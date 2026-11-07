# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Original-format weights attached to the pinned canonical module tree."""

import importlib.metadata
import json
from dataclasses import dataclass

import torch

from ignition.components.model import canonical_tree
from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import CheckpointError

from .checkpoint import OriginalCheckpoint, canonical_name
from .config import canonical_config

TRANSFORMERS_REVISION = "1d9654e78a23fb167ef539cbd699dfbd3e4830cf"


@dataclass(frozen=True)
class ModelWeights:
    checkpoint: OriginalCheckpoint
    tree: torch.nn.Module
    draft: None = None
    shard: TensorParallelShard = TensorParallelShard(0, 1)

    @property
    def head(self):
        return self.tree.lm_head.weight

    @property
    def allocated_bytes(self):
        return sum(value.nbytes for value in (*self.tree.parameters(), *self.tree.buffers()))

    @classmethod
    def load(cls, checkpoint, device, *, progress=None):
        return cls(checkpoint, load_model(checkpoint, device, progress=progress))

    def capacity_tree(self):
        """Independent forwards sharing only immutable parameter/buffer storage."""
        tree = create_model_tree(self.checkpoint)
        for name, value in (*self.tree.named_parameters(remove_duplicate=False),
                            *self.tree.named_buffers(remove_duplicate=False)):
            parent, _, field = name.rpartition(".")
            setattr(tree.get_submodule(parent), field, value)
        return tree


def create_model_tree(checkpoint: OriginalCheckpoint):
    """Allocate meta geometry only; the returned tree cannot execute yet."""
    distribution = importlib.metadata.distribution("transformers")
    source = json.loads(distribution.read_text("direct_url.json") or "{}")
    if source.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_REVISION:
        raise RuntimeError(f"ASR preparation requires Transformers {TRANSFORMERS_REVISION}")
    from transformers import Qwen3ASRForConditionalGeneration

    tree = canonical_tree(Qwen3ASRForConditionalGeneration, canonical_config(checkpoint.index.snapshot))
    parameters = dict(tree.named_parameters(remove_duplicate=False))
    if parameters.keys() != checkpoint.canonical_tensors.keys():
        raise CheckpointError("canonical ASR parameter coverage differs from the original checkpoint")
    for name, parameter in parameters.items():
        if tuple(parameter.shape) != checkpoint.canonical_tensors[name].shape:
            raise CheckpointError(f"{name}: canonical ASR parameter shape differs")
    return tree


@torch.no_grad()
def load_model(checkpoint: OriginalCheckpoint, device: torch.device | str, *, progress=None):
    """Consume every original parameter, retaining separately stored head weights.

    This is preparation, not a serving execution path. Nonpersistent positional
    buffers must be reconstructed on the real device rather than left on meta.
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from transformers.models.qwen3_asr.modeling_qwen3_asr import SinusoidsPositionEmbedding

    tree = create_model_tree(checkpoint)
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("ASR preparation requires a CUDA device")
    consumed = set()
    with DirectWeightLoader(checkpoint.index, device, progress=progress) as loader:
        for original in checkpoint.index.tensors:
            name = canonical_name(original)
            parent, _, field = name.rpartition(".")
            tensor = loader.load_tensor(original)
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != checkpoint.canonical_tensors[name].shape:
                raise CheckpointError(f"{original}: loaded ASR dtype or shape differs")
            if not torch.isfinite(tensor).all().item():
                raise CheckpointError(f"{original}: nonfinite ASR checkpoint values")
            setattr(tree.get_submodule(parent), field, torch.nn.Parameter(tensor, requires_grad=False))
            consumed.add(original)
    if consumed != checkpoint.index.tensors.keys():
        raise CheckpointError("original ASR parameter consumption is incomplete")
    # These are deterministic, nonpersistent buffers, absent from the checkpoint.
    # Construct the canonical FP32 tables on CPU, then transfer them unchanged.
    with torch.device("cpu"):
        rotary = Qwen3RotaryEmbedding(tree.config.text_config)
        positional = SinusoidsPositionEmbedding(13, 1024)
    tree.model.language_model.rotary_emb = rotary.to(device)
    tree.model.audio_tower.positional_embedding = positional.to(device)
    if any(value.is_meta for value in (*tree.parameters(), *tree.buffers())):
        raise CheckpointError("ASR preparation left unmaterialized meta storage")
    return tree
