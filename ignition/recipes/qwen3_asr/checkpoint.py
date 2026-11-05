# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate original-format Qwen3-ASR-1.7B storage before model preparation.

This module neither registers a serving recipe nor imports a model implementation.
The name mapping follows Transformers 1d9654e78a23fb167ef539cbd699dfbd3e4830cf,
models/qwen3_asr/convert_qwen3_asr_to_hf.py. Both stored vocabulary tensors
remain separate: a configuration declaration never authorizes weight aliasing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ignition.preparation.checkpoint import (
    CheckpointError, IndexedCheckpoint, TensorMetadata, _parse_json, inspect_indexed_checkpoint,
)

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"


def validate_config(config: dict) -> None:
    """Require the original, unquantized 1.7B inference geometry and arithmetic."""
    def require(actual, expected, path):
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                raise CheckpointError(f"{path}: expected an object")
            for key, value in expected.items():
                require(actual.get(key), value, f"{path}.{key}")
        elif actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
            raise CheckpointError(f"{path}: expected {expected!r}, got {actual!r}")

    require(config, {
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "model_type": "qwen3_asr",
        "thinker_config": {
            "dtype": "bfloat16",
            "audio_token_id": 151676,
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "audio_config": {
                "model_type": "qwen3_asr_audio_encoder",
                "num_mel_bins": 128, "encoder_layers": 24, "num_hidden_layers": 24,
                "encoder_attention_heads": 16, "encoder_ffn_dim": 4096, "d_model": 1024,
                "downsample_hidden_size": 480, "output_dim": 2048,
                "n_window": 50, "n_window_infer": 800, "conv_chunksize": 500,
                "max_source_positions": 1500, "activation_function": "gelu",
                "dropout": 0, "attention_dropout": 0, "activation_dropout": 0,
                "scale_embedding": False,
            },
            "text_config": {
                "model_type": "qwen3", "vocab_size": 151936,
                "hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 28,
                "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
                "hidden_act": "silu", "attention_bias": False, "attention_dropout": 0,
                "rms_norm_eps": 1e-6, "max_position_embeddings": 65536,
                "rope_theta": 1000000, "tie_word_embeddings": True,
                "rope_scaling": {
                    "rope_type": "default", "type": "default", "interleaved": True,
                    "mrope_interleaved": True, "mrope_section": [24, 20, 20],
                },
            },
        },
    }, "config")
    for section in (config, config["thinker_config"], config["thinker_config"]["audio_config"],
                    config["thinker_config"]["text_config"]):
        if section.get("quantization_config") is not None:
            raise CheckpointError("Qwen3-ASR-1.7B requires unquantized original BF16 storage")


def expected_tensors():
    """Original names/shapes; every stored tensor is consumed without conversion."""
    audio = "thinker.audio_tower."
    for name, shape in (
        ("conv2d1.weight", (480, 1, 3, 3)), ("conv2d1.bias", (480,)),
        ("conv2d2.weight", (480, 480, 3, 3)), ("conv2d2.bias", (480,)),
        ("conv2d3.weight", (480, 480, 3, 3)), ("conv2d3.bias", (480,)),
        ("conv_out.weight", (1024, 7680)),
        ("ln_post.weight", (1024,)), ("ln_post.bias", (1024,)),
        ("proj1.weight", (1024, 1024)), ("proj1.bias", (1024,)),
        ("proj2.weight", (2048, 1024)), ("proj2.bias", (2048,)),
    ):
        yield audio + name, shape
    for layer in range(24):
        prefix = f"{audio}layers.{layer}."
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            yield prefix + f"self_attn.{projection}.weight", (1024, 1024)
            yield prefix + f"self_attn.{projection}.bias", (1024,)
        for norm in ("self_attn_layer_norm", "final_layer_norm"):
            yield prefix + norm + ".weight", (1024,)
            yield prefix + norm + ".bias", (1024,)
        for name, shape in (("fc1.weight", (4096, 1024)), ("fc1.bias", (4096,)),
                            ("fc2.weight", (1024, 4096)), ("fc2.bias", (1024,))):
            yield prefix + name, shape
    yield "thinker.model.embed_tokens.weight", (151936, 2048)
    yield "thinker.model.norm.weight", (2048,)
    yield "thinker.lm_head.weight", (151936, 2048)
    for layer in range(28):
        prefix = f"thinker.model.layers.{layer}."
        for name, shape in (
            ("input_layernorm.weight", (2048,)), ("post_attention_layernorm.weight", (2048,)),
            ("self_attn.q_proj.weight", (2048, 2048)),
            ("self_attn.k_proj.weight", (1024, 2048)),
            ("self_attn.v_proj.weight", (1024, 2048)),
            ("self_attn.o_proj.weight", (2048, 2048)),
            ("self_attn.q_norm.weight", (128,)), ("self_attn.k_norm.weight", (128,)),
            ("mlp.gate_proj.weight", (6144, 2048)), ("mlp.up_proj.weight", (6144, 2048)),
            ("mlp.down_proj.weight", (2048, 6144)),
        ):
            yield prefix + name, shape


def canonical_name(name: str) -> str:
    """Map an anchored original-format name, rejecting other checkpoint formats."""
    for source, destination in (
        ("thinker.model.", "model.language_model."),
        ("thinker.lm_head.", "lm_head."),
        ("thinker.audio_tower.proj1.", "model.multi_modal_projector.linear_1."),
        ("thinker.audio_tower.proj2.", "model.multi_modal_projector.linear_2."),
        ("thinker.audio_tower.", "model.audio_tower."),
    ):
        if name.startswith(source):
            return destination + name[len(source):]
    raise CheckpointError(f"not an original Qwen3-ASR tensor: {name}")


@dataclass(frozen=True)
class OriginalCheckpoint:
    index: IndexedCheckpoint
    canonical_tensors: Mapping[str, TensorMetadata]

    def embedding_and_head_equal(self) -> bool:
        """Compare every stored byte with bounded CPU reads, before any aliasing.

        Call before modifying or moving source storage. This checks these two
        tensors only, not revision identity or finite values elsewhere.
        """
        embedding = self.index.tensors["thinker.model.embed_tokens.weight"]
        head = self.index.tensors["thinker.lm_head.weight"]
        with (self.index.snapshot / embedding.shard).open("rb") as left, (
            self.index.snapshot / head.shard
        ).open("rb") as right:
            left.seek(embedding.offset)
            right.seek(head.offset)
            remaining = embedding.nbytes
            while remaining:
                size = min(remaining, 1 << 20)
                a, b = left.read(size), right.read(size)
                if len(a) != size or len(b) != size:
                    raise CheckpointError("truncated embedding or LM-head payload")
                if a != b:
                    return False
                remaining -= size
        return True


def inspect_original_checkpoint(snapshot: str | Path) -> OriginalCheckpoint:
    """Validate config, complete names/shapes/dtypes and indexed file coverage.

    Payloads, tokenizer/template semantics and upstream revision identity are
    separate checks. Inspection does not imply GPU or native serving support.
    """
    index = inspect_indexed_checkpoint(snapshot)
    validate_config(_parse_json((index.snapshot / "config.json").read_bytes(), "config.json"))
    unmatched = set(index.tensors)
    canonical = {}
    for name, shape in expected_tensors():
        tensor = index.tensors.get(name)
        if tensor is None:
            raise CheckpointError(f"missing required tensor: {name}")
        if tensor.shape != shape or tensor.dtype != "BF16":
            raise CheckpointError(f"{name}: expected BF16 {shape}, got {tensor.dtype} {tensor.shape}")
        destination = canonical_name(name)
        if destination in canonical:
            raise CheckpointError(f"duplicate canonical tensor: {destination}")
        canonical[destination] = tensor
        unmatched.remove(name)
    if unmatched:
        raise CheckpointError(f"unmapped tensor: {min(unmatched)}")
    return OriginalCheckpoint(index, MappingProxyType(canonical))
