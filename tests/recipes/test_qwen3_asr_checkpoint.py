# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Original-format preparation checks; no audio serving support is implied."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from ignition.preparation.checkpoint import CheckpointError, IndexedCheckpoint, TensorMetadata
from ignition.recipes.qwen3_asr.checkpoint import (
    OriginalCheckpoint, canonical_name, expected_tensors, inspect_original_checkpoint,
)


@pytest.mark.parametrize("original,canonical", [
    ("thinker.model.layers.0.self_attn.q_proj.weight", "model.language_model.layers.0.self_attn.q_proj.weight"),
    ("thinker.audio_tower.proj1.bias", "model.multi_modal_projector.linear_1.bias"),
    ("thinker.audio_tower.proj2.weight", "model.multi_modal_projector.linear_2.weight"),
    ("thinker.audio_tower.conv_out.weight", "model.audio_tower.conv_out.weight"),
    ("thinker.lm_head.weight", "lm_head.weight"),
])
def test_original_names_map_without_changing_suffix(original, canonical):
    assert canonical_name(original) == canonical


@pytest.mark.parametrize("name", ["model.language_model.norm.weight", "x.thinker.model.norm.weight", "score.weight"])
def test_other_formats_and_unanchored_names_are_rejected(name):
    with pytest.raises(CheckpointError, match="not an original"):
        canonical_name(name)


def test_inventory_matches_independent_canonical_module_tree():
    import torch
    from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration

    config = Qwen3ASRConfig(audio_config={"output_dim": 2048})
    with torch.device("meta"):
        model = Qwen3ASRForConditionalGeneration(config)
    expected = {canonical_name(name): shape for name, shape in expected_tensors()}
    actual = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    assert expected == actual
    assert len(expected) == 708


def _small_checkpoint(tmp_path):
    # Distinct equal-shape tensors, with a mismatch after the first read block.
    size = (1 << 20) + 2
    (tmp_path / "weights.bin").write_bytes(b"a" * size * 2)
    tensors = {
        name: TensorMetadata(name, "weights.bin", offset, size, (size // 2,), "BF16")
        for name, offset in (("thinker.model.embed_tokens.weight", 0), ("thinker.lm_head.weight", size))
    }
    return OriginalCheckpoint(IndexedCheckpoint(tmp_path, tensors, ()), {})


def test_tied_storage_requires_all_payload_bytes_equal(tmp_path):
    checkpoint = _small_checkpoint(tmp_path)
    assert checkpoint.embedding_and_head_equal()
    head = checkpoint.index.tensors["thinker.lm_head.weight"]
    with (tmp_path / head.shard).open("r+b") as stream:
        stream.seek(head.offset + head.nbytes - 1)
        stream.write(b"b")
    assert not checkpoint.embedding_and_head_equal()


def test_truncated_tied_payload_is_an_error(tmp_path):
    checkpoint = _small_checkpoint(tmp_path)
    with (tmp_path / "weights.bin").open("r+b") as stream:
        stream.truncate((1 << 21) + 3)
    with pytest.raises(CheckpointError, match="truncated"):
        checkpoint.embedding_and_head_equal()


@pytest.fixture(scope="module")
def snapshot():
    configured = os.environ.get("FLARE_QWEN3_ASR_SNAPSHOT")
    if configured is None:
        pytest.skip("set FLARE_QWEN3_ASR_SNAPSHOT to the original checkpoint")
    path = Path(configured)
    assert path.is_dir(), f"missing explicitly configured checkpoint: {path}"
    return path


@pytest.fixture(scope="module")
def checkpoint(snapshot):
    return inspect_original_checkpoint(snapshot)


def test_real_original_checkpoint_keeps_both_vocabulary_tensors(checkpoint):
    assert len(checkpoint.canonical_tensors) == len(checkpoint.index.tensors) == 708
    embedding = checkpoint.canonical_tensors["model.language_model.embed_tokens.weight"]
    head = checkpoint.canonical_tensors["lm_head.weight"]
    assert embedding.name != head.name
    assert embedding.offset != head.offset or embedding.shard != head.shard
    assert checkpoint.embedding_and_head_equal()


@pytest.mark.parametrize("field,value", [
    ("audio_config.encoder_layers", 23),
    ("audio_config.num_hidden_layers", 23),
    ("audio_config.activation_function", "silu"),
    ("text_config.rms_norm_eps", 1e-5),
    ("text_config.num_key_value_heads", 16),
    ("text_config.attention_bias", True),
    ("text_config.head_dim", True),
    ("text_config.rope_scaling.mrope_section", [16, 24, 24]),
])
def test_real_config_rejects_changed_semantics(snapshot, field, value):
    from ignition.recipes.qwen3_asr.checkpoint import validate_config

    config = json.loads((snapshot / "config.json").read_text())
    target = config["thinker_config"]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(CheckpointError, match=parts[-1]):
        validate_config(config)


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype"])
def test_real_index_rejects_incomplete_or_changed_storage(snapshot, checkpoint, monkeypatch, mutation):
    from ignition.recipes.qwen3_asr import checkpoint as module

    tensors = dict(checkpoint.index.tensors)
    name = "thinker.audio_tower.layers.0.self_attn.q_proj.weight"
    if mutation == "missing":
        del tensors[name]
    elif mutation == "extra":
        tensors["unused.weight"] = tensors[name]
    elif mutation == "shape":
        tensors[name] = replace(tensors[name], shape=(1024, 1023))
    else:
        tensors[name] = replace(tensors[name], dtype="F32")
    # Header parsing is covered by the shared checkpoint suite; this isolates
    # recipe storage validation without manufacturing multi-GB payload files.
    monkeypatch.setattr(module, "inspect_indexed_checkpoint", lambda _: replace(checkpoint.index, tensors=tensors))
    with pytest.raises(CheckpointError, match="required tensor|unmapped tensor|expected BF16"):
        inspect_original_checkpoint(snapshot)
