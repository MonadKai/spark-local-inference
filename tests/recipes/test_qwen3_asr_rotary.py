# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed native positions use the full original one-dimensional RoPE table."""

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 17])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_packed_rotary_preserves_axes_head_geometry_and_full_frequencies(tokens, dtype):
    config = Qwen3Config(hidden_size=2048, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, rope_parameters={"rope_type": "default", "rope_theta": 1000000.})
    rotary = Qwen3RotaryEmbedding(config)
    positions = (torch.arange(tokens, dtype=torch.int64) * 511) - 1
    hidden = torch.zeros((tokens, 2048), dtype=dtype)
    cos, sin = rotary(hidden, positions)
    assert cos.shape == sin.shape == (tokens, 128)
    assert cos.dtype == sin.dtype == dtype
    # Every original mRoPE axis receives this same sequential position in ASR.
    # Interleaved section selection consequently retains all 64 frequencies.
    frequencies = positions.float()[:, None] * rotary.inv_freq.float()[None, :]
    expected_cos = torch.cat((frequencies.cos(), frequencies.cos()), dim=-1).to(dtype)
    expected_sin = torch.cat((frequencies.sin(), frequencies.sin()), dim=-1).to(dtype)
    assert torch.equal(cos.view(torch.uint8), expected_cos.view(torch.uint8))
    assert torch.equal(sin.view(torch.uint8), expected_sin.view(torch.uint8))
    q = torch.arange(tokens * 16 * 128).reshape(tokens, 16, 128).remainder(47).to(dtype) / 16
    k = torch.arange(tokens * 8 * 128).reshape(tokens, 8, 128).remainder(31).to(dtype) / 16
    actual = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    for source, result in zip((q, k), actual, strict=True):
        rotated = torch.cat((-source[..., 64:], source[..., :64]), dim=-1)
        expected = source * expected_cos[:, None] + rotated * expected_sin[:, None]
        assert result.shape == source.shape
        assert torch.equal(result.view(torch.uint8), expected.view(torch.uint8))
