# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint ownership and the BF16 boundary before the affine multiply."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from ignition.components.canonical_rms_norm import CanonicalRMSNorm


def test_canonical_gain_identity_and_pre_affine_rounding_are_preserved():
    fixture = load_file(Path(__file__).resolve().parents[1] / "fixtures/asr-rmsnorm-replay.safetensors")
    weight = torch.nn.Parameter(fixture["weight"], requires_grad=False)
    module = CanonicalRMSNorm(weight, 1e-6, minimum_rows=64)
    canonical = Qwen3RMSNorm(2048, eps=1e-6).to(dtype=torch.bfloat16)
    canonical.weight = weight
    source = fixture["hidden"].repeat(64, 1)
    expected = canonical(source)
    assert module.weight is weight and dict(module.named_parameters())["weight"] is weight
    assert torch.equal(module(source).view(torch.uint8), expected.view(torch.uint8))
    floating = source.float()
    normalized = floating * torch.rsqrt(floating.pow(2).mean(-1, keepdim=True) + 1e-6)
    fused_rounding = (weight.float() * normalized).bfloat16()
    assert not torch.equal(fused_rounding, expected), "control lost the pre-affine BF16 boundary"
    assert torch.equal(module(source[:8]).view(torch.uint8), expected[:8].view(torch.uint8))


@pytest.mark.parametrize("defect", ["gain-dtype", "trainable", "gain-shape", "epsilon", "rows"])
def test_invalid_canonical_norm_declaration_is_rejected(defect):
    weight = torch.nn.Parameter(torch.ones(2048, dtype=torch.bfloat16), requires_grad=False)
    eps, rows = 1e-6, 64
    if defect == "gain-dtype": weight = torch.nn.Parameter(weight.float(), requires_grad=False)
    if defect == "trainable": weight.requires_grad_(True)
    if defect == "gain-shape": weight = torch.nn.Parameter(weight.view(1, 2048), requires_grad=False)
    if defect == "epsilon": eps = float("nan")
    if defect == "rows": rows = True
    with pytest.raises(ValueError, match="canonical RMSNorm"):
        CanonicalRMSNorm(weight, eps, minimum_rows=rows)
