# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent variable-width convolutions check fixed-width short-tail handling."""

import pytest
import torch
import torch.nn.functional as F

from ignition.recipes.qwen3_asr.audio import _ConvolutionChunks, convolution_chunks


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
@pytest.mark.parametrize("width", range(1, 101))
@pytest.mark.parametrize("short_width_path", [False, True])
@torch.no_grad()
def test_fixed_convolutions_preserve_original_global_width(width, dtype, short_width_path):
    torch.manual_seed(71)
    convolutions = [torch.nn.Conv2d(a, b, 3, stride=2, padding=1).to(dtype=dtype).eval()
                    for a, b in ((1, 3), (3, 4), (4, 2))]
    source = torch.randn((2, 1, 16, 100), dtype=dtype)
    source[:, :, :, width:] = 0
    source[1, :, :, max(1, width // 2):] = 0
    original = source[:, :, :, :width].clone()
    for convolution in convolutions:
        original = F.gelu(convolution(original))
    live_width = torch.tensor([width], dtype=torch.int32)
    fixed = (_ConvolutionChunks(convolutions, 100, source.device)(source, live_width, live_width) if short_width_path else
        convolution_chunks(source, live_width, convolutions, [torch.arange(value) for value in (50, 25, 13)]))
    assert torch.equal(fixed[:, :, :, :original.shape[-1]], original)
    assert torch.count_nonzero(fixed[:, :, :, original.shape[-1]:]) == 0


@torch.no_grad()
def test_final_layer_mask_alone_changes_valid_values():
    torch.manual_seed(71)
    convolutions = [torch.nn.Conv2d(a, b, 3, stride=2, padding=1).double().eval()
                    for a, b in ((1, 3), (3, 4), (4, 2))]
    source = torch.zeros((1, 1, 16, 100), dtype=torch.float64)
    source[:, :, :, :1] = 0.3
    wrong = source.clone()
    for convolution in convolutions:
        wrong = F.gelu(convolution(wrong))
    original = source[:, :, :, :1].clone()
    for convolution in convolutions:
        original = F.gelu(convolution(original))
    assert not torch.equal(wrong[:, :, :, :1], original)
