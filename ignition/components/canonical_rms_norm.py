# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical BF16 RMSNorm with a minimum physical row extent for its mean."""

import math

import torch
import torch.nn.functional as F


class CanonicalRMSNorm(torch.nn.Module):
    def __init__(self, weight, eps, *, minimum_rows):
        super().__init__()
        if (not isinstance(weight, torch.nn.Parameter) or weight.ndim != 1
                or not weight.numel() or weight.dtype != torch.bfloat16
                or not weight.is_contiguous() or weight.requires_grad
                or not math.isfinite(eps) or eps <= 0
                or type(minimum_rows) is not int or minimum_rows <= 0):
            raise ValueError("canonical RMSNorm requires immutable BF16 gain, positive epsilon and row extent")
        self.weight = weight
        self.variance_epsilon, self.minimum_rows = float(eps), minimum_rows

    def forward(self, hidden_states):
        if (hidden_states.ndim != 2 or hidden_states.shape[1] != self.weight.numel()
                or hidden_states.shape[0] == 0 or hidden_states.dtype != torch.bfloat16):
            raise ValueError("canonical RMSNorm requires a nonempty BF16 row matrix matching its gain")
        floating = hidden_states.to(torch.float32)
        square = floating.pow(2)
        rows = square.shape[0]
        # PyTorch's mean changes its FP32 reduction tree with physical row
        # count. Pad only that operand; each row still reduces its own width.
        if rows < self.minimum_rows:
            square = F.pad(square, (0, 0, 0, self.minimum_rows - rows))
            variance = square.mean(-1, keepdim=True)[:rows]
        else:
            variance = square.mean(-1, keepdim=True)
        normalized = floating * torch.rsqrt(variance + self.variance_epsilon)
        # This checkpoint rounds before the affine multiply as well as after it.
        return self.weight * normalized.to(hidden_states.dtype)
