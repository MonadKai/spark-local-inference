# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Project logical rows using one fixed prepared matrix geometry."""

import torch

from .linear import UnquantizedProjection


class PreparedRowProjection:
    """Pad into caller-owned storage before one actual cross-row GEMM.

    A recipe can require identical projection geometry across logical graph
    capacities. Padding is internal scratch: it never changes live-row metadata,
    cache writes, completion capacity, or the graph's declared logical inputs.
    The ordinary b12x preparation and numerical candidate selection still apply.
    """

    def __init__(self, source, weight, output, *, row_capacity, buffers, session, name, bias=None):
        if (type(row_capacity) is not int or source.ndim != 2 or output.ndim != 2
                or not 1 <= source.shape[0] <= row_capacity or output.shape[0] != source.shape[0]
                or weight.shape != (output.shape[1], source.shape[1]) or buffers is None):
            raise ValueError("padded projection requires matching matrices and bounded logical rows")
        if (source.dtype not in (torch.bfloat16, torch.float32)
                or output.dtype not in (torch.bfloat16, torch.float32)
                or source.device != output.device or source.device != weight.device):
            raise ValueError("padded projection requires matching devices and BF16 or FP32 storage")
        self.rows, self.weight, self.output = source.shape[0], weight, output
        self.source = buffers.zeros((row_capacity, source.shape[1]), source.dtype)
        self.projected = buffers.zeros((row_capacity, output.shape[1]), output.dtype)
        self.operation = UnquantizedProjection(self.source, weight, self.projected,
            session=session, name=name, bias=bias)
        self.plan = self.operation.plan

    def __call__(self, source):
        self.source.zero_()
        self.source[:self.rows].copy_(source)
        self.output.copy_(self.operation(self.source)[:self.rows])
        return self.output
