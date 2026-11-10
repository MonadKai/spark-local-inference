# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical ATen flash attention over one bounded bidirectional window."""

import torch


class PreparedFlashWindow:
    """Static storage with a live GPU length, including an empty invalid window.

    Complete-forward preparation compiles and captures the ATen operation. No
    host length read, runtime plan selection or Python call occurs at replay.
    The caller validates cumulative lengths as [0, N], with 0 <= N <= capacity.
    """

    def __init__(self, q, k, v, cu, output):
        if (q.ndim != 3 or not q.shape[0] or q.dtype != torch.bfloat16 or q.device.type != "cuda"
                or any(t.shape != q.shape or t.dtype != q.dtype or t.device != q.device
                       or not t.is_contiguous() for t in (q, k, v, output))
                or cu.shape != (2,) or cu.dtype != torch.int32 or cu.device != q.device
                or not cu.is_contiguous()):
            raise ValueError("flash window requires matching contiguous BF16 CUDA Q/K/V/output and int32 cu[2]")
        begin, end = output.data_ptr(), output.data_ptr() + output.nbytes
        if any(begin < t.data_ptr() + t.nbytes and t.data_ptr() < end for t in (q, k, v, cu)):
            raise ValueError("flash window output must not alias its inputs")
        self.q, self.k, self.v, self.cu, self.output = q, k, v, cu, output
        self.positions = torch.arange(q.shape[0], device=q.device, dtype=torch.int32)

    def __call__(self):
        value, _, _, _, _ = torch.ops.aten._flash_attention_forward(
            self.q, self.k, self.v, self.cu, self.cu, self.q.shape[0], self.k.shape[0],
            0., False, False)
        self.output.copy_(value)
        self.output.masked_fill_((self.positions >= self.cu[-1])[:, None, None], 0)
        return self.output
