# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Place a contiguous input fragment in a fixed full-sequence projection."""

import torch

from .canonical_projection import PreparedCanonicalProjection


class PreparedPositionedProjection:
    """Preserve physical projection rows across fragments of one sequence.

    The caller owns the full-sequence staging and result buffers. Each call
    initializes all staging rows, projects at the declared physical extent,
    and publishes only its current fragment. No future activation is read.
    This is a numerical mechanism, not a claim of reduced projection work.
    """

    def __init__(self, source, weight, output, count, positions, sequence_count,
                 staging, projected, *, bias=None, subrow_extents=()):
        if (source.ndim != 2 or output.ndim != 2 or weight.ndim != 2 or source.shape[0] < 1
                or output.shape != (source.shape[0], weight.shape[0])
                or staging.ndim != 2 or staging.shape[1] != source.shape[1]
                or staging.shape[0] < source.shape[0]
                or projected.shape != (staging.shape[0], weight.shape[0])
                or staging.device != source.device or projected.device != source.device
                or any(t.dtype != torch.bfloat16 or t.device != source.device
                    or not t.is_contiguous() for t in (source, output))
                or positions.shape != (source.shape[0],)
                or positions.dtype != torch.int64 or positions.device != source.device
                or not positions.is_contiguous()
                or count.shape != (1,) or count.dtype != torch.int32
                or count.device != source.device or not count.is_contiguous()
                or sequence_count is None):
            raise ValueError("positioned projection requires contiguous BF16 fragments and int64 positions")
        self.operation = PreparedCanonicalProjection(staging, weight, projected,
            sequence_count, bias=bias, subrow_extents=subrow_extents)
        self.output, self.count, self.positions = output, count, positions
        self.sequence_count, self.staging, self.projected = sequence_count, staging, projected
        self.shape, self.device = source.shape, source.device
        self.local_rows = torch.arange(source.shape[0], device=source.device)
        self.full_rows = torch.arange(staging.shape[0], device=source.device)
        self._aliases(source)

    def _aliases(self, source):
        reads = (source, self.operation.weight, self.count, self.positions, self.sequence_count,
            *((self.operation.bias,) if self.operation.bias is not None else ()))
        writes = (self.output, self.staging, self.projected)
        for index, target in enumerate(writes):
            begin, end = target.data_ptr(), target.data_ptr() + target.nbytes
            if any(begin < value.data_ptr() + value.nbytes and value.data_ptr() < end
                    for value in (*reads, *writes[index + 1:])):
                raise ValueError("positioned projection writes must not overlap inputs or each other")

    def __call__(self, source):
        if (source.shape != self.shape or source.dtype != torch.bfloat16
                or source.device != self.device or not source.is_contiguous()):
            raise ValueError("positioned projection source differs from prepared fragment")
        if not torch.compiler.is_compiling():
            self._aliases(source)
        live = self.local_rows < self.count
        start = self.positions[0]
        valid = ((self.count >= 0) & (self.count <= source.shape[0])
            & (self.sequence_count >= self.count)
            & (self.sequence_count <= self.staging.shape[0]) & (start >= 0)
            & (start <= self.sequence_count.long())
            & (self.count.long() <= self.sequence_count.long() - start)
            & (~live | (self.positions == start + self.local_rows)).all()
            & (~live | torch.isfinite(source).all(dim=1)).all())
        source_rows = self.full_rows - start
        gathered = torch.index_select(source, 0, source_rows.clamp(0, source.shape[0] - 1).long())
        self.staging.copy_(torch.where(
            ((source_rows >= 0) & (source_rows < self.count) & valid)[:, None], gathered, 0))
        self.operation(self.staging)
        result = torch.index_select(self.projected, 0,
            self.positions.clamp(0, self.projected.shape[0] - 1).long())
        self.output.copy_(torch.where(live[:, None] & valid, result, 0))
        # An invalid active fragment must reach the caller's nonfinite guard.
        self.output.masked_fill_((self.count != 0) & ~valid, float("nan"))
        return self.output
