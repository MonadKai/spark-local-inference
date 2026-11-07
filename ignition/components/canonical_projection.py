# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bind canonical BF16 linear nodes into captured caller-owned outputs."""

import torch
import torch.nn.functional as F

from ignition.preparation.tensor_metadata import record_numeric_tensor


class PreparedCanonicalProjection:
    """Keep the checkpoint's BF16 linear nodes with fixed physical row extent.

    This preserves the declared BF16 nodes when alternative prepared reductions
    fail model-level numerical gates. It introduces neither
    a runtime fallback nor kernel selection. Optimized replacements need their
    own qualification against this arithmetic and the independent reference.
    """

    def __init__(self, source, weight, output, count=None, *, bias=None,
                 staging=None, singleton_count=None, subrow_extents=()):
        if source.ndim != 2 or output.ndim != 2 or weight.shape != (output.shape[1], source.shape[1]) \
                or source.shape[0] != output.shape[0] or source.shape[0] <= 0 \
                or any(t.dtype != torch.bfloat16 or t.device != source.device or not t.is_contiguous()
                    for t in (source, weight, output)) or source.device.type != "cuda":
            raise ValueError("canonical projection requires matching contiguous CUDA BF16 matrices")
        for value in (count, singleton_count):
            if value is not None and (value.shape != (1,) or value.dtype != torch.int32
                    or value.device != source.device or not value.is_contiguous()):
                raise ValueError("canonical projection counts require device int32 storage")
        if bias is not None and (bias.shape != (output.shape[1],) or bias.dtype != torch.bfloat16
                or bias.device != source.device or not bias.is_contiguous()):
            raise ValueError("canonical projection bias must retain checkpoint BF16 storage")
        if staging is not None and (staging.ndim != 2 or staging.shape[1] != source.shape[1]
                or staging.shape[0] < source.shape[0] or staging.dtype != source.dtype
                or staging.device != source.device or not staging.is_contiguous()):
            raise ValueError("canonical physical row staging has invalid geometry")
        self.weight, self.bias, self.output = weight, bias, output
        if (not isinstance(subrow_extents, tuple)
                or any(type(extent) is not int or not 0 < extent < source.shape[0] for extent in subrow_extents)
                or tuple(sorted(set(subrow_extents))) != subrow_extents
                or subrow_extents and count is None):
            raise ValueError("subrow projections require ordered distinct extents and a logical row count")
        self.subrow_extents = subrow_extents
        self.count, self.singleton_count, self.staging = count, singleton_count, staging
        self.rows, self.columns = source.shape
        self.positions = None if count is None else torch.arange(self.rows, device=source.device)
        self._aliases(source)
        record_numeric_tensor(weight)
        if bias is not None:
            record_numeric_tensor(bias)

    def _aliases(self, source):
        reads = (source, self.weight, *((self.bias,) if self.bias is not None else ()),
            *((self.count,) if self.count is not None else ()),
            *((self.singleton_count,) if self.singleton_count is not None else ()))
        writes = (self.output, *((self.staging,) if self.staging is not None else ()))
        for index, target in enumerate(writes):
            begin, end = target.data_ptr(), target.data_ptr() + target.nbytes
            if any(begin < value.data_ptr() + value.nbytes and value.data_ptr() < end
                    for value in (*reads, *writes[index + 1:])):
                raise ValueError("canonical projection writes must not overlap inputs or each other")

    def __call__(self, source):
        if source.shape != (self.rows, self.columns) or source.dtype != torch.bfloat16 \
                or source.device != self.output.device or not source.is_contiguous():
            raise ValueError("canonical projection source differs from prepared node")
        if not torch.compiler.is_compiling():
            self._aliases(source)
        actual = source
        if self.staging is not None:
            self.staging.zero_()
            self.staging[:self.rows].copy_(source)
            actual = self.staging
        self.output.copy_(self._linear(actual)[:self.rows])
        # Each prepared extent executes; the device count selects its BF16
        # result. This fixes measured small-M arithmetic groups without host
        # shape changes, runtime dispatch or a claim of reduced GPU work.
        for extent in reversed(self.subrow_extents):
            shorter = self._linear(source[:extent])
            self.output[:extent].copy_(torch.where(self.count <= extent, shorter, self.output[:extent]))
        if self.singleton_count is not None:
            singleton = self._linear(source[:1])
            self.output[:1].copy_(torch.where(self.singleton_count == 1, singleton, self.output[:1]))
        if self.count is not None:
            self.output.masked_fill_((self.positions >= self.count)[:, None], 0)
        return self.output

    def _linear(self, source):
        if self.bias is None:
            return F.linear(source, self.weight)
        # The vector-bias cuBLASLt path returns zero instead of the represented
        # 1/256 residual for K=7680 on SM121, even with reduced precision
        # disabled. A row bias retains the single final BF16 node
        # without introducing a BF16 matrix-product rounding before bias.
        return torch.addmm(self.bias[None, :], source, self.weight.T)


class PreparedRowwiseProjection(PreparedCanonicalProjection):
    """A captured singleton-reduction baseline for numerical attribution.

    Every row issues its own matrix product. This deliberately forgoes shared
    weight reads across rows; it is not a batched-GEMM optimization. The parent
    retains the same output, alias, count and BF16-node contracts.
    """

    def _linear(self, source):
        return torch.cat(tuple(super(PreparedRowwiseProjection, self)._linear(source[row:row + 1])
            for row in range(source.shape[0])), dim=0)
