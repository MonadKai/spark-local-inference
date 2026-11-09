# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare causal attention at fixed original-sequence positions."""

from functools import partial

import torch
from b12x.attention import varlen
from b12x.preparation import PreparedCall

from ignition.preparation.fixtures import varlen_attention as fixtures
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer


class PreparedPositionedCausalAttention:
    """Place one fragment in a bounded causal sequence before attention.

    The sequence extent stays constant across fragments. Unavailable future
    keys and queries are zero; causal masking keeps them out of published
    results. Only the current fragment is published. Cache pages are read-only.
    """

    def __init__(self, query, key, value, pages, lengths, count, positions,
                 sequence_count, output, *, capacity, session, workspace, buffers, name):
        if (query.ndim != 3 or query.shape[0] < 1 or query.shape[2] != 128
                or query.device.type != "cuda" or query.dtype != torch.bfloat16
                or not query.is_contiguous() or key.ndim != 4 or min(key.shape) < 1
                or key.shape[3] != query.shape[2] or query.shape[1] % key.shape[2]
                or key.dtype != query.dtype or key.device != query.device or not key.is_contiguous()
                or value.shape != key.shape or value.dtype != key.dtype
                or value.device != key.device or not value.is_contiguous()
                or output.shape != query.shape or output.dtype != query.dtype
                or output.device != query.device or not output.is_contiguous()):
            raise ValueError("positioned attention requires contiguous BF16 CUDA Q/paged KV/output")
        if (pages.ndim != 2 or pages.shape[0] != 1 or pages.shape[1] < 1
                or pages.dtype != torch.int32 or pages.device != query.device or not pages.is_contiguous()
                or any(t.shape != (1,) or t.dtype != torch.int32 or t.device != query.device
                       or not t.is_contiguous() for t in (lengths, count, sequence_count))
                or positions.shape != (query.shape[0],) or positions.dtype != torch.int64
                or positions.device != query.device or not positions.is_contiguous()
                or type(capacity) is not int or not 1 <= capacity <= pages.shape[1] * key.shape[1]
                or capacity > torch.iinfo(torch.int32).max):
            raise ValueError("positioned attention metadata differs from its bounded single sequence")
        begin, end = output.data_ptr(), output.data_ptr() + output.nbytes
        reads = (query, key, value, pages, lengths, count, positions, sequence_count)
        if any(begin < tensor.data_ptr() + tensor.nbytes and tensor.data_ptr() < end for tensor in reads):
            raise ValueError("positioned attention output must not alias inputs")
        self.query, self.key, self.value, self.output = query, key, value, output
        self.pages, self.lengths, self.count = pages, lengths, count
        self.positions, self.sequence_count = positions, sequence_count
        self.capacity, self.workspace = capacity, workspace
        self.rows = torch.arange(query.shape[0], device=query.device)
        self.sequence_rows = torch.arange(capacity, device=query.device)
        self.query_staging = buffers.zeros((capacity, *query.shape[1:]), query.dtype)
        self.key_staging = buffers.zeros((capacity, *key.shape[2:]), key.dtype)
        self.value_staging = buffers.zeros((capacity, *value.shape[2:]), value.dtype)
        self.attended = buffers.zeros(self.query_staging.shape, query.dtype)
        self.cu = buffers.zeros((2,), torch.int32)
        self.errors = buffers.zeros((2,), torch.int32)
        self.plan = varlen.plan(self.query_staging, self.key_staging, self.value_staging,
            self.cu, self.cu, max_seqlen_q=capacity, max_seqlen_k=capacity,
            causal=True, math_mode="flash")
        arguments = dict(q=self.query_staging, k=self.key_staging, v=self.value_staging,
            out=self.attended, cu_seqlens_q=self.cu, cu_seqlens_k=self.cu,
            max_seqlen_q=capacity, max_seqlen_k=capacity,
            softmax_scale=query.shape[-1] ** -0.5, causal=True)

        def prepare(state):
            scratch = planned_scratch(state.scratch_plan.scratch_specs(), workspace)
            binding = state.bind(scratch=scratch, **{**arguments, "cu_seqlens_q": torch.zeros_like(self.cu),
                "cu_seqlens_k": torch.zeros_like(self.cu)})
            return PreparedCall(run=lambda: state.run(binding), owners=prime_owners(workspace, scratch, binding))

        declare(session, (self.plan.request(name=name, prepare_call=prepare,
            benchmark_call=partial(fixtures.benchmark_call, arguments=arguments, workspace=workspace,
                inputs=fixtures.input_factory(arguments, storage=getattr(session, "benchmark_storage", None)))),))
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = planned_scratch(self.plan.scratch_specs(), self.workspace)

    def __call__(self):
        count, sequence, length = self.count[0], self.sequence_count[0], self.lengths[0]
        start = self.positions[0]
        active = count != 0
        live = self.rows < count
        valid = ((count > 0) & (count <= self.query.shape[0])
            & (sequence > 0) & (sequence <= self.capacity)
            & (start >= 0) & (start <= sequence.long())
            & (count.long() <= sequence.long() - start)
            & (length >= start + count.long()) & (length <= sequence)
            & (~live | (self.positions == start + self.rows)).all()
            & (~live | torch.isfinite(self.query).all(dim=(1, 2))).all())
        page_ids = torch.index_select(self.pages[0], 0, self.sequence_rows // self.key.shape[1])
        present = self.sequence_rows < length
        valid &= (~present | ((page_ids >= 0) & (page_ids < self.key.shape[0]))).all()
        addresses = page_ids.long().clamp(0, self.key.shape[0] - 1) * self.key.shape[1]
        addresses += self.sequence_rows % self.key.shape[1]
        keys = torch.index_select(self.key.flatten(0, 1), 0, addresses)
        values = torch.index_select(self.value.flatten(0, 1), 0, addresses)
        valid &= (~present | (torch.isfinite(keys).all(dim=(1, 2))
            & torch.isfinite(values).all(dim=(1, 2)))).all()
        source_rows = self.sequence_rows - start
        query = torch.index_select(self.query, 0, source_rows.clamp(0, self.query.shape[0] - 1).long())
        placed = (source_rows >= 0) & (source_rows < count) & valid
        self.query_staging.copy_(torch.where(placed[:, None, None], query, 0))
        self.key_staging.copy_(torch.where((present & valid)[:, None, None], keys, 0))
        self.value_staging.copy_(torch.where((present & valid)[:, None, None], values, 0))
        self.cu[0].zero_()
        self.cu[1].copy_(torch.where(valid, sequence, 0))
        self.attended.zero_()
        varlen.run_out(self.query_staging, self.key_staging, self.value_staging, self.cu,
            cu_seqlens_k=self.cu, out=self.attended, scratch=self.scratch, plan=self.plan,
            max_seqlen_q=self.capacity, max_seqlen_k=self.capacity,
            softmax_scale=self.query.shape[-1] ** -0.5)
        selected = torch.index_select(self.attended, 0, self.positions.clamp(0, self.capacity - 1).long())
        bad = active & (~valid | (live & ~torch.isfinite(selected).all(dim=(1, 2))).any())
        self.errors[0].zero_()
        self.errors[1].copy_(bad.int())
        self.output.copy_(torch.where(live[:, None, None] & valid, selected, 0))
        self.output.masked_fill_(bad, float("nan"))
        return self.output
