# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compose original-prompt and generated-history arithmetic during prefill."""

from dataclasses import dataclass

import torch

from ignition.components.positioned_projection import PreparedPositionedProjection
from ignition.components.reduction_projection import PreparedReductionProjection
from ignition.components.request_flash import PreparedRequestFlashAttention


def history_projection_count(rows, count, positions, original):
    """Skip operand reads only when no published row selects history arithmetic."""
    selected = (rows < count) & (positions >= original)
    valid = (count >= 0) & (count <= rows.numel())
    return torch.where(valid & ~selected.any(), 0, count)


@dataclass(frozen=True)
class ReplayScratch:
    context: int
    full_input: torch.Tensor
    full_output: torch.Tensor
    history_output: torch.Tensor

    @classmethod
    def allocate(cls, tokens, context, buffers):
        if type(context) is not int or not 1 <= context <= 8192 \
                or type(tokens) is not int or not 1 <= tokens <= 8192:
            raise ValueError("ASR prompt and fragment storage require independent bounded extents")
        return cls(context, buffers.zeros((context * 6144,), torch.bfloat16),
            buffers.zeros((context * 6144,), torch.bfloat16),
            buffers.zeros((tokens * 6144,), torch.bfloat16))


class PreparedReplayProjection(torch.nn.Module):
    def __init__(self, source, module, output, count, *, prompt_count, positions,
                 original, history_count, scratch, session, name, buffers, workspace):
        super().__init__()
        if module.bias is not None or max(source.shape[1], output.shape[1]) > 6144:
            raise ValueError("ASR replay projections require checkpoint bias-free widths at most6144")
        self.output, self.positions, self.original = output, positions, original
        self.count = count
        self.rows = torch.arange(source.shape[0], device=source.device)
        # Every projection commits its result before the next projection starts.
        # Flat prefixes retain each projection's contiguous matrix geometry.
        staging = scratch.full_input[:scratch.context * source.shape[1]].view(scratch.context, source.shape[1])
        projected = scratch.full_output[:scratch.context * output.shape[1]].view(scratch.context, output.shape[1])
        self.history_output = scratch.history_output[:output.numel()].view_as(output)
        self.prompt = PreparedPositionedProjection(source, module.weight, output,
            prompt_count, positions, original, staging, projected,
            subrow_extents=tuple(extent for extent in (19, 27, 32, 64) if extent < scratch.context))
        self.history = PreparedReductionProjection(source, module.weight, self.history_output,
            history_count, session=session, name=name + ".history")

    def forward(self, source):
        self.prompt(source)
        self.history(source)
        history = (self.rows < self.count) & (self.positions >= self.original)
        self.output.view(torch.int16).copy_(torch.where(history[:, None],
            self.history_output.view(torch.int16), self.output.view(torch.int16)))
        return self.output


class PreparedReplayAttention:
    def __init__(self, query, key, value, metadata, lengths, output, *, original,
                 prompt_count, buffers):
        if metadata.lengths.numel() != 1:
            raise ValueError("ASR replay attention requires its sole prefill lane")
        self.query, self.output, self.metadata = query, output, metadata
        self.lengths, self.original, self.prompt_count = lengths, original, prompt_count
        rows, device = query.shape[0], query.device
        self.rows = torch.arange(rows, device=device)
        self.cumulative_rows = torch.arange(rows + 1, dtype=torch.int32, device=device)
        self.prompt_lengths = buffers.zeros((1,), torch.int32)
        self.prompt_cu = buffers.zeros((2,), torch.int32)
        self.prompt_cu_key = buffers.zeros((2,), torch.int32)
        self.prompt_output = buffers.zeros(query.shape, torch.bfloat16)
        self.history_query = buffers.zeros(query.shape, torch.bfloat16)
        self.history_output = buffers.zeros(query.shape, torch.bfloat16)
        self.history_pages = buffers.zeros((rows, metadata.page_table.shape[1]), torch.int32)
        self.history_lengths = buffers.zeros((rows,), torch.int32)
        self.history_cu = buffers.zeros((rows + 1,), torch.int32)
        self.history_cu_key = buffers.zeros((rows + 1,), torch.int32)
        self.errors = buffers.zeros((2,), torch.int32)
        self.prompt = PreparedRequestFlashAttention(query, key, value, metadata.page_table,
            self.prompt_lengths, self.prompt_cu, self.prompt_cu_key, self.prompt_output,
            max_query=rows, sequence_lengths=original)
        self.history = PreparedRequestFlashAttention(self.history_query, key, value,
            self.history_pages, self.history_lengths, self.history_cu, self.history_cu_key,
            self.history_output, max_query=1)

    def __call__(self):
        total, prompt = self.metadata.live_counts[0], self.prompt_count[0]
        history = (total - prompt).clamp(0, self.query.shape[0])
        self.prompt_lengths.copy_(torch.minimum(self.lengths, self.original))
        self.prompt_cu[:1].zero_(); self.prompt_cu[1:].copy_(self.prompt_count)
        self.prompt_cu_key[:1].zero_(); self.prompt_cu_key[1:].copy_(self.prompt_lengths)
        self.prompt()
        # Virtual rows refer to this request's real KV pages. They introduce
        # no native owner or write; the caller has written each current KV once.
        indices = (self.rows + prompt).clamp(0, self.query.shape[0] - 1).long()
        query = torch.index_select(self.query, 0, indices)
        live = self.rows < history
        self.history_query.copy_(torch.where(live[:, None, None], query, 0))
        positions = torch.index_select(self.metadata.positions, 0, indices)
        self.history_lengths.copy_(torch.where(live, positions + 1, 0).int())
        self.history_pages.copy_(self.metadata.page_table[:1])
        self.history_cu.copy_(torch.minimum(self.cumulative_rows, history))
        self.history_cu_key[:1].zero_()
        torch.cumsum(self.history_lengths, dim=0, dtype=torch.int32, out=self.history_cu_key[1:])
        self.history()
        restored = torch.index_select(self.history_output, 0,
            (self.rows - prompt).clamp(0, self.query.shape[0] - 1).long())
        bits = torch.where((self.rows < prompt)[:, None, None],
            self.prompt_output.view(torch.int16), restored.view(torch.int16))
        self.output.view(torch.int16).copy_(torch.where((self.rows < total)[:, None, None], bits, 0))
        self.errors[0].copy_(self.prompt.errors[0] | self.history.errors[0])
        self.errors[1].copy_(((self.prompt.errors[1:] != 0).any()
            | (self.history.errors[1:] != 0).any()).int())
