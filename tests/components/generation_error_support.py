# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Small real attention computation for shared generation failure tests."""

from types import SimpleNamespace

import torch

from ignition.components.request_flash import PreparedRequestFlashAttention


class AttentionDecoder:
    """Exercise real captured attention errors without loading a checkpoint."""

    def __init__(self, metadata, pool, faults, *, fault_kind="metadata"):
        device = metadata.token_ids.device
        rows, capacity = metadata.lengths.numel(), metadata.token_ids.numel()
        self.metadata, self.pool, self.faults = metadata, pool, faults
        self.fault_kind = fault_kind
        self.weights = SimpleNamespace(draft=None)
        self.head = SimpleNamespace(gather=SimpleNamespace(runtime=None))
        self.scratch = SimpleNamespace(
            hidden=torch.zeros((rows, 2048), dtype=torch.bfloat16, device=device),
            logits=torch.zeros((rows, 256), dtype=torch.bfloat16, device=device))
        self.query = torch.zeros((capacity, 16, 128), dtype=torch.bfloat16, device=device)
        self.key = torch.zeros((rows, 256, 8, 128), dtype=torch.bfloat16, device=device)
        self.value = torch.full_like(self.key, 2.)
        self.output = torch.zeros_like(self.query)
        self.key_lengths = torch.zeros_like(metadata.lengths)
        self.cumulative_keys = torch.zeros_like(metadata.query_start_loc)
        self.execution_errors = torch.zeros_like(metadata.lengths)
        self.attention = PreparedRequestFlashAttention(self.query, self.key, self.value,
            metadata.page_table, self.key_lengths, metadata.query_start_loc, self.cumulative_keys,
            self.output, max_query=metadata.max_query_tokens)
        self.auxiliary = None

    def __call__(self):
        m = self.metadata
        self.key_lengths.copy_(torch.where(self.faults.bool(), 0, m.starts + m.lengths)
            if self.fault_kind == "metadata" else m.starts + m.lengths)
        if self.fault_kind == "query":
            positions = torch.arange(self.query.shape[0], device=self.query.device)
            bad = ((positions[:, None] >= m.query_start_loc[:-1])
                & (positions[:, None] < m.query_start_loc[1:])
                & self.faults.bool()[None, :]).any(dim=1)
            self.query.copy_(torch.where(bad[:, None, None], float("nan"), 0.))
        if self.fault_kind == "value":
            pages = torch.arange(self.value.shape[0], device=self.value.device)
            bad = ((pages[:, None] == m.page_table[:, 0][None, :])
                & self.faults.bool()[None, :] & (m.lengths > 0)[None, :]).any(dim=1)
            self.value[:, 0, 0, 0].copy_(torch.where(bad, float("nan"), 2.))
        self.attention()
        self.execution_errors.copy_((self.attention.errors[0] | self.attention.errors[1:]).int())
        self.scratch.hidden.copy_(self.output.view(-1, 2048)[m.sample_indices])
        self.scratch.logits.copy_(self.scratch.hidden[:, :256])
