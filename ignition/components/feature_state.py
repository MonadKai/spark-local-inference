# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Generation-scoped feature storage consumed by a causal prefill graph."""

import torch


class FeatureState:
    def __init__(self, inputs, requests, *, indices, feature_slot, feature_token,
                 slots, features, hidden):
        if not 0 < slots <= requests.capacity or min(features, hidden) <= 0:
            raise ValueError("feature storage must fit positive prepared capacities")
        self.inputs, self.requests = inputs, requests
        self.indices, self.feature_slot, self.feature_token = indices, feature_slot, feature_token
        device = inputs.token_ids.device
        self.arena = torch.zeros((slots, features, hidden), dtype=torch.bfloat16, device=device)
        self.generation = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        self.count = torch.zeros(requests.capacity, dtype=torch.int32, device=device)

    def validate_prefill(self, tokens, lengths):
        inputs = self.inputs
        request = inputs.row_slot_ids[:1]
        safe = request.long().clamp(0, self.requests.capacity - 1)
        indices = self.indices[:tokens.numel()]
        live = torch.arange(tokens.numel(), device=tokens.device) < lengths[0]
        feature = tokens == self.feature_token
        valid = (~feature & (indices == -1)) | (feature & (indices >= 0) & (indices < self.count[safe]))
        owner = (request >= 0) & (request < self.requests.capacity)
        owner &= (self.generation[safe] == inputs.row_generations[0])
        owner &= (self.count[safe] > 0) & (self.count[safe] <= self.arena.shape[1])
        owner &= (self.feature_slot >= 0) & (self.feature_slot < self.arena.shape[0])
        return (~live | (valid & (~feature | owner))).all()

    def inject(self, metadata, embedding):
        slot = self.feature_slot.long().clamp(0, self.arena.shape[0] - 1)
        indices = self.indices[:embedding.shape[0]].long().clamp(0, self.arena.shape[1] - 1)
        embedding.copy_(torch.where((metadata.token_ids == self.feature_token)[:, None],
                                    self.arena[slot, indices], embedding))
