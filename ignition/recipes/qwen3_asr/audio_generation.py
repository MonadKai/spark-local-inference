# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate immutable audio windows and publish their generation-owned features."""

import torch

from ignition.components.feature_delivery import PreparedFeaturePublication


def validate_window(inputs, state, frame_capacity, features):
    requests = state.requests
    safe = inputs.request_slot.long().clamp(0, requests.capacity - 1)
    valid = (inputs.request_slot >= 0) & (inputs.request_slot < requests.capacity)
    valid &= inputs.generation > 0
    valid &= (inputs.generation == requests.generation[safe]) & (requests.status[safe] == 1)
    valid &= (state.feature_slot >= 0) & (state.feature_slot < state.arena.shape[0])
    count = 13 * (inputs.frames // 100) + ((inputs.frames % 100) + 7) // 8
    valid &= (inputs.frames > 0) & (inputs.frames <= frame_capacity)
    valid &= (inputs.feature_count == count) & (count <= inputs.gather_indices.numel())
    total = inputs.total_input_frames.long()
    offset = inputs.feature_offset.long()
    total_features = 13 * (total // 100) + ((total % 100) + 7) // 8
    begin = (offset // 104) * 800
    valid &= (total > 0) & (total_features <= state.arena.shape[1])
    valid &= (offset >= 0) & (offset % 104 == 0) & (begin < total)
    valid &= inputs.frames.long() == (total - begin).clamp(min=0, max=800)
    width = total.clamp(max=100)
    valid &= inputs.convolution_width == width
    valid &= (inputs.feature_offset >= 0) & (inputs.feature_offset <= state.arena.shape[1] - count)
    valid &= (inputs.feature_offset == 0) | ((state.generation[safe] == inputs.generation)
                                           & (state.count[safe] == inputs.feature_offset))
    valid &= (inputs.cu_seqlens[0] == 0) & (inputs.cu_seqlens[1] == count)
    expected = torch.where(features < count, features, -1)
    valid &= (inputs.gather_indices == expected).all() & torch.isfinite(inputs.mel).all()
    return valid


class PreparedAudioGeneration:
    def __init__(self, encoder, inputs, state, completion, frame, sequence, *, buffers):
        self.encoder, self.inputs, self.state = encoder, inputs, state
        self.errors = buffers.zeros((1,), torch.int32)
        self.mel = buffers.zeros(inputs.mel.shape, torch.float32)
        self.publish = PreparedFeaturePublication(state, completion, frame, sequence,
            request_slot=inputs.request_slot, generation=inputs.generation, offset=inputs.feature_offset)
        self.features = torch.arange(inputs.gather_indices.numel(), device=inputs.mel.device)

    def __call__(self):
        inputs, metadata = self.inputs, self.encoder.metadata
        valid = validate_window(inputs, self.state, self.encoder.capacity, self.features)
        self.errors.copy_((~valid).int())
        self.mel.copy_(torch.where(valid, inputs.mel, 0))
        metadata.frames.copy_(torch.where(valid, inputs.frames, 0))
        metadata.total_input_frames.copy_(torch.where(valid, inputs.total_input_frames, 1))
        metadata.convolution_width.copy_(torch.where(valid, inputs.convolution_width, 1))
        metadata.gather_indices.copy_(torch.where(valid, inputs.gather_indices, -1))
        metadata.cu_seqlens.copy_(torch.where(valid, inputs.cu_seqlens, 0))
        metadata.feature_count.copy_(torch.where(valid, inputs.feature_count, 0))
        output = self.encoder(self.mel)
        active = self.features < metadata.feature_count
        nonfinite = (~((~active[:, None]) | torch.isfinite(output)).all()).int() * 64
        self.errors.copy_(self.errors | nonfinite)
        self.publish(output, metadata.feature_count, self.errors)
