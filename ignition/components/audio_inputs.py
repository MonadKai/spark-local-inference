# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Typed uploads for one bounded audio encoder window and text prefill."""

from dataclasses import dataclass, fields

import torch


@dataclass(frozen=True)
class PrefillAudioInputs:
    audio_token_indices: torch.Tensor
    audio_feature_slot: torch.Tensor

    @classmethod
    def allocate(cls, tokens, device):
        return cls(torch.full((tokens,), -1, dtype=torch.int32, device=device),
                   torch.full((1,), -1, dtype=torch.int32, device=device))

    def buffers(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class AudioInputs:
    mel: torch.Tensor
    frames: torch.Tensor
    total_input_frames: torch.Tensor
    convolution_width: torch.Tensor
    gather_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    feature_count: torch.Tensor
    feature_offset: torch.Tensor
    request_slot: torch.Tensor
    generation: torch.Tensor

    @classmethod
    def allocate(cls, frames, features, device):
        if min(frames, features) <= 0:
            raise ValueError("audio inputs require positive prepared capacities")
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)
        return cls(zeros((128, frames), torch.float32), zeros((1,)), zeros((1,)), zeros((1,)),
                   zeros((features,)).fill_(-1), zeros((2,)), zeros((1,)), zeros((1,)),
                   zeros((1,)).fill_(-1), zeros((1,), torch.int64))

    def buffers(self, prefix):
        return {prefix + field.name: getattr(self, field.name) for field in fields(self)}
