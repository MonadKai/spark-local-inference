# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise typed audio metadata guards and generation-scoped prefill inputs."""

import pytest
import torch

from ignition.components.audio_inputs import AudioInputs, PrefillAudioInputs
from ignition.components.feature_state import FeatureState
from ignition.components.paged_batch import BatchInputs
from ignition.components.sampling import DeviceRequestState
from ignition.recipes.qwen3_asr.audio_generation import validate_window


def state():
    inputs = BatchInputs.allocate(4, 1, "cpu", capacity=2)
    requests = DeviceRequestState.allocate("cpu", vocab_size=100, capacity=2)
    prefill = PrefillAudioInputs.allocate(4, "cpu")
    features = FeatureState(inputs, requests, indices=prefill.audio_token_indices,
        feature_slot=prefill.audio_feature_slot, feature_token=99, slots=1, features=208, hidden=4)
    inputs.row_slot_ids[0], inputs.row_generations[0] = 1, 12
    requests.generation[1], requests.status[1] = 12, 1
    features.feature_slot.fill_(0)
    return inputs, requests, prefill, features


def window(frames=801, index=0):
    length = min(800, frames - index * 800)
    count = 13 * (length // 100) + ((length % 100) + 7) // 8
    inputs = AudioInputs.allocate(800, 104, "cpu")
    inputs.frames.fill_(length)
    inputs.total_input_frames.fill_(frames)
    inputs.feature_count.fill_(count)
    inputs.convolution_width.fill_(min(frames, 100))
    inputs.feature_offset.fill_(index * 104)
    inputs.request_slot.fill_(1)
    inputs.generation.fill_(12)
    inputs.gather_indices[:count].copy_(torch.arange(count, dtype=torch.int32))
    inputs.cu_seqlens[1] = count
    return inputs


def valid(inputs, features):
    return validate_window(inputs, features, 800, torch.arange(104))


def test_windows_require_contiguous_publication_for_the_current_generation():
    _, _, _, features = state()
    assert valid(window(), features)
    second = window(index=1)
    assert not valid(second, features)
    features.generation[1], features.count[1] = 12, 104
    assert valid(second, features)
    features.count[1] = 103
    assert not valid(second, features)
    features.count[1], features.generation[1] = 104, 11
    assert not valid(second, features)


@pytest.mark.parametrize("field,value", [("request_slot", -1), ("request_slot", 2),
    ("generation", 11), ("frames", 0), ("frames", 801), ("feature_count", 103),
    ("convolution_width", 99), ("feature_offset", -1), ("feature_offset", 208),
    ("total_input_frames", 0), ("total_input_frames", -1),
    ("total_input_frames", 600), ("total_input_frames", 2**31 - 1),
    ("total_input_frames", -(2**31))])
def test_audio_window_rejects_invalid_scalar_metadata(field, value):
    _, _, _, features = state()
    inputs = window()
    getattr(inputs, field).fill_(value)
    assert not valid(inputs, features)


def test_whole_input_length_distinguishes_equal_short_and_tail_windows():
    _, _, _, features = state()
    short, tail = window(150), window(950, index=1)
    assert short.frames.item() == tail.frames.item() == 150
    assert short.convolution_width.item() == tail.convolution_width.item() == 100
    assert short.total_input_frames.item() == 150
    assert tail.total_input_frames.item() == 950
    assert valid(short, features)
    features.generation[1], features.count[1] = 12, 104
    assert valid(tail, features)
    # This total would describe a 51-frame tail, not the supplied 150 frames.
    tail.total_input_frames.fill_(851)
    assert not valid(tail, features)


def test_contiguous_feature_count_does_not_waive_window_alignment():
    _, _, _, features = state()
    inputs = window()
    features.generation[1], features.count[1] = 12, 1
    inputs.feature_offset.fill_(1)
    assert not valid(inputs, features)


def test_audio_window_rejects_nonfinite_input_bad_gathers_and_bad_lengths():
    _, _, _, features = state()
    for field, offset, value in [("mel", (0, 0), float("nan")),
                                  ("gather_indices", 103, 104), ("cu_seqlens", 0, 1)]:
        inputs = window()
        getattr(inputs, field)[offset] = value
        assert not valid(inputs, features)
    short = window(99)
    assert valid(short, features)
    short.convolution_width.fill_(100)
    assert not valid(short, features)
    short = window(99)
    short.gather_indices[13] = 0
    assert not valid(short, features)


def test_prefill_injects_exact_features_and_rejects_stale_or_out_of_range_state():
    inputs, _, prefill, features = state()
    inputs.token_ids.copy_(torch.tensor([7, 99, 99, 0]))
    prefill.audio_token_indices.copy_(torch.tensor([-1, 0, 1, -1]))
    features.generation[1], features.count[1] = 12, 2
    features.arena[0, 0].fill_(3)
    features.arena[0, 1].fill_(5)
    assert features.validate_prefill(inputs.token_ids, torch.tensor([3]))
    embedding = torch.ones((4, 4), dtype=torch.bfloat16)
    features.inject(inputs, embedding)
    assert torch.equal(embedding, torch.tensor([[1]*4, [3]*4, [5]*4, [1]*4], dtype=torch.bfloat16))
    for epoch, count in [(11, 2), (12, 1), (12, 209)]:
        features.generation[1], features.count[1] = epoch, count
        assert not features.validate_prefill(inputs.token_ids, torch.tensor([3]))
    features.generation[1], features.count[1] = 12, 2
    features.feature_slot.fill_(1)
    assert not features.validate_prefill(inputs.token_ids, torch.tensor([3]))
