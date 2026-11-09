# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU snapshot controls with the actual Transformers DynamicCache.

These exercise observation/storage, not the original checkpoint forward.
"""

import pytest
import torch
from transformers.cache_utils import DynamicCache

from qualify.asr_canonical_stages import snapshot_cached_state


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU cache snapshot control attempted CUDA initialization")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def cache():
    result = DynamicCache()
    for layer in range(28):
        key = torch.arange(8 * 3 * 128, dtype=torch.float32).reshape(1, 8, 3, 128).to(torch.bfloat16)
        result.update(key + layer, key.neg() - layer, layer)
    return result


def test_actual_dynamic_cache_snapshot_keeps_complete_past_and_owns_copies():
    actual = cache()
    first = snapshot_cached_state(actual, 3)
    assert len(first) == 56
    for layer in range(28):
        for name in ("keys", "values"):
            value = first[f"thinker.model.cache.{layer}.{name}"]
            original = getattr(actual.layers[layer], name)
            assert value.shape == (1, 8, 3, 128) and value.dtype == torch.bfloat16
            assert value.device.type == "cpu" and value.is_contiguous()
            assert torch.equal(value.view(torch.uint8), original.view(torch.uint8))
            assert value.data_ptr() != original.data_ptr()
        new = torch.full((1, 8, 1, 128), layer + 100., dtype=torch.bfloat16)
        actual.update(new, -new, layer)
    second = snapshot_cached_state(actual, 4)
    for key, before in first.items():
        assert before.shape[2] == 3
        assert torch.equal(second[key][:, :, :3].contiguous().view(torch.uint8), before.view(torch.uint8))
    actual.layers[0].keys.fill_(0)
    assert second["thinker.model.cache.0.keys"].count_nonzero() > 0


@pytest.mark.parametrize("defect", ["layer", "past-row", "head", "dtype", "nan", "infinity", "consumed"])
def test_incomplete_or_corrupt_actual_cache_cannot_be_published(defect):
    actual = cache()
    consumed = 3
    if defect == "layer": actual.layers.pop()
    if defect == "past-row": actual.layers[17].keys = actual.layers[17].keys[:, :, :-1]
    if defect == "head": actual.layers[17].values = actual.layers[17].values[:, :-1]
    if defect == "dtype": actual.layers[17].values = actual.layers[17].values.float()
    if defect == "nan": actual.layers[17].keys[0, 0, 0, 0] = float("nan")
    if defect == "infinity": actual.layers[17].values[0, 0, 0, 0] = float("inf")
    if defect == "consumed": consumed += 1
    with pytest.raises(ValueError, match="canonical"):
        snapshot_cached_state(actual, consumed)
