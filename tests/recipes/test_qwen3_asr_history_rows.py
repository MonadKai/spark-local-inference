# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Only unused history arithmetic can skip its operand reads."""

import pytest
import torch

from ignition.recipes.qwen3_asr.replay import history_projection_count


@pytest.mark.parametrize("count", [-1, 0, 1, 2, 7, 8, 9, 2**31 - 1])
@pytest.mark.parametrize("original", [-1, 0, 1, 4, 8, 8192])
def test_history_count_preserves_selection_and_invalid_counts(count, original):
    rows = torch.arange(8)
    live = torch.tensor([count], dtype=torch.int32)
    boundary = torch.tensor([original], dtype=torch.int32)
    # Non-contiguous positions ensure this does not infer selection from the
    # first position, a prompt count, or an assumed contiguous history suffix.
    positions = torch.tensor([0, 7, -1, 8191, 2, 1, 8, 3])
    actual = history_projection_count(rows, live, positions, boundary)
    selected = [i for i, position in enumerate(positions.tolist())
        if i < count and position >= original]
    expected = 0 if 0 <= count <= len(rows) and not selected else count
    assert actual.shape == live.shape and actual.dtype == live.dtype
    assert actual.item() == expected
    assert live.item() == count


def test_history_in_inactive_rows_does_not_trigger_operand_reads():
    rows = torch.arange(8)
    count = torch.tensor([2], dtype=torch.int32)
    original = torch.tensor([19], dtype=torch.int32)
    positions = torch.tensor([0, 1, 19, 20, 21, 22, 23, 24])
    assert history_projection_count(rows, count, positions, original).item() == 0
    positions[1] = 19
    assert history_projection_count(rows, count, positions, original).item() == 2
