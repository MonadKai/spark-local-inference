# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import random

import pytest

from qualify.asr_corpus import select


def _rows():
    return [dict(id=f"{speaker:08}-{utterance:03}", speaker=f"{speaker:08}", samples=utterance * 160)
            for speaker in range(20) for utterance in range(22)]


def test_selection_is_independent_of_enumeration_and_uses_exact_duration_quantiles():
    rows = _rows()
    random.Random(9).shuffle(rows)
    selected = select(rows)
    assert len(selected) == 128
    assert {row["speaker"] for row in selected} == {f"{speaker:08}" for speaker in range(16)}
    assert [row["id"] for row in selected[:8]] == [f"00000000-{position:03}" for position in range(0, 22, 3)]
    assert selected == select(list(reversed(rows)))


def test_insufficient_first_speaker_cannot_be_replaced_after_selection():
    rows = [row for row in _rows() if row["speaker"] != "00000000" or row["samples"] < 7 * 160]
    with pytest.raises(ValueError, match="fewer than eight"):
        select(rows)


def test_duplicate_selected_utterance_is_not_silently_replaced():
    rows = _rows()
    rows[0]["id"] = rows[3]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        select(rows)
