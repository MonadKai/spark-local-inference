# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal

import pytest

from qualify.asr_long_corpus import interval, safe_boundary, check_rttm


def test_boundary_never_cuts_an_overlapping_speaker_or_adds_silence_anchors():
    protected = [interval("0", "30.00001", "a", "a1", "first"),
                 interval("29", "33.615", "b", "b1", "overlap"),
                 interval("33.662", "40", "a", "a2", "next")]
    assert safe_boundary(protected, 30, 100 * 16000) == 537840
    # An anchor in silence does not manufacture a new candidate; the next
    # original annotation boundary wins even though it is later than anchor.
    assert safe_boundary(protected, Decimal("33.64"), 100 * 16000) == 538592


def test_boundary_rejects_continuous_speech_instead_of_fabricating_a_cut():
    with pytest.raises(ValueError, match="no safe original"):
        safe_boundary([interval("0", "100", "a", "a1", "continuous")], 30, 100 * 16000)


def test_rttm_is_only_a_rounded_crosscheck_and_cannot_replace_original_times(tmp_path):
    row = interval("1.123456789", "2.234567890", "speaker", "1", "word")
    path = tmp_path / "meeting.rttm"
    path.write_text("SPEAKER meeting 1 1.1235 1.1111 <NA> <NA> speaker <NA> <NA>\n")
    assert check_rttm(path, [row], "meeting")["rows"] == 1
    assert row["start"] == Decimal("1.123456789")
    path.write_text("SPEAKER meeting 1 1.1235 1.1113 <NA> <NA> speaker <NA> <NA>\n")
    with pytest.raises(ValueError, match="does not match"):
        check_rttm(path, [row], "meeting")
    path.write_text("")
    with pytest.raises(ValueError, match="does not match"):
        check_rttm(path, [row], "meeting")


@pytest.mark.parametrize("start,end", [("NaN", "1"), ("0", "Infinity"), ("-1", "2"), ("2", "1")])
def test_annotation_nonfinite_and_reversed_extents_fail(start, end):
    with pytest.raises(ValueError, match="invalid annotation"):
        interval(start, end, "a", "id", "text")
