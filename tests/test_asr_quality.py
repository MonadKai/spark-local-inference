# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from qualify.asr_quality import edit_distance, evaluate, metric_units, normalize, paired_upper_bound


def test_frozen_normalization_preserves_digits_and_word_characters():
    assert normalize(" Ｈello,  WORLD！ １２３_四\n") == "hello world 123_四"
    assert metric_units("你 好！", "zh", "cer") == "你好"
    assert metric_units("A B!", "en", "cer") == "a b"
    assert metric_units("A B!", "en", "wer") == ["a", "b"]


@pytest.mark.parametrize("left,right,distance", [("", "abc", 3), ("kitten", "sitting", 3),
    (["hello", "world"], ["world"], 1), ("中文", "中问", 1), ("same", "same", 0)])
def test_edit_counts(left, right, distance):
    assert edit_distance(left, right) == edit_distance(right, left) == distance


def test_bootstrap_resamples_edit_totals_instead_of_utterance_rates():
    # The corpus degradation is near 1/101. A mean of utterance rates would
    # produce an upper bound above .5 and fails this independent range check.
    rows = [(0, 1, 1)] * 64 + [(0, 0, 100)] * 64
    assert 0.009 < paired_upper_bound(rows, replicates=500, seed=0) < 0.02
    assert paired_upper_bound([(2, 1, 100)] * 8, replicates=50) == -0.01
    assert paired_upper_bound(rows, replicates=100, seed=42) == paired_upper_bound(rows, replicates=100, seed=42)


def _inputs():
    utterances = [{"id": f"{lang}-{index:03}", "language": lang, "speaker": str(index // 8),
        "text": "one two" if lang == "en" else "你好"} for lang in ("en", "zh") for index in range(128)]
    results = [{"id": row["id"], "text": row["text"], "status": "success"} for row in utterances]
    return {"split": "qualification", "utterances": utterances}, results


def test_identical_engines_pass_all_predeclared_gates():
    manifest, reference = _inputs()
    result = evaluate(manifest, reference, reference)
    assert result["accepted"] and result["valid_complete_pairs"]
    assert all(metric["degradation"] == metric["upper_95_degradation"] == 0
               for key, metric in result["metrics"].items() if key != "en_cer")


def test_missing_and_truncated_requests_stay_in_denominator_and_invalidate():
    manifest, reference = _inputs()
    candidate = [dict(row) for row in reference[1:]]
    candidate[0]["status"] = "truncated"
    result = evaluate(manifest, reference, candidate)
    assert not result["accepted"] and not result["valid_complete_pairs"]
    assert len(result["utterances"]) == 256
    assert result["metrics"]["en_wer"]["units"] == 256
    assert result["metrics"]["en_wer"]["candidate_edits"] == 2


def test_planted_recognition_regression_fails():
    manifest, reference = _inputs()
    candidate = [dict(row, text="wrong") if row["id"].startswith("en-") else row for row in reference]
    result = evaluate(manifest, reference, candidate)
    assert not result["accepted"]
    assert not result["metrics"]["en_wer"]["point_gate"]
    assert not result["metrics"]["en_wer"]["confidence_gate"]


def test_duplicate_results_cannot_replace_a_failed_row():
    manifest, reference = _inputs()
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(manifest, reference, reference + reference[:1])


@pytest.mark.parametrize("split", ["development", None])
def test_development_or_unidentified_corpus_cannot_qualify(split):
    manifest, reference = _inputs()
    manifest["split"] = split
    with pytest.raises(ValueError, match="qualification split"):
        evaluate(manifest, reference, reference)
