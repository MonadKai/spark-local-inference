# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Negative controls for the independent teacher-row decision gates."""

import pytest
import torch
import subprocess
import sys

from qualify.asr_candidate_teacher import (compare_prefill, compare_row, compare_state, compare_trace,
    save_failed_group, save_teacher_traces)


def test_failed_group_preserves_partial_nonfinite_rows_without_reading_undrained_state(tmp_path):
    from safetensors.torch import load_file

    rows = torch.tensor([[1., 2.], [float("nan"), float("inf")]], dtype=torch.bfloat16)
    def forbidden():
        raise AssertionError("unconfirmed fence authorized a device-state read")
    result = save_failed_group({"finished": [row.clone() for row in rows], "not_started": []},
        tmp_path, "failed", False, forbidden)
    assert result["device_state"] == "unavailable_unconfirmed_drain"
    assert result["traces"] == [dict(id="finished", file="failed-finished-partial.safetensors", rows=2)]
    saved = load_file(tmp_path / result["traces"][0]["file"])["logits_bf16"]
    assert torch.equal(saved.view(torch.uint8), rows.view(torch.uint8))


def test_confirmed_failure_and_completed_trace_survive_a_later_probe_error(tmp_path):
    from safetensors.torch import load_file

    rows = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
    complete = save_teacher_traces({"request": rows}, tmp_path, "complete")
    try:
        raise ValueError("later independent projection probe failed")
    except ValueError:
        result = save_failed_group({"request": list(rows)}, tmp_path, "failed", True,
            lambda: {"error_code": torch.tensor([7], dtype=torch.int32)})
    assert result["device_state"] == "saved_after_confirmed_drain"
    assert load_file(tmp_path / result["state_file"])["error_code"].item() == 7
    for entry in [*complete, *result["traces"]]:
        assert torch.equal(load_file(tmp_path / entry["file"])["logits_bf16"].view(torch.uint8), rows.view(torch.uint8))


def test_teacher_top1_gate_rejects_even_with_loose_scalar_budgets():
    reference = torch.tensor([1., 2., 0.], dtype=torch.bfloat16)
    wrong = torch.tensor([3., 2., 0.], dtype=torch.bfloat16)
    result = compare_row(wrong, reference, {"max_abs": 100., "relative_l2": 100.}, 1)
    assert not result["passed"] and result["top1"] == 0


def test_teacher_scalar_gate_rejects_without_a_top1_change():
    reference = torch.tensor([1., 2., 0.], dtype=torch.bfloat16)
    result = compare_row(reference + 1, reference, {"max_abs": .5, "relative_l2": .1}, 1)
    assert not result["passed"] and result["top1"] == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("nonfinite_reference", [False, True])
def test_nonfinite_logits_never_form_a_passing_row(value, nonfinite_reference):
    finite = torch.tensor([1., 2., 0.], dtype=torch.bfloat16)
    bad = finite.clone()
    bad[0] = value
    with pytest.raises(ValueError, match="nonfinite"):
        compare_row(finite if nonfinite_reference else bad, bad if nonfinite_reference else finite,
            {"max_abs": 100., "relative_l2": 100.}, 1)


def test_first_maximum_ties_and_exact_rows_pass():
    values = torch.tensor([2., 2., 0.], dtype=torch.bfloat16)
    result = compare_row(values, values.clone(), {"max_abs": 0., "relative_l2": 0.}, 0)
    assert result["passed"] and result["max_abs"] == result["relative_l2"] == 0


def test_missing_teacher_rows_fail_even_when_every_returned_row_is_exact():
    reference = torch.tensor([[1., 2., 0.], [2., 0., 1.], [0., 1., 2.]], dtype=torch.bfloat16)
    case = dict(teacher_rows=[dict(max_abs=0., relative_l2=0.)] * 3, teacher_top1=[1, 0, 2])
    result = compare_trace(reference[:2], reference, case, reference)
    assert not result["passed"] and not result["complete_teacher_trace"]
    complete = compare_trace(reference, reference, case, reference)
    assert complete["passed"] and complete["complete_teacher_trace"]
    retired = compare_trace(reference[:2], reference, case, reference, retired_after=2)
    assert retired["passed"] and not retired["complete_teacher_trace"] and retired["required_rows"] == 2
    missing_retired = compare_trace(reference[:1], reference, case, reference, retired_after=2)
    assert not missing_retired["passed"]


def test_state_gate_rejects_a_changed_kv_row_and_signed_zero():
    baseline = dict(keys=torch.zeros((28, 2, 8, 128), dtype=torch.bfloat16),
        layers=torch.arange(28, dtype=torch.int32))
    assert compare_state(baseline, baseline)["passed"]
    changed = {name: value.clone() for name, value in baseline.items()}
    changed["keys"][27, 1, 7, 127] = 1
    result = compare_state(changed, baseline)
    assert not result["passed"] and result["fields"]["keys"]["differing_elements"] == 1
    changed["keys"].zero_()
    changed["keys"].view(torch.int16)[27, 1, 7, 127] = -32768
    result = compare_state(changed, baseline)
    assert not result["passed"] and result["fields"]["keys"]["differing_elements"] == 0


@pytest.mark.parametrize("change", ["missing", "shape", "dtype", "nonfinite", "empty"])
def test_state_gate_rejects_incomplete_or_invalid_observations(change):
    baseline = dict(features=torch.ones((3, 2048), dtype=torch.bfloat16))
    actual = {name: value.clone() for name, value in baseline.items()}
    if change == "missing":
        actual.clear()
    elif change == "shape":
        actual["features"] = actual["features"][:2]
    elif change == "dtype":
        actual["features"] = actual["features"].float()
    elif change == "nonfinite":
        actual["features"][0, 0] = float("nan")
    else:
        actual["features"] = torch.empty((0, 2048), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        compare_state(actual, baseline)


def test_top1_report_retains_both_cross_logits_and_tie_margins():
    actual = torch.tensor([3, 2, 3], dtype=torch.bfloat16)
    reference = torch.tensor([2, 3, 1], dtype=torch.bfloat16)
    result = compare_row(actual, reference, dict(max_abs=9, relative_l2=9), 1)
    assert not result["passed"] and result["top1"] == 0
    assert result["candidate_top1_logit"] == result["reference_top1_logit"] == 3
    assert result["candidate_logit_at_reference_top1"] == result["reference_logit_at_candidate_top1"] == 2
    assert result["candidate_margin"] == 0 and result["reference_margin"] == 1


def prefill_case():
    reference = torch.tensor([[[2., 3., 0.], [4., 1., 0.], [0., 2., 5.]]], dtype=torch.bfloat16)
    case = dict(prompt_ids=[1, 2, 3], teacher_rows=[dict(max_abs=0., relative_l2=0.)],
        teacher_top1=[2], stages={"thinker.lm_head.output": dict(shape=[1, 3, 3], max_abs=0., relative_l2=0.)})
    return reference, case


def test_full_prefill_gate_checks_nondecision_positions_and_the_whole_matrix():
    reference, case = prefill_case()
    assert compare_prefill(reference[0], reference, case)["passed"]
    changed = reference[0].clone()
    changed[0, 2] = 100
    result = compare_prefill(changed, reference, case)
    assert not result["passed"] and result["generation_decision"]["passed"]
    assert result["top1_mismatch_count"] == 1
    assert result["top1_mismatches"][0]["position"] == 0
    assert result["max_abs"] == 100


def test_full_prefill_report_retains_lowest_id_ties_cross_logits_and_second_ids():
    reference, case = prefill_case()
    case["stages"]["thinker.lm_head.output"].update(max_abs=10., relative_l2=10.)
    actual = reference[0].clone()
    actual[0] = torch.tensor([4., 4., 4.], dtype=torch.bfloat16)
    result = compare_prefill(actual, reference, case)
    assert result["passed"]  # Nondecision top-1 is reported, not silently excluded.
    gap, = result["top1_mismatches"]
    assert gap["candidate"]["first_id"] == 0 and gap["candidate"]["second_id"] == 1
    assert gap["reference"]["first_id"] == 1 and gap["reference"]["second_id"] == 0
    assert gap["candidate"]["margin"] == 0 and gap["reference"]["margin"] == 1
    assert gap["candidate_logit_at_reference_top1"] == 4
    assert gap["reference_logit_at_candidate_top1"] == 2


@pytest.mark.parametrize("change", ["short", "dtype", "nonfinite", "reference_nonfinite", "decision"])
def test_full_prefill_rejects_missing_invalid_or_changed_decision_rows(change):
    reference, case = prefill_case()
    actual = reference[0].clone()
    if change == "short":
        actual = actual[:-1]
    elif change == "dtype":
        actual = actual.float()
    elif change == "nonfinite":
        actual[0, 0] = float("nan")
    elif change == "reference_nonfinite":
        reference[0, 0, 0] = float("inf")
    else:
        case["stages"]["thinker.lm_head.output"].update(max_abs=10., relative_l2=10.)
        actual[-1, 0] = 8
        assert not compare_prefill(actual, reference, case)["passed"]
        return
    with pytest.raises(ValueError):
        compare_prefill(actual, reference, case)


@pytest.mark.parametrize("name", ["SIGINT", "SIGTERM"])
def test_ignored_inherited_signals_enter_cleanup_and_restore_the_prior_handlers(name):
    code = """
import os, signal, sys
from qualify.asr_candidate_teacher import diagnostic_signals
sig = getattr(signal, sys.argv[1])
for value in (signal.SIGINT, signal.SIGTERM):
    signal.signal(value, signal.SIG_IGN)
closed = False
try:
    with diagnostic_signals():
        try:
            try:
                os.kill(os.getpid(), sig)
                raise AssertionError('termination signal was ignored')
            except Exception as error:
                raise RuntimeError('termination was swallowed by ordinary case reporting') from error
        finally:
            assert all(signal.getsignal(value) == signal.SIG_IGN for value in (signal.SIGINT, signal.SIGTERM))
            closed = True
except KeyboardInterrupt as error:
    assert str(int(sig)) in str(error)
else:
    raise AssertionError('termination did not enter cleanup')
assert closed
assert all(signal.getsignal(value) == signal.SIG_IGN for value in (signal.SIGINT, signal.SIGTERM))
print('termination reached cleanup')
"""
    result = subprocess.run([sys.executable, "-c", code, name], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "termination reached cleanup"
