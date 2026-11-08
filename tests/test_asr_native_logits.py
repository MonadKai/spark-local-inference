# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU controls for the native BF16 evidence validator, not model acceptance."""

import json

import pytest
import torch
from safetensors.torch import save_file

from qualify.asr_native_logits import check


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU evidence validation must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def evidence(tmp_path, *, history=False):
    source, references = tmp_path / "native", tmp_path / "reference"
    source.mkdir(); references.mkdir()
    cases = [dict(id=f"input{i}", prompt_ids=[151644, 151676, 151645],
        teacher_history=[2, 151645]) for i in range(4)]
    contract = dict(numerical_policy="qwen3-asr.bf16-fp32.v1", candidate_used_for_budgets=False,
        generation_decision_top1_mismatches_allowed=0,
        cases=[dict(**case, teacher_top1=[2, 151645], teacher_rows=[
        dict(max_abs=0., relative_l2=0.), dict(max_abs=0., relative_l2=0.)]) for case in cases])
    logits = torch.zeros((2, 151936), dtype=torch.bfloat16)
    logits[0, 2] = 1; logits[1, 151645] = 1
    data = logits.view(torch.uint8).numpy().tobytes()
    for case in cases:
        save_file(dict(logits_bf16=logits, target_ids=torch.tensor([[2, 151645]])),
            references / (case["id"] + "-cached-teacher.safetensors"))
    orders = [[0], [1], [2], [3], [0, 1], [2, 3], [0, 1, 2, 3], [3, 2, 1, 0],
        [0, 1, 2, 3, 0, 1, 2, 3], [3, 2, 1, 0, 3, 2, 1, 0], [0, 1, 2, 3, 0, 1, 2]]
    if history:
        orders = [[0], [1], [2], [3], [0], [0], [1], [1], [2], [2], [3], [3]]
    raw = dict(status="complete_native_four_input_diagnostic", errors=[],
        shutdown=dict(drained=True, error=None), prepared_identity="0" * 64,
        generated_history_probe=history, actual_decode_capacity_and_live_rows=[
            [1, 1], [2, 2], [4, 4], [8, 8], [8, 7]], cohorts=[])
    for index, order in enumerate(orders):
        evicted = bool((index - 4) % 2) if history and index >= 4 else None
        raw["cohorts"].append(dict(cohort=index, order=order, checkpoint_evicted=evicted,
            natural_stop=[True] * len(order), actual_tokens=[[2, 151645] for _ in order],
            real_snapshot_parking=evicted is not None, readmitted=evicted is not None,
            actual_history_prefill=evicted is not None, poisoned_prefill_submissions=int(evicted is not None)))
        for request in range(len(order)):
            (source / f"cohort{index}-request{request}-logits.bf16").write_bytes(data)
    (source / "result.json").write_text(json.dumps(raw))
    return source, cases, contract, {case["id"]: references for case in cases}, raw


def evaluate(inputs, *, history=False, identity="0" * 64):
    source, cases, contract, references, raw = inputs
    (source / "result.json").write_text(json.dumps(raw))
    return check(source, cases, contract, references, history=history, expected_identity=identity)


@pytest.mark.parametrize("history,requests", [(False, 39), (True, 12)])
def test_fixed_schedules_apply_every_frozen_row(tmp_path, history, requests):
    result = evaluate(evidence(tmp_path, history=history), history=history)
    assert result["status"] == "complete_finite_native_scalar_checks_only"
    assert not result["errors"] and len(result["cases"]) == result["expected_requests"] == requests
    assert sum(len(row["rows"]) for row in result["cases"]) == requests * 2
    assert all(row["same_path_exact_bytes"] for row in result["cases"])
    assert not result["full_model_numerical_qualification"] and not result["performance_qualified"]


@pytest.mark.parametrize("failure", ["missing", "partial", "nonfinite", "scalar", "top1", "exact"])
def test_corrupted_native_bytes_are_rejected_without_dropping_requests(tmp_path, failure):
    inputs = evidence(tmp_path)
    source, _, contract, _, raw = inputs
    path = source / "cohort4-request0-logits.bf16"
    if failure == "missing":
        path.unlink()
    elif failure == "partial":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        value = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16).reshape(2, -1)
        value[0, 5] = float("nan") if failure == "nonfinite" else .5
        if failure == "top1":
            value[0, 5] = 2
            raw["cohorts"][4]["actual_tokens"][0][0] = 5
        if failure == "exact":
            for budget in contract["cases"][0]["teacher_rows"]:
                budget.update(max_abs=1., relative_l2=1.)
        path.write_bytes(value.view(torch.uint8).numpy().tobytes())
    result = evaluate(inputs)
    assert result["status"] == "failed_or_incomplete" and result["errors"]
    assert len(result["cases"]) == result["expected_requests"] == 39
    failed = result["cases"][4]
    assert failed["cohort"] == 4 and not failed["passed"]
    if failure == "exact":
        assert all(row["passed"] for row in failed["rows"])
        assert not failed["same_path_exact_bytes"]


@pytest.mark.parametrize("failure", ["identity", "schedule", "mode", "park", "token", "extra-token", "partial-batch", "drain"])
def test_metadata_cannot_promote_an_incomplete_run(tmp_path, failure):
    history = failure == "park"
    inputs = evidence(tmp_path, history=history)
    raw = inputs[-1]
    if failure == "identity": raw["prepared_identity"] = "other"
    if failure == "schedule": raw["cohorts"].pop()
    if failure == "mode": raw["generated_history_probe"] = True
    if failure == "park": raw["cohorts"][4]["actual_history_prefill"] = False
    if failure == "token": raw["cohorts"][4]["actual_tokens"] = None
    if failure == "extra-token": raw["cohorts"][0]["actual_tokens"].append([2, 151645])
    if failure == "partial-batch": raw["actual_decode_capacity_and_live_rows"].pop()
    if failure == "drain": raw["shutdown"]["drained"] = False
    result = evaluate(inputs, history=history)
    assert result["status"] == "failed_or_incomplete" and result["errors"]
    assert len(result["cases"]) == (12 if history else 39)


def test_matching_traces_cannot_continue_after_eos(tmp_path):
    inputs = evidence(tmp_path)
    source, _, _, _, raw = inputs
    for path in source.glob("*-logits.bf16"):
        data = path.read_bytes()
        path.write_bytes(data + data[151936 * 2:])
    for cohort in raw["cohorts"]:
        for tokens in cohort["actual_tokens"]:
            tokens.append(151645)
    result = evaluate(inputs)
    assert result["status"] == "failed_or_incomplete" and len(result["cases"]) == 39
    assert not any(row["passed"] for row in result["cases"])


def test_full_natural_trace_may_exceed_the_frozen_teacher_prefix(tmp_path):
    inputs = evidence(tmp_path)
    source, cases, contract, references, raw = inputs
    cases[0]["teacher_history"] = [2, 3]
    contract["cases"][0].update(teacher_history=[2, 3], teacher_top1=[2, 3])
    first = torch.frombuffer(bytearray((source / "cohort0-request0-logits.bf16").read_bytes()),
        dtype=torch.bfloat16).reshape(2, -1)
    value = torch.cat((first[:1], torch.zeros_like(first[:1]), first[1:]))
    value[1, 3] = 1
    for cohort in raw["cohorts"]:
        for request, index in enumerate(cohort["order"]):
            if index == 0:
                (source / f'cohort{cohort["cohort"]}-request{request}-logits.bf16').write_bytes(
                    value.view(torch.uint8).numpy().tobytes())
                cohort["actual_tokens"][request] = [2, 3, 151645]
    save_file(dict(logits_bf16=value[:2].contiguous(), target_ids=torch.tensor([[2, 3]])),
        references["input0"] / "input0-cached-teacher.safetensors")
    result = evaluate(inputs)
    assert not result["errors"] and len(result["cases"]) == 39
    assert all(row["actual_natural_rows"] == 3 and len(row["rows"]) == 2
        for row in result["cases"] if row["id"] == "input0")


def test_corrupt_reference_keeps_all_planned_requests(tmp_path):
    inputs = evidence(tmp_path)
    (inputs[3]["input0"] / "input0-cached-teacher.safetensors").write_bytes(b"truncated")
    result = evaluate(inputs)
    assert result["status"] == "failed_or_incomplete" and len(result["cases"]) == 39
    assert all(not row["passed"] and row["error"]["type"] == "SafetensorError"
        for row in result["cases"] if row["id"] == "input0")


@pytest.mark.parametrize("field,value", [("numerical_policy", "other"),
    ("candidate_used_for_budgets", True), ("generation_decision_top1_mismatches_allowed", 1)])
def test_existing_frozen_contract_rejections_apply(tmp_path, field, value):
    inputs = evidence(tmp_path)
    inputs[2][field] = value
    result = evaluate(inputs)
    assert result["status"] == "failed_or_incomplete" and len(result["cases"]) == 39
    assert not any(row["passed"] for row in result["cases"])


@pytest.mark.parametrize("failure", ["null", "mapping", "duplicate"])
def test_invalid_contract_case_index_keeps_planned_requests(tmp_path, failure):
    inputs = evidence(tmp_path)
    contract = inputs[2]
    contract["cases"] = (None if failure == "null" else {} if failure == "mapping"
        else [contract["cases"][0], contract["cases"][0]])
    result = evaluate(inputs)
    assert result["status"] == "failed_or_incomplete" and len(result["cases"]) == 39
    assert not any(row["passed"] for row in result["cases"])


@pytest.mark.parametrize("failure", ["missing", "truncated", "null", "list"])
def test_invalid_summary_keeps_all_planned_requests(tmp_path, failure):
    inputs = evidence(tmp_path)
    path = inputs[0] / "result.json"
    if failure == "missing": path.unlink()
    else: path.write_text({"truncated": '{"status":', "null": "null", "list": "[]"}[failure])
    result = check(*inputs[:4], history=False, expected_identity="0" * 64)
    assert result["status"] == "failed_or_incomplete" and len(result["cases"]) == 39
    assert not any(row["passed"] for row in result["cases"])
