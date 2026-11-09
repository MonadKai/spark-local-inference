# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Apply frozen teacher budgets to the separate Rust Worker observation.

Batch/slot and generated-history runs have different fixed cohort schedules.
Every planned request remains in the result, including missing and partial
files. This CPU check does not qualify all model inputs, operators, asynchronous
cancellation, recognition quality or performance.
"""

import argparse
import json
from pathlib import Path
import re
import sys


VOCABULARY = 151936


def cohorts(history, count=4):
    if history:
        return [([i], None) for i in range(count)] + [([i], evicted)
            for i in range(count) for evicted in (False, True)]
    if count == 20:
        result = [([i], None) for i in range(count)]
        for batch in (2, 4, 8):
            for start in range(0, count, batch):
                order = [(start + row) % count for row in range(batch)]
                result.extend(((order, None), (order[::-1], None)))
        for start in range(0, count, 7):
            order = [(start + row) % count for row in range(7)]
            result.extend(((order, None), (order[::-1], None)))
        return result
    return [(order, None) for order in ([0], [1], [2], [3], [0, 1], [2, 3],
        [0, 1, 2, 3], [3, 2, 1, 0], [0, 1, 2, 3, 0, 1, 2, 3],
        [3, 2, 1, 0, 3, 2, 1, 0], [0, 1, 2, 3, 0, 1, 2])]


def check(source, cases, contract, references, *, history, expected_identity):
    import torch
    from safetensors import SafetensorError, safe_open
    from qualify.asr_candidate_teacher import compare_row, validate_frozen_contract

    if sys.byteorder != "little" or len(cases) not in (4, 20) \
            or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("four pilot or twenty complete distinct native cases and little-endian BF16 storage required")
    if type(history) is not bool or not re.fullmatch(r"[0-9a-f]{64}", expected_identity):
        raise ValueError("explicit observation mode and complete prepared identity required")
    planned = cohorts(history, len(cases))
    result = dict(scope=__doc__.strip(), status="checking", history=history,
        expected_requests=sum(len(order) for order, _ in planned), cases=[], errors=[],
        full_model_numerical_qualification=False, performance_qualified=False)
    raw = {}
    try:
        raw = json.loads((source / "result.json").read_bytes())
        if not isinstance(raw, dict):
            raise ValueError("native summary must be a JSON object")
    except (OSError, ValueError) as error:
        raw = {}
        result["errors"].append(dict(reason="missing or invalid native summary", type=type(error).__name__))
    result.update(native_status=raw.get("status"), prepared_identity=raw.get("prepared_identity"))
    contract_error = None
    try:
        validate_frozen_contract(contract)
    except (KeyError, TypeError, ValueError) as error:
        contract_error = "invalid frozen independent contract: " + type(error).__name__
        result["errors"].append(contract_error)
    expected_status = ("complete_native_four_input_diagnostic" if len(cases) == 4
        else "complete_native_twenty_input_diagnostic")
    if raw.get("status") != expected_status or raw.get("errors") != [] \
            or raw.get("shutdown") != dict(drained=True, error=None) \
            or raw["shutdown"]["drained"] is not True:
        result["errors"].append("native observation did not complete and drain without errors")
    if raw.get("generated_history_probe") is not history or raw.get("prepared_identity") != expected_identity:
        result["errors"].append("native observation mode or prepared identity differs")
    actual_cohorts = raw.get("cohorts", [])
    if not isinstance(actual_cohorts, list):
        actual_cohorts = []
    if len(actual_cohorts) != len(planned):
        result["errors"].append("native cohort count differs from the selected fixed schedule")
    if not history:
        observed_capacities = raw.get("actual_decode_capacity_and_live_rows", [])
        valid_capacities = isinstance(observed_capacities, list) and all(
            isinstance(row, list) and len(row) == 2 and all(type(n) is int for n in row)
            for row in observed_capacities)
        capacities = {tuple(row) for row in observed_capacities} if valid_capacities else set()
        if not all((b, b) in capacities for b in (1, 2, 4, 8)) \
                or not any(0 < live < capacity for capacity, live in capacities):
            result["errors"].append("native full and partial decode batches were not observed")
    expected_files = {f"cohort{cohort}-request{request}-logits.bf16"
        for cohort, (order, _) in enumerate(planned) for request in range(len(order))}
    if {p.name for p in source.glob("cohort*-request*-logits.bf16")} - expected_files:
        result["errors"].append("unexpected request trace outside the fixed schedule")
    frozen_by_id = {}
    try:
        frozen_cases = contract["cases"]
        if not isinstance(frozen_cases, list):
            raise ValueError("frozen cases must be a list")
        for case in frozen_cases:
            if not isinstance(case, dict) or not isinstance(case.get("id"), str) \
                    or case["id"] in frozen_by_id:
                raise ValueError("frozen cases must have distinct string identities")
            frozen_by_id[case["id"]] = case
    except (KeyError, TypeError, ValueError) as error:
        contract_error = "invalid frozen case index: " + type(error).__name__
        result["errors"].append(contract_error)
    rebuilt = None
    if len(cases) == 20:
        from qualify.asr_native_frames import check_frames

        if set(frozen_by_id) != {case["id"] for case in cases}:
            contract_error = "native twenty-input identity set differs from the complete frozen contract"
            result["errors"].append(contract_error)
        try:
            rebuilt, witnesses = check_frames(source, raw, cases, planned, history=history)
            result["physical_frames_and_complete_same_path_kv_checked"] = True
            result["actual_case_decode_witnesses"] = {name: sorted(rows) for name, rows in witnesses.items()}
        except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as error:
            rebuilt = {}
            result["physical_frames_and_complete_same_path_kv_checked"] = False
            result["errors"].append(dict(reason="native physical frame or KV validation failed",
                type=type(error).__name__, message=str(error)))
    baseline = {}
    for cohort, (order, eviction) in enumerate(planned):
        observed = actual_cohorts[cohort] if cohort < len(actual_cohorts) else {}
        if not isinstance(observed, dict):
            observed = {}
        cohort_valid = (observed.get("cohort") == cohort and observed.get("order") == order
            and observed.get("checkpoint_evicted") is eviction
            and isinstance(observed.get("natural_stop"), list)
            and len(observed["natural_stop"]) == len(order)
            and all(value is True for value in observed["natural_stop"]))
        if eviction is not None:
            cohort_valid = cohort_valid and all(observed.get(key) is True for key in
                ("real_snapshot_parking", "readmitted", "actual_history_prefill")) \
                and type(observed.get("poisoned_prefill_submissions")) is int \
                and observed["poisoned_prefill_submissions"] > 0
        for request, case_index in enumerate(order):
            case = cases[case_index]
            row = dict(cohort=cohort, request=request, id=case["id"], rows=[], passed=False)
            result["cases"].append(row)
            try:
                if contract_error:
                    raise ValueError(contract_error)
                if case.get("endpoint", "original_sdk_chat") != "original_sdk_chat":
                    raise ValueError("the frozen numerical contract requires original SDK/chat inputs")
                if "case_endpoints" in raw and raw["case_endpoints"] != [dict(id=value["id"],
                        endpoint=value.get("endpoint", "original_sdk_chat")) for value in cases]:
                    raise ValueError("observed native input endpoints differ from the requested cases")
                frozen = frozen_by_id[case["id"]]
                if case["prompt_ids"] != frozen["prompt_ids"] or case["teacher_history"] != frozen["teacher_history"]:
                    raise ValueError("native input prompt or teacher history differs from the frozen contract")
                row["expected_teacher_rows"] = len(frozen["teacher_rows"])
                data = (source / f"cohort{cohort}-request{request}-logits.bf16").read_bytes()
                if not data or len(data) % (VOCABULARY * 2):
                    raise ValueError("native trace has no complete BF16 rows or an incomplete final row")
                value = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(-1, VOCABULARY)
                row["actual_natural_rows"] = len(value)
                if not torch.isfinite(value).all():
                    raise ValueError("native trace contains nonfinite logits")
                tokens = observed.get("actual_tokens", [])
                if not isinstance(tokens, list) or len(tokens) != len(order) \
                        or not isinstance(tokens[request], list) \
                        or any(type(token) is not int for token in tokens[request]) \
                        or tokens[request] != value.argmax(-1).tolist() \
                        or not tokens[request] or tokens[request][-1] not in (151643, 151645) \
                        or any(token in (151643, 151645) for token in tokens[request][:-1]):
                    raise ValueError("natural token trace does not match every observed BF16 argmax and EOS")
                with safe_open(references[case["id"]] / (case["id"] + "-cached-teacher.safetensors"),
                        framework="pt") as reference:
                    expected = reference.get_tensor("logits_bf16")
                    target = reference.get_tensor("target_ids").reshape(-1).tolist()
                if target != frozen["teacher_history"] or expected.shape != (len(target), VOCABULARY) \
                        or expected.argmax(-1).tolist() != frozen["teacher_top1"]:
                    raise ValueError("independent cached teacher reference differs from the frozen contract")
                row["rows"] = [dict(step=i, **compare_row(value[i], expected[i],
                    frozen["teacher_rows"][i], frozen["teacher_top1"][i]))
                    for i in range(min(len(value), len(expected)))]
                if rebuilt is not None and rebuilt.get((cohort, request)) != data:
                    raise ValueError("native request trace differs from its actual completed physical rows")
                if cohort < len(cases):
                    baseline[case_index] = data
                row["same_path_exact_bytes"] = baseline.get(case_index) == data
                row["passed"] = (cohort_valid and row["same_path_exact_bytes"]
                    and len(value) >= len(expected) and len(row["rows"]) == len(expected)
                    and all(item["passed"] for item in row["rows"]))
                if not row["passed"]:
                    raise ValueError("missing or failing teacher rows, cohort metadata or exact same-path bytes")
            except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError, SafetensorError) as error:
                message = ("unreadable native or reference file" if isinstance(error, OSError)
                    else "invalid independent safetensors reference" if isinstance(error, SafetensorError)
                    else str(error))
                row["error"] = dict(type=type(error).__name__, message=message)
                result["errors"].append(dict(cohort=cohort, request=request, id=case["id"],
                    reason=message))
    result["status"] = "complete_finite_native_scalar_checks_only" if not result["errors"] else "failed_or_incomplete"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--canonical-observations", type=Path, action="append", required=True)
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--suite", choices=("batch-slot", "generated-history"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve the previous result; choose a new output")
    contract = json.loads(args.contract.read_bytes())
    references = {}
    for root in args.canonical_observations:
        record = json.loads((root / "result.json").read_bytes())
        if record["errors"] or record["candidate_executed"] or record["layout"] not in ("canonical", "original") \
                or record["transformers_revision"] != contract["transformers_revision"]:
            parser.error("complete independent canonical observations at the frozen revision required")
        for case in record["cases"]:
            if case["id"] in references:
                parser.error("duplicate independent input identity")
            references[case["id"]] = root
    import torch

    torch.set_num_threads(2)
    result = check(args.source, json.loads(args.cases.read_bytes()), contract, references,
        history=args.suite == "generated-history", expected_identity=args.expected_identity)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    raise SystemExit(int(bool(result["errors"])))


if __name__ == "__main__":
    main()
