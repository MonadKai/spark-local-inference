# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU corruption controls for actual native frame and KV file validation."""

from copy import deepcopy

import pytest
import torch

from qualify.asr_native_frames import PLANES, VOCABULARY, check_frames, scope
from qualify.asr_native_logits import cohorts


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU native frame validation attempted CUDA initialization")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def inputs(tmp_path):
    cases = [dict(id=f"input{i}", prompt_ids=[1, 2, 3], teacher_history=[7, 8, 151645])
        for i in range(2)]
    planned = [([0], None), ([1], None), ([0], False), ([0], True), ([1], False), ([1], True)]
    raw = dict(frames=[], kv_frames=[], parking=[])
    fence = 0
    expected = {}
    for cohort, (order, eviction) in enumerate(planned):
        case = cases[order[0]]
        expected[cohort, 0] = bytearray()
        generation = cohort * 2 + 1
        sequence = [(0, 1, 0, 0, "Prefill"), (1, 2, 1, 0, "Prefill"), (3, 1, 1, 1, "Decode")]
        if eviction is None:
            sequence.append((4, 1, 1, 2, "Decode"))
        else:
            sequence.append((4, 0, 0, 2, "Snapshot"))
            if eviction: sequence.append((0, 1, 0, 2, "Prefill"))
            sequence.append((1 if eviction else 4, 4 if eviction else 1, 1, 2, "Prefill"))
        readmitted = False
        for start, tokens, emitted, history_step, kind in sequence:
            fence += 1
            if kind == "Snapshot":
                raw["parking"].append(dict(cohort=cohort, event="actual_snapshot_completed", fence=fence,
                    request_slot=0, generation=generation, consumed=4, lane_start=4, lane_tokens=0,
                    completion_record=[generation, fence, 4, 2, -1, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, -1]))
                raw["parking"].append(dict(cohort=cohort, event="checkpoint_residency_verified",
                    evicted=eviction, resident_before=4, resident_after=0 if eviction else 4,
                    pressure_reservations=int(eviction)))
                generation += 1
                readmitted = True
                continue
            token = case["teacher_history"][history_step] if emitted else -1
            value = torch.zeros(VOCABULARY, dtype=torch.bfloat16)
            value[3] = (order[0] + 1) / 16
            if emitted: value[token] = 1
            data = value.view(torch.uint8).numpy().tobytes()
            file = f"frame-{fence}.bf16"
            (tmp_path / file).write_bytes(data)
            if emitted: expected[cohort, 0].extend(data)
            lane = dict(row=0, request=0, case=case["id"], scope=scope(cohort, 0), slot=0,
                generation=generation, start=start, tokens=tokens, history_step=history_step)
            raw["frames"].append(dict(cohort=cohort, fence=fence, graph=kind, kind=kind,
                initial_prompt_capacity=8 if kind == "Prefill" else 0, token_capacity=8 if kind == "Prefill" else 1,
                capacity=1, live=1, slots=[0], epochs=[generation], lanes=[lane], raw_frame=file,
                completion_records=[[generation, fence, start + tokens, history_step + emitted,
                    token, 2 if token == 151645 else 1, 0, 0, emitted, 0, 0, 0, -1, 0, 0, 1, token]]))
            planes = []
            for index, name in enumerate(sorted(PLANES)):
                value = torch.full((start + tokens, 8, 128), (order[0] + 1) + index / 64,
                    dtype=torch.bfloat16)
                data = value.view(torch.uint8).numpy().tobytes()
                file = f"kv-{fence}-{name}.bf16"
                (tmp_path / file).write_bytes(data)
                planes.append(dict(buffer=name, file=file, nbytes=len(data)))
                if readmitted:
                    poison = data[:start * 2048] + b"\xc1\x7f" * (tokens * 1024)
                    (tmp_path / f"cohort{cohort}-start{start}-{name}.poison").write_bytes(poison)
            raw["kv_frames"].append(dict(cohort=cohort, request=0, fence=fence, case=case["id"],
                graph=kind, kind=kind, initial_prompt_capacity=8 if kind == "Prefill" else 0,
                token_capacity=8 if kind == "Prefill" else 1, consumed=start + tokens,
                request_slot=0, generation=generation, lane_start=start, lane_tokens=tokens,
                history_step=history_step, original_prompt_tokens=3, actual_page_table_matches=True, planes=planes))
            if readmitted:
                raw["parking"].append(dict(cohort=cohort, event="device_KV_poison", fence=fence,
                    slot=0, generation=generation, start=start, end=start + tokens, planes=56,
                    submitted=True, readback_verified=True, earlier_prefix_unchanged=True,
                    same_comparator_rejected_poison=True))
    return raw, cases, planned, {key: bytes(value) for key, value in expected.items()}


def test_physical_frames_rebuild_distinct_case_traces_and_complete_kv(tmp_path):
    raw, cases, planned, expected = inputs(tmp_path)
    rebuilt, _ = check_frames(tmp_path, raw, cases, planned, history=True)
    assert rebuilt == expected
    assert rebuilt[0, 0] != rebuilt[1, 0]
    assert scope(0, 0) != scope(32, 0)


@pytest.mark.parametrize("defect", ["scope-high-byte", "case", "generation", "fence", "consumed",
    "physical-tail", "token", "missing-zero-prefill", "missing-prefill-and-kv", "missing-plane", "duplicate-plane",
    "kv-tail", "kv-nan", "kv-exact", "kv-case-swap", "missing-parking", "missing-poison",
    "wrong-poison", "snapshot-fence", "snapshot-record-zero-fence", "snapshot-record-old-fence", "residency"])
def test_corrupt_actual_routing_or_complete_kv_is_rejected(tmp_path, defect):
    raw, cases, planned, _ = inputs(tmp_path)
    frame = raw["frames"][8]
    event = raw["kv_frames"][8]
    if defect == "scope-high-byte": frame["lanes"][0]["scope"][12] ^= 1
    if defect == "case": frame["lanes"][0]["case"] = "input1"
    if defect == "generation": frame["lanes"][0]["generation"] -= 1
    if defect == "fence": frame["fence"] -= 1
    if defect == "consumed": event["consumed"] += 1
    if defect == "physical-tail": (tmp_path / frame["raw_frame"]).write_bytes(b"truncated")
    if defect == "token": raw["frames"][9]["completion_records"][0][4] = 9
    if defect == "missing-zero-prefill": raw["kv_frames"].pop(8)
    if defect == "missing-prefill-and-kv":
        raw["frames"].pop(8)
        raw["kv_frames"].pop(8)
    if defect == "missing-plane": event["planes"].pop()
    if defect == "duplicate-plane": event["planes"][-1] = deepcopy(event["planes"][0])
    if defect in ("kv-tail", "kv-nan", "kv-exact"):
        path = tmp_path / event["planes"][0]["file"]
        if defect == "kv-tail": path.write_bytes(path.read_bytes()[:-1])
        else:
            value = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16)
            value[0] = float("nan") if defect == "kv-nan" else 9
            path.write_bytes(value.view(torch.uint8).numpy().tobytes())
    if defect == "kv-case-swap":
        event["planes"][0]["file"] = raw["kv_frames"][4]["planes"][0]["file"]
    if defect == "missing-parking": raw["parking"] = []
    if defect in ("missing-poison", "wrong-poison"):
        path = next(tmp_path.glob("*.poison"))
        if defect == "missing-poison": path.unlink()
        else: path.write_bytes(b"\0" * path.stat().st_size)
    if defect == "snapshot-fence": raw["parking"][0]["fence"] += 1
    if defect in ("snapshot-record-zero-fence", "snapshot-record-old-fence"):
        fence = 0 if defect == "snapshot-record-zero-fence" else raw["parking"][0]["fence"] - 1
        raw["parking"][0]["fence"] = fence
        raw["parking"][0]["completion_record"][1] = fence
    if defect == "residency": raw["parking"][1]["resident_after"] = 1
    with pytest.raises((ValueError, OSError)):
        check_frames(tmp_path, raw, cases, planned, history=True)


def test_complete_schedule_covers_each_case_and_full_scope_without_byte_wrap():
    planned = cohorts(False, 20)
    coverage = {case: set() for case in range(20)}
    scopes = set()
    for cohort, (order, _) in enumerate(planned):
        capacity = 1 if len(order) == 1 else 8 if len(order) == 7 else len(order)
        for row, case in enumerate(order):
            identity = tuple(scope(cohort, row))
            assert identity not in scopes
            scopes.add(identity)
            coverage[case].add((capacity, len(order), row))
    for seen in coverage.values():
        for batch in (1, 2, 4, 8):
            assert any(b == live == batch for b, live, _ in seen)
        for batch in (2, 4, 8):
            assert len({row for b, live, row in seen if b == live == batch}) >= 2
        assert any(b == 8 and live == 7 for b, live, _ in seen)
