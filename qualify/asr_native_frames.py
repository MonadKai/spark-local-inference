# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate completed native rows and complete request-owned KV observations.

This diagnostic checks same-path state and routing. It declares no new
cross-engine scalar budget and does not qualify production scheduling.
"""

from collections import defaultdict


VOCABULARY = 151936
PLANES = {f"cache.layer{layer}.{field}" for layer in range(28)
    for field in ("key", "value")}


def scope(cohort, request):
    return list(b"ASRLOGIT" + cohort.to_bytes(8, "little") + request.to_bytes(8, "little"))


def check_frames(source, raw, cases, planned, *, history):
    """Rebuild traces from physical frames; fail closed on missing metadata."""
    import torch

    traces = defaultdict(bytearray)
    witnesses = defaultdict(set)
    expected_kv = {}
    seen_fences = set()
    owners = {}
    consumed_by_handle = {}
    handles_by_request = defaultdict(set)
    last_fence = 0
    for frame in raw["frames"]:
        cohort, fence = frame["cohort"], frame["fence"]
        if type(cohort) is not int or not 0 <= cohort < len(planned):
            raise ValueError("native frame is outside its fixed cohort schedule")
        order, _ = planned[cohort]
        capacity, live = frame["capacity"], frame["live"]
        if type(fence) is not int or fence <= last_fence or fence in seen_fences:
            raise ValueError("native completion fence is stale or repeated")
        last_fence = fence
        seen_fences.add(fence)
        if type(capacity) is not int or capacity not in (1, 2, 4, 8) \
                or type(live) is not int or not 0 < live <= capacity \
                or len(frame["lanes"]) != live or len(frame["slots"]) != capacity \
                or len(frame["epochs"]) != capacity:
            raise ValueError("native physical frame geometry differs")
        data = (source / frame["raw_frame"]).read_bytes()
        if len(data) != capacity * VOCABULARY * 2:
            raise ValueError("native physical logit frame is incomplete")
        active_slots, active_requests = set(), set()
        for row, lane in enumerate(frame["lanes"]):
            request = lane["request"]
            if type(request) is not int or not 0 <= request < len(order):
                raise ValueError("native frame request is outside its cohort")
            case = cases[order[request]]
            slot, generation = lane["slot"], lane["generation"]
            if type(slot) is not int or not 0 <= slot < len(frame["completion_records"]) \
                    or any(type(lane[name]) is not int or lane[name] < 0
                        for name in ("start", "tokens", "history_step")):
                raise ValueError("native frame has an invalid slot or lane extent")
            identity = (cohort, request)
            if lane["row"] != row or lane["case"] != case["id"] \
                    or lane["scope"] != scope(cohort, request) \
                    or request in active_requests or slot in active_slots \
                    or frame["slots"][row] != slot or frame["epochs"][row] != generation \
                    or type(generation) is not int or generation <= 0:
                raise ValueError("native row, full request scope, slot or generation differs")
            active_slots.add(slot)
            active_requests.add(request)
            previous = owners.get(slot)
            if previous is not None and (generation < previous[0]
                    or (generation == previous[0] and identity != previous[1])):
                raise ValueError("native slot was reused with a stale generation")
            owners[slot] = generation, identity
            record = frame["completion_records"][slot]
            if len(record) < 17 or record[:2] != [generation, fence] \
                    or record[5] not in (1, 2) or record[6] != 0 \
                    or record[8] not in (0, 1) or record[2] != lane["start"] + lane["tokens"]:
                raise ValueError("native completion identity, status or consumed extent differs")
            step = len(traces[identity]) // (VOCABULARY * 2)
            if lane["history_step"] != step:
                raise ValueError("native history step differs from actual preceding emitted rows")
            handle = (identity, slot, generation)
            if handle not in consumed_by_handle:
                previous = handles_by_request[identity]
                initial = 0
                if previous:
                    eviction = planned[cohort][1]
                    if not history or eviction is None or len(previous) != 1 or step != 2 \
                            or consumed_by_handle[next(iter(previous))] != len(case["prompt_ids"]) + 1:
                        raise ValueError("native request changed generation outside its real G2 boundary")
                    initial = 0 if eviction else len(case["prompt_ids"]) + 1
                consumed_by_handle[handle] = initial
                previous.add(handle)
            if lane["start"] != consumed_by_handle[handle] or lane["tokens"] <= 0:
                raise ValueError("native prefill/decode consumed prefix has a gap or overlap")
            consumed_by_handle[handle] = record[2]
            if record[8]:
                if record[2] != len(case["prompt_ids"]) + step:
                    raise ValueError("native sampled row consumed a different actual history")
                value = data[row * VOCABULARY * 2:(row + 1) * VOCABULARY * 2]
                logits = torch.frombuffer(bytearray(value), dtype=torch.bfloat16)
                if not torch.isfinite(logits).all() or logits.argmax().item() != record[4]:
                    raise ValueError("native sampled token skipped the finite BF16 logit node")
                traces[identity].extend(value)
            elif frame["kind"] not in ("Prefill", "ScoredPrefill"):
                raise ValueError("non-prefill frame omitted its emitted row")
            if frame["kind"] == "Decode":
                witnesses[case["id"]].add((capacity, live, row, slot, generation))
            if frame["kind"] in ("Prefill", "ScoredPrefill") \
                    or (record[8] and step < len(case["teacher_history"])) or (history and step == 2):
                expected_kv[cohort, request, fence] = (frame, lane, record[2])

    if not history:
        for case in cases:
            seen = witnesses[case["id"]]
            for batch in (1, 2, 4, 8):
                if not any(b == batch and live == batch for b, live, _, _, _ in seen):
                    raise ValueError("one native input lacks a completed full logical batch")
            for batch in (2, 4, 8):
                if len({row for b, live, row, _, _ in seen if b == live == batch}) < 2:
                    raise ValueError("one native input lacks completed row permutations")
            if not any(b == 8 and live == 7 for b, live, _, _, _ in seen):
                raise ValueError("one native input lacks a completed partial batch")
            slots = defaultdict(set)
            for _, _, _, slot, generation in seen:
                slots[slot].add(generation)
            if len(slots) < 2 or not any(len(generations) > 1 for generations in slots.values()):
                raise ValueError("one native input lacks actual slot and generation reuse")

    baseline = {}
    observed_kv = set()
    for event in raw["kv_frames"]:
        key = event["cohort"], event["request"], event["fence"]
        if key in observed_kv or key not in expected_kv:
            raise ValueError("duplicate or unexpected native KV observation")
        observed_kv.add(key)
        frame, lane, consumed = expected_kv[key]
        order, _ = planned[key[0]]
        case = cases[order[key[1]]]
        expected = dict(case=case["id"], graph=frame["graph"], kind=frame["kind"],
            consumed=consumed, request_slot=lane["slot"], generation=lane["generation"],
            lane_start=lane["start"], lane_tokens=lane["tokens"], history_step=lane["history_step"],
            initial_prompt_capacity=frame["initial_prompt_capacity"], token_capacity=frame["token_capacity"],
            original_prompt_tokens=len(case["prompt_ids"]))
        if any(event.get(name) != value for name, value in expected.items()) \
                or event.get("actual_page_table_matches") is not True:
            raise ValueError("native KV extent, graph or request ownership differs")
        planes = event["planes"]
        if len(planes) != 56 or {plane["buffer"] for plane in planes} != PLANES:
            raise ValueError("native KV observation lacks complete distinct signed planes")
        values = {}
        for plane in planes:
            data = (source / plane["file"]).read_bytes()
            if plane["nbytes"] != len(data) or len(data) != consumed * 2048 \
                    or not torch.isfinite(torch.frombuffer(bytearray(data), dtype=torch.bfloat16)).all():
                raise ValueError("native KV plane is incomplete or nonfinite")
            values[plane["buffer"]] = data
        if key[0] < len(cases):
            previous = baseline.get(case["id"])
            if previous is not None and any(not values[name].startswith(before)
                    for name, before in previous.items()):
                raise ValueError("native B1 updated an earlier consumed KV prefix")
            baseline[case["id"]] = values
        else:
            previous = baseline[case["id"]]
            if any(value != previous[name][:len(value)] for name, value in values.items()):
                raise ValueError("native KV changed across batch, slot or generated-history replay")
    if observed_kv != expected_kv.keys():
        raise ValueError("missing native KV event, including a zero-emitted prefill fragment")
    if history:
        check_parking(source, raw, cases, planned, expected_kv, baseline, handles_by_request)
    return {key: bytes(value) for key, value in traces.items()}, witnesses


def check_parking(source, raw, cases, planned, expected_kv, baseline, handles_by_request):
    """Reconcile real snapshot, readmission and device poison with completed KV."""
    events = raw["parking"]
    for cohort, (order, eviction) in enumerate(planned):
        if eviction is None:
            continue
        case = cases[order[0]]
        selected = [event for event in events if event["cohort"] == cohort]
        snapshots = [event for event in selected if event["event"] == "actual_snapshot_completed"]
        residency = [event for event in selected if event["event"] == "checkpoint_residency_verified"]
        poison = [event for event in selected if event["event"] == "device_KV_poison"]
        if len(snapshots) != 1 or len(residency) != 1 or not poison \
                or len(handles_by_request[cohort, 0]) != 2:
            raise ValueError("native G2 lacks its actual snapshot, residency or readmitted generation")
        snapshot, resident = snapshots[0], residency[0]
        record = snapshot["completion_record"]
        consumed = len(case["prompt_ids"]) + 1
        if snapshot["consumed"] != consumed or snapshot["lane_start"] != consumed \
                or snapshot["lane_tokens"] != 0 or len(record) < 17 \
                or record[:3] != [snapshot["generation"], snapshot["fence"], consumed] \
                or record[5] not in (1, 2) or record[6] != 0 or record[8] != 0 \
                or resident["evicted"] is not eviction or resident["resident_before"] != consumed \
                or resident["resident_after"] != (0 if eviction else consumed) \
                or type(resident["pressure_reservations"]) is not int \
                or (resident["pressure_reservations"] > 0) is not eviction:
            raise ValueError("native G2 snapshot completion or checkpoint residency differs")
        old_handle = ((cohort, 0), snapshot["request_slot"], snapshot["generation"])
        if old_handle not in handles_by_request[cohort, 0]:
            raise ValueError("native G2 snapshot refers to an unobserved request generation")
        old_fences, new_fences = [], []
        for frame in raw["frames"]:
            if frame["cohort"] != cohort:
                continue
            for lane in frame["lanes"]:
                handle = ((cohort, lane["request"]), lane["slot"], lane["generation"])
                (old_fences if handle == old_handle else new_fences).append(frame["fence"])
        if type(snapshot["fence"]) is not int or not old_fences or not new_fences \
                or not max(old_fences) < snapshot["fence"] < min(new_fences) \
                or snapshot["fence"] in {frame["fence"] for frame in raw["frames"]}:
            raise ValueError("native G2 snapshot fence is outside its actual generation boundary")
        needed = {key for key, (frame, lane, _) in expected_kv.items()
            if key[0] == cohort and frame["kind"] == "Prefill"
            and (((cohort, 0), lane["slot"], lane["generation"]) != old_handle)}
        observed = set()
        for event in poison:
            key = cohort, 0, event["fence"]
            if key in observed or key not in needed:
                raise ValueError("native G2 poison lacks its completed history-prefill submission")
            observed.add(key)
            frame, lane, end = expected_kv[key]
            start = lane["start"]
            if event["start"] != start or event["end"] != end \
                    or event["slot"] != lane["slot"] or event["generation"] != lane["generation"] \
                    or event["planes"] != 56 or event["fence"] <= snapshot["fence"] \
                    or any(event.get(name) is not True for name in ("submitted", "readback_verified",
                        "earlier_prefix_unchanged", "same_comparator_rejected_poison")):
                raise ValueError("native G2 device poison metadata differs from actual completion")
            for name in PLANES:
                data = (source / f"cohort{cohort}-start{start}-{name}.poison").read_bytes()
                if len(data) != end * 2048 or data[:start * 2048] != baseline[case["id"]][name][:start * 2048] \
                        or data[start * 2048:] != b"\xc1\x7f" * ((end - start) * 1024):
                    raise ValueError("native G2 poison readback is missing or changed committed prefix")
        if observed != needed:
            raise ValueError("native G2 omitted poison for a completed history-prefill fragment")
