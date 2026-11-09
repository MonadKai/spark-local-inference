# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU raw-measurement corruption controls; no server or timing acceptance."""

from copy import deepcopy
import json

import pytest

from qualify.asr_performance_run import counter_delta, expected_chunks, reconcile_vllm


def measurements():
    def expected(count, prefix):
        return [dict(id=f"{prefix}{i}", language="en", samples=16000,
            pcm_s16le_sha256=f"{prefix}-pcm-{i}", group="short") for i in range(count)]

    def client(prefix, planned, seconds, origin):
        rows = []
        for i, item in enumerate(planned):
            text = "utterance " + str(i)
            rows.append(dict(**item, ordinal=i, status="response_received_unverified_termination",
                sent_ns=origin + i * seconds * 10**9, received_ns=origin + (i + 1) * seconds * 10**9,
                finished_ns=origin + (i + 1) * seconds * 10**9, http_status=200, text=text,
                response_headers={"content-type": "application/json"},
                response_body=json.dumps(dict(text=text, usage=dict(type="duration", seconds=1)))))
        return dict(engine="vllm", status="cohort_drained", errors=[], diagnostic_limit=None,
            suite="short", regime="unique-content", request_prefix=prefix, origin_ns=origin, rows=rows)

    planned, development = expected(100, "timed"), expected(2, "development")
    warmups = [client("warm0", development, 1, 0), client("warm1", development, 60, 10**12)]
    timed = client("timed", planned, 3, 2 * 10**12)
    chunk_counts, warm_counts = [2] + [1] * 99, [1, 2]
    observations = dict(errors={}, failures={}, active=[], started=0, terminal=0, closed=0,
        rows=[], mm_features={})
    before = dict(errors={}, pending=[], prepared={}, executed={})
    after = deepcopy(before)
    for phase in [*warmups, timed]:
        is_timed = phase is timed
        counts = chunk_counts if is_timed else warm_counts
        for parent, group in enumerate(expected_chunks(phase["request_prefix"], counts)):
            for child, identity in enumerate(group):
                observations["rows"].append([identity, 0, [7, 151645], "stop", None, identity, 0])
                feature = f"{'timed' if is_timed else 'warm'}-{parent}-{child}"
                observations["mm_features"][identity] = [["audio", feature, "hash-" + feature]]
                for boundary in [after] + ([] if is_timed else [before]):
                    for name in ("prepared", "executed"):
                        boundary[name][feature] = boundary[name].get(feature, 0) + 1
    for name in ("started", "terminal", "closed"):
        observations[name] = len(observations["rows"])
    observer = dict(status="server_returned", observations=observations)
    official = dict(stop=101, generated_tokens=202, length=0, abort=0, error=0, repetition=0)
    return [timed, warmups, planned, development, chunk_counts, warm_counts, observer, before, after, official]


def test_terminal_union_and_timed_encoder_delta_credit_parents_and_actual_tokens():
    values = measurements()
    result = reconcile_vllm(*values, cache_mode="disabled", suite="short")
    assert result["measurement_reconciled"] and not result["errors"]
    assert result["timed_parent_requests"] == 100 and result["timed_chunks"] == 101
    assert result["actual_generated_tokens"] == 202 and result["complete_audio_seconds"] == 100
    assert result["elapsed_seconds"] == 300 and result["additional_active_warmup_seconds"] == 120
    assert result["complete_audio_seconds_per_elapsed"] == pytest.approx(1 / 3)
    assert not result["performance_qualified"]


@pytest.mark.parametrize("defect", ["warmup-failure", "timed-failure", "missing-child", "unknown-child",
    "warmup-identity", "warmup-time", "short-run", "partial-corpus", "truncation", "counter-token",
    "counter-nonfinite", "encoder-reset", "encoder-missing", "encoder-pending", "warmup-encoder-incomplete",
    "cold-kv", "cold-mm", "body", "drain", "mode", "warmup-idle", "warmup-overlap",
    "warmup-after-timed", "short-labelled-long", "token-nan", "token-out-of-vocab", "float-eos"])
def test_failed_or_unknown_material_cannot_credit_complete_audio(defect):
    values = measurements()
    timed, warmups, planned, development, counts, warm_counts, observer, before, after, official = values
    raw = observer["observations"]
    if defect == "warmup-failure": warmups[0]["rows"][0]["status"] = "failed"
    if defect == "timed-failure": timed["rows"][0]["status"] = "failed"
    if defect == "missing-child": raw["rows"].pop()
    if defect == "unknown-child": raw["rows"][-1][0] = "unknown"
    if defect == "warmup-identity": warmups[1]["request_prefix"] = warmups[0]["request_prefix"]
    if defect == "warmup-time": warmups.pop()
    if defect == "short-run":
        for row in timed["rows"]:
            for name in ("sent_ns", "received_ns", "finished_ns"): row[name] //= 2
    if defect == "partial-corpus": timed["rows"].pop()
    if defect == "truncation": raw["rows"][-1][3] = "length"
    if defect == "counter-token": official["generated_tokens"] -= 1
    if defect == "counter-nonfinite": official["generated_tokens"] = float("nan")
    if defect == "encoder-reset": after["prepared"]["warm-0-0"] = after["executed"]["warm-0-0"] = 0
    if defect == "encoder-missing":
        after["prepared"].pop("timed-0-0"); after["executed"].pop("timed-0-0")
    if defect == "encoder-pending": after["pending"] = [["timed-0-0"]]
    if defect == "warmup-encoder-incomplete": before["executed"]["warm-0-0"] -= 1
    if defect == "cold-kv": raw["rows"][-1][6] = 1
    if defect == "cold-mm": raw["mm_features"][raw["rows"][-1][0]] = raw["mm_features"][raw["rows"][-2][0]]
    if defect == "body": timed["rows"][0]["response_body"] = "{}"
    if defect == "drain": observer["status"] = "running"
    if defect == "mode": timed["regime"] = "repeated-content"
    if defect == "warmup-idle":
        first, last = warmups[1]["rows"]
        first["received_ns"] = first["finished_ns"] = first["sent_ns"] + 10**9
        last["sent_ns"] = last["received_ns"] - 10**9
    if defect in ("warmup-overlap", "warmup-after-timed"):
        shift = -10**12 if defect == "warmup-overlap" else 3 * 10**12
        for row in warmups[1]["rows"]:
            for name in ("sent_ns", "received_ns", "finished_ns"): row[name] += shift
    if defect == "short-labelled-long":
        timed["suite"] = "long"
        for row in timed["rows"]:
            for name in ("sent_ns", "received_ns", "finished_ns"): row[name] //= 2
    if defect == "token-nan": raw["rows"][-1][2][0] = float("nan")
    if defect == "token-out-of-vocab": raw["rows"][-1][2][0] = 151936
    if defect == "float-eos": raw["rows"][-1][2][-1] = 151645.
    result = reconcile_vllm(*values, cache_mode="disabled", suite="short")
    assert result["errors"] and not result["measurement_reconciled"]
    assert result["planned_parent_requests"] == 100
    assert result["complete_audio_seconds"] == 0 and result["actual_generated_tokens"] is None
    assert not result["performance_qualified"]


def test_signed_union_counter_delta_cannot_hide_reset_or_removed_keys():
    assert counter_delta({"warm": 2}, {"warm": 2, "timed": 3}) == {"timed": 3}
    for after in ({}, {"warm": 1}):
        with pytest.raises(ValueError, match="decreased"):
            counter_delta({"warm": 2}, after)


@pytest.mark.parametrize("defect", [None, "missing", "hidden-wait", "wrong-origin", "warmup-overlaps-origin"])
def test_mixed_arrivals_match_frozen_offsets_and_retain_client_wait(defect):
    values = measurements()
    timed, _, planned = values[:3]
    timed["suite"] = "mixed"
    for i, (row, expected) in enumerate(zip(timed["rows"], planned, strict=True)):
        expected["scheduled_seconds"] = 3 * i
        row["scheduled_ns"] = timed["origin_ns"] + 3 * i * 10**9
    row = timed["rows"][5]
    row["sent_ns"] += 5 * 10**9
    row["received_ns"] += 5 * 10**9
    row["finished_ns"] += 5 * 10**9
    if defect == "missing": del row["scheduled_ns"]
    if defect == "hidden-wait": row["scheduled_ns"] = row["sent_ns"]
    if defect == "wrong-origin": timed["origin_ns"] += 1
    if defect == "warmup-overlaps-origin":
        timed["origin_ns"] = max(row["finished_ns"] for row in values[1][-1]["rows"]) - 10**9
        for row, expected in zip(timed["rows"], planned, strict=True):
            row["scheduled_ns"] = timed["origin_ns"] + expected["scheduled_seconds"] * 10**9
    result = reconcile_vllm(*values, cache_mode="disabled", suite="mixed")
    if defect:
        assert result["errors"] and not result["measurement_reconciled"]
    else:
        assert not result["errors"] and result["measurement_reconciled"]
        assert result["short_latency"]["samples"][5] == 8
