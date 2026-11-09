# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile raw ASR client phases with actual vLLM terminal observations.

This is an offline measurement check, not engine, quality, cache-reset,
instrumentation-overhead or statistical qualification. The run launcher must
establish those properties separately. Failed warmup and timed rows remain.
"""

import math

import httpx

from qualify.asr_vllm_performance_entry import validate_cohort
from qualify.asr_vllm_worker_observations import validate_encoder_cohort
from qualify.asr_workload import response_value, summarize


def counter_delta(before, after):
    """Keep missing and negative counts visible instead of Counter subtraction."""
    result = {}
    for key in before.keys() | after.keys():
        first, last = before.get(key, 0), after.get(key, 0)
        if type(first) is not int or type(last) is not int or first < 0 or last < first:
            raise ValueError("observation counter is invalid, reset or decreased")
        if last != first:
            result[key] = last - first
    return result


def expected_chunks(prefix, counts):
    if not counts or any(type(count) is not int or count < 1 for count in counts):
        raise ValueError("actual pinned processor chunk counts required")
    return [[f"transcribe-{prefix}-{index}"] if count == 1 else
        [f"transcribe-{prefix}-{index}-{chunk}" for chunk in range(count)]
        for index, count in enumerate(counts)]


def check_client_phase(client, expected, *, suite, timed):
    if client.get("engine") != "vllm" or client.get("status") != "cohort_drained" \
            or client.get("errors") != [] or client.get("diagnostic_limit") is not None \
            or client.get("suite") != suite or suite not in ("short", "long", "mixed"):
        raise ValueError("complete unabridged vLLM client phase required")
    rows = client["rows"]
    if len(rows) != len(expected):
        raise ValueError("client phase differs from its complete frozen request list")
    for index, (row, planned) in enumerate(zip(rows, expected, strict=True)):
        fields = ("id", "language", "samples", "pcm_s16le_sha256", "group")
        if row["ordinal"] != index or any(row[name] != planned[name] for name in fields) \
                or row["status"] != "response_received_unverified_termination" \
                or row["http_status"] != 200 or not row.get("text", "").strip() \
                or any(type(row[name]) is not int for name in ("sent_ns", "received_ns", "finished_ns")) \
                or not 0 <= row["sent_ns"] <= row["received_ns"] <= row["finished_ns"]:
            raise ValueError("failed, missing, reordered or malformed client request")
        if suite != "mixed" and (row["group"] != suite or "scheduled_ns" in row):
            raise ValueError("closed-loop request group or arrival semantics differ")
        if suite == "mixed":
            origin, offset = client["origin_ns"], planned["scheduled_seconds"]
            if type(origin) is not int or origin < 0 or type(offset) is not int or offset < 0 \
                    or type(row.get("scheduled_ns")) is not int \
                    or row["scheduled_ns"] != origin + offset * 10**9 \
                    or row["scheduled_ns"] > row["sent_ns"]:
                raise ValueError("mixed latency omitted its frozen arrival or client waiting")
        response = httpx.Response(row["http_status"], headers=row["response_headers"],
            content=row["response_body"].encode("utf8"))
        if response_value(response, "vllm", row["samples"])["text"] != row["text"]:
            raise ValueError("stored transcription differs from the actual HTTP body")
    summary = summarize(rows, engine="vllm", suite=suite, diagnostic=False)
    if summary["elapsed_seconds"] is None or summary["elapsed_seconds"] <= 0:
        raise ValueError("nonempty positive client measurement interval required")
    if timed and not summary["duration_count_client_criteria_passed"]:
        raise ValueError("timed phase misses frozen complete-count or duration criteria")
    return summary


def active_seconds(rows):
    intervals = sorted((row["sent_ns"], row["received_ns"]) for row in rows)
    total, end = 0, 0
    for start, stop in intervals:
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total / 1e9


def reconcile_vllm(timed, warmups, expected_timed, expected_warmup, chunk_counts,
                   warmup_chunk_counts, observer, encoder_before, encoder_after,
                   official_deltas, *, cache_mode, suite):
    """Credit timed audio only after all warmup and timed chunks terminate.

    Expected rows, chunk counts and suite come from the externally frozen plan,
    not from the client's labels or successful subset.
    """
    result = dict(scope=__doc__, errors=[], measurement_reconciled=False,
        planned_parent_requests=len(expected_timed), planned_warmup_passes=len(warmups),
        performance_qualified=False, actual_generated_tokens=None,
        complete_audio_seconds=0., complete_audio_seconds_per_elapsed=None)
    try:
        if cache_mode not in ("disabled", "allowed") or not warmups \
                or timed["regime"] not in ("unique-content", "repeated-content", "reasonable-deployment"):
            raise ValueError("explicit cache condition and actual warmup phases required")
        if (timed["regime"] == "unique-content") != (cache_mode == "disabled"):
            raise ValueError("declared timed cache regime differs from its reconciliation policy")
        if observer["status"] != "server_returned":
            raise ValueError("observer did not finish a normal serving shutdown")
        timed_summary = check_client_phase(timed, expected_timed, suite=suite, timed=True)
        for warmup in warmups:
            check_client_phase(warmup, expected_warmup, suite="short", timed=False)
        for first, last in zip(warmups, [*warmups[1:], timed], strict=True):
            start_field = "scheduled_ns" if last is timed and suite == "mixed" else "sent_ns"
            if max(row["finished_ns"] for row in first["rows"]) > min(row[start_field] for row in last["rows"]):
                raise ValueError("warmup phases overlap or extend into the timed phase")
        # The first full development pass precedes at least120 more active
        # seconds. Idle time between phases cannot satisfy this requirement.
        additional = sum(active_seconds(warmup["rows"]) for warmup in warmups[1:])
        if len(warmups) < 2 or additional < 120:
            raise ValueError("warmup lacks a full development pass plus120 active seconds")
        phases = [*warmups, timed]
        prefixes = [phase["request_prefix"] for phase in phases]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("warmup and timed request identities overlap")
        for counts, expected in [(chunk_counts, expected_timed), (warmup_chunk_counts, expected_warmup)]:
            if len(counts) != len(expected):
                raise ValueError("processor chunk plan differs from frozen parent set")
        parent_chunks = expected_chunks(timed["request_prefix"], chunk_counts)
        timed_ids = [identity for group in parent_chunks for identity in group]
        all_ids = [identity for warmup in warmups for group in
            expected_chunks(warmup["request_prefix"], warmup_chunk_counts) for identity in group] + timed_ids
        observations = observer["observations"]
        # Repeated warmup PCM is intentional. First reconcile the entire union
        # without asserting unique content, then check cold conditions on T.
        errors = validate_cohort(observations, all_ids, cache_mode="allowed")
        if errors:
            raise ValueError("warmup/timed terminal reconciliation failed: " + ",".join(errors))
        timed_set = set(timed_ids)
        selected = {row[0]: row for row in observations["rows"] if row[0] in timed_set}
        features = observations["mm_features"]
        values = [features[identity][0] for identity in timed_ids]
        if cache_mode == "disabled" and (any(selected[identity][6] != 0
                or type(selected[identity][6]) is not int for identity in timed_ids)
                or len({value[1] for value in values}) != len(timed_ids)
                or len({value[2] for value in values}) != len(timed_ids)):
            raise ValueError("timed cold KV or multimodal identities show reuse or unknown state")
        if encoder_before["errors"] or encoder_before["pending"] or encoder_after["errors"] \
                or encoder_after["pending"] or encoder_before["prepared"] != encoder_before["executed"] \
                or encoder_after["prepared"] != encoder_after["executed"]:
            raise ValueError("encoder observation failed or did not drain at phase boundaries")
        delta = dict(errors={}, pending=[],
            prepared=counter_delta(encoder_before["prepared"], encoder_after["prepared"]),
            executed=counter_delta(encoder_before["executed"], encoder_after["executed"]))
        errors = validate_encoder_cohort(delta, [value[1] for value in values], cache_mode=cache_mode)
        if errors:
            raise ValueError("timed encoder execution reconciliation failed: " + ",".join(errors))
        tokens = sum(len(selected[identity][2]) for identity in timed_ids)
        required = dict(stop=len(timed_ids), generated_tokens=tokens, length=0, abort=0, error=0, repetition=0)
        if any(type(official_deltas.get(name)) not in (int, float)
                or not math.isfinite(official_deltas[name]) or official_deltas[name] != value
                for name, value in required.items()):
            raise ValueError("actual timed chunk/token counts differ from official metric deltas")
        seconds = sum(row["samples"] for row in timed["rows"]) / 16000
        result.update(measurement_reconciled=True, timed_parent_requests=len(parent_chunks),
            timed_chunks=len(timed_ids), warmup_passes=len(warmups),
            additional_active_warmup_seconds=additional, actual_generated_tokens=tokens,
            complete_audio_seconds=seconds,
            complete_audio_seconds_per_elapsed=seconds / timed_summary["elapsed_seconds"],
            elapsed_seconds=timed_summary["elapsed_seconds"], latency=timed_summary["latency"],
            short_latency=timed_summary["short_latency"], long_latency=timed_summary["long_latency"],
            encoder_dispatch_deltas=delta)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, AttributeError) as error:
        result["errors"].append(dict(type=type(error).__name__, reason=str(error)))
    return result
