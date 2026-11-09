# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Collect frozen complete ASR phases from an exported package served by Rust.

The launcher owns package/binary/config identity, isolation, resource sampling
and normal shutdown. This client reconciles real HTTP token/stop headers with
drained native counters and actual CPU frontend plans. It does not qualify
model quality, stage timings, instrumentation cost or comparative performance.
"""

import argparse
import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import wave

import httpx

from qualify.asr_performance_run import active_seconds, check_client_phase, collect_client_phase, counter_delta
from qualify.asr_workload import ROOT, expected_rows, load_pcm, plan


COUNTERS = ("completed", "failed", "generated_tokens", "prefill_tokens", "replay_gpu_ns",
    "reused_tokens", "host_hits", "host_reused_tokens", "host_bytes_out", "host_bytes_in",
    "encoded_input_hits", "encoded_input_writes", "encoded_input_skipped", "encoded_input_upload_bytes",
    "encoded_input_readback_bytes", "encoded_input_upload_wall_ns", "encoded_input_retention_wall_ns",
    "input_cache_metadata_hits_total", "input_cache_payload_hits_total", "input_cache_memory_hits_total",
    "input_cache_disk_hits_total", "input_cache_failures_total")
GAUGES = ("active", "pending", "cached_pages", "free_pages", "total_pages", "page_tokens",
    "host_memory_records", "host_disk_records", "input_cache_memory_records", "input_cache_disk_records",
    "input_cache_memory_bytes", "input_cache_disk_bytes", "input_cache_queued", "input_cache_pending_writes",
    "input_cache_staging_bytes", "input_decoded_source_bytes", "input_cache_memory_limit")
DRAIN_GAUGES = ("http_inflight_requests", "frontend_active_workers", "retiring_request_slots",
    "active_requests", "pending_requests", "host_cache_outbound_in_flight", "host_cache_parked_records",
    "host_cache_inbound_in_flight", "host_cache_inbound_waiting", "input_cache_queued",
    "input_cache_pending_writes", "input_cache_staging_bytes", "input_decoded_source_bytes")
KINDS = ("init", "restore", "resume", "prefill", "decode", "vision", "snapshot", "speculative", "mixed", "audio_encode")
COLD_ZERO = ("cached_pages", "host_memory_records", "host_disk_records", "input_cache_memory_records",
    "input_cache_disk_records", "input_cache_memory_bytes", "input_cache_disk_bytes", "input_cache_memory_limit")
COLD_HITS = ("host_hits", "host_reused_tokens", "encoded_input_hits",
    "input_cache_metadata_hits_total", "input_cache_payload_hits_total", "input_cache_memory_hits_total",
    "input_cache_disk_hits_total")
CONTROL_COUNTERS = ("requests_admitted_total", "requests_cancelled_total", "preemptions_total")


def process_identity(pid):
    root = Path("/proc") / str(pid)
    stat = (root / "stat").read_text().rpartition(")")[2].split()
    executable = (root / "exe").stat()
    return dict(pid=pid, start_ticks=int(stat[19]), executable_device=executable.st_dev,
        executable_inode=executable.st_ino, argv=os.fsdecode((root / "cmdline").read_bytes()).rstrip("\0").split("\0"))


def check_policy(identity, *, cold):
    argv = identity["argv"]
    if len(argv) < 3 or argv[1] != "run":
        raise ValueError("actual server must run an exported package")
    required = {"--embedding-table-storage": "ram"}
    if cold:
        required.update({"--prefix-cache-memory": "0", "--prefix-cache-disk-limit": "0",
            "--prefix-cache-reuse": "request", "--prepared-input-cache": "off"})
    for name, value in required.items():
        at = [i for i, item in enumerate(argv) if item == name or item.startswith(name + "=")]
        if len(at) != 1 or argv[at[0]] != name or at[0] + 1 == len(argv) or argv[at[0] + 1] != value:
            raise ValueError("live native argv lacks the explicit frozen policy: " + name)


def observable_counters(text):
    """Retain all exposed cumulative series, including histogram components."""
    types = {}
    for line in text.splitlines():
        if line.startswith("# TYPE flare_"):
            match = re.fullmatch(r"# TYPE (flare_[a-zA-Z0-9_]+) (counter|gauge|histogram)", line)
            if match is None or match[1] in types:
                raise ValueError("invalid or duplicated native metric type")
            types[match[1]] = match[2]
    required = {"flare_" + name for name in CONTROL_COUNTERS}
    if not required.issubset(types) or any(types[name] != "counter" for name in required):
        raise ValueError("native admission, cancellation and preemption counters required")
    result, observed, histograms = {}, set(), {}
    for line in text.splitlines():
        if not line.startswith("flare_"):
            continue
        name = re.split(r"[ {\t]", line, maxsplit=1)[0]
        owner = name if types.get(name) == "counter" else next((name.removesuffix(suffix) for suffix in
            ("_bucket", "_sum", "_count") if name.endswith(suffix) and types.get(name.removesuffix(suffix)) == "histogram"), None)
        if owner is None:
            continue
        match = re.fullmatch(re.escape(name) + r'(?:\{([^}]*)\})? ([0-9]+)', line)
        if match is None:
            raise ValueError("invalid cumulative native metric series")
        labels = []
        for field in match[1].split(",") if match[1] else []:
            label = re.fullmatch(r'([a-zA-Z_][a-zA-Z_0-9]*)="([^"\\]*)"', field)
            if label is None or any(key == label[1] for key, _ in labels):
                raise ValueError("invalid cumulative native metric labels")
            labels.append((label[1], label[2]))
        key = name + json.dumps(sorted(labels), separators=(",", ":"))
        if key in result:
            raise ValueError("duplicate cumulative native metric series")
        result[key] = integer(int(match[2]))
        observed.add(owner)
        if types[owner] == "histogram":
            histograms.setdefault(owner, {})[(name.removeprefix(owner), tuple(sorted(labels)))] = result[key]
    expected = {name for name, kind in types.items() if kind == "counter" and name != "flare_graph_batch_submissions_total"}
    if not expected.issubset(observed):
        raise ValueError("declared native counter has no observed series")
    histogram_series = []
    for name, kind in types.items():
        if kind != "histogram":
            continue
        components = histograms.get(name, {})
        if ("_count", ()) not in components or ("_sum", ()) not in components \
                or ("_bucket", (("le", "+Inf"),)) not in components:
            raise ValueError("native histogram lacks count, sum or terminal bucket")
        buckets = []
        for (suffix, labels), count in components.items():
            if suffix in ("_count", "_sum") and not labels:
                continue
            if suffix != "_bucket" or len(labels) != 1 or labels[0][0] != "le" \
                    or labels[0][1] != "+Inf" and not labels[0][1].isdecimal():
                raise ValueError("native histogram component has invalid geometry")
            bound = 2**64 if labels[0][1] == "+Inf" else integer(int(labels[0][1]))
            buckets.append((bound, count))
        buckets.sort()
        if any(first[1] > last[1] for first, last in zip(buckets, buckets[1:])) \
                or buckets[-1][1] != components[("_count", ())]:
            raise ValueError("native histogram buckets disagree with its count")
        histogram_series.extend(key for key in result if key.startswith(name + "_"))
    return result, types, sorted(histogram_series)


def integer(value):
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError("missing or invalid native unsigned counter")
    return value


def prometheus(text):
    scalars, launches, batches = {}, {}, {}
    required = {*DRAIN_GAUGES, *CONTROL_COUNTERS, "graph_batch_catalog_entries", "uptime_seconds"}
    for line in text.splitlines():
        if line.startswith("flare_graph_launches_total{"):
            match = re.fullmatch(r'flare_graph_launches_total\{kind="([a-z_]+)"\} ([0-9]+)', line)
            if match is None or match[1] not in KINDS or match[1] in launches:
                raise ValueError("malformed or duplicate graph launch series")
            launches[match[1]] = integer(int(match[2]))
        elif line.startswith("flare_graph_batch_submissions_total{"):
            match = re.fullmatch(r'flare_graph_batch_submissions_total\{graph_index="([0-9]+)",capacity="([0-9]+)",requests="([0-9]+)"\} ([0-9]+)', line)
            if match is None:
                raise ValueError("malformed graph batch series")
            index, capacity, requests, count = map(int, match.groups())
            key = (index, capacity, requests)
            if key in batches or capacity < 1 or not 0 <= requests <= capacity:
                raise ValueError("duplicate or invalid graph batch geometry")
            batches[key] = integer(count)
        elif line.startswith("flare_"):
            name = re.split(r"[ {\t]", line.removeprefix("flare_"), maxsplit=1)[0]
            if name in required:
                match = re.fullmatch("flare_" + re.escape(name) + r" ([0-9]+)", line)
                if match is None or name in scalars:
                    raise ValueError("malformed or duplicated native gauge")
                scalars[name] = integer(int(match[1]))
    if set(scalars) != required or set(launches) != set(KINDS) or scalars["graph_batch_catalog_entries"] < 1:
        raise ValueError("required native metrics are absent")
    capacities = {}
    for index, capacity, _ in batches:
        if index >= scalars["graph_batch_catalog_entries"] or capacities.setdefault(index, capacity) != capacity:
            raise ValueError("batch series differs from the graph catalog")
    counters, types, histograms = observable_counters(text)
    return dict(gauges=scalars, launches=launches, batches=batches, observable_counters=counters,
        types=types, histogram_series=histograms)


def snapshot(value):
    if not isinstance(value, dict):
        raise ValueError("native JSON snapshot is not an object")
    result = {key: integer(value.get(key)) for key in (*COUNTERS, *GAUGES)}
    if result["total_pages"] < 1 or result["page_tokens"] < 1 \
            or result["cached_pages"] > result["total_pages"] or result["free_pages"] > result["total_pages"]:
        raise ValueError("invalid native page geometry")
    return result


def idle(observation, *, cold):
    first, last, prom = observation["first"], observation["last"], observation["prometheus"]
    if first != last or any(prom["gauges"][key] for key in DRAIN_GAUGES) or last["active"] or last["pending"]:
        return False
    if any(prom["gauges"][p] != last[j] for p, j in [("active_requests", "active"), ("pending_requests", "pending"),
            *[(key, key) for key in ("input_cache_queued", "input_cache_pending_writes", "input_cache_staging_bytes", "input_decoded_source_bytes")]]):
        return False
    return not cold or not any(last[key] for key in COLD_ZERO) and last["free_pages"] == last["total_pages"]


def deltas(before, after):
    if any(before["last"][key] != after["last"][key] for key in ("total_pages", "page_tokens")) \
            or before["prometheus"]["gauges"]["graph_batch_catalog_entries"] != after["prometheus"]["gauges"]["graph_batch_catalog_entries"] \
            or before["prometheus"]["types"] != after["prometheus"]["types"] \
            or before["prometheus"]["histogram_series"] != after["prometheus"]["histogram_series"] \
            or after["prometheus"]["gauges"]["uptime_seconds"] < before["prometheus"]["gauges"]["uptime_seconds"]:
        raise ValueError("native engine geometry changed or uptime reset")
    values = {key: after["last"][key] - before["last"][key] for key in COUNTERS}
    if any(value < 0 for value in values.values()):
        raise ValueError("native counter decreased or reset")
    launches = {key: after["prometheus"]["launches"][key] - before["prometheus"]["launches"][key] for key in KINDS}
    if any(value < 0 for value in launches.values()):
        raise ValueError("native launch counter decreased or reset")
    batches = counter_delta(before["prometheus"]["batches"], after["prometheus"]["batches"])
    return dict(counters=values, launches=launches,
        observable_counters=counter_delta(before["prometheus"]["observable_counters"], after["prometheus"]["observable_counters"]),
        graph_batches=[dict(graph_index=i, contract_capacity=b, requests=n, submissions=count)
            for (i, b, n), count in sorted(batches.items())])


def native_plan(content, expected, pcm, envelope):
    """Bind every real native plan row to the complete frozen PCM set."""
    import numpy as np

    rows = [json.loads(line) for line in content.splitlines()]
    if len(rows) != len(expected) + 2:
        raise ValueError("native plan is truncated or differs from the complete input set")
    header, *cases, final = rows
    if header.get("kind") != "header" or header.get("schema") != 1 or header.get("envelope_sha256") != envelope \
            or header.get("tensor_parallel") != 1 or header.get("inputs") != len(expected) \
            or final != dict(kind="summary", inputs=len(expected), failed=0):
        raise ValueError("native plan identity or terminal summary differs")
    settings, capacity = header["settings"], header["capacity"]
    if settings["sample_rate"] != 16000 or settings["hop_samples"] != 160 or settings["sample_padding"] not in (1, 2):
        raise ValueError("native plan has unsupported sample geometry")
    result = {}
    for ordinal, (row, audio) in enumerate(zip(cases, expected, strict=True)):
        with wave.open(io.BytesIO(pcm[audio["pcm_s16le_sha256"]]), "rb") as source:
            decoded = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype("<f4") / np.float32(32768)
        identity = hashlib.sha256(decoded.tobytes()).hexdigest()
        if row.get("kind") != "input" or row.get("status") != "passed" or row.get("ordinal") != ordinal \
                or row.get("id") != audio["id"] or audio["id"] in result or row.get("samples") != audio["samples"] \
                or row.get("pcm_f32le_sha256") != identity or row.get("encoded_sha256") != hashlib.sha256(pcm[audio["pcm_s16le_sha256"]]).hexdigest():
            raise ValueError("native plan row differs from actual frozen audio")
        end, windows = 0, 0
        for chunk in row["chunks"]:
            begin, stop = chunk["samples"]
            if type(begin) is not int or type(stop) is not int or begin != end or not begin < stop <= audio["samples"]:
                raise ValueError("native service chunks omit or overlap samples")
            frames = (stop - begin + (159 if settings["sample_padding"] == 2 else 0)) // 160
            frame_end, feature_end = 0, 0
            for window in chunk["windows"]:
                a, b = window["frames"]
                c, d = window["features"]
                length = b - a
                count = length // capacity["frames_per_chunk"] * capacity["features_per_chunk"] \
                    + (length % capacity["frames_per_chunk"] + capacity["subsampling_factor"] - 1) // capacity["subsampling_factor"]
                if not all(type(value) is int for value in (a, b, c, d)) or a != frame_end or c != feature_end \
                        or b != min(a + capacity["attention_window_frames"], frames) or length < 1 or d - c != count:
                    raise ValueError("native encoder windows omit or overlap chunk features")
                frame_end, feature_end = b, d
                windows += 1
            if frame_end != frames or chunk["frames"] != frames or chunk["features"] != feature_end or feature_end < 1:
                raise ValueError("native chunk frame/feature tail differs")
            end = stop
        if end != audio["samples"]:
            raise ValueError("native service plan omits decoded audio")
        result[audio["id"]] = dict(chunks=len(row["chunks"]), windows=windows)
    return result


async def metrics(client, result, save, *, timeout, previous=None):
    observation = dict(status="pending", raw=[])
    result["metric_observations"].append(dict(observation))
    save()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    try:
        bodies = []
        for format in ("json", "prometheus", "json"):
            response = await asyncio.wait_for(client.get("/metrics", params=dict(format=format)),
                timeout=max(.001, deadline - loop.time()))
            observation["raw"].append(dict(format=format, status=response.status_code, body=response.text))
            save()
            response.raise_for_status()
            if response.status_code != 200 or response.headers.get("content-type", "").partition(";")[0] \
                    != ("application/json" if format == "json" else "text/plain"):
                raise ValueError("native metrics did not return HTTP200 with the requested format")
            bodies.append(response.text)
        observation.update(first=snapshot(json.loads(bodies[0])),
            prometheus=prometheus(bodies[1]), last=snapshot(json.loads(bodies[2])), status="observed")
        if any(observation["last"][key] < observation["first"][key] for key in COUNTERS):
            raise ValueError("native JSON counter decreased within one observation")
        if previous is not None:
            if any(observation["first"][key] < previous["last"][key] for key in COUNTERS):
                raise ValueError("native JSON counter decreased since its previous observation")
            deltas(previous, observation)
        # Tuple keys are kept only in local parsed views; raw bodies are the
        # durable observations, not a lossy reserialization of the metrics.
        result["metric_observations"][-1] = dict(status="observed", raw=observation["raw"])
        save()
        return observation
    except BaseException as error:
        result["metric_observations"][-1].update(status="failed_or_interrupted",
            error=dict(type=type(error).__name__, message=str(error)))
        save()
        raise


async def drain(client, result, save, before, *, chunks, tokens, windows, cold):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 30
    record = dict(expected_chunks=chunks, expected_tokens=tokens, expected_windows=windows, cold=cold, attempts=0)
    result["drain_observations"].append(record)
    previous = before
    for _ in range(128):
        if loop.time() >= deadline:
            break
        current = await metrics(client, result, save, timeout=deadline - loop.time(), previous=previous)
        previous = current
        record["attempts"] += 1
        delta = deltas(before, current)
        counts = delta["counters"]
        record["last_deltas"] = delta
        save()
        controls = delta["observable_counters"]
        cancelled = controls.get("flare_requests_cancelled_total[]", 0)
        preemptions = controls.get("flare_preemptions_total[]", 0)
        if counts["completed"] > chunks or counts["generated_tokens"] > tokens or counts["failed"] \
                or counts["input_cache_failures_total"] or cancelled:
            raise ValueError("native phase failed or contains unplanned work")
        if idle(current, cold=cold) and counts["completed"] == chunks and counts["generated_tokens"] == tokens:
            if cold and (any(counts[key] for key in COLD_HITS) or delta["launches"]["audio_encode"] < windows
                    or delta["launches"]["audio_encode"] > windows and not preemptions):
                raise ValueError("cold native phase contradicts disabled stores or the encoder/readmission plan")
            delta["cache_attribution"] = dict(cross_request_prefix="disabled by live explicit Request namespace policy" if cold else "allowed",
                host_and_prepared_retention="disabled by live explicit argv" if cold else "launcher configuration",
                reused_tokens_scope=None, reused_tokens_reason="aggregate includes legitimate same-request readmission",
                baseline_encoder_windows=windows, preemptions=preemptions,
                extra_encoder_windows=max(0, delta["launches"]["audio_encode"] - windows),
                extra_encoder_request_identity=None,
                extra_encoder_reason="aggregate graph counters cannot attribute replayed windows to individual readmissions")
            return current, delta
        await asyncio.sleep(.25)
    raise TimeoutError("native frontend, scheduler, transfers or KV did not drain")


async def collect(args, result, requests, development, pcm, prepared, save, *, verify_server):
    task, loop = asyncio.current_task(), asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    cold = args.regime == "unique-content"

    def geometry(selected, key):
        return sum(prepared[item["input_set"]][item["utterance"]["id"]][key] for item in selected)

    try:
        limit = args.concurrency + (9 if args.suite == "mixed" else 0)
        async with httpx.AsyncClient(base_url=args.base_url, timeout=1800,
                limits=httpx.Limits(max_connections=limit, max_keepalive_connections=limit),
                follow_redirects=False, trust_env=False) as client:
            async def phase(target, selected, suite):
                verify_server()
                await collect_client_phase(client, target, selected, pcm, engine="flare", suite=suite,
                    regime=args.regime, concurrency=args.concurrency, save=save)
                verify_server()

            initial = await metrics(client, result, save, timeout=30)
            verify_server()
            if not idle(initial, cold=cold):
                raise ValueError("native server is not drained in the declared initial cache condition")
            before, additional = initial, 0.
            for index in range(64):
                warmup = dict(request_prefix=args.request_prefix + "-warm" + str(index))
                result["warmups"].append(warmup)
                await phase(warmup, development, "short")
                before, delta = await drain(client, result, save, before, chunks=geometry(development, "chunks"),
                    windows=geometry(development, "windows"), tokens=warmup["summary"]["actual_generated_tokens"], cold=cold)
                warmup["native_deltas"] = delta
                if index:
                    additional += active_seconds(warmup["rows"])
                if index and additional >= 120:
                    break
            else:
                raise RuntimeError("64 complete development passes did not provide120 active warmup seconds")
            boundary = await metrics(client, result, save, timeout=30, previous=before)
            bridge = deltas(before, boundary)
            if not idle(boundary, cold=cold) or any(bridge["counters"].values()) \
                    or any(bridge["launches"].values()) or bridge["graph_batches"] or bridge["observable_counters"]:
                raise ValueError("native work appeared between warmup drain and timed admission")
            result["additional_active_warmup_seconds"] = additional
            result["status"] = "timed_phase"
            await phase(result["timed"], requests, args.suite)
            after, result["native_metric_deltas"] = await drain(client, result, save, boundary,
                chunks=geometry(requests, "chunks"), windows=geometry(requests, "windows"),
                tokens=result["timed"]["summary"]["actual_generated_tokens"], cold=cold)
            summary = result["timed_client_summary"] = check_client_phase(result["timed"], expected_rows(requests),
                engine="flare", suite=args.suite, timed=True)
            result["reconciled_throughput"] = dict(complete_audio_seconds_per_elapsed=summary["complete_audio_seconds_per_elapsed"],
                successful_requests_per_second=len(requests) / summary["elapsed_seconds"],
                actual_generated_tokens_per_second=summary["actual_generated_tokens"] / summary["elapsed_seconds"])
            result["status"] = "http_phases_reconciled_pending_launcher_qualification"
            result["native_graph_catalog_entries"] = after["prometheus"]["gauges"]["graph_batch_catalog_entries"]
            save()
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--suite", choices=("short", "long", "mixed"), required=True)
    parser.add_argument("--regime", choices=("unique-content", "repeated-content", "reasonable-deployment"), required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--envelope-sha256", required=True, help="exact authenticated served package envelope identity")
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--server-binary-sha256", required=True)
    for name in ("performance", "development", "long"):
        parser.add_argument("--" + name + "-pcm-root", type=Path, required=True)
        parser.add_argument("--" + name + "-native-plan", type=Path, required=True)
    parser.add_argument("--request-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", args.request_prefix) or args.output.exists() \
            or not re.fullmatch(r"[0-9a-f]{64}", args.envelope_sha256) or args.server_pid < 1 \
            or not re.fullmatch(r"[0-9a-f]{64}", args.server_binary_sha256):
        parser.error("unique output,1..80 ASCII request prefix and exact package envelope SHA256 required")
    requests = plan(args.suite, args.regime, args.concurrency)
    development = [dict(utterance=row, group="short", input_set="development", scheduled_seconds=None)
        for row in json.loads((ROOT / "development.json").read_bytes())["utterances"]]
    result = dict(scope=__doc__, status="loading_inputs", errors=[], warmups=[],
        metric_observations=[], drain_observations=[], native_plan_files={}, performance_qualified=False,
        envelope_sha256=args.envelope_sha256, timed=dict(request_prefix=args.request_prefix + "-timed"),
        unavailable={key: dict(value=None, reason=reason) for key, reason in dict(raw_token_ids="transcription endpoint does not expose recognition IDs",
            model_ttft="non-streaming HTTP responses do not expose model TTFT", queue_wait="not exposed per request",
            stage_gpu_times="not exposed by phase metrics", gpu_busy="sum of overlapping replay intervals is not GPU busy",
            physical_compute_padding="contract batch capacity does not describe internal prepared compute rows").items()},
        frozen_files={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ("performance-protocol.json", "performance-inputs.json", "development.json", "long-speech.json")})

    def save():
        pending = args.output.with_suffix(args.output.suffix + ".pending")
        pending.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        pending.replace(args.output)

    save()
    try:
        identity = process_identity(args.server_pid)
        with (Path("/proc") / str(args.server_pid) / "exe").open("rb") as executable:
            if hashlib.file_digest(executable, "sha256").hexdigest() != args.server_binary_sha256:
                raise ValueError("actual native server binary differs from the supplied identity")
        check_policy(identity, cold=args.regime == "unique-content")
        result["server_identity"] = identity
        result["server_binary_sha256"] = args.server_binary_sha256

        def verify_server():
            if process_identity(args.server_pid) != identity:
                raise ValueError("native server PID, executable or argv identity changed")

        pcm, prepared = {}, {}
        for name, manifest in [("performance", "performance-inputs.json"), ("development", "development.json"), ("long", "long-speech.json")]:
            audio = json.loads((ROOT / manifest).read_bytes())["utterances"]
            pcm.update(load_pcm(getattr(args, name + "_pcm_root"), audio))
            content = getattr(args, name + "_native_plan").read_bytes()
            result["native_plan_files"][name] = hashlib.sha256(content).hexdigest()
            prepared[name] = native_plan(content, audio, pcm, args.envelope_sha256)
        result["loaded_pcm_bytes"] = sum(map(len, pcm.values()))
        asyncio.run(collect(args, result, requests, development, pcm, prepared, save, verify_server=verify_server))
    except BaseException as error:
        result["status"] = "failed_or_interrupted"
        result["errors"].append(dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
