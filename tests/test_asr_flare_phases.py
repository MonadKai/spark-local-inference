# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU corruption and real HTTP controls; these fixtures execute no model."""

import asyncio
from copy import deepcopy
from email import policy
from email.parser import BytesParser
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from threading import Event, Lock, Thread
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from qualify import asr_flare_phases as phases


def metrics(completed=0, *, busy=0):
    value = dict.fromkeys((*phases.COUNTERS, *phases.GAUGES), 0)
    value.update(completed=completed, generated_tokens=completed, prefill_tokens=completed * 32,
        total_pages=256, free_pages=256, page_tokens=32)
    gauges = dict.fromkeys((*phases.DRAIN_GAUGES, *phases.CONTROL_COUNTERS, "graph_batch_catalog_entries", "uptime_seconds"), 0)
    gauges.update(graph_batch_catalog_entries=55, uptime_seconds=123, retiring_request_slots=busy,
        requests_admitted_total=completed)
    text = [f"flare_{key} {count}" for key, count in gauges.items()]
    text += [f"# TYPE flare_{key} counter" for key in phases.CONTROL_COUNTERS]
    text += ["# TYPE flare_graph_launches_total counter", "# TYPE flare_graph_batch_submissions_total counter",
        "# TYPE flare_input_cache_stored_total counter", "flare_input_cache_stored_total 0",
        "# TYPE flare_host_cost histogram", 'flare_host_cost_bucket{le="1000"} 0', 'flare_host_cost_bucket{le="+Inf"} 0',
        "flare_host_cost_count 0", "flare_host_cost_sum 0"]
    text += [f'flare_graph_launches_total{{kind="{kind}"}} {completed * 3 // 2 if kind == "audio_encode" else 0}' for kind in phases.KINDS]
    if completed:
        text += [f'flare_graph_batch_submissions_total{{graph_index="0",capacity="1",requests="1"}} {completed * 3 // 2}']
    return value, "\n".join(text)


def observation(completed=0, *, busy=0):
    value, text = metrics(completed, busy=busy)
    return dict(first=phases.snapshot(value), last=phases.snapshot(value), prometheus=phases.prometheus(text))


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nan", "negative", "fraction", "labels", "timestamp", "unknown-kind", "batch-capacity", "batch-index"])
def test_metric_corruption_is_not_missing_as_zero(defect):
    _, text = metrics(2)
    if defect == "missing": text = text.replace("flare_http_inflight_requests 0", "")
    if defect == "duplicate": text += "\nflare_http_inflight_requests 0"
    if defect in ("nan", "negative", "fraction"):
        text = text.replace("flare_http_inflight_requests 0", "flare_http_inflight_requests " + dict(nan="NaN", negative="-1", fraction=".5")[defect])
    if defect == "labels": text += '\nflare_http_inflight_requests{fake="1"} 0'
    if defect == "timestamp": text += "\nflare_http_inflight_requests 0 123"
    if defect == "unknown-kind": text = text.replace('kind="audio_encode"', 'kind="unknown"')
    if defect == "batch-capacity": text = text.replace('capacity="1"', 'capacity="0"')
    if defect == "batch-index": text = text.replace('graph_index="0"', 'graph_index="55"')
    with pytest.raises(ValueError): phases.prometheus(text)


@pytest.mark.parametrize("defect", ["missing", "bool", "nan", "negative", "fraction", "overflow"])
def test_json_snapshot_needs_actual_unsigned_fields(defect):
    value, _ = metrics()
    if defect == "missing": del value["completed"]
    else: value["completed"] = dict(bool=False, nan=float("nan"), negative=-1, fraction=.5, overflow=2**64)[defect]
    with pytest.raises(ValueError): phases.snapshot(value)


@pytest.mark.parametrize("defect", ["all", "count", "sum", "infinity", "inconsistent-count", "changed-buckets"])
def test_declared_histograms_need_complete_consistent_components(defect):
    _, text = metrics()
    if defect == "all": text = "\n".join(line for line in text.splitlines() if not line.startswith("flare_host_cost"))
    if defect == "count": text = text.replace("flare_host_cost_count 0", "")
    if defect == "sum": text = text.replace("flare_host_cost_sum 0", "")
    if defect == "infinity": text = text.replace('flare_host_cost_bucket{le="+Inf"} 0', "")
    if defect == "inconsistent-count": text = text.replace("flare_host_cost_count 0", "flare_host_cost_count 1")
    if defect == "changed-buckets":
        after = observation()
        after["prometheus"] = phases.prometheus(text.replace('le="1000"', 'le="2000"'))
        with pytest.raises(ValueError): phases.deltas(observation(), after)
    else:
        with pytest.raises(ValueError): phases.prometheus(text)


@pytest.mark.parametrize("defect", [None, "busy", "cache", "unreleased-kv", "unstable", "counter-reset", "uptime-reset", "batch-reset"])
def test_real_drain_conditions_and_counter_continuity(defect):
    before, after = observation(2), observation(4)
    if defect == "busy": after["prometheus"]["gauges"]["retiring_request_slots"] = 1
    if defect == "cache":
        for value in (after["first"], after["last"]): value["input_cache_memory_records"] = 1
    if defect == "unreleased-kv":
        for value in (after["first"], after["last"]): value["free_pages"] -= 1
    if defect == "unstable": after["first"]["completed"] += 1
    if defect == "counter-reset": after["last"]["completed"] = 0
    if defect == "uptime-reset": after["prometheus"]["gauges"]["uptime_seconds"] = 0
    if defect == "batch-reset": after["prometheus"]["batches"] = {}
    if defect in ("counter-reset", "uptime-reset", "batch-reset"):
        with pytest.raises(ValueError): phases.deltas(before, after)
    else:
        assert phases.idle(after, cold=True) is (defect is None)
        if defect == "cache": assert phases.idle(after, cold=False)
        delta = phases.deltas(before, after)
        assert delta["counters"]["completed"] == 2 and delta["launches"]["audio_encode"] == 3
        assert delta["graph_batches"] == [dict(graph_index=0, contract_capacity=1, requests=1, submissions=3)]


def wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(32000))
    return output.getvalue()


@pytest.mark.parametrize("defect", [None, "request-scope", "input-store", "host-store", "missing", "duplicate", "not-run"])
def test_cold_policy_requires_live_explicit_cli_overrides(defect):
    argv = ["flare", "run", "package", "--embedding-table-storage", "ram", "--prefix-cache-memory", "0",
        "--prefix-cache-disk-limit", "0", "--prefix-cache-reuse", "request", "--prepared-input-cache", "off"]
    if defect == "request-scope": argv[10] = "shared"
    if defect == "input-store": argv[-1] = "memory"
    if defect == "host-store": argv[6] = "1GiB"
    if defect == "missing": argv = argv[:-2]
    if defect == "duplicate": argv += ["--prepared-input-cache=off"]
    if defect == "not-run": argv[1] = "debug-serve"
    if defect:
        with pytest.raises(ValueError): phases.check_policy(dict(argv=argv), cold=True)
    else:
        phases.check_policy(dict(argv=argv), cold=True)
        actual = phases.process_identity(os.getpid())
        assert actual["pid"] == os.getpid() and actual["start_ticks"] > 0 and actual["argv"]
        assert actual == phases.process_identity(os.getpid())


@pytest.mark.parametrize("defect", [None, "truncated", "summary", "ordinal", "pcm", "envelope", "chunk-gap", "window-gap"])
def test_native_plan_binds_complete_actual_pcm_and_independent_window_count(defect):
    data = wav()
    audio = dict(id="one", samples=16000, pcm_s16le_sha256="pcm")
    header = dict(kind="header", schema=1, inputs=1, envelope_sha256="a" * 64, tensor_parallel=1,
        settings=dict(sample_rate=16000, hop_samples=160, sample_padding=2),
        capacity=dict(frames_per_chunk=100, features_per_chunk=13, subsampling_factor=8, attention_window_frames=800))
    row = dict(kind="input", status="passed", ordinal=0, id="one", samples=16000,
        pcm_f32le_sha256=hashlib.sha256(np.zeros(16000, dtype="<f4").tobytes()).hexdigest(),
        encoded_sha256=hashlib.sha256(data).hexdigest(), chunks=[dict(samples=[0, 16000], frames=100,
            features=13, windows=[dict(frames=[0, 100], features=[0, 13])])])
    summary = dict(kind="summary", inputs=1, failed=0)
    if defect == "summary": summary["failed"] = 1
    if defect == "ordinal": row["ordinal"] = 1
    if defect == "pcm": row["pcm_f32le_sha256"] = "b" * 64
    if defect == "envelope": header["envelope_sha256"] = "b" * 64
    if defect == "chunk-gap": row["chunks"][0]["samples"][0] = 1
    if defect == "window-gap": row["chunks"][0]["windows"][0]["features"][0] = 1
    content = "\n".join(map(json.dumps, [header, row, summary][:2] if defect == "truncated" else [header, row, summary]))
    if defect:
        with pytest.raises(ValueError): phases.native_plan(content, [audio], dict(pcm=data), "a" * 64)
    else:
        assert phases.native_plan(content, [audio], dict(pcm=data), "a" * 64) == dict(one=dict(chunks=1, windows=1))


@pytest.mark.parametrize("defect", [None, "bad-metrics", "missing-header", "length", "wrong-tokens", "cache-not-empty", "boundary-work", "boundary-cancelled", "boundary-stored", "cancel", "warmup-limit", "delayed-drain", "readmission", "pid-change", "json-regression", "intermediate-regression"])
def test_real_http_phases_keep_complete_multipart_and_failure_attempts(monkeypatch, defect):
    # Synthetic warmup accounting keeps this CPU fixture short. It establishes
    # neither measured active warmup nor formal model/performance qualification.
    monkeypatch.setattr(phases, "active_seconds", lambda rows: 0 if defect == "warmup-limit" else 120)
    state, snapshots = dict(parents=0, probes=0, after_second=0), []
    lock, reached, release = Lock(), Event(), Event()
    payload = wav()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def respond(self, body, content_type, headers=()):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for key, value in headers: self.send_header(key, value)
            self.end_headers()
            try: self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError): pass

        def do_GET(self):
            with lock:
                state["probes"] += 1
                parents = state["parents"]
                if parents == 2: state["after_second"] += 1
                extra = defect == "boundary-work" and state["after_second"] >= 4
                busy = defect in ("delayed-drain", "intermediate-regression") and parents == 1 and state["probes"] <= 6
                value, text = metrics(parents * 2 + (2 if extra else 0), busy=int(busy))
                if defect == "json-regression" and parents == 1 and state["probes"] == 4:
                    value["generated_tokens"] += 1
                if defect == "intermediate-regression" and parents:
                    # Keep every later observation at1: the sole regression
                    # is the target2→1, not another1→0 in a following phase.
                    text = text.replace("flare_preemptions_total 0", "flare_preemptions_total " +
                        ("2" if parents == 1 and state["probes"] <= 6 else "1"))
                if state["after_second"] >= 4 and defect == "boundary-cancelled":
                    text = text.replace("flare_requests_cancelled_total 0", "flare_requests_cancelled_total 1")
                if state["after_second"] >= 4 and defect == "boundary-stored":
                    text = text.replace("flare_input_cache_stored_total 0", "flare_input_cache_stored_total 1")
                if defect == "readmission":
                    value["reused_tokens"] = parents * 32
                    text = text.replace("flare_preemptions_total 0", "flare_preemptions_total " + str(parents))
                    text = text.replace('kind="audio_encode"} ' + str(parents * 3), 'kind="audio_encode"} ' + str(parents * 4))
                if defect == "wrong-tokens" and parents: value["generated_tokens"] += 1
                if defect == "cache-not-empty": value["input_cache_memory_records"] = 1
            if self.path == "/metrics?format=json":
                self.respond(b"invalid JSON" if defect == "bad-metrics" else json.dumps(value).encode(), "application/json")
            else:
                assert self.path == "/metrics?format=prometheus"
                self.respond(text.encode(), "text/plain; version=0.0.4")

        def do_POST(self):
            assert self.path == "/v1/audio/transcriptions"
            body = self.rfile.read(int(self.headers["Content-Length"]))
            message = BytesParser(policy=policy.default).parsebytes(
                ("Content-Type: " + self.headers["Content-Type"] + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + body)
            parts = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True) for part in message.iter_parts()}
            assert parts.pop("file") == payload
            assert parts == {key: value.encode() for key, value in dict(model="Qwen/Qwen3-ASR-1.7B", language="en",
                prompt="", temperature="0", seed="0", response_format="json", max_completion_tokens="1024").items()}
            with lock:
                state["parents"] += 1
                ordinal = state["parents"]
            if defect == "cancel" and ordinal > 2:
                reached.set()
                release.wait(timeout=10)
            headers = [] if defect == "missing-header" else [("x-flare-generated-tokens", "2"),
                ("x-flare-finish-reason", "length" if defect == "length" else "stop")]
            self.respond(json.dumps(dict(text="complete fixture", usage=dict(type="duration", seconds=1))).encode(),
                "application/json", headers)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    args = SimpleNamespace(base_url=f"http://127.0.0.1:{server.server_port}", concurrency=2,
        suite="long", regime="unique-content", request_prefix="control")
    audio = dict(id="one", language="en", samples=16000, pcm_s16le_sha256="pcm")
    development = [dict(utterance=audio, group="short", input_set="development", scheduled_seconds=None)]
    selected = [dict(utterance=dict(audio, id=str(i)), group="long", input_set="long", scheduled_seconds=None) for i in range(2)]
    prepared = dict(development=dict(one=dict(chunks=2, windows=3)),
        long={str(i): dict(chunks=2, windows=3) for i in range(2)})
    result = dict(warmups=[], timed=dict(request_prefix="control-timed"), metric_observations=[], drain_observations=[])

    def save():
        # Actual CLI serialization must succeed after every observation.
        json.dumps(result, allow_nan=False)
        snapshots.append(deepcopy(result))

    async def exercise():
        checks = 0
        def verify_server():
            nonlocal checks
            checks += 1
            if defect == "pid-change" and checks == 3:
                raise ValueError("fixture process identity changed")
        task = asyncio.create_task(phases.collect(args, result, selected, development, dict(pcm=payload), prepared, save,
            verify_server=verify_server))
        if defect == "cancel":
            for _ in range(1000):
                if reached.is_set(): break
                if task.done(): await task
                await asyncio.sleep(.01)
            assert reached.is_set()
            task.cancel()
        await task

    try:
        if defect not in (None, "delayed-drain", "readmission"):
            with pytest.raises((ValueError, RuntimeError, asyncio.CancelledError)): asyncio.run(exercise())
        else:
            asyncio.run(exercise())
            assert result["status"] == "http_phases_reconciled_pending_launcher_qualification"
            delta = result["native_metric_deltas"]
            assert delta["counters"]["completed"] == delta["counters"]["generated_tokens"] == 4
            assert delta["launches"]["audio_encode"] == (8 if defect == "readmission" else 6)
            if defect == "readmission":
                assert delta["counters"]["reused_tokens"] == 64
                assert delta["cache_attribution"]["reused_tokens_scope"] is None
                assert delta["cache_attribution"]["preemptions"] == 2
                assert delta["cache_attribution"]["extra_encoder_windows"] == 2
            assert len(result["warmups"]) == 2 and len(result["timed"]["rows"]) == 2
            if defect == "delayed-drain": assert result["drain_observations"][0]["attempts"] == 2
        if defect == "bad-metrics":
            assert result["metric_observations"][-1]["raw"][0]["body"] == "invalid JSON"
        if defect in ("missing-header", "length"):
            assert result["warmups"][0]["rows"][0]["status"] == "failed"
        if defect == "cancel":
            assert result["timed"]["status"] == "failed_or_interrupted"
            assert all(row["status"] == "interrupted" for row in result["timed"]["rows"])
        if defect == "intermediate-regression":
            assert len(result["warmups"]) == 1 and result["drain_observations"][0]["attempts"] == 1
            assert result["metric_observations"][-1]["error"]["type"] == "ValueError"
            assert "decreased" in result["metric_observations"][-1]["error"]["message"]
        assert snapshots
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
