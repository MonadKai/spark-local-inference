# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Client-side corruption controls; no model or performance qualification."""

import asyncio
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Lock, Thread
from types import SimpleNamespace

import httpx
import pytest

from qualify.asr_vllm_phases import (REASONS, check_boundary_bridge, chunk_counts,
    expected_rows, get_metrics, official_counters, reset_caches)
from qualify import asr_vllm_phases


def metrics():
    labels = 'engine="0",model_name="Qwen/Qwen3-ASR-1.7B"'
    rows = [f'vllm:request_success_total{{{labels},finished_reason="{reason}"}} {12 if reason == "stop" else 0}'
        for reason in REASONS]
    rows += [f"vllm:{name}{{{labels}}} {value}" for name, value in
        [("generation_tokens_total", 456), ("num_requests_running", 0), ("num_requests_waiting", 0)]]
    return "\n".join(rows)


def test_official_counters_preserve_actual_chunks_tokens_and_idle_gauges():
    assert official_counters(metrics()) == dict(stop=12, length=0, abort=0, error=0,
        repetition=0, generated_tokens=456, running=0, waiting=0)


@pytest.mark.parametrize("defect", [None, "completed-work", "reset", "encoder-work"])
def test_cache_reset_interval_cannot_absorb_finished_work_or_a_counter_reset(defect):
    before = official_counters(metrics())
    after = dict(before)
    first = dict(prepared=dict(audio=1), executed=dict(audio=1))
    last = deepcopy(first)
    if defect == "completed-work": after.update(stop=13, generated_tokens=457)
    if defect == "reset": after.update(stop=0, generated_tokens=0)
    if defect == "encoder-work": last.update(prepared=dict(audio=1, other=1), executed=dict(audio=1, other=1))
    if defect:
        with pytest.raises(ValueError): check_boundary_bridge(before, after, first, last)
    else:
        check_boundary_bridge(before, after, first, last)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nan", "inf", "negative", "fraction",
    "other-model", "other-engine", "duplicate-label", "spoof-label", "unknown-reason", "timestamp"])
def test_unknown_or_corrupt_official_counts_do_not_become_zero(defect):
    text = metrics()
    if defect == "missing": text = "\n".join(text.splitlines()[:-1])
    if defect == "duplicate": text += "\n" + text.splitlines()[0]
    if defect in ("nan", "inf", "negative", "fraction"):
        text = text.replace("} 456", "} " + dict(nan="NaN", inf="Inf", negative="-1", fraction="1.5")[defect])
    if defect == "other-model": text = text.replace("Qwen/Qwen3-ASR-1.7B", "other")
    if defect == "other-engine": text = text.replace('engine="0"', 'engine="1"')
    if defect == "duplicate-label": text = text.replace('engine="0"', 'engine="0",engine="0"')
    if defect == "spoof-label": text = text.replace('engine="0"', 'other_engine="0"')
    if defect == "unknown-reason": text = text.replace('finished_reason="abort"', 'finished_reason="unknown"')
    if defect == "timestamp": text = text.replace("} 456", "} 456 123")
    with pytest.raises(ValueError): official_counters(text)


@pytest.mark.parametrize("defect", [None, "id", "hash", "samples", "lost-tail", "empty", "float"])
def test_chunk_counts_bind_the_actual_pcm_and_complete_tail(defect):
    audio = dict(id="long", language="en", samples=1000, pcm_s16le_sha256="pcm")
    selected = [dict(utterance=audio, group="long", input_set="long", scheduled_seconds=30)]
    source = dict(audio, chunk_samples=[400, 600])
    if defect == "id": source["id"] = "other"
    if defect == "hash": source["pcm_s16le_sha256"] = "other"
    if defect == "samples": source["samples"] += 1
    if defect == "lost-tail": source["chunk_samples"] = [400]
    if defect == "empty": source["chunk_samples"] = []
    if defect == "float": source["chunk_samples"] = [400., 600]
    if defect:
        with pytest.raises(ValueError): chunk_counts(selected, dict(long=dict(long=source)))
    else:
        assert chunk_counts(selected, dict(long=dict(long=source))) == [2]
        assert expected_rows(selected) == [dict(audio, group="long", scheduled_seconds=30)]


@pytest.mark.parametrize("prefix_status,payload,mm_status", [
    (200, {"success": True}, 200), (200, {"success": False}, 200),
    (200, {"success": 1}, 200), (200, {}, 200), (500, {"success": True}, 200),
    (200, {"success": True}, 500)])
def test_actual_http_reset_order_and_failed_acknowledgments(prefix_status, payload, mm_status):
    calls, saved, acknowledgments = [], [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def do_POST(self):
            calls.append(self.path)
            route = self.path.partition("?")[0]
            status = prefix_status if route == "/reset_prefix_cache" else mm_status if route == "/reset_mm_cache" else 200
            content = json.dumps(payload).encode() if route == "/reset_prefix_cache" else b""
            self.send_response(status)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    async def exercise():
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{server.server_port}", trust_env=False) as client:
            await reset_caches(client, acknowledgments, lambda: saved.append(deepcopy(acknowledgments)))

    try:
        valid = prefix_status == mm_status == 200 and payload.get("success") is True
        if valid:
            asyncio.run(exercise())
            assert [row["route"] for row in acknowledgments] == ["reset_prefix_cache", "reset_mm_cache", "reset_encoder_cache"]
        else:
            with pytest.raises((ValueError, httpx.HTTPStatusError)): asyncio.run(exercise())
        assert calls[0] == "/reset_prefix_cache?reset_running_requests=false&reset_external=false"
        assert len(calls) == len(acknowledgments) == len(saved)
        assert saved[-1] == acknowledgments
        if prefix_status != 200 or payload.get("success") is not True:
            assert len(calls) == 1
        if mm_status != 200:
            assert len(calls) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("defect", [None, "bad-metrics", "bad-rpc", "boundary-work", "cancel", "warmup-limit"])
def test_phase_collection_preserves_http_failures_cancellation_and_bounded_warmup(monkeypatch, defect):
    # Tiny real HTTP client/control-plane fixture. This endpoint executes no
    # audio model. Synthetic active-time selection only makes warmup bounded
    # in a CPU test and cannot establish measured warmup or performance.
    monkeypatch.setattr(asr_vllm_phases, "active_seconds", lambda rows: 0 if defect == "warmup-limit" else 120)
    state, snapshots = dict(stop=0, metrics=0), []
    lock, reached, release = Lock(), Event(), Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def respond(self, content):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            try: self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError): pass

        def do_GET(self):
            with lock:
                state["metrics"] += 1
                stop, ordinal = state["stop"], state["metrics"]
            if defect == "boundary-work" and ordinal == 4: stop += 1
            text = metrics().replace("} 12", "} " + str(stop)).replace("} 456", "} " + str(stop * 2))
            if defect == "bad-metrics" and ordinal == 2: text = "broken raw metrics"
            self.respond(text.encode())

        def do_POST(self):
            content = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            route = self.path.partition("?")[0]
            if route == "/v1/audio/transcriptions":
                with lock:
                    state["stop"] += 1
                    stop = state["stop"]
                if defect == "cancel" and stop > 4:
                    reached.set()
                    release.wait(timeout=10)
                value = dict(text="complete fixture", usage=dict(type="duration", seconds=1))
            elif route == "/collective_rpc":
                assert json.loads(content)["method"] == "get_asr_encoder_observations"
                with lock: stop = state["stop"]
                value = dict(results=[dict(errors={"fixture_failure": 1} if defect == "bad-rpc" else {},
                    pending=[], prepared=dict(audio=stop), executed=dict(audio=stop))])
            else:
                value = dict(success=True) if route == "/reset_prefix_cache" else None
            self.respond(json.dumps(value).encode() if value is not None else b"")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    args = SimpleNamespace(base_url=f"http://127.0.0.1:{server.server_port}", concurrency=2,
        suite="short", regime="unique-content", request_prefix="control")
    rows = [dict(utterance=dict(id=str(i), language="en", samples=16000, pcm_s16le_sha256=str(i)),
        group="short", input_set="development", scheduled_seconds=None) for i in range(2)]
    result = dict(warmups=[], timed=dict(request_prefix="control-timed"), cache_reset_acknowledgments=[],
        metric_observations=[], rpc_observations=[], drain_observations=[], warmup_chunk_counts=[1, 1], chunk_counts=[1, 1])

    async def exercise():
        task = asyncio.create_task(asr_vllm_phases.collect(args, result, rows, rows,
            {"0": b"fixture audio 0", "1": b"fixture audio 1"}, lambda: snapshots.append(deepcopy(result))))
        if defect == "cancel":
            for _ in range(1000):
                if reached.is_set(): break
                if task.done(): await task
                await asyncio.sleep(.01)
            assert reached.is_set()
            task.cancel()
        await task

    try:
        if defect:
            with pytest.raises((ValueError, RuntimeError, asyncio.CancelledError)): asyncio.run(exercise())
        else:
            asyncio.run(exercise())
            assert result["status"] == "http_phases_drained_pending_terminal_reconciliation"
            assert result["official_metric_deltas"] == dict(stop=2, length=0, abort=0, error=0, repetition=0, generated_tokens=4)
            assert len(result["warmups"]) == 2 and len(result["timed"]["rows"]) == 2
        if defect == "bad-metrics":
            assert result["metric_observations"][-1]["body"] == "broken raw metrics"
        if defect == "bad-rpc":
            assert json.loads(result["rpc_observations"][-1]["body"])["results"][0]["errors"]
        if defect == "boundary-work":
            assert official_counters(result["metrics_before"])["stop"] == 5
        if defect == "cancel":
            assert len(result["timed"]["rows"]) == 2
            assert result["timed"]["status"] == "failed_or_interrupted"
            assert any(row["status"] == "interrupted" for row in result["timed"]["rows"])
            assert snapshots[-1]["timed"] == result["timed"]
        if defect == "warmup-limit": assert len(result["warmups"]) == 64
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_stalled_metrics_obeys_total_deadline_and_retains_timeout():
    reached, release, observations, saved = Event(), Event(), [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def do_GET(self):
            reached.set()
            release.wait(timeout=5)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def exercise():
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{server.server_port}", timeout=1800, trust_env=False) as client:
            await get_metrics(client, observations, lambda: saved.append(deepcopy(observations)), timeout=.05)

    try:
        with pytest.raises(TimeoutError): asyncio.run(exercise())
        assert reached.is_set()
        assert observations[0]["status"] == "pending" and observations[0]["error"]["type"] == "TimeoutError"
        assert saved[-1] == observations
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
