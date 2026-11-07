# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading
import time
import wave

import httpx
import pytest

from qualify.asr_workload import load_pcm, plan, run, summarize


def cohort(count):
    audio = dict(id="speech", samples=16000, language="en", pcm_s16le_sha256="pcm")
    requests = [dict(utterance=audio, group="short", scheduled_seconds=None) for _ in range(count)]
    rows = [dict(ordinal=i, id="speech", samples=16000, group="short", status="not_attempted")
        for i in range(count)]
    return requests, rows


@contextmanager
def server():
    state = dict(lock=threading.Lock(), received=[], active=0, peak=0)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            identity = self.headers["X-Request-Id"]
            with state["lock"]:
                state["received"].append((identity, body))
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                time.sleep(.02 if identity != "failure-1" else .06)
                value = json.dumps(dict(text="speech", usage=dict(type="duration", seconds=1))).encode()
                self.send_response(500 if identity == "failure-1" else 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(value)))
                self.send_header("X-Flare-Generated-Tokens", "3")
                self.send_header("X-Flare-Finish-Reason", "length" if identity == "truncated-0" else "stop")
                self.end_headers()
                try:
                    self.wfile.write(value)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            finally:
                with state["lock"]:
                    state["active"] -= 1

    listener = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=listener.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(listener.server_port), state
    finally:
        listener.shutdown()
        listener.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


async def collect(url, *, count=7, concurrency=2, prefix="success", engine="flare"):
    requests, rows = cohort(count)
    async with httpx.AsyncClient(base_url=url, trust_env=False) as client:
        await run(client, requests, {"pcm": b"verified PCM stand-in at transport seam"},
            engine=engine, suite="short", concurrency=concurrency, request_prefix=prefix, rows=rows)
    return rows


def test_closed_loop_real_http_is_bounded_and_every_request_drains():
    with server() as (url, state):
        rows = asyncio.run(collect(url))
    assert len(state["received"]) == 7 and state["peak"] == 2
    assert {value[0] for value in state["received"]} == {f"success-{i}" for i in range(7)}
    assert all(b'name="max_completion_tokens"\r\n\r\n1024' in body for _, body in state["received"])
    summary = summarize(rows, engine="flare", suite="short", diagnostic=False)
    assert summary["complete_audio_seconds"] == 7
    assert summary["actual_generated_tokens"] == 21
    assert not summary["failed_or_unattempted_ordinals"]
    assert not summary["duration_count_client_criteria_passed"]
    assert not summary["performance_qualified"]


def test_http_failure_stays_in_denominator_and_does_not_abandon_siblings():
    with server() as (url, state):
        rows = asyncio.run(collect(url, prefix="failure"))
    summary = summarize(rows, engine="flare", suite="short", diagnostic=False)
    assert len(state["received"]) == summary["planned_requests"] == 7
    assert summary["failed_or_unattempted_ordinals"] == [1]
    assert summary["complete_audio_seconds"] == 6
    assert summary["actual_generated_tokens"] == 21
    assert rows[1]["http_status"] == 500 and rows[1]["response_body"]
    assert summary["elapsed_seconds"] >= .06


def test_output_cap_is_not_credited_as_complete_audio():
    with server() as (url, _):
        rows = asyncio.run(collect(url, count=1, prefix="truncated"))
    summary = summarize(rows, engine="flare", suite="short", diagnostic=False)
    assert summary["failed_or_unattempted_ordinals"] == [0]
    assert summary["complete_audio_seconds"] == 0
    assert summary["actual_generated_tokens"] == 3 and rows[0]["finish_reason"] == "length"


def test_vllm_http_success_is_not_natural_stop_or_a_token_counter():
    with server() as (url, _):
        rows = asyncio.run(collect(url, count=1, engine="vllm"))
    summary = summarize(rows, engine="vllm", suite="short", diagnostic=False)
    assert rows[0]["status"] == "response_received_unverified_termination"
    assert summary["complete_audio_seconds"] is None and summary["actual_generated_tokens"] is None
    assert summary["termination_reconciliation_required"]


def test_cancellation_preserves_active_and_unattempted_rows():
    async def cancel(url):
        requests, rows = cohort(12)
        async with httpx.AsyncClient(base_url=url, trust_env=False) as client:
            task = asyncio.create_task(run(client, requests, {"pcm": b"audio"}, engine="flare",
                suite="short", concurrency=1, request_prefix="cancel", rows=rows))
            while rows[0]["status"] == "not_attempted":
                await asyncio.sleep(.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        return rows
    with server() as (url, _):
        rows = asyncio.run(cancel(url))
    summary = summarize(rows, engine="flare", suite="short", diagnostic=False)
    assert len(summary["failed_or_unattempted_ordinals"]) == 12
    assert rows[0]["status"] == "interrupted" and rows[-1]["status"] == "not_attempted"
    assert summary["actual_generated_tokens"] is None


def test_mixed_latency_includes_waiting_for_client_capacity():
    rows = [dict(ordinal=0, group="short", status="success", samples=16000, generated_tokens=2,
        scheduled_ns=0, sent_ns=10_000_000_000, received_ns=12_000_000_000, finished_ns=12_000_100_000)]
    summary = summarize(rows, engine="flare", suite="mixed", diagnostic=False)
    assert summary["short_latency"]["p95"] == 12
    assert summary["elapsed_seconds"] == 12
    assert not summarize(rows, engine="flare", suite="mixed", diagnostic=True)["duration_count_client_criteria_passed"]
    assert summarize([], engine="flare", suite="mixed", diagnostic=False)["elapsed_seconds"] is None


def test_mixed_real_http_long_request_does_not_wait_for_short_client_credits():
    async def mixed(url):
        requests, rows = cohort(9)
        for item in requests:
            item["scheduled_seconds"] = 0
        requests[-1]["group"] = rows[-1]["group"] = "long"
        async with httpx.AsyncClient(base_url=url, trust_env=False) as client:
            await run(client, requests, {"pcm": b"audio"}, engine="flare", suite="mixed",
                concurrency=4, request_prefix="mixed", rows=rows)
        return rows
    with server() as (url, state):
        rows = asyncio.run(mixed(url))
    assert len(state["received"]) == 9 and state["peak"] <= 5
    assert all(row["status"] == "success" for row in rows)
    assert rows[-1]["sent_ns"] < rows[4]["sent_ns"]
    assert rows[4]["sent_ns"] >= min(row["received_ns"] for row in rows[:4])
    summary = summarize(rows, engine="flare", suite="mixed", diagnostic=True)
    assert summary["short_latency"]["count"] == 8 and summary["long_latency"]["count"] == 1
    assert summary["short_latency"]["samples"][4] > (rows[4]["received_ns"] - rows[4]["sent_ns"]) / 1e9


def test_mixed_cancellation_records_waiting_and_active_requests():
    async def cancel(url):
        requests, rows = cohort(9)
        for item in requests:
            item["scheduled_seconds"] = 0
        async with httpx.AsyncClient(base_url=url, trust_env=False) as client:
            task = asyncio.create_task(run(client, requests, {"pcm": b"audio"}, engine="flare",
                suite="mixed", concurrency=4, request_prefix="mixed-cancel", rows=rows))
            while any("scheduled_ns" not in row for row in rows):
                await asyncio.sleep(.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        return rows
    with server() as (url, _):
        rows = asyncio.run(cancel(url))
    assert all(row["status"] == "interrupted" for row in rows)
    assert sum("sent_ns" in row for row in rows) == 4
    summary = summarize(rows, engine="flare", suite="mixed", diagnostic=True)
    assert summary["failed_or_unattempted_ordinals"] == list(range(9))


def test_frozen_workload_membership_and_explicit_diagnostic():
    unique = plan("short", "unique-content", 8)
    repeated = plan("short", "repeated-content", 4)
    mixed = plan("mixed", "reasonable-deployment", 4)
    assert len(unique) == len(repeated) == 8169
    assert len({row["utterance"]["pcm_s16le_sha256"] for row in unique}) == 8169
    assert len({row["utterance"]["pcm_s16le_sha256"] for row in repeated}) == 256
    assert [row["scheduled_seconds"] for row in mixed if row["group"] == "short"] == list(range(301))
    assert [row["scheduled_seconds"] for row in mixed if row["group"] == "long"] == list(range(0, 241, 30))
    assert len(plan("short", "unique-content", 1, 3)) == 3
    with pytest.raises(ValueError):
        plan("mixed", "unique-content", 8)
    with pytest.raises(ValueError):
        plan("long", "repeated-content", 1)


def test_pcm_identity_geometry_and_path_are_checked_before_sampling(tmp_path):
    stream = io.BytesIO()
    pcm = b"\x01\x02" * 201
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(pcm)
    (tmp_path / "audio.wav").write_bytes(stream.getvalue())
    row = dict(id="input", path="audio.wav", samples=201, pcm_s16le_sha256=hashlib.sha256(pcm).hexdigest())
    assert load_pcm(tmp_path, [row]) == {row["pcm_s16le_sha256"]: stream.getvalue()}
    for changed in [row | dict(samples=202), row | dict(pcm_s16le_sha256="0" * 64), row | dict(path="../audio.wav")]:
        with pytest.raises(ValueError):
            load_pcm(tmp_path, [changed])
