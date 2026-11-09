# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Run one frozen ASR HTTP workload against an independently managed server.

All PCM is verified and loaded before sampling. Responses and timestamps stay
in bounded memory until the cohort drains or is interrupted. The launcher owns
warmup, isolation, engine identity, resource sampling and cache/compile checks.
For vLLM, terminal tokens and complete audio require the separate engine observer;
an HTTP 200 alone never qualifies complete-ASR throughput.
"""

import argparse
import asyncio
from collections import deque
import hashlib
import io
import json
from pathlib import Path
import re
import signal
import statistics
import time
import wave

import httpx


ROOT = Path(__file__).resolve().parents[1] / "benchmarks/qwen3_asr"


def load_pcm(root, utterances):
    result = {}
    for row in utterances:
        identity = row["pcm_s16le_sha256"]
        if identity in result:
            continue
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("PCM path escapes its input directory")
        data = path.read_bytes()
        with wave.open(io.BytesIO(data), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) \
                    != (1, 2, 16000, row["samples"]):
                raise ValueError("PCM geometry differs: " + row["id"])
            actual = hashlib.sha256(audio.readframes(audio.getnframes())).hexdigest()
        if actual != identity:
            raise ValueError("PCM content differs: " + row["id"])
        result[identity] = data
    return result


def plan(suite, regime, concurrency, diagnostic_limit=None):
    if concurrency not in (1, 2, 4, 8):
        raise ValueError("concurrency must be 1, 2, 4 or 8")
    if suite not in ("short", "long", "mixed") or regime not in \
            ("unique-content", "repeated-content", "reasonable-deployment"):
        raise ValueError("unknown workload or cache regime")
    if suite != "short" and regime == "repeated-content":
        raise ValueError("repeated-content is a short-workload cell")
    if suite == "long" and concurrency != 1 or suite == "mixed" and concurrency != 4:
        raise ValueError("long and mixed client capacities are frozen at one and four")
    if diagnostic_limit is not None and (type(diagnostic_limit) is not int or diagnostic_limit < 1):
        raise ValueError("diagnostic limit must be a positive integer")
    unique = json.loads((ROOT / "performance-inputs.json").read_bytes())["utterances"]
    development = json.loads((ROOT / "development.json").read_bytes())["utterances"]
    long = json.loads((ROOT / "long-speech.json").read_bytes())["utterances"]
    if suite == "short":
        chosen = unique if regime != "repeated-content" else \
            [development[i % len(development)] for i in range(len(unique))]
        rows = [dict(utterance=row, group="short", input_set="development" if
            regime == "repeated-content" else "performance", scheduled_seconds=None) for row in chosen]
    elif suite == "long":
        rows = [dict(utterance=row, group="long", input_set="long", scheduled_seconds=None) for row in long]
    else:
        rows = [dict(utterance=row, group="short", input_set="performance", scheduled_seconds=i)
            for i, row in enumerate(unique[:301])]
        rows += [dict(utterance=row, group="long", input_set="long", scheduled_seconds=i * 30)
            for i, row in enumerate(long)]
        rows.sort(key=lambda row: (row["scheduled_seconds"], row["group"] != "short"))
    if diagnostic_limit is not None:
        rows = rows[:diagnostic_limit]
    return rows


def response_value(response, engine, samples):
    if response.status_code != 200 or response.headers.get("content-type", "").partition(";")[0].strip().lower() \
            != "application/json":
        raise ValueError("transcription did not return HTTP 200 JSON")
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("text"), str) or not body["text"].strip():
        raise ValueError("transcription text is empty or invalid")
    if body.get("usage") != dict(type="duration", seconds=(samples + 15999) // 16000):
        raise ValueError("duration usage differs")
    if engine == "flare":
        count = response.headers.get("x-flare-generated-tokens", "")
        if not count.isdecimal() or int(count) < 1 or response.headers.get("x-flare-finish-reason") != "stop":
            raise ValueError("missing actual token count or non-natural terminal")
        return dict(text=body["text"], generated_tokens=int(count), finish_reason="stop", status="success")
    return dict(text=body["text"], generated_tokens=None, finish_reason=None,
        status="response_received_unverified_termination")


async def run(client, requests, pcm, *, engine, suite, concurrency, request_prefix, rows):
    """Run a complete cohort; interruption still preserves every planned row."""
    origin = time.perf_counter_ns()
    short_limit = asyncio.Semaphore(concurrency)
    long_limit = asyncio.Semaphore(9 if suite == "mixed" else 1)

    async def submit(index):
        item, row = requests[index], rows[index]
        audio = item["utterance"]
        scheduled = item["scheduled_seconds"]
        if scheduled is not None:
            row["scheduled_ns"] = origin + int(scheduled * 1e9)
        try:
            if scheduled is not None:
                await asyncio.sleep(max(0, (row["scheduled_ns"] - time.perf_counter_ns()) / 1e9))
            limit = long_limit if item["group"] == "long" else short_limit
            async with limit:
                form = dict(model="Qwen/Qwen3-ASR-1.7B", language=audio["language"], prompt="",
                    temperature="0", seed="0", response_format="json", max_completion_tokens="1024")
                row.update(status="attempting", sent_ns=time.perf_counter_ns())
                response = await client.post("/v1/audio/transcriptions", data=form,
                    headers={"X-Request-Id": request_prefix + "-" + str(index)},
                    files={"file": (audio["id"] + ".wav", pcm[audio["pcm_s16le_sha256"]], "audio/wav")})
                row.update(received_ns=time.perf_counter_ns(), http_status=response.status_code,
                    response_body=response.text, response_headers=dict(response.headers))
                if engine == "flare":
                    count = response.headers.get("x-flare-generated-tokens", "")
                    row["generated_tokens"] = int(count) if count.isdecimal() else None
                    row["finish_reason"] = response.headers.get("x-flare-finish-reason")
                row.update(response_value(response, engine, audio["samples"]))
        except BaseException as error:
            row.update(status="failed" if isinstance(error, Exception) else "interrupted",
                error=dict(type=type(error).__name__, message=str(error)))
            if not isinstance(error, Exception):
                raise
        finally:
            # Validation time is retained separately from the wire response.
            row["finished_ns"] = time.perf_counter_ns()

    async def worker():
        while indices:
            await submit(indices.popleft())

    indices = deque(range(len(requests)))
    tasks = [asyncio.create_task(submit(i)) for i in indices] if suite == "mixed" else \
        [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return origin


def summarize(rows, *, engine, suite, diagnostic):
    sent = [row["sent_ns"] for row in rows if "sent_ns" in row]
    received = [row["received_ns"] for row in rows if "received_ns" in row]
    # Failed transport attempts and drain are included; do not stop at last 200.
    finished = [row.get("received_ns", row["finished_ns"]) for row in rows if "sent_ns" in row]
    start = min((row["scheduled_ns"] for row in rows if "scheduled_ns" in row), default=None) \
        if suite == "mixed" else min(sent, default=None)
    elapsed = (max(finished) - start) / 1e9 if start is not None and finished else None
    successful = [row for row in rows if row["status"] in ("success", "response_received_unverified_termination")]
    failed = [row["ordinal"] for row in rows if row["status"] not in ("success", "response_received_unverified_termination")]

    def latencies(group=None):
        values = [(row.get("received_ns", row["finished_ns"]) - row.get("scheduled_ns", row["sent_ns"])) / 1e9
            for row in rows if "sent_ns" in row and (group is None or row["group"] == group)]
        if not values:
            return dict(count=0, samples=[], mean=None, p50=None, p95=None)
        ordered = sorted(values)
        # Linear interpolation with index (n-1)*p, frozen before sampling.
        def percentile(p):
            index = (len(ordered) - 1) * p
            lo = int(index)
            return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (index - lo)
        return dict(count=len(values), samples=values, mean=statistics.mean(values),
            p50=percentile(.5), p95=percentile(.95))

    complete_seconds = sum(row["samples"] for row in rows if row["status"] == "success") / 16000
    observed_tokens = sum(row.get("generated_tokens") or 0 for row in rows) if engine == "flare" else None
    token_count_complete = engine == "flare" and len(sent) == len(rows) and all(
        type(row.get("generated_tokens")) is int and row["generated_tokens"] > 0 for row in rows)
    return dict(planned_requests=len(rows), attempted_requests=len(sent), http_responses=len(received),
        successful_http_requests=len(successful), failed_or_unattempted_ordinals=failed,
        elapsed_seconds=elapsed, latency=latencies(), short_latency=latencies("short"),
        long_latency=latencies("long"),
        complete_audio_seconds=complete_seconds if engine == "flare" else None,
        complete_audio_seconds_per_elapsed=complete_seconds / elapsed if engine == "flare" and elapsed else None,
        observed_generated_tokens=observed_tokens,
        actual_generated_tokens=observed_tokens if token_count_complete else None,
        duration_count_client_criteria_passed=(not diagnostic and not failed and elapsed is not None
            and (suite == "long" or elapsed >= 300 and len(successful) >= 100)),
        termination_reconciliation_required=engine == "vllm", performance_qualified=False,
        internal_ttft=None, internal_ttft_unavailable_reason="HTTP client does not observe model TTFT")


async def collect(args, result, requests, pcm):
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        limit = args.concurrency + (9 if args.suite == "mixed" else 0)
        async with httpx.AsyncClient(base_url=args.base_url, timeout=1800,
                limits=httpx.Limits(max_connections=limit, max_keepalive_connections=limit),
                follow_redirects=False, trust_env=False) as client:
            result["origin_ns"] = await run(client, requests, pcm, engine=args.engine, suite=args.suite,
                concurrency=args.concurrency, request_prefix=args.request_prefix, rows=result["rows"])
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--engine", choices=("flare", "vllm"), required=True)
    parser.add_argument("--suite", choices=("short", "long", "mixed"), required=True)
    parser.add_argument("--regime", choices=("unique-content", "repeated-content", "reasonable-deployment"), required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--performance-pcm-root", type=Path, required=True)
    parser.add_argument("--development-pcm-root", type=Path, required=True)
    parser.add_argument("--long-pcm-root", type=Path, required=True)
    parser.add_argument("--request-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-limit", type=int, help="explicitly ineligible abbreviated diagnostic")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", args.request_prefix):
        parser.error("request prefix must have 1..96 ASCII letters, digits, underscores or hyphens")
    if args.output.exists():
        parser.error("preserve every prior attempt; output already exists")
    requests = plan(args.suite, args.regime, args.concurrency, args.diagnostic_limit)
    result = dict(scope=__doc__, status="loading_inputs", engine=args.engine, suite=args.suite,
        regime=args.regime, concurrency=args.concurrency, diagnostic_limit=args.diagnostic_limit,
        request_prefix=args.request_prefix, performance_qualified=False, errors=[],
        frozen_files={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ("performance-protocol.json", "performance-inputs.json", "development.json", "long-speech.json")},
        rows=[dict(ordinal=i, id=item["utterance"]["id"], group=item["group"],
            language=item["utterance"]["language"], samples=item["utterance"]["samples"],
            pcm_s16le_sha256=item["utterance"]["pcm_s16le_sha256"], status="not_attempted")
            for i, item in enumerate(requests)])

    def save():
        pending = args.output.with_suffix(args.output.suffix + ".pending")
        pending.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        pending.replace(args.output)

    save()
    try:
        pcm = {}
        roots = dict(performance=args.performance_pcm_root, development=args.development_pcm_root,
            long=args.long_pcm_root)
        for name, root in roots.items():
            pcm.update(load_pcm(root, [item["utterance"] for item in requests if item["input_set"] == name]))
        result["loaded_pcm_bytes"] = sum(map(len, pcm.values()))
        result["status"] = "running"
        save()
        asyncio.run(collect(args, result, requests, pcm))
        result["status"] = "cohort_drained"
    except BaseException as error:
        result.update(status="failed_or_interrupted")
        result["errors"].append(dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        result["summary"] = summarize(result["rows"], engine=args.engine,
            suite=args.suite, diagnostic=args.diagnostic_limit is not None)
        save()
    if result["summary"]["failed_or_unattempted_ordinals"]:
        raise RuntimeError("ASR cohort contains failed or unattempted requests")


if __name__ == "__main__":
    main()
