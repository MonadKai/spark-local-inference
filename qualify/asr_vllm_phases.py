# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Collect complete frozen warmup and timed HTTP phases from pinned vLLM.

The external launcher owns the server, image/configuration verification, worker
observer installation, resource isolation/sampling and normal shutdown. Reconcile
this output with the terminal observer after shutdown. Neither HTTP reset
acknowledgments nor this client establish performance qualification.
"""

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import re
import signal

import httpx

from qualify.asr_performance_run import active_seconds, collect_client_phase
from qualify.asr_workload import ROOT, expected_rows, load_pcm, plan


REASONS = ("stop", "length", "abort", "error", "repetition")


def official_counters(text):
    """Require exactly one observed engine/model series per required counter."""
    result = {}
    for key in (*REASONS, "generated_tokens", "running", "waiting"):
        metric = "vllm:request_success_total" if key in REASONS else {
            "generated_tokens": "vllm:generation_tokens_total",
            "running": "vllm:num_requests_running", "waiting": "vllm:num_requests_waiting"}[key]
        selected = []
        for line in text.splitlines():
            if not line.startswith(metric + "{"):
                continue
            match = re.fullmatch(re.escape(metric) + r'\{([^}]+)\}\s+([^\s]+)', line)
            if match is None:
                raise ValueError("malformed official metric")
            raw, value = match.groups()
            fields = raw.split(",")
            labels = [re.fullmatch(r'([a-zA-Z_][a-zA-Z_0-9]*)="([^"\\]*)"', field) for field in fields]
            if any(label is None for label in labels) or len({label[1] for label in labels}) != len(labels):
                raise ValueError("invalid official metric labels")
            labels = {label[1]: label[2] for label in labels}
            expected = {"engine", "model_name", "finished_reason"} if key in REASONS else {"engine", "model_name"}
            if set(labels) != expected or labels["engine"] != "0" or labels["model_name"] != "Qwen/Qwen3-ASR-1.7B":
                raise ValueError("unexpected metric engine or model")
            if key in REASONS and labels["finished_reason"] not in REASONS:
                raise ValueError("unknown official finish reason")
            if key in REASONS and labels["finished_reason"] != key:
                continue
            number = float(value)
            if not math.isfinite(number) or number < 0 or not number.is_integer():
                raise ValueError("invalid official counter")
            selected.append(int(number))
        if len(selected) != 1:
            raise ValueError("missing or duplicated official counter: " + key)
        result[key] = selected[0]
    return result


async def reset_caches(client, acknowledgments, save):
    for route, params in [("reset_prefix_cache", dict(reset_running_requests="false", reset_external="false")),
            ("reset_mm_cache", None), ("reset_encoder_cache", None)]:
        response = await asyncio.wait_for(client.post("/" + route, params=params), timeout=120)
        acknowledgments.append(dict(route=route, status=response.status_code, body=response.text))
        save()
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("cache reset did not acknowledge HTTP200")
        if route == "reset_prefix_cache":
            payload = response.json()
            if not isinstance(payload, dict) or set(payload) != {"success"} or payload["success"] is not True:
                raise ValueError("prefix cache reset did not acknowledge strict success")


async def get_metrics(client, observations, save, *, timeout):
    observation = dict(status="pending", timeout_seconds=timeout)
    observations.append(observation)
    save()
    try:
        response = await asyncio.wait_for(client.get("/metrics"), timeout=timeout)
        observation.update(status=response.status_code, body=response.text)
        save()
        response.raise_for_status()
        return response.text
    except BaseException as error:
        observation["error"] = dict(type=type(error).__name__, message=str(error))
        save()
        raise


def chunk_counts(requests, prepared):
    result = []
    for item in requests:
        row = item["utterance"]
        source = prepared[item["input_set"]][row["id"]]
        if any(source[name] != row[name] for name in ("id", "samples", "pcm_s16le_sha256")) \
                or not source["chunk_samples"] or any(type(n) is not int or n < 1 for n in source["chunk_samples"]) \
                or sum(source["chunk_samples"]) != row["samples"]:
            raise ValueError("pinned processor chunk geometry differs from frozen audio")
        result.append(len(source["chunk_samples"]))
    return result


def check_boundary_bridge(warm_metrics, before_metrics, warm_encoder, before_encoder):
    if warm_metrics != before_metrics:
        raise ValueError("official counters changed between warmup drain and timed boundary")
    if any(warm_encoder[name] != before_encoder[name] for name in ("prepared", "executed")):
        raise ValueError("encoder dispatch changed between warmup drain and timed boundary")


async def collect(args, result, requests, development, pcm, save):
    task, loop = asyncio.current_task(), asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        limit = args.concurrency + (9 if args.suite == "mixed" else 0)
        async with httpx.AsyncClient(base_url=args.base_url, timeout=1800,
                limits=httpx.Limits(max_connections=limit, max_keepalive_connections=limit),
                follow_redirects=False, trust_env=False) as client:
            async def rpc():
                response = await asyncio.wait_for(client.post("/collective_rpc", json=dict(
                    method="get_asr_encoder_observations", timeout=60)), timeout=70)
                result["rpc_observations"].append(dict(status=response.status_code, body=response.text))
                save()
                response.raise_for_status()
                values = response.json()["results"]
                if len(values) != 1 or values[0]["errors"] or values[0]["pending"] \
                        or values[0]["prepared"] != values[0]["executed"]:
                    raise ValueError("single drained encoder observer required")
                return values[0]

            async def metrics(timeout=30):
                return await get_metrics(client, result["metric_observations"], save, timeout=timeout)

            async def drain(before, chunks):
                deadline = loop.time() + 30
                observation = dict(deadline_monotonic_seconds=deadline, expected_chunks=chunks,
                    before=dict(before), attempts=0)
                result["drain_observations"].append(observation)
                for _ in range(128):
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise TimeoutError("official phase counters did not drain")
                    text = await metrics(remaining)
                    observation.update(attempts=observation["attempts"] + 1,
                        last_monotonic_seconds=loop.time())
                    current = official_counters(text)
                    deltas = {key: current[key] - before[key] for key in (*REASONS, "generated_tokens")}
                    observation.update(last_counters=current, last_deltas=deltas)
                    save()
                    if any(value < 0 for value in deltas.values()) or any(deltas[key] for key in REASONS[1:]) \
                            or deltas["stop"] > chunks:
                        raise ValueError("official phase counters reset or report failure/unplanned work")
                    if current["running"] == current["waiting"] == 0 and deltas["stop"] == chunks:
                        return text, deltas
                    if loop.time() >= deadline:
                        raise TimeoutError("official phase counters did not drain")
                    await asyncio.sleep(.25)
                raise TimeoutError("official phase drain exceeded128 observations")

            async def phase(target, selected, suite):
                await collect_client_phase(client, target, selected, pcm, engine="vllm", suite=suite,
                    regime=args.regime, concurrency=args.concurrency, save=save)

            initial_text = await metrics()
            initial = official_counters(initial_text)
            if initial["running"] or initial["waiting"]:
                raise ValueError("server already has active or waiting work")
            result["metrics_initial"] = initial_text
            result["encoder_initial"] = await rpc()
            additional = 0.
            for index in range(64):
                warmup = dict(request_prefix=args.request_prefix + "-warm" + str(index))
                result["warmups"].append(warmup)
                await phase(warmup, development, "short")
                text, _ = await drain(initial, (index + 1) * sum(result["warmup_chunk_counts"]))
                if index:
                    additional += active_seconds(warmup["rows"])
                if index and additional >= 120:
                    break
            else:
                raise RuntimeError("64 complete development passes did not provide120 active warmup seconds")
            result["metrics_after_warmup"] = text
            result["encoder_after_warmup"] = await rpc()
            if args.regime == "unique-content":
                await reset_caches(client, result["cache_reset_acknowledgments"], save)
            # These snapshots are outside all request timing. MM/encoder HTTP200
            # alone is not a successful cache-state observation.
            result["encoder_before"] = await rpc()
            result["metrics_before"] = await metrics()
            before = official_counters(result["metrics_before"])
            if before["running"] or before["waiting"]:
                raise ValueError("work appeared between warmup and timed admission")
            check_boundary_bridge(official_counters(result["metrics_after_warmup"]), before,
                result["encoder_after_warmup"], result["encoder_before"])
            result["status"] = "timed_phase"
            await phase(result["timed"], requests, args.suite)
            result["metrics_after"], result["official_metric_deltas"] = await drain(before, sum(result["chunk_counts"]))
            result["encoder_after"] = await rpc()
            result["status"] = "http_phases_drained_pending_terminal_reconciliation"
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
    for name in ("performance", "development", "long"):
        parser.add_argument("--" + name + "-pcm-root", type=Path, required=True)
        parser.add_argument("--" + name + "-chunk-plan", type=Path, required=True)
    parser.add_argument("--request-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", args.request_prefix) or args.output.exists():
        parser.error("unique output and1..80 ASCII request prefix required")
    requests = plan(args.suite, args.regime, args.concurrency)
    development = [dict(utterance=row, group="short", input_set="development", scheduled_seconds=None)
        for row in json.loads((ROOT / "development.json").read_bytes())["utterances"]]
    result = dict(scope=__doc__, status="loading_inputs", errors=[], warmups=[],
        cache_reset_acknowledgments=[], rpc_observations=[], metric_observations=[], drain_observations=[],
        chunk_plan_files={}, performance_qualified=False,
        timed=dict(request_prefix=args.request_prefix + "-timed"),
        frozen_files={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ("performance-protocol.json", "performance-inputs.json", "development.json", "long-speech.json")})

    def save():
        pending = args.output.with_suffix(args.output.suffix + ".pending")
        pending.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        pending.replace(args.output)

    save()
    try:
        pcm, prepared = {}, {}
        for name in ("performance", "development", "long"):
            chosen = [item["utterance"] for item in [*development, *requests] if item["input_set"] == name]
            pcm.update(load_pcm(getattr(args, name + "_pcm_root"), chosen))
            content = getattr(args, name + "_chunk_plan").read_bytes()
            result["chunk_plan_files"][name] = hashlib.sha256(content).hexdigest()
            value = json.loads(content)
            rows = value["cases"]
            if value["configuration"] != dict(sample_rate=16000, max_audio_clip_s=30,
                    overlap_chunk_second=1, min_energy_split_window_size=1600) \
                    or value["errors"] or len({row["id"] for row in rows}) != len(rows):
                raise ValueError("failed or duplicate pinned processor plan")
            prepared[name] = {row["id"]: row for row in rows}
        result.update(loaded_pcm_bytes=sum(map(len, pcm.values())), chunk_counts=chunk_counts(requests, prepared),
            warmup_chunk_counts=chunk_counts(development, prepared))
        asyncio.run(collect(args, result, requests, development, pcm, save))
    except BaseException as error:
        result["status"] = "failed_or_interrupted"
        result["errors"].append(dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
