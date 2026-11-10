# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Finite HTTP concurrency/order and post-cancellation reuse checks.

This observes actual returned text/counts and terminal headers, not native token
IDs or teacher logits. The launcher must bind the binary/package/source and
resource isolation. HTTP calls and atomic evidence writes exclude performance
qualification. The complete fixed request schedule is recorded before admission.
"""

import argparse
import asyncio
import hashlib
import io
import json
from pathlib import Path
import wave

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--pcm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-pcm-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = Path(__file__).resolve().parents[1] / "benchmarks/qwen3_asr/development.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["split"] != "development" or len(manifest["utterances"]) != 256:
        raise ValueError("fixed independent development split required")
    selected = manifest["utterances"][::32]
    if len(selected) != 8 or {row["language"] for row in selected} != {"en", "zh"}:
        raise ValueError("eight fixed mixed-language inputs required")
    if args.output.exists():
        raise ValueError("preserve previous attempts")
    requests = []
    for row in selected:
        data = (args.pcm_root / row["path"]).read_bytes()
        with wave.open(io.BytesIO(data), "rb") as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) \
                    != (1, 2, 16000, row["samples"]):
                raise ValueError("frozen upload geometry differs")
            if hashlib.sha256(stream.readframes(stream.getnframes())).hexdigest() != row["pcm_s16le_sha256"]:
                raise ValueError("frozen PCM differs")
        requests.append((row, data))
    long_manifest = json.loads((manifest_path.parent / "long-speech.json").read_text())
    long_row = long_manifest["utterances"][-1]
    long_data = (args.long_pcm_root / long_row["path"]).read_bytes()
    with wave.open(io.BytesIO(long_data), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) \
                != (1, 2, 16000, long_row["samples"]) or hashlib.sha256(stream.readframes(stream.getnframes())).hexdigest() != long_row["pcm_s16le_sha256"]:
            raise ValueError("frozen cancellation PCM differs")
    schedule = [dict(batch=1, order="baseline", repeat=0, members=list(range(8)))]
    for batch in (1, 2, 4, 8):
        for order in ("forward", "reverse"):
            for repeat in (0, 1):
                ids = list(range(8)) if order == "forward" else list(reversed(range(8)))
                schedule.append(dict(batch=batch, order=order, repeat=repeat, members=ids))
    schedule.append(dict(batch=8, order="after-parent-cancel", repeat=0, members=list(reversed(range(8)))))
    record = dict(scope=__doc__, cancellation_control=dict(id=long_row["id"], status="planned",
        requirement="disconnect after observing both active and queued child requests, then await retirement and submit eight concurrent requests"),
        performance_qualified=False, native_token_ids_observed=False,
        batch_dimension="HTTP client concurrency; actual GPU batch rows/slot identities are not observed",
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        input_ids=[row["id"] for row, _ in requests], schedule=schedule,
        planned_requests=8 * len(schedule), cases=[], errors=[], status="running")
    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".pending")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)
    save()
    async def run():
        async def metrics_snapshot(client):
            response = await client.get("/metrics", timeout=2)
            response.raise_for_status()
            values = {}
            for line in response.text.splitlines():
                if not line or line.startswith("#"):
                    continue
                name, value = line.rsplit(" ", 1)
                if "{" not in name:
                    values[name] = int(value) if value.isdigit() else float(value)
            return values
        baseline = {}
        async with httpx.AsyncClient(base_url=args.base_url, timeout=1800) as client:
            models = await client.get("/v1/models"); models.raise_for_status()
            if [(value["id"], value.get("max_model_len")) for value in models.json()["data"]] != [("Qwen/Qwen3-ASR-1.7B", 8192)]:
                raise ValueError("native model/capacity differs")
            async def submit(index, plan_index):
                row, data = requests[index]
                result = dict(id=row["id"], plan_index=plan_index, input_index=index, status="failed")
                record["cases"].append(result)
                try:
                    response = await client.post("/v1/audio/transcriptions", data=dict(model="Qwen/Qwen3-ASR-1.7B",
                        language=row["language"], prompt="", temperature="0", seed="0", response_format="json",
                        max_completion_tokens="1024"), files={"file": (row["id"] + ".wav", data, "audio/wav")})
                    result.update(http_status=response.status_code, body=response.text,
                        content_type=response.headers.get("content-type"), finish_reason=response.headers.get("x-flare-finish-reason"),
                        generated_tokens_header=response.headers.get("x-flare-generated-tokens"))
                    if response.status_code != 200 or (result["content_type"] or "").partition(";")[0] != "application/json":
                        raise ValueError("native HTTP response failed")
                    body = response.json()
                    if not isinstance(body.get("text"), str) or result["finish_reason"] != "stop" \
                            or int(result["generated_tokens_header"]) < 1:
                        raise ValueError("missing complete natural ASR response/count")
                    if body.get("usage") != dict(type="duration", seconds=(row["samples"] + 15999) // 16000):
                        raise ValueError("duration usage differs")
                    value = body, int(result["generated_tokens_header"])
                    if plan_index == 0:
                        baseline[index] = value
                    elif baseline.get(index) != value:
                        raise ValueError("same input/parameters changed text/count/usage across batch or row schedule")
                    result["status"] = "success"
                except Exception as error:
                    result["error"] = dict(type=type(error).__name__, message=str(error))
                    record["errors"].append(dict(id=row["id"], plan_index=plan_index))
                finally:
                    save()
            for plan_index, plan in enumerate(schedule):
                if plan["order"] == "after-parent-cancel":
                    before = await metrics_snapshot(client)
                    cancellation = record["cancellation_control"]
                    async with httpx.AsyncClient(base_url=args.base_url, timeout=1800) as cancelled_client:
                        task = asyncio.create_task(cancelled_client.post("/v1/audio/transcriptions",
                            data=dict(model="Qwen/Qwen3-ASR-1.7B", language=long_row["language"], prompt="",
                                temperature="0", seed="0", response_format="json", max_completion_tokens="1024"),
                            files={"file": (long_row["id"] + ".wav", long_data, "audio/wav")}))
                        try:
                            deadline = asyncio.get_running_loop().time() + 30
                            while True:
                                current = await metrics_snapshot(client)
                                if current["flare_active_requests"] > 0 and current["flare_pending_requests"] > 0:
                                    cancellation["metrics_at_disconnect"] = current
                                    break
                                if task.done():
                                    response = await task
                                    cancellation.update(http_status=response.status_code, body=response.text)
                                    raise ValueError("long request finished before the required active/queued cancellation window")
                                if asyncio.get_running_loop().time() >= deadline:
                                    raise TimeoutError("active and queued child state not observed")
                                await asyncio.sleep(.001)
                        finally:
                            cancellation["client_task_cancel_requested"] = task.cancel()
                            cancelled_result = await asyncio.gather(task, return_exceptions=True)
                            if isinstance(cancelled_result[0], httpx.Response):
                                cancellation.update(http_status=cancelled_result[0].status_code, body=cancelled_result[0].text)
                    if not cancellation["client_task_cancel_requested"]:
                        raise ValueError("parent completed before actual client cancellation")
                    deadline = asyncio.get_running_loop().time() + 30
                    while True:
                        current = await metrics_snapshot(client)
                        if current["flare_requests_cancelled_total"] > before["flare_requests_cancelled_total"] and all(current[key] == 0 for key in ("flare_active_requests", "flare_pending_requests", "flare_retiring_request_slots")):
                            cancellation.update(status="passed_active_queued_disconnect_and_retirement", metrics_after=current)
                            save()
                            break
                        if asyncio.get_running_loop().time() >= deadline:
                            cancellation.update(status="failed_cancel_or_retirement", metrics_after=current)
                            raise TimeoutError("cancelled long children did not release request ownership")
                        await asyncio.sleep(.01)
                for start in range(0, 8, plan["batch"]):
                    await asyncio.gather(*(submit(index, plan_index)
                        for index in plan["members"][start:start + plan["batch"]]))
                # Slots retire asynchronously after HTTP completion. Observe
                # that this finite cohort eventually releases request ownership.
                deadline = asyncio.get_running_loop().time() + 30
                while True:
                    metrics = await metrics_snapshot(client)
                    if all(metrics[key] == 0 for key in ("flare_active_requests", "flare_retiring_request_slots", "flare_pending_requests")):
                        break
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError("request ownership did not drain")
                    await asyncio.sleep(.01)
            record["final_metrics"] = metrics
    try:
        asyncio.run(run())
        record["status"] = "passed_finite_text_count_schedule" if ( not record["errors"] and len(record["cases"]) == record["planned_requests"]
            and record["cancellation_control"]["status"] == "passed_active_queued_disconnect_and_retirement") else "failed_finite_schedule"
    except BaseException as error:
        record["status"] = "failed_or_interrupted"
        record["errors"].append(dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        save()
    raise SystemExit(int(record["status"] != "passed_finite_text_count_schedule"))


if __name__ == "__main__":
    main()
