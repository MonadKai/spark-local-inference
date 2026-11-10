# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Collect actual raw audio-chat tokens and worker observation timestamps.

The frozen development inputs test native schedule invariance, not recognition
quality or performance. HTTP concurrency does not establish actual GPU batch or
slot identity. Server timestamps include CPU completion observation/delivery.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
from threading import RLock
import wave

import httpx

COHORT = ("en-84-121123-0007", "en-652-130737-0008", "en-1673-143397-0009", "en-2035-147960-0015",
    "zh-BAC009S0724W0235", "zh-BAC009S0728W0367", "zh-BAC009S0732W0169", "zh-BAC009S0736W0424")


def _request(client, row, body, observed, record, save, record_lock, *,
        schedule="baseline", concurrency=1):
    body = dict(body)
    if observed:
        body.update(return_token_ids=True, return_token_timestamps=True)
    with record_lock:
        result = dict(attempt=len(record["attempts"]), id=row["id"], observed=observed,
            status="requesting", context=dict(schedule=schedule, concurrency=concurrency))
        record["attempts"].append(result)
        save()
    try:
        response = client.post("/v1/chat/completions", json=body)
        # Each worker saves its raw response before validation or a
        # sibling's exception can abort collection of this batch.
        with record_lock:
            result.update(http_status=response.status_code,
                content_type=response.headers.get("content-type"), response_body=response.text)
            save()
        assert response.status_code == 200, result
        data = response.json()
        choice = data["choices"][0]
        assert len(data["choices"]) == 1 and choice["finish_reason"] == "stop", result
        assert isinstance(choice["message"]["content"], str), result
        assert data["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, result
        if observed:
            ids, stamps = choice["token_ids"], choice["token_timestamps_ns"]
            assert ids and len(ids) == len(stamps) == data["usage"]["completion_tokens"], result
            assert all(type(value) is int and 0 <= value < 151936 for value in ids), result
            assert ids[-1] == 151645, result
            assert all(type(value) is int for value in stamps) and stamps == sorted(stamps), result
            prompt = data["prompt_token_ids"]
            assert prompt and len(prompt) == data["usage"]["prompt_tokens"], result
            assert all(type(value) is int and 0 <= value < 151936 for value in prompt), result
            timing = data["performance"]
            assert timing["ingress_ns"] <= timing["enqueue_ns"] <= stamps[0], result
            with record_lock:
                result["server_observed_first_raw_token_ns"] = stamps[0] - timing["ingress_ns"]
                result["queue_submission_to_first_raw_token_ns"] = stamps[0] - timing["enqueue_ns"]
        with record_lock:
            result["status"] = "response_validated"
            save()
        return result, data
    except BaseException as error:
        with record_lock:
            result.update(status="failed_or_interrupted",
                error=dict(type=type(error).__name__, message=str(error)))
            save()
        raise


def _compare(result, actual, previous, save, *, ids):
    fields = [("message", actual["choices"][0]["message"], previous["choices"][0]["message"]),
        ("usage", actual["usage"], previous["usage"])]
    if ids:
        fields.extend((("prompt_token_ids", actual["prompt_token_ids"], previous["prompt_token_ids"]),
            ("token_ids", actual["choices"][0]["token_ids"], previous["choices"][0]["token_ids"])))
    for name, value, expected in fields:
        if value != expected:
            result["comparison_failed_field"] = name
            save()
            raise AssertionError(dict(attempt=result["attempt"], id=result["id"],
                context=result["context"], differing_field=name))


def _refuse(client, body, addition, record, save):
    result = dict(addition=addition, state="requesting")
    record["refusals"].append(result)
    save()
    try:
        response = client.post("/v1/chat/completions", json=dict(body, **addition))
        result.update(status=response.status_code, body=response.text, state="response_received")
        save()
        assert response.status_code == 400 and "error" in response.json()
        result["state"] = "response_validated"
        save()
    except BaseException as error:
        result.update(state="failed_or_interrupted",
            error=dict(type=type(error).__name__, message=str(error)))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--pcm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = Path(__file__).resolve().parents[1] / "benchmarks/qwen3_asr/development.json"
    manifest = json.loads(manifest_path.read_bytes())
    by_id = {row["id"]: row for row in manifest["utterances"]}
    assert manifest["split"] == "development" and len(by_id) == len(manifest["utterances"]) == 256
    cases = [by_id[identity] for identity in COHORT]
    assert {row["language"] for row in cases} == {"en", "zh"}
    if args.output.exists():
        raise ValueError("preserve the previous native observation attempt")
    record = dict(scope=__doc__.strip(), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        input_ids=list(COHORT), planned_requests=83,
        language="automatic audio chat, separate from forced-language qualification",
        independent_sdk_prompt_match=False, actual_gpu_batch_and_slot_observed=False,
        binary_package_identity_verified_by_client=False, attempts=[], cases=[], refusals=[], errors=[], status="running")
    record_lock = RLock()

    def save():
        with record_lock:
            pending = args.output.with_suffix(".pending")
            pending.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            pending.replace(args.output)

    bodies = {}
    for row in cases:
        raw = (args.pcm_root / row["path"]).read_bytes()
        with wave.open(io.BytesIO(raw), "rb") as pcm:
            assert (pcm.getnchannels(), pcm.getsampwidth(), pcm.getframerate(), pcm.getnframes()) \
                == (1, 2, 16000, row["samples"])
            assert hashlib.sha256(pcm.readframes(pcm.getnframes())).hexdigest() == row["pcm_s16le_sha256"]
        bodies[row["id"]] = dict(model="Qwen/Qwen3-ASR-1.7B", temperature=0, seed=0,
            max_completion_tokens=1024, stream=False, messages=[dict(role="user", content=[
                dict(type="input_audio", input_audio=dict(format="wav", data=base64.b64encode(raw).decode()))])])
    save()
    try:
        with httpx.Client(base_url=args.base_url, timeout=1800) as client:
            models = client.get("/v1/models")
            models.raise_for_status()
            assert [(row["id"], row["max_model_len"]) for row in models.json()["data"]] == [("Qwen/Qwen3-ASR-1.7B", 8192)]
            baseline = {}

            def request(row, observed, *, schedule="baseline", concurrency=1):
                return _request(client, row, bodies[row["id"]], observed, record, save, record_lock,
                    schedule=schedule, concurrency=concurrency)

            def compare(result, actual, *, ids):
                return _compare(result, actual, baseline[result["id"]], save, ids=ids)

            for observed in (False, True):
                for row in cases:
                    result, data = request(row, observed)
                    if observed:
                        compare(result, data, ids=False)
                    baseline[row["id"]] = data
                    record["cases"].append(dict(schedule="baseline", **result))
                    save()
            for concurrency in (1, 2, 4, 8):
                orders = (list(reversed(cases)),) if concurrency == 1 else (cases, list(reversed(cases)))
                for order in orders:
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        results = list(pool.map(lambda row: request(row, True,
                            schedule="concurrent", concurrency=concurrency), order))
                    for result, data in results:
                        compare(result, data, ids=True)
                        record["cases"].append(dict(schedule="concurrent", concurrency=concurrency, **result))
                    save()
            for concurrency, count in ((4, 3), (8, 7)):
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    results = list(pool.map(lambda row: request(row, True,
                        schedule="partial", concurrency=concurrency), cases[:count]))
                for result, data in results:
                    compare(result, data, ids=True)
                    record["cases"].append(dict(schedule="partial", concurrency=concurrency, **result))
                save()
            for addition in (dict(return_token_ids=1), dict(return_token_timestamps=True),
                    dict(return_token_ids=True, return_token_timestamps="true"), dict(timestamps=False)):
                _refuse(client, bodies[cases[0]["id"]], addition, record, save)
            result, data = request(cases[0], True, schedule="after-refusals")
            compare(result, data, ids=True)
            record["cases"].append(dict(schedule="after-refusals", **result))
            assert len(record["cases"]) == len(record["attempts"]) == 83
            record["status"] = "complete_native_raw_token_schedule_diagnostic"
    except BaseException as error:
        record["status"] = "failed_or_interrupted"
        record["errors"].append(dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
