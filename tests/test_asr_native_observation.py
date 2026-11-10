# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Collector failure controls; these do not stand in for native model checks."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from threading import Barrier, RLock

import httpx
import pytest

from qualify.asr_native_observation import _compare, _refuse, _request


def response_data():
    return dict(choices=[dict(finish_reason="stop", message=dict(content="language English<asr_text>hi"),
        token_ids=[151645], token_timestamps_ns=[30])], prompt_token_ids=[1000],
        usage=dict(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=dict(cached_tokens=0)),
        performance=dict(ingress_ns=10, enqueue_ns=20))


def recorder(tmp_path):
    path = tmp_path / "responses.json"
    record = dict(attempts=[])
    lock = RLock()

    def save():
        with lock:
            pending = path.with_suffix(".pending")
            pending.write_text(json.dumps(record))
            pending.replace(path)

    return path, record, lock, save


def test_parallel_failure_keeps_the_failed_raw_response_and_completed_sibling(tmp_path):
    path, record, lock, save = recorder(tmp_path)
    barrier = Barrier(2)

    def respond(request):
        barrier.wait(timeout=5)
        if json.loads(request.content)["control"] == "fail":
            return httpx.Response(502, text="deliberate collector response failure")
        return httpx.Response(200, json=response_data())

    with httpx.Client(base_url="http://collector.test", transport=httpx.MockTransport(respond)) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_request, client, dict(id=identity), dict(control=identity), True,
                record, save, lock, schedule="parallel-control", concurrency=2) for identity in ("fail", "healthy")]
            with pytest.raises(AssertionError):
                futures[0].result(timeout=10)
            futures[1].result(timeout=10)
    saved = json.loads(path.read_text())
    assert len(saved["attempts"]) == 2
    by_id = {row["id"]: row for row in saved["attempts"]}
    assert by_id["fail"]["http_status"] == 502
    assert by_id["fail"]["response_body"] == "deliberate collector response failure"
    assert by_id["fail"]["status"] == "failed_or_interrupted"
    assert by_id["healthy"]["status"] == "response_validated"
    assert json.loads(by_id["healthy"]["response_body"]) == response_data()


def test_transport_failure_is_retained_before_any_response(tmp_path):
    path, record, lock, save = recorder(tmp_path)

    def fail(request):
        raise httpx.ReadError("deliberate interrupted response")

    with httpx.Client(base_url="http://collector.test", transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(httpx.ReadError):
            _request(client, dict(id="interrupted"), {}, True, record, save, lock)
    attempt, = json.loads(path.read_text())["attempts"]
    assert attempt["status"] == "failed_or_interrupted"
    assert attempt["error"]["type"] == "ReadError"
    assert "http_status" not in attempt


def test_comparison_failure_keeps_its_raw_response_and_request_context(tmp_path):
    path, record, lock, save = recorder(tmp_path)
    actual = response_data()
    with httpx.Client(base_url="http://collector.test", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=actual))) as client:
        result, data = _request(client, dict(id="different"), {}, True, record, save, lock,
            schedule="partial", concurrency=4)
    previous = deepcopy(actual)
    previous["choices"][0]["token_ids"] = [151644]
    with pytest.raises(AssertionError, match="token_ids"):
        _compare(result, data, previous, save, ids=True)
    saved, = json.loads(path.read_text())["attempts"]
    assert saved["comparison_failed_field"] == "token_ids"
    assert saved["context"] == dict(schedule="partial", concurrency=4)
    assert json.loads(saved["response_body"]) == actual


def test_natural_stop_without_the_actual_eos_is_rejected_and_preserved(tmp_path):
    path, record, lock, save = recorder(tmp_path)
    wrong = response_data()
    wrong["choices"][0]["token_ids"] = [1000]
    with httpx.Client(base_url="http://collector.test", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=wrong))) as client:
        with pytest.raises(AssertionError):
            _request(client, dict(id="missing-eos"), {}, True, record, save, lock)
    saved, = json.loads(path.read_text())["attempts"]
    assert saved["status"] == "failed_or_interrupted"
    assert json.loads(saved["response_body"]) == wrong


def test_nonempty_audio_chat_cannot_claim_an_empty_prompt(tmp_path):
    path, record, lock, save = recorder(tmp_path)
    wrong = response_data()
    wrong["prompt_token_ids"] = []
    wrong["usage"]["prompt_tokens"] = 0
    with httpx.Client(base_url="http://collector.test", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=wrong))) as client:
        with pytest.raises(AssertionError):
            _request(client, dict(id="empty-prompt"), {}, True, record, save, lock)
    saved, = json.loads(path.read_text())["attempts"]
    assert saved["status"] == "failed_or_interrupted"
    assert json.loads(saved["response_body"]) == wrong


def test_refusal_transport_failure_is_recorded_before_the_post(tmp_path):
    path, record, lock, save = recorder(tmp_path)
    record["refusals"] = []
    addition = dict(return_token_ids=1)

    def fail(request):
        pending, = json.loads(path.read_text())["refusals"]
        assert pending == dict(addition=addition, state="requesting")
        raise httpx.ReadError("deliberate refusal response interruption")

    with httpx.Client(base_url="http://collector.test", transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(httpx.ReadError):
            _refuse(client, {}, addition, record, save)
    attempt, = json.loads(path.read_text())["refusals"]
    assert attempt["state"] == "failed_or_interrupted"
    assert attempt["addition"] == addition
    assert attempt["error"]["type"] == "ReadError"
    assert "status" not in attempt


def test_wrong_refusal_response_is_preserved_before_validation(tmp_path):
    path, record, lock, save = recorder(tmp_path)
    record["refusals"] = []
    with httpx.Client(base_url="http://collector.test", transport=httpx.MockTransport(
            lambda request: httpx.Response(502, text="deliberate refusal failure"))) as client:
        with pytest.raises(AssertionError):
            _refuse(client, {}, dict(timestamps=False), record, save)
    attempt, = json.loads(path.read_text())["refusals"]
    assert attempt["state"] == "failed_or_interrupted"
    assert attempt["status"] == 502
    assert attempt["body"] == "deliberate refusal failure"
