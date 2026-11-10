# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared attention errors cross exported graphs, the scheduler and HTTP."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an assigned GPU and FLARE_RUN_CUDA_TESTS=1")
ROOT = Path(__file__).resolve().parents[2]


def metrics(client):
    result = {}
    response = client.get("/metrics")
    response.raise_for_status()
    for line in response.text.splitlines():
        if line.startswith("flare_") and "{" not in line:
            name, value = line.split()
            result[name] = float(value)
    return result


def idle(client):
    deadline = time.monotonic() + 20
    while True:
        values = metrics(client)
        if values["flare_active_requests"] == values["flare_retiring_request_slots"] == 0:
            assert values["flare_free_request_slots"] == 2
            return values
        assert time.monotonic() < deadline, values
        time.sleep(.05)


def test_native_errors_do_not_publish_failed_state_and_recover_slots(tmp_path):
    binary = Path(os.environ.get("FLARE_TEST_FLARE_BINARY", ROOT / "flare/target/release/flare"))
    assert binary.is_file(), "build Flare or set FLARE_TEST_FLARE_BINARY to the tested native build"
    package_root = tmp_path / "prepared"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    with (tmp_path / "prepare.log").open("w") as log:
        preparation = subprocess.run([sys.executable, str(ROOT / "helios/tests/generation_error_package_fixture.py"),
            str(package_root)], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    assert preparation.returncode == 0, (tmp_path / "prepare.log").read_text()
    # The preparation process has exited before the Rust server loads anything.
    from helios.tests.signing import publisher
    signer = publisher()
    profile = tmp_path / "publisher.json"
    profile.write_text(json.dumps([dict(name="Flare public test fixture", namespace=signer.namespace,
        public_key=list(signer.public_key), development=False)]))
    config = tmp_path / "config.toml"
    config.write_text("")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = None
    with (tmp_path / "server.log").open("w") as log:
        try:
            server = subprocess.Popen([str(binary), "run", str(package_root / "package"), "--config", str(config),
                "--publisher-keys", str(profile), "--trust-store", str(tmp_path / "trust.toml"),
                "--offline", "--dangerously-approve-key", signer.fingerprint.hex(),
                "--dangerously-accept-publisher", signer.namespace, "--verify-weights",
                "--host", "127.0.0.1", "--port", str(port), "--gpus", "0", "--max-concurrency", "2",
                "--embedding-table-storage", "ram", "--kv-cache-limit", "concurrency",
                "--prefix-cache-memory", "0", "--prefix-cache-disk-limit", "0"],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
                deadline = time.monotonic() + 180
                while True:
                    assert server.poll() is None, (tmp_path / "server.log").read_text()
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    assert time.monotonic() < deadline, (tmp_path / "server.log").read_text()
                    time.sleep(.1)
                mappings = Path(f"/proc/{server.pid}/maps").read_text().lower()
                assert "libpython" not in mappings and "libtorch" not in mappings
                def request(prompt, *, stream=False):
                    return client.post("/v1/completions", json=dict(model="generation-error-fixture",
                        prompt=prompt, max_tokens=4, seed=0, stream=stream))
                def error_response(response, streaming):
                    if streaming:
                        assert response.status_code == 200, response.text
                        data = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
                        assert "[DONE]" not in data
                        failures = [json.loads(item)["error"] for item in data if "error" in json.loads(item)]
                        assert len(failures) == 1, response.text
                        error = failures[0]
                    else:
                        assert response.status_code == 500 and "error" in response.json(), response.text
                        error = response.json()["error"]
                    assert error["message"] == "prepared model reported device error 1", error
                observations = []
                before = idle(client)
                assert before["flare_snapshot_capacity"] >= 4, before
                for streaming in (False, True):
                    response = request([3] + [2] * 128, stream=streaming)
                    error_response(response, streaming)
                    after = idle(client)
                    assert after["flare_snapshot_publications_total"] == before["flare_snapshot_publications_total"]
                    assert after["flare_requests_failed_total"] == before["flare_requests_failed_total"] + 1
                    for name in ("flare_free_kv_pages", "flare_free_snapshots"):
                        assert after[name] == before[name], (name, before, after)
                    observations.append(dict(kind="nonsampling-prefill", stream=streaming, status=response.status_code,
                        body=response.text, before=before, after=after))
                    before = after
                for repeat, streaming in enumerate((False, False, True)):
                    response = request([4] * 13, stream=streaming)
                    error_response(response, streaming)
                    after = idle(client)
                    assert after["flare_requests_failed_total"] == before["flare_requests_failed_total"] + 1
                    if repeat == 0:
                        # The successful prefill may retain its valid prompt,
                        # while the failed decode must not publish a terminal.
                        assert after["flare_snapshot_publications_total"] == before["flare_snapshot_publications_total"] + 1
                    else:
                        assert after["flare_snapshot_publications_total"] == before["flare_snapshot_publications_total"]
                        assert after["flare_cache_reused_tokens_total"] > before["flare_cache_reused_tokens_total"]
                        for name in ("flare_free_kv_pages", "flare_free_snapshots"):
                            assert after[name] == before[name], (name, before, after)
                    observations.append(dict(kind="decode", repeat=repeat, status=response.status_code,
                        body=response.text, before=before, after=after))
                    before = after
                healthy = request([2] * 13)
                assert healthy.status_code == 200, healthy.text
                assert healthy.json()["usage"]["completion_tokens"] == 4
                idle(client)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    responses = list(pool.map(request, ([4] * 13, [2] * 13)))
                assert [response.status_code for response in responses] == [500, 200]
                assert responses[1].json()["choices"] == healthy.json()["choices"]
                final = idle(client)
                assert client.get("/health").status_code == 200
                observations.append(dict(kind="healthy-sibling-and-slot-reuse", metrics=final))
                (tmp_path / "observations.json").write_text(json.dumps(observations, indent=2) + "\n")
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)
                    pytest.fail("native error fixture did not drain after SIGTERM")
        assert server is not None and server.returncode == 0, (tmp_path / "server.log").read_text()
