# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Authenticated CPU frontend plans; fixture graphs are never loaded or executed."""

import hashlib
import json
import subprocess
import wave

import numpy as np
import pytest

from helios.tests.signing import publisher
from ignition.preparation import runtime_package_pb2 as proto
from ignition.preparation.package_file import combine
from tests.frontend.test_audio_decode import binary
from tests.preparation.payload_fixture import manifest, rank_source, schema
from tests.preparation.test_package_protection import profiles


@pytest.fixture(params=[proto.AUDIO_SAMPLE_PADDING_NONE, proto.AUDIO_SAMPLE_PADDING_HOP_MULTIPLE])
def package(tmp_path, request):
    index = proto.ModelIndex(version=4)
    settings = index.frontend.audio
    settings.sample_rate, settings.mel_bins, settings.fft_size, settings.hop_samples = 16000, 128, 400, 160
    settings.max_file_bytes, settings.max_samples, settings.decode_timeout_millis = 25 << 20, 600 * 16000, 60000
    settings.service_chunk_samples, settings.silence_search_samples, settings.silence_window_samples = 480000, 16000, 1600
    settings.sample_padding = request.param
    index.contract.model_contract.version = 3
    capacity = index.contract.model_contract.audio
    capacity.feature_slots, capacity.features_per_request, capacity.feature_width = 2, 7800, 2048
    capacity.frame_capacities[:] = [100, 800]
    capacity.mel_bins, capacity.frames_per_chunk, capacity.features_per_chunk = 128, 100, 13
    capacity.subsampling_factor, capacity.attention_window_frames = 8, 800
    for name in ("features", "generations", "counts"):
        view = getattr(capacity, name)
        view.buffer, view.stride, view.nbytes = name, 1, 1
    source = rank_source(tmp_path / "source", index=index)
    destination = tmp_path / "package"
    signer = publisher()
    combine(destination, [source], [], publisher=signer)
    return destination, profiles(tmp_path / "publishers.json", signer), request.param


def source(tmp_path, name, samples):
    path = tmp_path / (name + ".wav")
    values = np.zeros(samples, dtype="<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(values.tobytes())
    decoded = values.astype("<f4") / 32768
    return dict(id=name, path=str(path), samples=samples,
        pcm_f32le_sha256=hashlib.sha256(decoded.tobytes()).hexdigest())


def run(binary, tmp_path, package, inputs, *, tp="1", namespace="flare-tests"):
    request, output = tmp_path / "inputs.json", tmp_path / "plans.jsonl"
    request.write_text(json.dumps(inputs))
    result = subprocess.run([str(binary), "plan", str(package[0]), str(package[1]), tp,
        str(request), str(output), namespace], capture_output=True, text=True, timeout=90)
    rows = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    return result, rows


def test_real_service_chunks_and_encoder_windows_are_distinct(binary, package, tmp_path):
    inputs = [source(tmp_path, "nonhop", 480001), source(tmp_path, "exact", 480000),
        source(tmp_path, "short", 301)]
    result, rows = run(binary, tmp_path, package, inputs)
    assert result.returncode == 0, result.stderr
    header, *cases, summary = rows
    assert header["settings"]["sample_padding"] == package[2] and header["tensor_parallel"] == 1
    assert len(header["envelope_sha256"]) == 64 and header["capacity"]["features"]["buffer"] == "features"
    assert summary == dict(kind="summary", inputs=3, failed=0)
    assert [row["id"] for row in cases] == [row["id"] for row in inputs]
    assert [row["ordinal"] for row in cases] == [0, 1, 2]
    first, exact, short = [row["chunks"] for row in cases]
    # Equal-energy silence chooses the earliest reference split candidate.
    assert [row["samples"] for row in first] == [[0, 464000], [464000, 480001]]
    padded = package[2] == proto.AUDIO_SAMPLE_PADDING_HOP_MULTIPLE
    assert [row["frames"] for row in first] == [2900, 101 if padded else 100]
    assert [len(row["windows"]) for row in first] == [4, 1]
    assert len(exact) == 1 and len(exact[0]["windows"]) == 4
    assert short[0]["frames"] == (2 if padded else 1) and short[0]["features"] == 1
    for case, frozen in zip(cases, inputs, strict=True):
        assert case["pcm_f32le_sha256"] == frozen["pcm_f32le_sha256"]
        for chunk in case["chunks"]:
            assert chunk["windows"][0]["frames"][0] == chunk["windows"][0]["features"][0] == 0
            assert chunk["windows"][-1]["frames"][1] == chunk["frames"]
            assert chunk["windows"][-1]["features"][1] == chunk["features"]


@pytest.mark.parametrize("defect", ["hash", "samples", "invalid-audio"])
def test_failed_input_retained_and_following_input_attempted(binary, package, tmp_path, defect):
    bad, good = source(tmp_path, "bad", 1600), source(tmp_path, "good", 1601)
    if defect == "hash":
        bad["pcm_f32le_sha256"] = "a" * 64
    elif defect == "samples":
        bad["samples"] += 1
    else:
        (tmp_path / "bad.wav").write_bytes(b"invalid audio")
    result, rows = run(binary, tmp_path, package, [bad, good])
    assert result.returncode == 1
    assert rows[1]["status"] == "failed" and rows[1]["ordinal"] == 0
    assert rows[2]["status"] == "passed" and rows[2]["ordinal"] == 1
    assert rows[-1] == dict(kind="summary", inputs=2, failed=1)


@pytest.mark.parametrize("defect", ["namespace", "tp", "tampered-signature"])
def test_invalid_plan_identity_rejected_before_input_execution(binary, package, tmp_path, defect):
    inputs = [source(tmp_path, "input", 1600)]
    options = {}
    if defect == "namespace":
        options["namespace"] = "another-publisher"
    elif defect == "tp":
        options["tp"] = "2"
    else:
        from safetensors.numpy import load_file, save_file

        primary = package[0] / "model.safetensors"
        envelope = manifest(primary)
        body = schema()("PackageBody").FromString(envelope.package_body)
        body.frontend.audio.sample_rate = 8000
        envelope.package_body = body.SerializeToString(deterministic=True)
        tensors = load_file(primary)
        tensors["flare.manifest"] = np.frombuffer(envelope.SerializeToString(deterministic=True), dtype=np.uint8)
        replacement = primary.with_suffix(".tampered")
        save_file(tensors, replacement, metadata={"format": "flare"})
        replacement.replace(primary)
    result, rows = run(binary, tmp_path, package, inputs, **options)
    assert result.returncode == 1 and rows == []
    assert result.stderr.strip()


@pytest.mark.parametrize("defect", ["duplicate", "empty-id", "invalid-hash", "zero-samples"])
def test_invalid_manifest_rows_retained_with_independent_good_input(binary, package, tmp_path, defect):
    bad, good = source(tmp_path, "bad", 1600), source(tmp_path, "good", 1601)
    bad["path"] = str(tmp_path / "must-not-open")
    if defect == "empty-id":
        bad["id"] = ""
    elif defect == "invalid-hash":
        bad["pcm_f32le_sha256"] = "F" * 64
    elif defect == "zero-samples":
        bad["samples"] = 0
    inputs = [bad, bad.copy(), good] if defect == "duplicate" else [bad, good]
    result, rows = run(binary, tmp_path, package, inputs)
    assert result.returncode == 1
    assert rows[-2]["id"] == "good" and rows[-2]["status"] == "passed"
    failed = rows[1:-2]
    assert len(failed) == len(inputs) - 1
    assert all(row["status"] == "failed" and "invalid or duplicate identities" in row["error"] for row in failed)
    assert rows[-1] == dict(kind="summary", inputs=len(inputs), failed=len(failed))


@pytest.mark.parametrize("defect", ["missing-field", "wrong-type", "unknown-field"])
def test_manifest_schema_errors_reject_globally_before_execution(binary, package, tmp_path, defect):
    bad, good = source(tmp_path, "bad", 1600), source(tmp_path, "good", 1601)
    if defect == "missing-field":
        del bad["path"]
    elif defect == "wrong-type":
        bad["samples"] = "1600"
    else:
        bad["unknown"] = True
    result, rows = run(binary, tmp_path, package, [bad, good])
    assert result.returncode == 1 and rows == [] and result.stderr.strip()
