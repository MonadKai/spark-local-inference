# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Real FFmpeg decode rejection controls through the native probe binary."""

import os
from pathlib import Path
import shutil
import struct
import subprocess

import pytest


@pytest.fixture(scope="module")
def binary():
    explicit = os.environ.get("FLARE_TEST_AUDIO_BINARY")
    path = Path(explicit) if explicit else Path(__file__).resolve().parents[2] / "helios/target/debug/test_audio"
    if not path.is_file():
        if explicit:
            pytest.fail(f"explicit native audio binary does not exist: {path}")
        pytest.skip("build test_audio with audio,test-tools")
    return path.resolve()


def _decode(binary, tmp_path, name, payload):
    source, output = tmp_path / (name + ".audio"), tmp_path / (name + ".f32le")
    source.write_bytes(payload)
    result = subprocess.run([str(binary), "decode", str(source), str(output)],
                            capture_output=True, text=True, timeout=30)
    return result, output


def test_speech_transport_fixtures_decode_with_separate_waveforms(binary, tmp_path):
    import numpy as np
    import wave

    root = Path(__file__).with_name("audio-inputs")
    wav = root / "en-61-70968-0058.wav"
    with wave.open(str(wav), "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000)
        expected = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
    decoded = []
    for suffix in ("wav", "mp3"):
        result, output = _decode(binary, tmp_path, suffix, wav.with_suffix("." + suffix).read_bytes())
        assert result.returncode == 0, result.stderr
        values = np.fromfile(output, dtype="<f4")
        assert values.size == 29120 and np.isfinite(values).all()
        decoded.append(values)
    assert np.array_equal(decoded[0], expected)
    assert not np.array_equal(decoded[0], decoded[1]), "lossy MP3 was silently treated as identical PCM"


def _aac(rate, channels):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg CLI is needed only to encode the AAC regression fixture")
    return subprocess.check_output([ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        f"sine=frequency=440:sample_rate={rate}:duration=0.4", "-ac", str(channels),
        "-c:a", "aac", "-f", "adts", "pipe:1"], timeout=30)


@pytest.mark.parametrize("first,second", [((16000, 2), (16000, 1)),
    ((16000, 1), (16000, 2)), ((16000, 1), (48000, 1))])
def test_reject_midstream_geometry_change_without_crashing(binary, tmp_path, first, second):
    initial, following = _aac(*first), _aac(*second)
    # Both individual source geometries are valid; their in-stream transition
    # must not reach a resampler configured with stale pointers or sample rate.
    for index, payload in enumerate((initial, following)):
        result, output = _decode(binary, tmp_path, f"valid-{index}", payload)
        assert result.returncode == 0, result.stderr
        assert output.stat().st_size > 200 * 4
    result, output = _decode(binary, tmp_path, "changed", initial + following)
    assert result.returncode == 1, result.stderr
    assert not output.exists()


def _float_wav(declared, actual, value=0.25):
    return (b"RIFF" + struct.pack("<I", 36 + declared * 4) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 3, 1, 16000, 64000, 4, 32)
        + b"data" + struct.pack("<I", declared * 4) + struct.pack("<f", value) * actual)


@pytest.mark.parametrize("signature", [b"RF64", b"BW64"])
def test_extended_wav_rejects_truncated_data(binary, tmp_path, signature):
    def payload(actual):
        return (signature + struct.pack("<I", 0xFFFFFFFF) + b"WAVEds64"
            + struct.pack("<IQQQI", 28, 72 + 2048 * 4, 2048 * 4, 2048, 0)
            + b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, 16000, 64000, 4, 32)
            + b"data" + struct.pack("<I", 0xFFFFFFFF) + struct.pack("<f", 0.25) * actual)
    result, output = _decode(binary, tmp_path, "complete", payload(2048))
    assert result.returncode == 0, result.stderr
    assert output.stat().st_size == 2048 * 4
    result, output = _decode(binary, tmp_path, "truncated", payload(1024))
    assert result.returncode == 1, result.stderr
    assert not output.exists()


@pytest.mark.parametrize("declared,actual", [(1600, 400), (2048, 1024), (4096, 2048), (8192, 4096)])
@pytest.mark.parametrize("outer_length", ["declared", "actual", "empty", "fmt-only"])
def test_truncated_packet_cannot_be_successful_partial_audio(binary, tmp_path, declared, actual, outer_length):
    result, output = _decode(binary, tmp_path, "complete", _float_wav(declared, declared))
    assert result.returncode == 0, result.stderr
    assert output.stat().st_size == declared * 4
    payload = _float_wav(declared, actual)
    if outer_length != "declared":
        # A repaired outer length must not conceal an oversized inner chunk.
        length = {"actual": len(payload) - 8, "empty": 4, "fmt-only": 28}[outer_length]
        payload = payload[:4] + struct.pack("<I", length) + payload[8:]
    result, output = _decode(binary, tmp_path, "truncated", payload)
    assert result.returncode == 1, result.stderr
    assert not output.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_pcm_is_rejected(binary, tmp_path, value):
    result, output = _decode(binary, tmp_path, "nonfinite", _float_wav(1600, 1600, value))
    assert result.returncode == 1
    assert "nonfinite" in result.stderr
    assert not output.exists()
