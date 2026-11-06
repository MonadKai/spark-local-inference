# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import random

import pytest

from qualify.asr_corpus import select


def _rows():
    return [dict(id=f"{speaker:08}-{utterance:03}", speaker=f"{speaker:08}", samples=utterance * 160)
            for speaker in range(20) for utterance in range(22)]


def test_selection_is_independent_of_enumeration_and_uses_exact_duration_quantiles():
    rows = _rows()
    random.Random(9).shuffle(rows)
    selected = select(rows)
    assert len(selected) == 128
    assert {row["speaker"] for row in selected} == {f"{speaker:08}" for speaker in range(16)}
    assert [row["id"] for row in selected[:8]] == [f"00000000-{position:03}" for position in range(0, 22, 3)]
    assert selected == select(list(reversed(rows)))


def test_insufficient_first_speaker_cannot_be_replaced_after_selection():
    rows = [row for row in _rows() if row["speaker"] != "00000000" or row["samples"] < 7 * 160]
    with pytest.raises(ValueError, match="fewer than eight"):
        select(rows)


def test_duplicate_selected_utterance_is_not_silently_replaced():
    rows = _rows()
    rows[0]["id"] = rows[3]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        select(rows)


def test_duplicate_unselected_candidate_cannot_change_quantile_positions():
    rows = _rows()
    duplicate = rows[1].copy()
    assert duplicate["id"] not in {row["id"] for row in select(rows)}
    with pytest.raises(ValueError, match="duplicate candidate"):
        select(rows + [duplicate])


def test_source_sample_count_decodes_before_accepting_duration_metadata(tmp_path):
    import shutil
    import subprocess
    import wave
    from qualify.asr_corpus import source_samples

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("native corpus preparation requires FFmpeg and ffprobe")
    wav = tmp_path / "original.wav"
    with wave.open(str(wav), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x10" * 16001)
    assert source_samples(wav) == 16001
    flac = tmp_path / "source.flac"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(wav), str(flac)], check=True)
    assert source_samples(flac) == 16001
    malformed = bytearray(flac.read_bytes())
    assert malformed[:4] == b"fLaC"
    packed = int.from_bytes(malformed[18:26], "big")
    assert packed & ((1 << 36) - 1) == 16001
    malformed[18:26] = ((packed & ~((1 << 36) - 1)) | 32002).to_bytes(8, "big")
    bad = tmp_path / "wrong-total-samples.flac"
    bad.write_bytes(malformed)
    # A metadata-only probe accepts this doubled duration. Full decoding must
    # reject it before it can alter the qualification quantile selection.
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        source_samples(bad)


def test_unlabeled_aishell_files_are_declared_before_quantile_selection(tmp_path):
    import shutil
    import wave
    from qualify.asr_corpus import chinese_rows

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("native corpus preparation requires FFmpeg and ffprobe")
    transcript = tmp_path / "transcript"
    transcript.mkdir()
    (transcript / "aishell_transcript_v0.8.txt").write_text("labeled 有 标 注\n")
    speaker = tmp_path / "wav/dev/S0724"
    speaker.mkdir(parents=True)
    for name in ["labeled", "unlabeled"]:
        with wave.open(str(speaker / (name + ".wav")), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b"\x00\x00" * 1600)
    rows, excluded = chinese_rows(tmp_path, "dev")
    assert [row["source_id"] for row in rows] == ["labeled"]
    assert excluded == ["unlabeled"]
