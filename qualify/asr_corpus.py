# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare deterministic ASR qualification/development inputs before inference.

The caller supplies verified official LibriSpeech and AISHELL-1 extractions.
Selection never reads hypotheses: first 16 sorted speakers, eight duration
quantiles per speaker, with utterance ID breaking duration ties.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import wave


def select(rows):
    speakers = defaultdict(list)
    for row in rows:
        speakers[row["speaker"]].append(row)
    if len(speakers) < 16:
        raise ValueError("corpus needs 16 speakers")
    selected = []
    for speaker in sorted(speakers)[:16]:
        candidates = sorted(speakers[speaker], key=lambda row: (row["samples"], row["id"]))
        if len(candidates) < 8:
            raise ValueError(f"speaker {speaker} has fewer than eight utterances")
        selected.extend(candidates[round(i * (len(candidates) - 1) / 7)] for i in range(8))
    if len({row["id"] for row in selected}) != 128:
        raise ValueError("selection has duplicate utterances")
    return selected


def source_samples(path):
    value = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
        "stream=sample_rate,duration_ts,time_base,channels", "-of", "json", str(path)]))
    streams = value["streams"]
    if len(streams) != 1 or streams[0].get("sample_rate") != "16000" or streams[0].get("channels") != 1:
        raise ValueError(f"official source is not mono 16 kHz: {path}")
    stream = streams[0]
    if stream["time_base"] != "1/16000" or stream["duration_ts"] <= 200:
        raise ValueError(f"invalid source sample extent: {path}")
    return stream["duration_ts"]


def english_rows(root):
    rows = []
    speakers = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: int(p.name))[:16]
    for speaker in speakers:
        for transcript in sorted(speaker.rglob("*.trans.txt")):
            for line in transcript.read_text().splitlines():
                identity, text = line.split(" ", 1)
                source = transcript.parent / (identity + ".flac")
                rows.append(dict(id="en-" + identity, language="en", speaker=speaker.name.zfill(8),
                    text=text, source=source, source_id=identity, samples=source_samples(source)))
    return rows


def chinese_rows(root, split):
    transcripts = {}
    for line in (root / "transcript/aishell_transcript_v0.8.txt").read_text().splitlines():
        identity, text = line.split(maxsplit=1)
        if identity in transcripts:
            raise ValueError(f"duplicate AISHELL transcript: {identity}")
        transcripts[identity] = text
    rows = []
    speakers = sorted(p for p in (root / "wav" / split).iterdir() if p.is_dir())[:16]
    for speaker in speakers:
        for source in sorted(speaker.glob("*.wav")):
            identity = source.stem
            rows.append(dict(id="zh-" + identity, language="zh", speaker=speaker.name,
                text=transcripts[identity], source=source, source_id=identity, samples=source_samples(source)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech", type=Path, required=True, help="LibriSpeech root containing test-clean/dev-clean")
    parser.add_argument("--aishell", type=Path, required=True, help="data_aishell root with transcript/ and unpacked wav/")
    parser.add_argument("--split", choices=("qualification", "development"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pcm = args.output / "pcm"
    pcm.mkdir()
    english = "test-clean" if args.split == "qualification" else "dev-clean"
    chinese = "test" if args.split == "qualification" else "dev"
    selected = select(english_rows(args.librispeech / english)) + select(chinese_rows(args.aishell, chinese))
    output = []
    for row in selected:
        target = pcm / (row["id"] + ".wav")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(row["source"]),
            "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)], check=True)
        with wave.open(str(target)) as decoded:
            if (decoded.getnchannels(), decoded.getsampwidth(), decoded.getframerate(), decoded.getnframes()) != (
                    1, 2, 16000, row["samples"]):
                raise ValueError(f"canonical PCM extent differs for {row['id']}")
            samples = decoded.readframes(decoded.getnframes())
        output.append({key: value for key, value in row.items() if key != "source"} | {
            "path": target.relative_to(args.output).as_posix(), "pcm_s16le_sha256": hashlib.sha256(samples).hexdigest()})
    manifest = dict(split=args.split, selection="first 16 numeric English/lexical Chinese speakers; "
        "8 round(i*(n-1)/7) quantiles sorted by exact source samples then ID; i=0..7", utterances=output,
        sources={"en": {"dataset": "OpenSLR/12 LibriSpeech", "split": english,
            "archive_md5": "32fa31d27d2e1cad72775fee3f4849a9" if english == "test-clean" else "42e2234ba48799c1f50f24a7926300a1",
            "license": "CC BY 4.0", "source": "https://www.openslr.org/12/"},
            "zh": {"dataset": "OpenSLR/33 AISHELL-1", "split": chinese,
                "archive_md5": "2f494334227864a8a8fec932999db9d8", "license": "Apache-2.0",
                "source": "https://www.openslr.org/33/"}},
        decode={"program": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            "format": "mono 16000 Hz PCM16 WAV; source has same channels/rate; no duration trimming"},
        request={"temperature": 0, "seed": 0, "max_model_len": 8192,
            "context": "", "language": "utterance.language", "max_completion_tokens": 1024})
    with (args.output / "manifest.json").open("x") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
