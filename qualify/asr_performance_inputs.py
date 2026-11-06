# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Freeze distinct development speech before ASR performance experiments.

This does not run a recognizer. Warmup and quality utterances, including equal
PCM under different source names, are excluded before content ordering.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import wave


def order(rows):
    """Alternate languages; ordering depends only on the fixed ID salt."""
    groups = {language: sorted((row for row in rows if row["language"] == language),
        key=lambda row: hashlib.sha256(("asr-performance-20261106:" + row["id"]).encode()).digest())
        for language in ("en", "zh")}
    result = []
    for index in range(max(map(len, groups.values()), default=0)):
        for language in ("en", "zh"):
            if index < len(groups[language]):
                result.append(groups[language][index])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", type=Path, required=True, help="complete official dev-clean directory")
    parser.add_argument("--chinese", type=Path, required=True, help="acquired official dev speaker directories")
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("decoder workers must be in [1,8]")
    development = json.loads(args.development_manifest.read_bytes())
    qualification = json.loads(args.qualification_manifest.read_bytes())
    if development["split"] != "development" or qualification["split"] != "qualification":
        raise ValueError("independent development and qualification manifests required")
    excluded = development["utterances"] + qualification["utterances"]
    excluded_ids = {row["id"] for row in excluded}
    excluded_pcm = {row["pcm_s16le_sha256"] for row in excluded}
    candidates = []
    for transcript in sorted(args.english.rglob("*.trans.txt")):
        for line in transcript.read_text().splitlines():
            identity, text = line.split(" ", 1)
            candidates.append(dict(id="en-" + identity, language="en", text=text,
                speaker=identity.split("-")[0], source=transcript.parent / (identity + ".flac")))
    labels = {}
    for line in args.transcript.read_text().splitlines():
        identity, text = line.split(maxsplit=1)
        if identity in labels:
            raise ValueError("duplicate Chinese annotation")
        labels[identity] = text
    missing = []
    for source in sorted(args.chinese.glob("*/*.wav")):
        if source.stem not in labels:
            missing.append(source.stem)
        else:
            candidates.append(dict(id="zh-" + source.stem, language="zh", text=labels[source.stem],
                speaker=source.parent.name, source=source))
    if len({row["id"] for row in candidates}) != len(candidates) or not candidates:
        raise ValueError("empty or duplicate source candidate IDs")
    if not {row["id"] for row in development["utterances"]} <= {row["id"] for row in candidates}:
        raise ValueError("performance source pool does not contain its declared warmup development split")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "pcm").mkdir()

    def decode(row):
        target = args.output / "pcm" / (row["id"] + ".wav")
        metadata = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
            "stream=sample_rate,duration_ts,time_base,channels", "-of", "json", str(row["source"])], timeout=30))
        streams = metadata["streams"]
        if len(streams) != 1 or streams[0].get("sample_rate") != "16000" or streams[0].get("channels") != 1 \
                or streams[0].get("time_base") != "1/16000" or not 200 < streams[0]["duration_ts"] <= 600 * 16000:
            raise ValueError(f"source geometry differs: {row['id']}")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-n", "-threads", "1",
            "-i", str(row["source"]), "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-threads", "1", "-fs", str(600 * 16000 * 2 + 65536), str(target)], check=True, timeout=60)
        with wave.open(str(target)) as source:
            samples = source.getnframes()
            if (source.getnchannels(), source.getsampwidth(), source.getframerate(), samples) != \
                    (1, 2, 16000, streams[0]["duration_ts"]):
                raise ValueError(f"decoded source extent differs: {row['id']}")
            digest = hashlib.sha256(source.readframes(samples)).hexdigest()
        return {key: value for key, value in row.items() if key != "source"} | dict(
            path=target.relative_to(args.output).as_posix(), samples=samples, pcm_s16le_sha256=digest)

    selected, duplicates = [], []
    seen = set(excluded_pcm)
    candidates = sorted((row for row in candidates if row["id"] not in excluded_ids), key=lambda row: row["id"])
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(decode, candidates):
            if row["pcm_s16le_sha256"] in seen:
                duplicates.append(row["id"])
            else:
                seen.add(row["pcm_s16le_sha256"])
                selected.append(row)
    sources = development["sources"]
    sources["en"]["acquisition"]["candidates"] = "complete official dev-clean; all numeric speakers before fixed warmup/qualification exclusions"
    result = dict(split="performance-development", selection=__doc__.strip(),
        ordering="SHA256(asr-performance-20261106: + ID), per-language ascending digest, alternate en/zh until exhausted",
        duplicate_policy="lexical ID first; exclude every PCM matching warmup/qualification or an earlier candidate",
        excluded_without_reference={"zh": missing}, excluded_pcm_duplicates=duplicates,
        warmup_manifest_sha256=hashlib.sha256(args.development_manifest.read_bytes()).hexdigest(),
        qualification_manifest_sha256=hashlib.sha256(args.qualification_manifest.read_bytes()).hexdigest(),
        sources=sources,
        english_candidate_scope="all numeric speakers in complete official dev-clean, not only the quality selection's 16",
        chinese_candidate_scope="same acquired official dev speaker subset as the development selection",
        decode=dict(program=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            format="mono 16000 Hz PCM16 WAV; source same channels/rate; no trimming"),
        request=development["request"], utterances=order(selected))
    (args.output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(requests=len(selected), audio_seconds=sum(row["samples"] for row in selected) / 16000,
                         excluded_pcm_duplicates=len(duplicates)), ensure_ascii=False))


if __name__ == "__main__":
    main()
