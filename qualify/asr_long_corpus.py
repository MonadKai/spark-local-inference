# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Freeze continuous annotated meeting crops without looking at ASR output."""

import argparse
from collections import Counter
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import wave
import xml.etree.ElementTree as ET
import zipfile

RATE = 16000
NITE = "{http://nite.sourceforge.net/}"


def interval(start, end, speaker, identity, text, **extra):
    start, end = Decimal(start), Decimal(end)
    if not start.is_finite() or not end.is_finite() or start < 0 or end < start:
        raise ValueError("invalid annotation interval")
    return dict(start=start, end=end, speaker=speaker, id=identity, text=text, **extra)


def ami_annotations(archive, recording):
    protected, lexical, gaps = [], [], []
    with zipfile.ZipFile(archive) as source:
        names = sorted(name for name in source.namelist()
                       if name.startswith("words/" + recording + ".") and name.endswith(".words.xml"))
        if len(names) != 4:
            raise ValueError("AMI recording needs all four speaker annotation files")
        for name in names:
            speaker = Path(name).name.split(".")[1]
            words = list(ET.fromstring(source.read(name)))
            ids = {word.attrib[NITE + "id"]: index for index, word in enumerate(words)}
            if len(ids) != len(words):
                raise ValueError("duplicate AMI annotation IDs")
            references = Counter()
            segments = ET.fromstring(source.read(f"segments/{recording}.{speaker}.segments.xml"))
            for segment in segments:
                span = interval(segment.attrib["transcriber_start"], segment.attrib["transcriber_end"],
                                speaker, segment.attrib[NITE + "id"], "")
                protected.append(span)
                for child in segment:
                    href = child.attrib["href"]
                    match = re.fullmatch(r"([^#]+)#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?", href)
                    if not match or match[1] != Path(name).name:
                        raise ValueError("unexpected AMI word reference")
                    first, last = ids[match[2]], ids[match[3] or match[2]]
                    if first > last:
                        raise ValueError("reversed AMI word reference")
                    for word in words[first:last + 1]:
                        identity = word.attrib[NITE + "id"]
                        references[identity] += 1
                        if word.tag != "w" or "starttime" not in word.attrib or "endtime" not in word.attrib:
                            gaps.append(dict(speaker=speaker, segment_id=span["id"], id=identity,
                                             tag=word.tag, text=word.text or "", attributes=word.attrib))
                            continue
                        item = interval(word.attrib["starttime"], word.attrib["endtime"], speaker,
                                        identity, word.text or "", attributes={
                                            key: value for key, value in word.attrib.items()
                                            if key not in (NITE + "id", "starttime", "endtime")})
                        if not span["start"] <= item["start"] <= item["end"] <= span["end"]:
                            raise ValueError("AMI word escapes its containing original segment")
                        lexical.append(item)
            if any(references[word.attrib[NITE + "id"]] != 1 for word in words
                   if word.tag == "w" and "starttime" in word.attrib and "endtime" in word.attrib):
                raise ValueError("AMI timed word is missing or multiply referenced")
    return protected, lexical, gaps


def aishell_annotations(path, markers):
    source = path.read_text()
    tiers = re.split(r"(?m)^\s*item \[\d+\]:\s*$", source)[1:]
    if len(tiers) != 6:
        raise ValueError("AISHELL-4 recording needs six original speaker tiers")
    protected, lexical, speakers = [], [], set()
    for tier in tiers:
        if not re.search(r'class = "IntervalTier"', tier):
            raise ValueError("unsupported TextGrid tier")
        speaker = re.search(r'name = "((?:[^"]|"")*)"', tier)[1].replace('""', '"')
        if speaker in speakers:
            raise ValueError("duplicate TextGrid speaker tier")
        speakers.add(speaker)
        extent = re.search(r'xmin = (\S+)\s*xmax = (\S+)', tier)
        if not extent or Decimal(extent[1]) != 0:
            raise ValueError("invalid TextGrid tier extent")
        expected = int(re.search(r"intervals: size = (\d+)", tier)[1])
        rows = re.findall(r'intervals \[(\d+)\]:\s*xmin = (\S+)\s*xmax = (\S+)\s*text = "((?:[^"]|"")*)"', tier)
        if len(rows) != expected or [int(row[0]) for row in rows] != list(range(1, expected + 1)):
            raise ValueError("incomplete TextGrid interval list")
        previous = Decimal(0)
        for identity, start, end, text in rows:
            text = text.replace('""', '"')
            span = interval(start, end, speaker, speaker + ":" + identity, text)
            if span["start"] != previous:
                raise ValueError("TextGrid tier is not continuous")
            previous = span["end"]
            if not text.strip():
                continue
            protected.append(span)
            found = {marker: text.count(marker) for marker in markers if marker in text}
            cleaned = text
            for marker in markers:
                cleaned = cleaned.replace(marker, "")
            lexical.append(span | {"text": cleaned, "original_text": text, "markers": found})
        if previous != Decimal(extent[2]):
            raise ValueError("incomplete TextGrid tier extent")
    return protected, lexical, []


def check_rttm(path, words, recording):
    expected = sorted((row["speaker"], row["start"], row["end"] - row["start"])
                      for row in words if row["text"].strip())
    actual = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 10 or fields[:3] != ["SPEAKER", recording, "1"]:
            raise ValueError("unexpected RTTM row")
        start, duration = Decimal(fields[3]), Decimal(fields[4])
        if not start.is_finite() or not duration.is_finite() or start < 0 or duration < 0:
            raise ValueError("invalid RTTM interval")
        actual.append((fields[7], start, duration))
    actual.sort()
    if len(actual) != len(expected) or any(a[0] != e[0] or abs(a[1] - e[1]) > Decimal(".0001")
            or abs(a[2] - e[2]) > Decimal(".0001") for a, e in zip(actual, expected)):
        raise ValueError("RTTM does not match the original precision TextGrid")
    return dict(rows=len(actual), maximum_rounding_error_seconds=str(max(
        (max(abs(a[1] - e[1]), abs(a[2] - e[2])) for a, e in zip(actual, expected)), default=Decimal(0))))


def safe_boundary(protected, anchor, maximum):
    anchor = int(anchor * RATE)
    candidates = set()
    for row in protected:
        for time in (row["start"], row["end"]):
            scaled = time * RATE
            candidates.update(int(scaled.to_integral_value(rounding=rule)) for rule in (ROUND_FLOOR, ROUND_CEILING))
    for sample in sorted(candidates):
        if not anchor <= sample <= min(anchor + 15 * RATE, maximum):
            continue
        time = Decimal(sample) / RATE
        if not any(row["start"] < time < row["end"] for row in protected):
            return sample
    raise ValueError("no safe original segment boundary within the frozen 15-second window")


def decode_source(path, channels, channel):
    metadata = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
        "stream=sample_rate,duration_ts,time_base,channels", "-of", "json", str(path)], timeout=30))
    streams = metadata["streams"]
    if len(streams) != 1:
        raise ValueError("meeting source needs exactly one audio stream")
    stream = streams[0]
    if (stream["sample_rate"], stream["time_base"], stream["channels"]) != ("16000", "1/16000", channels):
        raise ValueError("meeting source geometry changed")
    samples = stream["duration_ts"]
    if not 0 < samples <= 3 * 3600 * RATE:
        raise ValueError("meeting source duration is outside acquisition capacity")
    with tempfile.TemporaryFile() as output:
        command = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-i", str(path), "-map", "0:a:0"]
        if channels != 1:
            command += ["-af", f"pan=mono|c0=c{channel}"]
        command += ["-c:a", "pcm_s16le", "-threads", "1", "-f", "s16le", "-fs", str((samples + 1) * 2), "pipe:1"]
        subprocess.run(command, stdout=output, check=True, timeout=600)
        if output.tell() != samples * 2:
            raise ValueError("full decoded meeting extent differs from metadata")
        output.seek(0)
        return output.read()


def write_crop(path, pcm):
    with wave.open(str(path), "wb") as target:
        target.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        target.writeframes(pcm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "pcm").mkdir()
    result = dict(scope=config["scope"], configuration=config, fixed_before_recognizer_execution=True,
                  request=config["request"],
                  decode=dict(program=subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=30).splitlines()[0],
                              format="mono 16000 Hz PCM16 WAV; full source decode before sample-index cropping; AISHELL channel0 extraction without mixing"),
                  sources=[], utterances=[], acquisition_failures=[], regression_inputs=[])
    for family in ("ami", "aishell4"):
        for recording in config[family]["recordings"]:
            identity = recording["id"]
            audio = args.source_dir / (identity + (".Mix-Headset.wav" if family == "ami" else ".flac"))
            if family == "ami":
                annotation = args.source_dir / "ami_public_manual_1.6.2.zip"
                protected, words, gaps = ami_annotations(annotation, identity)
            else:
                annotation = args.source_dir / (identity + ".TextGrid")
                protected, words, gaps = aishell_annotations(annotation, config[family]["reference_markers"]["markers"])
                rttm = check_rttm(args.source_dir / (identity + ".rttm"), words, identity)
            pcm = decode_source(audio, 1 if family == "ami" else 8, config[family].get("channel", 0))
            count = len(pcm) // 2
            if any(row["end"] > Decimal(count) / RATE for row in protected):
                raise ValueError("annotation escapes decoded meeting")
            result["sources"].append(dict(recording=identity, dataset=family,
                audio_file_sha256=hashlib.file_digest(audio.open("rb"), "sha256").hexdigest(),
                annotation_file_sha256=hashlib.file_digest(annotation.open("rb"), "sha256").hexdigest(),
                selected_channel=config[family]["channel"], decoded_samples=count,
                decoded_channel_pcm_s16le_sha256=hashlib.sha256(pcm).hexdigest(), annotation_gaps=gaps,
                speakers=sorted({row["speaker"] for row in protected})))
            if family == "aishell4":
                result["sources"][-1]["rttm_crosscheck"] = rttm
            for segment in config["segments"]:
                target = segment["target_seconds"]
                try:
                    first = safe_boundary(protected, segment["anchor_seconds"], count)
                    last = safe_boundary(protected, segment["anchor_seconds"] + target, count)
                except ValueError as error:
                    result["acquisition_failures"].append(dict(recording=identity, target_seconds=target, reason=str(error)))
                    continue
                if not first < last <= first + 600 * RATE:
                    raise ValueError("crop escapes the service duration limit")
                rows = sorted((row for row in words if Decimal(first) / RATE <= row["start"] <= row["end"] <= Decimal(last) / RATE),
                              key=lambda row: (row["start"], row["end"], row["speaker"], row["id"]))
                name = f"{identity}-{target}s"
                crop = pcm[first * 2:last * 2]
                write_crop(args.output / "pcm" / (name + ".wav"), crop)
                result["utterances"].append(dict(id=name, language="en" if family == "ami" else "zh", recording=identity,
                    path="pcm/" + name + ".wav", source_sample_range=[first, last], samples=last - first,
                    pcm_s16le_sha256=hashlib.sha256(crop).hexdigest(), text=(" " if family == "ami" else "").join(row["text"] for row in rows),
                    annotations=rows))
            # A separately labelled repeated-content stress case is not a real
            # 600-second annotated meeting and is excluded from the long set.
            if identity == "IS1000a":
                seed = next(row for row in result["utterances"] if row["id"] == "IS1000a-90s")
                a, b = seed["source_sample_range"]
                unit = pcm[a * 2:b * 2]
                repeated = (unit * ((600 * RATE * 2 + len(unit) - 1) // len(unit)))[:600 * RATE * 2]
                write_crop(args.output / "pcm" / "repeated-600s.wav", repeated)
                result["regression_inputs"].append(dict(id="repeated-600s", path="pcm/repeated-600s.wav", samples=600 * RATE,
                    source=seed["id"], pcm_s16le_sha256=hashlib.sha256(repeated).hexdigest(),
                    scope="repeated content regression stress only; not long-audio quality or unique-content performance"))
    expected = config["ami"]["known_boundary_acquisition_failures"]
    if result["acquisition_failures"] != expected:
        raise ValueError("acquisition failures changed from the pre-inference configuration")
    (args.output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(dict(annotated_crops=len(result["utterances"]), acquisition_failures=result["acquisition_failures"],
                          regression_inputs=len(result["regression_inputs"])), ensure_ascii=False))


if __name__ == "__main__":
    main()
