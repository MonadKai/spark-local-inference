# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Paired recognition scoring for the frozen English/Mandarin ASR corpus.

Inputs are a split="qualification" corpus manifest with utterances
[{id, language, speaker, text}]
and two result files [{id, text, status}]. Missing results remain failed rows.
Status is success, failed or truncated. No failed row leaves the denominator.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import ceil
from pathlib import Path
import random
import re
import unicodedata

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261106


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", "", value).split())


def edit_distance(reference, hypothesis) -> int:
    # The shorter sequence is the row dimension; the recurrence is symmetric.
    if len(reference) > len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(reference) + 1))
    for row, actual in enumerate(hypothesis, 1):
        current = [row]
        for column, expected in enumerate(reference, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (actual != expected)))
        previous = current
    return previous[-1]


def metric_units(text: str, language: str, metric: str):
    text = normalize(text)
    if metric == "wer":
        return text.split()
    return text.replace(" ", "") if language == "zh" else text


def paired_upper_bound(rows: list[tuple[int, int, int]], *, replicates=BOOTSTRAP_REPLICATES,
                       seed=BOOTSTRAP_SEED) -> float:
    """One-sided 95% percentile bound from paired utterance resampling.

    Each tuple is (reference-engine edits, candidate edits, truth unit count).
    Recompute corpus edit/reference totals per draw, never mean utterance rates.
    Python's Random(seed), stable input order and nearest-rank percentile define
    this protocol. The qualification entry point fixes 10,000 draws/seed.
    """
    if not rows or replicates < 1 or any(total <= 0 for _, _, total in rows):
        raise ValueError("bootstrap requires positive truth extents and repetitions")
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        difference = total = 0
        for _ in rows:
            reference, candidate, units = rows[rng.randrange(len(rows))]
            difference += candidate - reference
            total += units
        samples.append(difference / total)
    samples.sort()
    return samples[ceil(0.95 * replicates) - 1]


def _results(values: list[dict], expected: set[str]) -> dict[str, dict]:
    result = {}
    for value in values:
        identity = value["id"]
        if identity not in expected or identity in result:
            raise ValueError(f"unknown or duplicate result ID: {identity}")
        if value.get("status") not in ("success", "failed", "truncated") or not isinstance(value.get("text"), str):
            raise ValueError(f"invalid transcription result: {identity}")
        result[identity] = value
    return result


def evaluate(manifest: dict, reference: list[dict], candidate: list[dict]) -> dict:
    if manifest.get("split") != "qualification":
        raise ValueError("qualification scoring requires the frozen qualification split")
    utterances = manifest["utterances"]
    expected = {row["id"] for row in utterances}
    if len(expected) != len(utterances):
        raise ValueError("duplicate corpus utterance")
    counts = Counter(row["language"] for row in utterances)
    if counts != {"en": 128, "zh": 128}:
        raise ValueError("qualification requires exactly 128 English and 128 Mandarin utterances")
    for language in ("en", "zh"):
        speakers = {row["speaker"] for row in utterances if row["language"] == language}
        if len(speakers) < 16:
            raise ValueError(f"qualification requires at least 16 {language} speakers")
    reference, candidate = _results(reference, expected), _results(candidate, expected)
    per_utterance = []
    valid = True
    for row in sorted(utterances, key=lambda row: row["id"]):
        identity, language = row["id"], row["language"]
        missing = {"text": "", "status": "failed"}
        ref, cand = reference.get(identity, missing), candidate.get(identity, missing)
        valid &= ref["status"] == cand["status"] == "success"
        scores = {}
        for metric in (("wer", "cer") if language == "en" else ("cer",)):
            truth = metric_units(row["text"], language, metric)
            if not truth:
                raise ValueError(f"empty normalized truth for {identity}/{metric}")
            scores[metric] = {
                "units": len(truth),
                "reference_edits": edit_distance(truth, metric_units(ref["text"], language, metric)),
                "candidate_edits": edit_distance(truth, metric_units(cand["text"], language, metric)),
            }
        per_utterance.append({"id": identity, "language": language, "speaker": row["speaker"],
            "truth": row["text"], "reference_text": ref["text"], "candidate_text": cand["text"],
            "reference_status": ref["status"], "candidate_status": cand["status"], "scores": scores})
    metrics = {}
    for language, metric, point_numerator in (("en", "wer", 2), ("en", "cer", 1), ("zh", "cer", 2)):
        rows = [row["scores"][metric] for row in per_utterance if row["language"] == language]
        units = sum(row["units"] for row in rows)
        ref = sum(row["reference_edits"] for row in rows)
        cand = sum(row["candidate_edits"] for row in rows)
        primary = metric == "wer" or language == "zh"
        upper = paired_upper_bound([(row["reference_edits"], row["candidate_edits"], row["units"])
                                    for row in rows]) if primary else None
        point_pass = (cand - ref) * 1000 <= point_numerator * units
        bound_pass = upper <= 0.005 if primary else True
        metrics[f"{language}_{metric}"] = {"units": units, "reference_edits": ref,
            "candidate_edits": cand, "reference_rate": ref / units, "candidate_rate": cand / units,
            "degradation": (cand - ref) / units, "upper_95_degradation": upper,
            "point_gate": point_pass, "confidence_gate": bound_pass}
    return {"split": "qualification", "valid_complete_pairs": bool(valid), "accepted": bool(valid) and
            all(m["point_gate"] and m["confidence_gate"] for m in metrics.values()),
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "unit": "paired utterance", "interval": "one-sided 95% percentile, nearest rank"},
        "metrics": metrics, "utterances": per_utterance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(*(json.loads(path.read_text()) for path in (args.manifest, args.reference, args.candidate)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
