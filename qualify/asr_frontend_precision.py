# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterize independent FP32/FP64 Log-Mel references on fixed development PCM.

The absolute and RMS frontend budgets are fixed fractions of the BF16 unit roundoff (2^-8) at
unit magnitude (1/32 and 1/512 of 2^-8). They are downstream engineering budgets,
not FFT error theorems and not candidate-fitted tolerances. This process imports
no candidate code and performs no ASR forward or recognition qualification.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import wave

MAX_ABS = 2. ** -13
MAX_RMS = 2. ** -17


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pcm-root", type=Path, required=True)
    parser.add_argument("--sdk-observations", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from transformers import WhisperFeatureExtractor
    from safetensors.torch import save_file

    if any(importlib.util.find_spec(name) is not None for name in ("ignition", "flare_heliopause")):
        raise ValueError("reference environment contains candidate packages")
    source = json.loads(importlib.metadata.distribution("transformers").read_text("direct_url.json"))
    if source["vcs_info"]["commit_id"] != "1d9654e78a23fb167ef539cbd699dfbd3e4830cf":
        raise ValueError("canonical feature reference revision changed")
    manifest = json.loads(args.manifest.read_bytes())
    if manifest["split"] != "development": raise ValueError("use the independent development set")
    originals = {row["id"]: row for row in manifest["utterances"]}
    selected = []
    for directory in args.sdk_observations:
        observations = json.loads((directory / "result.json").read_bytes())
        if observations["errors"] or observations["candidate_executed"]:
            raise ValueError("incomplete or candidate-dependent case selection")
        selected.extend(observations["cases"])
    if len({row["id"] for row in selected}) != len(selected): raise ValueError("duplicate case")
    feature = WhisperFeatureExtractor.from_pretrained(args.checkpoint, local_files_only=True)
    if (feature.feature_size, feature.n_fft, feature.hop_length, feature.sampling_rate) != (128, 400, 160, 16000):
        raise ValueError("canonical frontend geometry changed")
    torch.set_num_threads(2)
    window = .5 * (1 - torch.cos(torch.arange(400, dtype=torch.float64) * (2 * math.pi / 400)))
    mel = torch.from_numpy(feature.mel_filters).double()
    args.output.mkdir(exist_ok=False)
    record = dict(scope=__doc__.strip(), max_abs_budget=MAX_ABS, rms_budget=MAX_RMS,
        selection="existing independent SDK development-stage cases, unchanged PCM and hop padding",
        fp64="periodic FP64 Hann, complex FP64 STFT, original FP64 Slaney matrix, clamp/log/global-per-input normalization",
        input_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        transformers_revision=source["vcs_info"]["commit_id"], torch=torch.__version__,
        candidate_executed=False, cases=[], failures=[])

    def publish():
        (args.output / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    publish()
    for case in selected:
        original = originals[case.get("source_id", case["id"])]
        with wave.open(str(args.pcm_root / original["path"]), "rb") as source:
            assert (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()) \
                == (1, 2, 16000, original["samples"])
            pcm = source.readframes(source.getnframes())
        assert hashlib.sha256(pcm).hexdigest() == original["pcm_s16le_sha256"]
        pcm = pcm[:case["original_samples"] * 2]
        if "prefix_pcm_s16le_sha256" in case:
            assert hashlib.sha256(pcm).hexdigest() == case["prefix_pcm_s16le_sha256"]
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768)
        samples = np.pad(samples, (0, (-len(samples)) % 160))
        canonical = torch.from_numpy(feature._torch_extract_fbank_features(samples[None], device="cpu"))[0]
        stft = torch.stft(torch.from_numpy(samples).double(), 400, 160, window=window, return_complex=True)
        powers = stft[:, :-1].abs().square()
        high = (mel.T @ powers).clamp_min(1e-10).log10()
        high = torch.maximum(high, high.max() - 8)
        high = (high + 4) / 4
        if canonical.shape != high.shape or canonical.shape != (128, case["frames"]):
            raise ValueError("frontend frame geometry differs")
        if not torch.isfinite(canonical).all() or not torch.isfinite(high).all():
            raise ValueError("nonfinite frontend reference")
        repeated = torch.from_numpy(feature._torch_extract_fbank_features(samples[None], device="cpu"))[0]
        assert torch.equal(canonical.view(torch.uint8), repeated.view(torch.uint8))
        error = canonical.double() - high
        row = dict(id=case["id"], samples=len(samples), frames=case["frames"],
            max_abs=error.abs().max().item(), rms=error.square().mean().sqrt().item(),
            fp32_same_path_exact_bytes=True)
        row["within_fixed_budget"] = row["max_abs"] <= MAX_ABS and row["rms"] <= MAX_RMS
        record["cases"].append(row)
        if not row["within_fixed_budget"]: record["failures"].append(case["id"])
        save_file(dict(canonical_fp32=canonical.contiguous(), independent_fp64=high.contiguous()),
            args.output / (case["id"] + ".safetensors"))
        publish()
    print(json.dumps(dict(cases=len(record["cases"]), failures=len(record["failures"]), candidate_executed=False)))
    if record["failures"]: raise RuntimeError("independent frontend calibration exceeded fixed budgets")


if __name__ == "__main__":
    main()
