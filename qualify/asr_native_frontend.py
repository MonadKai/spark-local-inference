# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Check the native same-PCM frontend against frozen independent FP32 fixtures.

This invokes the real Rust/FFmpeg probe and does not execute a model. File
decoding, resampler equivalence and recognition remain separate qualifications.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pcm-root", type=Path, required=True)
    parser.add_argument("--sdk-observations", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from safetensors import safe_open

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    reference = json.loads((args.reference / "result.json").read_bytes())
    if manifest["split"] != "development" or reference["candidate_executed"] or reference["failures"]:
        raise ValueError("use complete independent development frontend fixtures")
    if reference["input_manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("frontend fixtures belong to different PCM selection")
    if (reference["max_abs_budget"], reference["rms_budget"]) != (2. ** -13, 2. ** -17):
        raise ValueError("frozen frontend budgets changed")
    originals = {row["id"]: row for row in manifest["utterances"]}
    cases = {}
    for directory in args.sdk_observations:
        sdk = json.loads((directory / "result.json").read_bytes())
        if sdk["errors"] or sdk["candidate_executed"] \
                or sdk["input_manifest_sha256"] != reference["input_manifest_sha256"]:
            raise ValueError("incomplete or different development case selection")
        for case in sdk["cases"]:
            if case["id"] in cases: raise ValueError("duplicate development case")
            cases[case["id"]] = case
    if cases.keys() != {row["id"] for row in reference["cases"]}:
        raise ValueError("reference and native cases differ")
    args.output.mkdir(exist_ok=False)
    record = dict(scope=__doc__.strip(), max_abs_budget=2. ** -13, rms_budget=2. ** -17,
        input_manifest_sha256=reference["input_manifest_sha256"], cases=[], failures=[])
    baseline = {}

    def publish():
        (args.output / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    publish()
    torch.set_num_threads(2)
    # Each probe starts a fresh Rust process. Reverse order in the second pass;
    # repeatability is independent of the cross-engine floating-point budget.
    # Persistent plan reuse is covered separately by the Rust pool tests.
    for repetition, order in enumerate((list(cases), list(reversed(cases)))):
        for name in order:
            case = cases[name]
            original = originals[case.get("source_id", name)]
            with wave.open(str(args.pcm_root / original["path"]), "rb") as source:
                if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()) \
                        != (1, 2, 16000, original["samples"]):
                    raise ValueError("PCM geometry changed")
                pcm = source.readframes(source.getnframes())
            if hashlib.sha256(pcm).hexdigest() != original["pcm_s16le_sha256"]:
                raise ValueError("PCM identity changed")
            pcm = pcm[:case["original_samples"] * 2]
            if "prefix_pcm_s16le_sha256" in case and hashlib.sha256(pcm).hexdigest() != case["prefix_pcm_s16le_sha256"]:
                raise ValueError("prefix PCM identity changed")
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768)
            samples = np.pad(samples, (0, (-len(samples)) % 160))
            input_path, output_path = args.output / (name + ".pcm.f32le"), args.output / (name + f"-{repetition}.mel.f32le")
            if repetition == 0: input_path.write_bytes(samples.astype("<f4").tobytes())
            process = subprocess.run([str(args.binary.resolve()), "features", str(input_path), str(output_path)],
                capture_output=True, text=True, timeout=60)
            row = dict(id=name, repetition=repetition, exit_code=process.returncode,
                stdout=process.stdout, stderr=process.stderr)
            record["cases"].append(row)
            if process.returncode:
                record["failures"].append(dict(id=name, repetition=repetition, reason="native probe failed"))
                publish()
                continue
            actual_bytes = output_path.read_bytes()
            if repetition == 0: baseline[name] = actual_bytes
            row["fresh_process_exact_bytes"] = repetition == 0 or baseline.get(name) == actual_bytes
            actual = torch.from_numpy(np.frombuffer(actual_bytes, dtype="<f4").copy())
            with safe_open(args.reference / (name + ".safetensors"), framework="pt") as saved:
                expected = saved.get_tensor("canonical_fp32").flatten()
            if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                record["failures"].append(dict(id=name, repetition=repetition,
                    reason="native frontend has different geometry or nonfinite values"))
                publish()
                continue
            error = actual.double() - expected.double()
            row.update(frames=case["frames"], max_abs=error.abs().max().item(), rms=error.square().mean().sqrt().item())
            if not row["fresh_process_exact_bytes"] or row["max_abs"] > record["max_abs_budget"] or row["rms"] > record["rms_budget"]:
                record["failures"].append(dict(id=name, repetition=repetition, reason="frontend gate failed"))
            publish()
    print(json.dumps(dict(cases=len(cases), native_calls=len(record["cases"]), failures=len(record["failures"]))))
    if record["failures"]: raise RuntimeError("native frontend qualification failed; all observations retained")


if __name__ == "__main__":
    main()
