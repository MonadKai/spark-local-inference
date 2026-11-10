# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Freeze development-stage scalar budgets from independent references only.

Each empirical budget is the observed canonical/SDK discrepancy plus one final
BF16 grid step. This explicitly chosen downstream allowance is not a theorem
about propagated network error. Local arithmetic, exact state/determinism,
decoded top-1 and blind recognition gates remain independent requirements.
"""

import argparse
import json
from pathlib import Path

import torch


def finite(value):
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError("nonfinite reference value cannot enter a frozen budget")


def scalar_budget(canonical, independent):
    if canonical.dtype != torch.bfloat16 or independent.dtype != canonical.dtype \
            or independent.shape != canonical.shape:
        raise ValueError("empirical stage budgets require matching BF16 tensors")
    finite(canonical)
    finite(independent)
    reference, other = canonical.double(), independent.double()
    difference = reference - other
    norm = reference.norm().item()
    if norm == 0:
        if torch.count_nonzero(other):
            raise ValueError("zero canonical stage needs an explicit absolute-error policy")
        return dict(max_abs=0., relative_l2=0., independent_max_abs=0., independent_relative_l2=0.,
            grid_step_max=0., grid_step_relative_l2=0.)
    _, exponent = torch.frexp(reference.abs())
    # Maximum adjacent spacing at the represented value, with gradual BF16
    # subnormals. At a binade boundary use the larger adjacent spacing.
    step = torch.exp2((exponent - 8).double()).clamp_min(2. ** -133)
    step = torch.where(reference == 0, 2. ** -133, step)
    maximum = difference.abs().max().item()
    relative = difference.norm().item() / norm
    gap_maximum, gap_relative = step.max().item(), step.norm().item() / norm
    return dict(max_abs=maximum + gap_maximum, relative_l2=relative + gap_relative,
        independent_max_abs=maximum, independent_relative_l2=relative,
        grid_step_max=gap_maximum, grid_step_relative_l2=gap_relative)


def check(value, reference, budget):
    if value.dtype != reference.dtype or value.shape != reference.shape:
        raise ValueError("numerical probe dtype/geometry changed")
    finite(value)
    finite(reference)
    if not reference.is_floating_point():
        if not torch.equal(value, reference): raise ValueError("integer probe changed")
        return
    error = (value.double() - reference.double()).abs()
    denominator = reference.double().norm().item()
    relative = error.norm().item() / denominator if denominator else (0. if not error.any() else float("inf"))
    if error.max().item() > budget["max_abs"] or relative > budget["relative_l2"]:
        raise ValueError("frozen empirical numerical budget exceeded")


def validate_case_layout(primary, case, sdk_case):
    if case["frames"] < 100 and (primary.get("layout") != "original"
            or primary.get("short_width_adapter") !=
            "conv2d1 pre-hook slices padded input to true F; no arithmetic replacement"):
        raise ValueError("short original-format inputs require the declared true-width reference adapter")
    if case["frames"] != sdk_case["frames"]:
        raise ValueError("canonical and SDK frame counts differ")


def validate_identical_features(canonical, independent):
    if canonical.dtype != independent.dtype or canonical.shape != independent.shape \
            or not torch.equal(canonical.view(torch.uint8), independent.view(torch.uint8)):
        raise ValueError("model-stage references require byte-identical represented convolution inputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("CANONICAL", "SDK"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from safetensors import safe_open

    torch.set_num_threads(2)
    result = dict(version=1, scope=__doc__.strip(), primary="canonical Transformers with original-format layout",
        checkpoint="Qwen/Qwen3-ASR-1.7B", checkpoint_revision="7278e1e70fe206f11671096ffdd38061171dd6e5",
        transformers_revision="1d9654e78a23fb167ef539cbd699dfbd3e4830cf",
        sdk_revision="7c6daf77a2421100f5fb066495372c00129d39ff", numerical_policy="qwen3-asr.bf16-fp32.v1",
        denominator="canonical tensor L2; errors include all elements; nonfinite is an unconditional failure",
        generation_decision_top1_mismatches_allowed=0, top1_ties="lowest token ID; no margin exclusions",
        full_prefill_top1="report every mismatch and both logits/margins, including non-decoding positions",
        same_path="byte-identical stages/logits/tokens at B=1/2/4/8, permutations, partial rows, slot reuse, cancellation and fresh processes",
        candidate_used_for_budgets=False, cases=[])
    seen = set()
    for primary_path, sdk_path in args.pair:
        primary_path, sdk_path = Path(primary_path), Path(sdk_path)
        primary = json.loads((primary_path / "result.json").read_bytes())
        sdk = json.loads((sdk_path / "result.json").read_bytes())
        if primary["errors"] or sdk["errors"] or primary["candidate_executed"] or sdk["candidate_executed"] \
                or primary.get("short_control") or primary["transformers_revision"] != result["transformers_revision"] \
                or sdk["sdk_revision"] != result["sdk_revision"] \
                or sdk["checkpoint_revision"] != result["checkpoint_revision"] \
                or primary["bf16_reduced_precision_reduction"] or sdk["bf16_reduced_precision_reduction"]:
            raise ValueError("incomplete, different or candidate-dependent references")
        if primary["original_tensor_values_verified"] != 708 or not primary["used_fp32_constants_exact_to_sdk"]:
            raise ValueError("independent original-weight or constant validation is missing")
        independent_cases = {case["id"]: case for case in sdk["cases"]}
        if {case["id"] for case in primary["cases"]} != independent_cases.keys():
            raise ValueError("reference pairs have different case sets")
        for case in primary["cases"]:
            name = case["id"]
            if name in seen: raise ValueError("duplicate development case")
            seen.add(name)
            sdk_case = independent_cases[name]
            validate_case_layout(primary, case, sdk_case)
            row = dict(id=name, frames=case["frames"], prompt_ids=sdk_case["prompt_ids"], stages={})
            with safe_open(primary_path / (name + "-prefill.safetensors"), framework="pt") as a, \
                    safe_open(sdk_path / (name + "-prefill.safetensors"), framework="pt") as b:
                input_key = "thinker.audio_tower.conv2d1.input.0"
                validate_identical_features(a.get_tensor(input_key), b.get_tensor(input_key))
                for key in sorted(a.keys()):
                    if key not in b.keys(): raise ValueError(f"unmatched primary probe: {key}")
                    value, other = a.get_tensor(key), b.get_tensor(key)
                    finite(value)
                    finite(other)
                    if (key.startswith("thinker.model.rotary_emb.") or key == "thinker.model.kwargs.position_ids") \
                            and other.shape == (3, *value.shape):
                        # Original mRoPE repeats the same sequence on all axes.
                        if not all(torch.equal(other[i].view(torch.uint8), other[0].view(torch.uint8)) for i in range(3)):
                            raise ValueError("original rotary axes are not equivalent")
                        other = other[0]
                    if not value.is_floating_point():
                        if not torch.equal(value, other): raise ValueError(f"integer layout differs: {key}")
                        continue
                    budget = scalar_budget(value, other)
                    row["stages"][key] = dict(dtype="BF16", shape=list(value.shape), **budget)
                if a.get_tensor("thinker.lm_head.output").dtype != torch.bfloat16:
                    raise ValueError("required BF16 logit node missing")
                if not torch.equal(a.get_tensor("thinker.lm_head.output")[:, -1].argmax(-1),
                        b.get_tensor("thinker.lm_head.output")[:, -1].argmax(-1)):
                    raise ValueError("independent last-prefill top1 disagreement requires an explicit decision")
            with safe_open(primary_path / (name + "-cached-teacher.safetensors"), framework="pt") as a, \
                    safe_open(sdk_path / (name + "-cached-teacher.safetensors"), framework="pt") as b:
                logits, other = a.get_tensor("logits_bf16"), b.get_tensor("logits_bf16")
                if not torch.equal(a.get_tensor("target_ids"), b.get_tensor("target_ids")):
                    raise ValueError("teacher histories differ")
                if not torch.equal(logits.argmax(-1), other.argmax(-1)):
                    raise ValueError("independent decoded top1 disagreement requires an explicit decision")
                row["teacher_history"] = a.get_tensor("target_ids").tolist()[0]
                row["teacher_top1"] = logits.argmax(-1).tolist()
                row["teacher_rows"] = [scalar_budget(x, y) for x, y in zip(logits, other, strict=True)]
            result["cases"].append(row)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(dict(cases=len(seen), prefill_scalar_probes=sum(len(c["stages"]) for c in result["cases"]),
        decoded_rows=sum(len(c["teacher_rows"]) for c in result["cases"]), candidate_used=False)))


if __name__ == "__main__":
    main()
