# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Check original SDK BF16 linear probes against independent FP64 sums.

Every observed linear is checked at first/middle/last rows and six fixed output
channels, including both vocabulary ends. This is an operator calibration, not
a proof about unobserved elements, arbitrary kernels or accumulated network error.
The conditional envelope assumes at most n RNE FP32 updates plus bounded FTZ;
an environment flag or a Tensor Core instruction alone does not prove that model.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import torch


def gamma(n, precision):
    unit = 2. ** -precision
    if n * unit >= 1:
        raise ValueError("reduction length exceeds the stated accumulation model")
    return n * unit / (1 - n * unit)


def bf16_rounding_sup(absolute_upper):
    """Conservative half-gap over an entire finite, non-overflowing interval.

    The maximum magnitude fixes the largest binade touched by the interval;
    using the gap at its centre would fail when the interval crosses a binade.
    BF16 subnormal spacing is constant, including intervals that contain zero.
    """
    if not torch.isfinite(absolute_upper).all() or (absolute_upper < 0).any() \
            or (absolute_upper > torch.finfo(torch.bfloat16).max).any():
        raise ValueError("classify nonfinite/overflow intervals separately")
    _, exponent = torch.frexp(absolute_upper)
    half_gap = torch.exp2((exponent - 9).double()).clamp_min(2. ** -134)
    return torch.where(absolute_upper == 0, 2. ** -134, half_gap)


def envelope(source, weight, bias):
    if source.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 \
            or bias is not None and bias.dtype != torch.bfloat16:
        raise ValueError("this bound requires exact BF16 operands")
    for value in (source, weight, bias):
        if value is not None and (not torch.isfinite(value).all()
                or ((value != 0) & (value.abs() < torch.finfo(torch.bfloat16).tiny)).any()):
            raise ValueError("classify nonfinite/subnormal BF16 operands separately")
    x, w = source.double(), weight.double()
    products = x[:, None, :] * w[None, :, :]
    if not torch.isfinite(products).all() or ((products != 0) & (products.abs() < torch.finfo(torch.float32).tiny)).any():
        raise ValueError("classify FP32 product underflow/nonfinite cases separately")
    length = x.shape[-1] + int(bias is not None)
    g64, g32 = gamma(length, 53), gamma(length, 24)
    infinity = torch.full((), torch.inf, dtype=torch.float64)
    magnitude = products.abs().sum(-1)
    if bias is not None: magnitude = magnitude + bias.double().abs()
    # Outward padding covers this FP64 evaluation of the envelope itself.
    magnitude = torch.nextafter(magnitude / (1 - g64) * (1 + 8 * 2. ** -53), infinity)
    exact = products.sum(-1)
    if bias is not None: exact = exact + bias.double()
    reference_error = magnitude * g64
    # Even normal products can cancel into a subnormal partial sum. A declared
    # K(+bias)-update reduction may lose at most tiny per update under FTZ;
    # later rounded updates amplify that absolute term by <= 1/(1-n*u).
    underflow_error = length * torch.finfo(torch.float32).tiny / (1 - length * 2. ** -24)
    accumulation_error = torch.nextafter((magnitude * g32 + underflow_error)
        * (1 + 8 * 2. ** -53), infinity)
    if (torch.nextafter(magnitude + accumulation_error, infinity) > torch.finfo(torch.float32).max).any():
        raise ValueError("the FP32 partial sums might overflow")
    radius = torch.nextafter((reference_error + accumulation_error) * (1 + 8 * 2. ** -53), infinity)
    maximum = torch.nextafter(exact.abs() + radius, infinity)
    bound = torch.nextafter(radius + bf16_rounding_sup(maximum), infinity)
    return exact, bound, reference_error, accumulation_error


def result_ratio(observed, exact, bound):
    if observed.dtype != torch.bfloat16 or observed.shape != exact.shape:
        raise ValueError("observed linear output must have matching BF16 geometry")
    if not torch.isfinite(observed).all() or not torch.isfinite(exact).all() \
            or not torch.isfinite(bound).all() or not (bound > 0).all():
        raise ValueError("nonfinite output/reference or invalid bound")
    ratio = ((observed.double() - exact).abs() / bound).max().item()
    if not 0 <= ratio < float("inf"):
        raise ValueError("nonfinite envelope ratio")
    return ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sdk-observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from safetensors import safe_open

    torch.set_num_threads(2)
    sdk = json.loads((args.sdk_observations / "result.json").read_bytes())
    if sdk["errors"] or sdk["bf16_reduced_precision_reduction"] is not False:
        raise ValueError("independent reference is incomplete or has different reduction semantics")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_bytes())["weight_map"]
    record = dict(scope=__doc__.strip(), cases=[], failures=[],
        checkpoint_revision=sdk["checkpoint_revision"], sdk_revision=sdk["sdk_revision"],
        scalar_limit=1., finite_bf16_linear_only=True,
        arithmetic="gamma32(n)*sum_abs(products,bias) + n*FP32tiny/(1-n*u32) + FP64-reference-error + sup BF16 rounding over full interval; n=K+bias",
        domain="normal-or-zero finite BF16 operands and exact products; non-overflowing declared FP32 reduction with at most n RNE updates, bounded partial-sum FTZ, RNE gradual BF16 output conversion",
        arbitrary_tensor_core_arithmetic_qualified=False,
        same_operand_comparison=True, candidate_executed=False, network_thresholds_admitted=False)

    def publish():
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    if args.output.exists(): raise ValueError("refusing to replace an earlier calibration attempt")
    publish()
    with ExitStack() as owners:
        shards = {name: owners.enter_context(safe_open(args.checkpoint / name, framework="pt"))
            for name in sorted(set(index.values()))}
        for case in sdk["cases"]:
            row = dict(id=case["id"], probes=[])
            record["cases"].append(row)
            with safe_open(args.sdk_observations / (case["id"] + "-prefill.safetensors"), framework="pt") as stages:
                for key in sorted(stages.keys()):
                    if not key.endswith(".output"): continue
                    module = key.removesuffix(".output")
                    name = module + ".weight"
                    if name not in index or module + ".input.0" not in stages.keys(): continue
                    tensor = shards[index[name]].get_tensor(name)
                    if tensor.ndim != 2: continue
                    x, y = stages.get_tensor(module + ".input.0"), stages.get_tensor(key)
                    x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
                    if tensor.shape != (y.shape[-1], x.shape[-1]): raise ValueError("linear shape mismatch")
                    rows = sorted({0, x.shape[0] // 2, x.shape[0] - 1})
                    channels = sorted({0, 1, tensor.shape[0] // 4, tensor.shape[0] // 2,
                        tensor.shape[0] - 2, tensor.shape[0] - 1})
                    bias_name = module + ".bias"
                    bias = shards[index[bias_name]].get_tensor(bias_name)[channels] if bias_name in index else None
                    exact, bound, error64, error32 = envelope(x[rows], tensor[channels], bias)
                    observed = y[rows][:, channels]
                    ratio = result_ratio(observed, exact, bound)
                    # The same numerical check must reject a wrong BF16 result.
                    wrong = (observed.double() + 4 * bound).bfloat16()
                    control = result_ratio(wrong, exact, bound) > 1
                    probe = dict(module=module, reduction=x.shape[-1], rows=rows, channels=channels,
                        max_envelope_ratio=ratio, max_abs=(observed.double()-exact).abs().max().item(),
                        max_bound=bound.max().item(), fp64_reference_error_max=error64.max().item(),
                        fp32_accumulation_error_max=error32.max().item(), negative_control_detected=control)
                    row["probes"].append(probe)
                    if ratio > 1 or not control: record["failures"].append(dict(id=case["id"], **probe))
            publish()
    print(json.dumps(dict(cases=len(record["cases"]), probes=sum(len(c["probes"]) for c in record["cases"]),
        failures=len(record["failures"]), network_thresholds_admitted=False)))
    if record["failures"]: raise RuntimeError("reference linear calibration failed")


if __name__ == "__main__":
    main()
