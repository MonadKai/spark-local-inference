# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Cross-check unprepared canonical stages against the independent original SDK.

Run with Python -I in the isolated pinned canonical environment. The original
layout crops only conv2d1's padded input for F<100; all arithmetic forwards stay
canonical. The canonical layout retains the unmodified padded-100 control.
Neither mode runs Ignition or admits a candidate.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import traceback


def snapshot_cached_state(cache, consumed):
    """Copy the actual full, post-RoPE canonical KV before the next update.

    The fixed checkpoint has28 full-attention layers. Newly projected operands
    alone cannot represent the consumed past used by a cached attention call.
    """
    import torch

    if type(consumed) is not int or consumed <= 0 or len(cache.layers) != 28 \
            or cache.get_seq_length() != consumed:
        raise ValueError("complete canonical28-layer consumed KV required")
    result = {}
    for index, layer in enumerate(cache.layers):
        for name in ("keys", "values"):
            value = getattr(layer, name)
            if not isinstance(value, torch.Tensor) or value.shape != (1, 8, consumed, 128) \
                    or value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                raise ValueError("canonical cached KV shape, dtype or finite values differ")
            result[f"thinker.model.cache.{index}.{name}"] = value.detach().to("cpu", copy=True).contiguous()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sdk-observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("canonical", "original"), default="canonical")
    parser.add_argument("--short-control", action="store_true",
                        help="run only the F<100 unmodified canonical padding controls")
    parser.add_argument("--cached-stages", action="store_true",
                        help="also record actual cached teacher decoder operands; diagnostic only")
    parser.add_argument("--frozen-canonical", type=Path,
                        help="require cached teacher logits to remain byte-identical to this independent reference")
    args = parser.parse_args()
    if args.short_control and args.layout != "canonical":
        parser.error("short-control requires the unmodified canonical layout")
    if args.cached_stages != (args.frozen_canonical is not None):
        parser.error("cached-stages and frozen-canonical must be supplied together")
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    spec = importlib.util.spec_from_file_location("canonical_loading_probe",
        Path(__file__).with_name("asr_canonical_reference.py"))
    loading = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loading)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("canonical reference requires one assigned SM121 GPU")
    reference, index, renamed = loading.load_reference(args.checkpoint)
    # Loading uses the unmodified upstream converter, independently of Ignition.
    # Verify all values again in this process before the first model forward.
    from contextlib import ExitStack
    with ExitStack() as owners:
        shards = {name: owners.enter_context(safe_open(args.checkpoint / name, framework="pt"))
            for name in sorted(set(index.values()))}
        actual = reference.state_dict()
        assert actual.keys() == renamed.keys()
        for destination, original in renamed.items():
            source, target = shards[index[original]].get_tensor(original), actual[destination]
            assert source.dtype == target.dtype == torch.bfloat16 and source.shape == target.shape
            assert torch.isfinite(source).all() and torch.equal(source.view(torch.uint8), target.view(torch.uint8))
    reference.set_attn_implementation("sdpa")
    constants = {"audio_position": reference.model.audio_tower.positional_embedding.positional_embedding,
        "rotary_inv_freq": reference.model.language_model.rotary_emb.inv_freq,
        "rotary_original_inv_freq": reference.model.language_model.rotary_emb.original_inv_freq}
    with safe_open(args.sdk_observations / "fp32-constants.safetensors", framework="pt") as original:
        for name, value in constants.items():
            expected = original.get_tensor(name)
            if name == "audio_position":
                # The SDK retains 1500 positions but its stride-2 CNN emits at
                # most thirteen per chunk. Canonical stores this used prefix.
                assert value.shape == (13, 1024) and expected.shape == (1500, 1024)
                expected = expected[:13]
            assert value.dtype == expected.dtype == torch.float32 and torch.equal(value, expected)
    reference.to("cuda:0").eval()
    args.output.mkdir(exist_ok=False)
    sdk = json.loads((args.sdk_observations / "result.json").read_bytes())
    if sdk["errors"] or sdk["checkpoint_revision"] != "7278e1e70fe206f11671096ffdd38061171dd6e5":
        raise ValueError("original SDK observations are incomplete or have a different checkpoint")
    record = dict(scope=__doc__.strip(), transformers_revision=loading.REVISION,
        torch=torch.__version__, attention="unmodified canonical SDPA; explicit per-window encoder attention",
        original_tensor_values_verified=len(index), bf16_reduced_precision_reduction=False,
        used_fp32_constants_exact_to_sdk=True, audio_position_rows_compared=13,
        unused_sdk_audio_position_rows=1487,
        layout=args.layout,
        short_control=args.short_control,
        cached_stages=args.cached_stages,
        short_width_adapter="conv2d1 pre-hook slices padded input to true F; no arithmetic replacement"
            if args.layout == "original" else None,
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        cases=[], errors=[], candidate_executed=False, cross_engine_tolerances_admitted=False)

    def publish():
        (args.output / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    def cpu(value):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("canonical reference produced nonfinite values")
        return value.detach().to("cpu", copy=True).contiguous()

    def same(left, right):
        return left.shape == right.shape and left.dtype == right.dtype \
            and torch.equal(left.view(torch.uint8), right.view(torch.uint8))

    def metrics(left, right):
        if left.shape != right.shape or left.dtype != right.dtype:
            return dict(comparable=False, canonical_shape=list(left.shape), sdk_shape=list(right.shape),
                canonical_dtype=str(left.dtype), sdk_dtype=str(right.dtype))
        if not left.is_floating_point():
            return dict(comparable=True, exact_bytes=same(left, right))
        a, b = left.double(), right.double()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("nonfinite reference comparison")
        difference = a - b
        result = dict(comparable=True, exact_bytes=same(left, right),
            max_abs=difference.abs().max().item(), rmse=difference.square().mean().sqrt().item(),
            relative_l2=(difference.norm() / b.norm().clamp_min(1e-300)).item(),
            differing_elements=(left != right).sum().item(), elements=left.numel(),
            max_over_reference_max=(difference.abs().max() / b.abs().max().clamp_min(1e-300)).item())
        if left.dtype == torch.bfloat16:
            def ordered(value):
                bits = value.view(torch.int16).int()
                return torch.where(bits < 0, 32768 - (bits & 32767), 32768 + bits)
            result["max_bf16_ulp"] = (ordered(left) - ordered(right)).abs().max().item()
        return result

    def top1_differences(canonical, original):
        a, b = canonical.reshape(-1, canonical.shape[-1]), original.reshape(-1, original.shape[-1])
        ta, tb = a.argmax(-1), b.argmax(-1)
        result = []
        for index in (ta != tb).nonzero().flatten().tolist():
            ca, sa = ta[index].item(), tb[index].item()
            result.append(dict(row=index, canonical_top1=ca, sdk_top1=sa,
                canonical_logits=[a[index, ca].item(), a[index, sa].item()],
                sdk_logits=[b[index, ca].item(), b[index, sa].item()],
                canonical_margin=torch.topk(a[index].float(), 2).values.diff().abs().item(),
                sdk_margin=torch.topk(b[index].float(), 2).values.diff().abs().item(),
                row_max_abs=(a[index].float()-b[index].float()).abs().max().item()))
        return result

    def original_name(name):
        for source, target in [("model.multi_modal_projector.linear_1", "thinker.audio_tower.proj1"),
                               ("model.multi_modal_projector.linear_2", "thinker.audio_tower.proj2"),
                               ("model.multi_modal_projector.act", "thinker.audio_tower.act"),
                               ("model.audio_tower", "thinker.audio_tower"),
                               ("model.language_model", "thinker.model"), ("lm_head", "thinker.lm_head")]:
            if name == source or name.startswith(source + "."):
                return target + name[len(source):]
        return name

    def observe(inputs, *, use_cache=False, return_output=False):
        values, handles = {}, []

        def store(name, value):
            if isinstance(value, torch.Tensor):
                if name in values:
                    raise ValueError(f"canonical stage invoked more than once: {name}")
                values[name] = cpu(value)
            elif isinstance(value, (tuple, list)):
                for i, item in enumerate(value): store(f"{name}.{i}", item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, torch.Tensor): store(f"{name}.{key}", item)

        for path, module in reference.named_modules():
            name = original_name(path)
            ordinary = isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm, torch.nn.SiLU)) \
                or name.endswith(("q_norm", "k_norm", "input_layernorm", "post_attention_layernorm",
                                  "rotary_emb", "model.norm"))
            if ordinary or name in ("thinker.audio_tower.layers.0", "thinker.model"):
                def before(module, positional, keywords, name=name):
                    store(name + ".input", positional)
                    store(name + ".kwargs", keywords)
                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            if ordinary or name.rsplit(".", 1)[-1].isdigit() and ".layers." in name:
                handles.append(module.register_forward_hook(
                    lambda module, positional, output, name=name: store(name + ".output", output)))
        try:
            past = inputs.get("past_key_values")
            consumed = inputs["input_ids"].shape[-1] + (past.get_seq_length() if past is not None else 0)
            output = reference(**inputs, use_cache=use_cache)
            if args.cached_stages and use_cache:
                snapshot = snapshot_cached_state(output.past_key_values, consumed)
                if values.keys() & snapshot.keys():
                    raise ValueError("cached KV observation collides with a module stage")
                values.update(snapshot)
            assert same(cpu(output.logits), values["thinker.lm_head.output"])
        finally:
            for handle in handles: handle.remove()
        return (values, output) if return_output else values

    if args.cached_stages:
        frozen = json.loads((args.frozen_canonical / "result.json").read_bytes())
        if frozen["errors"] or frozen["candidate_executed"] \
                or frozen["transformers_revision"] != loading.REVISION \
                or frozen["layout"] != args.layout \
                or {case["id"] for case in frozen["cases"]} != {case["id"] for case in sdk["cases"]}:
            raise ValueError("complete matching independent canonical observations required")

    publish()
    with torch.inference_mode():
        for case in sdk["cases"]:
            if args.short_control and case["frames"] >= 100: continue
            row = dict(id=case["id"], frames=case["frames"], metrics={})
            record["cases"].append(row)
            short_layout = None
            try:
                if args.layout == "original" and case["frames"] < 100:
                    frames = case["frames"]
                    short_layout = reference.model.audio_tower.conv2d1.register_forward_pre_hook(
                        lambda module, positional, frames=frames: (positional[0][..., :frames],))
                with safe_open(args.sdk_observations / (case["id"] + "-prefill.safetensors"), framework="pt") as original:
                    features = original.get_tensor("processor.input_features").to("cuda")
                    extra = (-features.shape[-1]) % 100
                    padded = torch.nn.functional.pad(features, (0, extra))
                    inputs = dict(input_ids=torch.tensor([case["prompt_ids"]], device="cuda"),
                        input_features=padded, input_features_mask=(torch.arange(padded.shape[-1],
                            device="cuda") < features.shape[-1])[None].long())
                    positions = torch.arange(len(case["prompt_ids"])).reshape(1, -1)
                    sdk_positions = original.get_tensor("thinker.model.kwargs.position_ids")
                    assert sdk_positions.shape == (3, *positions.shape)
                    assert torch.equal(sdk_positions, positions.expand(3, -1, -1))
                    inputs.update(attention_mask=torch.ones_like(inputs["input_ids"]),
                        position_ids=positions.to("cuda"))
                    row["position_ids_equal_all_sdk_axes"] = True
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                        values = observe(inputs)
                    row["observed_sdpa_dispatch"] = sorted({event.key for event in profile.key_averages()
                        if "scaled_dot_product" in event.key})
                    if not row["observed_sdpa_dispatch"]:
                        raise ValueError("no actual SDPA dispatcher operation was observed")
                    repeated = observe(inputs)
                    assert values.keys() == repeated.keys()
                    for name, value in values.items(): assert same(value, repeated[name]), name
                    del repeated
                    missing = set(original.keys()) - values.keys() - {"processor.input_features"}
                    row["sdk_probes_without_canonical_name"] = sorted(missing)
                    for name, value in values.items():
                        if name in original.keys(): row["metrics"][name] = metrics(value, original.get_tensor(name))
                    logits = values["thinker.lm_head.output"]
                    sdk_logits = original.get_tensor("thinker.lm_head.output")
                    row["prefill_top1_differences"] = (logits.argmax(-1) != sdk_logits.argmax(-1)).sum().item()
                    row["prefill_top1_details"] = top1_differences(logits, sdk_logits)
                    row["prefill_rows"] = logits.shape[-2]
                    row["last_prefill_top1_equal"] = torch.equal(logits[:, -1].argmax(-1), sdk_logits[:, -1].argmax(-1))
                    save_file(values, args.output / (case["id"] + "-prefill.safetensors"))
                    del values
                with safe_open(args.sdk_observations / (case["id"] + "-cached-teacher.safetensors"), framework="pt") as original:
                    target = original.get_tensor("target_ids")
                    teacher = target.to("cuda")
                    baseline = None
                    cached_observations = {}
                    def cached_forward(step, arguments):
                        if not args.cached_stages:
                            return reference(**arguments, use_cache=True)
                        values, output = observe(arguments, use_cache=True, return_output=True)
                        # These are real cached-forward operands, not a slice
                        # of prefill presented as decode observations.
                        if step in cached_observations:
                            previous = cached_observations[step]
                            assert values.keys() == previous.keys()
                            for name, value in values.items():
                                assert same(value, previous[name]), (step, name)
                        else:
                            cached_observations[step] = values
                            save_file(values, args.output / (case["id"] + f"-teacher-step-{step}.safetensors"))
                        return output

                    for _ in range(2):
                        output = cached_forward(0, inputs)
                        rows = [cpu(output.logits[:, -1])]
                        for i in range(1, teacher.shape[1]):
                            output = cached_forward(i, dict(input_ids=teacher[:, i-1:i],
                                past_key_values=output.past_key_values))
                            assert output.past_key_values.get_seq_length() == len(case["prompt_ids"]) + i
                            rows.append(cpu(output.logits[:, -1]))
                        actual = torch.cat(rows)
                        if baseline is not None: assert same(actual, baseline)
                        else: baseline = actual
                        del output
                    if args.cached_stages:
                        with safe_open(args.frozen_canonical / (case["id"] + "-cached-teacher.safetensors"),
                                framework="pt") as previous:
                            assert same(target, previous.get_tensor("target_ids"))
                            assert same(baseline, previous.get_tensor("logits_bf16"))
                        row["cached_stage_steps"] = sorted(cached_observations)
                        row["cached_stage_repeated_exact"] = True
                        row["cached_logits_exact_frozen_canonical"] = True
                    del cached_observations
                    row["teacher_metrics"] = metrics(baseline, original.get_tensor("logits_bf16"))
                    row["teacher_top1_differences"] = (baseline.argmax(-1) != original.get_tensor("logits_bf16").argmax(-1)).sum().item()
                    row["teacher_top1_details"] = top1_differences(baseline, original.get_tensor("logits_bf16"))
                    row["teacher_rows"] = teacher.shape[1]
                    save_file(dict(logits_bf16=baseline, target_ids=target),
                        args.output / (case["id"] + "-cached-teacher.safetensors"))
                row["status"] = "reference_crosscheck_only"
            except Exception as error:
                row.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                record["errors"].append(case["id"])
            finally:
                if short_layout is not None: short_layout.remove()
            publish()
    if record["errors"]: raise RuntimeError("canonical reference checks failed; retained results contain all attempts")


if __name__ == "__main__":
    main()
