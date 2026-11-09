# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Check bounded prompt projection geometry against the fixed8192-row path.

All20 independent frozen inputs and all28 layers retain their actual operands.
Only the prompt staging extent changes:256 or512 instead of8192. Each result
is assembled from captured64-row fragments. This is a component experiment,
not a native graph selector, full-network qualification or serving benchmark.
"""

import argparse
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--canonical-observations", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output; retain every previous attempt")
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    from ignition.planning import discover_gpus
    apply(Configuration.load(args.config), discover_gpus())

    import torch
    from safetensors import safe_open
    from ignition.components.positioned_projection import PreparedPositionedProjection
    from ignition.preparation.failure import retain_on_failed_drain
    from ignition.recipes.qwen3_asr.checkpoint import inspect_original_checkpoint
    from ignition.recipes.qwen3_asr.numerics import fp32_reductions
    from qualify.asr_projection_tuning import component_interval
    from qualify.asr_freeze_numerics import validate_case_layout

    inspect_original_checkpoint(args.checkpoint)
    contract = json.loads((Path(__file__).resolve().parents[1]
        / "benchmarks/qwen3_asr/numerical-contract.json").read_bytes())
    expected_cases = {case["id"]: (case["frames"], len(case["prompt_ids"])) for case in contract["cases"]}
    frozen_cases = {case["id"]: case for case in contract["cases"]}
    cases = []
    for root in args.canonical_observations:
        result = json.loads((root / "result.json").read_bytes())
        if result["errors"] or result["candidate_executed"] or result.get("short_control") \
                or result["layout"] not in ("canonical", "original") \
                or any(case["frames"] < 100 and result["layout"] != "original" for case in result["cases"]) \
                or result["transformers_revision"] != "1d9654e78a23fb167ef539cbd699dfbd3e4830cf":
            raise ValueError("unchanged complete independent canonical observations required")
        for case in result["cases"]:
            validate_case_layout(result, case, frozen_cases[case["id"]])
        cases.extend((root, case) for case in result["cases"])
    if len(cases) != 20 or len({case["id"] for _, case in cases}) != 20 \
            or {case["id"]: (case["frames"], case["prefill_rows"]) for _, case in cases} != expected_cases:
        raise ValueError("the complete frozen20-input set is required")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_bytes())["weight_map"]
    args.output.mkdir()
    record = dict(scope=__doc__.strip(), status="running", errors=[], observations=[],
        torch=torch.__version__, cuda_runtime=torch.version.cuda, physical_fragment_rows=64,
        baseline_prompt_rows=8192, candidate_prompt_rows=[256, 512],
        planned_inputs=20, planned_projections=20 * 28 * 7,
        expected_checkpoint_revision="7278e1e70fe206f11671096ffdd38061171dd6e5",
        checkpoint_payload_identity_rehashed_by_this_tool=False,
        reference_scope="Existing canonical stage files selected by frozen contract case IDs/frame counts/prompt extents and original short-width layout; stage files contain embeddings, not prompt token IDs, which are not revalidated here",
        reference_revision="1d9654e78a23fb167ef539cbd699dfbd3e4830cf",
        timing_case="zh-BAC009S0729W0241", timing_blocks=9, replays_per_sample=10,
        timing_fragment_begin=0, timing_fragment_live_rows=64, timing_original_prompt_rows=165,
        timing_scope="One captured64-row component fragment, not a complete165-token forward; descriptive within-session intervals with independence unestablished",
        timing_order_seed=20261106, active_cell=None, model_qualified=False,
        native_selector_implemented=False, service_performance_qualified=False)
    def save():
        pending = args.output / "result.pending"
        pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        pending.replace(args.output / "result.json")
    save()
    random_order = random.Random(record["timing_order_seed"])
    torch.set_num_threads(2)
    suffixes = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    allocation_fields = ("allocation.all.allocated", "allocation.all.freed",
        "segment.all.allocated", "segment.all.freed")
    try:
        with torch.inference_mode(), fp32_reductions():
            if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 1):
                raise ValueError("one assigned SM121 GPU required")
            for root, case in cases:
                n = case["prefill_rows"]
                capacity = 256 if n <= 256 else 512
                if not 0 < n <= capacity:
                    raise ValueError("frozen prompt does not fit its declared measured bucket")
                with safe_open(root / (case["id"] + "-prefill.safetensors"), framework="pt") as frozen:
                    for layer in range(28):
                        for suffix in suffixes:
                            name = f"thinker.model.layers.{layer}." + suffix
                            record["active_cell"] = dict(case=case["id"], name=name, stage="construct")
                            owners, graphs = [], []
                            with retain_on_failed_drain(0, owners, graphs):
                                operand = frozen.get_tensor(name + ".input.0").reshape(n, -1).contiguous()
                                reference = frozen.get_tensor(name + ".output").reshape(n, -1).contiguous()
                                with safe_open(args.checkpoint / index[name + ".weight"], framework="pt") as shard:
                                    weight = shard.get_tensor(name + ".weight").cuda()
                                if operand.dtype != torch.bfloat16 or reference.dtype != torch.bfloat16 \
                                        or not torch.isfinite(operand).all() or not torch.isfinite(reference).all() \
                                        or weight.dtype != torch.bfloat16 or not torch.isfinite(weight).all():
                                    raise ValueError("actual independent BF16 operands must be finite")
                                source = torch.empty((64, operand.shape[1]), device="cuda", dtype=torch.bfloat16)
                                positions = torch.arange(64, device="cuda", dtype=torch.int64)
                                count = torch.zeros((1,), device="cuda", dtype=torch.int32)
                                sequence = torch.tensor([n], device="cuda", dtype=torch.int32)
                                owners.extend((source, positions, count, sequence, weight))
                                variants = []
                                for extent in (8192, capacity):
                                    staging = torch.empty((extent, weight.shape[1]), device="cuda", dtype=torch.bfloat16)
                                    projected = torch.empty((extent, weight.shape[0]), device="cuda", dtype=torch.bfloat16)
                                    guard = torch.full((64 * weight.shape[0] + 16,), 37., device="cuda", dtype=torch.bfloat16)
                                    output = guard[8:-8].view(64, weight.shape[0])
                                    operation = PreparedPositionedProjection(source, weight, output, count,
                                        positions, sequence, staging, projected, subrow_extents=(19, 27, 32, 64))
                                    owners.extend((staging, projected, guard, output, operation))
                                    for _ in range(3): operation(source)
                                    torch.cuda.synchronize()
                                    graph = torch.cuda.CUDAGraph(); graphs.append(graph)
                                    with torch.cuda.graph(graph): operation(source)
                                    variants.append((extent, graph, output, guard, staging, projected))
                                row = dict(case=case["id"], name=name, original_rows=n, candidate_rows=capacity,
                                    fragments=[], baseline_reference_changed=0, candidate_reference_changed=0,
                                    candidate_baseline_changed=0, timings=[])
                                record["observations"].append(row)
                                for begin in range(0, n, 64):
                                    live = min(64, n - begin)
                                    source.fill_(float("nan")); source[:live].copy_(operand[begin:begin + live])
                                    positions.copy_(torch.arange(begin, begin + 64, device="cuda")); count.fill_(live)
                                    results = []
                                    for extent, graph, output, guard, staging, projected in variants:
                                        prior = None
                                        for repeat in range(5):
                                            record["active_cell"] = dict(case=case["id"], name=name,
                                                extent=extent, begin=begin, repeat=repeat, stage="captured-check")
                                            guard.fill_(37); staging.fill_(float("nan")); projected.fill_(float("nan"))
                                            source[live:].fill_(float("inf") if repeat % 2 else float("nan"))
                                            torch.cuda.synchronize(); before = torch.cuda.memory_stats()
                                            graph.replay(); torch.cuda.synchronize(); after = torch.cuda.memory_stats()
                                            actual = output[:live].cpu().contiguous()
                                            if any(before[key] != after[key] for key in allocation_fields) \
                                                    or not torch.isfinite(actual).all() or not output[live:].eq(0).all() \
                                                    or not guard[:8].eq(37).all() or not guard[-8:].eq(37).all() \
                                                    or prior is not None and not torch.equal(actual.view(torch.uint8), prior.view(torch.uint8)):
                                                raise ValueError("capture repeat, finite, inactive, guards or allocation control failed")
                                            prior = actual
                                        results.append(prior)
                                    target = reference[begin:begin + live]
                                    changes = [int((value.view(torch.int16) != target.view(torch.int16)).sum()) for value in results]
                                    changed = int((results[0].view(torch.int16) != results[1].view(torch.int16)).sum())
                                    row["baseline_reference_changed"] += changes[0]
                                    row["candidate_reference_changed"] += changes[1]
                                    row["candidate_baseline_changed"] += changed
                                    row["fragments"].append(dict(begin=begin, live=live,
                                        candidate_baseline_changed=changed, identical_replays=5, allocation_stable=True))
                                    if changed: raise ValueError("candidate prompt extent changes baseline published BF16 bytes")
                                if case["id"] == record["timing_case"]:
                                    source.copy_(operand[:64]); positions.copy_(torch.arange(64, device="cuda")); count.fill_(64)
                                    for block in range(9):
                                        order = list(range(2)); random_order.shuffle(order); values = {}
                                        for variant in order:
                                            extent, graph = variants[variant][:2]
                                            record["active_cell"] = dict(case=case["id"], name=name,
                                                extent=extent, block=block, stage="timing")
                                            for _ in range(3): graph.replay()
                                            torch.cuda.synchronize()
                                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                            owners.extend((start, end)); before = torch.cuda.memory_stats(); start.record()
                                            for _ in range(10): graph.replay()
                                            end.record(); end.synchronize(); after = torch.cuda.memory_stats()
                                            if any(before[key] != after[key] for key in allocation_fields):
                                                raise ValueError("timed replay changed allocations")
                                            values[variant] = start.elapsed_time(end) * 1000 / 10
                                        row["timings"].append(dict(block=block, order=order,
                                            baseline_us=values[0], candidate_us=values[1]))
                                    row["component_interval"] = component_interval([
                                        (item["baseline_us"], item["candidate_us"]) for item in row["timings"]])
                                torch.cuda.synchronize()
                                for graph in graphs: graph.reset()
                            # Do not retain retired CUDA buffers across projection cells.
                            del owners, graphs, variants, operation, graph, source, positions, count, sequence
                            del staging, projected, guard, output, weight
                        save()
            if len(record["observations"]) != record["planned_projections"]:
                raise ValueError("incomplete planned projection denominator")
            record.update(status="complete_component_geometry_only", active_cell=None)
    except BaseException as error:
        record["status"] = "failed_or_interrupted"
        record["errors"].append(dict(type=type(error).__name__, message=str(error), active_cell=record["active_cell"]))
        try: save()
        except BaseException as reporting_error:
            error.add_note(f"could not publish prompt-capacity failure: {reporting_error!r}")
        raise
    else:
        save()


if __name__ == "__main__":
    main()
