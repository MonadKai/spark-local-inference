# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Measure fixed-reduction column tiles without changing production candidates.

Use original checkpoint weights and independent represented operands. Time graphs
rotating every decoder layer's weights to avoid a single small matrix remaining
in cache. These are component diagnostics, never complete ASR throughput or
evidence that an unqualified tactic may enter production autotuning.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    from ignition.planning import discover_gpus
    apply(Configuration.load(args.config), discover_gpus())

    import torch
    from safetensors import safe_open
    from b12x._lib.compile_plan import load_programs, retain_compiled_programs
    from ignition.components.reduction_projection import CONTRACT, Query, State, compile_projection
    from ignition.preparation.failure import retain_on_failed_drain, drain_or_retain, retain_undrained
    from ignition.preparation.module_images import ModuleImages
    from ignition.recipes.qwen3_asr.numerics import fp32_reductions
    from ignition.recipes.qwen3_asr.checkpoint import inspect_original_checkpoint

    torch.set_num_threads(2)
    inspect_original_checkpoint(args.checkpoint)
    reference = json.loads((args.reference / "result.json").read_bytes())
    if reference["errors"] or reference["candidate_executed"] \
            or reference["transformers_revision"] != "1d9654e78a23fb167ef539cbd699dfbd3e4830cf":
        raise ValueError("complete independent pinned represented operands required")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_bytes())["weight_map"]
    args.output.mkdir(exist_ok=False)
    tactics = ((4, 4), (2, 4), (8, 4), (16, 4), (8, 8), (16, 8),
        (32, 4), (32, 8), (64, 4), (64, 8))
    record = dict(scope=__doc__, status="preparing", source_case="en-422-122949-0013-first-2f",
        expected_checkpoint_revision="7278e1e70fe206f11671096ffdd38061171dd6e5",
        checkpoint_content_identity_verified_by_this_tool=False,
        torch=torch.__version__, cuda_runtime=torch.version.cuda,
        numerical_control="all output bytes versus existing (4,4) fixed-reduction tactic on identical represented operands; not an independent arithmetic oracle",
        full_model_numerical_qualification=False, production_candidates_changed=False,
        physical_rows=8, logical_rows=[1, 2, 4, 8], timing_blocks=9,
        graph_replays_per_sample=10, order_seed=20261106,
        timings="CUDA events per projection in a graph rotating actual layer weights; no serving comparison",
        candidates=[], errors=[], active_candidate=None,
        checkpoint_names_shapes_and_dtypes_validated=True)
    def save():
        (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    save()
    owners, graphs, cases = [], [], []
    with (retain_on_failed_drain(0, owners, graphs), retain_compiled_programs() as programs,
            torch.inference_mode(), fp32_reductions(), ModuleImages() as images):
        owners.extend((programs, images))
        try:
            torch.cuda.set_device(0)
            if torch.cuda.get_device_capability() != (12, 1):
                raise ValueError("assigned SM121 GPU required")
            record["compute_capability"] = list(torch.cuda.get_device_capability())
            record["multiprocessors"] = torch.cuda.get_device_properties(0).multi_processor_count
            with safe_open(args.reference / (record["source_case"] + "-prefill.safetensors"), framework="pt") as frozen:
                for operation, parts in (("qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
                        ("attention_output", ("self_attn.o_proj",)),
                        ("gate_up", ("mlp.gate_proj", "mlp.up_proj")),
                        ("down", ("mlp.down_proj",)), ("head", ("head",)),
                        ("query", ("self_attn.q_proj",)), ("key", ("self_attn.k_proj",)),
                        ("value", ("self_attn.v_proj",)), ("gate", ("mlp.gate_proj",)),
                        ("up", ("mlp.up_proj",))):
                    weights = []
                    expected_shape = dict(qkv=(4096, 2048), attention_output=(2048, 2048),
                        gate_up=(12288, 2048), down=(2048, 6144), head=(151936, 2048),
                        query=(2048, 2048), key=(1024, 2048), value=(1024, 2048),
                        gate=(6144, 2048), up=(6144, 2048))[operation]
                    for layer in range(1 if operation == "head" else 28):
                        pieces = []
                        for part in parts:
                            name = "thinker.lm_head.weight" if part == "head" else f"thinker.model.layers.{layer}.{part}.weight"
                            with safe_open(args.checkpoint / index[name], framework="pt") as shard:
                                pieces.append(shard.get_tensor(name).clone())
                        weight = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
                        if weight.dtype != torch.bfloat16 or tuple(weight.shape) != expected_shape \
                                or not torch.isfinite(weight).all():
                            raise ValueError("invalid original weight geometry, dtype or finiteness")
                        weights.append(weight.to("cuda"))
                        owners.append(weights[-1])
                    width, columns = weights[0].shape[1], weights[0].shape[0]
                    module = "thinker.lm_head" if operation == "head" else "thinker.model.layers.0." + parts[0]
                    represented = frozen.get_tensor(module + ".input.0")
                    if represented.dtype != torch.bfloat16 or represented.ndim != 3 \
                            or represented.shape[0] != 1 or represented.shape[-1] != width \
                            or represented.shape[1] < 8 or not torch.isfinite(represented).all():
                        raise ValueError("independent input geometry or finiteness differs")
                    rows = list(range(represented.shape[1] - 1, represented.shape[1] - 9, -1))
                    basis = represented[0, rows].contiguous()
                    for part in parts[1:]:
                        other = frozen.get_tensor("thinker.model.layers.0." + part + ".input.0")
                        if other.shape != represented.shape or other.dtype != represented.dtype \
                                or not torch.equal(other[0, rows].contiguous().view(torch.uint8), basis.view(torch.uint8)):
                            raise ValueError("packed projections do not share the same represented input")
                    source = basis.to("cuda")
                    guard = torch.full((8 * columns + 16,), 137, dtype=torch.bfloat16, device="cuda")
                    output = guard[8:-8].view(8, columns)
                    count = torch.full((1,), 8, dtype=torch.int32, device="cuda")
                    owners.extend((source, guard, count))
                    query = Query(columns, width)
                    payload = CONTRACT.encode_query(query)
                    case = dict(operation=operation, source=source, basis=basis, weights=weights,
                        guard=guard, output=output, count=count, states=[], graphs=[])
                    cases.append(case)
                    for tactic in tactics:
                        record["active_candidate"] = dict(operation=operation, query=asdict(query), tactic=list(tactic))
                        save()
                        program = compile_projection(payload, tactic, 0)
                        owners.append(program)
                        load_programs(program)
                        state = State(query, tactic, program)
                        owners.append(state)
                        for weight in weights:
                            state.run(source, weight, output, count)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        graphs.append(graph)
                        with torch.cuda.graph(graph):
                            for weight in weights:
                                state.run(source, weight, output, count)
                        case["states"].append(state)
                        case["graphs"].append(graph)
                        row = dict(operation=operation, query=asdict(query), tactic=list(tactic),
                            composition_matches_current_decode=operation not in ("qkv", "gate_up", "head"),
                            source_rows=rows, includes_prefill_first_token=operation == "head",
                            rotated_weight_sets=len(weights), weight_payload_bytes=sum(weight.nbytes for weight in weights),
                            loaded_kernel=dict(registers_per_thread=int(program.n_regs),
                                local_bytes_per_thread=int(program.n_spills) * 4,
                                maximum_threads_per_block=int(program.n_max_threads),
                                shared_memory_bytes=int(program.metadata.shared)),
                            observations=[], timings=[], admitted_to_production=False)
                        record["candidates"].append(row)
                        case.setdefault("rows", []).append(row)
                    save()
            # Resolve/compile/capture every candidate before any timed sample.
            record["status"] = "checking"
            save()
            for case in cases:
                source, guard, output, count = (case[name] for name in ("source", "guard", "output", "count"))
                for logical in (1, 2, 4, 8):
                    source.copy_(case["basis"])
                    source[logical:].fill_(float("nan"))
                    count.fill_(logical)
                    baseline = []
                    for index, (state, row) in enumerate(zip(case["states"], case["rows"], strict=True)):
                        record["active_candidate"] = dict(operation=case["operation"], query=row["query"],
                            tactic=row["tactic"], logical_rows=logical, stage="checking")
                        exact, guards, finite = True, True, True
                        for layer, weight in enumerate(case["weights"]):
                            guard.fill_(137)
                            state.run(source, weight, output, count)
                            torch.cuda.synchronize()
                            actual = output.cpu().clone()
                            finite &= bool(torch.isfinite(actual).all())
                            guards &= bool(guard[:8].eq(137).all() and guard[-8:].eq(137).all() and actual[logical:].eq(0).all())
                            if index == 0:
                                baseline.append(actual)
                            else:
                                exact &= torch.equal(actual.view(torch.uint8), baseline[layer].view(torch.uint8))
                        row["observations"].append(dict(logical_rows=logical, exact_baseline_bytes=exact,
                            finite=finite, guards_and_inactive_rows=guards))
                        if not exact or not guards or not finite:
                            raise ValueError("candidate arithmetic/control mismatch: " + str(row["tactic"]))
                        # Gate the actual rotating graph used for timing, rather
                        # than inferring its binding correctness from eager run.
                        graph = case["graphs"][index]
                        for repeat in range(5):
                            guard.fill_(137)
                            torch.cuda.synchronize()
                            before = torch.cuda.memory_stats()
                            graph.replay()
                            torch.cuda.synchronize()
                            after = torch.cuda.memory_stats()
                            actual = output.cpu().clone()
                            exact = torch.equal(actual.view(torch.uint8), baseline[-1].view(torch.uint8))
                            finite = bool(torch.isfinite(actual).all())
                            guards = bool(guard[:8].eq(137).all() and guard[-8:].eq(137).all()
                                and actual[logical:].eq(0).all())
                            stable = all(before[key] == after[key] for key in ("allocation.all.allocated",
                                "allocation.all.freed", "segment.all.allocated", "segment.all.freed"))
                            row["observations"].append(dict(logical_rows=logical, captured_repeat=repeat,
                                exact_last_weight_baseline_bytes=exact, finite=finite,
                                guards_and_inactive_rows=guards, torch_allocation_stable=stable))
                            if not exact or not guards or not finite or not stable:
                                raise ValueError("captured candidate binding/arithmetic/control mismatch: " + str(row["tactic"]))
                save()
            record["status"] = "timing"
            save()
            order = random.Random(record["order_seed"])
            for case in cases:
                source, count = case["source"], case["count"]
                source.copy_(case["basis"])
                cells = [(logical, index) for logical in (1, 2, 4, 8) for index in range(len(tactics))]
                for block in range(record["timing_blocks"]):
                    order.shuffle(cells)
                    for logical, index in cells:
                        record["active_candidate"] = dict(operation=case["operation"],
                            tactic=list(tactics[index]), logical_rows=logical, timing_block=block)
                        count.fill_(logical)
                        graph = case["graphs"][index]
                        for _ in range(3):
                            graph.replay()
                        torch.cuda.synchronize()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        owners.extend((begin, end))
                        before = torch.cuda.memory_stats()
                        begin.record()
                        for _ in range(record["graph_replays_per_sample"]):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        after = torch.cuda.memory_stats()
                        for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                            if before[key] != after[key]:
                                raise ValueError("candidate replay changed GPU allocations")
                        case["rows"][index]["timings"].append(dict(block=block, logical_rows=logical,
                            microseconds_per_projection=begin.elapsed_time(end) * 1000 /
                                (record["graph_replays_per_sample"] * len(case["weights"])),
                            torch_allocation_stable=True))
                    save()
            torch.cuda.synchronize()
            for graph in graphs:
                graph.reset()
        except BaseException as error:
            record.update(status="failed_or_interrupted")
            record["errors"].append(dict(type=type(error).__name__, message=str(error)))
            save()
            if drain_or_retain(error, 0, owners, graphs):
                try:
                    for graph in graphs:
                        graph.reset()
                except BaseException as cleanup_error:
                    if not drain_or_retain(cleanup_error, 0, owners, graphs):
                        retain_undrained(error, owners, graphs)
                    error.add_note(f"candidate graph cleanup failed: {cleanup_error!r}")
            raise
    record["status"] = "complete_component_diagnostic_only"
    save()


if __name__ == "__main__":
    main()
