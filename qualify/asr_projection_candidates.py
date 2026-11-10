# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Measure existing b12x matrix candidates on actual ASR projection geometry.

Overrides enumerate diagnostic candidates only; production selection is not
changed. Rotate real layer weights for timing. Numerical controls use frozen
layer-zero operands, including the first-token head input. These are kernel
diagnostics, not complete ASR throughput, numeric admission or a serving win.
"""

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
from pathlib import Path


def retained_call(state, x, weights, y, owners):
    """Publish native launch owners before a raw session can prime the call."""
    from b12x.preparation import PreparedCall

    owners.append(state)
    call = PreparedCall(run=lambda: state.run(x, weights[0], out=y), owners=(state, x, y, weights))
    owners.append(call)
    return call


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
    from safetensors.torch import save_file
    from b12x.gemm import bf16_gemv
    from b12x.gemm.bf16_gemv._tuning import TUNING
    from b12x.preparation import PreparationSession
    from b12x._lib.compile_plan import retain_compiled_programs
    from ignition.preparation.failure import retain_on_failed_drain, resources_retained
    from ignition.preparation.module_images import ModuleImages
    from ignition.recipes.qwen3_asr.numerics import fp32_reductions
    from asr_linear_bounds import envelope, result_ratio

    torch.set_num_threads(2)
    dependency = importlib.metadata.distribution("flashinfer-python")
    source_identity = json.loads(dependency.read_text("direct_url.json") or "{}")
    flashinfer_revision = source_identity.get("vcs_info", {}).get("commit_id")
    if flashinfer_revision != "1f8ed8a3c46e27d03cff54e08f23e9bc50476d57":
        raise ValueError("pinned b12x source distribution required")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_bytes())["weight_map"]
    reference_record = json.loads((args.reference / "result.json").read_bytes())
    if reference_record["errors"] or reference_record["candidate_executed"] \
            or reference_record["transformers_revision"] != "1d9654e78a23fb167ef539cbd699dfbd3e4830cf":
        raise ValueError("complete independent pinned operands required")
    args.output.mkdir(exist_ok=False)
    result = dict(scope=__doc__, status="preparing", candidates=[], ineligible=[], errors=[],
        torch=torch.__version__, expected_checkpoint_revision="7278e1e70fe206f11671096ffdd38061171dd6e5",
        checkpoint_content_identity_verified_by_this_tool=False,
        flashinfer=dependency.version, flashinfer_revision=flashinfer_revision,
        cuda_runtime=torch.version.cuda, compute_capability=[12, 1],
        conditional_rne_fp32_envelope_only=True, arbitrary_mma_arithmetic_qualified=False,
        timings="CUDA events around repeated CUDA graphs; rotating actual layer weights; no serving comparison",
        source_case="en-422-122949-0013-first-2f", timing_repetitions=5, graph_replays_per_sample=10)

    def publish():
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    publish()
    owners, graphs, declarations, prepared_cases = [], [], [], []
    error = None
    with (retain_on_failed_drain(0, owners, graphs), retain_compiled_programs() as programs,
            torch.inference_mode(), fp32_reductions(), ModuleImages() as images):
        owners.extend((programs, images))
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability() != (12, 1):
            raise ValueError("assigned SM121 GPU required")
        session = PreparationSession(device=torch.device("cuda:0"), compile_workers=2, autotune=False)
        owners.append(session)
        try:
            with retain_on_failed_drain(0, owners, graphs):
                with safe_open(args.reference / (result["source_case"] + "-prefill.safetensors"), framework="pt") as frozen:
                    for operation, parts in (("qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
                            ("attention_output", ("self_attn.o_proj",)),
                            ("gate_up", ("mlp.gate_proj", "mlp.up_proj")), ("down", ("mlp.down_proj",)),
                            ("head", ("head",))):
                        weights, first_cpu = [], None
                        for layer in range(1 if operation == "head" else 28):
                            pieces = []
                            for part in parts:
                                name = "thinker.lm_head.weight" if part == "head" else f"thinker.model.layers.{layer}.{part}.weight"
                                with safe_open(args.checkpoint / index[name], framework="pt") as shard:
                                    pieces.append(shard.get_tensor(name).clone())
                            packed = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
                            if packed.dtype != torch.bfloat16 or not torch.isfinite(packed).all():
                                raise ValueError("invalid original projection weight")
                            if layer == 0:
                                first_cpu = packed
                            weight = packed.to("cuda")
                            weights.append(weight)
                            owners.append(weight)
                        module = "thinker.lm_head" if operation == "head" else "thinker.model.layers.0." + parts[0]
                        source = frozen.get_tensor(module + ".input.0").reshape(-1, first_cpu.shape[1])
                        row_indices = list(range(len(source) - 1, len(source) - 9, -1))
                        basis = source[row_indices].contiguous()
                        for part in parts[1:]:
                            other = frozen.get_tensor("thinker.model.layers.0." + part + ".input.0").reshape(-1, basis.shape[1])
                            if not torch.equal(basis.view(torch.uint8), other[row_indices].contiguous().view(torch.uint8)):
                                raise ValueError("packed projections do not share an identical input node")
                        columns = first_cpu.shape[0]
                        channels = sorted({0, 1, columns // 4, columns // 2, columns - 2, columns - 1})
                        exact, bound, _, _ = envelope(basis, first_cpu[channels], None)
                        regimes = (("decode", 8),) if operation == "head" else (("decode", 8), ("prefill", 4096))
                        for regime, capacity in regimes:
                            x = torch.zeros((capacity, basis.shape[1]), dtype=torch.bfloat16, device="cuda")
                            owners.append(x)
                            x[:8].copy_(basis)
                            guard = torch.full((capacity * columns + 32,), 137, dtype=torch.bfloat16, device="cuda")
                            owners.append(guard)
                            y = guard[16:-16].view(capacity, columns)
                            query = replace(bf16_gemv.query_from_call(x, weights[0], out=y), autotune_effort="audit")
                            configs = [bf16_gemv.GemvConfig(backend=backend) for backend in ("torch", "mma", "tc", "prefill")]
                            configs += [bf16_gemv.GemvConfig(backend="simt", rows_per_tile=rows, block_threads=threads)
                                for rows in (1, 2, 4, 8) for threads in (128, 256)]
                            for config in configs:
                                name = f"{operation}-{regime}-{config.backend}-{config.rows_per_tile}-{config.block_threads}"
                                try:
                                    TUNING.validate_config(query, config, None)
                                except ValueError as invalid:
                                    result["ineligible"].append(dict(name=name, reason=str(invalid)))
                                    continue
                                plan = bf16_gemv.plan(query, override=config)
                                owners.append(plan)
                                def bind(state, x=x, weights=weights, y=y):
                                    return retained_call(state, x, weights, y, owners)
                                declarations.append(plan.request(name=name, prepare_call=bind))
                                prepared_cases.append(dict(name=name, operation=operation, regime=regime,
                                    config=config, query=query, plan=plan, x=x, y=y, guard=guard, weights=weights,
                                    basis=basis, exact=exact, bound=bound, channels=channels, source_rows=row_indices))
                publish()
                prepared = session.prepare(tuple(declarations), autotune=False)
                prepared.close()
                result["status"] = "running"
                publish()
                for case in prepared_cases:
                    x, y, guard, plan, weights = (case[name] for name in ("x", "y", "guard", "plan", "weights"))
                    if plan.selection.config != case["config"]:
                        raise ValueError("diagnostic override did not select the requested candidate")
                    row = dict(name=case["name"], operation=case["operation"], regime=case["regime"],
                        query=asdict(case["query"]), selected=asdict(plan.selection.config), source_rows=case["source_rows"],
                        rotated_weight_sets=len(weights), weight_payload_bytes=sum(w.nbytes for w in weights),
                        observations=[], timings=[], failed=False)
                    result["candidates"].append(row)
                    anchor = None
                    sizes = (8, 1, 2, 4) if case["regime"] == "decode" else (8, 19, 31, 64, 128, 256, 512, 4096)
                    for physical in sizes:
                        for offset in (0, 1):
                            if offset >= physical:
                                continue
                            live = min(8, physical - offset)
                            x.fill_(float("nan"))
                            x[offset:offset + live].copy_(case["basis"][:live])
                            guard.fill_(137)
                            bf16_gemv.mm(x[:physical], weights[0], out=y[:physical], plan=plan)
                            torch.cuda.synchronize()
                            actual = y[offset:offset + live].cpu().clone()
                            if not torch.isfinite(actual).all() or not guard[:16].eq(137).all() \
                                    or not guard[-16:].eq(137).all() or not y[physical:].eq(137).all():
                                raise ValueError("projection nonfinite or output guard corruption: " + case["name"])
                            if anchor is None:
                                anchor = actual
                                save_file(dict(output=anchor), args.output / (case["name"] + "-anchor.safetensors"))
                            stable = torch.equal(actual.view(torch.uint8), anchor[:live].contiguous().view(torch.uint8))
                            ratio = result_ratio(actual[:, case["channels"]], case["exact"][:live], case["bound"][:live])
                            wrong = (actual[:, case["channels"]].double() + 4 * case["bound"][:live]).bfloat16()
                            negative = result_ratio(wrong, case["exact"][:live], case["bound"][:live]) > 1
                            row["observations"].append(dict(physical_rows=physical, offset=offset, live_rows=live,
                                exact_to_same_candidate_eight_rows=stable, conditional_envelope_ratio=ratio,
                                wrong_output_rejected=negative, guards_intact=True))
                            row["failed"] |= not stable or ratio > 1 or not negative
                    if case["regime"] == "decode":
                        for logical in (1, 2, 4, 8):
                            for physical in sorted({logical, 8}):
                                x.zero_()
                                x[:logical].copy_(case["basis"][:logical])
                                for weight in weights:
                                    bf16_gemv.mm(x[:physical], weight, out=y[:physical], plan=plan)
                                torch.cuda.synchronize()
                                graph = torch.cuda.CUDAGraph()
                                graphs.append(graph)
                                with session.capture(), torch.cuda.graph(graph):
                                    for weight in weights:
                                        bf16_gemv.mm(x[:physical], weight, out=y[:physical], plan=plan)
                                for _ in range(5):
                                    graph.replay()
                                torch.cuda.synchronize()
                                allocated = torch.cuda.memory_stats()
                                samples = []
                                for _ in range(5):
                                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                    owners.extend((begin, end))
                                    begin.record()
                                    for _ in range(10):
                                        graph.replay()
                                    end.record()
                                    end.synchronize()
                                    samples.append(begin.elapsed_time(end) * 1000 / (10 * len(weights)))
                                after = torch.cuda.memory_stats()
                                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                                    if allocated[key] != after[key]:
                                        raise ValueError("projection replay allocated GPU tensors")
                                row["timings"].append(dict(logical_rows=logical, physical_rows=physical,
                                    microseconds_per_projection=samples, torch_allocation_stable=True))
                                graph.reset()
                    publish()
                torch.cuda.synchronize()
        except BaseException as failure:
            error = failure
            result["errors"].append(repr(failure))
            result["status"] = "interrupted"
            try:
                publish()
            except BaseException as reporting_error:
                failure.add_note(f"could not publish candidate failure: {reporting_error!r}")
            raise
        finally:
            if error is None or not resources_retained(error):
                try:
                    for graph in graphs:
                        graph.reset()
                    session.close()
                except BaseException as cleanup_error:
                    if error is None:
                        raise
                    error.add_note(f"candidate cleanup failed after confirmed drain: {cleanup_error!r}")
    result["status"] = "complete_diagnostic_with_rejections" if any(c["failed"] for c in result["candidates"]) else "complete_diagnostic_only"
    publish()


if __name__ == "__main__":
    main()
