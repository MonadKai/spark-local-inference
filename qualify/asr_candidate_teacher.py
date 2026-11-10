# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inspect production captured ASR graph logits against frozen independent rows.

This retains the Python preparer to inspect existing scratch. It is numerical
diagnosis, never native serving or comparative performance evidence. The same
recipe prepares the audio, prefill and decode graphs; no model forward or graph
is replaced. Requests use the documented bootstrap admission uploads. Natural
stopping, HTTP, asynchronous cancellation and exported Rust replay require their
separate checks.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import traceback
from time import monotonic


@contextmanager
def diagnostic_signals():
    """Turn inherited ignored termination signals into fenced cleanup."""
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(signum, frame):
        # Repeated requests must not interrupt the owner's drain/close path.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        # Ordinary per-case reporting catches Exception and continues. Process
        # termination must bypass that path and enter the outer fenced cleanup.
        raise KeyboardInterrupt("diagnostic received signal " + str(signum))
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def preparation_context(mode):
    """Choose actual tensor dispatch identity, independently of the caller's mode."""
    import torch

    if mode not in ("inference", "no-grad"):
        raise ValueError("unknown diagnostic preparation mode")
    with torch.inference_mode(mode == "inference"), torch.no_grad():
        yield


def save_teacher_traces(traces, output, label, *, suffix=""):
    """Save completed CPU observations, including failing/nonfinite rows."""
    import torch
    from safetensors.torch import save_file

    saved = []
    for identity, rows in traces.items():
        if isinstance(rows, list):
            if not rows:
                continue
            if any(row.device.type != "cpu" for row in rows):
                raise ValueError("partial teacher evidence must already be on the CPU")
            rows = torch.stack(rows)
        if rows.device.type != "cpu":
            raise ValueError("teacher evidence must already be on the CPU")
        name = f"{label}-{identity}{suffix}.safetensors"
        save_file({"logits_bf16": rows}, output / name)
        saved.append(dict(id=identity, file=name, rows=len(rows)))
    return saved


def save_failed_group(traces, output, label, drained, state):
    """An undrained error may save CPU rows but must never read device state."""
    from safetensors.torch import save_file

    result = dict(traces=save_teacher_traces(traces, output, label, suffix="-partial"),
        device_state="unavailable_unconfirmed_drain")
    if drained:
        values = state()
        name = label + "-failure-state.safetensors"
        save_file(values, output / name)
        result.update(device_state="saved_after_confirmed_drain", state_file=name)
    return result


def compare_row(actual, reference, budget, top1):
    import torch

    if actual.dtype != torch.bfloat16 or reference.dtype != torch.bfloat16 or actual.shape != reference.shape:
        raise ValueError("teacher rows require equal BF16 shapes")
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("nonfinite teacher logits")
    a, b = actual.double(), reference.double()
    delta = a - b
    absolute = delta.abs().max().item()
    relative = (delta.norm() / b.norm().clamp_min(1e-300)).item()
    decision = actual.argmax().item()
    def margin(value):
        values = value.reshape(-1).double().topk(min(2, value.numel())).values
        return (values[0] - values[1]).item() if len(values) == 2 else None
    return dict(max_abs=absolute, relative_l2=relative, top1=decision, reference_top1=top1,
        candidate_top1_logit=actual.reshape(-1)[decision].item(),
        reference_top1_logit=reference.reshape(-1)[top1].item(),
        candidate_logit_at_reference_top1=actual.reshape(-1)[top1].item(),
        reference_logit_at_candidate_top1=reference.reshape(-1)[decision].item(),
        candidate_margin=margin(actual), reference_margin=margin(reference),
        passed=absolute <= budget["max_abs"] and relative <= budget["relative_l2"] and decision == top1)


def validate_frozen_contract(frozen):
    from ignition.recipes.qwen3_asr.numerics import POLICY

    if frozen["numerical_policy"] != POLICY or frozen["candidate_used_for_budgets"] \
            or frozen["generation_decision_top1_mismatches_allowed"] != 0:
        raise ValueError("a frozen independent zero-top1-mismatch contract is required")


def compare_trace(actual, reference, case, baseline, *, retired_after=None):
    import torch

    total = len(case["teacher_rows"])
    required = total if retired_after is None else min(total, retired_after)
    if not 0 < required <= total or actual.shape[1:] != reference.shape[1:] or len(actual) > total:
        raise ValueError("invalid teacher trace geometry")
    metrics = [compare_row(value, reference[i], case["teacher_rows"][i], case["teacher_top1"][i])
        for i, value in enumerate(actual)]
    stable = torch.equal(actual.view(torch.uint8), baseline[:len(actual)].view(torch.uint8))
    return dict(rows=len(actual), expected_rows=total, required_rows=required,
        retired_after_rows=retired_after, complete_teacher_trace=len(actual) == total,
        metrics=metrics, same_path_exact_bytes=stable,
        passed=len(actual) == required and stable and all(metric["passed"] for metric in metrics))


def compare_state(actual, baseline):
    """Reject incomplete, nonfinite or changed meaningful state on the CPU."""
    import torch

    if not actual or actual.keys() != baseline.keys():
        raise ValueError("state observation fields differ")
    fields = {}
    for name, value in actual.items():
        expected = baseline[name]
        if value.device.type != "cpu" or expected.device.type != "cpu" \
                or value.dtype != expected.dtype or value.shape != expected.shape or not value.numel():
            raise ValueError("state observation geometry differs: " + name)
        if not torch.isfinite(value).all() or not torch.isfinite(expected).all():
            raise ValueError("nonfinite state observation: " + name)
        exact = torch.equal(value.contiguous().reshape(-1).view(torch.uint8),
            expected.contiguous().reshape(-1).view(torch.uint8))
        fields[name] = dict(shape=list(value.shape), dtype=str(value.dtype), exact_bytes=exact,
            differing_elements=(value != expected).sum().item())
    return dict(passed=all(row["exact_bytes"] for row in fields.values()), fields=fields)


def compare_prefill(actual, reference, case):
    """Apply the frozen whole-matrix scalar budget and report every top-1 gap."""
    import torch

    budget = case["stages"]["thinker.lm_head.output"]
    shape = (1, len(case["prompt_ids"]), actual.shape[-1])
    if tuple(budget["shape"]) != shape or tuple(reference.shape) != shape \
            or tuple(actual.shape) != shape[1:] \
            or actual.dtype != torch.bfloat16 or reference.dtype != torch.bfloat16:
        raise ValueError("full-prefill BF16 geometry differs from the frozen matrix")
    reference = reference[0]
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("nonfinite full-prefill logits")
    a, b = actual.double(), reference.double()
    delta = a - b
    absolute = delta.abs().max().item()
    relative = (delta.norm() / b.norm().clamp_min(1e-300)).item()
    candidate_ids, reference_ids = actual.argmax(-1), reference.argmax(-1)
    mismatches = []
    for position in (candidate_ids != reference_ids).nonzero().reshape(-1).tolist():
        candidate, expected = int(candidate_ids[position]), int(reference_ids[position])
        def decision(row, first):
            remaining = row.clone()
            remaining[first] = -float("inf")
            second = remaining.argmax().item()
            return dict(first_id=first, second_id=second, first_logit=row[first].item(),
                second_logit=row[second].item(), margin=(row[first].double() - row[second].double()).item())
        mismatches.append(dict(position=position, candidate=decision(actual[position], candidate),
            reference=decision(reference[position], expected),
            candidate_logit_at_reference_top1=actual[position, expected].item(),
            reference_logit_at_candidate_top1=reference[position, candidate].item()))
    last = compare_row(actual[-1], reference[-1], case["teacher_rows"][0], case["teacher_top1"][0])
    return dict(shape=list(actual.shape), max_abs=absolute, relative_l2=relative,
        frozen_budget=budget, top1_mismatches=mismatches, top1_mismatch_count=len(mismatches),
        top1_ties="lowest ID for both first and second; no margin exclusions",
        generation_decision=last,
        passed=absolute <= budget["max_abs"] and relative <= budget["relative_l2"] and last["passed"])


def load_reference_tensors(references, *, inference_mode, grad_enabled):
    """Read unchanged independent CPU tensors after compiler owners have retired."""
    import torch
    from safetensors import safe_open
    if __package__:
        from .asr_freeze_numerics import validate_identical_features
    else:
        from asr_freeze_numerics import validate_identical_features

    inputs, expected, encoder_reference, encoder_stages = {}, {}, {}, {}
    with torch.inference_mode(inference_mode), torch.set_grad_enabled(grad_enabled):
        for case, sdk_root, canonical_root in references:
            identity, target_rows = case["id"], len(case["teacher_rows"])
            with safe_open(sdk_root / (identity + "-prefill.safetensors"), framework="pt") as file:
                stored = file.get_tensor("processor.input_features").squeeze(0).contiguous()
                original_convolution = file.get_tensor("thinker.audio_tower.conv2d1.input.0")
            if stored.shape != (128, case["frames"]) or stored.dtype != torch.bfloat16 or not torch.isfinite(stored).all():
                raise ValueError("SDK Mel geometry or precision differs")
            # The independent stage reference records the BF16 model input. Exact
            # widening preserves that node through this graph's FP32 upload ABI.
            inputs[identity] = stored.float()
            with safe_open(canonical_root / (identity + "-prefill.safetensors"), framework="pt") as file:
                validate_identical_features(file.get_tensor("thinker.audio_tower.conv2d1.input.0"), original_convolution)
                encoder_reference[identity] = file.get_tensor("thinker.audio_tower.proj2.output")
                names = ["thinker.audio_tower.conv_out.output", "thinker.audio_tower.proj1.output", "thinker.audio_tower.proj2.output"]
                for layer in range(24):
                    prefix = f"thinker.audio_tower.layers.{layer}"
                    names += [prefix + f".self_attn.{field}.output" for field in ("q_proj", "k_proj", "v_proj", "out_proj")]
                    names += [prefix + ".self_attn.out_proj.input.0", prefix + ".fc1.output", prefix + ".fc2.output"]
                encoder_stages[identity] = {name: file.get_tensor(name) for name in names}
            with safe_open(canonical_root / (identity + "-cached-teacher.safetensors"), framework="pt") as file:
                expected[identity] = file.get_tensor("logits_bf16")
                if file.get_tensor("target_ids").reshape(-1).tolist() != case["teacher_history"]:
                    raise ValueError("canonical teacher history differs from frozen contract")
            if expected[identity].shape != (target_rows, 151936) or expected[identity].dtype != torch.bfloat16 \
                    or not torch.isfinite(expected[identity]).all():
                raise ValueError("canonical teacher logits geometry differs")
            if expected[identity].argmax(-1).tolist() != case["teacher_top1"]:
                raise ValueError("canonical top1 differs from frozen contract")

    return inputs, expected, encoder_reference, encoder_stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=2,
        help="bound preparation compiler processes, including diagnostic captures")
    parser.add_argument("--prefill-chunk-size", type=int, default=4096,
        help="explicit prepared prefill capacity; smaller diagnostic catalogs do not qualify larger capacities")
    parser.add_argument("--preparation-mode", choices=("inference", "no-grad"), default="inference",
        help="retain historical inference diagnosis or match ordinary CLI no-grad tensor identity")
    parser.add_argument("--snapshot-budget-bytes", type=int, default=256 << 20,
        help="explicit recipe continuation-storage budget; match the package under diagnosis")
    parser.add_argument("--preparation-headroom-bytes", type=int, default=12 << 30,
        help="explicit preparation headroom, including the recipe reserve")
    parser.add_argument("--retain-graphs", action="store_true",
        help="match ordinary export's retained graph templates during preparation")
    parser.add_argument("--contract", type=Path, default=Path("benchmarks/qwen3_asr/numerical-contract.json"))
    parser.add_argument("--sdk-observations", type=Path, action="append", required=True)
    parser.add_argument("--canonical-observations", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-inputs", action="store_true", help="validate independent rows without preparing or running the candidate")
    parser.add_argument("--decoder-encoder-features", choices=("captured", "canonical"), default="captured",
        help="canonical is attribution only: replace the encoder output with frozen independent features")
    parser.add_argument("--encoder-operator-probes", action="store_true",
        help="diagnose selected encoder operators and all 24 layers on the two minimal frozen inputs")
    parser.add_argument("--prefill-operator-probes", action="store_true",
        help="diagnose actual selected prefill projections with frozen operands")
    parser.add_argument("--decoder-operator-probes", action="store_true",
        help="after teacher collection, diagnose all 28 layers on actual frozen prefill/cached operands")
    parser.add_argument("--canonical-cached-observations", type=Path, action="append",
        help="independent cached-stage observations required by decoder-operator-probes")
    parser.add_argument("--paged-attention-probes", action="store_true",
        help="diagnose selected layer-0 paged operators with independently frozen Q/K/V")
    parser.add_argument("--torch-paged-attention-probes", action="store_true",
        help="diagnose canonical ATen paged operation with the same represented Q/K/V; not a serving path")
    parser.add_argument("--probe-repack-page-tokens", type=int, choices=(256,),
        help="repack only independent canonical paged probe operands; does not change the recipe/cache geometry")
    parser.add_argument("--prefix-kv-layers", type=int, choices=(2, 28), default=28,
        help="check this many prefix layers in every cell; two is an explicitly partial diagnostic")
    parser.add_argument("--prefill-capacity", choices=("smallest", "largest"), default="smallest",
        help="exercise a legal prepared prefill capacity; both remain production graph paths")
    parser.add_argument("--audio-capacity", choices=("smallest", "largest"), default="smallest",
        help="exercise a legal prepared audio capacity, including cost-refresh paths for short windows")
    parser.add_argument("--generation-error-probes", action="store_true",
        help="after teacher gates, fault actual decoder private KV and check failed-state/healthy-row isolation")
    parser.add_argument("--full-prefill", action="store_true",
        help="capture diagnostic residual observations, bridge all persistent bytes, and check every prompt logit")
    parser.add_argument("--prefill-network-nodes", action="store_true",
        help="also capture actual decoder module nodes and apply their existing frozen network budgets; local operator qualification remains separate")
    parser.add_argument("--encoder-network-nodes", action="store_true",
        help="also bridge and capture every frozen encoder module node; local operator qualification remains separate")
    parser.add_argument("--focus-network-nodes", action="store_true",
        help="diagnose every frozen input on B1 baseline only; never qualifies the complete teacher/state schedule matrix")
    args = parser.parse_args()
    if args.compile_workers < 1:
        parser.error("compile-workers must be positive")
    if args.snapshot_budget_bytes <= 0 or args.preparation_headroom_bytes < 0:
        parser.error("positive snapshot budget and nonnegative preparation headroom required")
    if args.prefill_network_nodes and not args.full_prefill:
        parser.error("prefill-network-nodes requires the full production-prefill bridge")
    if args.focus_network_nodes and not (args.prefill_network_nodes or args.encoder_network_nodes):
        parser.error("focus-network-nodes requires an actual network-node observer")
    if args.decoder_operator_probes != bool(args.canonical_cached_observations):
        parser.error("decoder-operator-probes and canonical-cached-observations must be supplied together")
    # Apply the ordinary explicit dependency configuration before importing any
    # preparation libraries, just as the CLI does.
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    from ignition.cli.preparation import Host

    configuration, gpus = Configuration.load(args.config), Host.discover().gpus
    if len(gpus) != 1:
        raise ValueError("assign one physical GPU to the teacher diagnostic")
    apply(configuration, gpus)

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from ignition.recipes.qwen3_asr.numerics import POLICY, fp32_reductions
    if __package__:
        from .asr_freeze_numerics import check, validate_case_layout, validate_identical_features
        from .asr_encoder_observation import observe as observe_encoder
        from .asr_encoder_observation import independent_operands
        from .asr_decoder_observation import independent_operands as prefill_operands, projection_count
        from .asr_decoder_observation import independent_attention
        from .asr_prefill_observation import PrefillObservation, bridge_buffers, compare_bridge, restore, check_decoder_nodes, check_head_nodes
        from .asr_audio_network_nodes import AudioObservation, check_encoder_nodes
    else:
        from asr_freeze_numerics import check, validate_case_layout, validate_identical_features
        from asr_encoder_observation import observe as observe_encoder
        from asr_encoder_observation import independent_operands
        from asr_decoder_observation import independent_operands as prefill_operands, projection_count
        from asr_decoder_observation import independent_attention
        from asr_prefill_observation import PrefillObservation, bridge_buffers, compare_bridge, restore, check_decoder_nodes, check_head_nodes
        from asr_audio_network_nodes import AudioObservation, check_encoder_nodes

    frozen = json.loads(args.contract.read_bytes())
    validate_frozen_contract(frozen)

    def index(roots, revision_key, revision):
        result = {}
        for root in roots:
            data = json.loads((root / "result.json").read_bytes())
            if data["errors"] or data[revision_key] != revision:
                raise ValueError("incomplete or mismatched independent observations")
            if data["candidate_executed"] or data.get("short_control"):
                raise ValueError("independent original-layout observations required")
            for case in data["cases"]:
                if case["id"] in result:
                    raise ValueError("ambiguous independent case identity")
                result[case["id"]] = root, case, data
        return result

    sdk = index(args.sdk_observations, "checkpoint_revision", frozen["checkpoint_revision"])
    canonical = index(args.canonical_observations, "transformers_revision", frozen["transformers_revision"])
    cached = {}
    if args.decoder_operator_probes:
        cached = index(args.canonical_cached_observations, "transformers_revision", frozen["transformers_revision"])
        if set(cached) != {case["id"] for case in frozen["cases"]} or any(
                not data.get("cached_stages") or not case.get("cached_stage_repeated_exact")
                or not case.get("cached_logits_exact_frozen_canonical")
                for _, case, data in cached.values()):
            raise ValueError("complete repeated cached operands with unchanged frozen logits required")
    cases = frozen["cases"]
    references = []
    reference_context = (torch.is_inference_mode_enabled(), torch.is_grad_enabled())
    for case in cases:
        identity = case["id"]
        sdk_root, source, sdk_record = sdk[identity]
        if sdk_record["sdk_revision"] != frozen["sdk_revision"]:
            raise ValueError("SDK source revision differs from frozen contract")
        target_rows = len(case["teacher_rows"])
        if not target_rows or target_rows != len(case["teacher_history"]) or target_rows != len(case["teacher_top1"]):
            raise ValueError("inconsistent frozen teacher row counts")
        if (source["frames"], source["prompt_ids"]) != (case["frames"], case["prompt_ids"]):
            raise ValueError("SDK input differs from frozen contract")
        canonical_root, canonical_case, canonical_record = canonical[identity]
        validate_case_layout(canonical_record, canonical_case, source)
        references.append((case, sdk_root, canonical_root))

    if args.check_inputs:
        load_reference_tensors(references, inference_mode=reference_context[0], grad_enabled=reference_context[1])
        print(json.dumps(dict(cases=len(cases), teacher_rows=sum(len(case["teacher_rows"]) for case in cases),
            independent_inputs_validated=True, candidate_executed=False)))
        return

    # Input-only validation reads independent CPU files without importing the
    # GPU candidate or FlashInfer's device-initializing sampling modules.
    from ignition.components.sampling import RequestAdmission, RUNNING, FINISHED
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.failure import drain_or_retain, retain_undrained, retain_on_failed_drain
    from ignition.recipes.qwen3_asr.bootstrap import PreparationConfig, prepare
    from ignition.recipes.qwen3_asr.audio_generation import PreparedAudioGeneration
    from ignition.recipes.qwen3_asr.pool import PAGE_TOKENS

    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(scope=__doc__.strip(), status="preparing", numerical_policy=POLICY, source_contract=str(args.contract),
        production_graphs=True, native_package_serving=False, cells=[], errors=[], encoder_probes=[], projection_probes=[],
        encoder_stage_probes=[],
        encoder_operator_probes=[],
        prefill_operator_probes=[],
        decoder_operator_probes=[],
        paged_attention_probes=[], prefill_graphs={}, same_path_state_probes=[],
        prefix_kv_layers_in_every_cell=args.prefix_kv_layers,
        full_prefill_qualification=False, full_model_numerical_qualification=False,
        full_prefill_requested=args.full_prefill, full_prefill_probes=[],
        compile_workers=args.compile_workers, diagnostic_capture_memory=[],
        bridge_snapshot_storage="three complete bounded device copies plus transient comparison work, allocated outside model replay; not serving allocation or physical DRAM peaks",
        prefill_capacity_selection=args.prefill_capacity,
        audio_capacity_selection=args.audio_capacity,
        focused_network_node_diagnostic=args.focus_network_nodes,
        complete_schedule_matrix_requested=not args.focus_network_nodes,
        encoder_attribution_only=args.decoder_encoder_features == "canonical",
        input_source="frozen SDK BF16 model-input Mel, widened exactly to FP32 upload; frontend FP32 and decoded PCM have separate gates",
        interpretation="teacher forcing through existing captured graphs; between-replay row retirement is not asynchronous HTTP cancellation")

    def publish():
        (args.output / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    def publish_failure(error):
        try:
            publish()
        except BaseException as reporting_error:
            error.add_note(f"could not record diagnostic failure: {reporting_error!r}")

    publish()
    config = PreparationConfig(kernel_cache=configuration.directory("kernel_cache"), compile_workers=args.compile_workers,
        prefill_chunk_size=args.prefill_chunk_size,
        snapshot_budget_bytes=args.snapshot_budget_bytes, headroom_bytes=args.preparation_headroom_bytes,
        retain_graphs=args.retain_graphs)
    record["requested_prefill_chunk_size"] = args.prefill_chunk_size
    publish()
    with diagnostic_signals(), preparation_context(args.preparation_mode), fp32_reductions(), ModuleImages():
        record["preparation_context"] = dict(requested_mode=args.preparation_mode,
            inference_mode=torch.is_inference_mode_enabled(), grad_enabled=torch.is_grad_enabled(),
            num_threads=torch.get_num_threads(), snapshot_budget_bytes=config.snapshot_budget_bytes,
            headroom_bytes=config.headroom_bytes, retain_graphs=config.retain_graphs,
            allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            allow_fp16_accumulation=torch.backends.cuda.matmul.allow_fp16_accumulation)
        publish()
        torch.cuda.set_device(0)
        with PackageStorage(device=0) as storage:
            record["blas_before_preparation"] = str(torch.backends.cuda.preferred_blas_library())
            started = monotonic()
            prefill_observers, audio_observers = {}, {}
            try:
                prepared = prepare(args.checkpoint, config, package_storage=storage)
            except BaseException as error:
                record["status"] = "preparation_interrupted" if isinstance(error, KeyboardInterrupt) else "preparation_failed"
                record["errors"].append(dict(stage="preparation", error_type=type(error).__name__,
                    error=str(error), traceback=traceback.format_exc()))
                publish_failure(error)
                raise
            try:
                record["preparation_seconds"] = monotonic() - started
                record["prepared_identity"] = prepared.server_config["identity"]
                record["blas_after_preparation"] = str(torch.backends.cuda.preferred_blas_library())
                record["prepared_operations"] = prepared.memory_report["tuning"]["selections"]
                record["status"] = "running"
                publish()
                # These are preparation-owned objects, not a runtime plugin or an
                # alternate serving path. The graph catalog already retains them.
                catalog, = (value for value in prepared.resources if isinstance(value, dict)
                    and isinstance(value.get("audio100"), PreparedAudioGeneration))
                # All variants share request/cache owners; execution resolves C below.
                exemplar = next(catalog[name] for name, signature in prepared.signatures.items()
                    if signature.kind == "prefill")
                requests, uploads, pool = exemplar.batch.requests, exemplar.batch.inputs, exemplar.decoder.pool
                features = exemplar.decoder.feature_state
                baseline, state_baseline = {}, {}
                observed_encoder, observed_projection = set(), {}
                observed_prefill = set()
                observed_attention = set()

                def prefill_graph(prompt_tokens, initial_tokens=None):
                    from ignition.preparation.execution import select_prefill_graph

                    return select_prefill_graph(prepared.signatures, tokens=prompt_tokens,
                        initial_tokens=prompt_tokens if initial_tokens is None else initial_tokens,
                        largest=args.prefill_capacity != "smallest")
                observed_nodes = set()
                record["prefill_network_nodes"], record["head_network_nodes"] = [], []
                record["encoder_network_nodes"], record["encoder_observation_bridges"] = [], []
                if args.encoder_network_nodes:
                    capacities = (100, 800) if args.audio_capacity == "smallest" else (800,)
                    for capacity in capacities:
                        name = f"audio{capacity}"
                        observer = AudioObservation(catalog[name])
                        audio_observers[name] = observer
                        observer.prepare(prepared, prepared.signatures[name], storage,
                            config.kernel_cache, pool.clear_for_preparation)
                    pool.clear_for_preparation()
                    torch.cuda.synchronize()
                if args.full_prefill:
                    names = sorted({prefill_graph(len(case["prompt_ids"])) for case in cases})
                    node_storage = {} if args.prefill_network_nodes else None
                    node_rows = max(prepared.signatures[name].token_capacity for name in names)
                    # Observers execute serially and are read before reuse.
                    # Share their diagnostic destinations across capacities.
                    for name in names:
                        free, total = torch.cuda.mem_get_info()
                        record["diagnostic_capture_memory"].append(dict(graph=name,
                            device_free_bytes=free, device_total_bytes=total,
                            pytorch_allocated_bytes=torch.cuda.memory_allocated(),
                            pytorch_reserved_bytes=torch.cuda.memory_reserved()))
                        publish()
                        observer = PrefillObservation(catalog[name], node_storage=node_storage,
                            node_storage_rows=node_rows)
                        prefill_observers[name] = observer
                        observer.prepare(prepared, prepared.signatures[name], storage, config.kernel_cache)
                    pool.clear_for_preparation()
                    torch.cuda.synchronize()
                # These CPU references are not inputs to preparation or capture.
                # Keep them out of the compiler's unified-memory residency window.
                record["reference_tensor_loading"] = "after_observer_capture"
                publish()
                inputs, expected, encoder_reference, encoder_stages = load_reference_tensors(references,
                    inference_mode=reference_context[0], grad_enabled=reference_context[1])
                record["reference_tensor_context"] = dict(inference_mode=reference_context[0],
                    grad_enabled=reference_context[1])
                record["independent_reference_tensors_validated"] = True
                publish()
                epoch = 0

                def observe_state(identity, stage, values, context):
                    key = (identity, stage)
                    baseline_cell = context["batch_capacity"] == 1 and not context["reverse"] and not context["retire"]
                    entry = dict(id=identity, stage=stage, **context)
                    try:
                        if key not in state_baseline:
                            if not baseline_cell:
                                raise ValueError("the complete B1 state baseline is absent")
                            comparison = compare_state(values, values)
                            state_baseline[key] = {name: value.clone() for name, value in values.items()}
                        else:
                            comparison = compare_state(values, state_baseline[key])
                        entry.update(comparison)
                        if not comparison["passed"]:
                            raise ValueError("same-path state bytes changed: " + identity + " " + stage)
                    except Exception as error:
                        entry.update(passed=False, error=str(error))
                        raise
                    finally:
                        record["same_path_state_probes"].append(entry)

                def replay_graph(graph):
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                        if before[key] != after[key]:
                            raise ValueError("graph replay changed GPU allocation counters: " + key)

                def replay(name):
                    last_replay["graph"] = name
                    replay_graph(prepared.graphs[name])

                last_replay = {}

                def audio(case, slot, context):
                    mel, offset = inputs[case["id"]], 0
                    stages, network_nodes = {}, {}
                    features.feature_slot.fill_(slot)
                    for start in range(0, mel.shape[1], 800):
                        frames = min(800, mel.shape[1] - start)
                        capacity = 100 if frames <= 100 and args.audio_capacity == "smallest" else 800
                        raw = catalog[f"audio{capacity}"].inputs
                        count = 13 * (frames // 100) + ((frames % 100) + 7) // 8
                        raw.mel.zero_()
                        raw.mel[:, :frames].copy_(mel[:, start:start + frames])
                        raw.frames.fill_(frames)
                        raw.total_input_frames.fill_(case["frames"])
                        raw.convolution_width.fill_(min(case["frames"], 100))
                        raw.feature_count.fill_(count)
                        raw.feature_offset.fill_(offset)
                        raw.request_slot.fill_(slot)
                        raw.generation.fill_(epoch)
                        raw.cu_seqlens.copy_(torch.tensor([0, count], dtype=torch.int32))
                        raw.gather_indices.fill_(-1)
                        raw.gather_indices[:count].copy_(torch.arange(count, dtype=torch.int32))
                        name = f"audio{capacity}"
                        if args.encoder_network_nodes and case["id"] not in observed_encoder:
                            nodes, comparison = audio_observers[name].observe(prepared, exemplar,
                                prepared.graphs[name], replay_graph, frames, case["frames"])
                            record["encoder_observation_bridges"].append(dict(id=case["id"],
                                graph=name, frame_start=start, **comparison))
                            if not comparison["passed"]:
                                save_file(nodes, args.output / (case["id"] + f"-audio-bridge-failure-{start}.safetensors"))
                                raise ValueError("observed audio changed production output/state")
                            for field, value in nodes.items():
                                network_nodes.setdefault(field, []).append(value)
                        else:
                            replay(name)
                        for name, value in observe_encoder(catalog[f"audio{capacity}"].encoder,
                                frames, min(case["frames"], 100)).items():
                            stages.setdefault(name, []).append(value)
                        if requests.error_code[slot].item() != 0:
                            raise ValueError("audio graph rejected diagnostic input")
                        offset += count
                    if offset != case["prompt_ids"].count(151676) or features.count[slot].item() != offset:
                        raise ValueError("encoder feature count differs from prompt")
                    identity = case["id"]
                    encoded = features.arena[slot, :offset].cpu().clone()
                    stage_values = {name: torch.cat(windows).reshape(encoder_stages[identity][name].shape)
                        for name, windows in stages.items()}
                    observed = dict(encoder_features_bf16=encoded, **stage_values)
                    save_file(observed, args.output / (context["label"] + "-" + identity + "-encoder-state.safetensors"))
                    observe_state(identity, "encoder", observed, context)
                    if case["id"] not in observed_encoder:
                        if args.encoder_network_nodes:
                            nodes = {name: torch.cat(windows, dim=0) for name, windows in network_nodes.items()}
                            with safe_open(canonical[identity][0] / (identity + "-prefill.safetensors"), framework="pt") as reference:
                                try:
                                    node_probe = dict(id=identity, production_bridge_passed=True,
                                        **check_encoder_nodes(nodes, reference, case))
                                finally:
                                    save_file(nodes, args.output / (identity + "-encoder-network-nodes.safetensors"))
                            record["encoder_network_nodes"].append(node_probe)
                            publish()
                            if not node_probe["passed"]:
                                raise ValueError("frozen encoder network-node budget failed")
                        actual = encoded
                        reference = encoder_reference[identity]
                        probe = dict(id=identity, passed=True)
                        try:
                            check(actual, reference, case["stages"]["thinker.audio_tower.proj2.output"])
                        except ValueError as error:
                            probe.update(passed=False, error=str(error))
                        delta = actual.double() - reference.double()
                        probe.update(max_abs=delta.abs().max().item(),
                            relative_l2=(delta.norm() / reference.double().norm()).item())
                        record["encoder_probes"].append(probe)
                        save_file({"encoder_features_bf16": actual}, args.output / (identity + "-encoder.safetensors"))
                        stage_probes = {}
                        for name, actual in stage_values.items():
                            reference = encoder_stages[identity][name]
                            probe = dict(passed=True)
                            try:
                                check(actual, reference, case["stages"][name])
                            except ValueError as error:
                                probe.update(passed=False, error=str(error))
                            delta = actual.double() - reference.double()
                            probe.update(exact_bytes=torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)),
                                max_abs=delta.abs().max().item(),
                                relative_l2=(delta.norm() / reference.double().norm().clamp_min(1e-300)).item())
                            stage_probes[name] = probe
                        record["encoder_stage_probes"].append(dict(id=identity, stages=stage_probes))
                        save_file(stage_values, args.output / (identity + "-encoder-stages.safetensors"))
                        if args.encoder_operator_probes and identity in (
                                "zh-BAC009S0729W0241-first-13f", "zh-BAC009S0729W0241-first-100f",
                                "en-422-122949-0013-first-2f", "zh-BAC009S0729W0241-first-2f",
                                "zh-BAC009S0727W0418"):
                            layers = (tuple(range(24)) if case["frames"] == 2 else
                                (10,) if identity == "zh-BAC009S0727W0418" else (0,))
                            with safe_open(canonical[identity][0] / (identity + "-prefill.safetensors"), framework="pt") as reference:
                                values = independent_operands(catalog[f"audio{capacity}"].encoder, reference, layers=layers)
                            probes, saved = {}, {}
                            for name, value in values.items():
                                a, b = value["actual"], value["reference"]
                                delta = a.double() - b.double()
                                probe = dict(physical_rows=value["physical_rows"], logical_rows=value["logical_rows"],
                                    max_abs=delta.abs().max().item(), relative_l2=(delta.norm() / b.double().norm().clamp_min(1e-300)).item(),
                                    exact_reference_bytes=torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
                                if "fp64_rounded" in value:
                                    probe["max_abs_to_fp64_rounded"] = (a.double() - value["fp64_rounded"].double()).abs().max().item()
                                for field in ("fp64_rounded", "torch_logical", "torch_padded"):
                                    if field in value:
                                        measured = value[field]
                                        difference = measured.double() - b.double()
                                        probe[field] = dict(exact_reference_bytes=torch.equal(measured.view(torch.uint8), b.view(torch.uint8)),
                                            max_abs=difference.abs().max().item(),
                                            relative_l2=(difference.norm() / b.double().norm().clamp_min(1e-300)).item())
                                probes[name] = probe
                                for field in ("actual", "reference", "fp64_rounded", "torch_logical", "torch_padded"):
                                    if field in value:
                                        saved[name + "." + field] = value[field]
                            controls_exact = all(probe.get("torch_logical", {}).get("exact_reference_bytes", False)
                                for probe in probes.values())
                            record["encoder_operator_probes"].append(dict(id=identity, attribution_only=True,
                                independent_controls_exact=controls_exact,
                                layers=list(layers), operators=probes))
                            save_file(saved, args.output / (identity + "-encoder-operators.safetensors"))
                        observed_encoder.add(identity)
                    if args.decoder_encoder_features == "canonical":
                        # Same graph, metadata, generation and storage; change
                        # only this diagnostic's feature values. Such a run
                        # cannot qualify the end-to-end candidate.
                        features.arena[slot, :offset].copy_(encoder_reference[case["id"]])

                def local_projection(case, capacity):
                    from dataclasses import asdict

                    # A fixed represented input separates cross-shape GEMM
                    # changes from preceding norm/attention computation. This
                    # is attribution, not a substitute for the teacher gates.
                    identity = case["id"]
                    tree = exemplar.decoder.weights.tree.model.language_model
                    embedding = tree.embed_tokens.weight[case["teacher_history"][0]].cpu().double()
                    norm = tree.layers[0].input_layernorm
                    normalized = (embedding * torch.rsqrt(embedding.square().mean() + norm.variance_epsilon)).to(torch.bfloat16)
                    normalized = normalized * norm.weight.cpu()
                    projection = catalog[f"decode{capacity}"].decoder.tree.layers[0].self_attn.v_proj
                    from ignition.components.row_projection import PreparedRowProjection
                    from ignition.components.reduction_projection import PreparedReductionProjection
                    wrapper = projection.operation
                    from ignition.components.canonical_projection import PreparedCanonicalProjection, PreparedRowwiseProjection
                    operation = wrapper.operation if isinstance(wrapper, PreparedRowProjection) else wrapper
                    physical_rows = projection.operation.output.shape[0]
                    owners = [projection]
                    with retain_on_failed_drain(0, owners):
                        source = torch.zeros((physical_rows, normalized.numel()), dtype=torch.bfloat16, device="cuda")
                        owners.append(source)
                        source[:capacity].copy_(normalized)
                        from contextlib import nullcontext

                        temporary_count = (projection_count(operation, capacity, owners)
                            if isinstance(operation, (PreparedReductionProjection, PreparedCanonicalProjection))
                            else nullcontext())
                        with temporary_count:
                            value = projection(source)[:capacity].cpu().clone()
                            torch.cuda.synchronize()
                    weight = projection.operation.weight.cpu().double()
                    golden = (normalized.double() @ weight.T).to(torch.bfloat16)
                    if capacity == 1:
                        observed_projection[identity] = value[0]
                    anchor = observed_projection[identity]
                    record["projection_probes"].append(dict(id=identity, logical_capacity=capacity,
                        physical_rows=physical_rows,
                        query=(dict(rows=1 if isinstance(operation, PreparedRowwiseProjection) else operation.rows,
                            columns=operation.output.shape[1], reduction=operation.columns)
                            if isinstance(operation, PreparedCanonicalProjection) else asdict(operation.plan.query)),
                        selected_config=(None if isinstance(operation, PreparedCanonicalProjection) else
                            operation.plan.contract.config_payload(operation.plan.selection.config).to_dict()),
                        selected_source=("rowwise singleton eager linear; not captured"
                            if isinstance(operation, PreparedRowwiseProjection) else
                            "canonical eager linear; not captured" if isinstance(operation, PreparedCanonicalProjection)
                            else operation.plan.selection.source),
                        exact_to_B1=torch.equal(value[0].view(torch.uint8), anchor.view(torch.uint8)),
                        rows_exact=torch.equal(value.view(torch.uint8), value[:1].expand_as(value).contiguous().view(torch.uint8)),
                        max_abs_to_B1=(value[0].float() - anchor.float()).abs().max().item(),
                        max_abs_to_fp64_bf16=(value[0].float() - golden.float()).abs().max().item()))
                    save_file(dict(input_bf16=normalized, output_bf16=value, fp64_rounded_bf16=golden),
                        args.output / f"b{capacity}-{identity}-layer0-v-projection.safetensors")
                    if args.prefill_operator_probes and identity not in observed_prefill and identity in (
                            "en-422-122949-0013-first-2f", "zh-BAC009S0729W0241-first-100f"):
                        name = prefill_graph(len(case["prompt_ids"]))
                        with safe_open(canonical[identity][0] / (identity + "-prefill.safetensors"), framework="pt") as reference:
                            observations = prefill_operands(catalog[name].decoder, reference)
                        probes, saved = {}, {}
                        for path, observation in observations.items():
                            data = observation.pop("data")
                            reference = data["reference"]
                            comparisons = {}
                            for field, value in data.items():
                                if field == "reference":
                                    continue
                                delta = value.double() - reference.double()
                                comparisons[field] = dict(exact_bytes=torch.equal(value.view(torch.uint8), reference.view(torch.uint8)),
                                    max_abs=delta.abs().max().item(),
                                    relative_l2=(delta.norm() / reference.double().norm().clamp_min(1e-300)).item())
                            probes[path] = dict(**observation, comparisons=comparisons)
                            saved.update({path + "." + field: value for field, value in data.items()})
                        record["prefill_operator_probes"].append(dict(id=identity, graph=name, attribution_only=True, operators=probes))
                        save_file(saved, args.output / (identity + "-prefill-operators.safetensors"))
                        observed_prefill.add(identity)
                    if (args.paged_attention_probes or args.torch_paged_attention_probes) and identity == "zh-BAC009S0729W0241" \
                            and identity not in observed_attention:
                        for name in (prefill_graph(len(case["prompt_ids"])), "decode8"):
                            with safe_open(canonical[identity][0] / (identity + "-prefill.safetensors"), framework="pt") as reference:
                                observation = independent_attention(catalog[name].decoder, reference,
                                    canonical_paged=args.torch_paged_attention_probes,
                                    repack_page_tokens=args.probe_repack_page_tokens)
                            values = observation.pop("data")
                            record["paged_attention_probes"].append(dict(id=identity, graph=name,
                                attribution_only=True, **observation))
                            save_file(values, args.output / (identity + "-" + name + "-paged-attention.safetensors"))
                            publish()
                        observed_attention.add(identity)

                def group(rows, capacity, reverse, retire, result):
                    nonlocal epoch
                    epoch += 1
                    pool.clear_for_preparation()
                    requests.status.zero_()
                    uploads.request_state_rows.zero_()
                    uploads.page_tables.fill_(-1)
                    uploads.sample_enabled.fill_(1)
                    slots = list(range(len(rows)))
                    if reverse:
                        slots = [7 - slot for slot in slots]
                    context = dict(batch_capacity=capacity, reverse=reverse, retire=retire,
                        label=f"b{capacity}-reverse{int(reverse)}-retire{int(retire)}")
                    page = 0
                    for case, slot in zip(rows, slots, strict=True):
                        prompt = case["prompt_ids"]
                        pages = (len(prompt) + len(case["teacher_history"]) + PAGE_TOKENS - 1) // PAGE_TOKENS
                        if page + pages > pool.memory_capacity[1].item():
                            raise ValueError("diagnostic group exceeds prepared physical KV")
                        uploads.page_tables[slot, :pages].copy_(torch.arange(page, page + pages, dtype=torch.int32))
                        page += pages
                        uploads.request_state_rows[slot, 0] = slot + 1
                        requests.admit(slot, RequestAdmission(epoch, 0, max_new_tokens=1024,
                            temperature=1., top_k=1, top_p=1.), prompt)
                        audio(case, slot, context)
                        uploads.live_rows.fill_(1)
                        uploads.row_slot_ids.fill_(-1)
                        uploads.row_slot_ids[0] = slot
                        uploads.row_generations.fill_(epoch)
                        uploads.query_lengths.fill_(len(prompt))
                        prepared.buffers["original_prompt_tokens"].zero_()
                        prepared.buffers["original_prompt_tokens"][:1].fill_(len(case["prompt_ids"]))
                        uploads.token_ids.zero_()
                        uploads.token_ids[:len(prompt)].copy_(torch.tensor(prompt))
                        features.indices.fill_(-1)
                        positions = [i for i, token in enumerate(prompt) if token == 151676]
                        features.indices[torch.tensor(positions, device="cuda")] = torch.arange(len(positions), device="cuda", dtype=torch.int32)
                        name = prefill_graph(len(prompt), len(case["prompt_ids"]))
                        choice = dict(graph=name, token_capacity=prepared.signatures[name].token_capacity,
                            initial_prompt_capacity=prepared.signatures[name].initial_prompt_capacity)
                        previous = record["prefill_graphs"].setdefault(case["id"], choice)
                        if previous != choice:
                            raise ValueError("prefill graph selection changed within the fixed diagnostic")
                        bridge = bridge_buffers(prepared, catalog[name]) if args.full_prefill else None
                        observer = prefill_observers.get(name)
                        initial_state = observer.bridge.snapshot(bridge) if bridge is not None else None
                        replay(name)
                        # Retain the real production row before any observer
                        # or state gate can fail and the next group clears KV.
                        result[case["id"]].append(catalog[name].decoder.scratch.logits[0].cpu().clone())
                        if bridge is not None:
                            production_state = observer.bridge.snapshot(bridge)
                            restore(bridge, initial_state)
                            replay_graph(observer.graphs["observed_prefill"])
                            observed_state = observer.bridge.snapshot(bridge)
                            bridge_result = compare_bridge(observed_state, production_state)
                            probe = dict(id=case["id"], graph=name, **context,
                                diagnostic_graph_only=True, production_bridge=bridge_result,
                                bridge_snapshot_bytes=sum(value.nbytes for value in initial_state.values()),
                                passed=False)
                            record["full_prefill_probes"].append(probe)
                            if not bridge_result["passed"]:
                                failure = {}
                                for field, comparison in bridge_result["fields"].items():
                                    if not comparison["exact_bytes"]:
                                        failure["expected." + field] = production_state[field].cpu().contiguous()
                                        failure["actual." + field] = observed_state[field].cpu().contiguous()
                                filename = context["label"] + "-" + case["id"] + "-bridge-failure.safetensors"
                                save_file(failure, args.output / filename)
                                probe["failure_tensors"] = filename
                                raise ValueError("observed prefill changed production output/state")
                            production_logits = catalog[name].decoder.scratch.logits[0].cpu().clone()
                            observed_logits = observer.logits(len(prompt), replay_graph)
                            with safe_open(canonical[case["id"]][0] / (case["id"] + "-prefill.safetensors"),
                                    framework="pt") as reference:
                                full_reference = reference.get_tensor("thinker.lm_head.output")
                                if not torch.equal(full_reference[0, -1].view(torch.uint8),
                                        expected[case["id"]][0].view(torch.uint8)):
                                    raise ValueError("full-prefill reference differs from frozen first teacher row")
                                probe.update(compare_prefill(observed_logits,
                                    full_reference, case))
                                if args.prefill_network_nodes and case["id"] not in observed_nodes:
                                    nodes = observer.node_values(len(prompt))
                                    try:
                                        row = dict(id=case["id"], graph=name, production_bridge_passed=True,
                                            **check_decoder_nodes(nodes, reference, case))
                                    finally:
                                        save_file(nodes, args.output / (case["id"] + "-prefill-network-nodes.safetensors"))
                                    record["prefill_network_nodes"].append(row)
                                    # normalized contains the actual BF16
                                    # scratch.hidden read after each captured
                                    # position-head invocation, not a new norm.
                                    head_nodes = {"thinker.lm_head.input.0": torch.stack(observer.normalized),
                                        "thinker.lm_head.output": observed_logits}
                                    try:
                                        head_row = dict(id=case["id"], graph=name, production_bridge_passed=True,
                                            **check_head_nodes(head_nodes, reference, case))
                                    finally:
                                        save_file(head_nodes, args.output / (case["id"] + "-head-network-nodes.safetensors"))
                                    record["head_network_nodes"].append(head_row)
                                    publish()
                                    if not row["passed"] or not head_row["passed"]:
                                        raise ValueError("frozen decoder or head network-node budget failed")
                                    observed_nodes.add(case["id"])
                            probe["last_row_exact_production"] = torch.equal(observed_logits[-1].view(torch.uint8),
                                production_logits.view(torch.uint8))
                            probe["passed"] &= probe["last_row_exact_production"]
                            save_file(dict(logits_bf16=observed_logits,
                                final_residual_bf16=observer.residual[:len(prompt)].cpu().clone()),
                                args.output / (context["label"] + "-" + case["id"] + "-full-prefill.safetensors"))
                            observe_state(case["id"], "full_prefill_logits",
                                dict(logits_bf16=observed_logits), context)
                            # The head-only observation changes shared output scratch.
                            # Restore the exact ordinary-graph end before teacher decode.
                            restore(bridge, production_state)
                            del initial_state, production_state, observed_state
                            observer.bridge.release()
                            if not probe["passed"]:
                                raise ValueError("full-prefill scalar or production-head bridge gate failed")
                        if requests.error_code[slot].item() or requests.consumed_tokens[slot].item() != len(prompt):
                            raise ValueError("prefill graph state differs from the complete prompt")
                        page_ids = uploads.page_tables[slot, :(len(prompt) + PAGE_TOKENS - 1) // PAGE_TOKENS].cpu().tolist()
                        keys, values = [], []
                        prefix_layers = args.prefix_kv_layers
                        for layer in range(prefix_layers):
                            key, value = pool.attention(layer)
                            keys.append(torch.cat([key[page].cpu().clone() for page in page_ids])[:len(prompt)].transpose(0, 1).contiguous())
                            values.append(torch.cat([value[page].cpu().clone() for page in page_ids])[:len(prompt)].transpose(0, 1).contiguous())
                        state = dict(keys=torch.stack(keys), values=torch.stack(values),
                            layer_indices=torch.arange(prefix_layers, dtype=torch.int32))
                        save_file(state,
                            args.output / f"b{capacity}-reverse{int(reverse)}-retire{int(retire)}-{case['id']}-prefix-kv.safetensors")
                        observe_state(case["id"], "prefix_kv", state, context)
                    for step in range(1, max(len(case["teacher_history"]) for case in rows)):
                        active = []
                        for logical, (case, slot) in enumerate(zip(rows, slots, strict=True)):
                            if step >= len(case["teacher_history"]) or retire and logical == 0 and step >= 2:
                                requests.status[slot] = FINISHED
                                continue
                            requests.status[slot] = RUNNING
                            requests.pending_token[slot] = case["teacher_history"][step - 1]
                            active.append((case, slot))
                        if not active:
                            break
                        if reverse:
                            active.reverse()
                        uploads.live_rows.fill_(len(active))
                        uploads.row_slot_ids.fill_(-1)
                        uploads.row_generations.fill_(epoch)
                        uploads.query_lengths.fill_(1)
                        uploads.row_slot_ids[:len(active)].copy_(torch.tensor([slot for _, slot in active], dtype=torch.int32))
                        replay(f"decode{capacity}")
                        logits = catalog[f"decode{capacity}"].decoder.scratch.logits.cpu()
                        for packed, (case, slot) in enumerate(active):
                            result[case["id"]].append(logits[packed].clone())
                        for packed, (case, slot) in enumerate(active):
                            if requests.error_code[slot].item() or requests.consumed_tokens[slot].item() != len(case["prompt_ids"]) + step:
                                raise ValueError("decode graph state differs from teacher position")
                    for case, slot in zip(rows, slots, strict=True):
                        if len(case["teacher_history"]) < 2:
                            continue
                        position = len(case["prompt_ids"])
                        page = int(uploads.page_tables[slot, position // PAGE_TOKENS].item())
                        keys, values = [], []
                        for layer in range(28):
                            key, value = pool.attention(layer)
                            keys.append(key[page, position % PAGE_TOKENS].cpu().clone())
                            values.append(value[page, position % PAGE_TOKENS].cpu().clone())
                        state = dict(keys=torch.stack(keys), values=torch.stack(values))
                        save_file(state,
                            args.output / f"b{capacity}-reverse{int(reverse)}-retire{int(retire)}-{case['id']}-first-decode-kv.safetensors")
                        observe_state(case["id"], "first_decode_kv", state, context)
                    return {name: torch.stack(values) for name, values in result.items()}

                for capacity in ((1,) if args.focus_network_nodes else (1, 2, 4, 8)):
                    schedules = ((False, False),) if args.focus_network_nodes else ((False, False), (True, False), (True, True))
                    for reverse, retire in schedules:
                        cell = dict(batch_capacity=capacity, reverse_slots_and_rows=reverse, retire_other_row=retire,
                            complete_teacher_qualification=not args.focus_network_nodes and not retire
                                and args.decoder_encoder_features == "captured", cases=[], errors=[])
                        record["cells"].append(cell)
                        for start in range(0, len(cases), capacity):
                            rows = cases[start:start + capacity]
                            partial = {case["id"]: [] for case in rows}
                            label = f"b{capacity}-reverse{int(reverse)}-retire{int(retire)}"
                            try:
                                actual = group(rows, capacity, reverse, retire, partial)
                                save_teacher_traces(actual, args.output, label)
                                if not reverse:
                                    for case in rows:
                                        local_projection(case, capacity)
                                for logical, case in enumerate(rows):
                                    identity, values = case["id"], actual[case["id"]]
                                    if capacity == 1 and not reverse:
                                        baseline[identity] = values
                                    row = dict(id=identity, **compare_trace(values, expected[identity], case,
                                        baseline[identity], retired_after=2 if retire and logical == 0 else None))
                                    cell["cases"].append(row)
                                    if not row["passed"]:
                                        cell["errors"].append(identity)
                            except BaseException as error:
                                failure = dict(ids=[case["id"] for case in rows], error_type=type(error).__name__,
                                    error=str(error), traceback=traceback.format_exc(), last_graph=last_replay.get("graph"))
                                cell["errors"].append(failure)
                                drained = drain_or_retain(error, 0, prepared, storage, prefill_observers, audio_observers)

                                def failure_state():
                                    values = {"request." + name: value.cpu().clone()
                                        for name, value in vars(requests).items() if isinstance(value, torch.Tensor)}
                                    values.update({"input." + name: value.cpu().clone()
                                        for name, value in vars(uploads).items() if isinstance(value, torch.Tensor)})
                                    values["completion_records"] = prepared.buffers["completion_records"].cpu().clone()
                                    forward = catalog.get(last_replay.get("graph"))
                                    if forward is not None and hasattr(forward, "decoder"):
                                        values["last_logits_bf16"] = forward.decoder.scratch.logits.cpu().clone()
                                    return values

                                try:
                                    failure["raw_evidence"] = save_failed_group(partial, args.output,
                                        label + f"-group{start}", drained, failure_state)
                                except BaseException as evidence_error:
                                    failure["raw_evidence_error"] = repr(evidence_error)
                                    error.add_note(f"could not retain failed group evidence: {evidence_error!r}")
                                    publish_failure(error)
                                    raise error
                                if not drained or not isinstance(error, Exception):
                                    record["errors"].append(len(record["cells"]) - 1)
                                    publish_failure(error)
                                    raise
                                for observer in prefill_observers.values():
                                    observer.bridge.release()
                            publish()
                        if cell["errors"]:
                            record["errors"].append(len(record["cells"]) - 1)
                        publish()
                if args.generation_error_probes:
                    if __package__:
                        from .asr_generation_errors import check as check_generation_errors
                    else:
                        from asr_generation_errors import check as check_generation_errors
                    check_generation_errors(prepared, args.output / "generation-errors.json")
                if args.decoder_operator_probes:
                    # Run only after all complete teacher traces are saved.
                    # These calls mutate shared scratch; they cannot qualify
                    # serving or replace the independent frozen logits.
                    for identity in ("en-777-126732-0081", "en-422-122949-0013-first-2f"):
                        root, cached_case, _ = cached[identity]
                        for step in (0, 1, 2):
                            if step not in cached_case["cached_stage_steps"]:
                                raise ValueError("required real cached teacher step is absent")
                            graph_name = record["prefill_graphs"][identity]["graph"] if step == 0 else "decode8"
                            with safe_open(root / (identity + f"-teacher-step-{step}.safetensors"),
                                    framework="pt") as reference:
                                if not torch.equal(reference.get_tensor("thinker.lm_head.output").reshape(-1, 151936)[-1].view(torch.uint8),
                                        expected[identity][step].view(torch.uint8)):
                                    raise ValueError("cached operator reference changed the frozen teacher logit row")
                                observations = prefill_operands(catalog[graph_name].decoder, reference, layers=range(28))
                            probes, saved = {}, {}
                            for path, observation in observations.items():
                                data = observation.pop("data")
                                target = data["reference"]
                                comparisons = {}
                                for field, value in data.items():
                                    if field == "reference":
                                        continue
                                    delta = value.double() - target.double()
                                    comparisons[field] = dict(exact_bytes=torch.equal(value.view(torch.uint8), target.view(torch.uint8)),
                                        max_abs=delta.abs().max().item(), differing_elements=(value != target).sum().item(),
                                        relative_l2=(delta.norm() / target.double().norm().clamp_min(1e-300)).item())
                                probes[path] = dict(**observation, comparisons=comparisons)
                                saved.update({path + "." + field: value for field, value in data.items()})
                            record["decoder_operator_probes"].append(dict(id=identity, teacher_step=step,
                                graph=graph_name, attribution_only=True, actual_cached_operands=step > 0, operators=probes))
                            save_file(saved, args.output / (identity + f"-teacher-step-{step}-projection-operators.safetensors"))
                            publish()
            except BaseException as error:
                record["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed"
                record["errors"].append(dict(stage="execution_or_observation",
                    error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
                drained = drain_or_retain(error, 0, prepared, storage, prefill_observers, audio_observers)
                record["interrupted_cleanup"] = dict(drained=drained,
                    observers_closed=False, prepared_closed=False)
                publish_failure(error)
                if drained:
                    try:
                        for observer in (*prefill_observers.values(), *audio_observers.values()):
                            observer.close()
                        record["interrupted_cleanup"]["observers_closed"] = True
                        prepared.close()
                        record["interrupted_cleanup"]["prepared_closed"] = True
                    except BaseException as cleanup_error:
                        record["interrupted_cleanup"]["error"] = dict(type=type(cleanup_error).__name__, message=str(cleanup_error))
                        record["interrupted_cleanup"]["drained"] = drain_or_retain(cleanup_error, 0, prepared, storage, prefill_observers, audio_observers)
                        if not record["interrupted_cleanup"]["drained"]:
                            retain_undrained(error, prepared, storage, prefill_observers, audio_observers)
                        error.add_note(f"teacher diagnostic cleanup failed: {cleanup_error!r}")
                publish_failure(error)
                raise
            else:
                try:
                    for observer in (*prefill_observers.values(), *audio_observers.values()):
                        observer.close()
                    prepared.close()
                except BaseException as error:
                    record["status"] = "failed"
                    record["errors"].append(dict(stage="cleanup", error_type=type(error).__name__,
                        error=str(error), traceback=traceback.format_exc()))
                    publish_failure(error)
                    drain_or_retain(error, 0, prepared, storage, prefill_observers, audio_observers)
                    raise
    if any(not probe["passed"] for probe in record["encoder_probes"]):
        record["errors"].append("encoder-stages")
    if any(not stage["passed"] for probe in record["encoder_stage_probes"] for stage in probe["stages"].values()):
        record["errors"].append("encoder-projection-and-attention-stages")
    record["full_prefill_qualification"] = (args.full_prefill and not args.focus_network_nodes and not record["errors"]
        and len(record["full_prefill_probes"]) == 12 * len(cases)
        and all(probe["passed"] for probe in record["full_prefill_probes"]))
    record["prefill_network_node_budgets_passed"] = (args.prefill_network_nodes
        and len(record["prefill_network_nodes"]) == len(cases)
        and {probe["id"] for probe in record["prefill_network_nodes"]} == {case["id"] for case in cases}
        and all(probe["passed"] for probe in record["prefill_network_nodes"]))
    if args.prefill_network_nodes and not record["prefill_network_node_budgets_passed"]:
        record["errors"].append("decoder-network-node-coverage-or-budget")
    record["head_network_node_budgets_passed"] = (args.prefill_network_nodes
        and len(record["head_network_nodes"]) == len(cases)
        and {probe["id"] for probe in record["head_network_nodes"]} == {case["id"] for case in cases}
        and all(probe["passed"] for probe in record["head_network_nodes"]))
    if args.prefill_network_nodes and not record["head_network_node_budgets_passed"]:
        record["errors"].append("head-network-node-coverage-or-budget")
    record["encoder_network_node_budgets_passed"] = (args.encoder_network_nodes
        and len(record["encoder_network_nodes"]) == len(cases)
        and {probe["id"] for probe in record["encoder_network_nodes"]} == {case["id"] for case in cases}
        and all(probe["passed"] for probe in record["encoder_network_nodes"])
        and bool(record["encoder_observation_bridges"])
        and all(probe["passed"] for probe in record["encoder_observation_bridges"]))
    if args.encoder_network_nodes and not record["encoder_network_node_budgets_passed"]:
        record["errors"].append("encoder-network-node-coverage-or-budget")
    record["status"] = "failed" if record["errors"] else "complete"
    publish()
    if record["errors"]:
        raise RuntimeError("candidate teacher gates failed; all attempted cells retained")


if __name__ == "__main__":
    main()
