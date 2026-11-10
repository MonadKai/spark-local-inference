# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare original BF16 audio encoding and full-attention text CUDA graphs."""

import contextlib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ignition.components.audio_inputs import AudioInputs, PrefillAudioInputs
from ignition.components.capacity import batch_capacities, prefill_capacities, request_capacity
from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.feature_state import FeatureState
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState, SamplingSelection
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.progress import phase

from .audio import AudioMetadata, PreparedAudioEncoder
from .audio_generation import PreparedAudioGeneration
from .checkpoint import inspect_original_checkpoint
from .contract import FEATURES, FRAME_CAPACITIES, prepare as prepare_contract
from .decoder import PreparedDecoder, Scratch, _Vocabulary
from .frontend import prepare_frontend, runtime_assets
from .model import ModelWeights
from .numerics import POLICY, fp32_reductions
from .pool import ContinuationPool, PAGE_TOKENS, continuation_layout, page_bytes


@dataclass(frozen=True)
class PreparationConfig:
    device: int = 0
    max_active: int = 8
    context: int = 8192
    prefill_chunk_size: int = 4096
    snapshot_budget_bytes: int = 256 << 20
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    compile_workers: int | None = None
    collective_tuning: str = "static"
    collective_ladder: dict | None = None
    native_dense_activations: bool = False
    lock_gpu_clocks: bool = False
    autotune_effort: str = "default"
    tuning_replica: object | None = None

    def __post_init__(self):
        request_capacity(self.max_active)
        if not 1 <= self.max_active <= 8 or self.device < 0:
            raise ValueError("ASR preparation requires one GPU and concurrency in [1,8]")
        if not PAGE_TOKENS <= self.context <= 8192 or self.context % PAGE_TOKENS:
            raise ValueError(f"ASR context must be {PAGE_TOKENS}-aligned in [{PAGE_TOKENS},8192]")
        if not 128 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 128:
            raise ValueError("ASR prefill capacity must be 128-aligned and fit context")
        if self.snapshot_budget_bytes < continuation_layout().row_bytes or self.headroom_bytes < 0:
            raise ValueError("ASR preparation requires complete continuation storage and nonnegative headroom")


@torch.no_grad()
@with_autotune_effort
@fp32_reductions()
def prepare(snapshot, config, *, progress=None, package_storage):
    from ignition.preparation.failure import drain_or_retain, resources_retained, retain_undrained

    transient, failed_owners = contextlib.ExitStack(), contextlib.ExitStack()
    prepared = None
    try:
        prepared = _prepare(snapshot, config, progress=progress, package_storage=package_storage,
            transient=transient, failed_owners=failed_owners)
        transient.close()
    except BaseException as error:
        if (resources_retained(error) or torch.cuda.is_initialized()) and not drain_or_retain(
                error, config.device, transient, failed_owners, package_storage, prepared):
            raise
        try:
            # No native handoff has occurred. Drain before resetting captured
            # executables or closing the session's GPU resource owners.
            failed_owners.close()
        except BaseException as cleanup_error:
            error.add_note(f"ASR preparation owner cleanup failed: {cleanup_error!r}")
        try:
            transient.close()
        except BaseException as cleanup_error:
            if resources_retained(cleanup_error):
                retain_undrained(error, transient, failed_owners, package_storage, prepared)
            error.add_note(f"transient allocation scope cleanup failed: {cleanup_error!r}")
        raise
    failed_owners.pop_all()
    return prepared


def _prepare(snapshot, config, *, progress, package_storage, transient, failed_owners):
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    device = torch.device("cuda", config.device)
    torch.cuda.set_device(device)
    # Validate structural frontend identity before transferring any checkpoint bytes.
    frontend, assets = prepare_frontend(snapshot), runtime_assets(snapshot)
    with phase(progress, "Inspect original ASR checkpoint"):
        checkpoint = inspect_original_checkpoint(snapshot)
    transient.enter_context(package_storage.transient())
    weights = ModelWeights.load(checkpoint, device, progress=progress)
    release_unused()
    layout = continuation_layout()
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    slots, blocks = request_capacity(config.max_active), config.context // PAGE_TOKENS
    state_rows = slots + snapshots
    pages = config.prefill_chunk_size // PAGE_TOKENS + 2 * slots
    total = torch.cuda.mem_get_info(device)[1]
    # Host prefix records outlive this process: headers alone cannot identify
    # weights when a same-shaped checkpoint changes its tensor payloads.
    payloads = []
    for path in checkpoint.index.shard_paths:
        with path.open("rb") as source:
            payloads.append(hashlib.file_digest(source, "sha256").hexdigest())
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))}
    sources.update({"components/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((Path(__file__).parents[2] / "components").glob("*.py"))})
    initial_capacities = tuple(sorted({min(config.context, value) for value in (256, 512, 8192)}))
    identity = hashlib.sha256(json.dumps(dict(
        payloads=payloads, sources=sources,
        headers=[shard.header_sha256 for shard in checkpoint.index.shards],
        config=json.loads((snapshot / "config.json").read_bytes()),
        contract="qwen3-asr.bf16.graph-inputs.v1", numerical_policy=POLICY,
        original_prompt_capacity=config.context,
        initial_prompt_capacities=initial_capacities,
        bf16_reduction_split_k=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k),
        sort_keys=True).encode()).hexdigest()
    with phase(progress, "Allocate ASR state"):
        pool = ContinuationPool.for_model(state_rows + 1, pages, device, storage=package_storage,
            maximum_pages=total // page_bytes(), maximum_rows=total // layout.row_bytes,
            backed_slots=min(state_rows + 1, 1 + 2 * slots))
        requests = DeviceRequestState.allocate(device, vocab_size=151936, capacity=slots)
        inputs = BatchInputs.allocate(max(config.prefill_chunk_size, slots), blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=151936, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        original_prompt_tokens = torch.zeros(slots, dtype=torch.int32, device=device)
        completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + 1), device=device)
        prefill_audio = PrefillAudioInputs.allocate(config.prefill_chunk_size, device)
        feature_state = FeatureState(inputs, requests, indices=prefill_audio.audio_token_indices,
            feature_slot=prefill_audio.audio_feature_slot, feature_token=151676,
            slots=slots, features=FEATURES, hidden=2048,
            prompt_lengths=original_prompt_tokens[:1])
        session = RecordedPreparationSession(device=device, checkpoint=identity, workload_schema="qwen3-asr",
            numerical_policy=POLICY, coordinator=None, headroom_bytes=config.headroom_bytes,
            compile_workers=config.compile_workers, collective_tuning=config.collective_tuning,
            collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort,
            tuning_replica=config.tuning_replica, native_dense_activations=config.native_dense_activations,
            lock_gpu_clocks=config.lock_gpu_clocks, benchmark_storage=package_storage)
        failed_owners.callback(session.close)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)
        graph_buffers, private_buffers = GraphBufferPool(device), GraphBufferPool(device)
        buffers = dict(inputs.buffers(), **prefill_audio.buffers(), **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token, sampling_grammar_mask=requests.grammar_mask,
            frame_index=frame, submission_sequence=sequence,
            original_prompt_tokens=original_prompt_tokens, **completion.buffers())
    forwards, signatures, metadata, raw_audio = {}, {}, {}, {}
    prefills = (64, *prefill_capacities(config.prefill_chunk_size))
    batches = batch_capacities(config.max_active)

    def add(name, forward, kind, batch, tokens, fields, *, initial_prompt_capacity=0):
        forwards[name] = forward
        signatures[name] = GraphSignature(kind, batch, tokens, tuple(fields),
            initial_prompt_capacity=initial_prompt_capacity)

    with phase(progress, "Construct ASR graph catalog"):
        for capacity in reversed(FRAME_CAPACITIES):
            name, prefix = f"audio{capacity}", f"audio{capacity}_"
            raw = raw_audio[prefix] = AudioInputs.allocate(capacity, 13 * (capacity // 100), device)
            buffers.update(raw.buffers(prefix))
            private = private_buffers.cursor()
            tree = weights.capacity_tree()
            encoder = PreparedAudioEncoder(tree.model.audio_tower, tree.model.multi_modal_projector,
                AudioMetadata.allocate(capacity, device, buffers=private), capacity=capacity,
                device=device, session=session, attention_workspace=arena, buffers=private)
            forward = PreparedAudioGeneration(encoder, raw, feature_state, completion, frame, sequence, buffers=private)
            add(name, forward, "audio_encode", 1, capacity, (*raw.buffers(prefix), "audio_feature_slot"))
        for mode, capacities in (("prefill", prefills), ("decode", reversed(batches))):
            for capacity in capacities:
                for initial in (initial_capacities if mode == "prefill" else (0,)):
                    name = f"{mode}{capacity}" + (f"_initial{initial}" if mode == "prefill" else "")
                    rows = 1 if mode == "prefill" else capacity
                    private = private_buffers.cursor()
                    current = metadata[name] = Metadata.allocate(capacity, rows, blocks, device, buffers=private)
                    scratch = Scratch.allocate(capacity if mode == "prefill" else 8,
                        rows, device, buffers=graph_buffers.cursor())
                    decoder = PreparedDecoder(weights, weights.capacity_tree().model.language_model,
                        scratch, current, pool, mode=mode, session=session, name=name,
                        workspace=arena, buffers=private, feature_state=feature_state,
                        original_prompt_tokens=original_prompt_tokens[:1] if mode == "prefill" else None,
                        original_capacity=initial if mode == "prefill" else config.context)
                    batch = PreparedBatch(inputs, requests, current, mode=mode, shard=weights.shard, pool=pool,
                        validate_prefill=feature_state.validate_prefill if mode == "prefill" else None,
                        page_tokens=PAGE_TOKENS, buffers=private)
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=None,
                        coordinator=None, session=session, name=name + ".generation", workspace=arena, buffers=private,
                        sampling_selection=SamplingSelection.FIRST_MAXIMUM,
                        execution_errors=decoder.execution_errors)
                    fields = (*inputs.buffers(), "sampling_grammar_mask",
                        *(() if mode == "decode" else (*prefill_audio.buffers(), "original_prompt_tokens")))
                    add(name, forward, mode, rows, capacity if mode == "prefill" else 1, fields,
                        initial_prompt_capacity=initial)
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for capacity in reversed(batches):
            name = f"resume{capacity}"
            add(name, PreparedResume(weights, requests, pool, inputs, completion, frame, sequence,
                rows=capacity, runtime=None, vocabulary_runtime=None, coordinator=None, session=session,
                name=name, buffers=private_buffers.cursor(), sampling_selection=SamplingSelection.FIRST_MAXIMUM,
                projection_row_capacity=8, vocabulary_projection=_Vocabulary.prepare_resume),
                "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"))
            for mode in ("init", "restore", "snapshot"):
                name = f"{mode}{capacity}"
                forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                    runtime=None, coordinator=None, session=session, name=name, capacity=capacity,
                    admission_buffers=admission, draft_state=False, page_tokens=PAGE_TOKENS,
                    buffers=private_buffers.cursor())
                fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()))
    from ignition.preparation.failure import retain_on_failed_drain

    with retain_on_failed_drain(device, session, forwards, weights, package_storage, transient), phase(progress, "Prepare and tune ASR kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device,
        stream=stream, pool=graph_pool, compile_decoder=config.compile_decoder, progress=progress,
        cache_root=config.kernel_cache, retain_graphs=config.retain_graphs,
        materialize=pool.clear_for_preparation, warmup_storage=package_storage)
    for graph in graphs.values():
        failed_owners.callback(graph.reset)
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    contract = prepare_contract(signatures, buffers, pool, feature_state, raw_audio)
    server = dict(identity=identity, model_id="Qwen/Qwen3-ASR-1.7B", max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=PAGE_TOKENS, frontend=frontend, assets=assets,
        sampling_defaults=RECIPE.sampling_defaults.as_dict(), max_new_tokens=1024, model_contract=contract)
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=0, world_size=1, device=config.device, weights_bytes=weights.allocated_bytes,
        mapped_host_bytes=0, offloaded_bytes=0, state_rows=state_rows, continuation_bytes=pool.backing.nbytes,
        preparation_state_rows=int(pool.memory_capacity[0].item()), pages=pages, page_bytes=page_bytes(),
        page_tokens=PAGE_TOKENS, kv_bytes=pages * page_bytes(), kv_dtype="bfloat16",
        completion_output_bytes=completion.nbytes, feature_bytes=feature_state.arena.nbytes,
        operator_scratch_bytes=arena.nbytes, graph_buffer_bytes=graph_buffers.nbytes,
        private_buffer_bytes=private_buffers.nbytes, graphs=len(graphs), prefill_capacities=prefills,
        audio_frame_capacities=FRAME_CAPACITIES, free_device_bytes=free, total_device_bytes=total,
        headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder,
        numerical_policy=POLICY, bf16_reduced_precision_reduction=False,
        decoder_projection_physical_rows=8,
        decoder_compute_physical_rows=8,
        prefill_original_prompt_physical_rows=config.context,
        prefill_initial_prompt_capacities=initial_capacities,
        bf16_reduction_split_k=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
    roots = dict(audio=weights.tree.model.audio_tower, projector=weights.tree.model.multi_modal_projector,
                 decoder=weights.tree.model.language_model, head=weights.head)
    return PreparedModel(rebuild_weights=declare_weights(roots, components=tuple(roots)),
        device=config.device, graphs=graphs, buffers=buffers, signatures=signatures, max_active=config.max_active,
        session=session, server_config=server, memory_report=memory,
        export_zero_ranges=(*storage_ranges([metadata, graph_buffers.chunks, private_buffers.chunks,
                                             arena.tensor, feature_state.arena]), *completion.zero_ranges()),
        resources=(weights, pool, requests, inputs, admission, commands, frame, sequence, completion,
                   prefill_audio, feature_state, original_prompt_tokens, session, arena, graph_buffers, private_buffers,
                   forwards, metadata, raw_audio, compiled, stream, graph_pool))
