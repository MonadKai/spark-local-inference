# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Export real paged generation whose token operands trigger attention errors."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.paged_attention import PagedAttentionMetadata, PreparedPagedKVWriter
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.paged_state import PagedStateStorage
from ignition.components.parallel import TensorParallelShard
from ignition.components.request_flash import PreparedRequestFlashAttention
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState, SamplingSelection
from ignition.components.state import StateLayout
from ignition.components.state_transfer import StateCommands
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation import contract as c, runtime_package_pb2 as p
from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature
from ignition.preparation.model import PreparedModel
from ignition.preparation.failure import drain_or_retain, retain_on_failed_drain, retain_undrained
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.package_file import combine
from ignition.preparation.package_storage import PackageStorage
from ignition.preparation.scratch import GraphBufferPool
from ignition.preparation.tuning import RecordedPreparationSession


class Decoder:
    """Synthetic operands, real KV writes/attention/head and error reporting.

    Prompt token 3 fails prefill. Token 4 emits 7, whose next decode fails.
    Other tokens emit 2 until the requested output limit. No status or native
    request policy is injected by this test computation.
    """

    def __init__(self, metadata, pool, weights, *, mode, session, name, buffers):
        self.metadata, self.pool, self.weights = metadata, pool, weights
        self.mode = mode
        rows, tokens = metadata.lengths.numel(), metadata.token_ids.numel()
        zero = lambda shape, dtype=torch.bfloat16: buffers.zeros(shape, dtype)
        self.query = zero((tokens, 16, 128))
        self.key, self.value = zero((tokens, 8, 128)), zero((tokens, 8, 128))
        self.attended = zero(self.query.shape)
        self.scratch = SimpleNamespace(hidden=zero((rows, 2048)), logits=zero((rows, 256)))
        local_logits = zero((rows, 256))
        self.kv = PagedAttentionMetadata(metadata, page_tokens=256, buffers=buffers)
        key_cache, value_cache = (pool.pages[field].view(torch.bfloat16).view(-1, 256, 8, 128)
            for field in ("key", "value"))
        self.writer = PreparedPagedKVWriter(self.key, self.value, key_cache, value_cache,
            self.kv.slots, session=session, name=name + ".write")
        cumulative = zero(metadata.query_start_loc.shape, torch.int32)
        self.attention = PreparedRequestFlashAttention(self.query, key_cache, value_cache,
            metadata.page_table, self.kv.lengths, metadata.query_start_loc, cumulative,
            self.attended, max_query=metadata.max_query_tokens)
        self.execution_errors = zero((rows,), torch.int32)
        self.head = PreparedVocabularyProjection(self.scratch.hidden, weights.head, local_logits,
            self.scratch.logits, runtime=None, session=session, coordinator=None, name=name + ".head")
        self.auxiliary = None

    def __call__(self):
        m = self.metadata
        bad = (m.token_ids == 3) | ((m.token_ids == 7) if self.mode == "decode" else False)
        self.query.copy_(torch.where(bad[:, None, None], float("nan"), 0.))
        self.key.zero_()
        self.value.fill_(2.)
        self.kv()
        self.writer()
        self.attention()
        self.execution_errors.copy_((self.attention.errors[0] | self.attention.errors[1:]).int())
        self.scratch.hidden.copy_(self.attended.view(-1, 2048)[m.sample_indices])
        special = m.token_ids[m.sample_indices] == 4
        self.scratch.hidden[:, 0].copy_(special)
        self.scratch.hidden[:, 1].copy_(~special)
        self.head(self.scratch.hidden)


def frontend(root):
    from fixture_frontend import server_config
    config = server_config(root)
    config["sampling_defaults"]["top_k"] = 1
    vocab = {"<unk>" if i == 0 else "<eos>" if i == 1 else f"t{i}": i for i in range(256)}
    special_tokens = [dict(id=i, content=token, single_word=False, lstrip=False,
        rstrip=False, normalized=False, special=True) for i, token in ((0, "<unk>"), (1, "<eos>"))]
    raw = json.dumps(dict(version="1.0", model=dict(type="WordLevel", vocab=vocab, unk_token="<unk>"),
        added_tokens=special_tokens,
        decoder=dict(type="ByteLevel", add_prefix_space=False, trim_offsets=False, use_regex=False))).encode()
    template = b"{% for message in messages %}{{ message.content }}{% mark %}{% endfor %}"
    for name, contents in (("tokenizer", raw), ("chat_template", template)):
        path = root / f"{name}.txt"
        path.write_bytes(contents)
        config["assets"][name] = dict(path=str(path), sha256=hashlib.sha256(contents).hexdigest())
    return config


@torch.no_grad()
def _prepare(root, owners, sessions, captured):
    from signing import publisher
    from ignition.preparation.runtime_package import export_model

    torch.cuda.set_device(0)
    with retain_on_failed_drain(0, owners):
        with ModuleImages() as images, PackageStorage() as storage:
            owners.extend((images, storage))
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
            page_bytes = 256 * 8 * 128 * 2
            specs = [(name, (page_bytes,), torch.uint8) for name in ("key", "value")]
            layout = StateLayout.from_specs([*specs, ("last_hidden", (2048,), torch.bfloat16),
                ("consumed_tokens", (1,), torch.int64)])
            pool = PagedStateStorage.allocate(layout, [(name, shape) for name, shape, _ in specs],
                7, 32, device, storage=storage, maximum_rows=7, maximum_pages=32)
            requests = DeviceRequestState.allocate(device, vocab_size=256, capacity=2)
            inputs = BatchInputs.allocate(128, 4, device, capacity=2)
            commands = StateCommands.allocate(device, capacity=2)
            admission = AdmissionBuffers.allocate(device, vocab_size=256, capacity=2)
            completion = CompletionOutputs.allocate((2, 2, 17), device=device)
            frame = torch.zeros(1, dtype=torch.int32, device=device)
            sequence = torch.zeros(1, dtype=torch.int64, device=device)
            head = torch.zeros((256, 2048), dtype=torch.bfloat16, device=device)
            head[7, 0], head[2, 1] = 1., 1.
            weights = SimpleNamespace(head=head, draft=None, shard=TensorParallelShard(0, 1))
            buffers = dict(inputs.buffers(), **commands.buffers(),
                **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
                state_pool=pool.backing, memory_capacity=pool.memory_capacity,
                request_pending_token=requests.pending_token, sampling_grammar_mask=requests.grammar_mask,
                frame_index=frame, submission_sequence=sequence, **completion.buffers())
            private = GraphBufferPool(device)
            forwards, signatures = {}, {}
            owners.extend((pool, requests, inputs, commands, admission, completion, head, weights, buffers, private, forwards))
            session = RecordedPreparationSession(device=device, checkpoint="generation-error-fixture-v1",
                workload_schema="generation-error-fixture", numerical_policy="bf16-fixture-v1",
                compile_workers=1, autotune_effort="off", benchmark_storage=storage)
            sessions.append(session)
            session.__enter__()
            owners.append(session)
            def add(name, operation, kind, rows, tokens, fields):
                forwards[name] = operation
                signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields))
            for mode, capacities in (("prefill", (128,)), ("decode", (2, 1))):
                for capacity in capacities:
                    name = f"{mode}{capacity}"
                    rows = 1 if mode == "prefill" else capacity
                    scratch = private.cursor()
                    metadata = Metadata.allocate(capacity, rows, 4, device, buffers=scratch)
                    decoder = Decoder(metadata, pool, weights, mode=mode, session=session, name=name, buffers=scratch)
                    batch = PreparedBatch(inputs, requests, metadata, mode=mode, shard=weights.shard,
                        pool=pool, page_tokens=256, buffers=scratch)
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=None,
                        coordinator=None, session=session, name=name, workspace=None, buffers=scratch,
                        sampling_selection=SamplingSelection.FIRST_MAXIMUM,
                        execution_errors=decoder.execution_errors)
                    add(name, forward, mode, rows, capacity if mode == "prefill" else 1,
                        (*inputs.buffers(), "sampling_grammar_mask"))
            row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
            for rows in (2, 1):
                name = f"resume{rows}"
                operation = PreparedResume(weights, requests, pool, inputs, completion, frame, sequence,
                    rows=rows, runtime=None, vocabulary_runtime=None, coordinator=None, session=session,
                    name=name, buffers=private.cursor(), sampling_selection=SamplingSelection.FIRST_MAXIMUM)
                add(name, operation, "resume", rows, 1, (*row_inputs, "sampling_grammar_mask"))
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{rows}"
                    operation = PreparedMaintenance(mode, requests, pool, commands, inputs, completion,
                        frame, sequence, runtime=None, coordinator=None, session=session, name=name,
                        capacity=rows, admission_buffers=admission, draft_state=False,
                        page_tokens=256, buffers=private.cursor())
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, operation, mode, rows, 1, (*fields, "page_tables", *commands.buffers()))
            session.tune()
            graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device,
                stream=torch.cuda.Stream(), pool=storage.graph_pool.id, cache_root=root / "forwards",
                retain_graphs=True, materialize=pool.clear_for_preparation, warmup_storage=storage)
            owners.extend((graphs, compiled))
            captured.extend(graphs.values())
            session.freeze()
            pool.clear_for_preparation()
            continuation = c.state_view("state_pool", layout.row_bytes)
            pending = c.state_view("request_pending_token", 8)
            graph_inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()}}
            regions = [dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0)]
            regions += [c.history_region(buffers, "cache." + name, value, p.PAGED_KEY_VALUE)
                for name, value in pool.pages.items()]
            syntax = c.frontend(tools=p.QWEN_CODER, assistant_marker="assistant", message_end_markers=[], prefix_boundary=None)
            syntax.update(tools=None, reasoning=None)
            contract = c.model_contract(vocabulary=256, page_tokens=256,
                page_tables=[dict(buffer="page_tables", page_offset=0)], regions=regions,
                graphs=c.graph_contracts(signatures, graph_inputs, continuation=continuation, pending=pending), frontend=syntax)
            server = dict(frontend(root), identity=hashlib.sha256(b"generation-error-fixture-v1").hexdigest(),
                model_id="generation-error-fixture", max_sequence_tokens=1024, prefill_chunk_size=128,
                pages=32, page_tokens=256, state_rows=6, snapshots=4, max_new_tokens=8,
                model_contract=contract)
            prepared = PreparedModel(device=0, graphs=graphs, buffers=buffers, signatures=signatures,
                max_active=2, resources=(storage, pool, requests, forwards, private, session, *compiled.values()),
                server_config=server, memory_report={}, session=session)
            owners.append(prepared)
        export_model(prepared, images, storage, root / "rank.json", options={}, sources=True)
        combine(root / "package", [root / "rank.json"], [], publisher=publisher())
        prepared.close()
        print("exported real generation-error fixture", flush=True)


@torch.no_grad()
def prepare(root):
    owners, sessions, captured = [], [], []
    try:
        _prepare(root, owners, sessions, captured)
    except BaseException as error:
        if drain_or_retain(error, 0, owners, sessions, captured):
            try:
                for graph in captured:
                    graph.reset()
                for session in sessions:
                    session.close()
            except BaseException as cleanup_error:
                retain_undrained(error, owners, sessions, captured)
                error.add_note(f"fixture cleanup failed: {cleanup_error!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    from helios.tests.generation_error_package_fixture import prepare as prepare_importable
    prepare_importable(args.directory)
