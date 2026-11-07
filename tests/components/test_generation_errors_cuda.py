# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Execution failure precedes hidden publication and token consumption."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import DeviceRequestState, RequestAdmission, SamplingSelection, ERROR, RUNNING
from ignition.preparation.compilation import compile_forward
from ignition.preparation.failure import retain_on_failed_drain
from ignition.preparation.module_images import ModuleImages
from tests.components.generation_error_support import AttentionDecoder

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an assigned GPU and FLARE_RUN_CUDA_TESTS=1")


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("sample_enabled", [False, True])
@pytest.mark.parametrize("fault_kind", ["metadata", "query", "value"])
@torch.no_grad()
def test_execution_error_precedes_all_state_commit_and_completion(mode, sample_enabled, fault_kind):
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        owners.append(images)
        device = torch.device("cuda")
        rows, capacity = (1, 64) if mode == "prefill" else (2, 2)
        requests = DeviceRequestState.allocate(device, vocab_size=256, capacity=rows)
        inputs = BatchInputs.allocate(capacity, 1, device, capacity=rows)
        metadata = Metadata.allocate(capacity, rows, 1, device)
        pool = SimpleNamespace(memory_capacity=torch.tensor([rows + 1, rows], device=device),
            last_hidden=torch.full((rows + 1, 2048), 37., dtype=torch.bfloat16, device=device))
        faults = torch.zeros(rows, dtype=torch.int32, device=device)
        decoder = AttentionDecoder(metadata, pool, faults, fault_kind=fault_kind)
        batch = PreparedBatch(inputs, requests, metadata, mode=mode,
            shard=TensorParallelShard(0, 1), pool=pool, page_tokens=256)
        outputs = CompletionOutputs.allocate((2, rows, 24), device=device)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.full((1,), 7, dtype=torch.int64, device=device)
        forward = PreparedGeneration(decoder, batch, outputs, frame, sequence, runtime=None,
            coordinator=None, session=None, name="execution-errors", workspace=None,
            sampling_selection=SamplingSelection.FIRST_MAXIMUM,
            execution_errors=decoder.execution_errors)
        owners.extend((requests, inputs, metadata, pool, faults, decoder, batch, outputs, frame, sequence, forward))
        compiled = compile_forward(forward)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        for generation in (1, 2):
            for failed in (True, False):
                for slot in range(rows):
                    requests.admit(slot, RequestAdmission(generation=generation, seed=0, max_new_tokens=8), [7] * 13)
                    if mode == "decode":
                        requests.pending_token[slot] = 7
                pool.last_hidden.fill_(37)
                inputs.row_slot_ids.copy_(torch.arange(rows - 1, -1, -1, device=device, dtype=torch.int32))
                inputs.row_generations.fill_(generation)
                inputs.live_rows.fill_(rows)
                inputs.request_state_rows[:, 0].copy_(torch.arange(1, rows + 1, device=device))
                inputs.page_tables[:, 0].copy_(torch.arange(rows, device=device, dtype=torch.int32))
                inputs.query_lengths.fill_(13 if mode == "prefill" else 1)
                inputs.token_ids.fill_(7)
                inputs.sample_enabled.fill_(int(sample_enabled))
                faults.zero_()
                faults[0] = int(failed)
                frozen = {name: getattr(requests, name).clone() for name in
                    ("consumed_tokens", "generated_tokens", "rng_counter", "histogram", "pending_token")}
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[name] == after[name], name
                failed_slot = rows - 1
                assert decoder.execution_errors[0] == int(failed)
                if fault_kind != "metadata":
                    assert torch.equal(decoder.key_lengths, metadata.starts + metadata.lengths)
                if fault_kind == "value":
                    # Page tables are request-slot indexed before packing. A
                    # reversed row map must poison the selected request's page.
                    assert torch.isnan(decoder.value[metadata.page_table[0, 0], 0, 0, 0]) == failed
                if failed:
                    assert requests.status[failed_slot] == ERROR
                    assert requests.error_code[failed_slot] != 0
                    for name, previous in frozen.items():
                        assert torch.equal(getattr(requests, name)[failed_slot], previous[failed_slot]), name
                    assert pool.last_hidden[failed_slot + 1].eq(37).all()
                    record = outputs.records[0, failed_slot]
                    assert record[5] == ERROR and record[6] != 0 and record[4] == -1
                    assert record[8] == 0 and record[15] == 1 and record[16:].eq(-1).all()
                    assert record[2] == frozen["consumed_tokens"][failed_slot]
                    assert record[3] == frozen["generated_tokens"][failed_slot]
                else:
                    assert requests.status.eq(RUNNING).all()
                    assert pool.last_hidden[1:].eq(2).all()
                for slot in range(rows):
                    if failed and slot == failed_slot:
                        continue
                    expected = 13 if mode == "prefill" else 1
                    assert requests.consumed_tokens[slot] == expected
                    assert requests.generated_tokens[slot] == int(sample_enabled)
                    assert requests.rng_counter[slot] == int(sample_enabled)
        graph.reset()


@pytest.mark.parametrize("world", [1, 4])
@torch.no_grad()
def test_fault_mapping_filters_stale_rows_and_preserves_existing_errors(world):
    # Scaling the reduction input checks the mapping arithmetic, not an actual
    # multi-rank collective or its transport/lifetime guarantees.
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        owners.append(images)
        device = torch.device("cuda")
        requests = DeviceRequestState.allocate(device, vocab_size=256, capacity=4)
        inputs = BatchInputs.allocate(4, 1, device, capacity=4)
        metadata = Metadata.allocate(4, 4, 1, device)
        pool = SimpleNamespace(memory_capacity=torch.tensor([5, 4], device=device))
        batch = PreparedBatch(inputs, requests, metadata, mode="decode",
            shard=TensorParallelShard(0, world), pool=pool, page_tokens=256)
        errors = torch.ones(4, dtype=torch.int32, device=device)
        owners.extend((requests, inputs, metadata, pool, batch, errors))
        def step():
            batch.fail_execution(batch.execution_validation(errors) * world)
        compiled = compile_forward(step)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        for _ in range(2):
            for slot in range(4):
                requests.admit(slot, RequestAdmission(generation=2, seed=0), [7])
            requests.status[1], requests.error_code[1] = ERROR, 32
            inputs.row_slot_ids.copy_(torch.tensor([3, 2, 1, -1], device=device, dtype=torch.int32))
            inputs.row_generations.copy_(torch.tensor([2, 1, 2, 2], device=device, dtype=torch.int64))
            inputs.live_rows.fill_(4)
            errors.fill_(1)
            batch.validation_input.fill_(42)
            graph.replay()
            torch.cuda.synchronize()
            assert requests.status.tolist() == [RUNNING, ERROR, RUNNING, ERROR]
            assert requests.error_code.tolist() == [0, 32, 0, 1]
            assert requests.consumed_tokens.eq(0).all() and requests.rng_counter.eq(0).all()
            # No current live participant: leftover flags must disappear.
            inputs.live_rows.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert batch.validation_input.eq(0).all() and batch.token_state.active.eq(0).all()
            assert requests.error_code.tolist() == [0, 32, 0, 1]
        graph.reset()


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "stride", "draft"])
def test_execution_fault_operand_rejects_bad_storage_and_unintegrated_draft(invalid):
    device = torch.device("cuda")
    decoder = SimpleNamespace(weights=SimpleNamespace(draft=None),
        scratch=SimpleNamespace(logits=torch.zeros((2, 256), device=device)))
    batch = SimpleNamespace(rows=2, validation_input=torch.zeros((1, 8), device=device))
    errors = torch.zeros(2, dtype=torch.int32, device=device)
    if invalid == "shape":
        errors = torch.zeros(3, dtype=torch.int32, device=device)
    elif invalid == "dtype":
        errors = errors.float()
    elif invalid == "device":
        errors = errors.cpu()
    elif invalid == "stride":
        errors = torch.zeros(4, dtype=torch.int32, device=device)[::2]
    else:
        decoder.weights.draft = object()
    with pytest.raises(ValueError, match="execution.errors"):
        PreparedGeneration(decoder, batch, None, None, None, runtime=None, coordinator=None,
            session=None, name="invalid-errors", workspace=None, execution_errors=errors)
