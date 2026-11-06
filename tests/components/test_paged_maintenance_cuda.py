# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native maintenance replay uses the cache's physical token page geometry."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_state import PagedStateStorage
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState, RequestAdmission
from ignition.components.state import StateLayout
from ignition.components.state_transfer import StateCommands
from ignition.preparation.graphs import PreparedGraphBundle
from tests.components.completion_support import completion_outputs


CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@CUDA
@pytest.mark.parametrize("page_tokens,maintenance_tokens", [(128, 128), (256, 256), (256, 128)])
def test_native_snapshot_restore_page_boundaries(page_tokens, maintenance_tokens):
    from ignition.preparation.failure import drain_or_retain, retain_undrained

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    width, slots = 8192 // page_tokens, 4
    specs = (("key", (page_tokens * 2,)), ("value", (page_tokens * 2,)))
    layout = StateLayout.from_specs([
        *((name, shape, torch.uint8) for name, shape in specs),
        ("last_hidden", (8,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)])
    pool = PagedStateStorage.allocate(layout, specs, 8, 2 * width + 1, "cuda")
    requests = DeviceRequestState.allocate("cuda", vocab_size=128, capacity=slots)
    commands = StateCommands.allocate("cuda", capacity=slots)
    admission = AdmissionBuffers.allocate("cuda", vocab_size=128, capacity=slots)
    inputs = SimpleNamespace(
        request_state_rows=torch.arange(1, 5, dtype=torch.int64, device="cuda")[:, None].expand(4, 8).contiguous(),
        row_slot_ids=torch.full((slots,), -1, dtype=torch.int32, device="cuda"),
        row_generations=torch.zeros(slots, dtype=torch.int64, device="cuda"),
        live_rows=torch.zeros(1, dtype=torch.int32, device="cuda"),
        page_tables=torch.full((slots, width), -1, dtype=torch.int32, device="cuda"))
    records = torch.full((2, slots, 32), -22, dtype=torch.int64, device="cuda")
    outputs = completion_outputs(records)
    frame = torch.zeros(1, dtype=torch.int32, device="cuda")
    sequence = torch.ones(1, dtype=torch.int64, device="cuda")
    graphs, operations, worker = {}, {}, None
    retained = False
    owners = (pool, requests, commands, admission, inputs, outputs, frame, sequence, operations, graphs)
    try:
        for mode in ("snapshot", "restore"):
            operations[mode] = PreparedMaintenance(mode, requests, pool, commands, inputs,
                outputs, frame, sequence, runtime=None, coordinator=None, session=None,
                name=mode, capacity=slots, admission_buffers=admission, draft_state=False,
                page_tokens=maintenance_tokens)
        for mode, operation in operations.items():
            graph = torch.cuda.CUDAGraph()
            graphs[mode] = graph
            with torch.cuda.graph(graph):
                operation()
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={"records": records},
            resources=owners)
        worker = bundle.start_worker()

        def replay(mode):
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(bundle.graph_indices[mode])
            result = worker.poll(timeout_ms=30000)
            assert result is not None and result.ticket == ticket and result.error is None
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key], key

        # Unassigned table entries remain -1, as under native incremental reservation.
        endpoints = (127, 128, 129, 255, 256, 257, 384, 4096, 4097, 8191, 8192)
        if maintenance_tokens != page_tokens:
            endpoints = (128, 129)
        for epoch, endpoint in enumerate(endpoints, 1):
            pool.backing.fill_(231)
            pool.region("consumed_tokens").zero_()
            pool.backing[1].fill_(17)
            pool.region("consumed_tokens")[1] = endpoint
            pool.last_hidden[1].fill_(0.5)
            for index, (_, cache) in enumerate(pool.partial_pages()):
                cache.copy_((torch.arange(cache.shape[1], device="cuda")[None, :]
                    + 37 * torch.arange(cache.shape[0], device="cuda")[:, None] + 29 * index).to(torch.uint8))
            inputs.page_tables.fill_(-1)
            pages = (endpoint + page_tokens - 1) // page_tokens
            inputs.page_tables[0, :pages] = 1 + (torch.arange(pages, device="cuda") * 7) % width
            inputs.page_tables[1, :pages] = width + 1 + (torch.arange(pages, device="cuda") * 11) % width
            requests.admit(0, RequestAdmission(epoch, 13, max_new_tokens=8), [5] * endpoint,
                consumed_tokens=endpoint)
            inputs.row_slot_ids[0], inputs.row_generations[0] = 0, epoch
            inputs.live_rows.fill_(1)
            commands.request_slots[0], commands.source_slots[0], commands.destination_slots[0] = 0, 1, 5
            commands.operation[0], commands.expected_consumed[0] = 0, endpoint
            commands.num_ops.fill_(1)
            records.fill_(-22)
            before = pool.backing.clone()
            cached = {name: cache.clone() for name, cache in pool.partial_pages()}
            expected = before[1].clone()
            if endpoint % page_tokens:
                physical = 1 + (endpoint // page_tokens * 7) % width
                for name, cache in pool.partial_pages():
                    region = layout.region(name)
                    expected[region.offset:region.offset + region.nbytes].copy_(cache[physical])
                if maintenance_tokens == page_tokens:
                    inputs.page_tables[0, endpoint // page_tokens] = -1
                    replay("snapshot")
                    assert commands.errors[0] == 8 and records[0, 0, 5:7].tolist() == [3, 72]
                    assert torch.equal(pool.backing, before)
                    for name, cache in pool.partial_pages():
                        assert torch.equal(cache, cached[name])
                    inputs.page_tables[0, endpoint // page_tokens] = physical
                    records.fill_(-22)
            replay("snapshot")

            if maintenance_tokens != page_tokens and endpoint == 129:
                # The old 128-token binding asks for missing logical page 1.
                assert commands.errors[0] == 8
                assert records[0, 0, 5:7].tolist() == [3, 72]
                assert torch.equal(pool.backing, before)
                for name, cache in pool.partial_pages():
                    assert torch.equal(cache, cached[name])
                continue
            assert not commands.errors.count_nonzero()
            assert records[0, 0, 2] == endpoint and records[0, 0, 5:7].tolist() == [1, 0]
            if maintenance_tokens != page_tokens:
                # At 128 the wrong binding reports success but loses the partial page.
                assert torch.equal(pool.backing[5], before[1])
                assert not torch.equal(pool.backing[5], expected)
                continue
            assert torch.equal(pool.backing[5], expected)
            assert torch.equal(pool.backing[:5], before[:5])
            assert torch.equal(pool.backing[6:], before[6:])
            assert records[0, 1:, :8].tolist() == [[0, 0, 0, 0, -1, 0, 0, 0]] * 3
            assert records[1].eq(-22).all()
            for name, cache in pool.partial_pages():
                assert torch.equal(cache, cached[name])
            immutable = pool.backing[5].clone()

            # Restore into another physical request slot and private tail page.
            admission.stage(0, 1, RequestAdmission(epoch, 99, max_new_tokens=8), [5] * endpoint)
            admission.admission_count.fill_(1)
            commands.request_slots[0], commands.source_slots[0], commands.destination_slots[0] = 1, 5, 2
            if endpoint % page_tokens:
                destination_page = width + 1 + (endpoint // page_tokens * 11) % width
                for _, cache in pool.partial_pages():
                    cache[destination_page].fill_(201)
            restore_before = pool.backing.clone()
            restore_cache = {name: cache.clone() for name, cache in pool.partial_pages()}
            if endpoint % page_tokens:
                inputs.page_tables[1, endpoint // page_tokens] = -1
                replay("restore")
                assert commands.errors[0] == 8 and records[0, 1, 5:7].tolist() == [3, 72]
                assert torch.equal(pool.backing, restore_before)
                for name, cache in pool.partial_pages():
                    assert torch.equal(cache, restore_cache[name])
                inputs.page_tables[1, endpoint // page_tokens] = destination_page
            replay("restore")
            assert not commands.errors.count_nonzero()
            assert torch.equal(pool.backing[2], immutable)
            assert torch.equal(pool.backing[:2], restore_before[:2])
            assert torch.equal(pool.backing[3:], restore_before[3:])
            for name, cache in pool.partial_pages():
                wanted = restore_cache[name]
                if endpoint % page_tokens:
                    wanted[destination_page].copy_(cached[name][physical])
                assert torch.equal(cache, wanted)
            assert requests.consumed_tokens[1] == endpoint and requests.seed[1] == 99
            assert requests.histogram[1, 5] == endpoint and requests.rng_counter[1] == 0
            assert records[0, 1, 2] == endpoint and records[0, 1, 5:7].tolist() == [1, 0]
    except BaseException as error:
        retained = not drain_or_retain(error, torch.device("cuda:0"), owners, worker)
        raise
    finally:
        if not retained:
            try:
                if worker is not None:
                    worker.close()
                torch.cuda.synchronize()
            except BaseException as error:
                retain_undrained(error, owners, worker)
                raise
            for graph in graphs.values():
                graph.reset()
