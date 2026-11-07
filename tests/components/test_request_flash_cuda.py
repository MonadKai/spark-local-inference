# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured request attention keeps original GQA arithmetic and explicit errors."""

import os

import pytest
import torch

from ignition.components.request_flash import PreparedRequestFlashAttention
from ignition.preparation.compilation import compile_forward
from ignition.preparation.failure import retain_on_failed_drain
from ignition.preparation.module_images import ModuleImages
from tests.components.test_paged_flash_cuda import (
    test_paged_flash_decode_permutations_retirement_and_poisoned_storage as _check_decode,
    test_paged_flash_package_replays_without_python as _check_package,
    test_paged_flash_prefill_prefix_causal_alignment_and_page_boundaries as _check_prefill,
)

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


@pytest.mark.parametrize("physical", [1, 2, 4, 8])
def test_existing_decode_lifecycle_contract(physical):
    _check_decode(physical, attention_type=PreparedRequestFlashAttention)


@pytest.mark.parametrize("capacity", [128, 4096])
def test_existing_prefill_alignment_contract(capacity):
    _check_prefill(capacity, attention_type=PreparedRequestFlashAttention)


@pytest.mark.parametrize("mode,count", [("decode", 0), ("decode", 2), ("prefill", 13)])
def test_package_replay_after_preparation_exits(tmp_path, mode, count):
    _check_package(tmp_path, mode, count, arithmetic="request")


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@torch.no_grad()
def test_actual_gqa_long_keys_and_packed_holes_are_exact(batch):
    from torch.nn.attention import sdpa_kernel, SDPBackend

    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        owners.append(images)
        def allocate(shape, dtype=torch.bfloat16):
            tensor = torch.zeros(shape, dtype=dtype, device="cuda")
            owners.append(tensor)
            return tensor
        query = allocate((batch, 16, 128))
        key, value = allocate((32 * batch, 256, 8, 128)), allocate((32 * batch, 256, 8, 128))
        guard = allocate((batch + 2, 16, 128))
        output = guard[1:-1]
        pages = allocate((batch, 32), torch.int32)
        lengths = allocate((batch,), torch.int32)
        cu, cu_key = allocate((batch + 1,), torch.int32), allocate((batch + 1,), torch.int32)
        operation = PreparedRequestFlashAttention(query, key, value, pages, lengths, cu, cu_key,
            output, max_query=1)
        owners.append(operation)
        compiled = compile_forward(operation)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        source_q = allocate(query.shape)
        source_k, source_v = allocate((batch, 8192, 8, 128)), allocate((batch, 8192, 8, 128))
        for identity in range(batch):
            # Request operands stay identical across the B=1/2/4/8 cases.
            generator = torch.Generator(device="cuda").manual_seed(20261106 + identity)
            for source in (source_q[identity], source_k[identity], source_v[identity]):
                source.normal_(std=.2, generator=generator)
        baselines = {}
        for size in (123, 165, 444, 768, 769, 897, 8192, 769):
            references = [torch.nn.functional.scaled_dot_product_attention(
                source_q[i:i + 1].transpose(0, 1)[None],
                source_k[i, :size].transpose(0, 1)[None], source_v[i, :size].transpose(0, 1)[None],
                is_causal=False, enable_gqa=True)[0, :, 0].cpu() for i in range(batch)]
            for slots in (list(range(batch)), list(reversed(range(batch))), [batch - 1], [], list(range(batch))):
                # A request identity moves between slots, while query rows are
                # packed in ascending slot order and absent slots retain poison.
                ordered = sorted((slot, identity) for identity, slot in enumerate(slots))
                query.fill_(float("nan"))
                key.fill_(float("nan"))
                value.fill_(float("nan"))
                pages.fill_(-1)
                lengths.zero_()
                cumulative = [0]
                for slot in range(batch):
                    cumulative.append(cumulative[-1] + int(slot in slots))
                cu.copy_(torch.tensor(cumulative, device="cuda", dtype=torch.int32))
                for packed, (slot, identity) in enumerate(ordered):
                    query[packed].copy_(source_q[identity])
                    key[32 * slot:32 * (slot + 1)].view(8192, 8, 128)[:size].copy_(source_k[identity, :size])
                    value[32 * slot:32 * (slot + 1)].view(8192, 8, 128)[:size].copy_(source_v[identity, :size])
                    lengths[slot] = size
                    pages[slot, :(size + 255) // 256].copy_(torch.arange(
                        32 * slot, 32 * slot + (size + 255) // 256, device="cuda", dtype=torch.int32))
                frozen = [tensor.view(torch.uint8).clone() for tensor in (query, key, value)]
                for _ in range(2):
                    guard.fill_(137)
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                        assert before[name] == after[name], name
                    actual = output.cpu().contiguous()
                    assert operation.errors.eq(0).all()
                    assert torch.isfinite(actual).all() and actual[len(slots):].eq(0).all()
                    assert guard[[0, -1]].eq(137).all()
                    for packed, (_, identity) in enumerate(ordered):
                        assert torch.equal(actual[packed].view(torch.uint8), references[identity].contiguous().view(torch.uint8)), (batch, size, slots, identity)
                        identity_key = (size, identity)
                        if identity_key in baselines:
                            assert torch.equal(actual[packed].view(torch.uint8), baselines[identity_key].view(torch.uint8))
                        baselines[identity_key] = actual[packed].clone()
                    for tensor, original in zip((query, key, value), frozen, strict=True):
                        assert torch.equal(tensor.view(torch.uint8), original)
        graph.reset()


@torch.no_grad()
def test_bad_metadata_is_visible_and_recovers_without_allocating():
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        owners.append(images)
        def allocate(shape, dtype=torch.bfloat16):
            tensor = torch.zeros(shape, dtype=dtype, device="cuda")
            owners.append(tensor)
            return tensor
        q, output = allocate((2, 16, 128)), allocate((2, 16, 128))
        k, v = allocate((1, 256, 8, 128)), allocate((1, 256, 8, 128))
        pages, lengths = allocate((2, 1), torch.int32), allocate((2,), torch.int32)
        cu, cu_key = allocate((3,), torch.int32), allocate((3,), torch.int32)
        operation = PreparedRequestFlashAttention(q, k, v, pages, lengths, cu, cu_key, output, max_query=1)
        owners.append(operation)
        compiled = compile_forward(operation)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        cases = [([0, 1, 1], [0, 0], [[0], [-1]]),
            ([0, 1, 1], [257, 0], [[0], [-1]]),
            ([0, 2, 2], [2, 0], [[0], [-1]]),
            ([0, 1, 1], [1, 0], [[1], [-1]]),
            ([0, 1, 0], [1, 0], [[0], [-1]]),
            ([1, 2, 2], [1, 0], [[0], [-1]]),
            ([0, 1, 3], [1, 2], [[0], [0]])]
        for bad_cu, bad_lengths, bad_pages in cases:
            cu.copy_(torch.tensor(bad_cu, device="cuda", dtype=torch.int32))
            lengths.copy_(torch.tensor(bad_lengths, device="cuda", dtype=torch.int32))
            pages.copy_(torch.tensor(bad_pages, device="cuda", dtype=torch.int32))
            graph.replay()
            torch.cuda.synchronize()
            assert operation.errors.ne(0).any() and not torch.isfinite(output[0]).all()
            # Inactive slots may keep arbitrary unusable length/page values.
            cu.copy_(torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32))
            lengths.copy_(torch.tensor([1, -1], device="cuda", dtype=torch.int32))
            pages.copy_(torch.tensor([[0], [-1]], device="cuda", dtype=torch.int32))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[name] == after[name], name
            assert operation.errors.eq(0).all() and output.eq(0).all()
        graph.reset()
