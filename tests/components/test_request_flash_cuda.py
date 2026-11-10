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


@pytest.mark.parametrize("capacity", [1, 64, 128])
@torch.no_grad()
def test_sequence_geometry_keeps_full_prompt_bytes_in_intermediate_fragments(capacity):
    from torch.nn.attention import sdpa_kernel, SDPBackend

    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        owners.append(images)
        def allocate(shape, dtype=torch.bfloat16):
            result = torch.zeros(shape, dtype=dtype, device="cuda")
            owners.append(result)
            return result
        query = allocate((2 * capacity, 16, 128))
        key, value = allocate((64, 256, 8, 128)), allocate((64, 256, 8, 128))
        guard = allocate((2 * capacity + 2, 16, 128))
        output = guard[1:-1]
        pages = allocate((2, 32), torch.int32)
        lengths, original = allocate((2,), torch.int32), allocate((2,), torch.int32)
        cu, cu_key = allocate((3,), torch.int32), allocate((3,), torch.int32)
        operation = PreparedRequestFlashAttention(query, key, value, pages, lengths, cu, cu_key,
            output, max_query=capacity, sequence_lengths=original)
        owners.append(operation)
        assert len(operation.family_geometry) // 3 < 104
        compiled = compile_forward(operation)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        sources = [allocate((2, 8192, heads, 128)) for heads in (16, 8, 8)]
        for identity in range(2):
            generator = torch.Generator(device="cuda").manual_seed(20261106 + identity)
            for source in sources:
                source[identity].normal_(std=.2, generator=generator)
        for counts in [(1, 19), (19, 38), (123, 165), (129, 192), (165, 443), (8192, 8192)]:
            references = [torch.nn.functional.scaled_dot_product_attention(
                sources[0][identity, :count].transpose(0, 1)[None],
                sources[1][identity, :count].transpose(0, 1)[None],
                sources[2][identity, :count].transpose(0, 1)[None],
                is_causal=True, enable_gqa=True)[0].transpose(0, 1).contiguous().cpu()
                for identity, count in enumerate(counts)]
            for part in ("first", "middle", "tail", "remainder", "last"):
                for identities in ([0, 1], [1, 0], [None, 0], [None, None], [0, 1]):
                    query.fill_(float("nan")); key.fill_(float("nan")); value.fill_(float("nan"))
                    pages.fill_(-1); lengths.zero_(); original.fill_(-2147483648)
                    cumulative, consumed, expected = [0], [0], []
                    for slot, identity in enumerate(identities):
                        if identity is None:
                            cumulative.append(cumulative[-1]); consumed.append(consumed[-1])
                            continue
                        count = counts[identity]
                        end = min(count, capacity) if part == "first" else min(count, 2 * capacity) if part == "middle" else count
                        start = (end - 1 if part == "last" else (count - 1) // capacity * capacity
                            if part == "remainder" else max(0, end - capacity))
                        live = end - start
                        query[cumulative[-1]:cumulative[-1] + live].copy_(sources[0][identity, start:end])
                        key[32 * slot:32 * (slot + 1)].view(8192, 8, 128)[:end].copy_(sources[1][identity, :end])
                        value[32 * slot:32 * (slot + 1)].view(8192, 8, 128)[:end].copy_(sources[2][identity, :end])
                        pages[slot, :(end + 255) // 256].copy_(torch.arange(
                            32 * slot, 32 * slot + (end + 255) // 256, dtype=torch.int32, device="cuda"))
                        lengths[slot], original[slot] = end, count
                        cumulative.append(cumulative[-1] + live); consumed.append(consumed[-1] + end)
                        expected.append(references[identity][start:end])
                    cu.copy_(torch.tensor(cumulative, dtype=torch.int32, device="cuda"))
                    cu_key.copy_(torch.tensor(consumed, dtype=torch.int32, device="cuda"))
                    frozen = [tensor.contiguous().view(torch.uint8).clone()
                        for tensor in (query, key, value, pages, lengths, original, cu, cu_key)]
                    for _ in range(5):
                        guard.fill_(137); torch.cuda.synchronize()
                        before = torch.cuda.memory_stats(); graph.replay(); torch.cuda.synchronize()
                        after = torch.cuda.memory_stats()
                        for field in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                            assert before[field] == after[field], field
                        assert operation.errors.eq(0).all(), (capacity, counts, part, identities)
                        actual = output[:cumulative[-1]].cpu().contiguous()
                        reference = torch.cat(expected) if expected else actual
                        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)), (capacity, counts, part, identities)
                        assert output[cumulative[-1]:].eq(0).all() and guard[[0, -1]].eq(137).all()
                    for tensor, saved in zip((query, key, value, pages, lengths, original, cu, cu_key), frozen, strict=True):
                        assert torch.equal(tensor.contiguous().view(torch.uint8), saved)
        # Invalid geometry remains a device error, including int32 overflow
        # controls. An inactive slot's poisoned metadata stays ignored.
        query.fill_(0); key.fill_(0); value.fill_(0)
        pages.copy_(torch.arange(64, dtype=torch.int32, device="cuda").view(2, 32))
        lengths.copy_(torch.tensor([0, 165], dtype=torch.int32, device="cuda"))
        cu.copy_(torch.tensor([0, 0, 1], dtype=torch.int32, device="cuda"))
        cu_key.copy_(torch.tensor([0, 0, 165], dtype=torch.int32, device="cuda"))
        for bad in (-2147483648, -1, 0, 1, 164, 8193, 2147483647):
            original.copy_(torch.tensor([-2147483648, bad], dtype=torch.int32, device="cuda"))
            graph.replay(); torch.cuda.synchronize()
            assert operation.errors.cpu().tolist() == [0, 0, 1]
            assert torch.isnan(output[:1]).all() and output[1:].eq(0).all()
        original[1] = 165
        graph.replay(); torch.cuda.synchronize()
        assert operation.errors.eq(0).all() and output.eq(0).all()
        for invalid in (original.float(), original[:1], original.repeat_interleave(2)[::2]):
            with pytest.raises(ValueError):
                PreparedRequestFlashAttention(query, key, value, pages, lengths, cu, cu_key,
                    output, max_query=capacity, sequence_lengths=invalid)
        with pytest.raises(ValueError):
            PreparedRequestFlashAttention(query, key, value, pages, lengths, cu, cu_key,
                output, max_query=capacity, num_splits=1, sequence_lengths=original)
        with pytest.raises(ValueError):
            PreparedRequestFlashAttention(query, key, value, pages, lengths, cu, cu_key,
                output, max_query=capacity, sequence_lengths=output.view(torch.int32).reshape(-1)[:2])
        graph.reset()


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
        baselines, covered = {}, set()
        # Visit both sides and the exact edge of every 128-key split block.
        # The original small selection missed most prepared split families.
        boundaries = sorted({1, 8192, *[edge + delta for edge in range(128, 8192, 128)
            for delta in (-1, 0, 1)]})
        for size in (123, 165, 444, 768, 769, 897, 8192, 769, *boundaries):
            covered.add(int(operation.lookup[0, (size + 127) // 128]))
            references = [torch.nn.functional.scaled_dot_product_attention(
                source_q[i:i + 1].transpose(0, 1)[None],
                source_k[i, :size].transpose(0, 1)[None], source_v[i, :size].transpose(0, 1)[None],
                is_causal=False, enable_gqa=True)[0, :, 0].cpu() for i in range(batch)]
            independent = []
            for identity in range(batch):
                keys = source_k[identity, :size].double().repeat_interleave(2, dim=1)
                values = source_v[identity, :size].double().repeat_interleave(2, dim=1)
                scores = torch.einsum("hd,thd->ht", source_q[identity].double(), keys) * (128 ** -.5)
                independent.append(torch.einsum("ht,thd->hd", torch.softmax(scores, -1), values).cpu())
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
                for _ in range(5):
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
                        assert torch.isfinite(independent[identity]).all()
                        assert torch.allclose(actual[packed].double(), independent[identity], atol=.002, rtol=.02), (batch, size, slots, identity)
                        identity_key = (size, identity)
                        if identity_key in baselines:
                            assert torch.equal(actual[packed].view(torch.uint8), baselines[identity_key].view(torch.uint8))
                        baselines[identity_key] = actual[packed].clone()
                    for tensor, original in zip((query, key, value), frozen, strict=True):
                        assert torch.equal(tensor.view(torch.uint8), original)
        assert covered == set(range(len(operation.family_geometry) // 3))
        graph.reset()


@pytest.mark.parametrize("num_splits", [None, 1])
@torch.no_grad()
def test_compiled_general_query_catalog_preserves_every_original_family(num_splits):
    from ignition.components.request_flash import _attend

    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        owners.append(images)
        def allocate(shape, dtype=torch.bfloat16):
            tensor = torch.zeros(shape, dtype=dtype, device="cuda")
            owners.append(tensor)
            return tensor
        query = allocate((128, 16, 128))
        key, value = allocate((32, 256, 8, 128)), allocate((32, 256, 8, 128))
        guard, original = allocate((130, 16, 128)), allocate(query.shape)
        output = guard[1:-1]
        pages, lengths = allocate((1, 32), torch.int32), allocate((1,), torch.int32)
        cu, cu_key = allocate((2,), torch.int32), allocate((2,), torch.int32)
        operation = PreparedRequestFlashAttention(query, key, value, pages, lengths, cu, cu_key,
            output, max_query=128, num_splits=num_splits)
        owners.append(operation)
        errors = allocate(operation.errors.shape, torch.int32)
        compiled = compile_forward(operation)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        table = operation.lookup.cpu().tolist()
        representatives = {}
        for queries in (1, 2, 65, 128):
            row = 0 if queries == 1 else (queries + 63) // 64
            for blocks in range(1, len(table[row])):
                if blocks * 128 >= queries:
                    representatives.setdefault(table[row][blocks], (queries, blocks * 128))
        assert set(representatives) == set(range(len(operation.family_geometry) // 3))
        cases = list(representatives.values()) + [(0, 0), (13, 257), (123, 8192), (128, 8192), (13, 257)]
        pages.copy_(torch.arange(32, device="cuda", dtype=torch.int32)[None])
        for pattern in ("signed_zero", "rounding"):
            query.zero_()
            query[:, :, 0] = 1
            key.zero_()
            value.fill_(1)
            if pattern == "signed_zero":
                value.zero_()
                key[:, -128:, :, 0] = -640
                value[:, -128:, :, 0] = -2.**-80
            else:
                value[:, ::2] += 2.**-7
            for queries, keys in cases:
                cu[1], lengths[0] = queries, keys
                _attend(query, key, value, pages, lengths, cu, original, errors,
                    operation.lookup, operation.family_geometry, 128)
                torch.cuda.synchronize()
                expected = original.cpu().clone()
                expected_errors = errors.cpu().clone()
                assert torch.isfinite(expected).all() and expected_errors.eq(0).all()
                for _ in range(5):
                    guard.fill_(137)
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                        assert before[name] == after[name], name
                    assert torch.equal(output.cpu().view(torch.uint8), expected.view(torch.uint8)), (pattern, queries, keys)
                    assert torch.equal(operation.errors.cpu(), expected_errors)
                    assert output[queries:].eq(0).all() and guard[[0, -1]].eq(137).all()
        # Actual error classes, including a nonfinite key whose softmax weight
        # can be zero, must survive compilation without changes or recapture.
        cu[1], lengths[0] = 13, 257
        for tensor, index in ((query, (0, 0, 0)), (key, (0, 0, 0, 0)), (value, (0, 0, 0, 0))):
            saved = tensor[index].clone()
            for poison in (float("nan"), float("inf"), -float("inf")):
                tensor[index] = poison
                _attend(query, key, value, pages, lengths, cu, original, errors,
                    operation.lookup, operation.family_geometry, 128)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(operation.errors, errors)
                assert torch.equal(output.view(torch.uint8), original.view(torch.uint8))
            tensor[index] = saved
        for malformed in ("pages", "length", "cu"):
            if malformed == "pages": pages[0, 0] = -1
            if malformed == "length": lengths[0] = 12
            if malformed == "cu": cu[0] = 1
            _attend(query, key, value, pages, lengths, cu, original, errors,
                operation.lookup, operation.family_geometry, 128)
            graph.replay()
            torch.cuda.synchronize()
            assert errors.ne(0).any() and torch.equal(operation.errors, errors)
            assert torch.equal(output.view(torch.uint8), original.view(torch.uint8))
            pages[0, 0], lengths[0], cu[0] = 0, 257, 0
        _attend(query, key, value, pages, lengths, cu, original, errors,
            operation.lookup, operation.family_geometry, 128)
        graph.replay()
        torch.cuda.synchronize()
        assert operation.errors.eq(0).all() and torch.equal(output.view(torch.uint8), original.view(torch.uint8))
        graph.reset()


@torch.no_grad()
def test_single_split_prefill_partitions_keep_identical_rows():
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        owners.append(images)
        def allocate(shape, dtype=torch.bfloat16):
            value = torch.zeros(shape, dtype=dtype, device="cuda")
            owners.append(value)
            return value
        generator = torch.Generator(device="cuda").manual_seed(20261106)
        q = allocate((443, 16, 128)).normal_(std=.2, generator=generator)
        k = allocate((2, 256, 8, 128)).normal_(std=.2, generator=generator)
        v = allocate(k.shape).normal_(std=.2, generator=generator)
        results = []
        for capacity, begin, count in ((512, 0, 443), (128, 0, 128), (256, 128, 256), (64, 384, 59)):
            query, guard = allocate((capacity, 16, 128)), allocate((capacity + 2, 16, 128))
            output = guard[1:-1]
            query[:count].copy_(q[begin:begin + count])
            pages = allocate((1, 32), torch.int32).fill_(-1)
            pages[0, :2] = torch.arange(2, dtype=torch.int32, device="cuda")
            lengths = allocate((1,), torch.int32).fill_(begin + count)
            cu, cu_key = allocate((2,), torch.int32), allocate((2,), torch.int32)
            cu[1], cu_key[1] = count, begin + count
            operation = PreparedRequestFlashAttention(query, k, v, pages, lengths, cu, cu_key,
                output, max_query=capacity, num_splits=1)
            owners.append(operation)
            compiled = compile_forward(operation)
            owners.append(compiled)
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with torch.cuda.graph(graph):
                compiled()
            prior = None
            for _ in range(5):
                guard.fill_(137)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[name] == after[name]
                actual = output.cpu()
                assert operation.errors.eq(0).all() and torch.isfinite(actual).all()
                assert actual[count:].eq(0).all() and guard[[0, -1]].eq(137).all()
                if prior is not None:
                    assert torch.equal(actual.view(torch.uint8), prior.view(torch.uint8))
                prior = actual
            results.append(prior[:count].contiguous())
        full, partitioned = results[0], torch.cat(results[1:])
        assert torch.equal(full.view(torch.uint8), partitioned.view(torch.uint8))
        # Independent causal FP64 softmax checks both sides of every boundary.
        keys, values = k.view(-1, 8, 128).double().repeat_interleave(2, 1), v.view(-1, 8, 128).double().repeat_interleave(2, 1)
        owners.extend((keys, values))
        for row in (0, 127, 128, 383, 384, 442):
            scores = torch.einsum("hd,thd->ht", q[row].double(), keys[:row + 1]) * (128 ** -.5)
            expected = torch.einsum("ht,thd->hd", scores.softmax(-1), values[:row + 1]).cpu()
            assert torch.allclose(full[row].double(), expected, atol=.002, rtol=.02)
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()


@pytest.mark.parametrize("num_splits", [None, 1])
@torch.no_grad()
def test_bad_metadata_is_visible_and_recovers_without_allocating(num_splits):
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
        operation = PreparedRequestFlashAttention(q, k, v, pages, lengths, cu, cu_key,
            output, max_query=1, num_splits=num_splits)
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


@pytest.mark.parametrize("num_splits", [None, 1])
@torch.no_grad()
@pytest.mark.parametrize("slots", [64, 4096])
def test_large_q1_catalog_preserves_sparse_owners_shared_pages_and_late_errors(slots, num_splits):
    from ignition.components.request_flash import _attend

    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        owners.append(images)
        def allocate(shape, dtype=torch.bfloat16):
            tensor = torch.zeros(shape, dtype=dtype, device="cuda")
            owners.append(tensor)
            return tensor
        query = allocate((slots, 16, 128))
        key, value = allocate((64, 256, 8, 128)), allocate((64, 256, 8, 128))
        guard = allocate((slots + 2, 16, 128))
        output = guard[1:-1]
        pages, lengths = allocate((slots, 32), torch.int32), allocate((slots,), torch.int32)
        cu, cu_key = allocate((slots + 1,), torch.int32), allocate((slots + 1,), torch.int32)
        operation = PreparedRequestFlashAttention(query, key, value, pages, lengths, cu, cu_key,
            output, max_query=1, num_splits=num_splits)
        owners.append(operation)
        compiled = compile_forward(operation)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        generator = torch.Generator(device="cuda").manual_seed(20261106)
        prototypes = allocate((8, 16, 128))
        for tensor in (prototypes, key, value):
            tensor.normal_(std=.2, generator=generator)
        representatives = {}
        for blocks, family in enumerate(operation.lookup[0].cpu().tolist()[1:], 1):
            representatives.setdefault(family, blocks * 128)
        assert set(representatives) == set(range(len(operation.family_geometry) // 3))
        sizes = sorted({1, 123, 165, 444, 768, 769, 897, 8192, *representatives.values()})
        reference = allocate((8 * len(sizes), 16, 128))
        one_query, one_output = allocate((1, 16, 128)), allocate((1, 16, 128))
        one_pages, one_length = allocate((1, 32), torch.int32), allocate((1,), torch.int32)
        one_cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        one_errors = allocate((2,), torch.int32)
        owners.append(one_cu)
        for identity in range(8):
            one_query.copy_(prototypes[identity:identity + 1])
            one_pages.copy_(torch.arange(32, device="cuda", dtype=torch.int32)[None] + 32 * (identity % 2))
            for index, size in enumerate(sizes):
                one_length.fill_(size)
                # The retained general program supplies the same original
                # single-request arithmetic without the batched owner packing.
                _attend(one_query, key, value, one_pages, one_length, one_cu,
                    one_output, one_errors, operation.lookup, operation.family_geometry, 1)
                assert one_errors.eq(0).all() and torch.isfinite(one_output).all()
                reference[identity * len(sizes) + index].copy_(one_output[0])
        arrangements = [list(range(slots)), [i for i in range(slots) if i % 3 != 1],
            [slots - 1], [], list(range(slots - 1, -1, -1)), list(range(slots))]
        frozen = [tensor.view(torch.uint8).clone() for tensor in (key, value)]
        for arrangement in arrangements:
            live = len(arrangement)
            occupied = set(arrangement)
            cumulative = [0]
            for slot in range(slots):
                cumulative.append(cumulative[-1] + int(slot in occupied))
            # Reverse identity assignment changes packed order without changing
            # the sorted slot packing and also exercises empty slots in front.
            ids = torch.tensor(list(reversed(arrangement)), device="cuda", dtype=torch.int64)
            sorted_slots = torch.tensor(sorted(arrangement), device="cuda", dtype=torch.int64)
            query.fill_(float("nan"))
            query[:live].copy_(torch.index_select(prototypes, 0, ids % 8))
            pages.fill_(-1)
            pages[sorted_slots] = torch.arange(32, device="cuda", dtype=torch.int32)[None] + 32 * (ids[:, None] % 2).int()
            lengths.zero_()
            chosen_sizes = torch.tensor(sizes, device="cuda", dtype=torch.int32)
            lengths[sorted_slots] = torch.index_select(chosen_sizes, 0, (ids // 8) % len(sizes))
            cu.copy_(torch.tensor(cumulative, device="cuda", dtype=torch.int32))
            target = torch.zeros_like(output)
            reference_rows = (ids % 8) * len(sizes) + (ids // 8) % len(sizes)
            target[:live].copy_(torch.index_select(reference, 0, reference_rows))
            for _ in range(5):
                guard.fill_(137)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for field in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                    assert before[field] == after[field], field
                assert operation.errors.eq(0).all()
                assert torch.equal(output.view(torch.uint8), target.view(torch.uint8))
                assert guard[[0, -1]].eq(137).all()
                for tensor, saved in zip((key, value), frozen, strict=True):
                    assert torch.equal(tensor.view(torch.uint8), saved)
        expected = output.clone()
        saved = query[-1].clone()
        query[-1, 0, 0] = float("nan")
        graph.replay()
        torch.cuda.synchronize()
        assert operation.errors[-1].item() == 1 and operation.errors[:-1].eq(0).all()
        assert not torch.isfinite(output[-1]).all()
        assert torch.equal(output[:-1].view(torch.uint8), expected[:-1].view(torch.uint8))
        query[-1].copy_(saved)
        graph.replay()
        torch.cuda.synchronize()
        assert operation.errors.eq(0).all()
        assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
        graph.reset()


@pytest.mark.parametrize("num_splits", [None, 1])
@torch.no_grad()
def test_mixed_split_families_match_fp64_and_survive_row_movement(num_splits):
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        owners.append(images)
        def allocate(shape, dtype=torch.bfloat16):
            value = torch.zeros(shape, dtype=dtype, device="cuda")
            owners.append(value)
            return value
        query = allocate((8, 16, 128))
        key, value = allocate((256, 256, 8, 128)), allocate((256, 256, 8, 128))
        guard = allocate((10, 16, 128))
        output = guard[1:-1]
        pages, lengths = allocate((8, 32), torch.int32), allocate((8,), torch.int32)
        cu, cu_key = allocate((9,), torch.int32), allocate((9,), torch.int32)
        operation = PreparedRequestFlashAttention(query, key, value, pages, lengths, cu, cu_key,
            output, max_query=1, num_splits=num_splits)
        owners.append(operation)
        compiled = compile_forward(operation)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        source_q = allocate(query.shape)
        source_k, source_v = allocate((8, 8192, 8, 128)), allocate((8, 8192, 8, 128))
        counts, references = [1, 13, 123, 257, 769, 897, 1025, 8192], []
        for identity, count in enumerate(counts):
            generator = torch.Generator(device="cuda").manual_seed(20261108 + identity)
            for source in (source_q[identity], source_k[identity], source_v[identity]):
                source.normal_(std=.2, generator=generator)
            keys = source_k[identity, :count].double().repeat_interleave(2, dim=1)
            values = source_v[identity, :count].double().repeat_interleave(2, dim=1)
            scores = torch.einsum("hd,thd->ht", source_q[identity].double(), keys) * (128 ** -.5)
            target = torch.einsum("ht,thd->hd", torch.softmax(scores, dim=-1), values).cpu()
            assert torch.isfinite(target).all()
            references.append(target)
        baselines = {}
        arrangements = [list(enumerate(range(8))), list(zip(range(8), reversed(range(8)))),
            [(7, 0)], [(0, 2), (3, 4), (7, 1)], [], list(enumerate(range(8)))]
        for arrangement in arrangements:
            arrangement = sorted(arrangement)
            query.fill_(float("nan"))
            key.fill_(float("nan"))
            value.fill_(float("nan"))
            pages.fill_(-1)
            lengths.zero_()
            cumulative, live = [0], {slot: identity for slot, identity in arrangement}
            for slot in range(8):
                cumulative.append(cumulative[-1] + int(slot in live))
            cu.copy_(torch.tensor(cumulative, device="cuda", dtype=torch.int32))
            for packed, (slot, identity) in enumerate(arrangement):
                count = counts[identity]
                query[packed].copy_(source_q[identity])
                lengths[slot] = count
                for index in range((count + 255) // 256):
                    page = slot * 32 + (31 - index)
                    pages[slot, index] = page
                    amount = min(256, count - index * 256)
                    key[page, :amount].copy_(source_k[identity, index * 256:index * 256 + amount])
                    value[page, :amount].copy_(source_v[identity, index * 256:index * 256 + amount])
            original = [tensor.view(torch.uint8).clone() for tensor in (query, key, value)]
            for repeat in range(5):
                guard.fill_(137)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[name] == after[name], name
                actual = output.cpu().contiguous()
                assert operation.errors.eq(0).all() and torch.isfinite(actual).all()
                assert guard[[0, -1]].eq(137).all() and actual[len(arrangement):].eq(0).all()
                for packed, (_, identity) in enumerate(arrangement):
                    assert torch.allclose(actual[packed].double(), references[identity], atol=.002, rtol=.02), (identity, repeat)
                    if identity in baselines:
                        assert torch.equal(actual[packed].view(torch.uint8), baselines[identity].view(torch.uint8))
                    baselines[identity] = actual[packed].clone()
                for tensor, saved in zip((query, key, value), original, strict=True):
                    assert torch.equal(tensor.view(torch.uint8), saved)
        # A failed row must remain visible without changing healthy rows. The
        # original represented request must recover without recapture.
        before = output.cpu().clone()
        query[0, 0, 0] = float("nan")
        graph.replay()
        torch.cuda.synchronize()
        assert operation.errors[1].item() != 0 and not torch.isfinite(output[0]).all()
        assert torch.equal(output[1:].cpu().view(torch.uint8), before[1:].view(torch.uint8))
        query[0].copy_(source_q[0])
        graph.replay()
        torch.cuda.synchronize()
        assert operation.errors.eq(0).all() and torch.equal(output.cpu().view(torch.uint8), before.view(torch.uint8))
        graph.reset()


@torch.no_grad()
def test_all_split_families_preserve_rounding_and_nonfinite_isolation():
    from ignition.components.request_flash import _attend

    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        owners.append(images)
        def allocate(shape, dtype=torch.bfloat16):
            tensor = torch.zeros(shape, dtype=dtype, device="cuda")
            owners.append(tensor)
            return tensor
        query = allocate((8, 16, 128))
        key, value = allocate((256, 256, 8, 128)), allocate((256, 256, 8, 128))
        guard = allocate((10, 16, 128))
        output, original = guard[1:-1], allocate(query.shape)
        pages, lengths = allocate((8, 32), torch.int32), allocate((8,), torch.int32)
        cu, cu_key = allocate((9,), torch.int32), allocate((9,), torch.int32)
        operation = PreparedRequestFlashAttention(query, key, value, pages, lengths, cu, cu_key, output, max_query=1)
        owners.append(operation)
        original_errors = allocate(operation.errors.shape, torch.int32)
        compiled = compile_forward(operation)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        lookup = operation.lookup[0].cpu().tolist()
        representatives = {}
        for blocks in range(1, len(lookup)):
            representatives.setdefault(lookup[blocks], blocks * 128)
        assert set(representatives) == set(range(len(operation.family_geometry) // 3))
        sizes = list(representatives.values())
        groups = [[128 * (i + 1) for i in range(8)]]
        groups += [[sizes[(begin + i) % len(sizes)] for i in range(8)] for begin in range(0, len(sizes), 8)]
        pages.copy_(torch.arange(256, device="cuda", dtype=torch.int32).reshape(8, 32))
        cu.copy_(torch.arange(9, device="cuda", dtype=torch.int32))
        for pattern in ("signed_zero", "rounding"):
            for counts in groups:
                query.zero_()
                query[:, :, 0] = 1
                key.fill_(float("nan"))
                value.fill_(float("nan"))
                lengths.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
                for identity, count in enumerate(counts):
                    keys = key[identity * 32:(identity + 1) * 32].view(8192, 8, 128)[:count]
                    values = value[identity * 32:(identity + 1) * 32].view(8192, 8, 128)[:count]
                    keys.zero_()
                    values.zero_()
                    if pattern == "signed_zero":
                        # Nonzero finite operands produce an underflowing
                        # contribution. Extra +0 FMAs may alter its zero sign.
                        keys[-128:, :, 0] = -640
                        values[-128:, :, 0] = (-1 if identity % 2 == 0 else 1) * 2.**-80
                    else:
                        values.fill_(1 + identity / 8)
                        values[::2] += 2.**-7
                _attend(query, key, value, pages, lengths, cu, original, original_errors,
                    operation.lookup, operation.family_geometry, 1)
                torch.cuda.synchronize()
                assert original_errors.eq(0).all() and torch.isfinite(original).all()
                expected = original.cpu().contiguous()
                for _ in range(5):
                    guard.fill_(137)
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                        assert before[name] == after[name], name
                    assert operation.errors.eq(0).all()
                    assert torch.equal(output.cpu().contiguous().view(torch.uint8), expected.view(torch.uint8)), (pattern, counts)
                    assert guard[[0, -1]].eq(137).all()
                for tensor, index in ((query, (0, 0, 0)), (key, (0, 0, 0, 0)), (value, (0, 0, 0, 0))):
                    saved = tensor[index].clone()
                    for poison in (float("nan"), float("inf"), -float("inf")):
                        tensor[index] = poison
                        _attend(query, key, value, pages, lengths, cu, original, original_errors,
                            operation.lookup, operation.family_geometry, 1)
                        graph.replay()
                        torch.cuda.synchronize()
                        # Preserve the original nonfinite classes. A -Inf key
                        # can have zero softmax weight and finite output; every
                        # nonfinite output must still publish an error.
                        assert torch.equal(operation.errors, original_errors)
                        assert torch.equal(output.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8))
                        if not torch.isfinite(output[0]).all():
                            assert operation.errors[1].item() != 0
                        if tensor is query:
                            assert operation.errors[1].item() != 0
                        assert operation.errors[2:].eq(0).all()
                        assert torch.equal(output[1:].cpu().contiguous().view(torch.uint8), expected[1:].view(torch.uint8))
                        tensor[index] = saved
                        graph.replay()
                        torch.cuda.synchronize()
                        assert operation.errors.eq(0).all()
                        assert torch.equal(output.cpu().contiguous().view(torch.uint8), expected.view(torch.uint8))
        graph.reset()


@pytest.fixture(scope="module")
def single_split_request_rows():
    # CPU rows persist across the parametrized B=1/2/4/8 programs.
    return {}


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@torch.no_grad()
def test_single_split_continuation_matches_prefill_and_packed_holes(batch, single_split_request_rows):
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
            output, max_query=1, num_splits=1)
        owners.append(operation)
        compiled = compile_forward(operation)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        # A separately captured prefill program supplies the bytewise baseline
        # for this same fixed-split arithmetic. Standard fixed-length SDPA below
        # remains an independent numerical reference, not a varlen byte oracle.
        prefills = {}
        for capacity in (512, 4096, 8192):
            pq = allocate((capacity, 16, 128))
            pg = allocate((capacity + 2, 16, 128))
            po = pg[1:-1]
            pk, pv = allocate((32, 256, 8, 128)), allocate((32, 256, 8, 128))
            pp = allocate((1, 32), torch.int32)
            pp[0].copy_(torch.arange(32, device="cuda", dtype=torch.int32))
            pl = allocate((1,), torch.int32)
            pc, pck = allocate((2,), torch.int32), allocate((2,), torch.int32)
            prefill = PreparedRequestFlashAttention(pq, pk, pv, pp, pl, pc, pck,
                po, max_query=capacity, num_splits=1)
            owners.append(prefill)
            prefill_compiled = compile_forward(prefill)
            owners.append(prefill_compiled)
            prefill_compiled()
            prefill_graph = torch.cuda.CUDAGraph()
            graphs.append(prefill_graph)
            with torch.cuda.graph(prefill_graph):
                prefill_compiled()
            prefills[capacity] = (pq, pk, pv, pp, pl, pc, pck, pg, po, prefill, prefill_graph)
        source_q = allocate(query.shape)
        source_k, source_v = allocate((batch, 8192, 8, 128)), allocate((batch, 8192, 8, 128))
        for identity in range(batch):
            # Request operands stay identical across the B=1/2/4/8 cases.
            generator = torch.Generator(device="cuda").manual_seed(20261106 + identity)
            for source in (source_q[identity], source_k[identity], source_v[identity]):
                source.normal_(std=.2, generator=generator)
        baselines, covered = {}, set()
        # Visit both sides and the exact edge of every 128-key split block.
        # The original small selection missed most prepared split families.
        boundaries = [1, 2, 13, 19, 27, 32, 63, 64, 127, 128, 129, 255, 256, 257, 442, 443, 769, 8192]
        for size in (123, 165, 444, 768, 769, 897, 8192, 769, *boundaries):
            covered.add(int(operation.lookup[0, (size + 127) // 128]))
            references, sdk_references = [], []
            for identity in range(batch):
                # The independent full causal prefill computes the same last
                # query at its original absolute key position. It does not use
                # the request packing or the continuation component.
                full_query = torch.zeros((1, 16, size, 128), device="cuda", dtype=torch.bfloat16)
                full_query[0, :, -1].copy_(source_q[identity])
                reference = torch.nn.functional.scaled_dot_product_attention(
                    full_query, source_k[identity, :size].transpose(0, 1)[None],
                    source_v[identity, :size].transpose(0, 1)[None],
                    is_causal=True, enable_gqa=True)[0, :, -1].cpu()
                sdk_references.append(reference)
                capacity = 512 if size <= 512 else 4096 if size <= 4096 else 8192
                pq, pk, pv, pp, pl, pc, pck, pg, po, prefill, prefill_graph = prefills[capacity]
                count = size
                pq.zero_()
                pq[count - 1].copy_(source_q[identity])
                pk.fill_(float("nan"))
                pv.fill_(float("nan"))
                pk.view(8192, 8, 128)[:size].copy_(source_k[identity, :size])
                pv.view(8192, 8, 128)[:size].copy_(source_v[identity, :size])
                pl.fill_(size)
                pc[1], pck[1] = count, size
                prefill_inputs = (pq, pk, pv, pp, pl, pc, pck)
                frozen_prefill = [tensor.contiguous().view(torch.uint8).clone() for tensor in prefill_inputs]
                previous = None
                for _ in range(5):
                    pg.fill_(137)
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    prefill_graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                        assert before[name] == after[name], name
                    assert prefill.errors.eq(0).all() and torch.isfinite(po).all()
                    assert po[count:].eq(0).all() and pg[[0, -1]].eq(137).all()
                    current = po[count - 1].cpu().clone()
                    if previous is not None:
                        assert torch.equal(current.view(torch.uint8), previous.view(torch.uint8))
                    previous = current
                for tensor, frozen_tensor in zip(prefill_inputs, frozen_prefill, strict=True):
                    assert torch.equal(tensor.contiguous().view(torch.uint8), frozen_tensor)
                references.append(previous)
            independent = []
            for identity in range(batch):
                keys = source_k[identity, :size].double().repeat_interleave(2, dim=1)
                values = source_v[identity, :size].double().repeat_interleave(2, dim=1)
                scores = torch.einsum("hd,thd->ht", source_q[identity].double(), keys) * (128 ** -.5)
                independent.append(torch.einsum("ht,thd->hd", torch.softmax(scores, -1), values).cpu())
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
                for _ in range(5):
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
                        assert torch.isfinite(sdk_references[identity]).all()
                        assert torch.allclose(actual[packed], sdk_references[identity], atol=.002, rtol=.02), (batch, size, slots, identity, "SDK")
                        assert torch.isfinite(independent[identity]).all()
                        assert torch.allclose(actual[packed].double(), independent[identity], atol=.002, rtol=.02), (batch, size, slots, identity)
                        identity_key = (size, identity)
                        if identity_key in baselines:
                            assert torch.equal(actual[packed].view(torch.uint8), baselines[identity_key].view(torch.uint8))
                        baselines[identity_key] = actual[packed].clone()
                        if identity_key in single_split_request_rows:
                            assert torch.equal(actual[packed].view(torch.uint8),
                                single_split_request_rows[identity_key].view(torch.uint8)), (batch, size, identity, "cross batch")
                        single_split_request_rows[identity_key] = actual[packed].clone()
                    for tensor, original in zip((query, key, value), frozen, strict=True):
                        assert torch.equal(tensor.view(torch.uint8), original)
        assert covered == set(range(len(operation.family_geometry) // 3))
        for captured in graphs:
            captured.reset()
