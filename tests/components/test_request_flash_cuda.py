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


@torch.no_grad()
def test_mixed_split_families_match_fp64_and_survive_row_movement():
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
        operation = PreparedRequestFlashAttention(query, key, value, pages, lengths, cu, cu_key, output, max_query=1)
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
