# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured paged flash preserves packed rows, tails and changing requests."""

import os

import pytest
import torch

from ignition.components.paged_flash import PreparedPagedFlashAttention
from ignition.preparation.compilation import compile_forward
from ignition.preparation.failure import retain_on_failed_drain
from ignition.preparation.module_images import ModuleImages

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


@torch.no_grad()
@pytest.mark.parametrize("physical", [1, 2, 4, 8])
def test_paged_flash_decode_permutations_retirement_and_poisoned_storage(
        physical, attention_type=PreparedPagedFlashAttention):
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        owners.append(images)
        values = []
        owners.append(values)
        def allocate(shape, dtype=torch.bfloat16):
            value = torch.zeros(shape, device="cuda:0", dtype=dtype)
            values.append(value)
            return value
        q = allocate((physical, 16, 128))
        k, v = allocate((2 * physical, 256, 8, 128)), allocate((2 * physical, 256, 8, 128))
        guard = allocate((physical * 16 * 128 + 32,))
        output = guard[16:-16].view_as(q)
        pages, lengths = allocate((physical, 32), torch.int32), allocate((physical,), torch.int32)
        cu, cu_key = allocate((physical + 1,), torch.int32), allocate((physical + 1,), torch.int32)
        op = attention_type(q, k, v, pages, lengths, cu, cu_key, output, max_query=1)
        owners.append(op)
        with pytest.raises(ValueError, match="alias"):
            attention_type(q, k, v, pages, lengths, cu, cu_key, q, max_query=1)
        with pytest.raises(ValueError, match="metadata"):
            attention_type(q, k, v, pages, lengths.cpu(), cu, cu_key, output, max_query=1)
        def forward():
            cu_key[:1].zero_()
            torch.cumsum(lengths, dim=0, dtype=torch.int32, out=cu_key[1:])
            return op()
        compiled = compile_forward(forward)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        generator = torch.Generator(device="cuda:0").manual_seed(20261106)
        sources = []
        owners.append(sources)
        for shape in ((physical, 16, 128), (physical, 512, 8, 128), (physical, 512, 8, 128)):
            sources.append(allocate(shape))
            sources[-1].normal_(std=.2, generator=generator)
        source_q, source_k, source_v = sources
        sizes = [13, 33, 129, 257, 511, 1, 256, 512][:physical]
        expected, baseline = {}, {}
        for identity, size in enumerate(sizes):
            expected[identity] = torch.nn.functional.scaled_dot_product_attention(
                source_q[identity:identity+1].transpose(0, 1)[None],
                source_k[identity, :size].repeat_interleave(2, dim=1).transpose(0, 1)[None],
                source_v[identity, :size].repeat_interleave(2, dim=1).transpose(0, 1)[None],
                is_causal=False)[0, :, 0].cpu()
        counts_to_test = [value for value in (1, 2, 4, 8) if value <= physical]
        layouts = [(count, reverse) for reverse in (False, True) for count in counts_to_test]
        for count, reverse in [*layouts, (0, False), (physical, False)]:
            identities = list(range(count))
            slots = [physical - 1 - i for i in range(count)] if reverse else list(range(count))
            q.fill_(float("nan"))
            k.fill_(float("nan"))
            v.fill_(float("nan"))
            pages.fill_(-1)
            lengths.zero_()
            counts = [0] * physical
            for packed, (identity, slot) in enumerate(zip(identities, slots, strict=True)):
                counts[slot] = 1
                size = sizes[identity]
                q[packed].copy_(source_q[identity])
                k[2*slot:2*slot+2].view(512, 8, 128)[:size].copy_(source_k[identity, :size])
                v[2*slot:2*slot+2].view(512, 8, 128)[:size].copy_(source_v[identity, :size])
                lengths[slot] = size
                pages[slot, :(size + 255) // 256].copy_(torch.arange(2*slot, 2*slot+(size+255)//256,
                    device="cuda:0", dtype=torch.int32))
            # Query rows are packed in ascending request order.
            ordered = sorted(zip(slots, identities))
            for packed, (_, identity) in enumerate(ordered):
                q[packed].copy_(source_q[identity])
            cu.copy_(torch.tensor([0] + list(torch.tensor(counts).cumsum(0).tolist()), device="cuda:0", dtype=torch.int32))
            for _ in range(5):
                guard.fill_(137)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[name] == after[name], name
                actual = output.cpu().clone()
                assert torch.isfinite(actual).all() and actual[count:].eq(0).all()
                assert guard[:16].eq(137).all() and guard[-16:].eq(137).all()
                for packed, (_, identity) in enumerate(ordered):
                    assert torch.allclose(actual[packed], expected[identity], atol=.002, rtol=.02), {
                        "count": count, "reverse": reverse, "identity": identity, "slot": ordered[packed][0],
                        "cu": cu.cpu().tolist(), "cu_key": cu_key.cpu().tolist(),
                        "lengths": lengths.cpu().tolist(), "actual_max": actual[packed].abs().max().item()}
                    if identity in baseline:
                        assert torch.equal(actual[packed].view(torch.uint8), baseline[identity].view(torch.uint8))
                    baseline[identity] = actual[packed].clone()
        lengths[0] = sizes[0] + 1
        graph.replay()
        torch.cuda.synchronize()
        assert not torch.isfinite(output[0]).all()
        lengths[0] = sizes[0]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output[0].cpu().view(torch.uint8), baseline[0].view(torch.uint8))
        graph.reset()


@pytest.mark.parametrize("capacity", [128, 4096])
@torch.no_grad()
def test_paged_flash_prefill_prefix_causal_alignment_and_page_boundaries(
        capacity, attention_type=PreparedPagedFlashAttention):
    from torch.nn.attention.bias import causal_lower_right

    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        owners.append(images)
        values = []
        owners.append(values)
        def allocate(shape, dtype=torch.bfloat16):
            tensor = torch.zeros(shape, device="cuda", dtype=dtype)
            values.append(tensor)
            return tensor
        q = allocate((capacity, 16, 128))
        k, v = allocate((32, 256, 8, 128)), allocate((32, 256, 8, 128))
        guard = allocate((capacity * 16 * 128 + 32,))
        output = guard[16:-16].view_as(q)
        pages, lengths = allocate((1, 32), torch.int32), allocate((1,), torch.int32)
        cu, cu_key = allocate((2,), torch.int32), allocate((2,), torch.int32)
        op = attention_type(q, k, v, pages, lengths, cu, cu_key, output, max_query=capacity)
        owners.append(op)
        def forward():
            cu_key[:1].zero_()
            torch.cumsum(lengths, dim=0, dtype=torch.int32, out=cu_key[1:])
            return op()
        compiled = compile_forward(forward)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        generator = torch.Generator(device="cuda").manual_seed(20261106)
        source_q = allocate(q.shape)
        source_k, source_v = allocate((8192, 8, 128)), allocate((8192, 8, 128))
        for tensor in (source_q, source_k, source_v):
            tensor.normal_(std=.2, generator=generator)
        q_cpu, k_cpu, v_cpu = [tensor.cpu().double() for tensor in (source_q, source_k, source_v)]
        baseline = None
        for query_count, key_count in [(13, 257), (capacity, 8192), (capacity - 1, capacity + 256),
                (1, 256), (0, 0), (13, 257)]:
            q.fill_(float("nan"))
            k.fill_(float("nan"))
            v.fill_(float("nan"))
            pages.fill_(-1)
            q[:query_count].copy_(source_q[:query_count])
            for logical in range((key_count + 255) // 256):
                physical = 31 - logical
                pages[0, logical] = physical
                count = min(256, key_count - 256 * logical)
                k[physical, :count].copy_(source_k[logical*256:logical*256+count])
                v[physical, :count].copy_(source_v[logical*256:logical*256+count])
            lengths[0], cu[1] = key_count, query_count
            reference = None
            if query_count:
                # Full contiguous canonical comparison. Its lower-right mask
                # retains a real cache prefix when query_count != key_count.
                reference = torch.nn.functional.scaled_dot_product_attention(
                    source_q[:query_count].transpose(0, 1)[None],
                    source_k[:key_count].repeat_interleave(2, dim=1).transpose(0, 1)[None],
                    source_v[:key_count].repeat_interleave(2, dim=1).transpose(0, 1)[None],
                    attn_mask=causal_lower_right(query_count, key_count))[0].transpose(0, 1).cpu()
            repeated = None
            for _ in range(3):
                guard.fill_(137)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[name] == after[name], name
                actual = output.cpu().clone()
                assert torch.isfinite(actual).all() and actual[query_count:].eq(0).all()
                assert guard[:16].eq(137).all() and guard[-16:].eq(137).all()
                if reference is not None:
                    assert torch.allclose(actual[:query_count], reference, atol=.002, rtol=.02)
                if repeated is not None:
                    assert torch.equal(actual.view(torch.uint8), repeated.view(torch.uint8))
                repeated = actual
            if query_count:
                # Explicit FP64 softmax independently checks the alignment and
                # reduction on every short-case row and selected boundary rows
                # of the 4096-capacity case. The full tensor gate above is not
                # relabelled as an all-element FP64 comparison.
                rows = list(range(query_count)) if query_count <= 128 else sorted({0, 1, 12, 127, 128,
                    255, 256, query_count // 2, query_count - 2, query_count - 1})
                indices = torch.tensor(rows)
                expected = torch.empty((len(rows), 16, 128), dtype=torch.float64)
                visible = torch.arange(key_count)[None] <= key_count - query_count + indices[:, None]
                for head in range(16):
                    scores = q_cpu[indices, head] @ k_cpu[:key_count, head // 2].T / (128 ** .5)
                    scores.masked_fill_(~visible, -torch.inf)
                    expected[:, head] = scores.softmax(-1) @ v_cpu[:key_count, head // 2]
                assert torch.allclose(actual[indices], expected.bfloat16(), atol=.002, rtol=.02)
            if (query_count, key_count) == (13, 257):
                if baseline is not None:
                    assert torch.equal(actual.view(torch.uint8), baseline.view(torch.uint8))
                baseline = actual
        # Reading one poisoned key beyond the correct prefix must trip the
        # finite gate on the last causal query and then recover exactly.
        lengths[0] = 258
        graph.replay()
        torch.cuda.synchronize()
        assert not torch.isfinite(output[12]).all()
        lengths[0] = 257
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output.cpu().view(torch.uint8), baseline.view(torch.uint8))
        graph.reset()


@torch.no_grad()
def test_paged_flash_exact_packed_queries_and_split_combine_failure_control():
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        owners.append(images)
        generator = torch.Generator(device="cuda").manual_seed(20261106)
        q = torch.randn((4, 16, 128), device="cuda", dtype=torch.bfloat16, generator=generator) * .2
        k = torch.randn((2, 256, 8, 128), device="cuda", dtype=torch.bfloat16, generator=generator) * .2
        v = torch.randn(k.shape, device="cuda", dtype=torch.bfloat16, generator=generator) * .2
        v[0] += 1
        v[1] -= 1
        k[:, 165:].fill_(float("nan"))
        v[:, 165:].fill_(float("nan"))
        cu = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
        cu_key = torch.tensor([0, 165, 330], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([165, 165], device="cuda", dtype=torch.int32)
        pages = torch.tensor([[0], [1]], device="cuda", dtype=torch.int32)
        output = torch.empty_like(q)
        owners.extend((q, k, v, cu, cu_key, lengths, pages, output))
        targets = []
        for batch in range(2):
            query = q[2*batch:2*batch+2].cpu().double().transpose(0, 1)
            key = k[batch, :165].cpu().double().repeat_interleave(2, 1).transpose(0, 1)
            value = v[batch, :165].cpu().double().repeat_interleave(2, 1).transpose(0, 1)
            scores = query @ key.transpose(-1, -2) / (128 ** .5)
            visible = torch.arange(165)[None] <= 163 + torch.arange(2)[:, None]
            scores.masked_fill_(~visible[None], -torch.inf)
            targets.append((scores.softmax(-1) @ value).transpose(0, 1).bfloat16())
        expected = torch.cat(targets)
        op = PreparedPagedFlashAttention(q, k, v, pages, lengths, cu, cu_key, output, max_query=2)
        compiled = compile_forward(op)
        owners.extend((op, compiled))
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        for _ in range(5):
            output.fill_(137)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.allclose(output.cpu(), expected, atol=.002, rtol=.02)
        graph.reset()

        # Exact total Q and no empty slots isolate the pinned implementation's
        # combine defect from Flare's separately qualified capacity extension.
        # The sentinel marks the raw ATen return itself, not a copied output.
        warm = torch.ops.aten._flash_attention_forward(q, k, v, cu, cu_key, 2, 256,
            0., True, False, seqused_k=lengths, block_table=pages, num_splits=2)[0]
        owners.append(warm)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            rejected = torch.ops.aten._flash_attention_forward(q, k, v, cu, cu_key, 2, 256,
                0., True, False, seqused_k=lengths, block_table=pages, num_splits=2)[0]
        owners.append(rejected)
        for _ in range(5):
            rejected.fill_(137)
            graph.replay()
            torch.cuda.synchronize()
            actual = rejected.cpu()
            assert not torch.allclose(actual, expected, atol=.002, rtol=.02)
            assert torch.isfinite(actual[:2]).all() and actual[:2].ne(137).all()
            assert actual[2:].eq(137).all()
        graph.reset()


@pytest.mark.parametrize("mode,count", [("decode", 0), ("decode", 2), ("prefill", 13)])
def test_paged_flash_package_replays_without_python(tmp_path, mode, count, arithmetic="paged"):
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    subprocess.run(["cargo", "build", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools", "--bin", "test_fixed_runtime_package"],
        cwd=root, check=True, timeout=300)
    package = tmp_path / "paged-flash"
    preparation = subprocess.run([sys.executable, "-m", "helios.tests.paged_flash_package_fixture", str(package),
        "--mode", mode, "--count", str(count), "--arithmetic", arithmetic],
        cwd=root, text=True, capture_output=True, timeout=900)
    assert preparation.returncode == 0, preparation.stdout + preparation.stderr
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(root / "helios/target")))
    replay = subprocess.run([str(target / "debug/test_fixed_runtime_package"), str(package), "--replays", "5"],
        cwd=root, text=True, capture_output=True, timeout=180)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert "native fixed-address replay passed" in replay.stdout
