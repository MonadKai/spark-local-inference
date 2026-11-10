# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prompt/history composition must retain the complete reference computation."""

import os

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.preparation.compilation import compile_forward
from ignition.preparation.failure import retain_on_failed_drain
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.scratch import GraphBufferPool
from ignition.recipes.qwen3_asr.replay import PreparedReplayAttention

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an assigned GPU and FLARE_RUN_CUDA_TESTS=1")


@torch.no_grad()
def test_shared_projection_scratch_preserves_published_rows_and_decode_arithmetic(tmp_path):
    from ignition.components.reduction_projection import PreparedReductionProjection
    from ignition.recipes.qwen3_asr.numerics import fp32_reductions
    from ignition.recipes.qwen3_asr.replay import (
        PreparedReplayProjection, ReplayScratch, history_projection_count)
    from tests.components.test_reduction_projection_cuda import _session

    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images, fp32_reductions():
        private = GraphBufferPool("cuda")
        owners.extend((images, private))
        buffers = private.cursor()
        count, original, prompt = [buffers.zeros((1,), torch.int32) for _ in range(3)]
        history_count = buffers.zeros((1,), torch.int32)
        rows = torch.arange(128, device="cuda")
        positions = buffers.zeros((128,), torch.int64)
        scratch = ReplayScratch.allocate(128, 8192, buffers)
        sources, full_sources, modules, outputs, guards, operations, reference_ops = [], [], [], [], [], [], []
        generator = torch.Generator(device="cuda").manual_seed(20261106)
        with _session(tmp_path, owners, graphs) as session:
            for index, (width, columns) in enumerate(((2048, 2048), (2048, 1024), (2048, 1024),
                    (2048, 6144), (2048, 6144), (6144, 2048))):
                source = buffers.zeros((128, width), torch.bfloat16)
                full = buffers.zeros((8192, width), torch.bfloat16)
                full.normal_(std=.2, generator=generator)
                module = torch.nn.Linear(width, columns, bias=False, dtype=torch.bfloat16, device="cuda")
                module.weight.normal_(std=.02, generator=generator)
                guard = buffers.full((128 * columns + 16,), 37, torch.bfloat16)
                output = guard[8:-8].view(128, columns)
                operation = PreparedReplayProjection(source, module, output, count, prompt_count=prompt,
                    positions=positions, original=original, history_count=history_count,
                    scratch=scratch, session=session,
                    name=f"replay{index}", buffers=buffers, workspace=None)
                rs = buffers.zeros((8, width), torch.bfloat16)
                ro = buffers.zeros((8, columns), torch.bfloat16)
                rc = buffers.zeros((1,), torch.int32)
                reference = PreparedReductionProjection(rs, module.weight, ro, rc,
                    session=session, name=f"ordinary_decode{index}")
                sources.append(source); full_sources.append(full); modules.append(module)
                guards.append(guard); outputs.append(output); operations.append(operation)
                reference_ops.append((reference, rs, ro, rc))
            owners.extend((sources, full_sources, modules, outputs, guards, operations, reference_ops, scratch))
            session.tune()
            def forward():
                prompt.copy_(torch.minimum(count, (original.long() - positions[:1]).clamp(min=0)).int())
                history_count.copy_(history_projection_count(rows, count, positions, original))
                for operation, source in zip(operations, sources, strict=True):
                    operation(source)
            compiled = compile_forward(forward)
            owners.append(compiled)
            compiled()
            graph = torch.cuda.CUDAGraph(); graphs.append(graph)
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            distinguished = False
            for n, start, live in ((165, 0, 0), (19, 0, 19), (19, 0, 128), (165, 0, 128), (165, 128, 128),
                    (165, 165, 128), (165, 256, 128), (165, 8064, 128), (165, 0, 0)):
                original.fill_(n); count.fill_(live)
                positions.fill_(n + 999)
                positions[:live].copy_(torch.arange(start, start + live, device="cuda"))
                expected = []
                for source, full, module, (reference, rs, ro, rc) in zip(
                        sources, full_sources, modules, reference_ops, strict=True):
                    source.fill_(float("nan")); source[:live].copy_(full[start:start + live])
                    canonical = torch.nn.functional.linear(full[:n], module.weight).cpu()
                    ordinary = torch.zeros_like(outputs[len(expected)], device="cpu")
                    for begin in range(0, live, 8):
                        active = min(8, live - begin)
                        rs.fill_(float("nan")); rs[:active].copy_(source[begin:begin + active]); rc.fill_(active)
                        reference(rs)
                        ordinary[begin:begin + active].copy_(ro[:active].cpu())
                    target = ordinary.clone()
                    p = min(live, max(n - start, 0))
                    if p:
                        distinguished |= not torch.equal(canonical[start:start + p].view(torch.uint8), ordinary[:p].view(torch.uint8))
                        target[:p].copy_(canonical[start:start + p])
                    expected.append(target)
                for repeat in range(5):
                    for source in sources:
                        source[live:].fill_(float("inf") if repeat % 2 else float("nan"))
                    scratch.full_input.fill_(float("nan")); scratch.full_output.fill_(float("nan"))
                    scratch.history_output.fill_(float("nan"))
                    for guard in guards: guard.fill_(37)
                    torch.cuda.synchronize(); before = torch.cuda.memory_stats()
                    graph.replay(); torch.cuda.synchronize(); after = torch.cuda.memory_stats()
                    assert history_count.item() == (live if start + live > n and live else 0)
                    for field in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                        assert before[field] == after[field], field
                    # Check every published value after the final, differently
                    # shaped projection has reused both full and history scratch.
                    for index, (output, target, guard) in enumerate(zip(outputs, expected, guards, strict=True)):
                        assert torch.equal(output.cpu().contiguous().view(torch.uint8), target.view(torch.uint8)), (index, n, start, live)
                        assert guard[:8].eq(37).all() and guard[-8:].eq(37).all()
            assert distinguished, "operands failed to distinguish original and ordinary-decode arithmetic"
            for invalid in (-1, 129):
                count.fill_(invalid)
                graph.replay(); torch.cuda.synchronize()
                assert history_count.item() == invalid
                assert operations[-1].history_output.isnan().all()
            count.zero_()
            graph.replay(); torch.cuda.synchronize()
            assert history_count.item() == 0
            assert all(output.eq(0).all() for output in outputs)
            graph.reset()


@pytest.mark.parametrize("capacity", [64, 128])
@torch.no_grad()
def test_prompt_history_attention_matches_original_and_each_decode_position(capacity):
    from torch.nn.attention import sdpa_kernel, SDPBackend

    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        private = GraphBufferPool("cuda")
        owners.extend((images, private))
        buffers = private.cursor()
        def allocate(shape, dtype=torch.bfloat16):
            return buffers.zeros(shape, dtype)
        query = allocate((capacity, 16, 128))
        guard = allocate((capacity + 2, 16, 128))
        output = guard[1:-1]
        key, value = allocate((32, 256, 8, 128)), allocate((32, 256, 8, 128))
        original, prompt = allocate((1,), torch.int32), allocate((1,), torch.int32)
        lengths = allocate((1,), torch.int32)
        metadata = Metadata.allocate(capacity, 1, 32, "cuda", buffers=buffers)
        operation = PreparedReplayAttention(query, key, value, metadata, lengths, output,
            original=original, prompt_count=prompt, buffers=buffers)
        owners.append(operation)
        compiled = compile_forward(operation)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        generator = torch.Generator(device="cuda").manual_seed(20261106)
        source_q = allocate((8192, 16, 128))
        source_k, source_v = allocate((8192, 8, 128)), allocate((8192, 8, 128))
        for tensor in (source_q, source_k, source_v):
            tensor.normal_(std=.2, generator=generator)
        canonical = torch.nn.functional.scaled_dot_product_attention(
            source_q[:165].transpose(0, 1)[None], source_k[:165].transpose(0, 1)[None],
            source_v[:165].transpose(0, 1)[None], is_causal=True, enable_gqa=True)[0].transpose(0, 1).contiguous().cpu()
        cases = [(0, capacity), (64, capacity), (128, capacity), (164, 3), (165, 9),
            (192, capacity), (255, capacity), (8192 - capacity, capacity), (8191, 1), (0, 0)]
        for start, count in cases:
            original.fill_(165)
            prompt.fill_(min(count, max(165 - start, 0)))
            metadata.page_table.copy_(torch.arange(32, dtype=torch.int32, device="cuda")[None])
            metadata.live_counts[0] = count
            metadata.lengths.fill_(count)
            metadata.positions.fill_(-1)
            metadata.positions[:count].copy_(torch.arange(start, start + count, device="cuda"))
            lengths.fill_(start + count if count else 0)
            query.fill_(float("nan"))
            query[:count].copy_(source_q[start:start + count])
            key.view_as(source_k).copy_(source_k); value.view_as(source_v).copy_(source_v)
            references = []
            for position in range(start, start + count):
                if position < 165:
                    references.append(canonical[position])
                else:
                    references.append(torch.nn.functional.scaled_dot_product_attention(
                        source_q[position:position + 1].transpose(0, 1)[None],
                        source_k[:position + 1].transpose(0, 1)[None],
                        source_v[:position + 1].transpose(0, 1)[None],
                        is_causal=False, enable_gqa=True)[0, :, 0].cpu())
            expected = torch.stack(references) if references else torch.empty((0, 16, 128), dtype=torch.bfloat16)
            for _ in range(5):
                guard.fill_(137); torch.cuda.synchronize()
                before = torch.cuda.memory_stats(); graph.replay(); torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for field in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[field] == after[field], field
                assert operation.errors.eq(0).all(), (capacity, start, count)
                assert torch.equal(output[:count].cpu().contiguous().view(torch.uint8), expected.view(torch.uint8)), (capacity, start, count)
                assert output[count:].eq(0).all() and guard[[0, -1]].eq(137).all()
            key.view_as(source_k)[start + count:].fill_(float("nan"))
            value.view_as(source_v)[start + count:].fill_(float("nan"))
            graph.replay(); torch.cuda.synchronize()
            assert operation.errors.eq(0).all()
            assert torch.equal(output[:count].cpu().contiguous().view(torch.uint8), expected.view(torch.uint8))
        # A fault in the final virtual history row must reach the real owner.
        query.copy_(source_q[192:192 + capacity])
        key.view_as(source_k).copy_(source_k); value.view_as(source_v).copy_(source_v)
        prompt.zero_(); lengths.fill_(192 + capacity)
        metadata.live_counts[0] = capacity; metadata.lengths.fill_(capacity)
        metadata.positions.copy_(torch.arange(192, 192 + capacity, device="cuda"))
        graph.replay(); torch.cuda.synchronize()
        assert operation.errors.eq(0).all()
        recovered_reference = output.cpu().contiguous()
        metadata.positions[-1] = 8192
        graph.replay(); torch.cuda.synchronize()
        assert operation.history.errors[-1] != 0 and operation.errors[1] != 0
        assert torch.isnan(output[-1]).all() and torch.isfinite(output[:-1]).all()
        metadata.positions[-1] = 192 + capacity - 1
        for _ in range(5):
            graph.replay(); torch.cuda.synchronize()
            assert operation.errors.eq(0).all()
            assert torch.equal(output.cpu().contiguous().view(torch.uint8), recovered_reference.view(torch.uint8))
        graph.reset()


@torch.no_grad()
def test_captured_feature_injection_stops_at_the_original_prompt_boundary():
    from ignition.components.audio_inputs import PrefillAudioInputs
    from ignition.components.feature_state import FeatureState
    from ignition.components.paged_batch import BatchInputs
    from ignition.components.sampling import DeviceRequestState

    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        inputs = BatchInputs.allocate(4, 1, "cuda", capacity=8)
        requests = DeviceRequestState.allocate("cuda", vocab_size=100, capacity=8)
        audio = PrefillAudioInputs.allocate(4, "cuda")
        original = torch.tensor([2], dtype=torch.int32, device="cuda")
        features = FeatureState(inputs, requests, indices=audio.audio_token_indices,
            feature_slot=audio.audio_feature_slot, feature_token=99, slots=1, features=208,
            hidden=4, prompt_lengths=original)
        metadata = Metadata.allocate(4, 1, 1, "cuda")
        embedding = torch.ones((4, 4), dtype=torch.bfloat16, device="cuda")
        valid = torch.zeros((), dtype=torch.bool, device="cuda")
        count = torch.tensor([3], dtype=torch.int32, device="cuda")
        owners.extend((images, inputs, requests, audio, original, features, metadata, embedding, valid, count))
        def operation():
            valid.copy_(features.validate_prefill(inputs.token_ids, count))
            features.inject(metadata, embedding)
        compiled = compile_forward(operation)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph(); graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        inputs.row_slot_ids[0], inputs.row_generations[0] = 7, 12
        requests.generation[7], requests.consumed_tokens[7] = 12, 1
        features.generation[7], features.count[7], features.feature_slot[0] = 12, 1, 0
        features.arena[0, 0].fill_(3)
        inputs.token_ids.copy_(torch.tensor([99, 99, 7, 0], device="cuda"))
        metadata.token_ids.copy_(inputs.token_ids)
        metadata.positions.copy_(torch.tensor([1, 2, 3, -1], device="cuda"))
        audio.audio_token_indices.copy_(torch.tensor([0, -1, -1, -1], device="cuda"))
        for _ in range(5):
            embedding.fill_(1); graph.replay(); torch.cuda.synchronize()
            assert valid
            assert torch.equal(embedding.cpu(), torch.tensor([[3]*4, [1]*4, [1]*4, [1]*4], dtype=torch.bfloat16))
        audio.audio_token_indices[0] = -1
        graph.replay(); torch.cuda.synchronize()
        assert not valid
        requests.consumed_tokens[7] = 2
        features.generation.zero_(); features.count.zero_(); features.feature_slot.fill_(-1)
        metadata.positions.copy_(torch.tensor([2, 3, 4, -1], device="cuda"))
        for _ in range(5):
            embedding.fill_(1); graph.replay(); torch.cuda.synchronize()
            assert valid and embedding.eq(1).all()
        graph.reset()


@torch.no_grad()
def test_prompt_extent_smaller_than_fragment_keeps_prompt_and_history_bytes(tmp_path):
    from ignition.components.reduction_projection import PreparedReductionProjection
    from ignition.recipes.qwen3_asr.numerics import fp32_reductions
    from ignition.recipes.qwen3_asr.replay import (
        PreparedReplayProjection, ReplayScratch, history_projection_count)
    from tests.components.test_reduction_projection_cuda import _session

    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images, fp32_reductions():
        pool = GraphBufferPool("cuda")
        buffers = pool.cursor()
        owners.extend((images, pool))
        rows = torch.arange(4096, device="cuda")
        count, original, prompt, history = [buffers.zeros((1,), torch.int32) for _ in range(4)]
        positions = buffers.zeros((4096,), torch.int64)
        scratch = ReplayScratch.allocate(4096, 128, buffers)
        source = buffers.zeros((4096, 2048), torch.bfloat16)
        guard = buffers.full((4096 * 256 + 16,), 37, torch.bfloat16)
        output = guard[8:-8].view(4096, 256)
        generator = torch.Generator(device="cuda").manual_seed(20261106)
        full = torch.randn((8192, 2048), generator=generator, device="cuda", dtype=torch.bfloat16)
        module = torch.nn.Linear(2048, 256, bias=False, device="cuda", dtype=torch.bfloat16)
        module.weight.normal_(std=.02, generator=generator)
        rs, ro = buffers.zeros((8, 2048), torch.bfloat16), buffers.zeros((8, 256), torch.bfloat16)
        rc = buffers.zeros((1,), torch.int32)
        with _session(tmp_path, owners, graphs) as session:
            operation = PreparedReplayProjection(source, module, output, count, prompt_count=prompt,
                positions=positions, original=original, history_count=history, scratch=scratch,
                session=session, name="wide_fragment", buffers=buffers, workspace=None)
            reference = PreparedReductionProjection(rs, module.weight, ro, rc,
                session=session, name="ordinary_eight_rows")
            owners.extend((operation, reference, scratch, source, guard, full, module, rs, ro, rc))
            session.tune()
            def forward():
                prompt.copy_(torch.minimum(count, (original.long() - positions[:1]).clamp(min=0)).int())
                history.copy_(history_projection_count(rows, count, positions, original))
                operation(source)
            compiled = compile_forward(forward)
            owners.append(compiled)
            compiled()
            graph = torch.cuda.CUDAGraph(); graphs.append(graph)
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            distinguished = False
            for n, start, live in ((19, 0, 0), (19, 0, 19), (19, 0, 130), (123, 64, 130),
                    (123, 256, 4096), (19, 8175, 17), (19, 0, 0)):
                original.fill_(n); count.fill_(live)
                positions.copy_(torch.arange(start, start + 4096, device="cuda"))
                source.fill_(float("nan")); source[:live].copy_(full[start:start + live])
                target = torch.zeros_like(output, device="cpu")
                for begin in range(0, live, 8):
                    active = min(8, live - begin)
                    rs.fill_(float("nan")); rs[:active].copy_(source[begin:begin + active]); rc.fill_(active)
                    reference(rs)
                    target[begin:begin + active].copy_(ro[:active].cpu())
                p = min(live, max(n - start, 0))
                if p:
                    extent = next((c for c in (19, 27, 32, 64) if n <= c), scratch.context)
                    padded = torch.zeros((extent, 2048), device="cuda", dtype=torch.bfloat16)
                    padded[:n].copy_(full[:n])
                    canonical = torch.nn.functional.linear(padded, module.weight)[start:start + p].cpu()
                    distinguished |= not torch.equal(canonical.view(torch.uint8), target[:p].view(torch.uint8))
                    target[:p].copy_(canonical)
                for repeat in range(5):
                    source[live:].fill_(float("inf") if repeat % 2 else float("nan"))
                    scratch.full_input.fill_(float("nan")); scratch.full_output.fill_(float("nan"))
                    scratch.history_output.fill_(float("nan")); guard.fill_(37)
                    torch.cuda.synchronize(); before = torch.cuda.memory_stats()
                    graph.replay(); torch.cuda.synchronize(); after = torch.cuda.memory_stats()
                    for field in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                        assert before[field] == after[field], field
                    assert prompt.item() == p
                    assert history.item() == (live if start + live > n and live else 0)
                    assert torch.equal(output.cpu().view(torch.uint8), target.view(torch.uint8)), (n, start, live)
                    assert guard[:8].eq(37).all() and guard[-8:].eq(37).all()
                if live:
                    for row in sorted({0, live - 1}):
                        source[row, 0] = float("nan")
                        graph.replay(); torch.cuda.synchronize()
                        assert not torch.isfinite(output[row]).all()
                        source[row].copy_(full[start + row])
                        graph.replay(); torch.cuda.synchronize()
                        assert torch.equal(output.cpu().view(torch.uint8), target.view(torch.uint8))
            assert distinguished, "control operands must distinguish the two arithmetic paths"
            for invalid in (-1, 4097):
                count.fill_(invalid); graph.replay(); torch.cuda.synchronize()
                assert history.item() == invalid and operation.history_output.isnan().all()
            count.zero_(); graph.replay(); torch.cuda.synchronize()
            assert output.eq(0).all()
            original.fill_(19); count.fill_(19)
            positions.copy_(torch.arange(4096, device="cuda")); source[:19].copy_(full[:19])
            graph.replay(); torch.cuda.synchronize()
            recovered = torch.nn.functional.linear(full[:19], module.weight).cpu()
            assert torch.equal(output[:19].cpu().view(torch.uint8), recovered.view(torch.uint8))
            assert output[19:].eq(0).all() and guard[:8].eq(37).all() and guard[-8:].eq(37).all()
            graph.reset()
