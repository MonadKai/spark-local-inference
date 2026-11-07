# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""ASR head extent is graph state, even when graph scratch is shared."""

import os
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.compilation import compile_forward
from ignition.preparation.failure import retain_on_failed_drain
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.scratch import ScratchArena
from tests.components.test_canonical_projection_cuda import Forward, prepared_session

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


@torch.no_grad()
def test_asr_vocabulary_republishes_extent_after_shared_scratch_is_overwritten():
    from ignition.preparation.scratch import GraphBufferPool
    from ignition.recipes.qwen3_asr.decoder import _Vocabulary

    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs):
        with ModuleImages() as images:
            device = torch.device("cuda:0")
            pool = GraphBufferPool(device, chunk_bytes=4096)
            arena = ScratchArena(device, overlap=False)
            arena.reserve(8 * 128 * 4)
            weight = torch.ones((128, 64), device=device, dtype=torch.bfloat16)
            owners.extend((images, pool, arena, weight))
            variants = {}
            with prepared_session(device, owners, graphs) as session:
                for rows in (8, 4, 2, 1):
                    buffers = pool.cursor()
                    source = buffers.zeros((rows, 64), torch.bfloat16)
                    output = buffers.zeros((rows, 128), torch.bfloat16)
                    count = buffers.full((1,), rows, torch.int32)
                    head = _Vocabulary(source, weight, output, count, session=session,
                        name=f"head{rows}", workspace=arena, buffers=buffers)
                    forward = compile_forward(Forward(head, source))
                    owners.extend((source, output, count, head, forward))
                    forward()
                    graph = torch.cuda.CUDAGraph()
                    graphs.append(graph)
                    with torch.cuda.graph(graph):
                        forward()
                    variants[rows] = (source, output, count, graph)
                # Each cursor reuses the same int32 chunk. A constructor fill
                # is lost when another variant or state initialization writes it.
                assert len({variant[2].data_ptr() for variant in variants.values()}) == 1
                for rows in (8, 1, 4, 2, 8, 1):
                    source, output, count, graph = variants[rows]
                    source.fill_(.25)
                    output.fill_(float("inf"))
                    count.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert count.item() == rows
                    assert output.eq(16).all()
                for graph in graphs:
                    graph.reset()


@torch.no_grad()
def test_real_vocabulary_resume_matches_normal_across_graphs_and_row_reuse():
    """Use independent normalized hidden rows and the original full head.

    This checks projection, resume gathering and completion, not the preceding
    decoder or asynchronous native scheduler. Explicit fixture locations make
    missing requested inputs fail rather than silently substituting tensors.
    """
    from safetensors import safe_open

    from ignition.components.paged_resume import PreparedResume
    from ignition.components.sampling import DeviceRequestState, RequestAdmission, SamplingSelection
    from ignition.preparation.scratch import GraphBufferPool
    from ignition.recipes.qwen3_asr.decoder import _Vocabulary
    from tests.components.completion_support import completion_outputs

    checkpoint = os.environ.get("FLARE_QWEN3_ASR_SNAPSHOT")
    observations = os.environ.get("FLARE_QWEN3_ASR_CANONICAL_OBSERVATIONS")
    if checkpoint is None or observations is None:
        pytest.skip("requires the original checkpoint and independent canonical observations")
    checkpoint, observations = Path(checkpoint), Path(observations)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    head_name = "thinker.lm_head.weight"
    with safe_open(checkpoint / index["weight_map"][head_name], framework="pt") as file:
        cpu_weight = file.get_tensor(head_name)
    assert cpu_weight.shape == (151936, 2048) and cpu_weight.dtype == torch.bfloat16
    values = []
    for identity in ("en-777-126732-0081", "en-422-122949-0013",
                     "zh-BAC009S0727W0418", "zh-BAC009S0729W0241"):
        with safe_open(observations / (identity + "-prefill.safetensors"), framework="pt") as file:
            hidden = file.get_tensor("thinker.model.norm.output").reshape(-1, 2048)
        values.extend((hidden[-1], hidden[hidden.shape[0] // 2]))
    hidden_rows = torch.stack(values)
    assert hidden_rows.dtype == torch.bfloat16 and torch.isfinite(hidden_rows).all()
    assert torch.unique(hidden_rows, dim=0).shape[0] == 8, "row-routing control needs distinct rows"
    owners, graphs = [], []
    previous = (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, previous[1])
        with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
            device = torch.device("cuda:0")
            weight, hidden_rows = cpu_weight.to(device), hidden_rows.to(device)
            saved = torch.zeros((10, 2048), device=device, dtype=torch.bfloat16)
            saved[1:9].copy_(hidden_rows)
            pool = SimpleNamespace(backing=saved.view(torch.uint8), last_hidden=saved)
            weights = SimpleNamespace(head=weight, shard=SimpleNamespace(world_size=1))
            requests = DeviceRequestState.allocate(device, vocab_size=151936, capacity=8)
            buffers = GraphBufferPool(device)
            records = torch.zeros((2, 8, 32), device=device, dtype=torch.int64)
            frame = torch.zeros(1, device=device, dtype=torch.int32)
            sequence = torch.ones(1, device=device, dtype=torch.int64)
            state_rows = torch.arange(1, 9, device=device)[:, None].expand(-1, 8).contiguous()
            owners.extend((images, weight, hidden_rows, saved, pool, requests, buffers,
                           records, frame, sequence, state_rows))
            expected = torch.nn.functional.linear(hidden_rows, weight)
            assert not torch.equal(expected[0], expected[1]), "wrong-row control must be observable"
            variants = {}
            with prepared_session(device, owners, graphs) as session:
                for rows in (8, 4, 2, 1):
                    inputs = SimpleNamespace(
                        row_slot_ids=torch.arange(rows, device=device, dtype=torch.int32),
                        row_generations=torch.full((rows,), 7, device=device, dtype=torch.int64),
                        live_rows=torch.zeros(1, device=device, dtype=torch.int32),
                        request_state_rows=state_rows)
                    resume = PreparedResume(weights, requests, pool, inputs,
                        completion_outputs(records), frame, sequence, rows=rows,
                        runtime=None, vocabulary_runtime=None, coordinator=None,
                        session=session, name=f"resume{rows}", buffers=buffers.cursor(),
                        sampling_selection=SamplingSelection.FIRST_MAXIMUM, projection_row_capacity=8,
                        vocabulary_projection=_Vocabulary.prepare_resume)
                    source = torch.zeros((rows, 2048), device=device, dtype=torch.bfloat16)
                    output = torch.empty((rows, 151936), device=device, dtype=torch.bfloat16)
                    count = torch.zeros(1, device=device, dtype=torch.int32)
                    normal = _Vocabulary(source, weight, output, count, session=session,
                        name=f"normal{rows}", workspace=None, buffers=buffers.cursor())
                    calls = (compile_forward(Forward(normal, source)), compile_forward(resume))
                    owners.extend((inputs, resume, source, output, count, normal, *calls))
                    captured = []
                    for call in calls:
                        call()
                        graph = torch.cuda.CUDAGraph()
                        graphs.append(graph)
                        with torch.cuda.graph(graph):
                            call()
                        captured.append(graph)
                    variants[rows] = inputs, resume, source, output, count, captured
                for rows in (8, 1, 4, 2, 8, 1):
                    inputs, resume, source, output, count, (normal_graph, resume_graph) = variants[rows]
                    for reverse in (False, True):
                        slots = list(range(8))[::-1] if reverse else list(range(8))
                        for live in sorted({0, 1, rows // 2, rows}):
                            for hole in (False, True):
                                selected = slots[:rows]
                                if hole:
                                    selected[rows // 2] = -1
                                indices = torch.tensor(selected, device=device)
                                active = (torch.arange(rows, device=device) < live) & (indices >= 0)
                                routed = hidden_rows[indices.clamp_min(0)].clone()
                                routed.masked_fill_(~active[:, None], 0)
                                target = expected[indices.clamp_min(0)].clone()
                                target.masked_fill_(~active[:, None], 0)
                                for _ in range(2):
                                    for slot in range(8):
                                        requests.admit(slot, RequestAdmission(7, 0, max_new_tokens=100),
                                            [1, 2, 3], consumed_tokens=3)
                                    inputs.row_slot_ids.copy_(indices)
                                    inputs.live_rows.fill_(live)
                                    source.copy_(routed)
                                    output.fill_(float("nan"))
                                    resume.logits.fill_(float("nan"))
                                    resume.local_logits.fill_(float("nan"))
                                    count.zero_()
                                    resume.projection.count.zero_()
                                    torch.cuda.synchronize()
                                    before = torch.cuda.memory_stats()
                                    normal_graph.replay()
                                    resume_graph.replay()
                                    torch.cuda.synchronize()
                                    after = torch.cuda.memory_stats()
                                    for key in ("allocation.all.allocated", "allocation.all.freed",
                                                "segment.all.allocated", "segment.all.freed"):
                                        assert before[key] == after[key], key
                                    assert torch.equal(output.view(torch.uint8), target.view(torch.uint8))
                                    assert torch.equal(resume.logits.view(torch.uint8), output.view(torch.uint8))
                                    assert torch.equal(resume.local_logits.view(torch.uint8), output.view(torch.uint8))
                                    assert torch.equal(resume.state.hidden, routed)
                                    assert requests.consumed_tokens.eq(3).all()
                                    assert requests.error_code.eq(0).all()
                                    for row, slot in enumerate(selected):
                                        if active[row]:
                                            assert requests.pending_token[slot] == target[row].argmax()
                for graph in graphs:
                    graph.reset()
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
