# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Real checkpoint short convolutions preserve nodes and longer-width results."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ignition.preparation.compilation import compile_forward
from ignition.preparation.failure import retain_on_failed_drain
from ignition.preparation.module_images import ModuleImages
from ignition.recipes.qwen3_asr.audio import _ConvolutionChunks, convolution_chunks
from ignition.recipes.qwen3_asr.numerics import fp32_reductions

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an assigned GPU and FLARE_RUN_CUDA_TESTS=1")


@torch.no_grad()
def test_original_weights_short_widths_and_full_path_transitions():
    configured = os.environ.get("FLARE_QWEN3_ASR_SNAPSHOT")
    if not configured:
        pytest.skip("requires the original-format ASR checkpoint")
    root = Path(configured)
    index = json.loads((root / "model.safetensors.index.json").read_bytes())["weight_map"]
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images, fp32_reductions():
        owners.append(images)
        convolutions = []
        owners.append(convolutions)
        for number, channels in enumerate((1, 480, 480), 1):
            convolution = torch.nn.Conv2d(channels, 480, 3, stride=2, padding=1, dtype=torch.bfloat16).eval()
            convolutions.append(convolution)
            convolution.cuda()
            for field in ("weight", "bias"):
                name = f"thinker.audio_tower.conv2d{number}.{field}"
                with safe_open(root / index[name], framework="pt") as shard:
                    value = shard.get_tensor(name)
                    assert value.dtype == torch.bfloat16 and value.shape == getattr(convolution, field).shape
                    getattr(convolution, field).copy_(value)
        source = torch.zeros((1, 1, 128, 100), device="cuda", dtype=torch.float32)
        width = torch.zeros((1,), device="cuda", dtype=torch.int32)
        positions = torch.arange(100, device="cuda")
        output = torch.zeros((1, 480, 16, 13), device="cuda", dtype=torch.bfloat16)
        operation = _ConvolutionChunks(tuple(convolutions), 100, source.device)
        owners.extend((source, width, positions, output, operation))
        def forward():
            masked = source.masked_fill((positions >= width)[None, None, None, :], 0).bfloat16()
            output.copy_(operation(masked, width, width))
        compiled = compile_forward(forward)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        generator = torch.Generator(device="cuda").manual_seed(20261106)
        for count in [0, *range(1, 42), 50, 99, 100, 13, 0]:
            source.fill_(float("nan"))
            source[..., :count].normal_(std=.5, generator=generator)
            width.fill_(count)
            if 0 < count <= 40:
                # Independent original physically cropped convolution stack.
                reference = source[..., :count].bfloat16().contiguous()
                for convolution in convolutions:
                    reference = F.gelu(convolution(reference))
            elif count:
                # The pre-existing full-width path retains bytewise behavior.
                masked = source.masked_fill((positions >= width)[None, None, None, :], 0).bfloat16()
                reference = convolution_chunks(masked, width, convolutions, operation.positions)
            previous = None
            for _ in range(3):
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[name] == after[name], name
                actual = output.cpu().clone()
                assert torch.isfinite(actual).all()
                if count:
                    expected = reference.cpu().contiguous()
                    assert torch.equal(actual[..., :expected.shape[-1]].contiguous().view(torch.uint8), expected.view(torch.uint8)), count
                    assert actual[..., (count + 7) // 8:].eq(0).all()
                else:
                    assert actual.eq(0).all()
                if previous is not None:
                    assert torch.equal(actual.view(torch.uint8), previous.view(torch.uint8))
                previous = actual
        graph.reset()


@pytest.mark.parametrize("capacity", [100, 800])
@torch.no_grad()
def test_whole_input_batch_classes_and_short_tail_windows(capacity):
    configured = os.environ.get("FLARE_QWEN3_ASR_SNAPSHOT")
    if not configured:
        pytest.skip("requires the original-format ASR checkpoint")
    root = Path(configured)
    index = json.loads((root / "model.safetensors.index.json").read_bytes())["weight_map"]
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images, fp32_reductions():
        owners.append(images)
        convolutions = []
        owners.append(convolutions)
        for number, channels in enumerate((1, 480, 480), 1):
            convolution = torch.nn.Conv2d(channels, 480, 3, stride=2, padding=1,
                device="cuda", dtype=torch.bfloat16).eval()
            convolutions.append(convolution)
            for field in ("weight", "bias"):
                name = f"thinker.audio_tower.conv2d{number}.{field}"
                with safe_open(root / index[name], framework="pt") as shard:
                    getattr(convolution, field).copy_(shard.get_tensor(name))
        chunks = capacity // 100
        source = torch.zeros((chunks, 1, 128, 100), device="cuda", dtype=torch.bfloat16)
        guard = torch.full((chunks + 2, 480, 16, 13), 37., device="cuda", dtype=torch.bfloat16)
        output = guard[1:-1]
        width = torch.tensor([100], device="cuda", dtype=torch.int32)
        total = torch.zeros((1,), device="cuda", dtype=torch.int32)
        operation = _ConvolutionChunks(tuple(convolutions), capacity, source.device)
        owners.extend((source, guard, output, width, total, operation))
        def forward():
            output.copy_(operation(source, width, total))
        compiled = compile_forward(forward)
        owners.append(compiled)
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        generator = torch.Generator(device="cuda").manual_seed(20261106)
        for frames in (100, 200, 300, 400, 500, 1200, 1300, 1600, 3300, 200, 100,
                2, 13, 32, 33, 40, 41, 99, 100, 101, 199, 200, 201, 300, 400,
                500, 799, 800, 801, 1199, 1200, 1201, 1299, 1300, 1301, 1600,
                3265, 3300, 200, 2):
            logical = (frames + 99) // 100
            common_width = min(frames, 100)
            original = torch.randn((logical, 1, 128, common_width), device="cuda", dtype=torch.bfloat16,
                generator=generator) * .5
            if frames % 100:
                original[-1, ..., frames % 100:] = 0
            reference = original
            for convolution in convolutions:
                reference = F.gelu(convolution(reference))
            reference = F.pad(reference, (0, 13 - reference.shape[-1]))
            total.fill_(frames)
            width.fill_(common_width)
            for start in range(0, logical, 8):
                count = min(8, logical - start)
                if count > chunks:
                    continue
                source.fill_(float("nan"))
                source[:count].zero_()
                source[:count, ..., :common_width].copy_(original[start:start + count])
                frozen_source = source.clone()
                expected = reference[start:start + count].cpu().contiguous()
                for _ in range(3):
                    output.fill_(float("inf"))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                        assert before[key] == after[key], key
                    actual = output[:count].cpu().contiguous()
                    assert torch.isfinite(actual).all()
                    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (capacity, frames, start)
                    assert guard[[0, -1]].eq(37).all()
                    assert torch.equal(source.view(torch.uint8), frozen_source.view(torch.uint8))
                if capacity == 800 and frames in (200, 1300) and start == 0:
                    wrong = convolution_chunks(source, width, convolutions, operation.positions)[:count]
                    assert not torch.equal(wrong.cpu().contiguous().view(torch.uint8), expected.view(torch.uint8))
                source[0, 0, 0, 0] = float("nan")
                graph.replay()
                torch.cuda.synchronize()
                assert not torch.isfinite(output[:count]).all()
                source.copy_(frozen_source)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(output[:count].cpu().contiguous().view(torch.uint8), expected.view(torch.uint8))
        graph.reset()
