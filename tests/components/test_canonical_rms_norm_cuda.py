# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Actual replay operand, complete physical row catalog, capture and row isolation."""

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from ignition.components.canonical_rms_norm import CanonicalRMSNorm
from ignition.preparation.compilation import compile_forward
from ignition.preparation.failure import retain_on_failed_drain
from ignition.preparation.module_images import ModuleImages

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


class Forward:
    def __init__(self, operation, source, output):
        self.operation, self.source, self.output = operation, source, output

    def __call__(self):
        self.output.copy_(self.operation(self.source))
        return self.output


def same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@torch.no_grad()
def test_original_canonical_mean_exposes_the_real_small_batch_replay_defect():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign the validation GPU explicitly"
    fixture = load_file(Path(__file__).resolve().parents[1] / "fixtures/asr-rmsnorm-replay.safetensors")
    owners = []
    with retain_on_failed_drain(0, owners):
        canonical = Qwen3RMSNorm(2048, eps=1e-6)
        owners.append(canonical)
        canonical.to(device="cuda:0", dtype=torch.bfloat16)
        canonical.weight.copy_(fixture["weight"])
        row = fixture["hidden"].cuda()
        owners.append(row)
        small_input = row.repeat(8, 1)
        owners.append(small_input)
        replay_input = row.repeat(64, 1)
        owners.append(replay_input)
        small = canonical(small_input)
        owners.append(small)
        replay = canonical(replay_input)
        owners.append(replay)
        # The same original implementation previously changed twenty BF16 affine
        # values, then L25 KV, across decode and readmitted history prefill.
        assert not same(small[0], replay[31]), "negative control no longer reproduces the measured mean-layout defect"


@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_canonical_norm_captured_rows_positions_poison_guards_and_allocations(compiled):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign the validation GPU explicitly"
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        owners.append(images)
        fixture = load_file(Path(__file__).resolve().parents[1] / "fixtures/asr-rmsnorm-replay.safetensors")
        gain = fixture["weight"].cuda()
        owners.append(gain)
        weight = torch.nn.Parameter(gain, requires_grad=False)
        owners.append(weight)
        row = fixture["hidden"].cuda()
        owners.append(row)
        canonical = Qwen3RMSNorm(2048, eps=1e-6)
        owners.append(canonical)
        canonical.to(device="cuda:0", dtype=torch.bfloat16)
        canonical.weight = weight
        reference_input = row.repeat(64, 1)
        owners.append(reference_input)
        reference_output = canonical(reference_input)
        owners.append(reference_output)
        expected = reference_output[0].clone()
        owners.append(expected)
        operation = CanonicalRMSNorm(weight, 1e-6, minimum_rows=64)
        owners.append(operation)
        # Physical decode is eight rows; final-head selection may be one row.
        # Every current prefill extent is included, along with small boundaries.
        for rows in (1, 2, 4, 8, 31, 63, 64, 128, 256, 512, 1024, 1536,
                     2048, 2560, 3072, 3584, 4096):
            source = torch.empty((rows, 2048), dtype=torch.bfloat16, device="cuda:0")
            owners.append(source)
            guard = torch.empty((rows + 2, 2048), dtype=torch.bfloat16, device="cuda:0")
            owners.append(guard)
            guard.fill_(137.)
            output = guard[1:-1]
            forward = Forward(operation, source, output)
            owners.extend((output, forward))
            source.copy_(row.expand_as(source))
            if compiled:
                forward = compile_forward(forward)
                owners.append(forward)
            forward()
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with torch.cuda.graph(graph):
                forward()
            for active in sorted({0, rows // 2, rows - 1}):
                for poison in (float("nan"), float("inf")):
                    source.fill_(poison)
                    source[active].copy_(row)
                    for _ in range(3):
                        output.fill_(float("inf"))
                        torch.cuda.synchronize()
                        before = torch.cuda.memory_stats()
                        graph.replay()
                        torch.cuda.synchronize()
                        after = torch.cuda.memory_stats()
                        assert same(output[active], expected), (compiled, rows, active, poison)
                        assert torch.isfinite(output[active]).all()
                        assert torch.isnan(output[:active]).all() and torch.isnan(output[active + 1:]).all()
                        assert guard[0].eq(137).all() and guard[-1].eq(137).all()
                        for key in ("allocation.all.allocated", "allocation.all.freed",
                                    "segment.all.allocated", "segment.all.freed"):
                            assert before[key] == after[key], (compiled, rows, key)
            # No output is allowed to alias and overwrite the input matrix.
            assert same(source[active], row)
            graph.reset()
