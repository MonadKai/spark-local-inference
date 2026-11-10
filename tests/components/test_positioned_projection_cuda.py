# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Full physical projection geometry survives partial and poisoned fragments."""

import os

import pytest
import torch

from ignition.components.positioned_projection import PreparedPositionedProjection
from ignition.preparation.compilation import compile_forward
from ignition.preparation.failure import retain_on_failed_drain
from ignition.preparation.module_images import ModuleImages

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


@pytest.mark.parametrize("capacity,length,extents,fragment_rows", [
    (64, 19, (19, 27, 32), 19), (256, 165, (19, 27, 32, 64), 165),
    (8192, 165, (19, 27, 32, 64), 165),
    (128, 19, (19, 27, 32, 64), 4096),
    (256, 123, (19, 27, 32, 64), 4096),
    (512, 443, (19, 27, 32, 64), 4096),
])
@torch.no_grad()
def test_full_and_fragmented_projection_keep_bytes_and_recover(capacity, length, extents, fragment_rows):
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        owners.append(images)
        generator = torch.Generator(device="cuda").manual_seed(20261106)
        source = torch.randn((length, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
        weight = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
        owners.extend((source, weight))
        expected_input = torch.zeros((capacity, 128), device="cuda", dtype=torch.bfloat16)
        expected_input[:length].copy_(source)
        selected_extent = next((extent for extent in extents if length <= extent), capacity)
        expected = torch.nn.functional.linear(expected_input[:selected_extent], weight)[:length].cpu()
        owners.append(expected_input)
        observed = []
        for begin, count, physical in [(0, length, fragment_rows), (0, 7, 8), (7, length - 7, length)]:
            inputs = torch.full((physical, 128), float("nan"), device="cuda", dtype=torch.bfloat16)
            inputs[:count].copy_(source[begin:begin + count])
            guards = [torch.full((rows + 2, columns), 123., device="cuda", dtype=torch.bfloat16)
                for rows, columns in ((physical, 256), (capacity, 128), (capacity, 256))]
            output, staging, projected = [value[1:-1] for value in guards]
            positions = torch.arange(begin, begin + physical, device="cuda", dtype=torch.int64)
            live = torch.tensor([count], device="cuda", dtype=torch.int32)
            total = torch.tensor([length], device="cuda", dtype=torch.int32)
            operation = PreparedPositionedProjection(inputs, weight, output, live, positions, total,
                staging, projected, subrow_extents=extents)
            owners.extend((inputs, *guards, positions, live, total, operation))
            compiled = compile_forward(lambda op=operation, x=inputs: op(x))
            owners.append(compiled)
            compiled()
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with torch.cuda.graph(graph):
                compiled()
            baseline = None
            for _ in range(5):
                output.fill_(float("nan"))
                staging.fill_(float("nan"))
                projected.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for field in ("allocation.all.allocated", "allocation.all.freed",
                              "segment.all.allocated", "segment.all.freed"):
                    assert before[field] == after[field]
                actual = output[:count].cpu().contiguous()
                assert torch.isfinite(actual).all()
                assert torch.equal(actual.view(torch.uint8), expected[begin:begin + count].view(torch.uint8))
                assert output[count:].eq(0).all()
                for guard in guards:
                    assert guard[0].eq(123.).all() and guard[-1].eq(123.).all()
                if baseline is not None:
                    assert torch.equal(actual.view(torch.uint8), baseline.view(torch.uint8))
                baseline = actual
            observed.append(baseline)
            inputs[0, 0] = float("nan")
            graph.replay()
            torch.cuda.synchronize()
            assert not torch.isfinite(output).all()
            inputs[0].copy_(source[begin])
            # Metadata defects must also reach the completion finite gate.
            for invalid in ("count", "position", "sequence", "position_int64_max", "position_int32_wrap"):
                if invalid == "count":
                    live[0] = physical + 1
                elif invalid == "position":
                    positions[0] = capacity
                elif invalid == "sequence":
                    total[0] = capacity + 1
                else:
                    live[0] = 1
                    positions[0] = 2**63 - 1 if invalid == "position_int64_max" else 2**32
                graph.replay()
                torch.cuda.synchronize()
                assert not torch.isfinite(output).all(), invalid
                live[0] = count
                positions.copy_(torch.arange(begin, begin + physical, device="cuda"))
                total[0] = length
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output[:count].cpu().view(torch.uint8), baseline.view(torch.uint8))
            live.zero_()
            total.zero_()
            inputs.fill_(float("nan"))
            positions.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            assert output.eq(0).all()
            graph.reset()
        assert torch.equal(torch.cat(observed[1:]).view(torch.uint8), observed[0].view(torch.uint8))


@torch.no_grad()
def test_positioned_projection_rejects_aliases():
    owners = []
    with retain_on_failed_drain(0, owners):
        source = torch.zeros((8, 128), device="cuda", dtype=torch.bfloat16)
        weight = torch.zeros((128, 128), device="cuda", dtype=torch.bfloat16)
        output = torch.zeros_like(source)
        staging = torch.zeros((64, 128), device="cuda", dtype=torch.bfloat16)
        projected = torch.zeros_like(staging)
        positions = torch.arange(8, device="cuda", dtype=torch.int64)
        count = torch.tensor([8], device="cuda", dtype=torch.int32)
        total = torch.tensor([19], device="cuda", dtype=torch.int32)
        owners.extend((source, weight, output, staging, projected, positions, count, total))
        for target, value in (("output", source), ("staging", projected), ("projected", staging)):
            buffers = dict(output=output, staging=staging, projected=projected)
            buffers[target] = value
            with pytest.raises(ValueError, match="overlap"):
                PreparedPositionedProjection(source, weight, buffers["output"], count, positions, total,
                    buffers["staging"], buffers["projected"])


@pytest.mark.parametrize("capacity,count,start", [(256, 165, 0), (128, 128, 0), (64, 37, 128), (8, 0, 0)])
def test_positioned_projection_package_replays_without_python(tmp_path, capacity, count, start):
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    subprocess.run(["cargo", "build", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools", "--bin", "test_fixed_runtime_package"],
        cwd=root, check=True, timeout=300)
    package = tmp_path / "positioned-projection"
    preparation = subprocess.run([sys.executable, "-m", "helios.tests.canonical_projection_package_fixture",
        str(package), "--capacity", str(capacity), "--count", str(count), "--positioned", "--start", str(start)],
        cwd=root, text=True, capture_output=True, timeout=900)
    assert preparation.returncode == 0, preparation.stdout + preparation.stderr
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(root / "helios/target")))
    replay = subprocess.run([str(target / "debug/test_fixed_runtime_package"), str(package), "--replays", "5"],
        cwd=root, text=True, capture_output=True, timeout=180)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert "native fixed-address replay passed" in replay.stdout
