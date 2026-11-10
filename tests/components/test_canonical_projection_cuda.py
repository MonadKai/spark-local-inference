# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured canonical projections preserves BF16 nodes, row identity and scratch bounds."""

from contextlib import contextmanager
import os

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.canonical_projection import PreparedCanonicalProjection, PreparedRowwiseProjection
from ignition.preparation.compilation import compile_forward
from ignition.preparation.failure import retain_on_failed_drain, retain_undrained
from ignition.preparation.module_images import ModuleImages

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an assigned CUDA device")


class Forward:
    def __init__(self, operation, source):
        self.operation, self.source = operation, source

    def __call__(self):
        return self.operation(self.source)


@contextmanager
def prepared_session(device, owners, graphs, *, autotune=False):
    session = PreparationSession(device=device, autotune=autotune, compile_workers=1, prime=False)
    owners.append(session)
    try:
        yield session
    finally:
        try:
            torch.cuda.synchronize(device)
        except BaseException as error:
            retain_undrained(error, owners, graphs)
            raise
        session.close()


@pytest.fixture(autouse=True)
def production_reduction_policy():
    from ignition.recipes.qwen3_asr.numerics import fp32_reductions

    with fp32_reductions():
        yield


@torch.no_grad()
@pytest.mark.parametrize("projection_type", [PreparedCanonicalProjection, PreparedRowwiseProjection])
@pytest.mark.parametrize("width,columns", [(2048, 4096), (2048, 2048),
    (2048, 12288), (6144, 2048), (2048, 151936), (7680, 1024), (1024, 4096)])
def test_captured_canonical_bias_rounding_rows_guards_and_poison(width, columns, projection_type):
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        device = torch.device("cuda:0")
        weight = torch.zeros((columns, width), device=device, dtype=torch.bfloat16)
        weight[:, :2] = 1
        bias = torch.full((columns,), -1., device=device, dtype=torch.bfloat16)
        original = torch.zeros((8, width), device=device, dtype=torch.bfloat16)
        original[:, 0] = 1
        original[:, 1] = torch.arange(1, 9, device=device) / 256
        expected = original[:, 1:2].expand(8, columns).clone()
        premature = torch.nn.functional.linear(original, weight).bfloat16() + bias
        assert not torch.equal(premature[0], expected[0]), "control lost the BF16 bias boundary"
        staging_guard = torch.full((8 * width + 2,), 37., device=device, dtype=torch.bfloat16)
        staging = staging_guard[1:-1].view(8, width)
        owners.extend((images, weight, bias, original, expected, staging_guard))
        variants = {}
        with prepared_session(device, owners, graphs, autotune=True) as session:
            for rows in (1, 2, 4, 8):
                source = original[:rows].clone()
                guard = torch.full((rows + 2, columns), 137., device=device, dtype=torch.bfloat16)
                output = guard[1:-1]
                count = torch.tensor([rows], device=device, dtype=torch.int32)
                operation = projection_type(source, weight, output, count,
                    staging=staging,
                    bias=bias)
                forward = compile_forward(Forward(operation, source))
                owners.extend((source, guard, count, operation, forward))
                forward()
                graph = torch.cuda.CUDAGraph()
                graphs.append(graph)
                with torch.cuda.graph(graph):
                    forward()
                variants[rows] = source, output, count, guard, graph
            for rows in (8, 1, 4, 2, 8, 1):
                source, output, count, guard, graph = variants[rows]
                for length in sorted({0, 1, rows // 2, rows}):
                    count.fill_(length)
                    source.fill_(float("nan"))
                    source[:length].copy_(original[:length])
                    target = torch.zeros_like(output)
                    target[:length].copy_(expected[:length])
                    for _ in range(3):
                        staging.fill_(float("nan"))
                        output.fill_(float("inf"))
                        torch.cuda.synchronize()
                        before = torch.cuda.memory_stats()
                        graph.replay()
                        torch.cuda.synchronize()
                        after = torch.cuda.memory_stats()
                        for key in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                            assert before[key] == after[key], key
                        assert torch.equal(output.view(torch.uint8), target.view(torch.uint8))
                        assert staging.view(8, width)[rows:].eq(0).all()
                        assert guard[[0, -1]].eq(137).all()
                        assert staging_guard[[0, -1]].eq(37).all()
                    if length:
                        source[0, 0] = float("nan")
                        graph.replay()
                        torch.cuda.synchronize()
                        assert not torch.isfinite(output[0]).all()
                        source[:length].copy_(original[:length])
                        graph.replay()
                        torch.cuda.synchronize()
                        assert torch.equal(output.view(torch.uint8), target.view(torch.uint8))
            for graph in graphs:
                graph.reset()


@torch.no_grad()
def test_canonical_rejects_binding_and_replay_input_aliases():
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        device = torch.device("cuda:0")
        source = torch.ones((2, 64), device=device, dtype=torch.bfloat16)
        weight = torch.ones((64, 64), device=device, dtype=torch.bfloat16)
        output = torch.zeros_like(source)
        shared = torch.empty((4, 64), device=device, dtype=torch.bfloat16)
        alias_source = shared[:2]
        owners.extend((images, source, weight, output, shared, alias_source))
        with prepared_session(device, owners, graphs) as session:
            with pytest.raises(ValueError, match="must not overlap"):
                PreparedCanonicalProjection(alias_source, weight, output, staging=shared)
            operation = PreparedCanonicalProjection(source, weight, output, staging=shared)
            owners.append(operation)
            with pytest.raises(ValueError, match="must not overlap"):
                operation(alias_source)


@torch.no_grad()
def test_captured_subrow_boundaries_rounding_poison_guards_and_allocations():
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        device = torch.device("cuda:0")
        rows, width, columns = 64, 2048, 4096
        original = torch.zeros((rows, width), device=device, dtype=torch.bfloat16)
        original[:, 0] = 1
        original[:, 1] = torch.arange(1, rows + 1, device=device) / 256
        source = original.clone()
        weight = torch.zeros((columns, width), device=device, dtype=torch.bfloat16)
        weight[:, :2] = 1
        bias = torch.full((columns,), -1., device=device, dtype=torch.bfloat16)
        guard = torch.full((rows + 2, columns), 37., device=device, dtype=torch.bfloat16)
        output = guard[1:-1]
        count = torch.zeros((1,), device=device, dtype=torch.int32)
        expected = torch.zeros_like(output)
        owners.extend((images, original, source, weight, bias, guard, count, expected))
        with prepared_session(device, owners, graphs) as session:
            for extents in ((0,), (64,), (27, 19), (19, 19), [19], (True,)):
                with pytest.raises(ValueError, match="subrow"):
                    PreparedCanonicalProjection(source, weight, output, count, subrow_extents=extents)
            with pytest.raises(ValueError, match="subrow"):
                PreparedCanonicalProjection(source, weight, output, subrow_extents=(19,))
            operation = PreparedCanonicalProjection(source, weight, output, count,
                bias=bias, subrow_extents=(19, 27, 32))
            forward = compile_forward(Forward(operation, source))
            owners.extend((operation, forward))
            forward()
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with torch.cuda.graph(graph):
                forward()
            for length in (0, 1, 18, 19, 20, 26, 27, 28, 31, 32, 33, 63, 64, 19, 0):
                count.fill_(length)
                source.fill_(float("nan"))
                source[:length].copy_(original[:length])
                expected.zero_()
                expected[:length].copy_(original[:length, 1:2].expand(length, columns))
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
                    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
                    assert guard[[0, -1]].eq(37).all()
                if length:
                    source[0, 0] = float("nan")
                    graph.replay()
                    torch.cuda.synchronize()
                    assert not torch.isfinite(output[0]).all()
                    source[:length].copy_(original[:length])
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
            graph.reset()


@pytest.mark.parametrize("capacity,count,with_bias,rowwise", [(8, 0, False, False), (64, 19, True, False),
    (64, 32, True, False), (64, 64, True, False), (8, 2, False, True)])
def test_canonical_projection_package_replays_without_python(tmp_path, capacity, count, with_bias, rowwise):
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    subprocess.run(["cargo", "build", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools", "--bin", "test_fixed_runtime_package"],
        cwd=root, check=True, timeout=300)
    package = tmp_path / "canonical-projection"
    args = [sys.executable, "-m", "helios.tests.canonical_projection_package_fixture", str(package),
        "--capacity", str(capacity), "--count", str(count)]
    if with_bias:
        args.append("--with-bias")
    if rowwise:
        args.append("--rowwise")
    preparation = subprocess.run(args, cwd=root, text=True, capture_output=True, timeout=900)
    assert preparation.returncode == 0, preparation.stdout + preparation.stderr
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(root / "helios/target")))
    replay = subprocess.run([str(target / "debug/test_fixed_runtime_package"), str(package), "--replays", "5"],
        cwd=root, text=True, capture_output=True, timeout=180)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert "native fixed-address replay passed" in replay.stdout
