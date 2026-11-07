# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Actual tuning, capture, row isolation and native replay of fixed reductions."""

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ignition.components.reduction_projection import PreparedReductionProjection
from ignition.preparation.compilation import compile_forward
from ignition.preparation.failure import resources_retained, retain_on_failed_drain, retain_undrained
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.tuning import RecordedPreparationSession
from qualify.asr_linear_bounds import envelope, result_ratio
from tests.components.test_canonical_projection_cuda import Forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an assigned CUDA device")


def _guards_intact(guard):
    assert guard[:8].eq(37).all() and guard[-8:].eq(37).all()


@contextmanager
def _session(path, owners, graphs):
    session = RecordedPreparationSession(device=torch.device("cuda:0"), checkpoint=str(path),
        workload_schema="fixed-reduction", numerical_policy="default", compile_workers=1)
    owners.append(session)
    failure = None
    try:
        yield session
    except BaseException as error:
        failure = error
        raise
    finally:
        if failure is not None and resources_retained(failure):
            retain_undrained(failure, owners, graphs)
        else:
            try:
                torch.cuda.synchronize()
                for graph in graphs:
                    graph.reset()
            except BaseException as error:
                retain_undrained(error, owners, graphs)
                raise
            session.close()


@torch.no_grad()
@pytest.mark.parametrize("width,columns", [(64, 1), (64, 3), (64, 17),
    (2048, 4096), (2048, 2048), (2048, 12288), (6144, 2048), (2048, 151936)])
def test_tuned_compiled_rows_rounding_guards_nonfinite_and_recovery(tmp_path, width, columns):
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        source = torch.zeros((8, width), device="cuda", dtype=torch.bfloat16)
        source[:, 0] = 1
        source[:, 16] = torch.arange(1, 9, device="cuda") / 256
        source[:, 32] = -1
        original = source.clone()
        weight = torch.ones((columns, width), device="cuda", dtype=torch.bfloat16)
        guard = torch.full((8 * columns + 16,), 37., device="cuda", dtype=torch.bfloat16)
        output = guard[8:-8].view(8, columns)
        count = torch.full((1,), 8, device="cuda", dtype=torch.int32)
        expected = original[:, 16:17].expand(8, columns).clone()
        premature = ((original[:, :1] + original[:, 16:17]).bfloat16()
            + original[:, 32:33]).bfloat16()
        assert not torch.equal(premature[0], expected[0, :1]), "control lost FP32 accumulation"
        owners.extend((images, source, original, weight, guard, count, expected))
        guard[0] = 0
        with pytest.raises(AssertionError):
            _guards_intact(guard)
        guard[0] = 37
        with _session(tmp_path, owners, graphs) as session:
            operation = PreparedReductionProjection(source, weight, output, count,
                session=session, name="projection")
            owners.append(operation)
            # Tuning must use its own live metadata and inputs, not these
            # deliberately invalid serving values.
            source.fill_(float("nan"))
            count.fill_(2**30)
            report = session.tune()
            assert report["benchmarked_candidates"] > 1
            source.copy_(original)
            count.fill_(8)
            forward = compile_forward(Forward(operation, source))
            owners.append(forward)
            forward()
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with session.capture(), torch.cuda.graph(graph):
                forward()
            for live in (1, 2, 4, 8, 3, 7, 0, 1, 8):
                for order in (list(range(live)), list(reversed(range(live)))):
                    source.fill_(float("nan"))
                    source[:live].copy_(original[order])
                    count.fill_(live)
                    target = torch.zeros_like(output)
                    target[:live].copy_(expected[order])
                    for _ in range(5):
                        output.fill_(float("inf"))
                        torch.cuda.synchronize()
                        before = torch.cuda.memory_stats()
                        graph.replay()
                        torch.cuda.synchronize()
                        after = torch.cuda.memory_stats()
                        for field in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                            assert before[field] == after[field], field
                        assert torch.equal(output.view(torch.uint8), target.view(torch.uint8))
                        _guards_intact(guard)
                    if live:
                        source[0, 0] = float("nan")
                        graph.replay()
                        torch.cuda.synchronize()
                        assert not torch.isfinite(output[0]).all()
                        assert torch.equal(output[1:].view(torch.uint8), target[1:].view(torch.uint8))
                        source[0].copy_(original[order[0]])
                        graph.replay()
                        torch.cuda.synchronize()
                        assert torch.equal(output.view(torch.uint8), target.view(torch.uint8))
            count.fill_(2)
            weight[0, 0] = float("nan")
            graph.replay()
            torch.cuda.synchronize()
            assert not torch.isfinite(output[:2, 0]).any()
            assert output[2:].eq(0).all()
            weight[0, 0] = 1
            for invalid in (-1, 9):
                count.fill_(invalid)
                graph.replay()
                torch.cuda.synchronize()
                assert not torch.isfinite(output).any()
            source.copy_(original)
            count.fill_(8)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
            _guards_intact(guard)
            # Exercise every accumulation chain with normal random operands.
            # Six fixed channels span the full output, including both ends;
            # the independent FP64 envelope is the already frozen linear bound.
            generator = torch.Generator(device=source.device).manual_seed(20261106)
            original.normal_(std=.1, generator=generator)
            weight.normal_(std=.1, generator=generator)
            channels = sorted({0, 1, columns // 4, columns // 2, max(0, columns - 2), columns - 1})
            channels = [value for value in channels if value < columns]
            exact, bound, _, _ = envelope(original.cpu(), weight[channels].cpu(), None)
            for live in (1, 2, 4, 8, 3, 7, 0, 8):
                for order in (list(range(live)), list(reversed(range(live)))):
                    source.fill_(float("nan"))
                    source[:live].copy_(original[order])
                    count.fill_(live)
                    previous = None
                    for _ in range(5):
                        graph.replay()
                        torch.cuda.synchronize()
                        actual = output[:live].cpu().contiguous()
                        _guards_intact(guard)
                        assert output[live:].eq(0).all()
                        assert torch.isfinite(actual).all()
                        if previous is not None:
                            assert torch.equal(actual.view(torch.uint8), previous.view(torch.uint8))
                        previous = actual
                        if live:
                            assert result_ratio(actual[:, channels], exact[order], bound[order]) <= 1
                    if live:
                        wrong = (actual[:, channels].double() + 4 * bound[order]).bfloat16()
                        assert result_ratio(wrong, exact[order], bound[order]) > 1
            graph.reset()
            graphs.remove(graph)


@torch.no_grad()
def test_binding_alias_and_replay_alias_rejections(tmp_path):
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        source = torch.ones((8, 64), device="cuda", dtype=torch.bfloat16)
        weight = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(source)
        count = torch.ones((1,), device="cuda", dtype=torch.int32)
        owners.extend((images, source, weight, output, count))
        with _session(tmp_path, owners, graphs) as session:
            with pytest.raises(ValueError, match="must not alias"):
                PreparedReductionProjection(source, weight, source, count, session=session, name="alias")
            operation = PreparedReductionProjection(source, weight, output, count,
                session=session, name="projection")
            owners.append(operation)
            with pytest.raises(ValueError, match="must not alias"):
                operation(output)
            with pytest.raises(ValueError, match="eight contiguous"):
                PreparedReductionProjection(source[:4], weight, output[:4], count,
                    session=session, name="short")
            bindings = (source, weight, output, count)
            for index in range(len(bindings)):
                changed = list(bindings)
                value = bindings[index]
                storage = torch.empty(value.numel() + 1, device=value.device, dtype=value.dtype)
                changed[index] = storage[1:].view_as(value)
                owners.append(storage)
                with pytest.raises(ValueError, match="16-byte aligned"):
                    PreparedReductionProjection(*changed, session=session, name="unaligned")
                if index == 0:
                    with pytest.raises(ValueError, match="16-byte aligned"):
                        operation(changed[0])


@torch.no_grad()
def test_zero_subnormal_overflow_and_nonfinite_classes_are_explicit(tmp_path):
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        cpu_source = torch.zeros((8, 64), dtype=torch.bfloat16)
        maximum = torch.finfo(torch.bfloat16).max
        cpu_source[:, 0] = torch.tensor([maximum, -maximum, 2.**-133, -2.**-133,
            float("inf"), float("nan"), float("nan"), float("nan")], dtype=torch.float64).bfloat16()
        cpu_weight = torch.zeros((17, 64), dtype=torch.bfloat16)
        cpu_weight[:2, 0] = torch.tensor([1., 2.], dtype=torch.bfloat16)
        reference = (cpu_source.double() @ cpu_weight.double().T).bfloat16()
        assert reference[2, 0].view(torch.int16).item() == 1
        assert torch.isinf(reference[0, 1]) and torch.isfinite(reference[0, 0])
        source, weight = cpu_source.cuda(), cpu_weight.cuda()
        guard = torch.full((8 * 17 + 16,), 37., device="cuda", dtype=torch.bfloat16)
        output = guard[8:-8].view(8, 17)
        count = torch.ones((1,), device="cuda", dtype=torch.int32)
        owners.extend((images, source, weight, guard, count))
        with _session(tmp_path, owners, graphs) as session:
            operation = PreparedReductionProjection(source, weight, output, count,
                session=session, name="classes")
            owners.append(operation)
            session.tune()
            forward = compile_forward(Forward(operation, source))
            owners.append(forward)
            forward()
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with session.capture(), torch.cuda.graph(graph):
                forward()
            for live in (0, 1, 2, 4, 6, 8, 0, 8):
                count.fill_(live)
                expected = reference.clone()
                expected[live:].zero_()
                for _ in range(5):
                    graph.replay()
                    torch.cuda.synchronize()
                    actual = output.cpu()
                    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
                    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
                    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
                    finite = torch.isfinite(expected)
                    assert torch.equal(actual[finite].view(torch.uint8), expected[finite].view(torch.uint8))
                    _guards_intact(guard)
            graph.reset()
            graphs.remove(graph)


@pytest.mark.parametrize("count", [0, 3, 8])
def test_exported_reduction_replays_after_preparation_exits(tmp_path, count):
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["cargo", "build", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools", "--bin", "test_fixed_runtime_package"],
        cwd=root, check=True, timeout=300)
    package = tmp_path / "reduction-projection"
    preparation = subprocess.run([sys.executable, "-m", "helios.tests.canonical_projection_package_fixture",
        str(package), "--capacity", "8", "--count", str(count), "--reduction"],
        cwd=root, text=True, capture_output=True, timeout=900)
    assert preparation.returncode == 0, preparation.stdout + preparation.stderr
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(root / "helios/target")))
    replay = subprocess.run([str(target / "debug/test_fixed_runtime_package"), str(package), "--replays", "5"],
        cwd=root, text=True, capture_output=True, timeout=180)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert "native fixed-address replay passed" in replay.stdout
