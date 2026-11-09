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
from b12x._lib.compile_plan import load_programs

from ignition.components.reduction_projection import CONTRACT, Query, State, PreparedReductionProjection, compile_projection
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
def test_logical_benchmark_counts_and_reopened_selection_cache(tmp_path):
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        source = torch.full((8, 64), float("nan"), device="cuda", dtype=torch.bfloat16)
        weight = torch.ones((17, 64), device="cuda", dtype=torch.bfloat16)
        count = torch.full((1,), 2**30, device="cuda", dtype=torch.int32)
        owners.extend((images, source, weight, count))
        for reopened in (False, True):
            with _session(tmp_path, owners, graphs) as session:
                operations = []
                for rows in (1, 2, 4, 8):
                    output = torch.empty((8, 17), device="cuda", dtype=torch.bfloat16)
                    operation = PreparedReductionProjection(source, weight, output, count,
                        session=session, name=f"b{rows}", benchmark_rows=rows)
                    operations.append(operation)
                    owners.extend((output, operation))
                for invalid in (True, 0, -1, 9, 1.5):
                    with pytest.raises(ValueError, match="benchmark rows"):
                        PreparedReductionProjection(source, weight, operations[0].output, count,
                            session=session, name="invalid", benchmark_rows=invalid)
                requests = dict(session._recorded_requests)
                report = session.tune()
                assert not session._recorded_requests, "tuning must release its production declaration owners"
                if reopened:
                    assert report["cache_hits"] >= 4
                    assert report["benchmarked_candidates"] == 0
                else:
                    assert report["benchmarked_candidates"] >= 24
                for rows, operation in zip((1, 2, 4, 8), operations, strict=True):
                    request = requests[f"b{rows}"]
                    call = request.benchmark_call(operation.plan.prepared.state)
                    owners.append(call)
                    trial_count = call.owners[-1]
                    assert trial_count.tolist() == [rows]
                    assert trial_count.data_ptr() != count.data_ptr()
                    assert call.output.shape == (8, 17)
                    call.produce()
                    call.invoke()
                    torch.cuda.synchronize()
                    assert torch.isfinite(call.output[:rows]).all()
                    assert call.output[rows:].eq(0).all()
                    assert count.tolist() == [2**30]
                    assert torch.isnan(source).all()
                source.fill_(1)
                count.fill_(8)
                for operation in operations:
                    forward = compile_forward(Forward(operation, source))
                    owners.append(forward)
                    forward()
                    graph = torch.cuda.CUDAGraph()
                    graphs.append(graph)
                    with session.capture(), torch.cuda.graph(graph):
                        forward()
                    for live in (0, 1, 2, 4, 8):
                        count.fill_(live)
                        graph.replay()
                        torch.cuda.synchronize()
                        assert operation.output[:live].eq(64).all()
                        assert operation.output[live:].eq(0).all()
                    graph.reset()
                    graphs.remove(graph)
                source.fill_(float("nan"))
                count.fill_(2**30)


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
            with pytest.raises(ValueError, match="groups of eight"):
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


@torch.no_grad()
@pytest.mark.parametrize("width,columns,rows", [(64, 17, 40), (64, 17, 4096),
    (2048, 4096, 16), (2048, 2048, 16), (2048, 12288, 16),
    (6144, 2048, 16), (2048, 151936, 16)])
def test_all_tactics_preserve_eight_row_arithmetic_at_larger_capacities(width, columns, rows):
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        generator = torch.Generator(device="cuda").manual_seed(20261106)
        source = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16, generator=generator) * .1
        original = source.clone()
        weight = torch.randn((columns, width), device="cuda", dtype=torch.bfloat16, generator=generator) * .1
        guard = torch.full((rows * columns + 16,), 37., device="cuda", dtype=torch.bfloat16)
        output = guard[8:-8].view(rows, columns)
        count = torch.full((1,), rows, device="cuda", dtype=torch.int32)
        small_source = torch.empty((8, width), device="cuda", dtype=torch.bfloat16)
        small_output = torch.empty((8, columns), device="cuda", dtype=torch.bfloat16)
        small_count = torch.full((1,), 8, device="cuda", dtype=torch.int32)
        small_query = Query(columns, width)
        small_program = compile_projection(CONTRACT.encode_query(small_query), (4, 4), 0)
        load_programs(small_program)
        small = State(small_query, (4, 4), small_program)
        expected = torch.empty_like(output)
        owners.extend((images, source, original, weight, guard, count, small_source,
            small_output, small_count, small_program, small, expected))
        for start in range(0, rows, 8):
            small_source.copy_(source[start:start + 8])
            small.run(small_source, weight, small_output, small_count)
            expected[start:start + 8].copy_(small_output)
        torch.cuda.synchronize()
        channels = sorted({0, 1, columns // 2, columns - 1})
        exact, bound, _, _ = envelope(original[:8].cpu(), weight[channels].cpu(), None)
        assert result_ratio(expected[:8, channels].cpu(), exact, bound) <= 1
        query = Query(columns, width, rows)
        for tactic in ((4, 4), (2, 4), (8, 4), (16, 4), (8, 8), (16, 8)):
            program = compile_projection(CONTRACT.encode_query(query), tactic, 0)
            load_programs(program)
            state = State(query, tactic, program)
            owners.extend((program, state))
            count.fill_(rows)
            source.copy_(original)
            state.run(source, weight, output, count)
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with torch.cuda.graph(graph):
                state.run(source, weight, output, count)
            for live in dict.fromkeys((0, 1, 2, 4, 8, 3, 7, 9, rows - 1, rows)):
                for reverse in (False, True):
                    source.fill_(float("nan"))
                    values = original[:live].flip(0) if reverse else original[:live]
                    source[:live].copy_(values)
                    target = torch.zeros_like(output)
                    target[:live].copy_(expected[:live].flip(0) if reverse else expected[:live])
                    count.fill_(live)
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
                        assert torch.equal(output.view(torch.uint8), target.view(torch.uint8)), (tactic, live, reverse)
                        _guards_intact(guard)
            # An error in a later row group must remain visible; inactive
            # groups must ignore the same poisoned operand and clear output.
            source.copy_(original)
            source[-1, 0] = float("nan")
            count.fill_(rows)
            graph.replay()
            torch.cuda.synchronize()
            assert not torch.isfinite(output[-1]).all()
            assert torch.equal(output[:-1].view(torch.uint8), expected[:-1].view(torch.uint8))
            count.fill_(rows - 1)
            graph.replay()
            torch.cuda.synchronize()
            assert output[-1].eq(0).all()
            saved = weight[0, 0].clone()
            weight[0, 0] = float("nan")
            count.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert output.eq(0).all()
            count.fill_(9)
            graph.replay()
            torch.cuda.synchronize()
            assert not torch.isfinite(output[:9, 0]).any() and output[9:].eq(0).all()
            weight[0, 0].copy_(saved)
            for invalid in (-1, rows + 1, 2**31 - 1):
                count.fill_(invalid)
                graph.replay()
                torch.cuda.synchronize()
                assert not torch.isfinite(output).any()
            source.copy_(original)
            count.fill_(rows)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
            _guards_intact(guard)
            graph.reset()
            graphs.remove(graph)


@torch.no_grad()
def test_largest_legal_row_grid_captures_and_replays():
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        rows = 8 * 65535
        source = torch.full((rows, 16), .5, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones((1, 16), device="cuda", dtype=torch.bfloat16)
        guard = torch.full((rows + 16,), 37., device="cuda", dtype=torch.bfloat16)
        output = guard[8:-8].view(rows, 1)
        count = torch.full((1,), rows, device="cuda", dtype=torch.int32)
        query = Query(1, 16, rows)
        program = compile_projection(CONTRACT.encode_query(query), (4, 4), 0)
        load_programs(program)
        state = State(query, (4, 4), program)
        owners.extend((images, source, weight, guard, count, program, state))
        state.run(source, weight, output, count)
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            state.run(source, weight, output, count)
        for live in (0, 1, rows - 1, rows):
            count.fill_(live)
            for _ in range(5):
                output.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for field in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                    assert before[field] == after[field], field
                assert output[:live].eq(8).all() and output[live:].eq(0).all()
                _guards_intact(guard)
        graph.reset()
        graphs.remove(graph)


@torch.no_grad()
def test_larger_capacity_uses_normal_tuning_complete_compilation_and_capture(tmp_path):
    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images:
        source = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
        weight = torch.ones((17, 64), device="cuda", dtype=torch.bfloat16)
        output = torch.empty((64, 17), device="cuda", dtype=torch.bfloat16)
        count = torch.full((1,), 37, device="cuda", dtype=torch.int32)
        owners.extend((images, source, weight, output, count))
        with _session(tmp_path, owners, graphs) as session:
            operation = PreparedReductionProjection(source, weight, output, count,
                session=session, name="larger")
            owners.append(operation)
            report = session.tune()
            assert report["benchmarked_candidates"] > 1
            forward = compile_forward(Forward(operation, source))
            owners.append(forward)
            forward()
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with session.capture(), torch.cuda.graph(graph):
                forward()
            for live in (37, 0, 1, 2, 4, 8, 63, 64, 9):
                count.fill_(live)
                for _ in range(5):
                    graph.replay()
                    torch.cuda.synchronize()
                    assert output[:live].eq(64).all() and output[live:].eq(0).all()


@pytest.mark.parametrize("capacity,count", [(8, 0), (8, 3), (8, 8),
    (64, 0), (64, 37), (128, 127), (256, 256)])
def test_exported_reduction_replays_after_preparation_exits(tmp_path, capacity, count):
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["cargo", "build", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools", "--bin", "test_fixed_runtime_package"],
        cwd=root, check=True, timeout=300)
    package = tmp_path / "reduction-projection"
    preparation = subprocess.run([sys.executable, "-m", "helios.tests.canonical_projection_package_fixture",
        str(package), "--capacity", str(capacity), "--count", str(count), "--reduction"],
        cwd=root, text=True, capture_output=True, timeout=900)
    assert preparation.returncode == 0, preparation.stdout + preparation.stderr
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(root / "helios/target")))
    replay = subprocess.run([str(target / "debug/test_fixed_runtime_package"), str(package), "--replays", "5"],
        cwd=root, text=True, capture_output=True, timeout=180)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert "native fixed-address replay passed" in replay.stdout
