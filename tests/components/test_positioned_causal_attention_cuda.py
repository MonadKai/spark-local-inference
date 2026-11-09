# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Causal fragments preserve sequence arithmetic, bounds and reusable storage."""

from contextlib import contextmanager
import os
import traceback

import pytest
import torch

from ignition.components.positioned_causal_attention import PreparedPositionedCausalAttention
from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature
from ignition.preparation.failure import drain_or_retain, require_retention, retain_undrained
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


@contextmanager
def _prepared(capacity, length, rows, cache):
    require_retention()
    images = ModuleImages()
    data, operations, guards, graphs, compiled = {}, {}, {}, {}, {}
    owners = [images, data, operations, guards, graphs, compiled]
    session, initialized, entered, primary = None, False, False, None
    try:
        images.__enter__()
        entered = True
        torch.cuda.set_device(0)
        generator = torch.Generator(device="cuda:0").manual_seed(20261106)
        for name, heads in (("original_q", 16), ("original_k", 8), ("original_v", 8)):
            data[name] = torch.randn((length, heads, 128), device="cuda:0",
                dtype=torch.bfloat16, generator=generator)
        data["q"] = torch.empty((rows, 16, 128), device="cuda:0", dtype=torch.bfloat16)
        pages = (length + 255) // 256
        data["k"] = torch.empty((pages, 256, 8, 128), device="cuda:0", dtype=torch.bfloat16)
        data["v"] = torch.empty_like(data["k"])
        data["pages"] = torch.full((1, (capacity + 255) // 256), -1, device="cuda:0", dtype=torch.int32)
        for name in ("length", "count", "sequence"):
            data[name] = torch.zeros(1, device="cuda:0", dtype=torch.int32)
        data["positions"] = torch.full((rows,), -1, device="cuda:0", dtype=torch.int64)
        data["position_rows"] = torch.arange(max(rows, length), device="cuda:0", dtype=torch.int64)
        data["page_rows"] = torch.arange(pages, device="cuda:0", dtype=torch.int32)
        pool = GraphBufferPool("cuda:0")
        owners.append(pool)
        arena = ScratchArena("cuda:0", overlap=False)
        owners.append(arena)
        session = object.__new__(RecordedPreparationSession)
        owners.append(session)
        RecordedPreparationSession.__init__(session, device=0, checkpoint="positioned-causal-tests",
            workload_schema="positioned-causal-tests.v1", numerical_policy="BF16-FP32",
            compile_workers=1, headroom_bytes=4 << 30, autotune_effort="off")
        initialized = True
        session.__enter__()
        session.register_scratch(arena)
        for physical in sorted({rows, 64}):
            guards[physical] = torch.full((physical + 2, 16, 128), 123.,
                device="cuda:0", dtype=torch.bfloat16)
            operation = object.__new__(PreparedPositionedCausalAttention)
            operations[physical] = operation
            PreparedPositionedCausalAttention.__init__(operation, data["q"][:physical], data["k"], data["v"],
                data["pages"], data["length"], data["count"], data["positions"][:physical], data["sequence"],
                guards[physical][1:-1], capacity=capacity, session=session, workspace=arena,
                buffers=pool.cursor(), name=f"positioned{physical}")
        session.tune(show_progress=False)
        stream = torch.cuda.Stream()
        owners.append(stream)
        graph_pool = torch.cuda.graph_pool_handle()
        owners.append(graph_pool)

        def inactive(_name=None):
            for name in ("q", "k", "v"):
                data[name].fill_(float("nan"))
            for name in ("length", "count", "sequence"):
                data[name].zero_()
            data["positions"].fill_(-1)
            data["pages"].fill_(-1)
            torch.cuda.synchronize()

        signatures = {str(n): GraphSignature("prefill", 1, n, (), initial_prompt_capacity=capacity)
            for n in operations}
        g, c = capture_forwards({str(n): op for n, op in operations.items()}, signatures,
            session=session, device=0, stream=stream, pool=graph_pool, cache_root=cache,
            before_run=inactive, report_catalog=False)
        owners.append((g, c))
        graphs.update(g)
        compiled.update(c)
        session.freeze()

        def install(start, count, known):
            data["q"].fill_(float("nan"))
            data["q"][:count].copy_(data["original_q"][start:start + count])
            for name, original in (("k", "original_k"), ("v", "original_v")):
                data[name].fill_(float("nan"))
                data[name].flatten(0, 1)[:known].copy_(data[original][:known])
            data["pages"].fill_(-1)
            data["pages"][0, :pages].copy_(data["page_rows"])
            data["length"].fill_(known)
            data["count"].fill_(count)
            data["sequence"].fill_(length)
            data["positions"].fill_(-1)
            data["positions"][:count].copy_(data["position_rows"][start:start + count])

        def replay(physical):
            guards[physical].fill_(123.)
            if arena.tensor is not None:
                arena.tensor.fill_(255)
            torch.cuda.synchronize()
            inputs = {name: data[name].cpu().clone() for name in
                ("q", "k", "v", "pages", "length", "count", "sequence", "positions")}
            before = torch.cuda.memory_stats()
            graphs[str(physical)].replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for field in ("allocation.all.allocated", "allocation.all.freed",
                          "segment.all.allocated", "segment.all.freed"):
                assert before[field] == after[field]
            assert guards[physical][0].eq(123.).all() and guards[physical][-1].eq(123.).all()
            for name, expected in inputs.items():
                assert torch.equal(data[name].cpu().view(torch.uint8), expected.view(torch.uint8)), name
            return operations[physical].output.cpu().contiguous()

        yield data, operations, install, inactive, replay, session, arena, pool
    except BaseException as error:
        primary = error
        if not entered:
            retain_undrained(error, owners)
        raise
    finally:
        error = primary or RuntimeError("positioned-attention test cleanup could not confirm drain")
        if not drain_or_retain(error, 0, owners):
            traceback.print_exception(error)
            pytest.exit("CUDA drain failed; retained owners require discarding this process", returncode=1)
        try:
            for graph in graphs.values():
                graph.reset()
            compiled.clear()
            if initialized:
                session.close()
            images.__exit__(None, None, None)
        except BaseException as cleanup:
            retain_undrained(cleanup, owners)
            if primary is not None:
                primary.add_note(f"positioned-attention test cleanup also failed: {cleanup!r}")
            traceback.print_exception(primary or cleanup)
            pytest.exit("CUDA owner close failed; discard this process", returncode=1)


@pytest.mark.parametrize("capacity,length,rows", [(256, 19, 64), (256, 165, 4096),
    (512, 394, 512), (8192, 394, 4096)])
@torch.no_grad()
def test_fragments_keep_bytes_with_future_poison_and_recover(tmp_path, capacity, length, rows):
    with _prepared(capacity, length, rows, tmp_path) as (data, operations, install, inactive, replay, session, arena, pool):
        from b12x.attention import varlen

        reference = {}
        data["direct_reference"] = reference
        for name, heads in (("q", 16), ("k", 8), ("v", 8)):
            reference[name] = torch.zeros((capacity, heads, 128), device="cuda:0", dtype=torch.bfloat16)
            reference[name][:length].copy_(data[f"original_{name}"])
        reference["out"] = torch.empty_like(reference["q"])
        reference["cu"] = torch.tensor([0, length], device="cuda:0", dtype=torch.int32)
        varlen.run_out(reference["q"], reference["k"], reference["v"], reference["cu"],
            cu_seqlens_k=reference["cu"], out=reference["out"], scratch=operations[rows].scratch,
            plan=operations[rows].plan, max_seqlen_q=capacity, max_seqlen_k=capacity,
            softmax_scale=128**-0.5)
        torch.cuda.synchronize()
        expected = reference["out"][:length].cpu().contiguous()
        install(0, length, length)
        whole = replay(rows)[:length].clone()
        assert torch.isfinite(whole).all() and not operations[rows].errors.any()
        assert torch.equal(whole.view(torch.uint8), expected.view(torch.uint8))
        saved = {name: value.cpu().clone() for name, value in data.items() if name in ("original_q", "original_k", "original_v")}
        for start in range(0, length, 64):
            count = min(64, length - start)
            install(start, count, start + count)
            actual = replay(64)
            assert not operations[64].errors.any()
            assert torch.equal(actual[:count].view(torch.uint8), whole[start:start + count].view(torch.uint8))
            assert actual[count:].eq(0).all()
        if data["k"].shape[0] > 1:
            install(0, length, length)
            for name in ("k", "v"):
                data[f"permuted_{name}"] = data[name].flip(0)
                data[name].copy_(data[f"permuted_{name}"])
            pages = data["k"].shape[0]
            data["permuted_pages"] = data["page_rows"].flip(0)
            data["pages"][0, :pages].copy_(data["permuted_pages"])
            assert torch.equal(replay(rows)[:length].view(torch.uint8), whole.view(torch.uint8))
            assert not operations[rows].errors.any()
        for invalid in ("q_nan", "k_nan", "v_inf", "negative_page", "page_limit", "count_limit",
                        "negative_count", "gap", "huge_position", "wrapped_position",
                        "sequence_zero", "sequence_limit", "short_cache"):
            install(0, 2, 2)
            if invalid == "q_nan":
                data["q"][0, 0, 0] = float("nan")
            elif invalid == "k_nan":
                data["k"][0, 0, 0, 0] = float("nan")
            elif invalid == "v_inf":
                data["v"][0, 0, 0, 0] = float("inf")
            elif invalid == "negative_page":
                data["pages"][0, 0] = -1
            elif invalid == "page_limit":
                data["pages"][0, 0] = data["k"].shape[0]
            elif invalid == "count_limit":
                data["count"].fill_(rows + 1)
            elif invalid == "negative_count":
                data["count"].fill_(-1)
            elif invalid == "gap":
                data["positions"][1] = 2
            elif invalid == "huge_position":
                data["positions"][0] = 2**63 - 1
            elif invalid == "wrapped_position":
                data["positions"][0] = 2**32
            elif invalid == "sequence_zero":
                data["sequence"].zero_()
            elif invalid == "sequence_limit":
                data["sequence"].fill_(capacity + 1)
            else:
                data["length"].fill_(1)
            actual = replay(rows)
            assert operations[rows].errors.any(), invalid
            assert (~torch.isfinite(actual)).all(), invalid
        if length >= 65:
            install(0, 65, 65)
            assert (~torch.isfinite(replay(64))).all()
            assert operations[64].errors.any()
        inactive()
        assert replay(rows).eq(0).all() and not operations[rows].errors.any()
        install(0, length, length)
        for _ in range(5):
            assert torch.equal(replay(rows)[:length].view(torch.uint8), whole.view(torch.uint8))
        for name, expected in saved.items():
            assert torch.equal(data[name].cpu().view(torch.uint8), expected.view(torch.uint8))
        with pytest.raises(ValueError, match="alias"):
            PreparedPositionedCausalAttention(data["q"], data["k"], data["v"], data["pages"], data["length"],
                data["count"], data["positions"], data["sequence"], data["q"], capacity=capacity,
                session=session, workspace=arena, buffers=pool.cursor(), name="bad-alias")


@torch.no_grad()
def test_uniform_attention_keeps_constant_values_exact(tmp_path):
    with _prepared(512, 394, 512, tmp_path) as (data, operations, install, _, replay, *_owners):
        data["original_q"].zero_()
        data["original_k"].zero_()
        for head in range(8):
            data["original_v"][:, head].fill_(head + 1)
        install(0, 394, 394)
        actual = replay(512)[:394]
        expected = torch.tensor([head // 2 + 1 for head in range(16)], dtype=torch.bfloat16)
        expected = expected[None, :, None].expand(394, 16, 128).contiguous()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        assert not operations[512].errors.any()


@pytest.mark.parametrize("explicit_keys", [False, True])
@torch.no_grad()
def test_tuning_fixture_uses_private_full_key_lengths(tmp_path, explicit_keys):
    from b12x.preparation.types import require_prepared
    from ignition.preparation.fixtures import varlen_attention as fixtures

    with _prepared(256, 19, 64, tmp_path) as (data, operations, _install, _inactive, _replay, _session, arena, _pool):
        operation = operations[64]
        state = require_prepared(operation.plan, operation.plan.component_id)
        captured = {}
        data["benchmark_bindings"] = captured

        class ObservedState:
            scratch_plan = state.scratch_plan

            def bind(self, **arguments):
                captured.update(arguments)
                return state.bind(**arguments)

            def run(self, binding):
                return state.run(binding)

        arguments = dict(q=operation.query_staging, k=operation.key_staging,
            v=operation.value_staging, out=operation.attended,
            cu_seqlens_q=operation.cu, max_seqlen_q=256, max_seqlen_k=256,
            softmax_scale=128**-0.5, causal=True)
        if explicit_keys:
            arguments["cu_seqlens_k"] = operation.cu
        allocate = fixtures.input_factory(arguments)

        def acquire():
            inputs = allocate()
            data["benchmark_inputs"] = inputs
            return inputs

        call = fixtures.benchmark_call(ObservedState(), arguments=arguments,
            inputs=acquire, workspace=arena)
        data["benchmark_call"] = call
        expected = torch.tensor([0, 256], dtype=torch.int32)
        assert torch.equal(captured["cu_seqlens_q"].cpu(), expected)
        assert captured["cu_seqlens_q"].data_ptr() != operation.cu.data_ptr()
        if explicit_keys:
            assert captured["cu_seqlens_k"] is captured["cu_seqlens_q"]
            assert torch.equal(captured["cu_seqlens_k"].cpu(), expected)
        assert operation.cu.eq(0).all()
        assert captured["out"].data_ptr() != operation.attended.data_ptr()
        captured["out"].fill_(float("nan"))
        call.produce()
        call.run()
        torch.cuda.synchronize()
        assert torch.isfinite(captured["out"]).all()
        assert operation.cu.eq(0).all()
