# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare shared-weight BF16 row projections with a fixed FP32 reduction."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import _NATIVE_JITS, attach_programs, compile_only_launches, launch_triton, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.fixtures.projection import input_factory
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare

from .native_operator import native_contract, native_plan


@dataclass(frozen=True)
class Query:
    columns: int
    reduction: int
    rows: int = 8

    def __post_init__(self):
        if (type(self.columns) is not int or type(self.reduction) is not int or type(self.rows) is not int
                or self.columns <= 0 or self.reduction <= 0 or self.reduction % 16
                or self.rows < 8 or self.rows % 8 or self.rows // 8 > 65535
                or max(self.rows * self.columns, self.rows * self.reduction,
                    self.columns * self.reduction) > 2**31 - 1):
            raise ValueError("fixed reduction extents must fit positive int32 element indexing and CUDA row-group bounds")


CONTRACT = native_contract(component_id="flare.reduction_projection", query_type=Query,
    tactics=lambda q: ((4, 4), (2, 4), (8, 4), (16, 4), (8, 8), (16, 8)))


@triton.jit
def _project(source, weight, output, count, K: tl.constexpr, N: tl.constexpr,
             ROWS: tl.constexpr, TILE: tl.constexpr):
    row_start = tl.program_id(1) * 8
    rows = row_start + tl.arange(0, 8)
    columns = tl.program_id(0) * TILE + tl.arange(0, TILE)
    lane = tl.arange(0, 16)
    part = tl.arange(0, 1)
    live = tl.load(count)
    active = rows < live
    accumulators = tl.full((8, TILE, 16, 1), 0, tl.float32)
    if live > row_start:
        for block in range(tl.cdiv(K, 16)):
            indices = block * 16 + part[None, :] * 16 + lane[:, None]
            x = tl.load(source + rows[:, None, None] * K + indices[None, :, :],
                active[:, None, None] & (indices[None, :, :] < K), other=0).to(tl.float32)
            w = tl.load(weight + columns[:, None, None] * K + indices[None, :, :],
                (columns[:, None, None] < N) & (indices[None, :, :] < K), other=0).to(tl.float32)
            accumulators = tl.fma(x[:, None, :, :], w[None, :, :, :], accumulators)
    result = tl.sum(tl.sum(accumulators, 3), 2)
    # Mask after reduction as well: inactive rows stay zero even if a weight
    # contains nonfinite data. Invalid metadata must not turn into success.
    result = tl.where(active[:, None], result, 0.)
    result = tl.where((live >= 0) & (live <= ROWS), result, float("nan"))
    tl.store(output + rows[:, None] * N + columns[None, :], result, columns[None, :] < N)


# b12x must evict discovery-only deferred entries before its workers compile.
_NATIVE_JITS.add(_project)


def _arguments(tensors, query, tactic):
    tile, warps = tactic
    return (*tensors, query.reduction, query.columns, query.rows, tile), \
        (triton.cdiv(query.columns, tile), query.rows // 8, 1), warps


@program_cache(scope="preparation")
def compile_projection(payload, tactic, ordinal):
    query = Query(**dict(payload))
    tensors = (torch.empty((query.rows, query.reduction), dtype=torch.bfloat16, device="meta"),
        torch.empty((query.columns, query.reduction), dtype=torch.bfloat16, device="meta"),
        torch.empty((query.rows, query.columns), dtype=torch.bfloat16, device="meta"),
        torch.empty((1,), dtype=torch.int32, device="meta"))
    args, grid, warps = _arguments(tensors, query, tactic)
    with torch.cuda.device(ordinal), compile_only_launches():
        return launch_triton(_project, grid, *args, num_warps=warps, enable_fp_fusion=False)


@dataclass(frozen=True)
class State:
    query: Query
    tactic: tuple
    program: object

    def run(self, source, weight, output, count):
        args, grid, _ = _arguments((source, weight, output, count), self.query, self.tactic)
        self.program[grid](*args)


@torch.library.custom_op("flare::reduction_projection", mutates_args=("output",))
def _execute(source: torch.Tensor, weight: torch.Tensor, output: torch.Tensor,
             count: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, source.device).run(source, weight, output, count)


@_execute.register_fake
def _fake(source, weight, output, count, handle):
    pass


def _overlap(left, right):
    return left.data_ptr() < right.data_ptr() + right.nbytes \
        and right.data_ptr() < left.data_ptr() + left.nbytes


class PreparedReductionProjection:
    """Project a live prefix in prepared storage in one kernel.

    Each program computes eight physical rows for a column tile; the launch
    covers a fixed positive multiple of eight rows. Entirely inactive row
    groups skip operand reads and still clear their outputs. The
    sixteen FP32 accumulation chains and final sum order are fixed; autotuning varies
    column tiles and thread counts, preserving the terminal BF16 node. There
    is no bias or intermediate BF16 product. No candidate changes arithmetic
    according to the live count. Inactive outputs are zero and active
    nonfinite values remain observable to the caller's completion guard.
    Inputs, output and count require 16-byte base alignment, matching the
    prepared program's pointer specializations. Element extents fit int32 and
    the row-group launch fits CUDA's 65535-entry grid Y bound.

    ``benchmark_rows`` describes the preparation trial's live prefix within
    that physical storage. Its invocation identity separates tuning choices
    without changing kernel geometry or the caller-owned runtime count.
    """

    def __init__(self, source, weight, output, count, *, session, name, benchmark_rows=None):
        if (source.ndim != 2 or source.shape[0] < 8 or source.shape[0] % 8 or source.shape[1] <= 0
                or source.shape[1] % 16 or output.ndim != 2 or output.shape[0] != source.shape[0]
                or output.shape[1] <= 0 or weight.shape != (output.shape[1], source.shape[1])
                or any(t.dtype != torch.bfloat16 or t.device != source.device or not t.is_contiguous()
                    for t in (source, weight, output)) or source.device.type != "cuda"
                or count.shape != (1,) or count.dtype != torch.int32
                or count.device != source.device or not count.is_contiguous()):
            raise ValueError("fixed reduction requires contiguous CUDA BF16 rows in groups of eight and an int32 count")
        if any(t.data_ptr() % 16 for t in (source, weight, output, count)):
            raise ValueError("fixed reduction bindings must be 16-byte aligned")
        if any(_overlap(output, value) for value in (source, weight, count)):
            raise ValueError("fixed reduction output must not alias its inputs")
        self.weight, self.output, self.count = weight, output, count
        self.shape = source.shape
        record_numeric_tensor(weight)
        query = Query(output.shape[1], source.shape[1], source.shape[0])
        if benchmark_rows is not None and (type(benchmark_rows) is not int
                or not 1 <= benchmark_rows <= query.rows):
            raise ValueError("benchmark rows must fit the prepared physical row capacity")
        trial_rows = query.rows if benchmark_rows is None else benchmark_rows
        payload = CONTRACT.encode_query(query)

        def materialize(tactic):
            program = compile_projection(payload, tactic, source.device.index)
            load_programs(program)
            return attach_programs(State(query, tactic, program), program)

        self.plan = native_plan(contract=CONTRACT, query=query, device=source.device,
            scratch_bytes=lambda t: 0, materialize=materialize,
            invocation={} if benchmark_rows is None else {"benchmark_rows": benchmark_rows},
            compile_jobs=lambda t, d: (CompileJob.create(
                "ignition.components.reduction_projection:compile_projection", payload, t, d.ordinal),))
        inputs = input_factory(source, output, storage=getattr(session, "benchmark_storage", None))

        def benchmark(state):
            operands = inputs()
            trial_count = torch.full((1,), trial_rows, dtype=torch.int32, device=source.device)
            return PreparedCall(run=lambda: state.run(operands.source, weight, operands.output, trial_count),
                output=operands.output, produce=operands.produce, owners=(operands, weight, trial_count))

        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.run(source, weight, output, count),
                output=output, owners=(source, weight, output, count)),
            benchmark_call=benchmark),))

    def __call__(self, source):
        if (source.shape != self.shape or source.dtype != torch.bfloat16
                or source.device != self.output.device or not source.is_contiguous()):
            raise ValueError("fixed reduction source differs from its prepared geometry")
        if not torch.compiler.is_compiling():
            if source.data_ptr() % 16:
                raise ValueError("fixed reduction source must be 16-byte aligned")
            if any(_overlap(self.output, value) for value in (source, self.weight, self.count)):
                raise ValueError("fixed reduction output must not alias its inputs")
        _execute(source, self.weight, self.output, self.count, self.plan.handle)
        return self.output
