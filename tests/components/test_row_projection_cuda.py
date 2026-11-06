# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed physical GEMM rows preserve logical rows and captured storage."""

import os

import pytest
import torch

from ignition.components.row_projection import PreparedRowProjection
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import GraphBufferPool
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("width,columns", [(2048, 1024), (2048, 4096), (2048, 2048), (2048, 6144), (2048, 12288),
    (6144, 2048), (2048, 151936)])
@torch.no_grad()
def test_fixed_rows_cover_decoder_projections_and_inactive_padding(width, columns):
    device = torch.device("cuda:0")
    # Integer products/sums are exactly representable in FP32. This gives a
    # mathematical BF16 output control independently of a competing GEMM.
    a = (torch.arange(width, device=device) % 7 - 3).float()
    b = (torch.arange(width, device=device) % 3 - 1).float()
    factors = (torch.arange(columns, device=device) % 5 - 2).float()
    weight = (factors[:, None] * b[None] / 64).bfloat16()
    source = ((a[None] + torch.arange(8, device=device)[:, None]) / 32).bfloat16()
    expected = (((a * b).sum() + torch.arange(8, device=device) * b.sum())[:, None]
        * factors[None] / 2048).bfloat16()
    buffers = GraphBufferPool(device)
    graphs, compiled, operations, outputs, guards = {}, {}, {}, {}, {}
    with RecordedPreparationSession(device=device, checkpoint="fixed-row-projection",
            workload_schema="fixed-eight-rows", numerical_policy="bf16-fp32", compile_workers=2) as session:
        for rows in (1, 2, 4, 8):
            guarded = guards[rows] = torch.full((rows, columns + 2), 137, device=device, dtype=torch.bfloat16)
            output = outputs[rows] = guarded[:, 1:-1]
            operations[rows] = PreparedRowProjection(source[:rows], weight, output,
                row_capacity=8, buffers=buffers.cursor(), session=session, name=f"rows{rows}")
        session.tune()
        for rows, operation in operations.items():
            compiled[rows] = compile_forward(lambda rows=rows, operation=operation: operation(source[:rows]))
            compiled[rows]()
            graphs[rows] = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graphs[rows]):
                compiled[rows]()
        session.freeze()
        try:
            for rows in (1, 2, 4, 8, 2, 1):
                operation = operations[rows]
                operation.source.fill_(float("nan"))
                operation.projected.fill_(float("nan"))
                outputs[rows].fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graphs[rows].replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                assert torch.equal(outputs[rows], expected[:rows])
                assert operation.source[rows:].eq(0).all()
                assert operation.projected[rows:].eq(0).all()
                assert guards[rows][:, (0, -1)].eq(137).all()
            for rows in (1, 2, 4):
                assert torch.equal(outputs[rows].view(torch.uint8), outputs[8][:rows].view(torch.uint8))
        finally:
            torch.cuda.synchronize()
            for graph in graphs.values():
                graph.reset()
