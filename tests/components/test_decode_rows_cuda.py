# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiled decode metadata padding preserves changing packed holes and tails."""

import os

import pytest
import torch

from ignition.components.decode_rows import PreparedDecodeRows
from ignition.components.paged_batch import Metadata
from ignition.preparation.compilation import compile_forward
from ignition.preparation.failure import retain_on_failed_drain
from ignition.preparation.module_images import ModuleImages

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@torch.no_grad()
def test_compiled_decode_row_padding_tracks_holes_without_allocating(rows):
    with ModuleImages() as images:
        source = Metadata.allocate(rows, rows, 3, "cuda")
        target = Metadata.allocate(8, 8, 3, "cuda")
        operation = PreparedDecodeRows(source, target)
        owners, graphs = [source, target, operation, images], []
        with retain_on_failed_drain(0, owners, graphs):
            # Production mutates metadata in place and discards the descriptor.
            # Do not ask Inductor to export the dataclass (including its static
            # scalar capacity) as an output of this standalone control.
            def forward():
                operation()

            compiled = compile_forward(forward)
            owners.append(compiled)
            compiled()
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with torch.cuda.graph(graph):
                compiled()
            for counts in ([1] * rows, [i % 2 for i in range(rows)], [0] * rows, [1] * rows):
                source.set(torch.arange(rows, device="cuda") * 13,
                    torch.tensor(counts, dtype=torch.int32, device="cuda"),
                    torch.arange(rows, device="cuda") + 31,
                    torch.arange(rows * 3, dtype=torch.int32, device="cuda").reshape(rows, 3),
                    torch.arange(rows, device="cuda")[:, None] + 101,
                    torch.tensor([rows], device="cuda"))
                source.sample_indices.copy_((source.query_start_loc[1:] - 1).clamp_min(0))
                for name in target.__dataclass_fields__:
                    if name != "max_query_tokens":
                        getattr(target, name).fill_(137)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key], key
                for name in source.__dataclass_fields__:
                    if name == "max_query_tokens":
                        continue
                    actual, expected = getattr(target, name).cpu(), getattr(source, name).cpu()
                    n = rows + 1 if name == "query_start_loc" else rows
                    if name == "live_counts":
                        assert torch.equal(actual, expected)
                    else:
                        assert torch.equal(actual[:n], expected)
                        fill = (int(source.query_start_loc[-1]) if name == "query_start_loc"
                                else -1 if name in ("positions", "page_table", "token_page_table") else 0)
                        assert actual[n:].eq(fill).all()
            graph.reset()
