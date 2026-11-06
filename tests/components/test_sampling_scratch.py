# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speculative scratch follows graph capacities and operation lifetimes."""

import pytest
import torch

from ignition.components.capacity import speculative_shapes
from ignition.components.parallel_draft import prepare_sampling_scratch
from ignition.preparation.scratch import GraphBufferPool, bind_scratch
from ignition.components.sampling import PreparedSampling, SamplingSelection


@pytest.mark.parametrize("width,proposal,verification,keys", [
    (6, True, True, {"proposal", "target", "verify"}),
    (6, True, False, {"proposal"}),
    (6, False, True, {"target", "verify"}),
    (1, True, True, {"target", "verify"}),
])
def test_sampling_allocates_only_executed_operations(width, proposal, verification, keys):
    pool = GraphBufferPool("cpu", chunk_bytes=256)
    work = prepare_sampling_scratch(4, width, 64, "cpu", buffers=pool.cursor(),
                                   proposal=proposal, verification=verification)
    assert work.keys() == keys
    for name, layout in work.items():
        live = list(layout.values())
        if name == "verify":
            live.extend(work["target"][key] for key in ("probabilities", "valid"))
        tensors = [tensor for tensor in live if tensor.numel()]
        spans = sorted((t.data_ptr(), t.data_ptr() + t.nbytes) for t in tensors)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        for index, tensor in enumerate(tensors, 1):
            tensor.fill_(bool(index % 2) if tensor.dtype == torch.bool else index)
        for index, tensor in enumerate(tensors, 1):
            assert torch.all(tensor == (bool(index % 2) if tensor.dtype == torch.bool else index))


def test_qwen_sampling_catalog_fits_bounded_shared_backing():
    pool = GraphBufferPool("meta", chunk_bytes=2 << 20)
    shapes = sorted(speculative_shapes(32, (1, 2, 3, 4, 5, 6), 32),
                    key=lambda shape: shape[0] * shape[1], reverse=True)
    for batch, width in shapes:
        cursor = pool.cursor()
        work = prepare_sampling_scratch(batch, width, 248320, "meta", buffers=cursor)
        cursor.zeros((batch, width - 1, 248320), torch.float32)
        assert work["target"]["scores"].numel() <= 32 * 248320
    assert pool.nbytes < 288 << 20
    capacity = pool.nbytes
    for batch, width in reversed(shapes):
        cursor = pool.cursor()
        prepare_sampling_scratch(batch, width, 248320, "meta", buffers=cursor)
        cursor.zeros((batch, width - 1, 248320), torch.float32)
    assert pool.nbytes == capacity


def test_fixed_selection_scratch_has_disjoint_typed_outputs_at_every_capacity():
    pool = GraphBufferPool("cpu", chunk_bytes=256)
    capacity = None
    for rows in (8, 1, 2, 4, 8):
        buffers = bind_scratch(PreparedSampling.scratch_layout(
            rows, 151936, selection=SamplingSelection.FIRST_MAXIMUM), device="cpu", workspace=pool.cursor())
        for index, tensor in enumerate(buffers.values(), 1):
            tensor.fill_(True if tensor.dtype == torch.bool else index)
        assert buffers["argmax_tokens"].dtype == torch.int64
        assert buffers["sampled_tokens"].dtype == torch.int32
        for index, tensor in enumerate(buffers.values(), 1):
            assert torch.all(tensor == (True if tensor.dtype == torch.bool else index))
        # The common planner must retain the first catalog's physical backing.
        if rows == 8:
            if capacity is not None:
                assert pool.nbytes == capacity
            capacity = pool.nbytes
