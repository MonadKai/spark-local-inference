# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The approved fixed selection retains common native completion/state guards."""

import io
import os
from contextlib import contextmanager

import pytest
import torch

from ignition.components.sampling import (
    DUPLICATE_SLOT, ERROR, FINISHED, INVALID_DISTRIBUTION, RUNNING,
    DeviceRequestState, PreparedSampling, RequestAdmission, SamplingSelection,
)
from ignition.preparation.graphs import PreparedGraphBundle
from tests.components.completion_support import completion_outputs

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)
VOCAB = 151936


@contextmanager
def captured(rows, *, compiled=False, aot=False, serialized=None):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    state = DeviceRequestState.allocate("cuda", vocab_size=VOCAB, capacity=8)
    logits = torch.zeros((rows, VOCAB), device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(rows, device="cuda", dtype=torch.int32)
    epochs = torch.ones(rows, device="cuda", dtype=torch.int64)
    live = torch.zeros(1, device="cuda", dtype=torch.int32)
    records = torch.full((2, 8, 32), -99, device="cuda", dtype=torch.int64)
    frame = torch.zeros(1, device="cuda", dtype=torch.int32)
    sequence = torch.ones(1, device="cuda", dtype=torch.int64)
    enabled = torch.ones(rows, device="cuda", dtype=torch.int32)
    sampler = PreparedSampling(logits, state, slots, epochs, live, completion_outputs(records),
        frame, sequence, sample_enabled=enabled, selection=SamplingSelection.FIRST_MAXIMUM)
    sampler.prepare()
    call = sampler
    if aot or serialized is not None:
        from ignition.preparation.aot import aot_forward
        if serialized is None:
            artifact = aot_forward(sampler)
            artifact(sampler)
            serialized = type(artifact).serialize(artifact).serialized_data
        restored = torch.compiler.load_compiled_function(io.BytesIO(serialized))
        call = lambda: restored(sampler)
    elif compiled:
        from ignition.preparation.compilation import compile_forward
        call = compile_forward(sampler)
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    bundle = PreparedGraphBundle(device=0, graphs={"sample": graph},
        buffers={"logits": logits, "records": records},
        resources=(state, sampler, call, slots, epochs, live, frame, sequence, enabled))
    worker = bundle.start_worker()

    class Owner:
        def admit(self, slot, *, counter=0, **kwargs):
            values = dict(generation=1, seed=0, max_new_tokens=100, temperature=1.,
                          top_k=1, top_p=1., eos_token_ids=())
            values.update(kwargs)
            state.admit(slot, RequestAdmission(**values), [])
            state.rng_counter[slot] = counter

        def replay(self):
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            addresses = {key: value.data_ptr() for key, value in sampler.scratch_tensors.items()}
            ticket = worker.submit(0)
            result = worker.poll(timeout_ms=30000)
            assert result is not None and result.ticket == ticket and result.error is None
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            assert addresses == {key: value.data_ptr() for key, value in sampler.scratch_tensors.items()}
            return records[frame.item()].cpu()

    owner = Owner()
    owner.state, owner.logits, owner.slots = state, logits, slots
    owner.serialized = serialized
    owner.epochs, owner.live, owner.enabled, owner.sampler = epochs, live, enabled, sampler
    try:
        yield owner
    finally:
        worker.close()


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_first_maximum_is_invariant_to_rows_permutations_slots_seeds_and_bf16_ties(rows):
    with captured(rows) as owner:
        # Distinct FP32 dot products become an exact BF16 tie at the required
        # LM-head boundary. Skipping that rounding would choose token 1024.
        hidden = torch.tensor([[1., 2.**-10]], device="cuda")
        head = torch.tensor([[1., 0.], [1., 1.]], device="cuda")
        products = hidden @ head.T
        assert products.argmax().item() == 1
        rounded = products.to(torch.bfloat16)
        assert rounded[0, 0] == rounded[0, 1]
        cases = [([VOCAB - 1], VOCAB - 1), ([31, 32], 31),
                 ([1023, 1024, VOCAB - 1], 1023), (None, 0), ([0, VOCAB - 1], 0)]
        for rotation in (0, 3, 7):
            order = [(rotation + row) % 8 for row in range(rows)]
            owner.slots.copy_(torch.tensor(order, dtype=torch.int32))
            for live in sorted({1, rows}):
                owner.live.fill_(live)
                for iteration, (ties, expected) in enumerate(cases):
                    for slot in range(8):
                        owner.admit(slot, seed=(2**64 - 1 - slot), counter=2**32 + iteration)
                    owner.logits.fill_(-2.)
                    if ties is None:
                        owner.logits.zero_()
                    else:
                        owner.logits[:, ties] = 1.
                    if ties == [1023, 1024, VOCAB - 1]:
                        owner.logits[:, 1023:1025].copy_(rounded.expand(rows, -1))
                    result = owner.replay()
                    active = order[:live]
                    assert result[active, 4].tolist() == [expected] * live
                    assert owner.state.rng_counter.cpu().tolist() == [
                        2**32 + iteration + int(slot in active) for slot in range(8)]
                    assert owner.state.generated_tokens.cpu().tolist() == [int(slot in active) for slot in range(8)]
                    assert owner.state.histogram.sum().item() == live
                    assert (result[[slot for slot in range(8) if slot not in active], 4] == -1).all()


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_first_maximum_keeps_distinct_requests_isolated_across_capacities_and_row_orders(rows):
    anchors = [0, 31, 32, 255, 256, 65535, 65536, VOCAB - 1]
    with captured(rows) as owner:
        iteration = 0
        for begin in range(0, 8, rows):
            group = list(range(begin, begin + rows))
            for logical in (group, list(reversed(group))):
                for rotation in (0, 3, 7):
                    for live in sorted({1, rows}):
                        iteration += 1
                        slots = [(3 * index + rotation) % 8 for index in logical]
                        for slot in range(8):
                            owner.admit(slot, generation=iteration, seed=19 + slot)
                        owner.slots.copy_(torch.tensor(slots, dtype=torch.int32))
                        owner.epochs.fill_(iteration)
                        owner.live.fill_(live)
                        owner.logits.fill_(-2.)
                        for row, index in enumerate(logical):
                            # Distinct simultaneous minima of each tie set.
                            # Higher FP32 values round to the same BF16 value.
                            owner.logits[row, anchors[index]] = 1.
                            owner.logits[row, VOCAB - 1] = torch.tensor(1.001,
                                dtype=torch.bfloat16, device="cuda")
                        owner.logits[live:].fill_(torch.nan)
                        result = owner.replay()
                        expected = [anchors[index] for index in logical[:live]]
                        assert result[slots[:live], 4].tolist() == expected
                        for slot, token in zip(slots[:live], expected, strict=True):
                            assert owner.state.histogram[slot, token].item() == 1
                        assert owner.state.histogram.sum().item() == live
                        assert owner.state.rng_counter.cpu().tolist() == [int(slot in slots[:live]) for slot in range(8)]
                        if rows > 1 and live == rows:
                            # Deliberately misroute every request to row zero's
                            # input, then run the real native graph. This must
                            # fail the same per-request comparison, establishing
                            # that a broadcast/row-routing fault is observable.
                            wrong = owner.logits[:1].expand_as(owner.logits).clone()
                            owner.logits.copy_(wrong)
                            for slot in range(8):
                                owner.admit(slot, generation=iteration, seed=19 + slot)
                            wrong_result = owner.replay()
                            with pytest.raises(AssertionError):
                                assert wrong_result[slots, 4].tolist() == expected


@pytest.mark.parametrize("compiled,aot", [(True, False), (False, True)])
def test_first_maximum_compiled_and_restored_aot_replay(compiled, aot):
    with captured(8, compiled=compiled, aot=aot) as owner:
        owner.live.fill_(8)
        for iteration in range(3):
            for slot in range(8):
                owner.admit(slot, generation=iteration + 1)
            owner.epochs.fill_(iteration + 1)
            owner.logits.zero_()
            owner.logits[:, VOCAB - 2:] = 3.
            assert owner.replay()[:, 4].tolist() == [VOCAB - 2] * 8
        serialized = owner.serialized
    if aot:
        # Rebind to fresh physical buffers and request state after closing the
        # original replay owner. A callable over its original owner is weaker.
        with captured(8, serialized=serialized) as fresh:
            fresh.live.fill_(8)
            fresh.logits.zero_()
            fresh.logits[:, 31:33] = 2.
            for slot in range(8):
                fresh.admit(slot)
            assert fresh.replay()[:, 4].tolist() == [31] * 8


def test_first_maximum_nonfinite_all_masked_and_masked_nan_guards():
    with captured(4) as owner:
        for slot in range(8):
            owner.admit(slot)
        owner.live.fill_(4)
        owner.logits.zero_()
        owner.logits[0, 5] = torch.nan
        owner.logits[1, 5] = torch.inf
        owner.logits[3, 5] = torch.nan
        owner.state.grammar_enabled[2:4] = 1
        owner.state.grammar_mask[2:4].zero_()
        owner.state.grammar_mask[3, 0] = 1 << 7
        records = owner.replay()
        for slot in (0, 1, 2):
            assert records[slot, :8].tolist() == [1, 1, 0, 0, -1, ERROR, INVALID_DISTRIBUTION, 0]
        assert records[3, :8].tolist() == [1, 1, 0, 1, 7, RUNNING, 0, 0]
        assert owner.state.rng_counter.cpu().tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
        assert owner.state.histogram.sum().item() == 1


def test_first_maximum_stale_duplicate_disabled_and_terminal_rows():
    with captured(8) as owner:
        for slot in range(8):
            owner.admit(slot)
        owner.admit(4, max_new_tokens=1)
        owner.admit(5, eos_token_ids=(7,))
        owner.slots.copy_(torch.tensor([0, 0, 2, 3, 4, 5, 6, 7], dtype=torch.int32))
        owner.epochs[2] = 99
        owner.enabled[3] = 0
        owner.live.fill_(8)
        owner.logits.fill_(-torch.inf)
        owner.logits[:, 7] = 1.
        owner.logits[3].fill_(torch.nan)
        result = owner.replay()
        assert result[0, :8].tolist() == [1, 1, 0, 0, -1, ERROR, DUPLICATE_SLOT, 0]
        assert result[3, :8].tolist() == [1, 1, 0, 0, -1, RUNNING, 0, 0]
        assert result[[4, 5], 5].tolist() == [FINISHED] * 2
        assert owner.state.rng_counter.cpu().tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
        # Remove still-running rows; a replay after EOS/budget completion must
        # ignore even NaNs and leave every committed state tensor unchanged.
        owner.live.fill_(6)
        before = {key: getattr(owner.state, key).cpu().clone() for key in owner.state.__dataclass_fields__}
        owner.logits.fill_(torch.nan)
        owner.replay()
        for key, value in before.items():
            assert torch.equal(value, getattr(owner.state, key).cpu()), key
