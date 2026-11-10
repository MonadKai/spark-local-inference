# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Real device snapshot controls; full model bridges run in the teacher tool."""

import os

import pytest
import torch

from ignition.preparation.failure import retain_on_failed_drain
from qualify.asr_candidate_teacher import compare_state
from qualify.asr_prefill_observation import BridgeSnapshots, compare_bridge, restore

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


def cpu(values):
    return {name: value.cpu() for name, value in values.items()}


def test_device_bridge_matches_cpu_bytes_and_restores_independent_snapshots():
    bridge = BridgeSnapshots()
    values = dict(kv=torch.arange(8192, device="cuda", dtype=torch.int32).view(torch.uint8),
        hidden=torch.zeros((3, 2048), device="cuda", dtype=torch.bfloat16),
        seed=torch.tensor([0, 2**64-1], dtype=torch.uint64, device="cuda"))
    with retain_on_failed_drain(0, bridge, values):
        original = bridge.snapshot(values)
        same = bridge.snapshot(values)
        assert compare_bridge(same, original)["passed"]
        assert compare_state(cpu(same), cpu(original))["passed"]
        values["kv"][-1] ^= 1
        values["hidden"].view(torch.int16)[-1, -1] = -32768
        changed = bridge.snapshot(values)
        gpu, host = compare_bridge(changed, original), compare_state(cpu(changed), cpu(original))
        assert not gpu["passed"] and not host["passed"]
        assert all(gpu["fields"][key]["exact_bytes"] == host["fields"][key]["exact_bytes"] for key in values)
        assert gpu["fields"]["kv"]["differing_bytes"] == 1
        assert gpu["fields"]["hidden"]["differing_bytes"] == 1
        assert gpu["fields"]["seed"]["exact_bytes"]
        restore(values, original)
        assert compare_bridge(values, original)["passed"]
        # The second snapshot is independent of both the input and first copy.
        original["kv"][-2] ^= 1
        assert not compare_bridge(original, same)["passed"]
        assert compare_bridge(values, same)["passed"]
        bridge.release()
        assert not bridge.owners


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_identical_nonfinite_device_states_cannot_pass(value):
    bridge = BridgeSnapshots()
    values = dict(hidden=torch.tensor([value], dtype=torch.bfloat16, device="cuda"))
    with retain_on_failed_drain(0, bridge, values):
        actual, expected = bridge.snapshot(values), bridge.snapshot(values)
        with pytest.raises(ValueError, match="nonfinite"):
            compare_bridge(actual, expected)
        bridge.release()


def test_device_bridge_rejects_missing_fields_geometry_dtype_and_host_copies():
    bridge = BridgeSnapshots()
    values = dict(kv=torch.ones((16,), device="cuda", dtype=torch.bfloat16))
    with retain_on_failed_drain(0, bridge, values):
        baseline = bridge.snapshot(values)
        for actual in ({}, dict(other=values["kv"]), dict(kv=values["kv"][:1]),
                dict(kv=values["kv"].float()), dict(kv=values["kv"].cpu())):
            with pytest.raises(ValueError):
                compare_bridge(actual, baseline)
        with pytest.raises(ValueError, match="nonempty CUDA"):
            bridge.snapshot({})
        bridge.release()
