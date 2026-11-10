# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Observer control-flow faults with actual CUDA work, not ASR qualification."""

import gc
import os
import traceback
from types import SimpleNamespace
import weakref

import pytest
import torch

from ignition.preparation.failure import resources_retained, retain_on_failed_drain
from qualify import asr_audio_network_nodes as audio
from qualify.asr_prefill_observation import BridgeSnapshots

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires exclusive CUDA and FLARE_RUN_CUDA_TESTS=1")


class Owner:
    pass


def graph(operation):
    operation()
    torch.cuda.synchronize()
    result = torch.cuda.CUDAGraph()
    with torch.cuda.graph(result):
        operation()
    return result


def dispose(error):
    traceback.clear_frames(error.__traceback__)
    error.__traceback__ = error.__cause__ = error.__context__ = None
    error.__dict__.clear()


@torch.no_grad()
@pytest.mark.parametrize("failure", ["replay_fence", "cleanup_fence"])
def test_failed_observer_fences_retain_owners_without_early_restore(monkeypatch, failure):
    synchronize, restore = torch.cuda.synchronize, audio.restore
    state = torch.ones((32,), dtype=torch.bfloat16, device="cuda")
    references, restores = [], []
    armed = False

    def run():
        nonlocal armed
        observer = audio.AudioObservation.__new__(audio.AudioObservation)
        observer.forward = None
        observer.bridge = BridgeSnapshots()
        observer.graphs = {"observed_audio": graph(lambda: state.add_(1))}
        production = graph(lambda: state.add_(1))
        observer.production = production
        prepared = Owner()
        prepared.device = 0
        references.extend((weakref.ref(observer), weakref.ref(prepared)))
        state.fill_(1)

        def fence(*args, **kwargs):
            if armed:
                raise RuntimeError("unconfirmed observer fence")
            return synchronize(*args, **kwargs)

        def restore_checked(values, saved):
            nonlocal armed
            assert not armed, "restored shared storage before confirming drain"
            restores.append("restore")
            restore(values, saved)
            if failure == "cleanup_fence" and len(restores) == 2:
                # Real work is outstanding when the cleanup exception occurs.
                state.add_(1)
                armed = True
                raise RuntimeError("injected cleanup failure after CUDA submission")

        def replay(value):
            nonlocal armed
            value.replay()
            if value is observer.graphs["observed_audio"]:
                armed = failure == "replay_fence"
                raise ValueError("injected observation failure after CUDA submission")

        with retain_on_failed_drain(0, observer, prepared), monkeypatch.context() as patch:
            # Isolate this controller property from model geometry; all state,
            # graph submissions, snapshots, fences and native retention are real.
            patch.setattr(audio, "bridge_buffers", lambda *args: {"state": state})
            patch.setattr(audio, "restore", restore_checked)
            patch.setattr(torch.cuda, "synchronize", fence)
            try:
                observer.observe(prepared, None, production, replay, 1, 1)
            except ValueError as error:
                assert resources_retained(error), "original observation error lost retained owners"
                assert len(observer.bridge.owners) == 2
                return error
        raise AssertionError("planned observer failure did not propagate")

    error = run()
    # Independent confirmation makes this deliberate test fault safe. It never
    # authorizes releasing the owner's already-retained preparation resources.
    synchronize()
    assert len(restores) == (1 if failure == "replay_fence" else 2)
    assert state.eq(2 if failure == "replay_fence" else 3).all()
    dispose(error)
    del error
    gc.collect()
    assert all(reference() is not None for reference in references)


@torch.no_grad()
def test_projector_plane_difference_fails_bridge_and_restores_production(monkeypatch):
    from qualify import asr_prefill_observation

    def zeros(shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype, device="cuda")

    projected = zeros((13, 1024))
    encoder = SimpleNamespace(encoder=SimpleNamespace(layers=[]),
        output=zeros((13, 2048)), convolution=zeros((13, 1024)),
        projector=SimpleNamespace(linear_1=SimpleNamespace(projection=SimpleNamespace(output=projected))))
    forward = SimpleNamespace(encoder=encoder, errors=zeros((1,), torch.int32), mel=zeros((128, 100), torch.float32),
        publish=SimpleNamespace(records=zeros((1, 8), torch.int64), frame=zeros((1,), torch.int32),
            sequence=zeros((1,), torch.int64)))
    observer = audio.AudioObservation.__new__(audio.AudioObservation)
    observer.forward, observer.bridge = forward, BridgeSnapshots()
    observer.nodes = SimpleNamespace(values=lambda *args: {"projector": projected.cpu().clone()})
    production = graph(lambda: projected.fill_(2))
    observed = graph(lambda: projected.fill_(4))
    observer.graphs = {"observed_audio": observed}
    projected.zero_()
    try:
        # Supply no decoder state; test the real audio-plane declaration and
        # comparison, rather than replacing this bridge's projection fields.
        monkeypatch.setattr(asr_prefill_observation, "bridge_buffers", lambda *args: {})
        nodes, result = observer.observe(SimpleNamespace(device=0), None, production,
            lambda value: value.replay(), 100, 100)
        assert nodes["projector"].eq(4).all()
        assert not result["passed"], "changed projector plane escaped the production bridge"
        assert not result["fields"]["audio.projector.linear_1"]["exact_bytes"]
        assert projected.eq(2).all() and not observer.bridge.owners
    finally:
        torch.cuda.synchronize()
        production.reset()
        observed.reset()
