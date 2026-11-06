# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Keep failed capture tracebacks alive while checking actual CUDA graph cleanup."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required")


@pytest.mark.parametrize("failure_stage", ["warmup", "capture", "instantiate"])
@torch.no_grad()
def test_partial_capture_failure_resets_all_created_graphs_before_return(monkeypatch, failure_stage):
    original_graph = torch.cuda.CUDAGraph
    observed = []

    class PlannedFailure(RuntimeError):
        pass

    class ObservedGraph(original_graph):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reset_calls = 0
            self.index = len(observed)
            observed.append(self)

        def reset(self):
            self.reset_calls += 1
            return super().reset()

        def instantiate(self):
            if self.index == 1 and failure_stage == "instantiate":
                raise PlannedFailure("planned graph instantiation failure")
            return super().instantiate()

    monkeypatch.setattr(torch.cuda, "CUDAGraph", ObservedGraph)
    source = torch.ones(32, device="cuda")
    output = torch.empty_like(source)
    second_calls = 0

    def first():
        output.copy_(source + 1)

    def second():
        nonlocal second_calls
        second_calls += 1
        if failure_stage == "warmup" and second_calls == 1:
            raise PlannedFailure("planned graph warmup failure")
        output.copy_(source * 2)
        if failure_stage == "capture" and second_calls == 2:
            raise PlannedFailure("planned graph capture failure")

    try:
        with PreparationSession(device="cuda:0", autotune=False) as session:
            with pytest.raises(PlannedFailure) as retained:
                capture_forwards(dict(first=first, second=second),
                    {name: GraphSignature("decode", 1, 1) for name in ("first", "second")},
                    session=session, device="cuda:0", stream=torch.cuda.Stream(), pool=None,
                    compile_decoder=False, retain_graphs=True)
            assert retained.value.__traceback__ is not None
            assert len(observed) == (1 if failure_stage == "warmup" else 2)
            assert all(graph.reset_calls == 1 for graph in observed)
            # A later independent graph still captures and runs on this context.
            graph = original_graph()
            with torch.cuda.graph(graph):
                output.copy_(source + 3)
            source.fill_(2)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output, torch.full_like(output, 5))
            graph.reset()
    finally:
        # The red control must also release its real graphs after observing the
        # missing cleanup, without making the assertion pass through GC.
        torch.cuda.synchronize()
        for graph in observed:
            original_graph.reset(graph)


def test_failed_drain_retains_real_graph_after_exception_disposal():
    # A separate context is required: retention is deliberately irreversible.
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "failed-drain"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("stage", ["warmup", "serialize"])
def test_raw_compile_retains_real_artifact_and_detects_unprotected_control(tmp_path, stage):
    for protected in (True, False):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            "raw-compile", stage, str(tmp_path / (stage + str(protected))), str(int(protected))],
            capture_output=True, text=True, timeout=180)
        if protected:
            assert result.returncode == 0, result.stdout + result.stderr
        else:
            assert result.returncode != 0 and "raw compilation lost undrained owners" in result.stderr, result.stdout + result.stderr


def _failed_drain_control():
    import gc
    import traceback
    import weakref
    from unittest.mock import patch
    from ignition.preparation.failure import resources_retained

    references, resets = [], []
    original_graph, synchronize = torch.cuda.CUDAGraph, torch.cuda.synchronize

    class ObservedGraph(original_graph):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            references.append(weakref.ref(self))

        def reset(self):
            resets.append(True)
            return super().reset()

    @torch.no_grad()
    def run():
        source, output = torch.ones(32, device="cuda"), torch.empty(32, device="cuda")
        session = PreparationSession(device="cuda:0", autotune=False)
        session.__enter__()
        failed = False

        def first():
            output.copy_(source + 1)

        def second():
            nonlocal failed
            output.copy_(source + 2)
            failed = True
            raise ValueError("planned second warmup failure")

        def fence(*args, **kwargs):
            if failed:
                raise RuntimeError("unconfirmed fence")
            return synchronize(*args, **kwargs)

        with patch.object(torch.cuda, "CUDAGraph", ObservedGraph), \
                patch.object(torch.cuda, "synchronize", fence):
            try:
                capture_forwards(dict(first=first, second=second),
                    {name: GraphSignature("decode", 1, 1) for name in ("first", "second")},
                    session=session, device="cuda:0", stream=torch.cuda.Stream(), pool=None,
                    compile_decoder=False, retain_graphs=True)
            except ValueError as error:
                assert str(error) == "planned second warmup failure" and resources_retained(error)
                return error
        raise AssertionError("failure did not propagate")

    error = run()
    # Independently confirm work completed for this control before process exit;
    # this does not change the preparation's unconfirmed-fence outcome.
    synchronize()
    traceback.clear_frames(error.__traceback__)
    error.__traceback__ = error.__cause__ = error.__context__ = None
    error.__dict__.clear()
    del error
    gc.collect()
    assert len(references) == 1 and references[0]() is not None and not resets


def _raw_compile_control(stage, root, protected):
    import gc
    import traceback
    import weakref
    from contextlib import nullcontext
    from unittest.mock import patch
    from torch import _inductor
    from ignition.preparation import compilation, failure
    from ignition.preparation.module_images import ModuleImages

    synchronize, compile_artifact = torch.cuda.synchronize, _inductor.standalone_compile
    references, invoked = [], []
    armed = False

    class Owner:
        pass

    class Artifact:
        def __init__(self, actual):
            self.actual = actual
            references.append(weakref.ref(self))

        def __call__(self, *inputs):
            nonlocal armed
            result = self.actual(*inputs)
            invoked.append("warmup")
            if stage == "warmup":
                armed = True
                raise ValueError("planned raw warmup failure")
            return result

        def serialize(self):
            nonlocal armed
            invoked.append("serialize")
            armed = True
            raise ValueError("planned raw serialization failure")

    def fence(*args, **kwargs):
        if armed:
            raise RuntimeError("unconfirmed raw compiler fence")
        return synchronize(*args, **kwargs)

    @torch.no_grad()
    def run():
        source, output = torch.ones(32, device="cuda"), torch.empty(32, device="cuda")
        owner, control = Owner(), Owner()
        references.extend((weakref.ref(owner), weakref.ref(control)))

        def forward():
            assert owner is not None
            output.copy_(source + 3)

        scope = nullcontext() if protected else patch.object(failure, "retain_on_failed_drain", lambda *a: nullcontext())
        with patch.object(compilation, "DEFAULT_CACHE_ROOT", Path(root)), \
                patch.object(_inductor, "standalone_compile", lambda *a, **k: Artifact(compile_artifact(*a, **k))), \
                patch.object(torch.cuda, "synchronize", fence), scope:
            try:
                compilation.compile_forward(forward)
            except ValueError as error:
                return error, output
        raise AssertionError("planned raw compiler fault did not run")

    with ModuleImages():
        torch.cuda.set_device(0)
        error, output = run()
        # The independent real fence makes the deliberately unprotected red
        # control safe before disposal. It does not release retained owners.
        synchronize()
        assert output.cpu().eq(4).all()
        retained = failure.resources_retained(error)
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = error.__cause__ = error.__context__ = None
        error.__dict__.clear()
        del error
        gc.collect()
        assert invoked == (["warmup"] if stage == "warmup" else ["warmup", "serialize"])
        assert references[1]() is None  # Uncaptured control is collected.
        assert retained and references[0]() is not None and references[2]() is not None, "raw compilation lost undrained owners"


if __name__ == "__main__":
    if sys.argv[1:] == ["failed-drain"]:
        _failed_drain_control()
    else:
        assert len(sys.argv) == 5 and sys.argv[1] == "raw-compile" and sys.argv[2] in ("warmup", "serialize")
        _raw_compile_control(sys.argv[2], sys.argv[3], bool(int(sys.argv[4])))
