# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise native strong references in subprocesses, without initializing CUDA.

Faults replace GPU/compiler calls; these tests qualify exception ownership, not
CUDA fences or tensor correctness. Real capture/fence tests are separate.
"""

import gc
import subprocess
import sys
import traceback
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ignition.preparation.failure import (
    resources_retained, retain_undrained, retain_on_failed_drain, restore_backing,
)


class Owner:
    pass


def discard(error):
    pending, seen = [error], set()
    while pending:
        value = pending.pop()
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        pending.extend((value.__cause__, value.__context__))
        if value.__traceback__ is not None:
            traceback.clear_frames(value.__traceback__)
        value.__traceback__ = value.__cause__ = value.__context__ = None
        value.__dict__.clear()
    gc.collect()


def native_reference():
    owner, control = Owner(), Owner()
    kept, released = weakref.ref(owner), weakref.ref(control)
    error = RuntimeError("no fence")
    retain_undrained(error, owner)
    del owner, control
    discard(error)
    assert released() is None  # Prove collection is active.
    assert kept() is not None


def grouped_failure():
    def run():
        inner, outer = Owner(), Owner()
        references = [weakref.ref(inner), weakref.ref(outer)]
        primary = RuntimeError("unconfirmed drain")
        retain_undrained(primary, inner)
        grouped = BaseExceptionGroup("cleanup", [KeyboardInterrupt(),
            ExceptionGroup("nested", [ValueError("ordinary"), primary])])
        try:
            with retain_on_failed_drain("cuda:0", outer):
                raise grouped
        except BaseExceptionGroup as error:
            assert resources_retained(error)
            return error, references
    error, references = run()
    discard(error)
    assert all(reference() is not None for reference in references)


def nested_transient(body_fails):
    from ignition.preparation.package_storage import PackageStorage

    references, cleanup = [], []

    class Pool(Owner):
        id = (1, 1)

        def __init__(self, allocator):
            references.append(weakref.ref(self))

    def synchronize(*args):
        raise RuntimeError("planned drain failure")

    def run():
        storage = PackageStorage.__new__(PackageStorage)
        storage.device, storage._retention_scope, storage._preparation_pools = 0, False, set()
        storage._allocator = Owner()
        storage._allocator.allocator = lambda: None
        storage.seal_low_window = lambda: cleanup.append("seal")
        references.extend((weakref.ref(storage), weakref.ref(storage._allocator)))
        primary = ValueError("planned body failure")
        with patch.object(torch.cuda, "MemPool", Pool), \
                patch.object(torch.cuda, "use_mem_pool", lambda *a, **k: nullcontext()), \
                patch.object(torch.cuda, "synchronize", synchronize), \
                patch.object(torch.cuda, "empty_cache", lambda: cleanup.append("empty")):
            try:
                with storage.transient(), storage.transient():
                    if body_fails:
                        raise primary
            except BaseException as error:
                assert resources_retained(error)
                assert (error is primary) if body_fails else str(error) == "planned drain failure"
                return error
        pytest.fail("failure did not propagate")

    error = run()
    discard(error)
    assert len(references) == 4 and all(reference() is not None for reference in references)
    assert cleanup == []


def partial_aot():
    from ignition.preparation import aot, compile_pool
    from ignition.preparation.inductor_tuning import InductorSelections
    import tempfile

    references = []

    class Pool:
        workers = 1

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Artifact(Owner):
        _artifacts = SimpleNamespace(compiled_fn=SimpleNamespace(artifact=SimpleNamespace(_flare_graph=True)))

        def __call__(self, owner):
            if owner.index == 2:
                raise ValueError("second forward failed")

        def save_compiled_function(self, path):
            pass

    def prepare(cache, owner, *, name, **kwargs):
        bound = aot.BoundForward(Artifact(), owner, True)
        references.extend((weakref.ref(bound), weakref.ref(bound.artifact), weakref.ref(owner)))
        return bound, cache.root / name

    def register(selections, bound):
        selections.forwards[id(bound)] = (SimpleNamespace(_tuning_frozen=False), None, {})

    def synchronize(*args):
        raise RuntimeError("planned drain failure")

    def run(root):
        cache = object.__new__(aot.ForwardCache)
        cache.device, cache.root, cache.options = "cuda:0", Path(root), {}
        cache.headroom_bytes, cache.compile_workers, cache.lock_gpu_clocks = 0, 1, False
        first, second = Owner(), Owner()
        first.index, second.index = 1, 2
        with patch.object(compile_pool, "CompilePool", Pool), \
                patch.object(aot.ForwardCache, "_prepare", prepare), \
                patch.object(aot, "_guard_code_cache", lambda path: nullcontext()), \
                patch.object(InductorSelections, "register", register), \
                patch.object(InductorSelections, "_reconcile", lambda *args: None), \
                patch.object(InductorSelections, "freeze", lambda *args: None), \
                patch.object(torch.cuda, "current_stream", lambda *a: SimpleNamespace(synchronize=lambda: None)), \
                patch.object(torch.cuda, "synchronize", synchronize):
            try:
                cache._prepare_many({"first": first, "second": second}, {"first": None, "second": None},
                    progress=None, coordinator=None, materialize=None, warmup=nullcontext, before_run=None)
            except RuntimeError as error:
                assert resources_retained(error)
                assert "second" in str(error)
                return error
        pytest.fail("second warmup did not fail")

    with tempfile.TemporaryDirectory() as root:
        error = run(root)
    discard(error)
    assert len(references) == 6 and all(reference() is not None for reference in references)


def capture_roots():
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.execution import GraphSignature

    references, closed = [], []

    def run():
        root, callback_root = Owner(), Owner()
        references.extend((weakref.ref(root), weakref.ref(callback_root)))

        def forward():
            assert root is not None
            raise ValueError("warmup failed")

        def materialize():
            assert callback_root is not None

        stream = SimpleNamespace(wait_stream=lambda other: None)
        with patch.object(torch.cuda, "current_stream", lambda *a: None), \
                patch.object(torch.cuda, "stream", lambda *a: nullcontext()), \
                patch.object(torch.cuda, "synchronize", side_effect=RuntimeError("no fence")):
            try:
                with restore_backing(SimpleNamespace(materialize=lambda: closed.append("remap"))):
                    capture_forwards({"one": forward}, {"one": GraphSignature("decode", 1, 1)},
                        session=Owner(), device="cuda:0", stream=stream, pool=None, compile_decoder=False,
                        materialize=materialize)
            except ValueError as error:
                assert str(error) == "warmup failed" and resources_retained(error)
                return error
        pytest.fail("warmup did not fail")

    error = run()
    discard(error)
    assert all(reference() is not None for reference in references)
    assert closed == []


def peer_retention(failed_exchange=None):
    from ignition.preparation.inductor_tuning import InductorSelections

    def run():
        selections = InductorSelections("cuda:0")
        owner = Owner()
        reference = weakref.ref(owner)
        exchanges = []
        primary = ValueError("local warmup failed")

        def gather(value):
            exchanges.append(value)
            if len(exchanges) == failed_exchange:
                raise RuntimeError("transport failed")
            return [value, "peer failed"] if len(exchanges) == 1 else [value, True]

        selections._gather = gather
        try:
            with retain_on_failed_drain("cuda:0", owner):
                selections._check(None, error=primary)
        except RuntimeError as error:
            assert resources_retained(error) and error.__cause__ is primary
            assert exchanges == ([None] if failed_exchange == 1 else [None, False])
            return error, reference
        pytest.fail("peer failure did not propagate")

    error, reference = run()
    discard(error)
    assert reference() is not None


SCENARIOS = {
    "native": native_reference,
    "group": grouped_failure,
    "transient_body": lambda: nested_transient(True),
    "transient_exit": lambda: nested_transient(False),
    "partial_aot": partial_aot,
    "capture": capture_roots,
    "peer": peer_retention,
    "first_gather_failure": lambda: peer_retention(1),
    "second_gather_failure": lambda: peer_retention(2),
}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_retained_owners_survive_exception_disposal(scenario):
    result = subprocess.run([sys.executable, __file__, scenario], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    SCENARIOS[sys.argv[1]]()
    assert not torch.cuda.is_initialized()
