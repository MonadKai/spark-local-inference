# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Real compiler workers bind mixed inference metadata without parent retracing."""

from contextlib import nullcontext
import os
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.aot import BindingMismatch, ForwardCache
from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature
from ignition.preparation.failure import resources_retained, retain_on_failed_drain, retain_undrained
from ignition.preparation.module_images import ModuleImages

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an assigned CUDA device")


class MixedInferenceForward:
    def __init__(self):
        with torch.inference_mode(False):
            self.normal_backing = torch.arange(16, dtype=torch.float32, device="cuda")
            self.normal = torch.nn.Parameter(self.normal_backing[::2], requires_grad=False)
            self.output = torch.empty(8, device="cuda")
        with torch.inference_mode():
            self.inference_backing = torch.arange(16, dtype=torch.float32, device="cuda")
            self.inference = self.inference_backing[1::2]
        self.again = self.normal

    def __call__(self):
        self.output.copy_(self.normal + self.inference)
        return self.output


@pytest.mark.parametrize("ambient", [False, True])
@torch.no_grad()
def test_cold_workers_hot_capture_and_inference_identity_rejection(tmp_path, monkeypatch, ambient):
    from ignition.preparation import aot
    from ignition.preparation.compile_pool import CompilePool

    owners, graphs = [], []
    with retain_on_failed_drain(0, owners, graphs), ModuleImages() as images, torch.inference_mode(ambient):
        session = SimpleNamespace(capture=nullcontext,
            namespace=SimpleNamespace(to_dict=lambda: {"model": "mixed-inference-metadata"}),
            requested_compile_workers=1)
        stream, pool = torch.cuda.Stream(), torch.cuda.graph_pool_handle()
        signature = GraphSignature("decode", 1, 1)
        options = dict(session=session, device=torch.device("cuda:0"), stream=stream,
            pool=pool, cache_root=tmp_path, materialize=lambda: None)
        owners.extend((images, session, stream, pool))
        failure = None
        try:
            def forbidden(*args, **kwargs):
                raise AssertionError("parent retracing or unexpected warm compiler submission")

            monkeypatch.setattr(aot, "aot_forward", forbidden)
            submitted = []
            submit = CompilePool.submit

            def record_submission(compiler, forward, path):
                submitted.append(path)
                return submit(compiler, forward, path)

            monkeypatch.setattr(CompilePool, "submit", record_submission)
            for warm in (False, True):
                forward = MixedInferenceForward()
                functions = {"decode": forward}
                owners.extend((forward, functions))
                if warm:
                    monkeypatch.setattr(CompilePool, "submit", forbidden)
                captured, compiled = capture_forwards(functions, {"decode": signature}, **options)
                graphs.extend(captured.values())
                owners.extend((captured, compiled))
                # Deferred binding reloads even a just-compiled artifact.
                # Count actual worker submissions to distinguish cold/hot work.
                assert len(submitted) == 1
                if warm:
                    assert compiled["decode"].cache_hit
                for value in (3, 5, 7):
                    forward.normal.fill_(value)
                    with torch.inference_mode():
                        forward.inference.fill_(11)
                    expected = torch.full((8,), value + 11, dtype=torch.float32)
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    captured["decode"].replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for field in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                        assert before[field] == after[field], field
                    torch.testing.assert_close(forward.output.cpu(), expected, rtol=0, atol=0)

            wrong = MixedInferenceForward()
            owners.append(wrong)
            with torch.inference_mode():
                alias = torch.empty(0, device="cuda").set_(wrong.normal.untyped_storage(),
                    wrong.normal.storage_offset(), wrong.normal.shape, wrong.normal.stride())
                wrong.normal = torch.nn.Parameter(alias, requires_grad=False)
                wrong.again = wrong.normal
            assert wrong.normal.is_inference()
            wrong.output.fill_(-91)
            cache = ForwardCache(namespace=session.namespace.to_dict(), device="cuda:0", root=tmp_path)
            owners.append(cache)
            path = cache._path(wrong, name="decode", signature=signature)
            assert path.is_file()
            with pytest.raises(BindingMismatch, match="dispatch key set mismatch"):
                cache._load(wrong, path, required=True)
            assert wrong.output.eq(-91).all()
        except BaseException as error:
            failure = error
            raise
        finally:
            if failure is not None and resources_retained(failure):
                retain_undrained(failure, owners, graphs)
            else:
                try:
                    torch.cuda.synchronize()
                    for graph in graphs:
                        graph.reset()
                except BaseException as error:
                    retain_undrained(error, owners, graphs)
                    raise
