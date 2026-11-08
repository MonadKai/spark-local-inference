# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inference tensor identity survives compiler metadata and guard serialization."""

import io
import inspect
import json
import pickle
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses import FakeTensor, FakeTensorMode

from ignition.preparation.aot import _GuardPickler, _GUARD_TENSOR_MODE
from ignition.preparation.forward_metadata import _Storage, _tensor, dumps, loads


def mixed_tensors():
    with torch.inference_mode(False):
        storage = torch.empty(256, dtype=torch.uint8)
        normal = torch.empty(0).set_(storage.untyped_storage(), 3, (3, 4), (5, 1))
        parameter = torch.nn.Parameter(normal, requires_grad=True)
        independent = torch.empty(8)
    with torch.inference_mode():
        inference = torch.empty(0, dtype=torch.bfloat16).set_(
            storage.untyped_storage(), 5, (4, 3), (7, 2))
        inference_parameter = torch.nn.Parameter(inference, requires_grad=True)
    return SimpleNamespace(storage=storage, normal=normal, parameter=parameter,
        inference=inference, inference_parameter=inference_parameter,
        independent=independent, again=inference)


class LegacyTensorMetadata:
    def __init__(self, kind):
        self.kind = kind

    def __reduce__(self):
        # The historical wire call has eight arguments and no inference bit.
        return _tensor, (_Storage(512), (3, 4), (8, 1), 5, torch.float32,
            torch.device("cpu"), True, self.kind)


@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize("kind", [torch.Tensor, torch.nn.Parameter])
def test_legacy_tensor_metadata_retains_loading_context_default(ambient, kind):
    legacy = LegacyTensorMetadata(kind)
    payload = pickle.dumps({"value": legacy, "again": legacy})
    with torch.inference_mode(ambient):
        restored = loads(payload)
        assert torch.is_inference_mode_enabled() == ambient
    value = restored["value"]
    assert value is restored["again"] and isinstance(value, FakeTensor)
    assert value.is_inference() == ambient and value.requires_grad
    assert isinstance(value, torch.nn.Parameter) == (kind is torch.nn.Parameter)
    assert (tuple(value.shape), value.stride(), value.storage_offset(), value.dtype,
            value.device, value.untyped_storage().nbytes()) == (
        (3, 4), (8, 1), 5, torch.float32, torch.device("cpu"), 512)


@pytest.mark.parametrize("ambient", [False, True])
def test_forward_metadata_preserves_mixed_inference_aliases_independently_of_ambient(ambient):
    original = mixed_tensors()
    assert not original.normal.is_inference() and original.inference.is_inference()
    with torch.inference_mode(ambient):
        restored = loads(dumps(original))
        assert torch.is_inference_mode_enabled() == ambient
    assert restored.again is restored.inference
    for name, source in vars(original).items():
        target = getattr(restored, name)
        assert isinstance(target, FakeTensor)
        assert (target.shape, target.stride(), target.storage_offset(), target.dtype,
                target.device, target.requires_grad, target.is_inference()) == (
            source.shape, source.stride(), source.storage_offset(), source.dtype,
            source.device, source.requires_grad, source.is_inference())
        assert isinstance(target, torch.nn.Parameter) == isinstance(source, torch.nn.Parameter)
    for left in vars(original):
        for right in vars(original):
            source_alias = (getattr(original, left).untyped_storage()._cdata
                == getattr(original, right).untyped_storage()._cdata)
            target_alias = (getattr(restored, left).untyped_storage()._cdata
                == getattr(restored, right).untyped_storage()._cdata)
            assert source_alias == target_alias


@pytest.mark.parametrize("ambient", [False, True])
def test_fresh_worker_keeps_mixed_tensor_identity(tmp_path, ambient):
    original = mixed_tensors()
    payload, expected = tmp_path / "forward.pkl", tmp_path / "expected.json"
    payload.write_bytes(dumps(original))
    expected.write_text(json.dumps({name: dict(inference=value.is_inference(),
        requires_grad=value.requires_grad, parameter=isinstance(value, torch.nn.Parameter),
        shape=list(value.shape), stride=list(value.stride()), offset=value.storage_offset())
        for name, value in vars(original).items()}))
    code = """
from pathlib import Path
import json,sys,torch
from ignition.preparation.forward_metadata import loads
torch.set_grad_enabled(False)
with torch.inference_mode(bool(int(sys.argv[3]))):
    value=loads(Path(sys.argv[1]).read_bytes())
    assert torch.is_inference_mode_enabled()==bool(int(sys.argv[3]))
    for name,expected in json.loads(Path(sys.argv[2]).read_text()).items():
        tensor=getattr(value,name)
        actual=dict(inference=tensor.is_inference(),requires_grad=tensor.requires_grad,
            parameter=isinstance(tensor,torch.nn.Parameter),shape=list(tensor.shape),
            stride=list(tensor.stride()),offset=tensor.storage_offset())
        assert actual==expected,(name,actual,expected)
    assert value.again is value.inference
    shared=value.storage.untyped_storage()._cdata
    assert all(getattr(value,name).untyped_storage()._cdata==shared
        for name in ('normal','inference','parameter','inference_parameter'))
    assert value.independent.untyped_storage()._cdata!=shared
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", code, str(payload), str(expected), str(int(ambient))],
        check=True, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize("guard_mode", [False, True])
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_guard_pickle_preserves_inference_and_dispatch_keys(ambient, guard_mode, inference, requires_grad):
    with torch.inference_mode(inference):
        source = torch.empty_strided((3, 4), (8, 1), requires_grad=requires_grad)
    original = loads(dumps(source))
    expected_keys = torch._C._dispatch_keys(original).raw_repr()
    output = io.BytesIO()
    _GuardPickler({id(original): original}, {}, {}, output).dump(original)
    with torch.inference_mode(ambient):
        token = _GUARD_TENSOR_MODE.set(FakeTensorMode() if guard_mode else None)
        try:
            restored = pickle.loads(output.getvalue())
            assert torch.is_inference_mode_enabled() == ambient
        finally:
            _GUARD_TENSOR_MODE.reset(token)
    assert restored.is_inference() == inference
    assert restored.requires_grad == requires_grad
    assert (restored.shape, restored.stride(), restored.dtype, restored.device) == (
        original.shape, original.stride(), original.dtype, original.device)
    assert torch._C._dispatch_keys(restored).raw_repr() == expected_keys


@pytest.mark.parametrize("inference,grad", [(False, False), (False, True), (True, False), (True, True)])
def test_compiler_job_carries_ambient_mode_separately_from_tensor_identity(
        tmp_path, monkeypatch, inference, grad):
    from torch.fx.experimental.proxy_tensor import make_fx
    from ignition.preparation import aot, compile_pool

    submitted, observed = [], []
    pool = object.__new__(compile_pool.CompilePool)
    pool.device, pool.options = torch.device("cuda:0"), {}
    pool.pool = SimpleNamespace(submit=lambda target, *args: submitted.append((target, args)))
    output = tmp_path / "context.bin"
    with torch.inference_mode(inference), torch.set_grad_enabled(grad):
        pool.submit_payload(dumps(mixed_tensors()), output)
    [(target, arguments)] = submitted
    job = inspect.signature(target).bind(*arguments).arguments
    assert job["inference_enabled"] == inference and job["grad_enabled"] == grad

    def record_context():
        observed.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))

    class Artifact:
        def __init__(self, result):
            self.result = result

        def save_compiled_function(self, path):
            record_context()
            Path(path).write_text(str(self.result))

    def trace_cpu_control(forward, *, options):
        record_context()
        assert not forward.normal.is_inference() and forward.inference.is_inference()
        # Exercise actual tracing of an ambient-mode branch. This is a CPU
        # context control; real CUDA compilation/binding is a separate gate.
        source = torch.ones(1)
        assert source.is_inference() == inference
        graph = make_fx(lambda value: value + 1 if torch.is_inference_mode_enabled()
            else value - 1)(source)
        return Artifact(graph(source).item())

    monkeypatch.setattr(aot, "aot_forward", trace_cpu_control)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    threads = torch.get_num_threads()
    precision = (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
    cuda_initialized = torch.cuda.is_initialized()
    try:
        # The reused worker begins in the opposite state; every submission
        # must set its own two flags and restore the worker afterwards.
        with torch.inference_mode(not inference), torch.set_grad_enabled(not grad):
            before = torch.is_inference_mode_enabled(), torch.is_grad_enabled()
            target(*arguments)
            assert (torch.is_inference_mode_enabled(), torch.is_grad_enabled()) == before
    finally:
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = precision
    assert observed == [(inference, grad)] * 2
    assert float(output.read_text()) == (2 if inference else 0)
    assert torch.cuda.is_initialized() == cuda_initialized
