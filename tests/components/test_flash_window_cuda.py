# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical flash windows retain empty/tail semantics under compiled replay."""

import os

import pytest
import torch

from ignition.components.flash_window import PreparedFlashWindow
from ignition.preparation.compilation import compile_forward
from ignition.preparation.failure import retain_on_failed_drain
from ignition.preparation.module_images import ModuleImages

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


@pytest.mark.parametrize("capacity", [13, 104])
@torch.no_grad()
def test_flash_window_compiled_empty_poison_and_repeated_valid_rows(capacity):
    with ModuleImages() as images:
        q, k, v, output = [torch.zeros((capacity, 16, 64), device="cuda", dtype=torch.bfloat16) for _ in range(4)]
        cu = torch.tensor([0, capacity], device="cuda", dtype=torch.int32)
        operation = PreparedFlashWindow(q, k, v, cu, output)
        owners, graphs = [operation, images], []
        with retain_on_failed_drain(0, owners, graphs):
            compiled = compile_forward(operation)
            owners.append(compiled)
            compiled()
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with torch.cuda.graph(graph):
                compiled()
            generator = torch.Generator(device="cuda").manual_seed(20261106)
            for count in (0, 1, 2, capacity - 1, capacity, 2, 0):
                cu[1] = count
                for value in (q, k, v):
                    value.fill_(float("nan"))
                    value[:count].normal_(std=.2, generator=generator)
                golden = None
                for _ in range(5):
                    output.fill_(float("nan"))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                        assert before[key] == after[key], key
                    actual = output.cpu().clone()
                    assert torch.isfinite(actual).all() and actual[count:].eq(0).all()
                    if golden is not None:
                        assert torch.equal(actual.view(torch.uint8), golden.view(torch.uint8))
                    golden = actual
                if count:
                    reference = torch.nn.functional.scaled_dot_product_attention(
                        q[:count].transpose(0, 1)[None], k[:count].transpose(0, 1)[None],
                        v[:count].transpose(0, 1)[None])[0].transpose(0, 1).cpu().contiguous()
                    assert torch.equal(actual[:count].reshape(-1).view(torch.uint8), reference.reshape(-1).view(torch.uint8))
                if 0 < count < capacity:
                    # A bounded but wrong live length exposes poisoned keys;
                    # prove the actual finite gate detects this failure.
                    cu[1] = capacity
                    graph.replay()
                    torch.cuda.synchronize()
                    assert not torch.isfinite(output[:count]).all()
                    cu[1] = count
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(output.cpu().view(torch.uint8), actual.view(torch.uint8))
            graph.reset()


def test_flash_window_rejects_output_alias_and_host_lengths():
    q = torch.zeros((13, 16, 64), device="cuda", dtype=torch.bfloat16)
    cu = torch.zeros(2, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="alias"):
        PreparedFlashWindow(q, q, q, cu, q)
    with pytest.raises(ValueError, match="CUDA"):
        PreparedFlashWindow(q, q, q, cu.cpu(), torch.empty_like(q))


@pytest.mark.parametrize("capacity,count", [(13, 0), (104, 103)])
def test_flash_window_package_replays_without_python_or_rng(tmp_path, capacity, count):
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    subprocess.run(["cargo", "build", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools", "--bin", "test_fixed_runtime_package"],
        cwd=root, check=True, timeout=300)
    package = tmp_path / "flash-window"
    preparation = subprocess.run([sys.executable, "-m", "helios.tests.flash_window_package_fixture", str(package),
        "--capacity", str(capacity), "--count", str(count)], cwd=root, text=True, capture_output=True, timeout=900)
    assert preparation.returncode == 0, preparation.stdout + preparation.stderr
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(root / "helios/target")))
    replay = subprocess.run([str(target / "debug/test_fixed_runtime_package"), str(package), "--replays", "5"],
        cwd=root, text=True, capture_output=True, timeout=180)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert "native fixed-address replay passed" in replay.stdout


@pytest.mark.parametrize("injection", ["replay", "storage_exit", "recorder_exit"])
def test_package_fixture_retains_post_capture_owners_after_failed_fence(tmp_path, injection):
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    script = r'''
import contextlib,gc,sys,weakref
from pathlib import Path
import torch
sys.path.insert(0,str(Path.cwd()/"helios/tests"))
from helios.tests import flash_window_package_fixture as fixture
import ignition.preparation.capture as capture
import ignition.preparation.package_storage as storage
import ignition.preparation.module_images as images
refs={}
real_sync=torch.cuda.synchronize
original_capture=capture.capture_forwards
original_init=storage.PackageStorage.__init__
def tracked_init(owner,*args,**kwargs):
    original_init(owner,*args,**kwargs)
    refs["storage"]=weakref.ref(owner)
storage.PackageStorage.__init__=tracked_init
def tracked_capture(functions,*args,**kwargs):
    graphs,compiled=original_capture(functions,*args,**kwargs)
    refs.update(graph=weakref.ref(graphs["decode"]),compiled=weakref.ref(compiled["decode"]),
        forward=weakref.ref(functions["decode"]))
    return graphs,compiled
capture.capture_forwards=tracked_capture
def failed_sync(*args,**kwargs):raise RuntimeError("injected unconfirmed fixture fence")
def inject(graph,output,expected):
    graph.replay()
    torch.cuda.synchronize=failed_sync
    raise RuntimeError("injected failure after real captured replay")
if sys.argv[3]=="replay":
    fixture.verify_replay=inject
else:
    target=storage.PackageStorage if sys.argv[3]=="storage_exit" else images.ModuleImages
    original_exit=target.__exit__
    def failed_exit(owner,*args):
        original_exit(owner,*args)
        refs["graph"]().replay()
        torch.cuda.synchronize=failed_sync
        raise RuntimeError("injected failure after real captured replay")
    target.__exit__=failed_exit
if sys.argv[2]=="disabled":
    fixture.retain_on_failed_drain=lambda *owners: contextlib.nullcontext()
path=Path(sys.argv[1]);path.mkdir()
try:
    fixture.prepare(path,13,12)
except RuntimeError as error:
    assert str(error)=="injected failure after real captured replay",str(error)
    error.__traceback__=None;error.__cause__=None;error.__context__=None;error.__dict__.clear()
else:raise AssertionError("fixture failure was not injected")
# Safely finish the negative control's real GPU work before collecting objects.
# A confirmed later fence does not release terminal native OwnerLease roots.
real_sync()
gc.collect()
assert set(refs)=={"storage","graph","compiled","forward"}
assert all(reference() is not None for reference in refs.values()),"post-capture fixture lost undrained owners"
assert refs["forward"]().output[:12].eq(2).all()
print("post-capture fixture owners survived exception disposal",flush=True)
'''
    for control in ("enabled", "disabled"):
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path / control), control, injection],
            cwd=root, text=True, capture_output=True, timeout=900)
        if control == "enabled":
            assert result.returncode == 0, result.stdout + result.stderr
            assert "owners survived exception disposal" in result.stdout
        else:
            assert result.returncode != 0
            assert "post-capture fixture lost undrained owners" in result.stdout + result.stderr
