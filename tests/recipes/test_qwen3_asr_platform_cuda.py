# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM121 stack probes; not model, cross-path numerical or serving acceptance."""

import os

import pytest
import torch

from helios.tests.conv_package_fixture import ConvolutionNorm

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.fixture(autouse=True)
def assigned_device():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign the validation GPU explicitly"


class AttentionForward:
    def __init__(self, attention, output):
        self.attention, self.output = attention, output

    def __call__(self):
        self.output.zero_()
        self.attention()


@pytest.mark.parametrize("rows", [13, 104])
@torch.no_grad()
def test_prepared_audio_attention_handles_partial_and_empty_windows(rows):
    from ignition.components.varlen_attention import PreparedVarlenAttention
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    q = torch.zeros((rows, 16, 64), device=device, dtype=torch.bfloat16)
    k, v, output = torch.zeros_like(q), torch.ones_like(q), torch.empty_like(q)
    cu = torch.tensor([0, rows], device=device, dtype=torch.int32)
    with RecordedPreparationSession(device=device, checkpoint="asr-platform-analytic-window",
            workload_schema="varlen-16x64", numerical_policy="default", compile_workers=1) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedVarlenAttention(q, k, v, cu, output, session=session, workspace=arena,
            name="audio-window", max_seqlen=rows)
        session.tune()
        forward = compile_forward(AttentionForward(attention, output))
        forward()
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                forward()
            session.freeze()
            for count in (rows, rows - 1, 1, 0, rows):
                q.zero_()
                k.zero_()
                v.fill_(2)
                for value in (q, k, v):
                    value[count:].fill_(float("nan"))
                cu[1] = count
                expected = torch.zeros_like(output)
                expected[:count].fill_(2)
                output.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                # Uniform zero-QK attention on constant V has an exact BF16
                # answer. Poisoned inactive storage must not enter that answer.
                assert torch.isfinite(output).all()
                assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
        finally:
            graph.reset()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("chunks,width", [(1, 1), (1, 99), (1, 100), (2, 100)])
@torch.no_grad()
def test_convolution_layernorm_capture_and_replay(chunks, width, compiled):
    # This gate compares each path to itself before capture. Canonical stage
    # tolerances and complete checkpoint quality are separate, later gates.
    torch.manual_seed(20261106)
    model = ConvolutionNorm().eval().to(device="cuda:0", dtype=torch.bfloat16)
    source = torch.randn((chunks, 1, 128, width), device="cuda:0", dtype=torch.bfloat16) * 0.1
    output = torch.empty((chunks, (width + 7) // 8, 1024), device="cuda:0", dtype=torch.bfloat16)
    with torch._inductor.config.patch(emulate_precision_casts=True):
        forward = torch.compile(model, fullgraph=True, dynamic=False) if compiled else model
        stream = torch.cuda.Stream(device=source.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                output.copy_(forward(source))
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        reference = output.clone()
        assert torch.isfinite(reference).all()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                output.copy_(forward(source))
            for _ in range(5):
                output.fill_(float("nan"))
                before = torch.cuda.memory_stats(source.device)
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats(source.device)
                assert torch.equal(output.view(torch.uint8), reference.view(torch.uint8))
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
        finally:
            graph.reset()


@pytest.mark.parametrize("chunks", [1, 8])
def test_convolution_package_replays_in_rust_after_python_exit(tmp_path, chunks):
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    subprocess.run(["cargo", "build", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools", "--bin", "test_fixed_runtime_package"],
        cwd=root, check=True, timeout=300)
    package = tmp_path / "convolution"
    preparation = subprocess.run([sys.executable, "-m", "helios.tests.conv_package_fixture",
        str(package), "--chunks", str(chunks)], cwd=root, text=True, capture_output=True, timeout=900)
    assert preparation.returncode == 0, preparation.stdout + preparation.stderr
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(root / "helios/target")))
    replay = subprocess.run([str(target / "debug/test_fixed_runtime_package"), str(package), "--replays", "5"],
        cwd=root, text=True, capture_output=True, timeout=180)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert "native fixed-address replay passed" in replay.stdout
