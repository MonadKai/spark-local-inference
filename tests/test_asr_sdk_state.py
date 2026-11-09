# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU controls for original SDK position observation, without a model forward."""

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from qualify.asr_sdk_stages import snapshot_sdk_positions


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU SDK position control attempted CUDA initialization")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


@pytest.mark.parametrize("consumed,tokens", [(31, 31), (32, 1), (33, 1)])
def test_sdk_actual_fp32_three_axis_positions_keep_bytes_and_own_storage(consumed, tokens):
    positions = torch.arange(consumed - tokens, consumed, dtype=torch.float32)
    source = positions.view(1, 1, tokens).expand(3, 1, tokens)
    actual = snapshot_sdk_positions(source, consumed, tokens)
    assert actual.dtype == torch.float32 and actual.shape == (3, 1, tokens)
    assert actual.is_contiguous() and actual.data_ptr() != source.data_ptr()
    assert torch.equal(actual.view(torch.uint8), source.contiguous().view(torch.uint8))
    positions.zero_()
    assert actual[0, 0, -1].item() == consumed - 1


@pytest.mark.parametrize("defect", ["integer-cast", "axis", "step", "fraction", "nan", "infinity"])
def test_corrupt_or_reinterpreted_sdk_positions_cannot_be_published(defect):
    value = torch.full((3, 1, 1), 31., dtype=torch.float32)
    if defect == "integer-cast": value = value.long()
    if defect == "axis": value = value[:2]
    if defect == "step": value[2, 0, 0] = 30.
    if defect == "fraction": value[0, 0, 0] = 31.5
    if defect == "nan": value[0, 0, 0] = float("nan")
    if defect == "infinity": value[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="SDK rotary"):
        snapshot_sdk_positions(value, 32, 1)


@pytest.mark.parametrize("option", ["--cached-state", "--frozen-sdk"])
def test_state_collection_requires_its_matching_frozen_sdk_before_imports(tmp_path, option):
    script = Path(__file__).resolve().parents[1] / "qualify/asr_sdk_stages.py"
    output = tmp_path / "observations"
    command = [sys.executable, "-I", str(script)]
    for name in ("checkpoint", "sdk-source", "manifest", "pcm-root"):
        command += ["--" + name, str(tmp_path)]
    command += ["--output", str(output), option]
    if option == "--frozen-sdk":
        command.append(str(tmp_path))
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "cached-state and frozen-sdk must be supplied together" in result.stderr
    assert not output.exists()
