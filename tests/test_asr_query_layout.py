# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CLI device selection controls only; these tests execute no GPU workload."""

import os
from unittest.mock import patch

import pytest

from ignition import planning
from ignition.cli import environment
from qualify import asr_query_layout


def devices():
    return tuple(planning.Gpu(i, "test", f"GPU-{i}", 1 << 30, 1 << 30, (12, 1))
        for i in range(3))


@pytest.mark.parametrize("visible", ["GPU-2", "2"])
def test_query_layout_uses_the_callers_single_assigned_device(monkeypatch, tmp_path, visible):
    monkeypatch.setattr(planning, "discover_gpus", devices)
    monkeypatch.setattr(environment.patches, "apply", lambda: ())
    config = tmp_path / "config.toml"
    config.write_text(f'[paths]\nkernel_cache = "{tmp_path}/kernels"\n')
    monkeypatch.setattr("sys.argv", ["asr_query_layout", "--config", str(config),
        "--output", str(tmp_path / "result")])
    called = []
    apply = environment.apply
    def record_apply(configuration, gpus):
        assert [(gpu.index, gpu.uuid) for gpu in gpus] == [(0, "GPU-2")]
        apply(configuration, gpus)
    monkeypatch.setattr(environment, "apply", record_apply)
    def run(output):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-2"
        called.append(output)
    monkeypatch.setattr(asr_query_layout, "run", run)
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}):
        asr_query_layout.main()
    assert called == [tmp_path / "result"]


@pytest.mark.parametrize("visible", [None, "", "0,2"])
def test_query_layout_rejects_zero_or_multiple_devices_before_apply(monkeypatch, tmp_path, visible):
    monkeypatch.setattr(planning, "discover_gpus", devices)
    monkeypatch.setattr("sys.argv", ["asr_query_layout", "--config", str(tmp_path / "unused.toml"),
        "--output", str(tmp_path / "result")])
    def unexpected(*args):
        pytest.fail("an unassigned GPU reached dependency setup or inference")
    monkeypatch.setattr(environment, "apply", unexpected)
    monkeypatch.setattr(asr_query_layout, "run", unexpected)
    with patch.dict(os.environ):
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
        with pytest.raises(SystemExit) as error:
            asr_query_layout.main()
        assert error.value.code == 2
