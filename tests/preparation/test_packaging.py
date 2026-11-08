# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU preparation package and command-line smoke checks."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tomllib
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_pyproject_declares_preparation_entrypoint_and_packages():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    includes = set(pyproject["tool"]["setuptools"]["packages"]["find"]["include"])

    assert pyproject["project"]["name"] == "ignition"
    assert "ignition*" in includes
    assert not any(name.startswith(("flare", "serve")) for name in includes)
    assert pyproject["project"]["dependencies"] == []
    assert pyproject["project"]["scripts"] == {"ignition": "ignition.__main__:main"}


def test_package_import_has_no_runtime_or_cuda_side_effects():
    result = subprocess.run(
        [sys.executable, "-c", "import sys, ignition; "
         "assert ignition.__all__ == ['__version__']; "
         "assert 'torch' not in sys.modules; "
         "assert 'flare_heliopause' not in sys.modules; "
         "assert not any(n == 'serve' or n.startswith('serve.') for n in sys.modules)"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (ROOT / "serve").exists()


def test_server_cli_help_does_not_prepare_model():
    result = subprocess.run(
        [sys.executable, "-m", "ignition", "--help"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    for command in ("export", "debug-serve", "bench"):
        assert f"\n    {command} " in result.stdout
    for command in ("serve", "run", "prepare", "plan", "models", "pull", "status", "config", "skill"):
        assert f"\n    {command} " not in result.stdout


def test_built_wheel_keeps_asr_template_asset(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    shutil.copytree(ROOT / "ignition", source / "ignition",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "LICENSE", "NOTICE", "helios/licenses/PyTorch.txt"):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    built = subprocess.run([sys.executable, "-I", "-c",
        "import sys; from setuptools import build_meta; build_meta.build_wheel(sys.argv[1])",
        str(wheels)], cwd=source, capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout + built.stderr
    wheel, = wheels.glob("*.whl")
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
    result = subprocess.run([sys.executable, "-I", "-c", """
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import ignition
from ignition.preparation.frontend import annotated_template_asset
root = Path(ignition.__file__).resolve().parent
assert root.is_relative_to(Path(sys.argv[1]))
patch = root / 'recipes/qwen3_asr/assets/chat_template.patch.json'
assert patch.read_bytes() == Path(sys.argv[2]).read_bytes()
try:
    annotated_template_asset(b'wrong checkpoint template', patch)
except ValueError as error:
    assert str(error).startswith('checkpoint chat template does not match ')
else:
    raise AssertionError('installed template must retain its checkpoint identity check')
assert 'torch' not in sys.modules and 'flare_heliopause' not in sys.modules
""", str(installed), str(ROOT / "ignition/recipes/qwen3_asr/assets/chat_template.patch.json")],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
