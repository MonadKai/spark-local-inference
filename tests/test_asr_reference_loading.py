# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Real CPU reference files and context preservation; no model acceptance."""

import pytest
import torch
from safetensors import SafetensorError
from safetensors.torch import save_file

from qualify.asr_candidate_teacher import load_reference_tensors, preparation_context


@pytest.fixture
def reference_files(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("reference loading must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    sdk, canonical = tmp_path / "sdk", tmp_path / "canonical"
    sdk.mkdir(); canonical.mkdir()
    mel = torch.arange(384).reshape(1, 128, 3).to(torch.bfloat16)
    convolution = mel.unsqueeze(1).clone()
    sdk_data = {"processor.input_features": mel,
        "thinker.audio_tower.conv2d1.input.0": convolution}
    canonical_data = {"thinker.audio_tower.conv2d1.input.0": convolution.clone()}
    names = ["thinker.audio_tower.conv_out.output", "thinker.audio_tower.proj1.output",
        "thinker.audio_tower.proj2.output"]
    for layer in range(24):
        prefix = f"thinker.audio_tower.layers.{layer}"
        names += [prefix + f".self_attn.{field}.output" for field in ("q_proj", "k_proj", "v_proj", "out_proj")]
        names += [prefix + ".self_attn.out_proj.input.0", prefix + ".fc1.output", prefix + ".fc2.output"]
    for index, name in enumerate(names):
        canonical_data[name] = torch.tensor([index / 4], dtype=torch.bfloat16)
    logits = torch.zeros((1, 151936), dtype=torch.bfloat16)
    logits[0, 2] = 5
    teacher_data = dict(logits_bf16=logits, target_ids=torch.tensor([[2]]))
    case = dict(id="input", frames=3, teacher_rows=[{}], teacher_history=[2], teacher_top1=[2])

    def write():
        save_file(sdk_data, sdk / "input-prefill.safetensors")
        save_file(canonical_data, canonical / "input-prefill.safetensors")
        save_file(teacher_data, canonical / "input-cached-teacher.safetensors")
    write()
    return [(case, sdk, canonical)], sdk_data, canonical_data, teacher_data, write


@pytest.mark.parametrize("preparation_mode", ["inference", "no-grad"])
@pytest.mark.parametrize("reference_inference", [False, True])
@pytest.mark.parametrize("reference_grad", [False, True])
def test_reference_loading_preserves_bytes_and_both_contexts(reference_files,
        preparation_mode, reference_inference, reference_grad):
    references, sdk, canonical, teacher, _ = reference_files
    rng = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    with preparation_context(preparation_mode):
        ambient = torch.is_inference_mode_enabled(), torch.is_grad_enabled()
        inputs, expected, encoder, stages = load_reference_tensors(references,
            inference_mode=reference_inference, grad_enabled=reference_grad)
        assert (torch.is_inference_mode_enabled(), torch.is_grad_enabled()) == ambient
        assert inputs.keys() == expected.keys() == encoder.keys() == stages.keys() == {"input"}
        assert inputs["input"].is_inference() == reference_inference
        assert inputs["input"].dtype == torch.float32
        assert torch.equal(inputs["input"], sdk["processor.input_features"][0].float())
        assert torch.equal(expected["input"].view(torch.uint8), teacher["logits_bf16"].view(torch.uint8))
        assert torch.equal(encoder["input"].view(torch.uint8), canonical["thinker.audio_tower.proj2.output"].view(torch.uint8))
        assert stages["input"].keys() == canonical.keys() - {"thinker.audio_tower.conv2d1.input.0"}
        for name, value in stages["input"].items():
            assert torch.equal(value.view(torch.uint8), canonical[name].view(torch.uint8))
    assert torch.get_num_threads() == threads
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("defect", ["mel-dtype", "mel-shape", "mel-nonfinite", "features",
    "teacher-shape", "teacher-dtype", "teacher-nonfinite", "targets", "top1", "missing-stage"])
def test_bad_reference_files_still_fail_during_loading(reference_files, defect):
    references, sdk, canonical, teacher, write = reference_files
    if defect == "mel-dtype": sdk["processor.input_features"] = sdk["processor.input_features"].float()
    if defect == "mel-shape": sdk["processor.input_features"] = sdk["processor.input_features"][:, :, :2].contiguous()
    if defect == "mel-nonfinite": sdk["processor.input_features"][0, 0, 0] = float("nan")
    if defect == "features": canonical["thinker.audio_tower.conv2d1.input.0"][0, 0, 0, 0] = 1
    if defect == "teacher-shape": teacher["logits_bf16"] = teacher["logits_bf16"][:, :5].contiguous()
    if defect == "teacher-dtype": teacher["logits_bf16"] = teacher["logits_bf16"].float()
    if defect == "teacher-nonfinite": teacher["logits_bf16"][0, 0] = float("inf")
    if defect == "targets": teacher["target_ids"][0, 0] = 3
    if defect == "top1": teacher["logits_bf16"][0, 3] = 6
    if defect == "missing-stage": del canonical["thinker.audio_tower.layers.23.fc2.output"]
    write()
    with preparation_context("inference"):
        with pytest.raises((ValueError, RuntimeError, SafetensorError)):
            load_reference_tensors(references, inference_mode=False, grad_enabled=True)
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
