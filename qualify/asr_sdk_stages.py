# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Record original SDK GPU stages before evaluating the candidate model.

This independent development-set probe does not admit cross-engine tolerances,
recognition quality, native package execution, or comparative performance.
"""

import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sdk-source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pcm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-tokens", type=int, default=16,
                        help="diagnostic prefix length; zero records the complete generated sequence")
    parser.add_argument("--boundary-frames", nargs="+", type=int,
                        help="instead probe these prefix lengths of the longest development clip per language")
    args = parser.parse_args()
    if args.teacher_tokens < 0:
        parser.error("teacher token count must be nonnegative")
    if args.boundary_frames and (len(set(args.boundary_frames)) != len(args.boundary_frames)
            or any(not 2 <= value <= 8000 for value in args.boundary_frames)):
        parser.error("boundary frame counts must be unique and in [2,8000]")
    import numpy as np
    import torch
    import transformers
    import flash_attn
    import qwen_asr
    from qwen_asr.core.transformers_backend import configuration_qwen3_asr as configuration
    from qwen_asr.core.transformers_backend import modeling_qwen3_asr as modeling
    from safetensors.torch import save_file

    if (torch.__version__, transformers.__version__, flash_attn.__version__) != (
            "2.13.0+cu130", "4.57.6", "2.8.3"):
        raise ValueError("independent SDK environment changed")
    if any(importlib.util.find_spec(name) is not None for name in ("ignition", "flare_heliopause")):
        raise ValueError("independent SDK environment contains candidate packages")
    source, installed = args.sdk_source / "qwen_asr", Path(qwen_asr.__file__).parent
    files = {path.relative_to(source) for path in source.rglob("*.py")}
    if files != {path.relative_to(installed) for path in installed.rglob("*.py")}:
        raise ValueError("SDK installed source file set differs")
    for name in files:
        if (source / name).read_bytes() != (installed / name).read_bytes():
            raise ValueError(f"SDK installed source differs: {name}")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("reference requires one assigned SM121 GPU")
    manifest = json.loads(args.manifest.read_bytes())
    if manifest.get("split") != "development":
        raise ValueError("stage calibration requires the independent development set")
    selected = []
    for language in ("en", "zh"):
        rows = sorted((row for row in manifest["utterances"] if row["language"] == language),
                      key=lambda row: (row["samples"], row["id"]))
        if args.boundary_frames:
            if rows[-1]["samples"] < max(args.boundary_frames) * 160:
                raise ValueError("longest development clip is shorter than a requested boundary")
            selected.extend(dict(rows[-1], boundary_frames=frames) for frames in args.boundary_frames)
        else:
            selected.extend((rows[0], rows[-1]))
    # Establish FP32 projection reductions before the first checkpoint forward.
    # This is an explicit reference numerical setting, not an SDK source patch.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    args.output.mkdir(exist_ok=False)
    record = dict(scope=__doc__.strip(), sdk_revision="7c6daf77a2421100f5fb066495372c00129d39ff",
        checkpoint_revision="7278e1e70fe206f11671096ffdd38061171dd6e5",
        input_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        selection="prefixes of longest (samples,id) per language" if args.boundary_frames else
            "first and last (samples,id) per en/zh development language, fixed before inference",
        boundary_frames=args.boundary_frames,
        preprocessing="canonical PCM16/32768, explicit zero-pad to hop160, original SDK processor",
        prompt="original SDK raw audio-chat prompt, empty system context, forced English/Chinese",
        long_input_scope="whole-audio feature/model profile; transcription service chunking is a separate check",
        torch=torch.__version__, transformers=transformers.__version__, flash_attention=flash_attn.__version__,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        cases=[], errors=[], candidate_executed=False, cross_engine_tolerances_admitted=False)
    result_path = args.output / "result.json"

    def publish():
        result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    publish()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    config = configuration.Qwen3ASRConfig.from_pretrained(args.checkpoint, local_files_only=True)
    # Preserve the CPU-constructed FP32 nonpersistent constants already checked
    # by the independent loading probe; migrate devices without a second cast.
    model, loading = modeling.Qwen3ASRForConditionalGeneration.from_pretrained(args.checkpoint,
        config=config, dtype=torch.bfloat16, device_map={"": "cpu"}, attn_implementation="flash_attention_2",
        local_files_only=True, trust_remote_code=False, output_loading_info=True)
    if any(loading.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"original SDK loading mismatch: {loading}")
    constants = {"audio_position": model.thinker.audio_tower.positional_embedding.positional_embedding.clone(),
        "rotary_inv_freq": model.thinker.model.rotary_emb.inv_freq.clone(),
        "rotary_original_inv_freq": model.thinker.model.rotary_emb.original_inv_freq.clone()}
    model.eval().to("cuda:0")
    actual_constants = {"audio_position": model.thinker.audio_tower.positional_embedding.positional_embedding,
        "rotary_inv_freq": model.thinker.model.rotary_emb.inv_freq,
        "rotary_original_inv_freq": model.thinker.model.rotary_emb.original_inv_freq}
    assert all(value.dtype == torch.bfloat16 for value in model.parameters())
    for name, value in actual_constants.items():
        assert value.dtype == torch.float32 and torch.equal(value.cpu(), constants[name])
    assert model.thinker.audio_tower.config._attn_implementation == "flash_attention_2"
    assert model.thinker.model.config._attn_implementation == "flash_attention_2"
    processor = transformers.AutoProcessor.from_pretrained(args.checkpoint,
        local_files_only=True, fix_mistral_regex=True)
    wrapper = qwen_asr.Qwen3ASRModel("transformers", model, processor,
        max_inference_batch_size=1, max_new_tokens=1024)
    save_file(constants, args.output / "fp32-constants.safetensors")

    def cpu(value):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("reference produced nonfinite tensor")
        return value.detach().to(device="cpu", copy=True).contiguous()

    def same_bytes(left, right):
        return left.dtype == right.dtype and left.shape == right.shape \
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))

    class Stages:
        def __init__(self):
            self.values, self.handles = {}, []

        def store(self, name, value):
            if isinstance(value, torch.Tensor):
                if name in self.values:
                    raise ValueError(f"stage invoked more than once in a prefill: {name}")
                self.values[name] = cpu(value)
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    self.store(f"{name}.{index}", item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, torch.Tensor):
                        self.store(f"{name}.{key}", item)

        def attach(self):
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm, torch.nn.SiLU)) \
                        or name.endswith(("q_norm", "k_norm", "input_layernorm", "post_attention_layernorm",
                                          "rotary_emb", "model.norm")) \
                        or name in ("thinker.audio_tower.layers.0", "thinker.model"):
                    def before(module, positional, keywords, name=name):
                        self.store(name + ".input", positional)
                        self.store(name + ".kwargs", keywords)
                    self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm, torch.nn.SiLU)) \
                        or name.endswith(("q_norm", "k_norm", "input_layernorm", "post_attention_layernorm",
                                          "rotary_emb", "model.norm")) \
                        or name.rsplit(".", 1)[-1].isdigit() and ".layers." in name:
                    self.handles.append(module.register_forward_hook(
                        lambda module, positional, output, name=name: self.store(name + ".output", output)))

        def detach(self):
            for handle in self.handles:
                handle.remove()
            self.handles.clear()

    with torch.inference_mode():
        for row in selected:
            case_id = row["id"] + (f"-first-{row['boundary_frames']}f" if "boundary_frames" in row else "")
            case = dict(id=case_id, source_id=row["id"], language=row["language"], original_samples=row["samples"],
                        teacher_token_limit=args.teacher_tokens)
            record["cases"].append(case)
            try:
                with wave.open(str(args.pcm_root / row["path"]), "rb") as source:
                    assert (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()) \
                        == (1, 2, 16000, row["samples"])
                    pcm = source.readframes(source.getnframes())
                assert hashlib.sha256(pcm).hexdigest() == row["pcm_s16le_sha256"]
                if "boundary_frames" in row:
                    pcm = pcm[:row["boundary_frames"] * 160 * 2]
                    case.update(original_samples=len(pcm) // 2, prefix_pcm_s16le_sha256=hashlib.sha256(pcm).hexdigest())
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768)
                samples = np.pad(samples, (0, (-len(samples)) % 160))
                prompt = wrapper._build_text_prompt("", {"en": "English", "zh": "Chinese"}[row["language"]])
                inputs = processor(text=[prompt], audio=[samples], sampling_rate=16000,
                    return_tensors="pt", padding=True).to(model.device).to(model.dtype)
                frames = len(samples) // 160
                assert inputs["input_features"].shape[-1] == frames
                assert inputs["feature_attention_mask"].sum().item() == frames
                assert inputs["input_ids"].eq(151676).sum().item() == 13 * (frames // 100) + (frames % 100 + 7) // 8
                p = inputs["input_ids"].shape[1]
                case.update(padded_samples=len(samples), frames=frames, prompt=prompt,
                    prompt_ids=inputs["input_ids"][0].tolist())
                model.thinker.rope_deltas = None
                warmup = model.thinker(**inputs, use_cache=False)
                del warmup
                stages = Stages()
                stages.attach()
                try:
                    model.thinker.rope_deltas = None
                    measured = model.thinker(**inputs, use_cache=False)
                finally:
                    stages.detach()
                assert measured.logits.dtype == torch.bfloat16
                reference_logits = cpu(measured.logits)
                del measured
                required = {f"thinker.audio_tower.layers.{index}.output.0" for index in range(24)} \
                    | {f"thinker.model.layers.{index}.output" for index in range(28)} \
                    | {"thinker.lm_head.output", "thinker.audio_tower.proj2.output",
                       "thinker.model.kwargs.inputs_embeds", "thinker.model.norm.output"}
                assert required <= stages.values.keys(), "a required stage observer did not run"
                assert stages.values["thinker.lm_head.output"].dtype == torch.bfloat16
                # An unobserved call independently checks that observation did
                # not alter the final arithmetic or logits.
                model.thinker.rope_deltas = None
                unobserved = model.thinker(**inputs, use_cache=False)
                assert same_bytes(cpu(unobserved.logits), reference_logits)
                del unobserved
                for _ in range(5):
                    repeated_stages = Stages()
                    repeated_stages.attach()
                    model.thinker.rope_deltas = None
                    try:
                        repeated = model.thinker(**inputs, use_cache=False)
                    finally:
                        repeated_stages.detach()
                    assert same_bytes(cpu(repeated.logits), reference_logits), "same-path prefill logits changed"
                    assert repeated_stages.values.keys() == stages.values.keys()
                    for name, value in stages.values.items():
                        assert same_bytes(repeated_stages.values[name], value), f"same-path stage changed: {name}"
                    del repeated, repeated_stages
                stages.values["processor.input_features"] = cpu(inputs["input_features"])
                save_file(stages.values, args.output / (case_id + "-prefill.safetensors"))
                case["stages"] = {name: dict(shape=list(value.shape), dtype=str(value.dtype))
                                  for name, value in stages.values.items()}
                del stages, reference_logits
                sequences = []
                for _ in range(2):
                    model.thinker.rope_deltas = None
                    generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                        num_beams=1, use_cache=True, eos_token_id=[151643, 151645], pad_token_id=151643)
                    sequences.append(cpu(generated.sequences[:, p:]))
                    del generated
                assert same_bytes(*sequences), "same-path greedy IDs changed"
                y = sequences[0]
                assert y.numel() > 0
                case.update(generated_ids=y[0].tolist(), generated_tokens=y.shape[1],
                    natural_stop=int(y[0, -1]) in (151643, 151645),
                    raw_text=processor.batch_decode(y, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False)[0],
                    prefill_same_path_repetitions=5, natural_generation_repetitions=2)
                assert case["natural_stop"], "reference natural generation reached its output cap"
                t = y.shape[1] if args.teacher_tokens == 0 else min(args.teacher_tokens, y.shape[1])
                teacher, reference = y[:, :t].to(model.device), None
                for repetition in range(5):
                    model.thinker.rope_deltas = None
                    result = model.thinker(**inputs, use_cache=True, cache_position=torch.arange(p, device=model.device))
                    rows = [cpu(result.logits[:, -1])]
                    for index in range(1, t):
                        result = model.thinker(input_ids=teacher[:, index-1:index],
                            attention_mask=torch.cat((inputs["attention_mask"], torch.ones((1, index),
                                device=model.device, dtype=inputs["attention_mask"].dtype)), dim=1),
                            past_key_values=result.past_key_values, input_features=None,
                            cache_position=torch.tensor([p + index - 1], device=model.device), use_cache=True)
                        assert result.past_key_values.get_seq_length() == p + index
                        rows.append(cpu(result.logits[:, -1]))
                    values = torch.cat(rows, dim=0)
                    if repetition == 0:
                        reference = values
                    else:
                        assert same_bytes(reference, values), "same-path cached teacher logits changed"
                    del result, rows
                save_file({"logits_bf16": reference, "target_ids": teacher.cpu()},
                    args.output / (case_id + "-cached-teacher.safetensors"))
                case.update(teacher_rows=t, cached_teacher_same_path_repetitions=5, status="repeatable_controls_only")
            except Exception as error:
                case.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                record["errors"].append(case_id)
            publish()
    if record["errors"]:
        raise RuntimeError("independent reference stage controls failed; inspect retained cases")


if __name__ == "__main__":
    main()
