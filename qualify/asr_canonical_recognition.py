# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate frozen recognition hypotheses with isolated canonical Transformers.

Use Python -I in the unpatched reference environment. Input processing comes
from the pinned real vLLM frontend, independently of the candidate Rust path.
The original short-width hook is the same layout adaptation already qualified
by the independent SDK stage probes; no arithmetic forward is replaced.
"""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    spec = importlib.util.spec_from_file_location("independent_loading",
        Path(__file__).with_name("asr_canonical_reference.py"))
    loading = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loading)
    manifest = json.loads(args.manifest.read_bytes())
    inputs = json.loads((args.inputs / "result.json").read_bytes())
    if (manifest["split"] != "qualification" or len(manifest["utterances"]) != 256
            or inputs["manifest_sha256"] != hashlib.sha256(args.manifest.read_bytes()).hexdigest()
            or inputs["errors"] or inputs["status"] != "inputs_prepared_no_model_execution"
            or inputs["vllm_revision"] != "98dff2a81d747d1dba01a47f939f48c3526d4206"
            or inputs["checkpoint_revision"] != "7278e1e70fe206f11671096ffdd38061171dd6e5"
            or manifest["request"] != dict(temperature=0, seed=0, max_model_len=8192,
                context="", language="utterance.language", max_completion_tokens=1024)
            or inputs["request"] != manifest["request"]
            or inputs["model_executed"] or inputs["cuda_initialized"]
            or [c["id"] for c in inputs["cases"]] != [c["id"] for c in manifest["utterances"]]):
        raise ValueError("complete independent frozen qualification inputs required")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if torch.__version__ != "2.14.1+cu132" or torch.cuda.device_count() != 1 \
            or torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("pinned canonical SM121 stack required")
    model, index, renamed = loading.load_reference(args.checkpoint)
    with ExitStack() as owners:
        shards = {name: owners.enter_context(safe_open(args.checkpoint / name, framework="pt"))
            for name in sorted(set(index.values()))}
        actual = model.state_dict()
        if actual.keys() != renamed.keys():
            raise ValueError("canonical weight coverage differs")
        for target, original in renamed.items():
            source, destination = shards[index[original]].get_tensor(original), actual[target]
            if source.dtype != destination.dtype or not torch.isfinite(source).all() \
                    or not torch.equal(source.view(torch.uint8), destination.view(torch.uint8)):
                raise ValueError("canonical weight bytes differ: " + original)
    model.set_attn_implementation("sdpa")
    if (model.model.audio_tower.positional_embedding.positional_embedding.dtype != torch.float32
            or model.model.language_model.rotary_emb.inv_freq.dtype != torch.float32):
        raise ValueError("canonical positional buffers lost FP32 precision")
    model.to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False,
        local_files_only=True, fix_mistral_regex=True)
    args.output.mkdir(exist_ok=False)
    result = dict(scope=__doc__, source_revision=loading.REVISION, torch=torch.__version__,
        checkpoint_revision=inputs["checkpoint_revision"], manifest_sha256=inputs["manifest_sha256"],
        original_tensor_values_verified=len(index), candidate_executed=False,
        request=inputs["request"], cases=[], errors=[], status="running")

    def publish():
        (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    publish()
    with torch.inference_mode():
        for case in inputs["cases"]:
            row = dict(id=case["id"], language=case["language"], chunks=[], status="success")
            result["cases"].append(row)
            try:
                for chunk in case["chunks"]:
                    with safe_open(args.inputs / chunk["file"], framework="pt") as file:
                        ids = file.get_tensor("input_ids").reshape(1, -1).to("cuda")
                        features = file.get_tensor("input_features_bf16").to("cuda").unsqueeze(0)
                    frames, prompt = chunk["frames"], ids.shape[1]
                    if features.shape != (1, 128, frames) or features.dtype != torch.bfloat16 \
                            or ids[0].tolist() != chunk["prompt_ids"] \
                            or prompt + 1024 > 8192 or not torch.isfinite(features).all():
                        raise ValueError("invalid canonical chunk input")
                    padded = torch.nn.functional.pad(features, (0, (-frames) % 100))
                    mask = (torch.arange(padded.shape[-1], device="cuda") < frames)[None].long()
                    generated = None
                    try:
                        with ExitStack() as hooks:
                            if frames < 100:
                                handle = model.model.audio_tower.conv2d1.register_forward_pre_hook(
                                    lambda module, positional, frames=frames: (positional[0][..., :frames],))
                                hooks.callback(handle.remove)
                            consumed, query_rows = 0, []

                            def positions(module, positional):
                                nonlocal consumed
                                value = positional[1]
                                count = value.shape[-1]
                                expected = torch.arange(consumed, consumed + count, device=value.device)[None]
                                if value.shape != expected.shape or not torch.equal(value, expected):
                                    raise ValueError("canonical generation positions differ from sequential ASR positions")
                                consumed += count
                                query_rows.append(count)

                            handle = model.model.language_model.rotary_emb.register_forward_pre_hook(positions)
                            hooks.callback(handle.remove)
                            torch.manual_seed(0)
                            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                input_features=padded, input_features_mask=mask, max_new_tokens=1024,
                                do_sample=False, num_beams=1, use_cache=True, eos_token_id=[151643, 151645],
                                pad_token_id=151643, return_dict_in_generate=True)
                        sequence = generated.sequences.cpu()
                        if not torch.equal(sequence[:, :prompt], ids.cpu()):
                            raise ValueError("canonical generation did not retain the prompt prefix")
                        tokens = sequence[0, prompt:].tolist()
                        natural = bool(tokens) and tokens[-1] in (151643, 151645)
                        if query_rows != [prompt] + [1] * (len(tokens) - 1):
                            raise ValueError("unexpected canonical cached-generation calls")
                        item = dict(index=chunk["index"], samples=chunk["samples"], frames=frames,
                            prompt_ids=chunk["prompt_ids"],
                            generated_ids=tokens, generated_tokens=len(tokens),
                            finish_reason="stop" if natural else "length", status="success" if natural else "truncated",
                            raw_text=tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                            decoded_text_diagnostic=tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                        row["chunks"].append(item)
                        if not natural:
                            row["status"] = "truncated"
                    finally:
                        generated = None
            except Exception as error:
                row.update(status="failed", error=repr(error))
                result["errors"].append(case["id"])
            publish()
    result["status"] = "complete_reference_hypotheses" if not result["errors"] else "failed"
    publish()
    if result["errors"]:
        raise RuntimeError("canonical recognition failed; all rows retained")


if __name__ == "__main__":
    main()
