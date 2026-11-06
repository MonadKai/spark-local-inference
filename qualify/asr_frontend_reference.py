# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate CPU frontend cases with the isolated, original-format SDK reference.

Run under its Transformers 4.57.6 environment. This never imports Ignition or
candidate audio code, and does not run model inference or select quality audio.
"""

import argparse
import ast
import json
import hashlib
import math
from pathlib import Path

import numpy as np
from transformers import Qwen2Tokenizer, Qwen2TokenizerFast


VLLM_REVISION = "98dff2a81d747d1dba01a47f939f48c3526d4206"
VLLM_AUDIO_SOURCE_SHA256 = "66a7766caf7bc4977006e9dce3d95b702bae71f1e1a6549318247d74a67e6cfc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vllm-audio-source", type=Path, required=True)
    parser.add_argument("--transcription-reference", type=Path,
        help="actual vLLM CPU frontend output to check with the independent slow tokenizer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import transformers
    if transformers.__version__ != "4.57.6" or np.__version__ != "2.2.6":
        raise ValueError("this reference requires Transformers 4.57.6 and NumPy 2.2.6")
    slow = Qwen2Tokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    fast = Qwen2TokenizerFast.from_pretrained(args.checkpoint, local_files_only=True)
    if slow.is_fast or not fast.is_fast:
        raise ValueError("independent slow tokenizer is required")
    args.output.mkdir(exist_ok=False)
    template = json.loads((args.checkpoint / "chat_template.json").read_bytes())["chat_template"]
    cases = []
    for index, context in enumerate(["", "Names: Alice and GB10.", "请识别普通话。", "Café, naïve — 東京 😀"]):
        for generation in [False, True]:
            messages = ([dict(role="system", content=context)] if context else []) + [
                dict(role="user", content=[dict(type="audio", audio="external")])]
            text = slow.apply_chat_template(messages, chat_template=template, tokenize=False,
                add_generation_prompt=generation)
            ids = slow.encode(text, add_special_tokens=False)
            if fast.encode(text, add_special_tokens=False) != ids:
                raise ValueError("original fast and slow prompt tokenizer disagree")
            cases.append(dict(name=f"context{index}-generation{int(generation)}", messages=messages,
                options=dict(add_generation_prompt=generation), rendered=text, token_ids=ids))
    texts = ["", "hello", "你好，世界！", "Café naïve 東京 😀", "<asr_text>",
        "<|audio_start|><|audio_pad|><|audio_end|>", "language English<asr_text>hello",
        "language Chinese<asr_text>你好", "<|endoftext|><|im_end|>", "a\n\n b\t中"]
    encode, decode = [], []
    for index, text in enumerate(texts):
        for add in [False, True]:
            ids = slow.encode(text, add_special_tokens=add)
            if fast.encode(text, add_special_tokens=add) != ids:
                raise ValueError("original fast and slow tokenizer disagree")
            encode.append(dict(name=f"encode{index}-{int(add)}", text=text, add_special_tokens=add, token_ids=ids))
        for skip in [False, True]:
            ids = slow.encode(text, add_special_tokens=False)
            decoded = slow.decode(ids, skip_special_tokens=skip, clean_up_tokenization_spaces=False)
            if fast.decode(ids, skip_special_tokens=skip, clean_up_tokenization_spaces=False) != decoded:
                raise ValueError("original fast and slow decoder disagree")
            decode.append(dict(name=f"decode{index}-{int(skip)}", token_ids=ids, skip_special_tokens=skip, text=decoded))
    frontend = dict(cases=cases, encode_cases=encode, decode_cases=decode, default_overrides=[],
        template_defaults=dict(add_generation_prompt=True))
    transcription_cases = []
    if args.transcription_reference:
        recorded = json.loads(args.transcription_reference.read_bytes())
        if recorded["expected_vllm_revision"] != VLLM_REVISION or not recorded["critical_installed_frontend_files_equal"]:
            raise ValueError("transcription reference identity differs")
        transcription_cases = recorded["cases"]
        for case in transcription_cases:
            if slow.encode(case["rendered"], add_special_tokens=False) != case["token_ids"]:
                raise ValueError(f"independent slow transcription tokens differ: {case['name']}")
        frontend["transcription_cases"] = transcription_cases
    (args.output / "frontend.json").write_text(json.dumps(frontend, ensure_ascii=False, indent=2) + "\n")
    # Execute the pinned baseline functions unchanged, with only their module
    # dependencies supplied. Loading the whole vLLM module would initialize CUDA.
    source_bytes = args.vllm_audio_source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != VLLM_AUDIO_SOURCE_SHA256:
        raise ValueError("vLLM audio source differs from the pinned reference revision")
    source = ast.parse(source_bytes)
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
        and node.name in ("split_audio", "find_split_point")]
    if len(functions) != 2:
        raise ValueError("pinned vLLM split functions are missing")
    scope = dict(np=np, math=math)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(args.vllm_audio_source), "exec"), scope)
    split_cases = []
    waveforms = [(f"silence-{n}", np.zeros(n, dtype=np.float32)) for n in (201, 479999, 480000, 480001, 960001)]
    for seed in range(12):
        rng = np.random.default_rng(seed)
        waveforms.append((f"random-{seed}", rng.uniform(-0.5, 0.5, 960001).astype(np.float32)))
    crafted = np.full(600000, 0.5, dtype=np.float32)
    crafted[478400:480000] = 0
    crafted[467200:468800] = 0.25
    waveforms.append(("excluded-last-candidate", crafted.copy()))
    crafted[465600:467200] = 0.25
    waveforms.append(("first-equal-minimum", crafted.copy()))
    tied = np.full(600000, 0.6, dtype=np.float32)
    tied[464000:465600] = 0.5
    tied[464000] = np.float32(0.5000477)
    tied[465600:467200] = 0.5
    first_mean = (tied[464000:465600] ** 2).mean()
    second_mean = (tied[465600:467200] ** 2).mean()
    if not first_mean > second_mean or first_mean ** 0.5 != second_mean ** 0.5:
        raise ValueError("FP32 square-root tie control did not trigger")
    waveforms.append(("fp32-square-root-tie", tied))
    for name, pcm in waveforms:
        pcm.astype("<f4").tofile(args.output / (name + ".f32"))
        chunks = scope["split_audio"](pcm, 16000, 30.0, 1.0, 1600)
        cursor = 0
        ranges = []
        for chunk in chunks:
            end = cursor + len(chunk)
            if not np.array_equal(chunk, pcm[cursor:end]):
                raise ValueError("reference splitting did not return a continuous partition")
            ranges.append([cursor, end])
            cursor = end
        if cursor != len(pcm):
            raise ValueError("reference split omitted input samples")
        split_cases.append(dict(name=name, samples=len(pcm), ranges=ranges))
    (args.output / "split-cases.json").write_text(json.dumps(dict(numpy=np.__version__,
        vllm_revision=VLLM_REVISION, vllm_audio_source_sha256=VLLM_AUDIO_SOURCE_SHA256,
        sqrt_scalar_type=type(np.float32(0.5) ** 0.5).__name__, cases=split_cases), indent=2) + "\n")
    print(json.dumps(dict(template_cases=len(cases), encode_cases=len(encode), decode_cases=len(decode),
        split_cases=len(split_cases), transcription_cases=len(transcription_cases), model_inference=False)))


if __name__ == "__main__":
    main()
