# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Apply the pinned real vLLM chunk cleanup/joiner to reference hypotheses.

Run in the isolated CPU reference image. Failed/truncated rows and partial
outputs remain in the result, with their original non-success status.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import vllm
    from vllm import SamplingParams
    from vllm.config import ModelConfig
    from vllm.entrypoints.speech_to_text.base.serving import asr_inter_chunk_separator
    from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration as Model
    from vllm.tokenizers import cached_tokenizer_from_config
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer

    if vllm.__version__ != "0.29.0":
        raise ValueError("pinned cleanup implementation required")
    for name in ("model_executor/models/qwen3_asr.py", "entrypoints/speech_to_text/base/serving.py",
            "v1/engine/__init__.py", "v1/engine/detokenizer.py", "tokenizers/detokenizer_utils.py",
            "v1/engine/output_processor.py", "model_executor/models/interfaces.py"):
        if (Path(vllm.__file__).parent / name).read_bytes() != (args.vllm_source / "vllm" / name).read_bytes():
            raise ValueError("installed cleanup differs from pinned source")
    config = ModelConfig(model=str(args.checkpoint), tokenizer=str(args.checkpoint), trust_remote_code=False,
        dtype="bfloat16", max_model_len=8192, limit_mm_per_prompt={"audio": 1},
        mm_processor_cache_gb=0, enforce_eager=True)
    tokenizer = cached_tokenizer_from_config(config)

    def replay(prompt, tokens, finish):
        request = EngineCoreRequest(request_id="canonical-recognition-replay", prompt_token_ids=list(prompt),
            mm_features=None, sampling_params=SamplingParams(temperature=0, seed=0, max_tokens=1024,
                skip_special_tokens=True, spaces_between_special_tokens=True, include_stop_str_in_output=False),
            pooling_params=None, arrival_time=0., lora_request=None, cache_salt=None, data_parallel_rank=None)
        decoder = IncrementalDetokenizer.from_new_request(tokenizer, request)
        matched = decoder.update(list(tokens), stop_terminated=finish == "stop")
        if matched is not None or list(decoder.output_token_ids) != list(tokens):
            raise ValueError("detokenizer replay changed generated IDs or introduced a string stop")
        return decoder.get_next_output_text(finished=True, delta=False)

    if args.controls_only:
        prompt = tokenizer.encode("<|im_start|>assistant\nlanguage Chinese<asr_text>", add_special_tokens=False)
        tokens = tokenizer.encode("𠀀", add_special_tokens=False)
        partial, complete = replay(prompt, tokens[:1], "length"), replay(prompt, tokens + [151645], "stop")
        whole = tokenizer.decode(tokens[:1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if partial != "" or complete != "𠀀" or partial == whole or replay(prompt, [151645], "stop") != "":
            raise ValueError("incremental UTF-8/EOS controls failed")
        controls = dict(scope="actual CPU detokenizer controls; no qualification hypotheses or model execution",
            truncated_piece=partial, whole_decode_negative=whole, complete=complete, generated_ids_preserved=True)
        with args.output.open("x") as output:
            json.dump(controls, output, indent=2, ensure_ascii=False)
            output.write("\n")
        return
    if args.raw is None:
        parser.error("--raw is required outside --controls-only")
    manifest, raw = json.loads(args.manifest.read_bytes()), json.loads(args.raw.read_bytes())
    if raw["manifest_sha256"] != hashlib.sha256(args.manifest.read_bytes()).hexdigest() \
            or [c["id"] for c in raw["cases"]] != [c["id"] for c in manifest["utterances"]]:
        raise ValueError("raw hypotheses differ from frozen qualification selection")
    values = []
    for case in raw["cases"]:
        if case["status"] not in ("success", "failed", "truncated") \
                or [c["index"] for c in case["chunks"]] != list(range(len(case["chunks"]))):
            raise ValueError("invalid status or chunk order")
        chunks = [Model.post_process_output(replay(chunk["prompt_ids"], chunk["generated_ids"], chunk["finish_reason"]))
            for chunk in case["chunks"]]
        separator = asr_inter_chunk_separator(case["language"], Model.no_space_languages)
        values.append(dict(id=case["id"], status=case["status"], text=separator.join(chunks),
            chunks=chunks, raw_chunks=case["chunks"], error=case.get("error")))
    with args.output.open("x") as output:
        json.dump(values, output, indent=2, ensure_ascii=False)
        output.write("\n")


if __name__ == "__main__":
    main()
