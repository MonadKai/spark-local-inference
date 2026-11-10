# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare canonical recognition inputs with the pinned real vLLM frontend.

Run in the isolated vLLM image without a GPU. This does not execute a model or
change the frozen qualification sample. Preserve all failures and actual chunk
boundaries; the independent canonical model consumes these represented inputs.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pcm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import vllm
    from safetensors.torch import save_file
    from vllm.config import ModelConfig
    from vllm.config.speech_to_text import SpeechToTextParams
    from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
    from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration as Model
    from vllm.multimodal import MULTIMODAL_REGISTRY

    if vllm.__version__ != "0.29.0" or torch.cuda.is_initialized():
        raise ValueError("pinned CPU-only vLLM frontend required")
    installed = Path(vllm.__file__).parent
    for name in ("model_executor/models/qwen3_asr.py", "model_executor/models/qwen3_omni_moe_thinker.py",
            "model_executor/models/qwen2_5_omni_thinker.py", "transformers_utils/processors/qwen3_asr.py",
            "config/speech_to_text.py", "entrypoints/speech_to_text/base/serving.py",
            "multimodal/media/audio.py", "multimodal/audio.py"):
        if (installed / name).read_bytes() != (args.vllm_source / "vllm" / name).read_bytes():
            raise ValueError("installed frontend source differs: " + name)
    manifest = json.loads(args.manifest.read_bytes())
    if manifest["split"] != "qualification" or len(manifest["utterances"]) != 256:
        raise ValueError("the fixed qualification manifest is required")
    if manifest["request"] != dict(temperature=0, seed=0, max_model_len=8192,
            context="", language="utterance.language", max_completion_tokens=1024):
        raise ValueError("qualification request differs from the fixed executable protocol")
    torch.set_num_threads(2)
    config = ModelConfig(model=args.checkpoint, tokenizer=args.checkpoint, trust_remote_code=False,
        dtype="bfloat16", max_model_len=8192, limit_mm_per_prompt={"audio": 1},
        mm_processor_cache_gb=0, enforce_eager=True)
    stt = Model.get_speech_to_text_config(config, "transcribe")
    receiver = SimpleNamespace(asr_config=stt, max_audio_decode_duration_s=600,
        max_audio_decode_bytes=64 * 1024 * 1024)
    processor = MULTIMODAL_REGISTRY.create_processor(config, cache=None)
    if type(processor).__name__ != "Qwen3ASRMultiModalProcessor":
        raise ValueError("unexpected registered processor")
    args.output.mkdir(exist_ok=False)
    result = dict(scope=__doc__, vllm_revision="98dff2a81d747d1dba01a47f939f48c3526d4206",
        checkpoint_revision="7278e1e70fe206f11671096ffdd38061171dd6e5",
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        split="qualification", model_executed=False, cuda_initialized=False,
        feature_dtype="bfloat16", feature_boundary="real registered processor output cast to configured model dtype",
        request=manifest["request"], cases=[], errors=[], status="running")

    def publish():
        (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    publish()
    for row in manifest["utterances"]:
        case = dict(id=row["id"], language=row["language"], samples=row["samples"], chunks=[])
        result["cases"].append(case)
        try:
            path = args.pcm_root / row["path"]
            with wave.open(str(path), "rb") as source:
                if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()) != (1, 2, 16000, row["samples"]):
                    raise ValueError("PCM file geometry differs from frozen input")
                pcm = source.readframes(source.getnframes())
            if hashlib.sha256(pcm).hexdigest() != row["pcm_s16le_sha256"]:
                raise ValueError("PCM bytes differ from frozen input")
            chunks, duration = SpeechToTextBaseServing._decode_and_chunk_speech(receiver, path.read_bytes())
            if sum(len(chunk) for chunk in chunks) != row["samples"] or duration != row["samples"] / 16000:
                raise ValueError("decoded sample count or splitter coverage differs")
            case["duration"] = duration
            offset = 0
            for index, chunk in enumerate(chunks):
                if chunk.dtype != np.float32 or not np.isfinite(chunk).all() or not len(chunk):
                    raise ValueError("invalid decoded chunk")
                params = SpeechToTextParams(audio=chunk, stt_config=stt, model_config=config,
                    language=row["language"], task_type="transcribe", request_prompt="")
                prompt = Model.get_generation_prompt(params)
                items = processor.info.parse_mm_data(prompt["multi_modal_data"])
                processed = processor(prompt["prompt_token_ids"], mm_items=items, hf_processor_mm_kwargs={})
                data = processed["mm_kwargs"]["audio"][0].get_data()
                features = data["input_audio_features"].contiguous()
                frames = int(data["audio_feature_lengths"].item())
                tokens = processed["prompt_token_ids"]
                if (features.shape != (128, frames) or features.dtype != torch.bfloat16
                        or not torch.isfinite(features).all() or frames != (len(chunk) + 159) // 160
                        or tokens.count(151676) != 13 * (frames // 100) + (frames % 100 + 7) // 8
                        or len(tokens) + 1024 > 8192):
                    raise ValueError("registered processor returned unexpected model input: " + str(dict(
                        shape=list(features.shape), dtype=str(features.dtype), frames=frames,
                        expected_frames=(len(chunk) + 159) // 160,
                        audio_tokens=tokens.count(151676),
                        expected_audio_tokens=13 * (frames // 100) + (frames % 100 + 7) // 8,
                        prompt_tokens=len(tokens))))
                filename = f"{row['id']}-chunk{index}.safetensors"
                save_file(dict(input_features_bf16=features, input_ids=torch.tensor(tokens, dtype=torch.int64),
                    pcm_fp32=torch.from_numpy(chunk.copy())), args.output / filename)
                case["chunks"].append(dict(index=index, start_sample=offset, samples=len(chunk), frames=frames,
                    file=filename, prompt_ids=tokens))
                offset += len(chunk)
                if torch.cuda.is_initialized():
                    raise ValueError("CPU frontend unexpectedly initialized CUDA")
            case["status"] = "prepared"
        except Exception as error:
            case.update(status="failed", error=repr(error))
            result["errors"].append(row["id"])
        publish()
    result["status"] = "failed" if result["errors"] else "inputs_prepared_no_model_execution"
    publish()
    if result["errors"]:
        raise RuntimeError("qualification input preparation failed; all cases retained")


if __name__ == "__main__":
    main()
