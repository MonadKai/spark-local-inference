# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise pinned vLLM file decode/splitting and processor boundaries on CPU.

Only ordinary configuration attributes are supplied to the real unbound serving
method; its decoder/splitter and the registered processor are not replaced.
No engine, model forward or HTTP response is established by this check.
"""

import argparse
import importlib.metadata
import io
import json
from pathlib import Path
from types import SimpleNamespace
import wave


def wav(samples):
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        target.writeframes(bytes(samples * 2))
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import vllm
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration
    from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing

    if vllm.__version__ != "0.29.0":
        raise ValueError("vLLM version changed")
    installed = Path(vllm.__file__).parent
    for name in ["model_executor/models/qwen3_asr.py", "config/speech_to_text.py",
                 "entrypoints/speech_to_text/base/serving.py", "multimodal/media/audio.py", "multimodal/audio.py"]:
        if (installed / name).read_bytes() != (args.vllm_source / "vllm" / name).read_bytes():
            raise ValueError(f"installed file differs from supplied pinned source: {name}")
    config = ModelConfig(model=args.checkpoint, tokenizer=args.checkpoint, trust_remote_code=False,
        dtype="bfloat16", max_model_len=8192, limit_mm_per_prompt={"audio": 1},
        mm_processor_cache_gb=0, enforce_eager=True)
    asr_config = Qwen3ASRForConditionalGeneration.get_speech_to_text_config(config, "transcribe")
    receiver = SimpleNamespace(asr_config=asr_config,
        max_audio_decode_duration_s=600, max_audio_decode_bytes=64 * 1024 * 1024)
    processor = MULTIMODAL_REGISTRY.create_processor(config, cache=None)
    if type(processor).__name__ != "Qwen3ASRMultiModalProcessor":
        raise ValueError("unexpected registered multimodal processor")
    cases = []
    for samples in [0, 1, 159, 160, 161, 200, 201, 7999, 16000, 16001, 480001, 9600000, 9600001]:
        row = dict(samples=samples)
        try:
            chunks, duration = SpeechToTextBaseServing._decode_and_chunk_speech(receiver, wav(samples))
        except Exception as error:
            row["decode"] = dict(error_type=type(error).__name__, message=str(error))
            cases.append(row)
            continue
        row["decode"] = dict(duration=duration, chunks=[len(chunk) for chunk in chunks],
                             dtypes=[str(chunk.dtype) for chunk in chunks])
        assert sum(len(chunk) for chunk in chunks) == samples
        assert all(chunk.dtype == np.float32 and np.isfinite(chunk).all() and (chunk == 0).all() for chunk in chunks)
        if samples <= 16001:
            try:
                items = processor.info.parse_mm_data({"audio": [(chunks[0], 16000)]})
                output = processor("<|audio_start|><|audio_pad|><|audio_end|>",
                    mm_items=items, hf_processor_mm_kwargs={})
                data = output["mm_kwargs"]["audio"][0].get_data()
                row["processor"] = dict(frames=int(data["audio_feature_lengths"].item()),
                    audio_tokens=output["prompt_token_ids"].count(151676))
            except Exception as error:
                row["processor"] = dict(error_type=type(error).__name__, message=str(error))
        cases.append(row)
    dependencies = {}
    for name in ["vllm", "torch", "transformers", "torchaudio", "librosa", "soundfile", "av", "resampy"]:
        try: dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: dependencies[name] = None
    observations = {row["samples"]: row for row in cases}
    required_controls_pass = all("error_type" not in observations[count]["decode"] for count in [201,16000,9600000]) \
        and "error_type" in observations[9600001]["decode"]
    result = dict(scope="real vLLM CPU file decode/splitter/registered processor only; not HTTP or model inference",
        expected_vllm_revision="98dff2a81d747d1dba01a47f939f48c3526d4206", source_revision_verified_by_this_check=False,
        critical_installed_files_match_supplied_source=True,
        explicit_test_decode_limits=dict(max_duration_s=600, max_decode_bytes=64 * 1024 * 1024,
            scope="downstream test configuration, not universal vLLM defaults"),
        asr_config=dict(sample_rate=asr_config.sample_rate, max_audio_clip_s=asr_config.max_audio_clip_s,
            overlap_chunk_second=asr_config.overlap_chunk_second,
            min_energy_split_window_size=asr_config.min_energy_split_window_size),
        installed_audio_dependencies=dependencies,
        vllm_declared_audio_requirements=[requirement for requirement in importlib.metadata.requires("vllm")
            if 'extra == "audio"' in requirement or "extra == 'audio'" in requirement],
        required_decode_controls_pass=required_controls_pass, cases=cases, model_forward_executed=False)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result))
    if not required_controls_pass:
        raise RuntimeError("mandatory positive decode or duration rejection controls failed; no decode qualification")


if __name__ == "__main__":
    main()
