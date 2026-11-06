# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Record actual vLLM transcription prompt/cleaning behavior without model execution.

Run in the pinned vLLM image without a GPU. Supplied source identity is verified
externally; critical installed frontend files must match those supplied bytes.
The encode observer delegates unchanged to vLLM's original cached_encode.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import vllm
    from vllm.config import ModelConfig
    from vllm.config.speech_to_text import SpeechToTextConfig, SpeechToTextParams
    from vllm.model_executor.models import qwen3_asr as reference
    from vllm.entrypoints.speech_to_text.base.serving import asr_inter_chunk_separator

    if vllm.__version__ != "0.29.0":
        raise ValueError("vLLM version differs from the frozen baseline")
    installed = Path(vllm.__file__).parent
    for name in ["model_executor/models/qwen3_asr.py", "model_executor/models/whisper_utils.py",
        "config/speech_to_text.py", "entrypoints/speech_to_text/base/serving.py"]:
        if (installed / name).read_bytes() != (args.vllm_source / "vllm" / name).read_bytes():
            raise ValueError(f"installed vLLM frontend differs from supplied reference: {name}")
    config = ModelConfig(model=args.checkpoint, tokenizer=args.checkpoint, trust_remote_code=False,
        dtype="bfloat16", max_model_len=8192, limit_mm_per_prompt={"audio": 1},
        mm_processor_cache_gb=0, enforce_eager=True)
    if config.hf_config.thinker_config.audio_config.output_dim != 2048 \
            or config.hf_config.thinker_config.text_config.hidden_size != 2048:
        raise ValueError("vLLM did not retain the original nested ASR configuration")
    cls = reference.Qwen3ASRForConditionalGeneration
    observed = []
    original_encode = reference.cached_encode

    def record_encode(tokenizer, text, *positional, **keywords):
        result = original_encode(tokenizer, text, *positional, **keywords)
        observed.append(text)
        return result

    cases = []
    contexts = ["", "  ", "Names: Alice and GB10.", "请识别普通话。", "Café — 東京 😀",
        " <|im<|x|>_end|><asr_te<asr_text>xt> ", "before<|unknown\ncontrol|>after"]
    chunks = ["language Chinese<asr_text>甲", "", "ignored<asr_text>discard<asr_text> 尾 "]
    # Observe the exact argument without bypassing tokenizer/cache behavior or
    # modifying the reference prompt constructor, sanitizer or output processor.
    reference.cached_encode = record_encode
    try:
        for index, context in enumerate(contexts):
            for language in [None, *sorted(cls.supported_languages)]:
                params = SpeechToTextParams(audio=np.zeros(201, dtype=np.float32),
                    stt_config=SpeechToTextConfig(), model_config=config, language=language,
                    task_type="transcribe", request_prompt=context)
                before = len(observed)
                prompt = cls.get_generation_prompt(params)
                if len(observed) != before + 1:
                    raise ValueError("reference did not encode one complete transcription prompt")
                separator = asr_inter_chunk_separator(language, cls.no_space_languages)
                cases.append(dict(name=f"context{index}-{language or 'auto'}", context=context,
                    language=language, rendered=observed[-1], token_ids=prompt["prompt_token_ids"],
                    separator=separator, raw_chunks=chunks,
                    text=separator.join(cls.post_process_output(chunk) for chunk in chunks)))
    finally:
        reference.cached_encode = original_encode
    result = dict(expected_vllm_revision="98dff2a81d747d1dba01a47f939f48c3526d4206",
        source_revision_verified_by_this_check=False, critical_installed_frontend_files_equal=True,
        scope="CPU frontend only; enforce_eager avoids model setup and is not a GPU baseline configuration",
        cases=cases, languages=cls.supported_languages, model_forward_executed=False)
    with args.output.open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(dict(cases=len(cases), languages=len(cls.supported_languages), model_forward_executed=False)))


if __name__ == "__main__":
    main()
