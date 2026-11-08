# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the original tokenizer, template and consumed audio settings."""

import json
from pathlib import Path

from ignition.preparation.frontend import annotated_template_asset, prepare_frontend as structural_frontend
from ignition.preparation import runtime_package_pb2 as p


# Endpoint language labels from vLLM v0.29.0, whisper_utils.py at
# 98dff2a81d747d1dba01a47f939f48c3526d4206; these labels do not claim quality in every language.
TRANSCRIPTION_LANGUAGES = {
    'af': 'Afrikaans',
    'ar': 'Arabic',
    'az': 'Azerbaijani',
    'be': 'Belarusian',
    'bg': 'Bulgarian',
    'bs': 'Bosnian',
    'ca': 'Catalan',
    'cs': 'Czech',
    'cy': 'Welsh',
    'da': 'Danish',
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'es': 'Spanish',
    'et': 'Estonian',
    'fa': 'Persian',
    'fi': 'Finnish',
    'fr': 'French',
    'gl': 'Galician',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hr': 'Croatian',
    'hu': 'Hungarian',
    'hy': 'Armenian',
    'id': 'Indonesian',
    'is': 'Icelandic',
    'it': 'Italian',
    'ja': 'Japanese',
    'kk': 'Kazakh',
    'kn': 'Kannada',
    'ko': 'Korean',
    'lt': 'Lithuanian',
    'lv': 'Latvian',
    'mi': 'Maori',
    'mk': 'Macedonian',
    'mr': 'Marathi',
    'ms': 'Malay',
    'ne': 'Nepali',
    'nl': 'Dutch',
    'no': 'Norwegian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'sr': 'Serbian',
    'sv': 'Swedish',
    'sw': 'Swahili',
    'ta': 'Tamil',
    'th': 'Thai',
    'tl': 'Tagalog',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'zh': 'Chinese',
}

# The pinned vLLM SupportsTranscription validator additionally accepts these
# Whisper language codes. Qwen's prompt constructor leaves their labels as the
# literal code, rather than inventing a model-supported language name. This is
# frontend compatibility, not recognition-quality qualification.
TRANSCRIPTION_LANGUAGES.update({code: code for code in (
    'am', 'as', 'ba', 'bn', 'bo', 'br', 'eu', 'fo', 'gu', 'ha', 'haw', 'ht', 'jw',
    'ka', 'km', 'la', 'lb', 'ln', 'lo', 'mg', 'ml', 'mn', 'mt', 'my', 'nn', 'oc',
    'pa', 'ps', 'sa', 'sd', 'si', 'sn', 'so', 'sq', 'su', 'te', 'tg', 'tk', 'tt',
    'uz', 'yi', 'yo', 'yue')})

TRANSCRIPTION = dict(
    context_prefix="<|im_start|>system\n", context_suffix="<|im_end|>\n",
    audio_prefix="<|im_start|>user\n", audio_suffix="<|im_end|>\n",
    assistant_prefix="<|im_start|>assistant\n", language_prefix="language ",
    text_delimiter="<asr_text>", default_separator=" ",
    languages=[dict(code=code, label=label, chunk_separator="" if code in ("zh", "ja") else " ")
               for code, label in TRANSCRIPTION_LANGUAGES.items()],
    sanitization=dict(prefix="<|", suffix="|>", excluded_body_character="|"))


def prepare_frontend(snapshot):
    snapshot = Path(snapshot)
    processor = json.loads((snapshot / "preprocessor_config.json").read_bytes())
    expected = dict(chunk_length=30, dither=0.0, feature_extractor_type="WhisperFeatureExtractor",
        feature_size=128, hop_length=160, n_fft=400, n_samples=480000, nb_max_frames=3000,
        padding_side="right", padding_value=0.0, processor_class="Qwen3ASRProcessor", return_attention_mask=True)
    if processor != expected:
        raise ValueError("original ASR audio processor differs from the validated frontend")
    result = structural_frontend(snapshot, eos_token_ids=[151643, 151645])
    result["audio"] = dict(sample_rate=16000, mel_bins=128, fft_size=400, hop_samples=160,
        max_file_bytes=25 * 1024 * 1024, max_samples=600 * 16000, decode_timeout_millis=60000,
        service_chunk_samples=30 * 16000, silence_search_samples=16000, silence_window_samples=1600,
        sample_padding=p.AUDIO_SAMPLE_PADDING_HOP_MULTIPLE, numerics=p.AUDIO_NUMERICS_FP32_FMA)
    return result


def runtime_assets(snapshot):
    from transformers import Qwen2TokenizerFast

    snapshot = Path(snapshot)
    tokenizer = Qwen2TokenizerFast.from_pretrained(snapshot, local_files_only=True)
    required = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645,
                "<|audio_start|>": 151669, "<|audio_end|>": 151670, "<|audio_pad|>": 151676,
                "<asr_text>": 151704}
    if any(tokenizer.convert_tokens_to_ids(text) != index for text, index in required.items()):
        raise ValueError("original ASR tokenizer marker identities differ")
    source = json.loads((snapshot / "chat_template.json").read_bytes())["chat_template"].encode("utf-8")
    return {"tokenizer": {"text": tokenizer.backend_tokenizer.to_str()},
            "chat_template": annotated_template_asset(source, Path(__file__).with_name("assets") / "chat_template.patch.json")}
