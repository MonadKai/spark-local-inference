# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit original-format configuration for the canonical ASR module tree."""

from pathlib import Path

from ignition.preparation.checkpoint import CheckpointError, _parse_json
from .checkpoint import validate_config


def canonical_config(snapshot: str | Path):
    """Build structural configuration without importing original remote code.

    Generation metadata supplies EOS/padding only. Sampling recommendations
    belong to the recipe declaration, not these checkpoint files.
    """
    from transformers import Qwen3ASRConfig, Qwen3ASREncoderConfig, Qwen3Config

    snapshot = Path(snapshot)
    config = _parse_json((snapshot / "config.json").read_bytes(), "config.json")
    validate_config(config)
    generation = _parse_json((snapshot / "generation_config.json").read_bytes(), "generation_config.json")
    tokenizer = _parse_json((snapshot / "tokenizer_config.json").read_bytes(), "tokenizer_config.json")
    if generation.get("eos_token_id") != [151643, 151645] or generation.get("pad_token_id") != 151643:
        raise CheckpointError("original ASR requires EOS [151643,151645] and input pad 151643")
    if (tokenizer.get("pad_token") != "<|endoftext|>" or tokenizer.get("eos_token") != "<|im_end|>"
            or tokenizer.get("add_bos_token") is not False or tokenizer.get("bos_token") is not None
            or tokenizer.get("add_prefix_space") is not False
            or tokenizer.get("split_special_tokens") is not False
            or tokenizer.get("tokenizer_class") != "Qwen2Tokenizer"):
        raise CheckpointError("original ASR tokenizer padding/EOS/BOS semantics differ")
    expected = {151643: ("<|endoftext|>", True), 151644: ("<|im_start|>", True),
                151645: ("<|im_end|>", True), 151669: ("<|audio_start|>", True),
                151670: ("<|audio_end|>", True), 151676: ("<|audio_pad|>", True),
                151704: ("<asr_text>", False)}
    for identity, (content, special) in expected.items():
        declaration = tokenizer.get("added_tokens_decoder", {}).get(str(identity), {})
        if declaration.get("content") != content or declaration.get("special") is not special:
            raise CheckpointError(f"original ASR token {identity} content/special semantics differ")
    if "151705" in tokenizer["added_tokens_decoder"]:
        raise CheckpointError("the original ASR tokenizer does not declare a timestamp token")
    thinker = config["thinker_config"]
    original_audio, original_text = thinker["audio_config"], thinker["text_config"]
    audio = Qwen3ASREncoderConfig(**{field: original_audio[field] for field in (
        "num_mel_bins", "encoder_layers", "encoder_attention_heads", "encoder_ffn_dim", "d_model",
        "dropout", "attention_dropout", "activation_function", "activation_dropout", "scale_embedding",
        "initializer_range", "n_window", "output_dim", "n_window_infer", "downsample_hidden_size")},
        max_position_embeddings=13)
    text = Qwen3Config(**{field: original_text[field] for field in (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "hidden_act", "max_position_embeddings", "initializer_range",
        "rms_norm_eps", "use_cache", "tie_word_embeddings", "attention_bias", "attention_dropout")},
        rope_parameters={"rope_type": "default", "rope_theta": original_text["rope_theta"]},
        sliding_window=None, layer_types=["full_attention"] * 28,
        pad_token_id=151643, eos_token_id=[151643, 151645])
    # The original ASR positions repeat the same sequential text positions on
    # all three mRoPE axes. Section/interleave selection then retains the full
    # ordinary RoPE frequencies; no Qwen3.8 partial rotary configuration enters.
    return Qwen3ASRConfig(audio_config=audio, text_config=text, audio_token_id=thinker["audio_token_id"],
        initializer_range=thinker["initializer_range"], tie_word_embeddings=True,
        pad_token_id=151643, eos_token_id=[151643, 151645])
