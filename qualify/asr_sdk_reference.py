# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Check original-format loading in the isolated, pinned Qwen SDK environment.

No candidate adapter, renamed weights or model forward is used. Checkpoint
content identity is an external precondition, separate from exact loading.
"""

import argparse
from contextlib import ExitStack
import importlib.metadata
import importlib.util
import json
from pathlib import Path


SDK_REVISION = "7c6daf77a2421100f5fb066495372c00129d39ff"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sdk-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import transformers
    from safetensors import safe_open
    import qwen_asr
    from qwen_asr.core.transformers_backend import configuration_qwen3_asr as configuration
    from qwen_asr.core.transformers_backend import modeling_qwen3_asr as modeling

    if transformers.__version__ != "4.57.6" or importlib.metadata.version("qwen-asr") != "0.0.6":
        raise ValueError("original SDK reference dependency versions differ")
    if importlib.util.find_spec("ignition") is not None or importlib.util.find_spec("flare_heliopause") is not None:
        raise ValueError("reference environment must exclude candidate preparation/runtime packages")
    installed = Path(qwen_asr.__file__).parent
    source = args.sdk_source / "qwen_asr"
    original_files = {path.relative_to(source) for path in source.rglob("*.py")}
    installed_files = {path.relative_to(installed) for path in installed.rglob("*.py")}
    if not original_files or installed_files != original_files:
        raise ValueError("installed SDK Python source inventory differs from supplied fixed source")
    for path in sorted(original_files):
        if (installed / path).read_bytes() != (source / path).read_bytes():
            raise ValueError(f"installed SDK Python source differs: {path}")
    torch.set_num_threads(2)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("SDK positional constants require the original FP32 default")
    config = configuration.Qwen3ASRConfig.from_pretrained(args.checkpoint, local_files_only=True)
    model, loading = modeling.Qwen3ASRForConditionalGeneration.from_pretrained(args.checkpoint,
        config=config, dtype=torch.bfloat16, device_map={"": "cpu"}, attn_implementation="eager",
        local_files_only=True, trust_remote_code=False, output_loading_info=True)
    for name in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]:
        if loading.get(name):
            raise ValueError(f"SDK loading reported {name}: {loading[name]}")
    model.eval()
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_bytes())["weight_map"]
    actual = model.state_dict()
    if len(index) != 708 or set(actual) != set(index):
        raise ValueError("SDK state does not consume the complete original tensor inventory without renaming")
    with ExitStack() as owners:
        shards = {name: owners.enter_context(safe_open(args.checkpoint / name, framework="pt", device="cpu"))
            for name in sorted(set(index.values()))}
        for filename, shard in shards.items():
            if set(shard.keys()) != {name for name, file in index.items() if file == filename}:
                raise ValueError("checkpoint shard contains unindexed or missing tensors")
        for name, shard in index.items():
            original = shards[shard].get_tensor(name)
            target = actual[name]
            if original.dtype != torch.bfloat16 or target.dtype != original.dtype or target.shape != original.shape:
                raise ValueError(f"SDK tensor shape/precision differs: {name}")
            if not torch.isfinite(original).all() or not torch.equal(
                    original.contiguous().view(torch.uint8), target.contiguous().view(torch.uint8)):
                raise ValueError(f"SDK tensor is nonfinite or differs from original: {name}")
    buffers = []
    for name, value in model.named_buffers():
        if value.is_meta:
            raise ValueError(f"SDK buffer is unmaterialized: {name}")
        if ("inv_freq" in name or "positional_embedding" in name) and value.dtype != torch.float32:
            raise ValueError(f"SDK positional constants lost FP32 precision: {name}")
        buffers.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype)))
    audio_config = config.thinker_config.audio_config
    text_config = config.thinker_config.text_config
    position = model.thinker.audio_tower.positional_embedding.positional_embedding
    rotary = model.thinker.model.rotary_emb
    expected_position = modeling.SinusoidsPositionEmbedding(
        audio_config.max_source_positions, audio_config.d_model).positional_embedding
    expected_rotary = 1.0 / (text_config.rope_theta ** (
        torch.arange(0, text_config.head_dim, 2, dtype=torch.int64).float() / text_config.head_dim))
    if position.shape != (1500, 1024) or not torch.equal(position, expected_position):
        raise ValueError("SDK original sinusoidal constants differ")
    for frequency in [rotary.inv_freq, rotary.original_inv_freq]:
        if frequency.dtype != torch.float32 or frequency.shape != (64,) or not torch.equal(frequency, expected_rotary):
            raise ValueError("SDK original rotary constants differ")
    embedding = actual["thinker.model.embed_tokens.weight"]
    head = actual["thinker.lm_head.weight"]
    result = dict(reference="original-format Qwen SDK without candidate adapters or weight conversion",
        expected_sdk_revision=SDK_REVISION, sdk_python_files_equal_to_supplied_source=len(original_files),
        sdk_source_revision_verified_by_this_check=False, transformers=transformers.__version__,
        expected_checkpoint="Qwen/Qwen3-ASR-1.7B",
        expected_checkpoint_revision="7278e1e70fe206f11671096ffdd38061171dd6e5",
        checkpoint_content_identity_verified_by_this_check=False, original_tensors_compared=len(index),
        all_original_tensor_bytes_equal=True,
        embedding_head_storage_alias=embedding.untyped_storage().data_ptr() == head.untyped_storage().data_ptr(),
        original_fp32_positional_constants_equal=True,
        loading_device="cpu", model_forward_executed=False, buffers=buffers)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "buffers"}))


if __name__ == "__main__":
    main()
