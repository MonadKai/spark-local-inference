# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Load and verify the original checkpoint with the independent upstream converter.

Run in the isolated canonical Transformers environment, using Python -I.
This CPU loading check performs no model forward and imports no Ignition code.
The original SDK independently checks the original-format loading semantics.
The caller must separately establish the supplied checkpoint's content identity;
this tool checks complete value consumption relative to those supplied files.
"""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path

REVISION = "1d9654e78a23fb167ef539cbd699dfbd3e4830cf"


def load_reference(snapshot):
    import torch
    import transformers
    from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration

    direct = json.loads(importlib.metadata.distribution("transformers").read_text("direct_url.json"))
    if direct.get("vcs_info", {}).get("commit_id") != REVISION:
        raise ValueError("canonical Transformers revision differs")
    if importlib.util.find_spec("ignition") is not None or importlib.util.find_spec("flare_heliopause") is not None:
        raise ValueError("reference environment must exclude candidate preparation/runtime packages")
    converter_path = Path(transformers.__file__).parent / "models/qwen3_asr/convert_qwen3_asr_to_hf.py"
    spec = importlib.util.spec_from_file_location("canonical_asr_upstream_converter", converter_path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    values = upstream.clean_config(snapshot, "asr")
    generation = json.loads((snapshot / "generation_config.json").read_bytes())
    # The upstream writer supplies newer frontend/generation defaults. Use only
    # its structural conversion, retaining the original checkpoint's token roles.
    for name in ["bos_token_id", "pad_token_id", "eos_token_id"]:
        values[name] = generation.get(name)
        values["text_config"][name] = generation.get(name)
    config = Qwen3ASRConfig(**values)
    index = json.loads((snapshot / "model.safetensors.index.json").read_bytes())["weight_map"]
    renamed = upstream.convert_state_dict({name: name for name in index}, upstream.STATE_DICT_MAPPING_ASR)
    if len(index) != 708 or len(renamed) != 708 or set(renamed.values()) != set(index):
        raise ValueError("upstream conversion does not consume the complete original inventory bijectively")
    reference, loading = Qwen3ASRForConditionalGeneration.from_pretrained(snapshot,
        config=config, key_mapping=upstream.STATE_DICT_MAPPING_ASR, dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager", local_files_only=True,
        trust_remote_code=False, output_loading_info=True)
    for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]:
        if loading.get(key):
            raise ValueError(f"canonical native loading reported {key}: {loading[key]}")
    reference.eval()
    return reference, index, renamed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    reference, index, renamed = load_reference(args.checkpoint)
    actual = reference.state_dict()
    if set(actual) != set(renamed):
        raise ValueError("canonical state dictionary differs from the complete mapped original inventory")
    from contextlib import ExitStack
    with ExitStack() as owners:
        shards = {name: owners.enter_context(safe_open(args.checkpoint / name, framework="pt", device="cpu"))
            for name in sorted(set(index.values()))}
        for filename, shard in shards.items():
            if set(shard.keys()) != {name for name, file in index.items() if file == filename}:
                raise ValueError("checkpoint shard contains unindexed or missing tensors")
        for destination, original in renamed.items():
            source = shards[index[original]].get_tensor(original)
            target = actual[destination]
            if source.dtype != torch.bfloat16 or target.dtype != source.dtype or target.shape != source.shape:
                raise ValueError(f"canonical tensor shape/precision differs: {original}")
            if not torch.isfinite(source).all() or not torch.equal(
                    source.contiguous().view(torch.uint8), target.contiguous().view(torch.uint8)):
                raise ValueError(f"canonical tensor is nonfinite or differs from original: {original}")
    buffers = []
    for name, value in reference.named_buffers():
        if value.is_meta:
            raise ValueError(f"canonical buffer is unmaterialized: {name}")
        if "inv_freq" in name and value.dtype != torch.float32:
            raise ValueError(f"canonical rotary frequencies lost FP32 precision: {name}")
        buffers.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype)))
    result = dict(reference="canonical Transformers with its unmodified upstream structural converter",
        transformers_revision=REVISION, expected_checkpoint="Qwen/Qwen3-ASR-1.7B",
        expected_checkpoint_revision="7278e1e70fe206f11671096ffdd38061171dd6e5",
        checkpoint_content_identity_verified_by_this_check=False,
        original_tensors_compared=len(index), all_original_tensor_bytes_equal=True,
        loading_device="cpu", model_forward_executed=False, buffers=buffers,
        frontend_semantics="original generation token roles retained; no upstream writer/template used",
        range_limits="F<100 convolution width and SDK eager F>800 window semantics differ; separate original-format layout qualification is required")
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "buffers"}))


if __name__ == "__main__":
    main()
