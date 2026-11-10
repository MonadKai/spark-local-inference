# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare observed native audio-chat prompt IDs with the isolated original SDK.

This executes the original prompt builder and processor on the same PCM, on
CPU. It neither loads model weights nor establishes an HTTP/model forward.
Decoded prompt bytes are a reconstruction from observed IDs, not a server-side
capture of its renderer's string allocation.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "sdk-source", "manifest", "pcm-root", "native-observations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve the earlier reference attempt")
    import numpy as np
    import torch
    import transformers
    import qwen_asr
    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

    assert transformers.__version__ == "4.57.6"
    assert importlib.metadata.version("qwen-asr") == "0.0.6"
    assert importlib.util.find_spec("ignition") is None
    assert importlib.util.find_spec("flare_heliopause") is None
    original, installed = args.sdk_source / "qwen_asr", Path(qwen_asr.__file__).parent
    files = {p.relative_to(original) for p in original.rglob("*.py")}
    assert files and files == {p.relative_to(installed) for p in installed.rglob("*.py")}
    assert all((original / p).read_bytes() == (installed / p).read_bytes() for p in files)
    assert torch.get_default_dtype() == torch.float32 and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    observed = json.loads(args.native_observations.read_bytes())
    assert manifest["split"] == "development"
    assert observed["status"] == "complete_native_raw_token_schedule_diagnostic" and not observed["errors"]
    assert observed["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    inputs = {row["id"]: row for row in manifest["utterances"]}
    native = {row["id"]: row for row in observed["cases"]
        if row["schedule"] == "baseline" and row["observed"]}
    assert len(native) == len(observed["input_ids"]) == 8 and set(native) == set(observed["input_ids"])
    processor = Qwen3ASRProcessor.from_pretrained(args.checkpoint, local_files_only=True,
        trust_remote_code=False)
    # The prompt builder only needs its original processor. No model initializer,
    # candidate adapter or generated replacement implementation is involved.
    reference = object.__new__(Qwen3ASRModel)
    reference.processor = processor
    result = dict(scope=__doc__.strip(), status="running", cases=[], errors=[],
        expected_sdk_revision="7c6daf77a2421100f5fb066495372c00129d39ff",
        expected_checkpoint_revision="7278e1e70fe206f11671096ffdd38061171dd6e5",
        sdk_installed_python_files_equal=len(files), source_revision_verified_by_this_check=False,
        model_weights_loaded=False, model_forward_executed=False,
        native_renderer_bytes_directly_observed=False,
        native_observations_sha256=hashlib.sha256(args.native_observations.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        context="", force_language=None, tokenizer_class=type(processor.tokenizer).__name__)

    def save():
        temporary = args.output.with_suffix(".pending")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(args.output)

    save()
    try:
        for identity, observation in native.items():
            row = dict(id=identity, passed=False)
            result["cases"].append(row)
            save()
            item = inputs[identity]
            with wave.open(str(args.pcm_root / item["path"]), "rb") as audio:
                assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) \
                    == (1, 2, 16000, item["samples"])
                pcm = audio.readframes(audio.getnframes())
            assert hashlib.sha256(pcm).hexdigest() == item["pcm_s16le_sha256"]
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768)
            text = reference._build_text_prompt(context="", force_language=None)
            processed = processor(text=[text], audio=[samples], sampling_rate=16000,
                return_tensors="pt", padding=True)
            expected = processed["input_ids"][0].tolist()
            actual = json.loads(observation["response_body"])["prompt_token_ids"]
            expanded = processor.replace_multimodal_special_tokens([text], iter([expected.count(151676)]))[0]
            decoded = processor.tokenizer.decode(actual, skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            row.update(pcm_s16le_sha256=item["pcm_s16le_sha256"], sdk_prompt=text,
                sdk_expanded_prompt=expanded, native_ids=actual, sdk_ids=expected,
                sdk_frames=int(processed["feature_attention_mask"].sum().item()),
                ids_exact=actual == expected, decoded_native_prompt=decoded,
                reconstructed_bytes_exact=decoded.encode("utf-8") == expanded.encode("utf-8"))
            save()
            assert row["ids_exact"] and row["reconstructed_bytes_exact"], identity
            assert not torch.cuda.is_initialized()
            row["passed"] = True
            save()
        result["status"] = "complete_original_sdk_native_prompt_comparison"
    except BaseException as error:
        result["status"] = "failed_or_interrupted"
        result["errors"].append(dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
