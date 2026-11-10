# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate actual vLLM chunk completions against the frozen qualification input.

Run in the isolated CPU reference image. Engine chunks may finish out of order;
the pinned API encodes their original ordinal in each multi-chunk request ID.
Keep the original client verdict as well as the corrected ordered validation.
No hypothesis, request, corpus member or token ID is replaced or discarded.
"""

import argparse
import hashlib
import json
from pathlib import Path
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "manifest", "inputs", "vllm-source", "pcm-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    import torch
    import vllm
    from vllm.entrypoints.speech_to_text.base.serving import asr_inter_chunk_separator
    from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration as Model

    if vllm.__version__ != "0.29.0" or torch.cuda.is_initialized():
        raise ValueError("pinned CPU-only result validation required")
    for name in ("entrypoints/speech_to_text/base/serving.py", "model_executor/models/qwen3_asr.py",
            "model_executor/models/interfaces.py"):
        if (Path(vllm.__file__).parent / name).read_bytes() != (args.vllm_source / "vllm" / name).read_bytes():
            raise ValueError("installed chunk ordering/cleanup differs from pinned source")
    manifest, inputs, raw = (json.loads(p.read_bytes()) for p in (args.manifest, args.inputs, args.raw))
    ids = [row["id"] for row in manifest["utterances"]]
    request_policy = dict(temperature=0, seed=0, max_model_len=8192, context="",
        language="utterance.language", max_completion_tokens=1024)
    if manifest["split"] != "qualification" or len(ids) != 256 or len(set(ids)) != 256 \
            or inputs["manifest_sha256"] != hashlib.sha256(args.manifest.read_bytes()).hexdigest() \
            or inputs["errors"] or [row["id"] for row in inputs["cases"]] != ids \
            or [row["id"] for row in raw["requests"]] != ids \
            or inputs["status"] != "inputs_prepared_no_model_execution" \
            or inputs["model_executed"] or inputs["cuda_initialized"] \
            or inputs["vllm_revision"] != "98dff2a81d747d1dba01a47f939f48c3526d4206" \
            or inputs["checkpoint_revision"] != "7278e1e70fe206f11671096ffdd38061171dd6e5" \
            or manifest["request"] != request_policy or inputs["request"] != request_policy:
        raise ValueError("complete unchanged frozen qualification requests required")
    expected_image = "sha256:027ee28ab4fbb8b2621454e1ccacb6bac1fb3cf5577a5c301ca84637fe843ae1"
    if raw["image_id"] != expected_image or any(raw[name]["Image"] != expected_image
            for name in ("before_start", "after_ready", "before_cleanup")):
        raise ValueError("actual recognition image differs from the fixed reference")
    expected_arguments = ["--model", "/checkpoint", "--served-model-name", "Qwen/Qwen3-ASR-1.7B",
        "--revision", inputs["checkpoint_revision"], "--dtype", "bfloat16", "--host", "0.0.0.0", "--port", "8000",
        "--max-model-len", "8192", "--max-num-seqs", "8", "--max-num-batched-tokens", "4096",
        "--gpu-memory-utilization", "0.5", "--kv-cache-memory-bytes", "8589934592", "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0", "--limit-mm-per-prompt", '{"audio":1}']
    if raw["arguments"] != expected_arguments \
            or [model["id"] for model in raw["models"]["data"]] != ["Qwen/Qwen3-ASR-1.7B"]:
        raise ValueError("recognition model/compile/cache configuration differs")
    result = dict(scope=__doc__, vllm_revision="98dff2a81d747d1dba01a47f939f48c3526d4206",
        checkpoint_revision=inputs["checkpoint_revision"], manifest_sha256=inputs["manifest_sha256"],
        vllm_revision_verification_scope="selected installed Python sources; full binary build provenance not established",
        observed_request_forms_pcm_and_sampling_required=True,
        sampling_observation_scope="selected fields at AsyncLLM.generate entry; later InputProcessor cloning/generation-config normalization not observed",
        cache_keys_observed=False,
        request=inputs["request"], candidate_executed=False, cases=[], errors=[],
        status="validating", client_order_check_reinterpreted=0)
    for request, source, utterance in zip(raw["requests"], inputs["cases"], manifest["utterances"], strict=True):
        row = dict(id=utterance["id"], language=utterance["language"], status="failed",
            original_client_status=request.get("status"), original_client_error=request.get("error"),
            completion_order=[], chunks=[], request_form=request.get("request_form"),
            uploaded_pcm_s16le_sha256=request.get("uploaded_pcm_s16le_sha256"),
            uploaded_wav_sha256=request.get("uploaded_wav_sha256"),
            content_type=request.get("content_type"), response_body=request.get("body"))
        result["cases"].append(row)
        response_text = ""
        try:
            expected_form = dict(model="Qwen/Qwen3-ASR-1.7B", language=utterance["language"], prompt="",
                temperature="0", seed="0", response_format="json", max_completion_tokens="1024")
            if request.get("request_form") != expected_form:
                raise ValueError("actual multipart request parameters differ")
            path = args.pcm_root / utterance["path"]
            with wave.open(str(path), "rb") as audio:
                if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) \
                        != (1, 2, 16000, utterance["samples"]):
                    raise ValueError("actual upload PCM geometry differs")
                pcm_digest = hashlib.sha256(audio.readframes(audio.getnframes())).hexdigest()
            if pcm_digest != utterance["pcm_s16le_sha256"] \
                    or request.get("uploaded_pcm_s16le_sha256") != pcm_digest \
                    or request.get("uploaded_wav_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("actual upload bytes differ from the frozen input")
            if request.get("http_status") != 200 or not request.get("content_type", "").startswith("application/json"):
                raise ValueError("parent transcription response failed")
            response_text = json.loads(request["body"])["text"]
            if not isinstance(response_text, str):
                raise ValueError("parent transcription text is not a string")
            chunks = request.get("chunks")
            if not isinstance(chunks, list) or not source["chunks"] or len(chunks) != len(source["chunks"]) \
                    or any(not isinstance(chunk, dict) for chunk in chunks):
                raise ValueError("actual terminal chunk coverage differs")
            row["completion_order"] = [chunk.get("request_id") for chunk in chunks]
            if any(not isinstance(value, str) or not value for value in row["completion_order"]):
                raise ValueError("invalid engine chunk identity")
            reversed_order = False
            if len(chunks) > 1:
                prefixes = {chunk["request_id"].rsplit("-", 1)[0] for chunk in chunks}
                ordinals = [int(chunk["request_id"].rsplit("-", 1)[1]) for chunk in chunks]
                if len(prefixes) != 1 or sorted(ordinals) != list(range(len(chunks))):
                    raise ValueError("invalid original engine chunk identities")
                reversed_order = ordinals != sorted(ordinals)
                chunks = [chunk for _, chunk in sorted(zip(ordinals, chunks, strict=True))]
            texts = []
            for chunk, expected in zip(chunks, source["chunks"], strict=True):
                if chunk["status"] != "finished" or not chunk["finished"] \
                        or chunk["output_kind"] not in ("CUMULATIVE", "FINAL_ONLY") \
                        or chunk["prompt_ids"] != expected["prompt_ids"] or len(chunk["outputs"]) != 1:
                    raise ValueError("terminal engine chunk input differs")
                output, = chunk["outputs"]
                sampling = chunk["sampling"]
                if (sampling["temperature"], sampling["seed"], sampling["max_tokens"], sampling["n"],
                        sampling["top_k"], sampling["top_p"], sampling["min_p"], sampling["ignore_eos"],
                        sampling["stop"], sampling["stop_token_ids"]) != (0., 0, 1024, 1, 0, 1., 0., False, [], []):
                    raise ValueError("selected sampling fields at AsyncLLM.generate entry differ")
                tokens = output["token_ids"]
                if output["index"] != 0 or output["finish_reason"] != "stop" or not tokens \
                        or len(tokens) > 1024 or tokens[-1] not in (151643, 151645) \
                        or any(token in (151643, 151645) for token in tokens[:-1]):
                    raise ValueError("engine chunk failed or did not naturally stop within the cap")
                text = Model.post_process_output(output["text"])
                texts.append(text)
                row["chunks"].append(dict(index=expected["index"], prompt_ids=chunk["prompt_ids"],
                    generated_ids=tokens, generated_tokens=len(tokens), finish_reason=output["finish_reason"],
                    stop_reason=output["stop_reason"], text=text, raw_text=output["text"], sampling=sampling))
            text = asr_inter_chunk_separator(utterance["language"], Model.no_space_languages).join(texts)
            if text != response_text:
                raise ValueError("actual API text differs from ordered engine chunk cleanup/join")
            if request["status"] != "success":
                if request["status"] != "failed" or not reversed_order or request.get("error") != "":
                    raise ValueError("original client failure lacks the confirmed chunk-order explanation")
                result["client_order_check_reinterpreted"] += 1
            row.update(status="success", text=text)
        except Exception as error:
            row.update(error=str(error), text=response_text)
            result["errors"].append(row["id"])
    if torch.cuda.is_initialized():
        raise ValueError("result validation unexpectedly initialized CUDA")
    result["status"] = "complete_reference_hypotheses" if not result["errors"] else "failed"
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
        output.write("\n")
    if result["errors"]:
        raise RuntimeError("vLLM reference validation failed; every request retained")


if __name__ == "__main__":
    main()
