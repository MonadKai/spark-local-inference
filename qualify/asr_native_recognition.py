# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Collect native transcription responses on frozen qualification or long speech.

Run against an independently launched exported package with flare run. The
client verifies frozen uploaded PCM and actual terminal token-count headers;
it does not infer tokens by tokenizing cleaned text or identify the server's
binary/package/checkpoint. The launcher must bind those identities separately.
Request observation adds IO, so this is recognition qualification, not timing.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import wave

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pcm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("qualification", "long-speech"), default="qualification")
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    policy = dict(temperature=0, seed=0, max_model_len=8192, context="",
        language="utterance.language", max_completion_tokens=1024)
    utterances = manifest["utterances"]
    ids = [row["id"] for row in utterances]
    if args.suite == "qualification":
        if manifest["split"] != "qualification" or manifest["request"] != policy \
                or len(ids) != 256 or len(set(ids)) != 256:
            raise ValueError("unchanged 256-request frozen qualification input required")
    else:
        frozen = Path(__file__).resolve().parents[1] / "benchmarks/qwen3_asr/long-speech.json"
        if manifest_bytes != frozen.read_bytes() or manifest["request"] != policy \
                or not manifest["fixed_before_recognizer_execution"] \
                or len(ids) != 9 or len(set(ids)) != 9:
            raise ValueError("unchanged nine-request supplemental long-speech input required")
    if args.output.exists():
        raise ValueError("preserve all prior attempts; choose a fresh output")
    result = dict(scope=__doc__, manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        suite=args.suite, primary_qualification=args.suite == "qualification",
        candidate_executed=True, request=policy, cases=[], errors=[], status="running",
        generated_token_ids_observed=False, binary_package_identity_verified_by_client=False)

    def save():
        pending = args.output.with_suffix(args.output.suffix + ".pending")
        pending.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        pending.replace(args.output)

    save()
    try:
        with httpx.Client(base_url=args.base_url, timeout=1800) as client:
            models = client.get("/v1/models")
            models.raise_for_status()
            result["models"] = models.json()
            if [row["id"] for row in result["models"]["data"]] != ["Qwen/Qwen3-ASR-1.7B"]:
                raise ValueError("native server model ID differs")
            if result["models"]["data"][0].get("max_model_len") != policy["max_model_len"]:
                raise ValueError("native server context capacity differs")
            for utterance in utterances:
                row = dict(id=utterance["id"], language=utterance["language"], status="failed", text="")
                result["cases"].append(row)
                try:
                    path = args.pcm_root / utterance["path"]
                    uploaded = path.read_bytes()
                    with wave.open(io.BytesIO(uploaded), "rb") as pcm:
                        if (pcm.getnchannels(), pcm.getsampwidth(), pcm.getframerate(), pcm.getnframes()) \
                                != (1, 2, 16000, utterance["samples"]):
                            raise ValueError("upload PCM geometry differs")
                        pcm_digest = hashlib.sha256(pcm.readframes(pcm.getnframes())).hexdigest()
                    if pcm_digest != utterance["pcm_s16le_sha256"]:
                        raise ValueError("upload PCM bytes differ")
                    form = dict(model="Qwen/Qwen3-ASR-1.7B", language=utterance["language"], prompt="",
                        temperature="0", seed="0", response_format="json", max_completion_tokens="1024")
                    row.update(request_form=form, uploaded_pcm_s16le_sha256=pcm_digest,
                        uploaded_wav_sha256=hashlib.sha256(uploaded).hexdigest())
                    response = client.post("/v1/audio/transcriptions", data=form,
                        files={"file": (utterance["id"] + ".wav", uploaded, "audio/wav")})
                    row.update(http_status=response.status_code, content_type=response.headers.get("content-type"),
                        response_body=response.text, generated_tokens_header=response.headers.get("x-flare-generated-tokens"),
                        finish_reason=response.headers.get("x-flare-finish-reason"))
                    media_type = (row["content_type"] or "").partition(";")[0].strip().lower()
                    if response.status_code != 200 or media_type != "application/json":
                        raise ValueError("native transcription HTTP response failed")
                    body = response.json()
                    if not isinstance(body.get("text"), str):
                        raise ValueError("native transcription text is not a string")
                    row["text"] = body["text"]
                    if body.get("usage") != dict(type="duration", seconds=(utterance["samples"] + 15999) // 16000):
                        raise ValueError("native duration usage differs")
                    count = int(row["generated_tokens_header"])
                    if count < 1:
                        raise ValueError("missing actual generated-token count")
                    row["generated_tokens"] = count
                    if row["finish_reason"] == "length":
                        row["status"] = "truncated"
                        raise ValueError("at least one native child hit its output limit")
                    if row["finish_reason"] != "stop":
                        raise ValueError("native response lacks a natural terminal reason")
                    row["status"] = "success"
                except Exception as error:
                    row["error"] = str(error)
                    result["errors"].append(row["id"])
                save()
    finally:
        # Interrupted/not-yet-attempted members stay in the scoring denominator.
        seen = {row["id"] for row in result["cases"]}
        for utterance in utterances:
            if utterance["id"] not in seen:
                result["cases"].append(dict(id=utterance["id"], language=utterance["language"],
                    status="failed", text="", error="not attempted before client interruption"))
                result["errors"].append(utterance["id"])
        result["errors"] = [row["id"] for row in result["cases"] if row["status"] != "success"]
        result["status"] = "complete_native_recognition" if not result["errors"] else "failed"
        save()
    if result["errors"]:
        raise RuntimeError("native recognition failed; every frozen request retained")


if __name__ == "__main__":
    main()
