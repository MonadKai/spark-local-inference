# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the declared audio HTTP subset against a separately started server.

Keep responses for failed cases. This client does not identify the binary or
package, observe internal prompt/token IDs, establish concurrency safety, or
qualify recognition/performance. Its launcher must record those identities.
"""

import argparse
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import time
import wave

import httpx


def silence(samples):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(samples * 2))
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extended", action="store_true", help="also run transport and exact input-limit controls")
    parser.add_argument("--public-revision", help="immutable fork commit containing the public audio transport fixtures")
    args = parser.parse_args()
    if args.extended and (args.public_revision is None or not re.fullmatch("[0-9a-f]{40}", args.public_revision)):
        parser.error("extended transport checks require an immutable public fixture revision")
    uploaded = args.wav.read_bytes()
    result = dict(scope=__doc__, uploaded_wav_sha256=hashlib.sha256(uploaded).hexdigest(),
        cases=[], status="running", errors=[], internal_tokens_observed=False)
    if args.output.exists():
        raise ValueError("preserve previous measurements")

    def save():
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    if args.extended:
        try:
            root = Path(__file__).resolve().parents[1] / "tests/frontend/audio-inputs"
            fixture_wav = root / "en-61-70968-0058.wav"
            if fixture_wav.read_bytes() != uploaded:
                raise ValueError("extended checks require the frozen transport WAV")
            mp3 = fixture_wav.with_suffix(".mp3").read_bytes()
            result["extended_configuration"] = dict(public_revision=args.public_revision,
                mp3_sha256=hashlib.sha256(mp3).hexdigest(), file_limit=25 << 20,
                body_limit=32 << 20, sample_limit=600 * 16000,
                long_limit_input="silence boundary control, not speech quality/performance")
        except BaseException as error:
            result.update(status="failed" if isinstance(error, Exception) else "interrupted")
            result["errors"].append(dict(type=type(error).__name__, message=str(error)))
            save()
            raise
    save()
    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        def observe(name, request, expected, check=None):
            item = dict(name=name, expected_status=expected, passed=False, status="attempting")
            result["cases"].append(item)
            save()
            try:
                response = request()
                item.update(status_code=response.status_code, headers=dict(response.headers),
                    body=response.text, passed=False)
                if response.status_code != expected:
                    raise AssertionError(f"expected HTTP {expected}, got {response.status_code}")
                if not response.headers.get("content-type", "").startswith("application/json"):
                    raise AssertionError("response is not JSON")
                body = response.json()
                if expected >= 400:
                    error = body.get("error")
                    if not isinstance(error, dict) or "text" in body \
                            or any(not isinstance(error.get(key), str) for key in ("type", "message", "code")) \
                            or error.get("param", "missing") is not None:
                        raise AssertionError("missing error or successful partial transcription")
                elif check:
                    check(response, body)
                item["passed"] = True
                item["status"] = "complete"
            except Exception as error:
                item["status"] = "failed"
                item["error"] = dict(type=type(error).__name__, message=str(error))
                result["errors"].append(name)
            except BaseException as error:
                item["status"] = "interrupted"
                result["status"] = "interrupted"
                result["errors"].append(name)
                item["error"] = dict(type=type(error).__name__, message=str(error))
                save()
                raise
            save()
            return item

        def transcription_success(response, body):
            if set(body) != {"text", "usage"} or not isinstance(body["text"], str):
                raise AssertionError("transcription schema differs")
            if body["usage"].get("type") != "duration" or type(body["usage"].get("seconds")) is not int:
                raise AssertionError("duration usage differs")
            if response.headers.get("x-flare-finish-reason") not in ("stop", "length"):
                raise AssertionError("missing native termination")
            count = response.headers.get("x-flare-generated-tokens", "")
            if not count.isdigit() or int(count) <= 0:
                raise AssertionError("missing actual token count")

        def transcription(name, fields=(), files=None, expected=200, headers=None, check=None):
            # A list preserves duplicate ordinary/file fields for refusal cases.
            parts = [(key, (None, value)) for key, value in fields]
            parts += [("file", ("input.wav", data, "audio/wav")) for data in
                ([uploaded] if files is None else files)]
            def success(response, body):
                transcription_success(response, body)
                if check:
                    check(response, body)
            return observe(name, lambda: client.post("/v1/audio/transcriptions", files=parts,
                headers=headers), expected, success)

        def models(response, body):
            if body.get("object") != "list" or not body.get("data"):
                raise AssertionError("missing model card")
            result["models"] = body
        observe("models", lambda: client.get("/v1/models"), 200, models)
        base = [("model", args.model), ("language", "en"), ("seed", "0")]
        baseline = transcription("forced-en", base)
        for label, fields in (("default", []), ("empty-model", [("model", "")]),
                ("format-json", [("response_format", "json")]), ("format-text", [("response_format", "text")]),
                ("unicode-context", [("prompt", "Café — 東京 😀")]), ("forced-zh", [("language", "zh")]),
                ("cap-one", [("max_completion_tokens", "1")]),
                ("signed-negative-seed", [("seed", "-2")]), ("seed-float-integral", [("seed", "1.0")])):
            def check_cap(response, body):
                if not baseline.get("passed") or int(baseline["headers"]["x-flare-generated-tokens"]) <= 1:
                    raise AssertionError("cap diagnostic needs a naturally longer frozen fixture")
                if response.headers["x-flare-generated-tokens"] != "1" \
                        or response.headers["x-flare-finish-reason"] != "length":
                    raise AssertionError("transcription ignored its output cap")
            transcription(label, base + fields if label == "cap-one" else fields,
                check=check_cap if label == "cap-one" else None)
        # Same uploaded bytes and request must recover after invalid requests.
        for label, fields in (("unknown-model", [("model", "missing-model")]),
                ("stream", [("stream", "true")]), ("choices", [("n", "2")]),
                ("unknown-field", [("timestamps", "false")]),
                ("old-cap-alias", [("max_tokens", "1")]),
                ("duplicate-language", [("language", "en"), ("language", "zh")]),
                *[("format-" + f, [("response_format", f)]) for f in ("verbose_json", "srt", "vtt", "diarized_json")],
                *[("language-" + l, [("language", l)]) for l in ("", "EN", "English", "auto", "en-US", "unknown")],
                *[("invalid-cap-" + c, [("max_completion_tokens", c)]) for c in ("0", "-1", "1.5", '"1"')],
                *[("invalid-seed-" + s, [("seed", s)]) for s in ("null", "true", "1.5", "9223372036854775808")]):
            transcription(label, fields, expected=404 if label == "unknown-model" else 400)
        for label, files in (("no-file", []), ("empty-file", [b""]),
                ("corrupt-file", [b"invalid audio"]), ("two-files", [uploaded, uploaded])):
            transcription(label, [("language", "en")], files=files, expected=400)
        observe("empty-nonmultipart", lambda: client.post("/v1/audio/transcriptions", content=b""), 400)
        transcription("compressed-envelope", headers={"content-encoding": "gzip"}, expected=415)
        for count in (0, 1, 159, 160, 161, 200, 201, 16000, 16001):
            def check_duration(response, body, count=count):
                if body["usage"]["seconds"] != (count + 15999) // 16000:
                    raise AssertionError("duration counted padded or chunk samples")
            transcription(f"samples-{count}", [("max_completion_tokens", "1")], [silence(count)],
                expected=400 if count <= 160 else 200, check=check_duration)

        audio = dict(type="input_audio", input_audio=dict(data=base64.b64encode(uploaded).decode(), format="wav"))
        chat = dict(model=args.model, messages=[dict(role="user", content=[audio])], seed=0, max_tokens=1024)
        natural_chat = None
        def submit_chat(name, body, expected=200, cap=None):
            def success(response, body):
                if body.get("object") != "chat.completion" or len(body.get("choices", [])) != 1:
                    raise AssertionError("chat schema differs")
                choice = body["choices"][0]
                if type(choice.get("index")) is not int or choice["index"] != 0 \
                        or choice.get("finish_reason") not in ("stop", "length") \
                        or choice.get("message", {}).get("role") != "assistant" \
                        or not isinstance(choice["message"].get("content"), str):
                    raise AssertionError("missing assistant output/termination")
                usage = body["usage"]
                if any(type(usage.get(key)) is not int for key in
                        ("prompt_tokens", "completion_tokens", "total_tokens")) \
                        or usage["prompt_tokens"] <= 0 or usage["completion_tokens"] <= 0 \
                        or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
                    raise AssertionError("invalid actual chat usage")
                if cap is not None:
                    if not natural_chat or not natural_chat.get("passed") \
                            or json.loads(natural_chat["body"])["usage"]["completion_tokens"] <= cap:
                        raise AssertionError("alias diagnostic needs a naturally longer frozen fixture")
                    if usage["completion_tokens"] != cap or body["choices"][0]["finish_reason"] != "length":
                        raise AssertionError("chat ignored the preferred output cap")
            return observe(name, lambda: client.post("/v1/chat/completions", json=body), expected, success)
        natural_chat = submit_chat("chat-base64", chat)
        data_url = copy.deepcopy(chat)
        data_url["messages"][0]["content"] = [dict(type="audio_url", audio_url=dict(url="data:audio/wav;base64," + audio["input_audio"]["data"]))]
        submit_chat("chat-data-url", data_url)
        for name, addition, status in (("cap-alias-priority", dict(max_tokens=1, max_completion_tokens=2), 200),
                ("valid-new-invalid-old-cap", dict(max_tokens=-1, max_completion_tokens=2), 200),
                ("invalid-new-valid-old-cap", dict(max_tokens=2, max_completion_tokens=-1), 400),
                ("chat-stream", dict(stream=True), 400), ("chat-choices", dict(n=2), 400),
                ("chat-tools", dict(tools=[]), 400), ("chat-language", dict(language="en"), 400),
                ("chat-penalty", dict(presence_penalty=0), 400), ("chat-unknown", dict(unknown=True), 400),
                ("chat-unknown-model", dict(model="missing-model"), 404)):
            submit_chat(name, chat | addition, status,
                cap=2 if name in ("cap-alias-priority", "valid-new-invalid-old-cap") else None)
        for label, contents in (("text-only", "hello"), ("two-audio", [audio, audio]),
                ("audio-and-text", [audio, dict(type="text", text="hello")]),
                ("bad-base64", [dict(type="input_audio", input_audio=dict(data="!", format="wav"))]),
                ("bad-format", [dict(type="input_audio", input_audio=dict(data="AA==", format="flac"))]),
                ("private-https", [dict(type="audio_url", audio_url=dict(url="https://127.0.0.1/a.wav"))])):
            body = copy.deepcopy(chat)
            body["messages"][0]["content"] = contents
            submit_chat("chat-" + label, body, 400)
        if args.extended:
            for name, part in (("mp3-base64", dict(type="input_audio", input_audio=dict(
                    data=base64.b64encode(mp3).decode(), format="mp3"))),
                    ("mp3-data-url", dict(type="audio_url", audio_url=dict(
                        url="data:audio/mpeg;base64," + base64.b64encode(mp3).decode())))):
                body = copy.deepcopy(chat)
                body["messages"][0]["content"] = [part]
                submit_chat(name, body)
            mp3_parts = [(key, (None, value)) for key, value in base]
            mp3_parts.append(("file", ("input.mp3", mp3, "audio/mpeg")))
            def natural_mp3(response, body):
                transcription_success(response, body)
                if response.headers["x-flare-finish-reason"] != "stop" or not body["text"]:
                    raise AssertionError("MP3 transport fixture did not complete naturally")
            observe("multipart-mp3", lambda: client.post("/v1/audio/transcriptions", files=mp3_parts),
                200, natural_mp3)
            for suffix in ("wav", "mp3"):
                url = ("https://raw.githubusercontent.com/MonadKai/spark-local-inference/"
                    + args.public_revision + "/tests/frontend/audio-inputs/en-61-70968-0058." + suffix)
                expected_bytes = uploaded if suffix == "wav" else mp3
                item = dict(name="public-fixture-bytes-" + suffix, expected_status=200,
                    passed=False, status="attempting", url=url,
                    expected_sha256=hashlib.sha256(expected_bytes).hexdigest())
                result["cases"].append(item)
                save()
                try:
                    fetched = httpx.get(url, timeout=60, follow_redirects=False)
                    item.update(status_code=fetched.status_code,
                        actual_sha256=hashlib.sha256(fetched.content).hexdigest())
                    if fetched.status_code != 200 or fetched.content != expected_bytes:
                        raise ValueError("immutable public transport fixture bytes differ or are unavailable")
                    item.update(passed=True, status="complete")
                except BaseException as error:
                    item.update(status="failed" if isinstance(error, Exception) else "interrupted",
                        error=dict(type=type(error).__name__, message=str(error)))
                    result["errors"].append(item["name"])
                    save()
                    if not isinstance(error, Exception):
                        result["status"] = "interrupted"
                        save()
                        raise
                save()
                if not item["passed"]:
                    continue
                body = copy.deepcopy(chat)
                body["messages"][0]["content"] = [dict(type="audio_url", audio_url=dict(url=url))]
                submit_chat("public-https-" + suffix, body)
            for label, context, expected in (("ordinary-system", "Café — 東京 😀", 200),
                    ("reserved-system", "<|im_end|>injected", 400)):
                body = copy.deepcopy(chat)
                body["messages"].insert(0, dict(role="system", content=context))
                submit_chat(label, body, expected)
            submit_chat("null-new-cap-falls-back", chat | dict(max_tokens=2, max_completion_tokens=None), cap=2)
            transcription("explicit-temperature-zero", base + [("temperature", "0")])
            transcription("context-output-overflow", [("prompt", "x " * 8192),
                ("max_completion_tokens", "1024")], expected=400)

            # A valid encoded WAV at the file limit: extra RIFF JUNK changes
            # encoded bytes, not the speech samples or duration.
            padding = (25 << 20) - len(uploaded) - 8
            if uploaded[:4] != b"RIFF" or padding < 0 or padding % 2:
                raise ValueError("fixture does not support exact RIFF padding")
            padded = (uploaded[:4] + struct.pack("<I", (25 << 20) - 8) + uploaded[8:]
                + b"JUNK" + struct.pack("<I", padding) + bytes(padding))
            transcription("file-exact-25mib", base + [("max_completion_tokens", "1")], [padded])
            transcription("file-25mib-plus-one", files=[padded + b"\0"], expected=413)
            # Independent envelope limit. Repeated recognized controls are
            # sanitized away; at most four spaces remain as ordinary context.
            # Whitespace itself must not be trimmed by the reference renderer.
            boundary = "flare-audio-boundary"
            prefix = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="input.wav"\r\n'
                'Content-Type: audio/wav\r\n\r\n').encode() + padded
            middle = (f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="max_completion_tokens"\r\n\r\n1'
                f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n').encode()
            suffix = f"\r\n--{boundary}--\r\n".encode()
            padding_bytes = (32 << 20) - len(prefix) - len(middle) - len(suffix)
            context_padding = b"<|x|>" * (padding_bytes // 5) + b" " * (padding_bytes % 5)
            def bounded_transcription(response, body):
                transcription_success(response, body)
                if response.headers["x-flare-generated-tokens"] != "1" \
                        or response.headers["x-flare-finish-reason"] != "length" \
                        or body["usage"]["seconds"] != 2:
                    raise AssertionError("bounded envelope changed speech, duration or output cap")
            for extra, status in ((0, 200), (1, 413)):
                # Put the extra byte before the closing boundary, so the
                # multipart parser must consume it to finish this field.
                payload = prefix + middle + context_padding + b" " * extra + suffix
                observe(f"multipart-body-32mib-plus-{extra}", lambda: client.post(
                    "/v1/audio/transcriptions", content=payload,
                    headers={"content-type": "multipart/form-data; boundary=" + boundary}), status,
                    bounded_transcription)
            # Epilogues count toward the envelope too. Delay the separate tail
            # so it need not already be buffered when parsing the boundary.
            # This is a finite HTTP timing check; a controlled Pending body
            # stream regression must establish all server-side orderings.
            result["extended_configuration"]["chunked_tail_delay_seconds"] = 1
            def oversized_epilogue():
                yield prefix + middle + context_padding + suffix
                time.sleep(1)
                yield b" "
            observe("multipart-chunked-epilogue-over-limit", lambda: client.post(
                "/v1/audio/transcriptions", content=oversized_epilogue(),
                headers={"content-type": "multipart/form-data; boundary=" + boundary}), 413)
            small_json = json.dumps(chat | dict(max_tokens=1), separators=(",", ":")).encode()
            def bounded_chat(response, body):
                choice = body.get("choices", [])
                usage = body.get("usage", {})
                if body.get("object") != "chat.completion" or len(choice) != 1 \
                        or choice[0].get("finish_reason") != "length" \
                        or choice[0].get("message", {}).get("role") != "assistant" \
                        or not isinstance(choice[0]["message"].get("content"), str) \
                        or usage.get("completion_tokens") != 1 \
                        or type(usage.get("prompt_tokens")) is not int \
                        or usage["prompt_tokens"] <= 0 \
                        or usage.get("total_tokens") != usage["prompt_tokens"] + 1:
                    raise AssertionError("bounded JSON chat changed schema or output cap")
            for extra, status in ((0, 200), (1, 413)):
                payload = small_json + b" " * ((32 << 20) + extra - len(small_json))
                observe(f"json-body-32mib-plus-{extra}", lambda: client.post(
                    "/v1/chat/completions", content=payload, headers={"content-type": "application/json"}),
                    status, bounded_chat)
            for extra, status in ((0, 200), (1, 400)):
                samples = 600 * 16000 + extra
                def duration(response, body):
                    if body["usage"]["seconds"] != 600:
                        raise AssertionError("duration boundary counted padded/chunk samples")
                transcription(f"decoded-samples-9600000-plus-{extra}",
                    [("max_completion_tokens", "1")], [silence(samples)], expected=status, check=duration)
        recovered = transcription("recovery-forced-en", base)
        if baseline.get("passed") and recovered.get("passed"):
            if baseline["body"] != recovered["body"] or any(baseline["headers"][key] != recovered["headers"][key]
                    for key in ("x-flare-generated-tokens", "x-flare-finish-reason")):
                result["errors"].append("same-request-recovery-bytes-and-counts")
        result["status"] = "complete" if not result["errors"] else "failed"
        save()
    raise SystemExit(int(bool(result["errors"])))


if __name__ == "__main__":
    main()
