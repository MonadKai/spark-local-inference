# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Observe pinned request validation and sampling normalization without inference.

Model construction here does not exercise multipart HTTP transport. In particular,
Python None and a multipart field containing the text "null" are distinct cases.
"""

import argparse
import io
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import vllm
    import pydantic, pydantic_core
    from starlette.datastructures import UploadFile
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionRequest

    if vllm.__version__ != "0.29.0":
        raise ValueError("vLLM version differs from the frozen reference")
    installed = Path(vllm.__file__).parent
    for name in ["sampling_params.py", "entrypoints/openai/chat_completion/protocol.py",
                 "entrypoints/speech_to_text/transcription/protocol.py", "entrypoints/speech_to_text/base/protocol.py"]:
        if (installed / name).read_bytes() != (args.vllm_source / "vllm" / name).read_bytes():
            raise ValueError(f"installed sampling reference differs: {name}")
    # Seed inputs include both JSON/Python values and literal form-field text.
    seeds = [None, -2**63-1, -2**63, -2, -1, 0, 1, 2**63-1, 2**63, 2**64-1,
             1.0, -1.0, 1.5, True, False, "-1", "-2", "0", "1.0", "1.5", "null", "", "  19  ",
             "+19", "0019", "-0", "+0", "1e0", "1E0", "1_0", "_10", "10_", "1__0", "1.00",
             "-2.000", "+19.000", "19.", "19.0_0", "  +0019.000  ", "١٩", "１９", "\t19\n", "0x10",
             "9223372036854775807.0", "-9223372036854775808.0", "-9223372036854775809.0",
             "9223372036854775806.0", 9223372036854774784.0, -9223372036854775808.0, 1e20]
    requests = [("omitted", {})] + [(f"seed-{index}", {"seed": value}) for index, value in enumerate(seeds)]
    requests += [(f"{name}-{index}", {name: value})
        for name, values in [("temperature", [-1, 0, .0001, .01, 1, 2, 2.001]),
                             ("top_k", [-2, -1, 0, 1, 1.0, 1.5]),
                             ("top_p", [-.001, 0, .5, 1, 1.001]),
                             ("min_p", [-.001, 0, .5, 1, 1.001])]
        for index, value in enumerate(values)]
    cases = []
    for endpoint in ("chat", "transcription"):
        for name, options in requests:
            case = dict(endpoint=endpoint, name=name, options=options)
            cases.append(case)
            source = io.BytesIO(b"not-decoded")
            try:
                if endpoint == "chat":
                    request = ChatCompletionRequest(model="Qwen/Qwen3-ASR-1.7B",
                        messages=[{"role": "user", "content": "probe"}], **({"temperature": 0} | options))
                else:
                    request = TranscriptionRequest(file=UploadFile(source, filename="probe.wav"),
                        model="Qwen/Qwen3-ASR-1.7B", **({"temperature": 0} | options))
                case.update(request_valid=True, request_seed=request.seed)
                parameters = request.to_sampling_params(1024, {})
                case.update(sampling_valid=True, seed=parameters.seed,
                    sampling_type=parameters.sampling_type.name, temperature=parameters.temperature,
                    top_k=parameters.top_k, top_p=parameters.top_p, min_p=parameters.min_p)
            except Exception as error:
                case.update(error_type=type(error).__name__, error=str(error))
                case.setdefault("request_valid", False)
                case.setdefault("sampling_valid", False)
            finally:
                source.close()
    for endpoint in ("chat", "transcription"):
        def seed_case(value):
            return next(case for case in cases if case["endpoint"] == endpoint
                and case["options"] == {"seed": value} and type(case["options"]["seed"]) is type(value))
        assert seed_case(-1)["seed"] is None
        assert seed_case(-2)["seed"] == -2
        assert seed_case(-2**63)["seed"] == -2**63
        assert seed_case(2**63-1)["seed"] == 2**63-1
        for value in (-2**63-1, 2**63, 1.5):
            assert not seed_case(value)["request_valid"]
        assert all(case["sampling_type"] == "GREEDY" for case in cases if case["endpoint"] == endpoint
                   and (case["name"] == "omitted" or case["name"].startswith("seed-")) and case["sampling_valid"])
    result = dict(expected_vllm_revision="98dff2a81d747d1dba01a47f939f48c3526d4206",
        source_revision_verified_by_this_check=False, critical_installed_files_equal=True,
        scope=__doc__.strip(), pydantic=pydantic.__version__, pydantic_core=pydantic_core.__version__,
        model_forward_executed=False, cases=cases)
    with args.output.open("x") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(dict(cases=len(cases), mandatory_seed_controls_passed=True, model_forward_executed=False)))


if __name__ == "__main__":
    main()
