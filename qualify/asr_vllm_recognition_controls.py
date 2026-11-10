# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real reference-result gates with deliberate corruptions of saved data.

Run in the isolated CPU image after the unchanged full reference run. Mutated
records are negative controls only and never replace qualification inputs.
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "manifest", "inputs", "vllm-source", "pcm-root", "validator", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("actual_result_validator", args.validator)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    original = json.loads(args.raw.read_bytes())
    multiple = next(i for i, row in enumerate(original["requests"]) if len(row["chunks"]) == 2)
    args.output.mkdir(exist_ok=False)
    result = dict(scope=__doc__, controls=[], model_executed=False)

    def mutate(data, name):
        row = data["requests"][multiple]
        if name == "valid_completion_order":
            row["chunks"].reverse()
            row["status"] = "success"
            row.pop("error", None)
        elif name == "missing_chunk":
            row["chunks"].pop()
        elif name == "duplicate_chunk_ordinal":
            row["chunks"][1]["request_id"] = row["chunks"][0]["request_id"]
        elif name == "wrong_prompt":
            row["chunks"][0]["prompt_ids"][0] += 1
        elif name == "length_stop":
            row["chunks"][0]["outputs"][0]["finish_reason"] = "length"
        elif name == "missing_eos":
            row["chunks"][0]["outputs"][0]["token_ids"][-1] = 0
        elif name == "wrong_join":
            row["body"] = json.dumps(dict(text="intentionally incorrect join"))
        elif name == "http_error":
            row["http_status"] = 500
        elif name == "malformed_json":
            row["body"] = "{"
        elif name == "missing_chunk_identity":
            row["chunks"][0].pop("request_id")
        elif name == "malformed_chunk":
            row["chunks"][0] = None
        elif name == "delta_output":
            row["chunks"][0]["output_kind"] = "DELTA"
        elif name == "wrong_seed":
            row["chunks"][0]["sampling"]["seed"] = 1
        elif name == "wrong_form":
            row["request_form"]["prompt"] = "changed context"
        elif name == "wrong_pcm":
            row["uploaded_pcm_s16le_sha256"] = "0" * 64
        else:
            raise ValueError(name)

    for name in ("valid_completion_order", "missing_chunk", "duplicate_chunk_ordinal", "wrong_prompt",
            "length_stop", "missing_eos", "wrong_join", "http_error", "malformed_json",
            "missing_chunk_identity", "malformed_chunk", "delta_output", "wrong_seed", "wrong_form", "wrong_pcm"):
        data = copy.deepcopy(original)
        mutate(data, name)
        raw, output = args.output / (name + "-input.json"), args.output / (name + "-output.json")
        raw.write_text(json.dumps(data, ensure_ascii=False))
        sys.argv = [str(args.validator), "--raw", str(raw), "--manifest", str(args.manifest),
            "--inputs", str(args.inputs), "--vllm-source", str(args.vllm_source), "--pcm-root", str(args.pcm_root),
            "--output", str(output)]
        rejected = False
        try:
            validator.main()
        except RuntimeError as error:
            if str(error) != "vLLM reference validation failed; every request retained":
                raise
            rejected = True
        record = json.loads(output.read_bytes())
        assert len(record["cases"]) == 256
        expected_rejection = name != "valid_completion_order"
        assert rejected == expected_rejection
        assert record["cases"][multiple]["status"] == ("failed" if expected_rejection else "success")
        assert record["errors"] == ([data["requests"][multiple]["id"]] if expected_rejection else [])
        result["controls"].append(dict(name=name, rejected=rejected, all_requests_retained=True))
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
