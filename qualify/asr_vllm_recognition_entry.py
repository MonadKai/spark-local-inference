# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Observe actual pinned vLLM chunk IDs/stops during non-streaming qualification.

Run in the isolated reference image. The wrapper delegates every engine request
unchanged and yields the same objects. File observation affects timing: this
entry is exclusively for recognition/API diagnosis, never performance evidence.
"""

import argparse
from functools import wraps
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    args, server_arguments = parser.parse_known_args()
    import vllm
    from vllm._version import __commit_id__
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    if vllm.__version__ != "0.29.0" or __commit_id__ != "g98dff2a81" \
            or importlib.metadata.version("torch") != "2.13.0+cu130":
        raise ValueError("pinned vLLM recognition image required")
    installed = Path(vllm.__file__).parent
    paths = sorted((args.vllm_source / "vllm").rglob("*.py"))
    if not paths or not any(p.name == "async_llm.py" for p in paths):
        raise ValueError("pinned observer boundary source missing")
    for path in paths:
        if path.read_bytes() != (installed / path.relative_to(args.vllm_source / "vllm")).read_bytes():
            raise ValueError("installed vLLM differs from pinned source: " + str(path.relative_to(args.vllm_source)))
    with args.observations.open("x") as output:
        output.write("")
    with args.observations.with_name("installed-identity.json").open("x") as output:
        json.dump(dict(vllm=vllm.__version__, commit_label=__commit_id__,
            torch=importlib.metadata.version("torch"), critical_python_files_equal=len(paths),
            source_revision="98dff2a81d747d1dba01a47f939f48c3526d4206",
            scope="actual installed versions/commit label and selected Python sources; not full binary build provenance"),
            output, indent=2)

    def record(value):
        with args.observations.open("a") as output:
            output.write(json.dumps(value, ensure_ascii=False) + "\n")

    original = AsyncLLM.generate

    @wraps(original)
    async def observed(self, prompt, sampling_params, request_id, *positional, **keywords):
        if sampling_params.output_kind == RequestOutputKind.DELTA:
            raise ValueError("recognition observer requires complete cumulative chunk outputs")
        finished = False
        try:
            async for value in original(self, prompt, sampling_params, request_id, *positional, **keywords):
                if value.finished:
                    if finished:
                        raise ValueError("engine emitted duplicate terminal chunk")
                    record(dict(request_id=request_id, status="finished", finished=True,
                        prompt_ids=value.prompt_token_ids, output_kind=sampling_params.output_kind.name,
                        engine_input_kind=type(prompt).__name__,
                        multimodal_hashes=prompt.get("mm_hashes") if isinstance(prompt, dict) else None,
                        sampling_observation_scope="AsyncLLM.generate entry before InputProcessor cloning/generation-config normalization",
                        sampling=dict(temperature=sampling_params.temperature, seed=sampling_params.seed,
                            max_tokens=sampling_params.max_tokens, top_k=sampling_params.top_k,
                            top_p=sampling_params.top_p, min_p=sampling_params.min_p, n=sampling_params.n,
                            stop=sampling_params.stop, stop_token_ids=sampling_params.stop_token_ids,
                            ignore_eos=sampling_params.ignore_eos),
                        outputs=[dict(index=row.index, token_ids=list(row.token_ids), text=row.text,
                            finish_reason=row.finish_reason, stop_reason=row.stop_reason) for row in value.outputs]))
                    finished = True
                yield value
        except BaseException as error:
            record(dict(request_id=request_id, status="failed", error_type=type(error).__name__,
                terminal_observed=finished))
            raise
        finally:
            if not finished:
                record(dict(request_id=request_id, status="incomplete", finished=False))

    AsyncLLM.generate = observed
    sys.argv = ["vllm.entrypoints.openai.api_server", *server_arguments]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
