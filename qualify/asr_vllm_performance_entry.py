# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain bounded terminal chunk observations without IO inside the cohort.

Use only in the isolated pinned vLLM reference image. All engine arguments and
outputs are delegated unchanged. Observation overhead must be calibrated before
performance use. The launcher must drain the cohort, stop the server, bind its
image/model/configuration and reconcile HTTP IDs and official counters. This
entry alone establishes neither benchmark eligibility nor performance.
"""

import argparse
from collections import Counter
from functools import wraps
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys


class ChunkObservations:
    """Bounded host snapshots; a recording failure invalidates the entire cohort."""

    def __init__(self, capacity, token_capacity=1024):
        if type(capacity) is not int or not 1 <= capacity <= 100000 \
                or type(token_capacity) is not int or not 1 <= token_capacity <= 1024:
            raise ValueError("bounded positive chunk/token observation capacities required")
        self.capacity, self.token_capacity = capacity, token_capacity
        self.rows, self.failures = [], []
        self.mm_features = {}
        self.errors = Counter()
        self.started = self.terminal = self.closed = 0
        self.active = set()
        self.seen = set()

    def begin(self, request_id):
        self.started += 1
        if not isinstance(request_id, str) or len(request_id) > 256:
            self.errors["invalid_request_id"] += 1
            return
        if request_id in self.seen:
            self.errors["duplicate_request_id"] += 1
        if len(self.seen) >= self.capacity:
            self.errors["request_capacity_exceeded"] += 1
            return
        self.seen.add(request_id)
        self.active.add(request_id)

    def inputs(self, request):
        identity, features = request.request_id, request.mm_features
        if not isinstance(identity, str) or len(identity) > 256:
            self.errors["invalid_input_request_id"] += 1
            return
        if identity in self.mm_features:
            self.errors["duplicate_input_request_id"] += 1
            return
        if len(self.mm_features) >= self.capacity:
            self.errors["input_capacity_exceeded"] += 1
            return
        if not features or len(features) != 1:
            self.errors["audio_feature_count"] += 1
            self.mm_features[identity] = ()
            return
        feature = features[0]
        values = (feature.modality, feature.identifier, feature.mm_hash)
        if feature.modality != "audio" or any(not isinstance(value, str) or not value
                or len(value) > 1024 for value in values):
            self.errors["invalid_feature_identity"] += 1
            return
        self.mm_features[identity] = (values,)

    def finish(self, request_id, value):
        self.terminal += 1
        returned_id = value.request_id
        if returned_id != request_id:
            self.errors["returned_request_id_mismatch"] += 1
        if not isinstance(returned_id, str) or len(returned_id) > 256:
            self.errors["invalid_returned_request_id"] += 1
            returned_id = None
        cached = value.num_cached_tokens
        if cached is not None and (type(cached) is not int or cached < 0):
            self.errors["invalid_cached_count"] += 1
            cached = None
        outputs = value.outputs
        if len(outputs) != 1:
            self.errors["output_count"] += 1
            return
        row = outputs[0]
        if len(row.token_ids) > self.token_capacity:
            self.errors["token_capacity_exceeded"] += 1
            return
        if type(row.index) is not int or row.index != 0 or any(
                type(token) is not int or not 0 <= token < 151936 for token in row.token_ids):
            self.errors["invalid_output_identity"] += 1
            return
        if type(row.finish_reason) is not str or row.finish_reason not in ("stop", "length", "abort", "error", "repetition"):
            self.errors["invalid_terminal_reason"] += 1
            return
        if row.stop_reason is not None and (type(row.stop_reason) not in (int, str)
                or isinstance(row.stop_reason, str) and len(row.stop_reason) > 4096):
            self.errors["invalid_stop_reason"] += 1
            return
        if len(self.rows) >= self.capacity or not isinstance(request_id, str) or len(request_id) > 256:
            self.errors["record_capacity_exceeded"] += 1
            return
        if row.finish_reason != "stop" or not row.token_ids or row.token_ids[-1] != 151645 \
                or 151645 in row.token_ids[:-1] or row.stop_reason is not None:
            self.errors["non_natural_terminal"] += 1
        self.rows.append((request_id, row.index, tuple(row.token_ids), row.finish_reason, row.stop_reason,
                          returned_id, cached))

    def fail(self, request_id, error_type):
        self.errors["engine_exception"] += 1
        if len(self.failures) < self.capacity:
            self.failures.append((request_id if isinstance(request_id, str) and len(request_id) <= 256 else None,
                                  error_type))
        else:
            self.errors["failure_capacity_exceeded"] += 1

    def close(self, request_id, terminal):
        self.closed += 1
        if isinstance(request_id, str):
            self.active.discard(request_id)
        if not terminal:
            self.errors["incomplete_engine_request"] += 1

    def snapshot(self):
        return dict(started=self.started, terminal=self.terminal, closed=self.closed,
            active=sorted(self.active), errors=dict(self.errors),
            chunk_capacity=self.capacity, token_capacity=self.token_capacity,
            # Rows include supplied/returned IDs, actual tokens, reasons and cached count.
            rows=self.rows, failures=self.failures, mm_features=self.mm_features)


def wrap_inputs(original, observations):
    @wraps(original)
    def observed(self, *arguments, **keywords):
        request = original(self, *arguments, **keywords)
        try:
            observations.inputs(request)
        except Exception:
            observations.errors["input_collector_exception"] += 1
        return request
    return observed


def wrap_generate(original, observations, final_kind):
    """Observe outputs while closing the original generator at the same boundary."""
    @wraps(original)
    async def observed(self, prompt, sampling_params, request_id, *positional, **keywords):
        observations.begin(request_id)
        finished = False
        if sampling_params.output_kind != final_kind:
            observations.errors["non_final_output_kind"] += 1
        engine = original(self, prompt, sampling_params, request_id, *positional, **keywords)
        try:
            async for value in engine:
                if value.finished:
                    if finished:
                        observations.errors["duplicate_terminal"] += 1
                    try:
                        observations.finish(request_id, value)
                    except Exception:
                        observations.errors["collector_exception"] += 1
                    finished = True
                yield value
        except GeneratorExit:
            # Closing after a terminal value is ordinary consumer cleanup;
            # nonterminal closure is rejected by close's incomplete check.
            raise
        except BaseException as error:
            observations.fail(request_id, type(error).__name__)
            raise
        finally:
            try:
                # Exhausted generators close immediately. Early consumer closure
                # must await the original cleanup, including its abort behavior.
                await engine.aclose()
            except BaseException as error:
                observations.fail(request_id, type(error).__name__)
                raise
            finally:
                observations.close(request_id, finished)
    return observed


def validate_cohort(observations, expected_chunk_ids, *, cache_mode):
    """Reconcile a drained raw cohort against every externally planned chunk.

    This runs after sampling; it cannot establish model quality, timing validity
    or actual encoder execution. Cold mode also requires observable zero KV reuse
    and unique multimodal processor/encoder identifiers across all requests.
    """
    if cache_mode not in ("disabled", "allowed"):
        raise ValueError("explicit disabled/allowed cache condition required")
    expected = list(expected_chunk_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("nonempty unique externally planned chunk IDs required")
    errors = []
    count = len(expected)
    if observations["errors"] or observations["failures"] or observations["active"]:
        errors.append("collector_or_engine_failure")
    if any(observations[key] != count for key in ("started", "terminal", "closed")):
        errors.append("planned_chunk_count")
    rows, features = observations["rows"], observations["mm_features"]
    actual = [row[0] for row in rows]
    if len(actual) != count or len(set(actual)) != count or set(actual) != set(expected):
        errors.append("planned_chunk_identity")
    if set(features) != set(expected):
        errors.append("planned_input_identity")
    for row in rows:
        identity, index, tokens, reason, stop, returned, cached = row
        if index != 0 or identity != returned or reason != "stop" or stop is not None \
                or not tokens or len(tokens) > 1024 or tokens[-1] != 151645 \
                or 151645 in tokens[:-1]:
            errors.append("non_natural_terminal")
        if cache_mode == "disabled" and (type(cached) is not int or cached != 0):
            errors.append("cold_kv_reuse_or_unknown")
    identifiers, hashes = [], []
    for values in features.values():
        if len(values) != 1 or len(values[0]) != 3 or values[0][0] != "audio":
            errors.append("audio_feature_identity")
            continue
        identifiers.append(values[0][1]); hashes.append(values[0][2])
    if cache_mode == "disabled" and (len(set(identifiers)) != count or len(set(hashes)) != count):
        errors.append("cold_multimodal_reuse_or_unknown")
    return sorted(set(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--observation-capacity", type=int, required=True)
    args, server_arguments = parser.parse_known_args()
    # Spawned official workers inherit this explicit observation-module path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import vllm
    from vllm._version import __commit_id__
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.input_processor import InputProcessor

    if vllm.__version__ != "0.29.0" or __commit_id__ != "g98dff2a81" \
            or importlib.metadata.version("torch") != "2.13.0+cu130":
        raise ValueError("pinned vLLM reference image required")
    installed = Path(vllm.__file__).parent
    paths = sorted((args.vllm_source / "vllm").rglob("*.py"))
    required = {"v1/engine/async_llm.py", "v1/engine/input_processor.py",
        "v1/worker/worker_base.py", "v1/worker/gpu/mm/encoder_runner.py",
        "v1/worker/gpu/model_states/interface.py", "entrypoints/serve/dev/rpc/api_router.py"}
    if not paths or not required.issubset({path.relative_to(args.vllm_source / "vllm").as_posix() for path in paths}):
        raise ValueError("pinned observer boundary source missing")
    for path in paths:
        if path.read_bytes() != (installed / path.relative_to(args.vllm_source / "vllm")).read_bytes():
            raise ValueError("installed vLLM differs from pinned source: " + str(path.relative_to(args.vllm_source)))
    observations = ChunkObservations(args.observation_capacity)
    record = dict(scope=__doc__, performance_qualified=False, status="running",
        source_revision="98dff2a81d747d1dba01a47f939f48c3526d4206", vllm=vllm.__version__,
        commit_label=__commit_id__, torch=importlib.metadata.version("torch"),
        critical_python_files_equal=len(paths))
    with args.observations.open("x") as output:
        json.dump(record, output, indent=2)
    original = AsyncLLM.generate

    InputProcessor.process_inputs = wrap_inputs(InputProcessor.process_inputs, observations)
    AsyncLLM.generate = wrap_generate(original, observations, RequestOutputKind.FINAL_ONLY)
    try:
        sys.argv = ["vllm.entrypoints.openai.api_server", *server_arguments]
        runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
        record["status"] = "server_returned"
    except BaseException as error:
        record.update(status="failed_or_interrupted", server_exception=type(error).__name__)
        raise
    finally:
        record["observations"] = observations.snapshot()
        if observations.errors or observations.active or observations.started != observations.closed or observations.terminal != observations.started \
                or set(observations.mm_features) != observations.seen:

            record["status"] = "failed_observation"
        pending = args.observations.with_suffix(args.observations.suffix + ".pending")
        pending.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        pending.replace(args.observations)


if __name__ == "__main__":
    main()
