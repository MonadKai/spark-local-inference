# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU dispatch controls only; real worker/GPU integration is separate."""

from types import SimpleNamespace

import pytest

from qualify.asr_vllm_worker_observations import EncoderObservations, WorkerObservations, validate_encoder_cohort, wrap_execute, wrap_prepare


def test_observed_encoder_preserves_actual_objects_arguments_and_counts_items():
    observations = EncoderObservations(2)
    inputs = [("audio", object()), ("audio", object())]
    prepared = (["first", "second"], inputs)
    outputs = [object(), object()]
    marker = object()
    def prepare(*args, **kwargs):
        assert args == (marker,) and kwargs == {"option": 7}
        return prepared
    def execute(value, *args, **kwargs):
        assert value is inputs and args == (marker,) and kwargs == {"option": 9}
        return outputs
    assert wrap_prepare(prepare, observations)(marker, option=7) is prepared
    assert observations.pending and not observations.executed
    assert wrap_execute(execute, observations)(inputs, marker, option=9) is outputs
    assert observations.snapshot()["executed"] == {"first": 1, "second": 1}
    assert not observations.pending and not observations.errors
    assert observations.successful_execute_calls == 1


def test_recycled_equivalent_list_missing_execution_and_failed_execution_cannot_pass():
    observations = EncoderObservations(2)
    inputs = [("audio", object())]
    observations.prepare((["id"], inputs))
    assert observations.begin_execute(list(inputs)) is None
    assert observations.errors["unassociated_encoder_execute"] == 1 and observations.pending
    def fail(value):
        assert value is inputs
        raise RuntimeError("real encoder failure")
    with pytest.raises(RuntimeError, match="real encoder failure"):
        wrap_execute(fail, observations)(inputs)
    assert observations.errors["encoder_exception"] == 1 and not observations.executed


def test_capacity_modality_and_output_count_fail_without_altering_original_execution():
    observations = EncoderObservations(1)
    first, second = [("audio", object())], [("audio", object())]
    observations.prepare((["first"], first))
    observations.prepare((["second"], second))
    observations.prepare((["image"], [("image", object())]))
    assert len(observations.prepared) == len(observations.pending) == 1
    result = []
    assert wrap_execute(lambda value: result, observations)(first) is result
    assert observations.errors["encoder_output_count"] == 1 and not observations.executed
    assert observations.errors["encoder_capacity_or_duplicate_dispatch"] == 1
    assert observations.errors["invalid_encoder_dispatch"] == 1


def test_extension_binds_exact_model_state_runner_and_rejects_double_install():
    runner = SimpleNamespace(prepare_mm_inputs=lambda value: ([], []), execute_mm_encoder=lambda value: [])
    worker = WorkerObservations()
    worker.model_runner = SimpleNamespace(model_state=SimpleNamespace(encoder_runner=runner))
    assert not worker.install_asr_encoder_observations("2")["prepared"]
    assert runner.prepare_mm_inputs({}) == ([], []) and runner.execute_mm_encoder([]) == []
    assert worker.get_asr_encoder_observations()["successful_execute_calls"] == 1
    with pytest.raises(ValueError, match="already installed"):
        worker.install_asr_encoder_observations("2")
    missing = WorkerObservations(); missing.model_runner = SimpleNamespace()
    with pytest.raises(AttributeError):
        missing.install_asr_encoder_observations("2")


def test_actual_dispatch_qualification_rejects_missing_execution_and_pending_work():
    store = EncoderObservations(2)
    inputs = [("audio", object())]
    store.prepare((["a"], inputs))
    failures = validate_encoder_cohort(store.snapshot(), ["a", "b"], cache_mode="disabled")
    assert "encoder_observation_failure" in failures
    assert "incomplete_encoder_dispatch" in failures
    assert "cold_encoder_execution_missing" in failures
    identities = store.begin_execute(inputs); store.finish_execute(identities, [object()])
    assert not validate_encoder_cohort(store.snapshot(), ["a"], cache_mode="disabled")
    assert "cold_encoder_execution_missing" in validate_encoder_cohort(store.snapshot(), ["a", "b"], cache_mode="disabled")
    assert not validate_encoder_cohort(store.snapshot(), ["a", "b"], cache_mode="allowed")
    assert "unassociated_encoder_identity" in validate_encoder_cohort(store.snapshot(), ["b"], cache_mode="allowed")


def test_preparation_exception_is_retained_and_propagated_unchanged():
    observations = EncoderObservations(1)
    failure = RuntimeError("actual preparation failure")
    def fail(value):
        raise failure
    with pytest.raises(RuntimeError) as caught:
        wrap_prepare(fail, observations)({})
    assert caught.value is failure
    assert observations.prepare_calls == observations.errors["prepare_exception"] == 1
    assert not observations.pending and not observations.prepared
