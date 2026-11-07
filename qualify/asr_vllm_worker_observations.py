# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU observations at pinned vLLM encoder dispatch boundaries.

Install after model startup through the official worker extension/RPC mechanism.
No GPU read, synchronization or file IO occurs inside an observed operation.
Returned tensors and the original call arguments are never changed. Calibrate
this instrumentation before performance use; dispatch is not a GPU busy timer.
"""

from collections import Counter
from functools import wraps


class EncoderObservations:
    def __init__(self, capacity):
        if type(capacity) is not int or not 1 <= capacity <= 100000:
            raise ValueError("bounded positive encoder observation capacity required")
        self.capacity = capacity
        self.prepared, self.executed, self.errors = Counter(), Counter(), Counter()
        self.pending = {}
        self.prepare_calls = self.execute_calls = self.successful_execute_calls = 0

    def prepare(self, value):
        self.prepare_calls += 1
        identities, inputs = value
        if not identities and not inputs:
            return
        if len(identities) != len(inputs) or any(not isinstance(identity, str)
                or not identity or len(identity) > 1024 for identity in identities) \
                or any(modality != "audio" for modality, _ in inputs):
            self.errors["invalid_encoder_dispatch"] += 1
            return
        new_ids = {identity for identity in identities if identity not in self.prepared}
        if id(inputs) in self.pending or len(self.pending) >= self.capacity \
                or len(self.prepared) + len(new_ids) > self.capacity:
            self.errors["encoder_capacity_or_duplicate_dispatch"] += 1
            return
        # Retain only the original pending list until its normal execute call.
        # Identity prevents an unrelated/recycled Python object from matching.
        self.pending[id(inputs)] = inputs, tuple(identities)
        self.prepared.update(identities)

    def begin_execute(self, inputs):
        self.execute_calls += 1
        if not inputs:
            return ()
        value = self.pending.pop(id(inputs), None)
        if value is None or value[0] is not inputs:
            self.errors["unassociated_encoder_execute"] += 1
            return None
        return value[1]

    def finish_execute(self, identities, outputs):
        self.successful_execute_calls += 1
        if identities is None:
            return
        if len(outputs) != len(identities):
            self.errors["encoder_output_count"] += 1
            return
        self.executed.update(identities)

    def snapshot(self):
        return dict(scope=__doc__, capacity=self.capacity, errors=dict(self.errors),
            prepare_calls=self.prepare_calls, execute_calls=self.execute_calls,
            successful_execute_calls=self.successful_execute_calls,
            prepared=dict(self.prepared), executed=dict(self.executed),
            pending=[list(value[1]) for value in self.pending.values()])


def wrap_prepare(original, observations):
    @wraps(original)
    def observed(*arguments, **keywords):
        try:
            value = original(*arguments, **keywords)
        except BaseException:
            observations.prepare_calls += 1
            observations.errors["prepare_exception"] += 1
            raise
        try:
            observations.prepare(value)
        except Exception:
            observations.errors["prepare_collector_exception"] += 1
        return value
    return observed


def wrap_execute(original, observations):
    @wraps(original)
    def observed(inputs, *arguments, **keywords):
        try:
            identities = observations.begin_execute(inputs)
        except Exception:
            observations.errors["execute_collector_exception"] += 1
            identities = None
        try:
            outputs = original(inputs, *arguments, **keywords)
        except BaseException:
            observations.errors["encoder_exception"] += 1
            raise
        try:
            observations.finish_execute(identities, outputs)
        except Exception:
            observations.errors["finish_collector_exception"] += 1
        return outputs
    return observed


def validate_encoder_cohort(snapshot, expected_identifiers, *, cache_mode):
    """Check actual successful encoder dispatches after the HTTP cohort drains."""
    if cache_mode not in ("disabled", "allowed"):
        raise ValueError("explicit cache condition required")
    expected_counts = Counter(expected_identifiers)
    expected = set(expected_counts)
    if not expected:
        raise ValueError("externally observed audio identifiers required")
    errors = []
    if snapshot["errors"] or snapshot["pending"]:
        errors.append("encoder_observation_failure")
    prepared, executed = snapshot["prepared"], snapshot["executed"]
    if prepared != executed:
        errors.append("incomplete_encoder_dispatch")
    if any(not set(boundary).issubset(expected) or any(
            type(count) is not int or count <= 0 for count in boundary.values())
            for boundary in (prepared, executed)):
        errors.append("unassociated_encoder_identity")
    if cache_mode == "disabled" and set(executed) != expected:
        errors.append("cold_encoder_execution_missing")
    if cache_mode == "disabled" and (prepared != expected_counts or executed != expected_counts):
        errors.append("cold_encoder_execution_multiplicity")
    return errors


class WorkerObservations:
    def install_asr_encoder_observations(self, capacity):
        if hasattr(self, "_asr_encoder_observations"):
            raise ValueError("encoder observations already installed")
        # MRV2's model state owns this runner. A different structure is a
        # failed qualification setup, never a silently unobserved fallback.
        runner = self.model_runner.model_state.encoder_runner
        observations = EncoderObservations(int(capacity))
        runner.prepare_mm_inputs = wrap_prepare(runner.prepare_mm_inputs, observations)
        runner.execute_mm_encoder = wrap_execute(runner.execute_mm_encoder, observations)
        self._asr_encoder_observations = observations
        return observations.snapshot()

    def get_asr_encoder_observations(self):
        return self._asr_encoder_observations.snapshot()
