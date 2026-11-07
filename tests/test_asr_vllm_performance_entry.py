# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Recording contracts only; these do not execute or qualify vLLM inference."""

from types import SimpleNamespace

import pytest

from qualify.asr_vllm_performance_entry import ChunkObservations, validate_cohort, wrap_generate, wrap_inputs


def output(tokens=None, reason="stop", stop=None):
    return SimpleNamespace(index=0, token_ids=[4, 151645] if tokens is None else tokens,
        finish_reason=reason, stop_reason=stop)


def packet(identity, row=None, cached=0):
    return SimpleNamespace(request_id=identity, outputs=[output() if row is None else row],
        num_cached_tokens=cached, finished=True)


def test_out_of_order_snapshots_preserve_actual_ids_and_remain_independent():
    store = ChunkObservations(2)
    store.begin("parent-0"); store.begin("parent-1")
    ids = [7, 151645]
    store.finish("parent-1", packet("parent-1", output(ids))); store.close("parent-1", True)
    ids[0] = 999
    store.finish("parent-0", packet("parent-0")); store.close("parent-0", True)
    result = store.snapshot()
    assert result["rows"] == [("parent-1", 0, (7, 151645), "stop", None, "parent-1", 0),
                              ("parent-0", 0, (4, 151645), "stop", None, "parent-0", 0)]
    assert result["started"] == result["terminal"] == result["closed"] == 2
    assert result["active"] == [] and not result["errors"]


def test_failed_and_incomplete_requests_are_not_counted_as_terminal_successes():
    store = ChunkObservations(2)
    store.begin("failed"); store.fail("failed", "EngineDeadError"); store.close("failed", False)
    store.begin("unfinished")
    result = store.snapshot()
    assert result["started"] == 2 and result["terminal"] == 0 and result["closed"] == 1
    assert result["active"] == ["unfinished"]
    assert result["failures"] == [("failed", "EngineDeadError")]
    assert result["errors"] == {"engine_exception": 1, "incomplete_engine_request": 1}


def test_capacity_excess_and_duplicate_requests_invalidate_without_unbounded_storage():
    store = ChunkObservations(1)
    for identity in ("first", "first", "excess"):
        store.begin(identity); store.finish(identity, packet(identity)); store.close(identity, True)
    result = store.snapshot()
    assert len(result["rows"]) == len(store.seen) == 1
    assert result["started"] == result["closed"] == 3
    assert result["errors"]["duplicate_request_id"] == 1
    assert result["errors"]["request_capacity_exceeded"] == 2
    assert result["errors"]["record_capacity_exceeded"] == 2


@pytest.mark.parametrize("tokens", [[4, 5, 151645], [-1], [151936], [True]])
def test_oversized_or_invalid_tokens_cannot_become_eligible_records(tokens):
    store = ChunkObservations(1, token_capacity=2)
    store.begin("invalid"); store.finish("invalid", packet("invalid", output(tokens))); store.close("invalid", True)
    assert not store.rows and store.errors


def test_length_stop_is_retained_for_the_cohort_eligibility_check():
    store = ChunkObservations(1)
    store.begin("truncated"); store.finish("truncated", packet("truncated", output([4, 5], "length", None)))
    store.close("truncated", True)
    assert store.rows == [("truncated", 0, (4, 5), "length", None, "truncated", 0)]
    assert store.errors["non_natural_terminal"] == 1


@pytest.mark.parametrize("capacity", [0, -1, 100001, True])
def test_record_capacity_must_be_explicit_and_bounded(capacity):
    with pytest.raises(ValueError):
        ChunkObservations(capacity)


def test_returned_id_mismatch_and_unknown_cache_observation_are_explicit():
    store = ChunkObservations(1)
    store.begin("input"); store.finish("input", packet("other", cached=None)); store.close("input", True)
    assert store.errors["returned_request_id_mismatch"] == 1
    assert store.rows[0][-2:] == ("other", None)


def test_consumer_close_awaits_original_cleanup_without_changing_yielded_object():
    import asyncio

    async def run():
        store, events = ChunkObservations(1), []
        value = packet("id")
        async def original(self, prompt, sampling, identity, *args, **kwargs):
            assert (self, prompt, identity, args, kwargs) == ("engine", "prompt", "id", (7,), {"option": 3})
            try:
                yield value
            finally:
                await asyncio.sleep(0)
                events.append("closed-original")
        generator = wrap_generate(original, store, "final")("engine", "prompt",
            SimpleNamespace(output_kind="final"), "id", 7, option=3)
        assert await anext(generator) is value
        await generator.aclose()
        assert events == ["closed-original"] and store.closed == 1 and not store.active
        assert not store.errors
    asyncio.run(run())


def test_cancellation_drains_original_and_retains_incomplete_failure():
    import asyncio

    async def run():
        store, entered, closed = ChunkObservations(1), asyncio.Event(), asyncio.Event()
        async def original(*args, **kwargs):
            try:
                entered.set()
                await asyncio.Event().wait()
                yield packet("id")
            finally:
                closed.set()
        generator = wrap_generate(original, store, "final")(None, None,
            SimpleNamespace(output_kind="final"), "id")
        task = asyncio.create_task(anext(generator))
        await entered.wait(); task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set() and store.closed == 1 and not store.active
        assert store.errors["incomplete_engine_request"] == 1
    asyncio.run(run())


def test_input_observer_preserves_returned_object_and_actual_cache_identifiers():
    store = ChunkObservations(2)
    request = SimpleNamespace(request_id="id", mm_features=[SimpleNamespace(
        modality="audio", identifier="renderer0-mm-42-audio-0", mm_hash="renderer0-mm-42-audio-0")])
    marker = object()
    def original(self, *arguments, **keywords):
        assert self is marker and arguments == (7,) and keywords == {"value": 9}
        return request
    observed = wrap_inputs(original, store)
    assert observed(marker, 7, value=9) is request
    assert store.snapshot()["mm_features"] == {"id": (("audio", "renderer0-mm-42-audio-0", "renderer0-mm-42-audio-0"),)}
    assert not store.errors
    assert observed(marker, 7, value=9) is request
    assert store.errors["duplicate_input_request_id"] == 1


def test_input_identity_snapshot_rejects_missing_audio_and_remains_bounded():
    store = ChunkObservations(1)
    store.inputs(SimpleNamespace(request_id="id", mm_features=[]))
    store.inputs(SimpleNamespace(request_id="excess", mm_features=[]))
    assert store.snapshot()["mm_features"] == {"id": ()}
    assert store.errors == {"audio_feature_count": 1, "input_capacity_exceeded": 1}


def complete_cohort():
    store = ChunkObservations(2)
    for identity in ("a", "b"):
        store.begin(identity)
        store.inputs(SimpleNamespace(request_id=identity, mm_features=[SimpleNamespace(
            modality="audio", identifier="mm-" + identity, mm_hash="hash-" + identity)]))
        store.finish(identity, packet(identity)); store.close(identity, True)
    return store.snapshot()


def test_post_cohort_validation_requires_all_external_attempts_and_cache_observations():
    snapshot = complete_cohort()
    assert validate_cohort(snapshot, ["a", "b"], cache_mode="disabled") == []
    assert "planned_chunk_count" in validate_cohort(snapshot, ["a", "b", "failed-before-engine"], cache_mode="disabled")
    with pytest.raises(ValueError):
        validate_cohort(snapshot, [], cache_mode="disabled")
    with pytest.raises(ValueError):
        validate_cohort(snapshot, ["a", "a"], cache_mode="allowed")


@pytest.mark.parametrize("cached", [None, 39])
def test_cold_cache_unknown_positive_and_reused_mm_identity_are_rejected(cached):
    snapshot = complete_cohort()
    row = list(snapshot["rows"][0]); row[-1] = cached; snapshot["rows"][0] = row
    snapshot["mm_features"]["b"] = snapshot["mm_features"]["a"]
    errors = validate_cohort(snapshot, ["a", "b"], cache_mode="disabled")
    assert "cold_kv_reuse_or_unknown" in errors
    assert "cold_multimodal_reuse_or_unknown" in errors
    assert not validate_cohort(snapshot, ["a", "b"], cache_mode="allowed")


@pytest.mark.parametrize("stop", [151645, "custom-stop"])
def test_nondefault_stop_reason_is_retained_and_rejected(stop):
    store = ChunkObservations(1)
    store.begin("id"); store.finish("id", packet("id", output(stop=stop))); store.close("id", True)
    assert store.rows[0][4] == stop and store.errors["non_natural_terminal"] == 1


def test_engine_error_duplicate_terminal_and_early_close_keep_original_behavior():
    import asyncio

    async def run():
        async def execute(original, *, early=False):
            store = ChunkObservations(2)
            generator = wrap_generate(original, store, "final")(None, None,
                SimpleNamespace(output_kind="final"), "id")
            if early:
                await anext(generator); await generator.aclose()
            else:
                with pytest.raises(RuntimeError, match="original failure"):
                    async for _ in generator:
                        pass
            assert not store.active and store.closed == 1
            return store
        async def failing(*args, **kwargs):
            raise RuntimeError("original failure")
            yield
        failed = await execute(failing)
        assert failed.errors["engine_exception"] == failed.errors["incomplete_engine_request"] == 1
        async def duplicate(*args, **kwargs):
            yield packet("id"); yield packet("id")
            raise RuntimeError("original failure")
        duplicated = await execute(duplicate)
        assert duplicated.errors["duplicate_terminal"] == 1 and duplicated.terminal == 2
        async def unfinished(*args, **kwargs):
            value = packet("id"); value.finished = False
            yield value
        incomplete = await execute(unfinished, early=True)
        assert incomplete.errors["incomplete_engine_request"] == 1
        async def cleanup_error(*args, **kwargs):
            try:
                yield packet("id")
            finally:
                raise RuntimeError("original failure")
        broken = await execute(cleanup_error)
        assert broken.errors["engine_exception"] >= 1
    asyncio.run(run())
