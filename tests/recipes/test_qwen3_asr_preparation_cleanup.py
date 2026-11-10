# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Host exception ownership, separate from actual CUDA capture cleanup tests."""

from contextlib import contextmanager

import pytest

from ignition.recipes.qwen3_asr import bootstrap


@pytest.mark.parametrize("failure", [RuntimeError("construction failed"), KeyboardInterrupt()])
def test_failure_drains_then_closes_owners_and_transient_with_retained_traceback(monkeypatch, failure):
    events = []

    @contextmanager
    def allocator():
        events.append("enter allocator")
        try:
            yield
        finally:
            events.append("exit allocator")

    def prepare(*args, transient, failed_owners, **kwargs):
        transient.enter_context(allocator())
        failed_owners.callback(events.append, "close session")
        failed_owners.callback(events.append, "reset graph")
        raise failure

    monkeypatch.setattr(bootstrap, "_prepare", prepare)
    monkeypatch.setattr(bootstrap.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(bootstrap.torch.cuda, "synchronize", lambda device: events.append("drain"))
    with pytest.raises(type(failure)) as retained:
        bootstrap.prepare("unused", bootstrap.PreparationConfig(autotune_effort="off"), package_storage=object())
    assert retained.value is failure and retained.value.__traceback__ is not None
    assert events == ["enter allocator", "drain", "reset graph", "close session", "exit allocator"]


def test_success_exits_transient_and_transfers_owner_cleanup_to_prepared_model(monkeypatch):
    events = []
    model = object()

    @contextmanager
    def allocator():
        try:
            yield
        finally:
            events.append("exit allocator")

    def prepare(*args, transient, failed_owners, **kwargs):
        transient.enter_context(allocator())
        failed_owners.callback(events.append, "close session")
        return model

    monkeypatch.setattr(bootstrap, "_prepare", prepare)
    assert bootstrap.prepare("unused", bootstrap.PreparationConfig(autotune_effort="off"), package_storage=object()) is model
    assert events == ["exit allocator"]


def test_cleanup_failure_does_not_mask_primary_or_skip_transient_exit(monkeypatch):
    primary = RuntimeError("primary construction failure")
    events = []

    def cleanup():
        raise ValueError("session cleanup failed")

    def prepare(*args, transient, failed_owners, **kwargs):
        transient.callback(events.append, "exit allocator")
        failed_owners.callback(cleanup)
        raise primary

    monkeypatch.setattr(bootstrap, "_prepare", prepare)
    monkeypatch.setattr(bootstrap.torch.cuda, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError) as retained:
        bootstrap.prepare("unused", bootstrap.PreparationConfig(autotune_effort="off"), package_storage=object())
    assert retained.value is primary and events == ["exit allocator"]
    assert any("session cleanup failed" in note for note in primary.__notes__)
