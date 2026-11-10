# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain undrained preparation resources until this process exits.

This is a fatal preparation outcome, not a recovery path. Callers must not
close, remap, clear or reuse the retained owners, even after dropping the error.
"""

from contextlib import contextmanager


def require_retention():
    from flare_heliopause import retain_undrained_preparation

    return retain_undrained_preparation


def resources_retained(error):
    pending, seen = [error], set()
    while pending:
        value = pending.pop()
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        if getattr(value, "resources_retained", False):
            return True
        pending.extend((value.__cause__, value.__context__))
        if isinstance(value, BaseExceptionGroup):
            pending.extend(value.exceptions)
    return False


def retain_undrained(error, *owners):
    # Take the native strong reference before publishing the marker. A Python
    # container or error attribute alone can disappear when a caller clears it.
    require_retention()(owners)
    error.resources_retained = True
    error.add_note("GPU drain is unconfirmed; preparation owners are retained until process exit")


def drain_or_retain(error, device, *owners):
    """Confirm the device fence, or retain every caller-owned resource root."""
    if resources_retained(error):
        retain_undrained(error, *owners)
        return False
    import torch

    try:
        torch.cuda.synchronize(device)
    except BaseException as cleanup_error:
        retain_undrained(error, *owners)
        error.add_note(f"preparation cleanup could not confirm GPU drain: {cleanup_error!r}")
        return False
    return True


@contextmanager
def retain_on_failed_drain(device, *owners):
    require_retention()
    try:
        yield
    except BaseException as error:
        drain_or_retain(error, device, *owners)
        raise


@contextmanager
def restore_backing(*pools):
    """Restore released scratch only after capture has confirmed a fence."""
    try:
        yield
    except BaseException as error:
        if resources_retained(error):
            retain_undrained(error, *pools)
        else:
            for pool in pools:
                if pool is not None:
                    try:
                        pool.materialize()
                    except BaseException as cleanup_error:
                        error.add_note(f"scratch backing restoration failed: {cleanup_error!r}")
        raise
    else:
        for pool in pools:
            if pool is not None:
                pool.materialize()
