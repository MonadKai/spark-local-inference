# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Immutable preparation choices must survive guard serialization by identity."""

import io
import pickle

import pytest

from ignition.components.sampling import SamplingSelection
from ignition.preparation.aot import _GuardPickler


@pytest.mark.parametrize("selection", list(SamplingSelection))
@pytest.mark.parametrize("guarded", [False, True])
def test_prepared_enum_round_trip_preserves_identity_and_branch_comparisons(selection, guarded):
    output = io.BytesIO()
    values = {id(value): value for value in vars(selection).values()} if guarded else {}
    _GuardPickler(values, {}, {}, output).dump({"selection": selection})
    restored = pickle.loads(output.getvalue())["selection"]
    assert restored is selection
    assert restored.value == selection.value
    assert (restored == SamplingSelection.CATEGORICAL) == (selection == SamplingSelection.CATEGORICAL)
