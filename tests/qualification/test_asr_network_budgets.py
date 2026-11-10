# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from qualify.asr_freeze_numerics import check, scalar_budget, validate_case_layout, validate_identical_features


def test_relative_budget_uses_canonical_denominator_and_bf16_grid_at_the_reference():
    reference = torch.tensor([2., 4.], dtype=torch.bfloat16)
    independent = torch.tensor([1., 2.], dtype=torch.bfloat16)
    budget = scalar_budget(reference, independent)
    assert budget["independent_relative_l2"] == .5
    assert budget["grid_step_max"] == .03125
    assert budget["grid_step_relative_l2"] == .0078125
    assert budget["max_abs"] == 2.03125
    assert budget["relative_l2"] == .5078125
    check(reference, reference, budget)
    check(independent, reference, budget)
    with pytest.raises(ValueError, match="budget exceeded"):
        check(-reference, reference, budget)


def test_zero_probe_retains_zero_budget_and_rejects_even_one_subnormal():
    reference = torch.zeros(2, dtype=torch.bfloat16)
    budget = scalar_budget(reference, reference)
    assert budget["max_abs"] == budget["relative_l2"] == 0
    check(reference, reference, budget)
    changed = reference.clone()
    changed[0] = 2. ** -133
    with pytest.raises(ValueError, match="budget exceeded"):
        check(changed, reference, budget)


@pytest.mark.parametrize("value", [torch.nan, torch.inf, -torch.inf])
def test_nonfinite_among_valid_values_cannot_enter_or_pass_a_network_budget(value):
    reference = torch.ones(4, dtype=torch.bfloat16)
    changed = reference.clone()
    changed[0] = value
    with pytest.raises(ValueError, match="nonfinite"):
        scalar_budget(reference, changed)
    with pytest.raises(ValueError, match="nonfinite"):
        check(changed, reference, scalar_budget(reference, reference))


def test_fp32_logits_cannot_bypass_the_declared_bf16_node_even_if_values_match():
    reference = torch.ones((2, 8), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="dtype/geometry"):
        check(reference.float(), reference, scalar_budget(reference, reference))


@pytest.mark.parametrize("frames", [2, 13, 99])
def test_short_original_layout_requires_the_explicit_width_adapter(frames):
    case = {"frames": frames}
    valid = dict(layout="original", short_width_adapter=
        "conv2d1 pre-hook slices padded input to true F; no arithmetic replacement")
    validate_case_layout(valid, case, case)
    for changed in ({}, dict(valid, layout="canonical"), dict(valid, short_width_adapter="")):
        with pytest.raises(ValueError, match="true-width"):
            validate_case_layout(changed, case, case)
    with pytest.raises(ValueError, match="frame counts"):
        validate_case_layout(valid, case, {"frames": frames + 1})


def test_different_represented_input_cannot_inflate_a_model_stage_budget():
    reference = torch.ones((1, 1, 128, 100), dtype=torch.bfloat16)
    validate_identical_features(reference, reference.clone())
    changed = reference.clone()
    changed[0, 0, 0, 0] = 1.0078125
    with pytest.raises(ValueError, match="byte-identical"):
        validate_identical_features(reference, changed)
