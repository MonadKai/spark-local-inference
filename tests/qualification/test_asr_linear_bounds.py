# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from qualify.asr_linear_bounds import bf16_rounding_sup, envelope, result_ratio


def test_interval_bound_covers_a_crossed_binade_where_centre_half_ulp_fails():
    centre = torch.tensor(1.999, dtype=torch.float64)
    radius = torch.tensor(.01, dtype=torch.float64)
    midpoint = torch.tensor(2.0078125, dtype=torch.float64)
    error = (midpoint.bfloat16().double() - midpoint).abs()
    assert error > bf16_rounding_sup(centre)
    assert error <= bf16_rounding_sup(centre.abs() + radius)


def test_interval_bound_handles_zero_subnormal_and_negative_intervals():
    values = torch.tensor([0., 2. ** -140, 2. ** -126], dtype=torch.float64)
    assert torch.equal(bf16_rounding_sup(values), torch.full_like(values, 2. ** -134))
    with pytest.raises(ValueError): bf16_rounding_sup(torch.tensor([-1.], dtype=torch.float64))
    with pytest.raises(ValueError): bf16_rounding_sup(torch.tensor([torch.inf], dtype=torch.float64))


def test_linear_bias_bound_detects_rounding_the_product_before_adding_bias():
    source = torch.tensor([[1., 1.]], dtype=torch.bfloat16)
    weight = torch.tensor([[1., 2. ** -10]], dtype=torch.bfloat16)
    bias = torch.tensor([-1.], dtype=torch.bfloat16)
    exact, bound, _, _ = envelope(source, weight, bias)
    assert exact.item() == 2. ** -10
    correct = exact.bfloat16().double()
    wrong = ((exact - bias.double()).bfloat16() + bias).double()
    assert ((correct-exact).abs() <= bound).all()
    assert ((wrong-exact).abs() > bound).all()


def test_linear_envelope_refuses_unmodelled_fp32_product_underflow():
    with pytest.raises(ValueError, match="underflow"):
        envelope(torch.tensor([[2. ** -100]], dtype=torch.bfloat16),
            torch.tensor([[2. ** -50]], dtype=torch.bfloat16), None)


def test_linear_envelope_refuses_denormal_operands_even_when_the_product_is_normal():
    with pytest.raises(ValueError, match="subnormal BF16 operands"):
        envelope(torch.tensor([[2. ** -130]], dtype=torch.bfloat16),
            torch.tensor([[2. ** 10]], dtype=torch.bfloat16), None)


def test_normal_products_cancelling_to_subnormal_admit_the_declared_ftz_class():
    source = torch.tensor([[2. ** -63, 2. ** -63]], dtype=torch.bfloat16)
    weight = torch.tensor([[2. ** -63, -(1 + 2. ** -7) * 2. ** -63]], dtype=torch.bfloat16)
    exact, bound, _, _ = envelope(source, weight, None)
    assert exact.item() == -(2. ** -133)
    assert result_ratio(exact.bfloat16(), exact, bound) <= 1
    assert result_ratio(torch.zeros_like(exact, dtype=torch.bfloat16), exact, bound) <= 1


def test_linear_bound_refuses_possible_fp32_partial_sum_overflow():
    with pytest.raises(ValueError, match="partial sums might overflow"):
        envelope(torch.tensor([[2. ** 64]], dtype=torch.bfloat16),
            torch.tensor([[2. ** 64]], dtype=torch.bfloat16), None)


def test_linear_bound_refuses_bf16_output_interval_overflow():
    with pytest.raises(ValueError, match="overflow intervals"):
        envelope(torch.tensor([[torch.finfo(torch.bfloat16).max]], dtype=torch.bfloat16),
            torch.tensor([[1.]], dtype=torch.bfloat16), None)


@pytest.mark.parametrize("invalid", [torch.nan, torch.inf, -torch.inf])
def test_output_gate_rejects_nonfinite_among_otherwise_valid_elements(invalid):
    source = torch.tensor([[1., 1.]], dtype=torch.bfloat16)
    weight = torch.tensor([[1., 1.], [1., -1.]], dtype=torch.bfloat16)
    exact, bound, _, _ = envelope(source, weight, None)
    output = exact.bfloat16()
    assert result_ratio(output, exact, bound) <= 1
    output[0, 1] = invalid
    with pytest.raises(ValueError, match="nonfinite"):
        result_ratio(output, exact, bound)
