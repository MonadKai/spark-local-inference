# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Block uncertainty must not turn bracketing drift or one fast run into a win."""

from fractions import Fraction

import pytest

from qualify.asr_performance_statistics import paired_interval


def blocks(ratios):
    # Geometric reference is two; an arithmetic bracketing mean would be 4.25.
    return [dict(id=str(index), vllm_before=.5, flare=2 * value, vllm_after=8)
        for index, value in enumerate(ratios)]


def test_bracketing_geometric_mean_preserves_equal_and_constant_ratios():
    result = paired_interval(blocks([1] * 5))
    assert result["geometric_mean_ratio"] == pytest.approx(1)
    assert result["lower_95"] == pytest.approx(1)
    assert result["upper_95"] == pytest.approx(1)
    assert not result["lower_95"] > 1
    result = paired_interval(blocks([1.04] * 5))
    assert result["lower_95"] == pytest.approx(1.04)
    assert result["upper_95"] <= 1.05


@pytest.mark.parametrize("scale", [1, 2 ** 48])
def test_exact_geometric_equality_cannot_become_a_one_ulp_win(scale):
    before, candidate, after = 4863618921601801, 838919602212206, 144704202841636
    assert candidate ** 2 == before * after
    rows = [dict(id=str(index), vllm_before=before / scale,
        flare=candidate / scale, vllm_after=after / scale) for index in range(5)]
    result = paired_interval(rows)
    assert result["block_log_ratios"] == [0] * 5
    assert result["lower_95"] == result["upper_95"] == 1


def test_binary_conversion_cannot_hide_a_latency_budget_violation():
    candidate = 3.1500000000000004
    assert Fraction(candidate) / 3 > Fraction(1.05)
    rows = [dict(id=str(index), vllm_before=3, flare=candidate, vllm_after=3)
        for index in range(5)]
    result = paired_interval(rows)
    assert result["geometric_mean_ratio"] == 1.05
    assert result["upper_95"] > 1.05


def test_one_fast_block_is_inconclusive_and_order_does_not_select_it():
    rows = blocks([1, 1, 1, 1, 2])
    result = paired_interval(rows)
    assert result["geometric_mean_ratio"] == pytest.approx(1.148698354997035)
    # Independent closed form: for four zero logs and one log(2), the standard
    # error is log(2)/5. Decimal evaluation of exp(log(2)/5 * (1 +/- t4)).
    assert result["lower_95"] == pytest.approx(.7817130236513406)
    assert result["upper_95"] == pytest.approx(1.6879697163155117)
    assert result["lower_95"] < 1 < result["upper_95"]
    reversed_result = paired_interval(list(reversed(rows)))
    for name in ("geometric_mean_ratio", "lower_95", "upper_95"):
        assert result[name] == reversed_result[name]


@pytest.mark.parametrize("count", [0, 1, 3, 4, 6, 100])
def test_requests_or_selected_repeats_cannot_replace_five_blocks(count):
    with pytest.raises(ValueError, match="exactly five"):
        paired_interval(blocks([1] * count))


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), float("-inf"), True, "1",
                                 pytest.param(1 << 4096, id="overflowing_integer")])
@pytest.mark.parametrize("role", ["vllm_before", "flare", "vllm_after"])
def test_bad_measurement_invalidates_the_cell_without_dropping_a_block(bad, role):
    rows = blocks([1] * 5)
    rows[2][role] = bad
    with pytest.raises(ValueError, match="finite and positive"):
        paired_interval(rows)


@pytest.mark.parametrize("identity", ["", " ", "0", None])
def test_missing_or_duplicate_block_identity_is_rejected(identity):
    rows = blocks([1] * 5)
    rows[2]["id"] = identity
    with pytest.raises(ValueError, match="distinct nonempty"):
        paired_interval(rows)


def test_unrepresentable_interval_is_rejected_instead_of_reporting_infinite_confidence():
    rows = blocks([1e-300, 1e300, 1e-300, 1e300, 1])
    with pytest.raises(ValueError, match="representable range"):
        paired_interval(rows)
