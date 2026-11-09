# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU controls for descriptive component statistics, not service acceptance."""

import math
import pytest

from qualify.asr_projection_tuning import component_interval


def test_equal_times_cannot_become_a_strict_improvement():
    result=component_interval([(1.,1.)]*9)
    assert result["legacy_over_logical_geometric_ratio"]==1
    assert result["lower_95"]<=1<=result["upper_95"]
    assert not result["service_performance_qualified"]


def test_one_exceptional_block_is_not_a_reliable_gain():
    result=component_interval([(1.,1.)]*8+[(100.,1.)])
    assert result["legacy_over_logical_geometric_ratio"]>1
    assert result["lower_95"]<1<result["upper_95"]


def test_ratio_orientation_and_block_order_are_explicit():
    values=[(20.+i,10.+i) for i in range(9)]
    a,b=component_interval(values),component_interval(list(reversed(values)))
    assert a==b
    reverse=component_interval([(new,old) for old,new in values])
    assert a["lower_95"]>1 and reverse["upper_95"]<1
    assert math.isclose(a["legacy_over_logical_geometric_ratio"]*reverse["legacy_over_logical_geometric_ratio"],1.)


@pytest.mark.parametrize("bad", [None,[],[(1.,1.)]*8,[(1.,1.)]*8+[None],
    [(1.,1.)]*8+[(1.,)],[(1.,1.)]*8+[(1.,0.)],[(1.,1.)]*8+[(float("nan"),1.)],
    [(1.,1.)]*8+[(1.,float("inf"))],[(1.,1.)]*8+[(True,1.)]])
def test_invalid_or_missing_blocks_cannot_be_scored(bad):
    with pytest.raises(ValueError):component_interval(bad)
