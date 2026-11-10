# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate element-index bounds before compiler or GPU allocation."""

import pytest

from b12x.preparation.session import _coalesce_requests

from ignition.components.native_operator import native_plan
from ignition.components.reduction_projection import CONTRACT, Query


@pytest.mark.parametrize("columns,reduction", [(4096, 2048), (2048, 2048),
    (12288, 2048), (2048, 6144), (151936, 2048), (1048575, 2048)])
def test_projection_query_accepts_representable_extents(columns, reduction):
    query = Query(columns, reduction)
    assert max(8 * query.columns, 8 * query.reduction, query.columns * query.reduction) < 2**31


@pytest.mark.parametrize("columns,reduction", [(1048576, 2048), (1, 2**28),
    (2**28, 16), (True, 16), (8, True), (0, 16), (1, 15), (-1, 16)])
def test_projection_query_rejects_overflow_or_invalid_geometry(columns, reduction):
    with pytest.raises(ValueError, match="int32 element indexing"):
        Query(columns, reduction)


@pytest.mark.parametrize("rows", [8, 16, 40, 128, 4096, 8192])
def test_projection_rows_enter_the_prepared_query(rows):
    query = Query(151936, 2048, rows)
    assert query.rows == rows


@pytest.mark.parametrize("rows", [True, 0, -8, 4, 9, 2**24, 2**31])
def test_projection_rows_reject_invalid_or_overflowing_storage(rows):
    with pytest.raises(ValueError, match="int32 element indexing"):
        Query(151936, 2048, rows)


def test_projection_rows_reject_grid_y_overflow_before_compilation():
    assert Query(1, 16, 8 * 65535).rows == 8 * 65535
    with pytest.raises(ValueError, match="CUDA row-group bounds"):
        Query(1, 16, 8 * 65536)


def test_logical_benchmark_workloads_do_not_coalesce_at_equal_physical_geometry():
    def request(name, invocation=None):
        plan = native_plan(contract=CONTRACT, query=Query(4096, 2048), device="cpu",
            scratch_bytes=lambda tactic: 0, materialize=lambda tactic: None,
            invocation=invocation)
        return plan.request(name=name, prepare_call=lambda state: None)

    # Names alone did not distinguish the former full-eight-row trials.
    legacy = tuple(request(f"decode-{rows}") for rows in (1, 2, 4, 8))
    assert len(_coalesce_requests(legacy)) == 1
    assert all(not item.plan.invocation for item in legacy)

    current = tuple(request(f"layer-{layer}.decode-{rows}", {"benchmark_rows": rows})
        for rows in (1, 2, 4, 8) for layer in (0, 1))
    groups = _coalesce_requests(current)
    assert len(groups) == 4
    assert [len(group) for group in groups] == [2] * 4
    assert [group[0].plan.invocation.to_dict() for group in groups] == [
        {"benchmark_rows": rows} for rows in (1, 2, 4, 8)]
    assert all(item.plan.query == legacy[0].plan.query for item in current)
    assert all(not item.plan.same_declaration_except_override(legacy[0].plan) for item in current)
