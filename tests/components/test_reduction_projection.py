# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate element-index bounds before compiler or GPU allocation."""

import pytest

from ignition.components.reduction_projection import Query


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
