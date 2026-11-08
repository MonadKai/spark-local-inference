# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Pinned GQA split boundaries differ from explicit KV-head repetition."""

import pytest

from ignition.components.request_flash import _catalog, _sequence_catalog


@pytest.mark.parametrize("capacity", [256, 8192])
@pytest.mark.parametrize("heads,kv_heads", [(16, 8), (8, 8)])
def test_full_sequence_catalog_preserves_every_reachable_original_family(capacity, heads, kv_heads):
    original, plans = _catalog(capacity, capacity, heads, kv_heads, 96)
    selected, compact = _sequence_catalog(capacity, heads, kv_heads, 96)
    covered = set()
    for count in range(1, capacity + 1):
        row, column = (0 if count == 1 else (count + 63) // 64), (count + 127) // 128
        old, new = original[row][column], selected[row][column]
        assert old >= 0 and new >= 0
        assert plans[old * 3:old * 3 + 3] == compact[new * 3:new * 3 + 3]
        covered.add(new)
    assert covered == set(range(len(compact) // 3))
    assert len(compact) < len(plans)


def test_native_gqa_and_square_prefill_boundaries():
    lookup, flat = _catalog(4096, 8192, 16, 8, 96)
    plans = [flat[i:i + 3] for i in range(0, len(flat), 3)]
    assert len(plans) == 104
    for blocks, expected in [(1, 1), (6, 6), (7, 7), (8, 8), (12, 12), (64, 10)]:
        capacity, split, bound = plans[lookup[0][blocks]]
        assert capacity == 1 and split == expected
        if split > 1:
            assert (bound // 128 + split - 1) // split == (blocks + split - 1) // split
        else:
            assert bound == 8192
    # Q165/K165 needs two splits; Q443/K443 needs one. The query
    # capacity is independent of the original live count in each case.
    assert plans[lookup[3][2]][1] == 2
    assert plans[lookup[7][4]][1] == 1
    single, flat_single = _catalog(1, 8192, 16, 8, 96)
    assert len(flat_single) // 3 == 31
    assert all(identifier >= 0 for identifier in single[0][1:])
