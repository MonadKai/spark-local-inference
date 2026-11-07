# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Pinned GQA split boundaries differ from explicit KV-head repetition."""

from ignition.components.request_flash import _catalog


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
