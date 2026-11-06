# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed compute rows preserve packed holes and retire previously live tails."""

import pytest
import torch

from ignition.components.decode_rows import PreparedDecodeRows
from ignition.components.paged_batch import Metadata


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_decode_row_padding_preserves_packing_and_retires_tails(rows):
    source = Metadata.allocate(rows, rows, 3, "cpu")
    target = Metadata.allocate(8, 8, 3, "cpu")
    operation = PreparedDecodeRows(source, target)
    for counts in (torch.ones(rows, dtype=torch.int32),
                   (torch.arange(rows) % 2).int(), torch.zeros(rows, dtype=torch.int32)):
        starts = torch.arange(rows, dtype=torch.int64) * 13
        states = torch.arange(rows, dtype=torch.int64) + 31
        pages = torch.arange(rows * 3, dtype=torch.int32).reshape(rows, 3)
        tokens = torch.arange(rows, dtype=torch.int64)[:, None] + 101
        source.set(starts, counts, states, pages, tokens, torch.tensor([rows]))
        source.sample_indices.copy_((source.query_start_loc[1:] - 1).clamp_min(0))
        # Unused tail values must not survive a previously larger invocation.
        for name in target.__dataclass_fields__:
            if name != "max_query_tokens":
                getattr(target, name).fill_(137)
        operation()
        for name in source.__dataclass_fields__:
            if name == "max_query_tokens":
                continue
            actual, expected = getattr(target, name), getattr(source, name)
            n = rows + 1 if name == "query_start_loc" else rows
            if name == "live_counts":
                assert torch.equal(actual, expected)
            else:
                assert torch.equal(actual[:n], expected)
                fill = (int(source.query_start_loc[-1]) if name == "query_start_loc"
                        else -1 if name in ("positions", "page_table", "token_page_table") else 0)
                assert actual[n:].eq(fill).all()
        active = int(counts.sum())
        assert target.token_ids[:active].tolist() == tokens[counts.bool(), 0].tolist()
        assert target.positions[active:].eq(-1).all()
        assert target.lengths.sum() == active


def test_decode_row_padding_rejects_aliases_and_multi_token_queries():
    source = Metadata.allocate(2, 2, 3, "cpu")
    with pytest.raises(ValueError, match="disjoint"):
        PreparedDecodeRows(source, source)
    with pytest.raises(ValueError, match="single-token"):
        PreparedDecodeRows(Metadata.allocate(4, 2, 3, "cpu"), Metadata.allocate(8, 8, 3, "cpu"))
