# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Pad packed single-token decode metadata without adding live requests."""


class PreparedDecodeRows:
    """Give model arithmetic fixed rows while preserving logical batch ownership.

    The source already packs live tokens, including batches with inactive holes.
    Copy that packing unchanged; appending empty request and token tails must not
    interpret packed token offsets as request-row indices. Sampling and state
    completion continue to consume the original metadata.
    """

    def __init__(self, source, target):
        rows, capacity = source.lengths.numel(), target.lengths.numel()
        if (source.max_query_tokens != 1 or target.max_query_tokens != 1
                or source.token_ids.numel() != rows or target.token_ids.numel() != capacity
                or not 1 <= rows <= capacity
                or source.sample_indices.numel() != rows or target.sample_indices.numel() != capacity
                or source.page_table.shape[1] != target.page_table.shape[1]):
            raise ValueError("decode row padding requires bounded single-token metadata")
        for name in source.__dataclass_fields__:
            if name == "max_query_tokens":
                continue
            a, b = getattr(source, name), getattr(target, name)
            if a.dtype != b.dtype or a.device != b.device or a.data_ptr() == b.data_ptr():
                raise ValueError("decode row padding requires disjoint matching metadata")
        self.source, self.target, self.rows = source, target, rows

    def __call__(self):
        source, target, rows = self.source, self.target, self.rows
        for name in ("token_ids", "sample_indices", "state_rows", "starts", "lengths", "requests"):
            value = getattr(target, name)
            value.zero_()
            value[:rows].copy_(getattr(source, name))
        for name in ("positions", "page_table", "token_page_table"):
            value = getattr(target, name)
            value.fill_(-1)
            value[:rows].copy_(getattr(source, name))
        target.query_start_loc[:rows + 1].copy_(source.query_start_loc)
        target.query_start_loc[rows + 1:].copy_(source.query_start_loc[-1])
        target.live_counts.copy_(source.live_counts)
        return target
