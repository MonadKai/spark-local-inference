# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Capture independent request attention with explicit split geometry."""

import math
import struct

import torch

from .paged_flash import PreparedPagedFlashAttention


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _splits(blocks, work, sms):
    # Pinned ATen FA2's occupancy heuristic uses float32 efficiencies, with a
    # double 0.85 threshold. This is preparation arithmetic, never native policy.
    if work >= _f32(_f32(.8) * sms):
        return 1
    efficiencies = []
    for split in range(1, min(128, sms, blocks) + 1):
        if split > 1 and (blocks + split - 1) // split == (blocks + split - 2) // (split - 1):
            efficiencies.append(0.)
            continue
        waves = _f32(_f32(work * split) / sms)
        efficiencies.append(_f32(waves / math.ceil(waves)))
    best = max(efficiencies)
    return next(split for split, value in enumerate(efficiencies, 1) if value >= .85 * best)


def _catalog(max_query, max_key, heads, kv_heads, sms):
    """Return device lookup data and fixed (query rows, split, key bound) calls."""
    groups, identifiers = [], {}
    table = [[-1] * ((max_key + 127) // 128 + 1)
        for _ in range((max_query + 63) // 64 + 1)]
    for row in range(len(table)):
        if row and max_query == 1:
            continue
        work = kv_heads * ((heads // kv_heads + 63) // 64) if row == 0 else heads * row
        for blocks in range(1, len(table[row])):
            split = _splits(blocks, work, sms)
            partition = 0 if split == 1 else (blocks + split - 1) // split
            key = (row == 0, split, partition)
            if key not in identifiers:
                identifiers[key] = len(groups)
                groups.append([1 if row == 0 else min(max_query, row * 64), split,
                    max_key if split == 1 else min(partition * split * 128, max_key)])
            elif row:
                groups[identifiers[key]][0] = max(groups[identifiers[key]][0], min(max_query, row * 64))
            table[row][blocks] = identifiers[key]
    return table, [value for group in groups for value in group]


def _attend_general(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            pages: torch.Tensor, lengths: torch.Tensor, cu: torch.Tensor,
            output: torch.Tensor, errors: torch.Tensor, lookup: torch.Tensor,
            plans: list[int], max_query: int) -> None:
    # Expose integer packing and bit selection to complete-forward compilation.
    # The original FA2 calls and their split/partition geometry stay unchanged.
    positions = torch.arange(query.shape[0], device=query.device, dtype=torch.int32)
    query_bits = query.view(torch.int16)
    output_bits = output.view(torch.int16)
    output_bits.zero_()
    ordered = ((cu[0] == 0) & (cu >= 0).all() & (cu <= query.shape[0]).all()
        & (cu[1:] >= cu[:-1]).all())
    errors.zero_()
    errors[0].copy_((~ordered).int())
    bad_rows = torch.zeros_like(positions, dtype=torch.bool)
    finite_query = torch.isfinite(query).all(dim=(1, 2))
    for slot in range(pages.shape[0]):
        count = cu[slot + 1] - cu[slot]
        valid = ((count > 0) & (count <= max_query) & (lengths[slot] >= count)
            & (lengths[slot] <= pages.shape[1] * key.shape[1])
            & ordered)
        page_positions = torch.arange(pages.shape[1], device=query.device, dtype=torch.int32)
        bad_pages = (((pages[slot] < 0) | (pages[slot] >= key.shape[0]))
            & (page_positions < (lengths[slot] + key.shape[1] - 1) // key.shape[1])).any()
        valid &= ~bad_pages
        valid &= ~((~finite_query) & (positions >= cu[slot]) & (positions < cu[slot + 1])).any()
        row = torch.where(count == 1, 0, (count + 63) // 64).clamp(0, lookup.shape[0] - 1)
        column = ((lengths[slot] + 127) // 128).clamp(0, lookup.shape[1] - 1)
        chosen = torch.index_select(lookup.reshape(-1), 0,
            (row * lookup.shape[1] + column).reshape(1).long())[0]
        valid &= (chosen >= 0) & (chosen < len(plans) // 3)
        bad = (count > 0) & ~valid
        row_bits = torch.zeros((max(plans[::3]), *query.shape[1:]),
            device=query.device, dtype=torch.int16)
        zero = count * 0
        for identifier, begin in enumerate(range(0, len(plans), 3)):
            capacity, split, bound = plans[begin:begin + 3]
            selected = valid & (chosen == identifier)
            indices = torch.arange(capacity, device=query.device, dtype=torch.int32)
            source_indices = cu[slot] + indices - (capacity - count)
            live = (indices >= capacity - count) & valid
            gathered = torch.index_select(query_bits, 0, source_indices.clamp(0, query.shape[0] - 1).long())
            padded = torch.where(live[:, None, None], gathered, 0).view(torch.bfloat16)
            used = torch.where(selected, lengths[slot], zero).reshape(1)
            cumulative_q = torch.stack((zero, zero + capacity))
            cumulative_k = torch.cat((zero.reshape(1), used))
            result = torch.ops.aten._flash_attention_forward(
                padded, key, value, cumulative_q, cumulative_k, capacity, bound,
                0., True, False, seqused_k=used, block_table=pages[slot:slot + 1],
                num_splits=split)[0]
            # Every physical query is valid to FA2, including dummy prefix
            # queries, so split partials and LSE are written before combination.
            # Right alignment preserves the original lower-right causal bound.
            original_order = torch.index_select(result.view(torch.int16), 0,
                (capacity - count + indices).clamp(0, capacity - 1).long())
            row_bits[:capacity].copy_(torch.where(
                ((indices < count) & selected)[:, None, None], original_order, row_bits[:capacity]))
        row_positions = torch.arange(max_query, device=query.device, dtype=torch.int32)
        bad |= ((row_positions < count) & ~torch.isfinite(row_bits.view(torch.bfloat16)).all(dim=(1, 2))).any()
        errors[slot + 1].copy_(bad.int())
        bad_rows |= (positions >= cu[slot]) & (positions < cu[slot + 1]) & bad
        restored = torch.index_select(row_bits, 0,
            (positions - cu[slot]).clamp(0, max_query - 1).long())
        output_bits.copy_(torch.where(((positions >= cu[slot]) & (positions < cu[slot + 1])
            & valid)[:, None, None], restored, output_bits))
    # Invalid active inputs must reach the caller's existing nonfinite/error
    # completion guard. Zero output would turn a metadata failure into success.
    output.masked_fill_((bad_rows | ~ordered)[:, None, None], float("nan"))


@torch.library.custom_op("flare::request_flash", mutates_args=("output", "errors"))
def _attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            pages: torch.Tensor, lengths: torch.Tensor, cu: torch.Tensor,
            output: torch.Tensor, errors: torch.Tensor, lookup: torch.Tensor,
            plans: list[int], max_query: int) -> None:
    # Retain the original opaque operation for existing compiled artifacts and
    # direct arithmetic controls. Serving graphs compile the visible program.
    _attend_general(query, key, value, pages, lengths, cu, output, errors, lookup,
        plans, max_query)


@_attend.register_fake
def _fake(query, key, value, pages, lengths, cu, output, errors, lookup, plans, max_query):
    pass



def _attend_batched(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            pages: torch.Tensor, lengths: torch.Tensor, cu: torch.Tensor,
            output: torch.Tensor, errors: torch.Tensor, lookup: torch.Tensor,
            plans: list[int], max_query: int) -> None:
    # The fixed catalog is visible to complete-forward compilation. Exported
    # execution contains CUDA nodes, with no Python dispatch in Rust serving.
    if max_query != 1:
        raise ValueError("batched request attention requires logical Q1")
    slots = pages.shape[0]
    positions = torch.arange(query.shape[0], device=query.device, dtype=torch.int32)
    query_bits, output_bits = query.view(torch.int16), output.view(torch.int16)
    output_bits.zero_()
    ordered = ((cu[0] == 0) & (cu >= 0).all() & (cu <= query.shape[0]).all()
        & (cu[1:] >= cu[:-1]).all())
    errors.zero_()
    errors[0].copy_((~ordered).int())
    counts = cu[1:] - cu[:-1]
    valid = ((counts > 0) & (counts <= max_query) & (lengths >= counts)
        & (lengths <= pages.shape[1] * key.shape[1]) & ordered)
    page_positions = torch.arange(pages.shape[1], device=query.device, dtype=torch.int32)
    bad_pages = (((pages < 0) | (pages >= key.shape[0]))
        & (page_positions[None, :] < (lengths[:, None] + key.shape[1] - 1) // key.shape[1])).any(1)
    # This program has at most one logical query per slot. Check that query
    # directly, without a slots-by-packed-rows membership matrix. Invalid
    # larger counts already fail the logical extent guard above.
    finite_query = torch.isfinite(query).all(dim=(1, 2))
    first_finite = torch.index_select(finite_query, 0,
        cu[:-1].clamp(0, query.shape[0] - 1).long())
    valid &= ~bad_pages & first_finite
    row = torch.where(counts == 1, 0, (counts + 63) // 64).clamp(0, lookup.shape[0] - 1)
    column = ((lengths + 127) // 128).clamp(0, lookup.shape[1] - 1)
    chosen = torch.index_select(lookup.reshape(-1), 0, (row * lookup.shape[1] + column).long())
    valid &= (chosen >= 0) & (chosen < len(plans) // 3)
    bad = (counts > 0) & ~valid
    row_bits = torch.zeros((slots, max(plans[::3]), *query.shape[1:]),
        device=query.device, dtype=torch.int16)
    zero = cu[0] * 0
    slot_indices = torch.arange(slots, device=query.device, dtype=torch.int32)
    for identifier, begin in enumerate(range(0, len(plans), 3)):
        capacity, split, bound = plans[begin:begin + 3]
        selected = valid & (chosen == identifier)
        indices = torch.arange(capacity, device=query.device, dtype=torch.int32)
        source_indices = cu[:-1, None] + indices[None, :] - (capacity - counts[:, None])
        live = (indices[None, :] >= capacity - counts[:, None]) & valid[:, None]
        gathered = torch.index_select(query_bits, 0,
            source_indices.clamp(0, query.shape[0] - 1).reshape(-1).long()).reshape(
                slots, capacity, *query.shape[1:])
        padded = torch.where(live[:, :, None, None], gathered, 0).view(torch.bfloat16).reshape(
            slots * capacity, *query.shape[1:])
        used = torch.where(selected, lengths, zero)
        cumulative_q = torch.arange(slots + 1, device=query.device, dtype=torch.int32) * capacity
        cumulative_k = torch.cat((zero.reshape(1), torch.cumsum(used, 0, dtype=torch.int32)))
        result = torch.ops.aten._flash_attention_forward.default(
            padded, key, value, cumulative_q, cumulative_k, capacity, bound,
            0., True, False, seqused_k=used, block_table=pages, num_splits=split)[0]
        # Fixed split geometry is still chosen per request from the original
        # catalog. A family call contains every independent request instead
        # of issuing a separate FA2 call for each slot.
        restored_indices = ((capacity - counts[:, None] + indices[None, :]).clamp(0, capacity - 1)
            + slot_indices[:, None] * capacity).reshape(-1).long()
        original_order = torch.index_select(result.view(torch.int16), 0, restored_indices).reshape(
            slots, capacity, *query.shape[1:])
        row_bits[:, :capacity].copy_(torch.where(
            ((indices[None, :] < counts[:, None]) & selected[:, None])[:, :, None, None],
            original_order, row_bits[:, :capacity]))
    row_positions = torch.arange(max_query, device=query.device, dtype=torch.int32)
    bad |= ((row_positions[None, :] < counts[:, None])
        & ~torch.isfinite(row_bits.view(torch.bfloat16)).all(dim=(2, 3))).any(1)
    errors[1:].copy_(bad.int())
    # Upper-bound search skips empty slots, so a packed row maps to its actual
    # owner even when physical request slots contain holes. Every Q1 result is
    # in that slot's first logical row, including the two-query fixed-split
    # program. Unordered metadata is masked as an error after bounded gathers.
    owner = torch.searchsorted(cu[1:], positions, right=True).clamp(0, slots - 1)
    packed = positions < cu[-1]
    restored = torch.index_select(row_bits[:, 0], 0, owner)
    selected = packed & torch.index_select(valid, 0, owner)
    output_bits.copy_(torch.where(selected[:, None, None], restored, 0))
    bad_rows = packed & torch.index_select(bad, 0, owner)
    output.masked_fill_((bad_rows | ~ordered)[:, None, None], float("nan"))


class PreparedRequestFlashAttention(PreparedPagedFlashAttention):
    """Keep request arithmetic while batching compatible Q1/GQA launches.

    All prepared split/partition groups execute. Compatible Q1/GQA rows share
    one FA2 call per family; prefill and MHA retain their original calls.
    Device lengths select their represented BF16 results by bits. Empty request slots cannot overwrite a
    packed live query. Padding never changes KV or recomputes rotary positions.
    An explicit single split uses one full-capacity family and at least two
    physical queries, preserving the general FA2 reduction across prefill
    partitions and single-query continuation. The extra query is a causal
    prefix dummy; only logical request rows are published.
    """

    def __init__(self, query, key, value, pages, lengths, cu_query, cu_key, output, *, max_query,
                 num_splits=None):
        if num_splits is not None and (type(num_splits) is not int or num_splits != 1):
            raise ValueError("request flash accepts automatic request geometry or one fixed split")
        super().__init__(query, key, value, pages, lengths, cu_query, cu_key, output, max_query=max_query)
        if query.shape[2] != 128:
            raise ValueError("request flash requires head dimension 128")
        if max_query == 1 and query.shape[1] > key.shape[2] \
                and pages.shape[0] * self.max_key > 2**31 - 1:
            raise ValueError("batched request cumulative key extents must fit int32")
        self.logical_max_query = max_query
        self.errors = torch.zeros((pages.shape[0] + 1,), dtype=torch.int32, device=query.device)
        table, self.family_geometry = _catalog(max_query, self.max_key, query.shape[1], key.shape[2],
            2 * torch.cuda.get_device_properties(query.device).multi_processor_count)
        if num_splits == 1:
            # A prefill can be partitioned differently as other requests arrive.
            # Keep one reduction geometry rather than selecting splits from the
            # current chunk size. Physical padding retains causal alignment.
            table = [[0 for _ in row] for row in table]
            self.family_geometry = [max(2, max_query), 1, self.max_key]
        self.lookup = torch.tensor(table, dtype=torch.int32, device=query.device)

    def __call__(self):
        # Pinned FA2's GQA host swap sets the batch output stride only for
        # this query geometry. Varlen prefill/MHA retain their original calls.
        operation = (_attend_batched if self.logical_max_query == 1
            and self.query.shape[1] > self.key.shape[2] else _attend_general)
        operation(self.query, self.key, self.value, self.pages, self.lengths, self.cu_query,
            self.output, self.errors, self.lookup, self.family_geometry, self.logical_max_query)
        return self.output
