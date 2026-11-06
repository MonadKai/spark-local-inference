# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical ATen flash attention over explicitly paged BF16 history."""

import torch


class PreparedPagedFlashAttention:
    """Fixed buffers and capacities; live lengths stay on the CUDA device.

    The caller supplies valid packed query/key cumulative lengths, per-request
    key lengths and page IDs. Complete-forward preparation compiles and captures
    the operation, including output-tail clearing. The native server replays its
    exported executable without invoking ATen or Python.
    """

    def __init__(self, query, key, value, pages, lengths, cu_query, cu_key, output, *, max_query):
        if (query.ndim != 3 or key.ndim != 4 or query.dtype != torch.bfloat16
                or query.device.type != "cuda" or not query.is_contiguous()
                or key.dtype != query.dtype or key.device != query.device or not key.is_contiguous()
                or value.shape != key.shape or value.dtype != key.dtype or value.device != key.device
                or not value.is_contiguous() or key.shape[1] % 256 or not key.shape[1]
                or query.shape[2] != key.shape[3] or not key.shape[2]
                or query.shape[1] % key.shape[2] or output.shape != query.shape
                or output.dtype != query.dtype or output.device != query.device or not output.is_contiguous()):
            raise ValueError("paged flash requires contiguous BF16 CUDA Q/K/V/output and 256-token cache pages")
        if (pages.ndim != 2 or not pages.shape[0] or not pages.shape[1]
                or lengths.shape != (pages.shape[0],) or cu_query.shape != (pages.shape[0] + 1,)
                or cu_key.shape != cu_query.shape or any(t.dtype != torch.int32 or t.device != query.device
                    or not t.is_contiguous() for t in (pages, lengths, cu_query, cu_key))
                or type(max_query) is not int or not 0 < max_query <= query.shape[0]
                or pages.numel() * key.shape[1] > torch.iinfo(torch.int32).max):
            raise ValueError("paged flash metadata differs from its bounded query/key capacities")
        begin, end = output.data_ptr(), output.data_ptr() + output.nbytes
        if any(begin < tensor.data_ptr() + tensor.nbytes and tensor.data_ptr() < end
                for tensor in (query, key, value, pages, lengths, cu_query, cu_key)):
            raise ValueError("paged flash output must not alias an input")
        self.query, self.key, self.value, self.output = query, key, value, output
        self.pages, self.lengths, self.cu_query, self.cu_key = pages, lengths, cu_query, cu_key
        # ATen's max_query=1 GQA shortcut reshapes as one query per request and
        # discards cu_query. Packed batches can contain empty request slots, so
        # use the ordinary varlen path even for single-token decode. Two is a
        # conservative launch capacity; the device cumulative lengths still
        # define exactly which queries exist.
        self.max_query = max(2, max_query) if query.shape[1] > key.shape[2] else max_query
        self.max_key = pages.shape[1] * key.shape[1]
        self.positions = torch.arange(query.shape[0], device=query.device, dtype=torch.int32)

    def __call__(self):
        result, _, _, _, _ = torch.ops.aten._flash_attention_forward(
            self.query, self.key, self.value, self.cu_query, self.cu_key, self.max_query, self.max_key,
            0., True, False, seqused_k=self.lengths, block_table=self.pages)
        self.output.copy_(result)
        self.output.masked_fill_((self.positions >= self.cu_query[-1])[:, None, None], 0)
        return self.output
