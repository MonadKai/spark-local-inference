# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""BF16 full-attention pages and complete per-request continuation state."""

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

PAGE_TOKENS = 256


def page_bytes():
    return 28 * 2 * PAGE_TOKENS * 8 * 128 * 2


def page_specs():
    # PagedStateStorage allocates byte planes, including the BF16 element size.
    for layer in range(28):
        for field in ("key", "value"):
            yield f"layer{layer}.{field}", (PAGE_TOKENS * 8 * 128 * 2,)


def continuation_layout():
    return StateLayout.from_specs([
        *((name, shape, torch.uint8) for name, shape in page_specs()),
        ("last_hidden", (2048,), torch.bfloat16),
        ("consumed_tokens", (1,), torch.int64),
    ])


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, slots, pages, device, **kwargs):
        return cls.allocate(continuation_layout(), tuple(page_specs()), slots, pages, device, **kwargs)

    def attention(self, layer):
        if not 0 <= layer < 28:
            raise ValueError("ASR decoder layer is outside its complete KV catalog")
        return tuple(self.pages[f"layer{layer}.{field}"].view(torch.bfloat16)
            .view(-1, PAGE_TOKENS, 8, 128) for field in ("key", "value"))
