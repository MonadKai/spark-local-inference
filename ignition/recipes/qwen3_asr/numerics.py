# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The ASR arithmetic configuration is fixed before tuning and capture."""

from contextlib import contextmanager

import torch

POLICY = "qwen3-asr.bf16-fp32.v1"


@contextmanager
def fp32_reductions():
    previous = (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
    try:
        # b12x's legal candidates include Torch GEMMs. Their intermediate
        # reductions must obey the same FP32 policy as the independent oracle.
        # The captured executable retains this choice without a Rust selector.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, previous[1])
        yield
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
