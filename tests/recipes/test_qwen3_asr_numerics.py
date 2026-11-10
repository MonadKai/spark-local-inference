# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from ignition.recipes.qwen3_asr.numerics import fp32_reductions


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("initial", [(False, False), (False, True), (True, True)])
def test_preparation_reduction_policy_covers_the_call_and_restores_on_failure(failure, initial):
    previous = (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
    try:
        # Torch rejects (True, False); cover all three valid states.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = initial
        with pytest.raises(RuntimeError) if failure else fp32_reductions():
            if failure:
                with fp32_reductions():
                    assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k == initial[1]
                    raise RuntimeError("preparation interrupted")
            else:
                assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k == initial[1]
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == initial[0]
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k == initial[1]
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
