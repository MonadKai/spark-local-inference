# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Actual spawned compiler workers retain the submitting arithmetic state."""

import os

import pytest
import torch

from ignition.preparation.compile_pool import CompilePool

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required")


class MatmulPolicyForward:
    def __init__(self, reduction, split_k):
        self.reduction, self.split_k = reduction, split_k
        self.source = torch.ones((8, 2048), device="cuda", dtype=torch.bfloat16)
        self.weight = torch.ones((2048, 64), device="cuda", dtype=torch.bfloat16)
        self.output = torch.empty((8, 64), device="cuda", dtype=torch.bfloat16)

    def __call__(self):
        if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction != self.reduction \
                or torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k != self.split_k:
            raise RuntimeError("compiler arithmetic policy differs from its submitting process")
        self.output.copy_(self.source @ self.weight)


@torch.no_grad()
def test_reused_spawn_worker_preserves_reduction_and_split_k_before_aot(tmp_path):
    previous = (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
    previous_blas = torch.backends.cuda.preferred_blas_library()
    try:
        # Torch permits split-K=False only with cuBLASLt. This fixture covers
        # transport of all valid settings, not default-cuBLAS support for them.
        torch.backends.cuda.preferred_blas_library("cublaslt")
        with CompilePool("cuda:0", compile_workers=1) as pool:
            for index, policy in enumerate(((False, False), (True, True), (False, True))):
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = policy
                forward = MatmulPolicyForward(*policy)
                path = tmp_path / f"policy{index}.bin"
                pool.submit(forward, path).result()
                with path.open("rb") as file:
                    compiled = torch.compiler.load_compiled_function(file)
                compiled(forward)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    compiled(forward)
                forward.source.fill_(2)
                forward.output.fill_(torch.nan)
                allocated = torch.cuda.memory_allocated()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == allocated
                assert torch.equal(forward.output, torch.full_like(forward.output, 4096))
                graph.reset()
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (True, True)
            with pytest.raises(Exception, match="compiler arithmetic policy differs"):
                pool.submit(MatmulPolicyForward(False, False), tmp_path / "wrong-policy.bin").result()
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
        torch.backends.cuda.preferred_blas_library(previous_blas)
