# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Actual captured attention observer controls; these do not qualify a model."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.request_flash import PreparedRequestFlashAttention
from ignition.preparation.failure import retain_on_failed_drain
from qualify.asr_decoder_observation import independent_attention

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


@torch.no_grad()
def test_projection_count_restores_after_success_and_confirmed_error():
    from qualify.asr_decoder_observation import projection_count

    owners = []
    with retain_on_failed_drain(0, owners):
        operation = SimpleNamespace(count=torch.ones((1,), dtype=torch.int32, device="cuda"))
        output = torch.zeros_like(operation.count)
        owners.extend((operation, output))
        with projection_count(operation, 4, owners):
            output.copy_(operation.count)
        assert operation.count.item() == 1 and output.item() == 4
        with pytest.raises(ValueError, match="confirmed failure"):
            with projection_count(operation, 3, owners):
                output.copy_(operation.count)
                raise ValueError("confirmed failure")
        assert operation.count.item() == 1 and output.item() == 3


@torch.no_grad()
@pytest.mark.parametrize("failure", ["projection", "restore"])
def test_projection_count_failure_confirms_drain_before_restoring_and_retains_original_error(monkeypatch, failure):
    import gc
    import traceback
    import weakref
    from ignition.preparation.failure import resources_retained
    from qualify import asr_decoder_observation as observation

    synchronize, restore = torch.cuda.synchronize, observation._restore_count
    armed, calls, references = False, [], []
    count = torch.ones((1,), dtype=torch.int32, device="cuda")
    output = torch.zeros_like(count)

    class Owner:
        pass

    def run():
        nonlocal armed
        operation = Owner()
        operation.count = count
        references.append(weakref.ref(operation))
        owners = [operation, output]

        def fence(*args, **kwargs):
            if armed:
                raise RuntimeError("unconfirmed projection fence")
            return synchronize(*args, **kwargs)

        def restore_checked(value, saved):
            nonlocal armed
            assert not armed, "projection count restored before confirmed drain"
            calls.append("restore")
            if failure == "restore":
                output.copy_(count)
                armed = True
                raise RuntimeError("count restoration failed after CUDA submission")
            restore(value, saved)

        with retain_on_failed_drain(0, owners), monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "synchronize", fence)
            patch.setattr(observation, "_restore_count", restore_checked)
            try:
                with observation.projection_count(operation, 4, owners):
                    output.copy_(count)
                    armed = failure == "projection"
                    raise ValueError("original projection failure after CUDA submission")
            except ValueError as error:
                assert resources_retained(error), "original projection failure lost retained owners"
                return error
        raise AssertionError("planned projection failure did not propagate")

    error = run()
    synchronize()
    assert calls == ([] if failure == "projection" else ["restore"])
    assert count.item() == output.item() == 4
    traceback.clear_frames(error.__traceback__)
    error.__traceback__ = error.__cause__ = error.__context__ = None
    error.__dict__.clear()
    del error
    gc.collect()
    assert all(reference() is not None for reference in references)


@pytest.mark.parametrize("decode", [False, True])
def test_observer_uses_actual_request_buffers_and_bounded_backing(decode):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    generator = torch.Generator().manual_seed(20261106)
    tokens, slots, rows = 13, 8, 8 if decode else 64
    owners = []
    def keep(value):
        owners.append(value)
        return value
    with retain_on_failed_drain(0, owners):
        q = keep(torch.randn((tokens, 16, 128), generator=generator).bfloat16().cuda())
        k = keep(torch.randn((tokens, 8, 128), generator=generator).bfloat16().cuda())
        v = keep(torch.randn((tokens, 8, 128), generator=generator).bfloat16().cuda())
        angle = keep(torch.arange(tokens, device="cuda")[:, None] * torch.arange(128, device="cuda")[None, :] / 128)
        cosine, sine = keep(angle.cos().bfloat16()), keep(angle.sin().bfloat16())
        rotated_q, rotated_k = apply_rotary_pos_emb(q, k, cosine, sine, unsqueeze_dim=1)
        expected = torch.nn.functional.scaled_dot_product_attention(
            rotated_q.transpose(0, 1)[None], rotated_k.repeat_interleave(2, dim=1).transpose(0, 1)[None],
            v.repeat_interleave(2, dim=1).transpose(0, 1)[None], is_causal=True)[0].transpose(0, 1).contiguous()
        owners.extend((rotated_q, rotated_k, expected))
        query = keep(torch.zeros((rows, 16, 128), dtype=torch.bfloat16, device="cuda"))
        key = keep(torch.zeros((2, 256, 8, 128), dtype=torch.bfloat16, device="cuda"))
        value, output = keep(torch.zeros_like(key)), keep(torch.zeros_like(query))
        pages = keep(torch.full((slots, 2), -1, dtype=torch.int32, device="cuda"))
        lengths = keep(torch.zeros((slots,), dtype=torch.int32, device="cuda"))
        cu = keep(torch.zeros((slots + 1,), dtype=torch.int32, device="cuda"))
        attention = PreparedRequestFlashAttention(query, key, value, pages, lengths, cu,
            keep(torch.zeros_like(cu)), output, max_query=1 if decode else rows)
        owners.append(attention)
        # These are the current production object's actual fields: no mocked
        # plan, tma, completion or attention execution is involved.
        decoder = SimpleNamespace(tree=SimpleNamespace(layers=[
            SimpleNamespace(self_attn=SimpleNamespace(attention=attention))]),
            scratch=SimpleNamespace(logits=keep(torch.zeros((1,), device="cuda"))),
            decode_rows=object() if decode else None,
            pool=SimpleNamespace(memory_capacity=keep(torch.tensor([1, 2], device="cuda"))))
        owners.append(decoder)
        prefix = "thinker.model.layers.0.self_attn."
        tensors = {prefix + "q_norm.output": q.cpu()[None],
            prefix + "k_norm.output": k.cpu()[None], prefix + "v_proj.output": v.cpu(),
            prefix + "o_proj.input.0": expected.cpu(),
            "thinker.model.rotary_emb.output.0": cosine.cpu()[None],
            "thinker.model.rotary_emb.output.1": sine.cpu()[None]}
        reference = SimpleNamespace(get_tensor=tensors.__getitem__)
        result = independent_attention(decoder, reference)
        assert result["existing_component_tolerance_passed"]
        assert result["exact_canonical_operand_control"] and result["repeated_allocation_stable"]
        assert result["wrong_length_nonfinite_rejected"]
        assert result["component"] == "aten._flash_attention_forward.request"
        assert result["selected"]["family_geometry"] == attention.family_geometry
        # A large reserved extent cannot authorize unbacked poisoning/writes.
        decoder.pool.memory_capacity[1] = 0
        with pytest.raises(ValueError, match="physically backed"):
            independent_attention(decoder, reference)
        torch.cuda.synchronize()
