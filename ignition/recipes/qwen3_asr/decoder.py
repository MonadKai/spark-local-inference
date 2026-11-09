# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical Qwen3 arithmetic with prepared BF16 projections and paged KV."""

from dataclasses import dataclass
from functools import partial

import torch

from ignition.components.embedding import PreparedEmbedding
from ignition.components.collectives import PreparedAllGather
from ignition.components.decode_rows import PreparedDecodeRows
from ignition.components.canonical_projection import PreparedCanonicalProjection
from ignition.components.canonical_rms_norm import CanonicalRMSNorm
from ignition.components.reduction_projection import PreparedReductionProjection
from ignition.components.paged_attention import PagedAttentionMetadata, PreparedPagedKVWriter
from ignition.components.request_flash import PreparedRequestFlashAttention

from .pool import PAGE_TOKENS
from .replay import PreparedReplayAttention, PreparedReplayProjection, ReplayScratch, history_projection_count


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, device, *, buffers):
        return cls(*(buffers.zeros(shape, torch.bfloat16) for shape in (
            (tokens, 2048), (rows, 2048), (rows, 151936))))


class _Projection(torch.nn.Module):
    def __init__(self, source, module, output, count, *, session, name, buffers, workspace,
                 subrow_extents=()):
        super().__init__()
        self.operation = PreparedCanonicalProjection(source, module.weight, output, count,
            bias=module.bias, subrow_extents=subrow_extents)

    def forward(self, source):
        return self.operation(source)


class _DecodeProjection(_Projection):
    def __init__(self, source, module, output, count, *, session, name, buffers, workspace,
                 benchmark_rows):
        torch.nn.Module.__init__(self)
        if module.bias is not None:
            raise ValueError("decode reduction requires the checkpoint's bias-free projection")
        self.operation = PreparedReductionProjection(source, module.weight, output, count,
            session=session, name=name, benchmark_rows=benchmark_rows)


@dataclass(frozen=True)
class _LayerScratch:
    query_raw: torch.Tensor
    key_raw: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    projected: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    intermediate: torch.Tensor
    mlp_output: torch.Tensor

    @classmethod
    def allocate(cls, tokens, buffers):
        nodes = tuple(buffers.zeros((tokens, *shape), torch.bfloat16) for shape in (
            (2048,), (1024,), (16, 128), (8, 128), (8, 128), (2048,), (2048,),
            (6144,), (6144,), (6144,), (2048,)))
        return cls(*nodes)


class _MLP(torch.nn.Module):
    def __init__(self, canonical, source, scratch, count, *, session, name, buffers, workspace, projection):
        super().__init__()
        self.activation, self.intermediate = canonical.act_fn, scratch.intermediate
        self.gate = projection(source, canonical.gate_proj, scratch.gate, count,
            session=session, name=name + ".gate_proj", buffers=buffers, workspace=workspace)
        self.up = projection(source, canonical.up_proj, scratch.up, count,
            session=session, name=name + ".up_proj", buffers=buffers, workspace=workspace)
        self.down = projection(scratch.intermediate, canonical.down_proj, scratch.mlp_output, count,
            session=session, name=name + ".down_proj", buffers=buffers, workspace=workspace)

    def forward(self, source):
        gate, up = self.gate(source), self.up(source)
        self.intermediate.copy_(self.activation(gate) * up)
        return self.down(self.intermediate)


class _Attention(torch.nn.Module):
    def __init__(self, canonical, source, metadata, cache_metadata, pool, layer, *, mode,
                 session, workspace, scratch, name, buffers, cumulative_keys, projection,
                 original=None, prompt_count=None):
        super().__init__()
        self.canonical = canonical
        tokens = source.shape[0]
        count = metadata.live_counts[:1]
        self.query, self.key, self.value = scratch.query, scratch.key, scratch.value
        self.attended, self.output = scratch.attended, scratch.projected
        self.q_proj = projection(source, canonical.q_proj, scratch.query_raw, count,
            session=session, name=name + ".q_proj", buffers=buffers, workspace=workspace)
        self.k_proj = projection(source, canonical.k_proj, scratch.key_raw, count,
            session=session, name=name + ".k_proj", buffers=buffers, workspace=workspace)
        self.v_proj = projection(source, canonical.v_proj, self.value.view(tokens, 1024),
            count, session=session, name=name + ".v_proj", buffers=buffers, workspace=workspace)
        self.o_proj = projection(self.attended, canonical.o_proj, self.output, count,
            session=session, name=name + ".o_proj", buffers=buffers, workspace=workspace)
        key, value = pool.attention(layer)
        self.writer = PreparedPagedKVWriter(self.key, self.value, key, value, cache_metadata.slots,
            session=session, name=name + ".cache_write")
        if mode == "prefill":
            self.attention = PreparedReplayAttention(self.query, key, value, metadata,
                cache_metadata.lengths, self.attended.view(tokens, 16, 128), original=original,
                prompt_count=prompt_count, buffers=buffers)
        else:
            self.attention = PreparedRequestFlashAttention(self.query, key, value, metadata.page_table,
                cache_metadata.lengths, metadata.query_start_loc, cumulative_keys,
                self.attended.view(tokens, 16, 128), max_query=1)

    def forward(self, hidden_states, position_embeddings, **kwargs):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        tokens = hidden_states.shape[0]
        # Canonical Qwen3RMSNorm rounds the normalized value to BF16 before
        # the BF16 affine multiply; zero-centered Qwen3.8 norms cannot replace it.
        query = self.canonical.q_norm(self.q_proj(hidden_states).view(tokens, 16, 128))
        key = self.canonical.k_norm(self.k_proj(hidden_states).view(tokens, 8, 128))
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings, unsqueeze_dim=1)
        self.query.copy_(query)
        self.key.copy_(key)
        self.v_proj(hidden_states)
        self.writer()
        self.attended.zero_()
        self.attention()
        return self.o_proj(self.attended), None


class _Vocabulary:
    def __init__(self, source, weight, output, count, *, session, name, workspace, buffers, local_output=None):
        self.count, self.rows = count, source.shape[0]
        local_output = output if local_output is None else local_output
        if local_output.shape != output.shape or local_output.dtype != output.dtype:
            raise ValueError("single-GPU ASR vocabulary outputs must have matching BF16 geometry")
        self.projection = PreparedCanonicalProjection(source, weight, local_output, count,
            staging=buffers.zeros((8, source.shape[1]), torch.bfloat16))
        self.gather = PreparedAllGather(local_output, output, runtime=None, coordinator=None,
            session=session, name=name + ".gather")

    @classmethod
    def prepare_resume(cls, source, weight, local_output, output, *, runtime, coordinator,
            session, name, activation_mode, buffers, row_capacity):
        if runtime is not None or coordinator is not None or activation_mode != "a16" or row_capacity != 8:
            raise ValueError("ASR resume vocabulary requires its single-GPU fixed-eight-row arithmetic")
        count = buffers.zeros((1,), torch.int32)
        return cls(source, weight, output, count, session=session, name=name, workspace=None,
            buffers=buffers, local_output=local_output)

    def __call__(self, source):
        # GraphBufferPool scratch is shared between variants. Initialization
        # during construction cannot publish a value for a later replay.
        self.count.fill_(self.rows)
        return self.gather(self.projection(source))


class PreparedDecoder:
    def __init__(self, weights, tree, scratch, metadata, pool, *, mode, session, name,
                 workspace, buffers, feature_state=None, original_prompt_tokens=None,
                 original_capacity=None):
        if mode not in ("prefill", "decode"):
            raise ValueError("ASR decoder prepares ordinary prefill and decode")
        self.weights, self.tree, self.scratch = weights, tree, scratch
        self.metadata, self.pool = metadata, pool
        self.original, self.original_capacity = original_prompt_tokens, original_capacity
        if mode == "prefill" and (original_prompt_tokens is None
                or original_prompt_tokens.shape != (1,) or original_prompt_tokens.dtype != torch.int32
                or original_prompt_tokens.device != metadata.token_ids.device
                or not original_prompt_tokens.is_contiguous() or type(original_capacity) is not int
                or not 1 <= original_capacity <= 8192):
            raise ValueError("ASR prefill requires its immutable original prompt count and context extent")
        if mode == "decode" and original_prompt_tokens is not None:
            raise ValueError("ordinary ASR decode does not select original-prompt arithmetic")
        self.decode_rows = None
        if mode == "decode":
            from ignition.components.paged_batch import Metadata

            padded = Metadata.allocate(8, 8, metadata.page_table.shape[1],
                metadata.token_ids.device, buffers=buffers)
            self.decode_rows = PreparedDecodeRows(metadata, padded)
            metadata = padded
        self.feature_state = feature_state if mode == "prefill" else None
        self.embedding = PreparedEmbedding(tree.embed_tokens.weight, metadata.token_ids,
            scratch.activations, num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.cache_metadata = PagedAttentionMetadata(metadata, page_tokens=PAGE_TOKENS, buffers=buffers)
        self.cumulative_keys = buffers.zeros((metadata.lengths.numel() + 1,), torch.int32)
        self.execution_errors = buffers.zeros((self.metadata.lengths.numel(),), torch.int32)
        # All layers run sequentially. Q/K/V and branch outputs survive only
        # until the corresponding canonical residual addition consumes them.
        layer_scratch = _LayerScratch.allocate(scratch.activations.shape[0], buffers)
        self.prompt_count = None
        projection = partial(_DecodeProjection, benchmark_rows=self.metadata.lengths.numel())
        if mode == "prefill":
            self.prompt_count = buffers.zeros((1,), torch.int32)
            self.history_count = buffers.zeros((1,), torch.int32)
            self.projection_rows = torch.arange(scratch.activations.shape[0], device=metadata.positions.device)
            replay = ReplayScratch.allocate(scratch.activations.shape[0], original_capacity, buffers)
            projection = partial(PreparedReplayProjection, prompt_count=self.prompt_count,
                positions=metadata.positions, original=original_prompt_tokens,
                history_count=self.history_count, scratch=replay)
        sequential = buffers.fork()
        stages = []
        for index, layer in enumerate(tree.layers):
            prefix = f"{name}.layers.{index}"
            stage = sequential.fork()
            stages.append(stage)
            # Match the physical mean layout of the smallest prefill variant.
            # Q/K norms already have at least 64 flattened head rows in decode.
            layer.input_layernorm = CanonicalRMSNorm(layer.input_layernorm.weight,
                layer.input_layernorm.variance_epsilon, minimum_rows=64)
            layer.post_attention_layernorm = CanonicalRMSNorm(layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.variance_epsilon, minimum_rows=64)
            layer.self_attn = _Attention(layer.self_attn, scratch.activations, metadata,
                self.cache_metadata, pool, index, mode=mode, session=session, workspace=workspace,
                scratch=layer_scratch, name=prefix + ".self_attn", buffers=stage,
                cumulative_keys=self.cumulative_keys, projection=projection,
                original=original_prompt_tokens, prompt_count=self.prompt_count)
            layer.mlp = _MLP(layer.mlp, scratch.activations, layer_scratch, metadata.live_counts[:1],
                session=session, name=prefix + ".mlp", buffers=buffers, workspace=workspace,
                projection=projection)
        buffers.join(*stages)
        tree.norm = CanonicalRMSNorm(tree.norm.weight, tree.norm.variance_epsilon, minimum_rows=64)
        # BF16 is the required LM-head result node. Sampling may convert this
        # tensor to FP32 afterwards; the projection must not bypass its rounding.
        # These are request rows, which may have inactive holes. Compute their
        # declared extent; completion still masks each original request row.
        head_count = buffers.zeros((1,), torch.int32)
        self.head = _Vocabulary(scratch.hidden, weights.tree.lm_head.weight, scratch.logits,
            head_count, session=session, name=name + ".head", buffers=buffers, workspace=workspace)
        self.auxiliary = None

    def __call__(self):
        self.execution_errors.zero_()
        if self.original is not None:
            invalid = (self.original <= 0) | (self.original > self.original_capacity)
            self.execution_errors.bitwise_or_(invalid.int() * (self.metadata.lengths > 0).int())
            self.prompt_count.copy_(torch.minimum(self.metadata.live_counts[:1],
                (self.original.long() - self.metadata.positions[:1]).clamp(min=0)).int())
            self.history_count.copy_(history_projection_count(self.projection_rows,
                self.metadata.live_counts[:1], self.metadata.positions, self.original))
        if self.decode_rows is not None:
            self.decode_rows()
        self.cache_metadata()
        self.cumulative_keys[:1].zero_()
        torch.cumsum(self.cache_metadata.lengths, dim=0, dtype=torch.int32, out=self.cumulative_keys[1:])
        hidden = self.embedding()
        if self.feature_state is not None:
            self.feature_state.inject(self.metadata, hidden)
        metadata = self.metadata if self.decode_rows is None else self.decode_rows.target
        positions = self.tree.rotary_emb(hidden, metadata.positions)
        for layer in self.tree.layers:
            hidden = layer(hidden, position_embeddings=positions)
            errors = layer.self_attn.attention.errors
            self.execution_errors.bitwise_or_(((errors[0] != 0)
                | (errors[1:1 + self.execution_errors.numel()] != 0)).int()
                * (self.metadata.lengths > 0).int())
        if self.decode_rows is None:
            selected = self.tree.norm(hidden[self.metadata.sample_indices])
        else:
            selected = self.tree.norm(hidden)[self.metadata.sample_indices]
        self.scratch.hidden.copy_(selected)
        self.head(self.scratch.hidden)
