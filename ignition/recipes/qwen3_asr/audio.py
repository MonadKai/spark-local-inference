# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-capacity encoder windows with original convolution-tail semantics."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ignition.components.canonical_projection import PreparedCanonicalProjection
from ignition.components.flash_window import PreparedFlashWindow


def convolution_chunks(source, global_width, convolutions, positions):
    """Preserve the original batch's common padded width inside a fixed graph."""
    for convolution, columns in zip(convolutions, positions, strict=True):
        source = F.gelu(convolution(source))
        global_width = (global_width + 1) // 2
        source = source.masked_fill((columns >= global_width)[None, None, None, :], 0)
    return source


class _ConvolutionChunks:
    def __init__(self, convolutions, frame_capacity, device):
        if frame_capacity not in (100, 800):
            raise ValueError("audio convolution windows require 100 or 800 frame capacity")
        self.convolutions = convolutions
        self.chunks = frame_capacity // 100
        self.positions = tuple(torch.arange(width, device=device) for width in (50, 25, 13))
        self.short_positions = tuple(torch.arange(width, device=device) for width in (20, 10, 5))

    @staticmethod
    def _pad_batch(source, rows):
        return F.pad(source, (0, 0, 0, 0, 0, 0, 0, rows - source.shape[0]))

    def __call__(self, source, global_width, total_input_frames):
        # These groups were measured on the original represented operands and
        # supplementary logical batches through 600. Whole-input N is distinct
        # from the current window's valid chunks, especially at a short tail.
        whole_chunks = (total_input_frames.long() + 99) // 100
        result = source
        width = global_width
        for number, (convolution, columns) in enumerate(zip(self.convolutions, self.positions, strict=True), 1):
            ordinary = F.gelu(convolution(result))
            if number == 1:
                larger = F.gelu(convolution(self._pad_batch(result, 16)))[:self.chunks]
                ordinary = torch.where(whole_chunks >= 13, larger, ordinary)
            elif number == 2:
                if self.chunks == 1:
                    multiple = F.gelu(convolution(self._pad_batch(result, 2)))[:1]
                    ordinary = torch.where(whole_chunks >= 2, multiple, ordinary)
                else:
                    singleton = self._pad_batch(F.gelu(convolution(result[:1].contiguous())), self.chunks)
                    ordinary = torch.where(whole_chunks == 1, singleton, ordinary)
            else:
                if self.chunks == 1:
                    middle = F.gelu(convolution(self._pad_batch(result, 2)))[:1]
                else:
                    middle = self._pad_batch(F.gelu(convolution(result[:4].contiguous())), self.chunks)
                ordinary = torch.where((whole_chunks >= 2) & (whole_chunks <= 4), middle, ordinary)
            width = (width + 1) // 2
            result = ordinary.masked_fill((columns >= width)[None, None, None, :], 0)
        # A short whole input has one original chunk. Keep that physical batch
        # and the measured short width even when replay uses the larger window.
        short_source = source[:1] if self.chunks == 8 else source
        short = convolution_chunks(short_source[..., :40].contiguous(), global_width,
            self.convolutions, self.short_positions)
        if self.chunks == 8:
            short = self._pad_batch(short, self.chunks)
        short = F.pad(short, (0, result.shape[-1] - short.shape[-1]))
        return torch.where(global_width <= 40, short, result)


@dataclass(frozen=True)
class AudioMetadata:
    frames: torch.Tensor
    total_input_frames: torch.Tensor
    convolution_width: torch.Tensor
    gather_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    feature_count: torch.Tensor

    @classmethod
    def allocate(cls, capacity, device, *, buffers=None):
        if capacity not in (100, 800):
            raise ValueError("audio encoder windows require 100 or 800 Mel-frame capacity")
        def zeros(shape):
            return (torch.zeros(shape, dtype=torch.int32, device=device) if buffers is None
                    else buffers.zeros(shape, torch.int32))
        return cls(zeros((1,)), zeros((1,)), zeros((1,)), zeros((13 * (capacity // 100),)), zeros((2,)), zeros((1,)))


class _Linear(torch.nn.Module):
    def __init__(self, source, module, output, *, session, name, singleton_count,
                 buffers, physical_rows=104, large_frames=None):
        super().__init__()
        self.projection = PreparedCanonicalProjection(source, module.weight, output,
            bias=module.bias, singleton_count=singleton_count,
            staging=buffers.zeros((physical_rows, source.shape[1]), source.dtype))
        self.whole_features, self.large_frames = singleton_count, large_frames
        self.small = PreparedCanonicalProjection(source[:13], module.weight,
            buffers.zeros((13, output.shape[1]), output.dtype), bias=module.bias,
            singleton_count=singleton_count)
        self.large = None if large_frames is None else PreparedCanonicalProjection(source,
            module.weight, buffers.zeros(output.shape, output.dtype), bias=module.bias,
            staging=buffers.zeros((512, source.shape[1]), source.dtype))

    def forward(self, source):
        output = self.projection(source)
        small = self.small(source[:13])
        output[:13].copy_(torch.where(self.whole_features <= 13, small, output[:13]))
        if self.large is not None:
            chunks = (self.large_frames.long() + 99) // 100
            use_large = (chunks > 12) | ((chunks >= 5) & (chunks <= 6))
            output.copy_(torch.where(use_large, self.large(source), output))
        return output


class _Attention(torch.nn.Module):
    def __init__(self, module, hidden, metadata, *, session, name, workspace, buffers, singleton_count):
        super().__init__()
        rows = hidden.shape[0]
        self.query, self.key, self.value = (buffers.zeros((rows, 16, 64), hidden.dtype) for _ in range(3))
        self.attended = buffers.zeros(hidden.shape, hidden.dtype)
        self.projected = buffers.zeros(hidden.shape, hidden.dtype)
        for field, output in (("q_proj", self.query), ("k_proj", self.key), ("v_proj", self.value)):
            setattr(self, field, _Linear(hidden, getattr(module, field), output.view(rows, 1024),
                session=session, name=name + "." + field, singleton_count=singleton_count, buffers=buffers))
        self.out_proj = _Linear(self.attended, module.out_proj, self.projected,
            session=session, name=name + ".out_proj", singleton_count=singleton_count, buffers=buffers)
        self.attention = PreparedFlashWindow(
            self.query, self.key, self.value, metadata.cu_seqlens, self.attended.view(rows, 16, 64))

    def forward(self, hidden_states, cu_seqlens, **kwargs):
        self.q_proj(hidden_states)
        self.k_proj(hidden_states)
        self.v_proj(hidden_states)
        self.attended.zero_()
        self.attention()
        return self.out_proj(self.attended)


class PreparedAudioEncoder:
    """One window of an already normalized full-clip Mel input.

    The caller supplies a separate canonical module tree per captured capacity.
    Its parameters may reference the same immutable loaded weights. The graph
    consumes explicit host-prepared indices; it performs no dynamic packing.
    """

    def __init__(self, encoder, projector, metadata, *, capacity, device, session,
                 attention_workspace=None, buffers):
        if capacity not in (100, 800):
            raise ValueError("unsupported prepared audio window capacity")
        if (encoder.config.d_model, encoder.config.encoder_layers, encoder.config.encoder_attention_heads,
                encoder.config.encoder_ffn_dim, encoder.config.downsample_hidden_size) != (1024, 24, 16, 4096, 480):
            raise ValueError("audio encoder geometry differs from the original checkpoint")
        self.encoder, self.projector, self.metadata = encoder, projector, metadata
        self.capacity, self.chunks = capacity, capacity // 100
        prefix = f"audio{capacity}"
        rows = 13 * self.chunks
        self.global_features = buffers.zeros((1,), torch.int32)
        singleton_count = self.global_features
        if metadata.gather_indices.shape != (rows,) or metadata.cu_seqlens.shape != (2,):
            raise ValueError("audio metadata differs from captured capacity")
        self.mel_positions = torch.arange(capacity, device=device)
        self.feature_positions = torch.arange(rows, device=device)
        self.convolution_chunks = _ConvolutionChunks(
            (encoder.conv2d1, encoder.conv2d2, encoder.conv2d3), capacity, device)
        self.hidden = buffers.zeros((rows, 1024), torch.bfloat16)
        self.convolution = buffers.zeros(self.hidden.shape, self.hidden.dtype)
        self.conv_out = _Linear(buffers.zeros((rows, 7680), torch.bfloat16),
            encoder.conv_out, self.convolution, session=session, name=prefix + ".conv_out",
            singleton_count=singleton_count, buffers=buffers, large_frames=metadata.total_input_frames)
        for index, layer in enumerate(encoder.layers):
            layer_prefix = f"{prefix}.layers.{index}"
            layer.self_attn = _Attention(layer.self_attn, self.hidden, metadata,
                session=session, name=layer_prefix + ".self_attn", workspace=attention_workspace, buffers=buffers,
                singleton_count=singleton_count)
            intermediate = buffers.zeros((rows, 4096), torch.bfloat16)
            projected = buffers.zeros(self.hidden.shape, self.hidden.dtype)
            layer.fc1 = _Linear(self.hidden, layer.fc1, intermediate,
                session=session, name=layer_prefix + ".fc1", singleton_count=singleton_count, buffers=buffers)
            layer.fc2 = _Linear(intermediate, layer.fc2, projected,
                session=session, name=layer_prefix + ".fc2", singleton_count=singleton_count, buffers=buffers)
        projector_hidden = buffers.zeros(self.hidden.shape, self.hidden.dtype)
        self.output = buffers.zeros((rows, 2048), torch.bfloat16)
        projector.linear_1 = _Linear(self.hidden, projector.linear_1, projector_hidden,
            session=session, name=prefix + ".projector.linear_1", singleton_count=singleton_count, buffers=buffers)
        projector.linear_2 = _Linear(projector_hidden, projector.linear_2, self.output,
            session=session, name=prefix + ".projector.linear_2", singleton_count=singleton_count, buffers=buffers)

    def __call__(self, mel):
        metadata, encoder = self.metadata, self.encoder
        if mel.shape != (128, self.capacity) or mel.dtype != torch.float32:
            raise ValueError("audio graph input requires FP32 [128,frame_capacity] Log-Mel")
        total = metadata.total_input_frames.long()
        self.global_features.copy_(13 * (total // 100) + ((total % 100) + 7) // 8)
        features = mel.masked_fill((self.mel_positions >= metadata.frames)[None, :], 0).to(torch.bfloat16)
        features = features.view(128, self.chunks, 100).permute(1, 0, 2).unsqueeze(1)
        features = self.convolution_chunks(features, metadata.convolution_width, metadata.total_input_frames)
        features = features.permute(0, 3, 1, 2).contiguous().view(-1, 7680)
        features = self.conv_out(features).view(self.chunks, 13, 1024)
        features = features + encoder.positional_embedding.positional_embedding[:13].to(features.dtype)
        features = torch.index_select(features.view(-1, 1024), 0, metadata.gather_indices.clamp_min(0))
        features = features.masked_fill((self.feature_positions >= metadata.feature_count)[:, None], 0)
        for layer in encoder.layers:
            features = layer(features, metadata.cu_seqlens)[0]
        features = encoder.ln_post(features)
        result = self.projector(features)
        result.masked_fill_((self.feature_positions >= metadata.feature_count)[:, None], 0)
        return result
