# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Audio-window publication and ordinary paged text graph contracts."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_package_pb2 as p

from .pool import PAGE_TOKENS
from .frontend import TRANSCRIPTION

FRAME_CAPACITIES = (100, 800)
FEATURES = 7800

FRONTEND = c.frontend(tools=p.QWEN_CODER, assistant_marker="<|im_start|>assistant\n",
    message_end_markers=["<|im_end|>"],
    prefix_boundary=c.prefix_boundary(["system"], append_role="user", minimum_messages=0,
        required_start="<|im_start|>system\n", required_marker="<|im_end|>"))
FRONTEND.update(audio_preparation=p.CENTERED_SLANEY_LOG_MEL,
    audio_tokens=dict(feature_token=151676, start_token=151669, end_token=151670),
    transcription=TRANSCRIPTION, request_seed_syntax=p.SIGNED64_OPTIONAL, tools=None, reasoning=None)


def prepare(signatures, buffers, pool, state, audio_inputs):
    inputs = {kind: c.text_inputs(kind) for kind in {signature.kind for signature in signatures.values()}}
    for kind in ("prefill", "scored_prefill"):
        if kind in inputs:
            inputs[kind].update({
                "original_prompt_tokens": c.binding("original_prompt_tokens", p.INITIAL_PROMPT_TOKENS),
                "audio_token_indices": c.binding("audio_token_indices", p.AUDIO_FEATURE_INDEX, p.PACKED_TOKEN, padding=-1),
                "audio_feature_slot": c.binding("audio_feature_slot", p.AUDIO_FEATURE_SLOT, p.SCALAR, padding=-1),
            })
    encoder = inputs["audio_encode"] = {
        "audio_feature_slot": c.binding("audio_feature_slot", p.AUDIO_FEATURE_SLOT, p.SCALAR, padding=-1),
    }
    fields = dict(mel=p.AUDIO_LOG_MEL, frames=p.AUDIO_VALID_FRAMES,
        total_input_frames=p.AUDIO_TOTAL_INPUT_FRAMES,
        convolution_width=p.AUDIO_CONVOLUTION_WIDTH, gather_indices=p.AUDIO_GATHER_INDICES,
        cu_seqlens=p.AUDIO_CUMULATIVE_LENGTHS, feature_count=p.AUDIO_FEATURE_COUNT,
        feature_offset=p.AUDIO_FEATURE_OFFSET)
    for prefix, raw in audio_inputs.items():
        for name in raw.buffers(""):
            source = p.REQUEST_SLOT if name == "request_slot" else p.REQUEST_GENERATION if name == "generation" else p.AUDIO_TENSOR
            encoder[prefix + name] = c.binding(prefix + name, source, p.SCALAR,
                audio_tensor=fields.get(name))
    continuation, pending = c.state_view("state_pool", pool.layout.row_bytes), c.state_view("request_pending_token", 8)
    views = [c.state_view("audio_features", FEATURES * 2048 * 2),
             c.state_view("audio_generations", 8), c.state_view("audio_counts", 4)]
    buffers.update(audio_features=state.arena, audio_generations=state.generation, audio_counts=state.count)
    graphs = c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending)
    for graph in graphs:
        if graph["kind"] in (p.AUDIO_ENCODE, p.PREFILL, p.SCORED_PREFILL):
            for index, view in enumerate(views):
                access = (p.WRITE if index == 0 else p.READ_WRITE) if graph["kind"] == p.AUDIO_ENCODE else p.READ
                graph["state"].append(dict(view=view, access=access))
    regions = [c.history_region(buffers, "cache." + name, value, p.PAGED_KEY_VALUE)
               for name, value in pool.pages.items()]
    return c.model_contract(vocabulary=151936, page_tokens=PAGE_TOKENS,
        page_tables=[dict(buffer="page_tables", page_offset=0)],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0), *regions],
        graphs=graphs, frontend=FRONTEND, immutable_request_values=[p.INITIAL_PROMPT_TOKENS],
        audio=dict(feature_slots=state.arena.shape[0], features_per_request=FEATURES, feature_width=2048,
            frame_capacities=FRAME_CAPACITIES, mel_bins=128, frames_per_chunk=100, features_per_chunk=13,
            subsampling_factor=8, attention_window_frames=800,
            features=views[0], generations=views[1], counts=views[2]))
