# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation-only encoder nodes, accepted only after a production-state bridge."""

from functools import partial

import torch

if __package__:
    from .asr_prefill_observation import BridgeSnapshots, compare_bridge, restore
else:
    from asr_prefill_observation import BridgeSnapshots, compare_bridge, restore


def convolution_selection(capacity, total_frames):
    """Declared physical calls in audio._ConvolutionChunks, never error fitting."""
    if capacity not in (100, 800) or type(total_frames) is not int or total_frames <= 0:
        raise ValueError("invalid audio observation geometry")
    chunks = capacity // 100
    whole_chunks = (total_frames + 99) // 100
    if total_frames <= 40:
        return ((1, 40), (1, 20), (1, 10))
    first = 16 if whole_chunks >= 13 else chunks
    second = (2 if whole_chunks >= 2 else 1) if chunks == 1 else (1 if whole_chunks == 1 else 8)
    third = (2 if chunks == 1 else 4) if 2 <= whole_chunks <= 4 else chunks
    return ((first, 100), (second, 50), (third, 25))


def check_encoder_nodes(values, reference, case):
    if __package__:
        from .asr_freeze_numerics import check
    else:
        from asr_freeze_numerics import check

    expected = {name for name in case["stages"] if name.startswith("thinker.audio_tower.")}
    if not expected or values.keys() != expected:
        raise ValueError("encoder network node coverage differs from the frozen contract")
    result = dict(stages={}, passed=False, local_operator_qualification=False)
    for name, actual in values.items():
        target, budget = reference.get_tensor(name), case["stages"][name]
        if budget["dtype"] != "BF16" or actual.dtype != torch.bfloat16 or target.dtype != actual.dtype \
                or tuple(actual.shape) != tuple(budget["shape"]) or actual.shape != target.shape \
                or not actual.numel() or not torch.isfinite(actual).all() or not torch.isfinite(target).all():
            raise ValueError("encoder network node geometry, dtype or finiteness differs: " + name)
        gate = dict(passed=True)
        try:
            check(actual, target, budget)
        except ValueError as error:
            gate.update(passed=False, error=str(error))
        difference = actual.double() - target.double()
        gate.update(max_abs=difference.abs().max().item(),
            relative_l2=(difference.norm() / target.double().norm().clamp_min(1e-300)).item())
        result["stages"][name] = gate
    result["passed"] = all(gate["passed"] for gate in result["stages"].values())
    return result


def _input(module, arguments, *, output):
    output.copy_(arguments[0])


def _output(module, arguments, result, *, output):
    output.copy_(result)


def _convolution_input(module, arguments, *, destinations):
    source = arguments[0]
    destination, _ = destinations[source.shape[0], source.shape[-1]]
    destination.copy_(source)


def _convolution_output(module, arguments, result, *, destinations):
    source = arguments[0]
    _, destination = destinations[source.shape[0], source.shape[-1]]
    destination.copy_(result)


def _layer_output(module, arguments, result, *, output):
    output.copy_(result[0])


class EncoderNodeCopies:
    def __init__(self, encoder, owners):
        self.handles, self.buffers, self.convolutions = [], {}, []
        owners.append(self)
        self.encoder = encoder
        rows, device = encoder.hidden.shape[0], encoder.hidden.device

        def destination(name, shape):
            value = torch.zeros(shape, dtype=torch.bfloat16, device=device)
            self.buffers[name] = value
            return value

        def node(module, name, source_columns, output_columns):
            before = destination(name + ".input.0", (rows, source_columns))
            after = destination(name + ".output", (rows, output_columns))
            self.handles.append(module.register_forward_pre_hook(partial(_input, output=before)))
            self.handles.append(module.register_forward_hook(partial(_output, output=after)))

        chunks = encoder.chunks
        # Every invocation has a distinct, declared physical N/W pair. Keep
        # all candidates; the final short call cannot overwrite an ordinary one.
        calls = (((chunks, 100), (16, 100), (1, 40)),
            ((chunks, 50), (2 if chunks == 1 else 1, 50), (1, 20)),
            ((chunks, 25), (2 if chunks == 1 else 4, 25), (1, 10)))
        for index, (module, shapes) in enumerate(zip(encoder.convolution_chunks.convolutions, calls, strict=True)):
            destinations = {}
            self.convolutions.append(destinations)
            for batch, width in shapes:
                source_shape = (batch, module.in_channels, 128 // (2**index), width)
                result_shape = (batch, module.out_channels, 128 // (2**(index + 1)), (width + 1) // 2)
                pair = destinations[batch, width] = []
                pair.append(torch.zeros(source_shape, dtype=torch.bfloat16, device=device))
                pair.append(torch.zeros(result_shape, dtype=torch.bfloat16, device=device))
            self.handles.append(module.register_forward_pre_hook(
                partial(_convolution_input, destinations=destinations)))
            self.handles.append(module.register_forward_hook(
                partial(_convolution_output, destinations=destinations)))
        node(encoder.conv_out, "thinker.audio_tower.conv_out", 7680, 1024)
        for index, layer in enumerate(encoder.encoder.layers):
            prefix = f"thinker.audio_tower.layers.{index}"
            for name in ("self_attn_layer_norm", "final_layer_norm"):
                node(getattr(layer, name), prefix + "." + name, 1024, 1024)
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                node(getattr(layer.self_attn, name), prefix + ".self_attn." + name, 1024, 1024)
            node(layer.fc1, prefix + ".fc1", 1024, 4096)
            node(layer.fc2, prefix + ".fc2", 4096, 1024)
            target = destination(prefix + ".output.0", (rows, 1024))
            self.handles.append(layer.register_forward_hook(partial(_layer_output, output=target)))
            if index == 0:
                target = destination(prefix + ".input.0", (rows, 1024))
                self.handles.append(layer.register_forward_pre_hook(partial(_input, output=target)))
        node(encoder.encoder.ln_post, "thinker.audio_tower.ln_post", 1024, 1024)
        node(encoder.projector.linear_1, "thinker.audio_tower.proj1", 1024, 1024)
        node(encoder.projector.linear_2, "thinker.audio_tower.proj2", 1024, 2048)

    def values(self, frames, total_frames):
        if not 0 < frames <= self.encoder.capacity or total_frames < frames:
            raise ValueError("audio observation exceeds the actual window")
        width, chunks = min(total_frames, 100), (frames + 99) // 100
        columns, count = (width + 7) // 8, 13 * (frames // 100) + ((frames % 100) + 7) // 8
        values = {}
        for index, (destinations, selected) in enumerate(zip(self.convolutions,
                convolution_selection(self.encoder.capacity, total_frames), strict=True), 1):
            source, result = destinations[selected]
            values[f"thinker.audio_tower.conv2d{index}.input.0"] = source[:chunks, :, :, :width].cpu().clone()
            width = (width + 1) // 2
            values[f"thinker.audio_tower.conv2d{index}.output"] = result[:chunks, :, :, :width].cpu().clone()
        for name, value in self.buffers.items():
            if name.startswith("thinker.audio_tower.conv_out."):
                # Canonical CNN linears retain padded tail chunk positions.
                value = value.view(self.encoder.chunks, 13, value.shape[1])[:chunks, :columns]
            else:
                value = value[:count]
            values[name] = value.cpu().clone()
        return values

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class AudioObservation:
    def __init__(self, forward):
        from ignition.preparation.failure import retain_on_failed_drain

        self.forward, self.graphs, self.compiled, self.node_owners = forward, {}, {}, []
        self.bridge = BridgeSnapshots()
        with retain_on_failed_drain(forward.inputs.mel.device, self, forward):
            self.nodes = EncoderNodeCopies(forward.encoder, self.node_owners)
            self.stream = torch.cuda.Stream(device=forward.inputs.mel.device)
            self.pool = torch.cuda.graph_pool_handle()

    def prepare(self, prepared, signature, storage, cache_root, materialize):
        from ignition.preparation.capture import capture_forwards

        self.graphs, self.compiled = capture_forwards(
            {"observed_audio": self.forward}, {"observed_audio": signature},
            session=prepared.session, device=prepared.device, stream=self.stream,
            pool=self.pool, cache_root=cache_root, warmup_storage=storage,
            materialize=materialize, report_catalog=False)

    def close(self):
        torch.cuda.synchronize(self.forward.inputs.mel.device)
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.compiled.clear()
        self.bridge.release()
        for owner in self.node_owners:
            owner.close()

    def observe(self, prepared, exemplar, production, replay, frames, total_frames):
        from ignition.preparation.failure import drain_or_retain, retain_undrained

        values = bridge_buffers(prepared, exemplar, self.forward)
        production_state = None
        try:
            initial = self.bridge.snapshot(values)
            replay(production)
            production_state = self.bridge.snapshot(values)
            restore(values, initial)
            replay(self.graphs["observed_audio"])
            observed = self.bridge.snapshot(values)
            comparison = compare_bridge(observed, production_state)
            nodes = self.nodes.values(frames, total_frames)
        except BaseException as error:
            # A failed or interrupted replay may still reference every byte in
            # values. Confirm its drain before restoring any shared storage.
            if drain_or_retain(error, prepared.device, self, prepared):
                try:
                    if production_state is not None:
                        restore(values, production_state)
                    self.bridge.release()
                except BaseException as cleanup_error:
                    if not drain_or_retain(cleanup_error, prepared.device, self, prepared):
                        retain_undrained(error, self, prepared)
                    error.add_note(f"audio observation cleanup failed: {cleanup_error!r}")
            raise
        else:
            try:
                restore(values, production_state)
                self.bridge.release()
            except BaseException as error:
                drain_or_retain(error, prepared.device, self, prepared)
                raise
            return nodes, comparison


def bridge_buffers(prepared, exemplar, forward):
    if __package__:
        from .asr_prefill_observation import bridge_buffers as decoder_buffers
    else:
        from asr_prefill_observation import bridge_buffers as decoder_buffers

    values = decoder_buffers(prepared, exemplar)
    values.update({"audio." + name: getattr(forward, name) for name in ("errors", "mel")})
    values.update({"audio.publication." + name: getattr(forward.publish, name)
        for name in ("records", "frame", "sequence")})
    values["audio.output"] = forward.encoder.output
    # Existing encoder observations read these private planes. Bridge them too
    # before retaining their old same-path observations alongside new copies.
    for index, layer in enumerate(forward.encoder.encoder.layers):
        for name in ("query", "key", "value", "attended", "projected"):
            values[f"audio.layer.{index}.{name}"] = getattr(layer.self_attn, name)
        for name in ("fc1", "fc2"):
            values[f"audio.layer.{index}.{name}"] = getattr(layer, name).projection.output
    values["audio.convolution"] = forward.encoder.convolution
    values["audio.projector.linear_1"] = forward.encoder.projector.linear_1.projection.output
    return values
