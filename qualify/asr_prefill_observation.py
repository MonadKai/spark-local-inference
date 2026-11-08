# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Observe full prefill residuals without changing the serving graph catalog.

These additional preparation-only graphs are never exported or timed. A bridge
must establish exact production outputs and persistent state before an observed
residual can be used. The position graph retains the production singleton norm
and fixed-eight-row vocabulary projection instead of inventing a full-M head.
"""

from dataclasses import fields
from functools import partial

import torch


def _copy_residual(module, arguments, result, *, output):
    output.copy_(result)


def _copy_input(module, arguments, *, output):
    output.copy_(arguments[0])


def _copy_rotary(module, arguments, result, *, cosine, sine):
    cosine.copy_(result[0])
    sine.copy_(result[1])


def check_decoder_nodes(values, reference, case):
    """Apply existing network budgets; unnamed SiLU nodes remain attribution."""
    if __package__:
        from .asr_freeze_numerics import check
    else:
        from asr_freeze_numerics import check

    expected = {name for name in case["stages"] if name.startswith("thinker.model.")}
    extras = set(values) - expected
    unnamed = {f"thinker.model.layers.{index}.mlp.act_fn.{suffix}"
        for index in range(28) for suffix in ("input.0", "output")}
    if expected - values.keys() or not extras.issubset(unnamed):
        raise ValueError("decoder network node coverage differs from the frozen contract")
    result = dict(stages={}, unbudgeted_activation_nodes=sorted(extras), passed=False,
        local_operator_qualification=False)
    for name, value in values.items():
        if value.dtype != torch.bfloat16 or not value.numel() or not torch.isfinite(value).all():
            raise ValueError("nonfinite or non-BF16 decoder network node: " + name)
        if name in extras:
            continue
        target = reference.get_tensor(name)
        budget = case["stages"][name]
        shape = tuple(budget["shape"])
        # The prepared decoder flattens only the canonical singleton batch.
        # Equal element counts cannot authorize axis swaps or new layouts.
        if len(shape) < 3 or shape[0] != 1 or budget["dtype"] != "BF16" \
                or tuple(target.shape) != shape or tuple(value.shape) != shape[1:] \
                or target.dtype != torch.bfloat16 or not torch.isfinite(target).all():
            raise ValueError("frozen decoder network node geometry or finiteness differs: " + name)
        value = value.unsqueeze(0)
        gate = dict(passed=True)
        try:
            check(value, target, budget)
        except ValueError as error:
            gate.update(passed=False, error=str(error))
        difference = value.double() - target.double()
        gate.update(max_abs=difference.abs().max().item(),
            relative_l2=(difference.norm() / target.double().norm().clamp_min(1e-300)).item())
        result["stages"][name] = gate
    result["passed"] = bool(result["stages"]) and all(gate["passed"] for gate in result["stages"].values())
    return result


def check_head_nodes(values, reference, case):
    """Check actual position-head inputs/results against both frozen nodes."""
    if __package__:
        from .asr_freeze_numerics import check
    else:
        from asr_freeze_numerics import check

    expected = {"thinker.lm_head.input.0", "thinker.lm_head.output"}
    if values.keys() != expected:
        raise ValueError("vocabulary network node coverage differs")
    result = dict(stages={}, passed=False, local_operator_qualification=False)
    for name, actual in values.items():
        target, budget = reference.get_tensor(name), case["stages"][name]
        shape = tuple(budget["shape"])
        if len(shape) != 3 or shape[0] != 1 or budget["dtype"] != "BF16" \
                or actual.dtype != torch.bfloat16 or target.dtype != torch.bfloat16 \
                or tuple(actual.shape) != shape[1:] or tuple(target.shape) != shape \
                or not actual.numel() or not torch.isfinite(actual).all() or not torch.isfinite(target).all():
            raise ValueError("vocabulary network node geometry, dtype or finiteness differs")
        gate = dict(passed=True)
        try:
            check(actual.unsqueeze(0), target, budget)
        except ValueError as error:
            gate.update(passed=False, error=str(error))
        difference = actual.double() - target[0].double()
        gate.update(max_abs=difference.abs().max().item(),
            relative_l2=(difference.norm() / target.double().norm().clamp_min(1e-300)).item())
        result["stages"][name] = gate
    result["passed"] = all(gate["passed"] for gate in result["stages"].values())
    return result


class DecoderNodeCopies:
    """Preparation-only copies of actual decoder module inputs and BF16 nodes.

    Register before complete-forward compilation, and compare its outputs and
    persistent state with the unobserved production graph. These copies alone
    do not qualify arithmetic or establish a bridge.
    """

    def __init__(self, decoder, *, storage, storage_rows):
        self.decoder, self.handles, self.buffers = decoder, [], {}
        rows = decoder.scratch.activations.shape[0]
        self.rows = rows

        def destination(name, shape):
            if storage_rows < rows:
                raise ValueError("decoder node storage is smaller than the observed graph")
            if name not in storage:
                storage[name] = torch.zeros((storage_rows, *shape[1:]), dtype=torch.bfloat16,
                    device=decoder.scratch.activations.device)
            buffer = storage[name]
            if buffer.shape != (storage_rows, *shape[1:]) or buffer.dtype != torch.bfloat16 \
                    or buffer.device != decoder.scratch.activations.device:
                raise ValueError("shared decoder node storage geometry differs")
            result = buffer[:rows]
            self.buffers[name] = result
            return result

        def node(module, prefix, source_shape, result_shape):
            before = destination(prefix + ".input.0", source_shape)
            after = destination(prefix + ".output", result_shape)
            self.handles.append(module.register_forward_pre_hook(
                partial(_copy_input, output=before)))
            self.handles.append(module.register_forward_hook(
                partial(_copy_residual, output=after)))

        for index, layer in enumerate(decoder.tree.layers):
            prefix = f"thinker.model.layers.{index}"
            for name in ("input_layernorm", "post_attention_layernorm"):
                node(getattr(layer, name), prefix + "." + name, (rows, 2048), (rows, 2048))
            for name, columns in (("q_proj", 2048), ("k_proj", 1024),
                                  ("v_proj", 1024), ("o_proj", 2048)):
                node(getattr(layer.self_attn, name), prefix + ".self_attn." + name,
                    (rows, 2048), (rows, columns))
            for name, heads in (("q_norm", 16), ("k_norm", 8)):
                node(getattr(layer.self_attn.canonical, name), prefix + ".self_attn." + name,
                    (rows, heads, 128), (rows, heads, 128))
            for name, original, reduction, columns in (("gate", "gate_proj", 2048, 6144),
                    ("up", "up_proj", 2048, 6144), ("down", "down_proj", 6144, 2048)):
                node(getattr(layer.mlp, name), prefix + ".mlp." + original,
                    (rows, reduction), (rows, columns))
            node(layer.mlp.activation, prefix + ".mlp.act_fn", (rows, 6144), (rows, 6144))
            result = destination(prefix + ".output", (rows, 2048))
            self.handles.append(layer.register_forward_hook(partial(_copy_residual, output=result)))
        cosine = destination("thinker.model.rotary_emb.output.0", (rows, 128))
        sine = destination("thinker.model.rotary_emb.output.1", (rows, 128))
        self.handles.append(decoder.tree.rotary_emb.register_forward_hook(
            partial(_copy_rotary, cosine=cosine, sine=sine)))

    def values(self, tokens):
        if not 0 < tokens <= self.rows:
            raise ValueError("decoder node observation exceeds its physical rows")
        return {name: value[:tokens].cpu().clone() for name, value in self.buffers.items()}

    def close(self):
        # The caller first drains its observed graphs, which retain these
        # destinations until reset. This only removes Python preparation hooks.
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class _PositionHead:
    def __init__(self, decoder, residual):
        self.decoder, self.residual = decoder, residual
        self.position = torch.zeros((1,), dtype=torch.int64, device=residual.device)

    def __call__(self):
        selected = self.decoder.tree.norm(self.residual[self.position])
        self.decoder.scratch.hidden.copy_(selected)
        self.decoder.head(self.decoder.scratch.hidden)


class BridgeSnapshots:
    """Keep bounded device copies alive through every comparison and fence.

    These are diagnostic allocations outside graph replay. Keeping the complete
    bounded state on device avoids copying every KV plane to pageable memory
    several times per case. Semantic prefix/teacher observations are still
    saved independently on the CPU by the qualification collector.
    """

    def __init__(self):
        self.owners = []

    def snapshot(self, values):
        if not values:
            raise ValueError("bridge snapshots require nonempty CUDA state")
        saved = {}
        # Publish ownership before any copy can be submitted or fail.
        self.owners.append(saved)
        for name, value in values.items():
            if value.device.type != "cuda" or not value.numel():
                raise ValueError("bridge snapshots require nonempty CUDA state")
            saved[name] = torch.empty_like(value)
            saved[name].copy_(value)
        torch.cuda.synchronize()
        return saved

    def release(self):
        torch.cuda.synchronize()
        self.owners.clear()


def compare_bridge(actual, expected):
    if not actual or actual.keys() != expected.keys():
        raise ValueError("bridge observation fields differ")
    fields = {}
    for name, value in actual.items():
        other = expected[name]
        if value.device.type != "cuda" or other.device != value.device \
                or other.dtype != value.dtype or other.shape != value.shape or not value.numel():
            raise ValueError("bridge observation geometry differs: " + name)
        if (value.is_floating_point() or value.is_complex()) \
                and (not torch.isfinite(value).all() or not torch.isfinite(other).all()):
            raise ValueError("nonfinite bridge observation: " + name)
        a = value.contiguous().reshape(-1).view(torch.uint8)
        b = other.contiguous().reshape(-1).view(torch.uint8)
        differing = (a != b).sum().item()
        fields[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
            exact_bytes=differing == 0, differing_bytes=differing)
    return dict(passed=all(row["exact_bytes"] for row in fields.values()), fields=fields)


class PrefillObservation:
    def __init__(self, forward, *, node_storage=None, node_storage_rows=None):
        self.forward = forward
        self.residual = torch.zeros_like(forward.decoder.scratch.activations)
        self.head = _PositionHead(forward.decoder, self.residual)
        self.stream = torch.cuda.Stream(device=self.residual.device)
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs, self.compiled, self.hook = {}, {}, None
        self.bridge = BridgeSnapshots()
        self.nodes = None if node_storage is None else DecoderNodeCopies(forward.decoder,
            storage=node_storage, storage_rows=node_storage_rows)
        self.normalized = []

    def prepare(self, prepared, signature, storage, cache_root):
        from ignition.preparation.capture import capture_forwards
        from ignition.preparation.execution import GraphSignature
        from ignition.preparation.failure import drain_or_retain

        self.hook = self.forward.decoder.tree.layers[-1].register_forward_hook(
            partial(_copy_residual, output=self.residual))
        try:
            self.graphs, self.compiled = capture_forwards(
                {"observed_prefill": self.forward, "position_head": self.head},
                {"observed_prefill": signature,
                 "position_head": GraphSignature("prefill", 1, 1, ())},
                session=prepared.session, device=prepared.device, stream=self.stream,
                pool=self.pool, cache_root=cache_root, warmup_storage=storage,
                materialize=self.forward.decoder.pool.clear_for_preparation,
                report_catalog=False)
        except BaseException as error:
            if drain_or_retain(error, prepared.device, self, prepared, storage):
                self.close()
            raise

    def close(self):
        # The caller retains this owner together with PreparedModel and its
        # ModuleImages context if this fence or graph destruction fails.
        torch.cuda.synchronize(self.residual.device)
        for graph in self.graphs.values():
            graph.reset()
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        self.graphs.clear()
        self.compiled.clear()
        self.bridge.release()
        if self.nodes is not None:
            self.nodes.close()

    def logits(self, tokens, replay):
        if not 0 < tokens <= self.residual.shape[0]:
            raise ValueError("prefill observation exceeds its captured capacity")
        rows = []
        self.normalized = []
        for position in range(tokens):
            self.head.position.fill_(position)
            replay(self.graphs["position_head"])
            rows.append(self.forward.decoder.scratch.logits[0].cpu().clone())
            if self.nodes is not None:
                self.normalized.append(self.forward.decoder.scratch.hidden[0].cpu().clone())
        return torch.stack(rows)

    def node_values(self, tokens):
        if self.nodes is None or len(self.normalized) != tokens:
            raise ValueError("complete bridged decoder node observations are absent")
        values = self.nodes.values(tokens)
        values["thinker.model.kwargs.inputs_embeds"] = self.forward.decoder.scratch.activations[:tokens].cpu().clone()
        values["thinker.model.rotary_emb.input.0"] = values["thinker.model.kwargs.inputs_embeds"].clone()
        values["thinker.model.norm.input.0"] = self.residual[:tokens].cpu().clone()
        values["thinker.model.norm.output"] = torch.stack(self.normalized)
        return values


def bridge_buffers(prepared, forward):
    """All bounded persistent storage plus this graph's observable outputs.

    Page and continuation tensors expose a reserved virtual extent. Snapshot
    only the physically backed units declared by memory_capacity, including
    inactive rows/pages so a side-effect outside the request is still caught.
    """
    decoder = forward.decoder
    pool = decoder.pool
    rows, pages = (int(value) for value in pool.memory_capacity.cpu().tolist())
    # The exported cache.* bindings alias these same reserved tensors. Bound
    # those aliases too, not just the separately named page observations.
    bounded = {id(pool.backing): pool.backing[:rows],
        **{id(value): value[:pages] for value in pool.pages.values()}}
    values = {"input." + name: bounded.get(id(value), value)
        for name, value in prepared.buffers.items()}
    values.update({"request." + field.name: getattr(forward.batch.requests, field.name)
        for field in fields(forward.batch.requests)})
    feature = decoder.feature_state
    values.update({"feature." + name: getattr(feature, name)
        for name in ("arena", "generation", "count")})
    values.update({"decoder." + name: getattr(decoder.scratch, name)
        for name in ("hidden", "logits")})
    values["decoder.execution_errors"] = decoder.execution_errors
    return values


def restore(values, saved):
    if values.keys() != saved.keys():
        raise ValueError("prefill bridge storage changed")
    for name, value in values.items():
        source = saved[name]
        if source.shape != value.shape or source.dtype != value.dtype:
            raise ValueError("prefill bridge geometry changed: " + name)
        value.copy_(source)
    torch.cuda.synchronize()
