# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare actual prepared prefill projections on identical frozen operands."""

from contextlib import contextmanager


def _restore_count(count, saved):
    import torch

    count.copy_(saved)
    torch.cuda.synchronize(count.device)


@contextmanager
def projection_count(operation, rows, owners):
    """Restore diagnostic metadata only after its consumers have drained."""
    import torch
    from ignition.preparation.failure import drain_or_retain, retain_undrained

    count = operation.count
    saved = count.clone()
    owners.extend((operation, saved))
    try:
        count.fill_(rows)
        yield
        torch.cuda.synchronize(count.device)
    except BaseException as error:
        if drain_or_retain(error, count.device, owners):
            try:
                _restore_count(count, saved)
            except BaseException as cleanup_error:
                if not drain_or_retain(cleanup_error, count.device, owners):
                    retain_undrained(error, owners)
                error.add_note(f"projection count restoration failed: {cleanup_error!r}")
        raise
    else:
        try:
            _restore_count(count, saved)
        except BaseException as error:
            drain_or_retain(error, count.device, owners)
            raise


def independent_operands(decoder, reference, *, layers=(0,)):
    from dataclasses import asdict
    import torch
    from b12x.preparation.types import require_prepared
    from ignition.preparation.failure import retain_on_failed_drain

    from ignition.components.row_projection import PreparedRowProjection
    from ignition.components.canonical_projection import PreparedCanonicalProjection, PreparedRowwiseProjection
    from ignition.components.reduction_projection import PreparedReductionProjection

    modules = {}
    for index in layers:
        layer = decoder.tree.layers[index]
        prefix = f"thinker.model.layers.{index}."
        modules.update({prefix + "self_attn." + name: getattr(layer.self_attn, name)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")})
        modules.update({prefix + "mlp." + original: getattr(layer.mlp, name)
            for original, name in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down"))})
    owners, values = [decoder], {}
    with retain_on_failed_drain(decoder.scratch.logits.device, owners):
        for name, module in modules.items():
            wrapper = module.operation
            operation = wrapper.operation if isinstance(wrapper, PreparedRowProjection) else wrapper
            canonical = isinstance(operation, PreparedCanonicalProjection)
            fixed = isinstance(operation, PreparedReductionProjection)
            weight = operation.weight
            bias = getattr(operation, "bias", None)
            owners.extend((weight, bias))
            source = reference.get_tensor(name + ".input.0")
            expected = reference.get_tensor(name + ".output").reshape(-1, weight.shape[0])
            original = source.to(weight.device)
            owners.append(original)
            flat = original.reshape(-1, weight.shape[1])
            rows, capacity = flat.shape[0], operation.output.shape[0]
            if rows > capacity:
                raise ValueError("frozen projection operands exceed the selected physical capacity")
            padded = torch.zeros((capacity, flat.shape[1]), dtype=flat.dtype, device=flat.device)
            owners.append(padded)
            padded[:rows].copy_(flat)
            logical_output = torch.empty_like(operation.output[:rows])
            owners.append(logical_output)
            if canonical or fixed:
                with projection_count(operation, rows, owners):
                    actual = module(padded)[:rows].cpu().clone()
                selected = {}
            else:
                actual = module(padded)[:rows].cpu().clone()
                # Only this launcher accepts a smaller physical shape. Fixed
                # projections keep their capacity and use the device count.
                state = require_prepared(operation.plan, operation.plan.component_id)
                state.run(flat, weight, out=logical_output, bias=bias)
                selected = dict(selected_logical=logical_output.cpu().clone())
            logical_gpu = torch.nn.functional.linear(original, weight, bias)
            owners.append(logical_gpu)
            logical = logical_gpu.reshape(expected.shape).cpu().clone()
            flattened_gpu = torch.nn.functional.linear(flat, weight, bias)
            owners.append(flattened_gpu)
            flattened = flattened_gpu.cpu().clone()
            padded_gpu = torch.nn.functional.linear(padded, weight, bias)
            owners.append(padded_gpu)
            torch_padded = padded_gpu[:rows].cpu().clone()
            double_source, double_weight = flat.double(), weight.double()
            owners.extend((double_source, double_weight))
            double_output = double_source @ double_weight.T
            owners.append(double_output)
            if bias is not None:
                double_bias = bias.double()
                owners.append(double_bias)
                double_output.add_(double_bias)
            nearest = double_output.to(torch.bfloat16)
            owners.append(nearest)
            data = dict(actual=actual, reference=expected, **selected,
                torch_logical=logical, torch_flattened=flattened, torch_padded=torch_padded,
                fp64_rounded=nearest.cpu().clone())
            for value in data.values():
                if value.dtype != torch.bfloat16 or value.shape != expected.shape or not torch.isfinite(value).all():
                    raise ValueError("nonfinite or mismatched prefill projection observation")
            selection = None if canonical else operation.plan.selection
            values[name] = dict(data=data, logical_rows=rows, physical_rows=capacity,
                selected_logical_shape_available=not (canonical or fixed),
                input_shape=list(original.shape), input_stride=list(original.stride()),
                weight_stride=list(operation.weight.stride()),
                query=(dict(rows=1 if isinstance(operation, PreparedRowwiseProjection) else capacity,
                    columns=weight.shape[0], reduction=weight.shape[1], subrow_extents=operation.subrow_extents)
                    if canonical else asdict(operation.plan.query)),
                selection=(dict(component_id=None, config=None,
                    source="rowwise singleton eager linear; not captured" if isinstance(operation, PreparedRowwiseProjection)
                        else "canonical eager linear; not captured")
                    if canonical else dict(component_id=selection.component_id,
                        config=asdict(selection.config), source=selection.source)))
        torch.cuda.synchronize()
    return values


def independent_attention(decoder, reference, *, canonical_paged=False, repack_page_tokens=None, layer=0):
    """Probe the selected production paged operator with frozen canonical Q/K/V.

    This is a single-operator diagnostic, not the complete serving graph. It
    overwrites scratch/cache after the caller saved the complete teacher trace;
    the next request group must clear the pool and upload fresh metadata.
    """
    from ignition.preparation.failure import retain_on_failed_drain

    owners, graphs = [decoder], []
    with retain_on_failed_drain(decoder.scratch.logits.device, owners, graphs):
        return _independent_attention(decoder, reference, owners, graphs, canonical_paged, repack_page_tokens, layer)


def _independent_attention(decoder, reference, owners, graphs, canonical_paged, repack_page_tokens, layer):
    import torch
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.failure import retain_on_failed_drain

    if type(layer) is not int or not 0 <= layer < len(decoder.tree.layers):
        raise ValueError("attention probe layer is outside the prepared decoder")
    prefix = f"thinker.model.layers.{layer}.self_attn."
    q = reference.get_tensor(prefix + "q_norm.output").squeeze(0).to("cuda")
    owners.append(q)
    k = reference.get_tensor(prefix + "k_norm.output").squeeze(0).to("cuda")
    owners.append(k)
    v = reference.get_tensor(prefix + "v_proj.output").reshape(-1, 8, 128).to("cuda")
    owners.append(v)
    positions = []
    owners.append(positions)
    for i in (0, 1):
        positions.append(reference.get_tensor("thinker.model.rotary_emb.output." + str(i)).squeeze(0).to("cuda"))
    cosine, sine = positions
    q, k = apply_rotary_pos_emb(q, k, cosine, sine, unsqueeze_dim=1)
    owners.extend((q, k))
    expected = reference.get_tensor(prefix + "o_proj.input.0").reshape(-1, 16, 128)
    tokens = q.shape[0]
    # Require exact reconstruction of the independent represented operands,
    # rather than silently treating a new attention invocation as an oracle.
    canonical = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.repeat_interleave(2, dim=1).transpose(0, 1)[None],
        v.repeat_interleave(2, dim=1).transpose(0, 1)[None], is_causal=True)[0].transpose(0, 1).cpu().contiguous()
    if not torch.equal(canonical.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)):
        raise ValueError("canonical Q/K/V reconstruction did not reproduce frozen attention")
    attention = decoder.tree.layers[layer].self_attn.attention
    query, key, value, pages, lengths, cu, output = (getattr(attention, name) for name in (
        "query", "key", "value", "pages", "lengths", "cu_query", "output"))
    decode = decoder.decode_rows is not None
    count = 1 if decode else tokens
    if repack_page_tokens is not None:
        if not canonical_paged or repack_page_tokens != 256:
            raise ValueError("only the independent canonical paged probe may repack into 256-token pages")
        maximum_tokens = pages.shape[1] * key.shape[1]
        blocks = (tokens + repack_page_tokens - 1) // repack_page_tokens
        key = torch.empty((blocks, repack_page_tokens, 8, 128), device=query.device, dtype=query.dtype)
        owners.append(key)
        value = torch.empty_like(key)
        owners.append(value)
        pages = torch.full((lengths.numel(), (maximum_tokens + repack_page_tokens - 1) // repack_page_tokens),
            -1, device=query.device, dtype=torch.int32)
        owners.append(pages)
    with retain_on_failed_drain(query.device, owners, graphs):
        if tokens > pages.shape[1] * key.shape[1] or count > query.shape[0]:
            raise ValueError("independent Q/K/V do not fit the selected prepared operator")
        query.fill_(float("nan"))
        query[:count].copy_(q[-1:] if decode else q)
        used_pages = (tokens + key.shape[1] - 1) // key.shape[1]
        if repack_page_tokens is None and used_pages > int(decoder.pool.memory_capacity[1].item()):
            raise ValueError("independent operands exceed physically backed prepared cache pages")
        # PagedStateStorage exposes a larger reserved virtual range. Only
        # explicitly backed pages may be touched, including poisoning controls.
        key[:used_pages].fill_(float("nan"))
        value[:used_pages].fill_(float("nan"))
        key[:used_pages].view(-1, 8, 128)[:tokens].copy_(k)
        value[:used_pages].view(-1, 8, 128)[:tokens].copy_(v)
        pages.fill_(-1)
        pages[0, :used_pages].copy_(torch.arange(used_pages, device=pages.device, dtype=pages.dtype))
        lengths.zero_()
        lengths[0] = tokens
        cu.fill_(count)
        cu[0] = 0
        if canonical_paged:
            maximum_query = 1 if decode else query.shape[0]
            row_indices = torch.arange(query.shape[0], device=query.device)
            owners.append(row_indices)

            def attend():
                cumulative_keys = torch.cat((cu.new_zeros(1), torch.cumsum(lengths, 0, dtype=torch.int32)))
                result, _, _, _, _ = torch.ops.aten._flash_attention_forward(
                    query, key, value, cu, cumulative_keys, maximum_query, pages.shape[1] * key.shape[1],
                    0., True, False, seqused_k=lengths, block_table=pages, num_splits=1)
                output.copy_(result)
                output.masked_fill_((row_indices >= cu[-1])[:, None, None], 0)
        else:
            attend = attention
        owners.append(attend)
        compiled = compile_forward(attend)
        owners.append(compiled)
        output.zero_()
        compiled()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            compiled()
        actual = None
        for _ in range(5):
            output.zero_()
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                if before[name] != after[name]:
                    raise ValueError("selected paged attention replay allocated storage")
            current = output[:count].cpu().clone()
            if not torch.isfinite(current).all():
                raise ValueError("selected paged attention exposed poisoned unused storage")
            if actual is not None and not torch.equal(current.view(torch.uint8), actual.view(torch.uint8)):
                raise ValueError("selected paged attention changed repeated output bytes")
            actual = current
        negative = None
        if tokens % key.shape[1]:
            lengths[0] = tokens + 1
            graph.replay()
            torch.cuda.synchronize()
            negative = not torch.isfinite(output[:count]).all().item()
            if not negative:
                raise ValueError("wrong-length poisoned-tail control did not fail")
            lengths[0] = tokens
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(output[:count].cpu().reshape(-1).view(torch.uint8), actual.reshape(-1).view(torch.uint8)):
                raise ValueError("selected paged attention did not recover after wrong length")
        graph.reset()
    target = expected[-1:] if decode else expected
    delta = actual.double() - target.double()
    return dict(data=dict(actual=actual, reference=target, query=q.cpu(), key=k.cpu(), value=v.cpu()),
        layer=layer, mode="decode" if decode else "prefill", tokens=tokens, physical_query_rows=query.shape[0],
        page_tokens=key.shape[1], diagnostic_repacked_cache=repack_page_tokens is not None,
        component="aten._flash_attention_forward.paged" if canonical_paged else "aten._flash_attention_forward.request",
        selected=(dict(num_splits=1) if canonical_paged else
            dict(family_geometry=list(attention.family_geometry), max_query=attention.logical_max_query)),
        exact_canonical_operand_control=True,
        exact_bytes=torch.equal(actual.reshape(-1).view(torch.uint8), target.reshape(-1).view(torch.uint8)),
        max_abs=delta.abs().max().item(), relative_l2=(delta.norm() / target.double().norm().clamp_min(1e-300)).item(),
        existing_component_tolerance_passed=torch.allclose(actual, target, atol=.002, rtol=.02),
        wrong_length_nonfinite_rejected=negative, repeated_allocation_stable=True)
