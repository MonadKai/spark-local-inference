# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Read already-written encoder projection buffers after a captured window.

No forward is replaced and no diagnostic tensor is added to the captured
program. Values must be read before another graph reuses its private scratch.
"""


def observe(encoder, frames, convolution_width):
    count = 13 * (frames // 100) + ((frames % 100) + 7) // 8
    chunks, columns = (frames + 99) // 100, (convolution_width + 7) // 8
    values = {"thinker.audio_tower.conv_out.output":
        encoder.convolution.view(encoder.chunks, 13, 1024)[:chunks, :columns].cpu().reshape(-1, 1024).clone()}
    for index, layer in enumerate(encoder.encoder.layers):
        prefix = f"thinker.audio_tower.layers.{index}"
        for field in ("q_proj", "k_proj", "v_proj", "out_proj"):
            values[prefix + f".self_attn.{field}.output"] = \
                getattr(layer.self_attn, field).projection.output[:count].cpu().clone()
        values[prefix + ".self_attn.out_proj.input.0"] = layer.self_attn.attended[:count].cpu().clone()
        for field in ("fc1", "fc2"):
            values[prefix + f".{field}.output"] = getattr(layer, field).projection.output[:count].cpu().clone()
    for field, original in (("linear_1", "proj1"), ("linear_2", "proj2")):
        values[f"thinker.audio_tower.{original}.output"] = \
            getattr(encoder.projector, field).projection.output[:count].cpu().clone()
    return values


def independent_operands(encoder, reference, *, layers=(0,)):
    """Run selected prepared operators on frozen reference inputs.

    This runs outside serving graphs, after their outputs have been saved. The
    FP64 calculations below are attribution controls, never replacement oracles.
    """
    import torch
    from ignition.preparation.failure import retain_on_failed_drain

    values = {}
    if not layers or len(set(layers)) != len(layers) or any(
            type(index) is not int or not 0 <= index < len(encoder.encoder.layers) for index in layers):
        raise ValueError("encoder probe layers must be unique valid indices")
    operations = {"thinker.audio_tower.conv_out": encoder.conv_out,
        "thinker.audio_tower.proj1": encoder.projector.linear_1,
        "thinker.audio_tower.proj2": encoder.projector.linear_2}
    for index in layers:
        layer = encoder.encoder.layers[index]
        prefix = f"thinker.audio_tower.layers.{index}"
        operations.update({prefix + ".fc1": layer.fc1, prefix + ".fc2": layer.fc2})
        operations.update({prefix + f".self_attn.{field}": getattr(layer.self_attn, field)
            for field in ("q_proj", "k_proj", "v_proj", "out_proj")})
    owners = [encoder]
    with retain_on_failed_drain(encoder.convolution.device, owners):
        for name, module in operations.items():
            operation = module.projection
            source = reference.get_tensor(name + ".input.0")
            expected = reference.get_tensor(name + ".output")
            source = source.reshape(-1, operation.weight.shape[1])
            expected = expected.reshape(-1, operation.weight.shape[0])
            physical_rows, rows = operation.output.shape[0], source.shape[0]
            if source.dtype != torch.bfloat16 or rows > physical_rows:
                raise ValueError("frozen operator operands do not fit the prepared projection")
            padded = torch.zeros((physical_rows, source.shape[1]), dtype=source.dtype, device=operation.weight.device)
            owners.append(padded)
            padded[:rows].copy_(source)
            actual = module(padded)[:rows].cpu().clone()
            logical_gpu = torch.nn.functional.linear(padded[:rows], operation.weight, operation.bias)
            owners.append(logical_gpu)
            logical = logical_gpu.cpu().clone()
            padded_gpu = torch.nn.functional.linear(padded, operation.weight, operation.bias)
            owners.append(padded_gpu)
            torch_padded = padded_gpu[:rows].cpu().clone()
            exact = source.double() @ operation.weight.cpu().double().T
            if operation.bias is not None:
                exact += operation.bias.cpu().double()
            values[name] = dict(actual=actual, reference=expected,
                fp64_rounded=exact.bfloat16(), torch_logical=logical, torch_padded=torch_padded,
                physical_rows=physical_rows, logical_rows=rows)
        for index in layers:
            attention = encoder.encoder.layers[index].self_attn
            prefix = f"thinker.audio_tower.layers.{index}.self_attn."
            rows = reference.get_tensor(prefix + "q_proj.output").shape[0]
            if not 0 < rows <= attention.query.shape[0]:
                raise ValueError("frozen attention operands exceed one prepared window")
            operands = []
            for field, output in (("q_proj", attention.query), ("k_proj", attention.key),
                                  ("v_proj", attention.value)):
                source = reference.get_tensor(prefix + field + ".output")
                output.zero_()
                output[:rows].copy_(source.reshape(rows, 16, 64))
                operands.append(output[:rows].transpose(0, 1).unsqueeze(0))
            encoder.metadata.cu_seqlens.copy_(torch.tensor([0, rows], dtype=torch.int32))
            attention.attended.zero_()
            attention.attention()
            logical_gpu = torch.nn.functional.scaled_dot_product_attention(*operands,
                dropout_p=0.0, is_causal=False)
            owners.append(logical_gpu)
            torch.cuda.synchronize()
            values[prefix + "attention"] = dict(actual=attention.attended[:rows].cpu().clone(),
                reference=reference.get_tensor(prefix + "out_proj.input.0"),
                torch_logical=logical_gpu.squeeze(0).transpose(0, 1).contiguous().view(rows, 1024).cpu(),
                physical_rows=attention.attended.shape[0], logical_rows=rows)
    for result in values.values():
        for value in result.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("nonfinite independent-operand diagnostic")
    return values
