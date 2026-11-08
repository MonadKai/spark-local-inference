# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Attribute prompt-fragment attention geometry with identical BF16 operands.

This direct-ATen diagnostic measures numerical differences, not serving speed
or model qualification. It never reads beyond the actual consumed KV endpoint.
"""

from pathlib import Path
import datetime
import json


def metric(actual, reference):
    import torch
    assert actual.shape == reference.shape and actual.dtype == reference.dtype == torch.bfloat16
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    delta = actual.double() - reference.double()
    return dict(changed=int((actual.view(torch.int16) != reference.view(torch.int16)).sum()),
        max_abs=float(delta.abs().max()),
        relative_l2=float(delta.norm() / reference.double().norm().clamp_min(1e-300)))


def family(capacity, original):
    import torch
    from ignition.components.request_flash import _catalog
    table, plans = _catalog(capacity, 8192, 16, 8, 2 * torch.cuda.get_device_properties(0).multi_processor_count)
    row = 0 if original == 1 else (original + 63) // 64
    column = (original + 127) // 128
    index = table[row][column]
    assert index >= 0
    return plans[3 * index:3 * index + 3]


def run(out):
    import torch
    from safetensors.torch import save_file
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from ignition.preparation.failure import retain_on_failed_drain
    out.mkdir()
    record = dict(scope='Frozen synthetic BF16 Q/K/V. Direct pinned ATen FA2 full versus intermediate fragments under original, rounded and merged query geometry. True usedK=end throughout. Arithmetic attribution only; no recipe, metadata, native, ASR, quality or performance qualification.',
        seed=20261106, paged_key_capacity=8192, pinned_fa2_sms_factor=2, physical_sms=torch.cuda.get_device_properties(0).multi_processor_count, originals=[19, 38, 123, 129, 165, 192, 443],
        status='running', observations=[], errors=[],
        started_at=datetime.datetime.now(datetime.UTC).isoformat())
    def save():
        (out / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    save()
    try:
        for original in record['originals']:
            owners, graphs = [], []
            with retain_on_failed_drain(0, owners, graphs), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                generator = torch.Generator(device='cuda').manual_seed(record['seed'] + original)
                page_count = 8192 // 256
                q = torch.randn((original, 16, 128), device='cuda', dtype=torch.bfloat16, generator=generator) * .2
                k = torch.randn((page_count, 256, 8, 128), device='cuda', dtype=torch.bfloat16, generator=generator) * .2
                v = torch.randn(k.shape, device='cuda', dtype=torch.bfloat16, generator=generator) * .2
                key, value = k.clone(), v.clone()
                pages = torch.arange(page_count, dtype=torch.int32, device='cuda').reshape(1, page_count)
                owners.extend((q, k, v, key, value, pages))
                sdk = torch.nn.functional.scaled_dot_product_attention(q.transpose(0, 1)[None],
                    k.flatten(0, 1)[:original].transpose(0, 1)[None],
                    v.flatten(0, 1)[:original].transpose(0, 1)[None],
                    is_causal=True, enable_gqa=True)[0].transpose(0, 1).contiguous().cpu()
                outputs = dict(query=q.cpu(), key=k.cpu(), value=v.cpu(), sdk=sdk)
                canonical = family(original, original)
                merged = family(8192, original)
                geometries = [('original-query', canonical),
                    ('rounded', family(((original + 63) // 64) * 64, original)),
                    ('merged', merged)]
                boundaries = sorted(set([0, min(64, original), min(128, original), original]))
                parts = [('full', 0, original)] + [(f'part{i}', a, z)
                    for i, (a, z) in enumerate(zip(boundaries, boundaries[1:]))]
                if original > 128:
                    parts.append(('prefix128', 0, 128))
                for label, (physical, split, bound) in geometries:
                    for name, start, end in parts:
                        count = end - start
                        key.copy_(k); value.copy_(v)
                        query = torch.zeros((physical, 16, 128), dtype=torch.bfloat16, device='cuda')
                        query[-count:].copy_(q[start:end])
                        used = torch.tensor([end], dtype=torch.int32, device='cuda')
                        cuq = torch.tensor([0, physical], dtype=torch.int32, device='cuda')
                        cuk = torch.tensor([0, end], dtype=torch.int32, device='cuda')
                        output = torch.empty((count, 16, 128), dtype=torch.bfloat16, device='cuda')
                        owners.extend((query, used, cuq, cuk, output))
                        def operation():
                            result = torch.ops.aten._flash_attention_forward(query, key, value,
                                cuq, cuk, physical, bound, 0., True, False, seqused_k=used,
                                block_table=pages, num_splits=split)[0]
                            output.copy_(result[-count:])
                        for _ in range(3): operation()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph(); graphs.append(graph)
                        with torch.cuda.graph(graph): operation()
                        actual = None
                        for _ in range(5):
                            output.fill_(float('nan')); torch.cuda.synchronize()
                            before = torch.cuda.memory_stats(); graph.replay(); torch.cuda.synchronize()
                            after = torch.cuda.memory_stats()
                            for field in ('allocation.all.allocated', 'allocation.all.freed',
                                    'segment.all.allocated', 'segment.all.freed'):
                                assert before[field] == after[field], field
                            current = output.cpu().contiguous()
                            if actual is not None: assert metric(current, actual)['changed'] == 0
                            actual = current
                        key.flatten(0, 1)[end:].fill_(float('nan'))
                        value.flatten(0, 1)[end:].fill_(float('nan'))
                        graph.replay(); torch.cuda.synchronize()
                        assert metric(output.cpu().contiguous(), actual)['changed'] == 0
                        key.copy_(k); value.copy_(v)
                        key[0, 0].fill_(float('nan')); graph.replay(); torch.cuda.synchronize()
                        assert not bool(torch.isfinite(output).all()), 'active key poison did not trigger'
                        key.copy_(k); graph.replay(); torch.cuda.synchronize()
                        assert metric(output.cpu().contiguous(), actual)['changed'] == 0
                        outputs[f'{label}_{name}'] = actual
                        row = dict(original=original, geometry=label, physical_query=physical,
                            split=split, key_bound=bound, start=start, end=end,
                            canonical_query_index_shift=physical-end,
                            actual_used_keys=end, repeated_replays=5, allocation_stable=True,
                            future_kv_poison_unread=True, active_poison_and_recovery=True,
                            versus_sdk=metric(actual, sdk[start:end]))
                        if name != 'full':
                            row['versus_same_geometry_full'] = metric(actual,
                                outputs[f'{label}_full'][start:end])
                        record['observations'].append(row); save()
                        graph.reset()
                save_file(outputs, out / f'n{original}-operands-and-outputs.safetensors')
                torch.cuda.synchronize()
            save()
        record['status'] = 'completed_attribution_only'
    except BaseException as error:
        record['status'] = 'failed_or_interrupted'
        record['errors'].append(dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        record['finished_at'] = datetime.datetime.now(datetime.UTC).isoformat(); save()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    from ignition.cli.preparation import Host
    host = Host.discover()
    if len(host.gpus) != 1:
        parser.error("assign exactly one GPU with CUDA_VISIBLE_DEVICES")
    apply(Configuration.load(args.config), host.gpus)
    import torch
    with torch.no_grad():
        run(args.output)


if __name__ == "__main__":
    main()
