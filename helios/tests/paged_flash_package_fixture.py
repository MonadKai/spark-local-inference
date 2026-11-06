# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Export paged causal attention for Rust replay after Python exits."""

import argparse
import json
from pathlib import Path

import torch
from ignition.preparation.failure import retain_on_failed_drain


def verify_replay(graph, output, expected):
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))


class _Forward:
    def __init__(self, operation):
        self.operation = operation

    def __call__(self):
        op = self.operation
        op.cu_key[:1].zero_()
        torch.cumsum(op.lengths, dim=0, dtype=torch.int32, out=op.cu_key[1:])
        return op()


@torch.no_grad()
def prepare(root, mode, count):
    from b12x.preparation import PreparationSession
    from fixed_package_fixture import allocation_reference
    from fixture_frontend import server_config
    from signing import publisher
    from ignition.components.paged_flash import PreparedPagedFlashAttention
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_file import combine
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.runtime_package import export_model

    retained = []
    with retain_on_failed_drain(0, retained):
        with ModuleImages() as images:
            retained.append(images)
            torch.cuda.set_device(0)
            with PackageStorage() as storage:
                retained.append(storage)
                with retain_on_failed_drain(0, retained):
                    capacity, batch = (8, 8) if mode == "decode" else (128, 1)
                    values = []
                    retained.append(values)
                    def allocate(shape, dtype=torch.bfloat16):
                        tensor = torch.zeros(shape, device="cuda", dtype=dtype)
                        values.append(tensor)
                        return tensor
                    q, output = allocate((capacity, 16, 128)), allocate((capacity, 16, 128))
                    k, v = allocate((4, 256, 8, 128)), allocate((4, 256, 8, 128))
                    q[count:].fill_(float("nan"))
                    k.fill_(float("nan"))
                    v.fill_(float("nan"))
                    pages = allocate((batch, 32), torch.int32)
                    pages.fill_(-1)
                    lengths = allocate((batch,), torch.int32)
                    cu, cu_key = allocate((batch + 1,), torch.int32), allocate((batch + 1,), torch.int32)
                    if mode == "decode":
                        # Two separated live slots; cumulative query lengths
                        # remain authoritative despite single-token GQA.
                        if count:
                            lengths[4], lengths[7] = 13, 257
                            pages[4, 0], pages[7, 0], pages[7, 1] = 0, 1, 2
                            cu[5:8].fill_(1)
                            cu[8] = 2
                            k[0, :13].zero_()
                            v[0, :13].fill_(2)
                            k[1:3].view(512, 8, 128)[:257].zero_()
                            v[1:3].view(512, 8, 128)[:257].fill_(2)
                    else:
                        lengths[0], cu[1] = 257, count
                        pages[0, :2].copy_(torch.tensor([0, 1], device="cuda", dtype=torch.int32))
                        k[:2].view(512, 8, 128)[:257].zero_()
                        v[:2].view(512, 8, 128)[:257].fill_(2)
                    operation = PreparedPagedFlashAttention(q, k, v, pages, lengths, cu, cu_key, output,
                        max_query=1 if mode == "decode" else capacity)
                    retained.append(operation)
                    forward = _Forward(operation)
                    retained.append(forward)
                    signature = GraphSignature("decode", 1, 1, ("cu",))
                    with PreparationSession(device=torch.device("cuda:0"), autotune=False, compile_workers=1, prime=False) as session:
                        graphs, compiled = capture_forwards({"decode": forward}, {"decode": signature},
                            session=session, device=torch.device("cuda:0"), stream=torch.cuda.Stream(),
                            pool=storage.graph_pool.id, cache_root=root / "forwards", retain_graphs=True)
                        retained.extend((graphs, compiled, session))
                        session.freeze()
                    expected = torch.zeros_like(output)
                    retained.append(expected)
                    expected[:count].fill_(2)
                    verify_replay(graphs["decode"], output, expected)
                    expected_bytes = expected.view(torch.uint8).cpu().numpy().tobytes()
                    buffers = dict(cu=cu, output=output)
                    retained.append(buffers)
                    buffers["frame_index"] = torch.zeros(1, device="cuda", dtype=torch.int32)
                    buffers["submission_sequence"] = torch.zeros(1, device="cuda", dtype=torch.int64)
                    buffers["completion_records"] = torch.zeros((2, 1, 24), device="cuda", dtype=torch.int64)
                    records = buffers["completion_records"]
                    records[..., 4] = -1
                    records[..., 12:14] = -1
                    records[..., 15] = 1
                    records[..., 16:] = -1
                    output.zero_()
                    prepared = PreparedExecution(device=0, graphs=graphs, buffers=buffers,
                        signatures={"decode": signature}, max_active=1,
                        resources=(storage, forward, *compiled.values()),
                        export_zero_ranges=((output.data_ptr(), output.nbytes),))
                    prepared.server_config, prepared.package_data = server_config(root), None
                    retained.append(prepared)
        with retain_on_failed_drain(0, retained):
            package = export_model(prepared, images, storage, root / "rank.json", options={}, sources=True)
            combine(root / "package", [root / "rank.json"], [], publisher=publisher())
            reference = allocation_reference(package.allocations, output)
            (root / "expected.json").write_text(json.dumps(dict(reference=reference, bytes=list(expected_bytes),
                addresses={allocation.name: allocation.address for allocation in package.allocations},
                zero_reference=reference, zero_bytes=output.nbytes)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--mode", choices=("decode", "prefill"), required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    if args.count not in ((0, 2) if args.mode == "decode" else (13,)):
        parser.error("use decode count 0/2 or prefill count 13")
    args.directory.mkdir(parents=True, exist_ok=False)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from helios.tests.paged_flash_package_fixture import prepare as prepare_importable
    prepare_importable(args.directory, args.mode, args.count)
