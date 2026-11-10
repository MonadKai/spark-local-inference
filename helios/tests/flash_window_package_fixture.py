# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Export a canonical flash window for replay after Python and its RNG exit."""

import argparse
import json
from pathlib import Path

import torch
from ignition.preparation.failure import retain_on_failed_drain


def verify_replay(graph, output, expected):
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))


@torch.no_grad()
def prepare(root, capacity, count):
    from b12x.preparation import PreparationSession
    from fixed_package_fixture import allocation_reference
    from fixture_frontend import server_config
    from signing import publisher
    from ignition.components.flash_window import PreparedFlashWindow
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
                    values = []
                    retained.append(values)
                    for _ in range(4):
                        values.append(torch.zeros((capacity, 16, 64), device="cuda", dtype=torch.bfloat16))
                    q, k, v, output = values
                    v[:count].fill_(2)
                    for value in (q, k, v):
                        value[count:].fill_(float("nan"))
                    cu = torch.tensor([0, count], device="cuda", dtype=torch.int32)
                    retained.append(cu)
                    forward = PreparedFlashWindow(q, k, v, cu, output)
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
    parser.add_argument("--capacity", type=int, choices=(13, 104), required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.count <= args.capacity:
        parser.error("count must fit capacity")
    args.directory.mkdir(parents=True, exist_ok=False)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from helios.tests.flash_window_package_fixture import prepare as prepare_importable
    prepare_importable(args.directory, args.capacity, args.count)
