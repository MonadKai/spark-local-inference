# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Export a prepared canonical BF16 projection for independent native replay."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from ignition.preparation.failure import retain_on_failed_drain, retain_undrained
from ignition.recipes.qwen3_asr.numerics import fp32_reductions


@contextmanager
def prepared_session(device, owners):
    from b12x.preparation import PreparationSession
    session = PreparationSession(device=device, autotune=False, compile_workers=1, prime=False)
    owners.append(session)
    try:
        yield session
    finally:
        # b12x close clears prepared owners even when its own fence fails.
        # Confirm the last use before allowing that irreversible cleanup.
        try:
            torch.cuda.synchronize(device)
        except BaseException as error:
            retain_undrained(error, owners)
            raise
        session.close()


class Forward:
    def __init__(self, operation, source):
        self.operation, self.source = operation, source

    def __call__(self):
        return self.operation(self.source)


def verify_replay(graph, output, expected):
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))


@torch.no_grad()
@fp32_reductions()
def prepare(root, capacity, count, with_bias=False, rowwise=False):
    from fixed_package_fixture import allocation_reference
    from fixture_frontend import server_config
    from signing import publisher
    from ignition.components.canonical_projection import PreparedCanonicalProjection, PreparedRowwiseProjection
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
                    width, columns = (7680, 1024) if with_bias else (2048, 4096)
                    source = torch.full((capacity, width), float('nan'), device='cuda', dtype=torch.bfloat16)
                    source[:count].fill_(.5)
                    weight = torch.ones((columns, width), device='cuda', dtype=torch.bfloat16)
                    bias = None
                    if with_bias:
                        source[:count].zero_()
                        source[:count, 0] = 1
                        source[:count, 1] = 1 / 256
                        bias = torch.full((columns,), -1., device="cuda", dtype=torch.bfloat16)
                        retained.append(bias)
                    output = torch.zeros((capacity, columns), device='cuda', dtype=torch.bfloat16)
                    cu = torch.tensor([count], device='cuda', dtype=torch.int32)
                    retained.extend((source, weight, output, cu))
                    signature = GraphSignature("decode", 1, 1, ("cu",))
                    with prepared_session(torch.device("cuda:0"), retained) as session:
                        projection_type = PreparedRowwiseProjection if rowwise else PreparedCanonicalProjection
                        forward = projection_type(source, weight, output, cu, bias=bias,
                            subrow_extents=(19, 27, 32) if capacity == 64 else ())
                        retained.append(forward)
                        # capture_forwards invokes zero-argument graph programs.
                        callable_forward = Forward(forward, source)
                        retained.append(callable_forward)
                        graphs, compiled = capture_forwards({"decode": callable_forward}, {"decode": signature},
                            session=session, device=torch.device("cuda:0"), stream=torch.cuda.Stream(),
                            pool=storage.graph_pool.id, cache_root=root / "forwards", retain_graphs=True)
                        retained.extend((graphs, compiled, session))
                        session.freeze()
                    expected = torch.zeros_like(output)
                    retained.append(expected)
                    expected[:count].fill_(1 / 256 if with_bias else 1024)
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
    parser.add_argument("--capacity", type=int, choices=(8, 64), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--with-bias", action="store_true")
    parser.add_argument("--rowwise", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.count <= args.capacity:
        parser.error("count must fit capacity")
    args.directory.mkdir(parents=True, exist_ok=False)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from helios.tests.canonical_projection_package_fixture import prepare as prepare_importable
    prepare_importable(args.directory, args.capacity, args.count, args.with_bias, args.rowwise)
