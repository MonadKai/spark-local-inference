# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Export the ASR convolution stack for replay after the preparer exits.

Synthetic operands exercise package mechanics, not checkpoint numerical quality.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


class ConvolutionNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutions = torch.nn.ModuleList([
            torch.nn.Conv2d(1, 480, 3, stride=2, padding=1),
            torch.nn.Conv2d(480, 480, 3, stride=2, padding=1),
            torch.nn.Conv2d(480, 480, 3, stride=2, padding=1),
        ])
        self.projection = torch.nn.Linear(7680, 1024, bias=False)
        self.norm = torch.nn.LayerNorm(1024, eps=1e-5)

    def forward(self, source):
        for convolution in self.convolutions:
            source = F.gelu(convolution(source))
        source = source.permute(0, 3, 1, 2).flatten(2)
        return self.norm(self.projection(source))


class Forward:
    def __init__(self, model, source, output):
        self.model, self.source, self.output = model, source, output

    def __call__(self):
        self.output.copy_(self.model(self.source))


@torch.no_grad()
def prepare(root, chunks):
    from b12x.preparation import PreparationSession
    from fixed_package_fixture import allocation_reference
    from fixture_frontend import server_config
    from signing import publisher

    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_file import combine
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.runtime_package import export_model

    with ModuleImages() as images:
        torch.cuda.set_device(0)
        torch.manual_seed(20261106)
        with PackageStorage() as storage:
            model = ConvolutionNorm().eval().to(device="cuda:0", dtype=torch.bfloat16)
            source = torch.randn((chunks, 1, 128, 100), device="cuda:0", dtype=torch.bfloat16) * .1
            output = torch.empty((chunks, 13, 1024), device="cuda:0", dtype=torch.bfloat16)
            forward = Forward(model, source, output)
            signature = GraphSignature("decode", 1, 1, ("source",))
            with PreparationSession(device=torch.device("cuda:0"), autotune=False,
                    compile_workers=1, prime=False) as session:
                graphs, compiled = capture_forwards({"decode": forward}, {"decode": signature},
                    session=session, device=torch.device("cuda:0"), stream=torch.cuda.Stream(),
                    pool=storage.graph_pool.id, cache_root=root / "forwards", retain_graphs=True)
                session.freeze()
                compiled["decode"]()
                torch.cuda.synchronize()
                reference = output.clone()
                assert torch.isfinite(reference).all()
                for _ in range(5):
                    output.fill_(float("nan"))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    graphs["decode"].replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    assert torch.equal(output.view(torch.uint8), reference.view(torch.uint8))
                    for key in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                        assert before[key] == after[key]
                expected = reference.view(torch.uint8).cpu().numpy().tobytes()
            buffers = {"frame_index": torch.zeros(1, device="cuda:0", dtype=torch.int32),
                "submission_sequence": torch.zeros(1, device="cuda:0", dtype=torch.int64),
                "completion_records": torch.zeros((2, 1, 24), device="cuda:0", dtype=torch.int64),
                "source": source, "output": output}
            records = buffers["completion_records"]
            records[..., 4] = -1
            records[..., 12:14] = -1
            records[..., 15] = 1
            records[..., 16:] = -1
            output.zero_()
            prepared = PreparedExecution(device=0, graphs=graphs, buffers=buffers,
                signatures={"decode": signature}, max_active=1,
                resources=(storage, model, forward, *compiled.values()),
                export_zero_ranges=((output.data_ptr(), output.nbytes),))
            prepared.server_config, prepared.package_data = server_config(root), None
    package = export_model(prepared, images, storage, root / "rank.json", options={}, sources=True)
    combine(root / "package", [root / "rank.json"], [], publisher=publisher())
    reference = allocation_reference(package.allocations, output)
    (root / "expected.json").write_text(json.dumps({
        "reference": reference, "bytes": list(expected),
        "addresses": {allocation.name: allocation.address for allocation in package.allocations},
        "zero_reference": reference, "zero_bytes": output.nbytes,
    }))
    print(f"exported {chunks} convolution chunks with {package.export_counts['modules']} cubins", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--chunks", type=int, choices=(1, 8), required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    # The legacy fixture helpers are scripts in this directory. Keep their
    # existing import path while exporting classes by a stable package name.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # Compilation workers import this module by its stable package name.
    from helios.tests.conv_package_fixture import prepare as prepare_importable
    prepare_importable(args.directory, args.chunks)
