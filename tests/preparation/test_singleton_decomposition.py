# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare the closed compiler pass with its pinned original on real FX graphs."""

import ast
import inspect
from importlib.resources import files

import pytest
import torch
from torch._guards import detect_fake_mode
from torch._inductor import pattern_matcher
from torch._inductor.fx_passes import post_grad
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from ignition import patches


@torch.library.custom_op("flare_decomposition_test::update", mutates_args=("x",))
def update(x: torch.Tensor, y: torch.Tensor) -> None:
    x.add_(y)


@update.register_fake
def _(x, y):
    pass


@torch.library.custom_op("flare_decomposition_test::list_update", mutates_args=("xs", "extra"))
def list_update(xs: list[torch.Tensor], extra: torch.Tensor | None, y: torch.Tensor) -> None:
    for x in xs:
        x.add_(y)
    if extra is not None:
        extra.add_(y)


@list_update.register_fake
def _(xs, extra, y):
    pass


_out_library = torch.library.Library("flare_decomposition_out_test", "FRAGMENT")
_out_library.define("fill(Tensor x, *, Tensor(a!) out) -> Tensor(a!)", tags=(torch.Tag.out,))
_out_library.impl("fill", lambda x, *, out: out.copy_(x * 2), "CompositeExplicitAutograd")
_hop = torch.ops.higher_order.auto_functionalized
_hop2 = torch.ops.higher_order.auto_functionalized_v2
_targets = (_hop, _hop2)


@pytest.fixture(scope="module")
def original_decomposition():
    # Reconstruct only this original function from the exact bundled reverse
    # hunk. Execute both real handlers; installation mocks cannot prove aliases.
    source = inspect.getsource(post_grad)
    patch = files(patches).joinpath("torch-inductor-singleton-decomposition.patch").read_text()
    for _, before, after in patches._hunks(patch):
        assert source.count(after) == 1, "apply preparation patches before starting these tests"
        source = source.replace(after, before, 1)
    node = next(node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "decompose_auto_functionalized")
    namespace = post_grad.__dict__.copy()
    exec(compile(ast.get_source_segment(source, node), "<pinned-original-decomposition>", "exec"), namespace)
    return namespace["decompose_auto_functionalized"]


def _fixture(kind):
    op = torch.ops.flare_decomposition_test.update.default
    list_op = torch.ops.flare_decomposition_test.list_update.default
    if kind == "v1":
        def forward(x, y):
            a = _hop(op, x=x, y=y)[1]
            b = _hop(op, x=a, y=y)[1]
            return a, b, b.transpose(0, 1)
        inputs = [torch.arange(12.).reshape(3, 4), torch.ones(3, 4)]
    elif kind in ("v2_view", "control_deps"):
        def forward(x, y):
            kwargs = dict(_x_base_index=0, _x_size=(3, 4), _x_stride=(5, 1), _x_storage_offset=1, y=y)
            a = _hop2(op, _all_bases=[x], **kwargs)[1]
            b = _hop2(op, _all_bases=[a], **kwargs)[1]
            return a, b, b.transpose(0, 1)
        inputs = [torch.arange(20.).reshape(4, 5), torch.ones(3, 4)]
    elif kind in ("list_none", "list_optional"):
        optional = kind == "list_optional"
        def forward(x, y):
            kwargs = dict(_xs_length=2, _xs_0_base_index=0, _xs_0_size=(3, 4), _xs_0_stride=(5, 1),
                _xs_0_storage_offset=0, _xs_1_base_index=0, _xs_1_size=(3, 4), _xs_1_stride=(5, 1),
                _xs_1_storage_offset=1, _extra_base_index=1 if optional else None, y=y)
            a = _hop2(list_op, _all_bases=[x, torch.zeros(3, 4)] if optional else [x], **kwargs)
            b = _hop2(list_op, _all_bases=list(a[1:]), **kwargs)
            return a[1], b[1], b[1].transpose(0, 1)
        inputs = [torch.arange(20.).reshape(4, 5), torch.ones(3, 4)]
    elif kind == "out":
        def forward(x):
            return _hop2(torch.ops.flare_decomposition_out_test.fill.default, x=x, _all_bases=[],
                _out_size=(2, 3), _out_stride=(1, 2), _out_dtype=x.dtype, _out_device=x.device)
        inputs = [torch.arange(6.).reshape(2, 3)]
    elif kind in ("mutable_hop", "mutable_hop_mixed"):
        from torch._higher_order_ops.auto_functionalize import SchemaHolder
        from torch._higher_order_ops.invoke_subgraph import invoke_subgraph

        def inner_forward(x, y):
            x.add_(y)
            return (x * y,)
        inputs = [torch.arange(6.).reshape(2, 3), torch.ones(2, 3)]
        inner = make_fx(inner_forward, tracing_mode="fake")(*inputs)
        schema = invoke_subgraph.gen_schema(inner, "decomposition_test", *inputs)
        spec = pytree.tree_flatten(SchemaHolder(schema))[1]
        def forward(x, y):
            result = _hop2(invoke_subgraph, subgraph=inner, identifier="decomposition_test",
                _arg0_base_index=0, arg1=y, _all_bases=[x], _op_schema=spec)
            if kind == "mutable_hop":
                return result
            updated = _hop2(op, _all_bases=[result[1]], _x_base_index=0,
                _x_size=(2, 3), _x_stride=(3, 1), _x_storage_offset=0, y=y)[1]
            return result[0], updated
    else:
        raise AssertionError(kind)
    model = make_fx(forward, tracing_mode="fake")(*inputs)
    if kind in ("mutable_hop", "mutable_hop_mixed"):
        wrapper = next(node for node in model.graph.nodes if node.target is _hop2)
        for key, expected in (("subgraph", torch.fx.GraphModule), ("_op_schema", pytree.TreeSpec)):
            node = wrapper.kwargs[key]
            assert node.op == "get_attr" and "val" not in node.meta
            assert isinstance(getattr(model, node.target), expected)
    if kind == "control_deps":
        from torch._inductor.fx_passes.control_dependencies import control_deps
        root = torch.nn.Module()
        root.inner = model
        graph, args = torch.fx.Graph(), []
        for old in model.graph.nodes:
            if old.op == "placeholder":
                node = graph.placeholder(old.target)
                node.meta = old.meta.copy()
                args.append(node)
        inner = graph.get_attr("inner")
        graph.output(graph.call_function(control_deps, args=((args[1],), inner, *args)))
        model = torch.fx.GraphModule(root, graph)
    return model, inputs


def _run(model, inputs, transform, clone):
    wrappers = [node for module in model.modules() if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes if node.target in _targets]
    assert wrappers
    for index, node in enumerate(wrappers):
        if clone is not None:
            node.meta["only_clone_these_tensors"] = clone
        node.meta["custom"] = dict(stream=index, mempool=index + 4, mempool_device="cpu")
    mode = detect_fake_mode([node.meta["val"] for node in model.graph.nodes if node.op == "placeholder"])
    assert mode is not None
    with mode, V.set_fake_mode(mode):
        transform(model.graph)
    metadata, region_checks, region_states = [], [], []
    for name, module in model.named_modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        regions = [("mutation_region_id" in node.meta, node.meta.get("mutation_region_id"))
            for node in module.graph.nodes]
        metadata.extend((name, node.op, str(node.target), node.meta.get("custom")) for node in module.graph.nodes)
        pattern_matcher.compute_mutation_region_ids(module.graph)
        canonical = [node.meta["mutation_region_id"] for node in module.graph.nodes]
        region_checks.append(all(present and value == expected
            for (present, value), expected in zip(regions, canonical, strict=True)))
        for node, (present, value), expected in zip(module.graph.nodes, regions, canonical, strict=True):
            region_states.append((name, node.name, present, value, expected))
            if present:
                node.meta["mutation_region_id"] = value
            else:
                node.meta.pop("mutation_region_id", None)
        assert not any(node.target in _targets for node in module.graph.nodes)
        module.recompile()
    values = pytree.tree_leaves(model(*inputs))
    return model, inputs, values, metadata, region_checks, region_states


@pytest.mark.parametrize("kind", ["v1", "v2_view", "control_deps", "list_none", "list_optional", "out",
                                 "mutable_hop", "mutable_hop_mixed"])
@pytest.mark.parametrize("clone_mode", ["all", "none", "partial"])
def test_closed_pass_matches_original_values_mutations_and_aliases(original_decomposition, kind, clone_mode):
    clone = None if clone_mode == "all" else [] if clone_mode == "none" else ["x"] if kind == "v1" else [0]
    a = _run(*_fixture(kind), original_decomposition, clone)
    b = _run(*_fixture(kind), post_grad.decompose_auto_functionalized, clone)
    assert str(a[0].graph) == str(b[0].graph)
    assert a[3] == b[3]
    if kind not in ("mutable_hop", "mutable_hop_mixed"):
        assert all(a[4]) and all(b[4])
    # The original pass does not recurse through invoke_subgraph, and its
    # mutation tracker does not classify the replacement HOP as a mutating op.
    # A mixed pass can legitimately materialize more labels in the final scan.
    # Preserve every original label, and check all present labels against the
    # canonical scan, without altering the executable graph's saved metadata.
    for left_region, right_region in zip(a[5], b[5], strict=True):
        assert left_region[:2] == right_region[:2] and left_region[4] == right_region[4]
        for _, _, present, value, canonical in (left_region, right_region):
            assert not present or value == canonical
        if left_region[2]:
            assert right_region[2] and left_region[3] == right_region[3]
    left, right = a[1] + a[2], b[1] + b[2]
    assert len(left) == len(right)
    for x, y in zip(left, right, strict=True):
        assert torch.equal(x, y) and x.stride() == y.stride() and x.storage_offset() == y.storage_offset()
    for x, y in zip(left, right, strict=True):
        for other_x, other_y in zip(left, right, strict=True):
            assert (x.untyped_storage().data_ptr() == other_x.untyped_storage().data_ptr()) == (
                y.untyped_storage().data_ptr() == other_y.untyped_storage().data_ptr())
    if kind == "out":
        assert b[2][0].stride() == (1, 2)
        assert torch.equal(b[2][0], b[1][0] * 2)


@pytest.mark.parametrize("defect", ["missing_refresh", "wrong_clone"])
def test_metadata_and_input_guards_detect_planted_defects(defect):
    candidate = post_grad.decompose_auto_functionalized
    transform, clone = candidate, None
    if defect == "missing_refresh":
        source = inspect.getsource(candidate)
        needle = "    if changed_regions:\n        pm.compute_mutation_region_ids(graph)\n"
        assert source.count(needle) == 1
        namespace = post_grad.__dict__.copy()
        exec(compile(source.replace(needle, "    if changed_regions:\n        pass\n"),
            "<planted-missing-region-refresh>", "exec"), namespace)
        transform = namespace["decompose_auto_functionalized"]
    else:
        clone = []
    model, inputs = _fixture("v1")
    frozen = inputs[0].clone()
    with pytest.raises(AssertionError):
        result = _run(model, inputs, transform, clone)
        assert all(result[4])
        assert torch.equal(inputs[0], frozen)
    model, inputs = _fixture("v1")
    frozen = inputs[0].clone()
    result = _run(model, inputs, candidate, None)
    assert all(result[4])
    assert torch.equal(inputs[0], frozen)
    assert torch.equal(result[2][1], frozen + 2 * inputs[1])
