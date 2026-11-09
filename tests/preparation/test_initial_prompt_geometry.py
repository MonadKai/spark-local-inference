# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Immutable initial length selects C before the current fragment selects M."""

import pytest

from ignition.preparation.execution import GraphSignature, select_prefill_graph
from ignition.preparation import runtime_package_pb2 as p


def catalog(capacities=(128, 512, 8192)):
    return {f"c{initial}m{rows}": GraphSignature("prefill", 1, rows,
        initial_prompt_capacity=initial)
        for initial in capacities for rows in (64, 128, 512, 4096)}


@pytest.mark.parametrize("initial,expected", [(1, 128), (128, 128), (129, 512),
    (512, 512), (513, 8192), (8192, 8192)])
def test_initial_extent_does_not_grow_with_history_or_fragment(initial, expected):
    graphs = catalog()
    for ordered in (graphs, dict(reversed(list(graphs.items())))):
        for fragment in (1, 64, 65, 512, 4096):
            name = select_prefill_graph(ordered, initial_tokens=initial, tokens=fragment)
            assert ordered[name].initial_prompt_capacity == expected
            assert ordered[name].token_capacity == min(n for n in (64, 128, 512, 4096) if n >= fragment)
            widest = select_prefill_graph(ordered, initial_tokens=initial, tokens=fragment, largest=True)
            assert ordered[widest].initial_prompt_capacity == expected
            assert ordered[widest].token_capacity == 4096


def test_legacy_zero_catalog_keeps_its_existing_fragment_selection():
    graphs = catalog((0,))
    assert select_prefill_graph(graphs, initial_tokens=8192, tokens=65) == "c0m128"
    assert select_prefill_graph(graphs, initial_tokens=19, tokens=65, largest=True) == "c0m4096"


@pytest.mark.parametrize("fault", ["mixed-zero", "missing-combination", "duplicate", "outside-N", "outside-M"])
def test_invalid_or_incomplete_catalogs_cannot_silently_choose_other_arithmetic(fault):
    graphs = catalog()
    initial, fragment = 19, 64
    if fault == "mixed-zero": graphs.update(catalog((0,)))
    if fault == "missing-combination": del graphs["c128m128"]
    if fault == "duplicate": graphs["alias"] = graphs["c128m64"]
    if fault == "outside-N": initial = 8193
    if fault == "outside-M": fragment = 4097
    with pytest.raises(ValueError):
        select_prefill_graph(graphs, initial_tokens=initial, tokens=fragment)


@pytest.mark.parametrize("kind,capacity", [("decode", 128), ("prefill", -1),
    ("prefill", 2**31), ("prefill", True)])
def test_signature_rejects_invalid_initial_geometry(kind, capacity):
    with pytest.raises(ValueError):
        GraphSignature(kind, 1, 64, initial_prompt_capacity=capacity)


def test_actual_old_protobuf_wire_defaults_to_zero_and_roundtrips_unchanged():
    legacy = b"\x0a\x09prefill64\x12\x07prefill\x1a\x07general\x20\x01\x28\x40"
    graph = p.ExecutionGraph.FromString(legacy)
    assert graph.name == "prefill64" and graph.token_capacity == 64
    assert graph.initial_prompt_capacity == 0
    assert graph.SerializeToString() == legacy
    graph.initial_prompt_capacity = 128
    restored = p.ExecutionGraph.FromString(graph.SerializeToString())
    assert restored.initial_prompt_capacity == 128
    graph.initial_prompt_capacity = 0
    assert graph.SerializeToString() == legacy
