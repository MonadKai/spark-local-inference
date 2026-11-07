# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Pure input-repair contracts; actual solver/CBMC controls are separate."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("cvc5_smt",
    Path(__file__).resolve().parents[1] / "verification/helios-lifecycle/cvc5_smt.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

DECLARATION = "(declare-fun |dead (TLS)| () (_ BitVec 0))"
QUERY = "(get-value (|dead (TLS)|))"
BODY = "(declare-fun x () (_ BitVec 8))\n(assert (= x x))\n(check-sat)"
SOURCE = DECLARATION + "\n" + BODY + "\n" + QUERY


def test_only_isolated_declaration_and_single_value_query_are_removed():
    value, metadata = adapter.normalize(SOURCE)
    assert value == "\n" + BODY + "\n"
    assert metadata == dict(removed_dead_symbols=["dead (TLS)"], removed_commands=2,
        assertions_preserved=1, all_retained_commands_byte_identical=True)
    assert adapter.normalize(BODY)[0] == BODY
    assert adapter.normalize("; ignored (()\n" + SOURCE)[1]["removed_commands"] == 2


@pytest.mark.parametrize("source", [
    SOURCE + "\n(assert (= |dead (TLS)| x))",
    SOURCE.replace(QUERY, "(get-value (|dead (TLS)| x))"),
    SOURCE + "\n" + DECLARATION,
    DECLARATION + "\n" + BODY,
    "(declare-fun f ((_ BitVec 8)) (_ BitVec 0))",
    "(assert (= x x)",
    r"(declare-fun |bad\symbol| () (_ BitVec 0))",
    SOURCE.replace("|dead (TLS)|", "|dead|") + "\n(assert (= dead x))",
])
def test_constrained_compound_or_malformed_zero_width_use_is_rejected(source):
    with pytest.raises(ValueError):
        adapter.normalize(source)
