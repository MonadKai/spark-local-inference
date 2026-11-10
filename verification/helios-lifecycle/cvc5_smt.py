#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Reject every BV0 use except an isolated declaration/model-query pair.

This repairs a specific invalid solver input; it is not a stock Kani proof.
Only whole top-level commands are removed. All other original bytes survive.
"""


def commands(source):
    tokens, result, i, begin, depth = [], [], 0, None, 0
    while i < len(source):
        char = source[i]
        if char.isspace():
            i += 1
            continue
        if char == ";":
            newline = source.find("\n", i)
            i = len(source) if newline < 0 else newline + 1
            continue
        start = i
        if char in ("|", '"'):
            delimiter = char
            i += 1
            while i < len(source):
                if source[i] == delimiter:
                    if delimiter == '"' and i + 1 < len(source) and source[i + 1] == '"':
                        i += 2
                        continue
                    i += 1
                    break
                if delimiter == "|" and source[i] == "\\":
                    raise ValueError("unsupported quoted-symbol escape")
                i += 1
            else:
                raise ValueError("unterminated quoted token")
            value = source[start:i]
        elif char in "()":
            i += 1
            value = char
        else:
            while i < len(source) and not source[i].isspace() and source[i] not in '();|"':
                i += 1
            value = source[start:i]
        if depth == 0:
            if value != "(":
                raise ValueError("expected a top-level command")
            begin, tokens = start, []
        tokens.append(value)
        if value == "(":
            depth += 1
        elif value == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced command")
            if depth == 0:
                result.append((begin, i, tuple(tokens)))
    if depth:
        raise ValueError("unterminated command")
    return result


def symbol(token):
    if token.startswith('"') or token in ("(", ")"):
        return None
    return token[1:-1] if token.startswith("|") else token


def normalize(source):
    parsed = commands(source)
    removed, names = set(), []
    for index, (_, _, tokens) in enumerate(parsed):
        zero_width = any(tokens[k:k + 3] == ("_", "BitVec", "0") for k in range(len(tokens) - 2))
        if not zero_width:
            continue
        if not (len(tokens) == 11 and tokens[:2] == ("(", "declare-fun")
                and tokens[3:] == ("(", ")", "(", "_", "BitVec", "0", ")", ")")):
            raise ValueError("BV0 appears outside an isolated constant declaration")
        name = symbol(tokens[2])
        uses = [(j, position) for j, (_, _, values) in enumerate(parsed)
            for position, value in enumerate(values) if symbol(value) == name]
        if len(uses) != 2 or (index, 2) not in uses:
            raise ValueError("BV0 constant participates in more than a declaration and query")
        other, _ = next(use for use in uses if use != (index, 2))
        query = parsed[other][2]
        if len(query) != 6 or query[:3] != ("(", "get-value", "(") \
                or query[4:] != (")", ")") or symbol(query[3]) != name:
            raise ValueError("BV0 constant participates in a constraint or compound query")
        removed.update((index, other))
        names.append(name)
    output = source
    for index in sorted(removed, reverse=True):
        begin, end, _ = parsed[index]
        output = output[:begin] + output[end:]
    reparsed = commands(output)
    retained = [source[begin:end] for i, (begin, end, _) in enumerate(parsed) if i not in removed]
    if retained != [output[begin:end] for begin, end, _ in reparsed]:
        raise AssertionError("a retained solver command changed")
    assertions = [source[begin:end] for begin, end, tokens in parsed if tokens[1] == "assert"]
    if assertions != [output[begin:end] for begin, end, tokens in reparsed if tokens[1] == "assert"]:
        raise AssertionError("a solver assertion changed")
    return output, dict(removed_dead_symbols=names, removed_commands=len(removed),
        assertions_preserved=len(assertions), all_retained_commands_byte_identical=True)



def main():
    import os
    from pathlib import Path
    import shutil
    import signal
    import subprocess
    import sys
    import tempfile

    solver = shutil.which("cvc5")
    if solver is None:
        raise ValueError("cvc5 must be available on PATH")
    if sys.argv[1:] == ["--version"]:
        os.execv(solver, [solver, "--version"])
    inputs = [(index, Path(value)) for index, value in enumerate(sys.argv[1:])
              if Path(value).is_file()]
    if len(inputs) != 1:
        raise ValueError("exactly one SMT input file required")
    index, source = inputs[0]
    updated, _ = normalize(source.read_text())
    with tempfile.TemporaryDirectory(prefix="flare-cvc5-") as directory:
        normalized = Path(directory) / "input.smt2"
        normalized.write_text(updated)
        arguments = list(sys.argv[1:])
        arguments[index] = str(normalized)
        result = subprocess.run([solver, "--lang=smt2", "--incremental",
            "--solve-bv-as-int=sum", "--check-models", *arguments])
    if result.returncode < 0:
        number = -result.returncode
        if number not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(number, signal.SIG_DFL)
        os.kill(os.getpid(), number)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        import sys
        print("restricted CVC5 input adapter rejected input: " + str(error), file=sys.stderr)
        raise SystemExit(2) from error
