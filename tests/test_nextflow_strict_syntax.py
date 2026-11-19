#!/usr/bin/env python3
"""Guards for Nextflow 26's strict (v2) parser.

Nextflow 26 uses only the v2 syntax parser, which rejects several constructs
the v1 parser accepted; a violation aborts config parsing with
`ERROR ~ Config parsing failed`. Reproduce the parser's verdict with:

    NXF_SYNTAX_PARSER=v2 nextflow -q run . -profile test -stub

Each rule is its own test function so a failure names the rule that broke.
Plain pytest; no Nextflow install required.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONF_DIR = ROOT / "conf"

# A function declaration at the top level of a config file. Under the strict
# parser this is `Unexpected input: '('`, reported against the open paren.
# Leading `\s*`, not column 0: an indented declaration is just as illegal.
_CONFIG_DEF_FUNCTION_RE = re.compile(r"^\s*def\s+\w+\s*\(")


def _config_files() -> list[Path]:
    """Every parsed config file. `conf/site.config.template` is not matched:
    it is a template to copy, never included."""
    return sorted(CONF_DIR.glob("*.config"))


def test_config_files_are_present_to_scan():
    """Fail if the glob stops finding conf/modules.config, so the rule below
    cannot pass vacuously over an empty file list."""
    files = _config_files()
    assert files, f"no conf/*.config files found under {CONF_DIR}"
    assert CONF_DIR / "modules.config" in files, (
        f"conf/modules.config is not in the scanned set: {[f.name for f in files]}"
    )


def test_rule_1_no_top_level_def_functions_in_config():
    """Rule 1 — no `def helper(...)` in any `conf/*.config`.

    There is no legal way to share a helper between config blocks under the
    strict parser, so the fix is always to inline the body at each call site:

      * `def helper(params) { ... }`  -> `Unexpected input: '('`
      * `helper = { ... }`            -> `Dynamic config options only allowed
                                          in process scope`
      * moving it to `lib/*.groovy`   -> not visible from conf/*.config at all;
                                         the name resolves silently against the
                                         enclosing ConfigObject and blows up
                                         only when the closure runs, with a
                                         misleading MissingMethodException.
    """
    offenders: list[str] = []
    for path in _config_files():
        lines = path.read_text().splitlines()
        for lineno, line in enumerate(lines, start=1):
            if _CONFIG_DEF_FUNCTION_RE.match(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Nextflow 26's strict config parser rejects a function declaration in a "
        "config file with `Unexpected input: '('`. Inline the body at each call "
        "site instead — no other form is legal (see this test's docstring). "
        "Offenders:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Rules 2-5 — the `.nf` sources.
#
# These scan `main.nf` plus every `*.nf` under workflows/, subworkflows/ and
# modules/. Out of scope: `lib/` (plain Groovy, parses unchanged under v2) and
# `*.nf.test` (read by nf-test's own parser).
# ---------------------------------------------------------------------------

_NF_SOURCE_DIRS = ("workflows", "subworkflows", "modules")


def _nf_files() -> list[Path]:
    """Every Nextflow source file the strict parser will compile.

    `*.nf` matches `foo.nf` but not `foo.nf.test`, so the nf-test suite stays
    out of the sweep without needing an explicit exclusion.
    """
    files = [ROOT / "main.nf"]
    for d in _NF_SOURCE_DIRS:
        files.extend(sorted((ROOT / d).rglob("*.nf")))
    return files


def test_nf_files_are_present_to_scan():
    """Fail if the globs stop finding the known sources, so the sweeps below
    cannot pass vacuously."""
    files = _nf_files()
    rel = {str(f.relative_to(ROOT)) for f in files}
    assert ROOT / "main.nf" in files, "main.nf missing from the scanned set"
    for expected in (
        "workflows/mirage.nf",
        "subworkflows/local/final_qc.nf",
        "subworkflows/local/add_cycle.nf",
        "subworkflows/local/segmentation.nf",
    ):
        assert expected in rel, f"{expected} missing from the scanned set"
    assert not any(str(f).endswith(".nf.test") for f in files), (
        "the nf-test suite leaked into the .nf sweep"
    )


# ---------------------------------------------------------------------------
# Rule 2 — no Groovy `import` declaration.
# ---------------------------------------------------------------------------

_IMPORT_RE = re.compile(r"^\s*import\s+")


def test_rule_2_no_groovy_import_declarations():
    """Rule 2 — the strict parser has no `import` declaration at all.

    `include { X } from '...'` is the only import form v2 knows. A Groovy
    `import groovy.json.JsonOutput` is rejected outright. The fix is to use the
    fully-qualified name at each use site (`groovy.json.JsonOutput.toJson(...)`).

    Matched with a leading `\\s*` rather than at column 0 because an indented
    import is just as illegal. `include` lines do not match: the keyword differs.
    """
    offenders: list[str] = []
    for path in _nf_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if _IMPORT_RE.match(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Nextflow 26's strict parser rejects Groovy `import` declarations in "
        "`.nf` files. Use the fully-qualified class name at each use site "
        "instead. Offenders:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Rule 3 — no two-argument `assert`.
# ---------------------------------------------------------------------------

_ASSERT_RE = re.compile(r"^\s*assert\b")
# A line that plainly continues onto the next one. Used only to decide how far
# to accumulate a multi-line `assert` statement.
_CONTINUES_RE = re.compile(r"(,|\+|&&|\|\||\\)\s*$")


def _has_top_level_comma(text: str) -> bool:
    """True if `text` contains a comma outside every bracket and quote.

    This is the whole reason Rule 3 is a scanner and not a bare regex. A regex
    like `^\\s*assert\\s+[^,]+,` cannot tell these apart:

        assert col in Checkpoint.columns(X),   <- two-arg assert  (VIOLATION)
        assert foo.method(a, b)                <- one-arg assert  (fine)
        assert report.contains('a, b')         <- one-arg assert  (fine)

    Only the first has a comma at bracket depth 0 outside a string, and only the
    first is what the strict parser rejects.
    """
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            return True
        i += 1
    return False


def test_rule_3_no_two_argument_assert():
    """Rule 3 — `assert expr, "message"` is not valid strict syntax.

    v2 keeps single-argument `assert` but drops the two-argument form. Where
    the message matters, use an explicit conditional that throws it:

        if (!(cond)) { throw new IllegalStateException("<same message>") }

    Detection is `^\\s*assert\\b` to find the statement, then a bracket/quote
    aware scan for a comma at depth 0 (see `_has_top_level_comma`). The scan
    continues onto following lines while the statement is plainly unfinished, so
    an `assert` whose first argument wraps is still caught.
    """
    offenders: list[str] = []
    for path in _nf_files():
        lines = path.read_text().splitlines()
        for lineno, line in enumerate(lines, start=1):
            if not _ASSERT_RE.match(line):
                continue
            stmt = line
            j = lineno  # index of the NEXT line, 0-based
            while (
                not _has_top_level_comma(stmt)
                and _CONTINUES_RE.search(stmt)
                and j < len(lines)
            ):
                stmt += " " + lines[j].strip()
                j += 1
            if _has_top_level_comma(stmt):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Nextflow 26's strict parser accepts only single-argument `assert`. "
        "Replace each with `if (!(cond)) { throw new IllegalStateException(msg) }`, "
        "keeping the message verbatim. Offenders:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Rule 4 — no bare top-level assignment in a subworkflow.
# ---------------------------------------------------------------------------

# Anchored at column 0 and to SCREAMING_CASE (a script-level pseudo-constant).
# `(?!=)` keeps a comparison such as `FOO == bar` out.
_BARE_TOP_LEVEL_ASSIGN_RE = re.compile(r"^[A-Z_][A-Z0-9_]*\s*=(?!=)")


def test_rule_4_no_bare_top_level_assignment_in_subworkflows():
    """Rule 4 — a bare `FOO = ...` at the top level of a subworkflow is a
    statement, and the strict parser allows only declarations there
    (`Statements cannot be mixed with script declarations`).

    A script-level `def` is not a fix either: it is scoped to the file's
    implicit `run()` method and invisible to functions in the same file. Remove
    the need for file-level state instead (see subworkflows/local/final_qc.nf).
    """
    offenders: list[str] = []
    for path in sorted((ROOT / "subworkflows").rglob("*.nf")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if _BARE_TOP_LEVEL_ASSIGN_RE.match(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Nextflow 26's strict parser rejects a bare top-level assignment in a "
        "`.nf` file — only declarations (include/def/process/workflow) may sit "
        "at the top level. Offenders:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Rule 5 — no top-level `workflow.onComplete` / `workflow.onError`.
# ---------------------------------------------------------------------------

# Anchored at column 0, not `^\s*`: a handler registered inside a
# `workflow { }` block is legal under v2 (it is the fix) and is indented.
_TOP_LEVEL_HANDLER_RE = re.compile(r"^workflow\.on(Complete|Error)\b")


def test_rule_5_no_top_level_workflow_handlers():
    """Rule 5 — `workflow.onComplete { }` at the top level of a script is a
    statement, so v2 rejects it with `Statements cannot be mixed with script
    declarations -- move statements into a process or workflow`.

    Registering the handler from inside the entry `workflow { }` block is
    accepted by both parsers, but the closure's binding is NOT the script
    binding: `params`, `workflow` and `projectDir` all resolve to null inside
    it under v1 *and* v2. Anything the handler needs must therefore be captured
    into a local in the workflow body first, or reached through a script-level
    function (where `log` and friends resolve normally).
    """
    offenders: list[str] = []
    for path in _nf_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if _TOP_LEVEL_HANDLER_RE.match(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Nextflow 26's strict parser rejects a top-level `workflow.onComplete` "
        "/ `workflow.onError` handler. Register it from inside the entry "
        "`workflow { }` block, capturing what it needs into a local first. "
        "Offenders:\n  " + "\n  ".join(offenders)
    )
