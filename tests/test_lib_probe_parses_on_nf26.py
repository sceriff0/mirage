"""tests/lib_probe.nf must COMPILE on Nextflow 26, not just 25.

lib_probe.nf is the only unit-test surface lib/*.groovy has (nf-test's assertion
context cannot see lib/ classes). CI runs it on Nextflow 25 and 26; a script that
fails to compile skips every assertion in it. Two constructs NF26 rejects:

  1. `assert cond, 'message'` ("Unexpected input: ','"); use `assert cond : 'message'`.
  2. `someClosure(arg)` for a `def`-bound closure ("is not defined"); use
     `someClosure.call(arg)`.

A static stand-in for the CI matrix, which is the real check.
"""

import re
from pathlib import Path

from tests.nfmodel import strip_comments

PROBE = Path(__file__).resolve().parents[1] / "tests" / "lib_probe.nf"
SRC = strip_comments(PROBE.read_text())

# `assert <anything>, <quoted message>` at end of line, or with the message on
# the following line. Both are the comma form.
_COMMA_ASSERT = re.compile(r"^\s*assert\s+.*,\s*(?:'[^']*'|\"[^\"]*\")\s*$", re.M)
_COMMA_ASSERT_CONTINUED = re.compile(
    r"^\s*assert\s+[^\n]*,\s*\n\s*(?:'[^']*'|\"[^\"]*\")\s*$", re.M
)


def test_no_assert_uses_the_comma_message_form():
    offenders = [
        m.group(0).strip()
        for pat in (_COMMA_ASSERT, _COMMA_ASSERT_CONTINUED)
        for m in pat.finditer(SRC)
    ]
    assert not offenders, (
        "Nextflow 26's parser rejects `assert cond, 'msg'` with "
        "\"Unexpected input: ','\" and the whole script fails to compile, "
        "skipping every assertion in it. Use `assert cond : 'msg'`:\n  "
        + "\n  ".join(offenders)
    )


def test_no_def_bound_closure_is_invoked_as_a_function():
    """`def f = { ... }` then `f(x)` compiles on 25 and does not on 26."""
    closures = set(re.findall(r"^\s*def\s+(\w+)\s*=\s*\{", SRC, re.M))
    offenders = []
    for name in sorted(closures):
        for m in re.finditer(rf"(?<![.\w]){re.escape(name)}\s*\(", SRC):
            offenders.append(
                f"`{name}(...)` at offset {m.start()} -- a def-bound closure "
                f'invoked as a function; NF26 reports "`{name}` is not '
                f'defined". Use {name}.call(...)'
            )
    assert not offenders, "\n".join(offenders)


def test_the_scan_sees_the_probe():
    """If lib_probe.nf were renamed or emptied, both checks above would pass
    while covering nothing -- and the file they protect is the only unit-test
    surface lib/*.groovy has."""
    assert PROBE.exists(), "tests/lib_probe.nf is gone"
    n_asserts = len(re.findall(r"^\s*assert\s", SRC, re.M))
    assert n_asserts >= 100, (
        f"only {n_asserts} assertion(s) in lib_probe.nf -- it has been gutted, "
        "not merely reformatted"
    )
    assert re.search(r"^\s*def\s+\w+\s*=\s*\{", SRC, re.M), (
        "lib_probe.nf no longer binds any closure, so the second check above "
        "has nothing to walk"
    )


# ---------------------------------------------------------------------------
# ... and on Nextflow 25, whose legacy parser caps a workflow body at 64 K
# ---------------------------------------------------------------------------

# Nextflow 25's legacy parser keeps a `workflow {}` body's source as one Java string
# constant (max 65,535 bytes); over it, NF25 fails with "String too long" while NF26
# compiles. New checks go in def check...() functions; this pins a margin.
WORKFLOW_BODY_CAP = 65_535
WORKFLOW_BODY_MARGIN = 8_192


def _workflow_block_bytes() -> int:
    text = PROBE.read_text()
    start = text.index("\nworkflow {\n") + 1
    return len(text[start:].encode("utf-8"))


def test_the_workflow_block_stays_well_under_the_legacy_parser_string_cap():
    size = _workflow_block_bytes()
    assert size < WORKFLOW_BODY_CAP - WORKFLOW_BODY_MARGIN, (
        f"tests/lib_probe.nf's workflow {{}} block is {size} bytes; Nextflow 25's legacy "
        f"parser stores it as ONE string constant capped at {WORKFLOW_BODY_CAP} and the "
        f"probe then fails to COMPILE on the `NF 25.04.0 stub` leg (every assertion "
        f"skipped). Put new checks in a `def checkX()` ABOVE the block and call it, "
        f"as checkKeepSetRule() and the others do."
    )


def test_the_workflow_block_measure_sees_the_block():
    """The cap test is only worth anything if the measure found the block."""
    assert 10_000 < _workflow_block_bytes() < WORKFLOW_BODY_CAP
