"""tests/lib_probe.nf must COMPILE on Nextflow 26, not just 25.

lib_probe.nf is the only unit-test surface lib/*.groovy has -- nf-test's
assertion context cannot see lib/ classes (tests/layout.nf.test:5 records this),
and the repo has no JVM test runner. CI's `nextflow-stub` job runs it in BOTH
matrix legs, `25.04.0` and `latest-everything`.

It did not compile on the second one. Two constructs Nextflow 26's strict parser
rejects had accumulated, and the failure mode is the worst available: the script
fails to COMPILE, so every assertion in it is skipped, and the only signal is one
red step in one matrix leg.

  1. `assert cond, 'message'`  -- the old Groovy comma form.
     NF26: "Unexpected input: ','". The documented form is
     `assert cond : 'message'`, which parses on both. 31 sites.

  2. `someClosure(arg)` where `someClosure` is a `def`-bound closure.
     NF26: "`keysOf` is not defined". It must be `someClosure.call(arg)`.
     workflows/mirage.nf already carries a comment recording this same
     restriction ("the strict Nextflow parser cannot invoke a closure-typed
     local as a function"); the probe had not learned it.

Verified after the fix, both directions on both engines: unmodified exits 0 on
25.04.7 and 26.04.6, and with one assertion deliberately falsified it exits 1 on
both -- so the assertions genuinely run rather than merely compiling.

This guard is a cheap static stand-in for that. A full compile check would mean
launching Nextflow twice from pytest, which the Python CI job has no engine for;
CI's own two-leg matrix is the real check, and this is what fails fast in the
suite the author runs first.
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

# Java's class-file format stores a string constant in at most 65,535 UTF-8 bytes,
# and the LEGACY parser (Nextflow 25) keeps a `workflow {}` body's SOURCE as one such
# constant. Measured 2026-09-13: the block stood at 65,513 bytes, three added lines
# made it 65,676, and Nextflow 25.04.7 refused the whole script with
#     Script compilation error - cause: String too long. The given string is 65663
#     Unicode code units long, but only a maximum of 65535 is allowed.
# while 26.04.6 (the v2 parser) compiled it fine -- the exact "green on one matrix
# leg" failure this file exists for, in the other direction. Two sections were moved
# into def check...() functions above the block (48,577 bytes after), and this pins a
# margin so the NEXT inline addition fails here, with the reason, not in CI's
# `NF 25.04.0 stub` leg with a message that names no line.
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
