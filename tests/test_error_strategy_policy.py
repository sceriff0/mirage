"""errorStrategy closures in conf/modules.config: only bound identifiers, only named policies.

An identifier config scope does not bind (e.g. `log` under Nextflow 25's config parser)
resolves against ConfigObject and throws when the closure runs past its retry budget,
aborting the run. `manifest.nextflowVersion` accepts 25.04, so this must hold there too.

Two properties are asserted:

  1. No errorStrategy closure names an identifier that config scope does not bind.
  2. Every errorStrategy closure is one of a small set of named policies, inlined
     verbatim (conf/*.config cannot share a helper between withName: blocks).
"""

import re

from tests.nfmodel import block_extent, strip_comments, with_name_blocks

# Bound inside a process-scope config closure. Everything else resolves against
# ConfigObject and throws only when the closure runs, which a green run does not reach.
BOUND = {
    "task",
    "params",
    "workflow",
    "meta",
    "Math",
    "System",
    "it",
    "file",
    "true",
    "false",
    "null",
    "def",
    "return",
    "if",
    "else",
    "in",
}

_IDENT = re.compile(r"\b([a-zA-Z_]\w*)\s*\.")

# The named policies. Keys are the names used in the conf/modules.config
# comments; values are the exact closure body each name stands for.
#
# The matrix is (which failures are worth retrying) x (what happens when the
# budget runs out). The fourth cell, selective retry then drop, is absent: a
# narrow cause selector ending in 'ignore' silently discards every cause outside
# the selector on attempt 1.
CANONICAL = {
    # retry-then-fail: signals are transient, anything else is a real bug.
    "retry-then-fail": "task.exitStatus in ((130..145) + 104) && task.attempt <= 3 "
    "? 'retry' : 'finish'",
    # retry-exit1-then-fail: this process's known-transient failures surface as
    # a plain exit 1 (VALIS tile reads, a JVM OOM inside a wrapper that exits
    # 1), so the cause selector cannot be narrowed to signals. The artifact is
    # still required, so the terminal branch fails the run.
    "retry-exit1-then-fail": "task.exitStatus in [1, 104, 134, 135, 137, 139, 140, 143] "
    "? 'retry' : 'finish'",
    # retry-then-drop: the artifact is optional; losing it degrades QC only.
    # Retries regardless of cause -- the common failure IS resource exhaustion
    # and the memory ramp is what fixes it.
    "retry-then-drop": "task.attempt <= 3 ? 'retry' : 'ignore'",
    # fail-fast: no retry is meaningful.
    "fail-fast": "'finish'",
}


def _normalise(s):
    return " ".join(s.split())


def _error_strategy_closures():
    """Yield (selector, line, code_view, text_view) per errorStrategy closure.

    Two views of the same span, because the two assertions below need opposite
    things from a string literal:

      code_view  -- comments AND strings blanked. The identifier scan reads
                    this, so `'a.b'` inside a quoted message cannot be reported
                    as an unbound `a.`.
      text_view  -- comments blanked, strings VERBATIM. The policy comparison
                    reads this, because `'retry'` and `'ignore'` ARE the
                    policy; on the code view they are blanked to spaces and
                    every policy would compare equal to every other.

    Both views come from tests/nfmodel, which preserves length and line count
    exactly, so one set of offsets indexes either one. The `errorStrategy = {`
    match itself is made on the fully-stripped view, so a commented-out or
    quoted occurrence cannot be picked up as a real closure.
    """
    for block in with_name_blocks():
        commentless = strip_comments(block.raw_body)
        for m in re.finditer(r"errorStrategy\s*=\s*\{", block.body):
            start = m.end()
            end = block_extent(block.body, start)  # index just past the `}`
            yield (
                block.selector,
                block.start_line,
                block.body[start : end - 1],
                commentless[start : end - 1],
            )


def test_no_error_strategy_closure_calls_an_unbound_identifier():
    offenders = []
    for selector, line, code, _text in _error_strategy_closures():
        for m in _IDENT.finditer(code):
            name = m.group(1)
            if name not in BOUND and not name[0].isupper():
                offenders.append(
                    f"conf/modules.config:{line} withName: '{selector}' "
                    f"calls `{name}.` -- unbound in config scope, resolves "
                    f"against ConfigObject and throws at closure-run time"
                )
    assert not offenders, "\n".join(offenders)


def test_every_error_strategy_is_one_of_the_named_policies():
    """Every errorStrategy closure is one of the named policies, inlined verbatim
    (conf/*.config cannot share a helper)."""
    allowed = {_normalise(v) for v in CANONICAL.values()}
    offenders = []
    for selector, line, _code, text in _error_strategy_closures():
        if _normalise(text) not in allowed:
            offenders.append(
                f"conf/modules.config:{line} withName: '{selector}': "
                f"{_normalise(text)!r} is not a named policy"
            )
    assert not offenders, "\n".join(offenders) + (
        "\n\nUse one of: " + ", ".join(CANONICAL)
    )


def test_no_policy_selects_on_cause_and_then_drops():
    """The empty cell of the matrix, asserted rather than merely described.

    A closure that both narrows on `task.exitStatus` and terminates in
    'ignore' discards every cause outside its selector on attempt 1 without
    saying so: a failed default-on QC task would give exit 0 and no output.
    """
    # `exitStatus` is read off the code view (a quoted "(exit ${task.exitStatus})" is
    # not a selector) and `'ignore'` off the text view (the code view blanks it).
    offenders = [
        f"conf/modules.config:{line} withName: '{selector}' selects on "
        f"task.exitStatus and terminates in 'ignore'"
        for selector, line, code, text in _error_strategy_closures()
        if "exitStatus" in code and "'ignore'" in text
    ]
    assert not offenders, "\n".join(offenders)


def test_the_scan_is_not_vacuous():
    """conf/modules.config has errorStrategy closures, and each one has a
    non-empty body under BOTH views. If this finds none -- or finds spans that
    the string-blanking view emptied -- the extractor is stale, not the config.
    """
    found = list(_error_strategy_closures())
    assert found, "no errorStrategy closures found in conf/modules.config"
    for selector, line, code, text in found:
        assert code.strip(), f"conf/modules.config:{line} '{selector}': empty code view"
        assert text.strip(), f"conf/modules.config:{line} '{selector}': empty text view"
    # Both views must disagree, or `text` is silently the blanked one and the
    # policy comparison is testing nothing.
    assert any(code != text for _s, _l, code, text in found), (
        "code and text views are identical -- strip_comments is not preserving "
        "string literals, so the policy comparison is vacuous"
    )
