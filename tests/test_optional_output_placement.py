"""`optional: true` written inside `path(...)` is silently ignored.

`optional` is a qualifier on the output DECLARATION, not an argument to
`path()`. It belongs beside `emit:`, outside the parens:

    tuple val(meta), path("qc/*.png"), emit: qc, optional: true

Written inside them, Nextflow parses `optional: true` as one more entry in
`path()`'s own argument map -- the same slot that carries `stageAs:`,
`arity:` and `followLinks:` -- accepts it, and drops it. The declaration stays
mandatory, so the task FAILS when the glob matches nothing.

Under conf/modules.config's retry-then-fail QC selector, that failure stops
the run.
"""

import re

from tests.nfmodel import REPO_ROOT, processes

# `[^)]*` stops at the first `)`: `optional:` has to appear inside
# path()'s OWN argument list to be the defect. A declaration that closes the
# parens first and then writes `, optional: true` is the correct form and must
# not be reported.
_BAD = re.compile(r"path\s*\([^)]*\boptional\s*:")

# The correct form, asserted separately so this file cannot pass by finding
# nothing anywhere.
_GOOD = re.compile(r"\)\s*,[^\n]*\boptional\s*:")


def test_optional_is_never_written_inside_path():
    offenders = []
    for name, proc in processes().items():
        for m in _BAD.finditer(proc.outputs):
            offenders.append(
                f"{proc.path.relative_to(REPO_ROOT)} ({name}): {m.group(0).strip()}..."
            )
    assert not offenders, (
        "`optional:` inside path(...) is parsed as one of path()'s own "
        "arguments and dropped; the output stays mandatory and the task fails "
        "on a missing file. Put the qualifier on the declaration, after the "
        "closing paren:\n  " + "\n  ".join(offenders)
    )


def test_the_scan_sees_real_outputs():
    """A `processes()` walk that returned empty output sections would pass the
    check above vacuously."""
    with_outputs = [n for n, p in processes().items() if p.outputs.strip()]
    assert len(with_outputs) >= 10, (
        f"only {len(with_outputs)} process(es) with a parsed output: section -- "
        "the model is stale, not the modules"
    )


def test_the_correct_form_is_what_the_repo_actually_uses():
    """Pins the shape the guard is steering toward. If this stops matching, the
    guard above is enforcing a form that no longer exists anywhere and the next
    person will 'fix' it by deleting the qualifier instead of moving it."""
    users = sorted(n for n, p in processes().items() if _GOOD.search(p.outputs))
    assert len(users) >= 4, (
        "no process writes `optional:` outside path(...) any more; the guard "
        f"above has no positive example left to point at. Found: {users}"
    )
