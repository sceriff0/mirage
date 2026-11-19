#!/usr/bin/env python3
"""Guard: nothing in this repository ends a command with `|| true`.

`|| true` makes a failed command report success, so a check passes while proving
nothing. The rule is enforced here over discovered files rather than kept as a list.

Where the exit status carries data (`grep -c` exits 1 for zero matches), write the
assignment (`n=$(...) || n=0`). Beware `A && B || true`: it also swallows a failed `B`.

Scope: every `run:` body in `.github/workflows/**` and `.github/actions/**` (via
`tests/ci_actions.py`), every tracked shell script, `.readthedocs.yaml`, every
Nextflow process `script:`/`stub:` body (via `tests/nfmodel`), every
`containers/*/Dockerfile`, and the `Makefile`.
`test_the_scan_covers_the_build_files` probes the last two.

Shell comments, whole-line and trailing, are stripped: a comment explaining a removed
`|| true` quotes it.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
ci_actions = importlib.import_module("ci_actions")

from tests.nfmodel import processes, strip_comments  # noqa: E402

NEEDLE = "|| true"

# Allowed occurrences, keyed by (path, exact command) so an exemption cannot cover a
# second one added to the same file. Keep it as close to empty as possible;
# test_the_allowlist_has_no_dead_entries fails once the command changes.
ALLOWED: dict[tuple[str, str], str] = {
    (
        "modules/local/register.nf",
        'find -L ref input_* -maxdepth 1 -type f \\\\( -name "*.ome.tif" -o -name '
        '"*.ome.tiff" \\\\) 2>/dev/null > files_to_copy.txt || true',
    ): (
        "The same 'nonzero means zero matches' idiom as the two rewritten above: "
        "`find` exits 1 when a named path does not exist, and a REGISTER task whose "
        "`input_*` glob matched nothing is a case the script handles rather than an "
        "error. The real, named assertion is twenty lines below in the same block -- "
        '`if [ "$file_count" -eq 0 ]; then ... exit 1; fi` -- so nothing is '
        "swallowed; it is simply not on the NEXT line, which is why this is an entry "
        "here instead of a rewrite.\n"
        "It was left alone deliberately, and the cost of changing it is the reason: "
        "this is inside a process `script:` block, so any edit changes the rendered "
        "command, changes the task hash, and invalidates the cached REGISTER task in "
        "every existing work directory -- the most expensive process in the pipeline "
        "(measured at 483 GB and hours per task). A CI-hygiene phase is not where "
        "that gets spent. If REGISTER's script is being rewritten for another reason, "
        "fold the `if` in then and delete this entry."
    ),
}

# Non-vacuity floors: a "must not find" scan over an empty set passes. Each scope can
# go quietly empty (a resolver, a `git ls-files` filter, the process model).
MIN_CI_COMMAND_LINES = 100
MIN_SHELL_SCRIPTS = 3
MIN_PROCESSES = 15
# containers/images.json lists eleven images and each has a Dockerfile, plus the
# Makefile: twelve. Expressed as a floor rather than an equality so a twelfth image
# does not fail this, but low enough that a broken `git ls-files` pattern does.
MIN_BUILD_FILES = 12


def _tracked(*patterns: str) -> list[Path]:
    """Tracked files matching `patterns`, via `git ls-files`.

    Tracked rather than globbed: `.gitignore` covers per-worktree scratch (`tests/testdata/`, `.nf-test/`, virtualenvs), and a glob would scan
    those and report a `|| true` nobody committed.
    """
    out = subprocess.run(
        ["git", "ls-files", *patterns],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [REPO / rel for rel in out]


def _shell_scripts() -> list[Path]:
    """Every tracked shell script: ``*.sh``, and the SLURM job scripts (``*.sbatch``),
    which are bash run by the scheduler. The second pattern was added on 2026-10-17
    (regbench fleet wave 2, F5): five ``|| true`` sat in the benchmark's job scripts, one
    of them hiding a failed ``sacct`` while its log said the table was written."""
    return _tracked("*.sh", "*.sbatch")


def _build_files() -> list[Path]:
    """The container Dockerfiles and the Makefile.

    Both are shell: a Dockerfile `RUN` body is `/bin/sh -c`, and a Makefile recipe
    line is a shell command whose exit status Make reads. `|| true` swallows a
    failure in either exactly as it does in a workflow step. Each Dockerfile ends in
    `RUN bash /usr/local/bin/mirage-smoke.sh`, which runs on the release path.
    """
    return _tracked("containers/*/Dockerfile", "Makefile")


def _clean(text: str) -> str:
    """Commands only: whole-line and trailing shell comments removed."""
    return ci_actions.strip_shell_comments(text)


def _hits(where: str, text: str) -> list[tuple[str, str]]:
    return [(where, line.strip()) for line in text.splitlines() if NEEDLE in line]


def _all_hits() -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []

    for path in ci_actions.scanned_files():
        rel = path.relative_to(REPO).as_posix()
        hits += _hits(rel, _clean(ci_actions.run_scripts(path)))

    for path in _shell_scripts():
        hits += _hits(path.relative_to(REPO).as_posix(), _clean(path.read_text()))

    rtd = REPO / ".readthedocs.yaml"
    if rtd.is_file():
        hits += _hits(".readthedocs.yaml", _clean(rtd.read_text()))

    for path in _build_files():
        hits += _hits(path.relative_to(REPO).as_posix(), _clean(path.read_text()))

    for proc in processes().values():
        rel = proc.path.relative_to(REPO).as_posix()
        body = strip_comments(proc.script_body) + "\n" + strip_comments(proc.stub_body)
        hits += _hits(rel, body)

    # Deduplicate: `run_scripts` resolves a composite action into every workflow that
    # uses it, so one line in an action would otherwise be reported many times.
    return sorted(set(hits))


def test_the_scan_actually_read_something():
    """Each scope must be non-empty: a scan over nothing finds nothing."""
    ci_text = "\n".join(
        ci_actions.run_scripts(path) for path in ci_actions.scanned_files()
    )
    ci_lines = [line for line in ci_text.splitlines() if line.strip()]
    assert len(ci_lines) >= MIN_CI_COMMAND_LINES, (
        f"only {len(ci_lines)} command line(s) resolved out of "
        f"{[p.name for p in ci_actions.scanned_files()]}; the workflows and actions "
        "run far more than that, so ci_actions stopped resolving them."
    )
    assert len(_shell_scripts()) >= MIN_SHELL_SCRIPTS, (
        f"git ls-files '*.sh' returned {len(_shell_scripts())} script(s); this repo "
        "tracks several under tests/, so the scope glob is wrong."
    )
    assert len(processes()) >= MIN_PROCESSES, (
        f"tests.nfmodel found {len(processes())} process(es); modules/local/ holds far "
        "more, so the model read nothing."
    )
    assert len(_build_files()) >= MIN_BUILD_FILES, (
        f"git ls-files matched {len(_build_files())} build file(s); this repo tracks "
        "eleven containers/*/Dockerfile plus a Makefile, so the pattern is wrong and "
        "the scope this file was widened to cover is being read as empty."
    )


def test_the_scan_is_comment_blind():
    """Comments, whole-line and trailing, are ignored; real commands are still seen.

    Checked on synthetic input, including a `#` inside a quoted string.
    """
    assert NEEDLE not in _clean("# nf-core lint --dir . || true")
    assert NEEDLE not in _clean("echo hello  # this used to be `foo || true`")
    # ...and a REAL one is still seen, including after a string containing a `#`.
    assert NEEDLE in _clean("foo || true")
    assert NEEDLE in _clean("""echo "a # b" && foo || true""")


def test_nothing_swallows_a_failure_with_or_true():
    offenders = [
        f"{where}: {line}"
        for where, line in _all_hits()
        if (where, line) not in ALLOWED
    ]
    assert not offenders, (
        "`|| true` makes a command that failed report success, which is the visible "
        "way a check passes while proving nothing -- `nf-core lint --dir . || true` "
        "ran for months over a command that could not lint this repository at all.\n\n"
        "If the exit status genuinely carries data rather than failure (`grep -c` "
        "exits 1 to mean ZERO MATCHES), write the assignment: `n=$(...) || n=0`, with "
        "the real assertion next to it. If the command may legitimately be a no-op, "
        "ask the question directly (`git rev-parse --is-shallow-repository`) instead "
        "of discarding the answer. Only if neither is possible, add it to ALLOWED "
        "with the command and a reason.\n\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "rel",
    ["containers/merge/Dockerfile", "Makefile"],
)
def test_the_scan_covers_the_build_files(rel):
    """A `|| true` appended to a Dockerfile or the Makefile is reported.

    Appends a probe to a real file on disk, asserts the scan reports it, and restores
    the file. This checks the reader (`_all_hits`), not just the scope constant.
    """
    path = REPO / rel
    assert path.is_file(), f"{rel} is gone; this probe is checking nothing"
    original = path.read_text()
    probe = (
        "RUN echo scope-probe || true\n"
        if rel.endswith("Dockerfile")
        else "scope-probe:\n\techo scope-probe || true\n"
    )
    try:
        path.write_text(original + "\n" + probe)
        seen = {where for where, _ in _all_hits()}
        assert rel in seen, (
            f"a `|| true` appended to {rel} is invisible to this scan. {rel} is shell "
            "that CI and the release path execute; add it to _build_files()."
        )
    finally:
        path.write_text(original)
    # ...and the tree is back the way it was, so the main scan is clean again.
    assert rel not in {where for where, _ in _all_hits()}


@pytest.mark.parametrize(
    "entry", sorted(ALLOWED), ids=[f"{w}:{c[:40]}" for w, c in sorted(ALLOWED)]
)
def test_the_allowlist_has_no_dead_entries(entry):
    """An exemption that exempts nothing reads as a reviewed decision; it must go."""
    assert entry in _all_hits(), (
        f"ALLOWED entry {entry} matches no `|| true` in the tree any more. Either the "
        "command was rewritten -- in which case delete the entry in the same commit -- "
        "or it moved and the entry now exempts nothing while reading as though it does."
    )
