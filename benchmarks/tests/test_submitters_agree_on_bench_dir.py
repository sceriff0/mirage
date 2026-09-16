"""The three submitters must name ONE benchmark directory, and it must be overridable.

submit_matrix.sh writes bench_matrix/, submit_sweep.sh reads it and writes
bench_results/, submit_arms.sh writes arm_results/. They are three jobs against one
directory, so a disagreement is not a style question: whichever one drifts either
writes where nothing reads, or dies on `cd`.

Both happened. submit_matrix.sh kept a hardcoded BENCH_DIR pointing at a directory
that no longer exists while the other two had moved on and made theirs overridable,
so job 6831427 exited in under a second with "cannot cd to ...", and the matrix the
sweep waits on was never generated (2026-09-16).

Pinned here:
  * all three take the directory from the environment with the same default;
  * each header's documented `Submit: cd <path>` line names that same default, because
    a doc line that disagrees with the code is how this drifted in the first place.

Comment-blind through ci_actions.strip_line_comment for the code, and deliberately
comment-BLIND-in-reverse for the header: the Submit line IS a comment, so it is read
from the raw text on purpose.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks"
sys.path.insert(0, str(REPO / "tests"))
strip_line_comment = importlib.import_module("ci_actions").strip_line_comment

SUBMITTERS = ("submit_matrix.sh", "submit_arms.sh", "submit_sweep.sh")


def _code(name: str) -> str:
    text = (BENCH / name).read_text()
    return "\n".join(strip_line_comment(line) for line in text.splitlines())


def _default(name: str) -> str:
    """The BENCH_DIR default, and proof it is an env override rather than a literal."""
    m = re.search(r'^BENCH_DIR="\$\{BENCH_DIR:-([^}"]+)\}"$', _code(name), re.M)
    assert m, (
        f"{name} does not take BENCH_DIR from the environment. Write it as "
        'BENCH_DIR="${BENCH_DIR:-<default>}" so one run can be pointed elsewhere '
        "without editing the script."
    )
    return m.group(1)


def test_all_three_submitters_default_to_the_same_benchmark_directory():
    defaults = {name: _default(name) for name in SUBMITTERS}
    assert len(set(defaults.values())) == 1, (
        "the submitters disagree about where the benchmark lives, so one of them reads "
        "or writes a directory the others do not:\n  "
        + "\n  ".join(f"{n}: {d}" for n, d in sorted(defaults.items()))
    )


def test_each_header_documents_the_directory_the_script_actually_uses():
    offenders = []
    for name in SUBMITTERS:
        raw = (BENCH / name).read_text()
        m = re.search(r"^#\s*Submit:\s*cd\s+(\S+)", raw, re.M)
        assert m, f"{name} header no longer documents where to submit from"
        if m.group(1) != _default(name):
            offenders.append(
                f"{name}: header says {m.group(1)}, code uses {_default(name)}"
            )
    assert not offenders, (
        "a submitter's documented directory disagrees with the one it uses:\n  "
        + "\n  ".join(offenders)
    )
