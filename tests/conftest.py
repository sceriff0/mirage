"""Session configuration for the MIRAGE Python suite, and the floor under it.

1. ``sys.path``: put the repository root and ``bin/`` on ``sys.path`` so tests
   can import the pipeline's Python without an installed package.

2. A floor under the suite (everything below ``STRICT_ENV``). Many modules call
   ``pytest.importorskip`` at module scope, so a degraded environment deselects
   much of the suite and still exits 0. In strict mode:

   * every skip whose nodeid is not in ``tests/expected_skips.txt`` fails the
     session, named, with its reason;
   * a module never collected because a module-scope ``importorskip`` fired is
     reported by file (it has no test nodeids);
   * ``session.testscollected`` must be at least ``tests/collected_floor.txt``,
     which catches a loss that leaves no skip behind (a deleted file, a grown
     ``--ignore``);
   * an allowlist entry that matched nothing also fails, so the allowlist can
     only shrink.

   Both committed files are ratchets; moving either is a one-line diff.

Gating: inert unless ``MIRAGE_STRICT_SKIPS`` is set to something other than
``0``/``false``/``no``/``off``/empty. CI sets it on the full-suite step only
(step level, not job level), because the same jobs also run single files, which
collect far below the floor.

Scope: the floor is measured with the suite's standard exclusions
(``pytest tests/ --ignore=tests/testdata --ignore=tests/modules
--ignore=tests/subworkflows --ignore=tests/integration``). A narrower selection
with the variable set fails on the count.

Not xdist-aware: the bookkeeping is module state in one process. Under
``pytest-xdist`` these hooks would have to move onto ``config`` and be merged
across workers.
"""

import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN_DIR = os.path.join(ROOT, "bin")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if BIN_DIR not in sys.path:
    sys.path.insert(0, BIN_DIR)


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------

STRICT_ENV = "MIRAGE_STRICT_SKIPS"
_HERE = Path(__file__).resolve().parent
ALLOWLIST_PATH = _HERE / "expected_skips.txt"
FLOOR_PATH = _HERE / "collected_floor.txt"

# Values that mean "off". Anything else -- including the documented `1` -- is on.
_OFF_VALUES = {"", "0", "false", "no", "off"}

# How far the collected count may run ahead of the committed floor before the
# floor must be re-ratcheted. The floor is a `>=` check so test additions never
# fail it; this band stops it going stale (a floor far below the suite would
# not notice a large loss). The failure message names the number to write.
FLOOR_DRIFT = 50

# nodeid -> (kind, reason), populated by the report hooks below.
_unexpected_skips = {}
# allowlist entries that actually matched a skip this session
_allowlist_hits = set()
# human-readable problem blocks, computed in sessionfinish, printed in the summary
_problems = []
_allowlist_cache = None


def _strict_mode_on():
    return os.environ.get(STRICT_ENV, "").strip().lower() not in _OFF_VALUES


def _allowlist():
    """The committed set of nodeids that are allowed to skip.

    Exact match, no globs and no prefixes: the entry names one test (or, for a
    module-scope skip, one file), so a NEW skip in an already-listed file --
    including a new parametrised case of an already-listed function -- is still
    reported.

    An entry prefixed `?` is environment-conditional: it suppresses the skip when
    it happens but is exempt from the stale-entry check, because whether it fires
    depends on the environment (e.g. CI's python-tests job has no `nextflow` on
    PATH). Use `?` only for a skip another valid environment would not produce;
    an unconditional entry that stops matching still fails.
    """
    global _allowlist_cache
    if _allowlist_cache is None:
        entries, conditional = set(), set()
        if ALLOWLIST_PATH.exists():
            for raw in ALLOWLIST_PATH.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("?"):
                    nodeid = line[1:].strip()
                    entries.add(nodeid)
                    conditional.add(nodeid)
                else:
                    entries.add(line)
        _allowlist_cache = (entries, conditional)
    return _allowlist_cache[0]


def _conditional_allowlist():
    """Entries marked `?` -- exempt from the stale check. See _allowlist()."""
    _allowlist()
    return _allowlist_cache[1]


def _committed_floor():
    if not FLOOR_PATH.exists():
        raise RuntimeError(
            f"{STRICT_ENV} is set but {FLOOR_PATH} is missing. The floor is a "
            "committed ratchet; a run cannot be strict without it."
        )
    for raw in FLOOR_PATH.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            return int(line)
    raise RuntimeError(f"{FLOOR_PATH} contains no count, only comments/blanks.")


def _skip_reason(report):
    longrepr = getattr(report, "longrepr", None)
    # A skip's longrepr is (filename, lineno, "Skipped: <reason>") for both
    # TestReport and CollectReport.
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2]).strip()
    return str(longrepr).strip() if longrepr else "<no reason recorded>"


def _record_skip(kind, nodeid, reason):
    if nodeid in _allowlist():
        _allowlist_hits.add(nodeid)
        return
    _unexpected_skips.setdefault(nodeid, (kind, reason))


def pytest_runtest_logreport(report):
    """Skips of tests that were collected and then declined to run."""
    if not _strict_mode_on() or report.outcome != "skipped":
        return
    # xfail also surfaces as `skipped`; it is a different, already-visible
    # mechanism and is not what this guard is about.
    if hasattr(report, "wasxfail"):
        return
    _record_skip("test", report.nodeid, _skip_reason(report))


def pytest_collectreport(report):
    """Skips that happened during COLLECTION -- the dangerous kind.

    A module-scope `pytest.importorskip` raises here, so the module's tests are
    never collected and never appear in `testscollected`. There is no test
    nodeid to report; the file is the finest granularity available.
    """
    if not _strict_mode_on() or report.outcome != "skipped":
        return
    _record_skip("module", report.nodeid, _skip_reason(report))


def pytest_sessionfinish(session, exitstatus):
    if not _strict_mode_on():
        return

    collected = session.testscollected
    floor = _committed_floor()

    if _unexpected_skips:
        lines = [
            f"{len(_unexpected_skips)} unexpected skip(s) -- a skipped test is "
            "not a passing test:",
        ]
        for nodeid, (kind, reason) in sorted(_unexpected_skips.items()):
            marker = "MODULE NOT COLLECTED" if kind == "module" else "skipped"
            lines.append(f"  [{marker}] {nodeid}")
            lines.append(f"      {reason}")
        lines.append(
            "If the environment is wrong, fix the environment. If the skip is "
            f"legitimate and permanent, add the nodeid to "
            f"{ALLOWLIST_PATH.relative_to(ROOT)} with a reason."
        )
        _problems.append("\n".join(lines))

    if collected < floor:
        _problems.append(
            f"only {collected} test(s) collected, below the committed floor of "
            f"{floor} ({FLOOR_PATH.relative_to(ROOT)}). Tests went missing "
            "without failing. If they were deliberately removed, lower the "
            "floor in the same commit."
        )
    else:
        if collected > floor + FLOOR_DRIFT:
            _problems.append(
                f"{collected} test(s) collected, more than {FLOOR_DRIFT} above "
                f"the committed floor of {floor}. A stale floor stops being a "
                f"floor. Re-ratchet: write {collected} into "
                f"{FLOOR_PATH.relative_to(ROOT)}."
            )
        # Only meaningful on a full run -- which `collected >= floor` establishes.
        stale = sorted(_allowlist() - _allowlist_hits - _conditional_allowlist())
        if stale:
            _problems.append(
                "allowlist entr(y/ies) in "
                f"{ALLOWLIST_PATH.relative_to(ROOT)} matched no skip this "
                "session, so they are exempting nothing. Delete them:\n"
                + "\n".join(f"  {entry}" for entry in stale)
            )

    if _problems:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _problems:
        return
    terminalreporter.write_sep("=", "MIRAGE suite floor", red=True, bold=True)
    for problem in _problems:
        terminalreporter.write_line(problem)
