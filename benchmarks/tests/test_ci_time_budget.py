"""The benchmarks workflow runs every test, slow ones included, inside its time limit.

`slow` is a label for developers (`-m "not slow"`), never a way to shorten CI: a test
that is deselected there is a test that runs nowhere. This guard reads the workflow's
commands with their comments removed (a comment naming a command satisfies a text match
and runs nothing).
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "benchmarks.yml"
JOB = "benchmarks"
# The whole directory was measured at 17 to 22 minutes on a laptop (2026-10-11).
MEASURED_MINUTES = 22

if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))
ci_actions = importlib.import_module("ci_actions")


def _job():
    jobs = ci_actions.load_jobs(WORKFLOW)
    assert JOB in jobs, f"{WORKFLOW.name} has no `{JOB}` job; found {sorted(jobs)}"
    return jobs[JOB]


def _pytest_commands(job):
    script = ci_actions.strip_shell_comments(ci_actions.job_run_text(job))
    return [ln.strip() for ln in script.splitlines() if "pytest" in ln]


def test_the_job_has_room_for_the_measured_run():
    limit = _job().get("timeout-minutes")
    assert isinstance(limit, int), "the job must state its own time limit"
    assert limit >= 2 * MEASURED_MINUTES - 1, (
        f"timeout-minutes is {limit}; the suite was measured at up to "
        f"{MEASURED_MINUTES} minutes locally, and a runner is slower"
    )
    assert limit <= 60, "a hang should fail within the hour"


def test_ci_runs_every_test_slow_ones_included():
    commands = _pytest_commands(_job())
    assert commands, "the benchmarks job runs no pytest command"
    whole = [c for c in commands if re.search(r"pytest\s+benchmarks/tests(\s|$)", c)]
    assert whole, f"no command runs all of benchmarks/tests: {commands}"
    for command in commands:
        options = command.split("pytest", 1)[1] + " "  # not python's own `-m`
        for flag in (" -m ", " -m=", " -k ", " -k=", "--deselect", "--ignore"):
            assert flag not in options, (
                f"`{command}` leaves tests out ({flag.strip()}): `slow` is a label, "
                "CI runs everything"
            )


def test_the_slow_marker_is_registered_where_every_benchmark_test_sees_it():
    text = (REPO / "benchmarks" / "tests" / "conftest.py").read_text()
    assert 'addinivalue_line(\n        "markers",\n        "slow:' in text
    marked = [
        p
        for p in (REPO / "benchmarks" / "tests").rglob("test_*.py")
        if "@pytest.mark.slow" in p.read_text() and p.name != Path(__file__).name
    ]
    assert marked, "the marker is registered and no test carries it"
