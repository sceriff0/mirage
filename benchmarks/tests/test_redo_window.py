"""REDO_LAUNCHED_BY / REDO_SINCE find the runs to redo from the results root, not logs.

The head logs of the two colliding jobs were deleted (2026-09-29); what survives is each
run's .nextflow/history in its launch dir and SLURM's accounting of when the jobs ran.
These tests drive submit_arms.sh's own REDO block (extracted, so nothing is launched)
with a fake `sacct` and hand-written histories, and pin that:

  * the window is the jobs' earliest start to latest end, a running job ending "now";
  * a run launched in it is selected -- a base, a resumption (-rN) and a QC cross whose
    history line sits in its BASE's launch dir -- and nothing launched before it is;
  * REDO_SINCE/REDO_UNTIL give the window without sacct;
  * the selection becomes an anchored ONLY regex with ARMS_REPLACE=1.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]
SUBMIT = BENCH / "submit_arms.sh"


def _block() -> str:
    s = SUBMIT.read_text()
    m = re.search(
        r'^REDO_LAUNCHED_BY="\$\{REDO_LAUNCHED_BY:-\}"\n.*?^fi\n', s, re.S | re.M
    )
    assert m, "the REDO block moved; update this test's extraction"
    return m.group(0)


def _hist(root: Path, launch_dir: str, *lines: tuple[str, str, str]) -> None:
    d = root / ".launch" / launch_dir / ".nextflow"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "history", "a") as fh:
        for ts, name, status in lines:
            fh.write(f"{ts}\t1h\t{name}\t{status}\tabc\tsess-{name}\tnextflow run x\n")


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "arm_results"
    _hist(
        r, "preprocess_shared", ("2026-09-20 08:00:00", "arms-preprocess_shared", "OK")
    )
    _hist(
        r,
        "valis_high_micro2",
        ("2026-09-21 09:00:00", "arms-valis_high_micro2", "OK"),  # before: kept
        ("2026-09-29 10:40:00", "arms-valis_high_micro2_segstardist", "-"),  # cross, in
    )
    _hist(
        r,
        "tiled_low_s128",
        ("2026-09-28 09:00:00", "arms-tiled_low_s128", "-"),
        ("2026-09-29 11:00:00", "arms-tiled_low_s128-r2", "-"),  # resumption, in
    )
    _hist(r, "tiled_high_gate1", ("2026-09-29 10:30:00", "arms-tiled_high_gate1", "OK"))
    _hist(r, "valis_low_micro0", ("2026-09-29 13:30:00", "arms-valis_low_micro0", "OK"))
    return r


def _run(tmp_path, root, sacct_out: str | None = None, **env):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    if sacct_out is not None:
        (bindir / "sacct").write_text(f"#!/usr/bin/env bash\nprintf '{sacct_out}'\n")
        (bindir / "sacct").chmod(0o755)
    script = tmp_path / "block.sh"
    script.write_text(
        f'RESULTS="{root}"\nSRC_DIR="{BENCH.parent}"\n'
        + _block()
        + 'echo "ONLY=$ONLY"\necho "ARMS_REPLACE=${ARMS_REPLACE:-}"\n'
    )
    e = {k: v for k, v in os.environ.items() if not k.startswith(("REDO_", "ONLY"))}
    e["PATH"] = f"{bindir}:{e['PATH']}"
    e.update(env)
    return subprocess.run(["bash", str(script)], env=e, capture_output=True, text=True)


def _only(r) -> set[str]:
    (line,) = [ln for ln in r.stdout.splitlines() if ln.startswith("ONLY=")]
    m = re.fullmatch(r"ONLY=\^\((.*)\)\$", line)
    assert m, line
    return set(m.group(1).split("|"))


def test_the_jobs_window_selects_what_was_launched_in_it(tmp_path, root):
    r = _run(
        tmp_path,
        root,
        "2026-09-29T10:15:00|2026-09-29T12:00:00\\n2026-09-29T10:20:00|2026-09-29T11:30:00\\n",
        REDO_LAUNCHED_BY="7268624+7268693",
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert _only(r) == {
        "tiled_high_gate1",
        "tiled_low_s128",  # its -r2 resumption fell in the window
        "valis_high_micro2_segstardist",  # a cross, logged in its base's launch dir
    }
    assert "ARMS_REPLACE=1" in r.stdout
    assert "Redo window: 2026-09-29 10:15:00 .. 2026-09-29 12:00:00" in r.stdout


def test_a_job_still_running_keeps_the_window_open(tmp_path, root):
    r = _run(
        tmp_path, root, "2026-09-29T10:15:00|Unknown\\n", REDO_LAUNCHED_BY="7268693"
    )
    assert "valis_low_micro0" in _only(r), "launched at 13:30, while the job still ran"


def test_redo_since_needs_no_sacct(tmp_path, root):
    r = _run(
        tmp_path,
        root,
        REDO_SINCE="2026-09-29 10:35:00",
        REDO_UNTIL="2026-09-29 11:30:00",
    )
    assert r.returncode == 0, r.stderr
    assert _only(r) == {"valis_high_micro2_segstardist", "tiled_low_s128"}


def test_no_accounting_is_an_error_naming_the_manual_route(tmp_path, root):
    r = _run(tmp_path, root, "", REDO_LAUNCHED_BY="1")
    assert r.returncode == 1 and "REDO_SINCE" in r.stderr


def test_an_empty_window_redoes_nothing(tmp_path, root):
    r = _run(tmp_path, root, REDO_SINCE="2030-01-01 00:00:00")
    assert r.returncode == 0 and "nothing to redo" in r.stderr
    assert "ONLY=" not in r.stdout


def test_redo_and_only_are_exclusive(tmp_path, root):
    r = _run(tmp_path, root, REDO_SINCE="2026-09-29 10:00:00", ONLY="^x$")
    assert r.returncode == 1 and "set one" in r.stderr
