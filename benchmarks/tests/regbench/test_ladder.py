"""The memory ladder's rule: a unit goes up a rung only when it ran out of memory.

Every rung of a run is submitted up front, so the task of a later rung decides when it
starts (``run --retry-oom``, ``selfcheck --retry-memory``). These tests pin that
decision, and that a task which must not re-run really leaves the results alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from benchmarks.regbench import cli, ladder
from benchmarks.regbench.cases import Case, result_dir, write_json


def a_case(root, case_id, group):
    """A prepared case with no images and no nuclei: enough to be listed and run."""
    case = Case(
        dataset="multiplex",
        case_id=case_id,
        group=group,
        ref_nuclear="",
        mov_nuclear="",
        ref_image="",
        mov_image="",
        modality="fluorescence",
        diagonal=622.0,
        pixel_size_um=0.5,
        ref_hw=[440, 440],
        mov_hw=[440, 440],
    )
    d = root / "multiplex" / case_id
    d.mkdir(parents=True)
    np.savez(d / "points.npz", cells_mov_xy=np.zeros((0, 2)))
    case.dir = str(d)
    write_json(d / "case.json", asdict(case))
    return case


def record(path, ok=False, cause="", **more):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ok": ok, "cause": cause, "attempt": 1, **more}))
    return path


# ── a unit ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "cause, goes_up",
    [
        ("oom", True),
        ("timeout", False),
        ("exception", False),
        ("gpu_oom", False),
        ("killed", False),
    ],
)
def test_only_out_of_memory_units_move_up(tmp_path, cause, goes_up):
    path = record(tmp_path / "v" / "run.json", cause=cause)
    go, why = ladder.unit_retry([path])
    assert go is goes_up, why


def test_a_unit_that_succeeded_is_never_re_run(tmp_path):
    path = record(tmp_path / "v" / "run.json", ok=True)
    assert ladder.unit_retry([path]) == (False, "every result is ok")


def test_a_unit_that_never_started_is_not_retried(tmp_path):
    go, why = ladder.unit_retry([tmp_path / "nothing" / "run.json"])
    assert not go and "never started" in why


def test_one_round_out_of_memory_takes_the_whole_patient_up(tmp_path):
    paths = [
        record(tmp_path / "a" / "run.json", ok=True),
        record(tmp_path / "b" / "run.json", cause="oom"),
        record(tmp_path / "c" / "run.json", cause="exception"),
    ]
    go, why = ladder.unit_retry(paths)
    assert go and "1 of 3" in why


@pytest.mark.parametrize(
    "state, cause, goes_up",
    [
        ("OUT_OF_MEMORY", "oom", True),
        ("OUT_OF_ME+", "oom", True),  # sacct truncates the state to its column
        ("TIMEOUT", "timeout", False),
        ("FAILED", "killed", False),
        ("", "killed", False),
    ],
)
def test_the_scheduler_decides_only_where_the_record_still_says_killed(
    tmp_path, state, cause, goes_up
):
    """The driver writes `killed` before it starts; a process the cgroup limit ended
    never replaces it. The scheduler's state for that task then names the cause, and
    the record is rewritten so that the scorer counts the attempt under it."""
    path = record(tmp_path / "v" / "run.json", cause="killed")
    go, _ = ladder.unit_retry([path], state)
    assert go is goes_up
    rec = json.loads(path.read_text())
    assert rec["cause"] == cause
    assert ("cause_from" in rec) is (cause != "killed")


def test_the_scheduler_never_overrules_the_drivers_own_cause(tmp_path):
    path = record(tmp_path / "v" / "run.json", cause="exception")
    go, _ = ladder.unit_retry([path], "OUT_OF_MEMORY")
    assert not go and json.loads(path.read_text())["cause"] == "exception"


# ── `run --retry-oom`: what the array task of a later rung does ───────────────
def _run(root, out, *extra):
    """The array task's command, with the method that maps every point to itself."""
    argv = ["run", "--cases", str(root), "--out", str(out), "--dataset", "multiplex"]
    return cli.main([*argv, "--method", "initial", "--index", "0", *extra])


def test_a_later_rung_exits_at_once_unless_the_unit_ran_out_of_memory(
    tmp_path, monkeypatch
):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = a_case(root, "P1_r1", "P1")
    path = result_dir(out, "initial", case) / "run.json"
    for cause in ("exception", "timeout", "gpu_oom"):
        record(path, cause=cause, marker="left alone")
        assert _run(root, out, "--retry-oom") == 0, cause
        assert json.loads(path.read_text())["marker"] == "left alone", cause
    record(path, ok=True, marker="left alone")
    assert _run(root, out, "--retry-oom") == 0
    assert json.loads(path.read_text())["marker"] == "left alone"

    record(path, cause="oom", marker="replaced")
    monkeypatch.setenv("REGBENCH_ATTEMPT", "2")
    monkeypatch.setenv("REGBENCH_MEM_GB", "64")
    assert _run(root, out, "--retry-oom") == 0
    rec = json.loads(path.read_text())
    assert "marker" not in rec
    assert rec["ok"] and rec["attempt"] == 2 and rec["mem_requested_gb"] == 64
    # the attempt that ran out of memory is kept, so its cost is not lost
    assert [p["cause"] for p in rec["previous_attempts"]] == ["oom"]


def test_without_the_flag_the_first_rung_always_runs(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = a_case(root, "P1_r1", "P1")
    path = record(result_dir(out, "initial", case) / "run.json", cause="exception")
    assert _run(root, out) == 0
    assert json.loads(path.read_text())["ok"]


# ── the selfcheck climbs too ──────────────────────────────────────────────────
def _selfcheck(tmp_path, pairs, status="not_checked"):
    path = tmp_path / "selfcheck.json"
    path.write_text(json.dumps({"status": status, "pairs": pairs}))
    return path


def test_a_selfcheck_that_ended_for_memory_is_run_again(tmp_path):
    oom = [{"variants": {"x": {"status": "not_checked", "cause": "oom"}}}]
    assert ladder.selfcheck_retry(_selfcheck(tmp_path, oom))[0]
    raised = [{"variants": {"x": {"status": "not_checked", "cause": "oom: boom"}}}]
    assert ladder.selfcheck_retry(_selfcheck(tmp_path, raised))[0]
    killed = [{"cause": "the selfcheck process exited -9 (killed: out of memory?)"}]
    assert ladder.selfcheck_retry(_selfcheck(tmp_path, killed))[0]


@pytest.mark.parametrize(
    "pairs, status",
    [
        ([{"variants": {"x": {"status": "ok"}}}], "ok"),
        ([{"variants": {"x": {"status": "inconsistent"}}}], "inconsistent"),
        ([{"variants": {"x": {"status": "crashed", "cause": "exception"}}}], "crashed"),
        ([{"variants": {"x": {"status": "not_checked", "cause": "timeout"}}}], "x"),
        ([{"variants": {"x": {"status": "not_checked", "cause": "gpu_oom"}}}], "x"),
    ],
)
def test_any_other_selfcheck_result_stands(tmp_path, pairs, status):
    go, why = ladder.selfcheck_retry(_selfcheck(tmp_path, pairs, status))
    assert not go, why


def test_a_selfcheck_with_no_record_is_run_again_unless_it_ran_out_of_time(tmp_path):
    missing = tmp_path / "none.json"
    assert ladder.selfcheck_retry(missing)[0]
    assert ladder.selfcheck_retry(missing, "OUT_OF_MEMORY")[0]
    assert not ladder.selfcheck_retry(missing, "TIMEOUT")[0]
    assert not ladder.selfcheck_retry(missing, "CANCELLED by 1")[0]


def test_selfcheck_retry_memory_leaves_a_standing_record_alone(tmp_path, monkeypatch):
    """`selfcheck --retry-memory` on a record that did not end for memory runs nothing
    and exits with that record's status."""
    from benchmarks.regbench import selfcheck

    work = tmp_path / "work"
    work.mkdir()
    (work / "selfcheck.json").write_text(
        json.dumps(
            {
                "status": "inconsistent",
                "variants": {"stare": "inconsistent"},
                "pairs": [{"variants": {"stare": {"status": "inconsistent"}}}],
            }
        )
    )
    monkeypatch.setattr(
        selfcheck, "run", lambda *a, **k: pytest.fail("the selfcheck ran again")
    )
    code = cli.main(
        ["selfcheck", "--method", "stare", "--work", str(work), "--retry-memory"]
    )
    assert code == selfcheck.EXIT_CODE["inconsistent"] == 1


def test_the_rungs_are_the_frozen_ones():
    assert ladder.RUNGS_GB == (16, 64, 256, 512)
    assert ladder.SELFCHECK_RUNGS_GB == (64, 256, 512)
    assert ladder.CPUS == 4 and ladder.MAX_HOURS == 168
