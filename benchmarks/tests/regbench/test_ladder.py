"""The memory ladder's rule: a unit goes up a rung only when it ran out of memory.

Every rung of a run is submitted up front, so the task of a later rung decides when it
starts (``run --retry-oom``, ``selfcheck --retry-memory``). These tests pin that
decision, and that a task which must not re-run really leaves the results alone.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import cli, ladder
from benchmarks.regbench.cases import Case, result_dir, write_json

REPO = Path(__file__).resolve().parents[3]


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


def test_a_unit_without_a_record_does_not_go_up_but_is_reported_lost(tmp_path):
    """More memory is not what it lacked. It is not 'never started, nothing to do'
    either: the climb task says it was lost, and the caller writes that down."""
    paths = [tmp_path / "nothing" / "run.json"]
    go, why = ladder.unit_retry(paths)
    assert not go and "lost to the scheduler" in why
    assert ladder.unit_decision(paths, ladder.CLIMB, 2) == (False, why, True)


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
    # lost to the scheduler: run again, never read as not_checked (review item 12)
    for state in ("CANCELLED by 1", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"):
        assert ladder.selfcheck_retry(missing, state)[0], state


def test_a_selfcheck_record_of_another_submission_is_no_record(tmp_path):
    """Review item 13: the new first attempt died, the old `ok` is still on disk."""
    path = tmp_path / "selfcheck.json"
    path.write_text(json.dumps({"status": "ok", "pairs": [], "submission": "old"}))
    go, why = ladder.selfcheck_retry(path, "", "new")
    assert go and "of this submission" in why
    assert not ladder.selfcheck_retry(path, "", "old")[0]
    assert not ladder.selfcheck_retry(path, "")[0]  # outside the launcher: as before


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
    assert ladder.RUNGS_GB == (16, 64, 256, 512, 700)
    assert ladder.SELFCHECK_RUNGS_GB == (64, 256, 512)
    assert ladder.CPUS == 4 and ladder.MAX_HOURS == 168


# ── infrastructure is not the method (review item 12) ─────────────────────────
@pytest.mark.parametrize(
    "state",
    ["NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "CANCELLED by 1234", "CANCELLED+"],
)
def test_a_task_the_scheduler_lost_is_infrastructure_and_does_not_go_up(
    tmp_path, state
):
    path = record(tmp_path / "v" / "run.json", cause="killed")
    go, why = ladder.unit_retry([path], state)
    rec = json.loads(path.read_text())
    assert (
        not go
        and rec["cause"] == "infrastructure"
        and state.upper() in rec["cause_from"]
    )
    assert "infrastructure" in why


@pytest.mark.parametrize("state", ["FAILED", "COMPLETED", "", "SOMETHING_NEW"])
def test_a_process_that_died_on_its_own_stays_the_methods(tmp_path, state):
    path = record(tmp_path / "v" / "run.json", cause="killed")
    ladder.unit_retry([path], state)
    assert json.loads(path.read_text())["cause"] == "killed"


def test_a_lost_attempt_is_tried_again_at_the_same_rung_once(tmp_path):
    path = record(tmp_path / "v" / "run.json", cause="killed", attempt=2)
    go, why, _ = ladder.unit_decision([path], ladder.AGAIN, 2, "NODE_FAIL")
    assert go and "the same rung again" in why
    # the retry was lost too: the bound is one, and the cause stays infrastructure
    record(path, cause="killed", attempt=2, retry=1)
    go, why, _ = ladder.unit_decision([path], ladder.AGAIN, 2, "PREEMPTED")
    assert not go
    assert json.loads(path.read_text())["cause"] == "infrastructure"
    # and the rung above does not take it for a memory failure
    assert not ladder.unit_decision([path], ladder.CLIMB, 3)[0]


@pytest.mark.parametrize("cause", ["oom", "exception", "timeout", "writer", "killed"])
def test_again_runs_nothing_the_scheduler_did_not_lose(tmp_path, cause):
    path = record(tmp_path / "v" / "run.json", cause=cause, attempt=1)
    assert not ladder.unit_decision([path], ladder.AGAIN, 1, "COMPLETED")[0]
    ok = record(tmp_path / "w" / "run.json", ok=True)
    assert not ladder.unit_decision([ok], ladder.AGAIN, 1)[0]


def test_a_unit_that_never_started_is_run_by_the_again_task(tmp_path):
    missing = [tmp_path / "none" / "run.json"]
    go, why, _ = ladder.unit_decision(missing, ladder.AGAIN, 1)
    assert go and "never started" in why
    # a higher rung's `again` has nothing to run for a unit that never ran at all
    assert not ladder.unit_decision(missing, ladder.AGAIN, 2)[0]


def test_a_climb_task_lost_before_its_placeholder_is_run_by_again(tmp_path):
    """Rung 1 said oom, the rung-2 task died before the driver wrote anything: the
    record is still rung 1's, and rung 2's `again` runs the unit at rung 2."""
    path = record(tmp_path / "v" / "run.json", cause="oom", attempt=1)
    assert ladder.unit_decision([path], ladder.AGAIN, 2)[0]
    assert not ladder.unit_decision([path], ladder.AGAIN, 1)[0]
    record(path, cause="oom", attempt=2)  # the rung-2 attempt did run
    assert not ladder.unit_decision([path], ladder.AGAIN, 2)[0]


@pytest.mark.parametrize("cause", ["oom", "writer_oom", "benchmark_oom"])
def test_every_memory_cause_goes_up(tmp_path, cause):
    path = record(tmp_path / "v" / "run.json", cause=cause)
    go, why = ladder.unit_retry([path])
    assert go and cause in why


def test_a_record_is_settled_by_the_task_that_wrote_it(tmp_path):
    """The record's own launch decides, not whichever task ran just before: an attempt
    the cgroup killed, followed by a retry task the node lost, is still `oom`."""
    path = record(tmp_path / "v" / "run.json", cause="killed", launch_id="900_3")
    acct = ladder.parse_accounting(
        "900_3|OUT_OF_MEMORY|4711|03:25:10\n901_3|NODE_FAIL|12|00:00:01\n"
    )
    go, _ = ladder.unit_retry([path], "NODE_FAIL", acct)
    rec = json.loads(path.read_text())
    assert go and rec["cause"] == "oom"
    assert rec["scheduler"] == {
        "state": "OUT_OF_MEMORY",
        "elapsed_s": 4711.0,
        "cpu_s": 3 * 3600 + 25 * 60 + 10.0,
    }


# ── one look at the accounting is not the answer ──────────────────────────────
@pytest.mark.parametrize("state", ["RUNNING", "PENDING", "COMPLETING", ""])
def test_a_task_that_has_not_ended_settles_nothing(tmp_path, state):
    """The accounting can lag behind a task that has just ended: a state that is not
    an end state is no evidence, and the record is left exactly as it was."""
    path = record(tmp_path / "v" / "run.json", cause="killed", launch_id="900_3")
    before = path.read_text()
    acct = ladder.parse_accounting(f"900_3|{state}|10|00:00:09\n")
    assert ladder.settle(path, "", acct) == "killed"
    assert path.read_text() == before
    assert ladder.unsettled([path], "", acct) == [(path, "900_3")]
    assert not ladder.ended(state)


def test_a_record_with_an_end_state_or_another_cause_is_not_waited_for(tmp_path):
    died = record(tmp_path / "a" / "run.json", cause="killed", launch_id="900_3")
    other = record(tmp_path / "b" / "run.json", cause="exception", launch_id="900_4")
    fine = record(tmp_path / "c" / "run.json", ok=True, launch_id="900_5")
    stage = record(
        tmp_path / "d" / "run.json", cause="killed", launch_id="900_6", no_rerun=True
    )
    acct = ladder.parse_accounting("900_3|FAILED|10|00:00:09\n")
    assert ladder.unsettled([died, other, fine, stage], "", acct) == []
    assert ladder.unsettled([died, other, fine, stage]) == [(died, "900_3")]
    assert all(ladder.ended(s) for s in ("FAILED", "COMPLETED", "OUT_OF_ME+"))


def test_killed_without_an_end_state_is_not_final(tmp_path):
    """After the wait the climb task runs such a unit again at its own rung, and the
    record says why. Before the wait nothing is decided, and the `again` task (the
    same memory once more) never takes it: only the scheduler's word makes a loss."""
    path = record(tmp_path / "v" / "run.json", cause="killed", launch_id="900_3")
    assert not ladder.unit_decision([path], ladder.CLIMB, 2)[0]
    assert not ladder.unit_decision([path], ladder.AGAIN, 1, waited=True)[0]
    assert "cause_from" not in json.loads(path.read_text())
    go, why, lost = ladder.unit_decision(
        [path], ladder.CLIMB, 2, waited=True, mem_gb=64.0
    )
    assert go and not lost
    assert "no end state" in why and "this rung's memory" in why
    rec = json.loads(path.read_text())
    assert rec["cause"] == "killed"  # never known, so never renamed
    assert ladder.NO_END_STATE in rec["cause_from"] and "64 GB" in rec["cause_from"]
    assert rec["scheduler"]["state"] == ""


@pytest.mark.parametrize("state", ["FAILED", "COMPLETED"])
def test_killed_with_an_end_state_that_names_no_cause_stays_final(tmp_path, state):
    """A process that died on its own under a task the scheduler saw end: the
    method's, however long anybody waited."""
    path = record(tmp_path / "v" / "run.json", cause="killed", launch_id="900_3")
    acct = ladder.parse_accounting(f"900_3|{state}|10|00:00:09\n")
    go, why, _ = ladder.unit_decision([path], ladder.CLIMB, 2, "", acct, waited=True)
    assert not go and "killed" in why
    assert json.loads(path.read_text())["cause"] == "killed"


def test_memory_is_the_only_known_cause_that_climbs(tmp_path):
    for cause in ("timeout", "exception", "gpu_oom", "writer", "infrastructure"):
        path = record(tmp_path / cause / "run.json", cause=cause, launch_id="1_0")
        assert not ladder.unit_retry([path], waited=True)[0], cause
        if cause == "infrastructure":
            continue  # run once more by the next rung's task: a retry, not a climb
        assert not ladder.unit_decision([path], ladder.CLIMB, 2, waited=True)[0], cause


@pytest.mark.parametrize(
    "state, cause",
    [
        ("", "oom"),  # the accounting never answered
        ("COMPLETED", "oom"),  # the script exits 0 whatever became of Python
        ("FAILED", "oom"),
        ("OUT_OF_MEMORY", "oom"),
        ("NODE_FAIL", "infrastructure"),  # the scheduler names a cause: it stands
        ("TIMEOUT", "timeout"),
    ],
)
def test_python_ended_by_signal_9_under_a_script_that_went_on_is_memory(
    tmp_path, state, cause
):
    out = tmp_path / "out"
    notes = ladder.exits_dir(out)
    notes.mkdir(parents=True)
    (notes / "900_3").write_text("137\n")
    (notes / "900_4").write_text("1\n")
    (notes / "junk").write_text("not a number\n")
    assert ladder.read_exits(out) == {"900_3": 137, "900_4": 1}
    assert ladder.read_exits(tmp_path / "none") == {}
    path = record(out / "v" / "d" / "c" / "run.json", cause="killed", launch_id="900_3")
    acct = ladder.parse_accounting(f"900_3|{state}|10|00:00:09\n") if state else {}
    exits = ladder.read_exits(out)
    assert ladder.unsettled([path], "", acct, exits=exits) == []
    go, _ = ladder.unit_retry([path], "", acct, exits=exits)
    rec = json.loads(path.read_text())
    assert rec["cause"] == cause and go is (cause == "oom")
    assert ("signal 9" in rec["cause_from"]) is (state in ("", "COMPLETED", "FAILED"))
    # any other exit code is evidence of nothing
    other = record(
        out / "v" / "d" / "e" / "run.json", cause="killed", launch_id="900_4"
    )
    assert ladder.settle(other, "", {}, exits=exits) == "killed"


def test_the_settle_job_reads_the_scripts_notes_too(tmp_path):
    path = record(
        tmp_path / "m" / "d" / "c1" / "run.json", cause="killed", launch_id="5_0"
    )
    notes = ladder.exits_dir(tmp_path)
    notes.mkdir(parents=True)
    (notes / "5_0").write_text("137\n")
    acct = tmp_path / "acct.psv"
    acct.write_text("5_0|COMPLETED|30|00:00:20\n")
    assert cli.main(["settle", "--out", str(tmp_path), "--accounting", str(acct)]) == 0
    rec = json.loads(path.read_text())
    assert rec["cause"] == "oom" and rec["scheduler"]["elapsed_s"] == 30.0


def test_whichever_job_wrote_the_placeholder_is_settled_by_one_rule(tmp_path):
    """The last job of a round whose registration ran in other jobs is settled as any
    other unit's task: memory is `oom`, the time limit `timeout`, an unexplained death
    `killed`, all three the method's."""
    from benchmarks.regbench import failure_log

    def settled(state, **more):
        path = tmp_path / f"{state}{len(more)}" / "run.json"
        rec = {"cause": "killed", "launch_id": "7_0", **more}
        record(path, **rec)
        acct = {"7_0": {"state": state, "elapsed_s": 5.0, "cpu_s": 1.0}}
        return ladder.settle(path, "", acct)

    for more in ({}, {"collector": True}, {"phase": "writer"}):
        got = [settled(s, **more) for s in ("OUT_OF_MEMORY", "TIMEOUT", "FAILED")]
        assert got == ["oom", "timeout", "killed"], more
        assert all(failure_log.classify(c) == failure_log.METHOD for c in got)
        assert settled("NODE_FAIL", **more) == "infrastructure"


def test_a_settled_or_lost_record_is_replaced_whole(tmp_path, monkeypatch):
    """Both rewrites go through the atomic writer: a task killed in the middle of one
    leaves the record as it was, never half of a new one."""
    from benchmarks.regbench import cases as cases_mod

    path = record(tmp_path / "v" / "run.json", cause="killed", launch_id="900_3")
    before = path.read_text()

    def dies(tmp, *a, **k):
        Path(tmp).write_text("{")  # half a record, in the temporary file
        raise KeyboardInterrupt

    monkeypatch.setattr(
        cases_mod, "write_json", lambda p, obj: cases_mod._atomic(p, dies)
    )
    with pytest.raises(KeyboardInterrupt):
        ladder.settle(path, "OUT_OF_MEMORY")
    assert path.read_text() == before
    assert [p.name for p in path.parent.iterdir()] == ["run.json"]
    missing = tmp_path / "w" / "run.json"
    with pytest.raises(KeyboardInterrupt):
        ladder.mark_lost({missing: {"variant": "w"}}, 1, "s1")
    assert not missing.exists() and list(missing.parent.iterdir()) == []


# ── the rung task's script: the wait, and the note ────────────────────────────
PYTHON = r"""#!/bin/bash
# python: records its arguments; with --accounting-wanted and no --accounting-waited it
# names a task and exits 75; else it exits $FAKE/exit (default 0)
echo "$*" >> "$FAKE/python.log"
wanted="" waited=0
while [ $# -gt 0 ]; do
    case "$1" in
        --accounting-wanted) wanted="$2"; shift 2 ;;
        --accounting-waited) waited=1; shift ;;
        *) shift ;;
    esac
done
if [ -n "$wanted" ] && [ "$waited" = 0 ] && [ ! -e "$FAKE/settled" ]; then
    echo "901_3" > "$wanted"
    exit 75
fi
exit "$(cat "$FAKE/exit" 2>/dev/null || echo 0)"
"""
SACCT = r"""#!/bin/bash
# sacct: RUNNING for the first $FAKE/lag calls, then $FAKE/state
n=$(( $(cat "$FAKE/sacct.n" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$FAKE/sacct.n"
echo "$*" >> "$FAKE/sacct.log"
if [ "$n" -le "$(cat "$FAKE/lag" 2>/dev/null || echo 0)" ]; then
    echo "901_3|RUNNING|5|00:00:04"
else
    echo "901_3|$(cat "$FAKE/state")|5|00:00:04"
fi
"""


def _rung(tmp_path, sacct=True, task="climb", out="out", sacct_text=SACCT, **more):
    """run_rung.sbatch as the scheduler runs task 3 of array job 902, with a stand-in
    python (and sacct) on PATH."""
    fake, out = tmp_path / "bin", tmp_path / out
    fake.mkdir(exist_ok=True)
    for name, text in (("python", PYTHON), ("sacct", sacct_text)):
        if name == "sacct" and not sacct:
            continue
        (fake / name).write_text(text)
        (fake / name).chmod(0o755)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("REGBENCH_", "SLURM_"))
    }
    env.update(
        PATH=f"{fake}:/usr/bin:/bin",
        FAKE=str(fake),
        REGBENCH_REPO=str(REPO),
        REGBENCH_CASES=str(tmp_path / "cases"),
        REGBENCH_OUT=str(out),
        REGBENCH_ACCT_POLL_S="1",
        REGBENCH_ACCT_WAIT_S="3",
        SLURM_ARRAY_JOB_ID="902",
        SLURM_ARRAY_TASK_ID="3",
        SLURM_JOB_ID="950",
        SLURM_JOB_NAME="rb-initial-multiplex-64g",
    )
    env.update({k: str(v) for k, v in more.items()})
    done = subprocess.run(
        ["bash", str(REPO / "benchmarks/regbench/slurm/run_rung.sbatch"), "initial",
         "initial", "-", "-", "INITIAL", "0", "multiplex", "2", "64", "0", "900,901",
         task, "1-00:00:00"],
        env=env, capture_output=True, text=True, cwd=REPO, timeout=120,
    )  # fmt: skip
    calls = (fake / "python.log").read_text().splitlines()
    return done, calls, fake, out


def test_the_rung_task_asks_the_accounting_until_it_names_an_end_state(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "lag").write_text("2")  # the first look and one more say RUNNING
    (fake / "state").write_text("OUT_OF_MEMORY")
    done, calls, fake, out = _rung(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    acct = out / "logs" / "accounting" / "initial-multiplex-3-2-0.psv"
    # the first call: one look, and where to name the tasks it wants an end state of
    assert (
        f"--ladder climb --accounting {acct} --accounting-wanted {acct}.wanted"
        in (calls[0])
    )
    assert "--accounting-waited" not in calls[0]
    # the second, once the accounting has the end state: nothing run twice in between
    assert len(calls) == 2
    assert calls[1].endswith(f"--ladder climb --accounting-waited --accounting {acct}")
    assert acct.read_text() == "901_3|OUT_OF_MEMORY|5|00:00:04\n"
    asked = (fake / "sacct.log").read_text().splitlines()
    assert len(asked) == 3 and "-j 900_3,901_3" in asked[0] and "-j 901_3 " in asked[2]
    assert "accounting: an end state for 901_3 after 2s" in done.stdout
    assert done.stdout.strip().splitlines()[-1].startswith("REGBENCH_END kind=run")


def test_the_wait_for_the_accounting_is_bounded_and_said(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "lag").write_text("1000")
    done, calls, fake, out = _rung(tmp_path, REGBENCH_ACCT_WAIT_S=2)
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(calls) == 2 and "--accounting-waited" in calls[1]
    assert "accounting: NO end state for every task of 901_3 after 2s" in done.stdout
    assert "REGBENCH_ACCT_WAIT_S" in done.stdout
    assert len((fake / "sacct.log").read_text().splitlines()) == 3


def test_a_node_without_accounting_does_not_wait(tmp_path):
    done, calls, _, _ = _rung(tmp_path, sacct=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(calls) == 1 and calls[0].endswith("--ladder climb --accounting-waited")
    # the first attempt of a ladder asks nothing at all
    (tmp_path / "bin" / "python.log").unlink()
    done, calls, _, _ = _rung(tmp_path, task="first")
    assert len(calls) == 1 and not [w for w in calls[0].split() if "ladder" in w]
    assert not [w for w in calls[0].split() if w.startswith("--accounting")]


def test_the_script_notes_a_python_that_was_killed_and_only_that(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "settled").write_text("")
    (fake / "state").write_text("COMPLETED")
    (fake / "exit").write_text("137")
    done, _, fake, out = _rung(tmp_path)
    assert done.returncode == 0 and "python exited 137" in done.stdout
    assert ladder.read_exits(out) == {"902_3": ladder.EXIT_SIGKILL}
    (fake / "exit").write_text("1")
    done, _, _, out2 = _rung(tmp_path, out="out2")
    assert "python exited 1" in done.stdout
    assert ladder.read_exits(out2) == {} and not ladder.exits_dir(out2).exists()


def test_the_script_and_the_ladder_agree_on_what_has_not_ended():
    code = (REPO / "benchmarks/regbench/slurm/run_rung.sbatch").read_text()
    assert f"/^({'|'.join(ladder.STATES_NOT_ENDED)})/" in code
    assert f'[ "$code" = {ladder.EXIT_ACCOUNTING_PENDING} ]' in code
    assert f'[ "$code" = {ladder.EXIT_SIGKILL} ]' in code


def test_the_climb_task_waits_and_then_runs_a_unit_nobody_explained(
    tmp_path, monkeypatch
):
    """End to end with the identity method: the first call names the task it wants an
    end state of and decides nothing; the call after the wait runs the unit at this
    rung, and the result carries the attempt it replaces."""
    root, out = tmp_path / "cases", tmp_path / "out"
    case = a_case(root, "P1_r1", "P1")
    path = result_dir(out, "initial", case) / "run.json"
    monkeypatch.setenv("REGBENCH_SUBMISSION", "s1")
    monkeypatch.setenv("REGBENCH_ATTEMPT", "2")
    monkeypatch.setenv("REGBENCH_MEM_GB", "64")
    record(path, cause="killed", launch_id="77_0", submissions=["s1"])
    before = path.read_text()
    wanted = tmp_path / "w"
    code = _run(root, out, "--ladder", "climb", "--accounting-wanted", str(wanted))
    assert code == ladder.EXIT_ACCOUNTING_PENDING == 75
    assert wanted.read_text() == "77_0\n" and path.read_text() == before
    code = _run(
        root, out, "--ladder", "climb", "--accounting-wanted", str(wanted),
        "--accounting-waited",
    )  # fmt: skip
    assert code == 0
    rec = json.loads(path.read_text())
    assert rec["ok"] and rec["attempt"] == 2 and rec["mem_requested_gb"] == 64
    (old,) = rec["previous_attempts"]
    assert old["cause"] == "killed" and ladder.NO_END_STATE in old["cause_from"]
    assert "64 GB" in old["cause_from"]


def test_the_accounting_is_parsed_by_field_and_tolerates_junk():
    text = "12_0|COMPLETED|10|00:00:09.500\n\nnot a line\n13_1|CANCELLED by 7|5|1-00:00:01\n|x|\n"
    got = ladder.parse_accounting(text)
    assert got["12_0"] == {"state": "COMPLETED", "elapsed_s": 10.0, "cpu_s": 9.5}
    assert got["13_1"]["state"] == "CANCELLED BY 7"
    assert got["13_1"]["cpu_s"] == 86401.0 and set(got) == {"12_0", "13_1"}
    assert ladder.read_accounting(None) == {} and ladder.read_accounting("/nope") == {}


# sacct -n -P -o JobID,State,ElapsedRaw,TotalCPU for two array tasks and a plain job,
# in the two shapes the manual describes: with each job's steps, and with -X (the
# allocation alone, whose utilisation is reported as zero)
WITH_STEPS = """\
900_3|COMPLETED|4711|03:25:10
900_3.batch|COMPLETED|4711|03:25:09.500
900_3.extern|COMPLETED|4712|00:00:00.500
900_4|OUT_OF_MEMORY|120|00:00:00
900_4.batch|OUT_OF_MEMORY|120|00:07:30
900_4.extern|COMPLETED|121|00:00:00
900_4.0|CANCELLED|60|1-00:00:01
901|TIMEOUT|600|00:00:00
901.batch|CANCELLED|601|00:00:00
902.batch|COMPLETED|5|00:00:04
"""
ALLOCATIONS_ONLY = """\
900_3|COMPLETED|4711|00:00:00
900_4|OUT_OF_MEMORY|120|00:00:00
901|TIMEOUT|600|00:00:00
"""


def test_a_jobs_cpu_time_is_the_sum_of_its_steps():
    got = ladder.parse_accounting(WITH_STEPS)
    # one entry per job or array task: a step is never a job of its own, and steps of
    # a job that has no line are no job either
    assert set(got) == {"900_3", "900_4", "901"}
    # the job's own line already carries the total: it is not added to its steps'
    assert got["900_3"] == {
        "state": "COMPLETED",
        "elapsed_s": 4711.0,
        "cpu_s": 3 * 3600 + 25 * 60 + 10.0,
    }
    # a job's line with no CPU time: its steps have it. State and elapsed time stay
    # the job's own, whatever its steps ended as
    assert got["900_4"] == {
        "state": "OUT_OF_MEMORY",
        "elapsed_s": 120.0,
        "cpu_s": 7 * 60 + 30 + 86401.0,
    }
    # zero everywhere is "nothing was reported", never a figure
    assert got["901"] == {"state": "TIMEOUT", "elapsed_s": 600.0, "cpu_s": None}


def test_allocations_alone_name_the_end_state_and_no_cpu_time():
    got = ladder.parse_accounting(ALLOCATIONS_ONLY)
    assert {k: v["state"] for k, v in got.items()} == {
        "900_3": "COMPLETED",
        "900_4": "OUT_OF_MEMORY",
        "901": "TIMEOUT",
    }
    assert [v["elapsed_s"] for v in got.values()] == [4711.0, 120.0, 600.0]
    assert [v["cpu_s"] for v in got.values()] == [None, None, None]


@pytest.mark.parametrize("script", ["score_cohort.sbatch", "run_rung.sbatch"])
def test_the_accounting_is_asked_with_the_steps(script):
    """Both queries feed a CPU figure (`cpu_h_scheduler`, and a lost attempt's
    `scheduler_cpu_s`): with -X it could only ever be zero."""
    text = (Path(ladder.__file__).parent / "slurm" / script).read_text()
    calls = [
        line.split("#")[0]
        for line in text.splitlines()
        if "sacct " in line.split("#")[0] and "ElapsedRaw,TotalCPU" in line
    ]
    assert len(calls) == 1
    assert " -X" not in calls[0] and "--allocations" not in calls[0], calls[0]


SACCT_STEPS = r"""#!/bin/bash
# sacct as the manual describes it: with -X the allocation alone and no CPU time; else
# the job's line, with no CPU time of its own here, and one line per step
echo "$*" >> "$FAKE/sacct.log"
case " $* " in
    *" -X "*) echo "901_3|NODE_FAIL|120|00:00:00" ;;
    *)
        echo "901_3|NODE_FAIL|120|00:00:00"
        echo "901_3.batch|CANCELLED|120|00:07:30"
        echo "901_3.extern|COMPLETED|121|00:00:01"
        ;;
esac
"""


def test_the_rung_task_keeps_the_cpu_time_of_a_task_the_scheduler_ended(tmp_path):
    """The end state is the job's own line's, whatever its steps ended as, and the CPU
    time is its steps': 7 min 30 s + 1 s."""
    fake = tmp_path / "bin"
    fake.mkdir()
    done, calls, fake, out = _rung(tmp_path, sacct_text=SACCT_STEPS)
    assert done.returncode == 0, done.stdout + done.stderr
    acct = out / "logs" / "accounting" / "initial-multiplex-3-2-0.psv"
    # the step lines are no task Python asked about: the wait ends on the job's line
    assert len(calls) == 2 and len((fake / "sacct.log").read_text().splitlines()) == 1
    assert "accounting: an end state for 901_3 after 0s" in done.stdout
    assert ladder.read_accounting(acct) == {
        "901_3": {"state": "NODE_FAIL", "elapsed_s": 120.0, "cpu_s": 451.0}
    }


def test_a_record_of_another_submission_is_evidence_of_nothing(tmp_path):
    """Review item 13: an old `oom` must not send a unit of a new submission up a rung
    it never tried, and an old `ok` must not pass for this submission's."""
    old = record(tmp_path / "v" / "run.json", cause="oom", submissions=["old"])
    assert ladder.unit_decision([old], ladder.CLIMB, 2, submission="new")[:1] == (
        False,
    )
    assert ladder.unit_decision([old], ladder.CLIMB, 2, submission="new")[2] is True
    assert ladder.unit_decision([old], ladder.AGAIN, 1, submission="new")[0]
    assert ladder.unit_decision([old], ladder.CLIMB, 2, submission="old")[0]
    assert ladder.mine({"submissions": ["a", "b"]}, "b") and ladder.mine({}, "")
    assert not ladder.mine({"submission": "a"}, "b") and not ladder.mine(None, "b")


def test_a_unit_lost_twice_is_written_down_as_infrastructure(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    cases = [a_case(root, "P1_r1", "P1"), a_case(root, "P1_r2", "P1")]
    heads = ladder.unit_heads(out, "initial", None, cases)
    stale = record(list(heads)[0], cause="oom", submissions=["old"], marker=1)
    assert ladder.mark_lost(heads, 1, "new") == 2
    for path, head in heads.items():
        rec = json.loads(path.read_text())
        assert rec["cause"] == "infrastructure" and not rec["ok"]
        assert rec["submissions"] == ["new"] and rec["attempt"] == 1
        assert rec["case_id"] == head["case_id"] and rec["variant"] == "initial"
        assert rec["unit"] == "pair" and "marker" not in rec
    assert stale.exists() and ladder.mark_lost(heads, 1, "new") == 0  # once
    # the record another submission left is set aside, not destroyed
    first = json.loads(list(heads)[0].read_text())
    kept = Path(first["earlier_result"]["kept"])
    assert kept == stale.parent / "earlier" / "1"
    assert json.loads((kept / "run.json").read_text())["marker"] == 1
    assert first["earlier_result"] == {
        "kept": str(kept),
        "ok": False,
        "submissions": ["old"],
    }
    assert "earlier_result" not in json.loads(list(heads)[1].read_text())
    # written down, it is not tried at a higher rung
    assert not ladder.unit_decision(list(heads), ladder.AGAIN, 2, submission="new")[0]
    assert not ladder.unit_decision(list(heads), ladder.CLIMB, 3, submission="new")[0]


def test_an_attempt_the_accounting_names_late_is_retried_by_the_next_rung(tmp_path):
    """Rung 1 is lost to a node failure the accounting does not have yet when rung 1's
    `again` task decides. Rung 2's `climb` task finds it and runs the unit once; lost
    there too, rung 2's own retry runs it; lost a third time, it stays lost."""
    path = record(
        tmp_path / "v" / "run.json",
        cause="killed", launch_id="900_3", submissions=["s1"], attempt=1,
    )  # fmt: skip
    running = ladder.parse_accounting("900_3|RUNNING|10|00:00:09\n")
    go, why, lost = ladder.unit_decision(
        [path], ladder.AGAIN, 1, "", running, "s1", waited=True
    )
    assert (go, lost) == (False, False) and "not lost to the scheduler" in why
    assert json.loads(path.read_text())["cause"] == "killed"
    ended = ladder.parse_accounting("900_3|NODE_FAIL|10|00:00:09\n")
    go, why, lost = ladder.unit_decision(
        [path], ladder.CLIMB, 2, "", ended, "s1", waited=True, mem_gb=64
    )
    assert (go, lost) == (True, False) and ladder.LATE_LOST in why
    rec = json.loads(path.read_text())
    # the record says what it is: still the scheduler's loss, and no climb for memory
    assert rec["cause"] == "infrastructure" and rec["attempt"] == 1
    assert "NODE_FAIL" in rec["cause_from"] and "at 64 GB" in rec["cause_from"]
    assert rec["cause_from"].endswith(ladder.LATE_LOST)
    assert rec["cause"] not in ladder.resources.MEMORY_CAUSES
    # the run at rung 2, lost as well: rung 2's own retry, then nothing more
    record(path, cause="infrastructure", submissions=["s1"], attempt=2, retry=0)
    assert ladder.unit_decision([path], ladder.AGAIN, 2, submission="s1")[0]
    record(path, cause="infrastructure", submissions=["s1"], attempt=2, retry=1)
    assert not ladder.unit_decision([path], ladder.CLIMB, 3, submission="s1")[0]
    # an attempt already tried again at its own rung is not tried a third time
    record(path, cause="infrastructure", submissions=["s1"], attempt=1, retry=1)
    go, why, _ = ladder.unit_decision([path], ladder.CLIMB, 2, submission="s1")
    assert not go and "failed for another reason (infrastructure)" in why
    # and a failure that is the method's is never run again by this rule
    for cause in ("exception", "timeout", "writer"):
        record(path, cause=cause, submissions=["s1"], attempt=1, retry=0)
        assert not ladder.unit_decision([path], ladder.CLIMB, 2, submission="s1")[0]


def test_a_slide_that_is_not_there_is_not_the_methods_failure(tmp_path, capsys):
    """A fluorescence case whose slide cannot be read from here (its folder is not
    bound into the image): refused before the method starts, as `infrastructure`."""
    root, out = tmp_path / "cases", tmp_path / "out"
    case = a_case(root, "P1_r1", "P1")
    record_path = Path(case.dir) / "case.json"
    stored = json.loads(record_path.read_text())
    stored["mov_image"] = "/not/bound/slides/P1_r2.ome.tif"
    record_path.write_text(json.dumps(stored))
    _run(root, out)
    run = json.loads((result_dir(out, "initial", case) / "run.json").read_text())
    assert run["ok"] is False and run["cause"] == "infrastructure"
    assert "/not/bound/slides/P1_r2.ome.tif cannot be read from where" in run["error"]
    assert "REGBENCH_BIND" in run["error"]
    # a stage job of a registration in several jobs says the same before it starts
    with pytest.raises(SystemExit, match="stage coarse: multiplex/P1_r1: /not/bound"):
        cli.main(
            ["stage", "--cases", str(root), "--out", str(out), "--dataset", "multiplex",
             "--method", "stare", "--stage", "coarse", "--index", "0"]
        )  # fmt: skip


def test_the_last_rung_is_settled_from_the_accounting(tmp_path):
    """The last rung has no later task: `settle` gives every result that still says
    killed the scheduler's cause and its elapsed time."""
    a = record(
        tmp_path / "m" / "d" / "c1" / "run.json", cause="killed", launch_id="5_0"
    )
    b = record(
        tmp_path / "m" / "d" / "c2" / "run.json", cause="killed", launch_id="5_1"
    )
    c = record(
        tmp_path / "m" / "d" / "c3" / "run.json", cause="exception", launch_id="5_2"
    )
    d = record(
        tmp_path / "m" / "d" / "c4" / "run.json", cause="killed", launch_id="5_3"
    )
    acct = tmp_path / "acct.psv"
    acct.write_text(
        "5_0|NODE_FAIL|30|00:00:20\n5_1|TIMEOUT|86400|4-00:00:00\n5_2|OUT_OF_MEMORY|1|1\n5_3|FAILED|7|00:00:03\n"
    )
    assert cli.main(["settle", "--out", str(tmp_path), "--accounting", str(acct)]) == 0
    assert json.loads(a.read_text())["cause"] == "infrastructure"
    assert json.loads(b.read_text())["cause"] == "timeout"
    assert json.loads(b.read_text())["scheduler"]["elapsed_s"] == 86400.0
    assert json.loads(c.read_text())["cause"] == "exception"
    rec = json.loads(d.read_text())  # died on its own: still killed, cost now known
    assert rec["cause"] == "killed" and rec["scheduler"]["elapsed_s"] == 7.0


def test_a_lost_unit_keeps_the_valid_result_of_an_earlier_submission(tmp_path):
    """An `ok` result of submission A, then submission B loses the unit: A's record and
    its warped points are still there, and B's row says `infrastructure`."""
    root, out = tmp_path / "cases", tmp_path / "out"
    case = a_case(root, "P1_r1", "P1")
    heads = ladder.unit_heads(out, "initial", None, [case])
    (path,) = heads
    path.parent.mkdir(parents=True)
    good = {"ok": True, "cause": "", "submissions": ["A"], "register_s": 7.0}
    path.write_text(json.dumps(good))
    (path.parent / "warped.npz").write_bytes(b"the points A moved")
    (path.parent / "native").mkdir()
    assert ladder.mark_lost(heads, 1, "B") == 1
    now = json.loads(path.read_text())
    assert now["cause"] == "infrastructure" and now["submissions"] == ["B"]
    assert not (path.parent / "warped.npz").exists()  # B has no result of its own
    kept = Path(now["earlier_result"]["kept"])
    assert now["earlier_result"]["ok"] is True and "is kept under" in now["error"]
    assert json.loads((kept / "run.json").read_text()) == good
    assert (kept / "warped.npz").read_bytes() == b"the points A moved"
    assert (path.parent / "native").is_dir()  # the image stays where it is
    # a second loss (submission C) sets B's record aside too, beside A's
    assert ladder.mark_lost(heads, 1, "C") == 1
    assert sorted(p.name for p in (path.parent / "earlier").iterdir()) == ["1", "2"]
    assert json.loads((kept / "run.json").read_text()) == good
    # nothing set aside is taken for a result
    assert ladder.settle_all(out, {}) == {}
    assert [p.parent.name for p in out.glob("*/*/*/run.json")] == ["P1_r1"]


def test_a_unit_is_not_reused_when_it_was_computed_another_way(
    tmp_path, monkeypatch, capsys
):
    """The reuse key: how the unit was executed, by which code, from which bytes."""
    root, out = tmp_path / "cases", tmp_path / "out"
    case = a_case(root, "P1_r1", "P1")
    path = result_dir(out, "initial", case) / "run.json"
    assert _run(root, out) == 0
    first = json.loads(path.read_text())
    key = first["reuse_key"]
    assert key["execution"] == "one process" and len(key["code"]) == 64
    assert set(key) == {"execution", "code", "inputs", "method"}
    capsys.readouterr()
    assert _run(root, out) == 0 and "reused" in capsys.readouterr().out
    # made by other code
    path.write_text(json.dumps({**first, "reuse_key": {**key, "code": "0" * 64}}))
    assert _run(root, out) == 0 and "already ok for this configuration: reused" not in capsys.readouterr().out
    second = json.loads(path.read_text())
    assert second["launch_id"] != first["launch_id"] and second["reuse_key"] == key
    # a statement of who starts stage processes is nothing to a method without
    # stages: a stray one in the caller's shell does not make its results another's
    monkeypatch.setenv("REGBENCH_STAGES", "jobs")
    assert _run(root, out) == 0 and "reused" in capsys.readouterr().out
    assert json.loads(path.read_text())["reuse_key"]["execution"] == "one process"
    # ... and is the execution of a method that has them
    from benchmarks.regbench import methods

    assert methods.reuse_key("stare", [], out, {}, None)["execution"] == "jobs"
    monkeypatch.setenv("REGBENCH_STAGES", "local")
    assert methods.reuse_key("stare", [], out, {}, None)["execution"] == "local"
    monkeypatch.delenv("REGBENCH_STAGES")
    assert methods.reuse_key("initial", [], out, {}, None)["execution"] == "one process"
    # executed in another shape
    path.write_text(json.dumps({**second, "reuse_key": {**key, "execution": "jobs"}}))
    assert _run(root, out) == 0 and "already ok for this configuration: reused" not in capsys.readouterr().out
    # a record of before the key existed
    old = {k: v for k, v in second.items() if k != "reuse_key"}
    path.write_text(json.dumps(old))
    assert _run(root, out) == 0 and "already ok for this configuration: reused" not in capsys.readouterr().out


def test_a_file_fingerprint_is_its_size_time_and_sampled_bytes(tmp_path):
    from benchmarks.regbench import cases as cases_mod

    small, big = tmp_path / "small.bin", tmp_path / "big.bin"
    small.write_bytes(b"abc" * 1000)
    assert cases_mod.file_fingerprint(tmp_path / "none") == {"missing": True}
    one = cases_mod.file_fingerprint(small)
    assert one["size"] == 3000 and len(one["sample_sha256"]) == 64
    limit = cases_mod.SAMPLE_BLOCKS * cases_mod.SAMPLE_BYTES
    data = bytearray(os.urandom(limit + 300_000))
    big.write_bytes(data)
    st = big.stat()
    base = cases_mod.file_fingerprint(big)

    def with_byte(at):
        changed = bytearray(data)
        changed[at] ^= 0xFF
        big.write_bytes(changed)
        os.utime(big, ns=(st.st_atime_ns, st.st_mtime_ns))  # the same time
        cases_mod._FILE_FINGERPRINTS.clear()
        return cases_mod.file_fingerprint(big)

    # the first and the last byte are sampled; so is the start of every block
    assert with_byte(0) != base and with_byte(len(data) - 1) != base
    step = (len(data) - cases_mod.SAMPLE_BYTES) // (cases_mod.SAMPLE_BLOCKS - 1)
    assert with_byte(step * 7 + 10) != base
    # the stated residual risk: the same size and time, a byte between two samples
    assert with_byte(cases_mod.SAMPLE_BYTES + 5) == base
    # a new time is a new fingerprint whatever the bytes
    big.write_bytes(data)
    os.utime(big, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert cases_mod.file_fingerprint(big) != base


# ── the array tasks, end to end with the identity method ──────────────────────
def test_the_again_task_runs_a_lost_unit_at_the_same_rung(tmp_path, monkeypatch):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = a_case(root, "P1_r1", "P1")
    path = result_dir(out, "initial", case) / "run.json"
    monkeypatch.setenv("REGBENCH_SUBMISSION", "s1")
    monkeypatch.setenv("REGBENCH_ATTEMPT", "1")
    monkeypatch.setenv("REGBENCH_MEM_GB", "16")
    record(path, cause="killed", launch_id="77_0", submissions=["s1"], marker=1)
    acct = tmp_path / "a.psv"
    acct.write_text("77_0|NODE_FAIL|40|00:00:30\n")
    monkeypatch.setenv("REGBENCH_RETRY", "1")
    assert _run(root, out, "--ladder", "again", "--accounting", str(acct)) == 0
    rec = json.loads(path.read_text())
    assert rec["ok"] and rec["attempt"] == 1 and rec["retry"] == 1
    assert rec["mem_requested_gb"] == 16 and rec["submissions"] == ["s1"]
    (lost,) = rec["previous_attempts"]
    assert lost["cause"] == "infrastructure" and lost["scheduler_elapsed_s"] == 40.0
    assert lost["wall_s"] is None


def test_the_climb_task_writes_down_a_unit_nobody_ran(tmp_path, monkeypatch, capsys):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = a_case(root, "P1_r1", "P1")
    path = result_dir(out, "initial", case) / "run.json"
    monkeypatch.setenv("REGBENCH_SUBMISSION", "s2")
    monkeypatch.setenv("REGBENCH_ATTEMPT", "2")
    assert _run(root, out, "--ladder", "climb") == 0
    rec = json.loads(path.read_text())
    assert rec["cause"] == "infrastructure" and rec["attempt"] == 1
    assert "recorded as infrastructure" in capsys.readouterr().out


# ── resumability (review item 16) ─────────────────────────────────────────────
def test_a_resubmission_does_not_overwrite_a_unit_that_is_ok(
    tmp_path, monkeypatch, capsys
):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = a_case(root, "P1_r1", "P1")
    path = result_dir(out, "initial", case) / "run.json"
    monkeypatch.setenv("REGBENCH_SUBMISSION", "first")
    assert _run(root, out) == 0
    before = json.loads(path.read_text())
    assert before["ok"] and before["submissions"] == ["first"]
    monkeypatch.setenv("REGBENCH_SUBMISSION", "second")
    assert _run(root, out) == 0
    after = json.loads(path.read_text())
    assert "reused, nothing run" in capsys.readouterr().out
    assert after["launch_id"] == before["launch_id"]  # the same run, not a new one
    assert after["register_s"] == before["register_s"]
    assert after["submissions"] == ["first", "second"]
    assert [r["submission"] for r in after["reused_by"]] == ["second"]
    # forced, by the flag or by the environment: a new launch
    assert _run(root, out, "--force") == 0
    forced = json.loads(path.read_text())
    assert forced["launch_id"] != before["launch_id"] and "reused_by" not in forced
    monkeypatch.setenv("REGBENCH_FORCE", "1")
    assert _run(root, out) == 0
    assert json.loads(path.read_text())["launch_id"] != forced["launch_id"]


def test_a_unit_is_reused_only_for_the_same_configuration(tmp_path, capsys):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = a_case(root, "P1_r1", "P1")
    path = result_dir(out, "initial", case) / "run.json"
    assert _run(root, out, "--opt", "workers=2") == 0
    first = json.loads(path.read_text())["launch_id"]
    capsys.readouterr()
    assert _run(root, out, "--opt", "workers=3") == 0  # other options: run again
    assert "already ok for this configuration: reused" not in capsys.readouterr().out
    assert json.loads(path.read_text())["launch_id"] != first
    record(path, cause="exception", marker=1)  # a failed unit is never reused
    assert _run(root, out, "--opt", "workers=3") == 0
    assert json.loads(path.read_text())["ok"]


def test_the_ladder_ends_at_the_sites_largest_node():
    assert ladder.RUNGS_GB == (16, 64, 256, 512, 700)
    assert ladder.MAX_RUNG_GB == 700 == ladder.RUNGS_GB[-1]
    assert 700 not in ladder.SELFCHECK_RUNGS_GB
