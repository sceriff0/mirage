"""The memory ladder's rule: a unit goes up a rung only when it ran out of memory.

Every rung of a run is submitted up front, so the task of a later rung decides when it
starts (``run --retry-oom``, ``selfcheck --retry-memory``). These tests pin that
decision, and that a task which must not re-run really leaves the results alone.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

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
    assert ladder.RUNGS_GB == (16, 64, 256, 512)
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


def test_the_accounting_is_parsed_by_field_and_tolerates_junk():
    text = "12_0|COMPLETED|10|00:00:09.500\n\nnot a line\n13_1|CANCELLED by 7|5|1-00:00:01\n|x|\n"
    got = ladder.parse_accounting(text)
    assert got["12_0"] == {"state": "COMPLETED", "elapsed_s": 10.0, "cpu_s": 9.5}
    assert got["13_1"]["state"] == "CANCELLED BY 7"
    assert got["13_1"]["cpu_s"] == 86401.0 and set(got) == {"12_0", "13_1"}
    assert ladder.read_accounting(None) == {} and ladder.read_accounting("/nope") == {}


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
    assert _run(root, out) == 0 and "reused" not in capsys.readouterr().out
    second = json.loads(path.read_text())
    assert second["launch_id"] != first["launch_id"] and second["reuse_key"] == key
    # executed in another shape
    monkeypatch.setenv("REGBENCH_STAGES", "jobs")
    assert _run(root, out) == 0 and "reused" not in capsys.readouterr().out
    assert json.loads(path.read_text())["reuse_key"]["execution"] == "jobs"
    # a record of before the key existed
    monkeypatch.delenv("REGBENCH_STAGES")
    old = {k: v for k, v in second.items() if k != "reuse_key"}
    path.write_text(json.dumps(old))
    assert _run(root, out) == 0 and "reused" not in capsys.readouterr().out


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
    assert "reused" not in capsys.readouterr().out
    assert json.loads(path.read_text())["launch_id"] != first
    record(path, cause="exception", marker=1)  # a failed unit is never reused
    assert _run(root, out, "--opt", "workers=3") == 0
    assert json.loads(path.read_text())["ok"]
