"""The chains of a cohort submission read back (benchmarks/regbench/chains.py): a unit
whose ladder could not go on is the LAUNCHER's (``not_submitted``), and a result whose
task the scheduler had not ended when the benchmark was settled is ``infrastructure``,
never the method's. Fleet wave 2, F1 and F2."""

from __future__ import annotations

import json

from benchmarks.regbench import chains, cli, ladder
from benchmarks.tests.regbench.test_ladder import a_case

SUB = "s1"


def _cases(tmp_path):
    root = tmp_path / "cases"
    return root, [a_case(root, "P1_a", "P1"), a_case(root, "P2_a", "P2")]


def _rec(out, case, variant="palom", **fields):
    paths = ladder.unit_paths(out, "palom", "", [case])
    (path,) = [p for p in paths if p.parents[2].name == variant]
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ok": False, "cause": "oom", "attempt": 3, "submission": SUB, **fields}
    path.write_text(json.dumps(rec))
    return path


def _read(path):
    return json.loads(path.read_text())


def test_a_unit_whose_next_rung_was_not_submitted_is_the_launchers(tmp_path):
    root, (c0, c1) = _cases(tmp_path)
    out = tmp_path / "out"
    climbed = _rec(out, c0, mem_requested_gb=256)
    ok = _rec(out, c0, "palom_affine", ok=True, cause="")
    other = _rec(out, c1)  # a unit that is not named
    n = chains.mark(
        out, root, "multiplex", "palom", "", [0], 512, "not_submitted",
        "no node of partition cpu has 512 GB", SUB,
    )  # fmt: skip
    assert n == 1
    rec = _read(climbed)
    assert rec["cause"] == "not_submitted" and rec[chains.LADDER_CUT] == 512.0
    assert rec["cause_from"].startswith("oom at rung 3 (256 GB); the next rung, 512 GB")
    assert "no node of partition cpu has 512 GB" in rec["cause_from"]
    assert _read(ok)["ok"] is True and _read(other)["cause"] == "oom"


def test_only_a_failure_the_next_rung_would_run_again_changes(tmp_path):
    """The method's own failure (`writer`, `no_rerun`) and another submission's record
    keep their cause; a `killed` one is settled first by its script's note, as the
    next rung's task would have."""
    root, (c0, c1) = _cases(tmp_path)
    out = tmp_path / "out"
    writer = _rec(out, c0, cause="writer")
    final = _rec(out, c0, "palom_affine", no_rerun=True)
    theirs = _rec(out, c1, submission="s0")
    killed = _rec(out, c1, "palom_affine", cause="killed", launch_id="50_1")
    exits = out / "logs" / "accounting" / "exits"
    exits.mkdir(parents=True)
    (exits / "50_1").write_text("137\n")
    n = chains.mark(out, root, "multiplex", "palom", "", [0, 1, 7], 700, "not_submitted",
                    "refused", SUB)  # fmt: skip
    assert n == 1
    assert [_read(p)["cause"] for p in (writer, final, theirs)] == [
        "writer",
        "oom",
        "oom",
    ]
    rec = _read(killed)
    assert rec["cause"] == "not_submitted" and rec["cause_from"].startswith("oom at")


def test_the_watcher_writes_it_through_the_cli(tmp_path, monkeypatch, capsys):
    root, (c0, _) = _cases(tmp_path)
    out = tmp_path / "out"
    path = _rec(out, c0)
    monkeypatch.setenv("REGBENCH_SUBMISSION", SUB)
    argv = ["chain-mark", "--cases", str(root), "--out", str(out), "--dataset",
            "multiplex", "--method", "palom", "--index", "0", "--rung-gb", "64",
            "--cause", "not_submitted", "--why", "sbatch refused it"]  # fmt: skip
    assert cli.main(argv) == 0
    assert "1 result(s) of palom on multiplex" in capsys.readouterr().out
    assert _read(path)["cause"] == "not_submitted"


def _plan(out, name, last, done=None, state=None):
    plan = chains.retry_root(out, SUB) / name
    plan.mkdir(parents=True)
    (plan / "run").write_text("palom\npalom\n-\n-\nmultiplex\n")
    (plan / "last").write_text(last)
    if done:
        (plan / "done").write_text(done)
    if state:
        (plan / "state").write_text(state)
    return plan


def test_a_result_the_scheduler_had_not_ended_is_not_the_methods(tmp_path):
    """At settle time, a chain no link finished leaves its last job in the queue: the
    failed results of its units are infrastructure, the rung below's failure
    included. A finished chain, and a task the scheduler ended, keep their cause."""
    root, (c0, c1) = _cases(tmp_path)
    out = tmp_path / "out"
    running = _rec(out, c0, cause="killed", launch_id="60_0")
    ended = _rec(out, c0, "palom_affine", cause="killed", launch_id="61_0")
    below = _rec(out, c1)  # oom at 256 GB; its 512 GB task is still pending
    _plan(out, "rb-palom-multiplex", "0 rb-palom-multiplex-16g-61\n1 rb-palom-multiplex-512g-70\n",
          state="k=6\njob=rb-palom-multiplex-512g\nid=70\n")  # fmt: skip
    _plan(out, "rb-other-multiplex", "0 rb-palom-multiplex-16g-80\n", done="ok - 0\n")
    accounting = {
        "60_0": {"state": "RUNNING"},
        "61_0": {"state": "FAILED"},
        "70_1": {"state": "PENDING"},
    }
    changed = chains.mark_in_flight(out, accounting, SUB, root)
    assert changed == {"-": 1, "rb-palom-multiplex": 1}
    assert _read(running)["cause"] == ladder.INFRASTRUCTURE
    assert "RUNNING" in _read(running)["cause_from"]
    assert _read(ended)["cause"] == "killed"
    rec = _read(below)
    assert rec["cause"] == ladder.INFRASTRUCTURE
    assert rec["cause_from"].startswith("oom at rung 3; the scheduler had not ended")
    assert "rb-palom-multiplex" in rec["cause_from"]
    # what the scorer's summary says about it
    lines = chains.summary_lines(out, SUB)
    assert "1 chain(s) of submission s1 were not finished" in "".join(lines)
    assert (
        "- `rb-palom-multiplex`: at step 6 (rb-palom-multiplex-512g, job 70)" in lines
    )
    assert [p.name for p in chains.unfinished(out, SUB)] == ["rb-palom-multiplex"]
    assert chains.summary_lines(out, "s0") == []


def test_a_unit_whose_next_step_no_link_submitted_is_the_launchers(tmp_path):
    """T1: the chain's links all ended before its 64 GB step was submitted. A unit
    whose task the scheduler DID end, out of memory at 16 GB, never got the rungs above:
    that is the launcher's (`not_submitted`), not the method's `oom`. A unit no later
    step could run (`writer`) keeps its cause."""
    root, (c0, c1) = _cases(tmp_path)
    out = tmp_path / "out"
    oom = _rec(out, c0, attempt=1, mem_requested_gb=16)
    writer = _rec(out, c1, cause="writer", attempt=1)
    plan = _plan(out, "rb-palom-multiplex",
                 "0 rb-palom-multiplex-16g-60\n1 rb-palom-multiplex-16g-60\n",
                 state="k=1\njob=rb-palom-multiplex-16g-again\nid=61\n")  # fmt: skip
    (plan / "step-1").write_text("again\nrb-palom-multiplex-16g-again\n--mem=16G\n")
    (plan / "step-2").write_text("climb\nrb-palom-multiplex-64g\n--mem=64G\n")
    accounting = {"60_0": {"state": "COMPLETED"}, "60_1": {"state": "COMPLETED"}}
    changed = chains.mark_in_flight(out, accounting, SUB, root)
    rec = _read(oom)
    assert rec["cause"] == "not_submitted" and rec[chains.LADDER_CUT] == 64.0
    assert "step 2 (rb-palom-multiplex-64g)" in rec["cause_from"]
    assert _read(writer)["cause"] == "writer"
    assert changed == {"rb-palom-multiplex": 1}


def test_a_cut_rung_over_a_memory_failure_is_capped_and_the_methods(tmp_path):
    """Task 28, B6: the ladder reached the largest rung the partition grants. Over a
    memory failure that is `capped` (the method's: a cost result, imputed) with the cap
    in its error; over any other failure the next rung would run again, it stays
    `not_submitted`."""
    root, (c0, c1) = _cases(tmp_path)
    out = tmp_path / "out"
    oom = _rec(out, c0, mem_requested_gb=512)
    lost = _rec(out, c1, cause="infrastructure", mem_requested_gb=512)
    n = chains.mark(out, root, "multiplex", "palom", "", [0, 1], 720, "capped",
                    "the largest node of cpu has 515000 MB", SUB)  # fmt: skip
    assert n == 2
    rec = _read(oom)
    assert rec["cause"] == "capped" and rec[chains.LADDER_CUT] == 720.0
    assert rec["error"].startswith("capped at 512 GB: the next rung, 720 GB")
    assert _read(lost)["cause"] == "not_submitted"
    import pytest

    with pytest.raises(ValueError):
        chains.mark(out, root, "multiplex", "palom", "", [0], 1, "oom", "x", SUB)


def test_a_watcher_that_gave_up_leaves_its_units_as_watcher_gave_up(tmp_path):
    """S3: the chain's watcher gave up (`gave_up`): what it was waiting for, and what
    it never submitted, are `watcher_gave_up` (not the method's), not infrastructure
    or not_submitted."""
    root, (c0, c1) = _cases(tmp_path)
    out = tmp_path / "out"
    oom = _rec(out, c0, attempt=1, mem_requested_gb=16)
    waiting = _rec(out, c1, attempt=1)
    plan = _plan(out, "rb-palom-multiplex",
                 "0 rb-palom-multiplex-16g-60\n1 rb-palom-multiplex-16g-62\n",
                 state="k=1\njob=rb-palom-multiplex-16g-again\nid=61\n")  # fmt: skip
    (plan / "step-1").write_text("again\nrb-palom-multiplex-16g-again\n--mem=16G\n")
    (plan / "step-2").write_text("climb\nrb-palom-multiplex-64g\n--mem=64G\n")
    (plan / "gave_up").write_text("2026-10-18T10:00:00 the queue could not be read\n")
    accounting = {"60_0": {"state": "COMPLETED"}, "62_1": {"state": "PENDING"}}
    chains.mark_in_flight(out, accounting, SUB, root)
    rec = _read(oom)
    assert rec["cause"] == "watcher_gave_up" and "it gave up" in rec["cause_from"]
    rec = _read(waiting)
    assert rec["cause"] == "watcher_gave_up"
    assert "the watcher gave up: " in rec["cause_from"]


def test_a_chain_whose_units_cannot_be_listed_is_written_down(tmp_path, capsys):
    """S2: a run's chain whose `run` file is unreadable, or whose cases cannot be read,
    is never passed over in silence: each unit is a row of unmarked_units.csv, and the
    settle command says how many."""
    root, _ = _cases(tmp_path)
    out = tmp_path / "out"
    bad = _plan(out, "rb-broken-multiplex", "0 rb-x-16g-60\n3 rb-x-16g-60\n")
    (bad / "run").write_text("palom\n")  # short: unreadable
    gone = _plan(out, "rb-gone-multiplex", "1 rb-y-16g-61\n")
    (gone / "run").write_text("palom\npalom\n-\n-\nno_such_dataset\n")
    (gone / "step-1").write_text("climb\nrb-y-64g\n--mem=64G\n")
    chains.mark_in_flight(out, {}, SUB, root)
    rows = chains.read_unmarked(out)
    assert [(r["plan"], r["index"]) for r in rows] == [
        ("rb-broken-multiplex", "0"),
        ("rb-broken-multiplex", "3"),
        ("rb-gone-multiplex", "1"),
    ]
    assert "unreadable" in rows[0]["reason"] and "unit(s) its cases list" in rows[2]["reason"]
    # nothing to report: the file is the header alone
    other = tmp_path / "other"
    chains.mark_in_flight(other, {}, SUB, root)
    assert (other / "logs" / chains.UNMARKED).read_text().strip() == "plan,index,reason"
