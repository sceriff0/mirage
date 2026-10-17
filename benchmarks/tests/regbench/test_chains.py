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
