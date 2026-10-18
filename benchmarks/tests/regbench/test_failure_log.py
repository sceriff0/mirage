"""The failure log: one row per event, and whose each is (ruling R3.17)."""

import csv
import json
import re
import subprocess
from pathlib import Path

import pytest

from benchmarks.regbench import failure_log as fl
from benchmarks.regbench import ladder, losses, methods, resources, score, selfcheck
from benchmarks.regbench.cases import result_dir, write_json
from benchmarks.tests.regbench.test_pipeline import fake, good, toy_case, true_map

REGBENCH = Path(fl.__file__).parent


# ── whose: total, and one owner per cause ────────────────────────────────────────────
def _literal_causes():
    """Every cause written as a literal anywhere in the package's code."""
    found = {}
    pattern = re.compile(r"""["']?cause["']?\s*[:=]\s*["']([a-z_]+)["']""")
    for path in sorted(REGBENCH.rglob("*.py")):
        if path.name in ("selfcheck.py", "failure_log.py"):
            continue  # selfcheck.py's `cause` is a sentence, not this vocabulary
        for name in pattern.findall(path.read_text()):
            found.setdefault(name, path.name)
    return found


def test_every_cause_has_exactly_one_owner():
    vocabulary = {
        *resources.CAUSES,
        *score.CAUSES,
        *resources.MEMORY_CAUSES,
        *resources.NOT_THE_METHOD,
        *score.NOT_THE_METHOD,
        ladder.INFRASTRUCTURE,
        *fl.SELFCHECK_CAUSE.values(),
        *_literal_causes(),
        *fl.REALCHECK_CAUSE.values(),
        "realcheck_flagged",
        "tissue_missing",
        "unreadable_record",
        *losses.CAUSES,
    }
    assert "oom" in _literal_causes()  # the scan reads something
    # the gate's causes are gone with the gate: reported, never a gate
    assert "gate_blocked" not in fl.WHOSE and "dropped_row" not in fl.WHOSE
    assert "gate" not in fl.STAGES
    missing = sorted(c for c in vocabulary if c not in fl.WHOSE)
    assert not missing, f"causes with no owner in failure_log.WHOSE: {missing}"
    assert set(fl.WHOSE.values()) <= set(fl.OWNERS)
    for cause in vocabulary:
        assert fl.classify(cause) in fl.OWNERS
    with pytest.raises(KeyError):
        fl.classify("a_cause_nobody_classified")


def test_the_two_tables_of_what_is_not_the_methods_agree():
    """`NOT_THE_METHOD` (what the summaries leave out) is what `whose` says."""
    for cause in resources.CAUSES:
        if cause == "not_run":
            continue
        assert (fl.classify(cause) != fl.METHOD) == (
            cause in resources.NOT_THE_METHOD
        ), cause
    # resources.py is frozen: the causes Task 28 added are losses.NOT_THE_METHOD_EXTRA
    assert tuple(sorted(score.NOT_THE_METHOD)) == tuple(
        sorted({*resources.NOT_THE_METHOD, *losses.NOT_THE_METHOD_EXTRA})
    )
    for cause in losses.NOT_THE_METHOD_EXTRA:
        assert fl.classify(cause) != fl.METHOD, cause
    assert "capped" not in score.NOT_THE_METHOD and fl.classify("capped") == fl.METHOD


def test_every_non_ok_selfcheck_status_is_an_event_of_the_benchmark():
    assert set(selfcheck.NOT_OK) <= set(fl.SELFCHECK_CAUSE)
    for cause in [*fl.SELFCHECK_CAUSE.values(), *fl.REALCHECK_CAUSE.values()]:
        assert fl.classify(cause) == fl.BENCHMARK, cause


# ── one fixture per cause of a registration attempt ──────────────────────────────────
def record(cause, **kw):
    rec = {
        "method": "m",
        "label": "",
        "variant": "m",
        "dataset": "multiplex",
        "case_id": "P1_r1",
        "group": "P1",
        "unit": "pair",
        "ok": False,
        "cause": cause,
        "error": f"boom ({cause})\nsecond line",
        "attempt": 2,
        "retry": 0,
        "mem_requested_gb": 64.0,
        "submission": "s1",
        "launch_id": "77_3",
        "log_path": "/out/logs/rb-m-multiplex-r2-77_3.log",
        "resources": {"wall_s": 12.0, "peak_rss_mb": 2048.0},
        "previous_attempts": [],
    }
    rec.update(kw)
    return rec


EXPECTED = {
    "exception": ("register", "method", "failed"),
    "killed": ("register", "method", "failed"),
    "oom": ("register", "method", "failed"),
    "timeout": ("register", "method", "failed"),
    "gpu_oom": ("register", "method", "failed"),
    "writer": ("writer", "method", "failed"),
    "writer_oom": ("writer", "method", "failed"),
    "benchmark_oom": ("benchmark_points", "benchmark", "lost"),
    "benchmark": ("benchmark_points", "benchmark", "lost"),
    "infrastructure": ("register", "infrastructure", "lost"),
    "not_submitted": ("register", "launcher", "lost"),
    # the benchmark's own doing, found while the method was registering
    "identical_overlap": ("register", "benchmark", "lost"),
}


@pytest.mark.parametrize("cause", [c for c in resources.CAUSES if c != "not_run"])
def test_each_cause_of_an_attempt_gives_its_row(cause):
    (row,) = fl.run_events(record(cause))
    stage, whose, final = EXPECTED[cause]
    assert (row["stage"], row["cause"], row["whose"]) == (stage, cause, whose)
    assert row["final_outcome_of_unit"] == final
    assert row["message"] == f"boom ({cause})"
    assert (row["attempt"], row["rung_gb"], row["elapsed_s"]) == (2, 64.0, 12.0)
    assert row["peak_rss_gb"] == 2.0 and row["launch_id"] == "77_3"
    assert row["log_path"].endswith("77_3.log") and row["climbed_to"] == fl.NONE
    assert row["cause_from"] == "its own record" and row["run"] == "m"
    assert set(row) == set(fl.COLUMNS)


def test_expected_covers_every_cause_of_an_attempt():
    assert set(EXPECTED) == set(resources.CAUSES) - {"not_run"}


def test_a_result_nobody_ran_is_the_launchers():
    (row,) = fl.run_events({}, "multiplex", "P1_r1", "m")
    assert (row["cause"], row["whose"], row["final_outcome_of_unit"]) == (
        "not_run",
        "launcher",
        "lost",
    )


def test_the_scheduler_s_word_is_kept_when_it_decided_the_cause():
    (row,) = fl.run_events(
        record(
            "infrastructure",
            cause_from="scheduler state NODE_FAIL (the driver's record said killed)",
            resources=None,
            scheduler={"state": "NODE_FAIL", "elapsed_s": 301.0, "exit_code": "0:9"},
        )
    )
    assert row["scheduler_state"] == "NODE_FAIL" and row["elapsed_s"] == 301.0
    assert row["cause_from"].startswith("scheduler state NODE_FAIL")
    assert row["exit_code"] == "0:9" and row["peak_rss_gb"] == ""


def test_earlier_rungs_are_events_even_when_the_unit_ended_ok():
    earlier = [
        {
            "attempt": 1,
            "cause": "oom",
            "mem_requested_gb": 16.0,
            "launch_id": "70_3",
            "retry": 0,
            "wall_s": 5.0,
            "error": "MemoryError",
            "log_path": "/out/logs/a.log",
            "peak_rss_mb": 15000.0,
        },
        {
            "attempt": 2,
            "cause": "infrastructure",
            "mem_requested_gb": 64.0,
            "launch_id": "71_3",
            "retry": 0,
            "wall_s": None,
            "scheduler_state": "NODE_FAIL",
            "scheduler_elapsed_s": 40.0,
            "cause_from": "scheduler state NODE_FAIL",
        },
    ]
    rows = fl.run_events(
        record("", ok=True, attempt=2, retry=1, previous_attempts=earlier)
    )
    assert [(r["cause"], r["whose"], r["climbed_to"]) for r in rows] == [
        ("oom", "method", "64 GB"),
        ("infrastructure", "infrastructure", "the same rung again"),
    ]
    assert {r["final_outcome_of_unit"] for r in rows} == {"ok at rung 2"}
    assert rows[1]["elapsed_s"] == 40.0 and rows[1]["scheduler_state"] == "NODE_FAIL"
    assert rows[0]["log_path"] == "/out/logs/a.log"
    assert fl.not_ok(rows) == []  # the unit ended ok: nothing in the "not ok" list


def test_a_patient_launch_that_fails_is_one_event_per_result_name():
    rows = [
        r
        for case_id in ("P1_r1", "P1_r2", "P1_r3")
        for r in fl.run_events(record("oom", unit="patient", case_id=case_id))
    ]
    (row,) = fl.collapse(rows)
    assert row["case_id"] == "P1_r1;P1_r2;P1_r3"
    pair_rows = [r for c in ("a", "b") for r in fl.run_events(record("oom", case_id=c))]
    assert len(fl.collapse(pair_rows)) == 2


# ── selfcheck, realcheck, tissue, the scorer's own reading ──────────────────────────
def said(run, selfcheck="ok", realcheck="", log="", realcheck_log=""):
    """One run's verdicts, as `score.read_verdicts` reads them from verdicts.tsv."""
    rec = {
        "run": run,
        "selfcheck": selfcheck,
        "realcheck": realcheck,
        "reference": "",
        "log": log,
        "rows": "",
        "realcheck_row": run if realcheck else "",
    }
    if realcheck_log:
        rec["realcheck_log"] = realcheck_log
    return rec


def test_a_failed_selfcheck_and_a_flagged_realcheck_are_events_not_failures():
    verdicts = [
        said("valis120", "inconsistent", log="/out/logs/sc.log"),
        said("deeperhistreg_gpu"),
        said("valis_mirage", realcheck="flagged", realcheck_log="flagged on P1_r1"),
        said("valis120_hires", realcheck="ok"),
        said("palom", "not_checked", log="no selfcheck record"),
        said("stare"),
    ]
    rows = fl.selfcheck_events(verdicts, {"valis120": ["valis120_rigid", "valis120"]})
    got = [(r["run"], r["stage"], r["cause"], r["whose"]) for r in rows]
    assert got == [
        ("valis120", "selfcheck", "selfcheck_inconsistent", "benchmark"),
        ("valis_mirage", "realcheck", "realcheck_flagged", "benchmark"),
        ("palom", "selfcheck", "selfcheck_not_checked", "benchmark"),
    ]
    assert rows[0]["variant"] == "valis120_rigid;valis120"
    assert rows[0]["message"] == "selfcheck: inconsistent (/out/logs/sc.log)"
    assert rows[1]["variant"] == "valis_mirage"
    assert rows[1]["message"] == "realcheck: flagged (flagged on P1_r1)"
    assert rows[2]["message"] == "selfcheck: not_checked (no selfcheck record)"
    # reported, never a gate: the run went on, so no unit ENDED with any of them, and
    # none is the method's failure
    assert {r["final_outcome_of_unit"] for r in rows} == {fl.REPORTED}
    assert fl.not_ok(rows) == []
    assert all(r["whose"] != fl.METHOD for r in rows)
    assert all(t["ended_the_unit"] == 0 for t in fl.count_table(rows))
    assert "blocked" not in {r["final_outcome_of_unit"] for r in rows}


@pytest.mark.parametrize("status", sorted(fl.SELFCHECK_CAUSE))
def test_each_selfcheck_outcome_that_is_not_ok_gives_its_row(tmp_path, status):
    d = tmp_path / "logs" / "selfcheck" / "palom"
    d.mkdir(parents=True)
    rec = {
        "status": status,
        "submission": "s1",
        "variants": {"palom_affine": "ok", "palom": status},
        "attempts": [
            {
                "attempt": 1,
                "mem_requested_gb": 64.0,
                "status": "not_checked",
                "why": "palom: oom",
                "log_path": "/out/logs/sc1.log",
                "submission": "s1",
            },
            {
                "attempt": 2,
                "mem_requested_gb": 256.0,
                "status": status,
                "log_path": "/out/logs/sc2.log",
                "submission": "s1",
            },
        ],
    }
    (d / "selfcheck.json").write_text(json.dumps(rec))
    first, last = fl.selfcheck_attempt_events(tmp_path / "logs" / "selfcheck")
    assert (first["cause"], first["climbed_to"], first["message"]) == (
        "selfcheck_not_checked",
        "256 GB",
        "palom: oom",
    )
    assert (last["stage"], last["cause"], last["whose"]) == (
        "selfcheck",
        fl.SELFCHECK_CAUSE[status],
        "benchmark",
    )
    assert last["variant"] == "palom" and last["log_path"] == "/out/logs/sc2.log"
    assert last["final_outcome_of_unit"] == status


def test_a_selfcheck_that_ended_ok_leaves_only_its_failed_attempts(tmp_path):
    d = tmp_path / "stare"
    d.mkdir()
    rec = {
        "status": "ok",
        "variants": {"stare": "ok"},
        "attempts": [
            {"attempt": 1, "mem_requested_gb": 64.0, "status": "not_checked"},
            {"attempt": 2, "mem_requested_gb": 256.0, "status": "ok"},
        ],
    }
    (d / "selfcheck.json").write_text(json.dumps(rec))
    (row,) = fl.selfcheck_attempt_events(tmp_path)
    assert row["final_outcome_of_unit"] == "ok" and row["rung_gb"] == 64.0
    assert fl.not_ok([row]) == []


def test_a_realcheck_flag_on_an_ok_result_is_an_event():
    (row,) = fl.run_events(
        record("", ok=True, realcheck={"status": "flagged: 0.6 px", "flag": True})
    )
    assert (row["stage"], row["cause"], row["whose"]) == (
        "realcheck",
        "realcheck_flagged",
        "benchmark",
    )


def test_only_a_tissue_error_is_an_event():
    missing = [
        {"dataset": "multiplex", "case_id": "a", "group": "P", "variant": "stare_rigid",
         "reason": "the method writes no image for this variant"},
        {"dataset": "multiplex", "case_id": "a", "group": "P", "variant": "stare",
         "reason": "the written image could not be reduced: OSError: gone"},
    ]  # fmt: skip
    (row,) = fl.tissue_events(missing)
    assert (row["variant"], row["cause"], row["whose"]) == (
        "stare",
        "tissue_missing",
        "benchmark",
    )


# ── end to end: the scorer writes the log on every run ───────────────────────────────
def log_rows(out):
    with open(out / "tables" / "failure_log.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def test_the_log_is_written_even_when_nothing_failed(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    methods.run_case(good(), toy_case(root, "P1_r1", "P1"), out)
    res = score.score(root, out, ["multiplex"], with_tissue=False)
    assert res["failure_log"] == [] and log_rows(out) == []
    header = (out / "tables" / "failure_log.csv").read_text().strip().split(",")
    assert header == list(fl.COLUMNS)
    text = (out / "tables" / "summary.md").read_text()
    assert "No failed or lost event, and no check that was not ok." in text
    assert "blocked" not in text.lower()


def test_the_scorer_logs_failures_of_every_kind_and_names_whose(tmp_path, monkeypatch):
    root, out = tmp_path / "cases", tmp_path / "out"
    cases = [toy_case(root, f"P{k}_r1", f"P{k}") for k in (1, 2)]
    monkeypatch.setenv("REGBENCH_LOG", "/out/logs/rb-task-9.log")

    def register(case, work, opts):
        if case.case_id == "P2_r1":
            raise MemoryError("Unable to allocate 99 GiB")
        yield "flaky", true_map(), {}

    for c in cases:
        methods.run_case(fake("flaky", ["flaky"], register), c, out)
        methods.run_case(good(), c, out)
    # the scheduler lost `good`'s second case (ladder.mark_lost writes this record)
    lost = result_dir(out, "good", cases[1]) / "run.json"
    rec = json.loads(lost.read_text())
    rec.update(ok=False, cause="infrastructure", error=ladder.LOST, cause_from="none")
    write_json(lost, rec)
    (result_dir(out, "good", cases[1]) / "warped.npz").unlink()
    verdicts = out / "logs" / "verdicts.tsv"
    verdicts.parent.mkdir(parents=True, exist_ok=True)
    verdicts.write_text(
        "run\tcheck\tstatus\treference\trows\tlog\n"
        "ghost\tselfcheck\tcrashed\t-\tghost\t/out/logs/sc.log\n"
        "ghost\trealcheck\t-\t-\t-\t-\n"
    )
    res = score.score(
        root,
        out,
        ["multiplex"],
        expected=["flaky", "good", "ghost"],
        primary=["flaky", "good", "ghost"],
        verdicts=verdicts,
        with_tissue=False,
    )
    rows = {(r["run"], r["cause"]): r for r in log_rows(out)}
    oom = rows[("flaky", "oom")]
    assert (oom["whose"], oom["stage"], oom["case_id"]) == (
        "method",
        "register",
        "P2_r1",
    )
    assert oom["log_path"] == "/out/logs/rb-task-9.log"
    assert "Unable to allocate" in oom["message"]
    assert rows[("good", "infrastructure")]["whose"] == "infrastructure"
    assert rows[("good", "infrastructure")]["final_outcome_of_unit"] == "lost"
    # a crashed selfcheck is the benchmark's event; the run is never said blocked
    crashed = rows[("ghost", "selfcheck_crashed")]
    assert (crashed["whose"], crashed["final_outcome_of_unit"]) == (
        "benchmark",
        fl.REPORTED,
    )
    assert "/out/logs/sc.log" in crashed["message"]
    # a run nobody ran is not run, folded into one row, whatever its selfcheck said
    assert rows[("ghost", "not_run")]["case_id"] == "2 case(s)"
    assert rows[("ghost", "not_run")]["final_outcome_of_unit"] == "lost"
    text = (out / "tables" / "summary.md").read_text()
    assert "## Failures: every event, and whose it is" in text
    assert "| flaky | register | oom | method | 1 | 1 |" in text
    assert "### Units whose final outcome is not ok" in text
    assert "/out/logs/rb-task-9.log" in text
    assert "unit(s) ended for a reason that is NOT the method's" in text
    # and the lost round is out of `good`'s summary, named
    assert [e["case_id"] for e in res["excluded_rounds"]] == ["P2_r1"]


def test_an_earlier_attempt_carries_its_error_and_log_into_the_next(
    tmp_path, monkeypatch
):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = toy_case(root, "P1_r1", "P1")
    state = {"n": 0}

    def register(case, work, opts):
        state["n"] += 1
        if state["n"] == 1:
            raise MemoryError("first rung too small")
        yield "m", true_map(), {}

    monkeypatch.setenv("REGBENCH_ATTEMPT", "1")
    monkeypatch.setenv("REGBENCH_MEM_GB", "16")
    monkeypatch.setenv("REGBENCH_LOG", "/out/logs/rung1.log")
    methods.run_case(fake("m", ["m"], register), case, out)
    monkeypatch.setenv("REGBENCH_ATTEMPT", "2")
    monkeypatch.setenv("REGBENCH_MEM_GB", "64")
    monkeypatch.setenv("REGBENCH_LOG", "/out/logs/rung2.log")
    methods.run_case(fake("m", ["m"], register), case, out, resume=False)
    score.score(root, out, ["multiplex"], with_tissue=False)
    (row,) = log_rows(out)
    assert (row["cause"], row["rung_gb"], row["climbed_to"]) == ("oom", "16.0", "64 GB")
    assert row["log_path"] == "/out/logs/rung1.log"
    assert "first rung too small" in row["message"]
    assert row["final_outcome_of_unit"] == "ok at rung 2"


# ── the last line of a task's log ────────────────────────────────────────────────────
def test_the_end_line_uses_the_log_s_words(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("REGBENCH_LOG", "/out/logs/x.log")
    monkeypatch.setenv("REGBENCH_END_FILE", str(tmp_path / "end" / "x.end"))
    line = fl.emit_end("run", "failed", "writer_oom", run="palom", unit="P1 r1")
    assert line == (
        "REGBENCH_END kind=run run=palom unit=P1_r1 status=failed cause=writer_oom "
        "whose=method log=/out/logs/x.log"
    )
    assert (tmp_path / "end" / "x.end").read_text() == line + "\n"
    assert capsys.readouterr().out.strip() == line
    ok = fl.end_line("score", "ok")
    assert "cause=- whose=-" in ok


def test_unit_end_reads_the_first_failure():
    assert fl.unit_end([{"ok": True}, {"ok": True}]) == ("ok", "")
    assert fl.unit_end([{"ok": True}, {"ok": False, "cause": "oom"}]) == (
        "failed",
        "oom",
    )
    assert fl.unit_end([{"ok": False, "cause": "infrastructure"}]) == (
        "lost",
        "infrastructure",
    )
    assert fl.unit_end([{}, {}]) == ("lost", "not_run")


# ── what a chain's watcher may still submit for a unit (fleet wave 2, F3/F4) ──────
def _rec(cause, attempt=2, **more):
    if cause == "ok":
        return {"ok": True, "cause": "", "attempt": attempt, **more}
    return {"ok": False, "cause": cause, "attempt": attempt, **more}


FINAL_KILLED = {"scheduler": {"state": "COMPLETED"}, "launch_id": "7_0"}
SITUATIONS = {
    "ok": [_rec("ok")],
    "oom": [_rec("oom")],
    "writer_oom": [_rec("writer_oom")],
    "benchmark_oom": [_rec("benchmark_oom")],
    "killed, no end state yet": [_rec("killed", launch_id="7_0")],
    "killed, ended COMPLETED": [_rec("killed", **FINAL_KILLED)],
    "lost once": [_rec("infrastructure", retry=0)],
    "lost twice": [_rec("infrastructure", retry=1)],
    "exception": [_rec("exception")],
    "writer": [_rec("writer")],
    "timeout": [_rec("timeout")],
    "gpu_oom": [_rec("gpu_oom")],
    "stage oom retried where it ran": [_rec("oom", no_rerun=True)],
    # F3: the round that failed first is not the one that ran out of memory
    "writer, then writer_oom": [_rec("writer"), _rec("writer_oom")],
    "exception, then oom": [_rec("exception"), _rec("oom")],
    "ok, then lost twice": [_rec("ok"), _rec("infrastructure", retry=1)],
    "killed final, then oom": [_rec("killed", **FINAL_KILLED), _rec("oom")],
}


def test_the_watcher_takes_a_unit_whenever_one_of_its_records_climbs():
    """F3: a unit goes up as a whole when ANY of its records ran out of memory
    (ladder.unit_decision), whatever cause the first failed record has. F4: a unit no
    later task can run (killed with an end state, lost twice, a cause of the method's
    own) is taken by no step: a task for it would only exit at once, after days of
    queueing for a large slot."""
    wants = {k: fl.chain_wants(v) for k, v in SITUATIONS.items()}
    assert wants["writer, then writer_oom"] == ("climb",)
    assert wants["exception, then oom"] == ("climb",)
    assert wants["killed final, then oom"] == ("climb",)
    for memory in ("oom", "writer_oom", "benchmark_oom"):
        assert wants[memory] == ("climb",)
    assert wants["killed, no end state yet"] == ("again", "climb")
    assert wants["lost once"] == ("again", "climb")
    for final in (
        "ok", "killed, ended COMPLETED", "lost twice", "exception", "writer",
        "timeout", "gpu_oom", "stage oom retried where it ran", "ok, then lost twice",
    ):  # fmt: skip
        assert wants[final] == (), final
    # no record of this submission: the task never started; every step may need it
    assert fl.chain_wants([]) == fl.chain_wants([{}, {}]) == ("again", "climb")


@pytest.mark.parametrize("name", sorted(SITUATIONS))
def test_the_watchers_filter_never_drops_a_unit_the_ladder_would_run(tmp_path, name):
    """One predicate, two readers: the watcher's filter (``chain_wants``, read off the
    task's end line) and the ladder's decision (``ladder.unit_decision``, made by the
    task itself). The filter may submit a task that then finds nothing to do; it must
    never keep out a unit the ladder would run. The accounting is what the next task
    is given (sacct), so a record with an end state is settled as it will be."""
    records = SITUATIONS[name]
    accounting = {
        r["launch_id"]: {"state": r["scheduler"]["state"], "elapsed_s": 1, "cpu_s": 1}
        for r in records
        if r.get("scheduler")
    }
    wants = fl.chain_wants(records)
    for task, rung in ((ladder.AGAIN, 2), (ladder.CLIMB, 3)):
        paths = []
        for i, rec in enumerate(records):
            path = tmp_path / task / f"v{i}" / "run.json"
            write_json(path, rec)
            paths.append(path)
        go, why, _ = ladder.unit_decision(
            paths, task, rung, accounting=accounting, waited=True, mem_gb=64
        )
        if go:
            assert task in wants, (name, task, why)


SLURM = REGBENCH / "slurm"


def _run_env_function(tmp_path, body, **env):
    script = f'set -u\nsource "{SLURM}/env.sh"\n{body}\n'
    base = {
        "PATH": "/usr/bin:/bin",
        "REGBENCH_REPO": str(REGBENCH.parents[1]),
        "REGBENCH_OUT": str(tmp_path / "out"),
        "REGBENCH_CASES": str(tmp_path / "cases"),
    }
    done = subprocess.run(
        ["bash", "-c", script], env={**base, **env}, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_a_task_names_its_own_log_as_the_launcher_does(tmp_path):
    out = _run_env_function(
        tmp_path,
        'echo "$REGBENCH_LOG"; echo "$REGBENCH_END_FILE"',
        SLURM_JOB_ID="123",
        SLURM_JOB_NAME="rb-stare-multiplex-r1",
        SLURM_ARRAY_JOB_ID="120",
        SLURM_ARRAY_TASK_ID="4",
    ).split()
    assert out[0] == f"{tmp_path}/out/logs/rb-stare-multiplex-r1-120_4.log"
    assert out[1] == f"{tmp_path}/out/logs/end/rb-stare-multiplex-r1-120_4.end"
    # the launcher's own pattern, so the two cannot drift apart
    launcher = (SLURM / "submit_cohort.sh").read_text()
    assert 'LOG_ARRAY="--output=$REGBENCH_OUT/logs/%x-%A_%a.log"' in launcher
    assert 'LOG_ONE="--output=$REGBENCH_OUT/logs/%x-%j.log"' in launcher
    one = _run_env_function(
        tmp_path, 'echo "$REGBENCH_LOG"', SLURM_JOB_ID="9", SLURM_JOB_NAME="rb-sc-x"
    ).strip()
    assert one == f"{tmp_path}/out/logs/rb-sc-x-9.log"


def test_the_batch_script_s_last_line_is_the_step_s_or_a_fallback(tmp_path):
    env = {"SLURM_JOB_ID": "9", "SLURM_JOB_NAME": "rb-x"}
    died = _run_env_function(
        tmp_path, "regbench_end run 137 run=stare unit=index_3", **env
    )
    # the last line; the operator's `[regbench] end` line just before it (B1)
    assert died.strip().splitlines()[-2].startswith("[regbench] end kind=")
    assert " outcome=no_end_line/killed " in died.strip().splitlines()[-2]
    assert died.strip().splitlines()[-1] == (
        "REGBENCH_END kind=run run=stare unit=index_3 status=no_end_line cause=killed "
        f"whose=method exit=137 log={tmp_path}/out/logs/rb-x-9.log"
    )
    assert fl.classify("killed") == "method"  # the fallback's words are the table's
    wrote = _run_env_function(
        tmp_path,
        'echo "REGBENCH_END kind=run status=ok" > "$REGBENCH_END_FILE"\n'
        "regbench_end run 0 run=stare",
        **env,
    )
    assert wrote.strip().splitlines()[-1] == "REGBENCH_END kind=run status=ok"
    assert " outcome=ok " in wrote.strip().splitlines()[-2]


START = re.compile(
    r"^\[regbench\] start kind=\S+ job=\S+ run=\S+ label=\S+ dataset=\S+ unit=\S+ "
    r"rung_gb=\S+ attempt=\S+( \S+=\S+)* out=\S+ commit=\S+ host=\S+ at=\S+$"
)
END = re.compile(
    r"^\[regbench\] end kind=\S+ job=\S+ outcome=\S+ wall=(\d+s|-) "
    r"peak_rss=(\d+MB|-) record=\S+ exit=\S+$"
)


def test_a_job_says_how_it_started_and_ended_in_one_format(tmp_path):
    """Fleet wave 2, B1: the operator greps one format across every job's log."""
    env = {"SLURM_JOB_ID": "9", "SLURM_JOB_NAME": "rb-x", "REGBENCH_COMMIT": "abc"}
    said = _run_env_function(
        tmp_path,
        "regbench_start run run=stare dataset=multiplex unit=index_3 rung_gb=64 "
        "attempt=2\n"
        'echo "REGBENCH_END kind=run status=failed cause=oom whose=method '
        'peak_rss_mb=61234 record=/o/r/run.json exit=0" > "$REGBENCH_END_FILE"\n'
        "regbench_end run 0",
        **env,
    ).splitlines()
    assert START.match(said[0]), said[0]
    assert " run=stare label=- dataset=multiplex unit=index_3 rung_gb=64 " in said[0]
    assert " commit=abc " in said[0]
    assert END.match(said[1]), said[1]
    assert " outcome=failed/oom " in said[1] and " peak_rss=61234MB " in said[1]
    assert " record=/o/r/run.json " in said[1]
    assert said[2].startswith("REGBENCH_END ") and len(said) == 3
    # a job that ends without its end line (set -e, a die) still says how
    done = subprocess.run(
        ["bash", "-c", f'source "{SLURM}/env.sh"\nregbench_start retry\nexit 3'],
        env={"PATH": "/usr/bin:/bin", "REGBENCH_OUT": str(tmp_path / "out"), **env},
        capture_output=True, text=True,
    )  # fmt: skip
    assert done.returncode == 3
    last = done.stdout.splitlines()[-1]
    assert END.match(last) and " outcome=exit_3 " in last, last


@pytest.mark.parametrize(
    "name", sorted(p.name for p in SLURM.glob("*.sbatch"))
)
def test_every_job_prints_the_start_line(name):
    code = "\n".join(
        ln for ln in (SLURM / name).read_text().splitlines()
        if not ln.lstrip().startswith("#")
    )  # fmt: skip
    assert "regbench_start " in code or 'echo "[regbench] start kind=' in code, name
    # its REGBENCH_END lines go out through the function that prints the end line first
    for line in code.splitlines():
        if (
            "REGBENCH_END kind=" in line
            and line.lstrip().startswith("echo ")
            and not line.rstrip().endswith("\\")  # redirected on the next line
        ):
            assert "> " in line or name == "inputs.sbatch", (name, line)


@pytest.mark.parametrize(
    "name",
    ["run_rung.sbatch", "selfcheck.sbatch", "score_cohort.sbatch"],
)
def test_every_batch_script_ends_with_an_end_line(name):
    lines = [
        line.split("#")[0].rstrip()
        for line in (SLURM / name).read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    code = "\n".join(lines)
    # after EVERY Python step, not only on an early exit: the last one included
    after = code[code.rindex("regbench_py ") :]
    assert "REGBENCH_END" in after or "regbench_end " in after, name
    # every fallback line written in shell uses a cause and an owner of the table
    for cause, whose in re.findall(r"cause=([a-z_]+) whose=([a-z]+)", code):
        assert fl.WHOSE[cause] == whose, (name, cause, whose)


# ── a registration in several scheduler jobs: every failed stage attempt ──────
def stage_job(start, wall, **more):
    rec = {"started": start, "ended": start + wall, "wall_s": wall, "cpu_s": wall}
    return {**rec, "peak_rss_mb": 500.0, "cpus": 2, "attempt": 1, "ok": True, **more}


def staged_run(ok=True):
    """A round whose tile 1 was killed for memory and retried, whose tile 2 hit its
    time limit without recording an end and was retried."""
    from benchmarks.regbench import stages

    oom = stage_job(
        100.0, 20.0, ok=False, exit=137, cause="oom", mem_gb=4, launch_id="50_1",
        log_path="/logs/tile-50_1.log", error="Killed",
    )  # fmt: skip
    cut = {
        "attempt": 1, "ok": False, "cause": "killed", "exit": None, "started": 100.0,
        "cpus": 2, "mem_gb": 4, "time_limit": "0-02:00:00", "launch_id": "50_2",
    }  # fmt: skip
    tiles = [
        stage_job(100.0, 40.0),
        {**stage_job(200.0, 50.0), "attempt": 2, "history": [oom]},
        {**stage_job(100.0 + 7200.0, 60.0), "attempt": 2, "history": [cut]},
    ]
    jobs = stages.summarise(stage_job(0.0, 60.0), tiles, stage_job(8000.0, 10.0))
    return jobs, {
        "ok": ok,
        "cause": "" if ok else "exception",
        "method": "stare",
        "variant": "stare",
        "dataset": "multiplex",
        "case_id": "P1_a",
        "unit": "pair",
        "attempt": 1,
        "info": {"jobs": {**jobs, "execution": "jobs"}},
        "native": {
            "stitch_job": {
                "failed_attempts": [
                    {"attempt": 1, "exit": 143, "cause": "timeout", "wall_s": 14400.0}
                ]
            }
        },
    }


def test_every_retried_or_failed_stage_attempt_is_a_row():
    jobs, run = staged_run()
    assert jobs["unmeasured_attempts"] == 1 and jobs["wasted_attempts"] == 1
    rows = fl.run_events(run)
    assert [(r["message"].split(":")[0], r["cause"], r["whose"]) for r in rows] == [
        ("stage job TILE row 1", "oom", "method"),
        ("stage job TILE row 2", "timeout", "method"),
        ("stage job STITCH", "writer", "method"),
    ]
    oom, cut, stitch = rows
    assert (oom["exit_code"], oom["elapsed_s"], oom["rung_gb"]) == (137, 20.0, 4)
    assert oom["launch_id"] == "50_1" and oom["log_path"] == "/logs/tile-50_1.log"
    assert oom["climbed_to"] == "the next attempt of that job"
    assert oom["final_outcome_of_unit"] == "ok at rung 1" and oom["retry"] == "stage"
    # ended by its time limit: no end recorded, and the next attempt started 2 h later
    assert cut["cause_from"].startswith("no end recorded, and the next attempt started")
    assert cut["elapsed_s"] == "" and stitch["stage"] == "writer"
    assert set(oom) == set(fl.COLUMNS)


def test_a_stage_job_with_no_end_is_a_timeout_only_where_the_evidence_allows():
    from benchmarks.regbench import stages

    cut = {
        "cause": "killed",
        "exit": None,
        "started": 100.0,
        "time_limit": "0-02:00:00",
    }
    late, soon = {"started": 100.0 + 7200.0}, {"started": 100.0 + 600.0}
    assert stages.attempt_cause(cut, late)[0] == "timeout"
    # the next attempt started before the limit: the job was ended for another reason
    assert stages.attempt_cause(cut, soon)[0] == "killed"
    assert stages.attempt_cause(cut, None)[0] == "killed"  # no later attempt
    assert stages.attempt_cause({**cut, "time_limit": None}, late)[0] == "killed"
    # an attempt that recorded its exit speaks for itself
    assert stages.attempt_cause({"cause": "oom", "exit": 137}, late) == (
        "oom",
        "its own record",
    )


def test_a_rounds_stage_attempts_are_listed_once_for_its_two_result_names():
    _, run = staged_run()
    rigid = {**run, "variant": "stare_rigid", "native": None}
    rows = fl.once_per_round([*fl.run_events(rigid), *fl.run_events(run)])
    assert [r["message"].split(":")[0] for r in rows] == [
        "stage job TILE row 1",
        "stage job TILE row 2",
        "stage job STITCH",
    ]
