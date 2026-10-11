"""How a run ends, for every method alike: writer failures, the three memory causes, the
watcher's record, stale records, and how the tables count what is not the method's.

Review of 2026-10-11, items 3, 11 (cost), 12, 13, 15, 16. The ladder's decisions are in
``test_ladder.py``; the VALIS classification in ``test_valis_adapters.py``.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import cost, ladder, methods, resources, score
from benchmarks.regbench.cases import result_dir
from benchmarks.regbench.mapping import MOV_TO_REF, PointMap

from .test_pipeline import fake, toy_case, true_map

REPO = Path(__file__).resolve().parents[3]
JAVA_OOM = "java.lang.OutOfMemoryError: Java heap space"


@pytest.fixture
def toy(tmp_path):
    root = tmp_path / "cases"
    return root, toy_case(root, "P1_r1", "P1"), tmp_path / "out"


def run_json(out, variant, case):
    return json.loads((result_dir(out, variant, case) / "run.json").read_text())


def image(dest, name="registered.tif"):
    import tifffile

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(dest / name, np.zeros((16, 16), np.uint16))
    return {"files": [str(dest / name)], "format": "tiff"}


def two_stage(write_native=None, native_at_yield=None, name="m"):
    """A method with an early result and a final one. Its writer is ``write_native``
    (as STARE, PALOM, DeeperHistReg) or runs at the yield (as VALIS, ASHLAR)."""

    def register(case, work, opts):
        yield f"{name}_rigid", true_map(), {}
        info = {"_state": {"work": str(work)}}
        if native_at_yield is not None:
            mon = opts["monitor"]
            mon.pause()
            try:
                info["native"] = native_at_yield(opts["native_dir"](case.case_id, name))
            finally:
                mon.resume()
        yield name, true_map(), info

    extra = {"write_native": write_native} if write_native else {}
    return fake(name, [f"{name}_rigid", name], register, **extra)


# ── a writer that fails leaves no ok row, whichever kind of writer it is ──────
def _raising(exc):
    def write_native(state, case_id, dest):
        raise exc

    return write_native


WRITER_FAILURES = [
    (OSError("No space left on device"), "writer"),
    (RuntimeError("the stitch produced nothing"), "writer"),
    (MemoryError("Unable to allocate 300. GiB for an array"), "writer_oom"),
    (RuntimeError(JAVA_OOM), "writer_oom"),
]


@pytest.mark.parametrize("exc, cause", WRITER_FAILURES)
@pytest.mark.parametrize("kind", ["write_native", "at_the_yield"])
def test_a_failed_writer_fails_the_row_it_wrote_for(toy, exc, cause, kind):
    """Review item 3. The row scored through a point map of an image that does not exist
    was `ok` for STARE, PALOM and DeeperHistReg and failed for VALIS and ASHLAR."""
    _, case, out = toy
    text = f"{type(exc).__name__}: {exc}"
    if kind == "write_native":
        method = two_stage(write_native=_raising(exc))
    else:  # the method's writer caught its own error and reported it, as VALIS's does
        method = two_stage(native_at_yield=lambda dest: {"error": text})
    assert methods.run_case(method, case, out) == ["m_rigid"]
    run = run_json(out, "m", case)
    assert run["ok"] is False and run["cause"] == cause
    assert "the method's own writer failed" in run["error"] and str(exc) in run["error"]
    assert not (result_dir(out, "m", case) / "warped.npz").exists()
    # the registration's cost is kept: the attempt is not free
    assert run["register_s"] >= 0 and run["resources"]["rss_method"]
    # the earlier result has no image of its own and stands
    assert run_json(out, "m_rigid", case)["ok"] is True
    # and the ladder sees it: a memory death in a writer climbs, another does not
    go, _ = ladder.unit_retry(ladder.unit_paths(out, method, None, [case]))
    assert go is (cause == "writer_oom")


@pytest.mark.parametrize("kind", ["write_native", "at_the_yield"])
def test_a_writer_that_works_leaves_an_ok_row_with_its_image(toy, kind):
    _, case, out = toy
    if kind == "write_native":
        method = two_stage(write_native=lambda state, case_id, dest: image(dest))
    else:
        method = two_stage(native_at_yield=image)
    assert methods.run_case(method, case, out) == ["m_rigid", "m"]
    run = run_json(out, "m", case)
    assert run["ok"] is True and Path(run["native"]["files"][0]).exists()
    assert "phase" not in run and run["cause"] == ""


def test_while_the_writer_runs_the_row_is_not_ok(toy):
    """A writer the scheduler kills leaves what is on disk at that moment."""
    _, case, out = toy
    seen = {}

    def write_native(state, case_id, dest):
        seen.update(run_json(out, "m", case))
        seen["rigid_ok"] = run_json(out, "m_rigid", case)["ok"]
        return image(dest)

    methods.run_case(two_stage(write_native=write_native), case, out)
    assert seen["ok"] is False and seen["cause"] == "killed"
    assert seen["phase"] == "writer" and "writer" in seen["error"]
    assert seen["rigid_ok"] is True
    assert seen["register_s"] > 0  # the registration's bill is already in it
    # the ladder reads such a record as it reads any killed one
    path = result_dir(out, "m", case) / "run.json"
    path.write_text(json.dumps(seen))
    assert ladder.unit_retry([path], "OUT_OF_MEMORY")[0]


def test_the_writer_stays_outside_the_registration_clock(toy):
    _, case, out = toy

    def write_native(state, case_id, dest):
        time.sleep(0.6)
        return image(dest)

    methods.run_case(two_stage(write_native=write_native), case, out)
    run = run_json(out, "m", case)
    assert run["register_s"] < 0.5 <= run["write_s"]


# ── three stretches, three memory causes ──────────────────────────────────────
def test_the_benchmarks_own_out_of_memory_is_not_the_methods(toy):
    """Review item 12: the paused point warp is the benchmark's work."""
    _, case, out = toy

    def hungry(xy):
        raise MemoryError("Unable to allocate 40. GiB for an array")

    def register(case, work, opts):
        yield "b", PointMap(hungry, MOV_TO_REF), {}

    method = fake("b", ["b"], register)
    assert methods.run_case(method, case, out) == []
    run = run_json(out, "b", case)
    assert run["cause"] == "benchmark_oom" and "benchmark's own step" in run["error"]
    assert run["cause"] in resources.NOT_THE_METHOD
    # more memory is still the remedy
    assert ladder.unit_retry(ladder.unit_paths(out, method, None, [case]))[0]


def test_the_methods_own_out_of_memory_is_oom(toy):
    _, case, out = toy

    def register(case, work, opts):
        raise MemoryError("Unable to allocate 40. GiB for an array")
        yield  # pragma: no cover

    methods.run_case(fake("o", ["o"], register), case, out)
    assert run_json(out, "o", case)["cause"] == "oom"


def test_a_method_may_name_the_cause_of_a_failed_round(toy):
    root, case, out = toy
    other = toy_case(root, "P1_r2", "P1")

    def register_group(cases, work, opts):
        yield cases[0].case_id, "g", None, {"error": "micro ran out", "cause": "oom"}
        yield cases[1].case_id, "g", None, {"error": "odd", "cause": "nonsense"}

    methods.run_group(
        fake("g", ["g"], register_group=register_group), [case, other], out
    )
    assert run_json(out, "g", case)["cause"] == "oom"
    assert run_json(out, "g", other)["cause"] == "exception"


@pytest.fixture
def counter(tmp_path, monkeypatch):
    """A cgroup events file the test can raise, and a fast watcher."""
    path = tmp_path / "memory.events"
    path.write_text("oom 0\noom_kill 0\n")
    monkeypatch.setattr(resources, "oom_events_path", lambda: path)
    monkeypatch.setattr(methods, "OOM_POLL_S", 0.05)
    return path


def _killed(counter):
    counter.write_text("oom 1\noom_kill 1\n")
    time.sleep(60)  # a parent waiting for a worker that is gone


def test_a_kill_during_write_native_ends_the_task_as_writer_oom(toy, counter):
    """Review item 3, the hang: the watcher used to stop before the writer, so a pool
    worker killed there left the task waiting for its time limit."""
    _, case, out = toy

    def write_native(state, case_id, dest):
        _killed(counter)

    t0 = time.perf_counter()
    done = methods.run_case(two_stage(write_native=write_native), case, out)
    assert time.perf_counter() - t0 < 20
    assert done == ["m_rigid"]
    run = run_json(out, "m", case)
    assert run["ok"] is False and run["cause"] == "writer_oom"


def test_a_kill_while_the_method_has_the_clock_paused_is_its_writers(toy, counter):
    _, case, out = toy

    def at_yield(dest):
        _killed(counter)

    done = methods.run_case(two_stage(native_at_yield=at_yield), case, out)
    assert done == ["m_rigid"]
    assert run_json(out, "m", case)["cause"] == "writer_oom"


def test_a_kill_while_the_benchmark_moves_the_nuclei_is_the_benchmarks(toy, counter):
    _, case, out = toy

    def slow(xy):
        _killed(counter)
        return np.asarray(xy, dtype=float)

    def register(case, work, opts):
        yield "k", PointMap(slow, MOV_TO_REF), {}

    methods.run_case(fake("k", ["k"], register), case, out)
    assert run_json(out, "k", case)["cause"] == "benchmark_oom"


def test_a_writer_that_never_returns_is_ended_by_the_watcher(tmp_path):
    """Native code waiting on a dead worker does not come back to the interpreter: after
    the grace period the watcher records the row and ends the process."""
    script = tmp_path / "stuck.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import signal, sys, time
            from pathlib import Path
            sys.path.insert(0, {str(REPO)!r})
            from benchmarks.regbench import methods, resources
            from benchmarks.tests.regbench.test_failures import two_stage
            from benchmarks.tests.regbench.test_pipeline import toy_case

            tmp = Path({str(tmp_path)!r})
            counter = tmp / "memory.events"
            counter.write_text("oom_kill 0\\n")
            resources.oom_events_path = lambda: counter
            methods.OOM_POLL_S, methods.OOM_GRACE_S = 0.05, 0.5

            def write_native(state, case_id, dest):
                signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])
                counter.write_text("oom_kill 1\\n")
                time.sleep(120)

            case = toy_case(tmp / "cases", "P1_r1", "P1")
            methods.run_case(two_stage(write_native=write_native), case, tmp / "out")
            """
        )
    )
    t0 = time.perf_counter()
    done = subprocess.run(
        [sys.executable, str(script)], capture_output=True, timeout=90
    )
    assert time.perf_counter() - t0 < 60
    assert done.returncode == methods.OOM_EXIT_CODE, done.stderr.decode()[-2000:]
    path = tmp_path / "out" / "m" / "multiplex" / "P1_r1" / "run.json"
    run = json.loads(path.read_text())
    assert run["ok"] is False and run["cause"] == "writer_oom"
    assert "did not return" in run["error"]


# ── the watcher's own record (review item 15) ─────────────────────────────────
def test_run_json_says_whether_the_oom_counter_was_readable(toy, counter, monkeypatch):
    _, case, out = toy
    methods.run_case(two_stage(), case, out)
    watch = run_json(out, "m", case)["oom_watch"]
    assert watch == {"readable": True, "started": True, "path": str(counter)}

    monkeypatch.setattr(resources, "oom_events_path", lambda: None)
    methods.run_case(two_stage(name="n"), case, out)
    for variant in ("n_rigid", "n"):
        assert run_json(out, variant, case)["oom_watch"] == {
            "readable": False,
            "started": False,
            "path": "",
        }


def test_the_cost_header_counts_the_runs_without_a_watcher():
    rows = [
        {"cause": "", "oom_watch": "yes", "adapter": "stare", "unit": "pair"},
        {"cause": "timeout", "oom_watch": "no", "adapter": "stare", "unit": "pair"},
        {"cause": "", "oom_watch": "no", "adapter": "valis", "unit": "patient"},
        {"cause": "not_run", "oom_watch": "", "adapter": "", "unit": ""},
    ]
    (watch, reused) = cost.header_lines(rows)
    assert "2 of 3 results ran WITHOUT one" in watch
    assert "stare 1, valis 1" in watch and "may be a memory death" in watch
    assert reused.startswith("reused: 0 of 3 results")
    assert "unknown" in cost.header_lines([{"cause": "", "oom_watch": ""}])[0]
    rows[0].update(time_limit="1-00:00:00", reused=True)
    rows[2].update(time_limit="3-00:00:00")
    lines = cost.header_lines(rows)
    assert lines[1].startswith("reused: 1 of 3 results")
    assert "pair 1-00:00:00; patient 3-00:00:00" in lines[2]
    assert "more time per round" in lines[2]


# ── stale records (review item 13) ────────────────────────────────────────────
def test_an_attempt_removes_what_an_earlier_one_left_for_a_gate(toy):
    _, case, out = toy
    d = result_dir(out, "m", case)
    d.mkdir(parents=True)
    (d / "realcheck.json").write_text(json.dumps({"flag": False, "status": "ok"}))

    def register(case, work, opts):
        assert not (d / "realcheck.json").exists(), "removed before the method starts"
        raise RuntimeError("this attempt fails")
        yield  # pragma: no cover

    methods.run_case(fake("m", ["m"], register), case, out)
    assert not (d / "realcheck.json").exists()


def test_the_scorer_takes_no_result_of_another_submission(toy, monkeypatch):
    """A blocked run's old folders must not be scored as if it had run."""
    _, case, out = toy
    monkeypatch.setenv("REGBENCH_SUBMISSION", "old")
    methods.run_case(two_stage(), case, out)
    run = run_json(out, "m", case)
    assert run["submissions"] == ["old"]
    assert score._load_run(out, "m", case)[3] == ""
    monkeypatch.setenv(score.SUBMISSION_ENV, "new")
    got = score._load_run(out, "m", case)
    assert got[0] == {} and got[1] is None and got[3] == "not_run"
    assert "another submission" in got[2]
    monkeypatch.setenv(score.SUBMISSION_ENV, "old")
    assert score._load_run(out, "m", case)[3] == ""


def test_selfcheck_removes_the_old_record_before_it_runs(tmp_path, monkeypatch):
    """The record is written last. If the old one stayed while the new attempt ran, a
    killed attempt would leave the OLD verdict for the gate."""
    from benchmarks.regbench import selfcheck

    work = tmp_path / "w"
    work.mkdir()
    (work / "selfcheck.json").write_text(
        json.dumps({"status": "ok", "attempts": [{"submission": "old", "attempt": 1}]})
    )
    seen = {}

    def run_pair(*args, **kwargs):
        seen["old_record_there"] = (work / "selfcheck.json").exists()
        return {"status": "ok", "variants": {"initial": {"status": "ok"}}}

    monkeypatch.setattr(selfcheck, "run_pair", run_pair)
    monkeypatch.setenv("REGBENCH_SUBMISSION", "new")
    record = selfcheck.run(
        "initial", work, sizes=[((64, 64), (64, 64))], in_process=True
    )
    assert seen["old_record_there"] is False
    assert record["submission"] == "new"
    # another launch's attempts are not this one's
    assert [a["submission"] for a in record["attempts"]] == ["new"]
    assert json.loads((work / "selfcheck.json").read_text())["submission"] == "new"


# ── what is not the method's is counted apart ─────────────────────────────────
def _res(method, cause, launch, ok=False, **more):
    return {
        "dataset": "multiplex", "group": "P1", "method": method, "ok": ok,
        "cause": cause, "launch_id": launch, "unit": "pair", **more,
    }  # fmt: skip


def test_the_failure_table_counts_infrastructure_apart():
    rows = [
        _res("a", "", "1", ok=True),
        _res("a", "oom", "2"),
        _res("a", "writer", "3"),
        _res("a", "infrastructure", "4"),
        _res("a", "benchmark_oom", "5"),
        _res("a", "not_run", ""),
    ]
    (rec,) = score.failures(rows)
    assert rec["launches_failed"] == 4
    assert rec["launches_failed_method"] == 2 and rec["launches_lost_not_method"] == 2
    assert rec["launches_failed_infrastructure"] == 1
    assert rec["launches_failed_writer"] == 1 and rec["rounds_lost_not_method"] == 2
    text = score.summary_markdown([], [rec], [])
    assert "2 round(s) are lost for a reason that is not the method's" in text
    clean = score.failures(rows[:3])
    assert "No round is lost for a reason" in score.summary_markdown([], clean, [])
    assert set(resources.CAUSES) == set(score.CAUSES)
    assert set(resources.NOT_THE_METHOD) == set(score.NOT_THE_METHOD)


def test_a_killed_attempt_is_billed_by_the_scheduler_or_counted_unknown():
    """Review item 11: the placeholder carries no cost, so wasted_* was zero for exactly
    the attempts a scheduler ends."""
    run = {
        "ok": True, "register_s": 10.0, "resources": {"wall_s": 10.0, "cpu_s": 30.0},
        "previous_attempts": [
            {"cause": "oom", "wall_s": 100.0, "cpu_s": 360.0},
            {"cause": "oom", "wall_s": None, "cpu_s": None,
             "scheduler_elapsed_s": 7200.0, "scheduler_cpu_s": 3600.0},
            {"cause": "killed", "wall_s": None, "cpu_s": None},
        ],
    }  # fmt: skip
    row = score.resource_row(run)
    assert row["wasted_attempts"] == 3
    assert row["wasted_wall_s"] == 7300.0 and row["wasted_cpu_h"] == 1.1
    assert row["wasted_from_scheduler"] == 1 and row["wasted_unknown"] == 1
    # the result's own last attempt, killed at the last rung and settled afterwards
    last = {"ok": False, "cause": "oom", "launch_id": "9_1",
            "scheduler": {"state": "OUT_OF_MEMORY", "elapsed_s": 500.0, "cpu_s": 900.0}}  # fmt: skip
    row = score.resource_row(last)
    assert row["register_s"] == 500.0 and row["wasted_from_scheduler"] == 1
    lost = score.resource_row({"ok": False, "cause": "killed", "launch_id": "9_2"})
    assert lost["wasted_unknown"] == 1 and np.isnan(lost["register_s"])
    assert score.resource_row({"cause": "not_run"})["wasted_unknown"] == 0


def test_the_file_says_how_many_channels_and_the_driver_how_many_the_patient_has(toy):
    root, case, out = toy
    assert methods.patient_channels([case]) is None  # a case without channel lists
    case.extra.update(channels_ref=["DAPI", "a"], channels_mov=["DAPI", "b", "c"])
    other = toy_case(root, "P1_r2", "P1")
    other.extra.update(channels_ref=["DAPI", "a"], channels_mov=["DAPI", "d"])
    assert methods.patient_channels([case]) == 5
    assert methods.patient_channels([case, other]) == 7  # the reference once
    methods.run_case(two_stage(native_at_yield=image), case, out)
    written = run_json(out, "m", case)["native"]["written"]
    assert written["patient_channels"] == 5 and written["channels"] == 1
    assert cost.complete_image(written) is False


def test_sigint_from_anyone_else_is_not_swallowed(toy):
    """Only the watcher's interrupt is turned into a cause."""
    _, case, out = toy

    def write_native(state, case_id, dest):
        signal.raise_signal(signal.SIGINT)

    with pytest.raises(KeyboardInterrupt):
        methods.run_case(two_stage(write_native=write_native), case, out)
    assert run_json(out, "m", case)["ok"] is False  # the placeholder of the writer


@pytest.mark.parametrize(
    "text, cause",
    [
        ("Unable to allocate 64.0 GiB for an array with shape (9, 9)", "oom"),
        ("OSError: [Errno 12] Cannot allocate memory", "oom"),
        ("terminate called after throwing an instance of 'std::bad_alloc'", "oom"),
        ("java.lang.OutOfMemoryError: GC overhead limit exceeded", "oom"),
        ("RuntimeError: CUDA out of memory. Tried to allocate 2 GiB", "gpu_oom"),
        ("torch.cuda.OutOfMemoryError: CUDA out of memory.", "gpu_oom"),
        ("ValueError: need at least one array to concatenate", "exception"),
        ("", "exception"),
        (None, "exception"),
    ],
)
def test_a_captured_error_text_names_its_cause(text, cause):
    """For a package that prints its exception and carries on: only the text is left."""
    assert resources.classify_text(text) == cause
    if text and cause != "gpu_oom":
        assert resources.classify_error(RuntimeError(text)) == cause
