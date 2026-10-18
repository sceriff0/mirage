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

from benchmarks.regbench import (
    cost,
    failure_log,
    ladder,
    losses,
    methods,
    resources,
    score,
)
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
def test_the_benchmarks_own_out_of_memory_is_not_the_methods(toy, monkeypatch):
    """Review item 12: the benchmark's own lines around the map, clock stopped."""
    _, case, out = toy

    def register(case, work, opts):
        yield "b", true_map(), {}

    def full(path, **arrays):
        raise MemoryError("Unable to allocate 40. GiB for an array")

    monkeypatch.setattr(methods, "write_npz", full)
    method = fake("b", ["b"], register)
    assert methods.run_case(method, case, out) == []
    run = run_json(out, "b", case)
    assert run["cause"] == "benchmark_oom" and "benchmark's own step" in run["error"]
    assert run["cause"] in resources.NOT_THE_METHOD
    # more memory is still the remedy
    assert ladder.unit_retry(ladder.unit_paths(out, method, None, [case]))[0]


def test_the_benchmarks_own_error_is_not_the_methods(toy, monkeypatch):
    _, case, out = toy

    def register(case, work, opts):
        yield "b", true_map(), {}

    def broken(path, **arrays):
        raise OSError("No space left on device")

    monkeypatch.setattr(methods, "write_npz", broken)
    assert methods.run_case(fake("b", ["b"], register), case, out) == []
    run = run_json(out, "b", case)
    assert run["cause"] == "benchmark" and "benchmark's own step" in run["error"]
    assert failure_log.classify(run["cause"]) == failure_log.BENCHMARK


# ── the method's own point mapping is the method's ────────────────────────────
def _bad_shape(xy):
    return np.zeros((len(xy), 3))


def _raises(xy):
    raise ValueError("no transform for this patch")


def _not_numbers(xy):
    return [["a", "b"]] * len(xy)


@pytest.mark.parametrize(
    "mapping, said",
    [
        (_raises, "raised ValueError: no transform for this patch"),
        (_bad_shape, "returned an array of shape"),
        (_not_numbers, "not numbers"),
    ],
)
def test_a_mapping_that_fails_is_the_methods_failure(toy, mapping, said):
    """An error raised inside the method's mapping function, or a result that is not
    one X, Y pair per point, fails that row as the method's; the run goes on."""
    _, case, out = toy

    def register(case, work, opts):
        yield "p_first", PointMap(mapping, MOV_TO_REF), {}
        yield "p", true_map(), {}

    method = fake("p", ["p_first", "p"], register)
    assert methods.run_case(method, case, out) == ["p"]
    run = run_json(out, "p_first", case)
    assert run["cause"] == "exception" and said in run["error"]
    assert "the method's point mapping" in run["error"]
    assert "benchmark's own step" not in run["error"]
    assert failure_log.classify(run["cause"]) == failure_log.METHOD
    assert run["cause"] not in resources.NOT_THE_METHOD
    assert not ladder.unit_retry(ladder.unit_paths(out, method, None, [case]))[0]


def test_a_mapping_that_runs_out_of_memory_is_the_methods_oom(toy):
    _, case, out = toy

    def hungry(xy):
        raise MemoryError("Unable to allocate 40. GiB for an array")

    def register(case, work, opts):
        yield "h_first", PointMap(hungry, MOV_TO_REF), {}
        yield "h", true_map(), {}  # pragma: no cover

    method = fake("h", ["h_first", "h"], register)
    assert methods.run_case(method, case, out) == []
    for variant in ("h_first", "h"):  # the run ends there, and the unit goes up
        run = run_json(out, variant, case)
        assert run["cause"] == "oom" and run["cause"] not in resources.NOT_THE_METHOD
    assert "point mapping" in run_json(out, "h_first", case)["error"]
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


def test_a_kill_inside_the_methods_mapping_is_the_methods(toy, counter):
    _, case, out = toy

    def slow(xy):
        _killed(counter)
        return np.asarray(xy, dtype=float)

    def register(case, work, opts):
        yield "k", PointMap(slow, MOV_TO_REF), {}

    methods.run_case(fake("k", ["k"], register), case, out)
    assert run_json(out, "k", case)["cause"] == "oom"


def test_a_kill_in_the_benchmarks_own_lines_is_the_benchmarks(
    toy, counter, monkeypatch
):
    _, case, out = toy

    def register(case, work, opts):
        yield "k", true_map(), {}

    monkeypatch.setattr(methods, "write_npz", lambda path, **a: _killed(counter))
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
    (watch, threads, reused) = cost.header_lines(rows)
    assert threads.startswith("thread variables: the benchmark sets none.")
    assert "2 of 3 results ran WITHOUT one" in watch
    assert "stare 1, valis 1" in watch and "may be a memory death" in watch
    assert reused.startswith("reused: 0 of 3 results")
    assert "unknown" in cost.header_lines([{"cause": "", "oom_watch": ""}])[0]
    rows[0].update(time_limit="1-00:00:00", reused=True)
    rows[2].update(time_limit="3-00:00:00")
    lines = cost.header_lines(rows)
    assert lines[2].startswith("reused: 1 of 3 results")
    assert "pair 1-00:00:00; patient 3-00:00:00" in lines[3]
    assert "more time per round" in lines[3]


# ── stale records (review item 13) ────────────────────────────────────────────
def test_an_attempt_removes_what_an_earlier_one_left_for_the_verdict(toy):
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
    """A run's old folders of another submission must not be scored as if it had run."""
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
    killed attempt would leave the OLD verdict for the scorer to report."""
    from benchmarks.regbench import selfcheck

    work = tmp_path / "w"
    work.mkdir()
    (work / "selfcheck.json").write_text(
        json.dumps({"status": "ok", "attempts": [{"submission": "old", "attempt": 1}]})
    )
    seen = {}

    def run_case(*args, **kwargs):  # what `run` calls per case since the patient cases
        seen["old_record_there"] = (work / "selfcheck.json").exists()
        return [{"status": "ok", "variants": {"initial": {"status": "ok"}}}]

    monkeypatch.setattr(selfcheck, "run_case", run_case)
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
    # resources.py is frozen: score's extra causes are Task 28's (regbench.losses)
    assert set(score.CAUSES) - set(resources.CAUSES) == {
        "capped", *losses.NOT_THE_METHOD_EXTRA,
    } - {"score_error"}
    assert set(score.NOT_THE_METHOD) == {
        *resources.NOT_THE_METHOD, *losses.NOT_THE_METHOD_EXTRA,
    }


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


# ── CPU time: the method's, the writer's and the benchmark's, each once ───────
BUSY = (
    "import time\n"
    "t = time.process_time()\n"
    "while time.process_time() - t < {cpu}: pass\n"
    "print(time.process_time())\n"
)


def busy_children(n, cpu):
    """``n`` short children one after another: the CPU time they report, summed, plus
    what THIS process used meanwhile (starting them, and whatever its other threads
    did): the counter is the process's, so both are in the stretch."""
    import subprocess

    total, own = 0.0, time.process_time()
    for _ in range(n):
        done = subprocess.run(
            [sys.executable, "-S", "-c", BUSY.format(cpu=cpu)],
            check=True,
            capture_output=True,
            text=True,
        )
        total += float(done.stdout)
    OWN["cpu_s"] += time.process_time() - own
    return total


OWN = {"cpu_s": 0.0}  # this process's own CPU inside `busy_children`, by test


@pytest.mark.parametrize("kind", ["write_native", "at_the_yield"])
def test_the_writers_cpu_is_off_the_methods_bill_and_in_its_own_figure(toy, kind):
    import tifffile  # noqa: F401  (its first import is CPU of whoever triggers it)

    from benchmarks.regbench import billing, score

    _, case, out = toy
    used, OWN["cpu_s"] = {}, 0.0

    def write(dest):
        used["writer"] = busy_children(4, 0.15)
        return image(dest)

    def register(case, work, opts):
        yield "m_rigid", true_map(), {}
        used["method"] = busy_children(4, 0.05)
        info = {"_state": {}}
        if kind == "at_the_yield":
            mon = opts["monitor"]
            mon.pause()
            try:
                info["native"] = write(opts["native_dir"](case.case_id, "m"))
                info["write_s"] = 1.0
            finally:
                mon.resume()
        yield "m", true_map(), info

    extra = {}
    if kind == "write_native":
        extra["write_native"] = lambda state, case_id, dest: write(dest)
    methods.run_case(fake("m", ["m_rigid", "m"], register, **extra), case, out)
    run = run_json(out, "m", case)
    res = run["resources"]
    assert res["cpu_instrument"] in resources.CPU_INSTRUMENTS
    assert res["cpu_instrument"] != resources.POLLED
    # the method's children in full, and none of the writer's 0.6 s
    assert res["cpu_s"] >= 0.99 * used["method"]
    assert res["cpu_s"] < used["method"] + OWN["cpu_s"] + 0.3
    # the writer's, in its own figure
    assert run["write_cpu_s"] >= 0.99 * used["writer"]
    assert run["write_cpu_s"] < used["writer"] + OWN["cpu_s"] + 0.3
    assert run["warp_cpu_s"] >= 0.0  # the benchmark's own point mapping, apart
    # the whole launch holds all three
    assert res["launch_cpu_s"] >= 0.99 * (used["method"] + used["writer"])
    row = score.resource_row(run)
    assert row["cpu_instrument"] == res["cpu_instrument"]
    assert row["write_cpu_h_used"] == pytest.approx(run["write_cpu_s"] / 3600.0)
    if kind == "write_native":
        assert run["write_cpu_from"] == billing.WRITE_CPU_CALL
        assert row["billing_note"] == ""
        # the first result was yielded before any of it
        assert "write_cpu_s" not in run_json(out, "m_rigid", case)
    else:
        assert run["write_cpu_from"] == billing.WRITE_CPU_STRETCH
        assert res["paused_cpu_s"] >= 0.99 * used["writer"]
        assert billing.WRITE_CPU_IS_THE_STRETCH in row["billing_note"]


def test_one_writer_call_for_a_patient_is_shared_between_its_rounds(tmp_path):
    """A per-patient writer draws every round in one stretch (VALIS, ASHLAR): its CPU
    is shared between the rows as their ``write_s`` is, and adds up to the stretch."""
    root, out = tmp_path / "cases", tmp_path / "out"
    cases = [toy_case(root, "P1_r1", "P1"), toy_case(root, "P1_r2", "P1")]
    used = {}

    def register_group(cases, work, opts):
        mon = opts["monitor"]
        mon.pause()
        try:
            OWN["cpu_s"] = 0.0
            used["writer"] = busy_children(4, 0.1)
            used["own"] = OWN["cpu_s"]
            ready = [
                (
                    c.case_id,
                    "pp",
                    true_map(),
                    {
                        "native": image(opts["native_dir"](c.case_id, "pp")),
                        "write_s": share,
                    },
                )
                for c, share in zip(cases, (1.0, 3.0))
            ]
        finally:
            mon.resume()
        yield from ready

    def benchmark(case, variant, pmap, info):  # the driver's own stretch, at each yield
        used["benchmark"] = used.get("benchmark", 0.0) + busy_children(1, 0.2)

    method = fake("pp", ["pp"], register_group=register_group)
    methods.run_group(method, cases, out, on_variant=benchmark)
    first, second = (run_json(out, "pp", c) for c in cases)
    assert second["write_cpu_s"] == pytest.approx(3 * first["write_cpu_s"])
    total = first["write_cpu_s"] + second["write_cpu_s"]
    # the writer's stretch, and none of the 0.4 s the benchmark used at the two yields
    assert total >= 0.99 * used["writer"]
    assert total < used["writer"] + used["own"] + 0.3
    # neither is the method's
    assert first["resources"]["cpu_s"] < 0.2 and second["resources"]["cpu_s"] < 0.2
    # every stretch taken off the clock before the second result: the writer's and
    # the benchmark's at the first yield
    assert second["resources"]["paused_cpu_s"] >= 0.99 * (
        used["writer"] + used["benchmark"] / 2
    )


# ── every run's failures beside its scores (C5) ───────────────────────────────
def _run_summary(method, **more):
    return {
        "dataset": "multiplex", "method": method, "n_patients": 1, "n_cases": 7,
        "n_ok": 3, **more,
    }  # fmt: skip


def test_the_failure_columns_count_rounds_by_owner_and_name_the_top_cause():
    rows = [_res("a", "", str(i), ok=True) for i in range(3)] + [
        _res("a", "oom", "4"),
        _res("a", "identical_overlap", "5"),
        _res("a", "not_submitted", ""),
        _res("a", "not_run", ""),
    ]
    (rec,) = score.failures(rows)
    assert rec["n_failed_method"] == 1
    assert rec["n_failed_benchmark"] == 1
    assert rec["n_not_run_or_lost"] == 2
    # a tie of one each: the earlier in CAUSES
    assert rec["top_cause"] == "oom"
    assert (rec["n_fallback_stage_positions"], rec["n_cycle_fully_modelled"]) == (0, 0)


def test_a_not_run_row_is_never_a_method_failure():
    rows = [_res("a", c, "") for c in ("not_run", "not_submitted", "infrastructure")]
    (rec,) = score.failures(rows)
    assert rec["n_failed_method"] == 0 and rec["n_failed_benchmark"] == 0
    assert rec["n_not_run_or_lost"] == 3
    assert "n_not_run_or_blocked" not in rec  # there is no blocked run any more
    assert score.failures([_res("a", "", "1", ok=True)])[0]["top_cause"] == ""


def test_ashlar_flags_are_counted_and_a_failed_row_without_them_counts_zero():
    rows = [
        _res("ashlar", "", "1", ok=True, fallback_stage_positions=True,
             cycle_fully_modelled=False),
        _res("ashlar", "", "1", ok=True, fallback_stage_positions=True,
             cycle_fully_modelled=True),
        _res("ashlar", "identical_overlap", "2"),  # a failed row carries no flag key
    ]  # fmt: skip
    (rec,) = score.failures(rows)
    assert rec["n_fallback_stage_positions"] == 2
    assert rec["n_cycle_fully_modelled"] == 1
    assert rec["n_failed_benchmark"] == 1 and rec["top_cause"] == "identical_overlap"


def test_resource_row_carries_the_flags_and_an_absent_one_is_none():
    got = score.resource_row(
        {
            "ok": True,
            "info": {"fallback_stage_positions": True, "cycle_fully_modelled": False},
        }
    )
    assert got["fallback_stage_positions"] is True
    assert got["cycle_fully_modelled"] is False
    assert (
        score.resource_row({"ok": False, "cause": "identical_overlap"})[
            "fallback_stage_positions"
        ]
        is None
    )


def test_the_headline_table_carries_the_failure_columns_ashlar_cells_only_on_ashlar():
    rows = (
        [_res("a", "", "1", ok=True), _res("a", "oom", "2"),
         _res("a", "identical_overlap", "3"), _res("a", "not_submitted", ""),
         _res("a", "not_run", ""),
         _res("ashlar", "", "9", ok=True, fallback_stage_positions=True,
              cycle_fully_modelled=True)]
    )  # fmt: skip
    from benchmarks.tests.regbench.test_score_units import by, cell

    cells = [cell(m, f"P{k}_r1", f"P{k}") for m in ("a", "ashlar") for k in range(2)]
    summary = score.summarise_cells(cells)
    a = by(summary, method="a", subset="all")
    b = by(summary, method="ashlar", subset="all")
    lines = score._summary_table(
        [a, b], ["a", "ashlar"], score.run_failures(rows)
    ).splitlines()
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    cells = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]
    assert all(len(c) == len(header) for c in cells)
    col = lambda row, name: row[header.index(name)]  # noqa: E731
    got = [col(cells[0], n) for n in (
        "failed (method)", "failed (benchmark)", "not run / lost", "top cause",
        "fallback", "fully modelled")]  # fmt: skip
    assert got == ["1", "1", "2", "oom", "", ""]
    got = [col(cells[1], n) for n in ("failed (method)", "fallback", "fully modelled")]
    assert got == ["0", "1", "1"]
    # without `fails` the table is the old one
    assert "top cause" not in score._summary_table([a], ["a"])


def _cells_of(table):
    lines = table.splitlines()
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    return header, [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]


def test_the_failure_cells_are_blank_on_the_common_subset_and_on_non_rows():
    from benchmarks.tests.regbench.test_score_units import cell

    rows = [_res("a", "oom", "1"), _res("a", "", "2", ok=True)]
    cells = [cell("a", f"P{k}_r1", f"P{k}") for k in range(2)]
    summary = score.summarise_cells(cells)
    run = score.run_failures(rows)
    common = [r for r in summary if r["subset"] == "common"]
    assert common
    header, got = _cells_of(score._summary_table(common, ["a"], run))
    assert got[0][header.index("failed (method)")] == ""
    assert got[0][header.index("top cause")] == ""
    non = [{**r, "method": f"a{score.NON}ashlar", "subset": "all"} for r in summary[:1]]
    header, got = _cells_of(score._summary_table(non, ["a"], run))
    assert got[0][header.index("failed (method)")] == ""
    assert len(got[0]) == len(header)


def test_failures_csv_has_the_new_columns_and_no_is_ashlar():
    (rec,) = score.failures([_res("a", "oom", "1")])
    assert "n_failed_method" in rec and "top_cause" in rec
    assert "is_ashlar" not in rec


def test_the_counting_unit_is_stated_under_the_table():
    from benchmarks.tests.regbench.test_score_units import cell

    summary = score.summarise_cells([cell("a", "P0_r1", "P0")])
    text = score.summary_markdown(
        summary, [], [], run_fails=score.run_failures([_res("a", "oom", "1")])
    )
    assert "count CASES" in text
    assert "count CASES" not in score.summary_markdown(summary, [], [])


# ── pre-freeze review wave 1: F19, F20, F21 ───────────────────────────────────
def _with_slide(case, tmp_path):
    """The toy case naming a moving slide on disk (the toy has none)."""
    import dataclasses

    slide = tmp_path / "slides" / "moving.tif"
    slide.parent.mkdir(parents=True, exist_ok=True)
    slide.write_bytes(b"\x01" * 4096)
    return dataclasses.replace(case, mov_image=str(slide)), slide


def _one_variant(name="g"):
    def register(case, work, opts):
        yield name, true_map(), {}

    return fake(name, [name], register)


def test_unreadable_inputs_leave_an_earlier_ok_result_as_it_was(
    toy, tmp_path, monkeypatch
):
    """F19: a slide not visible from the job says nothing of whether the inputs changed:
    the earlier ok record and its warped.npz stay as they were."""
    _, case, out = toy
    case, slide = _with_slide(case, tmp_path)
    monkeypatch.setenv("REGBENCH_SUBMISSION", "S1")
    assert methods.run_case(_one_variant(), case, out, resume=True) == ["g"]
    d = result_dir(out, "g", case)
    kept = (d / "run.json").read_bytes(), (d / "warped.npz").read_bytes()
    slide.rename(slide.with_suffix(".moved"))
    monkeypatch.setenv("REGBENCH_SUBMISSION", "S2")
    assert methods.run_case(_one_variant(), case, out, resume=True) == []
    assert ((d / "run.json").read_bytes(), (d / "warped.npz").read_bytes()) == kept


def test_unreadable_inputs_fail_a_row_without_an_ok_result_as_infrastructure(
    toy, tmp_path
):
    _, case, out = toy
    case, slide = _with_slide(case, tmp_path)
    slide.unlink()
    assert methods.run_case(_one_variant(), case, out, resume=True) == []
    run = run_json(out, "g", case)
    assert not run["ok"] and run["cause"] == "infrastructure"
    assert "cannot be read" in run["error"]


def test_another_submissions_ok_result_is_set_aside_not_destroyed(
    toy, tmp_path, monkeypatch
):
    """F19: a reuse refused (here the slide's bytes changed) runs the unit again; the
    earlier ok record of another submission is moved to earlier/<n>/, named in the new
    record, and survives a new attempt that fails."""
    _, case, out = toy
    case, slide = _with_slide(case, tmp_path)
    monkeypatch.setenv("REGBENCH_SUBMISSION", "S1")
    methods.run_case(_one_variant(), case, out, resume=True)
    d = result_dir(out, "g", case)
    old = json.loads((d / "run.json").read_text())
    slide.write_bytes(b"\x02" * 4096)

    def broken(case, work, opts):
        raise RuntimeError("this attempt fails")
        yield  # pragma: no cover

    monkeypatch.setenv("REGBENCH_SUBMISSION", "S2")
    methods.run_case(fake("g", ["g"], broken), case, out, resume=True)
    new = run_json(out, "g", case)
    assert not new["ok"] and new["earlier_result"]["ok"] is True
    kept = Path(new["earlier_result"]["kept"])
    assert kept == d / ladder.SET_ASIDE / "1"
    assert json.loads((kept / "run.json").read_text()) == old
    assert (kept / "warped.npz").exists()


def test_the_final_row_of_a_method_with_its_own_writer_is_held_until_it_wrote(toy):
    """F20: between the final yield and the writer, the row on disk is the writer's
    holding record, never ok without an image."""
    _, case, out = toy
    seen = {}

    def register(case, work, opts):
        yield "m_rigid", true_map(), {}
        yield "m", true_map(), {"_state": {}}
        # the generator resumes after the driver handled the yield, before the writer
        run = run_json(out, "m", case)
        seen.update(ok=run["ok"], cause=run["cause"], phase=run.get("phase"))

    mod = fake("m", ["m_rigid", "m"], register, write_native=lambda s, c, d: image(d))
    assert methods.run_case(mod, case, out) == ["m_rigid", "m"]
    assert seen == {"ok": False, "cause": "killed", "phase": "writer"}
    final = run_json(out, "m", case)
    assert final["ok"] is True and final["native"]["files"]
    assert run_json(out, "m_rigid", case)["ok"] is True  # not a writer's row


def test_a_rise_the_main_thread_saw_first_is_never_signalled(counter, monkeypatch):
    """F21: the main thread saw the rise and handles it: the watcher must not send a
    SIGINT for it later, into code without a handler. A LATER rise still is."""
    monkeypatch.setattr(methods, "OOM_POLL_S", 0.3)
    watch = methods._OomWatch(lambda: None).start()
    try:
        counter.write_text("oom 1\noom_kill 1\n")
        assert watch.seen("method") is True and watch.phase == "method"
        try:
            time.sleep(1.0)  # three polls of the watcher
        except KeyboardInterrupt:  # pragma: no cover - the defect
            pytest.fail("the watcher signalled a rise the main thread had handled")
        assert watch.fired is False
        counter.write_text("oom 2\noom_kill 2\n")
        with pytest.raises(KeyboardInterrupt):
            time.sleep(2.0)
        assert watch.fired is True
    finally:
        watch.stop()


def test_a_rise_seen_at_a_yield_leaves_the_rounds_already_done_to_their_writer(
    tmp_path, counter, monkeypatch
):
    """F21: a rise seen at round 2's yield fails round 2 as the method's oom; round 1,
    done before it, is written by the writer, uninterrupted, and stays ok."""
    monkeypatch.setattr(methods, "OOM_POLL_S", 0.3)
    root, out = tmp_path / "cases", tmp_path / "out"
    first, second = toy_case(root, "P1_r1", "P1"), toy_case(root, "P1_r2", "P1")

    def register_group(cases, work, opts):
        yield first.case_id, "g", true_map(), {"_state": {}}
        counter.write_text("oom 1\noom_kill 1\n")
        yield second.case_id, "g", true_map(), {"_state": {}}

    def write_native(state, case_id, dest):
        time.sleep(1.0)  # long enough for the watcher to poll
        return image(dest)

    mod = fake("g", ["g"], register_group=register_group, write_native=write_native)
    methods.run_group(mod, [first, second], out)
    assert run_json(out, "g", second)["cause"] == "oom"
    done = run_json(out, "g", first)
    assert done["ok"] is True and done["native"]["files"]
