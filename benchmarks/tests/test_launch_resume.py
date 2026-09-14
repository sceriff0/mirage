"""An interrupted benchmark can be resumed, and a finished one is not re-run.

scancel of a head job leaves every in-flight run with status '-' in its launch
dir's .nextflow/history (Nextflow writes the OK/ERR only at completion). The
launchers used to refuse any run whose name was already in the history, so the
only way forward after an interruption was ARMS_REPLACE=1 -- a restart from
scratch. Now:

  * a run whose LAST attempt is OK is skipped as DONE (a relaunch is idempotent);
  * an interrupted or failed one is continued under ARMS_RESUME=1 / SWEEP_RESUME=1
    as <name>-r<N> with `-resume <its last session>`, so Nextflow serves every
    cached task from work/ and runs only the rest;
  * without the switch it is refused, naming both switches;
  * every launch pins cleanup_work=false, or a finished base arm would have no
    work/ left for its crosses to resume from.

Behavioural, through the same stub nextflow the launch tests use: it records
its argv (run name, -resume value) and writes a history line with status OK,
which the tests then flip to '-' to simulate an interruption.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from benchmarks.build_arm_plan import build_arm_plan
from benchmarks.tests.test_build_arm_plan import _fake_nextflow, _plan_csv
from benchmarks.tests.test_subset_rerun_equivalence import _launch_cfg

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent


def _interrupt(hist: Path, run_name: str) -> str:
    """Flip `run_name`'s history line to status '-' (what scancel leaves); return its session."""
    lines = hist.read_text().splitlines()
    sid = ""
    for i, ln in enumerate(lines):
        f = ln.split("\t")
        if f[2] == run_name:
            f[1], f[3] = "-", "-"
            sid = f[5]
            lines[i] = "\t".join(f)
    assert sid, f"{run_name} not in {hist}"
    hist.write_text("\n".join(lines) + "\n")
    return sid


@pytest.fixture
def arms(tmp_path):
    plan = build_arm_plan(_launch_cfg())
    root = tmp_path / "arm_results"
    root.mkdir()
    sheet = tmp_path / "input.csv"
    sheet.write_text(
        "patient_id,path_to_file,is_reference,channels\nP1,/x/a.tif,true,DAPI\n"
    )
    log = tmp_path / "launches.log"
    _fake_nextflow(tmp_path / "bin", log)
    env = dict(
        os.environ,
        PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        ARMS_CONCURRENCY="4",
    )
    for k in ("ARMS_REPLACE", "ARMS_RESUME"):
        env.pop(k, None)
    plan_csv = tmp_path / "plan.csv"
    plan_csv.write_text(_plan_csv(plan))

    def run(**extra_env):
        before = len(log.read_text().splitlines()) if log.exists() else 0
        r = subprocess.run(
            ["bash", str(BENCH / "run_arms.sh"), str(plan_csv), str(sheet), str(root)],
            env=dict(env, **extra_env),
            capture_output=True,
            text=True,
            timeout=300,
        )
        new = (
            [ln.split("|") for ln in log.read_text().splitlines()[before:]]
            if log.exists()
            else []
        )
        launches = {
            ln[1]: ln[2] for ln in new if ln[0] != "LOCKFAIL"
        }  # run name -> -resume value
        return r, launches

    r, launches = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert set(launches) == {f"arms-{p['run_id']}" for p in plan}
    return plan, root, run


def test_a_finished_plan_relaunched_with_resume_runs_nothing(arms):
    plan, root, run = arms
    r, launches = run(ARMS_RESUME="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert launches == {}
    assert r.stdout.count("DONE:") == len(plan)


def test_an_interrupted_base_arm_is_refused_without_the_switch_and_resumed_with_it(
    arms,
):
    plan, root, run = arms
    base = "valis_high_micro2"
    sid = _interrupt(root / ".launch" / base / ".nextflow" / "history", f"arms-{base}")

    r, launches = run()
    assert launches == {}, "an interrupted run was relaunched without ARMS_RESUME"
    assert "ARMS_RESUME=1" in r.stderr and "ARMS_REPLACE=1" in r.stderr
    assert f"[{base}] SKIP" in r.stderr

    r, launches = run(ARMS_RESUME="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert launches == {f"arms-{base}-r2": sid}, launches
    assert f"[{base}] RESUME" in r.stdout
    # the other arms are done and stay done
    assert r.stdout.count("DONE:") == len(plan) - 1


def test_a_twice_interrupted_arm_counts_its_attempts(arms):
    plan, root, run = arms
    base = "tiled_low_gate1"
    hist = root / ".launch" / base / ".nextflow" / "history"
    _interrupt(hist, f"arms-{base}")
    r, launches = run(ARMS_RESUME="1")
    assert set(launches) == {f"arms-{base}-r2"}
    sid2 = _interrupt(hist, f"arms-{base}-r2")
    r, launches = run(ARMS_RESUME="1")
    assert launches == {f"arms-{base}-r3": sid2}, launches


def test_an_interrupted_cross_resumes_its_own_session_not_its_base(arms):
    plan, root, run = arms
    base, cross = "valis_high_micro2", "valis_high_micro2_segstardist"
    hist = root / ".launch" / base / ".nextflow" / "history"
    cross_sid = _interrupt(hist, f"arms-{cross}")
    r, launches = run(ARMS_RESUME="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert launches == {f"arms-{cross}-r2": cross_sid}, launches


def test_a_resumed_base_arms_crosses_resume_its_latest_session(arms):
    """After a base arm is resumed, a NEW cross of it must resume the base's latest
    (completed) session, not the interrupted first one."""
    plan, root, run = arms
    base = "valis_high_micro2"
    hist = root / ".launch" / base / ".nextflow" / "history"
    _interrupt(hist, f"arms-{base}")
    r, launches = run(ARMS_RESUME="1")
    assert set(launches) == {f"arms-{base}-r2"}
    # now the base's last attempt is arms-<base>-r2 (OK); free one cross so it launches again
    lines = [
        ln
        for ln in hist.read_text().splitlines()
        if ln.split("\t")[2] != f"arms-{base}_pairmutual_nn"
    ]
    hist.write_text("\n".join(lines) + "\n")
    latest = [
        ln.split("\t")[5] for ln in lines if ln.split("\t")[2] == f"arms-{base}-r2"
    ][0]
    r, launches = run(ARMS_RESUME="1")
    assert launches == {f"arms-{base}_pairmutual_nn": latest}, launches


def test_replace_still_recognises_a_resumed_cross_by_its_base_name(arms):
    plan, root, run = arms
    base, cross = "valis_high_micro2", "valis_high_micro2_segstardist"
    hist = root / ".launch" / base / ".nextflow" / "history"
    _interrupt(hist, f"arms-{cross}")
    run(ARMS_RESUME="1")  # leaves arms-<cross>-r2 in the base's history
    # a replace of the base with its crosses in the plan must not be refused on the -r2 name
    r, launches = run(ARMS_REPLACE="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "refused" not in r.stderr
    assert f"arms-{base}" in launches and f"arms-{cross}" in launches


def test_a_resumed_arm_keeps_its_params_file_and_only_pins_cleanup_work(arms):
    """The interrupted attempt's params file is reused -- an arms.yaml edit made after the
    launch must not change what a half-finished arm IS -- with one hash-neutral change:
    cleanup_work=false (no process script reads it; tests/test_resume_param_hash_neutrality.py),
    so an arm launched before the launchers pinned it keeps work/ for its crosses."""
    plan, root, run = arms
    base = "valis_high_micro2"
    params = root / ".launch" / base / f"params.{base}.json"
    marked = json.loads(params.read_text())
    marked.pop("cleanup_work", None)  # as a launch from before the pin wrote it
    marked["__marker__"] = "from the interrupted attempt"
    params.write_text(json.dumps(marked))
    _interrupt(root / ".launch" / base / ".nextflow" / "history", f"arms-{base}")
    r, launches = run(ARMS_RESUME="1")
    assert set(launches) == {f"arms-{base}-r2"}
    assert json.loads(params.read_text()) == {**marked, "cleanup_work": False}
    assert "cleanup_work pinned false" in r.stdout


def test_resume_params_regenerate_replaces_the_file_and_says_what_it_costs(arms):
    plan, root, run = arms
    base = "valis_high_micro2"
    params = root / ".launch" / base / f"params.{base}.json"
    marked = json.loads(params.read_text())
    marked.pop("cleanup_work", None)  # as a launch from before the pin wrote it
    marked["__marker__"] = "old"
    params.write_text(json.dumps(marked))
    _interrupt(root / ".launch" / base / ".nextflow" / "history", f"arms-{base}")
    r, launches = run(ARMS_RESUME="1", ARMS_RESUME_PARAMS="regenerate")
    assert set(launches) == {f"arms-{base}-r2"}
    fresh = json.loads(params.read_text())
    assert "__marker__" not in fresh and fresh.get("cleanup_work") is False
    assert "params REGENERATED" in r.stdout
    assert "re-run only where a param they read changed" in r.stdout


def test_every_arm_launch_keeps_its_work_directory(arms):
    plan, root, run = arms
    for p in plan:
        if p["arm_kind"] == "external":
            continue
        d = root / ".launch" / (p["resume_run"] or p["run_id"])
        params = json.loads((d / f"params.{p['run_id']}.json").read_text())
        assert params.get("cleanup_work") is False, (p["run_id"], params)


# --- the sweep launcher ------------------------------------------------------------
@pytest.fixture
def sweep(tmp_path):
    log = tmp_path / "launches.log"
    _fake_nextflow(tmp_path / "bin", log)
    matrix = tmp_path / "matrix_manifest.csv"
    (tmp_path / "px256_ch2.ome.tif").write_bytes(b"")
    (tmp_path / "mov.ome.tif").write_bytes(b"")
    matrix.write_text(
        f"cell_id,path,moving_paths\npx256_ch2,{tmp_path / 'px256_ch2.ome.tif'},{tmp_path / 'mov.ome.tif'}\n"
    )
    plan = tmp_path / "plan.csv"
    plan.write_text(
        "run_id,varied_axis,config_id,rep,target_px,n_channels,n_register_images,registration_method\n"
        "run0000,baseline,cfg000,0,256,2,2,valis\n"
    )
    root = tmp_path / "bench_results"
    env = dict(
        os.environ,
        PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        SWEEP_PROFILE="test",
    )
    for k in ("SWEEP_REPLACE", "SWEEP_RESUME"):
        env.pop(k, None)

    def run(**extra_env):
        before = len(log.read_text().splitlines()) if log.exists() else 0
        r = subprocess.run(
            ["bash", str(BENCH / "run_sweep.sh"), str(plan), str(matrix), str(root)],
            env=dict(env, **extra_env),
            capture_output=True,
            text=True,
            timeout=300,
            cwd=tmp_path,
        )
        new = (
            [ln.split("|") for ln in log.read_text().splitlines()[before:]]
            if log.exists()
            else []
        )
        return r, {ln[1]: ln[2] for ln in new if ln[0] != "LOCKFAIL"}

    r, launches = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert set(launches) == {"bench_run0000"}, (launches, r.stdout, r.stderr)
    return root, run


def test_sweep_skips_finished_runs_and_resumes_interrupted_ones(sweep):
    root, run = sweep
    r, launches = run(SWEEP_RESUME="1")
    assert launches == {} and "DONE:" in r.stdout
    sid = _interrupt(root / "run0000" / ".nextflow" / "history", "bench_run0000")
    r, launches = run()
    assert launches == {} and "SWEEP_RESUME=1" in r.stderr
    params_file = root / "run0000" / "params.json"
    marked = json.loads(params_file.read_text())
    assert marked.get("cleanup_work") is False
    marked.pop("cleanup_work")  # as a launch from before the pin wrote it
    marked["__marker__"] = "from the interrupted attempt"
    params_file.write_text(json.dumps(marked))
    r, launches = run(SWEEP_RESUME="1")
    assert launches == {"bench_run0000-r2": sid}, (launches, r.stderr)
    assert json.loads(params_file.read_text()) == {**marked, "cleanup_work": False}
    assert "cleanup_work pinned false" in r.stdout


def test_a_cross_never_starts_on_an_unfinished_base(arms):
    """Measured on the cluster 2026-09-14 (job 6812701): with its base arms refused as
    interrupted, the QC pass still launched their crosses, each resuming a half-finished
    session and so re-running the registration. A cross now waits for its base."""
    plan, root, run = arms
    base, cross = "valis_high_micro2", "valis_high_micro2_segcellsam"
    hist = root / ".launch" / base / ".nextflow" / "history"
    _interrupt(hist, f"arms-{base}")
    kept = [
        ln
        for ln in hist.read_text().splitlines()
        if ln.split("\t")[2] != f"arms-{cross}"
    ]
    hist.write_text("\n".join(kept) + "\n")
    r, launches = run()
    assert launches == {}, launches
    line = next(ln for ln in r.stderr.splitlines() if ln.startswith(f"[{cross}] SKIP"))
    assert "has not finished" in line and "ARMS_RESUME=1" in line, line
    # with the switch the base resumes in the registration pass, and only then the cross starts
    r, launches = run(ARMS_RESUME="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert set(launches) == {f"arms-{base}-r2", f"arms-{cross}"}, launches


def _external_root(tmp_path, reference_status):
    from benchmarks.tests.test_build_arm_plan import _plan_csv as plan_csv

    cfg = _launch_cfg()
    cfg["external_baseline"]["ashlar"]["enabled"] = True
    plan = [p for p in build_arm_plan(cfg) if p["arm_kind"] == "external"]
    assert plan, "no external rows planned"
    root = tmp_path / "arm_results"
    (root / "preprocess_shared" / "csv").mkdir(parents=True)
    (root / "preprocess_shared" / "csv" / "preprocessed.csv").write_text("patient_id\n")
    ref = plan[0]["ext_from_arm"]
    (root / ref).mkdir()
    hist = root / ".launch" / ref / ".nextflow" / "history"
    hist.parent.mkdir(parents=True)
    hist.write_text(f"now\t1s\tarms-{ref}\t{reference_status}\t-\tsess\tnextflow run\n")
    plan_file = tmp_path / "plan.csv"
    plan_file.write_text(plan_csv(plan))
    sheet = tmp_path / "input.csv"
    sheet.write_text(
        "patient_id,path_to_file,is_reference,channels\nP1,/x/a.tif,true,DAPI\n"
    )

    def run():
        env = dict(os.environ, ARMS_CONCURRENCY="4")
        return subprocess.run(
            ["bash", str(BENCH / "run_arms.sh"), str(plan_file), str(sheet), str(root)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    return plan, root, run


def test_ashlar_waits_for_the_arm_whose_nuclei_it_scores_against(tmp_path):
    plan, root, run = _external_root(tmp_path, "-")
    r = run()
    for p in plan:
        line = next(
            ln for ln in r.stderr.splitlines() if ln.startswith(f"[{p['run_id']}] SKIP")
        )
        assert "has not finished" in line and p["ext_from_arm"] in line, line
        assert not (root / p["arm"] / "ashlar.stdout.log").exists(), (
            "ASHLAR was started"
        )


def test_a_finished_ashlar_arm_is_done_and_not_rerun(tmp_path):
    plan, root, run = _external_root(tmp_path, "OK")
    for p in plan:
        (root / p["arm"]).mkdir(parents=True, exist_ok=True)
        (root / p["arm"] / ".external_done").write_text("2026-09-14T12:00:00\n")
    r = run()
    for p in plan:
        assert f"[{p['run_id']}] DONE" in r.stdout, r.stdout + r.stderr
        assert not (root / p["arm"] / "ashlar.stdout.log").exists(), (
            "ASHLAR was started"
        )


def test_the_ashlar_pass_runs_right_after_registration():
    import re

    code = "\n".join(
        ln
        for ln in (BENCH / "run_arms.sh").read_text().splitlines()
        if not ln.lstrip().startswith("#")
    )
    order = re.search(r"^for kind in ([a-z_ ]+); do", code, re.M).group(1).split()
    assert order.index("registration") + 1 == order.index("external"), order
    assert order.index("external") < order.index("registration_qc"), order
