"""One results root, every method: VALIS, STARE, ASHLAR, seg, compute.

The properties the single-launcher design rests on:

  * every row says which method it is, so METHODS= and the analysis can select on it;
  * a METHODS= selection launches what it reads, and ARMS_REPLACE never moves that;
  * cost-by-tier keys each method on its own depth column.
"""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from benchmarks import impact
from benchmarks.analysis.lib import quality
from benchmarks.build_arm_plan import METHODS, build_arm_plan, select_methods
from benchmarks.tests.test_build_arm_plan import _fake_nextflow, _plan_csv
from benchmarks.tests.test_subset_rerun_equivalence import _launch_cfg

BENCH = Path(__file__).resolve().parents[1]


def test_every_row_names_its_method_and_tiled_here_is_stare():
    plan = build_arm_plan(_launch_cfg())
    assert {r["method"] for r in plan} <= set(METHODS)
    assert all(r["method"] for r in plan), [
        r["run_id"] for r in plan if not r["method"]
    ]
    for r in plan:
        if r.get("registration_method") == "tiled":
            assert r["method"] == "stare", r["run_id"]
        assert r["role"] == ""


def test_upstream_is_the_transitive_reverse_of_closure():
    plan = build_arm_plan(_launch_cfg())
    cross = [r for r in plan if r["run_id"].startswith("tiled_low_s")][-1:]
    got = [r["run_id"] for r in impact.upstream(plan, cross)]
    assert got[0] == "preprocess_shared", got


def test_a_method_selection_launches_what_it_reads_as_upstream():
    cfg = _launch_cfg()
    cfg["external_baseline"]["ashlar"]["enabled"] = True
    plan = build_arm_plan(cfg)
    rows = select_methods(plan, ["ashlar"])
    up = {r["run_id"] for r in rows if r["role"] == "upstream"}
    assert up == {"preprocess_shared", "valis_high_micro2"}
    assert {r["method"] for r in rows if r["role"] != "upstream"} == {"ashlar"}
    with pytest.raises(ValueError, match="unknown method"):
        select_methods(plan, ["stare_v1"])


# ---------------------------------------------------------------- the launcher --
@pytest.fixture
def launcher(tmp_path):
    """run_arms.sh over the plan with a stub nextflow."""
    plan = build_arm_plan(_launch_cfg())
    root = tmp_path / "arm_results"
    sheet = tmp_path / "input.csv"
    sheet.write_text(
        "patient_id,path_to_file,is_reference,channels\nP1,/x/a.tif,true,DAPI\n"
    )
    _fake_nextflow(tmp_path / "bin", tmp_path / "launches.log")
    env = dict(
        os.environ,
        PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        ARMS_CONCURRENCY="4",
    )
    for k in ("ARMS_REPLACE", "ARMS_RESUME"):
        env.pop(k, None)

    def run(rows, **extra):
        p = tmp_path / "plan.csv"
        p.write_text(_plan_csv(rows))
        return subprocess.run(
            ["bash", str(BENCH / "run_arms.sh"), str(p), str(sheet), str(root)],
            env=dict(env, **extra),
            capture_output=True,
            text=True,
            timeout=300,
        )

    return plan, root, run


def test_every_launch_records_the_commit_that_ran_it(launcher):
    plan, root, run = launcher
    r = run(plan)
    assert r.returncode == 0, r.stdout + r.stderr
    recs = list((root / ".launch").rglob("code.*"))
    assert recs, "no code.<run_id> record was written"


def test_replace_never_moves_an_upstream_row(launcher):
    plan, root, run = launcher
    assert run(plan).returncode == 0
    sub = select_methods(plan, ["stare"])
    assert [r["run_id"] for r in sub if r["role"] == "upstream"] == [
        "preprocess_shared"
    ]
    base = next(r["run_id"] for r in sub if r["role"] != "upstream")
    r = run(sub, ARMS_REPLACE="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (root / "preprocess_shared").is_dir(), "an upstream row's results were moved"
    assert "[preprocess_shared] replaced" not in r.stdout
    assert f"[{base}] replaced" in r.stdout
    assert "[preprocess_shared] DONE" in r.stdout


# ---------------------------------------------------------------- the analysis --
def test_cost_by_tier_keys_each_method_on_its_own_depth():
    def rows(run, method, backend, **depth):
        procs = (
            ("TILED_COARSE", "TILED_SOLVE", "WARP_SEG_QC")
            if backend == "tiled"
            else ("REGISTER", "WARP_SEG_QC")
        )
        return [
            dict(
                run_id=run,
                arm_kind="registration",
                registration_method=backend,
                method=method,
                reg_tiled_mode="high",
                memory_mode="high",
                process=p,
                realtime_s=3600.0,
                cpus=2,
                peak_rss_gb=4.0,
                **depth,
            )
            for p in procs
        ]

    df = pd.DataFrame(
        rows(
            "tiled_high_s128",
            "stare",
            "tiled",
            reg_tiled_stride=128.0,
            reg_micro_reg=float("nan"),
        )
        + rows(
            "valis_high_micro2",
            "valis",
            "valis",
            reg_tiled_stride=float("nan"),
            reg_micro_reg=2.0,
        )
    )
    out = quality.registration_cost_by_tier(df, "/nonexistent").set_index("run_id")
    assert out.loc["tiled_high_s128", "backend"] == "stare"
    assert out.loc["tiled_high_s128", "depth"] == "128"
    assert out.loc["valis_high_micro2", "backend"] == "valis"
    assert out.loc["valis_high_micro2", "depth"] == "2"
    assert out.loc["tiled_high_s128", "n_tasks"] == 2, "QC is not registration cost"


def test_every_plan_column_the_analysis_splits_on_is_written(tmp_path):
    """The full plan the analysis reads must carry `method`, or cost-by-tier falls back
    to registration_method."""
    p = tmp_path / "plan.csv"
    p.write_text(_plan_csv(build_arm_plan(_launch_cfg())))
    header = next(csv.reader(p.open()))
    assert {"method", "role"} <= set(header)
    assert not {"code_ref", "pinned_params"} & set(header)


def test_waves_put_the_reference_then_segmentation_first_and_qc_last(tmp_path):
    """The supplementary order: preprocess -> the reference arm (what seg reads) ->
    segmentation BEFORE the other registration arms, high tier first -> QC crosses last."""
    cfg = _launch_cfg()
    cfg["registration_arms"]["valis"]["memory_mode"] = ["low", "high"]
    cfg["registration_arms"]["valis"]["reg_micro_reg"] = [2]
    cfg["registration_arms"]["tiled"]["reg_tiled_mode"] = ["low", "high"]
    cfg["segmentation_arms"]["seg_method"] = ["instantseg", "stardist"]
    plan = build_arm_plan(cfg)
    root = tmp_path / "arm_results"
    sheet = tmp_path / "input.csv"
    sheet.write_text(
        "patient_id,path_to_file,is_reference,channels\nP1,/x/a.tif,true,DAPI\n"
    )
    log = tmp_path / "launches.log"
    _fake_nextflow(tmp_path / "bin", log)
    # the seg arms resume from the reference arm's registered checkpoint
    nf = tmp_path / "bin" / "nextflow"
    nf.write_text(
        nf.read_text().replace(
            'echo "patient_id" > "$outdir/csv/preprocessed.csv"',
            'echo "patient_id" > "$outdir/csv/preprocessed.csv"; '
            'echo "patient_id" > "$outdir/csv/registered.csv"',
        )
    )
    p = tmp_path / "plan.csv"
    p.write_text(_plan_csv(plan))
    env = dict(
        os.environ,
        PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        ARMS_CONCURRENCY="1",
    )
    for k in ("ARMS_REPLACE", "ARMS_RESUME"):
        env.pop(k, None)
    r = subprocess.run(
        ["bash", str(BENCH / "run_arms.sh"), str(p), str(sheet), str(root)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    order = [
        ln.split("|")[1].removeprefix("arms-") for ln in log.read_text().splitlines()
    ]
    kind = {row["run_id"]: row["arm_kind"] for row in plan}
    pos = {rid: i for i, rid in enumerate(order)}
    ref = cfg["segmentation_arms"]["from_arm"]
    segs = [x for x in order if kind.get(x) == "segmentation"]
    regs = [x for x in order if kind.get(x) == "registration" and x != ref]
    qcs = [x for x in order if kind.get(x) == "registration_qc"]
    assert order[0] == "preprocess_shared" and order[1] == ref, order[:3]
    assert len(segs) == 2 and max(pos[x] for x in segs) < min(pos[x] for x in regs)
    assert "_high_" in regs[0] and "_low_" in regs[-1], regs
    assert qcs and min(pos[x] for x in qcs) > max(pos[x] for x in segs + regs)


def test_arms_rerun_continues_a_finished_arm_from_its_own_session(launcher):
    """ARMS_RERUN=<regex>: a FINISHED arm is resumed from its own session with params
    regenerated (only tasks reading a changed param re-run), not skipped as DONE and not
    moved aside like ARMS_REPLACE. Every other finished arm stays DONE."""
    plan, root, run = launcher
    assert run(plan).returncode == 0
    r = run(plan, ARMS_RESUME="1", ARMS_RERUN="^preprocess_shared$")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[preprocess_shared] RERUN" in r.stdout, r.stdout
    assert "arms-preprocess_shared-r2" in r.stdout
    assert "params REGENERATED" in r.stdout
    others = [x["run_id"] for x in plan if x["run_id"] != "preprocess_shared"]
    assert all(f"[{o}] DONE" in r.stdout for o in others[:3]), r.stdout
    assert (root / "preprocess_shared").is_dir(), "a rerun must not move results aside"
