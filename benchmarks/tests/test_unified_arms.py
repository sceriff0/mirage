"""One results root, every method: VALIS, DRAPE, STARE v1 (pinned code), ASHLAR, seg.

STARE v1 and DRAPE cannot live in one tree -- both are registration_method=tiled and v1's
solvers and reg_tiled_gate_tre are deleted here -- so STARE runs as PINNED-CODE rows:
built by the pinned tree's own plan builder, launched from a `git archive` snapshot at
that commit (benchmarks/code_snapshot.py). These tests pin the four properties the
single-launcher design rests on:

  * every row says which method it is, so METHODS= and the analysis can split STARE
    from DRAPE although both are `tiled`;
  * a pinned row runs from its snapshot, with its own params forwarded, and is refused
    if it was recorded under another commit (never kept or resumed under other code);
  * a METHODS= selection launches what it reads, and ARMS_REPLACE never moves that;
  * cost-by-tier keys STARE and DRAPE apart, each with its own depth column.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from benchmarks import code_snapshot, impact
from benchmarks.analysis.lib import quality
from benchmarks.build_arm_plan import (
    METHODS,
    build_arm_plan,
    check_pinned_params,
    pinned_rows,
    select_methods,
)
from benchmarks.tests.test_build_arm_plan import _fake_nextflow, _plan_csv
from benchmarks.tests.test_subset_rerun_equivalence import _launch_cfg

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent
SHA = "a" * 40


def _old_plan() -> list[dict]:
    """A STARE v1 plan as the pinned builder writes it: text values, its own columns."""
    common = dict(
        start="registration",
        stop="registration",
        from_arm="preprocess_shared",
        from_csv="preprocessed",
        rep="0",
        backend="tiled",
        registration_method="tiled",
        memory_mode="",
        reg_micro_reg="",
        reg_qc="2",
        seg_qc_pairing="lsa",
        reg_tiled_mode="low",
        reg_tiled_gate_tre="1.0",
        reg_tiled_solver="legacy",
    )
    return [
        dict(
            common,
            run_id="preprocess_shared",
            arm="preprocess_shared",
            arm_kind="preprocess",
            registration_method="",
            backend="",
            from_arm="",
            reg_tiled_mode="",
            reg_tiled_gate_tre="",
            reg_tiled_solver="",
            resume_run="",
            seg_method="instantseg",
        ),
        dict(
            common,
            run_id="valis_high_micro2",
            arm="valis_high_micro2",
            arm_kind="registration",
            registration_method="valis",
            backend="valis",
            reg_tiled_mode="",
            reg_tiled_gate_tre="",
            reg_tiled_solver="",
            resume_run="",
            seg_method="instantseg",
        ),
        dict(
            common,
            run_id="tiled_low_gate1",
            arm="tiled_low_gate1",
            arm_kind="registration",
            resume_run="",
            seg_method="instantseg",
        ),
        dict(
            common,
            run_id="tiled_low_gate1_segstardist",
            arm="tiled_low_gate1_segstardist",
            arm_kind="registration_qc",
            resume_run="tiled_low_gate1",
            seg_method="stardist",
        ),
        dict(
            common,
            run_id="tiled_low_gate1_solver_robust",
            arm="tiled_low_gate1_solver_robust",
            arm_kind="registration_solver",
            resume_run="tiled_low_gate1",
            seg_method="instantseg",
            reg_tiled_solver="robust",
        ),
    ]


def _pinned(current_columns) -> list[dict]:
    return pinned_rows(
        _old_plan(),
        {"tiled_low_gate1": "tiled (STARE, low, gate 1 px)"},
        method="stare",
        code_ref=SHA,
        select="tiled",
        current_columns=current_columns,
    )


# ------------------------------------------------------------------ the plan --
def test_every_row_names_its_method_and_tiled_here_is_drape():
    plan = build_arm_plan(_launch_cfg())
    assert {r["method"] for r in plan} <= set(METHODS)
    assert all(r["method"] for r in plan), [
        r["run_id"] for r in plan if not r["method"]
    ]
    for r in plan:
        if r.get("registration_method") == "tiled":
            assert r["method"] == "drape", r["run_id"]
        assert r["code_ref"] == "" and r["role"] == ""


def test_pinned_rows_take_the_tiled_closure_and_forward_only_what_this_tree_lacks():
    current = {k for r in build_arm_plan(_launch_cfg()) for k in r}
    rows = _pinned(current)
    assert [r["run_id"] for r in rows] == [
        "tiled_low_gate1",
        "tiled_low_gate1_segstardist",
        "tiled_low_gate1_solver_robust",
    ], "the VALIS and preprocess rows of the pinned plan must never be re-run from it"
    for r in rows:
        assert r["method"] == "stare" and r["code_ref"] == SHA
        # reg_tiled_mode exists in this tree's config: add_param forwards it already.
        assert r["pinned_params"] == "reg_tiled_gate_tre;reg_tiled_solver"


def test_a_pinned_param_the_pinned_schema_does_not_declare_is_refused(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps({"properties": {"reg_tiled_gate_tre": {"type": "number"}}})
    )
    rows = _pinned({k for r in build_arm_plan(_launch_cfg()) for k in r})
    bad = check_pinned_params(rows, schema)
    assert bad and all("reg_tiled_solver" in b for b in bad)
    schema.write_text(
        json.dumps(
            {
                "$defs": {
                    "x": {
                        "properties": {"reg_tiled_gate_tre": {}, "reg_tiled_solver": {}}
                    }
                }
            }
        )
    )
    assert check_pinned_params(rows, schema) == []


def test_upstream_is_the_transitive_reverse_of_closure():
    plan = _old_plan()
    cross = [r for r in plan if r["run_id"] == "tiled_low_gate1_segstardist"]
    got = [r["run_id"] for r in impact.upstream(plan, cross)]
    assert got == ["preprocess_shared", "tiled_low_gate1"]


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


# -------------------------------------------------------------- the snapshot --
def test_a_snapshot_is_the_commit_and_is_reused(tmp_path):
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if not head:
        pytest.skip("not a git checkout")
    d = code_snapshot.materialise(REPO, head, tmp_path)
    assert (
        d == tmp_path / head
        and (d / code_snapshot.COMPLETE).read_text().strip() == head
    )
    assert os.access(d / "benchmarks" / "run_arms.sh", os.X_OK), "modes must survive"
    (d / "marker").write_text("x")
    assert code_snapshot.materialise(REPO, head, tmp_path) == d
    assert (d / "marker").exists(), "a complete snapshot is reused, not re-extracted"
    (d / code_snapshot.COMPLETE).unlink()
    code_snapshot.materialise(REPO, head, tmp_path)
    assert not (d / "marker").exists(), "an incomplete snapshot is redone"


# ---------------------------------------------------------------- the launcher --
@pytest.fixture
def launcher(tmp_path):
    """run_arms.sh over a plan with one pinned STARE base + cross, stub nextflow, and a
    stand-in snapshot (this tree's benchmarks/ + schema) at <root>/.code/<SHA>."""
    base = build_arm_plan(_launch_cfg())
    plan = base + _pinned({k for r in base for k in r})
    root = tmp_path / "arm_results"
    snap = root / ".code" / SHA
    (snap / "benchmarks").mkdir(parents=True)
    for f in ("params_json.py", "__init__.py"):
        shutil.copy(BENCH / f, snap / "benchmarks" / f)
    (snap / "benchmarks" / "configs").mkdir()
    (snap / "benchmarks" / "configs" / "benchmark.config").write_text("")
    shutil.copy(REPO / "nextflow_schema.json", snap / "nextflow_schema.json")
    (snap / code_snapshot.COMPLETE).write_text(SHA + "\n")
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

    return plan, root, snap, run


def _history_cmd(root: Path, launch_dir: str, run_name: str) -> str:
    for ln in (
        (root / ".launch" / launch_dir / ".nextflow" / "history")
        .read_text()
        .splitlines()
    ):
        f = ln.split("\t")
        if f[2] == run_name:
            return f[6]
    raise AssertionError(f"{run_name} not launched")


def test_a_pinned_row_runs_from_its_snapshot_with_its_own_params(launcher):
    plan, root, snap, run = launcher
    r = run(plan)
    assert r.returncode == 0, r.stdout + r.stderr
    cmd = _history_cmd(root, "tiled_low_gate1", "arms-tiled_low_gate1")
    assert f" run {snap} " in cmd, cmd
    assert f"-c {snap}/benchmarks/configs/benchmark.config" in cmd
    params = json.loads(
        (root / ".launch/tiled_low_gate1/params.tiled_low_gate1.json").read_text()
    )
    assert params["reg_tiled_gate_tre"] == 1.0 or params["reg_tiled_gate_tre"] == "1.0"
    assert params["reg_tiled_solver"] == "legacy" and "reg_tiled_stride" not in params
    # The cross is chained in the QC pass (registration_solver too) and resumes its base.
    assert f" run {snap} " in _history_cmd(
        root, "tiled_low_gate1", "arms-tiled_low_gate1_solver_robust"
    )
    assert (
        root / ".launch/tiled_low_gate1/code.tiled_low_gate1"
    ).read_text().strip() == SHA
    # A current-checkout row runs from this tree and records its HEAD.
    assert f" run {REPO} " in _history_cmd(
        root, "valis_high_micro2", "arms-valis_high_micro2"
    )


def test_a_pinned_arm_recorded_under_another_commit_is_refused(launcher):
    plan, root, snap, run = launcher
    assert run(plan).returncode == 0
    (root / ".launch/tiled_low_gate1/code.tiled_low_gate1").write_text("b" * 40 + "\n")
    r = run(plan)
    assert "tiled_low_gate1] SKIP: this arm ran under commit bbbb" in r.stderr, r.stderr
    assert "arms-tiled_low_gate1] DONE" not in r.stdout


def test_replace_never_moves_an_upstream_row(launcher):
    plan, root, snap, run = launcher
    assert run(plan).returncode == 0
    sub = select_methods(plan, ["stare"])
    assert [r["run_id"] for r in sub if r["role"] == "upstream"] == [
        "preprocess_shared"
    ]
    r = run(sub, ARMS_REPLACE="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (root / "preprocess_shared").is_dir(), "an upstream row's results were moved"
    assert "[preprocess_shared] replaced" not in r.stdout
    assert "[tiled_low_gate1] replaced" in r.stdout
    assert "[preprocess_shared] DONE" in r.stdout


# ---------------------------------------------------------------- the analysis --
def test_cost_by_tier_keys_stare_and_drape_apart():
    def rows(run, method, **depth):
        return [
            dict(
                run_id=run,
                arm_kind="registration",
                registration_method="tiled",
                method=method,
                reg_tiled_mode="high",
                process=p,
                realtime_s=3600.0,
                cpus=2,
                peak_rss_gb=4.0,
                **depth,
            )
            for p in ("TILED_COARSE", "TILED_SOLVE", "WARP_SEG_QC")
        ]

    df = pd.DataFrame(
        rows(
            "tiled_high_gate1",
            "stare",
            reg_tiled_gate_tre=1.0,
            reg_tiled_stride=float("nan"),
        )
        + rows(
            "tiled_high_s128",
            "drape",
            reg_tiled_gate_tre=float("nan"),
            reg_tiled_stride=128.0,
        )
    )
    out = quality.registration_cost_by_tier(df, "/nonexistent").set_index("run_id")
    assert out.loc["tiled_high_gate1", "backend"] == "stare"
    assert out.loc["tiled_high_gate1", "depth"] == "1.0"
    assert out.loc["tiled_high_s128", "backend"] == "drape"
    assert out.loc["tiled_high_s128", "depth"] == "128"
    assert (out["n_tasks"] == 2).all(), "QC processes are not registration cost"


def test_every_plan_column_the_analysis_splits_on_is_written(tmp_path):
    """The full plan the analysis reads must carry `method`, or cost-by-tier falls back
    to registration_method and pools STARE with DRAPE."""
    p = tmp_path / "plan.csv"
    p.write_text(_plan_csv(build_arm_plan(_launch_cfg())))
    header = next(csv.reader(p.open()))
    assert {"method", "code_ref", "pinned_params", "role"} <= set(header)
