"""Guards for the opt-in synthetic placeholders (benchmarks/analysis/lib/placeholders.py).

A placeholder is a fabricated number drawn so a figure can be judged before every run
has landed. These tests pin the five properties that keep it from ever passing for a
measurement: off by default, every synthetic row marked, every figure holding one
watermarked, deterministic under the seed, and real points never altered -- including
the CSVs, fits and emitted config, which stay byte-identical to a default run.

The watermark is checked on the ARTIFACT, not on an in-memory flag: figures are saved
as SVG with svg.fonttype='none', so text lands in the file as <text> and a figure
missing its watermark cannot be reported as carrying one.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from benchmarks.analysis import make_figures
from benchmarks.analysis.lib import placeholders

REPO = Path(__file__).resolve().parents[2]
PATIENTS = ("P1", "P2")
TRACE_HEAD = (
    "task_id\tprocess\ttag\tname\tstatus\texit\tsubmit\tstart\tcomplete\tduration"
    "\trealtime\t%cpu\tcpus\tmemory\tpeak_rss\tpeak_vmem\trchar\twchar\n"
)

# arm -> (backend, memory_mode, patients whose QC has landed, trace written?)
# stare_mid has NOTHING yet; valis_low is half-done (P2's QC missing).
ARMS = {
    "valis_high": ("valis", "high", PATIENTS, True),
    "valis_low": ("valis", "low", ("P1",), True),
    "stare_high": ("tiled", "high", PATIENTS, True),
    "stare_mid": ("tiled", "medium", (), False),
}


def _trace(d: Path, rss_gb: int):
    (d / "trace").mkdir(parents=True, exist_ok=True)
    rows = [
        f"1\tREGISTER\tP1\tREGISTER (P1)\tCOMPLETED\t0\t-\t-\t-\t10m 0s\t9m 0s"
        f"\t400%\t4\t64 GB\t{rss_gb} GB\t{rss_gb + 2} GB\t1 GB\t1 GB",
        f"2\tSEGMENT\tP1\tSEGMENT (P1)\tCOMPLETED\t0\t-\t-\t-\t5m 0s\t4m 0s"
        f"\t200%\t2\t32 GB\t{rss_gb // 2} GB\t{rss_gb} GB\t1 GB\t1 GB",
    ]
    (d / "trace" / "trace.txt").write_text(TRACE_HEAD + "\n".join(rows) + "\n")
    (d / "size_logs").mkdir(exist_ok=True)
    (d / "size_logs" / "input_sizes.csv").write_text(
        "process,sample_id,filename,bytes\n"
        f"REGISTER,P1,a.tif,{rss_gb * 2**28}\nSEGMENT,P1,b.tif,{rss_gb * 2**27}\n"
    )


def _seg_qc(d: Path, patient: str, dice: float, disp: float):
    q = d / patient / "qc" / "registration"
    q.mkdir(parents=True, exist_ok=True)
    (q / f"{patient}_R2_seg_qc.json").write_text(
        json.dumps(
            {
                "patient_id": patient,
                "moving": "R2",
                "stage_order": ["rigid", "non_rigid"],
                "stages": {
                    "rigid": {
                        "n_pairs": 100,
                        "dice_matched": dice - 0.2,
                        "displacement_um_p50": disp + 2,
                    },
                    "non_rigid": {
                        "n_pairs": 100,
                        "dice_matched": dice,
                        "displacement_um_p50": disp,
                    },
                },
                "delta_vs_anchor": {},
                "matching": {"pair_fraction": 0.9},
            }
        )
    )


@pytest.fixture
def partial_tree(tmp_path):
    """An arms results root in the ARMS layout with runs at three stages of done."""
    root = tmp_path / "arm_results"
    plan_rows = []
    for i, (arm, (backend, mode, done, traced)) in enumerate(ARMS.items()):
        d = root / arm
        d.mkdir(parents=True)
        if traced:
            _trace(d, rss_gb=8 + 4 * i)
        for j, p in enumerate(done):
            _seg_qc(d, p, dice=0.7 + 0.05 * i + 0.01 * j, disp=3.0 - 0.3 * i + 0.1 * j)
        plan_rows.append(
            {
                "run_id": arm,
                "arm": arm,
                "arm_kind": "registration",
                "backend": backend,
                "registration_method": backend,
                "memory_mode": mode,
            }
        )
    plan = tmp_path / "arm_plan.csv"
    pd.DataFrame(plan_rows).to_csv(plan, index=False)
    return root, plan


def _run(tree, outdir, **kw):
    root, plan = tree
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        return make_figures.run(
            results_root=root,
            run_plan_csv=plan,
            reg_eval_csv="none",
            outdir=outdir,
            formats=("svg",),
            **kw,
        )


def _watermarked(outdir: Path) -> set[str]:
    return {
        str(p.relative_to(outdir).with_suffix(""))
        for p in (outdir / "figures").glob("*.svg")
        if "PLACEHOLDER" in p.read_text()
    }


# ── off by default ────────────────────────────────────────────────────────────
def test_default_run_has_no_placeholders(partial_tree, tmp_path):
    out = tmp_path / "default"
    res = _run(partial_tree, out)
    assert res["placeholders"].empty
    assert not (out / placeholders.SIDECAR).exists()
    assert not (out / placeholders.MARKER).exists()
    assert list((out / "figures").glob("*.svg")), "the default run drew nothing"
    assert _watermarked(out) == set()


def test_env_var_is_the_only_other_switch():
    assert placeholders.resolve_enabled(False, env={}) is False
    assert (
        placeholders.resolve_enabled(False, env={"PLACEHOLDER_MISSING": "0"}) is False
    )
    assert placeholders.resolve_enabled(False, env={"PLACEHOLDER_MISSING": "1"}) is True
    assert placeholders.resolve_enabled(True, env={}) is True


# ── every synthetic point marked, every such figure watermarked ───────────────
def test_placeholder_run_fills_exactly_the_missing_points(partial_tree, tmp_path):
    out = tmp_path / "ph"
    res = _run(partial_tree, out, placeholder_missing=True)
    side = pd.read_csv(out / placeholders.SIDECAR)
    assert (out / placeholders.MARKER).is_file()
    assert (
        side["placeholder_rule"].notna().all()
        and (side["placeholder_rule"] != "").all()
    )

    reg = side[side["figure"] == "figures/registration_accuracy_by_run"]
    got = set(zip(reg["run_id"], reg["patient_id"]))
    # valis_low's P2 and all of stare_mid -- and nothing measured
    assert got == {("valis_low", "P2"), ("stare_mid", "P1"), ("stare_mid", "P2")}
    cost = side[side["figure"] == "figures/cost_by_run"]
    assert set(cost["run_id"]) == {"stare_mid"}
    assert set(side["run_id"]) <= {"valis_low", "stare_mid"}
    assert len(side) == len(res["placeholders"])


def test_every_figure_with_a_placeholder_is_watermarked(partial_tree, tmp_path):
    out = tmp_path / "ph"
    _run(partial_tree, out, placeholder_missing=True)
    side = pd.read_csv(out / placeholders.SIDECAR)
    with_points = set(side["figure"])
    assert with_points, "the partial tree produced no placeholder at all"
    # both directions: every figure holding one says so, and no clean figure does
    assert _watermarked(out) == with_points
    marker = (out / placeholders.MARKER).read_text()
    for f in with_points:
        assert f in marker


def test_ledger_fill_marks_every_synthetic_row_and_keeps_real_cells():
    obs = pd.DataFrame(
        {"run_id": ["a", "b"], "fam": ["x", "x"], "m": [1.0, None], "n": [5.0, 6.0]}
    )
    exp = pd.DataFrame({"run_id": ["a", "b", "c"], "fam": ["x", "x", "x"]})
    led = placeholders.Ledger(enabled=True, seed=0)
    out = led.fill(
        obs,
        exp,
        keys=["run_id"],
        metrics=["m", "n"],
        levels=[("neighbours", ["fam"]), ("global", [])],
        figure="f",
    )
    a, b, c = (out[out["run_id"] == r].iloc[0] for r in "abc")
    assert not a["is_placeholder"] and a["m"] == 1.0 and a["n"] == 5.0
    # b: real row, NaN cell filled -> flagged, its real `n` untouched
    assert b["is_placeholder"] and b["n"] == 6.0 and pd.notna(b["m"])
    assert b["placeholder_rule"] == "m=neighbours(fam)"
    assert (
        c["is_placeholder"]
        and "m=" in c["placeholder_rule"]
        and "n=" in c["placeholder_rule"]
    )
    assert len(led.frame()) == 3  # b.m, c.m, c.n


def test_rules_fall_through_in_order_and_clip():
    led = placeholders.Ledger(enabled=True, seed=0)
    obs = pd.DataFrame(
        {"run_id": ["r1"], "patient": ["P1"], "fam": ["x"], "dice_matched": [0.99]}
    )
    exp = pd.DataFrame(
        {
            "run_id": ["r1", "r2", "r3"],
            "patient": ["P2", "P1", "P1"],
            "fam": ["x", "x", "y"],
        }
    )
    out = led.fill(
        obs,
        exp,
        keys=["run_id", "patient"],
        metrics=["dice_matched"],
        levels=[("same_run", ["run_id"]), ("neighbours", ["fam"]), ("global", [])],
        figure="f",
    )
    rules = dict(
        zip(
            zip(led.frame()["run_id"], led.frame()["patient"]),
            led.frame()["placeholder_rule"],
        )
    )
    assert rules == {
        ("r1", "P2"): "same_run(run_id)",
        ("r2", "P1"): "neighbours(fam)",
        ("r3", "P1"): "global",
    }
    assert out["dice_matched"].between(0, 1).all()
    # no observation of the metric at all -> the documented prior
    led2 = placeholders.Ledger(enabled=True, seed=0)
    led2.fill(
        pd.DataFrame({"run_id": []}),
        pd.DataFrame({"run_id": ["z"]}),
        keys=["run_id"],
        metrics=["cpu_hours"],
        levels=[("global", [])],
        always_expected=True,
        figure="f",
    )
    assert list(led2.frame()["placeholder_rule"]) == ["prior"]
    assert led2.frame()["value"].iloc[0] >= 0


def test_disabled_ledger_is_a_no_op():
    obs = pd.DataFrame({"run_id": ["a"], "m": [None]})
    led = placeholders.Ledger(enabled=False)
    out = led.fill(
        obs,
        pd.DataFrame({"run_id": ["a", "b"]}),
        keys=["run_id"],
        metrics=["m"],
        levels=[],
        always_expected=True,
        figure="f",
    )
    assert out is obs and led.frame().empty


# ── deterministic ────────────────────────────────────────────────────────────
def test_deterministic_under_the_seed(partial_tree, tmp_path):
    a = _run(partial_tree, tmp_path / "a", placeholder_missing=True)["placeholders"]
    b = _run(partial_tree, tmp_path / "b", placeholder_missing=True)["placeholders"]
    c = _run(
        partial_tree, tmp_path / "c", placeholder_missing=True, placeholder_seed=7
    )["placeholders"]
    pd.testing.assert_frame_equal(a, b)
    assert not a["value"].equals(c["value"])


# ── real data never altered ──────────────────────────────────────────────────
REAL_OUTPUTS = (
    "measurements.csv",
    "resource_models.csv",
    "resource_stats.csv",
    "run_cost.csv",
    "registration_accuracy.csv",
    "quality.csv",
    "modules.optimized.config",
)


def test_real_outputs_identical_with_and_without_placeholders(partial_tree, tmp_path):
    _run(partial_tree, tmp_path / "d")
    _run(partial_tree, tmp_path / "p", placeholder_missing=True)
    for name in REAL_OUTPUTS:
        assert (tmp_path / "d" / name).read_bytes() == (
            tmp_path / "p" / name
        ).read_bytes(), name


def test_a_default_run_clears_a_previous_preview(partial_tree, tmp_path):
    out = tmp_path / "same"
    _run(partial_tree, out, placeholder_missing=True)
    assert _watermarked(out)
    _run(partial_tree, out)
    assert not (out / placeholders.SIDECAR).exists()
    assert not (out / placeholders.MARKER).exists()
    assert _watermarked(out) == set()


def test_hand_off_refuses_a_preview_directory(partial_tree, tmp_path):
    handoff = tmp_path / "handoff"
    (handoff / "sweep").mkdir(parents=True)
    (handoff / "sweep" / placeholders.MARKER).write_text("preview\n")
    ihc = tmp_path / "ihc"
    ihc.mkdir()
    root, _ = partial_tree
    r = subprocess.run(
        [
            "bash",
            str(REPO / "benchmarks" / "pull_to_ihc_method.sh"),
            str(root),
            str(ihc),
            "--handoff",
            str(handoff),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1, r.stdout + r.stderr
    assert "REFUSED" in r.stderr
    assert not (ihc / "data").exists() or not any((ihc / "data").rglob("*.csv"))
