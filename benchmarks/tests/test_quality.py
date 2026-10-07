import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.analysis.lib import quality


# ── cost ──
def test_run_cost_summary_cpu_gpu_hours_and_bottleneck():
    df = pd.DataFrame(
        [
            {
                "process": "MIRAGE:PREPROCESSING:PREPROCESS",
                "realtime_s": 100,
                "cpus": 4,
                "run_id": "r0",
                "varied_axis": "baseline",
            },
            {
                "process": "MIRAGE:REGISTRATION:VALIS_ADAPTER:REGISTER",
                "realtime_s": 400,
                "cpus": 8,
                "run_id": "r0",
                "varied_axis": "baseline",
            },
            {
                "process": "MIRAGE:POSTPROCESSING:SEGMENT",
                "realtime_s": 200,
                "cpus": 2,
                "run_id": "r0",
                "varied_axis": "baseline",
            },
        ]
    )
    out = quality.run_cost_summary(df).set_index("run_id")
    row = out.loc["r0"]
    assert row["total_realtime_s"] == 700
    assert row["cpu_hours"] == (100 * 4 + 400 * 8 + 200 * 2) / 3600
    assert row["gpu_hours"] == 200 / 3600  # only SEGMENT is a GPU leaf
    assert row["bottleneck_stage"] == "REGISTER"  # largest single realtime
    assert abs(row["bottleneck_frac"] - 400 / 700) < 1e-9


def test_run_cost_summary_wall_clock_from_timestamps():
    ts = pd.to_datetime(["2026-07-01 10:00:00", "2026-07-01 10:02:00"])
    tc = pd.to_datetime(["2026-07-01 10:01:00", "2026-07-01 10:07:00"])
    df = pd.DataFrame(
        {
            "process": ["A:P", "A:REGISTER"],
            "realtime_s": [60, 300],
            "cpus": [1, 1],
            "run_id": ["r0", "r0"],
            "start_ts": ts,
            "complete_ts": tc,
        }
    )
    out = quality.run_cost_summary(df).set_index("run_id")
    assert (
        out.loc["r0", "wall_clock_s"] == 7 * 60
    )  # 10:00:00 start -> 10:07:00 complete


# ── registration accuracy (staged QC, reg_qc=2) ──
def _write_seg_qc(root, run_id, patient, moving, stages, deltas, pair_fraction=0.9):
    d = root / run_id / "out" / patient / "qc" / "registration"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{patient}_{moving}_seg_qc.json").write_text(
        json.dumps(
            {
                "patient_id": patient,
                "moving": moving,
                "reference": f"{patient}_ref",
                "stages_separable": True,
                "stage_order": list(stages),
                "stages": stages,
                "delta_vs_anchor": deltas,
                "matching": {"pair_fraction": pair_fraction, "n_pairs": 1000},
            }
        )
    )


def test_harvest_registration_qc(tmp_path):
    plan = tmp_path / "plan.csv"
    pd.DataFrame({"run_id": ["r0", "r1"]}).to_csv(plan, index=False)
    stages = {
        "rigid": {
            "n_pairs": 1000,
            "iou_mean": 0.42,
            "iou_p50": 0.40,
            "dice_matched": 0.55,
            "displacement_px_p50": 4.1,
            "displacement_um_p50": 1.33,
        },
        "non_rigid": {
            "n_pairs": 1000,
            "iou_mean": 0.71,
            "iou_p50": 0.70,
            "dice_matched": 0.80,
            "displacement_px_p50": 1.6,
            "displacement_um_p50": 0.52,
        },
    }
    deltas = {
        "non_rigid": {
            "dice_matched": 0.25,
            "displacement_um_p50": -0.81,
            "displacement_px_p50": -2.5,
        }
    }
    _write_seg_qc(tmp_path, "r0", "P001", "cycle2", stages, deltas, pair_fraction=0.91)
    # r1 has no seg_qc -> skipped
    out = quality.harvest_registration_qc(tmp_path, plan)
    assert set(out["run_id"]) == {"r0"}
    assert set(out["stage"]) == {"rigid", "non_rigid"}
    nr = out[out["stage"] == "non_rigid"].iloc[0]
    assert nr["dice_matched"] == 0.80
    assert nr["displacement_um_p50"] == 0.52
    assert nr["delta_dice_vs_rigid"] == 0.25
    assert nr["pair_fraction"] == 0.91
    rigid = out[out["stage"] == "rigid"].iloc[0]
    assert np.isnan(rigid["delta_dice_vs_rigid"])  # anchor has no delta


def test_harvest_registration_qc_tolerates_bad_json(tmp_path):
    plan = tmp_path / "plan.csv"
    pd.DataFrame({"run_id": ["r0"]}).to_csv(plan, index=False)
    d = tmp_path / "r0" / "out" / "P001" / "qc" / "registration"
    d.mkdir(parents=True)
    (d / "x_seg_qc.json").write_text("{ not json")
    assert quality.harvest_registration_qc(tmp_path, plan).empty  # no crash, just empty


# ── registration accuracy (VALIS-reported rTRE / D) ──
def _write_valis_summary(root, run_id, patient, rows):
    d = root / run_id / "out" / patient / "registered" / "summary"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / f"{patient}_summary.csv", index=False)


def test_harvest_valis_rtre_and_per_run(tmp_path):
    plan = tmp_path / "plan.csv"
    pd.DataFrame({"run_id": ["r0", "r1"]}).to_csv(plan, index=False)
    _write_valis_summary(
        tmp_path,
        "r0",
        "P001",
        [
            {
                "name": "cycle1",
                "original_D": 40.0,
                "rigid_D": 8.0,
                "non_rigid_D": 3.0,
                "n_matches": 200,
            },
            {
                "name": "cycle2",
                "original_D": 44.0,
                "rigid_D": 9.0,
                "non_rigid_D": 5.0,
                "n_matches": 180,
            },
        ],
    )
    # r1 has no summary -> contributes nothing
    long = quality.harvest_valis_rtre(tmp_path, plan)
    assert set(long["run_id"]) == {"r0"}
    assert len(long) == 2  # two slides
    assert "non_rigid_D" in long.columns and "summary_csv" in long.columns

    per_run = quality.valis_rtre_per_run(long).set_index("run_id")
    # median across the two slides, prefixed valis_
    assert per_run.loc["r0", "valis_non_rigid_D"] == 4.0  # median(3, 5)
    assert per_run.loc["r0", "valis_original_D"] == 42.0  # median(40, 44)


def test_harvest_valis_rtre_tolerates_missing(tmp_path):
    plan = tmp_path / "plan.csv"
    pd.DataFrame({"run_id": ["r0"]}).to_csv(plan, index=False)
    assert quality.harvest_valis_rtre(tmp_path, plan).empty
    assert quality.valis_rtre_per_run(pd.DataFrame(columns=["run_id"])).empty


# ── segmentation counts + agreement (injected mask reader) ──
def _mk_seg_run(tmp_path, run_id, patient="P001"):
    d = tmp_path / run_id / "out" / patient / "segment"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{patient}_cell_mask.tif"
    p.write_bytes(b"")
    return p


def test_harvest_segmentation_counts(tmp_path):
    plan = tmp_path / "plan.csv"
    pd.DataFrame({"run_id": ["r0", "r1"]}).to_csv(plan, index=False)
    _mk_seg_run(tmp_path, "r0")
    _mk_seg_run(tmp_path, "r1")
    # r1 has a NON-contiguous label (5) for a single cell — distinct-count = 1 (correct); max-label = 5.
    masks = {"r0": np.array([[0, 1], [2, 3]]), "r1": np.array([[0, 0], [0, 5]])}

    def reader(p):
        return masks["r0" if "r0" in str(p) else "r1"]

    out = quality.harvest_segmentation_counts(tmp_path, plan, reader=reader).set_index(
        "run_id"
    )
    assert out.loc["r0", "n_cells"] == 3 and out.loc["r1", "n_cells"] == 1


def test_instance_f1_iou_matching():
    ma = np.array([[1, 1, 1], [2, 2, 2]])
    mb = np.array(
        [[1, 1, 1], [0, 0, 2]]
    )  # cell1 exact match (IoU=1); cell2 IoU=1/3 < 0.5 -> unmatched
    r = quality.instance_f1(ma, mb, iou_thresh=0.5)
    assert r["n_a"] == 2 and r["n_b"] == 2 and r["matched"] == 1
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["f1"] == 0.5
    # identical masks -> perfect agreement
    r2 = quality.instance_f1(ma, ma)
    assert r2["f1"] == 1.0 and r2["matched"] == 2


def test_harvest_segmentation_counts_finds_global_default_location(tmp_path):
    # SEGMENT uses the global-default publishDir -> masks land at out/segment/ (NOT out/<patient>/...),
    # so the harvest must search recursively. Regression for the wrong */segment*/ glob.
    plan = tmp_path / "plan.csv"
    pd.DataFrame({"run_id": ["r0"]}).to_csv(plan, index=False)
    d = tmp_path / "r0" / "out" / "segment"
    d.mkdir(parents=True)  # no patient dir
    (d / "P001_cell_mask.tif").write_bytes(b"")

    def reader(_p):
        return np.array([[0, 1], [2, 3]])

    out = quality.harvest_segmentation_counts(tmp_path, plan, reader=reader)
    assert len(out) == 1 and out.iloc[0]["n_cells"] == 3


def test_segmentation_agreement_pairwise(tmp_path):
    plan = tmp_path / "plan.csv"
    pd.DataFrame(
        {
            "run_id": ["s", "c"],
            "target_px": [4096, 4096],
            "n_channels": [2, 2],
            "seg_method": ["stardist", "cellsam"],
        }
    ).to_csv(plan, index=False)
    _mk_seg_run(tmp_path, "s")
    _mk_seg_run(tmp_path, "c")
    # contiguous labels 1..N (as real masks are relabeled), so max label == cell count
    m = {
        "s": np.array([[1, 1], [0, 2]]),
        "c": np.array([[1, 0], [0, 2]]),
    }  # 2 vs 2 cells, partial overlap

    def reader(p):
        return m["s" if "/s/" in str(p) else "c"]

    out = quality.segmentation_agreement(tmp_path, plan, reader=reader)
    assert len(out) == 1
    r = out.iloc[0]
    assert {r["method_a"], r["method_b"]} == {"stardist", "cellsam"}
    # foreground: s = {(0,0),(0,1),(1,1)}=3px, c = {(0,0),(1,1)}=2px, inter=2, union=3
    assert abs(r["foreground_iou"] - 2 / 3) < 1e-9
    assert r["n_cells_a"] == 2 and r["n_cells_b"] == 2
    # Dice of the same foreground, via the exact identity Dice = 2J / (1 + J).
    assert abs(r["foreground_dice"] - 2 * (2 / 3) / (1 + 2 / 3)) < 1e-9


def _mk_arm_seg(tmp_path, arm, patient):
    # arms layout: <root>/<arm>/<patient>/..., no out/ segment (run_arms.sh)
    d = tmp_path / arm / patient / "segment"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{patient}_cell_mask.tif").write_bytes(b"")


def test_segmentation_agreement_on_arms_pairs_segmentation_arms_per_patient(tmp_path):
    # An arm plan has no target_px/n_channels: the old grouping raised and the table
    # came out empty. Only arm_kind=segmentation rows are compared, per patient, within
    # one from_arm; a registration arm's seg_method is its QC segmenter, not a candidate.
    plan = tmp_path / "arm_plan.csv"
    pd.DataFrame(
        {
            "run_id": [
                "valis_high_micro2",
                "seg_instantseg",
                "seg_stardist",
                "seg_cellsam",
            ],
            "arm_kind": [
                "registration",
                "segmentation",
                "segmentation",
                "segmentation",
            ],
            "from_arm": [
                "",
                "valis_high_micro2",
                "valis_high_micro2",
                "valis_high_micro2",
            ],
            "seg_method": ["instantseg", "instantseg", "stardist", "cellsam"],
        }
    ).to_csv(plan, index=False)
    for arm in ("valis_high_micro2", "seg_instantseg", "seg_stardist", "seg_cellsam"):
        for pid in ("P1", "P2"):
            _mk_arm_seg(tmp_path, arm, pid)
    masks = {
        "seg_instantseg": np.array([[1, 1], [0, 2]]),
        "seg_stardist": np.array([[1, 0], [0, 2]]),
        "seg_cellsam": np.array([[1, 1], [2, 2]]),
    }

    def reader(p):
        arm = Path(p).parts[-4]
        assert arm != "valis_high_micro2", "registration arm must not be read"
        return masks[arm]

    out = quality.segmentation_agreement(tmp_path, plan, reader=reader)
    assert len(out) == 6  # 3 method pairs x 2 patients
    assert set(out["patient_id"]) == {"P1", "P2"}
    assert set(out["from_arm"]) == {"valis_high_micro2"}
    pairs = {frozenset((a, b)) for a, b in zip(out["method_a"], out["method_b"])}
    assert pairs == {
        frozenset(("cellsam", "instantseg")),
        frozenset(("cellsam", "stardist")),
        frozenset(("instantseg", "stardist")),
    }
    r = out[(out["method_a"] == "instantseg") & (out["method_b"] == "stardist")].iloc[0]
    assert abs(r["foreground_iou"] - 2 / 3) < 1e-9
    assert abs(r["foreground_dice"] - 0.8) < 1e-9


def _trace_rows(run, backend, tier, depth, procs):
    # procs: (process, realtime_s, cpus, peak_rss_gb, start_s)
    base = pd.Timestamp("2026-01-01")
    return [
        {
            "run_id": run,
            "arm_kind": "registration",
            "registration_method": backend,
            "memory_mode": tier if backend == "valis" else "",
            "reg_micro_reg": depth if backend == "valis" else "",
            "reg_tiled_mode": tier if backend == "tiled" else "",
            "reg_tiled_stride": depth if backend == "tiled" else "",
            "seg_method": "instantseg",
            "seg_qc_pairing": "lsa",
            "process": f"MIRAGE:REGISTRATION:{p}",
            "realtime_s": rt,
            "cpus": c,
            "peak_rss_gb": rss,
            "start_ts": base + pd.Timedelta(seconds=st),
            "complete_ts": base + pd.Timedelta(seconds=st + rt),
        }
        for p, rt, c, rss, st in procs
    ]


def _registered_csv(root, run, n):
    d = root / run / "csv"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"patient_id": ["P1"] * n, "image": [f"s{i}" for i in range(n)]}
    ).to_csv(d / "registered.csv", index=False)


def test_registration_cost_by_tier_counts_registration_processes_only(tmp_path):
    rows = _trace_rows(
        "valis_high_micro2",
        "valis",
        "high",
        "2",
        [
            ("REGISTER", 3600, 8, 40.0, 0),
            # the reg_qc=2 QC layer: not registration cost, must not count
            ("WARP_SEG_QC", 7200, 16, 90.0, 3600),
            ("SEGMENT", 7200, 16, 99.0, 3600),
        ],
    ) + _trace_rows(
        "stare_high",
        "tiled",
        "high",
        "128",
        [
            ("TILED_COARSE", 600, 4, 10.0, 0),
            ("TILED_REG_TILE", 1200, 2, 5.0, 600),
            ("TILED_REG_TILE", 1200, 2, 6.0, 600),
            ("TILED_SOLVE", 300, 1, 2.0, 1800),
            ("TILED_STITCH", 900, 4, 30.0, 2100),
        ],
    )
    runs = pd.DataFrame(rows)
    _registered_csv(tmp_path, "valis_high_micro2", 4)
    _registered_csv(tmp_path, "stare_high", 4)
    out = quality.registration_cost_by_tier(runs, tmp_path).set_index("run_id")

    v = out.loc["valis_high_micro2"]
    assert v["backend"] == "valis" and v["tier"] == "high" and v["depth"] == "2"
    assert v["n_slides"] == 4
    assert v["reg_peak_rss_gb"] == pytest.approx(40.0)  # QC's 99 GB excluded
    assert v["reg_cpu_hours"] == pytest.approx(8.0)
    assert v["reg_wall_h"] == pytest.approx(1.0)
    assert v["cpu_hours_per_slide"] == pytest.approx(2.0)
    assert v["wall_h_per_slide"] == pytest.approx(0.25)

    s = out.loc["stare_high"]
    assert s["backend"] == "tiled" and s["tier"] == "high" and s["depth"] == "128"
    assert s["reg_wall_h"] == pytest.approx(3000 / 3600)  # first start -> last end
    assert s["reg_cpu_hours"] == pytest.approx((2400 + 2400 + 2400 + 300 + 3600) / 3600)
    assert s["reg_peak_rss_gb"] == pytest.approx(30.0)


def test_registration_cost_by_tier_without_checkpoint_has_nan_per_slide(tmp_path):
    runs = pd.DataFrame(
        _trace_rows(
            "valis_low_micro0", "valis", "low", "0", [("REGISTER", 3600, 4, 8.0, 0)]
        )
    )
    out = quality.registration_cost_by_tier(runs, tmp_path)
    assert len(out) == 1
    assert np.isnan(out.iloc[0]["n_slides"])
    assert np.isnan(out.iloc[0]["cpu_hours_per_slide"])
    assert out.iloc[0]["reg_cpu_hours"] == pytest.approx(4.0)


def test_registration_cost_by_tier_skips_non_registration_arms(tmp_path):
    rows = _trace_rows("x", "valis", "high", "2", [("REGISTER", 60, 1, 1.0, 0)])
    rows[0]["arm_kind"] = "segmentation"
    assert quality.registration_cost_by_tier(pd.DataFrame(rows), tmp_path).empty


def test_full_transform_row_is_the_headline_and_the_deltas_stay_on_the_ladder(tmp_path):
    """The scorer's `full_transform` record (final stage, paired after the whole transform)
    becomes its own row and is what the per-run reduction quotes; the *_vs_rigid deltas are
    only defined on the rigid-anchored ladder and keep coming from its last stage."""
    plan = tmp_path / "plan.csv"
    pd.DataFrame({"run_id": ["r0", "old"]}).to_csv(plan, index=False)
    stages = {
        "rigid": {"n_pairs": 1000, "dice_matched": 0.55, "displacement_um_p50": 1.33},
        "micro": {"n_pairs": 1000, "dice_matched": 0.80, "displacement_um_p50": 0.52},
    }
    deltas = {"micro": {"dice_matched": 0.25, "displacement_um_p50": -0.81}}
    _write_seg_qc(tmp_path, "r0", "P001", "cycle2", stages, deltas, pair_fraction=0.70)
    _write_seg_qc(tmp_path, "old", "P001", "cycle2", stages, deltas, pair_fraction=0.70)
    js = next((tmp_path / "r0").rglob("*_seg_qc.json"))
    d = json.loads(js.read_text())
    d["full_transform"] = {
        "stage": "micro",
        "n_pairs": 1300,
        "dice_matched": 0.86,
        "displacement_um_p50": 0.40,
        "matching": {"anchor_stage": "micro", "pair_fraction": 0.95},
    }
    js.write_text(json.dumps(d))

    long = quality.harvest_registration_qc(tmp_path, plan)
    full = long[long["stage"] == quality.FULL_TRANSFORM_STAGE]
    assert list(full["run_id"]) == ["r0"]  # a JSON without the record gets no such row
    row = full.iloc[0]
    assert (row["dice_matched"], row["n_pairs"], row["pair_fraction"]) == (
        0.86,
        1300,
        0.95,
    )
    assert row["paired_at"] == "micro"
    assert np.isnan(row["delta_dice_vs_rigid"])

    per_run = quality.registration_accuracy_per_run(long).set_index("run_id")
    assert per_run.loc["r0", "reg_dice_matched"] == 0.86
    assert per_run.loc["r0", "reg_displacement_um_p50"] == 0.40
    assert per_run.loc["r0", "reg_pair_fraction"] == 0.95
    assert per_run.loc["r0", "reg_delta_dice_vs_rigid"] == 0.25
    # no record -> the ladder's last stage, as before
    assert per_run.loc["old", "reg_dice_matched"] == 0.80
    assert per_run.loc["old", "reg_pair_fraction"] == 0.70


def test_registration_cost_by_patient_splits_phases_and_drops_failed_attempts(tmp_path):
    """Per patient and phase, from the attempts that FINISHED: an out-of-memory attempt
    that was retried is counted as a failure, never summed into the cost."""

    def rows(run, backend, depth, procs):
        out = _trace_rows(run, backend, "high", depth, [p[:5] for p in procs])
        for r, p in zip(out, procs):
            r["tag"], r["status"], r["pcpu"] = p[5], p[6], p[7]
        return out

    runs = pd.DataFrame(
        rows(
            "valis_high_micro2",
            "valis",
            "2",
            [
                ("REGISTER", 7200, 8, 90.0, 0, "P1", "FAILED", 400.0),  # OOM, retried
                ("REGISTER", 3600, 8, 40.0, 0, "P1", "COMPLETED", 400.0),
                ("REGISTER", 1800, 8, 30.0, 0, "P2", "COMPLETED", 400.0),
                ("WARP_SEG_QC", 7200, 16, 99.0, 0, "P1", "COMPLETED", 100.0),  # QC
            ],
        )
        + rows(
            "tiled_high_s64",
            "tiled",
            "64",
            [
                ("TILED_COARSE", 600, 4, 10.0, 0, "P1:DAPI_CD3", "COMPLETED", 100.0),
                ("TILED_REG_TILE", 1200, 2, 5.0, 0, "P1:0_0", "COMPLETED", 200.0),
                ("TILED_REG_TILE", 1200, 2, 6.0, 0, "P1:0_1", "CACHED", 200.0),
                ("TILED_SOLVE", 300, 1, 2.0, 0, "P1:DAPI_CD3", "COMPLETED", 100.0),
                ("TILED_STITCH", 900, 4, 30.0, 0, "P1:DAPI_CD3", "COMPLETED", 100.0),
            ],
        )
    )
    for arm, n in (
        ("valis_high_micro2", {"P1": 4, "P2": 2}),
        ("tiled_high_s64", {"P1": 4}),
    ):
        d = tmp_path / arm / "csv"
        d.mkdir(parents=True)
        pd.DataFrame(
            {"patient_id": [p for p, k in n.items() for _ in range(k)]}
        ).to_csv(d / "registered.csv", index=False)

    out = quality.registration_cost_by_patient(runs, tmp_path)
    assert "reg_wall_h" not in out.columns and "wall_h_per_slide" not in out.columns
    key = out.set_index(["run_id", "patient_id", "phase"])

    v = key.loc[("valis_high_micro2", "P1", "REGISTER")]
    assert v["cpu_hours"] == pytest.approx(8.0)  # the failed 16 core-h are not in it
    assert v["cpu_hours_per_slide"] == pytest.approx(2.0)
    assert v["cpu_hours_used"] == pytest.approx(4.0)  # 400% of one hour
    assert v["peak_rss_gb"] == pytest.approx(40.0)  # not the failed attempt's 90
    assert v["n_failed_attempts"] == 1 and v["n_tasks"] == 1
    assert key.loc[("valis_high_micro2", "P2", "REGISTER")]["n_slides"] == 2
    assert set(out[out["run_id"] == "valis_high_micro2"]["phase"]) == {"REGISTER"}

    s = out[out["run_id"] == "tiled_high_s64"].set_index("phase")
    assert list(s.index) == sorted(s.index) and set(s.index) == set(
        quality.PHASE_ORDER[1:]
    )
    assert set(s["patient_id"]) == {"P1"}  # every tag's first field
    assert s.loc["TILED_REG_TILE", "cpu_hours"] == pytest.approx(4800 / 3600)
    assert s.loc["TILED_REG_TILE", "n_tasks"] == 2
    assert s.loc["TILED_REG_TILE", "peak_rss_gb"] == pytest.approx(6.0)
    assert s["cpu_hours"].sum() == pytest.approx((2400 + 4800 + 300 + 3600) / 3600)
    assert s["n_failed_attempts"].sum() == 0
