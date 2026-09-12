"""Harvest QUALITY + COST signals the pipeline already writes but the resource trace ignores.

The sweep's trace.txt gives RAM/time; this module joins in the *result-quality* the pipeline emits:
  - registration accuracy: staged QC JSON (reg_qc=2)
      <root>/<run_id>/out/<patient>/qc/registration/*_seg_qc.json
    Segments DAPI nuclei, fixes cell correspondence once at the rigid stage, then re-measures
    dice_matched + centroid displacement (px/um) after each transform stage — the landmark-free
    registration-accuracy signal (bin/warp_seg_qc.py; docs/registration_qc.md).
  - segmentation output size: cell count = max label of <...>/segment*/*cell_mask*.tif
  - segmentation cross-method agreement: pairwise mask IoU + cell-count ratio between methods on the
    SAME image (segmentation_grid runs share the baseline cell).
Plus per-run COST derived from the trace (cpu-hours, gpu-hours, wall-clock, bottleneck stage).

Everything is BEST-EFFORT and robust to missing/failed runs (a CellSAM run that OOMs just contributes
no rows) — so the analysis + plots keep working when some runs fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contract import identity_columns, load_sweep

# The two experiment definitions. run_cost_summary serves BOTH -- the Makefile's
# `arm-tables` target runs make_tables against the arms results root -- so the
# identity is the union of what either varies, filtered per table by
# `if c in df.columns`.
SWEEP_PATH = Path(__file__).resolve().parents[2] / "configs" / "sweep.yaml"
ARMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "arms.yaml"

# Processes that run on the GPU (approximate — for a GPU-hours estimate).
GPU_LEAVES = {
    "SEGMENT",
    "QUANTIFY",
    "EXTRACT_CELL_PROPERTIES",
    "GENERATE_POSTPROCESSING_QC",
}


def _leaf(process: str) -> str:
    return str(process).split(":")[-1]


def _run_out(results_root, run_id) -> Path:
    """Where a run's PIPELINE OUTPUTS are, under either results layout.

    The two benchmarks publish differently (load.load_runs documents the same
    split for the size logs):

      sweep  <root>/<run_id>/out/...   run_sweep.sh isolates --outdir under out/
      arms   <root>/<arm>/...          run_arms.sh publishes --outdir straight to
                                       <root>/<arm>, because <arm>/<patient>/qc/... IS
                                       ihc_method's consumer contract

    The harvesters below hardcoded the `out/` segment, so on an arm results root
    every one of them returned nothing: registration_accuracy.csv came out empty
    for the whole arm benchmark while the trace-based tables filled normally, and
    nothing raised. `out/` wins when it exists, so the sweep layout is unchanged.
    """
    run_dir = Path(results_root) / str(run_id)
    out = run_dir / "out"
    return out if out.is_dir() else run_dir


# ─────────────────────────────────────────────────────────── registration accuracy ──
# The anchor stage the staged QC reports deltas against (see docs/registration_qc.md).
REG_QC_ANCHOR = "rigid"


def _f(x):
    """Coerce a JSON value to float, or NaN if missing/non-numeric."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def harvest_registration_qc(results_root, run_plan_csv) -> pd.DataFrame:
    """Per-(run, moving-slide, stage) registration accuracy from the staged QC JSONs (reg_qc=2).

    One row per registration stage (native/rigid/non_rigid/micro) of each moving slide of each run.
    dice_matched and centroid displacement are the co-headline accuracy metrics; the *_vs_rigid
    deltas isolate the registration effect from segmentation noise (docs/registration_qc.md).
    Missing/unparseable JSONs are skipped (best-effort)."""
    root = Path(results_root)
    plan = pd.read_csv(run_plan_csv)
    rows = []
    for run_id in plan["run_id"]:
        for j in _run_out(root, run_id).rglob("*_seg_qc.json"):
            try:
                d = json.loads(j.read_text())
            except Exception:
                continue
            stages = d.get("stages") or {}
            deltas = d.get("delta_vs_anchor") or {}
            pair_fraction = _f((d.get("matching") or {}).get("pair_fraction"))
            for stage in d.get("stage_order") or list(stages):
                s = stages.get(stage) or {}
                dv = deltas.get(stage) or {}
                rows.append(
                    {
                        "run_id": run_id,
                        "patient_id": d.get("patient_id"),
                        "moving": d.get("moving"),
                        "stage": stage,
                        "n_pairs": int(s.get("n_pairs") or 0),
                        "pair_fraction": pair_fraction,
                        "iou_mean": _f(s.get("iou_mean")),
                        "iou_p50": _f(s.get("iou_p50")),
                        "frac_iou_ge_0.5": _f(s.get("frac_iou_ge_0.5")),
                        "dice_matched": _f(s.get("dice_matched")),
                        "displacement_px_p50": _f(s.get("displacement_px_p50")),
                        "displacement_px_p90": _f(s.get("displacement_px_p90")),
                        "displacement_um_p50": _f(s.get("displacement_um_p50")),
                        "displacement_um_p90": _f(s.get("displacement_um_p90")),
                        "delta_dice_vs_rigid": _f(dv.get("dice_matched")),
                        "delta_disp_um_p50_vs_rigid": _f(dv.get("displacement_um_p50")),
                        "delta_disp_px_p50_vs_rigid": _f(dv.get("displacement_px_p50")),
                    }
                )
    return pd.DataFrame(rows)


# Stage ordering so a per-run reduction can pick the final (most-registered) stage.
#
# THE BACKENDS DO NOT SHARE A STAGE VOCABULARY -- lib/WarpBackends.groovy gives VALIS
# native/rigid/non_rigid/micro and the manifest backends (tiled/STARE, ashlar)
# native/rigid/refined. Only `native` and `rigid` are shared as both a spelling and a meaning.
#
# `refined` was MISSING here, which was a live bug for STARE and not merely a gap waiting for
# ashlar: `.map(...).fillna(-1)` sent it to -1, the same rank as the equally-unranked `native`,
# so `sort_values("_rank").groupby(...).tail(1)` picked STARE's terminal stage only by the
# stable-sort tie-break happening to put it after `native`. Any reordering upstream -- a
# different glob order, a pandas version whose sort is not stable here -- would silently have
# reported STARE's UNREGISTERED stage as its headline accuracy.
#
# Ranked, not derived: the ranks encode "how much registration has been applied", which is a
# fact about the backends, not about the data in any one table.
_STAGE_RANK = {"native": 0, "rigid": 1, "non_rigid": 2, "refined": 2, "micro": 3}


def registration_accuracy_per_run(reg_qc_long: pd.DataFrame) -> pd.DataFrame:
    """Reduce the per-(run, moving, stage) QC table to ONE headline row per run: the final
    (most-registered) stage of each moving slide, then the median across moving slides. Columns:
    run_id, reg_dice_matched, reg_displacement_um_p50, reg_delta_disp_um_p50_vs_rigid,
    reg_delta_dice_vs_rigid, reg_pair_fraction. Empty in -> empty out."""
    if reg_qc_long.empty:
        return pd.DataFrame(columns=["run_id"])
    df = reg_qc_long.copy()
    df["_rank"] = df["stage"].map(_STAGE_RANK).fillna(-1)
    # final stage per (run, moving)
    final = (
        df.sort_values("_rank").groupby(["run_id", "moving"], as_index=False).tail(1)
    )
    agg = final.groupby("run_id", as_index=False).agg(
        reg_dice_matched=("dice_matched", "median"),
        reg_displacement_um_p50=("displacement_um_p50", "median"),
        reg_delta_disp_um_p50_vs_rigid=("delta_disp_um_p50_vs_rigid", "median"),
        reg_delta_dice_vs_rigid=("delta_dice_vs_rigid", "median"),
        reg_pair_fraction=("pair_fraction", "median"),
    )
    return agg


# ─────────────────────────────────────────────────── registration accuracy (VALIS rTRE) ──
def harvest_valis_rtre(results_root, run_plan_csv) -> pd.DataFrame:
    """Per-(run, slide) VALIS-reported registration error from the summary CSVs VALIS writes
    during register() (published to <patient>/registered/summary/*.csv). This is the pipeline's
    OTHER accuracy signal — feature-based rTRE / median distance (D), independent of the
    segmentation-based dice/displacement. Columns are whatever VALIS wrote (e.g. original_D,
    rigid_D, non_rigid_D, rTRE variants, n_matches), plus run_id and source. Best-effort."""
    root = Path(results_root)
    plan = pd.read_csv(run_plan_csv)
    frames = []
    for run_id in plan["run_id"]:
        out = _run_out(root, run_id)
        # VALIS summaries are published under .../registered/summary/ (conf/modules.config REGISTER).
        csvs = [
            c
            for c in out.rglob("*.csv")
            if "registered/summary" in c.as_posix() or c.parent.name == "summary"
        ]
        for c in csvs:
            try:
                df = pd.read_csv(c)
            except Exception:
                continue
            if df.empty:
                continue
            df.insert(0, "run_id", run_id)
            df.insert(1, "summary_csv", c.name)
            frames.append(df)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["run_id"])
    )


def valis_rtre_per_run(valis_long: pd.DataFrame) -> pd.DataFrame:
    """Reduce the per-slide VALIS summary to one row per run: median of every numeric column,
    prefixed ``valis_`` (so e.g. valis_non_rigid_D is the post-registration median distance).
    Column-agnostic — works whatever VALIS's exact schema is. Empty in -> empty out."""
    if valis_long.empty or "run_id" not in valis_long.columns:
        return pd.DataFrame(columns=["run_id"])
    num = valis_long.select_dtypes("number")
    if num.empty:
        return pd.DataFrame({"run_id": valis_long["run_id"].unique()})
    num = num.assign(run_id=valis_long["run_id"].values)
    agg = num.groupby("run_id", as_index=False).median()
    agg.columns = ["run_id"] + [f"valis_{c}" for c in agg.columns if c != "run_id"]
    return agg


# ─────────────────────────────────────────────────────────── segmentation counts ──
def _cell_masks(run_out_dir):
    # SEGMENT has no explicit publishDir, so masks land under the global-default dir (out/segment/),
    # NOT out/<patient>/segment*/. Search recursively so we find them wherever they are published.
    p = Path(run_out_dir)
    return sorted(set(p.rglob("*_cell_mask.tif")) | set(p.rglob("*_cell_mask.tiff")))


def _n_cells(path, reader):
    # Count DISTINCT non-zero labels (exact even if labels aren't a contiguous 1..N range — max-label
    # would over-count with gaps).
    try:
        u = np.unique(np.asarray(reader(path)))
        return int((u != 0).sum())
    except Exception:
        return None


def harvest_segmentation_counts(
    results_root, run_plan_csv, reader=None
) -> pd.DataFrame:
    """Per-run cell count = max label of the cell mask (I/O-light: one max() per run). Robust to
    missing masks. reader is injectable for tests; defaults to tifffile.imread."""
    if reader is None:
        import tifffile

        reader = tifffile.imread
    root = Path(results_root)
    plan = pd.read_csv(run_plan_csv)
    rows = []
    for run_id in plan["run_id"]:
        masks = _cell_masks(_run_out(root, run_id))
        counts = [c for c in (_n_cells(m, reader) for m in masks) if c is not None]
        if not counts:
            continue
        rows.append({"run_id": run_id, "n_cells": int(max(counts))})
    return pd.DataFrame(rows)


def instance_f1(ma, mb, iou_thresh=0.5) -> dict:
    """Instance-level agreement between two label masks by GREEDY IoU matching (the standard detection
    metric — far more meaningful than foreground IoU). Returns {f1, precision, recall, matched, n_a, n_b}.
    Efficient: intersections via a flat (a,b) label histogram, not an N×N matrix."""
    a = np.asarray(ma).ravel().astype(np.int64)
    b = np.asarray(mb).ravel().astype(np.int64)
    na, nb = int(a.max()) if a.size else 0, int(b.max()) if b.size else 0
    nan = float("nan")
    if na == 0 or nb == 0:
        return {
            "f1": nan,
            "precision": nan,
            "recall": nan,
            "matched": 0,
            "n_a": na,
            "n_b": nb,
        }
    fg = (a > 0) & (b > 0)
    pair = a[fg] * (nb + 1) + b[fg]  # unique key per (a_label, b_label)
    counts = np.bincount(pair)
    idx = np.nonzero(counts)[0]
    inter = counts[idx].astype(np.float64)
    al, bl = idx // (nb + 1), idx % (nb + 1)
    area_a = np.bincount(a, minlength=na + 1).astype(np.float64)
    area_b = np.bincount(b, minlength=nb + 1).astype(np.float64)
    iou = inter / (area_a[al] + area_b[bl] - inter)
    used_a, used_b, matched = set(), set(), 0
    for k in np.argsort(-iou):  # greedy, highest IoU first
        if iou[k] < iou_thresh:
            break
        if al[k] in used_a or bl[k] in used_b:
            continue
        used_a.add(int(al[k]))
        used_b.add(int(bl[k]))
        matched += 1
    precision = matched / nb
    recall = matched / na
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "matched": matched,
        "n_a": na,
        "n_b": nb,
    }


def segmentation_agreement(results_root, run_plan_csv, reader=None) -> pd.DataFrame:
    """Cross-method agreement on shared cells: for each (target_px, n_channels) segmented by >1 method,
    compare each pair of methods' cell masks — foreground IoU (spatial agreement) + cell-count ratio
    (over/under-segmentation rate). Uses one representative run per (cell, method). Best-effort."""
    if reader is None:
        import tifffile

        reader = tifffile.imread
    root = Path(results_root)
    plan = pd.read_csv(run_plan_csv)
    if "seg_method" not in plan.columns:
        return pd.DataFrame()
    keys = [k for k in ("target_px", "n_channels") if k in plan.columns]
    rows = []
    for cell, g in plan.groupby(keys):
        by_method = {}
        for _, r in (
            g.groupby("seg_method").head(1).iterrows()
        ):  # one run per method at this cell
            masks = _cell_masks(_run_out(root, r["run_id"]))
            if masks:
                by_method[r["seg_method"]] = masks[0]
        methods = sorted(by_method)
        for i in range(len(methods)):
            for k in range(i + 1, len(methods)):
                a, b = methods[i], methods[k]
                try:
                    ma, mb = (
                        np.asarray(reader(by_method[a])),
                        np.asarray(reader(by_method[b])),
                    )
                except Exception:
                    continue
                if ma.shape != mb.shape:
                    continue
                fa, fb = ma > 0, mb > 0
                inter = int(np.logical_and(fa, fb).sum())
                union = int(np.logical_or(fa, fb).sum())
                na, nb = int(ma.max()), int(mb.max())
                inst = instance_f1(ma, mb)  # IoU-matched per-cell agreement
                row = dict(zip(keys, cell if isinstance(cell, tuple) else (cell,)))
                row.update(
                    method_a=a,
                    method_b=b,
                    foreground_iou=(inter / union if union else float("nan")),
                    instance_f1=inst["f1"],
                    instance_precision=inst["precision"],
                    instance_recall=inst["recall"],
                    matched_cells=inst["matched"],
                    n_cells_a=na,
                    n_cells_b=nb,
                    cell_count_ratio=(na / nb if nb else float("nan")),
                )
                rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────── per-run cost ──
def run_cost_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Per-run cost derived from the trace: total compute, CPU-hours, GPU-hours, end-to-end wall-clock
    (if timestamps present), and the bottleneck stage (largest single-process realtime). One row / run."""
    if runs_df.empty:
        return pd.DataFrame()
    df = runs_df.copy()
    df["_leaf"] = df["process"].map(_leaf)
    key = "run_id" if "run_id" in df.columns else "config_id"
    # DERIVED, never hardcoded. This was 14 columns maintained by hand against a
    # sweep with 30+ axes, and it is the carry that actually RUNS -- make_tables'
    # 33-entry CONFIG_COLS was documentation nothing imported. Axes missing from
    # this list do not reach any table, so two runs differing only on them produce
    # the same row: reg_max_image_dim was swept over [2000, 4000, 8000] and absent,
    # and sweep.yaml itself calls it the dominant runtime-and-accuracy axis of
    # VALIS.
    #
    # `if c in df.columns` is kept: a run plan from a DIFFERENT sweep (the arms
    # experiment, or an older results tree) carries a different column set, and a
    # missing column must degrade the table rather than raise.
    wanted = identity_columns(load_sweep(SWEEP_PATH), load_sweep(ARMS_PATH))
    carry = [c for c in wanted if c in df.columns and c != key]
    out = []
    for run, g in df.groupby(key):
        rt = g["realtime_s"].fillna(0)
        cpus = g["cpus"].fillna(1) if "cpus" in g else pd.Series(1, index=g.index)
        gpu = g["_leaf"].isin(GPU_LEAVES)
        row = {key: run}
        for c in carry:
            row[c] = g[c].iloc[0]
        row["total_realtime_s"] = float(rt.sum())
        row["cpu_hours"] = float((rt * cpus).sum() / 3600.0)
        row["gpu_hours"] = float(rt[gpu].sum() / 3600.0)
        # end-to-end wall-clock from timestamps if load parsed them (else NaN)
        if {"start_ts", "complete_ts"} <= set(g.columns):
            s, c = g["start_ts"].dropna(), g["complete_ts"].dropna()
            row["wall_clock_s"] = (
                float((c.max() - s.min()).total_seconds())
                if len(s) and len(c)
                else float("nan")
            )
        else:
            row["wall_clock_s"] = float("nan")
        if len(rt) and rt.max() > 0:
            row["bottleneck_stage"] = g.loc[rt.idxmax(), "_leaf"]
            row["bottleneck_frac"] = (
                float(rt.max() / rt.sum()) if rt.sum() else float("nan")
            )
        out.append(row)
    return pd.DataFrame(out)
