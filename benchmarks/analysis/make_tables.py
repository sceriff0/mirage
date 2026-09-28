"""Emit the method-paper DATA tables from a completed sweep — tidy CSVs, no plots.

This is the DATA-first deliverable (the figures/notebook are optional views of the same numbers).
It reads the pipeline's own trace + QC outputs and writes ``<outdir>/`` (default benchmarks/paper_data):

  runs_master.csv            one row per run: input dims + per-stage peak RAM / wall-time + cost
  scaling_fits.csv           resource scaling laws: peak_rss_gb ~ input_gb and realtime_s ~ input_gb,
                             per stage (slope = marginal cost per input GiB, intercept = fixed overhead)
  registration_accuracy.csv  per (run, moving slide, stage): dice_matched + centroid displacement (um),
                             deltas vs the rigid anchor — the segmentation-based registration-accuracy signal
  registration_valis_rtre.csv per (run, slide): VALIS's own feature-based registration error (rTRE / D)
                             from the summary CSVs it writes during register() — the independent, second
                             accuracy signal (also rendered in the pipeline's final QC report)
  segmentation_agreement.csv per method-pair on the same section (sweep cell, or arm patient):
                             instance-F1 + foreground IoU/Dice (stability)
  param_matrix.csv           runs_master joined with the registration + segmentation-quality headlines —
                             every knob's cost AND quality in one wide table (for parameter tuning)

Each CSV is written with a sibling ``<name>.dict.md`` data dictionary. Run:

    python benchmarks/analysis/make_tables.py \
        --results-root bench_runs --run-plan bench_run_plan.csv \
        --outdir benchmarks/paper_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .lib import load, quality, regress

# Runs whose input size genuinely varies — the only rows the scaling fit may use (an OFAT knob
# swept at fixed input piles many points at one x and corrupts the regression). Mirrors make_figures.
SIZE_VARYING_AXES = {
    "baseline",
    "scaling_grid",
    "registration_grid",
    "target_px",
    "n_channels",
}

# The run's IDENTITY -- which config columns reach the tables -- is NOT decided
# here. It is derived from benchmarks/configs/sweep.yaml by
# benchmarks/analysis/lib/contract.py and applied in quality.run_cost_summary,
# whose output is the config half of every table build_runs_master assembles.
#
# A 33-entry tuple used to sit at this spot, documented as exactly that and
# imported by NOTHING -- the only occurrence of its name in the tree was its own
# definition -- while the carry that actually ran was 14 hardcoded columns in
# quality.py. Two swept axes were missing from that one, so runs differing only
# on them produced indistinguishable rows in the paper tables, including
# reg_max_image_dim, which sweep.yaml itself calls the dominant
# runtime-and-accuracy axis of VALIS.
#
# Do not reintroduce a constant here. A second name for the run's identity is
# precisely how that drift started: the authoritative-looking one went stale
# because nothing depended on it.


def _size_varying(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty or "varied_axis" not in runs_df.columns:
        return runs_df
    return runs_df[runs_df["varied_axis"].isin(SIZE_VARYING_AXES)]


def _stage_pivot(
    runs_ok: pd.DataFrame, value: str, suffix: str, aggfunc: str
) -> pd.DataFrame:
    """Wide per-run pivot: one column per pipeline stage, named ``<STAGE>_<suffix>``."""
    if runs_ok.empty or value not in runs_ok.columns:
        return pd.DataFrame(columns=["run_id"])
    df = runs_ok.copy()
    df["_leaf"] = df["process"].map(quality._leaf)
    wide = df.pivot_table(
        index="run_id", columns="_leaf", values=value, aggfunc=aggfunc
    )
    wide.columns = [f"{c}_{suffix}" for c in wide.columns]
    return wide.reset_index()


def build_runs_master(runs_ok: pd.DataFrame) -> pd.DataFrame:
    """One row per run: config + cost (run_cost_summary) + per-stage peak RAM (GiB) and wall-time (s)."""
    cost = quality.run_cost_summary(
        runs_ok
    )  # run_id + config + cpu/gpu-hours + bottleneck
    ram = _stage_pivot(
        runs_ok, "peak_rss_gb", "peak_ram_gb", "max"
    )  # RAM: peak across a stage's tasks
    wall = _stage_pivot(
        runs_ok, "realtime_s", "wall_s", "sum"
    )  # time: summed across a stage's tasks
    master = cost if not cost.empty else pd.DataFrame(columns=["run_id"])
    for extra in (ram, wall):
        if not extra.empty:
            master = master.merge(extra, on="run_id", how="left")
    return master


def build_scaling_fits(runs_ok: pd.DataFrame) -> pd.DataFrame:
    """Resource scaling laws on the size-varying runs: fit each metric ~ input_gb per stage."""
    scaling = _size_varying(runs_ok)
    frames = []
    for target in ("peak_rss_gb", "realtime_s"):
        if scaling.empty or target not in scaling.columns:
            continue
        models = regress.fit_per_process(scaling, predictor="input_gb", target=target)
        f = regress.models_to_frame(models, predictor="input_gb", target=target)
        f.insert(0, "stage", f["process"].map(quality._leaf))
        frames.append(f)
    if not frames:
        return pd.DataFrame(
            columns=[
                "stage",
                "process",
                "predictor",
                "target",
                "slope",
                "intercept",
                "r2",
                "sigma",
                "n",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def build_paper_data(results_root, run_plan_csv, outdir) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs_all = load.load_runs(results_root, run_plan_csv)
    runs_ok = load.only_successful(runs_all)

    master = build_runs_master(runs_ok)
    scaling = build_scaling_fits(runs_ok)
    reg_acc = quality.harvest_registration_qc(results_root, run_plan_csv)
    valis_rtre = quality.harvest_valis_rtre(results_root, run_plan_csv)
    try:
        seg_agree = quality.segmentation_agreement(results_root, run_plan_csv)
    except Exception:
        seg_agree = pd.DataFrame()

    # param_matrix: master + BOTH registration-accuracy headlines (seg-based + VALIS rTRE) +
    # cell count.
    reg_run = quality.registration_accuracy_per_run(reg_acc)
    valis_run = quality.valis_rtre_per_run(valis_rtre)
    try:
        seg_cnt = quality.harvest_segmentation_counts(results_root, run_plan_csv)
    except Exception:
        seg_cnt = pd.DataFrame(columns=["run_id"])
    param_matrix = master if not master.empty else pd.DataFrame(columns=["run_id"])
    for extra in (reg_run, valis_run, seg_cnt):
        if not extra.empty:
            param_matrix = param_matrix.merge(extra, on="run_id", how="left")

    tables = {
        "runs_master": master,
        "scaling_fits": scaling,
        "registration_accuracy": reg_acc,
        "registration_valis_rtre": valis_rtre,
        "segmentation_agreement": seg_agree,
        "param_matrix": param_matrix,
    }
    for name, df in tables.items():
        df.to_csv(outdir / f"{name}.csv", index=False)
        _write_dict(outdir / f"{name}.dict.md", name, df)
    return {"outdir": outdir, "tables": {k: len(v) for k, v in tables.items()}}


# ─────────────────────────────────────────────────────────────── data dictionaries ──
# Curated definitions for the STABLE columns of each table. Dynamic columns (per-stage RAM/time
# pivots) are described by their pattern in the prose, since their exact set
# depends on which stages/methods a given sweep exercised.
_DICTS = {
    "runs_master": (
        "One row per pipeline run: input scale, per-stage resource peaks, and derived cost.",
        [
            ("run_id", "-", "Unique run identifier (one pipeline launch)."),
            (
                "varied_axis",
                "-",
                "Which sweep dimension this run varied (baseline / scaling_grid / an OFAT axis / ...).",
            ),
            ("target_px", "px", "Synthetic input edge length."),
            ("n_channels", "-", "Channels per slide."),
            (
                "n_register_images",
                "-",
                "Slides registered together (1 reference + N-1 moving).",
            ),
            ("registration_method", "-", "Registration method (valis | tiled=STARE)."),
            ("memory_mode", "-", "Registration memory profile (low/medium/high)."),
            (
                "reg_micro_reg",
                "-",
                "Micro-registration depth (0=none | 1=micro-rigid | 2=+micro non-rigid).",
            ),
            (
                "reg_tiled_tile",
                "px",
                "STARE tile core = one REG_TILE task (only when registration_method=tiled).",
            ),
            (
                "reg_tiled_stride",
                "px",
                "STARE vector-lattice stride = mesh resolution; window 2*stride (only when registration_method=tiled).",
            ),
            (
                "reg_tiled_coarse_max_dim",
                "px",
                "STARE coarse-anchor thumbnail, longest side — the resolution M0 is solved at; the counterpart to VALIS reg_max_image_dim (only when registration_method=tiled).",
            ),
            ("seg_method", "-", "Segmentation backend (stardist/cellsam/instantseg)."),
            ("total_realtime_s", "s", "Sum of all process realtimes."),
            (
                "cpu_hours",
                "cpu-h",
                "Sum of realtime x allocated cpus over all processes.",
            ),
            ("gpu_hours", "gpu-h", "Sum of realtime over GPU-leaf processes."),
            (
                "wall_clock_s",
                "s",
                "End-to-end wall clock from first start to last completion (NaN if no timestamps).",
            ),
            (
                "bottleneck_stage",
                "-",
                "Stage with the largest single-process realtime.",
            ),
            ("bottleneck_frac", "-", "That stage's share of total realtime."),
            (
                "<STAGE>_peak_ram_gb",
                "GiB",
                "Peak resident RAM for the named stage (max across its tasks).",
            ),
            ("<STAGE>_wall_s", "s", "Summed realtime for the named stage."),
        ],
    ),
    "scaling_fits": (
        "Resource scaling laws: a LINEAR fit metric ~ input_gb per stage. slope is the marginal cost "
        "per input GiB, intercept the fixed overhead, r2 the fit quality.",
        [
            ("stage", "-", "Pipeline stage (process leaf name)."),
            ("process", "-", "Full Nextflow process name."),
            ("predictor", "-", "Independent variable (input_gb)."),
            (
                "target",
                "-",
                "Dependent variable: peak_rss_gb (RAM law) or realtime_s (time law).",
            ),
            ("slope", "target/GiB", "Marginal target per input GiB."),
            ("intercept", "target", "Fixed overhead at zero input."),
            ("r2", "-", "Coefficient of determination of the fit."),
            ("sigma", "target", "Residual standard deviation (retry-buffer headroom)."),
            ("n", "-", "Number of size-varying runs the fit used."),
        ],
    ),
    "registration_accuracy": (
        "Landmark-free registration accuracy from the staged DAPI-nuclei QC (reg_qc=2). One row per "
        "(run, moving slide, transform stage). Quote dice_matched and displacement_um together; the "
        "*_vs_rigid deltas isolate the registration effect from segmentation noise.",
        [
            ("run_id", "-", "Run identifier."),
            ("patient_id", "-", "Patient."),
            ("moving", "-", "Moving slide (cycle) registered to the reference."),
            (
                "stage",
                "-",
                "Transform stage. The backends do NOT share a vocabulary: VALIS emits "
                "native / rigid / non_rigid / micro, the manifest backends (tiled/STARE, ashlar) emit "
                "native / rigid / refined. Only native and rigid are shared as both a spelling and a "
                "meaning, so each arm is ranked on ITS OWN final stage.",
            ),
            ("n_pairs", "-", "Matched nucleus pairs at this stage."),
            (
                "pair_fraction",
                "-",
                "Fraction of cells that paired (sanity: <~0.5 means a thin, biased match).",
            ),
            ("iou_mean", "-", "Mean per-pair nucleus IoU."),
            ("iou_p50", "-", "Median per-pair nucleus IoU."),
            ("frac_iou_ge_0.5", "-", "Fraction of pairs with IoU >= 0.5."),
            (
                "dice_matched",
                "-",
                "Areal Dice over matched pairs (co-headline; bounded by segmentation agreement).",
            ),
            ("displacement_px_p50", "px", "Median centroid residual displacement."),
            (
                "displacement_px_p90",
                "px",
                "90th-percentile centroid residual displacement.",
            ),
            (
                "displacement_um_p50",
                "um",
                "Median centroid residual displacement (co-headline; physical units).",
            ),
            (
                "displacement_um_p90",
                "um",
                "90th-percentile centroid residual displacement.",
            ),
            (
                "delta_dice_vs_rigid",
                "-",
                "dice_matched at this stage minus the rigid anchor.",
            ),
            (
                "delta_disp_um_p50_vs_rigid",
                "um",
                "displacement_um_p50 minus rigid (negative = improvement).",
            ),
            (
                "delta_disp_px_p50_vs_rigid",
                "px",
                "displacement_px_p50 minus rigid (negative = improvement).",
            ),
        ],
    ),
    "registration_valis_rtre": (
        "VALIS's own registration error, from the summary CSVs it writes during register() (published "
        "to <patient>/registered/summary/ and also shown in the final QC report). Feature-based — an "
        "independent second accuracy signal alongside registration_accuracy.csv. One row per slide; "
        "columns are exactly what VALIS wrote.",
        [
            ("run_id", "-", "Run identifier."),
            ("summary_csv", "-", "Source VALIS summary CSV filename."),
            (
                "original_D",
                "px",
                "Median matched-feature distance BEFORE registration (if VALIS emits it).",
            ),
            ("rigid_D", "px", "Median matched-feature distance after the rigid stage."),
            (
                "non_rigid_D",
                "px",
                "Median matched-feature distance after the non-rigid stage (post-registration).",
            ),
            (
                "<...>rTRE",
                "-",
                "Relative target registration error variants, if present.",
            ),
            ("n_matches", "-", "Number of matched features, if present."),
        ],
    ),
    "segmentation_agreement": (
        "Cross-method segmentation stability on the same section (no ground truth): each method pair "
        "compared by IoU-matched instance-F1 and foreground IoU/Dice. Consensus = mean instance_f1. "
        "Sweep: one row per pair per (target_px, n_channels). Arms: one row per pair per patient, "
        "over the segmentation arms resuming one registration arm (from_arm).",
        [
            ("target_px", "px", "Sweep only: input edge length of the shared cell."),
            ("n_channels", "-", "Sweep only: channels of the shared cell."),
            ("from_arm", "-", "Arms only: the registration arm every compared segmentation resumed."),
            ("patient_id", "-", "Arms only: the patient (whole section) compared."),
            ("method_a", "-", "First segmentation method."),
            ("method_b", "-", "Second segmentation method."),
            ("foreground_iou", "-", "Foreground-mask IoU between the two methods."),
            (
                "foreground_dice",
                "-",
                "Foreground-mask Dice between the two methods, = 2*IoU / (1 + IoU).",
            ),
            (
                "instance_f1",
                "-",
                "IoU-matched (>=0.5) per-cell F1 between the two methods.",
            ),
            ("instance_precision", "-", "Matched cells / method_b cells."),
            ("instance_recall", "-", "Matched cells / method_a cells."),
            ("matched_cells", "-", "Number of IoU-matched cells."),
            ("n_cells_a", "-", "Cell count, method_a."),
            ("n_cells_b", "-", "Cell count, method_b."),
            (
                "cell_count_ratio",
                "-",
                "n_cells_a / n_cells_b (over/under-segmentation).",
            ),
        ],
    ),
    "param_matrix": (
        "runs_master joined with the registration-accuracy and segmentation-quality headlines: every "
        "swept knob's cost AND quality in one wide table, for parameter tuning. Slice by varied_axis.",
        [
            (
                "run_id",
                "-",
                "Run identifier (see runs_master for cost + per-stage columns).",
            ),
            (
                "reg_dice_matched",
                "-",
                "Final-stage matched Dice (median over moving slides).",
            ),
            (
                "reg_displacement_um_p50",
                "um",
                "Final-stage median centroid displacement.",
            ),
            (
                "reg_delta_disp_um_p50_vs_rigid",
                "um",
                "Final-stage displacement improvement vs rigid.",
            ),
            ("reg_delta_dice_vs_rigid", "-", "Final-stage Dice improvement vs rigid."),
            ("reg_pair_fraction", "-", "Final-stage nucleus-pair fraction."),
            (
                "valis_non_rigid_D",
                "px",
                "VALIS post-registration median feature distance (per-run median).",
            ),
            (
                "valis_<col>",
                "varies",
                "Per-run median of each numeric VALIS summary column.",
            ),
            ("n_cells", "-", "Segmented cell count (max over the run's masks)."),
        ],
    ),
}


def _write_dict(path: Path, name: str, df: pd.DataFrame) -> None:
    desc, cols = _DICTS.get(name, ("", []))
    documented = {c for c, _, _ in cols}
    lines = [
        f"# `{name}.csv` — data dictionary",
        "",
        desc,
        "",
        f"Rows in this build: **{len(df)}**. Columns present: **{len(df.columns)}**.",
        "",
        "| Column | Unit | Definition |",
        "|---|---|---|",
    ]
    lines += [f"| `{c}` | {u} | {d} |" for c, u, d in cols]
    # Flag any actual column not covered by a curated definition or a documented `<...>` pattern.
    # Any `<placeholder>` in a documented column name is treated as a wildcard.
    import re

    patterns = [
        re.compile(
            "^"
            + re.sub(
                r"<[^>]+>", ".+", re.escape(c).replace(r"\<", "<").replace(r"\>", ">")
            )
            + "$"
        )
        for c in documented
        if "<" in c
    ]
    undoc = [
        c
        for c in df.columns
        if c not in documented and not any(p.match(c) for p in patterns)
    ]
    if undoc:
        lines += [
            "",
            "Additional columns in this build (see the emitter for provenance): "
            + ", ".join(f"`{c}`" for c in undoc)
            + ".",
        ]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Emit method-paper DATA tables from a completed sweep."
    )
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--run-plan", required=True)
    ap.add_argument("--outdir", default="benchmarks/paper_data")
    a = ap.parse_args()
    res = build_paper_data(a.results_root, a.run_plan, a.outdir)
    print(f"Wrote paper DATA to {res['outdir']}/:")
    for name, n in res["tables"].items():
        print(f"  {name}.csv  ({n} rows)  + {name}.dict.md")


if __name__ == "__main__":
    main()
