"""Headless end-to-end benchmark analysis: parse -> regress -> plot -> emit config.

This is the reproducible artifact the notebook mirrors. Run:
  python -m benchmarks.analysis.make_figures --results-root <dir> \
      --run-plan run_plan.csv \
      --reg-eval reg_eval.csv --outdir benchmarks/analysis/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from .lib import emit_config, load, plotting, quality, regress

# Only these runs change input_gb: the scaling grid (size x channels), the
# registration grid (size x n_register_images — REGISTER's input is the SUM of all
# rounds, and merged channels grow with round count, so this genuinely varies input),
# the baseline cell, and — for back-compat with older OFAT plans — the target_px /
# n_channels axes. Every OTHER axis (memory_mode, seg_gpu, ...) holds input fixed at
# the baseline cell, so pooling them into a peak_rss_gb ~ input_gb regression piles
# many points at one x — inflating sigma, depressing r2, perturbing the intercept.
# The memory-scaling fit must use only the size-varying runs.
SIZE_VARYING_AXES = {
    "baseline",
    "scaling_grid",
    "registration_grid",
    "target_px",
    "n_channels",
}


def _size_varying(runs_df):
    """Rows whose varied_axis changes input size; fall back to all rows if the
    column is absent (older run plans)."""
    if "varied_axis" not in runs_df.columns:
        return runs_df
    return runs_df[runs_df["varied_axis"].isin(SIZE_VARYING_AXES)]


NO_GROUND_TRUTH = "none"

_NO_GT_NOTE = """\
NO GROUND-TRUTH REGISTRATION ACCURACY IN THIS ANALYSIS.

It was run with --reg-eval {marker}, which is the explicit opt-out. Every
accuracy number in this directory is either a COST measure or a method scoring
its own transform (valis_rtre, the STARE intrinsic TRE, the reg_qc=2
segmentation overlap). None of them is independent.

The producer of that landmark TRE table is benchmarks/anhir/ (since
2026-09-12): its evaluate.py writes anhir_reg_eval.csv -- one row per
(pair_id, mode), `mode` naming the registration method, carrying
load.GROUND_TRUTH_COLS -- from the public ANHIR landmarks. It needs the challenge download and a registration run
per backend, so it is not part of the sweep; benchmarks/anhir/README.md is the
run guide and benchmarks/README.md section B the history. Pass that file as
--reg-eval and this note is not written. --reg-eval none remains the explicit
opt-out, not the normal path.

This file exists because the alternative -- a cost-only result that reads like a
complete one -- is how the number came to be missing for so long: it sat unread
in make_figures.run()'s signature.
"""


def _join_ground_truth(runs_df, reg_eval_csv, outdir):
    """Merge per-method landmark TRE onto the run table, or record its absence.

    Returns runs_df unchanged when the opt-out was taken; the marker file is the
    record. Raises when the argument is missing entirely, because "nobody passed
    it" and "we decided not to" must not look the same.
    """
    if reg_eval_csv is None:
        raise ValueError(
            "reg_eval_csv is required. It carries an EXTERNALLY produced landmark "
            "TRE table -- the only ground-truth accuracy measure this analysis can "
            "use, and one this repository cannot produce; running without it gives "
            "a cost-only result that reads like a complete one. Pass --reg-eval "
            f"<path>, or --reg-eval {NO_GROUND_TRUTH} to record deliberately running "
            "without it."
        )
    if str(reg_eval_csv).strip().lower() == NO_GROUND_TRUTH:
        (Path(outdir) / "NO_GROUND_TRUTH.txt").write_text(
            _NO_GT_NOTE.format(marker=NO_GROUND_TRUTH)
        )
        print(
            "[analysis] WARNING: no ground-truth registration accuracy; see "
            f"{outdir}/NO_GROUND_TRUTH.txt"
        )
        return runs_df

    tre = load.load_reg_eval(reg_eval_csv)
    if "registration_method" not in runs_df.columns:
        raise ValueError(
            "the run table has no registration_method column, so the per-method "
            "landmark TRE cannot be joined onto it. Was this run plan built from "
            "a sweep that varies the registration method?"
        )
    merged = runs_df.merge(tre, on="registration_method", how="left")
    matched = merged["gt_n_pairs"].notna().sum() if "gt_n_pairs" in merged else 0
    if not matched:
        print(
            "[analysis] WARNING: reg_eval carries methods "
            f"{sorted(tre['registration_method'])} but the runs use "
            f"{sorted(runs_df['registration_method'].dropna().unique())} -- "
            "no ground-truth row matched any run"
        )
    return merged


def run(
    results_root, run_plan_csv, reg_eval_csv, outdir, formats=("pdf", "svg")
) -> dict:
    outdir = Path(outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plotting.set_paper_theme()

    runs_all = load.load_runs(results_root, run_plan_csv)
    # measurements.csv keeps EVERY row (incl. failed processes, with status/exit) — lossless. But the
    # fits/aggregates/comparisons use only SUCCESSFUL processes: a failed CellSAM/timeout has bogus
    # peak_rss + realtime that would corrupt the means and the scaling model.
    runs_df = load.only_successful(runs_all)
    n_dropped = len(runs_all) - len(runs_df)
    if n_dropped:
        print(
            f"[analysis] excluded {n_dropped} failed/incomplete process rows from the aggregates "
            f"(measurements.csv keeps them)"
        )

    # ── GROUND TRUTH ──────────────────────────────────────────────────────────
    # reg_eval_csv, if passed, carries an OPTIONAL EXTERNALLY produced landmark
    # table -- this benchmark ships no producer for one, and none of its own
    # accuracy numbers are ground-truth. Everything else here is cost, or a
    # method scoring its own transform.
    #
    # It used to sit UNREAD in this signature, its docstring and its call site,
    # with every test passing None -- so the number reached no table and no
    # figure, and a cost-only result read like a complete one. A parameter that
    # looks like a data dependency and is not is worse than an absent one.
    #
    # Required, with an EXPLICIT opt-out ("none"), for the same reason the QC
    # errorStrategy stopped defaulting to 'ignore': running without ground truth
    # has to be a decision somebody wrote down, and it has to be visible in the
    # OUTPUT, not only in an argument nobody passed.
    runs_df = _join_ground_truth(runs_df, reg_eval_csv, outdir)
    # Fit the memory model on the size-varying runs only (avoids the confound).
    scaling_df = _size_varying(runs_df)
    models = regress.fit_per_process(
        scaling_df, predictor="input_gb", target="peak_rss_gb"
    )

    # Tidy CSVs for downstream analysis (R, etc.) — these are the primary data
    # artifacts; figures and the notebook are optional views of the same numbers.
    measurements_csv = outdir / "measurements.csv"
    models_csv = outdir / "resource_models.csv"
    stats_csv = outdir / "resource_stats.csv"
    runs_all.to_csv(
        measurements_csv, index=False
    )  # lossless (all rows, incl. failures)
    regress.models_to_frame(models).to_csv(models_csv, index=False)
    # per-config replicate variance (mean/std/CV across repeats) — empty CSV with
    # no runs, but always written so the artifact set is stable.
    stats_df = load.aggregate_repeats(runs_df)
    stats_df.to_csv(stats_csv, index=False)

    # per-process memory scaling figures (size-varying runs only, matching the fit)
    for proc, g in scaling_df.groupby("process"):
        sub = g[["input_gb", "peak_rss_gb"]].dropna()
        if sub.empty:
            continue
        m = models[proc]
        fig = plotting.scatter_with_fit(
            sub["input_gb"],
            sub["peak_rss_gb"],
            m["slope"],
            m["intercept"],
            xlabel="input (GiB)",
            ylabel="peak RSS (GiB)",
            title=proc,
        )
        plotting.save_fig(fig, figdir / f"scaling_{proc}", formats=formats)

    emit_config.write_optimized_config(models, outdir / "modules.optimized.config")

    # ── QUALITY + COST (the result-quality the trace ignores + derived cost). All best-effort and
    #    robust to missing/failed runs, so a CellSAM run that OOMs simply contributes no rows. ──
    # Per-run cost (cpu/gpu-hours, wall-clock, bottleneck) — from the trace, always available.
    quality.run_cost_summary(runs_df).to_csv(outdir / "run_cost.csv", index=False)
    # Registration accuracy (staged QC, reg_qc=2) — the full per-(run, moving, stage) table, and a
    # per-run headline reduction (final stage, median over moving slides).
    reg_qc_long = quality.harvest_registration_qc(results_root, run_plan_csv)
    reg_qc_long.to_csv(outdir / "registration_accuracy.csv", index=False)
    reg_run = quality.registration_accuracy_per_run(reg_qc_long)
    # VALIS's own feature-based registration error (the independent second accuracy signal).
    quality.harvest_valis_rtre(results_root, run_plan_csv).to_csv(
        outdir / "registration_valis_rtre.csv", index=False
    )
    # Per-run quality join: swept params + registration headline + segmentation cell count.
    per_run = (
        runs_df.drop_duplicates("run_id")[
            [
                c
                for c in (
                    "run_id",
                    "varied_axis",
                    "target_px",
                    "n_channels",
                    "n_register_images",
                    "registration_method",
                    "memory_mode",
                    "reg_micro_reg",
                    "reg_tiled_tile",
                    "reg_tiled_gate_tre",
                    "reg_tiled_coarse_max_dim",
                    "seg_method",
                )
                if c in runs_df.columns
            ]
        ]
        if not runs_df.empty
        else pd.DataFrame(columns=["run_id"])
    )
    try:
        seg_cnt = quality.harvest_segmentation_counts(
            results_root, run_plan_csv
        )  # reads masks (I/O)
    except Exception:
        seg_cnt = pd.DataFrame(columns=["run_id"])
    quality_df = per_run
    for extra in (reg_run, seg_cnt):
        if not extra.empty:
            quality_df = quality_df.merge(extra, on="run_id", how="left")
    # Ground-truth accuracy against cost, per method. The point of having the
    # landmark TRE at all: a method that is 3x cheaper and 3x less accurate is a
    # different answer from one that is 3x cheaper for free, and neither the cost
    # tables nor the method-native self-reports can tell them apart.
    gt_cols = [c for c in runs_df.columns if c.startswith("gt_true_")]
    if gt_cols and "registration_method" in runs_df.columns:
        cost = quality.run_cost_summary(runs_df)
        if not cost.empty and "cpu_hours" in cost.columns:
            gt = runs_df.drop_duplicates("run_id")[
                ["run_id", "registration_method"] + gt_cols
            ]
            acc = cost.merge(gt, on="run_id", how="left", suffixes=("", "_gt"))
            acc.to_csv(outdir / "accuracy_vs_cost.csv", index=False)
            metric = (
                "gt_true_median_um" if "gt_true_median_um" in gt_cols else gt_cols[0]
            )
            sub = acc[["cpu_hours", metric, "registration_method"]].dropna(
                subset=["cpu_hours", metric]
            )
            if not sub.empty:
                fig = plotting.scatter(
                    sub["cpu_hours"],
                    sub[metric],
                    xlabel="CPU-hours per run",
                    ylabel=f"landmark TRE ({metric.replace('gt_true_', '')})",
                    title="Ground-truth accuracy vs cost",
                    labels=sub["registration_method"],
                )
                plotting.save_fig(fig, figdir / "accuracy_vs_cost", formats=formats)

    quality_df.to_csv(outdir / "quality.csv", index=False)
    # Segmentation cross-method agreement (pairwise mask IoU + cell-count ratio) — best-effort.
    try:
        seg_agree = quality.segmentation_agreement(results_root, run_plan_csv)
    except Exception:
        seg_agree = pd.DataFrame()
    seg_agree.to_csv(outdir / "segmentation_agreement.csv", index=False)

    return {
        "runs_df": runs_df,
        "models": models,
        "outdir": outdir,
        "measurements_csv": measurements_csv,
        "models_csv": models_csv,
        "stats_csv": stats_csv,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark analysis: figures + optimal config."
    )
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--run-plan", required=True)
    ap.add_argument(
        "--reg-eval",
        required=True,
        help=(
            "an EXTERNALLY produced landmark TRE table, one row per "
            "(pair_id, mode) — this repository ships no producer for it. Keyed "
            "on `mode` (the registration method) and carrying at least one of "
            f"load.GROUND_TRUTH_COLS. Pass '{NO_GROUND_TRUTH}' to run "
            "deliberately without it; that writes a NO_GROUND_TRUTH.txt into "
            "the output so the omission is visible in the deliverable rather "
            "than only in the command line."
        ),
    )
    ap.add_argument("--outdir", default="benchmarks/analysis")
    a = ap.parse_args()
    res = run(a.results_root, a.run_plan, a.reg_eval, a.outdir)
    print(
        f"Wrote {res['measurements_csv']} ({len(res['runs_df'])} rows), "
        f"{res['models_csv']} ({len(res['models'])} processes), "
        f"{res['stats_csv']} (per-config variance), "
        f"{res['outdir']}/modules.optimized.config, and figures"
    )


if __name__ == "__main__":
    main()
