"""Load + join Nextflow trace, size logs and the run plan into a tidy frame.

Generalises notebooks/resource_regression.ipynb:load_resource_data over a results
tree laid out by benchmarks/run_sweep.sh: <root>/<run_id>/trace/trace.txt and
<root>/<run_id>/out/size_logs/input_sizes.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .parsing import parse_duration, parse_to_gb


def parse_trace(trace_txt) -> pd.DataFrame:
    df = pd.read_csv(trace_txt, sep="\t")
    out = pd.DataFrame(
        {
            "process": df["process"],
            "tag": df.get("tag"),
            "status": df.get("status"),
            "exit": pd.to_numeric(df.get("exit"), errors="coerce"),
            "peak_rss_gb": df["peak_rss"].map(parse_to_gb),
            "peak_vmem_gb": df["peak_vmem"].map(parse_to_gb),
            "realtime_s": df["realtime"].map(parse_duration),
            "duration_s": df["duration"].map(parse_duration),
            "cpus": pd.to_numeric(df.get("cpus"), errors="coerce"),
        }
    )
    # I/O volume: rchar/wchar are the cumulative bytes a process moved through read()/write()
    # syscalls (Nextflow trace fields; nextflow.config enables both). This is transferred VOLUME,
    # not throughput. Present-only, so a trace lacking the fields degrades to NaN, never an error.
    # %cpu is the task's MEASURED CPU use (100 = one core busy for its whole runtime), as
    # opposed to `cpus`, the cores it reserved. Present-only, like the I/O fields.
    out["pcpu"] = (
        pd.to_numeric(
            df["%cpu"].astype(str).str.rstrip("%").str.strip(), errors="coerce"
        )
        if "%cpu" in df.columns
        else float("nan")
    )
    for src, dst in (("rchar", "read_gb"), ("wchar", "write_gb")):
        out[dst] = df[src].map(parse_to_gb) if src in df.columns else float("nan")
    # Timestamps (start/complete) enable an end-to-end wall-clock; present only if the trace config
    # includes them. Parsed leniently so a missing/odd format degrades to NaT, never an error.
    for src, dst in (
        ("start", "start_ts"),
        ("complete", "complete_ts"),
        ("submit", "submit_ts"),
    ):
        if src in df.columns:
            out[dst] = pd.to_datetime(df[src], errors="coerce")
    return out


def parse_size_logs(input_sizes_csv) -> pd.DataFrame:
    """Per-process input size in GiB, averaged over the process's tasks.

    ``input_sizes.csv`` has one row per task (e.g. PREPROCESS emits one row per
    channel image). The trace also has one row per task, but there is no shared
    task-unique key to join on (both sides carry only ``process``/``sample_id``),
    so ``input_gb`` is broadcast to every trace row of a process. Averaging — not
    summing — makes that broadcast value the *per-task* input the memory model is
    fit against; summing would inflate x by the task count for multi-task
    processes and bias the regression. Single-task processes (one row) are
    unaffected: mean == the single value.
    """
    df = pd.read_csv(input_sizes_csv)
    agg = df.groupby("process")["bytes"].mean().to_frame()
    agg["input_gb"] = agg["bytes"] / 2**30
    return agg


def only_successful(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows from processes that COMPLETED successfully — a failed/aborted process (a CellSAM
    OOM, a timeout) has partial/bogus peak_rss + realtime that would pollute the fits and means. Uses
    Nextflow ``status`` (COMPLETED/CACHED) when present, else ``exit == 0``; keeps rows with no signal
    (so older traces without those columns are unaffected). measurements.csv still keeps EVERY row
    (with status/exit) so failures remain visible — this filter only feeds the aggregates."""
    if runs_df.empty:
        return runs_df
    ok = pd.Series(True, index=runs_df.index)
    if "status" in runs_df.columns and runs_df["status"].notna().any():
        ok &= runs_df["status"].isna() | runs_df["status"].isin(["COMPLETED", "CACHED"])
    if "exit" in runs_df.columns and runs_df["exit"].notna().any():
        ok &= runs_df["exit"].isna() | (runs_df["exit"] == 0)
    return runs_df[ok]


def aggregate_repeats(
    runs_df: pd.DataFrame,
    metrics=(
        "peak_rss_gb",
        "peak_vmem_gb",
        "realtime_s",
        "duration_s",
        "input_gb",
        "read_gb",
        "write_gb",
    ),
) -> pd.DataFrame:
    """Collapse replicate runs into per-(process, config) mean / std / CV.

    Replicates share a ``config_id`` (from build_run_plan --repeats); this reports
    the spread across them — the variance estimate a single-shot (n=1) sweep can't
    give. Falls back to grouping on ``process`` alone if ``config_id`` is absent.
    For each metric: ``<m>_mean``, ``<m>_std`` (population, ddof=0), ``<m>_cv``
    (std/mean, NaN when mean==0), plus ``n_reps``. Descriptive param columns
    (varied_axis, target_px, n_channels) are carried through when present.
    """
    if runs_df.empty:
        return pd.DataFrame()
    keys = ["process"] + [k for k in ("config_id",) if k in runs_df.columns]
    carry = [
        c for c in ("varied_axis", "target_px", "n_channels") if c in runs_df.columns
    ]
    metrics = [m for m in metrics if m in runs_df.columns]

    out_rows = []
    for key_vals, g in runs_df.groupby(keys):
        key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        row = dict(zip(keys, key_vals))
        row["n_reps"] = len(g)
        for c in carry:
            row[c] = g[c].iloc[0]
        for m in metrics:
            vals = g[m].dropna()
            mean = float(vals.mean()) if len(vals) else float("nan")
            std = float(vals.std(ddof=0)) if len(vals) else float("nan")
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_cv"] = (std / mean) if mean else float("nan")
        out_rows.append(row)
    cols = (
        keys
        + ["n_reps"]
        + carry
        + [f"{m}_{s}" for m in metrics for s in ("mean", "std", "cv")]
    )
    return pd.DataFrame(out_rows, columns=cols)


def load_runs(results_root, run_plan_csv) -> pd.DataFrame:
    """Tidy (run x process) frame from each run's trace + size logs.

    TWO RESULT LAYOUTS ARE SUPPORTED, because the two benchmarks publish
    differently and neither is free to change:

      sweep  <root>/<run_id>/out/size_logs/...   run_sweep.sh isolates each run's
                                                 pipeline outputs under out/
      arms   <root>/<arm>/size_logs/...          run_arms.sh publishes --outdir
                                                 straight to <root>/<arm>, because
                                                 that IS ihc_method's consumer
                                                 contract (<arm>/<patient>/qc/...)

    The arm plan sets run_id to the arm name so both resolve through the same
    <root>/<run_id> lookup. There is deliberately no `manifest` argument: the
    matrix manifest was accepted here for a long time and never read — every
    number comes from the trace and the size logs.
    """
    root = Path(results_root)
    plan = pd.read_csv(run_plan_csv)
    rows = []
    for _, prow in plan.iterrows():
        run_id = prow["run_id"]
        trace = root / run_id / "trace" / "trace.txt"
        sizes = root / run_id / "out" / "size_logs" / "input_sizes.csv"
        if not sizes.exists():
            sizes = root / run_id / "size_logs" / "input_sizes.csv"
        if not trace.exists():
            continue
        t = parse_trace(trace)
        if sizes.exists():
            s = parse_size_logs(sizes)
            t = t.merge(s["input_gb"], left_on="process", right_index=True, how="left")
        else:
            t["input_gb"] = float("nan")
        for col in plan.columns:
            t[col] = prow[col]
        rows.append(t)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return df


# The ground-truth accuracy columns an external landmark table may carry, keyed
# per (pair_id, mode). Only the landmark-derived ones: valis_rtre/stare_* are
# METHOD-NATIVE self-reports (each method scoring its own transform), which is
# a different claim and already has its own table via quality.harvest_valis_rtre.
GROUND_TRUTH_COLS = (
    "true_median_px",
    "true_mean_px",
    "true_p90_px",
    "true_median_rtre",
    "true_median_um",
    "true_p90_um",
)


def load_reg_eval(reg_eval_csv) -> pd.DataFrame:
    """Landmark TRE per registration METHOD, ready to join onto the run table.

    This benchmark ships no producer for a ground-truth registration accuracy
    number; `reg_eval_csv` is an OPTIONAL EXTERNALLY produced table. The column
    contract this function enforces is what a caller needs to know: a CSV with a
    `mode` column naming the registration method, and at least one of
    GROUND_TRUTH_COLS. The table is KEYED ON `mode`; `pair_id` is not read by
    this function — values are medianed per `mode` (non-numeric GROUND_TRUTH_COLS
    are silently dropped by `median(numeric_only=True)`), and `gt_n_pairs` is a
    count of non-null rows per `mode`, not a distinct pair_id count. It used to
    reach nothing:
    `make_figures.run()` took `reg_eval_csv` in its signature, its docstring and
    its call site and never read it in the body, and every test passed None, so
    nothing could detect it.

    KEYED ON METHOD, NOT ON run_id, and that is not a shortcut. The pairs a
    landmark table carries are not sweep runs -- there is no run_id in it and
    there cannot be. So the honest join is "this run used method M, and M's
    measured ground-truth TRE is X", with the median taken across pairs.

    The returned frame is keyed `registration_method` so it merges directly onto
    the run table, and its columns are prefixed `gt_` so a reader cannot mistake
    a per-method constant for a per-run measurement.
    """
    df = pd.read_csv(reg_eval_csv)
    if "mode" not in df.columns:
        raise ValueError(
            f"{reg_eval_csv} has no 'mode' column; it needs a 'mode' column "
            f"naming the registration method"
        )
    present = [c for c in GROUND_TRUTH_COLS if c in df.columns]
    if not present:
        raise ValueError(
            f"{reg_eval_csv} carries none of the ground-truth columns "
            f"{list(GROUND_TRUTH_COLS)}; joining it would add nothing"
        )
    out = df.groupby("mode")[present].median(numeric_only=True).reset_index()
    out["gt_n_pairs"] = df.groupby("mode")[present[0]].count().values
    return out.rename(
        columns={"mode": "registration_method", **{c: f"gt_{c}" for c in present}}
    )
