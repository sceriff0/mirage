"""Resource profile of ONE real MIRAGE run: CPU, wall-time, peak RSS, against input size.

The synthetic sweep prices how cost SCALES; the arm benchmark prices a cohort under
the shipped configuration. Neither answers the question an operator asks after an
ordinary run -- "what did THIS run of MY slides cost, per process, and was the memory
we reserved anywhere near what was used?" -- so this module turns the two files every
traced run already writes into tidy tables the `ihc_method` page `run_resources.Rmd`
plots:

    <trace_dir>/trace.txt              Nextflow's per-task trace (realtime, %cpu, cpus,
                                       memory requested, peak_rss, peak_vmem, rchar, wchar)
    <outdir>/size_logs/input_sizes.csv per-task input bytes, written by every process's
                                       ProcessEnvelope.sizeLog (enable_size_logs)
    <outdir>/qc/run_summary.json       run identity (optional)

Parsing is NOT reimplemented here. `bin/generate_resource_report.py` -- the script
main.nf's onComplete runs to render `qc/mirage_resource_report.html` -- already owns
the trace/size-log grammar (byte and duration units, the `-` not-run sentinel, the
(process, tag) join with its prefix fallback). It is stdlib-only and importable by
path, so this module loads it and adds only what the HTML report does not have: a
per-task CSV, per-process rollups with CPU-hours and memory utilisation, and per-process
linear fits of peak RSS and wall-time against input size (the same
`regress.fit_memory_model` the sweep uses, so the two fits are comparable).

ONE RUN, by design. A fit inside one run uses the run's own tasks as points -- a
process that ran once per slide over a cohort of eight has eight (input, RSS) pairs;
a process that ran once has one, and gets no fit (`n < 3` -> flat fallback, flagged
by `fit_ok = False`). Pooling runs would mix configurations; the sweep is where that
comparison belongs.

Outputs, all beside each other in --outdir (a `.dict.md` documents every column):

    run_resources_tasks.csv       one row per trace task
    run_resources_processes.csv   one row per process
    run_resources_fits.csv        one row per process x target (peak_rss_gb, realtime_s)
    run_resources_summary.csv     one row: the whole run

CLI:  python -m benchmarks.analysis.run_resources --run <outdir> [--trace <trace.txt>] --outdir <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd

from .lib.regress import fit_memory_model

REPO = Path(__file__).resolve().parents[2]
_GIB = 1024**3

TASK_COLUMNS = (
    "process",
    "tag",
    "status",
    "exit",
    "failed",
    "cpus",
    "realtime_s",
    "duration_s",
    "cpu_pct",
    "cpu_h_used",
    "cpu_h_reserved",
    "memory_req_gb",
    "peak_rss_gb",
    "peak_vmem_gb",
    "rss_utilisation",
    "gb_h_reserved",
    "read_gb",
    "write_gb",
    "input_gb",
)

# What each column means, for the .dict.md. Kept as data so the dictionary and the
# frame cannot drift: test_run_resources asserts every emitted column is described.
COLUMN_DOCS = {
    "process": "Nextflow process name, as the trace records it (no workflow prefix).",
    "tag": "the task's `tag` (patient / slide / cell id) -- the join key to the size log.",
    "status": "Nextflow task status (COMPLETED, FAILED, CACHED, ...).",
    "exit": "exit code as recorded; `-` (not run / cached) and empty are kept verbatim.",
    "failed": "True when `exit` is a non-zero code (the trace's `-` sentinel is not a failure).",
    "cpus": "CPUs requested for the task.",
    "realtime_s": "wall-clock seconds the task actually ran (trace `realtime`).",
    "duration_s": "seconds from submission to completion, queue wait included (trace `duration`).",
    "cpu_pct": "trace `%cpu`: mean CPU utilisation over the task's lifetime; 100 = one core busy.",
    "cpu_h_used": "CPU-hours actually consumed: realtime_s x cpu_pct / 100 / 3600.",
    "cpu_h_reserved": "CPU-hours reserved: realtime_s x cpus / 3600. The gap to cpu_h_used is idle reservation.",
    "memory_req_gb": "memory requested for the task (GiB); the scheduler's reservation.",
    "peak_rss_gb": "peak resident set size (GiB): the memory the task really used.",
    "peak_vmem_gb": "peak virtual memory (GiB).",
    "rss_utilisation": "peak_rss_gb / memory_req_gb; 0.25 means three quarters of the reservation sat idle.",
    "gb_h_reserved": "memory-hours reserved: memory_req_gb x realtime_s / 3600 (peak_rss when no request was recorded).",
    "read_gb": "bytes read through read() syscalls (trace `rchar`), GiB. Volume, not throughput.",
    "write_gb": "bytes written through write() syscalls (trace `wchar`), GiB.",
    "input_gb": "the task's input bytes from size_logs/input_sizes.csv, joined on (process, tag) with the report's prefix fallback; NaN when the size log has no row for it.",
    # process rollup
    "n_tasks": "tasks of this process in the trace.",
    "n_failed": "tasks with a non-zero exit.",
    "realtime_total_h": "sum of realtime_s over the process's tasks, in hours.",
    "realtime_max_h": "the longest single task, in hours -- the critical-path contribution if tasks ran in parallel.",
    "wall_share": "realtime_total_h / the run's total realtime hours: this process's share of all task time.",
    "cpu_h_used_total": "sum of cpu_h_used.",
    "cpu_h_reserved_total": "sum of cpu_h_reserved.",
    "cpu_efficiency": "cpu_h_used_total / cpu_h_reserved_total.",
    "peak_rss_max_gb": "largest peak_rss_gb among the process's tasks.",
    "memory_req_max_gb": "largest memory request among them.",
    "rss_utilisation_max": "largest per-task rss_utilisation -- how close the tightest task came to its reservation.",
    "gb_h_reserved_total": "sum of gb_h_reserved.",
    "input_gb_total": "sum of input_gb over tasks with a size-log row.",
    "input_gb_mean": "mean input_gb over those tasks.",
    "read_gb_total": "sum of read_gb.",
    "write_gb_total": "sum of write_gb.",
    # fits
    "target": "the fitted quantity: peak_rss_gb or realtime_s.",
    "predictor": "always input_gb.",
    "slope": "target per GiB of input (least squares).",
    "intercept": "target at zero input.",
    "r2": "coefficient of determination; NaN when n < 3.",
    "sigma": "residual standard deviation (the sweep's retry-headroom unit).",
    "n": "points the fit used: tasks of the process with both target and input_gb present.",
    "fit_ok": "True when n >= 3 AND the inputs span more than one distinct size; otherwise slope/intercept are the flat fallback and must not be read as a scaling law.",
    # summary
    "run": "run label: the --run directory's basename (run_summary.json records no run name).",
    "mirage_version": "pipeline version from run_summary.json, when present.",
    "n_processes": "distinct processes in the trace.",
    "wall_clock_h": "end-to-end hours from the earliest task start to the latest completion (needs `start`/`complete` in the trace; NaN otherwise).",
    "task_realtime_total_h": "sum of every task's realtime, in hours -- the serial cost.",
    "raw_input_gb": "total bytes CONVERT_IMAGE read, i.e. the run's raw slide input, GiB; NaN when the size log has no CONVERT_IMAGE rows.",
}


def _load_report_module():
    """Import bin/generate_resource_report.py by path (bin/ is not a package)."""
    path = REPO / "bin" / "generate_resource_report.py"
    spec = importlib.util.spec_from_file_location("mirage_resource_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_trace(run_dir: Path, explicit: str | None) -> Path | None:
    """Locate trace.txt: an explicit path wins; otherwise the layouts a run can have.

    `trace_dir` defaults to `.trace` beside the LAUNCH directory, independent of
    --outdir, so a bare run keeps its trace next to where `nextflow run` was typed
    (searched as `<outdir>/../.trace`); the benchmark launchers point it INTO the
    results tree (`<run>/trace/`). The current directory is deliberately NOT
    searched: a stale `.trace/` beside whoever runs this tool would profile the wrong
    run and look right (the first test run did exactly that).
    """
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for cand in (
        run_dir / "trace" / "trace.txt",
        run_dir / ".trace" / "trace.txt",
        run_dir.parent / ".trace" / "trace.txt",
    ):
        if cand.exists():
            return cand
    return None


def find_size_log(run_dir: Path) -> Path | None:
    for cand in (
        run_dir / "size_logs" / "input_sizes.csv",
        run_dir / "out" / "size_logs" / "input_sizes.csv",
    ):
        if cand.exists():
            return cand
    return None


def _gb(b):
    return float("nan") if b is None else b / _GIB


def tasks_frame(trace_rows, size_map, report_mod) -> pd.DataFrame:
    """One row per trace task, with derived CPU-hours, utilisation and joined input size."""
    joined = report_mod.join_size(trace_rows, size_map)
    rows = []
    for r in joined:
        rt = r.get("realtime_s")
        cpus = r.get("cpus")
        pct = r.get("cpu_pct")
        req = r.get("memory_b")
        rss = r.get("peak_rss_b")
        rows.append(
            {
                "process": r["process"],
                "tag": r.get("tag", ""),
                "status": r.get("status", ""),
                "exit": r.get("exit", ""),
                "failed": report_mod._is_failure(r),
                "cpus": cpus,
                "realtime_s": rt,
                "duration_s": r.get("duration_s"),
                "cpu_pct": pct,
                "cpu_h_used": (rt * pct / 100.0 / 3600.0)
                if rt is not None and pct is not None
                else float("nan"),
                "cpu_h_reserved": (rt * cpus / 3600.0)
                if rt is not None and cpus is not None
                else float("nan"),
                "memory_req_gb": _gb(req),
                "peak_rss_gb": _gb(rss),
                "peak_vmem_gb": _gb(r.get("peak_vmem_b")),
                "rss_utilisation": (rss / req)
                if rss is not None and req
                else float("nan"),
                "gb_h_reserved": report_mod._reserved_gb_h(r),
                "read_gb": _gb(r.get("rchar_b")),
                "write_gb": _gb(r.get("wchar_b")),
                "input_gb": _gb(r.get("input_bytes")),
            }
        )
    df = pd.DataFrame(rows, columns=list(TASK_COLUMNS))
    return df


def processes_frame(tasks: pd.DataFrame) -> pd.DataFrame:
    """Per-process rollup of the task frame."""
    total_rt_h = tasks["realtime_s"].sum(min_count=1) / 3600.0
    out = []
    for proc, g in tasks.groupby("process", sort=True):
        rt_h = g["realtime_s"].sum(min_count=1) / 3600.0
        used = g["cpu_h_used"].sum(min_count=1)
        reserved = g["cpu_h_reserved"].sum(min_count=1)
        out.append(
            {
                "process": proc,
                "n_tasks": int(len(g)),
                "n_failed": int(g["failed"].sum()),
                "realtime_total_h": rt_h,
                "realtime_max_h": g["realtime_s"].max() / 3600.0,
                "wall_share": (rt_h / total_rt_h)
                if total_rt_h and not math.isnan(total_rt_h)
                else float("nan"),
                "cpu_h_used_total": used,
                "cpu_h_reserved_total": reserved,
                "cpu_efficiency": (used / reserved)
                if reserved and not math.isnan(reserved)
                else float("nan"),
                "peak_rss_max_gb": g["peak_rss_gb"].max(),
                "memory_req_max_gb": g["memory_req_gb"].max(),
                "rss_utilisation_max": g["rss_utilisation"].max(),
                "gb_h_reserved_total": g["gb_h_reserved"].sum(),
                "input_gb_total": g["input_gb"].sum(min_count=1),
                "input_gb_mean": g["input_gb"].mean(),
                "read_gb_total": g["read_gb"].sum(min_count=1),
                "write_gb_total": g["write_gb"].sum(min_count=1),
            }
        )
    return pd.DataFrame(out)


FIT_TARGETS = ("peak_rss_gb", "realtime_s")


def fits_frame(tasks: pd.DataFrame) -> pd.DataFrame:
    """Per process x target: target ~ input_gb over the run's own tasks.

    `fit_ok` is the honesty flag. `fit_memory_model` returns a flat fallback below
    three points, and three tasks of the SAME size (one slide, three attempts) give a
    degenerate design a least-squares call will still happily solve; both are
    reported with fit_ok=False so the page draws the points but not a line.
    """
    rows = []
    for proc, g in tasks.groupby("process", sort=True):
        for target in FIT_TARGETS:
            sub = g[["input_gb", target]].dropna()
            n_distinct = sub["input_gb"].round(9).nunique()
            model = fit_memory_model(sub["input_gb"].to_numpy(), sub[target].to_numpy())
            ok = model["n"] >= 3 and n_distinct >= 2
            rows.append(
                {
                    "process": proc,
                    "target": target,
                    "predictor": "input_gb",
                    "slope": model["slope"] if ok else float("nan"),
                    "intercept": model["intercept"] if ok else float("nan"),
                    "r2": model["r2"] if ok else float("nan"),
                    "sigma": model["sigma"] if ok else float("nan"),
                    "n": int(model["n"]),
                    "fit_ok": bool(ok),
                }
            )
    return pd.DataFrame(rows)


def _wall_clock_h(trace_path: Path) -> float:
    """Earliest start -> latest complete, from the raw trace; NaN without timestamps."""
    try:
        raw = pd.read_csv(
            trace_path, sep="\t", usecols=lambda c: c in ("start", "complete")
        )
    except ValueError:
        return float("nan")
    if "start" not in raw or "complete" not in raw:
        return float("nan")
    # `-` is Nextflow's not-run sentinel; mask it first so an all-sentinel column is
    # simply empty rather than a dateutil fallback with a warning per row.
    start = pd.to_datetime(raw["start"].where(raw["start"] != "-"), errors="coerce")
    end = pd.to_datetime(raw["complete"].where(raw["complete"] != "-"), errors="coerce")
    if start.isna().all() or end.isna().all():
        return float("nan")
    return (end.max() - start.min()).total_seconds() / 3600.0


def summary_frame(
    tasks: pd.DataFrame, run_label: str, version: str, wall_clock_h: float
) -> pd.DataFrame:
    conv = tasks.loc[tasks["process"] == "CONVERT_IMAGE", "input_gb"]
    return pd.DataFrame(
        [
            {
                "run": run_label,
                "mirage_version": version,
                "n_tasks": int(len(tasks)),
                "n_processes": int(tasks["process"].nunique()),
                "n_failed": int(tasks["failed"].sum()),
                "wall_clock_h": wall_clock_h,
                "task_realtime_total_h": tasks["realtime_s"].sum(min_count=1) / 3600.0,
                "cpu_h_used_total": tasks["cpu_h_used"].sum(min_count=1),
                "cpu_h_reserved_total": tasks["cpu_h_reserved"].sum(min_count=1),
                "gb_h_reserved_total": tasks["gb_h_reserved"].sum(),
                "peak_rss_max_gb": tasks["peak_rss_gb"].max(),
                "raw_input_gb": conv.sum(min_count=1) if len(conv) else float("nan"),
            }
        ]
    )


def _run_identity(run_dir: Path) -> tuple[str, str]:
    summary = run_dir / "qc" / "run_summary.json"
    label, version = run_dir.name, ""
    if summary.exists():
        try:
            d = json.loads(summary.read_text())
        except json.JSONDecodeError:
            return label, version
        # buildRunSummary (subworkflows/local/final_qc.nf) writes pipeline.{name,version}
        # and run.{timestamp,mode,start,stop} -- no run name, so the directory is the label.
        if isinstance(d, dict):
            version = str((d.get("pipeline") or {}).get("version") or "")
    return label, version


def write_dictionary(outdir: Path, frames: dict[str, pd.DataFrame]) -> Path:
    lines = [
        "# run_resources -- data dictionary",
        "",
        "One real MIRAGE run's resource profile, from Nextflow's trace.txt joined with",
        "size_logs/input_sizes.csv. Produced by `python -m benchmarks.analysis.run_resources`.",
        "",
    ]
    for name, df in frames.items():
        lines += [f"## {name}.csv", "", "| column | meaning |", "|---|---|"]
        for col in df.columns:
            doc = COLUMN_DOCS.get(col)
            if doc is None:
                raise KeyError(f"{name}.{col} has no entry in COLUMN_DOCS")
            lines.append(f"| `{col}` | {doc} |")
        lines.append("")
    path = outdir / "run_resources.dict.md"
    path.write_text("\n".join(lines))
    return path


def profile_run(run_dir: Path, trace: str | None, outdir: Path) -> dict[str, Path]:
    """Read one run, write the four CSVs + dictionary, return their paths.

    Raises FileNotFoundError when no trace can be found and ValueError when the
    trace has a header but no tasks -- a run launched without enable_trace, or one
    that never started a process, must not produce an empty-but-plausible table.
    """
    report_mod = _load_report_module()
    trace_path = find_trace(run_dir, trace)
    if trace_path is None:
        raise FileNotFoundError(
            f"no trace.txt for {run_dir}: pass --trace, or run the pipeline with "
            "enable_trace (the default) and point --trace at <trace_dir>/trace.txt"
        )
    trace_rows = report_mod.parse_trace(trace_path)
    if not trace_rows:
        raise ValueError(f"{trace_path} has no task rows -- nothing to profile")
    size_log = find_size_log(run_dir)
    size_map = report_mod.parse_size_log(size_log) if size_log else {}

    tasks = tasks_frame(trace_rows, size_map, report_mod)
    procs = processes_frame(tasks)
    fits = fits_frame(tasks)
    label, version = _run_identity(run_dir)
    summary = summary_frame(tasks, label, version, _wall_clock_h(trace_path))

    outdir.mkdir(parents=True, exist_ok=True)
    frames = {
        "run_resources_tasks": tasks,
        "run_resources_processes": procs,
        "run_resources_fits": fits,
        "run_resources_summary": summary,
    }
    paths = {}
    for name, df in frames.items():
        p = outdir / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    paths["dictionary"] = write_dictionary(outdir, frames)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="a MIRAGE run's --outdir")
    ap.add_argument(
        "--trace", default=None, help="trace.txt (default: searched, see find_trace)"
    )
    ap.add_argument(
        "--outdir", required=True, help="where the CSVs and .dict.md are written"
    )
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        print(f"--run {run_dir} is not a directory", file=sys.stderr)
        return 2
    try:
        paths = profile_run(run_dir, args.trace, Path(args.outdir))
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    summary = pd.read_csv(paths["run_resources_summary"])
    row = summary.iloc[0]
    print(
        f"run {row['run']}: {row['n_tasks']} tasks / {row['n_processes']} processes, "
        f"{row['n_failed']} failed; task realtime {row['task_realtime_total_h']:.2f} h, "
        f"CPU-h used {row['cpu_h_used_total']:.2f} of {row['cpu_h_reserved_total']:.2f} reserved, "
        f"peak RSS {row['peak_rss_max_gb']:.1f} GiB, raw input {row['raw_input_gb']:.2f} GiB"
    )
    for p in paths.values():
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
