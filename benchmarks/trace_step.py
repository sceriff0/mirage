#!/usr/bin/env python3
"""Run one command and append its resource usage as a Nextflow trace row.

The ASHLAR external arms (benchmarks/run_ashlar_arm.sh) run OUTSIDE Nextflow, so
nothing wrote a trace for them: they were scored for accuracy into the same
seg_qc tree as every other arm and were simply absent from measurements.csv,
run_cost and resource_stats -- a baseline with no cost column, and no error to
say so. This wrapper is what puts them in. It writes exactly the file
benchmarks.analysis.lib.load.load_runs reads for every other arm,
<root>/<arm>/trace/trace.txt, with the columns parse_trace consumes, so the
analysis layer needs no ashlar-specific path.

    python3 benchmarks/trace_step.py --trace <root>/<arm>/trace/trace.txt \\
        --process ASHLAR_SOLVE --tag <patient> [--input <path> ...] -- <command...>

One row per invocation: process, tag, status (COMPLETED/FAILED), exit, submit/
start/complete timestamps, duration and realtime (wall-clock), %cpu, cpus,
peak_rss, and `-` for what a process outside Nextflow cannot know (peak_vmem,
memory, rchar, wchar -- parse_trace maps `-` to NaN). --input records the input's
size in <root>/<arm>/size_logs/input_sizes.csv, the pipeline's own size-log
format, so the arm gets an input_gb like every other; a directory is summed.

WHY resource.getrusage AND NOT GNU `time -v`. Both read the same kernel counter
(wait4's rusage: ru_maxrss is what `time -v` prints as "Maximum resident set
size"), so the number is the same; the Python path runs wherever the harness
already runs (python3 is required by every launcher) and has one code path to
test instead of a platform branch that a test could only exercise on one
platform. RUSAGE_CHILDREN is the max over every waited-for descendant, so a
`singularity exec ...` prefix is measured through to the process inside the
container, as `time -v` would.

The exit status is the child's, so a wrapped step fails its script exactly as
the bare command did.
"""

from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

COLUMNS = [
    "task_id",
    "process",
    "tag",
    "name",
    "status",
    "exit",
    "submit",
    "start",
    "complete",
    "duration",
    "realtime",
    "%cpu",
    "cpus",
    "memory",
    "peak_rss",
    "peak_vmem",
    "rchar",
    "wchar",
]


def fmt_bytes(n: float) -> str:
    """Nextflow's human form ('1.2 GB'), which parsing.parse_to_gb reads back."""
    for unit, div in (("GB", 2**30), ("MB", 2**20), ("KB", 2**10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{int(n)} B"


def fmt_duration(seconds: float) -> str:
    """Nextflow's human form ('1h 2m 3s' / '1.5s' / '120ms'), parsed by
    parsing.parse_duration."""
    if seconds < 1:
        return f"{int(round(seconds * 1000))}ms"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h >= 1:
        parts.append(f"{int(h)}h")
    if m >= 1:
        parts.append(f"{int(m)}m")
    parts.append(f"{s:.1f}s" if not parts else f"{int(round(s))}s")
    return " ".join(parts)


def fmt_ts(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def path_bytes(p: Path) -> int:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size


def maxrss_bytes(ru) -> int:
    # ru_maxrss is kilobytes on Linux and BYTES on macOS (getrusage(2)).
    return int(ru.ru_maxrss) if sys.platform == "darwin" else int(ru.ru_maxrss) * 1024


def default_cpus() -> int:
    # The allocation the step could use: SLURM's per-task count inside a job,
    # the machine's count otherwise -- the same meaning as Nextflow's `cpus`.
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(var)
        if v and v.isdigit():
            return int(v)
    return os.cpu_count() or 1


def append_row(trace: Path, row: dict) -> None:
    trace.parent.mkdir(parents=True, exist_ok=True)
    existing = 0
    if trace.exists() and trace.stat().st_size > 0:
        with open(trace) as fh:
            existing = sum(1 for _ in fh) - 1
    else:
        trace.write_text("\t".join(COLUMNS) + "\n")
    row = dict(row, task_id=existing + 1)
    with open(trace, "a") as fh:
        fh.write("\t".join(str(row.get(c, "-")) for c in COLUMNS) + "\n")


def append_size_log(size_log: Path, process: str, tag: str, inputs: list[Path]) -> None:
    size_log.parent.mkdir(parents=True, exist_ok=True)
    if not size_log.exists() or size_log.stat().st_size == 0:
        size_log.write_text("process,sample_id,filename,bytes\n")
    with open(size_log, "a") as fh:
        for p in inputs:
            fh.write(f"{process},{tag},{p.name},{path_bytes(p)}\n")


def run(
    trace: Path,
    process: str,
    tag: str,
    command: list[str],
    cpus: int,
    inputs: list[Path] = (),
    size_log: Path | None = None,
) -> int:
    if not command:
        raise SystemExit("trace_step: no command after --")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    submit = time.time()
    t0 = time.perf_counter()
    proc = subprocess.run(command)
    wall = time.perf_counter() - t0
    complete = time.time()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    rc = proc.returncode
    append_row(
        trace,
        {
            "process": process,
            "tag": tag,
            "name": f"{process} ({tag})",
            "status": "COMPLETED" if rc == 0 else "FAILED",
            "exit": rc,
            "submit": fmt_ts(submit),
            "start": fmt_ts(submit),
            "complete": fmt_ts(complete),
            "duration": fmt_duration(wall),
            "realtime": fmt_duration(wall),
            "%cpu": f"{100.0 * cpu / wall:.1f}%" if wall > 0 else "-",
            "cpus": cpus,
            "memory": "-",
            "peak_rss": fmt_bytes(maxrss_bytes(after)),
            "peak_vmem": "-",
            "rchar": "-",
            "wchar": "-",
        },
    )
    if inputs:
        size_log = size_log or trace.parent.parent / "size_logs" / "input_sizes.csv"
        append_size_log(size_log, process, tag, list(inputs))
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run a command and append its resource usage as a Nextflow trace row"
    )
    ap.add_argument(
        "--trace", required=True, type=Path, help="<root>/<arm>/trace/trace.txt"
    )
    ap.add_argument("--process", required=True, help="e.g. ASHLAR_SOLVE")
    ap.add_argument("--tag", required=True, help="the patient")
    ap.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="default: SLURM_CPUS_PER_TASK or cpu_count",
    )
    ap.add_argument(
        "--input",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="record this input's size (a directory is summed) in size_logs/input_sizes.csv",
    )
    ap.add_argument(
        "--size-log",
        type=Path,
        default=None,
        help="default: <trace dir>/../size_logs/input_sizes.csv",
    )
    ap.add_argument("command", nargs=argparse.REMAINDER, help="-- <command...>")
    a = ap.parse_args(argv)
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    return run(
        a.trace,
        a.process,
        a.tag,
        cmd,
        a.cpus if a.cpus is not None else default_cpus(),
        a.input,
        a.size_log,
    )


if __name__ == "__main__":
    sys.exit(main())
