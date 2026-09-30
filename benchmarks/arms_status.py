#!/usr/bin/env python3
"""arms_status.py -- where is the arm benchmark? One screen, from what is on disk + squeue.

    python3 ~/pipelines/mirage/benchmarks/arms_status.py            # the default bench dir
    python3 .../arms_status.py /beegfs/.../benchmark --method stare # one method
    python3 .../arms_status.py --all                                # every run, not only open ones
    python3 .../arms_status.py --watch 120                          # refresh every 2 min

Read-only, standard library only (runs on a login node without the conda env). Per plan
row it combines:

    arm_plan.csv                        which runs exist, their method and kind
    .launch/<dir>/.nextflow/history     the last attempt's status (OK / ERR / '-' = open)
                                        -- a QC cross is logged in its BASE's launch dir
    <arm>/trace/trace.txt               tasks COMPLETED / FAILED / CACHED so far
    .launch/<dir>/code.<run_id>         the commit that ran it
    <arm>/.external_done                ASHLAR's own done marker (no Nextflow history)
    squeue (work dir column)            SLURM jobs in flight, attributed to their run

and prints one status per run:

    DONE         last attempt OK (ASHLAR: marker present)
    RUNNING      its SLURM jobs are in the queue now (R running / P pending)
    ACTIVE?      open attempt, no job queued, but its log moved in the last 20 min
    INTERRUPTED  open attempt ('-') and nothing moving: resume with ARMS_RESUME=1
    FAILED       last attempt ERR: see the log path printed
    WAITING      never launched (its turn has not come, or its upstream is not done)
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_BENCH = "/beegfs/scratch/ieo7660/ihc_method/benchmark"
ACTIVE_WINDOW_S = 20 * 60
ORDER = ["RUNNING", "ACTIVE?", "FAILED", "INTERRUPTED", "WAITING", "DONE"]


def _history(hist: Path, run_id: str) -> dict | None:
    """The last attempt of arms-<run_id> (or arms-<run_id>-rN) in a history file."""
    if not hist.is_file():
        return None
    name, last, attempts = f"arms-{run_id}", None, 0
    for ln in hist.read_text(errors="replace").splitlines():
        f = ln.split("\t")
        if len(f) < 6:
            continue
        if f[2] == name or (
            f[2].startswith(name + "-r") and f[2][len(name) + 2 :].isdigit()
        ):
            attempts += 1
            last = {"time": f[0], "duration": f[1], "name": f[2], "status": f[3]}
    if last:
        last["attempts"] = attempts
    return last


def _last_launch(hist: Path) -> str | None:
    """The run name of the last line in a launch dir's history."""
    lines = [ln for ln in hist.read_text(errors="replace").splitlines() if ln.strip()]
    return (
        lines[-1].split("\t")[2] if lines and len(lines[-1].split("\t")) > 2 else None
    )


def _trace_counts(trace: Path) -> Counter:
    c: Counter = Counter()
    if not trace.is_file():
        return c
    with open(trace, newline="", errors="replace") as fh:
        rows = csv.reader(fh, delimiter="\t")
        header = next(rows, None)
        if not header or "status" not in header:
            return c
        i = header.index("status")
        for r in rows:
            if len(r) > i:
                c[r[i]] += 1
    return c


def _squeue(results: Path) -> dict[str, Counter]:
    """run_id -> Counter of job states, from each queued job's work dir
    (<results>/.launch/<launch dir>/work/..). A cross runs in its base's launch dir, so
    its jobs count under the base: the dir is what SLURM can see."""
    try:
        out = subprocess.run(
            ["squeue", "--me", "-h", "-o", "%T|%Z"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    marker = str(results / ".launch") + "/"
    per: dict[str, Counter] = defaultdict(Counter)
    for ln in out.splitlines():
        state, _, wd = ln.partition("|")
        if marker in wd:
            per[wd.split(marker, 1)[1].split("/", 1)[0]][state] += 1
    return per


def _heads() -> list[str] | None:
    try:
        out = subprocess.run(
            ["squeue", "--me", "-h", "-o", "%i %j %T %M %R"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None  # not on a SLURM host: say so, never read it as "no head job"
    return [
        ln
        for ln in out.splitlines()
        if ln.split()[1:2] and ln.split()[1].startswith("mirage_")
    ]


def _age(p: Path) -> float | None:
    try:
        return time.time() - p.stat().st_mtime
    except OSError:
        return None


def collect(bench: Path, results: Path, plan: Path) -> list[dict]:
    with open(plan, newline="") as fh:
        rows = list(csv.DictReader(fh))
    jobs = _squeue(results)
    out = []
    for r in rows:
        run_id, arm = r["run_id"], r.get("arm") or r["run_id"]
        launch_dir = r.get("resume_run") or run_id
        ld = results / ".launch" / launch_dir
        rec = {
            "run_id": run_id,
            "method": r.get("method") or r.get("registration_method") or "?",
            "kind": r.get("arm_kind", ""),
            "attempts": 0,
            "when": "",
            "tasks": _trace_counts(results / arm / "trace" / "trace.txt"),
            "jobs": Counter(),
            "log": ld / ".nextflow.log",
        }
        if r.get("arm_kind") == "external":
            d = results / arm
            rec["log"] = d / "ashlar.stderr.log"
            if (d / ".external_done").is_file():
                rec["status"] = "DONE"
            elif d.is_dir():
                age = _age(rec["log"])
                rec["status"] = (
                    "ACTIVE?" if age is not None and age < ACTIVE_WINDOW_S else "FAILED"
                )
            else:
                rec["status"] = "WAITING"
            out.append(rec)
            continue
        h = _history(ld / ".nextflow" / "history", run_id)
        if h is None:
            rec["status"] = "WAITING"
        else:
            rec.update(attempts=h["attempts"], when=h["time"])
            if h["status"] == "OK":
                rec["status"] = "DONE"
            elif h["status"] == "ERR":
                rec["status"] = "FAILED"
            elif h["name"] != _last_launch(ld / ".nextflow" / "history"):
                # A base and its crosses share one launch dir (and its jobs and log); only
                # the run launched LAST there can be the one running. An earlier open
                # attempt in the same dir was interrupted.
                rec["status"] = "INTERRUPTED"
            elif jobs.get(launch_dir):
                rec["status"] = "RUNNING"
                rec["jobs"] = jobs[launch_dir]
            else:
                age = _age(rec["log"])
                rec["status"] = (
                    "ACTIVE?"
                    if age is not None and age < ACTIVE_WINDOW_S
                    else "INTERRUPTED"
                )
        out.append(rec)
    return out


def render(bench: Path, results: Path, recs: list[dict], show_all: bool) -> str:
    lines = [f"arm benchmark  {bench}   {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    heads = _heads()
    if heads is None:
        lines.append("head jobs: unknown (squeue not available on this host)")
    else:
        lines.append(
            f"head jobs ({len(heads)}):" if heads else "head jobs: NONE running"
        )
        lines += [f"  {h}" for h in heads]
    queued = Counter()
    for r in recs:
        queued.update(r["jobs"])
    lines.append(
        f"process jobs attributed to runs: {queued.get('RUNNING', 0)} running, "
        f"{queued.get('PENDING', 0)} pending"
    )
    replaced = results / ".replaced"
    if replaced.is_dir():
        lines.append(
            f"moved aside earlier: {len(list(replaced.iterdir()))} batch(es) in {replaced}"
        )
    lines.append("")

    methods = sorted({r["method"] for r in recs})
    by = defaultdict(Counter)
    for r in recs:
        by[r["method"]][r["status"]] += 1
    cols = [s for s in ORDER if any(by[m][s] for m in methods)]
    w = max(10, *(len(m) for m in methods))
    lines.append(
        f"{'method':<{w}} " + " ".join(f"{c:>11}" for c in cols) + f" {'total':>6}"
    )
    for m in methods:
        lines.append(
            f"{m:<{w}} "
            + " ".join(f"{by[m][c] or '':>11}" for c in cols)
            + f" {sum(by[m].values()):>6}"
        )
    tot = Counter()
    for m in methods:
        tot.update(by[m])
    lines.append(
        f"{'ALL':<{w}} "
        + " ".join(f"{tot[c] or '':>11}" for c in cols)
        + f" {sum(tot.values()):>6}"
    )
    done = tot["DONE"] / max(1, sum(tot.values()))
    bar = "#" * int(40 * done) + "." * (40 - int(40 * done))
    lines += ["", f"[{bar}] {100 * done:.0f}% of runs done", ""]

    shown = [r for r in recs if show_all or r["status"] != "DONE"]
    shown.sort(key=lambda r: (ORDER.index(r["status"]), r["method"], r["run_id"]))
    if shown:
        lines.append(
            f"{'status':<12}{'method':<11}{'run':<44}{'try':>4}  {'tasks ok/fail/cached':<22}"
            f"{'jobs R/P':<10}last launch"
        )
        for r in shown:
            t = r["tasks"]
            tasks = f"{t['COMPLETED']}/{t['FAILED']}/{t['CACHED']}" if t else "-"
            j = r["jobs"]
            jobs = f"{j.get('RUNNING', 0)}/{j.get('PENDING', 0)}" if j else "-"
            lines.append(
                f"{r['status']:<12}{r['method']:<11}{r['run_id'][:43]:<44}"
                f"{r['attempts'] or '':>4}  {tasks:<22}{jobs:<10}{r['when']}"
            )
    fails = [r for r in recs if r["status"] == "FAILED"]
    if fails:
        lines += ["", "logs of failed runs:"]
        lines += [f"  {r['run_id']}: {r['log']}" for r in fails[:15]]
        if len(fails) > 15:
            lines.append(f"  ... and {len(fails) - 15} more")
    nxt = []
    if tot["INTERRUPTED"] and heads == []:
        nxt.append(
            "interrupted runs and no head job: resubmit with --export=ALL,ARMS_RESUME=1"
        )
    if tot["FAILED"]:
        nxt.append(
            "failed runs: read the logs above; ARMS_RESUME=1 retries them from cache"
        )
    if nxt:
        lines += ["", "next:"] + [f"  - {n}" for n in nxt]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Status of the arm benchmark (read-only).")
    ap.add_argument(
        "bench", nargs="?", default=os.environ.get("BENCH_DIR", DEFAULT_BENCH)
    )
    ap.add_argument("--results", default=None, help="default: <bench>/arm_results")
    ap.add_argument("--plan", default=None, help="default: <bench>/arm_plan.csv")
    ap.add_argument(
        "--method", default=None, help="only this method (valis, stare, stare, ...)"
    )
    ap.add_argument("--all", action="store_true", help="list DONE runs too")
    ap.add_argument(
        "--watch", type=int, default=0, metavar="SEC", help="refresh every SEC s"
    )
    a = ap.parse_args(argv)
    bench = Path(a.bench)
    results = Path(a.results) if a.results else bench / "arm_results"
    plan = Path(a.plan) if a.plan else bench / "arm_plan.csv"
    if not plan.is_file():
        print(f"no plan at {plan} (has submit_arms.sh run yet?)", file=sys.stderr)
        return 1
    while True:
        recs = collect(bench, results, plan)
        if a.method:
            recs = [r for r in recs if r["method"] == a.method]
        text = render(bench, results, recs, a.all)
        if a.watch:
            print("\033[2J\033[H" + text, flush=True)
            time.sleep(a.watch)
        else:
            print(text)
            return 0


if __name__ == "__main__":
    sys.exit(main())
