"""``regbench status``: where a results folder stands, for the operator. READ-ONLY.

Per run and result name (variant) and dataset, against what the cohort must have (every
unit of the case list, ``variants.runs_for``): how many results are ``ok``, how many of
those this submission REUSED from an earlier one, how many failed (by cause), and how
many are MISSING (no ``run.json``). The jobs of the results folder still in the queue
are counted when ``squeue`` can be asked (a node without it says so). Chains no link
of whose watcher finished (``chains.unfinished``) are named. The last line says what to
do next: wait for the queue, submit again into the same folder (every ok unit is
reused), or nothing.

    python -m benchmarks.regbench status --out O [--cases C] [--dataset D]
                                         [--submission S] [--csv]

``--csv`` prints the table as CSV (one row per run, variant and dataset), for a
spreadsheet; the default is an aligned text table. ``--summary-to FILE`` appends the
unfinished-chains block to the scorer's ``summary.md`` (score_cohort.sbatch) and prints
nothing else.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

COLUMNS = (
    "run", "variant", "dataset", "expected", "ok", "reused", "failed", "missing",
    "causes",
)  # fmt: skip


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _submission(out, given=None):
    if given:
        return given
    try:
        return Path(out, "logs", "submission").read_text().split()[0]
    except (OSError, IndexError):
        return ""


def _reused(rec, submission):
    entries = rec.get("reused_by") or []
    if not entries:
        return False
    if submission:
        return any((e or {}).get("submission") == submission for e in entries)
    return True


def tally(out, cases_root, dataset="multiplex", submission="", runs=None):
    """``[row]``, one per run, variant and dataset (``COLUMNS``), and
    ``{run name: [unit index, ...]}`` of the units that are not all ok. ``runs``: only
    these run names (default: every active run)."""
    from . import ladder, variants
    from .chains import _units

    rows, todo = [], {}
    for run in variants.runs_for(None):
        name = variants.run_name(run)
        if runs and name not in runs:
            continue
        units = _units(cases_root, dataset, run.method)
        by_variant = {
            v: {
                "ok": 0,
                "reused": 0,
                "failed": 0,
                "missing": 0,
                "expected": 0,
                "causes": {},
            }  # fmt: skip
            for v in variants.variants_of(run)
        }
        for index, cases in enumerate(units):
            unit_ok = True
            for path in ladder.unit_paths(out, run.method, run.label, cases):
                variant = Path(path).parents[2].name
                c = by_variant.setdefault(
                    variant, {"ok": 0, "reused": 0, "failed": 0, "missing": 0,
                              "expected": 0, "causes": {}},
                )  # fmt: skip
                c["expected"] += 1
                rec = _read(path)
                if not isinstance(rec, dict):
                    c["missing"] += 1
                    unit_ok = False
                    continue
                if rec.get("ok"):
                    c["ok"] += 1
                    c["reused"] += _reused(rec, submission)
                    continue
                cause = str(rec.get("cause") or "exception")
                c["failed"] += 1
                c["causes"][cause] = c["causes"].get(cause, 0) + 1
                unit_ok = False
            if not unit_ok:
                todo.setdefault(name, []).append(index)
        for variant, c in by_variant.items():
            rows.append(
                {
                    "run": name,
                    "variant": variant,
                    "dataset": dataset,
                    **{
                        k: c[k]
                        for k in ("expected", "ok", "reused", "failed", "missing")
                    },
                    "causes": ";".join(
                        f"{k}:{v}" for k, v in sorted(c["causes"].items())
                    ),
                }
            )
    return rows, todo


def queued(out):
    """``(jobs, note)``: the jobs of this results folder still in the queue
    (``[(job id, name, state)]`` of the names its ``logs/jobs.tsv`` lists), or None
    with why they were not read."""
    if not shutil.which("squeue"):
        return None, "not read (no squeue on this node)"
    try:
        names = {
            line.split("\t")[1]
            for line in Path(out, "logs", "jobs.tsv").read_text().splitlines()
            if line.count("\t") >= 2
        }
    except OSError:
        names = set()
    user = os.environ.get("USER") or ""
    argv = ["squeue", "-h", "-o", "%i|%j|%T"] + (["-u", user] if user else [])
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"not read (squeue: {e})"
    if done.returncode != 0:
        return (
            None,
            f"not read (squeue exited {done.returncode}: {done.stderr.strip()})",
        )
    jobs = []
    for line in done.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3 and (
            parts[1] in names or (not names and parts[1].startswith("rb-"))
        ):
            jobs.append(tuple(parts))
    return jobs, ""


def report(out, cases_root, datasets=("multiplex",), submission="", as_csv=False,
           queue=True, runs=None):  # fmt: skip
    """The report as text (what ``regbench status`` prints)."""
    from . import chains

    submission = _submission(out, submission)
    rows, todo = [], {}
    for ds in datasets:
        got, left = tally(out, cases_root, ds, submission, runs)
        rows += got
        for k, v in left.items():
            todo.setdefault(k, []).extend(v)
    buf = io.StringIO()
    if as_csv:
        w = csv.DictWriter(buf, fieldnames=list(COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue()
    say = buf.write
    say(
        f"# results: {out}  (cases {cases_root}; submission {submission or 'none recorded'})\n"
    )
    widths = {
        c: max(len(c), *(len(str(r[c])) for r in rows)) if rows else len(c)
        for c in COLUMNS
    }
    say("  ".join(c.ljust(widths[c]) for c in COLUMNS).rstrip() + "\n")
    for r in rows:
        say("  ".join(str(r[c]).ljust(widths[c]) for c in COLUMNS).rstrip() + "\n")
    total = {
        k: sum(r[k] for r in rows)
        for k in ("expected", "ok", "reused", "failed", "missing")
    }
    say(
        f"# {total['expected']} result(s) expected: {total['ok']} ok "
        f"({total['reused']} reused), {total['failed']} failed, {total['missing']} missing\n"
    )
    jobs, note = queued(out) if queue else (None, "not read (--no-queue)")
    if jobs is None:
        say(f"# queue: {note}\n")
    else:
        say(f"# queue: {len(jobs)} job(s) of this folder still queued\n")
        for job_id, name, state in jobs[:20]:
            say(f"#   {job_id} {name} {state}\n")
        if len(jobs) > 20:
            say(f"#   ... and {len(jobs) - 20} more\n")
    left = chains.unfinished(out, submission) if submission else []
    for plan in left:
        say(
            f"# WARNING chain not finished by any link of its watcher: {plan.name} ({plan})\n"
        )
    if jobs:
        say(
            f"# next: wait; {len(jobs)} job(s) are still queued (squeue), then run this again\n"
        )
    elif todo:
        runs = " ".join(sorted(todo))
        n = sum(len(v) for v in todo.values())
        say(
            f"# next: {n} unit(s) of {len(todo)} run(s) are not all ok: submit again into "
            f"the same folder, which reuses every ok unit: REGBENCH_OUT={out} "
            f'submit_cohort.sh --runs "{runs}"\n'
        )
    else:
        say("# next: nothing; every expected result is ok\n")
    return buf.getvalue()


def main(a):
    from . import chains

    if getattr(a, "summary_to", None):
        lines = chains.summary_lines(a.out, _submission(a.out, a.submission))
        if lines:
            with open(a.summary_to, "a") as fh:
                fh.write("\n" + "\n".join(lines) + "\n")
        return 0
    cases_root = a.cases or os.environ.get("REGBENCH_CASES") or ""
    if not cases_root or not Path(cases_root).is_dir():
        print(
            f"status: no cases folder ({cases_root or 'neither --cases nor REGBENCH_CASES'}"
            "): the expected results are read off the case list",
            file=sys.stderr,
        )
        return 2
    datasets = (a.dataset,) if a.dataset else ("multiplex",)
    sys.stdout.write(
        report(
            a.out,
            cases_root,
            datasets,
            a.submission or "",
            a.csv,
            not a.no_queue,
            (a.runs or "").split() or None,
        )  # fmt: skip
    )
    return 0
