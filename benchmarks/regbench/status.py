"""``regbench status``: where a results folder stands, for the operator. READ-ONLY.

Per run and result name (variant) and dataset, against what the cohort must have (every
unit of the case list, ``variants.runs_for``): how many results are ``ok``, how many of
those this submission REUSED from an earlier one, how many failed (by cause), and how
many are MISSING (no ``run.json``). ``ok`` counts only what the scorer would load for
the submission (``score._load_run``: the record is this submission's, ``ladder.mine``,
and was made for the case's points and landmarks as they are now,
``methods.record_gaps``); an ok record of ANOTHER submission is ``other_sub``, and one
made for points or landmarks the case no longer has is ``stale``. Both are reported
apart, so the counts agree with the tables. The jobs of the results folder still in the queue
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
    "other_sub", "stale", "causes",
)  # fmt: skip
COUNTS = ("expected", "ok", "reused", "failed", "missing", "other_sub", "stale")


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
    from . import ladder, landmark_io, variants
    from .cases import stale_points
    from .chains import _units

    def fresh(rec, case):
        made_for = rec.get("points_sha256")
        return not stale_points(made_for, case) and (
            rec.get("truth_sha256") == landmark_io.truth_sha256(case)
        )

    rows, todo = [], {}
    for run in variants.runs_for(None):
        name = variants.run_name(run)
        if runs and name not in runs:
            continue
        units = _units(cases_root, dataset, run.method)

        def zero():
            return {**dict.fromkeys(COUNTS, 0), "causes": {}}

        by_variant = {v: zero() for v in variants.variants_of(run)}
        for index, cases in enumerate(units):
            unit_ok = True
            heads = ladder.unit_heads(out, run.method, run.label, cases)
            by_id = {c.case_id: c for c in cases}
            for path, head in heads.items():
                variant = head["variant"]
                c = by_variant.setdefault(variant, zero())
                c["expected"] += 1
                rec = _read(path)
                if not isinstance(rec, dict):
                    c["missing"] += 1
                    unit_ok = False
                    continue
                if rec.get("ok"):
                    if not ladder.mine(rec, submission):
                        c["other_sub"] += 1
                        unit_ok = False
                    elif not fresh(rec, by_id[head["case_id"]]):
                        c["stale"] += 1
                        unit_ok = False
                    else:
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
                    **{k: c[k] for k in COUNTS},
                    "causes": ";".join(
                        f"{k}:{v}" for k, v in sorted(c["causes"].items())
                    ),
                }
            )
    return rows, todo


def queued(out):
    """``(jobs, note)``: the jobs of this results folder still in the queue
    (``[(job id, name, state)]``), or None with why they were not read. A job is the
    folder's when its id is one ``logs/jobs.tsv`` lists (column 3, an array task's
    ``_<task>`` stripped): squeue's ``%F`` (the array job id; the job id of a job
    that is not an array) or ``%A`` (the job's own id). Not by name: another folder's
    submission names its jobs the same."""
    if not shutil.which("squeue"):
        return None, "not read (no squeue on this node)"
    try:
        ids = {
            line.split("\t")[2].strip().split("_")[0]
            for line in Path(out, "logs", "jobs.tsv").read_text().splitlines()
            if line.count("\t") >= 2
        }
    except OSError:
        ids = set()
    ids.discard("")
    user = os.environ.get("USER") or ""
    argv = ["squeue", "-h", "-o", "%i|%j|%T|%F|%A"] + (["-u", user] if user else [])
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
        if len(parts) == 5 and (parts[3] in ids or parts[4] in ids):
            jobs.append(tuple(parts[:3]))
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
    total = {k: sum(r[k] for r in rows) for k in COUNTS}
    say(
        f"# {total['expected']} result(s) expected: {total['ok']} ok "
        f"({total['reused']} reused), {total['failed']} failed, {total['missing']} missing"
        + (
            f"; not scored for this submission: {total['other_sub']} ok of another "
            f"submission, {total['stale']} made for points or landmarks the case no "
            "longer has"
            if total["other_sub"] or total["stale"]
            else ""
        )
        + "\n"
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
    queued_names = {name for _, name, _ in jobs or []}
    for plan in left:
        # a watcher still in the queue is finishing its chain: not a warning (its links
        # are <prefix>-retry and <prefix>-retry-<n>)
        if any(
            n == f"{plan.name}-retry" or n.startswith(f"{plan.name}-retry-")
            for n in queued_names
        ):
            continue
        say(
            f"# WARNING chain not finished by any link of its watcher: {plan.name} ({plan})"
            + (" (the queue was not read)" if jobs is None else "")
            + "\n"
        )
    if jobs:
        say(
            f"# next: wait; {len(jobs)} job(s) are still queued (squeue), then run this again\n"
        )
    elif todo:
        runs = " ".join(sorted(todo))
        n = sum(len(v) for v in todo.values())
        # the FULL run list: the scorer loads only what the LAST submission into the
        # folder ran or reused, so a --runs subset here would leave every other run
        # `not run` in the tables (README, "A subset submitted into a results folder")
        say(
            f"# next: {n} unit(s) of {len(todo)} run(s) ({runs}) are not all ok: submit "
            f"the FULL run list again into the same folder, which reuses every ok unit: "
            f"REGBENCH_OUT={out} submit_cohort.sh\n"
            "#   (a --runs subset needs its own REGBENCH_OUT: submitted here, it leaves "
            "every other run `not run` in the tables)\n"
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
