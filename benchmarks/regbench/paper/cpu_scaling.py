"""The CPU-scaling table (``cpu_scaling``): wall time, CPU-hours and peak memory of the
primary rows at 4, 8 and 16 CPUs, per patient.

``points`` maps a CPU count to the results folders that ran at it. Each folder's
``tables/summary_resources.csv`` gives one line per (patient, result name); only COMPLETE
patients count (``tables._complete``: a patient with a failed round has the cost of some
of its rounds, not of itself).

**Execution.** A row is one of ``jobs`` (STARE's stage jobs, one scheduler job each),
``local`` (STARE's stages in one job: the CPU cross) or ``one process`` (a method with no
stage chain). It is read from the folder's run records as ``subset.staged_stare`` reads
it: ``reuse_key.execution``, else ``info.jobs.execution``. Speedup and efficiency are
only ever taken inside one (row, execution): a ``local`` point is never compared with a
``jobs`` one.

**Patients.** For one (row, execution), a patient is kept only if it is complete at EVERY
count that (row, execution) has.

**Wall time.** ``wall_unlimited_parallel_s`` for a row run as several scheduler jobs
(``n_jobs`` present: the queue-free wall time with every tile at once), ``register_s``
otherwise: the same rule as ``tables._cost``.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from . import schema, tables

ONE_PROCESS = "one process"
BASE_CPUS = 4


def _execution_of(rec):
    """``reuse_key.execution``, else ``info.jobs.execution`` (``subset.staged_stare``),
    else None."""
    key = rec.get("reuse_key")
    execution = key.get("execution") if isinstance(key, dict) else None
    if execution is None:
        jobs = (rec.get("info") or {}).get("jobs")
        execution = jobs.get("execution") if isinstance(jobs, dict) else None
    return execution


def _execution_label(out, row, dataset):
    """The execution the folder's records of ``row`` say; ``one process`` for a result
    with none. A folder whose records disagree is labelled by the most common one."""
    seen = Counter()
    for rec in tables._records(out, row, dataset):
        if isinstance(rec, dict):
            ex = _execution_of(rec)
            if ex:
                seen[str(ex)] += 1
    return seen.most_common(1)[0][0] if seen else ONE_PROCESS


def build(points, rows, dataset=tables.COHORT):
    wanted = list(dict.fromkeys(rows))
    cells = {}  # (row, execution, cpus, patient) -> measured values
    for cpus, folders in sorted(points.items()):
        for out in folders:
            for r in tables._complete(tables._cost_rows(out, dataset)):
                name = r["method"]
                if name not in wanted:
                    continue
                execution = _execution_label(out, name, dataset)
                jobs = not tables._missing(r.get("n_jobs"))
                wall = tables._num(r.get("wall_unlimited_parallel_s" if jobs else "register_s"))
                key = (name, execution, int(cpus), r["group"])
                if key in cells:
                    raise ValueError(
                        f"cpu_scaling: {name} ({execution}) of {r['group']} at {cpus} CPUs is "
                        f"in two folders ({cells[key]['out_root']} and {out})"
                    )
                cells[key] = {
                    "out_root": str(out),
                    "wall_s": wall,
                    "cpu_h_used": tables._num(r.get("cpu_h_used")),
                    "peak_rss_gb": tables._num(r.get("peak_rss_gb")),
                }
    counts, patients = {}, {}
    for name, ex, cpus, pid in cells:
        counts.setdefault((name, ex), set()).add(cpus)
        patients.setdefault((name, ex, pid), set()).add(cpus)
    out_rows = []
    for name in wanted:
        for ex in sorted(e for n, e in counts if n == name):
            every = counts[(name, ex)]
            keep = sorted(p for (n, e, p), c in patients.items() if n == name and e == ex and c == every)
            for pid in keep:
                base = cells.get((name, ex, BASE_CPUS, pid))
                for cpus in sorted(every):
                    cell = cells[(name, ex, cpus, pid)]
                    speedup = math.nan
                    if (
                        base
                        and math.isfinite(base["wall_s"])
                        and math.isfinite(cell["wall_s"])
                        and cell["wall_s"] > 0
                    ):
                        speedup = base["wall_s"] / cell["wall_s"]
                    out_rows.append(
                        schema.empty_row(
                            "cpu_scaling",
                            family=tables.family_of(name) or schema.NA,
                            row=name,
                            cpus=cpus,
                            patient_id=pid,
                            execution=ex,
                            wall_s=cell["wall_s"],
                            cpu_h_used=cell["cpu_h_used"],
                            peak_rss_gb=cell["peak_rss_gb"],
                            speedup_vs_4=speedup,
                            efficiency=speedup / (cpus / BASE_CPUS),
                            out_root=cell["out_root"],
                        )
                    )
    return out_rows


def main(a):
    points = {}
    for spec in a.point:
        cpus, _, out = spec.partition("=")
        if not out or not cpus.isdigit():
            raise SystemExit(f"cpu-scaling: --point wants CPUS=OUT, got {spec!r}")
        points.setdefault(int(cpus), []).append(Path(out))
    rows = [x for x in a.rows.split(",") if x]
    result = build(points, rows, a.dataset)
    Path(a.to).parent.mkdir(parents=True, exist_ok=True)
    schema.write(a.to, "cpu_scaling", result)
    print(f"cpu-scaling: {len(result)} row(s) of {len(rows)} named, to {a.to}")
    return 0
