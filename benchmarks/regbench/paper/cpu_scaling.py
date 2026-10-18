"""The CPU cross (``cpu_scaling``): wall time, CPU-hours and peak memory of each family's
best-Dice row at 4 and 8 CPUs, per patient and repeat (the user's ruling of 2026-10-18).

**The rows** (``cross_rows``, ``regbench cross-runs --out MAIN``): each family's
best-Dice row of the MAIN run's tables, by the paper's own rule (``tables.best_of``),
DeeperHistReg's among its CPU rows only; ``stare_local4`` is never one (it is a cost row
of ``stare``). A GPU row is refused (``build``): the cross measures what more CPUs buy,
and a GPU row's host side is not the method.

**The points.** ``points`` maps a CPU count to the results folders that ran at it: every
cross row runs at BOTH counts, in the same submission style, each count in its own folder
(``out_c4``, ``out_c8``), ``REGBENCH_REPEAT`` times (the launcher's ``rep<k>``
subfolders; a folder with its own ``tables/`` is one repeat). Each folder's
``tables/summary_resources.csv`` gives one line per (patient, result name); only
COMPLETE patients count (``tables._complete``).

**What is checked, per point** (never hidden):

* ``cpus_checked``: the CPUs every record of the point asked for (``resources.csv:
  cpus_requested``) against the point's count: ``yes``, or ``no: <seen>``;
* ``cpu_model`` of the point's records, and ``cpu_model_matches``: ``yes`` when every
  point of the patient (every count and repeat) ran on one CPU model, else ``no``;
* ``n_patients_incomplete``: for the (row, execution), the patients that are not
  complete at every count, left out of it and counted.

**Execution.** ``jobs``, ``local`` (STARE's stages in one job: the cross) or ``one
process``, read from the records (``reuse_key.execution``, else ``info.jobs.execution``).
Speed-up and efficiency are only taken inside one (row, execution, repeat): a ``local``
point is never compared with a ``jobs`` one.

**Wall time.** ``wall_unlimited_parallel_s`` for a row run as several scheduler jobs
(``n_jobs`` present), ``register_s`` otherwise (``costs.wall_of``).
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from . import costs, schema, tables

ONE_PROCESS = "one process"
BASE_CPUS = 4
DEEPERHISTREG = "deeperhistreg"


# ── the rows ─────────────────────────────────────────────────────────────────
def _is_gpu_row(name):
    run = tables._run(name)
    return bool(run is not None and run.gpu)


def cross_rows(out, dataset=tables.COHORT):
    """``[(family, row)]``: each family's best-Dice row in MAIN's tables
    (``tables.best_of``), DeeperHistReg's among its CPU rows. Refuses a folder with no
    scored tables."""
    pp = tables.per_pair(out, dataset)
    if not pp:
        raise SystemExit(
            f"cross-runs: {tables._tables(out, dataset)} holds no scored cells table "
            "(cells.csv): score the main run first (`regbench score`)"
        )
    cost = tables._cpu_by_row(tables._cost_rows(out, dataset))
    got = []
    for fam in tables._families(dataset):
        only = (lambda n: not _is_gpu_row(n)) if fam == DEEPERHISTREG else None
        best = tables.best_of(pp, fam, dataset, cost, candidates=only)
        if not tables._missing(best["best_row"]):
            got.append((fam, best["best_row"]))
    return got


def main_cross_runs(a):
    rows = cross_rows(a.out, a.dataset)
    for fam, row in rows:
        print(f"{fam}\t{row}")
    print(f"# --rows {','.join(r for _, r in rows)}")
    return 0


# ── the points ───────────────────────────────────────────────────────────────
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


def repeats(folder):
    """``[(repeat, folder)]``: the launcher's ``rep<k>`` subfolders (``REGBENCH_REPEAT``),
    else the folder itself as repeat 1."""
    folder = Path(folder)
    subs = sorted(
        (int(p.name[3:]), p)
        for p in folder.glob("rep*")
        if p.is_dir() and p.name[3:].isdigit() and (p / "tables").is_dir()
    )
    return subs or [(1, folder)]


def _hosts(out, dataset, name, group):
    rows = [r for r in tables._res_rows(out, dataset) if r.get("method") == name and r.get("group") == group]
    asked = sorted({str(r.get("cpus_requested")) for r in rows if not tables._missing(r.get("cpus_requested"))})
    models = sorted({str(r.get("cpu_model")) for r in rows if not tables._missing(r.get("cpu_model"))})
    return asked, models


def build(points, rows, dataset=tables.COHORT):
    """The ``cpu_scaling`` rows (module docstring)."""
    wanted = list(dict.fromkeys(rows))
    gpu = [n for n in wanted if _is_gpu_row(n)]
    if gpu:
        raise SystemExit(
            f"cpu-scaling: {', '.join(gpu)} run(s) on a GPU: the CPU cross takes CPU rows "
            "only (`regbench cross-runs` gives DeeperHistReg's best CPU row)"
        )
    cells = {}  # (row, execution, cpus, patient, repeat) -> measured values
    for cpus, folders in sorted(points.items()):
        for given in folders:
            for rep, out in repeats(given):
                for r in tables._complete(tables._cost_rows(out, dataset)):
                    name = r["method"]
                    if name not in wanted:
                        continue
                    if r.get("device") == "gpu":
                        raise SystemExit(f"cpu-scaling: {name} ran on a GPU in {out}: refused")
                    execution = _execution_label(out, name, dataset)
                    key = (name, execution, int(cpus), r["group"], rep)
                    if key in cells:
                        raise ValueError(
                            f"cpu_scaling: {name} ({execution}) of {r['group']} at {cpus} CPUs, "
                            f"repeat {rep}, is in two folders ({cells[key]['out_root']} and {out})"
                        )
                    asked, models = _hosts(out, dataset, name, r["group"])
                    cells[key] = {
                        "out_root": str(out),
                        "wall_s": costs.wall_of(r),
                        "cpu_h_used": tables._num(r.get("cpu_h_used")),
                        "peak_rss_gb": tables._num(r.get("peak_rss_gb")),
                        "cpus_checked": "yes" if asked and all(
                            math.isclose(tables._num(a), cpus) for a in asked
                        ) else f"no: {'|'.join(asked) or 'none recorded'}",
                        "cpu_model": "|".join(models),
                    }
    counts, patients = {}, {}
    for name, ex, cpus, pid, rep in cells:
        counts.setdefault((name, ex), set()).add(cpus)
        patients.setdefault((name, ex, pid), set()).add(cpus)
    out_rows = []
    for name in wanted:
        for ex in sorted(e for n, e in counts if n == name):
            every = counts[(name, ex)]
            mine = {p: c for (n, e, p), c in patients.items() if n == name and e == ex}
            keep = sorted(p for p, c in mine.items() if c == every)
            n_incomplete = len(mine) - len(keep)
            for pid in keep:
                mine_cells = {k: v for k, v in cells.items() if k[:2] == (name, ex) and k[3] == pid}
                models = {v["cpu_model"] for v in mine_cells.values() if v["cpu_model"]}
                same = "yes" if len(models) <= 1 else "no"
                for (_, _, cpus, _, rep), cell in sorted(mine_cells.items()):
                    base = cells.get((name, ex, BASE_CPUS, pid, rep))
                    speedup = math.nan
                    if (
                        base
                        and math.isfinite(base["wall_s"])
                        and math.isfinite(cell["wall_s"])
                        and cell["wall_s"] > 0
                    ):
                        speedup = base["wall_s"] / cell["wall_s"]
                    clock = (costs.LOCAL4 if ex == "local" else costs.WHOLE
                             if ex == ONE_PROCESS else costs.CRITICAL)
                    out_rows.append(
                        schema.empty_row(
                            "cpu_scaling",
                            family=tables.family_of(name) or schema.NA,
                            row=name,
                            cpus=cpus,
                            patient_id=pid,
                            execution=ex,
                            repeat=rep,
                            wall_s=cell["wall_s"],
                            wall_clock=clock,
                            clock=clock,  # of the CPU-hours and the peak too (T31 fix I5)
                            unit=costs.PATIENT,
                            cpu_h_used=cell["cpu_h_used"],
                            peak_rss_gb=cell["peak_rss_gb"],
                            speedup_vs_4=speedup,
                            efficiency=speedup / (cpus / BASE_CPUS),
                            cpus_checked=cell["cpus_checked"],
                            cpu_model=cell["cpu_model"],
                            cpu_model_matches=same,
                            n_patients_incomplete=n_incomplete,
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
    if a.rows:
        rows = [x for x in a.rows.split(",") if x]
    elif getattr(a, "main", None):
        rows = [r for _, r in cross_rows(a.main, a.dataset)]
    else:
        raise SystemExit(
            "cpu-scaling: name the rows (--rows) or the main run they are chosen from "
            "(--main MAIN: each family's best-Dice row, `regbench cross-runs`)"
        )
    result = build(points, rows, a.dataset)
    Path(a.to).parent.mkdir(parents=True, exist_ok=True)
    schema.write(a.to, "cpu_scaling", result)
    bad = sorted({(r["row"], r["cpus"], r["repeat"]) for r in result if r["cpus_checked"] != "yes"})
    mixed = sorted({(r["row"], r["patient_id"]) for r in result if r["cpu_model_matches"] != "yes"})
    gone = sorted({(r["row"], r["n_patients_incomplete"]) for r in result if r["n_patients_incomplete"]})
    print(f"cpu-scaling: {len(result)} row(s) of {len(rows)} named, to {a.to}")
    for row, cpus, rep in bad:
        print(f"cpu-scaling: {row} at {cpus} CPUs (repeat {rep}): the records asked for other CPUs")
    for row, pid in mixed:
        print(f"cpu-scaling: {row} {pid}: its points ran on different CPU models (cpu_model)")
    for row, k in gone:
        print(f"cpu-scaling: {row}: {k} patient(s) not complete at every count, left out")
    return 0
