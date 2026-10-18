"""Cost against input size (Tables ``cost_points``, ``cost_fits``, ``pipeline_cost_fits``).

``points``  one point per ok unit of each row: a round (``resources.csv``) for a pair
            method, a patient (``summary_resources.csv``) for a per-patient one. Only the
            COMPLETE patients of the latter (every round registered: the cost of some of a
            patient's rounds set against all of its input would bias the slope down), and
            never an excluded round.
``method_fits``  ``target ~ predictor`` per row, by ordinary least squares.
``pipeline_fits``  the same per (backend, process) over the tasks of the compute-all runs,
            ``<out>/tables/pipeline_cost.csv`` joined to each run's
            ``compute/<backend>/<patient>/results/size_logs/input_sizes.csv`` by
            (process, tag) with ``bin/generate_resource_report.py``'s ``parse_size_log`` and
            ``join_size`` (loaded by path: ``bin/`` is not a package).

A fit is ``fit_ok`` only with three points and two distinct sizes (rounded to 1e-9): three
tasks of one size are a degenerate design that least squares still solves. Otherwise its
slope, intercept, r2 and sigma are blank.

The size of a unit's input is read from the slide files named by the prepared cases
(``case.json: ref_image, mov_image``), so it needs ``cases_root``; without it ``input_gb``
is blank and only the ``input_gpx`` fits can be ok.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import numpy as np

from . import schema

SOURCE = "regbench"
GIB = 2**30
METHOD_TARGETS = ("wall_s", "peak_rss_gb")
METHOD_PREDICTORS = ("input_gb", "input_gpx")
PIPELINE_TARGETS = ("realtime_s", "peak_rss_gb")
REPO = Path(__file__).resolve().parents[3]


def _num(v):
    if v is None or isinstance(v, bool):
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def fit(x, y):
    """Ordinary least squares ``y = slope * x + intercept``; see the module docstring."""
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(_num(a)) and math.isfinite(_num(b))]
    n = len(pairs)
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    distinct = len({round(v, 9) for v in xs.tolist()})
    out = {"slope": None, "intercept": None, "r2": None, "sigma": None, "n": n,
           "n_distinct_x": distinct, "fit_ok": False}
    if n < 3 or distinct < 2:
        return out
    if float(np.ptp(ys)) == 0.0:
        # a flat target: polyfit's round-off would give a slope of 1e-16, not zero
        slope, intercept = 0.0, float(ys[0])
    else:
        slope, intercept = (float(v) for v in np.polyfit(xs, ys, 1))
    resid = ys - (slope * xs + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    out.update(
        slope=slope,
        intercept=intercept,
        r2=None if ss_tot == 0 else 1.0 - ss_res / ss_tot,
        sigma=float(np.std(resid)),
        fit_ok=True,
    )
    return out


# ── cost_points ──────────────────────────────────────────────────────────────
def _file_bytes(paths):
    total = 0
    for p in dict.fromkeys(paths):
        try:
            total += os.path.getsize(p)
        except (OSError, TypeError):
            return None
    return total


def _cases_by_id(cases_root, dataset):
    if not cases_root:
        return {}
    from ..cases import list_cases

    return {str(c.case_id): c for c in list_cases(cases_root, dataset)}


def _gb(paths):
    got = _file_bytes(paths)
    return None if got is None else got / GIB


def points(out, dataset, cases_root=None, excluded=None):
    """The ``cost_points`` rows (see the module docstring). ``excluded`` is
    ``{(case_id, row)}`` of the rounds excluded from the summaries (``tables._excluded_rounds``);
    by default it is read from the cell tables."""
    from . import tables as t

    if excluded is None:
        excluded = t._excluded_rounds(t.per_pair(out, dataset))
    cases = _cases_by_id(cases_root, dataset)
    res = t._res_rows(out, dataset)
    summary = t._cost_rows(out, dataset)
    rows = []
    for row in dict.fromkeys(t.terminal_rows(dataset).values()):
        fam = t.family_of(row)
        if fam is None:
            continue
        mine = [r for r in summary if r["method"] == row]
        if any(r.get("unit") == "patient" for r in mine):
            for r in t._complete(mine):
                patient = r["group"]
                rounds = [x for x in res if x["method"] == row and x["group"] == patient]
                if any((x["case_id"], row) in excluded for x in rounds):
                    continue
                files = [
                    f
                    for x in rounds
                    if x["case_id"] in cases
                    for f in (cases[x["case_id"]].ref_image, cases[x["case_id"]].mov_image)
                ]
                rows.append(_point(dataset, fam, row, "patient", patient, r, files if cases else None))
        else:
            for r in res:
                if r["method"] != row or not t._flag(r.get("ok")) or (r["case_id"], row) in excluded:
                    continue
                c = cases.get(r["case_id"])
                files = [c.ref_image, c.mov_image] if c is not None else None
                rows.append(_point(dataset, fam, row, "round", r["case_id"], r, files))
    return rows


def _point(dataset, fam, row, kind, unit_id, r, files):
    staged = r.get("n_jobs") not in (None, "", "nan", schema.NA)
    wall = r.get("wall_unlimited_parallel_s") if staged else r.get("register_s")
    return schema.empty_row(
        "cost_points",
        source=SOURCE,
        dataset=dataset,
        family=fam,
        row=row,
        unit_kind=kind,
        unit_id=unit_id,
        input_gb=_gb(files) if files else None,
        input_gpx=_finite_or_none(r.get("moving_gpx")),
        wall_s=_finite_or_none(wall),
        peak_rss_gb=_finite_or_none(r.get("peak_rss_gb")),
        cpu_h_used=_finite_or_none(r.get("cpu_h_used")),
        device=r.get("device") or None,
    )


def _finite_or_none(v):
    f = _num(v)
    return f if math.isfinite(f) else None


# ── cost_fits ────────────────────────────────────────────────────────────────
def _fit_row(name, xs, ys, **ident):
    return schema.empty_row(name, source=SOURCE, **ident, **fit(xs, ys))


def method_fits(points_rows):
    """The ``cost_fits`` rows: per row, target and predictor."""
    out = []
    keys = dict.fromkeys((p["dataset"], p["family"], p["row"]) for p in points_rows)
    for ds, fam, row in keys:
        mine = [p for p in points_rows if (p["dataset"], p["family"], p["row"]) == (ds, fam, row)]
        for target in METHOD_TARGETS:
            for pred in METHOD_PREDICTORS:
                both = [(_num(p[pred]), _num(p[target])) for p in mine]
                both = [b for b in both if math.isfinite(b[0]) and math.isfinite(b[1])]
                out.append(
                    _fit_row(
                        "cost_fits",
                        [b[0] for b in both],
                        [b[1] for b in both],
                        dataset=ds, family=fam, row=row, target=target, predictor=pred,
                    )
                )
    return out


# ── pipeline_cost_fits ───────────────────────────────────────────────────────
def _report_module():
    """``bin/generate_resource_report.py`` by path (``bin/`` is not a package)."""
    path = REPO / "bin" / "generate_resource_report.py"
    spec = importlib.util.spec_from_file_location("mirage_resource_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pipeline_fits(out):
    """The ``pipeline_cost_fits`` rows: per (backend, process, target), over the traced
    tasks that ran (a ``status`` other than COMPLETED, when the table has one, is not a
    cost) and whose input size the size log gives."""
    from . import tables as t

    tasks = t._csv(Path(out) / "tables" / "pipeline_cost.csv")
    if not tasks:
        return []
    report = _report_module()
    runs = {}
    for r in tasks:
        runs.setdefault((r["backend"], r["patient"]), []).append(r)
    points_by = {}
    for (backend, patient), rows in runs.items():
        log = Path(out) / "compute" / backend / patient / "results" / "size_logs" / "input_sizes.csv"
        joined = report.join_size(rows, report.parse_size_log(log))
        for r in joined:
            if r.get("status") not in (None, "", "COMPLETED"):
                continue
            if r["input_bytes"] is None:
                continue
            points_by.setdefault((backend, r["process"]), []).append(r)
    result = []
    for (backend, process), rows in sorted(points_by.items()):
        for target in PIPELINE_TARGETS:
            both = [(r["input_bytes"] / GIB, _num(r.get(target))) for r in rows]
            both = [b for b in both if math.isfinite(b[1])]
            result.append(
                _fit_row(
                    "pipeline_cost_fits",
                    [b[0] for b in both],
                    [b[1] for b in both],
                    backend=backend, process=process, target=target, predictor="input_gb",
                )
            )
    return result
