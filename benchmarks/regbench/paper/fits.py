"""Cost against input size (Tables ``cost_points``, ``cost_fits``, ``pipeline_cost_fits``).

``points``  (``paper.costs``, Task 31) one point per PATIENT of every row, whatever
            became of it (``status``; only ``complete`` ones enter a median or a fit),
            for every method: a pair method's patient is the sums over its rounds the
            scorer made; plus fig. 2c's per-round points (``unit = round``), where
            every method is per round: a pair method's ok rounds, and a per-patient
            method's rounds only where its launch timed each round
            (``register_s_per_round_from = info``).
``method_fits``  ``log10(target) ~ log10(predictor)`` per row over its complete patients
            (``patient_set = survivors``), by ordinary least squares on log-log axes,
            with the 95% interval of the slope (Student's t, n - 2 degrees of freedom)
            and the residual s.d. (ddof = 1). Below ``MIN_FIT_POINTS`` (5) points there
            is no fit (the figure shows the count, no line).
``pipeline_fits``  the same per (backend, process) over the tasks of the compute-all runs,
            ``<out>/tables/pipeline_cost.csv`` joined to each run's
            ``compute/<backend>/<patient>/results/size_logs/input_sizes.csv`` by
            (process, tag) with ``bin/generate_resource_report.py``'s ``parse_size_log`` and
            ``join_size`` (loaded by path: ``bin/`` is not a package).

A fit is ``fit_ok`` only with ``min_n`` points (5 for a method fit, 3 for a pipeline fit)
and two distinct sizes (rounded to 1e-9): tasks of one size are a degenerate design that
least squares still solves. Otherwise its slope, intercept, r2 and sigma are blank. Every
fit row counts the points it left out (``n_dropped``: no finite target or predictor, a
value <= 0 on a log axis, or for a pipeline task no input size), and the figures say "k
dropped" (Task 28, S12: a point is never lost silently).

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
METHOD_TARGETS = ("wall_s", "peak_rss_gb", "cpu_h_used")
MIN_FIT_POINTS = 5  # P10: no method fit line below five patients
# two-sided 97.5% quantiles of Student's t, by degrees of freedom (1..30); above 30 the
# normal's 1.96 is within 2% of it
T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201,
        2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074,
        2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)
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


def _t975(df):
    return T975[df - 1] if 1 <= df <= len(T975) else 1.96


def fit(x, y, log=False, min_n=3):
    """Ordinary least squares ``y = slope * x + intercept`` (on ``log10`` of both with
    ``log``, where a value <= 0 is left out); see the module docstring."""
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(_num(a)) and math.isfinite(_num(b))]
    if log:
        pairs = [(math.log10(a), math.log10(b)) for a, b in pairs if a > 0 and b > 0]
    n = len(pairs)
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    distinct = len({round(v, 9) for v in xs.tolist()})
    out = {"slope": None, "intercept": None, "r2": None, "sigma": None, "n": n,
           "n_distinct_x": distinct, "fit_ok": False, "slope_ci_low": None,
           "slope_ci_high": None, "scale": "log-log" if log else "linear",
           "min_n": min_n}
    if n < max(3, min_n) or distinct < 2:
        return out
    if float(np.ptp(ys)) == 0.0:
        # a flat target: polyfit's round-off would give a slope of 1e-16, not zero
        slope, intercept = 0.0, float(ys[0])
    else:
        slope, intercept = (float(v) for v in np.polyfit(xs, ys, 1))
    resid = ys - (slope * xs + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    sxx = float(np.sum((xs - xs.mean()) ** 2))
    half = _t975(n - 2) * math.sqrt(ss_res / (n - 2) / sxx) if sxx > 0 else math.nan
    out.update(
        slope=slope,
        intercept=intercept,
        r2=None if ss_tot == 0 else 1.0 - ss_res / ss_tot,
        sigma=float(np.std(resid, ddof=1)),
        fit_ok=True,
        slope_ci_low=slope - half if half == half else None,
        slope_ci_high=slope + half if half == half else None,
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


def gb_of(cases):
    """``gb(rounds)``: the bytes of the slide files a patient's rounds read, in GB
    (``case.json: ref_image, mov_image``), or None without the cases."""
    if not cases:
        return None

    def gb(rounds):
        files = [
            f
            for x in rounds
            if x.get("case_id") in cases
            for f in (cases[x["case_id"]].ref_image, cases[x["case_id"]].mov_image)
        ]
        return _gb(files) if files else None

    return gb


def points(out, dataset, cases_root=None):
    """The ``cost_points`` rows: ``tables.build``'s, by ``paper.costs``'s rules (the
    rounds excluded from summaries are read from the scored cells there, so there is no
    ``excluded`` argument any more)."""
    from . import tables as t

    return t.build(out, dataset, cases_root)["cost_points"]


def _finite_or_none(v):
    f = _num(v)
    return f if math.isfinite(f) else None


# ── cost_fits ────────────────────────────────────────────────────────────────
def _fit_row(name, xs, ys, n_dropped=0, log=False, min_n=3, **ident):
    """One fit row; ``n_dropped`` counts the points left out of it (S12)."""
    got = fit(xs, ys, log=log, min_n=min_n)
    cols = set(schema.columns(name))
    got = {k: v for k, v in got.items() if k in cols}
    return schema.empty_row(name, source=SOURCE, **ident, **got, n_dropped=n_dropped)


def method_fits(points_rows):
    """The ``cost_fits`` rows: per row, target and predictor, log-log, over the row's
    complete patients (``unit = patient``, ``status = complete``; P10, P13)."""
    out = []
    pts = [
        p
        for p in points_rows
        if p.get("unit_kind", "patient") == "patient" and p.get("status", "complete") == "complete"
    ]
    keys = dict.fromkeys((p["dataset"], p["family"], p["row"]) for p in pts)
    for ds, fam, row in keys:
        mine = [p for p in pts if (p["dataset"], p["family"], p["row"]) == (ds, fam, row)]
        for target in METHOD_TARGETS:
            for pred in METHOD_PREDICTORS:
                both = [(_num(p.get(pred)), _num(p.get(target))) for p in mine]
                both = [b for b in both if b[0] > 0 and b[1] > 0]  # log axes: > 0 only
                out.append(
                    _fit_row(
                        "cost_fits",
                        [b[0] for b in both],
                        [b[1] for b in both],
                        n_dropped=len(mine) - len(both),
                        log=True,
                        min_n=MIN_FIT_POINTS,
                        dataset=ds, family=fam, row=row, target=target, predictor=pred,
                        unit="patient", patient_set="survivors",
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
    tasks that ran (COMPLETED, or CACHED: a task of an earlier session whose figures the
    trace keeps; any other ``status``, when the table has one, is not a cost) and whose
    input size the size log gives."""
    from . import tables as t

    tasks = t._csv(Path(out) / "tables" / "pipeline_cost.csv")
    if not tasks:
        return []
    report = _report_module()
    runs = {}
    for r in tasks:
        runs.setdefault((r["backend"], r["patient"]), []).append(r)
    points_by, no_size = {}, {}
    for (backend, patient), rows in runs.items():
        log = Path(out) / "compute" / backend / patient / "results" / "size_logs" / "input_sizes.csv"
        joined = report.join_size(rows, report.parse_size_log(log))
        for r in joined:
            # P15: a CACHED task ran (in an earlier session) and its trace has the
            # figures of that run: a point like a COMPLETED one
            if r.get("status") not in (None, "", "COMPLETED", "CACHED"):
                continue
            if r["input_bytes"] is None:
                key = (backend, r["process"])
                no_size[key] = no_size.get(key, 0) + 1
                points_by.setdefault(key, [])
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
                    n_dropped=no_size.get((backend, process), 0) + len(rows) - len(both),
                    backend=backend, process=process, target=target, predictor="input_gb",
                )
            )
    return result
