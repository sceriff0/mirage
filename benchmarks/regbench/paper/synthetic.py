"""The synthetic scaling set's tables (T32): how the cost of STARE and VALIS grows with
the size of a slide, on slides made for it (``regbench synth``). COST only.

Three tables (``table_schema``), written by the scorer into ``tables/synthetic/`` and by
``paper --dataset synthetic`` into ``paper/synthetic/``; never pooled with the cohort:

``synthetic_cost``           one row per (run, size): every cost stem of ``paper.costs``
                             (``STEMS``: CPU-hours used and reserved, GB-hours, wall on its
                             clock, the peaks; STARE's per-task peak is
                             ``peak_largest_job_gb``), by its rules (the patient unit: one
                             synthetic patient per size; clock and unit columns; used vs
                             reserved; wasted; nothing silent: a size a run did not
                             complete is a row with its ``status``), plus the design's
                             ``input_gpx``, ``input_bytes``, ``n_channels``, ``n_panels`` and
                             ``n_repeats`` (1: no interval is ever given).
``synthetic_scaling``        per run and quantity (``QUANTITIES``), ``log10(quantity) ~
                             log10(input_gpx)`` over the run's complete sizes
                             (``fits.fit``, Student's t slope interval); no fit below
                             ``MIN_SIZES`` sizes, and the row says why and how many sizes
                             it had.
``synthetic_extrapolation``  each fit used at the cohort's median and largest common
                             patient (``input_gpx``), beside the cohort's MEASURED value of
                             the same result name on its common patients; the gap, and
                             ``extrapolated = 1`` outside the sizes the fit used.

A run left out of the set (``variants.SYNTHETIC_RUNS``) is a row of NA with its reason in
each table (T29 M8, ``tables.add_excluded``), and nothing else is invented. Every row
carries ``synth.CAVEAT``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from .. import synth
from . import costs, fits, schema
from . import tables as T

DATASET = synth.DATASET
SOURCE = T.SOURCE
QUANTITIES = (
    ("cpu_h_used", "CPU-h"),
    ("wall_s", "s"),
    ("peak_rss_gb", "GB"),
    ("gb_h_used", "GB-h"),
    ("peak_largest_job_gb", "GB"),
)
MIN_SIZES = 3
AT = ("cohort_median", "cohort_largest")
TABLES = ("synthetic_cost", "synthetic_scaling", "synthetic_extrapolation")
COHORT_POINTS = "paper/cost_points.csv"
COHORT_SUMMARY = "tables/summary_resources.csv"
ABSENT = "absent"


def size_of(patient):
    """The size of a synthetic patient ``s<size>`` (``synth.patient_of``), or None."""
    m = re.fullmatch(r"s(\d+)", str(patient or ""))
    return int(m.group(1)) if m else None


def input_gpx(size):
    """Moving gigapixels of a synthetic patient by design: 10 panels of size^2."""
    return synth.N_MOVING * size * size / 1e9


def input_bytes(size):
    return synth.N_PANELS * len(synth.CHANNELS) * 2 * size * size


def _finite_or_none(v):
    f = costs.num(v)
    return f if math.isfinite(f) else None


def _common_extra():
    return {
        "n_repeats": 1,
        "caveat": synth.CAVEAT,
    }


def _foreground():
    """The set's tissue fraction by design (synth.FOREGROUND_FRACTION, the same at every
    size; each case's measured value is in its synth_transform.json)."""
    return {"foreground_fraction": synth.FOREGROUND_FRACTION}


# ── synthetic_cost ───────────────────────────────────────────────────────────
def cost_table(cost_rows, res_rows, dataset=DATASET):
    """``synthetic_cost`` rows of the runs of the set (excluded runs: ``add``)."""
    out = []
    cols = set(schema.columns("synthetic_cost"))
    for run, final in T.terminal_rows(dataset).items():
        fam = T.family_of(final)
        mine = [r for r in cost_rows if r.get("method") == final]
        kinds = costs.row_kinds(final, (), ())
        pts = costs.patient_points(final, fam, mine, res_rows, set(), kinds, {})
        for p in pts:
            size = size_of(p["unit_id"])
            rounds = [
                x
                for x in res_rows
                if x.get("method") == final and x.get("group") == p["unit_id"]
            ]
            vals = {k: v for k, v in p.items() if k in cols}
            vals.update(
                run=run,
                patient_id=p["unit_id"],
                size_px=size,
                input_gpx=input_gpx(size) if size else None,
                input_bytes=input_bytes(size) if size else None,
                n_channels=len(synth.CHANNELS),
                n_panels=synth.N_PANELS,
                n_rounds_ok=sum(costs.flag(x.get("ok")) for x in rounds),
                cpus=T._most_common(x.get("cpus_requested") for x in rounds),
                lower_bound=p.get("lower_bound") or None,
                peak_instrument=p.get("peak_instrument") or None,
                **_foreground(),
                **_common_extra(),
            )
            vals.pop("unit_id", None)
            for k, v in list(vals.items()):
                if isinstance(v, float) and not math.isfinite(v):
                    vals[k] = None
            out.append(
                schema.empty_row(
                    "synthetic_cost", source=SOURCE, dataset=dataset, **vals
                )
            )
    out.sort(key=lambda r: (str(r["row"]), size_of(r["patient_id"]) or 0))
    return out


# ── synthetic_scaling ────────────────────────────────────────────────────────
def _fit_of(points, quantity):
    """``(fit, sizes used)`` of one quantity over complete points."""
    done = [p for p in points if p["status"] == costs.COMPLETE]
    pairs = []
    for p in done:
        x, y = costs.num(p["input_gpx"]), costs.num(p.get(quantity))
        if math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0:
            pairs.append((x, y, size_of(p["patient_id"])))
    got = fits.fit([a for a, _, _ in pairs], [b for _, b, _ in pairs], log=True,
                   min_n=MIN_SIZES)  # fmt: skip
    return got, pairs


def _no_fit_reason(got, pairs, n_complete, quantity, n_grid):
    if got["fit_ok"]:
        return None
    if n_complete == 0:
        return "no size complete: nothing to fit"
    if len(pairs) < MIN_SIZES:
        return (
            f"{len(pairs)} size(s) with a finite, positive {quantity} (of "
            f"{n_complete} complete, {n_grid} measured); a fit needs {MIN_SIZES}"
        )
    return f"fewer than 2 distinct sizes ({got['n_distinct_x']})"


def scaling_table(cost, dataset=DATASET):
    """``synthetic_scaling`` rows: one per (run, quantity), a fit or the reason none."""
    out = []
    by_row = {}
    for r in cost:
        if r.get("excluded_reason") in (None, schema.NA):
            by_row.setdefault(r["row"], []).append(r)
    for run, final in T.terminal_rows(dataset).items():
        mine = by_row.get(final, [])
        n_complete = sum(p["status"] == costs.COMPLETE for p in mine)
        clock = T._most_common(p.get("clock") for p in mine)
        for quantity, qunit in QUANTITIES:
            got, pairs = _fit_of(mine, quantity)
            xs = [a for a, _, _ in pairs]
            out.append(
                schema.empty_row(
                    "synthetic_scaling",
                    source=SOURCE,
                    dataset=dataset,
                    family=T.family_of(final),
                    row=final,
                    quantity=quantity,
                    quantity_unit=qunit,
                    predictor="input_gpx",
                    scale="log-log",
                    clock=costs.STEM_CLOCK.get(quantity, clock),
                    unit=costs.PATIENT,
                    min_n=MIN_SIZES,
                    n_sizes_used=len(pairs),
                    sizes_used="|".join(str(s) for _, _, s in sorted(pairs)) or None,
                    slope=got["slope"],
                    slope_ci_low=got["slope_ci_low"],
                    slope_ci_high=got["slope_ci_high"],
                    intercept=got["intercept"],
                    r2=got["r2"],
                    sigma=got["sigma"],
                    n=got["n"],
                    n_distinct_x=got["n_distinct_x"],
                    fit_ok=got["fit_ok"],
                    no_fit_reason=_no_fit_reason(
                        got, pairs, n_complete, quantity, len(mine)
                    ),
                    fit_lo_gpx=min(xs) if xs else None,
                    fit_hi_gpx=max(xs) if xs else None,
                    **_common_extra(),
                )
            )
    return out


# ── the cohort, measured ─────────────────────────────────────────────────────
def _typed_point(r):
    return {
        **r,
        "input_gpx": costs.num(r.get("input_gpx")),
        **{q: costs.num(r.get(q)) for q, _ in QUANTITIES},
    }


def cohort_points(out):
    """``(source, points)``: the cohort's complete patient points on its common
    patients. From ``paper/cost_points.csv`` when ``paper`` wrote it (the paper's own
    common set, over the primary and best rows), else from the scorer's
    ``tables/summary_resources.csv`` (common over the primary rows, ``variants.PRIMARY``),
    else ``("absent", [])``."""
    from .. import variants

    out = Path(out)
    pts_path = out / COHORT_POINTS
    if pts_path.is_file():
        rows = [
            _typed_point(r)
            for r in schema.read(pts_path)
            if r.get("unit_kind") == costs.PATIENT
            and r.get("status") == costs.COMPLETE
            and costs.flag(r.get("in_common"))
            and (r.get("dataset") or T.COHORT) == T.COHORT
        ]
        return COHORT_POINTS, rows
    cost_rows = T._cost_rows(out, T.COHORT)
    if not cost_rows:
        return ABSENT, []
    res_rows = T._res_rows(out, T.COHORT)
    common = costs.common_set(cost_rows, variants.primary_for(None), T.COHORT)
    got = []
    for final in dict.fromkeys(T.terminal_rows(T.COHORT).values()):
        fam = T.family_of(final)
        if fam is None:
            continue
        mine = [r for r in cost_rows if r.get("method") == final]
        kinds = costs.row_kinds(final, (), ())
        for p in costs.patient_points(final, fam, mine, res_rows, common, kinds, {}):
            if p["status"] == costs.COMPLETE and p["in_common"]:
                got.append(_typed_point(p))
    return COHORT_SUMMARY + " (common over the primary rows)", got


def cohort_sizes(points):
    """``{at: (gpx, patient)}``: the median and the largest common patient's
    ``input_gpx`` (a patient's size is the largest any row recorded for it)."""
    size = {}
    for p in points:
        x = p["input_gpx"]
        if math.isfinite(x):
            size[p["unit_id"]] = max(size.get(p["unit_id"], 0.0), x)
    if not size:
        return {}
    vals = sorted(size.values())
    largest = max(size, key=lambda k: (size[k], k))
    return {
        "cohort_median": (costs.median(vals), None),
        "cohort_largest": (size[largest], largest),
    }


# ── synthetic_extrapolation ──────────────────────────────────────────────────
def extrapolation_table(scaling, out, dataset=DATASET):
    source, points = cohort_points(out)
    sizes = cohort_sizes(points)
    rows = []
    for s in scaling:
        if s.get("excluded_reason") not in (None, schema.NA):
            continue
        row, quantity = s["row"], s["quantity"]
        mine = [p for p in points if p.get("row") == row]
        cohort_clock = T._most_common(p.get("clock") for p in mine)
        fit_ok = s["fit_ok"] is True
        lo, hi = s["fit_lo_gpx"], s["fit_hi_gpx"]
        for at in AT:
            gpx, patient = sizes.get(at, (None, None))
            note = []
            fitted = None
            if not sizes:
                note.append(f"no cohort cost under the results folder ({source})")
            if fit_ok and gpx is not None:
                fitted = 10 ** (s["intercept"] + s["slope"] * math.log10(gpx))
            elif not fit_ok:
                note.append(f"no synthetic fit: {s['no_fit_reason']}")
            measured, low, high = None, None, None
            vals = [p[quantity] for p in mine if math.isfinite(p[quantity])]
            if at == "cohort_median":
                if vals:
                    measured = costs.median(vals)
                    low, high = costs.interval(vals)
                    low = _finite_or_none(low)
                    high = _finite_or_none(high)
            elif patient is not None:
                theirs = [p[quantity] for p in mine if p["unit_id"] == patient]
                theirs = [v for v in theirs if math.isfinite(v)]
                measured = theirs[0] if theirs else None
            if sizes and measured is None:
                note.append(f"the cohort has no measured {quantity} for {row}")
            extrapolated = None
            if gpx is not None and lo not in (None, schema.NA) and fit_ok:
                extrapolated = not (float(lo) <= gpx <= float(hi))
            rows.append(
                schema.empty_row(
                    "synthetic_extrapolation",
                    source=SOURCE,
                    dataset=dataset,
                    family=s["family"],
                    row=row,
                    quantity=quantity,
                    quantity_unit=s["quantity_unit"],
                    at=at,
                    at_gpx=gpx,
                    fit_lo_gpx=lo,
                    fit_hi_gpx=hi,
                    extrapolated=extrapolated,
                    fit_ok=fit_ok,
                    fitted=fitted,
                    measured_cohort=measured,
                    measured_ci_low=low,
                    measured_ci_high=high,
                    measured_patient=patient if at == "cohort_largest" else None,
                    gap=None
                    if fitted is None or measured is None
                    else fitted - measured,
                    gap_ratio=None
                    if fitted is None or not measured
                    else fitted / measured,
                    cohort_row=row,
                    n_cohort_patients=len(vals) if mine else 0,
                    cohort_source=source,
                    clock=s["clock"],
                    cohort_clock=cohort_clock,
                    unit=costs.PATIENT,
                    note="; ".join(note) or None,
                    **_foreground(),
                    **_common_extra(),
                )
            )
    return rows


# ── all three ────────────────────────────────────────────────────────────────
def build(out, dataset=DATASET):
    """``{table: rows}`` of the three tables, from the scorer's
    ``tables/<dataset>/summary_resources.csv`` and ``resources.csv``, with every run of
    the set's ``excluded_runs.csv`` a row of NA with its reason."""
    cost_rows = T._cost_rows(out, dataset)
    res_rows = T._res_rows(out, dataset)
    cost = cost_table(cost_rows, res_rows, dataset)
    scaling = scaling_table(cost, dataset)
    built = {
        "synthetic_cost": cost,
        "synthetic_scaling": scaling,
        "synthetic_extrapolation": extrapolation_table(scaling, out, dataset),
    }
    extra = {
        "synthetic_cost": {
            "n_channels": len(synth.CHANNELS),
            "n_panels": synth.N_PANELS,
            **_foreground(),
            **_common_extra(),
        },
        "synthetic_scaling": _common_extra(),
        "synthetic_extrapolation": {**_foreground(), **_common_extra()},
    }
    return T.add_excluded(built, T.excluded_runs(out, dataset), dataset, extra)


def write(out, target, dataset=DATASET):
    """Write the three tables into ``target``; ``{table: path}``."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    built = build(out, dataset)
    paths = {}
    for name in TABLES:
        paths[name] = target / f"{name}.csv"
        schema.write(paths[name], name, built[name])
    return paths


def write_all(out, dataset=DATASET):
    """``paper --dataset synthetic``: the three tables into ``paper/synthetic/``, and
    that folder's ``losses.csv`` (the scorer's rows of the set, unchanged: the paper
    stage adds none here)."""
    from .. import losses

    target = T._paper_dir(out, dataset)
    paths = write(out, target, dataset)
    paths["losses"] = target / "losses.csv"
    rows = losses.read_csv(T._tables(out, dataset) / "losses.csv")
    losses.write(paths["losses"], rows)
    return paths
