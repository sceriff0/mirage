"""The paper tables, built from the tables the scorer has already written.

Reads ``<out>/tables/{cells,cells_nuclei,summary_resources,resources,stare_stage_peaks,
ashlar_discard}.csv`` and every ``<out>/<row>/<dataset>/<case>/run.json``; computes nothing
the scorer computes. Writes ``<out>/paper/<table>.csv`` for the cohort (``multiplex``) and
``<out>/paper/<dataset>/<table>.csv`` for any other dataset, in the columns of
``paper.schema`` (``schema.columns``), one table per name of ``WRITTEN``. Every row has
``source = regbench``.

**Missing values.** ``""``, ``nan`` and ``NA`` are missing. A missing number never enters a
median and is never turned into a number; a row whose number is missing still counts in
the counts (``n_pairs``, ``n_ok``). An imputed failure (``imputed_initial``: scored at the
unregistered pose) is never dropped; a row ``excluded_from_summaries`` (lost for a cause
that is not the method's) stays in ``per_pair``, flagged, and enters no other table.

**The best row of a method** (``best_of``; written into every ``method_best`` row as
``rule``): the candidates are the family's terminal rows (``terminal_rows``: the last
result name of each active run); outlines only; a case counts for a candidate unless it is
``excluded_from_summaries`` (imputed failures COUNT); the comparison is on the cases every
candidate has a non-excluded row on (``n_cases_common``); per candidate, the patient median
of ``dice_matched`` over those cases, then the median over patients; the highest wins; ties
go to the lower median of ``cpu_h_used`` (``summary_resources.csv``) over the patients every
candidate completed, then to the name.

**No matched pair is Dice 0.** ``pairs.record`` gives ``dice_matched = nan`` to a case with
no pair (an imputed failure on a large offset, or a bad registration). On a non-excluded row
it counts as 0.0 in the selection and in every Dice statistic of ``method_summary``,
``stare_tiers`` and ``valis_stages`` (``n_dice_zeroed`` counts them), so a method is never
read on its easy rounds only. A case where EVERY candidate (``best_of``), or both rows of a
paired difference, have no pair is the case's, not a method's, and is left out.

**Cost** reads the COMPLETE patients of ``summary_resources.csv`` only (every round
registered; as ``score.summarise_cost``): a patient's sums there cover its registered rounds
only. An excluded round makes its patient incomplete, and its ``resources.csv`` row is left
out of ``cpu_model`` and ``hosts``. ``in_sample_selection`` is True for every family: the row is chosen on this cohort
with this metric.

**VALIS stages** (ruling R18, ``methods.valis_mirage.STAGES_AT_DEPTH``). At depth 0 there
is no micro pass. At depth 1 the micro-RIGID pass runs inside ``register()``, so the
``_rigid`` row is affine + micro-rigid and the final row equals ``_nonrigid``. A non-rigid
``micro`` stage exists only at depth 2. ``valis_stages`` therefore has five stage slots per
run: ``initial``, ``rigid`` (the ``_rigid`` row at depth 0 only), ``micro_rigid`` (the
``_rigid`` row at depth 1 and 2), ``nonrigid`` and ``micro`` (the final row at depth 2
only); a slot the run does not have is ``NA``.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from .. import stages, variants
from . import schema

SOURCE = "regbench"
COHORT = "multiplex"
OUTLINES, NUCLEI = "outlines", "nuclei"
INITIAL = "initial"
STARE_FAMILY = "stare"
VALIS_FAMILY = "valis_mirage"

WRITTEN = (
    "per_pair",
    "method_best",
    "method_summary",
    "valis_stages",
    "stare_tiers",
    "stare_stage_peaks",
    "ashlar_discard",
    "versions",
    "run_parameters",
    "hosts",
    "stare_tier_table",
    "pipeline_params",
    "cohort_counts",
)

RULE = (
    "candidates are the family's terminal rows; outlines only; a case counts for a "
    "candidate unless excluded_from_summaries (imputed failures count); compared on the "
    "cases every candidate has a non-excluded row on (n_cases_common); per candidate the "
    "patient median of dice_matched over those cases, then the median over patients; a "
    "missing dice_matched (no matched pair) on a non-excluded row counts as 0, and a case "
    "where every candidate has none is the case's and leaves the common cases; the "
    "highest wins; ties by the lower cpu_h_used median over the patients every candidate "
    "completed, then by name"
)
CRITERION = "median over patients of the patient-median dice_matched (outlines)"
TIE_BREAK = "lower cpu_h_used median over the patients every candidate completed, then the name"

# a module-level name, so a test can give the rows a toy family map
family_of = variants.family

NUMERIC = (
    "dice_matched",
    "iou_p50",
    "pair_fraction",
    "pair_fraction_null_max",
    "displacement_um_p50",
    "displacement_um_p90",
    "within_1um",
    "n_pairs",
    "n_ref",
    "n_moving",
)
FLAGS = ("imputed_initial", "excluded_from_summaries", "displacement_readable")

REPO = Path(__file__).resolve().parents[3]
SCHEMA_JSON = REPO / "nextflow_schema.json"


# ── reading ──────────────────────────────────────────────────────────────────
def _num(v):
    """A float, or nan for a missing value (``""``, ``nan``, ``NA``, None)."""
    if v is None or isinstance(v, bool):
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _flag(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _missing(v):
    return v is None or v == "" or v == schema.NA or (isinstance(v, float) and math.isnan(v))


def _finite(values):
    out = []
    for v in values:
        f = _num(v)
        if math.isfinite(f):
            out.append(f)
    return out


def _median(values):
    vals = _finite(values)
    return float(statistics.median(vals)) if vals else math.nan


def _mean(values):
    vals = _finite(values)
    return float(statistics.fmean(vals)) if vals else math.nan


def _quartiles(values):
    vals = _finite(values)
    if not vals:
        return math.nan, math.nan
    q25, q75 = np.percentile(vals, [25, 75])
    return float(q25), float(q75)


def _distinct(values):
    seen = [str(v) for v in values if not _missing(v)]
    return "|".join(dict.fromkeys(sorted(seen)))


def _int(v):
    f = _num(v)
    return int(f) if math.isfinite(f) else math.nan


def _csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _tables(out):
    return Path(out) / "tables"


def _paper_dir(out, dataset):
    base = Path(out) / "paper"
    return base if dataset == COHORT else base / dataset


def _records(out, row, dataset):
    """Every ``run.json`` of one result name on a dataset, in case order."""
    found = []
    for path in sorted(Path(out, row, dataset).glob("*/run.json")):
        try:
            found.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return found


def _run(name):
    """The run stored under ``name``, or None for a name of no run (a toy row)."""
    try:
        return variants.run_of(name)
    except KeyError:
        return None


def _variants_of_name(name):
    run = _run(name)
    return list(variants.variants_of(run)) if run is not None else [name]


# ── the rows and their stages ────────────────────────────────────────────────
def terminal_rows(dataset):
    """``{run name: its last result name}`` for every active run of ``dataset``:
    ``variants.variants_of(run)[-1]``. STARE's final row, VALIS's final row (the micro
    stage only at depth 2), DeeperHistReg's one row (the run's own name), ASHLAR's cell,
    PALOM's ``palom``."""
    ds = None if dataset == COHORT else dataset
    return {
        variants.run_name(r): list(variants.variants_of(r))[-1]
        for r in variants.runs_for(ds)
    }


def _valis_depth(run):
    from ..methods import valis_mirage

    return int(run.opts.get("micro_reg", valis_mirage.ARGS["micro_reg"]))


def _valis_tier(run):
    from ..methods import valis_mirage

    return str(run.opts.get("memory_mode", valis_mirage.ARGS["memory_mode"]))


def _owner(variant):
    """``(run, index of the variant in its run's names)`` or ``(None, None)``."""
    for run in variants.RUNS:
        names = list(variants.variants_of(run))
        if variant in names:
            return run, names.index(variant)
    return None, None


def stage_of(variant):
    """The stage a result name is: ``initial``, ``rigid``, ``micro_rigid``, ``nonrigid``,
    ``micro`` or ``final`` (the terminal row of a method with no named stage). The VALIS
    rows follow R18 (module docstring)."""
    name = str(variant).split(":", 1)[0]
    if name == INITIAL:
        return INITIAL
    run, at = _owner(name)
    if run is not None and run.method == VALIS_FAMILY:
        depth = _valis_depth(run)
        kind = ("rigid", "nonrigid", "final")[at]
        if kind == "rigid":
            return "micro_rigid" if depth >= 1 else "rigid"
        if kind == "nonrigid":
            return "nonrigid"
        return "micro" if depth >= 2 else "nonrigid"
    if name.endswith("_rigid") or name.endswith("_affine"):
        return "rigid"
    if name.endswith("_nonrigid"):
        return "nonrigid"
    if name.endswith("_micro"):
        return "micro"
    if run is not None and at < len(list(variants.variants_of(run))) - 1:
        return "nonrigid"  # an unsuffixed middle row (VALIS 1.2.0's `valis120`)
    return "final"


def per_pair(out, dataset):
    """One row per (case, result, polygon set) of ``cells.csv`` / ``cells_nuclei.csv``,
    typed; off-seam rows (``<row>:non_<owner>``) are not in it."""
    rows = []
    for name in ("cells.csv", "cells_nuclei.csv"):
        for r in _csv(_tables(out) / name):
            if r.get("dataset") != dataset or ":" in r.get("method", ""):
                continue
            num = {k: _num(r.get(k)) for k in NUMERIC}
            flags = {k: _flag(r.get(k)) for k in FLAGS}
            method = r["method"]
            fam = family_of(method)
            rows.append(
                schema.empty_row(
                    "per_pair",
                    source=SOURCE,
                    dataset=r["dataset"],
                    patient_id=r.get("patient_id") or r.get("group"),
                    panel=r.get("panel"),
                    case_id=r["case_id"],
                    row=method,
                    family=fam if fam is not None else schema.NA,
                    set=r.get("set") or (NUCLEI if name == "cells_nuclei.csv" else OUTLINES),
                    stage=stage_of(method),
                    tool_version=r.get("tool_version"),
                    frame=r.get("frame"),
                    dice_matched=num["dice_matched"],
                    iou_p50=num["iou_p50"],
                    pair_fraction=num["pair_fraction"],
                    pair_fraction_null_max=num["pair_fraction_null_max"],
                    displacement_um_p50=num["displacement_um_p50"],
                    displacement_um_p90=num["displacement_um_p90"],
                    displacement_readable=flags["displacement_readable"],
                    within_1um=num["within_1um"],
                    n_pairs=_int(num["n_pairs"]),
                    n_ref=_int(num["n_ref"]),
                    n_mov=_int(num["n_moving"]),
                    imputed_initial=flags["imputed_initial"],
                    excluded_from_summaries=flags["excluded_from_summaries"],
                    cause=r.get("cause"),
                    selfcheck_status=r.get("selfcheck_status"),
                )
            )
    return rows


def _rows_of(pp, row, dataset, kept=True):
    """The outline rows of one result name on a dataset; ``kept``: without the excluded."""
    return [
        r
        for r in pp
        if r["row"] == row
        and r["dataset"] == dataset
        and r["set"] == OUTLINES
        and not (kept and r["excluded_from_summaries"] is True)
    ]


def _dice0(r):
    """A kept row's Dice, with a missing one (no matched pair) counted as 0.0."""
    d = _num(r["dice_matched"])
    return d if math.isfinite(d) else 0.0


def _no_pair(r):
    return not math.isfinite(_num(r["dice_matched"]))


def _patient_median(rows):
    by = {}
    for r in rows:
        by.setdefault(r["patient_id"], []).append(_dice0(r))
    return {p: _median(v) for p, v in by.items()}


# ── the best row of a method ─────────────────────────────────────────────────
def best_of(per_pair, family, dataset, cost):
    """The ``method_best`` row of one family (the rule: module docstring, ``RULE``).
    ``cost`` is ``{result name: {patient: cpu_h_used}}`` over complete patients
    (``_cpu_by_row``; may be empty)."""
    cands = [
        t for t in dict.fromkeys(terminal_rows(dataset).values()) if family_of(t) == family
    ]
    have = {c: _rows_of(per_pair, c, dataset) for c in cands}
    cands = [c for c in cands if _rows_of(per_pair, c, dataset, kept=False)]
    common = None
    for c in cands:
        cases = {r["case_id"] for r in have[c]}
        common = cases if common is None else common & cases
    common = common or set()
    # a case where no candidate has a matched pair is the case's, not a method's
    common = {
        k
        for k in common
        if not all(_no_pair(r) for c in cands for r in have[c] if r["case_id"] == k)
    }
    cost = cost or {}
    shared = None  # the patients every candidate completed
    for c in cands:
        mine_p = set((cost.get(c) or {}))
        shared = mine_p if shared is None else shared & mine_p
    shared = shared or set()
    scored = []
    zeroed = {}
    for c in cands:
        mine = [r for r in have[c] if r["case_id"] in common]
        zeroed[c] = sum(_no_pair(r) for r in mine)
        value = _median(_patient_median(mine).values())
        cpu = _median((cost.get(c) or {}).get(p) for p in shared)
        scored.append(
            (
                0 if math.isfinite(value) else 1,
                -value if math.isfinite(value) else 0.0,
                cpu if math.isfinite(cpu) else math.inf,
                c,
                value,
            )
        )
    scored.sort()
    best = scored[0] if scored else None
    return schema.empty_row(
        "method_best",
        source=SOURCE,
        dataset=dataset,
        family=family,
        best_row=best[3] if best else schema.NA,
        n_candidates=len(cands),
        criterion=CRITERION,
        criterion_value=best[4] if best else math.nan,
        n_cases_common=len(common),
        n_dice_zeroed=zeroed[best[3]] if best else math.nan,
        tie_break=TIE_BREAK,
        in_sample_selection=True,
        rule=RULE,
    )


def primary_of(family, dataset=COHORT):
    """The family's primary result name on ``dataset`` (``variants.primary_for``), or
    None: also when the family's primary was not run on that dataset."""
    for v in variants.primary_for(None if dataset == COHORT else dataset):
        if family_of(v) == family:
            return v
    return None


def _families(dataset):
    out = []
    for t in terminal_rows(dataset).values():
        f = family_of(t)
        if f is not None and f not in out:
            out.append(f)
    return out


# ── cost ─────────────────────────────────────────────────────────────────────
def _cost_rows(out, dataset):
    return [r for r in _csv(_tables(out) / "summary_resources.csv") if r.get("dataset") == dataset]


def _res_rows(out, dataset):
    return [r for r in _csv(_tables(out) / "resources.csv") if r.get("dataset") == dataset]


def _complete(cost_rows):
    """The patients whose every round registered (``summary_resources.csv: complete``),
    as ``score.summarise_cost`` reads them: the cost of some of a patient's rounds is not
    the cost of the patient."""
    return [r for r in cost_rows if _flag(r.get("complete"))]


def _cpu_by_row(cost_rows):
    """``{result name: {patient: cpu_h_used}}`` over complete patients."""
    by = {}
    for r in _complete(cost_rows):
        v = _num(r.get("cpu_h_used"))
        if math.isfinite(v):
            by.setdefault(r["method"], {})[r["group"]] = v
    return by


def _most_common(values):
    vals = [str(v) for v in values if not _missing(v)]
    return Counter(vals).most_common(1)[0][0] if vals else schema.NA


def _peak_unit(mine):
    if any(not _missing(r.get("n_jobs")) for r in mine):
        return "per task (largest job)"
    if any(r.get("unit") == "patient" for r in mine):
        return "per patient launch"
    return "per round"


# ── method_summary ───────────────────────────────────────────────────────────
def _accuracy(pp, row, dataset):
    mine = _rows_of(pp, row, dataset)
    dice = [_dice0(r) for r in mine]
    q25, q75 = _quartiles(dice)
    readable = [r["displacement_um_p50"] for r in mine if r["displacement_readable"] is True]
    d25, d75 = _quartiles(readable)
    return mine, {
        "n_dice_zeroed": sum(_no_pair(r) for r in mine),
        "tool_version": _distinct(r["tool_version"] for r in mine),
        "n_patients": len({r["patient_id"] for r in mine}),
        "n_pairs": len(mine),
        "n_ok": sum(r["imputed_initial"] is not True for r in mine),
        "dice_med": _median(dice),
        "dice_q25": q25,
        "dice_q75": q75,
        "dice_mean": _mean(dice),
        "disp_um_med": _median(readable),
        "disp_um_q25": d25,
        "disp_um_q75": d75,
        "n_readable": len(_finite(readable)),
    }


def _paired(pp, row, stare_row, dataset):
    """Per case, ``row - stare_row`` on the cases both have and neither is excluded. A
    missing Dice counts as 0, except on a case where both are missing (the case's)."""
    if stare_row is None:
        return {}
    a = {r["case_id"]: _num(r["dice_matched"]) for r in _rows_of(pp, row, dataset)}
    b = {r["case_id"]: _num(r["dice_matched"]) for r in _rows_of(pp, stare_row, dataset)}
    both = [c for c in a if c in b and (math.isfinite(a[c]) or math.isfinite(b[c]))]
    a = {c: (v if math.isfinite(v) else 0.0) for c, v in a.items()}
    b = {c: (v if math.isfinite(v) else 0.0) for c, v in b.items()}
    diffs = [a[c] - b[c] for c in both]
    q25, q75 = _quartiles(diffs)
    return {
        "d_dice_vs_stare_med": _median(diffs),
        "d_dice_vs_stare_q25": q25,
        "d_dice_vs_stare_q75": q75,
        "n_pairs_vs_stare": len(both),
        "n_pairs_stare_ge": sum(b[c] >= a[c] for c in both),
    }


def _excluded_rounds(pp):
    """``{(case_id, row)}`` of the rounds excluded from the summaries."""
    return {(r["case_id"], r["row"]) for r in pp if r["excluded_from_summaries"] is True}


def _cost(cost_rows, res_rows, row, excluded=frozenset()):
    """Cost per patient over the COMPLETE patients (``_complete``); an excluded round
    makes its patient incomplete, and its ``resources.csv`` row is left out of
    ``cpu_model``."""
    every = [r for r in cost_rows if r["method"] == row]
    mine = _complete(every)
    jobs = any(not _missing(r.get("n_jobs")) for r in mine)
    wall_key = "wall_unlimited_parallel_s" if jobs else "register_s"
    peaks = [r.get("peak_rss_gb") for r in mine]
    return {
        "peak_rss_gb_med": _median(peaks),
        "peak_rss_gb_max": max(_finite(peaks)) if _finite(peaks) else math.nan,
        "peak_rss_unit": _peak_unit(mine or every) if every else schema.NA,
        "n_patients_complete": len({r["group"] for r in mine}),
        "cpu_h_used_med": _median(r.get("cpu_h_used") for r in mine),
        "wall_s_med": _median(r.get(wall_key) for r in mine),
        "wall_kind": "time_unlimited_parallel_tiles_s" if jobs else "register_s",
        "gpu_h_med": _median(r.get("gpu_h") for r in mine),
        "device": _distinct(r.get("device") for r in mine),
        "gpu_model": _distinct(r.get("gpu_model") for r in mine),
        "cpu_model": _most_common(
            r.get("cpu_model")
            for r in res_rows
            if r["method"] == row and (r.get("case_id"), row) not in excluded
        ),
    }


def method_summary(pp, best_rows, cost_rows, res_rows, dataset):
    """Two rows per family: ``best`` (the ``method_best`` row) and ``primary``
    (``primary_of``; a family with none has the ``best`` row only)."""
    best = {r["family"]: r["best_row"] for r in best_rows}
    stare_best = best.get(STARE_FAMILY)
    stare_best = None if _missing(stare_best) else stare_best
    excluded = _excluded_rounds(pp)
    stare_peak = (
        _cost(cost_rows, res_rows, stare_best, excluded)["peak_rss_gb_med"]
        if stare_best
        else math.nan
    )
    out = []
    for fam in _families(dataset):
        picks = [("best", best.get(fam)), ("primary", primary_of(fam, dataset))]
        for which, row in picks:
            if _missing(row):
                continue
            _, acc = _accuracy(pp, row, dataset)
            cost = _cost(cost_rows, res_rows, row, excluded)
            fold = (
                cost["peak_rss_gb_med"] / stare_peak
                if math.isfinite(_num(cost["peak_rss_gb_med"]))
                and math.isfinite(stare_peak)
                and stare_peak > 0
                else math.nan
            )
            out.append(
                schema.empty_row(
                    "method_summary",
                    source=SOURCE,
                    dataset=dataset,
                    family=fam,
                    row=row,
                    which=which,
                    **acc,
                    **_paired(pp, row, stare_best, dataset),
                    **cost,
                    fold_peak_vs_stare=fold,
                )
            )
    return out


# ── per-method tables ────────────────────────────────────────────────────────
def _stage_stats(pp, row, dataset):
    if _missing(row):
        return {}
    mine = _rows_of(pp, row, dataset)
    dice = [_dice0(r) for r in mine]
    q25, q75 = _quartiles(dice)
    readable = [r["displacement_um_p50"] for r in mine if r["displacement_readable"] is True]
    return {
        "dice_med": _median(dice),
        "dice_q25": q25,
        "dice_q75": q75,
        "n_pairs": len(mine),
        "disp_um_med": _median(readable),
    }


def valis_stages(pp, dataset):
    """Five stage slots per ``valis_mirage`` run (R18, module docstring)."""
    out = []
    for name, final in terminal_rows(dataset).items():
        run = _run(name)
        if run is None or run.method != VALIS_FAMILY or family_of(final) != VALIS_FAMILY:
            continue
        rigid, nonrigid, last = list(variants.variants_of(run))
        depth = _valis_depth(run)
        slots = (
            ("initial", INITIAL),
            ("rigid", rigid if depth == 0 else None),
            ("micro_rigid", rigid if depth >= 1 else None),
            ("nonrigid", nonrigid),
            ("micro", last if depth >= 2 else None),
        )
        for stage, row in slots:
            out.append(
                schema.empty_row(
                    "valis_stages",
                    source=SOURCE,
                    dataset=dataset,
                    row=final,
                    memory_mode=_valis_tier(run),
                    micro_reg=str(depth),
                    stage=stage,
                    stage_row=row if row is not None else schema.NA,
                    **_stage_stats(pp, row, dataset),
                )
            )
    return out


def stare_tiers(pp, cost_rows, dataset):
    """One row per STARE run: tier, stride and preprocessing from its options (defaults
    ``high``, 64, ``none``), accuracy and cost."""
    out = []
    for name, final in terminal_rows(dataset).items():
        if family_of(final) != STARE_FAMILY:
            continue
        run = _run(name)
        opts = run.opts if run is not None else {}
        stats = _stage_stats(pp, final, dataset)
        mine = [r for r in _complete(cost_rows) if r["method"] == final]
        peaks = _finite(r.get("peak_rss_gb") for r in mine)
        out.append(
            schema.empty_row(
                "stare_tiers",
                source=SOURCE,
                dataset=dataset,
                row=final,
                tier=opts.get("tier", "high"),
                stride=int(opts.get("stride", 64)),
                sprep=opts.get("sprep", "none"),
                dice_med=stats["dice_med"],
                dice_q25=stats["dice_q25"],
                dice_q75=stats["dice_q75"],
                n_pairs=stats["n_pairs"],
                peak_task_gb_max=max(peaks) if peaks else math.nan,
                cpu_h_used_med=_median(r.get("cpu_h_used") for r in mine),
            )
        )
    return out


def _copied(out, dataset, name):
    """The scorer's own ``<name>.csv``, with ``source``: it has the schema's columns but
    the first."""
    cols = schema.columns(name)
    rows = []
    for r in _csv(_tables(out) / f"{name}.csv"):
        if r.get("dataset") != dataset:
            continue
        rows.append(
            schema.empty_row(
                name,
                source=SOURCE,
                **{c: (None if r.get(c) in ("nan",) else r.get(c)) for c in cols[1:] if c in r},
            )
        )
    return rows


def versions(out, dataset):
    """Per run: the tool and the distinct ``info.version`` values over its records."""
    rows = []
    for name, final in terminal_rows(dataset).items():
        run = _run(name)
        recs = [rec for v in _variants_of_name(name) for rec in _records(out, v, dataset)]
        mine = _records(out, final, dataset)
        infos = [rec.get("info") or {} for rec in recs]
        tools = [((i.get("implementation") or {}).get("package")) for i in infos]
        vers = [i.get("version") for i in infos]
        rows.append(
            schema.empty_row(
                "versions",
                source=SOURCE,
                row=final,
                method=run.method if run is not None else (recs[0].get("method") if recs else None),
                tool=_most_common(tools) if any(not _missing(t) for t in tools) else (
                    run.method if run is not None else None),
                tool_version=_most_common(vers),
                versions_seen=_distinct(vers),
                image=f"REGBENCH_SIF_{run.image}" if run is not None else None,
                n_results=len(mine),
            )
        )
    return rows


def _row(table, values):
    """``schema.empty_row`` for values whose names clash with its own argument
    (``name``)."""
    row = schema.empty_row(table)
    unknown = set(values) - set(row)
    if unknown:
        raise KeyError(f"{table}: no column {sorted(unknown)}")
    row.update(values)
    return row


def _value(v):
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True)


def run_parameters(out, dataset):
    """Per run: every key of ``info.implementation.parameters`` when the run recorded a
    dict there (Table S6), else the run's options."""
    rows = []
    for name, final in terminal_rows(dataset).items():
        run = _run(name)
        recs = _records(out, final, dataset)
        params = None
        for rec in recs:
            p = ((rec.get("info") or {}).get("implementation") or {}).get("parameters")
            if isinstance(p, dict):
                params = p
                break
        method = run.method if run is not None else (recs[0].get("method") if recs else None)
        if params is not None:
            items, src = params.items(), "run.json info.implementation.parameters"
        else:
            opts = run.opts if run is not None else ((recs[0].get("opts") or {}) if recs else {})
            items, src = opts.items(), "the run's options"
            if not opts:
                items, src = [(None, None)], "the run's options (none: the method's defaults)"
        for k, v in items:
            rows.append(
                schema.empty_row(
                    "run_parameters",
                    source=SOURCE,
                    row=final,
                    method=method,
                    key=k,
                    value=None if v is None else _value(v),
                    **{"from": src},
                )
            )
    return rows


def hosts(res_rows, dataset, excluded=frozenset()):
    """Where each terminal row ran, from ``resources.csv``, without its excluded rounds."""
    rows = []
    for final in terminal_rows(dataset).values():
        mine = [
            r
            for r in res_rows
            if r["method"] == final and (r.get("case_id"), final) not in excluded
        ]
        rows.append(
            schema.empty_row(
                "hosts",
                source=SOURCE,
                row=final,
                cpu_model=_distinct(r.get("cpu_model") for r in mine),
                cpus_requested=_distinct(
                    r.get("cpus_requested") for r in mine if r.get("cpus_requested") != "nan"
                ),
                mem_requested_gb=_distinct(
                    r.get("mem_requested_gb") for r in mine if r.get("mem_requested_gb") != "nan"
                ),
                n_results=len(mine),
            )
        )
    return rows


def stare_tier_table():
    """Table S3: one row per tier of ``methods.stare.TIERS`` (``lib/RegPresets.groovy``).
    ``read_box_px_s64`` is STARE's read box for an interior tile at stride 64 (tile +
    2 x 3 x 64); the memories are ``stages.ALLOCATION`` attempt 1 (the high tier's; at
    stride 64 ``conf/modules.config``'s formula floors at the same 4 GB for every tier)."""
    from ..methods import stare

    tile_gb = stages.allocation(stages.TILE, 1)["mem_gb"]
    stitch_gb = stages.allocation(stages.STITCH, 1)["mem_gb"]
    return [
        schema.empty_row(
            "stare_tier_table",
            source=SOURCE,
            tier=tier,
            tile=row["tile"],
            halo=row["halo"],
            out_tile=row["out_tile"],
            coarse_max_dim=row["max_dim"],
            read_box_px_s64=row["tile"] + 2 * 3 * 64,
            reg_tile_mem_gb=tile_gb,
            stitch_mem_gb=stitch_gb,
        )
        for tier, row in stare.TIERS.items()
    ]


def pipeline_params(path=SCHEMA_JSON):
    """Table S1: every property of ``nextflow_schema.json`` (its ``$defs`` /
    ``definitions`` and their ``properties``, and any top-level ``properties``)."""
    doc = json.loads(Path(path).read_text())
    groups = [doc.get("$defs") or {}, doc.get("definitions") or {}]
    props = {}
    for group in groups:
        for block in group.values():
            props.update(block.get("properties") or {})
    props.update(doc.get("properties") or {})
    rows = []
    for name, p in props.items():
        typ = p.get("type")
        rows.append(
            _row(
                "pipeline_params",
                {
                    "source": SOURCE,
                    "name": name,
                    "default": None if p.get("default") is None else _value(p["default"]),
                    "type": "|".join(typ) if isinstance(typ, list) else typ,
                    "description": p.get("description"),
                },
            )
        )
    return rows


def cohort_counts(pp, rows, dataset):
    """Per patient, per primary and best row: moving panels, and matched nuclei per pair
    (min, median, max over the row's cases of the patient)."""
    out = []
    for row in rows:
        mine = _rows_of(pp, row, dataset)  # an excluded round is in no count
        for patient in dict.fromkeys(r["patient_id"] for r in mine):
            of_p = [r for r in mine if r["patient_id"] == patient]
            n = _finite(r["n_pairs"] for r in of_p)
            out.append(
                schema.empty_row(
                    "cohort_counts",
                    source=SOURCE,
                    dataset=dataset,
                    patient_id=patient,
                    row=row,
                    n_panels=len({r["case_id"] for r in of_p}),
                    n_pairs_min=int(min(n)) if n else math.nan,
                    n_pairs_med=_median(n),
                    n_pairs_max=int(max(n)) if n else math.nan,
                )
            )
    return out


# ── build and write ──────────────────────────────────────────────────────────
def build(out, dataset=COHORT):
    """``{table name: rows}`` for every table of ``WRITTEN``."""
    pp = per_pair(out, dataset)
    cost_rows = _cost_rows(out, dataset)
    res_rows = _res_rows(out, dataset)
    cpu = _cpu_by_row(cost_rows)
    best = [best_of(pp, fam, dataset, cpu) for fam in _families(dataset)]
    summary = method_summary(pp, best, cost_rows, res_rows, dataset)
    counted = list(dict.fromkeys(r["row"] for r in summary))
    return {
        "per_pair": pp,
        "method_best": best,
        "method_summary": summary,
        "valis_stages": valis_stages(pp, dataset),
        "stare_tiers": stare_tiers(pp, cost_rows, dataset),
        "stare_stage_peaks": _copied(out, dataset, "stare_stage_peaks"),
        "ashlar_discard": _copied(out, dataset, "ashlar_discard"),
        "versions": versions(out, dataset),
        "run_parameters": run_parameters(out, dataset),
        "hosts": hosts(res_rows, dataset, _excluded_rounds(pp)),
        "stare_tier_table": stare_tier_table(),
        "pipeline_params": pipeline_params(),
        "cohort_counts": cohort_counts(pp, counted, dataset),
    }


def write_all(out, dataset=COHORT):
    """Write every table of ``WRITTEN``; ``{table name: path}``."""
    built = build(out, dataset)
    target = _paper_dir(out, dataset)
    target.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in WRITTEN:
        path = target / f"{name}.csv"
        schema.write(path, name, built[name])
        paths[name] = path
    return paths


def main(a):
    paths = write_all(a.out, a.dataset)
    for name, path in paths.items():
        print(f"{name}\t{path}")
    return 0
