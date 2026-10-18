"""The paper tables, built from the tables the scorer has already written.

Reads ``<out>/tables/{cells,cells_nuclei,summary_resources,resources,stare_stage_peaks,
ashlar_discard}.csv`` (``<out>/tables/<dataset>/`` for a public dataset) and every ``<out>/<row>/<dataset>/<case>/run.json``; computes nothing
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
go to the name, device-blind (T31 R11: no cost decides a tie, since a GPU row's CPU-hours
are not its cost); a cost-only row (``variants.COST_ONLY``: ``stare_local4``) is never a
candidate.

**No matched pair is Dice 0.** ``pairs.record`` gives ``dice_matched = nan`` to a case with
no pair (an imputed failure on a large offset, or a bad registration). On a non-excluded row
it counts as 0.0 in the selection and in every Dice statistic of ``method_summary``,
``stare_tiers`` and ``valis_stages`` (``n_dice_zeroed`` counts them), so a method is never
read on its easy rounds only. A case where EVERY candidate (``best_of``), or both rows of a
paired difference, have no pair is the case's, not a method's, and is left out.

**Cost** (Task 31, ``paper.costs``: one unit, the patient; the common patients first; a
clock and a unit beside every number; a bootstrap interval beside every median; nothing
dropped silently) reads the COMPLETE patients of ``summary_resources.csv`` (every round
registered; as ``score.summarise_cost``) and counts the others (``n_capped``, ``n_failed``,
``n_excluded``). The common set is computed ONCE per dataset over the primary and the
best rows (``costs.common_set``). An excluded round makes its patient incomplete, and its
``resources.csv`` row is left out of ``cpu_model`` and ``hosts``. The old ``*_med``
columns of ``method_summary`` (over each row's own complete patients) and ``wall_kind``
stay one release beside the new ones (deprecated: ``<stem>_common_med`` /
``<stem>_survivors_med`` and ``clock`` replace them); ``fold_peak_vs_stare`` is gone
(ruling R4: it divided a sampled tree peak by one stage job's high-water mark). ``in_sample_selection`` is True for every family: the row is chosen on this cohort
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

from .. import cost as cost_mod
from .. import stages, variants
from . import costs, fits, schema

SOURCE = "regbench"
COHORT = "multiplex"
OUTLINES, NUCLEI = "outlines", "nuclei"
INITIAL = "initial"
LAPTOP_SUFFIX = "_laptop"  # score.LAPTOP_SUFFIX: a `regbench local` result
STARE_FAMILY = "stare"
VALIS_FAMILY = "valis_mirage"

WRITTEN = (
    "per_pair",
    "end_to_end",
    "method_best",
    "method_best_per_method",
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
    "cost_points",
    "cost_fits",
    "pipeline_cost_fits",
    "laptop",
)

RULE = (
    "candidates are the family's terminal rows; outlines only; a case counts for a "
    "candidate unless excluded_from_summaries (imputed failures count); compared on the "
    "cases every candidate has a non-excluded row on (n_cases_common); per candidate the "
    "patient median of dice_matched over those cases, then the median over patients; a "
    "missing dice_matched (no matched pair) on a non-excluded row counts as 0, and a case "
    "where every candidate has none is the case's and leaves the common cases; the "
    "highest wins; ties by name (device-blind: a GPU row's CPU-hours are not its cost, so "
    "no cost decides a tie); a cost-only row (variants.COST_ONLY: stare_local4) is never a "
    "candidate"
)
CRITERION = "median over patients of the patient-median dice_matched (outlines)"
TIE_BREAK = "the name (device-blind, T31 R11: no cost decides a tie)"

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


def _tables(out, dataset=COHORT):
    """The scorer's tables of ``dataset``: ``<out>/tables`` for the cohort,
    ``<out>/tables/<dataset>`` for a public dataset (``score.tables_dir``: never pooled)."""
    base = Path(out) / "tables"
    return base if dataset == COHORT else base / dataset


def _paper_dir(out, dataset):
    base = Path(out) / "paper"
    return base if dataset == COHORT else base / dataset


def _records(out, row, dataset, unreadable=None):
    """Every ``run.json`` of one result name on a dataset, in case order. One that
    cannot be read (or is no JSON object) is not a record: its path is appended to
    ``unreadable`` when a list is given, so the table can count it (S13: never a
    silent skip)."""
    found = []
    for path in sorted(Path(out, row, dataset).glob("*/run.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            rec = None
        if not isinstance(rec, dict):
            if unreadable is not None:
                unreadable.append(str(path))
            continue
        found.append(rec)
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
def _run_terminals(dataset):
    """``{run name: its last result name}`` for every active run of ``dataset``, the
    cost-only rows included (``terminal_rows`` filters them)."""
    ds = None if dataset == COHORT else dataset
    return {
        variants.run_name(r): list(variants.variants_of(r))[-1]
        for r in variants.runs_for(ds)
    }


def is_cost_only(name):
    """A COST row (``variants.COST_ONLY``: ``stare_local4``): it is in the cost tables and
    figures only, never in an accuracy table or figure (T31 fix C1)."""
    return name in set(getattr(variants, "COST_ONLY", ()))


def terminal_rows(dataset):
    """``{run name: its last result name}`` for every active run of ``dataset``:
    ``variants.variants_of(run)[-1]``. STARE's final row, VALIS's final row (the micro
    stage only at depth 2), DeeperHistReg's one row (the run's own name), ASHLAR's cell,
    PALOM's ``palom``. A cost-only run (``is_cost_only``) is left out: every accuracy
    table and figure reads this list (T31 fix C1); the cost tables add it themselves
    (``cost_points``, ``_summary_kinds``). ``versions``, ``run_parameters`` and ``hosts``
    read this list too: the cost row is ``stare``'s code on another allocation."""
    return {
        name: final
        for name, final in _run_terminals(dataset).items()
        if not (is_cost_only(name) or is_cost_only(final))
    }


def accuracy_pairs(pp):
    """``per_pair`` without the cost-only rows: what every accuracy table and figure is
    made from (T31 fix C1)."""
    return [r for r in pp if not is_cost_only(r.get("row"))]


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
        for r in _csv(_tables(out, dataset) / name):
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
STRICT, PER_METHOD = "strict", "per_method"  # score.COMMON_BLOCKS (B2)


def best_of(per_pair, family, dataset, cost=None, rule=STRICT, candidates=None):
    """The ``method_best`` row of one family (the rule: module docstring, ``RULE``).
    ``cost`` is kept for the callers and decides nothing: the tie-break is the name
    (T31 R11: device-blind). ``candidates(name)`` narrows the candidates (the CPU
    cross takes DeeperHistReg's CPU rows only); a ``variants.COST_ONLY`` row is never
    one."""
    cost_only = set(getattr(variants, "COST_ONLY", ()))
    cands = [
        t
        for t in dict.fromkeys(terminal_rows(dataset).values())
        if family_of(t) == family
        and t not in cost_only
        and (candidates is None or candidates(t))
    ]
    have = {c: _rows_of(per_pair, c, dataset) for c in cands}
    cands = [c for c in cands if _rows_of(per_pair, c, dataset, kept=False)]
    # B2: STRICT (the headline) is the cases every candidate has a kept row on; on
    # PER_METHOD a round lost for a cause that is not its own (excluded) takes the
    # case out of that candidate's own set only
    every = {c: {r["case_id"] for r in _rows_of(per_pair, c, dataset, kept=False)}
             for c in cands}  # fmt: skip
    common = None
    for c in cands:
        cases = {r["case_id"] for r in have[c]} if rule == STRICT else every[c]
        common = cases if common is None else common & cases
    common = common or set()
    # a case where no candidate has a matched pair is the case's, not a method's
    common = {
        k
        for k in common
        if not all(_no_pair(r) for c in cands for r in have[c] if r["case_id"] == k)
    }
    own = {c: common & {r["case_id"] for r in have[c]} for c in cands}
    scored = []
    zeroed = {}
    for c in cands:
        mine = [r for r in have[c] if r["case_id"] in own[c]]
        zeroed[c] = sum(_no_pair(r) for r in mine)
        value = _median(_patient_median(mine).values())
        scored.append(
            (
                0 if math.isfinite(value) else 1,
                -value if math.isfinite(value) else 0.0,
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
        best_row=best[2] if best else schema.NA,
        n_candidates=len(cands),
        criterion=CRITERION,
        criterion_value=best[3] if best else math.nan,
        n_cases_common=len(own[best[2]]) if best else len(common),
        common_block=rule,
        n_dice_zeroed=zeroed[best[2]] if best else math.nan,
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
    return [r for r in _csv(_tables(out, dataset) / "summary_resources.csv") if r.get("dataset") == dataset]


def _res_rows(out, dataset):
    return [r for r in _csv(_tables(out, dataset) / "resources.csv") if r.get("dataset") == dataset]


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


def _adapter_of(res_rows, row):
    return next((r.get("adapter") for r in res_rows if r["method"] == row and r.get("adapter")), "")


def _summary_kinds(fam, best, primary, cost_rows, dataset):
    """``[(row_kind, row)]`` of one family in ``method_summary``: its best row, its
    primary, STARE's 4-CPU row (``variants.LOCAL_4CPU``) and a GPU row's CPU twin
    (``variants.cpu_twin``), each where it has a summary line."""
    have = {r["method"] for r in cost_rows}
    picks = [(costs.BEST, best.get(fam)), (costs.PRIMARY, primary.get(fam))]
    for name in getattr(variants, "LOCAL_4CPU", ()):
        if family_of(name) == fam and name in have:
            picks.append((costs.LOCAL_4CPU, name))
    for _, row in list(picks):
        twin = variants.cpu_twin(row) if not _missing(row) else None
        if twin and twin in have and all(twin != r for _, r in picks):
            picks.append((costs.CPU_TWIN, twin))
    return [(k, r) for k, r in picks if not _missing(r)]


def _prior(row, cost_rows, res_rows, common, peaks):
    """``(prior_cpu_h, prior_s)`` of an ``affine_ashlar`` row: the median, over the
    common patients, of what the row it reads its anchor from cost up to that stage
    (R15). ``affine=stare``: STARE's COARSE (``stare_stage_peaks.coarse_cpu_h`` /
    ``coarse_s``, summed over the patient's rounds); ``affine=valis``: the
    ``valis_mirage_rigid`` row's ``cpu_h_used`` and ``register_s``. NaN otherwise."""
    try:
        run = variants.run_of(row)
    except KeyError:
        return math.nan, math.nan
    anchor = (run.opts or {}).get("affine")
    if run.method != "ashlar" or anchor in (None, "", "none"):
        return math.nan, math.nan
    if anchor == "stare":
        by = {}
        for r in peaks:
            if r.get("row") == "stare":
                p = str(r.get("patient_id"))
                c, t = _num(r.get("coarse_cpu_h")), _num(r.get("coarse_s"))
                got = by.setdefault(p, [0.0, 0.0])
                got[0] += c
                got[1] += t
        cpu = [by[p][0] for p in common if p in by]
        sec = [by[p][1] for p in common if p in by]
        return _median(cpu), _median(sec)
    rigid = f"{VALIS_FAMILY}_rigid"
    mine = [r for r in _complete(cost_rows) if r["method"] == rigid and r["group"] in common]
    return _median(r.get("cpu_h_used") for r in mine), _median(r.get("register_s") for r in mine)


def method_summary(pp, best_rows, cost_rows, res_rows, dataset, points=None, common=None, peaks=()):
    """The rows of each family (``_summary_kinds``): ``best`` (the ``method_best`` row),
    ``primary`` (``primary_of``), STARE's 4-CPU row and a GPU row's CPU twin. Accuracy as
    before; cost by ``paper.costs`` (module docstring), over the common patients first."""
    best = {r["family"]: r["best_row"] for r in best_rows}
    primary = {fam: primary_of(fam, dataset) for fam in _families(dataset)}
    stare_best = best.get(STARE_FAMILY)
    stare_best = None if _missing(stare_best) else stare_best
    excluded = _excluded_rounds(pp)
    if points is None or common is None:
        common, points = cost_points(pp, best_rows, cost_rows, res_rows, dataset)
    local4 = next(iter(getattr(variants, "LOCAL_4CPU", ())), None)
    local4_peak = (
        costs.summary_columns(points, local4, common, costs.LOCAL4)["peak_rss_gb_common_med"]
        if local4
        else math.nan
    )
    out = []
    for fam in _families(dataset):
        for which, row in _summary_kinds(fam, best, primary, cost_rows, dataset):
            # a cost-only row has no accuracy here: it is no configuration of its own
            # (T31 fix C1); its cost columns are its own
            cost_row = is_cost_only(row)
            acc = {} if cost_row else _accuracy(pp, row, dataset)[1]
            old = _cost(cost_rows, res_rows, row, excluded)
            mine = [r for r in cost_rows if r["method"] == row]
            clock = costs.clock_of(row, mine[0]) if mine else costs.WHOLE
            sc = costs.summary_columns(points, row, common, clock)
            peak = _num(sc["peak_rss_gb_common_med"])
            shared = [
                p
                for p in points
                if p["row"] == row
                and p["unit_kind"] == costs.PATIENT
                and p["status"] == costs.COMPLETE
                and p["unit_id"] in common
            ]
            adapter = _adapter_of(res_rows, row)
            work = cost_mod.WORK.get(adapter, {})
            heap = _finite(r.get("jvm_max_heap_gb") for r in mine)
            prior_cpu, _ = _prior(row, cost_rows, res_rows, common, peaks)
            cpu = _num(sc["cpu_h_used_common_med"])
            out.append(
                schema.empty_row(
                    "method_summary",
                    source=SOURCE,
                    dataset=dataset,
                    family=fam,
                    row=row,
                    which=which,
                    row_kind=which,
                    **acc,
                    **({} if cost_row else _paired(pp, row, stare_best, dataset)),
                    **old,
                    **sc,
                    total_cpu_h_used_med=sc["total_cpu_h_used_common_med"],
                    cpu_h_reserved_med=sc["cpu_h_reserved_common_med"],
                    total_cpu_h_reserved_med=sc["total_cpu_h_reserved_common_med"],
                    rung_gb_med=sc["rung_gb_common_med"],
                    wasted_attempts=int(sum(_num(p.get("wasted_attempts")) for p in shared
                                            if math.isfinite(_num(p.get("wasted_attempts"))))),
                    fold_peak_vs_stare_local4=peak / local4_peak
                    if math.isfinite(peak) and math.isfinite(_num(local4_peak)) and local4_peak > 0
                    else math.nan,
                    peak_largest_job_gb=sc["peak_largest_job_gb_common_med"],
                    peak_concurrent_gb=sc["peak_concurrent_gb_common_med"],
                    gpu_peak_gb=sc["gpu_peak_gb_common_med"],
                    convert_s_med=sc["convert_s_common_med"],
                    convert_inside_clock=work.get("convert_inside_clock", schema.NA),
                    working_px=work.get("working_px", schema.NA),
                    channels_read=work.get("channels_read", schema.NA),
                    jvm_max_heap_gb=max(heap) if heap else math.nan,
                    jvm_peak_heap_gb=JVM_PEAK_NA if heap else schema.NA,
                    prior_cpu_h=prior_cpu,
                    with_prior_cpu_h=cpu + prior_cpu
                    if math.isfinite(cpu) and math.isfinite(prior_cpu)
                    else math.nan,
                )
            )
    return out


JVM_PEAK_NA = (
    "NA: not measured (the JVM's own heap use needs a call into the JVM from the "
    "method's process, whose code is in every result's key)"
)


def _imputed(pp, res_rows):
    """``{(row, patient): rounds that registered and were scored as imputed failures}``
    (P13: costed, and counted)."""
    ok = {(r["method"], r["case_id"]) for r in res_rows if _flag(r.get("ok"))}
    got = {}
    for r in pp:
        if r["set"] != OUTLINES or r["imputed_initial"] is not True:
            continue
        if (r["row"], r["case_id"]) in ok:
            key = (r["row"], r["patient_id"])
            got[key] = got.get(key, 0) + 1
    return got


def cost_points(pp, best_rows, cost_rows, res_rows, dataset, cases=None):
    """``(common patients, cost_points rows)``: one point per patient of every terminal
    row (and of STARE's 4-CPU row), whatever became of the patient, then fig. 2c's
    per-round points (``costs.round_points``). The common set is computed ONCE, over the
    primary and the best rows (``costs.common_set``)."""
    best = [r["best_row"] for r in best_rows if not _missing(r.get("best_row"))]
    primary = [primary_of(f, dataset) for f in _families(dataset)]
    primary = [p for p in primary if p]
    compared = costs.compared_rows(best_rows, primary)
    common = costs.common_set(cost_rows, compared, dataset)
    local4 = tuple(getattr(variants, "LOCAL_4CPU", ()))
    twins = tuple(t for t in (variants.cpu_twin(r) for r in compared) if t)
    imputed = _imputed(pp, res_rows)
    excluded = _excluded_rounds(pp)
    gb = fits.gb_of(cases)
    rows = list(dict.fromkeys(terminal_rows(dataset).values()))
    rows += [r for r in local4 if r not in rows and any(c["method"] == r for c in cost_rows)]
    out = []
    for row in rows:
        fam = family_of(row)
        if fam is None:
            continue
        kinds = costs.row_kinds(row, primary, best, local4, twins)
        mine = [r for r in cost_rows if r["method"] == row]
        pts = costs.patient_points(row, fam, mine, res_rows, common, kinds, imputed, gb)
        rounds, without = costs.round_points(row, fam, res_rows, excluded, kinds, gb)
        for p in pts + rounds:
            out.append(_point_row(dataset, p))
    return common, out


def _point_row(dataset, p):
    cols = set(schema.columns("cost_points"))
    vals = {k: v for k, v in p.items() if k in cols}
    for k, v in list(vals.items()):
        if isinstance(v, float) and not math.isfinite(v):
            vals[k] = None
    return schema.empty_row("cost_points", source=SOURCE, dataset=dataset, **vals)


def end_to_end(summary, cost_rows, res_rows, e2e_rows, peaks, dataset, common):
    """``end_to_end`` (R15, P11): per ``method_summary`` row, the medians over the
    common patients of the method's own conversion, its registration, its writer and,
    for an anchored ASHLAR row, the prior it reads, from the scorer's
    ``cost_end_to_end.csv``; each row says whether the conversion is inside the
    registration's clock (``convert_inside_clock``)."""
    out = []
    for m in summary:
        row = m["row"]
        mine = [
            r for r in e2e_rows if r.get("method") == row and r.get("group") in common
        ]
        out.append(
            schema.empty_row(
                "end_to_end",
                source=SOURCE,
                dataset=dataset,
                family=m["family"],
                row=row,
                row_kind=m["row_kind"],
                clock=m["clock"],
                n_patients=len({r.get("group") for r in mine}),
                convert_s_med=_median(r.get("method_convert_s") for r in mine),
                estimate_s_med=_median(r.get("estimate_s") for r in mine),
                write_s_med=_median(r.get("write_s") for r in mine),
                prior_cpu_h=m.get("prior_cpu_h"),
                prior_s_med=_prior(row, cost_rows, res_rows, common, peaks)[1],
                convert_inside_clock=m.get("convert_inside_clock"),
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
    for r in _csv(_tables(out, dataset) / f"{name}.csv"):
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
        bad = []
        recs = [
            rec
            for v in _variants_of_name(name)
            for rec in _records(out, v, dataset, bad)
        ]
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
                n_unreadable=len(bad),
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
        bad = []
        recs = _records(out, final, dataset, bad)
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
                    n_unreadable=len(bad),
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


def laptop(out, dataset, pp):
    """One row per laptop result (``regbench local``, a row ``<run>_laptop``) and case:
    the wall to the written image (``total_to_image_s`` = register + write), the process
    tree's peak, the CPU-hours, the machine ``run.json: machine`` recorded, the tool
    version and the Dice from ``cells.csv`` (outlines). A case it was not run on has none."""
    res = {
        (r.get("method"), r.get("case_id")): r
        for r in _res_rows(out, dataset)
    }
    rows = []
    for p in pp:
        if not (p["dataset"] == dataset and p["set"] == OUTLINES):
            continue
        if not p["row"].endswith(LAPTOP_SUFFIX) or p["cause"] == "not_run":
            continue  # a case the laptop was not run on is not a laptop row
        rec = {}
        path = Path(out, p["row"], dataset, p["case_id"], "run.json")
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            pass
        mach = rec.get("machine") if isinstance(rec.get("machine"), dict) else {}
        r = res.get((p["row"], p["case_id"]), {})
        version = (rec.get("info") or {}).get("version")
        rows.append(
            schema.empty_row(
                "laptop",
                source=SOURCE,
                dataset=p["dataset"],
                patient_id=p["patient_id"],
                panel=p["panel"],
                case_id=p["case_id"],
                row=p["row"],
                wall_s=_num(r.get("total_to_image_s")),
                peak_rss_gb=_num(r.get("peak_rss_gb")),
                cpu_h_used=_num(r.get("cpu_h_used")),
                cpu_model=mach.get("cpu_model"),
                n_cores_logical=mach.get("n_cores_logical"),
                n_cores_physical=mach.get("n_cores_physical"),
                mem_total_gb=mach.get("mem_total_gb"),
                os=mach.get("os"),
                python=mach.get("python"),
                stare_version=version if version else p.get("tool_version"),
                dice_matched=p["dice_matched"],
            )
        )
    return rows


# ── build and write ──────────────────────────────────────────────────────────
def build_best(pp, dataset, cpu):
    """``method_best`` (the strict common set: the headline every other table reads)
    and ``method_best_per_method`` beside it (B2)."""
    fams = _families(dataset)
    return {
        "method_best": [best_of(pp, f, dataset, cpu) for f in fams],
        "method_best_per_method": [
            best_of(pp, f, dataset, cpu, PER_METHOD) for f in fams
        ],
    }


def build(out, dataset=COHORT, cases_root=None):
    """``{table name: rows}`` for every table of ``WRITTEN``."""
    every = per_pair(out, dataset)  # the cost tables and local4_differs read these
    pp = accuracy_pairs(every)  # every accuracy table and per_pair.csv (T31 fix C1)
    cost_rows = _cost_rows(out, dataset)
    res_rows = _res_rows(out, dataset)
    cpu = _cpu_by_row(cost_rows)
    got = build_best(pp, dataset, cpu)
    best = got["method_best"]
    peaks = _copied(out, dataset, "stare_stage_peaks")
    common, points = cost_points(
        every, best, cost_rows, res_rows, dataset, fits._cases_by_id(cases_root, dataset)
    )
    summary = method_summary(every, best, cost_rows, res_rows, dataset, points, common, peaks)
    counted = [r for r in dict.fromkeys(r["row"] for r in summary) if not is_cost_only(r)]
    e2e = _csv(_tables(out, dataset) / "cost_end_to_end.csv")
    return {
        # every row: local4_differs reads the cost row's pairs; per_pair.csv is written
        # without it (write_all: accuracy_pairs, T31 fix C1)
        "per_pair": every,
        "end_to_end": end_to_end(summary, cost_rows, res_rows, e2e, peaks, dataset, common),
        "method_best": best,
        "method_best_per_method": got["method_best_per_method"],
        "method_summary": summary,
        "valis_stages": valis_stages(pp, dataset),
        "stare_tiers": stare_tiers(pp, cost_rows, dataset),
        "stare_stage_peaks": _copied(out, dataset, "stare_stage_peaks"),
        "ashlar_discard": _copied(out, dataset, "ashlar_discard"),
        "versions": versions(out, dataset),
        "run_parameters": run_parameters(out, dataset),
        "hosts": hosts(res_rows, dataset, _excluded_rounds(every)),
        "stare_tier_table": stare_tier_table(),
        "pipeline_params": pipeline_params(),
        "cohort_counts": cohort_counts(pp, counted, dataset),
        "cost_points": points,
        "cost_fits": fits.method_fits(points),
        "pipeline_cost_fits": fits.pipeline_fits(out),
        "laptop": laptop(out, dataset, pp),
    }


LOCAL4_TOLERANCE = 1e-6  # the same registration scores the same dice, to round-off


def local4_differs(per_pair, shown):
    """The ``stare_local4_differs`` loss rows (T31 R2): ``stare_local4`` is a COST row
    of ``stare`` (its stages in one job of 4 CPUs) and must score every case it shares
    with ``stare`` the same (``dice_matched`` of the outlines, per stage). A case where the
    two differ, or where one has a dice and the other none, is a row (blame
    ``integrity``, effect ``flagged``): the cost row is then not of the same
    registration."""
    from .. import losses

    local = getattr(variants, "STARE_LOCAL4", None)
    if not local:
        return []

    def dice(row):
        return {
            (r["case_id"], r.get("stage")): (r.get("patient_id"), _num(r.get("dice_matched")))
            for r in per_pair
            if r.get("row") == row and r.get("set") == OUTLINES
        }

    mine, theirs = dice(local), dice("stare")
    out = []
    for key in sorted(set(mine) & set(theirs), key=str):
        case = key[0]
        pid, a = mine[key]
        b = theirs[key][1]
        same = (math.isnan(a) and math.isnan(b)) or (
            math.isfinite(a) and math.isfinite(b) and abs(a - b) <= LOCAL4_TOLERANCE)
        if not same:
            out.append(losses.row(
                stage="paper", dataset=shown, patient_id=pid or "", case_id=case, run=local,
                method="stare", cause="stare_local4_differs", effect="flagged",
                detail=f"dice_matched {a:g} ({local}) against {b:g} (stare)",
                source="paper/per_pair.csv",
            ))
    return out


def paper_losses(built, dataset):
    """The paper stage's loss rows (M3): one per cost fit that left points out (S12)
    and one per run whose run.json files the paper tables could not all read (S13),
    (one row per run, from versions, which reads every variant, else from
    run_parameters), so ``paper/losses.csv`` holds them beside ``tables/losses.csv``."""
    from .. import losses

    shown = "multiplex" if dataset == COHORT else dataset
    rows = []
    for table in ("cost_fits", "pipeline_cost_fits"):
        for f in built[table]:
            n = _int(f.get("n_dropped"))
            if not n > 0:  # nan (no count) is not a loss
                continue
            what = ", ".join(
                f"{k}={f[k]}"
                for k in ("dataset", "row", "backend", "process", "target", "predictor")
                if not _missing(f.get(k))
            )
            rows.append(losses.row(
                stage="paper", dataset=shown, run=f.get("row") or "",
                cause="fit_points_dropped", effect="dropped",
                detail=f"{n} point(s) left out of the fit ({what})",
                source=f"paper/{table}.csv",
            ))
    rows += local4_differs(built["per_pair"], shown)
    seen = set()  # one row per run: versions reads every variant, so it comes first
    for table in ("versions", "run_parameters"):
        for r in built[table]:
            n = _int(r.get("n_unreadable"))
            if not n > 0 or r["row"] in seen:
                continue
            seen.add(r["row"])
            rows.append(losses.row(
                stage="paper", dataset=shown, run=r["row"], method=r.get("method") or "",
                cause="paper_record_unreadable", effect="dropped",
                detail=f"{n} run.json file(s) could not be read",
                source=f"paper/{table}.csv",
            ))
    return rows


def write_all(out, dataset=COHORT, cases_root=None):
    """Write every table of ``WRITTEN`` and ``losses.csv`` (the paper stage's losses,
    ALWAYS written); ``{table name: path}``."""
    from .. import losses

    built = build(out, dataset, cases_root)
    target = _paper_dir(out, dataset)
    target.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in WRITTEN:
        path = target / f"{name}.csv"
        rows = accuracy_pairs(built[name]) if name == "per_pair" else built[name]
        schema.write(path, name, rows)
        paths[name] = path
    paths["losses"] = target / "losses.csv"
    losses.write(paths["losses"], paper_losses(built, dataset))
    return paths


def main(a):
    paths = write_all(a.out, a.dataset, getattr(a, "cases", None))
    for name, path in paths.items():
        print(f"{name}\t{path}")
    return 0
