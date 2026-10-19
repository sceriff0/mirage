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

**Poolings, sets and Dice rules** (T29). Every accuracy table (``method_summary``,
``valis_stages``, ``stare_tiers``) has a row per ``set`` (``outlines``, the headline, and
the bare ``nuclei``) and per ``pooling``: ``case`` (the median over cases), ``patient``
(the median of the per-patient medians, every case of the row) and ``common`` (the
patient pooling on the cases every compared row of the dataset has, ``common_sets``),
which is the HEADLINE (``is_headline``; a reader that wants one row per family takes it).
``n_units`` counts what the pooling is over. Dice is given both ways: ``dice_zero_med``
(no pair counts as 0, the headline ``dice_med``) and ``dice_skip_med`` (finite values
only), with ``n_dice_zeroed`` beside. ``disp_um_common_*`` is the displacement over the
cases EVERY compared row has a readable displacement for (WD11), beside ``disp_um_*``
over the row's own readable cases. ``method_best`` has a row per family per set (chosen
on that set); every other table and figure reads the outlines' row (``headline_best``).
Paired differences against STARE are per patient (``paired_unit``), a tie counted as
STARE at least as good (b >= a). ``n_candidates`` and ``in_sample_selection`` are on every
best row: the best is an in-sample CEILING over ``n_candidates`` rows, not an estimate of
a new cohort. DeeperHistReg's primary is its best row (``BEST_IS_PRIMARY``: user ruling
2026-10-18), so its primary is in-sample too; ASHLAR's primary is unchanged.

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
    "pair_fraction_max",
    "pair_fraction_max_null_max",
    "pair_fraction_sym",
    "pair_fraction_sym_null_max",
    "displacement_um_p50",
    "displacement_um_p90",
    "within_1um",
    "n_pairs",
    "n_ref",
    "n_moving",
)
# Task 21b: beside the headline, never ranked (score.FULL_PAIRINGS): the per_pair column
# and the cells column it is read from
FULL_PAIR_COLUMNS = {
    "dice_matched_full_lsa": "dice_matched_full_lsa",
    "dice_matched_full_mnn": "dice_matched_mnn",
    "pair_fraction_full_lsa": "pair_fraction_full_lsa",
    "pair_fraction_full_mnn": "pair_fraction_mnn",
    "fallback_frac": "fallback_frac",
}
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
                    pair_fraction_max=num["pair_fraction_max"],
                    pair_fraction_max_null_max=num["pair_fraction_max_null_max"],
                    pair_fraction_sym=num["pair_fraction_sym"],
                    pair_fraction_sym_null_max=num["pair_fraction_sym_null_max"],
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
                    **{k: _num(r.get(c)) for k, c in FULL_PAIR_COLUMNS.items()},
                    full_lsa_status=r.get("full_lsa_status"),
                )
            )
    return rows


def full_stats(mine, pooling=None):
    """The Task 21b columns of a summary row over the rows ``mine`` (schema ``_FULL``):
    the full pairings' Dice and the fallback share, each the median over patients of
    the patient medians of the FINITE values (a missing one is left out, never 0: these
    numbers rank nothing), with ``fallback_frac_max`` and the cases with a full LSA.
    ``pooling`` (T29) ``case`` takes the median over the cases instead; the others (and
    None) pool on patients, ``mine`` being already the pooled rows."""

    def per_patient(key):
        if pooling == CASE_POOL:
            return _median(r.get(key) for r in mine)
        by = {}
        for r in mine:
            v = _num(r.get(key))
            if math.isfinite(v):
                by.setdefault(r["patient_id"], []).append(v)
        return _median(_median(v) for v in by.values())

    fallback = _finite(r.get("fallback_frac") for r in mine)
    return {
        "dice_full_lsa_med": per_patient("dice_matched_full_lsa"),
        "dice_full_mnn_med": per_patient("dice_matched_full_mnn"),
        "fallback_frac_med": per_patient("fallback_frac"),
        "fallback_frac_max": max(fallback) if fallback else math.nan,
        "n_full_lsa": len(_finite(r.get("pair_fraction_full_lsa") for r in mine)),
    }


def _rows_of(pp, row, dataset, kept=True, set_=OUTLINES):
    """The rows of one result name on a dataset and polygon set (the outlines unless
    told); ``kept``: without the excluded."""
    return [
        r
        for r in pp
        if r["row"] == row
        and r["dataset"] == dataset
        and r["set"] == set_
        and not (kept and r["excluded_from_summaries"] is True)
    ]


def _dice0(r):
    """A kept row's Dice, with a missing one (no matched pair) counted as 0.0."""
    d = _num(r["dice_matched"])
    return d if math.isfinite(d) else 0.0


def _no_pair(r):
    return not math.isfinite(_num(r["dice_matched"]))


def _dice_skip(r):
    """A kept row's Dice with a missing one left out (NaN: no statistic reads it)."""
    return _num(r["dice_matched"])


# ── T29: sets and poolings (the fair comparison, accuracy side) ─────────────
SETS = (OUTLINES, NUCLEI)
CASE_POOL, PATIENT_POOL, COMMON_POOL = "case", "patient", "common"
POOLINGS = (COMMON_POOL, PATIENT_POOL, CASE_POOL)  # the headline first
HEADLINE_POOLING = COMMON_POOL


def is_headline(r):
    """A row of the headline slot: the outlines, pooled on the common cases. A row of a
    table written before T29 (no ``set`` / ``pooling``) is the headline too."""
    def blank(v):
        return v is None or v == "" or v == schema.NA

    return (blank(r.get("set")) or r.get("set") == OUTLINES) and (
        blank(r.get("pooling")) or r.get("pooling") == HEADLINE_POOLING)


def headline_best(best_rows):
    """The method_best rows the other tables and figures read: the outlines'."""
    return [r for r in best_rows if is_headline(r)]


def compared_rows(best_rows, primary):
    """The rows the cross-method common cases are among: the headline best rows and
    the primary rows (the cost tables' set, ``costs.compared_rows``), no cost-only row."""
    rows = [r.get("best_row") for r in headline_best(best_rows)] + list(primary)
    return [r for r in dict.fromkeys(rows) if not _missing(r) and not is_cost_only(r)]


def common_sets(pp, dataset, compared):
    """``{set: {"cases", "readable"}}``: the cases every compared row has a kept row on,
    and those of them where every compared row's displacement is readable (WD11). A
    compared row with no row at all on a set does not empty it (it is not there)."""
    out = {}
    for set_ in SETS:
        cases = readable = None
        for row in compared:
            if not _rows_of(pp, row, dataset, kept=False, set_=set_):
                continue
            mine = _rows_of(pp, row, dataset, set_=set_)
            have = {r["case_id"] for r in mine}
            ok = {r["case_id"] for r in mine if r["displacement_readable"] is True}
            cases = have if cases is None else cases & have
            readable = ok if readable is None else readable & ok
        out[set_] = {"cases": cases or set(), "readable": readable or set()}
    return out


def _pooled(rows, pooling, common):
    """The rows a pooling reads: on ``common`` only the common cases."""
    if pooling == COMMON_POOL:
        cases = (common or {}).get("cases", set())
        return [r for r in rows if r["case_id"] in cases]
    return rows


def _units(rows, pooling, value):
    """One value per unit: per case (``case``), or per patient the median of its cases'
    finite values (``patient``, ``common``)."""
    if pooling == CASE_POOL:
        return [value(r) for r in rows]
    by = {}
    for r in rows:
        by.setdefault(r["patient_id"], []).append(value(r))
    return [_median(v) for v in by.values()]

def _patient_median(rows):
    by = {}
    for r in rows:
        by.setdefault(r["patient_id"], []).append(_dice0(r))
    return {p: _median(v) for p, v in by.items()}


# ── the best row of a method ─────────────────────────────────────────────────
STRICT, PER_METHOD = "strict", "per_method"  # score.COMMON_BLOCKS (B2)


def best_of(per_pair, family, dataset, cost=None, rule=STRICT, candidates=None,
            set_=OUTLINES):
    """The ``method_best`` row of one family (the rule: module docstring, ``RULE``).
    ``cost`` is kept for the callers and decides nothing: the tie-break is the name
    (T31 R11: device-blind). ``candidates(name)`` narrows the candidates (the CPU
    cross takes DeeperHistReg's CPU rows only); a ``variants.COST_ONLY`` row is never
    one. ``set_`` (T29): the polygon set it is chosen on; the outlines' is the headline
    every other table reads (``headline_best``), the nuclei's is beside."""
    cost_only = set(getattr(variants, "COST_ONLY", ()))
    cands = [
        t
        for t in dict.fromkeys(terminal_rows(dataset).values())
        if family_of(t) == family
        and t not in cost_only
        and (candidates is None or candidates(t))
    ]
    have = {c: _rows_of(per_pair, c, dataset, set_=set_) for c in cands}
    cands = [c for c in cands if _rows_of(per_pair, c, dataset, kept=False, set_=set_)]
    # B2: STRICT (the headline) is the cases every candidate has a kept row on; on
    # PER_METHOD a round lost for a cause that is not its own (excluded) takes the
    # case out of that candidate's own set only
    every = {c: {r["case_id"] for r in _rows_of(per_pair, c, dataset, kept=False, set_=set_)}
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
    zeroed, skipped = {}, {}
    for c in cands:
        mine = [r for r in have[c] if r["case_id"] in own[c]]
        zeroed[c] = sum(_no_pair(r) for r in mine)
        skipped[c] = _median(_units(mine, PATIENT_POOL, _dice_skip))
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
        criterion_value_skip=skipped[best[2]] if best else math.nan,
        set=set_,
        unit=PATIENT_POOL,
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


# Ruling 193 (user 2026-10-18 "the best one as_drawn"): a family whose primary is its
# dataset's method_best row (every DeeperHistReg row is as-drawn since 2026-10-17).
# variants.PRIMARY keeps a placeholder (PRIMARY_DEEPERHISTREG) for the scorer's common
# block and the realcheck; the paper resolves the primary here, at table time.
BEST_IS_PRIMARY = {
    "deeperhistreg": "primary = best (user 2026-10-18): the dataset's method_best row, "
    "every DeeperHistReg row as-drawn; chosen on this dataset with this metric (in-sample)",
}


def primary_row(family, dataset, best_rows):
    """The family's primary on ``dataset`` as the tables use it: its headline
    method_best row for a family of ``BEST_IS_PRIMARY``, else ``primary_of`` (also the
    fallback when the family has no best row)."""
    if family in BEST_IS_PRIMARY:
        for r in headline_best(best_rows):
            if r.get("family") == family and not _missing(r.get("best_row")):
                return r["best_row"]
    return primary_of(family, dataset)


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
def _stats(values, prefix):
    q25, q75 = _quartiles(values)
    return {f"{prefix}_med": _median(values), f"{prefix}_q25": q25, f"{prefix}_q75": q75}


def _accuracy(pp, row, dataset, set_=OUTLINES, pooling=CASE_POOL, common=None):
    """``(rows, columns)`` of one row's accuracy on one set and pooling (T29): Dice
    zero-filled (the headline) and skipped, the three pair fractions, the displacement
    on the row's readable cases and on the common readable ones, the full pairings."""
    common = common or {"cases": set(), "readable": set()}
    mine = _pooled(_rows_of(pp, row, dataset, set_=set_), pooling, common)
    dice = _units(mine, pooling, _dice0)
    skip = _units(mine, pooling, _dice_skip)
    readable_rows = [r for r in mine if r["displacement_readable"] is True]
    shared_rows = [r for r in readable_rows if r["case_id"] in common["readable"]]

    def disp(rows):
        return _units(rows, pooling, lambda r: r["displacement_um_p50"])

    readable, shared = disp(readable_rows), disp(shared_rows)
    return mine, {
        "n_units": len(dice),
        "n_dice_zeroed": sum(_no_pair(r) for r in mine),
        "tool_version": _distinct(r["tool_version"] for r in mine),
        "n_patients": len({r["patient_id"] for r in mine}),
        "n_pairs": len(mine),
        "n_ok": sum(r["imputed_initial"] is not True for r in mine),
        **_stats(dice, "dice"),
        "dice_mean": _mean(dice),
        "dice_zero_med": _median(dice),
        **_stats(skip, "dice_skip"),
        "dice_skip_mean": _mean(skip),
        **{f"{k}_med": _median(_units(mine, pooling, lambda r, k=k: r.get(k)))
           for k in ("pair_fraction", "pair_fraction_max", "pair_fraction_sym")},
        **_stats(readable, "disp_um"),
        "n_readable": len(readable_rows),
        **_stats(shared, "disp_um_common"),
        "n_readable_common": len(shared_rows),
        **full_stats(mine, pooling),
    }


def _paired(pp, row, stare_row, dataset, set_=OUTLINES, pooling=CASE_POOL, common=None):
    """``row - stare_row`` on the cases both have and neither is excluded, per unit
    (T29: a case, or per patient the median over its cases both have, minus STARE's;
    ``paired_unit``). A missing Dice counts as 0, except on a case where both are
    missing (the case's); ``d_dice_skip_vs_stare_med`` leaves it out instead. A tie
    counts for STARE (``n_units_stare_ge``: b >= a)."""
    if stare_row is None:
        return {}
    mine = _pooled(_rows_of(pp, row, dataset, set_=set_), pooling, common)
    theirs = _pooled(_rows_of(pp, stare_row, dataset, set_=set_), pooling, common)
    a = {r["case_id"]: r for r in mine}
    b = {r["case_id"]: r for r in theirs}
    both = [c for c in a if c in b and not (_no_pair(a[c]) and _no_pair(b[c]))]

    def per_unit(value):
        if pooling == CASE_POOL:
            return {c: (value(a[c]), value(b[c])) for c in both}
        by = {}
        for c in both:
            by.setdefault(a[c]["patient_id"], []).append(c)
        return {p: (_median(value(a[c]) for c in cs), _median(value(b[c]) for c in cs))
                for p, cs in by.items()}

    zero, skip = per_unit(_dice0), per_unit(_dice_skip)
    diffs = [x - y for x, y in zero.values()]
    q25, q75 = _quartiles(diffs)
    return {
        "d_dice_vs_stare_med": _median(diffs),
        "d_dice_vs_stare_q25": q25,
        "d_dice_vs_stare_q75": q75,
        "d_dice_skip_vs_stare_med": _median(x - y for x, y in skip.values()),
        "paired_unit": CASE_POOL if pooling == CASE_POOL else PATIENT_POOL,
        "n_pairs_vs_stare": len(both),
        "n_pairs_stare_ge": sum(_dice0(b[c]) >= _dice0(a[c]) for c in both),
        "n_units_vs_stare": len(zero),
        "n_units_stare_ge": sum(y >= x for x, y in zero.values()),
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


def _prior_by_patient(row, cost_rows, res_rows, common, peaks):
    """``{patient: (prior_cpu_h, prior_s, n_rounds_missing)}`` over ``common`` of an
    ``affine_ashlar`` row, what the row it reads its anchor from cost up to that stage
    (R15); ``{}`` for a row with no prior. ``affine=stare``: STARE's
    COARSE (``stare_stage_peaks.coarse_cpu_h`` / ``coarse_s``) summed over the patient's
    rounds, and the count of its rounds (``resources.csv`` rows of ``stare``) with no
    stage-peaks line: a patient with one is a LOWER bound (T31 fix M8). ``affine=valis``:
    the ``valis_mirage_rigid`` row's ``cpu_h_used`` and ``register_s`` (complete patients)."""
    try:
        run = variants.run_of(row)
    except KeyError:
        return {}
    anchor = (run.opts or {}).get("affine")
    if run.method != "ashlar" or anchor in (None, "", "none"):
        return {}
    out = {}
    if anchor == "stare":
        rounds = {}
        for r in res_rows:
            if r.get("method") == "stare":
                rounds.setdefault(str(r.get("group")), set()).add(r.get("case_id"))
        for r in peaks:
            p = str(r.get("patient_id"))
            if r.get("row") != "stare" or p not in common:
                continue
            c, t, seen = out.get(p, (0.0, 0.0, set()))
            out[p] = (c + _num(r.get("coarse_cpu_h")), t + _num(r.get("coarse_s")),
                      seen | {r.get("case_id")})
        return {p: (c, t, len(rounds.get(p, set()) - seen)) for p, (c, t, seen) in out.items()}
    rigid = f"{VALIS_FAMILY}_rigid"
    for r in _complete(cost_rows):
        if r["method"] == rigid and r["group"] in common:
            out[r["group"]] = (_num(r.get("cpu_h_used")), _num(r.get("register_s")), 0)
    return out


def _with_prior(own, prior):
    """The median over patients of the row's own CPU-hours plus its prior's (the sum per
    patient, never a median plus a median; T31 fix M8): ``own`` is ``{patient: cpu_h}``."""
    return _median(own[p] + prior[p][0] for p in own if p in prior)


def method_summary(pp, best_rows, cost_rows, res_rows, dataset, points=None, common=None, peaks=()):
    """The rows of each family (``_summary_kinds``): ``best`` (the ``method_best`` row),
    ``primary`` (``primary_of``), STARE's 4-CPU row and a GPU row's CPU twin. Accuracy as
    before; cost by ``paper.costs`` (module docstring), over the common patients first."""
    heads = headline_best(best_rows)
    best = {r["family"]: r["best_row"] for r in heads}
    n_cands = {r["family"]: r.get("n_candidates") for r in heads}
    primary = {fam: primary_row(fam, dataset, best_rows) for fam in _families(dataset)}
    stare_best = best.get(STARE_FAMILY)
    stare_best = None if _missing(stare_best) else stare_best
    excluded = _excluded_rounds(pp)
    if points is None or common is None:
        common, points = cost_points(pp, best_rows, cost_rows, res_rows, dataset)
    sets = common_sets(pp, dataset, compared_rows(best_rows, primary.values()))
    local4 = next(iter(getattr(variants, "LOCAL_4CPU", ())), None)
    out = []
    for fam in _families(dataset):
        for which, row in _summary_kinds(fam, best, primary, cost_rows, dataset):
            # a cost-only row has no accuracy here: it is no configuration of its own
            # (T31 fix C1); its cost columns are its own, once, in the headline slot
            cost_row = is_cost_only(row)
            old = _cost(cost_rows, res_rows, row, excluded)
            mine = [r for r in cost_rows if r["method"] == row]
            clock = costs.clock_of(row, mine[0]) if mine else costs.WHOLE
            sc = costs.summary_columns(points, row, common, clock)
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
            prior = _prior_by_patient(row, cost_rows, res_rows, common, peaks)
            prior_cpu = _median(v[0] for v in prior.values())
            own = {p["unit_id"]: _num(p.get("cpu_h_used")) for p in shared}
            cost_cols = dict(
                **old,
                **sc,
                total_cpu_h_used_med=sc["total_cpu_h_used_common_med"],
                cpu_h_reserved_med=sc["cpu_h_reserved_common_med"],
                total_cpu_h_reserved_med=sc["total_cpu_h_reserved_common_med"],
                rung_gb_med=sc["rung_gb_common_med"],
                wasted_attempts=int(sum(_num(p.get("wasted_attempts")) for p in shared
                                        if math.isfinite(_num(p.get("wasted_attempts"))))),
                **fold_vs_local4(points, row, local4, common),
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
                with_prior_cpu_h=_with_prior(own, prior) if prior else math.nan,
                n_prior_rounds_missing=sum(v[2] for v in prior.values()) if prior
                else schema.NA,
            )
            slots = [(OUTLINES, COMMON_POOL)] if cost_row else [
                (set_, pooling) for set_ in SETS for pooling in POOLINGS]
            for set_, pooling in slots:
                acc = {} if cost_row else {
                    **_accuracy(pp, row, dataset, set_, pooling, sets[set_])[1],
                    **_paired(pp, row, stare_best, dataset, set_, pooling, sets[set_]),
                }
                out.append(
                    schema.empty_row(
                        "method_summary",
                        source=SOURCE,
                        dataset=dataset,
                        family=fam,
                        row=row,
                        which=which,
                        row_kind=which,
                        set=set_,
                        pooling=pooling,
                        **_selection(which, fam, n_cands),
                        **acc,
                        **cost_cols,
                    )
                )
    return out


def _selection(which, family, n_cands):
    """``selection``, ``n_candidates`` and ``in_sample_selection`` of a method_summary
    row (T29, WD1: best stays Dice and is an in-sample ceiling, with how many rows
    competed; the headline rows are the primaries)."""
    if which == costs.BEST:
        n = n_cands.get(family)
        return {"selection": f"best of {n} by Dice on this dataset: an in-sample ceiling",
                "n_candidates": n, "in_sample_selection": True}
    if which == costs.PRIMARY and family in BEST_IS_PRIMARY:
        return {"selection": BEST_IS_PRIMARY[family], "n_candidates": n_cands.get(family),
                "in_sample_selection": True}
    if which == costs.PRIMARY:
        return {"selection": "primary: fixed before scoring (variants.PRIMARY)",
                "in_sample_selection": False}
    return {"selection": f"{which}: a cost row of its family", "in_sample_selection": False}


FOLD_STAGED = ("STARE's stage jobs: a stage job's high-water mark is not the sampler's "
               "process-tree peak (no fold)")
FOLD_GPU = "GPU row: its host RSS only (the GPU's memory is gpu_peak_gb)"


def fold_vs_local4(points, row, local4, common):
    """``fold_peak_vs_stare_local4`` of one row (T31 fix I3): the median of ``row``'s
    peaks over the median of ``stare_local4``'s, over the SAME patients: common ones
    both rows completed with a comparable peak (``n_patients_fold``). NaN for a row in
    several jobs (its peak is a stage job's, another instrument); a GPU row's is its
    host RSS and says so (``fold_note``)."""
    def peaks(name):
        return {
            p["unit_id"]: _num(p.get("peak_rss_gb"))
            for p in points
            if p.get("row") == name and p.get("unit_kind") == costs.PATIENT
            and p.get("status") == costs.COMPLETE and p["unit_id"] in common
            and p.get("peak_rss_comparable") is not False
            and math.isfinite(_num(p.get("peak_rss_gb")))
        }

    mine_pts = [p for p in points if p.get("row") == row and p.get("unit_kind") == costs.PATIENT]
    if any(p.get("clock") == costs.CRITICAL for p in mine_pts):
        return {"fold_peak_vs_stare_local4": math.nan, "n_patients_fold": 0,
                "fold_note": FOLD_STAGED}
    note = FOLD_GPU if any("gpu" in str(p.get("device") or "") for p in mine_pts) else ""
    a, b = peaks(row), peaks(local4) if local4 else {}
    both = sorted(set(a) & set(b))
    num, den = _median(a[p] for p in both), _median(b[p] for p in both)
    fold = num / den if both and math.isfinite(den) and den > 0 else math.nan
    return {"fold_peak_vs_stare_local4": fold, "n_patients_fold": len(both), "fold_note": note}


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
    best_rows = headline_best(best_rows)
    best = [r["best_row"] for r in best_rows if not _missing(r.get("best_row"))]
    primary = [primary_row(f, dataset, best_rows) for f in _families(dataset)]
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
        rounds, _ = costs.round_points(row, fam, res_rows, excluded, kinds, gb, common)
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


def end_to_end(summary, cost_rows, res_rows, e2e_rows, peaks, dataset, common, points=None):
    """``end_to_end`` (R15, P11): per ``method_summary`` row, the medians over the
    common patients of the method's own conversion, its registration, its writer and,
    for an anchored ASHLAR row, the prior it reads, from the scorer's
    ``cost_end_to_end.csv``; each row says whether the conversion is inside the
    registration's clock (``convert_inside_clock``). A common patient the row did not
    complete (``points``: its cost_points) is no part of its medians (T31 fix I5), and
    every median carries its bootstrap interval."""
    out = []
    for m in summary:
        row = m["row"]
        done = common if points is None else {
            p["unit_id"] for p in points
            if p["row"] == row and p["unit_kind"] == costs.PATIENT
            and p["status"] == costs.COMPLETE and p["unit_id"] in common
        }
        mine = [
            r for r in e2e_rows if r.get("method") == row and r.get("group") in done
        ]
        stats = {}
        for col, src in (("convert_s", "method_convert_s"), ("estimate_s", "estimate_s"),
                         ("write_s", "write_s")):
            vals = [r.get(src) for r in mine]
            stats[f"{col}_med"] = _median(vals)
            stats[f"{col}_ci_low"], stats[f"{col}_ci_high"] = costs.interval(vals)
        prior_s = [v[1] for v in _prior_by_patient(row, cost_rows, res_rows, done, peaks).values()]
        stats["prior_s_med"] = _median(prior_s)
        stats["prior_s_ci_low"], stats["prior_s_ci_high"] = costs.interval(prior_s)
        out.append(
            schema.empty_row(
                "end_to_end",
                source=SOURCE,
                dataset=dataset,
                family=m["family"],
                row=row,
                row_kind=m["row_kind"],
                clock=m["clock"],
                unit=costs.PATIENT,
                n_patients=len({r.get("group") for r in mine}),
                prior_cpu_h=m.get("prior_cpu_h"),
                **stats,
                convert_inside_clock=m.get("convert_inside_clock"),
            )
        )
    return out


# ── per-method tables ────────────────────────────────────────────────────────
def _stage_stats(pp, row, dataset, set_=OUTLINES, pooling=CASE_POOL, common=None):
    if _missing(row):
        return {}
    mine = _pooled(_rows_of(pp, row, dataset, set_=set_), pooling, common)
    dice = _units(mine, pooling, _dice0)
    q25, q75 = _quartiles(dice)
    readable = _units([r for r in mine if r["displacement_readable"] is True], pooling,
                      lambda r: r["displacement_um_p50"])
    return {
        "n_units": len(dice),
        "dice_med": _median(dice),
        "dice_q25": q25,
        "dice_q75": q75,
        "dice_zero_med": _median(dice),
        "dice_skip_med": _median(_units(mine, pooling, _dice_skip)),
        "n_dice_zeroed": sum(_no_pair(r) for r in mine),
        "n_pairs": len(mine),
        "disp_um_med": _median(readable),
        **full_stats(mine, pooling),
    }


def accuracy_sets(pp, dataset, best_rows=None):
    """``common_sets`` among the dataset's compared rows: its headline best rows (made
    here by ``best_of`` when not given) and its primaries (``primary_row``)."""
    if best_rows is None:
        best_rows = build_best(pp, dataset, {})["method_best"]
    primary = [primary_row(f, dataset, best_rows) for f in _families(dataset)]
    return common_sets(pp, dataset, compared_rows(best_rows, [p for p in primary if p]))


def _slots(sets):
    """``[(set, pooling, that set's common cases)]``, the headline first (T29)."""
    return [(s, p, sets.get(s)) for s in SETS for p in POOLINGS]


def valis_stages(pp, dataset, sets=None):
    """Five stage slots per ``valis_mirage`` run (R18, module docstring), per polygon set
    and pooling (T29; ``sets``: ``common_sets``, empty for a table derived alone)."""
    sets = accuracy_sets(pp, dataset) if sets is None else sets
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
        for set_, pooling, common in _slots(sets):
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
                        set=set_,
                        pooling=pooling,
                        **_stage_stats(pp, row, dataset, set_, pooling, common),
                    )
                )
    return out


def stare_tiers(pp, cost_rows, dataset, points=(), common=(), sets=None):
    """One row per STARE run: tier, stride and preprocessing from its options (defaults
    ``high``, 64, ``none``), accuracy and cost. The cost is ``costs.summary_columns``'
    (the cost rules: the common patients, an interval, the clock and the unit; T31 fix
    I5), from the row's patient points. Per polygon set and pooling (T29)."""
    sets = accuracy_sets(pp, dataset) if sets is None else sets
    out = []
    for name, final in terminal_rows(dataset).items():
        if family_of(final) != STARE_FAMILY:
            continue
        run = _run(name)
        opts = run.opts if run is not None else {}
        mine = [r for r in cost_rows if r["method"] == final]
        clock = costs.clock_of(final, mine[0]) if mine else costs.WHOLE
        sc = costs.summary_columns(points, final, set(common), clock)
        cost = {k: sc[k] for k in (
            "n_patients_common", "n_patients_survivors", "n_peak_not_comparable", "clock",
            "unit", "cpu_h_used_common_med", "cpu_h_used_common_ci_low",
            "cpu_h_used_common_ci_high", "peak_rss_gb_common_med", "peak_rss_gb_common_ci_low",
            "peak_rss_gb_common_ci_high")}
        keep = ("n_units", "dice_med", "dice_q25", "dice_q75", "dice_zero_med",
                "dice_skip_med", "n_dice_zeroed", "n_pairs", *full_stats([]))
        for set_, pooling, sets_common in _slots(sets):
            stats = _stage_stats(pp, final, dataset, set_, pooling, sets_common)
            out.append(
                schema.empty_row(
                    "stare_tiers",
                    source=SOURCE,
                    dataset=dataset,
                    row=final,
                    tier=opts.get("tier", "high"),
                    stride=int(opts.get("stride", 64)),
                    sprep=opts.get("sprep", "none"),
                    set=set_,
                    pooling=pooling,
                    **{k: stats.get(k, math.nan) for k in keep},
                    **cost,
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


NOT_RECORDED = "not recorded"
LAUNCHER = Path(__file__).resolve().parents[1] / "slurm" / "launch.sbatch"


def launcher_images():
    """``{image key: reference}`` of the launcher's ``IMAGES`` block (the image each
    ``REGBENCH_SIF_<key>`` is pulled from)."""
    text = LAUNCHER.read_text()
    start = text.index('IMAGES="') + len('IMAGES="')
    block = text[start:text.index('"', start)]
    return dict(line.split(None, 1) for line in block.strip().splitlines() if line.strip())


def _recorded(recs, *paths):
    """The distinct values a record gives at any of ``paths`` (dotted), '|'-joined;
    ``NOT_RECORDED`` when none does."""
    seen = []
    for rec in recs:
        for path in paths:
            v = rec
            for part in path.split("."):
                v = v.get(part) if isinstance(v, dict) else None
            if not _missing(v) and str(v) not in seen:
                seen.append(str(v))
    return "|".join(seen) if seen else NOT_RECORDED


def versions(out, dataset):
    """Per run: the tool and the distinct ``info.version`` values over its records, the
    image tag the launcher pins, and the digest, Python and StarDist version as far as a
    record gives them (T29, M7: ``not recorded`` where none does)."""
    from .. import nuclei

    images = launcher_images()
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
                image_tag=images.get(run.image, NOT_RECORDED) if run is not None
                else NOT_RECORDED,
                image_digest=_recorded(recs, "image_digest", "info.image_digest",
                                       "host.image_digest"),
                python=_recorded(recs, "info.python", "info.libraries.python",
                                 "host.python", "python"),
                stardist_image=nuclei.PIPELINE_IMAGE,
                stardist_version=_recorded(recs, "info.stardist", "stardist"),
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


VALIS_PRESETS = Path(__file__).resolve().parents[3] / "bin" / "utils" / "valis_config.py"
_VALIS_SIZES = ("max_processed_image_dim_px", "max_non_rigid_registration_dim_px")


def _valis_presets():
    """``{tier: {size key: px}}`` of ``bin/utils/valis_config.py`` ``MEMORY_PRESETS``,
    READ from its source (``ast``): importing it builds VALIS's matchers (torch)."""
    import ast

    tree = ast.parse(VALIS_PRESETS.read_text())
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "MEMORY_PRESETS"):
            out = {}
            for k, v in zip(node.value.keys, node.value.values):
                row = {}
                for kk, vv in zip(v.keys, v.values):
                    key = getattr(kk, "value", None)
                    if key in _VALIS_SIZES and isinstance(vv, ast.Constant):
                        row[key] = vv.value
                out[k.value] = row
            return out
    raise KeyError(f"{VALIS_PRESETS}: no MEMORY_PRESETS")


def module_parameters(run):
    """``[(key, value, from)]``: a run's settings as its method module resolves them
    (T29, M6), from the module's own constants and resolvers, imported read-only (the
    method files are frozen; nothing here changes what they do). ``[]`` for a method with
    no settings (``initial``) or a toy row."""
    if run is None:
        return []
    from ..methods import ashlar, deeperhistreg, palom, stare, valis, valis_mirage

    opts = {**run.opts, **({"label": run.label} if run.label else {})}
    m = run.method
    if m == "stare":
        src = "benchmarks.regbench.methods.stare (TIERS, tier_of, stride_of; sprep)"
        tier = stare.tier_of(opts)
        values = {**stare.TIERS[tier], "stride": stare.stride_of(opts)}
        values.update({k: opts[k] for k in stare._FLAGS if k in opts and k != "stride"})
        out = [("tier", tier, src)] + [(k, v, src) for k, v in values.items()]
        out.append(("sprep", str(opts.get("sprep", "none")), src))
        if "proxy" in opts:
            out.append(("proxy", opts["proxy"], src))
        return out
    if m == "valis_mirage":
        src = "benchmarks.regbench.methods.valis_mirage (ARGS, call_args)"
        args = valis_mirage.call_args(opts)
        out = [(k, v, src) for k, v in args.items() if v is not None]
        sizes = _valis_presets().get(args["memory_mode"], {})
        psrc = f"bin/utils/valis_config.py MEMORY_PRESETS[{args['memory_mode']!r}] (its source)"
        return out + [(k, sizes[k], psrc) for k in _VALIS_SIZES if k in sizes]
    if m == "valis":
        src = "benchmarks.regbench.methods.valis (MICRO_REG_FRACTION, CHANNEL; hires)"
        return [("hires", str(opts.get("hires", 0)) in ("1", "true", "True"), src),
                ("micro_reg_fraction", valis.MICRO_REG_FRACTION, src),
                ("channel", valis.CHANNEL, src)]
    if m == "palom":
        src = "benchmarks.regbench.methods.palom (THUMBNAIL_SIZE, N_KEYPOINTS, LEVEL)"
        return [("thumbnail_size", palom.THUMBNAIL_SIZE, src),
                ("n_keypoints", palom.N_KEYPOINTS, src), ("level", palom.LEVEL, src),
                ("preread", str(opts.get("preread", "auto")), src)]
    if m == "ashlar":
        src = "benchmarks.regbench.methods.ashlar (settings, TILE, OVERLAP, SEED)"
        frac, affine, shift = ashlar.settings(opts)
        return [("max_shift_um", shift, src), ("noise_frac", frac, src),
                ("affine", affine, src), ("tile", ashlar.TILE, src),
                ("overlap", ashlar.OVERLAP, src), ("seed", ashlar.SEED, src)]
    if m == "deeperhistreg":
        src = "benchmarks.regbench.methods.deeperhistreg (settings, CONFIGS)"
        config, device, fluo = deeperhistreg.settings(opts)
        return [("config", config, src), ("paper_resolution_px", deeperhistreg.CONFIGS[config], src),
                ("device", device, src), ("fluo", fluo, src)]
    return []


def run_parameters(out, dataset):
    """Per run: its settings as its method module resolves them (``module_parameters``,
    T29), then every key of ``info.implementation.parameters`` when the run recorded a
    dict there (Table S6), else the run's options, that the module rows do not name."""
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
        mod = module_parameters(run)
        named = {k for k, _, _ in mod}
        for k, v, msrc in mod:
            rows.append(
                schema.empty_row(
                    "run_parameters",
                    source=SOURCE,
                    row=final,
                    method=method,
                    key=k,
                    value=_value(v),
                    n_unreadable=len(bad),
                    **{"from": msrc},
                )
            )
        if mod and items and list(items)[0][0] is None:
            items = []  # the module rows ARE the defaults
        items = [(k, v) for k, v in items if k not in named and k != "label"]
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
    for set_, row in ((s, r) for s in SETS for r in rows):
        mine = _rows_of(pp, row, dataset, set_=set_)  # an excluded round is in no count
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
                    set=set_,
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
                clock=costs.WHOLE,
                unit="case",
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
        "method_best": [best_of(pp, f, dataset, cpu, set_=s) for s in SETS for f in fams],
        "method_best_per_method": [
            best_of(pp, f, dataset, cpu, PER_METHOD, set_=s) for s in SETS for f in fams
        ],
    }


def excluded_runs(out, dataset):
    """``[(run, reason)]`` of the dataset's ``excluded_runs.csv`` (the scorer's record;
    ``[]`` when it is absent or a header alone)."""
    return [(r["run"], r.get("reason") or schema.NA)
            for r in _csv(_tables(out, dataset) / "excluded_runs.csv") if r.get("run")]


def _excluded_final(name):
    """``(final result name, family, method)`` of a run left out (its own name when it
    names no run)."""
    run = _run(name)
    final = list(variants.variants_of(run))[-1] if run is not None else name
    return final, family_of(final) or family_of(name), run.method if run is not None else None


def add_excluded(built, excluded, dataset, summary_keys=None):
    """T29 (M8), in place: each run of ``excluded`` (``excluded_runs``) as a row of NA
    with its reason in ``method_summary`` (``which = excluded``) and ``versions``, and in
    ``method_best`` the family's ``excluded_runs`` (a family with no run left: a row of NA).
    ``summary_keys``: ``{table: extra values}`` of a landmark dataset's summary instead."""
    if not excluded:
        return built
    by_family = {}
    for name, reason in excluded:
        final, fam, method = _excluded_final(name)
        by_family.setdefault(fam, []).append(f"{name}: {reason}")
        if "method_summary" in built:
            built["method_summary"].append(schema.empty_row(
                "method_summary", source=SOURCE, dataset=dataset, family=fam, row=final,
                which="excluded", row_kind="excluded", set=OUTLINES, pooling=COMMON_POOL,
                excluded_reason=reason))
        for table, extra in (summary_keys or {}).items():
            built[table].append(schema.empty_row(
                table, source=SOURCE, dataset=dataset, family=fam, row=final,
                excluded_reason=reason, **extra))
        if "versions" in built:
            built["versions"].append(schema.empty_row(
                "versions", source=SOURCE, row=final, method=method, image_tag=NOT_RECORDED,
                image_digest=NOT_RECORDED, python=NOT_RECORDED, stardist_version=NOT_RECORDED,
                excluded_reason=reason))
    best = built.get("method_best")
    if best is not None:
        have = {r["family"] for r in best}
        for fam, said in by_family.items():
            text = " | ".join(said)
            for r in best:
                if r["family"] == fam:
                    r["excluded_runs"] = text
            if fam not in have:
                best.append(schema.empty_row(
                    "method_best", source=SOURCE, dataset=dataset, family=fam, set=OUTLINES,
                    excluded_runs=text))
    return built


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
    sets = accuracy_sets(pp, dataset, best)
    counted = [r for r in dict.fromkeys(r["row"] for r in summary) if not is_cost_only(r)]
    e2e = _csv(_tables(out, dataset) / "cost_end_to_end.csv")
    return add_excluded({
        # every row: local4_differs reads the cost row's pairs; per_pair.csv is written
        # without it (write_all: accuracy_pairs, T31 fix C1)
        "per_pair": every,
        "end_to_end": end_to_end(summary, cost_rows, res_rows, e2e, peaks, dataset, common,
                                 points),
        "method_best": best,
        "method_best_per_method": got["method_best_per_method"],
        "method_summary": summary,
        "valis_stages": valis_stages(pp, dataset, sets),
        "stare_tiers": stare_tiers(pp, cost_rows, dataset, points, common, sets),
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
    }, excluded_runs(out, dataset), dataset)


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


def cost_losses(points, summary, shown):
    """Rule 6 for the tables (T31 fix I6): per ``method_summary`` row and common patient
    it completed, one ``cost_figure_missing`` row naming the costs it has no figure for
    (that the row has for another such patient: they are left out of those medians), one
    ``cost_peak_not_comparable`` row (a rusage peak, out of every peak median) and one
    ``cost_lower_bound`` row (kept and flagged)."""
    from .. import losses

    rows = []
    for m in {m["row"]: m for m in summary}.values():
        row = m["row"]
        mine = [p for p in points
                if p["row"] == row and p["unit_kind"] == costs.PATIENT
                and p["status"] == costs.COMPLETE and p.get("in_common") is True]
        have = [s for s in costs.STEMS
                if any(math.isfinite(costs.num(p.get(s))) for p in mine)]

        def loss(p, cause, effect, detail):
            rows.append(losses.row(
                stage="paper", dataset=shown, patient_id=p["unit_id"], run=row,
                method=m.get("family") or "", cause=cause, effect=effect, detail=detail,
                source="paper/method_summary.csv"))

        for p in sorted(mine, key=lambda p: str(p["unit_id"])):
            lost = [s for s in have if not math.isfinite(costs.num(p.get(s)))]
            if lost:
                loss(p, "cost_figure_missing", "dropped",
                     f"no {', '.join(lost)}: left out of those common-patient medians")
            if p.get("peak_rss_comparable") is False:
                loss(p, "cost_peak_not_comparable", "dropped",
                     "a rusage peak: left out of every peak median")
            if not costs.missing(p.get("lower_bound")):
                loss(p, "cost_lower_bound", "flagged",
                     f"times and CPU-hours are lower bounds ({p['lower_bound']})")
    return rows


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
    rows += cost_losses(built["cost_points"], built["method_summary"], shown)
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
    from . import landmarks

    if landmarks.is_landmark_set(dataset):
        # a public landmark set has its own tables (paper.landmarks): its scorer wrote
        # no cells, so the cohort's tables would be empty
        return landmarks.write_all(out, dataset)
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
    from .cost_plots import PLOT_CAUSES  # the figures' rows are theirs (T31 fix M7)

    mine = [c for c in losses.CAUSES if c not in PLOT_CAUSES]
    losses.merge(paths["losses"], paper_losses(built, dataset), mine)
    return paths


def main(a):
    paths = write_all(a.out, a.dataset, getattr(a, "cases", None))
    for name, path in paths.items():
        print(f"{name}\t{path}")
    return 0
