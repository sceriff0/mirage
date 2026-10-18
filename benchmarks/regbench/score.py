"""Score every variant found under ``--out`` against the cases; needs no method's software.

**Cells.** Nuclei are segmented once on each NATIVE slide. A method's point mapping has
moved one of the two sets in the method's own direction (``warped.npz``), and the two are
compared in the frame that leads to: the reference frame for a ``mov_to_ref`` map, the
moving frame for a ``ref_to_mov`` one. Every number is ``regbench.pairs.record``: the
pipeline's pairing and IoU under a gate sized once per case, a null and a two-sided curve.

This is SELF-CONSISTENCY on the registration channel, not accuracy: it says the two
segmentations overlap after warping, with no independent truth. Three things make it
readable:

``*_null``      the same score with one set shifted four nuclear radii, averaged over eight
                directions: what chance pairing gives in this tissue. It depends strongly on
                density, so every case reports ``coverage`` and ``nn_distance_um_p50``.
``fold_pct``, ``sd_log_jac``   the share of a lattice where the method's map folds and the
                spread of its log Jacobian (forward evaluation only). A map that snaps
                nuclei onto their nearest neighbour is rewarded by the paired numbers; this
                and the two-sided curve are what object.

**A failed run is a row, never a gap.** A variant that failed, was killed, or never ran is
scored at the unregistered pose (``imputed_initial``), with ``problem`` and ``cause``
filled. ``expected`` names variants that must appear even with no directory.

**Reading rules, fixed before any cohort run.** Displacement is censored at the gate and
conditional on being paired, so it is not used to rank; it is summarised over the cases
flagged ``displacement_readable`` only, with their count. Matched Dice is summarised over
all cases and, as ``dice_matched_ranked``, over the cases where pair fraction clears the
null's maximum; only the second may be set beside another row's. The summary claims no
ranking at any number of patients, and says so for every dataset.

**Summaries.** Per patient the median over its rounds, then the median, minimum and maximum
over patients; beside every median its mean over the same patients, ``n_ok / n_cases``
and the minimum over CASES, because a median hides a method that fails on some patients. The ``common`` block repeats the summary on the cases
every primary variant registered.

**Cost.** ``resources.csv`` has one row per round, ``summary_resources.csv`` one per
patient. Peak memory is never divided by rounds. ``cgroup_peak_gb`` is the whole job,
writers included, for every method; ``peak_rss_gb`` is the registration's own peak.

**Off the seams** (``nonband_*.csv``, never in the tables above). A method that draws its
image from tiles or blocks (``SEAM_OWNERS``) writes a seam mask with its primary row: the
nuclei that straddle its joins, from its geometry alone. EVERY variant of the case is then
scored a second time on the nuclei that survive that mask, as ``<variant>:non_<owner>``:
all methods on identical nuclei, with the number kept. These rows are a block of their
own: they enter no summary of the primary rows, no paired difference and no ranking, and
a number from them is never set beside a full-slide number. A case the owner's primary run
did not register has no such row for any method, and the count of those cases is printed.

**Tissue-level agreement of the written images** (``tissue*.csv``, ``regbench.tissue``;
never in the tables above). ACCREDIT's score and its pieces between the nuclear channel of
each image a method wrote and the reference's, at 4000 px, with an ``initial`` row for
scale. Secondary: a block of its own, in no summary of the primary rows, no paired
difference and no ranking. A variant without a written image has no row there and a line
in ``tissue_missing.csv``.

``cost_end_to_end.csv`` and ``cost_end_to_end_summary.csv`` are a second, secondary view
of cost (``regbench.cost``): the method's own conversion, the estimate and the write in
separate columns and their sum, all measured here; nothing is read from another run.
``pipeline_cost.csv``, ``pipeline_cost_summary.csv`` and ``pipeline_cost_by_process.csv``
are a THIRD, separate one (``regbench.compute_all``): the whole pipeline per patient
and registration backend, read from the Nextflow traces of compute-all
(``<out>/compute/<backend>/<patient>/trace/trace.txt``), a different clock that no
total adds to the ones above.
``launch.csv`` lists what every run's selfcheck and realcheck said (``verdicts``,
``regbench verdict``): reported, never a gate, so no run and no row is left out
because of them.

``dev_*`` cases are left out of every table unless ``dev`` is given.

**Records** (``<out>/cache/scores``, "one case" below). What is computed is kept: each
pair record (one ``pairs.record`` call) the moment it is done, and each case's rows.
A record is read again only when its whole key is equal (``pair_key``, ``case_key``);
``score_case`` makes one case's record without writing a table, and ``score`` computes
whatever has none. No table depends on which of them ran: every path writes the same
bytes (``test_score_records.py``).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import (
    DATASETS,
    ashlar_discard,
    billing,
    compute_all,
    cost,
    failure_log,
    pairs,
    plugins,
    stage_peaks,
    table_schema,
    tissue,
)
from .cases import (
    NUCLEI,
    OUTLINES,
    list_cases,
    load_points,
    nuclei_reason,
    polys,
    result_dir,
    set_info,
    stale_points,
    write_json,
)
from .mapping import FRAME, MOV_TO_REF, REF_TO_MOV

# "cache": what methods share between runs (ASHLAR's reference tiles), never a variant
RESERVED = ("tables", "tables_dev", "logs", "cache")
TIERS = ("headline", "secondary")
PSEUDO = ("initial",)  # not registrations run by the driver
# <variant>:non_<owner>: the variant scored without the nuclei on <owner>'s seams
NON = ":non_"
# Methods whose primary row carries a seam mask (``warped.npz``: ``band`` on the moving
# nuclei, ``band_ref`` on the reference nuclei). The owner's name is its primary variant.
SEAM_OWNERS = ("ashlar", "palom") + tuple(
    plugins.get("SEAM_OWNERS", ())  # of a private method plug-in, when one is enabled
)
# Every overlap number is computed on two polygon sets (``cases.SETS``). The OUTLINES are
# the headline and keep every table and column they had; the bare NUCLEI get the same
# tables under ``<name>_nuclei.csv`` with the same columns, and both carry ``set``.
NUCLEI_SUFFIX = "_nuclei"
# what a set's arrays are called in ``warped.npz``: the moved polygons, and the owner's
# seam mask on the moving and on the reference side
WARPED_KEYS = {
    OUTLINES: ("cells_xy", "band", "band_ref"),
    NUCLEI: ("nuclei_xy", "band_nuc", "band_ref_nuc"),
}
# the tables that exist once per polygon set
NUCLEI_TABLES = (
    "cells",
    "summary_cells",
    "common_block",
    "paired_vs_primary",
    "paired_summary",
    "nonband_cells",
    "nonband_summary",
)
# Every table the scorer writes (the file is ``<name>.csv`` under ``tables/``), in one list:
# the loop that clears them before a write iterates it, and the harness figure's test
# reads it. ``selfcheck_masks`` is written and cleared apart, with the selfcheck records.
TABLES = (
    "cells",
    "summary_cells",
    "paired_vs_primary",
    "failures",
    "resources",
    "summary_resources",
    "stare_stage_peaks",
    "ashlar_discard",
    "nonband_cells",
    "nonband_summary",
    "nonband_missing",
    "cost_end_to_end",
    "cost_end_to_end_summary",
    "pipeline_cost",
    "pipeline_cost_summary",
    "pipeline_cost_by_process",
    "launch",
    "common_block",
    "paired_summary",
    "excluded_rounds",
    "cost_summary",
    "failure_log",
    "nuclei_missing",
    "attrition",
    *(f"{name}{NUCLEI_SUFFIX}" for name in NUCLEI_TABLES),
    *tissue.TABLES,
)
# tables of earlier versions, still cleared from a folder that holds them
REMOVED_TABLES = (
    "dropped",
    "truth",
    "truth_by_family",
    "nonband_truth",
    "nonband_truth_by_family",
    "tissue_dose",
    "tissue_dose_summary",
)
NO_SEAM_MASK_ON_NUCLEI = (
    "no off-seam row on the bare nuclei for any method: the seam owner's primary run "
    "left no mask on this case's nuclei (it did not register the case, or its result "
    "was made before the case had nuclei)"
)
NO_NUCLEI_IN_RESULT = (
    "this result holds no bare nuclei: it was made before the case had them. "
    "Run it again (REGBENCH_FORCE=1) to score it on nuclei; its outline rows stand"
)
# pair_fraction_excess is already "above the null", so it has no null column of its own
HEADLINES = (
    "pair_fraction",
    "pair_fraction_excess",
    "dice_matched",
    "within_1um",
    "within_3um",
)
HAS_NULL = ("pair_fraction", "dice_matched", "within_1um", "within_3um")
# the headlines that depend on which cells are paired: the ones the exact assignment
# (``pairs.exact_pairs``) has a value of its own for. Shown beside the headline, in
# no ranking, no shared subset and no paired difference.
EXACT_HEADLINES = ("pair_fraction", "dice_matched")
PAIRED_AGAINST = "stare"
# The rows whose settings their authors chose on the cohort, with this metric: STARE's
# tier and the pipeline's VALIS settings. Said in the caption of the cohort's tables.
IN_SAMPLE = ("stare", "valis_mirage")
IN_SAMPLE_DATASET = "multiplex"
# said of every dataset, at every number of units: nothing here is a test
DESCRIPTIVE = "descriptive; no ranking is claimed"
COVERAGE_BIN_RADII = 16.0  # side of the occupancy bins, in median nuclear radii
# the method's own failures first, then what is NOT the method's (the scheduler lost the
# attempt; the benchmark's own step ran out of memory), then a result nobody ran
CAUSES = (
    "exception",
    "killed",
    "oom",
    "timeout",
    "gpu_oom",
    "writer",
    "writer_oom",
    "benchmark_oom",
    "infrastructure",
    "not_submitted",
    "benchmark",
    "identical_overlap",
    "not_run",  # last: `CAUSES[:-1]` below names the causes of a launch that ran
)
NOT_THE_METHOD = (
    "benchmark_oom",
    "infrastructure",
    "not_submitted",
    "benchmark",
    "identical_overlap",
)
# What becomes of a round lost for a cause in NOT_THE_METHOD. "exclude": it is left out
# of that method's summaries and, being unregistered, of the common block for every
# method; it is counted and named, and its row stays in cells.csv. "impute": it is
# scored at the unregistered pose like a failure of the method (the rule before
# 2026-10-11). ONE switch: every summary reads `_excluded`.
NOT_THE_METHOD_ROUNDS = "exclude"
# What one independent unit is, per dataset: `n_patients`, the "under six" line and
# every across-unit statistic count THIS, not the case's group.
UNIT_IS = {
    "multiplex": "patient",
}
# the paired bootstrap of the differences against PAIRED_AGAINST
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261011
BOOTSTRAP_LEVEL = 0.95


def tier_of(case):
    if case.dataset == "multiplex":
        return "headline"
    return "secondary"


JACOBIAN_KEYS = ("fold_pct", "sd_log_jac")


def jacobian_stats(warped):
    """``fold_pct`` and ``sd_log_jac`` of a method's map, from the lattice the driver pushed
    through it (``warped.npz``: ``grid``, ``grid_shape``, ``grid_step``). Forward evaluation
    only: the map is differentiated where the method defined it."""
    out = dict.fromkeys(JACOBIAN_KEYS, math.nan)
    if warped is None or "grid" not in warped or min(warped["grid_shape"]) < 3:
        return out
    rows, cols = (int(v) for v in warped["grid_shape"])
    step = float(warped["grid_step"])
    g = np.asarray(warped["grid"], dtype=float).reshape(rows, cols, 2)
    jac = np.gradient(g[..., 0], step, axis=1) * np.gradient(
        g[..., 1], step, axis=0
    ) - np.gradient(g[..., 0], step, axis=0) * np.gradient(g[..., 1], step, axis=1)
    jac = jac[np.isfinite(jac)]
    if len(jac):
        out["fold_pct"] = float(100.0 * np.mean(jac <= 0))
        if np.any(jac > 0):
            out["sd_log_jac"] = float(np.std(np.log(jac[jac > 0])))
    return out


def unit_of(case):
    """The independent unit a case belongs to (``UNIT_IS``): the cohort's patient."""
    return case.group


def _unit(row):
    return row.get("unit") or row["group"]


def _excluded(row):
    """True for a round lost for a cause that is not the method's, when such rounds
    are left out of the summaries (``NOT_THE_METHOD_ROUNDS``)."""
    return (
        NOT_THE_METHOD_ROUNDS == "exclude"
        and bool(row.get("imputed_initial"))
        and row.get("cause") in NOT_THE_METHOD
    )


def _is_run(method):
    """A registration the driver ran (or should have): not ``initial``, with or without
    an off-seam suffix."""
    return method.split(NON)[0] not in PSEUDO


def density(ps, pixel_size_um, geometry=None):
    """``coverage`` and ``nn_distance_um_p50`` of one native set of nuclei.

    ``coverage`` is the nuclear area over the area of the bins that hold nuclei, with bins
    ``COVERAGE_BIN_RADII`` median radii a side: an area fraction of the tissue, not of the
    slide. The null and every number read against it depend on it.
    """
    from scipy.spatial import cKDTree

    out = {"coverage": math.nan, "nn_distance_um_p50": math.nan}
    ps = pairs.as_polyset(ps)
    if ps.n_features < 2:
        return out
    area, cent = pairs.area_centroid(ps, geometry)
    ok = np.isfinite(cent).all(axis=1) & np.isfinite(area)
    area, cent = area[ok], cent[ok]
    radius = pairs.cp.median_equivalent_radius(area)
    if len(cent) < 2 or not radius > 0:
        return out
    side = COVERAGE_BIN_RADII * radius
    occupied = len(np.unique(np.floor(cent / side).astype(np.int64), axis=0))
    out["coverage"] = float(area.sum() / (occupied * side * side))
    if pixel_size_um:
        d, _ = cKDTree(cent).query(cent, k=2)
        out["nn_distance_um_p50"] = float(np.median(d[:, 1]) * float(pixel_size_um))
    return out


def _keep_features(ps, keep):
    """A ``PolySet`` with only the features where ``keep`` is true, renumbered."""
    keep = np.asarray(keep, dtype=bool)
    ring_keep = keep[ps.ring_feat]
    lens = np.diff(ps.ring_off)
    vertex_keep = np.repeat(ring_keep, lens)
    new_feat = np.cumsum(keep) - 1
    return pairs.cp.PolySet(
        xy=ps.xy[vertex_keep],
        ring_off=np.concatenate([[0], np.cumsum(lens[ring_keep])]).astype(np.int64),
        ring_feat=new_feat[ps.ring_feat[ring_keep]].astype(np.int64),
        ring_hole=ps.ring_hole[ring_keep],
        n_features=int(keep.sum()),
    )


SHRUNK_SOURCE = "shrunk_outlines"


def set_marks(case, which):
    """What a case's ``case.json`` says of one polygon set, as columns of every cell
    row computed on it: where each side's objects come from, whether either side is
    declared inexact (``set_inexact``: the objects are not the segmentation they stand
    for, object for object) and whether either side is the outlines with their rim
    taken off (``set_shrunk``: the fallback for a slide without bare nuclei). Empty
    sources and ``False`` for a case that records nothing about its sets."""
    info = {side: set_info(case, side, which) for side in ("ref", "mov")}
    return {
        "set_source_ref": str(info["ref"].get("source") or ""),
        "set_source_mov": str(info["mov"].get("source") or ""),
        "set_inexact": any(i.get("exact") is False for i in info.values()),
        "set_shrunk": any(i.get("source") == SHRUNK_SOURCE for i in info.values()),
    }


def _f(value):
    """A cell as a number: NaN for a missing one and for a blank one (a column that
    does not apply to that row, as the writer's job for a row without an image)."""
    return math.nan if value is None or value == "" else float(value)


def _blank(value):
    """A recorded number, or ``""`` where the run recorded none."""
    v = _f(value) if value not in (None, "") else math.nan
    return v if v == v else ""


def jvm_heap_lines(res_rows):
    """One line for the cost tables: the JVM heap each row of a JVM method really had,
    as its runs recorded it, beside what its adapter asked for. Nothing without one."""
    seen = {}
    for r in res_rows:
        if r.get("jvm_max_heap_gb") in ("", None):
            continue
        asked = r.get("jvm_heap_asked_gb")
        seen.setdefault(r["method"], set()).add(
            (round(float(r["jvm_max_heap_gb"]), 1), "" if asked == "" else float(asked))
        )
    if not seen:
        return []
    parts = []
    for method, values in seen.items():
        parts.append(
            f"`{method}` "
            + ", ".join(
                f"{got:g} GiB" + (f" (asked {asked:g})" if asked != "" else "")
                for got, asked in sorted(values, key=str)
            )
        )
    return [
        "JVM heap, as the JVM reported it in each run (`jvm_max_heap_gb`, with what "
        "the adapter asked for; a memory rung changes what is asked): "
        + "; ".join(parts)
        + ". Where the two differ the request did not reach the JVM, and the reported "
        "value is the one the run had."
    ]


NOT_RECORDED = "not recorded"


def panel_of(case):
    """The moving slide's channel panel as ``A_B_C``: the manifest stores the names
    ``|``-separated (a string), a prepared case holds them as a list; either is read."""
    extra = case.extra or {}
    for key in ("channels_mov", "channel_names"):
        names = extra.get(key)
        if isinstance(names, str):
            names = [n for n in names.split("|") if n]
        if isinstance(names, (list, tuple)) and names:
            return "_".join(str(n) for n in names)
    return NOT_RECORDED


def version_of(run):
    """The tool version the run recorded (``info.version``), else ``not recorded``."""
    version = ((run or {}).get("info") or {}).get("version")
    return str(version) if version else NOT_RECORDED


def resource_row(run, moving_px=None, stage=None, accounting=None):
    """One run's cost, as recorded at ITS variant's yield.

    The CPU-hours and memory-hours are ``regbench.billing``'s, one rule for every
    method. ``moving_px`` is the pixel count of the round's moving image
    (``cpu_h_used_per_gpx``). ``stage`` is ``billing.stage_attempts`` of the round's
    stage records, for a registration that ran as several scheduler jobs: without it
    such a row has its CPU-hours from what its run recorded and no ``gb_h_reserved``.
    For a failed row whose record has no summary of its stage jobs (``info.jobs``),
    ``stage`` is the only source of what they cost.
    ``accounting`` is the scheduler's (``ladder.read_accounting``), for
    ``cpu_h_scheduler``; without it that column is blank."""
    r = run.get("resources") or {}
    info = run.get("info") or {}
    ok = bool(run.get("ok"))
    wall, cpu = _f(run.get("register_s", r.get("wall_s"))), _f(r.get("cpu_s"))
    cpus = _f(run.get("cpus_requested") or run.get("n_cpus"))
    rounds = len(run.get("group_cases") or []) or 1
    patient = run.get("unit") == "patient"
    per_round = info.get("register_s_round")
    if per_round is None:
        per_round = wall / rounds if patient else wall
    write_s = _f(run.get("write_s") or 0.0)
    # The method's own writer where its adapter times it apart from the benchmark's own
    # copying (ASHLAR's split of the mosaic, DeeperHistReg's stacking): that copying is
    # in no total. `cost.end_to_end` takes the same number: ONE definition of write_s.
    own_writer = (run.get("native") or {}).get("write_parts_s")
    if isinstance(own_writer, dict) and own_writer.get("writer") is not None:
        write_s = _f(own_writer["writer"])

    def gb(key, src=r):
        return src[key] / 1024.0 if src.get(key) is not None else math.nan

    after = gb("peak_rss_after_register_mb", info)
    on_gpu = str(info.get("device", "")).startswith("cuda")
    adapter = str(run.get("method") or "")
    convert_key = cost.CLOCK.get(adapter, {}).get("convert", cost.NONE)
    native = run.get("native") if isinstance(run.get("native"), dict) else {}
    written = native.get("written")
    complete = cost.complete_image(written)
    parts = native.get("write_parts_s")
    parts = parts if isinstance(parts, dict) else {}
    previous = list(run.get("previous_attempts") or [])
    # An attempt the process could not bill (the scheduler killed it) is billed by the
    # scheduler's own figures for the whole task where the launcher got them; where it
    # did not, the attempt is counted as UNKNOWN, never as zero. An attempt that ended
    # for a cause that is not the method's is LOST, not wasted: it is in the `lost_*`
    # columns and in no `wasted_*` and no `total_*` figure.
    earlier, from_scheduler, unknown = [], 0, 0
    kinds = [billing.kind_of(p.get("cause")) for p in previous]
    for p, kind in zip(previous, kinds):
        wall_p, cpu_p = p.get("wall_s"), p.get("cpu_s")
        if wall_p is None and p.get("scheduler_elapsed_s") is not None:
            wall_p, cpu_p = p["scheduler_elapsed_s"], p.get("scheduler_cpu_s")
            from_scheduler += kind == billing.WASTED
        elif wall_p is None:
            unknown += kind == billing.WASTED
        earlier.append((_f(wall_p), _f(cpu_p)))
    wasted = [e for e, kind in zip(earlier, kinds) if kind == billing.WASTED]
    # this row's own last attempt, when it registered nothing
    own_kind = billing.kind_of(run.get("cause"))
    sched = run.get("scheduler") or {}
    if not ok and run.get("resources") is None and run.get("cause") != "not_run":
        # this result's own last attempt, killed before it could bill itself
        if sched.get("elapsed_s") is not None:
            wall, cpu = _f(sched["elapsed_s"]), _f(sched.get("cpu_s"))
            from_scheduler += own_kind == billing.WASTED
        elif run.get("launch_id"):
            unknown += own_kind == billing.WASTED
    watch = run.get("oom_watch") if isinstance(run.get("oom_watch"), dict) else None
    # ── the attempts of this row's registration (regbench.billing) ──
    registered = ok or bool(run.get("registered"))
    mem = _f(run.get("mem_requested_gb"))
    # the rungs of the ladder before this one: the job's CPUs are the same at every
    # rung (ladder.CPUS), its memory is the rung's
    attempts = [
        billing.attempt(
            kind,
            cpus,
            p.get("mem_requested_gb"),
            wall_p,
            cpu_p,
            p.get("launch_id"),
        )
        for p, (wall_p, cpu_p), kind in zip(previous, earlier, kinds)
    ]
    notes, good_jobs = [], []
    # A registration that ran as several scheduler jobs per round (regbench.stages):
    # its cost is summed over those jobs, and the CPUs behind its wall time are the
    # jobs', not this record's own allocation (the round's last job, the writer).
    jobs = info.get("jobs") if isinstance(info.get("jobs"), dict) else {}
    staged = jobs.get("execution") == "jobs"
    allocated_h = _f(jobs.get("cpu_h_allocated")) if staged else math.nan
    # what `cpu_efficiency` of a staged row divides by: the allocation of the attempts
    # whose CPU time `cpu` holds
    spent_alloc_h = allocated_h
    stitch = (
        native.get("stitch_job") if isinstance(native.get("stitch_job"), dict) else {}
    )
    unmeasured = 0
    if staged:
        # ONE rule for every method: an attempt that failed is wasted, and the cost is
        # the attempts that succeeded. Here the attempts are the stage jobs'.
        unmeasured = int(jobs.get("unmeasured_attempts") or 0)
        spent = [
            (
                _f(jobs.get("wasted_wall_s")),
                _f(jobs.get("wasted_cpu_h_measured")) * 3600,
            )
        ] * bool(jobs.get("wasted_attempts"))
        if not ok and not run.get("registered"):
            # a round that failed: what its stage jobs cost, not this job's own clock
            # and not the writer's (STITCH's failed attempts are the writer's time)
            good = _f(jobs.get("cpu_h_measured")) * 3600.0
            wall = _f(jobs.get("wall_unlimited_parallel_s")) + float(
                np.nansum([w for w, _ in spent])
            )
            cpu = good + float(np.nansum([c for _, c in spent]))
            # the allocation that CPU time was used on: the failed attempts' too
            if jobs.get("wasted_attempts"):
                spent_alloc_h = float(
                    np.nansum([allocated_h, _f(jobs.get("wasted_cpu_h_allocated"))])
                )
        else:
            wasted += spent + [
                (_f(a.get("wall_s")), math.nan)
                for a in stitch.get("failed_attempts") or []
            ]
    if staged:
        recs = None
        if stage is not None:
            recs = list(stage[0])
            mine = math.fsum(a["reserved_h"] for a in recs if a["kind"] == billing.GOOD)
            if not math.isclose(mine, allocated_h, rel_tol=1e-6, abs_tol=1e-9):
                recs = None  # the records are of another attempt at this round
                notes.append(billing.RECORDS_CHANGED)
        else:
            notes.append(billing.NOT_IN_RECORDS)
        if recs is not None and registered:
            attempts += recs
            good_jobs = [a["job"] for a in recs if a["kind"] == billing.GOOD]
        elif recs is not None:
            # a round that failed: every attempt of its stage jobs is wasted, or lost
            # when the round ended for a cause that is not the method's
            attempts += [
                {**a, "kind": own_kind if a["kind"] == billing.GOOD else a["kind"]}
                for a in recs
            ]
        else:
            good_h = (_f(jobs.get("cpu_h_measured")), allocated_h)
            lost_h = (
                (
                    _f(jobs.get("wasted_cpu_h_measured")),
                    _f(jobs.get("wasted_cpu_h_allocated")),
                )
                if jobs.get("wasted_attempts")
                else (0.0, 0.0)
            )
            if registered:
                attempts.append(billing.hours(billing.GOOD, *good_h))
                if jobs.get("wasted_attempts"):
                    attempts.append(billing.hours(billing.WASTED, *lost_h))
            else:
                attempts.append(
                    billing.hours(
                        own_kind,
                        float(np.nansum([good_h[0], lost_h[0]])),
                        float(np.nansum([good_h[1], lost_h[1]])),
                    )
                )
    elif registered:
        attempts.append(
            billing.attempt(billing.GOOD, cpus, mem, wall, cpu, run.get("launch_id"))
        )
        good_jobs = [run.get("launch_id") or ""]
    elif run.get("cause") != "not_run" and wall == wall and not (stage and stage[0]):
        # this row's own last attempt, which registered nothing (a round billed from its
        # stage records below is not also billed its last job's clock, as a round that
        # holds their summary is not)
        attempts.append(
            billing.attempt(own_kind, cpus, mem, wall, cpu, run.get("launch_id"))
        )
    if not staged and stage is not None and not registered:
        # a round of a registration in several jobs whose record holds no summary of
        # them (its last job was killed, or the run was refused before it read them):
        # what its stage jobs did is spent all the same, and none of it registered
        attempts += [
            {**a, "kind": own_kind if a["kind"] == billing.GOOD else a["kind"]}
            for a in stage[0]
        ]
    bill = billing.sums(attempts)
    patient_unit = run.get("unit") == "patient"
    gpx = _f(moving_px) / 1e9 if moving_px else math.nan
    sched_h, sched_n = (
        billing.scheduler_hours(good_jobs, accounting)
        if accounting is not None and registered
        else (math.nan, 0)
    )
    if staged and stitch:
        write_bill = {
            "write_cpu_h_used": _f(stitch.get("cpu_s")) / 3600.0,
            "write_cpu_h_reserved": _f(stitch.get("cpus"))
            * _f(stitch.get("wall_s"))
            / 3600.0,
            "write_wasted_cpu_h_reserved": float(
                np.nansum(
                    [
                        _f(a.get("cpus")) * _f(a.get("wall_s")) / 3600.0
                        for a in stitch.get("failed_attempts") or []
                    ]
                )
            ),
        }
    else:
        # a writer inside the row's own job: its CPUs are the job's, its CPU time is
        # what the run's counter advanced by while it ran (`write_cpu_s`), and a failed
        # one is the row's own failure
        write_cpu = _f(run.get("write_cpu_s"))
        beside = [k for k, v in sorted(parts.items()) if k != "writer" and _f(v) > 0]
        if ok and write_cpu == write_cpu and not staged:
            if run.get("write_cpu_from") == billing.WRITE_CPU_STRETCH:
                notes.append(billing.WRITE_CPU_IS_THE_STRETCH)
            elif beside:
                notes.append(billing.WRITE_CPU_HOLDS.format(", ".join(beside)))
        write_bill = {
            "write_cpu_h_used": write_cpu / 3600.0
            if ok and write_cpu == write_cpu and not staged
            else "",
            "write_cpu_h_reserved": cpus * write_s / 3600.0
            if ok and write_s > 0 and not staged
            else "",
            "write_wasted_cpu_h_reserved": "",
        }
    billed = {
        "device": (billing.GPU if on_gpu else billing.CPU) if adapter else "",
        "device_name": str(info.get("device") or "cpu") if adapter else "",
        **bill,
        **write_bill,
        "moving_gpx": gpx,
        # per round only where the row's CPU time is the round's own: a per-patient
        # launch has one clock for all its rounds (per patient in the summary)
        "cpu_h_used_per_gpx": billing.ratio(bill["cpu_h_used"], gpx)
        if not patient_unit
        else math.nan,
        "cpu_h_scheduler": sched_h if sched_h == sched_h else "",
        "scheduler_jobs": sched_n if sched_h == sched_h else "",
        "cpu_h_used_vs_scheduler": billing.ratio(bill["cpu_h_used"], sched_h)
        if sched_h == sched_h
        else "",
        "billing_note": "; ".join(notes),
    }
    return {
        "ok": ok,
        "cause": run.get("cause", ""),
        # ASHLAR's per-round flags (methods/ashlar.py); a row that failed carries none,
        # and an absent flag is None here, never False
        "fallback_stage_positions": info.get("fallback_stage_positions"),
        "cycle_fully_modelled": info.get("cycle_fully_modelled"),
        "unit": run.get("unit", ""),
        "launch_id": run.get("launch_id") or "",
        "rounds_in_launch": rounds,
        "attempt": run.get("attempt", math.nan),
        "mem_requested_gb": _f(run.get("mem_requested_gb")),
        "cpus_requested": cpus,
        "register_s": wall,
        "register_s_per_round": _f(per_round),
        # the measured CPU time of THIS attempt, whatever became of it: `cpu_h_used`
        # for a row that registered, wasted for one that did not (`wasted_cpu_h_used`)
        "cpu_h": cpu / 3600.0,
        # `cpu_h_used / cpu_h_reserved`: above 1 where a method's threads are not
        # confined to its CPUs
        "cpu_efficiency": (
            cpu / 3600.0 / spent_alloc_h
            if staged and spent_alloc_h
            else cpu / (wall * cpus)
            if wall and cpus and not staged
            else math.nan
        ),
        # only for a registration that ran as several scheduler jobs per round, blank
        # elsewhere (definitions: regbench.stages). `register_s` of such a row IS
        # `wall_unlimited_parallel_s`, `cpu_h` the measured CPU time summed over its
        # jobs, and `peak_rss_gb` the peak of its largest single job.
        "elapsed_with_queue_s": _f(jobs.get("elapsed_with_queue_s")) if staged else "",
        "wall_unlimited_parallel_s": _f(jobs.get("wall_unlimited_parallel_s"))
        if staged
        else "",
        "cpu_h_allocated": allocated_h if staged else "",
        "largest_job": str(jobs.get("largest_job") or "") if staged else "",
        "peak_cpus_at_once": _f(jobs.get("peak_cpus_at_once")) if staged else "",
        "n_jobs": _f(jobs.get("n_jobs")) if staged else "",
        "n_tiles": _f(jobs.get("n_tiles")) if staged else "",
        "n_job_retries": _f(jobs.get("n_retries")) if staged else "",
        # the writer's own job: allocated CPUs x its duration, and its peak
        "write_cpu_h_allocated": _f(stitch.get("cpus"))
        * _f(stitch.get("wall_s"))
        / 3600.0
        if staged and stitch
        else "",
        "write_peak_gb": _f(stitch.get("peak_rss_mb")) / 1024.0
        if staged and stitch
        else "",
        "peak_rss_gb": gb("peak_rss_mb"),
        # for a method whose writer cannot be paused, the peak it read after registering
        "peak_rss_after_register_gb": after if after == after else gb("peak_rss_mb"),
        "cgroup_peak_gb": gb("cgroup_peak_mb"),  # whole job, writers included
        "gpu_h": wall / 3600.0 * (run.get("gpus") or 1) if on_gpu else 0.0,
        "gpu_model": run.get("gpu_model", "") if on_gpu else "",
        "write_s": write_s,
        "total_to_image_s": wall + write_s,
        "workers_seen": info.get("workers_seen", run.get("workers", "")),
        "wasted_attempts": len(wasted),
        "wasted_cpu_h": float(np.nansum([c for _, c in wasted])) / 3600.0,
        "wasted_wall_s": float(np.nansum([w for w, _ in wasted])),
        # attempts whose cost is the scheduler's elapsed time of the whole task (an
        # upper bound: container start and the benchmark's own steps are in it), and
        # attempts whose cost nobody recorded (in NO wasted total)
        "wasted_from_scheduler": int(from_scheduler),
        "wasted_unknown": int(unknown),
        # attempts that ended for a cause that is not the method's (a node failure, a
        # cancelled job, the benchmark's own step): billed in `lost_*`, in no
        # `wasted_*` and no `total_*`
        "lost_attempts": sum(kind == billing.LOST for kind in kinds)
        + int(
            own_kind == billing.LOST
            and not registered
            and run.get("cause") != "not_run"
        ),
        # stage attempts that ended without a recorded duration (a job the scheduler
        # ended at its time limit): every time and CPU figure of the row is then a
        # LOWER BOUND. Blank for a run that is one job
        "unmeasured_attempts": unmeasured if staged else "",
        # the transform existed when this row failed: its writer, or the job around
        # the writer, failed. The registration is then no wasted attempt
        "registered": bool(run.get("registered")),
        "retry": run.get("retry", 0),
        "time_limit": run.get("time_limit", ""),
        # "" when the run recorded nothing about its watcher (an older record)
        "oom_watch": "" if watch is None else ("yes" if watch.get("started") else "no"),
        "reused": bool(run.get("reused_by")),
        "rss_method": r.get("rss_method", ""),
        # the counter `cpu_h` was read from (regbench.resources): the same in kind for
        # every row unless this says "polled (may undercount)"
        "cpu_instrument": r.get("cpu_instrument", ""),
        # for the end-to-end view (regbench.cost): the method's own conversion where it
        # is timed apart, the device, and what its writer left on disk
        "adapter": adapter,
        # what the scheduler was asked for, with no fallback (NaN outside a scheduler)
        "cpus_asked": _f(run.get("cpus_requested")),
        **billed,
        # a method that starts a JVM (VALIS): the heap its adapter asked for, and the
        # heap the JVM itself reports (`Runtime.maxMemory()`), as recorded. Blank for
        # a method without one
        "jvm_heap_asked_gb": _blank(info.get("jvm_heap_gb")),
        "jvm_max_heap_gb": _blank(info.get("jvm_max_heap_gb")),
        # extra cost columns a private method plug-in declares (none without one)
        **{
            column: info.get(key, "")
            for column, key in (plugins.get("COST_COLUMNS", {}) or {}).items()
        },
        "method_convert_s": _f(info.get(convert_key))
        if convert_key not in (cost.NONE, cost.NOT_SEPARABLE)
        else math.nan,
        # where the adapter times the parts of its write step apart: the package's own
        # writer, a conversion that writer needed, and the benchmark's own copying
        "write_writer_s": _f(parts.get("writer")),
        "write_convert_s": _f(parts.get("convert")),
        "write_benchmark_s": _f(parts.get("benchmark")),
        "written": cost.written_text(written),
        "written_complete": "" if complete is None else complete,
        "written_file": str((native.get("files") or [""])[0]),
        "cpus_affinity": _f(run.get("cpus_affinity")),
        "cpus_node": _f(run.get("cpus_node")),
        "host": run.get("host", ""),
        "cpu_model": run.get("cpu_model", ""),
        "blas": run.get("blas", ""),
        # the thread variables the run found SET in its own process (run.json:
        # threads_env), as text: the benchmark exports none, so one that is here came
        # with the method's image (ASHLAR's sets OMP_NUM_THREADS=1)
        "threads_env": "; ".join(
            f"{k}={v}"
            for k, v in sorted((run.get("threads_env") or {}).items())
            if v not in (None, "")
        )
        if isinstance(run.get("threads_env"), dict)
        else "",
    }


def moving_pixels(case):
    """Pixels of a case's moving image (one plane), None when the case has no size."""
    hw = getattr(case, "mov_hw", None)
    return int(hw[0]) * int(hw[1]) if hw and len(hw) == 2 else None


def _in_stages(row):
    """Is the row of a method whose registration is several stage jobs per round (its
    adapter declares ``STAGES``)?"""
    from . import methods

    adapter = str(row.get("adapter") or "")
    if adapter not in methods.METHODS:
        return False
    try:
        return bool(getattr(methods.load(adapter), "STAGES", None))
    except Exception:  # an adapter that does not import here has no stage records
        return False


def bill_from_disk(res_rows, out, cases, accounting=None):
    """Complete the billing columns of the resource rows with what is not in a
    ``run.json``, in place:

    * a registration in several scheduler jobs gets its attempts from its stage jobs'
      records (``billing.stage_attempts``): each attempt's own memory and job, hence
      ``gb_h_reserved`` and the jobs the scheduler is asked about. Also a failed round
      whose ``run.json`` has no ``info.jobs`` (its last job was killed, or the run was
      refused before it read them): the stage jobs ran, and their cost is its wasted
      (or lost) cost;
    * every row that registered gets ``cpu_h_scheduler`` from ``accounting``
      (``ladder.read_accounting`` of ``<out>/logs/accounting.psv``), when there is one.

    Done at table time and kept in no record: the stage records and the accounting
    are not in a case's key, so nothing read from them may be stored under it."""
    by_id = {(c.dataset, str(c.case_id)): c for c in cases}
    for row in res_rows:
        case = by_id.get((row["dataset"], str(row["case_id"])))
        if case is None:
            continue
        if row.get("n_jobs") in ("", None):  # one job: its own launch
            if accounting is not None and (row["ok"] or row.get("registered")):
                h, n = billing.scheduler_hours([row.get("launch_id")], accounting)
                if h == h:
                    row["cpu_h_scheduler"], row["scheduler_jobs"] = h, n
                    row["cpu_h_used_vs_scheduler"] = billing.ratio(row["cpu_h_used"], h)
            # ... unless it is a failed round of a registration in several jobs whose
            # record never got their summary: its stage records are read below
            if row["ok"] or row.get("registered") or not _in_stages(row):
                continue
        folder = billing.find_stage_dir(
            out, row["dataset"], row["case_id"], row["method"], row.get("launch_id")
        )
        stage = billing.stage_attempts(folder) if folder is not None else None
        if stage is None:
            continue
        try:
            run = json.loads(
                (result_dir(out, row["method"], case) / "run.json").read_text()
            )
        except (OSError, ValueError):
            continue
        again = resource_row(run, moving_pixels(case), stage, accounting)
        row.update({k: again[k] for k in billing.COLUMNS})
    return res_rows


def find_variants(out, dataset, expected=None):
    """The variants to score: ``initial``, every result folder of the dataset, and the
    ``expected`` names that have none. Nothing is left out for what its selfcheck
    said: the verdict is beside the row (``selfcheck_status``, ``selfcheck_row``)."""
    out = Path(out)
    found = (
        [
            d.name
            for d in sorted(out.iterdir())
            if d.is_dir()
            and d.name not in RESERVED
            and d.name != "initial"
            and (d / dataset).is_dir()
        ]
        if out.exists()
        else []
    )
    extra = [v for v in expected or [] if v not in found and v != "initial"]
    return ["initial", *found, *extra]


# `score --submission ID`: only what that submission ran or reused is scored. In the
# environment because the scorer's pool workers are spawned and read it there.
SUBMISSION_ENV = "REGBENCH_SCORE_SUBMISSION"
STALE = (
    "not run by this submission (the result on disk is another submission's and was "
    "not reused: a unit that never started)"
)


def of_this_submission(run):
    """False for a record another submission left, when one is being scored alone."""
    from . import ladder

    return ladder.mine(run, os.environ.get(SUBMISSION_ENV) or "")


STALE_CASE = (
    "the result was made for an earlier version of this case (its points.npz has "
    "changed since: the case was prepared again): not run on the case as it is"
)


def _load_run(out, variant, case):
    """``(run, warped, problem, cause)`` of one (variant, case); ``warped`` is None on failure."""
    d = result_dir(out, variant, case)
    run = {}
    if (d / "run.json").exists():
        try:
            run = json.loads((d / "run.json").read_text())
        except ValueError:
            return {}, None, "run.json is not readable", "exception"
    if not run:
        return {}, None, "not run", "not_run"
    if not of_this_submission(run):
        return {}, None, STALE, "not_run"
    changed = stale_points(run.get("points_sha256"), case)
    if changed:
        # not the method's failure: its polygons no longer fit because the CASE changed
        return {}, None, f"{STALE_CASE} ({', '.join(changed)})", "not_run"
    if not run.get("ok") or not (d / "warped.npz").exists():
        return (
            run,
            None,
            run.get("error") or "no output",
            run.get("cause") or "exception",
        )
    with np.load(d / "warped.npz") as z:
        warped = {k: z[k] for k in z.files}
    return run, warped, str(run.get("problem") or ""), ""


def _seam_masks(
    out, case, variants, owners, ref, mov, points, which=OUTLINES, load=None
):
    """``({owner: (keep_ref | None, keep_mov | None)}, [owners without
    a mask])`` for the owners that have a row on this dataset. The mask is read from the
    owner's own primary run and from nothing else: no other method's result can change it.

    ``which`` is the polygon set: the mask of the bare nuclei is the owner's ``band_nuc``
    / ``band_ref_nuc`` (``methods._nuclei_band`` says how the driver got them).
    ``load`` reads a variant's result (default
    ``_load_run``): a caller that has read it already passes its own."""
    _, key, key_ref = WARPED_KEYS[which]
    masks, missing = {}, []
    for owner in owners:
        if owner not in variants:
            continue
        _, warped, _, _ = _load_run(out, owner, case) if load is None else load(owner)
        band = warped.get(key) if warped is not None else None
        band_ref = warped.get(key_ref) if warped is not None else None
        fits = (band is None or len(band) == mov.n_features) and (
            band_ref is None or len(band_ref) == ref.n_features
        )
        if (band is None and band_ref is None) or not fits:
            missing.append(owner)
            continue
        keep_mov = None if band is None else ~np.asarray(band, dtype=bool)
        masks[owner] = (
            None if band_ref is None else ~np.asarray(band_ref, dtype=bool),
            keep_mov,
        )
    return masks, missing


# what a row's `selfcheck_row` says when its stage has no image to be checked against
NEVER_CHECKED = "no_stage_image"


# ── one case: what is read, what is computed, what is kept ───────────────────────────
# Bump when anything a case's rows or a pair record depend on changes in a way the
# hashes below cannot see. They see the code (`case_code`, `pair_code`: the files of
# this package and the modules they import, the pipeline's cell_pairs.py),
# the libraries' versions and the constants in `_settings`; this
# number is for the rest (a changed meaning with unchanged code).
RECORD_VERSION = 1
# where the records are kept, under the results folder (`cache` is RESERVED)
RECORDS = ("cache", "scores")
# a pair record's source: the unregistered pose, the true map, or a variant's result
INITIAL, VARIANT = "initial", "variant"
PROGRESS_EVERY_S = 30.0

_FILE_SHA = {}


def file_sha256(path):
    """sha256 of a file's bytes, ``""`` when it is not there. Kept per file state (path,
    size, modification time), so a file is read once per process."""
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return ""
    state = (str(path), st.st_size, st.st_mtime_ns)
    if state not in _FILE_SHA:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        _FILE_SHA[state] = h.hexdigest()
    return _FILE_SHA[state]


def _array_sha256(a):
    if a is None:
        return ""
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype}:{a.shape}:".encode() + a.tobytes()).hexdigest()


def _libraries():
    import platform

    import scipy
    import skimage

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-image": skimage.__version__,
    }


def _settings():
    """The constants a record's numbers depend on, as they are NOW (not as written in
    the files: a constant changed in a running process is a different scorer)."""
    return {
        "gate_factor": pairs.GATE_FACTOR,
        "null_shift_radii": pairs.NULL_SHIFT_RADII,
        "null_directions": pairs.NULL_DIRECTIONS,
        "curve_um": list(pairs.CURVE_UM),
        "iou_thresh": pairs.IOU_THRESH,
        "readable_margin": pairs.READABLE_MARGIN,
        "max_component_cells": pairs.cp.DEFAULT_MAX_COMPONENT_CELLS,
        "exact_max_work": pairs.EXACT_MAX_WORK,
        "max_pair_window_px": pairs.cp.DEFAULT_MAX_PAIR_WINDOW_PX,
        "coverage_bin_radii": COVERAGE_BIN_RADII,
        "not_the_method": list(NOT_THE_METHOD),
        "not_the_method_rounds": NOT_THE_METHOD_ROUNDS,
    }


_CODE_FILES = {}


def _states(paths):
    """Size and modification time of each file: what `code_files` keeps its answer by."""
    out = []
    for path in paths:
        try:
            st = os.stat(path)
            out.append((str(path), st.st_size, st.st_mtime_ns))
        except OSError:
            out.append((str(path), None, None))
    return out


def code_files(starts, package=None):
    """``{name: path}`` of ``starts`` (files of ``package``, default this one) and of
    every module of the package they import, followed through: ``from . import x``,
    ``from .x import y``, ``from .. import z``, at module level or inside a function.
    ``name`` is the path inside the package. An import that leaves the package (NumPy,
    the pipeline's ``cell_pairs.py``, which is loaded by path) is not followed."""
    import ast

    package = Path(__file__).resolve().parent if package is None else Path(package)
    package = package.resolve()
    found, todo = {}, [Path(f).resolve() for f in starts]
    asked = (str(package), tuple(sorted(map(str, todo))))
    held = _CODE_FILES.get(asked)
    if held is not None and held[0] == _states(held[1].values()):
        return dict(held[1])  # no file it was found through has changed since
    while todo:
        path = todo.pop()
        try:
            name = path.relative_to(package).as_posix()
        except ValueError:
            continue
        if name in found or not path.is_file():
            continue
        found[name] = path
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            names = [node.module] if node.module else [a.name for a in node.names]
            if node.module:  # `from .x import y`: y may be a module of the package x
                names += [f"{node.module}.{a.name}" for a in node.names]
            for dotted in names:
                target = base.joinpath(*dotted.split("."))
                # by its exact name: a file system that ignores case would take the
                # constant `DATASETS` for the package `datasets`
                if target.parent.is_dir() and any(
                    target.name == Path(f).stem for f in os.listdir(target.parent)
                ):
                    todo += [target.with_suffix(".py"), target / "__init__.py"]
    found = dict(sorted(found.items()))
    _CODE_FILES[asked] = (_states(found.values()), found)
    return dict(found)


# What of THIS file decides a pair record: how a result is read and judged usable, how
# the polygon sets of a record are put together, the seam masks, and the names the
# arrays are read under. The rest of this file (summaries, tables) decides no pair
# record, so a change there keeps them: they are the expensive part of a scoring.
PAIR_CODE_MEMBERS = (
    "_Case",
    "_keep_features",
    "_load_run",
    "of_this_submission",
    "_seam_masks",
)
PAIR_CODE_CONSTANTS = (
    "WARPED_KEYS",
    "INITIAL",
    "VARIANT",
    "OUTLINES",
    "NUCLEI",
    "MOV_TO_REF",
    "REF_TO_MOV",
    "SUBMISSION_ENV",
)
# the modules a pair record is computed by, each followed through its imports
PAIR_CODE_STARTS = ("pairs.py", "cases.py", "mapping.py")


_OWN_SOURCE = {}


def pair_code():
    """What turns two polygon sets into a pair record: the pipeline's ``cell_pairs.py``;
    ``pairs.py``, ``cases.py`` (how a set is read) and ``mapping.py`` with every module
    of this package they import (``code_files``); and, of this file, the source of
    ``PAIR_CODE_MEMBERS`` and the values of ``PAIR_CODE_CONSTANTS``, as they are in the
    running process."""
    import inspect

    here = Path(__file__).resolve().parent
    module = sys.modules[__name__]
    members = [getattr(module, n) for n in PAIR_CODE_MEMBERS]
    # the source of the objects that are there NOW, read once per object and file state
    state = (*_states([__file__])[0], *map(id, members))
    if _OWN_SOURCE.get("state") != state:
        _OWN_SOURCE.update(
            state=state,
            members=members,  # kept alive: an id is theirs only while they live
            source="".join(inspect.getsource(m) for m in members),
        )
    own = _OWN_SOURCE["source"]
    own += repr([(n, getattr(module, n)) for n in PAIR_CODE_CONSTANTS])
    return {
        "cell_pairs.py": file_sha256(pairs.CELL_PAIRS_PATH),
        **{
            name: file_sha256(path)
            for name, path in code_files(here / f for f in PAIR_CODE_STARTS).items()
        },
        "score.py:" + ",".join(PAIR_CODE_MEMBERS): hashlib.sha256(
            own.encode()
        ).hexdigest(),
    }


def case_code():
    """Everything a case's rows could depend on: every file of this package's top
    level, every module this file imports inside the package, followed through
    (``code_files``: a module in a sub-package too) and the pipeline's
    ``cell_pairs.py``. Deliberately wide: a case record that is not reused costs
    minutes (its pair records are), a wrong one costs a table."""
    here = Path(__file__).resolve().parent
    files = {f.name: f for f in sorted(here.glob("*.py"))}
    files.update(code_files([Path(__file__)]))
    return {
        "cell_pairs.py": file_sha256(pairs.CELL_PAIRS_PATH),
        **{name: file_sha256(path) for name, path in sorted(files.items())},
    }


def _key_sha256(key):
    return hashlib.sha256(
        json.dumps(key, sort_keys=True, default=str).encode()
    ).hexdigest()


def _differs(mine, theirs, prefix=""):
    """The items of two keys that are not the same, as dotted names."""
    if not isinstance(mine, dict) or not isinstance(theirs, dict):
        return [] if mine == theirs else [prefix or "key"]
    out = []
    for k in sorted({*mine, *theirs}, key=str):
        name = f"{prefix}.{k}" if prefix else str(k)
        if k not in mine or k not in theirs:
            out.append(name)
        else:
            out += _differs(mine[k], theirs[k], name)
    return out


def _write_record(path, key, payload):
    """One record file, written whole or not at all: the sha256 of the body on the
    first line, then the pickled ``{"key", "payload"}``. Pickle because the rows hold
    NumPy scalars whose type decides how a table prints them, and the pool that scores
    cases side by side has always moved them between processes that way."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = pickle.dumps({"key": key, "payload": payload}, protocol=4)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(hashlib.sha256(body).hexdigest().encode() + b"\n" + body)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_record(path, key):
    """``(payload, "")`` of a record made for exactly ``key``, else ``(None, why)``. A
    record is evidence only when it is whole and its key is this one: anything else is
    computed again, never trusted."""
    path = Path(path)
    if not path.exists():
        return None, "no record"
    try:
        digest, _, body = path.read_bytes().partition(b"\n")
        if hashlib.sha256(body).hexdigest().encode() != digest:
            return None, "unreadable record (cut short or damaged)"
        held = pickle.loads(body)
        theirs, payload = held["key"], held["payload"]
    except Exception as exc:  # whatever a damaged pickle raises
        return None, f"unreadable record ({type(exc).__name__})"
    if theirs != key:
        changed = _differs(key, theirs)
        return None, "made for other inputs: " + ", ".join(changed[:6]) + (
            f" and {len(changed) - 6} more" if len(changed) > 6 else ""
        )
    return payload, ""


def records_dir(out, case):
    """Where a case's records are: ``<out>/cache/scores/<dataset>/<case>/``, with
    ``case.rec`` (its rows), ``case.json`` (the key, to be read by a person) and
    ``pairs/<key>.rec`` (one per pair record)."""
    return Path(out).joinpath(*RECORDS, case.dataset, str(case.case_id))


def case_key(job):
    """Everything that decides a case's rows. A case record is reused only when this is
    EQUAL: the case as prepared, the variants and their order, the bytes of every
    result the case reads (``run.json`` and ``warped.npz``; the seam masks are in the
    owners' ``warped.npz``), the seam owners, each row's selfcheck, the submission
    being scored, the code, the libraries and the constants."""
    out, case, variants, owners = job[:4]
    checks = job[4] if len(job) > 4 else {}
    results = {}
    for v in variants:
        if v == "initial":
            continue
        d = result_dir(out, v, case)
        results[v] = {
            "run.json": file_sha256(d / "run.json"),
            "warped.npz": file_sha256(d / "warped.npz"),
        }
    return {
        "record_version": RECORD_VERSION,
        "code": case_code(),
        "libraries": _libraries(),
        "settings": _settings(),
        "dataset": case.dataset,
        "case_id": str(case.case_id),
        "case.json": file_sha256(Path(case.dir) / "case.json"),
        "points.npz": file_sha256(case.points_path),
        "variants": list(variants),
        "results": results,
        "seam_owners": list(owners),
        "checks": {v: checks.get(v, "") for v in variants},
        "submission": os.environ.get(SUBMISSION_ENV) or "",
    }


class _Progress:
    """Lines on stderr, so a log is never silent for hours: every case as it starts and
    ends, and its pair records as they finish, at most one line per PROGRESS_EVERY_S."""

    def __init__(self, case):
        self.name = f"{case.dataset}/{case.case_id}"
        self.started = self.last = time.monotonic()

    def say(self, text):
        print(f"score: {self.name}: {text}", file=sys.stderr, flush=True)
        self.last = time.monotonic()

    def elapsed(self):
        return time.monotonic() - self.started

    def step(self, done, total, what, seconds):
        if done == total or time.monotonic() - self.last >= PROGRESS_EVERY_S:
            self.say(
                f"pair record {done}/{total} ({what}: {seconds:.1f} s), "
                f"{self.elapsed():.0f} s so far"
            )


class _Case:
    """What scoring one case reads, each thing read once: its points, its polygon sets
    and their geometry, every variant's result, the owners' seam masks.

    The arrays of the last ``KEEP_WARPED`` results are held, the rest are read again
    when asked for: a slide has hundreds of megabytes per result and a case has dozens
    of results. What a result SAID (its ``run.json``, problem and cause) is kept for
    all of them."""

    KEEP_WARPED = 2

    def __init__(self, job):
        self.job = job
        self.out, self.case, self.variants, self.owners = job[:4]
        self.checks = job[4] if len(job) > 4 else {}
        self.points = load_points(self.case)
        self.px = self.case.pixel_size_um
        self._said, self._warped = {}, {}
        self._sets, self._geometry, self._masks, self._jac = {}, {}, {}, {}

    def load(self, v):
        """``_load_run`` of one variant of this case."""
        if v in self._warped:
            warped = self._warped.pop(v)
        else:
            run, warped, problem, cause = _load_run(self.out, v, self.case)
            self._said.setdefault(v, (run, problem, cause))
            while len(self._warped) >= self.KEEP_WARPED:
                del self._warped[next(iter(self._warped))]
        self._warped[v] = warped  # the most recent last
        run, problem, cause = self._said[v]
        return run, warped, problem, cause

    def sets(self, which):
        """``(reference, moving)`` native polygons of one set; None where there is none."""
        if which not in self._sets:
            self._sets[which] = (
                polys(self.points, "ref", which),
                polys(self.points, "mov", which),
            )
        return self._sets[which]

    def geometry(self, which):
        """Areas and centroids of the native sets, kept for the whole case."""
        return self._geometry.setdefault(which, pairs.Geometry())

    def masks(self, which):
        """``_seam_masks`` of one set: ``({owner: keeps}, [owners without a mask])``."""
        if which not in self._masks:
            ref, mov = self.sets(which)
            self._masks[which] = _seam_masks(
                self.out,
                self.case,
                self.variants,
                self.owners,
                ref,
                mov,
                self.points,
                which,
                self.load,
            )
        return self._masks[which]

    def jacobian(self, v):
        if v not in self._jac:
            self._jac[v] = jacobian_stats(self.load(v)[1])
        return self._jac[v]

    def inputs(self, which, source, owner=None):
        """The four arguments of ``pairs.record`` for one pair record: the two native
        sets, and the two sets in the frame of ``source`` (``(INITIAL, "")`` or
        ``(VARIANT, name)``), without the nuclei on ``owner``'s
        seams when one is named."""
        ref, mov = self.sets(which)
        kind, name = source
        ref_in, mov_in = ref, mov
        if kind == VARIANT:
            warped = self.load(name)[1]
            xy = warped[WARPED_KEYS[which][0]].astype(float)
            if str(warped["direction"]) == REF_TO_MOV:
                ref_in = pairs.with_xy(ref, xy)
            else:
                mov_in = pairs.with_xy(mov, xy)
        if owner is not None:
            keep_ref, keep_mov = self.masks(which)[0][owner]
            if keep_ref is not None:
                ref_in = _keep_features(ref_in, keep_ref)
            if keep_mov is not None:
                mov_in = _keep_features(mov_in, keep_mov)
        return ref, mov, ref_in, mov_in

    def record(self, which, source, owner=None):
        """``pairs.record`` of one pair record. The gate comes from ALL native nuclei,
        with or without an owner: one gate per case and set."""
        ref, mov, ref_in, mov_in = self.inputs(which, source, owner)
        geometry = self.geometry(which)
        geometry.area_cent(ref)
        geometry.area_cent(mov)
        return pairs.record(ref, mov, ref_in, mov_in, self.px, geometry)

    def pair_key(self, which, source, owner=None):
        """What decides one pair record, and nothing else: the case's points, the set,
        the pixel size, the bytes of the result that moved the polygons, the owner's
        mask itself (not the file it came from), the code, libraries and constants.
        What a result SAYS (status, cause, selfcheck) is in no pair record: a changed
        ``run.json`` changes a case's rows and none of its pair records."""
        kind, name = source
        keep_ref = keep_mov = None
        if owner is not None:
            keep_ref, keep_mov = self.masks(which)[0][owner]
        return {
            "record_version": RECORD_VERSION,
            "code": pair_code(),
            "libraries": _libraries(),
            "settings": _settings(),
            "points.npz": file_sha256(self.case.points_path),
            "set": which,
            "pixel_size_um": repr(self.px),
            "source": kind,
            "warped.npz": file_sha256(
                result_dir(self.out, name, self.case) / "warped.npz"
            )
            if kind == VARIANT
            else "",
            "masked": owner is not None,
            "keep_ref": _array_sha256(keep_ref),
            "keep_mov": _array_sha256(keep_mov),
        }


def _spec_name(spec):
    which, (kind, name), owner = spec
    return f"{which} {name or kind}" + (f"{NON}{owner}" if owner else "")


_IN_WORKER = {}


def _pair_task(task):
    """One pair record, in a pool worker. The worker reads the case itself (nothing but
    the job and the record's name crosses the process boundary on the way in, and the
    record's numbers on the way out) and keeps it for the next task of the same case."""
    i, job, spec = task
    name = (job[0], job[1].dataset, str(job[1].case_id))
    if _IN_WORKER.get("name") != name:
        _IN_WORKER.clear()
        _IN_WORKER.update(name=name, case=_Case(job))
    started = time.monotonic()
    return i, _IN_WORKER["case"].record(*spec), time.monotonic() - started


def _records(ctx, specs, pool=None, fresh=False, progress=None):
    """``{spec: pair record}`` for the pair records a case needs, each exactly once.

    A record kept from an earlier scoring of the same inputs (``pair_key``) is read;
    the others are computed, side by side when there is a ``pool``, and each is saved
    as it finishes, so a scoring that is killed loses the records in flight and no
    more. The result does not depend on the order they finish in. ``fresh`` reads
    nothing."""
    folder = records_dir(ctx.out, ctx.case) / "pairs"
    keys = {spec: ctx.pair_key(*spec) for spec in specs}
    paths = {spec: folder / f"{_key_sha256(keys[spec])}.rec" for spec in specs}
    recs, todo = {}, []
    for spec in specs:
        got = None if fresh else _read_record(paths[spec], keys[spec])[0]
        if got is None:
            todo.append(spec)
        else:
            recs[spec] = got
    if progress is not None:
        progress.say(
            f"{len(specs)} pair record(s): {len(recs)} kept from an earlier scoring, "
            f"{len(todo)} to compute"
            + (", side by side" if pool is not None and len(todo) > 1 else "")
        )

    def keep(spec, rec, seconds):
        recs[spec] = rec
        _write_record(paths[spec], keys[spec], rec)
        if progress is not None:
            progress.step(
                len(recs) - (len(specs) - len(todo)),
                len(todo),
                _spec_name(spec),
                seconds,
            )

    if pool is not None and len(todo) > 1:
        # the outlines first: their pairs are the larger windows, and the longest
        # tasks must not be the last to start
        order = sorted(range(len(todo)), key=lambda i: (todo[i][0] != OUTLINES, i))
        tasks = [(i, ctx.job, todo[i]) for i in order]
        for i, rec, seconds in pool.imap_unordered(_pair_task, tasks):
            keep(todo[i], rec, seconds)
    else:
        for spec in todo:
            started = time.monotonic()
            rec = ctx.record(*spec)
            keep(spec, rec, time.monotonic() - started)
    return recs


def _kept(ps, keep):
    """How many features ``_keep_features(ps, keep)`` has (all of them without a mask)."""
    return int(ps.n_features) if keep is None else int(np.asarray(keep, bool).sum())


def _score_case(job, pool=None, fresh=False, progress=None):
    """Every row of one case: ``(cell rows, resource rows, off-seam cell rows, owners
    without a mask, nuclei)``. The first four are the OUTLINES'
    (and what does not depend on a polygon set); ``nuclei`` is ``{"cells", "non_cells",
    "missing"}``: the same cell rows on the bare nuclei, and, where there are none, why.

    In two steps. The rows are laid out first, each naming the pair record it reads
    (``(set, source, owner)``); then every DISTINCT pair record is computed once
    (``_records``) and the rows are filled. A variant that failed, was killed or never
    ran is scored at the unregistered pose: all of them read the one pair record of
    that pose, and one more per seam owner."""
    ctx = _Case(job)
    case, variants, checks = ctx.case, ctx.variants, ctx.checks
    points, px = ctx.points, ctx.px
    base = {
        "dataset": case.dataset,
        "tier": tier_of(case),
        "case_id": case.case_id,
        "group": case.group,
        "patient_id": case.group,
        "panel": panel_of(case),
        "unit": unit_of(case),
        "family": case.family,
        "pose": case.pose,
    }
    res_rows = []
    versions = {INITIAL: "-"}  # variant -> the tool version its run recorded
    flat = {"fold_pct": 0.0, "sd_log_jac": 0.0}
    wanted = {}  # the pair records the rows read, in the order they are first named
    initial = (INITIAL, "")

    def one_set(which):
        """The cell rows of one polygon set, each a function of the pair records. The
        resource rows do not depend on the set: they are made once, with the outlines."""
        first = which == OUTLINES
        xy_key = WARPED_KEYS[which][0]
        ref, mov = ctx.sets(which)
        scorable = bool(ref is not None and mov is not None and ref.n_features)
        scorable = scorable and bool(mov.n_features)
        dens = density(ref, px, ctx.geometry(which)) if scorable else {}
        cell_rows, non_cells, absent = [], [], []
        masks, missing = ctx.masks(which) if scorable else ({}, [])
        marks = set_marks(case, which)

        def row(method, frame, status, spec, jac, imputed, problem, cause):
            wanted.setdefault(spec)

            def fill(recs):
                rec = recs[spec]
                return {
                    **base,
                    "tool_version": versions.get(method.split(NON)[0], NOT_RECORDED),
                    "set": which,
                    **marks,
                    "method": method,
                    "frame": frame,
                    "selfcheck_status": status,
                    # THIS row's own check (variants.row_checks); selfcheck_status
                    # above is its run's verdict (set at score time from verdicts.tsv;
                    # `set_selfcheck_status`), which a row that was never checked
                    # shares with its run
                    "selfcheck_row": checks.get(method.split(NON)[0], ""),
                    **{k: rec[k] for k in pairs.RECORD_KEYS},
                    **dens,
                    **jac,
                    "imputed_initial": imputed,
                    "problem": problem[:200],
                    "cause": cause,
                    # the row stays; no summary of this method reads it
                    # (NOT_THE_METHOD_ROUNDS)
                    "excluded_from_summaries": _excluded(
                        {"imputed_initial": imputed, "cause": cause}
                    ),
                }

            return fill

        def off_seams(v, frame, status, source, jac, imputed, problem, cause):
            """The same variant on the nuclei each owner's mask leaves, one row per owner."""
            for owner, (keep_ref, keep_mov) in masks.items():
                # the gate still comes from ALL native nuclei: one gate per case and set
                fill = row(
                    f"{v}{NON}{owner}",
                    frame,
                    status,
                    (which, source, owner),
                    jac,
                    imputed,
                    problem,
                    cause,
                )
                more = {
                    "variant": v,
                    "owner": owner,
                    "n_mov_all": int(mov.n_features),
                    "n_mov_kept": _kept(mov, keep_mov),
                    "n_ref_all": int(ref.n_features),
                    "n_ref_kept": _kept(ref, keep_ref),
                }
                non_cells.append(
                    lambda recs, fill=fill, more=more: {**fill(recs), **more}
                )

        for v in variants:
            if v == "initial":
                run, warped, problem, cause = {}, None, "", ""
            else:
                run, warped, problem, cause = ctx.load(v)
                versions[v] = version_of(run)
                if first:
                    res_rows.append(
                        {
                            **base,
                            "method": v,
                            "tool_version": versions[v],
                            **resource_row(
                                run or {"cause": "not_run"}, moving_pixels(case)
                            ),
                        }
                    )
            if not scorable:
                continue
            if warped is not None and xy_key not in warped:
                # a result that registered but never moved THIS set (the nuclei, for a
                # run made before the case had them): no row, and the reason; never the
                # unregistered pose, which would charge the method for it
                absent.append({**base, "method": v, "reason": NO_NUCLEI_IN_RESULT})
                continue
            direction = str(warped["direction"]) if warped is not None else MOV_TO_REF
            if warped is not None:
                moved_side = mov if direction == MOV_TO_REF else ref
                if direction not in FRAME or len(warped[xy_key]) != len(moved_side.xy):
                    problem = "warped.npz does not fit the case's nuclei"
                    warped, cause, direction = None, "exception", MOV_TO_REF
            imputed = v != "initial" and warped is None
            status = run.get("selfcheck_status", "") if run else ""
            if warped is None:
                # the unregistered pose: ONE pair record for every such variant of the
                # case (and one per owner), whatever the reason it has no result
                said = (flat, imputed, problem, cause)
                cell_rows.append(
                    row(v, "reference", status, (which, initial, None), *said)
                )
                off_seams(v, "reference", status, initial, *said)
                continue
            jac = ctx.jacobian(v)
            frame = FRAME[direction]
            said = (jac, False, problem, cause)
            cell_rows.append(row(v, frame, status, (which, (VARIANT, v), None), *said))
            off_seams(v, frame, status, (VARIANT, v), *said)
        lost = [{**base, "owner": owner} for owner in missing]
        return cell_rows, non_cells, lost, absent, scorable

    cell_rows, non_cells, lost, _, scorable = one_set(OUTLINES)
    why = nuclei_reason(case, points)
    nuc_rows, nuc_non, absent = [], [], []
    if not scorable:
        # no outlines either (a landmark set): there is nothing the nuclei are missing from
        pass
    elif why:
        # a case without bare nuclei is scored on its outlines; nothing is imputed
        absent = [{**base, "method": "", "reason": why}]
    else:
        nuc_rows, nuc_non, nuc_lost, absent, _ = one_set(NUCLEI)
        # an owner without a seam mask on the bare nuclei of this case: no off-seam
        # nuclei row for ANY method here, and that has to be said somewhere
        absent += [
            {
                **base,
                "method": f"*{NON}{r['owner']}",
                "owner": r["owner"],
                "reason": NO_SEAM_MASK_ON_NUCLEI,
            }
            for r in nuc_lost
        ]
    recs = _records(ctx, list(wanted), pool, fresh, progress)

    def filled(rows):
        return [fill(recs) for fill in rows]

    nuclei = {
        "cells": filled(nuc_rows),
        "non_cells": filled(nuc_non),
        "missing": absent,
    }
    return (
        filled(cell_rows),
        res_rows,
        filled(non_cells),
        lost,
        nuclei,
    )


def _case_record(job, key, why, pool=None, fresh=False):
    """The rows of one case, computed now and saved at once as its record (``why`` is
    the reason there was none to read). Progress goes to stderr."""
    out, case = job[0], job[1]
    folder = records_dir(out, case)
    progress = _Progress(case)
    progress.say(f"started ({why})")
    got = _score_case(job, pool, fresh, progress)
    _write_record(folder / "case.rec", key, got)
    # the same key where a person can read it; `case.rec` is what the scorer trusts
    write_json(
        folder / "case.json",
        {
            "key_sha256": _key_sha256(key),
            "seconds": round(progress.elapsed(), 3),
            "rows": {
                "cells": len(got[0]),
                "nonband_cells": len(got[2]),
                "cells_nuclei": len(got[4]["cells"]),
                "nonband_cells_nuclei": len(got[4]["non_cells"]),
            },
            "seam_masks": {
                OUTLINES: sorted({r["owner"] for r in got[2]}),
                NUCLEI: sorted({r["owner"] for r in got[4]["non_cells"]}),
            },
            "key": key,
        },
    )
    progress.say(f"finished in {progress.elapsed():.1f} s; record saved")
    return got


def _case_task(task):
    """One whole case in a pool worker (serial inside): ``(index, rows)``."""
    i, job, key, why, fresh = task
    return i, _case_record(job, key, why, None, fresh)


def score_jobs(jobs, workers=1, fresh=False):
    """The rows of every job (``_score_case``), in the order of ``jobs``.

    A case whose record is valid (``case_key``) is read. The others are computed and
    saved as each finishes, so nothing depends on anything having scored a case
    before, and an interrupted scoring keeps what it finished. With ``workers`` above
    one the work runs at ONE level, never both: whole cases side by side when at least
    ``workers`` of them are to be computed, else one case after the other with its
    pair records side by side. One worker is purely serial: no process is started.
    ``fresh`` reads no record, of a case or of a pair, and replaces them all."""
    done = [None] * len(jobs)
    keys, todo = [], []
    for i, job in enumerate(jobs):
        keys.append(case_key(job))
        got, why = (
            (None, "asked to compute it again")
            if fresh
            else _read_record(records_dir(job[0], job[1]) / "case.rec", keys[i])
        )
        if got is None:
            todo.append((i, why))
        else:
            done[i] = got
            _Progress(job[1]).say(
                "rows read from its record (nothing it rests on has changed)"
            )
    if not todo:
        return done
    if workers <= 1:
        for i, why in todo:
            done[i] = _case_record(jobs[i], keys[i], why, None, fresh)
        return done
    import multiprocessing

    spawn = multiprocessing.get_context("spawn")
    if len(todo) >= workers:
        print(
            f"score: {len(todo)} case(s) to compute, {workers} at once",
            file=sys.stderr,
            flush=True,
        )
        tasks = [(i, jobs[i], keys[i], why, fresh) for i, why in todo]
        with spawn.Pool(workers) as pool:
            for i, got in pool.imap_unordered(_case_task, tasks):
                done[i] = got
        return done
    print(
        f"score: {len(todo)} case(s) to compute, one after the other, the pair "
        f"records of each in {workers} processes",
        file=sys.stderr,
        flush=True,
    )
    with spawn.Pool(workers) as pool:
        for i, why in todo:
            done[i] = _case_record(jobs[i], keys[i], why, pool, fresh)
    return done


def case_jobs(
    cases_root,
    out,
    dataset,
    dev=False,
    expected=None,
    owners=None,
    row_checks=None,
):
    """One job per case of a dataset, as ``score`` and ``score_case`` both make them
    (the two must agree, or a case's record would never be reused)."""
    owners = tuple(SEAM_OWNERS if owners is None else owners)
    cs = [
        c
        for c in list_cases(cases_root, dataset)
        if c.case_id.startswith("dev_") == dev
    ]
    if not cs:
        return []
    variants = find_variants(out, dataset, expected)
    return [(str(out), c, variants, owners, dict(row_checks or {})) for c in cs]


def refuse_slimmed(out, what="score"):
    """``SystemExit`` when ``out`` was slimmed to ``csv`` (``<out>/SLIMMED.txt``,
    ``regbench slim``, ``regbench.slimmed``): its maps (``warped.npz``) and records are
    gone, so a re-score would find every result missing and read as failures. A folder
    slimmed to ``images`` is scored as usual: what a score reads is kept."""
    from . import slimmed

    if slimmed.level(out) == "csv":
        raise SystemExit(
            f"{what}: refusing {out}: slimmed: csv ({Path(out) / slimmed.NAME}, `regbench "
            "slim --level csv`): its maps and records are removed, so it can no longer be "
            "scored. Its CSVs (tables/) are what is left; run the runs again to score them."
        )


def score_case(
    cases_root,
    out,
    dataset,
    index=None,
    case_id=None,
    workers=1,
    dev=False,
    expected=None,
    seam_owners=None,
    row_checks=None,
    fresh=False,
):
    """Score ONE case and leave its record under ``<out>/cache/scores``; no table is
    written. ``score`` then reads the record instead of computing the case, provided
    it is called with the same ``expected``, ``seam_owners``, ``row_checks`` and
    submission (``case_key``).

    The case is named by ``case_id`` or by its ``index`` among ALL prepared cases of
    the dataset (``list_cases``: the order every array of the launcher uses). Returns
    ``"scored"``, ``"kept"`` (its record was valid already) or ``"skipped"`` (a
    ``dev_*`` case when ``dev`` is not given, and the reverse)."""
    refuse_slimmed(out, "score-case")
    everything = list_cases(cases_root, dataset)
    if case_id is None:
        if index is None or not 0 <= index < len(everything):
            raise SystemExit(
                f"score-case: --index {index} is not one of the {len(everything)} "
                f"prepared {dataset} case(s) under {cases_root}"
            )
        case_id = everything[index].case_id
    jobs = [
        j
        for j in case_jobs(
            cases_root, out, dataset, dev, expected, seam_owners, row_checks
        )
        if str(j[1].case_id) == str(case_id)
    ]
    if not jobs:
        if any(str(c.case_id) == str(case_id) for c in everything):
            print(
                f"score: {dataset}/{case_id}: not scored here (a dev_* case is scored "
                "with --dev only, and only those are then)",
                file=sys.stderr,
            )
            return "skipped"
        raise SystemExit(f"score-case: no prepared {dataset} case named {case_id}")
    (job,) = jobs
    key = case_key(job)
    valid = _read_record(records_dir(out, job[1]) / "case.rec", key)[0] is not None
    score_jobs(jobs, workers, fresh)
    return "kept" if valid and not fresh else "scored"


def _in_set(rows, which):
    """The rows of a summary with the polygon set they were computed on (column ``set``)."""
    return [{**r, "set": which} for r in rows]


# ── summaries ─────────────────────────────────────────────────────────────────
def _values(rows, key):
    return np.array([_f(r.get(key)) for r in rows], dtype=float)


def _stat(values, fn):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(fn(v)) if v.size else math.nan


def _whole(values, fn):
    """``fn`` of the values when every one of them is a figure, else NaN: a sum over
    the rows that happen to have one is a sum over another set of rows."""
    v = np.asarray(values, dtype=float)
    return float(fn(v)) if v.size and np.isfinite(v).all() else math.nan


def _per_patient(rows, key):
    """``{independent unit: median over its rounds}`` (``UNIT_IS``)."""
    by = {}
    for r in rows:
        by.setdefault(_unit(r), []).append(_f(r.get(key)))
    return {g: _stat(v, np.median) for g, v in by.items()}


def _methods(rows):
    return list(dict.fromkeys(r["method"] for r in rows))


NOT_LAUNCHED = "not launched by this submission (it has no result on this dataset)"
NO_ROW = "it has no row on this dataset (not expected here)"


def compared_rows(rows, primary=None):
    """``(inside, outside)`` for the rows of ONE dataset: the rows the common block and
    the shared subsets are computed among, and ``{row: why it is not one of them}``.

    A row is compared when it is a primary row (every registration row when no primary
    is named) that actually RAN in what is being scored: at least one of its cases has
    a record. A run that was never launched (a submission of some of the runs) has
    every case ``not run``; counting it would empty the block for everyone. What a
    selfcheck said never takes a row out: the checks are reported, never a gate."""
    names = list(primary) if primary else [m for m in _methods(rows) if _is_run(m)]
    inside, outside = [], {}
    for name in names:
        mine = [r for r in rows if r["method"] == name]
        if not mine:
            outside[name] = NO_ROW
        elif all(r.get("cause") == "not_run" for r in mine):
            outside[name] = NOT_LAUNCHED
        else:
            inside.append(name)
    return inside, outside


def common_cases(rows, primary=None):
    """Case ids every compared row registered (``compared_rows``). A round lost for a
    cause that is not the method's is unregistered, so it is in no common block."""
    inside, _ = compared_rows(rows, primary)
    judged = [r for r in rows if r["method"] in inside]
    failed = {r["case_id"] for r in judged if r["imputed_initial"]}
    return {r["case_id"] for r in rows} - failed


def _shared(rows, inside, qualifies):
    """Case ids where EVERY compared row qualifies. Empty when nothing is compared."""
    if not inside:
        return set()
    cases = None
    for name in inside:
        mine = {
            r["case_id"]
            for r in rows
            if r["method"] == name and not _excluded(r) and qualifies(r)
        }
        cases = mine if cases is None else cases & mine
    return cases or set()


def _clears_null(r):
    return r["pair_fraction"] > r["pair_fraction_null_max"]


def _readable(r):
    return bool(r["displacement_readable"])


def common_block(rows, primary=None):
    """One row per (dataset, compared-or-not row): who is in the common block, and why
    the others are not. Written as ``common_block.csv`` and printed, always."""
    out = []
    for ds in sorted({r["dataset"] for r in rows}):
        of_ds = [r for r in rows if r["dataset"] == ds]
        inside, outside = compared_rows(of_ds, primary)
        common = common_cases(of_ds, primary)
        n_all = len({r["case_id"] for r in of_ds})
        for name in [*inside, *outside]:
            out.append(
                {
                    "dataset": ds,
                    "method": name,
                    "in_common_block": name in inside,
                    "why_not": outside.get(name, ""),
                    "n_cases_common": len(common) if inside else 0,
                    "n_cases": n_all,
                }
            )
    return out


def summarise_cells(rows, primary=None):
    """One row per (dataset, family, method, subset): ``all`` and ``common``.

    ``family`` is empty on the cohort (the column stays, so a table keeps its shape).
    ``n_patients`` counts independent units (``UNIT_IS``). A round lost for a cause that is not the
    method's is left out of that method's rows and counted (``n_excluded_not_method``).

    Two numbers are read on a subset of cases, and each is given twice. ``*_ranked`` /
    the readable displacement: on the cases where THIS method qualifies (its pair
    fraction clears its null's maximum / by ``READABLE_MARGIN``), with their count; a
    method that fails its hard cases is then read on its easy ones. ``*_shared``: on the
    cases where EVERY compared row qualifies (``compared_rows``), the same cases for
    every method: the only one of the two that may be set beside another method's.
    ``<headline>_n_nan`` counts the cases whose value does not exist (no pair): they
    cannot enter a median, and are shown instead of vanishing."""
    out = []
    for ds in sorted({r["dataset"] for r in rows}):
        of_ds = [r for r in rows if r["dataset"] == ds]
        inside, _ = compared_rows(of_ds, primary)
        common = common_cases(of_ds, primary)
        for fam in ("",):  # the cohort has one stratum; `family` stays an empty column
            of_fam = of_ds
            for subset in ("all", "common"):
                pool = of_fam
                if subset == "common":
                    if not inside:
                        continue  # nothing to be common to: said in the summary
                    pool = [r for r in of_fam if r["case_id"] in common]
                clear_all = _shared(pool, inside, _clears_null)
                readable_all = _shared(pool, inside, _readable)
                for m in _methods(of_fam):
                    every = [r for r in pool if r["method"] == m]
                    mine = [r for r in every if not _excluded(r)]
                    if not every:
                        continue
                    n_ok = sum(not r["imputed_initial"] for r in mine)
                    rec = {
                        "dataset": ds,
                        "family": fam,
                        "method": m,
                        "subset": subset,
                        "n_patients": len({_unit(r) for r in mine}),
                        "n_cases": len(mine),
                        "n_ok": n_ok,
                        "n_excluded_not_method": len(every) - len(mine),
                        "selfcheck_row": every[0].get("selfcheck_row", ""),
                        # cases scored on objects declared inexact, and those of
                        # them that are outlines with the rim taken off
                        "n_inexact_cases": sum(
                            bool(r.get("set_inexact")) for r in mine
                        ),
                        "n_shrunk_cases": sum(bool(r.get("set_shrunk")) for r in mine),
                    }
                    for h in HEADLINES:
                        per = list(_per_patient(mine, h).values())
                        rec[f"{h}_med"] = _stat(per, np.median)
                        # over the same units: where it leaves the median, some
                        # units are far from the others (a method failing on a few)
                        rec[f"{h}_mean"] = _stat(per, np.mean)
                        rec[f"{h}_min"] = _stat(per, np.min)
                        rec[f"{h}_max"] = _stat(per, np.max)
                        rec[f"{h}_min_case"] = _stat(_values(mine, h), np.min)
                        rec[f"{h}_n_nan"] = int(np.isnan(_values(mine, h)).sum())
                        if h in HAS_NULL:
                            null = list(_per_patient(mine, f"{h}_null").values())
                            rec[f"{h}_null_med"] = _stat(null, np.median)
                    # Dice where pair fraction clears the null's maximum: this method's
                    # own cases, then the cases every compared row clears
                    clear = [r for r in mine if _clears_null(r)]
                    per = list(_per_patient(clear, "dice_matched").values())
                    rec["dice_matched_ranked_med"] = _stat(per, np.median)
                    rec["n_dice_ranked"] = len(clear)
                    shared = [r for r in mine if r["case_id"] in clear_all]
                    per = list(_per_patient(shared, "dice_matched").values())
                    rec["dice_matched_shared_med"] = _stat(per, np.median)
                    rec["n_dice_shared"] = len(shared)
                    # displacement is read only on the readable cases
                    readable = [r for r in mine if _readable(r)]
                    per = list(_per_patient(readable, "displacement_um_p50").values())
                    rec["displacement_um_p50_med"] = _stat(per, np.median)
                    rec["n_readable"] = len(readable)
                    rec["readable_share"] = len(readable) / len(mine) if mine else 0.0
                    shared = [r for r in mine if r["case_id"] in readable_all]
                    per = list(_per_patient(shared, "displacement_um_p50").values())
                    rec["displacement_um_p50_shared_med"] = _stat(per, np.median)
                    rec["n_readable_shared"] = len(shared)
                    rec["unmapped_frac_max"] = _stat(
                        _values(mine, "unmapped_frac"), np.max
                    )
                    rec["fold_pct_max"] = _stat(_values(mine, "fold_pct"), np.max)
                    rec["coverage_med"] = _stat(_values(mine, "coverage"), np.median)
                    # how much of the pipeline's pairing was its fallback, and the
                    # exact assignment's numbers beside the headline (never ranked)
                    share = _values(mine, "fallback_frac")
                    rec["n_fallback_cases"] = int((share > 0).sum())
                    rec["fallback_frac_max"] = _stat(share, np.max)
                    for h in EXACT_HEADLINES:
                        per = list(_per_patient(mine, f"{h}_exact").values())
                        rec[f"{h}_exact_med"] = _stat(per, np.median)
                    rec["n_exact"] = int(
                        np.isfinite(_values(mine, "pair_fraction_exact")).sum()
                    )
                    out.append(rec)
    return out


def excluded_rounds(rows):
    """The rounds no summary of their method reads, named: one dict per (dataset,
    method, case, cause). Empty when ``NOT_THE_METHOD_ROUNDS`` is ``impute``."""
    return [
        {
            "dataset": r["dataset"],
            "method": r["method"],
            "case_id": r["case_id"],
            "group": r["group"],
            "cause": r["cause"],
        }
        for r in rows
        if _excluded(r)
    ]


def paired_vs_primary(rows, against=PAIRED_AGAINST):
    """Per independent unit and variant: each headline's unit median minus the
    ``against`` row's, on the cases BOTH have (a round either lost for a cause that is
    not the method's is in neither median)."""
    out = []
    for ds in sorted({r["dataset"] for r in rows}):
        of_ds = [r for r in rows if r["dataset"] == ds]
        for fam in ("",):  # the cohort has one stratum; `family` stays an empty column
            of_fam = of_ds
            ref = [r for r in of_fam if r["method"] == against and not _excluded(r)]
            if not ref:
                continue
            for m in _methods(of_fam):
                if m == against:
                    continue
                mine = [r for r in of_fam if r["method"] == m and not _excluded(r)]
                both = {r["case_id"] for r in mine} & {r["case_id"] for r in ref}
                mine = [r for r in mine if r["case_id"] in both]
                theirs = [r for r in ref if r["case_id"] in both]
                per = {h: _per_patient(mine, h) for h in HEADLINES}
                base = {h: _per_patient(theirs, h) for h in HEADLINES}
                for g in dict.fromkeys(_unit(r) for r in mine):
                    rec = {
                        "dataset": ds,
                        "family": fam,
                        "group": g,
                        "method": m,
                        "against": against,
                    }
                    rec["n_rounds"] = sum(_unit(r) == g for r in mine)
                    for h in HEADLINES:
                        rec[f"d_{h}"] = per[h][g] - base[h][g]
                    out.append(rec)
    return out


def bootstrap_interval(values, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """``(low, high)`` of the median of ``values`` (one per independent unit) by a
    percentile bootstrap: units are drawn with replacement ``n`` times. NaN with fewer
    than two units. With a handful of units the interval is coarse (the median of a
    resample can only take a few values) and narrower than the truth; it is a
    description of the spread between units, not a test."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = np.median(v[rng.integers(0, v.size, size=(n, v.size))], axis=1)
    tail = (1.0 - BOOTSTRAP_LEVEL) / 2.0 * 100.0
    low, high = np.percentile(draws, [tail, 100.0 - tail])
    return float(low), float(high)


def summarise_paired(paired):
    """Per (dataset, family, variant): the median over units of each paired difference
    against ``PAIRED_AGAINST``, with a bootstrap interval over units. The rounds of a
    unit stay together (they are already one median per unit), so the interval is over
    what is independent. No p-value and no ranking are derived from it."""
    out = []
    keys = dict.fromkeys((r["dataset"], r["family"], r["method"]) for r in paired)
    for ds, fam, m in keys:
        mine = [
            r
            for r in paired
            if (r["dataset"], r["family"], r["method"]) == (ds, fam, m)
        ]
        rec = {
            "dataset": ds,
            "family": fam,
            "method": m,
            "against": mine[0]["against"],
            "n_patients": len(mine),
        }
        for h in HEADLINES:
            d = _values(mine, f"d_{h}")
            low, high = bootstrap_interval(d)
            rec[f"d_{h}_med"] = _stat(d, np.median)
            rec[f"d_{h}_mean"] = _stat(d, np.mean)
            rec[f"d_{h}_lo"] = low
            rec[f"d_{h}_hi"] = high
            rec[f"d_{h}_n_nan"] = int(np.isnan(d).sum())
        out.append(rec)
    return out


def failures(res_rows):
    """Per variant: launches made and failed, by cause, and rounds lost.

    A per-patient method loses every round of a patient in one failed launch, so the two
    counts differ. A launch is failed when any of its rounds is; its cause is the first
    failed round's.
    """
    out = []
    for ds in sorted({r["dataset"] for r in res_rows}):
        of_ds = [r for r in res_rows if r["dataset"] == ds]
        for m in _methods(of_ds):
            mine = [r for r in of_ds if r["method"] == m]
            launches, failed = {}, {}
            for r in mine:
                if not r["launch_id"]:
                    continue
                launches[r["launch_id"]] = True
                if not r["ok"]:
                    failed.setdefault(r["launch_id"], r["cause"] or "exception")
            rec = {
                "dataset": ds,
                "method": m,
                "n_cases": len(mine),
                "launches": len(launches),
                "launches_failed": len(failed),
            }
            for cause in CAUSES[:-1]:
                rec[f"launches_failed_{cause}"] = sum(
                    c == cause for c in failed.values()
                )
            # apart: what the method did, and what was done to it
            rec["launches_failed_method"] = sum(
                c not in NOT_THE_METHOD for c in failed.values()
            )
            rec["launches_lost_not_method"] = sum(
                c in NOT_THE_METHOD for c in failed.values()
            )
            rec["rounds_not_run"] = sum(r["cause"] == "not_run" for r in mine)
            rec["rounds_lost"] = sum(not r["ok"] for r in mine)
            rec["rounds_lost_not_method"] = sum(
                (not r["ok"]) and r["cause"] in NOT_THE_METHOD for r in mine
            )
            rec.update(run_failure_columns(mine))
            out.append(rec)
    return out


def _owner_column(cause):
    """Which failure column a lost round is counted in, by ``failure_log.WHOSE``: the
    method's, the benchmark's, or "not run / lost" (the launcher's, the
    infrastructure's, a result nobody ran). A cause the table does not know is the
    method's, as ``failures`` already treats it."""
    if cause == "not_run":
        return "n_not_run_or_lost"
    whose = failure_log.WHOSE.get(cause, failure_log.METHOD)
    return {
        failure_log.METHOD: "n_failed_method",
        failure_log.BENCHMARK: "n_failed_benchmark",
    }.get(whose, "n_not_run_or_lost")


def run_failure_columns(mine):
    """One run's failures, by owner and cause, from its result rows (one per round).

    A round that did not register is counted once, by who is responsible: the method
    (``n_failed_method``), the benchmark (``n_failed_benchmark``), or no one's result
    (``n_not_run_or_lost``: launcher, infrastructure, ``not_run``). ``top_cause`` is
    the most frequent of those causes, a tie going to the earlier in ``CAUSES``.
    ``n_fallback_stage_positions`` and ``n_cycle_fully_modelled`` count rounds whose
    ASHLAR flag is true; a row that carries no flag (a failed one) counts 0."""
    rec = dict.fromkeys(
        ("n_failed_method", "n_failed_benchmark", "n_not_run_or_lost"), 0
    )
    causes = {}
    for r in mine:
        if r["ok"]:
            continue
        cause = r.get("cause") or "exception"
        rec[_owner_column(cause)] += 1
        causes[cause] = causes.get(cause, 0) + 1
    rank = {c: i for i, c in enumerate(CAUSES)}
    rec["top_cause"] = (
        min(causes, key=lambda c: (-causes[c], rank.get(c, len(rank)), c))
        if causes
        else ""
    )
    rec["n_fallback_stage_positions"] = sum(
        bool(r.get("fallback_stage_positions")) for r in mine
    )
    rec["n_cycle_fully_modelled"] = sum(
        bool(r.get("cycle_fully_modelled")) for r in mine
    )
    return rec


def run_failures(res_rows):
    """``run_failure_columns`` per (dataset, family, method): the counts beside a
    summary row, over exactly the cases that row summarises (the whole dataset).
    ``is_ashlar``
    says whether the cells ``fallback`` / ``fully modelled`` apply; it is read by the
    table only and is in no CSV."""
    out = {}
    keys = dict.fromkeys((r["dataset"], "", r["method"]) for r in res_rows)
    for ds, fam, m in keys:
        mine = [r for r in res_rows if (r["dataset"], "", r["method"]) == (ds, fam, m)]
        out[(ds, fam, m)] = {
            **run_failure_columns(mine),
            "is_ashlar": any(
                "ashlar" in str(r.get("method", ""))
                or r.get("fallback_stage_positions") is not None
                for r in mine
            ),
        }
    return out


FAILURE_COLUMN_RULE = (
    "`failed (method)`, `failed (benchmark)` and `not run / lost` count CASES "
    "(rounds) of the run that did not register, by whose cause it was "
    "(`failure_log.WHOSE`): the method's; the benchmark's; or the launcher's, the "
    "infrastructure's or a result nobody ran. They count the same cases as "
    "`ok/cases` beside them, including the rounds "
    "`ok/cases` leaves out as not the method's. `top cause` is the most frequent "
    "of them. `fallback` and `fully modelled` (ASHLAR runs only) count rounds whose "
    "reference stitch kept the stage positions, and whose cycle was entirely replaced "
    "by the linear model. The cells are blank on an off-seam view row (its failures are "
    "its run's own row's) and in the table of cases every row registered (the counts "
    "are of all cases).\n"
)


def cpu_instruments_line(res_summary):
    """The CPU counters behind a cost table, for ``summary.md``; the one that can miss
    CPU time is said to."""
    from . import resources

    seen = sorted(
        {
            part
            for r in res_summary
            for part in str(r.get("cpu_instrument") or "").split("; ")
            if part
        }
    )
    if not seen:
        return "no row of this table recorded one"
    line = "here: " + ", ".join(f"`{name}`" for name in seen)
    if resources.POLLED in seen:
        line += (
            f"; a `{resources.POLLED}` row sampled its process tree, misses the CPU "
            "of a child that ends between two samples, and its `cpu_h_used` is a "
            "LOWER BOUND"
        )
    return line


def summarise_resources(res_rows):
    """Per (dataset, patient, method). A patient's cost is the sum over its rounds for a
    pair method and the launch's own total for a per-patient one; the peak is the maximum
    and is never divided by rounds. Only completed runs are a cost. ``wasted_*`` is the
    earlier attempts of the ladder plus a last attempt that failed outright, and
    ``lost_*`` the attempts among them that ended for a cause that is not the method's.

    ``write_s`` and ``total_to_image_s`` are the patient's totals for both units. A
    patient is ``complete`` when every round of it registered; ``summarise_cost`` reads
    only complete patients, because the cost of some of a patient's rounds is not the
    cost of the patient."""
    out = []
    keys = dict.fromkeys((r["dataset"], r["group"], r["method"]) for r in res_rows)
    for ds, g, m in keys:
        mine = [
            r for r in res_rows if (r["dataset"], r["group"], r["method"]) == (ds, g, m)
        ]
        ok = [r for r in mine if r["ok"]]
        patient = any(r["unit"] == "patient" for r in mine)
        total = np.max if patient else np.sum
        rec = {
            "dataset": ds,
            "group": g,
            "method": m,
            "unit": "patient" if patient else "pair",
            "n_rounds": len(mine),
            "n_ok": len(ok),
            "complete": len(ok) == len(mine),
            # False when no round of it has a record (a run that was not launched)
            "launched": any(r["cause"] != "not_run" for r in mine),
            "attempt": _stat(_values(mine, "attempt"), np.max),
            "mem_requested_gb": _stat(_values(mine, "mem_requested_gb"), np.max),
            "cpus_requested": _stat(_values(mine, "cpus_requested"), np.max),
        }
        for key in ("register_s", "cpu_h", "gpu_h"):
            rec[key] = _stat(_values(ok, key), total)
        # regbench.billing, per patient. Good: over the rounds that registered (a
        # round whose writer failed registered: its registration is a cost like any
        # other), and blank when one of them lacks the figure, so that every column
        # of a patient is over the same rounds. Wasted, and lost: every such attempt
        # of the patient; a failed round inside a per-patient launch that otherwise
        # finished shares that launch's clock and is no attempt of its own, so there
        # they are the registered rounds' earlier ones. Total: good + wasted.
        rec["device"] = (
            (
                billing.GPU
                if any(r.get("device") == billing.GPU for r in ok)
                else billing.CPU
            )
            if ok
            else ""
        )
        reg = [r for r in mine if r["ok"] or r.get("registered")]
        spent = reg if patient and reg else mine
        for stem, _ in billing.SUMS:
            good = _whole(_values(reg, stem), total)
            failed = _stat(_values(spent, f"wasted_{stem}"), total)
            rec[stem] = good
            rec[f"wasted_{stem}"] = failed
            both = [v for v in (good, failed) if v == v]
            rec[f"total_{stem}"] = (
                float(sum(both)) if both and not (reg and good != good) else math.nan
            )
            rec[f"lost_{stem}"] = _stat(_values(spent, f"lost_{stem}"), total)
        rec["cpu_h_scheduler"] = _whole(_values(reg, "cpu_h_scheduler"), total)
        rec["cpu_h_used_vs_scheduler"] = billing.ratio(
            rec["cpu_h_used"], rec["cpu_h_scheduler"]
        )
        # the moving images of the rounds that registered, together
        rec["moving_gpx"] = _stat(_values(ok, "moving_gpx"), np.sum)
        rec["cpu_h_used_per_gpx"] = billing.ratio(rec["cpu_h_used"], rec["moving_gpx"])
        for key in (
            "write_cpu_h_used",
            "write_cpu_h_reserved",
            "write_wasted_cpu_h_reserved",
        ):
            rec[key] = _stat(_values(ok, key), np.sum)
        # a registration in several scheduler jobs per round: the patient's sums over
        # its rounds, and the most CPUs one round had at once (blank elsewhere)
        split = [r for r in ok if r.get("n_jobs") not in ("", None)]
        for key in (
            "elapsed_with_queue_s",
            "wall_unlimited_parallel_s",
            "cpu_h_allocated",
            "n_jobs",
            "n_tiles",
            "n_job_retries",
            "write_cpu_h_allocated",
            "unmeasured_attempts",
        ):
            rec[key] = _stat(_values(split, key), np.sum) if split else ""
        for key in ("peak_cpus_at_once", "write_peak_gb"):
            rec[key] = _stat(_values(split, key), np.max) if split else ""
        # The writer's time is the SUM over the rounds for both units: a per-patient
        # method records each round's share of one writer call (or each slide's own
        # time), a pair method each round's own. The time to an image is the
        # registration plus that sum. (Until 2026-10-11 a per-patient launch reported
        # the largest single round: one share of the writer, not the whole.)
        rec["write_s"] = _stat(_values(ok, "write_s"), np.sum)
        rec["total_to_image_s"] = (
            float(np.nansum([rec["register_s"], rec["write_s"]])) if ok else math.nan
        )
        # used over reserved of the patient; the median of the rounds' own where a
        # row carries no billing (rows made by hand)
        rec["cpu_efficiency"] = billing.ratio(rec["cpu_h_used"], rec["cpu_h_reserved"])
        if rec["cpu_efficiency"] != rec["cpu_efficiency"]:
            rec["cpu_efficiency"] = _stat(_values(ok, "cpu_efficiency"), np.median)
        for key in ("peak_rss_gb", "peak_rss_after_register_gb", "cgroup_peak_gb"):
            rec[key] = _stat(_values(ok, key), np.max)
        for key in ("jvm_heap_asked_gb", "jvm_max_heap_gb"):  # blank without a JVM
            rec[key] = _blank(_stat(_values(ok, key), np.max))
        rec["gpu_model"] = "; ".join(sorted({r["gpu_model"] for r in ok} - {""}))
        # the counter(s) the patient's CPU time was read from (regbench.resources)
        rec["cpu_instrument"] = "; ".join(
            sorted({str(r.get("cpu_instrument") or "") for r in ok} - {""})
        )
        # A failed round inside a per-patient launch that otherwise finished is not a
        # wasted attempt; a launch that produced nothing is.
        lost = [
            r
            for r in mine
            if not r["ok"]
            and r["cause"] not in ("not_run", *NOT_THE_METHOD)
            and not r.get("registered")
        ]
        if patient and ok:
            lost = []
        last = {
            "wasted_attempts": (1 if patient else len(lost)) if lost else 0,
            "wasted_cpu_h": _stat(_values(lost, "cpu_h"), total),
            "wasted_wall_s": _stat(_values(lost, "register_s"), total),
        }
        for key, extra in last.items():
            earlier = _stat(_values(mine, key), total)
            rec[key] = float(np.nansum([earlier, extra]))
        for key in ("wasted_from_scheduler", "wasted_unknown", "lost_attempts"):
            rec[key] = _stat(_values(mine, key), total)
        out.append(rec)
    return out


# CPU-hours first (regbench.billing): the cost that can be set beside another row's
COST_KEYS = (
    "cpu_h_used",
    "cpu_h_reserved",
    "cpu_efficiency",
    "gb_h_reserved",
    "total_cpu_h_used",
    "total_cpu_h_reserved",
    "total_gb_h_reserved",
    "cpu_h_used_per_gpx",
    "gpu_h",
    "cpu_h_scheduler",
    "cpu_h_used_vs_scheduler",
    "register_s",
    "write_s",
    "total_to_image_s",
    "peak_rss_gb",
    "cgroup_peak_gb",
)
# The two times of a row whose registration ran as several scheduler jobs per round
# are not a single process's and are not called like one: each has a name of its own
# in `cost_summary.csv` and in `summary.md`, and the plain column is blank there.
PARALLEL_TIMES = {
    "register_s": "time_unlimited_parallel_tiles_s",
    "total_to_image_s": "to_image_unlimited_parallel_tiles_s",
}
# costs a patient can lack while it is complete (no stage record read, nothing in the
# accounting): beside each of their medians, the number of patients it is over
COUNTED_KEYS = (
    "gb_h_reserved",
    "total_gb_h_reserved",
    "cpu_h_scheduler",
    "cpu_h_used_vs_scheduler",
)
VALID_RATIOS = (
    "Valid between rows: CPU-hours (`cpu_h_used`, `cpu_h_reserved`, `total_*`), read "
    "on the common patients, with `device` beside them. NOT valid: the time with "
    "unlimited parallel tiles of a row in several jobs against the time of a single "
    "process; a `gpu` row's CPU-hours or time against a `cpu` row's as if they had "
    "the same resources; and totals that end at different deliverables "
    "(`merged_image` in the cost-by-step block).\n"
)


def common_patients(res_summary, primary=None):
    """``{dataset: (patients complete for every compared row, [compared rows])}``. A
    row is compared as in the cell tables: a primary row (any run when none is named)
    with at least one result that is not ``not run``."""
    out = {}
    for ds in dict.fromkeys(r["dataset"] for r in res_summary):
        of_ds = [r for r in res_summary if r["dataset"] == ds]
        names = list(primary) if primary else _methods(of_ds)
        inside = [
            m for m in names if any(r["method"] == m and r["launched"] for r in of_ds)
        ]
        patients = {r["group"] for r in of_ds}
        for m in inside:
            patients &= {
                r["group"] for r in of_ds if r["method"] == m and r["complete"]
            }
        out[ds] = (patients if inside else set(), inside)
    return out


def summarise_cost(res_summary, primary=None):
    """Per (dataset, method): the median over patients of each cost, TWICE. Over the
    patients this method completed (``n_patients_complete`` of ``n_patients``): a
    method that fails on the largest patients is then read on the small ones and looks
    cheaper. And over the patients EVERY compared row completed
    (``n_patients_common``, ``*_common``): the same patients for every method, the only
    numbers that may be set beside each other."""
    common = common_patients(res_summary, primary)
    out = []
    for ds, m in dict.fromkeys((r["dataset"], r["method"]) for r in res_summary):
        mine = [r for r in res_summary if (r["dataset"], r["method"]) == (ds, m)]
        done = [r for r in mine if r["complete"]]
        shared = [r for r in done if r["group"] in common[ds][0]]
        rec = {
            "dataset": ds,
            "method": m,
            "unit": mine[0]["unit"],
            "n_patients": len(mine),
            "n_patients_complete": len(done),
            "n_patients_partial": sum(0 < r["n_ok"] < r["n_rounds"] for r in mine),
            "n_patients_common": len(shared),
            "common_among": ", ".join(common[ds][1]),
        }
        # in several scheduler jobs per round: its times are another quantity
        split = any(r.get("n_jobs") not in ("", None) for r in done)
        rec["device"] = "; ".join(sorted({r.get("device") or "" for r in done} - {""}))
        rec["gpu_model"] = "; ".join(
            sorted({r.get("gpu_model") or "" for r in done} - {""})
        )
        rec["peak_cpus_at_once"] = (
            _stat(_values(done, "peak_cpus_at_once"), np.max) if split else ""
        )
        for key in COST_KEYS:
            name = PARALLEL_TIMES[key] if split and key in PARALLEL_TIMES else key
            rec[f"{name}_med"] = _stat(_values(done, key), np.median)
            rec[f"{name}_mean"] = _stat(_values(done, key), np.mean)
            rec[f"{name}_common_med"] = _stat(_values(shared, key), np.median)
            rec[f"{name}_common_mean"] = _stat(_values(shared, key), np.mean)
            if key in COUNTED_KEYS:
                rec[f"{name}_n"] = int(np.isfinite(_values(done, key)).sum())
                rec[f"{name}_common_n"] = int(np.isfinite(_values(shared, key)).sum())
        out.append(rec)
    return out


# ── output ────────────────────────────────────────────────────────────────────
def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def _fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.4g}"
    return str(v)


def markdown(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += [
        "| " + " | ".join(_fmt(r.get(c, "")) for c in columns) + " |" for r in rows
    ]
    return "\n".join(lines)


def _check_text(value):
    """A row's own selfcheck for a table: loud where it was never checked."""
    if value == NEVER_CHECKED:
        return "NEVER CHECKED (no image of this stage)"
    return value or "-"


def _summary_table(rows, primary, run_fails=None):
    """Markdown rows: each headline as ``median (mean) [minimum over cases]`` with
    ``ok/cases`` on the same line, its null beside it. With ``run_fails``
    (``run_failures``), each row also carries its run's failures beside its scores: ``failed (method)``,
    ``failed (benchmark)``, ``not run / lost`` (rounds), ``top cause``, and for an
    ASHLAR run ``fallback`` and ``fully modelled`` (rounds; empty on other rows)."""

    order = [p.split(NON)[0] for p in primary or []]
    families = list(dict.fromkeys(r.get("family", "") for r in rows))
    rows = sorted(
        rows,
        key=lambda r: (
            families.index(r.get("family", "")),
            order.index(r["method"].split(NON)[0])
            if r["method"].split(NON)[0] in order
            else len(order),
        ),
    )
    table = []
    for r in rows:
        line = {"family": r.get("family", "")} if any(families) else {}
        line.update(
            {
                "method": r["method"],
                "units": r["n_patients"],
                "selfcheck of this row": _check_text(r.get("selfcheck_row", "")),
                "ok/cases": f"{r['n_ok']}/{r['n_cases']}",
                "left out (not the method's)": r.get("n_excluded_not_method", 0),
            }
        )
        for h in HEADLINES:
            line[h] = (
                f"{_fmt(r[f'{h}_med'])} ({_fmt(r.get(f'{h}_mean', math.nan))}) "
                f"[{_fmt(r[f'{h}_min_case'])}]"
            )
            if h in HAS_NULL:
                line[f"{h}_null"] = r[f"{h}_null_med"]
        line["dice_shared (n)"] = (
            f"{_fmt(r['dice_matched_shared_med'])} ({r['n_dice_shared']})"
        )
        line["dice_own_cases (n)"] = (
            f"{_fmt(r['dice_matched_ranked_med'])} ({r['n_dice_ranked']})"
        )
        line["dice NaN cases"] = r["dice_matched_n_nan"]
        line["displacement_um_p50 shared (n)"] = (
            f"{_fmt(r['displacement_um_p50_shared_med'])} ({r['n_readable_shared']})"
        )
        line["displacement_um_p50 own cases (n)"] = (
            f"{_fmt(r['displacement_um_p50_med'])} ({r['n_readable']})"
        )
        line["unmapped_frac_max"] = r["unmapped_frac_max"]
        line["fold_pct_max"] = r["fold_pct_max"]
        line["fallback cases (largest share of cells)"] = (
            f"{r.get('n_fallback_cases', 0)} ({_fmt(r.get('fallback_frac_max', 0.0))})"
        )
        for h in EXACT_HEADLINES:
            line[f"{h}_exact"] = r.get(f"{h}_exact_med", math.nan)
        line["cases with exact numbers"] = f"{r.get('n_exact', 0)}/{r['n_cases']}"
        line["cases on inexact objects (shrunk outlines)"] = (
            f"{r.get('n_inexact_cases', 0)} ({r.get('n_shrunk_cases', 0)})"
        )
        if run_fails is not None:
            # the counts are of all cases of the run: blank beside a subset, and on a
            # `:non_` view (its failures are the base run's own row's)
            own = r.get("subset", "all") == "all" and NON not in r["method"]
            f = run_fails.get((r["dataset"], r.get("family", ""), r["method"]), {})
            f = f if own else {}
            ashlar = bool(f.get("is_ashlar"))
            line["failed (method)"] = f.get("n_failed_method", "")
            line["failed (benchmark)"] = f.get("n_failed_benchmark", "")
            line["not run / lost"] = f.get("n_not_run_or_lost", "")
            line["top cause"] = f.get("top_cause", "")
            line["fallback"] = f["n_fallback_stage_positions"] if ashlar else ""
            line["fully modelled"] = f["n_cycle_fully_modelled"] if ashlar else ""
        table.append(line)
    return markdown(table, list(table[0])) if table else ""


SUBSET_RULE = (
    "`dice_shared` and the shared displacement are read on the cases where EVERY "
    "compared row qualifies (pair fraction above its null's maximum; for displacement "
    "by the readable margin): the same cases for every row, with their count. Only "
    "these may be set beside another row's. `own cases` is the same number on the "
    "cases where that row alone qualifies: a row that fails its hard cases is read on "
    "its easy ones there. `dice NaN cases` are cases with no pair at all: they have no "
    "Dice and are in no median.\n"
)


PAIRING_RULE = (
    "Every number above is the pipeline's pairing (`bin/utils/cell_pairs.py`): an "
    "assignment per connected component of the candidate pairs, and mutual nearest "
    "neighbour in a component of more than "
    f"{pairs.cp.DEFAULT_MAX_COMPONENT_CELLS} cells. `fallback cases` counts the cases "
    "where some component was paired that way, with the largest share of a case's "
    "cells that sat in such a component. The `*_exact` columns are the same numbers "
    "from the assignment solved exactly in every component (most pairs first, then "
    "least total distance): extra numbers, in no ranking, no shared subset and no "
    "difference. `cases with exact numbers` are those where it was solved; a case "
    "with a component above the solver's bound has none (`exact_status` in "
    "`cells.csv` says why).\n"
)


SETS_RULE = (
    "Every overlap number exists on two polygon sets. The tables below are the "
    "**outlines** (the headline: on the cohort, each StarDist nucleus grown by 10 px, "
    "what the pipeline's QC publishes and the arms benchmark scores). The **bare "
    "nuclei** have the same tables in `<name>_nuclei.csv`, with their own gate (1.5 x "
    "the median equivalent radius of the nuclei), their own null and their own shared "
    "subset; 'Outlines and bare nuclei, side by side' sets the two beside each other "
    "for the primary rows. Neither set depends on any method.\n"
)
# the headline numbers set side by side, as (label, summary key)
SIDE_BY_SIDE = (
    ("pair_fraction", "pair_fraction_med"),
    ("pair_fraction_null", "pair_fraction_null_med"),
    ("pair_fraction_excess", "pair_fraction_excess_med"),
    ("dice_matched", "dice_matched_med"),
    ("dice_shared", "dice_matched_shared_med"),
    ("within_1um", "within_1um_med"),
    ("displacement_um_p50 shared", "displacement_um_p50_shared_med"),
)


def _nuclei_missing_lines(missing, n_cases, any_scored):
    """What the summary says of the cases (and results) without bare nuclei."""
    if not missing:
        return []
    cases = {(r["dataset"], r["case_id"]) for r in missing if not r["method"]}
    seams = [r for r in missing if r.get("owner")]
    results = [r for r in missing if r["method"] and not r.get("owner")]
    lines = []
    if seams:
        owners = sorted({r["owner"] for r in seams})
        lines.append(
            f"**{len(seams)} case-and-owner pair(s) have no off-seam rows on the bare "
            f"nuclei** (owner(s): {', '.join(f'`{o}`' for o in owners)}): the owner's "
            "primary run left no seam mask on those nuclei, so `nonband_cells_nuclei."
            "csv` has no row for any method there (`nuclei_missing.csv`, `method` = "
            "`*:non_<owner>`).\n"
        )
    if cases:
        reasons = {}
        for r in missing:
            if not r["method"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        lines.append(
            f"**{len(cases)} of {n_cases} case(s) have no bare nuclei** and are scored "
            "on the outlines alone; their nuclei numbers are absent, never imputed "
            "(`nuclei_missing.csv`): "
            + "; ".join(f"{n} x {why}" for why, n in sorted(reasons.items())[:5])
            + ("; ..." if len(reasons) > 5 else "")
            + ("." if any_scored else ". No nuclei table is written.")
            + "\n"
        )
    if results:
        names = sorted({r["method"] for r in results})
        lines.append(
            f"**{len(results)} result(s) hold no bare nuclei** (made before their case "
            f"had them: {', '.join(f'`{n}`' for n in names)}); they have outline rows "
            "and no nuclei row until they are run again (`nuclei_missing.csv`).\n"
        )
    return lines


def _side_by_side(block, nuclei_summary, primary):
    """The headline numbers of each primary row on both sets: ``outlines / nuclei``."""
    if not nuclei_summary:
        return []

    def key(r):
        return r["dataset"], r.get("family", ""), r["method"], r["subset"]

    def show(v):
        nothing = v is None or (isinstance(v, float) and math.isnan(v))
        return "-" if nothing else _fmt(v)

    by = {key(r): r for r in nuclei_summary}
    wanted = [p.split(NON)[0] for p in primary or []]
    table = []
    for r in block:
        if wanted and r["method"] not in wanted:
            continue
        n = by.get(key(r))
        line = {"family": r["family"]} if r.get("family") else {}
        line["method"] = r["method"]
        line["cases: outlines / nuclei"] = (
            f"{r['n_cases']} / {n['n_cases'] if n else 0}"
        )
        for label, k in SIDE_BY_SIDE:
            line[f"{label}: outlines / nuclei"] = (
                f"{show(r.get(k))} / {show(n.get(k) if n else None)}"
            )
        line["n_dice_shared: outlines / nuclei"] = (
            f"{show(r.get('n_dice_shared'))} / "
            f"{show(n.get('n_dice_shared') if n else None)}"
        )
        table.append(line)
    if not table:
        return []
    return [
        "#### Outlines and bare nuclei, side by side\n",
        "Each cell is `outlines / nuclei`: the median over units of the same number "
        "on the two sets. Matched Dice on the outlines is dominated by the 10 px rim "
        "and is clipped where cells touch; on the nuclei it is stricter per "
        "micrometre. `dice_shared` is read on each set's own shared subset.\n",
        markdown(table, list(table[0])),
        "",
    ]


def _common_lines(block, ds):
    """Who is in the common block of one dataset and who is not, always said."""
    mine = [r for r in block if r["dataset"] == ds]
    inside = [r["method"] for r in mine if r["in_common_block"]]
    outside = [r for r in mine if not r["in_common_block"]]
    if not mine:
        return []
    n = f"{mine[0]['n_cases_common']} of {mine[0]['n_cases']} case(s)"
    if inside:
        text = (
            "Compared rows (they ran in what is scored here): "
            + ", ".join(f"`{m}`" for m in inside)
            + f". Cases every one of them registered: {n}."
        )
    else:
        text = "No compared row ran in what is scored here: there is no common block."
    if outside:
        text += (
            " NOT among them: "
            + "; ".join(f"`{r['method']}` ({r['why_not']})" for r in outside)
            + "."
        )
    return [text + "\n"]


def _paired_markdown(summary, ds):
    """Differences against ``PAIRED_AGAINST`` with their bootstrap interval."""
    mine = [r for r in summary if r["dataset"] == ds and _is_run(r["method"])]
    if not mine:
        return []
    families = any(r["family"] for r in mine)
    table = []
    for r in mine:
        line = {"family": r["family"]} if families else {}
        line.update({"method": r["method"], "units": r["n_patients"]})
        for h in HEADLINES:
            interval = (
                "no interval: one unit"
                if math.isnan(r[f"d_{h}_lo"])
                else f"{_fmt(r[f'd_{h}_lo'])}, {_fmt(r[f'd_{h}_hi'])}"
            )
            line[f"d_{h}"] = (
                f"{_fmt(r[f'd_{h}_med'])} ({_fmt(r.get(f'd_{h}_mean', math.nan))}) "
                f"[{interval}]"
            )
        table.append(line)
    return [
        f"### Differences against `{mine[0]['against']}` (paired within unit)\n",
        "Each cell is `median (mean) [interval]` over units of (this row minus "
        f"`{mine[0]['against']}`), each unit's rounds first reduced to their median on "
        "the cases both rows have; `units` is how many have both. The interval is a "
        f"{BOOTSTRAP_LEVEL:.0%} percentile bootstrap OVER UNITS "
        f"({BOOTSTRAP_N} resamples of the units, seed {BOOTSTRAP_SEED}; a unit's "
        "rounds stay together), of the median. It is narrower than the truth: a "
        "resample of a handful of units can only take a few values, and the units "
        "that were measured stand in for all units. It describes the spread between "
        f"the units and is not a test. {DESCRIPTIVE.capitalize()}.\n",
        markdown(table, list(table[0])),
        "",
    ]


def summarise_nonband(rows, primary=None):
    """One row per (dataset, owner, variant): the summary of ``summarise_cells`` on the
    off-seam rows, with the nuclei kept. Never merged with the primary summary.

    ``within_ceiling`` is what the mask alone allows ``within_*um`` to reach for ANY
    method: the curve is the smaller of the two one-sided shares, and a mask that
    removes nuclei on one side only leaves at most (fewer kept nuclei) / (more kept
    nuclei) of the larger side with a partner, one partner per nucleus."""
    out = []
    for owner in dict.fromkeys(r["owner"] for r in rows):
        mine = [r for r in rows if r["owner"] == owner]
        names = [f"{p}{NON}{owner}" for p in primary] if primary else None
        for rec in summarise_cells(mine, names):
            if rec["subset"] != "all":
                continue
            of = [
                r
                for r in mine
                if r["method"] == rec["method"] and r["dataset"] == rec["dataset"]
            ]
            rec = {"owner": owner, **rec}
            del rec["subset"]
            for key in ("n_mov_kept", "n_mov_all", "n_ref_kept", "n_ref_all"):
                rec[key] = int(sum(r[key] for r in of))
            kept = sorted((rec["n_mov_kept"], rec["n_ref_kept"]))
            rec["within_ceiling"] = kept[0] / kept[1] if kept[1] else math.nan
            out.append(rec)
    return out


def nonband_missing(lost, jobs, owners, variants_of):
    """Per (dataset, owner) with a row on that dataset: the cases, and how many of them
    have no off-seam rows because the owner's primary run left no mask there."""
    out = []
    for ds in dict.fromkeys(job[1].dataset for job in jobs):
        n_cases = sum(job[1].dataset == ds for job in jobs)
        for owner in owners:
            if owner not in variants_of.get(ds, ()):
                continue
            n = sum(r["dataset"] == ds and r["owner"] == owner for r in lost)
            out.append(
                {
                    "dataset": ds,
                    "owner": owner,
                    "n_cases": n_cases,
                    "n_cases_without_mask": n,
                }
            )
    return out


def _nonband_markdown(summary, missing):
    """The off-seam block: apart from every other table, and saying so."""
    if not missing:
        return []
    parts = [
        "## Off the seams (a block of its own; not ranked)\n",
        "Every variant scored a second time WITHOUT the nuclei that straddle one "
        "method's joins (`<variant>:non_<owner>`): all rows of a block use the same "
        "nuclei, chosen by where the owner put its tiles or blocks and by no score. "
        "These numbers are not "
        "comparable with the full-slide tables above and enter no ranking. A case the "
        "owner did not register has no row here for any method.\n",
    ]
    for m in missing:
        ds, owner = m["dataset"], m["owner"]
        block = [r for r in summary if r["dataset"] == ds and r["owner"] == owner]
        parts.append(f"### {ds}: off `{owner}`'s seams\n")
        kept = (
            f"{block[0]['n_mov_kept']} of {block[0]['n_mov_all']} moving and "
            f"{block[0]['n_ref_kept']} of {block[0]['n_ref_all']} reference nuclei "
            "kept, the same for every row. The mask alone caps `within_*um` at about "
            f"{_fmt(block[0]['within_ceiling'])} for every row here (the fewer kept "
            "nuclei over the more, one partner per nucleus); pair fraction is over "
            "the smaller kept set and has no such cap. "
            if block
            else ""
        )
        parts.append(
            f"{kept}{m['n_cases_without_mask']} of {m['n_cases']} case(s) have no "
            f"mask (`{owner}` did not register them) and no row.\n"
        )
        if block:
            parts += [_summary_table(block, None), ""]
    return parts


ATTRITION_COLUMNS = (
    "dataset",
    "method",
    "manifest_rounds",
    "prepared",
    "with_outlines",
    "with_nuclei",
    "run",
    "registered",
    "scored",
    "scored_nuclei",
)


def manifest_rounds(path):
    """How many rounds a ``prepare multiplex`` manifest lists (rows with a
    ``case_id``); None without a manifest or when it cannot be read."""
    if not path:
        return None
    try:
        with open(path, newline="") as fh:
            return sum(
                bool((r.get("case_id") or "").strip()) for r in csv.DictReader(fh)
            )
    except (OSError, csv.Error, UnicodeDecodeError):
        return None


def attrition(jobs, cells, nuclei_cells, nuclei_missing, res_rows, manifest=None):
    """Where the rounds go, per dataset and registration row (``attrition.csv``):

    ``manifest_rounds``  rounds the cohort's manifest lists (``multiplex`` only, and
                         only when the manifest was given to the scorer; blank else)
    ``prepared``         cases prepared from them, as scored here
    ``with_outlines``    of those, cases with outlines on both sides (a cell row exists)
    ``with_nuclei``      of those, cases with bare nuclei on both sides
    ``run``              cases this row has a record for (anything but ``not run``)
    ``registered``       cases it registered (``ok``)
    ``scored``           cases whose outline row is its own result: not scored at the
                         unregistered pose and not left out of its summaries
    ``scored_nuclei``    the same on the bare nuclei

    One line per dataset without any registration row, so a dataset is never absent."""
    out = []
    for ds in dict.fromkeys(job[1].dataset for job in jobs):
        ids = {str(job[1].case_id) for job in jobs if job[1].dataset == ds}

        def of(rows):
            return [r for r in rows if r["dataset"] == ds]

        outlined = {str(r["case_id"]) for r in of(cells)}
        without = {str(r["case_id"]) for r in of(nuclei_missing) if not r["method"]}
        head = {
            "dataset": ds,
            "manifest_rounds": manifest
            if ds == "multiplex" and manifest is not None
            else "",
            "prepared": len(ids),
            "with_outlines": len(outlined),
            "with_nuclei": len(outlined - without),
        }
        names = [m for m in _methods(of(res_rows)) if _is_run(m)]
        for m in names:
            mine = [r for r in of(res_rows) if r["method"] == m]

            def own(rows, m=m):
                return sum(
                    r["method"] == m and not r["imputed_initial"] and not _excluded(r)
                    for r in of(rows)
                )

            out.append(
                {
                    **head,
                    "method": m,
                    "run": sum(r["cause"] != "not_run" for r in mine),
                    "registered": sum(bool(r["ok"]) for r in mine),
                    "scored": own(cells),
                    "scored_nuclei": own(nuclei_cells),
                }
            )
        if not names:
            out.append({**head, "method": ""})
    return out


def attrition_markdown(rows):
    if not rows:
        return []
    return [
        "## From the manifest to a scored round\n",
        "Rounds at each step, per dataset and registration row (`attrition.csv`). "
        "`manifest_rounds` is blank when the scorer was not given the manifest "
        "(`score --manifest`) and for datasets that have none. `scored` counts the "
        "rounds whose row is the method's own result: a round it did not register is "
        "scored at the unregistered pose and is not counted, and a round lost for a "
        "cause that is not the method's is in no summary.\n",
        markdown(rows, list(ATTRITION_COLUMNS)),
        "",
    ]


LAUNCH_COLUMNS = ("run", "selfcheck", "realcheck", "reference", "log")


def read_verdicts(path):
    """What every run's selfcheck and realcheck said (``<out>/logs/verdicts.tsv``,
    written by ``regbench verdict``): one dict per run, ``LAUNCH_COLUMNS``, in the
    order of the file, plus ``rows`` (the result names its selfcheck status is read
    from) and ``realcheck_row``. Reported, never a gate. Empty without a file."""
    out = {}
    if not path or not Path(path).is_file():
        return []
    lines = Path(path).read_text().splitlines()
    for line in lines[1:]:  # the first line is the header
        fields = line.split("\t")
        if len(fields) < 6:
            continue
        run, check, status, reference, rows, log = fields[:6]
        rec = out.setdefault(
            run,
            {
                "run": run,
                "selfcheck": "",
                "realcheck": "",
                "reference": "",
                "log": "",
                "rows": "",
                "realcheck_row": "",
            },
        )
        if check == "selfcheck":
            rec["selfcheck"] = status
            rec["reference"] = "" if reference == "-" else reference
            rec["log"] = "" if log == "-" else log
            rec["rows"] = "" if rows == "-" else rows
        elif check == "realcheck":
            rec["realcheck"] = "" if status == "-" else status
            rec["realcheck_row"] = "" if rows == "-" else rows
            if log not in ("-", ""):
                rec["realcheck_log"] = log
    return list(out.values())


def verdict_status(launch, run_variants=None):
    """``{result name: its run's selfcheck verdict}`` from ``read_verdicts``: column
    ``selfcheck_status`` of every row. A run's name stands for itself when
    ``run_variants`` does not list its result names."""
    out = {}
    for g in launch or []:
        for v in (run_variants or {}).get(g["run"], [g["run"]]):
            out[v] = g["selfcheck"]
    return out


def set_selfcheck_status(rows, statuses):
    """Column ``selfcheck_status`` of every row of a run that has a verdict: what its
    run's selfcheck said (``verdict_status``), whatever the run's own record holds (a
    run is never told its selfcheck's verdict: it does not wait for it)."""
    for r in rows or []:
        if isinstance(r, dict) and "selfcheck_status" in r:
            got = statuses.get(str(r.get("method", "")).split(NON)[0])
            if got is not None:
                r["selfcheck_status"] = got
    return rows


def narrow_dots(selfcheck_rows):
    """Per (run, result name) of the selfcheck records: the dots its detector found in
    the images that result wrote, and how many of them were narrow (``n_narrow``:
    under ``selfcheck.NARROW_PX``, a dot the writer shrank to under about 0.8 of its
    size). Each comparison is counted once, whatever its dot sets."""
    seen, out = set(), {}
    for r in selfcheck_rows or []:
        key = (r.get("run", ""), r.get("variant", ""), r.get("case_id", ""))
        if key in seen or r.get("n_narrow") in ("", None):
            continue
        seen.add(key)
        rec = out.setdefault(
            key[:2],
            {"run": key[0], "variant": key[1], "n_detected": 0, "n_narrow": 0},
        )
        rec["n_detected"] += int(r.get("n_detected") or 0)
        rec["n_narrow"] += int(r["n_narrow"])
    return list(out.values())


def narrow_lines(selfcheck_rows):
    """What the selfcheck block says of the narrow dots: every result that has some, by
    name and count, and how many results have none."""
    counted = narrow_dots(selfcheck_rows)
    if not counted:
        return []
    some = [r for r in counted if r["n_narrow"]]
    text = (
        "Narrow dots (`n_narrow` in `selfcheck_masks.csv`): the selfcheck's detector "
        "takes a dot from about half its drawn size to about twice, so that its "
        "verdict is about WHERE a dot is and not about the registration's local "
        "scale; a dot the writer drew at under about 0.8 of its size is counted as a "
        "dot and counted here. "
    )
    if some:
        text += (
            "Results with some: "
            + "; ".join(
                f"`{r['variant']}` {r['n_narrow']} of {r['n_detected']} detected"
                + (f" (run `{r['run']}`)" if r["run"] != r["variant"] else "")
                for r in some
            )
            + f". None in the other {len(counted) - len(some)} checked result(s).\n"
        )
    else:
        text += f"None in the {len(counted)} checked result(s).\n"
    return [text]


SELFCHECKS_SAID = "## What the selfchecks said\n"


def launch_markdown(launch, selfcheck_rows=None):
    """What every run's selfcheck and realcheck said, one line per run, and the narrow
    dots of the selfchecks. Reported, never a gate: every run started and every row of
    theirs is in the tables above whatever its line here says."""
    narrow = narrow_lines(selfcheck_rows)
    if not launch:
        return [SELFCHECKS_SAID, *narrow] if narrow else []
    parts = [
        SELFCHECKS_SAID,
        "The selfcheck (do a method's points follow the image its own writer drew, on "
        "a generated pair) and the realcheck (the same on the run's first real unit) "
        "are REPORTED: no run waited for them and no row is left out because of them. "
        "Every row carries its run's verdict (`selfcheck_status`) and its own "
        "(`selfcheck_row`).\n",
    ]
    for g in launch:
        line = f"- `{g['run']}`: selfcheck {g['selfcheck'] or 'not_checked'}"
        if g.get("reference"):
            line += f" (reference comparison {g['reference']})"
        if g.get("realcheck"):
            line += f"; realcheck {g['realcheck']}"
            if g.get("realcheck_log"):
                line += f" ({g['realcheck_log']})"
        if g.get("log"):
            line += f"; log {g['log']}"
        parts.append(line)
    return [*parts, "", *narrow]


MASK_FLAG = "selfcheck_mask_over_cap"
MASK_SUMMARY_COLUMNS = (
    "run",
    "variant",
    "case_id",
    "set",
    "n_dots",
    "found_frac",
    "err_px_p95",
    "err_px_mean",
    "worst_block_err_px_p95",
    "masked_frac",
)


def flag_rows(rows, flags):
    """Column ``selfcheck_mask_over_cap`` on every row that names a method: what the
    selfcheck's mask hid for that row's run (``variants.mask_flags``)."""
    for r in rows or []:
        if isinstance(r, dict) and "method" in r:
            r[MASK_FLAG] = (flags or {}).get(str(r["method"]).split(NON)[0], "")
    return rows


def mask_section(flags, rows):
    """The summary's lines on the selfcheck's masks: the runs over their cap, by name,
    and the numbers with and without the mask for every masked selfcheck."""
    over = {k: v for k, v in (flags or {}).items() if str(v).startswith("yes")}
    masked = [r for r in rows or [] if r.get("mask")]
    if not over and not masked:
        return []
    parts = ["## Selfcheck masks: nothing is hidden behind a cap\n"]
    if over:
        parts.append(
            "**The selfcheck's mask hid more of the dots than its cap for these rows. "
            "They ran and are scored all the same (the cap is a flag, not a gate); "
            f"every table row of theirs says so in `{MASK_FLAG}`:** "
            + "; ".join(f"`{k}` ({v[5:]})" for k, v in over.items())
            + ".\n"
        )
    else:
        parts.append("No row's selfcheck mask hid more of the dots than its cap.\n")
    if masked:
        caps = sorted({k for r in masked for k in r if k.startswith("passes_cap_")})
        parts.append(
            "The selfcheck's own numbers on ALL dots (no mask), on the unmasked dots "
            "(what the verdict rests on) and on the masked dots alone; `passes_cap_*`: "
            "would the row pass with the mask allowed to hide at most that share "
            "(`selfcheck_masks.csv` has every column and every unmasked method too).\n"
        )
        parts += [markdown(masked, [*MASK_SUMMARY_COLUMNS, *caps]), ""]
    return parts


def in_sample_lines(dataset, methods):
    """The caption line of the cohort's tables for the rows tuned on the cohort."""
    mine = [m for m in IN_SAMPLE if m in methods]
    if dataset != IN_SAMPLE_DATASET or not mine:
        return []
    names = " and ".join(f"`{m}`" for m in mine)
    return [
        f"**In-sample: {names}.** The settings of "
        + ("these rows" if len(mine) > 1 else "this row")
        + " (STARE's tier, the pipeline's VALIS settings) were chosen by their authors "
        "on this cohort with this metric, and the other results of the same runs "
        "share them. No other row was tuned on these slides. Their numbers below are "
        "those of a method scored on the data it was tuned on.\n"
    ]


def summary_markdown(
    summary,
    fails,
    res_summary,
    primary=None,
    nonband=None,
    nonband_lost=None,
    tissue_tables=None,
    confined="",
    end_to_end=None,
    launch=None,
    header=None,
    common=None,
    paired=None,
    excluded=None,
    events=None,
    cost_summary=None,
    nuclei=None,
    masks=None,
    attrition_rows=None,
    run_fails=None,
    pipeline=None,
):
    nuc, nuclei_missing, n_cases = nuclei or ({}, [], 0)
    nuc_summary = nuc.get("summary_cells") or []
    parts = ["# Registration benchmark\n"]
    parts.append(
        "Nuclear overlap after registration, on nuclei segmented on the native slides. "
        "It is self-consistency on the registration channel, not accuracy. Each cell is "
        "`median over units (mean over the same units) [minimum over cases]`; "
        "`ok/cases` on the same line counts the runs that registered, and a run that "
        "did not is scored at the unregistered pose and is in the median, the mean and "
        "the minimum. A mean far from its median means some units are far from the "
        "others.\n"
    )
    parts.append(SUBSET_RULE)
    parts.append(PAIRING_RULE)
    parts.append(SETS_RULE)
    parts += _nuclei_missing_lines(nuclei_missing, n_cases, bool(nuc_summary))
    parts.append(
        "`selfcheck of this row` is what the selfcheck measured for THAT row, not for "
        "its run: `NEVER CHECKED` marks a stage the method's own writer draws no image "
        "of (its points were compared with nothing), `via <run>` a row whose run is "
        "checked by another run's selfcheck.\n"
    )
    parts += mask_section(*(masks or ({}, [])))
    parts += attrition_markdown(attrition_rows or [])
    excluded = excluded or []
    if excluded:
        parts.append(
            f"**{len(excluded)} round(s) were lost for a cause that is not the "
            "method's (`infrastructure`, `benchmark_oom`). They are left out of that "
            "method's rows (column `left out`), of its differences, and, being "
            "unregistered, of the common block for every method; their rows stay in "
            "`cells.csv` (`excluded_from_summaries`). Submit them again:** "
            + "; ".join(
                f"`{e['method']}` {e['dataset']}/{e['case_id']} ({e['cause']})"
                for e in excluded[:50]
            )
            + (f"; and {len(excluded) - 50} more" if len(excluded) > 50 else "")
            + ".\n"
        )
    for ds in sorted({r["dataset"] for r in summary}):
        of_ds = [r for r in summary if r["dataset"] == ds]
        n_units = max(r["n_patients"] for r in of_ds)
        parts.append(f"## {ds}\n")
        what = UNIT_IS.get(ds, "patient")
        parts.append(
            f"The independent unit is the {what}. At most {n_units} unit(s)"
            + f": {DESCRIPTIVE}.\n"
        )
        parts += in_sample_lines(ds, [r["method"] for r in of_ds])
        block = [r for r in of_ds if r["subset"] == "all"]
        if block:
            parts += [
                "### All cases\n",
                _summary_table(block, primary, run_fails),
                "",
            ]
            if run_fails is not None:
                parts += [FAILURE_COLUMN_RULE]
            parts += _side_by_side(block, nuc_summary, primary)
        parts.append("### Cases every compared row registered\n")
        parts += _common_lines(common or [], ds)
        block = [r for r in of_ds if r["subset"] == "common"]
        if block:
            parts += [_summary_table(block, primary, run_fails), ""]
            parts += _side_by_side(block, nuc_summary, primary)
        parts += _paired_markdown(paired or [], ds)
    parts += _nonband_markdown(nonband or [], nonband_lost or [])
    if tissue_tables:
        parts += tissue.markdown(tissue_tables, markdown)
    parts += launch_markdown(launch or [], (masks or ({}, []))[1])
    if events is not None:
        parts += failure_log.markdown(events, markdown)
    if fails:
        parts += ["## Failures, counted (launches and rounds)\n"]
        lost = sum(r["rounds_lost_not_method"] for r in fails)
        parts.append(
            "`launches_failed_method` counts what the method did (it raised, ran out "
            "of memory, ran out of time, its writer failed, its process died). "
            "`launches_lost_not_method` counts what was done to it: `infrastructure` "
            "(the scheduler lost the attempt twice at one rung, or the task never "
            "started) and `benchmark_oom` (the benchmark's own step ran out of memory "
            "at the last rung). "
            + (
                f"**{lost} round(s) are lost for a reason that is not the method's "
                + (
                    "and are left out of that method's rows above"
                    if NOT_THE_METHOD_ROUNDS == "exclude"
                    else "and are still scored at the unregistered pose above"
                )
                + ": submit them again before reading those rows.**\n"
                if lost
                else "No round is lost for a reason that is not the method's.\n"
            )
        )
        parts += [markdown(fails, list(fails[0])), ""]
    if res_summary:
        table = []
        for r in res_summary:
            split = r.get("n_jobs") not in ("", None)
            line = {
                "dataset": r["dataset"],
                "group": r["group"],
                "method": r["method"],
                "unit": r["unit"],
                "n_ok": r["n_ok"],
                "cpu_h_used": r.get("cpu_h_used", math.nan),
                "cpu_h_reserved": r.get("cpu_h_reserved", math.nan),
                "cpu_efficiency": r["cpu_efficiency"],
                "gpu_h": r["gpu_h"],
                "device": r.get("device", ""),
                "gb_h_reserved": r.get("gb_h_reserved", math.nan),
                "wasted_cpu_h_reserved": r.get("wasted_cpu_h_reserved", math.nan),
                "total_cpu_h_reserved": r.get("total_cpu_h_reserved", math.nan),
                "lost_cpu_h_reserved": r.get("lost_cpu_h_reserved", math.nan),
                "cpu_h_scheduler": r.get("cpu_h_scheduler", math.nan),
                "cpu_h_used_vs_scheduler": r.get("cpu_h_used_vs_scheduler", math.nan),
                "cpu_h_used_per_gpx": r.get("cpu_h_used_per_gpx", math.nan),
                "register_s": "" if split else r["register_s"],
                PARALLEL_TIMES["register_s"]: r["register_s"] if split else "",
                "peak_cpus_at_once": r.get("peak_cpus_at_once", "") if split else "",
                "write_s": r["write_s"],
                "total_to_image_s": "" if split else r["total_to_image_s"],
                PARALLEL_TIMES["total_to_image_s"]: r["total_to_image_s"]
                if split
                else "",
                "peak_rss_gb": r["peak_rss_gb"],
                "cgroup_peak_gb": r["cgroup_peak_gb"],
                "attempt": r["attempt"],
                "mem_requested_gb": r["mem_requested_gb"],
            }
            for key in (
                "wasted_attempts",
                "wasted_from_scheduler",
                "wasted_unknown",
                "lost_attempts",
                "unmeasured_attempts",
            ):
                line[key] = r.get(key, "")
            table.append(line)
        parts += [
            "## Resources (per patient; registration only)\n",
            "CPU-hours first, by one rule for every row (`regbench.billing`): "
            "`cpu_h_used` is the user + system CPU time of the registration's own "
            "processes, `cpu_h_reserved` the allocated CPUs x duration and "
            "`gb_h_reserved` the requested memory x duration, each summed over the "
            "attempts that succeeded, every job of the row; `cpu_efficiency` is used "
            "over reserved; `wasted_*` are the same over the attempts that failed (a "
            "rung of the memory ladder, a retried stage job) and `total_*` is both. "
            "`lost_*` are the attempts lost to causes that are not the method's (a "
            "node that failed, a cancelled job, the benchmark's own step), counted in "
            "`lost_attempts`: they are in no `wasted_*` and no `total_*` figure. A "
            "patient's figure is blank when one of its registered rounds lacks it. "
            "The method's writer and the benchmark's own work (moving the nuclei, "
            "scoring) are in none of them. CPU time is read from a counter the kernel "
            "keeps, never by sampling: `cpu_instrument` in `resources.csv` names it "
            "for every row ("
            + cpu_instruments_line(res_summary)
            + "). `cpu_h_scheduler` is the scheduler's own CPU time of the row's jobs, "
            "which for a row that is one job also hold the benchmark's work and the "
            "writer: a cross-check, blank when the scheduler gave nothing. "
            "`gpu_h` is its own column and `device` says which "
            "rows had a GPU: a `gpu` row's CPU-hours are never read without it. "
            f"`{PARALLEL_TIMES['register_s']}` is the time of a registration in "
            "several scheduler jobs with every tile running at once and no queue "
            "(`peak_cpus_at_once` is how many CPUs that took); it is not a "
            "`register_s`, which is one process on its allocation. "
            + VALID_RATIOS.strip()
            + " `peak_rss_gb` is the process tree's peak up to that result. Nothing a "
            "paused stretch allocates and frees is in it, but what an earlier result's "
            "writer left resident is (VALIS and ASHLAR write between results). "
            "`cgroup_peak_gb` is the whole job, writers included. For a result whose "
            "registration ran as several scheduler jobs per round (the header line "
            "below names it) `peak_rss_gb` is instead the peak of its largest single "
            "stage job, and `cgroup_peak_gb`, `attempt`, `mem_requested_gb` and "
            "`cpus_requested` are those of the round's LAST job, which holds the "
            "benchmark's own work and the writer and none of the registration. "
            "`write_s` is the method's own writer; the benchmark's own copying of a "
            "written image is in no column. `unmeasured_attempts` counts stage "
            "attempts with no recorded duration: where it is not 0, that row's times "
            "and CPU-hours are lower bounds. No single column is "
            "the cost. `wasted_from_scheduler` attempts are billed by the scheduler's "
            "elapsed time of the whole task (an upper bound); `wasted_unknown` "
            "attempts are in no wasted total, because nobody recorded their cost.\n",
            *([confined + "\n"] if confined else []),
            *[line + "\n" for line in (header or [])],
            *[line + "\n" for line in jvm_heap_lines(res_summary)],
            markdown(table, list(table[0])),
            "",
        ]
    if cost_summary:
        shown = (
            "cpu_h_used",
            "cpu_h_reserved",
            "cpu_efficiency",
            "gpu_h",
            "total_cpu_h_reserved",
            "gb_h_reserved",
            "cpu_h_used_per_gpx",
            "cpu_h_used_vs_scheduler",
            "register_s",
            PARALLEL_TIMES["register_s"],
            "write_s",
            "total_to_image_s",
            PARALLEL_TIMES["total_to_image_s"],
            "peak_rss_gb",
            "cgroup_peak_gb",
        )

        def cell(r, key):
            if f"{key}_med" not in r:
                return "-"
            text = (
                f"{_fmt(r[f'{key}_med']) or '-'} / "
                f"{_fmt(r[f'{key}_common_med']) or '-'}"
            )
            if f"{key}_n" in r:  # a cost a complete patient can lack
                text += f" (n {r[f'{key}_n']} / {r[f'{key}_common_n']})"
            return text

        table = []
        for r in cost_summary:
            line = {
                "dataset": r["dataset"],
                "method": r["method"],
                "patients complete / all": f"{r['n_patients_complete']}/"
                f"{r['n_patients']}",
                "common": r["n_patients_common"],
            }
            for key in shown:
                line[f"{key}: complete / common"] = cell(r, key)
                if key == "gpu_h":
                    line["device"] = r.get("device", "") or "-"
                if key == PARALLEL_TIMES["register_s"]:
                    line["peak_cpus_at_once"] = _fmt(r.get("peak_cpus_at_once", ""))
            table.append(line)
        parts += [
            "### Cost per method (median over patients), on two sets of patients\n",
            "A patient is `complete` for a method when every round of it registered; "
            "a cost is read on complete patients only (the cost of some of a "
            "patient's rounds is not the patient's). Each cell is `over the patients "
            "this method completed / over the patients EVERY compared row completed`, "
            "with the two counts in `patients complete / all` and `common`. "
            "The first is over survivors: a method that fails on the largest patients "
            "looks cheaper there. Only the second is on the same patients for every "
            "method (compared rows: "
            + "; ".join(
                f"{ds}: {among or 'none ran'}"
                for ds, among in dict.fromkeys(
                    (r["dataset"], r["common_among"]) for r in cost_summary
                )
            )
            + "). A cell with `(n a / b)` is a cost a complete patient can lack (no "
            "stage record, nothing in the accounting): its two medians are over `a` "
            "and `b` patients. The columns are CPU-hours used, reserved and their ratio, GPU-hours "
            "with the device, then the totals with failed attempts, memory-hours, "
            "CPU-hours per gigapixel of moving image and the scheduler cross-check, "
            "and only then times. "
            + VALID_RATIOS.strip()
            + f" A row in several jobs has `-` under `register_s` and "
            "`total_to_image_s` and its own columns instead "
            f"(`{PARALLEL_TIMES['register_s']}`, "
            f"`{PARALLEL_TIMES['total_to_image_s']}`, with `peak_cpus_at_once` beside "
            "them). `write_s` and the time to the image are the patient's totals for "
            "both kinds of unit. The means of the same patients are in "
            "`cost_summary.csv` (`*_mean`, `*_common_mean`).\n",
            markdown(table, list(table[0])),
            "",
        ]
    if end_to_end:
        parts += cost.markdown(*end_to_end, markdown)
    # the whole pipeline per backend (compute-all): its own block, its own clock
    if pipeline is not None:
        pipe, pipe_summary = pipeline
        if pipe:
            parts += compute_all.markdown(pipe, pipe_summary)
        else:
            parts += [
                "Full pipeline cost: not measured (no <out>/compute/*/*/trace/trace.txt; "
                "submit_cohort.sh --compute-all).",
                "",
            ]
    return "\n".join(parts)


def score(
    cases_root,
    out,
    datasets=None,
    workers=1,
    dev=False,
    expected=None,
    primary=None,
    seam_owners=None,
    with_tissue=True,
    verdicts=None,
    run_variants=None,
    row_checks=None,
    mask_flags=None,
    selfcheck_masks=None,
    fresh=False,
    manifest=None,
    accounting=None,
):
    """Score everything under ``out``; writes ``<out>/tables`` and returns the rows.

    ``accounting`` is the scheduler's accounting of the jobs (the file ``sacct -n -P
    -o JobID,State,ElapsedRaw,TotalCPU`` wrote, with the jobs' steps; default ``<out>/logs/
    accounting.psv``, which the launcher's settle step leaves): the cross-check column
    ``cpu_h_scheduler``. Without one that column is blank.

    ``manifest`` is the ``prepare multiplex`` CSV the cohort's cases were made from:
    its number of rounds is the first column of ``attrition.csv``. Without it that
    column is blank; nothing else reads it.

    A case whose record is valid is read, not computed (``score_jobs``, ``case_key``;
    ``score_case`` makes one case's record on its own, which is how the cohort launcher
    scores a case per job). ``fresh`` computes everything again. ``workers`` is how
    many processes compute what is missing.

    ``row_checks`` maps a result name to what the selfcheck says of THAT row
    (``variants.row_checks``): column ``selfcheck_row`` of every table, so a row whose
    stage was never compared with an image (``no_stage_image``) says so beside its
    numbers. Without it the column is empty.

    ``run_variants`` maps a run's name to the result names it writes
    (``variants.run_variants()``): with the verdicts it says which rows belong to a
    run. Without it a verdict names its run only.

    ``verdicts`` is ``<out>/logs/verdicts.tsv`` (``regbench verdict``): what every
    run's selfcheck and realcheck said, written to ``launch.csv``, named in the summary
    and set in column ``selfcheck_status`` of every row of the run. REPORTED, never a
    gate: no row is left out and no run is taken out of the common block for it.

    ``with_tissue`` adds the tissue-level block of the written images (``tissue*.csv``,
    ``regbench.tissue``): secondary, its own tables, never part of the ones above.

    ``expected`` lists variants that must appear even with no directory. ``primary`` lists
    the primary variants: they come first in the summary and define the ``common`` block.
    ``seam_owners`` (default ``SEAM_OWNERS``) names the variants whose seam mask gives
    every variant a second, off-seam row (``nonband_*``).

    A results folder slimmed to ``csv`` (``refuse_slimmed``) is refused here, before
    anything is read.
    """
    refuse_slimmed(out, "score")
    owners = tuple(SEAM_OWNERS if seam_owners is None else seam_owners)
    variants_of = {}
    jobs = []
    for ds in datasets or DATASETS:
        made = case_jobs(cases_root, out, ds, dev, expected, owners, row_checks)
        if made:
            variants_of[ds] = made[0][2]
            jobs += made
    if not jobs:
        raise SystemExit(f"no prepared cases under {cases_root}")
    done = score_jobs(jobs, workers, fresh)
    cells = [r for rows in done for r in rows[0]]
    res = [r for rows in done for r in rows[1]]
    # off the seams: tables of their own, never merged into the ones above
    non_cells = [r for rows in done for r in rows[2]]
    non_lost = nonband_missing(
        [r for rows in done for r in rows[3]], jobs, owners, variants_of
    )
    launch = read_verdicts(verdicts)
    said = verdict_status(launch, run_variants)
    for rows in (cells, non_cells):
        set_selfcheck_status(rows, said)
    non_summary = _in_set(summarise_nonband(non_cells, primary), OUTLINES)
    summary = _in_set(summarise_cells(cells, primary), OUTLINES)
    common = _in_set(common_block(cells, primary), OUTLINES)
    excluded = excluded_rounds(cells)
    paired = _in_set(paired_vs_primary(cells), OUTLINES)
    paired_summary = _in_set(summarise_paired(paired), OUTLINES)
    # the same tables on the BARE NUCLEI: tables of their own (``<name>_nuclei.csv``),
    # same columns, read with the nuclei's own gate, null and shared subset. A case
    # without nuclei has no row there and is listed with its reason, never imputed.
    nuc = {
        "cells": [r for rows in done for r in rows[4]["cells"]],
        "nonband_cells": [r for rows in done for r in rows[4]["non_cells"]],
    }
    nuclei_missing = [r for rows in done for r in rows[4]["missing"]]
    for rows in (nuc["cells"], nuc["nonband_cells"]):
        set_selfcheck_status(rows, said)
    nuc["summary_cells"] = _in_set(summarise_cells(nuc["cells"], primary), NUCLEI)
    nuc["common_block"] = _in_set(common_block(nuc["cells"], primary), NUCLEI)
    nuc["paired_vs_primary"] = _in_set(paired_vs_primary(nuc["cells"]), NUCLEI)
    nuc["paired_summary"] = _in_set(summarise_paired(nuc["paired_vs_primary"]), NUCLEI)
    nuc["nonband_summary"] = _in_set(
        summarise_nonband(nuc["nonband_cells"], primary), NUCLEI
    )
    fails = failures(res)  # every run
    run_fails = run_failures(res)
    lost_rounds = attrition(
        jobs, cells, nuc["cells"], nuclei_missing, res, manifest_rounds(manifest)
    )
    from . import ladder

    booked = Path(accounting or Path(out) / "logs" / "accounting.psv")
    bill_from_disk(
        res,
        out,
        [job[1] for job in jobs],
        ladder.read_accounting(booked) if booked.is_file() else None,
    )
    res_summary = summarise_resources(res)
    # STARE's peak memory per stage per round (score side: reads the stage records)
    peaks = [
        {k: table_schema.NA if v is None else v for k, v in r.items()}
        for _, case, names, *_ in jobs
        for r in stage_peaks.rows(out, [case], names)
    ]
    # ASHLAR's replaced tiles per round and cell (score side: reads run.json)
    discards = [
        {k: table_schema.NA if v is None else v for k, v in r.items()}
        for case in {
            (j[1].dataset, str(j[1].case_id)): j[1] for j in jobs
        }.values()  # a case once, not per job
        for r in ashlar_discard.rows(out, [case])
    ]
    cost_summary = summarise_cost(res_summary, primary)
    confined = cost.cores_confined(res)
    cost_header = cost.header_lines(res)
    e2e = cost.end_to_end(res)
    e2e_summary = cost.summarise(
        e2e,
        {ds: got[0] for ds, got in common_patients(res_summary, primary).items()},
        {(r["dataset"], r["method"]): r["n_patients"] for r in cost_summary},
    )
    # the whole pipeline per patient and backend, from compute-all's traces
    pipe = compute_all.rows(out)
    pipe_summary = compute_all.summarise(pipe)
    pipe_by_process = compute_all.by_process(pipe, pipe_summary)

    tables = Path(out) / ("tables_dev" if dev else "tables")
    tables.mkdir(parents=True, exist_ok=True)
    for name in (*TABLES, *REMOVED_TABLES):
        (tables / f"{name}.csv").unlink(missing_ok=True)
    # what the selfcheck's mask hid for each row's run, on every row of every table
    for rows in (
        cells,
        summary,
        paired,
        res,
        res_summary,
        cost_summary,
        non_cells,
        non_summary,
        *(nuc[name] for name in NUCLEI_TABLES),
    ):
        if mask_flags is not None:  # a score outside the launcher has no selfcheck
            flag_rows(rows, mask_flags)
    (tables / "selfcheck_masks.csv").unlink(missing_ok=True)
    if selfcheck_masks is not None:
        write_csv(tables / "selfcheck_masks.csv", list(selfcheck_masks))
    for name in NUCLEI_TABLES:
        write_csv(tables / f"{name}{NUCLEI_SUFFIX}.csv", nuc[name])
    write_csv(tables / "nuclei_missing.csv", nuclei_missing)
    write_csv(tables / "attrition.csv", lost_rounds, list(ATTRITION_COLUMNS))
    write_csv(tables / "cells.csv", cells)
    write_csv(tables / "summary_cells.csv", summary)
    write_csv(tables / "paired_vs_primary.csv", paired)
    write_csv(tables / "failures.csv", fails)
    write_csv(tables / "resources.csv", res)
    write_csv(tables / "summary_resources.csv", res_summary)
    write_csv(
        tables / "stare_stage_peaks.csv",
        peaks,
        table_schema.columns("stare_stage_peaks")[1:],
    )
    write_csv(
        tables / "ashlar_discard.csv",
        discards,
        table_schema.columns("ashlar_discard")[1:],
    )
    write_csv(tables / "cost_summary.csv", cost_summary)
    write_csv(tables / "common_block.csv", common)
    write_csv(tables / "paired_summary.csv", paired_summary)
    write_csv(tables / "excluded_rounds.csv", excluded)
    write_csv(tables / "cost_end_to_end.csv", e2e, list(cost.COLUMNS))
    write_csv(tables / "cost_end_to_end_summary.csv", e2e_summary)
    write_csv(tables / "pipeline_cost.csv", list(pipe), list(compute_all.COLUMNS))
    write_csv(
        tables / "pipeline_cost_summary.csv",
        pipe_summary,
        list(compute_all.SUMMARY_COLUMNS),
    )
    write_csv(
        tables / "pipeline_cost_by_process.csv",
        pipe_by_process,
        list(compute_all.BY_PROCESS_COLUMNS),
    )
    write_csv(
        tables / "launch.csv",
        [{k: g.get(k, "") for k in LAUNCH_COLUMNS} for g in launch],
        list(LAUNCH_COLUMNS),
    )
    write_csv(tables / "nonband_cells.csv", non_cells)
    write_csv(tables / "nonband_summary.csv", non_summary)
    write_csv(tables / "nonband_missing.csv", non_lost)
    # the written images against the reference: tables of their own, never ranked
    tis = (
        tissue.score_all([(j[0], j[1], j[2]) for j in jobs], workers)
        if with_tissue
        else {}
    )
    for name, rows in tis.items():
        write_csv(tables / f"{name}.csv", rows)
    events = failure_log.build(
        out,
        [(j[1], j[2]) for j in jobs],
        launch,
        run_variants,
        tis.get("tissue_missing", ()),
        of_this_submission,
    )
    # always written, a header alone when nothing failed: an absent log is not "none"
    with open(tables / "failure_log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(failure_log.COLUMNS), restval="")
        w.writeheader()
        w.writerows(events)
    text = summary_markdown(
        summary,
        fails,
        res_summary,
        primary,
        non_summary,
        non_lost,
        tis,
        confined,
        (e2e, e2e_summary),
        launch,
        cost_header,
        common,
        paired_summary,
        excluded,
        events,
        cost_summary,
        (nuc, nuclei_missing, len(jobs)),
        (mask_flags, selfcheck_masks),
        lost_rounds,
        run_fails=run_fails,
        pipeline=(pipe, pipe_summary),
    )
    (tables / "summary.md").write_text(text)
    print(text)
    print(f"tables -> {tables}")
    return {
        "cells": cells,
        "summary_cells": summary,
        "paired_vs_primary": paired,
        "paired_summary": paired_summary,
        "common_block": common,
        "excluded_rounds": excluded,
        "cost_summary": cost_summary,
        "failure_log": events,
        "failures": fails,
        "resources": res,
        "summary_resources": res_summary,
        "stare_stage_peaks": peaks,
        "ashlar_discard": discards,
        "nonband_cells": non_cells,
        "nonband_summary": non_summary,
        "nonband_missing": non_lost,
        "cost_end_to_end": e2e,
        "cost_end_to_end_summary": e2e_summary,
        "pipeline_cost": list(pipe),
        "pipeline_cost_summary": pipe_summary,
        "pipeline_cost_by_process": pipe_by_process,
        "cores_confined": confined,
        "cost_header": cost_header,
        "launch": launch,
        **{f"{name}{NUCLEI_SUFFIX}": rows for name, rows in nuc.items()},
        "nuclei_missing": nuclei_missing,
        "attrition": lost_rounds,
        **tis,
    }
