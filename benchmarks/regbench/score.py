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
``truth``       (cases with a known map) a pseudo-method: the moving cells under the TRUE map.
``fold_pct``, ``sd_log_jac``   the share of a lattice where the method's map folds and the
                spread of its log Jacobian (forward evaluation only). A map that snaps
                nuclei onto their nearest neighbour is rewarded by the paired numbers; this
                and the two-sided curve are what object.

**Known-truth error** (``regbench.truth``). On cases made by a known map, ``truth.csv`` has
the distance between where a method puts each probe and where it truly belongs, in the
method's own direction, and ``truth_by_family.csv`` summarises it per (dataset, family,
pose, variant). Families are never pooled. The summary's known-truth header names the
spec's hash and the cohort patients the semi-synthetic cases were built from
(``truth_patients.json``); cases of two patient lists are refused, like cases of two
specs. Cases of the frozen spec are left out of the
cost tables, and cases of two spec hashes are refused.

**A failed run is a row, never a gap.** A variant that failed, was killed, or never ran is
scored at the unregistered pose (``imputed_initial``), with ``problem`` and ``cause``
filled. ``expected`` names variants that must appear even with no directory.

**Reading rules, fixed before any cohort run.** Displacement is censored at the gate and
conditional on being paired, so it is not used to rank; it is summarised over the cases
flagged ``displacement_readable`` only, with their count. Matched Dice is summarised over
all cases and, as ``dice_matched_ranked``, over the cases where pair fraction clears the
null's maximum; only the second may enter a ranking. With fewer than six patients the
summary says so and claims no ranking.

**Summaries.** Per patient the median over its rounds, then the median, minimum and maximum
over patients; beside them ``n_ok / n_cases`` and the minimum over CASES, because with a few
patients a median hides failures. The ``common`` block repeats the summary on the cases
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
separate columns, and the pipeline's convert and merge-and-pyramid steps read from a
Nextflow trace when one is given (``pipeline_trace``). ``launch.csv`` lists what the
launcher's gates decided (``gates``).

``dev_*`` cases are left out of every table unless ``dev`` is given.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from . import DATASETS, cost, failure_log, pairs, tissue, truth
from .cases import (
    NUCLEI,
    OUTLINES,
    list_cases,
    load_points,
    nuclei_reason,
    polys,
    result_dir,
)
from .mapping import FRAME, MOV_TO_REF, REF_TO_MOV

# "cache": what methods share between runs (ASHLAR's reference tiles), never a variant
RESERVED = ("tables", "tables_dev", "logs", "cache")
TIERS = ("headline", "secondary")
# semisynth families with parameters from papers
HEADLINE_FAMILIES = ("grid", "multiscale")
PSEUDO = ("initial", "truth")  # not registrations run by the driver
# <variant>:non_<owner>: the variant scored without the nuclei on <owner>'s seams
NON = ":non_"
# Methods whose primary row carries a seam mask (``warped.npz``: ``band`` on the moving
# nuclei, ``band_ref`` on the reference nuclei, ``band_truth`` on the known-truth probes
# of a reference-side owner). The owner's name is its primary variant.
SEAM_OWNERS = ("ashlar", "palom", "palom_b1024")
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
TRUTH_XY = {OUTLINES: "cells_mov_xy_truth", NUCLEI: "nuclei_mov_xy_truth"}
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
PAIRED_AGAINST = "stare"
MIN_PATIENTS_TO_RANK = 6
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
    "not_run",
)
NOT_THE_METHOD = ("benchmark_oom", "infrastructure")
# What becomes of a round lost for a cause in NOT_THE_METHOD. "exclude": it is left out
# of that method's summaries and, being unregistered, of the common block for every
# method; it is counted and named, and its row stays in cells.csv. "impute": it is
# scored at the unregistered pose like a failure of the method (the rule before
# 2026-10-11). ONE switch: every summary reads `_excluded`.
NOT_THE_METHOD_ROUNDS = "exclude"
# Datasets made by known maps: their cell summaries are per deformation family, never
# pooled over families (a family outside a method's model must show in its own row).
PER_FAMILY = ("synthetic", "semisynth")
# What one independent unit is, per dataset: `n_patients`, the "under six" line and
# every across-unit statistic count THIS, not the case's group.
UNIT_IS = {
    "multiplex": "patient",
    "synthetic": "generated scene (one per family; its two poses are the same scene "
    "and are not independent)",
    "semisynth": "real patient (its windows, families and poses are cut from one "
    "slide and are not independent)",
}
# the paired bootstrap of the differences against PAIRED_AGAINST
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261011
BOOTSTRAP_LEVEL = 0.95


def tier_of(case):
    if case.dataset == "multiplex":
        return "headline"
    if case.dataset == "semisynth" and case.family in HEADLINE_FAMILIES:
        return "headline"
    return "secondary"


jacobian_stats = truth.jacobian_stats


def unit_of(case):
    """The independent unit a case belongs to (``UNIT_IS``). A semi-synthetic case is
    cut from a real patient's slide: that patient, whatever its window, family or pose.
    A synthetic case's unit is its generated scene (its group: one per family)."""
    if case.dataset == "semisynth":
        return str(case.extra.get("truth_patient") or case.group)
    return case.group


def _unit(row):
    return row.get("unit") or row["group"]


def _stratum(row):
    """The family a row is summarised in: its own on a known-truth dataset, none else."""
    return (row.get("family") or "") if row["dataset"] in PER_FAMILY else ""


def _excluded(row):
    """True for a round lost for a cause that is not the method's, when such rounds
    are left out of the summaries (``NOT_THE_METHOD_ROUNDS``)."""
    return (
        NOT_THE_METHOD_ROUNDS == "exclude"
        and bool(row.get("imputed_initial"))
        and row.get("cause") in NOT_THE_METHOD
    )


def _is_run(method):
    """A registration the driver ran (or should have): not ``initial`` or ``truth``,
    with or without an off-seam suffix."""
    return method.split(NON)[0] not in PSEUDO


def density(ps, pixel_size_um):
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
    area, cent = pairs.cp.feature_area_centroid(ps)
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


def _f(value):
    return math.nan if value is None else float(value)


def resource_row(run):
    """One run's cost, as recorded at ITS variant's yield."""
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
    # did not, the attempt is counted as UNKNOWN, never as zero.
    wasted, from_scheduler, unknown = [], 0, 0
    for p in previous:
        wall_p, cpu_p = p.get("wall_s"), p.get("cpu_s")
        if wall_p is None and p.get("scheduler_elapsed_s") is not None:
            wall_p, cpu_p = p["scheduler_elapsed_s"], p.get("scheduler_cpu_s")
            from_scheduler += 1
        elif wall_p is None:
            unknown += 1
        wasted.append((_f(wall_p), _f(cpu_p)))
    sched = run.get("scheduler") or {}
    if not ok and run.get("resources") is None and run.get("cause") != "not_run":
        # this result's own last attempt, killed before it could bill itself
        if sched.get("elapsed_s") is not None:
            wall, cpu = _f(sched["elapsed_s"]), _f(sched.get("cpu_s"))
            from_scheduler += 1
        elif run.get("launch_id"):
            unknown += 1
    watch = run.get("oom_watch") if isinstance(run.get("oom_watch"), dict) else None
    return {
        "ok": ok,
        "cause": run.get("cause", ""),
        "unit": run.get("unit", ""),
        "launch_id": run.get("launch_id") or "",
        "rounds_in_launch": rounds,
        "attempt": run.get("attempt", math.nan),
        "mem_requested_gb": _f(run.get("mem_requested_gb")),
        "cpus_requested": cpus,
        "register_s": wall,
        "register_s_per_round": _f(per_round),
        "cpu_h": cpu / 3600.0,
        # CPU time over wall x CPUs: above 1 where a method's threads are not confined
        "cpu_efficiency": cpu / (wall * cpus) if wall and cpus else math.nan,
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
        "wasted_from_scheduler": from_scheduler,
        "wasted_unknown": unknown,
        "retry": run.get("retry", 0),
        "time_limit": run.get("time_limit", ""),
        # "" when the run recorded nothing about its watcher (an older record)
        "oom_watch": "" if watch is None else ("yes" if watch.get("started") else "no"),
        "reused": bool(run.get("reused_by")),
        "rss_method": r.get("rss_method", ""),
        # for the end-to-end view (regbench.cost): the method's own conversion where it
        # is timed apart, the device, and what its writer left on disk
        "adapter": adapter,
        # what the scheduler was asked for, with no fallback (NaN outside a scheduler)
        "cpus_asked": _f(run.get("cpus_requested")),
        "device": str(info.get("device") or "cpu") if run.get("method") else "",
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
    }


DROPPED_REASON = "its selfcheck failed and it does not gate its method"


def find_variants(out, dataset, expected=None, dropped=()):
    """The variants to score: ``initial``, every result folder of the dataset, and the
    ``expected`` names that have none. A name in ``dropped`` is in neither part."""
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
    return ["initial", *(v for v in [*found, *extra] if v not in set(dropped))]


# `score --submission ID`: only what that submission ran or reused is scored. In the
# environment because the scorer's pool workers are spawned and read it there.
SUBMISSION_ENV = "REGBENCH_SCORE_SUBMISSION"
STALE = (
    "not run by this submission (the result on disk is another submission's and was "
    "not reused: a blocked run, or a unit that never started)"
)


def of_this_submission(run):
    """False for a record another submission left, when one is being scored alone."""
    from . import ladder

    return ladder.mine(run, os.environ.get(SUBMISSION_ENV) or "")


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


def _truth_rows(case, points, base, checks=None):
    """A function ``add(method, frame, direction, mapped, ...)`` and the rows it fills, for
    a case with paired truth probes; ``(None, [])`` for a case without."""
    if "truth_mov" not in points or not len(points["truth_mov"]):
        return None, []
    t_mov = np.asarray(points["truth_mov"], dtype=float)
    t_ref = np.asarray(points["truth_ref"], dtype=float)
    head = {
        **base,
        "in_stare_model": case.extra.get("in_stare_model", ""),
        "truth_spec": case.extra.get("truth_spec_sha256", ""),
    }
    rows = []

    def add(
        method,
        frame,
        direction,
        mapped,
        jac,
        imputed,
        problem,
        cause,
        keep=None,
        into=None,
        owner=None,
    ):
        # the method's own direction: its map of the side it starts from, against the
        # partner on the other side
        source, target = (t_mov, t_ref) if direction == MOV_TO_REF else (t_ref, t_mov)
        if mapped is None:  # the unregistered pose
            mapped = source
        n_all = len(source)
        if keep is not None:
            mapped, source, target = mapped[keep], source[keep], target[keep]
        seam = {} if owner is None else {"owner": owner, "n_probes_all": n_all}
        (rows if into is None else into).append(
            {
                **head,
                **seam,
                "method": method,
                "frame": frame,
                "selfcheck_row": (checks or {}).get(method.split(NON)[0], ""),
                **truth.error_stats(mapped, target, source, case.pixel_size_um),
                **jac,
                "imputed_initial": imputed,
                "problem": problem[:200],
                "cause": cause,
            }
        )

    return add, rows


def _seam_masks(out, case, variants, owners, ref, mov, points, which=OUTLINES):
    """``({owner: (keep_ref | None, keep_mov | None, keep_truth | None)}, [owners without
    a mask])`` for the owners that have a row on this dataset. The mask is read from the
    owner's own primary run and from nothing else: no other method's result can change it.

    ``which`` is the polygon set: the mask of the bare nuclei is the owner's ``band_nuc``
    / ``band_ref_nuc`` (``methods._nuclei_band`` says how the driver got them).
    ``keep_truth`` is one bool per known-truth probe (``_truth_keep``); the probes are
    scored once, with the outlines."""
    _, key, key_ref = WARPED_KEYS[which]
    masks, missing = {}, []
    for owner in owners:
        if owner not in variants:
            continue
        _, warped, _, _ = _load_run(out, owner, case)
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
            _truth_keep(points, keep_mov, warped.get("band_truth"))
            if which == OUTLINES
            else None,
        )
    return masks, missing


def _truth_keep(points, keep_mov, band_truth):
    """Which known-truth probes an owner's seam mask leaves: one bool per probe, or None
    when the mask says nothing about probes (no off-seam truth row is then made).

    A probe is a pair: the centroid of moving nucleus ``truth_cell`` (``truth_mov``) and
    where the truth puts it on the reference (``truth_ref``). It is dropped when

    * the owner's mask is on MOVING nuclei and marks the probe's own nucleus, or
    * the owner's mask is on REFERENCE nuclei and its run marked the probe itself
      (``band_truth`` in its ``warped.npz``, one bool per probe: the owner applies its
      seam rule to the probe's nucleus at ``truth_ref``; PALOM's is in
      ``methods/palom.py::probe_band``). Reference nuclei carry no probe index, so the
      scorer cannot derive this from ``band_ref``.

    Either way only the owner's geometry enters, never a score."""
    if "truth_mov" not in points:
        return None
    n = len(points["truth_mov"])
    keep = None
    if keep_mov is not None and "truth_cell" in points:
        keep = keep_mov[np.asarray(points["truth_cell"], dtype=np.int64)]
    if band_truth is not None and len(band_truth) == n:
        off = ~np.asarray(band_truth, dtype=bool)
        keep = off if keep is None else keep & off
    return keep


# what a row's `selfcheck_row` says when its stage has no image to be checked against
NEVER_CHECKED = "no_stage_image"


def _score_case(job):
    """Every row of one case: ``(cell rows, resource rows, truth rows, off-seam cell rows,
    off-seam truth rows, owners without a mask, nuclei)``. The first six are the OUTLINES'
    (and what does not depend on a polygon set); ``nuclei`` is ``{"cells", "non_cells",
    "missing"}``: the same cell rows on the bare nuclei, and, where there are none, why."""
    out, case, variants, owners = job[:4]
    checks = job[4] if len(job) > 4 else {}
    points = load_points(case)
    px = case.pixel_size_um
    base = {
        "dataset": case.dataset,
        "tier": tier_of(case),
        "case_id": case.case_id,
        "group": case.group,
        "unit": unit_of(case),
        "family": case.family,
        "pose": case.pose,
    }
    res_rows, non_truth = [], []
    add_truth, truth_rows = _truth_rows(case, points, base, checks)
    flat = {"fold_pct": 0.0, "sd_log_jac": 0.0}
    nan = {"fold_pct": math.nan, "sd_log_jac": math.nan}
    loaded = {}

    def load(v):
        if v not in loaded:
            loaded[v] = _load_run(out, v, case)
        return loaded[v]

    def one_set(which):
        """The cell rows of one polygon set. The resource rows and the known-truth rows
        do not depend on the set: they are made once, with the outlines."""
        first = which == OUTLINES
        xy_key = WARPED_KEYS[which][0]
        ref, mov = polys(points, "ref", which), polys(points, "mov", which)
        scorable = bool(ref is not None and mov is not None and ref.n_features)
        scorable = scorable and bool(mov.n_features)
        dens = density(ref, px) if scorable else {}
        cell_rows, non_cells, absent = [], [], []
        masks, missing = (
            _seam_masks(out, case, variants, owners, ref, mov, points, which)
            if scorable
            else ({}, [])
        )
        unregistered = {}

        def at_initial():
            if not unregistered:
                unregistered.update(pairs.record(ref, mov, ref, mov, px))
            return unregistered

        def row(method, frame, status, rec, jac, imputed, problem, cause):
            return {
                **base,
                "set": which,
                "method": method,
                "frame": frame,
                "selfcheck_status": status,
                # THIS row's own check (variants.row_checks); selfcheck_status above is
                # its run's gate, which a row that was never checked shares with its run
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

        def off_seams(v, frame, status, ref_in, mov_in, jac, imputed, problem, cause):
            """The same variant on the nuclei each owner's mask leaves, one row per owner."""
            for owner, (keep_ref, keep_mov, _) in masks.items():
                r_in = ref_in if keep_ref is None else _keep_features(ref_in, keep_ref)
                m_in = mov_in if keep_mov is None else _keep_features(mov_in, keep_mov)
                # the gate still comes from ALL native nuclei: one gate per case and set
                rec = pairs.record(ref, mov, r_in, m_in, px)
                non_cells.append(
                    {
                        **row(
                            f"{v}{NON}{owner}",
                            frame,
                            status,
                            rec,
                            jac,
                            imputed,
                            problem,
                            cause,
                        ),
                        "variant": v,
                        "owner": owner,
                        "n_mov_all": int(mov.n_features),
                        "n_mov_kept": int(m_in.n_features),
                        "n_ref_all": int(ref.n_features),
                        "n_ref_kept": int(r_in.n_features),
                    }
                )

        for v in variants:
            if v == "initial":
                run, warped, problem, cause = {}, None, "", ""
            else:
                run, warped, problem, cause = load(v)
                if first:
                    res_rows.append(
                        {
                            **base,
                            "method": v,
                            **resource_row(run or {"cause": "not_run"}),
                        }
                    )
            failed = v != "initial" and warped is None
            if first and add_truth is not None:
                direction = (
                    str(warped["direction"]) if warped is not None else MOV_TO_REF
                )
                mapped = warped.get("truth_xy") if warped is not None else None
                t_problem, t_cause, t_failed = problem, cause, failed
                if warped is not None and (
                    direction not in FRAME
                    or mapped is None
                    or len(mapped) != len(points["truth_mov"])
                ):
                    mapped, direction, t_failed = None, MOV_TO_REF, True
                    t_problem = "warped.npz does not fit the case's truth probes"
                    t_cause = "exception"
                jac = flat if mapped is None else jacobian_stats(warped)
                args = (jac, t_failed, t_problem, t_cause)
                add_truth(v, FRAME[direction], direction, mapped, *args)
                # the same probes for every variant of the case: those the owner's seam
                # mask leaves (``_truth_keep``)
                for owner, (_, _, keep_truth) in masks.items():
                    if keep_truth is None:
                        continue
                    add_truth(
                        f"{v}{NON}{owner}",
                        FRAME[direction],
                        direction,
                        mapped,
                        *args,
                        keep_truth,
                        into=non_truth,
                        owner=owner,
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
                cell_rows.append(
                    row(
                        v,
                        "reference",
                        status,
                        at_initial(),
                        flat,
                        imputed,
                        problem,
                        cause,
                    )
                )
                off_seams(
                    v, "reference", status, ref, mov, flat, imputed, problem, cause
                )
                continue
            xy = warped[xy_key].astype(float)
            if direction == REF_TO_MOV:
                ref_in, mov_in = pairs.with_xy(ref, xy), mov
            else:
                ref_in, mov_in = ref, pairs.with_xy(mov, xy)
            jac = jacobian_stats(warped)
            rec = pairs.record(ref, mov, ref_in, mov_in, px)
            frame = FRAME[direction]
            cell_rows.append(row(v, frame, status, rec, jac, False, problem, cause))
            off_seams(v, frame, status, ref_in, mov_in, jac, False, problem, cause)
        if first and add_truth is not None:
            # the true map itself: zero by construction, shown so the table has its floor
            add_truth(
                "truth",
                "reference",
                MOV_TO_REF,
                points["truth_ref"],
                nan,
                False,
                "",
                "",
            )
        if scorable and TRUTH_XY[which] in points:
            moved = pairs.with_xy(mov, points[TRUTH_XY[which]].astype(float))
            rec = pairs.record(ref, mov, ref, moved, px)
            cell_rows.append(row("truth", "reference", "", rec, nan, False, "", ""))
            off_seams("truth", "reference", "", ref, moved, nan, False, "", "")
        lost = [{**base, "owner": owner} for owner in missing]
        return cell_rows, non_cells, lost, absent, scorable

    cell_rows, non_cells, lost, _, scorable = one_set(OUTLINES)
    why = nuclei_reason(case, points)
    if not scorable:
        # no outlines either (a landmark set): there is nothing the nuclei are missing from
        nuclei = {"cells": [], "non_cells": [], "missing": []}
    elif why:
        # a case without bare nuclei is scored on its outlines; nothing is imputed
        nuclei = {
            "cells": [],
            "non_cells": [],
            "missing": [{**base, "method": "", "reason": why}],
        }
    else:
        nuc_rows, nuc_non, _, absent, _ = one_set(NUCLEI)
        nuclei = {"cells": nuc_rows, "non_cells": nuc_non, "missing": absent}
    return cell_rows, res_rows, truth_rows, non_cells, non_truth, lost, nuclei


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


def _per_patient(rows, key):
    """``{independent unit: median over its rounds}`` (``UNIT_IS``)."""
    by = {}
    for r in rows:
        by.setdefault(_unit(r), []).append(_f(r.get(key)))
    return {g: _stat(v, np.median) for g, v in by.items()}


def _methods(rows):
    return list(dict.fromkeys(r["method"] for r in rows))


NOT_LAUNCHED = "not launched by this submission (it has no result on this dataset)"
NO_ROW = "it has no row on this dataset (dropped, or not expected here)"


def compared_rows(rows, primary=None, blocked=None):
    """``(inside, outside)`` for the rows of ONE dataset: the rows the common block and
    the shared subsets are computed among, and ``{row: why it is not one of them}``.

    A row is compared when it is a primary row (every registration row when no primary
    is named) that actually RAN in what is being scored: a gate did not block its run
    (``blocked``: ``{row: reason}``, from the launcher's gate records) and at least one
    of its cases has a record. A run that was never launched (a submission of some of
    the runs) or that a gate stopped has every case ``not run``; counting it would
    empty the block for everyone."""
    blocked = blocked or {}
    names = list(primary) if primary else [m for m in _methods(rows) if _is_run(m)]
    inside, outside = [], {}
    for name in names:
        mine = [r for r in rows if r["method"] == name]
        if name in blocked:
            outside[name] = blocked[name]
        elif not mine:
            outside[name] = NO_ROW
        elif all(r.get("cause") == "not_run" for r in mine):
            outside[name] = NOT_LAUNCHED
        else:
            inside.append(name)
    return inside, outside


def common_cases(rows, primary=None, blocked=None):
    """Case ids every compared row registered (``compared_rows``). A round lost for a
    cause that is not the method's is unregistered, so it is in no common block."""
    inside, _ = compared_rows(rows, primary, blocked)
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


def common_block(rows, primary=None, blocked=None):
    """One row per (dataset, compared-or-not row): who is in the common block, and why
    the others are not. Written as ``common_block.csv`` and printed, always."""
    out = []
    for ds in sorted({r["dataset"] for r in rows}):
        of_ds = [r for r in rows if r["dataset"] == ds]
        inside, outside = compared_rows(of_ds, primary, blocked)
        common = common_cases(of_ds, primary, blocked)
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


def summarise_cells(rows, primary=None, blocked=None):
    """One row per (dataset, family, method, subset): ``all`` and ``common``.

    ``family`` is empty on the cohort and one deformation family on a known-truth
    dataset (``PER_FAMILY``): those are never pooled over families. ``n_patients``
    counts independent units (``UNIT_IS``). A round lost for a cause that is not the
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
        inside, _ = compared_rows(of_ds, primary, blocked)
        common = common_cases(of_ds, primary, blocked)
        for fam in dict.fromkeys(_stratum(r) for r in of_ds):
            of_fam = [r for r in of_ds if _stratum(r) == fam]
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
                    }
                    for h in HEADLINES:
                        per = list(_per_patient(mine, h).values())
                        rec[f"{h}_med"] = _stat(per, np.median)
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
        for fam in dict.fromkeys(_stratum(r) for r in of_ds):
            of_fam = [r for r in of_ds if _stratum(r) == fam]
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
            out.append(rec)
    return out


def summarise_resources(res_rows):
    """Per (dataset, patient, method). A patient's cost is the sum over its rounds for a
    pair method and the launch's own total for a per-patient one; the peak is the maximum
    and is never divided by rounds. Only completed runs are a cost. ``wasted_*`` is the
    earlier attempts of the ladder plus a last attempt that failed outright.

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
        # The writer's time is the SUM over the rounds for both units: a per-patient
        # method records each round's share of one writer call (or each slide's own
        # time), a pair method each round's own. The time to an image is the
        # registration plus that sum. (Until 2026-10-11 a per-patient launch reported
        # the largest single round: one share of the writer, not the whole.)
        rec["write_s"] = _stat(_values(ok, "write_s"), np.sum)
        rec["total_to_image_s"] = (
            float(np.nansum([rec["register_s"], rec["write_s"]])) if ok else math.nan
        )
        rec["cpu_efficiency"] = _stat(_values(ok, "cpu_efficiency"), np.median)
        for key in ("peak_rss_gb", "peak_rss_after_register_gb", "cgroup_peak_gb"):
            rec[key] = _stat(_values(ok, key), np.max)
        rec["gpu_model"] = "; ".join(sorted({r["gpu_model"] for r in ok} - {""}))
        # A failed round inside a per-patient launch that otherwise finished is not a
        # wasted attempt; a launch that produced nothing is.
        lost = [r for r in mine if not r["ok"] and r["cause"] != "not_run"]
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
        for key in ("wasted_from_scheduler", "wasted_unknown"):
            rec[key] = _stat(_values(mine, key), total)
        out.append(rec)
    return out


COST_KEYS = (
    "register_s",
    "cpu_h",
    "gpu_h",
    "write_s",
    "total_to_image_s",
    "peak_rss_gb",
    "cgroup_peak_gb",
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
        for key in COST_KEYS:
            rec[f"{key}_med"] = _stat(_values(done, key), np.median)
            rec[f"{key}_common_med"] = _stat(_values(shared, key), np.median)
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


def _summary_table(rows, primary):
    """Markdown rows: each headline as ``median [minimum over cases]``, its null beside it."""
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
            line[h] = f"{_fmt(r[f'{h}_med'])} [{_fmt(r[f'{h}_min_case'])}]"
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
    results = [r for r in missing if r["method"]]
    lines = []
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
            line[f"d_{h}"] = f"{_fmt(r[f'd_{h}_med'])} [{interval}]"
        table.append(line)
    few = max(r["n_patients"] for r in mine) < MIN_PATIENTS_TO_RANK
    return [
        f"### Differences against `{mine[0]['against']}` (paired within unit)\n",
        "Each cell is the median over units of (this row minus "
        f"`{mine[0]['against']}`), each unit's rounds first reduced to their median on "
        "the cases both rows have, with a "
        f"{BOOTSTRAP_LEVEL:.0%} percentile bootstrap interval over units "
        f"({BOOTSTRAP_N} resamples of the units, seed {BOOTSTRAP_SEED}; a unit's "
        "rounds stay together). With few units the interval is coarse and too narrow; "
        "it describes the spread between units and is not a test"
        + (f". {DESCRIPTIVE.capitalize()}.\n" if few else ".\n"),
        markdown(table, list(table[0])),
        "",
    ]


TRUTH_COLUMNS = (
    "dataset",
    "family",
    "pose",
    "in_stare_model",
    "method",
    "n_ok",
    "n_cases",
    "initial_tre_um_p50_med",
    "tre_um_p50_med",
    "tre_um_p50_max",
    "tre_um_p90_med",
    "tre_um_max_max",
    "frac_under_1um_med",
    "frac_under_2um_med",
    "fold_pct_med",
    "n_excluded_not_method",
    "selfcheck_row",
)


def truth_header(spec_hashes, record, prepared, folded=(), reseeded=()):
    """The lines that say what the known-truth numbers rest on: the frozen spec, and the
    cohort patients the semi-synthetic cases were built from."""
    lines = []
    for h in spec_hashes:
        lines.append(f"Frozen spec `truth_spec.json`, sha256 `{h}`.")
    if not spec_hashes:
        lines.append("These cases are not the frozen ones (no spec hash).")
    if record:
        ids = record["patients"]
        have = [p for p in ids if p in prepared]
        source = (
            f"manifest `{record['manifest']}` (sha256 "
            f"`{record['manifest_sha256'][:12]}`)"
            if record["manifest"]
            else "given on the command line, no manifest"
        )
        lines.append(
            f"Semi-synthetic cases: {len(ids)} cohort patient(s): {', '.join(ids)}; "
            f"cases are prepared for {len(have)} of them ({', '.join(have) or 'none'}). "
            f"List `{record['sha256'][:12]}` ({truth.PATIENTS_FILE}), {source}."
        )
    if folded:
        lines.append(
            f"{len(folded)} case(s) were made by a true map that folds somewhere "
            "(`extra.field.fold_pct` > 0 in their case.json): "
            + ", ".join(folded)
            + "."
        )
    if reseeded:
        lines.append(
            f"{len(reseeded)} case(s) were re-seeded: the true map of the seed their "
            "seed rule asks for folds, so the next seed that does not was used "
            "(`extra.reseed` in their case.json has the skipped seeds): "
            + ", ".join(reseeded)
            + "."
        )
    return lines


def truth_by_family(rows):
    """``truth.by_family`` without the rounds lost for a cause that is not the
    method's, each (dataset, family, pose, method) saying how many it left out. A key
    with nothing left keeps a row (``n_cases`` 0): a row is never missing silently."""
    kept = [r for r in rows if not _excluded(r)]
    out = truth.by_family(kept)

    def key(r):
        return r["dataset"], r["family"], r["pose"], r["method"]

    gone = {}
    for r in rows:
        if _excluded(r):
            gone[key(r)] = gone.get(key(r), 0) + 1
    have = {key(r) for r in out}
    checks = {key(r): r.get("selfcheck_row", "") for r in rows}
    for r in out:
        r["n_excluded_not_method"] = gone.get(key(r), 0)
        r["selfcheck_row"] = checks.get(key(r), "")
    for k, n in gone.items():
        if k not in have:
            sample = next(r for r in rows if key(r) == k)
            out.append(
                {
                    "dataset": k[0],
                    "family": k[1],
                    "pose": k[2],
                    "in_stare_model": sample["in_stare_model"],
                    "method": k[3],
                    "n_cases": 0,
                    "n_ok": 0,
                    "n_excluded_not_method": n,
                }
            )
    return out


def summarise_nonband(rows, primary=None, blocked=None):
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
        gone = {f"{k}{NON}{owner}": v for k, v in (blocked or {}).items()}
        for rec in summarise_cells(mine, names, gone):
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


def read_gates(directory):
    """What the launcher's gate jobs decided (``<out>/logs/gates/*.tsv``): one dict per
    record with ``run``, ``gate`` (``selfcheck`` | ``realcheck``), ``status``,
    ``dropped`` (result names), ``blocked`` and ``reference``. Empty without a
    directory."""
    out = []
    if not directory or not Path(directory).is_dir():
        return out
    for path in sorted(Path(directory).glob("*.tsv")):
        fields = path.read_text().strip().split("\t")
        if len(fields) < 4:
            continue
        run, status, dropped, blocked = fields[:4]
        reference = fields[4] if len(fields) > 4 else "-"
        log_path = fields[5] if len(fields) > 5 else "-"
        out.append(
            {
                "run": run,
                "gate": "realcheck"
                if path.name.endswith(".realcheck.tsv")
                else "selfcheck",
                "status": status,
                "blocked": blocked.strip() != "0",
                "dropped": "" if dropped == "-" else dropped,
                "reference": "" if reference == "-" else reference,
                "log_path": "" if log_path == "-" else log_path,
            }
        )
    return out


def launch_markdown(launch):
    """The block that names every run a gate blocked and every row a gate dropped."""
    if not launch:
        return []
    blocked = [g for g in launch if g["blocked"]]
    unchecked = [
        g
        for g in launch
        if not g["blocked"] and g["status"] not in ("ok", "realcheck_ok")
    ]
    dropped = [g for g in launch if g["dropped"]]
    parts = ["## What the gates decided\n"]
    if blocked:
        parts.append(
            "Runs that did NOT start (their rows above are `not run`, scored at the "
            "unregistered pose, and they are not among the compared rows): "
            + "; ".join(f"`{g['run']}` ({g['gate']}: {g['status']})" for g in blocked)
            + ".\n"
        )
    else:
        parts.append("No run was blocked.\n")
    if unchecked:
        parts.append(
            "Runs that started although a check did not say ok: "
            + "; ".join(f"`{g['run']}` ({g['gate']}: {g['status']})" for g in unchecked)
            + ".\n"
        )
    if dropped:
        parts.append(
            "Rows a gate said to leave out: "
            + "; ".join(f"`{g['dropped']}` (of `{g['run']}`)" for g in dropped)
            + ".\n"
        )
    return parts


def blocked_rows(launch, run_variants=None):
    """``{result name: why}`` for every row of a run a gate blocked."""
    out = {}
    for g in launch or []:
        if g["blocked"]:
            for v in (run_variants or {}).get(g["run"], [g["run"]]):
                out[v] = (
                    f"its run `{g['run']}` was blocked ({g['gate']}: {g['status']})"
                )
    return out


def summary_markdown(
    summary,
    fails,
    res_summary,
    primary=None,
    by_family=None,
    known_truth=None,
    nonband=None,
    nonband_lost=None,
    tissue_tables=None,
    dropped=None,
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
):
    nuc, nuclei_missing, n_cases = nuclei or ({}, [], 0)
    nuc_summary = nuc.get("summary_cells") or []
    parts = ["# Registration benchmark\n"]
    parts.append(
        "Nuclear overlap after registration, on nuclei segmented on the native slides. "
        "It is self-consistency on the registration channel, not accuracy. Each cell is "
        "`median over units [minimum over cases]`; `ok/cases` counts the runs that "
        "registered, and a run that did not is scored at the unregistered pose.\n"
    )
    parts.append(SUBSET_RULE)
    parts.append(SETS_RULE)
    parts += _nuclei_missing_lines(nuclei_missing, n_cases, bool(nuc_summary))
    parts.append(
        "`selfcheck of this row` is what the selfcheck measured for THAT row, not for "
        "its run: `NEVER CHECKED` marks a stage the method's own writer draws no image "
        "of (its points were compared with nothing), `via <run>` a row whose run is "
        "gated by another run's selfcheck.\n"
    )
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
        per = " per family" if ds in PER_FAMILY else ""
        parts.append(
            f"The independent unit is the {what}. At most {n_units} unit(s){per}"
            + (f": {DESCRIPTIVE}.\n" if n_units < MIN_PATIENTS_TO_RANK else ".\n")
        )
        if ds in PER_FAMILY:
            parts.append(
                "One row per deformation family: families are never pooled, so a "
                "family outside a method's model shows in its own row.\n"
            )
        block = [r for r in of_ds if r["subset"] == "all"]
        if block:
            parts += ["### All cases\n", _summary_table(block, primary), ""]
            parts += _side_by_side(block, nuc_summary, primary)
        parts.append("### Cases every compared row registered\n")
        parts += _common_lines(common or [], ds)
        block = [r for r in of_ds if r["subset"] == "common"]
        if block:
            parts += [_summary_table(block, primary), ""]
            parts += _side_by_side(block, nuc_summary, primary)
        parts += _paired_markdown(paired or [], ds)
    if by_family:
        parts += [
            "## Known-truth error, per deformation family\n",
            "Distance between where a method puts a probe and where the generating map "
            "says it belongs, in the method's own direction (micrometres). One row per "
            "family, pose and variant; families are never pooled. `in_stare_model` is "
            "False for the families a B-spline mesh on STARE's lattice cannot "
            "represent. A run that did not register is scored at the unregistered "
            "pose; a point a method could not map counts as an infinite error. The "
            "amplitudes are set against published measurements and are probably "
            "larger than what rounds of cyclic IF do (README, 'Where the families "
            "come from'); the families, their smoothness and the multiscale spectrum "
            "were chosen by STARE's authors. One generated scene per family: "
            "descriptive, never ranked.\n",
            *[line + "\n" for line in known_truth or []],
            markdown(by_family, list(TRUTH_COLUMNS)),
            "",
        ]
    parts += _nonband_markdown(nonband or [], nonband_lost or [])
    if tissue_tables:
        parts += tissue.markdown(tissue_tables, markdown)
    if dropped:
        parts += [
            "## Rows left out\n",
            "These result rows are in no table above. Each is a row whose own selfcheck "
            "failed (its points do not follow an image the method wrote) while the row "
            "that gates the method passed, so the method ran and this row alone is not "
            "reported: "
            + ", ".join(
                f"`{r['variant']}`" + ("" if r["had_results"] else " (no results)")
                for r in dropped
            )
            + ".\n",
        ]
    parts += launch_markdown(launch or [])
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
        cols = [
            "dataset",
            "group",
            "method",
            "unit",
            "n_ok",
            "attempt",
            "mem_requested_gb",
            "register_s",
            "cpu_h",
            "cpu_efficiency",
            "peak_rss_gb",
            "cgroup_peak_gb",
            "gpu_h",
            "write_s",
            "total_to_image_s",
            "wasted_attempts",
            "wasted_from_scheduler",
            "wasted_unknown",
        ]
        parts += [
            "## Resources (per patient; registration only)\n",
            "`peak_rss_gb` is the process tree's peak up to that result. Nothing a "
            "paused stretch allocates and frees is in it, but what an earlier result's "
            "writer left resident is (VALIS and ASHLAR write between results). "
            "`cgroup_peak_gb` is the whole job, writers included. No single column is "
            "the cost. `wasted_from_scheduler` attempts are billed by the scheduler's "
            "elapsed time of the whole task (an upper bound); `wasted_unknown` "
            "attempts are in no wasted total, because nobody recorded their cost.\n",
            *([confined + "\n"] if confined else []),
            *[line + "\n" for line in (header or [])],
            markdown(res_summary, cols),
            "",
        ]
    if cost_summary:
        table = []
        for r in cost_summary:
            line = {
                "dataset": r["dataset"],
                "method": r["method"],
                "patients complete / all": f"{r['n_patients_complete']}/"
                f"{r['n_patients']}",
                "common": r["n_patients_common"],
            }
            for key in COST_KEYS:
                line[f"{key}: complete / common"] = (
                    f"{_fmt(r[f'{key}_med']) or '-'} / "
                    f"{_fmt(r[f'{key}_common_med']) or '-'}"
                )
            table.append(line)
        parts += [
            "### Cost per method (median over patients), on two sets of patients\n",
            "A patient is `complete` for a method when every round of it registered; "
            "a cost is read on complete patients only (the cost of some of a "
            "patient's rounds is not the patient's). Each cell is `over the patients "
            "this method completed / over the patients EVERY compared row completed`. "
            "The first is over survivors: a method that fails on the largest patients "
            "looks cheaper there. Only the second is on the same patients for every "
            "method (compared rows: "
            + (cost_summary[0]["common_among"] or "none ran")
            + "). `write_s` and `total_to_image_s` are the patient's totals for both "
            "kinds of unit.\n",
            markdown(table, list(table[0])),
            "",
        ]
    if end_to_end:
        parts += cost.markdown(*end_to_end, markdown)
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
    dropped=None,
    pipeline_trace=None,
    pipeline_trace_label=None,
    pipeline_processes=(None, None),
    gates=None,
    run_variants=None,
    row_checks=None,
):
    """Score everything under ``out``; writes ``<out>/tables`` and returns the rows.

    ``row_checks`` maps a result name to what the selfcheck says of THAT row
    (``variants.row_checks``): column ``selfcheck_row`` of every table, so a row whose
    stage was never compared with an image (``no_stage_image``) says so beside its
    numbers. Without it the column is empty.

    ``run_variants`` maps a run's name to the result names it writes
    (``variants.run_variants()``): with the gate records it says which rows belong to a
    run a gate blocked. Without it a gate record names its run only.

    ``pipeline_trace`` is a Nextflow ``trace.txt`` of an earlier pipeline run on the
    cohort: the end-to-end cost view reads the pipeline's convert and merge-and-pyramid
    steps from it (``regbench.cost``; nothing is run). ``pipeline_trace_label`` names
    that run in every imported number; ``pipeline_processes`` overrides the two process
    names. ``gates`` is the launcher's directory of gate records; what they decided is
    written to ``launch.csv`` and named in the summary.

    ``dropped`` names variants that are left out of every table although they may have
    results: rows whose own selfcheck failed without blocking their method
    (``variants.dropped_variants``). They are listed in ``dropped.csv`` and in the
    summary, so a row is never missing without a reason.

    ``with_tissue`` adds the tissue-level block of the written images (``tissue*.csv``,
    ``regbench.tissue``): secondary, its own tables, never part of the ones above.

    ``expected`` lists variants that must appear even with no directory. ``primary`` lists
    the primary variants: they come first in the summary and define the ``common`` block.
    ``seam_owners`` (default ``SEAM_OWNERS``) names the variants whose seam mask gives
    every variant a second, off-seam row (``nonband_*``).
    """
    owners = tuple(SEAM_OWNERS if seam_owners is None else seam_owners)
    dropped = list(dict.fromkeys(dropped or []))
    dropped_rows = [
        {
            "variant": v,
            "reason": DROPPED_REASON,
            "had_results": (Path(out) / v).is_dir(),
        }
        for v in dropped
    ]
    variants_of = {}
    jobs, hashes, lists, prepared, folded, reseeded = [], {}, {}, set(), [], []
    for ds in datasets or DATASETS:
        cs = [
            c for c in list_cases(cases_root, ds) if c.case_id.startswith("dev_") == dev
        ]
        if cs:
            variants = find_variants(out, ds, expected, dropped)
            variants_of[ds] = variants
            jobs += [
                (str(out), c, variants, owners, dict(row_checks or {})) for c in cs
            ]
            for c in cs:
                if c.extra.get("truth_spec_sha256"):
                    hashes.setdefault(c.extra["truth_spec_sha256"], f"{ds}/{c.case_id}")
                if (c.extra.get("field") or {}).get("fold_pct", 0) > 0:
                    folded.append(f"{ds}/{c.case_id}")
                skipped = (c.extra.get("reseed") or {}).get("skipped") or []
                if skipped:
                    reseeded.append(f"{ds}/{c.case_id} ({len(skipped)} seed)")
                if c.extra.get("truth_patients_sha256"):
                    lists.setdefault(
                        (ds, c.extra["truth_patients_sha256"]), f"{ds}/{c.case_id}"
                    )
                    prepared.add(str(c.extra.get("truth_patient", "")))
    if not jobs:
        raise SystemExit(f"no prepared cases under {cases_root}")
    if len(hashes) > 1:
        raise SystemExit(
            "the known-truth cases were made from more than one truth_spec.json: "
            + "; ".join(f"{h[:12]} (e.g. {c})" for h, c in hashes.items())
            + ". Score them separately; a changed spec is a post hoc change."
        )
    if len(lists) > 1:
        raise SystemExit(
            "the semi-synthetic known-truth cases were built from more than one "
            "patient list: "
            + "; ".join(f"{h[:12]} (e.g. {c})" for (_, h), c in lists.items())
            + ". A result rests on one declared list of patients; score them "
            "separately."
        )
    patients = None
    for ds, sha in lists:
        patients = truth.load_patients(truth.patients_path(cases_root, ds))
        if patients[truth.HASH_KEY] != sha:
            raise SystemExit(
                f"{truth.patients_path(cases_root, ds)} (list "
                f"{patients[truth.HASH_KEY][:12]}: {', '.join(patients['patients'])}) "
                f"does not match the list the cases were built from ({sha[:12]}, "
                f"e.g. {lists[(ds, sha)]})."
            )
    if workers > 1 and len(jobs) > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            done = pool.map(_score_case, jobs)
    else:
        done = [_score_case(j) for j in jobs]
    cells = [r for rows in done for r in rows[0]]
    res = [r for rows in done for r in rows[1]]
    truth_rows = [r for rows in done for r in rows[2]]
    # off the seams: tables of their own, never merged into the ones above
    non_cells = [r for rows in done for r in rows[3]]
    non_truth = [r for rows in done for r in rows[4]]
    non_lost = nonband_missing(
        [r for rows in done for r in rows[5]], jobs, owners, variants_of
    )
    launch = read_gates(gates)
    blocked = blocked_rows(launch, run_variants)
    non_summary = _in_set(summarise_nonband(non_cells, primary, blocked), OUTLINES)
    by_family = truth_by_family(truth_rows)
    summary = _in_set(summarise_cells(cells, primary, blocked), OUTLINES)
    common = _in_set(common_block(cells, primary, blocked), OUTLINES)
    excluded = excluded_rounds(cells)
    paired = _in_set(paired_vs_primary(cells), OUTLINES)
    paired_summary = _in_set(summarise_paired(paired), OUTLINES)
    # the same tables on the BARE NUCLEI: tables of their own (``<name>_nuclei.csv``),
    # same columns, read with the nuclei's own gate, null and shared subset. A case
    # without nuclei has no row there and is listed with its reason, never imputed.
    nuc = {
        "cells": [r for rows in done for r in rows[6]["cells"]],
        "nonband_cells": [r for rows in done for r in rows[6]["non_cells"]],
    }
    nuclei_missing = [r for rows in done for r in rows[6]["missing"]]
    nuc["summary_cells"] = _in_set(
        summarise_cells(nuc["cells"], primary, blocked), NUCLEI
    )
    nuc["common_block"] = _in_set(common_block(nuc["cells"], primary, blocked), NUCLEI)
    nuc["paired_vs_primary"] = _in_set(paired_vs_primary(nuc["cells"]), NUCLEI)
    nuc["paired_summary"] = _in_set(summarise_paired(nuc["paired_vs_primary"]), NUCLEI)
    nuc["nonband_summary"] = _in_set(
        summarise_nonband(nuc["nonband_cells"], primary, blocked), NUCLEI
    )
    fails = failures(res)  # every run, the frozen known-truth cases included
    # cost is read on the cohort: the frozen known-truth cases are left out of it
    frozen = [bool(job[1].extra.get("truth_spec_sha256")) for job in jobs]
    res = [r for rows, skip in zip(done, frozen) if not skip for r in rows[1]]
    res_summary = summarise_resources(res)
    cost_summary = summarise_cost(res_summary, primary)
    confined = cost.cores_confined(res)
    cost_header = cost.header_lines(res)
    steps, source = None, ""
    if pipeline_trace:
        convert, merge = pipeline_processes
        steps = cost.pipeline_steps(
            cost.read_trace(pipeline_trace),
            tuple(convert or cost.PIPELINE_CONVERT),
            tuple(merge or cost.PIPELINE_MERGE),
        )
        trace_path = Path(pipeline_trace)
        label = pipeline_trace_label or trace_path.resolve().parent.name
        source = f"{label} ({trace_path})"
    e2e = cost.end_to_end(res, steps, source)
    e2e_summary = cost.summarise(
        e2e,
        {ds: got[0] for ds, got in common_patients(res_summary, primary).items()},
        {(r["dataset"], r["method"]): r["n_patients"] for r in cost_summary},
    )

    tables = Path(out) / ("tables_dev" if dev else "tables")
    tables.mkdir(parents=True, exist_ok=True)
    for name in (
        "cells",
        "summary_cells",
        "paired_vs_primary",
        "failures",
        "resources",
        "summary_resources",
        "truth",
        "truth_by_family",
        "nonband_cells",
        "nonband_summary",
        "nonband_truth",
        "nonband_truth_by_family",
        "nonband_missing",
        "cost_end_to_end",
        "cost_end_to_end_summary",
        "launch",
        "common_block",
        "paired_summary",
        "excluded_rounds",
        "cost_summary",
        "failure_log",
        "nuclei_missing",
        *(f"{name}{NUCLEI_SUFFIX}" for name in NUCLEI_TABLES),
        *tissue.TABLES,
    ):
        (tables / f"{name}.csv").unlink(missing_ok=True)
    for name in NUCLEI_TABLES:
        write_csv(tables / f"{name}{NUCLEI_SUFFIX}.csv", nuc[name])
    write_csv(tables / "nuclei_missing.csv", nuclei_missing)
    write_csv(tables / "cells.csv", cells)
    write_csv(tables / "summary_cells.csv", summary)
    write_csv(tables / "paired_vs_primary.csv", paired)
    write_csv(tables / "failures.csv", fails)
    write_csv(tables / "resources.csv", res)
    write_csv(tables / "summary_resources.csv", res_summary)
    write_csv(tables / "cost_summary.csv", cost_summary)
    write_csv(tables / "common_block.csv", common)
    write_csv(tables / "paired_summary.csv", paired_summary)
    write_csv(tables / "excluded_rounds.csv", excluded)
    write_csv(tables / "cost_end_to_end.csv", e2e, list(cost.COLUMNS))
    write_csv(tables / "cost_end_to_end_summary.csv", e2e_summary)
    write_csv(tables / "launch.csv", launch)
    write_csv(tables / "truth.csv", truth_rows)
    write_csv(tables / "truth_by_family.csv", by_family)
    write_csv(tables / "nonband_cells.csv", non_cells)
    write_csv(tables / "nonband_summary.csv", non_summary)
    write_csv(tables / "nonband_truth.csv", non_truth)
    write_csv(tables / "nonband_truth_by_family.csv", truth_by_family(non_truth))
    write_csv(tables / "nonband_missing.csv", non_lost)
    (tables / "dropped.csv").unlink(missing_ok=True)  # never a stale list
    write_csv(tables / "dropped.csv", dropped_rows)
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
    known = truth_header(list(hashes), patients, prepared, folded, reseeded)
    text = summary_markdown(
        summary,
        fails,
        res_summary,
        primary,
        by_family,
        known,
        non_summary,
        non_lost,
        tis,
        dropped_rows,
        confined,
        (e2e, e2e_summary, source),
        launch,
        cost_header,
        common,
        paired_summary,
        excluded,
        events,
        cost_summary,
        (nuc, nuclei_missing, len(jobs)),
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
        "truth": truth_rows,
        "truth_by_family": by_family,
        "truth_patients": patients,
        "nonband_cells": non_cells,
        "nonband_summary": non_summary,
        "nonband_truth": non_truth,
        "nonband_missing": non_lost,
        "dropped": dropped_rows,
        "cost_end_to_end": e2e,
        "cost_end_to_end_summary": e2e_summary,
        "cores_confined": confined,
        "cost_header": cost_header,
        "launch": launch,
        **{f"{name}{NUCLEI_SUFFIX}": rows for name, rows in nuc.items()},
        "nuclei_missing": nuclei_missing,
        **tis,
    }
