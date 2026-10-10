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

``dev_*`` cases are left out of every table unless ``dev`` is given.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from . import DATASETS, pairs, tissue, truth
from .cases import list_cases, load_points, polys, result_dir
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
CAUSES = ("exception", "killed", "oom", "timeout", "gpu_oom", "not_run")


def tier_of(case):
    if case.dataset == "multiplex":
        return "headline"
    if case.dataset == "semisynth" and case.family in HEADLINE_FAMILIES:
        return "headline"
    return "secondary"


jacobian_stats = truth.jacobian_stats


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
    previous = list(run.get("previous_attempts") or [])
    wasted = [(_f(p.get("wall_s")), _f(p.get("cpu_s"))) for p in previous]
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
        "rss_method": r.get("rss_method", ""),
        "host": run.get("host", ""),
        "cpu_model": run.get("cpu_model", ""),
        "blas": run.get("blas", ""),
    }


def find_variants(out, dataset, expected=None):
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


def _truth_rows(case, points, base):
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
                **truth.error_stats(mapped, target, source, case.pixel_size_um),
                **jac,
                "imputed_initial": imputed,
                "problem": problem[:200],
                "cause": cause,
            }
        )

    return add, rows


def _seam_masks(out, case, variants, owners, ref, mov, points):
    """``({owner: (keep_ref | None, keep_mov | None, keep_truth | None)}, [owners without
    a mask])`` for the owners that have a row on this dataset. The mask is read from the
    owner's own primary run and from nothing else: no other method's result can change it.

    ``keep_truth`` is one bool per known-truth probe (``_truth_keep``)."""
    masks, missing = {}, []
    for owner in owners:
        if owner not in variants:
            continue
        _, warped, _, _ = _load_run(out, owner, case)
        band = warped.get("band") if warped is not None else None
        band_ref = warped.get("band_ref") if warped is not None else None
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
            _truth_keep(points, keep_mov, warped.get("band_truth")),
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


def _score_case(job):
    out, case, variants, owners = job
    points = load_points(case)
    ref, mov = polys(points, "ref"), polys(points, "mov")
    px = case.pixel_size_um
    scorable = bool(ref is not None and mov is not None and ref.n_features)
    scorable = scorable and bool(mov.n_features)
    base = {
        "dataset": case.dataset,
        "tier": tier_of(case),
        "case_id": case.case_id,
        "group": case.group,
        "family": case.family,
        "pose": case.pose,
    }
    dens = density(ref, px) if scorable else {}
    cell_rows, res_rows, non_cells, non_truth = [], [], [], []
    add_truth, truth_rows = _truth_rows(case, points, base)
    masks, missing = (
        _seam_masks(out, case, variants, owners, ref, mov, points)
        if scorable
        else ({}, [])
    )
    unregistered = {}
    flat = {"fold_pct": 0.0, "sd_log_jac": 0.0}
    nan = {"fold_pct": math.nan, "sd_log_jac": math.nan}

    def at_initial():
        if not unregistered:
            unregistered.update(pairs.record(ref, mov, ref, mov, px))
        return unregistered

    def row(method, frame, status, rec, jac, imputed, problem, cause):
        return {
            **base,
            "method": method,
            "frame": frame,
            "selfcheck_status": status,
            **{k: rec[k] for k in pairs.RECORD_KEYS},
            **dens,
            **jac,
            "imputed_initial": imputed,
            "problem": problem[:200],
            "cause": cause,
        }

    def off_seams(v, frame, status, ref_in, mov_in, jac, imputed, problem, cause):
        """The same variant on the nuclei each owner's mask leaves, one row per owner."""
        for owner, (keep_ref, keep_mov, _) in masks.items():
            r_in = ref_in if keep_ref is None else _keep_features(ref_in, keep_ref)
            m_in = mov_in if keep_mov is None else _keep_features(mov_in, keep_mov)
            # the gate still comes from ALL native nuclei: one gate per case
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
            run, warped, problem, cause = _load_run(out, v, case)
            res_rows.append(
                {**base, "method": v, **resource_row(run or {"cause": "not_run"})}
            )
        failed = v != "initial" and warped is None
        if add_truth is not None:
            direction = str(warped["direction"]) if warped is not None else MOV_TO_REF
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
        direction = str(warped["direction"]) if warped is not None else MOV_TO_REF
        if warped is not None:
            moved_side = mov if direction == MOV_TO_REF else ref
            if direction not in FRAME or len(warped["cells_xy"]) != len(moved_side.xy):
                problem = "warped.npz does not fit the case's nuclei"
                warped, cause, direction = None, "exception", MOV_TO_REF
        imputed = v != "initial" and warped is None
        status = run.get("selfcheck_status", "") if run else ""
        if warped is None:
            cell_rows.append(
                row(v, "reference", status, at_initial(), flat, imputed, problem, cause)
            )
            off_seams(v, "reference", status, ref, mov, flat, imputed, problem, cause)
            continue
        xy = warped["cells_xy"].astype(float)
        if direction == REF_TO_MOV:
            ref_in, mov_in = pairs.with_xy(ref, xy), mov
        else:
            ref_in, mov_in = ref, pairs.with_xy(mov, xy)
        jac = jacobian_stats(warped)
        rec = pairs.record(ref, mov, ref_in, mov_in, px)
        frame = FRAME[direction]
        cell_rows.append(row(v, frame, status, rec, jac, False, problem, cause))
        off_seams(v, frame, status, ref_in, mov_in, jac, False, problem, cause)
    if add_truth is not None:
        # the true map itself: zero by construction, shown so the table has its floor
        add_truth(
            "truth", "reference", MOV_TO_REF, points["truth_ref"], nan, False, "", ""
        )
    if scorable and "cells_mov_xy_truth" in points:
        moved = pairs.with_xy(mov, points["cells_mov_xy_truth"].astype(float))
        rec = pairs.record(ref, mov, ref, moved, px)
        cell_rows.append(row("truth", "reference", "", rec, nan, False, "", ""))
        off_seams("truth", "reference", "", ref, moved, nan, False, "", "")
    lost = [{**base, "owner": owner} for owner in missing]
    return cell_rows, res_rows, truth_rows, non_cells, non_truth, lost


# ── summaries ─────────────────────────────────────────────────────────────────
def _values(rows, key):
    return np.array([_f(r.get(key)) for r in rows], dtype=float)


def _stat(values, fn):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(fn(v)) if v.size else math.nan


def _per_patient(rows, key):
    """``{patient: median over its rounds}``."""
    by = {}
    for r in rows:
        by.setdefault(r["group"], []).append(_f(r.get(key)))
    return {g: _stat(v, np.median) for g, v in by.items()}


def _methods(rows):
    return list(dict.fromkeys(r["method"] for r in rows))


def common_cases(rows, primary=None):
    """Case ids every primary variant registered (every run variant when none is named)."""
    judged = [r for r in rows if NON not in r["method"]]
    if primary:
        judged = [r for r in judged if r["method"] in primary]
    else:
        judged = [r for r in judged if r["method"] not in PSEUDO]
    failed = {r["case_id"] for r in judged if r["imputed_initial"]}
    return {r["case_id"] for r in rows} - failed


def summarise_cells(rows, primary=None):
    """One row per (dataset, method, subset): ``all`` and ``common``."""
    out = []
    for ds in sorted({r["dataset"] for r in rows}):
        of_ds = [r for r in rows if r["dataset"] == ds]
        common = common_cases(of_ds, primary)
        for subset in ("all", "common"):
            for m in _methods(of_ds):
                mine = [r for r in of_ds if r["method"] == m]
                if subset == "common":
                    mine = [r for r in mine if r["case_id"] in common]
                if not mine:
                    continue
                n_ok = sum(not r["imputed_initial"] for r in mine)
                rec = {
                    "dataset": ds,
                    "method": m,
                    "subset": subset,
                    "n_patients": len({r["group"] for r in mine}),
                    "n_cases": len(mine),
                    "n_ok": n_ok,
                }
                for h in HEADLINES:
                    per = list(_per_patient(mine, h).values())
                    rec[f"{h}_med"] = _stat(per, np.median)
                    rec[f"{h}_min"] = _stat(per, np.min)
                    rec[f"{h}_max"] = _stat(per, np.max)
                    rec[f"{h}_min_case"] = _stat(_values(mine, h), np.min)
                    if h in HAS_NULL:
                        null = list(_per_patient(mine, f"{h}_null").values())
                        rec[f"{h}_null_med"] = _stat(null, np.median)
                # Dice may be ranked only where pair fraction clears the null's maximum
                clear = [
                    r for r in mine if r["pair_fraction"] > r["pair_fraction_null_max"]
                ]
                per = list(_per_patient(clear, "dice_matched").values())
                rec["dice_matched_ranked_med"] = _stat(per, np.median)
                rec["n_dice_ranked"] = len(clear)
                # displacement is read only on the readable cases
                readable = [r for r in mine if r["displacement_readable"]]
                per = list(_per_patient(readable, "displacement_um_p50").values())
                rec["displacement_um_p50_med"] = _stat(per, np.median)
                rec["n_readable"] = len(readable)
                rec["readable_share"] = len(readable) / len(mine)
                rec["fold_pct_max"] = _stat(_values(mine, "fold_pct"), np.max)
                rec["coverage_med"] = _stat(_values(mine, "coverage"), np.median)
                out.append(rec)
    return out


def paired_vs_primary(rows, against=PAIRED_AGAINST):
    """Per patient and variant: each headline's patient median minus the ``against`` row's."""
    out = []
    for ds in sorted({r["dataset"] for r in rows}):
        of_ds = [r for r in rows if r["dataset"] == ds]
        ref = [r for r in of_ds if r["method"] == against]
        if not ref:
            continue
        base = {h: _per_patient(ref, h) for h in HEADLINES}
        for m in _methods(of_ds):
            if m == against:
                continue
            mine = [r for r in of_ds if r["method"] == m]
            per = {h: _per_patient(mine, h) for h in HEADLINES}
            for g in dict.fromkeys(r["group"] for r in mine):
                if g not in base[HEADLINES[0]]:
                    continue
                rec = {"dataset": ds, "group": g, "method": m, "against": against}
                rec["n_rounds"] = sum(r["group"] == g for r in mine)
                for h in HEADLINES:
                    rec[f"d_{h}"] = per[h][g] - base[h][g]
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
            rec["rounds_not_run"] = sum(r["cause"] == "not_run" for r in mine)
            rec["rounds_lost"] = sum(not r["ok"] for r in mine)
            out.append(rec)
    return out


def summarise_resources(res_rows):
    """Per (dataset, patient, method). A patient's cost is the sum over its rounds for a
    pair method and the launch's own total for a per-patient one; the peak is the maximum
    and is never divided by rounds. Only completed runs are a cost. ``wasted_*`` is the
    earlier attempts of the ladder plus a last attempt that failed outright."""
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
            "attempt": _stat(_values(mine, "attempt"), np.max),
            "mem_requested_gb": _stat(_values(mine, "mem_requested_gb"), np.max),
            "cpus_requested": _stat(_values(mine, "cpus_requested"), np.max),
        }
        for key in ("register_s", "cpu_h", "gpu_h", "total_to_image_s"):
            rec[key] = _stat(_values(ok, key), total)
        rec["write_s"] = _stat(_values(ok, "write_s"), np.sum)
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
        out.append(rec)
    return out


# ── output ────────────────────────────────────────────────────────────────────
def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
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


def _summary_table(rows, primary):
    """Markdown rows: each headline as ``median [minimum over cases]``, its null beside it."""
    order = list(primary or [])
    rows = sorted(
        rows,
        key=lambda r: order.index(r["method"]) if r["method"] in order else len(order),
    )
    table = []
    for r in rows:
        line = {
            "method": r["method"],
            "patients": r["n_patients"],
            "ok/cases": f"{r['n_ok']}/{r['n_cases']}",
        }
        for h in HEADLINES:
            line[h] = f"{_fmt(r[f'{h}_med'])} [{_fmt(r[f'{h}_min_case'])}]"
            if h in HAS_NULL:
                line[f"{h}_null"] = r[f"{h}_null_med"]
        line["dice_ranked (n)"] = (
            f"{_fmt(r['dice_matched_ranked_med'])} ({r['n_dice_ranked']})"
        )
        line["displacement_um_p50 (readable n)"] = (
            f"{_fmt(r['displacement_um_p50_med'])} ({r['n_readable']})"
        )
        line["fold_pct_max"] = r["fold_pct_max"]
        table.append(line)
    return markdown(table, list(table[0])) if table else ""


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


def summarise_nonband(rows):
    """One row per (dataset, owner, variant): the summary of ``summarise_cells`` on the
    off-seam rows, with the nuclei kept. Never merged with the primary summary."""
    out = []
    for owner in dict.fromkeys(r["owner"] for r in rows):
        mine = [r for r in rows if r["owner"] == owner]
        for rec in summarise_cells(mine):
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
        "nuclei, chosen from the owner's geometry alone. These numbers are not "
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
            "kept, the same for every row. "
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
):
    parts = ["# Registration benchmark\n"]
    parts.append(
        "Nuclear overlap after registration, on nuclei segmented on the native slides. "
        "It is self-consistency on the registration channel, not accuracy. Each cell is "
        "`median over patients [minimum over cases]`; `ok/cases` counts the runs that "
        "registered, and a run that did not is scored at the unregistered pose.\n"
    )
    for ds in sorted({r["dataset"] for r in summary}):
        of_ds = [r for r in summary if r["dataset"] == ds]
        n_patients = max(r["n_patients"] for r in of_ds)
        parts.append(f"## {ds}\n")
        if n_patients < MIN_PATIENTS_TO_RANK:
            parts.append(f"{n_patients} patient(s): {DESCRIPTIVE}.\n")
        for subset, title in (
            ("all", "All cases"),
            ("common", "Cases every primary variant registered"),
        ):
            block = [r for r in of_ds if r["subset"] == subset]
            if block:
                parts += [f"### {title}\n", _summary_table(block, primary), ""]
    if by_family:
        parts += [
            "## Known-truth error, per deformation family\n",
            "Distance between where a method puts a probe and where the generating map "
            "says it belongs, in the method's own direction (micrometres). One row per "
            "family, pose and variant; families are never pooled. `in_stare_model` is "
            "False for the families a B-spline mesh on STARE's lattice cannot "
            "represent. A run that did not register is scored at the unregistered "
            "pose; a point a method could not map counts as an infinite error. The "
            "amplitudes are literature values as recorded in STARE's README; the "
            "families, their smoothness and the multiscale spectrum were chosen by "
            "STARE's authors.\n",
            *[line + "\n" for line in known_truth or []],
            markdown(by_family, list(TRUTH_COLUMNS)),
            "",
        ]
    parts += _nonband_markdown(nonband or [], nonband_lost or [])
    if tissue_tables:
        parts += tissue.markdown(tissue_tables, markdown)
    if fails:
        parts += ["## Failures (launches and rounds)\n"]
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
        ]
        parts += [
            "## Resources (per patient; registration only)\n",
            "`peak_rss_gb` is the process tree's peak while registering. "
            "`cgroup_peak_gb` is the whole job, writers included. No single column is "
            "the cost.\n",
            markdown(res_summary, cols),
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
):
    """Score everything under ``out``; writes ``<out>/tables`` and returns the rows.

    ``with_tissue`` adds the tissue-level block of the written images (``tissue*.csv``,
    ``regbench.tissue``): secondary, its own tables, never part of the ones above.

    ``expected`` lists variants that must appear even with no directory. ``primary`` lists
    the primary variants: they come first in the summary and define the ``common`` block.
    ``seam_owners`` (default ``SEAM_OWNERS``) names the variants whose seam mask gives
    every variant a second, off-seam row (``nonband_*``).
    """
    owners = tuple(SEAM_OWNERS if seam_owners is None else seam_owners)
    variants_of = {}
    jobs, hashes, lists, prepared, folded, reseeded = [], {}, {}, set(), [], []
    for ds in datasets or DATASETS:
        cs = [
            c for c in list_cases(cases_root, ds) if c.case_id.startswith("dev_") == dev
        ]
        if cs:
            variants = find_variants(out, ds, expected)
            variants_of[ds] = variants
            jobs += [(str(out), c, variants, owners) for c in cs]
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
    non_summary = summarise_nonband(non_cells)
    non_lost = nonband_missing(
        [r for rows in done for r in rows[5]], jobs, owners, variants_of
    )
    by_family = truth.by_family(truth_rows)
    summary = summarise_cells(cells, primary)
    paired = paired_vs_primary(cells)
    fails = failures(res)  # every run, the frozen known-truth cases included
    # cost is read on the cohort: the frozen known-truth cases are left out of it
    frozen = [bool(job[1].extra.get("truth_spec_sha256")) for job in jobs]
    res = [r for rows, skip in zip(done, frozen) if not skip for r in rows[1]]
    res_summary = summarise_resources(res)

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
        *tissue.TABLES,
    ):
        (tables / f"{name}.csv").unlink(missing_ok=True)
    write_csv(tables / "cells.csv", cells)
    write_csv(tables / "summary_cells.csv", summary)
    write_csv(tables / "paired_vs_primary.csv", paired)
    write_csv(tables / "failures.csv", fails)
    write_csv(tables / "resources.csv", res)
    write_csv(tables / "summary_resources.csv", res_summary)
    write_csv(tables / "truth.csv", truth_rows)
    write_csv(tables / "truth_by_family.csv", by_family)
    write_csv(tables / "nonband_cells.csv", non_cells)
    write_csv(tables / "nonband_summary.csv", non_summary)
    write_csv(tables / "nonband_truth.csv", non_truth)
    write_csv(tables / "nonband_truth_by_family.csv", truth.by_family(non_truth))
    write_csv(tables / "nonband_missing.csv", non_lost)
    # the written images against the reference: tables of their own, never ranked
    tis = (
        tissue.score_all([(j[0], j[1], j[2]) for j in jobs], workers)
        if with_tissue
        else {}
    )
    for name, rows in tis.items():
        write_csv(tables / f"{name}.csv", rows)
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
    )
    (tables / "summary.md").write_text(text)
    print(text)
    print(f"tables -> {tables}")
    return {
        "cells": cells,
        "summary_cells": summary,
        "paired_vs_primary": paired,
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
        **tis,
    }
