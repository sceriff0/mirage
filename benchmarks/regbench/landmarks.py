"""Score HyReCo (``hyreco``), a public landmark set: its own tables, never
the cohort's; and the per-pair landmark error.

Restored from commit 5a46f778 (``benchmarks/regbench/landmarks.py``). What changed:
the landmarks are the case's ``truth.npz`` (``regbench.landmark_io``), moved by the
driver into ``warped.npz: truth_xy`` on the side ``truth_side`` names; a result whose
``run.json: truth_sha256`` is not the case's sidecar as it is now was made for other
landmarks and is ``not_run`` (``notrun.STALE_TRUTH``: an integrity refusal, a
``stale_result`` row of ``losses.csv``); the tuning lock (subset A chooses, subset B
reports only with A's settings) is REMOVED (ruling 2026-10-18) and the two subsets are
ONE dataset (ruling 2026-10-19): subset A and, when it was downloaded
(``download/hyreco.sh`` fetches it by default; ``HYRECO_ADDITIONAL=0`` skips it), subset
B, in one cases folder, every pair scored by every run
of the list, the four ids both subsets list kept in both and flagged (``shared_id``);
no settings file gates anything; nothing
blocks (2026-10-18): a pair whose landmarks fail a sanity check
(``datasets/hyreco.py``: ``extra.flags``) is scored and FLAGGED (column ``flags``, a count
in the summary, a ``landmark_flagged`` row of ``losses.csv``), never removed.

``score(cases_root, out, dataset)`` reads the same ``warped.npz`` / ``run.json`` as the
cohort's scorer and writes ``<out>/tables/<dataset>/``. Nothing here enters
``<out>/tables/*.csv``, the cohort's summary, a paired comparison or a ranking.

**The error.** A case holds paired landmarks (the moving PHH3 slide's and the reference
H&E slide's, full-resolution pixels). A method's map has moved the landmarks of the side
it starts from; the error of a landmark is the distance to its partner on the other side,
times the pixel size: target registration error in micrometres, in the method's own
direction (``regbench.truth.error_stats``). A point a method could not map has an
infinite error. A run that failed is scored at the unregistered pose
(``imputed_initial``) and counts in every pooled number; a pair never registered for a
reason that is not the method's is in none (``n_not_run``, ``hyreco_not_run.csv``).

**Two aggregations, labelled apart** (``hyreco_summary.csv``):

``per_pair_*``  each pair's median error, then the median, minimum and maximum over pairs.
``pooled_*``    one median over ALL landmarks of all pairs of the set: the definition of
                Lotz et al. 2023 ("MTRE": the median "over all N_images image pairs and
                over all available landmarks"). It weights a pair by its landmark count.

Each pair row also carries the relative numbers (``rtre_*``, robustness: ``pair_stats``),
so the paper's ``public_landmarks`` reads one set of columns for every landmark set. Their
definitions are BIRL's (Borda/BIRL at ``f1648b29`` ``[PARTIAL: read
birl/utilities/evaluate.py:1-123 (compute_tre, compute_target_regist_error_statistic,
compute_tre_robustness) and the repository's evaluate_submission.py:225-345
(compute_scores); the rest of the repository not read; read for Task 21, not re-read
since]``): rTRE is TRE over the diagonal of the target (reference) image, robustness the
share of a pair's landmarks whose TRE after registration is strictly below their initial
TRE.

**Sets** (column ``set``). ``all_pairs`` (every prepared pair), ``subset_a`` and
``subset_b`` (each subset's pairs: the sets the published numbers were computed on). The
four ids both subsets list (29, 108, 361, 464) are in both subsets' sets, flagged
(``shared_id``, ``n_shared_id``): whether they are the same slides is stated by no
source read.

**Published numbers** (``PUBLISHED``, ``hyreco_published.csv``) are quoted, not
reproduced: neither method was run here. Only a number whose aggregation is defined and
whose subset is stated is in the table; CORE's 0.41 um ("AMTRE", aggregation undefined,
subset unstated) is deliberately absent (ruling R3.12 c).

The tissue-level block and the off-seam rows are not computed for these cases
(``SKIPPED``).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import landmark_io, losses, truth, variants
from . import score as cohort
from .cases import list_cases
from .mapping import FRAME, MOV_TO_REF
from .notrun import (
    NOT_RUN_CAUSES,
    STALE_TRUTH,
    UNKNOWN_FRAME,
    not_run_rows,
    settle_frames,
)

HYRECO = "hyreco"
HYRECO_DATASETS = (HYRECO,)
LOTZ = (
    "Lotz et al. 2023, J Med Imaging 10(6):067501, doi:10.1117/1.JMI.10.6.067501 "
    "[PARTIAL: main text searched and its results paragraphs read in the Europe PMC "
    "copy; figures and supplement not seen]"
)
NOT_REPRODUCED = "published, not reproduced here"
# One row per published number that may stand beside this benchmark's own: the method's
# authors' value, with the aggregation and the subset it was computed on.
PUBLISHED = (
    {
        "dataset": HYRECO,
        "set": "subset_b",
        "method": "HistokatFusion, deformable (Fraunhofer MEVIS)",
        "n_pairs": 54,
        "n_landmarks": 2357,
        "pooled_median_tre_um": 0.9,
        "status": NOT_REPRODUCED,
        "conditions": "best results at a pixel size <= 1.9 um; parameters set on "
        "ANHIR, not tuned on HyReCo; licensed software, no public code",
        "source": LOTZ,
    },
    {
        "dataset": HYRECO,
        "set": "subset_b",
        "method": "HistokatFusion, affine only (Fraunhofer MEVIS)",
        "n_pairs": 54,
        "n_landmarks": 2357,
        "pooled_median_tre_um": 1.7,
        "status": NOT_REPRODUCED,
        "conditions": "as above",
        "source": LOTZ,
    },
    {
        "dataset": HYRECO,
        "set": "subset_a",
        "method": "HistokatFusion, deformable (Fraunhofer MEVIS)",
        "n_pairs": 9,
        "n_landmarks": 138,
        "pooled_median_tre_um": 2.2,
        "status": NOT_REPRODUCED,
        "conditions": "at 1.0 um/px; landmarks placed while viewing five sections at "
        "once",
        "source": LOTZ,
    },
)
# The annotation floor: what two observers disagree by on the same section.
ANNOTATION_FLOOR = {
    "inter_observer_um": 0.57,
    "inter_observer_sd_um": 0.36,
    "intra_observer_um": 0.53,
    "intra_observer_sd_um": 0.32,
    "basis": "about 20 landmarks annotated independently by two researchers, same "
    "section (mean +- standard deviation)",
    "source": LOTZ,
}
PUBLISHED_COUNTS = (
    "2303 on the dataset page and in arXiv:2106.13150v2, 2357 in the paper"
)
SKIPPED = {
    "tissue-level block": "it compares the nuclear channel of a written image with the "
    "reference's; a brightfield case has none, and the methods write different kinds "
    "of image (RGB, a one-channel proxy)",
    "off-seam rows": "they are defined on nuclei segmented on the slides and on the "
    "seams of ASHLAR and PALOM; these cases carry no nuclei and neither method runs "
    "on them",
}
TABLES = cohort.HYRECO_TABLES
NO_LANDMARKS = "the case has no truth.npz: it was prepared without its landmarks"


# ── one result, one pair ─────────────────────────────────────────────────────
def load_result(out, variant, case):
    """``(run, warped, problem, cause)`` as the cohort's tables read it
    (``score._load_result``), with the integrity refusal of a landmark set: a record
    whose ``truth_sha256`` is not the case's ``truth.npz`` now is ``not_run``
    (``STALE_TRUTH``), whatever it holds."""
    run, warped, problem, cause = cohort._load_result(out, variant, case)
    if run and cause not in ("not_run", cohort.UNREADABLE):
        if run.get("truth_sha256") != landmark_io.truth_sha256(case):
            return {}, None, STALE_TRUTH, "not_run"
    return run, warped, problem, cause


def landmark_errors(case, truth_pts, warped):
    """``(error px, initial px, mapped, source, target, direction)`` of one run on one
    case; ``warped`` None scores the unregistered pose. ``truth_pts`` is
    ``landmark_io.read_truth(case)``. The side moved is ``warped.npz: truth_side``
    (``mov`` under ``mov_to_ref``, ``ref`` under ``ref_to_mov``). Raises ValueError when
    the result does not fit the case's landmarks."""
    t_mov = np.asarray(truth_pts["mov"], dtype=float).reshape(-1, 2)
    t_ref = np.asarray(truth_pts["ref"], dtype=float).reshape(-1, 2)
    direction = str(warped["direction"]) if warped is not None else MOV_TO_REF
    if direction not in FRAME:
        raise ValueError(f"unknown direction {direction!r}")
    side = "mov" if direction == MOV_TO_REF else "ref"
    source, target = (t_mov, t_ref) if side == "mov" else (t_ref, t_mov)
    if warped is None:
        mapped = source
    else:
        if "truth_xy" not in warped:
            raise ValueError("warped.npz holds no moved landmarks (truth_xy)")
        said = str(warped["truth_side"]) if "truth_side" in warped else side
        if said != side:
            raise ValueError(
                f"warped.npz moved the {said} landmarks under {direction}: not its side"
            )
        mapped = np.asarray(warped["truth_xy"], dtype=float).reshape(-1, 2)
        if len(mapped) != len(source):
            raise ValueError("warped.npz does not fit the case's landmarks")
    bad = ~np.isfinite(mapped).all(axis=1)
    err = np.full(len(target), math.inf)
    err[~bad] = np.hypot(*(mapped[~bad] - target[~bad]).T)
    init = np.hypot(*(source - target).T)
    return err, init, mapped, source, target, direction


def pair_stats(err, init, diagonal):
    """The per-pair statistics from the errors of one pair's landmarks, in pixels (BIRL
    ``compute_scores``' definitions): ``err`` after registration (a
    non-finite entry = a landmark the method did not map: it stays at its unregistered
    position), ``init`` before."""
    err, init = np.asarray(err, float), np.asarray(init, float)
    bad = ~np.isfinite(err)
    err = np.where(bad, init, err)
    rtre = err / float(diagonal)
    return {
        "rtre_median": float(np.median(rtre)),
        "rtre_mean": float(np.mean(rtre)),
        "rtre_max": float(np.max(rtre)),
        "robustness": float(np.sum(err < init) / len(err)),
        "initial_rtre_median": float(np.median(init / float(diagonal))),
        "n_landmarks": int(len(err)),
        "n_nonfinite": int(bad.sum()),
    }


def flags_of(case):
    """The sanity flags the dataset module wrote (``extra.flags``), ``;``-joined."""
    return "; ".join(str(f) for f in case.extra.get("flags") or [])


def _resource(base, case, v, run):
    return {
        **base,
        "group": case.group,
        "patient_id": case.group,
        "unit": "pair",
        "method": v,
        **cohort.resource_row(run or {"cause": "not_run"}, cohort.moving_pixels(case)),
    }


def _score_case(job):
    out, case, names = job
    pts = landmark_io.read_truth(case)
    if pts is None or pts["ref"] is None or not len(pts["mov"]):
        raise ValueError(f"{case.dataset}/{case.case_id}: {NO_LANDMARKS}")
    px = float(case.pixel_size_um) if case.pixel_size_um else None
    base = {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "group": case.group,
        "subset": str(case.extra.get("subset", "")),
        "shared_id": bool(case.extra.get("shared_id")),
        "flags": flags_of(case),
        "flagged": bool(case.extra.get("flags")),
        "settings_sha256": case.extra.get("settings_sha256", ""),
    }
    pairs, pts_rows, res = [], [], []
    for v in names:
        if v == "initial":
            run, warped, problem, cause = {}, None, "", ""
        else:
            run, warped, problem, cause = load_result(out, v, case)
            res.append(_resource(base, case, v, run))
        try:
            err, init, mapped, source, target, direction = landmark_errors(
                case, pts, warped
            )
        except ValueError as exc:
            warped, problem, cause = None, str(exc), "exception"
            err, init, mapped, source, target, direction = landmark_errors(
                case, pts, None
            )
        imputed = v != "initial" and warped is None
        if imputed and not cause:
            cause = "exception"
        if imputed:
            # a failed pair has no map, hence no direction of its own: the run's record
            # says it when there is one, else `settle_frames` takes it from the row's
            # other pairs; never a default
            declared = str((run or {}).get("direction") or "")
            direction = declared if declared in FRAME else ""
        stats = pair_stats(err, init, case.diagonal)
        not_run = imputed and cause in NOT_RUN_CAUSES
        pairs.append(
            {
                **base,
                "method": v,
                "frame": FRAME.get(direction, ""),
                "failed": imputed,
                "not_run": not_run,
                **stats,
                **truth.error_stats(mapped, target, source, px),
                **truth.jacobian_stats(warped),
                "pixel_size_um": px if px is not None else math.nan,
                "pixel_size_source": case.extra.get("pixel_size_source", ""),
                "imputed_initial": imputed,
                "excluded_from_summaries": not_run,
                "problem": str(problem)[:200],
                "cause": cause,
            }
        )
        if not_run:
            continue  # a pair never registered: none of its points is in a statistic
        pts_rows += [
            {
                **{k: base[k] for k in ("dataset", "case_id", "group")},
                "method": v,
                "landmark": i,
                "tre_um": float(e * px) if px is not None else math.nan,
                "initial_tre_um": float(s * px) if px is not None else math.nan,
                "imputed_initial": imputed,
            }
            for i, (e, s) in enumerate(zip(err, init))
        ]
    return pairs, pts_rows, res


def score_jobs(jobs, one, workers=1):
    """``one(job)`` of every job, in a spawned pool when ``workers`` > 1."""
    if workers > 1 and len(jobs) > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            return pool.map(one, jobs)
    return [one(j) for j in jobs]


# ── aggregation ──────────────────────────────────────────────────────────────
def _median(values):
    """The median of errors that may hold ``inf`` (a lost point is not hidden)."""
    v = np.sort(np.asarray(values, dtype=float))
    v = v[~np.isnan(v)]
    return truth._percentile(v, 50) if len(v) else math.nan


def _p90(values):
    v = np.sort(np.asarray(values, dtype=float))
    v = v[~np.isnan(v)]
    return truth._percentile(v, 90) if len(v) else math.nan


def sets_of(dataset):
    """``[(set name, keep(pair row))]`` of a dataset: all pairs, then each subset (a set
    with no pair has no row)."""
    return [
        ("all_pairs", lambda r: True),
        ("subset_a", lambda r: r["subset"] == "A"),
        ("subset_b", lambda r: r["subset"] == "B"),
    ]


def summarise(pairs, points):
    """One row per (dataset, set, method): the per-pair and the pooled numbers."""
    out = []
    for ds in dict.fromkeys(r["dataset"] for r in pairs):
        for name, keep in sets_of(ds):
            for m in dict.fromkeys(r["method"] for r in pairs):
                mine = [r for r in pairs if (r["dataset"], r["method"]) == (ds, m)]
                mine = [r for r in mine if keep(r)]
                n_not_run = sum(bool(r.get("not_run")) for r in mine)
                # a pair never registered for a reason that is not the method's is in
                # no statistic of it (`n_not_run` counts it)
                mine = [r for r in mine if not r.get("not_run")]
                if not mine:
                    continue
                ids = {r["case_id"] for r in mine}
                mp = [
                    p
                    for p in points
                    if (p["dataset"], p["method"]) == (ds, m) and p["case_id"] in ids
                ]
                per_pair = [r["tre_um_p50"] for r in mine]
                err = np.array([p["tre_um"] for p in mp], dtype=float)
                init = np.array([p["initial_tre_um"] for p in mp], dtype=float)
                finite_pp = [v for v in per_pair if not math.isnan(v)]
                out.append(
                    {
                        "dataset": ds,
                        "set": name,
                        "method": m,
                        "n_pairs": len(mine),
                        "n_ok": sum(not r["imputed_initial"] for r in mine),
                        "n_not_run": n_not_run,
                        "n_flagged": sum(bool(r.get("flagged")) for r in mine),
                        "n_shared_id": sum(bool(r.get("shared_id")) for r in mine),
                        "n_landmarks": len(mp),
                        "per_pair_median_tre_um_med": _median(per_pair),
                        "per_pair_median_tre_um_min": float(np.min(finite_pp))
                        if finite_pp
                        else math.nan,
                        "per_pair_median_tre_um_max": float(np.max(finite_pp))
                        if finite_pp
                        else math.nan,
                        "per_pair_p90_tre_um_med": _median(
                            [r["tre_um_p90"] for r in mine]
                        ),
                        "pooled_median_tre_um": _median(err),
                        "pooled_p90_tre_um": _p90(err),
                        "pooled_frac_under_1um": float(np.mean(err < 1.0))
                        if len(err)
                        else math.nan,
                        "pooled_frac_improved": float(np.mean(err < init))
                        if len(err)
                        else math.nan,
                        "initial_pooled_median_tre_um": _median(init),
                        "median_rtre_median": _median([r["rtre_median"] for r in mine]),
                        "avg_robustness": float(
                            np.mean([r["robustness"] for r in mine])
                        ),
                        "n_nonfinite": int(sum(r["n_nonfinite"] for r in mine)),
                        "fold_pct_max": cohort._stat(
                            [r["fold_pct"] for r in mine], np.max
                        ),
                    }
                )
    return out


def _table(rows, columns):
    return cohort.markdown(rows, columns)


SUMMARY_COLUMNS = (
    "method",
    "n_ok",
    "n_pairs",
    "n_not_run",
    "n_flagged",
    "n_shared_id",
    "n_landmarks",
    "initial_pooled_median_tre_um",
    "per_pair_median_tre_um_med",
    "per_pair_median_tre_um_min",
    "per_pair_median_tre_um_max",
    "pooled_median_tre_um",
    "pooled_p90_tre_um",
    "pooled_frac_under_1um",
)
LIKE_COLUMNS = (
    "method",
    "n_pairs",
    "n_landmarks",
    "pooled_median_tre_um",
    "status",
    "source",
)


def like_for_like(summary, dataset):
    """The rows of the like-for-like table: this benchmark's own pooled medians on the
    set a published number was computed on, then the published rows."""
    quoted = [dict(r) for r in PUBLISHED if r["dataset"] == dataset]
    wanted = {r["set"] for r in quoted}
    own = [
        {
            "method": r["method"],
            "n_pairs": r["n_pairs"],
            "n_landmarks": r["n_landmarks"],
            "pooled_median_tre_um": r["pooled_median_tre_um"],
            "status": f"run here ({r['n_ok']} of {r['n_pairs']} registered)",
            "source": "this benchmark",
        }
        for r in summary
        if r["dataset"] == dataset and r["set"] in wanted
    ]
    return own + quoted


SET_TITLE = {
    "all_pairs": "All prepared pairs (both subsets)",
    "subset_a": "Subset A (HyReCo)",
    "subset_b": "Subset B (HyReCo-Additional)",
}


def markdown(dataset, cases, summary, fails, primary, lost=(), frames=None):
    floor = ANNOTATION_FLOOR
    n_landmarks = sum(int(c.extra.get("n_landmarks", 0)) for c in cases)
    px_sources = sorted({str(c.extra.get("pixel_size_source", "")) for c in cases})
    complete = all(c.extra.get("subset_complete", True) for c in cases)
    settings = sorted({str(c.extra.get("settings_sha256", "")) for c in cases})
    flagged = [c for c in cases if c.extra.get("flags")]
    parts = [f"# HyReCo, {dataset}: a public landmark set\n"]
    parts.append(
        "A block of its own: these numbers are in none of the cohort's tables, in no "
        "paired comparison with them and in no ranking. Brightfield H&E / PHH3 pairs "
        "of the same section, scored by manual landmarks (target registration error, "
        "micrometres). The same landmarks for every method; a run that did not "
        "register is scored at the unregistered pose and counted.\n"
    )
    subsets = sorted({str(c.extra.get("subset", "")) for c in cases})
    shared = [c for c in cases if c.extra.get("shared_id")]
    parts.append(
        f"One dataset (ruling 2026-10-19): subset(s) {', '.join(subsets)} prepared "
        "(subset B is downloaded by default; HYRECO_ADDITIONAL=0 skips it). Nothing "
        "was tuned on "
        "it: every run of the list runs with its own fixed settings. **Pairs whose id "
        f"both subsets list: {len(shared)}** (column `shared_id`; kept in both, "
        "whether they are the same slides is stated by no source read).\n"
    )
    proxies = sorted(
        {str((c.extra.get("settings") or {}).get("proxy", "")) for c in cases}
    )
    parts.append(
        f"STARE's own one-channel proxy of each RGB slide: {', '.join(proxies)} "
        "(benchmark-side preprocessing; the STARE levels that name another proxy or "
        "STARE's own `--prep` say so in their names; VALIS and DeeperHistReg read the "
        "original RGB slides).\n"
    )
    if len(settings) > 1:
        parts.append(
            f"**The cases carry {len(settings)} different preparation settings** "
            f"({', '.join(s[:12] for s in settings)}): each pair row names its own "
            "(`settings_sha256`). Kept and scored, not pooled silently.\n"
        )
    parts.append(
        f"{len(cases)} pair(s), {n_landmarks} landmark pairs found in the files "
        f"(published counts for subset B: {PUBLISHED_COUNTS}). Pixel size from: "
        f"{'; '.join(px_sources)}.\n"
    )
    if any(s.startswith("paper") for s in px_sources):
        parts.append(
            "**The pixel size of at least one pair is the paper's two-digit 0.24 um, "
            "not a value from the file.** Landmarks are given in millimetres, so a 1 % "
            "error in pixel size misplaces a landmark by 1 % of its distance from the "
            "image origin. Check before reading any number below.\n"
        )
    parts.append(
        f"**Flagged pairs: {len(flagged)}** (a landmark sanity check failed; kept and "
        "scored, column `flags` of `hyreco_pairs.csv`, a `landmark_flagged` row of "
        "`losses.csv` each)"
        + (
            ": " + "; ".join(f"{c.case_id}: {flags_of(c)}" for c in flagged)
            if flagged
            else ""
        )
        + ".\n"
    )
    if not complete:
        parts.append(
            "**INCOMPLETE SUBSET** (`--allow-incomplete`): not every pair of the "
            "subset was prepared. No number here is the subset's.\n"
        )
    parts.append(
        f"Annotation floor: two observers differ by {floor['inter_observer_um']} +- "
        f"{floor['inter_observer_sd_um']} um on the same section (intra-observer "
        f"{floor['intra_observer_um']} +- {floor['intra_observer_sd_um']} um; "
        f"{floor['basis']}; {floor['source']}). A difference between two methods "
        "below about 0.5 um is annotation noise: no ranking is claimed.\n"
    )
    if lost:
        parts.append("## Not run, for a reason that is not the method's\n")
        parts.append(
            "These pairs were never registered: the task never started, the attempt "
            "was lost to the scheduler or to the benchmark's own step, or the result "
            "was made for other landmarks (`hyreco_not_run.csv`). They are in NO "
            "median below (`n_not_run`); a row with no other pair is in no table at "
            "all. A pair the METHOD failed on is different: it counts at its "
            "unregistered landmarks (`n_ok` is less than `n_pairs`).\n"
        )
        parts += [_table(list(lost), list(lost[0])), ""]
    unknown = sorted(m for m, f in (frames or {}).items() if f == [UNKNOWN_FRAME])
    if unknown:
        parts.append(
            "No pair of " + ", ".join(f"`{m}`" for m in unknown) + " registered, so "
            "the frame its map is in is unknown (a failed pair has no map).\n"
        )
    order = list(primary or [])
    for name, _ in sets_of(dataset):
        block = [r for r in summary if r["set"] == name]
        block.sort(
            key=lambda r: (
                order.index(r["method"]) if r["method"] in order else len(order)
            )
        )
        parts += [f"## {SET_TITLE[name]}\n"]
        parts.append(
            "`per_pair_*`: each pair's median error, then the median / minimum / "
            "maximum over pairs. `pooled_*`: one median over all landmarks of all "
            "pairs of the set. The two are different aggregations.\n"
        )
        parts += [_table(block, list(SUMMARY_COLUMNS)), ""]
    like = like_for_like(summary, dataset)
    if any(r["status"] == NOT_REPRODUCED for r in like):
        short = [
            f"{q['set']}: {own} of {q['n_pairs']}"
            for q in PUBLISHED
            if q["dataset"] == dataset
            for own in [
                max((r["n_pairs"] for r in summary if r["set"] == q["set"]), default=0)
            ]
            if own != q["n_pairs"]
        ]
        parts.append("## Like for like: pooled median over all landmarks (MTRE)\n")
        parts.append(
            "The definition of Lotz et al. 2023: one median over all landmarks of all "
            "pairs. The rows marked `" + NOT_REPRODUCED + "` are quoted from the "
            "method's authors; that software was not run and its number could not be "
            "checked here. Its resolution and parameters are its authors' "
            "(`hyreco_published.csv`, column `conditions`).\n"
        )
        if short:
            parts.append(
                "**NOT like for like where fewer pairs are scored here than were "
                f"published** ({'; '.join(dict.fromkeys(short))}).\n"
            )
        parts += [_table(like, list(LIKE_COLUMNS)), ""]
    parts.append("\n## Not computed for these cases\n")
    parts += [f"- {k}: {v}" for k, v in SKIPPED.items()]
    if fails:
        parts += ["\n## Failures (launches and pairs)\n"]
        parts += [_table(fails, list(fails[0])), ""]
    return "\n".join(parts) + "\n"


# ── losses ───────────────────────────────────────────────────────────────────
def loss_rows(out, cases_root, dataset, cases, pairs):
    """Every row of the set's ``losses.csv``: the pairs ``prepare`` refused
    (``prepare_dropped``), every flagged pair (``landmark_flagged``), every pair a run
    did not register (the cohort's ``_run_loss``: imputed, not run, stale) and the
    per-stage tables the cohort's scorer reads (``score.loss_rows``)."""
    got = cohort.loss_rows(
        out, cases_root, [dataset], None, [], [], [], [], [], run_level=False
    )
    p_rows, where = losses.read_prepare_dropped(cases_root)
    got += [
        r
        for r in losses.prepare_rows(p_rows, cohort._shown(where, out, cases_root))
        if r["dataset"] == dataset
    ]
    for c in cases:
        if c.extra.get("flags"):
            got.append(
                losses.row(
                    dataset=dataset,
                    patient_id=c.group,
                    case_id=str(c.case_id),
                    cause="landmark_flagged",
                    effect="flagged",
                    detail=flags_of(c)[:300],
                    source=f"{dataset}/{c.case_id}/case.json",
                )
            )
    of_run = {v: run for run, vs in variants.run_variants().items() for v in vs}
    rows = []
    for r in pairs:
        if not r["failed"] or r["method"] == "initial":
            continue
        rr = dict(r)
        if rr["cause"] == "not_run" and str(rr["problem"]).startswith(STALE_TRUTH[:40]):
            rr["cause"], rr["problem"] = "stale_result", rr["problem"]
        rows.append(cohort._run_loss(
            {**rr, "excluded_from_summaries": bool(r.get("not_run"))}, of_run))  # fmt: skip
    return got + cohort._fold_not_run(rows)


def write_tables(tables, named, summary_text, lost, n_cases, n_scored, dataset):
    """Write the set's tables (``named``: ``{table: rows}``), ``excluded_runs.csv``,
    ``losses.csv`` / ``README.txt`` / ``summary.json`` and ``summary.md`` (``summary_text``
    with the runs left out and the loss summary)."""
    tables = Path(tables)
    tables.mkdir(parents=True, exist_ok=True)
    for name in (*cohort.TABLES, *cohort.REMOVED_TABLES):
        (tables / f"{name}.csv").unlink(missing_ok=True)
    left_out = cohort.write_excluded_runs(tables, [dataset])
    for name, rows in named.items():
        cohort.write_csv(tables / f"{name}.csv", rows)
    block = cohort._write_losses_block(tables, lost, n_cases, n_scored)
    text = summary_text + cohort.excluded_markdown(left_out)
    (tables / "summary.md").write_text(text + "\n" + losses.summary_text(lost))
    print(text)
    print(f"tables -> {tables}")
    return left_out, block


def score(
    cases_root, out, dataset, workers=1, expected=None, primary=None, tables=None
):
    """Score one HyReCo set; writes ``tables`` (default ``<out>/tables/<dataset>/``) and
    returns the rows. ``expected`` and ``primary`` are
    ``variants.expected_variants(dataset)`` and ``variants.primary_for(dataset)`` (None
    scores only what has results)."""
    if dataset not in HYRECO_DATASETS:
        raise SystemExit(f"{dataset!r} is not a HyReCo set {HYRECO_DATASETS}")
    tables = Path(tables) if tables else cohort.tables_dir(out, dataset)
    cases = list_cases(cases_root, dataset)
    if not cases:
        lost = loss_rows(out, cases_root, dataset, [], [])
        return {**cohort.nothing_scored(
            tables, lost, f"no prepared {dataset} cases under {cases_root}"),
            "tables": str(tables)}  # fmt: skip
    names = cohort.find_variants(out, dataset, expected)
    done = score_jobs([(str(out), c, names) for c in cases], _score_case, workers)
    pairs = [r for d in done for r in d[0]]
    points = [r for d in done for r in d[1]]
    res = [r for d in done for r in d[2]]
    frames = settle_frames(pairs)
    lost_pairs = not_run_rows(pairs)
    summary = summarise(pairs, points)
    fails = cohort.failures(res)
    res_summary = cohort.summarise_resources(res)
    published = [dict(r) for r in PUBLISHED if r["dataset"] == dataset]
    lost = loss_rows(out, cases_root, dataset, cases, pairs)
    text = markdown(dataset, cases, summary, fails, primary, lost_pairs, frames)
    named = {
        "hyreco_pairs": pairs,
        "hyreco_points": points,
        "hyreco_summary": summary,
        "hyreco_published": published,
        "hyreco_failures": fails,
        "hyreco_resources": res,
        "hyreco_summary_resources": res_summary,
        "hyreco_not_run": lost_pairs,
    }
    excluded, block = write_tables(
        tables, named, text, lost, len(cases), len(cases), dataset
    )
    return {
        "pairs": pairs,
        "points": points,
        "summary": summary,
        "published": published,
        "failures": fails,
        "resources": res,
        "summary_resources": res_summary,
        "excluded": excluded,
        "not_run": lost_pairs,
        "frames": frames,
        "losses": lost,
        "n_scored_cases": block["n_scored_cases"],
        "tables": str(tables),
    }
