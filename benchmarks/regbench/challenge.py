"""Score ANHIR's training pairs (``anhir``, ``anhir_tune``): its own tables, never the cohort's.

``score(cases_root, out, dataset)`` reads the same ``warped.npz`` / ``run.json`` as the
cohort's scorer and writes ``<out>/tables/<dataset>/``. Nothing here enters
``<out>/tables/*.csv``, a paired comparison or a ranking of the cohort.

**What these numbers are, and are not.** They are computed on the challenge's TRAINING
pairs, whose landmarks are public. The challenge's leaderboard ranks methods on the
hidden evaluation pairs (and shows a training column beside it); nothing here is
comparable row for row with a leaderboard entry, nothing is exported for submission and
no rank among published methods is claimed. The ``stare_prep`` row's preprocessing was
chosen on the tuning side of this same training set (``regbench.prep``), so its
tuning-side numbers are fitted, and its report-side numbers are from sets it did not see
but from the same challenge data. STARE is built for re-imaged sections with the same
nuclei in both rounds; consecutive, differently stained sections at 1 to 9 um per pixel
are outside what it was designed for.

**The measures, per pair** (the challenge's Evaluation page ``[FULL: the saved copy of
https://anhir.grand-challenge.org/Performance_Metrics/ fetched 2026-10-10]``; the
evaluation code, Borda/BIRL at ``f1648b29`` ``[PARTIAL: read
birl/utilities/evaluate.py:1-123 (compute_tre, compute_target_regist_error_statistic,
compute_tre_robustness), bm_ANHIR/evaluate_submission.py:225-345 (compute_scores and its
two helpers) and :371-376 (the tissue name), birl/benchmark.py:755-801; the rest of the
repository not read]``):

``TRE``         the Euclidean distance between a landmark's target position and its
                position after registration, pixels.
``rTRE``        TRE divided by the diagonal of the target (reference) image,
                ``sqrt(w^2 + h^2)``. Per pair: its median, mean and maximum over the
                landmarks (``rtre_median``, ``rtre_mean``, ``rtre_max``).
``robustness``  the share of a pair's landmarks whose TRE after registration is strictly
                below their initial TRE (``compute_tre_robustness``).

**Aggregation, the official way** (``compute_scores``): over the pairs of a group, the
mean and the median of each per-pair statistic (BIRL's ``Average-Median-rTRE``,
``Median-Median-rTRE``, ``Average-Average-rTRE``, ``Median-Average-rTRE``,
``Average-Max-rTRE``, ``Median-Max-rTRE``, ``Average-Robustness``,
``Median-Robustness``), and the same over the "robust" pairs alone, those with robustness
above 0.5 (BIRL's ``-Robust`` columns). Groups: the split's side (``report``, ``tune``,
and ``all``), crossed with the tissue (BIRL groups by the text before the first
underscore of the set name) and with ``all`` tissues; per tissue SET in
``anhir_by_set.csv``. The challenge's ranking criterion, the average over pairs of a
method's RANK by median rTRE among all submitted methods, is not computed: it needs the
other methods' results on the same pairs (BIRL's own script leaves it NaN).

**Failures.** The Evaluation page: "All missing or incomplete registrations in the
submission are considered to have the initial rTRE". So in every official column a pair
that did not register (failed, killed, never run) has its landmarks at their unregistered
positions and robustness 0, and a landmark a method returned as non-finite is at its
unregistered position and not improved (BIRL itself would turn a NaN written into a
submission into 0; a submission written from these results would hold the unregistered
point instead, which is what is scored here). Failures are also shown apart: ``n_failed``,
``n_nonfinite``, the ``*_registered`` columns over the pairs that did register, and
``anhir_failures.csv``. The server's ``--min_landmarks`` threshold is not public and is
not applied: every pair has all its landmarks or none.

**Micrometres.** ``tre_um_*`` is the same error times the pair's pixel size, where the
pixel size is established (``datasets/anhir.py``: every tissue but ``mammary-gland``,
which is reported in pixels only and counted in ``n_pairs_um``). Unlike the official
columns, a non-finite landmark has an INFINITE error there (``regbench.truth``).

**Frame.** A method whose map goes moving -> reference (STARE) moves the source landmarks
into the target image: the challenge's own quantity. A method whose map goes reference
-> moving (VALIS, DeeperHistReg) moves the TARGET landmarks into the source image and is
compared with the source landmarks there: exact, with no inverted transform, but
measured in the source image's pixels (column ``frame``). Both are divided by the target
image's diagonal. Source and target differ in size in most pairs, so the two are close
but not the same quantity; the summary says which rows are which.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import ANHIR, ANHIR_DATASETS, ANHIR_TUNE, prep, resources, truth, variants
from . import score as cohort
from .cases import list_cases, load_points
from .landmarks import landmark_errors
from .mapping import FRAME, MOV_TO_REF

ROBUST_ABOVE = 0.5  # evaluate_submission.py:281
SIDES = ("report", "tune", "all")
SIDE_TITLE = {
    "report": "Report side (the sets the preprocessing choice never saw)",
    "tune": "Tuning side (the sets the preprocessing was chosen on)",
    "all": "All training pairs (both sides together)",
}
PAIR_STATS = ("rtre_median", "rtre_mean", "rtre_max", "robustness")
TABLES = (
    "anhir_pairs",
    "anhir_summary",
    "anhir_by_set",
    "anhir_failures",
    "anhir_resources",
    "anhir_summary_resources",
    "anhir_excluded",
    "anhir_prep_evidence",
    "anhir_not_run",
)
# A pair that was never registered for a reason that is not the method's: its run was
# blocked by a gate, its task never started, or the attempt was lost to the scheduler or
# to the benchmark's own step. Such a pair is in NO median of the method (it says
# nothing about it); it is counted (`n_not_run`) and listed (`anhir_not_run.csv`). A
# pair the method itself failed on counts at its unregistered landmarks, as the
# challenge counts a missing registration.
NOT_RUN_CAUSES = ("not_run", "blocked", *resources.NOT_THE_METHOD)
UNKNOWN_FRAME = "unknown (no pair of this row registered)"
SKIPPED = {
    "tissue-level block": "it compares the nuclear channel of a written image with the "
    "reference's; a brightfield case has none",
    "off-seam rows": "they are defined on nuclei and on the seams of ASHLAR and PALOM; "
    "these cases carry no nuclei and neither method runs on them",
    "rank among methods": "the challenge's ranking criterion needs every other "
    "method's result on the same pairs",
}
SUMMARY_COLUMNS = (
    "method",
    "n_pairs",
    "n_not_run",
    "n_failed",
    "median_rtre_median",
    "avg_rtre_median",
    "avg_rtre_max",
    "avg_robustness",
    "n_robust",
    "median_rtre_median_registered",
    "n_pairs_um",
    "median_tre_um_p50",
    "median_tre_px_p50",
)


def pair_stats(err, init, diagonal):
    """The challenge's per-pair statistics from the errors of one pair's landmarks, in
    pixels: ``err`` after registration (a non-finite entry = a landmark the method did
    not map: it stays at its unregistered position), ``init`` before."""
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


def _score_case(job):
    out, case, names = job
    points = load_points(case)
    if "truth_mov" not in points or not len(points["truth_mov"]):
        raise SystemExit(f"{case.dataset}/{case.case_id}: the case has no landmarks")
    px = float(case.pixel_size_um) if case.pixel_size_um else None
    base = {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "side": case.extra.get("side", ""),
        "tissue": case.extra.get("tissue", ""),
        "set": case.extra.get("set", ""),
        "pair": f"{case.extra.get('stain_mov', '')} -> {case.extra.get('stain_ref', '')}",
    }
    pairs, res = [], []
    for v in names:
        if v == "initial":
            run, warped, problem, cause = {}, None, "", ""
        else:
            run, warped, problem, cause = cohort._load_run(out, v, case)
            res.append(
                {
                    **base,
                    "group": case.group,
                    "method": v,
                    **cohort.resource_row(run or {"cause": "not_run"}),
                }
            )
        try:
            err, init, mapped, source, target, direction = landmark_errors(
                case, points, warped
            )
        except ValueError as exc:
            warped, problem, cause = None, str(exc), "exception"
            err, init, mapped, source, target, direction = landmark_errors(
                case, points, None
            )
        failed = v != "initial" and warped is None
        if failed and not cause:
            cause = "exception"
        # a failed pair has no map, hence no direction of its own: the run's record
        # says it when there is one, else `score` takes it from the row's other pairs
        declared = str((run or {}).get("direction") or "")
        if failed:
            direction = declared if declared in FRAME else ""
        stats = pair_stats(err, init, case.diagonal)
        if failed:  # the official rule: the initial rTRE, and no landmark improved
            stats["robustness"] = 0.0
        um = truth.error_stats(mapped, target, source, px)
        pairs.append(
            {
                **base,
                "method": v,
                "frame": FRAME.get(direction, ""),
                "challenge_quantity": direction == MOV_TO_REF,
                "not_run": failed and cause in NOT_RUN_CAUSES,
                **stats,
                "tre_px_p50": float(np.median(np.where(np.isfinite(err), err, init))),
                "tre_um_p50": um["tre_um_p50"],
                "tre_um_p90": um["tre_um_p90"],
                "initial_tre_um_p50": um["initial_tre_um_p50"],
                **truth.jacobian_stats(warped),
                "diagonal_px": float(case.diagonal),
                "pixel_size_um": px if px is not None else math.nan,
                "pixel_size_established": bool(
                    case.extra.get("pixel_size_established")
                ),
                "failed": failed,
                "problem": str(problem)[:200],
                "cause": cause,
            }
        )
    return pairs, res


def _mean(v):
    return float(np.mean(v)) if len(v) else math.nan


def _median(v):
    return float(np.median(v)) if len(v) else math.nan


def aggregate(rows):
    """BIRL's aggregates over one group of per-pair rows (module docstring). Pairs that
    were never registered for a reason that is not the method's (``not_run``) are in no
    statistic: ``n_not_run`` counts them, ``n_pairs`` is what is left."""
    n_not_run = sum(bool(r.get("not_run")) for r in rows)
    rows = [r for r in rows if not r.get("not_run")]
    robust = [r for r in rows if r["robustness"] > ROBUST_ABOVE]
    registered = [r for r in rows if not r["failed"]]
    # NaN = no pixel size for the pair. An INFINITE median (most landmarks lost) is a
    # result: it counts, and is the worst there is.
    with_um = [r for r in rows if not math.isnan(r["tre_um_p50"])]
    out = {
        "n_pairs": len(rows),
        "n_not_run": n_not_run,
        "n_failed": len(rows) - len(registered),
        "n_landmarks": int(sum(r["n_landmarks"] for r in rows)),
        "n_nonfinite": int(sum(r["n_nonfinite"] for r in rows)),
        "n_robust": len(robust),
    }
    for stat in PAIR_STATS:
        out[f"avg_{stat}"] = _mean([r[stat] for r in rows])
        out[f"median_{stat}"] = _median([r[stat] for r in rows])
    for stat in ("rtre_median", "rtre_max"):
        out[f"avg_{stat}_robust"] = _mean([r[stat] for r in robust])
        out[f"median_{stat}_robust"] = _median([r[stat] for r in robust])
    out["median_rtre_median_registered"] = _median(
        [r["rtre_median"] for r in registered]
    )
    out["avg_robustness_registered"] = _mean([r["robustness"] for r in registered])
    out["median_initial_rtre_median"] = _median(
        [r["initial_rtre_median"] for r in rows]
    )
    out["n_pairs_um"] = len(with_um)
    out["median_tre_um_p50"] = _median([r["tre_um_p50"] for r in with_um])
    out["median_tre_px_p50"] = _median([r["tre_px_p50"] for r in rows])
    out["frames"] = "+".join(sorted({r["frame"] for r in rows if r["frame"]}))
    return out


def sides_of(dataset):
    return ("tune",) if dataset == ANHIR_TUNE else SIDES


def summarise(pairs, dataset):
    """One row per (side, tissue or ``all``, method)."""
    out = []
    methods = list(dict.fromkeys(r["method"] for r in pairs))
    tissues = ["all", *sorted({r["tissue"] for r in pairs})]
    for side in sides_of(dataset):
        for tissue in tissues:
            for m in methods:
                mine = [
                    r
                    for r in pairs
                    if r["method"] == m
                    and (side == "all" or r["side"] == side)
                    and (tissue == "all" or r["tissue"] == tissue)
                ]
                if mine and not all(r.get("not_run") for r in mine):
                    out.append(
                        {
                            "dataset": dataset,
                            "side": side,
                            "tissue": tissue,
                            "method": m,
                            **aggregate(mine),
                        }
                    )
    return out


def settle_frames(pairs, blocked=None):
    """Fill in what a failed row cannot know by itself, in place.

    **Frame.** A pair that did not register has no map; its row takes the frame of the
    same result name's registered pairs (a method maps one way), else the direction its
    run record declares, else ``UNKNOWN_FRAME``. Never a default: a reference -> moving
    method's failed pair is not "the challenge's own quantity".

    **Blocked.** ``blocked`` is ``{result name: why}`` for the rows of a run its gate did
    not let start: their pairs are ``not_run`` with cause ``blocked``, whatever is on
    disk from an earlier submission. Returns ``{result name: frames}``."""
    frames = {}
    for r in pairs:
        if not r["failed"] and r["frame"]:
            frames.setdefault(r["method"], set()).add(r["frame"])
    reference = FRAME[MOV_TO_REF]
    for r in pairs:
        why = (blocked or {}).get(r["method"])
        if why and r["method"] != "initial":
            r.update(failed=True, not_run=True, cause="blocked", problem=why[:200])
            r["robustness"] = 0.0
        if not r["failed"]:
            continue
        known = frames.get(r["method"]) or ({r["frame"]} if r["frame"] else set())
        r["frame"] = "+".join(sorted(known)) if known else UNKNOWN_FRAME
        r["challenge_quantity"] = known == {reference}
    return {
        m: sorted({r["frame"] for r in pairs if r["method"] == m})
        for m in dict.fromkeys(r["method"] for r in pairs)
    }


def not_run_rows(pairs):
    """One row per result name with pairs that were never registered for a reason that
    is not the method's (``anhir_not_run.csv``), by cause."""
    out = []
    for m in dict.fromkeys(r["method"] for r in pairs):
        mine = [r for r in pairs if r["method"] == m]
        for cause in dict.fromkeys(r["cause"] for r in mine if r.get("not_run")):
            hit = [r for r in mine if r.get("not_run") and r["cause"] == cause]
            out.append(
                {
                    "method": m,
                    "cause": cause,
                    "n_pairs": len(hit),
                    "of": len(mine),
                    "in_no_median": True,
                    "why": hit[0]["problem"],
                    "pairs": " ".join(r["case_id"] for r in hit[:20])
                    + (" ..." if len(hit) > 20 else ""),
                }
            )
    return out


def blocked_rows(gates):
    """``{result name: why}`` for every row of a run a gate of this submission blocked
    (``<out>/logs/gates/*.tsv``, as the cohort's scorer reads them)."""
    out = {}
    for g in cohort.read_gates(gates):
        if not g["blocked"]:
            continue
        try:
            run = variants.run_of(g["run"])
        except KeyError:
            continue
        for v in variants.variants_of(run):
            out[v] = (
                f"the run {g['run']} was blocked by its {g['gate']} gate "
                f"({g['status']}): no pair was registered"
            )
    return out


def by_set(pairs, dataset):
    """One row per (tissue set, method), with the set's side."""
    out = []
    for name in sorted({r["set"] for r in pairs}):
        for m in dict.fromkeys(r["method"] for r in pairs):
            mine = [r for r in pairs if r["method"] == m and r["set"] == name]
            if mine:
                out.append(
                    {
                        "dataset": dataset,
                        "side": mine[0]["side"],
                        "tissue": mine[0]["tissue"],
                        "set": name,
                        "method": m,
                        **aggregate(mine),
                    }
                )
    return out


def prep_evidence(pairs, res=()):
    """The table the preprocessing is chosen from: one row per candidate, from the
    tuning-side rows of its FINAL transform, in the criterion's order. ``never_run`` counts
    the pairs with no result at all (a selection over those is refused by the caller)."""
    rows = []
    for c in prep.CANDIDATES:
        name = variants.tuning_final(c)
        mine = [r for r in pairs if r["method"] == name and r["side"] == "tune"]
        if not mine:
            continue
        a = aggregate(mine)
        if not a["n_pairs"]:  # nothing of this candidate ran at all
            a["median_rtre_median"] = a["avg_robustness"] = math.nan
        rows.append(
            {
                "candidate": c,
                "result": name,
                "n_pairs": a["n_pairs"],
                "n_failed": a["n_failed"],
                "never_run": sum(bool(r.get("not_run")) for r in mine),
                "median_of_median_rtre": a["median_rtre_median"],
                "mean_robustness": a["avg_robustness"],
                "avg_of_median_rtre": a["avg_rtre_median"],
                "median_tre_px_p50": a["median_tre_px_p50"],
                "initial_median_of_median_rtre": a["median_initial_rtre_median"],
            }
        )
    try:
        ranked = prep.rank(rows)
    except ValueError:
        ranked = rows
    for i, r in enumerate(ranked, 1):
        r["order"] = i
    return ranked


def _check_split(cases):
    """Every case of the frozen split, and of no other."""
    from .datasets import anhir

    split = anhir.load_split()
    found = dict.fromkeys(c.extra.get("split_sha256") for c in cases)
    if list(found) != [split["sha256"]]:
        raise SystemExit(
            f"the {cases[0].dataset} cases carry the split(s) "
            f"{', '.join(str(h)[:12] for h in found)}; the frozen one is "
            f"{split['sha256'][:12]}. Results of two splits are never put together: "
            "prepare the cases again."
        )
    for c in cases:
        if anhir.side_of(split, c.extra.get("set")) != c.extra.get("side"):
            raise SystemExit(f"{c.dataset}/{c.case_id}: its side is not the split's")
        if c.dataset == ANHIR_TUNE and c.extra.get("side") != "tune":
            raise SystemExit(
                f"{c.dataset}/{c.case_id} is a report-side pair in the tuning dataset"
            )
    return split


def _table(rows, columns):
    return cohort.markdown(rows, columns)


def markdown(
    dataset, cases, split, summary, evidence, fails, primary, frames=None, lost=()
):
    n_tune = sum(c.extra.get("side") == "tune" for c in cases)
    want = split["pairs"]["tune"] + (
        split["pairs"]["report"] if dataset == ANHIR else 0
    )
    no_um = sorted(
        {c.extra["tissue"] for c in cases if not c.extra.get("pixel_size_established")}
    )
    parts = [f"# ANHIR training pairs, `{dataset}`: an external landmark set\n"]
    parts.append(
        "A block of its own: these numbers are in none of the cohort's tables and in "
        "no ranking. Differently stained consecutive sections, brightfield, scored by "
        "the challenge's public manual landmarks.\n"
    )
    parts.append(
        "**These are the challenge's TRAINING pairs.** Published leaderboard numbers "
        "are computed on the hidden evaluation pairs (with a training column beside "
        "them) and are not comparable row for row with anything here; no submission "
        "was made and no rank among published methods is claimed. **The `stare_prep` "
        "row's preprocessing was chosen on the tuning side of this same training "
        "set.** STARE is not designed for consecutive sections: its rows here are an "
        "out-of-domain test.\n"
    )
    if dataset == ANHIR_TUNE:
        parts.append(
            "**TUNING SIDE. Nothing on this page is a result.** Only the preprocessing "
            "candidates of the `stare_prep` row run here; the choice is made by "
            f"this rule: {prep.CRITERION}.\n"
        )
    parts.append(
        f"{len(cases)} pair(s) prepared ({n_tune} on the tuning side, "
        f"{len(cases) - n_tune} on the report side); split `{split['sha256'][:12]}` "
        "by tissue set (`anhir_split.json`).\n"
    )
    if len(cases) != want:
        parts.append(
            f"**INCOMPLETE: {len(cases)} of {want} pairs are prepared.** No number "
            "here is the set's.\n"
        )
    parts.append(prep.frozen_summary() + "\n")
    parts.append(
        "STARE reads one channel: the plain `stare` row is given each image as "
        "inverted luminance (`255 - (0.299 R + 0.587 G + 0.114 B)`, 8 bit, full "
        "resolution), written by the benchmark; `stare_prep` is given the frozen "
        "candidate instead. VALIS and DeeperHistReg read the RGB images.\n"
    )
    if no_um:
        parts.append(
            f"Micrometres are not given for {', '.join(no_um)} (the pixel size is not "
            "established from the challenge's own page); those pairs are in every "
            "rTRE and pixel column and not in `median_tre_um_p50` (`n_pairs_um`).\n"
        )
    parts.append(
        "`median_rtre_median`: the median over pairs of the per-pair median rTRE; "
        "`avg_*`: the mean over pairs. A pair that did not register counts at its "
        "unregistered landmarks with robustness 0 (the official rule) and is counted "
        "in `n_failed`; `median_rtre_median_registered` leaves those pairs out. "
        "`n_robust`: pairs with robustness above 0.5.\n"
    )
    if lost:
        parts.append("## Not run, for a reason that is not the method's\n")
        parts.append(
            "These pairs were never registered: the run was blocked by its gate, the "
            "task never started, or the attempt was lost to the scheduler or to the "
            "benchmark's own step. They are in NO median below (`n_not_run`); a row "
            "with no other pair is in no table at all. A pair the METHOD failed on is "
            "different: it counts at its unregistered landmarks (`n_failed`).\n"
        )
        parts += [_table(list(lost), list(lost[0])), ""]
    order = list(primary or [])
    frames = frames or {}
    for side in sides_of(dataset):
        block = [r for r in summary if r["side"] == side and r["tissue"] == "all"]
        block.sort(
            key=lambda r: (
                order.index(r["method"]) if r["method"] in order else len(order)
            )
        )
        if not block:
            continue
        parts += [f"## {SIDE_TITLE[side]}\n", _table(block, list(SUMMARY_COLUMNS)), ""]
    # every frame any pair of the row was measured in: one reference -> moving pair is
    # enough for the row to be named here
    reference = FRAME[MOV_TO_REF]
    inverse = sorted(
        m for m, f in frames.items() if m != "initial" and set(f) - {reference}
    )
    if inverse:
        parts.append(
            "Measured in the SOURCE image's frame, not the challenge's own quantity "
            "(the method maps reference to moving; no transform is inverted): "
            + ", ".join(f"`{m}`" for m in inverse)
            + ".\n"
        )
    parts.append(
        "Per tissue: `anhir_summary.csv` (column `tissue`); per tissue set: "
        "`anhir_by_set.csv`; per pair: `anhir_pairs.csv`.\n"
    )
    if evidence:
        parts.append("## The preprocessing candidates on the tuning side\n")
        parts += [_table(evidence, list(evidence[0])), ""]
    excluded = variants.excluded_for(dataset)
    if dataset == ANHIR_TUNE:
        parts.append(f"## Runs\n\n{variants.TUNING_ONLY}\n")
    elif excluded:
        parts.append("## Runs of the cohort's list that are not run here\n")
        by_reason = {}
        for name, why in excluded:
            by_reason.setdefault(why, []).append(name)
        parts += [
            f"- {', '.join(f'`{n}`' for n in names)}: {why}"
            for why, names in by_reason.items()
        ]
    parts.append("\n## Not computed for these cases\n")
    parts += [f"- {k}: {v}" for k, v in SKIPPED.items()]
    if fails:
        parts += ["\n## Failures (launches and pairs)\n"]
        parts += [_table(fails, list(fails[0])), ""]
    return "\n".join(parts) + "\n"


def score(
    cases_root,
    out,
    dataset,
    workers=1,
    expected=None,
    primary=None,
    dropped=(),
    write=True,
    gates=None,
):
    """Score ``anhir`` or ``anhir_tune``; writes ``<out>/tables/<dataset>/`` and returns
    the rows. ``expected`` / ``primary`` are ``variants.expected_variants(dataset)`` and
    ``variants.primary_for(dataset)`` (None scores only what has results)."""
    if dataset not in ANHIR_DATASETS:
        raise SystemExit(f"{dataset!r} is not an ANHIR dataset {ANHIR_DATASETS}")
    cases = list_cases(cases_root, dataset)
    if not cases:
        raise SystemExit(f"no prepared {dataset} cases under {cases_root}")
    split = _check_split(cases)
    names = cohort.find_variants(out, dataset, expected, dropped)
    jobs = [(str(out), c, names) for c in cases]
    if workers > 1 and len(jobs) > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            done = pool.map(_score_case, jobs)
    else:
        done = [_score_case(j) for j in jobs]
    pairs = [r for d in done for r in d[0]]
    res = [r for d in done for r in d[1]]
    frames = settle_frames(pairs, blocked_rows(gates))
    lost = not_run_rows(pairs)
    summary = summarise(pairs, dataset)
    sets = by_set(pairs, dataset)
    evidence = prep_evidence(pairs) if dataset == ANHIR_TUNE else []
    fails = cohort.failures(res)
    res_summary = cohort.summarise_resources(res)
    excluded = [
        {"dataset": dataset, "run": name, "reason": why}
        for name, why in variants.excluded_for(dataset)
    ]
    result = {
        "pairs": pairs,
        "summary": summary,
        "by_set": sets,
        "prep_evidence": evidence,
        "failures": fails,
        "resources": res,
        "summary_resources": res_summary,
        "excluded": excluded,
        "split": split,
        "not_run": lost,
        "frames": frames,
    }
    if not write:
        return result
    tables = Path(out) / "tables" / dataset
    tables.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        (tables / f"{name}.csv").unlink(missing_ok=True)
    cohort.write_csv(tables / "anhir_pairs.csv", pairs)
    cohort.write_csv(tables / "anhir_summary.csv", summary)
    cohort.write_csv(tables / "anhir_by_set.csv", sets)
    cohort.write_csv(tables / "anhir_failures.csv", fails)
    cohort.write_csv(tables / "anhir_resources.csv", res)
    cohort.write_csv(tables / "anhir_summary_resources.csv", res_summary)
    cohort.write_csv(tables / "anhir_excluded.csv", excluded)
    if evidence:
        cohort.write_csv(tables / "anhir_prep_evidence.csv", evidence)
    cohort.write_csv(tables / "anhir_not_run.csv", lost)
    text = markdown(
        dataset, cases, split, summary, evidence, fails, primary, frames, lost
    )
    (tables / "summary.md").write_text(text)
    print(text)
    print(f"tables -> {tables}")
    result["tables"] = str(tables)
    return result


def select_prep(cases_root, out, freeze=False, path=None, workers=1):
    """Choose the ``stare_prep`` preprocessing from the tuning side's results. Prints
    the evidence table and the winner; with ``freeze`` rewrites ``prep_frozen.json``.
    Refuses when a candidate has no result on a tuning pair (a selection on part of the
    evidence), and when the prepared tuning side is not the whole of the split's."""
    result = score(
        cases_root,
        out,
        ANHIR_TUNE,
        workers,
        expected=variants.expected_variants(ANHIR_TUNE),
        write=False,
    )
    split, evidence = result["split"], result["prep_evidence"]
    n_cases = len({r["case_id"] for r in result["pairs"]})
    if n_cases != split["pairs"]["tune"]:
        raise SystemExit(
            f"{n_cases} of the {split['pairs']['tune']} tuning-side pairs are "
            "prepared: the selection is made on the whole tuning side or not at all"
        )
    missing = [r["candidate"] for r in evidence if r["never_run"]]
    missing += [
        c for c in prep.CANDIDATES if c not in {r["candidate"] for r in evidence}
    ]
    if missing:
        raise SystemExit(
            f"no result on some tuning pair for: {', '.join(missing)}. Every "
            "candidate runs on every tuning pair before one is chosen "
            "(submit_cohort.sh --datasets anhir_tune)."
        )
    print(_table(evidence, list(evidence[0])))
    winner = evidence[0]["candidate"]
    print(f"\ncriterion: {prep.CRITERION}\nselected: {winner}")
    if freeze:
        import datetime

        record = prep.make_record(
            winner,
            "selected",
            evidence,
            split_sha256=split["sha256"],
            selected_on=f"{n_cases} tuning-side pairs of split "
            f"{split['sha256'][:12]}, {datetime.date.today().isoformat()}",
        )
        where = prep.write_frozen(record, path)
        print(
            f"frozen -> {where} ({record['sha256'][:12]}). Commit it; every "
            f"`{prep.ROW}` result made with another candidate is now stale and is "
            "not reused (its options differ)."
        )
    return winner, evidence
