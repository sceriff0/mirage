"""Score ANHIR's training pairs (``anhir``): its own tables, never the cohort's.

Restored from commit 5a46f778 (``benchmarks/regbench/challenge.py``). What changed: the
landmarks are the case's ``truth.npz`` (``regbench.landmark_io``, scored by
``regbench.landmarks``: one per-pair error for both landmark sets, with its integrity
refusal: a result made for other landmarks is ``not_run``); the tuning side's own
dataset, its preprocessing candidates, their selection and the tune / report split are
REMOVED (rulings 2026-10-18 and 2026-10-19: all 230 training pairs are scored together,
nothing is tuned); the per-pair RANK is back (``rank_median_rtre``, below); a pair whose
landmarks fail a sanity check is scored and flagged, never removed.

``score(cases_root, out, dataset)`` reads the same ``warped.npz`` / ``run.json`` as the
cohort's scorer and writes ``<out>/tables/<dataset>/``. Nothing here enters
``<out>/tables/*.csv``, a paired comparison or a ranking of the cohort.

**What these numbers are, and are not.** They are computed on the challenge's TRAINING
pairs, whose landmarks are public. The challenge's leaderboard ranks methods on the
hidden evaluation pairs (and shows a training column beside it); nothing here is
comparable row for row with a leaderboard entry, nothing is exported for submission and
no rank among published methods is claimed. No method is tuned on these pairs: every
run of the list runs with its own fixed settings, and the STARE rows' preprocessing
levels (``regimes.py``) were chosen by reasoning, not on these pairs. STARE is built for
re-imaged sections with the same nuclei in both rounds; consecutive, differently stained
sections at 1 to 9 um per pixel are outside what it was designed for.

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
above 0.5 (BIRL's ``-Robust`` columns). Groups: every pair together, and (BIRL groups by the text before the first
underscore of the set name) and with ``all`` tissues; per tissue SET in
``anhir_by_set.csv``.

**Rank** (``rank_median_rtre`` per pair, ``avg_rank_median_rtre`` per group). The
challenge ranks a method by the average over pairs of its rank by median rTRE among ALL
submitted methods. That needs the other submissions' results on the same pairs, which are
not public, so the rank here is among the METHODS PRESENT in this scoring (every result
name of the run list, ``initial`` excluded), not the leaderboard's: 1 is the best, ties
take the average rank, a pair a method failed on ranks at its unregistered rTRE (the
challenge's rule) and a pair it was never run on (``not_run``) is in no rank (ported from
``5a46f778:benchmarks/anhir/metrics.py`` ``add_ranks``, without pandas). The number
changes when the list of methods does; it is never comparable with a leaderboard rank.

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

from . import landmark_io, truth
from . import score as cohort
from .cases import list_cases
from .landmarks import (
    NO_LANDMARKS,
    flags_of,
    landmark_errors,
    load_result,
    loss_rows,
    pair_stats,
    score_jobs,
    write_tables,
)
from .mapping import FRAME, MOV_TO_REF
from .notrun import NOT_RUN_CAUSES, not_run_rows, settle_frames

ANHIR = "anhir"

ROBUST_ABOVE = 0.5  # evaluate_submission.py:281
PAIR_STATS = ("rtre_median", "rtre_mean", "rtre_max", "robustness")
TABLES = cohort.ANHIR_TABLES
SKIPPED = {
    "tissue-level block": "it compares the nuclear channel of a written image with the "
    "reference's; a brightfield case has none",
    "off-seam rows": "they are defined on nuclei and on the seams of ASHLAR and PALOM; "
    "these cases carry no nuclei and neither method runs on them",
    "the leaderboard's rank": "the challenge ranks among every submitted method on the "
    "same pairs, whose results are not public: `avg_rank_median_rtre` ranks among the "
    "methods present here only",
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
    "avg_rank_median_rtre",
    "n_flagged",
    "median_rtre_median_registered",
    "n_pairs_um",
    "median_tre_um_p50",
    "median_tre_px_p50",
)


def _score_case(job):
    out, case, names = job
    pts = landmark_io.read_truth(case)
    if pts is None or pts["ref"] is None or not len(pts["mov"]):
        raise ValueError(f"{case.dataset}/{case.case_id}: {NO_LANDMARKS}")
    px = float(case.pixel_size_um) if case.pixel_size_um else None
    base = {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "tissue": case.extra.get("tissue", ""),
        "set": case.extra.get("set", ""),
        "pair": f"{case.extra.get('stain_mov', '')} -> {case.extra.get('stain_ref', '')}",
        "flags": flags_of(case),
        "flagged": bool(case.extra.get("flags")),
    }
    pairs, res = [], []
    for v in names:
        if v == "initial":
            run, warped, problem, cause = {}, None, "", ""
        else:
            run, warped, problem, cause = load_result(out, v, case)
            res.append(
                {
                    **base,
                    "group": case.group,
                    "patient_id": case.group,
                    "unit": "pair",
                    "method": v,
                    **cohort.resource_row(
                        run or {"cause": "not_run"}, cohort.moving_pixels(case)
                    ),
                }
            )
        try:
            err, init, mapped, source, target, direction = landmark_errors(
                case, pts, warped
            )
        except ValueError as exc:
            warped, problem, cause = None, str(exc), "exception"
            err, init, mapped, source, target, direction = landmark_errors(
                case, pts, None
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
        not_run = failed and cause in NOT_RUN_CAUSES
        pairs.append(
            {
                **base,
                "method": v,
                "frame": FRAME.get(direction, ""),
                "challenge_quantity": direction == MOV_TO_REF,
                "not_run": not_run,
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
                "imputed_initial": failed,
                "excluded_from_summaries": not_run,
                "problem": str(problem)[:200],
                "cause": cause,
            }
        )
    return pairs, res


def _avg_rank(values):
    """1-based ranks of ``values`` (ascending), ties given their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def add_ranks(pairs):
    """``rank_median_rtre`` on every pair row, in place: the rank of the row's
    ``rtre_median`` among the methods present on the same case (module docstring,
    "Rank"). ``initial``, a ``not_run`` pair and a non-finite median have NaN."""
    by_case = {}
    for r in pairs:
        r["rank_median_rtre"] = math.nan
        if r["method"] == "initial" or r.get("not_run"):
            continue
        if not math.isfinite(r["rtre_median"]):
            continue
        by_case.setdefault((r["dataset"], r["case_id"]), []).append(r)
    for rows in by_case.values():
        for r, k in zip(rows, _avg_rank([r["rtre_median"] for r in rows])):
            r["rank_median_rtre"] = k
    return pairs


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
        "n_flagged": sum(bool(r.get("flagged")) for r in rows),
    }
    for stat in PAIR_STATS:
        out[f"avg_{stat}"] = _mean([r[stat] for r in rows])
        out[f"median_{stat}"] = _median([r[stat] for r in rows])
    for stat in ("rtre_median", "rtre_max"):
        out[f"avg_{stat}_robust"] = _mean([r[stat] for r in robust])
        out[f"median_{stat}_robust"] = _median([r[stat] for r in robust])
    ranks = [r["rank_median_rtre"] for r in rows]
    ranks = [k for k in ranks if not math.isnan(k)]
    out["avg_rank_median_rtre"] = _mean(ranks)
    out["n_ranked"] = len(ranks)
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


def summarise(pairs, dataset):
    """One row per (tissue or ``all``, method)."""
    out = []
    methods = list(dict.fromkeys(r["method"] for r in pairs))
    tissues = ["all", *sorted({r["tissue"] for r in pairs})]
    for tissue in tissues:
        for m in methods:
            mine = [
                r
                for r in pairs
                if r["method"] == m and (tissue == "all" or r["tissue"] == tissue)
            ]
            if mine and not all(r.get("not_run") for r in mine):
                out.append(
                    {
                        "dataset": dataset,
                        "tissue": tissue,
                        "method": m,
                        **aggregate(mine),
                    }
                )
    return out


def by_set(pairs, dataset):
    """One row per (tissue set, method)."""
    out = []
    for name in sorted({r["set"] for r in pairs}):
        for m in dict.fromkeys(r["method"] for r in pairs):
            mine = [r for r in pairs if r["method"] == m and r["set"] == name]
            if mine:
                out.append(
                    {
                        "dataset": dataset,
                        "tissue": mine[0]["tissue"],
                        "set": name,
                        "method": m,
                        **aggregate(mine),
                    }
                )
    return out


def _table(rows, columns):
    return cohort.markdown(rows, columns)


N_TRAINING = 230  # datasets/anhir.py: the training pairs of the cover table


def markdown(dataset, cases, summary, fails, primary, frames=None, lost=()):
    want = N_TRAINING
    flagged = [c for c in cases if c.extra.get("flags")]
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
        "was made and no rank among published methods is claimed: "
        "`avg_rank_median_rtre` is the mean rank of each pair's median rTRE among the "
        "METHODS PRESENT here (`initial` excluded), not the leaderboard's. Nothing was "
        "tuned on these pairs; the STARE rows' preprocessing levels (`regimes.py`, "
        "`bf_consecutive`) were chosen by reasoning. STARE is not designed for "
        "consecutive sections: its rows here are an out-of-domain test.\n"
    )
    parts.append(
        f"{len(cases)} pair(s) prepared, all scored together (no tune / report "
        "split: ruling 2026-10-19).\n"
    )
    if len(cases) != want:
        parts.append(
            f"**INCOMPLETE: {len(cases)} of {want} pairs are prepared.** No number "
            "here is the set's.\n"
        )
    parts.append(
        f"**Flagged pairs: {len(flagged)}** (a landmark sanity check failed; kept and "
        "scored, column `flags` of `anhir_pairs.csv`, a `landmark_flagged` row of "
        "`losses.csv` each)"
        + (
            ": " + "; ".join(f"{c.case_id}: {flags_of(c)}" for c in flagged[:20])
            if flagged
            else ""
        )
        + ".\n"
    )
    parts.append(
        "STARE reads one channel: the plain `stare` row is given each image as "
        "inverted luminance (`255 - (0.299 R + 0.587 G + 0.114 B)`, 8 bit, full "
        "resolution), written by the benchmark; the STARE levels that name another "
        "proxy or STARE's own `--prep` say so in their names (`regimes.py`). VALIS "
        "and DeeperHistReg read the RGB images.\n"
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
            "These pairs were never registered: the task never started, the attempt "
            "was lost to the scheduler or to the benchmark's own step, or the result "
            "was made for other landmarks. They are in NO median below (`n_not_run`); a row "
            "with no other pair is in no table at all. A pair the METHOD failed on is "
            "different: it counts at its unregistered landmarks (`n_failed`).\n"
        )
        parts += [_table(list(lost), list(lost[0])), ""]
    order = list(primary or [])
    frames = frames or {}
    block = [r for r in summary if r["tissue"] == "all"]
    block.sort(
        key=lambda r: order.index(r["method"]) if r["method"] in order else len(order)
    )
    if block:
        parts += ["## All training pairs\n", _table(block, list(SUMMARY_COLUMNS)), ""]
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
    parts.append("\n## Not computed for these cases\n")
    parts += [f"- {k}: {v}" for k, v in SKIPPED.items()]
    if fails:
        parts += ["\n## Failures (launches and pairs)\n"]
        parts += [_table(fails, list(fails[0])), ""]
    return "\n".join(parts) + "\n"


def score(cases_root, out, dataset=ANHIR, workers=1, expected=None, primary=None,
          tables=None):  # fmt: skip
    """Score ``anhir``; writes ``tables`` (default ``<out>/tables/anhir/``) and returns
    the rows. ``expected`` / ``primary`` are ``variants.expected_variants(dataset)`` and
    ``variants.primary_for(dataset)`` (None scores only what has results)."""
    if dataset != ANHIR:
        raise SystemExit(f"{dataset!r} is not ANHIR ({ANHIR!r})")
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
    res = [r for d in done for r in d[1]]
    frames = settle_frames(pairs)
    add_ranks(pairs)
    lost_pairs = not_run_rows(pairs)
    summary = summarise(pairs, dataset)
    sets = by_set(pairs, dataset)
    fails = cohort.failures(res)
    res_summary = cohort.summarise_resources(res)
    lost = loss_rows(out, cases_root, dataset, cases, pairs)
    text = markdown(dataset, cases, summary, fails, primary, frames, lost_pairs)
    named = {
        "anhir_pairs": pairs,
        "anhir_summary": summary,
        "anhir_by_set": sets,
        "anhir_failures": fails,
        "anhir_resources": res,
        "anhir_summary_resources": res_summary,
        "anhir_not_run": lost_pairs,
    }
    excluded, block = write_tables(
        tables, named, text, lost, len(cases), len(cases), dataset
    )
    return {
        "pairs": pairs,
        "summary": summary,
        "by_set": sets,
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
