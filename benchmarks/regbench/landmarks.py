"""Score an external landmark set (HyReCo): its own tables, never the cohort's.

``score(cases_root, out, dataset)`` reads the same ``warped.npz`` / ``run.json`` as the
cohort's scorer and writes ``<out>/tables/<dataset>/``. Nothing here enters
``<out>/tables/*.csv``, the cohort's summary, a paired comparison or a ranking; the
cohort's ``score`` never reads an external dataset.

**The error.** A case holds paired landmarks (``truth_mov`` on the PHH3 slide,
``truth_ref`` on the H&E slide, full-resolution pixels). A method's map has moved the
landmarks of the side it starts from (``warped.npz: truth_xy``); the error of a landmark
is the distance to its partner on the other side, times the pixel size: target
registration error in micrometres, in the method's own direction, as in
``regbench.truth.error_stats`` (which gives the per-pair numbers here). A point a method
could not map has an infinite error. A run that failed or never ran is scored at the
unregistered pose (``imputed_initial``), so a failure is a row and counts in every
pooled number.

**Two aggregations, labelled apart** (``hyreco_summary.csv``):

``per_pair_*``  each pair's median error, then the median, minimum and maximum over
                pairs.
``pooled_*``    one median over ALL landmarks of all pairs of the set: the definition of
                Lotz et al. 2023 ("MTRE": the median "over all N_images image pairs and
                over all available landmarks"). It weights a pair by its landmark count.

**Sets** (column ``set``). ``hyreco_a``: ``tuning`` (all nine pairs; a tuning set, never
a result). ``hyreco_b``: ``report`` (the pairs with ``extra["report_set"]``: without the
four ids subset A also has) and ``all_pairs`` (every prepared pair: with all 54 this is
the published definition's set, the like-for-like line beside the one published number
quoted below).

**Published numbers** (``PUBLISHED``, ``hyreco_published.csv``) are quoted, not
reproduced: neither method was run here. Only a number whose aggregation is defined and
whose subset is stated is in the table; CORE's 0.41 um ("AMTRE", aggregation undefined,
subset unstated) is deliberately absent (ruling R3.12 c).

**Refusals.** Cases of more than one settings hash; a ``hyreco_b`` scored when its cases'
settings are not the ones recorded as chosen on subset A. The tissue-level block and the
off-seam rows are not computed for these cases (``SKIPPED``).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import HYRECO_DATASETS, truth, variants
from . import score as cohort
from .cases import list_cases, load_points
from .mapping import FRAME, MOV_TO_REF

TUNING, REPORT = HYRECO_DATASETS
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
        "dataset": REPORT,
        "set": "all_pairs",
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
        "dataset": REPORT,
        "set": "all_pairs",
        "method": "HistokatFusion, affine only (Fraunhofer MEVIS)",
        "n_pairs": 54,
        "n_landmarks": 2357,
        "pooled_median_tre_um": 1.7,
        "status": NOT_REPRODUCED,
        "conditions": "as above",
        "source": LOTZ,
    },
    {
        "dataset": TUNING,
        "set": "tuning",
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
# what each run registers at, from its adapter's own documentation (not measured here)
WORKING_RESOLUTION = {
    "stare": "the proxy's own pixels (0.24 um, times the reduction), stride 64",
    "valis120": "upstream defaults: 512 px (rigid) and 2048 px (non-rigid) a side",
    "valis120_hires": "upstream's high-resolution recipe: micro pass at 0.25 x the slide",
}
TABLES = (
    "hyreco_pairs",
    "hyreco_points",
    "hyreco_summary",
    "hyreco_published",
    "hyreco_failures",
    "hyreco_resources",
    "hyreco_summary_resources",
    "hyreco_excluded",
)


def landmark_errors(case, points, warped):
    """``(error px, initial px, mapped, source, target, direction)`` of one run on one
    case; ``warped`` None scores the unregistered pose. Raises ValueError when the
    result does not fit the case's landmarks."""
    t_mov = np.asarray(points["truth_mov"], dtype=float).reshape(-1, 2)
    t_ref = np.asarray(points["truth_ref"], dtype=float).reshape(-1, 2)
    direction = str(warped["direction"]) if warped is not None else MOV_TO_REF
    if direction not in FRAME:
        raise ValueError(f"unknown direction {direction!r}")
    source, target = (t_mov, t_ref) if direction == MOV_TO_REF else (t_ref, t_mov)
    if warped is None:
        mapped = source
    else:
        mapped = np.asarray(warped.get("truth_xy"), dtype=float).reshape(-1, 2)
        if len(mapped) != len(source):
            raise ValueError("warped.npz does not fit the case's landmarks")
    bad = ~np.isfinite(mapped).all(axis=1)
    err = np.full(len(target), math.inf)
    err[~bad] = np.hypot(*(mapped[~bad] - target[~bad]).T)
    init = np.hypot(*(source - target).T)
    return err, init, mapped, source, target, direction


def _score_case(job):
    out, case, names = job
    points = load_points(case)
    if "truth_mov" not in points or not len(points["truth_mov"]):
        raise SystemExit(f"{case.dataset}/{case.case_id}: the case has no landmarks")
    px = float(case.pixel_size_um)
    base = {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "group": case.group,
        "report_set": bool(case.extra.get("report_set", True)),
        "shared_id_with_subset_a": bool(case.extra.get("shared_id_with_subset_a")),
    }
    pairs, pts, res = [], [], []
    for v in names:
        if v == "initial":
            run, warped, problem, cause = {}, None, "", ""
        else:
            run, warped, problem, cause = cohort._load_run(out, v, case)
            res.append(
                {
                    **base,
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
        imputed = v != "initial" and warped is None
        jac = truth.jacobian_stats(warped)
        pairs.append(
            {
                **base,
                "method": v,
                "frame": FRAME[direction],
                **truth.error_stats(mapped, target, source, px),
                **jac,
                "pixel_size_um": px,
                "pixel_size_source": case.extra.get("pixel_size_source", ""),
                "imputed_initial": imputed,
                "problem": str(problem)[:200],
                "cause": cause,
            }
        )
        pts += [
            {
                **base,
                "method": v,
                "landmark": i,
                "tre_um": float(e * px),
                "initial_tre_um": float(s * px),
                "imputed_initial": imputed,
            }
            for i, (e, s) in enumerate(zip(err, init))
        ]
    return pairs, pts, res


def _median(values):
    """The median of errors that may hold ``inf`` (a lost point is not hidden)."""
    v = np.sort(np.asarray(values, dtype=float))
    return truth._percentile(v, 50) if len(v) else math.nan


def _p90(values):
    v = np.sort(np.asarray(values, dtype=float))
    return truth._percentile(v, 90) if len(v) else math.nan


def sets_of(dataset):
    """``[(set name, keep(pair row))]`` of a dataset."""
    if dataset == TUNING:
        return [("tuning", lambda r: True)]
    return [("report", lambda r: r["report_set"]), ("all_pairs", lambda r: True)]


def summarise(pairs, points):
    """One row per (dataset, set, method): the per-pair and the pooled numbers."""
    out = []
    for ds in dict.fromkeys(r["dataset"] for r in pairs):
        for name, keep in sets_of(ds):
            for m in dict.fromkeys(r["method"] for r in pairs):
                mine = [r for r in pairs if (r["dataset"], r["method"]) == (ds, m)]
                mine = [r for r in mine if keep(r)]
                if not mine:
                    continue
                ids = {r["case_id"] for r in mine}
                mp = [
                    p
                    for p in points
                    if (p["dataset"], p["method"]) == (ds, m) and p["case_id"] in ids
                ]
                per_pair = [r["tre_um_p50"] for r in mine]
                err = [p["tre_um"] for p in mp]
                init = [p["initial_tre_um"] for p in mp]
                out.append(
                    {
                        "dataset": ds,
                        "set": name,
                        "method": m,
                        "n_pairs": len(mine),
                        "n_ok": sum(not r["imputed_initial"] for r in mine),
                        "n_landmarks": len(mp),
                        "per_pair_median_tre_um_med": _median(per_pair),
                        "per_pair_median_tre_um_min": float(np.min(per_pair)),
                        "per_pair_median_tre_um_max": float(np.max(per_pair)),
                        "per_pair_p90_tre_um_med": _median(
                            [r["tre_um_p90"] for r in mine]
                        ),
                        "pooled_median_tre_um": _median(err),
                        "pooled_p90_tre_um": _p90(err),
                        "pooled_frac_under_1um": float(np.mean(np.array(err) < 1.0)),
                        "pooled_frac_improved": float(
                            np.mean(np.array(err) < np.array(init))
                        ),
                        "initial_pooled_median_tre_um": _median(init),
                        "n_nonfinite": int(sum(r["n_nonfinite"] for r in mine)),
                        "fold_pct_max": cohort._stat(
                            [r["fold_pct"] for r in mine], np.max
                        ),
                    }
                )
    return out


def _settings(cases):
    """The one settings record of the cases; SystemExit when they carry several."""
    hashes = dict.fromkeys(c.extra.get("settings_sha256") for c in cases)
    if len(hashes) != 1 or None in hashes:
        raise SystemExit(
            f"the {cases[0].dataset} cases carry {len(hashes)} different settings "
            f"({', '.join(str(h)[:12] for h in hashes)}). One result rests on one "
            "setting, chosen on subset A."
        )
    return cases[0].extra["settings"], next(iter(hashes))


def _tuned_check(cases, settings_hash):
    """A report-set case must name where its settings were chosen, and those prepared
    tuning cases, if still there, must carry the same settings."""
    from .datasets import hyreco

    sources = dict.fromkeys(c.extra.get("settings_from") for c in cases)
    if len(sources) != 1 or not next(iter(sources)):
        raise SystemExit(
            f"the {REPORT} cases do not name one prepared tuning set their settings "
            "were chosen on (extra.settings_from); they were not prepared by "
            "`prepare hyreco --subset b`"
        )
    source = next(iter(sources))
    if Path(source, TUNING).is_dir():
        tuned = hyreco.tuned_settings(source)
        if hyreco.settings_hash(tuned) != settings_hash:
            raise SystemExit(
                f"the {REPORT} cases were prepared with settings {settings_hash[:12]}, "
                f"and the tuning set at {source} now carries "
                f"{hyreco.settings_hash(tuned)[:12]}: the report set does not follow "
                "the tuning set"
            )
    return source


def _table(rows, columns):
    return cohort.markdown(rows, columns)


SUMMARY_COLUMNS = (
    "method",
    "n_ok",
    "n_pairs",
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


def markdown(dataset, cases, settings, sha, source, summary, fails, primary):
    floor = ANNOTATION_FLOOR
    n_landmarks = sum(int(c.extra.get("n_landmarks", 0)) for c in cases)
    px_sources = sorted({str(c.extra.get("pixel_size_source", "")) for c in cases})
    complete = all(c.extra.get("subset_complete", True) for c in cases)
    parts = [f"# HyReCo, {dataset}: an external landmark set\n"]
    parts.append(
        "A block of its own: these numbers are in none of the cohort's tables, in no "
        "paired comparison with them and in no ranking. Brightfield H&E / PHH3 pairs "
        "of the same section, scored by manual landmarks (target registration error, "
        "micrometres). The same landmarks for every method; a run that did not "
        "register is scored at the unregistered pose and counted.\n"
    )
    if dataset == TUNING:
        parts.append(
            "**TUNING SET. Nothing on this page is a result.** Every setting is chosen "
            "here and then used unchanged on subset B (`hyreco_b`).\n"
        )
    else:
        parts.append(
            "**REPORT SET. Nothing was chosen on it.** Its settings were copied from "
            f"the prepared tuning set (`hyreco_a`, subset A's nine pairs) at `{source}`."
            "\n"
        )
    parts.append(
        f"Settings `{sha[:12]}`: STARE's one-channel proxy is `{settings['proxy']}`, "
        f"reduced {settings['reduce']} x (benchmark-side preprocessing; VALIS reads "
        "the original RGB slides).\n"
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
    order = list(primary or [])
    for name, _ in sets_of(dataset):
        block = [r for r in summary if r["set"] == name]
        block.sort(
            key=lambda r: (
                order.index(r["method"]) if r["method"] in order else len(order)
            )
        )
        title = {
            "tuning": "Tuning set (all nine pairs)",
            "report": "Report set (subset B without the four ids subset A also has)",
            "all_pairs": "All prepared pairs",
        }[name]
        parts += [f"## {title}\n"]
        parts.append(
            "`per_pair_*`: each pair's median error, then the median / minimum / "
            "maximum over pairs. `pooled_*`: one median over all landmarks of all "
            "pairs of the set. The two are different aggregations.\n"
        )
        parts += [_table(block, list(SUMMARY_COLUMNS)), ""]
    like = like_for_like(summary, dataset)
    if any(r["status"] == NOT_REPRODUCED for r in like):
        n_all = max(
            (r["n_pairs"] for r in summary if r["set"] in ("all_pairs", "tuning")),
            default=0,
        )
        n_published = max(r["n_pairs"] for r in PUBLISHED if r["dataset"] == dataset)
        parts.append("## Like for like: pooled median over all landmarks (MTRE)\n")
        parts.append(
            "The definition of Lotz et al. 2023: one median over all landmarks of all "
            "pairs. The rows marked `" + NOT_REPRODUCED + "` are quoted from the "
            "method's authors; that software was not run and its number could not be "
            "checked here. Its resolution and parameters are its authors' "
            "(`hyreco_published.csv`, column `conditions`).\n"
        )
        if n_all != n_published:
            parts.append(
                f"**NOT like for like: {n_all} of {n_published} pairs are scored "
                "here.**\n"
            )
        parts += [_table(like, list(LIKE_COLUMNS)), ""]
    parts.append("## What each run registers at (from its adapter; not measured)\n")
    parts += [f"- `{k}`: {v}" for k, v in WORKING_RESOLUTION.items()]
    excluded = variants.excluded_for(dataset)
    parts.append("\n## Runs of the cohort's list that are not run here\n")
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


def score(cases_root, out, dataset, workers=1, expected=None, primary=None, dropped=()):
    """Score one external dataset; writes ``<out>/tables/<dataset>/`` and returns the
    rows. ``expected`` and ``primary`` are ``variants.expected_variants(dataset)`` and
    ``variants.primary_for(dataset)`` (None scores only what has results)."""
    if dataset not in HYRECO_DATASETS:
        raise SystemExit(f"{dataset!r} is not a HyReCo set {HYRECO_DATASETS}")
    cases = list_cases(cases_root, dataset)
    if not cases:
        raise SystemExit(f"no prepared {dataset} cases under {cases_root}")
    settings, sha = _settings(cases)
    source = _tuned_check(cases, sha) if dataset == REPORT else ""
    names = cohort.find_variants(out, dataset, expected, dropped)
    jobs = [(str(out), c, names) for c in cases]
    if workers > 1 and len(jobs) > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            done = pool.map(_score_case, jobs)
    else:
        done = [_score_case(j) for j in jobs]
    pairs = [r for d in done for r in d[0]]
    points = [r for d in done for r in d[1]]
    res = [r for d in done for r in d[2]]
    summary = summarise(pairs, points)
    fails = cohort.failures(res)
    res_summary = cohort.summarise_resources(res)
    published = [dict(r) for r in PUBLISHED if r["dataset"] == dataset]
    excluded = [
        {"dataset": dataset, "run": name, "reason": why}
        for name, why in variants.excluded_for(dataset)
    ]
    tables = Path(out) / "tables" / dataset
    tables.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        (tables / f"{name}.csv").unlink(missing_ok=True)
    cohort.write_csv(tables / "hyreco_pairs.csv", pairs)
    cohort.write_csv(tables / "hyreco_points.csv", points)
    cohort.write_csv(tables / "hyreco_summary.csv", summary)
    cohort.write_csv(tables / "hyreco_published.csv", published)
    cohort.write_csv(tables / "hyreco_failures.csv", fails)
    cohort.write_csv(tables / "hyreco_resources.csv", res)
    cohort.write_csv(tables / "hyreco_summary_resources.csv", res_summary)
    cohort.write_csv(tables / "hyreco_excluded.csv", excluded)
    text = markdown(dataset, cases, settings, sha, source, summary, fails, primary)
    (tables / "summary.md").write_text(text)
    print(text)
    print(f"tables -> {tables}")
    return {
        "pairs": pairs,
        "points": points,
        "summary": summary,
        "published": published,
        "failures": fails,
        "resources": res,
        "summary_resources": res_summary,
        "excluded": excluded,
        "settings": settings,
        "tables": str(tables),
    }
