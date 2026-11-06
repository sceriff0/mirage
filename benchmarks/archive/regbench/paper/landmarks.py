"""The paper tables of a public LANDMARK dataset (``hyreco``): built from the tables its
scorer wrote (``<out>/tables/<dataset>/``: ``regbench.landmarks``) and nothing else;
computes no error itself.

``public_landmark_pairs``  one row per (pair, row): the terminal row of every run of the
                           dataset's list (``tables.terminal_rows``) and ``initial``.
``public_landmarks``       one row per (set, row): the pairs' statistics aggregated as
                           BIRL's ``compute_scores`` defines them (per-pair median /
                           maximum rTRE and robustness, then the median and the mean over
                           pairs; the robust subset, robustness above 0.5) and the
                           micrometre medians.
``method_best``            the best row of each family, in the cohort's columns: the lowest
                           ``rtre_median_med`` over every pair, EVERY terminal row of the
                           family a candidate (STARE's whole tier x stride x level cross;
                           ruling 2026-10-19), chosen on these same pairs
                           (``in_sample_selection`` True, as on every dataset).

A pair never registered for a cause that is not the method's (``not_run``) is in no
statistic and is counted (``n_not_run``); a pair the method failed on counts at its
unregistered landmarks (``n_failed``); a flagged pair is kept and counted (``n_flagged``).
Missing values are written ``NA`` (``schema``).
"""

from __future__ import annotations

import math

from . import schema
from . import tables as T

LANDMARK_WRITTEN = ("public_landmarks", "public_landmark_pairs", "method_best")
ROBUST_ABOVE = 0.5  # BIRL evaluate_submission.py:281
PAIRS_TABLE = {"hyreco": "hyreco_pairs"}
SUMMARY_TABLE = {"hyreco": "hyreco_summary"}
AGGREGATION = {
    "hyreco": "per-pair median TRE, then the median over pairs (tre_um_med); "
    "tre_um_pooled_med is one median over all landmarks of all pairs (Lotz et al. 2023's "
    "MTRE); rTRE (TRE over the target image's diagonal: the side the error is measured "
    "on) and robustness as BIRL compute_scores defines them; all_pairs counts an id "
    "both subsets list once (subset A's copy)",
}
CRITERION = "median over pairs of the per-pair median rTRE (lowest wins), every pair"
RULE = (
    "candidates are the family's terminal rows on the dataset (STARE: its whole tier x "
    "stride x level cross); every pair but those not run for a cause not the method's "
    "(a failed pair counts at its unregistered landmarks); the lowest median over pairs "
    "of the per-pair median rTRE wins; ties by name; chosen on these same pairs "
    "(in-sample)"
)


def _sets(dataset, r):
    """The sets a pair row belongs to: HyReCo ``all_pairs`` and its subset. An id both
    subsets list is in ``all_pairs`` once, subset A's copy (ruling 2026-10-21)."""
    sub = str(r.get("subset") or "").lower()
    own = [f"subset_{sub}"] if sub in ("a", "b") else []
    if sub == "b" and _bool(r.get("shared_id")) is True:
        return own
    return ["all_pairs"] + own


def _bool(v):
    f = T._flag(v)
    return f if isinstance(f, bool) else schema.NA


def _rows(out, dataset):
    """The scorer's pair rows of the terminal rows and ``initial``, typed."""
    keep = set(T.terminal_rows(dataset).values()) | {T.INITIAL}
    got = []
    for r in T._csv(T._tables(out, dataset) / f"{PAIRS_TABLE[dataset]}.csv"):
        if r.get("method") not in keep:
            continue
        got.append(r)
    return got


def pairs_table(rows, dataset):
    out = []
    for r in rows:
        fam = T.family_of(r["method"])
        out.append(
            schema.empty_row(
                "public_landmark_pairs",
                source=T.SOURCE,
                dataset=dataset,
                case_id=r["case_id"],
                set=r.get("subset"),
                family=fam if fam is not None else schema.NA,
                row=r["method"],
                frame=r.get("frame"),
                rtre_median=T._num(r.get("rtre_median")),
                rtre_max=T._num(r.get("rtre_max")),
                robustness=T._num(r.get("robustness")),
                tre_um_p50=T._num(r.get("tre_um_p50")),
                tre_um_p90=T._num(r.get("tre_um_p90")),
                imputed_initial=_bool(r.get("imputed_initial", r.get("failed"))),
                excluded_from_summaries=_bool(
                    r.get("excluded_from_summaries", r.get("not_run"))
                ),
                flagged=_bool(r.get("flagged")),
                flags=r.get("flags"),
                shared_id=_bool(r.get("shared_id"))
                if dataset == "hyreco"
                else schema.NA,
                cause=r.get("cause"),
            )
        )
    return out


def _pooled(out, dataset):
    """``{(set, row): pooled median TRE}`` from HyReCo's own summary."""
    if dataset not in SUMMARY_TABLE:
        return {}
    return {
        (r["set"], r["method"]): T._num(r.get("pooled_median_tre_um"))
        for r in T._csv(T._tables(out, dataset) / f"{SUMMARY_TABLE[dataset]}.csv")
    }


def _stat(values, fn):
    v = T._finite(values)
    return fn(v) if v else math.nan


def landmarks_table(rows, dataset, pooled):
    by = {}
    for r in rows:
        for name in _sets(dataset, r):
            by.setdefault((name, r["method"]), []).append(r)
    out = []
    for (name, row), mine in by.items():
        not_run = [r for r in mine if T._flag(r.get("not_run")) is True]
        kept = [r for r in mine if T._flag(r.get("not_run")) is not True]
        num = lambda k, rs=kept: [T._num(r.get(k)) for r in rs]  # noqa: E731
        robust = [r for r in kept if T._num(r.get("robustness")) > ROBUST_ABOVE]
        fam = T.family_of(row)
        out.append(
            schema.empty_row(
                "public_landmarks",
                source=T.SOURCE,
                dataset=dataset,
                set=name,
                family=fam if fam is not None else schema.NA,
                row=row,
                n_pairs=len(kept),
                n_failed=sum(T._flag(r.get("failed")) is True for r in kept),
                n_not_run=len(not_run),
                n_flagged=sum(T._flag(r.get("flagged")) is True for r in kept),
                rtre_median_med=_stat(num("rtre_median"), T._median),
                rtre_median_mean=_stat(num("rtre_median"), T._mean),
                rtre_max_med=_stat(num("rtre_max"), T._median),
                rtre_max_mean=_stat(num("rtre_max"), T._mean),
                n_robust=len(robust),
                rtre_median_robust_med=_stat(num("rtre_median", robust), T._median),
                rtre_median_robust_mean=_stat(num("rtre_median", robust), T._mean),
                rtre_max_robust_mean=_stat(num("rtre_max", robust), T._mean),
                robustness_med=_stat(num("robustness"), T._median),
                robustness_mean=_stat(num("robustness"), T._mean),
                tre_um_med=_stat(num("tre_um_p50"), T._median),
                tre_um_pooled_med=pooled.get((name, row), math.nan),
                best_of_family=False,
                in_sample_selection=True,
                frame="+".join(sorted({r["frame"] for r in kept if r.get("frame")}))
                or schema.NA,
                aggregation=AGGREGATION[dataset],
            )
        )
    return out


def best_rows(summary, dataset):
    """The ``method_best`` rows (module docstring) and, in place, ``best_of_family`` of
    the chosen rows of ``summary``."""
    out = []
    every_set = "all_pairs"
    terminals = list(dict.fromkeys(T.terminal_rows(dataset).values()))
    families = list(
        dict.fromkeys(T.family_of(t) for t in terminals if T.family_of(t) is not None)
    )
    for fam in families:
        cands = [t for t in terminals if T.family_of(t) == fam]
        for name in dict.fromkeys(r["set"] for r in summary):
            mine = [r for r in summary if r["set"] == name and r["row"] in cands]
            scored = sorted(
                (
                    (
                        0 if math.isfinite(v) else 1,
                        v if math.isfinite(v) else 0.0,
                        r["row"],
                        r,
                    )
                    for r in mine
                    for v in [T._num(r["rtre_median_med"])]
                ),
                key=lambda t: t[:3],
            )
            # a family none of whose rows has a value has no best row
            scored = [t for t in scored if t[0] == 0]
            if scored:
                scored[0][3]["best_of_family"] = True
            if name != every_set:
                continue
            best = scored[0] if scored else None
            out.append(
                schema.empty_row(
                    "method_best",
                    source=T.SOURCE,
                    dataset=dataset,
                    family=fam,
                    best_row=best[2] if best else schema.NA,
                    n_candidates=len(cands),
                    criterion=CRITERION,
                    criterion_value=T._num(best[3]["rtre_median_med"])
                    if best
                    else math.nan,
                    n_cases_common=best[3]["n_pairs"] if best else math.nan,
                    n_dice_zeroed=math.nan,
                    tie_break="the name",
                    in_sample_selection=True,
                    rule=RULE,
                    common_block="all pairs",
                )
            )
    return out


def build(out, dataset):
    rows = _rows(out, dataset)
    summary = landmarks_table(rows, dataset, _pooled(out, dataset))
    best = best_rows(summary, dataset)
    built = {
        "public_landmarks": summary,
        "public_landmark_pairs": pairs_table(rows, dataset),
        "method_best": best,
    }
    # T29 (M8): a run of excluded_runs.csv is a row of NA with its reason
    return T.add_excluded(built, T.excluded_runs(out, dataset), dataset,
                          summary_keys={"public_landmarks": {"set": "all_pairs"}},
                          best_sets=(None,))


def write_all(out, dataset):
    """Write ``LANDMARK_WRITTEN`` under ``<out>/paper/<dataset>/``; ``{table: path}``."""
    built = build(out, dataset)
    target = T._paper_dir(out, dataset)
    target.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in LANDMARK_WRITTEN:
        path = target / f"{name}.csv"
        schema.write(path, name, built[name])
        paths[name] = path
    return paths


def is_landmark_set(dataset):
    from .. import public_of

    spec = public_of(dataset) if dataset != T.COHORT else None
    return spec is not None and spec["kind"] == "landmarks"
