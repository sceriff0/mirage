"""Cross-case aggregation, mirroring the challenge's evaluator.

The per-case frame (one row per case x method) is reduced the way BIRL's
``bm_ANHIR/evaluate_submission.py`` does it:

* ``Average-<stat>`` / ``Median-<stat>`` are the mean / median over cases of the
  per-case median, mean and max rTRE, and of robustness;
* the primary ranking metric is ``a_d(r_m(m_i(rTRE)))``: rank the methods on
  each case by their median rTRE, then average the ranks over cases;
* the ``robust`` subset keeps cases with robustness > 0.5;
* a breakdown per tissue and per status is reported alongside ``all``.

Rows with ``scored == False`` (evaluation cases, whose target landmarks are
server-side) carry NaN metrics and are excluded from every reduction except the
case count, which is reported as scored cases only.
"""

from __future__ import annotations

import pandas as pd

CASE_COLUMNS = [
    "case_id",
    "tissue",
    "scale",
    "status",
    "source_image",
    "target_image",
    "method",
    "n_landmarks",
    "scored",
    "rtre_median",
    "rtre_mean",
    "rtre_max",
    "tre_median_px",
    "robustness",
    "rank_median_rtre",
    "time_min",
    # True when the method produced nothing for this case and the row carries
    # the INITIAL (unregistered) error instead -- the challenge's own rule:
    # "all missing or incomplete registrations ... are considered to have the
    # initial rTRE and they are ranked as last".
    "imputed_initial",
]

AGGREGATE_COLUMNS = [
    "method",
    "subset",
    "n_cases",
    "avg_median_rtre",
    "med_median_rtre",
    "avg_mean_rtre",
    "avg_max_rtre",
    "avg_robustness",
    "med_robustness",
    "avg_rank_median_rtre",
    "avg_time_min",
]

SUBSET_ALL = "all"
SUBSET_ROBUST = "robust"
ROBUST_THRESHOLD = 0.5


def add_ranks(cases: pd.DataFrame) -> pd.DataFrame:
    """Rank methods within each case by median rTRE (1 = best, ties averaged).

    Only scored rows are ranked. A method that produced nothing for a case is
    not absent from the ranking: ``evaluate.score_method`` gives it the initial
    (unregistered) error for that case, flagged ``imputed_initial``, so it ranks
    at the bottom -- the challenge's rule, and the reason a partial submission
    cannot look better than a complete one here.
    """
    out = cases.copy()
    out["rank_median_rtre"] = float("nan")
    scored = out["scored"].astype(bool) & out["rtre_median"].notna()
    if scored.any():
        out.loc[scored, "rank_median_rtre"] = (
            out.loc[scored].groupby("case_id")["rtre_median"].rank(method="average")
        )
    return out


def _reduce(method: str, subset: str, df: pd.DataFrame) -> dict:
    scored = df[df["scored"].astype(bool) & df["rtre_median"].notna()]
    n = int(len(scored))

    def m(col, fn):
        if n == 0 or col not in scored:
            return float("nan")
        s = scored[col].dropna()
        return float(fn(s)) if len(s) else float("nan")

    return {
        "method": method,
        "subset": subset,
        "n_cases": n,
        "avg_median_rtre": m("rtre_median", pd.Series.mean),
        "med_median_rtre": m("rtre_median", pd.Series.median),
        "avg_mean_rtre": m("rtre_mean", pd.Series.mean),
        "avg_max_rtre": m("rtre_max", pd.Series.mean),
        "avg_robustness": m("robustness", pd.Series.mean),
        "med_robustness": m("robustness", pd.Series.median),
        "avg_rank_median_rtre": m("rank_median_rtre", pd.Series.mean),
        "avg_time_min": m("time_min", pd.Series.mean),
    }


def aggregate(cases: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, subset). Ranks are recomputed here so a stale or
    blank ``rank_median_rtre`` column in the input cannot leak into the table."""
    cases = add_ranks(cases)
    rows = []
    for method, df in cases.groupby("method", sort=True):
        rows.append(_reduce(method, SUBSET_ALL, df))
        for status, sdf in df.groupby("status", sort=True):
            rows.append(_reduce(method, str(status), sdf))
        robust = df[df["robustness"] > ROBUST_THRESHOLD]
        rows.append(_reduce(method, SUBSET_ROBUST, robust))
        for tissue, tdf in df.groupby("tissue", sort=True):
            rows.append(_reduce(method, f"tissue:{tissue}", tdf))
    return pd.DataFrame(rows, columns=AGGREGATE_COLUMNS)
