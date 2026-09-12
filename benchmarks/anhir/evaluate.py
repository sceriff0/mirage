"""Score warped landmarks against the public target landmarks; package a submission.

Inputs are one warped-landmark directory per method (``warp.py``'s output) and
the challenge's landmark archive. Outputs, under ``--out``:

    anhir_cases.csv        one row per (case, method) -- the ihc_method hand-off
    anhir_aggregates.csv   one row per (method, subset)
    anhir_missing.csv      cases a method produced no warped landmarks for
    anhir_reg_eval.csv     the same scores in the shape make_figures --reg-eval
                           takes (``pair_id``, ``mode``, ``true_median_px``,
                           ``true_median_rtre``), so the ground truth joins onto
                           the arm and sweep tables by registration method
    submission/<method>/   the grand-challenge upload for that method (--submit)

Training cases are scored here. Evaluation cases have no public target
landmarks: their rows carry ``scored = False`` and NaN metrics, and they are
what the submission package is for.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from . import dataset as ds
from .landmarks import case_stats as _case_stats
from .landmarks import read_landmarks
from .metrics import CASE_COLUMNS, add_ranks, aggregate

SUBMISSION_LANDMARKS_DIR = "landmarks"
SUBMISSION_COVER = "registration-results.csv"


def _warp_index(warped_dir) -> pd.DataFrame:
    p = Path(warped_dir) / "warp_index.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(
        columns=["case_id", "method", "warped_path", "time_min", "error"]
    )


def _blank_row(c, method: str, n_landmarks: int, time_min: float) -> dict:
    return {
        "case_id": c.case_id,
        "tissue": c.tissue,
        "scale": c.scale,
        "status": c.status,
        "source_image": c.source_image,
        "target_image": c.target_image,
        "method": method,
        "n_landmarks": int(n_landmarks),
        "scored": False,
        "rtre_median": float("nan"),
        "rtre_mean": float("nan"),
        "rtre_max": float("nan"),
        "tre_median_px": float("nan"),
        "robustness": float("nan"),
        "rank_median_rtre": float("nan"),
        "time_min": time_min,
        "imputed_initial": False,
    }


def score_method(cases, method: str, warped_dir, landmarks_root) -> tuple:
    """``(rows, missing)`` for one method over ``cases``.

    A case with no warped file is listed in ``missing`` AND, when its landmarks
    are public, scored at the initial pose with ``imputed_initial = True`` --
    the challenge's rule for a missing or incomplete registration, so a method
    that silently dropped half its cases ranks last on them instead of
    vanishing from the comparison.
    """
    warped_dir = Path(warped_dir)
    index = _warp_index(warped_dir).set_index("case_id")
    rows, missing = [], []
    for c in cases:
        target_path = ds.resolve_landmarks(c, landmarks_root, "target")
        source_path = ds.resolve_landmarks(c, landmarks_root, "source")
        scorable = c.scored_locally and target_path.exists() and source_path.exists()
        warped_path = warped_dir / f"{c.case_id}.csv"
        if not warped_path.exists():
            missing.append(
                {"case_id": c.case_id, "method": method, "reason": "no warped file"}
            )
            if not scorable:
                continue
            source = read_landmarks(source_path)
            row = _blank_row(c, method, len(source), float("nan"))
            stats = _case_stats(source, read_landmarks(target_path), source, c.diagonal)
            row.update({k: v for k, v in stats.items() if k in row})
            row["scored"] = True
            row["imputed_initial"] = True
            rows.append(row)
            continue
        warped = read_landmarks(warped_path)
        time_min = float("nan")
        if c.case_id in index.index:
            time_min = float(index.loc[c.case_id, "time_min"])
        row = _blank_row(c, method, len(warped), time_min)
        if scorable:
            stats = _case_stats(
                warped,
                read_landmarks(target_path),
                read_landmarks(source_path),
                c.diagonal,
            )
            row.update({k: v for k, v in stats.items() if k in row})
            row["scored"] = True
        rows.append(row)
    return rows, missing


def reg_eval_table(cases_df: pd.DataFrame) -> pd.DataFrame:
    """Scored rows in ``benchmarks.analysis.lib.load.load_reg_eval``'s contract.

    ``mode`` is the registration method exactly as the pipeline names it
    (``valis``, ``tiled``), which is what the arm and sweep tables key on; the
    ``initial`` and ``bunwarpj`` rows are carried too and simply match no run.
    """
    scored = cases_df[cases_df["scored"].astype(bool)]
    return pd.DataFrame(
        {
            "pair_id": scored["case_id"].astype(int),
            "mode": scored["method"],
            "true_median_px": scored["tre_median_px"],
            "true_median_rtre": scored["rtre_median"],
        }
    ).reset_index(drop=True)


def evaluate(cases, warped_dirs: dict, landmarks_root) -> tuple:
    """``(cases_df, aggregates_df, missing_df)`` over ``{method: warped_dir}``."""
    rows, missing = [], []
    for method, d in warped_dirs.items():
        r, m = score_method(cases, method, d, landmarks_root)
        rows.extend(r)
        missing.extend(m)
    cases_df = add_ranks(pd.DataFrame(rows, columns=CASE_COLUMNS))
    agg = aggregate(cases_df)
    missing_df = pd.DataFrame(missing, columns=["case_id", "method", "reason"])
    return cases_df, agg, missing_df


def write_submission(
    cover: pd.DataFrame, cases, warped_dir, out_dir, method: str
) -> Path:
    """Build ``<out_dir>/<method>/`` in the challenge's upload layout and zip it.

    The cover table is the challenge's own CSV with two columns filled in:
    ``Warped source landmarks`` (a path relative to the package root) and
    ``Execution time [minutes]``. Only cases with a warped file are listed; the
    evaluator scores what is present and reports the coverage.
    """
    warped_dir = Path(warped_dir)
    pkg = Path(out_dir) / method
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / SUBMISSION_LANDMARKS_DIR).mkdir(parents=True)
    index = _warp_index(warped_dir)
    times = dict(zip(index["case_id"].astype(int), index["time_min"]))
    sub = cover.copy()
    sub[ds.COL_WARPED_SOURCE] = ""
    sub[ds.COL_TIME] = float("nan")
    keep = []
    for c in cases:
        src = warped_dir / f"{c.case_id}.csv"
        if not src.exists():
            continue
        rel = f"{SUBMISSION_LANDMARKS_DIR}/{c.case_id}.csv"
        shutil.copyfile(src, pkg / rel)
        sub.loc[c.case_id, ds.COL_WARPED_SOURCE] = rel
        sub.loc[c.case_id, ds.COL_TIME] = times.get(c.case_id, float("nan"))
        keep.append(c.case_id)
    sub = sub.loc[keep]
    sub.to_csv(pkg / SUBMISSION_COVER, index=True)
    archive = shutil.make_archive(str(pkg), "zip", root_dir=pkg)
    return Path(archive)


def _parse_warped(items) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--warped expects method=dir, got {item!r}")
        method, d = item.split("=", 1)
        out[method] = d
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Score ANHIR warped landmarks; package a submission."
    )
    ap.add_argument(
        "--dataset", required=True, help="the challenge cover CSV (dataset_medium.csv)"
    )
    ap.add_argument(
        "--landmarks-root", required=True, help="root of the landmark archive"
    )
    ap.add_argument(
        "--warped",
        action="append",
        required=True,
        metavar="METHOD=DIR",
        help="a warped-landmark directory from warp.py; repeatable",
    )
    ap.add_argument("--out", required=True, help="output directory for the tables")
    ap.add_argument("--status", default="all", choices=("all",) + ds.STATUSES)
    ap.add_argument(
        "--tissue", action="append", help="restrict to these tissues; repeatable"
    )
    ap.add_argument(
        "--submit",
        action="append",
        default=[],
        help="also build the challenge submission package for this method; repeatable",
    )
    a = ap.parse_args(argv)

    cases = ds.select_cases(ds.load_cases(a.dataset), status=a.status, tissues=a.tissue)
    warped = _parse_warped(a.warped)
    cases_df, agg, missing = evaluate(cases, warped, a.landmarks_root)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases_df.to_csv(out / "anhir_cases.csv", index=False)
    agg.to_csv(out / "anhir_aggregates.csv", index=False)
    missing.to_csv(out / "anhir_missing.csv", index=False)
    reg_eval_table(cases_df).to_csv(out / "anhir_reg_eval.csv", index=False)
    scored = int(cases_df["scored"].sum())
    print(
        f"{len(cases_df)} case x method rows ({scored} scored, {len(missing)} missing) -> {out}"
    )
    for _, r in agg[agg["subset"] == "all"].iterrows():
        print(
            f"  {r['method']:<10} n={int(r['n_cases']):<4} avg-median-rTRE={r['avg_median_rtre']:.5f} "
            f"avg-robustness={r['avg_robustness']:.3f} avg-rank={r['avg_rank_median_rtre']:.2f}"
        )

    if a.submit:
        cover = ds.load_cover(a.dataset)
        for method in a.submit:
            if method not in warped:
                raise SystemExit(f"--submit {method}: no --warped {method}=... given")
            z = write_submission(
                cover, cases, warped[method], out / "submission", method
            )
            print(f"  submission package for {method}: {z}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
