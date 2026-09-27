#!/usr/bin/env python3
"""Per-cell QC columns for one patient (CELL_QC).

Adds every "QC: ..." column (spec 2026-09-27 §2) to MERGE_QUANT_CSVS's table and
republishes it as merged_quant.csv, so EXPORT_GEOJSON, EXPORT_SPATIALDATA and the
postprocessed checkpoint all read ONE table. Also writes the long per-round table and
the round manifest for analysis. Recomputing is idempotent: QC columns this run owns are
dropped and rebuilt; round columns of rounds not in this run's manifest (add_cycle's
prior rounds) are kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "utils"))

from logger import configure_logging, get_logger  # noqa: E402
from measurements import (  # noqa: E402
    QC_NUCLEAR_RETENTION,
    QC_REG_DICE,
    QC_REG_DISPLACEMENT,
    QC_ROUND_METRICS,
    QC_TOTAL_INTENSITY,
    is_qc_column,
    measurement_key,
    qc_key,
)
from metadata import is_nuclear  # noqa: E402
from reg_residuals import join_one  # noqa: E402

logger = get_logger(__name__)
_CELL_MEDIAN = ": Cell: Median"


def _base_markers(columns) -> List[str]:
    return [
        c[: -len(_CELL_MEDIAN)]
        for c in columns
        if isinstance(c, str) and c.endswith(_CELL_MEDIAN) and not is_qc_column(c)
    ]


def total_intensity(df: pd.DataFrame, nuclear_markers: List[str]) -> pd.Series:
    cols = [
        measurement_key(m, "Cell", "Median")
        for m in _base_markers(df.columns)
        if not is_nuclear(m, nuclear_markers)
    ]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].sum(axis=1, min_count=1)


def normalised_retention(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref, dtype=float)
    mov = np.asarray(mov, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(ref > 0, mov / ref, np.nan)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return np.full(raw.shape, np.nan)
    med = float(np.median(finite))
    if med <= 0:
        logger.warning("round median retention is %s; the whole round reads NaN", med)
        return np.full(raw.shape, np.nan)
    return raw / med


def _round_markers(entry: Dict, nuclear_markers: List[str]) -> List[str]:
    return sorted(m for m in entry.get("markers", []) if not is_nuclear(m, nuclear_markers))


def _reference_nuclear(df: pd.DataFrame, nuclear_markers: List[str]) -> Optional[str]:
    for m in _base_markers(df.columns):
        if is_nuclear(m, nuclear_markers):
            return m
    return None


def add_qc_columns(
    quant: pd.DataFrame,
    rounds: List[Dict],
    retention_dir: Path,
    residual_dir: Path,
    pixel_size: float,
    join_max_px: float,
    nuclear_markers: List[str],
) -> pd.DataFrame:
    out = quant.copy()
    owned = {qc_key(QC_TOTAL_INTENSITY)}
    moving = [
        r for r in rounds
        if not r.get("is_reference") and _round_markers(r, nuclear_markers)
    ]
    for r in moving:
        owned.update(qc_key(m, _round_markers(r, nuclear_markers)) for m in QC_ROUND_METRICS)
    out = out.drop(columns=[c for c in out.columns if c in owned])

    new: Dict[str, pd.Series] = {qc_key(QC_TOTAL_INTENSITY): total_intensity(out, nuclear_markers)}
    ref_marker = _reference_nuclear(out, nuclear_markers)
    labels = out["label"].to_numpy()
    quant_xy = out  # centre-of-pixel x/y, straight off the table (see reg_residuals.py)
    centroids = quant_xy[["x", "y"]].to_numpy(dtype=float)

    for r in moving:
        markers = _round_markers(r, nuclear_markers)
        if r.get("retention_csv") and ref_marker is not None:
            ret = pd.read_csv(Path(retention_dir) / r["retention_csv"])
            comp = (
                "Nucleus"
                if "Nucleus" in ret.columns
                and measurement_key(ref_marker, "Nucleus", "Median") in out.columns
                else "Cell"
            )
            if comp in ret.columns:
                mov = ret.set_index("label")[comp].reindex(labels).to_numpy(dtype=float)
                ref = out[measurement_key(ref_marker, comp, "Median")].to_numpy(dtype=float)
                new[qc_key(QC_NUCLEAR_RETENTION, markers)] = pd.Series(
                    normalised_retention(ref, mov), index=out.index
                )
            else:
                logger.warning("%s: no retention values; no retention key for this round", r["round_id"])
        if r.get("residual_csv"):
            resid, iou, stats = join_one(
                str(Path(residual_dir) / r["residual_csv"]), centroids, join_max_px
            )
            logger.info("%s: residual join %s", r["round_id"], stats)
            if resid is not None:
                new[qc_key(QC_REG_DISPLACEMENT, markers)] = pd.Series(
                    resid * float(pixel_size), index=out.index
                )
                with np.errstate(invalid="ignore", divide="ignore"):
                    dice = 2.0 * iou / (1.0 + iou)
                new[qc_key(QC_REG_DICE, markers)] = pd.Series(dice, index=out.index)
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def round_long_table(table: pd.DataFrame, rounds: List[Dict], pixel_size: float) -> pd.DataFrame:
    frames = []
    nan = pd.Series(np.nan, index=table.index)
    for r in rounds:
        if r.get("is_reference") or not r.get("markers"):
            continue
        markers = sorted(r["markers"])
        try:
            keys = {m: qc_key(m, markers) for m in QC_ROUND_METRICS}
        except ValueError:
            continue
        if not any(k in table.columns for k in keys.values()):
            continue
        disp_um = table.get(keys[QC_REG_DISPLACEMENT], nan)
        frames.append(pd.DataFrame({
            "label": table["label"].to_numpy(),
            "round_id": r["round_id"],
            "markers": "|".join(markers),
            "nuclear_retention": table.get(keys[QC_NUCLEAR_RETENTION], nan).to_numpy(),
            "displacement_px": (disp_um / float(pixel_size)).to_numpy(),
            "displacement_um": disp_um.to_numpy(),
            "dice": table.get(keys[QC_REG_DICE], nan).to_numpy(),
        }))
    cols = ["label", "round_id", "markers", "nuclear_retention",
            "displacement_px", "displacement_um", "dice"]
    return pd.concat(frames, ignore_index=True)[cols] if frames else pd.DataFrame(columns=cols)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--merged", required=True)
    p.add_argument("--rounds", required=True)
    p.add_argument("--retention-dir", required=True)
    p.add_argument("--residual-dir", required=True)
    p.add_argument("--pixel-size", type=float, required=True)
    p.add_argument("--join-max-px", type=float, required=True)
    p.add_argument("--nuclear-markers", nargs="+", required=True)
    p.add_argument("--patient-id", required=True)
    p.add_argument("--prior-rounds", default=None)
    p.add_argument("--out-merged", required=True)
    p.add_argument("--out-round-qc", required=True)
    p.add_argument("--out-rounds", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    configure_logging()
    a = parse_args(argv)
    rounds = json.loads(Path(a.rounds).read_text())
    quant = pd.read_csv(a.merged)
    table = add_qc_columns(
        quant, rounds, Path(a.retention_dir), Path(a.residual_dir),
        a.pixel_size, a.join_max_px, a.nuclear_markers,
    )
    manifest = [
        {"round_id": r["round_id"], "is_reference": bool(r.get("is_reference")),
         "markers": _round_markers(r, a.nuclear_markers)}
        for r in rounds
    ]
    if a.prior_rounds:
        seen = {r["round_id"] for r in manifest}
        manifest = [r for r in json.loads(Path(a.prior_rounds).read_text())
                    if r["round_id"] not in seen] + manifest
    table.to_csv(a.out_merged, index=False)
    round_long_table(table, manifest, a.pixel_size).to_csv(a.out_round_qc, index=False)
    Path(a.out_rounds).write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("%s: %d QC columns", a.patient_id, sum(is_qc_column(c) for c in table.columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
