"""The one spatial join of warp_seg_qc's per-cell residuals onto segmentation labels.

Used by CELL_QC (per-round QC columns) and EXPORT_SPATIALDATA (obsm residuals), so the
two can never pair a QC nucleus to a different cell. Both sides are CENTRE-of-pixel:
the residual CSV's ref_x/ref_y trace back to mask_to_geojson.py's rings, and callers
must pass the quantification table's raw x/y (see
tests/test_join_reg_residuals_convention.py).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INSTANCE_KEY = "label"


def join_one(
    path: str, centroids_xy: np.ndarray, max_dist_px: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """Per-cell (residual_px, iou) from one moving slide's CSV.

    Several QC pairs can land on one cell (different segmentations); the WORST
    residual is kept, and the IoU of that same pair with it, because these columns
    exclude cells and the optimistic choice would hide exactly the cells they flag.
    """
    from scipy.spatial import cKDTree

    stats: Dict = {"qc_pairs": 0, "joined": 0}
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError) as exc:
        logger.warning("skipping unreadable residual CSV %s: %s", path, exc)
        return None, None, stats
    if df.empty or not {"ref_x", "ref_y", "residual_px"} <= set(df.columns):
        logger.warning("residual CSV %s has no usable rows; skipping", path)
        return None, None, stats
    n = len(centroids_xy)
    resid = np.full(n, np.nan, dtype=float)
    iou = np.full(n, np.nan, dtype=float)
    if n == 0:
        return resid, iou, stats

    # cKDTree raises on a non-finite row (a cell whose centroid could not be computed,
    # e.g. a degenerate mask region). Build the tree on the finite rows only and map
    # its (compacted) indices back to the original cell positions; a NaN-centroid cell
    # is simply never a query target and keeps its NaN, same as an unmatched one.
    finite_mask = np.all(np.isfinite(centroids_xy), axis=1)
    finite_idx = np.nonzero(finite_mask)[0]
    if finite_idx.size == 0:
        stats = {"qc_pairs": int(len(df)), "joined": 0}
        return resid, iou, stats

    tree = cKDTree(centroids_xy[finite_idx])
    dist, idx = tree.query(
        df[["ref_x", "ref_y"]].to_numpy(dtype=float),
        distance_upper_bound=float(max_dist_px),
    )
    ok = np.isfinite(dist)
    r_vals = df["residual_px"].to_numpy(dtype=float)
    i_vals = (
        df["iou"].to_numpy(dtype=float)
        if "iou" in df.columns
        else np.full(len(df), np.nan)
    )
    for compact_i, r, i in zip(idx[ok], r_vals[ok], i_vals[ok]):
        cell_i = finite_idx[compact_i]
        if np.isnan(resid[cell_i]) or r > resid[cell_i]:
            resid[cell_i] = r
            iou[cell_i] = i
    stats = {"qc_pairs": int(len(df)), "joined": int(ok.sum())}
    return resid, iou, stats


def join_reg_residuals(
    residual_paths: List[str],
    centroids_xy: np.ndarray,
    labels: np.ndarray,
    max_dist_px: float,
) -> Tuple[pd.DataFrame, Dict]:
    """Spatially join per-cell registration residuals onto segmentation labels.

    ``warp_seg_qc`` scores a *separate* StarDist run on each slide's native image
    (see ``modules/local/seg_qc_geojson.nf``), so its cells share **no label
    space** with ``cell_mask``. The two do share a coordinate frame though — the
    residual CSV reports reference centroids in the registered reference frame,
    which is the frame ``SEGMENT`` ran on — so the join is spatial: each QC pair
    is assigned to the nearest ``cell_mask`` centroid within ``max_dist_px``.

    Returns ``(residuals, stats)`` where ``residuals`` is a cells x moving-slides
    frame of displacement in pixels (NaN where a cell was never matched) and
    ``stats`` records how well the join went — an unmatched cell means "no QC
    evidence", NOT "well registered", and conflating those would invert the
    signal's meaning.
    """
    out = pd.DataFrame(index=pd.Index(labels, name=INSTANCE_KEY))
    stats: Dict = {"max_dist_px": float(max_dist_px), "slides": {}}
    if not residual_paths or centroids_xy.size == 0:
        return out, stats
    for p in residual_paths:
        resid, _iou, one = join_one(p, centroids_xy, max_dist_px)
        if resid is None:
            continue
        try:
            moving = str(pd.read_csv(p, usecols=["moving"], nrows=1)["moving"].iloc[0])
        except (ValueError, KeyError, IndexError):
            moving = Path(p).stem
        out[moving] = resid
        covered = int(np.isfinite(resid).sum())
        stats["slides"][moving] = {
            "qc_pairs": one["qc_pairs"],
            "joined": one["joined"],
            "join_fraction": one["joined"] / max(one["qc_pairs"], 1),
            "cells_with_residual": covered,
            "cell_coverage": covered / max(len(labels), 1),
        }
        logger.info(
            "  %s: %d/%d QC pairs joined, %.1f%% of cells covered",
            moving,
            one["joined"],
            one["qc_pairs"],
            100 * stats["slides"][moving]["cell_coverage"],
        )
    return out, stats
