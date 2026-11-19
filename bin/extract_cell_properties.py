#!/usr/bin/env python3
"""Extract cell morphology and contours from a segmentation mask.

Computes regionprops (morphology) and polygon contours once per mask,
so downstream processes (QUANTIFY, EXPORT_GEOJSON) don't need to recompute them.

Outputs:
    - morphology.csv: Per-cell morphological properties
    - contours.json: Per-cell simplified polygon contours for GeoJSON export
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from skimage.measure import (
    approximate_polygon,
    find_contours,
    regionprops_table,
)

# Add parent directory to path to import lib modules
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from image_utils import ensure_dir, load_image
from logger import configure_logging, get_logger
from pixel_convention import centre_to_corner

logger = get_logger(__name__)


def extract_morphology(
    mask: NDArray,
) -> Tuple[Optional[pd.DataFrame], Optional[NDArray], Optional[NDArray]]:
    """Compute morphological properties for all cells in the mask.

    Parameters
    ----------
    mask : ndarray, shape (Y, X)
        Segmentation mask with cell labels (background=0).

    Returns
    -------
    props_df : DataFrame or None
        Morphological properties with columns:
        label, y, x, area, eccentricity, perimeter, convex_area,
        axis_major_length, axis_minor_length, plus bbox columns
        (bbox-0..bbox-3) used internally by extract_contours.
    mask : ndarray or None
        Input mask (unchanged, no filtering applied).
    valid_labels : ndarray or None
        Array of valid (non-background) label IDs.
    """
    mask = np.ascontiguousarray(mask.squeeze())

    # `mask.max()` is one linear pass, unlike a full-mask np.unique sort. The label set
    # comes from regionprops_table below (tests/test_no_full_mask_unique.py).
    if int(mask.max()) == 0:
        logger.warning("No cells found in mask")
        return None, None, None

    props = regionprops_table(
        mask,
        properties=[
            "label",
            "centroid",
            "area",
            "eccentricity",
            "perimeter",
            "convex_area",
            "axis_major_length",
            "axis_minor_length",
            "solidity",
            "bbox",
        ],
    )
    props_df = pd.DataFrame(props).set_index("label")
    props_df.rename(
        columns={"centroid-0": "y", "centroid-1": "x"},
        inplace=True,
    )

    # The label set from regionprops_table: one row per non-zero label, ascending.
    valid_labels = props_df.index.to_numpy()

    logger.info(f"Morphology computed for {len(props_df)} cells")
    return props_df, mask, valid_labels


def _label_bboxes(props_df: pd.DataFrame):
    """Yield ``(label, minr, minc, maxr, maxc)`` for each row, in row order.

    Avoids ``props_df.iterrows()``, which builds one pandas Series per row.
    """
    labels = props_df.index.to_numpy()
    bboxes = props_df[["bbox-0", "bbox-1", "bbox-2", "bbox-3"]].to_numpy()
    for label, (minr, minc, maxr, maxc) in zip(labels, bboxes):
        yield label, int(minr), int(minc), int(maxr), int(maxc)


def extract_contours(
    mask_filtered: NDArray,
    props_df: pd.DataFrame,
    simplify_tolerance: float = 1.0,
    coord_precision: int = 2,
) -> Dict[str, List[List[float]]]:
    """Extract simplified polygon contours for each cell.

    Uses find_contours on each cell's bounding box crop for efficiency,
    then simplifies with Ramer-Douglas-Peucker via approximate_polygon.

    Parameters
    ----------
    mask_filtered : ndarray, shape (Y, X)
        Filtered segmentation mask.
    props_df : DataFrame
        Morphology DataFrame indexed by label, must contain bbox columns
        (bbox-0, bbox-1, bbox-2, bbox-3) from regionprops_table.
    simplify_tolerance : float
        Douglas-Peucker simplification tolerance in pixels. Higher = fewer
        polygon vertices = smaller GeoJSON and faster QuPath import. The exported
        polygon is display-only (all quantification is mask-derived), so this does
        not affect downstream gating/analysis.
    coord_precision : int
        Decimal places to round vertex coordinates to. 2 (0.01 px) is far below
        one pixel, so it shrinks the GeoJSON with no visible/analysis impact.

    Returns
    -------
    contours : dict
        Mapping of str(label) -> [[x, y], ...] closed polygon ring
        in pixel coordinates.
    """
    logger.info(
        f"Extracting contours for {len(props_df)} cells (tolerance={simplify_tolerance})..."
    )

    contours_dict: Dict[str, List[List[float]]] = {}
    skipped_no_contour = 0
    skipped_too_simple = 0

    for label, minr, minc, maxr, maxc in _label_bboxes(props_df):
        # Crop binary mask for this cell, padded by 1px to ensure closed contours
        # for cells touching the image border (find_contours returns open contours
        # at array edges, which would produce incorrect polygon geometry)
        binary_crop = np.pad(
            (mask_filtered[minr:maxr, minc:maxc] == label).astype(np.uint8),
            pad_width=1,
            mode="constant",
            constant_values=0,
        )
        contour_list = find_contours(binary_crop, level=0.5)

        if not contour_list:
            skipped_no_contour += 1
            continue

        # Take the longest contour (outer boundary)
        contour = max(contour_list, key=len)

        # Offset back to full image coordinates (subtract 1 for padding, add bbox origin)
        contour[:, 0] += minr - 1  # y offset
        contour[:, 1] += minc - 1  # x offset

        # skimage centre-of-pixel -> ImageJ/QuPath corner-of-pixel. This differs from
        # mask_to_geojson.py on purpose (that one feeds the VALIS reg-QC warp); see
        # bin/utils/pixel_convention.py before changing either.
        contour = centre_to_corner(contour)

        # Simplify with Douglas-Peucker
        simplified = approximate_polygon(contour, tolerance=simplify_tolerance)

        # Need at least 4 vertices for a valid polygon (3 unique + closing point)
        if len(simplified) < 4:
            skipped_too_simple += 1
            continue

        # Close the ring (first == last) for GeoJSON compliance
        if not np.array_equal(simplified[0], simplified[-1]):
            simplified = np.vstack([simplified, simplified[0:1]])

        # Convert from (y, x) to (x, y) for GeoJSON and round for compact JSON
        contours_dict[str(label)] = [
            [round(float(pt[1]), coord_precision), round(float(pt[0]), coord_precision)]
            for pt in simplified
        ]

    skipped = skipped_no_contour + skipped_too_simple
    logger.info(
        f"Extracted {len(contours_dict)} contours, skipped {skipped} "
        f"({skipped_no_contour} no contour, {skipped_too_simple} too few vertices)"
    )
    return contours_dict


def pair_labels_to_reference(
    label_mask: NDArray,
    reference_mask: NDArray,
) -> Dict[int, int]:
    """Map each label in ``label_mask`` to the dominant overlapping ``reference_mask`` label.

    Used to attribute nucleus instances (``label_mask``) to whole-cell instances
    (``reference_mask``). When the two masks share label IDs (StarDist / CellSAM,
    where the cell mask is the expanded nucleus mask) this resolves to the identity
    mapping; when they do not (InstanSeg or any independent dual-mask backend) it
    pairs by maximal pixel overlap. Only nonzero reference labels are considered.

    Returns
    -------
    dict
        ``{label: reference_label}`` for every label that overlaps a cell.
    """
    flat = np.ascontiguousarray(label_mask).ravel()
    ref = np.ascontiguousarray(reference_mask).ravel()
    fg = (flat != 0) & (ref != 0)
    if not np.any(fg):
        return {}

    # Overlap counts as a sparse matrix: rows = label, cols = reference, value = shared
    # pixels. coo -> csr sums the duplicates in one linear pass, without materialising
    # one row per foreground pixel.
    lbl = flat[fg].astype(np.int64, copy=False)
    rf = ref[fg].astype(np.int64, copy=False)
    overlaps = sparse.coo_matrix(
        (np.ones(lbl.size, dtype=np.int32), (lbl, rf)),
        shape=(int(lbl.max()) + 1, int(rf.max()) + 1),
    ).tocsr()

    # Ties go to the smallest reference label: csr indices are sorted ascending within
    # a row, so `.min()` over the tied columns picks it (tests/test_pair_labels_memory.py).
    indptr, indices, data = overlaps.indptr, overlaps.indices, overlaps.data
    mapping: Dict[int, int] = {}
    for label in range(1, overlaps.shape[0]):  # label 0 is background
        start, end = indptr[label], indptr[label + 1]
        if start == end:
            continue
        row = data[start:end]
        mapping[label] = int(indices[start:end][row == row.max()].min())
    return mapping


def rekey_contours_to_reference(
    contours: Dict[str, List[List[float]]],
    mapping: Dict[int, int],
    areas: Dict[int, float],
) -> Dict[str, List[List[float]]]:
    """Re-key nucleus contours from nucleus labels to their paired cell labels.

    If several nuclei map to the same cell, the largest nucleus (by area) wins, so
    each cell carries a single nucleus polygon (QuPath cell objects hold one nucleus).
    """
    out: Dict[str, List[List[float]]] = {}
    chosen_area: Dict[str, float] = {}
    for label_str, ring in contours.items():
        nuc = int(label_str)
        cell = mapping.get(nuc)
        if cell is None:
            continue
        cell_str = str(cell)
        area = float(areas.get(nuc, 0.0))
        if cell_str not in out or area > chosen_area[cell_str]:
            out[cell_str] = ring
            chosen_area[cell_str] = area
    return out


def run_extraction(
    mask_path: str,
    outdir: str,
    simplify_tolerance: float = 1.0,
    reference_mask_path: Optional[str] = None,
    coord_precision: int = 2,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """Run full extraction pipeline: morphology + contours.

    Parameters
    ----------
    mask_path : str
        Path to segmentation mask (.tif or .npy).
    outdir : str
        Output directory.
    simplify_tolerance : float
        Contour simplification tolerance.

    Returns
    -------
    morphology_df : DataFrame or None
    contours : dict or None
    """
    logger.info("=" * 60)
    logger.info("Extracting cell properties")
    logger.info("=" * 60)

    # Load mask
    logger.info(f"Loading mask: {mask_path}")
    if mask_path.endswith(".npy"):
        mask = np.load(mask_path).squeeze()
    else:
        mask, _ = load_image(mask_path)
        mask = mask.squeeze()
    logger.info(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")

    # Extract morphology
    props_df, mask_filtered, valid_labels = extract_morphology(mask)

    if props_df is None:
        logger.warning("No cells found in mask — writing empty outputs")
        empty_df = pd.DataFrame(
            columns=[
                "label",
                "y",
                "x",
                "area",
                "eccentricity",
                "perimeter",
                "convex_area",
                "axis_major_length",
                "axis_minor_length",
                "solidity",
            ]
        )
        morphology_path = str(Path(outdir) / "morphology.csv")
        contours_path = str(Path(outdir) / "contours.json")
        empty_df.to_csv(morphology_path, index=False)
        with open(contours_path, "w") as f:
            json.dump({}, f)
        return None, None

    # Extract contours (reuse bbox from morphology — no second regionprops call)
    contours = extract_contours(
        mask_filtered, props_df, simplify_tolerance, coord_precision
    )

    # Optionally re-key contours from this mask's labels to a reference mask's labels.
    # Used for the nucleus mask: emit nucleus contours keyed by CELL label so that
    # EXPORT_GEOJSON attaches each nucleus polygon to the right cell via identity lookup.
    if reference_mask_path:
        logger.info(f"Re-keying contours to reference mask: {reference_mask_path}")
        if reference_mask_path.endswith(".npy"):
            reference_mask = np.load(reference_mask_path).squeeze()
        else:
            reference_mask, _ = load_image(reference_mask_path)
            reference_mask = reference_mask.squeeze()
        mapping = pair_labels_to_reference(mask_filtered, reference_mask)
        areas = props_df["area"].to_dict() if "area" in props_df.columns else {}
        before = len(contours)
        contours = rekey_contours_to_reference(contours, mapping, areas)
        logger.info(f"Re-keyed {before} nucleus contours -> {len(contours)} cells")

    # Save morphology CSV
    morphology_path = str(Path(outdir) / "morphology.csv")
    morphology_out = props_df.reset_index()
    morpho_cols = [
        "label",
        "y",
        "x",
        "area",
        "eccentricity",
        "perimeter",
        "convex_area",
        "axis_major_length",
        "axis_minor_length",
        "solidity",
    ]
    morphology_out = morphology_out[morpho_cols]
    morphology_out.to_csv(morphology_path, index=False)
    logger.info(f"Saved morphology: {morphology_path} ({len(morphology_out)} cells)")

    # Save contours JSON
    contours_path = str(Path(outdir) / "contours.json")
    with open(contours_path, "w") as f:
        json.dump(contours, f)
    contours_size_mb = Path(contours_path).stat().st_size / (1024 * 1024)
    logger.info(
        f"Saved contours: {contours_path} ({len(contours)} cells, {contours_size_mb:.1f} MB)"
    )

    logger.info("Extraction complete")
    return morphology_out, contours


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract cell morphology and contours from segmentation mask",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mask_file",
        type=str,
        required=True,
        help="Path to segmentation mask (.tif or .npy)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory",
    )
    parser.add_argument(
        "--simplify_tolerance",
        type=float,
        default=1.0,
        help="Douglas-Peucker contour simplification tolerance in pixels. Higher = "
        "fewer polygon vertices = smaller GeoJSON and faster QuPath import. "
        "Display-only geometry; does not affect quantification.",
    )
    parser.add_argument(
        "--coord_precision",
        type=int,
        default=2,
        help="Decimal places for polygon vertex coordinates (2 = 0.01 px, sub-pixel). "
        "Lower = smaller GeoJSON; no analysis impact.",
    )
    parser.add_argument(
        "--reference_mask",
        type=str,
        default=None,
        help="Optional reference mask (.tif/.npy). When given, contours are re-keyed "
        "from this mask's labels to the dominant overlapping reference label "
        "(used to key nucleus contours by cell label).",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    configure_logging(level=logging.INFO)
    args = parse_args()

    ensure_dir(args.outdir)

    run_extraction(
        mask_path=args.mask_file,
        outdir=args.outdir,
        simplify_tolerance=args.simplify_tolerance,
        reference_mask_path=args.reference_mask,
        coord_precision=args.coord_precision,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
