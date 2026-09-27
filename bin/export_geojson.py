#!/usr/bin/env python3
"""Export cell data to QuPath-compatible GeoJSON.

Generates a QuPath-native GeoJSON FeatureCollection with raw marker intensities
and morphological measurements for all cells. No phenotype classification is
applied — gating is handled downstream by FlowPath in QuPath.

Outputs:
    - cells.geojson: QuPath-compatible cell detections (native array measurement format)
    - cells_data.csv: Full cell data with raw intensities and z-scores per marker
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from scipy.stats import zscore

sys.path.insert(0, str(Path(__file__).parent / "utils"))

from image_utils import ensure_dir
from logger import configure_logging, get_logger
from measurements import (
    MORPH_EXPORT,
    MORPHOLOGY_COLS,
    identify_marker_columns,
    is_qc_column,
    morph_key,
)
from pixel_convention import centre_to_corner  # noqa: E402
from pixel_size import resolve_pixel_size  # noqa: E402

logger = get_logger(__name__)

# Columns that are morphological / metadata, not marker intensities
# (single source of truth: bin/utils/measurements.py). Two derived names from
# the SAME import, for two different uses:
#   - _MORPHOLOGY_COL_ORDER keeps the tuple's producer order, captured before the
#     rebind below. build_measurements() below spreads it into field_cols, and
#     Python randomises string-hash order per process -- iterating a *set* there
#     would make field_cols order, dict insertion order, and (if anything ever
#     iterates a row instead of using row.get()) key order all vary run to run.
#     Nothing currently reads that order except positionally-fixed row.get()
#     calls, so this is inert today, but a GeoJSON export is a byte-exact
#     external contract (qupath-extension-flowpath) and should not depend on
#     hash-order luck.
#   - MORPHOLOGY_COLS is then rebound to a set, unchanged from before, for the
#     O(1) membership tests this module does in a loop.
_MORPHOLOGY_COL_ORDER = tuple(MORPHOLOGY_COLS)
MORPHOLOGY_COLS = set(MORPHOLOGY_COLS)

# Default classification color for "Cell" (cyan)
CELL_COLOR_RGB = (0, 255, 255)


def rgb_to_qupath_color(r: int, g: int, b: int, a: int = 255) -> int:
    """Convert RGB to QuPath's signed 32-bit ARGB color format."""
    value = (a << 24) | (r << 16) | (g << 8) | b
    if value >= 0x80000000:
        value -= 0x100000000
    return value


def compute_zscores(df: pd.DataFrame, marker_cols: List[str]) -> pd.DataFrame:
    """Compute z-scores for marker columns and add as new columns.

    Parameters
    ----------
    df : DataFrame
        Cell data with raw marker intensities.
    marker_cols : list of str
        Marker column names to z-score.

    Returns
    -------
    DataFrame
        Input dataframe with added z-score columns (e.g. 'CD45_zscore').
    """
    if not marker_cols:
        return df

    df = df.copy()
    marker_data = df[marker_cols]

    # Compute z-scores (handle constant columns gracefully)
    z_values = pd.DataFrame(
        zscore(marker_data, axis=0, nan_policy="omit"),
        index=marker_data.index,
        columns=marker_data.columns,
    )
    # For constant columns (std=0), zscore produces NaN for all rows — replace with 0.
    # But preserve NaN for cells that had missing raw data (those should stay NaN).
    for col in marker_cols:
        is_missing_raw = marker_data[col].isna()
        col_std = marker_data[col].std()
        is_constant_col = col_std == 0 or pd.isna(col_std)
        if is_constant_col:
            z_values.loc[~is_missing_raw, col] = 0.0

    for col in marker_cols:
        df[f"{col}_zscore"] = z_values[col]

    return df


def _iter_rows_positional(df: pd.DataFrame, marker_cols: List[str], qc_cols=()):
    """Yield ``(idx, x_px, y_px, row)`` for every row of ``df``, without ``iterrows()``.

    ``iterrows()`` materialises one pandas Series per row -- 500k of them on a
    whole-slide image, on the process with a 32 GB reservation. The columns
    ``build_measurements`` can read -- ``MORPHOLOGY_COLS`` plus ``marker_cols`` -- are
    instead pulled to a numpy array **once**, up front, and indexed positionally per row
    (the same seam ``extract_cell_properties._label_bboxes`` established for bboxes; see
    ``tests/test_no_full_mask_unique.py``). Deriving the column set from
    ``MORPHOLOGY_COLS`` rather than restating it here keeps this in sync with
    ``build_measurements`` automatically -- a second hand-copied list is exactly the
    kind of drift ``bin/utils/measurements.py`` was written to prevent.

    ``row`` is a plain ``dict`` of ``{column: value}`` for whichever columns are
    actually present in ``df`` -- a column absent from ``df`` is simply absent from
    every ``row``, so ``row.get(col)`` still returns ``None`` exactly as
    ``Series.get(col)`` did. That is a deliberate difference from a positional numpy
    lookup, which would raise ``KeyError``/``IndexError`` on a missing column instead.

    ``idx`` is ``df``'s own index value for that row (``df.index.to_numpy()``), not a
    freshly-counted position -- callers fall back to it (``row.get("label", idx)``) and
    a caller upstream may have already dropped rows (e.g. ``drop_duplicates``), leaving
    a non-contiguous index that a plain position counter would not reproduce.
    """
    field_cols = [*_MORPHOLOGY_COL_ORDER, *marker_cols, *qc_cols]
    idx_arr = df.index.to_numpy()
    col_arrays = {c: df[c].to_numpy() for c in field_cols if c in df.columns}
    x_arr = col_arrays.get("x")
    y_arr = col_arrays.get("y")

    for i in range(len(df)):
        row = {c: arr[i] for c, arr in col_arrays.items()}
        x_px = x_arr[i] if x_arr is not None else None
        y_px = y_arr[i] if y_arr is not None else None
        yield idx_arr[i], x_px, y_px, row


def build_measurements(
    row: pd.Series | Dict,
    marker_cols: List[str],
    pixel_size: float,
    qc_cols=(),
) -> List[Dict]:
    """Build QuPath-native measurement array for a single cell.

    Returns a list of {"name": ..., "value": ...} dicts matching QuPath's
    native GeoJSON serialization format, in order: ``label``, centroids,
    markers, ``QC: ...`` columns (in ``qc_cols`` order), then ``MORPH: ...``
    morphology fields.

    ``row`` only needs ``.get(key)`` / ``.get(key, default)`` -- a ``pd.Series`` (the
    original per-row shape from ``iterrows()``) and a plain ``dict`` (what
    ``_iter_rows_positional`` now hands in) are both accepted with identical
    behaviour, including that a key absent from either returns ``None``/the given
    default rather than raising.
    """
    measurements = []

    # Segmentation identity, emitted first. `label` ties this cell to its contour (the
    # lookup export_geojson() does ten lines below) and to every row of cells_data.csv,
    # and it was the one column the CSV kept and the GeoJSON dropped: MORPHOLOGY_COLS
    # excludes it from marker_cols, so nothing else in this function emits it.
    #
    # Joining CSV row i to feature i cannot stand in for it. A cell with a NaN centroid
    # stays in the CSV and never becomes a feature (the `skipped` counter in
    # export_geojson()), so the two orderings diverge the moment one row is skipped --
    # silently, and not detectably from the published artefacts alone.
    #
    # Safe on the QuPath side: FlowPath's MORPHOLOGY_PREFIXES filters "label" out of the
    # marker panel, so it does not appear as a gateable channel.
    label = row.get("label")
    if pd.notna(label):
        measurements.append({"name": "label", "value": int(label)})

    # Centroid in micrometres. Corner-of-pixel, matching the polygon this cell
    # is drawn as; the rule lives in bin/utils/pixel_convention.py.
    x_px = centre_to_corner(float(row.get("x", 0)))
    y_px = centre_to_corner(float(row.get("y", 0)))
    measurements.append({"name": "Centroid X µm", "value": round(x_px * pixel_size, 3)})
    measurements.append({"name": "Centroid Y µm", "value": round(y_px * pixel_size, 3)})

    # Raw marker intensities
    for col in marker_cols:
        val = row.get(col)
        if pd.notna(val):
            measurements.append({"name": col, "value": round(float(val), 4)})

    # QC: columns, verbatim (already carry their own "QC: " prefix), NaN omitted.
    for col in qc_cols:
        val = row.get(col)
        if pd.notna(val):
            measurements.append({"name": col, "value": round(float(val), 4)})

    # Morphological features, under their "MORPH: " names (converted to µm where
    # appropriate; unit_power is 0/1/2 for none/µm/µm²).
    for morph_col, display_name, unit_power in MORPH_EXPORT:
        val = row.get(morph_col)
        if pd.notna(val):
            measurements.append(
                {
                    "name": morph_key(display_name),
                    "value": round(float(val) * pixel_size**unit_power, 4),
                }
            )

    return measurements


def _polygon_geometry(
    contours: Optional[Dict[str, List[List[float]]]],
    label_str: str,
) -> Optional[Dict]:
    """Return a GeoJSON Polygon geometry for a label, or None if no contour exists."""
    if contours and label_str in contours:
        return {"type": "Polygon", "coordinates": [contours[label_str]]}
    return None


def build_feature(
    measurements: List[Dict],
    geometry: Dict,
    color_int: int,
    object_type: str = "detection",
    nucleus_geometry: Optional[Dict] = None,
    object_id: Optional[str] = "PathDetectionObject",
) -> Dict:
    """Assemble a single QuPath-native GeoJSON Feature.

    ``nucleus_geometry``, when provided, is written as a **top-level** member of the
    Feature (sibling of ``geometry``), exactly matching QuPath's own serialization
    of cell objects (qupath.lib.io.QuPathTypeAdapters). It is only meaningful when
    ``object_type == "cell"``.
    """
    feature: Dict = {"type": "Feature"}
    if object_id is not None:
        feature["id"] = object_id
    feature["geometry"] = geometry
    if nucleus_geometry is not None:
        feature["nucleusGeometry"] = nucleus_geometry
    feature["properties"] = {
        "objectType": object_type,
        "classification": {"name": "Cell", "colorRGB": color_int},
        "isLocked": False,
        "measurements": measurements,
    }
    return feature


def export_geojson(
    df: pd.DataFrame,
    output_path: str,
    pixel_size: float = 0.325,
    marker_cols: Optional[List[str]] = None,
    contours: Optional[Dict[str, List[List[float]]]] = None,
) -> int:
    """Export cells to QuPath-native GeoJSON format.

    Parameters
    ----------
    df : DataFrame
        Cell data with coordinates and marker intensities.
    output_path : str
        Path for output GeoJSON file.
    pixel_size : float
        Micrometers per pixel for coordinate conversion.
    marker_cols : list, optional
        Marker columns to include. If None, auto-detected.
    contours : dict, optional
        Pre-computed contours mapping str(label) -> [[x,y], ...] polygon ring.

    Returns
    -------
    int
        Number of cells exported.
    """
    if marker_cols is None:
        marker_cols = identify_marker_columns(df)
    qc_cols = [c for c in df.columns if is_qc_column(c)]

    logger.info(f"Exporting {len(df)} cells to GeoJSON: {output_path}")
    logger.info(f"  Markers: {marker_cols}")

    color_int = rgb_to_qupath_color(*CELL_COLOR_RGB)

    # A generator, not a list: _stream_collection writes each feature as it is produced, so a
    # slide's worth of features never coexists in memory. `skipped` is a one-element list
    # because the count is read after the generator has been consumed.
    skipped = [0]

    def _iter_features():
        for idx, x_px, y_px, row in _iter_rows_positional(df, marker_cols, qc_cols):
            if pd.isna(x_px) or pd.isna(y_px):
                skipped[0] += 1
                continue

            # regionprops centroids are centre-of-pixel; the contours this Point
            # falls back from are corner-of-pixel. Convert, so the Point fallback
            # and the Polygon land in the same place
            # (bin/utils/pixel_convention.py).
            x_px_corner = centre_to_corner(float(x_px))
            y_px_corner = centre_to_corner(float(y_px))

            cell_id = row.get("label", idx)
            cell_label_str = str(int(cell_id)) if pd.notna(cell_id) else str(idx)

            # Geometry: Polygon if contour available, else Point
            geometry = _polygon_geometry(contours, cell_label_str) or {
                "type": "Point",
                "coordinates": [x_px_corner, y_px_corner],
            }

            # Build measurements array (QuPath native format)
            measurements = build_measurements(row, marker_cols, pixel_size, qc_cols)
            # object_id=None: do not stamp every cell with the same constant id
            # ("PathDetectionObject"), which QuPath treats as a per-object UUID — a
            # shared id across all detections breaks re-import. (Matches the
            # compartment export path, which already passes object_id=None.)
            yield build_feature(measurements, geometry, color_int, object_id=None)

    n_exported = _stream_collection(_iter_features(), output_path)
    logger.info(f"  Exported {n_exported} cells, skipped {skipped[0]}")

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"  GeoJSON size: {file_size_mb:.1f} MB")

    return n_exported


def _stream_collection(features, output_path: str) -> int:
    """Write an iterable of features as a GeoJSON FeatureCollection, one at a time.

    Returns the number written. Nothing accumulates: the caller can hand this a generator and
    the whole document never exists in memory. Measured at 1.94x faster and -1004 MB peak RSS
    on the pipeline's largest artifact, vs. building the full feature list and a single
    `json.dump` (pre-dates this branch; not part of the 2026-08-26 measurement pass in
    `docs/perf/2026-08-26-rss.md`).

    The literals below reproduce `json.dump`'s default formatting exactly -- `", "` between
    array items, `": "` after a key -- so the file is BYTE-IDENTICAL to the single-dump version
    it replaces, including the empty collection. QuPath would not care about the whitespace, but
    byte-identity means this change cannot be the cause of any downstream difference.
    Guarded by tests/test_geojson_streaming_write.py.
    """
    n = 0
    with open(output_path, "w") as f:
        f.write('{"type": "FeatureCollection", "features": [')
        for feature in features:
            if n:
                f.write(", ")
            f.write(json.dumps(feature))
            n += 1
        f.write("]}")
    return n


def _write_collection(features: List[Dict], output_path: str) -> None:
    """Write a list of features as a GeoJSON FeatureCollection."""
    _stream_collection(iter(features), output_path)


def export_combined_geojson(
    df: pd.DataFrame,
    output_dir: str,
    pixel_size: float,
    marker_cols: List[str],
    cell_contours: Optional[Dict[str, List[List[float]]]],
    nucleus_contours: Optional[Dict[str, List[List[float]]]],
    prefix: str = "cells",
) -> Dict[str, int]:
    """Export one combined GeoJSON for per-compartment quantification.

    Writes a single ``<prefix>.geojson`` of QuPath-native **cell** objects: the
    whole-cell polygon as ``geometry`` and the nucleus polygon as top-level
    ``nucleusGeometry``, carrying all per-compartment measurements. This is the
    file FlowPath / qUMAP / annomask consume; QuPath toggles the drawn outline
    (nucleus vs cell) natively, so no separate nuclei/whole-cell files are needed.

    ``nucleus_contours`` must be keyed by **cell label** (re-keyed upstream by
    EXTRACT_NUCLEI_PROPERTIES via ``--reference_mask``), so lookup is a plain
    identity on the cell label.
    """
    color_int = rgb_to_qupath_color(*CELL_COLOR_RGB)
    qc_cols = [c for c in df.columns if is_qc_column(c)]

    cells_combined: List[Dict] = []
    n_with_nucleus = 0
    # This counter used to be incremented as `skipped[0] += 1` against a plain `int`
    # (`skipped = 0`), which raised `TypeError: 'int' object is not subscriptable` on
    # the first row with a NaN x/y centroid -- an expected input this function is meant
    # to handle (see compute_zscores' NaN-preservation comment above), not an edge case.
    skipped = 0

    for idx, x_px, y_px, row in _iter_rows_positional(df, marker_cols, qc_cols):
        if pd.isna(x_px) or pd.isna(y_px):
            skipped += 1
            continue
        # Corner-of-pixel, like the cell and nucleus contours below
        # (bin/utils/pixel_convention.py).
        x_corner = centre_to_corner(float(x_px))
        y_corner = centre_to_corner(float(y_px))

        cell_id = row.get("label", idx)
        label_str = str(int(cell_id)) if pd.notna(cell_id) else str(idx)

        cell_geom = _polygon_geometry(cell_contours, label_str) or {
            "type": "Point",
            "coordinates": [x_corner, y_corner],
        }
        nucleus_geom = _polygon_geometry(nucleus_contours, label_str)
        if nucleus_geom is not None:
            n_with_nucleus += 1

        measurements = build_measurements(row, marker_cols, pixel_size, qc_cols)

        cells_combined.append(
            build_feature(
                measurements,
                cell_geom,
                color_int,
                object_type="cell",
                nucleus_geometry=nucleus_geom,
                object_id=None,
            )
        )

    out = Path(output_dir)
    combined_path = str(out / f"{prefix}.geojson")
    _write_collection(cells_combined, combined_path)

    # Whole-cell-only companion: same cells and the SAME measurements, but WITHOUT
    # the nucleusGeometry polygon. Dropping the second polygon roughly halves the
    # geometry, so it imports much faster in QuPath; compartment gating still works
    # in FlowPath (which reads the measurement keys, not the nucleus polygon). Import
    # whichever you need — <prefix>.geojson for the nucleus outline / cell-vs-nucleus
    # toggle, <prefix>_wholecell.geojson for a fast, lean import.
    wholecell = [
        {k: v for k, v in feat.items() if k != "nucleusGeometry"}
        for feat in cells_combined
    ]
    wholecell_path = str(out / f"{prefix}_wholecell.geojson")
    _write_collection(wholecell, wholecell_path)

    counts = {prefix: len(cells_combined), f"{prefix}_wholecell": len(wholecell)}
    logger.info(
        f"  Combined export: {len(cells_combined)} cell objects "
        f"({n_with_nucleus} with nucleus), skipped {skipped}"
    )
    logger.info(
        f"  Whole-cell companion: {len(wholecell)} cells without nucleusGeometry "
        f"-> {Path(wholecell_path).name}"
    )
    return counts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export cell data to QuPath-compatible GeoJSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cell_data",
        required=True,
        help="Path to merged quantification CSV (from MERGE_QUANT_CSVS)",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--contours_json",
        type=str,
        default=None,
        help="Path to pre-computed whole-cell contours JSON (from extract_cell_properties.py)",
    )
    parser.add_argument(
        "--nucleus_contours_json",
        type=str,
        default=None,
        help="Path to nucleus contours JSON re-keyed to cell labels (from "
        "EXTRACT_NUCLEI_PROPERTIES). When given, each cell in the single combined "
        "cells.geojson gets a top-level nucleusGeometry field for per-compartment quantification.",
    )
    parser.add_argument(
        "--pixel_size",
        # str, not float: the run-level value may be the literal 'auto'. Unlike every
        # other consumer, EXPORT_GEOJSON stages no image -- it works from a
        # quantification CSV and contour JSON -- so there is nothing here to read a
        # scale off, and 'auto' is refused rather than silently substituted.
        type=str,
        default=None,
        help="Pixel size in micrometres (a number; 'auto' is not supported here)",
    )
    parser.add_argument(
        "--output_prefix",
        default="cells",
        help="Prefix for output files",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    configure_logging(level=logging.INFO)
    args = parse_args()

    # No image is staged for this process, so there is nothing to read a scale off:
    # resolve_pixel_size raises on 'auto' here rather than substituting a guess. A
    # number is required. Until this call existed, the argparse default silently made
    # every µm measurement below 0.325 regardless of what the run configured.
    args.pixel_size = resolve_pixel_size(args.pixel_size, None, source="EXPORT_GEOJSON")

    ensure_dir(args.output_dir)

    logger.info("=" * 80)
    logger.info("Cell GeoJSON Export (QuPath-native format)")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading cell data: {args.cell_data}")
    cell_df = pd.read_csv(args.cell_data)

    # Remove duplicates by label if present
    if "label" in cell_df.columns:
        cell_df = cell_df.drop_duplicates(subset="label", keep="first")

    logger.info(f"Loaded {len(cell_df)} cells")

    # Identify marker columns
    marker_cols = identify_marker_columns(cell_df)
    logger.info(f"Detected {len(marker_cols)} marker channels: {marker_cols}")

    # Load whole-cell contours
    contours = None
    if args.contours_json:
        logger.info(f"Loading cell contours: {args.contours_json}")
        with open(args.contours_json, "r") as f:
            contours = json.load(f)
        logger.info(f"Loaded contours for {len(contours)} cells")

    # Load nucleus contours (re-keyed to cell labels) for the combined per-compartment export
    nucleus_contours = None
    if args.nucleus_contours_json:
        logger.info(f"Loading nucleus contours: {args.nucleus_contours_json}")
        with open(args.nucleus_contours_json, "r") as f:
            nucleus_contours = json.load(f)
        logger.info(f"Loaded nucleus contours for {len(nucleus_contours)} cells")

    # Compute z-scores for CSV output
    cell_df_with_z = compute_zscores(cell_df, marker_cols)

    output_geojson = str(Path(args.output_dir) / f"{args.output_prefix}.geojson")
    if nucleus_contours is not None:
        # Per-compartment quantification: one combined cell+nucleus GeoJSON.
        counts = export_combined_geojson(
            df=cell_df,
            output_dir=args.output_dir,
            pixel_size=args.pixel_size,
            marker_cols=marker_cols,
            cell_contours=contours,
            nucleus_contours=nucleus_contours,
            prefix=args.output_prefix,
        )
        num_exported = counts[args.output_prefix]
    else:
        # Whole-cell-only export (legacy behaviour).
        num_exported = export_geojson(
            df=cell_df,
            output_path=output_geojson,
            pixel_size=args.pixel_size,
            marker_cols=marker_cols,
            contours=contours,
        )

    # Save CSV with both raw and z-score columns
    output_csv = str(Path(args.output_dir) / f"{args.output_prefix}_data.csv")
    logger.info(f"Saving CSV: {output_csv}")
    # No defensive copy: to_csv only reads, and cell_df_with_z is the widest frame in the
    # pipeline (every marker, raw AND z-scored, one row per cell), so copying it doubled the
    # peak at the very end of the run for nothing.
    cell_df_with_z.to_csv(output_csv, index=False)

    # Summary
    logger.info("=" * 80)
    logger.info("EXPORT COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("Output files:")
    logger.info(f"  GeoJSON: {output_geojson} ({num_exported} cells)")
    logger.info(
        f"  CSV:     {output_csv} ({len(cell_df_with_z)} cells, "
        f"{len(cell_df_with_z.columns)} columns)"
    )
    logger.info("")
    logger.info("To view in QuPath:")
    logger.info("  1. Open your pyramidal OME-TIFF image")
    logger.info("  2. Run import_geojson.groovy or use File > Import objects")
    logger.info("  3. Open FlowPath for interactive gating")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
