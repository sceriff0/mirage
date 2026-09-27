#!/usr/bin/env python3
"""Generate postprocessing quality control visualizations.

Creates QC images for visual inspection of segmentation and quantification
results after the postprocessing pipeline step.

Outputs
-------
For a given prefix, generates:
- {prefix}_seg_overlay.png: Cell boundaries from the segmentation mask
- {prefix}_cell_stats.png: Histograms of cell area, eccentricity, and cell count
- {prefix}_intensity_distributions.png: Per-marker intensity histograms

Examples
--------
Generate postprocessing QC:

    $ python generate_postprocessing_qc.py \\
        --mask seg_mask.npy \\
        --csv merged_quant.csv \\
        --output qc/ \\
        --prefix patient1
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries

# Add utils directory to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from logger import configure_logging, get_logger
from measurements import MORPHOLOGY_COLS, is_qc_column

__all__ = ["main", "generate_postprocessing_qc"]

# Morphology columns that should not appear in intensity distribution plots
# (single source of truth: bin/utils/measurements.py). Wrapped in a set here
# since this module does membership tests in a loop.
MORPHOLOGY_COLUMNS = set(MORPHOLOGY_COLS)


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a segmentation mask from .npy or .tif file.

    Parameters
    ----------
    mask_path : Path
        Path to the segmentation mask.

    Returns
    -------
    np.ndarray
        2D segmentation mask array.
    """
    if mask_path.suffix == ".npy":
        mask = np.load(str(mask_path)).squeeze()
    else:
        import tifffile

        mask = tifffile.imread(str(mask_path)).squeeze()
    return mask


def _generate_seg_overlay(
    mask: np.ndarray,
    output_path: Path,
    logger: Optional[logging.Logger] = None,
    n_cells_from_csv: Optional[int] = None,
) -> Path:
    """Generate a segmentation boundary overlay image.

    Parameters
    ----------
    mask : np.ndarray
        2D segmentation mask with cell labels (background=0).
    output_path : Path
        Output PNG path.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    if logger is None:
        logger = get_logger(__name__)

    logger.info(f"Generating segmentation overlay ({mask.shape[1]}x{mask.shape[0]})")

    boundaries = find_boundaries(mask, mode="thick")

    # Scale down for large images to keep output file size reasonable
    max_dim = 4096
    scale = 1.0
    if max(mask.shape) > max_dim:
        scale = max_dim / max(mask.shape)
        new_h = int(mask.shape[0] * scale)
        new_w = int(mask.shape[1] * scale)
        # Use proper resampling to avoid jagged boundary artifacts
        from skimage.transform import resize as skimage_resize

        boundaries = (
            skimage_resize(
                boundaries.astype(np.float32),
                (new_h, new_w),
                order=1,
                anti_aliasing=True,
                preserve_range=True,
            )
            > 0.5
        )  # Threshold back to binary after interpolation
        logger.info(
            f"  Downsampled to {boundaries.shape[1]}x{boundaries.shape[0]} (scale={scale:.3f})"
        )

    # Count cells from the CSV (post-filtering count), not from raw mask labels
    n_cells = (
        n_cells_from_csv if n_cells_from_csv is not None else len(np.unique(mask)) - 1
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(boundaries, cmap="gray", interpolation="nearest")
    ax.set_title(f"Cell Boundaries ({n_cells} cells)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  Saved: {output_path.name}")
    return output_path


def _generate_cell_stats(
    df: pd.DataFrame,
    output_path: Path,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Generate cell morphology statistics figure.

    Creates a figure with histograms of cell area and eccentricity,
    plus a bar chart of total cell count.

    Parameters
    ----------
    df : pd.DataFrame
        Merged quantification CSV with morphology columns.
    output_path : Path
        Output PNG path.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    if logger is None:
        logger = get_logger(__name__)

    logger.info(f"Generating cell statistics ({len(df)} cells)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Cell area histogram
    if "area" in df.columns:
        areas = df["area"].dropna()
        axes[0].hist(
            areas, bins=50, color="steelblue", edgecolor="black", linewidth=0.5
        )
        axes[0].set_xlabel("Cell Area (pixels)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Cell Area Distribution (n={len(areas)})")
        median_area = areas.median()
        axes[0].axvline(
            median_area,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Median={median_area:.0f}",
        )
        axes[0].legend()
    else:
        axes[0].text(
            0.5,
            0.5,
            "No area column",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_title("Cell Area Distribution")

    # Panel 2: Eccentricity histogram
    if "eccentricity" in df.columns:
        ecc = df["eccentricity"].dropna()
        axes[1].hist(ecc, bins=50, color="coral", edgecolor="black", linewidth=0.5)
        axes[1].set_xlabel("Eccentricity")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Eccentricity Distribution (n={len(ecc)})")
        axes[1].set_xlim(0, 1)
        median_ecc = ecc.median()
        axes[1].axvline(
            median_ecc,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Median={median_ecc:.2f}",
        )
        axes[1].legend()
    else:
        axes[1].text(
            0.5,
            0.5,
            "No eccentricity column",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("Eccentricity Distribution")

    # Panel 3: Cell count bar chart
    cell_count = len(df)
    axes[2].bar(
        ["Total Cells"],
        [cell_count],
        color="seagreen",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[2].set_ylabel("Count")
    axes[2].set_title("Cell Count")
    axes[2].text(
        0,
        cell_count,
        str(cell_count),
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=14,
    )

    fig.suptitle("Cell Morphology Statistics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  Saved: {output_path.name}")
    return output_path


def _intensity_marker_columns(df: pd.DataFrame) -> List[str]:
    """Numeric marker columns of ``df``, excluding morphology and QC columns.

    A "QC: ..." column (``measurements.is_qc_column``, e.g. ``QC: Total
    intensity``) is not a marker intensity and must not be plotted as one
    alongside DAPI/PANCK/etc.
    """
    return [
        col
        for col in df.columns
        if col not in MORPHOLOGY_COLUMNS
        and not is_qc_column(col)
        and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]


def _generate_intensity_distributions(
    df: pd.DataFrame,
    output_path: Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Generate per-marker intensity distribution histograms.

    Parameters
    ----------
    df : pd.DataFrame
        Merged quantification CSV containing marker intensity columns.
    output_path : Path
        Output PNG path.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    Path or None
        Path to the saved PNG, or None if no marker columns found.
    """
    if logger is None:
        logger = get_logger(__name__)

    # Identify marker columns (everything that is not morphology/metadata/QC)
    marker_cols = _intensity_marker_columns(df)

    if not marker_cols:
        logger.warning("No marker columns found for intensity distributions")
        return None

    logger.info(f"Generating intensity distributions for {len(marker_cols)} markers")

    # Compute grid layout
    n_markers = len(marker_cols)
    n_cols = min(4, n_markers)
    n_rows = math.ceil(n_markers / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    # Ensure axes is always a flat array for uniform indexing
    if n_markers == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    for i, marker in enumerate(marker_cols):
        ax = axes[i]
        values = df[marker].dropna()
        ax.hist(
            values,
            bins=50,
            color="steelblue",
            edgecolor="black",
            linewidth=0.3,
            alpha=0.8,
        )
        ax.set_title(marker, fontsize=10)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")

        # Add median line
        median_val = values.median()
        ax.axvline(median_val, color="red", linestyle="--", linewidth=0.8, alpha=0.7)

    # Hide unused subplot axes
    for j in range(n_markers, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Marker Intensity Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  Saved: {output_path.name}")
    return output_path


def generate_postprocessing_qc(
    mask_path: Path,
    csv_path: Path,
    output_dir: Path,
    prefix: str,
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    """Generate all postprocessing QC visualizations.

    Parameters
    ----------
    mask_path : Path
        Path to segmentation mask (.npy or .tif).
    csv_path : Path
        Path to merged quantification CSV.
    output_dir : Path
        Output directory for PNG files.
    prefix : str
        Prefix for output filenames.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    List[Path]
        List of generated PNG file paths.
    """
    if logger is None:
        logger = get_logger(__name__)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    # Load segmentation mask
    logger.info(f"Loading mask: {mask_path}")
    mask = load_mask(mask_path)
    logger.info(
        f"Mask shape: {mask.shape}, dtype: {mask.dtype}, unique labels: {len(np.unique(mask)) - 1}"
    )

    # Load merged CSV
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"CSV shape: {df.shape[0]} cells, {df.shape[1]} columns")

    # 1. Segmentation overlay
    seg_path = output_dir / f"{prefix}_seg_overlay.png"
    _generate_seg_overlay(mask, seg_path, logger, n_cells_from_csv=len(df))
    output_files.append(seg_path)

    # 2. Cell statistics
    stats_path = output_dir / f"{prefix}_cell_stats.png"
    _generate_cell_stats(df, stats_path, logger)
    output_files.append(stats_path)

    # 3. Intensity distributions
    intensity_path = output_dir / f"{prefix}_intensity_distributions.png"
    result = _generate_intensity_distributions(df, intensity_path, logger)
    if result is not None:
        output_files.append(result)

    return output_files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate postprocessing QC visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate postprocessing QC images
  %(prog)s \\
    --mask seg_mask.npy \\
    --csv merged_quant.csv \\
    --output qc/ \\
    --prefix patient1

Output:
  For each sample, creates:
    - {prefix}_seg_overlay.png           (cell boundary visualization)
    - {prefix}_cell_stats.png            (area, eccentricity, cell count)
    - {prefix}_intensity_distributions.png (per-marker histograms)
        """,
    )

    parser.add_argument(
        "--mask",
        type=Path,
        required=True,
        help="Path to segmentation mask (.npy or .tif)",
    )

    parser.add_argument(
        "--csv", type=Path, required=True, help="Path to merged quantification CSV"
    )

    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory for PNG files"
    )

    parser.add_argument(
        "--prefix", type=str, required=True, help="Prefix for output filenames"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose (DEBUG) logging"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for postprocessing QC generation.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    configure_logging(
        level=log_level, format_string="[%(asctime)s] %(levelname)s - %(message)s"
    )

    logger = get_logger(__name__)

    logger.info("=" * 80)
    logger.info("POSTPROCESSING QC GENERATION")
    logger.info("=" * 80)

    # Validate inputs
    if not args.mask.exists():
        logger.error(f"Mask not found: {args.mask}")
        return 1

    if not args.csv.exists():
        logger.error(f"CSV not found: {args.csv}")
        return 1

    logger.info(f"Mask: {args.mask}")
    logger.info(f"CSV: {args.csv}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Prefix: {args.prefix}")
    logger.info("")

    try:
        output_files = generate_postprocessing_qc(
            mask_path=args.mask,
            csv_path=args.csv,
            output_dir=args.output,
            prefix=args.prefix,
            logger=logger,
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("QC GENERATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Generated {len(output_files)} PNG files")
        for f in output_files:
            logger.info(f"  - {f.name}")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"Failed to generate QC: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
