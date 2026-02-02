#!/usr/bin/env python3
"""Cell quantification for single-channel processing.

Designed for Nextflow pipelines where each channel is processed separately
and results are merged afterward. Supports both CPU and GPU processing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import gzip
import json

import numpy as np
import pandas as pd
from skimage.measure import regionprops, find_contours, approximate_polygon
from numpy.typing import NDArray

# Add parent directory to path to import lib modules
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

# Import from lib for DRY principle
from logger import get_logger, configure_logging
from image_utils import ensure_dir, load_image

logger = get_logger(__name__)


__all__ = [
    "compute_morphology",
    "compute_channel_intensity",
    "quantify_single_channel",
    "run_quantification",
]


def compute_morphology(
    mask: NDArray,
    min_area: int = 0,
    export_contours: bool = False,
    simplify_tolerance: float = 1.0
) -> Tuple[Optional[pd.DataFrame], Optional[NDArray], Optional[NDArray], Optional[Dict[int, List]]]:
    """Compute morphological properties and optionally extract contours.

    Parameters
    ----------
    mask : ndarray, shape (Y, X)
        Segmentation mask with cell labels (background=0, cells>=1).
    min_area : int, default=0
        Minimum cell area in pixels. Cells smaller than this are excluded.
    export_contours : bool, default=False
        If True, extract cell boundary contours for each cell.
    simplify_tolerance : float, default=1.0
        Douglas-Peucker simplification tolerance in pixels.
        Higher values = fewer points, smaller file. Set to 0 to disable.

    Returns
    -------
    props_df : DataFrame or None
        Morphological properties indexed by label with columns:
        - label: Cell label ID
        - y, x: Centroid coordinates
        - area: Cell area in pixels
        - eccentricity: Shape eccentricity (0=circle, 1=line)
        - perimeter: Cell perimeter length
        - convex_area: Area of convex hull
        - axis_major_length, axis_minor_length: Ellipse fit axes
        Returns None if no valid cells found.
    mask_filtered : ndarray or None
        Binary mask containing only cells meeting area threshold.
        Returns None if no valid cells found.
    valid_labels : ndarray or None
        Array of label IDs for cells that passed filtering.
        Returns None if no valid cells found.
    contours : dict or None
        Mapping of label -> list of (x, y) coordinates forming closed polygon.
        Only returned if export_contours=True, otherwise None.

    Notes
    -----
    This function filters cells by area and computes morphological features
    using scikit-image's regionprops. The filtering removes artifacts and
    debris that don't meet size criteria.

    When export_contours=True, contours are extracted using marching squares
    on each cell's bounding box region for efficiency.

    Examples
    --------
    >>> mask = np.array([[0, 1, 1], [0, 1, 1], [2, 2, 0]])
    >>> props_df, mask_filt, valid, contours = compute_morphology(mask, min_area=2)
    >>> print(props_df['area'].values)
    [4 2]

    See Also
    --------
    skimage.measure.regionprops : Underlying morphology computation
    compute_channel_intensity : Intensity measurements per cell
    """
    logger.debug(f"Computing morphology: mask shape={mask.shape}, min_area={min_area}")
    mask = np.ascontiguousarray(mask.squeeze())

    # Filter cells by area
    labels, counts = np.unique(mask, return_counts=True)
    valid_labels = labels[(labels != 0) & (counts > min_area)]

    if len(valid_labels) == 0:
        logger.warning("[WARN] No valid cells found after area filtering")
        return None, None, None, None

    # Create filtered mask
    mask_filtered = np.where(np.isin(mask, valid_labels), mask, 0)

    if np.all(mask_filtered == 0):
        logger.warning("[WARN] Filtered mask is empty")
        return None, None, None, None

    # Compute morphological properties using regionprops (single pass)
    logger.debug("Computing morphological properties...")
    regions = regionprops(mask_filtered)

    rows = []
    contours = {} if export_contours else None

    for region in regions:
        # Extract morphology
        rows.append({
            'label': region.label,
            'y': region.centroid[0],
            'x': region.centroid[1],
            'area': region.area,
            'eccentricity': region.eccentricity,
            'perimeter': region.perimeter,
            'convex_area': region.convex_area,
            'axis_major_length': region.axis_major_length,
            'axis_minor_length': region.axis_minor_length,
        })

        # Extract contour from bounding box (efficient - only processes local region)
        if export_contours:
            minr, minc, maxr, maxc = region.bbox
            local_mask = (mask_filtered[minr:maxr, minc:maxc] == region.label)
            local_contours = find_contours(local_mask.astype(float), 0.5)

            if local_contours:
                # Take the largest contour (handles any holes)
                contour = max(local_contours, key=len)

                # Simplify with Douglas-Peucker algorithm
                if simplify_tolerance > 0 and len(contour) > 10:
                    contour = approximate_polygon(contour, tolerance=simplify_tolerance)

                # Offset to global coords, convert (row, col) to (x, y), close polygon
                coords = [(round(pt[1] + minc, 1), round(pt[0] + minr, 1)) for pt in contour]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])

                contours[region.label] = coords

    props_df = pd.DataFrame(rows).set_index('label')

    logger.debug(f"Found {len(props_df)} valid cells")
    if export_contours:
        logger.debug(f"Extracted {len(contours)} contours")

    return props_df, mask_filtered, valid_labels, contours


def compute_channel_intensity(
    mask_filtered: NDArray,
    channel: NDArray,
    valid_labels: NDArray,
    channel_name: str
) -> pd.Series:
    """Compute mean intensity per cell for a single channel.

    Parameters
    ----------
    mask_filtered : ndarray, shape (Y, X)
        Pre-filtered segmentation mask with valid cell labels only.
    channel : ndarray, shape (Y, X)
        Channel intensity image (same spatial dimensions as mask).
    valid_labels : ndarray
        Array of valid cell label IDs to compute intensities for.
    channel_name : str
        Name of the channel/marker for the output Series.

    Returns
    -------
    pd.Series
        Mean intensities indexed by cell label, with name=channel_name.
        Missing or out-of-bounds labels are assigned intensity 0.0.

    Notes
    -----
    This function uses np.bincount for efficient vectorized computation
    of mean intensities. It handles edge cases like missing labels or
    labels outside the computed range.

    The mean intensity is computed as the sum of pixel intensities
    divided by the number of pixels for each cell.

    Examples
    --------
    >>> mask = np.array([[1, 1, 0], [1, 0, 2], [0, 2, 2]])
    >>> channel = np.array([[10, 20, 0], [30, 0, 40], [0, 50, 60]])
    >>> valid_labels = np.array([1, 2])
    >>> intensities = compute_channel_intensity(mask, channel, valid_labels, "CD3")
    >>> print(intensities)
    1    20.0
    2    50.0
    Name: CD3, dtype: float64

    See Also
    --------
    compute_morphology : Compute morphological properties
    quantify_single_channel : Complete quantification pipeline
    """
    channel = channel.squeeze()
    
    # Efficient bincount-based mean computation
    flat_mask = mask_filtered.ravel()
    flat_channel = channel.ravel().astype(np.float64)

    sum_per_label = np.bincount(flat_mask, weights=flat_channel)
    count_per_label = np.bincount(flat_mask)

    # Compute means with safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_intensities = np.divide(sum_per_label, count_per_label)
        mean_intensities = np.nan_to_num(mean_intensities, nan=0.0)

    # Vectorized extraction
    max_label = len(mean_intensities) - 1
    safe_labels = np.clip(valid_labels, 0, max_label)
    intensities = mean_intensities[safe_labels]
    
    # Handle any labels that were out of bounds
    out_of_bounds = valid_labels > max_label
    if np.any(out_of_bounds):
        intensities[out_of_bounds] = 0.0

    return pd.Series(intensities, index=valid_labels, name=channel_name)


def quantify_single_channel(
    mask: NDArray,
    channel_image: NDArray,
    channel_name: str,
    min_area: int = 0,
    export_contours: bool = False,
    simplify_tolerance: float = 1.0
) -> Tuple[pd.DataFrame, Optional[Dict[int, List]]]:
    """Quantify a single channel.

    Parameters
    ----------
    mask : ndarray, shape (Y, X)
        Segmentation mask.
    channel_image : ndarray, shape (Y, X)
        Channel intensity image.
    channel_name : str
        Name of the channel/marker.
    min_area : int, optional
        Minimum cell area filter.
    export_contours : bool, default=False
        If True, extract cell boundary contours.
    simplify_tolerance : float, default=1.0
        Douglas-Peucker simplification tolerance in pixels.

    Returns
    -------
    result_df : DataFrame
        Morphological properties + channel intensity for all cells.
    contours : dict or None
        Cell contours if export_contours=True, else None.
    """
    # Compute morphology (and optionally contours)
    props_df, mask_filtered, valid_labels, contours = compute_morphology(
        mask, min_area,
        export_contours=export_contours,
        simplify_tolerance=simplify_tolerance
    )

    if props_df is None:
        return pd.DataFrame(), None

    # Compute intensity for this channel
    intensity_series = compute_channel_intensity(
        mask_filtered, channel_image, valid_labels, channel_name
    )

    # Add intensity to morphology dataframe
    result_df = props_df.copy()
    result_df[channel_name] = intensity_series

    # Reset index to make 'label' a column
    result_df = result_df.reset_index()

    # Reorder: label first, then morphology, then intensity
    morpho_cols = ['label', 'y', 'x', 'area', 'eccentricity', 'perimeter',
                   'convex_area', 'axis_major_length', 'axis_minor_length']
    cols_order = morpho_cols + [channel_name]
    result_df = result_df[cols_order]

    return result_df, contours


def run_quantification(
    mask_path: str,
    channel_path: str,
    output_path: str,
    min_area: int = 0,
    channel_name: str = None,
    export_contours: bool = False,
    simplify_tolerance: float = 1.0
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Run quantification for a single channel.

    Parameters
    ----------
    mask_path : str
        Path to segmentation mask (.npy or .tif file).
    channel_path : str
        Path to channel image file (.tif).
    output_path : str
        Path to save output CSV file.
    min_area : int, default=0
        Minimum cell area filter in pixels. Cells smaller than this
        are excluded from quantification.
    channel_name : str, optional
        Explicit channel/marker name. If not provided, will be parsed
        from the channel file basename.
    export_contours : bool, default=False
        If True, extract and save cell boundary contours to a gzipped JSON file.
    simplify_tolerance : float, default=1.0
        Douglas-Peucker simplification tolerance in pixels for contour reduction.
        Higher values = fewer points, smaller file. Set to 0 to disable.

    Returns
    -------
    result_df : pd.DataFrame
        Quantification results with columns:
        - label: Cell ID
        - y, x: Centroid coordinates
        - area, eccentricity, perimeter, etc.: Morphology features
        - {channel_name}: Mean intensity for this channel
    contour_path : str or None
        Path to saved contours file if export_contours=True, else None.

    Raises
    ------
    FileNotFoundError
        If mask_path or channel_path does not exist.
    ValueError
        If mask and channel have incompatible shapes.

    Notes
    -----
    This function orchestrates the complete quantification pipeline:
    1. Load segmentation mask and channel image
    2. Validate shapes match
    3. Compute morphology and filter by area
    4. Compute mean intensities per cell
    5. Save results to CSV
    6. Optionally save contours to gzipped JSON

    If no valid cells are found, an empty CSV with proper column
    headers is created.

    Examples
    --------
    >>> run_quantification(
    ...     mask_path="seg/P001_mask.npy",
    ...     channel_path="channels/P001_CD3.tif",
    ...     output_path="quant/P001_CD3_quant.csv",
    ...     min_area=10,
    ...     channel_name="CD3",
    ...     export_contours=True
    ... )

    See Also
    --------
    compute_morphology : Morphology computation
    compute_channel_intensity : Intensity computation
    quantify_single_channel : Core quantification logic
    """
    # Use provided channel name, or extract from filename as fallback
    if channel_name is None:
        channel_name = Path(channel_path).stem.split('_')[-1]
        logger.info(f"Channel name parsed from filename: {channel_name}")
    
    logger.info("=" * 60)
    logger.info(f"Quantifying channel: {channel_name}")
    logger.info("=" * 60)

    # Load segmentation mask
    logger.info(f"Loading mask: {mask_path}")
    try:
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path).squeeze()
        else:
            mask, _ = load_image(mask_path)
            mask = mask.squeeze()
        logger.debug(f"  Mask shape: {mask.shape}")
    except FileNotFoundError:
        logger.error(f"[FAIL] Segmentation mask not found: {mask_path}")
        raise FileNotFoundError(f"Segmentation mask not found: {mask_path}")
    except Exception as e:
        logger.error(f"[FAIL] Failed to load mask from {mask_path}: {e}")
        raise ValueError(f"Failed to load mask from {mask_path}: {e}") from e

    # Load channel image
    logger.info(f"Loading channel: {channel_path}")
    try:
        channel_image, _ = load_image(channel_path)
        logger.debug(f"  Channel shape: {channel_image.shape}")
    except FileNotFoundError:
        logger.error(f"[FAIL] Channel image not found: {channel_path}")
        raise FileNotFoundError(f"Channel image not found: {channel_path}")
    except Exception as e:
        logger.error(f"[FAIL] Failed to load channel from {channel_path}: {e}")
        raise ValueError(f"Failed to load channel from {channel_path}: {e}") from e

    # Validate shapes match
    if mask.shape != channel_image.squeeze().shape:
        logger.error(
            f"[FAIL] Shape mismatch: mask {mask.shape} vs channel {channel_image.squeeze().shape}"
        )
        raise ValueError(
            f"Shape mismatch: mask {mask.shape} vs channel {channel_image.squeeze().shape}. "
            f"Ensure the mask and channel images have the same spatial dimensions."
        )

    # Quantify
    result_df, contours = quantify_single_channel(
        mask, channel_image, channel_name,
        min_area=min_area,
        export_contours=export_contours,
        simplify_tolerance=simplify_tolerance
    )

    # Save CSV
    contour_path = None
    if not result_df.empty:
        result_df.to_csv(output_path, index=False)
        logger.info(f"[OK] Saved {len(result_df)} cells to {output_path}")

        # Save contours if extracted
        if contours:
            contour_path = output_path.replace('_quant.csv', '_contours.json.gz')
            if contour_path == output_path:
                # Fallback if filename doesn't match expected pattern
                contour_path = output_path.rsplit('.', 1)[0] + '_contours.json.gz'
            with gzip.open(contour_path, 'wt') as f:
                json.dump({str(k): v for k, v in contours.items()}, f)
            logger.info(f"[OK] Saved {len(contours)} contours to {contour_path}")
    else:
        logger.warning("[WARN] No results to save")
        # Create empty CSV with expected columns
        empty_df = pd.DataFrame(columns=[
            'label', 'y', 'x', 'area', 'eccentricity', 'perimeter',
            'convex_area', 'axis_major_length', 'axis_minor_length', channel_name
        ])
        empty_df.to_csv(output_path, index=False)

    logger.info("Quantification complete")

    return result_df, contour_path


def run_quantification_gpu(
    mask_path: str,
    channel_path: str,
    output_path: str,
    min_area: int = 0,
    channel_name: str = None
) -> pd.DataFrame:
    """Run GPU-accelerated quantification for a single channel.

    Parameters
    ----------
    mask_path : str
        Path to segmentation mask (.npy file).
    channel_path : str
        Path to channel image file (.tif).
    output_path : str
        Path to save output CSV.
    min_area : int, optional
        Minimum cell area filter.
    channel_name : str, optional
        Explicit channel name. If not provided, will parse from filename.

    Returns
    -------
    DataFrame
        Quantification results.
    """
    import cupy as cp
    from cucim.skimage.measure import regionprops_table as gpu_regionprops_table

    # Use provided channel name, or extract from filename as fallback
    if channel_name is None:
        channel_name = Path(channel_path).stem.split('_')[-1]
        logger.info(f"Channel name parsed from filename: {channel_name}")
    
    logger.info("=" * 60)
    logger.info(f"Quantifying channel (GPU): {channel_name}")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading mask: {mask_path}")
    if mask_path.endswith('.npy'):
        segmentation_mask = np.load(mask_path).squeeze()
    else:
        segmentation_mask, _ = load_image(mask_path)
        segmentation_mask = segmentation_mask.squeeze()
    
    logger.info(f"Loading channel: {channel_path}")
    channel_image, _ = load_image(channel_path)
    channel_image = channel_image.squeeze()

    # Validate shapes
    if segmentation_mask.shape != channel_image.shape:
        raise ValueError(
            f"Shape mismatch: mask {segmentation_mask.shape} vs channel {channel_image.shape}"
        )

    # Transfer to GPU
    logger.info("Transferring to GPU...")
    mask_gpu = cp.asarray(segmentation_mask)
    channel_gpu = cp.asarray(channel_image)

    # Filter by area
    labels, counts = cp.unique(mask_gpu, return_counts=True)
    valid_labels = labels[(labels != 0) & (counts > min_area)]

    if len(valid_labels) == 0:
        logger.warning("No valid cells found")
        empty_df = pd.DataFrame(columns=[
            'label', 'y', 'x', 'area', 'eccentricity', 'perimeter',
            'convex_area', 'axis_major_length', 'axis_minor_length', channel_name
        ])
        empty_df.to_csv(output_path, index=False)
        return empty_df

    # Filter mask
    mask_filtered = cp.where(cp.isin(mask_gpu, valid_labels), mask_gpu, 0)

    # Compute morphology on GPU
    logger.info("Computing morphology (GPU)...")
    props = gpu_regionprops_table(
        mask_filtered,
        properties=[
            'label', 'centroid', 'area', 'eccentricity', 'perimeter',
            'convex_area', 'axis_major_length', 'axis_minor_length'
        ]
    )
    props_cpu = {k: (v.get() if isinstance(v, cp.ndarray) else v) for k, v in props.items()}
    props_df = pd.DataFrame(props_cpu)
    props_df.rename(columns={'centroid-0': 'y', 'centroid-1': 'x'}, inplace=True)

    # Compute intensity
    logger.info("Computing intensity (GPU)...")
    flat_mask = mask_filtered.ravel()
    flat_channel = channel_gpu.ravel().astype(cp.float64)

    sum_per_label = cp.bincount(flat_mask, weights=flat_channel)
    count_per_label = cp.bincount(flat_mask)

    means = cp.where(count_per_label != 0, sum_per_label / count_per_label, 0.0)
    intensities = means[valid_labels]

    # Create result dataframe
    props_df[channel_name] = intensities.get()

    # Reorder columns
    morpho_cols = ['label', 'y', 'x', 'area', 'eccentricity', 'perimeter',
                   'convex_area', 'axis_major_length', 'axis_minor_length']
    cols_order = morpho_cols + [channel_name]
    result_df = props_df[cols_order]

    # Save
    logger.info(f"Saving: {output_path}")
    result_df.to_csv(output_path, index=False)
    logger.info(f"Cells: {len(result_df)}")

    # Cleanup GPU memory
    del mask_gpu, channel_gpu, mask_filtered, flat_mask, flat_channel
    cp.get_default_memory_pool().free_all_blocks()

    return result_df


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Single-channel cell quantification (Nextflow compatible)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--channel_tiff',
        type=str,
        required=True,
        help='Path to single channel TIFF image'
    )
    parser.add_argument(
        '--mask_file',
        type=str,
        required=True,
        help='Path to segmentation mask (.npy)'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='.',
        help='Output directory'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Output CSV filename (default: {channel}_quant.csv)'
    )
    parser.add_argument(
        '--min_area',
        type=int,
        default=0,
        help='Minimum cell area (pixels)'
    )
    parser.add_argument(
        '--channel-name',
        type=str,
        default=None,
        help='Explicit channel name (if not provided, will parse from filename)'
    )
    parser.add_argument(
        '--export-contours',
        action='store_true',
        default=False,
        help='Extract and save cell boundary contours to gzipped JSON'
    )
    parser.add_argument(
        '--simplify-tolerance',
        type=float,
        default=1.0,
        help='Douglas-Peucker simplification tolerance in pixels (0 to disable)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    configure_logging(level=logging.INFO)

    args = parse_args()

    # Ensure output directory exists
    ensure_dir(args.outdir)

    # Determine output path
    if args.output_file:
        output_path = os.path.join(args.outdir, args.output_file)
    else:
        channel_name = Path(args.channel_tiff).stem
        output_path = os.path.join(args.outdir, f"{channel_name}_quant.csv")

    # Run quantification (CPU mode)
    # Note: GPU mode removed - use GPU container if GPU acceleration needed
    logger.info('Running CPU quantification')
    if args.export_contours:
        logger.info(f'Contour export enabled (tolerance={args.simplify_tolerance})')

    run_quantification(
        mask_path=args.mask_file,
        channel_path=args.channel_tiff,
        output_path=output_path,
        min_area=args.min_area,
        channel_name=args.channel_name,
        export_contours=args.export_contours,
        simplify_tolerance=args.simplify_tolerance
    )

    return 0


if __name__ == '__main__':
    exit(main())