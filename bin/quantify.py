#!/usr/bin/env python3
"""Cell quantification using CPU or GPU.

This module provides cell quantification by computing marker intensities
and morphological properties from segmentation masks. Supports both CPU
(via scikit-image) and GPU (via quantify_gpu module) processing.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from numpy.typing import NDArray
from joblib import Parallel, delayed

from _common import (
    ensure_dir,
    load_image,
)

logger = logging.getLogger(__name__)


__all__ = [
    "compute_morphology",
    "compute_channel_intensity",
    "quantify_multichannel",
    "run_quantification",
]


def compute_morphology(
    mask: NDArray,
    min_area: int = 0
) -> Tuple[Optional[pd.DataFrame], Optional[NDArray], Optional[NDArray]]:
    """Compute morphological properties once for all cells.

    Parameters
    ----------
    mask : ndarray, shape (Y, X)
        Segmentation mask with cell labels.
    min_area : int, optional
        Minimum cell area in pixels.

    Returns
    -------
    props_df : DataFrame or None
        Morphological properties indexed by label.
    mask_filtered : ndarray or None
        Mask with only valid labels.
    valid_labels : ndarray or None
        Array of valid label IDs.
    """
    mask = np.ascontiguousarray(mask.squeeze())
    
    # Filter cells by area
    labels, counts = np.unique(mask, return_counts=True)
    valid_labels = labels[(labels != 0) & (counts > min_area)]

    if len(valid_labels) == 0:
        logger.warning("No valid cells found after area filtering")
        return None, None, None

    # Create filtered mask once
    mask_filtered = np.where(np.isin(mask, valid_labels), mask, 0)

    if np.all(mask_filtered == 0):
        logger.warning("Filtered mask is empty")
        return None, None, None

    # Compute morphological properties ONCE
    logger.info("Computing morphological properties...")
    props = regionprops_table(
        mask_filtered,
        properties=[
            'label', 'centroid', 'area',
            'eccentricity', 'perimeter',
            'convex_area', 'axis_major_length', 'axis_minor_length'
        ]
    )
    props_df = pd.DataFrame(props).set_index('label')
    props_df.rename(
        columns={'centroid-0': 'y', 'centroid-1': 'x'},
        inplace=True
    )

    logger.info(f"Found {len(props_df)} valid cells")
    return props_df, mask_filtered, valid_labels


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
        Pre-filtered segmentation mask.
    channel : ndarray, shape (Y, X)
        Channel image.
    valid_labels : ndarray
        Array of valid label IDs.
    channel_name : str
        Name of the channel.

    Returns
    -------
    Series
        Mean intensities indexed by cell label.
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

    # Vectorized extraction - NO Python loop
    # Ensure valid_labels doesn't exceed array bounds
    max_label = len(mean_intensities) - 1
    safe_labels = np.clip(valid_labels, 0, max_label)
    intensities = mean_intensities[safe_labels]
    
    # Handle any labels that were out of bounds
    out_of_bounds = valid_labels > max_label
    if np.any(out_of_bounds):
        intensities[out_of_bounds] = 0.0

    return pd.Series(intensities, index=valid_labels, name=channel_name)


def _load_and_compute_intensity(
    channel_path: str,
    mask_filtered: NDArray,
    valid_labels: NDArray
) -> Tuple[str, pd.Series]:
    """Helper for parallel channel processing."""
    channel_name = Path(channel_path).stem.split('_')[-1]
    channel_image, _ = load_image(channel_path)
    intensity = compute_channel_intensity(
        mask_filtered, channel_image, valid_labels, channel_name
    )
    return channel_name, intensity


def quantify_multichannel(
    mask: NDArray,
    channel_paths: List[str],
    min_area: int = 0,
    n_jobs: int = -1
) -> pd.DataFrame:
    """Quantify all channels for all cells.

    Parameters
    ----------
    mask : ndarray, shape (Y, X)
        Segmentation mask.
    channel_paths : list of str
        Paths to channel image files.
    min_area : int, optional
        Minimum cell area filter.
    n_jobs : int, optional
        Number of parallel jobs (-1 for all cores).

    Returns
    -------
    DataFrame
        Combined measurements for all channels.
    """
    logger.info(f"Quantifying {len(channel_paths)} channels")

    # Step 1: Compute morphology ONCE
    props_df, mask_filtered, valid_labels = compute_morphology(mask, min_area)

    if props_df is None:
        return pd.DataFrame()

    # Step 2: Process channels in parallel
    logger.info(f"Processing channels with {n_jobs} workers...")
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_load_and_compute_intensity)(
            path, mask_filtered, valid_labels
        )
        for path in channel_paths
    )

    # Step 3: Combine results efficiently
    result_df = props_df.copy()
    for channel_name, intensity_series in results:
        result_df[channel_name] = intensity_series

    logger.info(f"Total cells: {len(result_df)}")
    return result_df


def run_quantification(
    mask_path: str,
    channel_paths: List[str],
    output_path: str,
    min_area: int = 0,
    n_jobs: int = -1
) -> pd.DataFrame:
    """Run complete quantification pipeline.

    Parameters
    ----------
    mask_path : str
        Path to segmentation mask (.npy file).
    channel_paths : list of str
        Paths to channel image files.
    output_path : str
        Path to save output CSV.
    min_area : int, optional
        Minimum cell area filter.
    n_jobs : int, optional
        Number of parallel jobs for channel processing.

    Returns
    -------
    DataFrame
        Quantification results.
    """
    logger.info("=" * 80)
    logger.info("Starting Quantification Pipeline (CPU - Optimized)")
    logger.info("=" * 80)

    # Load segmentation mask
    logger.info(f"Loading segmentation mask: {mask_path}")
    mask = np.load(mask_path)
    logger.info(f"Mask shape: {mask.shape}")
    logger.info(f"Unique labels: {len(np.unique(mask)) - 1}")

    # Quantify
    result_df = quantify_multichannel(
        mask, channel_paths, min_area=min_area, n_jobs=n_jobs
    )

    # Save
    if not result_df.empty:
        logger.info(f"Saving results: {output_path}")
        result_df.to_csv(output_path, index=True)
        logger.info(f"Rows: {len(result_df)}, Columns: {len(result_df.columns)}")
    else:
        logger.warning("No results to save")

    logger.info("=" * 80)
    logger.info("Quantification Complete")
    logger.info("=" * 80)

    return result_df


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Cell quantification (CPU/GPU modes)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--mode',
        choices=['cpu', 'gpu'],
        default='cpu',
        help='Processing mode'
    )
    parser.add_argument(
        '--mask_file',
        type=str,
        required=True,
        help='Path to segmentation mask (.npy)'
    )
    parser.add_argument(
        '--indir',
        type=str,
        required=True,
        help='Directory containing channel images'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Exact output file path for CSV'
    )
    parser.add_argument(
        '--min_area',
        type=int,
        default=0,
        help='Minimum cell area (pixels)'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all cores)'
    )

    return parser.parse_args()


def main():
    """Main entry point dispatching to CPU or GPU processing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    ensure_dir(args.outdir)

    # Get channel files
    channel_files = sorted([
        os.path.join(args.indir, f)
        for f in os.listdir(args.indir)
        if f.lower().endswith(('.tif', '.tiff', '.ome.tif', '.ome.tiff'))
    ])

    if not channel_files:
        raise ValueError(f"No channel files found in {args.indir}")

    logger.info(f"Found {len(channel_files)} channel files")

    # Resolve output path
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(
            args.outdir, 
            f"{Path(args.mask_file).stem}_quantification.csv"
        )

    # Run quantification
    if args.mode == 'cpu':
        logger.info('Running CPU quantification (optimized)')
        run_quantification(
            mask_path=args.mask_file,
            channel_paths=channel_files,
            output_path=output_path,
            min_area=args.min_area,
            n_jobs=args.n_jobs
        )

    else:  # gpu mode
        logger.info('Running GPU quantification')
        try:
            from bin import quantify_gpu
        except ImportError:
            logger.warning('GPU module unavailable; falling back to CPU')
            run_quantification(
                mask_path=args.mask_file,
                channel_paths=channel_files,
                output_path=output_path,
                min_area=args.min_area,
                n_jobs=args.n_jobs
            )
        else:
            quantify_gpu.run_marker_quantification(
                indir=args.indir,
                mask_file=args.mask_file,
                outdir=args.outdir,
                size_cutoff=args.min_area,
            )

    return 0


if __name__ == '__main__':
    exit(main())