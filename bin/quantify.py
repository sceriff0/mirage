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
from typing import List

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from numpy.typing import NDArray

from _common import (
    ensure_dir,
    load_image,
)

logger = logging.getLogger(__name__)


__all__ = [
    "compute_cell_intensities",
    "quantify_multichannel",
    "run_quantification",
]


def compute_cell_intensities(
    mask: NDArray,
    channel: NDArray,
    channel_name: str,
    min_area: int = 0
) -> pd.DataFrame:
    """Compute mean intensity per cell for a single channel.

    Parameters
    ----------
    mask : ndarray, shape (Y, X)
        Segmentation mask with cell labels.
    channel : ndarray, shape (Y, X)
        Channel image.
    channel_name : str
        Name of the channel.
    min_area : int, optional
        Minimum cell area in pixels.

    Returns
    -------
    DataFrame
        Cell measurements with columns: label, channel_name, area, x, y, etc.
    """
    logger.info(f"Computing intensities for {channel_name}")

    # Filter cells by area
    labels, counts = np.unique(mask, return_counts=True)
    valid_labels = labels[(labels != 0) & (counts > min_area)]

    if len(valid_labels) == 0:
        logger.warning(f"No valid cells found for {channel_name}")
        return pd.DataFrame()

    # Create filtered mask
    mask_filtered = np.where(np.isin(mask, valid_labels), mask, 0)

    if np.all(mask_filtered == 0):
        logger.warning(f"Filtered mask is empty for {channel_name}")
        return pd.DataFrame()

    # Compute morphological properties
    props = regionprops_table(
        mask_filtered,
        properties=[
            'label', 'centroid', 'area',
            'eccentricity', 'perimeter',
            'convex_area', 'axis_major_length', 'axis_minor_length'
        ]
    )
    props_df = pd.DataFrame(props).set_index('label', drop=False)

    # Compute mean intensities using bincount (efficient)
    flat_mask = mask_filtered.ravel()
    flat_channel = channel.ravel()

    sum_per_label = np.bincount(flat_mask, weights=flat_channel)
    count_per_label = np.bincount(flat_mask)

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_intensities = np.true_divide(sum_per_label, count_per_label)
        mean_intensities[np.isnan(mean_intensities)] = 0

    # Extract intensities for valid labels
    intensities = np.array([
        mean_intensities[label] if label < len(mean_intensities) else 0
        for label in valid_labels
    ])

    # Create intensity dataframe
    intensity_df = pd.DataFrame(
        {channel_name: intensities},
        index=valid_labels
    )

    # Join with morphological properties
    result_df = intensity_df.join(props_df)
    result_df.rename(
        columns={'centroid-0': 'y', 'centroid-1': 'x'},
        inplace=True
    )

    logger.info(f"Cells quantified: {len(result_df)}")
    return result_df


def quantify_multichannel(
    mask: NDArray,
    channel_paths: List[str],
    min_area: int = 0
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

    Returns
    -------
    DataFrame
        Combined measurements for all channels.
    """
    logger.info(f"Quantifying {len(channel_paths)} channels")

    all_channel_dfs = []

    for idx, channel_path in enumerate(channel_paths):
        channel_name = Path(channel_path).stem.split('_')[-1]
        logger.info(f"[{idx + 1}/{len(channel_paths)}] {channel_name}")

        # Load channel
        channel_image, _ = load_image(channel_path)

        # Compute intensities
        channel_df = compute_cell_intensities(
            mask,
            channel_image,
            channel_name,
            min_area=min_area
        )

        if not channel_df.empty:
            all_channel_dfs.append(channel_df)

    # Combine all channels
    if all_channel_dfs:
        result_df = pd.concat(all_channel_dfs, axis=1)
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        logger.info(f"Total cells: {len(result_df)}")
        return result_df
    else:
        logger.warning("No valid data found")
        return pd.DataFrame()


def run_quantification(
    mask_path: str,
    channel_paths: List[str],
    output_path: str,
    min_area: int = 0
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

    Returns
    -------
    DataFrame
        Quantification results.
    """
    logger.info("=" * 80)
    logger.info("Starting Quantification Pipeline (CPU)")
    logger.info("=" * 80)

    # Load segmentation mask
    logger.info(f"Loading segmentation mask: {mask_path}")
    mask = np.load(mask_path)
    logger.info(f"Mask shape: {mask.shape}")
    logger.info(f"Unique labels: {len(np.unique(mask)) - 1}")

    # Quantify
    result_df = quantify_multichannel(mask, channel_paths, min_area=min_area)

    # Save
    if not result_df.empty:
        logger.info(f"Saving results: {output_path}")
        result_df.to_csv(output_path, index=False)
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
    # Removed --patient_id: not needed for this script or pipeline.
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
    channel_files = [
        os.path.join(args.indir, f)
        for f in os.listdir(args.indir)
        if f.lower().endswith(('.tif', '.tiff', '.ome.tif', '.ome.tiff'))
    ]

    if not channel_files:
        raise ValueError(f"No channel files found in {args.indir}")

    logger.info(f"Found {len(channel_files)} channel files")

    # Resolve output path
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(args.outdir, f"{Path(args.mask_file).stem}_quantification.csv")

    # Run quantification
    if args.mode == 'cpu':
        logger.info('Running CPU quantification')
        run_quantification(
            mask_path=args.mask_file,
            channel_paths=channel_files,
            output_path=output_path,
            min_area=args.min_area
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
                min_area=args.min_area
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
