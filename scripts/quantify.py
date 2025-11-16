#!/usr/bin/env python3
from __future__ import annotations
"""Cell quantification utilities.

Compute per-cell marker intensities from a segmentation mask and single-
channel images. The module provides a CPU implementation; an optional GPU
implementation lives in ``scripts/quantify_gpu.py`` and is invoked when
``--mode gpu`` is passed and the GPU helper module is available.

Functions follow NumPy docstring conventions and are lightweight and
functional for easy testing.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table
from numpy.typing import NDArray

try:
    from utils import logging_config
except Exception:
    # Minimal fallback so the module can be imported in test/dev environments
    import logging as _logging

    class _FallbackLoggingConfig:
        @staticmethod
        def setup_logging():
            _logging.basicConfig(level=_logging.INFO)

    logging_config = _FallbackLoggingConfig()
from scripts._common import ensure_dir, setup_file_logger

# Setup logging.
logging_config.setup_logging()
logger = logging.getLogger(__name__)

# Keep this module focused on CPU helpers.
# GPU helpers are implemented in scripts/quantify_gpu.py and imported dynamically when requested.

__all__ = [
    "load_channel_image",
    "compute_cell_intensities",
    "quantify_multichannel",
    "run_quantification",
]


def load_channel_image(
    channel_path: str
) -> Tuple[NDArray, dict]:
    """Load a single-channel image and return best-effort metadata.

    This function intentionally uses ``tifffile`` to keep the codebase
    lightweight and avoid extra imaging backends. The returned ``metadata``
    dict may be empty if no OME metadata is available.

    Parameters
    ----------
    channel_path : str
        Path to channel image file (TIFF/OME-TIFF recommended).

    Returns
    -------
    image : ndarray
        2D image array (YX).
    metadata : dict
        Best-effort metadata dictionary (may include OME XML under 'ome').
    """
    image_data = tifffile.imread(channel_path)
    metadata: dict = {}
    try:
        # TiffFile provides access to OME metadata when present.
        with tifffile.TiffFile(channel_path) as tf:
            ome = tf.ome_metadata
            if ome:
                metadata['ome'] = ome
    except Exception:
        # Do not fail on metadata extraction; image data is the primary output.
        pass

    return image_data, metadata


def compute_cell_intensities(
    mask: NDArray,
    channel: NDArray,
    channel_name: str,
    min_area: int = 0
) -> pd.DataFrame:
    """
    Compute mean intensity per cell for a single channel.

    Parameters
    ----------
    mask : ndarray, shape (Y, X)
        Segmentation mask with cell labels.
    channel : ndarray, shape (Y, X)
        Channel image.
    channel_name : str
        Name of the channel.
    min_area : int, optional
        Minimum cell area in pixels. Default is 0.

    Returns
    -------
    df : DataFrame
        Cell measurements with columns: label, [channel_name], area, centroid, etc.

    Notes
    -----
    Uses efficient bincount for intensity computation.
    """
    logger.info(f"  Computing intensities for {channel_name}")

    # Filter cells by area. We treat label 0 as background and keep labels
    # with pixel counts > min_area. Using numpy.unique is efficient for
    # relatively small label sets and keeps memory usage low.
    labels, counts = np.unique(mask, return_counts=True)
    valid_labels = labels[(labels != 0) & (counts > min_area)]

    if len(valid_labels) == 0:
        logger.warning(f"    No valid cells found")
        return pd.DataFrame()

    # Filter mask
    mask_filtered = np.where(np.isin(mask, valid_labels), mask, 0)

    if np.all(mask_filtered == 0):
        logger.warning(f"    Filtered mask is empty")
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

    # Compute mean intensities using np.bincount which is fast and avoids
    # per-label loops. We flatten arrays and compute sums/counts per label.
    flat_mask = mask_filtered.ravel()
    flat_channel = channel.ravel()

    sum_per_label = np.bincount(flat_mask, weights=flat_channel)
    count_per_label = np.bincount(flat_mask)

    # Avoid division by zero
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

    # Rename centroid columns
    result_df.rename(
        columns={'centroid-0': 'y', 'centroid-1': 'x'},
        inplace=True
    )

    logger.info(f"    Cells quantified: {len(result_df)}")

    return result_df


def quantify_multichannel(
    mask: NDArray,
    channel_paths: List[str],
    min_area: int = 0
) -> pd.DataFrame:
    """
    Quantify all channels for all cells.

    Parameters
    ----------
    mask : ndarray, shape (Y, X)
        Segmentation mask.
    channel_paths : list of str
        Paths to channel image files.
    min_area : int, optional
        Minimum cell area filter. Default is 0.

    Returns
    -------
    df : DataFrame
        Combined measurements for all channels.

    Notes
    -----
    Processes channels sequentially to avoid memory issues.
    """
    logger.info(f"Quantifying {len(channel_paths)} channels")

    all_channel_dfs = []

    for idx, channel_path in enumerate(channel_paths):
        channel_name = Path(channel_path).stem.split('_')[-1]
        logger.info(f"[{idx + 1}/{len(channel_paths)}] {channel_name}")

        # Load channel
        channel_image, _ = load_channel_image(channel_path)

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
        # Remove duplicate columns
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
    min_area: int = 0,
    log_file: str = None
) -> pd.DataFrame:
    """
    Run complete quantification pipeline.

    Parameters
    ----------
    mask_path : str
        Path to segmentation mask (.npy file).
    channel_paths : list of str
        Paths to channel image files.
    output_path : str
        Path to save output CSV.
    min_area : int, optional
        Minimum cell area filter. Default is 0.
    log_file : str, optional
        Log file path. Default is None.

    Returns
    -------
    df : DataFrame
        Quantification results.
    """
    # Setup logging.
    if log_file:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info("=" * 80)
    logger.info("Starting Quantification Pipeline (CPU)")
    logger.info("=" * 80)

    # Load segmentation mask
    logger.info(f"Loading segmentation mask: {mask_path}")
    mask = np.load(mask_path)
    logger.info(f"  Mask shape: {mask.shape}")
    logger.info(f"  Unique labels: {len(np.unique(mask)) - 1}")  # Subtract background

    # Quantify
    result_df = quantify_multichannel(
        mask,
        channel_paths,
        min_area=min_area
    )

    # Save
    if not result_df.empty:
        logger.info(f"Saving results: {output_path}")
        result_df.to_csv(output_path, index=False)
        logger.info(f"  Rows: {len(result_df)}")
        logger.info(f"  Columns: {len(result_df.columns)}")
    else:
        logger.warning("No results to save")

    logger.info("=" * 80)
    logger.info("Quantification Complete")
    logger.info("=" * 80)

    return result_df


def parse_args():
    """Parse command-line arguments.

    Supports two modes: cpu (default) and gpu. If GPU libraries are not
    available and mode=gpu is requested, the script will fall back to CPU
    processing with a warning.
    """
    parser = argparse.ArgumentParser(
        description='Cell quantification (CPU/GPU modes)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--mode',
        choices=['cpu', 'gpu'],
        default='cpu',
        help='Processing mode: cpu (default) or gpu (requires cupy/cucim)'
    )

    parser.add_argument(
        '--patient_id',
        type=str,
        required=False,
        help='Patient identifier (optional)'
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
        required=False,
        help='Exact output file path for CSV (overrides patient_id/outdir naming)'
    )

    parser.add_argument(
        '--min_area',
        type=int,
        default=0,
        help='Minimum cell area (pixels)'
    )

    parser.add_argument(
        '--log_file',
        type=str,
        help='Log file path'
    )

    return parser.parse_args()


def main():
    """Main entry point which dispatches to CPU or GPU processing based on args.mode."""
    args = parse_args()

    try:
        # Create output directory
        ensure_dir(args.outdir)
        if args.log_file:
            setup_file_logger(args.log_file)

        # Get channel files
        channel_files = [
            os.path.join(args.indir, f)
            for f in os.listdir(args.indir)
            if f.lower().endswith(('.tif', '.tiff', '.ome.tif', '.ome.tiff'))
        ]

        if not channel_files:
            raise ValueError(f"No channel files found in {args.indir}")

        logger.info(f"Found {len(channel_files)} channel files")

        # Resolve output path (explicit output_file wins)
        if args.output_file:
            output_path = args.output_file
        else:
            pid = args.patient_id or Path(args.mask_file).stem
            output_path = os.path.join(args.outdir, f"{pid}_quantification.csv")

        if args.mode == 'cpu':
            logger.info('Running CPU quantification')
            run_quantification(
                mask_path=args.mask_file,
                channel_paths=channel_files,
                output_path=output_path,
                min_area=args.min_area,
                log_file=args.log_file
            )

        else:  # gpu mode
            logger.info('Running GPU quantification (if GPU helper module available)')
            # Resolve patient id for naming
            pid = args.patient_id or Path(args.mask_file).stem

            try:
                from scripts import quantify_gpu
            except Exception:
                logger.warning('GPU helper module not available; falling back to CPU implementation')
                run_quantification(
                    mask_path=args.mask_file,
                    channel_paths=channel_files,
                    output_path=output_path,
                    min_area=args.min_area,
                    log_file=args.log_file
                )
            else:
                try:
                    # GPU helper writes output itself; pass patient id and outdir
                    quantify_gpu.run_marker_quantification(
                        indir=args.indir,
                        mask_file=args.mask_file,
                        outdir=args.outdir,
                        patient_id=pid,
                        size_cutoff=args.min_area,
                    )
                except Exception as e:
                    logger.error(f'GPU quantification failed: {e}', exc_info=True)
                    logger.info('Falling back to CPU quantification')
                    run_quantification(
                        mask_path=args.mask_file,
                        channel_paths=channel_files,
                        output_path=output_path,
                        min_area=args.min_area,
                        log_file=args.log_file
                    )

        return 0

    except Exception as e:
        logger.error(f"Quantification failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())