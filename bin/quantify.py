#!/usr/bin/env python3
"""Cell quantification for single-channel processing.

Designed for Nextflow pipelines where each channel is processed separately
and results are merged afterward. Supports both CPU and GPU processing.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


__all__ = [
    "compute_morphology",
    "compute_channel_intensity",
    "quantify_single_channel",
    "run_quantification",
]


def load_image(path: str) -> Tuple[NDArray, dict]:
    """Load image from file.
    
    Supports .npy, .tif, .tiff, .ome.tif, .ome.tiff formats.
    
    Parameters
    ----------
    path : str
        Path to image file.
        
    Returns
    -------
    image : ndarray
        Image data.
    metadata : dict
        Image metadata (pixel sizes, etc.).
    """
    path = str(path)
    
    if path.endswith('.npy'):
        return np.load(path), {}
    
    # Use aicsimageio for TIFF files
    try:
        from aicsimageio import AICSImage
        img = AICSImage(path)
        data = img.get_image_data("YX")
        
        metadata = {}
        try:
            metadata['pixel_sizes'] = img.physical_pixel_sizes
        except AttributeError:
            pass
            
        return data, metadata
        
    except ImportError:
        # Fallback to tifffile
        import tifffile
        return tifffile.imread(path), {}


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def compute_morphology(
    mask: NDArray,
    min_area: int = 0
) -> Tuple[Optional[pd.DataFrame], Optional[NDArray], Optional[NDArray]]:
    """Compute morphological properties for all cells.

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

    # Create filtered mask
    mask_filtered = np.where(np.isin(mask, valid_labels), mask, 0)

    if np.all(mask_filtered == 0):
        logger.warning("Filtered mask is empty")
        return None, None, None

    # Compute morphological properties
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
    min_area: int = 0
) -> pd.DataFrame:
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

    Returns
    -------
    DataFrame
        Morphological properties + channel intensity for all cells.
    """
    # Compute morphology
    props_df, mask_filtered, valid_labels = compute_morphology(mask, min_area)

    if props_df is None:
        return pd.DataFrame()

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

    return result_df


def run_quantification(
    mask_path: str,
    channel_path: str,
    output_path: str,
    min_area: int = 0,
    channel_name: str = None
) -> pd.DataFrame:
    """Run quantification for a single channel.

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
    # Use provided channel name, or extract from filename as fallback
    if channel_name is None:
        channel_name = Path(channel_path).stem.split('_')[-1]
        logger.info(f"Channel name parsed from filename: {channel_name}")
    
    logger.info("=" * 60)
    logger.info(f"Quantifying channel: {channel_name}")
    logger.info("=" * 60)

    # Load segmentation mask
    logger.info(f"Loading mask: {mask_path}")
    if mask_path.endswith('.npy'):
        mask = np.load(mask_path).squeeze()
    else:
        mask, _ = load_image(mask_path)
        mask = mask.squeeze()
    logger.info(f"Mask shape: {mask.shape}")
    # Load channel image
    logger.info(f"Loading channel: {channel_path}")
    channel_image, _ = load_image(channel_path)
    logger.info(f"Channel shape: {channel_image.shape}")

    # Validate shapes match
    if mask.shape != channel_image.squeeze().shape:
        raise ValueError(
            f"Shape mismatch: mask {mask.shape} vs channel {channel_image.shape}"
        )

    # Quantify
    result_df = quantify_single_channel(
        mask, channel_image, channel_name, min_area=min_area
    )

    # Save
    if not result_df.empty:
        logger.info(f"Saving: {output_path}")
        result_df.to_csv(output_path, index=False)
        logger.info(f"Cells: {len(result_df)}, Columns: {list(result_df.columns)}")
    else:
        logger.warning("No results to save")
        # Create empty CSV with expected columns
        empty_df = pd.DataFrame(columns=[
            'label', 'y', 'x', 'area', 'eccentricity', 'perimeter',
            'convex_area', 'axis_major_length', 'axis_minor_length', channel_name
        ])
        empty_df.to_csv(output_path, index=False)

    logger.info("=" * 60)
    logger.info("Complete")
    logger.info("=" * 60)

    return result_df


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

    return parser.parse_args()


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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
    run_quantification(
        mask_path=args.mask_file,
        channel_path=args.channel_tiff,
        output_path=output_path,
        min_area=args.min_area,
        channel_name=args.channel_name
    )

    return 0


if __name__ == '__main__':
    exit(main())