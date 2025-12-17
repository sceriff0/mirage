#!/usr/bin/env python3
"""GPU-accelerated marker quantification.

This module provides GPU-accelerated cell quantification using CuPy and CUCIM.
It gracefully falls back to CPU if GPU libraries are unavailable.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from _common import load_image, load_pickle, save_pickle

try:
    import psutil
except ImportError:
    psutil = None

try:
    import cupy as cp
    import cucim.skimage as cskimage
    from cucim.skimage.measure import regionprops_table as gpu_regionprops_table
except ImportError:
    cp = None
    cskimage = None
    gpu_regionprops_table = None

os.environ.setdefault("CUPY_CACHE_DIR", "/tmp/.cupy")

logger = logging.getLogger(__name__)


__all__ = [
    "print_memory_usage",
    "gpu_extract_features",
    "extract_features_gpu",
    "run_marker_quantification",
]


def print_memory_usage(prefix: str = "") -> None:
    """Print CPU and GPU memory usage for debugging.

    Parameters
    ----------
    prefix : str, optional
        Prefix string for the print statement.
    """
    if psutil is None:
        return

    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024**3

    gpu_mem = 0.0
    gpu_total = 0.0
    if cp is not None:
        try:
            mempool = cp.get_default_memory_pool()
            gpu_mem = mempool.used_bytes() / 1024**3
            gpu_total = mempool.total_bytes() / 1024**3
        except Exception:
            pass

    print(f"{prefix}CPU: {cpu_mem:.2f} GB | GPU: {gpu_mem:.2f}/{gpu_total:.2f} GB")


def gpu_extract_features(
    segmentation_mask: np.ndarray,
    channel_image: np.ndarray,
    chan_name: str,
    size_cutoff: int = 0,
    verbose: bool = True,
    chunk_size: int = 500000
) -> pd.DataFrame:
    """Extract features using GPU-accelerated regionprops.

    Parameters
    ----------
    segmentation_mask : ndarray
        Label mask with cell IDs (Y, X).
    channel_image : ndarray
        Channel image (Y, X).
    chan_name : str
        Channel name for column naming.
    size_cutoff : int, optional
        Minimum cell area in pixels.
    verbose : bool, optional
        Enable verbose logging.
    chunk_size : int, optional
        Process regionprops in chunks if cell count exceeds this. Default: 500000.

    Returns
    -------
    DataFrame
        Cell measurements with intensity and morphological properties.

    Raises
    ------
    RuntimeError
        If GPU libraries are not available.
    """
    if cp is None or gpu_regionprops_table is None:
        raise RuntimeError("GPU libraries (cupy/cucim) not available")

    if verbose:
        logger.info(f"Processing channel: {chan_name}")
        print_memory_usage(f"  Before GPU transfer ({chan_name}): ")

    # Transfer to GPU
    mask_gpu = cp.asarray(segmentation_mask.squeeze())
    channel_gpu = cp.asarray(channel_image.squeeze())

    if verbose:
        print_memory_usage(f"  After GPU transfer ({chan_name}): ")

    # Filter labels by size - use lookup table for efficiency
    labels, counts = cp.unique(mask_gpu, return_counts=True)
    valid_mask = (labels != 0) & (counts > size_cutoff)
    valid_ids = labels[valid_mask]

    num_cells = len(valid_ids)
    if num_cells == 0:
        logger.warning(f"No valid cells for {chan_name}")
        return pd.DataFrame()

    if verbose:
        logger.info(f"  Processing {num_cells} cells")

    # Create lookup table for fast filtering (avoids slow cp.isin)
    max_label = int(mask_gpu.max())
    lookup = cp.zeros(max_label + 1, dtype=cp.int32)
    lookup[valid_ids.get()] = valid_ids.get()

    # Create filtered mask using lookup table
    mask_filtered = lookup[mask_gpu]

    if (mask_filtered == 0).all():
        logger.warning(f"Filtered mask is empty for {chan_name}")
        return pd.DataFrame()

    # Compute mean intensities first (fast operation)
    flat_mask = mask_filtered.ravel()
    flat_image = channel_gpu.ravel()

    sum_per_label = cp.bincount(flat_mask, weights=flat_image)
    count_per_label = cp.bincount(flat_mask)

    means = cp.where(count_per_label != 0, sum_per_label / count_per_label, 0.0)

    # Index on GPU, then transfer to CPU (avoid CPU list comprehension)
    valid_ids_idx = valid_ids.astype(cp.int64)
    valid_ids_idx = cp.minimum(valid_ids_idx, len(means) - 1)
    mean_values = means[valid_ids_idx]

    # Transfer intensities to CPU
    mean_values_cpu = mean_values.get()
    valid_ids_cpu = valid_ids.get()

    # Free intermediate arrays
    del flat_mask, flat_image, sum_per_label, count_per_label, means, mean_values

    # Process regionprops in chunks if needed (regionprops is the bottleneck)
    if num_cells > chunk_size:
        if verbose:
            logger.info(f"  Using chunked processing ({num_cells} cells > {chunk_size} threshold)")

        props_list = []
        num_chunks = (num_cells + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, num_cells)
            chunk_labels = valid_ids_cpu[start_idx:end_idx]

            if verbose:
                logger.info(f"    Chunk {chunk_idx+1}/{num_chunks}: cells {start_idx} to {end_idx}")

            # Create chunk mask
            chunk_lookup = cp.zeros(max_label + 1, dtype=cp.int32)
            chunk_lookup[chunk_labels] = chunk_labels
            chunk_mask = chunk_lookup[mask_gpu]

            # Run regionprops on chunk
            chunk_props = gpu_regionprops_table(
                chunk_mask,
                properties=[
                    "label", "centroid", "area", "axis_major_length", "axis_minor_length"
                ]
            )

            # Transfer to CPU
            chunk_props_cpu = {k: (v.get() if hasattr(v, 'get') else v) for k, v in chunk_props.items()}
            props_list.append(pd.DataFrame(chunk_props_cpu))

            del chunk_mask, chunk_lookup
            cp.get_default_memory_pool().free_all_blocks()

        props_df = pd.concat(props_list, ignore_index=True).set_index("label", drop=False)
    else:
        # Process all at once
        props = gpu_regionprops_table(
            mask_filtered,
            properties=[
                "label", "centroid", "area", "axis_major_length", "axis_minor_length"
            ]
        )

        props_cpu = {k: (v.get() if hasattr(v, 'get') else v) for k, v in props.items()}
        props_df = pd.DataFrame(props_cpu).set_index("label", drop=False)

    intensity_df = pd.DataFrame({chan_name: mean_values_cpu}, index=valid_ids_cpu)
    df = intensity_df.join(props_df)
    df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)

    # Free GPU memory
    del mask_gpu, channel_gpu, mask_filtered, lookup
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass

    return df


def extract_features_gpu(
    multichannel_image: np.ndarray,
    channel_names: List[str],
    segmentation_mask: np.ndarray,
    output_file: Optional[str] = None,
    size_cutoff: int = 0,
    verbose: bool = True,
    write: bool = False,
    chunk_size: int = 500000
) -> pd.DataFrame:
    """Extract features for all channels using GPU.

    Parameters
    ----------
    multichannel_image : ndarray
        Multi-channel image array (C, H, W).
    channel_names : list of str
        Names for each channel.
    segmentation_mask : ndarray
        Segmentation mask.
    output_file : str, optional
        Path to save CSV output.
    size_cutoff : int, optional
        Minimum cell area filter.
    verbose : bool, optional
        Enable verbose output.
    write : bool, optional
        Write results to file.
    chunk_size : int, optional
        Process regionprops in chunks if cell count exceeds this. Default: 500000.

    Returns
    -------
    DataFrame
        Combined quantification results.
    """
    segmentation_mask = segmentation_mask.squeeze()
    results_all = []

    # Ensure multichannel_image is (C, H, W)
    if multichannel_image.ndim == 2:
        multichannel_image = multichannel_image[np.newaxis, ...]

    num_channels = multichannel_image.shape[0]

    for i in range(num_channels):
        chan_name = channel_names[i] if i < len(channel_names) else f"channel_{i}"

        if verbose:
            logger.info(f"Processing channel {i+1}/{num_channels}: {chan_name}")

        channel_image = multichannel_image[i]

        df = gpu_extract_features(
            segmentation_mask,
            channel_image,
            chan_name,
            size_cutoff,
            verbose,
            chunk_size
        )

        if not df.empty:
            results_all.append(df)

        # Free memory
        del channel_image
        try:
            if cp is not None:
                cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    if results_all:
        result_df = pd.concat(results_all, axis=1)
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]

        if write and output_file:
            result_df.to_csv(output_file, index=False)
            if verbose:
                logger.info(f"Saved output: {output_file}")

        return result_df

    return pd.DataFrame()


def run_marker_quantification(
    channel_tiff: str,
    mask_file: str,
    outdir: str,
    output_file: str = None,
    size_cutoff: int = 0,
    mode: str = 'gpu',
    verbose: bool = True,
    chunk_size: int = 500000
) -> pd.DataFrame:
    """Run marker quantification pipeline on a single-channel TIFF.

    Parameters
    ----------
    channel_tiff : str
        Path to single-channel TIFF file.
    mask_file : str
        Path to segmentation mask (.npy or .tif/.tiff).
    outdir : str
        Output directory for CSV file.
    output_file : str, optional
        Specific output filename. If not provided, will be auto-generated.
    size_cutoff : int, optional
        Minimum cell area.
    mode : str, optional
        Processing mode: 'gpu' or 'cpu'.
    verbose : bool, optional
        Enable verbose logging.
    chunk_size : int, optional
        Process regionprops in chunks if cell count exceeds this. Default: 500000.

    Returns
    -------
    DataFrame
        Quantification results.
    """
    os.makedirs(outdir, exist_ok=True)

    # Check GPU availability if mode is 'gpu'
    if mode == 'gpu':
        if cp is None or gpu_regionprops_table is None:
            logger.warning("GPU libraries not available, falling back to CPU mode")
            mode = 'cpu'

    if mode == 'cpu':
        raise NotImplementedError("CPU mode not yet implemented - use GPU mode")

    # Load the single channel TIFF file
    from pathlib import Path
    channel_file = Path(channel_tiff)

    if not channel_file.exists():
        raise ValueError(f"Channel TIFF file not found: {channel_tiff}")

    if verbose:
        logger.info(f"Loading channel: {channel_file.name}")

    # Load channel image
    channel_image, _ = load_image(str(channel_file))
    channel_image = channel_image.squeeze()

    # Extract channel name from filename
    # Format: <prefix>_<marker>.tiff
    channel_name = channel_file.stem.split('_')[-1]

    if verbose:
        logger.info(f"Channel name: {channel_name}")

    # Create single-channel array in (C, H, W) format
    multichannel_image = channel_image[np.newaxis, ...]
    channel_names = [channel_name]

    if verbose:
        logger.info(f"Image shape: {multichannel_image.shape}")
        logger.info(f"Channel names: {channel_names}")

    # Load segmentation mask - support both .npy and TIFF formats
    if verbose:
        logger.info(f"Loading segmentation mask: {mask_file}")

    if mask_file.endswith('.npy'):
        segmentation_mask = np.load(mask_file).squeeze()
    else:
        segmentation_mask, _ = load_image(mask_file)
        segmentation_mask = segmentation_mask.squeeze()

    # Determine output file path
    if output_file is None:
        from pathlib import Path
        output_file = os.path.join(outdir, f"{Path(mask_file).stem}_quantification.csv")

    # Run quantification
    markers_data = extract_features_gpu(
        multichannel_image=multichannel_image,
        channel_names=channel_names,
        segmentation_mask=segmentation_mask,
        output_file=output_file,
        size_cutoff=size_cutoff,
        verbose=verbose,
        write=True,
        chunk_size=chunk_size,
    )

    return markers_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='GPU-accelerated cell quantification on single-channel TIFFs'
    )
    parser.add_argument(
        "--mode",
        choices=['cpu', 'gpu'],
        default='gpu',
        help="Processing mode: gpu or cpu"
    )
    parser.add_argument(
        "--channel_tiff",
        required=True,
        help="Path to single-channel TIFF file"
    )
    parser.add_argument(
        "--mask_file",
        required=True,
        help="Path to segmentation mask (.npy or .tif/.tiff)"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for CSV file"
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output CSV filename (optional, auto-generated if not provided)"
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=0,
        help="Minimum cell area in pixels"
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Log file path (optional)"
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Enable verbose output"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500000,
        help="Process regionprops in chunks if cell count exceeds this (for very large datasets)"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    handlers = [logging.StreamHandler()]

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    # Run quantification
    try:
        run_marker_quantification(
            channel_tiff=args.channel_tiff,
            mask_file=args.mask_file,
            outdir=args.outdir,
            output_file=args.output_file,
            size_cutoff=args.min_area,
            mode=args.mode,
            verbose=args.verbose,
            chunk_size=args.chunk_size
        )
        logger.info("Quantification completed successfully")
    except Exception as e:
        logger.error(f"Quantification failed: {e}")
        raise
