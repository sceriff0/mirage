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
    verbose: bool = True
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

    # Filter labels by size
    labels, counts = cp.unique(mask_gpu, return_counts=True)
    valid_ids = labels[(labels != 0) & (counts > size_cutoff)]

    if len(valid_ids) == 0:
        logger.warning(f"No valid cells for {chan_name}")
        return pd.DataFrame()

    # Create filtered mask
    mask_filtered = cp.where(cp.isin(mask_gpu, valid_ids), mask_gpu, 0)

    if (mask_filtered == 0).all():
        logger.warning(f"Filtered mask is empty for {chan_name}")
        return pd.DataFrame()

    # GPU-accelerated regionprops
    props = gpu_regionprops_table(
        mask_filtered,
        properties=[
            "label", "centroid", "eccentricity", "perimeter",
            "convex_area", "area", "axis_major_length", "axis_minor_length"
        ]
    )

    # Transfer props to CPU
    props_cpu = {k: (v.get() if hasattr(v, 'get') else v) for k, v in props.items()}
    props_df = pd.DataFrame(props_cpu).set_index("label", drop=False)

    # Compute mean intensities
    flat_mask = mask_filtered.ravel()
    flat_image = channel_gpu.ravel()

    sum_per_label = cp.bincount(flat_mask, weights=flat_image)
    count_per_label = cp.bincount(flat_mask)

    means = cp.where(count_per_label != 0, sum_per_label / count_per_label, 0.0)
    mean_values = cp.array([means[int(lbl)] if lbl < len(means) else 0 for lbl in valid_ids])

    # Transfer to CPU
    mean_values_cpu = mean_values.get()
    valid_ids_cpu = valid_ids.get()

    intensity_df = pd.DataFrame({chan_name: mean_values_cpu}, index=valid_ids_cpu)
    df = intensity_df.join(props_df)
    df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)

    # Free GPU memory
    del mask_gpu, channel_gpu, mask_filtered, flat_mask, flat_image
    del sum_per_label, count_per_label, means, mean_values
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
    write: bool = False
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
            verbose
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
    merged_image_path: str,
    mask_file: str,
    output_file: str,
    size_cutoff: int = 0,
    mode: str = 'gpu',
    verbose: bool = True
) -> pd.DataFrame:
    """Run marker quantification pipeline.

    Parameters
    ----------
    merged_image_path : str
        Path to merged multi-channel OME-TIFF file.
    mask_file : str
        Path to segmentation mask (.npy or .tif/.tiff).
    output_file : str
        Path for output CSV file.
    size_cutoff : int, optional
        Minimum cell area.
    mode : str, optional
        Processing mode: 'gpu' or 'cpu'.
    verbose : bool, optional
        Enable verbose logging.

    Returns
    -------
    DataFrame
        Quantification results.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Check GPU availability if mode is 'gpu'
    if mode == 'gpu':
        if cp is None or gpu_regionprops_table is None:
            logger.warning("GPU libraries not available, falling back to CPU mode")
            mode = 'cpu'

    if mode == 'cpu':
        raise NotImplementedError("CPU mode not yet implemented - use GPU mode")

    # Load merged multi-channel image
    if verbose:
        logger.info(f"Loading merged image: {merged_image_path}")

    multichannel_image, _ = load_image(merged_image_path)

    # Extract channel names from OME metadata
    channel_names = []
    try:
        import tifffile
        import xml.etree.ElementTree as ET

        with tifffile.TiffFile(merged_image_path) as tif:
            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                # Parse OME-XML for channel names
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                channels = root.findall('.//ome:Channel', ns)
                channel_names = [ch.get('Name', '') for ch in channels]
                # Filter out empty names
                channel_names = [name for name in channel_names if name]

                if verbose and channel_names:
                    logger.info(f"Extracted {len(channel_names)} channel names from OME metadata")
    except Exception as e:
        if verbose:
            logger.warning(f"Could not extract channel names from OME metadata: {e}")

    # If metadata extraction failed or returned wrong number of channels, try looking at directory
    if len(channel_names) != multichannel_image.shape[0]:
        if verbose:
            logger.info(f"Metadata gave {len(channel_names)} names but image has {multichannel_image.shape[0]} channels")
            logger.info(f"Attempting to infer channel names from registered files...")

        try:
            from pathlib import Path

            # Look for registered files to infer channel composition
            parent_dir = Path(merged_image_path).parent
            search_paths = [
                parent_dir / '..' / 'registered',
                parent_dir / '..',
                parent_dir / '..' / 'gpu' / 'registered',
            ]

            registered_files = []
            for search_path in search_paths:
                if search_path.exists():
                    registered_files = list(search_path.glob('*registered*.tif*'))
                    if registered_files:
                        break

            if registered_files:
                # Collect all unique markers from all registered files
                all_markers = set()
                for reg_file in registered_files:
                    name_part = reg_file.stem.replace('_registered', '').replace('_corrected', '')
                    parts = name_part.split('_')
                    # Filter out patient ID (has dash and numbers)
                    file_markers = [p for p in parts if not (len(p) > 0 and p[0].isalpha() and any(c.isdigit() for c in p) and '-' in p)]
                    all_markers.update(file_markers)

                # Sort alphabetically for consistent ordering
                sorted_markers = sorted(list(all_markers))

                if len(sorted_markers) == multichannel_image.shape[0]:
                    channel_names = sorted_markers
                    if verbose:
                        logger.info(f"Inferred channel names from registered files: {channel_names}")
                elif verbose:
                    logger.warning(f"Found {len(sorted_markers)} unique markers but image has {multichannel_image.shape[0]} channels")
        except Exception as e:
            if verbose:
                logger.warning(f"Could not infer channel names from registered files: {e}")

    # Final fallback to generic names
    if len(channel_names) != multichannel_image.shape[0]:
        if verbose:
            logger.warning(f"Using generic channel names (channel_0, channel_1, ...)")
        channel_names = [f"channel_{i}" for i in range(multichannel_image.shape[0])]

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

    # Run quantification
    markers_data = extract_features_gpu(
        multichannel_image=multichannel_image,
        channel_names=channel_names,
        segmentation_mask=segmentation_mask,
        output_file=output_file,
        size_cutoff=size_cutoff,
        verbose=verbose,
        write=True,
    )

    return markers_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='GPU-accelerated cell quantification'
    )
    parser.add_argument(
        "--mode",
        choices=['cpu', 'gpu'],
        default='gpu',
        help="Processing mode: gpu or cpu"
    )
    parser.add_argument(
        "--mask_file",
        required=True,
        help="Path to segmentation mask (.npy or .tif/.tiff)"
    )
    parser.add_argument(
        "--merged_image",
        required=True,
        help="Path to merged multi-channel OME-TIFF file"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output CSV file path"
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
            merged_image_path=args.merged_image,
            mask_file=args.mask_file,
            output_file=args.output_file,
            size_cutoff=args.min_area,
            mode=args.mode,
            verbose=args.verbose
        )
        logger.info("Quantification completed successfully")
    except Exception as e:
        logger.error(f"Quantification failed: {e}")
        raise
