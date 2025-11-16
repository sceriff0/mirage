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

# Optional GPU libraries
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
    channels_files: List[str],
    segmentation_mask: np.ndarray,
    output_file: Optional[str] = None,
    size_cutoff: int = 0,
    verbose: bool = True,
    write: bool = False
) -> pd.DataFrame:
    """Extract features for all channels using GPU.

    Parameters
    ----------
    channels_files : list of str
        Paths to channel image files.
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

    for file in channels_files:
        chan_name = os.path.basename(file).split('.')[0].split('_')[-1]

        if verbose:
            logger.info(f"Processing channel: {chan_name}")

        channel_image, _ = load_image(file)

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
    indir: str,
    mask_file: str,
    outdir: str,
    patient_id: str,
    size_cutoff: int = 0,
    verbose: bool = True
) -> pd.DataFrame:
    """Run GPU-accelerated marker quantification pipeline.

    Parameters
    ----------
    indir : str
        Directory containing channel images.
    mask_file : str
        Path to segmentation mask (.npy).
    outdir : str
        Output directory.
    patient_id : str
        Patient identifier for output naming.
    size_cutoff : int, optional
        Minimum cell area.
    verbose : bool, optional
        Enable verbose logging.

    Returns
    -------
    DataFrame
        Quantification results.
    """
    os.makedirs(outdir, exist_ok=True)

    output_file = os.path.join(outdir, f"{patient_id}_quantification_gpu.csv")
    segmentation_mask = np.load(mask_file).squeeze()

    files = [
        os.path.join(indir, f)
        for f in os.listdir(indir)
        if f.endswith(('.tif', '.tiff', '.ome.tif', '.ome.tiff'))
    ]

    markers_data = extract_features_gpu(
        channels_files=files,
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
    parser.add_argument("--patient_id", required=True, help="Patient ID")
    parser.add_argument("--indir", required=True, help="Channel images directory")
    parser.add_argument("--mask_file", required=True, help="Segmentation mask path")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_marker_quantification(
        args.indir, args.mask_file, args.outdir, args.patient_id
    )
