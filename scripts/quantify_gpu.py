#!/usr/bin/env python3
"""GPU helpers for marker quantification.

This module contains GPU-accelerated helpers extracted from the original
`quantify.py`. It is safe to import on systems without GPU support because
imports are guarded.
"""
from __future__ import annotations

import os
import time
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
import tifffile

# Optional GPU libraries.
try:
    import psutil
except Exception:
    psutil = None

try:
    import cupy as cp
    import cucim.skimage as cskimage
    from cucim.skimage.measure import regionprops_table as gpu_regionprops_table
except Exception:
    cp = None
    cskimage = None
    gpu_regionprops_table = None


def print_memory_usage(prefix: str = "") -> None:
    """Print CPU and GPU memory usage (best-effort).

    Parameters
    ----------
    prefix : str
        Optional prefix string for the print statement.
    """
    if psutil is None:
        return
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024**3  # in GB

    gpu_mem = 0.0
    gpu_total = 0.0
    if cp is not None:
        try:
            mempool = cp.get_default_memory_pool()
            gpu_mem = mempool.used_bytes() / 1024**3
            gpu_total = mempool.total_bytes() / 1024**3
        except Exception:
            gpu_mem = 0.0
            gpu_total = 0.0

    print(f"{prefix}CPU Memory: {cpu_mem:.2f} GB | GPU Memory: {gpu_mem:.2f}/{gpu_total:.2f} GB")


def load_pickle(path: str):
    """Load a pickle file.

    Returns
    -------
    object
        Loaded Python object.
    """
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(obj, path: str) -> None:
    """Save object to pickle file.

    Parameters
    ----------
    obj
        Pickle-able Python object.
    path : str
        Destination path.
    """
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def import_images(path: str):
    """Import image and best-effort metadata using ``tifffile``.

    Parameters
    ----------
    path : str
        Path to image file.

    Returns
    -------
    ndarray, dict
        Image array and a metadata dict (may be empty if metadata not found).
    """
    image = tifffile.imread(path)
    metadata = {}
    try:
        with tifffile.TiffFile(path) as tf:
            ome = tf.ome_metadata
            if ome:
                metadata['ome'] = ome
    except Exception:
        pass

    return image, metadata


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
        Label mask (Y,X) with integer cell IDs.
    channel_image : ndarray
        Channel image (Y,X).
    chan_name : str
        Channel name used as column.
    size_cutoff : int
        Minimum pixel area for labels to keep.
    verbose : bool
        Verbosity flag.

    Returns
    -------
    pandas.DataFrame
        Data frame with properties and mean intensity per label.
    """
    if cp is None or gpu_regionprops_table is None:
        raise RuntimeError("GPU libraries not available")

    if verbose:
        print(f"Processing channel: {chan_name}")
        print_memory_usage(f"  Before GPU transfer ({chan_name}): ")

    # Transfer arrays to GPU memory. We squeeze to ensure 2D shapes, then
    # operate using CuPy-backed arrays and cucim regionprops for performance.
    mask_gpu = cp.asarray(segmentation_mask.squeeze())
    channel_gpu = cp.asarray(channel_image.squeeze())

    if verbose:
        print_memory_usage(f"  After GPU transfer ({chan_name}): ")

    # Get unique labels and filter by size
    labels, counts = cp.unique(mask_gpu, return_counts=True)
    valid_ids = labels[(labels != 0) & (counts > size_cutoff)]

    if len(valid_ids) == 0:
        return pd.DataFrame()

    # Filter mask to only include valid labels
    mask_filtered = cp.where(cp.isin(mask_gpu, valid_ids), mask_gpu, 0)

    if (mask_filtered == 0).all():
        return pd.DataFrame()

    # Use GPU-accelerated regionprops
    props = gpu_regionprops_table(
        mask_filtered,
        properties=[
            "label", "centroid", "eccentricity", "perimeter",
            "convex_area", "area", "axis_major_length", "axis_minor_length"
        ]
    )

    props_cpu = {k: (v.get() if hasattr(v, 'get') else v) for k, v in props.items()}
    props_df = pd.DataFrame(props_cpu).set_index("label", drop=False)

    # Calculate mean intensities using CuPy
    flat_mask = mask_filtered.ravel()
    flat_image = channel_gpu.ravel()

    sum_per_label = cp.bincount(flat_mask, weights=flat_image)
    count_per_label = cp.bincount(flat_mask)

    means = cp.where(count_per_label != 0, sum_per_label / count_per_label, 0.0)

    mean_values = cp.array([means[int(label)] if label < len(means) else 0 for label in valid_ids])

    # Move results back to host (CPU) memory for pandas consumption.
    mean_values_cpu = mean_values.get()
    valid_ids_cpu = valid_ids.get()

    intensity_df = pd.DataFrame({chan_name: mean_values_cpu}, index=valid_ids_cpu)

    df = intensity_df.join(props_df)
    df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)

    # Clear GPU memory explicitly to avoid lingering allocations between
    # channels. This helps when processing many channels sequentially.
    del mask_gpu, channel_gpu, mask_filtered, flat_mask, flat_image
    del sum_per_label, count_per_label, means, mean_values
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        # Best-effort free; continue even if the runtime does not expose
        # or allow explicit pool freeing on some CuPy versions.
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
    """Main feature extraction function using GPU acceleration.

    Returns combined DataFrame across channels and optionally writes CSV.
    """
    segmentation_mask = segmentation_mask.squeeze()
    results_all = []

    for file in channels_files:
        chan_name = os.path.basename(file).split('.')[0].split('_')[-1]

        if verbose:
            print(f"\n--- Processing channel: {chan_name} ---")

        # import_images returns a numpy array and metadata dict
        channel_image, _ = import_images(file)

        df = gpu_extract_features(
            segmentation_mask,
            channel_image,
            chan_name,
            size_cutoff,
            verbose
        )

        if not df.empty:
            results_all.append(df)

        # Free per-channel memory promptly
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
                print(f"Saved output to: {output_file}")

        return result_df

    return pd.DataFrame()


# Lightweight runner kept for convenience. In the pipeline, the Nextflow module
# should call appropriate functions directly.
def run_marker_quantification(
    indir: str,
    mask_file: str,
    outdir: str,
    patient_id: str,
    size_cutoff: int = 0,
    verbose: bool = True
) -> pd.DataFrame:
    """Run the marker quantification pipeline using GPU helpers."""
    os.makedirs(outdir, exist_ok=True)

    output_file = os.path.join(outdir, f"{patient_id}_segmentation_markers_data_FULL.csv")

    segmentation_mask = np.load(mask_file).squeeze()

    files = [os.path.join(indir, file) for file in os.listdir(indir)
             if file.endswith(('.tif', '.tiff', '.ome.tif', '.ome.tiff'))]

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
    # Minimal CLI for manual runs; keep simple to avoid coupling with Nextflow args.
    import argparse

    parser = argparse.ArgumentParser(description="Run GPU quantification (helper)")
    parser.add_argument("--patient_id", required=True)
    parser.add_argument("--indir", required=True)
    parser.add_argument("--mask_file", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    run_marker_quantification(args.indir, args.mask_file, args.outdir, args.patient_id)
