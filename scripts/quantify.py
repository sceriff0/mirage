#!/usr/bin/env python3
"""
Cell Quantification - CPU Version
==================================
Quantify marker expression per cell using Dask for parallel processing.

Pure functional implementation - NO CLASSES.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from skimage.measure import regionprops_table
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster
from tqdm.dask import TqdmCallback
from numpy.typing import NDArray

from utils.io import load_pickle
from utils import logging_config

# Setup logging
logging_config.setup_logging()
logger = logging.getLogger(__name__)


def load_channel_image(
    channel_path: str
) -> Tuple[NDArray, dict]:
    """
    Load single channel image.

    Parameters
    ----------
    channel_path : str
        Path to channel image file.

    Returns
    -------
    image : ndarray, shape (Y, X)
        Channel image data.
    metadata : dict
        Image metadata.
    """
    img = AICSImage(channel_path)
    image_data = img.get_image_data("YX")
    pixel_sizes = img.physical_pixel_sizes

    metadata = {
        'pixel_size_x': pixel_sizes.X,
        'pixel_size_y': pixel_sizes.Y
    }

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

    # Filter cells by area
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

    # Compute mean intensities using bincount (fast!)
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
    # Setup logging
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Cell quantification (CPU version)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--patient_id',
        type=str,
        required=True,
        help='Patient identifier'
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
    """Main entry point."""
    args = parse_args()

    try:
        # Create output directory
        os.makedirs(args.outdir, exist_ok=True)

        # Get channel files
        channel_files = [
            os.path.join(args.indir, f)
            for f in os.listdir(args.indir)
            if f.endswith(('.tif', '.tiff', '.TIF', '.TIFF'))
        ]

        if not channel_files:
            raise ValueError(f"No channel files found in {args.indir}")

        logger.info(f"Found {len(channel_files)} channel files")

        # Output path
        output_path = os.path.join(
            args.outdir,
            f"{args.patient_id}_quantification.csv"
        )

        # Run quantification
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
#!/usr/bin/env python3
# Standard library
import os
import time
import pickle
import argparse
import psutil

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from aicsimageio import AICSImage

# GPU libraries
import cupy as cp
import cucim.skimage as cskimage
from cucim.skimage.measure import regionprops_table as gpu_regionprops_table


def print_memory_usage(prefix=""):
    """Print CPU and GPU memory usage"""
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024**3  # in GB
    
    # GPU memory
    mempool = cp.get_default_memory_pool()
    gpu_mem = mempool.used_bytes() / 1024**3  # in GB
    gpu_total = mempool.total_bytes() / 1024**3
    
    print(f"{prefix}CPU Memory: {cpu_mem:.2f} GB | GPU Memory: {gpu_mem:.2f}/{gpu_total:.2f} GB")


def load_pickle(path):
    """Load a pickle file"""
    with open(path, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data


def save_pickle(obj, path):
    """Save object to pickle file"""
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def import_images(path):
    """Import image and metadata"""
    img = AICSImage(path)

    # Handle pixel size across versions
    try:
        pixel_microns = img.physical_pixel_sizes
    except AttributeError:
        pixel_microns = img.get_physical_pixel_size()

    # Handle dims across versions
    dims = img.dims
    #order = dims if isinstance(dims, str) else dims.order

    #shape = img.shape
    #series = img.scenesavailable_scenes

    return img, pixel_microns


def gpu_extract_features(
    segmentation_mask,
    channel_image,
    chan_name,
    size_cutoff=0,
    verbose=True
):
    """Extract features using GPU-accelerated regionprops"""
    
    if verbose:
        print(f"Processing channel: {chan_name}")
        print_memory_usage(f"  Before GPU transfer ({chan_name}): ")
    
    # Transfer to GPU
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
    valid_ids_cpu = valid_ids.get()  # Need CPU version for isin
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
    
    # Convert to pandas DataFrame
    props_cpu = {k: (v.get() if isinstance(v, cp.ndarray) else v) for k, v in props.items()}
    props_df = pd.DataFrame(props_cpu).set_index("label", drop=False)

    
    # Calculate mean intensities using GPU
    flat_mask = mask_filtered.ravel()
    flat_image = channel_gpu.ravel()
    
    # Use CuPy bincount for efficient computation
    #max_label = int(mask_filtered.max()) + 1
    sum_per_label = cp.bincount(flat_mask, weights=flat_image)#, minlength=max_label)
    count_per_label = cp.bincount(flat_mask)#, minlength=max_label)
    
    # Compute means
    # Note: cp.errstate was removed in newer CuPy versions
    # Using cp.where to avoid division by zero instead
    means = cp.where(count_per_label != 0,
                     sum_per_label / count_per_label,
                     0.0)
    
    # Extract mean values for valid labels
    mean_values = cp.array([means[int(label)] if label < len(means) else 0 
                            for label in valid_ids])
    
    # Convert to CPU and create DataFrame
    mean_values_cpu = mean_values.get()
    valid_ids_cpu = valid_ids.get()
    
    intensity_df = pd.DataFrame({chan_name: mean_values_cpu}, index=valid_ids_cpu)
    
    # Join with properties DataFrame
    df = intensity_df.join(props_df)
    df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)
    
    # Clear GPU memory
    del mask_gpu, channel_gpu, mask_filtered, flat_mask, flat_image
    del sum_per_label, count_per_label, means, mean_values
    cp.get_default_memory_pool().free_all_blocks()
    
    return df


def extract_features_gpu(
    channels_files,
    segmentation_mask,
    output_file=None,
    size_cutoff=0,
    verbose=True,
    write=False
):
    """Main feature extraction function using GPU acceleration"""
    
    segmentation_mask = segmentation_mask.squeeze()
    results_all = []
    
    for file in channels_files:
        chan_name = os.path.basename(file).split('.')[0].split('_')[-1]
        
        if verbose:
            print(f"\n--- Processing channel: {chan_name} ---")
        
        # Load channel image
        channel_data, _ = import_images(file)
        channel_image = channel_data.get_image_data("YX")
        
        # Extract features using GPU
        df = gpu_extract_features(
            segmentation_mask,
            channel_image,
            chan_name,
            size_cutoff,
            verbose
        )
        
        if not df.empty:
            results_all.append(df)
        
        # Clear memory after each channel
        del channel_image
        cp.get_default_memory_pool().free_all_blocks()
    
    # Combine results
    if results_all:
        result_df = pd.concat(results_all, axis=1)
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        
        if write and output_file:
            result_df.to_csv(output_file, index=False)
            if verbose:
                print(f"Saved output to: {output_file}")
        
        return result_df
    else:
        return pd.DataFrame()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run GPU-accelerated marker quantification for a patient.")
    parser.add_argument("--patient_id", required=True, help="Patient ID to process")
    parser.add_argument(
        "--indir", required=True, help="Input directory with registered channel images"
    )
    parser.add_argument(
        "--mask_file", required=True, help="Path to segmentation mask .npy file"
    )
    parser.add_argument(
        "--positions_file", required=True, help="Path to crop positions .pkl file"
    )
    parser.add_argument(
        "--outdir", required=True, help="Output directory to save quantification results"
    )
    return parser.parse_args()


def run_marker_quantification(
    indir, mask_file, outdir, patient_id, size_cutoff=0, verbose=True
):
    """Run the marker quantification pipeline"""
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    
    output_file = os.path.join(
        outdir, f"{patient_id}_segmentation_markers_data_FULL.csv"
    )
    
    print(f"Loading segmentation mask from: {mask_file}")
    segmentation_mask = np.load(mask_file).squeeze()
    print(f"Mask shape: {segmentation_mask.shape}")
    
    # Get all channel files
    files = [os.path.join(indir, file) for file in os.listdir(indir) 
             if file.endswith(('.tif', '.tiff', '.ome.tif', '.ome.tiff'))]
    print(f"Found {len(files)} channel files")
    
    # Run GPU-accelerated feature extraction
    markers_data = extract_features_gpu(
        channels_files=files,
        segmentation_mask=segmentation_mask,
        output_file=output_file,
        size_cutoff=0,
        verbose=verbose,
        write=True,
    )
    
    return markers_data


def main():
    """Main function"""
    args = parse_args()
    
    # Run the pipeline
    start_time = time.time()
    
    markers_data = run_marker_quantification(
        indir=args.indir,
        mask_file=args.mask_file,
        outdir=args.outdir,
        patient_id=args.patient_id,
        size_cutoff=0,
    )
    
    end_time = time.time()
    print(f"\n=== Completed in {end_time - start_time:.2f} seconds ===")
    
    # Clear GPU memory
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    
    if not markers_data.empty:
        print(f"Processed {len(markers_data)} cells")
        print(f"Columns: {list(markers_data.columns)}")


if __name__ == "__main__":
    main()