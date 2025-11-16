#!/usr/bin/env python3
"""BaSiC Illumination Correction Preprocessing.

This module provides utilities to apply BaSiC shading correction to large
multichannel images by tiling into FOVs and reconstructing the corrected image.
This version loads a single multichannel image, processes channels in parallel,
and saves the result back to a single TIFF file.
"""

from __future__ import annotations

import os
import time
import argparse
import logging
import re
from pathlib import Path
from typing import Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tifffile
from numpy.typing import NDArray

# BaSiC imports - optional. Keep import guarded so tests can import this module
# without installing basicpy. If BaSiC is unavailable, apply_basic_correction
# will raise if called (or users can handle fallback behavior).
try:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Force CPU for JAX
    from basicpy import BaSiC  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    BaSiC = None

# --- Minimal Fallbacks for Local Imports (Ensuring script is self-contained) ---
try:
    from utils import logging_config
except Exception:
    import logging as _logging

    class _FallbackLoggingConfig:
        @staticmethod
        def setup_logging():
            _logging.basicConfig(level=_logging.INFO)

    logging_config = _FallbackLoggingConfig()

# Dummy implementations for required local imports if missing
try:
    from _common import ensure_dir, setup_file_logger
except Exception:
    def ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)
    
    def setup_file_logger(log_file):
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
# --- End Fallbacks ---

# Setup logging.
logging_config.setup_logging()
logger = logging.getLogger(__name__)

__all__ = [
    "split_image_into_fovs",
    "reconstruct_image_from_fovs",
    "apply_basic_correction",
    "preprocess_multichannel_image",
]


def split_image_into_fovs(
    image: NDArray,
    fov_size: Tuple[int, int],
    overlap: int = 0
) -> Tuple[NDArray, List[Tuple[int, int, int, int]], Tuple[int, int]]:
    """
    Split image (H, W) into field-of-view (FOV) tiles for BaSiC processing.
    """
    if image.ndim != 2:
        # Enforcing 2D input as BaSiC works on single channels (H, W)
        raise ValueError(f"Image must be 2D, got shape {image.shape}")

    image_h, image_w = image.shape
    fov_h, fov_w = fov_size

    if overlap >= min(fov_h, fov_w):
        raise ValueError(f"Overlap ({overlap}) must be < FOV size {fov_size}")

    # Calculate stride
    stride_h = fov_h - overlap
    stride_w = fov_w - overlap

    # Calculate number of FOVs needed
    n_fovs_h = int(np.ceil((image_h - overlap) / stride_h))
    n_fovs_w = int(np.ceil((image_w - overlap) / stride_w))

    n_fovs = n_fovs_h * n_fovs_w

    logger.debug(
        f"Splitting {image.shape} image into {n_fovs} FOVs "
        f"({n_fovs_h}×{n_fovs_w}) of size {fov_size}"
    )

    # Prepare output array (N, fov_h, fov_w)
    fov_stack = np.zeros((n_fovs, fov_h, fov_w), dtype=image.dtype)
    positions = []
    idx = 0

    # Extract FOVs
    for i in range(n_fovs_h):
        row_start = i * stride_h
        row_end = min(row_start + fov_h, image_h)
        actual_h = row_end - row_start

        for j in range(n_fovs_w):
            col_start = j * stride_w
            col_end = min(col_start + fov_w, image_w)
            actual_w = col_end - col_start

            # Extract FOV
            fov_stack[idx, :actual_h, :actual_w] = \
                image[row_start:row_end, col_start:col_end]

            positions.append((row_start, col_start, actual_h, actual_w))
            idx += 1

    return fov_stack, positions, (fov_h, fov_w)


def reconstruct_image_from_fovs(
    fov_stack: NDArray,
    positions: List[Tuple[int, int, int, int]],
    original_shape: Tuple[int, ...]
) -> NDArray:
    """
    Reconstruct 2D image from 3D FOV tiles stack.
    """
    reconstructed = np.zeros(original_shape, dtype=fov_stack.dtype)

    for idx, (row_start, col_start, h, w) in enumerate(positions):
        # fov_stack is 3D (N, fov_h, fov_w), reconstructed is 2D (H, W)
        reconstructed[row_start:row_start + h, col_start:col_start + w] = \
            fov_stack[idx, :h, :w]

    return reconstructed


def apply_basic_correction(
    image: NDArray,
    fov_size: Tuple[int, int] = (1950, 1950),
    get_darkfield: bool = True,
    autotune: bool = False,
    n_iter: int = 100,
    **basic_kwargs
) -> Tuple[NDArray, object]:
    """
    Apply BaSiC illumination correction to a single channel image (H, W).
    """
    if image.ndim != 2:
        raise ValueError(f"apply_basic_correction requires a 2D image, got shape {image.shape}")

    logger.info(f"Applying BaSiC correction to {image.shape} image")
    start_time = time.time()
    
    if BaSiC is None:
        raise ImportError("basicpy (BaSiC) is required but not installed.")

    # Split into FOVs
    fov_stack, positions, _ = split_image_into_fovs(
        image, fov_size, overlap=0
    )

    # Initialize BaSiC model and (Autotune if requested)...
    basic = BaSiC(get_darkfield=get_darkfield, **basic_kwargs)

    if autotune:
        logger.info(f"Autotuning BaSiC parameters ({n_iter} iterations)...")
        autotune_start = time.time()
        basic.autotune(fov_stack, early_stop=True, n_iter=n_iter)
        logger.info(f"Autotuning completed in {time.time() - autotune_start:.2f}s")
        
    # Fit and transform
    logger.info("Fitting and transforming FOVs...")
    transform_start = time.time()
    corrected_fovs = basic.fit_transform(fov_stack)
    logger.info(f"Transformation completed in {time.time() - transform_start:.2f}s")

    # Reconstruct image
    reconstructed = reconstruct_image_from_fovs(
        corrected_fovs,
        positions,
        image.shape
    )

    total_time = time.time() - start_time
    logger.info(f"BaSiC correction completed in {total_time:.2f}s")

    return reconstructed, basic


def _process_single_channel_from_stack(
    channel_image: NDArray,
    channel_index: int,
    channel_name: str,
    fov_size: Tuple[int, int],
    skip_dapi: bool,
    autotune: bool,
    n_iter: int,
    basic_kwargs: dict
) -> Tuple[int, NDArray]:
    """Worker function to process a single channel slice from a stack."""
    logger.info(f"Processing channel #{channel_index} ({channel_name}) - shape: {channel_image.shape}")

    # Skip DAPI if requested
    if skip_dapi and 'DAPI' in channel_name.upper():
        logger.info(f"  Skipping BaSiC correction for DAPI")
        return channel_index, channel_image

    # Apply BaSiC correction
    try:
        corrected, _ = apply_basic_correction(
            channel_image,
            fov_size=fov_size,
            autotune=autotune,
            n_iter=n_iter,
            **basic_kwargs
        )
        return channel_index, corrected

    except Exception as e:
        logger.error(f"  BaSiC correction failed for {channel_name}: {e}")
        logger.warning(f"  Using uncorrected channel for {channel_name}")
        return channel_index, channel_image


def preprocess_multichannel_image(
    image_path: str,
    channel_names: List[str],
    output_path: str, # Now a TIFF path
    fov_size: Tuple[int, int] = (1950, 1950),
    skip_dapi: bool = True,
    autotune: bool = False,
    n_iter: int = 3,
    n_workers: int = 4,
    **basic_kwargs
) -> NDArray:
    """
    Apply BaSiC preprocessing to a single multichannel image in parallel and save as TIFF.
    """
    logger.info(f"Loading multichannel image from {image_path}")
    
    # 1. Load the entire multichannel image (C, H, W)
    try:
        # Reading as Dask array, then computing to ensure (C, H, W) or (Z, C, H, W) is handled
        # For simplicity with BaSiC, we primarily target (C, H, W).
        multichannel_stack = tifffile.imread(image_path)
        
        # Adjust dimensions if tifffile returns (H, W) or (H, W, C)
        if multichannel_stack.ndim == 2:
            multichannel_stack = np.expand_dims(multichannel_stack, axis=0)
        elif multichannel_stack.ndim == 3 and multichannel_stack.shape[2] == len(channel_names):
            # If (H, W, C), transpose to (C, H, W)
            multichannel_stack = np.transpose(multichannel_stack, (2, 0, 1))

    except Exception as e:
        logger.critical(f"Failed to load multichannel image {image_path}: {e}")
        raise
        
    n_channels, H, W = multichannel_stack.shape
    
    if n_channels != len(channel_names):
        logger.warning(
            f"Channel count mismatch: Found {n_channels} layers, "
            f"but detected {len(channel_names)} names from filename. "
            f"Using generic names for excess channels."
        )
        channel_names = channel_names[:n_channels] + [f"Channel_{i}" for i in range(len(channel_names), n_channels)]

    logger.info(
        f"Processing {n_channels} channels ({H}x{W}) in parallel "
        f"with {n_workers} workers."
    )

    results = {}
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_channels):
            # Pass the 2D slice (H, W) for processing
            future = executor.submit(
                _process_single_channel_from_stack,
                multichannel_stack[i, ...],
                i,
                channel_names[i],
                fov_size,
                skip_dapi,
                autotune,
                n_iter,
                basic_kwargs
            )
            futures.append(future)

        # Collect results, ensuring correct order via channel index
        for future in as_completed(futures):
            try:
                channel_index, result_array = future.result()
                results[channel_index] = result_array
            except Exception as e:
                logger.error(f"❌ Parallel channel processing failed: {e}")
                raise RuntimeError("Parallel channel processing failed.")
                
    # 2. Reconstruct the final stack in the original index order
    preprocessed_channels = [
        results[i] for i in range(n_channels)
    ]

    # Stack channels (C, Y, X format)
    preprocessed = np.stack(preprocessed_channels, axis=0)
    logger.info(f"Stacked shape: {preprocessed.shape}")

    # 3. Save as TIFF/OME-TIFF
    logger.info(f"Saving corrected multichannel image to {output_path}")
    try:
        # Use tifffile to save, potentially preserving OME-TIFF metadata 
        # (though complex metadata transfer is beyond simple tifffile use)
        tifffile.imwrite(
            output_path, 
            preprocessed, 
            imagej=True, # Common for biomedical images
            photometric='minisblack'
        )
        logger.info(f"Successfully saved corrected image to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save output TIFF to {output_path}: {e}")
        raise

    return preprocessed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Apply BaSiC illumination correction to multichannel images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to the multichannel image file (e.g., <ID>_DAPI_<MARKER1>...ome.tiff)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory'
    )

    parser.add_argument(
        '--fov_size',
        type=int,
        default=1950,
        help='FOV size for BaSiC tiling'
    )
    
    parser.add_argument(
        '--n_workers',
        type=int,
        default=4,
        help='Maximum number of channels to process in parallel.'
    )

    parser.add_argument(
        '--skip_dapi',
        action='store_true',
        help='Skip BaSiC correction for DAPI channel'
    )

    parser.add_argument(
        '--autotune',
        action='store_true',
        help='Autotune BaSiC parameters'
    )

    parser.add_argument(
        '--n_iter',
        type=int,
        default=3,
        help='Number of autotuning iterations'
    )

    parser.add_argument(
        '-l', '--log_file',
        type=str,
        help='Path to log file'
    )

    return parser.parse_args()

def find_channel_names_from_path(image_path: str) -> List[str]:
    """
    Extracts channel names from a multichannel image path assuming the format:
    <ID>_<CHANNEL1>_<CHANNEL2>...<EXT>
    e.g., 'id1_DAPI_Marker1_Marker2.ome.tiff' -> ['DAPI', 'Marker1', 'Marker2']
    """
    p = Path(image_path)
    base_name = p.stem # 'id1_DAPI_Marker1_Marker2'
    
    parts = base_name.split('_')
    
    if len(parts) < 2:
        logger.warning(f"Could not parse channel names from path: {image_path}. Using generic names.")
        return ["DAPI", "Channel_1", "Channel_2", "Channel_3", "Channel_4"] # Provide a sufficient list

    # Skip the first part (the ID)
    channel_names = parts[1:]
    
    if not channel_names:
        logger.warning("Filename contained an ID but no subsequent channel names. Using generic names.")
        return ["DAPI", "Channel_1", "Channel_2", "Channel_3", "Channel_4"]

    return channel_names


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging.
    if args.log_file:
        setup_file_logger(args.log_file)
        
    # Create output directory.
    ensure_dir(args.output_dir)

    image_path = args.image
    image_basename = os.path.basename(image_path)
    
    # Extract channel names from the filename
    channel_names = find_channel_names_from_path(image_path)

    # Determine the output path, appending '_corrected' before the extension
    base, ext = os.path.splitext(image_basename)
    
    # Handle extensions like .ome.tiff
    if base.endswith(".ome"):
        base = base[:-4]
        ext = ".ome" + ext # Recombine to keep the full extension

    output_filename = f"{base}_corrected{ext}"
    output_path = os.path.join(
        args.output_dir,
        output_filename
    )

    logger.info(f"Expected channel order (from filename): {channel_names}")

    # Process the single multichannel image.
    try:
        preprocess_multichannel_image(
            image_path=image_path,
            channel_names=channel_names,
            output_path=output_path,
            fov_size=(args.fov_size, args.fov_size),
            skip_dapi=args.skip_dapi,
            autotune=args.autotune,
            n_iter=args.n_iter,
            n_workers=args.n_workers
        )

        logger.info(f"Preprocessing completed successfully for {image_path}. Output saved to {output_path}")
        return 0

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())