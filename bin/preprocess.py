#!/usr/bin/env python3
"""BaSiC Illumination Correction Preprocessing.

This module provides utilities to apply BaSiC shading correction to large
multichannel images by tiling into FOVs and reconstructing the corrected image.
This version loads a single multichannel image, processes channels in parallel,
and saves the result back to a single TIFF file.
"""

from __future__ import annotations

import os
import argparse
import logging
import math
from pathlib import Path

# Add utils directory to path to import shared modules
import sys
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

from logger import get_logger, configure_logging
from typing import Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tifffile
from numpy.typing import NDArray

os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Force CPU for JAX

# Check BaSiCPy and Pydantic versions for compatibility
import basicpy
import pydantic
from packaging import version

BASICPY_VERSION = getattr(basicpy, '__version__', 'unknown')
PYDANTIC_VERSION = getattr(pydantic, '__version__', 'unknown')
AUTOTUNE_AVAILABLE = False

# autotune requires BaSiCPy >= 1.1.0 and Pydantic v2
if BASICPY_VERSION != 'unknown' and PYDANTIC_VERSION != 'unknown':
    try:
        basicpy_ok = version.parse(BASICPY_VERSION) >= version.parse("1.1.0")
        pydantic_ok = version.parse(PYDANTIC_VERSION) >= version.parse("2.0.0")
        AUTOTUNE_AVAILABLE = basicpy_ok and pydantic_ok
    except Exception:
        pass

from basicpy import BaSiC  # type: ignore

from image_utils import ensure_dir
from validation import log_image_stats, clip_negative_values, detect_negative_values

logger = get_logger(__name__)

# Log version info at import time
logger.info(f"BaSiCPy version: {BASICPY_VERSION}")
logger.info(f"Pydantic version: {PYDANTIC_VERSION}")
logger.info(f"Autotune available: {AUTOTUNE_AVAILABLE}")

__all__ = [
    "split_image_into_fovs",
    "reconstruct_image_from_fovs",
    "apply_basic_correction",
    "preprocess_multichannel_image",
]


def count_fovs(
    image_shape: Tuple[int, int],
    fov_size: Tuple[int, int],
    overlap: int = 0
) -> Tuple[int, int]:
    """Calculate how many FOVs are needed to cover an image with given FOV size and overlap."""
    height, width = image_shape[:2]
    fov_h, fov_w = fov_size

    if overlap >= fov_h or overlap >= fov_w:
        raise ValueError("Overlap cannot be >= FOV size")

    step_y = fov_h - overlap
    step_x = fov_w - overlap

    n_fovs_y = math.ceil((height - fov_h) / step_y) + 1 if height > fov_h else 1
    n_fovs_x = math.ceil((width - fov_w) / step_x) + 1 if width > fov_w else 1

    return n_fovs_y, n_fovs_x


def split_image_into_fovs(
    image: NDArray,
    n_fovs_x: int,
    n_fovs_y: int
) -> Tuple[NDArray, List[Tuple[int, int, int, int]], Tuple[int, int]]:
    """
    Split image (H, W) into FOV tiles with adaptive sizing to handle remainders.

    The image is divided into n_fovs_y * n_fovs_x tiles. Remainder pixels are
    distributed across tiles (some tiles get +1 pixel) to exactly cover the image
    without padding.
    """
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D, got shape {image.shape}")

    if n_fovs_x <= 0 or n_fovs_y <= 0:
        raise ValueError("Number of FOVs must be positive")

    height, width = image.shape

    # Calculate base FOV sizes and remainders
    base_w = width // n_fovs_x
    base_h = height // n_fovs_y
    remainder_x = width % n_fovs_x
    remainder_y = height % n_fovs_y

    # Calculate actual FOV sizes (some FOVs get +1 pixel to handle remainders)
    fov_widths = [base_w + (1 if j < remainder_x else 0) for j in range(n_fovs_x)]
    fov_heights = [base_h + (1 if i < remainder_y else 0) for i in range(n_fovs_y)]

    max_w = max(fov_widths)
    max_h = max(fov_heights)

    # Create FOV stack with padding to max dimensions
    n_fovs = n_fovs_y * n_fovs_x
    fov_stack = np.zeros((n_fovs, max_h, max_w), dtype=image.dtype)

    # Extract FOVs and store position info
    positions = []
    y_start = 0
    idx = 0

    for i in range(n_fovs_y):
        x_start = 0
        for j in range(n_fovs_x):
            h = fov_heights[i]
            w = fov_widths[j]

            fov_stack[idx, :h, :w] = image[y_start:y_start + h, x_start:x_start + w]

            positions.append((y_start, x_start, h, w))
            x_start += w
            idx += 1
        y_start += fov_heights[i]

    return fov_stack, positions, (max_h, max_w)


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

    n_fovs_y, n_fovs_x = count_fovs(image.shape, fov_size)
    fov_stack, positions, _ = split_image_into_fovs(image, n_fovs_x, n_fovs_y)

    basic = BaSiC(get_darkfield=get_darkfield, smoothness_flatfield=1)

    if autotune:
        logger.info(f"  [OK] Autotuning BaSiC parameters for {n_iter} iterations")
        basic.autotune(
            fov_stack,
            n_iter=n_iter
        )

    corrected_fovs = basic.fit_transform(fov_stack)

    # Checkpoint 2: Clip negative values from BaSiC darkfield subtraction
    # BaSiC can produce negative values when darkfield > original pixel value
    has_neg, neg_count, neg_pct = detect_negative_values(corrected_fovs, logger)
    if has_neg:
        logger.warning(
            f"BaSiC correction produced {neg_count} negative pixels ({neg_pct:.4f}%), "
            f"min value: {corrected_fovs.min()}"
        )
        corrected_fovs = np.clip(corrected_fovs, 0, None)
        logger.info(f"Clipped negative values to 0. New min: {corrected_fovs.min()}")

    reconstructed = reconstruct_image_from_fovs(
        corrected_fovs,
        positions,
        image.shape
    )

    return reconstructed, basic


def _process_single_channel_from_stack(
    channel_image: NDArray,
    channel_index: int,
    channel_name: str,
    fov_size: Tuple[int, int],
    skip_dapi: bool,
    autotune: bool,
    n_iter: int,
    basic_kwargs: dict,
    auto_detect: bool = True
) -> Tuple[int, NDArray, bool]:
    """Worker function to process a single channel slice from a stack.

    Returns
    -------
    channel_index : int
        Channel index
    processed_image : NDArray
        Corrected or original image
    was_corrected : bool
        Whether BaSiC was applied
    """
    logger.debug(f"Processing channel #{channel_index} ({channel_name})")

    # Automatic detection of whether to apply BaSiC
    if skip_dapi and 'DAPI' in channel_name.upper():
        logger.debug(f"  [SKIP] BaSiC correction for DAPI (user setting)")
        return channel_index, channel_image, False

    logger.debug(f"  [OK] Applying BaSiC correction")
    corrected, _ = apply_basic_correction(
        channel_image,
        fov_size=fov_size,
        autotune=autotune,
        n_iter=n_iter,
        #**basic_kwargs
    )
    return channel_index, corrected, True


def preprocess_multichannel_image(
    image_path: str,
    channel_names: List[str],
    output_path: str,
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

    # Read and log input file metadata
    with tifffile.TiffFile(image_path) as tif:
        has_ome = hasattr(tif, 'ome_metadata') and tif.ome_metadata
        has_physical_size = has_ome and 'PhysicalSizeX' in tif.ome_metadata
        if has_ome:
            logger.debug(f"  Input metadata: OME=True, PhysicalSize={has_physical_size}")
        else:
            logger.warning("  [WARN] Input file has no OME metadata")

        if tif.pages and len(tif.pages) > 0:
            first_page = tif.pages[0]
            logger.debug(f"  First page: shape={first_page.shape}, dtype={first_page.dtype}")

    multichannel_stack = tifffile.imread(image_path)
    logger.info(f"Loaded image shape: {multichannel_stack.shape}")

    # Checkpoint 1: Log input image statistics
    log_image_stats(multichannel_stack, "input", logger)

    if multichannel_stack.ndim == 2:
        logger.debug("  Converting 2D to 3D (adding channel dimension)")
        multichannel_stack = np.expand_dims(multichannel_stack, axis=0)
    elif multichannel_stack.ndim == 3 and multichannel_stack.shape[2] == len(channel_names):
        logger.debug("  Transposing from (Y, X, C) to (C, Y, X)")
        multichannel_stack = np.transpose(multichannel_stack, (2, 0, 1))

    n_channels, H, W = multichannel_stack.shape
    logger.info(f"Processing {n_channels} channels ({H}x{W}) with {n_workers} workers")

    if n_channels != len(channel_names):
        channel_names = channel_names[:n_channels] + [f"Channel_{i}" for i in range(len(channel_names), n_channels)]

    results = {}
    correction_applied = {}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_channels):
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

        for future in as_completed(futures):
            channel_index, result_array, was_corrected = future.result()
            results[channel_index] = result_array
            correction_applied[channel_index] = was_corrected

    preprocessed_channels = [
        results[i] for i in range(n_channels)
    ]

    # Log summary
    n_corrected = sum(correction_applied.values())
    n_skipped = n_channels - n_corrected
    logger.info(f"BaSiC Correction Summary: corrected={n_corrected}/{n_channels}, skipped={n_skipped}/{n_channels}")

    preprocessed = np.stack(preprocessed_channels, axis=0)

    # Log output statistics after all processing
    log_image_stats(preprocessed, "after_basic_correction", logger)

    logger.info(f"Saving corrected image to {output_path}")
    logger.debug(f"  Final stack shape: {preprocessed.shape}, channels: {channel_names[:preprocessed.shape[0]]}")

    # Save as OME-TIFF with proper metadata
    # VALIS expects OME-TIFF with proper channel dimension and physical size metadata
    # Default pixel size is 0.325 µm (matching convert_nd2.py)
    metadata = {
        'axes': 'CYX',
        'Channel': {'Name': channel_names[:preprocessed.shape[0]]},
        'PhysicalSizeX': 0.325,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': 0.325,
        'PhysicalSizeYUnit': 'µm'
    }

    logger.debug(f"  OME metadata: axes={metadata['axes']}, pixel_size=0.325µm")

    tifffile.imwrite(
        output_path,
        preprocessed,
        photometric='minisblack',
        metadata=metadata,
        bigtiff=True,
        ome=True,
        compression='zlib',
        tile=(2048, 2048)
    )

    logger.info(f"[OK] Saved OME-TIFF with {preprocessed.shape[0]} channels")

    # Verify the saved file
    with tifffile.TiffFile(output_path) as tif:
        has_ome = hasattr(tif, 'ome_metadata') and tif.ome_metadata
        if not has_ome:
            logger.warning("[WARN] No OME metadata in saved file")

    verify_img = tifffile.imread(output_path)
    if verify_img.ndim == 3:
        logger.debug(f"  Verification: shape={verify_img.shape}")

    # CHECKPOINT 1: Verify saved file has no negatives
    logger.info("[CHECKPOINT 1] Verifying saved preprocessed file...")
    verify_min = float(verify_img.min())
    verify_max = float(verify_img.max())
    if verify_min < 0:
        neg_count = int(np.sum(verify_img < 0))
        neg_pct = 100 * neg_count / verify_img.size
        logger.error(f"[CHECKPOINT 1 FAIL] Saved file has {neg_count} negatives ({neg_pct:.4f}%), min={verify_min}")
    else:
        logger.info(f"[CHECKPOINT 1 OK] No negatives in saved file. min={verify_min:.2f}, max={verify_max:.2f}")

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
        help='Path to the multichannel image file'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory'
    )

    parser.add_argument(
        '--channels',
        type=str,
        nargs='+',
        required=True,
        help='Channel names from metadata'
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
        '--overlap',
        type=int,
        default=0,
        help='Overlap between FOV tiles for BaSiC correction'
    )

    parser.add_argument(
        '--no_darkfield',
        action='store_true',
        help='Disable darkfield estimation in BaSiC'
    )

    return parser.parse_args()

def main():
    """Main entry point."""
    configure_logging(level=logging.INFO)

    args = parse_args()

    ensure_dir(args.output_dir)

    image_path = args.image
    image_basename = os.path.basename(image_path)

    channel_names = args.channels

    # Always save as .ome.tiff since we're writing OME-TIFF format
    if image_basename.endswith('.ome.tif'):
        base = image_basename[:-8]  # Remove .ome.tif
    elif image_basename.endswith('.ome.tiff'):
        base = image_basename[:-9]  # Remove .ome.tiff
    elif image_basename.endswith('.tif'):
        base = image_basename[:-4]  # Remove .tif
    elif image_basename.endswith('.tiff'):
        base = image_basename[:-5]  # Remove .tiff
    else:
        base = os.path.splitext(image_basename)[0]

    ext = '.ome.tif'  # Always use OME-TIFF extension
    output_filename = f"{base}_corrected{ext}"
    output_path = os.path.join(
        args.output_dir,
        output_filename
    )

    logger.info(f"Starting preprocessing: {image_path}")
    logger.info(f"Expected channel order: {channel_names}")

    # Build BaSiC kwargs
    basic_kwargs = {
        'overlap': args.overlap,
        'get_darkfield': not args.no_darkfield
    }

    preprocess_multichannel_image(
        image_path=image_path,
        channel_names=channel_names,
        output_path=output_path,
        fov_size=(args.fov_size, args.fov_size),
        skip_dapi=args.skip_dapi,
        autotune=args.autotune,
        n_iter=args.n_iter,
        n_workers=args.n_workers,
        **basic_kwargs
    )

    logger.info(f"Preprocessing completed successfully. Output: {output_path}")

    # Write dimensions to file for downstream processes
    img = tifffile.imread(output_path)
    shape = img.shape if img.ndim == 3 else (1, img.shape[0], img.shape[1])
    dims_filename = f"{base}_dims.txt"
    dims_path = os.path.join(args.output_dir, dims_filename)
    with open(dims_path, 'w') as f:
        f.write(f"{shape[0]} {shape[1]} {shape[2]}")
    logger.info(f"Image dimensions saved to: {dims_path} (C={shape[0]}, H={shape[1]}, W={shape[2]})")

    return 0


if __name__ == '__main__':
    exit(main())