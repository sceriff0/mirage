#!/usr/bin/env python3
"""BaSiC Illumination Correction Preprocessing.

This module provides utilities to apply BaSiC shading correction to large
multichannel images by tiling into FOVs and reconstructing the corrected image.
"""

from __future__ import annotations

import os
import time
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Optional

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

# Local imports (guarded fallbacks for utils helpers)
try:
    from utils.io import save_h5
except Exception:
    # Minimal HDF5 saver used when utils.io isn't available in test envs
    try:
        import h5py

        def save_h5(arr, path):
            with h5py.File(path, 'w') as f:
                f.create_dataset('data', data=arr)
    except Exception:
        def save_h5(arr, path):
            # Last-resort: write a numpy .npy file if h5py missing
            np.save(path.replace('.h5', '.npy'), arr)

try:
    from utils import logging_config
except Exception:
    import logging as _logging

    class _FallbackLoggingConfig:
        @staticmethod
        def setup_logging():
            _logging.basicConfig(level=_logging.INFO)

    logging_config = _FallbackLoggingConfig()

from _common import ensure_dir, setup_file_logger

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
    Split image into field-of-view (FOV) tiles for BaSiC processing.

    Parameters
    ----------
    image : ndarray, shape (H, W) or (H, W, C)
        Input image to split.
    fov_size : tuple of int
        FOV dimensions as (height, width).
    overlap : int, optional
        Overlap between adjacent FOVs. Default is 0.

    Returns
    -------
    fov_stack : ndarray, shape (N, fov_h, fov_w) or (N, fov_h, fov_w, C)
        Stack of FOV tiles.
    positions : list of tuple
        FOV positions as (row_start, col_start, height, width).
    max_fov_size : tuple of int
        Maximum FOV size used for padding.

    Notes
    -----
    FOVs are padded to uniform size. Use positions to reconstruct original image.
    """
    if image.ndim < 2:
        raise ValueError(f"Image must be at least 2D, got {image.ndim}D")

    image_h, image_w = image.shape[:2]
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

    logger.info(
        f"Splitting {image.shape} image into {n_fovs} FOVs "
        f"({n_fovs_h}×{n_fovs_w}) of size {fov_size}"
    )

    # Prepare output array
    if image.ndim == 2:
        fov_stack = np.zeros((n_fovs, fov_h, fov_w), dtype=image.dtype)
    else:
        fov_stack = np.zeros((n_fovs, fov_h, fov_w, image.shape[2]), dtype=image.dtype)

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
            if image.ndim == 2:
                fov_stack[idx, :actual_h, :actual_w] = \
                    image[row_start:row_end, col_start:col_end]
            else:
                fov_stack[idx, :actual_h, :actual_w, :] = \
                    image[row_start:row_end, col_start:col_end, :]

            positions.append((row_start, col_start, actual_h, actual_w))
            idx += 1

    return fov_stack, positions, (fov_h, fov_w)


def reconstruct_image_from_fovs(
    fov_stack: NDArray,
    positions: List[Tuple[int, int, int, int]],
    original_shape: Tuple[int, ...]
) -> NDArray:
    """
    Reconstruct image from FOV tiles.

    Parameters
    ----------
    fov_stack : ndarray, shape (N, fov_h, fov_w) or (N, fov_h, fov_w, C)
        Stack of FOV tiles.
    positions : list of tuple
        FOV positions as (row_start, col_start, height, width).
    original_shape : tuple of int
        Shape of the reconstructed image.

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image with original_shape.
    """
    reconstructed = np.zeros(original_shape, dtype=fov_stack.dtype)

    for idx, (row_start, col_start, h, w) in enumerate(positions):
        if reconstructed.ndim == 2:
            reconstructed[row_start:row_start + h, col_start:col_start + w] = \
                fov_stack[idx, :h, :w]
        else:
            reconstructed[row_start:row_start + h, col_start:col_start + w, :] = \
                fov_stack[idx, :h, :w, :]

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
    Apply BaSiC illumination correction to image.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Input image (single channel).
    fov_size : tuple of int, optional
        FOV size for tiling. Default is (1950, 1950).
    get_darkfield : bool, optional
        Whether to estimate darkfield. Default is True.
    autotune : bool, optional
        Whether to autotune BaSiC parameters. Default is False.
    n_iter : int, optional
        Number of autotuning iterations. Default is 3.
    **basic_kwargs
        Additional arguments passed to BaSiC constructor.

    Returns
    -------
    corrected : ndarray
        Illumination-corrected image.
    basic_model : BaSiC
        Fitted BaSiC model.

    Notes
    -----
    The image is split into FOVs for processing, then reconstructed.
    This is the correct way to apply BaSiC to large images.
    """
    logger.info(f"Applying BaSiC correction to {image.shape} image")
    start_time = time.time()

    # Split into FOVs
    fov_stack, positions, max_fov_size = split_image_into_fovs(
        image, fov_size, overlap=0
    )

    # Initialize BaSiC model
    basic = BaSiC(get_darkfield=get_darkfield, **basic_kwargs)

    logger.info(
        f"BaSiC parameters: smoothness_flatfield={basic.smoothness_flatfield}, "
        f"smoothness_darkfield={basic.smoothness_darkfield}, "
        f"sparse_cost_darkfield={basic.sparse_cost_darkfield}"
    )

    # Autotune if requested
    if autotune:
        logger.info(f"Autotuning BaSiC parameters ({n_iter} iterations)...")
        autotune_start = time.time()

        basic.autotune(fov_stack, early_stop=True, n_iter=n_iter)

        autotune_time = time.time() - autotune_start
        logger.info(f"Autotuning completed in {autotune_time:.2f}s")
        logger.info(
            f"Tuned parameters: smoothness_flatfield={basic.smoothness_flatfield}, "
            f"smoothness_darkfield={basic.smoothness_darkfield}, "
            f"sparse_cost_darkfield={basic.sparse_cost_darkfield}"
        )

    # Fit and transform
    logger.info("Fitting and transforming FOVs...")
    transform_start = time.time()

    corrected_fovs = basic.fit_transform(fov_stack)

    transform_time = time.time() - transform_start
    logger.info(f"Transformation completed in {transform_time:.2f}s")

    # Reconstruct image (FIXED: use corrected_fovs, not fov_stack!)
    reconstructed = reconstruct_image_from_fovs(
        corrected_fovs,  # CRITICAL FIX: was using fov_stack before
        positions,
        image.shape
    )

    total_time = time.time() - start_time
    logger.info(f"BaSiC correction completed in {total_time:.2f}s")

    return reconstructed, basic


def preprocess_multichannel_image(
    channels: List[str],
    output_path: str,
    fov_size: Tuple[int, int] = (1950, 1950),
    skip_dapi: bool = True,
    autotune: bool = False,
    n_iter: int = 3,
    **basic_kwargs
) -> NDArray:
    """
    Apply BaSiC preprocessing to multichannel image.

    Parameters
    ----------
    channels : list of str
        List of paths to single-channel TIFF files.
    output_path : str
        Path to save preprocessed H5 file.
    fov_size : tuple of int, optional
        FOV size for BaSiC tiling. Default is (1950, 1950).
    skip_dapi : bool, optional
        Skip BaSiC correction for DAPI channel. Default is True.
    autotune : bool, optional
        Autotune BaSiC parameters. Default is False.
    n_iter : int, optional
        Number of autotuning iterations. Default is 3.
    **basic_kwargs
        Additional BaSiC parameters.

    Returns
    -------
    preprocessed : ndarray, shape (C, H, W)
        Preprocessed multichannel image.

    Notes
    -----
    The output is saved as HDF5 in CYX format.
    """
    logger.info(f"Preprocessing {len(channels)} channels")

    preprocessed_channels = []

    for idx, channel_path in enumerate(channels):
        channel_name = Path(channel_path).stem.split('_')[-1]
        logger.info(f"[{idx + 1}/{len(channels)}] Processing channel: {channel_name}")

        # Load channel using tifffile (consistent, fast TIFF IO)
        load_start = time.time()
        channel_image = tifffile.imread(channel_path)
        load_time = time.time() - load_start
        logger.info(f"  Loaded in {load_time:.2f}s - shape: {channel_image.shape}")

        # Skip DAPI if requested
        if skip_dapi and 'DAPI' in channel_name.upper():
            logger.info(f"  Skipping BaSiC correction for DAPI")
            preprocessed_channels.append(channel_image)
            continue

        # Apply BaSiC correction
        try:
            corrected, _ = apply_basic_correction(
                channel_image,
                fov_size=fov_size,
                autotune=autotune,
                n_iter=n_iter,
                **basic_kwargs
            )
            preprocessed_channels.append(corrected)

        except Exception as e:
            logger.error(f"  BaSiC correction failed for {channel_name}: {e}")
            logger.warning(f"  Using uncorrected channel")
            preprocessed_channels.append(channel_image)

    # Stack channels (CYX format)
    preprocessed = np.stack(preprocessed_channels, axis=0)
    logger.info(f"Stacked shape: {preprocessed.shape}")

    # Save as HDF5
    logger.info(f"Saving to {output_path}")
    save_h5(preprocessed, output_path)

    return preprocessed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Apply BaSiC illumination correction to multichannel images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--channels',
        type=str,
        nargs='+',
        required=True,
        help='Paths to single-channel TIFF files'
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Original image path (for naming)'
    )

    parser.add_argument(
        '--patient_id',
        type=str,
        required=True,
        help='Patient ID'
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


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging.
    if args.log_file:
        handler = logging.FileHandler(args.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Create output directory.
    ensure_dir(args.output_dir)
    if args.log_file:
        setup_file_logger(args.log_file)

    # Determine output path
    image_basename = os.path.basename(args.image)
    output_path = os.path.join(
        args.output_dir,
        f"preprocessed_{image_basename}"
    )

    # Handle different extensions.
    for ext in ['.nd2', '.tiff', '.tif']:
        if output_path.endswith(ext):
            output_path = output_path.replace(ext, '.h5')
            break

    if not output_path.endswith('.h5'):
        output_path += '.h5'

    # Sort channels by expected order.
    channel_order = image_basename.split('.')[0].split('_')[1:][::-1]
    sorted_channels = []

    for ch_name in channel_order:
        matching = [f for f in args.channels if ch_name in f]
        if matching:
            sorted_channels.append(matching[0])

    if not sorted_channels:
        sorted_channels = args.channels

    logger.info(f"Channel order: {channel_order}")
    logger.info(f"Sorted files: {[Path(f).name for f in sorted_channels]}")

    # Process.
    try:
        preprocess_multichannel_image(
            channels=sorted_channels,
            output_path=output_path,
            fov_size=(args.fov_size, args.fov_size),
            skip_dapi=args.skip_dapi,
            autotune=args.autotune,
            n_iter=args.n_iter
        )

        logger.info(f"Preprocessing completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
