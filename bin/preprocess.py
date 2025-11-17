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
from pathlib import Path
from typing import Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tifffile
from numpy.typing import NDArray

os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Force CPU for JAX
from basicpy import BaSiC  # type: ignore

from _common import ensure_dir

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
        raise ValueError(f"Image must be 2D, got shape {image.shape}")

    image_h, image_w = image.shape
    fov_h, fov_w = fov_size

    if overlap >= min(fov_h, fov_w):
        raise ValueError(f"Overlap ({overlap}) must be < FOV size {fov_size}")

    stride_h = fov_h - overlap
    stride_w = fov_w - overlap

    n_fovs_h = int(np.ceil((image_h - overlap) / stride_h))
    n_fovs_w = int(np.ceil((image_w - overlap) / stride_w))

    n_fovs = n_fovs_h * n_fovs_w

    fov_stack = np.zeros((n_fovs, fov_h, fov_w), dtype=image.dtype)
    positions = []
    idx = 0

    for i in range(n_fovs_h):
        row_start = i * stride_h
        row_end = min(row_start + fov_h, image_h)
        actual_h = row_end - row_start

        for j in range(n_fovs_w):
            col_start = j * stride_w
            col_end = min(col_start + fov_w, image_w)
            actual_w = col_end - col_start

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

    fov_stack, positions, _ = split_image_into_fovs(
        image, fov_size, overlap=0
    )

    basic = BaSiC(get_darkfield=get_darkfield, **basic_kwargs)

    if autotune:
        basic.autotune(fov_stack, early_stop=True, n_iter=n_iter)

    corrected_fovs = basic.fit_transform(fov_stack)

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
    basic_kwargs: dict
) -> Tuple[int, NDArray]:
    """Worker function to process a single channel slice from a stack."""
    logger.info(f"Processing channel #{channel_index} ({channel_name})")

    if skip_dapi and 'DAPI' in channel_name.upper():
        logger.info(f"  Skipping BaSiC correction for DAPI")
        return channel_index, channel_image

    corrected, _ = apply_basic_correction(
        channel_image,
        fov_size=fov_size,
        autotune=autotune,
        n_iter=n_iter,
        **basic_kwargs
    )
    return channel_index, corrected


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
    multichannel_stack = tifffile.imread(image_path)

    if multichannel_stack.ndim == 2:
        multichannel_stack = np.expand_dims(multichannel_stack, axis=0)
    elif multichannel_stack.ndim == 3 and multichannel_stack.shape[2] == len(channel_names):
        multichannel_stack = np.transpose(multichannel_stack, (2, 0, 1))

    n_channels, H, W = multichannel_stack.shape
    logger.info(f"Processing {n_channels} channels ({H}x{W}) with {n_workers} workers")

    if n_channels != len(channel_names):
        channel_names = channel_names[:n_channels] + [f"Channel_{i}" for i in range(len(channel_names), n_channels)]

    results = {}

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
            channel_index, result_array = future.result()
            results[channel_index] = result_array

    preprocessed_channels = [
        results[i] for i in range(n_channels)
    ]

    preprocessed = np.stack(preprocessed_channels, axis=0)

    logger.info(f"Saving corrected image to {output_path}")
    tifffile.imwrite(
        output_path,
        preprocessed,
        imagej=True,
        photometric='minisblack'
    )

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

    return parser.parse_args()

def find_channel_names_from_path(image_path: str) -> List[str]:
    """
    Extracts channel names from a multichannel image path assuming the format:
    <ID>_<CHANNEL1>_<CHANNEL2>...<EXT>
    e.g., 'id1_DAPI_Marker1_Marker2.ome.tiff' -> ['DAPI', 'Marker1', 'Marker2']
    """
    p = Path(image_path)
    base_name = p.stem

    parts = base_name.split('_')

    if len(parts) < 2:
        return ["DAPI", "Channel_1", "Channel_2", "Channel_3", "Channel_4"]

    channel_names = parts[1:]

    if not channel_names:
        return ["DAPI", "Channel_1", "Channel_2", "Channel_3", "Channel_4"]

    return channel_names


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    ensure_dir(args.output_dir)

    image_path = args.image
    image_basename = os.path.basename(image_path)

    channel_names = find_channel_names_from_path(image_path)

    base, ext = os.path.splitext(image_basename)

    if base.endswith(".ome"):
        base = base[:-4]
        ext = ".ome" + ext

    output_filename = f"{base}_corrected{ext}"
    output_path = os.path.join(
        args.output_dir,
        output_filename
    )

    logger.info(f"Starting preprocessing: {image_path}")
    logger.info(f"Expected channel order: {channel_names}")

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

    logger.info(f"Preprocessing completed successfully. Output: {output_path}")
    return 0


if __name__ == '__main__':
    exit(main())