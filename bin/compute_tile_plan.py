#!/usr/bin/env python3
"""Compute tile plan for tiled CPU registration.

This script reads image metadata (without loading image data) and generates
a JSON tile plan containing coordinates for both affine and diffeomorphic
registration stages.

Usage:
    compute_tile_plan.py --reference ref.tiff --moving mov.tiff --output plan.json

The output JSON contains:
    - Image dimensions, dtype, channel info
    - affine_tiles: List of tile coordinates for affine stage (larger tiles)
    - diffeo_tiles: List of tile coordinates for diffeo stage (smaller tiles)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tifffile

# Add utils directory to path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

from logger import get_logger, configure_logging
from registration_utils import extract_crop_coords

logger = get_logger(__name__)


def get_image_metadata(image_path: Path) -> Dict[str, Any]:
    """Extract image metadata without loading pixel data.

    Parameters
    ----------
    image_path : Path
        Path to TIFF image

    Returns
    -------
    dict
        Dictionary containing:
        - shape: tuple of (channels, height, width) or (height, width)
        - dtype: string representation of dtype
        - n_channels: number of channels
        - height: image height
        - width: image width
    """
    with tifffile.TiffFile(str(image_path)) as tif:
        # Get shape from first page or series
        if tif.series:
            shape = tif.series[0].shape
            dtype = str(tif.series[0].dtype)
        else:
            page = tif.pages[0]
            shape = page.shape
            dtype = str(page.dtype)

    # Normalize to (C, H, W) or (H, W)
    if len(shape) == 2:
        height, width = shape
        n_channels = 1
    elif len(shape) == 3:
        n_channels, height, width = shape
    else:
        raise ValueError(f"Unexpected image shape: {shape}")

    return {
        'shape': shape,
        'dtype': dtype,
        'n_channels': n_channels,
        'height': height,
        'width': width
    }


def compute_tile_plan(
    reference_path: Path,
    moving_path: Path,
    affine_crop_size: int,
    diffeo_crop_size: int,
    overlap_percent: float
) -> Dict[str, Any]:
    """Generate tile plan for two-pass tiled registration.

    Parameters
    ----------
    reference_path : Path
        Path to reference image
    moving_path : Path
        Path to moving image
    affine_crop_size : int
        Crop size for affine registration (typically 10000px)
    diffeo_crop_size : int
        Crop size for diffeomorphic registration (typically 2000px)
    overlap_percent : float
        Overlap between tiles as percentage (0-100)

    Returns
    -------
    dict
        Tile plan containing all necessary information for tiled registration
    """
    logger.info(f"Reading reference metadata: {reference_path}")
    ref_meta = get_image_metadata(reference_path)

    logger.info(f"Reading moving metadata: {moving_path}")
    mov_meta = get_image_metadata(moving_path)

    # Validate dimensions match
    if ref_meta['height'] != mov_meta['height'] or ref_meta['width'] != mov_meta['width']:
        raise ValueError(
            f"Spatial dimensions mismatch: "
            f"reference ({ref_meta['height']}, {ref_meta['width']}) vs "
            f"moving ({mov_meta['height']}, {mov_meta['width']})"
        )

    logger.info(f"Image dimensions: {mov_meta['height']} x {mov_meta['width']}, {mov_meta['n_channels']} channels")

    # Compute overlap in pixels
    affine_overlap = int(affine_crop_size * overlap_percent / 100.0)
    diffeo_overlap = int(diffeo_crop_size * overlap_percent / 100.0)

    # Use moving image shape for crop coordinates
    mov_shape = mov_meta['shape']

    # Generate affine tile coordinates
    affine_coords = extract_crop_coords(mov_shape, affine_crop_size, affine_overlap)
    logger.info(f"Affine stage: {len(affine_coords)} tiles (size={affine_crop_size}, overlap={affine_overlap})")

    # Generate diffeo tile coordinates
    diffeo_coords = extract_crop_coords(mov_shape, diffeo_crop_size, diffeo_overlap)
    logger.info(f"Diffeo stage: {len(diffeo_coords)} tiles (size={diffeo_crop_size}, overlap={diffeo_overlap})")

    # Add tile_id to each coordinate dict
    affine_tiles = [
        {**coord, 'tile_id': f"affine_{i:04d}"}
        for i, coord in enumerate(affine_coords)
    ]

    diffeo_tiles = [
        {**coord, 'tile_id': f"diffeo_{i:04d}"}
        for i, coord in enumerate(diffeo_coords)
    ]

    # Build the tile plan
    tile_plan = {
        'version': '1.0',
        'reference': {
            'path': str(reference_path.name),
            'shape': list(ref_meta['shape']),
            'dtype': ref_meta['dtype'],
            'n_channels': ref_meta['n_channels'],
            'height': ref_meta['height'],
            'width': ref_meta['width']
        },
        'moving': {
            'path': str(moving_path.name),
            'shape': list(mov_meta['shape']),
            'dtype': mov_meta['dtype'],
            'n_channels': mov_meta['n_channels'],
            'height': mov_meta['height'],
            'width': mov_meta['width']
        },
        'affine_config': {
            'crop_size': affine_crop_size,
            'overlap': affine_overlap,
            'overlap_percent': overlap_percent,
            'n_tiles': len(affine_tiles)
        },
        'diffeo_config': {
            'crop_size': diffeo_crop_size,
            'overlap': diffeo_overlap,
            'overlap_percent': overlap_percent,
            'n_tiles': len(diffeo_tiles)
        },
        'affine_tiles': affine_tiles,
        'diffeo_tiles': diffeo_tiles
    }

    return tile_plan


def main():
    parser = argparse.ArgumentParser(
        description="Compute tile plan for tiled CPU registration"
    )
    parser.add_argument(
        "--reference", type=str, required=True,
        help="Path to reference image"
    )
    parser.add_argument(
        "--moving", type=str, required=True,
        help="Path to moving image"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output JSON tile plan"
    )
    parser.add_argument(
        "--affine-crop-size", type=int, default=10000,
        help="Crop size for affine registration (default: 10000)"
    )
    parser.add_argument(
        "--diffeo-crop-size", type=int, default=2000,
        help="Crop size for diffeomorphic registration (default: 2000)"
    )
    parser.add_argument(
        "--overlap-percent", type=float, default=40.0,
        help="Overlap between tiles as percentage (default: 40.0)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(level=getattr(logging, args.log_level.upper()))

    try:
        # Compute tile plan
        tile_plan = compute_tile_plan(
            reference_path=Path(args.reference),
            moving_path=Path(args.moving),
            affine_crop_size=args.affine_crop_size,
            diffeo_crop_size=args.diffeo_crop_size,
            overlap_percent=args.overlap_percent
        )

        # Write output
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(tile_plan, f, indent=2)

        logger.info(f"Tile plan written to: {output_path}")
        logger.info(f"  Affine tiles: {tile_plan['affine_config']['n_tiles']}")
        logger.info(f"  Diffeo tiles: {tile_plan['diffeo_config']['n_tiles']}")

        return 0

    except Exception as e:
        logger.error(f"Failed to compute tile plan: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
