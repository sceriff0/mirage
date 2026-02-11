#!/usr/bin/env python3
"""Stitch tiles into a complete image for tiled CPU registration.

This script assembles processed tiles (from either affine or diffeo stage)
into a complete image using hard-cutoff placement strategy.

Supports two modes:
1. Affine stitch: Creates intermediate TIFF (no OME metadata)
2. Diffeo stitch: Creates final OME-TIFF with proper metadata

Memory-efficient: Streams tiles one at a time using memmap output.

Usage:
    stitch_tiles.py --tile-plan plan.json --tiles-dir ./tiles \
                    --stage affine --output output.tiff
    stitch_tiles.py --tile-plan plan.json --tiles-dir ./tiles \
                    --stage diffeo --output output.ome.tiff --moving orig.tiff
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tifffile

# Add utils directory to path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

from logger import get_logger, configure_logging
from registration_utils import calculate_bounds
from metadata import create_ome_xml, extract_channel_names_from_ome, extract_channel_names_from_filename

logger = get_logger(__name__)

__all__ = ["main"]


def find_tile_files(tiles_dir: Path, stage: str) -> List[Path]:
    """Find all tile .npy files for a given stage.

    Parameters
    ----------
    tiles_dir : Path
        Directory containing tile files
    stage : str
        Stage name ('affine' or 'diffeo')

    Returns
    -------
    list
        Sorted list of tile file paths
    """
    pattern = f"{stage}_*.npy"
    tile_files = sorted(tiles_dir.glob(pattern))

    # Filter out metadata files
    tile_files = [f for f in tile_files if not f.name.endswith('_meta.json.npy')]

    return tile_files


def load_tile_metadata(tile_path: Path) -> Optional[Dict[str, Any]]:
    """Load metadata JSON for a tile.

    Parameters
    ----------
    tile_path : Path
        Path to tile .npy file

    Returns
    -------
    dict or None
        Tile metadata, or None if not found
    """
    meta_path = tile_path.with_suffix('.npy').parent / f"{tile_path.stem}_meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None


def stitch_tiles(
    tile_plan: Dict[str, Any],
    tiles_dir: Path,
    output_path: Path,
    stage: str,
    moving_path: Optional[Path] = None
) -> None:
    """Stitch tiles into a complete image.

    Parameters
    ----------
    tile_plan : dict
        Tile plan loaded from JSON
    tiles_dir : Path
        Directory containing tile .npy files
    output_path : Path
        Path to output image
    stage : str
        'affine' or 'diffeo'
    moving_path : Path, optional
        Path to original moving image (for extracting channel names in diffeo stage)
    """
    # Get configuration for this stage
    if stage == 'affine':
        config = tile_plan['affine_config']
        tiles_info = tile_plan['affine_tiles']
    elif stage == 'diffeo':
        config = tile_plan['diffeo_config']
        tiles_info = tile_plan['diffeo_tiles']
    else:
        raise ValueError(f"Unknown stage: {stage}")

    overlap = config['overlap']
    n_tiles_expected = config['n_tiles']

    # Get image dimensions
    height = tile_plan['moving']['height']
    width = tile_plan['moving']['width']

    # Find tile files
    tile_files = find_tile_files(tiles_dir, stage)
    logger.info(f"Found {len(tile_files)} tile files (expected {n_tiles_expected})")

    if len(tile_files) == 0:
        raise RuntimeError(f"No tile files found in {tiles_dir} for stage {stage}")

    # Load first tile to get dtype and number of channels
    first_tile = np.load(tile_files[0])
    if first_tile.ndim == 3:
        n_channels = first_tile.shape[0]
    else:
        n_channels = 1
        first_tile = first_tile[np.newaxis, ...]

    dtype = first_tile.dtype
    logger.info(f"Output shape: ({n_channels}, {height}, {width}), dtype: {dtype}")

    # Create output array using memmap for memory efficiency
    output_shape = (n_channels, height, width)

    # For large images, use a temporary memmap file
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"stitch_{stage}_"))
    memmap_path = tmp_dir / "output.npy"

    output_mem = np.memmap(
        str(memmap_path),
        dtype=np.float32,
        mode='w+',
        shape=output_shape
    )
    output_mem[:] = 0

    # Build a mapping from tile_id to tile_info
    tile_info_map = {t['tile_id']: t for t in tiles_info}

    # Process each tile
    tiles_processed = 0
    for tile_path in tile_files:
        # Extract tile_id from filename (e.g., "affine_0001.npy" -> "affine_0001")
        tile_id = tile_path.stem

        if tile_id not in tile_info_map:
            logger.warning(f"Tile {tile_id} not in plan, skipping")
            continue

        tile_info = tile_info_map[tile_id]
        y, x, h, w = tile_info['y'], tile_info['x'], tile_info['h'], tile_info['w']

        # Load tile data
        tile_data = np.load(tile_path)
        if tile_data.ndim == 2:
            tile_data = tile_data[np.newaxis, ...]

        # Calculate bounds for hard-cutoff placement
        y_start, y_end, y_slice = calculate_bounds(y, h, height, overlap)
        x_start, x_end, x_slice = calculate_bounds(x, w, width, overlap)

        # Place tile in output
        for c in range(n_channels):
            tile_channel = tile_data[c] if c < tile_data.shape[0] else tile_data[0]
            crop_trimmed = tile_channel[y_slice, x_slice]
            output_mem[c, y_start:y_end, x_start:x_end] = crop_trimmed.astype(np.float32)

        tiles_processed += 1
        if tiles_processed % 50 == 0:
            logger.info(f"Processed {tiles_processed}/{len(tile_files)} tiles")

        del tile_data
        gc.collect()

    logger.info(f"Processed {tiles_processed} tiles total")

    # Flush memmap
    output_mem.flush()

    # Convert to output dtype
    orig_dtype_str = tile_plan['moving']['dtype']
    if 'uint16' in orig_dtype_str:
        output_data = np.clip(output_mem, 0, 65535).astype(np.uint16)
    elif 'uint8' in orig_dtype_str:
        output_data = np.clip(output_mem, 0, 255).astype(np.uint8)
    else:
        output_data = output_mem.astype(np.float32)

    # Save output
    if stage == 'diffeo':
        # Final output - create OME-TIFF with proper metadata
        channel_names = None

        # Try to extract channel names from original moving image
        if moving_path and moving_path.exists():
            channel_names = extract_channel_names_from_ome(moving_path)
            if not channel_names or len(channel_names) != n_channels:
                channel_names = extract_channel_names_from_filename(moving_path, expected_channels=n_channels)

        if not channel_names or len(channel_names) != n_channels:
            channel_names = [f"Channel_{i}" for i in range(n_channels)]

        logger.info(f"Channel names: {channel_names}")

        # Create OME-XML
        ome_xml = create_ome_xml(channel_names, output_data.dtype, width, height)

        # Save as OME-TIFF
        tifffile.imwrite(
            str(output_path),
            output_data,
            metadata={'axes': 'CYX'},
            description=ome_xml,
            compression="zlib",
            bigtiff=True
        )
    else:
        # Intermediate affine output - simple TIFF (uncompressed for memmap compatibility)
        tifffile.imwrite(
            str(output_path),
            output_data,
            metadata={'axes': 'CYX'},
            bigtiff=True
        )

    logger.info(f"Saved stitched image: {output_path}")

    # Cleanup
    del output_mem, output_data
    gc.collect()

    # Remove temporary memmap file
    try:
        import shutil
        shutil.rmtree(tmp_dir)
    except Exception as e:
        logger.warning(f"Could not clean up temp dir {tmp_dir}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Stitch tiles into a complete image for tiled CPU registration"
    )
    parser.add_argument(
        "--tile-plan", type=str, required=True,
        help="Path to tile plan JSON"
    )
    parser.add_argument(
        "--tiles-dir", type=str, required=True,
        help="Directory containing tile .npy files"
    )
    parser.add_argument(
        "--stage", type=str, required=True, choices=['affine', 'diffeo'],
        help="Stage: 'affine' or 'diffeo'"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output image"
    )
    parser.add_argument(
        "--moving", type=str, default=None,
        help="Path to original moving image (for channel names, diffeo stage only)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(level=getattr(logging, args.log_level.upper()))

    try:
        # Load tile plan
        with open(args.tile_plan, 'r') as f:
            tile_plan = json.load(f)

        # Stitch tiles
        stitch_tiles(
            tile_plan=tile_plan,
            tiles_dir=Path(args.tiles_dir),
            output_path=Path(args.output),
            stage=args.stage,
            moving_path=Path(args.moving) if args.moving else None
        )

        return 0

    except Exception as e:
        logger.error(f"Failed to stitch tiles: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
