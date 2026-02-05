"""FOV Tiling Utilities for Pixie Clustering.

This module provides functions to split large images into 2048x2048 FOV tiles
for efficient processing with Pixie's multiprocessing capabilities.

The tiling approach allows Pixie to process multiple FOVs in parallel using
its internal multiprocessing (multiprocess=True, batch_size=N).

Examples
--------
>>> from utils.tiling import create_fov_directory_structure
>>> tiff_dir, seg_dir, fovs, tile_info = create_fov_directory_structure(
...     channel_tiffs=[Path('CD3.tiff'), Path('CD4.tiff')],
...     cell_mask=Path('cell_mask.tif'),
...     output_dir=Path('tiled_fovs'),
...     tile_size=2048,
...     patient_id='sample001'
... )
>>> print(fovs)  # ['sample001_tile_0_0', 'sample001_tile_0_1', ...]
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
from numpy.typing import NDArray

__all__ = [
    "TileInfo",
    "calculate_tile_grid",
    "split_image_to_fov_tiles",
    "create_fov_directory_structure",
    "needs_tiling",
    "save_tile_positions",
    "load_tile_positions",
]


@dataclass
class TileInfo:
    """Information about a single tile's position and dimensions."""
    row: int
    col: int
    y_start: int
    x_start: int
    height: int
    width: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TileInfo":
        """Create TileInfo from dictionary."""
        return cls(**d)


def calculate_tile_grid(
    height: int,
    width: int,
    tile_size: int = 2048
) -> Tuple[int, int]:
    """Calculate the number of tiles needed to cover an image.

    Parameters
    ----------
    height : int
        Image height in pixels.
    width : int
        Image width in pixels.
    tile_size : int
        Target tile size (default 2048).

    Returns
    -------
    Tuple[int, int]
        (n_rows, n_cols) - number of tiles in each dimension.
    """
    n_rows = math.ceil(height / tile_size)
    n_cols = math.ceil(width / tile_size)
    return n_rows, n_cols


def needs_tiling(height: int, width: int, tile_size: int = 2048) -> bool:
    """Check if image needs tiling (larger than tile_size in any dimension).

    Parameters
    ----------
    height : int
        Image height in pixels.
    width : int
        Image width in pixels.
    tile_size : int
        Target tile size (default 2048).

    Returns
    -------
    bool
        True if image needs tiling, False if it can be processed as single FOV.
    """
    return height > tile_size or width > tile_size


def split_image_to_fov_tiles(
    image: NDArray,
    tile_size: int = 2048,
    patient_id: str = "sample"
) -> Dict[str, Tuple[NDArray, TileInfo]]:
    """Split a 2D image into tiles with unique FOV names.

    Edge tiles may be smaller than tile_size if the image dimensions
    are not evenly divisible.

    Parameters
    ----------
    image : NDArray
        2D numpy array (H, W) to split.
    tile_size : int
        Target tile size (default 2048).
    patient_id : str
        Patient/sample ID used in FOV naming.

    Returns
    -------
    Dict[str, Tuple[NDArray, TileInfo]]
        Dictionary mapping FOV names to (tile_array, tile_info) tuples.
        FOV names follow pattern: {patient_id}_tile_{row}_{col}

    Raises
    ------
    ValueError
        If image is not 2D.
    """
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D, got shape {image.shape}")

    height, width = image.shape
    n_rows, n_cols = calculate_tile_grid(height, width, tile_size)

    result = {}

    for row in range(n_rows):
        for col in range(n_cols):
            y_start = row * tile_size
            x_start = col * tile_size

            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)

            tile = image[y_start:y_end, x_start:x_end].copy()

            fov_name = f"{patient_id}_tile_{row}_{col}"
            tile_info = TileInfo(
                row=row,
                col=col,
                y_start=y_start,
                x_start=x_start,
                height=y_end - y_start,
                width=x_end - x_start
            )

            result[fov_name] = (tile, tile_info)

    return result


def create_fov_directory_structure(
    channel_tiffs: List[Path],
    cell_mask: Path,
    output_dir: Path,
    tile_size: int = 2048,
    patient_id: str = "sample"
) -> Tuple[Path, Path, List[str], Dict[str, TileInfo]]:
    """Create Pixie-compatible FOV directory structure from full images.

    This function reads full-resolution channel images and cell mask,
    splits them into tiles, and creates the directory structure expected
    by Pixie's pixel clustering functions.

    Directory structure created:
    ```
    output_dir/
        tiff_dir/
            {patient_id}_tile_0_0/
                channel1.tif
                channel2.tif
                ...
            {patient_id}_tile_0_1/
                ...
        seg_dir/
            {patient_id}_tile_0_0_cell_mask.tif
            {patient_id}_tile_0_1_cell_mask.tif
            ...
    ```

    Parameters
    ----------
    channel_tiffs : List[Path]
        List of paths to single-channel TIFF files.
    cell_mask : Path
        Path to cell segmentation mask.
    output_dir : Path
        Base output directory for tiled structure.
    tile_size : int
        Target tile size (default 2048).
    patient_id : str
        Patient/sample ID used in FOV naming.

    Returns
    -------
    Tuple[Path, Path, List[str], Dict[str, TileInfo]]
        (tiff_dir, seg_dir, fov_names, tile_positions)
        - tiff_dir: Path to directory containing tiled channel images
        - seg_dir: Path to directory containing tiled masks
        - fov_names: List of FOV names for all tiles
        - tile_positions: Dict mapping FOV names to TileInfo

    Raises
    ------
    ValueError
        If channel images have different dimensions.
    """
    output_dir = Path(output_dir)
    tiff_dir = output_dir / "tiff_dir"
    seg_dir = output_dir / "seg_dir"

    os.makedirs(tiff_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    # Read reference image to get dimensions
    with tifffile.TiffFile(channel_tiffs[0]) as tif:
        ref_shape = tif.pages[0].shape

    height, width = ref_shape[:2]

    # Check if tiling is needed
    if not needs_tiling(height, width, tile_size):
        # No tiling needed - create single FOV structure
        fov_name = patient_id
        fov_dir = tiff_dir / fov_name
        os.makedirs(fov_dir, exist_ok=True)

        # Symlink channel files
        for tiff_path in channel_tiffs:
            channel_name = Path(tiff_path).stem
            dest = fov_dir / f"{channel_name}.tif"
            if dest.exists():
                dest.unlink()
            os.symlink(os.path.abspath(tiff_path), dest)

        # Symlink mask
        mask_dest = seg_dir / f"{fov_name}_cell_mask.tif"
        if mask_dest.exists():
            mask_dest.unlink()
        os.symlink(os.path.abspath(cell_mask), mask_dest)

        tile_info = TileInfo(
            row=0, col=0,
            y_start=0, x_start=0,
            height=height, width=width
        )

        return tiff_dir, seg_dir, [fov_name], {fov_name: tile_info}

    # Tiling needed - split all channels and mask
    n_rows, n_cols = calculate_tile_grid(height, width, tile_size)

    # Read and tile the mask first
    mask_image = tifffile.imread(cell_mask)
    if mask_image.ndim > 2:
        mask_image = mask_image[0]  # Take first channel/slice if multichannel

    mask_tiles = split_image_to_fov_tiles(mask_image, tile_size, patient_id)

    # Create FOV directories and save mask tiles
    fov_names = list(mask_tiles.keys())
    tile_positions = {}

    for fov_name, (mask_tile, tile_info) in mask_tiles.items():
        # Create FOV directory for channels
        fov_dir = tiff_dir / fov_name
        os.makedirs(fov_dir, exist_ok=True)

        # Save mask tile
        mask_path = seg_dir / f"{fov_name}_cell_mask.tif"
        tifffile.imwrite(mask_path, mask_tile)

        tile_positions[fov_name] = tile_info

    # Process each channel
    for tiff_path in channel_tiffs:
        channel_name = Path(tiff_path).stem
        channel_image = tifffile.imread(tiff_path)

        if channel_image.ndim > 2:
            channel_image = channel_image[0]  # Take first channel/slice

        # Verify dimensions match
        if channel_image.shape[:2] != (height, width):
            raise ValueError(
                f"Channel {channel_name} has shape {channel_image.shape}, "
                f"expected ({height}, {width})"
            )

        # Split channel into tiles
        channel_tiles = split_image_to_fov_tiles(channel_image, tile_size, patient_id)

        # Save each tile
        for fov_name, (tile, _) in channel_tiles.items():
            fov_dir = tiff_dir / fov_name
            tile_path = fov_dir / f"{channel_name}.tif"
            tifffile.imwrite(tile_path, tile)

    return tiff_dir, seg_dir, fov_names, tile_positions


def save_tile_positions(
    tile_positions: Dict[str, TileInfo],
    output_path: Path,
    patient_id: str,
    tile_size: int,
    original_height: int,
    original_width: int
) -> None:
    """Save tile position information to JSON file.

    Parameters
    ----------
    tile_positions : Dict[str, TileInfo]
        Dictionary mapping FOV names to TileInfo.
    output_path : Path
        Path for output JSON file.
    patient_id : str
        Original patient/sample ID.
    tile_size : int
        Tile size used for splitting.
    original_height : int
        Original image height.
    original_width : int
        Original image width.
    """
    data = {
        "patient_id": patient_id,
        "tile_size": tile_size,
        "original_height": original_height,
        "original_width": original_width,
        "n_tiles": len(tile_positions),
        "tiles": {name: info.to_dict() for name, info in tile_positions.items()}
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_tile_positions(input_path: Path) -> Tuple[Dict[str, TileInfo], dict]:
    """Load tile position information from JSON file.

    Parameters
    ----------
    input_path : Path
        Path to tile positions JSON file.

    Returns
    -------
    Tuple[Dict[str, TileInfo], dict]
        (tile_positions, metadata)
        - tile_positions: Dict mapping FOV names to TileInfo
        - metadata: Dict with patient_id, tile_size, original dimensions
    """
    with open(input_path) as f:
        data = json.load(f)

    tile_positions = {
        name: TileInfo.from_dict(info)
        for name, info in data["tiles"].items()
    }

    metadata = {
        "patient_id": data["patient_id"],
        "tile_size": data["tile_size"],
        "original_height": data["original_height"],
        "original_width": data["original_width"],
        "n_tiles": data["n_tiles"]
    }

    return tile_positions, metadata
