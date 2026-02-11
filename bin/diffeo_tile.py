#!/usr/bin/env python3
"""Process a single diffeomorphic tile for tiled CPU registration.

This script reads a specific tile region from reference and affine-transformed
images, computes the diffeomorphic deformation using DIPY, applies it to all
channels, and saves the registered tile.

Memory-efficient: Only loads one tile region at a time.

Usage:
    diffeo_tile.py --tile-id diffeo_0001 --tile-plan plan.json \
                   --reference ref.tiff --affine affine.tiff \
                   --output-prefix tile_diffeo_0001
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tifffile

# Add utils directory to path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

from logger import get_logger, configure_logging

logger = get_logger(__name__)

# CPU imports with availability check
try:
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.metrics import CCMetric
    DIPY_AVAILABLE = True
except Exception as e:
    DIPY_AVAILABLE = False
    _dipy_import_error = str(e)


def compute_diffeomorphic_mapping_dipy(
    y: np.ndarray,
    x: np.ndarray,
    sigma_diff: int = 20,
    radius: int = 20,
    opt_tol: float = 1e-5,
    inv_tol: float = 1e-5
):
    """Compute diffeomorphic mapping using DIPY (CPU-based).

    Parameters
    ----------
    y : np.ndarray
        Reference image (2D)
    x : np.ndarray
        Moving image (2D)
    sigma_diff : int
        Sigma for cross-correlation metric
    radius : int
        Radius for cross-correlation metric
    opt_tol : float
        Optimization tolerance
    inv_tol : float
        Inverse tolerance

    Returns
    -------
    mapping
        DIPY diffeomorphic mapping object
    """
    if y.shape != x.shape:
        raise ValueError("Reference and moving images must have the same shape.")

    # Input validation
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Reference image contains NaN or Inf values")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Moving image contains NaN or Inf values")

    # Check intensity ranges
    y_min, y_max = np.percentile(y, [0.1, 99.9])
    x_min, x_max = np.percentile(x, [0.1, 99.9])

    if y_max <= y_min:
        raise ValueError(f"Reference crop has uniform intensity (min={y_min:.2f}, max={y_max:.2f})")
    if x_max <= x_min:
        raise ValueError(f"Moving crop has uniform intensity (min={x_min:.2f}, max={x_max:.2f})")

    # Ensure contiguous float64 arrays
    y_cont = np.ascontiguousarray(y, dtype=np.float64)
    x_cont = np.ascontiguousarray(x, dtype=np.float64)

    # Scale parameters based on crop size
    crop_size = y.shape[0]
    scale_factor = crop_size / 2000
    radius = int(20 * scale_factor)
    sigma_diff = int(20 * np.sqrt(scale_factor))

    # Create metric and optimizer
    metric = CCMetric(2, sigma_diff=sigma_diff, radius=radius)
    sdr = SymmetricDiffeomorphicRegistration(metric, opt_tol=opt_tol, inv_tol=inv_tol)

    # Compute mapping
    mapping = sdr.optimize(y_cont, x_cont)

    return mapping


def apply_diffeo_mapping(mapping, x: np.ndarray) -> np.ndarray:
    """Apply diffeomorphic mapping using DIPY.

    Parameters
    ----------
    mapping
        DIPY diffeomorphic mapping object
    x : np.ndarray
        Input image (2D)

    Returns
    -------
    np.ndarray
        Transformed image
    """
    return mapping.transform(x)


def process_diffeo_tile(
    tile_id: str,
    tile_plan: Dict[str, Any],
    reference_path: Path,
    affine_path: Path,
    opt_tol: float = 1e-5,
    inv_tol: float = 1e-5
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Process a single diffeomorphic tile.

    Parameters
    ----------
    tile_id : str
        Tile identifier (e.g., "diffeo_0001")
    tile_plan : dict
        Tile plan loaded from JSON
    reference_path : Path
        Path to reference image
    affine_path : Path
        Path to affine-transformed intermediate image
    opt_tol : float
        Optimization tolerance for DIPY
    inv_tol : float
        Inverse tolerance for DIPY

    Returns
    -------
    tile_data : np.ndarray
        Registered tile data (C, H, W)
    tile_meta : dict
        Metadata about the tile processing
    """
    if not DIPY_AVAILABLE:
        raise RuntimeError(f"DIPY not available: {_dipy_import_error}")

    # Find tile coordinates
    tile_info = None
    for tile in tile_plan['diffeo_tiles']:
        if tile['tile_id'] == tile_id:
            tile_info = tile
            break

    if tile_info is None:
        raise ValueError(f"Tile {tile_id} not found in tile plan")

    y, x, h, w = tile_info['y'], tile_info['x'], tile_info['h'], tile_info['w']
    logger.info(f"Processing tile {tile_id} at ({y}, {x}) size ({h}, {w})")

    # Open images as memory maps
    ref_mem = tifffile.memmap(str(reference_path), mode='r')
    affine_mem = tifffile.memmap(str(affine_path), mode='r')

    # Extract reference crop (channel 0 for feature matching)
    if ref_mem.ndim == 3:
        ref_crop = ref_mem[0, y:y+h, x:x+w].astype(np.float32)
    else:
        ref_crop = ref_mem[y:y+h, x:x+w].astype(np.float32)

    # Extract affine-transformed crop (all channels)
    if affine_mem.ndim == 3:
        affine_crop = affine_mem[:, y:y+h, x:x+w].astype(np.float32)
        affine_crop_ch0 = affine_crop[0]
        n_channels = affine_mem.shape[0]
    else:
        affine_crop = affine_mem[y:y+h, x:x+w].astype(np.float32)
        affine_crop_ch0 = affine_crop
        n_channels = 1

    logger.debug(f"Reference crop shape: {ref_crop.shape}")
    logger.debug(f"Affine crop shape: {affine_crop.shape}")

    # Prepare tile metadata
    tile_meta = {
        'tile_id': tile_id,
        'y': y,
        'x': x,
        'h': h,
        'w': w
    }

    # Check for uniform intensity (skip diffeo)
    crop_min, crop_max = affine_crop_ch0.min(), affine_crop_ch0.max()
    if crop_max - crop_min < 1e-6:
        logger.warning(f"Tile {tile_id}: uniform intensity, using affine-only")
        tile_meta['diffeo_status'] = 'skipped_uniform'
        tile_meta['used_affine_only'] = True

        if affine_crop.ndim == 2:
            result = affine_crop[np.newaxis, ...]
        else:
            result = affine_crop

        del ref_crop, affine_crop
        # Explicitly close memory maps before deletion to prevent SIGBUS on BeeGFS
        for mmap_arr in [ref_mem, affine_mem]:
            if hasattr(mmap_arr, '_mmap') and mmap_arr._mmap is not None:
                mmap_arr._mmap.close()
        del ref_mem, affine_mem
        gc.collect()
        return result, tile_meta

    # Compute diffeomorphic mapping
    try:
        ref_crop_cont = np.ascontiguousarray(ref_crop)
        affine_crop_ch0_cont = np.ascontiguousarray(affine_crop_ch0)

        mapping = compute_diffeomorphic_mapping_dipy(
            ref_crop_cont, affine_crop_ch0_cont,
            opt_tol=opt_tol, inv_tol=inv_tol
        )

        # Apply mapping to all channels
        if affine_crop.ndim == 3:
            result = np.stack([
                apply_diffeo_mapping(mapping, np.ascontiguousarray(affine_crop[c]))
                for c in range(n_channels)
            ])
        else:
            result = apply_diffeo_mapping(mapping, np.ascontiguousarray(affine_crop))
            result = result[np.newaxis, ...]

        tile_meta['diffeo_status'] = 'success'
        tile_meta['used_affine_only'] = False
        logger.info(f"Tile {tile_id}: diffeo succeeded")

        del mapping

    except Exception as e:
        # Diffeo failed - use affine result
        logger.warning(f"Tile {tile_id}: diffeo failed ({e}), using affine-only")
        tile_meta['diffeo_status'] = 'failed'
        tile_meta['diffeo_error'] = str(e)
        tile_meta['used_affine_only'] = True

        if affine_crop.ndim == 2:
            result = affine_crop[np.newaxis, ...]
        else:
            result = affine_crop

    # Clean up
    del ref_crop, affine_crop
    # Explicitly close memory maps before deletion to prevent SIGBUS on BeeGFS
    for mmap_arr in [ref_mem, affine_mem]:
        if hasattr(mmap_arr, '_mmap') and mmap_arr._mmap is not None:
            mmap_arr._mmap.close()
    del ref_mem, affine_mem
    gc.collect()

    return result, tile_meta


def main():
    parser = argparse.ArgumentParser(
        description="Process a single diffeomorphic tile for tiled CPU registration"
    )
    parser.add_argument(
        "--tile-id", type=str, required=True,
        help="Tile identifier (e.g., diffeo_0001)"
    )
    parser.add_argument(
        "--tile-plan", type=str, required=True,
        help="Path to tile plan JSON"
    )
    parser.add_argument(
        "--reference", type=str, required=True,
        help="Path to reference image"
    )
    parser.add_argument(
        "--affine", type=str, required=True,
        help="Path to affine-transformed intermediate image"
    )
    parser.add_argument(
        "--output-prefix", type=str, required=True,
        help="Output prefix for tile files"
    )
    parser.add_argument(
        "--opt-tol", type=float, default=1e-5,
        help="Optimization tolerance (default: 1e-5)"
    )
    parser.add_argument(
        "--inv-tol", type=float, default=1e-5,
        help="Inverse tolerance (default: 1e-5)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(level=getattr(logging, args.log_level.upper()))

    # Check DIPY availability
    if not DIPY_AVAILABLE:
        logger.error(f"DIPY not available: {_dipy_import_error}")
        return 1

    try:
        # Load tile plan
        with open(args.tile_plan, 'r') as f:
            tile_plan = json.load(f)

        # Process tile
        tile_data, tile_meta = process_diffeo_tile(
            tile_id=args.tile_id,
            tile_plan=tile_plan,
            reference_path=Path(args.reference),
            affine_path=Path(args.affine),
            opt_tol=args.opt_tol,
            inv_tol=args.inv_tol
        )

        # Save tile data
        tile_path = Path(f"{args.output_prefix}.npy")
        np.save(tile_path, tile_data)
        logger.info(f"Saved tile data: {tile_path} (shape={tile_data.shape})")

        # Save tile metadata
        meta_path = Path(f"{args.output_prefix}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(tile_meta, f, indent=2)
        logger.info(f"Saved tile metadata: {meta_path}")

        return 0

    except Exception as e:
        logger.error(f"Failed to process diffeo tile: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
