#!/usr/bin/env python3
"""Process a single affine tile for tiled CPU registration.

This script reads a specific tile region from reference and moving images,
computes the affine transformation using ORB feature matching, applies it
to all channels, and saves the transformed tile.

Memory-efficient: Only loads one tile region at a time.

Usage:
    affine_tile.py --tile-id affine_0001 --tile-plan plan.json \
                   --reference ref.tiff --moving mov.tiff \
                   --output-prefix tile_affine_0001
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import tifffile

# Add utils directory to path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

from logger import get_logger, configure_logging

logger = get_logger(__name__)

__all__ = ["main"]


def compute_affine_mapping_cv2(
    y: np.ndarray,
    x: np.ndarray,
    n_features: int = 2000
) -> Tuple[Optional[np.ndarray], Dict]:
    """Compute affine transformation matrix using ORB feature matching.

    Parameters
    ----------
    y : np.ndarray
        Reference image (2D)
    x : np.ndarray
        Moving image (2D)
    n_features : int
        Number of ORB features to detect

    Returns
    -------
    matrix : np.ndarray or None
        2x3 affine transformation matrix, or None if failed
    info : dict
        Diagnostic information about the computation
    """
    # Normalize to uint8 for ORB
    y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create ORB detector
    orb = cv2.ORB_create(fastThreshold=0, edgeThreshold=0, nfeatures=n_features)

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(y, None)
    keypoints2, descriptors2 = orb.detectAndCompute(x, None)

    if descriptors1 is None or descriptors2 is None:
        fail_reason = []
        if descriptors1 is None:
            fail_reason.append(f"reference: {len(keypoints1) if keypoints1 else 0} keypoints but no descriptors")
        if descriptors2 is None:
            fail_reason.append(f"moving: {len(keypoints2) if keypoints2 else 0} keypoints but no descriptors")
        return None, {"status": "failed", "reason": "no_features", "detail": "; ".join(fail_reason)}

    # Ensure uint8 type for BFMatcher
    if descriptors1.dtype != np.uint8:
        descriptors1 = descriptors1.astype(np.uint8)
    if descriptors2.dtype != np.uint8:
        descriptors2 = descriptors2.astype(np.uint8)

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda m: m.distance)

    if len(matches) < 3:
        return None, {
            "status": "failed",
            "reason": "insufficient_matches",
            "detail": f"{len(matches)} matches < 3 required"
        }

    # Extract matched points
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate affine transformation
    matrix, mask = cv2.estimateAffinePartial2D(points2, points1)

    if matrix is None:
        return None, {
            "status": "failed",
            "reason": "ransac_failed",
            "detail": f"{len(matches)} matches but RANSAC rejected all"
        }

    inlier_count = np.sum(mask) if mask is not None else len(matches)
    return matrix, {
        "status": "success",
        "n_matches": len(matches),
        "n_inliers": int(inlier_count),
        "detail": f"{inlier_count}/{len(matches)} inliers"
    }


def apply_affine_cv2(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply affine transformation using OpenCV.

    Parameters
    ----------
    x : np.ndarray
        Input image (2D)
    matrix : np.ndarray
        2x3 affine transformation matrix

    Returns
    -------
    np.ndarray
        Transformed image
    """
    height, width = x.shape[:2]
    return cv2.warpAffine(x, matrix, (width, height), flags=cv2.INTER_LINEAR)


def process_affine_tile(
    tile_id: str,
    tile_plan: Dict[str, Any],
    reference_path: Path,
    moving_path: Path,
    n_features: int = 5000
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Process a single affine tile.

    Parameters
    ----------
    tile_id : str
        Tile identifier (e.g., "affine_0001")
    tile_plan : dict
        Tile plan loaded from JSON
    reference_path : Path
        Path to reference image
    moving_path : Path
        Path to moving image
    n_features : int
        Number of ORB features for affine matching

    Returns
    -------
    tile_data : np.ndarray
        Transformed tile data (C, H, W) or (H, W)
    tile_meta : dict
        Metadata about the tile processing
    """
    # Find tile coordinates
    tile_info = None
    for tile in tile_plan['affine_tiles']:
        if tile['tile_id'] == tile_id:
            tile_info = tile
            break

    if tile_info is None:
        raise ValueError(f"Tile {tile_id} not found in tile plan")

    y, x, h, w = tile_info['y'], tile_info['x'], tile_info['h'], tile_info['w']
    logger.info(f"Processing tile {tile_id} at ({y}, {x}) size ({h}, {w})")

    # Open images as memory maps
    ref_mem = tifffile.memmap(str(reference_path), mode='r')
    mov_mem = tifffile.memmap(str(moving_path), mode='r')

    # Extract reference crop (channel 0 for feature matching)
    if ref_mem.ndim == 3:
        ref_crop = ref_mem[0, y:y+h, x:x+w].astype(np.float32)
    else:
        ref_crop = ref_mem[y:y+h, x:x+w].astype(np.float32)

    # Extract moving crop (all channels)
    if mov_mem.ndim == 3:
        mov_crop = mov_mem[:, y:y+h, x:x+w].astype(np.float32)
        mov_crop_ch0 = mov_crop[0]
        n_channels = mov_mem.shape[0]
    else:
        mov_crop = mov_mem[y:y+h, x:x+w].astype(np.float32)
        mov_crop_ch0 = mov_crop
        n_channels = 1

    logger.debug(f"Reference crop shape: {ref_crop.shape}")
    logger.debug(f"Moving crop shape: {mov_crop.shape}")

    # Compute affine transformation
    matrix, info = compute_affine_mapping_cv2(
        np.ascontiguousarray(ref_crop),
        np.ascontiguousarray(mov_crop_ch0),
        n_features=n_features
    )

    # Prepare tile metadata
    tile_meta = {
        'tile_id': tile_id,
        'y': y,
        'x': x,
        'h': h,
        'w': w,
        'affine_info': info
    }

    if matrix is None:
        # Affine failed - return original crop (identity transform)
        logger.warning(f"Tile {tile_id}: affine failed ({info['reason']}), using identity")
        tile_meta['affine_matrix'] = None
        tile_meta['used_identity'] = True

        if mov_crop.ndim == 2:
            result = mov_crop[np.newaxis, ...]  # Add channel dim
        else:
            result = mov_crop
    else:
        # Apply affine to all channels
        logger.info(f"Tile {tile_id}: affine succeeded - {info['detail']}")
        tile_meta['affine_matrix'] = matrix.tolist()
        tile_meta['used_identity'] = False

        if mov_crop.ndim == 3:
            result = np.stack([
                apply_affine_cv2(np.ascontiguousarray(mov_crop[c]), matrix)
                for c in range(n_channels)
            ])
        else:
            result = apply_affine_cv2(np.ascontiguousarray(mov_crop), matrix)
            result = result[np.newaxis, ...]  # Add channel dim

    # Clean up
    del ref_crop, mov_crop, ref_mem, mov_mem
    gc.collect()

    return result, tile_meta


def main():
    parser = argparse.ArgumentParser(
        description="Process a single affine tile for tiled CPU registration"
    )
    parser.add_argument(
        "--tile-id", type=str, required=True,
        help="Tile identifier (e.g., affine_0001)"
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
        "--moving", type=str, required=True,
        help="Path to moving image"
    )
    parser.add_argument(
        "--output-prefix", type=str, required=True,
        help="Output prefix for tile files"
    )
    parser.add_argument(
        "--n-features", type=int, default=5000,
        help="Number of ORB features (default: 5000)"
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

        # Process tile
        tile_data, tile_meta = process_affine_tile(
            tile_id=args.tile_id,
            tile_plan=tile_plan,
            reference_path=Path(args.reference),
            moving_path=Path(args.moving),
            n_features=args.n_features
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
        logger.error(f"Failed to process affine tile: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
