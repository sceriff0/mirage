#!/usr/bin/env python3
"""Multi-resolution CPU-based image registration with coarse-to-fine strategy.

Three-stage registration pipeline for maximum accuracy:
- Stage 1 (Coarse Affine): Large crops, fewer features → rough global alignment
- Stage 2 (Fine Affine): Medium crops, many features → precise global alignment
- Stage 3 (Diffeomorphic): Optimal crops → local deformations

Memory-aware design:
- Uses memmap for large intermediate results
- Multi-threaded processing for better CPU utilization
- Aggressively frees memory after each stage

Usage: Enhanced CLI with multi-resolution parameters.
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tifffile
import tempfile
import gc
import traceback
from skimage.transform import rescale

# CPU imports with availability check
try:
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.metrics import CCMetric
    DIPY_AVAILABLE = True
except Exception as e:
    DIPY_AVAILABLE = False
    _dipy_import_error = str(e)

logger = logging.getLogger(__name__)


def print_dipy_diagnostics():
    """Print DIPY version and system information."""
    logger.info("=" * 80)
    logger.info("MULTI-RESOLUTION CPU REGISTRATION DIAGNOSTICS")
    logger.info("=" * 80)

    if not DIPY_AVAILABLE:
        logger.error(f"✗ DIPY not available: {_dipy_import_error}")
        logger.info("=" * 80)
        return

    try:
        import dipy
        logger.info(f"DIPY version: {dipy.__version__}")
    except Exception as e:
        logger.warning(f"Could not get DIPY version: {e}")

    import multiprocessing
    logger.info(f"CPU cores available: {multiprocessing.cpu_count()}")
    logger.info(f"Registration mode: CPU multi-threaded multi-resolution")
    logger.info(f"Pipeline: Coarse Affine → Fine Affine → Diffeomorphic")

    logger.info("=" * 80)


# ------------------------- Helper Functions ---------------------------------

def get_channel_names(filename: str) -> List[str]:
    base = os.path.basename(filename)
    # Remove all suffixes that might be present
    name_part = base.replace('_corrected', '').replace('_padded', '').replace('_preprocessed', '').replace('_registered', '').split('.')[0]
    parts = name_part.split('_')
    channels = parts[1:]
    return channels


def autoscale(img: np.ndarray) -> np.ndarray:
    """Normalize image to 0-255 using min-max scaling."""
    lo = img.min()
    hi = img.max()
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)


def create_qc_rgb_composite(reference_path: Path, registered_path: Path, output_path: Path) -> None:
    """Create QC RGB composite with registered (blue) and reference (green) DAPI channels.

    Uses double normalization:
    1. Normalize full-resolution images independently
    2. Downsample by 0.25 scale factor
    3. Normalize again after downsampling to ensure full dynamic range
    """
    logger.info(f"Creating QC composite: {output_path.name}")

    # Load images
    ref_img = tifffile.imread(str(reference_path))
    reg_img = tifffile.imread(str(registered_path))

    if ref_img.ndim == 2:
        ref_img = ref_img[np.newaxis, ...]
    if reg_img.ndim == 2:
        reg_img = reg_img[np.newaxis, ...]

    # Find DAPI channels
    ref_channels = get_channel_names(reference_path.name)
    reg_channels = get_channel_names(registered_path.name)

    ref_dapi_idx = next((i for i, ch in enumerate(ref_channels) if "DAPI" in ch.upper()), 0)
    reg_dapi_idx = next((i for i, ch in enumerate(reg_channels) if "DAPI" in ch.upper()), 0)

    ref_dapi = ref_img[ref_dapi_idx]
    reg_dapi = reg_img[reg_dapi_idx]

    # Stack images along channel axis for vectorized normalization
    dapi_stack = np.stack([reg_dapi, ref_dapi], axis=0)

    # First normalization: normalize each channel independently at full resolution
    min_val = dapi_stack.min(axis=(1, 2), keepdims=True)
    max_val = dapi_stack.max(axis=(1, 2), keepdims=True)
    dapi_stack = (dapi_stack - min_val) / np.maximum(max_val - min_val, 1e-6) * 255
    dapi_stack = dapi_stack.astype(np.uint8)

    # Downsample each channel separately
    downsampled = np.array([
        rescale(dapi_stack[0], scale=0.25, anti_aliasing=True, preserve_range=True),
        rescale(dapi_stack[1], scale=0.25, anti_aliasing=True, preserve_range=True)
    ])

    del dapi_stack
    gc.collect()

    # Second normalization: normalize again after downsampling
    min_val = downsampled.min(axis=(1, 2), keepdims=True)
    max_val = downsampled.max(axis=(1, 2), keepdims=True)
    downsampled = (downsampled - min_val) / np.maximum(max_val - min_val, 1e-6) * 255
    downsampled = downsampled.astype(np.uint8)

    reg_down = downsampled[0]
    ref_down = downsampled[1]

    # Create RGB composite: Red = registered, Green = reference
    h, w = reg_down.shape
    rgb_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_bgr[:, :, 2] = 0         # Blue channel
    rgb_bgr[:, :, 1] = ref_down  # Green channel
    rgb_bgr[:, :, 0] = reg_down  # Red channel

    # Save as PNG (OpenCV uses BGR order)
    png_output_path = output_path.with_suffix('.png')
    cv2.imwrite(str(png_output_path), rgb_bgr)
    logger.info(f"  Saved QC PNG: {png_output_path}")

    # Save as ImageJ-compatible TIFF (CYX order)
    rgb_stack = np.stack([
        reg_down,   # Red channel (registered)
        ref_down,   # Green channel (reference)
        np.zeros_like(ref_down, dtype=np.uint8)  # Blue channel
    ], axis=0)

    tiff_output_path = output_path.with_suffix('.tif')
    tifffile.imwrite(
        str(tiff_output_path),
        rgb_stack,
        imagej=True,
        metadata={'axes': 'CYX', 'mode': 'composite'}
    )
    logger.info(f"  Saved QC TIFF: {tiff_output_path}")


def apply_affine_cv2(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply affine transformation using OpenCV."""
    height, width = x.shape[:2]
    return cv2.warpAffine(x, matrix, (width, height), flags=cv2.INTER_LINEAR)


def apply_diffeo_mapping(mapping, x: np.ndarray) -> np.ndarray:
    """Apply diffeomorphic mapping using DIPY."""
    return mapping.transform(x)


def compute_affine_mapping_cv2(y: np.ndarray, x: np.ndarray, n_features: int = 2000) -> Tuple[Optional[np.ndarray], Dict]:
    """Compute affine transformation matrix using ORB feature matching."""
    y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    orb = cv2.ORB_create(fastThreshold=0, edgeThreshold=0, nfeatures=n_features)
    keypoints1, descriptors1 = orb.detectAndCompute(y, None)
    keypoints2, descriptors2 = orb.detectAndCompute(x, None)

    if descriptors1 is None or descriptors2 is None:
        fail_reason = []
        if descriptors1 is None:
            fail_reason.append(f"reference: {len(keypoints1) if keypoints1 else 0} keypoints but no descriptors")
        if descriptors2 is None:
            fail_reason.append(f"moving: {len(keypoints2) if keypoints2 else 0} keypoints but no descriptors")
        return None, {"reason": "no_features", "detail": "; ".join(fail_reason)}

    if descriptors1.dtype != np.uint8:
        descriptors1 = descriptors1.astype(np.uint8)
    if descriptors2.dtype != np.uint8:
        descriptors2 = descriptors2.astype(np.uint8)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda m: m.distance)

    if len(matches) < 3:
        return None, {
            "reason": "insufficient_matches",
            "detail": f"{len(matches)} matches < 3 required"
        }

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.estimateAffinePartial2D(points2, points1)

    if matrix is None:
        return None, {
            "reason": "ransac_failed",
            "detail": f"{len(matches)} matches but RANSAC rejected all"
        }

    inlier_count = np.sum(mask) if mask is not None else len(matches)
    return matrix, {"reason": "success", "detail": f"{inlier_count}/{len(matches)} inliers"}


def compute_diffeomorphic_mapping_dipy(y: np.ndarray, x: np.ndarray,
                                        sigma_diff: int = 20, radius: int = 20,
                                        opt_tol: float = 1e-5, inv_tol: float = 1e-5):
    """Compute diffeomorphic mapping using DIPY (CPU-based)."""
    if y.shape != x.shape:
        raise ValueError("Reference and moving images must have the same shape.")

    # Thorough input validation
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

    # Ensure contiguous arrays (DIPY works with NumPy arrays directly)
    y_cont = np.ascontiguousarray(y, dtype=np.float64)
    x_cont = np.ascontiguousarray(x, dtype=np.float64)

    crop_size = y.shape[0]
    scale_factor = crop_size / 2000
    radius = int(radius * scale_factor)
    sigma_diff = int(sigma_diff * np.sqrt(scale_factor))

    metric = CCMetric(2, sigma_diff=sigma_diff, radius=radius)
    sdr = SymmetricDiffeomorphicRegistration(metric, opt_tol=opt_tol, inv_tol=inv_tol)

    mapping = sdr.optimize(y_cont, x_cont)

    return mapping


def extract_crop_coords(image_shape: Tuple[int, ...], crop_size: int, overlap: int) -> List[Dict]:
    """Return list of coordinate dicts (y, x, h, w) for tiling the image."""
    if len(image_shape) == 3:
        _, height, width = image_shape
    else:
        height, width = image_shape

    # Validate overlap to prevent infinite loop
    if overlap >= crop_size:
        raise ValueError(f"Overlap ({overlap}) must be less than crop_size ({crop_size})")

    # Warn if crop size is larger than image
    if crop_size > height or crop_size > width:
        logger.warning(f"crop_size ({crop_size}) is larger than image dimensions ({width}x{height}). Using full image as single crop.")

    stride = crop_size - overlap
    coords = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + crop_size, height)
            x_end = min(x + crop_size, width)
            y_start = max(0, y_end - crop_size)
            x_start = max(0, x_end - crop_size)
            h = y_end - y_start
            w = x_end - x_start
            coords.append({"y": y_start, "x": x_start, "h": h, "w": w})
    return coords


def calculate_bounds(start: int, crop_dim: int, total_dim: int, overlap: int):
    """Calculate bounds for hard-cutoff crop placement."""
    if start == 0:
        start_idx = start
        end_idx = start + crop_dim - overlap // 2
        crop_slice = slice(0, crop_dim - overlap // 2)
    elif start == total_dim - crop_dim:
        start_idx = start + overlap // 2
        end_idx = start + crop_dim
        crop_slice = slice(overlap // 2, crop_dim)
    else:
        start_idx = start + overlap // 2
        end_idx = start + crop_dim - overlap // 2
        crop_slice = slice(overlap // 2, crop_dim - overlap // 2)
    return start_idx, end_idx, crop_slice


def create_memmaps_for_merge(output_shape: Tuple[int, ...],
                              dtype: np.dtype = np.float32,
                              prefix: str = "reg_merge_") -> Tuple[np.memmap, np.memmap, Path]:
    """Create memory-mapped arrays for accumulating merged results."""
    tmp_dir = Path(tempfile.mkdtemp(prefix=prefix))

    merged_path = tmp_dir / "merged.npy"
    weights_path = tmp_dir / "weights.npy"

    merged = np.memmap(str(merged_path), dtype=dtype, mode="w+", shape=output_shape)
    weights = np.memmap(str(weights_path), dtype=dtype, mode="w+", shape=output_shape)

    merged[:] = 0
    weights[:] = 0

    return merged, weights, tmp_dir


def cleanup_memmaps(tmp_dir: Path):
    """Clean up temporary memmap files."""
    import shutil
    try:
        shutil.rmtree(tmp_dir)
    except Exception as e:
        logger.warning(f"Could not clean up temp directory {tmp_dir}: {e}")


# --------------------- Affine Registration Stage (Generalized) ---------------------

def process_affine_crop(crop_idx: int, ref_mem: np.ndarray, mov_mem: np.ndarray,
                        coord: Dict, n_features: int) -> Dict:
    """Compute affine matrix for a single crop."""
    y, x, h, w = coord["y"], coord["x"], coord["h"], coord["w"]

    try:
        # Read reference channel 0
        if ref_mem.ndim == 3:
            ref_crop = ref_mem[0, y:y+h, x:x+w]
        else:
            ref_crop = ref_mem[y:y+h, x:x+w]

        # Read moving channel 0
        if mov_mem.ndim == 3:
            mov_crop = mov_mem[0, y:y+h, x:x+w]
        else:
            mov_crop = mov_mem[y:y+h, x:x+w]

        ref_2d = np.ascontiguousarray(ref_crop.astype(np.float32))
        mov_2d = np.ascontiguousarray(mov_crop.astype(np.float32))

        matrix, diag = compute_affine_mapping_cv2(ref_2d, mov_2d, n_features=n_features)

        if matrix is None:
            return {
                "status": "failed",
                "error": diag["reason"],
                "error_detail": diag["detail"],
                "crop_idx": crop_idx,
                "coord": coord,
                "matrix": None
            }

        return {
            "status": "success",
            "matrix": matrix,
            "crop_idx": crop_idx,
            "coord": coord,
            "match_info": diag["detail"]
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": "exception",
            "error_detail": str(e),
            "crop_idx": crop_idx,
            "coord": coord,
            "matrix": None
        }


def run_affine_stage(ref_mem: np.ndarray, mov_mem: np.ndarray,
                     mov_shape: Tuple[int, ...],
                     crop_size: int, overlap: int,
                     n_features: int, n_workers: int,
                     stage_name: str = "AFFINE") -> Tuple[np.memmap, Path, int, int]:
    """
    Run CPU affine stage: compute matrices and merge into affine-transformed image.

    Returns:
        affine_mem: memmap of affine-transformed moving image
        affine_tmp_dir: temp directory containing memmaps
        success_count: number of successful crops
        total_crops: total number of crops
    """
    logger.info("=" * 80)
    logger.info(f"STAGE: {stage_name}")
    logger.info("=" * 80)

    # Extract crop coordinates
    affine_coords = extract_crop_coords(mov_shape, crop_size, overlap)
    logger.info(f"{stage_name}: {len(affine_coords)} crops (size={crop_size}, overlap={overlap}, features={n_features})")

    # Compute affine matrices in parallel
    logger.info("Computing affine matrices...")
    cpu_results = [None] * len(affine_coords)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_affine_crop, i, ref_mem, mov_mem, affine_coords[i], n_features): i
            for i in range(len(affine_coords))
        }

        for future in as_completed(futures):
            res = future.result()
            idx = res["crop_idx"]
            cpu_results[idx] = res
            coord = res["coord"]

            if res["status"] == "success":
                logger.debug(f"  {stage_name} crop {idx+1}/{len(affine_coords)} at ({coord['y']},{coord['x']}): ✓")
            else:
                logger.warning(f"  {stage_name} crop {idx+1}/{len(affine_coords)} at ({coord['y']},{coord['x']}): ✗ {res.get('error', 'unknown')}")

    # Summarize results
    successful = [r for r in cpu_results if r["status"] == "success"]
    failed = [r for r in cpu_results if r["status"] != "success"]

    logger.info(f"{stage_name}: {len(successful)}/{len(affine_coords)} crops succeeded ({100*len(successful)/len(affine_coords):.1f}%)")

    if failed:
        failure_reasons = {}
        for r in failed:
            reason = r.get('error', 'unknown')
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            logger.warning(f"  - {reason}: {count} crops")

    if len(successful) == 0:
        raise RuntimeError(f"All {stage_name} crops failed. Cannot proceed.")

    # Create memmap for affine-transformed image
    affine_mem, affine_weights, affine_tmp_dir = create_memmaps_for_merge(mov_shape, np.float32, f"{stage_name.lower()}_")
    logger.info(f"Created {stage_name} memmap in {affine_tmp_dir}")

    # Apply affine transformations and merge using hard-cutoff
    logger.info("Applying affine transformations and merging...")

    for res in cpu_results:
        coord = res["coord"]
        y, x, h, w = coord["y"], coord["x"], coord["h"], coord["w"]

        # Calculate bounds using same logic as cropping.reconstruct_image
        if len(mov_shape) == 3:
            img_height, img_width = mov_shape[1], mov_shape[2]
        else:
            img_height, img_width = mov_shape[0], mov_shape[1]
        y_start, y_end, y_slice = calculate_bounds(y, h, img_height, overlap)
        x_start, x_end, x_slice = calculate_bounds(x, w, img_width, overlap)

        if res["status"] != "success":
            # For failed crops, use original (identity transform)
            if mov_mem.ndim == 3:
                for c in range(mov_mem.shape[0]):
                    crop = mov_mem[c, y:y+h, x:x+w].astype(np.float32)
                    crop_trimmed = crop[y_slice, x_slice]
                    affine_mem[c, y_start:y_end, x_start:x_end] = crop_trimmed
            else:
                crop = mov_mem[y:y+h, x:x+w].astype(np.float32)
                crop_trimmed = crop[y_slice, x_slice]
                affine_mem[y_start:y_end, x_start:x_end] = crop_trimmed
        else:
            # Apply affine transformation
            matrix = res["matrix"]
            if mov_mem.ndim == 3:
                for c in range(mov_mem.shape[0]):
                    crop = mov_mem[c, y:y+h, x:x+w].astype(np.float32)
                    transformed = apply_affine_cv2(np.ascontiguousarray(crop), matrix)
                    transformed_trimmed = transformed[y_slice, x_slice]
                    affine_mem[c, y_start:y_end, x_start:x_end] = transformed_trimmed
                    del crop, transformed
            else:
                crop = mov_mem[y:y+h, x:x+w].astype(np.float32)
                transformed = apply_affine_cv2(np.ascontiguousarray(crop), matrix)
                transformed_trimmed = transformed[y_slice, x_slice]
                affine_mem[y_start:y_end, x_start:x_end] = transformed_trimmed
                del crop, transformed

        gc.collect()

    # Flush to disk
    affine_mem.flush()

    # Clean up weights memmap
    del affine_weights
    gc.collect()

    logger.info(f"{stage_name} stage complete")
    return affine_mem, affine_tmp_dir, len(successful), len(affine_coords)


# --------------------- CPU Stage: Multi-threaded Diffeomorphic Registration ---------------------

def process_diffeo_crop(crop_idx: int, ref_mem: np.ndarray, affine_mem: np.memmap,
                        coord: Dict, sigma_diff: int, radius: int,
                        opt_tol: float, inv_tol: float) -> Dict:
    """Compute diffeomorphic mapping for a single crop (runs in worker thread)."""
    y, x, h, w = coord["y"], coord["x"], coord["h"], coord["w"]

    try:
        # Read reference crop (channel 0 for metric computation)
        if ref_mem.ndim == 3:
            ref_crop = ref_mem[0, y:y+h, x:x+w].astype(np.float32)
        else:
            ref_crop = ref_mem[y:y+h, x:x+w].astype(np.float32)

        # Read affine-transformed moving crop (all channels)
        if affine_mem.ndim == 3:
            affine_crop = affine_mem[:, y:y+h, x:x+w].astype(np.float32)
            affine_crop_ch0 = affine_crop[0]
        else:
            affine_crop = affine_mem[y:y+h, x:x+w].astype(np.float32)
            affine_crop_ch0 = affine_crop

        # Check for valid intensity range
        crop_min, crop_max = affine_crop_ch0.min(), affine_crop_ch0.max()
        if crop_max - crop_min < 1e-6:
            return {
                "status": "uniform",
                "crop_idx": crop_idx,
                "coord": coord,
                "affine_crop": affine_crop
            }

        # Compute diffeomorphic mapping
        ref_crop_cont = np.ascontiguousarray(ref_crop)
        affine_crop_ch0_cont = np.ascontiguousarray(affine_crop_ch0)

        mapping = compute_diffeomorphic_mapping_dipy(
            ref_crop_cont, affine_crop_ch0_cont,
            sigma_diff=sigma_diff, radius=radius,
            opt_tol=opt_tol, inv_tol=inv_tol
        )

        # Apply mapping to all channels
        registered_crop = []
        if affine_crop.ndim == 3:
            for c in range(affine_crop.shape[0]):
                channel = np.ascontiguousarray(affine_crop[c])
                mapped = apply_diffeo_mapping(mapping, channel)
                registered_crop.append(mapped)
            registered_crop = np.stack(registered_crop, axis=0)
        else:
            channel = np.ascontiguousarray(affine_crop)
            registered_crop = apply_diffeo_mapping(mapping, channel)

        return {
            "status": "success",
            "crop_idx": crop_idx,
            "coord": coord,
            "registered_crop": registered_crop
        }

    except Exception as e:
        logger.warning(f"Diffeo crop {crop_idx+1} failed: {e}")
        # Return affine crop as fallback
        if affine_mem.ndim == 3:
            affine_crop = affine_mem[:, y:y+h, x:x+w].astype(np.float32)
        else:
            affine_crop = affine_mem[y:y+h, x:x+w].astype(np.float32)

        return {
            "status": "failed",
            "crop_idx": crop_idx,
            "coord": coord,
            "error": str(e),
            "affine_crop": affine_crop
        }


def run_diffeo_stage(ref_mem: np.ndarray, affine_mem: np.memmap,
                     mov_shape: Tuple[int, ...],
                     diffeo_crop_size: int, diffeo_overlap: int,
                     sigma_diff: int, radius: int,
                     opt_tol: float, inv_tol: float,
                     n_workers: int) -> Tuple[np.memmap, Path, int, int, int]:
    """
    Run CPU multi-threaded diffeomorphic stage on the affine-transformed image.

    Returns:
        diffeo_mem: memmap of final registered image
        diffeo_tmp_dir: temp directory containing memmaps
        success_count: successful diffeo crops
        fallback_count: crops that fell back to affine-only
        total_crops: total diffeo crops
    """
    logger.info("=" * 80)
    logger.info("STAGE: DIFFEOMORPHIC")
    logger.info("=" * 80)

    # Extract crop coordinates for diffeo stage
    diffeo_coords = extract_crop_coords(mov_shape, diffeo_crop_size, diffeo_overlap)
    logger.info(f"Diffeo stage: {len(diffeo_coords)} crops (size={diffeo_crop_size}, overlap={diffeo_overlap})")
    logger.info(f"DIPY parameters: sigma_diff={sigma_diff}, radius={radius}, opt_tol={opt_tol}, inv_tol={inv_tol}")
    logger.info(f"Using {n_workers} worker threads")

    # Create memmap for final result
    diffeo_mem, diffeo_weights, diffeo_tmp_dir = create_memmaps_for_merge(mov_shape, np.float32, "diffeo_")
    logger.info(f"Created diffeo memmap in {diffeo_tmp_dir}")

    # Process crops in parallel
    success_count = 0
    fallback_count = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_diffeo_crop, i, ref_mem, affine_mem, diffeo_coords[i],
                          sigma_diff, radius, opt_tol, inv_tol): i
            for i in range(len(diffeo_coords))
        }

        for future in as_completed(futures):
            res = future.result()
            idx = res["crop_idx"]
            coord = res["coord"]
            y, x, h, w = coord["y"], coord["x"], coord["h"], coord["w"]

            # Calculate bounds using same logic as cropping.reconstruct_image
            if len(mov_shape) == 3:
                img_height, img_width = mov_shape[1], mov_shape[2]
            else:
                img_height, img_width = mov_shape[0], mov_shape[1]
            y_start, y_end, y_slice = calculate_bounds(y, h, img_height, diffeo_overlap)
            x_start, x_end, x_slice = calculate_bounds(x, w, img_width, diffeo_overlap)

            if res["status"] == "success":
                # Use registered crop
                registered_crop = res["registered_crop"]
                if registered_crop.ndim == 3:
                    for c in range(registered_crop.shape[0]):
                        crop_trimmed = registered_crop[c][y_slice, x_slice]
                        diffeo_mem[c, y_start:y_end, x_start:x_end] = crop_trimmed
                else:
                    crop_trimmed = registered_crop[y_slice, x_slice]
                    diffeo_mem[y_start:y_end, x_start:x_end] = crop_trimmed

                success_count += 1
                logger.info(f"Diffeo crop {idx+1}/{len(diffeo_coords)} at ({y},{x}): ✓")

            elif res["status"] == "uniform":
                # Use affine crop directly for uniform intensity
                affine_crop = res["affine_crop"]
                if affine_crop.ndim == 3:
                    for c in range(affine_crop.shape[0]):
                        crop_trimmed = affine_crop[c][y_slice, x_slice]
                        diffeo_mem[c, y_start:y_end, x_start:x_end] = crop_trimmed
                else:
                    crop_trimmed = affine_crop[y_slice, x_slice]
                    diffeo_mem[y_start:y_end, x_start:x_end] = crop_trimmed

                fallback_count += 1
                logger.warning(f"Diffeo crop {idx+1}/{len(diffeo_coords)} at ({y},{x}): uniform intensity, using affine")

            else:  # failed
                # Use affine crop as fallback
                affine_crop = res["affine_crop"]
                if affine_crop.ndim == 3:
                    for c in range(affine_crop.shape[0]):
                        crop_trimmed = affine_crop[c][y_slice, x_slice]
                        diffeo_mem[c, y_start:y_end, x_start:x_end] = crop_trimmed
                else:
                    crop_trimmed = affine_crop[y_slice, x_slice]
                    diffeo_mem[y_start:y_end, x_start:x_end] = crop_trimmed

                fallback_count += 1
                logger.warning(f"Diffeo crop {idx+1}/{len(diffeo_coords)} at ({y},{x}): fallback to affine")

    # Flush and cleanup
    diffeo_mem.flush()
    del diffeo_weights
    gc.collect()

    logger.info(f"Diffeo stage complete: {success_count} success, {fallback_count} fallback")
    return diffeo_mem, diffeo_tmp_dir, success_count, fallback_count, len(diffeo_coords)


# --------------------- Main Multi-Resolution Registration Pipeline ---------------------

def extract_channel_names(moving_path: Path, num_channels: int) -> List[str]:
    """Extract channel names from file metadata or filename."""
    channel_names = []

    # Try OME metadata first
    try:
        with tifffile.TiffFile(str(moving_path)) as tif:
            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                channels = root.findall('.//ome:Channel', ns)
                channel_names = [ch.get('Name') for ch in channels]
    except Exception as e:
        logger.debug(f"Could not extract channel names from OME metadata: {e}")

    # Validate or fallback to filename parsing
    if not channel_names or len(channel_names) != num_channels:
        filename = moving_path.stem
        name_part = filename.replace('_corrected', '').replace('_preprocessed', '').replace('_registered', '').replace('_padded', '')
        parts = name_part.split('_')

        if len(parts) > 1 and '-' in parts[0] and any(c.isdigit() for c in parts[0]):
            markers = parts[1:]
        else:
            markers = parts

        if len(markers) == num_channels:
            channel_names = markers
        else:
            channel_names = [f"channel_{i}" for i in range(num_channels)]

    return channel_names


def create_ome_xml(channel_names: List[str], dtype: np.dtype,
                   width: int, height: int) -> str:
    """Create OME-XML metadata string."""
    num_channels = len(channel_names)

    channel_xml = '\n'.join(
        f'            <Channel ID="Channel:0:{i}" Name="{name}" SamplesPerPixel="1" />'
        for i, name in enumerate(channel_names)
    )

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
    <Image ID="Image:0" Name="Registered">
        <Pixels ID="Pixels:0" Type="{dtype.name}"
                SizeX="{width}" SizeY="{height}" SizeZ="1" SizeC="{num_channels}" SizeT="1"
                DimensionOrder="XYCZT"
                PhysicalSizeX="0.325" PhysicalSizeY="0.325" PhysicalSizeXUnit="um" PhysicalSizeYUnit="um">
{channel_xml}
            <TiffData />
        </Pixels>
    </Image>
</OME>'''


def register_image_pair_multires(
    reference_path: Path,
    moving_path: Path,
    output_path: Path,
    # Coarse affine parameters
    coarse_crop_size: int = 4000,
    coarse_overlap_percent: float = 10.0,
    coarse_n_features: int = 1000,
    # Fine affine parameters
    fine_crop_size: int = 2500,
    fine_overlap_percent: float = 10.0,
    fine_n_features: int = 5000,
    # Diffeomorphic parameters
    diffeo_crop_size: int = 2000,
    diffeo_overlap_percent: float = 15.0,
    diffeo_sigma_diff: int = 20,
    diffeo_radius: int = 20,
    opt_tol: float = 1e-6,
    inv_tol: float = 1e-6,
    # General parameters
    n_workers: int = 4,
    qc_dir: Optional[Path] = None,
):
    """
    Register moving image to reference using multi-resolution coarse-to-fine strategy.

    Pipeline:
    1. Coarse Affine: Large crops, fewer features → rough global alignment
    2. Fine Affine: Medium crops, many features → precise global alignment
    3. Diffeomorphic: Optimal crops → local deformations
    """
    if not DIPY_AVAILABLE:
        raise RuntimeError(
            f"CPU registration requires DIPY library.\nImport error: {_dipy_import_error}"
        )

    # Open images as memmaps (read-only)
    logger.info(f"Opening reference: {reference_path}")
    ref_mem = tifffile.memmap(str(reference_path), mode='r')

    logger.info(f"Opening moving: {moving_path}")
    mov_mem = tifffile.memmap(str(moving_path), mode='r')

    # Normalize shapes to (C, H, W)
    if ref_mem.ndim == 2:
        ref_shape = (1, ref_mem.shape[0], ref_mem.shape[1])
    else:
        ref_shape = ref_mem.shape

    if mov_mem.ndim == 2:
        mov_shape = (1, mov_mem.shape[0], mov_mem.shape[1])
    else:
        mov_shape = mov_mem.shape

    logger.info(f"Reference shape: {ref_shape}, Moving shape: {mov_shape}")

    # Validate dimensions
    if ref_shape[1:] != mov_shape[1:]:
        raise ValueError(f"Spatial dimension mismatch: ref {ref_shape[1:]} != mov {mov_shape[1:]}")

    if ref_shape[0] != mov_shape[0]:
        logger.warning(f"Channel count mismatch: ref={ref_shape[0]}, mov={mov_shape[0]}")

    # Compute overlap sizes
    coarse_overlap = int(coarse_crop_size * coarse_overlap_percent / 100.0)
    fine_overlap = int(fine_crop_size * fine_overlap_percent / 100.0)
    diffeo_overlap = int(diffeo_crop_size * diffeo_overlap_percent / 100.0)

    logger.info("=" * 80)
    logger.info("MULTI-RESOLUTION REGISTRATION PARAMETERS")
    logger.info("=" * 80)
    logger.info(f"Coarse Affine: crop_size={coarse_crop_size}, overlap={coarse_overlap}px ({coarse_overlap_percent}%), features={coarse_n_features}")
    logger.info(f"Fine Affine:   crop_size={fine_crop_size}, overlap={fine_overlap}px ({fine_overlap_percent}%), features={fine_n_features}")
    logger.info(f"Diffeomorphic: crop_size={diffeo_crop_size}, overlap={diffeo_overlap}px ({diffeo_overlap_percent}%)")
    logger.info(f"               sigma_diff={diffeo_sigma_diff}, radius={diffeo_radius}")
    logger.info(f"               opt_tol={opt_tol}, inv_tol={inv_tol}")
    logger.info(f"Workers: {n_workers}")
    logger.info("=" * 80)

    orig_dtype = mov_mem.dtype
    coarse_tmp_dir = None
    fine_tmp_dir = None
    diffeo_tmp_dir = None

    try:
        # STAGE 1: Coarse Affine (rough global alignment)
        logger.info("\n🔍 Starting COARSE AFFINE stage...")
        coarse_mem, coarse_tmp_dir, coarse_success, coarse_total = run_affine_stage(
            ref_mem, mov_mem, mov_shape,
            coarse_crop_size, coarse_overlap,
            coarse_n_features, n_workers,
            stage_name="COARSE AFFINE"
        )

        # STAGE 2: Fine Affine (precise global alignment on coarse result)
        logger.info("\n🎯 Starting FINE AFFINE stage...")
        fine_mem, fine_tmp_dir, fine_success, fine_total = run_affine_stage(
            ref_mem, coarse_mem, mov_shape,
            fine_crop_size, fine_overlap,
            fine_n_features, n_workers,
            stage_name="FINE AFFINE"
        )

        # Clean up coarse memmap after fine affine is done
        del coarse_mem
        gc.collect()
        if coarse_tmp_dir:
            cleanup_memmaps(coarse_tmp_dir)
            coarse_tmp_dir = None

        # STAGE 3: Diffeomorphic (local deformations on fine affine result)
        logger.info("\n🌀 Starting DIFFEOMORPHIC stage...")
        diffeo_mem, diffeo_tmp_dir, diffeo_success, diffeo_fallback, diffeo_total = run_diffeo_stage(
            ref_mem, fine_mem, mov_shape,
            diffeo_crop_size, diffeo_overlap,
            diffeo_sigma_diff, diffeo_radius,
            opt_tol, inv_tol,
            n_workers
        )

        # Clean up fine memmap after diffeo is done
        del fine_mem
        gc.collect()
        if fine_tmp_dir:
            cleanup_memmaps(fine_tmp_dir)
            fine_tmp_dir = None

        # Convert to original dtype
        logger.info("Converting to output dtype...")
        if orig_dtype == np.uint16:
            out = np.clip(diffeo_mem, 0, 65535).astype(np.uint16)
        elif orig_dtype == np.uint8:
            out = np.clip(diffeo_mem, 0, 255).astype(np.uint8)
        else:
            out = diffeo_mem.astype(orig_dtype)

        # Ensure 3D shape
        if out.ndim == 2:
            out = out[np.newaxis, ...]

        # Extract channel names
        channel_names = extract_channel_names(moving_path, out.shape[0])
        logger.info(f"Channel names: {channel_names}")

        # Create OME-XML metadata
        _, height, width = out.shape
        ome_xml = create_ome_xml(channel_names, out.dtype, width, height)

        # Save output
        logger.info(f"Saving registered image: {output_path}")
        tifffile.imwrite(
            str(output_path),
            out,
            metadata={'axes': 'CYX'},
            description=ome_xml,
            compression="zlib",
            bigtiff=True
        )

        # Summary
        logger.info("=" * 80)
        logger.info("MULTI-RESOLUTION REGISTRATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Coarse Affine: {coarse_success}/{coarse_total} crops succeeded ({100*coarse_success/coarse_total:.1f}%)")
        logger.info(f"Fine Affine:   {fine_success}/{fine_total} crops succeeded ({100*fine_success/fine_total:.1f}%)")
        logger.info(f"Diffeomorphic: {diffeo_success}/{diffeo_total} success, {diffeo_fallback}/{diffeo_total} fallback")
        logger.info(f"Output saved:  {output_path}")
        logger.info("=" * 80)

        # Generate QC if requested
        if qc_dir:
            logger.info(f"Generating QC outputs: {qc_dir}")
            qc_dir.mkdir(parents=True, exist_ok=True)
            qc_filename = f"{moving_path.stem}_QC_RGB.png"
            qc_output_path = qc_dir / qc_filename

            try:
                create_qc_rgb_composite(reference_path, output_path, qc_output_path)
            except Exception as e:
                logger.warning(f"Failed to generate QC composite: {e}")

        logger.info("Multi-resolution registration complete ✅")

    finally:
        # Ensure memmaps are properly closed before cleanup
        if 'coarse_mem' in locals():
            try:
                coarse_mem.flush()
            except Exception:
                pass
            del coarse_mem
        if 'fine_mem' in locals():
            try:
                fine_mem.flush()
            except Exception:
                pass
            del fine_mem
        if 'diffeo_mem' in locals():
            try:
                diffeo_mem.flush()
            except Exception:
                pass
            del diffeo_mem
        gc.collect()

        # Cleanup temp directories
        if coarse_tmp_dir:
            cleanup_memmaps(coarse_tmp_dir)
        if fine_tmp_dir:
            cleanup_memmaps(fine_tmp_dir)
        if diffeo_tmp_dir:
            cleanup_memmaps(diffeo_tmp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-resolution CPU-based image registration with coarse-to-fine strategy"
    )
    parser.add_argument("--reference", type=str, required=True,
                        help="Path to reference image")
    parser.add_argument("--moving", type=str, required=True,
                        help="Path to moving image to register")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save registered image")
    parser.add_argument("--qc-dir", type=str, default=None,
                        help="Directory to save QC outputs (optional)")

    # Coarse affine parameters
    parser.add_argument("--coarse-crop-size", type=int, default=4000,
                        help="Crop size for coarse affine stage (default: 4000)")
    parser.add_argument("--coarse-overlap-percent", type=float, default=10.0,
                        help="Overlap for coarse affine stage (default: 10%%)")
    parser.add_argument("--coarse-n-features", type=int, default=1000,
                        help="Number of features for coarse affine (default: 1000)")

    # Fine affine parameters
    parser.add_argument("--fine-crop-size", type=int, default=2500,
                        help="Crop size for fine affine stage (default: 2500)")
    parser.add_argument("--fine-overlap-percent", type=float, default=10.0,
                        help="Overlap for fine affine stage (default: 10%%)")
    parser.add_argument("--fine-n-features", type=int, default=5000,
                        help="Number of features for fine affine (default: 5000)")

    # Diffeomorphic parameters
    parser.add_argument("--diffeo-crop-size", type=int, default=2000,
                        help="Crop size for diffeomorphic stage (default: 2000)")
    parser.add_argument("--diffeo-overlap-percent", type=float, default=15.0,
                        help="Overlap for diffeomorphic stage (default: 15%%)")
    parser.add_argument("--diffeo-sigma-diff", type=int, default=20,
                        help="Sigma for Gaussian smoothing in SyN (default: 20)")
    parser.add_argument("--diffeo-radius", type=int, default=20,
                        help="Neighborhood radius for SyN metric (default: 20)")
    parser.add_argument("--opt-tol", type=float, default=1e-6,
                        help="Optimization tolerance for diffeomorphic registration (default: 1e-6)")
    parser.add_argument("--inv-tol", type=float, default=1e-6,
                        help="Inverse tolerance for diffeomorphic registration (default: 1e-6)")

    # General parameters
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Number of parallel workers for all stages")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print_dipy_diagnostics()

    try:
        register_image_pair_multires(
            reference_path=Path(args.reference),
            moving_path=Path(args.moving),
            output_path=Path(args.output),
            coarse_crop_size=args.coarse_crop_size,
            coarse_overlap_percent=args.coarse_overlap_percent,
            coarse_n_features=args.coarse_n_features,
            fine_crop_size=args.fine_crop_size,
            fine_overlap_percent=args.fine_overlap_percent,
            fine_n_features=args.fine_n_features,
            diffeo_crop_size=args.diffeo_crop_size,
            diffeo_overlap_percent=args.diffeo_overlap_percent,
            diffeo_sigma_diff=args.diffeo_sigma_diff,
            diffeo_radius=args.diffeo_radius,
            opt_tol=args.opt_tol,
            inv_tol=args.inv_tol,
            n_workers=args.n_workers,
            qc_dir=Path(args.qc_dir) if args.qc_dir else None,
        )
        return 0
    except Exception as e:
        logger.error(f"Multi-resolution registration failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
