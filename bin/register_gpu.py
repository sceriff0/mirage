#!/usr/bin/env python3
"""GPU-based image registration (refactored streaming version).

Properly separates affine and diffeomorphic stages with different crop sizes:
- Stage 1 (CPU): Affine registration on affine_crop_size crops → merged affine-transformed image
- Stage 2 (GPU): Diffeomorphic registration on diffeo_crop_size crops → final registered image

Memory-aware design:
- Uses memmap for large intermediate results
- Streams crops one at a time in GPU stage
- Aggressively frees memory after each crop

Usage: same CLI as original script. Requires CuPy + cuDIPY for GPU diffeo.
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

# Keep a stable Cupy cache directory
os.environ.setdefault("CUPY_CACHE_DIR", "/tmp/.cupy")

# GPU imports with availability check
try:
    import cupy as cp
    from cudipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from cudipy.align.metrics import CCMetric
    GPU_AVAILABLE = True
except Exception as e:
    GPU_AVAILABLE = False
    _gpu_import_error = str(e)

# CUDA error types that indicate corrupted GPU state
CUDA_FATAL_ERRORS = (
    'cudaErrorIllegalAddress',
    'cudaErrorIllegalMemoryAccess', 
    'cudaErrorHardwareStackError',
    'cudaErrorIllegalInstruction',
    'cudaErrorMisalignedAddress',
    'cudaErrorInvalidAddressSpace',
    'cudaErrorInvalidPc',
    'cudaErrorLaunchFailure',
    'cudaErrorECCUncorrectable',
)

logger = logging.getLogger(__name__)


def print_cuda_diagnostics():
    """Print CUDA driver, runtime, and library version information."""
    logger.info("=" * 80)
    logger.info("CUDA DIAGNOSTICS")
    logger.info("=" * 80)

    if not GPU_AVAILABLE:
        logger.error(f"✗ GPU libraries not available: {_gpu_import_error}")
        logger.info("=" * 80)
        return

    try:
        logger.info(f"CuPy version: {cp.__version__}")
    except Exception as e:
        logger.error(f"Could not get CuPy version: {e}")

    try:
        runtime_version = cp.cuda.runtime.runtimeGetVersion()
        major = runtime_version // 1000
        minor = (runtime_version % 1000) // 10
        logger.info(f"CUDA Runtime Version (CuPy): {major}.{minor}")
    except Exception as e:
        logger.error(f"✗ CUDA Runtime check FAILED: {e}")

    try:
        driver_version = cp.cuda.runtime.driverGetVersion()
        major = driver_version // 1000
        minor = (driver_version % 1000) // 10
        logger.info(f"CUDA Driver Version: {major}.{minor}")
    except Exception as e:
        logger.error(f"✗ CUDA Driver check FAILED: {e}")

    try:
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        logger.info(f"GPU Device: {props['name'].decode('utf-8')}")
        logger.info(f"Compute Capability: {props['major']}.{props['minor']}")
        logger.info(f"Total Memory: {props['totalGlobalMem'] / 1024**3:.2f} GB")
    except Exception as e:
        logger.error(f"✗ GPU Device check FAILED: {e}")

    try:
        import cudipy
        if hasattr(cudipy, '__version__'):
            logger.info(f"cuDIPY version: {cudipy.__version__}")
        else:
            logger.info("cuDIPY version: (version not available)")
    except Exception as e:
        logger.warning(f"cuDIPY import: {e}")

    logger.info("=" * 80)


# ------------------------- Helper Functions ---------------------------------

def get_channel_names(filename: str) -> List[str]:
    base = os.path.basename(filename)
    # Remove all suffixes that might be present
    name_part = base.replace('_corrected', '').replace('_padded', '').replace('_preprocessed', '').replace('_registered', '').split('.')[0]
    parts = name_part.split('_')
    channels = parts[1:]
    return channels


def autoscale(img: np.ndarray, low_p: float = 1, high_p: float = 99) -> np.ndarray:
    lo = np.percentile(img, low_p)
    hi = np.percentile(img, high_p)
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)


def create_qc_rgb_composite(reference_path: Path, registered_path: Path, output_path: Path) -> None:
    logger.info(f"Creating QC composite: {output_path.name}")

    ref_img = tifffile.imread(str(reference_path))
    reg_img = tifffile.imread(str(registered_path))

    if ref_img.ndim == 2:
        ref_img = ref_img[np.newaxis, ...]
    if reg_img.ndim == 2:
        reg_img = reg_img[np.newaxis, ...]

    ref_channels = get_channel_names(reference_path.name)
    reg_channels = get_channel_names(registered_path.name)

    ref_dapi_idx = next((i for i, ch in enumerate(ref_channels) if "DAPI" in ch.upper()), 0)
    reg_dapi_idx = next((i for i, ch in enumerate(reg_channels) if "DAPI" in ch.upper()), 0)

    ref_dapi = ref_img[ref_dapi_idx]
    reg_dapi = reg_img[reg_dapi_idx]

    ref_dapi_scaled = autoscale(ref_dapi)
    reg_dapi_scaled = autoscale(reg_dapi)

    h, w = reg_dapi_scaled.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 2] = reg_dapi_scaled  # Blue channel
    rgb[:, :, 1] = ref_dapi_scaled  # Green channel
    rgb[:, :, 0] = 0                 # Red channel

    # Change extension to .png and write as PNG
    png_output_path = output_path.with_suffix('.png')
    cv2.imwrite(str(png_output_path), rgb)
    logger.info(f"  Saved QC composite: {png_output_path}")


def apply_affine_cv2(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply affine transformation using OpenCV."""
    height, width = x.shape[:2]
    return cv2.warpAffine(x, matrix, (width, height), flags=cv2.INTER_LINEAR)


def apply_diffeo_mapping(mapping, x: np.ndarray) -> np.ndarray:
    """Apply diffeomorphic mapping using cuDIPY."""
    mapped = mapping.transform(x)
    if hasattr(mapped, "get"):
        mapped = mapped.get()
    return mapped


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
    """Compute diffeomorphic mapping using cuDIPY."""
    if y.shape != x.shape:
        raise ValueError("Reference and moving images must have the same shape.")

    # Thorough input validation to prevent CUDA errors
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

    # Ensure contiguous arrays
    y_cont = np.ascontiguousarray(y)
    x_cont = np.ascontiguousarray(x)
    
    # Transfer to GPU
    y_gpu = cp.asarray(y_cont)
    x_gpu = cp.asarray(x_cont)
    
    # Free CPU copies
    del y_cont, x_cont

    crop_size = y.shape[0]
    scale_factor = crop_size / 2000
    radius =  int(20 * scale_factor)  # Minimum radius of 5
    sigma_diff = int(20 * np.sqrt(scale_factor))  # Minimum sigma of 5

    metric = CCMetric(2, sigma_diff=sigma_diff, radius=radius)
    sdr = SymmetricDiffeomorphicRegistration(metric, opt_tol=opt_tol, inv_tol=inv_tol)

    try:
        mapping = sdr.optimize(y_gpu, x_gpu)
    finally:
        # Always clean up GPU arrays, even on failure
        del y_gpu, x_gpu
    
    return mapping


def extract_crop_coords(image_shape: Tuple[int, ...], crop_size: int, overlap: int) -> List[Dict]:
    """Return list of coordinate dicts (y, x, h, w) for tiling the image."""
    if len(image_shape) == 3:
        _, height, width = image_shape
    else:
        height, width = image_shape

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


def create_weight_mask(h: int, w: int, overlap: int) -> np.ndarray:
    """Create a weight mask with hard cutoff at overlap midpoint (no blending)."""
    mask = np.ones((h, w), dtype=np.float32)
    if overlap > 0:
        # Hard cutoff: set edges to 0 (will be trimmed during merging)
        # This mimics the "bad" script's hard cutoff behavior
        half_overlap = overlap // 2
        mask[:half_overlap, :] = 0
        mask[-half_overlap:, :] = 0
        mask[:, :half_overlap] = 0
        mask[:, -half_overlap:] = 0
    return mask


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


# --------------------- CPU Stage: Affine Registration ---------------------

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
                     mov_shape: Tuple[int, ...], orig_dtype: np.dtype,
                     affine_crop_size: int, affine_overlap: int,
                     n_features: int, n_workers: int) -> Tuple[np.memmap, Path, int, int]:
    """
    Run CPU affine stage: compute matrices and merge into affine-transformed image.
    
    Returns:
        affine_mem: memmap of affine-transformed moving image
        affine_tmp_dir: temp directory containing memmaps
        success_count: number of successful crops
        total_crops: total number of crops
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: CPU AFFINE REGISTRATION")
    logger.info("=" * 80)
    
    # Extract crop coordinates
    affine_coords = extract_crop_coords(mov_shape, affine_crop_size, affine_overlap)
    logger.info(f"Affine stage: {len(affine_coords)} crops (size={affine_crop_size}, overlap={affine_overlap})")
    
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
                logger.debug(f"  Affine crop {idx+1}/{len(affine_coords)} at ({coord['y']},{coord['x']}): ✓")
            else:
                logger.warning(f"  Affine crop {idx+1}/{len(affine_coords)} at ({coord['y']},{coord['x']}): ✗ {res.get('error', 'unknown')}")
    
    # Summarize results
    successful = [r for r in cpu_results if r["status"] == "success"]
    failed = [r for r in cpu_results if r["status"] != "success"]
    
    logger.info(f"Affine stage: {len(successful)}/{len(affine_coords)} crops succeeded ({100*len(successful)/len(affine_coords):.1f}%)")
    
    if failed:
        failure_reasons = {}
        for r in failed:
            reason = r.get('error', 'unknown')
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            logger.warning(f"  - {reason}: {count} crops")
    
    if len(successful) == 0:
        raise RuntimeError("All affine crops failed. Cannot proceed.")
    
    # Create memmap for affine-transformed image
    affine_mem, affine_weights, affine_tmp_dir = create_memmaps_for_merge(mov_shape, np.float32, "affine_")
    logger.info(f"Created affine memmap in {affine_tmp_dir}")
    
    # Apply affine transformations and merge
    logger.info("Applying affine transformations and merging...")
    weight_cache = {}
    
    for res in cpu_results:
        coord = res["coord"]
        y, x, h, w = coord["y"], coord["x"], coord["h"], coord["w"]
        
        # Get or create weight mask
        if (h, w) not in weight_cache:
            weight_cache[(h, w)] = create_weight_mask(h, w, affine_overlap)
        weight_mask = weight_cache[(h, w)]
        
        if res["status"] != "success":
            # For failed crops, use original (identity transform)
            if mov_mem.ndim == 3:
                for c in range(mov_mem.shape[0]):
                    crop = mov_mem[c, y:y+h, x:x+w].astype(np.float32)
                    affine_mem[c, y:y+h, x:x+w] += crop * weight_mask
                    affine_weights[c, y:y+h, x:x+w] += weight_mask
            else:
                crop = mov_mem[y:y+h, x:x+w].astype(np.float32)
                affine_mem[y:y+h, x:x+w] += crop * weight_mask
                affine_weights[y:y+h, x:x+w] += weight_mask
        else:
            # Apply affine transformation
            matrix = res["matrix"]
            if mov_mem.ndim == 3:
                for c in range(mov_mem.shape[0]):
                    crop = mov_mem[c, y:y+h, x:x+w].astype(np.float32)
                    transformed = apply_affine_cv2(np.ascontiguousarray(crop), matrix)
                    affine_mem[c, y:y+h, x:x+w] += transformed * weight_mask
                    affine_weights[c, y:y+h, x:x+w] += weight_mask
                    del crop, transformed
            else:
                crop = mov_mem[y:y+h, x:x+w].astype(np.float32)
                transformed = apply_affine_cv2(np.ascontiguousarray(crop), matrix)
                affine_mem[y:y+h, x:x+w] += transformed * weight_mask
                affine_weights[y:y+h, x:x+w] += weight_mask
                del crop, transformed
        
        gc.collect()
    
    # Normalize by weights
    logger.info("Normalizing affine result...")
    nonzero_mask = affine_weights > 0
    affine_mem[nonzero_mask] /= affine_weights[nonzero_mask]
    
    # Flush to disk
    affine_mem.flush()
    
    # Clean up weights memmap
    del affine_weights
    gc.collect()
    
    logger.info("Affine stage complete")
    return affine_mem, affine_tmp_dir, len(successful), len(affine_coords)


# --------------------- GPU Stage: Diffeomorphic Registration ---------------------

def clear_gpu_memory():
    """Aggressively clear GPU memory pools."""
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize()
    except Exception:
        pass


def is_fatal_cuda_error(error: Exception) -> bool:
    """Check if the error is a fatal CUDA error requiring context reset."""
    error_str = str(error)
    return any(fatal in error_str for fatal in CUDA_FATAL_ERRORS)


def reset_cuda_context():
    """
    Fully reset CUDA context after fatal errors like cudaErrorIllegalAddress.
    
    This is necessary because once a CUDA error corrupts the context,
    all subsequent operations will fail until the context is reset.
    """
    logger.warning("Attempting full CUDA context reset...")
    
    try:
        # Step 1: Synchronize to ensure all pending operations complete/fail
        try:
            cp.cuda.Device().synchronize()
        except Exception:
            pass
        
        # Step 2: Clear all memory pools
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        
        # Step 3: Reset the device (this clears all allocations and state)
        try:
            device = cp.cuda.Device()
            device.synchronize()
            
            # cupy doesn't have direct device reset, but we can recreate pools
            # and clear the FFT plan cache which cuDIPY uses heavily
            cp.fft.config.clear_plan_cache()
        except Exception as e:
            logger.warning(f"Could not clear FFT cache: {e}")
        
        # Step 4: Force Python garbage collection
        gc.collect()
        
        # Step 5: Small allocation test to verify context is working
        try:
            test = cp.zeros((10, 10), dtype=cp.float32)
            _ = float(test.sum())
            del test
            logger.info("CUDA context reset successful - GPU is responsive")
            return True
        except Exception as e:
            logger.error(f"CUDA context still broken after reset: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to reset CUDA context: {e}")
        return False


def is_cuda_context_healthy():
    """Quick check if CUDA context is still usable."""
    try:
        test = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        result = float(test.sum())
        del test
        return abs(result - 6.0) < 0.01
    except Exception:
        return False


def log_gpu_memory_usage():
    """Log current GPU memory usage."""
    try:
        mempool = cp.get_default_memory_pool()
        used = mempool.used_bytes() / (1024**3)
        total = mempool.total_bytes() / (1024**3)
        
        # Also get device memory info
        device = cp.cuda.Device()
        free_mem, total_mem = device.mem_info
        free_gb = free_mem / (1024**3)
        total_gb = total_mem / (1024**3)
        used_gb = total_gb - free_gb
        
        logger.debug(f"GPU Memory: Pool={used:.2f}/{total:.2f} GB, Device={used_gb:.2f}/{total_gb:.2f} GB ({100*used_gb/total_gb:.1f}% used)")
        return used_gb, total_gb
    except Exception as e:
        logger.debug(f"Could not get GPU memory info: {e}")
        return None, None


def run_diffeo_stage(ref_mem: np.ndarray, affine_mem: np.memmap,
                     mov_shape: Tuple[int, ...],
                     diffeo_crop_size: int, diffeo_overlap: int,
                     opt_tol: float, inv_tol: float,
                     gpu_reset_interval: int = 50) -> Tuple[np.memmap, Path, int, int, int]:
    """
    Run GPU diffeomorphic stage on the affine-transformed image.
    
    Returns:
        diffeo_mem: memmap of final registered image
        diffeo_tmp_dir: temp directory containing memmaps
        success_count: successful diffeo crops
        fallback_count: crops that fell back to affine-only
        total_crops: total diffeo crops
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: GPU DIFFEOMORPHIC REGISTRATION")
    logger.info("=" * 80)
    
    # Clear GPU memory before starting
    gc.collect()
    clear_gpu_memory()
    
    # Extract crop coordinates for diffeo stage
    diffeo_coords = extract_crop_coords(mov_shape, diffeo_crop_size, diffeo_overlap)
    logger.info(f"Diffeo stage: {len(diffeo_coords)} crops (size={diffeo_crop_size}, overlap={diffeo_overlap})")
    
    # Create memmap for final result
    diffeo_mem, diffeo_weights, diffeo_tmp_dir = create_memmaps_for_merge(mov_shape, np.float32, "diffeo_")
    logger.info(f"Created diffeo memmap in {diffeo_tmp_dir}")
    
    # Process crops sequentially (GPU memory constraint)
    weight_cache = {}
    success_count = 0
    fallback_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 5  # If we hit this many in a row, GPU is likely broken
    cuda_context_broken = False
    
    # Log initial GPU memory state
    log_gpu_memory_usage()
    
    for idx, coord in enumerate(diffeo_coords):
        y, x, h, w = coord["y"], coord["x"], coord["h"], coord["w"]
        
        # Proactive GPU reset every N crops to prevent memory fragmentation
        # This happens BEFORE errors occur, not just after
        # Set gpu_reset_interval to 0 to disable
        if gpu_reset_interval > 0 and idx > 0 and idx % gpu_reset_interval == 0 and not cuda_context_broken:
            logger.info(f"Proactive GPU reset at crop {idx}/{len(diffeo_coords)} (every {gpu_reset_interval} crops)")
            log_gpu_memory_usage()
            reset_cuda_context()
            time.sleep(0.2)  # Brief pause to let GPU stabilize
        
        # Get or create weight mask
        if (h, w) not in weight_cache:
            weight_cache[(h, w)] = create_weight_mask(h, w, diffeo_overlap)
        weight_mask = weight_cache[(h, w)]
        
        # Read affine-transformed moving crop (all channels) - always needed
        if affine_mem.ndim == 3:
            affine_crop = affine_mem[:, y:y+h, x:x+w].astype(np.float32)
            affine_crop_ch0 = affine_crop[0]
        else:
            affine_crop = affine_mem[y:y+h, x:x+w].astype(np.float32)
            affine_crop_ch0 = affine_crop
        
        # Check if CUDA context is broken - skip GPU work entirely
        if cuda_context_broken:
            # Use affine result directly
            if affine_mem.ndim == 3:
                for c in range(affine_mem.shape[0]):
                    diffeo_mem[c, y:y+h, x:x+w] += affine_crop[c] * weight_mask
                    diffeo_weights[c, y:y+h, x:x+w] += weight_mask
            else:
                diffeo_mem[y:y+h, x:x+w] += affine_crop * weight_mask
                diffeo_weights[y:y+h, x:x+w] += weight_mask

            fallback_count += 1
            del affine_crop
            gc.collect()
            clear_gpu_memory()
            continue
        
        try:
            # Read reference crop (channel 0 for metric computation)
            if ref_mem.ndim == 3:
                ref_crop = ref_mem[0, y:y+h, x:x+w].astype(np.float32)
            else:
                ref_crop = ref_mem[y:y+h, x:x+w].astype(np.float32)
            
            # Check for valid intensity range
            crop_min, crop_max = affine_crop_ch0.min(), affine_crop_ch0.max()
            if crop_max - crop_min < 1e-6:
                logger.warning(f"Diffeo crop {idx+1}/{len(diffeo_coords)} at ({y},{x}): uniform intensity, using affine-only")
                fallback_count += 1
                
                # Use affine result directly
                if affine_mem.ndim == 3:
                    for c in range(affine_mem.shape[0]):
                        diffeo_mem[c, y:y+h, x:x+w] += affine_crop[c] * weight_mask
                        diffeo_weights[c, y:y+h, x:x+w] += weight_mask
                else:
                    diffeo_mem[y:y+h, x:x+w] += affine_crop * weight_mask
                    diffeo_weights[y:y+h, x:x+w] += weight_mask
                
                del ref_crop, affine_crop
                gc.collect()
                consecutive_failures = 0  # Uniform intensity is not a GPU failure
                continue
            
            # Compute diffeomorphic mapping
            ref_crop_cont = np.ascontiguousarray(ref_crop)
            affine_crop_ch0_cont = np.ascontiguousarray(affine_crop_ch0)
            
            mapping = None
            diffeo_succeeded = False
            
            try:
                mapping = compute_diffeomorphic_mapping_dipy(
                    ref_crop_cont, affine_crop_ch0_cont,
                    opt_tol=opt_tol, inv_tol=inv_tol
                )
                diffeo_succeeded = True
                consecutive_failures = 0
                
            except Exception as diffeo_err:
                error_msg = str(diffeo_err).lower()

                # Check for fatal CUDA errors that corrupt context
                # Note: Don't use generic 'cuda error' - too broad and matches benign messages
                is_fatal_error = any(kw in error_msg for kw in [
                    'illegalmemory', 'illegaladdress', 'cudaerrorillegal',
                    'an illegal memory access', 'unspecified launch failure',
                    'device-side assert', 'cudaerrorhardware',
                    'cudaerrorinvalidpc', 'cudaerrorlaunchfailure'
                ])

                if is_fatal_error:
                    logger.error(f"Diffeo crop {idx+1}: FATAL CUDA ERROR: {diffeo_err}")
                    consecutive_failures += 1
                    
                    # Try to reset CUDA context
                    if reset_cuda_context():
                        # Context reset succeeded, try one more time with very relaxed settings
                        logger.info(f"Retrying crop {idx+1} after context reset...")
                        time.sleep(0.5)  # Give GPU time to stabilize
                        try:
                            clear_gpu_memory()
                            mapping = compute_diffeomorphic_mapping_dipy(
                                ref_crop_cont, affine_crop_ch0_cont,
                                opt_tol=1e-2, inv_tol=1e-2  # Very relaxed
                            )
                            diffeo_succeeded = True
                            consecutive_failures = 0
                            logger.info(f"  → Retry after context reset succeeded")
                        except Exception as retry_err:
                            logger.error(f"  → Retry failed: {retry_err}")
                            consecutive_failures += 1
                    else:
                        logger.error("CUDA context reset failed!")
                        consecutive_failures += 1
                
                elif any(kw in error_msg for kw in ['memory', 'out of memory', 'oom']):
                    # Memory error - try clearing and retrying
                    logger.warning(f"Diffeo crop {idx+1}: GPU memory error, clearing and retrying...")
                    clear_gpu_memory()
                    gc.collect()
                    
                    try:
                        mapping = compute_diffeomorphic_mapping_dipy(
                            ref_crop_cont, affine_crop_ch0_cont,
                            opt_tol=1e-3, inv_tol=1e-3
                        )
                        diffeo_succeeded = True
                        consecutive_failures = 0
                    except Exception:
                        consecutive_failures += 1
                
                else:
                    # Other error (numerical, etc) - just log and fallback
                    logger.warning(f"Diffeo crop {idx+1}: {diffeo_err}")
                    consecutive_failures += 1
            
            del ref_crop_cont, affine_crop_ch0_cont
            
            # Check if we've hit too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                logger.error(f"Hit {consecutive_failures} consecutive GPU failures!")
                logger.error("GPU appears to be in an unrecoverable state.")
                logger.error("Falling back to affine-only for remaining crops.")
                cuda_context_broken = True
                
                # Fallback for this crop
                if affine_mem.ndim == 3:
                    for c in range(affine_mem.shape[0]):
                        diffeo_mem[c, y:y+h, x:x+w] += affine_crop[c] * weight_mask
                        diffeo_weights[c, y:y+h, x:x+w] += weight_mask
                else:
                    diffeo_mem[y:y+h, x:x+w] += affine_crop * weight_mask
                    diffeo_weights[y:y+h, x:x+w] += weight_mask
                
                fallback_count += 1
                del ref_crop, affine_crop
                gc.collect()
                continue
            
            if diffeo_succeeded and mapping is not None:
                # Apply mapping to all channels
                if affine_mem.ndim == 3:
                    for c in range(affine_mem.shape[0]):
                        channel = np.ascontiguousarray(affine_crop[c])
                        mapped = apply_diffeo_mapping(mapping, channel)
                        diffeo_mem[c, y:y+h, x:x+w] += mapped * weight_mask
                        diffeo_weights[c, y:y+h, x:x+w] += weight_mask
                        del channel, mapped
                else:
                    channel = np.ascontiguousarray(affine_crop)
                    mapped = apply_diffeo_mapping(mapping, channel)
                    diffeo_mem[y:y+h, x:x+w] += mapped * weight_mask
                    diffeo_weights[y:y+h, x:x+w] += weight_mask
                    del channel, mapped
                
                del mapping
                success_count += 1
                logger.info(f"Diffeo crop {idx+1}/{len(diffeo_coords)} at ({y},{x}): ✓")
            else:
                # Fallback to affine
                if affine_mem.ndim == 3:
                    for c in range(affine_mem.shape[0]):
                        diffeo_mem[c, y:y+h, x:x+w] += affine_crop[c] * weight_mask
                        diffeo_weights[c, y:y+h, x:x+w] += weight_mask
                else:
                    diffeo_mem[y:y+h, x:x+w] += affine_crop * weight_mask
                    diffeo_weights[y:y+h, x:x+w] += weight_mask
                
                fallback_count += 1
                logger.warning(f"Diffeo crop {idx+1}/{len(diffeo_coords)} at ({y},{x}): fallback to affine")
            
            del ref_crop, affine_crop
            gc.collect()
            clear_gpu_memory()
            
        except Exception as e:
            fallback_count += 1
            logger.error(f"Diffeo crop {idx+1} at ({y},{x}): unexpected error: {e}")
            
            # Fallback: use affine result
            if affine_mem.ndim == 3:
                for c in range(affine_mem.shape[0]):
                    diffeo_mem[c, y:y+h, x:x+w] += affine_crop[c] * weight_mask
                    diffeo_weights[c, y:y+h, x:x+w] += weight_mask
            else:
                diffeo_mem[y:y+h, x:x+w] += affine_crop * weight_mask
                diffeo_weights[y:y+h, x:x+w] += weight_mask
            
            del affine_crop
            gc.collect()
            clear_gpu_memory()
        
        # Periodic GPU health check every 100 crops
        if (idx + 1) % 100 == 0 and not cuda_context_broken:
            if not is_cuda_context_healthy():
                logger.warning(f"GPU health check failed at crop {idx+1}, attempting reset...")
                if not reset_cuda_context():
                    logger.error("Could not recover GPU, switching to affine-only mode")
                    cuda_context_broken = True
    
    # Normalize by weights
    logger.info("Normalizing diffeo result...")
    nonzero_mask = diffeo_weights > 0
    if not np.any(nonzero_mask):
        raise RuntimeError("All regions have zero weight - no valid diffeomorphic registration data")
    diffeo_mem[nonzero_mask] /= diffeo_weights[nonzero_mask]
    
    # Flush and cleanup
    diffeo_mem.flush()
    del diffeo_weights
    gc.collect()
    
    # Final summary
    if cuda_context_broken:
        logger.warning(f"GPU became unrecoverable during processing!")
        logger.warning(f"Crops after failure used affine-only fallback.")
    
    logger.info(f"Diffeo stage complete: {success_count} success, {fallback_count} fallback")
    return diffeo_mem, diffeo_tmp_dir, success_count, fallback_count, len(diffeo_coords)


# --------------------- Main Registration Pipeline ---------------------

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


def register_image_pair(
    reference_path: Path,
    moving_path: Path,
    output_path: Path,
    affine_crop_size: int = 2000,
    diffeo_crop_size: int = 2000,
    overlap_percent: float = 10.0,
    n_features: int = 2000,
    n_workers: int = 4,
    opt_tol: float = 1e-5,
    inv_tol: float = 1e-5,
    gpu_reset_interval: int = 50,
    qc_dir: Optional[Path] = None,
):
    """
    Register moving image to reference using affine + diffeomorphic registration.
    
    Pipeline:
    1. CPU Affine Stage: Compute affine transforms on affine_crop_size crops, merge
    2. GPU Diffeo Stage: Compute diffeomorphic transforms on diffeo_crop_size crops, merge
    """
    if not GPU_AVAILABLE:
        raise RuntimeError(
            f"GPU registration requires CuPy and cuDIPY.\nImport error: {_gpu_import_error}"
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
    
    # Validate overlap
    if not 0 <= overlap_percent < 100:
        raise ValueError(f"overlap_percent must be in [0, 100), got {overlap_percent}")
    
    # Compute overlap sizes
    affine_overlap = int(affine_crop_size * overlap_percent / 100.0)
    diffeo_overlap = int(diffeo_crop_size * overlap_percent / 100.0)
    
    logger.info(f"Overlap: {overlap_percent}%")
    logger.info(f"  Affine: crop_size={affine_crop_size}, overlap={affine_overlap}px")
    logger.info(f"  Diffeo: crop_size={diffeo_crop_size}, overlap={diffeo_overlap}px")
    
    orig_dtype = mov_mem.dtype
    affine_tmp_dir = None
    diffeo_tmp_dir = None
    
    try:
        # STAGE 1: CPU Affine
        affine_mem, affine_tmp_dir, affine_success, affine_total = run_affine_stage(
            ref_mem, mov_mem, mov_shape, orig_dtype,
            affine_crop_size, affine_overlap,
            n_features, n_workers
        )
        
        # STAGE 2: GPU Diffeomorphic
        diffeo_mem, diffeo_tmp_dir, diffeo_success, diffeo_fallback, diffeo_total = run_diffeo_stage(
            ref_mem, affine_mem, mov_shape,
            diffeo_crop_size, diffeo_overlap,
            opt_tol, inv_tol,
            gpu_reset_interval=gpu_reset_interval
        )
        
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
        logger.info("REGISTRATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Affine stage:  {affine_success}/{affine_total} crops succeeded ({100*affine_success/affine_total:.1f}%)")
        logger.info(f"Diffeo stage:  {diffeo_success}/{diffeo_total} success, {diffeo_fallback}/{diffeo_total} fallback")
        logger.info(f"Output saved:  {output_path}")
        
        # Generate QC if requested
        if qc_dir:
            logger.info(f"Generating QC outputs: {qc_dir}")
            qc_dir.mkdir(parents=True, exist_ok=True)
            qc_filename = f"{moving_path.stem}_QC_RGB.tif"
            qc_output_path = qc_dir / qc_filename
            
            try:
                create_qc_rgb_composite(reference_path, output_path, qc_output_path)
            except Exception as e:
                logger.warning(f"Failed to generate QC composite: {e}")
        
        logger.info("Registration complete")
        
    finally:
        # Cleanup temp directories
        if affine_tmp_dir:
            cleanup_memmaps(affine_tmp_dir)
        if diffeo_tmp_dir:
            cleanup_memmaps(diffeo_tmp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="GPU-based image registration with separate affine and diffeomorphic stages"
    )
    parser.add_argument("--reference", type=str, required=True, 
                        help="Path to reference image")
    parser.add_argument("--moving", type=str, required=True, 
                        help="Path to moving image to register")
    parser.add_argument("--output", type=str, required=True, 
                        help="Path to save registered image")
    parser.add_argument("--qc-dir", type=str, default=None, 
                        help="Directory to save QC outputs (optional)")
    parser.add_argument("--affine-crop-size", type=int, default=2000, 
                        help="Crop size for affine registration (default: 2000)")
    parser.add_argument("--diffeo-crop-size", type=int, default=2000, 
                        help="Crop size for diffeomorphic registration (default: 2000)")
    parser.add_argument("--overlap-percent", type=float, default=10.0, 
                        help="Overlap between crops as percentage (default: 10%%)")
    parser.add_argument("--n-features", type=int, default=2000, 
                        help="Number of features for affine registration")
    parser.add_argument("--n-workers", type=int, default=4, 
                        help="Number of parallel workers for CPU stage")
    parser.add_argument("--opt-tol", type=float, default=1e-5, 
                        help="Optimization tolerance for diffeomorphic registration")
    parser.add_argument("--inv-tol", type=float, default=1e-5, 
                        help="Inverse tolerance for diffeomorphic registration")
    parser.add_argument("--gpu-reset-interval", type=int, default=50,
                        help="Reset GPU every N crops to prevent memory fragmentation (default: 50, 0 to disable)")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        help="Logging level")
    
    # Legacy argument for backward compatibility
    parser.add_argument("--crop-size", type=int, default=None,
                        help="(Deprecated) Use --affine-crop-size and --diffeo-crop-size instead")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Handle legacy --crop-size argument
    affine_crop_size = args.affine_crop_size
    diffeo_crop_size = args.diffeo_crop_size
    
    if args.crop_size is not None:
        logger.warning("--crop-size is deprecated. Use --affine-crop-size and --diffeo-crop-size instead.")
        affine_crop_size = args.crop_size
        diffeo_crop_size = args.crop_size

    print_cuda_diagnostics()

    try:
        register_image_pair(
            reference_path=Path(args.reference),
            moving_path=Path(args.moving),
            output_path=Path(args.output),
            affine_crop_size=affine_crop_size,
            diffeo_crop_size=diffeo_crop_size,
            overlap_percent=args.overlap_percent,
            n_features=args.n_features,
            n_workers=args.n_workers,
            opt_tol=args.opt_tol,
            inv_tol=args.inv_tol,
            gpu_reset_interval=args.gpu_reset_interval,
            qc_dir=Path(args.qc_dir) if args.qc_dir else None,
        )
        return 0
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())