#!/usr/bin/env python3
"""GPU-based image registration using affine + diffeomorphic transformations.

This script performs pairwise registration of a moving image to a reference image
using overlapping crops processed in parallel on CPU (affine) and sequentially on GPU (diffeo).
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tifffile
import traceback

# Keep a stable Cupy cache directory (optional)
os.environ.setdefault('CUPY_CACHE_DIR', '/tmp/.cupy')

# GPU imports with availability check
try:
    import cupy as cp
    from cudipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from cudipy.align.metrics import CCMetric
    GPU_AVAILABLE = True
except Exception as e:
    GPU_AVAILABLE = False
    _gpu_import_error = str(e)

logger = logging.getLogger(__name__)


def get_channel_names(filename: str) -> List[str]:
    """Parse channel names from filename.

    Expected format: PatientID_DAPI_Marker1_Marker2_corrected.ome.tif

    Parameters:
        filename (str): Base filename (not full path)

    Returns:
        list of str: Channel names extracted from filename (excludes Patient ID)
    """
    base = os.path.basename(filename)
    name_part = base.split('_corrected')[0]  # Remove suffix
    parts = name_part.split('_')
    channels = parts[1:]  # Exclude Patient ID
    return channels


def autoscale(img: np.ndarray, low_p: float = 1, high_p: float = 99) -> np.ndarray:
    """Auto brightness/contrast similar to ImageJ's 'Auto'.

    Parameters:
        img (ndarray): Input image
        low_p (float): Lower percentile for scaling
        high_p (float): Upper percentile for scaling

    Returns:
        ndarray: Scaled uint8 image
    """
    lo = np.percentile(img, low_p)
    hi = np.percentile(img, high_p)
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)



def create_qc_rgb_composite(
    reference_path: Path,
    registered_path: Path,
    output_path: Path,
) -> None:
    """Create QC RGB composite for visual inspection of registration quality.

    RGB channels:
        - RED: registered DAPI
        - GREEN: reference DAPI
        - BLUE: zeros

    Good alignment appears yellow (red + green), misalignment shows red/green artifacts.

    Parameters:
        reference_path (Path): Path to reference image
        registered_path (Path): Path to registered image
        output_path (Path): Path to save QC composite
    """
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

    logger.info(f"  Reference DAPI channel: {ref_dapi_idx} ({ref_channels[ref_dapi_idx] if ref_dapi_idx < len(ref_channels) else 'channel_0'})")
    logger.info(f"  Registered DAPI channel: {reg_dapi_idx} ({reg_channels[reg_dapi_idx] if reg_dapi_idx < len(reg_channels) else 'channel_0'})")

    ref_dapi = ref_img[ref_dapi_idx]
    reg_dapi = reg_img[reg_dapi_idx]

    ref_dapi_scaled = autoscale(ref_dapi)
    reg_dapi_scaled = autoscale(reg_dapi)

    h, w = reg_dapi_scaled.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = reg_dapi_scaled
    rgb[:, :, 1] = ref_dapi_scaled
    rgb[:, :, 2] = 0

    # Write uncompressed to avoid requiring imagecodecs for JPEG compression
    tifffile.imwrite(str(output_path), rgb, photometric='rgb', compression=None)
    logger.info(f"  Saved QC composite: {output_path}")


def apply_mapping(mapping, x, method="dipy"):
    """
    Apply mapping to the image.

    - 'cv2': mapping is affine 2x3, warp using OpenCV (returns numpy)
    - 'dipy': mapping.transform(x) may return numpy or cupy; convert to numpy
    """
    if method not in ["cv2", "dipy"]:
        raise ValueError("Invalid method specified. Choose either 'cv2' or 'dipy'.")

    if method == "dipy":
        mapped = mapping.transform(x)
        if hasattr(mapped, "get"):
            # cupy array -> numpy
            mapped = mapped.get()
        return mapped

    elif method == "cv2":
        height, width = x.shape[:2]
        return cv2.warpAffine(x, mapping, (width, height))


def compute_affine_mapping_cv2(y: np.ndarray, x: np.ndarray, n_features=2000):
    """
    Compute affine mapping using OpenCV.

    Parameters:
        y (ndarray): Reference image.
        x (ndarray): Moving image to be registered.
        n_features (int, optional): Maximum number of features to detect. Default is 2000.

    Returns:
        matrix (ndarray): Affine transformation matrix.
    """
    # Normalize them to 8-bit (0-255) for feature detection
    y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    orb = cv2.ORB_create(fastThreshold=0, edgeThreshold=0, nfeatures=n_features)
    keypoints1, descriptors1 = orb.detectAndCompute(y, None)
    keypoints2, descriptors2 = orb.detectAndCompute(x, None)

    if descriptors1 is None or descriptors2 is None:
        return None

    if descriptors1.dtype != np.uint8:
        descriptors1 = descriptors1.astype(np.uint8)
    if descriptors2.dtype != np.uint8:
        descriptors2 = descriptors2.astype(np.uint8)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 3:
        return None

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.estimateAffinePartial2D(points2, points1)
    return matrix


def compute_diffeomorphic_mapping_dipy(y: np.ndarray, x: np.ndarray, sigma_diff=20, radius=20):
    """
    Compute diffeomorphic mapping using DIPY.

    Parameters:
        y (ndarray): Reference image.
        x (ndarray): Moving image to be registered.
        sigma_diff (int, optional): Standard deviation for the CCMetric. Default is 20.
        radius (int, optional): Radius for the CCMetric. Default is 20.

    Returns:
        mapping: A mapping object containing the transformation information.
    """
    if y.shape != x.shape:
        raise ValueError("Reference image (y) and moving image (x) must have the same shape.")

    # CRITICAL: Ensure contiguous memory layout before GPU transfer
    # Non-contiguous arrays can cause cudaErrorIllegalAddress
    y_contiguous = np.ascontiguousarray(y, dtype=np.float32)
    x_contiguous = np.ascontiguousarray(x, dtype=np.float32)

    y_gpu = cp.asarray(y_contiguous)
    x_gpu = cp.asarray(x_contiguous)

    # Free CPU copies immediately
    del y_contiguous, x_contiguous

    # Auto-detect crop_size from the input image
    crop_size = y.shape[0]  # Assuming square crops

    # Scale parameters based on detected crop size
    scale_factor = crop_size / 2000

    # Scale radius linearly with image size
    radius = int(20 * scale_factor)

    # Scale sigma_diff with square root of scale factor (empirically better)
    sigma_diff = int(20 * np.sqrt(scale_factor))

    # Define the metric and create the Symmetric Diffeomorphic Registration object
    metric = CCMetric(2, sigma_diff=sigma_diff, radius=radius)
    sdr = SymmetricDiffeomorphicRegistration(metric, opt_tol=1e-16, inv_tol=1e-16)

    mapping = sdr.optimize(y_gpu, x_gpu)
    return mapping


def _extract_single_crop(args):
    """
    Extract a single crop from an image. Helper function for parallel extraction.

    Parameters:
        args (tuple): (image, y_start, x_start, crop_size, is_multichannel)

    Returns:
        dict: Crop dictionary with 'data', 'y', 'x', 'h', 'w'
    """
    image, y_start, x_start, y_end, x_end, is_multichannel = args

    h = y_end - y_start
    w = x_end - x_start

    if is_multichannel:
        crop_data = image[:, y_start:y_end, x_start:x_end].copy()
    else:
        crop_data = image[y_start:y_end, x_start:x_end].copy()

    return {
        "data": crop_data,
        "y": y_start,
        "x": x_start,
        "h": h,
        "w": w
    }


def extract_crops(image, crop_size, overlap, n_workers=4):
    """
    Extract overlapping crops from an image in parallel.

    Parameters:
        image (ndarray): Input image (H, W) or (C, H, W)
        crop_size (int): Size of each crop
        overlap (int): Overlap between crops in pixels
        n_workers (int): Number of parallel workers for extraction

    Returns:
        crops (list): List of crop dictionaries with 'data', 'y', 'x', 'h', 'w'
    """
    if image.ndim == 3:
        _, height, width = image.shape
        is_multichannel = True
    else:
        height, width = image.shape
        is_multichannel = False

    stride = crop_size - overlap
    crop_coords = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + crop_size, height)
            x_end = min(x + crop_size, width)
            y_start = max(0, y_end - crop_size)
            x_start = max(0, x_end - crop_size)
            crop_coords.append((image, y_start, x_start, y_end, x_end, is_multichannel))

    if n_workers > 1 and len(crop_coords) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            crops = list(executor.map(_extract_single_crop, crop_coords))
    else:
        crops = [_extract_single_crop(coords) for coords in crop_coords]

    return crops


def _create_weight_mask(h, w, overlap):
    mask = np.ones((h, w), dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap)
        mask[:overlap, :] *= ramp[:, np.newaxis]
        mask[-overlap:, :] *= ramp[::-1, np.newaxis]
        mask[:, :overlap] *= ramp[np.newaxis, :]
        mask[:, -overlap:] *= ramp[::-1][np.newaxis, :]
    return mask


def _merge_single_crop(args):
    """
    Process a single crop for merging. Helper function for parallel merging.

    Parameters:
        args (tuple): (crop, output_shape, overlap)

    Returns:
        tuple: (merged_contribution, weights_contribution, y, x, h, w)
    """
    crop, output_shape, overlap = args
    y, x = crop["y"], crop["x"]
    h, w = crop["h"], crop["w"]
    crop_data = crop["data"]

    weight_mask = _create_weight_mask(h, w, overlap)

    if len(output_shape) == 3:
        merged_local = np.zeros((output_shape[0], h, w), dtype=np.float32)
        weights_local = np.zeros((output_shape[0], h, w), dtype=np.float32)
        for c in range(output_shape[0]):
            merged_local[c] = crop_data[c] * weight_mask
            weights_local[c] = weight_mask
    else:
        merged_local = crop_data * weight_mask
        weights_local = weight_mask

    return (merged_local, weights_local, y, x, h, w)


def merge_crops(crops, output_shape, overlap, n_workers=4):
    """
    Merge overlapping crops back into a full image using weighted averaging in overlap regions.

    Parameters:
        crops (list): List of crop dictionaries with 'data', 'y', 'x', 'h', 'w'
        output_shape (tuple): Shape of output image (C, H, W) or (H, W)
        overlap (int): Overlap between crops in pixels
        n_workers (int): Number of parallel workers for merging

    Returns:
        merged (ndarray): Merged image
    """
    # Initialize output arrays
    if len(output_shape) == 3:
        merged = np.zeros(output_shape, dtype=np.float32)
        weights = np.zeros(output_shape, dtype=np.float32)
    else:
        merged = np.zeros(output_shape, dtype=np.float32)
        weights = np.zeros(output_shape, dtype=np.float32)

    merge_args = [(crop, output_shape, overlap) for crop in crops]

    if n_workers > 1 and len(crops) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_merge_single_crop, merge_args))
    else:
        results = [_merge_single_crop(args) for args in merge_args]

    for merged_local, weights_local, y, x, h, w in results:
        if len(output_shape) == 3:
            merged[:, y : y + h, x : x + w] += merged_local
            weights[:, y : y + h, x : x + w] += weights_local
        else:
            merged[y : y + h, x : x + w] += merged_local
            weights[y : y + h, x : x + w] += weights_local

    weights[weights == 0] = 1
    merged /= weights

    return merged


def process_crop_cpu_stage(crop_idx: int, ref_crop: Dict, mov_crop: Dict, n_features: int) -> Dict:
    try:
        ref_2d = ref_crop["data"][0].astype(np.float32)
        mov_2d = mov_crop["data"][0].astype(np.float32)

        affine_matrix = compute_affine_mapping_cv2(ref_2d, mov_2d, n_features=n_features)
        if affine_matrix is None:
            raise ValueError(f"Affine transformation failed for crop {crop_idx}. Not enough matching features.")

        mov_affine = np.zeros_like(mov_crop["data"], dtype=np.float32)
        for c in range(mov_crop["data"].shape[0]):
            mov_affine[c] = apply_mapping(affine_matrix, mov_crop["data"][c].astype(np.float32), method="cv2")

        return {
            "status": "success",
            "mov_affine": mov_affine,
            "ref_2d": ref_2d,
            "mov_affine_2d": mov_affine[0].astype(np.float32),
            "y": ref_crop["y"],
            "x": ref_crop["x"],
            "h": ref_crop["h"],
            "w": ref_crop["w"],
            "crop_idx": crop_idx,
        }
    except Exception as e:
        logger.error(f"CPU stage failed for crop {crop_idx}: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "mov_affine": mov_crop["data"],
            "y": mov_crop["y"],
            "x": mov_crop["x"],
            "h": mov_crop["h"],
            "w": mov_crop["w"],
            "crop_idx": crop_idx,
        }


def register_image_pair(
    reference_path: Path,
    moving_path: Path,
    output_path: Path,
    crop_size: int = 2000,
    overlap: int = 200,
    n_features: int = 2000,
    n_workers: int = 4,
    qc_dir: Optional[Path] = None,
):
    if not GPU_AVAILABLE:
        raise RuntimeError(
            f"GPU registration requires CuPy and cuDIPY libraries.\n"
            f"Import error: {_gpu_import_error}\n"
            f"Please install with: pip install cupy-cuda12x cudipy"
        )

    logger.info(f"Loading reference image: {reference_path}")
    ref_img = tifffile.imread(str(reference_path))

    logger.info(f"Loading moving image: {moving_path}")
    mov_img = tifffile.imread(str(moving_path))

    logger.info(f"Reference shape: {ref_img.shape}, Moving shape: {mov_img.shape}")

    if ref_img.ndim == 2:
        ref_img = ref_img[np.newaxis, ...]
    if mov_img.ndim == 2:
        mov_img = mov_img[np.newaxis, ...]

    # Validate spatial dimensions match (channels may differ)
    if ref_img.shape[1:] != mov_img.shape[1:]:
        raise ValueError(
            f"Spatial dimension mismatch: reference {ref_img.shape[1:]} != moving {mov_img.shape[1:]}. "
            f"Images should be pre-padded to the same spatial dimensions by PAD_IMAGES process."
        )

    # Warn if channel counts differ
    if ref_img.shape[0] != mov_img.shape[0]:
        logger.warning(
            f"Channel count mismatch: reference has {ref_img.shape[0]} channels, "
            f"moving has {mov_img.shape[0]} channels. Registration will use first channel only."
        )

    logger.info(f"Reference shape: {ref_img.shape}, Moving shape: {mov_img.shape}")
    logger.info(f"Spatial dimensions match: {ref_img.shape[1:]} (H, W)")

    logger.info(f"Extracting crops with size={crop_size}, overlap={overlap}")
    logger.info(f"Using {n_workers} workers for parallel cropping")
    ref_crops = extract_crops(ref_img, crop_size, overlap, n_workers=n_workers)
    mov_crops = extract_crops(mov_img, crop_size, overlap, n_workers=n_workers)

    if len(ref_crops) != len(mov_crops):
        raise ValueError(f"Number of crops mismatch: ref={len(ref_crops)}, mov={len(mov_crops)}")

    logger.info(f"Extracted {len(ref_crops)} crops from each image")
    logger.info(f"Processing crops in two stages: CPU-parallel (affine) -> GPU-sequential (diffeo)")

    # Log cupy/runtime information
    try:
        logger.info("cupy %s", cp.__version__)
        try:
            logger.info("CUDA runtime version: %s", cp.cuda.runtime.runtimeGetVersion())
        except Exception:
            pass
    except Exception:
        pass

    # STEP A: CPU-only parallel stage (affine compute + apply_affine)
    logger.info("STEP A: running CPU-only affine stage in parallel...")
    cpu_stage_results = [None] * len(ref_crops)
    cpu_func = partial(process_crop_cpu_stage, n_features=n_features)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(cpu_func, i, ref_crops[i], mov_crops[i]): i for i in range(len(ref_crops))}
        for future in as_completed(futures):
            res = future.result()
            idx = res["crop_idx"]
            cpu_stage_results[idx] = res
            status = res.get("status", "failed")
            position = (res["y"], res["x"])
            logger.info(f"CPU stage crop {idx+1}/{len(ref_crops)} at {position}: {status}")
            if status == "failed":
                logger.warning(f"  CPU stage error: {res.get('error')}")

    # STEP B: Sequential GPU diffeomorphic registration and mapping (single-threaded)
    logger.info("STEP B: running GPU diffeomorphic registration sequentially (one crop at a time)...")
    registered_crops = [None] * len(ref_crops)

    # Reset GPU memory pool before starting to ensure clean state
    try:
        device = cp.cuda.Device()
        device.synchronize()
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        logger.info("Cleared GPU memory pools before processing")
        logger.info(f"GPU memory: {mempool.used_bytes() / 1e9:.2f} GB used, {mempool.total_bytes() / 1e9:.2f} GB total")
    except Exception as e:
        logger.warning(f"Could not clear GPU memory pool: {e}")

    for i, cpu_res in enumerate(cpu_stage_results):
        crop_idx = cpu_res["crop_idx"]
        y, x, h, w = cpu_res["y"], cpu_res["x"], cpu_res["h"], cpu_res["w"]

        if cpu_res.get("status") != "success":
            logger.warning(f"Skipping GPU stage for crop {crop_idx} due to CPU-stage failure.")
            registered_crops[crop_idx] = {
                "data": cpu_res["mov_affine"],
                "y": y,
                "x": x,
                "h": h,
                "w": w,
                "status": "failed",
                "crop_idx": crop_idx,
                "error": cpu_res.get("error"),
            }
            continue

        try:
            ref_2d = cpu_res["ref_2d"]
            mov_affine = cpu_res["mov_affine"]
            mov_affine_2d = cpu_res["mov_affine_2d"]

            # Aggressive pre-transfer cleanup
            try:
                cp.cuda.get_device().synchronize()
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

            # Compute mapping on GPU (cudipy)
            # Pass CPU arrays directly - compute_diffeomorphic_mapping_dipy handles GPU transfer
            diffeo_mapping = compute_diffeomorphic_mapping_dipy(ref_2d, mov_affine_2d)

            # Aggressive post-computation cleanup
            try:
                cp.cuda.get_device().synchronize()
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

            # Apply diffeo to all channels (mapping.transform may return cupy or numpy)
            registered_data = np.zeros_like(mov_affine, dtype=np.float32)
            for c in range(mov_affine.shape[0]):
                # Ensure channel slice is contiguous before GPU transform
                channel_data = np.ascontiguousarray(mov_affine[c], dtype=np.float32)
                result = apply_mapping(diffeo_mapping, channel_data, method="dipy")
                registered_data[c] = result
                # Free intermediate arrays
                del channel_data
                if hasattr(result, 'get'):
                    del result

            # Delete mapping object to free GPU memory
            del diffeo_mapping

            # Aggressive post-processing cleanup
            try:
                cp.cuda.get_device().synchronize()
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

            registered_crops[crop_idx] = {
                "data": registered_data,
                "y": y,
                "x": x,
                "h": h,
                "w": w,
                "status": "success",
                "crop_idx": crop_idx,
            }
            logger.info(f"GPU stage crop {crop_idx+1}/{len(ref_crops)} at ({y},{x}): success")

            # Every 10 crops, do extra aggressive cleanup and report memory
            if (crop_idx + 1) % 10 == 0:
                try:
                    device.synchronize()
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
                    logger.info(f"[Crop {crop_idx+1}] GPU memory: {mempool.used_bytes() / 1e9:.2f} GB used, {mempool.total_bytes() / 1e9:.2f} GB total")
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Failed to register crop {crop_idx}: {e}")
            logger.error(traceback.format_exc())

            # Aggressive error cleanup
            try:
                cp.cuda.get_device().synchronize()
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

            # Check if this is a recoverable GPU error
            error_str = str(e).lower()
            if 'cuda' in error_str and ('memory' in error_str or 'illegal' in error_str):
                logger.warning(f"GPU memory error detected - attempting GPU reset")
                try:
                    # Try to reset GPU context
                    cp.cuda.Device().synchronize()
                    cp.get_default_memory_pool().free_all_blocks()

                    # If we've processed more than 50% of crops, continue
                    if crop_idx > len(ref_crops) // 2:
                        logger.info("Processed majority of crops - continuing with remaining")

                except Exception as reset_err:
                    logger.error(f"GPU reset failed: {reset_err}")

            # fallback to CPU-affined data
            registered_crops[crop_idx] = {
                "data": cpu_res["mov_affine"],
                "y": y,
                "x": x,
                "h": h,
                "w": w,
                "status": "failed",
                "crop_idx": crop_idx,
                "error": str(e),
            }

    # Remove status fields before merging
    for crop in registered_crops:
        if isinstance(crop, dict):
            crop.pop("status", None)
            crop.pop("crop_idx", None)
            crop.pop("error", None)

    logger.info(f"Merging {len(registered_crops)} registered crops with {n_workers} workers...")
    registered_img = merge_crops(registered_crops, mov_img.shape, overlap, n_workers=n_workers)

    # Convert back to original dtype
    if mov_img.dtype == np.uint16:
        registered_img = np.clip(registered_img, 0, 65535).astype(np.uint16)
    elif mov_img.dtype == np.uint8:
        registered_img = np.clip(registered_img, 0, 255).astype(np.uint8)
    else:
        registered_img = registered_img.astype(mov_img.dtype)

    logger.info(f"Saving registered image to: {output_path}")
    tifffile.imwrite(str(output_path), registered_img, photometric="minisblack", compression="zlib")

    if qc_dir:
        logger.info(f"\nGenerating QC outputs to: {qc_dir}")
        qc_dir.mkdir(parents=True, exist_ok=True)
        qc_filename = f"{moving_path.stem}_QC_RGB.tif"
        qc_output_path = qc_dir / qc_filename
        try:
            create_qc_rgb_composite(
                reference_path=reference_path,
                registered_path=Path(output_path),
                output_path=qc_output_path,
            )
            logger.info(f"QC composite saved successfully")
        except Exception as e:
            logger.warning(f"Failed to generate QC composite: {e}")
            traceback.print_exc()

    logger.info("Registration complete")


def main():
    parser = argparse.ArgumentParser(description="GPU-based image registration with affine + diffeomorphic")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference image")
    parser.add_argument("--moving", type=str, required=True, help="Path to moving image to register")
    parser.add_argument("--output", type=str, required=True, help="Path to save registered image")
    parser.add_argument("--qc-dir", type=str, default=None, help="Directory to save QC outputs (optional)")
    parser.add_argument("--crop-size", type=int, default=2000, help="Size of crops for processing")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between crops in pixels")
    parser.add_argument("--n-features", type=int, default=2000, help="Number of features for affine registration")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        register_image_pair(
            reference_path=Path(args.reference),
            moving_path=Path(args.moving),
            output_path=Path(args.output),
            crop_size=args.crop_size,
            overlap=args.overlap,
            n_features=args.n_features,
            n_workers=args.n_workers,
            qc_dir=Path(args.qc_dir) if args.qc_dir else None,
        )
        return 0
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
