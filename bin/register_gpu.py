#!/usr/bin/env python3
"""GPU-based image registration (refactored streaming version).

Optimizations applied:
- extract_crops now returns only coordinates (y, x, h, w) — no .data copies
- CPU stage computes and stores only affine matrices (6 floats) per crop
- GPU stage loads crop data on-demand, applies affine, computes diffeo, applies transform
- Merging is streamed into on-disk numpy.memmap accumulators (merged + weights)
- Avoids building large result lists; processes results as generators/streams

Usage: same CLI as original script. Requires CuPy + cuDIPY for GPU diffeo.
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
import tempfile
import math
import gc
import traceback

# Keep a stable Cupy cache directory (optional)
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

logger = logging.getLogger(__name__)


# ------------------------- small helpers ---------------------------------

def get_channel_names(filename: str) -> List[str]:
    base = os.path.basename(filename)
    name_part = base.split('_corrected')[0]
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
    rgb[:, :, 0] = reg_dapi_scaled
    rgb[:, :, 1] = ref_dapi_scaled
    rgb[:, :, 2] = 0

    tifffile.imwrite(str(output_path), rgb, photometric='rgb', compression=None)
    logger.info(f"  Saved QC composite: {output_path}")


def apply_mapping(mapping, x, method="dipy"):
    if method not in ["cv2", "dipy"]:
        raise ValueError("Invalid method specified. Choose either 'cv2' or 'dipy'.")

    if method == "dipy":
        mapped = mapping.transform(x)
        if hasattr(mapped, "get"):
            mapped = mapped.get()
        return mapped
    elif method == "cv2":
        height, width = x.shape[:2]
        return cv2.warpAffine(x, mapping, (width, height))


def compute_affine_mapping_cv2(y: np.ndarray, x: np.ndarray, n_features=2000):
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
    if y.shape != x.shape:
        raise ValueError("Reference image (y) and moving image (x) must have the same shape.")

    y_contiguous = np.ascontiguousarray(y, dtype=np.float32)
    x_contiguous = np.ascontiguousarray(x, dtype=np.float32)

    y_gpu = cp.asarray(y_contiguous)
    x_gpu = cp.asarray(x_contiguous)

    del y_contiguous, x_contiguous

    crop_size = y.shape[0]
    scale_factor = crop_size / 2000
    radius = int(20 * scale_factor)
    sigma_diff = int(20 * np.sqrt(scale_factor))

    metric = CCMetric(2, sigma_diff=sigma_diff, radius=radius)
    sdr = SymmetricDiffeomorphicRegistration(metric, opt_tol=1e-16, inv_tol=1e-16)

    mapping = sdr.optimize(y_gpu, x_gpu)
    return mapping


# --------------------- streaming crop utilities ---------------------------

def extract_crop_coords(image_shape: Tuple[int, ...], crop_size: int, overlap: int) -> List[Dict]:
    """Return list of coordinate dicts (y,x,h,w) for the image shape.

    image_shape is either (C, H, W) or (H, W)
    """
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


def _create_weight_mask(h, w, overlap):
    mask = np.ones((h, w), dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap)
        mask[:overlap, :] *= ramp[:, np.newaxis]
        mask[-overlap:, :] *= ramp[::-1, np.newaxis]
        mask[:, :overlap] *= ramp[np.newaxis, :]
        mask[:, -overlap:] *= ramp[::-1][np.newaxis, :]
    return mask


def _create_memmaps_for_merge(output_shape, dtype=np.float32):
    tmp_dir = Path(tempfile.mkdtemp(prefix="reg_merge_"))
    if len(output_shape) == 3:
        c, H, W = output_shape
        merged_path = tmp_dir / "merged.npy"
        weights_path = tmp_dir / "weights.npy"
        merged = np.memmap(str(merged_path), dtype=dtype, mode="w+", shape=(c, H, W))
        weights = np.memmap(str(weights_path), dtype=dtype, mode="w+", shape=(c, H, W))
    else:
        H, W = output_shape
        merged_path = tmp_dir / "merged.npy"
        weights_path = tmp_dir / "weights.npy"
        merged = np.memmap(str(merged_path), dtype=dtype, mode="w+", shape=(H, W))
        weights = np.memmap(str(weights_path), dtype=dtype, mode="w+", shape=(H, W))
    merged[:] = 0
    weights[:] = 0
    return merged, weights, tmp_dir


# --------------------- CPU stage: affine-only -----------------------------

def process_crop_cpu_stage_affine_only(crop_idx: int, ref_mem, mov_mem, coord: Dict, n_features: int) -> Dict:
    """Read only the channel-0 crop on demand and compute affine matrix.

    ref_mem/mov_mem should be array-like (support slicing). Returns small dict with matrix.
    """
    y, x, h, w = coord["y"], coord["x"], coord["h"], coord["w"]
    try:
        # read slice lazily
        if ref_mem.ndim == 3:
            ref_crop = ref_mem[0, y : y + h, x : x + w]
        else:
            ref_crop = ref_mem[y : y + h, x : x + w]
        if mov_mem.ndim == 3:
            mov_crop = mov_mem[0, y : y + h, x : x + w]
        else:
            mov_crop = mov_mem[y : y + h, x : x + w]

        ref_2d = np.ascontiguousarray(ref_crop.astype(np.float32))
        mov_2d = np.ascontiguousarray(mov_crop.astype(np.float32))

        matrix = compute_affine_mapping_cv2(ref_2d, mov_2d, n_features=n_features)
        if matrix is None:
            return {"status": "failed", "error": "affine_fail", "crop_idx": crop_idx, "coord": coord, "matrix": None}
        return {"status": "success", "matrix": matrix, "crop_idx": crop_idx, "coord": coord}
    except Exception as e:
        return {"status": "failed", "error": str(e), "crop_idx": crop_idx, "coord": coord, "matrix": None}


# --------------------- main streaming registration -----------------------

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
            f"GPU registration requires CuPy and cuDIPY libraries.\nImport error: {_gpu_import_error}\nPlease install with: pip install cupy-cuda12x cudipy"
        )

    logger.info(f"Opening reference (memmap): {reference_path}")
    ref_mem = tifffile.memmap(str(reference_path))
    logger.info(f"Opening moving (memmap): {moving_path}")
    mov_mem = tifffile.memmap(str(moving_path))

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

    # Validate spatial dimensions match
    if ref_shape[1:] != mov_shape[1:]:
        raise ValueError(f"Spatial dimension mismatch: reference {ref_shape[1:]} != moving {mov_shape[1:]}")

    if ref_shape[0] != mov_shape[0]:
        logger.warning(f"Channel count mismatch: reference has {ref_shape[0]}, moving has {mov_shape[0]}. Using all channels but metric uses first channel.")

    # Create list of crop coords only
    logger.info(f"Extracting crop coordinates with size={crop_size}, overlap={overlap}")
    coords = extract_crop_coords(ref_shape, crop_size, overlap)
    logger.info(f"Total crops: {len(coords)}")

    # STEP A: CPU stage — compute affine matrices only (parallel)
    logger.info("STEP A: CPU affine-only stage — computing matrices (no big copies)")
    cpu_results = [None] * len(coords)
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(process_crop_cpu_stage_affine_only, i, ref_mem, mov_mem, coords[i], n_features): i for i in range(len(coords))}
        for future in as_completed(futures):
            res = future.result()
            idx = res["crop_idx"]
            cpu_results[idx] = res
            status = res.get("status", "failed")
            logger.info(f"CPU affine crop {idx+1}/{len(coords)}: {status}")
            if status != "success":
                logger.warning(f"  CPU affine error on crop {idx}: {res.get('error')}")

    # Prepare memmaps for merged accumulation
    merged_mem, weights_mem, tmp_dir = _create_memmaps_for_merge(mov_shape, dtype=np.float32)
    logger.info(f"Created memmaps for merge in {tmp_dir}")

    # Reset GPU memory pools
    try:
        device = cp.cuda.Device()
        device.synchronize()
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        logger.info("Cleared GPU memory pools before processing")
    except Exception as e:
        logger.warning(f"Could not clear GPU memory pool: {e}")

    # STEP B: Sequential GPU stage — process one crop at a time, write to memmap
    logger.info("STEP B: Sequential GPU diffeo stage (streaming)")
    weight_cache = {}
    for idx, cres in enumerate(cpu_results):
        coord = cres["coord"]
        y, x, h, w = coord["y"], coord["x"], coord["h"], coord["w"]

        if cres.get("status") != "success":
            logger.warning(f"Skipping crop {idx} due to CPU affine failure")
            continue

        matrix = cres["matrix"]

        # read moving crop (all channels) on demand
        if mov_mem.ndim == 3:
            mov_crop = mov_mem[:, y : y + h, x : x + w].astype(np.float32)
        else:
            mov_crop = mov_mem[y : y + h, x : x + w].astype(np.float32)

        # apply affine per-channel (cv2.warpAffine expects 2D)
        if mov_crop.ndim == 3:
            mov_affine = np.zeros_like(mov_crop, dtype=np.float32)
            for c in range(mov_crop.shape[0]):
                mov_affine[c] = apply_mapping(matrix, np.ascontiguousarray(mov_crop[c].astype(np.float32)), method="cv2")
        else:
            mov_affine = apply_mapping(matrix, np.ascontiguousarray(mov_crop.astype(np.float32)), method="cv2")

        # drop mov_crop ASAP
        del mov_crop
        gc.collect()

        # run diffeo on the first channel and apply mapping to all channels
        try:
            if ref_mem.ndim == 3:
                ref_crop = ref_mem[0, y : y + h, x : x + w].astype(np.float32)
            else:
                ref_crop = ref_mem[y : y + h, x : x + w].astype(np.float32)

            mapping = compute_diffeomorphic_mapping_dipy(ref_crop, mov_affine[0] if mov_affine.ndim == 3 else mov_affine)

            # apply mapping channel-wise and accumulate into memmaps using weight mask
            weight_mask = weight_cache.get((h, w))
            if weight_mask is None:
                weight_mask = _create_weight_mask(h, w, overlap).astype(np.float32)
                weight_cache[(h, w)] = weight_mask

            if mov_affine.ndim == 3:
                for c in range(mov_affine.shape[0]):
                    channel = np.ascontiguousarray(mov_affine[c].astype(np.float32))
                    mapped = apply_mapping(mapping, channel, method="dipy")
                    if hasattr(mapped, "get"):
                        mapped = mapped.get()
                    merged_mem[c, y : y + h, x : x + w] += mapped * weight_mask
                    weights_mem[c, y : y + h, x : x + w] += weight_mask
                    del channel
                    del mapped
                    gc.collect()
            else:
                channel = np.ascontiguousarray(mov_affine.astype(np.float32))
                mapped = apply_mapping(mapping, channel, method="dipy")
                if hasattr(mapped, "get"):
                    mapped = mapped.get()
                merged_mem[y : y + h, x : x + w] += mapped * weight_mask
                weights_mem[y : y + h, x : x + w] += weight_mask
                del channel
                del mapped
                gc.collect()

            # cleanup mapping and mov_affine
            del mapping
            del mov_affine
            gc.collect()

            # free cupy pools
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

            logger.info(f"Processed crop {idx+1}/{len(coords)} at ({y},{x})")

        except Exception as e:
            logger.error(f"Diffeo failed on crop {idx}: {e}")
            logger.error(traceback.format_exc())
            # fallback: accumulate affine result (no diffeo)
            weight_mask = weight_cache.get((h, w))
            if weight_mask is None:
                weight_mask = _create_weight_mask(h, w, overlap).astype(np.float32)
                weight_cache[(h, w)] = weight_mask
            if mov_affine.ndim == 3:
                for c in range(mov_affine.shape[0]):
                    merged_mem[c, y : y + h, x : x + w] += mov_affine[c] * weight_mask
                    weights_mem[c, y : y + h, x : x + w] += weight_mask
            else:
                merged_mem[y : y + h, x : x + w] += mov_affine * weight_mask
                weights_mem[y : y + h, x : x + w] += weight_mask
            del mov_affine
            gc.collect()

    # finalize merge: avoid division by zero
    weights_nonzero = weights_mem.copy()
    weights_nonzero[weights_nonzero == 0] = 1.0
    merged_mem[:] = merged_mem / weights_nonzero

    # cast back to original dtype
    final = merged_mem
    # if original moving file dtype is not float32, cast back
    orig_dtype = mov_mem.dtype
    if orig_dtype == np.uint16:
        out = np.clip(final, 0, 65535).astype(np.uint16)
    elif orig_dtype == np.uint8:
        out = np.clip(final, 0, 255).astype(np.uint8)
    else:
        out = final.astype(orig_dtype)

    logger.info(f"Saving registered image to: {output_path}")
    tifffile.imwrite(str(output_path), out, photometric="minisblack", compression="zlib")

    # optional QC
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
