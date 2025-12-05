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

    tifffile.imwrite(str(output_path), rgb, photometric='rgb', compression=None, bigtiff=True)
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
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 3:
        return None, {
            "reason": "insufficient_matches",
            "detail": f"{len(matches)} matches < 3 required (ref: {len(descriptors1)} features, mov: {len(descriptors2)} features)"
        }

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.estimateAffinePartial2D(points2, points1)

    if matrix is None:
        inlier_count = 0
        return None, {
            "reason": "ransac_failed",
            "detail": f"{len(matches)} matches but RANSAC rejected all as outliers (likely non-rigid deformation or false matches)"
        }

    # Count inliers for success case
    inlier_count = np.sum(mask) if mask is not None else len(matches)
    return matrix, {"reason": "success", "detail": f"{inlier_count}/{len(matches)} inliers"}



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
        return {"status": "failed", "error": "exception", "error_detail": str(e), "crop_idx": crop_idx, "coord": coord, "matrix": None}


# --------------------- main streaming registration -----------------------

def register_image_pair(
    reference_path: Path,
    moving_path: Path,
    output_path: Path,
    crop_size: int = 2000,
    overlap_percent: float = 10.0,
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

    # Validate overlap_percent
    if overlap_percent < 0 or overlap_percent >= 100:
        raise ValueError(f"overlap_percent must be in range [0, 100), got {overlap_percent}")
    if overlap_percent < 5:
        logger.warning(f"Overlap {overlap_percent}% is very low. Recommended minimum: 10%")
    if overlap_percent > 30:
        logger.warning(f"Overlap {overlap_percent}% is very high. Recommended maximum: 20-30%. High overlap increases processing time significantly.")

    # Compute overlap size from percentage
    overlap = int(crop_size * overlap_percent / 100.0)
    logger.info(f"Using overlap: {overlap_percent}% of crop_size ({overlap} pixels)")

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
            coord = res["coord"]

            if status == "success":
                logger.info(f"CPU affine crop {idx+1}/{len(coords)} at ({coord['y']},{coord['x']}): ✓ {res.get('match_info', 'success')}")
            else:
                error_type = res.get('error', 'unknown')
                error_detail = res.get('error_detail', 'no details')
                logger.warning(f"CPU affine crop {idx+1}/{len(coords)} at ({coord['y']},{coord['x']}): ✗ {error_type} - {error_detail}")

    # Analyze and summarize CPU affine results
    successful_crops = [r for r in cpu_results if r.get("status") == "success"]
    failed_crops = [r for r in cpu_results if r.get("status") != "success"]

    logger.info("=" * 80)
    logger.info(f"CPU Affine Stage Summary: {len(successful_crops)}/{len(coords)} crops succeeded ({len(successful_crops)/len(coords)*100:.1f}%)")

    if failed_crops:
        # Categorize failures
        failure_reasons = {}
        for r in failed_crops:
            reason = r.get('error', 'unknown')
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        logger.warning(f"Failure breakdown:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            logger.warning(f"  - {reason}: {count} crops ({count/len(coords)*100:.1f}%)")

    if len(successful_crops) == 0:
        raise RuntimeError(
            f"All {len(coords)} crops failed affine registration. Cannot proceed.\n"
            f"Common causes: uniform/blank image, severe misalignment, or low contrast.\n"
            f"Try adjusting --crop-size or check input image quality."
        )

    if len(failed_crops) > len(coords) * 0.3:
        logger.warning(
            f"High failure rate: {len(failed_crops)}/{len(coords)} crops failed ({len(failed_crops)/len(coords)*100:.1f}%).\n"
            f"Registration quality may be poor. Consider checking image quality or adjusting parameters."
        )

    logger.info("=" * 80)

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
    diffeo_fallback_count = 0
    diffeo_success_count = 0
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

            diffeo_success_count += 1
            logger.info(f"Processed crop {idx+1}/{len(coords)} at ({y},{x})")

        except Exception as e:
            diffeo_fallback_count += 1
            error_msg = str(e).lower()
            if "memory" in error_msg or "cuda" in error_msg or "out of memory" in error_msg:
                logger.error(f"Diffeo failed on crop {idx} - GPU MEMORY ERROR: {e}")
                logger.error("  → Consider reducing --crop-size to use less GPU memory")
            else:
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

    # Summarize GPU diffeomorphic stage
    logger.info("=" * 80)
    logger.info(f"GPU Diffeomorphic Stage Summary:")
    logger.info(f"  - Processed: {len(successful_crops)} crops (skipped {len(failed_crops)} due to affine failure)")
    logger.info(f"  - Diffeo success: {diffeo_success_count}/{len(successful_crops)} crops")
    if diffeo_fallback_count > 0:
        logger.warning(f"  - Diffeo fallback to affine-only: {diffeo_fallback_count}/{len(successful_crops)} crops ({diffeo_fallback_count/len(successful_crops)*100:.1f}%)")
        logger.warning(f"    → {diffeo_fallback_count} crops used only affine registration (no diffeomorphic refinement)")
    logger.info("=" * 80)

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

    # Extract channel names from moving image to preserve them in output
    channel_names = []
    try:
        with tifffile.TiffFile(str(moving_path)) as tif:
            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                channels = root.findall('.//ome:Channel', ns)
                channel_names = [ch.get('Name') for ch in channels]
    except Exception as e:
        logger.warning(f"Could not extract channel names from metadata: {e}")

    # Fallback: extract from filename
    if not channel_names or len(channel_names) != out.shape[0]:
        filename = Path(moving_path).stem
        # Extract markers from filename (e.g., "B19-10215_DAPI_SMA_panck_corrected" or "01B19-10215_DAPI_FOXP3_VIMENTIN_corrected")
        name_part = filename.replace('_corrected', '').replace('_preprocessed', '').replace('_registered', '')
        parts = name_part.split('_')

        # Skip first part if it looks like a patient/sample ID (contains dash and numbers)
        if len(parts) > 1 and '-' in parts[0] and any(c.isdigit() for c in parts[0]):
            markers = parts[1:]  # Skip first part (patient ID)
        else:
            markers = parts  # No obvious patient ID, use all parts

        if len(markers) == out.shape[0]:
            channel_names = markers
        else:
            channel_names = [f"channel_{i}" for i in range(out.shape[0])]

    logger.info(f"Channel names: {channel_names}")

    # Create OME-XML metadata with channel names
    num_channels, height, width = out.shape if out.ndim == 3 else (1, out.shape[0], out.shape[1])
    if out.ndim == 2:
        out = out[np.newaxis, ...]
        channel_names = ['channel_0']

    ome_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
    <Image ID="Image:0" Name="Registered">
        <Pixels ID="Pixels:0" Type="{out.dtype.name}"
                SizeX="{width}" SizeY="{height}" SizeZ="1" SizeC="{num_channels}" SizeT="1"
                DimensionOrder="XYCZT"
                PhysicalSizeX="0.325" PhysicalSizeY="0.325" PhysicalSizeXUnit="um" PhysicalSizeYUnit="um">
            {chr(10).join(f'            <Channel ID="Channel:0:{i}" Name="{name}" SamplesPerPixel="1" />' for i, name in enumerate(channel_names))}
            <TiffData />
        </Pixels>
    </Image>
</OME>'''

    tifffile.imwrite(
        str(output_path),
        out,
        metadata={'axes': 'CYX'},
        description=ome_xml,
        compression="zlib",
        bigtiff=True
    )

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
    parser.add_argument("--overlap-percent", type=float, default=10.0, help="Overlap between crops as percentage of crop size (default: 10%%, recommended: 10-20%%)")
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
            overlap_percent=args.overlap_percent,
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
