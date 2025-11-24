#!/usr/bin/env python3
"""GPU-based image registration using affine + diffeomorphic transformations.

This script performs pairwise registration of a moving image to a reference image
using overlapping crops processed in parallel on both CPU and GPU.
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

# GPU imports with availability check
try:
    import cupy as cp
    from cudipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from cudipy.align.metrics import CCMetric
    GPU_AVAILABLE = True
except ImportError as e:
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


def pad_to_reference(moving_img: np.ndarray, reference_shape: Tuple[int, ...], mode: str = 'constant') -> np.ndarray:
    """Pad moving image to match reference image dimensions.

    Uses symmetric padding to center the moving image within the reference dimensions.
    This ensures proper spatial alignment during registration.

    Parameters:
        moving_img (ndarray): Moving image in (C, H, W) format
        reference_shape (tuple): Target shape (C, H, W)
        mode (str): Padding mode - 'constant' (zeros), 'edge' (replicate edges),
                   'reflect' (mirror), or 'symmetric'. Default: 'constant'

    Returns:
        ndarray: Padded image with shape matching reference_shape

    Raises:
        ValueError: If moving image is larger than reference in any dimension
    """
    c_mov, h_mov, w_mov = moving_img.shape
    c_ref, h_ref, w_ref = reference_shape

    # Validate channel counts match
    if c_mov != c_ref:
        raise ValueError(f"Channel mismatch: moving has {c_mov} channels, reference has {c_ref}")

    # Check if moving is larger than reference
    if h_mov > h_ref or w_mov > w_ref:
        raise ValueError(
            f"Moving image ({h_mov}x{w_mov}) is larger than reference ({h_ref}x{w_ref}). "
            f"Cannot pad a larger image to smaller dimensions. Consider cropping instead."
        )

    # Calculate padding needed
    pad_h = h_ref - h_mov
    pad_w = w_ref - w_mov

    # No padding needed
    if pad_h == 0 and pad_w == 0:
        logger.info("  No padding required - dimensions match")
        return moving_img

    # Calculate symmetric padding (distribute padding evenly on both sides)
    pad_h_before = pad_h // 2
    pad_h_after = pad_h - pad_h_before
    pad_w_before = pad_w // 2
    pad_w_after = pad_w - pad_w_before

    logger.info(f"  Padding moving image from ({c_mov}, {h_mov}, {w_mov}) to ({c_ref}, {h_ref}, {w_ref})")
    logger.info(f"  Padding strategy: {mode}")
    logger.info(f"    Height: {pad_h_before} pixels before, {pad_h_after} pixels after")
    logger.info(f"    Width: {pad_w_before} pixels before, {pad_w_after} pixels after")

    # Create padding specification: ((before, after), ...) for each dimension
    # Format: ((C_before, C_after), (H_before, H_after), (W_before, W_after))
    pad_width = (
        (0, 0),                           # No padding on channel dimension
        (pad_h_before, pad_h_after),      # Height padding
        (pad_w_before, pad_w_after),      # Width padding
    )

    # Apply padding
    if mode == 'constant':
        # Pad with zeros (most common for microscopy images)
        padded = np.pad(moving_img, pad_width, mode='constant', constant_values=0)
    else:
        # Other padding modes (edge, reflect, symmetric)
        padded = np.pad(moving_img, pad_width, mode=mode)

    logger.info(f"  Padded image shape: {padded.shape}")

    return padded


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

    # Load images
    ref_img = tifffile.imread(str(reference_path))
    reg_img = tifffile.imread(str(registered_path))

    # Ensure (C, H, W) format
    if ref_img.ndim == 2:
        ref_img = ref_img[np.newaxis, ...]
    if reg_img.ndim == 2:
        reg_img = reg_img[np.newaxis, ...]

    # Get channel names to find DAPI
    ref_channels = get_channel_names(reference_path.name)
    reg_channels = get_channel_names(registered_path.name)

    # Find DAPI channel indices
    ref_dapi_idx = next((i for i, ch in enumerate(ref_channels) if "DAPI" in ch.upper()), 0)
    reg_dapi_idx = next((i for i, ch in enumerate(reg_channels) if "DAPI" in ch.upper()), 0)

    logger.info(f"  Reference DAPI channel: {ref_dapi_idx} ({ref_channels[ref_dapi_idx] if ref_dapi_idx < len(ref_channels) else 'channel_0'})")
    logger.info(f"  Registered DAPI channel: {reg_dapi_idx} ({reg_channels[reg_dapi_idx] if reg_dapi_idx < len(reg_channels) else 'channel_0'})")

    # Extract DAPI channels
    ref_dapi = ref_img[ref_dapi_idx]
    reg_dapi = reg_img[reg_dapi_idx]

    # Autoscale
    ref_dapi_scaled = autoscale(ref_dapi)
    reg_dapi_scaled = autoscale(reg_dapi)

    # Create RGB composite (R=registered, G=reference, B=0)
    h, w = reg_dapi_scaled.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = reg_dapi_scaled  # Red = registered
    rgb[:, :, 1] = ref_dapi_scaled  # Green = reference
    rgb[:, :, 2] = 0                # Blue = zeros

    # Save with JPEG compression for smaller file size
    tifffile.imwrite(str(output_path), rgb, photometric='rgb', compression='jpeg')
    logger.info(f"  Saved QC composite: {output_path}")


def apply_mapping(mapping, x, method="dipy"):
    """
    Apply mapping to the image.

    Parameters:
        mapping: A mapping object from either the DIPY or the OpenCV package.
        x (ndarray): 2-dimensional numpy array to transform.
        method (str, optional): Method used for mapping. Either 'cv2' or 'dipy'. Default is 'dipy'.

    Returns:
        mapped (ndarray): Transformed image as a 2D numpy array.
    """
    if method not in ["cv2", "dipy"]:
        raise ValueError("Invalid method specified. Choose either 'cv2' or 'dipy'.")

    if method == "dipy":
        mapped = mapping.transform(x).get()
    elif method == "cv2":
        height, width = x.shape[:2]
        mapped = cv2.warpAffine(x, mapping, (width, height))

    return mapped


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

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(fastThreshold=0, edgeThreshold=0, nfeatures=n_features)

    # Compute keypoints and descriptors for both images
    keypoints1, descriptors1 = orb.detectAndCompute(y, None)
    keypoints2, descriptors2 = orb.detectAndCompute(x, None)

    if descriptors1 is None:
        raise ValueError("Object 'descriptors1' is None")
    elif descriptors2 is None:
        raise ValueError("Object 'descriptors2' is None")

    # Convert descriptors to uint8 if they are not already in that format
    if descriptors1 is not None and descriptors1.dtype != np.uint8:
        descriptors1 = descriptors1.astype(np.uint8)

    if descriptors2 is not None and descriptors2.dtype != np.uint8:
        descriptors2 = descriptors2.astype(np.uint8)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute affine transformation matrix from matched points
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
    # Check if both images have the same shape
    if y.shape != x.shape:
        raise ValueError("Reference image (y) and moving image (x) must have the same shape.")

    y_gpu = cp.asarray(y)
    x_gpu = cp.asarray(x)

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
    sdr = SymmetricDiffeomorphicRegistration(metric, opt_tol=1e-12, inv_tol=1e-12)

    # Perform the diffeomorphic registration using the pre-alignment from affine registration
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

    # Extract crop
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
        # Multi-channel image (C, H, W)
        _, height, width = image.shape
        is_multichannel = True
    else:
        # Single channel (H, W)
        height, width = image.shape
        is_multichannel = False

    stride = crop_size - overlap

    # Generate crop coordinates
    crop_coords = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Calculate crop boundaries
            y_end = min(y + crop_size, height)
            x_end = min(x + crop_size, width)

            # Adjust start position if crop would be too small at edge
            y_start = max(0, y_end - crop_size)
            x_start = max(0, x_end - crop_size)

            crop_coords.append((image, y_start, x_start, y_end, x_end, is_multichannel))

    # Extract crops in parallel
    if n_workers > 1 and len(crop_coords) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            crops = list(executor.map(_extract_single_crop, crop_coords))
    else:
        # Sequential fallback
        crops = [_extract_single_crop(coords) for coords in crop_coords]

    return crops


def _create_weight_mask(h, w, overlap):
    """Create a weight mask with linear blending in overlap regions."""
    mask = np.ones((h, w), dtype=np.float32)

    if overlap > 0:
        # Create linear ramp for blending
        ramp = np.linspace(0, 1, overlap)

        # Apply ramps to edges
        mask[:overlap, :] *= ramp[:, np.newaxis]  # Top edge
        mask[-overlap:, :] *= ramp[::-1, np.newaxis]  # Bottom edge
        mask[:, :overlap] *= ramp[np.newaxis, :]  # Left edge
        mask[:, -overlap:] *= ramp[::-1][np.newaxis, :]  # Right edge

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

    # Create weight mask
    weight_mask = _create_weight_mask(h, w, overlap)

    # Create local arrays for this crop's contribution
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

    # Prepare arguments for parallel processing
    merge_args = [(crop, output_shape, overlap) for crop in crops]

    # Process crops in parallel
    if n_workers > 1 and len(crops) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_merge_single_crop, merge_args))
    else:
        # Sequential fallback
        results = [_merge_single_crop(args) for args in merge_args]

    # Accumulate results into output arrays
    for merged_local, weights_local, y, x, h, w in results:
        if len(output_shape) == 3:
            merged[:, y : y + h, x : x + w] += merged_local
            weights[:, y : y + h, x : x + w] += weights_local
        else:
            merged[y : y + h, x : x + w] += merged_local
            weights[y : y + h, x : x + w] += weights_local

    # Normalize by weights
    weights[weights == 0] = 1  # Avoid division by zero
    merged /= weights

    return merged


def process_crop_pair(
    crop_idx: int,
    ref_crop: Dict,
    mov_crop: Dict,
    n_features: int,
) -> Dict:
    """
    Process a single crop pair: compute transformations and apply to all channels.

    This function is designed to run in parallel across multiple CPU cores.
    Transformations are computed using ONLY the first channel.

    Parameters:
        crop_idx (int): Crop index for logging
        ref_crop (Dict): Reference crop with 'data', 'y', 'x', 'h', 'w'
        mov_crop (Dict): Moving crop with 'data', 'y', 'x', 'h', 'w'
        n_features (int): Number of ORB features for affine registration

    Returns:
        Dict: Registered crop with 'data', 'y', 'x', 'h', 'w'
    """
    try:
        # ALWAYS use first channel only for computing transformations
        ref_2d = ref_crop["data"][0].astype(np.float32)
        mov_2d = mov_crop["data"][0].astype(np.float32)

        # Step 1: Compute affine transformation using first channel only
        affine_matrix = compute_affine_mapping_cv2(ref_2d, mov_2d, n_features=n_features)

        # Check if affine estimation failed
        if affine_matrix is None:
            raise ValueError(f"Affine transformation failed for crop {crop_idx}. Not enough matching features found.")

        # Step 2: Apply affine to all channels in parallel
        mov_affine = np.zeros_like(mov_crop["data"], dtype=np.float32)
        for c in range(mov_crop["data"].shape[0]):
            mov_affine[c] = apply_mapping(affine_matrix, mov_crop["data"][c].astype(np.float32), method="cv2")

        # Step 3: Compute diffeomorphic transformation using first channel only
        mov_affine_2d = mov_affine[0].astype(np.float32)
        diffeo_mapping = compute_diffeomorphic_mapping_dipy(ref_2d, mov_affine_2d)

        # Step 4: Apply diffeomorphic to all channels
        registered_data = np.zeros_like(mov_affine, dtype=np.float32)
        for c in range(mov_affine.shape[0]):
            registered_data[c] = apply_mapping(diffeo_mapping, mov_affine[c], method="dipy")

        return {
            "data": registered_data,
            "y": ref_crop["y"],
            "x": ref_crop["x"],
            "h": ref_crop["h"],
            "w": ref_crop["w"],
            "status": "success",
            "crop_idx": crop_idx,
        }

    except Exception as e:
        logger.error(f"Failed to register crop {crop_idx}: {e}")
        # Return original crop on failure
        return {
            "data": mov_crop["data"],
            "y": mov_crop["y"],
            "x": mov_crop["x"],
            "h": mov_crop["h"],
            "w": mov_crop["w"],
            "status": "failed",
            "crop_idx": crop_idx,
            "error": str(e),
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
    """
    Register a moving image to a reference image using GPU-based registration.

    Note: Images should be pre-padded to matching dimensions by PAD_IMAGES process.

    Parameters:
        reference_path (Path): Path to reference image
        moving_path (Path): Path to moving image
        output_path (Path): Path to save registered image
        crop_size (int): Size of crops for processing
        overlap (int): Overlap between crops
        n_features (int): Number of features for affine registration
        n_workers (int): Number of parallel workers for CPU operations
        qc_dir (Path, optional): Directory to save QC outputs
    """
    # Check GPU availability
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

    # Ensure images are in (C, H, W) format
    if ref_img.ndim == 2:
        ref_img = ref_img[np.newaxis, ...]
    if mov_img.ndim == 2:
        mov_img = mov_img[np.newaxis, ...]

    # Validate images have matching dimensions (should be pre-padded by PAD_IMAGES process)
    if ref_img.shape != mov_img.shape:
        raise ValueError(
            f"Image dimension mismatch: reference {ref_img.shape} != moving {mov_img.shape}. "
            f"Images should be pre-padded to the same dimensions by PAD_IMAGES process."
        )

    logger.info(f"Images have matching dimensions: {ref_img.shape}")

    # Extract crops from both images in parallel
    logger.info(f"Extracting crops with size={crop_size}, overlap={overlap}")
    logger.info(f"Using {n_workers} workers for parallel cropping")
    ref_crops = extract_crops(ref_img, crop_size, overlap, n_workers=n_workers)
    mov_crops = extract_crops(mov_img, crop_size, overlap, n_workers=n_workers)

    if len(ref_crops) != len(mov_crops):
        raise ValueError(f"Number of crops mismatch: ref={len(ref_crops)}, mov={len(mov_crops)}")

    logger.info(f"Extracted {len(ref_crops)} crops from each image")
    logger.info(f"Processing crops in parallel with {n_workers} workers")
    logger.info(f"Transformations will be computed using FIRST CHANNEL ONLY")

    # Process crops in parallel using ThreadPoolExecutor (shared CUDA context)
    registered_crops = [None] * len(ref_crops)

    # Create partial function with fixed parameters
    process_func = partial(process_crop_pair, n_features=n_features)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all crop pairs
        futures = {
            executor.submit(process_func, i, ref_crops[i], mov_crops[i]): i
            for i in range(len(ref_crops))
        }

        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            crop_idx = result["crop_idx"]
            registered_crops[crop_idx] = result

            status = result["status"]
            position = (result["y"], result["x"])
            logger.info(f"Crop {crop_idx+1}/{len(ref_crops)} at {position}: {status}")

            if status == "failed":
                logger.warning(f"  Error: {result.get('error', 'Unknown')}")

    # Remove status fields before merging
    for crop in registered_crops:
        crop.pop("status", None)
        crop.pop("crop_idx", None)
        crop.pop("error", None)

    # Merge registered crops in parallel
    logger.info(f"Merging {len(registered_crops)} registered crops with {n_workers} workers...")
    registered_img = merge_crops(registered_crops, mov_img.shape, overlap, n_workers=n_workers)

    # Convert back to original dtype
    if mov_img.dtype == np.uint16:
        registered_img = np.clip(registered_img, 0, 65535).astype(np.uint16)
    elif mov_img.dtype == np.uint8:
        registered_img = np.clip(registered_img, 0, 255).astype(np.uint8)
    else:
        registered_img = registered_img.astype(mov_img.dtype)

    # Save registered image
    logger.info(f"Saving registered image to: {output_path}")
    tifffile.imwrite(str(output_path), registered_img, photometric="minisblack", compression="zlib")

    # Generate QC outputs if requested
    if qc_dir:
        logger.info(f"\nGenerating QC outputs to: {qc_dir}")
        qc_dir.mkdir(parents=True, exist_ok=True)

        # Create QC RGB composite for visual inspection
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
            import traceback
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

    # Setup logging
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
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
