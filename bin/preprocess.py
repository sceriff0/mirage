#!/usr/bin/env python3
"""BaSiC Illumination Correction Preprocessing with OD Space Support.

This module provides utilities to apply BaSiC shading correction to large
multichannel images by tiling into FOVs and reconstructing the corrected image.

Key Features:
- Standard intensity-space correction for fluorescence microscopy
- Optical Density (OD) space correction for brightfield/IHC microscopy
- Automatic background color estimation and normalization
- Per-channel parallel processing

For IHC/brightfield images with colored backgrounds, use --brightfield flag
to enable OD-space processing, which properly handles:
- Vignetting (shading) artifacts
- Colored mounting medium / yellowed slides
- Scanner color calibration differences

References:
- Peng et al. (2017) "A BaSiC tool for background and shading correction"
  Nature Communications 8, 14836
- Beer-Lambert Law for optical density transformation
"""

from __future__ import annotations

import os
import argparse
import logging
import math
from pathlib import Path

# Add parent directory to path to import lib modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger, configure_logging
from typing import Tuple, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tifffile
from numpy.typing import NDArray

os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Force CPU for JAX

# Check BaSiCPy and Pydantic versions for compatibility
import basicpy
import pydantic
from packaging import version

BASICPY_VERSION = getattr(basicpy, '__version__', 'unknown')
PYDANTIC_VERSION = getattr(pydantic, '__version__', 'unknown')
AUTOTUNE_AVAILABLE = False

# autotune requires BaSiCPy >= 1.1.0 and Pydantic v2
if BASICPY_VERSION != 'unknown' and PYDANTIC_VERSION != 'unknown':
    try:
        basicpy_ok = version.parse(BASICPY_VERSION) >= version.parse("1.1.0")
        pydantic_ok = version.parse(PYDANTIC_VERSION) >= version.parse("2.0.0")
        AUTOTUNE_AVAILABLE = basicpy_ok and pydantic_ok
    except Exception:
        pass

from basicpy import BaSiC  # type: ignore

from utils.image_utils import ensure_dir

logger = get_logger(__name__)

# Log version info at import time
logger.info(f"BaSiCPy version: {BASICPY_VERSION}")
logger.info(f"Pydantic version: {PYDANTIC_VERSION}")
logger.info(f"Autotune available: {AUTOTUNE_AVAILABLE}")

__all__ = [
    "split_image_into_fovs",
    "reconstruct_image_from_fovs",
    "apply_basic_correction",
    "preprocess_multichannel_image",
    "rgb_to_od",
    "od_to_rgb",
    "estimate_background_intensity",
]


# =============================================================================
# Optical Density (OD) Space Conversion Functions
# =============================================================================

def rgb_to_od(
    image: NDArray,
    background: Optional[float] = None,
    eps: float = 1e-6
) -> NDArray:
    """Convert intensity image to optical density space.
    
    Based on Beer-Lambert law: OD = -log10(I / I0)
    where I is transmitted intensity and I0 is incident (background) intensity.
    
    Parameters
    ----------
    image : NDArray
        Input image in intensity space. Can be uint8, uint16, or float.
    background : float, optional
        Background (white/incident) intensity value. If None, uses the
        maximum possible value for the dtype (255 for uint8, 65535 for uint16).
    eps : float
        Small value to prevent log(0). Default 1e-6.
        
    Returns
    -------
    NDArray
        Image in optical density space (float64). Higher values = more absorption.
        
    Notes
    -----
    In OD space:
    - Background (white) → 0
    - Dark staining → high positive values
    - Illumination artifacts become additive, matching BaSiC's model
    """
    # Determine background intensity
    if background is None:
        if image.dtype == np.uint8:
            background = 255.0
        elif image.dtype == np.uint16:
            background = 65535.0
        else:
            background = image.max() if image.max() > 1 else 1.0
    
    # Convert to float and normalize
    img_float = image.astype(np.float64)
    
    # Clip to avoid log(0) and negative values
    img_float = np.clip(img_float, eps, background)
    
    # Apply Beer-Lambert transformation
    od = -np.log10(img_float / background)
    
    return od


def od_to_rgb(
    od: NDArray,
    background: float = 255.0,
    output_dtype: np.dtype = np.uint8,
    clip_negative_od: bool = True
) -> NDArray:
    """Convert optical density back to intensity space.
    
    Inverse of Beer-Lambert: I = I0 * 10^(-OD)
    
    Parameters
    ----------
    od : NDArray
        Image in optical density space.
    background : float
        Background (white/incident) intensity for output. Default 255.0.
    output_dtype : np.dtype
        Output data type. Default np.uint8.
    clip_negative_od : bool
        If True, clip negative OD values to 0 before conversion.
        Negative OD can occur after correction (brighter than background).
        
    Returns
    -------
    NDArray
        Image in intensity space with specified dtype.
    """
    if clip_negative_od:
        od = np.clip(od, 0, None)
    
    # Inverse Beer-Lambert
    intensity = background * np.power(10, -od)
    
    # Clip to valid range for output dtype
    if output_dtype == np.uint8:
        intensity = np.clip(intensity, 0, 255)
    elif output_dtype == np.uint16:
        intensity = np.clip(intensity, 0, 65535)
    else:
        intensity = np.clip(intensity, 0, background)
    
    return intensity.astype(output_dtype)


def estimate_background_intensity(
    images: List[NDArray],
    percentile: float = 95.0,
    sample_fraction: float = 0.1,
    method: str = 'percentile'
) -> NDArray:
    """Estimate per-channel background (white/incident) intensity.
    
    For brightfield microscopy, background pixels are the brightest regions
    (no tissue/stain). This estimates the incident light intensity I0 needed
    for accurate OD conversion.
    
    Parameters
    ----------
    images : List[NDArray]
        List of image tiles/FOVs. Each should be 2D (single channel) or 3D (H,W,C).
    percentile : float
        Percentile of bright pixels to use for background estimation.
        Default 95.0 (avoids saturated pixels while capturing background).
    sample_fraction : float
        Fraction of pixels to sample from each image for efficiency.
        Default 0.1 (10%).
    method : str
        'percentile': Use specified percentile of all sampled pixels
        'corner': Use corner regions (assumes corners are background)
        
    Returns
    -------
    NDArray
        Estimated background intensity. Scalar for 2D input, array for 3D.
        
    Notes
    -----
    For IHC images with colored backgrounds (e.g., yellowed mounting medium),
    each channel will have a different background value, reflecting the
    color cast that needs to be normalized.
    """
    if len(images) == 0:
        raise ValueError("No images provided for background estimation")
    
    sample_image = images[0]
    is_multichannel = sample_image.ndim == 3
    n_channels = sample_image.shape[-1] if is_multichannel else 1
    
    all_samples = [[] for _ in range(n_channels)]
    
    for img in images:
        if method == 'corner':
            # Sample from corners (typically background in tissue images)
            h, w = img.shape[:2]
            corner_size = min(h, w) // 10
            corners = [
                img[:corner_size, :corner_size],
                img[:corner_size, -corner_size:],
                img[-corner_size:, :corner_size],
                img[-corner_size:, -corner_size:]
            ]
            for corner in corners:
                if is_multichannel:
                    for c in range(n_channels):
                        all_samples[c].extend(corner[..., c].ravel())
                else:
                    all_samples[0].extend(corner.ravel())
        else:
            # Random sampling across entire image
            if is_multichannel:
                for c in range(n_channels):
                    flat = img[..., c].ravel()
                    n_sample = max(1, int(len(flat) * sample_fraction))
                    indices = np.random.choice(len(flat), n_sample, replace=False)
                    all_samples[c].extend(flat[indices])
            else:
                flat = img.ravel()
                n_sample = max(1, int(len(flat) * sample_fraction))
                indices = np.random.choice(len(flat), n_sample, replace=False)
                all_samples[0].extend(flat[indices])
    
    # Compute percentile for each channel
    background = np.array([
        np.percentile(samples, percentile) for samples in all_samples
    ])
    
    if n_channels == 1:
        return background[0]
    return background


# =============================================================================
# FOV Tiling Functions
# =============================================================================

def count_fovs(
    image_shape: Tuple[int, int],
    fov_size: Tuple[int, int],
    overlap: int = 0
) -> Tuple[int, int]:
    """Calculate how many FOVs are needed to cover an image with given FOV size and overlap."""
    height, width = image_shape[:2]
    fov_h, fov_w = fov_size

    if overlap >= fov_h or overlap >= fov_w:
        raise ValueError("Overlap cannot be >= FOV size")

    step_y = fov_h - overlap
    step_x = fov_w - overlap

    n_fovs_y = math.ceil((height - fov_h) / step_y) + 1 if height > fov_h else 1
    n_fovs_x = math.ceil((width - fov_w) / step_x) + 1 if width > fov_w else 1

    return n_fovs_y, n_fovs_x


def split_image_into_fovs(
    image: NDArray,
    n_fovs_x: int,
    n_fovs_y: int
) -> Tuple[NDArray, List[Tuple[int, int, int, int]], Tuple[int, int]]:
    """
    Split image (H, W) into FOV tiles with adaptive sizing to handle remainders.

    The image is divided into n_fovs_y * n_fovs_x tiles. Remainder pixels are
    distributed across tiles (some tiles get +1 pixel) to exactly cover the image
    without padding.
    """
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D, got shape {image.shape}")

    if n_fovs_x <= 0 or n_fovs_y <= 0:
        raise ValueError("Number of FOVs must be positive")

    height, width = image.shape

    # Calculate base FOV sizes and remainders
    base_w = width // n_fovs_x
    base_h = height // n_fovs_y
    remainder_x = width % n_fovs_x
    remainder_y = height % n_fovs_y

    # Calculate actual FOV sizes (some FOVs get +1 pixel to handle remainders)
    fov_widths = [base_w + (1 if j < remainder_x else 0) for j in range(n_fovs_x)]
    fov_heights = [base_h + (1 if i < remainder_y else 0) for i in range(n_fovs_y)]

    max_w = max(fov_widths)
    max_h = max(fov_heights)

    # Create FOV stack with padding to max dimensions
    n_fovs = n_fovs_y * n_fovs_x
    fov_stack = np.zeros((n_fovs, max_h, max_w), dtype=image.dtype)

    # Extract FOVs and store position info
    positions = []
    y_start = 0
    idx = 0

    for i in range(n_fovs_y):
        x_start = 0
        for j in range(n_fovs_x):
            h = fov_heights[i]
            w = fov_widths[j]

            fov_stack[idx, :h, :w] = image[y_start:y_start + h, x_start:x_start + w]

            positions.append((y_start, x_start, h, w))
            x_start += w
            idx += 1
        y_start += fov_heights[i]

    return fov_stack, positions, (max_h, max_w)


def reconstruct_image_from_fovs(
    fov_stack: NDArray,
    positions: List[Tuple[int, int, int, int]],
    original_shape: Tuple[int, ...]
) -> NDArray:
    """
    Reconstruct 2D image from 3D FOV tiles stack.
    """
    reconstructed = np.zeros(original_shape, dtype=fov_stack.dtype)

    for idx, (row_start, col_start, h, w) in enumerate(positions):
        # fov_stack is 3D (N, fov_h, fov_w), reconstructed is 2D (H, W)
        reconstructed[row_start:row_start + h, col_start:col_start + w] = \
            fov_stack[idx, :h, :w]

    return reconstructed


# =============================================================================
# BaSiC Correction Functions
# =============================================================================

def apply_basic_correction(
    image: NDArray,
    fov_size: Tuple[int, int] = (1950, 1950),
    get_darkfield: bool = True,
    autotune: bool = False,
    n_iter: int = 100,
    use_od_space: bool = False,
    background_intensity: Optional[float] = None,
    **basic_kwargs
) -> Tuple[NDArray, object, Dict[str, Any]]:
    """
    Apply BaSiC illumination correction to a single channel image (H, W).
    
    Parameters
    ----------
    image : NDArray
        2D image (H, W) to correct.
    fov_size : Tuple[int, int]
        Size of FOV tiles for BaSiC fitting.
    get_darkfield : bool
        Whether to estimate darkfield (additive offset).
    autotune : bool
        Whether to autotune BaSiC parameters.
    n_iter : int
        Number of autotuning iterations.
    use_od_space : bool
        If True, convert to optical density space before correction.
        Recommended for brightfield/IHC images with colored backgrounds.
    background_intensity : float, optional
        Background (incident) intensity for OD conversion. If None, estimated
        from image dtype (255 for uint8, 65535 for uint16).
        
    Returns
    -------
    reconstructed : NDArray
        Corrected image in original intensity space.
    basic : BaSiC
        Fitted BaSiC model (contains flatfield/darkfield).
    correction_info : Dict
        Dictionary with correction metadata (background, space used, etc.)
    """
    if image.ndim != 2:
        raise ValueError(f"apply_basic_correction requires a 2D image, got shape {image.shape}")

    original_dtype = image.dtype
    correction_info = {
        'use_od_space': use_od_space,
        'original_dtype': str(original_dtype),
        'background_intensity': background_intensity,
    }
    
    # Convert to OD space if requested (for brightfield/IHC)
    if use_od_space:
        logger.info(f"    Converting to OD space (background={background_intensity})")
        working_image = rgb_to_od(image, background=background_intensity)
        correction_info['od_range'] = (float(working_image.min()), float(working_image.max()))
    else:
        working_image = image.astype(np.float64)

    # Tile into FOVs
    n_fovs_y, n_fovs_x = count_fovs(image.shape, fov_size)
    fov_stack, positions, _ = split_image_into_fovs(working_image, n_fovs_x, n_fovs_y)
    
    logger.info(f"    Tiled into {n_fovs_y}x{n_fovs_x} = {n_fovs_y * n_fovs_x} FOVs")

    # Fit BaSiC
    basic = BaSiC(get_darkfield=get_darkfield, smoothness_flatfield=1)

    if autotune and AUTOTUNE_AVAILABLE:
        logger.info(f"    Autotuning BaSiC parameters for {n_iter} iterations")
        basic.autotune(fov_stack, n_iter=n_iter)

    corrected_fovs = basic.fit_transform(fov_stack)
    
    # Store flatfield/darkfield info
    correction_info['flatfield_range'] = (float(basic.flatfield.min()), float(basic.flatfield.max()))
    if get_darkfield and basic.darkfield is not None:
        correction_info['darkfield_range'] = (float(basic.darkfield.min()), float(basic.darkfield.max()))
        correction_info['darkfield_mean'] = float(basic.darkfield.mean())

    # Reconstruct
    reconstructed = reconstruct_image_from_fovs(
        corrected_fovs,
        positions,
        working_image.shape
    )

    # Convert back from OD space if needed
    if use_od_space:
        logger.info(f"    Converting back from OD space")
        # Determine appropriate background for inverse transform
        if original_dtype == np.uint8:
            out_background = 255.0
        elif original_dtype == np.uint16:
            out_background = 65535.0
        else:
            out_background = background_intensity or 255.0
            
        reconstructed = od_to_rgb(
            reconstructed, 
            background=out_background,
            output_dtype=original_dtype
        )
    else:
        # Clip and convert back to original dtype
        if original_dtype == np.uint8:
            reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        elif original_dtype == np.uint16:
            reconstructed = np.clip(reconstructed, 0, 65535).astype(np.uint16)
        else:
            reconstructed = reconstructed.astype(original_dtype)

    return reconstructed, basic, correction_info


def _process_single_channel_from_stack(
    channel_image: NDArray,
    channel_index: int,
    channel_name: str,
    fov_size: Tuple[int, int],
    skip_dapi: bool,
    autotune: bool,
    n_iter: int,
    basic_kwargs: dict,
    use_od_space: bool = False,
    background_intensity: Optional[float] = None,
    auto_detect: bool = True
) -> Tuple[int, NDArray, bool, Dict[str, Any]]:
    """Worker function to process a single channel slice from a stack.

    Returns
    -------
    channel_index : int
        Channel index
    processed_image : NDArray
        Corrected or original image
    was_corrected : bool
        Whether BaSiC was applied
    correction_info : Dict
        Metadata about the correction applied
    """
    logger.info(f"Processing channel #{channel_index} ({channel_name})")

    correction_info = {'channel': channel_name, 'index': channel_index}

    # Skip DAPI if requested
    if skip_dapi and 'DAPI' in channel_name.upper():
        logger.info(f"  ⊘ Skipping BaSiC correction for DAPI (user setting)")
        correction_info['skipped'] = True
        correction_info['skip_reason'] = 'DAPI channel'
        return channel_index, channel_image, False, correction_info

    logger.info(f"  ✓ Applying BaSiC correction (OD space: {use_od_space})")
    
    corrected, basic_model, channel_correction_info = apply_basic_correction(
        channel_image,
        fov_size=fov_size,
        autotune=autotune,
        n_iter=n_iter,
        use_od_space=use_od_space,
        background_intensity=background_intensity,
        **basic_kwargs
    )
    
    correction_info.update(channel_correction_info)
    correction_info['skipped'] = False
    
    return channel_index, corrected, True, correction_info


def preprocess_multichannel_image(
    image_path: str,
    channel_names: List[str],
    output_path: str,
    fov_size: Tuple[int, int] = (1950, 1950),
    skip_dapi: bool = True,
    autotune: bool = False,
    n_iter: int = 3,
    n_workers: int = 4,
    use_od_space: bool = False,
    estimate_background: bool = True,
    background_percentile: float = 95.0,
    **basic_kwargs
) -> Tuple[NDArray, Dict[str, Any]]:
    """
    Apply BaSiC preprocessing to a single multichannel image in parallel and save as TIFF.
    
    Parameters
    ----------
    image_path : str
        Path to input multichannel image.
    channel_names : List[str]
        Names for each channel.
    output_path : str
        Path to save corrected image.
    fov_size : Tuple[int, int]
        FOV tile size for BaSiC.
    skip_dapi : bool
        Skip correction for DAPI channels.
    autotune : bool
        Autotune BaSiC parameters.
    n_iter : int
        Autotuning iterations.
    n_workers : int
        Parallel workers for channel processing.
    use_od_space : bool
        Use optical density space for correction. Recommended for brightfield/IHC.
    estimate_background : bool
        Estimate background intensity from images (for OD conversion).
    background_percentile : float
        Percentile for background estimation.
        
    Returns
    -------
    preprocessed : NDArray
        Corrected multichannel image stack.
    preprocessing_report : Dict
        Detailed report of corrections applied.
    """
    logger.info(f"Loading multichannel image from {image_path}")
    
    preprocessing_report = {
        'input_path': image_path,
        'output_path': output_path,
        'use_od_space': use_od_space,
        'fov_size': fov_size,
        'channels': {}
    }

    # Read and log input file metadata
    logger.info("Reading input file metadata...")
    with tifffile.TiffFile(image_path) as tif:
        if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
            logger.info(f"  ✓ Input has OME metadata (length: {len(tif.ome_metadata)} chars)")
            if 'PhysicalSizeX' in tif.ome_metadata:
                logger.info(f"  ✓ Input has PhysicalSizeX metadata")
            if 'PhysicalSizeXUnit' in tif.ome_metadata:
                logger.info(f"  ✓ Input has PhysicalSizeXUnit metadata")
        else:
            logger.warning(f"  ⚠ Input file has no OME metadata")

        if tif.pages and len(tif.pages) > 0:
            first_page = tif.pages[0]
            logger.info(f"  - First page shape: {first_page.shape}")
            logger.info(f"  - First page dtype: {first_page.dtype}")

    multichannel_stack = tifffile.imread(image_path)
    logger.info(f"Loaded image shape: {multichannel_stack.shape}")

    if multichannel_stack.ndim == 2:
        logger.info("  - Converting 2D to 3D (adding channel dimension)")
        multichannel_stack = np.expand_dims(multichannel_stack, axis=0)
    elif multichannel_stack.ndim == 3 and multichannel_stack.shape[2] == len(channel_names):
        logger.info("  - Transposing from (Y, X, C) to (C, Y, X)")
        multichannel_stack = np.transpose(multichannel_stack, (2, 0, 1))

    n_channels, H, W = multichannel_stack.shape
    logger.info(f"Processing {n_channels} channels ({H}x{W}) with {n_workers} workers")
    
    preprocessing_report['image_shape'] = {'channels': n_channels, 'height': H, 'width': W}
    preprocessing_report['original_dtype'] = str(multichannel_stack.dtype)

    if n_channels != len(channel_names):
        channel_names = channel_names[:n_channels] + [f"Channel_{i}" for i in range(len(channel_names), n_channels)]

    # Estimate background intensity for OD conversion if needed
    background_intensities = None
    if use_od_space and estimate_background:
        logger.info(f"Estimating background intensity (percentile={background_percentile})")
        
        # Sample tiles for background estimation
        sample_tiles = []
        n_sample_tiles = min(50, max(1, (H // fov_size[0]) * (W // fov_size[1])))
        
        for c in range(n_channels):
            # Simple sampling from image
            channel_img = multichannel_stack[c]
            # Estimate from the channel directly
            bg = np.percentile(channel_img, background_percentile)
            logger.info(f"  Channel {c} ({channel_names[c]}): background = {bg:.1f}")
            
        # For per-channel background, estimate separately
        background_intensities = [
            float(np.percentile(multichannel_stack[c], background_percentile))
            for c in range(n_channels)
        ]
        preprocessing_report['background_intensities'] = dict(zip(channel_names, background_intensities))
        logger.info(f"Background intensities: {dict(zip(channel_names[:n_channels], background_intensities))}")

    results = {}
    correction_applied = {}
    channel_reports = {}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_channels):
            # Get per-channel background if available
            bg = background_intensities[i] if background_intensities else None
            
            future = executor.submit(
                _process_single_channel_from_stack,
                multichannel_stack[i, ...],
                i,
                channel_names[i],
                fov_size,
                skip_dapi,
                autotune,
                n_iter,
                basic_kwargs,
                use_od_space=use_od_space,
                background_intensity=bg
            )
            futures.append(future)

        for future in as_completed(futures):
            channel_index, result_array, was_corrected, correction_info = future.result()
            results[channel_index] = result_array
            correction_applied[channel_index] = was_corrected
            channel_reports[channel_names[channel_index]] = correction_info

    preprocessed_channels = [
        results[i] for i in range(n_channels)
    ]
    
    preprocessing_report['channels'] = channel_reports

    # Log summary
    n_corrected = sum(correction_applied.values())
    n_skipped = n_channels - n_corrected
    logger.info(f"\nBaSiC Correction Summary:")
    logger.info(f"  ✓ Corrected: {n_corrected}/{n_channels} channels")
    logger.info(f"  ✗ Skipped: {n_skipped}/{n_channels} channels")
    if use_od_space:
        logger.info(f"  ✓ Used OD space transformation (brightfield mode)")
    
    preprocessing_report['summary'] = {
        'corrected': n_corrected,
        'skipped': n_skipped,
        'total': n_channels
    }

    preprocessed = np.stack(preprocessed_channels, axis=0)

    logger.info(f"Saving corrected image to {output_path}")
    logger.info(f"Final stack shape: {preprocessed.shape} (expecting C, Y, X)")
    logger.info(f"Channel names to save: {channel_names[:preprocessed.shape[0]]}")

    # Save as OME-TIFF with proper metadata
    metadata = {
        'axes': 'CYX',
        'Channel': {'Name': channel_names[:preprocessed.shape[0]]},
        'PhysicalSizeX': 0.325,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': 0.325,
        'PhysicalSizeYUnit': 'µm'
    }

    logger.info("Writing OME-TIFF with metadata:")
    logger.info(f"  - Axes: {metadata['axes']}")
    logger.info(f"  - Channels: {metadata['Channel']['Name']}")
    logger.info(f"  - PhysicalSizeX: {metadata['PhysicalSizeX']} {metadata['PhysicalSizeXUnit']}")
    logger.info(f"  - PhysicalSizeY: {metadata['PhysicalSizeY']} {metadata['PhysicalSizeYUnit']}")

    tifffile.imwrite(
        output_path,
        preprocessed,
        photometric='minisblack',
        metadata=metadata,
        bigtiff=True,
        ome=True,
        compression='zlib',
    )

    logger.info(f"Saved OME-TIFF with {preprocessed.shape[0]} channels")

    # Verify the saved file
    logger.info("Verifying saved file...")
    with tifffile.TiffFile(output_path) as tif:
        if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
            logger.info(f"  ✓ OME-XML metadata present (length: {len(tif.ome_metadata)} chars)")
            if 'PhysicalSizeX' in tif.ome_metadata:
                logger.info(f"  ✓ PhysicalSizeX found in metadata")
            if 'PhysicalSizeXUnit' in tif.ome_metadata or 'µm' in tif.ome_metadata:
                logger.info(f"  ✓ Physical size units found in metadata")
            else:
                logger.warning(f"  ⚠ Physical size units may not be in metadata")
        else:
            logger.warning(f"  ⚠ No OME metadata in saved file!")

    verify_img = tifffile.imread(output_path)
    logger.info(f"  - Reloaded image shape: {verify_img.shape}")
    if verify_img.ndim == 3:
        logger.info(f"  ✓ File saved correctly with {verify_img.shape[0]} channels")
    else:
        logger.warning(f"  ⚠ File may not have correct dimensions: {verify_img.shape}")

    return preprocessed, preprocessing_report


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Apply BaSiC illumination correction to multichannel images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to the multichannel image file'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory'
    )

    parser.add_argument(
        '--channels',
        type=str,
        nargs='+',
        required=True,
        help='Channel names from metadata'
    )

    parser.add_argument(
        '--fov_size',
        type=int,
        default=1950,
        help='FOV size for BaSiC tiling'
    )
    
    parser.add_argument(
        '--n_workers',
        type=int,
        default=4,
        help='Maximum number of channels to process in parallel.'
    )

    parser.add_argument(
        '--skip_dapi',
        action='store_true',
        help='Skip BaSiC correction for DAPI channel'
    )

    parser.add_argument(
        '--autotune',
        action='store_true',
        help='Autotune BaSiC parameters'
    )

    parser.add_argument(
        '--n_iter',
        type=int,
        default=3,
        help='Number of autotuning iterations'
    )

    parser.add_argument(
        '--overlap',
        type=int,
        default=0,
        help='Overlap between FOV tiles for BaSiC correction'
    )

    parser.add_argument(
        '--no_darkfield',
        action='store_true',
        help='Disable darkfield estimation in BaSiC'
    )
    
    # New arguments for OD space / brightfield support
    parser.add_argument(
        '--brightfield',
        action='store_true',
        help='Enable brightfield/IHC mode: use optical density (OD) space for correction. '
             'Recommended for images with colored backgrounds or vignetting artifacts.'
    )
    
    parser.add_argument(
        '--background_percentile',
        type=float,
        default=95.0,
        help='Percentile for background intensity estimation (for OD conversion). '
             'Higher values = brighter background estimate.'
    )
    
    parser.add_argument(
        '--no_background_estimation',
        action='store_true',
        help='Disable automatic background estimation. Use dtype max instead.'
    )
    
    parser.add_argument(
        '--save_report',
        action='store_true',
        help='Save preprocessing report as JSON file'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    configure_logging(level=logging.INFO)

    args = parse_args()

    ensure_dir(args.output_dir)

    image_path = args.image
    image_basename = os.path.basename(image_path)

    channel_names = args.channels

    # Always save as .ome.tiff since we're writing OME-TIFF format
    if image_basename.endswith('.ome.tif'):
        base = image_basename[:-8]  # Remove .ome.tif
    elif image_basename.endswith('.ome.tiff'):
        base = image_basename[:-9]  # Remove .ome.tiff
    elif image_basename.endswith('.tif'):
        base = image_basename[:-4]  # Remove .tif
    elif image_basename.endswith('.tiff'):
        base = image_basename[:-5]  # Remove .tiff
    else:
        base = os.path.splitext(image_basename)[0]

    ext = '.ome.tif'  # Always use OME-TIFF extension
    output_filename = f"{base}_corrected{ext}"
    output_path = os.path.join(
        args.output_dir,
        output_filename
    )

    logger.info(f"Starting preprocessing: {image_path}")
    logger.info(f"Expected channel order: {channel_names}")
    
    if args.brightfield:
        logger.info("=" * 60)
        logger.info("BRIGHTFIELD/IHC MODE ENABLED")
        logger.info("Using Optical Density (OD) space for BaSiC correction")
        logger.info("This handles colored backgrounds and vignetting properly")
        logger.info("=" * 60)

    # Build BaSiC kwargs
    basic_kwargs = {
        'overlap': args.overlap,
        'get_darkfield': not args.no_darkfield
    }

    preprocessed, report = preprocess_multichannel_image(
        image_path=image_path,
        channel_names=channel_names,
        output_path=output_path,
        fov_size=(args.fov_size, args.fov_size),
        skip_dapi=args.skip_dapi,
        autotune=args.autotune,
        n_iter=args.n_iter,
        n_workers=args.n_workers,
        use_od_space=True,
        estimate_background=not args.no_background_estimation,
        background_percentile=args.background_percentile,
        **basic_kwargs
    )

    logger.info(f"Preprocessing completed successfully. Output: {output_path}")
    
    # Save report if requested
    if args.save_report:
        import json
        report_path = os.path.join(args.output_dir, f"{base}_correction_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Correction report saved to: {report_path}")

    # Write dimensions to file for downstream processes
    import tifffile
    img = tifffile.imread(output_path)
    shape = img.shape if img.ndim == 3 else (1, img.shape[0], img.shape[1])
    dims_filename = f"{base}_dims.txt"
    dims_path = os.path.join(args.output_dir, dims_filename)
    with open(dims_path, 'w') as f:
        f.write(f"{shape[0]} {shape[1]} {shape[2]}")
    logger.info(f"Image dimensions saved to: {dims_path} (C={shape[0]}, H={shape[1]}, W={shape[2]})")

    return 0


if __name__ == '__main__':
    exit(main())