"""Quality control image generation for registration.

This module provides functions for creating QC visualizations to assess
registration quality, particularly RGB composite overlays of registered
and reference images.

Examples
--------
>>> from qc import create_registration_qc
>>> create_registration_qc(
...     reference_path="ref.tif",
...     registered_path="registered.tif",
...     output_path="qc.tif"
... )

Notes
-----
This module consolidates QC generation code that was previously embedded
in registration scripts (register_cpu.py, register_gpu.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tifffile
from numpy.typing import NDArray
from skimage.transform import rescale

from logger import get_logger
from metadata import get_channel_names
from registration_utils import autoscale

__all__ = [
    "create_registration_qc",
    "create_dapi_overlay",
    "autoscale_for_display",
]

logger = get_logger(__name__)


def autoscale_for_display(
    img: NDArray,
    method: str = "minmax"
) -> NDArray[np.uint8]:
    """Autoscale image for display purposes.

    Parameters
    ----------
    img : NDArray
        Input image of any numeric dtype.
    method : str, default="minmax"
        Scaling method:
        - 'minmax': Min-max normalization (linear)
        - 'percentile': Percentile-based (robust to outliers)

    Returns
    -------
    NDArray[np.uint8]
        Scaled image in 0-255 range.

    Examples
    --------
    >>> img = np.random.rand(100, 100) * 1000
    >>> scaled = autoscale_for_display(img, method='minmax')
    >>> scaled.min(), scaled.max()
    (0, 255)

    See Also
    --------
    autoscale : Percentile-based scaling from registration_utils
    """
    if method == "minmax":
        img_min = img.min()
        img_max = img.max()
        range_val = max(img_max - img_min, 1e-6)
        normalized = (img - img_min) / range_val
        return (normalized * 255).astype(np.uint8)
    elif method == "percentile":
        return autoscale(img, low_p=1.0, high_p=99.0)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'minmax' or 'percentile'.")


def create_dapi_overlay(
    reference_dapi: NDArray,
    registered_dapi: NDArray,
    scale_factor: float = 0.25
) -> Tuple[NDArray, NDArray]:
    """Create RGB overlay of reference and registered DAPI channels.

    Parameters
    ----------
    reference_dapi : NDArray
        Reference DAPI channel (2D array).
    registered_dapi : NDArray
        Registered DAPI channel (2D array).
    scale_factor : float, default=0.25
        Downsampling factor for output images (0.25 = 4x smaller).

    Returns
    -------
    rgb_bgr : NDArray
        RGB composite in BGR order (for OpenCV/PNG), shape (H, W, 3).
        Red = registered, Green = reference, Blue = 0.
    rgb_cyx : NDArray
        RGB composite in CYX order (for TIFF/ImageJ), shape (3, H, W).
        Same channel assignment as rgb_bgr.

    Notes
    -----
    Each channel is independently autoscaled using min-max normalization
    before creating the composite. This ensures both channels are visible
    even if they have different intensity ranges.

    The downsampling uses anti-aliasing to prevent artifacts.

    Examples
    --------
    >>> ref = np.random.randint(0, 4096, (2048, 2048), dtype=np.uint16)
    >>> reg = np.random.randint(0, 4096, (2048, 2048), dtype=np.uint16)
    >>> rgb_bgr, rgb_cyx = create_dapi_overlay(ref, reg, scale_factor=0.25)
    >>> rgb_bgr.shape
    (512, 512, 3)
    >>> rgb_cyx.shape
    (3, 512, 512)

    See Also
    --------
    create_registration_qc : High-level QC generation function
    autoscale_for_display : Scaling function used internally
    """
    # Autoscale each channel independently
    ref_scaled = autoscale_for_display(reference_dapi, method="minmax")
    reg_scaled = autoscale_for_display(registered_dapi, method="minmax")

    # Downsample if requested
    if scale_factor != 1.0:
        ref_down = rescale(
            ref_scaled,
            scale=scale_factor,
            anti_aliasing=True,
            preserve_range=True
        ).astype(np.uint8)
        reg_down = rescale(
            reg_scaled,
            scale=scale_factor,
            anti_aliasing=True,
            preserve_range=True
        ).astype(np.uint8)
    else:
        ref_down = ref_scaled
        reg_down = reg_scaled

    h, w = reg_down.shape

    # Create RGB composite in BGR order (for OpenCV)
    rgb_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_bgr[:, :, 0] = reg_down   # Blue = 0
    rgb_bgr[:, :, 1] = ref_down   # Green = reference
    rgb_bgr[:, :, 2] = 0          # Red = registered

    # Wait, the convention in your code is actually:
    # Red = registered, Green = reference
    # Let me fix this to match existing behavior
    rgb_bgr[:, :, 2] = reg_down   # Red = registered (BGR so index 2 is Red)
    rgb_bgr[:, :, 1] = ref_down   # Green = reference
    rgb_bgr[:, :, 0] = 0          # Blue = 0

    # Create CYX version for TIFF
    rgb_cyx = np.stack([
        reg_down,                           # Red channel (registered)
        ref_down,                           # Green channel (reference)
        np.zeros_like(ref_down, dtype=np.uint8)  # Blue channel
    ], axis=0)

    return rgb_bgr, rgb_cyx


def create_registration_qc(
    reference_path: str | Path,
    registered_path: str | Path,
    output_path: str | Path,
    scale_factor: float = 0.25,
    save_fullres: bool = True,
    save_png: bool = True,
    save_tiff: bool = True
) -> None:
    """Create QC visualizations for registration assessment.

    This function generates RGB composite images showing the overlay of
    reference and registered DAPI channels. Perfect registration appears
    as yellow (red + green), while misalignment shows red/green fringing.

    Parameters
    ----------
    reference_path : str or Path
        Path to reference image (OME-TIFF).
    registered_path : str or Path
        Path to registered image (OME-TIFF).
    output_path : str or Path
        Base path for output files (extension will be modified).
    scale_factor : float, default=0.25
        Downsampling factor for preview images (0.25 = 4x smaller).
        Full resolution is always saved if save_fullres=True.
    save_fullres : bool, default=True
        Save full-resolution TIFF (with compression).
    save_png : bool, default=True
        Save downsampled PNG preview.
    save_tiff : bool, default=True
        Save downsampled TIFF for ImageJ.

    Returns
    -------
    None
        Saves files to disk.

    Notes
    -----
    Output files created:
    - {output_path}_fullres.tif: Full resolution, compressed (if save_fullres=True)
    - {output_path}.png: Downsampled PNG (if save_png=True)
    - {output_path}.tif: Downsampled ImageJ TIFF (if save_tiff=True)

    Channel assignment in composites:
    - Red: Registered image
    - Green: Reference image
    - Blue: 0 (black)

    Perfect alignment appears yellow (red + green).
    Misalignment shows red/green fringing.

    Examples
    --------
    >>> create_registration_qc(
    ...     reference_path="reference_DAPI_SMA.tif",
    ...     registered_path="registered_DAPI_CD3.tif",
    ...     output_path="qc_output.tif"
    ... )
    # Creates: qc_output_fullres.tif, qc_output.png, qc_output.tif

    Custom scaling and selective output:
    >>> create_registration_qc(
    ...     reference_path="ref.tif",
    ...     registered_path="reg.tif",
    ...     output_path="qc.tif",
    ...     scale_factor=0.5,
    ...     save_fullres=False,
    ...     save_png=True,
    ...     save_tiff=False
    ... )

    See Also
    --------
    create_dapi_overlay : Lower-level overlay creation
    lib.metadata.get_channel_names : Extract channel names
    """
    reference_path = Path(reference_path)
    registered_path = Path(registered_path)
    output_path = Path(output_path)

    logger.info(f"Creating QC composite: {output_path.name}")

    # Load images
    ref_img = tifffile.imread(str(reference_path))
    reg_img = tifffile.imread(str(registered_path))

    # Ensure 3D shape (C, H, W)
    if ref_img.ndim == 2:
        ref_img = ref_img[np.newaxis, ...]
    if reg_img.ndim == 2:
        reg_img = reg_img[np.newaxis, ...]

    # Find DAPI channels
    ref_channels = get_channel_names(reference_path.name)
    reg_channels = get_channel_names(registered_path.name)

    # Find DAPI index (default to first channel if not found)
    ref_dapi_idx = next(
        (i for i, ch in enumerate(ref_channels) if "DAPI" in ch.upper()),
        0
    )
    reg_dapi_idx = next(
        (i for i, ch in enumerate(reg_channels) if "DAPI" in ch.upper()),
        0
    )

    logger.debug(f"Reference DAPI: channel {ref_dapi_idx} ({ref_channels[ref_dapi_idx]})")
    logger.debug(f"Registered DAPI: channel {reg_dapi_idx} ({reg_channels[reg_dapi_idx]})")

    ref_dapi = ref_img[ref_dapi_idx]
    reg_dapi = reg_img[reg_dapi_idx]

    # Save full-resolution QC (compressed)
    if save_fullres:
        ref_dapi_scaled = autoscale_for_display(ref_dapi, method="minmax")
        reg_dapi_scaled = autoscale_for_display(reg_dapi, method="minmax")

        rgb_stack_full = np.stack([
            reg_dapi_scaled,   # Red channel (registered)
            ref_dapi_scaled,   # Green channel (reference)
            np.zeros_like(ref_dapi_scaled, dtype=np.uint8)  # Blue channel
        ], axis=0)

        fullres_output_path = output_path.with_name(output_path.stem + '_fullres.tif')
        tifffile.imwrite(
            str(fullres_output_path),
            rgb_stack_full,
            imagej=True,
            metadata={'axes': 'CYX', 'mode': 'composite'},
            compression='zlib'
        )
        logger.info(f"  Saved full-res QC TIFF: {fullres_output_path}")
        del rgb_stack_full

    # Create downsampled overlay
    rgb_bgr, rgb_cyx = create_dapi_overlay(ref_dapi, reg_dapi, scale_factor=scale_factor)

    # Save PNG (OpenCV uses BGR order)
    if save_png:
        png_output_path = output_path.with_suffix('.png')
        cv2.imwrite(str(png_output_path), rgb_bgr)
        logger.info(f"  Saved QC PNG: {png_output_path}")

    # Save TIFF (ImageJ-compatible, CYX order)
    if save_tiff:
        tiff_output_path = output_path.with_suffix('.tif')
        tifffile.imwrite(
            str(tiff_output_path),
            rgb_cyx,
            imagej=True,
            metadata={'axes': 'CYX', 'mode': 'composite'}
        )
        logger.info(f"  Saved QC TIFF: {tiff_output_path}")

    logger.info(f"QC generation complete")
