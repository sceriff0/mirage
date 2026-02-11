"""Image processing utilities for the MIRAGE pipeline.

This module provides common image processing functions used across
multiple scripts, following DRY principles.

Examples
--------
>>> from image_utils import normalize_image_dimensions, load_image_grayscale
>>> img = np.random.rand(100, 100, 3)
>>> normalized = normalize_image_dimensions(img)
>>> normalized.shape
(3, 100, 100)

Notes
-----
This module consolidates image processing code that was previously
duplicated across multiple bin/ scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

import tifffile
from typing import Any, Dict

try:
    import pyvips
    HAS_PYVIPS = True
except ImportError:
    HAS_PYVIPS = False

__all__ = [
    "normalize_image_dimensions",
    "load_image_grayscale",
    "load_image",
    "save_tiff",
    "ensure_dir",
]


def normalize_image_dimensions(img: NDArray) -> NDArray:
    """Normalize image to (C, H, W) format.

    This function handles the common case of images that may be in
    different formats:
    - 2D (H, W) -> (1, H, W)
    - 3D (H, W, C) -> (C, H, W)
    - 3D (C, H, W) -> (C, H, W) [no change]

    Parameters
    ----------
    img : NDArray
        Input image array.
        Can be 2D (grayscale) or 3D (multichannel).

    Returns
    -------
    NDArray
        Normalized array in (C, H, W) format where C is the number of channels.

    Raises
    ------
    ValueError
        If image has incorrect number of dimensions (not 2D or 3D).

    Notes
    -----
    The function detects (H, W, C) vs (C, H, W) format by comparing dimensions:
    - If shape[0] is smallest, assumes (C, H, W) format - no change
    - If shape[2] is smallest, assumes (H, W, C) format - transposes to (C, H, W)

    This heuristic works well for typical microscopy images where:
    - Number of channels (C) is typically 1-10
    - Height and Width are typically 512-4096+ pixels

    Examples
    --------
    Grayscale image:
    >>> img = np.random.rand(1024, 1024)
    >>> result = normalize_image_dimensions(img)
    >>> result.shape
    (1, 1024, 1024)

    RGB image in (H, W, C) format:
    >>> img = np.random.rand(1024, 1024, 3)
    >>> result = normalize_image_dimensions(img)
    >>> result.shape
    (3, 1024, 1024)

    Already normalized:
    >>> img = np.random.rand(3, 1024, 1024)
    >>> result = normalize_image_dimensions(img)
    >>> result.shape
    (3, 1024, 1024)

    See Also
    --------
    numpy.transpose : Transpose array dimensions
    """
    if img.ndim == 2:
        # Grayscale (H, W) -> (1, H, W)
        return img[np.newaxis, ...]

    elif img.ndim == 3:
        # Detect format by comparing dimensions
        # Assumption: channels is the smallest dimension
        if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            # Already (C, H, W) format
            return img
        elif img.shape[2] < img.shape[0] and img.shape[2] < img.shape[1]:
            # (H, W, C) format - transpose to (C, H, W)
            return np.transpose(img, (2, 0, 1))
        else:
            # Ambiguous - assume (C, H, W) and keep as-is
            # This handles square images or unusual dimensions
            return img

    else:
        raise ValueError(
            f"Image must be 2D or 3D, got {img.ndim}D with shape {img.shape}"
        )


def load_image_grayscale(
    image_path: str | Path,
    max_dim: Optional[int] = None
) -> NDArray[np.uint8]:
    """Load image and convert to grayscale uint8 for feature detection.

    This function is optimized for loading images for computer vision tasks
    like feature detection and matching. It:
    1. Loads the image using pyvips (memory-efficient for large images)
    2. Downsamples if needed (before loading into memory)
    3. Extracts first channel if multichannel
    4. Converts to uint8 with proper scaling

    Parameters
    ----------
    image_path : str or Path
        Path to image file (TIFF, PNG, etc.)
    max_dim : int or None, optional
        Maximum dimension (width or height) for downsampling.
        If None, loads at full resolution.
        If image is larger than max_dim, it will be downscaled proportionally.

    Returns
    -------
    NDArray[np.uint8]
        Grayscale image as uint8 array with shape (H, W).

    Raises
    ------
    ImportError
        If pyvips is not installed.
    FileNotFoundError
        If image_path does not exist.

    Notes
    -----
    This function requires pyvips to be installed. If pyvips is not available,
    an ImportError will be raised.

    The function is designed for:
    - Feature detection (SuperPoint, DISK, ORB, etc.)
    - Registration preprocessing
    - QC image generation

    For loading full multichannel images, use tifffile.imread() or
    lib.metadata functions instead.

    Examples
    --------
    Load full resolution:
    >>> img = load_image_grayscale("sample.tif")
    >>> img.dtype
    dtype('uint8')

    Load with downsampling:
    >>> img = load_image_grayscale("large_image.tif", max_dim=2048)
    >>> max(img.shape) <= 2048
    True

    See Also
    --------
    pyvips.Image.new_from_file : Efficient image loading
    normalize_image_dimensions : Normalize multichannel images
    """
    if not HAS_PYVIPS:
        raise ImportError(
            "pyvips is required for load_image_grayscale(). "
            "Install with: pip install pyvips"
        )

    # Load image with pyvips (memory-efficient)
    vips_img = pyvips.Image.new_from_file(str(image_path))

    # Downsample if requested
    if max_dim is not None:
        current_max = max(vips_img.width, vips_img.height)
        if current_max > max_dim:
            scale = max_dim / current_max
            vips_img = vips_img.resize(scale)

    # Extract first channel if multichannel
    if vips_img.bands > 1:
        vips_img = vips_img.extract_band(0)

    # Convert to numpy array
    img_array = vips_img.numpy()

    # Ensure uint8 format with proper scaling
    if img_array.dtype != np.uint8:
        img_min = img_array.min()
        img_max = img_array.max()

        if img_max > img_min:
            # Scale to 0-255
            img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            # All same value - return zeros
            img_array = np.zeros_like(img_array, dtype=np.uint8)

    return img_array


def ensure_dir(directory: str | Path) -> Path:
    """Ensure directory exists, creating it if necessary.

    Parameters
    ----------
    directory : str or Path
        Path to directory to create.

    Returns
    -------
    Path
        Path object for the created/existing directory.

    Notes
    -----
    Creates parent directories as needed (parents=True) and does not
    raise an error if directory exists (exist_ok=True).

    Examples
    --------
    >>> output_dir = ensure_dir("results/analysis")
    >>> print(output_dir)
    PosixPath('results/analysis')

    See Also
    --------
    pathlib.Path.mkdir : Underlying method used
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_image(
    image_path: str | Path,
    memmap: bool = False
) -> Tuple[NDArray, Dict[str, Any]]:
    """Load image from TIFF file with metadata extraction.

    Parameters
    ----------
    image_path : str or Path
        Path to TIFF image file.
    memmap : bool, default=False
        If True, use memory-mapped I/O to avoid loading entire file into RAM.

    Returns
    -------
    image : NDArray
        Image data as numpy array. Shape depends on image dimensionality.
    metadata : dict
        Dictionary containing image metadata with keys:
        - 'shape': tuple - Image shape
        - 'dtype': np.dtype - Data type
        - 'ome': str or None - OME-XML metadata if available
        - 'axes': str or None - Dimension order (e.g., 'CYX')

    Raises
    ------
    FileNotFoundError
        If image_path does not exist.
    ValueError
        If file cannot be read as TIFF.

    Notes
    -----
    This function handles both standard TIFF and OME-TIFF formats.
    Memory-mapped mode is recommended for images >1GB.

    Examples
    --------
    >>> img, meta = load_image("sample.tif")
    >>> print(img.shape, img.dtype)
    (3, 2048, 2048) uint16

    >>> # Memory-mapped for large files
    >>> img, meta = load_image("large.tif", memmap=True)

    See Also
    --------
    lib.metadata.get_ome_metadata : Extract detailed OME metadata
    save_tiff : Save image to TIFF format
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with tifffile.TiffFile(str(image_path)) as tif:
            if memmap:
                image = tif.asarray(out='memmap')
            else:
                image = tif.asarray()

            metadata = {
                'shape': image.shape,
                'dtype': image.dtype,
                'ome': None,
                'axes': None
            }

            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                metadata['ome'] = tif.ome_metadata

            return image, metadata

    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}") from e


def save_tiff(
    image: NDArray,
    output_path: str | Path,
    compression: str = 'zlib',
    bigtiff: bool = True,
    **kwargs
) -> Path:
    """Save image array to TIFF file with compression.

    Parameters
    ----------
    image : NDArray
        Image data to save. Can be 2D, 3D, or 4D.
    output_path : str or Path
        Path where TIFF file will be saved.
    compression : str, default='zlib'
        Compression algorithm: 'zlib', 'lzw', 'jpeg', or None.
    bigtiff : bool, default=True
        Use BigTIFF format for files >4GB.
    **kwargs : dict
        Additional arguments passed to tifffile.imwrite.
        Common options: metadata, photometric, imagej.

    Returns
    -------
    Path
        Path object for the saved file.

    Notes
    -----
    Automatically creates parent directories if needed.
    Uses compression by default to save disk space.

    Examples
    --------
    >>> mask = np.zeros((2048, 2048), dtype=np.uint16)
    >>> save_tiff(mask, "output/mask.tif")
    PosixPath('output/mask.tif')

    >>> # Custom compression
    >>> save_tiff(image, "output/results.tif", compression='lzw')

    See Also
    --------
    load_image : Load TIFF image
    lib.metadata.create_ome_xml : Create OME-XML metadata
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    tifffile.imwrite(
        str(output_path),
        image,
        compression=compression,
        bigtiff=bigtiff,
        **kwargs
    )

    return output_path
