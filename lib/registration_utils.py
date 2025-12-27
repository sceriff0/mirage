"""Shared utilities for image registration.

This module consolidates common functions used across registration scripts
(register.py, register_cpu.py, register_gpu.py) to eliminate code duplication.

Examples
--------
>>> from lib.registration_utils import autoscale, extract_crop_coords
>>> scaled = autoscale(image, low_p=1.0, high_p=99.0)
>>> crops = extract_crop_coords(image.shape, crop_size=2000, overlap=200)

Notes
-----
This module implements the DRY principle by providing single implementations
of functions that were previously duplicated across multiple scripts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "autoscale",
    "extract_crop_coords",
    "calculate_bounds",
    "place_crop_hardcutoff",
    "create_memmaps_for_merge",
    "cleanup_memmaps",
]


def autoscale(
    img: NDArray,
    low_p: float = 1.0,
    high_p: float = 99.0
) -> NDArray[np.uint8]:
    """Normalize image to 0-255 using percentile-based scaling.

    This function implements the same auto-scaling behavior as ImageJ's
    "Auto" brightness adjustment, using percentile clipping to handle
    outliers robustly.

    Parameters
    ----------
    img : NDArray
        Input image array of any numeric dtype.
    low_p : float, default=1.0
        Lower percentile for clipping (0-100).
        Values below this percentile are mapped to 0.
    high_p : float, default=99.0
        Upper percentile for clipping (0-100).
        Values above this percentile are mapped to 255.

    Returns
    -------
    NDArray[np.uint8]
        Scaled image in uint8 format (0-255 range).

    Notes
    -----
    The scaling process:
    1. Compute percentile values at `low_p` and `high_p`
    2. Clip image values to [low, high] range
    3. Linearly scale to [0, 1]
    4. Multiply by 255 and convert to uint8

    Using percentiles instead of min/max makes the scaling robust to
    outliers and saturated pixels.

    Examples
    --------
    >>> img = np.random.randint(0, 4096, size=(100, 100), dtype=np.uint16)
    >>> scaled = autoscale(img)
    >>> scaled.dtype
    dtype('uint8')
    >>> scaled.min(), scaled.max()
    (0, 255)

    Conservative scaling (wider range):
    >>> scaled = autoscale(img, low_p=0.1, high_p=99.9)

    See Also
    --------
    numpy.percentile : Compute percentiles
    numpy.clip : Clip values to range
    """
    lo = np.percentile(img, low_p)
    hi = np.percentile(img, high_p)

    # Avoid division by zero
    range_val = max(hi - lo, 1e-6)

    # Clip and normalize to [0, 1]
    img_normalized = np.clip((img - lo) / range_val, 0, 1)

    # Scale to [0, 255] and convert to uint8
    return (img_normalized * 255).astype(np.uint8)


def extract_crop_coords(
    image_shape: Tuple[int, ...],
    crop_size: int,
    overlap: int
) -> List[Dict[str, int]]:
    """Generate crop coordinates for tiling a large image.

    Parameters
    ----------
    image_shape : Tuple[int, ...]
        Shape of the image. Can be (H, W) or (C, H, W).
        Height and width are extracted automatically.
    crop_size : int
        Size of each crop in pixels (assumes square crops).
    overlap : int
        Overlap between adjacent crops in pixels.

    Returns
    -------
    List[Dict[str, int]]
        List of crop coordinate dictionaries, each containing:
        - 'y': int - Y-coordinate of crop top-left corner
        - 'x': int - X-coordinate of crop top-left corner
        - 'h': int - Crop height (may be less than crop_size at edges)
        - 'w': int - Crop width (may be less than crop_size at edges)

    Notes
    -----
    The function uses a sliding window approach with the following properties:
    - Stride = crop_size - overlap
    - Edge crops are adjusted to fit within image bounds
    - All crops have the same size (crop_size × crop_size) by adjusting
      the starting position of edge crops

    This ensures consistent crop dimensions for batch processing, which is
    important for neural network inference and registration algorithms.

    Examples
    --------
    >>> shape = (3, 2048, 2048)  # 3-channel image
    >>> coords = extract_crop_coords(shape, crop_size=1000, overlap=100)
    >>> len(coords)
    9
    >>> coords[0]
    {'y': 0, 'x': 0, 'h': 1000, 'w': 1000}

    For 2D images:
    >>> shape = (1024, 1024)
    >>> coords = extract_crop_coords(shape, crop_size=512, overlap=50)
    >>> len(coords)
    4

    See Also
    --------
    calculate_bounds : Calculate bounds for crop reconstruction
    place_crop_hardcutoff : Place crop back into full image
    """
    # Extract height and width from shape
    if len(image_shape) == 3:
        _, height, width = image_shape
    elif len(image_shape) == 2:
        height, width = image_shape
    else:
        raise ValueError(
            f"image_shape must be 2D or 3D, got {len(image_shape)}D: {image_shape}"
        )

    stride = crop_size - overlap
    coords = []

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Calculate end coordinates
            y_end = min(y + crop_size, height)
            x_end = min(x + crop_size, width)

            # Adjust start coordinates to maintain crop_size
            # (shifts crop inward at image edges)
            y_start = max(0, y_end - crop_size)
            x_start = max(0, x_end - crop_size)

            # Actual crop dimensions
            h = y_end - y_start
            w = x_end - x_start

            coords.append({
                "y": y_start,
                "x": x_start,
                "h": h,
                "w": w
            })

    return coords


def calculate_bounds(
    start: int,
    crop_dim: int,
    total_dim: int,
    overlap: int
) -> Tuple[int, int, slice]:
    """Calculate bounds for hard-cutoff crop placement.

    This function determines which portion of a crop should be placed
    into the output image to avoid blending artifacts at crop boundaries.

    Parameters
    ----------
    start : int
        Starting position of the crop in the full image.
    crop_dim : int
        Dimension of the crop (height or width).
    total_dim : int
        Total dimension of the full image (height or width).
    overlap : int
        Overlap between adjacent crops.

    Returns
    -------
    start_idx : int
        Starting index in the output image.
    end_idx : int
        Ending index in the output image.
    crop_slice : slice
        Slice object for extracting the valid portion from the crop.

    Notes
    -----
    The hard-cutoff strategy:
    - First crop (start=0): Keep entire crop except overlap//2 at end
    - Last crop (start=total_dim-crop_dim): Keep entire crop except overlap//2 at start
    - Middle crops: Trim overlap//2 from both start and end

    This approach avoids blending artifacts that can occur with weighted
    averaging in overlap regions, at the cost of potentially visible seams.

    Examples
    --------
    >>> # First crop
    >>> start_idx, end_idx, crop_slice = calculate_bounds(0, 1000, 2048, 100)
    >>> start_idx, end_idx
    (0, 950)
    >>> crop_slice
    slice(0, 950, None)

    >>> # Last crop
    >>> start_idx, end_idx, crop_slice = calculate_bounds(1048, 1000, 2048, 100)
    >>> start_idx, end_idx
    (1098, 2048)

    See Also
    --------
    place_crop_hardcutoff : Use these bounds to place crop
    extract_crop_coords : Generate crop coordinates
    """
    if start == 0:
        # First crop: no trimming at start
        start_idx = start
        end_idx = start + crop_dim - overlap // 2
        crop_slice = slice(0, crop_dim - overlap // 2)
    elif start == total_dim - crop_dim:
        # Last crop: no trimming at end
        start_idx = start + overlap // 2
        end_idx = start + crop_dim
        crop_slice = slice(overlap // 2, crop_dim)
    else:
        # Middle crop: trim both sides
        start_idx = start + overlap // 2
        end_idx = start + crop_dim - overlap // 2
        crop_slice = slice(overlap // 2, crop_dim - overlap // 2)

    return start_idx, end_idx, crop_slice


def place_crop_hardcutoff(
    target_mem: np.memmap | NDArray,
    crop: NDArray,
    y: int,
    x: int,
    h: int,
    w: int,
    img_height: int,
    img_width: int,
    overlap: int
) -> None:
    """Place a crop into the target image using hard-cutoff strategy.

    Parameters
    ----------
    target_mem : np.memmap or NDArray
        Target memory-mapped array or numpy array to place crop into.
        Can be 2D (H, W) or 3D (C, H, W).
    crop : NDArray
        Crop to place. Shape must match target_mem dimensions.
    y : int
        Y-coordinate of crop's top-left corner in full image.
    x : int
        X-coordinate of crop's top-left corner in full image.
    h : int
        Crop height.
    w : int
        Crop width.
    img_height : int
        Full image height.
    img_width : int
        Full image width.
    overlap : int
        Overlap between crops used to determine trim regions.

    Returns
    -------
    None
        Modifies target_mem in-place.

    Notes
    -----
    This function handles both 2D and 3D arrays automatically.
    For 3D arrays, it processes each channel separately.

    The crop is trimmed according to hard-cutoff rules (see calculate_bounds)
    before being placed into the target array.

    Examples
    --------
    >>> target = np.zeros((2048, 2048), dtype=np.float32)
    >>> crop = np.random.rand(1000, 1000).astype(np.float32)
    >>> place_crop_hardcutoff(target, crop, y=0, x=0, h=1000, w=1000,
    ...                       img_height=2048, img_width=2048, overlap=100)

    For multi-channel:
    >>> target = np.zeros((3, 2048, 2048), dtype=np.float32)
    >>> crop = np.random.rand(3, 1000, 1000).astype(np.float32)
    >>> place_crop_hardcutoff(target, crop, y=0, x=0, h=1000, w=1000,
    ...                       img_height=2048, img_width=2048, overlap=100)

    See Also
    --------
    calculate_bounds : Calculate trimming bounds
    extract_crop_coords : Generate crop coordinates
    """
    # Calculate bounds for Y and X dimensions
    y_start, y_end, y_slice = calculate_bounds(y, h, img_height, overlap)
    x_start, x_end, x_slice = calculate_bounds(x, w, img_width, overlap)

    if crop.ndim == 3:
        # Multi-channel crop
        for c in range(crop.shape[0]):
            crop_trimmed = crop[c][y_slice, x_slice]
            target_mem[c, y_start:y_end, x_start:x_end] = crop_trimmed
    elif crop.ndim == 2:
        # Single-channel crop
        crop_trimmed = crop[y_slice, x_slice]
        target_mem[y_start:y_end, x_start:x_end] = crop_trimmed
    else:
        raise ValueError(f"crop must be 2D or 3D, got shape {crop.shape}")


def create_memmaps_for_merge(
    output_shape: Tuple[int, ...],
    dtype: np.dtype = np.float32,
    prefix: str = "reg_merge_"
) -> Tuple[np.memmap, np.memmap, Path]:
    """Create memory-mapped arrays for accumulating merged results.

    This function creates temporary memory-mapped files that allow processing
    of images larger than RAM by storing intermediate results on disk.

    Parameters
    ----------
    output_shape : Tuple[int, ...]
        Shape of the output array (C, H, W) or (H, W).
    dtype : np.dtype, default=np.float32
        Data type of the arrays.
    prefix : str, default="reg_merge_"
        Prefix for temporary directory name.

    Returns
    -------
    merged : np.memmap
        Memory-mapped array for accumulating merged values.
        Initialized to zeros.
    weights : np.memmap
        Memory-mapped array for accumulating weights (for weighted averaging).
        Initialized to zeros.
    tmp_dir : Path
        Path to temporary directory containing the memmap files.
        Must be cleaned up with cleanup_memmaps() when done.

    Notes
    -----
    The function creates a temporary directory with two memory-mapped files:
    - merged.npy: Stores accumulated pixel values
    - weights.npy: Stores accumulated weights (if using weighted merging)

    Both arrays are initialized to zero and have the same shape and dtype.

    The caller is responsible for calling cleanup_memmaps() to remove
    temporary files when processing is complete.

    Examples
    --------
    >>> shape = (3, 2048, 2048)
    >>> merged, weights, tmp_dir = create_memmaps_for_merge(shape)
    >>> merged.shape
    (3, 2048, 2048)
    >>> # ... do processing ...
    >>> cleanup_memmaps(tmp_dir)

    See Also
    --------
    cleanup_memmaps : Clean up temporary files
    place_crop_hardcutoff : Place crops into merged array
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix=prefix))

    merged_path = tmp_dir / "merged.npy"
    weights_path = tmp_dir / "weights.npy"

    merged = np.memmap(str(merged_path), dtype=dtype, mode="w+", shape=output_shape)
    weights = np.memmap(str(weights_path), dtype=dtype, mode="w+", shape=output_shape)

    # Initialize to zero
    merged[:] = 0
    weights[:] = 0

    return merged, weights, tmp_dir


def cleanup_memmaps(tmp_dir: Path) -> None:
    """Clean up temporary memmap files.

    Parameters
    ----------
    tmp_dir : Path
        Path to temporary directory containing memmap files.
        This is typically returned by create_memmaps_for_merge().

    Returns
    -------
    None

    Notes
    -----
    This function removes the entire temporary directory tree,
    including all memmap files.

    Errors during cleanup are logged as warnings but do not raise exceptions,
    as cleanup is a best-effort operation.

    Examples
    --------
    >>> merged, weights, tmp_dir = create_memmaps_for_merge((100, 100))
    >>> # ... processing ...
    >>> del merged, weights  # Release file handles
    >>> cleanup_memmaps(tmp_dir)

    See Also
    --------
    create_memmaps_for_merge : Create temporary memmaps
    """
    import shutil
    from lib.logger import get_logger

    logger = get_logger(__name__)

    try:
        shutil.rmtree(tmp_dir)
        logger.debug(f"Cleaned up temporary directory: {tmp_dir}")
    except Exception as e:
        logger.warning(f"Could not clean up temp directory {tmp_dir}: {e}")
