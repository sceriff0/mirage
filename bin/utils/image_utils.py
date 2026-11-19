"""Image loading, saving and axis-normalisation helpers shared by bin/ scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tifffile
from numpy.typing import NDArray
from ome_io import write_tiff

__all__ = [
    "normalize_image_dimensions",
    "load_image",
    "save_tiff",
    "ensure_dir",
]


def normalize_image_dimensions(img: NDArray) -> NDArray:
    """Normalize image to (C, H, W) format.

    The channel axis is taken to be the smallest dimension:
    - 2D (H, W) -> (1, H, W)
    - 3D (H, W, C) -> (C, H, W)
    - 3D (C, H, W), or ambiguous (e.g. square) -> unchanged

    Parameters
    ----------
    img : NDArray
        2D or 3D image.

    Returns
    -------
    NDArray
        Array in (C, H, W) order.

    Raises
    ------
    ValueError
        If the image is not 2D or 3D.
    """
    if img.ndim == 2:
        return img[np.newaxis, ...]

    elif img.ndim == 3:
        if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            return img
        elif img.shape[2] < img.shape[0] and img.shape[2] < img.shape[1]:
            return np.transpose(img, (2, 0, 1))
        else:
            # Ambiguous: keep as (C, H, W).
            return img

    else:
        raise ValueError(
            f"Image must be 2D or 3D, got {img.ndim}D with shape {img.shape}"
        )


def ensure_dir(directory: str | Path) -> Path:
    """Create ``directory`` (and parents) if missing and return it as a Path."""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_image(
    image_path: str | Path, memmap: bool = False
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
        - 'axes': always None (not read)

    Raises
    ------
    FileNotFoundError
        If image_path does not exist.
    ValueError
        If file cannot be read as TIFF.

    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with tifffile.TiffFile(str(image_path)) as tif:
            if memmap:
                image = tif.asarray(out="memmap")
            else:
                image = tif.asarray()

            metadata = {
                "shape": image.shape,
                "dtype": image.dtype,
                "ome": None,
                "axes": None,
            }

            if hasattr(tif, "ome_metadata") and tif.ome_metadata:
                metadata["ome"] = tif.ome_metadata

            return image, metadata

    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}") from e


def save_tiff(
    image: NDArray,
    output_path: str | Path,
    compression: str = "zlib",
    bigtiff: bool = True,
    **kwargs,
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

    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    # Writes go through ome_io (the only tifffile writer in bin/); the defaults are this
    # function's, since ome_io.write_tiff sets none.
    return write_tiff(
        output_path, image, compression=compression, bigtiff=bigtiff, **kwargs
    )
