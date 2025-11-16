"""Common utilities for pipeline scripts.

This module provides shared helpers to keep scripts DRY (Don't Repeat Yourself)
and KISS (Keep It Simple, Stupid). All functions are small, typed, and testable.

Functions
---------
- File/Directory utilities: ensure_dir, write_text
- Logging utilities: setup_logging, setup_file_logger
- Image I/O utilities: load_image, save_tiff, save_npy
- Pickle utilities: load_pickle, save_pickle
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import tifffile
from numpy.typing import NDArray


__all__ = [
    "ensure_dir",
    "setup_logging",
    "setup_file_logger",
    "write_text",
    "load_image",
    "save_tiff",
    "save_npy",
    "load_pickle",
    "save_pickle",
]


# ==============================================================================
# Directory and File Utilities
# ==============================================================================


def ensure_dir(path: Optional[str]) -> None:
    """Create directory if it doesn't exist.

    Parameters
    ----------
    path : str or None
        Directory path to create. If None, no action is taken.
    """
    if path is None:
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def write_text(path: str, text: str) -> None:
    """Write text to file, creating parent directories if needed.

    Parameters
    ----------
    path : str
        File path to write.
    text : str
        Text content to write.
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, 'w') as f:
        f.write(text)


# ==============================================================================
# Logging Utilities
# ==============================================================================


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for scripts.

    Parameters
    ----------
    level : int, optional
        Logging level (default: logging.INFO).
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def setup_file_logger(log_file: str, level: int = logging.INFO) -> None:
    """Add file handler to root logger.

    Parameters
    ----------
    log_file : str
        Path to log file.
    level : int, optional
        Logging level (default: logging.INFO).
    """
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logging.getLogger().addHandler(handler)


# ==============================================================================
# Image I/O Utilities
# ==============================================================================


def load_image(image_path: str) -> Tuple[NDArray, dict]:
    """Load image and metadata using tifffile.

    Parameters
    ----------
    image_path : str
        Path to image file (TIFF/OME-TIFF).

    Returns
    -------
    image : ndarray
        Image array.
    metadata : dict
        Metadata dictionary (includes 'ome' key if OME-XML present).
    """
    image_data = tifffile.imread(image_path)
    metadata: dict = {}

    try:
        with tifffile.TiffFile(image_path) as tf:
            ome = tf.ome_metadata
            if ome:
                metadata['ome'] = ome
    except Exception:
        pass

    return image_data, metadata


def save_tiff(image: NDArray, output_path: str, **kwargs) -> None:
    """Save image as TIFF file.

    Parameters
    ----------
    image : ndarray
        Image array to save.
    output_path : str
        Output file path.
    **kwargs
        Additional arguments passed to tifffile.imwrite.
    """
    ensure_dir(os.path.dirname(output_path) or ".")
    tifffile.imwrite(output_path, image, **kwargs)


def save_npy(array: NDArray, output_path: str) -> None:
    """Save numpy array to .npy file.

    Parameters
    ----------
    array : ndarray
        Array to save.
    output_path : str
        Output file path.
    """
    ensure_dir(os.path.dirname(output_path) or ".")
    np.save(output_path, array)


# ==============================================================================
# Pickle Utilities
# ==============================================================================


def load_pickle(path: str) -> Any:
    """Load object from pickle file.

    Parameters
    ----------
    path : str
        Path to pickle file.

    Returns
    -------
    object
        Loaded Python object.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str) -> None:
    """Save object to pickle file.

    Parameters
    ----------
    obj : object
        Python object to pickle.
    path : str
        Output file path.
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
