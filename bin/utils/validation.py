"""Image value validation and negative value handling utilities.

This module provides functions for detecting, logging, and handling negative
pixel values that can be introduced during image processing (e.g., BaSiC
illumination correction, bicubic interpolation).

Examples
--------
>>> from utils.validation import log_image_stats, clip_negative_values
>>> data = load_image()
>>> log_image_stats(data, "after_basic", logger)
>>> data = clip_negative_values(data, logger)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "log_image_stats",
    "detect_negative_values",
    "clip_negative_values",
    "validate_image_range",
    "detect_wrapped_values",
]


def log_image_stats(
    data: NDArray,
    stage_name: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """Log image statistics for debugging and validation.

    Parameters
    ----------
    data : NDArray
        Image data array.
    stage_name : str
        Name of the processing stage (e.g., "input", "after_basic", "registered").
    logger : logging.Logger, optional
        Logger to use. If None, creates one for this module.

    Returns
    -------
    None
    """
    _logger = logger or logging.getLogger(__name__)

    min_val = data.min()
    max_val = data.max()
    mean_val = data.mean()

    _logger.info(
        f"[{stage_name}] Image stats: dtype={data.dtype}, shape={data.shape}, "
        f"min={min_val}, max={max_val}, mean={mean_val:.2f}"
    )

    # Warn about suspicious values
    if min_val < 0:
        _logger.warning(f"[{stage_name}] Negative values detected: min={min_val}")

    if data.dtype == np.uint16 and max_val > 60000:
        high_count = np.sum(data > 60000)
        high_pct = high_count / data.size * 100
        if high_pct > 0.1:
            _logger.warning(
                f"[{stage_name}] Suspicious high values (potential wrapped negatives): "
                f"{high_count} pixels ({high_pct:.2f}%) > 60000"
            )


def detect_negative_values(
    data: NDArray,
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, int, float]:
    """Detect negative values in image data.

    For signed types (int16, float32), checks for actual negative values.
    For unsigned types (uint8, uint16), negative values cannot exist but
    may have wrapped around - use detect_wrapped_values() for those.

    Parameters
    ----------
    data : NDArray
        Image data array.
    logger : logging.Logger, optional
        Logger to use for warnings.

    Returns
    -------
    has_negatives : bool
        True if negative values were found.
    count : int
        Number of negative pixels.
    percentage : float
        Percentage of pixels that are negative.
    """
    _logger = logger or logging.getLogger(__name__)

    if not np.issubdtype(data.dtype, np.signedinteger) and \
       not np.issubdtype(data.dtype, np.floating):
        # Unsigned type - cannot have actual negatives
        return False, 0, 0.0

    neg_mask = data < 0
    count = int(np.sum(neg_mask))
    percentage = count / data.size * 100

    has_negatives = count > 0

    if has_negatives:
        min_val = data[neg_mask].min()
        _logger.warning(
            f"Detected {count} negative pixels ({percentage:.4f}%), "
            f"min value: {min_val}"
        )

    return has_negatives, count, percentage


def detect_wrapped_values(
    data: NDArray,
    threshold_percentile: float = 99.5,
    min_value_threshold: float = 0.9,
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, int, float]:
    """Detect potentially wrapped negative values in unsigned integer data.

    When negative float values are cast to uint16, they wrap around:
    -1 -> 65535, -100 -> 65435, etc. This function detects such values
    by looking for suspiciously high values.

    Parameters
    ----------
    data : NDArray
        Image data array.
    threshold_percentile : float, default=99.5
        Percentile threshold for detecting outliers.
    min_value_threshold : float, default=0.9
        Fraction of dtype max above which values are suspicious.
    logger : logging.Logger, optional
        Logger to use for warnings.

    Returns
    -------
    has_wrapped : bool
        True if wrapped values were likely detected.
    count : int
        Number of suspicious pixels.
    percentage : float
        Percentage of suspicious pixels.
    """
    _logger = logger or logging.getLogger(__name__)

    if data.dtype == np.uint16:
        dtype_max = 65535
    elif data.dtype == np.uint8:
        dtype_max = 255
    else:
        # Not an unsigned integer type
        return False, 0, 0.0

    # Calculate threshold: values above this are suspicious
    percentile_val = np.percentile(data, threshold_percentile)
    absolute_threshold = dtype_max * min_value_threshold
    threshold = min(percentile_val, absolute_threshold)

    # Only flag if both conditions are met (high value AND above percentile)
    suspicious_mask = data > max(threshold, absolute_threshold)
    count = int(np.sum(suspicious_mask))
    percentage = count / data.size * 100

    has_wrapped = count > 0 and percentage > 0.01  # More than 0.01% suspicious

    if has_wrapped:
        max_val = data[suspicious_mask].max() if count > 0 else 0
        _logger.warning(
            f"Detected {count} potentially wrapped pixels ({percentage:.4f}%), "
            f"max value: {max_val}"
        )

    return has_wrapped, count, percentage


def clip_negative_values(
    data: NDArray,
    logger: Optional[logging.Logger] = None,
    stage_name: str = "unknown"
) -> NDArray:
    """Clip negative values to zero with logging.

    For float or signed integer data, clips values below 0 to 0.
    For unsigned data, no clipping is needed (values can't be negative).

    Parameters
    ----------
    data : NDArray
        Image data array. Modified in place if possible.
    logger : logging.Logger, optional
        Logger to use.
    stage_name : str, default="unknown"
        Name of processing stage for logging.

    Returns
    -------
    NDArray
        Clipped image data (may be same array if modified in place).
    """
    _logger = logger or logging.getLogger(__name__)

    # Check if data can have negative values
    if np.issubdtype(data.dtype, np.unsignedinteger):
        _logger.debug(f"[{stage_name}] Unsigned dtype, no negative clipping needed")
        return data

    # Detect negatives before clipping
    has_neg, count, pct = detect_negative_values(data, _logger)

    if has_neg:
        min_before = data.min()
        data = np.clip(data, 0, None)
        _logger.info(
            f"[{stage_name}] Clipped {count} negative pixels ({pct:.4f}%) to 0. "
            f"Min before: {min_before}, after: {data.min()}"
        )
    else:
        _logger.debug(f"[{stage_name}] No negative values to clip")

    return data


def validate_image_range(
    data: NDArray,
    stage_name: str,
    expected_dtype: Optional[np.dtype] = None,
    logger: Optional[logging.Logger] = None,
    fix_issues: bool = False
) -> Tuple[bool, NDArray]:
    """Validate image values are within expected range.

    Parameters
    ----------
    data : NDArray
        Image data array.
    stage_name : str
        Name of processing stage for logging.
    expected_dtype : np.dtype, optional
        Expected data type. If None, uses data's current dtype.
    logger : logging.Logger, optional
        Logger to use.
    fix_issues : bool, default=False
        If True, clip values to valid range.

    Returns
    -------
    is_valid : bool
        True if all values are within valid range.
    data : NDArray
        Original or fixed data array.
    """
    _logger = logger or logging.getLogger(__name__)

    dtype = expected_dtype or data.dtype
    min_val = data.min()
    max_val = data.max()

    # Determine valid range
    if np.issubdtype(dtype, np.unsignedinteger):
        valid_min = 0
        valid_max = np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.signedinteger):
        valid_min = np.iinfo(dtype).min
        valid_max = np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        # For floats, we typically want non-negative for images
        valid_min = 0
        valid_max = np.finfo(dtype).max
    else:
        _logger.warning(f"[{stage_name}] Unknown dtype: {dtype}")
        return True, data

    issues = []

    if min_val < valid_min:
        issues.append(f"min={min_val} < {valid_min}")

    if max_val > valid_max:
        issues.append(f"max={max_val} > {valid_max}")

    # For uint16, also check for wrapped values
    if dtype == np.uint16:
        has_wrapped, wrap_count, wrap_pct = detect_wrapped_values(data, logger=_logger)
        if has_wrapped:
            issues.append(f"{wrap_count} potentially wrapped values ({wrap_pct:.4f}%)")

    is_valid = len(issues) == 0

    if not is_valid:
        _logger.warning(f"[{stage_name}] Value range issues: {', '.join(issues)}")

        if fix_issues:
            _logger.info(f"[{stage_name}] Fixing value range issues...")

            # Clip to valid range
            data = np.clip(data, valid_min, valid_max)

            # Convert dtype if needed
            if data.dtype != dtype:
                data = data.astype(dtype)

            _logger.info(
                f"[{stage_name}] After fix: min={data.min()}, max={data.max()}"
            )
    else:
        _logger.debug(f"[{stage_name}] Value range OK: min={min_val}, max={max_val}")

    return is_valid, data
