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
    "detect_wrapped_values",
    "StreamingNegativeClip",
    "StreamingImageStats",
    "StreamingWrappedValues",
]


def log_image_stats(
    data: NDArray, stage_name: str, logger: Optional[logging.Logger] = None
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
    data: NDArray, logger: Optional[logging.Logger] = None
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

    if not np.issubdtype(data.dtype, np.signedinteger) and not np.issubdtype(
        data.dtype, np.floating
    ):
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
    logger: Optional[logging.Logger] = None,
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
    absolute_threshold = dtype_max * min_value_threshold
    # Flag values above absolute threshold as suspicious
    suspicious_mask = data > absolute_threshold
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
    data: NDArray, logger: Optional[logging.Logger] = None, stage_name: str = "unknown"
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


# ---------------------------------------------------------------------------
# Streaming counterparts
#
# The whole-array functions above log one line computed over the whole image. Called per
# chunk they would log one line per chunk, with percentages against the chunk. These
# classes fold chunk statistics and log the same line(s) once, at the end; use them from
# chunked loops.
# ---------------------------------------------------------------------------


class StreamingNegativeClip:
    """Accumulates ``clip_negative_values``' whole-image statistics across chunked reads,
    and logs the same line(s) as one whole-array call, once, after the last chunk.

    Counts and sizes are summed and the percentage computed once from the sums; the mins
    are true minima across chunks. Clipping every chunk unconditionally is pixel-identical
    to the whole-array version, since clipping nonnegative data is a no-op.

    Pass only real pixels: padding in an edge chunk would inflate ``size`` and change the
    reported percentage.
    """

    def __init__(self, dtype, logger, stage_name):
        self._logger = logger
        self._stage = stage_name
        self._active = not np.issubdtype(dtype, np.unsignedinteger)
        self._count = 0
        self._size = 0
        self._min_before = None
        self._min_after = None
        if not self._active:
            logger.debug(f"[{stage_name}] Unsigned dtype, no negative clipping needed")

    def process(self, data):
        """Clip one chunk in place-equivalent fashion and fold its stats in."""
        if not self._active:
            return data

        min_before = data.min()
        count = int(np.sum(data < 0))

        self._count += count
        self._size += data.size
        if self._min_before is None or min_before < self._min_before:
            self._min_before = min_before

        clipped = np.clip(data, 0, None)
        min_after = clipped.min()
        if self._min_after is None or min_after < self._min_after:
            self._min_after = min_after

        return clipped

    def finalize(self):
        """Log the aggregate once, matching detect_negative_values' and
        clip_negative_values' whole-array log lines."""
        if not self._active:
            return

        if self._count == 0:
            self._logger.debug(f"[{self._stage}] No negative values to clip")
            return

        pct = self._count / self._size * 100
        # Matches detect_negative_values' warning.
        self._logger.warning(
            f"Detected {self._count} negative pixels ({pct:.4f}%), "
            f"min value: {self._min_before}"
        )
        # Matches clip_negative_values' info line.
        self._logger.info(
            f"[{self._stage}] Clipped {self._count} negative pixels ({pct:.4f}%) to 0. "
            f"Min before: {self._min_before}, after: {self._min_after}"
        )


class StreamingImageStats:
    """The chunked counterpart of :func:`log_image_stats`.

    Emits the same three possible lines (stats, negative-value warning, wrapped-value
    warning) from folded per-chunk aggregates. ``dtype`` and ``shape`` come from the caller,
    since a chunk knows neither.

    The mean is accumulated in float64, as ``ndarray.mean()`` does for integer arrays, so
    the printed mean matches the whole-array call on integer data.

    Pass only real pixels: padded chunk edges would enter the mean and the percentage.
    """

    def __init__(self, dtype, shape, stage_name, logger=None):
        self._logger = logger or logging.getLogger(__name__)
        self._dtype = np.dtype(dtype)
        self._shape = tuple(shape)
        self._stage = stage_name
        self._min = None
        self._max = None
        self._sum = 0.0
        self._size = 0
        self._high_count = 0

    def process(self, data):
        """Fold one chunk's statistics in. Returns the chunk untouched."""
        if data.size == 0:
            return data
        chunk_min = data.min()
        chunk_max = data.max()
        self._min = chunk_min if self._min is None else min(self._min, chunk_min)
        self._max = chunk_max if self._max is None else max(self._max, chunk_max)
        self._sum += float(np.sum(data, dtype=np.float64))
        self._size += data.size
        if self._dtype == np.uint16:
            self._high_count += int(np.count_nonzero(data > 60000))
        return data

    def finalize(self):
        """Log the aggregate lines, exactly once, matching log_image_stats."""
        if self._size == 0:
            return
        mean_val = self._sum / self._size

        self._logger.info(
            f"[{self._stage}] Image stats: dtype={self._dtype}, shape={self._shape}, "
            f"min={self._min}, max={self._max}, mean={mean_val:.2f}"
        )

        if self._min < 0:
            self._logger.warning(
                f"[{self._stage}] Negative values detected: min={self._min}"
            )

        if self._dtype == np.uint16 and self._max > 60000:
            high_pct = self._high_count / self._size * 100
            if high_pct > 0.1:
                self._logger.warning(
                    f"[{self._stage}] Suspicious high values (potential wrapped "
                    f"negatives): {self._high_count} pixels ({high_pct:.2f}%) > 60000"
                )


class StreamingWrappedValues:
    """The chunked counterpart of :func:`detect_wrapped_values`.

    Count and size are summed across chunks and the percentage computed once at the end:
    a per-chunk percentage would change both the number and, at the 0.01% threshold,
    whether a warning fires. ``max value`` is the running max over suspicious pixels.

    ``finalize`` returns the same ``(has_wrapped, count, percentage)`` triple and logs the
    same warning as the whole-array function.

    Pass only real pixels: padded chunk edges would enter ``size`` and move the percentage.
    """

    def __init__(self, dtype, min_value_threshold: float = 0.9, logger=None):
        self._logger = logger or logging.getLogger(__name__)
        dt = np.dtype(dtype)
        # Same dtype table as detect_wrapped_values: anything else is not an unsigned
        # integer type and the function returns "no" without looking at the pixels.
        if dt == np.uint16:
            dtype_max = 65535
        elif dt == np.uint8:
            dtype_max = 255
        else:
            dtype_max = None
        self._threshold = None if dtype_max is None else dtype_max * min_value_threshold
        self._count = 0
        self._size = 0
        self._max = 0

    def process(self, data):
        """Fold one chunk in. Returns the chunk untouched."""
        if self._threshold is None or data.size == 0:
            return data
        suspicious_mask = data > self._threshold
        count = int(np.sum(suspicious_mask))
        self._count += count
        self._size += data.size
        if count:
            chunk_max = data[suspicious_mask].max()
            if chunk_max > self._max:
                self._max = chunk_max
        return data

    def finalize(self):
        """Return ``(has_wrapped, count, percentage)`` and log detect_wrapped_values' warning."""
        if self._threshold is None or self._size == 0:
            return False, 0, 0.0

        percentage = self._count / self._size * 100
        has_wrapped = (
            self._count > 0 and percentage > 0.01
        )  # More than 0.01% suspicious

        if has_wrapped:
            self._logger.warning(
                f"Detected {self._count} potentially wrapped pixels ({percentage:.4f}%), "
                f"max value: {self._max}"
            )

        return has_wrapped, self._count, percentage
