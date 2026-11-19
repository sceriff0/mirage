"""Shared utilities for image registration.

Used by ``utils/qc.py``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["autoscale"]


def autoscale(
    img: NDArray, low_p: float = 1.0, high_p: float = 99.0
) -> NDArray[np.uint8]:
    """Normalize image to 0-255 using percentile-based scaling.

    Like ImageJ's "Auto" brightness: percentile clipping ignores outliers.

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

    """
    lo = np.percentile(img, low_p)
    hi = np.percentile(img, high_p)

    # Avoid division by zero
    range_val = max(hi - lo, 1e-6)

    img_normalized = np.clip((img - lo) / range_val, 0, 1)

    # Round before the cast so values are not truncated.
    return np.round(img_normalized * 255).astype(np.uint8)
