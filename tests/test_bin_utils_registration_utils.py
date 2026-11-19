"""bin/utils/registration_utils.py — percentile autoscaling.

autoscale is what every registration QC overlay is built from. Two of its
properties are the kind that fail silently: a constant image divides by a zero
range (the overlay comes out black or NaN, and the QC panel still renders), and
percentile clipping rather than min/max is the whole reason one saturated pixel
does not flatten a slide to nothing.
"""

from __future__ import annotations

import numpy as np
from utils import registration_utils as ru


def test_the_output_is_uint8_and_spans_the_full_range():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 4096, size=(64, 64)).astype(np.uint16)
    scaled = ru.autoscale(img)
    assert scaled.dtype == np.uint8
    assert scaled.min() == 0 and scaled.max() == 255


def test_a_constant_image_does_not_divide_by_zero():
    """hi == lo. Without the 1e-6 floor this is a RuntimeWarning and an array of
    NaN cast to uint8 -- i.e. a black QC overlay that still writes a valid PNG.

    ``np.errstate(all="raise")`` is required: the returned array is uint8, so an
    ``isfinite`` check on it is vacuously true even after a NaN cast. Raising the
    warning as FloatingPointError is what catches the division by zero."""
    with np.errstate(all="raise"):
        scaled = ru.autoscale(np.full((32, 32), 700, dtype=np.uint16))
    assert scaled.dtype == np.uint8
    assert set(np.unique(scaled)) <= {0, 255}


def test_one_saturated_pixel_does_not_flatten_the_image():
    """The reason this is percentile-based rather than min/max: a single hot
    pixel under min/max scaling maps everything else into the bottom of the
    range, and the overlay goes black."""
    img = np.full((64, 64), 1000, dtype=np.uint16)
    img[0, 0] = 65535
    img[1:, 1:] = np.linspace(500, 1500, 63 * 63).reshape(63, 63).astype(np.uint16)
    scaled = ru.autoscale(img)
    assert scaled.mean() > 40, (
        f"mean is {scaled.mean():.1f} -- one saturated pixel flattened the image, "
        "which is min/max behaviour, not percentile behaviour"
    )


def test_the_percentile_bounds_are_honoured():
    """Values at or below low_p map to 0, at or above high_p map to 255."""
    img = np.arange(0, 10000, dtype=np.uint16).reshape(100, 100)
    scaled = ru.autoscale(img, low_p=10.0, high_p=90.0)
    flat = scaled.ravel()
    assert flat[:900].max() == 0, "the bottom decile did not clip to 0"
    assert flat[9100:].min() == 255, "the top decile did not clip to 255"


def test_a_float_image_is_accepted():
    """Deconvolved channels arrive as float32; autoscale must not assume ints."""
    rng = np.random.default_rng(1)
    scaled = ru.autoscale(rng.random((32, 32)).astype(np.float32))
    assert scaled.dtype == np.uint8
