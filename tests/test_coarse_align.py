"""Tests for bin/utils/coarse_align.py — the global rigid anchor (M0) of the STARE method.

The COARSE step estimates one affine ``M0`` that places a whole moving slide into the reference
frame, absorbing inter-cycle rotation/translation. The numerically load-bearing core is
``estimate_transform_from_matches`` (given point correspondences, solve for the transform + a
residual TRE); it is tested deterministically against transforms it did not compute the same way.
The anchor itself (``estimate_rigid``: NCC rotation sweep + ORB fallback) is exercised as a
smoke test on a synthetic image here; its hard cases live in
packages/stare/tests/test_coarse_anchor.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
    ),
)
pytest.importorskip("skimage")

from coarse_align import estimate_transform_from_matches


def _apply(M, xy):
    h = np.column_stack([xy, np.ones(len(xy))])
    return (h @ np.asarray(M).T)[:, :2]


def test_recovers_a_known_euclidean_transform_from_clean_matches():
    from skimage.transform import EuclideanTransform

    rng = np.random.default_rng(1)
    src = rng.uniform(0, 500, size=(30, 2))
    true = EuclideanTransform(rotation=np.deg2rad(12.0), translation=(7.0, -4.0))
    dst = true(src)  # independent construction, not the estimator's method

    M, residual, n_inliers = estimate_transform_from_matches(
        src, dst, model="euclidean"
    )

    np.testing.assert_allclose(M, true.params, atol=1e-6)
    assert residual == pytest.approx(0.0, abs=1e-6)
    assert n_inliers == 30
    # and the recovered transform actually maps src onto dst
    np.testing.assert_allclose(_apply(M, src), dst, atol=1e-6)


def test_residual_reports_the_target_registration_error_of_the_fit():
    """A single perturbed correspondence must show up as a non-zero residual TRE."""
    from skimage.transform import EuclideanTransform

    rng = np.random.default_rng(2)
    src = rng.uniform(0, 500, size=(40, 2))
    true = EuclideanTransform(rotation=np.deg2rad(5.0), translation=(2.0, 3.0))
    dst = true(src)
    dst[0] += np.array([10.0, 0.0])  # one noisy match

    M, residual, n_inliers = estimate_transform_from_matches(
        src, dst, model="euclidean", robust=False
    )
    # least-squares spreads the 10px error across the fit -> small but clearly non-zero TRE
    assert residual > 0.1


def test_estimate_rigid_smoke_recovers_a_translation_on_a_textured_image():
    pytest.importorskip("scipy")
    from coarse_align import estimate_rigid

    rng = np.random.default_rng(0)
    # a textured reference: smoothed noise gives the matcher something to latch onto
    ref = rng.uniform(0, 1, size=(256, 256)).astype(np.float32)
    from scipy.ndimage import gaussian_filter

    ref = gaussian_filter(ref, 3.0)
    ref = (ref - ref.min()) / (ref.ptp() + 1e-9)
    # moving = reference shifted by a known (dx, dy)
    mov = np.zeros_like(ref)
    dx, dy = 12, -7
    mov[max(0, dy) : 256 + min(0, dy), max(0, dx) : 256 + min(0, dx)] = ref[
        max(0, -dy) : 256 - max(0, dy), max(0, -dx) : 256 - max(0, dx)
    ]

    M, residual, n_inliers = estimate_rigid(ref, mov)
    # M0 maps moving coords back onto the reference: recovers roughly (-dx, -dy)
    assert np.isfinite(residual) and n_inliers >= 0
    np.testing.assert_allclose([M[0, 2], M[1, 2]], [-dx, -dy], atol=2.0)


# ---------------------------------------------------------------------------------------------
# Intensity normalisation of the front-end's input.
#
# `normalize_intensity` is the one place raw microscopy counts become the [0, 1] range every
# consumer here needs (the Otsu tissue mask, ORB's FAST threshold): a float plane carrying raw
# uint16 counts is what `tiled_io.read_decimated` returns, and two cycles' exposures differ.
#
# The smoke test above could never have caught that: it normalises its own input to [0, 1], so
# it only ever exercised the healthy regime. These tests pin the normalisation itself, which is
# cheap and deterministic, plus one end-to-end case at production scale.
#
# The end-to-end raw-scale case below pins that the anchor is scale-invariant in practice.
# ---------------------------------------------------------------------------------------------


def _textured(n=256, scale=1.0):
    """A textured plane in [0, scale] -- the same field regardless of `scale`."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(7)
    a = gaussian_filter(rng.uniform(0, 1, size=(n, n)).astype(np.float32), 3.0)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return (a * scale).astype(np.float32)


def test_normalize_intensity_puts_raw_uint16_counts_into_unit_range():
    from coarse_align import normalize_intensity

    raw = _textured(scale=65535.0)
    out = normalize_intensity(raw)

    assert out.min() >= 0.0 and out.max() <= 1.0
    # and it actually used the range -- a degenerate all-zeros return would also satisfy the above
    assert out.max() > 0.5


def test_normalize_intensity_survives_a_constant_plane():
    """A blank/constant channel must not produce NaNs (0/0) -- there is nothing to match either
    way."""
    from coarse_align import normalize_intensity

    out = normalize_intensity(np.full((32, 32), 4096.0, dtype=np.float32))
    assert np.all(np.isfinite(out))


def test_estimate_rigid_recovers_a_translation_on_raw_uint16_scale_input():
    """The smoke test above, but in the scale production actually feeds it."""
    pytest.importorskip("scipy")
    from coarse_align import estimate_rigid

    ref = _textured(scale=65535.0)
    mov = np.zeros_like(ref)
    dx, dy = 12, -7
    mov[max(0, dy) : 256 + min(0, dy), max(0, dx) : 256 + min(0, dx)] = ref[
        max(0, -dy) : 256 - max(0, dy), max(0, -dx) : 256 - max(0, dx)
    ]

    M, residual, _n = estimate_rigid(ref, mov)
    assert np.isfinite(residual)
    np.testing.assert_allclose([M[0, 2], M[1, 2]], [-dx, -dy], atol=2.0)
