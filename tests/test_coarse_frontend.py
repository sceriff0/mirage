"""Tests for the COARSE global-alignment front-end in bin/utils/coarse_align.py.

There is exactly ONE front-end, reached through ``estimate_rigid`` / ``estimate_anchor``: an
FFT NCC rotation sweep refined at the thumbnail, with a scikit-image ORB fallback and a loud
refusal (packages/stare/tests/test_coarse_anchor.py holds the hard-case and fallback tests).
It replaced DISK + LightGlue on 2026-09-27. The three classical CPU alternatives and the
``estimate_affine`` dispatch table deleted for v1.0.0 stay deleted -- the ORB of the fallback
is a different, internal shape (``_orb_fallback``), not the old selectable front-end --
and ``test_the_deleted_frontends_are_really_gone`` below is what stops one coming back
without a decision.

Nothing here skips: the anchor needs no torch (the STARE image, containers/stare, has none),
and the suite step's MIRAGE_STRICT_SKIPS floor fails CI on any unexpected skip.

Import note: coarse_align.py does ``from logger import get_logger`` at module scope (an
unqualified import resolved against ``bin/utils`` on sys.path directly, the same convention
tests/test_coarse_align.py and tests/test_tiled_coarse_thumbnail.py already use) -- so this
module is imported the same way those sibling test files do, rather than via
``bin.utils.coarse_align``, which would fail that inner import when this file is run alone.

Fixture note: the roll signs below are ``(-7, +5)``: M0 maps mov onto ref, so a moving image
rolled by (+5 x, -7 y) is recovered as a (-5, +7) translation.
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

from coarse_align import estimate_rigid


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    ref = np.zeros((512, 512), dtype=np.float32)
    for _ in range(120):
        y, x = rng.integers(20, 492, 2)
        ref[y - 4 : y + 4, x - 4 : x + 4] = rng.uniform(0.5, 1.0)
    mov = np.roll(np.roll(ref, -7, axis=0), 5, axis=1)
    return ref, mov


@pytest.fixture
def rect_pair():
    """Non-square 256x512 fixture (rows x cols), same blob-fixture style as ``pair``.

    A SQUARE fixture cannot catch a (H, W) vs (W, H) ``image_size`` ordering bug: the two
    orderings are the same tuple when H == W. This one is 256 rows x 512 cols, and stays
    at or below 512 px in the larger dimension.
    """
    rng = np.random.default_rng(1)
    ref = np.zeros((256, 512), dtype=np.float32)
    for _ in range(120):
        y = rng.integers(20, 236)
        x = rng.integers(20, 492)
        ref[y - 4 : y + 4, x - 4 : x + 4] = rng.uniform(0.5, 1.0)
    mov = np.roll(np.roll(ref, -7, axis=0), 5, axis=1)
    return ref, mov


def test_the_deleted_frontends_are_really_gone():
    """A one-value dispatch table is dead config. These three were deleted for v1.0.0;
    this fails if one is reintroduced without a decision."""
    import coarse_align

    for gone in (
        "FRONTENDS",
        "_FRONTENDS",
        "estimate_affine",
        "normalize_for_orb",
        "_orb_features",
        "_frontend_orb",
        "_frontend_sift",
        "_frontend_fourier_mellin",
    ):
        assert not hasattr(coarse_align, gone), f"coarse_align.{gone} still exists"


def test_the_front_end_needs_no_torch(pair, monkeypatch):
    """The anchor must work with torch/kornia unimportable -- the tiled image is dropping them."""
    import builtins

    real = builtins.__import__

    def no_torch(name, *a, **k):
        if name.startswith(("torch", "kornia")):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    ref, mov = pair
    m0, _residual, _n = estimate_rigid(ref, mov)
    np.testing.assert_allclose(m0[:2, 2], [-5, 7], atol=1.5)


def test_the_anchor_recovers_a_pure_shift(pair):
    ref, mov = pair
    m0, residual_px, n_inliers = estimate_rigid(ref, mov)
    np.testing.assert_allclose(m0[0, 2], -5, atol=1.5)
    np.testing.assert_allclose(m0[1, 2], 7, atol=1.5)
    np.testing.assert_allclose(m0[:2, :2], np.eye(2), atol=0.01)
    assert n_inliers == 0  # the NCC sweep carries no correspondences
    assert np.isfinite(residual_px) and residual_px < 2.0


def test_the_anchor_recovers_a_pure_shift_on_a_rectangular_thumbnail(rect_pair):
    """Every production thumbnail is non-square (one shared decimation factor preserves the
    aspect ratio), so the square ``pair`` fixture cannot catch an (H, W) vs (W, H) mix-up in
    the canvas placement or the shift unwrapping."""
    ref, mov = rect_pair
    m0, residual_px, _n = estimate_rigid(ref, mov)
    np.testing.assert_allclose(m0[0, 2], -5, atol=1.5)
    np.testing.assert_allclose(m0[1, 2], 7, atol=1.5)
    np.testing.assert_allclose(m0[:2, :2], np.eye(2), atol=0.01)
    assert np.isfinite(residual_px) and residual_px < 2.0
