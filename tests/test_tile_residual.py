"""Tests for bin/utils/tile_residual.py — the per-tile refinement primitive of DRAPE.

After the global M0 anchor, each tile's moving crop is near-aligned to the reference crop, so the
leftover misalignment is a small (near-pure-translation) residual. Phase correlation recovers it
sub-pixel. The returned (dx, dy) is the mesh control-point displacement for that tile — the amount
to add to a moving coordinate to land on the reference — and its magnitude is the tile's local
rigid-stage TRE, which gates whether the tile needs refining.
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
pytest.importorskip("scipy")

from tile_residual import residual_displacement


def _textured(seed=0, n=128):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.uniform(0, 1, size=(n, n)), 2.0)
    return ((img - img.min()) / (img.ptp() + 1e-9)).astype(np.float32)


def test_recovers_a_known_subpixel_shift_as_the_control_displacement():
    from scipy.ndimage import shift as ndi_shift

    ref = _textured()
    row_s, col_s = 2.5, -3.0  # move ref content by (row=+2.5, col=-3.0) to make mov
    mov = ndi_shift(ref, shift=(row_s, col_s), order=1, mode="reflect")

    dx, dy, tre, error = residual_displacement(ref, mov, upsample=20)

    # (dx, dy) is the displacement that maps a moving coord back onto the reference:
    # ref = mov - (col_s, row_s)  ->  F = (-col_s, -row_s)
    assert dx == pytest.approx(-col_s, abs=0.15)
    assert dy == pytest.approx(-row_s, abs=0.15)
    # a genuine match is confident (STARE v1 gated tiles on this error; v2 validates the
    # per-window vectors by peak ratio and keeps this function as the whole-tile oracle)
    assert error < 0.5, (
        f"a real shift between two crops of one field scored error={error:.4f}"
    )


def test_local_tre_is_the_magnitude_of_the_residual_and_is_small_when_aligned():
    ref = _textured()
    dx, dy, tre, error = residual_displacement(ref, ref.copy(), upsample=20)
    # an already-aligned tile has ~zero residual displacement and ~zero local TRE
    assert abs(dx) < 0.1 and abs(dy) < 0.1
    assert tre == pytest.approx(np.hypot(dx, dy), abs=1e-6)
    assert tre < 0.1
    # a tile correlated against itself is the most confident match there is
    assert error == pytest.approx(0.0, abs=1e-6), (
        f"self-correlation scored error={error!r}"
    )
