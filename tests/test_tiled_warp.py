"""Tests for bin/utils/tiled_warp.py — the image warp of the STARE method (the WARP_TILE core).

Warps a moving image into the reference frame through the same transform the QC warper uses: the
global affine ``M0`` plus the smooth mesh residual. Uses the inverse map + bilinear resampling, so
a non-negative image stays non-negative — the guarantee downstream quantification relies on.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
    ),
)

from mesh_field import MeshField
from tiled_warp import source_region, warp_image


def _translation(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def test_identity_transform_returns_the_image_unchanged():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 100, size=(20, 24))
    out = warp_image(img, np.eye(3), None, (20, 24))
    np.testing.assert_allclose(out, img, atol=1e-9)


def test_rigid_translation_moves_content_by_M0():
    img = np.zeros((20, 20))
    img[5, 7] = 100.0  # (x=7, y=5)
    # M0 forward maps moving -> reference: (7,5) -> (10, 3)
    out = warp_image(img, _translation(3.0, -2.0), None, (20, 20))
    assert np.unravel_index(np.argmax(out), out.shape) == (3, 10)
    assert out.max() == 100.0


def test_mesh_residual_is_applied_on_top_of_the_rigid_warp():
    img = np.zeros((30, 30))
    img[10, 10] = 100.0  # (x=10, y=10)
    # rigid puts it at (13, 12); a constant mesh of (+2, +1) finishes at (15, 13)
    mesh = MeshField(
        np.array([0.0, 100.0]),
        np.array([0.0, 100.0]),
        np.array([[[2.0, 1.0], [2.0, 1.0]], [[2.0, 1.0], [2.0, 1.0]]]),
    )
    out = warp_image(img, _translation(3.0, 2.0), mesh, (30, 30))
    assert np.unravel_index(np.argmax(out), out.shape) == (13, 15)


def test_warp_never_produces_negative_pixels():
    rng = np.random.default_rng(3)
    img = rng.uniform(0.0, 65535.0, size=(40, 40))  # non-negative uint16-range
    mesh = MeshField(
        np.array([0.0, 40.0]),
        np.array([0.0, 40.0]),
        np.array([[[3.0, -2.0], [1.0, 4.0]], [[-4.0, 1.0], [2.0, 2.0]]]),
    )
    out = warp_image(img, _translation(1.5, -0.5), mesh, (40, 40))
    assert out.min() >= 0.0
    assert out.max() <= img.max() + 1e-9


def test_out_origin_warps_a_subregion_identical_to_cropping_the_full_warp():
    """The per-tile stitch guarantee: warping tile (out_origin, out_shape) equals the matching
    crop of the whole-image warp — so tiles reassemble seamlessly and memory stays per-tile."""
    rng = np.random.default_rng(5)
    img = rng.uniform(0, 100, size=(40, 40))
    mesh = MeshField(
        np.array([0.0, 40.0]),
        np.array([0.0, 40.0]),
        np.array([[[1.5, -1.0], [2.0, 0.5]], [[-1.0, 2.0], [0.5, 1.5]]]),
    )
    full = warp_image(img, _translation(2.3, -1.7), mesh, (40, 40))
    # rows 8..24, cols 20..32
    tile = warp_image(img, _translation(2.3, -1.7), mesh, (16, 12), out_origin=(20, 8))
    np.testing.assert_allclose(tile, full[8:24, 20:32], atol=1e-9)


def test_src_origin_samples_from_a_crop_identical_to_the_full_image():
    """Streaming stitch reads only a source crop per tile; warping from the crop (with src_origin)
    must equal warping from the whole image, so the streamed output is bit-identical."""
    rng = np.random.default_rng(7)
    img = np.asarray(rng.uniform(0, 100, size=(40, 40)))
    m0 = _translation(
        2.0, -3.0
    )  # forward moving->ref; source of the tile is offset by (-2,+3)
    # output tile: cols 20..32, rows 8..24  ->  source cols ~18..30, rows ~11..27
    full = warp_image(img, m0, None, (16, 12), out_origin=(20, 8))
    sx0, sy0, sx1, sy1 = 14, 7, 34, 30  # a crop comfortably covering that source region
    crop = img[sy0:sy1, sx0:sx1]
    tiled = warp_image(
        crop, m0, None, (16, 12), out_origin=(20, 8), src_origin=(sx0, sy0)
    )
    np.testing.assert_allclose(tiled, full, atol=1e-9)


def test_source_region_bounds_the_moving_pixels_a_tile_needs():
    # identity M0, no mesh: an output tile maps to the same source box, plus the margin, clamped.
    x0, y0, x1, y1 = source_region(
        np.eye(3),
        None,
        out_origin=(20, 8),
        out_shape=(16, 12),
        margin=4,
        src_shape=(40, 40),
    )
    # cols 20..32 -/+ margin -> 16..36 ; rows 8..24 -/+ margin -> 4..28
    assert (x0, y0, x1, y1) == (16, 4, 36, 28)
    # clamps to the image at the border
    cx0, cy0, cx1, cy1 = source_region(
        np.eye(3),
        None,
        out_origin=(0, 0),
        out_shape=(10, 10),
        margin=5,
        src_shape=(8, 8),
    )
    assert (cx0, cy0) == (0, 0) and cx1 <= 8 and cy1 <= 8


def test_multichannel_is_warped_per_channel():
    img = np.zeros((16, 16, 3))
    img[4, 4, 1] = 50.0  # (x=4, y=4) in channel 1
    out = warp_image(img, _translation(2.0, 3.0), None, (16, 16))
    assert out.shape == (16, 16, 3)
    # channel 1 impulse moved to (x=6, y=7) -> (row=7, col=6)
    assert np.unravel_index(np.argmax(out[..., 1]), out[..., 1].shape) == (7, 6)
