"""Tests for bin/utils/mesh_field.py — the smooth control-grid displacement field and the
bilinear intensity resampler used by the tiled ('STARE') registration method.

Two guarantees are pinned here:
  * the field is a single continuous function of position (exact at control nodes, bilinear
    between them), so warping through it cannot tear a cell at a tile seam; and
  * the intensity resampler is a convex combination of source pixels, so a non-negative image
    stays non-negative — no bicubic overshoot that would corrupt downstream quantification.
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

from mesh_field import MeshField, resample_bilinear


# ── the field ──────────────────────────────────────────────────────────────────
def test_zero_field_is_the_identity_warp():
    xs = np.array([0.0, 10.0, 20.0])
    ys = np.array([0.0, 10.0])
    disp = np.zeros((ys.size, xs.size, 2))
    mf = MeshField(xs, ys, disp)

    xy = np.array([[5.0, 5.0], [12.0, 3.0], [0.0, 0.0], [20.0, 10.0]])
    np.testing.assert_allclose(mf.displacement(xy), 0.0)
    np.testing.assert_allclose(mf.warp_points(xy), xy)


def test_field_is_exact_at_control_nodes():
    xs = np.array([0.0, 10.0, 20.0])
    ys = np.array([0.0, 10.0])
    disp = np.zeros((2, 3, 2))
    disp[0, 1] = [3.0, -2.0]  # node (x=10, y=0)
    disp[1, 2] = [-1.0, 5.0]  # node (x=20, y=10)
    mf = MeshField(xs, ys, disp)

    got = mf.displacement(np.array([[10.0, 0.0], [20.0, 10.0], [0.0, 0.0]]))
    np.testing.assert_allclose(got, [[3.0, -2.0], [-1.0, 5.0], [0.0, 0.0]])


def test_cell_centre_is_the_mean_of_its_four_corners():
    """The continuity guarantee: between nodes the field is a smooth bilinear blend, so a point
    on a tile boundary has one well-defined displacement (no seam)."""
    xs = np.array([0.0, 10.0])
    ys = np.array([0.0, 10.0])
    disp = np.array(
        [
            [[0.0, 0.0], [4.0, 0.0]],  # y=0 row: corners (0,0)->0, (10,0)->(4,0)
            [[0.0, 2.0], [8.0, 6.0]],  # y=10 row: (0,10)->(0,2), (10,10)->(8,6)
        ]
    )
    mf = MeshField(xs, ys, disp)

    # centre (5,5): mean of the four corners = ((0+4+0+8)/4, (0+0+2+6)/4) = (3, 2)
    np.testing.assert_allclose(mf.displacement(np.array([[5.0, 5.0]])), [[3.0, 2.0]])
    # edge midpoint (5,0): mean of the two x-corners on that edge = (2, 0)
    np.testing.assert_allclose(mf.displacement(np.array([[5.0, 0.0]])), [[2.0, 0.0]])


def test_points_outside_the_grid_clamp_to_the_boundary_displacement():
    xs = np.array([0.0, 10.0])
    ys = np.array([0.0, 10.0])
    disp = np.zeros((2, 2, 2))
    disp[1, 1] = [7.0, -3.0]  # only the (10,10) corner is displaced
    mf = MeshField(xs, ys, disp)

    # a point past (10,10) extrapolates as the constant boundary value, not a runaway
    np.testing.assert_allclose(
        mf.displacement(np.array([[999.0, 999.0]])), [[7.0, -3.0]]
    )


# ── the intensity resampler (non-negativity guarantee) ──────────────────────────
def test_resample_matches_analytic_bilinear():
    # xy is (x=col, y=row). Centre of a 2x2 image = mean of the four pixels.
    img = np.array([[0.0, 10.0], [20.0, 40.0]])
    got = resample_bilinear(img, np.array([[0.5, 0.5]]))
    np.testing.assert_allclose(got, [17.5])
    # exact at pixel centres
    corners = resample_bilinear(
        img, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    )
    np.testing.assert_allclose(corners, [0.0, 10.0, 20.0, 40.0])


def test_resample_is_convex_so_nonnegative_input_stays_nonnegative():
    """Bilinear is a convex combination of four source pixels, so the output can never leave
    ``[min, max]`` of the inputs — the property bicubic/Lanczos would break with overshoot, and
    the reason quantification never sees a negative pixel."""
    rng = np.random.default_rng(0)
    img = rng.uniform(
        0.0, 65535.0, size=(37, 41)
    )  # non-negative, like a uint16 channel
    xy = rng.uniform(
        -5.0, 45.0, size=(5000, 2)
    )  # some coords fall outside → edge-clamped
    out = resample_bilinear(img, xy)
    assert out.min() >= 0.0
    assert out.max() <= img.max() + 1e-9  # never overshoots the source maximum


def test_resample_preserves_multichannel_shape():
    img = np.zeros((4, 4, 3))
    img[..., 1] = 5.0
    out = resample_bilinear(img, np.array([[1.5, 1.5], [0.0, 0.0]]))
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[:, 1], [5.0, 5.0])
    np.testing.assert_allclose(out[:, 0], [0.0, 0.0])
