"""End-to-end test for bin/utils/tiled_pipeline.py — the STARE registration, in one process.

``register_slide`` chains the whole method: global rigid M0 (coarse_align) -> rigid pre-warp ->
per-tile residuals (tile_grid + tile_residual) -> TRE-gated control grid (tiled_manifest) ->
mesh warp (tiled_warp). This is the proof the pieces compose into a working registration: a
synthetically warped moving image is brought back into alignment with the reference, and the
output is non-negative.
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

from tiled_pipeline import register_slide


def _textured(seed=0, n=512):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.uniform(0, 1, size=(n, n)), 2.0)
    return ((img - img.min()) / (img.ptp() + 1e-9)).astype(np.float32)


def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def test_register_slide_realigns_a_synthetically_warped_moving_image():
    from skimage.transform import EuclideanTransform
    from skimage.transform import warp as sk_warp

    ref = _textured()
    # moving = reference displaced by a known small rotation + translation
    tform = EuclideanTransform(rotation=np.deg2rad(2.5), translation=(14.0, -9.0))
    mov = sk_warp(ref, tform, order=1, mode="reflect").astype(np.float32)

    inner = (slice(96, -96), slice(96, -96))  # ignore rotation borders
    before = _corr(ref[inner], mov[inner])

    result = register_slide(ref, mov, tile=128, halo=32, gate_tre=0.0, upsample=10)
    registered = result["registered"]

    after = _corr(ref[inner], registered[inner])
    assert after > before  # registration improved alignment
    assert after > 0.9  # ...to near-perfect correlation
    assert registered.min() >= 0.0  # non-negativity preserved end-to-end
    # the coarse anchor found the rotation (NCC sweep: no correspondences, n_inliers 0)
    m0 = np.asarray(result["entry"]["M0"])
    # sk_warp's tform is the output->input map, i.e. exactly moving -> reference
    assert abs(np.degrees(np.arctan2(m0[1, 0], m0[0, 0])) - 2.5) < 0.5


def test_a_pure_rigid_shift_needs_no_mesh_the_coarse_anchor_absorbs_it():
    """A global translation is fully captured by M0, so every tile is gated out (rigid-only)."""
    ref = _textured(seed=1, n=256)
    from scipy.ndimage import shift as ndi_shift

    mov = ndi_shift(ref, (4.0, -6.0), order=1, mode="reflect").astype(np.float32)
    result = register_slide(ref, mov, tile=128, halo=32, gate_tre=0.5)

    assert np.asarray(result["entry"]["M0"]).shape == (3, 3)
    assert result["entry"]["mesh"] is None  # nothing left for the mesh to fix
    assert len(result["tre_px"]) == len(result["tiles"])
    assert all(t >= 0 for t in result["tre_px"])


def test_local_deformation_is_captured_by_the_mesh():
    """A smooth non-rigid warp M0 cannot capture leaves tiles above the gate -> a mesh appears."""
    from skimage.transform import warp as sk_warp

    ref = _textured(seed=2, n=256)
    n = 256
    yy, xx = np.mgrid[0:n, 0:n]
    amp = 3.0
    map_x = xx + amp * np.sin(2 * np.pi * yy / 90.0)
    map_y = yy + amp * np.cos(2 * np.pi * xx / 90.0)
    mov = sk_warp(ref, np.array([map_y, map_x]), order=1, mode="reflect").astype(
        np.float32
    )

    result = register_slide(ref, mov, tile=64, halo=32, gate_tre=0.5)
    assert (
        result["entry"]["mesh"] is not None
    )  # local deformation -> refinement present
    assert len(result["tre_px"]) == len(result["tiles"])
    assert result["registered"].min() >= 0.0

    # intrinsic TRE: the post-refinement residual (final accuracy) beats the rigid-stage residual.
    assert len(result["tre_after_px"]) == len(result["tiles"])
    rigid = float(np.median(result["tre_px"]))
    final = float(np.median(result["tre_after_px"]))
    assert final < rigid  # the mesh reduced the misalignment
