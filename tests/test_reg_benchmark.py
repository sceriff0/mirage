"""Tests for bin/utils/reg_benchmark.py — the registration accuracy harness.

Measures how well a registered image aligns to the reference by the residual distance between
matched features (a post-registration Target Registration Error that needs no ground truth) plus
intensity correlation. Method-agnostic: run it on VALIS and on tiled outputs and compare the TRE.
Here it is validated on synthetic data where the answer is known — a well-registered pair scores a
small residual, a misaligned pair a large one.
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

from reg_benchmark import benchmark, feature_residual


def _textured(seed=0, n=384):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.uniform(0, 1, size=(n, n)), 2.0)
    return ((img - img.min()) / (img.ptp() + 1e-9)).astype(np.float32)


def test_identical_images_have_near_zero_residual():
    ref = _textured()
    res = feature_residual(ref, ref.copy())
    assert res["n_matches"] >= 8
    assert res["tre_px_median"] < 0.5


def test_a_misaligned_pair_scores_a_large_residual():
    from scipy.ndimage import shift as ndi_shift

    ref = _textured()
    moved = ndi_shift(ref, (12.0, -9.0), mode="reflect")
    res = feature_residual(ref, moved)
    # matched features sit ~15 px apart (hypot(12,9)) — the residual reflects the misalignment
    assert res["tre_px_median"] > 5.0


def test_benchmark_shows_registration_reduced_the_residual():
    """The end-to-end accuracy claim, on synthetic ground truth: DRAPE lowers the TRE.

    The only case in this file that runs a REGISTRATION rather than just scoring one, so it is
    also the only one needing the learned COARSE matcher. The ORB oracle this module measures
    with is deliberately classical and independent of it -- see the note in reg_benchmark.py.
    """
    pytest.importorskip("torch")
    pytest.importorskip("kornia")
    from tiled_pipeline import register_slide

    ref = _textured(seed=3)
    from skimage.transform import EuclideanTransform
    from skimage.transform import warp as sk_warp

    tform = EuclideanTransform(rotation=np.deg2rad(2.0), translation=(11.0, -7.0))
    mov = sk_warp(ref, tform, order=1, mode="reflect").astype(np.float32)
    registered = register_slide(ref, mov, tile=128, halo=32, stride=32)["registered"]

    report = benchmark(ref, mov, registered)
    assert report["after"]["tre_px_median"] < report["before"]["tre_px_median"]
    assert report["after"]["tre_px_median"] < 2.0
    assert report["correlation_after"] > report["correlation_before"]
    assert report["improvement_tre_px"] > 0
