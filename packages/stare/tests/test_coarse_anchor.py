"""COARSE's rigid anchor: NCC rotation sweep, ORB fallback, loud refusal.

The synthetic pairs are the scout's hard cases (research/stare-optimal-design-2026-09-27, the
``make2`` generator of its brute.py), ported from OpenCV to SciPy: a blobby tissue silhouette,
a density field, nuclei dots of which only ``keep`` survive into the moving cycle (0.2 = 80 %
nuclei turnover), 25 % of the moving tissue lost, a gamma of 0.6, an intensity gain and noise.
Log-polar spectrum registration failed 3/4 of these and ORB collapsed on them; the sweep is what recovers them.

Thumbnails are 512 px (the ``low`` tier's refine resolution) to keep the suite fast; the
1024 px numbers are measured separately and recorded in the commit that introduced this file.
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest
from scipy.ndimage import affine_transform, gaussian_filter
from stare import coarse_align as ca

N = 512


def _make(n, seed, keep):
    r = np.random.default_rng(seed)
    m = gaussian_filter(
        np.random.default_rng(1).random((n, n)).astype(np.float32), n / 12
    )
    m = (m > np.percentile(m, 55)).astype(np.float32)
    dens = gaussian_filter(
        np.random.default_rng(2).random((n, n)).astype(np.float32), n / 60
    )
    dens = (dens - dens.min()) / np.ptp(dens)
    base = np.random.default_rng(3).random((n, n)) < 0.03 * dens
    new = r.random((n, n)) < 0.03 * dens
    dots = np.where(r.random((n, n)) < keep, base, new).astype(np.float32)
    return gaussian_filter(dots, 1.2) * m + 0.3 * dens * m


def _m_true(theta, tx, ty, n):
    """moving -> reference: rotate by ``theta`` about the centre, then translate."""
    t = math.radians(theta)
    r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    c = np.array([(n - 1) / 2.0] * 2)
    m = np.eye(3)
    m[:2, :2] = r
    m[:2, 2] = c - r @ c + [tx, ty]
    return m


def _pair(keep, theta, tx, ty, n=N, seed=0):
    """(ref, mov, M_true) in raw uint16-scale counts, ``mov(q) = tissue(M_true q)``."""
    rng = np.random.default_rng(100 + seed)
    ref = _make(n, 10, keep)
    b0 = _make(n, 11, keep)
    loss = gaussian_filter(
        np.random.default_rng(5).random((n, n)).astype(np.float32), n / 20
    )
    b0[loss > np.percentile(loss, 75)] = 0  # 25 % tissue loss
    b0 = b0**0.6  # gamma
    m = _m_true(theta, tx, ty, n)
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    mov = affine_transform(
        b0, swap @ m[:2, :2] @ swap, offset=m[:2, 2][::-1], order=1, cval=0.0
    )
    mov = mov * rng.uniform(0.6, 1.4) + 0.1 * rng.standard_normal(mov.shape)
    return (
        (ref * 60000).astype(np.float32),
        (np.clip(mov, 0, None) * 60000).astype(np.float32),
        m,
    )


def _errors(m_est, m_true, n=N):
    a_est = math.degrees(math.atan2(m_est[1, 0], m_est[0, 0]))
    a_true = math.degrees(math.atan2(m_true[1, 0], m_true[0, 0]))
    ang = abs((a_est - a_true + 180.0) % 360.0 - 180.0)
    c = np.array([(n - 1) / 2.0, (n - 1) / 2.0, 1.0])
    return ang, float(np.linalg.norm((m_est @ c - m_true @ c)[:2]))


CASES = [(0, 20, -15), (37, 60, -45), (178, -75, 40), (-95, 30, 70), (133, -40, -60)]


@pytest.mark.parametrize("keep", [0.2, 1.0], ids=["turnover80", "same_nuclei"])
@pytest.mark.parametrize("theta,tx,ty", CASES, ids=[f"rot{c[0]}" for c in CASES])
def test_the_sweep_recovers_hard_rigid_cases(keep, theta, tx, ty):
    ref, mov, m_true = _pair(keep, theta, tx, ty)
    a = ca.estimate_anchor(ref, mov)
    ang, trans = _errors(a.M, m_true)
    assert ang <= 1.0, f"rotation off by {ang:.2f} deg ({a})"
    assert trans <= 3.0, f"centre off by {trans:.2f} thumbnail px ({a})"
    assert a.method == "ncc_sweep", a
    assert a.peak_ncc >= ca.MIN_PEAK_NCC and a.peak_ratio >= ca.MIN_PEAK_RATIO
    assert a.n_inliers == 0  # the sweep has no correspondences, by contract
    assert 0 < a.residual_px < 5  # the quantisation bound, finite for the JSON


def _sweep_curve(keep, theta, tx, ty):
    """NCC as a function of sweep angle, built exactly as ``_sweep`` builds it."""
    from skimage.transform import downscale_local_mean

    ref, mov, _m = _pair(keep, theta, tx, ty)
    k = max(1, math.ceil(max(*ref.shape, *mov.shape) / ca.SWEEP_SIDE))
    rs = ca._preprocess(downscale_local_mean(ref, (k, k)), ca.BLUR_SIGMA)
    ms = ca._preprocess(downscale_local_mean(mov, (k, k)), ca.BLUR_SIGMA)
    canvas = ca._Canvas.for_shapes(rs.shape, ms.shape)
    corr = ca._Correlator(rs, canvas)
    mc = canvas.place(ms)
    return lambda a: corr.peak(ca._rotate_canvas(mc, a, canvas))[0]


@pytest.mark.parametrize("keep", [0.2, 1.0], ids=["turnover80", "same_nuclei"])
@pytest.mark.parametrize("theta,tx,ty", CASES, ids=[f"rot{c[0]}" for c in CASES])
def test_half_a_sweep_step_off_the_peak_keeps_90_percent_of_the_correlation(
    keep, theta, tx, ty
):
    """The nearest sweep angle is at most SWEEP_STEP_DEG / 2 from the truth; the step is
    safe only if the correlation there still stands >= 90 % of the peak, so the true basin
    cannot lose to a runner-up by quantisation (coarse NOTES, "Rotation step: DERIVE")."""
    f = _sweep_curve(keep, theta, tx, ty)
    fine = np.arange(-0.6, 0.61, 0.2)
    vals = [f(theta + d) for d in fine]
    a0, p0 = theta + fine[int(np.argmax(vals))], max(vals)
    half = ca.SWEEP_STEP_DEG / 2.0
    worst = min(f(a0 + half), f(a0 - half))
    assert worst >= 0.9 * p0, (worst, p0)


def test_estimate_rigid_is_the_three_tuple_view():
    ref, mov, m_true = _pair(1.0, 37, 60, -45)
    m, residual, n_inliers = ca.estimate_rigid(ref, mov)
    assert _errors(m, m_true)[0] <= 1.0
    assert np.isfinite(residual) and n_inliers == 0


@pytest.mark.parametrize("kind", ["noise", "flat"])
def test_a_pair_with_nothing_in_common_is_refused_loudly(kind):
    rng = np.random.default_rng(0)
    if kind == "noise":
        ref = rng.random((N, N)).astype(np.float32) * 60000
        mov = rng.random((N, N)).astype(np.float32) * 60000
    else:
        ref = np.full((N, N), 1200.0, np.float32)
        mov = np.full((N, N), 1200.0, np.float32)
    with pytest.raises(ca.CoarseRefused) as ei:
        ca.estimate_anchor(ref, mov)
    msg = str(ei.value)
    assert "peak" in msg and "ORB" in msg and "inliers" in msg, msg


def test_the_orb_fallback_runs_when_the_sweep_is_not_trusted(monkeypatch):
    """Force the sweep's ratio gate shut: the ORB fallback must take over, and be right."""
    monkeypatch.setattr(ca, "MIN_PEAK_RATIO", 1e9)
    ref, mov, m_true = _pair(1.0, 37, 60, -45)
    a = ca.estimate_anchor(ref, mov)
    assert a.method == "orb", a
    assert a.n_inliers >= ca.ORB_MIN_INLIERS
    ang, trans = _errors(a.M, m_true)
    assert ang <= 1.0 and trans <= 3.0, (ang, trans, a)


def test_a_disagreeing_fallback_loses_to_a_better_scoring_sweep(monkeypatch):
    """Sweep ambiguous but has a candidate; ORB 'confidently' says something else -> the
    candidate with the higher correlation wins, which here is the (correct) sweep."""
    monkeypatch.setattr(ca, "MIN_PEAK_RATIO", 1e9)
    ref, mov, m_true = _pair(1.0, 37, 60, -45)
    wrong = _m_true(37 + 90, 0, 0, N)
    monkeypatch.setattr(ca, "_orb_fallback", lambda r, m, model: (wrong, 1.0, 500))
    a = ca.estimate_anchor(ref, mov)
    assert a.method == "ncc_sweep", a
    assert _errors(a.M, m_true)[0] <= 1.0


def test_the_fallback_without_enough_inliers_is_a_refusal(monkeypatch):
    monkeypatch.setattr(ca, "MIN_PEAK_RATIO", 1e9)
    monkeypatch.setattr(ca, "_orb_fallback", lambda r, m, model: (np.eye(3), 1.0, 5))
    ref, mov, _m = _pair(1.0, 37, 60, -45)
    with pytest.raises(ca.CoarseRefused, match="5 RANSAC inliers"):
        ca.estimate_anchor(ref, mov)


def test_only_the_rigid_model_is_offered():
    ref, mov, _m = _pair(1.0, 0, 0, 0, n=64)
    with pytest.raises(ValueError, match="euclidean"):
        ca.estimate_anchor(ref, mov, model="affine")


def test_rectangular_thumbnails_of_different_sizes():
    """Production thumbnails are non-square and the two slides differ in size."""
    ref, mov, m_true = _pair(1.0, -95, 30, 70)
    ref_r = ref[
        :, 40:
    ]  # 512 x 472: cropping columns keeps reference coordinates (shift 40)
    mov_r = mov[30:, :]  # 482 x 512: cropping rows shifts moving coordinates by 30
    a = ca.estimate_anchor(ref_r, mov_r)
    # M on the crops = T(-40, 0) @ M_true @ T(0, 30)
    t_ref = np.array([[1, 0, -40], [0, 1, 0], [0, 0, 1]], float)
    t_mov = np.array([[1, 0, 0], [0, 1, 30], [0, 0, 1]], float)
    ang, trans = _errors(a.M, t_ref @ m_true @ t_mov)
    assert ang <= 1.0 and trans <= 3.0, (ang, trans, a)


def test_coarse_imports_no_learned_stack():
    """COARSE must run in the tiled image with no torch/kornia at all (and no OpenCV): not at
    import, and not on the call path either (the retired front-end imported torch lazily)."""
    code = (
        "import sys, numpy as np, stare.stages.coarse, stare.coarse_align as ca\n"
        "rng = np.random.default_rng(0)\n"
        "ref = np.zeros((128, 128), np.float32)\n"
        "for _ in range(60):\n"
        "    y, x = rng.integers(10, 118, 2); ref[y - 3:y + 3, x - 3:x + 3] = 1\n"
        "ca.estimate_rigid(ref, np.roll(ref, 4, axis=1))\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
        "('torch', 'kornia', 'cv2'))\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
