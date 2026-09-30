"""stare.solve's ``dctpls`` on REG_TILE's window-vector lattice.

When the controls carry ``vectors`` (stare.vector_grid), SOLVE lays every vector on the
slide-global lattice, applies no correlation-error gate (it drops good window vectors; REG_TILE
already applied the peak-ratio floor), calibrates each vector's sigma from block-CV held-out
residuals binned by peak ratio, and re-indexes the field to the frame the stitch evaluates it
in. research/stare-sota-review-2026-09-27.md Part C §4, §6.
"""

from __future__ import annotations

import numpy as np
import pytest
from stare import solve
from stare import vector_grid as vg
from stare.mesh_field import MeshField
from stare.tile_grid import tile_grid
from stare_synthetic import make_pair, moving_point, tissue

S = 128


def _control(ix, iy, vectors, rejected=(), error=0.05):
    return {
        "ix": ix,
        "iy": iy,
        "cx": 0.0,
        "cy": 0.0,
        "dx": 0.0,
        "dy": 0.0,
        "tre": 0.0,
        "error": error,
        "lattice": {"stride": S, "window": 2 * S, "origin": S},
        "vectors": [list(v) for v in vectors],
        "rejected": [list(r) for r in rejected],
    }


def _vec(kx, ky, dx, dy, pr=5.0):
    return [kx, ky, S * (kx + 1.0), S * (ky + 1.0), dx, dy, pr, 2.0, 1.0]


def _field_controls(n, fn, noise_of_pr, seed=0, hole=0.3, tile=16):
    """An n x n vector lattice split into ``tile``-node tiles; noise sd depends on PR."""
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    X = S * (k + 1.0)
    out = []
    for ty in range(0, n, tile):
        for tx in range(0, n, tile):
            vec, rej = [], []
            for iy in range(ty, min(n, ty + tile)):
                for ix in range(tx, min(n, tx + tile)):
                    if rng.random() < hole:
                        rej.append([ix, iy, X[ix], X[iy], 0.0, None])
                        continue
                    pr = float(rng.uniform(1.2, 8.0))
                    dx, dy = fn(X[ix], X[iy])
                    sd = noise_of_pr(pr)
                    vec.append(
                        _vec(ix, iy, dx + rng.normal(0, sd), dy + rng.normal(0, sd), pr)
                    )
            out.append(_control(tx // tile, ty // tile, vec, rej))
    return out


def test_vectors_from_all_tiles_are_laid_on_one_lattice():
    c0 = _control(0, 0, [_vec(0, 0, 1.0, 2.0), _vec(1, 0, 3.0, 4.0)])
    c1 = _control(
        1, 0, [_vec(3, 1, 5.0, 6.0)], rejected=[[2, 2, 384.0, 384.0, 0.1, None]]
    )
    gx, gy, Y, W0, PR, counts, lattice = solve._lattice_from_vectors([c0, c1], None)
    # node k at origin + k * stride, the extent spans the rejected node too
    assert list(gx) == [128.0, 256.0, 384.0, 512.0]
    assert list(gy) == [128.0, 256.0, 384.0]
    assert Y[0, 1].tolist() == [3.0, 4.0] and Y[1, 3].tolist() == [5.0, 6.0]
    assert W0.sum() == 3 and W0[2, 2] == 0
    assert lattice["shape"] == [3, 4] and counts["n_vectors"] == 3


def test_a_node_reported_twice_takes_the_mean_vector():
    a = _control(0, 0, [_vec(1, 1, 1.0, 1.0, pr=2.0)])
    b = _control(1, 0, [_vec(1, 1, 3.0, -1.0, pr=4.0)])
    _gx, _gy, Y, W0, PR, counts, _lat = solve._lattice_from_vectors([a, b], None)
    assert Y[0, 0].tolist() == [2.0, 0.0]
    assert PR[0, 0] == 4.0 and counts["duplicates"] == 1


def test_the_tile_level_correlation_error_is_not_a_gate():
    """Top-level error 0.9999 rejected the TILE under STARE v1; its window vectors count."""
    vecs = [_vec(kx, ky, 1.0, -1.0) for ky in range(8) for kx in range(8)]
    c = _control(0, 0, vecs, error=0.9999)
    _gx, _gy, disp, report = solve.solve_dctpls([c], max_disp=256)
    assert report["input"] == "vectors"
    assert report["n_valid"] == 64
    assert np.allclose(np.asarray(disp), [1.0, -1.0], atol=1e-6)


def test_the_range_gate_still_rejects_a_vector_beyond_max_disp():
    vecs = [_vec(kx, ky, 1.0, 0.0) for ky in range(6) for kx in range(6)]
    vecs.append(_vec(6, 6, 300.0, 0.0))
    _gx, _gy, disp, report = solve.solve_dctpls([_control(0, 0, vecs)], max_disp=256)
    assert report["n_rejected_disp"] == 1 and report["n_valid"] == 36
    assert abs(np.asarray(disp)[6, 6, 0] - 1.0) < 0.2


def _smooth(x, y):
    return (
        3.0 + 0.002 * (y - 5000) + 2.0 * np.sin(2 * np.pi * y / 7000),
        -2.0 - 0.002 * (x - 5000) + 2.0 * np.cos(2 * np.pi * x / 6000),
    )


def test_sigma_is_calibrated_per_peak_ratio_bin():
    """Noise injected as 0.05 + 0.5/PR px: the bins' sigma must fall with PR and cover ~68 %."""
    controls = _field_controls(80, _smooth, lambda pr: 0.05 + 0.5 / pr)
    _gx, _gy, _disp, report = solve.solve_dctpls(controls, max_disp=256)
    bins = report["sigma_calibration"]
    assert bins is not None and len(bins) == 8, report["sigma_calibration_skipped"]
    sig = [b["sigma_px"] for b in bins]
    assert sig[0] > sig[-1] * 1.8, sig
    assert all(b["n"] >= 30 for b in bins)
    # the true sd at each bin's centre bounds the calibrated sigma from below (the held-out
    # residual adds prediction error), and not by much on a smooth field
    for b in bins:
        true = 0.05 + 0.5 / ((b["pr_lo"] + b["pr_hi"]) / 2)
        assert true * 0.8 < b["sigma_px"] < true * 1.6, (b, true)
    assert 0.55 < report["coverage_1sigma"] < 0.8


def test_too_few_vectors_skip_the_calibration_and_say_why():
    vecs = [_vec(kx, ky, 1.0, 0.5) for ky in range(7) for kx in range(7)]
    _gx, _gy, _disp, report = solve.solve_dctpls([_control(0, 0, vecs)], max_disp=256)
    assert report["sigma_calibration"] is None
    assert "60" in report["sigma_calibration_skipped"]
    assert report["coverage_1sigma"] is None


def test_the_field_is_reindexed_to_the_frame_the_stitch_evaluates_it_in():
    """F(g) = D(g + F(g)): the stitch reads the mesh at the MOVING point (warp._invert)."""
    n = 40
    gx = gy = S * (np.arange(n) + 1.0)
    GX, GY = np.meshgrid(gx, gy)
    th = np.radians(0.3)
    D = np.stack(
        [
            100.0 + (np.cos(th) - 1) * GX - np.sin(th) * GY,
            -50.0 + np.sin(th) * GX + (np.cos(th) - 1) * GY,
        ],
        axis=-1,
    )
    F = solve._reindex_to_moving_frame(gx, gy, D)
    mesh_d = MeshField(gx, gy, D)
    inner = (slice(5, -5), slice(5, -5))
    g = np.stack([GX[inner].ravel(), GY[inner].ravel()], axis=1)
    back = mesh_d.displacement(g + F[inner].reshape(-1, 2))
    assert np.abs(back - F[inner].reshape(-1, 2)).max() < 1e-3
    # and it matters: at a 112 px offset with 0.3 deg of rotation F and D differ by ~0.6 px
    assert np.abs(F[inner] - D[inner]).max() > 0.3


@pytest.mark.parametrize("case", ["base", "large100"])
def test_end_to_end_4096_through_the_stitch_semantics(case):
    """Four 2048 tiles -> vector grid -> solve_dctpls -> the mesh as warp._invert reads it.

    ``large100`` adds a (+100, -50) px offset on top of a 0.15 deg rotation and mid-wave
    deformation: two passes must hold the median under 0.4 px; a single pass (pass 1 off)
    must fail it -- research/stare-sota-review-2026-09-27.md Part C §6.
    """
    N = 4096
    off = 100.0 if case == "large100" else 0.0
    th, c = np.radians(0.15), N / 2

    def ufield(X, Y):
        ux = (np.cos(th) - 1) * (X - c) - np.sin(th) * (Y - c) + 3.0 + off
        ux = (
            ux + 3.0 * np.sin(2 * np.pi * Y / 3000) + 1.5 * np.sin(2 * np.pi * X / 1500)
        )
        uy = np.sin(th) * (X - c) + (np.cos(th) - 1) * (Y - c) - 2.0 - off / 2
        uy = (
            uy + 3.0 * np.cos(2 * np.pi * X / 3000) + 1.5 * np.cos(2 * np.pi * Y / 1500)
        )
        return ux, uy

    rng = np.random.default_rng(7)
    mask = tissue(N, rng)
    ref, mov = make_pair(N, ufield, seed=7, mask=mask)

    def run(two_pass):
        controls = []
        for t in tile_grid(N, N, 2048, 256):
            bx0, by0, bx1, by1 = vg.read_box(t.core, S, (N, N))
            v = vg.estimate_tile_vectors(
                ref[by0:by1, bx0:bx1],
                mov[by0:by1, bx0:bx1],
                (bx0, by0),
                t.core,
                S,
                two_pass=two_pass,
            )
            controls.append(_control(t.ix, t.iy, v["vectors"], v["rejected"]))
        gx, gy, disp, report = solve.solve_dctpls(controls, max_disp=256)
        mesh = MeshField(np.asarray(gx), np.asarray(gy), np.asarray(disp))
        ev = np.mgrid[256 : N - 256 : 32, 256 : N - 256 : 32].reshape(2, -1).T
        ev = ev[mask[ev[:, 0], ev[:, 1]]][:, ::-1].astype(float)  # (x, y) on tissue
        v = ev.copy()
        for _ in range(5):  # warp._invert: v = u - F(v)
            v = ev - mesh.displacement(v)
        err = np.hypot(*(v - moving_point(ufield, ev)).T)
        return float(np.median(err)), report

    med2, report = run(True)
    assert med2 < 0.4, (case, med2)
    assert report["input"] == "vectors" and report["sigma_calibration"] is not None
    # a normal field is nowhere near a fold: the certificate holds with room to spare
    assert report["lipschitz"] < 0.5 and report["fold_certificate_ok"], report[
        "lipschitz"
    ]
    assert report["min_det_jacobian"] > 0.5
    if case == "large100":
        med1, _ = run(False)
        assert med1 > 1.0, ("a single pass should not survive +100 px", med1)


def test_a_folding_control_set_fails_the_fold_certificate():
    """A reference-frame expansion steeper than 1 (dD/dx > 1) folds the moving frame.

    ``dx = 250 sin(2 pi x / 1024)`` peaks at a slope of 1.53: ``x -> x - D(x)`` is not
    monotone there, so the mesh the stitch reads has ``det(I + J) < 0`` somewhere. The
    certificate must say so rather than ship it silently.
    """
    n = 24
    X = S * (np.arange(n) + 1.0)
    vecs = [
        _vec(kx, ky, 250.0 * np.sin(2 * np.pi * X[kx] / 1024), 0.0)
        for ky in range(n)
        for kx in range(n)
    ]
    _gx, _gy, _disp, report = solve.solve_dctpls([_control(0, 0, vecs)], max_disp=None)
    assert report["min_det_jacobian"] < 0, report["min_det_jacobian"]
    assert report["fold_certificate_ok"] is False
    assert report["lipschitz"] >= 1.0


def test_a_gentle_synthetic_field_passes_the_fold_certificate():
    n = 24
    X = S * (np.arange(n) + 1.0)
    vecs = [
        _vec(kx, ky, 4.0 * np.sin(2 * np.pi * X[kx] / 1500), 2.0 * np.cos(X[ky] / 900))
        for ky in range(n)
        for kx in range(n)
    ]
    _gx, _gy, _disp, report = solve.solve_dctpls([_control(0, 0, vecs)], max_disp=256)
    assert report["lipschitz"] < 0.5 and report["fold_certificate_ok"] is True
    assert report["min_det_jacobian"] > 0


# ── Phase 5b: coverage on disjoint folds, re-index residual ──────────────────────
def test_sigma_coverage_is_scored_on_folds_the_calibration_never_saw():
    """Residuals in the scoring folds are 3x those in the calibration folds: a coverage
    computed on the calibration residuals would still read ~0.68; scored on the disjoint
    folds it must collapse, and the RMS ratio must read ~3."""
    rng = np.random.default_rng(0)
    n = 60
    W = np.ones((n, n))
    folds = solve._cv_folds(W)
    in_cal = np.isin(folds, solve.CAL_FOLDS)
    E = rng.normal(0, 0.3, (n, n, 2))
    E[~in_cal] *= 3.0
    PR = rng.uniform(1.2, 8.0, (n, n))
    sigma, bins, scores = solve._calibrate_sigma(PR, W > 0, E, folds)
    assert sigma is not None
    assert set(scores["calibration_folds"]).isdisjoint(scores["scoring_folds"])
    assert set(scores["calibration_folds"]) | set(scores["scoring_folds"]) == set(
        range(solve.CV_FOLDS)
    )
    assert scores["n_scored"] == int((~in_cal).sum())
    assert scores["coverage_1sigma"] < 0.35, scores
    assert scores["rms_error_over_rms_sigma"] == pytest.approx(3.0, rel=0.1)
    # every bin is built from calibration-fold vectors only
    assert sum(b["n"] for b in bins) == int(in_cal.sum())


def test_the_solve_reports_held_out_coverage_and_rms_ratio():
    controls = _field_controls(80, _smooth, lambda pr: 0.05 + 0.5 / pr)
    _gx, _gy, _disp, report = solve.solve_dctpls(controls, max_disp=256)
    assert report["coverage_scoring_folds"] == [1, 3]
    assert report["coverage_calibration_folds"] == [0, 2, 4]
    assert report["coverage_n_scored"] > 500
    assert 0.8 < report["rms_error_over_rms_sigma"] < 1.25, report
    assert report["smoothing_selection"] == solve.CV_LABELS[report["cv_buffer"]]


def test_the_reindex_fixed_point_residual_is_reported_and_small():
    n = 40
    gx = gy = S * (np.arange(n) + 1.0)
    GX, GY = np.meshgrid(gx, gy)
    th = np.radians(0.3)
    D = np.stack(
        [
            100.0 + (np.cos(th) - 1) * GX - np.sin(th) * GY,
            -50.0 + np.sin(th) * GX + (np.cos(th) - 1) * GY,
        ],
        axis=-1,
    )
    info = {}
    F = solve._reindex_to_moving_frame(gx, gy, D, interp="cubic", info=info)
    mesh_d = MeshField(gx, gy, D, interp="cubic")
    G = np.stack([GX.ravel(), GY.ravel()], axis=1)
    actual = np.abs(mesh_d.displacement(G + F.reshape(-1, 2)) - F.reshape(-1, 2)).max()
    assert info["reindex_residual_px"] == pytest.approx(actual, abs=1e-12)
    assert info["reindex_residual_px"] < solve.REINDEX_TOL_PX
    assert 1 <= info["reindex_iterations"] <= solve.REINDEX_MAX_ITERATIONS
