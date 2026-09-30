"""stare.solve's ``dctpls`` core: robust affine + robust DCT-PLS, no dead zone.

Why it exists (research/stare-optimal-design-2026-09-27.md §0, §3): STARE v1's ``robust``
solver's TRE gate hard-zeroed sub-gate vectors, its first-order Tikhonov penalty shrank the
affine residual M0 leaves, and its ``1 - error`` weights had no fixed scale. These tests pin
the behaviours that replaced each of those, on ``_dctpls_core`` -- the lattice solver that
``solve_dctpls`` feeds with REG_TILE's vector lattice (test_solve_vectors.py covers that
feed: the range gate, sigma calibration, the re-index). Synthetic lattices are laid out
here directly, one node per ``(ix, iy)``; a node is invalid (weight 0) when it is flagged
``valid=False``, non-finite, or at/beyond ``max_disp``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from stare import solve
from stare.manifest import slide_entry
from stare.mesh_field import MeshField

TILE = 1024.0


def _controls(n, fn, tile=TILE, noise=0.0, seed=0, ny=None):
    """An ``n x ny`` lattice of nodes sampling ``fn(cx, cy) -> (dx, dy)`` at cell centres."""
    rng = np.random.default_rng(seed)
    out = []
    for iy in range(ny or n):
        for ix in range(n):
            cx, cy = ix * tile + tile / 2, iy * tile + tile / 2
            dx, dy = fn(cx, cy)
            dx = float(dx) + rng.normal(0, noise)
            dy = float(dy) + rng.normal(0, noise)
            out.append(
                {
                    "ix": ix,
                    "iy": iy,
                    "cx": cx,
                    "cy": cy,
                    "dx": dx,
                    "dy": dy,
                    "valid": True,
                }
            )
    return out


def _truth(controls, fn):
    nx = max(c["ix"] for c in controls) + 1
    ny = max(c["iy"] for c in controls) + 1
    t = np.zeros((ny, nx, 2))
    for c in controls:
        t[c["iy"], c["ix"]] = fn(c["cx"], c["cy"])
    return t


def _rotation_about_centre(n, deg=0.15, t=(3.0, -2.0), tile=TILE):
    """A pure affine displacement: rotation about the slide centre plus a translation."""
    c = n * tile / 2
    th = np.radians(deg)

    def fn(x, y):
        return (
            (np.cos(th) - 1) * (x - c) - np.sin(th) * (y - c) + t[0],
            np.sin(th) * (x - c) + (np.cos(th) - 1) * (y - c) + t[1],
        )

    return fn


def _lay_out(controls, max_disp):
    nx = max(c["ix"] for c in controls) + 1
    ny = max(c["iy"] for c in controls) + 1
    gx, gy = np.zeros(nx), np.zeros(ny)
    Y, W0 = np.zeros((ny, nx, 2)), np.zeros((ny, nx))
    n_disp = 0
    for c in controls:
        gx[c["ix"]], gy[c["iy"]] = c["cx"], c["cy"]
        dx, dy = float(c["dx"]), float(c["dy"])
        if not (c["valid"] and np.isfinite(dx) and np.isfinite(dy)):
            continue
        if max_disp is not None and np.hypot(dx, dy) >= max_disp:
            n_disp += 1
            continue
        Y[c["iy"], c["ix"]] = (dx, dy)
        W0[c["iy"], c["ix"]] = 1.0
    return gx, gy, Y, W0, n_disp


def _solve(controls, max_disp=256):
    """``_dctpls_core`` on the laid-out lattice, with the jacobian the solve reports."""
    gx, gy, Y, W0, n_disp = _lay_out(controls, max_disp)
    field, info = solve._dctpls_core(Y, W0, gx, gy)
    # the Jacobian of the bilinear interpolant (the tests here build bilinear meshes)
    jac = solve.jacobian_report(gx, gy, field)
    report = {
        **info,
        "n_rejected_disp": n_disp,
        "lipschitz": jac["max_operator_norm"],
        "min_det_jacobian": jac["min_jacobian_det"],
        "fold_certificate_ok": bool(
            jac["max_operator_norm"] < solve.FOLD_CERTIFICATE_LIPSCHITZ
        ),
        "measured": (W0 > 0).astype(int).tolist(),
    }
    return list(gx), list(gy), field, report


# ── the affine is recovered, not shrunk ─────────────────────────────────────
@pytest.mark.parametrize("noise", [0.0, 0.05])
def test_a_pure_affine_field_is_recovered(noise):
    """With noise the selector smooths hard, and DCT-PLS alone (null space: a constant)
    would flatten the rotation -- the affine must come out first."""
    fn = _rotation_about_centre(10)
    controls = _controls(10, fn, noise=noise)
    _, _, disp, report = _solve(controls)
    assert np.abs(disp - _truth(controls, fn)).max() < 0.05
    assert report["affine"]["rotation_deg"] == pytest.approx(0.15, abs=2e-3)
    if noise:
        return
    # the translation is reported at the pixel origin: t + (R - I)(0 - c)
    c = 10 * TILE / 2
    th = np.radians(0.15)
    assert report["affine"]["tx"] == pytest.approx(
        3.0 - (np.cos(th) - 1) * c + np.sin(th) * c, abs=1e-3
    )


@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(
            lambda x, y: (1.2 * x / (10 * TILE), 0.0 * x), id="ramp-0-to-1.2px"
        ),
        pytest.param(lambda x, y: (0.9 + 0.0 * x, 0.0 * x), id="uniform-0.9px"),
    ],
)
def test_sub_pixel_fields_are_kept_not_zeroed(fn):
    """Every one of these displacements is below the 1 px TRE gate STARE v1 turned into
    [0, 0] (a dead zone); dctpls keeps them."""
    controls = _controls(10, fn, noise=0.02)
    _, _, disp, _ = _solve(controls)
    assert np.abs(disp - _truth(controls, fn)).max() < 0.1


# ── local structure survives; a lone wrong vector does not ───────────────────
def test_a_three_cell_bump_is_preserved_at_its_centre():
    """A smooth 3 px bump about three cells across (Gaussian, sigma = 1 cell, on a node).

    A SHARP 3x3 top-hat of 3 px is, to any smoothness prior with robust weights,
    indistinguishable from a coherent cluster of wrong vectors -- the case Garcia (2011)
    designs the bisquare to reject -- so the physically meaningful bump is a smooth one.

    Pinned at plain block CV (no buffer ring), which the adaptive rule picks on this white
    noise. With the true h-block ring a sigma = 1-cell bump is hidden from every training fit
    and smoothed away (measured 0/3 seeds here) -- one of the reasons the ring is kept only
    for residuals that correlate.
    """
    centre = (4 * TILE + TILE / 2, 4 * TILE + TILE / 2)

    def fn(x, y):
        return 3.0 * np.exp(
            -((x - centre[0]) ** 2 + (y - centre[1]) ** 2) / (2 * TILE**2)
        ), 0.0 * x

    for seed in range(3):
        controls = _controls(10, fn, noise=0.05, seed=seed)
        _, _, disp, _ = _solve(controls)
        assert abs(disp[4, 4, 0] - 3.0) < 0.5, seed


def test_a_single_wildly_wrong_cell_is_downweighted():
    fn = _rotation_about_centre(10)
    controls = _controls(10, fn, noise=0.05)
    bad = next(c for c in controls if c["ix"] == 5 and c["iy"] == 4)
    bad["dx"] += 50.0  # inside the range gate: no gate can see it
    _, _, disp, report = _solve(controls)
    assert np.hypot(*(disp[4, 5] - _truth(controls, fn)[4, 5])) < 1.0
    assert report["n_downweighted"] >= 1


# ── validity ─────────────────────────────────────────────────────────────────
def test_an_invalid_node_is_filled_from_the_field_not_zeroed():
    fn = _rotation_about_centre(8)
    controls = _controls(8, fn)
    hole = next(c for c in controls if c["ix"] == 3 and c["iy"] == 3)
    hole["valid"] = False
    hole["dx"], hole["dy"] = 40.0, 40.0  # whatever a rejected window returned
    _, _, disp, report = _solve(controls)
    assert report["n_valid"] == 63
    assert report["measured"][3][3] == 0 and report["measured"][3][4] == 1
    assert np.hypot(*(disp[3, 3] - _truth(controls, fn)[3, 3])) < 0.05


def test_all_invalid_controls_give_no_mesh():
    controls = _controls(4, lambda x, y: (2.0, 1.0))
    for c in controls:
        c["valid"] = False
    gx, gy, disp, report = _solve(controls)
    assert report["smoothing_selection"] == "none" and report["n_valid"] == 0
    assert slide_entry(np.eye(3), gx, gy, disp)["mesh"] is None


@pytest.mark.parametrize(
    ("n_valid", "selection"),
    [(1, "translation_only"), (2, "translation_only"), (4, "affine_only"), (9, "gcv")],
)
def test_few_valid_cells_degrade_to_a_simpler_model(n_valid, selection):
    fn = lambda x, y: (1.5 + 0.0 * x, -0.5 + 0.0 * y)  # noqa: E731
    controls = _controls(5, fn)
    for c in controls[n_valid:]:
        c["valid"] = False
    _, _, disp, report = _solve(controls)
    assert report["smoothing_selection"] == selection
    # no hole stays at zero: every node carries the model
    np.testing.assert_allclose(disp, _truth(controls, fn), atol=1e-6)


def test_block_cv_selects_s_on_a_large_enough_grid():
    controls = _controls(8, _rotation_about_centre(8), noise=0.1)
    report = _solve(controls)[3]
    # the label says which CV ran: white noise -> no ring, plain block CV
    assert report["cv_buffer"] == 0 and report["smoothing_selection"] == "block_cv"
    assert report["residual_lag1_rho"] < solve.CV_BUFFER_RHO
    assert report["holdout_rmse_px"] is not None and 0 < report["holdout_rmse_px"] < 1.0


# ── report ───────────────────────────────────────────────────────────────────
def test_report_has_every_key_and_is_json_serialisable():
    controls = _controls(6, _rotation_about_centre(6), noise=0.05)
    controls[0]["dx"] = 999.0  # out of range
    report = _solve(controls)[3]
    for k in (
        "n_rejected_disp",
        "n_valid",
        "n_downweighted",
        "affine",
        "smoothing_s",
        "smoothing_selection",
        "holdout_rmse_px",
        "residual_lag1_rho",
        "cv_buffer",
        "lipschitz",
        "min_det_jacobian",
        "fold_certificate_ok",
        "measured",
    ):
        assert k in report, k
    assert set(report["affine"]) == {
        "tx",
        "ty",
        "rotation_deg",
        "scale_x",
        "scale_y",
        "shear",
    }
    assert report["n_rejected_disp"] == 1 and report["n_valid"] == 35
    assert report["fold_certificate_ok"] is True and report["min_det_jacobian"] > 0
    json.dumps(report)


def test_the_field_is_never_rescaled_and_a_fold_is_reported():
    """A field steeper than the certificate allows is reported, not scaled down."""
    controls = _controls(6, lambda x, y: (0.8 * x, 0.0 * y), tile=10.0)
    _, _, disp, report = _solve(controls, max_disp=None)
    np.testing.assert_allclose(
        disp[..., 0], 0.8 * np.asarray([[c["cx"] for c in controls[:6]]] * 6), atol=1e-6
    )
    assert report["lipschitz"] == pytest.approx(0.8, abs=1e-6)
    assert report["fold_certificate_ok"] is False


# ── the 16-tile acceptance, analytic and fast ────────────────────────────────
def test_on_a_coarse_4x4_grid_dctpls_ties_raw_vectors():
    """The research §0 comparison without the 8192^2 image: an analytic pull-back field
    (0.15 deg rotation about the centre + translation + a 4 px sinusoid at 6000 px +
    a 4 px Gaussian bump) sampled at 4x4 tile centres with 0.2 px noise. Scored on a dense
    grid through the same bilinear MeshField the stitch uses."""
    n, tile = 4, 2048.0
    c = n * tile / 2
    th = np.radians(0.15)

    def fn(x, y):
        b = 4 * np.exp(-((x - 5500) ** 2 + (y - 2500) ** 2) / (2 * 300**2))
        ux = (
            (np.cos(th) - 1) * (x - c)
            - np.sin(th) * (y - c)
            + 3
            + 4 * np.sin(2 * np.pi * y / 6000)
            + b
        )
        uy = (
            np.sin(th) * (x - c)
            + (np.cos(th) - 1) * (y - c)
            - 2
            + 4 * np.cos(2 * np.pi * x / 6000)
            - b
        )
        return ux, uy

    controls = _controls(n, fn, tile=tile, noise=0.2, seed=0)
    ey, ex = np.mgrid[256 : n * tile - 256 : 64, 256 : n * tile - 256 : 64].astype(
        float
    )
    tx, ty = fn(ex, ey)

    def median_error(gx, gy, disp):
        mf = MeshField(np.asarray(gx), np.asarray(gy), np.asarray(disp))
        d = mf.displacement(np.column_stack([ex.ravel(), ey.ravel()])).reshape(
            ex.shape + (2,)
        )
        return float(np.median(np.hypot(d[..., 0] - tx, d[..., 1] - ty)))

    raw = np.zeros((n, n, 2))
    for ct in controls:
        raw[ct["iy"], ct["ix"]] = (ct["dx"], ct["dy"])
    gx = [i * tile + tile / 2 for i in range(n)]
    e_raw = median_error(gx, gx, raw)
    e_dct = median_error(*_solve(controls)[:3])
    # On a 4x4 grid the residual after the affine is mostly real, unresolved deformation,
    # so the best any smoother can do is interpolate: GCV picks s ~ 1e-4 and dctpls lands
    # on raw bilinear to within a fraction of a percent (either side, with the noise). The
    # 1 % allowance is that tie (STARE v1's robust solver was 7x worse, not 1 %).
    assert e_dct <= 1.01 * e_raw, (e_dct, e_raw)


# ── Phase 5b audit fixes (research/stare-step-support-2026-09-27.md, SOLVE S1-S6) ────
def test_the_robust_scale_of_residual_norms_is_the_rayleigh_one():
    """|r| of an isotropic Gaussian residual is Rayleigh: median(|r|) / 1.1774 is sigma."""
    rng = np.random.default_rng(0)
    r = np.linalg.norm(rng.normal(0, 0.7, (20000, 2)), axis=1)
    assert solve._norm_scale(r) == pytest.approx(0.7, rel=0.03)
    assert solve.HUBER_C == pytest.approx(2.4477, abs=1e-4)
    assert solve.BISQUARE_C == pytest.approx(5.0569, abs=1e-4)


def test_huber_downweights_few_clean_gaussian_residuals():
    """On clean isotropic Gaussian residuals the Huber IRLS down-weights ~5 %, not ~67 %.

    1.4826 x MAD of Rayleigh norms is ~0.66 sigma, and Huber's 1.345 on that scale cut at
    0.89 sigma: two thirds of perfectly clean vectors were treated as outliers.
    """
    n = 40
    g = 128.0 * (np.arange(n) + 1)
    GX, GY = np.meshgrid(g, g)
    rng = np.random.default_rng(1)
    truth = np.stack([1.0 + 1e-3 * GX, -2.0 + 5e-4 * GY], axis=-1)
    Y = truth + rng.normal(0, 0.5, (n, n, 2))
    _field, _coef, wr = solve._robust_affine(Y, np.ones((n, n)), g, g)
    frac = float(np.mean(wr < 1.0))
    assert frac <= 0.10, frac


def test_bisquare_rejects_almost_no_clean_gaussian_residual():
    rng = np.random.default_rng(2)
    R = rng.normal(0, 0.5, (40, 40, 2))
    w = solve._bisquare(R, np.zeros_like(R), np.ones((40, 40)), s=1e6)
    assert float(np.mean(w < 0.1)) < 0.005
    # a real outlier among them is still rejected
    R[5, 5] = (20.0, 0.0)
    w = solve._bisquare(R, np.zeros_like(R), np.ones((40, 40)), s=1e6)
    assert w[5, 5] == 0.0


def test_with_the_ring_an_hblock_scored_cell_is_never_next_to_a_training_cell():
    """The buffer ring: every scored cell is >= 2 lattice steps from every training cell."""
    from scipy.ndimage import binary_dilation

    W = np.ones((20, 23))
    folds = solve._cv_folds(W)
    for f in range(solve.CV_FOLDS):
        scored, excluded = solve._cv_split(folds, f, W > 0, buffer=1)
        assert scored.any()
        training = ~excluded
        near_scored = binary_dilation(scored, structure=np.ones((3, 3), bool))
        assert not (near_scored & training).any(), f
        # the ring is excluded but not scored
        assert (excluded & ~scored).any()
        # and without it (the default) the excluded cells are exactly the scored patch
        plain, excl0 = solve._cv_split(folds, f, W > 0, buffer=0)
        assert (plain == excl0).all()


def _correlated_noise_lattice(n=48, corr_cells=0.7, sd=0.3, seed=0):
    """A smooth field plus noise correlated over ~1 cell, as 50 %-overlap windows give."""
    from scipy.ndimage import gaussian_filter

    g = 128.0 * (np.arange(n) + 1)
    GX, GY = np.meshgrid(g, g)
    truth = np.stack(
        [2 * np.sin(2 * np.pi * GY / 3000), 2 * np.cos(2 * np.pi * GX / 3000)], axis=-1
    )
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (n, n, 2))
    noise = np.stack(
        [gaussian_filter(noise[..., k], corr_cells) for k in range(2)], axis=-1
    )
    noise *= sd / noise.std()
    return g, truth, truth + noise


@pytest.mark.parametrize("seed", [0, 1])
def test_hblock_cv_with_a_buffer_ring_does_not_undersmooth_correlated_noise(seed):
    """Burman, Chow & Nolan (1994): with correlated errors, CV that trains on a held-out
    cell's neighbours predicts the error along with the signal and picks s too small
    (interpolation). The 1-cell ring removes the leak: larger s, smaller true error."""
    g, truth, Y = _correlated_noise_lattice(seed=seed)
    n = g.size
    W = np.ones((n, n))
    Lam = solve._dct_eigenvalues(n, n, g, g)
    s_plain, _ = solve._block_cv_select(Y, W, Lam, buffer=0)
    s_ring, _ = solve._block_cv_select(Y, W, Lam, buffer=1)
    assert s_ring >= s_plain
    assert s_ring > 100 * s_plain, (s_plain, s_ring)

    def err(s):
        return float(np.sqrt(np.mean((solve._pls_fit(Y, W, Lam, s) - truth) ** 2)))

    assert err(s_ring) < 0.8 * err(s_plain), (err(s_ring), err(s_plain))


def test_the_sign_lag1_statistic_reads_the_noise_it_is_given():
    """The raw statistic on noise with no fit: iid ~0, Gaussian-filtered ~ its lag-1, and
    one huge residual moves it by one sign at most."""
    white = np.random.default_rng(0).normal(0, 1, (48, 48, 2))
    assert abs(solve._sign_lag1(white, np.ones((48, 48)))) < 0.05
    g, truth, Y = _correlated_noise_lattice(seed=0)
    rho = solve._sign_lag1(Y - truth, np.ones((48, 48)))
    assert 0.5 < rho < 0.7, rho  # sigma = 0.7 cells: exp(-1 / (4 * 0.49)) = 0.60
    E = white.copy()
    E[3, 3] = 1e6
    assert abs(solve._sign_lag1(E, np.ones((48, 48)))) < 0.05


@pytest.mark.parametrize("seed", [0, 1])
def test_correlated_residuals_turn_the_hblock_buffer_on(seed):
    """Noise correlated at lag 1 ~0.6 (the 50 %-overlap regime): the rule keeps the ring,
    and the field is closer to the truth than plain block CV's undersmoothed one."""
    g, truth, Y = _correlated_noise_lattice(seed=seed)
    n = g.size
    field, info = solve._dctpls_core(Y, np.ones((n, n)), g, g)
    assert info["cv_buffer"] == 1 and info["smoothing_selection"] == "hblock_cv", info
    assert info["residual_lag1_rho"] > solve.CV_BUFFER_RHO
    Lam = solve._dct_eigenvalues(n, n, g, g)
    s_plain, _ = solve._block_cv_select(Y, np.ones((n, n)), Lam, buffer=0)
    plain, _ = solve._robust_pls(Y, np.ones((n, n)), Lam, s_plain)
    rms = lambda F: float(np.sqrt(np.mean((F - truth) ** 2)))  # noqa: E731
    assert info["smoothing_s"] > 100 * s_plain, (info["smoothing_s"], s_plain)
    assert rms(field) < 0.8 * rms(plain), (rms(field), rms(plain))


@pytest.mark.parametrize("seed", [0, 1])
def test_white_residuals_keep_the_buffer_off(seed):
    """Same field, iid noise: the ring would only cost accuracy, and the rule leaves it off."""
    g = 128.0 * (np.arange(48) + 1)
    GX, GY = np.meshgrid(g, g)
    truth = np.stack(
        [2 * np.sin(2 * np.pi * GY / 3000), 2 * np.cos(2 * np.pi * GX / 3000)], axis=-1
    )
    Y = truth + np.random.default_rng(seed).normal(0, 0.3, truth.shape)
    _field, info = solve._dctpls_core(Y, np.ones((48, 48)), g, g)
    assert info["cv_buffer"] == 0 and info["smoothing_selection"] == "block_cv", info
    assert info["residual_lag1_rho"] < solve.CV_BUFFER_RHO


def test_a_ring_that_flattens_lattice_scale_signal_is_refused():
    """A near-noiseless wave of ~6 nodes period on a 15x15 lattice: the interpolating fit's
    residuals correlate (signal, not noise), and the ring would hide a whole period and
    choose the flattest field. The ring is refused and the wave kept."""
    n = 15
    g = 16.0 * (np.arange(n) + 1)
    GX, GY = np.meshgrid(g, g)
    truth = np.stack(
        [3 * np.sin(2 * np.pi * GY / 90), 3 * np.cos(2 * np.pi * GX / 90)], axis=-1
    )
    Y = truth + np.random.default_rng(0).normal(0, 0.005, truth.shape)
    field, info = solve._dctpls_core(Y, np.ones((n, n)), g, g)
    assert info["residual_lag1_rho"] > solve.CV_BUFFER_RHO  # the confound is real
    assert info["cv_buffer"] == 0 and info["smoothing_selection"] == "block_cv", info
    assert np.abs(field - truth).max() < 0.1


def test_the_calibrated_resolve_reuses_s_instead_of_rescanning(monkeypatch):
    """'s chosen once' is true: the 1/sigma^2 re-solve calls no selector (and on white noise
    the first solve calls it once: plain block CV, whose residuals keep the ring off)."""
    calls = []
    real = solve._block_cv_select

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(solve, "_block_cv_select", counting)
    fn = _rotation_about_centre(12)
    controls = _controls(12, fn, noise=0.1)
    gx, gy, Y, W0, _ = _lay_out(controls, 256)
    _f, info = solve._dctpls_core(Y, W0, gx, gy)
    assert calls == [1] and info["smoothing_selection"] == "block_cv"
    W1 = np.where(W0 > 0, 0.25, 0.0)  # uniform weights at a different scale
    _f2, info2 = solve._dctpls_core(Y, W1, gx, gy, s_fixed=info["smoothing_s"])
    assert calls == [1] and info2["smoothing_selection"] == "fixed"
    # s is in mean-weight units, so a uniform rescale of W leaves the fit unchanged
    np.testing.assert_allclose(_f2, _f, atol=1e-6)
    assert info2["smoothing_s"] == pytest.approx(info["smoothing_s"])


def test_the_fold_certificate_reads_the_cubic_interpolant_not_the_nodes():
    """An alternating node sequence: node central differences see slope 0 inside and
    2a/h at the edge, but the interpolating cubic B-spline through it swings at 3a/h
    between nodes. The certificate must see the interpolant's slope."""
    h, a, n = 100.0, 20.0, 12
    gx = h * np.arange(n)
    gy = h * np.arange(6)
    disp = np.zeros((gy.size, n, 2))
    disp[..., 0] = a * (-1.0) ** np.arange(n)[None, :]
    node = np.abs(np.gradient(disp[..., 0], gx, axis=1)).max()
    assert node <= 2 * a / h + 1e-9  # 0.4: what node differences certify
    r = solve.jacobian_report(gx, gy, disp, interp="cubic")
    assert r["max_operator_norm"] == pytest.approx(3 * a / h, rel=0.02), r
    assert r["max_operator_norm"] > solve.FOLD_CERTIFICATE_LIPSCHITZ
