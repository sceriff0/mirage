"""stare.solve's ``dctpls`` solver: robust affine + robust DCT-PLS, no dead zone.

Why it exists (research/stare-optimal-design-2026-09-27.md §0, §3): the ``robust``
solver's TRE gate hard-zeroes sub-gate vectors, its first-order Tikhonov penalty
shrinks the affine residual M0 leaves, and its ``1 - error`` weights have no fixed
scale. These tests pin the behaviours that replace each of those.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from stare import solve
from stare.manifest import slide_entry
from stare.mesh_field import MeshField

TILE = 1024.0


def _controls(n, fn, tile=TILE, noise=0.0, seed=0, error=0.05, ny=None):
    """An ``n x ny`` grid of controls sampling ``fn(cx, cy) -> (dx, dy)`` at tile centres."""
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
                    "tre": float(np.hypot(dx, dy)),
                    "error": error,
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


def _solve(controls, **kw):
    kw.setdefault("max_error", 0.99)
    kw.setdefault("max_disp", 256)
    gx, gy, disp, report = solve.solve_grid(controls, 1.0, solver="dctpls", **kw)
    return gx, gy, np.asarray(disp), report


# ── the default ──────────────────────────────────────────────────────────────
def test_dctpls_is_the_default_solver():
    controls = _controls(4, lambda x, y: (1.0, 0.5))
    assert solve.solve_grid(controls, 1.0, 0.99, 256)[3]["solver"] == "dctpls"
    assert "dctpls" in solve.SOLVERS


def test_gate_tre_is_ignored_so_there_is_no_dead_zone():
    controls = _controls(6, lambda x, y: (0.4, -0.3))  # every tile far below gate 5 px
    a = solve.solve_grid(controls, 0.0, 0.99, 256, solver="dctpls")
    b = solve.solve_grid(controls, 5.0, 0.99, 256, solver="dctpls")
    assert a[2] == b[2]
    np.testing.assert_allclose(
        np.asarray(a[2]), _truth(controls, lambda x, y: (0.4, -0.3)), atol=1e-6
    )


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
def test_sub_gate_fields_are_kept_not_zeroed(fn):
    """Every one of these displacements is below the default 1 px TRE gate that the
    other solvers turn into [0, 0]; dctpls keeps them."""
    controls = _controls(10, fn, noise=0.02)
    _, _, disp, _ = _solve(controls)
    assert np.abs(disp - _truth(controls, fn)).max() < 0.1
    legacy = np.asarray(solve.solve_grid(controls, 1.0, 0.99, 256, solver="legacy")[2])
    # ...which is the dead zone it replaces
    assert np.abs(legacy - _truth(controls, fn)).max() > 0.5


# ── local structure survives; a lone wrong vector does not ───────────────────
def test_a_three_cell_bump_is_preserved_at_its_centre():
    """A smooth 3 px bump about three cells across (Gaussian, sigma = 1 cell, on a node).

    A SHARP 3x3 top-hat of 3 px is, to any smoothness prior with robust weights,
    indistinguishable from a coherent cluster of wrong vectors -- the case Garcia (2011)
    designs the bisquare to reject -- so the physically meaningful bump is a smooth one.
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
    bad["dx"] += 50.0
    bad["error"] = 0.01  # more confident than its neighbours: no gate can see it
    _, _, disp, report = _solve(controls)
    assert np.hypot(*(disp[4, 5] - _truth(controls, fn)[4, 5])) < 1.0
    assert report["n_downweighted"] >= 1


# ── validity ─────────────────────────────────────────────────────────────────
def test_a_nan_error_is_rejected_and_its_node_filled_from_the_field():
    fn = _rotation_about_centre(8)
    controls = _controls(8, fn)
    hole = next(c for c in controls if c["ix"] == 3 and c["iy"] == 3)
    hole["error"] = float("nan")
    hole["dx"], hole["dy"] = 40.0, 40.0  # whatever an empty crop returned
    _, _, disp, report = _solve(controls)
    assert report["n_rejected_error"] == 1 and report["n_valid"] == 63
    assert report["measured"][3][3] == 0 and report["measured"][3][4] == 1
    assert np.hypot(*(disp[3, 3] - _truth(controls, fn)[3, 3])) < 0.05


def test_sigma_sets_the_prior_weight_and_error_does_not():
    fn = lambda x, y: (1.0 + 0.0 * x, 0.0 * x)  # noqa: E731
    controls = _controls(6, fn)
    for c in controls:
        c["error"] = 0.97  # an arbitrary-scale error must not change the fit
    base = _solve(controls)[2]
    for c in controls:
        c["error"] = 0.02
    np.testing.assert_allclose(_solve(controls)[2], base, atol=1e-9)
    controls[0]["sigma"] = 0.5
    controls[1]["sigma"] = 0.0  # not positive: default weight
    w = solve._lattice_from_controls(controls, 0.99, 256)[3]
    assert w[0, 0] == pytest.approx(4.0) and w[0, 1] == 1.0 and w[0, 2] == 1.0


def test_all_invalid_controls_give_no_mesh():
    controls = _controls(4, lambda x, y: (2.0, 1.0))
    for c in controls:
        c["error"] = float("nan")
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
        c["error"] = float("nan")
    _, _, disp, report = _solve(controls)
    assert report["smoothing_selection"] == selection
    # no hole stays at zero: every node carries the model
    np.testing.assert_allclose(disp, _truth(controls, fn), atol=1e-6)


def test_hblock_cv_selects_s_on_a_large_enough_grid():
    controls = _controls(8, _rotation_about_centre(8), noise=0.1)
    report = _solve(controls)[3]
    assert report["smoothing_selection"] == "hblock_cv"
    assert report["holdout_rmse_px"] is not None and 0 < report["holdout_rmse_px"] < 1.0


# ── report ───────────────────────────────────────────────────────────────────
def test_report_has_every_key_and_is_json_serialisable():
    controls = _controls(6, _rotation_about_centre(6), noise=0.05)
    controls[0]["dx"] = 999.0  # out of range
    report = _solve(controls)[3]
    for k in (
        "solver",
        "n_controls",
        "n_rejected_error",
        "n_rejected_disp",
        "n_valid",
        "n_downweighted",
        "affine",
        "smoothing_s",
        "smoothing_selection",
        "holdout_rmse_px",
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
    assert report["n_rejected_disp"] == 1 and report["n_controls"] == 36
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
def test_on_a_coarse_4x4_grid_dctpls_beats_raw_vectors_and_robust():
    """The research §0 comparison without the 8192^2 image: an analytic pull-back field
    (0.15 deg rotation about the centre + translation + a 4 px sinusoid at 6000 px +
    a 4 px Gaussian bump) sampled at 4x4 tile centres with 0.2 px noise and the
    whitened-correlation errors seen on the synthetic slide (~0.96). Scored on a dense
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

    controls = _controls(n, fn, tile=tile, noise=0.2, seed=0, error=0.96)
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
    e_dct = median_error(
        *solve.solve_grid(controls, 1.0, 0.99, 256, solver="dctpls")[:3]
    )
    e_rob = median_error(
        *solve.solve_grid(controls, 1.0, 0.99, 256, solver="robust")[:3]
    )
    # On a 4x4 grid the residual after the affine is mostly real, unresolved deformation,
    # so the best any smoother can do is interpolate: GCV picks s ~ 1e-4 and dctpls lands
    # on raw bilinear to within a fraction of a percent (either side, with the noise). The
    # 1 % allowance is that tie; the robust solver is 7x worse, not 1 %.
    assert e_dct <= 1.01 * e_raw, (e_dct, e_raw)
    assert e_raw <= e_rob, (e_raw, e_rob)
