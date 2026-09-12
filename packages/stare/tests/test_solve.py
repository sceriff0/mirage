"""stare.solve: the legacy solver is byte-identical to the pre-package stage; the
robust solver rejects inconsistent tiles, bridges dropped ones, smooths, and
keeps the field invertible."""

from __future__ import annotations

import ast
import types
from pathlib import Path
from statistics import median

import numpy as np
import pytest
from stare import solve

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_tiled_solve.py.txt"


def _legacy_oracle():
    """The three functions of bin/tiled_solve.py as it was before the package
    existed, executed from a verbatim copy, so the legacy path is pinned to the
    code that produced every manifest published before 2026-09-12."""
    tree = ast.parse(FIXTURE.read_text())
    keep = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_accept", "_median_filter_accepted", "_grid_from_controls")
    ]
    mod = types.ModuleType("legacy_tiled_solve")
    mod.np = np
    mod.median = median
    mod.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None, info=lambda *a, **k: None
    )
    exec(
        compile(ast.Module(body=keep, type_ignores=[]), str(FIXTURE), "exec"),
        mod.__dict__,
    )
    return mod


def _grid(nx, ny, tile=100.0, rng=None, base=(3.0, -2.0), noise=0.3, error=0.05):
    """A plausible control set: a smooth field plus noise, all confident."""
    rng = rng or np.random.default_rng(0)
    controls = []
    for iy in range(ny):
        for ix in range(nx):
            dx = base[0] + 0.02 * ix * tile / 100 + rng.normal(0, noise)
            dy = base[1] + 0.01 * iy * tile / 100 + rng.normal(0, noise)
            controls.append(
                {
                    "ix": ix,
                    "iy": iy,
                    "cx": ix * tile + tile / 2,
                    "cy": iy * tile + tile / 2,
                    "dx": float(dx),
                    "dy": float(dy),
                    "tre": float(np.hypot(dx, dy)),
                    "error": error,
                }
            )
    return controls


# ── legacy parity ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_legacy_is_byte_identical_to_the_pre_package_stage(seed):
    rng = np.random.default_rng(seed)
    controls = _grid(7, 5, rng=rng)
    # sprinkle the cases the gates see in the wild: a background tile, an
    # out-of-range peak, an empty crop (NaN), a below-gate tile, a legacy point
    controls[3]["error"] = 0.9995
    controls[8]["dx"], controls[8]["dy"] = 300.0, 1.0
    controls[12]["error"] = float("nan")
    controls[17]["tre"] = 0.2
    del controls[21]["error"]
    oracle = _legacy_oracle()
    want = oracle._grid_from_controls(controls, 1.0, max_error=0.99, max_disp=256)
    got = solve.solve_grid(controls, 1.0, max_error=0.99, max_disp=256, solver="legacy")
    assert got[0] == want[0] and got[1] == want[1]
    assert got[2] == want[2]  # exact list equality, no tolerance
    for c in controls:
        assert solve.accept(c, 0.99, 256) == oracle._accept(c, 0.99, 256)


def test_legacy_report_counts_what_the_gates_did(caplog):
    controls = _grid(4, 3)
    controls[0]["error"] = 0.9995
    controls[1]["dx"] = 999.0
    controls[2]["tre"] = 0.0
    del controls[3]["error"]
    with caplog.at_level("WARNING", logger="stare.solve"):
        _, _, _, report = solve.solve_grid(controls, 1.0, 0.99, 256, solver="legacy")
    assert report["solver"] == "legacy"
    assert report["n_rejected_error"] == 1 and report["n_rejected_disp"] == 1
    assert report["n_below_gate"] == 1 and report["n_legacy_unscored"] == 1
    assert "no 'error' key" in caplog.text


def test_unknown_solver_is_refused():
    with pytest.raises(ValueError, match="unknown solver"):
        solve.solve_grid(_grid(2, 2), 1.0, solver="tps")


# ── robust: neighbour consistency ─────────────────────────────────────────────
def test_normalized_median_test_flags_the_one_tile_that_disagrees():
    controls = _grid(6, 6, noise=0.03)
    # a confident tile that correlated against the wrong structure: 40 px off
    bad = next(c for c in controls if c["ix"] == 3 and c["iy"] == 3)
    bad["dx"] += 40.0
    bad["error"] = 0.03  # MORE confident than its neighbours -- the gate cannot see it
    _, _, disp, report = solve.solve_grid(controls, 0.0, 0.99, 256, solver="robust")
    assert report["n_rejected_inconsistent"] == 1
    assert report["n_infilled"] == 1
    _, _, raw, measured, *_ = solve._lay_out(controls, 0.0, 0.99, 256)
    assert solve.normalized_median_test(raw, measured)[3, 3]
    d = np.asarray(disp)
    # the cell now sits where its neighbours put it, not 40 px away
    assert abs(d[3, 3, 0] - np.median(d[2:5, 2:5, 0])) < 1.0
    assert abs(d[3, 3, 0] - (bad["dx"])) > 30.0


def test_normalized_median_test_keeps_a_consistent_field_intact():
    """Sub-pixel jitter well inside epsilon is not an inconsistency. (At jitter
    comparable to epsilon the test can flag a stray cell -- Westerweel & Scarano's
    known behaviour -- and the in-fill then restores it from its neighbours, so a
    false rejection costs nothing; see the test above.)"""
    controls = _grid(6, 6, noise=0.03)
    outlier = solve.normalized_median_test(
        *solve._lay_out(controls, 0.0, 0.99, 256)[2:4]
    )
    assert not outlier.any()


def test_cells_with_too_few_neighbours_are_not_judged():
    controls = _grid(2, 2, noise=0.0)
    controls[0]["dx"] += 50.0  # every cell has only 3 neighbours: exactly the minimum
    _, _, disp_raw, measured, *_ = solve._lay_out(controls, 0.0, 0.99, 256)
    outlier = solve.normalized_median_test(disp_raw, measured)
    assert outlier[0, 0]  # 3 neighbours suffice
    controls = _grid(2, 1, noise=0.0)
    _, _, disp_raw, measured, *_ = solve._lay_out(controls, 0.0, 0.99, 256)
    assert not solve.normalized_median_test(disp_raw, measured).any()


# ── robust: in-fill ───────────────────────────────────────────────────────────
def test_a_dropped_tile_is_bridged_from_its_neighbours_not_zeroed():
    controls = _grid(5, 5, noise=0.0, base=(10.0, 4.0))
    dropped = next(c for c in controls if c["ix"] == 2 and c["iy"] == 2)
    dropped["error"] = float("nan")  # an empty crop
    legacy = np.asarray(solve.solve_grid(controls, 0.0, 0.99, 256, solver="legacy")[2])
    robust = np.asarray(solve.solve_grid(controls, 0.0, 0.99, 256, solver="robust")[2])
    assert legacy[2, 2].tolist() == [0.0, 0.0]  # the step the legacy field carried
    assert abs(robust[2, 2, 0] - 10.0) < 1.0 and abs(robust[2, 2, 1] - 4.0) < 1.0


def test_infill_stops_at_its_radius_so_background_decays_to_zero():
    disp = np.zeros((9, 9, 2))
    trusted = np.zeros((9, 9), dtype=bool)
    disp[0, 0] = [8.0, 8.0]
    trusted[0, 0] = True
    filled, mask = solve.infill(disp, trusted, radius=2)
    assert mask[2, 2] and not mask[3, 3] and not mask[8, 8]
    assert filled[8, 8].tolist() == [0.0, 0.0]
    assert 0 < filled[2, 2, 0] <= 8.0


# ── robust: smoothing ─────────────────────────────────────────────────────────
def test_tikhonov_reproduces_a_constant_field_and_damps_noise():
    const = np.full((6, 6, 2), 5.0)
    w = np.ones((6, 6))
    np.testing.assert_allclose(
        solve.tikhonov_smooth(const, w, lam=2.0), const, atol=1e-9
    )
    rng = np.random.default_rng(1)
    noisy = const + rng.normal(0, 1.0, const.shape)
    smooth = solve.tikhonov_smooth(noisy, w, lam=2.0)
    assert np.std(smooth - const) < 0.5 * np.std(noisy - const)
    # a zero-weight cell follows its neighbours
    w[3, 3] = 0.0
    hole = noisy.copy()
    hole[3, 3] = [500.0, 500.0]
    assert abs(solve.tikhonov_smooth(hole, w, lam=2.0)[3, 3, 0] - 5.0) < 1.5


def test_tikhonov_degenerate_grids_and_zero_lambda_pass_through():
    d = np.array([[[1.0, 2.0]]])
    np.testing.assert_array_equal(solve.tikhonov_smooth(d, np.ones((1, 1)), 1.0), d)
    d = np.random.default_rng(0).normal(size=(3, 4, 2))
    np.testing.assert_array_equal(solve.tikhonov_smooth(d, np.ones((3, 4)), 0.0), d)
    np.testing.assert_array_equal(solve.tikhonov_smooth(d, np.zeros((3, 4)), 1.0), 0.0)


# ── robust: invertibility ─────────────────────────────────────────────────────
def test_jacobian_report_sees_a_fold_and_a_shear():
    gx, gy = [0.0, 100.0, 200.0], [0.0, 100.0, 200.0]
    flat = np.full((3, 3, 2), 7.0)
    r = solve.jacobian_report(gx, gy, flat)
    assert r["max_operator_norm"] == pytest.approx(0.0) and r[
        "min_jacobian_det"
    ] == pytest.approx(1.0)
    fold = np.zeros((3, 3, 2))
    fold[:, :, 0] = -1.5 * np.asarray(gx)[None, :]  # dux/dx = -1.5: det(I+J) < 0
    r = solve.jacobian_report(gx, gy, fold)
    assert r["min_jacobian_det"] < 0 and r["max_operator_norm"] == pytest.approx(1.5)


def test_robust_scales_a_field_whose_lipschitz_constant_would_break_the_inverse():
    controls = _grid(6, 6, noise=0.0, base=(0.0, 0.0), tile=10.0)
    for c in controls:  # dux/dx = 2.0 per pixel across a 10 px grid step
        c["dx"] = 2.0 * c["cx"]
        c["tre"] = 5.0
    _, _, disp, report = solve.solve_grid(
        controls, 0.0, 0.99, None, solver="robust", lam=0.0
    )
    assert report["lipschitz_scale"] < 1.0
    assert report["max_operator_norm"] <= solve.MAX_LIPSCHITZ + 1e-9
    assert report["min_jacobian_det"] > 0


def test_robust_report_carries_every_diagnostic_and_leaves_a_clean_field_close_to_input():
    controls = _grid(6, 6, noise=0.05)
    _, _, disp, report = solve.solve_grid(controls, 0.0, 0.99, 256, solver="robust")
    for k in (
        "solver",
        "n_controls",
        "n_rejected_inconsistent",
        "n_infilled",
        "n_unfilled",
        "n_refined",
        "lambda",
        "nmt_threshold",
        "lipschitz_scale",
        "max_operator_norm",
        "min_jacobian_det",
    ):
        assert k in report, k
    assert report["solver"] == "robust" and report["lipschitz_scale"] == 1.0
    raw = solve._lay_out(controls, 0.0, 0.99, 256)[2]
    assert np.abs(np.asarray(disp) - raw).max() < 0.5
