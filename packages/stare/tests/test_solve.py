"""stare.solve: the SOLVE contract around ``solve_dctpls``.

STARE v2 has one solver on one input shape -- REG_TILE's window vectors on the slide-global
lattice. These tests pin the edges of that contract: a pre-v2 control JSON (one point per tile,
no ``vectors``) is refused with a remedy, the per-tile ``accepted`` flag the TRE report carries
follows the same range rule the solve applies, and the Jacobian diagnostics behind the fold
certificate see a fold. The solver's numerics are in test_solve_dctpls.py and
test_solve_vectors.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from stare import solve

S = 128


def _vector_control(ix, iy, vectors):
    return {
        "ix": ix,
        "iy": iy,
        "cx": 0.0,
        "cy": 0.0,
        "dx": 0.0,
        "dy": 0.0,
        "tre": 0.0,
        "error": 0.05,
        "lattice": {"stride": S, "window": 2 * S, "origin": S},
        "vectors": [list(v) for v in vectors],
        "rejected": [],
    }


def _vec(kx, ky, dx, dy, pr=5.0):
    return [kx, ky, S * (kx + 1.0), S * (ky + 1.0), dx, dy, pr, 2.0, 1.0]


def _v1_control(ix, iy):
    """What REG_TILE wrote before the vector grid: one point per tile, no lattice."""
    return {
        "ix": ix,
        "iy": iy,
        "cx": 1024.0 * ix + 512,
        "cy": 1024.0 * iy + 512,
        "dx": 1.0,
        "dy": -0.5,
        "tre": 1.118,
        "error": 0.04,
    }


# ── pre-v2 control JSONs are refused, not silently mis-solved ────────────────
def test_a_control_without_vectors_is_refused_with_the_remedy():
    with pytest.raises(ValueError) as exc:
        solve.solve_dctpls([_v1_control(0, 0), _v1_control(1, 0)], max_disp=256)
    msg = str(exc.value)
    assert "2/2" in msg and "(0,0)" in msg and "(1,0)" in msg
    assert "re-run REG_TILE" in msg


def test_one_pre_v2_tile_in_a_v2_set_is_refused_too():
    """A mixed set is two estimators in one mesh; it used to be warned about and dropped."""
    good = _vector_control(0, 0, [_vec(0, 0, 1.0, 0.0)])
    with pytest.raises(ValueError, match="1/2"):
        solve.solve_dctpls([good, _v1_control(1, 0)], max_disp=256)


def test_no_controls_at_all_is_an_error():
    with pytest.raises(ValueError, match="no control points"):
        solve.solve_dctpls([], max_disp=256)


def test_no_lattice_node_anywhere_leaves_the_slide_rigid():
    """A stride larger than the slide: every tile carries an empty vector list."""
    _gx, _gy, disp, report = solve.solve_dctpls(
        [_vector_control(0, 0, []), _vector_control(1, 0, [])], max_disp=256
    )
    assert disp == [] and report["n_valid"] == 0
    assert report["smoothing_selection"] == "none"


# ── the per-tile accepted flag follows the range rule ────────────────────────
@pytest.mark.parametrize(
    ("vectors", "max_disp", "want"),
    [
        ([_vec(0, 0, 3.0, 4.0)], 256, True),
        ([_vec(0, 0, 3.0, 4.0)], 5.0, False),  # |d| == max_disp is out of range
        ([_vec(0, 0, 3.0, 4.0)], None, True),
        ([_vec(0, 0, float("nan"), 0.0)], None, False),
        ([_vec(0, 0, 300.0, 0.0), _vec(1, 0, 1.0, 0.0)], 256, True),
        ([], 256, False),
    ],
)
def test_tile_accepted_is_one_in_range_vector(vectors, max_disp, want):
    assert solve.tile_accepted(_vector_control(0, 0, vectors), max_disp) is want


def test_tile_accepted_agrees_with_what_the_solve_counted():
    far = _vector_control(0, 0, [_vec(0, 0, 400.0, 0.0)])
    near = _vector_control(1, 0, [_vec(1, 0, 1.0, 0.0), _vec(2, 0, 1.0, 0.0)])
    _gx, _gy, _disp, report = solve.solve_dctpls([far, near], max_disp=256)
    assert report["n_rejected_disp"] == 1
    assert [solve.tile_accepted(c, 256) for c in (far, near)] == [False, True]


# ── the Jacobian behind the fold certificate ─────────────────────────────────
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


def test_the_retired_solvers_are_gone():
    """STARE v2 removed legacy/robust with the one-point-per-tile path; nothing dispatches."""
    for name in (
        "solve_grid",
        "solve_legacy",
        "solve_robust",
        "accept",
        "SOLVERS",
        "_lattice_from_controls",
        "normalized_median_test",
        "tikhonov_smooth",
    ):
        assert not hasattr(solve, name), name
