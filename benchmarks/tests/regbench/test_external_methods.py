"""VALIS and DeeperHistReg, really run -- in the environments that have them.

Each test is skipped where its package is not importable, so the default suite needs only
STARE's dependencies; the benchmark workflow runs this file once inside each method's own
environment. What is pinned is the adapter: that the method's transform is read in the right
direction, in full-resolution reference pixels, for slides of equal and of different shape.
"""

import numpy as np
import pytest
from regbench import methods
from regbench.cases import load_points, result_dir
from regbench.methods import deeperhistreg as dhr


def _median_tre(out, variant, case):
    pts = load_points(case)
    with np.load(result_dir(out, variant, case) / "warped.npz") as z:
        return float(np.median(np.hypot(*(z["landmarks"] - pts["landmarks_ref"]).T)))


def _initial(case):
    pts = load_points(case)
    return float(np.median(np.hypot(*(pts["landmarks_mov"] - pts["landmarks_ref"]).T)))


# ── DeeperHistReg's field bookkeeping, without the package ───────────────────
@pytest.mark.parametrize("ref_hw,mov_hw,f", [((400, 600), (400, 600), 1), ((400, 600), (360, 520), 1),
                                             ((300, 500), (420, 640), 1), ((400, 600), (360, 520), 4)])
def test_field_warper_undoes_padding_stretch_and_reduction(ref_hw, mov_hw, f):
    """A known translation, written the way DeeperHistReg stores it: a backward field on the
    padded moving grid, at a coarser resolution than the images it was computed from."""
    t = np.array([7.0, -4.0])  # reference = moving + t, in the pixels of the registered images
    hp, wp = max(ref_hw[0], mov_hw[0]), max(ref_hw[1], mov_hw[1])
    (pr_y, pr_x), (pm_y, pm_x) = dhr.centre_pads(ref_hw, mov_hw)
    fh, fw = hp // 2, wp // 2
    # padded-reference = padded-moving + u  =>  u = t + pad_ref - pad_mov; stored in field px
    u = t + [pr_x - pm_x, pr_y - pm_y]
    field = np.empty((2, fh, fw), np.float32)
    field[0], field[1] = u[0] * fw / wp, u[1] * fh / hp
    warp = dhr.field_warper(field, ref_hw, mov_hw, f, f)
    xy = np.array([[10.0, 20.0], [300.0, 150.0], [511.0, 333.0]]) * f
    assert warp(xy) == pytest.approx(xy + t * f, abs=1e-3)


def test_block_mean_and_centre_pads_follow_the_package_rules():
    img = np.arange(6 * 8, dtype=float).reshape(6, 8)
    assert dhr.block_mean(img, 2).shape == (3, 4)
    assert dhr.block_mean(img, 2)[0, 0] == pytest.approx(img[:2, :2].mean())
    assert dhr.block_mean(img[:5, :7], 2).shape == (2, 3)  # the ragged edge is dropped
    # leading pad is floor(diff / 2), on the smaller image only
    assert dhr.centre_pads((10, 20), (15, 14)) == [(2, 0), (0, 3)]


# ── the real packages ────────────────────────────────────────────────────────
def test_valis_rigid_and_nonrigid_land_in_reference_pixels(synthetic_cases, tmp_path):
    pytest.importorskip("valis")
    for name in ("smooth", "crop"):
        case = synthetic_cases[name]
        done = methods.run_case("valis", case, tmp_path, opts={"micro": "0"}, workers=2)
        assert done == ["valis_rigid", "valis"], name
        assert _median_tre(tmp_path, "valis", case) < 1.0 < _initial(case), name
        assert _median_tre(tmp_path, "valis", case) < _median_tre(tmp_path, "valis_rigid", case)


def test_valis_micro_variant_is_produced(synthetic_cases, tmp_path):
    """Only that the stage runs and yields finite points: whether it IMPROVES on ``valis`` is
    a result the benchmark reports, not something this test asserts."""
    pytest.importorskip("valis")
    case = synthetic_cases["smooth"]
    assert methods.run_case("valis", case, tmp_path, workers=2) == ["valis_rigid", "valis", "valis_micro"]
    with np.load(result_dir(tmp_path, "valis_micro", case) / "warped.npz") as z:
        assert np.isfinite(z["landmarks"]).all()


def test_deeperhistreg_field_is_read_as_moving_to_reference(synthetic_cases, tmp_path):
    pytest.importorskip("deeperhistreg")
    # equal shapes at full size, then different shapes reduced 2x: padding and scaling both live
    for name, opts in (("smooth", {}), ("crop", {"max_dim": "512"})):
        case = synthetic_cases[name]
        assert methods.run_case("deeperhistreg", case, tmp_path, opts=opts, workers=2) == ["deeperhistreg"]
        assert _median_tre(tmp_path, "deeperhistreg", case) < 1.5 < _initial(case), name
