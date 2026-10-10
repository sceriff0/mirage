"""VALIS and DeeperHistReg, really run -- in the environments that have them.

Each test is skipped where its package is not importable (for VALIS: not at the pinned
version of that row), so the default suite needs only STARE's dependencies; run this file
inside each method's own image. For VALIS the test IS the selfcheck: the points must follow
the image VALIS's own writer produced, at every stage. ``test_valis_adapters.py`` holds
everything that can be checked without the package.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import methods, selfcheck
from benchmarks.regbench.cases import load_points, result_dir
from benchmarks.regbench.methods import deeperhistreg as dhr


def _median_tre(out, variant, case):
    pts = load_points(case)
    with np.load(result_dir(out, variant, case) / "warped.npz") as z:
        return float(np.median(np.hypot(*(z["truth_xy"] - pts["truth_ref"]).T)))


def _initial(case):
    pts = load_points(case)
    return float(np.median(np.hypot(*(pts["truth_mov"] - pts["truth_ref"]).T)))


# ── DeeperHistReg's field bookkeeping, without the package ───────────────────
@pytest.mark.parametrize(
    "ref_hw,mov_hw,f",
    [
        ((400, 600), (400, 600), 1),
        ((400, 600), (360, 520), 1),
        ((300, 500), (420, 640), 1),
        ((400, 600), (360, 520), 4),
    ],
)
def test_field_warper_undoes_padding_stretch_and_reduction(ref_hw, mov_hw, f):
    """A known translation, written the way DeeperHistReg stores it: a backward field on the
    padded moving grid, at a coarser resolution than the images it was computed from."""
    t = np.array(
        [7.0, -4.0]
    )  # reference = moving + t, in the pixels of the registered images
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
def need_valis(version):
    """Skip unless the installed VALIS is exactly the pinned one of that row."""
    valis = pytest.importorskip(
        "valis", reason=f"needs valis-wsi=={version}: run inside that row's image"
    )
    found = getattr(valis, "__version__", "?")
    if found != version:
        pytest.skip(f"needs valis-wsi=={version}, found {found}")


def _valis_selfcheck(method, work, label=None, opts=None):
    """The default pair, in its own process (the JVM cannot restart). The large pair
    of ``selfcheck.SIZES`` is the launcher's job: it needs tens of GB."""
    rec = selfcheck.run(
        method, work, opts, label=label, sizes=[(selfcheck.DEFAULT_HW, None)]
    )
    return rec["variants"], rec


def test_valis120_points_follow_its_images(tmp_path):
    need_valis("1.2.0")
    variants, rec = _valis_selfcheck("valis", tmp_path, "valis120")
    assert variants == dict.fromkeys(
        ("valis120_rigid", "valis120", "valis120_micro"), selfcheck.OK
    ), rec


def test_valis120_hires_points_follow_its_images(tmp_path):
    need_valis("1.2.0")
    variants, rec = _valis_selfcheck(
        "valis", tmp_path, "valis120_hires", {"hires": "1"}
    )
    names = ("valis120_hires_rigid", "valis120_hires_nonrigid", "valis120_hires")
    assert variants == dict.fromkeys(names, selfcheck.OK), rec


def test_valis_mirage_points_follow_the_pipelines_images(tmp_path):
    need_valis("1.0.0")
    variants, rec = _valis_selfcheck("valis_mirage", tmp_path)
    names = ("valis_mirage_rigid", "valis_mirage_nonrigid", "valis_mirage")
    assert variants == dict.fromkeys(names, selfcheck.OK), rec


_PATIENT = """
import json, sys
from pathlib import Path
from benchmarks.regbench import methods, selfcheck
from benchmarks.tests.regbench.test_valis_adapters import patient
from valis import registration

built = []
init = registration.Valis.__init__
def counting(self, *args, **kwargs):
    built.append(1)
    return init(self, *args, **kwargs)
registration.Valis.__init__ = counting
work = Path(sys.argv[1])
pair = selfcheck.make_pair(work / "pair", selfcheck.DEFAULT_HW)
done = methods.run_group(
    "valis", patient(work, pair), work / "out", {}, label="valis120"
)
(work / "answer.json").write_text(json.dumps({"done": done, "built": len(built)}))
"""


def test_valis120_registers_a_three_slide_patient_with_one_registrar(tmp_path):
    need_valis("1.2.0")
    root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [sys.executable, "-c", _PATIENT, str(tmp_path)], cwd=root, check=False
    )
    answer = json.loads((tmp_path / "answer.json").read_text())
    names = ["valis120_rigid", "valis120", "valis120_micro"]
    assert answer == {"done": {"P1_r1": names, "P1_r2": names}, "built": 1}


def need_ashlar():
    """Skip unless the installed ASHLAR is the pinned one (``labsyspharm/ashlar:1.20.0``)."""
    ashlar = pytest.importorskip(
        "ashlar", reason="needs ashlar==1.20.0: run inside labsyspharm/ashlar:1.20.0"
    )
    found = getattr(ashlar, "__version__", "?")
    if found != "1.20.0":
        pytest.skip(f"needs ashlar==1.20.0, found {found}")


def test_ashlar_points_follow_its_mosaic(tmp_path):
    """The selfcheck IS the test: the moving dots through the scored map against the
    mosaic ASHLAR wrote, and the reference dots against its re-stitched reference."""
    need_ashlar()
    rec = selfcheck.run(
        "ashlar", tmp_path, mask="overlap", sizes=[(selfcheck.DEFAULT_HW, None)]
    )
    assert rec["variants"] == {"ashlar": selfcheck.OK}, rec
    check = rec["pairs"][0]["variants"]["ashlar"]
    assert check["reference"]["status"] == selfcheck.OK, check
    (path,) = (tmp_path / "out").rglob("run.json")
    info = json.loads(path.read_text())["info"]
    for key in (
        "discard_over_budget",
        "discard_zero_offset",
        "discard_infinite_error",
        "jump_px_p95",
    ):
        assert key in info, key
    # the three causes, rebuilt from the notes, add up to the flag ASHLAR kept
    assert info["discard_rebuild_matches"] is True


def test_deeperhistreg_field_is_read_as_moving_to_reference(synthetic_cases, tmp_path):
    pytest.importorskip("deeperhistreg")
    # equal shapes at full size, then different shapes reduced 2x: padding and scaling both live
    for name, opts in (("smooth", {}), ("crop", {"max_dim": "512"})):
        case = synthetic_cases[name]
        done = methods.run_case(
            "deeperhistreg", case, tmp_path, opts={**opts, "workers": 2}
        )
        assert done == ["deeperhistreg"]
        assert _median_tre(tmp_path, "deeperhistreg", case) < 1.5 < _initial(case), name
