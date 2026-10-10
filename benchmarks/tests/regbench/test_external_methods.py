"""VALIS, ASHLAR and DeeperHistReg, really run -- in the environments that have them.

Each test is skipped where its package is not importable (or not at the pinned version of
that row), so the default suite needs only STARE's dependencies; run this file inside each
method's own image. For each method the test IS the selfcheck: the points must follow the
image the method's own writer produced. ``test_valis_adapters.py``,
``test_ashlar_adapter.py`` and ``test_dhr_adapter.py`` hold everything that can be checked
without the package.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import selfcheck
from benchmarks.regbench.methods import deeperhistreg as dhr


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


# ── DeeperHistReg 1.0.1 ───────────────────────────────────────────────────────
DHR_PIN = (
    "needs deeperhistreg==1.0.1 with torch >= 2.3, SimpleITK and pyvips: run inside "
    "the DeeperHistReg image"
)
# Small enough for a test on the CPU: loaded 1640 x 2048, below both working sizes, so
# S = 5.0 at either parameter set. The cohort-class pair of ``selfcheck.SIZES`` (S = 9.77
# at the 4096 px set) is the launcher's selfcheck job.
DHR_SMALL = [((8200, 10240), (7780, 9720))]


def need_dhr():
    """Skip unless the real DeeperHistReg, at the pinned release, is importable."""
    package = pytest.importorskip("deeperhistreg", reason=DHR_PIN)
    found = dhr.installed_version(package)
    if found != dhr.VERSION:
        pytest.skip(f"{DHR_PIN}; found {found}")
    return package


def test_hr_factory_is_the_8192_configuration():
    """What the row labels claim about the two shipped parameter sets, on the real
    package: a later release that changes a factory fails here."""
    from .test_dhr_adapter import check_factories

    check_factories(need_dhr().configs)


def test_params_equal_the_factory_except_changed_keys():
    from .test_dhr_adapter import check_params_equal_the_factory

    check_params_equal_the_factory(need_dhr().configs)


def test_centre_pads_equal_the_packages_own():
    package = need_dhr()
    from dhr_utils import utils  # the package puts its own folder on sys.path

    assert package.direct_registration.w.warp_landmarks is not None
    for a, b in (((10, 20), (15, 14)), ((7, 9), (7, 9)), ((301, 40), (300, 47))):
        pad_a, pad_b = utils.calculate_pad_value(a, b)
        assert dhr.centre_pads(a, b) == [
            (pad_a[0][0], pad_a[1][0]),
            (pad_b[0][0], pad_b[1][0]),
        ]


def test_a_zero_field_is_written_half_a_field_pixel_right_and_down(tmp_path):
    """The settling measurement. A zero displacement field of 200 x 240 on a 1000 x 1200
    dot image (S = 5): the package's own writer must draw every dot 2.5 px right and down
    of where it is, which is what the as-drawn row is built on. If this fails, the number
    in the message is the measurement: report it, do not adjust anything to fit."""
    package = need_dhr()
    import SimpleITK as sitk
    import tifffile

    pipeline = package.direct_registration
    hw, s = (1000, 1200), 5
    dots = selfcheck.lattice_dots(hw, seed=4)
    plane = np.rint(selfcheck.render_dots(dots, 0, 0, hw[1], hw[0])).astype(np.uint16)
    src, field = tmp_path / "dots.tif", tmp_path / "displacement_field.mha"
    tifffile.imwrite(src, plane, tile=(512, 512), photometric="minisblack")
    zero = np.zeros((2, hw[0] // s, hw[1] // s), np.float32)
    sitk.WriteImage(sitk.GetImageFromArray(zero), str(field))
    out = tmp_path / "warped.tiff"
    pipeline.adf.apply_deformation_pyvips(
        str(src),
        str(src),
        str(out),
        str(field),
        loader=pipeline.loader_mapper["tiff"],
        saver=pipeline.saver_mapper["tiff"],
        level=0,
        pad_value=255.0,
        save_source_only=True,
        to_template_shape=True,
    )
    with tifffile.TiffFile(out) as tif:
        written = tif.pages[0].asarray()
    assert written.shape == hw and written.dtype == np.uint16
    found = selfcheck.find_dots(written)
    hit, j = selfcheck._nearest(found, dots, 6.0)
    assert hit.mean() > 0.9, f"{hit.sum()} of {len(found)} detections near a dot"
    shift = (found[hit] - dots[j[hit]]).mean(axis=0)
    assert shift == pytest.approx([s / 2, s / 2], abs=0.1), f"measured shift {shift}"


@pytest.mark.parametrize("fluo", [False, True], ids=["shipped", "fluo"])
def test_deeperhistreg_points_follow_its_written_image(tmp_path, fluo):
    """The selfcheck IS the test, on the CPU: the as-drawn row must follow the image the
    package's own writer produced, and the estimated-field row must be exactly its
    declared half field pixel away from it."""
    need_dhr()
    opts = {"device": "cpu", **({"fluo": "1"} if fluo else {})}
    label = "deeperhistreg_fluo_cpu" if fluo else "deeperhistreg_cpu"
    rec = selfcheck.run("deeperhistreg", tmp_path, opts, label=label, sizes=DHR_SMALL)
    assert rec["variants"] == {label: selfcheck.OK, f"{label}_asdrawn": selfcheck.OK}, (
        rec
    )
    check = rec["pairs"][0]["variants"][label]
    # the measured offset between the two maps: S / 2 = 2.5 px on each axis
    assert check["declared_px_mean_x"] == pytest.approx(2.5, abs=0.3), check
    assert check["declared_px_mean_y"] == pytest.approx(2.5, abs=0.3), check
