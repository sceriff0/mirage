"""The VALIS adapters, without VALIS.

VALIS is not installed where this suite runs, so everything here runs against
``fake_valis``: a recording stand-in that keeps VALIS's geometry conventions and draws
images the way its writer is written to (see that module). What these tests pin:

* the exact call sequence and arguments of each row (``valis120`` is upstream's default
  call, ``valis120_hires`` upstream's example, ``valis_mirage`` the pipeline's own
  function, really executed from ``bin/register.py``);
* the point-map algebra against the emulated writer, in the selfcheck's own comparison,
  and that a wrong direction, swapped axes, an offset and a wrongly rounded crop are each
  reported ``inconsistent``;
* that fields are copied at the yield and each image is written before a later stage
  rewrites the fields.

What they cannot pin is VALIS itself: ``test_external_methods.py`` holds the tests that
need the real package, skipped here.
"""

import ast
import json
import math
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

from benchmarks.regbench import METHODS, methods
from benchmarks.regbench import selfcheck as sc
from benchmarks.regbench.cases import Case, load_case, result_dir, write_case
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap
from benchmarks.regbench.methods import _valis_points as vp
from benchmarks.regbench.methods import _valis_writer_field as wf
from benchmarks.regbench.methods import valis as valis120
from benchmarks.regbench.methods import valis_mirage

from . import fake_valis as fv

ROOT = Path(__file__).resolve().parents[3]
SMALL = (512, 640)
MIRAGE = ("valis_mirage_rigid", "valis_mirage_nonrigid", "valis_mirage")
V120 = ("valis120_rigid", "valis120", "valis120_micro")
HIRES = ("valis120_hires_rigid", "valis120_hires_nonrigid", "valis120_hires")


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    """The fake ``valis`` and ``pyvips`` for one test, gone afterwards."""
    monkeypatch.setenv("MPLCONFIGDIR", os.environ.get("MPLCONFIGDIR", "/tmp/mpl"))
    fv.install(monkeypatch.setitem)
    yield fv
    fv.configure()


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    return sc.make_pair(tmp_path_factory.mktemp("valis_pair"), SMALL)


def registrar(pair, **kwargs):
    """A fake registrar over the pair, registered."""
    reg = fv.Valis(
        str(Path(pair["reference"]).parent),
        str(Path(pair["reference"]).parent / "valis"),
        img_list=[pair["reference"], pair["moving"]],
        reference_img_f=pair["reference"],
        **kwargs,
    )
    reg.register()
    return reg, reg.get_ref_slide(), reg.get_slide(pair["moving"])


def warp_tools():
    return sys.modules["valis.warp_tools"]


def dots_image(pair, slide, non_rigid=True, rounding=None):
    dots = tifffile.imread(pair["moving"])[pair["dots_channel"]]
    return fv.draw(slide, [dots], non_rigid=non_rigid, rounding=rounding)[0]


def run_json(out, variant, case):
    return json.loads((result_dir(out, variant, case) / "run.json").read_text())


# ── names ─────────────────────────────────────────────────────────────────────
def test_registered_and_named():
    assert "valis" in METHODS and "valis_mirage" in METHODS
    assert valis120.SCOPE == valis_mirage.SCOPE == "group"
    assert valis120.VARIANTS == ("valis_rigid", "valis", "valis_micro")
    assert valis_mirage.VARIANTS == MIRAGE
    assert valis120.SELFCHECK_SUBPROCESS and valis_mirage.SELFCHECK_SUBPROCESS
    assert methods.scope_of("valis") == methods.scope_of("valis_mirage") == "group"
    assert methods.variant_names("valis_mirage") == list(MIRAGE)
    assert methods.variant_names("valis", "valis120") == list(V120)
    # the one label whose names do not follow the replace rule
    assert methods.variant_names("valis", "valis120_hires") == list(HIRES)
    assert not hasattr(valis120, "register") and not hasattr(valis_mirage, "register")


def test_a_label_table_of_the_wrong_length_is_refused():
    bad = types.SimpleNamespace(
        NAME="m", VARIANTS=("m_a", "m"), VARIANT_LABELS={"x": ("x",)}
    )
    with pytest.raises(SystemExit, match="names 1 variants"):
        methods.variant_names(bad, "x")
    assert methods.variant_names(bad, "y") == ["y_a", "y"]


def test_modules_import_without_valis_or_torch():
    code = (
        "import sys\n"
        "from benchmarks.regbench.methods import valis, valis_mirage, _valis_points\n"
        "bad = [m for m in ('valis', 'torch', 'pyvips', 'jpype') if m in sys.modules]\n"
        "raise SystemExit(','.join(bad) or 0)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


def test_selfcheck_sizes_reach_every_stage():
    """The default pair alone would leave stages unrun: 1.2.0's ``register_micro`` returns
    early unless its size exceeds the first pass's 2048 px (``registration.py:4862-4871``),
    and the hires size is a quarter of the slide; the pipeline's is an eighth. The large
    pair is also the only one whose field is applied at a scale near the cohort's."""
    for name in ("valis", "valis120", "valis120_hires", "valis_mirage"):
        sizes = sc.sizes_for(name)
        assert sizes[0] == (sc.DEFAULT_HW, None), name
        ref_hw, mov_hw = sizes[-1]
        smallest_long_side = min(max(ref_hw), max(mov_hw))
        assert smallest_long_side >= 29000, name
        assert math.floor(smallest_long_side * 0.125) > 2048  # valis_mirage's micro
        assert math.floor(smallest_long_side * valis120.MICRO_REG_FRACTION) > 2048
        assert smallest_long_side / 2048 > 14  # the first field, scaled up 14x
        # one patient of three slides, before the large pair
        assert sizes[2] == (sc.DEFAULT_HW, None, None) and sc.is_patient(sizes[2])
        assert len(sizes) == 4
    # on the default pair the hires micro pass cannot run, and the row says so
    assert math.floor(max(sc.DEFAULT_HW) * valis120.MICRO_REG_FRACTION) < 2048


# ── the point path ────────────────────────────────────────────────────────────
class Recorder:
    """A ``warp_tools`` that records what it is called with."""

    def __init__(self):
        self.calls = []

    def warp_xy_from_to(self, xy, **kw):
        self.calls.append(("warp_xy_from_to", xy, kw))
        return np.asarray(xy, float) + 1.0

    def warp_xy_inv(self, xy, **kw):
        self.calls.append(("warp_xy_inv", xy, kw))
        return np.asarray(xy, float) * 2.0

    def vips2numpy(self, vi):
        self.calls.append(("vips2numpy", vi, {}))
        return np.asarray(vi.array)


def slide_like(name, shift, fwd=None):
    return types.SimpleNamespace(
        name=name,
        M=np.array([[1.0, 0.0, shift], [0.0, 1.0, 2 * shift], [0.0, 0.0, 1.0]]),
        processed_img_shape_rc=(100 + shift, 120 + shift),
        reg_img_shape_rc=(140, 150),
        slide_dimensions_wh=np.array([[1200 + shift, 1000 + shift], [600, 500]]),
        aligned_slide_shape_rc=np.array([1400, 1500]),
        fwd_dxdy=fwd,
    )


def test_point_path_passes_the_backward_field_and_the_shapes():
    wt = Recorder()
    ref = slide_like("ref", 0, fwd=np.zeros((2, 7, 8)))
    mov = slide_like("mov", 3)
    bk = np.ones((2, 7, 8))
    pmap = vp.ref_to_mov(ref, mov, bk, wt)
    assert isinstance(pmap, PointMap) and pmap.direction == REF_TO_MOV
    xy = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert pmap(xy) == pytest.approx(xy + 1.0)
    (name, sent, kw) = wt.calls[-1]
    assert name == "warp_xy_from_to" and sent == pytest.approx(xy)
    assert sorted(kw) == sorted(
        [
            "from_M",
            "from_transformation_src_shape_rc",
            "from_transformation_dst_shape_rc",
            "from_src_shape_rc",
            "from_dst_shape_rc",
            "from_fwd_dxdy",
            "to_M",
            "to_transformation_src_shape_rc",
            "to_transformation_dst_shape_rc",
            "to_src_shape_rc",
            "to_dst_shape_rc",
            "to_bk_dxdy",
        ]
    )
    assert kw["to_bk_dxdy"] is bk  # the object given, not a re-read of the slide
    assert kw["from_M"] == pytest.approx(ref.M) and kw["to_M"] == pytest.approx(mov.M)
    assert kw["from_transformation_src_shape_rc"] == (100, 120)
    assert kw["to_transformation_src_shape_rc"] == (103, 123)
    assert kw["from_transformation_dst_shape_rc"] == (140, 150)
    assert kw["to_transformation_dst_shape_rc"] == (140, 150)
    assert kw["from_src_shape_rc"] == (1000, 1200)  # level 0, as (rows, cols)
    assert kw["to_src_shape_rc"] == (1003, 1203)
    assert kw["from_dst_shape_rc"] == kw["to_dst_shape_rc"] == (1400, 1500)
    assert isinstance(kw["from_fwd_dxdy"], np.ndarray)
    # a stage's own matrix can be handed in
    other = np.eye(3) * 2.0
    vp.ref_to_mov(ref, mov, bk, wt, mov_M=other)(xy)
    assert wt.calls[-1][2]["to_M"] == pytest.approx(other)


def test_rigid_stage_passes_no_field():
    wt = Recorder()
    ref = slide_like("ref", 0, fwd=np.zeros((2, 7, 8)))
    vp.ref_to_mov(ref, slide_like("mov", 3), None, wt)(np.zeros((1, 2)))
    kw = wt.calls[-1][2]
    assert kw["to_bk_dxdy"] is None and kw["from_fwd_dxdy"] is None


def test_the_reference_field_is_converted_and_zero():
    wt = Recorder()
    stored = fv.VipsImage(np.zeros((7, 8, 2), np.float32))  # a tiled run's field
    ref = slide_like("ref", 0, fwd=stored)
    vp.ref_to_mov(ref, slide_like("mov", 3), np.ones((2, 7, 8)), wt)(np.zeros((1, 2)))
    sent = wt.calls[-1][2]["from_fwd_dxdy"]
    assert isinstance(sent, np.ndarray) and sent.shape == (2, 7, 8)
    assert not sent.any()
    assert any(c[0] == "vips2numpy" for c in wt.calls)

    moved = np.zeros((7, 8, 2), np.float32)
    moved[3, 4, 0] = 0.25
    ref = slide_like("ref", 0, fwd=fv.VipsImage(moved))
    with pytest.raises(ValueError, match="non-zero displacement field"):
        vp.ref_to_mov(ref, slide_like("mov", 3), np.ones((2, 7, 8)), wt)


def test_a_field_is_copied_and_a_stored_one_is_loaded_whole():
    wt = Recorder()
    live = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    kept = vp.numpy_field(live, wt)
    live += 100.0
    assert kept[0, 0, 0] == 0.0  # a copy: what the slide held THEN
    assert vp.numpy_field(None, wt) is None
    assert vp.numpy_field([live[0], live[1]], wt).shape == (2, 3, 4)
    stored = fv.VipsImage(np.dstack([np.full((3, 4), 1.5), np.full((3, 4), -2.0)]))
    loaded = vp.numpy_field(stored, wt)
    assert (
        loaded.shape == (2, 3, 4) and loaded[0, 0, 0] == 1.5 and loaded[1, 0, 0] == -2
    )
    with pytest.raises(ValueError, match="displacement field"):
        vp.numpy_field(np.zeros((3, 4)), wt)


def test_the_image_map_is_warp_xy_inv_on_the_writers_crop(pair):
    reg, ref, mov = registrar(pair)
    snap = vp.snapshot(mov, ref, warp_tools(), True, "ceil")
    wt = Recorder()
    pmap = vp.image_map(snap, wt)
    assert pmap.direction == REF_TO_MOV
    xy = np.array([[10.0, 20.0], [300.0, 150.0]])
    pmap(xy)
    name, sent, kw = wt.calls[-1]
    assert name == "warp_xy_inv"
    assert sent == pytest.approx(xy + snap["origin_xy"])
    assert sorted(kw) == sorted(
        [
            "M",
            "transformation_src_shape_rc",
            "transformation_dst_shape_rc",
            "src_shape_rc",
            "dst_shape_rc",
            "bk_dxdy",
        ]
    )  # no forward field: nothing is inverted
    assert kw["bk_dxdy"] is snap["bk_dxdy"] and kw["M"] == pytest.approx(mov.M)
    assert kw["src_shape_rc"] == SMALL and kw["dst_shape_rc"] == tuple(
        int(v) for v in mov.aligned_slide_shape_rc
    )
    assert vp.image_map(snap, wt)(np.empty((0, 2))).shape == (0, 2)


@pytest.mark.parametrize("rounding", ["ceil", "trunc"])
def test_the_crop_probe_measures_the_installed_rounding(rounding):
    fv.configure(rounding=rounding)
    raw = np.array([212.36, 163.92])
    want = {"ceil": [213.0, 164.0], "trunc": [212.0, 163.0]}[rounding]
    assert vp.writer_rounding(raw, warp_tools()).tolist() == want
    # a whole number needs no rounding, whatever the rule
    assert vp.writer_rounding([40.0, 7.0], warp_tools()).tolist() == [40.0, 7.0]
    # the rule can be stated instead of measured
    assert vp.writer_rounding(raw, None, "ceil").tolist() == [213.0, 164.0]
    assert vp.writer_rounding(raw, None, "trunc").tolist() == [212.0, 163.0]
    assert vp.writer_rounding(raw, None, "round").tolist() == [212.0, 164.0]
    with pytest.raises(ValueError, match="outside the registered canvas"):
        vp.writer_rounding([-0.5, 3.0], warp_tools())
    with pytest.raises(ValueError, match="crop_round"):
        vp.writer_rounding(raw, warp_tools(), "floor")


def test_the_crop_origin_is_the_slides_own_box(pair):
    reg, ref, mov = registrar(pair)
    origin, raw = vp.crop_origin(mov, ref, warp_tools(), "probe")
    s = np.array(SMALL[::-1]) / np.array(ref.processed_img_shape_rc[::-1])
    assert raw == pytest.approx(-ref.M[:2, 2] * s)
    assert (raw % 1 > 0.05).all()  # the fake places the reference off the pixel grid
    assert origin.tolist() == np.ceil(raw).tolist()


# ── the algebra against the emulated writer ───────────────────────────────────
@pytest.mark.parametrize("rounding", ["ceil", "trunc"])
@pytest.mark.parametrize("stage", ["rigid", "nonrigid", "micro", "stored"])
def test_the_image_map_follows_the_emulated_writer(pair, stage, rounding):
    fv.configure(rounding=rounding)
    reg, ref, mov = registrar(pair)
    if stage in ("micro", "stored"):
        reg.max_non_rigid_registration_dim_px = 0
        reg.register_micro()
    mov.stored_dxdy = stage == "stored"
    non_rigid = stage != "rigid"
    snap = vp.snapshot(mov, ref, warp_tools(), non_rigid)
    assert (snap["bk_dxdy"] is None) == (not non_rigid)
    assert snap["stored"] == (stage == "stored")
    image = dots_image(pair, mov, non_rigid)
    res = sc.compare(vp.image_map(snap, warp_tools()), pair["dots_mov_xy"], image)
    assert res["status"] == sc.OK, res["failed"]
    assert res["err_px_p95"] < 0.25 and res["n_found"] >= 0.99 * res["n_dots"]


# ── the field as the writer reads it (methods/_valis_writer_field.py) ─────────
# A field about 25 canvas px per sample, zero outside the tissue box, as VALIS's is: the
# step where the point path's spline and the writer's libvips bicubic part.
STEP = {"tissue_step": True, "field_scale": 0.1}


@pytest.mark.parametrize("stored", [False, True])
def test_at_a_tissue_step_only_the_writers_reading_of_the_field_follows_its_image(
    pair, stored
):
    """The cluster's selfcheck of 2026-10-24: every non-rigid VALIS row passed on the
    whole image (p95 <= 0.18 px) and failed in the 512 px blocks on the tissue boundary
    (0.4 to 1.9 px), every rigid row passed. The spline reading must be seen to fail
    here, or this fake could not have caught it; the writer's reading must pass."""
    fv.configure(**STEP)
    reg, ref, mov = registrar(pair)
    mov.stored_dxdy = stored
    snap = vp.snapshot(mov, ref, warp_tools(), True)
    image = dots_image(pair, mov, True)
    dots = pair["dots_mov_xy"]
    new = sc.compare(wf.image_map(snap, warp_tools()), dots, image)
    assert new["status"] == sc.OK, new["failed"]
    old = sc.compare(vp.image_map(snap, warp_tools()), dots, image)
    assert old["status"] == sc.INCONSISTENT
    assert [c["how"] for c in fv.calls("interpolate")] == ["bicubic"] * len(
        fv.calls("interpolate")
    )


def test_a_rigid_stage_is_the_point_path_unchanged(pair):
    fv.configure(**STEP)
    reg, ref, mov = registrar(pair)
    snap = vp.snapshot(mov, ref, warp_tools(), False)
    xy = np.array([[3.5, 7.25], [400.0, 300.0], [639.0, 511.0]])
    assert np.array_equal(
        wf.image_map(snap, warp_tools()).fn(xy), vp.image_map(snap, warp_tools()).fn(xy)
    )
    assert not fv.calls("affine")


def test_the_writers_field_is_read_in_windows_and_nan_stays_nan(pair):
    fv.configure(**STEP)
    reg, ref, mov = registrar(pair)
    snap = vp.snapshot(mov, ref, warp_tools(), True)
    vips_dxdy, matrix = wf.writer_field(snap, warp_tools())
    xy = np.array([[1.0, 2.0], [300.5, 200.25], [np.nan, 4.0], [511.0, 400.0]])
    whole = wf.displacement(xy, vips_dxdy, matrix, warp_tools(), window=4096)
    small = wf.displacement(xy, vips_dxdy, matrix, warp_tools(), window=64)
    assert np.allclose(whole, small, equal_nan=True)
    assert np.isnan(whole[2]).all() and np.isfinite(whole[[0, 1, 3]]).all()
    assert len(fv.calls("affine")) == 1 + 3  # one window, then one per block
    assert len(wf.displacement(np.empty((0, 2)), vips_dxdy, matrix, warp_tools())) == 0


def test_only_the_valis_adapters_carry_the_writers_field():
    """It is a module of its own so that no other method's reuse_key moves with it."""
    for name in METHODS:
        files = {p.name for p in methods.code_files(name)}
        assert ("_valis_writer_field.py" in files) == (
            name in ("valis", "valis_mirage")
        )


WRONG = {
    "direction": lambda s, wt: PointMap(vp.image_map(s, wt).fn, MOV_TO_REF),
    "swapped_axes": lambda s, wt: PointMap(
        lambda xy: vp.image_map(s, wt).fn(np.asarray(xy)[:, ::-1])[:, ::-1], REF_TO_MOV
    ),
    "one_pixel": lambda s, wt: vp.image_map(
        {**s, "origin_xy": s["origin_xy"] + [1.0, 0.0]}, wt
    ),
    "third_of_a_pixel": lambda s, wt: vp.image_map(
        {**s, "origin_xy": s["origin_xy"] + [0.0, 0.3]}, wt
    ),
    "no_crop_origin": lambda s, wt: vp.image_map({**s, "origin_xy": np.zeros(2)}, wt),
    "unrounded_origin": lambda s, wt: vp.image_map(
        {**s, "origin_xy": s["origin_raw_xy"]}, wt
    ),
    "field_sign": lambda s, wt: vp.image_map({**s, "bk_dxdy": -s["bk_dxdy"]}, wt),
    "field_axes": lambda s, wt: vp.image_map({**s, "bk_dxdy": s["bk_dxdy"][::-1]}, wt),
    "no_field": lambda s, wt: vp.image_map({**s, "bk_dxdy": None}, wt),
    "the_reference_matrix": lambda s, wt: vp.image_map({**s, "M": s["ref"]["M"]}, wt),
    "valis_from_to": lambda s, wt: vp.from_to_map(s, wt),
}


@pytest.mark.parametrize("wrong", sorted(WRONG))
def test_a_wrong_map_is_inconsistent(pair, wrong):
    """Each of these maps is one mistake away from the right one; none may pass."""
    reg, ref, mov = registrar(pair)
    snap = vp.snapshot(mov, ref, warp_tools(), True, with_reference=True)
    assert not snap["ref"]["fwd_dxdy"].any()
    assert vp.snapshot(mov, ref, warp_tools(), True)["ref"]["fwd_dxdy"] is None
    image = dots_image(pair, mov)
    right = sc.compare(vp.image_map(snap, warp_tools()), pair["dots_mov_xy"], image)
    assert right["status"] == sc.OK
    res = sc.compare(WRONG[wrong](snap, warp_tools()), pair["dots_mov_xy"], image)
    assert res["status"] == sc.INCONSISTENT, wrong


def test_a_crop_rounded_the_other_way_is_inconsistent(pair):
    """The writer rounds up, the adapter is told it rounds down: one pixel apart."""
    fv.configure(rounding="ceil")
    reg, ref, mov = registrar(pair)
    image = dots_image(pair, mov)
    told = vp.snapshot(mov, ref, warp_tools(), True, "trunc")
    res = sc.compare(vp.image_map(told, warp_tools()), pair["dots_mov_xy"], image)
    assert res["status"] == sc.INCONSISTENT
    assert abs(res["offset_px_mean_x"]) > 0.5 and abs(res["offset_px_mean_y"]) > 0.5


def test_valis_own_point_function_leaves_its_writer_by_the_recorded_amount(pair):
    """``Slide.warp_xy_from_to`` places a reference pixel on the canvas through the
    reference's matrix; the writer crops at a rounded origin. The record is that gap."""
    reg, ref, mov = registrar(pair)
    snap = vp.snapshot(mov, ref, warp_tools(), True)
    gap = vp.stage_info(snap, warp_tools(), 0.325)["valis_from_to_minus_writer_px"]
    s = np.array(SMALL[::-1]) / np.array(ref.processed_img_shape_rc[::-1])
    k = np.array(ref.aligned_slide_shape_rc[::-1]) / np.array(
        ref.reg_img_shape_rc[::-1]
    )
    raw = -ref.M[:2, 2] * s
    assert gap["at_origin"] == pytest.approx(raw * k / s - np.ceil(raw), abs=1e-9)
    far = np.array(SMALL[::-1]) - 1.0
    want = (far / s + ref.M[:2, 2] * -1) * k - (far + np.ceil(raw))
    assert gap["at_far_corner"] == pytest.approx(want, abs=1e-9)
    assert max(abs(v) for v in gap["at_origin"]) > 0.15  # more than the gate allows


def test_stage_info_and_resolution(pair):
    reg, ref, mov = registrar(pair)
    rigid = vp.snapshot(mov, ref, warp_tools(), False)
    info = vp.stage_info(rigid, warp_tools(), 0.325)
    assert info["field_shape"] is None and info["um_per_px"]["field"] is None
    assert info["um_per_px"]["rigid"] == pytest.approx(0.325 * 640 / 200)
    assert info["crop"] == "reference" and info["crop_round"] == "probe"
    full = vp.snapshot(mov, ref, warp_tools(), True)
    info = vp.stage_info(full, warp_tools(), 0.325)
    assert info["field_shape"] == list(full["bk_dxdy"].shape[1:])
    assert 0.325 < info["um_per_px"]["field"] < info["um_per_px"]["rigid"]
    assert vp.stage_info(full, warp_tools())["um_per_px"] == {
        "rigid": None,
        "field": None,
    }
    json.dumps(info)


def test_the_valid_box_bounds_what_the_moving_slide_covers():
    def fn(xy):  # the moving slide is the reference shifted by (100, 40)
        return np.asarray(xy, float) - [100.0, 40.0]

    box = vp.footprint_box(fn, (600, 800), (300, 400), max_side=100)
    step = 8
    x0, y0, x1, y1 = box
    assert 100 - step <= x0 <= 100 and 40 - step <= y0 <= 40
    assert 500 <= x1 <= 500 + step + 1 and 340 <= y1 <= 340 + step + 1
    assert vp.footprint_box(lambda xy: xy + 1e6, (600, 800), (300, 400)) is None
    assert vp.footprint_box(lambda xy: xy, (600, 800), (600, 800)) == [0, 0, 800, 600]


# ── the patient's slides ──────────────────────────────────────────────────────
def case_of(root, pair, case_id, mov, group="P1", ref=None):
    h, w = pair["ref_hw"]
    case = Case(
        dataset="selfcheck",
        case_id=case_id,
        group=group,
        ref_nuclear=ref or pair["reference"],
        mov_nuclear=str(mov),
        ref_image=ref or pair["reference"],
        mov_image=str(mov),
        modality="fluorescence",
        diagonal=math.hypot(h, w),
        pixel_size_um=pair["pixel_size_um"],
        ref_hw=list(pair["ref_hw"]),
        mov_hw=list(pair["mov_hw"]),
        extra={"ref_channel": 0, "mov_channel": 0, "channel_names": ["DAPI", "dots"]},
    )
    return load_case(write_case(root, case))


def patient(tmp_path, pair, rounds=2):
    cases = []
    for i in range(rounds):
        mov = tmp_path / f"round{i + 1}.ome.tif"
        shutil.copyfile(pair["moving"], mov)
        cases.append(case_of(tmp_path / "cases", pair, f"P1_r{i + 1}", mov))
    return cases


def test_slide_names_follow_valis():
    assert vp.slide_name("/a/b/P1_DAPI_CD3.ome.tif") == "P1_DAPI_CD3"
    assert vp.slide_name("x.OME.TIFF") == "x" and vp.slide_name("y.tiff") == "y"


def test_one_reference_and_one_name_per_slide(tmp_path, pair):
    cases = patient(tmp_path, pair)
    ref, moving = vp.patient_slides(cases)
    assert ref == pair["reference"] and list(moving) == ["P1_r1", "P1_r2"]
    other = tmp_path / "elsewhere" / "reference.ome.tif"
    other.parent.mkdir()
    shutil.copyfile(pair["reference"], other)
    two_refs = [
        cases[0],
        case_of(tmp_path / "c2", pair, "P1_r9", other, ref=str(other)),
    ]
    with pytest.raises(ValueError, match="2 reference slides"):
        vp.patient_slides(two_refs)
    # two different files of one basename are placed apart (parent folder in front) ...
    clash = tmp_path / "elsewhere" / "round1.ome.tif"
    shutil.copyfile(pair["moving"], clash)
    same_name = [cases[0], case_of(tmp_path / "c3", pair, "P1_r8", clash)]
    assert vp.patient_slides(same_name)[1]["P1_r8"] == str(clash)
    # two cases naming ONE moving file are refused, with their own message
    shared = [cases[0], case_of(tmp_path / "c5", pair, "P1_r6", cases[0].mov_image)]
    with pytest.raises(ValueError, match="two cases name the same moving slide"):
        vp.patient_slides(shared)
    # ... and refused only where even that leaves two of one VALIS name
    twin = tmp_path / "x" / "elsewhere" / "round1.ome.tif"
    twin.parent.mkdir(parents=True)
    shutil.copyfile(pair["moving"], twin)
    cases3 = [*same_name, case_of(tmp_path / "c4", pair, "P1_r7", twin)]
    with pytest.raises(
        ValueError, match="would both be the VALIS slide 'elsewhere_round1'"
    ):
        vp.patient_slides(cases3)


def test_slides_are_linked_or_copied_never_symlinked(tmp_path, pair, monkeypatch):
    placed, how = vp.place_slides([pair["moving"]], tmp_path / "in")
    dst = Path(placed[pair["moving"]])
    assert dst.name == "moving.ome.tif" and not dst.is_symlink()
    assert how == {"moving.ome.tif": "hard link"}
    assert os.stat(dst).st_ino == os.stat(pair["moving"]).st_ino

    def refuse(*a, **k):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", refuse)
    placed, how = vp.place_slides([pair["moving"]], tmp_path / "in")
    assert how == {"moving.ome.tif": "copy"} and not dst.is_symlink()
    assert os.stat(dst).st_ino != os.stat(pair["moving"]).st_ino


# ── VALIS 1.2.0 as shipped ────────────────────────────────────────────────────
def selfcheck(tmp_path, label="valis120", opts=None, mov_hw=None, **config):
    fv.configure(**config)
    rec = sc.run(
        "valis",
        tmp_path / "w",
        {"workers": 1, **(opts or {})},
        label=label,
        sizes=[(SMALL, mov_hw)],
        in_process=True,
        keep_work=True,
    )
    return rec, rec["pairs"][0]["variants"]


def test_valis120_is_upstreams_default_call_and_its_points_follow_its_images(tmp_path):
    rec, variants = selfcheck(tmp_path)
    assert rec["status"] == sc.OK and rec["variants"] == dict.fromkeys(V120, sc.OK)
    for name in V120:
        assert variants[name]["err_px_p95"] < 0.25, name
        assert variants[name]["direction"] == REF_TO_MOV
    (built,) = fv.calls("Valis")
    assert sorted(built["kwargs"]) == ["img_list", "reference_img_f"]
    assert [Path(f).name for f in built["kwargs"]["img_list"]] == [
        "reference.ome.tif",
        "moving.ome.tif",
    ]
    assert Path(built["kwargs"]["reference_img_f"]).name == "reference.ome.tif"
    assert [c["kwargs"] for c in fv.calls("register")] == [{}]
    assert [c["kwargs"] for c in fv.calls("register_micro")] == [{}]  # no argument
    assert [c["kwargs"] for c in fv.calls("warp_and_save_slides")] == [
        {"crop": "reference"},  # `valis`
        {"non_rigid": False, "crop": "reference"},  # the rigid stage, selfcheck only
        {"crop": "reference"},  # `valis_micro`
    ]
    assert not fv.calls("get_inverse_field") and not fv.calls("warp_xy_from_to")
    # the JVM is killed once, and the points are still evaluated afterwards
    order = [c["call"] for c in fv.CALLS]
    assert order.count("kill_jvm") == 1
    assert "warp_xy_inv" in order[order.index("kill_jvm") :]


PATIENT = (SMALL, None, None)
ROUNDS = ["patient_512x640_round1", "patient_512x640_round2"]
SLIDES = ["reference.ome.tif", "moving.ome.tif", "moving_2.ome.tif"]


@pytest.mark.parametrize(
    "label, opts, names",
    [("valis120", None, V120), ("valis120_hires", {"hires": 1}, HIRES)],
)
def test_a_patient_of_three_slides_is_one_valis_run(tmp_path, label, opts, names):
    """The path every cohort patient takes: both moving rounds handed to ONE ``Valis``
    object, one ``register``, and each round's points against that round's image."""
    fv.configure()
    rec = sc.run(
        "valis",
        tmp_path / "w",
        {"workers": 1, **(opts or {})},
        label=label,
        sizes=[PATIENT],
        in_process=True,
        keep_work=True,
    )
    assert rec["status"] == sc.OK and rec["variants"] == dict.fromkeys(names, sc.OK)
    assert [p["case_id"] for p in rec["pairs"]] == ROUNDS
    (built,) = fv.calls("Valis")  # one run for the patient
    assert [Path(f).name for f in built["kwargs"]["img_list"]] == SLIDES
    assert Path(built["kwargs"]["reference_img_f"]).name == "reference.ome.tif"
    assert len(fv.calls("register")) == 1
    seen = set()
    for p in rec["pairs"]:
        for name in names:
            v = p["variants"][name]
            assert v["status"] == sc.OK and v["err_px_p95"] < 0.25, (p["round"], name)
            assert v["n_found"] >= 0.99 * v["n_dots"] > 20
            assert v["n_covered"] >= 0.99 * v["n_expected"] > 20
            seen.add(v["native"]["files"][0])
    assert len(seen) == 2 * len(names)  # an image of its own per round and stage
    assert rec["cases"][0]["group"] == "patient_512x640" and rec["cases"][0]["patient"]


def test_a_round_whose_image_is_another_rounds_is_inconsistent(tmp_path, monkeypatch):
    """What the patient case is for: round 2's points checked against round 1's image
    (a writer or an adapter that mixes the rounds up) cannot pass."""
    fv.configure()
    real = sc._check_variant

    def swapped(mod, variant, final, rec, got, extras, pair, mask, view=None):
        if pair["round"] == 2 and rec and rec.get("native"):
            native = dict(rec["native"])
            native["files"] = [
                f.replace("round2", "round1").replace("moving_2", "moving")
                for f in native["files"]
            ]
            assert all(Path(f).exists() for f in native["files"])
            rec = {**rec, "native": native}
        return real(mod, variant, final, rec, got, extras, pair, mask, view)

    monkeypatch.setattr(sc, "_check_variant", swapped)
    rec = sc.run(
        "valis",
        tmp_path / "w",
        {"workers": 1},
        label="valis120",
        sizes=[PATIENT],
        in_process=True,
        keep_work=True,
    )
    first, second = rec["pairs"]
    assert first["status"] == sc.OK and second["status"] == sc.INCONSISTENT
    assert rec["not_ok"]


def test_the_default_row_is_written_before_micro_runs(tmp_path, pair):
    """A cohort run (no stage images): two images per round, `valis` from the fields as
    they were before ``register_micro`` rewrote them."""
    cases = patient(tmp_path, pair)
    out = tmp_path / "out"
    done = methods.run_group("valis", cases, out, {"workers": 1}, label="valis120")
    assert done == {"P1_r1": list(V120), "P1_r2": list(V120)}
    assert len(fv.calls("Valis")) == 1  # one registrar for the patient
    order = [c["call"] for c in fv.CALLS]
    writes = [i for i, c in enumerate(order) if c == "warp_and_save_slides"]
    micro = order.index("register_micro")
    assert len(writes) == 2 and writes[0] < micro < writes[1]
    assert [c["generation"] for c in fv.calls("warp_and_save_slides")] == [0, 1]
    for case in cases:
        assert run_json(out, "valis120_rigid", case)["native"] is None
        for variant, slide in (("valis120", 0), ("valis120_micro", 1)):
            run = run_json(out, variant, case)
            native = run["native"]
            (path,) = native["files"]
            home = result_dir(out, variant, case) / "native"
            assert Path(path).parent == home and Path(path).exists()
            assert Path(path).name == f"{vp.slide_name(case.mov_image)}.ome.tiff"
            assert native["channels"] == 2 and "origin_xy" not in native
            x0, y0, x1, y1 = native["valid_box"]
            assert 0 <= x0 < x1 <= SMALL[1] and 0 <= y0 < y1 <= SMALL[0]
            assert "warp_and_save_slides(dst, crop='reference')" in native["writer"]
            assert run["write_s"] > 0 and run["direction"] == REF_TO_MOV
            # what the driver hands a method through opts is not an option of the run
            assert set(run["opts"]) == {"workers"} and run["label"] == "valis120"
            assert run["info"]["workers_seen"] >= 1
            assert run["info"]["last_slide_in_slide_dict"] == "round2"
            assert run["info"]["nuclear_channel"]["round1"] == {
                "index": 0,
                "name": "DAPI",
            }
            assert run["info"]["stage"]["um_per_px"]["rigid"] > 0.325
    assert run_json(out, "valis120_micro", cases[0])["info"]["micro_ran"] is True
    # the scratch is gone, the images are not
    assert not (result_dir(out, "valis120_micro", cases[0]) / "work").exists()


def test_fields_are_copied_at_the_yield(tmp_path, pair):
    """The fake's micro pass scribbles over the arrays the first pass handed out and its
    field property changes its answer. A map taken at the `valis` yield must not move."""
    cases = patient(tmp_path, pair, rounds=1)
    xy = np.array([[100.0, 120.0], [400.0, 300.0], [250.0, 60.0]])
    seen, kept = {}, {}

    def on_variant(case, variant, pmap, info):
        seen[variant] = np.array(pmap(xy))
        kept[variant] = pmap

    methods.run_group(
        "valis",
        cases,
        tmp_path / "out",
        {"workers": 1},
        label="valis120",
        on_variant=on_variant,
    )
    assert fv.CONFIG["mutate_in_place"]
    for variant in V120:
        assert np.array_equal(kept[variant](xy), seen[variant]), variant
    assert np.abs(seen["valis120"] - seen["valis120_micro"]).max() > 0.5
    assert np.abs(seen["valis120"] - seen["valis120_rigid"]).max() > 0.5


def test_a_stored_field_never_takes_valis_per_point_loop(tmp_path):
    """A tiled run stores its fields; the fake raises if a pyvips field reaches a point
    function, so ``ok`` here means every one was loaded as an array first."""
    rec, variants = selfcheck(tmp_path, stored=True, micro_stored=True)
    assert rec["status"] == sc.OK
    assert fv.calls("vips2numpy")


def test_hires_is_upstreams_example_under_its_own_names(tmp_path):
    smaller = (480, 600)  # the moving slide: the size comes from the SMALLEST slide
    rec, variants = selfcheck(
        tmp_path, "valis120_hires", {"hires": 1}, smaller, max_non_rigid=64
    )
    assert rec["status"] == sc.OK and rec["variants"] == dict.fromkeys(HIRES, sc.OK)
    (built,) = fv.calls("Valis")
    assert sorted(built["kwargs"]) == [
        "img_list",
        "micro_rigid_registrar_cls",
        "reference_img_f",
    ]
    assert built["kwargs"]["micro_rigid_registrar_cls"] == "MicroRigidRegistrar"
    size = math.floor(min(max(SMALL), max(smaller)) * 0.25)
    assert size == 150
    assert [c["kwargs"] for c in fv.calls("register_micro")] == [
        {"max_non_rigid_registration_dim_px": size}
    ]


def test_a_micro_pass_that_returns_early_says_so(tmp_path, pair):
    cases = patient(tmp_path, pair, rounds=1)
    out = tmp_path / "out"
    opts = {"workers": 1, "hires": 1}
    done = methods.run_group("valis", cases, out, opts, label="valis120_hires")
    assert done == {"P1_r1": list(HIRES)}
    run = run_json(out, "valis120_hires", cases[0])
    assert run["info"]["micro_ran"] is False
    assert "returned without registering" in run["problem"]
    assert run_json(out, "valis120_hires_nonrigid", cases[0])["problem"] == ""
    assert run["info"]["hires"] is True


@pytest.mark.parametrize(
    "label,opts", [("valis120_hires", {}), ("valis120", {"hires": 1})]
)
def test_the_hires_label_and_option_come_together(tmp_path, pair, label, opts):
    cases = patient(tmp_path, pair, rounds=1)
    out = tmp_path / "out"
    done = methods.run_group("valis", cases, out, {"workers": 1, **opts}, label=label)
    assert done == {"P1_r1": []}
    names = methods.variant_names("valis", label)
    run = run_json(out, names[-1], cases[0])
    assert run["ok"] is False and "must come together" in run["error"]
    assert not fv.calls("Valis")


def test_a_register_that_valis_swallowed_fails_the_patient(tmp_path, pair):
    fv.configure(fail_register=True)
    cases = patient(tmp_path, pair)
    done = methods.run_group("valis", cases, tmp_path / "o", {}, label="valis120")
    assert done == {"P1_r1": [], "P1_r2": []}
    run = run_json(tmp_path / "o", "valis120", cases[1])
    assert run["cause"] == "exception" and "returned None" in run["error"]
    assert "It reported: RuntimeError: the fake fails" in run["error"]
    assert fv.CALLS[-1]["call"] == "kill_jvm"


JAVA_OOM = "java.lang.OutOfMemoryError: Java heap space"
NUMPY_OOM = (
    "numpy.core._exceptions._ArrayMemoryError: Unable to allocate 64.0 GiB for an "
    "array with shape (92000, 93000) and data type float64"
)


@pytest.mark.parametrize(
    "text,cause",
    [
        (JAVA_OOM, "oom"),
        ("java.lang.OutOfMemoryError: GC overhead limit exceeded", "oom"),
        (NUMPY_OOM, "oom"),
        ("RuntimeError: DefaultCPUAllocator: not enough memory", "oom"),
        ("need at least one array to concatenate", "exception"),
    ],
)
def test_what_valis_swallowed_decides_the_cause(tmp_path, pair, text, cause):
    """VALIS prints its exception and returns None. A memory failure must come out as
    `oom` (the ladder climbs on nothing else), and any other as `exception`."""
    fv.configure(fail_register=text)
    cases = patient(tmp_path, pair)
    methods.run_group("valis", cases, tmp_path / "o", {}, label="valis120")
    for variant in methods.variant_names("valis", "valis120"):
        run = run_json(tmp_path / "o", variant, cases[1])
        assert run["ok"] is False and run["cause"] == cause, variant
        assert text in run["error"]
    assert fv.calls("print_warning"), "the original print_warning must still run"


def test_the_capture_leaves_valtils_as_it_was(tmp_path, pair):
    fv.configure(fail_register=JAVA_OOM)
    before = sys.modules["valis.valtils"].print_warning
    cases = patient(tmp_path, pair, rounds=1)
    methods.run_group("valis", cases, tmp_path / "o", {}, label="valis120")
    assert sys.modules["valis.valtils"].print_warning is before


def test_a_memory_error_in_the_micro_pass_of_valis_120_is_oom(tmp_path, pair):
    fv.configure(micro_raises=JAVA_OOM)
    cases = patient(tmp_path, pair, rounds=1)
    done = methods.run_group("valis", cases, tmp_path / "o", {}, label="valis120")
    assert done == {"P1_r1": ["valis120_rigid", "valis120"]}
    run = run_json(tmp_path / "o", "valis120_micro", cases[0])
    assert run["ok"] is False and run["cause"] == "oom"


@pytest.mark.parametrize(
    "exc, cause",
    [
        (ValueError("the crop probe is 2 px off"), "benchmark"),
        (MemoryError("Unable to allocate 3.1 GiB"), "benchmark_oom"),
    ],
)
def test_a_failed_capture_of_valis_120_is_the_benchmarks(
    tmp_path, pair, monkeypatch, exc, cause
):
    """F18: the benchmark's own capture (vp.snapshot: the crop probe, the field loaded
    whole) failed: not VALIS's failure."""

    def broken(*a, **kw):
        raise exc

    monkeypatch.setattr(vp, "snapshot", broken)
    cases = patient(tmp_path, pair, rounds=1)
    methods.run_group("valis", cases, tmp_path / "o", {}, label="valis120")
    for variant in ("valis120_rigid", "valis120", "valis120_micro"):
        run = run_json(tmp_path / "o", variant, cases[0])
        assert run["ok"] is False and run["cause"] == cause, variant
        assert "the benchmark's capture failed" in run["error"]


def test_valis_120_gets_a_heap_that_follows_the_rung(tmp_path, pair, monkeypatch):
    """VALIS 1.2.0 starts its JVM at 10 GB whatever it is asked; the adapter starts it
    first, through VALIS's own slide_io.init_jvm, with the module's formula."""
    want = {(16, 1): 12, (64, 2): 48, (256, 3): 80, (512, 4): 96, (700, 5): 96}
    for (mem, attempt), heap in want.items():
        fv.configure()
        monkeypatch.setenv("REGBENCH_MEM_GB", str(mem))
        monkeypatch.setenv("REGBENCH_ATTEMPT", str(attempt))
        (tmp_path / f"c{mem}").mkdir()
        cases = patient(tmp_path / f"c{mem}", pair, rounds=1)
        out = tmp_path / f"o{mem}"
        methods.run_group("valis", cases, out, {}, label="valis120")
        order = [c["call"] for c in fv.CALLS]
        assert [c["mem_gb"] for c in fv.calls("slide_io.init_jvm")] == [heap], mem
        assert order.index("slide_io.init_jvm") < order.index("Valis")
        info = run_json(out, "valis120", cases[0])["info"]
        assert info["jvm_heap_gb"] == heap and f"-Xmx{heap}G" in info["jvm_heap"]


def test_off_the_ladder_valis_120_keeps_its_own_heap(tmp_path, pair, monkeypatch):
    monkeypatch.delenv("REGBENCH_MEM_GB", raising=False)
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    cases = patient(tmp_path, pair, rounds=1)
    methods.run_group("valis", cases, tmp_path / "o", {}, label="valis120")
    assert not fv.calls("slide_io.init_jvm")
    info = run_json(tmp_path / "o", "valis120", cases[0])["info"]
    assert info["jvm_heap_gb"] is None and "10 GB" in info["jvm_heap"]


def test_a_failed_micro_pass_leaves_the_default_row(tmp_path, pair):
    fv.configure(micro_raises=True)
    cases = patient(tmp_path, pair, rounds=1)
    done = methods.run_group("valis", cases, tmp_path / "o", {}, label="valis120")
    assert done == {"P1_r1": ["valis120_rigid", "valis120"]}
    run = run_json(tmp_path / "o", "valis120_micro", cases[0])
    assert run["ok"] is False and "fake micro pass fails" in run["error"]
    assert run_json(tmp_path / "o", "valis120", cases[0])["native"]["files"]


def test_another_valis_version_is_refused(tmp_path, pair, monkeypatch):
    fv.install(monkeypatch.setitem, version="1.1.0")
    cases = patient(tmp_path, pair, rounds=1)
    methods.run_group("valis", cases, tmp_path / "o", {}, label="valis120")
    run = run_json(tmp_path / "o", "valis120", cases[0])
    assert "the installed valis is 1.1.0" in run["error"]
    assert not fv.calls("Valis")


@pytest.mark.parametrize(
    "opts,status", [({"crop_round": "trunc"}, sc.INCONSISTENT), ({}, sc.OK)]
)
def test_the_selfcheck_sees_a_wrong_crop_rule_through_the_adapter(
    tmp_path, opts, status
):
    rec, variants = selfcheck(tmp_path, opts=opts, rounding="ceil")
    assert rec["status"] == status
    assert rec["variants"] == dict.fromkeys(V120, status)


def test_valis_own_point_function_is_an_option_and_does_not_follow_the_image(tmp_path):
    rec, variants = selfcheck(tmp_path, opts={"points": "from_to"})
    assert rec["status"] == sc.INCONSISTENT
    assert fv.calls("warp_xy_from_to")
    last = fv.calls("warp_xy_from_to")[-1]
    assert len(last["keywords"]) == 12


# ── the clock ─────────────────────────────────────────────────────────────────
class Clock:
    """A monitor that writes its pauses into the fake's record of calls."""

    def pause(self):
        fv.record("pause")
        return self

    def resume(self):
        fv.record("resume")
        return self

    def snapshot(self):
        return {"wall_s": 0.0}


BENCHMARK_SIDE = ("warp_xy_inv", "warp_img", "get_bk_dxdy", "warp_and_save_slide")


def billed(calls):
    """The calls made while the clock was running."""
    paused, out = False, []
    for c in calls:
        if c["call"] in ("pause", "resume"):
            paused = c["call"] == "pause"
        elif not paused:
            out.append(c["call"])
    return out


def pipeline_stand_in():
    """The steps of ``bin/register.py::valis_registration`` that touch VALIS, in order,
    for the one test that needs the adapter's generator in THIS process."""
    registration = sys.modules["valis.registration"]

    def valis_registration(input_dir, out, reference, **kwargs):
        reg = registration.Valis(
            input_dir,
            os.path.dirname(out),
            reference_img_f=reference,
            align_to_reference=True,
            crop="reference",
        )
        reg.register()
        reg.register_micro(reference_img_f=reference, align_to_reference=True)
        for name, slide in reg.slide_dict.items():
            slide.warp_and_save_slide(
                src_f=slide.src_f,
                dst_f=os.path.join(out, f"{name}_registered.ome.tiff"),
                level=0,
                non_rigid=True,
                crop="reference",
                interp_method=kwargs["interp_method"],
            )
        registration.kill_jvm()
        return 0

    return types.SimpleNamespace(valis_registration=valis_registration)


@pytest.mark.parametrize("method", ["valis", "valis_mirage"])
def test_only_valis_own_work_is_billed(tmp_path, pair, monkeypatch, method):
    """Copying fields, the point path, the valid box and the writer all run with the
    clock paused; what is left running is VALIS registering."""
    cases = patient(tmp_path, pair)
    home = tmp_path / "native"
    opts = {
        "monitor": Clock(),
        "workers": 2,
        "stage_images": "1",
        "native_dir": lambda case_id, variant: home / variant / case_id,
    }
    if method == "valis":
        gen = valis120.register_group(cases, tmp_path / "work", opts)
    else:
        fv.install(monkeypatch.setitem, version="1.0.0", rounding="trunc")
        monkeypatch.setattr(valis_mirage, "pipeline", pipeline_stand_in)
        gen = valis_mirage.register_group(cases, tmp_path / "work", opts)
    seen = []
    for case_id, variant, pmap, info in gen:
        fv.record("pause")  # the driver, at the yield
        assert pmap is not None, info.get("error")
        pmap(np.array([[50.0, 60.0]]))
        assert Path(info["native"]["files"][0]).parent == home / variant / case_id
        seen.append((case_id, variant))
        fv.record("resume")
    assert len(seen) == 6
    ran = billed(fv.CALLS)
    assert not [c for c in ran if c in BENCHMARK_SIDE], ran
    assert "register" in ran and "register_micro" in ran
    order = [c["call"] for c in fv.CALLS]
    assert order.count("pause") == order.count("resume")


# ── VALIS 1.0.0 through the pipeline's own function ───────────────────────────
def function_signature():
    """``valis_registration``'s parameters and defaults, read from the source text."""
    tree = ast.parse((ROOT / "bin" / "register.py").read_text())
    (fn,) = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "valis_registration"
    ]
    names = [a.arg for a in fn.args.args]
    defaults = [ast.literal_eval(d) for d in fn.args.defaults]
    return names, dict(zip(names[len(names) - len(defaults) :], defaults))


def test_shipped_arguments_match_the_function_signature():
    names, defaults = function_signature()
    # every parameter is passed, by ARGS or per run, and nothing else is
    assert sorted(names) == sorted([*valis_mirage.ARGS, *valis_mirage.RUN_ARGS])
    assert not set(valis_mirage.ARGS) & set(valis_mirage.RUN_ARGS)
    # the shipped values are the function's own defaults ...
    for key, value in valis_mirage.ARGS.items():
        assert defaults[key] == value, key
    # ... and nextflow.config's, for the three the module passes from params
    config = (ROOT / "nextflow.config").read_text()
    for needle in (
        "memory_mode                = 'high'",
        "reg_micro_reg_fraction     = 0.125",
        "reg_max_image_dim          = 4000",
        "reg_micro_reg              = 2 ",
    ):
        assert needle in config, needle


def test_the_jvm_heap_follows_the_modules_formula():
    module = (ROOT / "modules" / "local" / "register.nf").read_text()
    assert "(32 + 16 * task.attempt)" in module
    assert "Math.min(heap_request, task.memory.toGiga() - 4)" in module
    heap = valis_mirage.jvm_heap_gb
    # F17: the attempt is the one the PIPELINE would be on with this memory (64 GB
    # doubling per attempt: conf/modules.config), whatever the ladder's rung
    config = (ROOT / "conf" / "modules.config").read_text()
    assert "64.GB * (2 ** (task.attempt - 1))" in config
    for mem, attempt, want in (
        (16, 1, 12),
        (64, 2, 48),  # the pipeline's first attempt at 64 GB: 32 + 16, not 60
        (128, 3, 64),
        (256, 3, 80),
        (512, 4, 96),
        (700, 5, 96),  # the pipeline's attempt 4 (512 GB) is the largest that fits
    ):
        host = {"mem_requested_gb": float(mem), "attempt": attempt}
        assert heap({}, host) == want, mem
        assert want <= mem - 4
    assert heap({}, {"mem_requested_gb": None, "attempt": 1}) is None
    assert heap({"jvm_heap_gb": "20"}, {"mem_requested_gb": 64.0, "attempt": 2}) == 20


def test_register_micro_wrapper_always_delegates():
    seen, state = [], {}

    def original(self, *args, **kwargs):
        seen.append((self, args, kwargs))
        return "micro", "errors"

    def broken(registrar, kwargs):
        raise RuntimeError("the snapshot failed")

    wrapped = valis_mirage.register_micro_wrapper(original, broken, state)
    out = wrapped("registrar", 1, tile_wh=2048)
    assert out == ("micro", "errors")
    assert seen == [("registrar", (1,), {"tile_wh": 2048})]  # once, arguments untouched
    assert "the snapshot failed" in state["capture_error"]
    assert "Traceback" in state["capture_traceback"]
    assert state["micro_called"] is True and state["micro_ran"] is True

    def failing(self, *args, **kwargs):
        raise ValueError("no SimpleElastix")

    state = {}
    wrapped = valis_mirage.register_micro_wrapper(failing, lambda r, k: None, state)
    with pytest.raises(ValueError, match="no SimpleElastix"):
        wrapped("registrar")  # the pipeline's own except sees its own error
    assert state["micro_called"] is True and "micro_ran" not in state
    assert "no SimpleElastix" in state["micro_error"]


def test_the_wrappers_are_removed_after_the_call():
    class Owner:
        def method(self):
            return "original"

    before = Owner.method
    with valis_mirage._replaced(Owner, "method", lambda f: lambda self: "wrapped"):
        assert Owner().method() == "wrapped"
        assert Owner.method.__name__ == "method"
    assert Owner.method is before
    with pytest.raises(RuntimeError):
        with valis_mirage._replaced(Owner, "method", lambda f: lambda self: "w"):
            raise RuntimeError("the call failed")
    assert Owner.method is before


def run_mirage(work, fake=None, opts=None, env=None, rounds=1):
    """``valis_mirage``'s selfcheck in a fresh process: the REAL ``bin/register.py`` on
    the fake VALIS. Returns the answer and the process."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    request = {
        "method": "valis_mirage",
        "work": str(work),
        "size": list(SMALL),
        "rounds": rounds,
        "opts": {"workers": 4, **(opts or {})},
        "fake": {"version": "1.0.0", "rounding": "trunc", **(fake or {})},
        "env": {"REGBENCH_MEM_GB": "64", "REGBENCH_ATTEMPT": "2", **(env or {})},
    }
    (work / "request.json").write_text(json.dumps(request))
    environ = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    environ["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    done = subprocess.run(
        [sys.executable, fv.__file__, str(work / "request.json")],
        cwd=ROOT,
        env=environ,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert (work / "answer.json").exists(), done.stdout[-3000:] + done.stderr[-3000:]
    return json.loads((work / "answer.json").read_text()), done


def mirage_run_json(work, variant):
    path = Path(work) / "out" / variant / "selfcheck" / "pair_512x640" / "run.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def mirage(tmp_path_factory):
    work = tmp_path_factory.mktemp("mirage")
    answer, done = run_mirage(work)
    return types.SimpleNamespace(
        work=work, answer=answer, calls=answer["calls"], log=done.stdout + done.stderr
    )


def named(calls, name):
    return [c for c in calls if c["call"] == name]


def test_valis_mirage_points_follow_the_images_the_pipeline_wrote(mirage):
    rec = mirage.answer["selfcheck"]
    assert rec["status"] == sc.OK, mirage.log[-3000:]
    assert rec["variants"] == dict.fromkeys(MIRAGE, sc.OK)
    for name, v in rec["pairs"][0]["variants"].items():
        assert v["err_px_p95"] < 0.25 and v["direction"] == REF_TO_MOV, name
    assert "REGISTRATION COMPLETED SUCCESSFULLY" in mirage.log


def test_valis_mirage_takes_a_patient_of_three_slides_in_one_run(tmp_path_factory):
    """The pipeline's function on one patient with two moving rounds, as on the cohort."""
    answer, done = run_mirage(tmp_path_factory.mktemp("mirage3"), rounds=2)
    rec = answer["selfcheck"]
    assert rec["status"] == sc.OK, (done.stdout + done.stderr)[-3000:]
    assert rec["variants"] == dict.fromkeys(MIRAGE, sc.OK)
    assert [p["case_id"] for p in rec["pairs"]] == ROUNDS
    for p in rec["pairs"]:
        for name, v in p["variants"].items():
            assert v["err_px_p95"] < 0.25 and v["direction"] == REF_TO_MOV, name
    assert len(named(answer["calls"], "Valis")) == 1
    assert len(named(answer["calls"], "register")) == 1


def test_the_pipelines_settings_reach_valis(mirage):
    (built,) = named(mirage.calls, "Valis")
    assert Path(built["src_dir"]).name == "preprocessed"
    assert built["dst_dir"] == str(Path(built["src_dir"]).parent)
    assert built["kwargs"] == {
        "reference_img_f": "reference.ome.tif",
        "align_to_reference": True,
        "crop": "reference",
        "max_processed_image_dim_px": 2048,
        "max_non_rigid_registration_dim_px": 2048,
        "max_image_dim_px": 4000,
        "feature_detector_cls": "SuperPointFD",
        "matcher": "SuperGlueMatcher",
        "non_rigid_registrar_cls": "OpticalFlowWarper",
        "affine_optimizer_cls": None,
        "micro_rigid_registrar_cls": "MicroRigidRegistrar",
        "create_masks": True,
    }
    assert [c["kwargs"] for c in named(mirage.calls, "register")] == [{}]
    assert [c["kwargs"] for c in named(mirage.calls, "register_micro")] == [
        {
            "max_non_rigid_registration_dim_px": math.floor(max(SMALL) * 0.125),
            "reference_img_f": "reference.ome.tif",
            "align_to_reference": True,
            "tile_wh": 2048,
        }
    ]
    # 64 GB at the ladder's second rung is the pipeline's FIRST attempt (F17):
    # min(32 + 16 * 1, 64 - 4)
    assert [c["mem_gb"] for c in named(mirage.calls, "init_jvm")] == [48]
    assert mirage.answer["cpu_count"] == 4  # what VALIS's pools were left with
    assert not named(mirage.calls, "get_inverse_field")


def test_the_pipelines_writer_drew_the_final_image_and_nothing_was_changed(mirage):
    final = [
        c
        for c in named(mirage.calls, "warp_and_save_slide")
        if c["dst_f"].endswith("_registered.ome.tiff")
    ]
    assert [c["slide"] for c in final] == ["moving", "reference"]
    for c in final:
        assert (c["level"], c["non_rigid"], c["crop"]) == (0, True, "reference")
        assert c["interp_method"] == "bicubic" and c["generation"] == 1
        assert Path(c["dst_f"]).parent.name == "registered_slides"
    stage = [
        c
        for c in named(mirage.calls, "warp_and_save_slide")
        if not c["dst_f"].endswith("_registered.ome.tiff")
    ]
    # the selfcheck's stage images: written before the micro pass, moving slide only
    assert [(c["non_rigid"], c["generation"], c["slide"]) for c in stage] == [
        (False, 0, "moving"),
        (True, 0, "moving"),
    ]
    order = [c["call"] for c in mirage.calls]
    assert order.index("register_micro") > max(
        i
        for i, c in enumerate(mirage.calls)
        if c["call"] == "warp_and_save_slide" and c["generation"] == 0
    )
    assert order[-1] == "kill_jvm" or "kill_jvm" in order


def test_valis_mirage_records_what_ran(mirage):
    run = mirage_run_json(mirage.work, "valis_mirage")
    info = run["info"]
    assert info["version"] == "1.0.0" and info["points"] == "image"
    assert info["ARGS"] == valis_mirage.ARGS
    assert info["run_args"]["jvm_heap_gb"] == 48 and info["run_args"]["n_cpus"] == 4
    assert info["run_args"]["reference"] == "reference.ome.tif"
    assert info["workers_seen"] == 4 and info["cpus"]["valis_pool_threads"] == 3
    assert info["micro_called"] and info["micro_ran"] and not info["repair_fired"]
    assert info["slides_written"] == ["moving", "reference"]
    assert info["slides_placed"] == {
        "reference.ome.tif": "hard link",
        "moving.ome.tif": "hard link",
    }
    # placing the slides is the benchmark's own arrangement: timed, clock paused
    assert 0 <= info["place_s"] < run["register_s"]
    assert info["cost_after_register"]["wall_s"] <= run["register_s"]
    assert info["nuclear_channel"]["moving"] == {"index": 0, "name": "DAPI"}
    assert info["stage"]["crop_round"] == "probe" and run["problem"] == ""
    native = run["native"]
    (path,) = native["files"]
    assert Path(path).name == "moving_registered.ome.tiff"
    assert Path(path).parent.name == "native" and Path(path).exists()
    assert "origin_xy" not in native and len(native["valid_box"]) == 4
    assert native["writer_kwargs"]["crop"] == "reference" and run["write_s"] > 0
    assert run["direction"] == REF_TO_MOV and run["workers"] == 4
    # the earlier stages carry the same matrix and their own fields
    rigid = mirage_run_json(mirage.work, "valis_mirage_rigid")["info"]["stage"]
    first = mirage_run_json(mirage.work, "valis_mirage_nonrigid")["info"]["stage"]
    assert rigid["field_shape"] is None and first["field_shape"] is not None
    assert first["field_shape"] != info["stage"]["field_shape"]


def test_a_micro_pass_the_pipeline_swallowed_is_reported(tmp_path):
    """The pipeline logs a failed micro pass and carries on; the row must say so, and its
    points must still follow the image that was then written."""
    answer, done = run_mirage(tmp_path / "w", fake={"micro_raises": True})
    rec = answer["selfcheck"]
    assert rec["variants"] == dict.fromkeys(MIRAGE, sc.OK), done.stdout[-2000:]
    run = mirage_run_json(tmp_path / "w", "valis_mirage")
    assert run["info"]["micro_called"] and not run["info"]["micro_ran"]
    assert "the micro pass did not run" in run["problem"]
    assert "fake micro pass fails" in run["problem"]
    assert mirage_run_json(tmp_path / "w", "valis_mirage_nonrigid")["problem"] == ""


def test_a_register_the_pipelines_valis_swallowed_keeps_its_cause(tmp_path):
    """Valis.register() swallows, bin/register.py raises a fixed text: the memory cause
    VALIS printed must still reach run.json, or the pipeline's VALIS never climbs."""
    _, done = run_mirage(tmp_path / "w", fake={"fail_register": JAVA_OOM})
    for variant in MIRAGE:
        run = mirage_run_json(tmp_path / "w", variant)
        assert run["ok"] is False and run["cause"] == "oom", done.stdout[-2000:]
        assert JAVA_OOM in run["error"] and "swallowed" in run["error"]


def test_another_swallowed_register_of_the_pipeline_stays_an_exception(tmp_path):
    text = "need at least one array to concatenate"
    run_mirage(tmp_path / "w", fake={"fail_register": text})
    run = mirage_run_json(tmp_path / "w", "valis_mirage")
    assert run["cause"] == "exception" and text in run["error"]


def test_a_micro_pass_that_ran_out_of_memory_fails_the_full_row(tmp_path):
    """bin/register.py catches a Java out-of-memory inside the micro pass and writes the
    non-rigid result. That is not the full method: the row fails with `oom`, so the
    patient goes up a rung, and the non-rigid row keeps its own result."""
    run_mirage(tmp_path / "w", fake={"micro_raises": JAVA_OOM})
    final = mirage_run_json(tmp_path / "w", "valis_mirage")
    assert final["ok"] is False and final["cause"] == "oom"
    assert (
        "micro pass ran out of memory" in final["error"] and JAVA_OOM in final["error"]
    )
    assert mirage_run_json(tmp_path / "w", "valis_mirage_nonrigid")["ok"] is True
    assert mirage_run_json(tmp_path / "w", "valis_mirage_rigid")["ok"] is True


def test_a_slide_the_pipeline_did_not_write_fails_its_round(tmp_path):
    """F15: the pipeline's writer failed for the slide (it catches that per slide and
    goes on): the row is the writer's failure, with the writer's own error."""
    answer, done = run_mirage(tmp_path / "w", fake={"skip_write": ["moving"]})
    rec = answer["selfcheck"]
    assert rec["status"] == sc.CRASHED
    for variant in MIRAGE:
        run = mirage_run_json(tmp_path / "w", variant)
        assert run["ok"] is False and run["cause"] == "writer", variant
        assert "the fake writer refuses moving" in run["error"]
        assert "the pipeline's writer failed for moving" in run["error"]


@pytest.mark.parametrize(
    "fake",
    [
        {"write_error": "MemoryError"},
        # VALIS 1.0.0's tile worker: the Java heap error printed, another raised (F16)
        {"printed": JAVA_OOM},
    ],
)
def test_a_writer_that_ran_out_of_memory_on_a_slide_climbs(tmp_path, fake):
    """F15, F16: a memory failure of the pipeline's writer is writer_oom, the unit goes
    up a rung."""
    run_mirage(tmp_path / "w", fake={"skip_write": ["moving"], **fake})
    run = mirage_run_json(tmp_path / "w", "valis_mirage")
    assert run["ok"] is False and run["cause"] == "writer_oom"


def test_a_java_heap_error_valis_printed_inside_register_is_oom(tmp_path):
    """F16: VALIS 1.0.0 hands print_warning only the UnboundLocalError; the Java heap
    error went to stdout. The row is oom, as the module docstring says."""
    unbound = "local variable 'tile' referenced before assignment"
    _, done = run_mirage(
        tmp_path / "w", fake={"fail_register": unbound, "printed": JAVA_OOM}
    )
    run = mirage_run_json(tmp_path / "w", "valis_mirage")
    assert run["ok"] is False and run["cause"] == "oom", done.stdout[-2000:]
    assert JAVA_OOM in run["error"] and unbound in run["error"]
    assert JAVA_OOM in done.stdout  # still printed: the tee passes everything on


def test_a_micro_pass_whose_memory_error_valis_printed_fails_the_full_row(tmp_path):
    """F16: the micro pass raised something else; the Java heap error was printed."""
    run_mirage(
        tmp_path / "w",
        fake={"micro_raises": "UnboundLocalError: tile", "printed": JAVA_OOM},
    )
    final = mirage_run_json(tmp_path / "w", "valis_mirage")
    assert final["ok"] is False and final["cause"] == "oom"


def test_a_capture_of_the_benchmark_that_failed_is_the_benchmarks(monkeypatch):
    """F18: the final transform was not captured because the benchmark's own probe or
    field load failed: benchmark / benchmark_oom, never the method's exception."""
    state = {
        "errors": {"m": ["ValueError: probe off by 2 px"]},
        "capture_causes": {"m": ["exception"]},
        "write_errors": {},
    }
    written = {"path": "x.ome.tiff"}
    got = valis_mirage._not_written({}, state, "m", written)
    assert got["cause"] == "benchmark" and "not captured" in got["error"]
    state["capture_causes"]["m"] = ["oom"]
    assert (
        valis_mirage._not_written({}, state, "m", written)["cause"] == "benchmark_oom"
    )
    # a slide the function never wrote, with no writer error: the method's
    assert "cause" not in valis_mirage._not_written({}, {"errors": {}}, "m", None)


def test_printed_memory_errors_are_kept_and_everything_passes_on(capsys):
    """F16: the tee keeps the chunks that name memory, bounded, and prints all."""
    from benchmarks.regbench.methods import _valis_points as vp

    with vp.printed_errors(most=2) as kept:
        print("reading tile 3")
        print(JAVA_OOM)
        print("Unable to allocate 4.0 GiB")
        print("MemoryError")
    assert kept == [JAVA_OOM, "Unable to allocate 4.0 GiB"]
    assert "reading tile 3" in capsys.readouterr().out


def test_placing_the_slides_is_outside_the_clock_for_both_adapters(
    tmp_path, pair, monkeypatch
):
    """A copy of every slide (two file systems) must not be billed as VALIS's time."""
    import inspect
    import time

    from benchmarks.regbench import cost, resources

    def slow(paths, dst_dir):
        time.sleep(0.6)
        return {str(p): str(p) for p in paths}, {Path(p).name: "copy" for p in paths}

    monkeypatch.setattr(vp, "place_slides", slow)
    mon = resources.Monitor()
    mon.start()
    placed, how, took = vp.place_slides_unbilled(
        [pair["moving"]], tmp_path / "in", {"monitor": mon}
    )
    assert not mon.paused  # running again for the method
    billed = mon.snapshot()["wall_s"]
    mon.stop()
    assert took >= 0.6 and billed < 0.3 and how == {"moving.ome.tif": "copy"}
    # without the driver's monitor (a call outside a run) it still times the step
    assert vp.place_slides_unbilled([pair["moving"]], tmp_path / "in", {})[2] >= 0.6
    for adapter in (valis120, valis_mirage):
        source = inspect.getsource(adapter)
        assert "vp.place_slides_unbilled(" in source and "place_s" in source
        assert "vp.place_slides(" not in source
    for name in ("valis", "valis_mirage"):
        assert "info.place_s" in cost.CLOCK[name]["reading"]


# ── a brightfield case: VALIS's own processing, recorded (methods.NO_BRIGHTFIELD) ──
@pytest.mark.parametrize("module", [valis120, valis_mirage])
def test_a_brightfield_case_is_accepted_and_valis_own_reading_recorded(
    tmp_path, pair, module, monkeypatch
):
    """Both adapters have an input rule for a brightfield case (VALIS chooses its own
    processing from its reader's is_rgb). Here the slides are the fluorescence pair
    labelled brightfield, which VALIS reads as fluorescence: every row still runs, and
    says so (``problem``); no channel is looked for (``nuclear_channel`` None)."""
    import dataclasses

    cases = [
        dataclasses.replace(c, modality="brightfield") for c in patient(tmp_path, pair)
    ]
    if module is valis_mirage:
        fv.install(monkeypatch.setitem, version="1.0.0", rounding="trunc")
        monkeypatch.setattr(valis_mirage, "pipeline", pipeline_stand_in)
    rows = list(module.register_group(cases, tmp_path / "work", {"workers": 2}))
    assert len(rows) == 6 and all(pm is not None for _, _, pm, _ in rows)
    for _, _, _, info in rows:
        assert info["modality"] == "brightfield" and info["nuclear_channel"] is None
        assert set(info["valis_img_type"].values()) == {"fluorescence"}
        assert "not brightfield" in info["problem"]


# ── the pipeline's three tiers x micro-registration depth 0, 1, 2 ─────────────
# the shipped call, written out: nextflow.config's defaults as modules/local/register.nf
# passes them. `valis_mirage` (high, depth 2) is made with exactly this.
SHIPPED_ARGS = {
    "reference_markers": None,
    "memory_mode": "high",
    "max_processed_dim": None,
    "max_non_rigid_dim": None,
    "micro_reg_fraction": 0.125,
    "max_image_dim_px": 4000,
    "micro_reg": 2,
    "interp_method": "bicubic",
    "stage_checkpoint_dir": None,
    "max_keypoints": None,
}


def test_the_shipped_call_is_unchanged():
    assert valis_mirage.call_args({}) == valis_mirage.ARGS == SHIPPED_ARGS
    assert list(valis_mirage.call_args({})) == list(SHIPPED_ARGS)
    # the primary's own label and options name the same call
    assert valis_mirage.call_args({"label": "valis_mirage"}) == SHIPPED_ARGS
    assert valis_mirage.call_args({"label": ""}) == SHIPPED_ARGS


def _preset_rows():
    """bin/utils/valis_config.py ``MEMORY_PRESETS``, read from the source without
    importing VALIS: ``{tier: {key: the value's source text}}``."""
    import re

    src = (ROOT / "bin" / "utils" / "valis_config.py").read_text()
    table = src[src.index("MEMORY_PRESETS = {") :]
    table = table[: table.index("\n}\n")]
    rows = {}
    for tier in ("high", "medium", "low"):
        body = table[table.index(f'"{tier}": {{') :]
        body = body[: body.index("\n    }")]
        rows[tier] = dict(re.findall(r'^\s+"(\w+)": (.+?),$', body, re.M))
    return rows


# the preset keys a row may differ in: the two sizes, and two keys nothing reads
SIZE_KEYS = ("max_processed_image_dim_px", "max_non_rigid_registration_dim_px")
DEAD_KEYS = ("tile_wh", "tile_buffer")  # nextflow.config:241-244


def test_the_tiers_and_depths_are_the_pipelines():
    assert valis_mirage.TIERS == ("high", "medium", "low")
    assert valis_mirage.MICRO == (0, 1, 2)
    rows = _preset_rows()
    assert set(rows) == set(valis_mirage.TIERS)
    assert {t: tuple(int(r[k]) for k in SIZE_KEYS) for t, r in rows.items()} == {
        "high": (2048, 2048),
        "medium": (1024, 1024),
        "low": (512, 512),
    }
    # the rows differ in the two sizes and in two keys no code reads, nothing else
    keys = set().union(*rows.values())
    differ = {k for k in keys if len({r.get(k) for r in rows.values()}) > 1}
    assert differ == {*SIZE_KEYS, *DEAD_KEYS}
    assert "max_image_dim_px" not in keys  # so the call's 4000 stands at every tier
    import re

    read = set()
    for path in ("bin/register.py", "bin/utils/valis_config.py"):
        text = (ROOT / path).read_text()
        read |= set(re.findall(r'preset\[\s*"(\w+)"\s*\]', text))
        read |= set(re.findall(r'preset\.get\(\s*"(\w+)"', text))
    assert read & differ == set(SIZE_KEYS)
    # modules/local/register.nf: of every param its script: block reads, memory_mode
    # is the only one that names a tier. The three tier-owned overrides are null by
    # default (rendered only when set), so at a named tier the rendered command
    # differs in --memory-mode alone.
    nf = (ROOT / "modules" / "local" / "register.nf").read_text()
    script = nf[nf.index("script:") : nf.index("stub:")]
    read_nf = set(re.findall(r"params\.(\w+)", script))
    assert read_nf == {
        "memory_mode",
        "reg_jvm_heap_gb",
        "reg_max_image_dim",
        "reg_micro_reg",
        "reg_micro_reg_fraction",
        "reg_qc",
        "reg_valis_max_keypoints",
        "reg_valis_max_non_rigid_dim",
        "reg_valis_max_processed_dim",
        "skip_registration_qc",
    }
    config = (ROOT / "nextflow.config").read_text()

    def default(name):
        (value,) = re.findall(rf"^\s+{name}\s+=\s+(\S+)", config, re.M)
        return value

    assert default("memory_mode") == "'high'"
    for name in (
        "reg_valis_max_processed_dim",
        "reg_valis_max_non_rigid_dim",
        "reg_valis_max_keypoints",
        "reg_jvm_heap_gb",
    ):
        assert default(name) == "null", name
    assert (default("reg_max_image_dim"), default("reg_micro_reg_fraction")) == (
        "4000",
        "0.125",
    )
    assert "--memory-mode ${memory_mode}" in script
    assert "--max-image-dim ${max_image_dim}" in script
    assert "--micro-reg-fraction ${micro_reg_fraction}" in script


@pytest.mark.parametrize("tier", ["high", "medium", "low"])
@pytest.mark.parametrize("micro", [0, 1, 2])
def test_a_tier_and_depth_reach_the_pipelines_function(tier, micro):
    got = valis_mirage.call_args({"memory_mode": tier, "micro_reg": str(micro)})
    assert got["memory_mode"] == tier and got["micro_reg"] == micro
    assert isinstance(got["micro_reg"], int)
    assert {k: v for k, v in got.items() if k not in ("memory_mode", "micro_reg")} == {
        k: v for k, v in SHIPPED_ARGS.items() if k not in ("memory_mode", "micro_reg")
    }
    # its own label agrees with it
    label = (
        "valis_mirage"
        if (tier, micro) == ("high", 2)
        else f"valis_mirage_{tier}_m{micro}"
    )
    assert (
        valis_mirage.call_args(
            {"memory_mode": tier, "micro_reg": str(micro), "label": label}
        )
        == got
    )


def test_other_values_are_refused():
    with pytest.raises(ValueError):
        valis_mirage.call_args({"memory_mode": "custom"})
    with pytest.raises(ValueError):
        valis_mirage.call_args({"micro_reg": "3"})
    with pytest.raises(ValueError):
        valis_mirage.call_args({"micro_reg": "-1"})


def test_a_label_naming_another_tier_or_depth_is_refused():
    with pytest.raises(ValueError, match="label"):
        valis_mirage.call_args({"label": "valis_mirage_low_m0"})
    with pytest.raises(ValueError, match="label"):
        valis_mirage.call_args(
            {"memory_mode": "low", "micro_reg": "1", "label": "valis_mirage_low_m0"}
        )
    with pytest.raises(ValueError, match="label"):
        valis_mirage.call_args({"memory_mode": "low", "label": "valis_mirage"})


@pytest.mark.parametrize(
    "label", ["valis_mirage_try2", "valis_mirage_high", "valis_mirage_low_m9", "mine"]
)
def test_a_free_label_runs_the_shipped_call_or_its_options(label):
    """Only a label of the grid's form (``valis_mirage_<tier>_m<0-2>``, or the primary's
    own ``valis_mirage``) names a cell; any other label is free."""
    assert valis_mirage.call_args({"label": label}) == SHIPPED_ARGS
    got = valis_mirage.call_args(
        {"label": label, "memory_mode": "low", "micro_reg": "0"}
    )
    assert (got["memory_mode"], got["micro_reg"]) == ("low", 0)


def test_each_depth_names_the_stages_it_has():
    """R18: ``valis_registration``'s docstring (bin/register.py:253-257 when written).
    Depth 1 is micro-RIGID only; a non-rigid `micro`
    stage exists only at depth 2."""
    assert valis_mirage.STAGES_AT_DEPTH == {
        0: ("rigid", "nonrigid"),
        1: ("rigid", "micro_rigid", "nonrigid"),
        2: ("rigid", "micro_rigid", "nonrigid", "micro"),
    }
    import ast

    tree = ast.parse((ROOT / "bin" / "register.py").read_text())
    (fn,) = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "valis_registration"
    ]
    doc = " ".join(ast.get_docstring(fn).split())
    said = doc[doc.index("micro_reg : int") :]
    said = said[: said.index("stage_checkpoint_dir")]
    assert "0 = neither micro pass" in said
    assert "1 = micro-rigid only" in said
    assert "2 = also the micro non-rigid pass" in said


def test_a_low_tier_at_depth_0_reaches_valis_and_has_no_micro_pass(tmp_path):
    """The real bin/register.py on the stand-in VALIS, at memory_mode=low, micro_reg=0:
    the low row's sizes reach Valis, no micro-rigid registrar, no register_micro, and
    the final row is the non-rigid row (with no problem: depth 0 has no micro pass)."""
    answer, done = run_mirage(
        tmp_path / "w", opts={"memory_mode": "low", "micro_reg": "0"}
    )
    log = (done.stdout + done.stderr)[-3000:]
    (built,) = named(answer["calls"], "Valis")
    assert built["kwargs"]["max_processed_image_dim_px"] == 512, log
    assert built["kwargs"]["max_non_rigid_registration_dim_px"] == 512
    assert built["kwargs"]["max_image_dim_px"] == 4000
    assert built["kwargs"]["micro_rigid_registrar_cls"] is None
    assert not named(answer["calls"], "register_micro")
    final = mirage_run_json(tmp_path / "w", "valis_mirage")
    nonrigid = mirage_run_json(tmp_path / "w", "valis_mirage_nonrigid")
    rigid = mirage_run_json(tmp_path / "w", "valis_mirage_rigid")
    info = final["info"]
    assert info["ARGS"]["memory_mode"] == "low" and info["ARGS"]["micro_reg"] == 0
    assert info["ARGS"] == {**SHIPPED_ARGS, "memory_mode": "low", "micro_reg": 0}
    assert info["implementation"]["tier"] == "low"
    assert info["implementation"]["micro_reg"] == 0
    assert info["implementation"]["stages"] == ["rigid", "nonrigid"]
    assert "memory_mode=low, micro_reg=0" in info["implementation"]["parameters"]
    assert not info["micro_called"] and not info["micro_ran"]
    assert final["ok"] and nonrigid["ok"] and rigid["ok"]
    assert final["problem"] == "" and nonrigid["problem"] == ""
    assert final["info"]["stage"] == nonrigid["info"]["stage"]
    assert rigid["info"]["stage"]["field_shape"] is None
    assert nonrigid["info"]["stage"]["field_shape"] is not None


def test_depth_1_has_the_micro_rigid_registrar_and_no_micro_pass(tmp_path):
    answer, done = run_mirage(
        tmp_path / "w", opts={"memory_mode": "medium", "micro_reg": "1"}
    )
    (built,) = named(answer["calls"], "Valis")
    assert built["kwargs"]["max_processed_image_dim_px"] == 1024, done.stdout[-2000:]
    assert built["kwargs"]["micro_rigid_registrar_cls"] == "MicroRigidRegistrar"
    assert not named(answer["calls"], "register_micro")
    final = mirage_run_json(tmp_path / "w", "valis_mirage")
    assert final["info"]["implementation"]["stages"] == [
        "rigid",
        "micro_rigid",
        "nonrigid",
    ]
    assert final["problem"] == ""
    assert (
        final["info"]["stage"]
        == mirage_run_json(tmp_path / "w", "valis_mirage_nonrigid")["info"]["stage"]
    )


def test_the_shipped_run_records_its_four_stages(mirage):
    impl = mirage_run_json(mirage.work, "valis_mirage")["info"]["implementation"]
    assert impl["tier"] == "high" and impl["micro_reg"] == 2
    assert impl["stages"] == ["rigid", "micro_rigid", "nonrigid", "micro"]


def test_the_jvm_heap_at_the_fifth_rung():
    from benchmarks.regbench.methods import _valis_points as vp

    assert vp.jvm_heap_gb({}, {"mem_requested_gb": 700, "attempt": 5}) == 96
