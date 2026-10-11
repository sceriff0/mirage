"""The DeeperHistReg adapter, without DeeperHistReg.

DeeperHistReg (pinned at 1.0.1), SimpleITK and pyvips are not installed where this suite
runs, so everything here runs against ``fake_dhr``: a recording stand-in that keeps the
package's bookkeeping and draws the warped image the way its writer is written to (see
that module). What these tests pin:

* the call is the package's own, in its native roles, with a shipped parameter set in
  which only the device keys and ``save_final_images`` (and, for ``fluo``, two keys) differ;
* the two point maps: the estimated field through the issue-#12 chain, and the image as
  written; the as-drawn map against the emulated writer at S = 5, 9.77 and 6.10 and on
  sizes that do not divide; that a wrong direction, a dropped pad origin, a wrong scale,
  a missing writer offset and a wrong chain are each reported ``inconsistent``;
* a swallowed failure is raised with what the package printed, and classified;
* what a run records: the method (RegWSI), the factory, the working resolution.

What they cannot pin is DeeperHistReg and libvips themselves: ``test_external_methods.py``
holds the tests that need the real package, skipped here.
"""

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from benchmarks.regbench import METHODS, cost, imageio, methods
from benchmarks.regbench import selfcheck as sc
from benchmarks.regbench.cases import result_dir
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap
from benchmarks.regbench.methods import deeperhistreg as dhr

from . import fake_dhr as fd

HR = "default_initial_nonrigid_high_resolution"
# With the fake's working short side set to 104 px: loaded 208 x 256, k = 2, a field of
# 104 x 128 and S = 10 on both axes; every division is exact, and the pads are unequal.
EXACT = [((1040, 1280), (1000, 1200))]
EXACT_RESOLUTION = 104
LABELS = {
    "deeperhistreg_gpu": {"device": "cuda:0"},
    "deeperhistreg_cpu": {"device": "cpu"},
    "deeperhistreg_hr_gpu": {"device": "cuda:0", "config": HR},
    "deeperhistreg_hr_cpu": {"device": "cpu", "config": HR},
    "deeperhistreg_fluo_gpu": {"device": "cuda:0", "fluo": 1},
    "deeperhistreg_fluo_cpu": {"device": "cpu", "fluo": 1},
    "deeperhistreg_hr_fluo_gpu": {"device": "cuda:0", "config": HR, "fluo": 1},
    "deeperhistreg_hr_fluo_cpu": {"device": "cpu", "config": HR, "fluo": 1},
}


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    """The fake ``deeperhistreg`` (and ``torch``, ``SimpleITK``) for one test."""
    fd.install(monkeypatch.setitem, initial_resolution=EXACT_RESOLUTION)
    yield fd
    fd.configure()
    dhr._LIVE.clear()


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    return sc.make_pair(tmp_path_factory.mktemp("dhr_pair"), *EXACT[0])


@pytest.fixture
def case(pair, tmp_path):
    return sc.pair_case(pair, tmp_path / "cases")


def run_json(out, variant, case):
    return json.loads((result_dir(out, variant, case) / "run.json").read_text())


def _selfcheck(tmp_path, **opts):
    rec = sc.run("deeperhistreg", tmp_path, opts=opts or None, sizes=EXACT)
    return rec, rec["pairs"][0]["variants"]


# ── names and settings ────────────────────────────────────────────────────────
def test_contract():
    assert "deeperhistreg" in METHODS and dhr.SCOPE == "pair"
    assert dhr.VARIANTS == ("deeperhistreg", "deeperhistreg_asdrawn")
    assert methods.scope_of("deeperhistreg") == "pair"
    assert methods.variant_names("deeperhistreg", "deeperhistreg_gpu") == [
        "deeperhistreg_gpu",
        "deeperhistreg_gpu_asdrawn",
    ]
    assert methods.variant_names("deeperhistreg", "deeperhistreg_hr_fluo_cpu") == [
        "deeperhistreg_hr_fluo_cpu",
        "deeperhistreg_hr_fluo_cpu_asdrawn",
    ]
    # both rows are the method's own direction: no inverse anywhere in the module
    assert dhr.DIRECTION == REF_TO_MOV and not hasattr(dhr, "inverse_warper")


def test_imports_without_torch():
    code = (
        "import sys\n"
        "from benchmarks.regbench.methods import deeperhistreg\n"
        "bad = [m for m in ('torch', 'deeperhistreg', 'SimpleITK', 'pyvips')"
        " if m in sys.modules]\n"
        "raise SystemExit(int(bool(bad)))\n"
    )
    root = Path(__file__).resolve().parents[3]
    assert subprocess.run([sys.executable, "-c", code], cwd=root).returncode == 0


@pytest.mark.parametrize("label", sorted(LABELS))
def test_every_row_of_the_run_list_is_accepted(label):
    opts = {**LABELS[label], "label": label}
    config, device, fluo = dhr.settings(opts)
    assert config == opts.get("config", "default_initial_nonrigid")
    assert device == opts["device"] and fluo == bool(opts.get("fluo"))


@pytest.mark.parametrize(
    "opts",
    [
        {"label": "deeperhistreg_gpu", "device": "cpu"},
        {"label": "deeperhistreg_cpu", "device": "cuda:0"},
        {"label": "deeperhistreg_hr_cpu", "device": "cpu"},
        {"label": "deeperhistreg_cpu", "device": "cpu", "config": HR},
        {"label": "deeperhistreg_fluo_cpu", "device": "cpu"},
        {"label": "deeperhistreg_cpu", "device": "cpu", "fluo": 1},
    ],
)
def test_a_label_and_its_options_come_together(opts):
    with pytest.raises(ValueError, match="come together"):
        dhr.settings(opts)


def test_unknown_settings_are_refused():
    with pytest.raises(ValueError, match="config="):
        dhr.settings({"config": "default_nonrigid"})
    with pytest.raises(ValueError, match="device="):
        dhr.settings({"device": "gpu"})
    assert dhr.settings({}) == ("default_initial_nonrigid", "cpu", False)


# ── parameters ────────────────────────────────────────────────────────────────
def _fixture_params():
    return {
        "device": "cuda:0",
        "save_final_images": True,
        "loading_params": {
            "loader": "tiff",
            "source_resample_ratio": 0.2,
            "target_resample_ratio": 0.2,
            "pad_value": 255.0,
        },
        "preprocessing_params": {"flip_intensity": True, "clahe": True},
        "initial_registration_params": {"device": "cuda:0", "cuda": True},
        "nonrigid_registration_params": {"device": "cuda:0"},
    }


def _without(d, dotted):
    d = copy.deepcopy(d)
    for key in dotted:
        node, *rest = key.split(".")
        target = d
        while rest:
            target, node = target[node], rest.pop(0)
        del target[node]
    return d


def test_changed_keys_default():
    params = _fixture_params()
    keys = dhr.changed_keys(params, "cpu", False)
    assert keys == [
        "device",
        "initial_registration_params.cuda",
        "initial_registration_params.device",
        "nonrigid_registration_params.device",
        "save_final_images",
    ]
    assert params["device"] == "cpu" and params["save_final_images"] is False
    assert params["initial_registration_params"] == {"device": "cpu", "cuda": False}
    assert params["nonrigid_registration_params"] == {"device": "cpu"}
    assert _without(params, keys) == _without(_fixture_params(), keys)


def test_changed_keys_fluo():
    params = _fixture_params()
    keys = dhr.changed_keys(params, "cuda:0", True)
    assert set(keys) - set(dhr.changed_keys(_fixture_params(), "cuda:0", False)) == {
        "loading_params.pad_value",
        "preprocessing_params.flip_intensity",
    }
    assert params["loading_params"]["pad_value"] == 0.0
    assert params["preprocessing_params"]["flip_intensity"] is False
    assert params["initial_registration_params"]["cuda"] is True
    assert _without(params, keys) == _without(_fixture_params(), keys)


def check_factories(configs):
    """What the labels claim about the two shipped sets (the RegWSI report's C5)."""
    hr = getattr(configs, HR)()
    assert hr["preprocessing_params"]["initial_resolution"] == 8192
    nonrigid = hr["nonrigid_registration_params"]
    assert nonrigid["registration_size"] == 8192
    assert nonrigid["num_levels"] == nonrigid["used_levels"] == 9
    assert sum(nonrigid["iterations"]) == 1100
    assert nonrigid["cost_function"] == "ncc_local"
    assert nonrigid["regularization_function"] == "diffusion_relative"
    default = configs.default_initial_nonrigid()
    assert default["preprocessing_params"]["initial_resolution"] == 4096
    nonrigid = default["nonrigid_registration_params"]
    assert nonrigid["registration_size"] == 4096
    assert nonrigid["num_levels"] == nonrigid["used_levels"] == 8
    assert sum(nonrigid["iterations"]) == 900
    for params in (hr, default):
        assert params["loading_params"]["source_resample_ratio"] == 0.2
        assert params["loading_params"]["target_resample_ratio"] == 0.2
        assert params["loading_params"]["pad_value"] == 255.0
        assert params["preprocessing_params"]["flip_intensity"] is True
        # the departures the run records are the shipped values
        assert params["initial_registration_params"]["run_sift_ransac"] is True
        assert params["initial_registration_params"]["transform_type"] == "rigid"
    assert len(hr["initial_registration_params"]["registration_sizes"]) == 9
    assert len(default["initial_registration_params"]["registration_sizes"]) == 8
    assert {
        k: v["preprocessing_params"]["initial_resolution"]
        for k, v in (("default_initial_nonrigid", default), (HR, hr))
    } == dhr.CONFIGS


def check_params_equal_the_factory(configs):
    for config in dhr.CONFIGS:
        for device, fluo in (("cpu", False), ("cuda:0", True)):
            params = getattr(configs, config)()
            keys = dhr.changed_keys(params, device, fluo)
            assert _without(params, keys) == _without(getattr(configs, config)(), keys)


def test_the_transcribed_factories_are_the_two_configurations():
    """The stand-in's transcription; ``test_external_methods.py`` runs the same
    assertions on the real package."""
    check_factories(fd.modules()["deeperhistreg"].configs)
    check_params_equal_the_factory(fd.modules()["deeperhistreg"].configs)


# ── inputs ────────────────────────────────────────────────────────────────────
def test_converted_page_equals_the_channel(tmp_path):
    rng = np.random.default_rng(3)
    planes = rng.integers(100, 4000, (2, 700, 900)).astype(np.uint16)
    planes[1, 4, 8] = 65535  # on the lattice the levels are counted on
    src = imageio.write_channels(tmp_path / "slide.ome.tif", planes, ("DAPI", "CD3"))
    meta = dhr.single_page_tiff(src, 1, tmp_path / "one.tif")
    with tifffile.TiffFile(tmp_path / "one.tif") as tif:
        assert len(tif.pages) == 1 and len(tif.series) == 1
        page = tif.pages[0]
        assert page.subifds is None and page.is_tiled and page.compression == 1
        got = page.asarray()
    assert got.dtype == np.uint16 and np.array_equal(got, planes[1])
    assert meta["hw"] == [700, 900] and meta["dtype"] == "uint16"
    # one hot pixel stretches the min-max range: the channel keeps few 8-bit levels
    assert 1 < meta["levels"] <= 17
    with pytest.raises(ValueError, match="out of range"):
        dhr.single_page_tiff(src, 2, tmp_path / "none.tif")


def test_grey_levels_of_a_well_spread_channel(tmp_path):
    ramp = np.tile(np.linspace(0, 65535, 800).astype(np.uint16), (640, 1))
    src = imageio.write_channels(tmp_path / "ramp.ome.tif", ramp[None], ("DAPI",))
    assert dhr.single_page_tiff(src, 0, tmp_path / "ramp.tif")["levels"] > 150


# ── the two maps ──────────────────────────────────────────────────────────────
def test_centre_pads_are_the_packages():
    assert dhr.centre_pads((10, 20), (15, 14)) == [(2, 0), (0, 3)]
    for a, b in (((10, 20), (15, 14)), ((7, 9), (7, 9)), ((301, 40), (300, 47))):
        pad_a, pad_b = fd.calculate_pad_value(a, b)
        assert dhr.centre_pads(a, b) == [
            (pad_a[0][0], pad_a[1][0]),
            (pad_b[0][0], pad_b[1][0]),
        ]


def test_working_scale_is_the_short_side_rule():
    """S and the working pixel on a 0.325 um cohort (the RegWSI report's table)."""
    want = {
        (40000, 4096): (9.766, 3.174),
        (50000, 4096): (12.207, 3.967),
        (40000, 8192): (5.0, 1.625),
        (50000, 8192): (6.104, 1.984),
    }
    for (short, resolution), (s, um) in want.items():
        got_s, k = dhr.working_scale(short, resolution)
        assert got_s == pytest.approx(s, abs=5e-4), (short, resolution)
        assert got_s * 0.325 == pytest.approx(um, abs=5e-4)
        assert k == pytest.approx(got_s * 0.2)
    assert dhr.working_scale(8192, 4096) == (5.0, 1.0)  # never enlarged


def _geometry(ref_hw, mov_hw, resolution):
    """A synthetic run's field and parameters, with the fake's bookkeeping."""
    fd.configure(initial_resolution=resolution)
    params = fd.default_initial_nonrigid()
    post, (h, w) = fd.geometry(mov_hw, ref_hw, params)
    post = json.loads(json.dumps(post))  # as read back from postprocessing_params.json
    return dhr.Geometry(fd.invent_field(h, w), post, ref_hw, mov_hw)


def _drawn_map(geo):
    """The writer's emulation applied to the moving slide's own coordinates: the plane it
    writes holds, at each pixel, where in the moving slide that pixel came from."""
    mh, mw = geo.mov_hw
    yy, xx = np.mgrid[0:mh, 0:mw].astype(np.float64)
    return [fd.draw(plane, geo.ref_hw, geo.field, -1e6) for plane in (xx, yy)]


def _lattice(hw, step=37):
    gx, gy = np.meshgrid(
        np.arange(11, hw[1] - 11, step), np.arange(13, hw[0] - 13, step)
    )
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)


EXACT_GEOMETRIES = {
    # name: (reference, moving, working short side) -> S
    "S5": ((400, 600), (360, 520), 4096),  # loaded 80 x 120, k = 1
    "S9.77": ((2500, 3750), (2420, 3590), 256),  # loaded 500 x 750, k = 1.953125
    "S6.10": ((3125, 6250), (3065, 6110), 512),  # loaded 625 x 1250, k = 625 / 512
}


@pytest.mark.parametrize("name", sorted(EXACT_GEOMETRIES))
def test_landmark_chain_and_asdrawn_chain_on_synthetic_params(name):
    ref_hw, mov_hw, resolution = EXACT_GEOMETRIES[name]
    geo = _geometry(ref_hw, mov_hw, resolution)
    s = float(name[1:])
    assert geo.scale_xy == pytest.approx([s, s], abs=5e-3)
    assert (geo.full_pad_mov_xy > 0).all() and geo.full_pad_mov_xy[
        0
    ] != geo.full_pad_mov_xy[1]
    registration = dhr.registration_map(geo, fd.warp_landmarks)
    asdrawn = dhr.asdrawn_map(geo)
    xy = _lattice(ref_hw)

    # the as-drawn map IS the emulated writer: where each written pixel came from
    drawn_x, drawn_y = _drawn_map(geo)
    got = asdrawn(xy)
    ix, iy = xy[:, 0].astype(int), xy[:, 1].astype(int)
    inside = (
        (got[:, 0] > 2)
        & (got[:, 0] < mov_hw[1] - 3)
        & (got[:, 1] > 2)
        & (got[:, 1] < mov_hw[0] - 3)
    )
    assert inside.mean() > 0.6
    seen = np.stack([drawn_x[iy, ix], drawn_y[iy, ix]], axis=1)
    assert np.abs(got[inside] - seen[inside]).max() < 0.05

    # the estimated field is S/2 away from it, through the local Jacobian
    gap = registration(xy) - got
    assert gap.mean(axis=0) == pytest.approx(geo.scale_xy / 2, abs=0.3)
    e = geo.scale_xy / 2
    jacobian_term = registration(xy + e / 2) - registration(xy - e / 2)
    assert np.abs(gap - jacobian_term)[inside].max() < 0.3
    declared = dhr.declared_offset(registration, geo)(xy)
    assert np.abs(gap - declared)[inside].max() < 0.05
    # ... and NOT a constant: the rotation alone moves it by more than the gate's mean
    assert np.ptp(gap[inside], axis=0).max() > 0.02 * s

    stats = dhr.gap_stats(registration, asdrawn, geo)
    assert stats["mean_px"] == pytest.approx(list(geo.scale_xy / 2), abs=0.3)
    assert stats["beyond_declared_px_p95"] < 0.05
    # the invented field is far steeper than a real one: only that the term is there
    assert 0 < stats["chain_halfpixel_px_p95"] < 2.0


def test_on_sizes_that_do_not_divide_only_the_asdrawn_map_follows_the_writer():
    """A slide whose sizes leave remainders: the writer pads at full resolution and
    stretches by canvas / field size, the registration padded the loaded image and
    resampled by k. The as-drawn map uses the writer's numbers and still reproduces it;
    the estimated-field chain is further from the image than its declared half pixel."""
    ref_hw, mov_hw = (2493, 3761), (2411, 3587)
    geo = _geometry(ref_hw, mov_hw, 256)
    assert geo.scale_xy[0] != pytest.approx(geo.scale_xy[1], abs=1e-3)
    assert geo.scale_xy[1] != pytest.approx(geo.k / 0.2, abs=1e-3)
    asdrawn = dhr.asdrawn_map(geo)
    xy = _lattice(ref_hw)
    drawn_x, drawn_y = _drawn_map(geo)
    got = asdrawn(xy)
    ix, iy = xy[:, 0].astype(int), xy[:, 1].astype(int)
    inside = (got[:, 0] > 2) & (got[:, 0] < mov_hw[1] - 3)
    inside &= (got[:, 1] > 2) & (got[:, 1] < mov_hw[0] - 3)
    seen = np.stack([drawn_x[iy, ix], drawn_y[iy, ix]], axis=1)
    assert inside.mean() > 0.6
    assert np.abs(got[inside] - seen[inside]).max() < 0.05
    registration = dhr.registration_map(geo, fd.warp_landmarks)
    stats = dhr.gap_stats(registration, asdrawn, geo)
    assert stats["beyond_declared_px_p95"] > 1.0


def test_the_estimated_field_goes_through_the_packages_own_function():
    geo = _geometry((400, 600), (360, 520), 4096)
    seen = []

    def spy(points, field):
        seen.append((np.array(points), field))
        return fd.warp_landmarks(points, field)

    xy = np.array([[100.0, 50.0], [333.0, 222.0]])
    out = dhr.registration_map(geo, spy)(xy)
    ((points, field),) = seen
    assert field is geo.field
    # issue #12's chain, written out: ratio, the TARGET's pad, k; then back with the
    # SOURCE's pad
    assert points == pytest.approx(xy * 0.2 + geo.pad_ref_xy)
    assert out == pytest.approx(
        (fd.warp_landmarks(points, field) - geo.pad_mov_xy) / 0.2
    )
    assert dhr.registration_map(geo, spy)(np.empty((0, 2))).shape == (0, 2)
    assert dhr.asdrawn_map(geo)(np.empty((0, 2))).shape == (0, 2)


def test_stage_times_are_read_from_the_packages_log():
    text = (
        "Loading time: 12.5 seconds.\nPreprocessing time: 3 seconds.\n"
        "Initial registration time: 1.5e1 seconds.\n"
        "Nonrigid registration time: 99.25 seconds.\n"
        "Total registration time: 130.0 seconds.\n"
    )
    assert dhr.stage_times(text) == {
        "loading_s": 12.5,
        "preprocessing_s": 3.0,
        "initial_registration_s": 15.0,
        "nonrigid_registration_s": 99.25,
        "total_registration_s": 130.0,
    }
    assert dhr.stage_times("Exception: boom") == {}


# ── the run, against the emulated writer ──────────────────────────────────────
def test_the_unmutated_adapter_passes_both_rows(tmp_path):
    rec, v = _selfcheck(tmp_path)
    assert rec["status"] == sc.OK and not rec["blocks_the_method"], v
    assert rec["variants"] == {"deeperhistreg": sc.OK, "deeperhistreg_asdrawn": sc.OK}
    drawn, field = v["deeperhistreg_asdrawn"], v["deeperhistreg"]
    assert drawn["declared_offset"] is False and field["declared_offset"] is True
    assert drawn["direction"] == REF_TO_MOV and drawn["n_blocks_judged"] >= 2
    assert drawn["n_found"] >= 0.99 * drawn["n_dots"] > 50
    # the measured offset between the two maps: S / 2 = 5 px on each axis
    assert field["declared_px_mean_x"] == pytest.approx(5.0, abs=0.3)
    assert field["declared_px_mean_y"] == pytest.approx(5.0, abs=0.3)
    assert drawn["native"]["origin_xy"] == [0.0, 0.0]


def _mutate_direction(monkeypatch):
    monkeypatch.setattr(dhr, "DIRECTION", MOV_TO_REF)


def _mutate_drop_the_pads(monkeypatch):
    monkeypatch.setattr(dhr, "centre_pads", lambda a, b: [(0, 0), (0, 0)])


def _mutate_swap_the_pads(monkeypatch):
    real = dhr.centre_pads
    monkeypatch.setattr(dhr, "centre_pads", lambda a, b: real(a, b)[::-1])


def _mutate_scale(monkeypatch):
    real = dhr.scale_xy
    monkeypatch.setattr(
        dhr,
        "scale_xy",
        lambda canvas, field: tuple(1.01 * v for v in real(canvas, field)),
    )


def _mutate_no_writer_offset(monkeypatch):
    monkeypatch.setattr(dhr, "WRITER_SHIFT", 0.0)


def _mutate_inverse_field(monkeypatch):
    """The field read as moving -> reference (its sign flipped)."""
    real = dhr.Geometry.__init__

    def flipped(self, field, post, ref_hw, mov_hw):
        real(self, -np.asarray(field), post, ref_hw, mov_hw)

    monkeypatch.setattr(dhr.Geometry, "__init__", flipped)


@pytest.mark.parametrize(
    "mutate",
    [
        _mutate_direction,
        _mutate_drop_the_pads,
        _mutate_swap_the_pads,
        _mutate_scale,
        _mutate_no_writer_offset,
        _mutate_inverse_field,
    ],
    ids=lambda f: f.__name__,
)
def test_a_wrong_asdrawn_map_is_inconsistent(tmp_path, monkeypatch, mutate):
    """Each of these is a mistake the adapter could contain; the selfcheck must object
    on the row that can be primary."""
    mutate(monkeypatch)
    rec, v = _selfcheck(tmp_path)
    assert rec["status"] == sc.INCONSISTENT and rec["blocks_the_method"], v
    assert v["deeperhistreg_asdrawn"]["status"] == sc.INCONSISTENT
    assert "whole image" in v["deeperhistreg_asdrawn"]["failed"]


def test_without_its_declared_offset_the_estimated_field_row_is_inconsistent(
    tmp_path, monkeypatch
):
    """The estimated field is half a field pixel away from the image: undeclared, the
    check must see it, and see it as the 5 px it is."""
    monkeypatch.setattr(dhr, "SELFCHECK_OFFSET", lambda info: {})
    rec, v = _selfcheck(tmp_path)
    assert rec["status"] == sc.INCONSISTENT
    assert v["deeperhistreg_asdrawn"]["status"] == sc.OK
    assert v["deeperhistreg"]["status"] == sc.INCONSISTENT


def test_a_wrong_chain_does_not_cancel_out_of_the_estimated_field_rows_check(
    tmp_path, monkeypatch
):
    """The declared offset is built from the registration map alone, so a wrong chain
    (here: the two pads exchanged) still fails that row, and only that row."""
    real = dhr.registration_map

    def swapped(geo, warp_landmarks):
        geo = copy.copy(geo)
        geo.pad_mov_xy, geo.pad_ref_xy = geo.pad_ref_xy, geo.pad_mov_xy
        return real(geo, warp_landmarks)

    monkeypatch.setattr(dhr, "registration_map", swapped)
    rec, v = _selfcheck(tmp_path)
    assert v["deeperhistreg_asdrawn"]["status"] == sc.OK
    assert v["deeperhistreg"]["status"] == sc.INCONSISTENT
    assert rec["status"] == sc.INCONSISTENT


def test_on_sizes_that_do_not_divide_the_asdrawn_row_still_passes(tmp_path, fake):
    """The floor of the package's resampling (a field of 85 x 104 for a loaded 208 x 256)
    stretches the written image by canvas / field size. The as-drawn row follows it; the
    estimated-field row, with its half-pixel declaration, does not."""
    fake.configure(initial_resolution=85)
    rec, v = _selfcheck(tmp_path)
    assert v["deeperhistreg_asdrawn"]["status"] == sc.OK, v["deeperhistreg_asdrawn"]
    assert v["deeperhistreg"]["status"] == sc.INCONSISTENT
    (path,) = (tmp_path / "out" / "deeperhistreg_asdrawn").rglob("run.json")
    info = json.loads(path.read_text())["info"]
    assert info["S_x"] != pytest.approx(info["S_y"], abs=1e-3)
    assert info["gap"]["beyond_declared_px_p95"] > 0.4


# ── what a run calls and records ──────────────────────────────────────────────
def test_the_call_is_the_packages_own_in_its_native_roles(tmp_path, case, fake):
    done = methods.run_case("deeperhistreg", case, tmp_path / "out", keep_work=True)
    assert done == ["deeperhistreg", "deeperhistreg_asdrawn"]
    (call,) = fake.calls("run_registration")
    work = result_dir(tmp_path / "out", "deeperhistreg_asdrawn", case) / "work"
    assert call["source_path"] == str(work / "moving.tif")  # the moving slide is SOURCE
    assert call["target_path"] == str(work / "reference.tif")
    assert call["save_displacement_field"] is True and call["copy_target"] is False
    assert call["delete_temporary_results"] is False
    assert call["temporary_path"] == str(work / "tmp")
    assert set(call) == {
        "call",
        "source_path",
        "target_path",
        "output_path",
        "case_name",
        "save_displacement_field",
        "copy_target",
        "delete_temporary_results",
        "temporary_path",
        "parameters",
    }
    # the inputs are single pages holding the nuclear channel, value for value
    with tifffile.TiffFile(case.mov_image) as tif:
        nuclear = tif.series[0].asarray()[0]
    with tifffile.TiffFile(work / "moving.tif") as tif:
        assert len(tif.pages) == 1 and np.array_equal(tif.pages[0].asarray(), nuclear)

    # the writer: what save_final passes, once per channel of the moving slide
    writes = fake.calls("apply_deformation_pyvips")
    assert len(writes) == 2
    for c, w in enumerate(writes):
        assert w["target"] == str(work / "reference.tif")
        assert w["field"] == str(work / "out" / "displacement_field.mha")
        assert (w["loader"], w["saver"]) == ("TIFFLoader", "TIFFSaver")
        assert (w["level"], w["pad_value"]) == (0, 255.0)
        assert w["save_source_only"] is True and w["to_template_shape"] is True
        assert w["save_params"] is None and w["to_save_target_path"] is None
        assert Path(w["warped"]).name == f"warped_c{c}.tiff"
    assert writes[0]["source"] == str(work / "moving.tif")


def test_parameters_are_the_factorys_but_for_the_changed_keys(tmp_path, case, fake):
    methods.run_case(
        "deeperhistreg",
        case,
        tmp_path,
        opts={"config": HR, "fluo": 1},
        label="deeperhistreg_hr_fluo_cpu",
    )
    (call,) = fake.calls("run_registration")
    keys = [
        "device",
        "initial_registration_params.cuda",
        "initial_registration_params.device",
        "loading_params.pad_value",
        "nonrigid_registration_params.device",
        "preprocessing_params.flip_intensity",
        "save_final_images",
    ]
    factory = fd.default_initial_nonrigid_high_resolution()
    assert _without(call["parameters"], keys) == _without(factory, keys)
    got = call["parameters"]
    assert got["device"] == "cpu" and got["save_final_images"] is False
    assert got["initial_registration_params"]["cuda"] is False
    assert got["loading_params"]["pad_value"] == 0.0
    run = run_json(tmp_path, "deeperhistreg_hr_fluo_cpu_asdrawn", case)
    impl = run["info"]["implementation"]
    assert impl["changed_keys"] == keys
    assert impl["parameters"] == json.loads(json.dumps(got))
    assert fake.calls("apply_deformation_pyvips")[0]["pad_value"] == 0.0
    assert run["native"]["fill"].startswith("0 counts inside")


def test_info_names_regwsi_and_the_departures(tmp_path, case):
    methods.run_case("deeperhistreg", case, tmp_path)
    for variant in dhr.VARIANTS:
        impl = run_json(tmp_path, variant, case)["info"]["implementation"]
        assert impl["method"] == (
            "RegWSI (Wodzinski et al. 2024, doi:10.1016/j.cmpb.2024.108187)"
        )
        assert impl["package"] == "deeperhistreg"
        assert impl["config_factory"] == "default_initial_nonrigid"
        assert impl["paper_resolution_px"] == 4096
        departures = impl["departures_from_paper_text"]
        assert len(departures) == 5
        for i, word in enumerate(
            ("scale-invariant", "rigid", "nine", "linear", "flip_intensity")
        ):
            assert word in departures[i], (i, departures[i])
        side = " ".join(impl["benchmark_side"])
        for word in (
            "one top-level page",
            "issue #12",
            "re-stated",
            "stacked",
            "stdout",
        ):
            assert word in side, word
    methods.run_case(
        "deeperhistreg",
        case,
        tmp_path,
        opts={"config": HR},
        label="deeperhistreg_hr_cpu",
    )
    impl = run_json(tmp_path, "deeperhistreg_hr_cpu", case)["info"]["implementation"]
    assert impl["paper_resolution_px"] == 8192 and impl["config_factory"] == HR


def test_working_resolution_recorded(tmp_path, case):
    methods.run_case("deeperhistreg", case, tmp_path)
    info = run_json(tmp_path, "deeperhistreg_asdrawn", case)["info"]
    impl = info["implementation"]
    # the fake's geometry: loaded 208 x 256, a working short side of 104 px
    assert impl["initial_resample_ratio"] == 2.0
    assert info["S_x"] == info["S_y"] == 10.0
    assert impl["working_um_per_px"] == pytest.approx(10.0 * sc.PIXEL_SIZE_UM)
    assert info["field_shape"] == [2, 104, 128] and info["canvas_hw"] == [1040, 1280]
    assert info["postprocessing_params"]["pad_1"] == [[4, 4], [8, 8]]
    assert info["postprocessing_params"]["pad_2"] == [[0, 0], [0, 0]]
    assert info["stage_s"]["nonrigid_registration_s"] == 30.5
    assert info["workers_seen"] == 4 and info["torch_num_threads"] == 4
    assert set(info["grey_levels"]) == {"reference", "moving"}
    assert info["gap"]["mean_px"] == pytest.approx([5.0, 5.0], abs=0.3)
    assert info["gap"]["beyond_declared_px_p95"] < 0.05
    assert info["device"] == "cpu" and info["version"] == "1.0.1"


def test_the_written_image_is_on_the_reference_grid(tmp_path, case, pair):
    methods.run_case("deeperhistreg", case, tmp_path)
    run = run_json(tmp_path, "deeperhistreg_asdrawn", case)
    native = run["native"]
    assert run["write_s"] > 0 and native["origin_xy"] == [0.0, 0.0]
    assert native["nuclear_channel"] == 0 and native["channels"] == ["DAPI", "dots"]
    # the write step in its three parts (the driver's write_s is the whole of it): the
    # conversion of the further channel, the package's writer, the benchmark's stack
    parts = native["write_parts_s"]
    assert set(parts) == {"convert", "writer", "benchmark"}
    assert all(v >= 0 for v in parts.values()) and parts["writer"] > 0
    assert sum(parts.values()) <= run["write_s"]
    # what the PACKAGE wrote, described before the benchmark stacked it: one file per
    # channel, so it is not one image that holds every channel
    own = native["written"]
    assert own["files"] == 2 and own["channels"] == 1 and own["bytes"] > 0
    assert "one per channel" in own["of"]
    assert cost.complete_image(own) is False
    assert native["dtype"] == "uint16" and "255 counts" in native["fill"]
    (path,) = native["files"]
    assert (
        Path(path).parent
        == result_dir(tmp_path, "deeperhistreg_asdrawn", case) / "native"
    )
    assert imageio.shape_of(path) == (2, *pair["ref_hw"])
    assert sorted(p.name for p in Path(path).parent.iterdir()) == ["registered.ome.tif"]
    x0, y0, x1, y1 = native["valid_box"]
    h, w = pair["ref_hw"]
    assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
    # the moving slide is smaller: some of the reference grid is fill
    assert (x1 - x0) * (y1 - y0) < h * w
    # the stack holds the writer's pixels, value for value
    with tifffile.TiffFile(case.mov_image) as tif:
        dots = tif.series[0].asarray()[1]
    with tifffile.TiffFile(path) as tif:
        stacked = tif.series[0].asarray()
    geo = _geometry(pair["ref_hw"], pair["mov_hw"], EXACT_RESOLUTION)
    assert np.array_equal(stacked[1], fd.draw(dots, pair["ref_hw"], geo.field, 255.0))
    # the estimated-field row has no image of its own
    assert run_json(tmp_path, "deeperhistreg", case)["native"] is None


def test_each_call_has_its_own_case_name(tmp_path, case, fake):
    methods.run_case("deeperhistreg", case, tmp_path / "a")
    methods.run_case("deeperhistreg", case, tmp_path / "b")
    names = [c["case_name"] for c in fake.calls("run_registration")]
    assert len(set(names)) == 2 and all(n.startswith("regbench_") for n in names)


# ── devices and failures ──────────────────────────────────────────────────────
def test_cuda_request_without_cuda_fails(tmp_path, case, fake):
    done = methods.run_case(
        "deeperhistreg",
        case,
        tmp_path,
        opts={"device": "cuda:0"},
        label="deeperhistreg_gpu",
    )
    assert done == [] and fake.calls("run_registration") == []
    for variant in ("deeperhistreg_gpu", "deeperhistreg_gpu_asdrawn"):
        run = run_json(tmp_path, variant, case)
        assert run["ok"] is False and run["cause"] == "exception"
        assert "no CUDA device" in run["error"]


def test_a_gpu_row_sets_every_device_key(tmp_path, case, fake):
    fake.configure(cuda=True, initial_resolution=EXACT_RESOLUTION)
    done = methods.run_case(
        "deeperhistreg",
        case,
        tmp_path,
        opts={"device": "cuda:0"},
        label="deeperhistreg_gpu",
    )
    assert done == ["deeperhistreg_gpu", "deeperhistreg_gpu_asdrawn"]
    params = fake.calls("run_registration")[0]["parameters"]
    assert params["device"] == "cuda:0"
    assert params["initial_registration_params"]["device"] == "cuda:0"
    assert params["initial_registration_params"]["cuda"] is True
    assert params["nonrigid_registration_params"]["device"] == "cuda:0"
    assert run_json(tmp_path, "deeperhistreg_gpu", case)["info"]["device"] == "cuda:0"


def test_a_cpu_row_must_rewrite_the_nested_devices(tmp_path, case, monkeypatch):
    """Why ``changed_keys`` walks the whole dict: with only the top-level device changed
    the package fails inside its swallowed ``try`` (nested blocks stay ``cuda:0``)."""

    def top_only(params, device, fluo):
        params["device"], params["save_final_images"] = device, False
        return ["device", "save_final_images"]

    monkeypatch.setattr(dhr, "changed_keys", top_only)
    assert methods.run_case("deeperhistreg", case, tmp_path) == []
    run = run_json(tmp_path, "deeperhistreg", case)
    assert "torch.cuda.is_available() is False" in run["error"]


@pytest.mark.parametrize(
    "text,cause",
    [
        ("boom at pyramid level 3", "exception"),
        ("CUDA out of memory. Tried to allocate 20.00 GiB", "gpu_oom"),
        ("DefaultCPUAllocator: can't allocate memory", "oom"),
    ],
)
def test_a_swallowed_failure_is_raised_with_what_the_package_printed(
    tmp_path, case, fake, text, cause
):
    fake.configure(fail=text, initial_resolution=EXACT_RESOLUTION)
    assert methods.run_case("deeperhistreg", case, tmp_path) == []
    for variant in dhr.VARIANTS:
        run = run_json(tmp_path, variant, case)
        assert run["ok"] is False and run["cause"] == cause
        assert "wrote no displacement field" in run["error"] and text in run["error"]


def test_a_failed_writer_keeps_both_rows(tmp_path, case, fake):
    fake.configure(
        writer_fails="vips: out of disc", initial_resolution=EXACT_RESOLUTION
    )
    done = methods.run_case("deeperhistreg", case, tmp_path)
    assert done == ["deeperhistreg", "deeperhistreg_asdrawn"]
    run = run_json(tmp_path, "deeperhistreg_asdrawn", case)
    assert run["ok"] is True and "out of disc" in run["native"]["error"]


def test_another_version_is_refused(tmp_path, case, monkeypatch):
    fd.install(monkeypatch.setitem, version="1.0.0")
    assert methods.run_case("deeperhistreg", case, tmp_path) == []
    run = run_json(tmp_path, "deeperhistreg", case)
    assert "1.0.1 is the pinned release; found 1.0.0" in run["error"]


def test_a_brightfield_case_is_refused(tmp_path, case):
    case.modality = "brightfield"
    with pytest.raises(ValueError, match="fluorescence channel"):
        next(dhr.register(case, tmp_path / "work", {}))


class _Clock:
    paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


def test_conversion_is_off_the_clock_and_the_registration_on_it(
    tmp_path, case, monkeypatch
):
    clock, seen = _Clock(), {}
    real_convert, real_run, real_gap = (
        dhr.single_page_tiff,
        dhr.run_package,
        dhr.gap_stats,
    )

    def converting(*a, **kw):
        seen.setdefault("convert", []).append(clock.paused)
        return real_convert(*a, **kw)

    def running(*a, **kw):
        seen["run"] = clock.paused
        return real_run(*a, **kw)

    def describing(*a, **kw):
        seen["gap"] = clock.paused
        return real_gap(*a, **kw)

    monkeypatch.setattr(dhr, "single_page_tiff", converting)
    monkeypatch.setattr(dhr, "run_package", running)
    monkeypatch.setattr(dhr, "gap_stats", describing)
    items = list(dhr.register(case, tmp_path / "work", {"monitor": clock}))
    assert seen == {"convert": [True, True], "run": False, "gap": True}
    assert not clock.paused  # handed back running
    assert [v for v, _, _ in items] == list(dhr.VARIANTS)
    assert all(
        isinstance(m, PointMap) and m.direction == REF_TO_MOV for _, m, _ in items
    )
    assert "_state" in items[1][2] and "_state" not in items[0][2]
    assert items[0][2]["convert_s"] > 0


def test_the_realcheck_is_the_asdrawn_rows(tmp_path, case):
    """A case without nuclei cannot be judged, but the wiring is visible: the row with
    the image gets a realcheck record, the estimated-field row none."""
    methods.run_case("deeperhistreg", case, tmp_path, realcheck=True)
    assert run_json(tmp_path, "deeperhistreg_asdrawn", case)["realcheck"]["status"] == (
        sc.NOT_CHECKED
    )
    assert "realcheck" not in run_json(tmp_path, "deeperhistreg", case)


def test_the_selfcheck_pair_is_the_cohorts_class_for_every_label():
    """One pair for all eight labels: 40000 px on the short side gives the cohort's
    S = 9.77 at the 4096 px set and S = 5.0 at the 8192 px set, and a coarsest pyramid
    level of 31 px under nine levels."""
    for label in LABELS:
        ((ref_hw, mov_hw),) = sc.sizes_for(label)
        short = min(max(ref_hw[0], mov_hw[0]), max(ref_hw[1], mov_hw[1]))
        resolution = 8192 if "_hr" in label else 4096
        s, k = dhr.working_scale(short, resolution)
        assert s == pytest.approx(5.0 if "_hr" in label else 9.765625), label
        # every division of the package's resamplings is exact on this pair: the
        # loaded sizes, the field's size, and the pads at both resolutions
        canvas = (max(ref_hw[0], mov_hw[0]), max(ref_hw[1], mov_hw[1]))
        assert all(v % 20 == 0 for v in (*ref_hw, *mov_hw))
        assert all(math.isclose((v * 0.2 / k) % 1, 0, abs_tol=1e-9) for v in canvas)
        loaded = [tuple(round(v * 0.2) for v in hw) for hw in (mov_hw, ref_hw)]
        for full, small in zip(
            dhr.centre_pads(mov_hw, ref_hw), dhr.centre_pads(*loaded)
        ):
            assert full == tuple(5 * v for v in small)
        levels = 9 if "_hr" in label else 8
        assert (short * 0.2 / k) / 2 ** (levels - 1) >= 31
