"""selfcheck and realcheck: the checks must be able to FAIL.

A "render" here is the moving slide's dots channel resampled through an image transform A
(moving -> reference) with ``scipy.ndimage.map_coordinates``: it stands in for a method's
own writer. P is the point mapping that is checked against it. P consistent with A must be
``ok`` whatever A is; a P with the wrong direction, an offset, a scale error, a wrong strip
or a wrong block must be ``inconsistent``; an output the dots cannot be seen in must be
``not_checked``, never ``inconsistent``.
"""

import json
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy.ndimage import map_coordinates

from benchmarks.regbench import cells, cli, imageio, methods
from benchmarks.regbench import selfcheck as sc
from benchmarks.regbench.cases import load_points, result_dir, write_case, write_json
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap

HW = (1024, 1280)
SMALL = (512, 640)
DIRECTIONS = (MOV_TO_REF, REF_TO_MOV)


class Pair:
    """The pair, its planes in memory, and renders of it."""

    def __init__(self, work, hw=HW, mov_hw=None):
        self.d = sc.make_pair(work, hw, mov_hw)
        self.mov = tifffile.imread(self.d["moving"])
        self.ref = tifffile.imread(self.d["reference"])
        self.dots = self.d["dots_mov_xy"]
        self.truth, self.inverse = self.d["truth"], self.d["truth_inverse"]
        h, w = self.d["ref_hw"]
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        self.grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
        self._renders = {}

    def a_inv(self, shift=(0.0, 0.0)):
        """reference -> moving of the image transform A = truth + shift."""
        shift = np.asarray(shift, dtype=float)
        return lambda q: self.inverse(np.asarray(q, dtype=float) - shift)

    def a(self, shift=(0.0, 0.0)):
        shift = np.asarray(shift, dtype=float)
        return lambda p: self.truth(p) + shift

    def render(self, channel=1, shift=(0.0, 0.0), order=1):
        key = (channel, tuple(shift), order)
        if key not in self._renders:
            h, w = self.d["ref_hw"]
            p = self.a_inv(shift)(self.grid)
            self._renders[key] = map_coordinates(
                self.mov[channel].astype(float),
                [p[:, 1].reshape(h, w), p[:, 0].reshape(h, w)],
                order=order,
            )
        return self._renders[key]

    def maps(self, shift=(0.0, 0.0)):
        """P consistent with A = truth + shift, in each direction."""
        return {
            MOV_TO_REF: PointMap(self.a(shift), MOV_TO_REF),
            REF_TO_MOV: PointMap(self.a_inv(shift), REF_TO_MOV),
        }


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    return Pair(tmp_path_factory.mktemp("selfcheck_pair"))


def plus(pmap, offset):
    """The mapping, moved by ``offset`` (a constant or a function of the input points)."""
    off = offset if callable(offset) else (lambda xy: np.asarray(offset, dtype=float))
    return PointMap(lambda xy: pmap.fn(xy) + off(np.asarray(xy)), pmap.direction)


# ── the pair ──────────────────────────────────────────────────────────────────
def test_the_pair_is_two_named_channels_on_a_canvas_that_is_not_square(pair):
    d = pair.d
    assert pair.mov.shape == (2, *HW) and pair.ref.shape == (2, *HW)
    assert pair.mov.dtype == np.uint16 and HW[0] != HW[1]
    for path in (d["reference"], d["moving"]):
        assert imageio.channel_names(path) == ["DAPI", "dots"]
        assert imageio.pixel_size_um(path) == pytest.approx(0.325)
    assert pair.ref[1].max() == 0  # the reference's second channel is empty
    assert 0.9 * sc.DOT_PEAK < pair.mov[1].max() <= sc.DOT_PEAK
    # the tissue does not start at the origin
    ys, xs = np.nonzero(pair.ref[0] > 0.3 * pair.ref[0].max())
    assert xs.min() > 0.05 * HW[1] and ys.min() > 0.03 * HW[0]


def test_dots_are_a_jittered_lattice_inside_the_moving_frame(pair):
    xy = pair.dots
    assert len(xy) > 100
    assert xy.min() >= sc.DOT_MARGIN
    assert (xy[:, 0] <= HW[1] - 1 - sc.DOT_MARGIN).all()
    assert (xy[:, 1] <= HW[0] - 1 - sc.DOT_MARGIN).all()
    off = (xy - sc.LATTICE / 2.0) % sc.LATTICE
    off = np.minimum(off, sc.LATTICE - off)
    assert off.max() <= sc.JITTER and off.std() > 4.0  # jittered, not a plain lattice


def test_the_inverse_of_the_generators_map_is_good_to_a_ten_thousandth_of_a_pixel(pair):
    q = pair.grid[::997]
    assert np.abs(pair.truth(pair.inverse(q)) - q).max() < 1e-4


def test_the_generator_is_unchanged_for_square_specs():
    from benchmarks.regbench.datasets import synthetic

    scene = synthetic.Scene({**synthetic._BASE, "family": "t", "n": 256, "seed": 3})
    assert scene.shape("ref") == (256, 256) and scene.shape("mov") == (256, 256)
    assert len(scene.ref[0]) == 55  # the same nuclei as before `ref_hw` existed


def test_sizes_are_the_ones_each_method_needs():
    assert sc.sizes_for("initial") == [(sc.DEFAULT_HW, None)]
    assert sc.sizes_for("stare")[0] == (sc.DEFAULT_HW, None)
    assert sc.sizes_for("stare")[1][0] == (4096, 5120)  # a four-tile junction
    for name in ("valis", "valis_mirage", "valis120", "valis120_hires"):
        assert max(sc.sizes_for(name)[-1][0]) >= 30000, name
    # every per-patient method has ONE patient of three slides; a pair method has none
    # of its own (`stare` runs ASHLAR's, round by round, for the anchored rows)
    for name in ("valis_mirage", "valis120", "valis120_hires"):
        assert [s for s in sc.sizes_for(name) if sc.is_patient(s)] == [sc.VALIS_PATIENT]
    for name in ("ashlar", "ashlar_clean", "affine_ashlar", "affine_ashlar_clean"):
        assert sc.sizes_for(name) == [(sc.ASHLAR_HW, None), sc.ASHLAR_PATIENT], name
        assert set(sc.sizes_for(name)) <= set(sc.sizes_for("stare"))  # the anchor
    assert sc.rounds_of(sc.ASHLAR_PATIENT) == (sc.ASHLAR_HW, [None, None])
    for name in ("deeperhistreg_hr_gpu", "palom", "initial"):
        assert not any(sc.is_patient(s) for s in sc.sizes_for(name)), name
    # one pair for every DeeperHistReg label, the high-resolution ones included
    assert sc.sizes_for("deeperhistreg_gpu") == sc.SIZES["deeperhistreg"]
    assert sc.sizes_for("deeperhistreg_hr_fluo_cpu") == sc.SIZES["deeperhistreg"]
    assert "deeperhistreg_hr" not in sc.SIZES
    (ref_hw, mov_hw), swapped = sc.SIZES["deeperhistreg"]
    assert swapped == (mov_hw, ref_hw)  # the same canvases, the reference padded
    assert min(ref_hw) > 20480 and min(mov_hw) > 20480
    assert all(v % 20 == 0 for v in (*ref_hw, *mov_hw))


# ── blind detection ───────────────────────────────────────────────────────────
def test_find_dots_locates_clean_dots_to_a_tenth_of_a_pixel(pair):
    found = sc.find_dots(pair.mov[1])
    assert len(found) == len(pair.dots)
    d = np.linalg.norm(found[:, None] - pair.dots[None], axis=2).min(axis=1)
    assert d.max() < 0.1


def test_nuclei_are_not_mistaken_for_dots(pair):
    assert pair.mov[0].max() > 10000  # sigma 5 px blobs, as bright as the dots
    assert len(sc.find_dots(pair.mov[0])) == 0
    assert len(sc.find_dots(pair.ref[0])) == 0


def test_a_lazy_channel_is_read_by_region(pair):
    from stare.slide_io import open_lazy

    arr, _, close = open_lazy(pair.d["moving"])
    try:
        found = sc.find_dots(sc._Plane(arr, 1))
    finally:
        close()
    assert len(found) == len(pair.dots)


# ── a dot is a dot at the size the writer drew it ─────────────────────────────
SQUEEZE_X0, SQUEEZE = 800.0, 0.5


def _squeeze(q):
    """reference -> reference: everything right of ``SQUEEZE_X0`` drawn at half width."""
    q = np.asarray(q, dtype=float).copy()
    right = q[:, 0] > SQUEEZE_X0
    q[right, 0] = SQUEEZE_X0 + SQUEEZE * (q[right, 0] - SQUEEZE_X0)
    return q


def _unsqueeze(q):
    q = np.asarray(q, dtype=float).copy()
    right = q[:, 0] > SQUEEZE_X0
    q[right, 0] = SQUEEZE_X0 + (q[right, 0] - SQUEEZE_X0) / SQUEEZE
    return q


def squeezed(pair):
    """A writer whose transform shrinks one strip of the image to half its width, and
    the two point mappings that follow it exactly."""
    h, w = pair.d["ref_hw"]
    p = pair.inverse(_unsqueeze(pair.grid))
    image = map_coordinates(
        pair.mov[1].astype(float), [p[:, 1].reshape(h, w), p[:, 0].reshape(h, w)]
    )
    forward = PointMap(lambda xy: _squeeze(pair.truth(xy)), MOV_TO_REF)
    back = PointMap(lambda xy: pair.inverse(_unsqueeze(xy)), REF_TO_MOV)
    return image, {MOV_TO_REF: forward, REF_TO_MOV: back}


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_a_dot_the_writer_shrank_is_still_a_dot(pair, direction):
    """The case STARE's selfcheck meets at 8192 x 10240: the writer's own
    transform shrinks the image somewhere, the points follow it exactly, and the dots
    there are narrower than a dot as drawn. They are found and measured like the rest."""
    image, maps = squeezed(pair)
    forward = maps[MOV_TO_REF].fn
    res = sc.compare(maps[direction], pair.dots, image, expect=forward)
    assert res["status"] == sc.OK, res["failed"]
    assert res["n_found"] == res["n_dots"] and res["err_px_p95"] < 0.25
    # the strip's dots are in the count, and said to be narrow
    at = forward(pair.dots)
    strip = (at[:, 0] > SQUEEZE_X0 + 8) & sc._inside(at, HW, sc.EDGE_PX)
    assert strip.sum() > 30 and 0.8 * strip.sum() < res["n_narrow"] <= strip.sum() + 12
    found, width = sc.find_dots(image, widths=True)
    narrow = width < sc.NARROW_PX
    assert narrow.sum() == res["n_narrow"] and (found[narrow, 0] > SQUEEZE_X0).all()
    assert sc.WIDTH_PX[0] < width.min() < sc.NARROW_PX < np.median(width) < 1.5
    if direction == REF_TO_MOV:  # and every drawn dot is shown
        assert res["coverage_checked"] and res["n_covered"] == res["n_expected"] > 100


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_a_shrunk_dot_in_the_wrong_place_is_still_inconsistent(pair, direction):
    """Counting narrow dots as dots loosens nothing: points that do not follow the
    shrunk strip are reported, by the strip's own blocks."""
    image, maps = squeezed(pair)
    res = sc.compare(pair.maps()[direction], pair.dots, image)  # the map without it
    assert res["status"] == sc.INCONSISTENT and res["n_found"] < res["n_dots"]
    assert any(f.startswith("block (") for f in res["failed"])
    off = sc.compare(plus(maps[direction], (0.3, 0.0)), pair.dots, image)
    assert off["status"] == sc.INCONSISTENT and "whole image" in off["failed"]
    assert off["n_found"] == off["n_dots"]


def test_what_is_narrower_than_half_a_dot_is_not_a_dot(pair):
    """The lower bound still has something to refuse: single hot pixels and pairs of
    them, as bright as a dot, are not detections; a nucleus is still too wide."""
    assert sc.WIDTH_PX == (0.7, 1.8) and sc.NARROW_PX == 1.2
    plane = pair.mov[1].astype(float)
    n = len(sc.find_dots(plane))
    rng = np.random.default_rng(3)
    free = np.argwhere(plane[8:-8, 8:-8] == 0) + 8
    for y, x in free[rng.choice(len(free), 40, replace=False)]:
        plane[y, x] = sc.DOT_PEAK
        if x % 2:
            plane[y, x + 1] = sc.DOT_PEAK
    assert len(sc.find_dots(plane)) == n == len(pair.dots)
    assert len(sc.find_dots(pair.mov[0])) == 0


# ── a consistent mapping is ok, whatever the writer does to the pixels ────────
def _pedestal(img):
    rng = np.random.default_rng(5)
    return rng.poisson(np.clip(img, 0, None) + 1600.0).astype(float)


def _eight_bit(img):
    return np.rint(255.0 * np.clip(img, 0, None) / img.max()).astype(np.uint8)


WRITERS = {
    "bilinear": (1, lambda img: img),
    "bicubic": (3, lambda img: img),
    "pedestal_poisson": (1, _pedestal),
    "eight_bit": (1, _eight_bit),
}


@pytest.mark.parametrize("writer", sorted(WRITERS))
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_a_consistent_map_is_ok(pair, direction, writer):
    order, post = WRITERS[writer]
    res = sc.compare(pair.maps()[direction], pair.dots, post(pair.render(order=order)))
    assert res["status"] == sc.OK, res["failed"]
    assert res["err_px_p95"] < 0.25
    assert res["n_found"] == res["n_dots"] > 100
    assert res["n_blocks_judged"] >= 4
    assert max(abs(res["offset_px_mean_x"]), abs(res["offset_px_mean_y"])) < 0.03


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_registration_error_does_not_matter(pair, direction):
    """A is 20 px away from the truth; P follows A: the check is about P and A only."""
    shift = (20.0, 0.0)
    res = sc.compare(pair.maps(shift)[direction], pair.dots, pair.render(shift=shift))
    assert res["status"] == sc.OK, res["failed"]
    assert res["err_px_p95"] < 0.25


def test_an_output_with_an_origin_is_compared_in_the_reference_frame(pair):
    out = pair.render()[100:, 60:]
    res = sc.compare(pair.maps()[MOV_TO_REF], pair.dots, out, origin_xy=(60, 100))
    assert res["status"] == sc.OK
    assert sc.compare(pair.maps()[MOV_TO_REF], pair.dots, out)["status"] != sc.OK


# ── a wrong mapping is inconsistent ───────────────────────────────────────────
def test_a_flipped_direction_is_inconsistent(pair):
    out = pair.render()
    flipped = {
        MOV_TO_REF: PointMap(pair.inverse, MOV_TO_REF),
        REF_TO_MOV: PointMap(pair.truth, REF_TO_MOV),
    }
    for direction in DIRECTIONS:
        res = sc.compare(flipped[direction], pair.dots, out)
        assert res["status"] == sc.INCONSISTENT, direction
        assert res["n_found"] < 0.5 * res["n_dots"]


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize(
    "offset", [(0.30, 0.0), (0.0, -0.30), (0.5, 0.5)], ids=["x.30", "y-.30", "half"]
)
def test_a_fraction_of_a_pixel_offset_is_inconsistent(pair, direction, offset):
    res = sc.compare(plus(pair.maps()[direction], offset), pair.dots, pair.render())
    assert res["status"] == sc.INCONSISTENT
    assert "whole image" in res["failed"]
    assert res["n_found"] == res["n_dots"]  # found, and measured as offset
    assert res["offset_px_mean_x"] == pytest.approx(offset[0], abs=0.03)
    assert res["offset_px_mean_y"] == pytest.approx(offset[1], abs=0.03)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_a_tenth_of_a_percent_scale_error_is_inconsistent(pair, direction):
    centre = np.array([HW[1] - 1, HW[0] - 1]) / 2.0
    good = pair.maps()[direction]
    scaled = PointMap(
        lambda xy: centre + 1.001 * (good.fn(xy) - centre), good.direction
    )
    res = sc.compare(scaled, pair.dots, pair.render())
    assert res["status"] == sc.INCONSISTENT
    # about the centre the mean offset cancels: the percentile and the blocks object
    assert abs(res["offset_px_mean_x"]) < sc.GATE_MEAN
    assert res["err_px_p95"] > sc.GATE_P95


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_an_offset_of_one_lattice_period_is_inconsistent(pair, direction):
    res = sc.compare(
        plus(pair.maps()[direction], (float(sc.LATTICE), 0.0)), pair.dots, pair.render()
    )
    assert res["status"] == sc.INCONSISTENT
    assert res["n_found"] < 0.1 * res["n_dots"]  # the jitter: neighbours do not line up


def _strip(pair, direction, share):
    """True for the given share of the dots, along the left edge of the output."""
    where = pair.truth(pair.dots)[:, 0]
    cut = np.sort(where)[int(round(share * len(where)))]
    if direction == MOV_TO_REF:
        return lambda xy: (pair.truth(xy)[:, 0] < cut)[:, None]
    return lambda xy: (np.asarray(xy)[:, 0] < cut)[:, None]


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_a_strip_that_is_five_pixels_wrong_is_inconsistent(pair, direction):
    """8 % of the dots, along one edge: the whole-image gate at 90 % found passed this."""
    strip = _strip(pair, direction, 0.08)
    wrong = plus(pair.maps()[direction], lambda xy: strip(xy) * np.array([5.0, 0.0]))
    res = sc.compare(wrong, pair.dots, pair.render())
    assert res["status"] == sc.INCONSISTENT
    assert 0.90 * res["n_dots"] <= res["n_found"] < sc.GATE_FOUND * res["n_dots"]
    assert any(f.startswith("block") for f in res["failed"])


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_one_pixel_wrong_on_four_percent_of_dots_is_inconsistent(pair, direction):
    """Five neighbouring dots, 1 px off: every whole-image number passes; a block does not."""
    out_xy = pair.truth(pair.dots)
    block_centre = np.array([768.0, 256.0])
    near = np.argsort(np.linalg.norm(out_xy - block_centre, axis=1))[:5]
    assert len(near) / len(pair.dots) < 0.045

    def off(xy):
        at = pair.truth(xy) if direction == MOV_TO_REF else np.asarray(xy)
        d = np.linalg.norm(at[:, None] - out_xy[near][None], axis=2).min(axis=1)
        return (d < 2.0)[:, None] * np.array([1.0, 0.0])

    res = sc.compare(plus(pair.maps()[direction], off), pair.dots, pair.render())
    assert res["status"] == sc.INCONSISTENT
    assert res["failed"] == ["block (1, 0)"]
    assert res["n_found"] == res["n_dots"]
    assert res["err_px_p95"] < sc.GATE_P95 and abs(res["offset_px_mean_x"]) < 0.15


def test_a_map_that_covers_half_of_the_image_is_inconsistent(pair):
    """Half the dots are sent off the canvas: what is left matches, the image disagrees."""
    good = pair.maps()[MOV_TO_REF]
    half = PointMap(
        lambda xy: good.fn(xy) + (np.asarray(xy)[:, :1] > 640) * 1e6, MOV_TO_REF
    )
    res = sc.compare(half, pair.dots, pair.render())
    assert res["n_found"] == res["n_dots"] < 0.6 * res["n_drawn"]
    assert res["n_unexplained"] > 40
    assert res["status"] == sc.INCONSISTENT


def test_a_map_that_returns_nan_is_inconsistent(pair):
    nan = PointMap(lambda xy: np.full((len(xy), 2), np.nan), MOV_TO_REF)
    assert sc.compare(nan, pair.dots, pair.render())["status"] == sc.INCONSISTENT
    nan = PointMap(lambda xy: np.full((len(xy), 2), np.nan), REF_TO_MOV)
    assert sc.compare(nan, pair.dots, pair.render())["status"] == sc.INCONSISTENT


# ── an output the dots cannot be seen in is not checked ───────────────────────
def test_blank_constant_and_noise_only_outputs_are_not_checked(pair):
    rng = np.random.default_rng(0)
    outputs = {
        "zeros": np.zeros(HW, np.uint16),
        "constant": np.full(HW, 255, np.uint8),
        "noise": rng.normal(1600.0, 40.0, HW),
    }
    for name, out in outputs.items():
        for direction in DIRECTIONS:
            res = sc.compare(pair.maps()[direction], pair.dots, out)
            assert res["status"] == sc.NOT_CHECKED, name
            assert res["n_detected"] < 0.5 * res["n_drawn"]


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_weak_dots_are_not_checked_rather_than_inconsistent(pair, direction):
    """Dots at 5 x the noise s.d.: too weak to place, so nothing is claimed."""
    rng = np.random.default_rng(1)
    sd = 40.0
    out = pair.render() * (5.0 * sd / sc.DOT_PEAK) + rng.normal(1600.0, sd, HW)
    res = sc.compare(pair.maps()[direction], pair.dots, out)
    assert res["status"] == sc.NOT_CHECKED
    assert "detected" in res["cause"]


# ── a declared offset ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_a_declared_offset_is_required_to_match(pair, direction):
    moved = plus(pair.maps()[direction], (2.5, 2.5))
    out = pair.render()

    def status(offset):
        return sc.compare(moved, pair.dots, out, expected_offset=offset)["status"]

    assert status((2.5, 2.5)) == sc.OK
    assert status((0.0, 0.0)) == sc.INCONSISTENT
    assert status(None) == sc.INCONSISTENT
    assert status((2.0, 2.5)) == sc.INCONSISTENT
    assert status((-2.5, -2.5)) == sc.INCONSISTENT  # the sign is part of the claim


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_a_declared_offset_is_a_function_of_position(pair, direction):
    """An offset that grows across the slide: its mean, declared as a constant, fails."""

    def declared(xy):
        x = np.asarray(xy, dtype=float)[:, 0]
        return np.stack([3.0 + 2.0 * x / HW[1], 4.0 - 2.0 * x / HW[1]], axis=1)

    moved = plus(pair.maps()[direction], declared)
    out = pair.render()
    res = sc.compare(moved, pair.dots, out, expected_offset=declared)
    assert res["status"] == sc.OK and res["err_px_p95"] < 0.25
    assert res["declared_px_mean_x"] == pytest.approx(4.0, abs=0.2)
    constant = (res["declared_px_mean_x"], res["declared_px_mean_y"])
    assert sc.compare(moved, pair.dots, out, constant)["status"] == sc.INCONSISTENT


def test_masked_dots_are_left_out_and_counted(pair):
    """A mapping wrong only where the method says not to look (ASHLAR's overlap bands)."""
    for direction in DIRECTIONS:
        strip = _strip(pair, direction, 0.08)
        wrong = plus(pair.maps()[direction], lambda xy, s=strip: s(xy) * [5.0, 0.0])
        res = sc.compare(
            wrong, pair.dots, pair.render(), drop=lambda xy, s=strip: s(xy)[:, 0]
        )
        assert res["status"] == sc.OK, res["failed"]
        assert 8 <= res["n_masked"] <= 12
        assert sc.compare(wrong, pair.dots, pair.render())["status"] == sc.INCONSISTENT


def test_the_overall_status_is_the_worst_checked_one():
    assert sc.worst([sc.OK, sc.NO_STAGE_IMAGE]) == sc.OK
    assert sc.worst([sc.OK, sc.NOT_CHECKED]) == sc.NOT_CHECKED
    assert sc.worst([sc.CRASHED, sc.NOT_CHECKED, sc.OK]) == sc.CRASHED
    assert sc.worst([sc.CRASHED, sc.INCONSISTENT]) == sc.INCONSISTENT
    assert sc.worst([sc.NO_STAGE_IMAGE]) == sc.NOT_CHECKED  # nothing was checked
    assert set(sc.BLOCKING) == {sc.INCONSISTENT, sc.CRASHED, sc.NOT_CHECKED}
    assert sc.EXIT_CODE[sc.OK] == 0 and all(sc.EXIT_CODE[s] for s in sc.BLOCKING)


# ── run(): fake methods through the real driver ───────────────────────────────
def _round(case):
    cid = str(case.case_id)
    return int(cid.rsplit("_round", 1)[1]) if "_round" in cid else 1


def _truths(case):
    spec = sc.round_spec(case.ref_hw, case.mov_hw, 0, _round(case))
    from benchmarks.regbench.datasets import synthetic

    return synthetic.forward_map(spec), sc.inverse_of(spec)


def _draw(case, src_path, dest, inverse, shift=(0.0, 0.0), channels=None, stem="out"):
    """A writer: every channel of ``src_path`` resampled into the reference frame."""
    src = tifffile.imread(src_path)
    src = src[None] if src.ndim == 2 else src
    h, w = case.ref_hw
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    p = inverse(np.stack([xx.ravel(), yy.ravel()], axis=1) - np.asarray(shift))
    at = [p[:, 1].reshape(h, w), p[:, 0].reshape(h, w)]
    planes = [
        np.rint(np.clip(map_coordinates(c.astype(float), at, order=1), 0, 65535))
        for c in (src if channels is None else src[list(channels)])
    ]
    path = dest / f"{stem}.ome.tif"
    imageio.write_channels(
        path, np.asarray(planes, np.uint16), [f"c{i}" for i in range(len(planes))]
    )
    return {"files": [str(path)], "format": "ome-tiff", "channels": len(planes)}


def fake_method(
    name="fake",
    direction=MOV_TO_REF,
    wrong=(0.0, 0.0),
    stage=None,
    raises=None,
    writer=True,
    one_channel=False,
    declare=None,
    group=False,
    wrong_round=None,
):
    """A method whose writer draws through the truth and whose points are the truth plus
    ``wrong``. ``stage``: ``"yield"`` (a first variant with its image handed over at the
    yield), ``"hook"`` (the same through ``write_stage``) or ``"none"`` (no stage image)."""
    variants = (f"{name}_rigid", name) if stage else (name,)
    seen = {}

    def register(case, work, opts):
        seen.update(opts=dict(opts))
        truth, inverse = _truths(case)
        if raises is not None:
            raise raises
        fn = truth if direction == MOV_TO_REF else inverse
        off = wrong if wrong_round in (None, _round(case)) else (0.0, 0.0)
        pmap = PointMap(lambda xy: fn(xy) + np.asarray(off), direction)
        if stage:
            info = {}
            if stage == "yield" and opts.get("stage_images"):
                opts["monitor"].pause()
                d = work / "rigid"
                d.mkdir(parents=True, exist_ok=True)
                info["native"] = _draw(case, case.mov_image, d, inverse)
                opts["monitor"].resume()
            yield variants[0], pmap, info
        yield name, pmap, {"_state": {"case": case, "inverse": inverse}, "k": 1.0}

    def register_group(cases, work, opts):
        seen.setdefault("groups", []).append([c.case_id for c in cases])
        rounds = {}  # one state for the patient, as a per-patient method keeps it
        for case in cases:
            for variant, pmap, info in register(case, work, opts):
                if "_state" in info:
                    rounds[str(case.case_id)] = info["_state"]
                    info["_state"] = {"rounds": rounds}
                yield case.case_id, variant, pmap, info

    def write_native(state, case_id, dest_dir):
        channels = [0] if one_channel else None
        state = state.get("rounds", {}).get(str(case_id), state)
        case = state["case"]
        return _draw(
            case, case.mov_image, dest_dir, state["inverse"], channels=channels
        )

    extra = {}
    if writer:
        extra["write_native"] = write_native
    if stage == "hook":
        extra["write_stage"] = lambda state, case_id, variant, dest: _draw(
            state["case"], state["case"].mov_image, dest, state["inverse"]
        )
    if one_channel:
        extra["warp_extra"] = lambda state, case_id, variant, image, dest: (
            _draw(state["case"], image, dest, state["inverse"], stem="dots")
            if variant == name
            else None
        )
    if declare is not None:
        extra["SELFCHECK_OFFSET"] = lambda info: {name: declare}
    if group:
        extra["register_group"] = register_group
    else:
        extra["register"] = register
    return types.SimpleNamespace(
        NAME=name,
        SCOPE="group" if group else "pair",
        VARIANTS=variants,
        seen=seen,
        **extra,
    )


def run_small(method, tmp_path, **kw):
    return sc.run(method, tmp_path / "w", {"workers": 1}, sizes=[(SMALL, None)], **kw)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_run_reports_a_consistent_method_as_ok(tmp_path, direction):
    rec = run_small(fake_method(direction=direction), tmp_path)
    assert rec["status"] == sc.OK and rec["variants"] == {"fake": sc.OK}
    assert rec["blocks_the_method"] is False
    (p,) = rec["pairs"]
    assert (
        p["ref_hw"] == list(SMALL) and p["variants"]["fake"]["direction"] == direction
    )
    assert p["variants"]["fake"]["err_px_p95"] < 0.25
    saved = json.loads((tmp_path / "w" / "selfcheck.json").read_text())
    assert saved["status"] == sc.OK and saved["gate"]["found_min"] == 0.99
    assert not (tmp_path / "w" / "work").exists()  # the method's scratch is removed


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("wrong", [(0.3, 0.0), (40.0, 0.0)], ids=["third", "forty"])
def test_run_reports_a_method_whose_points_leave_its_image_as_inconsistent(
    tmp_path, direction, wrong
):
    rec = run_small(fake_method(direction=direction, wrong=wrong), tmp_path)
    assert rec["status"] == sc.INCONSISTENT and rec["blocks_the_method"] is True


def test_run_reports_a_method_declared_in_the_wrong_direction_as_inconsistent(tmp_path):
    """The method's writer and its points agree, but it says the points go the other way."""
    good = fake_method(direction=REF_TO_MOV)

    def register(case, work, opts):
        for variant, pmap, info in good.register(case, work, opts):
            yield variant, PointMap(pmap.fn, MOV_TO_REF), info

    liar = types.SimpleNamespace(**{**vars(good), "register": register})
    assert run_small(liar, tmp_path)["status"] == sc.INCONSISTENT


def test_run_reports_a_method_that_raises_as_crashed(tmp_path):
    rec = run_small(fake_method(raises=RuntimeError("no keypoints")), tmp_path)
    assert rec["status"] == sc.CRASHED
    assert "no keypoints" in rec["pairs"][0]["variants"]["fake"]["error"]


def test_out_of_memory_is_not_checked_never_inconsistent(tmp_path):
    rec = run_small(fake_method(raises=MemoryError("Java heap space")), tmp_path)
    assert rec["status"] == sc.NOT_CHECKED
    assert rec["pairs"][0]["variants"]["fake"]["cause"] == "oom"
    assert rec["blocks_the_method"] is True


def test_a_final_variant_without_an_image_is_not_checked(tmp_path):
    rec = run_small(fake_method(writer=False), tmp_path)
    assert rec["status"] == sc.NOT_CHECKED
    assert "no native image" in rec["pairs"][0]["variants"]["fake"]["cause"]


def test_a_stage_without_an_image_is_reported_and_does_not_block(tmp_path):
    rec = run_small(fake_method(stage="none"), tmp_path)
    assert rec["variants"] == {"fake_rigid": sc.NO_STAGE_IMAGE, "fake": sc.OK}
    assert rec["status"] == sc.OK and rec["blocks_the_method"] is False


@pytest.mark.parametrize("stage", ["yield", "hook"])
def test_a_stage_image_is_checked_when_the_method_can_draw_it(tmp_path, stage):
    method = fake_method(stage=stage)
    rec = run_small(method, tmp_path)
    assert rec["variants"] == {"fake_rigid": sc.OK, "fake": sc.OK}
    assert method.seen["opts"]["stage_images"] == "1"
    rec = run_small(fake_method(stage=stage, wrong=(0.5, 0.5)), tmp_path / "b")
    assert rec["variants"] == {"fake_rigid": sc.INCONSISTENT, "fake": sc.INCONSISTENT}


def test_dots_come_through_warp_extra_when_the_output_has_one_channel(tmp_path):
    rec = run_small(fake_method(one_channel=True), tmp_path)
    assert rec["status"] == sc.OK
    native = rec["pairs"][0]["variants"]["fake"]["native"]
    assert native["files"][0].endswith("dots.ome.tif")
    assert (tmp_path / "w" / "pair" / "dots.ome.tif").exists()
    # without the hook nothing can be checked, and it says so
    method = fake_method(one_channel=True)
    del method.warp_extra
    rec = run_small(method, tmp_path / "b")
    assert rec["status"] == sc.NOT_CHECKED
    assert "no dots channel" in rec["pairs"][0]["variants"]["fake"]["cause"]


def test_the_methods_declared_offset_is_subtracted(tmp_path):
    def declared(xy):
        return np.broadcast_to([2.5, 2.5], (len(xy), 2))

    rec = run_small(fake_method(wrong=(2.5, 2.5), declare=declared), tmp_path)
    assert rec["status"] == sc.OK
    assert rec["pairs"][0]["variants"]["fake"]["declared_offset"] is True
    rec = run_small(fake_method(wrong=(2.5, 2.5)), tmp_path / "b")
    assert rec["status"] == sc.INCONSISTENT


def test_a_per_patient_method_and_a_label(tmp_path):
    rec = run_small(fake_method(group=True, stage="none"), tmp_path, label="other")
    assert rec["variants"] == {"other_rigid": sc.NO_STAGE_IMAGE, "other": sc.OK}
    assert rec["label"] == "other"


def test_every_size_is_run_and_the_worst_counts(tmp_path):
    rec = sc.run(
        fake_method(),
        tmp_path / "w",
        {"workers": 1},
        sizes=[((512, 640), None), ((640, 768), (600, 700))],
    )
    assert [p["ref_hw"] for p in rec["pairs"]] == [[512, 640], [640, 768]]
    assert rec["pairs"][1]["mov_hw"] == [600, 700]
    assert rec["status"] == sc.OK
    assert (tmp_path / "w" / "640x768" / "pair" / "moving.ome.tif").exists()


def test_m0_from_and_mask_reach_the_method(tmp_path):
    method = fake_method()
    with pytest.raises(SystemExit, match="SELFCHECK_MASK"):
        run_small(method, tmp_path, mask="overlap")
    calls = []

    def mask(info, name):
        calls.append((name, info.get("k")))
        return lambda xy: np.asarray(xy)[:, 0] < 300

    method.SELFCHECK_MASK = mask
    rec = run_small(method, tmp_path, mask="overlap", m0_from=tmp_path / "stare")
    assert calls == [("overlap", 1.0)] and rec["mask"] == "overlap"
    assert rec["pairs"][0]["variants"]["fake"]["n_masked"] > 10
    assert method.seen["opts"]["m0_from"] == str(tmp_path / "stare")


def test_a_killed_selfcheck_process_is_not_checked_and_a_broken_one_crashed(
    tmp_path, monkeypatch
):
    size = ((512, 640), None)

    def with_exit(code):
        done = types.SimpleNamespace(returncode=code)
        monkeypatch.setattr(sc.subprocess, "run", lambda *a, **k: done)
        return sc._in_subprocess("stare", tmp_path, size, {}, None, None, None, 0)

    (killed,) = with_exit(-9)
    assert killed["status"] == sc.NOT_CHECKED and killed["case_id"] == "pair_512x640"
    assert with_exit(137)[0]["status"] == sc.NOT_CHECKED
    assert "out of memory" in with_exit(137)[0]["cause"]
    assert with_exit(1)[0]["status"] == sc.CRASHED  # exit 1 and no selfcheck.json
    # a patient whose process died: one record per round, none of them checked
    size = ((512, 640), None, (500, 600))
    lost = with_exit(-9)
    assert [r["case_id"] for r in lost] == [
        "patient_512x640_round1",
        "patient_512x640_round2",
    ]
    assert [r["mov_hw"] for r in lost] == [[512, 640], [500, 600]]
    assert {r["status"] for r in lost} == {sc.NOT_CHECKED}
    both = [{"status": sc.INCONSISTENT}, {"status": sc.OK}]
    write_json(tmp_path / "selfcheck.json", {"pairs": both})
    assert with_exit(1) == both  # every round's record comes back


def test_a_method_that_asks_for_it_gets_one_process_per_pair(tmp_path, monkeypatch):
    from benchmarks.regbench.methods import stare

    calls = []

    real = sc.subprocess.run

    def run(argv, **kw):
        if "--work" not in argv:  # another caller of the module (platform, uname)
            return real(argv, **kw)
        calls.append(argv)
        work = argv[argv.index("--work") + 1]
        write_json(
            f"{work}/selfcheck.json",
            {"pairs": [{"status": sc.OK, "variants": dict(rows)}]},
        )
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(stare, "SELFCHECK_SUBPROCESS", True, raising=False)
    monkeypatch.setattr(sc.subprocess, "run", run)
    # a case that lacks one of the run's rows was not checked on it
    rows = {"stare": {"status": sc.OK}}
    rec = sc.run("stare", tmp_path, {"tile": 1024}, sizes=sc.sizes_for("stare"))
    assert rec["status"] == sc.NOT_CHECKED and rec["variants"] == {
        "stare": sc.OK,
        "stare_rigid": sc.NOT_CHECKED,
    }
    assert all(p["variants"]["stare_rigid"]["filled"] for p in rec["pairs"])
    calls.clear()
    rows["stare_rigid"] = {"status": sc.OK}
    rec = sc.run("stare", tmp_path, {"tile": 1024}, sizes=sc.sizes_for("stare"))
    # its two pairs, then ASHLAR's two cases (the anchored rows read their matrices)
    assert rec["status"] == sc.OK and len(calls) == 4
    assert calls[1][calls[1].index("--size") + 1] == "4096x5120"
    assert calls[2][calls[2].index("--size") + 1] == "8192x10240"  # ASHLAR's pair
    # a patient is ONE process: every round's canvas is named
    patient = "8192x10240:8192x10240:8192x10240"
    assert calls[3][calls[3].index("--size") + 1] == patient
    assert Path(calls[3][calls[3].index("--work") + 1]).name == "patient_8192x10240"
    assert cli.build_parser().parse_args(calls[3][3:]).size == [patient]
    assert sc.parse_size(patient) == ((8192, 10240),) * 3
    assert "--in-process" in calls[0] and "tile=1024" in calls[0]
    assert cli.build_parser().parse_args(calls[0][3:]).in_process is True


def test_the_cli_runs_stare_for_real_and_exits_by_status(tmp_path, capsys):
    """`stare register` 1.2.1, its own STITCH, and the gate, on a small pair."""
    argv = ["selfcheck", "--method", "stare", "--work", str(tmp_path / "w")]
    assert cli.main([*argv, "--size", "1024x1280", "--opt", "workers=2"]) == 0
    rec = json.loads((tmp_path / "w" / "selfcheck.json").read_text())
    assert rec["variants"] == {"stare_rigid": sc.NO_STAGE_IMAGE, "stare": sc.OK}
    stare = rec["pairs"][0]["variants"]["stare"]
    assert stare["err_px_p95"] < 0.25 and stare["n_found"] == stare["n_dots"] > 100
    assert stare["native"]["writer"] == "stare.stages.stitch.main"
    assert "selfcheck [stare]: ok" in capsys.readouterr().out


def test_parse_size():
    assert sc.parse_size("1024x1280") == ((1024, 1280), None)
    assert sc.parse_size("40x60:30X50") == ((40, 60), (30, 50))
    assert sc.parse_size("10x20:30x40:50x60") == ((10, 20), (30, 40), (50, 60))
    for size in (((10, 20), None), ((10, 20), (30, 40))):
        assert sc.parse_size(sc._size_text(*size)) == size
    # a patient's rounds are written out: "same canvas" is the reference's size
    text = sc._size_text((10, 20), None, (5, 6))
    assert text == "10x20:10x20:5x6"
    assert sc.parse_size(text) == ((10, 20), (10, 20), (5, 6))
    for bad in ("1024", "1x2x3", "axb", "1x2:3x4:5x6:7x8:9x10"):
        with pytest.raises(SystemExit):
            sc.parse_size(bad)


# ── a patient: three slides, one native run ───────────────────────────────────
PATIENT = (SMALL, None, (500, 600))


def test_a_patient_shares_its_reference_and_each_round_has_its_own_map(tmp_path):
    one = sc.make_pair(tmp_path / "pair", SMALL)
    first, second = sc.make_group(tmp_path / "group", PATIENT)
    assert first["reference"] == second["reference"]
    assert Path(second["moving"]).name == "moving_2.ome.tif"
    # round 1 IS the pair: the same files, the same dots, the same map
    for name in ("reference", "moving"):
        assert np.array_equal(tifffile.imread(one[name]), tifffile.imread(first[name]))
    assert np.array_equal(one["dots_mov_xy"], first["dots_mov_xy"])
    assert (first["case_id"], first["group"]) == (
        "patient_512x640_round1",
        "patient_512x640",
    )
    assert (second["round"], second["rounds"], second["mov_hw"]) == (2, 2, (500, 600))
    # round 2: another pose, another field, other dots, the same tissue
    xy = np.array([[100.0, 100.0], [400.0, 300.0]])
    assert np.abs(first["truth"](xy) - second["truth"](xy)).min() > 5.0
    assert second["spec"]["scene_seed"] == first["spec"]["seed"] == 0
    assert len(second["dots_mov_xy"]) != len(first["dots_mov_xy"]) or not np.allclose(
        second["dots_mov_xy"], first["dots_mov_xy"]
    )
    back = second["truth_inverse"](second["truth"](xy))
    assert np.abs(back - xy).max() < 1e-3
    assert sc.case_names(PATIENT) == (
        "patient_512x640",
        ["patient_512x640_round1", "patient_512x640_round2"],
    )
    assert sc.case_folder(PATIENT) == "patient_512x640"
    assert sc.case_folder((SMALL, None)) == "512x640"
    assert sc.case_folder((SMALL, (500, 600)), {"512x640"}) == "512x640_mov_500x600"
    with pytest.raises(SystemExit, match="at most"):
        sc.round_spec(SMALL, None, 0, len(sc.ROUND_POSES) + 2)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_a_per_patient_method_gets_every_round_in_one_run(tmp_path, direction):
    method = fake_method(group=True, direction=direction)
    rec = sc.run(method, tmp_path / "w", {"workers": 1}, sizes=[PATIENT])
    # ONE native run with both rounds, not one per round
    assert method.seen["groups"] == [
        ["patient_512x640_round1", "patient_512x640_round2"]
    ]
    assert rec["status"] == sc.OK and rec["variants"] == {"fake": sc.OK}
    first, second = rec["pairs"]  # one entry per ROUND
    assert (first["round"], second["round"]) == (1, 2)
    assert first["group"] == second["group"] == "patient_512x640"
    assert second["case_id"] == "patient_512x640_round2"
    assert second["mov_hw"] == [500, 600] and first["mov_hw"] == list(SMALL)
    for r in (first, second):  # each round against the gate, on its own dots
        v = r["variants"]["fake"]
        assert v["status"] == sc.OK and v["n_found"] >= 0.99 * v["n_dots"] > 20
        assert v["err_px_p95"] < 0.25
    assert first["n_drawn"] != second["n_drawn"]
    assert rec["cases"] == [
        {
            "folder": "",
            "group": "patient_512x640",
            "size": "512x640:512x640:500x600",
            "rounds": 2,
            "patient": True,
            "status": sc.OK,
        }
    ]


def test_a_second_round_that_is_wrong_blocks_the_method(tmp_path):
    """The round a pair never had: right on round 1, 0.6 px off on round 2."""
    method = fake_method(group=True, wrong=(0.6, 0.0), wrong_round=2)
    rec = sc.run(method, tmp_path / "w", {"workers": 1}, sizes=[PATIENT])
    first, second = rec["pairs"]
    assert first["status"] == sc.OK and second["status"] == sc.INCONSISTENT
    assert rec["status"] == sc.INCONSISTENT and rec["blocks_the_method"]
    assert rec["variants"] == {"fake": sc.INCONSISTENT}
    assert rec["cases"][0]["status"] == sc.INCONSISTENT
    assert "patient_512x640" not in rec["attempts"][-1]["why"]  # names the variant


def test_a_pair_method_takes_a_patient_round_by_round(tmp_path):
    method = fake_method(one_channel=True)  # and its dots image is the round's own
    sizes = [(SMALL, None), PATIENT]
    rec = sc.run(method, tmp_path / "w", {"workers": 1}, sizes=sizes)
    assert rec["status"] == sc.OK, rec
    assert [p["case_id"] for p in rec["pairs"]] == [
        "pair_512x640",
        "patient_512x640_round1",
        "patient_512x640_round2",
    ]
    assert [c["folder"] for c in rec["cases"]] == ["512x640", "patient_512x640"]
    out = tmp_path / "w" / "patient_512x640" / "out" / "fake" / "selfcheck"
    assert sorted(p.name for p in out.iterdir()) == [
        "patient_512x640_round1",
        "patient_512x640_round2",
    ]
    assert (tmp_path / "w" / "patient_512x640" / "pair" / "dots_2.ome.tif").exists()


def test_a_patient_that_raises_is_crashed_on_every_round(tmp_path):
    method = fake_method(group=True, raises=RuntimeError("no keypoints"))
    rec = sc.run(method, tmp_path / "w", {"workers": 1}, sizes=[PATIENT])
    assert [p["status"] for p in rec["pairs"]] == [sc.CRASHED, sc.CRASHED]


# ── realcheck ─────────────────────────────────────────────────────────────────
CROP = 768


class Real:
    """A real-like case: nuclei segmented on each native slide, with 10 % of them missed
    and about 1 px of centroid jitter, and the image a writer draws through the truth."""

    def __init__(self, pair, root):
        from benchmarks.regbench.datasets import synthetic

        self.pair = pair
        rng = np.random.default_rng(11)
        sets = {}
        for side, path in (("ref", pair.d["reference"]), ("mov", pair.d["moving"])):
            ps = cells.segment_slide(path, 0, synthetic.NUCLEUS_DIAMETER)
            ps = ps.take(np.flatnonzero(rng.random(len(ps)) >= 0.10))
            jitter = rng.normal(0.0, 0.7, (len(ps), 2))
            sets[side] = ps.with_xy(ps.xy + np.repeat(jitter, np.diff(ps.off), axis=0))
        case = sc.pair_case(pair.d, root / "cases")
        write_case(root / "cases", case, cells_mov=sets["mov"], cells_ref=sets["ref"])
        self.case = case
        self.points = load_points(case)
        out = np.rint(np.clip(pair.render(channel=0), 0, 65535)).astype(np.uint16)
        self.native = {
            "files": [
                str(
                    imageio.write_channels(
                        root / "registered.ome.tif", out[None], ["DAPI"]
                    )
                )
            ]
        }
        self.root = root

    def check(self, pmap, name, **kw):
        d = self.root / name
        d.mkdir(exist_ok=True)
        return sc.realcheck(self.case, d, pmap, self.native, crop=CROP, **kw)


@pytest.fixture(scope="module")
def real(pair, tmp_path_factory):
    return Real(pair, tmp_path_factory.mktemp("realcheck"))


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_realcheck_does_not_flag_a_correct_mapping(real, direction):
    res = real.check(real.pair.maps()[direction], f"ok_{direction}")
    assert res["status"] == "ok" and res["flag"] is False
    assert len(res["crops"]) == 5 and all(c["judged"] for c in res["crops"])
    for c in res["crops"]:
        assert c["n"] >= 50
        assert max(abs(c["offset_px_mean_x"]), abs(c["offset_px_mean_y"])) < 0.2
        assert c["matched_fraction"] > 0.9 * c["baseline_fraction"] > 0.6
    saved = json.loads((real.root / f"ok_{direction}" / "realcheck.json").read_text())
    assert saved["flag"] is False and saved["direction"] == direction


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_realcheck_flags_half_a_pixel_by_the_signed_mean(real, direction):
    res = real.check(plus(real.pair.maps()[direction], (0.5, 0.5)), f"half_{direction}")
    assert res["status"] == "flagged" and res["flag"] is True
    assert all(c["why"] == ["offset"] for c in res["crops"])
    for c in res["crops"]:
        assert c["offset_px_mean_x"] == pytest.approx(0.5, abs=0.2)
        assert c["offset_px_mean_y"] == pytest.approx(0.5, abs=0.2)


def test_realcheck_flags_a_flipped_direction_by_the_matched_fraction(real):
    flipped = {
        MOV_TO_REF: PointMap(real.pair.inverse, MOV_TO_REF),
        REF_TO_MOV: PointMap(real.pair.truth, REF_TO_MOV),
    }
    for direction in DIRECTIONS:
        res = real.check(flipped[direction], f"flip_{direction}")
        assert res["flag"] is True
        assert all("matched_fraction" in c["why"] for c in res["crops"])


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_realcheck_subtracts_the_declared_offset(real, direction):
    moved = plus(real.pair.maps()[direction], (2.5, 2.5))

    def declared(xy):
        return np.broadcast_to([2.5, 2.5], (len(xy), 2))

    assert real.check(moved, "declared", offset=declared)["flag"] is False
    assert real.check(moved, "undeclared")["flag"] is True
    assert real.check(moved, "wrong", offset=(2.0, 2.5))["flag"] is True


def test_realcheck_of_an_empty_output_is_not_checked(real):
    blank = imageio.write_channels(
        real.root / "blank.ome.tif", np.zeros((1, *HW), np.uint16), ["DAPI"]
    )
    d = real.root / "blank"
    d.mkdir()
    res = sc.realcheck(
        real.case, d, real.pair.maps()[MOV_TO_REF], {"files": [str(blank)]}, crop=CROP
    )
    assert res["status"] == sc.NOT_CHECKED and res["flag"] is False


def _real_method(real, wrong=(0.0, 0.0), declare=None):
    def register(case, work, opts):
        pmap = plus(real.pair.maps()[MOV_TO_REF], wrong)
        yield "rm", pmap, {"_state": {}, "k": 2}

    def write_native(state, case_id, dest_dir):
        return real.native

    extra = {"SELFCHECK_OFFSET": (lambda info: {"rm": declare})} if declare else {}
    return types.SimpleNamespace(
        NAME="rm",
        SCOPE="pair",
        VARIANTS=("rm",),
        register=register,
        write_native=write_native,
        **extra,
    )


def test_the_driver_runs_the_realcheck_after_the_writer(real, tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "REAL_CROP", CROP)
    out = tmp_path / "out"
    methods.run_case(_real_method(real), real.case, out, realcheck=True)
    d = result_dir(out, "rm", real.case)
    run = json.loads((d / "run.json").read_text())
    assert run["realcheck"] == {"status": "ok", "flag": False}
    assert json.loads((d / "realcheck.json").read_text())["crop_px"] == CROP

    methods.run_case(_real_method(real, (0.5, 0.5)), real.case, out, realcheck=True)
    run = json.loads((d / "run.json").read_text())
    assert run["ok"] is True and run["realcheck"] == {"status": "flagged", "flag": True}
    args = types.SimpleNamespace(out=out, dataset=real.case.dataset)
    assert cli._realcheck_flagged(args, [real.case], {real.case.case_id: ["rm"]})

    def declared(xy):
        return np.broadcast_to([0.5, 0.5], (len(xy), 2))

    method = _real_method(real, (0.5, 0.5), declared)
    methods.run_case(method, real.case, out, realcheck=True)
    assert json.loads((d / "run.json").read_text())["realcheck"]["flag"] is False
    assert not cli._realcheck_flagged(args, [real.case], {real.case.case_id: ["rm"]})

    methods.run_case(_real_method(real), real.case, tmp_path / "plain")
    plain = result_dir(tmp_path / "plain", "rm", real.case)
    assert "realcheck" not in json.loads((plain / "run.json").read_text())
    assert not (plain / "realcheck.json").exists()


def test_a_realcheck_that_raises_is_recorded_and_the_row_stays(real, tmp_path):
    method = _real_method(real)
    method.write_native = lambda state, case_id, dest: {"files": ["/nowhere.tif"]}
    methods.run_case(method, real.case, tmp_path, realcheck=True)
    run = json.loads((result_dir(tmp_path, "rm", real.case) / "run.json").read_text())
    assert run["ok"] is True and run["realcheck"]["status"] == "crashed"
    assert run["realcheck"]["flag"] is None


def test_run_takes_the_realcheck_flag():
    argv = ["run", "--cases", "c", "--out", "o", "--dataset", "multiplex"]
    argv += ["--method", "stare", "--index", "0"]
    assert cli.build_parser().parse_args(argv).realcheck is False
    assert cli.build_parser().parse_args([*argv, "--realcheck"]).realcheck is True


def test_realcheck_of_a_case_without_nuclei_is_not_checked(pair, tmp_path):
    case = sc.pair_case(pair.d, tmp_path / "cases")
    res = sc.realcheck(case, tmp_path, pair.maps()[MOV_TO_REF], {"files": ["x"]})
    assert res["status"] == sc.NOT_CHECKED and "no nuclei" in res["cause"]
    assert json.loads((tmp_path / "realcheck.json").read_text())["flag"] is False
