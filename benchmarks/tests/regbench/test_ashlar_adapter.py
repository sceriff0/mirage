"""The ASHLAR adapter, without ASHLAR.

ASHLAR (pinned at 1.20.0) is not installed where this suite runs, so everything here runs
against ``fake_ashlar``: a recording stand-in that keeps ASHLAR's geometry conventions and
draws the mosaic the way its writer is written to (see that module). What these tests pin:

* the command is ASHLAR's own, with nothing but the documented flags;
* the point map against the emulated writer, in the selfcheck's own comparisons, with a
  reference that is NOT stitched onto its slide's grid; and that a wrong origin, a wrong
  direction, a swapped tile index and a one-tile offset are each reported
  ``inconsistent``;
* the four rows: noise on the reference tiles or none, moving tiles on the grid or placed
  by STARE's anchor; a missing anchor fails its round alone; reference tiles cut once;
* an exception out of ASHLAR fails the patient with its message; the clock is paused
  while tiles are cut and while ASHLAR's writer runs.

What they cannot pin is ASHLAR itself: ``test_external_methods.py`` holds the test that
needs the real package, skipped here.
"""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile

from benchmarks.ashlar import original, retile
from benchmarks.regbench import METHODS, methods, tilemap
from benchmarks.regbench import selfcheck as sc
from benchmarks.regbench.cases import (
    Case,
    load_case,
    load_points,
    result_dir,
    write_case,
)
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap
from benchmarks.regbench.methods import ashlar as am
from benchmarks.regbench.score import score

from . import fake_ashlar as fa
from .test_pipeline import squares

SMALL = (768, 1024)  # with 256 px tiles: a 4 x 5 grid
SMALL_TILE = 256


def off_grid(n_rows, n_cols):
    """A reference stitch that is NOT on the slide's grid: every tile 4 px down and 7 px
    right of where it was cut, except the first (so the origin is not zero) and one tile
    another 0.3 px off (so the tiles do not share one offset either)."""
    shifts = np.zeros((n_rows * n_cols, 2))
    shifts[0] = [-4.0, -7.0]
    shifts[n_cols + 1] = [0.3, -0.2]
    return shifts


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    """The fake ``ashlar`` for one test, gone afterwards."""
    fa.install(monkeypatch.setitem, edge_shifts=off_grid)
    yield fa
    fa.configure()


@pytest.fixture
def small(monkeypatch):
    """256 px tiles, so a 768 x 1024 pair is a 4 x 5 grid and a run takes a second."""
    monkeypatch.setattr(am, "TILE", SMALL_TILE)
    return [(SMALL, None)]


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    return sc.make_pair(tmp_path_factory.mktemp("ashlar_pair"), SMALL, ref_dots=True)


def case_of(root, pair, case_id, mov, dataset="selfcheck", cells=None, **extra):
    h, w = pair["ref_hw"]
    case = Case(
        dataset=dataset,
        case_id=case_id,
        group="P1",
        ref_nuclear=pair["reference"],
        mov_nuclear=str(mov),
        ref_image=pair["reference"],
        mov_image=str(mov),
        modality="fluorescence",
        diagonal=math.hypot(h, w),
        pixel_size_um=pair["pixel_size_um"],
        ref_hw=list(pair["ref_hw"]),
        mov_hw=list(pair["mov_hw"]),
        extra={"ref_channel": 0, "mov_channel": 0, **extra},
    )
    kw = {} if cells is None else {"cells_mov": cells[0], "cells_ref": cells[1]}
    return load_case(write_case(root, case, **kw))


def patient(tmp_path, pair, rounds=2, **kw):
    cases = []
    for i in range(rounds):
        mov = tmp_path / f"round{i + 1}.ome.tif"
        shutil.copyfile(pair["moving"], mov)
        cases.append(case_of(tmp_path / "cases", pair, f"P1_r{i + 1}", mov, **kw))
    return cases


def run_json(out, variant, case):
    return json.loads((result_dir(out, variant, case) / "run.json").read_text())


def stare_run(out, case, m0=None, ok=True):
    """What a ``stare`` run of this round leaves for the anchored rows."""
    d = result_dir(out, "stare", case)
    d.mkdir(parents=True, exist_ok=True)
    info = {} if m0 is None else {"m0": np.asarray(m0).tolist()}
    (d / "run.json").write_text(json.dumps({"ok": ok, "info": info}))
    return d / "run.json"


M0 = np.array([[1.0, 0.0, 18.0], [0.0, 1.0, -12.0], [0.0, 0.0, 1.0]])


# ── names and settings ────────────────────────────────────────────────────────
def test_registered_and_named():
    assert "ashlar" in METHODS and am.SCOPE == "group" and am.VARIANTS == ("ashlar",)
    assert methods.scope_of("ashlar") == "group"
    for label in ("ashlar_clean", "affine_ashlar", "affine_ashlar_clean"):
        assert methods.variant_names("ashlar", label) == [label]
    assert methods.variant_names("ashlar") == ["ashlar"]
    assert not hasattr(am, "register")
    # frozen values of the plan
    # the cohort scanner's tile and the owner's 10 % overlap
    assert (am.TILE, am.OVERLAP, am.MAX_SHIFT_UM) == (1950, 0.1, 15.0)
    assert (am.NOISE_FRAC, am.NOISE_MIN_COUNTS) == (0.002, 3.0)
    assert sc.sizes_for("ashlar_clean") == sc.sizes_for("affine_ashlar_clean")
    # one canvas (5 x 6 tiles of 1950 px), as a pair and as a patient of three slides
    hw = (8192, 10240)
    assert sc.sizes_for("ashlar") == [(hw, None), (hw, None, None)]
    assert sc.sizes_for("ashlar") == sc.sizes_for("affine_ashlar_clean")


def test_the_primary_row_is_slight_noise_and_no_anchor():
    assert am.settings({}) == (0.002, "none", 15.0)
    assert am.settings({"label": ""}) == (0.002, "none", 15.0)
    # the four names that predate the factorial keep their cells
    old = ("ashlar", "ashlar_clean", "affine_ashlar", "affine_ashlar_clean")
    kept = {n: am.ASHLAR_CELLS[n] for n in old}
    assert kept == {
        "ashlar": ("none", "0.002", "15"),
        "ashlar_clean": ("none", "0", "15"),
        "affine_ashlar": ("stare", "0.002", "15"),
        "affine_ashlar_clean": ("stare", "0", "15"),
    }
    # the factorial is every combination of the three factor tables
    assert len(am.ASHLAR_CELLS) == len(am.AFFINES) * len(am.NOISE_FRACS) * len(
        am.MAX_SHIFTS_UM
    )
    assert set(am.ASHLAR_CELLS.values()) == {
        (a, f, s_)
        for a in am.AFFINES
        for f in am.NOISE_FRACS
        for s_ in am.MAX_SHIFTS_UM
    }


def test_the_noise_option_is_still_accepted_as_an_alias_of_noise_frac():
    free = {"label": "mine"}
    assert am.settings({"noise": "ref", **free}) == am.settings(
        {"noise_frac": "0.002", **free}
    )
    assert am.settings({"noise": "none", **free}) == am.settings(
        {"noise_frac": "0", **free}
    )
    with pytest.raises(ValueError, match="disagree"):
        am.settings({"noise": "none", "noise_frac": "0.01", **free})
    assert am.settings(
        {"noise_frac": "0.05", "max_shift_um": "240", "label": "mine"}
    ) == (
        0.05,
        "none",
        240.0,
    )


@pytest.mark.parametrize(
    "opts",
    [
        {"label": "ashlar", "noise": "none"},
        {"label": "ashlar_clean"},
        {"label": "affine_ashlar", "noise": "ref"},
        {"label": "affine_ashlar_clean", "affine": "stare"},
        {"label": "", "affine": "stare"},
    ],
)
def test_a_label_and_its_options_come_together(opts):
    with pytest.raises(ValueError, match="come together"):
        am.settings(opts)


def test_an_unknown_setting_is_refused():
    with pytest.raises(ValueError, match="noise="):
        am.settings({"noise": "lots", "label": "mine"})
    with pytest.raises(ValueError, match="max_shift_um="):
        am.settings({"max_shift_um": "30", "label": "mine"})
    with pytest.raises(ValueError, match="noise_frac="):
        am.settings({"noise_frac": "0.5", "label": "mine"})
    assert am.settings({"noise": "none", "affine": "stare", "label": "mine"}) == (
        0.0,
        "stare",
        15.0,
    )


def test_the_module_imports_without_ashlar():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from benchmarks.regbench.methods import ashlar\n"
        "raise SystemExit(int('ashlar' in sys.modules))\n"
    )
    root = Path(__file__).resolve().parents[3]
    assert subprocess.run([sys.executable, "-c", code], cwd=root).returncode == 0


# ── geometry read back from ASHLAR's notes ────────────────────────────────────
def _grid(n_rows=2, n_cols=3, stride=230, tile=256):
    true = [[r * stride, c * stride] for r in range(n_rows) for c in range(n_cols)]
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "stride": stride,
        "tile_size": tile,
        "true_positions_yx": true,
    }


def test_the_tile_order_is_read_from_ashlars_positions_not_assumed():
    grid = _grid()
    meta = np.array(grid["true_positions_yx"], dtype=float)
    np.testing.assert_array_equal(am.tile_order(meta, grid), np.arange(6))
    # the same tiles handed over column by column are found where they belong
    by_column = meta.reshape(2, 3, 2).transpose(1, 0, 2).reshape(-1, 2)
    np.testing.assert_array_equal(am.tile_order(by_column, grid), [0, 3, 1, 4, 2, 5])
    with pytest.raises(RuntimeError, match="are not the 2 x 3 grid"):
        am.tile_order(meta[:5], grid)
    with pytest.raises(RuntimeError, match="are not the 2 x 3 grid"):
        am.tile_order(meta + 40.0, grid)
    with pytest.raises(RuntimeError, match="are not the 2 x 3 grid"):
        am.tile_order(np.repeat(meta[:1], 6, axis=0), grid)


def _placement():
    c = np.array([[0.0, 0.0], [0.0, 230.0]])
    return {
        "tile": 256,
        "ref_true_yx": c.tolist(),
        "ref_pos_yx": (c + [[4.0, 7.0], [4.0, 7.0]]).tolist(),
        "mov_true_yx": c.tolist(),
        "mov_pos_yx": (c + [[10.0, 30.0], [12.0, 18.0]]).tolist(),
        "mosaic_shape": [260, 493],
    }


def test_the_origin_is_where_the_mosaics_first_pixel_sits_on_the_reference_slide():
    p = _placement()
    assert am.mosaic_origin_xy(p) == [-7.0, -4.0]
    # the reference slide (200 x 480) inside the mosaic, in the mosaic's own pixels
    assert am._valid_box(p, (200, 480)) == [7, 4, 487, 204]
    fn = am.point_map(p)
    assert fn.direction == MOV_TO_REF
    # +30 right and +10 down into the mosaic, then -7 and -4 back to the reference slide
    np.testing.assert_allclose(fn(np.array([[100.0, 100.0]])), [[123.0, 106.0]])


def _notes(shifts, errors, max_shift_px=46.0):
    """An edge note on the grid and a layer note with the given shifts and errors, the
    way ASHLAR would have left them (the discard flag from its own rule)."""
    shifts = np.asarray(shifts, dtype=float)
    errors = np.asarray(errors, dtype=float)
    n = len(shifts)
    meta = np.array([[230.0 * (i // 2), 230.0 * (i % 2)] for i in range(n)])
    edge = {
        "metadata_positions_yx": meta.tolist(),
        "positions_yx": meta.tolist(),
        "model_coef": np.eye(2).tolist(),
        "model_intercept": [0.0, 0.0],
    }
    zero = (np.rint(np.abs(shifts) * 10) / 10 == 0).all(axis=1)
    first = zero | np.isinf(errors)
    offset = np.median(shifts[~first], axis=0) if (~first).any() else 0.0
    over = np.linalg.norm(shifts - offset, axis=1) > max_shift_px
    layer = {
        "metadata_positions_yx": meta.tolist(),
        "shifts_yx": shifts.tolist(),
        "cycle_offset_yx": [0.0, 0.0],
        "reference_idx": list(range(n)),
        "errors": [None if np.isinf(e) else float(e) for e in errors],
        "discard": (first | over).tolist(),
    }
    return layer, edge


def test_discards_are_counted_once_each_by_a_fixed_precedence():
    # tile 0: no registration AND zero shift -> infinite error wins
    # tile 1: zero to 0.1 px -> zero offset; tile 2: 300 px off -> over budget
    # tiles 3 to 5: normal tiles
    shifts = [[0.0, 0.0], [0.04, -0.03], [300.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5, 5]]
    layer, edge = _notes(shifts, [np.inf, 0.2, 0.2, 0.2, 0.2, 0.2])
    assert layer["discard"] == [True, True, True, False, False, False]
    got = am.discards(layer, edge, 46.0)
    assert got == {
        "discard_infinite_error": 1,
        "discard_zero_offset": 1,
        "discard_over_budget": 1,
        "n_tiles": 6,
        "n_discarded": 3,
        "discard_frac": 0.5,
        "cycle_fully_modelled": False,
        "discard_rebuild_matches": True,
    }
    # an infinite error far over the budget is still counted once, as infinite
    layer, edge = _notes(
        [[300.0, 5.0], [5, 5], [5, 5], [5, 5]], [np.inf, 0.2, 0.2, 0.2]
    )
    got = am.discards(layer, edge, 46.0)
    assert (got["discard_infinite_error"], got["discard_over_budget"]) == (1, 0)


def test_a_rebuild_that_disagrees_with_ashlars_flag_says_so():
    layer, edge = _notes([[5.0, 5.0]] * 4, [0.2] * 4)
    layer["discard"][2] = True  # ASHLAR discarded a tile this reading cannot explain
    assert not am.discards(layer, edge, 46.0)["discard_rebuild_matches"]


# ── the selfcheck against the emulated writer ────────────────────────────────
def _selfcheck(tmp_path, sizes, **kw):
    rec = sc.run("ashlar", tmp_path, mask="overlap", sizes=sizes, **kw)
    (pair_rec,) = rec["pairs"]
    (name,) = pair_rec["variants"]
    return rec, pair_rec["variants"][name]


def test_the_map_follows_the_mosaic_at_ashlars_own_tile_size(tmp_path):
    """The frozen geometry (1950 px tiles, 1755 px pitch), a reference stitched off its
    slide's grid: 3 x 3 tiles on 4096 x 5120. (Smaller canvases, whose last tiles are
    slivers: `test_sliver_tiles_on_both_axes`.)"""
    assert (am.TILE, retile._stride(am.TILE, am.OVERLAP)) == (1950, 1755)
    rec, v = _selfcheck(tmp_path, [(sc.LARGE_HW, None)])
    assert rec["status"] == sc.OK and not rec["not_ok"], v
    assert v["direction"] == MOV_TO_REF and v["n_masked"] > 0
    assert v["n_found"] == v["n_dots"] > 300 and v["err_px_p95"] < 0.2
    # both comparisons read ASHLAR's own file, not the per-cycle copy
    native, ref = v["native"], v["reference"]
    assert Path(native["dots_file"]).name == original.OUTPUT_NAME
    assert ref["file"] == native["dots_file"] and ref["channel"] == 1
    assert native["dots_channel"] == 3 and native["mosaic_planes"] == [2, 4]
    # the mosaic is not on the reference slide's grid, and says where it is
    assert native["origin_xy"] == [-7.0, -4.0] == ref["origin_xy_reported"]
    assert ref["image_origin_px_median_x"] == pytest.approx(-7.0, abs=0.1)
    assert ref["image_origin_px_median_y"] == pytest.approx(-4.0, abs=0.1)
    assert native["valid_box"] == [7, 4, 5127, 4100]
    assert ref["status"] == sc.OK and ref["n_found"] == ref["n_dots"] > 300
    assert v["declared_offset"] and abs(v["declared_px_mean_x"]) > 0.5
    # the copy holds the mosaic's planes, pixel for pixel
    with tifffile.TiffFile(native["mosaic"]) as tf:
        planes = tf.series[0].asarray()
    np.testing.assert_array_equal(tifffile.imread(native["files"][0]), planes[2:4])
    np.testing.assert_array_equal(tifffile.imread(native["reference_file"]), planes[:2])


# Canvases whose last row and column of 1950 px tiles are slivers, as on every cohort
# slide (about 60000 x 80000 px, never a multiple of the 1755 px pitch). Per canvas: the
# grid, how much slide the last row and the last column hold, and the mosaic's origin
# with the stand-in's stitch (`off_grid`: the first tile 4 px up and 7 px left of all
# the others).
SLIVERS = {
    # thin on both axes: the last tiles hold 245 and 305 px, 50 and 110 px of them in
    # no other tile. The first tile holds most of the slide, so the origin is ITS offset
    (2000, 2060): ((2, 2), (245, 305), [0.0, 0.0]),
    # the last row and column hold 150 and 135 px, ALL of them also in the tile before
    (1905, 5400): ((2, 4), (150, 135), [-7.0, -4.0]),
}


def _cover(origins_yx, tile, xy, grow):
    """How many tile squares, each grown by ``grow`` px, hold each point: plain
    rectangle tests, no nearest-neighbour search."""
    o = np.asarray(origins_yx, dtype=float)
    x, y = xy[:, 0][:, None], xy[:, 1][:, None]
    inside = (
        (x > o[None, :, 1] - grow)
        & (x < o[None, :, 1] + tile + grow)
        & (y > o[None, :, 0] - grow)
        & (y < o[None, :, 0] + tile + grow)
    )
    return inside.sum(axis=1)


def _one(origins_yx, tile, xy, margin):
    """Inside one tile by more than ``margin`` and outside every other by as much."""
    deep = _cover(origins_yx, tile, xy, -margin) >= 1
    return deep & (_cover(origins_yx, tile, xy, margin) == 1)


@pytest.mark.parametrize("hw", sorted(SLIVERS), ids=lambda hw: f"{hw[0]}x{hw[1]}")
def test_sliver_tiles_on_both_axes(tmp_path, hw):
    """The map, the seam mask, the origin and the valid box on a canvas whose last row
    and column of tiles are mostly padding, at the frozen tile size."""
    (n_rows, n_cols), (last_h, last_w), origin = SLIVERS[hw]
    h, w = hw
    pitch = retile._stride(am.TILE, am.OVERLAP)
    assert (am.TILE, pitch) == (1950, 1755)
    grid = retile.tile_grid(w, h, am.TILE, am.OVERLAP)
    assert retile.grid_shape(w, h, am.TILE, am.OVERLAP) == (n_rows, n_cols)
    assert (grid[-1].h, grid[-1].w) == (last_h, last_w)
    assert last_h < am.TILE // 4 and last_w < am.TILE // 2  # slivers

    rec, v = _selfcheck(tmp_path, [(hw, None)], keep_work=True)
    assert rec["status"] == sc.OK and not rec["not_ok"], v
    assert v["n_found"] == v["n_dots"] > 50 and v["err_px_p95"] < 0.2
    native, ref = v["native"], v["reference"]
    assert ref["status"] == sc.OK and ref["n_found"] == ref["n_dots"] > 50
    # the origin: the offset of the tiles that hold most of the slide, as the image
    # itself shows it, and the slide's extent in the mosaic from it
    assert native["origin_xy"] == origin == ref["origin_xy_reported"]
    assert ref["image_origin_px_median_x"] == pytest.approx(origin[0], abs=0.1)
    assert ref["image_origin_px_median_y"] == pytest.approx(origin[1], abs=0.1)
    ox, oy = (int(-v) for v in origin)
    assert native["valid_box"] == [ox, oy, ox + w, oy + h]
    # (the stand-in's one tile another 0.3 px down is in the last row: 4.3, rounded up)
    assert native["mosaic_shape"] == [
        (n_rows - 1) * pitch + am.TILE + 5,
        (n_cols - 1) * pitch + am.TILE + 7,
    ]

    # the seam mask, against plain rectangle tests on the placement the run recorded
    (path,) = (tmp_path / "out").rglob("run.json")
    p = json.loads(path.read_text())["info"]["placement"]
    assert len(p["mov_pos_yx"]) == n_rows * n_cols and p["tile"] == am.TILE
    dots = sc.lattice_dots(hw, 0)
    # the tile nearest to each dot: the one that holds it, when only one does
    tile_of = tilemap._signed_depths(p["mov_true_yx"], am.TILE, dots, 1)[1][:, 0]
    d_mov = np.array(p["mov_pos_yx"]) - np.array(p["mov_true_yx"])
    m = dots + d_mov[tile_of, ::-1]
    margin = am.DOT_MARGIN_PX
    clean = (
        _one(p["mov_true_yx"], am.TILE, dots, margin)
        & _one(p["mov_pos_yx"], am.TILE, m, margin)
        & _one(p["ref_pos_yx"], am.TILE, m, margin)
    )
    seam = tilemap.on_seam(p, dots, margin)
    np.testing.assert_array_equal(seam, ~clean)
    assert v["n_masked"] == int(seam.sum()) and 0 < v["n_masked"] < len(dots)
    # every dot a second tile also holds is masked, the slivers' included
    twice = _cover(p["mov_true_yx"], am.TILE, dots, 0.0) > 1
    assert twice.any() and seam[twice].all()
    # and dots only a sliver tile holds are JUDGED where it has pixels of its own
    judged = {int(t) for t in tile_of[~seam]}
    own_w, own_h = last_w - (am.TILE - pitch), last_h - (am.TILE - pitch)
    if own_w > 2 * sc.LATTICE:
        assert n_cols - 1 in judged  # the first row's last tile
    if own_w <= 0 and own_h <= 0:  # nothing of their own: all of it is band
        assert judged == set(range(n_cols - 1))
        assert seam[(dots[:, 1] > pitch) | (dots[:, 0] > (n_cols - 1) * pitch)].all()
    # the map itself, at dots only one tile holds: that tile's two translations
    fn = am.point_map(p)
    lone = np.flatnonzero(~seam)
    in_mosaic = m[lone]
    ref_tile = tilemap._signed_depths(p["ref_pos_yx"], am.TILE, in_mosaic, 1)[1][:, 0]
    d_ref = (np.array(p["ref_true_yx"]) - np.array(p["ref_pos_yx"]))[ref_tile]
    np.testing.assert_allclose(fn(dots[lone]), in_mosaic + d_ref[:, ::-1], atol=1e-9)


def test_the_origin_is_weighted_by_the_slide_each_tile_holds():
    """Three sliver tiles agreeing with each other do not outvote the one tile that
    holds most of the slide; without the extents every tile counts the same."""
    c = np.array([[0.0, 0.0], [0.0, 230.0], [230.0, 0.0], [230.0, 230.0]])
    off = np.array([[0.0, 0.0], [4.0, 7.0], [4.0, 7.0], [4.3, 6.8]])
    p = {
        "tile": 256,
        "ref_true_yx": c.tolist(),
        "ref_pos_yx": (c + off).tolist(),
        "mosaic_shape": [500, 500],
    }
    # the plain median: the mean of two tiles' offsets on x, no tile's own
    assert am.mosaic_origin_xy(p) == pytest.approx([-6.9, -4.0])
    p["ref_valid_wh"] = [[256, 256], [30, 256], [256, 30], [30, 30]]
    assert am.mosaic_origin_xy(p) == [0.0, 0.0]
    p["ref_valid_wh"] = [[30, 30], [256, 256], [256, 256], [30, 30]]
    assert am.mosaic_origin_xy(p) == [-7.0, -4.0]
    # equal extents: the median of an odd number of tiles, as before
    q = {**p, "ref_true_yx": c[:3].tolist(), "ref_pos_yx": (c + off)[:3].tolist()}
    q["ref_valid_wh"] = [[256, 256]] * 3
    assert am.mosaic_origin_xy(q) == [-7.0, -4.0]


def test_the_command_is_ashlars_own_and_nothing_else(tmp_path, small):
    _selfcheck(tmp_path, small)
    (call,) = fa.calls("main")
    argv = call["argv"]
    assert argv[0] == "ashlar" and argv[3:] == [
        "-o",
        argv[4],
        "-c",
        "0",
        "-m",
        "15.0",
    ]
    assert Path(argv[4]).name == "ashlar_output.ome.tif"
    for spec in argv[1:3]:
        reader, _, pattern, overlap, px = spec.split("|")
        assert reader == "filepattern" and pattern == "pattern=r{row:03}_c{col:03}.tif"
        assert float(overlap.split("=")[1]) == pytest.approx(1 - 230 / 256)
        assert px == "pixel_size=0.325"
    assert len(fa.calls("EdgeAligner")) == 1 and len(fa.calls("LayerAligner")) == 1


def test_the_run_record_holds_the_transform_and_the_diagnostics(tmp_path, small):
    _selfcheck(tmp_path, small, keep_work=True)
    (path,) = (tmp_path / "out").rglob("run.json")
    run = json.loads(path.read_text())
    info = run["info"]
    assert run["ok"] and run["direction"] == MOV_TO_REF and run["unit"] == "patient"
    assert info["version"] == "1.20.0" and info["noise"] == "ref"
    assert info["affine"] == "none" and info["align_channel"] == 0
    for key in (
        "discard_over_budget",
        "discard_zero_offset",
        "discard_infinite_error",
        "jump_px_p50",
        "jump_px_p95",
        "reference_stitch_error",
        "retile_s",
        "edge_s",
        "layer_s",
        "register_s_round",
        "peak_rss_after_register_mb",
        "edge_positions_identical",
        "command",
    ):
        assert key in info, key
    assert info["discard_rebuild_matches"] is True
    assert info["register_s_round"] == pytest.approx(
        info["layer_s"][0] + info["edge_s"]
    )
    # the placement IS the transform: one row per tile, both cycles
    p = info["placement"]
    assert len(p["mov_pos_yx"]) == len(p["ref_pos_yx"]) == 20 and p["tile"] == 256
    assert info["reference_stitch_error"]["max"] > 5  # the tile stitched 8 px away
    # 0.2 % of the nuclear channel's range; the floor of 3 counts on the empty one
    assert info["reference_noise_sd"][0] > 3.0 == info["reference_noise_sd"][1]
    assert any(
        "noise on the reference" in s for s in info["implementation"]["benchmark_side"]
    )
    assert run["write_s"] > 0 and run["native"]["origin_xy"] == [-7.0, -4.0]
    # ASHLAR's writer and the benchmark's split, apart; together they are write_s
    parts = run["native"]["write_parts_s"]
    assert parts["convert"] == 0.0 and parts["writer"] >= 0 and parts["benchmark"] > 0
    assert parts["writer"] + parts["benchmark"] == pytest.approx(run["write_s"])
    # what ASHLAR itself wrote: the patient's mosaic, described before it was split
    own = run["native"]["written"]
    assert own["channels"] == 4 and "every cycle" in own["of"]
    assert "band" not in np.load(path.parent / "warped.npz").files  # no nuclei here


def _mutate_origin(monkeypatch):
    real = am.mosaic_origin_xy
    monkeypatch.setattr(am, "mosaic_origin_xy", lambda p: [-v for v in real(p)])


def _mutate_direction(monkeypatch):
    def backwards(placement):
        swapped = {
            **placement,
            "mov_pos_yx": placement["mov_true_yx"],
            "mov_true_yx": placement["mov_pos_yx"],
        }
        return PointMap(tilemap.placement_maps(swapped)[0], MOV_TO_REF)

    monkeypatch.setattr(am, "point_map", backwards)


def _mutate_label(monkeypatch):
    real = am.point_map
    monkeypatch.setattr(am, "point_map", lambda p: PointMap(real(p).fn, REF_TO_MOV))


def _mutate_tile_index(monkeypatch):
    real = am.placement_of

    def swapped(edge, layer, ref_grid, mov_grid):
        p = real(edge, layer, ref_grid, mov_grid)
        pos = np.array(p["mov_pos_yx"])
        pos[[6, 7]] = pos[[7, 6]]  # two interior neighbours take each other's place
        return {**p, "mov_pos_yx": pos.tolist()}

    monkeypatch.setattr(am, "placement_of", swapped)


def _mutate_one_tile_off(monkeypatch):
    real = am.placement_of

    def rolled(edge, layer, ref_grid, mov_grid):
        p = real(edge, layer, ref_grid, mov_grid)
        return {**p, "mov_true_yx": np.roll(p["mov_true_yx"], 1, axis=0).tolist()}

    monkeypatch.setattr(am, "placement_of", rolled)


def _mutate_reference_index(monkeypatch):
    real = am.placement_of

    def rolled(edge, layer, ref_grid, mov_grid):
        p = real(edge, layer, ref_grid, mov_grid)
        return {**p, "ref_true_yx": np.roll(p["ref_true_yx"], 1, axis=0).tolist()}

    monkeypatch.setattr(am, "placement_of", rolled)


def _mutate_reference_sign(monkeypatch):
    def wrong(placement):
        flipped = {
            **placement,
            "ref_pos_yx": placement["ref_true_yx"],
            "ref_true_yx": placement["ref_pos_yx"],
        }
        return PointMap(tilemap.placement_maps(flipped)[0], MOV_TO_REF)

    monkeypatch.setattr(am, "point_map", wrong)


def test_the_unmutated_adapter_passes_on_the_small_grid(tmp_path, small):
    rec, v = _selfcheck(tmp_path, small)
    assert rec["status"] == sc.OK, v
    assert v["reference"]["status"] == sc.OK and v["n_blocks_judged"] >= 2


@pytest.mark.parametrize(
    "mutate,where",
    [
        (_mutate_origin, "reference"),
        (_mutate_direction, "moving"),
        (_mutate_label, "moving"),
        (_mutate_tile_index, "moving"),
        (_mutate_one_tile_off, "moving"),
        (_mutate_reference_index, "reference"),
        (_mutate_reference_sign, "moving"),
    ],
    ids=lambda v: getattr(v, "__name__", v),
)
def test_a_wrong_map_is_inconsistent(tmp_path, small, monkeypatch, mutate, where):
    """Each of these is a mistake the adapter could contain; the selfcheck must object."""
    mutate(monkeypatch)
    rec, v = _selfcheck(tmp_path, small)
    assert rec["status"] == sc.INCONSISTENT and rec["not_ok"], v
    if where == "reference":
        assert v["reference"]["status"] == sc.INCONSISTENT
        assert "the reference as redrawn" in v["failed"]
    else:
        assert "whole image" in v["failed"]


def test_without_the_mask_the_seams_fail_the_check(tmp_path, small):
    """Dots in the overlap bands are drawn twice: unmasked, the check must object, which
    is why the launcher passes ``--mask overlap`` and the masked count is reported."""
    rec = sc.run("ashlar", tmp_path, sizes=small)
    assert rec["status"] == sc.INCONSISTENT


def test_an_unknown_mask_is_refused():
    with pytest.raises(SystemExit, match="knows the masks"):
        am.SELFCHECK_MASK({"placement": _placement()}, "everything")


# ── the four rows ─────────────────────────────────────────────────────────────
def _grid_of(run):
    return json.loads((Path(run["info"]["reference_tiles"]) / "grid.json").read_text())


def test_one_command_registers_every_round_of_the_patient(tmp_path, pair, small):
    cases = patient(tmp_path, pair)
    done = methods.run_group("ashlar", cases, tmp_path / "out")
    assert done == {"P1_r1": ["ashlar"], "P1_r2": ["ashlar"]}
    assert len(fa.calls("main")) == 1 and len(fa.calls("LayerAligner")) == 2
    runs = [run_json(tmp_path / "out", "ashlar", c) for c in cases]
    assert [r["info"]["cycle"] for r in runs] == [1, 2]
    # each round's own placement (the fake shifts every cycle differently)
    assert (
        runs[0]["info"]["placement"]["mov_pos_yx"]
        != (runs[1]["info"]["placement"]["mov_pos_yx"])
    )
    assert (
        runs[0]["info"]["placement"]["ref_pos_yx"]
        == (runs[1]["info"]["placement"]["ref_pos_yx"])
    )
    # per round: its own alignment plus its share of the reference stitch
    for k, r in enumerate(runs):
        info = r["info"]
        assert info["register_s_round"] == pytest.approx(
            info["layer_s"][k] + info["edge_s"] / 2
        )
        native = r["native"]
        assert Path(native["files"][0]).parent == (
            result_dir(tmp_path / "out", "ashlar", cases[k]) / "native"
        )
        assert native["channels"] == ["channel_0", "channel_1"]
        assert "dots_file" not in native and "mosaic" not in native  # a cohort run
    assert runs[0]["native"]["reference_file"] == runs[1]["native"]["reference_file"]
    assert runs[0]["launch_id"] == runs[1]["launch_id"]


def test_noise_goes_on_the_reference_tiles_only(tmp_path, pair, small):
    cases = patient(tmp_path, pair, rounds=1)
    methods.run_group("ashlar", cases, tmp_path / "out", keep_work=True)
    run = run_json(tmp_path / "out", "ashlar", cases[0])
    grid = _grid_of(run)
    assert grid["noise_frac"] == 0.002 and grid["noise_min_counts"] == 3.0
    assert all(sd >= 3.0 for sd in grid["noise_sd"])
    ref = tifffile.imread(pair["reference"])
    mov = tifffile.imread(pair["moving"])
    ref_tile = tifffile.imread(Path(run["info"]["reference_tiles"]) / "r001_c001.tif")
    assert not np.array_equal(ref_tile, ref[:, 230:486, 230:486])
    (mov_dir,) = (tmp_path / "out").rglob("cycle_1")
    mov_grid = json.loads((mov_dir / "grid.json").read_text())
    assert mov_grid["noise_sd"] == [0.0, 0.0] and mov_grid["tile_origins"] == "grid"
    np.testing.assert_array_equal(
        tifffile.imread(mov_dir / "r001_c001.tif"), mov[:, 230:486, 230:486]
    )


def test_the_clean_row_cuts_noise_free_tiles(tmp_path, pair, small):
    cases = patient(tmp_path, pair, rounds=1)
    done = methods.run_group(
        "ashlar", cases, tmp_path / "out", {"noise": "none"}, label="ashlar_clean"
    )
    assert done == {"P1_r1": ["ashlar_clean"]}
    run = run_json(tmp_path / "out", "ashlar_clean", cases[0])
    assert _grid_of(run)["noise_sd"] == [0.0, 0.0]
    ref = tifffile.imread(pair["reference"])
    ref_tile = tifffile.imread(Path(run["info"]["reference_tiles"]) / "r001_c001.tif")
    np.testing.assert_array_equal(ref_tile, ref[:, 230:486, 230:486])
    side = run["info"]["implementation"]["benchmark_side"]
    assert "no noise on any tile" in side


def test_a_missing_anchor_fails_its_round_and_the_others_proceed(tmp_path, pair, small):
    cases = patient(tmp_path, pair)
    out = tmp_path / "out"
    stare_run(out, cases[0], M0)
    opts = {"affine": "stare"}
    done = methods.run_group("ashlar", cases, out, opts, label="affine_ashlar")
    assert done == {"P1_r1": ["affine_ashlar"], "P1_r2": []}
    bad = run_json(out, "affine_ashlar", cases[1])
    missing = result_dir(out, "stare", cases[1]) / "run.json"
    # F11: no anchor is the benchmark's input missing, not the hybrid's own failure
    assert not bad["ok"] and bad["cause"] == "benchmark"
    assert bad["error"] == f"anchor unavailable: {missing}"
    assert bad["info"]["anchor_cause"] == "no record"
    good = run_json(out, "affine_ashlar", cases[0])
    assert good["ok"] and good["info"]["rounds_in_command"] == ["P1_r1"]
    assert good["info"]["m0"] == M0.tolist()
    assert good["info"]["anchor_from"].endswith("stare/selfcheck/P1_r1/run.json")
    # ONE command, the dropped round is not in it
    (call,) = fa.calls("main")
    assert len(call["argv"]) == 1 + 2 + 6 and len(fa.calls("LayerAligner")) == 1
    # with one round the reference stitch is not divided
    info = good["info"]
    assert info["register_s_round"] == pytest.approx(
        info["layer_s"][0] + info["edge_s"]
    )


def test_a_failed_stare_run_leaves_its_cause_beside_the_anchor_failure(
    tmp_path, pair, small
):
    """F11: STARE's own cause (here the scheduler's) is kept, the row is the
    benchmark's."""
    cases = patient(tmp_path, pair, rounds=1)
    out = tmp_path / "out"
    path = stare_run(out, cases[0], ok=False)
    path.write_text(json.dumps({"ok": False, "cause": "infrastructure", "info": {}}))
    methods.run_group("ashlar", cases, out, {"affine": "stare"}, label="affine_ashlar")
    bad = run_json(out, "affine_ashlar", cases[0])
    assert (bad["cause"], bad["info"]["anchor_cause"]) == (
        "benchmark",
        "infrastructure",
    )


def test_an_anchored_row_is_reused_only_while_stares_anchor_is_the_same(
    tmp_path, pair, small
):
    """F10 (b): STARE ran again and moved its matrix: the anchored row is not reused
    with the old one."""
    cases = patient(tmp_path, pair, rounds=1)
    out = tmp_path / "out"
    stare_run(out, cases[0], M0)
    opts = {"affine": "stare"}

    def again():
        methods.run_group(
            "ashlar", cases, out, opts, label="affine_ashlar", resume=True
        )
        return len(fa.calls("main"))

    assert again() == 1 and again() == 1  # the second is reused
    moved = M0.copy()
    moved[0, 2] += 3.0
    stare_run(out, cases[0], moved)
    assert again() == 2
    run = run_json(out, "affine_ashlar", cases[0])
    assert run["ok"] and run["info"]["m0"] == moved.tolist()
    assert run["reuse_key"]["method"] == {"anchor_m0": {"P1_r1": moved.tolist()}}
    plain = methods.reuse_key("ashlar", cases, out, {}, None)
    assert plain["method"] is None  # a row without an anchor reads none


@pytest.mark.parametrize(
    "exc, cause",
    [
        (OSError(28, "No space left on device"), "benchmark"),
        (MemoryError("Unable to allocate 12.0 GiB"), "benchmark_oom"),
    ],
)
def test_a_failure_of_the_tile_cutting_is_the_benchmarks(
    tmp_path, pair, small, monkeypatch, exc, cause
):
    """F12: the benchmark cuts the tiles, with the clock paused: a full disk or a memory
    error there is not ASHLAR's failure."""

    def broken(*a, **kw):
        raise exc

    monkeypatch.setattr(retile, "write_tiles", broken)
    cases = patient(tmp_path, pair, rounds=1)
    methods.run_group("ashlar", cases, tmp_path / "out")
    run = run_json(tmp_path / "out", "ashlar", cases[0])
    assert not run["ok"] and run["cause"] == cause
    assert "tile cutting failed" in run["error"]


def test_tiles_cut_by_other_code_are_not_found_in_the_cache(
    tmp_path, pair, small, monkeypatch
):
    """F10 (a): the tile cache is keyed by the code that cuts the tiles."""
    slides = [pair["reference"], pair["moving"]]
    px = pair["pixel_size_um"]
    first, _, _ = am.reference_tiles(slides[0], slides, px, 0.0, tmp_path, tmp_path)
    copy = tmp_path / "retile.py"
    copy.write_text(Path(retile.__file__).read_text() + "\n# another cut\n")
    monkeypatch.setattr(retile, "__file__", str(copy))
    second, _, reused = am.reference_tiles(
        slides[0], slides, px, 0.0, tmp_path, tmp_path
    )
    assert second != first and reused is False


@pytest.mark.parametrize(
    "ok,m0,text",
    [
        (False, M0, "holds no info.m0 of a run that ok"),
        (True, None, "holds no info.m0 of a run that ok"),
        (True, np.eye(2), "has no 3 x 3 finite info.m0"),
    ],
)
def test_a_stare_run_without_a_usable_matrix_is_no_anchor(tmp_path, pair, ok, m0, text):
    (case,) = patient(tmp_path, pair, rounds=1)
    path = stare_run(tmp_path / "out", case, m0, ok)
    got, why = am.read_anchor({}, case, tmp_path / "out")
    assert got is None and why == f"anchor unavailable: {path} {text}"


def test_the_anchor_is_found_in_a_stare_selfchecks_work_folder(tmp_path, pair):
    (case,) = patient(tmp_path, pair, rounds=1)
    h, w = case.ref_hw
    for sub in (f"{h}x{w}/out", "out"):
        work = tmp_path / sub.replace("/", "_")
        path = stare_run(work / sub, case, M0)
        got, where = am.read_anchor({"m0_from": str(work)}, case, None)
        np.testing.assert_array_equal(got, M0)
        assert where == str(path)


def test_stares_selfcheck_holds_the_anchors_of_ashlars_own_cases(tmp_path):
    """The anchored rows read STARE's matrix of the SAME case: so STARE's selfcheck
    list holds ASHLAR's pair and patient, on ASHLAR's canvas, and each of their rounds
    is found in the folder STARE's selfcheck leaves it in (a stand-in record here: the
    real registrations of that canvas take minutes and run on the cluster)."""
    cases = sc.sizes_for("ashlar")
    assert (
        cases
        == [(sc.ASHLAR_HW, None), sc.ASHLAR_PATIENT]
        == sc.sizes_for("affine_ashlar_clean")
    )
    assert sc.sizes_for("stare")[-2:] == cases and sc.ASHLAR_HW == (8192, 10240)
    work = tmp_path / "stare_work"
    used, wanted = set(), []
    for size in sc.sizes_for("stare"):
        folder = sc.case_folder(size, used)
        used.add(folder)
        group, ids = sc.case_names(size)
        wanted += [(size, folder, group, cid) for cid in ids if size in cases]
    assert [(f, c) for _, f, _, c in wanted] == [
        ("8192x10240", "pair_8192x10240"),
        ("patient_8192x10240", "patient_8192x10240_round1"),
        ("patient_8192x10240", "patient_8192x10240_round2"),
    ]
    for i, (_, folder, group, cid) in enumerate(wanted):
        case = Case(
            dataset=sc.DATASET, case_id=cid, group=group, ref_nuclear="",
            mov_nuclear="", ref_image="", mov_image="", modality="fluorescence",
            diagonal=1.0, ref_hw=list(sc.ASHLAR_HW), mov_hw=list(sc.ASHLAR_HW),
        )  # fmt: skip
        m0 = M0 + np.array([[0, 0, i], [0, 0, 0], [0, 0, 0]])  # one matrix per round
        path = stare_run(work / folder / "out", case, m0)
        got, where = am.read_anchor({"m0_from": str(work)}, case, None)
        np.testing.assert_array_equal(got, m0)
        assert where == str(path)
    # a round STARE's selfcheck did not leave is no anchor
    gone = Case(
        dataset=sc.DATASET, case_id="patient_8192x10240_round3",
        group="patient_8192x10240", ref_nuclear="", mov_nuclear="", ref_image="",
        mov_image="", modality="fluorescence", diagonal=1.0,
        ref_hw=list(sc.ASHLAR_HW), mov_hw=list(sc.ASHLAR_HW),
    )  # fmt: skip
    got, why = am.read_anchor({"m0_from": str(work)}, gone, None)
    assert got is None and why.startswith("anchor unavailable")


def test_anchored_moving_tiles_are_cut_where_the_anchor_puts_them(
    tmp_path, pair, small
):
    """The fourth combination: STARE's anchor, no noise anywhere."""
    cases = patient(tmp_path, pair, rounds=1)
    out = tmp_path / "out"
    stare_run(out, cases[0], M0)
    opts = {"affine": "stare", "noise": "none"}
    done = methods.run_group(
        "ashlar", cases, out, opts, label="affine_ashlar_clean", keep_work=True
    )
    assert done == {"P1_r1": ["affine_ashlar_clean"]}
    run = run_json(out, "affine_ashlar_clean", cases[0])
    assert _grid_of(run)["noise_sd"] == [0.0, 0.0]  # reference tiles, noise-free
    (mov_dir,) = out.rglob("cycle_1")
    grid = json.loads((mov_dir / "grid.json").read_text())
    assert grid["tile_origins"] == "given" and grid["noise_sd"] == [0.0, 0.0]
    nominal = am.nominal_origins([pair["reference"], pair["moving"]])
    # M0 is a shift of (+18, -12): the moving tile is cut 18 px left and 12 px below
    np.testing.assert_array_equal(
        np.array(grid["true_positions_yx"]), nominal + [12, -18]
    )
    mov = tifffile.imread(pair["moving"])
    np.testing.assert_array_equal(
        tifffile.imread(mov_dir / "r001_c001.tif"),
        mov[:, 242:498, 212:468],
    )
    # ASHLAR was told the regular grid all the same
    spec = fa.calls("main")[0]["argv"][2]
    assert "pattern=r{row:03}_c{col:03}.tif" in spec
    assert run["info"]["placement"]["mov_true_yx"] == grid["true_positions_yx"]
    assert any(
        "STARE's rigid matrix" in s
        for s in run["info"]["implementation"]["benchmark_side"]
    )


def test_a_rotated_anchor_cuts_each_tile_where_its_own_centre_goes(tmp_path, pair, small):
    """WT7 (T29): the anchor with a rotation. Moving tile ``k`` is cut, unrotated, around
    where M0's inverse carries reference tile ``k``'s centre, so the offset differs from
    tile to tile (a pure shift would move every tile alike). Computed here by hand, not by
    ``tilemap.anchor_tile_origins``."""
    cases = patient(tmp_path, pair, rounds=1)
    out = tmp_path / "out"
    t = math.radians(2.0)
    m0 = np.array([[math.cos(t), -math.sin(t), 18.0],
                   [math.sin(t), math.cos(t), -12.0], [0.0, 0.0, 1.0]])
    stare_run(out, cases[0], m0)
    done = methods.run_group("ashlar", cases, out, {"affine": "stare", "noise": "none"},
                             label="affine_ashlar_clean", keep_work=True)
    assert done == {"P1_r1": ["affine_ashlar_clean"]}
    (mov_dir,) = out.rglob("cycle_1")
    grid = json.loads((mov_dir / "grid.json").read_text())
    nominal = np.asarray(am.nominal_origins([pair["reference"], pair["moving"]]), float)
    half = SMALL_TILE / 2.0
    want = []
    for y, x in nominal:  # the reference tile's centre, back through M0 (x-y), rounded
        cx, cy = x + half - 18.0, y + half + 12.0
        mx = math.cos(t) * cx + math.sin(t) * cy
        my = -math.sin(t) * cx + math.cos(t) * cy
        want.append([round(my - half), round(mx - half)])
    got = np.asarray(grid["true_positions_yx"])
    np.testing.assert_array_equal(got, want)
    shifts = got - nominal
    assert len({tuple(d) for d in shifts.tolist()}) > 1  # the rotation moves tiles apart
    # a tile wholly inside the moving slide holds exactly those pixels, unrotated
    mov = tifffile.imread(pair["moving"])
    h, w = mov.shape[-2:]
    inside = [k for k, (y, x) in enumerate(got)
              if 0 <= y and 0 <= x and y + SMALL_TILE <= h and x + SMALL_TILE <= w]
    assert inside
    names = sorted(p.name for p in mov_dir.glob("r*_c*.tif"))
    for k in inside[:2]:
        y, x = (int(v) for v in got[k])
        np.testing.assert_array_equal(
            tifffile.imread(mov_dir / names[k]), mov[:, y:y + SMALL_TILE, x:x + SMALL_TILE])


def test_the_anchored_row_passes_its_selfcheck(tmp_path, small):
    """The same two comparisons with moving tiles cut off the grid."""
    h, w = SMALL
    case_dir = Path("stare") / "selfcheck" / f"pair_{h}x{w}"
    d = tmp_path / "stare_work" / "out" / case_dir
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"ok": True, "info": {"m0": M0.tolist()}}))
    rec = sc.run(
        "ashlar",
        tmp_path / "check",
        {"affine": "stare"},
        label="affine_ashlar",
        mask="overlap",
        m0_from=tmp_path / "stare_work",
        sizes=small,
    )
    v = rec["pairs"][0]["variants"]["affine_ashlar"]
    assert rec["status"] == sc.OK, v
    assert v["reference"]["status"] == sc.OK


# ── one patient of three slides: two moving cycles in ONE command ─────────────
def _patient(small):
    return [(small[0][0], None, None)]


def test_a_patient_of_three_slides_is_one_ashlar_command(tmp_path, small):
    rec = sc.run("ashlar", tmp_path, mask="overlap", sizes=_patient(small))
    assert rec["status"] == sc.OK and not rec["not_ok"], rec["pairs"]
    h, w = SMALL
    assert [p["case_id"] for p in rec["pairs"]] == [
        f"patient_{h}x{w}_round1",
        f"patient_{h}x{w}_round2",
    ]
    (call,) = fa.calls("main")  # every cycle in one run of ASHLAR's own command
    cycles = call["argv"][1:4]  # the reference and both moving cycles
    assert call["argv"][4] == "-o" and all(c.startswith("filepattern|") for c in cycles)
    assert [c.split("|")[1].rsplit("/", 1)[1] for c in cycles[1:]] == [
        "cycle_1",
        "cycle_2",
    ]
    first, second = (p["variants"]["ashlar"] for p in rec["pairs"])
    # one mosaic, each round's own planes of it, each round's own dots
    assert first["native"]["dots_file"] == second["native"]["dots_file"]
    assert first["native"]["mosaic_planes"] == [2, 4]
    assert second["native"]["mosaic_planes"] == [4, 6]
    assert (first["native"]["dots_channel"], second["native"]["dots_channel"]) == (3, 5)
    for v in (first, second):
        assert v["status"] == sc.OK and v["n_found"] == v["n_dots"] > 30
        assert v["err_px_p95"] < 0.2 and 0 < v["masked_frac"] <= sc.MASK_MAX["overlap"]
        assert v["reference"]["status"] == sc.OK  # the reference as redrawn, once each
    assert first["n_drawn"] != second["n_drawn"] or first["n_dots"] != second["n_dots"]


def test_a_second_cycle_drawn_elsewhere_than_its_map_is_inconsistent(
    tmp_path, small, monkeypatch
):
    """Round 2 of the patient is checked on its own: a map that is right for cycle 1
    and half a pixel off for cycle 2 blocks the method."""
    real = am.point_map
    seen = []

    def off_on_the_second(placement):
        seen.append(1)
        pmap = real(placement)
        if len(seen) < 2:
            return pmap
        return PointMap(lambda xy: pmap.fn(xy) + [0.5, 0.0], pmap.direction)

    monkeypatch.setattr(am, "point_map", off_on_the_second)
    rec = sc.run("ashlar", tmp_path, mask="overlap", sizes=_patient(small))
    assert len(seen) == 2
    assert [p["status"] for p in rec["pairs"]] == [sc.OK, sc.INCONSISTENT]
    assert rec["status"] == sc.INCONSISTENT and rec["not_ok"]


def test_the_anchored_row_reads_each_rounds_matrix_of_the_patient(tmp_path, small):
    """`--m0-from`: STARE's selfcheck ran the same patient round by round, in the folder
    named after the group."""
    h, w = SMALL
    group = f"patient_{h}x{w}"
    for i, shift in ((1, (18.0, -12.0)), (2, (-14.0, 9.0))):
        d = tmp_path / "stare_work" / group / "out" / "stare" / "selfcheck"
        d = d / f"{group}_round{i}"
        d.mkdir(parents=True)
        m0 = np.eye(3)
        m0[:2, 2] = shift
        (d / "run.json").write_text(
            json.dumps({"ok": True, "info": {"m0": m0.tolist()}})
        )
    rec = sc.run(
        "ashlar",
        tmp_path / "check",
        {"affine": "stare"},
        label="affine_ashlar",
        mask="overlap",
        m0_from=tmp_path / "stare_work",
        sizes=_patient(small),
    )
    assert rec["status"] == sc.OK, rec["pairs"]
    out = tmp_path / "check" / "out" / "affine_ashlar" / "selfcheck"
    anchors = [
        json.loads((out / f"{group}_round{i}" / "run.json").read_text())["info"]
        for i in (1, 2)
    ]
    for i, a in enumerate(anchors, 1):
        want = f"stare_work/{group}/out/stare/selfcheck/{group}_round{i}/run.json"
        assert a["anchor_from"].endswith(want)
    # without STARE's run of the patient the anchored row is not ok
    rec = sc.run(
        "ashlar",
        tmp_path / "none",
        {"affine": "stare"},
        label="affine_ashlar",
        mask="overlap",
        m0_from=tmp_path / "elsewhere",
        sizes=_patient(small),
    )
    assert rec["status"] != sc.OK


def test_reference_tiles_are_cut_once_per_noise_level(
    tmp_path, pair, small, monkeypatch
):
    cases = patient(tmp_path, pair, rounds=1)
    out = tmp_path / "out"
    stare_run(out, cases[0], M0)
    cut = []
    real = retile.write_tiles

    def counting(image, outdir, *a, **kw):
        if kw.get("cycle") == 0:
            cut.append(kw.get("noise_frac"))
        return real(image, outdir, *a, **kw)

    monkeypatch.setattr(retile, "write_tiles", counting)
    methods.run_group("ashlar", cases, out)
    methods.run_group("ashlar", cases, out, {"affine": "stare"}, label="affine_ashlar")
    assert cut == [0.002]  # the anchored row found the plain row's reference tiles
    plain = run_json(out, "ashlar", cases[0])["info"]
    anchored = run_json(out, "affine_ashlar", cases[0])["info"]
    assert plain["reference_tiles"] == anchored["reference_tiles"]
    assert (plain["reference_tiles_reused"], anchored["reference_tiles_reused"]) == (
        False,
        True,
    )
    assert Path(plain["reference_tiles"]).parent == (
        out / "cache" / "ashlar_tiles" / "selfcheck__P1"
    )
    # Review item 11: the cost of cutting the reference is the same number for both
    # rows, whichever ran first; it used to be 0 for the one that found the cache.
    assert plain["retile_reference_s"] == anchored["retile_reference_s"] > 0
    for info in (plain, anchored):
        assert info["retile_s"] == pytest.approx(
            info["retile_moving_s"] + info["retile_reference_s"]
        )
    assert anchored["retile_s"] > anchored["retile_moving_s"]
    # the stitch is NOT shared: each command made its own, and they are compared
    assert plain["edge_positions_identical"] is None
    assert anchored["edge_positions_identical"] is True
    # another noise level is another set of tiles
    methods.run_group("ashlar", cases, out, {"noise": "none"}, label="ashlar_clean")
    assert cut == [0.002, 0.0]
    clean = run_json(out, "ashlar_clean", cases[0])["info"]
    assert clean["reference_tiles"] != plain["reference_tiles"]


# ── failures ──────────────────────────────────────────────────────────────────
def test_an_ashlar_exception_fails_the_patient_with_its_message(tmp_path, pair, small):
    """Pixel-identical overlaps make ASHLAR's error metric raise (utils.py:84-86); its
    command does not catch that (scripts/ashlar.py:231), so the adapter must."""
    fa.configure(raise_in="edge")
    cases = patient(tmp_path, pair)
    out = tmp_path / "out"
    done = methods.run_group(
        "ashlar", cases, out, {"noise": "none"}, label="ashlar_clean"
    )
    assert done == {"P1_r1": [], "P1_r2": []}
    for case in cases:
        run = run_json(out, "ashlar_clean", case)
        assert not run["ok"] and run["cause"] == "identical_overlap"
        assert run["error"] == f"RuntimeError: {fa.RAISE_TEXT}"
        assert run["info"]["ashlar_raise"] == run["error"]
        assert run["info"]["noise"] == "none" and "retile_s" in run["info"]


def test_a_raise_in_a_later_cycle_keeps_the_alignments_already_made(
    tmp_path, pair, small
):
    fa.configure(raise_in=2, edge_shifts=off_grid)
    cases = patient(tmp_path, pair)
    done = methods.run_group("ashlar", cases, tmp_path / "out", keep_work=True)
    assert done == {"P1_r1": [], "P1_r2": []}  # one mosaic: the rounds fail together
    (notes,) = (tmp_path / "out").rglob("aligner_notes.json")
    kinds = [n["kind"] for n in json.loads(notes.read_text())]
    assert kinds == ["EdgeAligner", "LayerAligner"]


def test_the_wrapper_writes_each_aligners_note_as_it_finishes(tmp_path, pair, small):
    """``original.run_ashlar`` alone: the notes file exists after the first aligner when
    the second raises, and every note carries its own seconds."""
    cases = patient(tmp_path, pair, rounds=1)
    slides = [pair["reference"], cases[0].mov_image]
    dirs = []
    for k, slide in enumerate(slides):
        d = tmp_path / f"tiles{k}"
        retile.write_tiles(
            slide, d, 256, 0.1, cycle=k, canvas_like=slides, exact_overlap=True
        )
        dirs.append(d)
    grids = [json.loads((d / "grid.json").read_text()) for d in dirs]
    argv = original.ashlar_argv(dirs, grids, tmp_path / "m.ome.tif", 0, 15.0)
    seen = []
    code, notes, _ = original.run_ashlar(
        argv,
        notes_path=tmp_path / "notes.json",
        on_aligner=lambda i, note: seen.append((i, note["kind"])),
    )
    assert code == 0 and seen == [(0, "EdgeAligner"), (1, "LayerAligner")]
    assert all(n["seconds"] >= 0 for n in notes)
    assert json.loads((tmp_path / "notes.json").read_text()) == notes
    (tmp_path / "notes.json").unlink()
    fa.configure(raise_in=1)
    with pytest.raises(RuntimeError, match="correlation > total_amplitude"):
        original.run_ashlar(argv, notes_path=tmp_path / "notes.json")
    (first,) = json.loads((tmp_path / "notes.json").read_text())
    assert first["kind"] == "EdgeAligner" and "seconds" in first
    # the wrapper leaves the classes as it found them, even after a raise
    reg = __import__("sys").modules["ashlar.reg"]
    assert reg.EdgeAligner.run is fa.EdgeAligner.run


def test_a_memory_error_is_not_swallowed_as_an_ashlar_failure(
    tmp_path, pair, small, monkeypatch
):
    """Only ``oom`` moves a run up the ladder: it must keep its cause."""

    def out_of_memory(*a, **kw):
        raise MemoryError("Unable to allocate 91.2 GiB")

    monkeypatch.setattr(original, "run_ashlar", out_of_memory)
    cases = patient(tmp_path, pair, rounds=1)
    methods.run_group("ashlar", cases, tmp_path / "out")
    assert run_json(tmp_path / "out", "ashlar", cases[0])["cause"] == "oom"


def test_a_non_zero_exit_fails_the_patient(tmp_path, pair, small):
    fa.configure(exit_code=1)
    cases = patient(tmp_path, pair)
    done = methods.run_group("ashlar", cases, tmp_path / "out")
    assert done == {"P1_r1": [], "P1_r2": []}
    run = run_json(tmp_path / "out", "ashlar", cases[0])
    assert run["error"].startswith("ashlar exited 1")


def test_cycles_with_the_nucleus_at_different_indices_fail_the_patient(
    tmp_path, pair, small
):
    mov = tmp_path / "round1.ome.tif"
    shutil.copyfile(pair["moving"], mov)
    case = case_of(tmp_path / "cases", pair, "P1_r1", mov, mov_channel=1)
    methods.run_group("ashlar", [case], tmp_path / "out")
    run = run_json(tmp_path / "out", "ashlar", case)
    assert not run["ok"] and "ONE channel index across cycles" in run["error"]
    assert not fa.calls("main")


def test_another_ashlar_version_is_refused(tmp_path, pair, small, monkeypatch):
    fa.install(monkeypatch.setitem, version="1.19.0")
    cases = patient(tmp_path, pair, rounds=1)
    methods.run_group("ashlar", cases, tmp_path / "out")
    run = run_json(tmp_path / "out", "ashlar", cases[0])
    assert "ASHLAR 1.20.0 as shipped; the installed ashlar is 1.19.0" in run["error"]


# ── the clock ─────────────────────────────────────────────────────────────────
class _Clock:
    def __init__(self):
        self.paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def snapshot(self):
        return {"peak_rss_mb": 123.0}


def test_tile_cutting_and_ashlars_writer_run_with_the_clock_paused(
    tmp_path, pair, small, monkeypatch
):
    clock, seen = _Clock(), {}
    for cls, name in (
        (fa.EdgeAligner, "run"),
        (fa.LayerAligner, "run"),
        (fa.PyramidWriter, "run"),
    ):
        real = getattr(cls, name)

        def watched(self, real=real, key=cls.__name__):
            seen[key] = clock.paused
            return real(self)

        monkeypatch.setattr(cls, name, watched)
    real_tiles = retile.write_tiles

    def cutting(*a, **kw):
        seen.setdefault("tiles", []).append(clock.paused)
        return real_tiles(*a, **kw)

    monkeypatch.setattr(retile, "write_tiles", cutting)
    real_split = original.split_mosaic

    def splitting(*a, **kw):
        seen["split"] = clock.paused
        return real_split(*a, **kw)

    monkeypatch.setattr(original, "split_mosaic", splitting)
    cases = patient(tmp_path, pair, rounds=1)
    opts = {"monitor": clock, "tiles_cache": tmp_path / "cache"}
    items = list(am.register_group(cases, tmp_path / "work", opts))
    assert seen == {
        "tiles": [True, True],
        "EdgeAligner": False,
        "LayerAligner": False,
        "PyramidWriter": True,
        "split": True,
    }
    assert not clock.paused  # handed back running
    (item,) = items
    assert item[3]["peak_rss_after_register_mb"] == 123.0


# ── the seam mask on nuclei, and its rows ────────────────────────────────────
def test_nuclei_on_ashlars_seams_are_marked_and_every_method_is_scored_off_them(
    tmp_path, pair, small, monkeypatch
):
    # the fake places every tile exactly on the grid here, so the seams are the cut bands
    fa.configure(edge_shifts=None, layer_shifts=lambda c, r, k: np.zeros((r * k, 2)))
    # tile pitch 230, tiles 256 px: x in 230..256 is held by two tiles
    centres = np.array([[100.0, 100.0], [243.0, 100.0], [222.0, 100.0], [400.0, 300.0]])
    mov = tmp_path / "round1.ome.tif"
    shutil.copyfile(pair["moving"], mov)
    case = case_of(
        tmp_path / "cases",
        pair,
        "P1_r1",
        mov,
        dataset="multiplex",
        cells=(squares(centres), squares(centres)),
    )
    out = tmp_path / "out"
    methods.run_group("ashlar", [case], out)
    with np.load(result_dir(out, "ashlar", case) / "warped.npz") as z:
        # the square around x = 222 reaches 227..: its right edge (227) is 3 px short of
        # the band and stays; the one at 243 is inside it
        np.testing.assert_array_equal(z["band"], [False, True, False, False])
        # every tile sits where it was cut: the map is the identity
        np.testing.assert_allclose(z["cells_xy"], load_points(case)["cells_mov_xy"])
    run = run_json(out, "ashlar", case)
    assert run["info"]["band_fraction"] == 0.25 and "band_mov" not in run["info"]
    res = score(tmp_path / "cases", out, ["multiplex"])  # the default owners
    names = {r["method"] for r in res["nonband_cells"]}
    assert names == {"initial:non_ashlar", "ashlar:non_ashlar"}
    assert {r["n_mov_kept"] for r in res["nonband_cells"]} == {3}
    assert not [r for r in res["cells"] if ":non_" in r["method"]]
    # the shared tiles under <out>/cache are not taken for a method
    assert {r["method"] for r in res["cells"]} == {"initial", "ashlar"}


def test_edge_stitches_are_compared_only_at_the_same_tiles_and_shift(tmp_path):
    """The reference tiles are shared by every cell of one noise level; a cell of another
    maximum shift may stitch them differently and is never compared with."""
    edge_a = {"positions_yx": [[0.0, 0.0], [1755.0, 0.0]]}
    edge_b = {"positions_yx": [[0.0, 0.0], [1754.0, 0.0]]}

    def call(label, shift, edge):
        base = {"label": label, "max_shift_um": shift}
        return am._edge_identity(tmp_path, "ref_n0_x", base, edge)

    assert call("ashlar_clean", 15.0, edge_a) is None  # nothing to compare with yet
    assert (
        call("ashlar_anone_n0_s60", 60.0, edge_b) is None
    )  # other shift: not compared
    assert (
        call("affine_ashlar_clean", 15.0, edge_a) is True
    )  # same shift, plain vs anchored
    assert call("ashlar_astare_n0_s60", 60.0, edge_a) is False  # same shift, differs


# ── failure causes and silent fallbacks (C4) ──────────────────────────────────
def test_identical_overlap_is_the_benchmarks(tmp_path, pair, small):
    """ASHLAR's `correlation > total_amplitude` (utils.py:84-86) on pixel-identical
    overlaps: the benchmark's tiles, not the method's failure."""
    from benchmarks.regbench import failure_log, resources, score

    fa.configure(raise_in="edge")
    cases = patient(tmp_path, pair)
    methods.run_group(
        "ashlar", cases, tmp_path / "out", {"noise": "none"}, label="ashlar_clean"
    )
    run = run_json(tmp_path / "out", "ashlar_clean", cases[0])
    assert run["ok"] is False and run["cause"] == "identical_overlap"
    assert failure_log.WHOSE["identical_overlap"] == failure_log.BENCHMARK
    assert "identical_overlap" in resources.NOT_THE_METHOD
    assert score.CAUSES[-1] == "not_run" and "identical_overlap" in score.CAUSES[:-1]


def test_another_runtime_error_stays_the_methods(tmp_path, pair, small, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("something else")

    monkeypatch.setattr(original, "run_ashlar", boom)
    cases = patient(tmp_path, pair, rounds=1)
    methods.run_group("ashlar", cases, tmp_path / "out")
    assert run_json(tmp_path / "out", "ashlar", cases[0])["cause"] == "exception"


def test_the_fallback_to_stage_positions_is_flagged(tmp_path, pair, small):
    text = (
        "Could not align enough edges, proceeding anyway with original stage positions."
    )
    fa.configure(warn=text)
    cases = patient(tmp_path, pair, rounds=1)
    methods.run_group("ashlar", cases, tmp_path / "out")
    run = run_json(tmp_path / "out", "ashlar", cases[0])
    assert run["ok"] is True
    assert run["info"]["fallback_stage_positions"] is True
    assert run["info"]["zero_overlap"] is False
    assert run["info"]["ashlar_warnings"] == [text]


@pytest.mark.parametrize(
    "text",
    [
        "No tiles overlap, attempting alignment anyway.",
        "Some neighboring tiles have zero overlap.",
    ],
)
def test_a_zero_overlap_warning_is_flagged(tmp_path, pair, small, text):
    fa.configure(warn=text)
    cases = patient(tmp_path, pair, rounds=1)
    methods.run_group("ashlar", cases, tmp_path / "out")
    info = run_json(tmp_path / "out", "ashlar", cases[0])["info"]
    assert info["zero_overlap"] is True and info["fallback_stage_positions"] is False


def test_no_warning_leaves_the_flags_false(tmp_path, pair, small):
    cases = patient(tmp_path, pair, rounds=1)
    methods.run_group("ashlar", cases, tmp_path / "out")
    info = run_json(tmp_path / "out", "ashlar", cases[0])["info"]
    assert info["ashlar_warnings"] == []
    assert info["fallback_stage_positions"] is False and info["zero_overlap"] is False


def test_a_fully_modelled_cycle_is_flagged(tmp_path, pair, small):
    fa.configure(layer_errors=lambda cycle, n: np.full(n, np.inf))
    cases = patient(tmp_path, pair, rounds=1)
    methods.run_group("ashlar", cases, tmp_path / "out")
    info = run_json(tmp_path / "out", "ashlar", cases[0])["info"]
    assert info["discard_frac"] == 1.0 and info["cycle_fully_modelled"] is True


def test_a_cycle_with_some_tiles_kept_is_not_fully_modelled(tmp_path, pair, small):
    cases = patient(tmp_path, pair, rounds=1)
    methods.run_group("ashlar", cases, tmp_path / "out")
    info = run_json(tmp_path / "out", "ashlar", cases[0])["info"]
    assert info["discard_frac"] < 1.0 and info["cycle_fully_modelled"] is False


# ── the final review's minors: the cause of a raise, the shared tile caches ───
def test_a_raise_with_noisy_reference_tiles_is_not_identical_overlap(
    tmp_path, pair, small
):
    """Pixel-identical overlaps exist only on noise-free tiles: with noise on the
    reference the same text is ASHLAR's own failure (`exception`), not the benchmark's
    `identical_overlap`."""
    fa.configure(raise_in="edge")
    cases = patient(tmp_path, pair)
    out = tmp_path / "out"
    assert methods.run_group("ashlar", cases, out) == {"P1_r1": [], "P1_r2": []}
    for case in cases:
        run = run_json(out, "ashlar", case)
        assert not run["ok"] and run["cause"] == "exception"
        assert run["info"]["noise_frac"] > 0
        assert run["error"] == f"RuntimeError: {fa.RAISE_TEXT}"


def test_moving_tiles_are_cut_once_and_shared_by_every_cell(
    tmp_path, pair, small, monkeypatch
):
    """The moving cycles are noise-free in every cell: two cells of the same affine
    find the first one's tiles (content-keyed, in the patient's tile cache); an
    anchored cell cuts its own (its tiles sit where STARE's matrix puts them)."""
    cases = patient(tmp_path, pair, rounds=1)
    out = tmp_path / "out"
    stare_run(out, cases[0], M0)
    cut = []
    real = retile.write_tiles

    def counting(image, outdir, *a, **kw):
        if kw.get("cycle", 0) > 0:
            cut.append((kw["cycle"], kw.get("tile_origins_yx") is not None))
        return real(image, outdir, *a, **kw)

    monkeypatch.setattr(retile, "write_tiles", counting)
    methods.run_group("ashlar", cases, out)
    methods.run_group("ashlar", cases, out, {"noise": "none"}, label="ashlar_clean")
    assert cut == [(1, False)]  # the clean cell found the plain cell's moving tiles
    plain = run_json(out, "ashlar", cases[0])["info"]
    clean = run_json(out, "ashlar_clean", cases[0])["info"]
    assert plain["moving_tiles"] == clean["moving_tiles"]
    assert (plain["moving_tiles_reused"], clean["moving_tiles_reused"]) == (
        [False],
        [True],
    )
    (where,) = plain["moving_tiles"]
    assert Path(where).name == "cycle_1"
    assert Path(where).parent.parent == out / "cache" / "ashlar_tiles" / "selfcheck__P1"
    # what cutting them cost is the same number for both cells
    assert plain["retile_moving_s"] == clean["retile_moving_s"] > 0
    methods.run_group("ashlar", cases, out, {"affine": "stare"}, label="affine_ashlar")
    assert cut == [(1, False), (1, True)]
    anchored = run_json(out, "affine_ashlar", cases[0])["info"]
    assert anchored["moving_tiles"] != plain["moving_tiles"]


def test_two_jobs_cutting_the_same_tiles_at_once_cannot_clobber_each_other(
    tmp_path, pair, small, monkeypatch
):
    """Two jobs (on two nodes, whose process ids can be equal) cut the same reference
    tiles at once: each writes its own temporary folder and one rename wins. Here the
    second job runs to the end between the first one's cut and its rename."""
    monkeypatch.setattr(am.os, "getpid", lambda: 4242)
    real = retile.write_tiles
    calls = []
    slides = [str(pair["reference"]), str(pair["moving"])]
    cache = tmp_path / "cache"

    def racing(image, outdir, *a, **kw):
        got = real(image, outdir, *a, **kw)
        calls.append(Path(outdir).name)
        if len(calls) == 1:  # the other job, start to finish
            am.reference_tiles(slides[0], slides, 0.5, 0.0, cache, tmp_path / "w2")
        return got

    monkeypatch.setattr(retile, "write_tiles", racing)
    final, seconds, reused = am.reference_tiles(
        slides[0], slides, 0.5, 0.0, cache, tmp_path / "w1"
    )
    assert len(calls) == 2 and calls[0] != calls[1]  # two temporary folders
    assert (final / "grid.json").exists() and seconds is not None and not reused
    assert sorted(p.name for p in cache.iterdir()) == [final.name]  # nothing left over
