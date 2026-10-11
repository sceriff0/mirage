"""The PALOM adapter, without PALOM.

PALOM (pinned at 2026.4.1) is not installed where this suite runs, so everything here runs
against ``fake_palom``: a recording stand-in that keeps the package's bookkeeping and draws
the registered image the way its writer is written to (see that module). What these tests
pin:

* the call is the README's recipe, with its arguments and nothing else, and no dask,
  OpenCV or thread setting is touched;
* the point map: the affine alone, and the affine with each block's shift, against the
  emulated writer -- dot by dot through the selfcheck, and pixel by pixel on a ramp; that
  a reversed direction, swapped ``(dy, dx)``, negated shifts, a transposed grid, a block
  edge one pixel off and a half-pixel offset are each caught;
* the rechunked row draws its image on the blocks its map uses;
* the seam mask: which reference nuclei and which known-truth probes it marks, that it is
  a function of the block grid and the shifts alone, and what the scorer does with it;
* what a run records, and how its failures surface.

What they cannot pin is PALOM, OpenCV's ``warpAffine`` inside it, and its reader on real
files: ``test_external_methods.py`` holds the tests that need the real package, skipped
here.
"""

import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import METHODS, cells, methods, score
from benchmarks.regbench import selfcheck as sc
from benchmarks.regbench.cases import load_case, load_points, polys, result_dir
from benchmarks.regbench.datasets import synthetic
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap
from benchmarks.regbench.methods import palom as pa

from . import fake_palom as fp

# 5 x 6 blocks of 256 px with short last ones (176 and 220 px); the moving slide is
# smaller and odd-sized, so the two thumbnail down-factors differ once a level is used
SMALL = [((1200, 1500), (1141, 1421))]
TILE = 256
RAMP_TOL = 0.05  # px: OpenCV's bilinear warp works on a 1/32 px grid [MEMORY]
BLOCK_MAP = pa.block_map  # the real ones, whatever a test puts in their place
PAIR_TILE = pa.SELFCHECK_PAIR_TILE


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    """The fake ``palom`` for one test, and 256 px tiles for the selfcheck pair."""
    fp.install(monkeypatch.setitem)
    monkeypatch.setattr(pa, "SELFCHECK_PAIR_TILE", lambda opts: TILE)
    yield fp
    fp.configure()


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    return sc.make_pair(tmp_path_factory.mktemp("palom_pair"), *SMALL[0], tile=TILE)


@pytest.fixture
def case(pair, tmp_path):
    return sc.pair_case(pair, tmp_path / "cases")


def run_json(out, variant, case):
    return json.loads((result_dir(out, variant, case) / "run.json").read_text())


def _selfcheck(tmp_path, mask="seams", label=None, sizes=SMALL, **opts):
    opts = {"workers": 1, **opts}  # no process pool to write a small pair
    rec = sc.run("palom", tmp_path, opts=opts, label=label, mask=mask, sizes=sizes)
    return rec, rec["pairs"][0]["variants"]


# ── names and settings ────────────────────────────────────────────────────────
def test_registered_and_named():
    assert "palom" in METHODS and pa.SCOPE == "pair"
    assert pa.VARIANTS == ("palom_affine", "palom") and pa.DIRECTION == REF_TO_MOV
    assert methods.scope_of("palom") == "pair"
    assert methods.variant_names("palom") == ["palom_affine", "palom"]
    assert methods.variant_names("palom", "palom_b1024") == [
        "palom_b1024_affine",
        "palom_b1024",
    ]
    # each row's final variant owns a seam mask (R3.4)
    assert {"palom", "palom_b1024"} <= set(score.SEAM_OWNERS)
    assert pa.LABELS == {"palom": None, "palom_b1024": 1024}


def test_module_imports_without_palom_cv2_or_dask():
    code = (
        "import sys\n"
        "from benchmarks.regbench.methods import palom\n"
        "bad = [m for m in ('palom', 'cv2', 'dask', 'matplotlib') if m in sys.modules]\n"
        "raise SystemExit(int(bool(bad)))\n"
    )
    root = Path(__file__).resolve().parents[3]
    assert subprocess.run([sys.executable, "-c", code], cwd=root).returncode == 0


@pytest.mark.parametrize(
    "opts, want",
    [
        ({}, None),
        ({"label": "palom"}, None),
        ({"label": "palom_b1024", "block": "1024"}, 1024),
        ({"label": "palom_b256", "block": 256}, 256),
    ],
)
def test_every_row_is_accepted(opts, want):
    assert pa.settings(opts) == (want, "auto")


@pytest.mark.parametrize(
    "opts",
    [
        {"block": "1024"},  # the primary's name on the rechunked run
        {"label": "palom", "block": "1024"},
        {"label": "palom_b1024"},  # the rechunked row's name on the README's run
        {"label": "palom_b1024", "block": "2048"},
        {"label": "something", "block": "1024"},
        {"label": "palom_b8", "block": "8"},
        {"label": "palom_b1024", "block": "big"},
        {"preread": "maybe"},
    ],
)
def test_a_label_and_its_options_come_together(opts):
    with pytest.raises(ValueError):
        pa.settings(opts)


# ── the call ──────────────────────────────────────────────────────────────────
def test_the_recipe_is_the_readmes(tmp_path, case, fake):
    assert methods.run_case("palom", case, tmp_path / "out") == [
        "palom_affine",
        "palom",
    ]
    readers = fake.calls("OmePyramidReader")
    assert [(r["path"], r["pixel_size"]) for r in readers] == [
        (case.ref_image, None),
        (case.mov_image, None),
    ]
    (thumb,) = fake.calls("get_thumbnail_level_of_size")
    assert thumb == {
        "call": "get_thumbnail_level_of_size",
        "path": case.ref_image,
        "size": 2000,
    }
    (aligner,) = fake.calls("Aligner")
    assert (
        aligner["keywords"] == sorted(pa.ALIGNER_KEYWORDS)
        and len(aligner["keywords"]) == 6
    )
    assert [c["kwargs"] for c in fake.calls("coarse_register_affine")] == [
        {"n_keypoints": 4000}
    ]
    for name in ("compute_shifts", "constrain_shifts"):
        (call,) = fake.calls(name)
        assert call["args"] == () and call["kwargs"] == {}, name
    order = [
        c["call"]
        for c in fake.CALLS
        if c["call"] in ("Aligner", *[x.split("(")[0] for x in pa.RECIPE["calls"]])
    ]
    assert order == [
        "Aligner",
        "coarse_register_affine",
        "compute_shifts",
        "constrain_shifts",
    ]
    # level 0 for the registration, the nuclear channel by index
    reads = [(c["level"], c["channels"]) for c in fake.calls("read_level_channels")]
    assert reads == [(0, 0), (0, 0), (0, 0), (0, 0)]  # thumbnail level 0 on this pair
    info = run_json(tmp_path / "out", "palom", case)["info"]
    assert info["RECIPE"] == json.loads(json.dumps(pa.RECIPE))
    assert info["version"] == "2026.4.1" and info["block_option"] is None


def test_the_thumbnail_level_is_the_references_and_each_slide_has_its_factor(
    tmp_path, case, fake, monkeypatch
):
    seen = {}
    real = fp.Aligner.__init__

    def spy(self, **keywords):
        seen.update(keywords)
        real(self, **keywords)

    monkeypatch.setattr(fp.Aligner, "__init__", spy)
    monkeypatch.setattr(pa, "THUMBNAIL_SIZE", 500)  # level 1 of this small pair
    methods.run_case("palom", case, tmp_path / "out")
    info = run_json(tmp_path / "out", "palom_affine", case)["info"]
    assert info["thumbnail_level"] == 1 and info["n_levels"] == {
        "reference": 2,
        "moving": 2,
    }
    assert info["file_levels"] == {"reference": 1, "moving": 1}
    assert info["auto_pyramid"] == {"reference": True, "moving": True}
    assert seen["ref_thumbnail_down_factor"] == info["ref_down_factor"] == 2.0
    # 1141 x 1421 halves to 570 x 710 with the excess trimmed
    want = (1141 / 570 + 1421 / 710) / 2
    assert seen["moving_thumbnail_down_factor"] == pytest.approx(want, abs=1e-12)
    assert info["mov_down_factor"] == pytest.approx(want, abs=1e-12)
    assert seen["ref_thumbnail"].shape == (600, 750)  # computed, not lazy
    assert isinstance(seen["ref_thumbnail"], np.ndarray)
    assert info["thumb_um_per_px"] == pytest.approx(2.0 * case.pixel_size_um)
    levels = [c["level"] for c in fake.calls("read_level_channels")]
    assert sorted(levels) == [0, 0, 1, 1]


def test_a_moving_slide_with_fewer_levels_fails_the_case(tmp_path, monkeypatch):
    pair = sc.make_pair(tmp_path / "pair", (1200, 1500), (700, 900), tile=TILE)
    case = sc.pair_case(pair, tmp_path / "cases")
    monkeypatch.setattr(pa, "THUMBNAIL_SIZE", 500)
    assert methods.run_case("palom", case, tmp_path / "out") == []
    run = run_json(tmp_path / "out", "palom", case)
    assert "one level for both" in run["error"] and run["cause"] == "exception"


def _forbidden_settings(source):
    """Names in the adapter's CODE that would set a pool size or a dask option."""
    banned = {
        "setNumThreads",
        "set_num_threads",
        "threadpool_limits",
        "environ",
        "putenv",
        "num_workers",
        "scheduler",
        "config",
    }
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in banned:
            found.add(node.attr)
        elif isinstance(node, ast.Name) and node.id in banned:
            found.add(node.id)
        elif isinstance(node, ast.keyword) and node.arg in banned:
            found.add(node.arg)
    return found


def test_no_dask_or_thread_setting_is_touched(tmp_path, case):
    source = Path(pa.__file__).read_text()
    assert _forbidden_settings(source) == set()
    # the guard does fire on the things it is there for
    for line, name in (
        ("import dask\ndask.config.set(scheduler='threads')", "config"),
        ("import cv2\ncv2.setNumThreads(4)", "setNumThreads"),
        ("import os\nos.environ['OMP_NUM_THREADS'] = '4'", "environ"),
        ("x.compute(num_workers=4)", "num_workers"),
    ):
        assert name in _forbidden_settings(source + "\n" + line), name
    before = dict(os.environ)
    methods.run_case("palom", case, tmp_path / "out")
    assert dict(os.environ) == before
    info = run_json(tmp_path / "out", "palom", case)["info"]
    seen = info["concurrency"]
    assert seen["psutil"] is True and seen["os_cpu_count"] == os.cpu_count()
    assert info["workers_seen"] == seen["dask_cpu_count"] > 0


def test_psutil_is_required(tmp_path, case, monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)  # import psutil now fails
    assert methods.run_case("palom", case, tmp_path / "out") == []
    assert "psutil" in run_json(tmp_path / "out", "palom", case)["error"]


def test_another_version_is_refused(tmp_path, case, monkeypatch):
    fp.install(monkeypatch.setitem, version="2025.7.1")
    assert methods.run_case("palom", case, tmp_path / "out") == []
    assert (
        "2026.4.1 is the pinned release"
        in (run_json(tmp_path / "out", "palom", case)["error"])
    )


# ── the exact map ─────────────────────────────────────────────────────────────
def _geometry(shifts=None, chunks=((128, 128, 44), (128, 128, 128, 36))):
    rows, cols = len(chunks[0]), len(chunks[1])
    th = math.radians(1.3)
    affine = np.array(
        [
            [1.004 * math.cos(th), -1.004 * math.sin(th), 6.3],
            [1.004 * math.sin(th), 1.004 * math.cos(th), -4.6],
            [0.0, 0.0, 1.0],
        ]
    )
    if shifts is None:
        rng = np.random.default_rng(4)
        shifts = rng.integers(-4, 5, (rows * cols, 2)).astype(np.float32)
        shifts[3] = (0.37, -1.62)  # a block the filter replaced: not a whole pixel
    return pa.Geometry(affine, shifts, (rows, cols), chunks)


def test_block_edge_sits_half_a_pixel_before_the_edge():
    edges = np.array([0, 128, 256, 300])
    # pixel centres on the integers: pixel 127 is the last of block 0, 128 the first of 1
    assert pa.block_index(edges, [0, 127, 128, 255, 256, 299]).tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    # between two pixel centres a point belongs to the nearer pixel's block
    assert pa.block_index(edges, [127.49, 127.51, 255.49, 255.51]).tolist() == [
        0,
        1,
        1,
        2,
    ]
    # outside the image: the nearest block, never an index off the grid
    assert pa.block_index(edges, [-50.0, 299.6, 4000.0]).tolist() == [0, 2, 2]


def test_short_last_block_uses_the_real_chunks():
    geo = _geometry(chunks=((100, 60, 100), (128, 128, 128, 36)))
    assert geo.edges_y.tolist() == [0, 100, 160, 260]
    assert geo.edges_x.tolist() == [0, 128, 256, 384, 420]
    rows, cols = geo.block_of([[400.0, 159.0], [419.0, 160.0], [383.0, 259.0]])
    assert rows.tolist() == [1, 2, 2] and cols.tolist() == [3, 3, 2]
    with pytest.raises(RuntimeError, match="chunks"):
        pa.Geometry(geo.affine, geo.shifts, (4, 3), ((100, 60, 100), (128,) * 4))
    with pytest.raises(RuntimeError, match="shifts"):
        pa.Geometry(geo.affine, geo.shifts[:5], (3, 4), ((100, 60, 100), (128,) * 4))


def test_the_map_is_the_affines_inverse_after_the_blocks_shift():
    geo = _geometry()
    fn = pa.block_map(geo)
    p = np.array([[200.0, 50.0]])  # block (0, 1)
    dy, dx = geo.shifts[1]
    want = np.linalg.solve(geo.affine, [200.0 - dx, 50.0 - dy, 1.0])[:2]
    np.testing.assert_allclose(fn(p)[0], want, atol=1e-9)
    # and PALOM's own block matrix takes that moving point back to P
    m_b = fp.block_affine_matrices(geo.affine, geo.shifts)[1]
    np.testing.assert_allclose((m_b @ [*fn(p)[0], 1.0])[:2], p[0], atol=1e-9)
    assert fn(np.empty((0, 2))).shape == (0, 2)


def test_affine_stage_ignores_the_shifts():
    geo = _geometry()
    still = pa.Geometry(geo.affine, np.zeros_like(geo.shifts), geo.grid_shape, CHUNKS)
    xy = np.random.default_rng(0).uniform(0, 290, (500, 2))
    np.testing.assert_allclose(pa.affine_map(geo)(xy), pa.block_map(still)(xy))
    back = pa.affine_map(geo)(xy)
    np.testing.assert_allclose(
        back @ geo.affine[:2, :2].T + geo.affine[:2, 2], xy, atol=1e-9
    )
    assert np.abs(pa.block_map(geo)(xy) - back).max() > 1.0


CHUNKS = ((128, 128, 44), (128, 128, 128, 36))


def ramp_errors(align, geo, fn=None, mov_hw=(310, 400)):
    """Draw a moving image whose two channels hold each pixel's own x and y through
    ``align`` (the package's ``palom.align``, or the fake's) and compare every output
    pixel with the point map: ``(largest |image - map| in px, pixels compared)``.

    The ramp is linear, so linear interpolation returns the sampled POSITION itself:
    the image says, pixel by pixel, where it was read from. Pixels read from within a
    pixel of the moving slide's border are left out (zero fill mixes in there).
    """
    import dask.array as da

    h, w = int(geo.edges_y[-1]), int(geo.edges_x[-1])
    chunks = (tuple(np.diff(geo.edges_y)), tuple(np.diff(geo.edges_x)))
    ref = da.zeros((h, w), chunks=chunks, dtype=np.uint16)
    yy, xx = np.mgrid[0 : mov_hw[0], 0 : mov_hw[1]]
    moving = da.from_array(np.stack([xx, yy]).astype(np.float32), chunks=(1, 128, 128))
    mxs = align.block_affine_matrices(geo.affine, geo.shifts.astype(np.float32))
    mxs = align.block_affine_matrices_da(mxs, geo.grid_shape)
    drawn = np.asarray(
        align.block_affine_transformed_moving_img(ref, moving, mxs).compute()
    )
    py, px = np.mgrid[0:h, 0:w]
    centres = np.stack([px.ravel(), py.ravel()], axis=1).astype(float)
    got = (fn or pa.block_map(geo))(centres)
    truth = pa.block_map(geo)(centres)  # where to compare, whatever map is on trial
    inside = (
        (truth[:, 0] >= 1.5)
        & (truth[:, 0] <= mov_hw[1] - 2.5)
        & (truth[:, 1] >= 1.5)
        & (truth[:, 1] <= mov_hw[0] - 2.5)
    )
    image = np.stack([drawn[0].ravel(), drawn[1].ravel()], axis=1)
    return float(np.abs(image - got)[inside].max()), int(inside.sum())


def test_point_map_equals_the_emulated_writer_pixel_for_pixel():
    geo = _geometry()
    worst, n = ramp_errors(fp, geo)
    assert n > 0.8 * 300 * 420 and worst < RAMP_TOL, (worst, n)


def _edges_one_pixel_off(geo):
    late = pa.Geometry(geo.affine, geo.shifts, geo.grid_shape, CHUNKS)
    late.edges_x = geo.edges_x + np.array([0, 1, 1, 1, 0])
    late.edges_y = geo.edges_y + np.array([0, 1, 1, 0])
    return BLOCK_MAP(late)


def _edge_at_the_pixel_border(geo):
    """The edge taken as the first pixel of the next block minus ONE, not minus a half:
    an off-by-one that only shows on whole pixels."""
    late = pa.Geometry(geo.affine, geo.shifts, geo.grid_shape, CHUNKS)
    late.edges_x = geo.edges_x - np.array([0, 1, 1, 1, 0])
    late.edges_y = geo.edges_y - np.array([0, 1, 1, 0])
    return BLOCK_MAP(late)


def _shifted(by):
    return lambda geo: lambda xy, fn=BLOCK_MAP(geo): fn(xy) + by


def _with_shifts(change):
    def make(geo):
        return BLOCK_MAP(
            pa.Geometry(geo.affine, change(geo), geo.grid_shape, geo_chunks(geo))
        )

    return make


def geo_chunks(geo):
    return (tuple(np.diff(geo.edges_y)), tuple(np.diff(geo.edges_x)))


def _transposed(geo):
    """The shifts read column-major: block (r, c) takes the shift stored for the block
    that would sit at (r, c) if the list ran down the columns first."""
    rows, cols = geo.grid_shape
    return geo.shifts.reshape(cols, rows, 2).transpose(1, 0, 2).reshape(-1, 2)


def _forward(geo):
    """The affine and the shift applied forward, as if the map went moving -> reference."""
    cols = geo.grid_shape[1]

    def fn(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        row, col = geo.block_of(xy)
        s = geo.shifts[row * cols + col][:, ::-1]
        return xy @ geo.affine[:2, :2].T + geo.affine[:2, 2] + s

    return fn


WRONG_MAPS = {
    "reversed_direction": _forward,
    "swapped_dy_dx": _with_shifts(lambda g: g.shifts[:, ::-1]),
    "negated_shifts": _with_shifts(lambda g: -g.shifts),
    "transposed_grid": _with_shifts(_transposed),
    "half_pixel_offset": _shifted(0.5),
}
EDGE_MAPS = {
    "edge_one_pixel_late": _edges_one_pixel_off,
    "edge_one_pixel_early": _edge_at_the_pixel_border,
}


@pytest.mark.parametrize("name", sorted({**WRONG_MAPS, **EDGE_MAPS}))
def test_a_wrong_map_departs_from_the_emulated_writer(name):
    geo = _geometry()
    wrong = {**WRONG_MAPS, **EDGE_MAPS}[name](geo)
    worst, _ = ramp_errors(fp, geo, wrong)
    assert worst >= 0.5 - RAMP_TOL, (name, worst)


def test_opencv_draws_what_the_emulated_writer_draws():
    """The fake samples with scipy; the package with ``cv2.warpAffine``. Here the two are
    set side by side on one block, on whatever OpenCV this environment has: the source is
    read at the INVERSE of the matrix at each output pixel centre."""
    cv2 = pytest.importorskip("cv2", reason="OpenCV's own warpAffine, where installed")
    geo = _geometry()
    matrix = fp.block_affine_matrices(geo.affine, geo.shifts)[5]
    yy, xx = np.mgrid[0:310, 0:400]
    for plane in (xx.astype(np.float32), yy.astype(np.float32)):
        info = {
            None: {
                "array-location": [(128, 256), (128, 256)],
                "chunk-shape": (128, 128),
            }
        }
        emulated = fp.draw_block(matrix, src_array=plane, block_info=info)
        local = (
            matrix.copy()
        )  # the block's own frame, as block_affine.py:81-87 builds it
        local[:2, 2] -= (128, 128)
        real = cv2.warpAffine(plane, local[:2], (128, 128), flags=cv2.INTER_LINEAR)
        assert float(np.abs(real - emulated).max()) < RAMP_TOL


# ── the selfcheck: dots found in the written image ────────────────────────────
def test_the_unmutated_adapter_passes_both_rows(tmp_path):
    rec, variants = _selfcheck(tmp_path)
    assert rec["status"] == sc.OK and not rec["blocks_the_method"], rec
    assert rec["variants"] == {"palom_affine": sc.OK, "palom": sc.OK}
    assert rec["pairs"][0]["tile"] == TILE and rec["mask"] == "seams"
    final, stage = variants["palom"], variants["palom_affine"]
    assert final["direction"] == stage["direction"] == REF_TO_MOV
    assert final["n_found"] == final["n_dots"] > 100
    # the affine stage has an image of its own (the package's coarse-only path) and no seams
    assert stage["n_masked"] == 0 < final["n_masked"] < 0.3 * final["n_detected"]
    assert stage["native"]["writer"].endswith("(cli/align_he.py:165-172)")
    assert final["err_px_p95"] < 0.2 and stage["err_px_p95"] < 0.2


def test_without_the_seam_mask_torn_dots_fail_the_block_row(tmp_path):
    """Why PALOM's selfcheck is run with ``--mask seams``: a dot drawn across two blocks
    with different shifts is torn, and its centroid follows neither."""
    _, variants = _selfcheck(tmp_path, mask=None)
    assert variants["palom_affine"]["status"] == sc.OK
    assert variants["palom"]["status"] == sc.INCONSISTENT
    assert variants["palom"]["n_masked"] == 0


@pytest.mark.parametrize("name", sorted(WRONG_MAPS))
def test_a_wrong_block_map_is_inconsistent(tmp_path, monkeypatch, name):
    monkeypatch.setattr(pa, "block_map", WRONG_MAPS[name])
    rec, variants = _selfcheck(tmp_path)
    assert variants["palom"]["status"] == sc.INCONSISTENT, (name, variants["palom"])
    assert variants["palom_affine"]["status"] == sc.OK
    assert rec["status"] == sc.INCONSISTENT and rec["blocks_the_method"]


def test_a_reversed_affine_stage_is_inconsistent(tmp_path, monkeypatch):
    def forward(geo):
        return lambda xy: np.asarray(xy) @ geo.affine[:2, :2].T + geo.affine[:2, 2]

    monkeypatch.setattr(pa, "affine_map", forward)
    _, variants = _selfcheck(tmp_path)
    assert variants["palom_affine"]["status"] == sc.INCONSISTENT
    assert variants["palom"]["status"] == sc.OK


def test_an_edge_one_pixel_off_passes_the_dots_and_not_the_ramp(tmp_path, monkeypatch):
    """A statement of what the selfcheck CANNOT see: the pixels it concerns lie on the
    seams the mask leaves out. The pixel-for-pixel test above is the one that sees it,
    here on the emulated writer and in PALOM's image on its own."""

    def late(geo):
        other = pa.Geometry(geo.affine, geo.shifts, geo.grid_shape, geo_chunks(geo))
        other.edges_x = geo.edges_x + np.r_[0, np.ones(len(geo.edges_x) - 2, int), 0]
        return BLOCK_MAP(other)

    monkeypatch.setattr(pa, "block_map", late)
    _, variants = _selfcheck(tmp_path)
    assert variants["palom"]["status"] == sc.OK


def test_all_zero_shifts_are_not_checked(tmp_path, monkeypatch):
    fp.install(monkeypatch.setitem, max_shift=0)
    rec, variants = _selfcheck(tmp_path)
    assert variants["palom_affine"]["status"] == sc.OK
    assert variants["palom"]["status"] == sc.NOT_CHECKED
    assert "non-zero shift" in variants["palom"]["cause"]
    assert variants["palom"]["n_found"] == variants["palom"]["n_dots"]  # numbers kept
    assert rec["status"] == sc.NOT_CHECKED and rec["blocks_the_method"]
    # with all shifts at zero a wrong sign would indeed have passed
    assert pa.SELFCHECK_NOT_CHECKED({"shift_nonzero_fraction": 0.09}, "palom")
    assert pa.SELFCHECK_NOT_CHECKED({"shift_nonzero_fraction": 0.10}, "palom") is None
    assert (
        pa.SELFCHECK_NOT_CHECKED({"shift_nonzero_fraction": 0.0}, "palom_affine")
        is None
    )


def test_the_not_checked_hook_never_softens_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "SELFCHECK_NOT_CHECKED", lambda info, variant: "always")
    monkeypatch.setattr(pa, "block_map", WRONG_MAPS["negated_shifts"])
    _, variants = _selfcheck(tmp_path)
    assert variants["palom"]["status"] == sc.INCONSISTENT
    assert variants["palom_affine"]["status"] == sc.NOT_CHECKED


def test_the_command_line_takes_paloms_mask_and_rows():
    from benchmarks.regbench import cli

    argv = ["selfcheck", "--method", "palom", "--label", "palom_b1024"]
    argv += ["--opt", "block=1024", "--mask", "seams", "--work", "W"]
    a = cli.build_parser().parse_args(argv)
    assert (a.method, a.label, a.mask, a.opt) == (
        "palom",
        "palom_b1024",
        "seams",
        ["block=1024"],
    )
    assert pa.settings({**methods.parse_opts(a.opt), "label": a.label}) == (
        1024,
        "auto",
    )
    with pytest.raises(SystemExit):  # ASHLAR's mask is not PALOM's
        pa.SELFCHECK_MASK({}, "overlap")


def test_the_selfcheck_pairs_are_enough_blocks_for_both_labels():
    want = [((8192, 10240), None), ((8192, 10240), (7782, 9728))]
    assert sc.sizes_for("palom") == sc.sizes_for("palom_b1024") == want
    assert (8192 // 1024, 10240 // 1024) == (8, 10)
    # the README's row reads the generator's 1024 px tiles; the rechunked row reads
    # 2048 px tiles (the cohort's layout) cut to 1024 px blocks
    assert PAIR_TILE({}) is None and PAIR_TILE({"label": "palom"}) is None
    assert PAIR_TILE({"label": "palom_b1024", "block": "1024"}) == 2048


# ── the rechunked row ─────────────────────────────────────────────────────────
def test_the_rechunked_row_draws_on_the_blocks_of_its_map(tmp_path, monkeypatch, fake):
    monkeypatch.setattr(pa, "SELFCHECK_PAIR_TILE", lambda opts: 512)
    rec, variants = _selfcheck(tmp_path, label="palom_b256", block="256")
    assert rec["variants"] == {"palom_b256_affine": sc.OK, "palom_b256": sc.OK}, rec
    assert rec["pairs"][0]["tile"] == 512
    want = ((256,) * 4 + (176,), (256,) * 5 + (220,))
    drawn = fake.calls("block_affine_transformed_moving_img")
    assert drawn and all(d["ref_chunks"] == want for d in drawn)
    # the block matrices: one 3 x 3 chunk per block of the REchunked grid
    assert drawn[0]["mxs_shape"] == (15, 18) and drawn[0]["mxs_is_dask"]
    out = tmp_path / "out"
    case = load_case(tmp_path / "cases" / "selfcheck" / "pair_1200x1500")
    info = run_json(out, "palom_b256", case)["info"]
    assert info["block_px"] == {"y": [256, 176], "x": [256, 220]}
    assert info["file_block_px"] == {"y": [512, 176], "x": [512, 476]}
    assert info["block_option"] == 256 and info["grid_shape"] == [5, 6]
    side = info["implementation"]["benchmark_side"]
    assert any("rechunk" in s and "Configured by the benchmark" in s for s in side)
    plain = pa.RECIPE["calls"]
    assert info["RECIPE"]["calls"] == plain  # the same recipe otherwise


def test_without_the_rechunk_the_same_files_give_their_own_tiles(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "SELFCHECK_PAIR_TILE", lambda opts: 512)
    rec, _ = _selfcheck(tmp_path)
    case = load_case(tmp_path / "cases" / "selfcheck" / "pair_1200x1500")
    info = run_json(tmp_path / "out", "palom", case)["info"]
    assert (
        info["block_px"]
        == info["file_block_px"]
        == {
            "y": [512, 176],
            "x": [512, 476],
        }
    )
    assert not any("rechunk" in s for s in info["implementation"]["benchmark_side"])


def test_the_declared_row_palom_b1024(tmp_path, monkeypatch, fake):
    monkeypatch.setattr(pa, "SELFCHECK_PAIR_TILE", PAIR_TILE)
    sizes = [((2304, 2816), None)]
    rec, variants = _selfcheck(tmp_path, label="palom_b1024", sizes=sizes, block="1024")
    assert rec["variants"] == {"palom_b1024_affine": sc.OK, "palom_b1024": sc.OK}, rec
    assert rec["pairs"][0]["tile"] == 2048
    (first, *_) = fake.calls("block_affine_transformed_moving_img")
    assert first["ref_chunks"] == ((1024, 1024, 256), (1024, 1024, 768))


def test_the_writer_refuses_an_aligner_that_changed_after_the_yield():
    geo = _geometry()

    class Ref:
        chunks = CHUNKS

    aligner = type("A", (), {})()
    aligner.shifts, aligner.affine_matrix = geo.shifts.copy(), geo.affine.copy()
    aligner.grid_shape, aligner.ref_img = geo.grid_shape, Ref
    pa._still({"aligner": aligner, "geo": geo})
    aligner.shifts[2] += 1.0
    with pytest.raises(RuntimeError, match="changed after the point map"):
        pa._still({"aligner": aligner, "geo": geo})
    aligner.shifts = geo.shifts.copy()
    Ref.chunks = ((128, 172), CHUNKS[1])
    aligner.grid_shape = (2, 4)
    with pytest.raises(RuntimeError, match="changed after the point map"):
        pa._still({"aligner": aligner, "geo": geo})


# ── the written image ─────────────────────────────────────────────────────────
def test_write_native_passes_every_moving_channel_and_the_block_matrices(
    tmp_path, case, pair, fake
):
    out = tmp_path / "out"
    methods.run_case("palom", case, out)
    (drawn,) = fake.calls("block_affine_transformed_moving_img")
    assert drawn["moving_shape"] == (2, *pair["mov_hw"])  # both channels of the slide
    assert drawn["mxs_is_dask"] and drawn["mxs_shape"] == (15, 18)
    assert drawn["ref_chunks"] == ((256,) * 4 + (176,), (256,) * 5 + (220,))
    (written,) = fake.calls("write_pyramid")
    assert written["n_mosaics"] == 1 and written["shapes"] == [(2, 1200, 1500)]
    assert list(written["kwargs"]) == ["pixel_size"]  # writer defaults otherwise
    assert written["kwargs"]["pixel_size"] == pytest.approx(case.pixel_size_um)
    run = run_json(out, "palom", case)
    native = run["native"]
    assert Path(native["files"][0]).name == "registered.ome.tif"
    assert Path(native["files"][0]).parent == result_dir(out, "palom", case) / "native"
    assert native["channels"] == 2 and native["dtype"] == "uint16"
    assert native["nuclear_channel"] == 0 and "origin_xy" not in native
    assert native["fill"].startswith("0 ")
    x0, y0, x1, y1 = native["valid_box"]
    assert 0 <= x0 < x1 <= 1500 and 0 <= y0 < y1 <= 1200
    assert (x1 - x0) < 1500 or (y1 - y0) < 1200  # the moving slide is smaller
    # the affine row has no image on a cohort run (stage images are the selfcheck's)
    assert run_json(out, "palom_affine", case)["native"] is None
    assert run["write_s"] > 0 and run["register_s"] > 0


def test_the_written_image_is_on_the_reference_grid(tmp_path, case, pair):
    from stare.slide_io import open_lazy

    out = tmp_path / "out"
    got = {}
    methods.run_case(
        "palom", case, out, on_variant=lambda c, v, pmap, info: got.update({v: pmap})
    )
    native = run_json(out, "palom", case)["native"]
    arr, dtype, close = open_lazy(native["files"][0])
    try:
        assert tuple(arr.shape) == (2, *pair["ref_hw"])
        dots = sc.find_dots(sc._Plane(arr, pair["dots_channel"]))
    finally:
        close()
    # no origin: a dot found at pixel P is at reference position P
    back = got["palom"].fn(dots)
    found, j = sc._nearest(back, pair["dots_mov_xy"], sc.MATCH_PX)
    assert found.mean() > 0.8  # torn dots on seams are the rest


# ── failures and risks ────────────────────────────────────────────────────────
def test_identity_placeholder_is_a_problem(tmp_path, case, monkeypatch):
    fp.install(monkeypatch.setitem, identity=True)
    out = tmp_path / "out"
    assert methods.run_case("palom", case, out) == ["palom_affine", "palom"]
    for variant in ("palom_affine", "palom"):
        run = run_json(out, variant, case)
        assert run["ok"] and run["problem"] == pa.IDENTITY_PROBLEM
        assert run["info"]["coarse_identity_placeholder"] is True
    fp.install(monkeypatch.setitem)
    methods.run_case("palom", case, tmp_path / "ok")
    assert run_json(tmp_path / "ok", "palom", case)["problem"] == ""


def test_nonfinite_shift_fails_the_case(tmp_path, case, monkeypatch):
    fp.install(monkeypatch.setitem, leave_nonfinite=True, blank=0.3)
    out = tmp_path / "out"
    assert methods.run_case("palom", case, out) == ["palom_affine"]
    run = run_json(out, "palom", case)
    assert run["ok"] is False and "not finite after constrain_shifts" in run["error"]
    assert run_json(out, "palom_affine", case)["ok"] is True


def test_a_filter_that_raises_fails_the_block_row_and_keeps_the_affine(
    tmp_path, case, monkeypatch
):
    """What the ``rigid`` known-truth family may do to PALOM (every shift equal): the row
    fails with the package's own message and is reported as it is."""
    text = (
        "Found array with 0 sample(s) (shape=(0, 2)) while a minimum of 1 is required"
    )
    fp.install(monkeypatch.setitem, constrain_raises=text)
    out = tmp_path / "out"
    assert methods.run_case("palom", case, out) == ["palom_affine"]
    run = run_json(out, "palom", case)
    assert run["ok"] is False and run["cause"] == "exception" and text in run["error"]
    assert not (result_dir(out, "palom", case) / "warped.npz").exists()


def test_figures_the_call_opens_are_saved_and_closed(tmp_path, case, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fp.install(monkeypatch.setitem, figure=True)
    out, work = tmp_path / "out", tmp_path / "work"
    methods.run_case("palom", case, out, work=work, keep_work=True)
    figures = run_json(out, "palom", case)["info"]["figures"]
    assert figures["error"] == "" and len(figures["files"]) == 1
    assert Path(figures["files"][0]).parent == work / "qc"
    assert Path(figures["files"][0]).stat().st_size > 0
    assert plt.get_fignums() == []


def _opened(monkeypatch):
    seen = []

    def stand_in(reader):  # what the workaround does to the fake: reads work again
        seen.append(str(reader.path))
        fp.STATE["closed"] = False

    monkeypatch.setattr(pa, "first_block_persist", stand_in)
    return seen


def test_seek_of_closed_file_is_retried_once_with_the_maintainers_workaround(
    tmp_path, case, monkeypatch
):
    fp.install(monkeypatch.setitem, closed_file=True)
    seen = _opened(monkeypatch)
    out = tmp_path / "out"
    assert methods.run_case("palom", case, out) == ["palom_affine", "palom"]
    assert seen == [case.ref_image, case.mov_image]
    info = run_json(out, "palom", case)["info"]
    assert info["closed_file"]["applied"] and info["closed_file"]["mode"] == "auto"
    assert "seek of closed file" in info["closed_file"]["first_error"]
    side = info["implementation"]["benchmark_side"]
    assert any("personal script" in s and "not an official" in s for s in side)


def test_the_workaround_is_not_applied_when_reads_work(tmp_path, case, monkeypatch):
    seen = _opened(monkeypatch)
    methods.run_case("palom", case, tmp_path / "out")
    info = run_json(tmp_path / "out", "palom", case)["info"]
    assert seen == [] and info["closed_file"] == {
        "mode": "auto",
        "applied": False,
        "first_error": "",
        "failed_call_s": 0.0,
    }
    assert not any(
        "personal script" in s for s in info["implementation"]["benchmark_side"]
    )


def test_preread_off_leaves_the_error_and_on_applies_it_first(
    tmp_path, case, monkeypatch
):
    fp.install(monkeypatch.setitem, closed_file=True)
    seen = _opened(monkeypatch)
    assert methods.run_case("palom", case, tmp_path / "off", {"preread": "0"}) == []
    assert "seek of closed file" in run_json(tmp_path / "off", "palom", case)["error"]
    assert seen == []
    fp.install(monkeypatch.setitem, closed_file=True)
    done = methods.run_case("palom", case, tmp_path / "on", {"preread": "1"})
    assert done == ["palom_affine", "palom"] and len(seen) == 2
    info = run_json(tmp_path / "on", "palom", case)["info"]
    assert info["closed_file"]["first_error"] == "" and info["closed_file"]["applied"]


def test_another_value_error_is_not_retried(tmp_path, case, monkeypatch):
    seen = _opened(monkeypatch)
    reads = pa._Reads([], "auto")

    def boom():
        raise ValueError("something else")

    with pytest.raises(ValueError, match="something else"):
        reads.call(boom)
    assert seen == [] and not reads.applied


def test_the_workaround_is_the_maintainers_line():
    import dask.array as da

    read = []

    def block(b, block_info=None):
        read.append(block_info[0]["chunk-location"])
        return b

    level = da.zeros((2, 40, 60), chunks=(1, 20, 20)).map_blocks(block, dtype=float)
    del read[:]
    pa.first_block_persist(type("R", (), {"pyramid": [level]})())
    assert read == [(0, 0, 0)]  # the first block of level 0, once, and nothing else


# ── what a run records ────────────────────────────────────────────────────────
def test_info_describes_the_blocks_and_can_rebuild_the_map(tmp_path, case):
    out = tmp_path / "out"
    got = {}
    methods.run_case(
        "palom", case, out, on_variant=lambda c, v, pmap, info: got.update({v: pmap})
    )
    info = run_json(out, "palom", case)["info"]
    assert info["grid_shape"] == [5, 6] and info["n_blocks"] == 30
    assert info["block_px"] == {"y": [256, 176], "x": [256, 220]}
    assert 0 < info["n_blocks_blank"] <= info["n_blocks_replaced"] < 30
    assert 0 < info["integer_shifts"] <= 1.0
    assert info["shift_nonzero_fraction"] > 0.8
    assert info["shift_px_p50"] <= info["shift_px_p95"] <= info["shift_px_max"]
    assert info["n_block_edges"] == 5 * 5 + 4 * 6
    assert 0 < info["n_seams"] <= info["n_block_edges"]
    assert info["seam_jump_px_p50"] <= info["seam_jump_px_p95"]
    assert info["pixel_size_detected"] == pytest.approx(case.pixel_size_um)
    assert info["source"] == pa.SOURCE and "_state" not in info
    assert "band_ref" not in info and "band_truth" not in info
    blocks = info["blocks"]
    assert blocks["edges_y"][-1] == 1200 and blocks["edges_x"][-1] == 1500
    assert any(None in row for row in blocks["original_shifts_yx"])  # the blank blocks
    rebuilt = pa.block_map(pa.Geometry.from_info(blocks))
    xy = np.random.default_rng(1).uniform(0, 1190, (2000, 2))
    np.testing.assert_allclose(rebuilt(xy), got["palom"].fn(xy), atol=1e-9)
    # the affine row carries the affine and no blocks
    stage = run_json(out, "palom_affine", case)["info"]
    assert stage["affine"] == info["affine"] and "blocks" not in stage
    np.testing.assert_allclose(
        got["palom_affine"].fn(xy),
        pa.affine_map(pa.Geometry.from_info(blocks))(xy),
        atol=1e-9,
    )
    side = info["implementation"]["benchmark_side"]
    for word in ("by index", "LEVEL = 0", "2000", "after the clock", "attributes"):
        assert any(word in s for s in side), word


# ── seams ─────────────────────────────────────────────────────────────────────
def _squares(centres, half):
    return cells.from_rings(
        [
            np.array(
                [
                    [x - half, y - half],
                    [x + half, y - half],
                    [x + half, y + half],
                    [x - half, y + half],
                ]
            )  # fmt: skip
            for x, y in centres
        ]
    )


def _seam_geometry():
    """2 x 3 blocks of 100 px; the two left columns share one shift, the right column and
    the bottom-left block have others."""
    shifts = np.array(
        [[1, 0], [1, 0], [3, -2], [0, 4], [1, 0], [3, -2]],  # fmt: skip
        dtype=float,
    )
    return pa.Geometry(np.eye(3), shifts, (2, 3), ((100, 100), (100, 100, 100)))


def _points(ref_centres, half=5.0, **extra):
    from benchmarks.regbench.cases import polyset_arrays

    return {**polyset_arrays(_squares(ref_centres, half), "ref"), **extra}


def test_a_nucleus_is_on_a_seam_only_where_the_shifts_differ():
    geo = _seam_geometry()
    centres = [
        (50, 50),  # inside block (0, 0)
        (100, 50),  # across (0, 0) | (0, 1): the same shift, no seam
        (200, 50),  # across (0, 1) | (0, 2): different shifts
        (50, 100),  # across (0, 0) | (1, 0): different shifts
        (150, 100),  # across (0, 1) | (1, 1): the same shift
        (100, 100),  # the corner of four blocks, one of which differs
        (193.6, 30),  # 5 px half-width: its edge is 0.9 px from the seam (margin 1)
        (193.4, 30),  # 1.1 px away
        (205.4, 30),  # from the other side: 0.9 px
        (205.6, 30),  # 1.1 px
        (250, 150),  # inside block (1, 2)
    ]
    band = pa.nucleus_band(geo, _points(centres))
    want = [False, False, True, True, False, True, True, False, True, False, False]
    assert band.tolist() == want
    assert pa.nucleus_band(geo, {}) is None
    # every shift equal: no seam anywhere, whatever straddles an edge
    flat = pa.Geometry(
        np.eye(3), np.full((6, 2), 2.0), (2, 3), ((100, 100), (100, 100, 100))
    )
    assert not pa.nucleus_band(flat, _points(centres)).any()
    assert pa.seam_jumps(flat).max() == 0 and len(pa.seam_jumps(geo)) == 7


def test_a_probe_is_on_a_seam_when_its_own_nucleus_would_be():
    from benchmarks.regbench.cases import polyset_arrays

    geo = _seam_geometry()
    # three moving nuclei of half-width 6, 2 and 6 px; the truth is a shift by (150, 20)
    mov_centres = np.array([[40.0, 30.0], [47.0, 60.0], [-100.0, 100.0]])
    mov = cells.from_rings(
        [
            np.array([[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h]])
            for (x, y), h in zip(mov_centres, (6.0, 2.0, 6.0))
        ]
    )
    truth_ref = mov_centres + [150.0, 20.0]  # x = 190, 197, 50
    points = {
        **polyset_arrays(mov, "mov"),
        "truth_mov": mov_centres,
        "truth_ref": truth_ref,
        "truth_cell": np.array([0, 1, 2]),
    }
    # the seam is the edge at x = 199.5: 190 + 6 + 1 reaches it; 197 + 2 + 1 = 200 does
    # too; the third probe sits in block (1, 0) at (50, 120), 20 px below the seam at
    # y = 99.5
    assert pa.probe_band(geo, points).tolist() == [False, True, False]
    wide = dict(points, truth_ref=truth_ref + [3.0, 0.0])  # 193 + 6 + 1 >= 199.5
    assert pa.probe_band(geo, wide).tolist() == [True, True, False]
    # without the nucleus the probe is judged as a point with the same margin
    bare = {"truth_ref": np.array([[198.4, 50.0], [198.6, 50.0]])}
    assert pa.probe_band(geo, bare).tolist() == [False, True]
    assert pa.probe_band(geo, {}) is None


def test_the_dots_mask_is_rebuilt_from_the_run_record():
    geo = _seam_geometry()
    info = json.loads(json.dumps(pa.describe(geo, geo.shifts)))
    drop = pa.SELFCHECK_MASK(info, "seams")
    xy = np.array([[50.0, 50.0], [100.0, 50.0], [193.0, 50.0], [191.0, 50.0]])
    assert drop(xy).tolist() == [False, False, True, False]
    assert not pa.SELFCHECK_MASK({}, "seams")(xy).any()  # the affine stage
    with pytest.raises(SystemExit):
        pa.SELFCHECK_MASK(info, "overlap")


@pytest.fixture(scope="module")
def truth_case(tmp_path_factory):
    """One known-truth case at 1024 px (not a frozen one: no hash)."""
    root = tmp_path_factory.mktemp("palom_truth")
    suite = synthetic.truth_suite(n=1024, families=("bumps25",))
    name = sorted(suite)[0]
    synthetic.prepare_one(root, name, suite[name])
    return root, load_case(root / "synthetic" / name)


def _exact():
    import types

    def register(case, work, opts):
        fn = synthetic.forward_map(case.extra["spec"])
        yield "exact", PointMap(fn, MOV_TO_REF), {}

    return types.SimpleNamespace(
        NAME="exact", SCOPE="pair", VARIANTS=("exact",), register=register
    )


def _run_b256(case, out):
    return methods.run_case(
        "palom", case, out, {"block": "256"}, label="palom_b256", keep_work=False
    )


@pytest.mark.slow
def test_every_variant_gets_rows_off_paloms_seams_on_identical_nuclei(
    truth_case, tmp_path
):
    root, case = truth_case
    out = tmp_path / "out"
    assert _run_b256(case, out) == ["palom_b256_affine", "palom_b256"]
    methods.run_case(_exact(), case, out)
    points = load_points(case)
    n_ref, n_probes = polys(points, "ref").n_features, len(points["truth_ref"])
    with np.load(result_dir(out, "palom_b256", case) / "warped.npz") as z:
        band_ref, band_truth = z["band_ref"], z["band_truth"]
        assert "band" not in z.files
    assert band_ref.shape == (n_ref,) and band_truth.shape == (n_probes,)
    assert 0 < band_ref.sum() < n_ref and 0 < band_truth.sum() < n_probes
    # only the block grid and the accepted shifts enter: rebuilt from the run's record
    # of them, the mask is the same
    info = run_json(out, "palom_b256", case)["info"]
    geo = pa.Geometry.from_info(info["blocks"])
    assert np.array_equal(pa.nucleus_band(geo, points), band_ref)
    assert np.array_equal(pa.probe_band(geo, points), band_truth)
    assert info["band_fraction"] == pytest.approx(band_ref.mean())
    # the affine row carries no mask: the owner is the row's final variant
    with np.load(result_dir(out, "palom_b256_affine", case) / "warped.npz") as z:
        assert "band_ref" not in z.files and "band_truth" not in z.files

    res = score.score(root, out, ["synthetic"], seam_owners=("palom_b256",))
    rows = [r for r in res["nonband_cells"] if r["case_id"] == case.case_id]
    names = {r["method"] for r in rows}
    assert names == {
        v + ":non_palom_b256"
        for v in ("initial", "palom_b256_affine", "palom_b256", "exact", "truth")
    }
    kept = int((~band_ref).sum())
    assert {(r["n_ref_kept"], r["n_ref_all"]) for r in rows} == {(kept, n_ref)}
    assert {r["n_mov_kept"] == r["n_mov_all"] for r in rows} == {True}
    probes = [r for r in res["nonband_truth"] if r["case_id"] == case.case_id]
    assert {r["method"] for r in probes} == {
        v + ":non_palom_b256"
        for v in ("initial", "palom_b256_affine", "palom_b256", "exact")
    }
    left = int((~band_truth).sum())
    assert {(r["n_probes"], r["n_probes_all"]) for r in probes} == {(left, n_probes)}
    # the exact map is still exact on the probes that are left
    (exact,) = [r for r in probes if r["method"] == "exact:non_palom_b256"]
    assert exact["tre_um_max"] < 1e-6
    assert not [r for r in res["truth"] if ":non_" in r["method"]]
    assert not [r for r in res["cells"] if ":non_" in r["method"]]


def test_a_failed_palom_run_gives_no_off_seam_row_to_anyone(
    truth_case, tmp_path, monkeypatch
):
    root, case = truth_case
    out = tmp_path / "out"
    fp.install(monkeypatch.setitem, constrain_raises="the filter failed here")
    assert _run_b256(case, out) == ["palom_b256_affine"]
    methods.run_case(_exact(), case, out)
    res = score.score(root, out, ["synthetic"], seam_owners=("palom_b256",))
    assert res["nonband_cells"] == [] and res["nonband_truth"] == []
    (lost,) = res["nonband_missing"]
    assert lost["owner"] == "palom_b256" and lost["n_cases_without_mask"] == 1
    # the failed row itself is in the primary table, at the unregistered pose
    (row,) = [r for r in res["cells"] if r["method"] == "palom_b256"]
    assert row["imputed_initial"] and "the filter failed here" in row["problem"]
