"""The tissue-level block of the written images: ACCREDIT's score and its pieces.

Known answers on drawn masks and images (identical, shifted, rotated, blank, off the
reference grid, filled outside a valid box), the reduction read in bands, and the block's
place in the scorer: its own tables, never the primary ones.

The pieces are the released code's (LeeZhou-bearway/ACCREDIT at 9293473 ``[FULL]`` for the
files named in ``regbench.tissue``); the two rows of its published table used below are
from ``benchmark/evaluation/metrics_flat_all_4000px.csv`` ``[FULL: 96 rows]``.
"""

import csv
import json
import math
import types

import numpy as np
import pytest

from benchmarks.regbench import cells, cli, methods, tissue
from benchmarks.regbench.cases import Case, load_case, result_dir, write_case
from benchmarks.regbench.imageio import write_channels, write_nuclear
from benchmarks.regbench.mapping import MOV_TO_REF, PointMap
from benchmarks.regbench.score import score

cv2 = pytest.importorskip("cv2")


def disc(hw, centre, radius):
    y, x = np.mgrid[: hw[0], : hw[1]]
    return (x - centre[0]) ** 2 + (y - centre[1]) ** 2 <= radius**2


def textured(mask, seed=0, low=2000, high=30000):
    """A 16-bit image with fine structure inside ``mask`` and zero outside."""
    rng = np.random.default_rng(seed)
    blobs = cv2.GaussianBlur(rng.random(mask.shape).astype(np.float32), (0, 0), 2.0)
    blobs = (blobs - blobs.min()) / (blobs.max() - blobs.min())
    return np.where(mask, low + blobs * (high - low), 0).astype(np.uint16)


# ── the frozen values ─────────────────────────────────────────────────────────
def test_the_frozen_values_are_the_released_ones():
    assert tissue.FROZEN == {
        "eval_px": 4000,
        "band_width": 8,
        "nmi_bins": 64,
        "orb_features": 5000,
        "ransac_px": 5.0,
        "close_px": 15,
        "close_iterations": 2,
        "min_area_frac": 0.0001,
        "max_contour_points": 2000,
        "weights": {"td": 0.30, "bd": 0.30, "hd": 0.15, "ir": 0.10, "nmi": 0.15},
        "hd_max": 500.0,
        "ir_max": 0.15,
        "nmi_min": 1.0,
        "nmi_range": 0.15,
        "cov_min": 0.20,
    }
    assert sum(tissue.WEIGHTS.values()) == pytest.approx(1.0)
    assert tissue.SOURCE == "LeeZhou-bearway/ACCREDIT@9293473"
    assert "not part of any ranking" in tissue.LABEL and "H&E" in tissue.LABEL


def test_composite_matches_rows_of_the_released_table():
    """sample, method, qcs and the seven pieces, as published (rounded there)."""
    rows = [
        (0.7145, 0.9101, 0.6448, 113.03, 1.036, 0.2427, 0.6758, 0.0056),
        (0.6938, 0.9031, 0.6424, 106.04, 1.0161, 0.1872, 0.6663, 0.0059),
    ]
    for qcs, td, bd, hd, nmi, ir, cov, bg in rows:
        got = tissue.composite(
            {
                "tissue_dice": td,
                "boundary_dice": bd,
                "hausdorff_mean": hd,
                "nmi": nmi,
                "inlier_ratio": ir,
                "coverage": cov,
                "bg_coverage": bg,
            }
        )
        assert got == pytest.approx(qcs, abs=2e-4)


def test_the_two_penalties_multiply_the_weighted_sum():
    perfect = {
        "tissue_dice": 1.0,
        "boundary_dice": 1.0,
        "hausdorff_mean": 0.0,
        "nmi": 2.0,
        "inlier_ratio": 1.0,
        "coverage": 0.5,
        "bg_coverage": 0.0,
    }
    assert tissue.composite(perfect) == 1.0
    # coverage is of the FRAME: under 20 % of the frame scales the score down
    assert tissue.composite({**perfect, "coverage": 0.05}) == 0.25
    assert tissue.composite({**perfect, "bg_coverage": 0.4}) == 0.6
    # the contour term reaches zero at 500 px and stays there
    assert tissue.composite({**perfect, "hausdorff_mean": 999.0}) == 0.85
    assert tissue.composite({**perfect, "nmi": 1.0, "inlier_ratio": 0.0}) == 0.75


# ── the reduction ─────────────────────────────────────────────────────────────
def test_eval_shape_is_the_released_rule():
    assert tissue.eval_shape((40000, 50000)) == (3200, 4000)
    assert tissue.eval_shape((50000, 40000)) == (4000, 3200)
    assert tissue.eval_shape((1000, 800)) == (1000, 800)  # never enlarged
    assert tissue.eval_shape((4001, 3000)) == (4000, int(3000 * (4000 / 4001)))


def _brute(a, t):
    out = []
    for lo, hi in zip(t[:-1], t[1:]):
        total = 0.0
        for i, v in enumerate(a):
            total += v * max(0.0, min(hi, i + 1) - max(lo, i))
        out.append(total)
    return np.array(out)


def test_box_sums_are_exact_integrals():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 60000, 37).astype(np.uint16)
    for t in (
        np.linspace(0, 37, 9),  # non-integer factor
        np.linspace(-6.3, 44.1, 12),  # starts before and ends after the samples
        np.array(
            [0.0, 0.2, 0.7, 1.0, 5.5, 5.5, 36.9, 37.0]
        ),  # inside one sample, empty
        np.arange(38, dtype=float),  # unit boxes
    ):
        want = _brute(a, np.clip(t, 0, 37))
        assert np.allclose(tissue.box_sums(a, t), want, rtol=0, atol=1e-6)
    two = np.stack([a, a[::-1]])
    got = tissue.box_sums(two, np.linspace(0, 37, 9))
    assert np.allclose(got[1], _brute(a[::-1], np.linspace(0, 37, 9)))


class Lazy:
    """A ``(C, H, W)`` array that records what is asked of it."""

    def __init__(self, data):
        self.data, self.shape, self.dtype = data, data.shape, data.dtype
        self.reads = []

    def __getitem__(self, key):
        c, ys, xs = key
        self.reads.append((c, ys.stop - ys.start, xs.stop - xs.start))
        return self.data[c, ys, xs]


def test_reduce_is_the_area_mean():
    rng = np.random.default_rng(1)
    plane = rng.integers(0, 60000, (731, 912)).astype(np.uint16)
    eh, ew = 211, 263
    got = tissue.reduce_plane(Lazy(plane[None]), 0, (eh, ew), 912 / ew, 731 / eh)
    want = cv2.resize(plane.astype(np.float64), (ew, eh), interpolation=cv2.INTER_AREA)
    assert got.dtype == np.uint16 and got.shape == (eh, ew)
    assert np.abs(got.astype(float) - want).max() <= 0.5 + 1e-3  # rounding only


def test_reduce_reads_one_channel_in_bands_never_the_whole_plane():
    data = np.zeros((3, 4000, 600), dtype=np.uint16)
    data[1, 1000:3000, 100:500] = 500
    lazy = Lazy(data)
    got = tissue.reduce_plane(lazy, 1, (400, 60), 10.0, 10.0, band_rows=256)
    assert got[200, 30] == 500 and got[10, 30] == 0
    assert {c for c, _, _ in lazy.reads} == {1}
    assert len(lazy.reads) > 10
    assert max(rows for _, rows, _ in lazy.reads) <= 256 + 2


def test_origin_is_applied():
    """An image that starts off the reference grid (ASHLAR's mosaic) lands where its
    origin says: the same plane as the image cut to the reference grid."""
    rng = np.random.default_rng(5)
    on_grid = rng.integers(0, 60000, (600, 800)).astype(np.uint16)
    mosaic = np.full((700, 950), 7, dtype=np.uint16)
    mosaic[21:621, 37:837] = on_grid  # the reference's first pixel is at (37, 21)
    args = ((150, 200), 4.0, 4.0)
    want = tissue.reduce_plane(Lazy(on_grid[None]), 0, *args)
    got = tissue.reduce_plane(Lazy(mosaic[None]), 0, *args, origin_xy=(-37.0, -21.0))
    assert np.array_equal(got, want)
    # a wrong origin is a different plane
    off = tissue.reduce_plane(Lazy(mosaic[None]), 0, *args, origin_xy=(-33.0, -21.0))
    assert not np.array_equal(off, want)
    # an image that covers only part of the canvas is zero elsewhere
    part = tissue.reduce_plane(Lazy(on_grid[None, :200, :400]), 0, *args, (400.0, 0.0))
    assert np.array_equal(part[:50, 100:], want[:50, :100]) and not part[50:].any()
    assert not part[:, :100].any()


def test_a_fractional_origin_is_an_area_shift():
    plane = np.zeros((40, 40), dtype=np.uint16)
    plane[:, 20:] = 1000
    got = tissue.reduce_plane(Lazy(plane[None]), 0, (10, 10), 4.0, 4.0, (0.5, 0.0))
    assert got[3, 4] == 0 and got[3, 6] == 1000
    assert got[3, 5] == 875  # the edge sits half a pixel into this evaluation pixel


def test_fill_outside_valid_box_is_ignored():
    clean = np.zeros((600, 800), dtype=np.uint16)
    clean[100:500, 200:600] = 3000
    filled = clean.copy()
    filled[:, :150] = 65535  # a bright pad, as a writer with pad_value 255 leaves
    filled[550:] = 65535
    args = ((150, 200), 4.0, 4.0)
    want = tissue.reduce_plane(Lazy(clean[None]), 0, *args)
    box = [150, 0, 800, 550]
    got = tissue.reduce_plane(Lazy(filled[None]), 0, *args, valid_box=box)
    assert np.array_equal(got, want)
    assert tissue.reduce_plane(Lazy(filled[None]), 0, *args).max() == 65535


# ── the pieces ────────────────────────────────────────────────────────────────
def test_norm8_maps_the_positive_percentiles():
    a = np.zeros((100, 100), dtype=np.uint16)
    a[:50] = np.linspace(1000, 2000, 5000).reshape(50, 100)
    g = tissue.norm8(a)
    assert g.dtype == np.uint8 and not g[50:].any()  # zero stays zero
    p1, p99 = np.percentile(a[a > 0], [1, 99])
    assert g[a == a.max()].min() == 255 and g[0, 0] == 0
    mid = a[25, 50]
    assert int(g[25, 50]) == int((mid - p1) / (p99 - p1) * 255)
    few = np.zeros((100, 100), dtype=np.uint16)
    few[0, :99] = 500
    assert not tissue.norm8(few).any()  # under 100 positive pixels: nothing


def test_the_mask_closes_gaps_and_drops_specks():
    g = np.zeros((1000, 1000), dtype=np.uint8)
    g[200:500, 200:400] = 200
    g[200:500, 410:600] = 200  # a 10 px gap: closed by the 15 px ellipse
    g[800:809, 800:809] = 200  # 81 px, under 1e-4 of the frame (100 px): dropped
    g[700:720, 100:120] = 200  # 400 px: kept
    mask, thr = tissue.make_mask(g, with_threshold=True)
    assert mask[350, 405] and mask[350, 300]
    assert not mask[804, 804] and mask[710, 110]
    assert 0 <= thr < 200


def test_identical_images_score_one():
    mask = disc((900, 1000), (500, 450), 330)
    plane = textured(mask)
    got = tissue.score_planes(plane, plane, eval_um=4.0)
    assert got["status"] == "ok"
    assert got["tissue_dice"] == pytest.approx(1.0)
    assert got["boundary_dice"] == pytest.approx(1.0)
    assert got["hausdorff_mean"] == 0.0 and got["hausdorff_mean_um"] == 0.0
    assert got["inlier_ratio"] > 0.95 and got["n_matches"] > 500
    assert got["nmi"] == pytest.approx(2.0, abs=1e-6)
    assert got["ncc"] == pytest.approx(1.0)
    assert got["bg_coverage"] == 0.0
    assert got["coverage"] == pytest.approx(mask.mean(), abs=0.01)
    assert got["qcs"] == 1.0
    assert got["opencv"] == cv2.__version__ and got["definition"] == tissue.DEFINITION


def test_coverage_is_of_the_frame_so_a_small_tissue_scores_low_even_when_identical():
    """The released coverage is mask / frame, not mask / reference tissue: a perfect
    match of a tissue that fills 5 % of the frame is scaled by 0.05 / 0.20."""
    mask = disc((1000, 1000), (500, 500), 126)
    plane = textured(mask)
    got = tissue.score_planes(plane, plane)
    assert got["tissue_dice"] == pytest.approx(1.0)
    assert got["coverage"] == pytest.approx(0.05, abs=0.003)
    assert got["qcs"] == pytest.approx(got["coverage"] / 0.20, abs=1e-3)


def _lens(r, d):
    return 2 * r * r * math.acos(d / (2 * r)) - 0.5 * d * math.sqrt(4 * r * r - d * d)


def test_a_known_shift_moves_tissue_dice_by_the_lens_area():
    hw, r, d = (1200, 1400), 300, 60
    a, b = disc(hw, (600, 600), r), disc(hw, (600 + d, 600), r)
    want = _lens(r, d) / (math.pi * r * r)  # 2 |A & B| / (|A| + |B|), equal discs
    assert tissue.tissue_dice(a, b) == pytest.approx(want, abs=2e-3)
    # the band-restricted Dice falls much further for the same shift
    assert tissue.boundary_dice(a, b) < tissue.tissue_dice(a, b) - 0.3
    # and the background term is the crescent over the reference's background
    ref_mask, mov_mask = a, b
    crescent = (math.pi * r * r - _lens(r, d)) / (a.size - math.pi * r * r)
    got = tissue.components(
        textured(a).astype(np.uint8), ref_mask, textured(b).astype(np.uint8), mov_mask
    )
    assert got["bg_coverage"] == pytest.approx(crescent, rel=0.01)


def test_boundary_dice_of_concentric_discs_is_the_band_formula():
    """Discs of radius R and R + k, k > 16: the union of the two bands holds the annulus
    [R - 8, R] of the small disc and [R - 8, R + 8] + [R + k - 8, R + k] of the large."""
    hw, r, k = (1000, 1000), 250, 40
    a, b = disc(hw, (500, 500), r), disc(hw, (500, 500), r + k)

    def ring(lo, hi):
        return math.pi * (hi * hi - lo * lo)

    small = ring(r - 8, r)
    large = ring(r - 8, r + 8) + ring(r + k - 8, r + k)
    want = 2 * small / (small + large)
    assert tissue.boundary_dice(a, b) == pytest.approx(want, abs=0.02)
    assert tissue.boundary_dice(a, a) == pytest.approx(1.0)
    assert tissue.boundary_dice(a, np.zeros(hw, bool)) == 0.0


def test_the_contour_distance_is_a_mean_not_a_maximum():
    hw, r, k = (1000, 1000), 250, 40
    a, b = disc(hw, (500, 500), r), disc(hw, (500, 500), r + k)
    mean, p95 = tissue.contour_distance(a, b)
    assert mean == pytest.approx(k, abs=1.0) and p95 == pytest.approx(k, abs=1.5)
    # a 150 px spike on one mask: the maximum would be ~150, the mean barely moves
    spike = a.copy()
    spike[495:505, 740:900] = True
    mean, p95 = tissue.contour_distance(a, spike)
    assert mean < 15 and p95 > 50
    # only the single longest contour of each mask is read: a detached piece is unseen
    far = a.copy()
    far[50:90, 50:90] = True
    assert tissue.contour_distance(a, far) == (0.0, 0.0)
    assert tissue.contour_distance(a, np.zeros(hw, bool)) == (999.0, 999.0)


def test_nmi_is_one_for_independent_images_and_two_for_identical_ones():
    rng = np.random.default_rng(0)
    a = rng.integers(1, 256, (600, 600)).astype(np.uint8)
    b = rng.integers(1, 256, (600, 600)).astype(np.uint8)
    assert tissue.nmi(a, a) == pytest.approx(2.0, abs=1e-6)
    assert tissue.nmi(a, b) == pytest.approx(1.0, abs=0.01)
    # only pixels above zero in BOTH images count; under 100 of them is 1.0
    sparse = np.zeros_like(a)
    sparse[0, :50] = 9
    assert tissue.nmi(a, sparse) == 1.0


def test_the_inlier_ratio_is_blind_to_a_global_shift():
    """RANSAC fits its own similarity: a slide left 40 px off scores like a registered
    one on this piece, while the mask pieces fall."""
    hw = (900, 1100)
    scene = textured(np.ones(hw, bool), seed=4)
    a = np.where(disc(hw, (500, 450), 330), scene, 0).astype(np.uint16)
    b = np.roll(a, 40, axis=1)
    same, moved = tissue.score_planes(a, a), tissue.score_planes(a, b)
    assert moved["inlier_ratio"] > 0.8 and same["inlier_ratio"] > 0.8
    assert moved["s_ir"] == 1.0  # the term saturates at 0.15
    assert moved["tissue_dice"] < 0.95 and moved["boundary_dice"] < 0.6
    assert moved["hausdorff_mean"] > 5
    again = tissue.score_planes(a, b)  # the same call twice: OpenCV's RANSAC is seeded
    for key in ("inlier_ratio", "n_inliers", "n_matches", "qcs"):
        assert again[key] == moved[key], key


def test_the_inlier_model_is_a_similarity_not_an_affine():
    """A sheared copy: its matches fit an affine, not the released four-parameter
    similarity at 5 px."""
    hw = (900, 1100)
    a = tissue.norm8(textured(np.ones(hw, bool), seed=4))
    shear = np.float32([[1, 0.12, -50], [0, 1, 0]])
    b = cv2.warpAffine(a, shear, (hw[1], hw[0]))
    ratio, inliers, matches = tissue.orb_inlier_ratio(a, b)
    assert matches > 500 and ratio < 0.5
    assert tissue.orb_inlier_ratio(a, a)[0] > 0.95


def test_a_known_rotation_gives_the_dice_of_the_rotated_ellipse():
    hw, centre, axes = (1000, 1200), (600, 500), (420, 180)
    y, x = np.mgrid[: hw[0], : hw[1]]

    def ellipse(deg):
        t = math.radians(deg)
        u = (x - centre[0]) * math.cos(t) + (y - centre[1]) * math.sin(t)
        v = -(x - centre[0]) * math.sin(t) + (y - centre[1]) * math.cos(t)
        return (u / axes[0]) ** 2 + (v / axes[1]) ** 2 <= 1.0

    a, b = ellipse(0), ellipse(15)
    want = 2.0 * (a & b).sum() / (a.sum() + b.sum())
    got = tissue.score_planes(textured(a, 1), textured(b, 2))
    assert got["tissue_dice"] == pytest.approx(want, abs=0.01)
    assert 0.5 < want < 0.9
    flat = tissue.score_planes(textured(a, 1), textured(a, 2))
    assert flat["tissue_dice"] > 0.99 > got["tissue_dice"]
    assert got["boundary_dice"] < flat["boundary_dice"] - 0.3
    assert got["hausdorff_mean"] > flat["hausdorff_mean"] + 10
    assert got["qcs"] < flat["qcs"]


def test_a_blank_output_is_failed_not_nan():
    ref = textured(disc((800, 900), (450, 400), 300))
    for blank in (np.zeros_like(ref), np.where(disc(ref.shape, (10, 10), 5), ref, 0)):
        got = tissue.score_planes(ref, blank)
        assert got["status"] == "no_tissue" and got["qcs"] == 0.0
        assert got["coverage"] < 1e-3 and "tissue_dice" not in got
        assert all(v == v for v in got.values() if isinstance(v, float))  # no NaN
    # a blank reference is scored: every tissue pixel of the image is on background
    got = tissue.score_planes(np.zeros_like(ref), ref)
    assert got["status"] == "ok" and got["tissue_dice"] == 0.0
    assert got["bg_coverage"] == pytest.approx(got["coverage"])
    assert got["hausdorff_mean"] == 999.0 and got["hausdorff_mean_um"] is None


def test_the_pieces_are_computed_inside_the_written_images_bounding_box():
    """The released crop: reference tissue OUTSIDE the bounding box of the written
    image's mask is not seen by the five pieces. An image that holds one of two
    identical pieces of tissue, in place, has a tissue Dice of 1 (the full-frame Dice is
    2/3); only the frame-wide coverage differs."""
    hw = (1000, 1600)
    left, right = disc(hw, (400, 500), 300), disc(hw, (1200, 500), 300)
    ref = textured(left | right)
    got = tissue.score_planes(ref, np.where(left, ref, 0))
    assert got["crop_box"][2] < 800
    assert got["tissue_dice"] == pytest.approx(1.0, abs=1e-3)
    assert tissue.tissue_dice(left | right, left) == pytest.approx(2 / 3, abs=1e-3)
    assert got["bg_coverage"] < 1e-3
    assert got["coverage"] == pytest.approx(left.mean(), abs=0.01)
    assert got["qcs"] > 0.84  # 0.176 / 0.20 of an otherwise perfect score


def test_planes_of_two_sizes_are_refused():
    ref = textured(disc((800, 900), (450, 400), 300))
    with pytest.raises(ValueError, match="plane"):
        tissue.score_planes(ref, ref[:, :-1])


# ── in the driver and the scorer ──────────────────────────────────────────────
HW = (1200, 1600)
SHIFT = np.array([48.0, -36.0])
GRID = np.array(
    [(x, y) for x in range(500, 1100, 60) for y in range(300, 900, 60)], dtype=float
)


def _squares(centres, half=5.0):
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    return cells.from_rings([c + corners for c in centres])


@pytest.fixture(scope="module")
def slide_case(tmp_path_factory):
    """A case WITH images: a textured tissue, the moving slide the reference moved by
    ``-SHIFT`` (so the true map is ``+SHIFT``), three channels, the nuclear one second."""
    root = tmp_path_factory.mktemp("tissue_cases")
    scene = textured(np.ones(HW, bool), seed=11)
    y, x = np.mgrid[: HW[0], : HW[1]]
    blob = ((x - 800) / 520) ** 2 + ((y - 600) / 400) ** 2 <= 1.0
    ref = np.where(blob, scene, 0).astype(np.uint16)
    mov = np.roll(ref, (-int(SHIFT[1]), -int(SHIFT[0])), axis=(0, 1))
    d = root / "images"
    d.mkdir()
    write_nuclear(d / "ref.ome.tif", ref, 0.5)
    other = np.full(HW, 40000, dtype=np.uint16)
    write_channels(
        d / "mov.ome.tif", np.stack([other, mov, other]), ["a", "n", "b"], 0.5
    )
    case = Case(
        dataset="multiplex",
        case_id="P1_r1",
        group="P1",
        ref_nuclear=str(d / "ref.ome.tif"),
        mov_nuclear=str(d / "mov.ome.tif"),
        ref_image=str(d / "ref.ome.tif"),
        mov_image=str(d / "mov.ome.tif"),
        modality="fluorescence",
        diagonal=2000.0,
        pixel_size_um=0.5,
        ref_hw=list(HW),
        mov_hw=list(HW),
        extra={"ref_channel": 0, "mov_channel": 1},
    )
    write_case(root, case, cells_mov=_squares(GRID - SHIFT), cells_ref=_squares(GRID))
    return root, load_case(root / "multiplex" / "P1_r1"), ref, mov


def _writer(image_of, origin=None, valid_box=None, variants=("drawn",), fail=False):
    """A method whose map is the true shift and whose writer draws ``image_of(case)``."""
    final = variants[-1]

    def register(case, work, opts):
        for v in variants:
            yield (
                v,
                PointMap(lambda xy: np.asarray(xy, float) + SHIFT, MOV_TO_REF),
                {"_state": case},
            )

    def write_native(state, case_id, dest_dir):
        if fail:
            raise RuntimeError("the disk is full")
        image = image_of(state)
        path = write_channels(dest_dir / "out.ome.tif", image, ["a", "n", "b"], 0.5)
        native = {"files": [str(path)], "format": "ome-tiff", "nuclear_channel": 1}
        if origin is not None:
            native["origin_xy"] = list(origin)
        if valid_box is not None:
            native["valid_box"] = list(valid_box)
        return native

    return types.SimpleNamespace(
        NAME=final,
        SCOPE="pair",
        VARIANTS=tuple(variants),
        register=register,
        write_native=write_native,
    )


def _stack(plane):
    other = np.full(plane.shape, 40000, dtype=np.uint16)
    return np.stack([other, plane, other])


def _rows(out, table):
    path = out / "tables" / f"{table}.csv"
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def scored(slide_case, tmp_path_factory):
    """Four methods on one case: the reference redrawn exactly (``drawn``, after a rigid
    stage that writes nothing), the same on a padded mosaic with an origin and a bright
    pad (``mosaic``), the moving slide left where it was (``lazy``), and one whose writer
    fails (``broken``); then scored."""
    root, case, ref, mov = slide_case
    out = tmp_path_factory.mktemp("tissue_out")

    def mosaic(_):
        pad = np.full((HW[0] + 90, HW[1] + 130), 65535, dtype=np.uint16)
        pad[21 : 21 + HW[0], 37 : 37 + HW[1]] = ref
        pad[:, : 37 + 60] = 65535  # the fill reaches 60 px INTO the reference's extent
        return _stack(pad)

    box = [37 + 60, 21, 37 + HW[1], 21 + HW[0]]
    runs = {
        "drawn": _writer(lambda _: _stack(ref), variants=("drawn_rigid", "drawn")),
        "mosaic": _writer(mosaic, (-37.0, -21.0), box, variants=("mosaic",)),
        "lazy": _writer(lambda _: _stack(mov), variants=("lazy",)),
        "broken": _writer(None, variants=("broken",), fail=True),
    }
    for mod in runs.values():
        methods.run_case(mod, case, out, {"workers": 1})
    tables = score(root, out, datasets=["multiplex"], expected=["never_ran"])
    return root, case, out, tables


def _one(rows, variant):
    (row,) = [r for r in rows if r["variant"] == variant]
    return row


def test_the_driver_reduces_each_written_image(scored):
    _, case, out, _ = scored
    run = json.loads((result_dir(out, "drawn", case) / "run.json").read_text())
    rec = run["tissue"]
    assert rec["eval_hw"] == list(HW) and rec["channel"] == 1  # 1600 px: not enlarged
    assert rec["image_shape"] == [3, *HW] and rec["seconds"] >= 0
    plane = tissue.read_plane(rec["file"])
    assert plane.shape == HW and plane.dtype == np.uint16
    assert rec["file"].endswith("drawn/multiplex/P1_r1/tissue/moving_eval.tif")
    rigid = json.loads((result_dir(out, "drawn_rigid", case) / "run.json").read_text())
    assert "tissue" not in rigid and rigid["native"] is None
    broken = json.loads((result_dir(out, "broken", case) / "run.json").read_text())
    # a writer that failed leaves a failed row (cause `writer`), not an ok one
    assert "tissue" not in broken and broken["ok"] is False
    assert broken["cause"] == "writer" and "disk is full" in broken["error"]


def test_a_selfcheck_run_is_not_reduced(slide_case, tmp_path):
    _, case, ref, _ = slide_case
    mod = _writer(lambda _: _stack(ref))
    methods.run_case(mod, case, tmp_path, {"workers": 1, "stage_images": "1"})
    run = json.loads((result_dir(tmp_path, "drawn", case) / "run.json").read_text())
    assert run["native"]["files"] and "tissue" not in run


def test_every_written_image_and_the_unregistered_slide_have_a_row(scored):
    _, _, out, tables = scored
    rows = tables["tissue"]
    assert sorted(r["variant"] for r in rows) == ["drawn", "initial", "lazy", "mosaic"]
    assert {r["variant"]: r["source"] for r in rows} == {
        "drawn": "method",
        "mosaic": "method",
        "lazy": "method",
        "initial": "benchmark",
    }
    drawn, initial, lazy = (_one(rows, v) for v in ("drawn", "initial", "lazy"))
    assert drawn["qcs"] == 1.0 and drawn["tissue_dice"] == pytest.approx(1.0)
    assert drawn["eval_um_per_px"] == 0.5 and drawn["hausdorff_mean_um"] == 0.0
    # the moving slide left in place IS the unregistered slide
    for key in ("qcs", "tissue_dice", "boundary_dice", "hausdorff_mean", "nmi"):
        assert lazy[key] == initial[key], key
    assert initial["tissue_dice"] < 0.97 and initial["qcs"] < drawn["qcs"]
    # 48 px of shift is 24 um here; the mean contour distance sees part of it
    assert 5 < initial["hausdorff_mean_um"] < 30
    assert _rows(out, "tissue") and len(_rows(out, "tissue")) == 4


def test_an_image_off_the_reference_grid_scores_as_the_same_image_on_it(scored):
    """ASHLAR-like: a padded mosaic whose first pixel is not the reference's, with a
    bright pad outside its valid box."""
    _, _, _, tables = scored
    drawn, mosaic = (_one(tables["tissue"], v) for v in ("drawn", "mosaic"))
    for key in ("qcs", "tissue_dice", "boundary_dice", "nmi", "coverage", "n_matches"):
        assert mosaic[key] == drawn[key], key
    assert mosaic["bg_coverage"] == 0.0


def test_variants_without_a_written_image_are_listed_with_the_reason(scored):
    _, _, out, tables = scored
    why = {m["variant"]: m["reason"] for m in tables["tissue_missing"]}
    assert set(why) == {"drawn_rigid", "broken", "never_ran"}
    assert why["drawn_rigid"] == "the method writes no image for this variant"
    assert why["broken"] == "the run failed (writer): no image"
    assert why["never_ran"] == "not run"
    assert len(_rows(out, "tissue_missing")) == 3
    assert not any(r["variant"] in why for r in tables["tissue"])


def test_the_summary_counts_cases_and_composites_under_a_quarter(scored):
    _, _, out, tables = scored
    summary = {r["variant"]: r for r in tables["tissue_summary"]}
    assert set(summary) == {"drawn", "mosaic", "lazy", "initial"}
    assert summary["drawn"]["n_scored"] == summary["drawn"]["n_cases"] == 1
    assert (
        summary["drawn"]["qcs_median"] == 1.0 and summary["drawn"]["n_below_025"] == 0
    )
    rows = [
        {"dataset": "d", "variant": "v", "source": "method", "qcs": q}
        for q in (0.1, 0.2, 0.9)
    ]
    lost = [{"dataset": "d", "variant": "v", "reason": "not run"}]
    (got,) = tissue.summarise(rows, lost)
    assert (got["n_scored"], got["n_cases"], got["n_below_025"]) == (3, 4, 2)
    assert got["qcs_median"] == 0.2 and got["qcs_worst"] == 0.1
    text = (out / "tables" / "summary.md").read_text()
    assert (
        "## Tissue-level agreement of the written images (secondary; not ranked)"
        in text
    )
    assert tissue.LABEL in text and "outside what the score was validated for" in text
    assert "### No tissue row" in text


def test_tissue_never_enters_the_primary_tables(scored, tmp_path):
    root, _, out, tables = scored
    for name in ("cells", "summary_cells", "resources", "failures"):
        rows = _rows(out, name)
        assert rows, name
        for key in rows[0]:
            assert "qcs" not in key and "tissue" not in key and "nmi" not in key, key
    # and the primary rows are the same with the block switched off
    plain = score(
        root, out, datasets=["multiplex"], expected=["never_ran"], with_tissue=False
    )
    # (value by value: the rows come back from the case's record, where a NaN is no
    # longer the same object, and two lists holding different NaNs are not equal)
    diff = [
        (i, k, a[k], b[k])
        for i, (a, b) in enumerate(zip(plain["cells"], tables["cells"]))
        for k in a
        if not (a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]))
    ]
    assert not diff and len(plain["cells"]) == len(tables["cells"]), diff[:5]
    assert plain["summary_cells"] == tables["summary_cells"]
    assert "tissue" not in plain and not _rows(out, "tissue")
    assert "Tissue-level" not in (out / "tables" / "summary.md").read_text()
    score(
        root, out, datasets=["multiplex"], expected=["never_ran"]
    )  # leave it as found


def test_without_opencv_a_row_is_read_from_disk_or_listed_as_missing(
    scored, monkeypatch, tmp_path
):
    root, case, out, tables = scored
    monkeypatch.setattr(tissue, "have_cv2", lambda: False)
    job = (str(out), case, ["initial", "drawn", "lazy"], True)
    rows, missing, _ = tissue.score_case(job)  # computed before: read back
    assert {r["variant"]: r["qcs"] for r in rows} == {
        r["variant"]: r["qcs"] for r in tables["tissue"] if r["variant"] != "mosaic"
    }
    assert not missing
    (result_dir(out, "lazy", case) / "tissue" / tissue.RESULT).unlink()
    rows, missing, _ = tissue.score_case(job)
    assert [m["variant"] for m in missing] == ["lazy"]
    assert "OpenCV (cv2) is not importable" in missing[0]["reason"]
    monkeypatch.undo()
    rows, missing, _ = tissue.score_case(job)  # and computed again where it is
    assert len(rows) == 3 and not missing


def test_a_stale_result_is_computed_again(scored):
    _, case, out, _ = scored
    path = result_dir(out, "drawn", case) / "tissue" / tissue.RESULT
    rec = json.loads(path.read_text())
    path.write_text(json.dumps({**rec, "definition": "older", "qcs": 0.123}))
    rows, _, _ = tissue.score_case((str(out), case, ["drawn"], True))
    assert _one(rows, "drawn")["qcs"] == 1.0


def test_a_cached_plane_is_of_the_image_and_of_the_code_it_was_made_from(
    slide_case, tmp_path, monkeypatch
):
    """The reference's reduced plane is kept under ``cache/tissue``. It is read again
    only while the image it was reduced from and ``tissue.py`` are what they were: a
    plane of another image, of other code, or one with no record of either, is made
    again, and a result computed by other code is computed again."""
    _, case, ref, _ = slide_case
    out = tmp_path / "out"
    plane = tissue.cache_dir(out, case) / "reference_eval.tif"
    first = tissue.reference_plane(out, case)
    record = json.loads(tissue.plane_record(plane).read_text())
    assert record["code_sha256"] == tissue.code_sha256()
    assert record["image"]["sample_sha256"] and record["channel"] == 0
    made = []
    real = tissue.reduce

    def counted(*a, **k):
        made.append(a[0])
        return real(*a, **k)

    monkeypatch.setattr(tissue, "reduce", counted)
    assert np.array_equal(tissue.reference_plane(out, case), first) and not made
    # the image changes under the same name: the plane is the new image's
    image = tmp_path / "ref.ome.tif"
    write_nuclear(image, ref, 0.5)
    moved = types.SimpleNamespace(**{**vars(case), "ref_nuclear": str(image)})
    assert np.array_equal(tissue.reference_plane(out, moved), first) and len(made) == 1
    write_nuclear(image, ref // 2, 0.5)
    halved = tissue.reference_plane(out, moved)
    assert len(made) == 2 and not np.array_equal(halved, first)
    assert np.array_equal(tissue.reference_plane(out, moved), halved) and len(made) == 2
    # the code changes: every plane is made again
    monkeypatch.setattr(tissue, "code_sha256", lambda: "other code")
    tissue.reference_plane(out, moved)
    assert len(made) == 3
    # a plane with no record of what it was made from is not trusted
    tissue.plane_record(plane).unlink()
    tissue.reference_plane(out, moved)
    assert len(made) == 4


def test_a_result_computed_by_other_code_is_computed_again(scored, monkeypatch):
    _, case, out, _ = scored
    path = result_dir(out, "drawn", case) / "tissue" / tissue.RESULT
    tissue.score_case((str(out), case, ["drawn"], True))
    rec = json.loads(path.read_text())
    assert rec["code_sha256"] == tissue.code_sha256()
    path.write_text(json.dumps({**rec, "qcs": 0.123}))
    rows, _, _ = tissue.score_case((str(out), case, ["drawn"], True))
    assert _one(rows, "drawn")["qcs"] == 0.123  # same code, same planes: it is read
    assert "code_sha256" not in _one(rows, "drawn")
    path.write_text(json.dumps({**rec, "qcs": 0.123, "code_sha256": "other code"}))
    rows, _, _ = tissue.score_case((str(out), case, ["drawn"], True))
    assert _one(rows, "drawn")["qcs"] == 1.0


def test_a_plane_the_driver_did_not_reduce_is_filled_in_from_the_written_image(scored):
    _, case, out, _ = scored
    d = result_dir(out, "drawn", case) / "tissue"
    for f in d.iterdir():
        f.unlink()
    rows, missing, _ = tissue.score_case((str(out), case, ["drawn"], True))
    assert _one(rows, "drawn")["qcs"] == 1.0 and not missing
    assert (d / tissue.MOVING_PLANE).exists()


def test_the_cli_computes_the_pieces_for_score_to_read(scored, capsys):
    root, case, out, _ = scored
    path = result_dir(out, "drawn", case) / "tissue" / tissue.RESULT
    path.unlink()
    argv = ["tissue", "--cases", str(root), "--out", str(out), "--dataset", "multiplex"]
    assert cli.main([*argv, "--only-run", "--workers", "1"]) == 0
    assert "tissue: 4 row(s), 2 without one" in capsys.readouterr().out
    assert json.loads(path.read_text())["qcs"] == 1.0


# ── known-truth cases: the truth plane and what the block resolves ────────────
def test_the_truth_plane_and_its_shifted_copies(synthetic_cases, tmp_path):
    case = synthetic_cases["rigid"]
    assert tissue.has_known_map(case)
    rows, missing, dose = tissue.score_case((str(tmp_path), case, ["initial"], True))
    assert not missing, missing
    truth, initial = _one(rows, "truth"), _one(rows, "initial")
    assert truth["source"] == "benchmark"
    assert (
        truth["tissue_dice"] > 0.97 and truth["tissue_dice"] >= initial["tissue_dice"]
    )
    assert truth["boundary_dice"] > initial["boundary_dice"]
    assert [r["shift_um"] for r in dose] == [0.0, *tissue.DOSE_UM]
    assert dose[0]["boundary_dice"] == pytest.approx(truth["boundary_dice"], abs=0.02)
    bd = [r["boundary_dice"] for r in dose]
    assert bd[-1] < bd[0] - 0.2  # 100 um is seen
    assert all(r["variant"] == "truth" and r["family"] == case.family for r in dose)
    summary = tissue.summarise(dose, [], by=("dataset", "shift_um"))
    assert [r["shift_um"] for r in summary] == [0.0, *tissue.DOSE_UM]
