"""The selfcheck gate is as strict one way as the other, and a mask cannot hide a defect
(review item 4). Dots are drawn directly, so no method runs: these are seconds."""

import json

import numpy as np
import pytest

from benchmarks.regbench import methods, score, variants
from benchmarks.regbench import selfcheck as sc
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap
from benchmarks.tests.regbench.test_pipeline import good, toy_case

HW = (1024, 1536)
SHIFT = np.array([7.0, -5.0])


def to_ref(xy):
    return np.asarray(xy, dtype=float) + SHIFT


def to_mov(xy):
    return np.asarray(xy, dtype=float) - SHIFT


FORWARD, BACKWARD = PointMap(to_ref, MOV_TO_REF), PointMap(to_mov, REF_TO_MOV)


@pytest.fixture(scope="module")
def dots():
    return sc.lattice_dots(HW, seed=3)


def image(dots_ref, blank=None):
    """What a writer would draw: the dots in the reference frame; ``blank`` (x0, x1)
    leaves a vertical band of the image empty."""
    out = sc.render_dots(dots_ref, 0, 0, HW[1], HW[0])
    if blank:
        out[:, blank[0] : blank[1]] = 0.0
    return out


def both(dots, plane, **kw):
    return (
        sc.compare(FORWARD, dots, plane, **kw),
        sc.compare(BACKWARD, dots, plane, expect=to_ref, **kw),
    )


def test_a_consistent_image_passes_both_ways(dots):
    for res in both(dots, image(to_ref(dots))):
        assert res["status"] == sc.OK and res["failed"] == [], res["failed"]
    back = sc.compare(BACKWARD, dots, image(to_ref(dots)), expect=to_ref)
    assert back["coverage_checked"] and back["n_covered"] == back["n_expected"] > 100


@pytest.mark.parametrize("blank", [(0, 700), (600, 900), (900, 1000)])
def test_an_image_with_a_part_undrawn_fails_in_both_directions(dots, blank):
    """The identity of the review's case: the points are right wherever something was
    drawn, and 45 %, 20 % or 7 % of the image shows nothing."""
    forward, backward = both(dots, image(to_ref(dots), blank))
    assert forward["status"] == sc.INCONSISTENT
    assert backward["status"] == sc.INCONSISTENT
    assert "drawn dots the image does not show" in backward["failed"]
    assert any(f.startswith("uncovered block") for f in backward["failed"])
    # what is drawn is right: every detection still comes back to its dot
    assert backward["n_found"] == backward["n_dots"]


def test_without_the_generators_map_the_coverage_half_is_not_run_and_says_so(dots):
    res = sc.compare(BACKWARD, dots, image(to_ref(dots), (0, 700)))
    assert res["coverage_checked"] is False and res["status"] == sc.OK


def test_a_dot_the_image_legitimately_cannot_show_is_not_demanded(dots):
    """The moving slide overhangs the reference: its outer dots land off the image."""
    far = np.vstack([dots, [[-300.0, 200.0], [HW[1] + 250.0, 400.0]]])
    backward = sc.compare(BACKWARD, far, image(to_ref(dots)), expect=to_ref)
    assert backward["status"] == sc.OK
    assert backward["n_expected"] < len(far)
    # and a writer that declares where it drew is held to that box only
    box = [0, 0, 1100, HW[0]]
    res = sc.compare(
        BACKWARD, dots, image(to_ref(dots), (1100, HW[1])), expect=to_ref, valid_box=box
    )
    assert res["status"] == sc.OK, res["failed"]
    undeclared = sc.compare(
        BACKWARD, dots, image(to_ref(dots), (1100, HW[1])), expect=to_ref
    )
    assert undeclared["status"] == sc.INCONSISTENT


# ── a mask ───────────────────────────────────────────────────────────────────────────
def band(x0, x1):
    return lambda xy: (np.asarray(xy)[:, 0] >= x0) & (np.asarray(xy)[:, 0] < x1)


def test_a_mask_that_hides_most_dots_is_not_checked(dots):
    hide = band(100, HW[1])  # 93 % of the image
    for res in both(dots, image(to_ref(dots)), drop=hide, mask_max=0.70):
        assert res["status"] == sc.NOT_CHECKED and res["masked_frac"] > 0.9
        assert "the mask hides" in res["cause"]
    for res in both(dots, image(to_ref(dots)), drop=band(0, 400), mask_max=0.70):
        assert res["status"] == sc.OK and 0.2 < res["masked_frac"] < 0.35


def overlap_share(hw, rnd, offset=(0.0, 0.0), follow=True, seed=0):
    """The share of the selfcheck's own dots that ASHLAR's overlap mask hides, from the
    adapter's geometry alone: its tile grid (``retile.tile_grid`` at the adapter's tile
    and overlap, padded, not clamped to the canvas) and ``tilemap.on_seam`` at its dot
    margin. ``follow``: every moving tile drawn where the round's true map puts its
    centre, plus ``offset`` px; otherwise where it was cut."""
    from benchmarks.ashlar import retile
    from benchmarks.regbench import tilemap as tm
    from benchmarks.regbench.datasets import synthetic
    from benchmarks.regbench.methods import ashlar as am

    h, w = hw
    grid = retile.tile_grid(w, h, am.TILE, am.OVERLAP)
    origins = np.array([[t.y, t.x] for t in grid], dtype=float)
    shift = np.zeros_like(origins)
    if follow:
        g = synthetic.forward_map(sc.round_spec(hw, None, seed, rnd))
        centres = origins[:, ::-1] + am.TILE / 2.0
        shift = (g(centres) - centres + np.asarray(offset, dtype=float))[:, ::-1]
    placement = dict(
        tile=am.TILE,
        mov_true_yx=origins,
        mov_pos_yx=origins + shift,
        ref_true_yx=origins,
        ref_pos_yx=origins,
    )
    dots = sc.lattice_dots(hw, seed + sc.ROUND_DOT_SEED * (rnd - 1))
    return float(tm.on_seam(placement, dots, am.DOT_MARGIN_PX).mean()), len(origins)


def test_the_overlap_cap_stands_above_the_measured_share():
    """The measurement the frozen cap is derived from (2026-10-11), repeated: ASHLAR's
    only selfcheck canvas, both rounds of its patient."""
    from benchmarks.regbench.methods import ashlar as am

    assert sc.MASK_MAX == {"overlap": 0.50, "seams": 0.10}
    assert (am.TILE, am.OVERLAP, am.DOT_MARGIN_PX) == (1024, 0.1, 8.0)
    reach = am.MAX_SHIFT_UM / sc.PIXEL_SIZE_UM  # 46 px: ASHLAR's whole maximum shift
    assert 46.0 < reach < 46.5
    on_grid, followed, pushed = [], [], []
    for rnd in (1, 2):
        share, n_tiles = overlap_share(sc.LARGE_HW, rnd, follow=False)
        assert n_tiles == 30  # 5 x 6
        on_grid.append(share)
        followed.append(overlap_share(sc.LARGE_HW, rnd)[0])
        for sx in (-reach, reach):
            for sy in (-reach, reach):
                pushed.append(overlap_share(sc.LARGE_HW, rnd, (sx, sy))[0])
    assert 0.21 < min(on_grid) <= max(on_grid) < 0.23, on_grid
    assert 0.27 < min(followed) <= max(followed) < 0.31, followed
    worst = max(pushed)
    assert 0.34 < worst < 0.37, pushed
    # the cap: the worst measured case plus a margin, and never half of the dots
    assert worst + 0.10 < sc.MASK_MAX["overlap"] <= worst + 0.16
    # twice ASHLAR's limit is still under it; an endless mosaic holds 0.21
    assert overlap_share(sc.LARGE_HW, 1, (2 * reach, 2 * reach))[0] < 0.40
    assert abs(1 - (1 - 102 / 922) ** 2 - 0.209) < 1e-3
    # PALOM: 8 px either side of every edge of 1024 px blocks
    assert 1 - (1 - 16 / 1024) ** 2 < sc.MASK_MAX["seams"] / 3


def test_the_small_canvas_is_no_longer_one_of_ashlars():
    """Dropped on 2026-10-11: 3 x 3 tiles, eight of them on the border."""
    for name in ("ashlar", "ashlar_clean", "affine_ashlar", "affine_ashlar_clean"):
        canvases = {sc.rounds_of(s)[0] for s in sc.sizes_for(name)}
        assert canvases == {sc.LARGE_HW}, name
    assert overlap_share(sc.DEFAULT_HW, 1, follow=False)[1] == 9


def test_a_detection_nothing_explains_fails_outside_the_mask_and_is_excused_inside(
    dots,
):
    hide = band(0, 400)
    stray_out = np.array([[900.0 + SHIFT[0], 311.0]])
    plane = image(np.vstack([to_ref(dots), stray_out]))
    res = sc.compare(FORWARD, dots, plane, drop=hide, mask_max=0.70)
    assert res["n_unexplained"] == 1 and res["status"] == sc.OK  # 1 of ~100: under 1 %
    many = np.column_stack([np.linspace(700, 1400, 12), np.full(12, 333.0)])
    plane = image(np.vstack([to_ref(dots), many]))
    res = sc.compare(FORWARD, dots, plane, drop=hide, mask_max=0.70)
    assert "detections the mapping puts no dot on" in res["failed"]
    # a second copy of a MASKED dot, a few px off (two tiles drew it): excused
    masked = dots[hide(dots)][:12]
    plane = image(np.vstack([to_ref(dots), to_ref(masked) + [6.0, 0.0]]))
    res = sc.compare(FORWARD, dots, plane, drop=hide, mask_max=0.70)
    assert res["n_unexplained"] == 0 and res["status"] == sc.OK


def test_a_map_wrong_under_the_mask_only_still_passes_and_reports_the_share(dots):
    """What a mask cannot see stays unseen: the cap and the reported share are the answer."""
    hide = band(0, 400)

    def wrong_there(xy):
        xy = np.asarray(xy, dtype=float)
        return to_ref(xy) + np.where(hide(xy)[:, None], [5.0, 0.0], [0.0, 0.0])

    res = sc.compare(
        PointMap(wrong_there, MOV_TO_REF), dots, image(to_ref(dots)), drop=hide
    )
    assert res["status"] == sc.OK and res["masked_frac"] == pytest.approx(
        0.26, abs=0.06
    )


# ── the image's shape ────────────────────────────────────────────────────────────────
def test_the_written_image_must_have_the_shape_it_claims():
    assert sc.shape_problem({}, (100, 200), (100, 200)) is None
    assert "claims the reference grid" in sc.shape_problem({}, (90, 200), (100, 200))
    assert "claims the reference grid" in sc.shape_problem(
        {"origin_xy": [0.0, 0.0]}, (100, 190), (100, 200)
    )
    mosaic = {"origin_xy": [-3.0, 4.0], "mosaic_shape": [120, 230]}
    assert sc.shape_problem(mosaic, (120, 230), (100, 200), [-3.0, 4.0]) is None
    assert "declares 120 x 230" in sc.shape_problem(mosaic, (100, 200), (100, 200))
    # an image off the grid that declares no shape has nothing to be held to
    assert (
        sc.shape_problem({"origin_xy": [5, 0]}, (90, 200), (100, 200), [5, 0]) is None
    )
    assert "valid_box" in sc.shape_problem(
        {"valid_box": [0, 0, 300, 100]}, (100, 200), (100, 200)
    )


def test_a_wrong_shape_fails_the_variant(tmp_path, monkeypatch):
    """A method that writes its image on another canvas than the reference's."""
    import types

    pair = sc.make_pair(tmp_path / "pair", (512, 768), None, seed=1)
    truth, dots = pair["truth"], pair["dots_mov_xy"]

    def write(shape):
        path = tmp_path / f"native_{shape[0]}.ome.tif"
        plane = np.rint(sc.render_dots(truth(dots), 0, 0, shape[1], shape[0]))
        from benchmarks.regbench import imageio

        imageio.write_channels(
            path,
            [plane.astype(np.uint16) * 0, plane.astype(np.uint16)],
            sc.CHANNEL_NAMES,
        )
        return {"files": [str(path)]}

    mod = types.SimpleNamespace(VARIANTS=("m",))
    rec = {"ok": True}
    pmap = PointMap(truth, MOV_TO_REF)
    good_one = sc._check_variant(
        mod, "m", True, {**rec, "native": write((512, 768))}, (pmap, {}), {}, pair, None
    )
    assert good_one["status"] == sc.OK and good_one["image_hw"] == [512, 768]
    cropped = sc._check_variant(
        mod, "m", True, {**rec, "native": write((512, 700))}, (pmap, {}), {}, pair, None
    )
    assert cropped["status"] == sc.INCONSISTENT
    assert any(f.startswith("image shape") for f in cropped["failed"])


# ── the pairs ────────────────────────────────────────────────────────────────────────
def test_two_pairs_have_a_moving_slide_larger_than_the_reference():
    first, swapped = sc.sizes_for("deeperhistreg_hr_gpu")
    assert swapped == (first[1], first[0]) and swapped[1][1] > swapped[0][1]
    for name in ("valis_mirage", "valis120", "valis120_hires"):
        ref_hw, mov_hw = sc.sizes_for(name)[1]
        assert mov_hw[0] > ref_hw[0] and mov_hw[1] > ref_hw[1]
        # odd overhang: a centred placement is half a pixel off the grid
        assert (mov_hw[0] - ref_hw[0]) % 2 == 1 and (mov_hw[1] - ref_hw[1]) % 2 == 1


def test_a_pair_with_an_overhanging_moving_slide_is_checkable(tmp_path):
    """The generator's own maps stand in for a method, in both directions."""
    ref_hw, mov_hw = (512, 768), (641, 897)
    pair = sc.make_pair(tmp_path, ref_hw, mov_hw, seed=2)
    assert pair["mov_hw"] == mov_hw and pair["ref_hw"] == ref_hw
    dots, truth = pair["dots_mov_xy"], pair["truth"]
    plane = sc.render_dots(truth(dots), 0, 0, ref_hw[1], ref_hw[0])
    forward = sc.compare(PointMap(truth, MOV_TO_REF), dots, plane)
    backward = sc.compare(
        PointMap(pair["truth_inverse"], REF_TO_MOV), dots, plane, expect=truth
    )
    assert forward["status"] == backward["status"] == sc.OK
    assert backward["n_expected"] < len(dots)  # the overhang is not demanded


def test_two_pairs_on_one_reference_canvas_get_folders_of_their_own(
    tmp_path, monkeypatch
):
    seen = []

    def fake_case(method, sub, size, *rest):
        seen.append(sub.name)
        return [{"status": "ok", "variants": {"m": {"status": "ok"}}}]

    monkeypatch.setattr(sc, "run_case", fake_case)
    sizes = [((64, 64), None), ((64, 64), (65, 67)), ((128, 64), None)]
    sizes += [((64, 64), None, None), ((64, 64), None, (65, 67))]
    rec = sc.run(good("m"), tmp_path, sizes=sizes, in_process=True)
    assert seen == [
        "64x64",
        "64x64_mov_65x67",
        "128x64",
        "patient_64x64",
        "patient_64x64_again",
    ]
    assert [c["folder"] for c in rec["cases"]] == seen
    assert [c["patient"] for c in rec["cases"]] == [False] * 3 + [True] * 2


# ── a row that was never checked says so in every table ──────────────────────────────
def test_a_stage_without_an_image_is_marked_in_every_table(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = toy_case(root, "P1_r1", "P1")
    for name in ("stare_rigid", "stare", "ashlar_clean"):
        methods.run_case(good(name), case, out)
    checks = out / "logs" / "selfcheck"
    for run, rec in (
        ("stare", {"status": "ok", "variants": {"stare_rigid": "no_stage_image", "stare": "ok"}}),
        ("ashlar", {"status": "ok", "variants": {"ashlar": "ok"}}),
    ):  # fmt: skip
        (checks / run).mkdir(parents=True)
        (checks / run / "selfcheck.json").write_text(json.dumps(rec))
    rows = variants.row_checks(checks)
    assert rows["stare_rigid"] == "no_stage_image" and rows["stare"] == "ok"
    assert rows["ashlar_clean"] == "via ashlar: ok"  # gated by another run's check
    assert rows["palom"] == variants.NO_RECORD
    assert set(rows) == set(variants.expected_variants())
    res = score.score(root, out, ["multiplex"], row_checks=rows, with_tissue=False)
    cells = {r["method"]: r["selfcheck_row"] for r in res["cells"]}
    assert cells["stare_rigid"] == "no_stage_image" and cells["initial"] == ""
    summary = {r["method"]: r["selfcheck_row"] for r in res["summary_cells"]}
    assert summary["stare_rigid"] == "no_stage_image"
    text = (out / "tables" / "summary.md").read_text()
    assert "| stare_rigid | 1 | NEVER CHECKED (no image of this stage) |" in text
    assert "| stare | 1 | ok |" in text
