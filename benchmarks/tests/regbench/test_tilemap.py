"""Tile geometry of the ASHLAR adapter: one shift per tile, the overlap bands, the anchor
cut, and the two-step map through ASHLAR's mosaic."""

import numpy as np

from benchmarks.regbench import tilemap

ORIGINS = np.array([[0, 0], [0, 900]])  # (y, x), tile 1000, 100 px overlap band
D = np.array([[2.0, 3.0], [-1.0, 4.0]])  # (dy, dx) per tile


def test_inside_one_tile_a_point_takes_that_tiles_shift():
    warp = tilemap.tile_point_map(ORIGINS, 1000, D)
    out = warp(np.array([[100.0, 100.0], [1700.0, 100.0]]))
    np.testing.assert_allclose(out, [[103.0, 102.0], [1704.0, 99.0]])


def test_in_the_band_the_shift_is_weighted_by_depth():
    warp = tilemap.tile_point_map(ORIGINS, 1000, D)
    mid = warp(np.array([[950.0, 500.0]]))  # 50 px deep in each tile
    np.testing.assert_allclose(mid, [[953.5, 500.5]])
    quarter = warp(np.array([[925.0, 500.0]]))  # 75 px deep in tile 0, 25 px in tile 1
    np.testing.assert_allclose(quarter, [[928.25, 501.25]])


def test_the_map_is_continuous_across_a_band_edge():
    warp = tilemap.tile_point_map(ORIGINS, 1000, D)
    a, b = warp(np.array([[899.999, 500.0], [900.001, 500.0]]))
    assert np.abs(a - b).max() < 0.01


def test_a_point_no_tile_covers_takes_its_nearest_tiles_shift():
    warp = tilemap.tile_point_map(ORIGINS, 1000, D)
    np.testing.assert_allclose(warp(np.array([[-5000.0, 400.0]])), [[-4997.0, 402.0]])


def test_a_single_tile_works():
    warp = tilemap.tile_point_map(np.array([[0, 0]]), 1000, np.array([[1.0, 1.0]]))
    np.testing.assert_allclose(warp(np.array([[10.0, 10.0]])), [[11.0, 11.0]])


def test_in_band_marks_points_covered_twice():
    xy = np.array([[100.0, 10.0], [950.0, 10.0], [1500.0, 10.0], [990.0, 10.0]])
    np.testing.assert_array_equal(
        tilemap.in_band(ORIGINS, 1000, xy), [False, True, False, True]
    )


def test_anchor_origins_for_a_pure_shift():
    # moving -> reference, xy
    m0 = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -20.0], [0.0, 0.0, 1.0]])
    got = tilemap.anchor_tile_origins(m0, ORIGINS, 1000)
    np.testing.assert_array_equal(got, [[20, -10], [20, 890]])  # (y, x)


def test_anchor_origins_follow_a_rotation():
    t = np.deg2rad(1.0)
    m0 = np.array(
        [[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0, 0, 1.0]]
    )
    got = tilemap.anchor_tile_origins(m0, np.array([[0, 0], [0, 20000]]), 1000)
    np.testing.assert_array_equal(got[1], [-358, 20006])


def test_anchor_cut_tiles_placed_on_the_grid_reproduce_the_anchor():
    """Moving tiles cut at the anchor's origins and placed by ASHLAR exactly on the nominal
    grid (L = nominal, E = c_ref) give D(t) = nominal - c_mov, and a point then lands at
    M0 @ p."""
    m0 = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -20.0], [0.0, 0.0, 1.0]])
    c_mov = tilemap.anchor_tile_origins(m0, ORIGINS, 1000)
    d_yx = ORIGINS - c_mov
    warp = tilemap.tile_point_map(c_mov, 1000, d_yx)
    p = np.array([[300.0, 400.0], [1500.0, 700.0], [960.0, 480.0]])
    np.testing.assert_allclose(warp(p), p + [10.0, -20.0])


# ── the two-step map through the mosaic, and the seam mask ───────────────────
def _placement():
    """Two tiles a side by side. The reference was stitched 7 px right and 4 px down of
    its slide, its second tile another 0.3 px right; the moving tiles were placed with
    different shifts."""
    c = np.array([[0.0, 0.0], [0.0, 900.0]])
    return {
        "tile": 1000,
        "ref_true_yx": c,
        "ref_pos_yx": c + [[4.0, 7.0], [4.0, 7.3]],
        "mov_true_yx": c,
        "mov_pos_yx": c + [[10.0, 30.0], [12.0, 18.0]],
    }


def test_a_point_goes_to_the_mosaic_by_its_own_tile_and_back_by_the_reference_tile():
    p = _placement()
    to_ref, to_mosaic = tilemap.placement_maps(p)
    xy = np.array([[400.0, 500.0], [1500.0, 500.0]])
    # moving slide -> mosaic: the moving tile's placement
    np.testing.assert_allclose(to_mosaic(xy), [[430.0, 510.0], [1518.0, 512.0]])
    # mosaic -> reference slide: minus the placement of the reference tile THERE
    np.testing.assert_allclose(to_ref(xy), [[423.0, 506.0], [1510.7, 508.0]])


def test_the_reference_tile_is_looked_up_where_the_point_lands_in_the_mosaic():
    """Reference tiles drawn 200 and 260 px right of where they were cut: mosaic x = 1050
    is under reference tile 0 only (200..1200; tile 1 starts at 1160), although in the
    SLIDE x = 1050 belongs to tile 1 only. The footprint in the mosaic decides."""
    p = _placement()
    p["ref_pos_yx"] = p["ref_true_yx"] + [[0.0, 200.0], [0.0, 260.0]]
    p["mov_pos_yx"] = p["mov_true_yx"] + [[0.0, 550.0], [0.0, 550.0]]
    to_ref, to_mosaic = tilemap.placement_maps(p)
    xy = np.array([[500.0, 500.0]])
    np.testing.assert_allclose(to_mosaic(xy), [[1050.0, 500.0]])
    np.testing.assert_allclose(to_ref(xy), [[850.0, 500.0]])


def test_two_reference_tiles_drawn_over_a_point_put_it_on_a_seam():
    """Reference tiles drawn at 200 and 1050: they overlap from 1050 to 1200 in the mosaic.
    A point landing at 1060 is under both, although the reference pixel it maps to (863)
    came from one tile and the moving side is clean."""
    p = _placement()
    p["ref_pos_yx"] = p["ref_true_yx"] + [[0.0, 200.0], [0.0, 150.0]]
    p["mov_pos_yx"] = p["mov_true_yx"] + [[0.0, 660.0], [0.0, 900.0]]
    to_ref, to_mosaic = tilemap.placement_maps(p)
    xy = np.array([[400.0, 500.0], [300.0, 500.0]])
    np.testing.assert_allclose(to_mosaic(xy)[:, 0], [1060.0, 960.0])
    assert 860 < to_ref(xy)[0, 0] < 870
    np.testing.assert_array_equal(tilemap.on_seam(p, xy), [True, False])


def test_the_seam_mask_is_true_wherever_the_image_is_not_one_unblended_copy():
    p = _placement()
    xy = np.array(
        [
            [400.0, 500.0],  # one moving tile, one reference tile
            [950.0, 500.0],  # two moving tiles hold it
            [880.0, 500.0],  # one moving tile, but it lands at mosaic x = 910:
            #                  under the second MOVING tile's footprint (918 on) no,
            #                  under the second REFERENCE tile (907.3 on)
            [1500.0, 500.0],  # clean again
        ]
    )
    np.testing.assert_array_equal(tilemap.on_seam(p, xy), [False, True, True, False])


def test_two_moving_tiles_drawn_over_a_point_put_it_on_a_seam():
    """The second moving tile drawn 200 px left of where it was cut: its footprint starts
    at 700. A point at 750 is in ONE tile of the slide, yet the mosaic cross-fades it with
    the other tile's content."""
    p = _placement()
    p["ref_pos_yx"] = p["ref_true_yx"].copy()
    p["mov_pos_yx"] = p["mov_true_yx"] + [[0.0, 0.0], [0.0, -200.0]]
    xy = np.array([[750.0, 500.0], [600.0, 500.0]])
    np.testing.assert_array_equal(tilemap.on_seam(p, xy), [True, False])


def test_a_point_two_moving_tiles_hold_is_on_a_seam_wherever_it_lands():
    """The second moving tile drawn 300 px right: the footprints no longer overlap, but a
    point both tiles were cut with is drawn twice (at 910 and at 1210), and the map's
    single position (940) is neither."""
    p = _placement()
    p["ref_pos_yx"] = np.array([[0.0, 0.0], [0.0, 1200.0]])
    p["mov_pos_yx"] = p["mov_true_yx"] + [[0.0, 0.0], [0.0, 300.0]]
    _, to_mosaic = tilemap.placement_maps(p)
    xy = np.array([[910.0, 500.0]])
    np.testing.assert_allclose(to_mosaic(xy), [[940.0, 500.0]])
    assert tilemap.on_seam(p, xy).all()


def test_the_seam_mask_has_a_margin_on_both_sides_of_a_join():
    p = _placement()
    p["ref_pos_yx"] = p["ref_true_yx"].copy()
    p["mov_pos_yx"] = p["mov_true_yx"].copy()
    xs = np.array([897.0, 899.5, 1000.5, 1003.0])
    xy = np.stack([xs, np.full(4, 500.0)], axis=1)
    np.testing.assert_array_equal(
        tilemap.on_seam(p, xy, margin=1.0), [False, True, True, False]
    )
    np.testing.assert_array_equal(
        tilemap.on_seam(p, xy, margin=5.0), [True, True, True, True]
    )


def test_a_point_no_tile_draws_is_on_a_seam():
    p = _placement()
    assert tilemap.on_seam(p, np.array([[500.0, 1500.0]])).all()  # below both tiles


def test_neighbour_jumps_are_the_difference_of_adjacent_tiles_shifts():
    d = np.array([[0.0, 0.0], [0.0, 3.0], [4.0, 0.0], [4.0, 3.0]])  # 2 x 2, row-major
    jumps = tilemap.neighbour_jumps(d, 2, 2)
    np.testing.assert_allclose(sorted(jumps), [3.0, 3.0, 4.0, 4.0])
    assert len(tilemap.neighbour_jumps(d[:1], 1, 1)) == 0
