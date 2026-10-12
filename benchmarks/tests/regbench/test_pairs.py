"""The cell scores: mirage's pairing under a fixed gate, the null and the two-sided curve."""

import math

import numpy as np

from benchmarks.regbench import cells, pairs


def squares(centres, half=5.0):
    return cells.from_rings(
        [
            np.array(
                [
                    [x - half, y - half],
                    [x + half, y - half],
                    [x + half, y + half],
                    [x - half, y + half],
                ]
            )
            for x, y in centres
        ]
    )


GRID = [(x, y) for x in range(40, 400, 40) for y in range(40, 400, 40)]
ROWS = [(x, y) for x in range(20, 420, 20) for y in range(60, 420, 60)]
TOUCHING = [(x, y) for x in range(10, 310, 10) for y in range(10, 310, 10)]


def test_identical_cells_score_perfectly():
    ref = squares(GRID)
    rec = pairs.record(ref, ref, ref, ref, pixel_size_um=0.5)
    assert rec["pair_fraction"] == 1.0
    assert rec["dice_matched"] > 0.99
    assert rec["displacement_um_p50"] == 0.0
    assert rec["within_1um"] == 1.0
    assert rec["pair_fraction_mnn"] == 1.0
    assert rec["pair_fraction_excess"] == 1.0
    assert rec["displacement_readable"] is True
    assert rec["count_ratio"] == 1.0


def test_shift_shows_in_displacement_and_curve():
    ref = squares(GRID)
    # 3 px = 1.5 um at 0.5 um/px
    moved = pairs.with_xy(ref, ref.xy + [3.0, 0.0])
    rec = pairs.record(ref, ref, ref, moved, pixel_size_um=0.5)
    assert math.isclose(rec["displacement_um_p50"], 1.5, abs_tol=1e-6)
    assert rec["within_1um"] == 0.0
    assert rec["within_2um"] == 1.0


def test_both_frames_give_the_same_score_for_a_rigid_error():
    ref = squares(GRID)
    err = np.array([3.0, -2.0])
    in_ref_frame = pairs.record(ref, ref, ref, pairs.with_xy(ref, ref.xy + err), 0.5)
    in_mov_frame = pairs.record(ref, ref, pairs.with_xy(ref, ref.xy - err), ref, 0.5)
    for key in ("pair_fraction", "displacement_um_p50", "within_2um", "dice_matched"):
        assert math.isclose(in_ref_frame[key], in_mov_frame[key], abs_tol=1e-9)


def test_null_is_chance_level_on_a_sparse_grid():
    ref = squares(GRID)
    rec = pairs.record(ref, ref, ref, ref, pixel_size_um=0.5)
    # gate 1.5 * sqrt(100/pi) = 8.5 px; the null shift is 22.6 px on a 40 px grid
    assert rec["pair_fraction_null"] == 0.0
    assert rec["pair_fraction_null_max"] == 0.0
    assert rec["within_1um_null"] == 0.0


def test_one_null_direction_is_not_enough_on_rows_of_nuclei():
    ref = squares(ROWS)
    rec = pairs.record(ref, ref, ref, ref, pixel_size_um=0.5)
    assert rec["pair_fraction_null_max"] > 0.9  # along the rows
    assert rec["pair_fraction_null_min"] == 0.0  # across them
    assert rec["pair_fraction_null"] < 0.5
    # 1.0 does not clear the worst direction
    assert rec["displacement_readable"] is False


def test_piling_every_cell_on_one_place_is_not_rewarded():
    ref = squares(GRID)
    piled = pairs.with_xy(
        ref, ref.xy - np.repeat(np.array(GRID, dtype=float), 4, axis=0) + [41.0, 43.0]
    )
    rec = pairs.record(ref, ref, ref, piled, pixel_size_um=0.5)
    assert rec["within_5um_mov"] == 1.0  # one side alone is flattered
    assert rec["within_5um"] <= 1.0 / len(GRID) + 1e-9
    assert rec["pair_fraction"] <= 1.0 / len(GRID) + 1e-9


def test_snapping_onto_the_nearest_reference_cell_is_caught_by_the_two_sided_curve():
    """A folding map: from a FAILED pose, every moving nucleus is put exactly on its nearest
    reference nucleus. The pairs that exist look perfect; the reference nuclei nobody
    landed on are what the reference side of the curve still counts."""
    rng = np.random.default_rng(3)
    centres = np.array(
        [(x, y) for x in range(40, 640, 40) for y in range(40, 640, 40)], dtype=float
    )
    centres += rng.uniform(-10.0, 10.0, centres.shape)
    ref = squares(centres)
    failed_centres = centres + [57.0, -23.0]  # a pose that pairs wrong neighbours
    failed = squares(failed_centres)
    nearest = np.argmin(
        np.hypot(
            failed_centres[:, None, 0] - centres[None, :, 0],
            failed_centres[:, None, 1] - centres[None, :, 1],
        ),
        axis=1,
    )
    snapped = squares(centres[nearest])
    hit = len(set(nearest.tolist())) / len(centres)
    assert hit < 0.9  # several moving nuclei share one reference nucleus

    before = pairs.record(ref, failed, ref, failed, pixel_size_um=0.5)
    after = pairs.record(ref, failed, ref, snapped, pixel_size_um=0.5)
    # the paired numbers improve, and say nothing is wrong with the pairs they see
    assert after["pair_fraction"] > before["pair_fraction"]
    assert after["dice_matched"] > 0.95
    assert math.isclose(after["displacement_um_p50"], 0.0, abs_tol=1e-9)
    # the moving side of the curve is flattered, the reference side is not
    assert after["within_1um_mov"] == 1.0
    assert math.isclose(after["within_1um_ref"], hit, abs_tol=1e-9)
    assert after["within_1um"] == after["within_1um_ref"] < 0.9


def test_lost_cells_cap_the_curve_at_the_count_ratio():
    ref = squares(GRID)
    fewer = squares(GRID[: int(0.6 * len(GRID))])
    rec = pairs.record(ref, fewer, ref, fewer, pixel_size_um=0.5)
    assert rec["pair_fraction"] == 1.0
    assert math.isclose(rec["count_ratio"], len(fewer) / len(ref))
    assert math.isclose(rec["within_1um"], rec["count_ratio"])
    assert rec["within_1um_mov"] == 1.0


def test_touching_nuclei_make_a_perfect_registration_unreadable():
    """The null depends on density: where nuclei touch, a four-radius shift lands every
    nucleus on another one, so even a perfect registration cannot clear it."""
    ref = squares(TOUCHING)
    rec = pairs.record(ref, ref, ref, ref, pixel_size_um=0.5)
    assert rec["pair_fraction"] == 1.0
    assert rec["pair_fraction_null"] > 0.8
    assert rec["displacement_readable"] is False


def test_gate_comes_from_native_cells_not_warped_ones():
    ref = squares(GRID)
    a = pairs.record(ref, ref, ref, ref, 0.5)
    b = pairs.record(ref, ref, ref, pairs.with_xy(ref, ref.xy * 1.3), 0.5)
    assert a["match_radius_px"] == b["match_radius_px"]


def test_assignment_and_mutual_nn_disagree_where_expected():
    ref = squares([(0, 0), (19, 0)], half=2.0)
    mov = squares([(1, 0), (0, 18)], half=2.0)
    assert pairs.score(ref, mov, 20.0, None, "lsa")["n_pairs"] == 2
    assert pairs.score(ref, mov, 20.0, None, "mutual_nn")["n_pairs"] == 1


def test_holes_are_kept_when_cells_come_from_geojson():
    import cell_pairs as cp

    outer = [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]
    hole = [[5, 5], [15, 5], [15, 15], [5, 15], [5, 5]]

    def one(rings):
        return cp.from_feature_collection(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": rings},
                    }
                ],
            }
        )

    rec = pairs.score(one([outer, hole]), one([outer]), 30.0, None)
    assert 0.70 < rec["iou_p50"] < 0.80  # 300 / 400: the hole is not filled in


def test_cell_pairs_is_the_pipelines_file_and_nothing_else_is_put_on_the_path():
    """In a fresh interpreter: other benchmark modules put ``bin/utils`` on ``sys.path``
    themselves, so this process cannot say what ``pairs`` did."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(pairs.__file__).resolve().parents[2]
    probe = (
        "import sys\n"
        "from pathlib import Path\n"
        "before = list(sys.path)\n"
        "from benchmarks.regbench import pairs\n"
        "import cell_pairs\n"
        "utils = Path(pairs.__file__).resolve().parents[2] / 'bin' / 'utils'\n"
        "assert Path(cell_pairs.__file__).resolve() == utils / 'cell_pairs.py'\n"
        "assert pairs.cp is cell_pairs\n"
        "assert sys.path == before, 'pairs changed sys.path'\n"
        "assert not any(Path(p).resolve() == utils for p in sys.path if p)\n"
        "assert 'logger' not in sys.modules and 'retry' not in sys.modules\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], cwd=repo, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


def test_no_pixel_size_gives_nan_not_zero():
    ref = squares(GRID)
    rec = pairs.record(ref, ref, ref, ref, pixel_size_um=None)
    assert math.isnan(rec["displacement_um_p50"])
    assert math.isnan(rec["within_1um"])
    assert rec["displacement_px_p50"] == 0.0


def test_empty_side_does_not_crash():
    e = cells.empty_polys()
    rec = pairs.record(squares(GRID), e, squares(GRID), e, 0.5)
    assert rec["n_pairs"] == 0 and rec["pair_fraction"] == 0.0
    assert rec["displacement_readable"] is False


def test_record_keys_are_complete():
    g = squares(GRID)
    rec = pairs.record(g, g, g, g, 0.5)
    assert set(pairs.RECORD_KEYS) == set(rec)
    assert len(pairs.RECORD_KEYS) == len(set(pairs.RECORD_KEYS))


def test_a_pair_both_pairings_share_is_rasterised_once(monkeypatch):
    """The two pairings mostly pick the same pairs; a pair's IoU depends on its two
    cells alone, so it is computed once and the numbers do not move."""
    rng = np.random.default_rng(4)
    centres = rng.uniform(40, 900, size=(150, 2))
    ref = pairs.as_polyset(squares(centres, 6.0))
    mov = pairs.as_polyset(squares(centres + rng.normal(0, 2.0, centres.shape), 6.0))
    apart = {}
    for pairing, mid in (("lsa", ""), ("mutual_nn", "_mnn")):
        got = pairs.score(ref, mov, 14.0, 0.5, pairing)
        apart.update({f"{k}{mid}": v for k, v in got.items()})
    calls = []
    real = pairs.cp.pair_iou

    def counting(a, b, ia, ib, *args, **kw):
        calls.append(len(ia))
        return real(a, b, ia, ib, *args, **kw)

    monkeypatch.setattr(pairs.cp, "pair_iou", counting)
    together = pairs._both_pairings(ref, mov, 14.0, 0.5)
    assert together.keys() == apart.keys()
    for key, value in apart.items():
        same = together[key] == value or (
            value != value and together[key] != together[key]
        )
        assert same, key
    n_lsa, n_mnn = together["n_pairs"], together["n_pairs_mnn"]
    assert n_mnn > 100 and sum(calls) < n_lsa + 0.2 * n_mnn


# ── the pipeline's pairing says what it was, and the exact assignment beside it ──
def tissue_centroids(side, seed=0, pitch=26.0, noise=3.0, drop=0.05):
    """Two sets of centroids as dense tissue gives them: a jittered lattice, the
    second set the first moved by a little noise, each with some cells missing."""
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1).reshape(-1, 2)
    ref = g * pitch + rng.uniform(-0.35, 0.35, g.shape) * pitch
    mov = ref + rng.normal(0, noise, ref.shape)
    return ref[rng.random(len(ref)) > drop], mov[rng.random(len(mov)) > drop]


def as_set(ia, ib):
    return set(zip(ia.tolist(), ib.tolist()))


def test_without_a_fallback_the_exact_assignment_is_the_pipelines_pairing():
    """Every component at or below the pipeline's threshold is solved by the pipeline
    itself, so there the two pairings are one: the same pairs, the same numbers."""
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 4000, (1500, 2))
    b = a[rng.random(len(a)) > 0.1]
    b = b + rng.normal(0, 6, b.shape)
    ia, ib, dist, stats = pairs.cp.match_cells(a, b, 25.0, method="lsa")
    assert stats["n_fallback_cells"] == 0 and stats["largest_component_cells"] > 2
    ea, eb, edist, status = pairs.exact_pairs(a, b, 25.0, ia, ib, stats)
    assert status == pairs.EXACT_SAME
    assert np.array_equal(ea, ia) and np.array_equal(eb, ib)
    assert np.array_equal(edist, dist)
    # and in a record: the exact columns repeat the headline ones
    ref = squares(GRID)
    moved = pairs.with_xy(ref, ref.xy + [3.0, 1.0])
    rec = pairs.record(ref, ref, ref, moved, pixel_size_um=0.5)
    assert rec["exact_status"] == pairs.EXACT_SAME
    assert rec["n_fallback_cells"] == 0 and rec["fallback_frac"] == 0.0
    assert rec["largest_component_cells"] == 2 and rec["n_components"] == len(GRID)
    for key in ("n_pairs", "pair_fraction", "dice_matched", "displacement_um_p50"):
        assert rec[f"{key}_exact"] == rec[key]


def test_the_sparse_solver_is_the_pipelines_dense_assignment(monkeypatch):
    """Where the pipeline falls back, the exact assignment is what the pipeline's own
    dense solver gives when it is allowed every component: the same pairs. Shown with
    the threshold lowered to 3 cells, so that the dense solver can still be run."""
    a, b = tissue_centroids(24, seed=5)
    radius = 30.0
    dense = pairs.cp.match_lsa(a, b, radius, max_component_cells=10**9)
    ia, ib, _, stats = pairs.cp.match_lsa(a, b, radius, max_component_cells=3)
    assert stats["n_fallback_cells"] > 0 and len(ia) < len(dense[0])
    monkeypatch.setattr(pairs.cp, "DEFAULT_MAX_COMPONENT_CELLS", 3)
    ea, eb, edist, status = pairs.exact_pairs(a, b, radius, ia, ib, stats)
    assert status.startswith("solved exactly in the ")
    assert as_set(ea, eb) == as_set(dense[0], dense[1])
    assert np.allclose(edist, dense[2], rtol=0, atol=1e-12)
    # cells the map sent nowhere are in no pair, as in the pipeline's
    a[7], b[11] = np.nan, np.inf
    dense = pairs.cp.match_lsa(a, b, radius, max_component_cells=10**9)
    ia, ib, _, stats = pairs.cp.match_lsa(a, b, radius, max_component_cells=3)
    ea, eb, _, _ = pairs.exact_pairs(a, b, radius, ia, ib, stats)
    assert as_set(ea, eb) == as_set(dense[0], dense[1])
    assert 7 not in ea and 11 not in eb


def test_above_the_threshold_the_two_pairings_differ_and_the_record_says_so():
    """One component of more than 5,000 cells: the pipeline pairs it by mutual nearest
    neighbour, the exact assignment finds more pairs at a larger total distance."""
    a, b = tissue_centroids(60, seed=1)
    radius = 30.0
    ia, ib, dist, stats = pairs.cp.match_cells(a, b, radius, method="lsa")
    assert stats["largest_component_cells"] > pairs.cp.DEFAULT_MAX_COMPONENT_CELLS
    assert stats["n_fallback_cells"] == stats["largest_component_cells"]
    ea, eb, edist, status = pairs.exact_pairs(a, b, radius, ia, ib, stats)
    assert status == pairs.EXACT_SOLVED.format(n=1)
    assert len(ea) > len(ia) and edist.sum() > dist.sum()
    assert len(set(ea)) == len(ea) and len(set(eb)) == len(eb)  # one partner each
    assert (edist <= radius).all()
    # the optimum itself, from the pipeline's dense solver without its threshold
    dense = pairs.cp.match_lsa(a, b, radius, max_component_cells=10**9)
    assert len(ea) == len(dense[0])
    assert math.isclose(edist.sum(), dense[2].sum(), rel_tol=0, abs_tol=1e-6)
    assert as_set(ea, eb) == as_set(dense[0], dense[1])
    # through a record: the headline stays the pipeline's, the exact numbers beside it
    ref, mov = squares(a, half=6.0), squares(b, half=6.0)
    got = pairs.score(ref, mov, radius, 0.5, exact=True)
    assert got["n_pairs"] == len(ia) and got["n_pairs_exact"] == len(ea)
    assert got["n_fallback_cells"] == stats["n_fallback_cells"]
    assert got["fallback_frac"] == stats["n_fallback_cells"] / (len(a) + len(b))
    assert got["pair_fraction_exact"] > got["pair_fraction"]
    assert got["displacement_um_p50_exact"] > got["displacement_um_p50"]
    assert 0 < got["dice_matched_exact"] < got["dice_matched"]
    assert "n_fallback_cells" not in pairs.score(ref, mov, radius, 0.5, "mutual_nn")


def test_a_component_above_the_bound_is_blank_with_the_reason(monkeypatch):
    a, b = tissue_centroids(60, seed=1)
    ia, ib, _, stats = pairs.cp.match_cells(a, b, 30.0, method="lsa")
    work = pairs.exact_work(len(a), len(b))
    assert pairs.exact_pairs(a, b, 30.0, ia, ib, stats, max_work=work)[0] is not None
    got = pairs.exact_pairs(a, b, 30.0, ia, ib, stats, max_work=0.9 * work)
    assert got[:3] == (None, None, None) and got[3].startswith("not solved: a comp")
    assert "pairs.EXACT_MAX_WORK" in got[3]
    monkeypatch.setattr(pairs, "EXACT_MAX_WORK", 10.0)
    rec = pairs.score(squares(a[:200]), squares(b[:200]), 30.0, 0.5, exact=True)
    assert (
        rec["exact_status"] == pairs.EXACT_SAME
    )  # nothing fell back: nothing to solve
    monkeypatch.setattr(pairs.cp, "DEFAULT_MAX_COMPONENT_CELLS", 3)
    ref, mov = squares(a[:200]), squares(b[:200])
    cent = [pairs.area_centroid(s)[1] for s in (ref, mov)]
    ia, ib, _, stats = pairs.cp.match_lsa(*cent, 30.0, max_component_cells=3)
    assert stats["n_fallback_cells"] > 0
    assert pairs.exact_pairs(*cent, 30.0, ia, ib, stats)[0] is None


def test_the_exact_optimum_is_certified_by_a_count_that_knows_no_distance():
    """Two cells against two, where the nearest-first choice leaves a cell alone: the
    most pairs come first, then the least distance."""
    rows, cols = np.array([0, 0, 1]), np.array([0, 1, 0])
    dist = np.array([1.0, 9.0, 2.0])
    ri, ci, certified = pairs.solve_component(rows, cols, dist, 2, 2, 10.0)
    assert certified and as_set(ri, ci) == {(0, 1), (1, 0)}
    # distance 0 is a candidate like any other
    ri, ci, certified = pairs.solve_component(
        np.array([0, 1]), np.array([0, 0]), np.array([0.0, 3.0]), 2, 1, 10.0
    )
    assert certified and as_set(ri, ci) == {(0, 0)}
