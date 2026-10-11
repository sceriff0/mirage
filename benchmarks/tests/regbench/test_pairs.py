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
