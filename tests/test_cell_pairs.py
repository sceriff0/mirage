"""Tests for bin/utils/cell_pairs.py — the fixed-correspondence cell-pair primitives behind
the staged registration QC (reg_qc=2).

Everything here is pure geometry: no VALIS, no container, no GPU. The properties that matter
are (a) flattening a GeoJSON preserves feature order and identity across a warp, (b) mutual-NN
matching is symmetric and respects the radius, and (c) per-pair IoU on a local window agrees
with the closed-form answer for shapes whose overlap can be computed by hand.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
    ),
)

pytest.importorskip("skimage")
pytest.importorskip("scipy")

from utils import cell_pairs as cp


# ── fixtures ───────────────────────────────────────────────────────────────────
def _square(x0, y0, s):
    ring = [[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s], [x0, y0]]
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _square_with_hole(x0, y0, s, hs):
    outer = [[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s], [x0, y0]]
    m = (s - hs) / 2.0
    hx, hy = x0 + m, y0 + m
    hole = [[hx, hy], [hx + hs, hy], [hx + hs, hy + hs], [hx, hy + hs], [hx, hy]]
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [outer, hole]},
    }


def _multipolygon(*squares):
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [[s["geometry"]["coordinates"][0]] for s in squares],
        },
    }


def _fc(*features):
    return {"type": "FeatureCollection", "features": list(features)}


# ── flattening ─────────────────────────────────────────────────────────────────
def test_from_feature_collection_preserves_feature_order_and_count():
    ps = cp.from_feature_collection(_fc(_square(0, 0, 10), _square(100, 100, 10)))
    assert ps.n_features == 2
    assert ps.n_rings == 2
    assert ps.ring_feat.tolist() == [0, 1]
    assert not ps.ring_hole.any()
    # feature 1's vertices are the second ring's slice, not feature 0's
    lo, hi = ps.ring_off[1], ps.ring_off[2]
    assert ps.xy[lo:hi][:, 0].min() == 100


def test_degenerate_rings_are_dropped_but_feature_index_survives():
    bad = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1]]]},
    }
    ps = cp.from_feature_collection(_fc(bad, _square(50, 50, 10)))
    assert ps.n_features == 2  # index of the good square is still 1
    assert ps.n_rings == 1
    assert ps.ring_feat.tolist() == [1]


def test_multipolygon_contributes_one_ring_per_part():
    ps = cp.from_feature_collection(
        _fc(_multipolygon(_square(0, 0, 4), _square(20, 0, 4)))
    )
    assert ps.n_features == 1
    assert ps.n_rings == 2
    assert ps.ring_feat.tolist() == [0, 0]


def test_with_xy_shares_ring_bookkeeping():
    ps = cp.from_feature_collection(_fc(_square(0, 0, 10), _square(30, 0, 10)))
    shifted = ps.with_xy(ps.xy + np.array([5.0, 7.0]))
    assert shifted.n_features == ps.n_features
    assert np.array_equal(shifted.ring_off, ps.ring_off)
    assert np.array_equal(shifted.ring_feat, ps.ring_feat)
    _, c0 = cp.feature_area_centroid(ps)
    _, c1 = cp.feature_area_centroid(shifted)
    assert np.allclose(c1 - c0, [5.0, 7.0])


def test_with_xy_rejects_a_shape_change():
    ps = cp.from_feature_collection(_fc(_square(0, 0, 10)))
    with pytest.raises(ValueError):
        ps.with_xy(ps.xy[:-1])


# ── area / centroid ────────────────────────────────────────────────────────────
def test_area_and_centroid_of_a_square():
    ps = cp.from_feature_collection(_fc(_square(10, 20, 6)))
    area, cent = cp.feature_area_centroid(ps)
    assert area[0] == pytest.approx(36.0)
    assert cent[0] == pytest.approx([13.0, 23.0])


def test_hole_subtracts_from_area_and_leaves_centroid_centred():
    ps = cp.from_feature_collection(_fc(_square_with_hole(0, 0, 10, 4)))
    area, cent = cp.feature_area_centroid(ps)
    assert area[0] == pytest.approx(100.0 - 16.0)
    assert cent[0] == pytest.approx([5.0, 5.0])


def test_area_is_orientation_independent():
    ring = [[0, 0], [0, 4], [4, 4], [4, 0], [0, 0]]  # clockwise
    ft = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }
    area, _ = cp.feature_area_centroid(cp.from_feature_collection(_fc(ft)))
    assert area[0] == pytest.approx(16.0)


def test_median_equivalent_radius_matches_a_disc():
    r = 7.0
    assert cp.median_equivalent_radius(
        np.array([math.pi * r * r] * 5)
    ) == pytest.approx(r)


def test_median_equivalent_radius_of_nothing_is_zero():
    assert cp.median_equivalent_radius(np.array([])) == 0.0


# ── matching ───────────────────────────────────────────────────────────────────
def test_mutual_nn_pairs_the_obvious_partners():
    a = np.array([[0.0, 0.0], [100.0, 0.0]])
    b = np.array([[1.0, 0.0], [101.0, 0.0]])
    ia, ib, d = cp.match_mutual_nn(a, b, radius=5.0)
    assert ia.tolist() == [0, 1]
    assert ib.tolist() == [0, 1]
    assert np.allclose(d, [1.0, 1.0])


def test_mutual_nn_respects_the_radius():
    a = np.array([[0.0, 0.0]])
    b = np.array([[10.0, 0.0]])
    assert cp.match_mutual_nn(a, b, radius=5.0)[0].size == 0
    assert cp.match_mutual_nn(a, b, radius=15.0)[0].size == 1


def test_mutual_nn_rejects_a_two_onto_one_collapse():
    # Both a-cells are nearest to b[0]; only the closer one is b[0]'s nearest, so only it pairs.
    a = np.array([[0.0, 0.0], [3.0, 0.0]])
    b = np.array([[1.0, 0.0]])
    ia, ib, _ = cp.match_mutual_nn(a, b, radius=10.0)
    assert ia.tolist() == [0]
    assert ib.tolist() == [0]


def test_mutual_nn_is_symmetric_in_its_arguments():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 200, size=(40, 2))
    b = a + rng.normal(0, 1.0, size=a.shape)
    ia, ib, _ = cp.match_mutual_nn(a, b, radius=6.0)
    jb, ja, _ = cp.match_mutual_nn(b, a, radius=6.0)
    assert set(zip(ia.tolist(), ib.tolist())) == set(zip(ja.tolist(), jb.tolist()))


def test_mutual_nn_handles_empty_and_nonpositive_radius():
    a = np.array([[0.0, 0.0]])
    assert cp.match_mutual_nn(a, np.empty((0, 2)), radius=5.0)[0].size == 0
    assert cp.match_mutual_nn(a, a, radius=0.0)[0].size == 0


def test_mutual_nn_skips_nan_centroids():
    a = np.array([[np.nan, np.nan], [0.0, 0.0]])
    b = np.array([[0.5, 0.0]])
    ia, ib, _ = cp.match_mutual_nn(a, b, radius=5.0)
    assert ia.tolist() == [1] and ib.tolist() == [0]


# ── optimal (LSA) matching ─────────────────────────────────────────────────────
def _dense_lsa(a, b, radius):
    """Reference solve: one dense big-M assignment over the whole graph, no components."""
    from scipy.optimize import linear_sum_assignment

    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    big_m = (a.shape[0] + b.shape[0]) * radius + 1.0
    cost = np.where(d <= radius, d, big_m)
    ri, ci = linear_sum_assignment(cost)
    keep = cost[ri, ci] < big_m
    return set(zip(ri[keep].tolist(), ci[keep].tolist()))


def test_lsa_recovers_the_pair_mutual_nn_drops():
    # a1's nearest is b0, but b0's nearest is a0, so mutual-NN drops a1 entirely.
    # The optimal assignment takes (a0,b0) and (a1,b1) instead: 1 + 4 beats M + 2.
    a = np.array([[0.0, 0.0], [3.0, 0.0]])
    b = np.array([[1.0, 0.0], [7.0, 0.0]])

    ia, ib, _ = cp.match_mutual_nn(a, b, radius=5.0)
    assert set(zip(ia.tolist(), ib.tolist())) == {(0, 0)}, "premise: mutual-NN drops a1"

    ja, jb, dist, stats = cp.match_lsa(a, b, radius=5.0)
    assert set(zip(ja.tolist(), jb.tolist())) == {(0, 0), (1, 1)}
    assert np.allclose(dist, [1.0, 4.0])
    assert stats["n_fallback_components"] == 0


def test_lsa_component_decomposition_equals_a_dense_solve():
    # The core correctness claim: per-component LSA is the global optimum.
    rng = np.random.default_rng(7)
    a = rng.uniform(0, 300, size=(60, 2))
    b = a + rng.normal(0, 2.0, size=a.shape)
    radius = 6.0
    ia, ib, _, _ = cp.match_lsa(a, b, radius=radius)
    assert set(zip(ia.tolist(), ib.tolist())) == _dense_lsa(a, b, radius)


def test_lsa_pairs_at_least_as_many_cells_as_mutual_nn():
    # NB: LSA's *total* distance is usually HIGHER than mutual-NN's, because it pairs
    # strictly more cells. Cardinality is the property that holds.
    for seed in range(5):
        rng = np.random.default_rng(seed)
        a = rng.uniform(0, 200, size=(80, 2))
        b = a + rng.normal(0, 2.5, size=a.shape)
        n_nn = cp.match_mutual_nn(a, b, radius=5.0)[0].size
        n_lsa = cp.match_lsa(a, b, radius=5.0)[0].size
        assert n_lsa >= n_nn, f"seed {seed}: LSA paired {n_lsa} < mutual-NN {n_nn}"


def test_lsa_is_one_to_one_on_both_sides():
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 100, size=(50, 2))
    b = rng.uniform(0, 100, size=(50, 2))
    ia, ib, _, _ = cp.match_lsa(a, b, radius=20.0)
    assert len(set(ia.tolist())) == ia.size
    assert len(set(ib.tolist())) == ib.size


def test_lsa_respects_the_radius():
    a = np.array([[0.0, 0.0]])
    b = np.array([[10.0, 0.0]])
    assert cp.match_lsa(a, b, radius=5.0)[0].size == 0
    assert cp.match_lsa(a, b, radius=15.0)[0].size == 1


def test_lsa_falls_back_to_mutual_nn_on_a_percolated_component():
    # A 6-cell chain: every consecutive pair is 1 apart, so one component spans all of it.
    a = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    b = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
    ia, ib, _, stats = cp.match_lsa(a, b, radius=1.5, max_component_cells=4)
    assert stats["n_fallback_components"] == 1
    assert stats["n_fallback_cells"] == 6
    assert stats["largest_component_cells"] == 6
    assert len(set(ia.tolist())) == ia.size and len(set(ib.tolist())) == ib.size


def test_lsa_handles_empty_nan_and_nonpositive_radius():
    a = np.array([[0.0, 0.0]])
    assert cp.match_lsa(a, np.empty((0, 2)), radius=5.0)[0].size == 0
    assert cp.match_lsa(a, a, radius=0.0)[0].size == 0
    nan_a = np.array([[np.nan, np.nan], [0.0, 0.0]])
    ia, ib, _, _ = cp.match_lsa(nan_a, np.array([[0.5, 0.0]]), radius=5.0)
    assert ia.tolist() == [1] and ib.tolist() == [0]


def test_lsa_is_deterministic():
    rng = np.random.default_rng(11)
    a = rng.uniform(0, 150, size=(70, 2))
    b = a + rng.normal(0, 2.0, size=a.shape)
    first = cp.match_lsa(a, b, radius=6.0)
    second = cp.match_lsa(a, b, radius=6.0)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


# ── per-pair IoU ───────────────────────────────────────────────────────────────
def test_pair_iou_of_identical_squares_is_one():
    ps = cp.from_feature_collection(_fc(_square(0, 0, 10)))
    iou, scored = cp.pair_iou(ps, ps, [0], [0], supersample=2)
    assert scored[0]
    assert iou[0] == pytest.approx(1.0)


def test_pair_iou_of_half_overlapping_squares():
    # 10x10 squares offset by 5 in x: intersection 50, union 150 -> IoU 1/3.
    a = cp.from_feature_collection(_fc(_square(0, 0, 10)))
    b = cp.from_feature_collection(_fc(_square(5, 0, 10)))
    iou, scored = cp.pair_iou(a, b, [0], [0], supersample=4)
    assert scored[0]
    assert iou[0] == pytest.approx(1.0 / 3.0, abs=0.02)


def test_pair_iou_of_disjoint_squares_is_zero():
    a = cp.from_feature_collection(_fc(_square(0, 0, 10)))
    b = cp.from_feature_collection(_fc(_square(40, 0, 10)))
    iou, scored = cp.pair_iou(a, b, [0], [0], supersample=2)
    assert scored[0]
    assert iou[0] == pytest.approx(0.0)


def test_pair_iou_supersampling_converges_on_the_exact_answer():
    # A 10x10 vs a 10x10 offset by 2.5 px — a half-pixel offset the 1x grid cannot represent.
    a = cp.from_feature_collection(_fc(_square(0, 0, 10)))
    b = cp.from_feature_collection(_fc(_square(2.5, 0, 10)))
    exact = 75.0 / 125.0
    coarse = cp.pair_iou(a, b, [0], [0], supersample=1)[0][0]
    fine = cp.pair_iou(a, b, [0], [0], supersample=8)[0][0]
    assert abs(fine - exact) <= abs(coarse - exact)
    assert fine == pytest.approx(exact, abs=0.01)


def test_pair_iou_accounts_for_holes():
    # Identical outer squares, but one has a hole: IoU = (100-16)/100.
    a = cp.from_feature_collection(_fc(_square(0, 0, 10)))
    b = cp.from_feature_collection(_fc(_square_with_hole(0, 0, 10, 4)))
    iou, scored = cp.pair_iou(a, b, [0], [0], supersample=4)
    assert scored[0]
    assert iou[0] == pytest.approx(84.0 / 100.0, abs=0.02)


def test_pair_iou_refuses_a_window_over_the_cap():
    a = cp.from_feature_collection(_fc(_square(0, 0, 10)))
    b = cp.from_feature_collection(_fc(_square(100000, 0, 10)))
    iou, scored = cp.pair_iou(a, b, [0], [0], supersample=2, max_window_px=1000)
    assert not scored[0]
    assert math.isnan(iou[0])


def test_pair_iou_of_no_pairs_is_empty():
    ps = cp.from_feature_collection(_fc(_square(0, 0, 10)))
    iou, scored = cp.pair_iou(ps, ps, [], [])
    assert iou.size == 0 and scored.size == 0


# ── summarisation ──────────────────────────────────────────────────────────────
def test_summarize_stage_reports_pairs_iou_and_displacement():
    iou = np.array([1.0, 0.5, 0.25])
    scored = np.array([True, True, True])
    dist = np.array([0.0, 1.0, 2.0])
    areas = np.array([100.0, 100.0, 100.0])
    rec = cp.summarize_stage(iou, scored, dist, areas, areas, iou_thresh=0.5)
    assert rec["n_pairs"] == 3
    assert rec["n_pairs_scored"] == 3
    assert rec["iou_mean"] == pytest.approx((1.0 + 0.5 + 0.25) / 3)
    assert rec["iou_p50"] == pytest.approx(0.5)
    assert rec["frac_iou_ge_0.5"] == pytest.approx(2 / 3)
    assert rec["displacement_px_p50"] == pytest.approx(1.0)
    assert rec["displacement_px_max"] == pytest.approx(2.0)


def test_summarize_stage_adds_micron_stats_only_with_a_pixel_size():
    iou = np.array([1.0])
    rec_px = cp.summarize_stage(
        iou, np.array([True]), np.array([2.0]), np.array([1.0]), np.array([1.0])
    )
    assert "displacement_um_p50" not in rec_px
    rec_um = cp.summarize_stage(
        iou,
        np.array([True]),
        np.array([2.0]),
        np.array([1.0]),
        np.array([1.0]),
        pixel_size_um=0.5,
    )
    assert rec_um["displacement_um_p50"] == pytest.approx(1.0)


def test_summarize_stage_dice_matched_matches_the_closed_form():
    # Two identical pairs at IoU 1 -> matched Dice is 1.
    rec = cp.summarize_stage(
        np.array([1.0, 1.0]),
        np.array([True, True]),
        np.array([0.0, 0.0]),
        np.array([10.0, 10.0]),
        np.array([10.0, 10.0]),
    )
    assert rec["dice_matched"] == pytest.approx(1.0)
    # One pair at IoU 1/3 with equal areas a: inter = a/2, dice = 2*(a/2)/(2a) = 0.5.
    rec2 = cp.summarize_stage(
        np.array([1.0 / 3.0]),
        np.array([True]),
        np.array([0.0]),
        np.array([10.0]),
        np.array([10.0]),
    )
    assert rec2["dice_matched"] == pytest.approx(0.5)


def test_summarize_stage_excludes_unscored_pairs_but_counts_them():
    rec = cp.summarize_stage(
        np.array([1.0, np.nan]),
        np.array([True, False]),
        np.array([0.0, 5.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
    )
    assert rec["n_pairs"] == 2
    assert rec["n_pairs_scored"] == 1
    assert rec["iou_n"] == 1


def test_summarize_stage_of_no_pairs_is_degenerate_not_a_crash():
    rec = cp.summarize_stage(
        np.empty(0), np.empty(0, dtype=bool), np.empty(0), np.empty(0), np.empty(0)
    )
    assert rec["n_pairs"] == 0
    assert rec["n_pairs_scored"] == 0
    assert rec["iou_n"] == 0


# ── the property the whole design rests on ─────────────────────────────────────
def test_a_fixed_pairing_tracks_the_same_cells_through_successive_warps():
    """Pair once on a coarse alignment, then re-score the *same* indices after refinement.

    This is the staged-QC contract in miniature: matching is done once, and the improvement
    is read off the same pairs — never from a re-derived correspondence.
    """
    ref = cp.from_feature_collection(_fc(*[_square(i * 40, 0, 10) for i in range(6)]))
    mov_native = cp.from_feature_collection(
        _fc(*[_square(i * 40 + 7, 0, 10) for i in range(6)])
    )

    _, c_ref = cp.feature_area_centroid(ref)
    _, c_mov = cp.feature_area_centroid(mov_native)
    ia, ib, d0 = cp.match_mutual_nn(c_ref, c_mov, radius=12.0)
    assert ia.tolist() == list(range(6))  # every cell paired at the anchor

    iou0, s0 = cp.pair_iou(ref, mov_native, ia, ib, supersample=4)

    # "Refinement": shift the moving set 7 px back, i.e. perfect alignment.
    mov_refined = mov_native.with_xy(mov_native.xy - np.array([7.0, 0.0]))
    _, c_ref2 = cp.feature_area_centroid(ref)
    _, c_mov2 = cp.feature_area_centroid(mov_refined)
    d1 = cp.centroid_distance(c_ref2, c_mov2, ia, ib)  # SAME ia/ib, no rematching
    iou1, s1 = cp.pair_iou(ref, mov_refined, ia, ib, supersample=4)

    assert np.allclose(d0, 7.0)
    assert np.allclose(d1, 0.0, atol=1e-9)
    assert iou0[s0].mean() < iou1[s1].mean()
    assert iou1[s1] == pytest.approx(1.0)


# ── pairing dispatcher ─────────────────────────────────────────────────────────
def test_match_cells_defaults_to_lsa():
    a = np.array([[0.0, 0.0], [3.0, 0.0]])
    b = np.array([[1.0, 0.0], [7.0, 0.0]])
    ia, ib, _, stats = cp.match_cells(a, b, radius=5.0)
    assert stats["method"] == "lsa_centroid"
    assert set(zip(ia.tolist(), ib.tolist())) == {(0, 0), (1, 1)}
    assert stats["n_components"] >= 1


def test_match_cells_can_select_mutual_nn():
    a = np.array([[0.0, 0.0], [3.0, 0.0]])
    b = np.array([[1.0, 0.0], [7.0, 0.0]])
    ia, ib, _, stats = cp.match_cells(a, b, radius=5.0, method="mutual_nn")
    assert stats == {"method": "mutual_nn_centroid"}
    assert set(zip(ia.tolist(), ib.tolist())) == {(0, 0)}


def test_match_cells_rejects_an_unknown_method():
    a = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError, match="unknown pairing method"):
        cp.match_cells(a, a, radius=5.0, method="hungarian")
