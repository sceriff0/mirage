"""Tests for bin/utils/tre_report.py — STARE's intrinsic TRE report builder.

Used by the one entry point (tiled_solve) so the `_tre.json` has one shape. Reports the coarse
(rigid) feature-fit TRE, a per-tile spatial heatmap of the rigid-stage misalignment, and — when
the caller measured it — the post-refinement residual that is STARE's VALIS-comparable
final-accuracy number.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
    ),
)

from tre_report import build_tre_report


def test_percentiles_and_spatial_heatmap_from_rigid_tres():
    recs = [
        {"ix": 0, "iy": 0, "cx": 8.0, "cy": 8.0, "tre_rigid": 4.0},
        {"ix": 1, "iy": 0, "cx": 24.0, "cy": 8.0, "tre_rigid": 0.0},
    ]
    r = build_tre_report(2.5, 30, recs, mesh_refined=True)
    assert r["coarse_tre_px"] == 2.5
    assert r["n_inliers"] == 30
    assert r["n_tiles"] == 2
    assert r["mesh_refined"] is True
    assert r["rigid_tre_px"]["max"] == 4.0
    assert r["rigid_tre_px"]["p50"] == 2.0  # median of [4, 0]
    assert r["tiles"] == recs  # the spatial heatmap is the per-tile records
    assert "residual_after_px" not in r  # no post-refinement measured


def test_post_refinement_residual_reported_when_every_tile_has_it():
    recs = [
        {"ix": 0, "iy": 0, "cx": 8.0, "cy": 8.0, "tre_rigid": 4.0, "tre_after": 0.5},
        {"ix": 1, "iy": 0, "cx": 24.0, "cy": 8.0, "tre_rigid": 3.0, "tre_after": 0.1},
    ]
    r = build_tre_report(1.0, 10, recs, mesh_refined=True)
    assert r["residual_after_px"]["max"] == 0.5
    assert r["residual_after_px"]["p50"] == 0.3  # median of [0.5, 0.1]


def test_empty_tiles_gives_null_percentiles_not_a_crash():
    r = build_tre_report(0.0, 0, [], mesh_refined=False)
    assert r["n_tiles"] == 0
    assert r["rigid_tre_px"]["p50"] is None
    assert "residual_after_px" not in r
