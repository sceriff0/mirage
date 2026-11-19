"""Tests for bin/mask_to_geojson.py: label mask -> cell GeoJSON (segment.py's
cell_mask.tif to warp_seg_qc's warp input)."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("scipy")

import mask_to_geojson
from utils import cell_pairs as cp


def _two_square_mask():
    m = np.zeros((20, 20), dtype=np.uint32)
    m[2:8, 2:8] = 1
    m[12:18, 12:18] = 2
    return m


def test_each_label_becomes_one_closed_polygon():
    fc = mask_to_geojson.mask_to_feature_collection(_two_square_mask())
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 2
    assert sorted(f["properties"]["label"] for f in fc["features"]) == [1, 2]
    for ft in fc["features"]:
        ring = ft["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]


def test_ignores_background_only_mask():
    fc = mask_to_geojson.mask_to_feature_collection(np.zeros((8, 8), dtype=np.uint32))
    assert fc["features"] == []


def _square_fc(*boxes):
    """FeatureCollection of axis-aligned squares given as pixel-edge ``(x0, y0, x1, y1)``."""
    feats = []
    for x0, y0, x1, y1 in boxes:
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        feats.append(
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def test_polygons_land_on_the_pixels_they_came_from():
    """Centroid and bounding box must be exact; area is allowed the marching-squares bias.

    The contour runs through pixel centres at the corners, so a 6x6 block encloses 30.5 px^2
    rather than 36, a consequence of tracing an iso-line. What must not drift is where the
    polygon is, which the QC's centroid residual and per-pair IoU are read off.
    """
    m = _two_square_mask()
    ps = cp.from_feature_collection(mask_to_geojson.mask_to_feature_collection(m))
    area, cent = cp.feature_area_centroid(ps)
    bbox = cp.feature_bboxes(ps)

    order = np.argsort(cent[:, 0])
    # Coordinates are pixel-centre based: m[2:8, 2:8] covers centres 2..7, so its outline
    # runs 1.5..7.5. The warp and centroid residual inherit this convention.
    assert cent[order].ravel() == pytest.approx([4.5, 4.5, 14.5, 14.5])
    assert bbox[order].ravel() == pytest.approx(
        [1.5, 1.5, 7.5, 7.5, 11.5, 11.5, 17.5, 17.5]
    )
    assert area == pytest.approx([30.5, 30.5])


def test_polygons_overlap_the_source_squares(tmp_path):
    """IoU against the exact squares the mask was drawn from, scored the way the QC scores."""
    m = _two_square_mask()
    got = cp.from_feature_collection(mask_to_geojson.mask_to_feature_collection(m))
    # Pixel-centre convention: m[2:8, 2:8] is the square spanning 1.5..7.5, not 2..8.
    truth = cp.from_feature_collection(
        _square_fc((1.5, 1.5, 7.5, 7.5), (11.5, 11.5, 17.5, 17.5))
    )

    _, c_got = cp.feature_area_centroid(got)
    _, c_truth = cp.feature_area_centroid(truth)
    i_truth, i_got, _ = cp.match_mutual_nn(c_truth, c_got, radius=2.0)
    assert i_truth.size == 2

    iou, scored = cp.pair_iou(truth, got, i_truth, i_got, supersample=8)
    assert scored.all()
    assert iou.min() > 0.8


def test_write_geojson_roundtrips_through_disk(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    mask_path = tmp_path / "P001_cell_mask.tif"
    tifffile.imwrite(str(mask_path), _two_square_mask())
    out = tmp_path / "P001_cells.geojson"
    n = mask_to_geojson.write_geojson(str(mask_path), str(out))
    assert n == 2
    assert len(json.loads(out.read_text())["features"]) == 2
