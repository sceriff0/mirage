"""Making the cells: geometry, tile-wise segmentation, label images and GeoJSON.

Pairing and scores are ``test_pairs.py``."""

import json

import numpy as np
import pytest

from benchmarks.regbench import cells, pairs
from benchmarks.regbench.imageio import write_nuclear


def square(cx, cy, half):
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=float,
    )


def test_area_and_centroid_survive_slide_scale_coordinates():
    ps = cells.from_rings([square(10, 20, 3), square(52_000.25, 31_000.75, 5)])
    area, cent = cells.area_centroid(ps)
    assert area == pytest.approx([36.0, 100.0])
    assert cent == pytest.approx(np.array([[10, 20], [52_000.25, 31_000.75]]))


def _blobs(n=700, seed=0):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    im = np.zeros((n, n), np.float32)
    ys, xs = rng.integers(10, n - 10, 90), rng.integers(10, n - 10, 90)
    im[ys, xs] = 1.0
    return (
        (gaussian_filter(im, 4.0) * 60000 * 2 * np.pi * 16)
        .clip(0, 65535)
        .astype(np.uint16)
    )


def test_tilewise_segmentation_equals_whole_image_segmentation(tmp_path):
    """Every nucleus belongs to exactly one tile: no duplicates and none lost at the seams."""
    path = write_nuclear(tmp_path / "s.ome.tif", _blobs())
    whole = cells.segment_slide(path, diameter=10, tile=4096)
    tiled = cells.segment_slide(path, diameter=10, tile=200)
    assert len(whole) > 40
    assert len(tiled) == len(whole)
    _, cw = cells.area_centroid(whole)
    _, ct = cells.area_centroid(tiled)
    ia, _, dist, _ = pairs.cp.match_cells(cw, ct, 1.0)
    assert len(ia) == len(whole) and dist.max() < 1e-6


def test_label_image_and_geojson_give_the_same_cells(tmp_path):
    import tifffile

    labels = np.zeros((120, 160), np.int32)
    labels[20:40, 30:60] = 1
    labels[70:100, 90:110] = 2
    lab = tmp_path / "labels.tif"
    tifffile.imwrite(lab, labels)
    from_labels = cells.load_cells(lab, diameter=20, tile=64)
    area, cent = cells.area_centroid(from_labels)
    order = np.argsort(cent[:, 0])
    assert cent[order] == pytest.approx(np.array([[44.5, 29.5], [99.5, 84.5]]))
    # a contour traced at the half level is the pixel area minus the corner cuts
    assert area[order] == pytest.approx([600, 600], rel=0.02)

    feats = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        *from_labels.xy[
                            from_labels.off[k] : from_labels.off[k + 1]
                        ].tolist(),
                        from_labels.xy[from_labels.off[k]].tolist(),
                    ]
                ],
            },
        }
        for k in range(2)
    ]
    gj = tmp_path / "cells.geojson"
    gj.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    from_json = cells.load_cells(gj)
    assert isinstance(from_json, pairs.cp.PolySet)  # the pipeline's reader
    assert pairs.cp.feature_area_centroid(from_json)[0] == pytest.approx(area)


def test_geojson_cells_are_checked_against_their_slide(tmp_path):
    def write(name, features):
        path = tmp_path / name
        path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
        return path

    def cell(ring):
        geometry = {"type": "Polygon", "coordinates": [ring]}
        return {"type": "Feature", "properties": {}, "geometry": geometry}

    inside = write("in.geojson", [cell([[-0.5, -0.5], [9, 0], [9.5, 19.5], [0, 9]])])
    assert cells.cells_from_geojson(inside, hw=(20, 10)).n_features == 1
    assert cells.cells_from_geojson(inside).n_features == 1  # no slide: no check
    # [-1, W] x [-1, H]: one pixel of slack on every side, no more
    for hw in ((20, 9), (19, 10)):
        with pytest.raises(ValueError, match="in.geojson.*outside"):
            cells.cells_from_geojson(inside, hw=hw)
    low = write("low.geojson", [cell([[-1.5, 0], [5, 0], [5, 5]])])
    with pytest.raises(ValueError, match="low.geojson.*outside"):
        cells.cells_from_geojson(low, hw=(20, 10))

    with pytest.raises(ValueError, match="none.geojson.*no cell"):
        cells.cells_from_geojson(write("none.geojson", []))
    # features without a usable polygon are no cells either
    degenerate = write("thin.geojson", [cell([[0, 0], [1, 1]])])
    with pytest.raises(ValueError, match="thin.geojson.*no cell"):
        cells.cells_from_geojson(degenerate)
    broken = tmp_path / "broken.geojson"
    broken.write_text("{")
    with pytest.raises(ValueError, match="broken.geojson"):
        cells.cells_from_geojson(broken)
    with pytest.raises(ValueError, match="absent.geojson"):
        cells.cells_from_geojson(tmp_path / "absent.geojson")
