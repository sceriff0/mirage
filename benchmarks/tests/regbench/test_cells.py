"""The cell scorer: geometry, the one-to-one pairing, per-pair IoU, Dice, and tile-wise segmentation."""

import json

import numpy as np
import pytest
from regbench import cells
from regbench.imageio import write_nuclear


def square(cx, cy, half):
    return np.array([[cx - half, cy - half], [cx + half, cy - half],
                     [cx + half, cy + half], [cx - half, cy + half]], dtype=float)


def test_area_and_centroid_survive_slide_scale_coordinates():
    ps = cells.from_rings([square(10, 20, 3), square(52_000.25, 31_000.75, 5)])
    area, cent = cells.area_centroid(ps)
    assert area == pytest.approx([36.0, 100.0])
    assert cent == pytest.approx(np.array([[10, 20], [52_000.25, 31_000.75]]))


def test_match_is_one_to_one_and_minimises_total_distance():
    # Both a-cells are nearest to b0; the optimum gives a0 the far cell so both are paired.
    a = np.array([[0.0, 0.0], [2.0, 0.0]])
    b = np.array([[1.2, 0.0], [-1.5, 0.0]])
    ia, ib, dist = cells.match_lsa(a, b, radius=3.0)
    assert dict(zip(ia, ib)) == {0: 1, 1: 0}
    assert dist == pytest.approx([1.5, 0.8])


def test_match_radius_is_a_hard_gate_and_nan_cells_are_never_paired():
    a = np.array([[0.0, 0.0], [100.0, 0.0], [np.nan, 5.0]])
    b = np.array([[0.5, 0.0], [110.0, 0.0]])
    ia, ib, _ = cells.match_lsa(a, b, radius=3.0)
    assert list(ia) == [0] and list(ib) == [0]


def test_pair_iou_of_a_half_shifted_square_is_one_third():
    a = cells.from_rings([square(50, 50, 10)])
    same = cells.pair_iou(a, a, [0], [0])
    shifted = cells.pair_iou(a, cells.from_rings([square(60, 50, 10)]), [0], [0])
    assert same[0] == pytest.approx(1.0)
    assert shifted[0] == pytest.approx(1 / 3, abs=0.02)


def test_score_cells_reads_a_known_shift_as_displacement_and_dice():
    rng = np.random.default_rng(0)
    centres = rng.uniform(100, 2000, size=(200, 2))
    centres = centres[np.argsort(centres[:, 0])]
    ref = cells.from_rings([square(x, y, 6) for x, y in centres])
    rec0 = cells.score_cells(ref, ref, pixel_size_um=0.5)
    assert rec0["dice_matched"] == pytest.approx(1.0)
    assert rec0["displacement_px_p50"] == pytest.approx(0.0, abs=1e-9)
    assert rec0["pair_fraction"] == 1.0

    moved = ref.with_xy(ref.xy + [3.0, 0.0])  # 12 px squares shifted 3 px: Dice 9/12
    rec = cells.score_cells(ref, moved, pixel_size_um=0.5)
    assert rec["displacement_px_p50"] == pytest.approx(3.0)
    assert rec["displacement_um_p50"] == pytest.approx(1.5)
    assert rec["dice_matched"] == pytest.approx(0.75, abs=0.02)


def _blobs(n=700, seed=0):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    im = np.zeros((n, n), np.float32)
    ys, xs = rng.integers(10, n - 10, 90), rng.integers(10, n - 10, 90)
    im[ys, xs] = 1.0
    return (gaussian_filter(im, 4.0) * 60000 * 2 * np.pi * 16).clip(0, 65535).astype(np.uint16)


def test_tilewise_segmentation_equals_whole_image_segmentation(tmp_path):
    """Every nucleus belongs to exactly one tile: no duplicates and none lost at the seams."""
    path = write_nuclear(tmp_path / "s.ome.tif", _blobs())
    whole = cells.segment_slide(path, diameter=10, tile=4096)
    tiled = cells.segment_slide(path, diameter=10, tile=200)
    assert len(whole) > 40
    assert len(tiled) == len(whole)
    _, cw = cells.area_centroid(whole)
    _, ct = cells.area_centroid(tiled)
    ia, _, dist = cells.match_lsa(cw, ct, radius=1.0)
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

    feats = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [
        [*from_labels.xy[from_labels.off[k] : from_labels.off[k + 1]].tolist(),
         from_labels.xy[from_labels.off[k]].tolist()]]}} for k in range(2)]
    gj = tmp_path / "cells.geojson"
    gj.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    from_json = cells.load_cells(gj)
    assert cells.area_centroid(from_json)[0] == pytest.approx(area)
