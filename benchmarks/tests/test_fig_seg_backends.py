"""Supplementary Fig. S7 (benchmarks/figures/fig_seg_backends.py) on a tiny slide and
three tiny GeoJSONs whose whole-cell masks overlap by amounts known in advance."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("skimage")
pytest.importorskip("stare")

from benchmarks.figures import fig_seg_backends as F  # noqa: E402
from benchmarks.regbench.imageio import write_nuclear  # noqa: E402

PX_UM = 0.5  # a 40 µm region is 80 px


def square(x, y, s):
    return [[[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]]]


def cells(shift):
    """Four 10 px cells (each a 4 px nucleus), all shifted right by ``shift`` px."""
    feats = []
    for i, (x, y) in enumerate([(10, 10), (40, 10), (10, 40), (40, 40)]):
        feats.append({"type": "Feature", "id": f"c{i}",
                      "geometry": {"type": "Polygon", "coordinates": square(x + shift, y, 10)},
                      "nucleusGeometry": {"type": "Polygon",
                                          "coordinates": square(x + shift + 3, y + 3, 4)},
                      "properties": {}})  # fmt: skip
    return {"type": "FeatureCollection", "features": feats}


@pytest.fixture
def slide(tmp_path):
    img = np.zeros((80, 80), dtype=np.uint16)
    for x, y in [(10, 10), (40, 10), (10, 40), (40, 40)]:
        img[y + 3 : y + 7, x + 3 : x + 7] = 4000
    path = tmp_path / "slide.ome.tif"
    write_nuclear(path, img, pixel_size_um=PX_UM)
    gj = {}
    for name, shift in (("CellSAM", 0), ("InstanSeg", 0), ("StarDist", 5)):
        gj[name] = tmp_path / f"{name}.geojson"
        gj[name].write_text(json.dumps(cells(shift)))
    return path, gj


def test_dice_of_filled_masks():
    a = F.fill_mask([np.array(square(0, 0, 10)[0], float)], 0, 0, 20)
    b = F.fill_mask([np.array(square(5, 0, 10)[0], float)], 0, 0, 20)
    assert F.dice(a, a) == 1.0
    # half of each square overlaps the other
    assert F.dice(a, b) == pytest.approx(0.5, abs=0.1)
    assert np.isnan(F.dice(np.zeros((2, 2), bool), np.zeros((2, 2), bool)))


def test_backends_need_a_name_and_at_least_two():
    with pytest.raises(SystemExit):
        F.parse_backends(["CellSAM"])
    with pytest.raises(SystemExit):
        F.parse_backends(["CellSAM=a.geojson"])
    assert list(F.parse_backends(["B=b", "A=a"])) == ["B", "A"]


def test_each_panel_is_its_own_file_with_its_source_data(slide, tmp_path):
    image, gj = slide
    out = tmp_path / "figs"
    F.main(["--image", str(image), "--region-um", "40", "--upsample", "1",
            "--outdir", str(out), *[f"--backend={n}={p}" for n, p in gj.items()]])  # fmt: skip
    for p in "ab":
        for ext in ("pdf", "png", "svg"):
            assert (out / f"seg_backends_{p}.{ext}").stat().st_size > 0, (p, ext)
    a = pd.read_csv(out / "seg_backends_a_source_data.csv")
    assert list(a["backend"]) == ["CellSAM", "InstanSeg", "StarDist"]
    assert set(a["cells_drawn"]) == {4} and set(a["nuclei_drawn"]) == {4}
    assert set(a["region_side_px"]) == {80}
    b = pd.read_csv(out / "seg_backends_b_source_data.csv")
    assert len(b) == 3  # every pair once
    d = {(r.backend_a, r.backend_b): r.dice for r in b.itertuples()}
    assert d[("CellSAM", "InstanSeg")] == pytest.approx(1.0)
    # StarDist's cells are shifted by half their width: about half overlap
    assert 0.35 < d[("CellSAM", "StarDist")] < 0.65
    assert d[("InstanSeg", "StarDist")] == pytest.approx(d[("CellSAM", "StarDist")])
