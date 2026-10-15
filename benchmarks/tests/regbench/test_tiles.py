"""The tile-series format (``regbench.tiles``): a round trip and each refusal."""

import json

import numpy as np
import pytest
import tifffile

from benchmarks.regbench import tiles


def four_tiles(h=16, w=20):
    rng = np.random.default_rng(3)
    return [
        (t, rng.integers(0, 4000, (h, w), dtype=np.uint16), x, y)
        for t, (x, y) in enumerate([(0.0, 0.0), (18.0, 0.0), (0.0, 14.5), (18.0, 14.5)], 1)
    ]


def written(tmp_path, **kw):
    kw.setdefault("pixel_size_um", 0.65)
    kw.setdefault("channels", ("DAPI", "CD3"))
    kw.setdefault("nuclear_channel", 0)
    return tiles.write_cycle(tmp_path / "c", four_tiles(), **kw)


def test_a_cycle_round_trips(tmp_path):
    src = four_tiles()
    cyc = written(tmp_path, ffp=tmp_path / "ffp.tif")
    back = tiles.read_cycle(cyc.dir)
    assert back == cyc
    assert (back.pixel_size_um, back.channels, back.nuclear_channel) == (0.65, ("DAPI", "CD3"), 0)
    assert back.ffp == tmp_path / "ffp.tif" and back.dfp is None
    assert [(r.tile, r.x_px, r.y_px, r.h, r.w, r.channel) for r in back.tiles] == [
        (t, x, y, 16, 20, 0) for t, _, x, y in src
    ]
    for row, (_, plane, _, _) in zip(back.tiles, src):
        assert row.path.parent == cyc.dir
        np.testing.assert_array_equal(tifffile.imread(row.path), plane)


def test_tiles_csv_is_stares_mosaic_input(tmp_path):
    cyc = written(tmp_path)
    head, *rows = (cyc.dir / "tiles.csv").read_text().splitlines()
    assert head == "tile,path,channel,x_px,y_px"
    assert rows[1] == "2,t00002.tif,0,18.0,0.0"  # the path is relative to the CSV
    mosaic = pytest.importorskip("stare.stages.mosaic")
    ids, paths, *_ = mosaic._read_tiles(cyc.dir / "tiles.csv")
    assert ids == ["1", "2", "3", "4"] and all(p.is_file() for p in paths)


def test_a_generator_is_consumed_one_plane_at_a_time(tmp_path):
    seen = []

    def gen():
        for item in four_tiles():
            seen.append(len(list((tmp_path / "c").glob("t*.tif"))))
            yield item

    tiles.write_cycle(tmp_path / "c", gen(), pixel_size_um=1.0, channels=("A",), nuclear_channel=0)
    assert seen == [0, 1, 2, 3]  # each plane is on disk before the next is asked for


def test_write_refuses_duplicates_non_finite_and_bad_pixel_size(tmp_path):
    base = four_tiles()
    kw = dict(channels=("A",), nuclear_channel=0)
    with pytest.raises(ValueError, match="duplicate tile id 1"):
        tiles.write_cycle(tmp_path / "a", base + [base[0]], pixel_size_um=1.0, **kw)
    bad = [(9, base[0][1], float("nan"), 0.0)]
    with pytest.raises(ValueError, match="non-finite"):
        tiles.write_cycle(tmp_path / "b", bad, pixel_size_um=1.0, **kw)
    for px in (None, 0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="pixel_size_um"):
            tiles.write_cycle(tmp_path / "c", base, pixel_size_um=px, **kw)


def test_read_refuses_a_missing_pixel_size(tmp_path):
    cyc = written(tmp_path)
    doc = json.loads((cyc.dir / "cycle.json").read_text())
    del doc["pixel_size_um"]
    (cyc.dir / "cycle.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="no pixel_size_um"):
        tiles.read_cycle(cyc.dir)


def test_read_refuses_a_duplicate_tile_id(tmp_path):
    cyc = written(tmp_path)
    csv = cyc.dir / "tiles.csv"
    csv.write_text(csv.read_text().replace("\n2,t00002", "\n1,t00002"))
    with pytest.raises(ValueError, match="duplicate tile id 1"):
        tiles.read_cycle(cyc.dir)


def test_read_refuses_a_tile_of_another_shape(tmp_path):
    cyc = written(tmp_path)
    tifffile.imwrite(cyc.dir / "t00003.tif", np.zeros((16, 21), np.uint16))
    with pytest.raises(ValueError, match="16x21.*first tile is 16x20"):
        tiles.read_cycle(cyc.dir)


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "abc"])
def test_read_refuses_a_non_finite_position(tmp_path, bad):
    cyc = written(tmp_path)
    csv = cyc.dir / "tiles.csv"
    csv.write_text(csv.read_text().replace("18.0,0.0", f"{bad},0.0", 1))
    with pytest.raises(ValueError, match="non-finite|not a number"):
        tiles.read_cycle(cyc.dir)
