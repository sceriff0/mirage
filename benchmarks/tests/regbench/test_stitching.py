"""Nuclei per raw tile, the mosaic's deduplicated nuclei, and the stitching error.

The geometry tests need no image: nuclei are small squares ("dots") of one global scene,
cut into 2x2 tiles of 100 px at step 60 (overlap 40: a strip, and a 40x40 diagonal corner
that STARE's neighbour rule leaves out). ``tile_nuclei`` is measured on real tile files.

Memory (user ruling 2026-10-14): ``tile_nuclei`` is traced on a 2x2 and a 6x6 grid (9x the
tiles); the peak grows with the polygons, not by one tile's pixels.
"""

from __future__ import annotations

import gc
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import tifffile

from benchmarks.regbench import cells, frontends, stitching, tiles
from benchmarks.regbench.tiles import TileRow

from .test_frontends import scene

PS = 0.5  # um / px
TILE, STEP = 100, 60
POS = {0: (0.0, 0.0), 1: (60.0, 0.0), 2: (0.0, 60.0), 3: (60.0, 60.0)}


def square(cx, cy, s=3.0):
    return np.array([[cx - s, cy - s], [cx + s, cy - s], [cx + s, cy + s], [cx - s, cy + s]])


def dots(seed=3, n=60, extent=160, gap=14.0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        p = rng.uniform(8, extent - 8, 2)
        if all(np.hypot(*(p - q)) > gap for q in out):
            out.append(p)
        if len(out) < n and rng.random() < 0.0:
            break
    return np.array(out)


def rows(positions=POS, h=TILE, w=TILE):
    return [TileRow(t, Path(f"t{t}.tif"), 0, x, y, h, w) for t, (x, y) in positions.items()]


def cut(scene_dots, true_pos=POS, h=TILE, w=TILE):
    """Per tile: the dots wholly inside it, as polygons in TILE px at the TRUE position."""
    out = {}
    for t, (x, y) in true_pos.items():
        rings = [square(cx - x, cy - y) for cx, cy in scene_dots
                 if 3 <= cx - x <= w - 3 and 3 <= cy - y <= h - 3]
        out[t] = cells.from_rings(rings)
    return out


SCENE = dots()


def interior_overlap_count(a, b):
    """Dots whose square lies inside the pair's overlap, 1 px clear of its edge (by hand)."""
    (ax, ay), (bx, by) = POS[a], POS[b]
    x0, x1 = max(ax, bx), min(ax, bx) + TILE
    y0, y1 = max(ay, by), min(ay, by) + TILE
    return sum(x0 + 4 <= cx <= x1 - 4 and y0 + 4 <= cy <= y1 - 4 for cx, cy in SCENE)


# ── seam_error ───────────────────────────────────────────────────────────────
def test_exact_positions_have_no_seam_error_and_match_every_overlap_nucleus():
    tp = cut(SCENE)
    res = stitching.seam_error(tp, POS, rows(), PS)
    assert res["n_pairs"] == 4  # the 40x40 diagonal (0, 3) is not a strip
    assert sorted((e["a"], e["b"]) for e in res["per_edge"]) == [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert res["disp_um_p90"] < 0.1 and res["disp_um_p50"] < 0.1
    for e in res["per_edge"]:
        want = interior_overlap_count(e["a"], e["b"])
        assert want > 0
        assert e["n_matched"] == want == e["n_a"] == e["n_b"]
    assert res["n_matched"] == sum(e["n_matched"] for e in res["per_edge"]) == res["n_candidates"]


def test_a_tile_placed_3px_off_reports_3px_on_its_edges_only():
    tp = cut(SCENE)
    moved = dict(POS)
    moved[3] = (63.0, 60.0)  # tile 3's content is where POS says, but it is PLACED 3 px right
    res = stitching.seam_error(tp, moved, rows(moved), PS)
    by = {(e["a"], e["b"]): e for e in res["per_edge"]}
    for edge in ((1, 3), (2, 3)):
        assert by[edge]["n_matched"] > 0
        assert by[edge]["disp_um_p50"] == pytest.approx(3 * PS, abs=0.01)
    assert by[(1, 3)]["dx_um"] == pytest.approx(3 * PS, abs=0.01)  # b's copy sits +x of a's
    for edge in ((0, 1), (0, 2)):
        assert by[edge]["disp_um_p90"] < 0.1


def test_a_nucleus_the_overlap_edge_cuts_is_left_out_of_the_pairing():
    # a nucleus reaching tile 0's right edge (x = 100) is cut there
    tp = {t: cells.empty_polys() for t in POS}
    tp[0] = cells.from_rings([square(97.0, 50.0)])  # bbox reaches x = 100
    tp[1] = cells.from_rings([square(37.0, 50.0)])
    res = stitching.seam_error(tp, POS, rows(), PS)
    assert res["n_matched"] == 0 and res["disp_um_p50"] is None


def test_no_neighbours_is_zero_pairs_not_an_error():
    far = {0: (0.0, 0.0), 1: (500.0, 0.0)}
    tp = {t: cells.from_rings([square(50, 50)]) for t in far}
    res = stitching.seam_error(tp, far, rows(far), PS)
    assert res["n_pairs"] == 0 and res["n_matched"] == 0 and res["per_edge"] == []
    assert res["disp_um_p50"] is None and res["disp_um_p90"] is None


def test_the_neighbour_rule_is_stares():
    # STARE: overlap >= 32 px on both axes AND >= half the smaller tile on one axis
    for other, expect in (((70.0, 0.0), False),    # a 30 px strip: under the 32 px minimum
                          ((60.0, 0.0), True),     # 40 x 100
                          ((60.0, 60.0), False),   # a 40 x 40 corner square
                          ((50.0, 50.0), True)):   # 50 x 50: half the tile on one axis
        pos = {0: (0.0, 0.0), 1: other}
        tp = {t: cells.empty_polys() for t in pos}
        assert stitching.seam_error(tp, pos, rows(pos), PS)["n_pairs"] == int(expect), other


def test_seam_error_and_mosaic_nuclei_refuse_inconsistent_inputs():
    tp = cut(SCENE)
    with pytest.raises(ValueError, match="disagree"):
        stitching.seam_error(tp, {0: (0, 0)}, rows(), PS)
    with pytest.raises(ValueError, match="pixel_size_um"):
        stitching.seam_error(tp, POS, rows(), 0.0)
    with pytest.raises(ValueError, match="disagree"):
        stitching.mosaic_nuclei({0: tp[0]}, POS, rows())


# ── mosaic_nuclei ────────────────────────────────────────────────────────────
def test_every_nucleus_appears_once_in_the_mosaic():
    out = stitching.mosaic_nuclei(cut(SCENE), POS, rows())
    _, c = cells.area_centroid(out)
    assert len(out) == len(SCENE)
    got = c[np.lexsort(c.T)]
    want = SCENE[np.lexsort(SCENE.T)]
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_dedup_keeps_the_copy_nearest_its_own_tile_centre():
    # tile 0's centre is global (50, 50), tile 1's (110, 50); the two copies are 1 px apart
    tp = {t: cells.empty_polys() for t in POS}
    tp[0] = cells.from_rings([square(75.0, 30.0)])         # global (75, 30): 32.0 from (50, 50)
    tp[1] = cells.from_rings([square(76.0 - 60, 30.0)])    # global (76, 30): 38.3 from (110, 50)
    out = stitching.mosaic_nuclei(tp, POS, rows())
    _, c = cells.area_centroid(out)
    assert len(out) == 1 and c[0] == pytest.approx([75.0, 30.0])  # tile 0's copy
    tp[0] = cells.from_rings([square(85.0, 30.0)])         # 40.3 from tile 0's centre
    tp[1] = cells.from_rings([square(85.5 - 60, 30.0)])    # 33.6 from tile 1's
    out = stitching.mosaic_nuclei(tp, POS, rows())
    _, c = cells.area_centroid(out)
    assert len(out) == 1 and c[0] == pytest.approx([85.5, 30.0])  # tile 1's copy


def test_an_exact_tie_keeps_the_lower_tile_id_whatever_the_order():
    # the copies are 30 px from their own centre, tile 0's at y 50.5 and tile 1's at y 49.5
    tp = {t: cells.empty_polys() for t in POS}
    tp[0] = cells.from_rings([square(80.0, 50.5)])
    tp[1] = cells.from_rings([square(80.0 - 60, 49.5)])
    by_id = {r.tile: r for r in rows()}
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [1, 0, 3, 2]):
        out = stitching.mosaic_nuclei(tp, POS, [by_id[i] for i in order])
        _, c = cells.area_centroid(out)
        assert len(out) == 1 and c[0] == pytest.approx([80.0, 50.5])  # tile 0's


def test_a_nucleus_in_all_four_tiles_is_kept_once():
    tp = {t: cells.empty_polys() for t in POS}
    # global (90, 90) is in the 40x40 corner all four tiles see
    g = np.array([90.0, 90.0])
    for t, (x, y) in POS.items():
        tp[t] = cells.from_rings([square(g[0] - x, g[1] - y)])
    out = stitching.mosaic_nuclei(tp, POS, rows())
    _, c = cells.area_centroid(out)
    assert len(out) == 1 and c[0] == pytest.approx(g)


def test_mosaic_nuclei_sit_on_the_pixels_write_mosaic_pastes(tmp_path):
    # two tiles with a bright block at a known tile pixel; fractional positions
    d = tmp_path / "c"
    plane = np.zeros((64, 64), np.uint16)

    def planes():
        for t, (x, y) in enumerate([(0.0, 0.0), (39.6, 0.4)]):
            p = plane.copy()
            p[20:28, 30:38] = 9000 + t  # a block at tile px x 30..37, y 20..27
            yield t, p, x, y

    cyc = tiles.write_cycle(d, planes(), pixel_size_um=PS, channels=["DAPI"], nuclear_channel=0)
    pos = {0: (0.0, 0.0), 1: (39.6, 0.4)}
    tp = {t: cells.from_rings([square(34.0, 24.0, 4.0)]) for t in pos}  # the block's outline
    out = stitching.mosaic_nuclei(tp, pos, cyc.tiles)
    mosaic = frontends.write_mosaic(cyc, pos, tmp_path / "m.ome.tif")
    img = tifffile.imread(str(mosaic))
    o = frontends.placed_origins(cyc.tiles, pos)
    assert o.tolist() == [[0, 0], [40, 0]]  # rint(39.6), rint(0.4) -> (40, 0)
    _, c = cells.area_centroid(out)
    for cx, cy in c:  # every kept nucleus sits on the block it came from
        assert img[int(cy), int(cx)] >= 9000


def test_a_tile_placed_far_off_is_judged_by_where_its_nuclei_land():
    tp = cut(SCENE)
    assert len(stitching.mosaic_nuclei(tp, POS, rows())) == len(SCENE)
    for dx, dy in ((20.0, 0.0), (0.0, 20.0), (20.0, 20.0), (-20.0, 0.0), (35.0, -20.0)):
        moved = dict(POS)
        moved[3] = (60.0 + dx, 60.0 + dy)
        rs = rows(moved)
        out = stitching.mosaic_nuclei(tp, moved, rs)
        assert len(stitching.mosaic_nuclei(tp, moved, rs[::-1])) == len(out)
        o = frontends.placed_origins(rs, moved)
        _, c = cells.area_centroid(out)
        for cx, cy in c:  # every survivor lies on some placed tile
            assert any(o[k][0] <= cx < o[k][0] + TILE and o[k][1] <= cy < o[k][1] + TILE
                       for k in range(4))
        # no polygon is kept by two tiles at one spot: no two survivors share a centroid
        assert len({(round(x, 6), round(y, 6)) for x, y in c}) == len(out)


# ── tile_nuclei ──────────────────────────────────────────────────────────────
def small_cycle(d, grid=(2, 2), tile=64, with_dots=True):
    k = {"n": 0}

    def planes():
        for gy in range(grid[0]):
            for gx in range(grid[1]):
                p = np.zeros((tile, tile), np.uint16)
                if with_dots:
                    p[20:28, 20:28] = 5000
                yield k["n"], p, gx * 48.0, gy * 48.0
                k["n"] += 1

    return tiles.write_cycle(d, planes(), pixel_size_um=PS, channels=["DAPI"], nuclear_channel=0)


class Counting(stitching.Segmenter):
    def __init__(self, tag="fake:1"):
        self.calls = []
        super().__init__(self._run, tag)

    def _run(self, path):
        self.calls.append(path.name)
        img = tifffile.imread(str(path))
        ys, xs = np.nonzero(img)
        if not len(xs):
            return cells.empty_polys()
        return cells.from_rings([square(xs.mean() + 0.5, ys.mean() + 0.5, 4.0)])


def test_each_tile_is_segmented_once_and_cached(tmp_path):
    cyc = small_cycle(tmp_path / "c")
    seg, cache = Counting(), tmp_path / "_cache" / "tiles" / "ds" / "cyc1"
    first = stitching.tile_nuclei(cyc, seg, cache)
    assert sorted(first) == [0, 1, 2, 3] and len(seg.calls) == 4
    for t, p in first.items():
        _, c = cells.area_centroid(p)
        assert len(p) == 1 and c[0] == pytest.approx([24.0, 24.0])
    again = stitching.tile_nuclei(cyc, seg, cache)
    assert len(seg.calls) == 4  # all four from the cache
    for t in first:
        np.testing.assert_array_equal(first[t].xy, again[t].xy)
    # another segmenter, or a changed tile, is a miss
    other = Counting("fake:2")
    stitching.tile_nuclei(cyc, other, cache)
    assert len(other.calls) == 4
    tifffile.imwrite(cyc.tiles[2].path, np.zeros((64, 64), np.uint16))
    stitching.tile_nuclei(cyc, other, cache)
    assert other.calls[4:] == ["t00002.tif"]  # only the changed tile
    res = stitching.tile_nuclei(cyc, other, cache)
    assert len(res[2]) == 0


def test_a_missing_or_corrupt_cache_entry_is_recomputed(tmp_path):
    cyc = small_cycle(tmp_path / "c", grid=(1, 2))
    seg, cache = Counting(), tmp_path / "cache"
    stitching.tile_nuclei(cyc, seg, cache)
    (cache / "t00001.npz").write_bytes(b"not an npz")
    seg.calls.clear()
    out = stitching.tile_nuclei(cyc, seg, cache)
    assert seg.calls == ["t00001.tif"] and len(out[1]) == 1


def test_the_built_in_segmenter_finds_the_nuclei_of_a_tile(tmp_path):
    sc = scene(200, density=1 / 1500, sigma=5.0)
    d = tmp_path / "c"
    cyc = tiles.write_cycle(d, [(0, (sc * 20000).astype(np.uint16), 0.0, 0.0)],
                            pixel_size_um=PS, channels=["DAPI"], nuclear_channel=0)
    polys = stitching.builtin_segmenter(PS)(cyc.tiles[0].path)
    assert len(polys) >= 3
    area, c = cells.area_centroid(polys)
    assert (area > 5).all() and ((c >= 0) & (c <= 200)).all()
    blank = tiles.write_cycle(tmp_path / "b", [(0, np.zeros((64, 64), np.uint16), 0.0, 0.0)],
                              pixel_size_um=PS, channels=["DAPI"], nuclear_channel=0)
    assert len(stitching.builtin_segmenter(PS)(blank.tiles[0].path)) == 0


def test_stardist_is_the_route_when_the_model_is_set_and_runs_segment_py(tmp_path, monkeypatch):
    from benchmarks.regbench import nuclei

    cyc = small_cycle(tmp_path / "c", grid=(1, 1))
    assert stitching.default_segmenter(cyc, tmp_path / "w", env={}).tag.startswith("builtin:")
    env = {nuclei.ENV_MODEL: "2D_versatile_fluo"}
    seg = stitching.default_segmenter(cyc, tmp_path / "w", env=env)
    assert seg.tag == "stardist:2D_versatile_fluo"
    seen = {}

    def fake_command(slide, dest, e):
        seen["slide"] = slide
        return ["python", "segment.py"]

    def fake_run(argv, **kw):
        lab = np.zeros((64, 64), np.int32)
        lab[10:20, 10:20] = 1
        mask, _ = nuclei.expected_masks(seen["slide"], seen["dest"])
        tifffile.imwrite(mask, lab)
        class R:
            returncode, stdout, stderr = 0, "", ""
        return R()

    def command(slide, dest, e):
        seen["dest"] = dest
        return fake_command(slide, dest, e)

    monkeypatch.setattr(nuclei, "command", command)
    monkeypatch.setattr(stitching.subprocess, "run", fake_run)
    polys = seg(cyc.tiles[0].path)
    assert len(polys) == 1 and seen["slide"]["channel"] == 0


# ── memory ───────────────────────────────────────────────────────────────────
def test_tile_nuclei_memory_does_not_grow_with_the_tile_count(tmp_path):
    tile = 512  # 512 KiB per uint16 tile
    peaks = {}
    for g in (2, 6):
        cyc = small_cycle(tmp_path / f"g{g}", grid=(g, g), tile=tile)
        seg = Counting()
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        out = stitching.tile_nuclei(cyc, seg)
        peaks[g] = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        assert len(out) == g * g
    one_tile = tile * tile * 2
    assert peaks[6] - peaks[2] < one_tile, (peaks, one_tile)
    assert peaks[2] > 0
