"""The three stitching front-ends over one tile series (``frontends.py``).

A 3x3 grid is cut from one synthetic nuclear scene at stage positions plus a KNOWN jitter
(Part A's ``_cut`` idea, ``stare-prep-mosaic/tests/test_mosaic.py``), each tile with its own
noise so no two overlaps are pixel-identical. ``stage`` must return the stage positions,
``stare_mosaic`` must recover the jitter, ``ashlar_edge`` runs against ``fake_ashlar``'s
``EdgeAligner`` (which returns the positions it is given), and all three must write the SAME
mosaic when given the same positions: they share one writer.

Memory (user ruling 2026-10-14): the writer, ``stage`` and ``ashlar_edge`` are measured on a
2x2 and a 6x6 grid (9x the tiles); the traced peak may grow by less than one tile.
"""

from __future__ import annotations

import gc
import json
import tracemalloc

import numpy as np
import pytest
import tifffile
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndshift

from benchmarks.regbench import frontends, tiles
from benchmarks.regbench.methods import ashlar as ashlar_method

from . import fake_ashlar

GAIN = 20000.0


def scene(n, seed=1, density=1 / 900, sigma=5.0):
    """Gaussian nuclei on black, peak ~1 (``stare_synthetic.nuclei``'s recipe)."""
    rng = np.random.default_rng(seed)
    k = int(n * n * density)
    im = np.zeros((n, n), np.float32)
    np.add.at(im, (rng.integers(0, n, k), rng.integers(0, n, k)), rng.uniform(0.5, 1.5, k))
    im = gaussian_filter(im, sigma)
    return im / float(im.max())


def cut_cycle(d, grid=(3, 3), tile=256, overlap=0.15, jitter_px=4.0, origin=(0.0, 0.0),
              seed=2, noise=True, pixel_size_um=0.5, ffp=None, dfp=None, raw=None):
    """A tile-series cycle in ``d``: the stage positions are the nominal grid + ``origin``,
    the pixels come from the nominal grid + a Gaussian jitter (tile 1 unjittered).

    Returns ``(cycle, truth)``: ``truth[tile] = (x, y)`` where the tile's pixels really are,
    in the stage frame. ``raw(plane) -> plane`` may distort each tile before it is written
    (the illumination test darkens it with known profiles).
    """
    rng = np.random.default_rng(seed)
    step = int(tile * (1 - overlap))
    n = step * (max(grid) - 1) + tile + 64
    sc = scene(n)
    truth = {}

    def planes():
        k = 0
        for iy in range(grid[0]):
            for ix in range(grid[1]):
                k += 1
                tx, ty = ix * step, iy * step
                jx, jy = rng.normal(0, jitter_px, 2) if k > 1 else (0.0, 0.0)
                img = ndshift(sc, (-ty - jy, -tx - jx), order=1)[:tile, :tile]
                if noise:
                    img = img + rng.normal(0, 0.01 * float(sc.std()), img.shape)  # Part A's level
                plane = np.clip(np.rint(img * GAIN), 0, 65535).astype(np.uint16)
                if raw is not None:
                    plane = raw(plane)
                truth[k] = (tx + jx + origin[0], ty + jy + origin[1])
                yield k, plane, tx + origin[0], ty + origin[1]

    cyc = tiles.write_cycle(d, planes(), pixel_size_um=pixel_size_um, channels=["DAPI"],
                            nuclear_channel=0, ffp=ffp, dfp=dfp)
    return cyc, truth


def noise_cycle(d, grid, tile=256, overlap=0.1, seed=0):
    """A grid of random tiles (no scene): for the memory tests, where nothing registers."""
    step = int(tile * (1 - overlap))

    def planes():
        rng = np.random.default_rng(seed)
        for i in range(grid * grid):
            iy, ix = divmod(i, grid)
            yield i + 1, rng.integers(0, 999, (tile, tile)).astype(np.uint16), ix * step, iy * step

    return tiles.write_cycle(d, planes(), pixel_size_um=1.0, channels=["DAPI"], nuclear_channel=0)


def install_fake(monkeypatch, **config):
    fake_ashlar.install(monkeypatch.setitem, **config)


def given_positions(pos_rc):
    """``CONFIG["edge_positions"]``: the fake's aligner returns these (row, col) px."""
    return lambda metadata: np.asarray(pos_rc, dtype=float)


# ── the vocabulary ────────────────────────────────────────────────────────────
def test_front_ends_are_the_three():
    assert frontends.FRONT_ENDS == ("stage", "ashlar_edge", "stare_mosaic")


def test_an_unknown_front_end_is_refused(tmp_path):
    cyc, _ = cut_cycle(tmp_path / "c", grid=(1, 2), tile=64)
    with pytest.raises(ValueError, match="front-end"):
        frontends.stitch(cyc, "ashlar", tmp_path / "o")


# ── stage ─────────────────────────────────────────────────────────────────────
def test_stage_positions_are_the_stage_positions_at_a_zero_origin(tmp_path):
    cyc, _ = cut_cycle(tmp_path / "c", origin=(-1000.0, 250.0))
    st = frontends.stitch(cyc, "stage", tmp_path / "o")
    want = {r.tile: (r.x_px + 1000.0, r.y_px - 250.0) for r in cyc.tiles}
    assert st.positions == want
    assert min(x for x, _ in st.positions.values()) == 0.0
    assert min(y for _, y in st.positions.values()) == 0.0
    assert st.report["front_end"] == "stage"
    with tifffile.TiffFile(st.mosaic) as tf:
        assert tf.is_bigtiff and tf.pages[0].is_tiled
        assert tf.series[0].shape[-2:] == (2 * 217 + 256, 2 * 217 + 256)


def test_stage_mosaic_pastes_last_wins(tmp_path):
    """Two 64 px tiles, the second 32 px right of the first: the overlap is the second's."""
    a = np.full((64, 64), 100, np.uint16)
    b = np.full((64, 64), 200, np.uint16)
    cyc = tiles.write_cycle(tmp_path / "c", iter([(1, a, 0, 0), (2, b, 32, 0)]),
                            pixel_size_um=1.0, channels=["DAPI"], nuclear_channel=0)
    st = frontends.stitch(cyc, "stage", tmp_path / "o")
    m = tifffile.imread(st.mosaic).squeeze()
    assert m.shape == (64, 96)
    assert (m[:, :32] == 100).all() and (m[:, 32:] == 200).all()


# ── the shared writer ────────────────────────────────────────────────────────
def test_writer_blocks_tile_the_mosaic_and_match_a_whole_canvas_paste(tmp_path):
    cyc, truth = cut_cycle(tmp_path / "c", grid=(3, 3), tile=96, overlap=0.2)
    pos = {t: (x, y) for t, (x, y) in truth.items()}  # sub-pixel, some negative
    out = frontends.write_mosaic(cyc, pos, tmp_path / "m.ome.tif", out_tile=64)
    m = tifffile.imread(out).squeeze()
    # the reference: a whole canvas, positions rounded and moved to a zero origin
    xy = np.array([pos[r.tile] for r in cyc.tiles])
    o = np.rint(xy).astype(int)
    o -= o.min(axis=0)
    ref = np.zeros((int((o[:, 1] + 96).max()), int((o[:, 0] + 96).max())), np.uint16)
    for (x, y), r in zip(o, cyc.tiles):
        ref[y:y + 96, x:x + 96] = tifffile.imread(r.path)
    assert m.shape == ref.shape
    np.testing.assert_array_equal(m, ref)


def test_writer_refuses_an_out_tile_tiff_cannot_hold(tmp_path):
    cyc, _ = cut_cycle(tmp_path / "c", grid=(1, 2), tile=64)
    with pytest.raises(ValueError, match="multiple of 16"):
        frontends.write_mosaic(cyc, frontends.stage_positions(cyc), tmp_path / "m.tif",
                               out_tile=100)


def test_writer_refuses_positions_for_tiles_the_cycle_lacks(tmp_path):
    cyc, _ = cut_cycle(tmp_path / "c", grid=(1, 2), tile=64)
    with pytest.raises(ValueError, match="tile"):
        frontends.write_mosaic(cyc, {1: (0.0, 0.0)}, tmp_path / "m.tif")


# ── illumination: ASHLAR's own arithmetic, applied per window ───────────────
def ashlar_correct(raw, dfp, ffp):
    """``reg.Mosaic.correct_illumination`` + ``utils.paste``'s conversion (ashlar 1.20.0):
    ``img_as_float``, ``-= dfp / iinfo.max``, ``/= ffp``, clip [0, 1], then
    ``skimage.util.dtype.convert``: ``rint(f * max)``."""
    imax = np.iinfo(raw.dtype).max
    f = np.multiply(raw, 1.0 / imax, dtype=np.float64)
    f -= dfp / imax
    f /= ffp
    np.clip(f, 0, 1, out=f)
    return np.clip(np.rint(f * imax), 0, imax).astype(raw.dtype)


def profiles(d, tile, n_channels=1):
    yy, xx = np.mgrid[0:tile, 0:tile] / tile
    ffp = (0.6 + 0.8 * (1 - (xx - 0.5) ** 2 - (yy - 0.5) ** 2)).astype(np.float32)
    ffp /= ffp.mean()
    dfp = (50 + 30 * xx).astype(np.float32)
    d.mkdir(parents=True, exist_ok=True)
    if n_channels > 1:  # an MCMICRO-style per-cycle profile holds every channel
        tifffile.imwrite(d / "ffp.tif", np.stack([ffp * 0 + 7] + [ffp] * (n_channels - 1)), photometric="minisblack")
        tifffile.imwrite(d / "dfp.tif", np.stack([dfp * 0 + 9] + [dfp] * (n_channels - 1)), photometric="minisblack")
    else:
        tifffile.imwrite(d / "ffp.tif", ffp)
        tifffile.imwrite(d / "dfp.tif", dfp)
    return d / "ffp.tif", d / "dfp.tif", ffp, dfp


def test_illumination_is_ashlars_formula_on_every_pasted_window(tmp_path):
    tile = 64
    f_path, d_path, ffp, dfp = profiles(tmp_path / "prof", tile)
    cyc, _ = cut_cycle(tmp_path / "c", grid=(2, 2), tile=tile, overlap=0.25,
                       ffp=f_path, dfp=d_path)
    st = frontends.stitch(cyc, "stage", tmp_path / "o", out_tile=32)
    m = tifffile.imread(st.mosaic).squeeze()
    r = cyc.tiles[-1]  # the last tile is pasted whole (last-wins)
    x, y = (int(v) for v in st.positions[r.tile])
    want = ashlar_correct(tifffile.imread(r.path), dfp, ffp)
    np.testing.assert_array_equal(m[y:y + tile, x:x + tile], want)
    assert st.report["illumination"]["formula"].startswith("ashlar")


def test_illumination_takes_the_nuclear_plane_of_a_per_cycle_profile(tmp_path):
    tile = 64
    f_path, d_path, ffp, dfp = profiles(tmp_path / "prof", tile, n_channels=3)
    a = np.full((tile, tile), 30000, np.uint16)
    cyc = tiles.write_cycle(tmp_path / "c", iter([(1, a, 0, 0)]), pixel_size_um=1.0,
                            channels=["A", "DAPI", "C"], nuclear_channel=1,
                            ffp=f_path, dfp=d_path)
    st = frontends.stitch(cyc, "stage", tmp_path / "o")
    m = tifffile.imread(st.mosaic).squeeze()
    np.testing.assert_array_equal(m, ashlar_correct(a, dfp, ffp))


def test_relative_profile_paths_resolve_against_the_cycle(tmp_path):
    tile = 32
    profiles(tmp_path / "c", tile)
    a = np.full((tile, tile), 1000, np.uint16)
    cyc = tiles.write_cycle(tmp_path / "c", iter([(1, a, 0, 0)]), pixel_size_um=1.0,
                            channels=["DAPI"], nuclear_channel=0, ffp="ffp.tif", dfp="dfp.tif")
    st = frontends.stitch(cyc, "stage", tmp_path / "o")
    assert st.report["illumination"]["ffp"].endswith("ffp.tif")


def test_a_profile_of_the_wrong_size_is_refused(tmp_path):
    f_path, d_path, _, _ = profiles(tmp_path / "prof", 48)
    cyc, _ = cut_cycle(tmp_path / "c", grid=(1, 2), tile=64, ffp=f_path)
    with pytest.raises(ValueError, match="profile"):
        frontends.stitch(cyc, "stage", tmp_path / "o")


def test_a_flat_field_with_a_zero_is_refused(tmp_path):
    tifffile.imwrite(tmp_path / "ffp.tif", np.zeros((64, 64), np.float32))
    cyc, _ = cut_cycle(tmp_path / "c", grid=(1, 2), tile=64, ffp=tmp_path / "ffp.tif")
    with pytest.raises(ValueError, match="flat-field"):
        frontends.stitch(cyc, "stage", tmp_path / "o")


# ── ashlar_edge (fake_ashlar's EdgeAligner) ──────────────────────────────────
def test_ashlar_edge_positions_are_the_aligners_row_col_as_x_y(tmp_path, monkeypatch):
    cyc, truth = cut_cycle(tmp_path / "c", origin=(-300.0, -40.0))
    rc = [(truth[r.tile][1], truth[r.tile][0]) for r in cyc.tiles]  # (row, col)
    install_fake(monkeypatch, edge_positions=given_positions(rc))
    st = frontends.stitch(cyc, "ashlar_edge", tmp_path / "o", max_shift_um=7.5)
    xy = np.array([truth[r.tile] for r in cyc.tiles])
    xy -= xy.min(axis=0)
    got = np.array([st.positions[r.tile] for r in cyc.tiles])
    np.testing.assert_allclose(got, xy, atol=1e-9)
    (made,) = fake_ashlar.calls("EdgeAligner")
    assert made["channel"] == cyc.nuclear_channel and made["max_shift"] == 7.5  # um
    assert st.report["fallback_stage_positions"] is False


def test_ashlar_sees_the_tiles_through_a_reader_subclass_and_no_cache(tmp_path, monkeypatch):
    cyc, _ = cut_cycle(tmp_path / "c", grid=(2, 2), tile=64, origin=(-64.0, 0.0))
    install_fake(monkeypatch, edge_positions=lambda md: md.positions)
    reg = fake_ashlar.modules()["ashlar.reg"]
    reader = frontends.reader_of(cyc)
    assert isinstance(reader, reg.Reader) and isinstance(reader.metadata, reg.Metadata)
    md = reader.metadata
    assert md.num_images == 4 and md.pixel_size == cyc.pixel_size_um
    assert md.positions.tolist() == [[r.y_px, r.x_px + 64.0] for r in cyc.tiles]  # row, col
    assert md.size.tolist() == [64, 64]
    np.testing.assert_array_equal(reader.read(series=2, c=0), tifffile.imread(cyc.tiles[2].path))
    with pytest.raises(ValueError, match="channel"):
        reader.read(series=0, c=1)
    st = frontends.stitch(cyc, "ashlar_edge", tmp_path / "o")
    (ran,) = fake_ashlar.calls("EdgeAligner.run")
    assert ran["reader"] == "TileSeriesReader"  # not ASHLAR's CachingReader (reg.py:502)
    assert any("CachingReader" in s for s in st.report["benchmark_side"])
    assert st.report["thumbnail"] == {"shape": [6, 8], "bytes": 48, "scale": 0.05}


def test_ashlar_fallback_warning_is_recorded(tmp_path, monkeypatch):
    cyc, _ = cut_cycle(tmp_path / "c", grid=(2, 2), tile=64)
    install_fake(monkeypatch, edge_positions=lambda md: md.positions,
                 warn=ashlar_method.FALLBACK_WARNING + ", proceeding anyway with original"
                 " stage positions.")
    st = frontends.stitch(cyc, "ashlar_edge", tmp_path / "o")
    assert st.report["fallback_stage_positions"] is True
    assert st.report["ashlar_warnings"]


def test_ashlar_reader_serves_corrected_tiles(tmp_path, monkeypatch):
    tile = 64
    f_path, d_path, ffp, dfp = profiles(tmp_path / "prof", tile)
    cyc, _ = cut_cycle(tmp_path / "c", grid=(1, 2), tile=tile, ffp=f_path, dfp=d_path)
    install_fake(monkeypatch)
    reader = frontends.reader_of(cyc, frontends.Illumination.of(cyc))
    np.testing.assert_array_equal(
        reader.read(series=1, c=0), ashlar_correct(tifffile.imread(cyc.tiles[1].path), dfp, ffp)
    )


# ── stare_mosaic ─────────────────────────────────────────────────────────────
def test_stare_mosaic_is_called_with_the_brief_flags(tmp_path, monkeypatch):
    cyc, _ = cut_cycle(tmp_path / "c", grid=(2, 2), tile=64)
    seen = {}

    def fake_run(argv):
        seen["argv"] = list(argv)
        out = argv[argv.index("--out") + 1]
        doc = {"version": 1, "tiles": [
            {"tile": str(r.tile), "x": r.x_px + 5, "y": r.y_px + 5} for r in cyc.tiles
        ], "report": {"n_edges": 4, "isolated": []}}
        with open(out, "w") as fh:
            json.dump(doc, fh)

    monkeypatch.setattr(frontends, "_run_stare_mosaic", fake_run)
    monkeypatch.setattr(frontends, "_require_stare", lambda: "1.3.0")
    st = frontends.stitch(cyc, "stare_mosaic", tmp_path / "o", max_shift_um=9.0)
    argv = seen["argv"]
    assert argv[argv.index("--tiles") + 1] == str(cyc.dir / tiles.TILES_CSV)
    assert argv[argv.index("--max-shift-um") + 1] == "9.0"
    assert argv[argv.index("--pixel-size") + 1] == repr(cyc.pixel_size_um)
    assert st.positions == frontends.stage_positions(cyc)  # +5 moved back to the origin
    assert st.report["n_edges"] == 4
    assert "mosaic" not in st.report  # STARE's "mosaic": None would say no mosaic was written


def test_stare_mosaic_gets_corrected_tile_files_written_once(tmp_path, monkeypatch):
    tile = 64
    f_path, d_path, ffp, dfp = profiles(tmp_path / "prof", tile)
    cyc, _ = cut_cycle(tmp_path / "c", grid=(1, 2), tile=tile, ffp=f_path, dfp=d_path)
    seen = {}

    def fake_run(argv):
        csv_path = argv[argv.index("--tiles") + 1]
        seen["cycle"] = tiles.read_cycle(type(cyc.dir)(csv_path).parent)
        doc = {"tiles": [{"tile": str(r.tile), "x": r.x_px, "y": r.y_px} for r in cyc.tiles],
               "report": {}}
        type(cyc.dir)(argv[argv.index("--out") + 1]).write_text(json.dumps(doc))

    monkeypatch.setattr(frontends, "_run_stare_mosaic", fake_run)
    monkeypatch.setattr(frontends, "_require_stare", lambda: "1.3.0")
    frontends.stitch(cyc, "stare_mosaic", tmp_path / "o")
    got = seen["cycle"]
    assert got.dir != cyc.dir and [r.tile for r in got.tiles] == [r.tile for r in cyc.tiles]
    assert [(r.x_px, r.y_px) for r in got.tiles] == [(r.x_px, r.y_px) for r in cyc.tiles]
    for a, b in zip(cyc.tiles, got.tiles):
        np.testing.assert_array_equal(tifffile.imread(b.path),
                                      ashlar_correct(tifffile.imread(a.path), dfp, ffp))


def test_stare_older_than_1_3_is_refused(monkeypatch):
    import types

    monkeypatch.setitem(__import__("sys").modules, "stare", types.SimpleNamespace(__version__="1.2.1"))
    with pytest.raises(RuntimeError, match="1.3.0"):
        frontends._require_stare()


def _stare_1_3():
    stare = pytest.importorskip("stare")
    if tuple(int(v) for v in stare.__version__.split(".")[:3]) < (1, 3, 0):
        pytest.skip(f"stare {stare.__version__} < 1.3.0 has no `stare mosaic`")
    return stare


def test_stare_mosaic_recovers_the_jitter(tmp_path):
    """The real ``stare mosaic`` (a subprocess), on a 3x3 grid of 512 px tiles whose stage
    origin is NEGATIVE: the positions it returns are the true ones to < 0.5 px."""
    _stare_1_3()
    cyc, truth = cut_cycle(tmp_path / "c", tile=512, jitter_px=6.0, origin=(-2000.0, -700.0),
                           pixel_size_um=1.0)
    st = frontends.stitch(cyc, "stare_mosaic", tmp_path / "o", max_shift_um=20.0)
    ids = [r.tile for r in cyc.tiles]
    pos = np.array([st.positions[t] for t in ids])
    tru = np.array([truth[t] for t in ids])
    err = (pos - pos[0]) - (tru - tru[0])
    print("stare_mosaic max |err| px:", np.abs(err).max())
    assert np.abs(err).max() < 0.5
    assert pos.min(axis=0).tolist() == [0.0, 0.0]
    assert st.report["isolated"] == []


# ── one writer: equal positions give the same mosaic ─────────────────────────
def test_all_three_write_the_same_mosaic_from_the_same_positions(tmp_path, monkeypatch):
    cyc, _ = cut_cycle(tmp_path / "c", grid=(3, 3), tile=64, origin=(-10.0, 3.0))
    stage = frontends.stage_positions(cyc)
    install_fake(monkeypatch, edge_positions=lambda md: md.positions)

    def fake_run(argv):
        doc = {"tiles": [{"tile": str(t), "x": x, "y": y} for t, (x, y) in stage.items()],
               "report": {}}
        type(cyc.dir)(argv[argv.index("--out") + 1]).write_text(json.dumps(doc))

    monkeypatch.setattr(frontends, "_run_stare_mosaic", fake_run)
    monkeypatch.setattr(frontends, "_require_stare", lambda: "1.3.0")
    out = {fe: frontends.stitch(cyc, fe, tmp_path / fe, out_tile=32) for fe in frontends.FRONT_ENDS}
    arrays = {fe: tifffile.imread(s.mosaic) for fe, s in out.items()}
    assert len({a.shape for a in arrays.values()}) == 1
    for fe in ("ashlar_edge", "stare_mosaic"):
        assert out[fe].positions == stage
        np.testing.assert_array_equal(arrays[fe], arrays["stage"])


# ── memory: O(one tile + one write-tile), not O(grid) ────────────────────────
MEM_TILE = 512  # 0.5 MB a tile: tifffile's per-open bookkeeping (cyclic garbage) is far below it


def peak_of(fn):
    gc.collect()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def assert_flat(measure):
    """2x2 vs 6x6 (9x the tiles): the peak may grow by less than ONE tile (uint16); an
    O(n_tiles) buffer would grow by >= 32 tiles."""
    measure(2)
    measure(6)  # warm-up: lazy imports would otherwise land in the small peak
    small, big = measure(2), measure(6)
    one_tile = MEM_TILE * MEM_TILE * 2
    print(f"peak small={small} big={big} growth={big - small} tile={one_tile}")
    assert big - small < one_tile


def _fresh(tmp_path, tag):
    d = tmp_path / f"{tag}_{len(list(tmp_path.iterdir()))}"
    d.mkdir()
    return d


def test_writer_memory_does_not_grow_with_the_grid(tmp_path):
    def measure(grid):
        d = _fresh(tmp_path, f"w{grid}")
        cyc = noise_cycle(d / "c", grid, MEM_TILE)
        pos = frontends.stage_positions(cyc)
        return peak_of(lambda: frontends.write_mosaic(cyc, pos, d / "m.ome.tif", out_tile=512))

    assert_flat(measure)


def test_stage_memory_does_not_grow_with_the_grid_with_illumination(tmp_path):
    f_path, d_path, _, _ = profiles(tmp_path / "prof", MEM_TILE)

    def measure(grid):
        d = _fresh(tmp_path, f"s{grid}")
        cyc = noise_cycle(d / "c", grid, MEM_TILE)
        doc = json.loads((cyc.dir / tiles.CYCLE_JSON).read_text())
        doc.update(ffp=str(f_path), dfp=str(d_path))
        (cyc.dir / tiles.CYCLE_JSON).write_text(json.dumps(doc))
        cyc = tiles.read_cycle(cyc.dir)
        return peak_of(lambda: frontends.stitch(cyc, "stage", d / "o", out_tile=512))

    assert_flat(measure)


def test_ashlar_edge_memory_does_not_grow_with_the_grid(tmp_path, monkeypatch):
    """The fake aligner reads every tile through ``self.reader`` twice, as the real one does
    (thumbnail, then the pairs): through ASHLAR's ``CachingReader`` that would keep all of
    them (``reg.py:436-462``)."""

    def measure(grid):
        install_fake(monkeypatch, edge_positions=lambda md: md.positions)
        d = _fresh(tmp_path, f"a{grid}")
        cyc = noise_cycle(d / "c", grid, MEM_TILE)
        return peak_of(lambda: frontends.stitch(cyc, "ashlar_edge", d / "o", out_tile=512))

    assert_flat(measure)


def test_ashlar_edge_refuses_an_aligner_that_does_not_cache(tmp_path, monkeypatch):
    cyc, _ = cut_cycle(tmp_path / "c", grid=(2, 2), tile=64)
    install_fake(monkeypatch, edge_positions=lambda md: md.positions)
    reg = fake_ashlar.modules()["ashlar.reg"]
    orig = reg.EdgeAligner.__init__

    def init(self, reader, **kw):
        orig(self, reader, **kw)
        self.reader = reader  # a future ASHLAR that no longer wraps

    monkeypatch.setattr(reg.EdgeAligner, "__init__", init)
    with pytest.raises(RuntimeError, match="CachingReader"):
        frontends.stitch(cyc, "ashlar_edge", tmp_path / "o")
