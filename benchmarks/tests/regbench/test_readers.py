"""The three raw-tile readers, on stand-in trees written in each dataset's real layout."""

import gc
import json
import tracemalloc
import zipfile

import numpy as np
import pytest
import tifffile
from scipy import ndimage

from benchmarks.regbench.readers import (
    _common,
    bioformats_tiles,
    codex_keyence,
    phenocycler,
)

from . import fake_ashlar

SHARP_Z = 1


def planes_of_tile(tile, n_z, size, seed=0):
    """``n_z`` planes of one tile: only z == SHARP_Z is sharp; the others are blurred."""
    rng = np.random.default_rng([seed, tile])
    sharp = rng.integers(0, 60000, (size, size)).astype(np.uint16)
    blurred = ndimage.gaussian_filter(sharp.astype(np.float32), 4).astype(np.uint16)
    return [sharp if z == SHARP_Z else blurred for z in range(n_z)], sharp


# ── PhenoCycler ───────────────────────────────────────────────────────────────
def pheno_experiment(grid, size, n_z=3, n_ch=2, n_cyc=2):
    return {
        "regionWidth": grid,
        "regionHeight": grid,
        "tileWidth": size,
        "tileHeight": size,
        "tileOverlapX": 0.25,
        "tileOverlapY": 0.5,
        "tilingMode": "snakerows",
        "numZPlanes": n_z,
        "xyResolution": 325,
        "channelNames": {
            "channelNamesArray": [
                f"{'DAPI' if c == 0 else 'M' + str(k) + str(c)}" for k in range(n_cyc) for c in range(n_ch)
            ]
        },
    }


def write_pheno(root, grid=2, size=64, n_z=3, n_ch=2, n_cyc=2):
    out = root / "MICROSCOPE_OUTPUT"
    out.mkdir(parents=True)
    (out / "experiment.json").write_text(json.dumps(pheno_experiment(grid, size, n_z, n_ch, n_cyc)))
    truth = {}
    for cyc in range(1, n_cyc + 1):
        d = out / f"cyc{cyc:03d}_reg001"
        d.mkdir()
        for tile in range(1, grid * grid + 1):
            planes, sharp = planes_of_tile(tile, n_z, size, seed=cyc)
            truth[(cyc, tile)] = sharp
            for z, plane in enumerate(planes, 1):
                for ch in range(1, n_ch + 1):
                    # a different channel must never be read: it is all zeros
                    tifffile.imwrite(
                        d / f"1_{tile:05d}_Z{z:03d}_CH{ch}.tif",
                        plane if ch == 1 else np.zeros_like(plane),
                    )
    return truth


def test_phenocycler_follows_snakerows_and_picks_the_sharp_z(tmp_path):
    truth = write_pheno(tmp_path / "raw")
    cycles = phenocycler.export(tmp_path / "raw", tmp_path / "out")
    assert [c.dir.name for c in cycles] == ["cyc001_reg001", "cyc002_reg001"]
    c1 = cycles[0]
    pitch_x, pitch_y = 64 * 0.75, 64 * 0.5
    pos = {r.tile: (r.x_px, r.y_px) for r in c1.tiles}
    assert pos[1] == (0.0, 0.0) and pos[2] == (pitch_x, 0.0)
    assert pos[3] == (pitch_x, pitch_y)  # row 1 runs right to left: tile 3 is at col 1
    assert pos[4] == (0.0, pitch_y)  # ... and tile 4 at col 0
    assert c1.pixel_size_um == pytest.approx(0.325)
    assert c1.channels == ("DAPI", "M01") and c1.nuclear_channel == 0
    doc = json.loads((c1.dir / "cycle.json").read_text())
    assert doc["z"] == {"1": 2, "2": 2, "3": 2, "4": 2}  # 1-based z = file Z002 = index 1
    for row in c1.tiles:
        np.testing.assert_array_equal(tifffile.imread(row.path), truth[(1, row.tile)])
    assert cycles[1].channels == ("DAPI", "M11")  # the second cycle's slice of the names


def test_phenocycler_refuses_another_tiling_mode(tmp_path):
    write_pheno(tmp_path / "raw")
    p = tmp_path / "raw" / "MICROSCOPE_OUTPUT" / "experiment.json"
    p.write_text(p.read_text().replace("snakerows", "rows"))
    with pytest.raises(ValueError, match="snakerows"):
        phenocycler.export(tmp_path / "raw", tmp_path / "out")


# ── Keyence CODEX ─────────────────────────────────────────────────────────────
def codex_experiment(grid, size, n_z=3):
    return {
        "region_width": grid,
        "region_height": grid,
        "tile_width": size,
        "tile_height": size,
        "tile_overlap_X": size // 4,
        "tile_overlap_Y": size // 2,
        "tiling_mode": "snake",
        "num_z_planes": n_z,
        "per_pixel_XY_resolution": 377.442,
    }


def write_codex(folder, grid=2, size=64, n_z=3, n_ch=2, as_zip=True, reg=4):
    """``Cyc2_reg4.zip`` (stored) holding ``Cyc2_reg4/<reg>_<tile5>_Z<zzz>_CH<n>.tif``."""
    folder.mkdir(parents=True)
    (folder / "Experiment.json").write_text(json.dumps(codex_experiment(grid, size, n_z)))
    (folder / "channelnames.txt").write_text(
        "\n".join(["HOECHST", "CD3", "HOECHST2", "CD8"][: 2 * n_ch]) + "\n"
    )
    members = {}
    truth = {}
    for tile in range(1, grid * grid + 1):
        planes, sharp = planes_of_tile(tile, n_z, size)
        truth[tile] = sharp
        for z, plane in enumerate(planes, 1):
            for ch in range(1, n_ch + 1):
                members[f"Cyc2_reg{reg}/{reg}_{tile:05d}_Z{z:03d}_CH{ch}.tif"] = (
                    plane if ch == 1 else np.zeros_like(plane)
                )
    if as_zip:
        target = folder / f"Cyc2_reg{reg}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as zf:
            for name, arr in members.items():
                tmp = folder / "_m.tif"
                tifffile.imwrite(tmp, arr)
                zf.write(tmp, name)
                tmp.unlink()
    else:
        target = folder / f"Cyc2_reg{reg}"
        target.mkdir()
        for name, arr in members.items():
            tifffile.imwrite(target / name.split("/")[1], arr)
    return target, folder / "Experiment.json", truth


@pytest.mark.parametrize("as_zip", [True, False])
def test_codex_follows_the_snake_and_picks_the_sharp_z(tmp_path, as_zip):
    src, exp, truth = write_codex(tmp_path / "raw" / "sample", as_zip=as_zip)
    cyc = codex_keyence.export(src, exp, tmp_path / "out")
    pos = {r.tile: (r.x_px, r.y_px) for r in cyc.tiles}
    assert pos[1] == (0.0, 0.0) and pos[2] == (48.0, 0.0)  # pitch = tile - overlap (px)
    assert pos[3] == (48.0, 32.0) and pos[4] == (0.0, 32.0)
    assert cyc.pixel_size_um == pytest.approx(0.377442)
    assert cyc.channels == ("HOECHST2", "CD8")  # Cyc2's slice of the four names
    assert json.loads((cyc.dir / "cycle.json").read_text())["z"] == {
        "1": 2, "2": 2, "3": 2, "4": 2
    }
    for row in cyc.tiles:
        np.testing.assert_array_equal(tifffile.imread(row.path), truth[row.tile])


def test_codex_refuses_conflicting_channel_name_files():
    assert codex_keyence.pick_channel_names(
        {"channelNames.txt": "A\nB\n", "channelnames.txt": "A\nB\n"}
    ) == ["A", "B"]
    with pytest.raises(ValueError, match="different content"):
        codex_keyence.pick_channel_names(
            {"channelNames.txt": "A\nB\n", "channelnames.txt": "A\nC\n"}
        )
    assert codex_keyence.pick_channel_names({"other.txt": "x"}) is None


def test_codex_refuses_a_sample_folder_with_both_spellings(tmp_path):
    src, exp, _ = write_codex(tmp_path / "raw" / "sample")
    other = exp.parent / "channelNames.txt"
    other.write_text("X\nY\nZ\nW\n")
    if len(list(exp.parent.glob("channel[nN]ames.txt"))) < 2:
        pytest.skip("case-insensitive file system: the two spellings cannot coexist")
    with pytest.raises(ValueError, match="different content"):
        codex_keyence.export(src, exp, tmp_path / "out")


# ── Bio-Formats (through ASHLAR's reader) ────────────────────────────────────
def test_bioformats_positions_are_pixels_row_col(tmp_path, monkeypatch):
    """ashlar 1.20.0 ``reg.py:349-389``: stage (Y, X) um, y flipped, / pixel_size -> pixels
    in (row, col). ``x_px = col``, ``y_px = row``."""
    pixel = 0.65
    stage = [(0.0, 0.0), (0.0, 650.0), (-325.0, 0.0), (-325.0, 650.0)]  # (Y, X) um
    positions = fake_ashlar.stage_to_positions(stage, pixel)
    assert positions.tolist() == [[0.0, 0.0], [0.0, 1000.0], [500.0, 0.0], [500.0, 1000.0]]
    size = 32
    fake_ashlar.install(
        monkeypatch.setitem,
        bioformats={
            "positions": positions,
            "pixel_size": pixel,
            "num_channels": 2,
            "planes": lambda s, c: np.full((size, size), 10 * s + c, np.uint16),
        },
    )
    cyc = bioformats_tiles.export(tmp_path / "x.ome.tiff", tmp_path / "out", 1)
    assert [(r.tile, r.x_px, r.y_px) for r in cyc.tiles] == [
        (1, 0.0, 0.0), (2, 1000.0, 0.0), (3, 0.0, 500.0), (4, 1000.0, 500.0)
    ]
    assert cyc.pixel_size_um == pixel and cyc.nuclear_channel == 1
    assert cyc.channels == ("C1", "C2")
    for row, s in zip(cyc.tiles, range(4)):
        assert tifffile.imread(row.path)[0, 0] == 10 * s + 1  # channel 1 only
    assert {c["c"] for c in fake_ashlar.calls("BioformatsReader.read")} == {1}


def test_bioformats_refuses_a_nuclear_channel_the_file_lacks(tmp_path, monkeypatch):
    fake_ashlar.install(
        monkeypatch.setitem,
        bioformats={"positions": [[0, 0]], "pixel_size": 1.0, "num_channels": 2,
                    "planes": lambda s, c: np.zeros((8, 8), np.uint16)},
    )
    with pytest.raises(ValueError, match="2 channels"):
        bioformats_tiles.export(tmp_path / "x", tmp_path / "out", 5)


# ── shared pieces ─────────────────────────────────────────────────────────────
def test_snake_rowcol():
    assert [_common.snake_rowcol(i, 3) for i in range(6)] == [
        (0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)
    ]


# ── memory: a reader holds one tile, not the grid ────────────────────────────
TILE = 512  # 0.5 MB a tile: the per-file bookkeeping (paths, zip entries) is far below it


def peak_of(fn):
    gc.collect()  # garbage from the previous run is not this run's peak
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def assert_flat(measure):
    """``measure(grid)`` -> traced peak. 2x2 vs 6x6 (9x the tiles): the growth must stay
    under one tile; an O(n_tiles) buffer would grow by >= 8 tiles."""
    measure(2)  # warm-up: the first call's lazy imports (scipy, tifffile) would land in
    measure(6)  # the small peak and mask growth
    small, big = measure(2), measure(6)
    one_tile = TILE * TILE * 2
    print(f"peak small={small} big={big} growth={big - small} tile={one_tile}")
    assert big - small < one_tile


def test_phenocycler_memory_does_not_grow_with_the_grid(tmp_path):
    def measure(grid):
        root = tmp_path / f"p{grid}_{len(list(tmp_path.iterdir()))}"
        write_pheno(root, grid=grid, size=TILE, n_cyc=1)
        return peak_of(lambda: phenocycler.export(root, root / "out"))

    assert_flat(measure)


@pytest.mark.parametrize("as_zip", [True, False])
def test_codex_memory_does_not_grow_with_the_grid(tmp_path, as_zip):
    def measure(grid):
        d = tmp_path / f"c{grid}_{len(list(tmp_path.iterdir()))}"
        src, exp, _ = write_codex(d, grid=grid, size=TILE, as_zip=as_zip)
        return peak_of(lambda: codex_keyence.export(src, exp, d / "out"))

    assert_flat(measure)


def test_bioformats_memory_does_not_grow_with_the_grid(tmp_path, monkeypatch):
    def measure(grid):
        n = grid * grid
        fake_ashlar.install(
            monkeypatch.setitem,
            bioformats={
                "positions": [(i // grid * 200.0, i % grid * 200.0) for i in range(n)],
                "pixel_size": 1.0,
                "planes": lambda s, c: np.random.default_rng(s).integers(
                    0, 999, (TILE, TILE)
                ).astype(np.uint16),
            },
        )
        d = tmp_path / f"b{grid}_{len(list(tmp_path.iterdir()))}"
        d.mkdir()
        return peak_of(lambda: bioformats_tiles.export(d / "x", d / "out", 0))

    assert_flat(measure)


def test_best_focus_keeps_the_sharpest_plane_and_the_first_of_a_tie():
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 100, (8, 8)).astype(np.uint16)
    blurred = ndimage.uniform_filter(sharp.astype(np.float32), 5).astype(np.uint16)
    best = _common.BestFocus()
    for z, plane in enumerate([blurred, sharp, sharp.copy(), blurred]):
        best.offer(z, plane)
    assert best.z == 1 and best.plane is not None and best.score > 0
    np.testing.assert_array_equal(best.plane, sharp)
