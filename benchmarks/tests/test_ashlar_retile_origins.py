"""``retile.write_tiles`` with the tiles' origins given, and with a floor on the noise.

Two additions the registration benchmark needs (``benchmarks/regbench/methods/ashlar.py``):
moving tiles cut where an anchor puts them, and a noise s.d. that does not vanish on a
channel with a narrow range.
"""

import json

import numpy as np
import pytest
import tifffile

from benchmarks.ashlar import retile


def _slide(path, planes):
    tifffile.imwrite(
        path,
        np.asarray(planes),
        photometric="minisblack",
        metadata={"axes": "CYX"},
        ome=True,
    )
    return path


def _two_channel(tmp_path):
    rng = np.random.default_rng(3)
    img = rng.integers(100, 60000, (2, 300, 400)).astype(np.uint16)
    return _slide(tmp_path / "s.ome.tif", img), img


def _n_tiles(img, tile=128, overlap=0.1):
    rows, cols = retile.grid_shape(img.shape[2], img.shape[1], tile, overlap)
    return rows * cols


def test_a_tile_is_cut_where_its_origin_says_and_padded_on_every_side(tmp_path):
    path, img = _two_channel(tmp_path)
    nominal = [(t.y, t.x) for t in retile.tile_grid(400, 300, 128, 0.1)]
    origins = np.array(nominal)
    origins[0] = (-20, 30)
    origins[-1] = (250, 350)  # runs off the bottom and the right
    grid_path = retile.write_tiles(
        path, tmp_path / "t", 128, 0.1, tile_origins_yx=origins, exact_overlap=True
    )
    grid = json.loads(grid_path.read_text())
    first = tifffile.imread(tmp_path / "t" / "r000_c000.tif")
    assert first.shape == (2, 128, 128)  # both channels, uniform size
    assert not first[:, :20].any()  # the 20 rows above the slide are zero
    np.testing.assert_array_equal(first[:, 20:60, :40], img[:, 0:40, 30:70])
    assert grid["true_positions_yx"][0] == [-20, 30]
    assert grid["true_positions_yx"][1] == list(nominal[1])
    # valid extent: [row, col, w, h] of the part that is slide
    assert grid["valid_extent"][0] == [0, 0, 128, 108]
    last = grid["valid_extent"][-1]
    assert last[2:] == [50, 50]
    tile = tifffile.imread(
        tmp_path / "t" / retile.TILE_PATTERN.format(row=last[0], col=last[1])
    )
    np.testing.assert_array_equal(tile[:, :50, :50], img[:, 250:, 350:])
    assert not tile[:, 50:].any() and not tile[:, :, 50:].any()
    # ASHLAR is still told the regular grid: file names and overlap are the nominal ones
    assert grid["overlap"] == pytest.approx(1 - grid["stride"] / 128)
    assert grid["tile_origins"] == "given"


def test_without_origins_nothing_changes(tmp_path):
    path, img = _two_channel(tmp_path)
    grid = json.loads(retile.write_tiles(path, tmp_path / "t", 128, 0.1).read_text())
    nominal = [[t.y, t.x] for t in retile.tile_grid(400, 300, 128, 0.1)]
    assert grid["true_positions_yx"] == nominal
    assert grid["tile_origins"] == "grid"
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "t" / "r000_c000.tif"), img[:, :128, :128]
    )


def test_origins_and_stage_jitter_are_refused_together(tmp_path):
    path, img = _two_channel(tmp_path)
    origins = np.zeros((_n_tiles(img), 2), dtype=int)
    with pytest.raises(ValueError, match="stage jitter"):
        retile.write_tiles(
            path,
            tmp_path / "t",
            128,
            0.1,
            tile_origins_yx=origins,
            stage_jitter_um=2.0,
        )


def test_a_wrong_number_of_origins_is_refused(tmp_path):
    path, img = _two_channel(tmp_path)
    origins = np.zeros((_n_tiles(img) - 1, 2), dtype=int)
    with pytest.raises(ValueError, match="one origin per tile"):
        retile.write_tiles(path, tmp_path / "t", 128, 0.1, tile_origins_yx=origins)
    with pytest.raises(ValueError, match="whole pixels"):
        retile.write_tiles(
            path,
            tmp_path / "t",
            128,
            0.1,
            tile_origins_yx=np.full((_n_tiles(img), 2), 0.5),
        )


def test_the_noise_has_a_floor_in_counts(tmp_path):
    """A ramp whose range is 100: 0.2 % of it is 0.2 counts, the floor of 3 wins. The
    difference of two tilings holds two independent draws, hence the sqrt(2)."""
    ramp = np.tile(np.linspace(1000, 1100, 400), (300, 1)).astype(np.uint16)
    path = _slide(tmp_path / "ramp.ome.tif", ramp[None])
    tiles = []
    for seed in (1, 2):
        out = tmp_path / f"t{seed}"
        grid_path = retile.write_tiles(
            path, out, 128, 0.1, noise_frac=0.002, noise_min_counts=3.0, seed=seed
        )
        assert json.loads(grid_path.read_text())["noise_sd"] == [3.0]
        tiles.append(tifffile.imread(out / "r000_c000.tif").astype(float))
    sd = (tiles[0] - tiles[1]).std() / np.sqrt(2)
    assert 2.5 < sd < 3.5
    # without the floor the same call gives 0.2 % of the range
    grid_path = retile.write_tiles(path, tmp_path / "t0", 128, 0.1, noise_frac=0.002)
    assert json.loads(grid_path.read_text())["noise_sd"][0] == pytest.approx(
        0.2, abs=0.01
    )
