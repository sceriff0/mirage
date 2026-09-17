"""ASHLAR retile: every slide of a patient on ONE canvas, at the run's pixel size.

A real 033 run (job 6831633, 2026-09-16) died in ASHLAR_SOLVE with "reference and moving
grids disagree on n_rows (35 vs 31)": the reference and moving slides differ in size, each
was retiled on its own extent, and LayerAligner matches tiles one-for-one. The same retile
read 0.34533768547788 um/px out of the preprocessed OME header while the run was given
0.325, which rescales --maximum-shift. Both are fixed at retile time and pinned here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from benchmarks.ashlar import retile


def _slide(path, h, w, px_header=0.34533768547788, seed=0):
    rng = np.random.default_rng(seed)
    data = (rng.random((2, h, w)) * 4000).astype(np.uint16)
    tifffile.imwrite(
        str(path),
        data,
        ome=True,
        metadata={
            "axes": "CYX",
            "PhysicalSizeX": px_header,
            "PhysicalSizeY": px_header,
        },
    )
    return data


def test_slides_of_different_size_disagree_without_a_common_canvas(tmp_path):
    _slide(tmp_path / "ref.ome.tif", 700, 900)
    _slide(tmp_path / "mov.ome.tif", 600, 800, seed=1)
    g_ref = json.loads(
        retile.write_tiles(
            tmp_path / "ref.ome.tif", tmp_path / "r", 128, 0.1
        ).read_text()
    )
    g_mov = json.loads(
        retile.write_tiles(
            tmp_path / "mov.ome.tif", tmp_path / "m", 128, 0.1
        ).read_text()
    )
    assert (g_ref["n_rows"], g_ref["n_cols"]) != (
        g_mov["n_rows"],
        g_mov["n_cols"],
    )  # the failure


def test_a_common_canvas_gives_identical_grids_and_keeps_the_real_shape(tmp_path):
    ref = _slide(tmp_path / "ref.ome.tif", 700, 900)
    mov = _slide(tmp_path / "mov.ome.tif", 600, 950, seed=1)
    both = [tmp_path / "ref.ome.tif", tmp_path / "mov.ome.tif"]
    g_ref = json.loads(
        retile.write_tiles(
            both[0], tmp_path / "r", 128, 0.1, canvas_like=both
        ).read_text()
    )
    g_mov = json.loads(
        retile.write_tiles(
            both[1], tmp_path / "m", 128, 0.1, canvas_like=both
        ).read_text()
    )
    for key in ("n_rows", "n_cols", "tile_size", "overlap", "stride"):
        assert g_ref[key] == g_mov[key], key
    assert g_ref["canvas_shape"] == g_mov["canvas_shape"] == [700, 950]
    # the stitched output is sized from the reference's REAL shape, never the padded canvas
    assert g_ref["orig_shape"] == [700, 900] and g_mov["orig_shape"] == [600, 950]
    # padding is zero and not counted as valid
    stride = g_mov["stride"]
    last_row = g_mov["n_rows"] - 1
    tile = tifffile.imread(
        tmp_path / "m" / retile.TILE_PATTERN.format(row=last_row, col=0)
    )
    y0 = last_row * stride
    valid_h = max(0, min(128, 600 - y0))
    assert not tile[:, valid_h:, :].any()
    extent = {(r, c): (w, h) for r, c, w, h in g_mov["valid_extent"]}
    assert extent[(last_row, 0)][1] == valid_h
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "m" / retile.TILE_PATTERN.format(row=1, col=2)),
        mov[:, stride : stride + 128, 2 * stride : 2 * stride + 128],
    )
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "r" / retile.TILE_PATTERN.format(row=0, col=0)),
        ref[:, :128, :128],
    )


def test_the_run_pixel_size_overrides_the_file_header(tmp_path):
    _slide(tmp_path / "ref.ome.tif", 300, 300)
    g = json.loads(
        retile.write_tiles(
            tmp_path / "ref.ome.tif", tmp_path / "r", 128, 0.1, pixel_size_um=0.325
        ).read_text()
    )
    assert g["pixel_size_um"] == pytest.approx(0.325)
    g = json.loads(
        retile.write_tiles(
            tmp_path / "ref.ome.tif", tmp_path / "h", 128, 0.1
        ).read_text()
    )
    assert g["pixel_size_um"] == pytest.approx(
        0.34533768547788
    )  # header only when not given


def test_the_cli_takes_the_canvas_and_the_pixel_size(tmp_path):
    _slide(tmp_path / "ref.ome.tif", 300, 400)
    _slide(tmp_path / "mov.ome.tif", 350, 300, seed=1)
    argv = [
        "--image",
        str(tmp_path / "mov.ome.tif"),
        "--outdir",
        str(tmp_path / "m"),
        "--tile-size",
        "128",
        "--overlap",
        "0.1",
        "--pixel-size-um",
        "0.325",
        "--canvas-like",
        str(tmp_path / "ref.ome.tif"),
        str(tmp_path / "mov.ome.tif"),
    ]
    assert retile.main(argv) == 0
    g = json.loads((tmp_path / "m" / "grid.json").read_text())
    assert g["canvas_shape"] == [350, 400] and g["pixel_size_um"] == pytest.approx(
        0.325
    )
