"""Correctness test for the streaming (gigapixel-safe) stitch.

Proves the tiled, lazy-read, incremental-write path produces the *same* registered slide as a
single whole-image warp — so streaming buys bounded memory, not a different result. Also checks
non-negativity and that the output is a tiled OME-TIFF.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
    ),
)
pytest.importorskip("skimage")
pytest.importorskip("scipy")
pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")

import tiled_stitch  # noqa: E402
from mesh_field import MeshField  # noqa: E402
from tiled_manifest import slide_entry  # noqa: E402
from tiled_warp import warp_image  # noqa: E402


def _whole_image_reference(mov_chw, m0, mesh, out_shape, dtype):
    hwc = np.moveaxis(mov_chw, 0, -1).astype(float)
    # at the stitch's own sub-grid step: its nodes are global, so streaming changes nothing
    warped = warp_image(
        hwc, m0, mesh, out_shape, field_step=tiled_stitch.FIELD_STEP
    )  # (H, W, C)
    out = np.clip(warped, 0.0, None)
    info = np.iinfo(dtype)
    return np.moveaxis(np.clip(np.rint(out), info.min, info.max).astype(dtype), -1, 0)


def test_streaming_stitch_equals_the_whole_image_warp(tmp_path):
    rng = np.random.default_rng(0)
    h, w = 200, 180
    mov = (rng.uniform(0, 1, size=(2, h, w)) * 60000).astype(np.uint16)
    mov_f = tmp_path / "mov.ome.tiff"
    tifffile.imwrite(str(mov_f), mov, photometric="minisblack")

    m0 = [[1.0, 0.0, -3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]  # small translation
    # a gentle non-uniform mesh over a 2x2 control grid spanning the frame
    grid_x, grid_y = [0.0, float(w)], [0.0, float(h)]
    disp = [[[1.5, -1.0], [2.0, 0.5]], [[-1.0, 2.0], [0.5, 1.5]]]
    entry = slide_entry(m0, grid_x, grid_y, disp)
    entry["out_shape"] = [h, w]
    manifest = {
        "ref_slide": "ref",
        "slides": {
            "ref": {"M0": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "mesh": None},
            "mov": entry,
        },
    }
    man_f = tmp_path / "manifest.json"
    man_f.write_text(json.dumps(manifest))

    out_f = tmp_path / "registered.ome.tiff"
    # small out-tile forces many tiles / strips, exercising the streaming path hard
    tiled_stitch.main(
        [
            "--moving",
            str(mov_f),
            "--manifest",
            str(man_f),
            "--out",
            str(out_f),
            "--out-tile",
            "48",
            "--pixel-size",
            "0.325",
        ]
    )

    streamed = tifffile.imread(str(out_f))
    assert streamed.shape == (2, h, w)
    assert streamed.dtype == np.uint16
    assert streamed.min() >= 0

    mesh = MeshField(grid_x, grid_y, np.asarray(disp, dtype=float))
    reference = _whole_image_reference(
        mov, np.asarray(m0, float), mesh, (h, w), np.uint16
    )
    # identical up to per-tile integer rounding at a few sub-pixel boundaries
    assert (
        np.array_equal(streamed, reference)
        or np.abs(streamed.astype(int) - reference.astype(int)).max() <= 1
    )

    # the output is genuinely tiled (gigapixel-friendly), not one giant strip
    with tifffile.TiffFile(str(out_f)) as tif:
        assert tif.pages[0].is_tiled
        # ...and carries 64-bit offsets. A real registered slide (C x H x W x itemsize,
        # written uncompressed) crosses classic TIFF's 4 GB offset ceiling and dies mid-write
        # with "struct.error: 'I' format requires 0 <= number <= 4294967295". This assertion
        # is the only thing that catches it, since the test slide itself is tiny.
        assert tif.is_bigtiff
