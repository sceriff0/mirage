"""The pipeline's intermediates must carry a scale.

`SPLIT_CHANNELS` and `TILED_STITCH` both used to write their outputs with no scale of
any kind — no OME header, no resolution tags — so every file between registration and
the published pyramid claimed 1 px = 1 unit. That is why `MERGE_AND_PYRAMID` and
`EXPORT_GEOJSON` have nothing to read and must take `params.pixel_size` on faith: the
information was not withheld from them, it was destroyed upstream.

These tests pin the stamp. They are about the *tags*, not about any µm value changing:
`params.pixel_size` remains the one owner of every measurement conversion.
"""

from __future__ import annotations

import json
import logging
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
tifffile = pytest.importorskip("tifffile")

import split_multichannel  # noqa: E402


def _resolution_um_per_px(path) -> float:
    """Invert the TIFF resolution tags back to µm/px."""
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        assert page.tags["ResolutionUnit"].value == 3, "must be CENTIMETER"
        num, den = page.tags["XResolution"].value  # pixels per cm, as a rational
        return 1e4 / (num / den)


def _multichannel(tmp_path, **metadata):
    path = tmp_path / "registered.ome.tiff"
    tifffile.imwrite(
        str(path),
        np.zeros((2, 8, 8), dtype=np.uint16),
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "PANCK"]}, **metadata},
    )
    return path


def test_split_channels_stamps_the_configured_scale(tmp_path):
    out = tmp_path / "channels"
    saved = split_multichannel.split_multichannel_tiff(
        str(_multichannel(tmp_path)),
        str(out),
        is_reference=True,
        channel_names=["DAPI", "PANCK"],
        nuclear_markers=["DAPI"],
        pixel_size=0.5,
    )
    assert len(saved) == 2
    for path in saved:
        assert _resolution_um_per_px(path) == pytest.approx(0.5, rel=1e-9)


def test_split_channels_warns_when_the_input_disagrees(tmp_path, caplog):
    src = _multichannel(tmp_path, PhysicalSizeX=0.2125, PhysicalSizeY=0.2125)
    with caplog.at_level(logging.WARNING):
        split_multichannel.split_multichannel_tiff(
            str(src),
            str(tmp_path / "channels"),
            is_reference=True,
            channel_names=["DAPI", "PANCK"],
            nuclear_markers=["DAPI"],
            pixel_size=0.325,
        )
    assert "SCALE MISMATCH" in caplog.text


def test_tiled_stitch_stamps_the_configured_scale(tmp_path):
    """The STARE path's own writer, which dropped the scale on the whole slide."""
    pytest.importorskip("scipy")
    pytest.importorskip("zarr")
    import tiled_stitch
    from tiled_manifest import slide_entry

    mov = tmp_path / "mov.ome.tiff"
    tifffile.imwrite(
        str(mov), np.zeros((2, 32, 32), dtype=np.uint16), photometric="minisblack"
    )
    entry = slide_entry(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], None, None, None
    )
    entry["out_shape"] = [32, 32]
    man = tmp_path / "manifest.json"
    man.write_text(
        json.dumps(
            {
                "ref_slide": "ref",
                "slides": {
                    "ref": {"M0": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "mesh": None},
                    "mov": entry,
                },
            }
        )
    )
    out = tmp_path / "registered.ome.tiff"
    tiled_stitch.main(
        [
            "--moving",
            str(mov),
            "--manifest",
            str(man),
            "--out",
            str(out),
            "--out-tile",
            "16",
            "--pixel-size",
            "0.2125",
        ]
    )
    assert _resolution_um_per_px(out) == pytest.approx(0.2125, rel=1e-6)
