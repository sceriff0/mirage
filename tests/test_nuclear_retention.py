from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("tifffile")
BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))
sys.path.insert(0, str(BIN / "utils"))

import nuclear_retention as nr  # noqa: E402


def _masks():
    cell = np.zeros((20, 40), dtype=np.uint32)
    nuc = np.zeros_like(cell)
    cell[2:18, 2:18], nuc[6:14, 6:14] = 1, 1
    cell[2:18, 22:38], nuc[6:14, 26:34] = 2, 2
    return cell, nuc


def test_nuclear_channel_index_uses_the_shared_rule():
    assert nr.nuclear_channel_index(["CD3", "DAPI_nuclear", "CD8"], ["DAPI"]) == 1
    assert nr.nuclear_channel_index(["CD3", "CD8"], ["DAPI"]) is None


def test_measure_reports_nucleus_and_cell_medians():
    cell, nuc = _masks()
    plane = np.zeros(cell.shape, dtype=np.uint16)
    plane[nuc == 1] = 1000
    plane[nuc == 2] = 10  # this cell's nucleus is (nearly) gone
    out = nr.measure_nuclear(cell, nuc, plane)
    assert list(out.columns) == ["label", "Nucleus", "Cell"]
    row = out.set_index("label")
    assert row.loc[1, "Nucleus"] == 1000 and row.loc[2, "Nucleus"] == 10


def test_measure_without_nuclei_reports_cell_only():
    cell, _ = _masks()
    out = nr.measure_nuclear(cell, None, np.ones(cell.shape, dtype=np.uint16))
    assert list(out.columns) == ["label", "Cell"]


def test_resolve_channel_names_prefers_explicit_channels_over_ome(tmp_path):
    import tifffile

    img = tmp_path / "s.ome.tif"
    # No Name given -> OME falls back to generic Channel IDs (metadata.py's own
    # fallback), never a real marker name.
    tifffile.imwrite(str(img), np.zeros((2, 20, 40), dtype=np.uint16))
    names = nr.resolve_channel_names(str(img), 2, ["CD3", "DAPI"])
    assert names == ["CD3", "DAPI"]


def test_resolve_channel_names_falls_back_to_ome_when_channels_omitted(tmp_path):
    import tifffile

    img = tmp_path / "s.ome.tif"
    tifffile.imwrite(
        str(img),
        np.zeros((2, 20, 40), dtype=np.uint16),
        metadata={"axes": "CYX", "Channel": {"Name": ["CD3", "CD8"]}},
    )
    names = nr.resolve_channel_names(str(img), 2, None)
    assert names == ["CD3", "CD8"]


def test_resolve_channel_names_pads_and_truncates_length_mismatch_like_split_multichannel(
    tmp_path,
):
    img = tmp_path / "s.ome.tif"
    # Too few names: padded with generic Channel_i, same as split_multichannel.py.
    assert nr.resolve_channel_names(str(img), 3, ["CD3"]) == [
        "CD3",
        "Channel_1",
        "Channel_2",
    ]
    # Too many names: truncated.
    assert nr.resolve_channel_names(str(img), 1, ["CD3", "DAPI"]) == ["CD3"]


def test_channels_arg_finds_nuclear_plane_when_ome_names_are_generic(tmp_path):
    """SPLIT_CHANNELS prefers meta.channels over generic OME names (finding 1);
    NUCLEAR_RETENTION must agree on which plane is nuclear under the same input."""
    import tifffile

    cell, nuc = _masks()
    img = tmp_path / "s.ome.tif"
    plane0 = np.full(cell.shape, 5, dtype=np.uint16)
    plane1 = np.zeros(cell.shape, dtype=np.uint16)
    plane1[nuc == 1] = 1000
    plane1[nuc == 2] = 10
    # No explicit Name -> OME channel names are generic IDs, not "CD3"/"DAPI".
    tifffile.imwrite(str(img), np.stack([plane0, plane1]))
    np.save(tmp_path / "c.npy", cell)
    np.save(tmp_path / "n.npy", nuc)
    out = tmp_path / "r.csv"
    rc = nr.main(
        [
            "--image",
            str(img),
            "--mask_file",
            str(tmp_path / "c.npy"),
            "--nuclei_mask_file",
            str(tmp_path / "n.npy"),
            "--nuclear-markers",
            "DAPI",
            "--channels",
            "CD3",
            "DAPI",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    df = pd.read_csv(out)
    # If the generic OME names had been used instead, no channel would match
    # "DAPI" and this table would come back empty (label-only).
    assert list(df.columns) == ["label", "Nucleus", "Cell"]
    row = df.set_index("label")
    assert row.loc[1, "Nucleus"] == 1000 and row.loc[2, "Nucleus"] == 10


def test_slide_without_a_nuclear_channel_writes_an_empty_table(tmp_path, monkeypatch):
    import tifffile

    cell, nuc = _masks()
    img = tmp_path / "s.ome.tif"
    tifffile.imwrite(
        str(img),
        np.zeros((2, *cell.shape), dtype=np.uint16),
        metadata={"axes": "CYX", "Channel": {"Name": ["CD3", "CD8"]}},
    )
    np.save(tmp_path / "c.npy", cell)
    out = tmp_path / "r.csv"
    rc = nr.main(
        [
            "--image",
            str(img),
            "--mask_file",
            str(tmp_path / "c.npy"),
            "--nuclear-markers",
            "DAPI",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    df = pd.read_csv(out)
    assert list(df.columns) == ["label"] and df.empty
