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


def test_slide_without_a_nuclear_channel_writes_an_empty_table(tmp_path, monkeypatch):
    import tifffile

    cell, nuc = _masks()
    img = tmp_path / "s.ome.tif"
    tifffile.imwrite(str(img), np.zeros((2, *cell.shape), dtype=np.uint16),
                     metadata={"axes": "CYX", "Channel": {"Name": ["CD3", "CD8"]}})
    np.save(tmp_path / "c.npy", cell)
    out = tmp_path / "r.csv"
    rc = nr.main([
        "--image", str(img), "--mask_file", str(tmp_path / "c.npy"),
        "--nuclear-markers", "DAPI", "--output", str(out),
    ])
    assert rc == 0
    df = pd.read_csv(out)
    assert list(df.columns) == ["label"] and df.empty
