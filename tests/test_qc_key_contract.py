"""Every GeoJSON measurement is identity, a marker key, a QC: key or a MORPH: key."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))
sys.path.insert(0, str(BIN / "utils"))

import export_geojson  # noqa: E402
from measurements import MORPH_PREFIX, QC_PREFIX, parse_qc_key  # noqa: E402

IDENTITY = {"label", "Centroid X µm", "Centroid Y µm"}


def test_no_unprefixed_non_marker_key(tmp_path):
    df = pd.DataFrame(
        {
            "label": [1],
            "x": [5.0],
            "y": [5.0],
            "area": [10.0],
            "eccentricity": [0.1],
            "perimeter": [12.0],
            "solidity": [0.9],
            "convex_area": [11.0],
            "axis_major_length": [4.0],
            "axis_minor_length": [3.0],
            "CD3: Cell: Median": [2.0],
            "CD3": [2.1],
            "QC: Total intensity": [2.0],
            "QC: Nuclear retention: [CD3]": [0.9],
        }
    )
    out = tmp_path / "c.geojson"
    contours = {"1": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]}
    export_geojson.export_geojson(df, str(out), pixel_size=1.0, contours=contours)
    names = {
        m["name"]
        for f in json.loads(out.read_text())["features"]
        for m in f["properties"]["measurements"]
    }
    markers = {"CD3: Cell: Median", "CD3"}
    for n in names - IDENTITY - markers:
        assert n.startswith((QC_PREFIX, MORPH_PREFIX)), n
        if n.startswith(QC_PREFIX):
            assert parse_qc_key(n) is not None, n
