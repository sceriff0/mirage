"""The export fixtures must produce a feature per cell, not an empty collection.

The export_geojson nf-test's real case reads tests/testdata/sample_merged_quant.csv and
tests/testdata/sample_contours.json. Their shapes must match what the producers write:

  * the CSV carries `x`/`y` (bin/export_geojson._iter_rows_positional skips any row
    whose centroid is missing, so `centroid_x`/`centroid_y` gives 0 features);
  * the contours map str(label) -> [[x, y], ...], as
    bin/extract_cell_properties.extract_contours returns and _polygon_geometry reads.

That nf-test is tagged `real` and runs only nightly on the default branch; this runs in
the blocking suite in about a second.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "tests" / "testdata"
SCRIPT = ROOT / "bin" / "export_geojson.py"
CELL_CSV = TESTDATA / "sample_merged_quant.csv"
CONTOURS = TESTDATA / "sample_contours.json"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Run the real EXPORT_GEOJSON script over the real fixtures, once."""
    for path in (CELL_CSV, CONTOURS):
        if not path.is_file():
            pytest.skip(
                f"{path.relative_to(ROOT)} is absent -- run "
                "`python tests/testdata/generate_complete_testdata.py` first"
            )
    out = tmp_path_factory.mktemp("geojson")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cell_data",
            str(CELL_CSV),
            "-o",
            str(out),
            "--contours_json",
            str(CONTOURS),
            "--nucleus_contours_json",
            str(CONTOURS),
            "--pixel_size",
            "0.325",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"export_geojson.py exited {proc.returncode}:\n{proc.stderr}"
    )
    return out, proc.stderr + proc.stdout


def test_every_csv_row_becomes_a_feature(exported):
    out, log = exported
    features = json.loads((out / "cells.geojson").read_text())["features"]
    rows = len((out / "cells_data.csv").read_text().strip().splitlines()) - 1
    assert rows == 20, f"the fixture should carry 20 cells, not {rows}"
    assert len(features) == rows, (
        f"{len(features)} GeoJSON feature(s) against {rows} CSV row(s).\n"
        "Every row was dropped at export_geojson.py's `pd.isna(x_px)` guard, which "
        "means the quantification fixture does not carry the 'x'/'y' columns "
        "_iter_rows_positional looks up. The producer's column order is "
        "bin/merge_quant_csvs.reorder_columns' and is written down in "
        "tests/testdata/expected/merged_quant_columns.txt.\n\n" + log
    )


def test_the_markers_are_the_markers(exported):
    """A morphology column mistaken for a marker is the same defect seen from the
    other side: with `centroid_x` in the frame, the script reported SEVEN marker
    channels for a three-marker panel."""
    _, log = exported
    assert "Detected 3 marker channels: ['DAPI', 'PANCK', 'SMA']" in log, (
        "export_geojson.py did not resolve exactly the three markers of the fixture "
        "panel; a morphology column is being treated as marker signal.\n\n" + log
    )


def test_each_feature_carries_a_closed_polygon_ring(exported):
    """The contours fixture must be a bare ring, as
    bin/extract_cell_properties.extract_contours writes it -- not a
    {"coordinates": ...} wrapper, which yields a one-element 'ring' that is a dict."""
    out, log = exported
    features = json.loads((out / "cells.geojson").read_text())["features"]
    assert features, (
        "no features to inspect -- see test_every_csv_row_becomes_a_feature"
    )
    for i, feat in enumerate(features):
        geom = feat["geometry"]
        assert geom["type"] == "Polygon", (
            f"feature {i} has geometry {geom['type']}, not Polygon: the contour lookup "
            f"missed and export_geojson fell back to a Point.\n\n{log}"
        )
        ring = geom["coordinates"][0]
        assert isinstance(ring, list) and len(ring) >= 4, (
            f"feature {i}'s ring is {type(ring).__name__} of length "
            f"{len(ring) if hasattr(ring, '__len__') else 'n/a'}; a GeoJSON polygon "
            "ring is a list of at least 4 [x, y] pairs (3 unique + the closing point)."
        )
        assert all(isinstance(pt, list) and len(pt) == 2 for pt in ring), (
            f"feature {i}'s ring is not a list of [x, y] pairs: {ring[:2]}"
        )


def test_every_feature_has_a_nucleus_geometry(exported):
    """The combined export's whole point: cell polygon as `geometry`, nucleus polygon
    as top-level `nucleusGeometry`."""
    out, log = exported
    features = json.loads((out / "cells.geojson").read_text())["features"]
    assert features, (
        "no features to inspect -- the export produced 0 features (the same "
        "0-features regression test_every_csv_row_becomes_a_feature catches), so "
        "the nucleusGeometry check below would pass vacuously over an empty "
        f"list.\n\n{log}"
    )
    without = [i for i, f in enumerate(features) if not f.get("nucleusGeometry")]
    assert not without, (
        f"{len(without)} feature(s) carry no nucleusGeometry: {without[:5]}"
    )
