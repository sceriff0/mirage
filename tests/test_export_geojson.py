"""Tests for bin/export_geojson.py's exported GeoJSON contract.

This module is the contract seam to the sibling FlowPath QuPath extension, so what
`export_geojson()` writes into each Feature is pinned here rather than left to the
`real`-tagged nf-tests (which CI's `--tag stub` gate never runs).

Recovered from tests/test_export_geojson_phenotype.py, which was deleted with the
panel-phenotyping extraction (see CHANGELOG's [Unreleased] Removed entry). Only the
"phenotyping is off" case survived that removal, and it is the case that now describes
the module's whole behaviour: constant "Cell" classification, no stamped feature `id`,
no phenotype measurements.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "export_geojson",
    Path(__file__).resolve().parent.parent / "bin" / "export_geojson.py",
)
eg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eg)


def test_no_panel_supplied_is_constant_cell_classification(tmp_path):
    """Constant "Cell", no id, no phenotype extras.

    The absent `id` is load-bearing: export_geojson() passes object_id=None so that
    features are NOT all stamped with the constant "PathDetectionObject", which QuPath
    treats as a per-object UUID — a shared id across every detection breaks re-import
    for the whole cohort.
    """
    quant = pd.DataFrame(
        {
            "label": [1, 2],
            "x": [5, 6],
            "y": [7, 8],
            "PanCK: Cytoplasm: Mean": [10.0, 0.0],
            "area": [100, 100],
        }
    )
    out = tmp_path / "cells.geojson"
    eg.export_geojson(
        quant,
        str(out),
        pixel_size=0.325,
        marker_cols=["PanCK: Cytoplasm: Mean"],
        contours=None,
    )
    gj = json.loads(out.read_text())
    f0 = gj["features"][0]
    assert "id" not in f0
    assert f0["properties"]["classification"] == {
        "name": "Cell",
        "colorRGB": eg.rgb_to_qupath_color(*eg.CELL_COLOR_RGB),
    }
    names = {m["name"] for m in f0["properties"]["measurements"]}
    assert "PanCK: Cytoplasm: Mean" in names  # the marker key is still emitted
    assert "pheno_score: Tumour" not in names
    assert not any(n.startswith("pheno_score:") for n in names)


def test_label_is_emitted_as_a_measurement(tmp_path):
    """`label` rides along as a measurement, not just as the contour lookup key.

    It is the only join key a QuPath-side consumer can use to get back to
    cells_data.csv. FlowPath resolves a measurement named "label"
    (CellIndex.resolveMeasurementKey) and filters it out of the marker panel
    (DetectionIngest.MORPHOLOGY_PREFIXES), so emitting it costs nothing there.
    """
    quant = pd.DataFrame(
        {
            "label": [7, 9],
            "x": [5, 6],
            "y": [7, 8],
            "PanCK: Cytoplasm: Mean": [10.0, 0.0],
            "area": [100, 100],
        }
    )
    out = tmp_path / "cells.geojson"
    eg.export_geojson(
        quant,
        str(out),
        pixel_size=0.325,
        marker_cols=["PanCK: Cytoplasm: Mean"],
        contours=None,
    )
    features = json.loads(out.read_text())["features"]
    labels = [
        next(
            m["value"] for m in f["properties"]["measurements"] if m["name"] == "label"
        )
        for f in features
    ]
    assert labels == [7, 9]


def test_label_survives_a_skipped_row(tmp_path):
    """The case positional CSV<->feature joining gets wrong.

    A NaN centroid keeps the row in cells_data.csv but drops the feature, so feature
    index and CSV row index diverge. The emitted label is what stays correct: the
    second feature must be label 3, not the label of CSV row 1.
    """
    quant = pd.DataFrame(
        {
            "label": [1, 2, 3],
            "x": [5, float("nan"), 6],
            "y": [7, 8, 9],
            "area": [100, 100, 100],
        }
    )
    out = tmp_path / "cells.geojson"
    n = eg.export_geojson(
        quant, str(out), pixel_size=0.325, marker_cols=[], contours=None
    )
    features = json.loads(out.read_text())["features"]
    assert n == 2 and len(features) == 2
    labels = [
        next(
            m["value"] for m in f["properties"]["measurements"] if m["name"] == "label"
        )
        for f in features
    ]
    assert labels == [1, 3]


# ---------------------------------------------------------------------------
# Both feature loops taken off df.iterrows()
# ---------------------------------------------------------------------------
#
# `_iter_features` (inside export_geojson()) and export_combined_geojson() each drove a
# loop off `df.iterrows()`, which materialises one pandas Series per row -- 500k of them
# on a whole-slide image. Both now pull the columns they read to numpy once
# (`_iter_rows_positional`) and index positionally, following the precedent
# `extract_cell_properties._label_bboxes` set for bboxes (tests/test_no_full_mask_unique.py).
#
# This block also covers a crash fix: export_combined_geojson() incremented `skipped[0]`
# on a plain int (`skipped = 0`), which raised `TypeError: 'int' object is not
# subscriptable` on the first row with a NaN x/y centroid -- an input the function is
# meant to handle (a cell with missing raw data keeps a NaN centroid rather than being
# dropped upstream), not an edge case.


def _nan_safe_eq(a, b) -> bool:
    """Value equality where NaN == NaN, matching how `pd.isna`/`pd.notna` treat these rows."""
    a_is_nan = isinstance(a, float) and math.isnan(a)
    b_is_nan = isinstance(b, float) and math.isnan(b)
    if a_is_nan or b_is_nan:
        return a_is_nan and b_is_nan
    return a == b


def test_iter_rows_positional_matches_iterrows_exactly():
    """Seam test: `_iter_rows_positional` vs the `df.iterrows()` it replaces.

    Covers all three named traps: a NaN centroid coordinate, a NaN `label`, and an
    optional column ("area") that is entirely absent from the frame.
    """
    df = pd.DataFrame(
        {
            "label": [1, float("nan"), 3],
            "x": [10.0, float("nan"), 30.0],
            "y": [20.0, 25.0, float("nan")],
            "CD3: Cell: Mean": [1.0, 2.0, 3.0],
        }
        # "area" is deliberately absent -- exercises the missing-optional-column path.
    )
    field_cols = ["label", "x", "y", "area", "CD3: Cell: Mean"]

    expected = [
        (
            idx,
            row.get("x"),
            row.get("y"),
            {c: row.get(c) for c in field_cols if c in df.columns},
        )
        for idx, row in df.iterrows()
    ]

    got = list(eg._iter_rows_positional(df, field_cols))

    assert len(got) == len(expected)
    for (g_idx, g_x, g_y, g_row), (e_idx, e_x, e_y, e_row) in zip(got, expected):
        assert g_idx == e_idx
        assert _nan_safe_eq(g_x, e_x)
        assert _nan_safe_eq(g_y, e_y)
        assert set(g_row.keys()) == set(e_row.keys())
        for key in e_row:
            assert _nan_safe_eq(g_row[key], e_row[key]), key


def test_iter_rows_positional_returns_none_for_a_column_absent_from_the_frame():
    """Trap 1: `row.get("x")` is None for an absent *column*, not just an absent value.

    Positional numpy indexing on a column that doesn't exist would raise instead --
    _iter_rows_positional must not do that.
    """
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})

    for _idx, _x, _y, row in eg._iter_rows_positional(df, ["x", "y", "convex_area"]):
        assert "convex_area" not in row
        assert row.get("convex_area") is None


def test_iter_rows_positional_preserves_a_non_contiguous_index():
    """A caller upstream (main()'s drop_duplicates) can leave a non-contiguous index;
    the fallback `row.get("label", idx)` must see the real index value, not a fresh
    position counter."""
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}, index=[0, 5])

    idxs = [idx for idx, _x, _y, _row in eg._iter_rows_positional(df, ["x", "y"])]

    assert idxs == [0, 5]


# ---- byte-identity: the binding G1 evidence -------------------------------------------
#
# These reference functions are a verbatim copy of the pre-change, iterrows-driven loop
# bodies (the code this task replaces). They are kept ONLY to prove byte-for-byte output
# equality with the new numpy-positional loops on NaN-free input.
#
# NOTE on scope: on a frame with a NaN centroid, the pre-fix export_combined_geojson()
# crashed (`skipped[0] += 1` on a plain int), so there is no "before" output to compare
# bytes against for that case. Byte-identity below is therefore asserted on NaN-free
# input only; the NaN-centroid case is covered instead by
# test_export_combined_geojson_survives_a_nan_centroid below (a regression test, not a
# byte-identity comparison). This is a deliberate gap, not an oversight.


def _reference_iter_features(df, marker_cols, pixel_size, contours):
    color_int = eg.rgb_to_qupath_color(*eg.CELL_COLOR_RGB)
    for idx, row in df.iterrows():
        x_px = row.get("x")
        y_px = row.get("y")
        if pd.isna(x_px) or pd.isna(y_px):
            continue
        x_px_corner = float(x_px) + 0.5
        y_px_corner = float(y_px) + 0.5
        cell_id = row.get("label", idx)
        cell_label_str = str(int(cell_id)) if pd.notna(cell_id) else str(idx)
        geometry = eg._polygon_geometry(contours, cell_label_str) or {
            "type": "Point",
            "coordinates": [x_px_corner, y_px_corner],
        }
        measurements = eg.build_measurements(row, marker_cols, pixel_size)
        yield eg.build_feature(measurements, geometry, color_int, object_id=None)


def _reference_export_geojson(df, output_path, pixel_size, marker_cols, contours):
    eg._stream_collection(
        _reference_iter_features(df, marker_cols, pixel_size, contours), output_path
    )


def _reference_export_combined_geojson(
    df,
    output_dir,
    pixel_size,
    marker_cols,
    cell_contours,
    nucleus_contours,
    prefix="cells",
):
    color_int = eg.rgb_to_qupath_color(*eg.CELL_COLOR_RGB)
    cells_combined = []
    for idx, row in df.iterrows():
        x_px = row.get("x")
        y_px = row.get("y")
        if pd.isna(x_px) or pd.isna(y_px):
            # Pre-fix, this branch crashed (`skipped[0] += 1` on a plain int). Unreachable
            # here: this reference is only ever fed NaN-free centroids (see the docstring
            # above the byte-identity tests for why).
            continue
        x_corner = float(x_px) + 0.5
        y_corner = float(y_px) + 0.5
        cell_id = row.get("label", idx)
        label_str = str(int(cell_id)) if pd.notna(cell_id) else str(idx)
        cell_geom = eg._polygon_geometry(cell_contours, label_str) or {
            "type": "Point",
            "coordinates": [x_corner, y_corner],
        }
        nucleus_geom = eg._polygon_geometry(nucleus_contours, label_str)
        measurements = eg.build_measurements(row, marker_cols, pixel_size)
        cells_combined.append(
            eg.build_feature(
                measurements,
                cell_geom,
                color_int,
                object_type="cell",
                nucleus_geometry=nucleus_geom,
                object_id=None,
            )
        )
    out = Path(output_dir)
    combined_path = str(out / f"{prefix}.geojson")
    eg._write_collection(cells_combined, combined_path)
    wholecell = [
        {k: v for k, v in feat.items() if k != "nucleusGeometry"}
        for feat in cells_combined
    ]
    wholecell_path = str(out / f"{prefix}_wholecell.geojson")
    eg._write_collection(wholecell, wholecell_path)


def _nan_free_df():
    """A frame with NaN in a marker value (a legitimate, exercised path) but never in a
    centroid -- centroid NaN is the one case old and new code disagree on (see above).

    Carries all seven morphology columns build_measurements() reads (not just three),
    so a byte-identity failure here would actually catch a column dropped from -- or
    added to -- the set _iter_rows_positional derives from MORPHOLOGY_COLS.
    """
    return pd.DataFrame(
        {
            "label": [7, 9, 3, 12],
            "x": [10.0, 20.5, 0.0, 33.25],
            "y": [11.0, 5.5, 100.0, 7.75],
            "area": [100.0, 250.5, 12.0, 88.0],
            "eccentricity": [0.1, 0.9, 0.5, float("nan")],
            "perimeter": [40.0, 60.5, 20.0, 33.3],
            "solidity": [0.95, 0.80, 1.0, 0.6],
            "convex_area": [105.0, 260.0, 12.0, 90.5],
            "axis_major_length": [12.5, 18.0, 4.0, 10.75],
            "axis_minor_length": [8.5, 9.0, 3.5, float("nan")],
            "CD3: Cell: Mean": [1.234567, float("nan"), 0.0, 9.99995],
            "CD8: Nucleus: Median": [5.5, 6.6, float("nan"), 3.14159],
        }
    )


def test_export_geojson_is_byte_identical_to_iterrows_on_nan_free_input(tmp_path):
    """G1: emitted GeoJSON must be byte-identical before and after, on NaN-free input.

    See the module-level note above: a NaN centroid is excluded here on purpose because
    the pre-fix code crashed on it in export_combined_geojson() (not this function, but
    the byte-identity scope is documented once, above, for both).
    """
    df = _nan_free_df()
    marker_cols = ["CD3: Cell: Mean", "CD8: Nucleus: Median"]
    contours = {"7": [[9, 10], [11, 10], [11, 12], [9, 12], [9, 10]]}

    old_path = tmp_path / "old.geojson"
    new_path = tmp_path / "new.geojson"
    _reference_export_geojson(df, str(old_path), 0.325, marker_cols, contours)
    eg.export_geojson(
        df, str(new_path), pixel_size=0.325, marker_cols=marker_cols, contours=contours
    )

    assert new_path.read_bytes() == old_path.read_bytes()
    assert len(old_path.read_bytes()) > 0


def test_export_combined_geojson_is_byte_identical_to_iterrows_on_nan_free_input(
    tmp_path,
):
    """G1, for export_combined_geojson(): same byte-identity requirement, same NaN-free
    scope note as the export_geojson() case above."""
    df = _nan_free_df()
    marker_cols = ["CD3: Cell: Mean", "CD8: Nucleus: Median"]
    cell_contours = {"7": [[9, 10], [11, 10], [11, 12], [9, 12], [9, 10]]}
    nucleus_contours = {
        "7": [[9.5, 10.5], [10.5, 10.5], [10.5, 11.5], [9.5, 11.5], [9.5, 10.5]]
    }

    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    _reference_export_combined_geojson(
        df, str(old_dir), 0.325, marker_cols, cell_contours, nucleus_contours
    )
    eg.export_combined_geojson(
        df,
        str(new_dir),
        pixel_size=0.325,
        marker_cols=marker_cols,
        cell_contours=cell_contours,
        nucleus_contours=nucleus_contours,
    )

    for name in ("cells.geojson", "cells_wholecell.geojson"):
        old_bytes = (old_dir / name).read_bytes()
        new_bytes = (new_dir / name).read_bytes()
        assert new_bytes == old_bytes
        assert len(old_bytes) > 0


# ---- the NaN-centroid regression test (not a byte-identity comparison) ----------------


def test_export_combined_geojson_survives_a_nan_centroid(tmp_path, caplog):
    """Regression test for the `skipped[0] += 1` crash on a plain `skipped = 0` int.

    Pre-fix, this call raised `TypeError: 'int' object is not subscriptable` on the
    first row with a NaN x/y centroid. There is no "before" output to byte-compare
    against for this input (the old code never produced any -- see the byte-identity
    tests' module note); this test instead pins the fixed behaviour directly.
    """
    df = pd.DataFrame(
        {
            "label": [1, 2, 3],
            "x": [10.0, np.nan, 30.0],
            "y": [20.0, 25.0, 35.0],
            "CD3: Cell: Mean": [1.0, 2.0, 3.0],
        }
    )

    with caplog.at_level(logging.INFO, logger="export_geojson"):
        counts = eg.export_combined_geojson(
            df,
            str(tmp_path),
            pixel_size=0.325,
            marker_cols=["CD3: Cell: Mean"],
            cell_contours=None,
            nucleus_contours=None,
        )

    assert counts["cells"] == 2
    assert counts["cells_wholecell"] == 2

    features = json.loads((tmp_path / "cells.geojson").read_text())["features"]
    assert len(features) == 2
    labels = [
        next(
            m["value"] for m in f["properties"]["measurements"] if m["name"] == "label"
        )
        for f in features
    ]
    assert labels == [1, 3]

    assert any("skipped 1" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# QC: passthrough and MORPH: prefixing (2026-09-27 spec)
# ---------------------------------------------------------------------------


def test_measurements_carry_qc_and_prefixed_morphology():
    row = {
        "label": 3,
        "x": 1.0,
        "y": 1.0,
        "CD3: Cell: Median": 2.0,
        "QC: Total intensity": 7.0,
        "QC: Registration Dice: [CD3]": float("nan"),
        "area": 4.0,
        "eccentricity": 0.5,
    }
    ms = eg.build_measurements(
        row, ["CD3: Cell: Median"], 0.5,
        qc_cols=["QC: Total intensity", "QC: Registration Dice: [CD3]"],
    )
    names = [m["name"] for m in ms]
    assert "QC: Total intensity" in names
    assert "QC: Registration Dice: [CD3]" not in names          # NaN omitted
    assert "MORPH: Area µm²" in names and "Area µm²" not in names
    assert names.index("CD3: Cell: Median") < names.index("QC: Total intensity") < names.index("MORPH: Area µm²")
    assert next(m["value"] for m in ms if m["name"] == "MORPH: Area µm²") == 1.0
