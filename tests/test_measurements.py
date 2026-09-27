"""Unit tests for bin/utils/measurements.py: the single-owner measurement vocabulary.

Two things used to be declared independently (kept in sync only by a
comment) across bin/merge_quant_csvs.py, bin/export_geojson.py,
bin/export_spatialdata.py, bin/generate_postprocessing_qc.py, and
bin/quantify.py:

- the 12-entry morphology column list, in three different container types
  (list/set/tuple) and one under a different name (MORPHOLOGY_COLUMNS in
  generate_postprocessing_qc.py), with merge_quant_csvs.py's copy actually
  short by two entries (fov, cell_size);
- the measurement-key grammar "<marker>: <Compartment>: <Statistic>", built
  independently in quantify.py and parsed in export_spatialdata.py.

This test asserts all four former copies now resolve back to the same
canonical set, and pins measurement_key()'s exact output (G5: a
case-/space-sensitive contract with the sibling qupath-extension-flowpath
repo).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN = REPO_ROOT / "bin"
sys.path.insert(0, str(BIN / "utils"))

import measurements as m  # noqa: E402


def _load_bin_module(name: str):
    """Load a bin/*.py script as a module (it inserts bin/utils on sys.path itself)."""
    spec = importlib.util.spec_from_file_location(name, BIN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── MORPHOLOGY_COLS: one owner ──────────────────────────────────────────────────
def test_canonical_morphology_cols_is_12_entry_tuple():
    assert isinstance(m.MORPHOLOGY_COLS, tuple)
    assert len(m.MORPHOLOGY_COLS) == 12
    assert set(m.MORPHOLOGY_COLS) == {
        "label",
        "y",
        "x",
        "area",
        "eccentricity",
        "perimeter",
        "convex_area",
        "axis_major_length",
        "axis_minor_length",
        "solidity",
        "fov",
        "cell_size",
    }


def test_all_former_copies_resolve_to_the_canonical_set():
    """The four former independent copies must now all equal the shared set.

    merge_quant_csvs.py (a list), export_geojson.py (a set),
    export_spatialdata.py (a tuple), and generate_postprocessing_qc.py's
    MORPHOLOGY_COLUMNS (a set) were each declared by hand; merge_quant_csvs.py's
    was short by `fov` and `cell_size`. All four must now be identical sets
    sourced from bin/utils/measurements.py.
    """
    canonical = set(m.MORPHOLOGY_COLS)

    mqc = _load_bin_module("merge_quant_csvs")
    eg = _load_bin_module("export_geojson")
    esd = _load_bin_module("export_spatialdata")
    qc = _load_bin_module("generate_postprocessing_qc")

    assert set(mqc.MORPHOLOGY_COLS) == canonical
    assert set(eg.MORPHOLOGY_COLS) == canonical
    assert set(esd.MORPHOLOGY_COLS) == canonical
    assert set(qc.MORPHOLOGY_COLUMNS) == canonical

    # Container semantics preserved at each call site.
    assert isinstance(mqc.MORPHOLOGY_COLS, list)
    assert isinstance(eg.MORPHOLOGY_COLS, set)
    assert isinstance(esd.MORPHOLOGY_COLS, tuple)
    assert isinstance(qc.MORPHOLOGY_COLUMNS, set)


def test_quantify_compartment_names_matches_canonical_compartments():
    quantify = _load_bin_module("quantify")
    assert quantify.COMPARTMENT_NAMES == m.COMPARTMENTS


# ── measurement_key(): the G5 contract ──────────────────────────────────────────
def test_measurement_key_exact_literal_string():
    """G5: exact spacing and case, pinned as a literal.

    This is the format qupath-extension-flowpath parses from GeoJSON
    measurement names ("marker: Compartment: Statistic"), reproduced from
    quantify.py::compute_compartment_intensities
    (`f"{channel_name}: {comp}: Median"`, bin/quantify.py:180).
    """
    assert m.measurement_key("CD3", "Nucleus", "Median") == "CD3: Nucleus: Median"


def test_measurement_key_matches_quantify_producer_output():
    """The shared builder must reproduce quantify.py's own key exactly."""
    import numpy as np

    quantify = _load_bin_module("quantify")

    cell_mask = np.array([[1, 1], [2, 2]], dtype=np.int32)
    channel = np.array([[10.0, 20.0], [30.0, 40.0]])
    df = quantify.compute_compartment_intensities(
        cell_mask, None, channel, "CD3", expanded=False
    )
    expected_key = m.measurement_key("CD3", "Cell", "Median")
    assert expected_key in df.columns
    assert expected_key == "CD3: Cell: Median"


def test_measurement_key_matches_export_spatialdata_parser():
    esd = _load_bin_module("export_spatialdata")
    key = m.measurement_key("PanCK", "Cytoplasm", "Sum")
    assert esd.parse_measurement_key(key) == ("PanCK", "Cytoplasm", "Sum")


# ── identify_marker_columns(): the shared predicate ─────────────────────────────
def test_identify_marker_columns_excludes_morphology_and_non_numeric():
    df = pd.DataFrame(
        {
            "label": [1, 2],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "fov": ["p1", "p1"],
            "cell_size": [10, 20],
            "CD3: Cell: Median": [1.5, 2.5],
            "DAPI": [7.0, 8.0],
            "some_text": ["a", "b"],
        }
    )
    assert m.identify_marker_columns(df) == ["CD3: Cell: Median", "DAPI"]


# ── QC: and MORPH: measurement keywords (2026-09-27) ──────────────────────────
import pytest  # noqa: E402, F401
from measurements import (  # noqa: E402
    MORPH_EXPORT,
    MORPH_PREFIX,
    QC_NUCLEAR_RETENTION,
    QC_PREFIX,
    QC_REG_DICE,
    QC_REG_DISPLACEMENT,
    QC_TOTAL_INTENSITY,
    identify_marker_columns,
    is_qc_column,
    morph_key,
    parse_qc_key,
    qc_key,
)


def test_cell_level_qc_key():
    assert qc_key(QC_TOTAL_INTENSITY) == "QC: Total intensity"


def test_round_level_qc_key_sorts_markers():
    assert (
        qc_key(QC_NUCLEAR_RETENTION, ["FOXP3", "CD3", "CD8"])
        == "QC: Nuclear retention: [CD3, CD8, FOXP3]"
    )


def test_round_key_round_trips():
    key = qc_key(QC_REG_DISPLACEMENT, ["CD8", "CD3"])
    assert parse_qc_key(key) == (QC_REG_DISPLACEMENT, ["CD3", "CD8"])
    assert parse_qc_key(qc_key(QC_TOTAL_INTENSITY)) == (QC_TOTAL_INTENSITY, None)


def test_parse_rejects_non_qc_and_unknown_metrics():
    assert parse_qc_key("CD3: Cell: Median") is None
    assert parse_qc_key("QC: Mystery") is None


def test_unknown_metric_is_refused():
    with pytest.raises(ValueError):
        qc_key("Mystery")


def test_round_key_needs_markers_and_clean_names():
    with pytest.raises(ValueError):
        qc_key(QC_REG_DICE, [])
    with pytest.raises(ValueError):
        qc_key(QC_REG_DICE, ["CD3, CD8"])


def test_morph_key_and_export_table():
    assert morph_key("Area µm²") == "MORPH: Area µm²"
    assert [c for c, _, _ in MORPH_EXPORT] == [
        "area", "eccentricity", "perimeter", "solidity",
        "convex_area", "axis_major_length", "axis_minor_length",
    ]
    assert MORPH_PREFIX == "MORPH: " and QC_PREFIX == "QC: "


def test_qc_columns_are_not_markers():
    df = pd.DataFrame(
        {"label": [1], "x": [1.0], "CD3: Cell: Median": [2.0], "QC: Total intensity": [5.0]}
    )
    assert is_qc_column("QC: Total intensity")
    assert not is_qc_column("CD3: Cell: Median")
    assert identify_marker_columns(df) == ["CD3: Cell: Median"]
