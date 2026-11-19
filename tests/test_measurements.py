"""Unit tests for bin/utils/measurements.py: the single-owner measurement vocabulary.

Asserts that the morphology column list used by merge_quant_csvs.py,
export_geojson.py, export_spatialdata.py and generate_postprocessing_qc.py
resolves to the one canonical set, and pins measurement_key()'s exact output
("<marker>: <Compartment>: <Statistic>"; G5: a case- and space-sensitive
contract with the sibling qupath-extension-flowpath repo).
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
    """All four consumers' morphology columns equal the shared set.

    merge_quant_csvs.py (a list), export_geojson.py (a set),
    export_spatialdata.py (a tuple) and generate_postprocessing_qc.py's
    MORPHOLOGY_COLUMNS (a set), all sourced from bin/utils/measurements.py.
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
    (`f"{channel_name}: {comp}: Median"`).
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
