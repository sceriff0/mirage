"""Measurement vocabulary shared by the postprocessing scripts.

- ``MORPHOLOGY_COLS``: merged-CSV columns that describe cell geometry/identity, not markers.
- The key grammar ``"<marker>: <Compartment>: <Statistic>"``, produced by ``quantify.py``
  and read by ``export_geojson.py`` and ``export_spatialdata.py``.

The key format is read by qupath-extension-flowpath and is case- and space-sensitive: do
not change ``measurement_key()``, ``COMPARTMENTS`` or ``STATISTICS`` without a matching
change there.

Imported flat (``from measurements import ...``) after ``bin/utils`` is put on sys.path.
"""

from __future__ import annotations

from typing import List

import pandas as pd

# Morphology/identity columns (not marker intensities), in producer order. Callers wrap
# it as needed, e.g. set(MORPHOLOGY_COLS) for membership tests.
MORPHOLOGY_COLS: tuple = (
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
)

# The measurement-key grammar shared with QuPath/FlowPath:
# "<marker>: <Compartment>: <Statistic>".
COMPARTMENTS: tuple = ("Nucleus", "Cytoplasm", "Cell")
STATISTICS: tuple = ("Median", "Mean", "Sum")


def measurement_key(marker: str, compartment: str, statistic: str) -> str:
    """Build a measurement key: ``"<marker>: <Compartment>: <Statistic>"``.

    Exact spacing and case are a contract with qupath-extension-flowpath. Matches what
    ``quantify.py::compute_compartment_intensities`` writes and
    ``export_spatialdata.py::parse_measurement_key`` parses.
    """
    return f"{marker}: {compartment}: {statistic}"


def identify_marker_columns(df: "pd.DataFrame") -> List[str]:
    """Numeric columns that are marker measurements, not morphology/metadata.

    Shared predicate: "not in MORPHOLOGY_COLS and numeric dtype", in
    ``df.columns`` order. Formerly duplicated verbatim in
    ``export_geojson.py`` and ``export_spatialdata.py``.
    """
    return [
        col
        for col in df.columns
        if col not in MORPHOLOGY_COLS and pd.api.types.is_numeric_dtype(df[col])
    ]
