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

from typing import List, Optional, Sequence, Tuple

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

# ── Non-marker measurement keywords (docs/outputs.md, "Per-cell QC") ───────────
# Every non-marker, non-identity measurement in the per-patient table and in
# cells.geojson starts with exactly one of these. FlowPath classifies on the prefix
# alone, so a key without one is read as a marker.
QC_PREFIX = "QC: "
MORPH_PREFIX = "MORPH: "

QC_TOTAL_INTENSITY = "Total intensity"
QC_NUCLEAR_RETENTION = "Nuclear retention"
QC_REG_DISPLACEMENT = "Registration displacement µm"
QC_REG_DICE = "Registration Dice"

QC_CELL_METRICS: tuple = (QC_TOTAL_INTENSITY,)
QC_ROUND_METRICS: tuple = (QC_NUCLEAR_RETENTION, QC_REG_DISPLACEMENT, QC_REG_DICE)

# (morphology CSV column, GeoJSON display name, unit power: 0 none, 1 µm, 2 µm²),
# in export_geojson's output order.
MORPH_EXPORT: tuple = (
    ("area", "Area µm²", 2),
    ("eccentricity", "Eccentricity", 0),
    ("perimeter", "Perimeter µm", 1),
    ("solidity", "Solidity", 0),
    ("convex_area", "Convex Area µm²", 2),
    ("axis_major_length", "Major Axis Length µm", 1),
    ("axis_minor_length", "Minor Axis Length µm", 1),
)

_FORBIDDEN_IN_MARKER = ("[", "]", ",", ": ")


def qc_key(metric: str, markers: Optional[Sequence[str]] = None) -> str:
    """``"QC: <metric>"`` (cell-level) or ``"QC: <metric>: [a, b]"`` (round-level).

    Round markers are sorted so the key is a function of the round, not of file order.
    """
    if markers is None:
        if metric not in QC_CELL_METRICS:
            raise ValueError(f"not a cell-level QC metric: {metric!r}")
        return f"{QC_PREFIX}{metric}"
    if metric not in QC_ROUND_METRICS:
        raise ValueError(f"not a round-level QC metric: {metric!r}")
    names = sorted(str(m) for m in markers)
    if not names:
        raise ValueError(f"round-level QC key {metric!r} needs at least one marker")
    for m in names:
        if not m or any(tok in m for tok in _FORBIDDEN_IN_MARKER):
            raise ValueError(f"marker name {m!r} cannot appear in a round QC key")
    return f"{QC_PREFIX}{metric}: [{', '.join(names)}]"


def parse_qc_key(key: str) -> Optional[Tuple[str, Optional[List[str]]]]:
    """Inverse of :func:`qc_key`; ``None`` for anything that is not a known QC key."""
    if not isinstance(key, str) or not key.startswith(QC_PREFIX):
        return None
    rest = key[len(QC_PREFIX) :]
    if rest in QC_CELL_METRICS:
        return rest, None
    metric, sep, bracket = rest.partition(": [")
    if not sep or not bracket.endswith("]") or metric not in QC_ROUND_METRICS:
        return None
    markers = [m for m in bracket[:-1].split(", ") if m]
    return (metric, markers) if markers else None


def morph_key(display_name: str) -> str:
    """Build a morphology key: ``"MORPH: <display_name>"``.

    Parameters
    ----------
    display_name : str
        A morphology display name, e.g. one of ``MORPH_EXPORT``'s second elements.

    Returns
    -------
    str
    """
    return f"{MORPH_PREFIX}{display_name}"


def is_qc_column(col) -> bool:
    """True when ``col`` is a ``"QC: ..."`` measurement key.

    Parameters
    ----------
    col
        A candidate column name; any type is accepted, only a ``str`` starting
        with ``QC_PREFIX`` can be ``True``.

    Returns
    -------
    bool
    """
    return isinstance(col, str) and col.startswith(QC_PREFIX)


def measurement_key(marker: str, compartment: str, statistic: str) -> str:
    """Build a measurement key: ``"<marker>: <Compartment>: <Statistic>"``.

    Exact spacing and case are a contract with qupath-extension-flowpath. Matches what
    ``quantify.py::compute_compartment_intensities`` writes and
    ``export_spatialdata.py::parse_measurement_key`` parses.
    """
    return f"{marker}: {compartment}: {statistic}"


def identify_marker_columns(df: "pd.DataFrame") -> List[str]:
    """Numeric columns that are marker measurements, not morphology, metadata or QC.

    "Not in MORPHOLOGY_COLS, not QC, numeric dtype", in ``df.columns`` order.
    """
    return [
        col
        for col in df.columns
        if col not in MORPHOLOGY_COLS
        and not is_qc_column(col)
        and pd.api.types.is_numeric_dtype(df[col])
    ]
