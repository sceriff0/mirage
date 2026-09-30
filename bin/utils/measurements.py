"""Canonical measurement vocabulary shared across the postprocessing scripts.

Single source of truth for two things that used to be declared independently
in several `bin/*.py` scripts, kept in sync only by a comment:

- ``MORPHOLOGY_COLS``: the columns in the merged quantification CSV that
  describe cell geometry/identity rather than marker signal.
- The measurement-key grammar ``"<marker>: <Compartment>: <Statistic>"``
  produced by ``quantify.py`` and consumed by ``export_geojson.py`` and
  ``export_spatialdata.py``.

G5 contract: the measurement-key format is consumed by the sibling repo
``qupath-extension-flowpath`` and is case- and space-sensitive. Do not change
``measurement_key()``'s output format, or the ``COMPARTMENTS``/``STATISTICS``
vocabularies, without a coordinated change on that side.

Import convention: this module is imported flat (``from measurements import
...``) by scripts that do ``sys.path.insert(0, .../bin/utils)`` first,
matching the existing convention in e.g. ``bin/export_geojson.py``. It is
pure-stdlib plus ``pandas``, which every consumer already depends on.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pandas as pd

# Canonical morphology/metadata columns (geometry + identity), in producer
# order. Every column here is NOT a marker intensity. This is the 12-entry
# list; former copies of this list disagreed on count (10 vs 12) and
# container type (list/set/tuple) -- callers wrap this tuple in the
# container their own local semantics need (e.g. `set(MORPHOLOGY_COLS)` for
# membership testing in a loop).
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
# in the order export_geojson has always written them.
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

    G5 contract with qupath-extension-flowpath: exact spacing and case.
    Reproduces the format independently built by
    ``quantify.py::compute_compartment_intensities`` (the producer) and parsed
    by ``export_spatialdata.py::parse_measurement_key``.
    """
    return f"{marker}: {compartment}: {statistic}"


def identify_marker_columns(df: "pd.DataFrame") -> List[str]:
    """Numeric columns that are marker measurements, not morphology, metadata or QC.

    Shared predicate: "not in MORPHOLOGY_COLS and not QC and numeric dtype", in
    ``df.columns`` order. Formerly duplicated verbatim in
    ``export_geojson.py`` and ``export_spatialdata.py``.
    """
    return [
        col
        for col in df.columns
        if col not in MORPHOLOGY_COLS
        and not is_qc_column(col)
        and pd.api.types.is_numeric_dtype(df[col])
    ]
