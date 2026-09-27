"""A wrong pixel size does not just get carried through the pipeline -- it corrupts
every reported measurement.

`tests/test_pixel_size_is_actually_passed.py` proves the value ARRIVES at every
`bin/*.py` consumer that needs a scale. It does not prove that a wrong value produces
wrong numbers -- that is a different claim, and it is the one that makes the
PREFLIGHT_SCALE warning (see `bin/preflight_scale.py`) load-bearing rather than
decorative. Without a test like this one, someone could weaken that warning to a debug
line and nothing here would object.

Exercises the REAL production chain rather than reimplementing area math:

  1. ``bin/extract_cell_properties.py::extract_morphology`` -- regionprops on a
     synthetic mask, producing ``area`` in raw pixels (no scale involved yet).
  2. ``bin/export_geojson.py::build_measurements`` -- the function that actually
     converts pixel-space morphology into the µm measurements written into
     ``cells.geojson`` (and read by qupath-extension-flowpath). This is where
     ``pixel_size`` is applied, and applied squared for an area.

Area scales with the SQUARE of the pixel size (doubling the scale quadruples the
reported area), not linearly -- a linear result here would mean the assertion is
reading the wrong measurement, not that the relationship is actually linear.

The synthetic cell is 200x200 px (area = 40000 px, a whole number under a 0.325/0.650
µm scale) so the doubled-vs-quadrupled comparison is exact and is not confounded by
``build_measurements``'s own `round(..., 3)` display rounding -- that rounding is real
production behaviour, just not the thing this test is checking.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
    ),
)

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("skimage")

import export_geojson as eg  # noqa: E402
import extract_cell_properties as ecp  # noqa: E402


def _one_square_cell(side_px: int):
    """A single labeled square cell, side ``side_px``, on a background twice as large."""
    mask = np.zeros((side_px * 2, side_px * 2), dtype=np.uint32)
    mask[0:side_px, 0:side_px] = 1
    return mask


def _measure_area_um2(mask, pixel_size: float) -> float:
    """Run the mask through the REAL production morphology + µm-conversion chain and
    return the reported "MORPH: Area µm²" measurement for the one cell in ``mask``.
    """
    props_df, _mask_out, valid_labels = ecp.extract_morphology(mask)
    assert props_df is not None and len(valid_labels) == 1, (
        "expected exactly one labeled cell in the synthetic mask"
    )
    row = props_df.loc[valid_labels[0]]

    measurements = eg.build_measurements(row, marker_cols=[], pixel_size=pixel_size)
    area_entries = [m["value"] for m in measurements if m["name"] == "MORPH: Area µm²"]
    assert area_entries, "build_measurements did not emit a MORPH: Area µm² measurement"
    return area_entries[0]


def test_area_scales_with_the_square_of_the_pixel_size():
    """A wrong pixel size silently rescales every area. This is the failure the
    PREFLIGHT_SCALE warning exists to make visible, so it must be demonstrable."""
    mask = _one_square_cell(side_px=200)
    a1 = _measure_area_um2(mask, pixel_size=0.325)
    a2 = _measure_area_um2(mask, pixel_size=0.650)
    assert a2 == pytest.approx(a1 * 4.0, rel=1e-6), (
        "doubling the pixel size must quadruple the reported area; if it does not, "
        "the scale is not reaching the measurement and the PREFLIGHT_SCALE warning "
        "is guarding nothing"
    )
