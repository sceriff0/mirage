"""An absent compartment must be distinguishable from a measured, signal-free one.

In `bin/quantify.py` a cell with no nuclear overlap at all gets NaN for its nuclear
Median/Mean, while a cell whose nucleus was measured and dark gets 0.0. A 0.0 fill would
read downstream as a negative call. The export layer treats NaN as missing:
`build_measurements` skips it (`if pd.notna(val)`), so no bare `NaN` reaches the JSON.

`Sum` stays `0.0`: the sum over an empty set is zero, and that is a real answer.
Only `Median` and `Mean` become NaN, because the average of nothing is not a number.
"""

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
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from quantify import compute_compartment_intensities  # noqa: E402


def test_a_cell_with_no_nuclear_overlap_reports_nan_not_zero():
    """No nuclear overlap gives NaN, not a 0.0 indistinguishable from "measured, no signal"."""
    cell_mask = np.array([[1, 1, 1], [1, 1, 1]])
    nuclei_mask = np.zeros_like(cell_mask)
    channel = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    df = compute_compartment_intensities(cell_mask, nuclei_mask, channel, "CD8")

    assert np.isnan(df["CD8: Nucleus: Median"].values[0])


def test_a_measured_but_dark_compartment_still_reports_zero():
    """The other half of the distinction -- this one IS a real zero and must stay one."""
    cell_mask = np.array([[1, 1], [1, 1]])
    nuclei_mask = np.array([[1, 1], [0, 0]])
    channel = np.zeros((2, 2), dtype=float)  # measured, and genuinely dark

    df = compute_compartment_intensities(cell_mask, nuclei_mask, channel, "CD8")

    assert df["CD8: Nucleus: Median"].values[0] == 0.0
    assert not np.isnan(df["CD8: Nucleus: Median"].values[0])


def test_a_cell_that_is_entirely_nucleus_reports_nan_cytoplasm():
    cell_mask = np.array([[2, 2], [2, 2]])
    nuclei_mask = np.array([[1, 1], [1, 1]])
    channel = np.array([[10.0, 20.0], [30.0, 40.0]])

    df = compute_compartment_intensities(cell_mask, nuclei_mask, channel, "CD8")

    assert np.isnan(df["CD8: Cytoplasm: Median"].values[0])


def test_the_whole_cell_compartment_is_never_nan():
    """Every valid label has cell pixels by construction, so Cell must always be a number."""
    cell_mask = np.array([[1, 1, 2], [1, 0, 2]])
    channel = np.array([[5.0, 7.0, 9.0], [11.0, 0.0, 13.0]])

    df = compute_compartment_intensities(cell_mask, None, channel, "CD8")

    assert not df["CD8: Cell: Median"].isna().any()


def test_expanded_mean_of_an_absent_compartment_is_nan():
    cell_mask = np.array([[1, 1], [1, 1]])
    nuclei_mask = np.zeros_like(cell_mask)
    channel = np.array([[1.0, 2.0], [3.0, 4.0]])

    df = compute_compartment_intensities(
        cell_mask, nuclei_mask, channel, "CD8", expanded=True
    )

    assert np.isnan(df["CD8: Nucleus: Mean"].values[0])


def test_expanded_sum_of_an_absent_compartment_stays_zero():
    """The sum over an empty set is zero -- that is a real answer, not a missing one."""
    cell_mask = np.array([[1, 1], [1, 1]])
    nuclei_mask = np.zeros_like(cell_mask)
    channel = np.array([[1.0, 2.0], [3.0, 4.0]])

    df = compute_compartment_intensities(
        cell_mask, nuclei_mask, channel, "CD8", expanded=True
    )

    assert df["CD8: Nucleus: Sum"].values[0] == 0.0


def test_a_nan_measurement_is_omitted_from_the_geojson_not_written_as_nan():
    """Bare NaN is invalid JSON; the export layer already drops it. Pin that it still does."""
    import export_geojson

    row = pd.Series(
        {
            "label": 1,
            "x": 10.0,
            "y": 20.0,
            "CD8: Cell: Median": 5.0,
            "CD8: Nucleus: Median": np.nan,
        }
    )

    measurements = export_geojson.build_measurements(
        row, ["CD8: Cell: Median", "CD8: Nucleus: Median"], pixel_size=0.325
    )

    names = [m["name"] for m in measurements]
    assert "CD8: Cell: Median" in names
    assert "CD8: Nucleus: Median" not in names
