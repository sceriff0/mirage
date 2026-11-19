"""Neither quantification nor morphology should sort the whole mask to learn its label set.

`np.unique(mask)` is a comparison sort over every pixel -- O(n log n) on 1.6x10^9 values for a
40k x 40k slide -- and both callers have the answer more cheaply:

  * `extract_cell_properties.extract_morphology`: `regionprops_table` returns the label set.
  * `quantify.compute_compartment_intensities`: the non-zero bins of
    `np.bincount(flat_cell, ...)`, a single linear pass, are the label set.

Identity: `np.unique(mask)` returns the ascending distinct values; `np.nonzero(bincount)[0]`
returns the ascending indices with at least one pixel. Those are the same set in the same order,
so dropping background 0 from either gives identical arrays. `regionprops_table` likewise reports
one row per non-zero label, ascending.

These tests pin both the identity and the property, so the cheap path cannot regress into a sort.
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
pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("skimage")

import extract_cell_properties as ecp  # noqa: E402
import quantify as q  # noqa: E402

# A mask big enough that "did you sort every pixel?" is a meaningful question.
MASK_SIDE = 96


def _mask():
    rng = np.random.default_rng(20260816)
    return rng.integers(0, 8, size=(MASK_SIDE, MASK_SIDE)).astype(np.uint32)


def _expected_labels(mask):
    labels = np.unique(mask)
    return labels[labels != 0]


class _UniqueSpy:
    """Records the sizes np.unique was called with, then delegates."""

    def __init__(self, real):
        self.real = real
        self.sizes = []

    def __call__(self, ar, *a, **kw):
        self.sizes.append(np.asarray(ar).size)
        return self.real(ar, *a, **kw)


def test_morphology_reports_the_same_labels_as_a_full_unique():
    mask = _mask()

    _props, _mask_out, valid_labels = ecp.extract_morphology(mask)

    np.testing.assert_array_equal(np.asarray(valid_labels), _expected_labels(mask))


def test_morphology_does_not_sort_every_pixel(monkeypatch):
    mask = _mask()
    spy = _UniqueSpy(np.unique)
    monkeypatch.setattr(ecp.np, "unique", spy)

    ecp.extract_morphology(mask)

    assert all(size < mask.size for size in spy.sizes), (
        f"np.unique was called on the whole mask ({mask.size} px); the label set is already "
        "available from regionprops_table"
    )


def test_morphology_still_reports_no_cells_for_an_empty_mask():
    empty = np.zeros((16, 16), dtype=np.uint32)

    props_df, mask_out, valid_labels = ecp.extract_morphology(empty)

    assert props_df is None and mask_out is None and valid_labels is None


def test_quantify_labels_match_a_full_unique():
    mask = _mask()
    channel = np.arange(mask.size, dtype=np.float64).reshape(mask.shape)

    df = q.compute_compartment_intensities(mask, None, channel, "CD8")

    np.testing.assert_array_equal(df["label"].to_numpy(), _expected_labels(mask))


def test_quantify_does_not_sort_every_pixel(monkeypatch):
    mask = _mask()
    channel = np.arange(mask.size, dtype=np.float64).reshape(mask.shape)
    spy = _UniqueSpy(np.unique)
    monkeypatch.setattr(q.np, "unique", spy)

    q.compute_compartment_intensities(mask, None, channel, "CD8")

    assert all(size < mask.size for size in spy.sizes), (
        f"np.unique was called on the whole mask ({mask.size} px); np.bincount is computed "
        "on the next line and its non-zero bins are the label set"
    )


def test_quantify_values_are_unchanged_by_the_cheaper_label_lookup():
    """The labels are only an index -- prove the numbers they select are the same."""
    mask = _mask()
    channel = np.arange(mask.size, dtype=np.float64).reshape(mask.shape)

    df = q.compute_compartment_intensities(mask, None, channel, "CD8")

    for label in _expected_labels(mask):
        expected = float(np.median(channel[mask == label]))
        got = float(df.loc[df["label"] == label, "CD8: Cell: Median"].iloc[0])
        assert got == expected


def test_quantify_still_returns_empty_for_a_mask_with_no_cells():
    empty = np.zeros((16, 16), dtype=np.uint32)
    channel = np.zeros((16, 16), dtype=np.float64)

    df = q.compute_compartment_intensities(empty, None, channel, "CD8")

    assert df.empty


# ---------------------------------------------------------------------------
# extract_contours must not iterate props_df.iterrows()
# ---------------------------------------------------------------------------
#
# `iterrows()` builds a pandas Series per row (60k for 60k cells) to box four integers. The
# replacement changes only how the same integers are fetched, so it is tested against
# iterrows directly.


def test_label_bboxes_matches_iterrows_exactly():
    import pandas as pd

    props_df = pd.DataFrame(
        {
            "bbox-0": [1, 10, 100],
            "bbox-1": [2, 20, 200],
            "bbox-2": [3, 30, 300],
            "bbox-3": [4, 40, 400],
            "area": [9.0, 9.0, 9.0],
        },
        index=pd.Index([7, 3, 11], name="label"),
    )

    expected = [
        (
            label,
            int(row["bbox-0"]),
            int(row["bbox-1"]),
            int(row["bbox-2"]),
            int(row["bbox-3"]),
        )
        for label, row in props_df.iterrows()
    ]

    assert list(ecp._label_bboxes(props_df)) == expected


def test_label_bboxes_preserves_row_order_not_sorted_order():
    """The index here is 7, 3, 11 -- a reordering would re-key every contour."""
    import pandas as pd

    props_df = pd.DataFrame(
        {
            "bbox-0": [1, 2, 3],
            "bbox-1": [1, 2, 3],
            "bbox-2": [4, 5, 6],
            "bbox-3": [4, 5, 6],
        },
        index=pd.Index([7, 3, 11], name="label"),
    )

    assert [row[0] for row in ecp._label_bboxes(props_df)] == [7, 3, 11]


def test_extract_contours_does_not_use_iterrows(monkeypatch):
    import pandas as pd

    def forbidden(self):
        raise AssertionError(
            "extract_contours used DataFrame.iterrows(); that builds one pandas Series per "
            "cell to box four integers"
        )

    monkeypatch.setattr(pd.DataFrame, "iterrows", forbidden)

    mask = np.zeros((32, 32), dtype=np.uint32)
    mask[8:20, 8:20] = 1
    props_df, mask_out, _labels = ecp.extract_morphology(mask)

    contours = ecp.extract_contours(mask_out, props_df)

    assert "1" in contours
