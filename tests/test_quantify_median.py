"""Equivalence tests for the per-label median in compute_compartment_intensities.

The uint16 fast path, `_median_per_label_from_sorted`, sorts a composite
`(label << 16) | value` key and reads each compartment's median off contiguous
sorted runs, with no per-label callback. Any other dtype falls back to
`scipy.ndimage.labeled_comprehension`. Both are checked to be numerically
identical to:

  1. an independent per-label np.median reference, and
  2. a copy of a per-pixel-DataFrame groupby-median implementation,
     on seeded random synthetic images.

Empty-label and even-count (median-of-two-middles) cases are covered
explicitly.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "quantify", Path(__file__).resolve().parent.parent / "bin" / "quantify.py"
)
quantify = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(quantify)

compute_compartment_intensities = quantify.compute_compartment_intensities


def _old_compute_compartment_intensities(cell_mask, nuclei_mask, channel, channel_name):
    """Per-pixel-DataFrame median algorithm, kept as a golden reference.

    One row per foreground pixel in a pandas DataFrame, then
    groupby(label).median(), with Nucleus/Cytoplasm compartments reindexed onto
    valid_labels.
    """
    cell_mask = np.ascontiguousarray(cell_mask.squeeze())
    channel = channel.squeeze()
    has_nuclei = nuclei_mask is not None
    if has_nuclei:
        nuclei_mask = np.ascontiguousarray(nuclei_mask.squeeze())

    valid_labels = np.unique(cell_mask)
    valid_labels = valid_labels[valid_labels != 0]
    if len(valid_labels) == 0:
        return {}

    flat_cell = cell_mask.ravel()
    flat_val = channel.ravel().astype(np.float64)

    nuc_fg = None
    if has_nuclei:
        nuc_fg = nuclei_mask.ravel() > 0

    fg = flat_cell != 0
    px = {"label": flat_cell[fg], "val": flat_val[fg]}
    if has_nuclei:
        px["nuc"] = nuc_fg[fg]
    df_px = pd.DataFrame(px)
    medians = {"Cell": df_px.groupby("label")["val"].median()}
    if has_nuclei:
        medians["Nucleus"] = df_px[df_px["nuc"]].groupby("label")["val"].median()
        medians["Cytoplasm"] = df_px[~df_px["nuc"]].groupby("label")["val"].median()

    out = {}
    compartments = ["Cell"] + (["Nucleus", "Cytoplasm"] if has_nuclei else [])
    for comp in compartments:
        out[f"{channel_name}: {comp}: Median"] = (
            medians[comp].reindex(valid_labels).fillna(0.0).values
        )
    out["label"] = valid_labels
    return out


def _reference_median_per_label(cell_mask, values, labels):
    """Independent np.median-per-label reference (no pandas, no scipy)."""
    out = np.zeros(len(labels), dtype=np.float64)
    flat_cell = cell_mask.ravel()
    flat_val = values.ravel().astype(np.float64)
    for i, lab in enumerate(labels):
        sel = flat_val[flat_cell == lab]
        out[i] = np.median(sel) if sel.size else 0.0
    return out


def test_multiple_labels_varying_size():
    cell_mask = np.array(
        [
            [1, 1, 1, 2],
            [1, 1, 2, 2],
            [3, 0, 0, 2],
            [3, 3, 0, 2],
        ]
    )
    channel = np.array(
        [
            [10.0, 20.0, 30.0, 1.0],
            [40.0, 50.0, 2.0, 3.0],
            [60.0, 0.0, 0.0, 4.0],
            [70.0, 80.0, 0.0, 5.0],
        ]
    )
    df = compute_compartment_intensities(cell_mask, None, channel, "CD8")
    labels = df["label"].values
    ref = _reference_median_per_label(cell_mask, channel, labels)
    np.testing.assert_allclose(df["CD8: Cell: Median"].values, ref, rtol=0, atol=0)
    # Sanity on the expected values themselves.
    expected = {
        1: np.median([10, 20, 30, 40, 50]),
        2: np.median([1, 2, 3, 4, 5]),
        3: np.median([60, 70, 80]),
    }
    for lab, val in zip(labels, df["CD8: Cell: Median"].values):
        assert val == expected[lab]


def test_single_pixel_label():
    cell_mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    channel = np.array([[0.0, 0.0, 0.0], [0.0, 42.5, 0.0], [0.0, 0.0, 0.0]])
    df = compute_compartment_intensities(cell_mask, None, channel, "CD8")
    assert df["CD8: Cell: Median"].values[0] == 42.5


def test_even_count_label_averages_two_middles():
    # Label 1 has exactly 4 foreground pixels -> median = mean of the two middle
    # sorted values. This exercises the tie/interpolation convention.
    cell_mask = np.array([[1, 1], [1, 1]])
    channel = np.array([[1.0, 2.0], [3.0, 4.0]])
    df = compute_compartment_intensities(cell_mask, None, channel, "CD8")
    # sorted [1,2,3,4] -> (2+3)/2 = 2.5
    assert df["CD8: Cell: Median"].values[0] == 2.5
    assert df["CD8: Cell: Median"].values[0] == np.median([1.0, 2.0, 3.0, 4.0])


def test_empty_compartment_is_nan_not_zero():
    # NaN, not 0.0: 0.0 would make "this cell has no nucleus" indistinguishable from
    # "its nucleus was measured and is dark". bin/export_geojson.py treats NaN as missing.
    # Full contract in tests/test_absent_compartment_is_nan.py.
    cell_mask = np.array([[1, 1, 1], [1, 1, 1]])
    nuclei_mask = np.zeros_like(cell_mask)  # no nucleus anywhere
    channel = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    df = compute_compartment_intensities(cell_mask, nuclei_mask, channel, "CD8")
    assert np.isnan(df["CD8: Nucleus: Median"].values[0])
    assert df["CD8: Cytoplasm: Median"].values[0] == df["CD8: Cell: Median"].values[0]

    # And the reverse: cell 2 is ENTIRELY nucleus -> Cytoplasm was never measured.
    cell_mask2 = np.array([[2, 2], [2, 2]])
    nuclei_mask2 = np.array([[1, 1], [1, 1]])
    channel2 = np.array([[10.0, 20.0], [30.0, 40.0]])
    df2 = compute_compartment_intensities(cell_mask2, nuclei_mask2, channel2, "CD8")
    assert np.isnan(df2["CD8: Cytoplasm: Median"].values[0])
    assert df2["CD8: Nucleus: Median"].values[0] == df2["CD8: Cell: Median"].values[0]


def test_background_label_zero_excluded():
    cell_mask = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]])
    channel = np.array([[999.0, 999.0, 5.0], [999.0, 7.0, 9.0], [999.0, 999.0, 999.0]])
    df = compute_compartment_intensities(cell_mask, None, channel, "CD8")
    assert list(df["label"].values) == [1]
    assert df["CD8: Cell: Median"].values[0] == np.median([5.0, 7.0, 9.0])


def test_golden_old_vs_new_random_images_no_nuclei():
    rng = np.random.default_rng(0)
    for _trial in range(25):
        h, w = rng.integers(5, 30), rng.integers(5, 30)
        n_labels = rng.integers(1, 8)
        cell_mask = rng.integers(0, n_labels + 1, size=(h, w))
        channel = rng.random((h, w)) * 1000.0

        new_df = compute_compartment_intensities(cell_mask, None, channel, "CD8")
        old = _old_compute_compartment_intensities(cell_mask, None, channel, "CD8")

        if len(old) == 0:
            assert new_df.empty
            continue

        np.testing.assert_array_equal(new_df["label"].values, old["label"])
        np.testing.assert_allclose(
            new_df["CD8: Cell: Median"].values,
            old["CD8: Cell: Median"],
            rtol=0,
            atol=0,
        )


def test_golden_old_vs_new_random_images_with_nuclei():
    rng = np.random.default_rng(0)
    for _trial in range(25):
        h, w = rng.integers(5, 30), rng.integers(5, 30)
        n_labels = rng.integers(1, 8)
        cell_mask = rng.integers(0, n_labels + 1, size=(h, w))
        # Nuclei mask: independent random binary overlay (need not share labels with
        # cell_mask — compute_compartment_intensities only uses nuclei_mask > 0).
        nuclei_mask = (rng.random((h, w)) > 0.5).astype(np.int32)
        channel = rng.random((h, w)) * 1000.0

        new_df = compute_compartment_intensities(cell_mask, nuclei_mask, channel, "CD8")
        old = _old_compute_compartment_intensities(
            cell_mask, nuclei_mask, channel, "CD8"
        )

        if len(old) == 0:
            assert new_df.empty
            continue

        np.testing.assert_array_equal(new_df["label"].values, old["label"])
        for comp in ("Cell", "Nucleus", "Cytoplasm"):
            key = f"CD8: {comp}: Median"
            np.testing.assert_allclose(new_df[key].values, old[key], rtol=0, atol=0)


def test_no_per_pixel_dataframe_in_source():
    """Sanity: the median branch builds no per-pixel DataFrame, and its hot (uint16) path
    invokes no per-label Python callback.

    `labeled_comprehension` still appears in the function as the non-uint16 fallback (the
    composite-key sort is only valid while values fit 16 bits), so the check isolates the
    uint16 branch, from its `if` guard to the fallback's `else:`, and asserts that slice
    uses the vectorised helper and no labeled_comprehension.
    """
    import inspect

    src = inspect.getsource(compute_compartment_intensities)
    # Isolate the median block (between the MEDIAN comment and the `if expanded:`
    # branch) and make sure it contains no pd.DataFrame / groupby call.
    start = src.index("Default statistic: MEDIAN")
    end = src.index("if expanded:")
    median_block = src[start:end]
    assert "pd.DataFrame" not in median_block
    assert ".groupby(" not in median_block

    # Isolate just the uint16 fast (hot) path from its `if` guard to the fallback `else:`.
    hot_start = median_block.index("if flat_raw.dtype == np.uint16:")
    hot_end = median_block.index("\n    else:", hot_start)
    hot_path = median_block[hot_start:hot_end]
    assert "_median_per_label_from_sorted" in hot_path
    assert "labeled_comprehension" not in hot_path

    # The fallback is still present -- just outside the hot path -- and is exercised by
    # test_non_uint16_fallback_matches_fast_path, so it is not dead code.
    assert "labeled_comprehension" in median_block[hot_end:]


def test_golden_old_vs_new_random_uint16_full_range():
    """Fast-path (uint16) equivalence, 40 trials: irregular label sets, cells with no nuclear
    overlap (forcing the NaN branch), the full uint16 intensity range, and both odd and even
    per-label pixel counts (median tie-break vs. a single middle element). Assert at
    rtol=0, atol=0, NaN-aware.

    Unlike the two golden tests above (float64 channels, which never take the composite-key
    sort), this one uses a uint16 channel so it exercises the fast path, and cross-checks
    every value against both the per-pixel-DataFrame golden and an independent per-label
    np.median reference.
    """
    rng = np.random.default_rng(7)
    n_trials = 40
    saw_no_overlap = False
    saw_odd = False
    saw_even = False

    for _trial in range(n_trials):
        h, w = rng.integers(4, 40), rng.integers(4, 40)
        n_labels = rng.integers(1, 12)
        cell_mask = rng.integers(0, n_labels + 1, size=(h, w))
        # Full uint16 range, including the 0 and 65535 extremes.
        channel = rng.integers(0, 65536, size=(h, w), dtype=np.uint16)
        # Independent per-pixel nuclear overlay: with small cells this reliably produces some
        # cells with zero nuclear overlap and some that are entirely nucleus, across trials.
        nuclei_mask = (rng.random((h, w)) > 0.5).astype(np.int32)

        new_df = compute_compartment_intensities(cell_mask, nuclei_mask, channel, "CD8")
        old = _old_compute_compartment_intensities(
            cell_mask, nuclei_mask, channel, "CD8"
        )

        if len(old) == 0:
            assert new_df.empty
            continue

        labels = new_df["label"].values
        np.testing.assert_array_equal(labels, old["label"])

        flat_cell = cell_mask.ravel()
        flat_val = channel.ravel().astype(np.float64)
        nuc_fg_full = nuclei_mask.ravel() > 0

        for comp in ("Cell", "Nucleus", "Cytoplasm"):
            key = f"CD8: {comp}: Median"
            new_vals = new_df[key].values
            old_vals = np.asarray(old[key], dtype=np.float64)
            for i, lab in enumerate(labels):
                sel = flat_cell == lab
                if comp == "Nucleus":
                    sel = sel & nuc_fg_full
                elif comp == "Cytoplasm":
                    sel = sel & ~nuc_fg_full
                count = int(sel.sum())
                if count == 0:
                    assert np.isnan(new_vals[i])
                    saw_no_overlap = True
                    continue
                saw_even = saw_even or count % 2 == 0
                saw_odd = saw_odd or count % 2 == 1
                # rtol=0, atol=0: bit-identical to the per-pixel-DataFrame golden...
                assert new_vals[i] == old_vals[i], (comp, lab, new_vals[i], old_vals[i])
                # ...and to an independent reference computed fresh from the raw pixels.
                assert new_vals[i] == np.median(flat_val[sel])

    assert saw_no_overlap, (
        "no trial produced an empty compartment (NaN branch unexercised)"
    )
    assert saw_odd, "no trial produced an odd-count label (single middle element)"
    assert saw_even, "no trial produced an even-count label (tie-break averaging)"


def test_non_uint16_fallback_matches_fast_path():
    """Only a uint16 channel takes the composite-key fast path; anything else falls back to
    labeled_comprehension, which must produce identical values on the same data. Keeps the
    fallback reachable and numerically pinned.
    """
    rng = np.random.default_rng(11)
    cell_mask = rng.integers(0, 6, size=(20, 25))
    nuclei_mask = (rng.random((20, 25)) > 0.5).astype(np.int32)
    channel_uint16 = rng.integers(0, 65536, size=(20, 25), dtype=np.uint16)

    fast_df = compute_compartment_intensities(
        cell_mask, nuclei_mask, channel_uint16, "CD8"
    )

    # Every dtype below carries the exact same integer values as channel_uint16, just
    # widened/retyped, so `compute_compartment_intensities` routes to the labeled_comprehension
    # fallback (dtype != uint16) but must still return numbers identical to the fast path.
    for dtype in (np.float64, np.int32, np.uint32, np.int64):
        channel_other = channel_uint16.astype(dtype)
        fallback_df = compute_compartment_intensities(
            cell_mask, nuclei_mask, channel_other, "CD8"
        )
        np.testing.assert_array_equal(
            fallback_df["label"].values, fast_df["label"].values
        )
        for comp in ("Cell", "Nucleus", "Cytoplasm"):
            key = f"CD8: {comp}: Median"
            np.testing.assert_allclose(
                fallback_df[key].values,
                fast_df[key].values,
                rtol=0,
                atol=0,
                equal_nan=True,
            )
