import numpy as np

from quantify import (
    compute_channel_intensity,
    compute_morphology,
    quantify_single_channel,
)


def test_compute_channel_intensity_simple():
    mask = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int32)
    channel = np.array([[0, 5, 7], [0, 3, 9]], dtype=np.float32)

    props_df, mask_filtered, valid_labels = compute_morphology(mask, min_area=0)
    assert props_df is not None
    assert mask_filtered is not None
    assert valid_labels is not None

    intensities = compute_channel_intensity(
        mask_filtered=mask_filtered,
        channel=channel,
        valid_labels=valid_labels,
        channel_name="TEST",
    )

    assert "TEST" == intensities.name
    assert np.isclose(intensities.loc[1], 6.0)
    assert np.isclose(intensities.loc[2], 6.0)


def test_quantify_single_channel_outputs_expected_columns():
    mask = np.array([[0, 1], [0, 2]], dtype=np.int32)
    channel = np.array([[0, 10], [0, 20]], dtype=np.uint16)

    df = quantify_single_channel(mask, channel, channel_name="CH1", min_area=0)

    assert not df.empty
    assert {"label", "y", "x", "area", "CH1"}.issubset(df.columns)
    values = dict(zip(df["label"], df["CH1"]))
    assert np.isclose(values[1], 10.0)
    assert np.isclose(values[2], 20.0)
