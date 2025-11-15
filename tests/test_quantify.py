import numpy as np
import pandas as pd

from scripts.quantify import compute_cell_intensities, quantify_multichannel


def test_compute_cell_intensities_basic():
    # Create a simple mask with two cells (labels 1 and 2)
    mask = np.array([
        [0, 1, 1],
        [0, 2, 2],
        [0, 0, 0]
    ], dtype=int)

    # Channel image with distinct intensities
    channel = np.array([
        [0, 10, 10],
        [0, 20, 20],
        [0, 0, 0]
    ], dtype=float)

    df = compute_cell_intensities(mask, channel, 'TEST', min_area=0)

    # Expect two rows for labels 1 and 2
    assert 'TEST' in df.columns
    assert len(df) == 2
    # Mean intensities should match
    vals = df['TEST'].sort_index().values
    assert np.isclose(vals[0], 10.0)
    assert np.isclose(vals[1], 20.0)


def test_quantify_multichannel_combination():
    mask = np.array([
        [0, 1],
        [0, 1]
    ], dtype=int)

    ch1 = np.array([[0, 5], [0, 5]], dtype=float)
    ch2 = np.array([[0, 2], [0, 8]], dtype=float)

    # quantify_multichannel expects file paths; call compute directly per-channel
    df1 = compute_cell_intensities(mask, ch1, 'CH1')
    df2 = compute_cell_intensities(mask, ch2, 'CH2')

    result = pd.concat([df1, df2], axis=1)
    assert 'CH1' in result.columns and 'CH2' in result.columns
    assert np.isclose(result['CH1'].iloc[0], 5.0)
    assert np.isclose(result['CH2'].iloc[0], 5.0)
