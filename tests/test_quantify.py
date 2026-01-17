import numpy as np
import pandas as pd
import tifffile
import tempfile
from pathlib import Path

from quantify import compute_cell_intensities, quantify_multichannel


def test_compute_cell_intensities_simple():
    # mask with labels 1 and 2, each with two pixels
    mask = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int32)
    channel = np.array([[0, 5, 7], [0, 3, 9]], dtype=np.float32)

    df = compute_cell_intensities(mask, channel, channel_name="TEST", min_area=0)

    # Expect label indices 1 and 2 with mean 6.0 each
    assert "TEST" in df.columns
    vals = df["TEST"].to_dict()
    assert vals[1] == 6.0
    assert vals[2] == 6.0


def test_quantify_multichannel_reads_tmp_files(tmp_path):
    # create a small mask
    mask = np.array([[0, 1], [0, 2]], dtype=np.int32)

    # create two channel tif files
    ch1 = np.array([[0, 10], [0, 20]], dtype=np.uint16)
    ch2 = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    f1 = tmp_path / "patient_CH1.tiff"
    f2 = tmp_path / "patient_CH2.tiff"
    tifffile.imwrite(str(f1), ch1)
    tifffile.imwrite(str(f2), ch2)

    # call quantify_multichannel directly (bypassing CLI)
    df = quantify_multichannel(mask, [str(f1), str(f2)], min_area=0)

    # DataFrame should contain both channel columns and some morphological cols
    assert not df.empty
    # channel names should be derived from filename suffixes
    assert any("CH1" in c or "CH2" in c for c in df.columns)
import numpy as np
import pandas as pd

from quantify import compute_cell_intensities, quantify_multichannel


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
