"""bin/generate_postprocessing_qc.py — three PNGs, and what they are made of.

No behavioural test existed for this script. Its process now carries a
retry-then-fail policy, i.e. the pipeline treats a QC failure as fatal -- so
"it produced a file" is the minimum contract, and it was unasserted.

Uses the generator's own fixtures (P001_cell_mask.tif, sample_merged_quant.csv)
rather than synthesising a mask, because the point is that the script reads what
SEGMENT and MERGE_QUANT_CSVS actually write.
"""

from __future__ import annotations

from pathlib import Path

import generate_postprocessing_qc as gpq
import numpy as np
import pandas as pd
import pytest

TESTDATA = Path(__file__).resolve().parent / "testdata"


def _fixture(name: str) -> Path:
    path = TESTDATA / name
    if not path.exists():
        pytest.fail(
            f"{path} is missing. Run `python tests/testdata/generate_complete_testdata.py`."
        )
    return path


def test_it_produces_all_three_panels_and_none_of_them_is_empty(tmp_path):
    outputs = gpq.generate_postprocessing_qc(
        _fixture("P001_cell_mask.tif"),
        _fixture("sample_merged_quant.csv"),
        tmp_path,
        "P001",
    )
    names = sorted(p.name for p in outputs)
    assert names == [
        "P001_cell_stats.png",
        "P001_intensity_distributions.png",
        "P001_seg_overlay.png",
    ]
    for path in outputs:
        assert path.exists(), f"{path.name} was returned but not written"
        assert path.stat().st_size > 1000, (
            f"{path.name} is {path.stat().st_size} bytes -- a matplotlib figure that "
            "rendered nothing still writes a small valid PNG, so size is the only "
            "cheap signal that the panel has content"
        )


def test_the_prefix_names_every_output(tmp_path):
    """The prefix is the patient id, and these PNGs are published side by side
    for every patient in a cohort -- a prefix that failed to reach one filename
    is a silent overwrite between patients."""
    outputs = gpq.generate_postprocessing_qc(
        _fixture("P001_cell_mask.tif"),
        _fixture("sample_merged_quant.csv"),
        tmp_path,
        "PATIENT_XYZ",
    )
    assert all(p.name.startswith("PATIENT_XYZ_") for p in outputs)


def test_load_mask_reads_both_shapes_segment_can_write(tmp_path):
    """SEGMENT writes a uint32 TIFF; the fixtures also carry the .npy the older
    tests use. load_mask has to squeeze both to 2-D -- a (1, H, W) mask reaching
    regionprops counts zero cells and reports it as a successful QC panel."""
    tif = gpq.load_mask(_fixture("P001_cell_mask.tif"))
    npy = gpq.load_mask(_fixture("P001_cell_mask.npy"))
    assert tif.ndim == 2 and npy.ndim == 2
    assert tif.shape == npy.shape
    assert np.array_equal(tif.astype(np.int64), npy.astype(np.int64))


def test_a_mask_with_no_cells_is_still_reported_rather_than_crashing(tmp_path):
    """An empty mask is a real outcome (a blank tissue region), and a QC panel
    that raised on it would fail the whole run under retry-then-fail."""
    import tifffile

    empty = tmp_path / "empty_mask.tif"
    tifffile.imwrite(empty, np.zeros((64, 64), dtype=np.uint32), compression="zlib")
    outputs = gpq.generate_postprocessing_qc(
        empty, _fixture("sample_merged_quant.csv"), tmp_path / "out", "P001"
    )
    assert (tmp_path / "out" / "P001_seg_overlay.png").exists()
    assert outputs


def test_qc_columns_are_excluded_from_intensity_distributions():
    """A "QC: ..." column (e.g. QC: Total intensity) is not a marker; plotting it
    as an intensity histogram alongside DAPI/PANCK/etc mislabels it as one."""
    df = pd.DataFrame(
        {
            "label": [1, 2],
            "DAPI": [10.0, 20.0],
            "QC: Total intensity": [5.0, 6.0],
        }
    )
    cols = gpq._intensity_marker_columns(df)
    assert "QC: Total intensity" not in cols
    assert "DAPI" in cols
