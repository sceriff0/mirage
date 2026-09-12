import math

import pandas as pd
import pytest

from benchmarks.anhir import dataset as ds


def test_tissue_and_scale_are_read_off_the_relative_path():
    assert ds.tissue_of("COAD_01/scale-25pc/S1.jpg") == "COAD"
    assert ds.tissue_of("lung-lesion_2/scale-100pc/x.jpg") == "lung-lesion"
    assert ds.tissue_of("mice-kidney_1/scale-5pc/x.jpg") == "mice-kidney"
    assert ds.scale_of("COAD_01/scale-25pc/S1.jpg") == "scale-25pc"


def test_load_cases_types_every_row(anhir_root):
    cases = anhir_root["cases"]
    assert [c.case_id for c in cases] == [0, 1, 2]
    c = cases[0]
    assert c.tissue == "COAD" and c.scale == "scale-25pc" and c.status == "training"
    assert c.size == (1600, 1200)
    assert c.diagonal == pytest.approx(2000.0)
    assert c.source_stem == "S1" and c.target_stem == "HE"
    assert c.scored_locally
    assert not cases[2].scored_locally


def test_diagonal_falls_back_to_the_size_when_the_column_is_blank(tmp_path):
    cover = tmp_path / "c.csv"
    pd.DataFrame(
        [
            {
                ds.COL_DIAGONAL: float("nan"),
                ds.COL_SIZE: "(300, 400)",
                ds.COL_SOURCE: "a_1/scale-25pc/S1.jpg",
                ds.COL_SOURCE_LND: "a_1/scale-25pc/S1.csv",
                ds.COL_TARGET: "a_1/scale-25pc/HE.jpg",
                ds.COL_TARGET_LND: "a_1/scale-25pc/HE.csv",
                ds.COL_STATUS: "training",
            }
        ]
    ).to_csv(cover, index=True)
    (case,) = ds.load_cases(cover)
    assert case.diagonal == pytest.approx(math.hypot(300, 400))


def test_load_cases_refuses_an_unknown_status_and_a_missing_column(tmp_path):
    cover = tmp_path / "c.csv"
    pd.DataFrame(
        [
            {
                ds.COL_DIAGONAL: 5.0,
                ds.COL_SIZE: "(3, 4)",
                ds.COL_SOURCE: "a_1/s/S1.jpg",
                ds.COL_SOURCE_LND: "a_1/s/S1.csv",
                ds.COL_TARGET: "a_1/s/HE.jpg",
                ds.COL_TARGET_LND: "a_1/s/HE.csv",
                ds.COL_STATUS: "validation",
            }
        ]
    ).to_csv(cover, index=True)
    with pytest.raises(ValueError, match="status 'validation'"):
        ds.load_cases(cover)
    pd.DataFrame([{ds.COL_SOURCE: "x"}]).to_csv(cover, index=True)
    with pytest.raises(ValueError, match="missing cover column"):
        ds.load_cases(cover)


def test_select_cases_filters_and_limits(anhir_root):
    cases = anhir_root["cases"]
    assert [c.case_id for c in ds.select_cases(cases, status="training")] == [0, 1]
    assert [c.case_id for c in ds.select_cases(cases, status="all")] == [0, 1, 2]
    assert [c.case_id for c in ds.select_cases(cases, tissues=["lung-lesion"])] == [2]
    assert [c.case_id for c in ds.select_cases(cases, case_ids=[1])] == [1]
    assert len(ds.select_cases(cases, limit=1)) == 1


def test_resolve_paths_join_the_archive_roots(anhir_root):
    c = anhir_root["cases"][0]
    assert ds.resolve_image(c, anhir_root["images"], "source").name == "S1.jpg"
    assert ds.resolve_landmarks(c, anhir_root["landmarks"], "target").exists()
    assert not ds.resolve_landmarks(
        anhir_root["cases"][2], anhir_root["landmarks"], "target"
    ).exists()
