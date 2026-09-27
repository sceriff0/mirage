from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))
sys.path.insert(0, str(BIN / "utils"))

import cell_qc  # noqa: E402

NUC = ["DAPI"]
RET = "QC: Nuclear retention: [CD3, CD8]"
DISP = "QC: Registration displacement µm: [CD3, CD8]"
DICE = "QC: Registration Dice: [CD3, CD8]"


def _quant(n=4):
    return pd.DataFrame(
        {
            "label": np.arange(1, n + 1),
            "x": [10.0, 50.0, 90.0, 130.0][:n],
            "y": [10.0] * n,
            "DAPI: Nucleus: Median": [1000.0] * n,
            "DAPI: Cell: Median": [800.0] * n,
            "PANCK: Cell: Median": [5.0, np.nan, 1.0, 2.0][:n],
            "CD3: Cell: Median": [1.0, np.nan, 2.0, 3.0][:n],
            "CD8: Cell: Median": [2.0, np.nan, np.nan, 1.0][:n],
        }
    )


def _rounds(ret="mov1_nuclear_retention.csv", res=None):
    return [
        {"round_id": "ref", "is_reference": True, "markers": ["DAPI", "PANCK"],
         "retention_csv": None, "residual_csv": None},
        {"round_id": "mov1", "is_reference": False, "markers": ["CD8", "CD3", "DAPI"],
         "retention_csv": ret, "residual_csv": res},
    ]


def _retention(tmp_path, nucleus):
    d = tmp_path / "retention"
    d.mkdir(exist_ok=True)
    pd.DataFrame({"label": [1, 2, 3, 4], "Nucleus": nucleus, "Cell": nucleus}).to_csv(
        d / "mov1_nuclear_retention.csv", index=False
    )
    return d


def _residuals(tmp_path):
    d = tmp_path / "residuals"
    d.mkdir(exist_ok=True)
    (d / "P_mov1_reg_residuals.csv").write_text(
        "moving,ref_x,ref_y,residual_px,iou,stage\n"
        "m,10,10,2.0,0.6,micro\nm,50,10,0.0,1.0,micro\n"
    )
    return d


def _run(tmp_path, rounds, nucleus=(1000.0, 1000.0, 10.0, 1000.0)):
    return cell_qc.add_qc_columns(
        _quant(), rounds, _retention(tmp_path, list(nucleus)),
        _residuals(tmp_path), pixel_size=0.5, join_max_px=5.0, nuclear_markers=NUC,
    )


def test_total_intensity_is_the_non_nuclear_median_sum():
    t = cell_qc.total_intensity(_quant(), NUC)
    assert t.tolist()[0] == pytest.approx(8.0)       # 5 + 1 + 2, DAPI excluded
    assert np.isnan(t.tolist()[1])                    # every term NaN -> NaN


def test_retention_is_normalised_per_round():
    ref = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    mov = np.array([500.0, 500.0, 5.0, 500.0])        # a globally dimmer round
    r = cell_qc.normalised_retention(ref, mov)
    assert r[0] == pytest.approx(1.0) and r[2] == pytest.approx(0.01)


def test_round_keys_use_sorted_non_nuclear_markers(tmp_path):
    out = _run(tmp_path, _rounds(res="P_mov1_reg_residuals.csv"))
    assert RET in out.columns and DISP in out.columns and DICE in out.columns
    assert out.loc[2, RET] == pytest.approx(0.01)
    assert out.loc[0, DISP] == pytest.approx(1.0)     # 2 px * 0.5 µm
    assert out.loc[0, DICE] == pytest.approx(2 * 0.6 / 1.6)
    assert np.isnan(out.loc[2, DISP])                 # unmatched: no evidence
    assert "QC: Total intensity" in out.columns


def test_round_without_residuals_has_no_registration_keys(tmp_path):
    out = _run(tmp_path, _rounds(res=None))
    assert RET in out.columns
    assert DISP not in out.columns and DICE not in out.columns


def test_single_slide_patient_has_only_total_intensity(tmp_path):
    out = _run(tmp_path, _rounds()[:1])
    qc = [c for c in out.columns if c.startswith("QC: ")]
    assert qc == ["QC: Total intensity"]


def test_round_with_no_non_nuclear_markers_gets_no_keys(tmp_path):
    rounds = _rounds()
    rounds[1]["markers"] = ["DAPI"]
    out = _run(tmp_path, rounds)
    assert [c for c in out.columns if c.startswith("QC: ")] == ["QC: Total intensity"]


def test_recomputing_replaces_old_qc_columns(tmp_path):
    first = _run(tmp_path, _rounds())
    first[RET] = -1.0
    again = cell_qc.add_qc_columns(
        first, _rounds(), tmp_path / "retention", tmp_path / "residuals",
        pixel_size=0.5, join_max_px=5.0, nuclear_markers=NUC,
    )
    assert (again[RET] >= 0).all()
    assert list(again.columns).count(RET) == 1


def test_long_table_has_one_row_per_cell_and_moving_round(tmp_path):
    out = _run(tmp_path, _rounds(res="P_mov1_reg_residuals.csv"))
    # round_long_table reads the PUBLISHED manifest shape (nuclear markers already
    # removed by main()), not the raw per-slide rows.
    manifest = [{**r, "markers": [m for m in r["markers"] if m != "DAPI"]} for r in _rounds()]
    long = cell_qc.round_long_table(out, manifest, pixel_size=0.5)
    assert list(long.columns) == [
        "label", "round_id", "markers", "nuclear_retention",
        "displacement_px", "displacement_um", "dice",
    ]
    assert len(long) == 4 and set(long["round_id"]) == {"mov1"}
    assert long.loc[0, "markers"] == "CD3|CD8"
    assert long.loc[0, "displacement_px"] == pytest.approx(2.0)


def test_main_writes_three_outputs(tmp_path):
    base = tmp_path / "base.csv"
    _quant().to_csv(base, index=False)
    _retention(tmp_path, [1000.0, 1000.0, 10.0, 1000.0])
    _residuals(tmp_path)
    rounds = tmp_path / "rounds_in.json"
    rounds.write_text(json.dumps(_rounds(res="P_mov1_reg_residuals.csv")))
    rc = cell_qc.main([
        "--merged", str(base), "--rounds", str(rounds),
        "--retention-dir", str(tmp_path / "retention"),
        "--residual-dir", str(tmp_path / "residuals"),
        "--pixel-size", "0.5", "--join-max-px", "5", "--nuclear-markers", "DAPI",
        "--patient-id", "P",
        "--out-merged", str(tmp_path / "merged_quant.csv"),
        "--out-round-qc", str(tmp_path / "P_round_qc.csv"),
        "--out-rounds", str(tmp_path / "P_rounds.json"),
    ])
    assert rc == 0
    assert RET in pd.read_csv(tmp_path / "merged_quant.csv").columns
    assert json.loads((tmp_path / "P_rounds.json").read_text())[1] == {
        "round_id": "mov1", "is_reference": False, "markers": ["CD3", "CD8"],
    }
