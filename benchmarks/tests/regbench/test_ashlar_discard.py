"""ASHLAR's replaced tiles per round and cell, beside the case's rigid offset
(``regbench.ashlar_discard``). The anchor is read where the ASHLAR adapter reads it."""

import json

from benchmarks.regbench import ashlar_discard, score, table_schema
from benchmarks.regbench.methods import ashlar


def _run(path, doc):
    path.mkdir(parents=True, exist_ok=True)
    (path / "run.json").write_text(json.dumps(doc))


def test_rigid_offset_of_a_pure_translation():
    m0 = [[1.0, 0.0, 300.0], [0.0, 1.0, -400.0]]
    assert ashlar_discard.rigid_offset_um(m0, (1000, 2000), 0.5) == 250.0
    m3 = [[1.0, 0.0, 300.0], [0.0, 1.0, -400.0], [0.0, 0.0, 1.0]]
    assert ashlar_discard.rigid_offset_um(m3, (1000, 2000), 0.5) == 250.0
    assert ashlar_discard.rigid_offset_um(m3, (1000, 2000), None) is None


def test_rigid_offset_of_a_rotation_is_read_at_the_centre():
    # 90 degrees about the origin: the centre (W/2, H/2) = (10, 5) goes to (-5, 10)
    m0 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]
    got = ashlar_discard.rigid_offset_um(m0, (10, 20), 1.0)
    assert abs(got - (15**2 + 5**2) ** 0.5) < 1e-9


def test_rows_read_the_cells_record_and_stares_anchor(tmp_path, toy_cases):
    case = next(iter(toy_cases.values()))
    _run(
        tmp_path / "ashlar_anone_n0_s60" / case.dataset / case.case_id,
        {
            "ok": True,
            "opts": {"affine": "none", "noise_frac": "0", "max_shift_um": "60"},
            "info": {
                "discard_frac": 0.25,
                "discard_infinite_error": 1,
                "discard_zero_offset": 0,
                "discard_over_budget": 3,
                "n_tiles": 16,
                "cycle_fully_modelled": False,
            },
        },
    )
    stare = tmp_path / "stare" / case.dataset / case.case_id
    _run(stare, {"ok": True, "info": {"m0": [[1, 0, 10], [0, 1, 0], [0, 0, 1]]}})
    (row,) = ashlar_discard.rows(tmp_path, [case])
    assert row["row"] == "ashlar_anone_n0_s60"
    assert row["discard_frac"] == 0.25 and row["max_shift_um"] == "60"
    assert row["rigid_offset_um"] == 10 * case.pixel_size_um
    assert row["rigid_offset_from"] == str(stare / "run.json")
    assert set(row) == set(table_schema.columns("ashlar_discard")[1:])


def test_the_anchor_is_the_one_the_adapter_reads(tmp_path, toy_cases):
    case = next(iter(toy_cases.values()))
    _run(
        tmp_path / "ashlar" / case.dataset / case.case_id,
        {"ok": True, "opts": {}, "info": {}},
    )
    _run(
        tmp_path / "stare" / case.dataset / case.case_id,
        {"ok": True, "info": {"m0": [[1, 0, 7], [0, 1, 0], [0, 0, 1]]}},
    )
    m0, where = ashlar.read_anchor({}, case, tmp_path)
    (row,) = ashlar_discard.rows(tmp_path, [case])
    assert row["rigid_offset_from"] == where
    assert row["rigid_offset_um"] == ashlar_discard.rigid_offset_um(
        m0, case.ref_hw, case.pixel_size_um
    )


def test_round_without_discard_record(tmp_path, toy_cases):
    case = next(iter(toy_cases.values()))
    _run(
        tmp_path / "ashlar" / case.dataset / case.case_id,
        {"ok": False, "opts": {}, "info": {}},
    )
    (row,) = ashlar_discard.rows(tmp_path, [case])
    assert row["discard_frac"] is None and row["rigid_offset_um"] is None
    assert row["rigid_offset_from"] is None and row["ok"] is False


def test_a_name_that_never_ran_has_no_row(tmp_path, toy_cases):
    case = next(iter(toy_cases.values()))
    assert ashlar_discard.rows(tmp_path, [case]) == []


def test_score_writes_the_table():
    assert "ashlar_discard" in score.TABLES
