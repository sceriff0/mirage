import csv
import dataclasses

from benchmarks.regbench import score
from benchmarks.tests.regbench import scored_cohort


def test_panel_from_the_manifests_channel_list():
    case = scored_cohort._case("multiplex", "P1_r1", "P1")
    case = dataclasses.replace(case, extra={"channels_mov": "DAPI|CD8|CD3"})
    assert score.panel_of(case) == "DAPI_CD8_CD3"


def test_panel_from_a_prepared_cases_list():
    case = scored_cohort._case("multiplex", "P1_r1", "P1")
    case = dataclasses.replace(case, extra={"channels_mov": ["DAPI", "CD8", "CD3"]})
    assert score.panel_of(case) == "DAPI_CD8_CD3"
    case = dataclasses.replace(case, extra={"channel_names": ["A", "B"]})
    assert score.panel_of(case) == "A_B"


def test_panel_and_version_not_recorded():
    case = scored_cohort._case("multiplex", "P1_r1", "P1")
    assert score.panel_of(case) == "not recorded"
    assert score.version_of({"info": {}}) == "not recorded"
    assert score.version_of({"info": {"version": "1.3.0"}}) == "1.3.0"


def test_every_cell_row_has_the_three_columns(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    score.score(root, out, **scored_cohort.SCORE_KW)
    for name in ("cells.csv", "cells_nuclei.csv", "resources.csv"):
        rows = list(csv.DictReader(open(out / "tables" / name)))
        assert rows and all({"patient_id", "panel", "tool_version"} <= set(r) for r in rows)
        assert all(r["patient_id"] == r["group"] for r in rows)
    cells = list(csv.DictReader(open(out / "tables" / "cells.csv")))
    assert {r["tool_version"] for r in cells if r["method"] == "initial"} == {"-"}
    nb = list(csv.DictReader(open(out / "tables" / "nonband_cells.csv")))
    assert nb and all("tool_version" in r for r in nb)



# score.pair_code() at 0df7acec, before the labels: adding them moved no pair record code
PAIR_CODE_SCORE_PY = "1145a3e665506c6f68565eefe06e3a4d832044d6056f2c72d21e139dd32ec096"


def test_the_labels_leave_the_pair_code_alone():
    code = score.pair_code()
    assert code["score.py:" + ",".join(score.PAIR_CODE_MEMBERS)] == PAIR_CODE_SCORE_PY
