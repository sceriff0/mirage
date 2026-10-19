"""T29 (WD6): which channel each method registered on, per (run, case), read score-side
from what its run.json records (``regbench.reg_channels``); never a method's behaviour."""

import json

from benchmarks.regbench import reg_channels, score
from benchmarks.regbench.cases import Case, result_dir


def _case(tmp_path, ref_channel=0, mov_channel=2):
    return Case(
        dataset="multiplex", case_id="P1_r1", group="P1",
        ref_nuclear="", mov_nuclear="",
        ref_image=str(tmp_path / "P1_ref.ome.tiff"),
        mov_image=str(tmp_path / "P1_r1.ome.tiff"),
        modality="fluorescence", diagonal=100.0,
        extra={"ref_channel": ref_channel, "mov_channel": mov_channel,
               "channels_ref": ["DAPI", "CD3"],
               "channels_mov": ["CELLTOX", "CD8", "DAPI"]},
    )


def _record(out, name, case, **rec):
    d = result_dir(out, name, case)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"variant": name, "ok": True, **rec}))


def test_valis_picking_channel_zero_on_a_celltox_round_is_flagged(tmp_path):
    case = _case(tmp_path)
    out = tmp_path / "out"
    # VALIS's `dapi` regex: the reference has DAPI at 0; the moving round's names do
    # not match it once (two DAPI-like names), so VALIS fell to channel 0 (CELLTOX)
    _record(out, "valis_mirage", case, method="valis_mirage", info={"nuclear_channel": {
        "P1_ref": {"index": 0, "name": "DAPI"},
        "P1_r1": {"index": 0, "name": "CELLTOX"}}})
    _record(out, "stare", case, method="stare", info={"nuclear_index": 2})
    _record(out, "palom", case, method="palom", native={"nuclear_channel": 2})
    got = {r["run"]: r for r in reg_channels.rows(
        out, case, ["initial", "valis_mirage", "stare", "palom", "ashlar"])}
    v = got["valis_mirage"]
    assert (v["nuclear_channel_ref"], v["nuclear_channel_mov"]) == (0, 2)
    assert (v["registered_on_ref"], v["registered_on_mov"]) == (0, 0)
    assert v["registered_name_mov"] == "CELLTOX" and v["nuclear_name_mov"] == "DAPI"
    assert v["registration_channel_is_nuclear"] is False
    # STARE's one --nuclear-index is the case's: both sides
    s = got["stare"]
    assert s["registration_channel_is_nuclear"] is False  # ref is 0, STARE used 2
    p = got["palom"]
    assert p["registered_on_mov"] == 2 and p["registered_on_ref"] == reg_channels.NOT_RECORDED
    assert p["registration_channel_is_nuclear"] is True  # what is recorded matches
    # a run with no record for the case: said, never guessed
    assert got["ashlar"]["registration_channel_is_nuclear"] == reg_channels.NOT_RECORDED
    assert "initial" not in got


def test_a_matching_stare_round_and_an_unmatched_valis_slide(tmp_path):
    case = _case(tmp_path, ref_channel=0, mov_channel=0)
    out = tmp_path / "out"
    _record(out, "stare", case, method="stare", info={"nuclear_index": 0})
    _record(out, "valis_mirage", case, method="valis_mirage", info={"nuclear_channel": {
        "something_else": {"index": 1, "name": "CD8"}}})
    got = {r["run"]: r for r in reg_channels.rows(out, case, ["stare", "valis_mirage"])}
    assert got["stare"]["registration_channel_is_nuclear"] is True
    v = got["valis_mirage"]
    assert v["registration_channel_is_nuclear"] == reg_channels.NOT_RECORDED
    assert "something_else" in v["note"]


def test_the_table_is_one_of_the_scorers():
    assert "registration_channels" in score.TABLES
    assert reg_channels.COLUMNS[:3] == ("dataset", "patient_id", "case_id")


def test_the_scored_cohort_writes_one_row_per_run_and_case(tmp_path):
    import csv

    from benchmarks.tests.regbench import scored_cohort

    root, out, made = scored_cohort.build(tmp_path)
    got = score.score(root, out, **scored_cohort.SCORE_KW)
    with open(out / "tables" / "registration_channels.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows == [{k: str(v) for k, v in r.items()} for r in got["registration_channels"]]
    keys = [(r["run"], r["case_id"]) for r in rows]
    assert len(keys) == len(set(keys)) and {r["case_id"] for r in rows} == {
        c.case_id for c in made}
    # the toy methods record no channel: said, never guessed
    assert {r["registration_channel_is_nuclear"] for r in rows} == {reg_channels.NOT_RECORDED}
