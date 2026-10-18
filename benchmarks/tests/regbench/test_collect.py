"""The interim collector (``benchmarks/regbench/paper/collect.py``): the existing arm, sweep and
interim regbench results in the paper's table schema, NA for everything still to come.

Stand-in trees in the real layouts: an arm's ``<arm>/<patient>/qc/registration/*_seg_qc.json``,
``csv/registered.csv``, ``trace/trace.txt`` and ``size_logs/input_sizes.csv``; a sweep's
``<run_id>/trace/trace.txt`` and its plan; a regbench folder's ``tables/cells.csv``, its
``cache/scores/<dataset>/<case>/case.rec`` and ``<row>/<dataset>/<case>/run.json``."""
import hashlib
import json
import math
import pickle
from types import SimpleNamespace

import pytest

from benchmarks.regbench.paper import collect, schema


def _arm(root, arm, patient, panel, dice, stage="full_transform"):
    d = root / arm
    (d / "csv").mkdir(parents=True, exist_ok=True)
    (d / "csv" / "registered.csv").write_text(
        f"patient_id,id,channels\n{patient},{patient}_{panel},{panel.replace('_', '|')}\n")
    q = d / patient / "qc" / "registration"
    q.mkdir(parents=True)
    (q / f"{patient}_{panel}_seg_qc.json").write_text(json.dumps(
        {"moving": f"{patient}_{panel}", stage: {"dice_matched": dice}}))


def test_every_schema_table_is_written_even_empty(tmp_path):
    collect.run(tmp_path / "c", None, None, None, [], None)
    for name in schema.TABLES:
        assert (tmp_path / "c" / f"{name}.csv").read_text().startswith("source,")


def test_arm_dice_lands_on_the_regbench_row(tmp_path):
    _arm(tmp_path / "arms", "tiled_high_s64", "P1", "DAPI_CD8_CD3", 0.81)
    collect.run(tmp_path / "c", tmp_path / "arms", None, None, [], None)
    rows = schema.read(tmp_path / "c" / "per_pair.csv")
    (r,) = [x for x in rows if x["source"] == "arms:tiled_high_s64"]
    assert r["row"] == "stare" and r["panel"] == "DAPI_CD8_CD3" and r["dice_matched"] == "0.81"


def test_damaged_inputs_become_na_rows(tmp_path):
    _arm(tmp_path / "arms", "valis_high_micro2", "P1", "DAPI_CD8_CD3", 0.7)
    bad = tmp_path / "arms" / "valis_high_micro2" / "P1" / "qc" / "registration" / "x_seg_qc.json"
    bad.write_text("{truncated")
    (tmp_path / "arms" / "broken_arm").mkdir()
    collect.run(tmp_path / "c", tmp_path / "arms", None, None, [], None)
    notes = [r["note"] for r in schema.read(tmp_path / "c" / "crosscheck_arms.csv")] + [
        line for line in (tmp_path / "c" / "README.txt").read_text().splitlines()]
    assert any("x_seg_qc.json" in n for n in notes if n)
    assert any("broken_arm" in n for n in notes if n)


def test_the_collection_is_never_written_into_a_results_folder(tmp_path):
    out = tmp_path / "out"
    (out / "stare" / "multiplex" / "P1_r1").mkdir(parents=True)
    (out / "stare" / "multiplex" / "P1_r1" / "run.json").write_text("{}")
    with pytest.raises(SystemExit):
        collect.run(out / "tables", None, None, None, [out], None)


# ── beyond the brief's four ──────────────────────────────────────────────────
@pytest.mark.parametrize("where", ["", "paper", "stare/multiplex"])
def test_the_refusal_also_catches_the_folder_itself_and_anything_under_it(tmp_path, where):
    out = tmp_path / "out"
    out.mkdir()
    to = out / where if where else out
    with pytest.raises(SystemExit):
        collect.run(to, None, None, None, [out], None)
    assert not (out / "README.txt").exists()


def test_the_arm_row_table():
    m = collect.arm_row
    assert m("tiled_high_s64") == "stare"
    assert m("tiled_medium_s64") == "stare_medium"
    assert m("tiled_low_s64") == "stare_low"
    assert m("tiled_high_s128") == "stare_s128_none"
    assert m("tiled_high_s256") == "stare_s256_none"
    assert m("valis_high_micro2") == "valis_mirage"
    assert m("valis_medium_micro0") == "valis_mirage_medium_m0"
    assert m("valis_low_micro1") == "valis_mirage_low_m1"
    assert m("tiled_medium_s128") is None
    assert m("ashlar_s15") is None


def test_an_unmapped_arm_is_an_na_row_named_in_the_readme(tmp_path):
    _arm(tmp_path / "arms", "ashlar_s15", "P1", "DAPI_CD8", 0.5)
    collect.run(tmp_path / "c", tmp_path / "arms", None, None, [], None)
    (r,) = [x for x in schema.read(tmp_path / "c" / "per_pair.csv")
            if x["source"] == "arms:ashlar_s15"]
    assert r["row"] is None and r["dice_matched"] == "0.5"
    assert "ashlar_s15" in (tmp_path / "c" / "README.txt").read_text()


def test_the_most_registered_stage_without_a_full_transform(tmp_path):
    d = tmp_path / "arms" / "valis_high_micro2"
    (d / "csv").mkdir(parents=True)
    (d / "csv" / "registered.csv").write_text("patient_id,id,channels\nP1,slideA,DAPI|CD4\n")
    q = d / "P1" / "qc" / "registration"
    q.mkdir(parents=True)
    (q / "slideA_seg_qc.json").write_text(json.dumps({
        "moving": "slideA.ome.tif", "stage_order": ["native", "rigid", "micro"],
        "stages": {"native": {"dice_matched": 0.1}, "rigid": {"dice_matched": 0.5},
                   "micro": {"dice_matched": 0.6, "displacement_um_p50": 1.5}}}))
    collect.run(tmp_path / "c", tmp_path / "arms", None, None, [], None)
    (r,) = schema.read(tmp_path / "c" / "per_pair.csv")
    assert (r["panel"], r["dice_matched"], r["stage"]) == ("DAPI_CD4", "0.6", "micro")
    assert r["displacement_um_p50"] == "1.5"


def test_a_nan_dice_is_copied_as_found(tmp_path):
    d = tmp_path / "arms" / "tiled_high_s64"
    _arm(tmp_path / "arms", "tiled_high_s64", "P1", "DAPI_CD8", 0.0)
    f = d / "P1" / "qc" / "registration" / "P1_DAPI_CD8_seg_qc.json"
    f.write_text('{"moving": "P1_DAPI_CD8", "full_transform": {"dice_matched": NaN}, '
                 '"stages": {"rigid": {"dice_matched": 0.4}}}')
    collect.run(tmp_path / "c", tmp_path / "arms", None, None, [], None)
    (r,) = schema.read(tmp_path / "c" / "per_pair.csv")
    assert r["dice_matched"] == "nan"
    assert "NaN" in (tmp_path / "c" / "README.txt").read_text()


TRACE = ("task_id\tprocess\ttag\tname\tstatus\texit\tpeak_rss\trealtime\t%cpu\n"
         "1\tMIRAGE:REGISTER\tP1\tREGISTER (P1)\tCOMPLETED\t0\t2 GB\t1h\t200.0%\n"
         "2\tMIRAGE:REGISTER\tP2\tREGISTER (P2)\tFAILED\t137\t9 GB\t2h\t100.0%\n"
         "3\tMIRAGE:TILED_SOLVE\tP1\tTILED_SOLVE (P1)\tCOMPLETED\t0\t1 GB\t30m\t100.0%\n"
         "4\tMIRAGE:CONVERT_IMAGE\tP1\tCONVERT_IMAGE (P1)\tCOMPLETED\t0\t1 GB\t1m\t100.0%\n")


def test_arm_cost_points_from_the_trace(tmp_path):
    for arm in ("valis_high_micro2", "tiled_high_s64"):
        _arm(tmp_path / "arms", arm, "P1", "DAPI_CD8", 0.7)
        (tmp_path / "arms" / arm / "trace").mkdir()
        (tmp_path / "arms" / arm / "trace" / "trace.txt").write_text(TRACE)
    sizes = tmp_path / "arms" / "valis_high_micro2" / "size_logs"
    sizes.mkdir()
    (sizes / "input_sizes.csv").write_text(
        f"process,sample_id,bytes\nREGISTER,P1,{2**31}\nREGISTER,P2,{2**30}\n")
    collect.run(tmp_path / "c", tmp_path / "arms", None, None, [], None)
    pts = schema.read(tmp_path / "c" / "cost_points.csv")
    valis = [p for p in pts if p["source"] == "arms:valis_high_micro2"]
    tiled = [p for p in pts if p["source"] == "arms:tiled_high_s64"]
    (v,) = valis  # the FAILED task is not a cost point; CONVERT_IMAGE is not REGISTER
    assert v["row"] == "valis_mirage" and v["unit_id"] == "REGISTER:P1"
    assert float(v["wall_s"]) == 3600.0 and float(v["peak_rss_gb"]) == 2.0
    assert float(v["input_gb"]) == 1.5 and float(v["cpu_h_used"]) == 2.0
    # T31 P14: a tiled arm's point is its PATIENT (the TILED_* tasks summed), wall the
    # critical path with every task at once
    (t,) = tiled
    assert t["row"] == "stare" and t["unit_id"] == "P1" and t["unit_kind"] == "patient"
    assert t["clock"] == "critical_path_unlimited" and t["input_gb"] is None
    assert float(t["wall_s"]) == 1800.0 and float(t["cpu_h_used"]) == 0.5


def test_sweep_cost_points_are_flagged_synthetic(tmp_path):
    (tmp_path / "sweep" / "r01" / "trace").mkdir(parents=True)
    (tmp_path / "sweep" / "r01" / "trace" / "trace.txt").write_text(TRACE)
    plan = tmp_path / "plan.csv"
    plan.write_text("run_id,varied_axis\nr01,tile\nr02,tile\n")
    collect.run(tmp_path / "c", None, tmp_path / "sweep", plan, [], None)
    pts = schema.read(tmp_path / "c" / "cost_points.csv")
    assert pts and {p["dataset"] for p in pts} == {"synthetic_sweep"}
    assert {p["source"] for p in pts} == {"sweep:r01"}
    readme = (tmp_path / "c" / "README.txt").read_text()
    assert "synthetic sweep is removed from the paper" in readme and "r02" in readme


def _record(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = pickle.dumps({"key": {}, "payload": payload}, protocol=4)
    path.write_bytes(hashlib.sha256(body).hexdigest().encode() + b"\n" + body)


def _cell(case, method, dice, set_="outlines"):
    return {"dataset": "multiplex", "case_id": case, "group": "P1", "patient_id": "P1",
            "panel": "DAPI_CD8", "method": method, "set": set_, "dice_matched": dice,
            "n_moving": 10}


def test_case_records_become_per_pair_rows_and_a_damaged_one_is_skipped(tmp_path):
    rb = tmp_path / "rb"
    scores = rb / "cache" / "scores" / "multiplex"
    _record(scores / "P1_r1" / "case.rec",
            ([_cell("P1_r1", "stare", 0.9), _cell("P1_r1", "stare:non_ashlar", 0.1)], [], [], [],
             {"cells": [_cell("P1_r1", "stare", float("nan"), "nuclei")], "non_cells": [],
              "missing": []}))
    _record(scores / "P1_r2" / "case.rec", ([_cell("P1_r2", "stare", 0.5)], [], [], [], {}))
    damaged = scores / "P1_r2" / "case.rec"
    # a whole, loadable pickle under a sha256 line that is not its own: only the check
    # keeps it out
    body = damaged.read_bytes().partition(b"\n")[2]
    damaged.write_bytes(b"0" * 64 + b"\n" + body)
    collect.run(tmp_path / "c", None, None, None, [rb], None)
    rows = schema.read(tmp_path / "c" / "per_pair.csv")
    assert {(r["case_id"], r["set"], r["dice_matched"]) for r in rows} == {
        ("P1_r1", "outlines", "0.9"), ("P1_r1", "nuclei", "nan")}
    assert {r["source"] for r in rows} == {f"regbench-interim:{rb}"}
    assert [r["n_mov"] for r in rows] == ["10", "10"]
    assert "P1_r2" in (tmp_path / "c" / "README.txt").read_text()


def test_cells_csv_wins_over_the_case_records(tmp_path):
    rb = tmp_path / "rb"
    _record(rb / "cache" / "scores" / "multiplex" / "P1_r1" / "case.rec",
            ([_cell("P1_r1", "stare", 0.9)], [], [], [], {}))
    (rb / "tables").mkdir(parents=True)
    (rb / "tables" / "cells.csv").write_text(
        "dataset,case_id,group,panel,method,set,dice_matched\n"
        "multiplex,P1_r1,P1,DAPI_CD8,stare,outlines,0.42\n")
    collect.run(tmp_path / "c", None, None, None, [rb], None)
    (r,) = schema.read(tmp_path / "c" / "per_pair.csv")
    assert r["dice_matched"] == "0.42" and r["patient_id"] == "P1"


def _run_json(rb, row, case, **rec):
    d = rb / row / "multiplex" / case
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"variant": row, "dataset": "multiplex",
                                            "case_id": case, **rec}))


def test_regbench_cost_counts_one_launch_once(tmp_path):
    rb = tmp_path / "rb"
    res = {"resources": {"cpu_s": 7200, "peak_rss_mb": 2048}, "register_s": 100.0}
    for case in ("P1_r1", "P1_r2"):  # a per-patient launch: the same cost in every round
        _run_json(rb, "valis_mirage", case, ok=True, unit="patient", group="P1",
                  launch_id="L1", **res)
    _run_json(rb, "stare", "P1_r1", ok=True, unit="pair", group="P1", launch_id="L2", **res)
    _run_json(rb, "stare", "P1_r2", ok=False, unit="pair", group="P1", launch_id="L3")
    collect.run(tmp_path / "c", None, None, None, [rb], None)
    pts = schema.read(tmp_path / "c" / "cost_points.csv")
    got = {(p["row"], p["unit_kind"], p["unit_id"]) for p in pts}
    assert got == {("valis_mirage", "patient", "P1"), ("stare", "round", "P1_r1")}
    (v,) = [p for p in pts if p["row"] == "valis_mirage"]
    assert float(v["cpu_h_used"]) == 2.0 and float(v["peak_rss_gb"]) == 2.0
    assert float(v["wall_s"]) == 100.0


def test_expected_rows_are_replaced_by_found_ones_and_crosschecked(tmp_path, monkeypatch):
    case = SimpleNamespace(case_id="P1_r1", group="P1", dataset="multiplex",
                           extra={"channels_mov": "DAPI|CD8"})
    monkeypatch.setattr(collect, "_cases", lambda root: [case])
    monkeypatch.setattr(collect.tables, "terminal_rows",
                        lambda ds: {"stare": "stare", "deeperhistreg_gpu": "deeperhistreg_gpu"})
    rb = tmp_path / "rb"
    _record(rb / "cache" / "scores" / "multiplex" / "P1_r1" / "case.rec",
            ([{**_cell("P1_r1", "stare", 0.9), "displacement_um_p50": 0.5}], [], [], [], {}))
    _arm(tmp_path / "arms", "tiled_high_s64", "P1", "DAPI_CD8", 0.7)
    collect.run(tmp_path / "c", tmp_path / "arms", None, None, [rb], tmp_path / "cases")
    rows = schema.read(tmp_path / "c" / "per_pair.csv")
    expected = [r for r in rows if r["source"] == "expected"]
    assert [(r["row"], r["set"], r["dice_matched"]) for r in expected] == [
        ("deeperhistreg_gpu", "outlines", None)]
    assert expected[0]["panel"] == "DAPI_CD8" and expected[0]["case_id"] == "P1_r1"
    assert not [r for r in rows if r["row"] == "deeperhistreg_gpu_asdrawn"]
    (x,) = schema.read(tmp_path / "c" / "crosscheck_arms.csv")
    assert (x["row"], x["arm"], x["dice_regbench"], x["dice_arm"]) == (
        "stare", "tiled_high_s64", "0.9", "0.7")
    assert math.isclose(float(x["dice_diff"]), 0.2)
    assert x["disp_um_regbench"] == "0.5"


def test_the_readme_says_what_the_collection_is(tmp_path):
    counts = collect.run(tmp_path / "c", None, None, None, [], None)
    text = (tmp_path / "c" / "README.txt").read_text()
    assert "interim only: not regbench rows, never read by `score`" in text
    assert "every measurement still to come is NA" in text
    assert set(counts) == set(schema.TABLES) and not any(counts.values())


def test_the_cli_runs_it(tmp_path, capsys):
    from benchmarks.regbench import cli

    _arm(tmp_path / "arms", "tiled_high_s64", "P1", "DAPI_CD8", 0.81)
    assert cli.main(["collect", "--to", str(tmp_path / "c"), "--arms", str(tmp_path / "arms")]) == 0
    assert "per_pair\t1" in capsys.readouterr().out
    assert (tmp_path / "c" / "README.txt").exists()


def test_a_tiled_arm_is_one_point_per_patient_summed_over_its_tasks(tmp_path):
    """T31 P14: two tiles of one patient and its solve are ONE point: CPU-hours summed,
    wall the longest tile plus the solve, the peak the largest task's."""
    trace_text = (
        "task_id\tprocess\ttag\tname\tstatus\texit\tpeak_rss\trealtime\t%cpu\n"
        "1\tMIRAGE:TILED_REG_TILE\tP1:0_0\tT (P1)\tCOMPLETED\t0\t1 GB\t10m\t100.0%\n"
        "2\tMIRAGE:TILED_REG_TILE\tP1:0_1\tT (P1)\tCOMPLETED\t0\t3 GB\t20m\t100.0%\n"
        "3\tMIRAGE:TILED_SOLVE\tP1:DAPI_CD8\tS (P1)\tCOMPLETED\t0\t2 GB\t30m\t200.0%\n"
        "4\tMIRAGE:TILED_SOLVE\tP2:DAPI_CD8\tS (P2)\tCOMPLETED\t0\t2 GB\t30m\t\n")
    _arm(tmp_path / "arms", "tiled_high_s64", "P1", "DAPI_CD8", 0.7)
    (tmp_path / "arms" / "tiled_high_s64" / "trace").mkdir()
    (tmp_path / "arms" / "tiled_high_s64" / "trace" / "trace.txt").write_text(trace_text)
    collect.run(tmp_path / "c", tmp_path / "arms", None, None, [], None)
    pts = {p["unit_id"]: p for p in schema.read(tmp_path / "c" / "cost_points.csv")
           if p["source"] == "arms:tiled_high_s64"}
    assert set(pts) == {"P1", "P2"}
    p1 = pts["P1"]
    assert float(p1["wall_s"]) == 20 * 60 + 30 * 60  # the longest tile, then the solve
    assert float(p1["cpu_h_used"]) == pytest.approx((10 + 20 + 60) / 60)
    assert float(p1["peak_rss_gb"]) == 3.0
    assert pts["P2"]["cpu_h_used"] is None  # a task with no %cpu: no figure
