# benchmarks/tests/regbench/test_cpu_scaling.py
import csv
import json
import math

import pytest

from benchmarks.regbench.paper import cpu_scaling, schema

COLS = "dataset,group,method,register_s,cpu_h_used,peak_rss_gb,complete,n_jobs,wall_unlimited_parallel_s"


def folder(root, name, rows, executions=None):
    """A results folder: tables/summary_resources.csv, and one STARE run.json per row
    named in ``executions`` ({row: reuse-key execution, or None for a record without it,
    plus 'jobs:<x>' to put it only under info.jobs})."""
    out = root / name
    (out / "tables").mkdir(parents=True)
    with open(out / "tables" / "summary_resources.csv", "w", newline="") as fh:
        fh.write(COLS + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    for row, how in (executions or {}).items():
        d = out / row / "multiplex" / "c1"
        d.mkdir(parents=True)
        rec = {"method": "stare" if row.startswith("stare") else row}
        if how.startswith("jobs:"):
            rec["info"] = {"jobs": {"execution": how[5:]}}
        else:
            rec["reuse_key"] = {"execution": how}
        (d / "run.json").write_text(json.dumps(rec))
    return out


def row(patient, method, wall, cpu_h=1.0, rss=2.0, complete="True", n_jobs="", unl=""):
    return ["multiplex", patient, method, wall, cpu_h, rss, complete, n_jobs, unl]


@pytest.fixture
def points(tmp_path):
    c4 = folder(tmp_path, "c4", [
        row("P1", "valis_mirage", 100), row("P2", "valis_mirage", 90),
        row("P3", "valis_mirage", 80), row("P4", "valis_mirage", 70),
        row("P1", "stare", 999, n_jobs=5, unl=40), row("P1", "other", 5),
    ], {"stare": "jobs"})
    l4 = folder(tmp_path, "l4", [row("P1", "stare", 200)], {"stare": "local"})
    c8 = folder(tmp_path, "c8", [
        row("P1", "valis_mirage", 50), row("P3", "valis_mirage", 40),
        row("P4", "valis_mirage", 35, complete="False"),
        row("P1", "stare", 120),
    ], {"stare": "jobs:local"})
    return {4: [c4, l4], 8: [c8]}


def test_speedup_efficiency_and_present_in_every_count(points):
    rows = cpu_scaling.build(points, ["valis_mirage"])
    patients = {r["patient_id"] for r in rows}
    assert patients == {"P1", "P3"}  # P2 only at 4; P4 incomplete at 8
    p1 = {r["cpus"]: r for r in rows if r["patient_id"] == "P1"}
    assert p1[4]["speedup_vs_4"] == pytest.approx(1.0)
    assert p1[8]["speedup_vs_4"] == pytest.approx(100 / 50)
    assert p1[8]["efficiency"] == pytest.approx(1.0)
    assert p1[8]["execution"] == "one process" and p1[8]["wall_s"] == 50
    assert p1[8]["family"] and p1[8]["row"] == "valis_mirage"


def test_stare_executions_never_mix(points):
    rows = [r for r in cpu_scaling.build(points, ["stare"])]
    by = {(r["execution"], r["cpus"]): r for r in rows}
    assert by[("jobs", 4)]["wall_s"] == 40  # wall_unlimited_parallel_s for a jobs row
    assert by[("local", 4)]["wall_s"] == 200
    assert by[("local", 8)]["speedup_vs_4"] == pytest.approx(200 / 120)  # info.jobs fallback
    assert by[("local", 8)]["efficiency"] == pytest.approx(200 / 120 / 2)
    assert ("jobs", 8) not in by


def test_rows_not_named_are_left_out_and_columns_are_the_schema(points, tmp_path):
    rows = cpu_scaling.build(points, ["valis_mirage"])
    assert {r["row"] for r in rows} == {"valis_mirage"}
    assert set(rows[0]) == set(schema.columns("cpu_scaling"))
    p = tmp_path / "o.csv"
    schema.write(p, "cpu_scaling", rows)
    assert len(list(csv.DictReader(open(p)))) == len(rows)


def test_no_4_cpu_point_gives_no_speedup(tmp_path):
    c8 = folder(tmp_path, "c8", [row("P1", "valis_mirage", 50)])
    rows = cpu_scaling.build({8: [c8]}, ["valis_mirage"])
    assert math.isnan(rows[0]["speedup_vs_4"])


def test_cli_writes_the_table(points, tmp_path):
    from benchmarks.regbench import cli

    args = ["cpu-scaling", "--rows", "valis_mirage,stare", "--to", str(tmp_path / "p" / "cpu_scaling.csv")]
    for cpus, outs in points.items():
        for o in outs:
            args += ["--point", f"{cpus}={o}"]
    assert cli.main(args) == 0
    got = list(csv.DictReader(open(tmp_path / "p" / "cpu_scaling.csv")))
    assert {r["execution"] for r in got} == {"one process", "jobs", "local"}
    assert list(got[0]) == schema.columns("cpu_scaling")


# ── the 2026-10-18 ruling (T31 R12, R13) ─────────────────────────────────────
RES_COLS = "dataset,group,method,cpus_requested,cpu_model"


def resources(out, lines):
    """``tables/resources.csv`` with each record's CPU request and CPU model."""
    with open(out / "tables" / "resources.csv", "w", newline="") as fh:
        fh.write(RES_COLS + "\n")
        for patient, method, cpus, model in lines:
            fh.write(f"multiplex,{patient},{method},{cpus},{model}\n")


def test_a_gpu_row_is_refused(tmp_path):
    c4 = folder(tmp_path, "c4", [row("P1", "deeperhistreg_gpu", 10)])
    with pytest.raises(SystemExit, match="GPU"):
        cpu_scaling.build({4: [c4]}, ["deeperhistreg_gpu"])


def test_each_repeat_folder_is_its_own_point(tmp_path):
    root = tmp_path / "c4"
    for k, wall in ((1, 100), (2, 110)):
        folder(root, f"rep{k}", [row("P1", "valis_mirage", wall)])
    (root / "rep3").mkdir()  # no tables/: not a repeat
    c8 = folder(tmp_path, "c8", [row("P1", "valis_mirage", 50)])
    assert [k for k, _ in cpu_scaling.repeats(root)] == [1, 2]
    assert cpu_scaling.repeats(c8) == [(1, c8)]
    rows = cpu_scaling.build({4: [root], 8: [c8]}, ["valis_mirage"])
    at4 = {r["repeat"]: r["wall_s"] for r in rows if r["cpus"] == 4}
    assert at4 == {1: 100, 2: 110}
    # speed-up only inside one repeat: repeat 1 at 8 against repeat 1 at 4
    p8 = [r for r in rows if r["cpus"] == 8]
    assert len(p8) == 1 and p8[0]["speedup_vs_4"] == pytest.approx(100 / 50)


def test_every_point_checks_its_cpus_and_cpu_model_and_counts_the_incomplete(points):
    c4, c8 = points[4][0], points[8][0]
    resources(c4, [("P1", "valis_mirage", 4, "Xeon A"), ("P3", "valis_mirage", 4, "Xeon A")])
    resources(c8, [("P1", "valis_mirage", 8, "Xeon B"), ("P3", "valis_mirage", 4, "Xeon A")])
    rows = cpu_scaling.build(points, ["valis_mirage"])
    by = {(r["patient_id"], r["cpus"]): r for r in rows}
    assert by[("P1", 4)]["cpus_checked"] == "yes" and by[("P1", 8)]["cpus_checked"] == "yes"
    assert by[("P3", 8)]["cpus_checked"] == "no: 4"  # asked for 4 at the 8-CPU point
    assert by[("P1", 8)]["cpu_model"] == "Xeon B"
    assert by[("P1", 4)]["cpu_model_matches"] == "no"  # A at 4, B at 8
    assert by[("P3", 4)]["cpu_model_matches"] == "yes"
    # P2 ran at 4 only, P4 is incomplete at 8: two patients out, counted on every row
    assert {r["n_patients_incomplete"] for r in rows} == {2}


def test_a_point_with_no_record_of_its_cpus_is_not_checked(points):
    rows = cpu_scaling.build(points, ["valis_mirage"])
    assert {r["cpus_checked"] for r in rows} == {"no: none recorded"}


def test_the_cross_rows_are_each_familys_best_and_deeperhistregs_on_cpu(monkeypatch, tmp_path):
    from benchmarks.regbench.paper import tables

    asked = {}

    def best_of(pp, fam, dataset, cost=None, rule=None, candidates=None):
        asked[fam] = candidates
        return {"best_row": {"deeperhistreg": "deeperhistreg", "stare": "stare_s128_none"}[fam]}

    monkeypatch.setattr(tables, "per_pair", lambda out, dataset: [{"x": 1}])
    monkeypatch.setattr(tables, "_cost_rows", lambda out, dataset: [])
    monkeypatch.setattr(tables, "_families", lambda dataset: ["deeperhistreg", "stare"])
    monkeypatch.setattr(tables, "best_of", best_of)
    got = cpu_scaling.cross_rows(tmp_path)
    assert got == [("deeperhistreg", "deeperhistreg"), ("stare", "stare_s128_none")]
    only = asked["deeperhistreg"]
    assert only is not None and asked["stare"] is None
    assert only("deeperhistreg") and not only("deeperhistreg_gpu")


def test_cross_rows_refuses_an_unscored_folder(tmp_path):
    with pytest.raises(SystemExit, match="cross-runs"):
        cpu_scaling.cross_rows(tmp_path)
