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
