# benchmarks/tests/regbench/test_local.py
"""``regbench local``: STARE's stages on one cohort slide on a laptop (``local.py``)."""
import csv
import json
import os
import platform

import pytest

from benchmarks.regbench import cli, local, methods, score
from benchmarks.regbench.paper import schema, tables
from benchmarks.tests.regbench import scored_cohort

CASE = "P1_r1"


def test_machine_never_raises_and_has_every_key(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no")

    for name in ("system", "platform", "python_version"):
        monkeypatch.setattr(platform, name, boom)
    monkeypatch.setattr(os, "cpu_count", boom)
    monkeypatch.setattr(local, "_sysctl", boom)
    monkeypatch.setattr(local, "_read_text", boom)
    spec = local.machine()
    assert set(spec) == set(local.KEYS)
    assert spec["cpu_model"] is None and spec["os"] is None


def test_machine_on_linux_reads_proc(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    files = {
        "/proc/cpuinfo": "processor\t: 0\nmodel name\t: Fake Core 9\nmodel name\t: other\n",
        "/proc/meminfo": "MemTotal:       16777216 kB\n",
    }
    monkeypatch.setattr(local, "_read_text", lambda p: files.get(str(p)))
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    spec = local.machine()
    assert set(spec) == set(local.KEYS)
    assert spec["cpu_model"] == "Fake Core 9"
    assert spec["n_cores_logical"] == 8
    assert spec["os"] and spec["python"]
    assert spec["mem_total_gb"] is not None


def test_machine_on_darwin_asks_sysctl(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    asked = {
        "machdep.cpu.brand_string": "Apple M9",
        "hw.memsize": str(32 * 1024**3),
    }
    monkeypatch.setattr(local, "_sysctl", lambda k: asked.get(k))
    spec = local.machine()
    assert spec["cpu_model"] == "Apple M9"
    assert set(spec) == set(local.KEYS)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    root, out, made = scored_cohort.build(tmp_path)
    fake = scored_cohort.forward("stare", 0.4)
    real = methods.load
    monkeypatch.setattr(methods, "load", lambda m: fake if m == "stare" else real(m))
    return root, out


def _args(root, out, *extra, run="stare"):
    return cli.build_parser().parse_args(
        ["local", "--cases", str(root), "--out", str(out), "--case-id", CASE,
         "--run", run, *extra]
    )


def test_local_writes_a_laptop_result_with_the_machine_and_submission(cohort):
    root, out = cohort
    assert local.main(_args(root, out, "--workers", "2", "--submission", "S7")) == 0
    rec = json.loads((out / "stare_laptop" / "multiplex" / CASE / "run.json").read_text())
    assert set(rec["machine"]) == set(local.KEYS)
    assert rec["submissions"] == ["S7"]
    assert rec["workers"] == 2 and rec["label"] == "stare_laptop"
    assert not (out / "stare_laptop" / "multiplex" / CASE / "machine.json").exists()
    assert "REGBENCH_STAGES" not in os.environ or os.environ["REGBENCH_STAGES"] != "local"


def test_local_refuses_a_non_stare_run_and_a_missing_case(cohort, capsys):
    root, out = cohort
    with pytest.raises(SystemExit) as e:
        local.main(_args(root, out, run="valis_mirage"))
    assert e.value.code == 2
    bad = _args(root, out)
    bad.case_id = "nope"
    with pytest.raises(SystemExit) as e:
        local.main(bad)
    assert e.value.code == 2


def test_scored_as_a_row_never_compared(cohort, monkeypatch):
    root, out = cohort
    assert local.main(_args(root, out, "--submission", "S7")) == 0
    monkeypatch.setenv(score.SUBMISSION_ENV, "S7")  # the submission it belongs to
    score.score(root, out, **scored_cohort.SCORE_KW)
    cells = list(csv.DictReader(open(out / "tables" / "cells.csv")))
    mine = [r for r in cells if r["method"] == "stare_laptop" and r["case_id"] == CASE]
    assert mine and all(r["excluded_from_summaries"] in ("False", "") for r in mine)
    assert all(r["cause"] != "not_run" for r in mine)
    # every other row is not_run in S7 and excluded (ruling 2026-10-21), so a paired
    # table may have no row at all and is then not written
    for name in ("common_block.csv", "paired_vs_primary.csv"):
        path = out / "tables" / name
        assert not path.exists() or "stare_laptop" not in path.read_text(), name
    built = tables.build(out, "multiplex")
    rows = built["laptop"]
    assert [r["row"] for r in rows] == ["stare_laptop"] and rows[0]["case_id"] == CASE
    assert set(rows[0]) == set(schema.columns("laptop"))
    assert rows[0]["n_cores_logical"] is not None and rows[0]["dice_matched"] > 0
    tables.write_all(out, "multiplex")
    header = (out / "paper" / "laptop.csv").read_text().splitlines()[0].split(",")
    assert header == schema.columns("laptop")


def test_another_submission_does_not_score_it(cohort, monkeypatch):
    root, out = cohort
    local.main(_args(root, out, "--submission", "S7"))
    monkeypatch.setenv(score.SUBMISSION_ENV, "S8")
    score.score(root, out, **scored_cohort.SCORE_KW)
    cells = list(csv.DictReader(open(out / "tables" / "cells.csv")))
    mine = [r for r in cells if r["method"] == "stare_laptop" and r["case_id"] == CASE]
    assert all(r["cause"] == "not_run" for r in mine)
