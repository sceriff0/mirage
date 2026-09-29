"""arms_status.py reads the benchmark's state off disk + squeue, one status per run.

A fake bench dir (a real plan, hand-written histories and traces) and a fake `squeue`
whose jobs sit in launch-dir work dirs. Pinned: finished / failed / interrupted / waiting
/ running come from the right source; a QC cross sharing its base's launch dir is only
RUNNING when it was the last launch there; a pinned row under another commit is CODE≠;
ASHLAR is judged by its own marker; nothing is written.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from benchmarks import arms_status as st

PLAN = [
    # run_id, arm_kind, method, resume_run, code_ref
    ("preprocess_shared", "preprocess", "preprocess", "", ""),
    ("valis_high_micro2", "registration", "valis", "", ""),
    (
        "valis_high_micro2_segstardist",
        "registration_qc",
        "valis",
        "valis_high_micro2",
        "",
    ),
    (
        "valis_high_micro2_pairmutual_nn",
        "registration_qc",
        "valis",
        "valis_high_micro2",
        "",
    ),
    ("tiled_high_s128", "registration", "drape", "", ""),
    ("tiled_low_s64", "registration", "drape", "", ""),
    ("tiled_high_gate1", "registration", "stare", "", "a" * 40),
    ("tiled_low_gate1", "registration", "stare", "", "a" * 40),
    ("ashlar_t1024_s240", "external", "ashlar", "", ""),
    ("seg_cellsam", "segmentation", "seg", "", ""),
]


def _hist(results: Path, launch_dir: str, *runs: tuple[str, str]) -> Path:
    d = results / ".launch" / launch_dir
    (d / ".nextflow").mkdir(parents=True, exist_ok=True)
    with open(d / ".nextflow" / "history", "a") as fh:
        for name, status in runs:
            fh.write(
                f"2026-09-29 10:00:00\t1h\t{name}\t{status}\tx\tsess\tnextflow run\n"
            )
    return d


@pytest.fixture
def bench(tmp_path, monkeypatch):
    b = tmp_path / "benchmark"
    res = b / "arm_results"
    res.mkdir(parents=True)
    with open(b / "arm_plan.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "arm_kind", "arm", "method", "resume_run", "code_ref"])
        for run_id, kind, method, base, ref in PLAN:
            w.writerow([run_id, kind, run_id, method, base, ref])
    _hist(res, "preprocess_shared", ("arms-preprocess_shared", "OK"))
    # base OK, then cross 1 interrupted, then cross 2 launched LAST and running now
    _hist(
        res,
        "valis_high_micro2",
        ("arms-valis_high_micro2", "OK"),
        ("arms-valis_high_micro2_segstardist", "-"),
        ("arms-valis_high_micro2_pairmutual_nn", "-"),
    )
    _hist(res, "tiled_high_s128", ("arms-tiled_high_s128", "ERR"))
    old = _hist(
        res,
        "tiled_low_s64",
        ("arms-tiled_low_s64", "-"),
        ("arms-tiled_low_s64-r2", "-"),
    )
    (old / ".nextflow.log").write_text("x")
    os.utime(old / ".nextflow.log", (1, 1))  # long quiet: interrupted
    d = _hist(res, "tiled_high_gate1", ("arms-tiled_high_gate1", "OK"))
    (d / "code.tiled_high_gate1").write_text("b" * 40 + "\n")
    (res / "ashlar_t1024_s240").mkdir()
    (res / "ashlar_t1024_s240" / ".external_done").write_text("t")
    tr = res / "tiled_low_s64" / "trace"
    tr.mkdir(parents=True)
    (tr / "trace.txt").write_text(
        "task_id\tprocess\tstatus\n1\tA\tCOMPLETED\n2\tB\tCOMPLETED\n3\tC\tFAILED\n4\tD\tCACHED\n"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    wd = res / ".launch" / "valis_high_micro2" / "work" / "ab" / "cdef"
    (bindir / "squeue").write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *"%i %j"* ]]; then echo "7300001 mirage_arms RUNNING 1:00:00 node03"; '
        f'else echo "RUNNING|{wd}"; echo "PENDING|{wd}"; echo "RUNNING|/elsewhere/work/x"; fi\n'
    )
    (bindir / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    return b


def test_each_run_gets_its_status_from_the_right_source(bench):
    recs = {
        r["run_id"]: r
        for r in st.collect(bench, bench / "arm_results", bench / "arm_plan.csv")
    }
    status = {k: v["status"] for k, v in recs.items()}
    assert status == {
        "preprocess_shared": "DONE",
        "valis_high_micro2": "DONE",
        "valis_high_micro2_segstardist": "INTERRUPTED",  # not the last launch in the dir
        "valis_high_micro2_pairmutual_nn": "RUNNING",  # last launch + jobs in its dir
        "tiled_high_s128": "FAILED",
        "tiled_low_s64": "INTERRUPTED",
        "tiled_high_gate1": "CODE≠",
        "tiled_low_gate1": "WAITING",
        "ashlar_t1024_s240": "DONE",
        "seg_cellsam": "WAITING",
    }
    assert recs["valis_high_micro2_pairmutual_nn"]["jobs"] == {
        "RUNNING": 1,
        "PENDING": 1,
    }
    assert recs["tiled_low_s64"]["attempts"] == 2
    t = recs["tiled_low_s64"]["tasks"]
    assert (t["COMPLETED"], t["FAILED"], t["CACHED"]) == (2, 1, 1)


def test_the_report_summarises_by_method_and_says_what_to_do(bench, capsys):
    before = sorted(p for p in bench.rglob("*"))
    assert st.main([str(bench)]) == 0
    out = capsys.readouterr().out
    assert "7300001 mirage_arms" in out
    assert "1 running, 1 pending" in out
    assert "30% of runs done" in out  # 3 of 10
    assert "valis_high_micro2_segstardist" in out and "ARMS_REPLACE=1" in out
    assert "tiled_high_s128: " in out  # a failed run's log is named
    assert sorted(p for p in bench.rglob("*")) == before, "status must be read-only"


def test_method_filter_and_all(bench, capsys):
    st.main([str(bench), "--method", "stare", "--all"])
    out = capsys.readouterr().out
    assert "tiled_high_gate1" in out and "tiled_low_gate1" in out
    assert "valis_high_micro2" not in out
