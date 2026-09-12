"""The ASHLAR external arms get resource rows through benchmarks/trace_step.py.

Every Nextflow arm writes <root>/<arm>/trace/trace.txt for free; the ashlar arms
run outside Nextflow (benchmarks/run_ashlar_arm.sh) and wrote none, so ASHLAR sat
in the accuracy table with no cost column and no error saying so. These tests pin
the writer to the reader: a row the wrapper writes parses through
load.parse_trace with the numbers it measured, a failing child records FAILED and
propagates its exit status, run_ashlar_arm.sh routes every heavy invocation
through the wrapper, and an ashlar arm with a trace but NO size log (input_gb NaN)
lands in runs_master / param_matrix / measurements.csv without any analysis
change.

Watched failing: the parse test against a writer emitting bytes as a bare integer
with a `B` unit the regex did not accept, and the script test against the
un-wrapped run_ashlar_arm.sh.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from benchmarks import trace_step
from benchmarks.analysis import make_figures, make_tables
from benchmarks.analysis.lib import load
from benchmarks.tests.test_subset_rerun_equivalence import (
    FIXTURE_ARMS,
    _plan_file,
    build_root,
)

BENCH = Path(__file__).parents[1]
REPO = BENCH.parent
PY = sys.executable

_ALLOC = "import time; x = bytearray(60_000_000); x[-1] = 1; time.sleep(0.25)"


def _wrap(trace: Path, process: str, tag: str, cmd: list[str], *extra):
    return subprocess.run(
        [
            PY,
            str(BENCH / "trace_step.py"),
            "--trace",
            str(trace),
            "--process",
            process,
            "--tag",
            tag,
            "--cpus",
            "3",
            *extra,
            "--",
            *cmd,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def test_a_written_row_parses_back_with_the_numbers_it_measured(tmp_path):
    trace = tmp_path / "arm" / "trace" / "trace.txt"
    big = tmp_path / "input.bin"
    big.write_bytes(b"\0" * 12345)
    r = _wrap(trace, "ASHLAR_SOLVE", "P001", [PY, "-c", _ALLOC], "--input", str(big))
    assert r.returncode == 0, r.stderr
    df = load.parse_trace(trace)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["process"] == "ASHLAR_SOLVE" and row["tag"] == "P001"
    assert row["status"] == "COMPLETED" and row["exit"] == 0
    assert row["cpus"] == 3
    assert row["realtime_s"] >= 0.25 and row["duration_s"] >= 0.25
    assert row["peak_rss_gb"] >= 60_000_000 / 2**30 * 0.9, row["peak_rss_gb"]
    assert pd.isna(row["peak_vmem_gb"]) and pd.isna(row["read_gb"])
    assert pd.notna(row["start_ts"]) and pd.notna(row["complete_ts"])
    assert (row["complete_ts"] - row["start_ts"]).total_seconds() >= 0.25
    # the size log, in the pipeline's own format, one directory up from trace/
    sizes = load.parse_size_logs(tmp_path / "arm" / "size_logs" / "input_sizes.csv")
    assert sizes.loc["ASHLAR_SOLVE", "bytes"] == 12345
    # and the raw file is a Nextflow trace: tab-separated, the fixture's header
    header = trace.read_text().splitlines()[0].split("\t")
    fixture = BENCH / "tests" / "fixtures" / "runs" / "run0000" / "trace" / "trace.txt"
    assert header == fixture.read_text().splitlines()[0].split("\t")


def test_a_failing_child_is_recorded_failed_and_its_status_propagates(tmp_path):
    trace = tmp_path / "trace.txt"
    ok = _wrap(trace, "ASHLAR_RETILE", "P001", [PY, "-c", "pass"])
    bad = _wrap(trace, "ASHLAR_RETILE", "P002", [PY, "-c", "raise SystemExit(3)"])
    assert ok.returncode == 0 and bad.returncode == 3
    df = load.parse_trace(trace)
    assert list(df["tag"]) == ["P001", "P002"]
    assert list(df["status"]) == ["COMPLETED", "FAILED"]
    assert list(df["exit"]) == [0, 3]
    assert [ln.split("\t")[0] for ln in trace.read_text().splitlines()[1:]] == [
        "1",
        "2",
    ]
    # only_successful drops the failed row and keeps the good one
    df["run_id"] = "ashlar_t1024_s30"
    assert list(load.only_successful(df)["tag"]) == ["P001"]


def test_formatters_round_trip_through_the_parsers():
    from benchmarks.analysis.lib.parsing import parse_duration, parse_to_gb

    for n in (512, 3 * 2**20, 1.7 * 2**30, 12 * 2**30):
        assert (
            abs(parse_to_gb(trace_step.fmt_bytes(n)) - n / 2**30)
            < 0.05 * n / 2**30 + 1e-9
        )
    for s in (0.12, 1.5, 65.0, 3725.0):
        assert abs(parse_duration(trace_step.fmt_duration(s)) - s) < 0.6


def test_run_ashlar_arm_routes_every_heavy_invocation_through_the_wrapper():
    """Comment-blind: the script's prose names the bare commands it replaced."""
    code = "\n".join(
        ln
        for ln in (BENCH / "run_ashlar_arm.sh").read_text().splitlines()
        if not ln.lstrip().startswith("#")
    )
    for proc in ("ASHLAR_RETILE", "ASHLAR_SOLVE", "ASHLAR_SEG_QC"):
        assert re.search(rf"^\s*step {proc} ", code, re.M), f"{proc} is not wrapped"
    # no bare $ASHLAR_EXEC / $QC_EXEC invocation survives outside a `step ... -- \` line
    for ln in code.splitlines():
        if "$ASHLAR_EXEC python3" in ln or "$QC_EXEC python3" in ln:
            assert ln.strip().startswith("$"), ln
    joined = code.replace("\\\n", " ")
    for ln in joined.splitlines():
        if "$ASHLAR_EXEC python3" in ln or "$QC_EXEC python3" in ln:
            assert re.match(r"^\s*step ASHLAR_\w+ ", ln), (
                f"unwrapped: {ln.strip()[:80]}"
            )
    assert 'TRACE="$OUT/trace/trace.txt"' in code, "must land where load_runs reads it"
    assert 'trace_step.py" --trace "$TRACE"' in code


def test_an_ashlar_arm_with_a_trace_and_no_size_log_lands_in_the_tables(tmp_path):
    """Written by the real wrapper, into the real arm layout, beside the other
    arms' synthetic results -- and read with NO analysis change."""
    from benchmarks.build_arm_plan import build_arm_plan

    plan = build_arm_plan(FIXTURE_ARMS)
    root = tmp_path / "root"
    build_root(root, plan, lambda r: 1)
    ashlar = next(r["arm"] for r in plan if r["arm_kind"] == "external")
    trace = root / ashlar / "trace" / "trace.txt"
    assert not trace.exists() and not (root / ashlar / "size_logs").exists()
    for pat in ("P001", "P002"):
        for proc in ("ASHLAR_RETILE", "ASHLAR_SOLVE", "ASHLAR_SEG_QC"):
            r = _wrap(trace, proc, pat, [PY, "-c", "pass"])
            assert r.returncode == 0, r.stderr
    assert not (root / ashlar / "size_logs").exists(), "no --input, no size log"
    plan_csv = _plan_file(tmp_path, plan)

    runs = load.load_runs(root, plan_csv)
    mine = runs[runs["run_id"] == ashlar]
    assert len(mine) == 6 and mine["input_gb"].isna().all()
    assert set(mine["process"]) == {"ASHLAR_RETILE", "ASHLAR_SOLVE", "ASHLAR_SEG_QC"}

    out = tmp_path / "tables"
    make_tables.build_paper_data(root, plan_csv, out)
    master = pd.read_csv(out / "runs_master.csv").set_index("run_id")
    assert ashlar in master.index
    assert master.loc[ashlar, "cpu_hours"] >= 0
    assert "ASHLAR_SOLVE_peak_ram_gb" in master.columns
    assert pd.notna(master.loc[ashlar, "ASHLAR_SOLVE_wall_s"])
    pm = pd.read_csv(out / "param_matrix.csv").set_index("run_id")
    # cost AND accuracy on the same row: the whole point of scoring ashlar in-tree
    assert pd.notna(pm.loc[ashlar, "reg_dice_matched"])
    assert pd.notna(pm.loc[ashlar, "cpu_hours"])
    fits = pd.read_csv(out / "scaling_fits.csv")
    assert not fits.empty, "NaN input_gb on one arm must not empty the fits"
    # no input size -> the documented flat fallback (n=0, r2 NaN), never a crash
    solve = fits[fits["stage"] == "ASHLAR_SOLVE"]
    assert len(solve) == 2 and (solve["n"] == 0).all() and solve["r2"].isna().all()

    figs = tmp_path / "figs"
    make_figures.run(root, plan_csv, "none", figs, formats=())
    meas = pd.read_csv(figs / "measurements.csv")
    assert (meas["run_id"] == ashlar).sum() == 6
    stats = pd.read_csv(figs / "resource_stats.csv")
    assert "ASHLAR_SOLVE" in set(stats["process"])
