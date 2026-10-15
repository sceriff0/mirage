"""A registration in several scheduler jobs per round (``regbench.stages``), and STARE run
that way: the pipeline's allocation and retry rule read from the pipeline's own files,
the stage commands, the records, the cost of a round, and the one path to the
transform whoever starts the stage processes (the scheduler, or the run itself).

The real registrations are on the selfcheck's 2048 x 2560 pair (two tiles), made once
for the module.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib
import inspect
import io
import json
import math
import os
import re
import shutil
import signal
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import ladder, methods, stages
from benchmarks.regbench import selfcheck as sc
from benchmarks.regbench.cases import load_case, result_dir, write_case
from benchmarks.regbench.methods import stare as adapter

stare = pytest.importorskip("stare")

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))
nfmodel = importlib.import_module("nfmodel")
lex = importlib.import_module("nfmodel._lex")
PIPELINE = (stages.COARSE, stages.TILE, stages.SOLVE, stages.STITCH)
MODULE = {
    stages.COARSE: "tiled_coarse",
    stages.TILE: "tiled_reg_tile",
    stages.SOLVE: "tiled_solve",
    stages.STITCH: "tiled_stitch",
}


# ── the allocation and the retry rule are the pipeline's ──────────────────────
def config():
    return lex.strip_comments((REPO / "conf" / "modules.config").read_text())


def label_block(name):
    text = config()
    start = re.search(rf"withLabel:\s*'{name}'\s*\{{", text).end()
    return text[start : lex.block_extent(text, start)]


def name_block(process):
    (block,) = [b for b in nfmodel.with_name_blocks() if b.names == [process]]
    return lex.strip_comments(block.raw_body)


def high_tier():
    text = (REPO / "lib" / "RegPresets.groovy").read_text()
    table = re.search(r"\bSTARE\s*=\s*\[(.*?)\n\s*\]", text, re.S).group(1)
    row = re.search(r"\bhigh\s*:\s*\[([^\]]*)\]", table).group(1)
    return {k: int(v) for k, v in re.findall(r"(\w+)\s*:\s*(\d+)", row)}


FORMULA = (
    r"Math\.ceil\(Math\.max\((\d+)d, \((\d*\.?\d+)d \+ (\d*\.?\d+)d \* mpx\) \* "
    r"(\d+)d\)\) as int\)\.GB \* task\.attempt"
)


def pipeline_memory_gb(stage):
    """The first attempt's memory of a process at the high tier and stride 64, worked
    out from the text of its ``memory`` closure in conf/modules.config."""
    tier, body = high_tier(), name_block(stages.ALLOCATION[stage]["process"])
    if stage == stages.SOLVE:
        return int(re.search(r"memory\s*=\s*\{\s*(\d+)\.GB \* task\.attempt", body)[1])
    floor, a, b, k = (float(v) for v in re.search(FORMULA, body).groups())
    if stage == stages.COARSE:
        assert "def mpx  = ((_dim as long) * (_dim as long)) / 1_000_000d" in body
        side = tier["coarse_max_dim"]
    elif stage == stages.TILE:
        times = int(
            re.search(r"Math\.max\(_halo, (\d+) \* \(params\.reg_tiled_stride", body)[1]
        )
        assert "def win = ((_tile + 2 * _margin) as long)" in body
        stride = adapter.TIER_HIGH_STRIDE64["stride"]
        side = tier["tile"] + 2 * max(tier["halo"], times * stride)
    else:
        assert "def win = (_out as long)" in body
        side = tier["out_tile"]
    return math.ceil(max(floor, (a + b * side * side / 1e6) * k))


@pytest.mark.parametrize("stage", PIPELINE)
def test_each_job_has_the_pipelines_allocation(stage):
    row = stages.ALLOCATION[stage]
    module = lex.strip_comments(
        (REPO / "modules" / "local" / f"{MODULE[stage]}.nf").read_text()
    )
    assert f"process {row['process']} " in module
    assert re.search(r"\blabel\s+'(\w+)'", module)[1] == row["label"]
    label = label_block(row["label"])
    # the process's own block sets memory and nothing else: the label owns CPUs and time
    body = name_block(row["process"])
    assert re.search(r"\bmemory\s*=", body)
    assert not re.search(r"\b(cpus|time|errorStrategy|maxRetries)\s*=", body)
    assert int(re.search(r"cpus\s*=\s*\{\s*(\d+)\s*\}", label)[1]) == row["cpus"]
    hours = re.search(r"time\s*=\s*\{\s*(\d+)\.h\s*\*\s*task\.attempt\s*\}", label)[1]
    assert int(hours) == row["hours"]
    assert pipeline_memory_gb(stage) == row["mem_gb"]
    for attempt in (1, 2, 3, 4):  # memory and time x attempt, as there
        got = stages.allocation(stage, attempt)
        assert got["cpus"] == row["cpus"]
        assert got["mem_gb"] == row["mem_gb"] * attempt
        assert got["hours"] == row["hours"] * attempt
        days, clock = got["time"].split("-")
        assert int(days) * 24 + int(clock[:2]) == got["hours"] and clock[2:] == ":00:00"
    with pytest.raises(ValueError):
        stages.allocation(stage, 5)


def test_the_allocation_table_is_the_high_tier_at_stride_64():
    assert {s: stages.ALLOCATION[s]["mem_gb"] for s in PIPELINE} == {
        "coarse": 2,
        "tile": 4,
        "solve": 1,
        "stitch": 4,
    }
    assert {s: stages.ALLOCATION[s]["cpus"] for s in PIPELINE} == {
        "coarse": 2,
        "tile": 2,
        "solve": 1,
        "stitch": 4,
    }
    # the tier the benchmark runs is the one the table was worked out for
    tier = high_tier()
    assert adapter.TIER_HIGH_STRIDE64 == {
        "tile": tier["tile"],
        "halo": tier["halo"],
        "out_tile": tier["out_tile"],
        "max_dim": tier["coarse_max_dim"],
        "stride": 64,
    }
    # the benchmark's own step has the benchmark's allocation
    prep = stages.ALLOCATION[stages.PREP]
    assert (prep["cpus"], prep["mem_gb"]) == (ladder.CPUS, ladder.RUNGS_GB[0])


def test_the_retry_rule_is_the_one_those_processes_inherit():
    base = lex.strip_comments((REPO / "conf" / "base.config").read_text())
    listed = re.search(
        r"errorStrategy = \{\s*task\.exitStatus in \[(.*?)\]\s*\? 'retry' : 'finish'",
        base,
        re.S,
    )
    assert tuple(int(v) for v in re.findall(r"\d+", listed[1])) == stages.RETRY_EXIT
    assert int(re.search(r"maxRetries = (\d+)", base)[1]) == stages.MAX_RETRIES == 3
    assert stages.ATTEMPTS == 4


def record(exit_code=0, attempt=1, command=("x",), **more):
    base = {
        "ok": exit_code == 0,
        "exit": exit_code,
        "cause": "" if exit_code == 0 else stages.cause_of(exit_code),
        "attempt": attempt,
        "command": list(command),
        "cpus": 2,
    }
    return {**base, **more}


@pytest.mark.parametrize("code", [104, 134, 135, 137, 139, 140, 143])
def test_a_signal_exit_is_retried_with_the_next_attempt(code):
    assert stages.retryable(record(code))
    go, why = stages.decide(record(code), 2, ["x"])
    assert go and f"exit {code}" in why
    # ... once per attempt, and never the attempt that already ran
    assert stages.decide(record(code, attempt=2), 2, ["x"])[0] is False
    assert stages.decide(record(code, attempt=2), 3, ["x"])[0] is True


@pytest.mark.parametrize("code", [1, 2, 126, 127, 130, 141, 152])
def test_any_other_exit_ends_the_job(code):
    assert not stages.retryable(record(code))
    assert stages.decide(record(code), 2, ["x"])[0] is False


def test_a_killed_job_and_a_job_that_never_started_are_retried():
    killed = record(1, exit=None, cause=stages.KILLED)
    assert stages.retryable(killed) and stages.retryable(None)
    assert stages.decide(killed, 2, ["x"])[0] and stages.decide(None, 3, ["x"])[0]
    # a record of another command, or of another submission, is no record
    assert stages.decide(record(1), 2, ["y"])[0]
    other = record(1, submission="A")
    assert stages.decide(other, 2, ["x"], submission="B")[0]
    assert not stages.decide(other, 2, ["x"], submission="A")[0]


def test_an_ok_job_is_reused_unless_forced():
    ok = record(0)
    assert stages.decide(ok, 1, ["x"])[0] is False
    assert stages.decide(ok, 1, ["x"], force=True)[0] is True
    # outside the launcher (no submission id) a forced run happens at attempt 1
    assert stages.decide(ok, 2, ["x"], force=True)[0] is False
    assert stages.decide(ok, 1, ["other"])[0] is True
    assert stages.decide(record(1), 1, ["x"])[0] is True


def test_a_forced_submission_reuses_only_its_own_ok_records_at_every_attempt():
    """REGBENCH_FORCE=1: the ok record of an EARLIER submission is never reused, also
    when this submission's first attempt was lost and a later attempt finds it."""
    old = record(0, submission="A")
    for attempt in (1, 2, 3, 4):
        go, why = stages.decide(old, attempt, ["x"], "B", force=True)
        assert go and "forced" in why, attempt
        assert stages.decide(old, attempt, ["x"], "B")[0] is False  # not forced: reused
    mine = record(0, submission="B")
    assert not any(
        stages.decide(mine, a, ["x"], "B", force=True)[0] for a in (1, 2, 3, 4)
    )


def test_an_ok_job_is_reused_only_for_what_it_was_computed_from():
    ok = record(0, identity="abc")
    assert stages.decide(ok, 1, ["x"], identity="abc")[0] is False
    go, why = stages.decide(ok, 1, ["x"], identity="other")
    assert go and why == "the first attempt"
    assert stages.decide(record(0), 1, ["x"], identity="abc")[0] is True  # no identity
    # ... and a failed record of another job is no evidence for a later attempt
    failed = record(1, identity="abc", submission="A")
    assert stages.decide(failed, 2, ["x"], "A", identity="abc")[0] is False
    assert stages.decide(failed, 2, ["x"], "A", identity="other")[0] is True
    assert stages.same_job(ok, ["x"], "abc") and not stages.same_job(None, ["x"], "abc")


def test_the_cause_of_a_stage_exit():
    assert stages.cause_of(137) == "oom" and stages.cause_of(1, "MemoryError") == "oom"
    assert stages.cause_of(143) == stages.cause_of(140) == "timeout"
    assert stages.cause_of(1) == stages.cause_of(139) == "exception"


# ── one job's record ──────────────────────────────────────────────────────────
PY = sys.executable


def test_a_command_leaves_its_exit_time_and_peak(tmp_path):
    rec = stages.run_command(
        tmp_path, stages.SOLVE, [PY, "-c", "x = bytearray(60 << 20)"]
    )
    assert rec == stages.read_record(tmp_path, stages.SOLVE)
    assert rec["ok"] and rec["exit"] == 0 and rec["cause"] == "" and rec["error"] == ""
    assert rec["wall_s"] > 0 and rec["cpu_s"] > 0 and rec["ended"] >= rec["started"]
    assert 60 < rec["peak_rss_mb"] < 600  # the command's own process, in MB
    assert (rec["cpus"], rec["mem_gb"], rec["attempt"]) == (1, 1, 1)
    assert rec["history"] == []


def test_a_failed_command_keeps_what_it_printed(tmp_path):
    code = "import sys; print('no anchor'); sys.exit(3)"
    rec = stages.run_command(
        tmp_path, stages.COARSE, [PY, "-c", code], execution="local"
    )
    assert not rec["ok"] and rec["exit"] == 3 and rec["cause"] == "exception"
    assert "no anchor" in rec["error"]


def test_a_command_killed_by_a_signal_has_the_shells_status(tmp_path):
    command = [PY, "-c", f"import os; os.kill(os.getpid(), {int(signal.SIGKILL)})"]
    first = stages.run_command(tmp_path, stages.TILE, command, row=7)
    assert first["exit"] == 137 and first["cause"] == "oom" and stages.retryable(first)
    assert stages.record_path(tmp_path, stages.TILE, 7).name == "tile_000007.json"
    again = stages.run_command(tmp_path, stages.TILE, command, row=7, attempt=2)
    assert again["attempt"] == 2 and again["mem_gb"] == 8 and len(again["history"]) == 1
    assert again["history"][0]["exit"] == 137 and again["history"][0]["mem_gb"] == 4
    assert len(stages.attempts_of(again)) == 2


def test_the_record_says_killed_while_the_command_runs(tmp_path):
    held = {}

    def spy(p, rec):
        held.setdefault("first", dict(rec))
        return real(p, rec)

    real = stages._write
    stages._write = spy
    try:
        stages.run_command(
            tmp_path, stages.COARSE, [PY, "-c", "pass"], execution="local"
        )
    finally:
        stages._write = real
    assert held["first"]["ok"] is False and held["first"]["cause"] == stages.KILLED
    assert held["first"]["exit"] is None and stages.retryable(held["first"])


# ── the tile arrays ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "total, most, want",
    [
        (1, 5, [(0, 1)]),
        (5, 5, [(0, 5)]),
        (6, 5, [(0, 5), (5, 1)]),
        (10, 5, [(0, 5), (5, 5)]),
        (11, 5, [(0, 5), (5, 5), (10, 1)]),
        (0, 5, []),
    ],
)
def test_tile_arrays_are_cut_at_the_boundaries(total, most, want):
    assert stages.chunks(total, most) == want
    assert sum(size for _, size in want) == total
    assert all(1 <= size <= most for _, size in want)


def test_a_flat_tile_index_names_one_tile_of_one_round(tmp_path):
    counts = [2, 1, 3]
    got = [stages.locate(counts, i) for i in range(6)]
    assert got == [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert stages.locate(counts, 6) is None and stages.locate(counts, -1) is None
    rows = [("a", 2), ("b", 1), ("c", 3)]
    assert stages.write_layout(tmp_path / "l" / "x.tsv", rows) == 6
    assert stages.read_layout(tmp_path / "l" / "x.tsv") == rows
    assert (tmp_path / "l" / "x.tsv").read_text().splitlines()[2] == "2\tc\t3\t3"


# ── the cost of a round ───────────────────────────────────────────────────────
def job(start, wall, cpu, peak, cpus, **more):
    rec = {"started": start, "ended": start + wall, "wall_s": wall, "cpu_s": cpu}
    return {**rec, "peak_rss_mb": peak, "cpus": cpus, "attempt": 1, "ok": True, **more}


def round_records():
    coarse = job(1000.0, 60.0, 90.0, 400.0, 2)
    # tile 1 was retried: a killed attempt of 20 s, then 50 s
    lost = job(1500.0, 20.0, 30.0, 3900.0, 2, ok=False)
    tiles = [
        job(1500.0, 40.0, 70.0, 1200.0, 2),
        {**job(1700.0, 50.0, 80.0, 1300.0, 2), "attempt": 2, "history": [lost]},
        job(1510.0, 30.0, 50.0, 1100.0, 2),
    ]
    solve = job(2000.0, 10.0, 8.0, 100.0, 1)
    return coarse, tiles, solve


def test_the_cost_of_a_round_is_summed_over_its_jobs():
    coarse, tiles, solve = round_records()
    got = stages.summarise(coarse, tiles, solve)
    # from the start of COARSE to the end of SOLVE, queue waits included
    assert got["elapsed_with_queue_s"] == pytest.approx(2010.0 - 1000.0)
    # COARSE + the longest tile + SOLVE, each the attempt its job ended ok with: no
    # queue, every tile at once. Tile 1's killed attempt (20 s) is not in it
    assert got["longest_tile_s"] == pytest.approx(50.0)
    assert got["wall_unlimited_parallel_s"] == pytest.approx(60.0 + 50.0 + 10.0)
    # the attempts that succeeded: allocated CPUs x duration, and measured CPU time
    walls = 2 * 60.0 + 2 * (40.0 + 50.0 + 30.0) + 1 * 10.0
    assert got["cpu_h_allocated"] == pytest.approx(walls / 3600.0)
    assert got["cpu_h_measured"] == pytest.approx((90 + 70 + 80 + 50 + 8) / 3600.0)
    # the attempt that failed is wasted, as a failed attempt of any method is
    assert got["wasted_attempts"] == 1 and got["wasted_wall_s"] == pytest.approx(20.0)
    assert got["wasted_cpu_h_allocated"] == pytest.approx(2 * 20.0 / 3600.0)
    assert got["wasted_cpu_h_measured"] == pytest.approx(30.0 / 3600.0)
    # the peak is the largest attempt's, the failed one's here
    assert (got["largest_job_peak_mb"], got["largest_job"]) == (3900.0, "tile")
    assert got["largest_job_row"] == 1
    assert (got["n_jobs"], got["n_tiles"], got["n_retries"]) == (6, 3, 1)
    # tiles 0, 2 and the lost attempt of tile 1 overlapped: 3 jobs of 2 CPUs
    assert got["peak_cpus_at_once"] == 6
    assert got["unmeasured_attempts"] == 0 and got["complete"] is True
    assert got["allocation"]["tile"] == stages.allocation("tile")


def test_an_attempt_killed_without_a_duration_is_counted_and_in_no_sum():
    coarse, tiles, solve = round_records()
    killed = {
        "attempt": 1,
        "ok": False,
        "cause": "killed",
        "started": 1400.0,
        "cpus": 2,
    }
    tiles[0] = {**tiles[0], "attempt": 2, "history": [killed]}
    got, base = (
        stages.summarise(coarse, tiles, solve),
        stages.summarise(*round_records()),
    )
    assert got["unmeasured_attempts"] == 1 and got["n_jobs"] == base["n_jobs"] + 1
    assert got["n_retries"] == 2
    assert got["cpu_h_allocated"] == base["cpu_h_allocated"]
    assert got["wasted_wall_s"] == base["wasted_wall_s"]  # unknown, not zero: flagged


def test_a_round_that_failed_shows_what_its_jobs_cost():
    """SOLVE never ran and a tile failed for good: nothing is the cost of a transform,
    and every attempt that ran is there, the failed ones as wasted."""
    coarse, tiles, _ = round_records()
    tiles[2] = {**tiles[2], "ok": False, "exit": 1}
    got = stages.summarise(coarse, tiles, None)
    assert got["complete"] is False and got["elapsed_with_queue_s"] is None
    assert got["solve_s"] == 0 and got["coarse_s"] == pytest.approx(60.0)
    assert got["wasted_attempts"] == 2
    assert got["wasted_wall_s"] == pytest.approx(20.0 + 30.0)
    assert got["cpu_h_measured"] == pytest.approx((90 + 70 + 80) / 3600.0)
    assert stages.summarise(None, [], None)["n_jobs"] == 0


def test_the_queue_time_does_not_span_two_submissions():
    coarse, tiles, solve = round_records()
    one = stages.summarise(
        {**coarse, "submission": "A"}, tiles, {**solve, "submission": "A"}
    )
    assert one["elapsed_with_queue_s"] == pytest.approx(1010.0)
    two = stages.summarise(
        {**coarse, "submission": "A"}, tiles, {**solve, "submission": "B"}
    )
    assert two["elapsed_with_queue_s"] is None
    assert two["wall_unlimited_parallel_s"] == one["wall_unlimited_parallel_s"]


def test_one_retry_rule_for_a_staged_row_and_for_a_run_that_is_one_job():
    """A failed attempt's time and CPU go to `wasted_*` and `register_s` is the
    attempts that succeeded, whether the attempt was a stage job or a whole run."""
    from benchmarks.regbench import score

    jobs = {**stages.summarise(*round_records()), "execution": "jobs"}
    stitch = {
        "cpus": 4,
        "wall_s": 30.0,
        "peak_rss_mb": 900.0,
        "failed_attempts": [{"attempt": 1, "exit": 143, "wall_s": 14400.0, "cpus": 4}],
    }
    run = {
        "ok": True,
        "method": "stare",
        "unit": "pair",
        "launch_id": "9_0",
        "register_s": jobs["wall_unlimited_parallel_s"],
        "write_s": 30.0,
        "resources": {
            "wall_s": jobs["wall_unlimited_parallel_s"],
            "cpu_s": jobs["cpu_h_measured"] * 3600,
        },
        "info": {"jobs": jobs},
        "native": {"stitch_job": stitch},
    }
    row = score.resource_row(run)
    assert row["register_s"] == pytest.approx(120.0)  # no failed attempt in it
    assert row["total_to_image_s"] == pytest.approx(150.0)
    # the failed tile attempt and the failed STITCH attempt
    assert row["wasted_attempts"] == 2
    assert row["wasted_wall_s"] == pytest.approx(20.0 + 14400.0)
    assert row["wasted_cpu_h"] == pytest.approx(30.0 / 3600.0)
    assert row["unmeasured_attempts"] == 0
    # a run that is one job, with one failed attempt before it: the same split
    plain = score.resource_row(
        {
            "ok": True,
            "register_s": 120.0,
            "resources": {"wall_s": 120.0, "cpu_s": 300.0},
            "previous_attempts": [{"wall_s": 20.0, "cpu_s": 30.0}],
        }
    )
    assert plain["register_s"] == 120.0 and plain["wasted_wall_s"] == 20.0
    assert plain["unmeasured_attempts"] == ""
    # a staged round that failed: the row's cost is what its stage jobs spent
    coarse, tiles, _ = round_records()
    spent = {**stages.summarise(coarse, tiles, None), "execution": "jobs"}
    lost = score.resource_row(
        {
            "ok": False,
            "cause": "exception",
            "method": "stare",
            "launch_id": "9_1",
            "register_s": 0.4,
            "resources": {"wall_s": 0.4, "cpu_s": 0.1},
            "info": {"jobs": spent},
        }
    )
    assert lost["register_s"] == pytest.approx(60.0 + 50.0 + 20.0)
    assert lost["cpu_h"] * 3600 == pytest.approx(90 + 70 + 80 + 50 + 30)
    base = {"dataset": "multiplex", "group": "P1", "method": "stare", "unit": "pair"}
    (rec,) = score.summarise_resources([{**lost, **base}])
    assert rec["wasted_wall_s"] == pytest.approx(130.0) and rec["n_ok"] == 0
    # an attempt without a duration: the patient's row says its sums are lower bounds
    jobs["unmeasured_attempts"] = 2
    (rec,) = score.summarise_resources([{**score.resource_row(run), **base}])
    assert rec["unmeasured_attempts"] == 2


# ── STARE as the pipeline runs it, for real ───────────────────────────────────
OPTS = {"label": ""}


@contextlib.contextmanager
def stated(launcher):
    """`REGBENCH_STAGES` as a run states it; None: not stated at all."""
    old = os.environ.pop(adapter.JOBS_ENV, None)
    if launcher is not None:
        os.environ[adapter.JOBS_ENV] = launcher
    try:
        yield
    finally:
        os.environ.pop(adapter.JOBS_ENV, None)
        if old is not None:
            os.environ[adapter.JOBS_ENV] = old


def as_jobs():
    return stated(adapter.JOBS)


def stage_job(cases, out, stage, *where):
    """The arguments slurm/stage.sbatch gives `regbench stage`."""
    argv = ["stage", "--cases", str(cases), "--out", str(out), "--dataset", "multiplex"]
    argv += ["--method", "stare", "--stage", stage, "--attempt", "1"]
    return [*argv, *map(str, where)]


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """The pair registered twice, by the two launchers. As the cohort does it: every
    stage a `regbench stage` call of its own, the tiles by their place in the layout
    (behind another round's tile), then the driver, which collects; `due` keeps what
    the tile submitter is told between those jobs. And by the local launcher:
    `register` starts the same stage processes itself and collects."""
    from benchmarks.regbench import cli

    tmp = tmp_path_factory.mktemp("staged")
    pair = sc.make_pair(tmp / "pair")
    # the selfcheck's pair as a case of a dataset `regbench stage` takes, the second
    # round of two: the first (one tile) is never registered
    case = dataclasses.replace(sc.pair_case(pair, tmp / "made"), dataset="multiplex")
    first = dataclasses.replace(case, case_id="a_first", ref_hw=[1024, 1280])
    write_case(tmp / "cases", first)
    case = load_case(write_case(tmp / "cases", case))
    out, layout = tmp / "out", tmp / "layout.tsv"
    stage_dir = adapter.stage_dir_of(out, None, case)
    assert stage_dir == result_dir(out, "stare", case) / "stages"
    assert adapter.tile_count(case, OPTS) == 2
    base = ["--cases", str(tmp / "cases"), "--dataset", "multiplex", "--method"]
    sized = ["stages", *base, "stare", "--layout", str(layout)]
    assert cli.main(sized) == 0
    collected, real = [], adapter._collect

    def spy(rnd, jobs):
        collected.append((rnd.dir, jobs))
        return real(rnd, jobs)

    def job(stage, *where):
        assert cli.main(stage_job(tmp / "cases", out, stage, *where)) == 0

    def tile(flat):
        job("tile", "--flat", flat, "--layout", layout)

    due = {}

    def ask(when):
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            for attempt in ("1", "2"):
                assert cli.main([*sized, "--out", str(out), "--due", attempt]) == 0
                print("|")
        due[when] = said.getvalue().replace("\n", " ").strip()

    adapter._collect = spy
    try:
        with stated(None):  # a stage job needs no statement: it IS the scheduler's
            tile(1)  # before COARSE: the job does nothing
            due["tile before coarse"] = stages.read_record(stage_dir, "tile", 0)
            ask("before coarse")
            job("coarse", "--index", 1)
            ask("after coarse")
            tile(1)
            ask("after the first tile")
            tile(2)
            ask("after the tiles")
            job("solve", "--index", 1)
        with as_jobs():
            done = methods.run_case("stare", case, out, opts={"stage_images": True})
        assert done == ["stare_rigid", "stare"]
        with stated(adapter.LOCAL):
            names = [v for v, _, _ in adapter.register(case, tmp / "l", {"workers": 2})]
        assert names == ["stare_rigid", "stare"]
    finally:
        adapter._collect = real
    return {
        "case": case,
        "cases": tmp / "cases",
        "out": out,
        "layout": layout,
        "dir": stage_dir,
        "local": tmp / "l" / "stages",
        "collected": collected,
        "due": due,
        "tmp": tmp,
    }


def run_json(out, variant, case):
    return json.loads((result_dir(out, variant, case) / "run.json").read_text())


EVERY = (("coarse", None), ("tile", 0), ("tile", 1), ("solve", None))
# px. One `stare reg-tile` command on identical inputs does not give identical bytes
# in two fresh processes: its float64 values differ by about 1e-8 relative, and a
# control value, which the stage rounds to 4 decimals, by one unit of the fourth
# (1e-4 px), more often on a busy machine. SOLVE smooths the controls into the mesh,
# so the transform of two launches differs by less than that unit: 8e-5 px was the
# largest difference seen, in the manifest and in the mapped points, when this test
# ran ten times in a row on a machine under heavy load (three of the ten were equal
# to the byte). 1e-3 px is ten of those units and twelve times the largest difference
# seen: far above what a rounding flip can do, and 0.3 nm on a slide of 0.325 um per
# px.
TRANSFORM_TOL_PX = 1e-3


def same_but_floats(a, b, tol, at="manifest"):
    """Two JSON values with the same shape, the same keys and the same strings,
    integers, booleans and nulls; a float may differ by at most ``tol``."""
    if isinstance(a, float) and isinstance(b, float):
        assert abs(a - b) <= tol, (at, a, b)
    elif isinstance(a, dict) and isinstance(b, dict):
        assert list(a) == list(b), at
        for key in a:
            same_but_floats(a[key], b[key], tol, f"{at}.{key}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), at
        for i, (x, y) in enumerate(zip(a, b)):
            same_but_floats(x, y, tol, f"{at}[{i}]")
    else:
        assert type(a) is type(b) and a == b, (at, a, b)


def test_the_scheduler_and_the_local_launcher_give_one_transform(staged):
    """The same stage commands, started by `regbench stage` (one call per stage, as
    slurm/stage.sbatch makes it) and by the adapter's local launcher: the anchor and
    the tile plan are equal BYTE FOR BYTE, the records are the same jobs, and the
    transform is equal within `TRANSFORM_TOL_PX` (why not to the byte, and why that
    number: the comment on it)."""
    from stare.stage_warp import make_warper

    ours, local = staged["dir"], staged["local"]
    for name in ("moving_m0.json", "moving_tiles.csv"):
        assert (ours / name).read_bytes() == (local / name).read_bytes(), name
    for stage, row in EVERY:
        a, b = (stages.read_record(d, stage, row) for d in (ours, local))
        assert a["ok"] and b["ok"], (stage, row)
        assert (a["execution"], b["execution"]) == ("jobs", "local")
        # one command, but for the folder the round's files are in
        assert a["command"] == [c.replace(str(local), str(ours)) for c in b["command"]]
        assert a["attempt"] == b["attempt"] == 1 and a["mem_gb"] == b["mem_gb"]
    mans = [json.loads((d / "manifest.json").read_text()) for d in (ours, local)]
    assert mans[0]["slides"]["moving"]["mesh"] is not None  # a mesh, not the anchor
    assert mans[0]["slides"]["moving"]["M0"] == mans[1]["slides"]["moving"]["M0"]
    same_but_floats(*mans, TRANSFORM_TOL_PX)
    h, w = staged["case"].ref_hw
    ys, xs = np.mgrid[0:h:64, 0:w:64]
    points = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    warps = [make_warper(m) for m in mans]
    rigid = [np.asarray(f("moving", points, "rigid")) for f in warps]
    assert np.array_equal(*rigid)
    full = [np.asarray(f("moving", points, "refined")) for f in warps]
    assert np.abs(full[0] - full[1]).max() <= TRANSFORM_TOL_PX
    assert np.abs(full[0] - rigid[0]).max() > 100 * TRANSFORM_TOL_PX  # the mesh moves
    tres = [json.loads((d / "tre.json").read_text()) for d in (ours, local)]
    assert tres[0]["solve"]["solver"] == tres[1]["solve"]["solver"] == "dctpls"
    # STITCH wrote the cohort round's image from that manifest
    import tifffile

    run = run_json(staged["out"], "stare", staged["case"])
    assert tifffile.imread(run["native"]["files"][0]).any()


def test_both_launchers_reach_the_transform_through_the_same_collector(staged):
    """`register` obtains the transform by collecting the stage records, whoever
    started the stage processes: one call of `_collect` per round, on that round's
    stage folder."""
    assert staged["collected"] == [(staged["dir"], True), (staged["local"], False)]
    run = run_json(staged["out"], "stare", staged["case"])
    assert run["info"]["jobs"]["execution"] == "jobs"


def lists_in_register(ns, row, plan_row):
    """The four stage argument lists as the SOURCE of ``stare.cli.register`` writes
    them, for the parsed arguments ``ns`` and one plan row. The function is read, not
    called: its assignments to the names below and the list literals it hands
    ``coarse.main`` and ``jobs.append`` are evaluated one by one, an ``if`` by its own
    test, and no stage is imported or run."""
    import stare.cli

    names = {"ref_name", "mov_name", "work", "m0_f", "tiles_f", "out", "max_disp"}
    names |= {"solve_argv", "stitch_argv"}
    env = {"a": ns, "Path": Path, "_slide_name": stare.cli._slide_name}
    env.update(row=row, r=plan_row, rc=0)
    got = {}

    def value(node):
        code = compile(ast.Expression(node), "stare.cli.register", "eval")
        return eval(code, env)

    def handed(node, owner, method):
        """The list literal of a call ``owner.method([...])`` inside ``node``."""
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == method
                and getattr(call.func.value, "id", None) == owner
                and isinstance(call.args[0], ast.List)
            ):
                return value(call.args[0])
        return None

    def read(body):
        for st in body:
            if isinstance(st, ast.If):
                read(st.body if value(st.test) else st.orelse)
            elif isinstance(st, (ast.For, ast.With)):
                read(st.body)
            elif isinstance(st, ast.Try):
                read(st.body)
            elif isinstance(st, ast.AugAssign) and st.target.id in names:
                env[st.target.id] = env[st.target.id] + value(st.value)
            elif isinstance(st, ast.Assign) and st.targets[0].id in names:
                env[st.targets[0].id] = value(st.value)
            elif isinstance(st, (ast.Assign, ast.Expr)):
                for key, (owner, method) in {
                    "coarse": ("coarse", "main"),
                    "tile": ("jobs", "append"),
                }.items():
                    found = handed(st, owner, method)
                    if found is not None:
                        got[key] = found

    (func,) = ast.parse(textwrap.dedent(inspect.getsource(stare.cli.register))).body
    read(func.body)
    return {**got, "solve": env["solve_argv"], "stitch": env["stitch_argv"]}


def test_each_stage_command_is_what_stare_register_hands_that_stage(staged, tmp_path):
    """Every flag of every stage, against the package's own `register`, which is read
    and never run (`lists_in_register`): the four argument lists, compared whole."""
    case = staged["case"]
    ns = adapter.namespace(case, tmp_path, {"workers": 1})
    ours = adapter.stage_commands(ns)
    rows = adapter.Round(case, staged["dir"], dict(OPTS)).plan()
    assert [(r["ix"], r["iy"]) for r in rows] == [("0", "0"), ("1", "0")]
    for i, r in enumerate(rows):
        theirs = lists_in_register(ns, i, r)
        assert set(theirs) == {"coarse", "tile", "solve", "stitch"}
        assert ours["coarse"] == theirs["coarse"]
        assert ours["tile"](i, int(r["ix"]), int(r["iy"])) == theirs["tile"]
        assert ours["solve"] == theirs["solve"]
        assert ours["stitch"](ns.moving, ns.out) == theirs["stitch"]
    # the paths of the files one stage leaves for the next
    assert ours["m0"] == tmp_path / "moving_m0.json"
    assert ours["plan"] == tmp_path / "moving_tiles.csv"
    # the values the pipeline's modules render (modules/local/tiled_*.nf)
    flags = dict(zip(ours["coarse"][::2], ours["coarse"][1::2]))
    assert flags["--tile"] == "2048" and flags["--halo"] == "256"
    assert flags["--max-dim"] == "1024" and flags["--nuclear-index"] == "0"
    tile = ours["tile"](1, 1, 0)
    assert (
        tile[tile.index("--stride") + 1] == "64"
        and tile[tile.index("--row") + 1] == "1"
    )
    solve_flags = dict(zip(ours["solve"][::2], ours["solve"][1::2]))
    assert (
        solve_flags["--max-disp"] == "256" and solve_flags["--moving-name"] == "moving"
    )
    assert "--max-disp ${maxDisp}" in name_block("TILED_SOLVE")
    last = ours["stitch"]("/m.tif", "/o.tif")
    assert last[last.index("--out-tile") + 1] == "1024"
    assert last[last.index("--pixel-size") + 1] == repr(float(case.pixel_size_um))
    assert last[last.index("--channel-names") + 1 :] == adapter.channel_names(case)
    assert last[last.index("--moving-name") + 1] == "moving"
    # and each runs as the package's own command, in a process of its own
    assert adapter.cli("tile", ["--row", "1"]) == [
        PY,
        "-m",
        "stare.cli",
        "reg-tile",
        "--row",
        "1",
    ]
    assert (
        adapter.cli("coarse", [])[3] == "coarse"
        and adapter.cli("stitch", [])[3] == "stitch"
    )


# ── who starts the stage processes is stated, never assumed ───────────────────
def no_stage_process(monkeypatch):
    """Fail the test when a stage process is started; returns the calls seen."""
    started = []

    def run(stage_dir, stage, command, *a, **k):
        started.append((stage, command))
        raise AssertionError(f"a {stage} process was started")

    monkeypatch.setattr(stages, "run_command", run)
    return started


@pytest.mark.parametrize("value", [None, "", "1", "one process"])
def test_a_round_whose_launcher_is_not_stated_is_refused(
    staged, tmp_path, monkeypatch, value
):
    """No default: without `jobs` or `local` in REGBENCH_STAGES nothing is started
    and nothing collected, and the failure is the launcher's."""
    from benchmarks.regbench import failure_log

    started = no_stage_process(monkeypatch)
    monkeypatch.setattr(adapter, "_collect", lambda *a: started.append("collect"))
    with stated(value), pytest.raises(stages.StageFailed, match="who starts") as err:
        next(adapter.register(staged["case"], tmp_path / "w", {}))
    assert err.value.regbench_cause == stages.NOT_SUBMITTED
    assert err.value.regbench_no_rerun is True
    assert failure_log.classify(err.value.regbench_cause) == failure_log.LAUNCHER
    assert started == [] and not (tmp_path / "w").exists()
    for value in (adapter.JOBS, adapter.LOCAL):
        with stated(value):
            assert adapter.launcher() == value


def test_a_cohort_unit_without_the_launcher_stated_refuses_before_any_work(
    staged, tmp_path, monkeypatch, capsys
):
    """`regbench run`, as slurm/run_rung.sbatch calls it, for a run of stage jobs
    that was not told so: no stage process, no driver, no record; the end line says
    whose failure it is."""
    from benchmarks.regbench import cli

    started = no_stage_process(monkeypatch)
    monkeypatch.setattr(methods, "run_case", lambda *a, **k: started.append("driver"))
    out = tmp_path / "out"
    argv = ["run", "--cases", str(staged["cases"]), "--out", str(out)]
    argv += ["--dataset", "multiplex", "--method", "stare", "--index", "1"]
    capsys.readouterr()
    with stated(None):
        assert cli.main(argv) == 1
    said = capsys.readouterr().out.strip().splitlines()
    assert "REGBENCH_STAGES=unset" in said[-2] and "Nothing was started" in said[-2]
    assert said[-1].startswith("REGBENCH_END kind=run run=stare dataset=multiplex ")
    assert f"unit={staged['case'].case_id} " in said[-1]
    assert "status=lost cause=not_submitted whose=launcher" in said[-1]
    assert started == [] and not out.exists()
    # ... at every task of the ladder: a later rung reads no record either
    with stated(None):
        assert cli.main([*argv, "--ladder", "climb", "--accounting-waited"]) == 1
    assert started == [] and not out.exists()
    # stated on the command line (a run by hand), for the time of that run only
    seen = []
    monkeypatch.setattr(
        methods,
        "run_case",
        lambda *a, **k: seen.append(os.environ.get(adapter.JOBS_ENV)) or [],
    )
    with stated(None):
        assert cli.main([*argv, "--stages", "local"]) == 1  # the stub made no result
        assert seen == ["local"] and adapter.JOBS_ENV not in os.environ
    # a method that is one job per unit has no stages to state
    with pytest.raises(SystemExit, match="one job per unit"):
        cli.main(
            [*argv[:-4], "--method", "initial", "--index", "0", "--stages", "jobs"]
        )


def test_the_selfcheck_states_the_local_launcher_for_its_own_run(
    tmp_path, monkeypatch, capsys
):
    """`regbench selfcheck` has no stage jobs of the scheduler: it says `local`
    whatever its job's environment says, and leaves the environment as it was."""
    from benchmarks.regbench import cli

    seen = []

    def run(method, *a, **k):
        seen.append((method, os.environ.get(adapter.JOBS_ENV)))
        return {"variants": {}, "status": "ok"}

    monkeypatch.setattr(sc, "run", run)
    for before in (None, adapter.JOBS):
        with stated(before):
            argv = ["selfcheck", "--method", "stare", "--work", str(tmp_path)]
            assert cli.main(argv) == 0
            assert os.environ.get(adapter.JOBS_ENV) == before
    assert seen == [("stare", "local"), ("stare", "local")]
    # a method that is one job per unit is told nothing
    with stated(None):
        assert (
            cli.main(["selfcheck", "--method", "initial", "--work", str(tmp_path)]) == 0
        )
    assert seen[-1] == ("initial", None)


def scripted_stage(monkeypatch, stage, row, codes):
    """The process of one stage job replaced by scripted exits, one per attempt; any
    other stage process fails the test. Returns the attempts that were started."""
    calls = []

    def run(stage_dir, st, command, r=None, attempt=1, execution="jobs", **kw):
        assert (st, r) == (stage, row), f"a {st} process was started (row {r})"
        calls.append(attempt)
        code = codes[len(calls) - 1]
        rec = {
            **stages.read_record(stage_dir, st, r),
            "ok": False,
            "exit": code,
            "cause": stages.cause_of(code),
            "error": "Killed" if code == 137 else "boom",
            "attempt": attempt,
            "mem_gb": stages.allocation(st, attempt)["mem_gb"],
            "command": list(command),
            "identity": kw.get("identity"),
            "execution": execution,
        }
        stages._write(stages.record_path(stage_dir, st, r), rec)
        return rec

    monkeypatch.setattr(stages, "run_command", run)
    return calls


@pytest.mark.parametrize(
    "codes, attempts, cause",
    [
        ([137, 137, 137, 137], 4, "oom"),  # retried as the pipeline retries it
        ([1], 1, "exception"),  # an exit the pipeline does not retry
    ],
)
def test_a_stage_that_fails_under_the_local_launcher_fails_the_round(
    staged, tmp_path, monkeypatch, codes, attempts, cause
):
    """The local launcher starts a failed tile again by the pipeline's rule and no
    stage behind it; the round then fails where every round fails, in `_collect`,
    with the tile's own cause (the method's)."""
    from benchmarks.regbench import failure_log

    work = tmp_path / "w"
    shutil.copytree(staged["local"], work / "stages")
    for path in (work / "stages").rglob("*.json"):
        text = path.read_text().replace(str(staged["local"]), str(work / "stages"))
        path.write_text(text)
    rewrite(work / "stages", "tile", 1, ok=False, exit=137, cause="oom")
    calls = scripted_stage(monkeypatch, "tile", 1, codes)
    with stated(adapter.LOCAL), pytest.raises(stages.StageFailed) as err:
        next(adapter.register(staged["case"], work, {"workers": 2}))
    assert calls == list(range(1, attempts + 1))  # COARSE and tile 0 were ok already
    assert f"TILE (tile row 1) failed at attempt {attempts} of 4" in str(err.value)
    assert err.value.regbench_cause == cause
    assert failure_log.classify(cause) == failure_log.METHOD
    assert err.value.regbench_no_rerun is False
    spent = err.value.regbench_info["jobs"]
    assert spent["execution"] == "local" and spent["complete"] is False


def test_a_model_the_first_stage_refuses_is_refused_before_any_job(staged, tmp_path):
    """`stare register` lists three models and `stare coarse` takes one: an option the
    first job would die on is refused where the round's parameters are read."""
    from stare.stages import coarse

    case = staged["case"]
    source = Path(coarse.__file__).read_text()
    assert 'ap.add_argument("--model", default="euclidean", choices=["euclidean"])' in (
        source
    )
    assert adapter.COARSE_MODELS == ("euclidean",)
    assert (
        adapter.namespace(case, tmp_path, {"model": "euclidean"}).model == "euclidean"
    )
    for model in ("similarity", "affine"):
        with pytest.raises(SystemExit, match=f"model={model}: `stare coarse` accepts"):
            adapter.namespace(case, tmp_path, {"model": model})
        with pytest.raises(SystemExit, match="accepts euclidean"):
            adapter.run_stage(case, tmp_path / "s", {"model": model}, "coarse")
    with pytest.raises(SystemExit):  # a value no stage knows: the parser's own error
        adapter.namespace(case, tmp_path, {"model": "projective"})


def test_the_tile_plan_is_known_before_coarse(staged):
    """What lets the launcher submit every tile job up front: the plan COARSE writes is
    the grid of the reference's size, and the adapter counts the same grid."""
    case = staged["case"]
    rows = adapter.Round(case, staged["dir"], dict(OPTS)).plan()
    assert len(rows) == adapter.tile_count(case, OPTS) == 2
    assert [(r["ix"], r["iy"]) for r in rows] == [("0", "0"), ("1", "0")]
    # a larger slide, by the same rule: 60,000 x 80,000 px at tile 2048
    import dataclasses

    big = dataclasses.replace(case, ref_hw=[60000, 80000])
    assert adapter.tile_count(big, OPTS) == 30 * 40 == 1200


def test_run_json_bills_the_jobs_and_not_the_writer(staged):
    case, out = staged["case"], staged["out"]
    run, rigid = run_json(out, "stare", case), run_json(out, "stare_rigid", case)
    recs = {
        s: stages.read_record(staged["dir"], s) for s in ("coarse", "solve", "stitch")
    }
    tiles = [stages.read_record(staged["dir"], "tile", i) for i in (0, 1)]
    jobs = run["info"]["jobs"]
    assert jobs["execution"] == "jobs" and "_cost" not in run["info"]
    assert (jobs["n_jobs"], jobs["n_tiles"], jobs["n_retries"]) == (4, 2, 0)
    want = (
        recs["coarse"]["wall_s"]
        + max(t["wall_s"] for t in tiles)
        + recs["solve"]["wall_s"]
    )
    assert jobs["wall_unlimited_parallel_s"] == pytest.approx(want)
    # the clock is the jobs' and stops at the transform: STITCH is the writer
    assert run["register_s"] == rigid["register_s"] == pytest.approx(want)
    assert recs["stitch"]["ok"] and recs["stitch"]["wall_s"] > 0
    assert run["write_s"] >= recs["stitch"]["wall_s"] and rigid["write_s"] == 0
    assert jobs["elapsed_with_queue_s"] == pytest.approx(
        recs["solve"]["ended"] - recs["coarse"]["started"]
    )
    assert (
        jobs["elapsed_with_queue_s"]
        < recs["stitch"]["started"] - recs["coarse"]["started"]
    )
    every = [recs["coarse"], *tiles, recs["solve"]]
    res = run["resources"]
    assert res["cpu_s"] == pytest.approx(sum(r["cpu_s"] for r in every))
    assert res["peak_rss_mb"] == pytest.approx(max(r["peak_rss_mb"] for r in every))
    assert jobs["cpu_h_allocated"] * 3600 == pytest.approx(
        sum(r["cpus"] * r["wall_s"] for r in every)
    )
    assert [r["cpus"] for r in every] == [2, 2, 2, 1] and recs["stitch"]["cpus"] == 4
    assert run["native"]["stitch_job"]["cpus"] == 4
    assert run["native"]["stitch_job"]["wall_s"] == recs["stitch"]["wall_s"]
    # the driver's contract is as for any run
    m0 = np.asarray(run["info"]["m0"], dtype=float)
    assert m0.shape == (3, 3) and rigid["info"]["m0"] == run["info"]["m0"]
    assert isinstance(run["points_sha256"], dict)
    assert (result_dir(out, "stare", case) / "warped.npz").exists()
    assert (
        run["native"]["writer"] == "stare.stages.stitch.main"
        and rigid["native"] is None
    )
    side = run["info"]["implementation"]["benchmark_side"]
    assert len(side) == 2 and "scheduler job of its own" in side[1]


def test_a_round_that_is_done_is_reused_stage_by_stage(staged):
    case = staged["case"]
    before = stages.read_record(staged["dir"], "tile", 1)
    for stage, attempt in (("coarse", 1), ("tile", 1), ("tile", 3), ("solve", 2)):
        got = adapter.run_stage(case, staged["dir"], dict(OPTS), stage, None, attempt)
        assert got and all(go is False for _, _, go, _, _ in got), (stage, attempt)
    assert stages.read_record(staged["dir"], "tile", 1) == before
    with as_jobs():
        again = methods.run_case(
            "stare", case, staged["out"], opts={"stage_images": True}, resume=True
        )
    assert again == ["stare_rigid", "stare"]
    assert run_json(staged["out"], "stare", case)["reused_by"]
    # the local launcher on a round that is done starts nothing either
    started = []
    real = stages.run_command
    stages.run_command = lambda *a, **k: started.append(a[1]) or real(*a, **k)
    try:
        with stated(adapter.LOCAL):
            list(adapter.register(case, staged["local"].parent, {"workers": 2}))
    finally:
        stages.run_command = real
    assert started == []


def clone(staged, tmp_path):
    """The round's stage files under another results folder, its records rewritten for
    that place: a copy a test may break."""
    out = tmp_path / "out"
    new = adapter.stage_dir_of(out, None, staged["case"])
    shutil.copytree(staged["dir"], new)
    for path in new.rglob("*.json"):
        path.write_text(path.read_text().replace(str(staged["dir"]), str(new)))
    return out, new


def rewrite(stage_dir, stage, row=None, **changes):
    path = stages.record_path(stage_dir, stage, row)
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))


def test_solve_never_runs_on_fewer_tiles_than_the_plan(staged, tmp_path):
    case = staged["case"]
    out, new = clone(staged, tmp_path)
    rnd = adapter.Round(case, new, dict(OPTS))
    assert rnd.tiles_done()[1] == ""
    kept = rnd.control(1).read_bytes()
    rnd.control(1).unlink()  # the tile's record still says ok
    assert "1 missing" in rnd.tiles_done()[1]
    got = adapter.run_stage(case, new, dict(OPTS), "solve", force=True)
    assert [(go, rec) for _, _, go, _, rec in got] == [(False, None)]
    with as_jobs(), pytest.raises(stages.StageFailed, match="not complete"):
        next(
            adapter.register(
                case, tmp_path, {"native_dir": lambda c, v: new.parent / "native"}
            )
        )
    # a control file of no tile of this plan would be read by SOLVE's glob: refused too
    rnd.control(1).write_bytes(kept)
    (new / "moving_9_9_ctrl.json").write_bytes(kept)
    assert "1 of another plan" in rnd.tiles_done()[1]
    # and a tile whose record is not ok
    (new / "moving_9_9_ctrl.json").unlink()
    rewrite(new, "tile", 0, ok=False, exit=1, cause="exception")
    assert "1 of 2 tiles are not ok (first: row 0)" in rnd.tiles_done()[1]


def test_a_tile_that_fails_after_its_retries_fails_the_round(staged, tmp_path):
    case = staged["case"]
    out, new = clone(staged, tmp_path)
    rewrite(
        new,
        "tile",
        1,
        ok=False,
        exit=137,
        cause="oom",
        attempt=4,
        mem_gb=16,
        error="Killed",
    )
    with as_jobs():
        assert methods.run_case("stare", case, out, opts={"stage_images": True}) == []
    for variant in ("stare_rigid", "stare"):
        run = run_json(out, variant, case)
        assert run["ok"] is False and run["cause"] == "oom" and run["no_rerun"] is True
        assert "TILE (tile row 1) failed at attempt 4 of 4" in run["error"]
        assert "exit 137 at 16 GB" in run["error"]
    # no rung of the last job's ladder runs the round again: its tiles were retried
    paths = [result_dir(out, v, case) / "run.json" for v in ("stare_rigid", "stare")]
    for task in (ladder.CLIMB, ladder.AGAIN):
        assert ladder.unit_decision(paths, task, 2)[0] is False
    plain = {**run_json(out, "stare", case), "no_rerun": False}
    paths[1].write_text(json.dumps(plain))
    assert ladder.unit_decision(paths[1:], ladder.CLIMB, 2)[0] is True


def test_a_stage_without_a_record_is_the_schedulers_loss(staged, tmp_path):
    case = staged["case"]
    out, new = clone(staged, tmp_path)
    stages.record_path(new, "coarse").unlink()
    started, real = [], stages.run_command
    stages.run_command = lambda *a, **k: started.append(a[1]) or real(*a, **k)
    try:
        with as_jobs():
            done = methods.run_case("stare", case, out, opts={"stage_images": True})
    finally:
        stages.run_command = real
    # the round's last job fails the round instead of running the stage itself
    assert done == [] and started == []
    run = run_json(out, "stare", case)
    assert run["cause"] == "infrastructure" and run["no_rerun"] is True
    assert "COARSE: no attempt of this job left a record" in run["error"]
    # the failed round keeps what its stage jobs cost
    spent = run["info"]["jobs"]
    assert spent["complete"] is False and spent["execution"] == "jobs"
    assert spent["n_jobs"] == 3 and spent["cpu_h_measured"] > 0
    # the stages behind a stage that is not ok do nothing
    for stage in ("tile", "solve"):
        got = adapter.run_stage(case, new, dict(OPTS), stage, force=True)
        assert [(go, why, rec) for _, _, go, why, rec in got] == [
            (False, "COARSE is not ok", None)
        ]


def test_stages_are_not_reused_when_an_input_file_holds_other_bytes(
    staged, tmp_path, monkeypatch
):
    """Stages ok under one submission, the moving image replaced at the same path,
    a second submission: nothing is "already ok", and the old manifest is not applied
    to the new points."""
    from benchmarks.regbench import cases as cases_mod

    case = staged["case"]
    out, new = clone(staged, tmp_path)
    monkeypatch.setenv("REGBENCH_SUBMISSION", "A")
    with as_jobs():
        made = methods.run_case("stare", case, out, opts={"stage_images": True})
        assert made == ["stare_rigid", "stare"]
    rnd = adapter.Round(case, new, dict(OPTS))
    every = [("coarse", None), ("tile", 0), ("tile", 1), ("solve", None)]
    assert all(rnd.done(s, r) for s, r in every)
    ids = {s: stages.read_record(new, s, r)["identity"] for s, r in every}
    assert all(len(v) == 64 for v in ids.values()) and len(set(ids.values())) == 3
    assert ids == {s: rnd.identity(s) for s, _ in every}
    monkeypatch.setenv("REGBENCH_SUBMISSION", "B")
    with as_jobs():
        again = methods.run_case(
            "stare", case, out, opts={"stage_images": True}, resume=True
        )
    assert again == ["stare_rigid", "stare"]
    assert run_json(out, "stare", case)["submissions"] == ["A", "B"]  # reused
    image = Path(case.mov_image)
    kept, st = image.read_bytes(), image.stat()
    assert len(kept) > cases_mod.SAMPLE_BLOCKS * cases_mod.SAMPLE_BYTES  # sampled
    try:
        # other bytes at the same path (a byte between the samples; a new time)
        data = bytearray(kept)
        data[len(kept) // 2 + 12345] ^= 0xFF
        image.write_bytes(data)
        os.utime(image, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        rnd = adapter.Round(case, new, dict(OPTS))
        assert rnd.identity("coarse") != ids["coarse"]
        assert rnd.identity("tile") != ids["tile"]
        assert rnd.identity("solve") == ids["solve"]  # SOLVE reads no slide ...
        assert not rnd.done("coarse") and rnd.tiles_done()[0] is None
        # ... and is not run: the stage before it is not ok for these bytes
        got = adapter.run_stage(case, new, dict(OPTS), "solve")
        assert [(go, rec) for _, _, go, _, rec in got] == [(False, None)]
        go, why = stages.decide(
            rnd.record("coarse"), 1, rnd.command("coarse"), "B", False,
            rnd.identity("coarse"),
        )  # fmt: skip
        assert go and why == "the first attempt"
        assert adapter.run_stage(case, new, dict(OPTS), "tile", None, 2) == [
            ("tile", None, False, "COARSE is not ok", None)
        ]
        # the unit: its results rest on other bytes, so they are not this run's
        with as_jobs():
            again = methods.run_case(
                "stare", case, out, opts={"stage_images": True}, resume=True
            )
        assert again == []
        lost = run_json(out, "stare", case)
        assert lost["ok"] is False and "reused_by" not in lost
        assert not (result_dir(out, "stare", case) / "warped.npz").exists()
        # the same size AND the same time, other bytes inside a sampled block
        data = bytearray(kept)
        data[5000] ^= 0xFF
        image.write_bytes(data)
        os.utime(image, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert image.stat().st_mtime_ns == st.st_mtime_ns
        cases_mod._FILE_FINGERPRINTS.clear()  # a job is a new process: no cache
        assert adapter.Round(case, new, dict(OPTS)).identity("coarse") != ids["coarse"]
    finally:
        image.write_bytes(kept)
        os.utime(image, ns=(st.st_atime_ns, st.st_mtime_ns))
    cases_mod._FILE_FINGERPRINTS.clear()
    # the file as it was: the stage records stand again
    assert adapter.Round(case, new, dict(OPTS)).done("coarse")


def test_another_stare_version_or_other_preprocessing_code_is_another_job(
    staged, tmp_path, monkeypatch
):
    import stare as package

    from benchmarks.regbench import prep

    case = staged["case"]
    _, new = clone(staged, tmp_path)
    ids = {s: adapter.Round(case, new, dict(OPTS)).identity(s) for s in PIPELINE[:3]}
    monkeypatch.setattr(package, "__version__", "0.0.0-other")
    rnd = adapter.Round(case, new, dict(OPTS))
    assert all(rnd.identity(s) != ids[s] for s in ids) and not rnd.done("coarse")
    monkeypatch.undo()
    # the preprocessing row: its `prep` job is keyed on the preprocessing code too
    opts = {"prep": "grey_clahe", "label": f"{prep.ROW}_grey_clahe"}
    before = adapter.Round(case, tmp_path / "p", dict(opts)).identity("prep")
    monkeypatch.setattr(prep, "code_sha256", lambda: "other code")
    assert adapter.Round(case, tmp_path / "p", dict(opts)).identity("prep") != before
    monkeypatch.undo()
    other = {"prep": "grey", "label": f"{prep.ROW}_grey"}
    assert adapter.Round(case, tmp_path / "p", dict(other)).identity("prep") != before


def test_a_result_of_the_other_launcher_is_not_reused_as_the_staged_row(staged):
    """The unit's reuse key says which launcher started its stages and by which code."""
    from benchmarks.regbench import resources

    case, out = staged["case"], staged["out"]
    run = run_json(out, "stare", case)
    with as_jobs():
        jobs = methods.reuse_key("stare", [case], out, run["opts"], None)
    with stated(adapter.LOCAL):
        local = methods.reuse_key("stare", [case], out, run["opts"], None)
    assert (
        run["reuse_key"] == jobs
        and run_json(out, "stare_rigid", case)["reuse_key"] == jobs
    )
    assert (jobs["execution"], local["execution"]) == ("jobs", "local")
    assert jobs["method"]["stare"] == stare.__version__ == local["method"]["stare"]
    solve = stages.read_record(staged["dir"], "solve")["identity"]
    assert jobs["method"]["solve"] == {case.case_id: solve}
    assert "solve" not in local["method"] and jobs["code"] == local["code"]
    assert set(jobs["inputs"][case.case_id]) >= {"ref_image", "mov_image"}
    names = dict(zip(adapter.VARIANTS, methods.variant_names("stare", None)))
    args = (out, [case], names, "stare", None, run["opts"], False)
    host = resources.host_info()
    assert methods._reuse(*args, host, "x", local) is None
    for change in ({"code": "0" * 64}, {"inputs": {}}, {"method": None}):
        assert methods._reuse(*args, host, "x", {**jobs, **change}) is None, change
    # the adapter's code: the driver, the adapter, and what the adapter imports
    files = {p.name for p in methods.code_files("stare")}
    assert files >= {"__init__.py", "stare.py", "stages.py", "prep.py", "mapping.py"}
    assert "score.py" not in files and "valis.py" not in files
    assert {p.name for p in methods.code_files("valis")} >= {
        "valis.py", "_valis_points.py"
    }  # fmt: skip
    assert methods.code_fingerprint("stare") != methods.code_fingerprint("palom")


def test_what_is_due_and_a_tile_that_was_never_submitted(staged, tmp_path, monkeypatch):
    """The tile submitter's question (`tiles_due`), and what the round's last job says
    of a tile no job of this submission ran: the launcher's failure while the submitter
    has not finished, the scheduler's once it has."""
    case = staged["case"]
    out, new = clone(staged, tmp_path)
    assert adapter.tiles_due(case, new, dict(OPTS), 1) == []
    assert adapter.tiles_due(case, new, dict(OPTS), 2) == []
    rewrite(new, "tile", 1, ok=False, exit=137, cause="oom")
    assert adapter.tiles_due(case, new, dict(OPTS), 2) == [1]
    assert adapter.tiles_due(case, new, dict(OPTS), 1) == [1]
    rewrite(new, "tile", 1, ok=False, exit=1, cause="exception")
    assert adapter.tiles_due(case, new, dict(OPTS), 2) == []  # not a retried exit
    stages.record_path(new, "tile", 1).unlink()
    assert adapter.tiles_due(case, new, dict(OPTS), 1) == [1]
    monkeypatch.setenv("REGBENCH_SUBMISSION", "S-2")
    with as_jobs():
        assert methods.run_case("stare", case, out, opts={"stage_images": True}) == []
    run = run_json(out, "stare", case)
    assert run["cause"] == stages.NOT_SUBMITTED and run["no_rerun"] is True
    assert (
        "TILE (tile row 1): this job never ran: it was not submitted, or was"
        in (run["error"])
    )
    assert "the tile submitter did not finish" in run["error"]
    from benchmarks.regbench import failure_log, resources, score

    assert failure_log.classify(run["cause"]) == failure_log.LAUNCHER
    assert run["cause"] in resources.NOT_THE_METHOD and run["cause"] in score.CAUSES
    assert set(score.NOT_THE_METHOD) == set(resources.NOT_THE_METHOD)
    # the submitter finished: the tile's job was in the queue and left no record
    state = stages.feed_state(out, "S-2", "stare", case.dataset)
    state.parent.mkdir(parents=True)
    Path(f"{state}.done").write_text("")
    with as_jobs():
        assert methods.run_case("stare", case, out, opts={"stage_images": True}) == []
    assert run_json(out, "stare", case)["cause"] == "infrastructure"
    # COARSE not ok: no tile of that round is ever due
    stages.record_path(new, "coarse").unlink()
    assert adapter.tiles_due(case, new, dict(OPTS), 1) == []


def test_a_tile_the_submitter_did_not_see_through_is_the_launchers(
    staged, tmp_path, monkeypatch
):
    """A tile task that had not ended when the round's last job ran, or that ended the
    way the pipeline retries with attempts left: while the tile submitter has not
    finished, the attempt after it was never submitted, and that is the launcher's.
    Once the submitter has finished (it waits for every tile task to leave the queue),
    the tile's own record is the answer, and it is the method's."""
    from benchmarks.regbench import failure_log

    case = staged["case"]
    out, new = clone(staged, tmp_path)
    monkeypatch.setenv("REGBENCH_SUBMISSION", "S-3")

    def last_job():
        with as_jobs():
            assert methods.run_case("stare", case, out, opts={"stage_images": 1}) == []
        return run_json(out, "stare", case)

    # still running: the placeholder of its first attempt
    rewrite(new, "tile", 1, ok=False, exit=None, cause="killed", attempt=1,
            submission="S-3")  # fmt: skip
    run = last_job()
    assert run["cause"] == stages.NOT_SUBMITTED and run["no_rerun"] is True
    assert "attempt 1 of 4 had not recorded its end" in run["error"]
    assert "the attempt after it was never submitted" in run["error"]
    assert "the tile submitter did not finish" in run["error"]
    assert failure_log.classify(run["cause"]) == failure_log.LAUNCHER
    # the scheduler's accounting does not turn it into anybody else's
    path = result_dir(out, "stare", case) / "run.json"
    acct = {run["launch_id"]: {"state": "COMPLETED", "elapsed_s": 1.0, "cpu_s": 1.0}}
    assert ladder.settle(path, "", acct, "S-3") == stages.NOT_SUBMITTED
    # ended the way the pipeline retries, and attempt 3 was never fed
    rewrite(new, "tile", 1, exit=137, cause="oom", attempt=2)
    run = last_job()
    assert run["cause"] == stages.NOT_SUBMITTED
    assert (
        "attempt 2 of 4 ended with exit 137, which the pipeline retries"
        in (run["error"])
    )
    # the last attempt is the method's own, submitter or not
    rewrite(new, "tile", 1, exit=137, cause="oom", attempt=4)
    assert last_job()["cause"] == "oom"
    # and so is everything once the submitter saw the queue empty
    rewrite(new, "tile", 1, exit=None, cause="killed", attempt=1)
    done = Path(f"{stages.feed_state(out, 'S-3', 'stare', case.dataset)}.done")
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text("")
    run = last_job()
    assert run["cause"] == "killed" and "TILE (tile row 1) failed" in run["error"]


def test_a_retried_tile_runs_alone_with_the_next_attempts_memory(staged, tmp_path):
    case = staged["case"]
    out, new = clone(staged, tmp_path)
    rewrite(new, "tile", 1, ok=False, exit=137, cause="oom")
    got = adapter.run_stage(case, new, dict(OPTS), "tile", None, attempt=2)
    assert [(row, go) for _, row, go, _, _ in got] == [(0, False), (1, True)]
    rec = got[1][4]
    assert rec["ok"] and rec["attempt"] == 2 and rec["mem_gb"] == 8
    assert [h["exit"] for h in rec["history"]] == [137]
    assert rec["command"][:4] == [PY, "-m", "stare.cli", "reg-tile"]
    # an exit the pipeline does not retry stays as it is
    rewrite(new, "tile", 0, ok=False, exit=1, cause="exception")
    got = adapter.run_stage(case, new, dict(OPTS), "tile", None, attempt=3)
    assert [go for _, _, go, _, _ in got] == [False, False]
    cost = stages.summarise(
        stages.read_record(new, "coarse"),
        [stages.read_record(new, "tile", 1)],
        stages.read_record(new, "solve"),
    )
    assert (cost["n_jobs"], cost["n_retries"]) == (4, 1)
    assert cost["wasted_attempts"] == 1 and cost["wasted_wall_s"] > 0


def test_the_stage_command_of_the_cli_runs_a_round(staged, capsys):
    """`regbench stage`, as stage.sbatch calls it: by round, and a tile by its place in
    the layout the launcher wrote; and what the tile submitter is told between the
    jobs (`stages --due`, attempt 1 | attempt 2), as the fixture kept it."""
    from benchmarks.regbench import cli

    layout, out = staged["layout"], staged["out"]
    rows = stages.read_layout(layout)
    assert rows == [("a_first", 1), (staged["case"].case_id, 2)]
    assert layout.read_text().splitlines()[1].split("\t")[3] == "1"  # its offset
    due = staged["due"]
    # a tile job of a round without COARSE does nothing, and nothing of it is due
    assert due["tile before coarse"] is None and due["before coarse"] == "| |"
    # COARSE ok: both tiles by their place in the layout; at a later attempt a tile
    # without a record never started, so its round is due
    assert due["after coarse"] == "1 2 | 1 |"
    assert due["after the first tile"] == "2 | 1 |"
    assert due["after the tiles"] == "| |"  # no first attempt left, no retry due
    assert all(stages.read_record(staged["dir"], s, r)["ok"] for s, r in EVERY)
    assert (staged["dir"] / "manifest.json").exists()
    kept = {p: p.read_bytes() for p in sorted(staged["dir"].rglob("*.json"))}

    def tile(flat, where):
        argv = stage_job(
            staged["cases"], out, "tile", "--flat", flat, "--layout", where
        )
        return cli.main(argv)

    with stated(None):
        assert tile(99, layout) == 0  # past the last tile: nothing to do
        # COARSE planned another number of tiles than the arrays were sized for: the
        # task says so on its end line, as the launcher's failure, and exits through it
        wrong = staged["tmp"] / "wrong.tsv"
        stages.write_layout(wrong, [(c, n + 1 if n == 2 else n) for c, n in rows])
        capsys.readouterr()
        assert tile(1, wrong) == 1
        said = capsys.readouterr().out.strip().splitlines()
        assert "COARSE planned 2 tiles, the arrays were sized for 3" in said[-2]
        assert said[-1].startswith("REGBENCH_END kind=stage ")
        assert "status=failed cause=not_submitted whose=launcher" in said[-1]
        assert "planned=2 sized_for=3" in said[-1]
        # arrays sized for another set of cases are refused
        stages.write_layout(wrong, rows[:-1])
        with pytest.raises(SystemExit, match="does not list"):
            tile(0, wrong)
    assert kept == {p: p.read_bytes() for p in sorted(staged["dir"].rglob("*.json"))}


def test_the_preprocessing_row_runs_in_the_same_jobs(toy_cases, tmp_path):
    """`stare_prep` is the same four stages behind a `prep` job of the benchmark's own:
    STARE registers the preprocessed copies, STITCH draws the original slide."""
    from benchmarks.regbench import prep

    case = toy_cases["crop"]
    label = f"{prep.ROW}_grey_clahe"
    opts = {"prep": "grey_clahe", "label": label}
    out = tmp_path / "out"
    stage_dir = adapter.stage_dir_of(out, label, case)
    assert stage_dir == result_dir(out, label, case) / "stages"
    # nothing of STARE runs before the preprocessing is ok
    got = adapter.run_stage(case, stage_dir, dict(opts), "coarse")
    assert [(go, why) for _, _, go, why, _ in got] == [
        (False, "the preprocessing is not ok")
    ]
    for stage in ("prep", "coarse", "tile", "solve"):
        got = adapter.run_stage(case, stage_dir, dict(opts), stage)
        assert [(s, go, rec["ok"]) for s, _, go, _, rec in got] == [(stage, True, True)]
    made = prep.input_paths(stage_dir / "prep", "grey_clahe")
    assert all(p.exists() for p in made)
    coarse = stages.read_record(stage_dir, "coarse")["command"]
    assert coarse[coarse.index("--reference") + 1] == str(made[0])
    assert coarse[coarse.index("--moving") + 1] == str(made[1])
    assert coarse[coarse.index("--nuclear-index") + 1] == "0"
    with as_jobs():
        done = methods.run_case(
            "stare", case, out, opts={"prep": "grey_clahe"}, label=label
        )
    assert done == [f"{label}_rigid", label]
    run = run_json(out, label, case)
    assert run["info"]["prep"]["candidate"] == "grey_clahe"
    assert run["info"]["prep"]["prep_s"] > 0 and run["info"]["input"]["moving"] == str(
        made[1]
    )
    assert (
        run["info"]["jobs"]["execution"] == "jobs"
        and run["info"]["jobs"]["n_jobs"] == 3
    )
    # the written image is the ORIGINAL slide moved by this row's transform
    stitch = stages.read_record(stage_dir, "stitch")["command"]
    assert stitch[stitch.index("--moving") + 1] == str(case.mov_image) != str(made[1])
    # a plain row has no such job, and asking for one does nothing
    assert adapter.run_stage(case, tmp_path / "plain", dict(OPTS), "prep") == []
    assert (
        stages.allocation("prep")["mem_gb"] == 16
        and stages.allocation("prep", 4)["mem_gb"] == 64
    )


# ── what the tables publish ───────────────────────────────────────────────────
def test_the_cost_tables_say_what_a_run_in_several_jobs_cost(
    staged, toy_cases, tmp_path
):
    from benchmarks.regbench import cost, score

    run = run_json(staged["out"], "stare", staged["case"])
    jobs = run["info"]["jobs"]
    row = score.resource_row(run)
    # the headline columns, as defined for several jobs
    assert (
        row["register_s"]
        == row["wall_unlimited_parallel_s"]
        == pytest.approx(jobs["wall_unlimited_parallel_s"])
    )
    assert row["total_to_image_s"] == pytest.approx(row["register_s"] + run["write_s"])
    assert row["elapsed_with_queue_s"] == pytest.approx(jobs["elapsed_with_queue_s"])
    assert row["cpu_h"] == pytest.approx(jobs["cpu_h_measured"])
    assert row["cpu_h_allocated"] == pytest.approx(jobs["cpu_h_allocated"])
    assert row["cpu_efficiency"] == pytest.approx(
        jobs["cpu_h_measured"] / jobs["cpu_h_allocated"]
    )
    assert row["peak_rss_gb"] == pytest.approx(jobs["largest_job_peak_mb"] / 1024)
    assert row["largest_job"] in ("coarse", "tile", "solve")
    assert (row["n_jobs"], row["n_tiles"], row["n_job_retries"]) == (4, 2, 0)
    assert row["peak_cpus_at_once"] == 2  # the stages ran one after another here
    stitch = run["native"]["stitch_job"]
    assert row["write_cpu_h_allocated"] == pytest.approx(4 * stitch["wall_s"] / 3600)
    assert row["write_peak_gb"] == pytest.approx(stitch["peak_rss_mb"] / 1024)
    # a run that is one job: those columns are there and blank
    plain = score.resource_row(
        {**run, "info": {**run["info"], "jobs": {**jobs, "execution": "local"}}}
    )
    extra = (
        "elapsed_with_queue_s",
        "wall_unlimited_parallel_s",
        "cpu_h_allocated",
        "largest_job",
        "peak_cpus_at_once",
        "n_jobs",
        "n_tiles",
        "n_job_retries",
        "write_cpu_h_allocated",
        "write_peak_gb",
    )
    assert all(plain[k] == "" for k in extra) and all(row[k] != "" for k in extra)
    assert plain["cpu_efficiency"] == pytest.approx(
        run["resources"]["cpu_s"] / (run["register_s"] * row["cpus_requested"])
    )
    # the header of every cost table says it in one sentence, with the number
    base = {"dataset": "multiplex", "group": "P1", "method": "stare"}
    busy = {**row, **base, "peak_cpus_at_once": 2400.0, "n_tiles": 1200.0}
    (line,) = [
        ln
        for ln in cost.header_lines([busy, {**plain, **base}])
        if "several jobs" in ln
    ]
    assert "used more than 4 CPUs at once, up to 2400 for one round" in line
    assert "wall time is NOT comparable with the 4-CPU methods', its CPU-hours" in line
    assert "COARSE 2 CPUs, every tile 2 CPUs, SOLVE 1 CPU" in line
    assert "1200 tiles per round (median)" in line
    assert not [
        ln for ln in cost.header_lines([{**plain, **base}]) if "several jobs" in ln
    ]
    # per patient: sums over the rounds, and the most CPUs one round had at once
    other = {**busy, "peak_cpus_at_once": 100.0}
    (patient,) = score.summarise_resources(
        [{**busy, "unit": "pair"}, {**other, "unit": "pair"}]
    )
    assert patient["n_jobs"] == 8 and patient["n_tiles"] == 2400
    assert patient["peak_cpus_at_once"] == 2400
    assert patient["wall_unlimited_parallel_s"] == pytest.approx(2 * row["register_s"])
    assert patient["register_s"] == pytest.approx(2 * row["register_s"])
    assert patient["cpu_h_allocated"] == pytest.approx(2 * row["cpu_h_allocated"])
    (single,) = score.summarise_resources([{**plain, **base, "unit": "pair"}])
    assert single["n_jobs"] == "" and single["peak_cpus_at_once"] == ""


# ── the batch scripts, run as the scheduler runs them ─────────────────────────
def test_the_batch_scripts_register_a_round_end_to_end(toy_cases, cases_root, tmp_path):
    """stage.sbatch for COARSE, a tile by its place in the layout and SOLVE, then
    run_rung.sbatch as the round's last job: the scripts' own argument handling, with
    the python on PATH standing in for the image."""
    import subprocess

    from benchmarks.regbench import cli

    slurm = REPO / "benchmarks" / "regbench" / "slurm"
    names = sorted(toy_cases)
    index = names.index("crop")
    out = tmp_path / "out"
    layout = out / "logs" / "stages" / "S-1" / "stare-multiplex.tsv"
    base = ["--cases", str(cases_root), "--dataset", "multiplex", "--method", "stare"]
    assert cli.main(["stages", *base, "--layout", str(layout)]) == 0
    (out / "logs" / "gates").mkdir(parents=True)
    (out / "logs" / "gates" / "stare.tsv").write_text("stare\tok\t-\t0\t-\t-\n")
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("REGBENCH_", "SLURM_"))
    }
    env.update(
        PATH=f"{Path(PY).parent}:{env['PATH']}",
        REGBENCH_REPO=str(REPO),
        REGBENCH_CASES=str(cases_root),
        REGBENCH_OUT=str(out),
        REGBENCH_SUBMISSION="S-1",
        SLURM_ARRAY_TASK_ID=str(index),
        SLURM_ARRAY_JOB_ID="900",
        SLURM_JOB_ID="901",
        SLURM_JOB_NAME="rb-stare-multiplex",
    )
    head = ["stare", "stare", "-", "-", "STARE_REGBENCH", "multiplex"]

    def sbatch(script, *args, task=index):
        argv = [
            "bash",
            str(slurm / script),
            *head[: 5 if script == "run_rung.sbatch" else 6],
            *args,
        ]
        done = subprocess.run(
            argv,
            env={**env, "SLURM_ARRAY_TASK_ID": str(task)},
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
        return done.stdout

    said = sbatch("stage.sbatch", "coarse", "1", "2", "0-02:00:00")
    assert "REGBENCH_END kind=stage" in said and "status=ok" in said.splitlines()[-1]
    # the tile's array task: its place in the layout is offset + task id
    said = sbatch(
        "stage.sbatch",
        "tile",
        "1",
        "4",
        "0-02:00:00",
        str(layout),
        str(index - 1),
        task=1,
    )
    assert "tile row 0, attempt 1: the first attempt; ran, exit 0" in said
    # a later attempt of SOLVE whose first never started runs it, with its memory
    said = sbatch("stage.sbatch", "solve", "2", "2", "0-16:00:00")
    assert "solve, attempt 2" in said and "ran, exit 0" in said
    case = toy_cases["crop"]
    stage_dir = adapter.stage_dir_of(out, None, case)
    solve = stages.read_record(stage_dir, "solve")
    assert solve["ok"] and solve["attempt"] == 2 and solve["mem_gb"] == 2
    assert solve["launch_id"] == f"900_{index}" and solve["submission"] == "S-1"
    assert solve["log_path"].endswith(f"rb-stare-multiplex-900_{index}.log")
    # the round's last job: nothing of the registration runs in it
    said = sbatch(
        "run_rung.sbatch",
        "0",
        "multiplex",
        "1",
        "4",
        "0",
        "-",
        "first",
        "0-04:00:00",
        "jobs",
    )
    assert "multiplex/crop [stare]: stare_rigid, stare" in said
    run = run_json(out, "stare", case)
    assert run["ok"] and run["info"]["jobs"]["execution"] == "jobs"
    assert run["attempt"] == 1 and run["mem_requested_gb"] == 4
    assert run["submissions"] == ["S-1"] and run["time_limit"] == "0-04:00:00"
    assert run["native"]["files"] and stages.read_record(stage_dir, "stitch")["ok"]
    # ... and a round the launcher did not say is in stage jobs (`-`) is refused
    # before anything: nothing is registered here in the stage jobs' place
    other = toy_cases[names[1]]
    said = sbatch(
        "run_rung.sbatch",
        "0",
        "multiplex",
        "1",
        "4",
        "0",
        "-",
        "first",
        "0-04:00:00",
        "-",
        task=1,
    )
    end = said.strip().splitlines()[-1]
    assert end.startswith("REGBENCH_END kind=run run=stare dataset=multiplex ")
    assert f"unit={other.case_id} " in end
    assert "status=lost cause=not_submitted whose=launcher" in end
    assert not result_dir(out, "stare", other).exists()


# ── a staged result through the scorer, end to end ────────────────────────────
def register_as_jobs(case, out, label=None, opts=None):
    """One round as the cohort runs it: every stage a call of its own, then the driver."""
    stage_dir = adapter.stage_dir_of(out, label, case)
    for stage in ("coarse", "tile", "solve"):
        got = adapter.run_stage(case, stage_dir, {"label": label or ""}, stage)
        assert all(rec["ok"] for *_, rec in got), stage
    with as_jobs():
        return methods.run_case("stare", case, out, opts=opts, label=label)


def test_a_staged_result_is_scored_end_to_end(tmp_path, monkeypatch):
    """`score.score` on results made in several jobs: the rigid row (jobs, no image),
    the final row (jobs and STITCH's own job), and the same round with a writer that
    failed. One case in a cases folder of its own: scoring a case is the slow part."""
    from benchmarks.regbench import score
    from benchmarks.regbench.cases import load_case

    from . import toy_cases

    root = tmp_path / "cases"
    case = load_case(toy_cases.prepare_toy(root, "crop", toy_cases.SPECS["crop"]))
    out, lost = tmp_path / "out", tmp_path / "lost"
    assert register_as_jobs(case, out) == ["stare_rigid", "stare"]

    def scored(where):
        got = score.score(
            root,
            where,
            datasets=["multiplex"],
            expected=["stare_rigid", "stare"],
            primary=["stare"],
            with_tissue=False,
        )
        assert (where / "tables" / "resources.csv").exists()
        assert (where / "tables" / "summary.md").exists()
        rows = {(r["case_id"], r["method"]): r for r in got["resources"]}
        per = {(r["group"], r["method"]): r for r in got["summary_resources"]}
        return rows, per

    rows, per = scored(out)
    rigid, final = rows[("crop", "stare_rigid")], rows[("crop", "stare")]
    assert rigid["n_jobs"] == final["n_jobs"] == 3
    assert rigid["write_cpu_h_allocated"] == "" and rigid["write_peak_gb"] == ""
    assert final["write_cpu_h_allocated"] > 0 and final["write_peak_gb"] > 0
    assert math.isnan(per[(case.group, "stare_rigid")]["write_cpu_h_allocated"])
    assert math.isnan(per[(case.group, "stare_rigid")]["write_peak_gb"])
    mine = per[(case.group, "stare")]
    assert mine["write_cpu_h_allocated"] == pytest.approx(
        final["write_cpu_h_allocated"]
    )
    assert mine["write_peak_gb"] == pytest.approx(final["write_peak_gb"])
    # the same stage records under another results folder, and a writer that fails
    shutil.copytree(out, lost)
    old = adapter.stage_dir_of(out, None, case)
    for path in adapter.stage_dir_of(lost, None, case).rglob("*.json"):
        path.write_text(path.read_text().replace(str(old), str(path.parents[1])))

    def broken(state, case_id, dest_dir):
        raise RuntimeError("no space left on device")

    monkeypatch.setattr(adapter, "write_native", broken)
    with as_jobs():
        assert methods.run_case("stare", case, lost) == ["stare_rigid"]
    monkeypatch.undo()
    failed = run_json(lost, "stare", case)
    assert failed["cause"] == "writer" and failed["info"]["jobs"]["n_jobs"] == 3
    rows, per = scored(lost)
    assert rows[("crop", "stare")]["ok"] is False
    assert rows[("crop", "stare")]["cause"] == "writer"
    assert rows[("crop", "stare_rigid")]["ok"] is True
    assert per[(case.group, "stare")]["n_jobs"] == ""


def test_the_other_scorers_summarise_a_staged_row_through_the_same_function():
    """`landmarks.score` and `challenge.score` build their cost rows with
    `score.resource_row` and `score.summarise_resources`: a staged row without an
    image goes through both."""
    from benchmarks.regbench import challenge, landmarks, score

    assert landmarks.cohort is score and challenge.cohort is score
    jobs = {**stages.summarise(*round_records()), "execution": "jobs"}
    run = {
        "ok": True,
        "method": "stare",
        "unit": "pair",
        "launch_id": "9_0",
        "register_s": 140.0,
        "resources": {"wall_s": 140.0, "cpu_s": 300.0, "peak_rss_mb": 3900.0},
        "info": {"jobs": jobs},
        "native": None,
    }
    row = {**score.resource_row(run), "dataset": "hyreco_b", "group": "g"}
    (rec,) = score.summarise_resources([{**row, "method": "stare_rigid"}])
    assert rec["n_jobs"] == 6 and math.isnan(rec["write_cpu_h_allocated"])
    assert math.isnan(rec["write_peak_gb"])


# ── the round's last job: the benchmark's allocation, STITCH as its own process ──
def scripted_stitch(monkeypatch, outcomes):
    """STITCH's process replaced by scripted outcomes, one per attempt: an exit code
    to fail with, or None to run the real command. Returns what each call was given."""
    real, calls = stages.run_command, []

    def run(stage_dir, stage, command, row=None, attempt=1, execution="jobs", **kw):
        if stage != stages.STITCH:
            return real(stage_dir, stage, command, row, attempt, execution, **kw)
        calls.append({"attempt": attempt, **kw})
        code = outcomes[len(calls) - 1]
        if code is None:
            return real(stage_dir, stage, command, row, attempt, execution, **kw)
        alloc = stages.allocation(stage, attempt)
        return {
            "ok": False,
            "exit": code,
            "cause": stages.cause_of(code),
            "error": "Killed" if code == 137 else "boom",
            "attempt": attempt,
            "wall_s": 100.0 * attempt,
            "cpus": alloc["cpus"],
            "mem_gb": alloc["mem_gb"],
        }

    monkeypatch.setattr(stages, "run_command", run)
    return calls


def last_job(staged, tmp_path, monkeypatch, outcomes, **env):
    out, _ = clone(staged, tmp_path)
    calls = scripted_stitch(monkeypatch, outcomes)
    for key, value in {"REGBENCH_MEM_GB": "16", **env}.items():
        monkeypatch.setenv(key, value)
    with as_jobs():
        done = methods.run_case(
            "stare", staged["case"], out, opts={"stage_images": True}
        )
    return out, done, calls


def test_stitch_is_retried_in_the_last_job_as_the_pipeline_retries_it(
    staged, tmp_path, monkeypatch
):
    """A time limit (143) and a signal (139) are retried with the next attempt; each
    attempt has the pipeline's time limit; the row's write_s is the attempt that wrote
    the image and the failed ones are kept apart."""
    out, done, calls = last_job(staged, tmp_path, monkeypatch, [143, 139, None])
    assert done == ["stare_rigid", "stare"]
    assert [c["attempt"] for c in calls] == [1, 2, 3]
    assert [c["timeout_s"] for c in calls] == [4 * 3600, 8 * 3600, 12 * 3600]
    run = run_json(out, "stare", staged["case"])
    job = run["native"]["stitch_job"]
    assert (job["attempt"], job["cpus"], job["mem_gb"]) == (3, 4, 12)
    assert [(f["attempt"], f["exit"], f["cause"]) for f in job["failed_attempts"]] == [
        (1, 143, "timeout"),
        (2, 139, "exception"),
    ]
    assert job["over_allocation"] is False
    assert run["write_s"] == job["wall_s"] and "write_s" not in run["native"]
    row = __import__("benchmarks.regbench.score", fromlist=["x"]).resource_row(run)
    assert row["total_to_image_s"] == pytest.approx(run["register_s"] + job["wall_s"])
    # the allocation in the record is the benchmark's job, not STITCH's
    assert run["mem_requested_gb"] == 16


@pytest.mark.parametrize(
    "outcomes, cause, attempts",
    [
        ([143, 143, 143, 143], "writer", 4),  # the time limit, four times
        ([137, 137, 137, 137], "writer_oom", 4),
        ([1], "writer", 1),  # an exit the pipeline does not retry
    ],
)
def test_a_stitch_that_fails_after_a_good_registration_is_the_writers(
    staged, tmp_path, monkeypatch, outcomes, cause, attempts
):
    from benchmarks.regbench import failure_log, score

    out, done, calls = last_job(
        staged, tmp_path, monkeypatch, outcomes, REGBENCH_MEM_GB="64"
    )
    assert done == ["stare_rigid"] and len(calls) == attempts
    case = staged["case"]
    run, rigid = run_json(out, "stare", case), run_json(out, "stare_rigid", case)
    assert (run["ok"], run["cause"], run["no_rerun"]) == (False, cause, True)
    assert run["registered"] is True and rigid["ok"] is True
    assert f"stare stitch failed at attempt {attempts}" in run["error"]
    assert failure_log.classify(cause) == failure_log.METHOD
    assert failure_log.stage_of(cause) == "writer"
    # no rung of the benchmark's ladder runs the round again for it
    paths = [result_dir(out, v, case) / "run.json" for v in ("stare_rigid", "stare")]
    for task in (ladder.CLIMB, ladder.AGAIN):
        assert ladder.unit_decision(paths, task, 2)[0] is False
    # the registration is not a wasted attempt: it was done, and the rigid row has it
    base = {"dataset": "multiplex", "group": "P1"}
    rows = [
        {**score.resource_row(r), **base, "method": r["variant"]} for r in (rigid, run)
    ]
    assert rows[1]["registered"] is True and rows[1]["register_s"] > 0
    per = {r["method"]: r for r in score.summarise_resources(rows)}
    assert per["stare"]["wasted_wall_s"] == 0 and per["stare"]["wasted_cpu_h"] == 0
    assert per["stare_rigid"]["register_s"] == pytest.approx(rigid["register_s"])


def test_a_stitch_killed_for_the_jobs_memory_is_the_benchmarks_and_climbs(
    staged, tmp_path, monkeypatch
):
    """STITCH's allocation plus the benchmark's own memory above what the job has:
    the job's limit. The same death in a job with room is STITCH's own attempt."""
    out, done, calls = last_job(
        staged, tmp_path, monkeypatch, [137], REGBENCH_MEM_GB="4"
    )
    case = staged["case"]
    run = run_json(out, "stare", case)
    assert done == ["stare_rigid"] and len(calls) == 1
    assert run["cause"] == "benchmark_oom" and "no_rerun" not in run
    assert "the job's limit, not STITCH's allocation" in run["error"]
    assert run["error"].startswith("the benchmark's job ran out of memory")
    paths = [result_dir(out, v, case) / "run.json" for v in ("stare_rigid", "stare")]
    assert ladder.unit_decision(paths, ladder.CLIMB, 2)[0] is True
    # the next rung: the round's last job again, and what the lost one cost is ITS
    # own time, not the registration the other jobs ran
    monkeypatch.setenv("REGBENCH_ATTEMPT", "2")
    monkeypatch.setenv("REGBENCH_MEM_GB", "64")
    calls.clear()
    monkeypatch.setattr(stages, "run_command", stages.run_command)  # scripted: [137]
    with as_jobs():
        done = methods.run_case(
            "stare", case, out, opts={"stage_images": True}, resume=True
        )
    again = run_json(out, "stare", case)
    (before,) = again["previous_attempts"]
    assert before["cause"] == "benchmark_oom"
    assert before["wall_s"] == run["resources"]["driver_wall_s"]
    assert before["wall_s"] < run["register_s"] / 2
    row = __import__("benchmarks.regbench.score", fromlist=["x"]).resource_row(again)
    # the benchmark's own step ran out of memory: the attempt is lost, not wasted
    assert row["lost_attempts"] == 1 and row["wasted_wall_s"] == 0.0
    assert row["lost_cpu_h_reserved"] == pytest.approx(
        row["cpus_requested"] * before["wall_s"] / 3600
    )


def test_a_command_does_not_outlive_a_call_that_is_interrupted(tmp_path):
    """The driver interrupts the call (its out-of-memory watcher does) while the stage
    command runs: the command is terminated and waited for, and its record still says
    killed."""
    import threading
    import time

    pid_file = tmp_path / "pid"
    code = (
        "import os, time, pathlib; "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        "print('started', flush=True); time.sleep(120)"
    )

    def interrupt():
        while not pid_file.exists() or not pid_file.read_text():
            time.sleep(0.02)
        # as the driver's watcher does it (methods.__init__): a real signal to the
        # main thread of this process, which is blocked reading the command's output
        signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)

    threading.Thread(target=interrupt, daemon=True).start()
    try:
        with pytest.raises(KeyboardInterrupt):
            stages.run_command(tmp_path, stages.STITCH, [PY, "-c", code])
        pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):  # gone, and reaped
            os.kill(pid, 0)
    finally:
        try:  # this test's own child, by the pid it wrote, should the fix be undone
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
        except (OSError, ValueError):
            pass
    rec = stages.read_record(tmp_path, stages.STITCH)
    assert rec["ok"] is False and rec["cause"] == "killed" and rec["exit"] is None


def test_a_command_is_ended_at_its_time_limit_with_a_retried_status(tmp_path):
    marker = tmp_path / "quick"
    code = f"import os, time; time.sleep(0 if os.path.exists({str(marker)!r}) else 30)"
    rec = stages.run_command(tmp_path, stages.STITCH, [PY, "-c", code], timeout_s=0.5)
    assert rec["exit"] == 143 and rec["cause"] == "timeout" and stages.retryable(rec)
    assert rec["wall_s"] < 10
    # every failed attempt of the job is kept, whichever scheduler job ran it
    marker.write_text("")
    again = stages.run_command(tmp_path, stages.STITCH, [PY, "-c", code], timeout_s=20)
    assert again["ok"] and [h["exit"] for h in again["history"]] == [143]


def test_the_placeholder_of_a_last_job_is_any_units_placeholder(
    staged, tmp_path, monkeypatch
):
    """Nothing in it sets the round's last job apart: a death of it is settled by the
    rule of every other task (`test_ladder.py`)."""
    seen = {}
    real = adapter.register

    def spy(case, work, opts):
        path = result_dir(tmp_path / "out", "stare", case) / "run.json"
        seen["placeholder"] = json.loads(path.read_text())
        yield from real(case, work, opts)

    out, _ = clone(staged, tmp_path)
    monkeypatch.setattr(adapter, "register", spy)
    with as_jobs():
        methods.run_case("stare", staged["case"], out, opts={"stage_images": True})
    assert seen["placeholder"]["cause"] == "killed"
    assert "collector" not in seen["placeholder"]
    plain = {"ok", "error", "cause"} | set(seen["placeholder"])
    assert plain == set(seen["placeholder"]) and "phase" not in seen["placeholder"]
