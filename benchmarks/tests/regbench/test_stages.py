"""A registration in several scheduler jobs per round (``regbench.stages``), and STARE run
that way: the pipeline's allocation and retry rule read from the pipeline's own files,
the stage commands, the records, the cost of a round, and the same transform as one
``stare register``.

The real registrations are on the selfcheck's 2048 x 2560 pair (two tiles), made once
for the module.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import re
import shutil
import signal
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import ladder, methods, stages
from benchmarks.regbench import selfcheck as sc
from benchmarks.regbench.cases import result_dir
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
    assert stages.decide(ok, 2, ["x"], force=True)[0] is False  # only the first reruns
    assert stages.decide(ok, 1, ["other"])[0] is True
    assert stages.decide(record(1), 1, ["x"])[0] is True


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
    # COARSE + the longest tile (both its attempts) + SOLVE: no queue, every tile at once
    assert got["longest_tile_s"] == pytest.approx(70.0)
    assert got["wall_unlimited_parallel_s"] == pytest.approx(60.0 + 70.0 + 10.0)
    # every attempt of every job: allocated CPUs x duration, and measured CPU time
    walls = 2 * 60.0 + 2 * (40.0 + 20.0 + 50.0 + 30.0) + 1 * 10.0
    assert got["cpu_h_allocated"] == pytest.approx(walls / 3600.0)
    assert got["cpu_h_measured"] == pytest.approx((90 + 70 + 30 + 80 + 50 + 8) / 3600.0)
    assert (got["largest_job_peak_mb"], got["largest_job"]) == (3900.0, "tile")
    assert got["largest_job_row"] == 1
    assert (got["n_jobs"], got["n_tiles"], got["n_retries"]) == (6, 3, 1)
    # tiles 0, 2 and the lost attempt of tile 1 overlapped: 3 jobs of 2 CPUs
    assert got["peak_cpus_at_once"] == 6
    assert got["unmeasured_attempts"] == 0
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


# ── STARE as the pipeline runs it, for real ───────────────────────────────────
OPTS = {"label": ""}


@contextlib.contextmanager
def as_jobs():
    old = os.environ.get(adapter.JOBS_ENV)
    os.environ[adapter.JOBS_ENV] = "jobs"
    try:
        yield
    finally:
        os.environ.pop(adapter.JOBS_ENV, None)
        if old is not None:
            os.environ[adapter.JOBS_ENV] = old


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """The pair registered stage by stage, each stage a call of its own as a cohort job
    makes it, then collected by the driver; and once by a single `stare register`."""
    import stare.cli

    tmp = tmp_path_factory.mktemp("staged")
    pair = sc.make_pair(tmp / "pair")
    case = sc.pair_case(pair, tmp / "cases")
    out = tmp / "out"
    stage_dir = adapter.stage_dir_of(out, None, case)
    assert stage_dir == result_dir(out, "stare", case) / "stages"
    assert adapter.tile_count(case, OPTS) == 2
    ran = []
    for stage, row in (("coarse", None), ("tile", 0), ("tile", 1), ("solve", None)):
        ran += adapter.run_stage(case, stage_dir, dict(OPTS), stage, row)
    assert [(s, r, go) for s, r, go, _, _ in ran] == [
        ("coarse", None, True),
        ("tile", 0, True),
        ("tile", 1, True),
        ("solve", None, True),
    ]
    assert all(rec["ok"] and rec["execution"] == "jobs" for *_, rec in ran)
    with as_jobs():
        done = methods.run_case("stare", case, out, opts={"stage_images": True})
    assert done == ["stare_rigid", "stare"]
    direct = tmp / "direct"
    assert stare.cli.main(adapter.command(case, direct, {"workers": 2})) == 0
    return {"case": case, "out": out, "dir": stage_dir, "direct": direct, "tmp": tmp}


def run_json(out, variant, case):
    return json.loads((result_dir(out, variant, case) / "run.json").read_text())


def test_stage_by_stage_is_the_transform_of_one_stare_register(staged):
    """Byte for byte: the anchor, the tile plan, every tile's control points and the
    manifest (the transform); SOLVE's report, value for value except its own clock; and
    the written image, pixel for pixel. Equal bytes, not a tolerance: both paths run the
    same stage functions on the same arguments, and nothing in those files names a
    directory."""
    ours, direct = staged["dir"], staged["direct"]
    names = sorted(p.name for p in direct.iterdir() if p.suffix in (".json", ".csv"))
    assert names == [
        "manifest.json",
        "moving_0_0_ctrl.json",
        "moving_1_0_ctrl.json",
        "moving_m0.json",
        "moving_tiles.csv",
        "tre.json",
    ]
    for name in names:
        if name == "tre.json":
            continue
        assert (ours / name).read_bytes() == (direct / name).read_bytes(), name
    # SOLVE's report holds one clock reading, `solve.solve_seconds`: equal without it
    a, b = (json.loads((d / "tre.json").read_text()) for d in (ours, direct))
    assert a["solve"].pop("solve_seconds") > 0 and b["solve"].pop("solve_seconds") > 0
    assert a == b and a["solve"]["solver"] == "dctpls"
    man = json.loads((ours / "manifest.json").read_text())
    assert man["slides"]["moving"]["mesh"] is not None  # a real mesh, not the anchor
    import tifffile

    run = run_json(staged["out"], "stare", staged["case"])
    written = tifffile.imread(run["native"]["files"][0])
    assert np.array_equal(written, tifffile.imread(direct / "registered.ome.tif"))
    assert written.any()


def test_each_stage_command_is_what_stare_register_hands_that_stage(
    staged, monkeypatch, tmp_path
):
    """Every flag of every stage, against the package's own `register`: its four stage
    functions are replaced by recorders and the argument lists compared whole."""
    import stare.cli
    from stare.stages import coarse, reg_tile, solve, stitch

    case = staged["case"]
    ns = adapter.namespace(case, tmp_path, {"workers": 1})
    seen = {}

    def plan(argv):  # the recorder of COARSE leaves the plan `register` reads next
        seen.setdefault("coarse", []).append(list(argv))
        shutil.copy(staged["dir"] / "moving_tiles.csv", tmp_path / "moving_tiles.csv")
        return 0

    monkeypatch.setattr(coarse, "main", plan)
    for name, mod in (("tile", reg_tile), ("solve", solve), ("stitch", stitch)):
        monkeypatch.setattr(
            mod,
            "main",
            lambda argv, n=name: seen.setdefault(n, []).append(list(argv)) or 0,
        )
    assert stare.cli.register(ns) == 0
    ours = adapter.stage_commands(ns)
    assert ours["coarse"] == seen["coarse"][0]
    assert [ours["tile"](0, 0, 0), ours["tile"](1, 1, 0)] == seen["tile"]
    assert sorted(ours["solve"]) == sorted(seen["solve"][0])
    assert ours["stitch"](ns.moving, ns.out) == seen["stitch"][0]
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
    with as_jobs():
        assert methods.run_case("stare", case, out, opts={"stage_images": True}) == []
    run = run_json(out, "stare", case)
    assert run["cause"] == "infrastructure" and run["no_rerun"] is True
    assert "COARSE: no attempt of this job left a record" in run["error"]
    # the stages behind a stage that is not ok do nothing
    for stage in ("tile", "solve"):
        got = adapter.run_stage(case, new, dict(OPTS), stage, force=True)
        assert [(go, why, rec) for _, _, go, why, rec in got] == [
            (False, "COARSE is not ok", None)
        ]


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
    assert "TILE (tile row 1): this job was never submitted" in run["error"]
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


def test_the_stage_command_of_the_cli_runs_a_round(
    synthetic_cases, cases_root, tmp_path, capsys
):
    """`regbench stage`, as stage.sbatch calls it: by round, and a tile by its place in
    the layout the launcher wrote."""
    from benchmarks.regbench import cli

    names = sorted(synthetic_cases)
    index = names.index("crop")
    layout = tmp_path / "layout.tsv"
    base = ["--cases", str(cases_root), "--dataset", "synthetic", "--method", "stare"]
    assert cli.main(["stages", *base, "--layout", str(layout)]) == 0
    rows = stages.read_layout(layout)
    assert [c for c, _ in rows] == names and all(n == 1 for _, n in rows)
    flat = sum(n for _, n in rows[:index])
    run = ["stage", *base, "--out", str(tmp_path / "out")]
    assert (
        cli.main(
            [*run, "--stage", "tile", "--flat", str(flat), "--layout", str(layout)]
        )
        == 0
    )
    case = synthetic_cases["crop"]
    stage_dir = adapter.stage_dir_of(tmp_path / "out", None, case)
    assert not stages.record_path(stage_dir, "tile", 0).exists()  # COARSE has not run
    # what the tile submitter asks: nothing is due of a round without COARSE
    due = ["stages", *base, "--layout", str(layout), "--out", str(tmp_path / "out")]
    capsys.readouterr()
    assert cli.main([*due, "--due", "1"]) == 0 and capsys.readouterr().out == ""
    assert cli.main([*run, "--stage", "coarse", "--index", str(index)]) == 0
    # ... then that round's tile, by its place in the layout; no retry is due
    capsys.readouterr()
    assert cli.main([*due, "--due", "1"]) == 0
    assert capsys.readouterr().out == f"{flat}\n"
    # a tile without a record at a later attempt never started: its round is due
    assert cli.main([*due, "--due", "2"]) == 0
    assert capsys.readouterr().out == f"{index}\n"
    assert (
        cli.main(
            [*run, "--stage", "tile", "--flat", str(flat), "--layout", str(layout)]
        )
        == 0
    )
    capsys.readouterr()
    for attempt in ("1", "2"):
        assert cli.main([*due, "--due", attempt]) == 0
        assert capsys.readouterr().out == ""
    assert (
        cli.main([*run, "--stage", "solve", "--index", str(index), "--attempt", "2"])
        == 0
    )
    assert all(
        stages.read_record(stage_dir, s, r)["ok"]
        for s, r in (("coarse", None), ("tile", 0), ("solve", None))
    )
    assert (stage_dir / "manifest.json").exists()
    assert (
        cli.main([*run, "--stage", "tile", "--flat", "99", "--layout", str(layout)])
        == 0
    )
    # arrays sized for another set of cases are refused
    stages.write_layout(layout, rows[:-1])
    with pytest.raises(SystemExit, match="does not list"):
        cli.main([*run, "--stage", "tile", "--flat", "0", "--layout", str(layout)])


def test_the_preprocessing_row_runs_in_the_same_jobs(synthetic_cases, tmp_path):
    """`stare_prep` is the same four stages behind a `prep` job of the benchmark's own:
    STARE registers the preprocessed copies, STITCH draws the original slide."""
    from benchmarks.regbench import prep

    case = synthetic_cases["crop"]
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
    staged, synthetic_cases, tmp_path
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
def test_the_batch_scripts_register_a_round_end_to_end(
    synthetic_cases, cases_root, tmp_path
):
    """stage.sbatch for COARSE, a tile by its place in the layout and SOLVE, then
    run_rung.sbatch as the round's last job: the scripts' own argument handling, with
    the python on PATH standing in for the image."""
    import subprocess

    from benchmarks.regbench import cli

    slurm = REPO / "benchmarks" / "regbench" / "slurm"
    names = sorted(synthetic_cases)
    index = names.index("crop")
    out = tmp_path / "out"
    layout = out / "logs" / "stages" / "S-1" / "stare-synthetic.tsv"
    base = ["--cases", str(cases_root), "--dataset", "synthetic", "--method", "stare"]
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
        SLURM_JOB_NAME="rb-stare-synthetic",
    )
    head = ["stare", "stare", "-", "-", "STARE_REGBENCH", "synthetic"]

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
    case = synthetic_cases["crop"]
    stage_dir = adapter.stage_dir_of(out, None, case)
    solve = stages.read_record(stage_dir, "solve")
    assert solve["ok"] and solve["attempt"] == 2 and solve["mem_gb"] == 2
    assert solve["launch_id"] == f"900_{index}" and solve["submission"] == "S-1"
    assert solve["log_path"].endswith(f"rb-stare-synthetic-900_{index}.log")
    # the round's last job: nothing of the registration runs in it
    said = sbatch(
        "run_rung.sbatch",
        "0",
        "synthetic",
        "1",
        "4",
        "0",
        "-",
        "first",
        "0-04:00:00",
        "jobs",
    )
    assert "synthetic/crop [stare]: stare_rigid, stare" in said
    run = run_json(out, "stare", case)
    assert run["ok"] and run["info"]["jobs"]["execution"] == "jobs"
    assert run["attempt"] == 1 and run["mem_requested_gb"] == 4
    assert run["submissions"] == ["S-1"] and run["time_limit"] == "0-04:00:00"
    assert run["native"]["files"] and stages.read_record(stage_dir, "stitch")["ok"]
    # ... and without the stages, that job fails the round instead of running them
    other = synthetic_cases[names[index - 1]]
    said = sbatch(
        "run_rung.sbatch",
        "0",
        "synthetic",
        "1",
        "4",
        "0",
        "-",
        "first",
        "0-04:00:00",
        "jobs",
        task=index - 1,
    )
    lost = run_json(out, "stare", other)
    assert (
        lost["ok"] is False and lost["cause"] == "infrastructure" and lost["no_rerun"]
    )
    assert not adapter.stage_dir_of(out, None, other).exists()


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
    from benchmarks.regbench.datasets import synthetic

    root = tmp_path / "cases"
    synthetic.prepare_one(root, "crop", synthetic.SUITES["test"]["crop"])
    case = load_case(root / "synthetic" / "crop")
    out, lost = tmp_path / "out", tmp_path / "lost"
    assert register_as_jobs(case, out) == ["stare_rigid", "stare"]

    def scored(where):
        got = score.score(
            root,
            where,
            datasets=["synthetic"],
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
