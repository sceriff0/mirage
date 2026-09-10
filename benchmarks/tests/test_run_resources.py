"""benchmarks.analysis.run_resources: one real run -> tidy resource tables.

The fixture is a small synthetic run written into tmp_path: a four-slide cohort where
CONVERT_IMAGE ran once per slide with inputs 1, 2, 3, 4 GiB (so peak RSS and wall-time
CAN be fitted against input size), MERGE_AND_PYRAMID ran once (no fit possible), one
task failed, and one row carries Nextflow's `-` not-run sentinel everywhere. The
numbers are chosen so every derived column has a hand-computable expected value.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.analysis import run_resources as rr

TRACE_HEADER = (
    "task_id\tprocess\ttag\tname\tstatus\texit\tsubmit\tstart\tcomplete\tduration\t"
    "realtime\t%cpu\tcpus\tmemory\tpeak_rss\tpeak_vmem\trchar\twchar\n"
)


def _row(i, proc, tag, status, exit_, start, complete, realtime, pct, cpus, mem, rss):
    return (
        f"{i}\t{proc}\t{tag}\t{proc} ({tag})\t{status}\t{exit_}\t-\t{start}\t{complete}\t"
        f"{realtime}\t{realtime}\t{pct}\t{cpus}\t{mem}\t{rss}\t{rss}\t1 GB\t1 GB\n"
    )


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "my_run"
    (run / "trace").mkdir(parents=True)
    (run / "size_logs").mkdir()
    (run / "qc").mkdir()
    rows = [TRACE_HEADER]
    # CONVERT_IMAGE: 4 slides, 1..4 GiB in, RSS = 2 + 3 x input, realtime = 600 x input.
    for i, gb in enumerate((1, 2, 3, 4), start=1):
        rows.append(
            _row(
                i,
                "CONVERT_IMAGE",
                f"P00{i}",
                "COMPLETED",
                0,
                f"2026-09-10 10:0{i}:00.000",
                f"2026-09-10 11:0{i}:00.000",
                f"{10 * gb}m",
                "200%",
                4,
                "32 GB",
                f"{2 + 3 * gb} GB",
            )
        )
    # MERGE_AND_PYRAMID: once, 1 h, 50% of 8 cpus, 64 GB requested, 16 GB used.
    rows.append(
        _row(
            5,
            "MERGE_AND_PYRAMID",
            "P001",
            "COMPLETED",
            0,
            "2026-09-10 12:00:00.000",
            "2026-09-10 13:00:00.000",
            "1h",
            "400%",
            8,
            "64 GB",
            "16 GB",
        )
    )
    # a failed attempt and a cached row (the `-` sentinel is not a failure)
    rows.append(
        _row(
            6,
            "SEGMENT",
            "P001",
            "FAILED",
            137,
            "2026-09-10 13:00:00.000",
            "2026-09-10 13:30:00.000",
            "30m",
            "100%",
            2,
            "8 GB",
            "8 GB",
        )
    )
    rows.append(
        "7\tSEGMENT\tP002\tSEGMENT (P002)\tCACHED\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\n"
    )
    (run / "trace" / "trace.txt").write_text("".join(rows))

    size = ["process,sample_id,filename,bytes\n"]
    for i, gb in enumerate((1, 2, 3, 4), start=1):
        size.append(f"CONVERT_IMAGE,P00{i},P00{i}.ome.tif,{gb * 2**30}\n")
    size.append(f"MERGE_AND_PYRAMID,P001,merged.ome.tif,{10 * 2**30}\n")
    (run / "size_logs" / "input_sizes.csv").write_text("".join(size))
    (run / "qc" / "run_summary.json").write_text(
        '{"pipeline": {"name": "mirage", "version": "1.0.0"}, "run": {"mode": "standard"}}'
    )
    return run


def test_tasks_carry_derived_cost_columns_and_the_joined_input(run_dir, tmp_path):
    paths = rr.profile_run(run_dir, None, tmp_path / "out")
    tasks = pd.read_csv(paths["run_resources_tasks"])
    assert list(tasks.columns) == list(rr.TASK_COLUMNS)
    assert len(tasks) == 7
    c1 = tasks[(tasks.process == "CONVERT_IMAGE") & (tasks.tag == "P001")].iloc[0]
    assert c1.input_gb == 1.0
    assert c1.realtime_s == 600
    assert c1.cpu_h_used == pytest.approx(600 * 2.0 / 3600)  # 200% of one core
    assert c1.cpu_h_reserved == pytest.approx(600 * 4 / 3600)
    assert c1.peak_rss_gb == 5.0 and c1.memory_req_gb == 32.0
    assert c1.rss_utilisation == pytest.approx(5 / 32)
    assert c1.gb_h_reserved == pytest.approx(32 * 600 / 3600)
    # the failure flag: exit 137 is one, the `-` sentinel is not
    assert tasks.loc[tasks.exit.astype(str) == "137", "failed"].all()
    cached = tasks[tasks.status == "CACHED"].iloc[0]
    assert (
        not cached.failed
        and math.isnan(cached.realtime_s)
        and math.isnan(cached.input_gb)
    )


def test_process_rollup_sums_and_shares(run_dir, tmp_path):
    paths = rr.profile_run(run_dir, None, tmp_path / "out")
    procs = pd.read_csv(paths["run_resources_processes"]).set_index("process")
    conv = procs.loc["CONVERT_IMAGE"]
    assert conv.n_tasks == 4 and conv.n_failed == 0
    assert conv.realtime_total_h == pytest.approx((10 + 20 + 30 + 40) * 60 / 3600)
    assert conv.realtime_max_h == pytest.approx(40 * 60 / 3600)
    assert conv.input_gb_total == 10.0 and conv.input_gb_mean == 2.5
    assert conv.peak_rss_max_gb == 14.0 and conv.memory_req_max_gb == 32.0
    merge = procs.loc["MERGE_AND_PYRAMID"]
    assert merge.cpu_efficiency == pytest.approx(0.5)  # 400% of 8 cpus
    seg = procs.loc["SEGMENT"]
    assert seg.n_tasks == 2 and seg.n_failed == 1
    # shares of all task time add up to one
    assert procs.wall_share.sum() == pytest.approx(1.0)


def test_fits_exist_only_where_the_run_has_the_points(run_dir, tmp_path):
    paths = rr.profile_run(run_dir, None, tmp_path / "out")
    fits = pd.read_csv(paths["run_resources_fits"]).set_index(["process", "target"])
    rss = fits.loc[("CONVERT_IMAGE", "peak_rss_gb")]
    assert rss.fit_ok and rss.n == 4
    assert rss.slope == pytest.approx(3.0) and rss.intercept == pytest.approx(2.0)
    assert rss.r2 == pytest.approx(1.0)
    rt = fits.loc[("CONVERT_IMAGE", "realtime_s")]
    assert (
        rt.fit_ok
        and rt.slope == pytest.approx(600.0)
        and rt.intercept == pytest.approx(0.0, abs=1e-6)
    )
    # one task: no fit, and NOT a flat line pretending to be one
    merge = fits.loc[("MERGE_AND_PYRAMID", "peak_rss_gb")]
    assert not merge.fit_ok and merge.n == 1 and math.isnan(merge.slope)
    # SEGMENT has no size-log rows at all -> n == 0, no fit
    assert fits.loc[("SEGMENT", "peak_rss_gb")].n == 0


def test_three_tasks_of_one_size_do_not_get_a_fit(tmp_path):
    """Three retries of one slide are three points at one x: least squares would
    still return a line, and fit_ok must say it is not one."""
    tasks = pd.DataFrame(
        {
            "process": ["REGISTER"] * 3,
            "input_gb": [4.0, 4.0, 4.0],
            "peak_rss_gb": [10.0, 12.0, 11.0],
            "realtime_s": [100.0, 110.0, 105.0],
        }
    )
    fits = rr.fits_frame(tasks).set_index("target")
    assert fits.loc["peak_rss_gb"].n == 3
    assert not fits.loc["peak_rss_gb"].fit_ok
    assert math.isnan(fits.loc["peak_rss_gb"].slope)


def test_summary_totals_wall_clock_and_raw_input(run_dir, tmp_path):
    paths = rr.profile_run(run_dir, None, tmp_path / "out")
    s = pd.read_csv(paths["run_resources_summary"]).iloc[0]
    assert s.run == "my_run" and s.mirage_version == "1.0.0"
    assert s.n_tasks == 7 and s.n_processes == 3 and s.n_failed == 1
    # earliest start 10:01, latest complete 13:30
    assert s.wall_clock_h == pytest.approx(3.0 + 29 / 60)
    assert s.raw_input_gb == pytest.approx(10.0)
    assert s.peak_rss_max_gb == 16.0
    assert s.task_realtime_total_h == pytest.approx((100 * 60 + 3600 + 1800) / 3600)


def test_every_emitted_column_is_in_the_dictionary(run_dir, tmp_path):
    paths = rr.profile_run(run_dir, None, tmp_path / "out")
    doc = paths["dictionary"].read_text()
    for key in (
        "run_resources_tasks",
        "run_resources_processes",
        "run_resources_fits",
        "run_resources_summary",
    ):
        for col in pd.read_csv(paths[key], nrows=0).columns:
            assert f"| `{col}` |" in doc, f"{key}.{col} undocumented"
    # and the dictionary refuses an undocumented column rather than skipping it
    with pytest.raises(KeyError):
        rr.write_dictionary(tmp_path, {"x": pd.DataFrame({"mystery": [1]})})


def test_trace_is_found_in_the_bare_run_layout_and_explicit_path_wins(
    run_dir, tmp_path
):
    # bare-run layout: <outdir>/../.trace/trace.txt (trace_dir defaults beside the launch dir)
    bare = tmp_path / "bare"
    (bare / "results").mkdir(parents=True)
    (bare / ".trace").mkdir()
    (bare / ".trace" / "trace.txt").write_text(TRACE_HEADER)
    assert rr.find_trace(bare / "results", None) == bare / ".trace" / "trace.txt"
    explicit = run_dir / "trace" / "trace.txt"
    assert rr.find_trace(bare / "results", str(explicit)) == explicit
    assert rr.find_trace(bare / "results", str(tmp_path / "nope.txt")) is None


def test_missing_or_empty_trace_is_an_error_not_an_empty_table(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        rr.profile_run(empty, None, tmp_path / "out")
    (empty / "trace").mkdir()
    (empty / "trace" / "trace.txt").write_text(TRACE_HEADER)
    with pytest.raises(ValueError):
        rr.profile_run(empty, None, tmp_path / "out")
    assert rr.main(["--run", str(empty), "--outdir", str(tmp_path / "out")]) == 3


def test_cli_writes_the_five_files_and_reports(run_dir, tmp_path, capsys):
    rc = rr.main(["--run", str(run_dir), "--outdir", str(tmp_path / "out")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "7 tasks / 3 processes, 1 failed" in out
    names = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert names == [
        "run_resources.dict.md",
        "run_resources_fits.csv",
        "run_resources_processes.csv",
        "run_resources_summary.csv",
        "run_resources_tasks.csv",
    ]


def test_the_shipped_fixture_run_profiles(tmp_path):
    """The sweep's own fixture (benchmarks/tests/fixtures/runs/run0000) is a valid
    input too -- its trace sits at <run>/trace and its size log under out/."""
    fixture = Path(__file__).parent / "fixtures" / "runs" / "run0000"
    paths = rr.profile_run(fixture, None, tmp_path / "out")
    s = pd.read_csv(paths["run_resources_summary"]).iloc[0]
    assert s.n_tasks == 2 and s.raw_input_gb == pytest.approx(2.0)
