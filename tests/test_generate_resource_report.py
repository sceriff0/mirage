import importlib.util
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "bin" / "generate_resource_report.py"


def _load():
    spec = importlib.util.spec_from_file_location("grr", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_bytes():
    grr = _load()
    assert grr.parse_bytes("3.2 GB") == round(3.2 * 1024**3, 1)
    assert grr.parse_bytes("512 MB") == 512 * 1024**2
    assert grr.parse_bytes("-") is None
    assert grr.parse_bytes("") is None


def test_parse_bytes_inert_set_condition_is_behavior_preserving():
    """A bare "0" is a genuine zero-byte reading; only "-" (not-run/cached)
    and empty are missing, as in parse_duration and parse_percent."""
    grr = _load()
    assert grr.parse_bytes("0") == 0.0
    assert grr.parse_bytes("-") is None
    assert grr.parse_bytes("") is None


def test_parse_duration():
    grr = _load()
    assert grr.parse_duration("1.5s") == 1.5
    assert grr.parse_duration("12m 4s") == 724.0
    assert grr.parse_duration("2h 1m") == 7260.0
    assert grr.parse_duration("-") is None


def test_parse_percent():
    grr = _load()
    assert grr.parse_percent("142.3%") == 142.3
    assert grr.parse_percent("-") is None


def test_parse_trace(tmp_path):
    grr = _load()
    t = tmp_path / "trace.txt"
    t.write_text(
        "task_id\tprocess\ttag\tname\tstatus\texit\tsubmit\tstart\tcomplete\t"
        "duration\trealtime\t%cpu\tcpus\tmemory\tpeak_rss\tpeak_vmem\trchar\twchar\n"
        "1\tMIRAGE:PRE:CONVERT_IMAGE\tP001\tname\tCOMPLETED\t0\t-\t-\t-\t"
        "12m 4s\t10m\t142.3%\t8\t8 GB\t3.2 GB\t4 GB\t1 GB\t500 MB\n"
    )
    rows = grr.parse_trace(t)
    assert len(rows) == 1
    r = rows[0]
    assert r["process"] == "MIRAGE:PRE:CONVERT_IMAGE"
    assert r["tag"] == "P001"
    assert r["realtime_s"] == 600.0
    assert r["peak_rss_b"] == round(3.2 * 1024**3, 1)
    assert r["cpu_pct"] == 142.3
    assert r["exit"] == "0"


def test_parse_size_log(tmp_path):
    grr = _load()
    p = tmp_path / "input_sizes.csv"
    p.write_text(
        "process,sample_id,filename,bytes\n"
        "MIRAGE:PRE:CONVERT_IMAGE,P001,a.tiff,100\n"
        "MIRAGE:PRE:CONVERT_IMAGE,P001,b.tiff,50\n"
        # A stub row: the real process name, zero bytes.
        "MIRAGE:PRE:SPLIT_CHANNELS,P001,stub,0\n"
    )
    m = grr.parse_size_log(p)
    assert m[("MIRAGE:PRE:CONVERT_IMAGE", "P001")] == 150
    assert m[("MIRAGE:PRE:SPLIT_CHANNELS", "P001")] == 0


def test_parse_size_log_rejects_a_foreign_header(tmp_path):
    """A reordered or renamed header must fail loudly, not be attributed wrongly.

    csv.DictReader keys on names, so a swapped `filename`/`bytes` pair would make
    every row's byte count the string 'a.tiff' -- caught by the int() guard and
    silently counted as 0. A report of all-zero input sizes looks like a run that
    read nothing, not like a bug.
    """
    grr = _load()
    p = tmp_path / "input_sizes.csv"
    p.write_text("process,sample_id,bytes,filename\nA,P001,100,a.tiff\n")
    with pytest.raises(ValueError, match="unexpected size-log header"):
        grr.parse_size_log(p)


def test_rollup_by_process():
    grr = _load()
    rows = [
        {
            "process": "A",
            "tag": "P1",
            "exit": "0",
            "realtime_s": 10.0,
            "cpu_pct": 100.0,
            "peak_rss_b": 200.0,
            "peak_vmem_b": 300.0,
            "rchar_b": 5.0,
            "wchar_b": 2.0,
        },
        {
            "process": "A",
            "tag": "P2",
            "exit": "0",
            "realtime_s": 30.0,
            "cpu_pct": 150.0,
            "peak_rss_b": 400.0,
            "peak_vmem_b": 500.0,
            "rchar_b": 7.0,
            "wchar_b": 1.0,
        },
    ]
    roll = {r["process"]: r for r in grr.rollup_by_process(rows)}
    a = roll["A"]
    assert a["n_tasks"] == 2
    assert a["realtime_total_s"] == 40.0
    assert a["realtime_mean_s"] == 20.0
    assert a["peak_rss_max_b"] == 400.0
    assert a["cpu_max_pct"] == 150.0


def test_join_size_exact_and_fallback():
    grr = _load()
    trace = [
        {"process": "A", "tag": "P001", "realtime_s": 1.0, "peak_rss_b": 10.0},
        {"process": "A", "tag": "P001_slideX", "realtime_s": 2.0, "peak_rss_b": 20.0},
    ]
    size = {("A", "P001"): 999}
    joined = grr.join_size(trace, size)
    assert joined[0]["input_bytes"] == 999  # exact (process, tag)
    assert joined[1]["input_bytes"] == 999  # fallback: same process, sample prefix


def test_join_size_prefix_boundary_no_false_match():
    """A size sample "P1" must not prefix-match a trace tag "P10_slide" —
    only an exact match or a "<sample>_"-bounded prefix should join."""
    grr = _load()
    trace = [
        {"process": "A", "tag": "P10_slideX", "realtime_s": 1.0, "peak_rss_b": 10.0},
        {"process": "A", "tag": "P1_slideY", "realtime_s": 2.0, "peak_rss_b": 20.0},
        {"process": "A", "tag": "P1", "realtime_s": 3.0, "peak_rss_b": 30.0},
    ]
    size = {("A", "P1"): 111}
    joined = grr.join_size(trace, size)
    by_tag = {j["tag"]: j["input_bytes"] for j in joined}
    assert by_tag["P10_slideX"] is None  # P1 must not falsely match P10_...
    assert by_tag["P1_slideY"] == 111  # legitimate "<sample>_" boundary match
    assert by_tag["P1"] == 111  # exact match


def test_rollup_by_process_exit_dash_not_counted_as_failure():
    """A trace row with exit="-" (cached/aborted/not-run) must not be
    counted as a failure, either in the per-process rollup or the run-level
    failure count / Retries & Failures section."""
    grr = _load()
    rows = [
        {
            "process": "A",
            "tag": "P1",
            "exit": "-",
            "realtime_s": 1.0,
            "cpu_pct": None,
            "peak_rss_b": None,
            "peak_vmem_b": None,
            "rchar_b": None,
            "wchar_b": None,
        },
        {
            "process": "A",
            "tag": "P2",
            "exit": "0",
            "realtime_s": 1.0,
            "cpu_pct": None,
            "peak_rss_b": None,
            "peak_vmem_b": None,
            "rchar_b": None,
            "wchar_b": None,
        },
    ]
    roll = {r["process"]: r for r in grr.rollup_by_process(rows)}
    assert roll["A"]["n_failed"] == 0


def test_build_html_exit_dash_excluded_from_failures():
    grr = _load()
    trace_rows = [
        {
            "process": "A",
            "tag": "P1",
            "status": "ABORTED",
            "exit": "-",
            "realtime_s": 1.0,
            "peak_rss_b": None,
            "peak_vmem_b": None,
            "rchar_b": None,
            "wchar_b": None,
            "cpu_pct": None,
            "duration_s": None,
        },
    ]
    html_out = grr.build_html(trace_rows, {}, "ts")
    assert "<tr><th>Failed/non-zero exit</th><td>0</td></tr>" in html_out
    assert "No failed or non-zero-exit tasks." in html_out


def test_cli_writes_report(tmp_path):
    trace = tmp_path / "trace.txt"
    trace.write_text(
        "task_id\tprocess\ttag\tname\tstatus\texit\tsubmit\tstart\tcomplete\t"
        "duration\trealtime\t%cpu\tcpus\tmemory\tpeak_rss\tpeak_vmem\trchar\twchar\n"
        "1\tMIRAGE:PRE:CONVERT_IMAGE\tP001\tn\tCOMPLETED\t0\t-\t-\t-\t"
        "12m\t10m\t142%\t8\t8 GB\t3.2 GB\t4 GB\t1 GB\t500 MB\n"
        "2\tMIRAGE:REG:REGISTER\tP001\tn\tFAILED\t1\t-\t-\t-\t"
        "1h\t1h\t90%\t8\t8 GB\t7 GB\t9 GB\t2 GB\t1 GB\n"
    )
    size = tmp_path / "input_sizes.csv"
    size.write_text(
        "process,sample_id,filename,bytes\n"
        "MIRAGE:PRE:CONVERT_IMAGE,P001,a.tiff,1073741824\n"
    )
    out = tmp_path / "resource.html"
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace",
            str(trace),
            "--size-log",
            str(size),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    html = out.read_text()
    assert "Resource" in html
    assert "MIRAGE:PRE:CONVERT_IMAGE" in html
    assert "MIRAGE:REG:REGISTER" in html
    assert "Retries" in html or "Failures" in html


def test_cli_exits_nonzero_on_a_header_only_trace(tmp_path):
    """A header-only trace.txt (a run killed before its first task) exits
    non-zero, so the entry script's warning branch fires instead of a false
    success."""
    trace = tmp_path / "trace.txt"
    trace.write_text(
        "task_id\tprocess\ttag\tname\tstatus\texit\tsubmit\tstart\tcomplete\t"
        "duration\trealtime\t%cpu\tcpus\tmemory\tpeak_rss\tpeak_vmem\trchar\twchar\n"
    )
    out = tmp_path / "resource.html"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--trace", str(trace), "--output", str(out)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 3, r.stderr
    assert out.exists()
    assert "not available" in out.read_text().lower()


def test_no_trace_message_does_not_show_the_broken_cli_boolean_form():
    """Nextflow 26 rejects `--enable_trace` on the command line (every boolean
    param must come from a profile or -params-file); the operator-facing
    "not available" message must not tell someone to type the one form that
    cannot work."""
    grr = _load()
    html_out = grr.build_html([], {}, "ts")
    assert "--enable_trace" not in html_out
    assert "enable_trace" in html_out


def test_cli_missing_inputs_is_graceful(tmp_path):
    out = tmp_path / "resource.html"
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace",
            str(tmp_path / "nope.txt"),
            "--size-log",
            str(tmp_path / "nope.csv"),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert out.exists()
    assert "not available" in out.read_text().lower()


def test_build_html_drops_a_zero_byte_matched_input_from_the_scatter():
    """The scatter cannot place a zero on a log axis, so it drops the point and
    says how many it dropped."""
    grr = _load()
    trace_rows = [
        {
            "process": "A",
            "tag": "P001",
            "exit": "0",
            "status": "COMPLETED",
            "realtime_s": 1.0,
            "peak_rss_b": 100.0,
            "peak_vmem_b": None,
            "rchar_b": None,
            "wchar_b": None,
            "cpu_pct": None,
            "duration_s": None,
            "memory_b": None,
            "cpus": None,
        },
    ]
    html_out = grr.build_html(trace_rows, {("A", "P001"): 0}, "ts")

    assert isinstance(html_out, str)
    assert "not shown" in html_out or "class='empty'" in html_out


def test_build_html_escapes_a_hostile_process_and_tag_in_the_panels():
    """The tag reaches the report only through the scatter's tooltip, so the
    fixture needs a matching size-log entry or the escaping assertion would
    pass vacuously."""
    grr = _load()
    trace_rows = [
        {
            "process": "A & B",
            "tag": "<b>P001</b>",
            "status": "COMPLETED",
            "exit": "0",
            "realtime_s": 60.0,
            "peak_rss_b": 10.0 * GB,
            "peak_vmem_b": None,
            "rchar_b": None,
            "wchar_b": None,
            "cpu_pct": None,
            "duration_s": None,
            "memory_b": 32.0 * GB,
            "cpus": 8,
        },
    ]
    size_map = {("A & B", "<b>P001</b>"): 2 * GB}

    html_out = grr.build_html(trace_rows, size_map, "ts")

    # Both must actually be RENDERED before their escaping means anything.
    assert "&lt;b&gt;P001&lt;/b&gt;" in html_out
    assert "A &amp; B" in html_out
    assert "<b>P001</b>" not in html_out
    assert "A & B" not in html_out


def test_build_html_escapes_a_hostile_process_name_in_the_failure_panel():
    """The process name reaches the failure panel's labels and tooltips and
    must be escaped there."""
    grr = _load()
    trace_rows = [
        {
            "process": "A & <b>B</b>",
            "tag": "P1",
            "status": "<i>FAILED</i>",
            "exit": "1 & 2",
            "realtime_s": 1.0,
            "peak_rss_b": 1024.0,
            "peak_vmem_b": None,
            "rchar_b": None,
            "wchar_b": None,
            "cpu_pct": None,
            "duration_s": None,
            "memory_b": None,
            "cpus": None,
        },
    ]
    html_out = grr.build_html(trace_rows, {}, "ts")

    assert "<b>B</b>" not in html_out
    assert "&lt;b&gt;B&lt;/b&gt;" in html_out
    assert "<i>FAILED</i>" not in html_out


def test_run_totals_reports_the_peak_single_task_cpu():
    """Run Totals reports the peak single-task %CPU."""
    grr = _load()
    trace_rows = [
        {
            "process": "A",
            "tag": "P1",
            "status": "COMPLETED",
            "exit": "0",
            "realtime_s": 1.0,
            "peak_rss_b": None,
            "peak_vmem_b": None,
            "rchar_b": None,
            "wchar_b": None,
            "cpu_pct": 87.5,
            "duration_s": None,
            "memory_b": None,
            "cpus": None,
        },
    ]
    html_out = grr.build_html(trace_rows, {}, "ts")

    marker = "<h2>Run Totals</h2>"
    section = html_out[html_out.index(marker) :]
    section = section[: section.index("</section>")]
    assert "87.5%" in section


def test_parse_trace_reads_requested_memory_and_cpus():
    """The trace's `memory` column is the REQUEST. Without it there is no headroom
    to plot -- only observed peak_rss, which cannot show over-provisioning."""
    grr = _load()
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as d:
        t = _P(d) / "trace.txt"
        t.write_text(
            "task_id\tprocess\ttag\tname\tstatus\texit\tsubmit\tstart\tcomplete\t"
            "duration\trealtime\t%cpu\tcpus\tmemory\tpeak_rss\tpeak_vmem\trchar\twchar\n"
            "1\tA\tP001\tn\tCOMPLETED\t0\t-\t-\t-\t12m\t10m\t142%\t8\t100 GB\t3.2 GB\t4 GB\t1 GB\t500 MB\n"
            "2\tA\tP002\tn\tCOMPLETED\t0\t-\t-\t-\t12m\t10m\t142%\t-\t-\t3.2 GB\t4 GB\t1 GB\t500 MB\n"
        )
        rows = grr.parse_trace(t)

    assert rows[0]["memory_b"] == 100 * 1024**3
    assert rows[0]["cpus"] == 8
    assert rows[1]["memory_b"] is None  # "-" is missing, not zero
    assert rows[1]["cpus"] is None


def test_rollup_carries_the_max_request_and_the_cost_of_failures():
    grr = _load()
    rows = [
        # succeeded: 100 GB requested for 1 h
        {
            "process": "A",
            "tag": "P1",
            "exit": "0",
            "status": "COMPLETED",
            "realtime_s": 3600.0,
            "cpu_pct": None,
            "peak_rss_b": 50 * 1024**3,
            "peak_vmem_b": None,
            "rchar_b": None,
            "wchar_b": None,
            "memory_b": 100 * 1024**3,
            "cpus": 8,
        },
        # failed after 30 min holding a 200 GB reservation -> 100 GB.h thrown away
        {
            "process": "A",
            "tag": "P1",
            "exit": "137",
            "status": "FAILED",
            "realtime_s": 1800.0,
            "cpu_pct": None,
            "peak_rss_b": 190 * 1024**3,
            "peak_vmem_b": None,
            "rchar_b": None,
            "wchar_b": None,
            "memory_b": 200 * 1024**3,
            "cpus": 8,
        },
    ]
    a = {r["process"]: r for r in grr.rollup_by_process(rows)}["A"]

    assert a["n_failed"] == 1
    assert a["mem_req_max_b"] == 200 * 1024**3
    assert a["failed_realtime_s"] == 1800.0
    assert a["failed_gb_h"] == pytest.approx(100.0, rel=1e-6)


def test_failed_gb_h_falls_back_to_observed_rss_when_no_request_is_recorded():
    grr = _load()
    rows = [
        {
            "process": "A",
            "tag": "P1",
            "exit": "1",
            "status": "FAILED",
            "realtime_s": 7200.0,
            "cpu_pct": None,
            "peak_rss_b": 10 * 1024**3,
            "peak_vmem_b": None,
            "rchar_b": None,
            "wchar_b": None,
            "memory_b": None,
            "cpus": None,
        },
    ]
    a = {r["process"]: r for r in grr.rollup_by_process(rows)}["A"]
    assert a["failed_gb_h"] == pytest.approx(20.0, rel=1e-6)


def _roll_row(proc, total_s, **kw):
    row = {
        "process": proc,
        "n_tasks": 1,
        "realtime_total_s": total_s,
        "realtime_mean_s": total_s,
        "cpu_max_pct": None,
        "peak_rss_max_b": None,
        "peak_vmem_max_b": None,
        "rchar_total_b": 0.0,
        "wchar_total_b": 0.0,
        "n_failed": 0,
        "mem_req_max_b": None,
        "failed_realtime_s": 0.0,
        "failed_gb_h": 0.0,
    }
    row.update(kw)
    return row


def test_short_process_keeps_only_the_leaf_name():
    grr = _load()
    assert grr.short_process("MIRAGE:PRE:CONVERT_IMAGE") == "CONVERT_IMAGE"
    assert grr.short_process("CONVERT_IMAGE") == "CONVERT_IMAGE"
    assert grr.short_process("") == ""


def test_walltime_bars_draws_one_bar_per_process_longest_first():
    grr = _load()
    roll = [
        _roll_row("MIRAGE:A:SHORT", 10.0),
        _roll_row("MIRAGE:A:LONG", 1000.0),
        _roll_row("MIRAGE:A:MID", 100.0),
    ]

    svg = grr.walltime_bars_svg(roll)

    assert svg.count("<rect class='bar'") == 3
    # Ranked: the longest process's label is emitted before the shortest.
    assert svg.index("LONG") < svg.index("MID") < svg.index("SHORT")
    # The full name survives as a tooltip even though the label is the leaf.
    assert "MIRAGE:A:LONG" in svg


def test_walltime_bars_truncates_to_top_n():
    grr = _load()
    roll = [_roll_row(f"P{i}", float(i)) for i in range(1, 40)]
    svg = grr.walltime_bars_svg(roll, top=5)
    assert svg.count("<rect class='bar'") == 5
    assert "P39" in svg and "P1<" not in svg


def test_walltime_bars_is_empty_for_no_rows():
    grr = _load()
    assert grr.walltime_bars_svg([]) == ""


def test_walltime_bars_escapes_a_hostile_process_name():
    grr = _load()
    svg = grr.walltime_bars_svg([_roll_row("<b>A</b>", 5.0)])
    assert "<b>A</b>" not in svg
    assert "&lt;b&gt;A&lt;/b&gt;" in svg


GB = 1024**3


def test_memory_headroom_draws_a_request_track_and_an_observed_overlay():
    grr = _load()
    roll = [
        _roll_row(
            "MIRAGE:A:WASTEFUL", 1.0, mem_req_max_b=200.0 * GB, peak_rss_max_b=20.0 * GB
        ),
        _roll_row(
            "MIRAGE:A:TIGHT", 1.0, mem_req_max_b=32.0 * GB, peak_rss_max_b=30.0 * GB
        ),
    ]

    svg = grr.memory_headroom_svg(roll)

    assert svg.count("<rect class='req'") == 2
    assert svg.count("<rect class='obs'") == 2
    # Ranked by wasted bytes: 180 GB wasted beats 2 GB wasted.
    assert svg.index("WASTEFUL") < svg.index("TIGHT")
    assert "10%" in svg  # 20 of 200 GB used
    assert "94%" in svg  # 30 of 32 GB used


def test_memory_headroom_skips_a_process_with_no_recorded_request():
    """A process whose trace rows carry `memory` = '-' has no headroom to show.
    Drawing it against a request of zero would render as 'infinitely
    over-provisioned', which is the opposite of the truth."""
    grr = _load()
    roll = [
        _roll_row("A", 1.0, mem_req_max_b=None, peak_rss_max_b=10.0 * GB),
        _roll_row("B", 1.0, mem_req_max_b=100.0 * GB, peak_rss_max_b=None),
        _roll_row("C", 1.0, mem_req_max_b=100.0 * GB, peak_rss_max_b=50.0 * GB),
    ]
    svg = grr.memory_headroom_svg(roll)
    assert svg.count("<rect class='req'") == 1
    assert "C" in svg


def test_memory_headroom_marks_a_process_that_exceeded_its_request():
    """peak_rss above the request is the OOM-retry precursor and must be visible
    as such, not silently clipped to a full bar like a perfectly-sized task."""
    grr = _load()
    svg = grr.memory_headroom_svg(
        [_roll_row("A", 1.0, mem_req_max_b=10.0 * GB, peak_rss_max_b=15.0 * GB)]
    )
    assert "class='obs over'" in svg
    assert "150%" in svg


def test_memory_headroom_is_empty_when_nothing_is_plottable():
    grr = _load()
    assert grr.memory_headroom_svg([]) == ""
    assert grr.memory_headroom_svg([_roll_row("A", 1.0)]) == ""


def test_failure_cost_draws_one_bar_per_failing_process():
    grr = _load()
    roll = [
        _roll_row("MIRAGE:A:CLEAN", 100.0),
        _roll_row(
            "MIRAGE:A:FLAKY",
            100.0,
            n_failed=3,
            failed_gb_h=450.0,
            failed_realtime_s=5400.0,
        ),
        _roll_row(
            "MIRAGE:A:WORST",
            100.0,
            n_failed=1,
            failed_gb_h=900.0,
            failed_realtime_s=3600.0,
        ),
    ]

    svg = grr.failure_cost_svg(roll)

    assert svg.count("<rect class='fail'") == 2  # CLEAN is not drawn
    assert "CLEAN" not in svg
    # Ranked by cost, not by count: one 900 GB.h failure outranks three 450s.
    assert svg.index("WORST") < svg.index("FLAKY")
    assert "3 failed" in svg
    assert "1 failed" in svg


def test_failure_cost_reports_the_reserved_memory_hours_not_just_the_count():
    """A count alone cannot distinguish three cheap failures from one that held
    450 GB for six hours -- and the second is the whole reason this panel is
    here."""
    grr = _load()
    svg = grr.failure_cost_svg(
        [_roll_row("A", 1.0, n_failed=1, failed_gb_h=2700.0, failed_realtime_s=21600.0)]
    )
    assert "2700" in svg
    assert "GB" in svg and "h" in svg


def test_failure_cost_is_empty_when_nothing_failed():
    grr = _load()
    assert grr.failure_cost_svg([_roll_row("A", 1.0)]) == ""
    assert grr.failure_cost_svg([]) == ""


def _joined(proc, tag, size_b, rt_s):
    return {
        "process": proc,
        "tag": tag,
        "input_bytes": size_b,
        "realtime_s": rt_s,
        "peak_rss_b": None,
        "exit": "0",
        "status": "COMPLETED",
    }


def test_size_vs_runtime_plots_one_point_per_joinable_task():
    grr = _load()
    joined = [
        _joined("MIRAGE:A:CONVERT", "P001", 1 * GB, 60.0),
        _joined("MIRAGE:A:CONVERT", "P002", 8 * GB, 400.0),
        _joined("MIRAGE:A:REGISTER", "P001", 4 * GB, 3600.0),
    ]

    svg = grr.size_vs_runtime_svg(joined)

    assert svg.count("<circle") == 3
    # One legend entry per distinct process, by leaf name.
    assert svg.count("<rect class='key'") == 2
    assert "CONVERT" in svg and "REGISTER" in svg


def test_size_vs_runtime_drops_a_zero_byte_input_from_the_log_axis():
    """A log axis cannot place zero: the scatter drops the point and says so,
    rather than parking it at an arbitrary x."""
    grr = _load()
    joined = [
        _joined("A", "P001", 0, 10.0),
        _joined("A", "P002", 2 * GB, 10.0),
    ]

    svg = grr.size_vs_runtime_svg(joined)

    assert svg.count("<circle") == 1
    assert "1 task not shown" in svg


def test_size_vs_runtime_ignores_unjoined_rows():
    grr = _load()
    joined = [
        {
            "process": "A",
            "tag": "P1",
            "input_bytes": None,
            "realtime_s": 10.0,
            "peak_rss_b": None,
            "exit": "0",
            "status": "COMPLETED",
        },
    ]
    assert grr.size_vs_runtime_svg(joined) == ""


def test_size_vs_runtime_escapes_a_hostile_process_name():
    grr = _load()
    svg = grr.size_vs_runtime_svg([_joined("<b>A</b>", "P1", 2 * GB, 10.0)])
    assert "<b>A</b>" not in svg
    assert "&lt;b&gt;A&lt;/b&gt;" in svg


# ---------------------------------------------------------------------------
# +-inf/nan are treated like a missing value: skipped (and counted in a caption
# where the panel has one), never rendered as "nan"/"inf" and never a crash.
# ---------------------------------------------------------------------------

_NONFINITE = [float("inf"), float("-inf"), float("nan")]


def test_fmt_secs_returns_a_sentinel_for_non_finite_input():
    grr = _load()
    for bad in _NONFINITE:
        assert grr.fmt_secs(bad) == "n/a"
    assert grr.fmt_secs(None) == "N/A"  # unchanged: still the missing-value case


def test_fmt_bytes_returns_a_sentinel_for_non_finite_input():
    grr = _load()
    for bad in _NONFINITE:
        assert grr.fmt_bytes(bad) == "n/a"
    assert grr.fmt_bytes(None) == "N/A"  # unchanged: still the missing-value case


@pytest.mark.parametrize("bad", _NONFINITE)
def test_walltime_bars_drops_a_non_finite_realtime_total(bad):
    grr = _load()
    roll = [_roll_row("GOOD", 10.0), _roll_row("BAD", bad)]
    svg = grr.walltime_bars_svg(roll)
    ET.fromstring(svg)  # well-formed
    assert "BAD" not in svg
    assert svg.count("<rect class='bar'") == 1
    assert "nan" not in svg and "inf" not in svg


@pytest.mark.parametrize("bad", _NONFINITE)
def test_memory_headroom_drops_a_non_finite_request(bad):
    grr = _load()
    roll = [
        _roll_row("GOOD", 1.0, mem_req_max_b=100.0 * GB, peak_rss_max_b=50.0 * GB),
        _roll_row("BAD", 1.0, mem_req_max_b=bad, peak_rss_max_b=50.0 * GB),
    ]
    svg = grr.memory_headroom_svg(roll)
    ET.fromstring(svg)
    assert "BAD" not in svg
    assert svg.count("<rect class='req'") == 1
    assert "nan" not in svg and "inf" not in svg


@pytest.mark.parametrize("bad", _NONFINITE)
def test_memory_headroom_drops_a_non_finite_observed_peak(bad):
    grr = _load()
    roll = [
        _roll_row("GOOD", 1.0, mem_req_max_b=100.0 * GB, peak_rss_max_b=50.0 * GB),
        _roll_row("BAD", 1.0, mem_req_max_b=100.0 * GB, peak_rss_max_b=bad),
    ]
    svg = grr.memory_headroom_svg(roll)
    ET.fromstring(svg)
    assert "BAD" not in svg
    assert svg.count("<rect class='req'") == 1
    assert "nan" not in svg and "inf" not in svg


@pytest.mark.parametrize("bad", _NONFINITE)
def test_failure_cost_drops_a_non_finite_failed_gb_h(bad):
    grr = _load()
    roll = [
        _roll_row("GOOD", 1.0, n_failed=1, failed_gb_h=100.0, failed_realtime_s=60.0),
        _roll_row("BAD", 1.0, n_failed=1, failed_gb_h=bad, failed_realtime_s=60.0),
    ]
    svg = grr.failure_cost_svg(roll)
    ET.fromstring(svg)
    assert "BAD" not in svg
    assert svg.count("<rect class='fail'") == 1
    assert "nan" not in svg and "inf" not in svg


@pytest.mark.parametrize("bad", _NONFINITE)
def test_failure_cost_drops_a_non_finite_failed_realtime(bad):
    grr = _load()
    roll = [
        _roll_row("GOOD", 1.0, n_failed=1, failed_gb_h=100.0, failed_realtime_s=60.0),
        _roll_row("BAD", 1.0, n_failed=1, failed_gb_h=100.0, failed_realtime_s=bad),
    ]
    svg = grr.failure_cost_svg(roll)
    ET.fromstring(svg)
    assert "BAD" not in svg
    assert svg.count("<rect class='fail'") == 1
    assert "nan" not in svg and "inf" not in svg


@pytest.mark.parametrize("bad", _NONFINITE)
def test_size_vs_runtime_drops_and_counts_a_non_finite_input_bytes(bad):
    grr = _load()
    joined = [_joined("A", "P1", bad, 10.0), _joined("A", "P2", 2 * GB, 10.0)]
    svg = grr.size_vs_runtime_svg(joined)
    ET.fromstring(svg)
    assert svg.count("<circle") == 1
    assert "1 task not shown" in svg
    assert "nan" not in svg and "inf" not in svg


@pytest.mark.parametrize("bad", _NONFINITE)
def test_size_vs_runtime_drops_and_counts_a_non_finite_realtime(bad):
    grr = _load()
    joined = [_joined("A", "P1", 2 * GB, bad), _joined("A", "P2", 2 * GB, 10.0)]
    svg = grr.size_vs_runtime_svg(joined)
    ET.fromstring(svg)
    assert svg.count("<circle") == 1
    assert "1 task not shown" in svg
    assert "nan" not in svg and "inf" not in svg


def _trace_row(proc, tag, rt, rss, mem, exit_code="0"):
    return {
        "process": proc,
        "tag": tag,
        "status": "COMPLETED",
        "exit": exit_code,
        "realtime_s": rt,
        "duration_s": rt,
        "cpu_pct": 100.0,
        "peak_rss_b": rss,
        "peak_vmem_b": None,
        "rchar_b": None,
        "wchar_b": None,
        "memory_b": mem,
        "cpus": 8,
    }


def test_the_report_is_plots_not_tables():
    """One table survives (Run Totals); every other view is a panel. A count,
    not a substring, so reintroducing any table fails here."""
    grr = _load()
    trace_rows = [
        _trace_row("MIRAGE:A:CONVERT", "P001", 600.0, 3 * GB, 8 * GB),
        _trace_row("MIRAGE:A:REGISTER", "P001", 3600.0, 90 * GB, 100 * GB, "137"),
    ]
    size_map = {("MIRAGE:A:CONVERT", "P001"): 2 * GB}

    html_out = grr.build_html(trace_rows, size_map, "ts")

    assert html_out.count("<table>") == 1, "only Run Totals may remain a table"
    assert html_out.count("<svg ") == 4, "all four panels must render"
    assert "Per-Process Resource Rollup" not in html_out
    assert "Top 10 by" not in html_out


def test_run_totals_survives_with_every_number_it_had():
    grr = _load()
    html_out = grr.build_html([_trace_row("A", "P1", 600.0, 3 * GB, 8 * GB)], {}, "ts")
    for needle in (
        "Total tasks",
        "Total CPU wall-time",
        "Failed/non-zero exit",
        "Peak single-task RSS",
    ):
        assert needle in html_out


MB = 1024**2


# ---------------------------------------------------------------------------
# Legend ranking and capping, drop-count notes, non-finite %CPU, wall-time share.
# ---------------------------------------------------------------------------


def test_size_vs_runtime_legend_ranked_by_point_count_not_first_appearance():
    """The legend is ranked by point count, not first appearance, and anything
    past `top_processes` goes into one explicit grey 'other' row."""
    grr = _load()
    joined = []
    for i in range(8):
        joined.append(_joined(f"SMALL_{i}", f"S{i}", (i + 1) * MB, 10.0))
    for i in range(1000):
        joined.append(_joined("BIG", f"B{i}", (i % 50 + 1) * MB, 5.0 + i))

    svg = grr.size_vs_runtime_svg(joined)
    ET.fromstring(svg)
    # BIG has the most points (1000) and must be legended first even though it
    # appears LAST in the input list.
    assert svg.index("BIG") < svg.index("SMALL_0")
    # 9 distinct processes, default top_processes=8 -> one SMALL is bucketed
    # as "other" rather than silently unlegended grey.
    assert svg.count("<rect class='key'") == 9
    assert "other (1 process)" in svg


def test_size_vs_runtime_caps_points_per_series():
    """A single dominant process must not draw one <circle> per task -- each
    series is deterministically stride-subsampled to at most 2,000 points, and
    the panel states how many of how many are shown."""
    grr = _load()
    joined = [
        _joined("BIG", f"T{i}", (i % 1000 + 1) * 1024, 1.0 + i) for i in range(5000)
    ]
    svg = grr.size_vs_runtime_svg(joined)
    ET.fromstring(svg)
    # ceil(5000 / ceil(5000/2000)) = ceil(5000/3) = 1667, deterministic stride.
    assert svg.count("<circle") == 1667
    assert "1667 of 5000" in svg


def test_size_vs_runtime_reports_drop_count_when_nothing_is_plottable():
    """When EVERY point is dropped but at least one was matched, the panel
    must say so rather than return the empty string, which the caller would
    then paper over with the false "No size logs matched trace tasks."."""
    grr = _load()
    joined = [_joined("A", "P1", 2 * GB, 0.0)]  # realtime_s == 0.0: not positive
    svg = grr.size_vs_runtime_svg(joined)
    assert svg != ""
    assert "could not be plotted" in svg


def test_build_html_size_panel_is_not_falsely_empty_when_everything_dropped():
    grr = _load()
    trace_rows = [
        {
            "process": "A",
            "tag": "P1",
            "exit": "0",
            "status": "COMPLETED",
            "realtime_s": 0.0,
            "peak_rss_b": None,
            "peak_vmem_b": None,
            "rchar_b": None,
            "wchar_b": None,
            "cpu_pct": None,
            "duration_s": None,
            "memory_b": None,
            "cpus": None,
        }
    ]
    html_out = grr.build_html(trace_rows, {("A", "P1"): 2 * GB}, "ts")
    assert "No size logs matched trace tasks." not in html_out
    assert "could not be plotted" in html_out


def test_parse_percent_rejects_non_finite_strings():
    """A corrupted trace field like 'nan%' parses cleanly with a bare
    float(), silently introducing a non-finite value that later renders as
    the literal text 'nan%'/'inf%'. Reject it at the source."""
    grr = _load()
    assert grr.parse_percent("nan%") is None
    assert grr.parse_percent("inf%") is None
    assert grr.parse_percent("-inf%") is None


def test_run_totals_peak_cpu_guards_against_non_finite():
    grr = _load()
    trace_rows = [
        {
            "process": "A",
            "tag": "P1",
            "status": "COMPLETED",
            "exit": "0",
            "realtime_s": 1.0,
            "peak_rss_b": None,
            "peak_vmem_b": None,
            "rchar_b": None,
            "wchar_b": None,
            "cpu_pct": float("nan"),
            "duration_s": None,
            "memory_b": None,
            "cpus": None,
        },
    ]
    html_out = grr.build_html(trace_rows, {}, "ts")
    marker = "<h2>Run Totals</h2>"
    section = html_out[html_out.index(marker) :]
    section = section[: section.index("</section>")]
    assert "nan%" not in section
    assert "n/a" in section


def test_walltime_bars_percentage_is_share_of_total_not_longest():
    """Each bar's percentage is its share of the run's total wall-time, not of
    the longest bar (which would always read 100%)."""
    grr = _load()
    roll = [_roll_row("A", 300.0), _roll_row("B", 100.0)]
    svg = grr.walltime_bars_svg(roll)
    assert "(75%)" in svg
    assert "(25%)" in svg
    assert "(100%)" not in svg


def test_a_panel_with_no_data_degrades_to_a_sentence_not_a_crash():
    """A run whose trace has no `memory` column at all (an older trace, or a
    fields list edited down) must still produce a report."""
    grr = _load()
    rows = [
        {
            "process": "A",
            "tag": "P1",
            "status": "COMPLETED",
            "exit": "0",
            "realtime_s": 10.0,
            "duration_s": 10.0,
            "cpu_pct": None,
            "peak_rss_b": None,
            "peak_vmem_b": None,
            "rchar_b": None,
            "wchar_b": None,
            "memory_b": None,
            "cpus": None,
        }
    ]
    html_out = grr.build_html(rows, {}, "ts")
    assert "<html" in html_out and "</html>" in html_out
    assert "class='empty'" in html_out
