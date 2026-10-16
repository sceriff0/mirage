"""``benchmarks.regbench.trace``: Nextflow's ``trace.txt`` read with the standard library.

Ported from ``benchmarks/analysis/lib/load.py::parse_trace`` and ``parsing.py``
(``parse_to_gb``, ``parse_duration``), which go with the analysis layer. The parser
tests of ``benchmarks/tests/test_load.py::test_parse_trace_converts_units`` are here,
on the same two rows, written inline.
"""

from benchmarks.regbench import trace

HEADER = "task_id\tprocess\ttag\tname\tstatus\texit\tsubmit\tstart\tcomplete\tduration\trealtime\t%cpu\tcpus\tmemory\tpeak_rss\tpeak_vmem\trchar\twchar"
ROW = "1\tMIRAGE:REGISTRATION:REGISTER_PATIENT:VALIS_ADAPTER:REGISTER\tP1\tREGISTER (P1)\tCOMPLETED\t0\t2026-10-01 10:00:00.000\t2026-10-01 10:01:00.000\t2026-10-02 12:01:00.000\t1d 2h 1m\t1d 2h\t350.0%\t4\t64 GB\t30.5 GB\t40 GB\t1.5 GB\t512 MB"


def _write(tmp_path, header, row):
    p = tmp_path / "trace.txt"
    p.write_text(header + "\n" + row + "\n")
    return p


def test_units_and_days(tmp_path):
    (r,) = trace.parse(_write(tmp_path, HEADER, ROW))
    assert r["realtime_s"] == 26 * 3600 and r["duration_s"] == 26 * 3600 + 60
    assert (
        r["peak_rss_gb"] == 30.5
        and r["write_gb"] == 0.5
        and r["pcpu"] == 350.0
        and r["cpus"] == 4
    )
    assert r["memory_gb"] == 64.0 and r["peak_vmem_gb"] == 40.0 and r["read_gb"] == 1.5
    assert r["process"] == "MIRAGE:REGISTRATION:REGISTER_PATIENT:VALIS_ADAPTER:REGISTER"
    assert r["task_id"] == 1 and r["exit"] == 0 and r["status"] == "COMPLETED"
    assert r["submit"] == "2026-10-01 10:00:00.000"
    assert list(r) == list(trace.FIELDS)


def test_columns_are_read_by_name_not_position(tmp_path):
    h, row = HEADER.split("\t"), ROW.split("\t")
    order = list(reversed(range(len(h))))
    a = trace.parse(_write(tmp_path, HEADER, ROW))
    (tmp_path / "b").mkdir()
    b = trace.parse(
        _write(
            tmp_path / "b",
            "\t".join(h[i] for i in order),
            "\t".join(row[i] for i in order),
        )
    )
    assert a == b


def test_a_dash_is_not_a_number(tmp_path):
    (r,) = trace.parse(_write(tmp_path, HEADER, ROW.replace("30.5 GB", "-")))
    assert r["peak_rss_gb"] != r["peak_rss_gb"]
    (r,) = trace.parse(_write(tmp_path, HEADER, ROW.replace("\t0\t2026", "\t-\t2026")))
    assert r["exit"] != r["exit"]


def test_the_ported_trace_converts_units(tmp_path):
    """test_load.py::test_parse_trace_converts_units, on its fixture's rows."""
    text = (
        HEADER + "\n"
        "1\tCONVERT_IMAGE\tpx4096_ch2\tCONVERT_IMAGE (px4096_ch2)\tCOMPLETED\t0\t-\t-\t-\t2m 0s\t1m 50s\t180%\t4\t100 GB\t8 GB\t9 GB\t1 GB\t1 GB\n"
        "2\tSEGMENT\tpx4096_ch2\tSEGMENT (px4096_ch2)\tCOMPLETED\t0\t-\t-\t-\t5m 0s\t4m 30s\t200%\t2\t64 GB\t20 GB\t22 GB\t2 GB\t2 GB\n"
    )
    p = tmp_path / "trace.txt"
    p.write_text(text)
    seg = [r for r in trace.parse(p) if r["process"] == "SEGMENT"][0]
    assert seg["peak_rss_gb"] == 20.0
    assert seg["realtime_s"] == 270.0  # 4m 30s
    assert seg["cpus"] == 2
    assert seg["submit"] == ""  # a '-' timestamp is no time


def test_the_unit_conversions():
    nan = trace.to_gb("-")
    assert nan != nan and trace.to_gb("") != trace.to_gb("")
    assert trace.to_gb("0") == 0.0
    assert trace.to_gb("1073741824") == 1.0  # a bare number is bytes
    assert trace.to_gb("1024 KB") == 1 / 1024
    assert trace.to_gb("2 TB") == 2048.0
    assert trace.to_seconds("100ms") == 0.1
    assert trace.to_seconds("1.5s") == 1.5
    assert trace.to_seconds("2d 3h") == 2 * 86400 + 3 * 3600
    assert abs(trace.to_seconds("1m 5s 20ms") - 65.02) < 1e-9
    bad = trace.to_seconds("soon")
    assert bad != bad


def test_a_trace_without_the_optional_fields_reads_them_as_nan(tmp_path):
    p = tmp_path / "trace.txt"
    p.write_text("task_id\tprocess\tstatus\trealtime\n7\tX\tCOMPLETED\t1s\n")
    (r,) = trace.parse(p)
    assert r["realtime_s"] == 1.0 and r["pcpu"] != r["pcpu"]
    assert r["tag"] == "" and r["read_gb"] != r["read_gb"]
