"""The end-to-end cost view: every step in its own column, imported numbers labelled.

The pipeline's own steps (convert, merge and pyramid) are never run here: they are read
from a Nextflow trace of an earlier run. These tests use a small trace written by hand.
"""

from __future__ import annotations

import csv
import json
import math
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

from benchmarks.regbench import cli, cost, imageio, methods, score, variants
from benchmarks.regbench.cases import result_dir
from benchmarks.regbench.mapping import MOV_TO_REF, PointMap

from .test_pipeline import toy_case

README = Path(cost.__file__).with_name("README.md")
HEADER = (
    "task_id process tag name status exit submit start complete duration realtime "
    "%cpu cpus memory peak_rss peak_vmem rchar wchar"
).split()
TASKS = [
    # two slides of P1 converted, one merge; P2 converted only; a failed first try
    dict(process="MIRAGE:PREPROCESS:CONVERT_IMAGE", tag="P1", status="COMPLETED",
         realtime="1m 40s", cpus="8", memory="32 GB", peak_rss="6 GB"),
    dict(process="MIRAGE:PREPROCESS:CONVERT_IMAGE", tag="P1", status="COMPLETED",
         realtime="20s", cpus="8", memory="32 GB", peak_rss="9 GB"),
    dict(process="MIRAGE:POSTPROCESS:ASSEMBLE_EXPORT:MERGE_AND_PYRAMID", tag="P1",
         status="FAILED", realtime="5h", cpus="16", memory="100 GB", peak_rss="99 GB"),
    dict(process="MIRAGE:POSTPROCESS:ASSEMBLE_EXPORT:MERGE_AND_PYRAMID", tag="P1",
         status="COMPLETED", realtime="1h 0m 0s", cpus="16", memory="200 GB",
         peak_rss="150 GB"),
    dict(process="MIRAGE:PREPROCESS:CONVERT_IMAGE", tag="P2", status="CACHED",
         realtime="30s", cpus="8", memory="32 GB", peak_rss="512 MB"),
    dict(process="MIRAGE:SEGMENT", tag="P1", status="COMPLETED", realtime="9h",
         cpus="2", memory="64 GB", peak_rss="20 GB"),
]  # fmt: skip


def write_trace(path, columns=HEADER, tasks=TASKS):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(columns)
        for i, t in enumerate(tasks, 1):
            row = {c: "-" for c in columns}
            row.update({k: v for k, v in t.items() if k in columns}, task_id=str(i))
            w.writerow([row[c] for c in columns])
    return path


def res(method, group, adapter, **more):
    """One resource row as the scorer builds it, for a run that wrote an image."""
    base = dict(
        dataset="multiplex", group=group, method=method, case_id=f"{group}_r1",
        ok=True, cause="", unit="pair", adapter=adapter, register_s=100.0,
        write_s=10.0, method_convert_s=math.nan, peak_rss_gb=3.0, cgroup_peak_gb=5.0,
        device="cpu", gpu_model="", written="4 channel(s); no pyramid",
        written_file="/x/registered.ome.tif", written_complete=False,
        cpus_requested=4, cpus_asked=4, cpus_affinity=4, cpus_node=64,
        workers_seen=4, write_writer_s=math.nan, write_convert_s=math.nan,
        write_benchmark_s=math.nan,
    )  # fmt: skip
    return {**base, **more}


# ── the trace ─────────────────────────────────────────────────────────────────
def test_the_trace_is_read_by_header_name_not_by_position(tmp_path):
    """CLAUDE.md's gotcha: the column order of trace.txt is not fixed."""
    straight = cost.read_trace(write_trace(tmp_path / "a.txt"))
    shuffled = cost.read_trace(write_trace(tmp_path / "b.txt", HEADER[::-1]))
    assert straight == shuffled
    first = straight[0]
    assert first["process"] == "CONVERT_IMAGE" and first["tag"] == "P1"
    assert first["realtime_s"] == 100.0 and first["peak_rss_gb"] == 6.0
    assert first["cpus"] == 8 and first["memory_gb"] == 32.0


def test_a_trace_without_the_columns_it_needs_is_refused(tmp_path):
    cut = [c for c in HEADER if c != "peak_rss"]
    with pytest.raises(SystemExit, match="no column peak_rss"):
        cost.read_trace(write_trace(tmp_path / "t.txt", cut))
    with pytest.raises(SystemExit, match="no such file"):
        cost.read_trace(tmp_path / "missing.txt")


def test_durations_and_sizes_in_every_form_nextflow_writes():
    assert cost.parse_seconds("1h 2m 3s") == 3723.0
    assert cost.parse_seconds("250ms") == 0.25
    assert cost.parse_seconds("1.5s") == 1.5
    assert cost.parse_seconds("1d 1h") == 90000.0
    assert cost.parse_seconds("90000") == 90.0  # the raw format is milliseconds
    assert math.isnan(cost.parse_seconds("-"))
    assert cost.parse_gb("512 MB") == 0.5 and cost.parse_gb("1073741824") == 1.0
    assert math.isnan(cost.parse_gb("-"))


def test_a_patients_step_is_the_sum_of_its_completed_tasks(tmp_path):
    steps = cost.pipeline_steps(cost.read_trace(write_trace(tmp_path / "t.txt")))
    assert set(steps) == {"P1", "P2"}
    conv, merge = steps["P1"]["convert"], steps["P1"]["merge"]
    assert conv["s"] == 120.0 and conv["tasks"] == 2 and conv["peak_gb"] == 9.0
    # the failed 5 h attempt is not a cost of the step that completed
    assert merge["s"] == 3600.0 and merge["tasks"] == 1 and merge["memory_gb"] == 200.0
    assert steps["P2"]["merge"] is None and steps["P2"]["convert"]["cached"] == 1


# ── the table ─────────────────────────────────────────────────────────────────
@pytest.fixture
def steps(tmp_path):
    return cost.pipeline_steps(cost.read_trace(write_trace(tmp_path / "t.txt")))


def _row(rows, method, group="P1"):
    (r,) = [r for r in rows if r["method"] == method and r["group"] == group]
    return r


def test_every_step_has_its_own_column_and_imported_ones_name_their_source(steps):
    rows = cost.end_to_end([res("stare", "P1", "stare")], steps, "the compute arm")
    r = _row(rows, "stare")
    assert r["method_convert_s"] == cost.NONE  # STARE reads the files as they are
    assert (r["estimate_s"], r["write_s"]) == (100.0, 10.0)
    assert (r["pipeline_convert_s"], r["pipeline_merge_s"]) == (120.0, 3600.0)
    assert r["measured_here_s"] == 110.0 and r["total_s"] == 110.0 + 120.0 + 3600.0
    assert r["pipeline_source"] == "the compute arm"
    # those tasks did not run at this benchmark's 4 CPUs, and the row says so
    assert r["pipeline_convert_alloc"].startswith("8 CPUs, 32 GB; 2 task(s)")
    assert r["pipeline_merge_alloc"].startswith("16 CPUs, 200 GB; 1 task(s)")
    assert r["imported"] == "pipeline_convert, pipeline_merge"
    assert r["measured_here"] == "estimate, write"
    assert r["pipeline_merge_peak_gb"] == 150.0


def test_no_merge_is_added_where_the_method_already_wrote_a_complete_pyramid(steps):
    done = res("palom", "P1", "palom", written_complete=True, written="pyramid")
    r = _row(cost.end_to_end([done], steps, "arm"), "palom")
    assert r["pipeline_merge_s"] == cost.NOT_NEEDED
    assert r["total_s"] == 110.0 + 120.0  # the hour of merging is not added
    assert r["pipeline_merge_alloc"] == "" and r["imported"] == "pipeline_convert"
    assert math.isnan(r["pipeline_merge_peak_gb"])


def test_a_patient_the_trace_does_not_have_gets_no_total(steps):
    rows = cost.end_to_end(
        [res("stare", "P1", "stare"), res("stare", "P9", "stare")], steps, "arm"
    )
    absent = _row(rows, "stare", "P9")
    assert absent["pipeline_convert_s"] == absent["pipeline_merge_s"] == "not available"
    assert math.isnan(absent["total_s"]) and absent["measured_here_s"] == 110.0
    assert absent["pipeline_source"] == "patient P9 is not in the trace"
    # P2 was converted but never merged: no total either, and no guessed merge
    half = _row(
        cost.end_to_end([res("stare", "P2", "stare")], steps, "arm"), "stare", "P2"
    )
    assert (
        half["pipeline_convert_s"] == 30.0
        and half["pipeline_merge_s"] == "not available"
    )
    assert math.isnan(half["total_s"])
    assert "1 cached" in half["pipeline_convert_alloc"]


def test_without_a_trace_the_pipeline_columns_say_not_available():
    r = _row(cost.end_to_end([res("stare", "P1", "stare")]), "stare")
    assert r["pipeline_convert_s"] == r["pipeline_merge_s"] == "not available"
    assert math.isnan(r["total_s"]) and r["measured_here_s"] == 110.0
    assert r["pipeline_source"] == "no pipeline trace was given"
    assert r["imported"] == cost.NONE
    text = "\n".join(cost.markdown([r], cost.summarise([r]), "", score.markdown))
    assert "No pipeline trace was given" in text and "not available" in text


def test_a_step_the_package_does_inside_its_call_is_not_separable_not_guessed(steps):
    for adapter in ("valis", "valis_mirage"):
        r = _row(cost.end_to_end([res("v", "P1", adapter)], steps, "arm"), "v")
        assert r["method_convert_s"] == cost.NOT_SEPARABLE
        assert "conversion" in r["estimate_includes"]
        assert r["measured_here_s"] == 110.0  # nothing is added for it


def test_a_methods_own_conversion_is_a_column_of_its_own(steps):
    # DeeperHistReg converts per round; ASHLAR cuts the patient's tiles once
    dhr = [
        res("d", "P1", "deeperhistreg", case_id="a", method_convert_s=7.0),
        res("d", "P1", "deeperhistreg", case_id="b", method_convert_s=5.0),
    ]
    r = _row(cost.end_to_end(dhr, steps, "arm"), "d")
    assert r["method_convert_s"] == 12.0 and r["measured_here_s"] == 12.0 + 200 + 20
    assert r["measured_here"] == "method_convert, estimate, write"
    ash = [
        res("a", "P1", "ashlar", case_id=c, unit="patient", method_convert_s=40.0)
        for c in ("a", "b")
    ]
    r = _row(cost.end_to_end(ash, steps, "arm"), "a")
    # one launch per patient: its tile cutting and its clock are counted once
    assert r["method_convert_s"] == 40.0 and r["estimate_s"] == 100.0
    assert r["write_s"] == 20.0


def test_the_benchmarks_own_copying_is_shown_and_is_in_no_total(steps):
    """Where an adapter times its write step in parts, the writer column is the package's
    writer alone; a conversion that writer needed is conversion; the benchmark's own
    copy (ASHLAR's split, DeeperHistReg's stacking) is beside them and never summed."""
    row = res(
        "d", "P1", "deeperhistreg", method_convert_s=7.0, write_s=60.0,
        write_writer_s=40.0, write_convert_s=5.0, write_benchmark_s=15.0,
    )  # fmt: skip
    r = _row(cost.end_to_end([row], steps, "arm"), "d")
    assert (r["method_convert_s"], r["write_s"], r["benchmark_copy_s"]) == (
        12.0,
        40.0,
        15.0,
    )
    assert r["measured_here_s"] == 12.0 + 100.0 + 40.0
    assert r["total_s"] == r["measured_here_s"] + 120.0 + 3600.0
    plain = _row(cost.end_to_end([res("stare", "P1", "stare")], steps, "arm"), "stare")
    assert plain["write_s"] == 10.0 and math.isnan(plain["benchmark_copy_s"])


def test_a_merge_is_never_added_on_a_guess(steps):
    """A run that recorded no description of its file: the merge step is not added and
    there is no total, rather than a total that assumes one way or the other."""
    blind = res("stare", "P1", "stare", written="", written_complete="")
    r = _row(cost.end_to_end([blind], steps, "arm"), "stare")
    assert r["pipeline_merge_s"] == "not available" and math.isnan(r["total_s"])
    assert r["written"] == "not available"


def test_cpu_and_gpu_rows_share_one_table_with_the_device_in_its_own_column(steps):
    rows = cost.end_to_end(
        [
            res("deeperhistreg_hr_gpu_asdrawn", "P1", "deeperhistreg",
                device="cuda:0", gpu_model="NVIDIA H200", method_convert_s=1.0),
            res("deeperhistreg_hr_cpu_asdrawn", "P1", "deeperhistreg",
                method_convert_s=1.0),
        ],
        steps,
        "arm",
    )  # fmt: skip
    assert [(r["method"], r["device"], r["gpu_model"]) for r in rows] == [
        ("deeperhistreg_hr_gpu_asdrawn", "cuda:0", "NVIDIA H200"),
        ("deeperhistreg_hr_cpu_asdrawn", "cpu", ""),
    ]
    summary = cost.summarise(rows)
    assert {"device", "gpu_model"} <= set(cost.SUMMARY_COLUMNS)
    assert [s["device"] for s in summary] == ["cuda:0", "cpu"]


def test_a_result_that_wrote_no_image_has_no_row(steps):
    rigid = res("stare_rigid", "P1", "stare", written_file="", written="")
    failed = res("stare", "P1", "stare", ok=False)
    assert cost.end_to_end([rigid, failed], steps, "arm") == []


def test_the_summary_block_says_what_was_not_measured_here(steps):
    rows = cost.end_to_end([res("stare", "P1", "stare")], steps, "the compute arm")
    text = "\n".join(
        cost.markdown(rows, cost.summarise(rows), "the compute arm", score.markdown)
    )
    assert "NOT measured here" in text and "the compute arm" in text
    assert "8 CPUs, 32 GB" in text and "16 CPUs, 200 GB" in text
    assert "(secondary)" in text


# ── cores ─────────────────────────────────────────────────────────────────────
def test_the_cost_header_says_whether_the_cores_were_confined():
    held = [res("a", "P1", "stare")]
    assert cost.cores_confined(held).startswith("cores confined: yes (1 of 1")
    loose = [res("a", "P1", "stare", cpus_affinity=64)]
    assert cost.cores_confined(loose).startswith("cores confined: no (0 of 1")
    assert cost.cores_confined(held + loose).startswith("cores confined: partly")
    laptop = [res("a", "P1", "stare", cpus_asked=math.nan)]
    assert cost.cores_confined(laptop).startswith("cores confined: unknown")
    # a row for a run that never started says nothing about the cluster
    ghost = [res("a", "P1", "stare", cause="not_run", cpus_affinity=64)]
    assert cost.cores_confined(held + ghost).startswith("cores confined: yes (1 of 1")
    assert "Workers each method's own pool had: stare 4." in cost.cores_confined(held)


# ── what was written ──────────────────────────────────────────────────────────
def test_a_written_file_is_described_from_the_file_itself(tmp_path):
    plane = np.zeros((3, 256, 256), np.uint16)
    flat = tmp_path / "flat.ome.tif"
    tifffile.imwrite(flat, plane, ome=True, bigtiff=True, tile=(64, 64),
                     metadata={"axes": "CYX"})  # fmt: skip
    got = imageio.describe_written(flat)
    assert (got["channels"], got["levels"], got["pyramid"]) == (3, 1, False)
    assert got["compression"] == "NONE" and got["bytes"] == flat.stat().st_size
    assert cost.complete_image(got) is False
    assert cost.written_text(got).startswith(
        "3 channel(s); no pyramid; compression none"
    )

    pyr = tmp_path / "pyr.ome.tif"
    with tifffile.TiffWriter(pyr, ome=True, bigtiff=True) as tw:
        tw.write(plane, subifds=1, tile=(64, 64), compression="zlib",
                 metadata={"axes": "CYX"})  # fmt: skip
        tw.write(plane[:, ::2, ::2], subfiletype=1, tile=(64, 64), compression="zlib")
    got = imageio.describe_written(pyr)
    assert (got["channels"], got["levels"], got["pyramid"]) == (3, 2, True)
    assert got["compression"] == "ADOBE_DEFLATE"
    # Ruling R3.14: no merge only for ONE pyramidal file per PATIENT with every round's
    # channels. The file says how many channels it holds; the driver records how many
    # the patient's slides hold together.
    assert cost.complete_image({**got, "patient_channels": 3}) is True
    assert cost.complete_image({**got, "patient_channels": 2}) is True
    # a pyramidal file per ROUND, every channel of that round: the merge is still needed
    assert cost.complete_image({**got, "patient_channels": 6}) is False
    # how many the patient has is not recorded: never decided on a guess
    assert cost.complete_image(got) is None
    assert cost.complete_image({**got, "patient_channels": None}) is None
    # without a pyramid the merge is needed whatever it holds
    flat_got = {**got, "pyramid": False, "levels": 1}
    assert cost.complete_image({**flat_got, "patient_channels": 3}) is False
    got = {**got, "patient_channels": 3}
    # one file per channel is not one image holding every channel
    assert cost.complete_image({**got, "files": 3, "channels": 1}) is False
    assert "3 files" in cost.written_text({**got, "files": 3, "channels": 1})
    assert (
        cost.complete_image(None) is None
        and cost.complete_image({"unread": "x"}) is None
    )
    assert "unread" in imageio.describe_written(tmp_path / "missing.tif")


# ── through the driver and the scorer ─────────────────────────────────────────
def _writer(name, pyramid, channels=2):
    """A method that writes an image of its own at the yield, as VALIS does."""

    def register(case, work, opts):
        dest = Path(opts["native_dir"](case.case_id, name))
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / "registered.ome.tif"
        plane = np.zeros((channels, 440, 440), np.uint16)
        with tifffile.TiffWriter(out, ome=True) as tw:
            tw.write(plane, subifds=1 if pyramid else 0, tile=(32, 32),
                     metadata={"axes": "CYX"})  # fmt: skip
            if pyramid:
                tw.write(plane[:, ::2, ::2], subfiletype=1, tile=(32, 32))
        fn = PointMap(lambda xy: np.asarray(xy, dtype=float), MOV_TO_REF)
        yield name, fn, {"native": {"files": [str(out)]}, "write_s": 2.0}

    return types.SimpleNamespace(
        NAME=name, SCOPE="pair", VARIANTS=(name,), register=register
    )


def test_score_writes_the_table_and_the_summary_block(tmp_path, capsys):
    root, out = tmp_path / "cases", tmp_path / "out"
    for group in ("P1", "P9"):
        case = toy_case(root, f"{group}_r1", group)
        # the patient's slides: two channels on the reference, three on the round
        case.extra.update(channels_ref=["a", "b"], channels_mov=["c", "d", "e"])
        for name, pyramid, channels in (
            ("flat", False, 5),
            ("deep", True, 5),  # one pyramid holding the whole patient (ASHLAR's)
            ("round", True, 3),  # a pyramid of this round only (VALIS's, PALOM's)
        ):
            methods.run_case(_writer(name, pyramid, channels), case, out)
    native = json.loads((result_dir(out, "deep", case) / "run.json").read_text())[
        "native"
    ]
    assert native["written"]["pyramid"] and native["written"]["channels"] == 5
    assert native["written"]["patient_channels"] == 5

    trace = write_trace(tmp_path / "trace.txt")
    got = score.score(
        root, out, ["multiplex"], with_tissue=False,
        pipeline_trace=trace, pipeline_trace_label="the compute arm",
    )  # fmt: skip
    rows = got["cost_end_to_end"]
    flat, deep = _row(rows, "flat"), _row(rows, "deep")
    assert (
        flat["pipeline_merge_s"] == 3600.0 and deep["pipeline_merge_s"] == "not needed"
    )
    # a pyramid per round is not the patient's merged image: the step is added
    per_round = _row(rows, "round")
    assert per_round["pipeline_merge_s"] == 3600.0
    assert "pyramid (2 levels)" in per_round["written"]
    assert flat["write_s"] == deep["write_s"] == 2.0
    assert "no pyramid" in flat["written"] and "pyramid (2 levels)" in deep["written"]
    assert flat["written_from"] == "the file, read after the run"
    assert math.isnan(_row(rows, "flat", "P9")["total_s"])
    assert flat["pipeline_source"].startswith("the compute arm (")
    assert str(trace) in flat["pipeline_source"]

    with open(out / "tables" / "cost_end_to_end.csv", newline="") as fh:
        on_disk = list(csv.DictReader(fh))
    assert len(on_disk) == 6 and list(on_disk[0]) == list(cost.COLUMNS)
    assert {r["pipeline_merge_s"] for r in on_disk} >= {"3600.0", "not needed"}
    assert (out / "tables" / "cost_end_to_end_summary.csv").is_file()
    text = (out / "tables" / "summary.md").read_text()
    assert "## Cost from the cohort's files to a merged image (secondary)" in text
    assert "cores confined:" in text
    # the headline cost table is still the estimate alone
    assert "## Resources (per patient; registration only)" in text

    # the same through the command line; without a trace nothing else changes
    capsys.readouterr()
    argv = [
        "score",
        "--cases",
        str(root),
        "--out",
        str(out),
        "--only-run",
        "--no-tissue",
    ]
    cli.main([*argv, "--pipeline-trace", str(trace), "--pipeline-trace-label", "arm X"])
    assert "arm X" in (out / "tables" / "summary.md").read_text()
    cli.main(argv)
    with open(out / "tables" / "cost_end_to_end.csv", newline="") as fh:
        bare = list(csv.DictReader(fh))
    assert {r["pipeline_convert_s"] for r in bare} == {"not available"}
    assert [r["estimate_s"] for r in bare] == [r["estimate_s"] for r in on_disk]


# ── the table of what each clock includes ─────────────────────────────────────
def test_the_clock_table_covers_every_method_of_the_run_list():
    assert {r.method for r in variants.RUNS} == set(cost.CLOCK)
    for name, clock in cost.CLOCK.items():
        how = clock["convert"]
        assert set(clock) >= {"reading", "convert", "estimate", "write", "writes"}, name
        if how not in (cost.NONE, cost.NOT_SEPARABLE):
            assert clock.get("per") in ("round", "patient"), name


def test_the_readme_carries_the_clock_table_as_the_code_has_it():
    assert cost.clock_markdown() in README.read_text()
