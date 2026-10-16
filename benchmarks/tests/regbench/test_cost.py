"""The cost-by-step view: every step in its own column, every number measured here.

Nothing is read from a pipeline run: the import of a Nextflow trace (and the totals that
needed it) was removed on 2026-10-12, and one test below holds that it stays removed.
"""

from __future__ import annotations

import csv
import json
import math
import os
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
SLURM = Path(cost.__file__).with_name("slurm")


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


# ── nothing is imported ───────────────────────────────────────────────────────
def test_no_pipeline_trace_is_read_and_no_column_depends_on_one(capsys):
    """The owner's ruling of 2026-10-12: no import of the pipeline's Nextflow trace, and
    no total that depends on it. Not as an option, a variable, a reader or a column."""
    for name in ("read_trace", "pipeline_steps", "parse_seconds", "parse_gb"):
        assert not hasattr(cost, name), name
    gone = ("pipeline", "total_s", "imported", "alloc")
    for column in (*cost.COLUMNS, *cost.SUMMARY_COLUMNS):
        assert not any(word in column for word in gone), column
    # what this benchmark measures itself is all still there
    for column in (
        "method_convert_s",
        "estimate_s",
        "write_s",
        "benchmark_copy_s",
        "measured_here_s",
        "estimate_peak_gb",
        "job_peak_gb",
        "written",
        "device",
        "gpu_model",
    ):
        assert column in cost.COLUMNS, column
    assert {"measured_here_s", "estimate_peak_gb_max", "job_peak_gb_max"} <= set(
        cost.SUMMARY_COLUMNS
    )
    # the command refuses the old option, the launcher has no such variable
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["score", "--cases", "c", "--out", "o", "--pipeline-trace", "t.txt"]
        )
    assert "--pipeline-trace" in capsys.readouterr().err
    import inspect

    assert "pipeline" not in " ".join(inspect.signature(score.score).parameters)
    for script in sorted(SLURM.iterdir()):
        text = script.read_text()
        assert "PIPELINE_TRACE" not in text and "pipeline-trace" not in text, script
    # the README says it once, as history, and promises nothing
    readme = README.read_text()
    assert readme.count("REGBENCH_PIPELINE_TRACE") == 1
    assert "imported the" in readme and "are removed" in readme


# ── the table ─────────────────────────────────────────────────────────────────
def _row(rows, method, group="P1"):
    (r,) = [r for r in rows if r["method"] == method and r["group"] == group]
    return r


def test_every_step_has_its_own_column_and_the_total_is_their_sum():
    r = _row(cost.end_to_end([res("stare", "P1", "stare")]), "stare")
    assert list(r) == list(cost.COLUMNS)
    assert r["method_convert_s"] == cost.NONE  # STARE reads the files as they are
    assert (r["estimate_s"], r["write_s"]) == (100.0, 10.0)
    assert r["measured_here_s"] == 110.0
    assert (r["estimate_peak_gb"], r["job_peak_gb"]) == (3.0, 5.0)
    assert r["measured_here"] == "estimate, write"


def test_the_row_says_whether_the_written_image_is_the_patients_merged_pyramid():
    """Methods keep their own writers, so the totals stop at unlike images; the table
    says which is already one pyramidal file for the patient, from the file alone."""
    done = res("ashlar", "P1", "ashlar", written_complete=True, written="pyramid")
    r = _row(cost.end_to_end([done]), "ashlar")
    assert (r["merged_image"], r["written_from"]) == (
        "yes",
        "the file, read after the run",
    )
    per_round = _row(cost.end_to_end([res("stare", "P1", "stare")]), "stare")
    assert per_round["merged_image"] == "no"
    assert per_round["measured_here_s"] == 110.0  # nothing is added for a merge
    mixed = [
        res("v", "P1", "valis", case_id="a", written_complete=True),
        res("v", "P1", "valis", case_id="b", written_complete=False),
    ]
    r = _row(cost.end_to_end(mixed), "v")
    assert (r["merged_image"], r["written_from"]) == (
        "no",
        "the files differ between rounds",
    )


def test_what_a_run_did_not_record_is_not_decided_on_a_guess():
    blind = res("stare", "P1", "stare", written="", written_complete="")
    r = _row(cost.end_to_end([blind]), "stare")
    assert r["merged_image"] == cost.NOT_RECORDED == "not recorded"
    assert r["written_from"] == "not recorded by this run"
    assert r["written"] == "not available" and r["measured_here_s"] == 110.0


def test_a_step_the_package_does_inside_its_call_is_not_separable_not_guessed():
    for adapter in ("valis", "valis_mirage"):
        r = _row(cost.end_to_end([res("v", "P1", adapter)]), "v")
        assert r["method_convert_s"] == cost.NOT_SEPARABLE
        assert "conversion" in r["estimate_includes"]
        assert r["measured_here_s"] == 110.0  # nothing is added for it


def test_a_methods_own_conversion_is_a_column_of_its_own():
    # DeeperHistReg converts per round; ASHLAR cuts the patient's tiles once
    dhr = [
        res("d", "P1", "deeperhistreg", case_id="a", method_convert_s=7.0),
        res("d", "P1", "deeperhistreg", case_id="b", method_convert_s=5.0),
    ]
    r = _row(cost.end_to_end(dhr), "d")
    assert r["method_convert_s"] == 12.0 and r["measured_here_s"] == 12.0 + 200 + 20
    assert r["measured_here"] == "method_convert, estimate, write"
    ash = [
        res("a", "P1", "ashlar", case_id=c, unit="patient", method_convert_s=40.0)
        for c in ("a", "b")
    ]
    r = _row(cost.end_to_end(ash), "a")
    # one launch per patient: its tile cutting and its clock are counted once
    assert r["method_convert_s"] == 40.0 and r["estimate_s"] == 100.0
    assert r["write_s"] == 20.0 and r["measured_here_s"] == 160.0


def test_the_benchmarks_own_copying_is_shown_and_is_in_no_total():
    """Where an adapter times its write step in parts, the writer column is the package's
    writer alone; a conversion that writer needed is conversion; the benchmark's own
    copy (ASHLAR's split, DeeperHistReg's stacking) is beside them and never summed."""
    # `write_s` of a row is already the package's writer alone (score.resource_row)
    row = res(
        "d", "P1", "deeperhistreg", method_convert_s=7.0, write_s=40.0,
        write_writer_s=40.0, write_convert_s=5.0, write_benchmark_s=15.0,
    )  # fmt: skip
    r = _row(cost.end_to_end([row]), "d")
    assert (r["method_convert_s"], r["write_s"], r["benchmark_copy_s"]) == (
        12.0,
        40.0,
        15.0,
    )
    assert r["measured_here_s"] == 12.0 + 100.0 + 40.0
    plain = _row(cost.end_to_end([res("stare", "P1", "stare")]), "stare")
    assert plain["write_s"] == 10.0 and math.isnan(plain["benchmark_copy_s"])


def test_the_headline_write_time_and_the_by_step_one_are_the_same_number():
    """A writer timed in parts (the package's writer 152 s, the benchmark's own copy
    8 s, 160 s in all): the headline `write_s` and `total_to_image_s`, the patient's
    row and the by-step column all take 152, and the copy is in none of them."""
    run = {
        "ok": True,
        "method": "ashlar",
        "unit": "patient",
        "launch_id": "5_0",
        "register_s": 1000.0,
        "write_s": 160.0,
        "resources": {"wall_s": 1000.0, "cpu_s": 3000.0, "peak_rss_mb": 2048.0},
        "native": {
            "files": ["/x/a.ome.tif"],
            "write_parts_s": {"writer": 152.0, "benchmark": 8.0},
        },
    }
    row = score.resource_row(run)
    assert row["write_s"] == 152.0 and row["write_benchmark_s"] == 8.0
    assert row["total_to_image_s"] == 1152.0
    base = {"dataset": "multiplex", "group": "P1", "method": "ashlar"}
    full = {**row, **base, "written_file": "/x/a.ome.tif"}
    (patient,) = score.summarise_resources([full])
    (step,) = cost.end_to_end([full])
    assert patient["write_s"] == step["write_s"] == 152.0
    assert patient["total_to_image_s"] == 1152.0
    assert step["benchmark_copy_s"] == 8.0
    assert step["measured_here_s"] == step["estimate_s"] + step["write_s"] == 1152.0
    (line,) = cost.summarise([step])
    (head,) = score.summarise_cost([patient])
    assert head["write_s_med"] == line["write_s"] == 152.0
    assert head["total_to_image_s_med"] == line["measured_here_s"] == 1152.0
    # a writer that is not timed in parts: its write step, as recorded
    plain = score.resource_row({**run, "native": {"files": ["/x/a.ome.tif"]}})
    assert plain["write_s"] == 160.0


def test_cost_by_step_reads_complete_patients_only():
    """A patient two of whose nine rounds wrote an image is in no median, as in the
    per-method block: the sum over two rounds is not the patient's cost."""
    whole = [res("stare", "P1", "stare"), res("stare", "P3", "stare", register_s=300.0)]
    partial = [
        {**res("stare", "P2", "stare", register_s=1.0, write_s=1.0), "ok": i < 2}
        for i in range(9)
    ]
    rows = cost.end_to_end([*whole, *partial])
    by = {r["group"]: r for r in rows}
    assert by["P1"]["complete"] and by["P3"]["complete"] and not by["P2"]["complete"]
    assert by["P2"]["n_rounds"] == 2
    (line,) = cost.summarise(rows)
    assert (line["n_patients"], line["n_patients_partial"]) == (2, 1)
    assert line["estimate_s"] == (100.0 + 300.0) / 2  # P2's 2 s is not in it
    # the same rule as the headline block
    per = score.summarise_resources([*whole, *partial])
    (head,) = score.summarise_cost(per)
    assert head["n_patients_complete"] == 2 and head["n_patients_partial"] == 1
    assert head["register_s_med"] == line["estimate_s"]


def test_cpu_and_gpu_rows_share_one_table_with_the_device_in_its_own_column():
    rows = cost.end_to_end(
        [
            res("deeperhistreg_hr_gpu_asdrawn", "P1", "deeperhistreg",
                device="cuda:0", gpu_model="NVIDIA H200", method_convert_s=1.0),
            res("deeperhistreg_hr_cpu_asdrawn", "P1", "deeperhistreg",
                method_convert_s=1.0),
        ]
    )  # fmt: skip
    assert [(r["method"], r["device"], r["gpu_model"]) for r in rows] == [
        ("deeperhistreg_hr_gpu_asdrawn", "cuda:0", "NVIDIA H200"),
        ("deeperhistreg_hr_cpu_asdrawn", "cpu", ""),
    ]
    summary = cost.summarise(rows)
    assert {"device", "gpu_model"} <= set(cost.SUMMARY_COLUMNS)
    assert [s["device"] for s in summary] == ["cuda:0", "cpu"]


def test_a_result_that_wrote_no_image_has_no_row():
    rigid = res("stare_rigid", "P1", "stare", written_file="", written="")
    failed = res("stare", "P1", "stare", ok=False)
    assert cost.end_to_end([rigid, failed]) == []


def test_the_summary_has_the_total_measured_time_and_the_peak_memory_per_method():
    """Median over the patients of every step and of their sum, the largest peaks, and
    the same medians on the patients every compared row completed."""
    rows = cost.end_to_end(
        [
            res("stare", "P1", "stare"),
            res("stare", "P2", "stare", register_s=300.0, write_s=30.0,
                peak_rss_gb=7.0, cgroup_peak_gb=9.0),
            res("stare", "P3", "stare", register_s=500.0, write_s=50.0,
                peak_rss_gb=4.0, cgroup_peak_gb=6.0),
        ]
    )  # fmt: skip
    (line,) = cost.summarise(rows, {"multiplex": ["P1", "P2"]}, {})
    assert set(line) == set(cost.SUMMARY_COLUMNS)
    assert (line["estimate_s"], line["write_s"], line["measured_here_s"]) == (
        300.0,
        30.0,
        330.0,
    )
    assert (line["estimate_peak_gb_max"], line["job_peak_gb_max"]) == (7.0, 9.0)
    assert (line["n_patients"], line["n_patients_common"]) == (3, 2)
    assert line["n_patients_partial"] == 0
    assert line["measured_here_s_common"] == (110.0 + 330.0) / 2
    assert line["merged_image"] == "no"


def test_the_summary_block_promises_only_what_was_measured_here():
    rows = cost.end_to_end([res("stare", "P1", "stare")])
    text = "\n".join(cost.markdown(rows, cost.summarise(rows), score.markdown))
    assert "## Cost by step, as measured here (secondary)" in text
    assert "nothing is taken from another run" in text
    assert "`measured_here_s` is the sum" in text and "merged_image" in text
    assert "are not measured here and are in no number" in text
    for word in ("trace", "imported", "IMPORTED", "total_s", "not available"):
        assert word not in text, word
    assert cost.markdown([], [], score.markdown) == []


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


# ── threads ───────────────────────────────────────────────────────────────────
def test_a_thread_variable_an_image_sets_is_recorded_and_said(tmp_path, monkeypatch):
    """The harness sets no thread variable (ruling). ASHLAR's official image sets
    OMP_NUM_THREADS=1 and `--cleanenv` keeps what an image defines, so the run finds it
    set in its own process: run.json records it as seen there, and every cost table's
    header names it."""
    from benchmarks.regbench import resources

    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    for name in names:
        monkeypatch.delenv(name, raising=False)
    assert resources.host_info()["threads_env"] == dict.fromkeys(names)
    case = toy_case(tmp_path / "cases", "P1_r1", "P1")
    out = tmp_path / "out"
    methods.run_case(_writer("free", False), case, out)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # as inside ASHLAR's image
    methods.run_case(_writer("held", False), case, out)
    # running a method changed nothing in the environment: recorded, never set
    assert {n: os.environ.get(n) for n in names} == {
        **dict.fromkeys(names),
        names[0]: "1",
    }
    seen = {
        v: json.loads((result_dir(out, v, case) / "run.json").read_text())[
            "threads_env"
        ]
        for v in ("free", "held")
    }
    assert seen["free"] == dict.fromkeys(names)
    assert seen["held"] == {**dict.fromkeys(names), "OMP_NUM_THREADS": "1"}
    rows = {v: score.resource_row({"threads_env": seen[v]}) for v in seen}
    assert rows["free"]["threads_env"] == ""
    assert rows["held"]["threads_env"] == "OMP_NUM_THREADS=1"
    assert score.resource_row({})["threads_env"] == ""  # an older record
    held = res("ashlar", "P1", "ashlar", threads_env="OMP_NUM_THREADS=1")
    free = res("stare", "P1", "stare", threads_env="")
    (line,) = [
        ln for ln in cost.header_lines([held, free]) if ln.startswith("thread var")
    ]
    assert "the benchmark sets none" in line
    assert "by the method's own image: ashlar OMP_NUM_THREADS=1." in line
    assert "stare" not in line and "fewer threads than the 4 CPUs" in line
    (none,) = [ln for ln in cost.header_lines([free]) if ln.startswith("thread var")]
    assert none.endswith("No run found one set in its own process.")
    monkeypatch.delenv("OMP_NUM_THREADS")
    got = score.score(tmp_path / "cases", out, ["multiplex"], with_tissue=False)
    assert {r["method"]: r["threads_env"] for r in got["resources"]} == {
        "free": "",
        "held": "OMP_NUM_THREADS=1",
    }
    text = (out / "tables" / "summary.md").read_text()
    assert "by the method's own image: held OMP_NUM_THREADS=1." in text


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

    got = score.score(root, out, ["multiplex"], with_tissue=False)
    rows = got["cost_end_to_end"]
    flat, deep = _row(rows, "flat"), _row(rows, "deep")
    # one pyramid holding the whole patient is the merged image; a flat file is not
    assert (flat["merged_image"], deep["merged_image"]) == ("no", "yes")
    # a pyramid per round is not the patient's merged image either
    per_round = _row(rows, "round")
    assert per_round["merged_image"] == "no"
    assert "pyramid (2 levels)" in per_round["written"]
    assert flat["write_s"] == deep["write_s"] == 2.0
    assert "no pyramid" in flat["written"] and "pyramid (2 levels)" in deep["written"]
    assert flat["written_from"] == "the file, read after the run"
    for r in rows:  # the total is what was measured here, and nothing else
        assert r["measured_here_s"] == pytest.approx(r["estimate_s"] + r["write_s"])

    with open(out / "tables" / "cost_end_to_end.csv", newline="") as fh:
        on_disk = list(csv.DictReader(fh))
    assert len(on_disk) == 6 and list(on_disk[0]) == list(cost.COLUMNS)
    with open(out / "tables" / "cost_end_to_end_summary.csv", newline="") as fh:
        summary = list(csv.DictReader(fh))
    assert len(summary) == 3 and set(cost.SUMMARY_COLUMNS) <= set(summary[0])
    assert {r["merged_image"] for r in summary} == {"no", "yes"}
    text = (out / "tables" / "summary.md").read_text()
    assert "## Cost by step, as measured here (secondary)" in text
    assert "cores confined:" in text
    # the headline cost table is still the registration alone, and the headline block
    # beside it has the time to the written image and both memory peaks per method
    assert "## Resources (per patient; registration only)" in text
    head = text[text.index("### Cost per method") : text.index("## Cost by step")]
    for column in ("register_s", "total_to_image_s", "peak_rss_gb", "cgroup_peak_gb"):
        assert f"| {column}: complete / common |" in head, column
    # ... with VALUES for a method that completed every patient, not only the cells
    names = [c.strip() for c in head.splitlines()[4].strip("|").split("|")]
    # CPU-hours come first, the device beside the GPU-hours, the times after them
    assert names[:9] == [
        "dataset",
        "method",
        "patients complete / all",
        "common",
        "cpu_h_used: complete / common",
        "cpu_h_reserved: complete / common",
        "cpu_efficiency: complete / common",
        "gpu_h: complete / common",
        "device",
    ]
    assert names.index("register_s: complete / common") > names.index("device")
    assert "Valid between rows: CPU-hours" in head and "NOT valid:" in head
    for method in ("flat", "deep", "round"):
        (line,) = [ln for ln in head.splitlines() if f"| {method} |" in ln]
        cells = dict(zip(names, (c.strip() for c in line.strip("|").split("|"))))
        assert cells["patients complete / all"] == "2/2"
        for column in ("register_s", "write_s", "total_to_image_s", "peak_rss_gb"):
            both = cells[f"{column}: complete / common"].split(" / ")
            assert len(both) == 2 and all(float(v) >= 0 for v in both), (method, column)
        written = float(cells["write_s: complete / common"].split(" / ")[0])
        total = float(cells["total_to_image_s: complete / common"].split(" / ")[0])
        estimate = float(cells["register_s: complete / common"].split(" / ")[0])
        assert written == 2.0 and total == pytest.approx(estimate + written, rel=1e-3)
    for word in ("pipeline_", "trace", "total_s", "not available"):
        assert word not in text[text.index("## Resources") :], word

    # the same through the command line
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
    cli.main(argv)
    with open(out / "tables" / "cost_end_to_end.csv", newline="") as fh:
        again = list(csv.DictReader(fh))
    assert again == on_disk


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
