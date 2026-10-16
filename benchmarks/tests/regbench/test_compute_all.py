"""compute-all: the full Mirage pipeline, traced end to end per patient, once per
registration backend (``valis``, ``tiled``), run last and alone on its own lanes, into
``pipeline_cost.csv``, ``pipeline_cost_summary.csv``, ``pipeline_cost_by_process.csv``
and a block of ``summary.md``.

No Nextflow runs here: a stand-in ``nextflow`` on PATH records what it was called with
and writes a trace, and the launcher's graph is read off its dry run.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.regbench import compute_all, params_json
from benchmarks.regbench import score as score_mod

from . import scored_cohort
from .test_submit_cohort import jobs_of, launch, site  # noqa: F401  (site: a fixture)

REPO = Path(__file__).resolve().parents[3]
SLURM = REPO / "benchmarks" / "regbench" / "slurm"
SBATCH = SLURM / "compute_all.sbatch"
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))
ci_actions = importlib.import_module("ci_actions")

HEADER = (
    "task_id\tprocess\ttag\tname\tstatus\texit\tsubmit\tstart\tcomplete\tduration\t"
    "realtime\t%cpu\tcpus\tmemory\tpeak_rss\tpeak_vmem\trchar\twchar"
)


def _sheet(path, patients=("P1", "P2", "P3"), rows_per=2):
    lines = ["patient_id,path_to_file,is_reference,channels"]
    for p in patients:
        for i in range(rows_per):
            lines.append(
                f"{p},/s/{p}_{i}.ome.tif,{'true' if i == 0 else 'false'},DAPI|m{i}"
            )
    path.write_text("\n".join(lines) + "\n")
    return path


def _row(
    task_id,
    process,
    status="COMPLETED",
    start_h=0,
    hours=1.0,
    cpus=4,
    pcpu=200.0,
    mem=16,
    rss=3.0,
):
    t0 = f"2026-10-01 {10 + start_h:02d}:00:00.000"
    t1 = f"2026-10-01 {10 + start_h + int(hours):02d}:{int(hours % 1 * 60):02d}:00.000"
    real = f"{int(hours)}h {int(hours % 1 * 60)}m"
    return "\t".join(
        map(
            str,
            (task_id, process, "P1", f"{process.split(':')[-1]} (P1)", status,
             0 if status == "COMPLETED" else 137, t0, t0, t1, real, real, f"{pcpu}%",
             cpus, f"{mem} GB", f"{rss} GB", f"{rss + 1} GB", "1 GB", "1 GB"),
        )
    )  # fmt: skip


def _trace(out, backend, patient, rows, end="ok"):
    d = out / "compute" / backend / patient
    (d / "trace").mkdir(parents=True, exist_ok=True)
    (d / "trace" / "trace.txt").write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    (d / "input_gb.txt").write_text("2.5\n")
    if end is not None:
        e = out / "logs" / "compute" / backend
        e.mkdir(parents=True, exist_ok=True)
        (e / f"{patient}.end").write_text(
            f"REGBENCH_END kind=compute backend={backend} patient={patient} "
            f"status={end} exit={0 if end == 'ok' else 1} log=-\n"
        )
    return d


VALIS_ROWS = [
    _row(1, "MIRAGE:PREPROCESS:CONVERT_IMAGE", hours=1.0, cpus=2, pcpu=100.0),
    _row(2, "MIRAGE:REGISTRATION:REGISTER_PATIENT:VALIS_ADAPTER:REGISTER", start_h=1,
         hours=2.0, cpus=8, pcpu=400.0, mem=64, rss=40.0),
    _row(3, "MIRAGE:POSTPROCESS:SEGMENT", start_h=3, hours=0.5, cpus=4, pcpu=300.0),
]  # fmt: skip
TILED_ROWS = [
    _row(1, "MIRAGE:PREPROCESS:CONVERT_IMAGE", hours=1.0, cpus=2, pcpu=100.0),
    _row(2, "MIRAGE:REGISTRATION:REGISTER_PATIENT:TILED_ADAPTER:TILED_COARSE", start_h=1, hours=0.5),
    _row(3, "MIRAGE:REGISTRATION:REGISTER_PATIENT:TILED_ADAPTER:TILED_REG_TILE", start_h=1, hours=1.0),
    _row(4, "MIRAGE:REGISTRATION:REGISTER_PATIENT:TILED_ADAPTER:TILED_SOLVE", start_h=2, hours=0.5),
    _row(5, "MIRAGE:REGISTRATION:REGISTER_PATIENT:TILED_ADAPTER:TILED_STITCH", start_h=2, hours=1.0),
    _row(6, "MIRAGE:POSTPROCESS:SEGMENT", start_h=3, hours=0.5),
]  # fmt: skip


# ── the plan: one launch dir, work dir, trace dir and name per run ────────────
def test_every_run_has_its_own_launch_work_and_trace_dir_and_name(tmp_path):
    runs = compute_all.plan(_sheet(tmp_path / "in.csv"), tmp_path / "out")
    assert len(runs) == 6
    for field in (
        "launch_dir",
        "work_dir",
        "trace_dir",
        "name",
        "params_file",
        "sheet",
    ):
        assert len({getattr(r, field) for r in runs}) == 6, field
    assert all(Path(r.trace_dir).name == "trace" for r in runs)
    assert not any(".trace" in Path(r.trace_dir).parts for r in runs)
    # all valis patients first, in the samplesheet's order, then all tiled
    assert [(r.backend, r.patient) for r in runs] == [
        (b, p) for b in ("valis", "tiled") for p in ("P1", "P2", "P3")
    ]
    r = runs[0]
    assert Path(r.launch_dir) == tmp_path / "out" / "compute" / "valis" / "P1"
    assert Path(r.work_dir) == Path(r.launch_dir) / "work"
    assert Path(r.outdir) == Path(r.launch_dir) / "results"
    assert Path(r.sheet) == Path(r.launch_dir) / "samplesheet.csv"
    assert Path(r.params_file) == Path(r.launch_dir) / "params.json"
    assert r.name == "rb-compute-valis-p1"
    assert all(re.fullmatch(r"[a-z0-9_-]+", r.name) for r in runs)


def test_two_patients_whose_run_names_collide_are_refused(tmp_path):
    with pytest.raises(ValueError, match="same run name"):
        compute_all.plan(_sheet(tmp_path / "in.csv", ("P 1", "p-1")), tmp_path / "o")
    with pytest.raises(ValueError, match="patient"):
        compute_all.plan(_sheet(tmp_path / "in.csv", ("../x",)), tmp_path / "o")


def test_the_patient_sheet_is_the_header_and_that_patients_rows(tmp_path):
    src = _sheet(tmp_path / "in.csv")
    src.write_text(src.read_text().replace("\n", "\r\n"))
    slide = tmp_path / "s.ome.tif"
    slide.write_bytes(b"x" * 2048)
    text = src.read_text().replace("/s/P2_0.ome.tif", str(slide))
    src.write_text(text)
    out = tmp_path / "P2.csv"
    sizes = tmp_path / "input_gb.txt"
    n = compute_all.sheet(src, "P2", out, sizes)
    assert n == 2
    lines = out.read_text().splitlines()
    assert lines[0] == "patient_id,path_to_file,is_reference,channels"
    assert [ln.split(",")[0] for ln in lines[1:]] == ["P2", "P2"]
    assert "\r" not in out.read_text()
    # one of the two slides is not there: the sum is not known
    assert math.isnan(float(sizes.read_text()))
    (tmp_path / "x.tif").write_bytes(b"y" * 1024)
    src.write_text(text.replace("/s/P2_1.ome.tif", str(tmp_path / "x.tif")))
    compute_all.sheet(src, "P2", out, sizes)
    assert float(sizes.read_text()) == pytest.approx(3072 / 2**30)
    with pytest.raises(ValueError, match="no row"):
        compute_all.sheet(src, "P9", out)


# ── the guard: what compute-all pins is real and typed by the schema ──────────
def _schema_entries():
    doc = json.loads((REPO / "nextflow_schema.json").read_text())
    found = {}

    def walk(node):
        if isinstance(node, dict):
            for name, entry in (node.get("properties") or {}).items():
                if isinstance(entry, dict) and "type" in entry:
                    found[name] = entry
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(doc)
    return found


def _default(key):
    """The shipped default, read the way test_outline_definition._default reads it."""
    m = re.search(
        rf"^\s+{key}\s*=\s*([^\s/]+)", (REPO / "nextflow.config").read_text(), re.M
    )
    assert m, f"{key} has no default in nextflow.config"
    return m.group(1).strip("'\"")


def _as_config(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


JSON_TYPE = {bool: "boolean", str: "string", int: "integer", float: "number"}


def test_the_pinned_params_are_real_and_at_their_shipped_default():
    types = params_json.schema_types()
    entries = _schema_entries()
    for backend in compute_all.BACKENDS:
        for key, value in compute_all.params(backend).items():
            assert key in types, f"{key} is not a param of nextflow_schema.json"
            want = JSON_TYPE[type(value)]
            assert want in types[key], f"{key}={value!r}: the schema says {types[key]}"
            _schema_valid(key, value, entries[key])
    assert set(compute_all.BACKENDS) <= set(entries["registration_method"]["enum"])
    for key, value in compute_all.PINNED.items():
        assert _as_config(value) == _default(key), (
            f"{key}: compute-all restates {value!r}, nextflow.config ships {_default(key)}"
        )
    assert _default("registration_method") in compute_all.BACKENDS
    # the rulings (2026-10-15): StarDist, the benchmark's own nuclei, and the input
    # step's pixel size. NOT the shipped defaults: rulings, not restatements
    assert compute_all.RULED == {"seg_method": "stardist", "pixel_size": 0.325}
    assert not set(compute_all.RULED) & set(compute_all.PINNED)
    for key, value in compute_all.RULED.items():
        assert _as_config(value) != _default(key), f"{key} is a restatement: PINNED"
    # the same scale as the inputs the benchmark scores (slurm/inputs.sbatch)
    inputs = (REPO / "benchmarks/regbench/slurm/inputs.sbatch").read_text()
    (scale,) = re.findall(r"\spixel_size=([0-9.]+)", inputs)
    assert float(scale) == compute_all.RULED["pixel_size"]


def _schema_valid(key, value, entry):
    """The checks nextflow_schema.json declares for one value: enum, pattern (on its
    text, as nf-schema reads a number given for a string-or-number param), bounds."""
    if "enum" in entry:
        assert value in entry["enum"], f"{key}={value!r} not in its enum"
    if "pattern" in entry and not isinstance(value, bool):
        assert re.fullmatch(entry["pattern"], str(value)), (
            f"{key}={value!r} does not match {entry['pattern']}"
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "exclusiveMinimum" in entry:
            assert value > entry["exclusiveMinimum"], f"{key}={value!r} too small"
        if "minimum" in entry:
            assert value >= entry["minimum"], f"{key}={value!r} too small"
        if "maximum" in entry:
            assert value <= entry["maximum"], f"{key}={value!r} too large"


def test_compute_all_pins_nothing_else():
    for backend in compute_all.BACKENDS:
        assert set(compute_all.params(backend)) == set(compute_all.PINNED) | set(
            compute_all.RULED
        ) | {"registration_method"}
        assert compute_all.params(backend)["registration_method"] == backend
    with pytest.raises(ValueError):
        compute_all.params("ashlar")


def test_the_params_file_is_typed_through_params_json(tmp_path):
    out = tmp_path / "p.json"
    done = subprocess.run(
        [sys.executable, "-m", "benchmarks.regbench", "compute-all", "params",
         "--backend", "tiled", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    doc = json.loads(out.read_text())
    assert doc == compute_all.params("tiled")
    assert doc["enable_trace"] is True
    assert doc["pixel_size"] == 0.325 and isinstance(doc["pixel_size"], float)


# ── the launcher: never resumed, every typed param in the params file ─────────
def code_lines(path):
    """The script's commands, comment-blind, with backslash continuations joined."""
    out, cur = [], ""
    for line in Path(path).read_text().splitlines():
        line = ci_actions.strip_line_comment(line).rstrip()
        if line.endswith("\\"):
            cur += line[:-1] + " "
            continue
        cur += line
        if cur.strip():
            out.append(cur.strip())
        cur = ""
    return out


def test_the_launcher_never_resumes_and_types_every_param():
    lines = code_lines(SBATCH)
    code = "\n".join(lines)
    assert "-resume" not in code
    runs = [ln for ln in lines if re.search(r"(^|\s)nextflow\s", ln)]
    assert len(runs) == 1 and re.search(r"\snextflow\s+-q\s+run\s", runs[0]), runs
    nf = runs[0]
    for flag in (
        "-params-file",
        "--trace_dir",
        "-work-dir",
        "-name",
        "--input",
        "--outdir",
    ):
        assert re.search(rf"(^|\s){re.escape(flag)}\s", nf), flag
    assert set(re.findall(r"\s--([a-z_]+)", nf)) == {"input", "outdir", "trace_dir"}
    for typed in (
        "enable_trace",
        "cleanup_work",
        "registration_method",
        "seg_method",
        "pixel_size",
    ):
        assert f"--{typed}" not in code, typed
    assert "env -u PYTHONPATH nextflow" in nf
    assert "compute-all params" in code and "compute-all sheet" in code


def test_the_launcher_is_executable():
    mode = subprocess.run(
        ["git", "ls-files", "-s", str(SBATCH.relative_to(REPO))],
        capture_output=True, text=True, cwd=REPO,
    ).stdout.split()  # fmt: skip
    assert mode and mode[0] == "100755"


def _fake_nextflow(tmp_path, fail=False):
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    nf = bin_ / "nextflow"
    nf.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" > "$NF_LOG.argv"\n'
        'pwd > "$NF_LOG.cwd"\n'
        'echo "${PYTHONPATH-unset}" > "$NF_LOG.pythonpath"\n'
        'tr="";\n'
        'while [ $# -gt 0 ]; do case "$1" in\n'
        '  --trace_dir) tr="$2"; shift 2 ;;\n'
        "  *) shift ;;\n"
        "esac; done\n"
        'mkdir -p "$tr" && printf "task_id\\tprocess\\tstatus\\n1\\tX\\tCOMPLETED\\n" > "$tr/trace.txt"\n'
        'echo "nextflow said something"\n'
        + ('echo "it broke" >&2\nexit 3\n' if fail else "")
    )
    nf.chmod(0o755)
    return bin_


def _run_head(tmp_path, bin_, backend="valis", patient="P1", **more):
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("REGBENCH_", "SLURM_"))
    }
    env.pop("NXF_VER", None)
    env.update(
        PATH=f"{bin_}:{Path(sys.executable).parent}:{env['PATH']}",
        REGBENCH_REPO=str(REPO),
        REGBENCH_INPUT=str(_sheet(tmp_path / "in.csv")),
        REGBENCH_OUT=str(tmp_path / "out"),
        REGBENCH_CASES=str(tmp_path / "cases"),
        REGBENCH_SITE_CONFIG="none",
        REGBENCH_SUBMISSION="20261016T101010-77",
        SLURM_JOB_ID="4242",
        NF_LOG=str(tmp_path / "nf"),
    )
    env.update(more)
    return subprocess.run(
        ["bash", str(SBATCH), backend, patient],
        env=env, capture_output=True, text=True, cwd=tmp_path,
    )  # fmt: skip


def test_a_head_runs_the_whole_pipeline_in_its_own_folder(tmp_path):
    bin_ = _fake_nextflow(tmp_path)
    done = _run_head(tmp_path, bin_, "tiled", "P2")
    assert done.returncode == 0, done.stdout + done.stderr
    d = tmp_path / "out" / "compute" / "tiled" / "P2"
    argv = (tmp_path / "nf.argv").read_text().splitlines()
    assert argv[:3] == ["-q", "run", str(REPO)]
    assert "--start" not in argv and "--stop" not in argv
    assert argv[argv.index("-work-dir") + 1] == str(d / "work")
    assert argv[argv.index("--outdir") + 1] == str(d / "results")
    assert argv[argv.index("--trace_dir") + 1] == str(d / "trace")
    assert argv[argv.index("--input") + 1] == str(d / "samplesheet.csv")
    assert argv[argv.index("-params-file") + 1] == str(d / "params.json")
    assert (
        argv[argv.index("-name") + 1] == "rb-compute-tiled-p2-20261016t101010-77-4242"
    )
    assert (tmp_path / "nf.cwd").read_text().strip() == str(d)
    assert (tmp_path / "nf.pythonpath").read_text().strip() == "unset"
    assert json.loads((d / "params.json").read_text()) == compute_all.params("tiled")
    rows = (d / "samplesheet.csv").read_text().splitlines()
    assert len(rows) == 3 and all(r.startswith("P2,") for r in rows[1:])
    assert (d / "input_gb.txt").is_file()
    assert "nextflow said something" in (d / "nextflow.stdout.log").read_text()
    end = (tmp_path / "out" / "logs" / "compute" / "tiled" / "P2.end").read_text()
    assert end.startswith(
        "REGBENCH_END kind=compute backend=tiled patient=P2 status=ok exit=0"
    )
    assert end.strip() in done.stdout

    # a second head of the same run: the first is moved aside, never deleted or reused
    again = _run_head(tmp_path, bin_, "tiled", "P2", SLURM_JOB_ID="4243")
    assert again.returncode == 0, again.stdout + again.stderr
    (aside,) = (tmp_path / "out" / "compute" / ".replaced").iterdir()
    assert (aside / "tiled" / "P2" / "trace" / "trace.txt").is_file()
    assert (aside / "tiled" / "P2.end").is_file()
    assert compute_all.rows(tmp_path / "out")  # only the new run is read
    assert len(compute_all.rows(tmp_path / "out")) == 1


def test_a_failed_head_says_so_with_the_ends_of_its_logs(tmp_path):
    bin_ = _fake_nextflow(tmp_path, fail=True)
    done = _run_head(tmp_path, bin_)
    assert done.returncode == 3
    assert "status=failed exit=3" in done.stdout
    assert "it broke" in done.stderr and "nextflow said something" in done.stderr
    end = (tmp_path / "out" / "logs" / "compute" / "valis" / "P1.end").read_text()
    assert "status=failed" in end


def test_a_head_refuses_an_unknown_backend_or_patient(tmp_path):
    bin_ = _fake_nextflow(tmp_path)
    assert _run_head(tmp_path, bin_, "ashlar").returncode == 2
    assert _run_head(tmp_path, bin_, "valis", "P9").returncode == 2
    assert not (tmp_path / "nf.argv").exists()
    assert _run_head(tmp_path, bin_, REGBENCH_INPUT="").returncode == 2


# ── the tables ────────────────────────────────────────────────────────────────
def test_rows_and_summary(tmp_path):
    out = tmp_path / "out"
    _trace(out, "valis", "P1", VALIS_ROWS)
    _trace(out, "tiled", "P1", TILED_ROWS)
    rows = compute_all.rows(out)
    assert len(rows) == len(VALIS_ROWS) + len(TILED_ROWS)
    assert all(list(r) == list(compute_all.COLUMNS) for r in rows)
    reg = {(r["backend"], r["process"]) for r in rows if r["is_registration"] == 1}
    assert reg == {("valis", "REGISTER")} | {
        ("tiled", p)
        for p in ("TILED_COARSE", "TILED_REG_TILE", "TILED_SOLVE", "TILED_STITCH")
    }
    reg_row = next(r for r in rows if r["process"] == "REGISTER")
    assert reg_row["process_path"].endswith(":VALIS_ADAPTER:REGISTER")
    assert reg_row["cpu_h_used"] == pytest.approx(4.0 * 2.0)  # 400 % for 2 h
    assert reg_row["cpu_h_reserved"] == pytest.approx(8 * 2.0)
    assert reg_row["gb_h_reserved"] == pytest.approx(64 * 2.0)
    summary = compute_all.summarise(rows)
    assert [list(s) for s in summary] == [list(compute_all.SUMMARY_COLUMNS)] * len(
        summary
    )
    v = next(s for s in summary if s["backend"] == "valis" and s["patient"] == "P1")
    want = 1.0 * 1.0 + 4.0 * 2.0 + 3.0 * 0.5
    assert v["cpu_h_used"] == pytest.approx(want)
    assert v["registration_cpu_h_used"] == pytest.approx(8.0)
    assert v["registration_share_cpu_h_used"] == pytest.approx(8.0 / want)
    assert v["wall_s"] == pytest.approx(3.5 * 3600)
    assert v["registration_wall_s"] == pytest.approx(2 * 3600)
    assert v["peak_rss_gb"] == 40.0 and v["input_gb"] == 2.5
    assert v["complete_run"] == 1 and v["n_tasks"] == 3 and v["n_failed"] == 0
    t = next(s for s in summary if s["backend"] == "tiled" and s["patient"] == "P1")
    assert t["registration_wall_s"] == pytest.approx(2 * 3600)
    total = next(
        s for s in summary if s["backend"] == "valis" and s["patient"] == "all"
    )
    assert total["n_patients"] == 1 and total["cpu_h_used"] == pytest.approx(want)
    by = compute_all.by_process(rows, summary)
    assert {(b["backend"], b["process"]) for b in by} >= {("valis", "REGISTER")}
    assert all(list(b) == list(compute_all.BY_PROCESS_COLUMNS) for b in by)
    text = "\n".join(compute_all.markdown(rows, summary))
    assert "## Full pipeline cost, per backend (secondary)" in text
    assert "different clock" in text and "no total adds the two" in text


def test_a_cached_or_failed_task_keeps_the_patient_out_of_the_total(tmp_path):
    out = tmp_path / "out"
    _trace(out, "valis", "P1", VALIS_ROWS)
    cached = [VALIS_ROWS[0].replace("COMPLETED", "CACHED")] + VALIS_ROWS[1:]
    _trace(out, "valis", "P2", cached)
    failed = VALIS_ROWS + [
        _row(9, "MIRAGE:POSTPROCESS:QUANTIFY", status="FAILED", start_h=4)
    ]
    _trace(out, "valis", "P3", failed, end="failed")
    _trace(out, "valis", "P4", VALIS_ROWS, end=None)  # still running: no end line
    summary = compute_all.summarise(compute_all.rows(out))
    by = {s["patient"]: s for s in summary if s["backend"] == "valis"}
    assert by["P1"]["complete_run"] == 1
    assert by["P2"]["complete_run"] == 0 and by["P2"]["n_cached"] == 1
    assert by["P3"]["complete_run"] == 0 and by["P3"]["n_failed"] == 1
    assert by["P4"]["complete_run"] == 0
    # a failed task was paid for: it counts in its patient's row
    assert by["P3"]["cpu_h_used"] > by["P1"]["cpu_h_used"]
    assert by["all"]["n_patients"] == 1
    assert by["all"]["cpu_h_used"] == pytest.approx(by["P1"]["cpu_h_used"])
    assert by["all"]["n_tasks"] == by["P1"]["n_tasks"]
    assert by["all"]["complete_run"] == 0  # not every patient is in the total
    text = "\n".join(compute_all.markdown(compute_all.rows(out), summary))
    assert "kept out of the `all` row" in text


def test_a_task_retried_to_success_is_cost_not_a_failed_run(tmp_path):
    """The pipeline retries a task that ran out of memory (its errorStrategy). The
    failed attempt is cost, counted; the run is complete all the same."""
    out = tmp_path / "out"
    oom = _row(2, "MIRAGE:REGISTRATION:REGISTER_PATIENT:VALIS_ADAPTER:REGISTER",
               status="FAILED", start_h=1, hours=1.0, cpus=8, pcpu=400.0)  # fmt: skip
    retried = [VALIS_ROWS[0], oom, "4" + VALIS_ROWS[1][1:], VALIS_ROWS[2]]
    _trace(out, "valis", "P1", retried)
    (s, _all) = [x for x in compute_all.summarise(compute_all.rows(out))]
    assert s["n_failed"] == 1 and s["n_retried"] == 1 and s["complete_run"] == 1
    assert s["registration_cpu_h_used"] == pytest.approx(4.0 + 8.0)


def test_score_writes_the_pipeline_tables_and_the_block(tmp_path):
    root, out, _ = scored_cohort.build(tmp_path / "a")
    score_mod.score(root, out, **scored_cohort.SCORE_KW)
    tables = out / "tables"
    text = (tables / "summary.md").read_text()
    assert (
        "Full pipeline cost: not measured (no <out>/compute/*/*/trace/trace.txt; submit_cohort.sh --compute-all)."
        in text
    )
    for name in ("pipeline_cost", "pipeline_cost_summary", "pipeline_cost_by_process"):
        assert not (tables / f"{name}.csv").exists()

    _trace(out, "valis", "P1", VALIS_ROWS)
    _trace(out, "tiled", "P1", TILED_ROWS)
    got = score_mod.score(root, out, **scored_cohort.SCORE_KW)
    with open(tables / "pipeline_cost.csv") as fh:
        assert next(csv.reader(fh)) == list(compute_all.COLUMNS)
    with open(tables / "pipeline_cost_summary.csv") as fh:
        assert next(csv.reader(fh)) == list(compute_all.SUMMARY_COLUMNS)
    assert (tables / "pipeline_cost_by_process.csv").is_file()
    text = (tables / "summary.md").read_text()
    assert "## Full pipeline cost, per backend" in text and "not measured" not in text
    assert len(got["pipeline_cost"]) == len(VALIS_ROWS) + len(TILED_ROWS)
    assert got["pipeline_cost_summary"]

    # the traces gone again: no stale table is left behind
    import shutil

    shutil.rmtree(out / "compute")
    score_mod.score(root, out, **scored_cohort.SCORE_KW)
    assert not (tables / "pipeline_cost.csv").exists()


# ── the submitter: compute runs last, alone, on its own lanes ─────────────────
def _compute_launch(site, *args, env=None, log="compute"):  # noqa: F811
    sheet = _sheet(site[0] / f"{log}-input.csv", ("P1", "P2"))
    return launch(
        site, *args, env={"REGBENCH_INPUT": str(sheet), **(env or {})}, log=log
    )


def test_the_submitter_runs_compute_last_on_lanes(site):  # noqa: F811
    done, _, _ = _compute_launch(site, "--compute-all")
    jobs = jobs_of(done)
    names = list(jobs)
    heads = [
        "rb-compute-valis-P1",
        "rb-compute-valis-P2",
        "rb-compute-tiled-P1",
        "rb-compute-tiled-P2",
    ]
    assert [n for n in names if n.startswith("rb-compute-")] == heads
    # after the scorer, and everything else before them
    assert names.index("rb-score") < names.index(heads[0])
    assert names[-1] == "rb-score-compute"
    # two lanes (the default): the first head of each lane after the scorer, every
    # other head after the head before it on its lane
    assert jobs[heads[0]].deps == {("afterany", "<rb-score>")}
    assert jobs[heads[1]].deps == {("afterany", "<rb-score>")}
    assert jobs[heads[2]].deps == {("afterany", f"<{heads[0]}>")}
    assert jobs[heads[3]].deps == {("afterany", f"<{heads[1]}>")}
    assert jobs["rb-score-compute"].deps == {("afterany", f"<{h}>") for h in heads}
    assert jobs["rb-score-compute"].script == "score_cohort.sbatch"
    assert jobs["rb-score-compute"].args == ["score"]
    for h in heads:
        j = jobs[h]
        assert j.script == "compute_all.sbatch"
        assert j.args == h.split("-")[2:4]
        assert j.opt("--cpus-per-task") == "2" and j.opt("--mem") == "8G"
        assert j.opt("--time") == "7-00:00:00"
    # one lane: every head after the one before it
    one = jobs_of(
        _compute_launch(site, "--compute-all", env={"REGBENCH_COMPUTE_LANES": "1"})[0]
    )
    for a, b in zip(heads, heads[1:]):
        assert one[b].deps == {("afterany", f"<{a}>")}


def test_compute_only_submits_no_ladder(site):  # noqa: F811
    done, _, _ = _compute_launch(
        site,
        "--compute-only",
        env={"REGBENCH_SUBMISSION": "20261012T010203-9"},
        log="co",
    )
    jobs = jobs_of(done)
    assert list(jobs) == [
        "rb-compute-valis-P1", "rb-compute-valis-P2",
        "rb-compute-tiled-P1", "rb-compute-tiled-P2", "rb-score-compute",
    ]  # fmt: skip
    assert jobs["rb-compute-valis-P1"].deps == set()
    assert jobs["rb-compute-valis-P2"].deps == set()
    assert jobs["rb-compute-tiled-P1"].deps == {("afterany", "<rb-compute-valis-P1>")}


def test_compute_needs_the_samplesheet_and_is_its_own_step(site, tmp_path):  # noqa: F811
    for flag in ("--compute-all", "--compute-only"):
        done, _, _ = launch(site, flag, log="nosheet")
        assert done.returncode == 2 and "REGBENCH_INPUT" in done.stderr
        assert "sbatch " not in done.stdout
    for other in ("--prepare", "--inputs", "--score-only"):
        done, _, _ = _compute_launch(site, "--compute-all", other, log="both")
        assert done.returncode == 2 and "two different steps" in done.stderr
    both, _, _ = _compute_launch(site, "--compute-all", "--compute-only", log="both")
    assert both.returncode == 2
    bad, _, _ = _compute_launch(
        site, "--compute-all", env={"REGBENCH_COMPUTE_LANES": "0"}, log="lanes"
    )
    assert bad.returncode == 2 and "REGBENCH_COMPUTE_LANES" in bad.stderr


def test_launch_sbatch_passes_compute_through_and_needs_the_samplesheet(tmp_path):
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("REGBENCH_", "SLURM_"))
    }
    env.update(
        PATH=f"{Path(sys.executable).parent}:{env['PATH']}",
        REGBENCH_REPO=str(REPO),
        REGBENCH_ROOT=str(tmp_path / "root"),
        REGBENCH_SITE_CONFIG="none",
        REGBENCH_ALLOW_PATH_PYTHON="1",
        REGBENCH_PREFLIGHT="0",
    )
    for flag in ("--compute-all", "--compute-only"):
        done = subprocess.run(
            ["bash", str(SLURM / "launch.sbatch"), flag, "--dry-run"],
            env=env, capture_output=True, text=True, cwd=tmp_path,
        )  # fmt: skip
        assert done.returncode == 2, done.stdout + done.stderr
        assert (
            done.stderr.startswith("regbench-launch:")
            and "REGBENCH_INPUT" in done.stderr
        )
