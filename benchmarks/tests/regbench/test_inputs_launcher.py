"""The input step: regbench runs the pipeline's own preprocessing for the cohort
(``launch.sbatch --inputs`` -> ``slurm/inputs.sbatch``), and nothing of it reads the
arms' results any more.

No Nextflow runs here: a stand-in ``nextflow`` on PATH records what it was called with
and writes the checkpoint the real preprocessing step writes.
"""

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SLURM = REPO / "benchmarks" / "regbench" / "slurm"
INPUTS = SLURM / "inputs.sbatch"
LAUNCHER = SLURM / "submit_cohort.sh"
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))
ci_actions = importlib.import_module("ci_actions")


def code_lines(path):
    """The script's commands, comment-blind (``ci_actions.strip_line_comment``), with
    backslash continuations joined: one entry per command."""
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


def _nextflow_line():
    (line,) = [ln for ln in code_lines(INPUTS) if re.search(r"(^|\s)nextflow\s", ln)]
    return line


def test_inputs_runs_preprocessing_only_at_the_shipped_defaults():
    lines = code_lines(INPUTS)
    (typed,) = [ln for ln in lines if "benchmarks.regbench.params_json" in ln]
    pairs = re.findall(r"\s([a-z_]+=[^\s;]+)", typed)
    # user ruling 2026-10-15: pixel_size 0.325 um as the arms ran; nothing else pinned
    assert pairs == ["start=preprocessing", "stop=preprocessing", "pixel_size=0.325"]
    nf = _nextflow_line()
    for flag in (
        "-params-file",
        "--input",
        "--outdir",
        "--trace_dir",
        "-work-dir",
        "-name",
        "-profile",
    ):
        assert re.search(rf"(^|\s){re.escape(flag)}\s", nf), flag
    assert "-resume" not in nf
    # every typed param goes through the params file: on the command line only paths
    assert set(re.findall(r"\s--([a-z_]+)", nf)) == {"input", "outdir", "trace_dir"}
    assert '--trace_dir "$d/trace"' in nf and '-work-dir "$d/work"' in nf
    assert '--outdir "$d/results"' in nf and '-params-file "$d/params.json"' in nf
    assert 'd="$REGBENCH_INPUTS/preprocess"' in lines
    assert 'cd "$d"' in " ".join(lines)
    assert 'NXF_VER="${NXF_VER:-25.04.7}"' in " ".join(lines)


def _sheet(tmp_path):
    sheet = tmp_path / "input.csv"
    sheet.write_text("patient_id,path_to_file,is_reference\nP1,/s/a.ome.tif,true\n")
    return sheet


def _fake_nextflow(tmp_path, fail=False):
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    nf = bin_ / "nextflow"
    nf.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" > "$NF_LOG.argv"\n'
        'pwd > "$NF_LOG.cwd"\n'
        'echo "${NXF_VER:-}" > "$NF_LOG.ver"\n'
        'echo "${PYTHONPATH-unset}" > "$NF_LOG.pythonpath"\n'
        'out=""; pf=""\n'
        'while [ $# -gt 0 ]; do case "$1" in\n'
        '  --outdir) out="$2"; shift 2 ;;\n'
        '  -params-file) pf="$2"; shift 2 ;;\n'
        "  *) shift ;;\n"
        "esac; done\n"
        'cp "$pf" "$NF_LOG.params"\n'
        + ("exit 1\n" if fail else "")
        + 'mkdir -p "$out/csv" && echo patient_id > "$out/csv/preprocessed.csv"\n'
    )
    nf.chmod(0o755)
    return bin_


def _run(tmp_path, bin_, **more):
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("REGBENCH_", "SLURM_"))
    }
    env.pop("NXF_VER", None)
    env.update(
        PATH=f"{bin_}:{Path(sys.executable).parent}:{env['PATH']}",
        REGBENCH_REPO=str(REPO),
        REGBENCH_INPUT=str(_sheet(tmp_path)),
        REGBENCH_INPUTS=str(tmp_path / "inputs"),
        REGBENCH_SITE_CONFIG="none",
        REGBENCH_SUBMISSION="20261016T101010-77",
        SLURM_JOB_ID="4242",
        PYTHONPATH=str(REPO),  # as the launcher's env.sh exports it
        NF_LOG=str(tmp_path / "nf"),
    )
    env.update(more)
    return subprocess.run(
        ["bash", str(INPUTS)], env=env, capture_output=True, text=True, cwd=tmp_path
    )


def test_the_input_step_refuses_to_overwrite_and_moves_aside_when_forced(tmp_path):
    bin_ = _fake_nextflow(tmp_path)
    d = tmp_path / "inputs" / "preprocess"
    done = _run(tmp_path, bin_)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "REGBENCH_END kind=inputs status=ok" in done.stdout
    argv = (tmp_path / "nf.argv").read_text().splitlines()
    assert argv[:3] == ["-q", "run", str(REPO)]
    assert argv[argv.index("-profile") + 1] == "singularity,ieo"
    assert "-c" not in argv  # REGBENCH_SITE_CONFIG=none: no site config
    assert argv[argv.index("-work-dir") + 1] == str(d / "work")
    assert argv[argv.index("--outdir") + 1] == str(d / "results")
    assert argv[argv.index("--trace_dir") + 1] == str(d / "trace")
    assert argv[argv.index("--input") + 1] == str(tmp_path / "input.csv")
    name = argv[argv.index("-name") + 1]
    assert name == "rb-inputs-20261016t101010-77-4242"  # a run name Nextflow accepts
    # the repository root on PYTHONPATH never reaches the pipeline (nor its tasks)
    assert (tmp_path / "nf.pythonpath").read_text().strip() == "unset"
    assert (tmp_path / "nf.cwd").read_text().strip() == str(d)
    assert (tmp_path / "nf.ver").read_text().strip() == "25.04.7"
    params = json.loads((tmp_path / "nf.params").read_text())
    assert params == {
        "start": "preprocessing",
        "stop": "preprocessing",
        "pixel_size": 0.325,
    }
    assert isinstance(params["pixel_size"], float)
    first = (d / "results" / "csv" / "preprocessed.csv").stat().st_mtime_ns

    # a second launch: refused, nothing is touched, Nextflow is not started
    (tmp_path / "nf.argv").unlink()
    again = _run(tmp_path, bin_)
    assert again.returncode == 2
    assert (
        "already exists" in again.stderr and "REGBENCH_INPUTS_FORCE=1" in again.stderr
    )
    assert "REGBENCH_END kind=inputs status=failed" in again.stdout
    assert not (tmp_path / "nf.argv").exists()
    assert (d / "results" / "csv" / "preprocessed.csv").stat().st_mtime_ns == first

    # forced: the old folder is moved aside, never deleted, and the step runs again
    forced = _run(tmp_path, bin_, REGBENCH_INPUTS_FORCE="1")
    assert forced.returncode == 0, forced.stdout + forced.stderr
    (old,) = (tmp_path / "inputs" / "preprocess.replaced").iterdir()
    assert (old / "results" / "csv" / "preprocessed.csv").is_file()
    assert (old / "params.json").is_file()
    assert (d / "results" / "csv" / "preprocessed.csv").is_file()


def test_a_failed_preprocessing_is_said(tmp_path):
    bin_ = _fake_nextflow(tmp_path, fail=True)
    done = _run(tmp_path, bin_)
    assert done.returncode == 1
    assert "REGBENCH_END kind=inputs status=failed" in done.stdout
    # a site config is passed with -c, as the arms did
    site = tmp_path / "site.config"
    site.write_text("params {}\n")
    shutil_rm(tmp_path / "inputs")
    _run(tmp_path, bin_, REGBENCH_SITE_CONFIG=str(site), REGBENCH_NF_PROFILE="x,y")
    argv = (tmp_path / "nf.argv").read_text().splitlines()
    assert argv[argv.index("-c") + 1] == str(site)
    assert argv[argv.index("-profile") + 1] == "x,y"


def shutil_rm(path):
    import shutil

    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.parametrize("missing", ["REGBENCH_INPUT", "REGBENCH_INPUTS"])
def test_the_input_step_needs_its_samplesheet_and_folder(tmp_path, missing):
    bin_ = _fake_nextflow(tmp_path)
    done = _run(tmp_path, bin_, **{missing: ""})
    assert done.returncode == 2 and missing in done.stderr
    assert not (tmp_path / "nf.argv").exists()


def test_launch_has_no_arm_results_left():
    for path in sorted(SLURM.iterdir()):
        if not path.is_file():
            continue
        code = "\n".join(code_lines(path))
        assert "REGBENCH_ARMS" not in code, path.name
        assert "arm_results" not in code, path.name
        assert "arms_manifest" not in code, path.name


def test_the_input_step_is_executable_like_the_launcher():
    mode = subprocess.run(
        ["git", "ls-files", "-s", str(INPUTS.relative_to(REPO))],
        capture_output=True,
        text=True,
        cwd=REPO,
    ).stdout.split()
    assert mode and mode[0] == "100755"


def _launch(tmp_path, *args, **env):
    base = {
        k: v for k, v in os.environ.items() if not k.startswith(("REGBENCH_", "SLURM_"))
    }
    base.update(
        PATH=f"{Path(sys.executable).parent}:{base['PATH']}",
        REGBENCH_CASES=str(tmp_path / "cases"),
        REGBENCH_OUT=str(tmp_path / "out"),
        REGBENCH_SITE_CONFIG="none",
        REGBENCH_ALLOW_PATH_PYTHON="1",
        REGBENCH_PREFLIGHT="0",
        **env,
    )
    return subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", *args],
        env=base,
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def test_the_launcher_submits_the_input_step_as_one_head_job(tmp_path):
    inputs = tmp_path / "inputs"
    done = _launch(
        tmp_path,
        "--inputs",
        REGBENCH_INPUT=str(_sheet(tmp_path)),
        REGBENCH_INPUTS=str(inputs),
    )
    assert done.returncode == 0, done.stderr
    (line,) = [ln for ln in done.stdout.splitlines() if ln.startswith("sbatch ")]
    assert "--job-name=rb-inputs" in line and line.endswith(str(INPUTS))
    assert "--cpus-per-task=2" in line and "--mem=8G" in line
    assert "--time=2-00:00:00" in line
    # it is not a submission of the benchmark: nothing else is submitted
    assert "run_rung" not in done.stdout and "rb-nuclei" not in done.stdout
    longer = _launch(
        tmp_path,
        "--inputs",
        REGBENCH_INPUT=str(_sheet(tmp_path)),
        REGBENCH_INPUTS=str(inputs),
        REGBENCH_INPUTS_TIME="3-00:00:00",
    )
    assert "--time=3-00:00:00" in longer.stdout
    # refused up front: no samplesheet, or a preprocessing that is already there
    none = _launch(tmp_path, "--inputs", REGBENCH_INPUTS=str(inputs))
    assert none.returncode == 2 and "REGBENCH_INPUT" in none.stderr
    done_before = inputs / "preprocess" / "results" / "csv" / "preprocessed.csv"
    done_before.parent.mkdir(parents=True)
    done_before.write_text("x\n")
    there = _launch(
        tmp_path,
        "--inputs",
        REGBENCH_INPUT=str(_sheet(tmp_path)),
        REGBENCH_INPUTS=str(inputs),
    )
    assert there.returncode == 2 and "REGBENCH_INPUTS_FORCE=1" in there.stderr
    forced = _launch(
        tmp_path,
        "--inputs",
        REGBENCH_INPUT=str(_sheet(tmp_path)),
        REGBENCH_INPUTS=str(inputs),
        REGBENCH_INPUTS_FORCE="1",
    )
    assert forced.returncode == 0 and "moved aside" in forced.stdout
    both = _launch(
        tmp_path,
        "--inputs",
        "--prepare",
        REGBENCH_INPUT=str(_sheet(tmp_path)),
        REGBENCH_INPUTS=str(inputs),
    )
    assert both.returncode == 2 and "two different steps" in both.stderr


def _preprocessed(inputs):
    """A toy checkpoint of the input step: one patient, a reference and two rounds."""
    res = inputs / "preprocess" / "results"
    rows = ["patient_id,id,preprocessed_image,is_reference,channels,pixel_size"]
    for name, ref in (("P1.r1", "true"), ("P1.r2", "false"), ("P1.r3", "false")):
        image = res / "P1" / "converted" / f"{name}.ome.tif"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.touch()
        rows.append(f"P1,{name},{image},{ref},DAPI|CD3,0.325")
    (res / "csv").mkdir(parents=True, exist_ok=True)
    (res / "csv" / "preprocessed.csv").write_text("\n".join(rows) + "\n")


def test_prepare_builds_the_rounds_from_the_input_step(tmp_path):
    inputs = tmp_path / "inputs"
    env = dict(
        REGBENCH_INPUTS=str(inputs),
        REGBENCH_STARDIST_MODEL="site_model",
        REGBENCH_SBATCH_GPU="--gres=gpu:1",
    )
    early = _launch(tmp_path, "--prepare", **env)
    assert early.returncode == 2 and "launch.sbatch --inputs" in early.stderr
    _preprocessed(inputs)
    done = _launch(tmp_path, "--prepare", **env)
    assert done.returncode == 0, done.stderr
    rounds = inputs / "rounds.csv"
    text = rounds.read_text()
    assert str(inputs / "outlines" / "P1" / "P1.r2.geojson") in text
    assert "arm_results" not in done.stdout + text
    jobs = [ln for ln in done.stdout.splitlines() if ln.startswith("sbatch ")]
    assert [re.search(r"--job-name=(\S+)", j).group(1) for j in jobs] == [
        "rb-nuclei",
        "rb-prepare-multiplex",
    ]
    assert "--array=0-2%" in jobs[0]  # three slides
    assert f"--manifest {rounds}" in jobs[1]
    # a rounds.csv named by hand is taken as it is (one written from the arms' results)
    old = tmp_path / "old_rounds.csv"
    old.write_text(text)
    mine = _launch(tmp_path, "--prepare", REGBENCH_MANIFEST=str(old), **env)
    assert mine.returncode == 0, mine.stderr
    assert f"--manifest {old}" in mine.stdout
    # REGBENCH_NUCLEI_SKIP is gone: without outlines there is no case
    skip = _launch(tmp_path, "--prepare", REGBENCH_NUCLEI_SKIP="1", **env)
    assert "rb-nuclei" in skip.stdout


NF_NAME = re.compile(r"^[a-z](?:[a-z\d]|[-_](?=[a-z\d])){0,79}$")  # Nextflow's rule


@pytest.mark.parametrize(
    "submission,want",
    [
        ("a..b", "rb-inputs-a-b-4242"),
        ("x-", "rb-inputs-x-4242"),
        ("--A__B--", "rb-inputs-a-b-4242"),
        ("", None),  # unset or empty: a dated id
        ("z" * 200, None),
    ],
)
def test_the_run_name_is_one_nextflow_takes_and_unique_per_job(
    tmp_path, submission, want
):
    bin_ = _fake_nextflow(tmp_path)
    done = _run(tmp_path, bin_, REGBENCH_SUBMISSION=submission)
    assert done.returncode == 0, done.stdout + done.stderr
    argv = (tmp_path / "nf.argv").read_text().splitlines()
    name = argv[argv.index("-name") + 1]
    assert NF_NAME.match(name), name
    assert name.endswith("-4242") and len(name) <= 80
    if want is not None:
        assert name == want
