# ruff: noqa: F811  (``site`` is a fixture imported from test_submit_cohort)
"""launch.sbatch --all: the whole benchmark from ONE sbatch, as a chain of short head jobs
(Task L1). Each head runs one stage (inputs, prepare, launch) and submits the next head,
this same script, with ``afterok`` on that stage's jobs and ``--kill-on-invalid-dep=yes``;
no head sleeps, nothing waits on the login node. And the GPU throttle,
REGBENCH_GPU_ARRAY_LIMIT. Stand-in ``sbatch``/``scontrol``/``squeue`` on PATH, as in
test_submit_cohort.py, whose helpers are reused."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from benchmarks.regbench import variants

from .test_compute_all import _sheet
from .test_submit_cohort import (  # noqa: F401  (site is a fixture)
    GPU,
    REPO,
    SLURM,
    _launch_command,
    _manifest,
    _sbatch_lines,
    _watch,
    code_of,
    jobs_of,
    launch,
    site,
)

LAUNCH = SLURM / "launch.sbatch"
CHAIN = SLURM / "chain.sh"
README = REPO / "benchmarks" / "regbench" / "README.md"
SCONTROL = (
    "#!/bin/bash\n"
    'echo "JobId=${2:-0} JobName=regbench-launch"\n'
    'echo "   BatchFlag=${FAKE_BATCHFLAG:-1} Reboot=0 ExitCode=0:0"\n'
    'echo "   UserId=u(1) GroupId=g(1) MCS_label=N/A"\n'
    'echo "   Priority=1 Nice=0 Account=lab_acct QOS=long_qos"\n'
    'echo "   RunTime=00:00:01 TimeLimit=06:00:00 TimeMin=N/A"\n'
    'echo "   Partition=cpu_part AllocNode:Sid=login:1"\n'
    f'echo "   Command={LAUNCH} --all"\n'
)


def _head(site, tmp_path, *args, state="none", env=None, dry=False, job="7"):
    """One head of the chain, run as the scheduler would run it. ``state``: what is on
    disk already (``none``, ``inputs`` done, ``partial``: inputs and some cases but no
    record that prepare completed, ``prepared``: inputs, rounds, cases and that record).
    Returns (completed process, sbatch log lines as (id, argv), root)."""
    tmp, cases, fake = site
    root = tmp_path / "root"
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    if state in ("inputs", "partial", "prepared"):
        csvdir = inputs / "preprocess" / "results" / "csv"
        csvdir.mkdir(parents=True, exist_ok=True)
        (csvdir / "preprocessed.csv").write_text("patient_id\nP1\n")
    if state in ("partial", "prepared") and not (root / "cases").exists():
        shutil.copytree(cases, root / "cases")
    if state == "prepared":
        (root / "cases" / "prepared-multiplex.ok").write_text("ok\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    (bin_ / "scontrol").write_text(SCONTROL)
    # the next head's sbatch: its environment is kept (--export=ALL hands it on)
    (bin_ / "sbatch").write_text(
        "#!/bin/bash\n"
        # the chain record as the stage's first job is queued (I2: written BEFORE it)
        f'case "$*" in *--job-name=rb-inputs*) cat "{root}"/logs/chain/*.tsv '
        f'> "{tmp_path}/during.tsv" 2>/dev/null ;; esac\n'
        f'case "$*" in *launch.sbatch*) env > "{tmp_path}/next.env"; '
        f'pwd > "{tmp_path}/next.pwd"\n'
        '  [ -z "${NEXT_REFUSE:-}" ] || { echo "sbatch: error: QOS limit" >&2; exit 1; }\n'
        '  if [ -n "${NEXT_WARN:-}" ]; then "$0.real" "$@"; '
        'echo "sbatch: warning: a site notice after the id"; exit 0; fi ;; esac\n'
        f'exec "{fake}/sbatch" "$@"\n'
    )
    (bin_ / "sbatch.real").write_text(f'#!/bin/bash\nexec "{fake}/sbatch" "$@"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    sheet = _sheet(tmp_path / "input.csv", ("P1", "P2"))
    more = {
        "REGBENCH_CASES": None,
        "REGBENCH_OUT": None,
        "REGBENCH_RUN_ORDER": None,
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "REGBENCH_ROOT": str(root),
        "REGBENCH_MANIFEST": str(_manifest(tmp_path)),
        "REGBENCH_STARDIST_MODEL": "site_model",
        "REGBENCH_INPUT": str(sheet),
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "SLURM_JOB_ID": job,
        **(env or {}),
    }
    argv, base, _ = _launch_command(site, args, more, dry, f"all-{tmp_path.name}")
    argv[1] = str(LAUNCH)
    base["PATH"] = f"{bin_}:{base['PATH']}"
    if dry:  # --dry-run is the head's own argument here, as a user gives it
        argv.remove("--dry-run")
        argv.insert(2, "--dry-run")
    done = subprocess.run(argv, env=base, capture_output=True, text=True, cwd=tmp_path)
    log = Path(base["SBATCH_LOG"])
    calls = []
    if log.exists():
        for line in log.read_text().splitlines():
            n, rest = line.split("\t", 1)
            calls.append((n, shlex.split(rest)))
    return done, calls, root


def _next_head(calls):
    heads = [(n, a) for n, a in calls if any(w.endswith("launch.sbatch") for w in a)]
    assert len(heads) <= 1, heads
    return heads[0] if heads else None


def _opt(argv, flag):
    vals = [w.split("=", 1)[1] for w in argv if w.startswith(flag + "=")]
    return vals[-1] if vals else None


def _named(root, ids_suffix, name):
    """The ids of the jobs named ``name*`` in the launcher's jobs.tsv."""
    jobs = (root / "out" / "logs" / "jobs.tsv").read_text().splitlines()
    return [ln.split("\t")[2] for ln in jobs if ln.split("\t")[1].startswith(name)]


def _replay(site, tmp_path, stage_env_file, *args, state, job):
    """The next head, started with the environment the head handed to sbatch."""
    carried = dict(
        line.split("=", 1)
        for line in stage_env_file.read_text().splitlines()
        if "=" in line
    )
    keep = {k: v for k, v in carried.items() if k.startswith("REGBENCH_CHAIN")}
    return _head(site, tmp_path, *args, state=state, env=keep, job=job)


# ── which stage the first head runs: read from disk ─────────────────────────────────
@pytest.mark.parametrize(
    "state,stage,launcher_job",
    [("none", "inputs", "rb-inputs"), ("inputs", "prepare", "rb-prepare-multiplex"),
     ("prepared", "launch", "rb-settle")],
)  # fmt: skip
def test_the_first_head_runs_the_first_stage_not_done(
    site, tmp_path, state, stage, launcher_job
):
    done, calls, root = _head(site, tmp_path, "--all", state=state)
    assert done.returncode == 0, done.stderr[-3000:] + done.stdout[-3000:]
    assert f"--all: this head runs the stage {stage}" in done.stdout
    names = [_opt(a, "--job-name") for _, a in calls]
    assert launcher_job in names
    nxt = _next_head(calls)
    if stage == "launch":
        assert nxt is None and "the chain ends" in done.stdout
    else:
        assert nxt is not None
        want = {"inputs": "prepare", "prepare": "launch"}[stage]
        assert nxt[1][-1] == f"--all-stage={want}" or f"--all-stage={want}" in nxt[1]


def test_the_next_head_has_the_chains_sbatch_flags(site, tmp_path):
    done, calls, root = _head(site, tmp_path, "--all", state="none")
    assert done.returncode == 0, done.stderr[-3000:]
    n, argv = _next_head(calls)
    inputs_id = _named(root, "inputs", "rb-inputs")
    assert len(inputs_id) == 1
    assert argv[0] == "--parsable"
    assert _opt(argv, "--dependency") == f"afterok:{inputs_id[0]}"
    assert "--export=ALL" in argv and "--kill-on-invalid-dep=yes" in argv
    assert _opt(argv, "--job-name") == "regbench-launch"
    assert _opt(argv, "--chdir") == str(tmp_path)
    assert _opt(argv, "--output") == f"{root}/logs/regbench-launch-%j.log"
    # the first head's own options, read from the scheduler (scontrol show job)
    assert _opt(argv, "--partition") == "cpu_part"
    assert _opt(argv, "--account") == "lab_acct"
    assert _opt(argv, "--qos") == "long_qos"
    assert _opt(argv, "--time") == "06:00:00"
    assert not any(w.startswith(("--wait", "--requeue")) for w in argv)
    assert "--no-requeue" in argv  # I1: a head is never requeued (JobRequeue=1)
    script = next(i for i, w in enumerate(argv) if w.endswith("launch.sbatch"))
    assert argv[script] == str(LAUNCH)  # the checkout's file, never the spool copy
    assert argv[script + 1 :] == ["--all-stage=prepare"]
    # submitted from the first head's submit folder, so its $here holds
    assert (tmp_path / "next.pwd").read_text().strip() == str(tmp_path)


def test_launcher_arguments_reach_the_launch_and_extra_sbatch_args_every_stage(
    site, tmp_path
):
    args = ("--all", "--runs", "stare", "--", "--constraint=big")
    done, calls, root = _head(site, tmp_path, *args, state="none")
    assert done.returncode == 0, done.stderr[-3000:]
    _, argv = _next_head(calls)
    script = next(i for i, w in enumerate(argv) if w.endswith("launch.sbatch"))
    assert argv[script + 1 :] == [
        "--all-stage=prepare",
        "--runs",
        "stare",
        "--",
        "--constraint=big",
    ]
    inputs_job = next(a for _, a in calls if _opt(a, "--job-name") == "rb-inputs")
    assert "--constraint=big" in inputs_job and "stare" not in inputs_job


def test_the_prepare_head_waits_for_the_prepare_jobs_only(site, tmp_path):
    done, calls, root = _head(site, tmp_path, "--all", state="inputs")
    assert done.returncode == 0, done.stderr[-3000:]
    _, argv = _next_head(calls)
    prep = _named(root, "prepare", "rb-prepare-")
    nuclei = _named(root, "prepare", "rb-nuclei")
    assert prep and nuclei
    assert _opt(argv, "--dependency") == "afterok:" + ":".join(prep)
    assert not set(nuclei) & set(
        _opt(argv, "--dependency")[len("afterok:") :].split(":")
    )


def test_a_stage_that_queues_nothing_ends_the_chain(site, tmp_path):
    # the input step needs the samplesheet: the launcher refuses, nothing is queued
    done, calls, root = _head(
        site, tmp_path, "--all", state="none", env={"REGBENCH_INPUT": None}
    )
    assert done.returncode != 0
    assert _next_head(calls) is None and not calls
    assert "queued nothing" in done.stderr and "no next head" in done.stderr


def test_the_chain_record_and_its_cumulative_scancel_line(site, tmp_path):
    done, calls, root = _head(site, tmp_path, "--all", state="none", job="7")
    assert done.returncode == 0, done.stderr[-3000:]
    first_next, _ = _next_head(calls)
    tsvs = list((root / "logs" / "chain").glob("*.tsv"))
    assert len(tsvs) == 1
    rows = [ln.split("\t") for ln in tsvs[0].read_text().splitlines()]
    assert rows[0] == ["stage", "head_job", "ids_file", "next_head", "state", "at"]
    assert rows[1][:2] == ["inputs", "7"] and rows[1][3] == first_next
    assert rows[1][4] == "done"
    assert rows[1][2].endswith(".inputs.ids") and Path(rows[1][2]).is_file()
    cancel = [ln for ln in done.stdout.splitlines() if "scancel" in ln]
    assert f'scancel 7 {first_next} $(cat "{rows[1][2]}" 2>/dev/null)' in cancel[-1]
    # the next head, with the environment the first head handed on
    again, calls2, _ = _replay(
        site, tmp_path, tmp_path / "next.env", "--all-stage=prepare", state="inputs",
        job=first_next,
    )  # fmt: skip
    assert again.returncode == 0, again.stderr[-3000:]
    assert "this head runs the stage prepare" in again.stdout
    second_next, _ = _next_head(calls2[len(calls) :])
    rows = [ln.split("\t") for ln in tsvs[0].read_text().splitlines()]
    assert [r[0] for r in rows[1:]] == ["inputs", "prepare"]
    assert rows[2][1] == first_next and rows[2][3] == second_next
    cancel = [ln for ln in again.stdout.splitlines() if "scancel" in ln]
    assert f"scancel 7 {first_next} {second_next} $(cat " in cancel[-1]
    assert f'$(cat "{rows[1][2]}" "{rows[2][2]}" 2>/dev/null)' in cancel[-1]


def test_a_later_head_runs_the_stage_it_is_given_never_the_disks(site, tmp_path):
    """--all-stage: the chain can never loop or skip, whatever is on disk."""
    env = {"REGBENCH_CHAIN": "c1", "REGBENCH_CHAIN_SUBMIT_DIR": str(tmp_path)}
    done, calls, _ = _head(
        site, tmp_path, "--all-stage=prepare", state="prepared", env=env
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert "this head runs the stage prepare" in done.stdout
    bad, _, _ = _head(site, tmp_path, "--all-stage=nope", env=env)
    assert bad.returncode == 2 and "--all-stage" in bad.stderr


def test_the_launch_stage_names_its_datasets_and_compute_all(site, tmp_path):
    done, calls, _ = _head(site, tmp_path, "--all", "--dry-run", state="prepared")
    assert done.returncode == 0, done.stderr[-3000:]
    assert not calls
    # no public dataset has cases here: each is said to be left out, and why
    for ds in ("mcmicro", "phenocycler", "rashid", "hyreco"):
        assert re.search(rf"left out: .*\b{ds}\b", done.stdout), ds
    assert '--datasets "multiplex"' in done.stdout
    assert "--compute-all" in done.stdout and "rb-score-compute" in done.stdout
    # its peak, from the settings and the run list, before the first sbatch
    peak = [ln for ln in done.stdout.splitlines() if ln.startswith("# peak")]
    assert peak and "CPUs" in peak[0] and "GPU" in peak[0] and "GB" in peak[0]
    first = done.stdout.index(peak[0])
    assert first < done.stdout.index("\nsbatch ")
    off, _, _ = _head(
        site, tmp_path, "--all", "--dry-run", state="prepared",
        env={"REGBENCH_INPUT": None},
    )  # fmt: skip
    assert off.returncode == 0, off.stderr[-3000:]
    assert "compute-all is skipped" in off.stdout and "REGBENCH_INPUT" in off.stdout
    assert "rb-score-compute" not in off.stdout


@pytest.mark.parametrize("state", ["none", "inputs", "prepared"])
def test_all_dry_run_submits_nothing_and_prints_the_plan(site, tmp_path, state):
    done, calls, root = _head(site, tmp_path, "--all", "--dry-run", state=state)
    assert done.returncode == 0, done.stderr[-3000:]
    assert not calls and not (root / "logs" / "chain").exists()
    for stage in ("inputs", "prepare", "launch"):
        assert f"--all: stage {stage}:" in done.stdout
    heads = [
        ln for ln in done.stdout.splitlines() if "launch.sbatch --all-stage=" in ln
    ]
    if state == "prepared":
        assert not heads
    else:
        assert len(heads) == 1
        line = heads[0][heads[0].index("sbatch ") :]
        for flag in ("--parsable", "--export=ALL", "--kill-on-invalid-dep=yes",
                     "--dependency=afterok:", "--job-name=regbench-launch",
                     f"--chdir={tmp_path}", "--partition=cpu_part"):  # fmt: skip
            assert flag in line, flag
    # the launcher's own dry run of the stage it would run now
    own = {
        "none": "rb-inputs",
        "inputs": "rb-prepare-multiplex",
        "prepared": "rb-settle",
    }
    assert any(f"--job-name={own[state]}" in ln for ln in _sbatch_lines(done.stdout))


def test_all_refuses_a_step_flag_of_its_own(site, tmp_path):
    done, calls, _ = _head(site, tmp_path, "--all", "--prepare")
    assert done.returncode == 2 and not calls and "--all" in done.stderr


def test_all_refuses_repeats(site, tmp_path):
    # the launcher repeats a submission before its --inputs/--prepare: under --all a
    # repeat would queue one input step per repeat, and the chain would see none queued
    done, calls, _ = _head(
        site, tmp_path, "--all", state="prepared", env={"REGBENCH_REPEAT": "2"}
    )
    assert done.returncode == 2 and not calls and "REGBENCH_REPEAT" in done.stderr
    once, calls, _ = _head(
        site, tmp_path, "--all", state="prepared", env={"REGBENCH_REPEAT": "1"}
    )
    assert once.returncode == 0 and calls


def test_the_plain_launch_is_unchanged():
    code = code_of(LAUNCH)
    assert 'exec bash "$launcher" "$@"' in code
    assert not re.search(r"(^|[;&|(]\s*)sbatch\s", code, re.M)
    # the one sbatch of --all (chain.sh, sourced by launch.sbatch) is the next head
    chain = code_of(CHAIN)
    calls = re.findall(r"(?:^|[;&|(]\s*)sbatch\s", chain, re.M)
    assert len(calls) == 1
    assert "--kill-on-invalid-dep=yes" in chain and '--dependency="afterok:' in chain
    assert "--wait" not in chain and "--requeue" not in chain


# ── the GPU throttle ──────────────────────────────────────────────────────────────
GPU_RUNS = {variants.run_name(r) for r in variants.runs_for(None) if r.gpu}


def _first_rungs(jobs):
    return {
        j.args[0]: j
        for j in jobs.values()
        if j.script == "run_rung.sbatch" and j.name.endswith("-16g")
    }


def test_a_gpu_runs_arrays_carry_the_gpu_limit(site):
    assert GPU_RUNS
    done, _, _ = launch(site, env={"REGBENCH_GPU_ARRAY_LIMIT": "3"}, log="gpulim")
    firsts = _first_rungs(jobs_of(done))
    for run, job in firsts.items():
        want = "%3" if run in GPU_RUNS else "%20"
        assert job.opt("--array").endswith(want), (run, job.opt("--array"))
    assert {r for r in firsts if r in GPU_RUNS}


def test_the_retry_plan_carries_its_chains_limit(site):
    done, _, out = launch(
        site,
        env={"REGBENCH_GPU_ARRAY_LIMIT": "3", "REGBENCH_SUBMISSION": "gpu-1",
             "REGBENCH_ALLOW_PATH_PYTHON": "1"},
        dry=False, log="gpuplan",
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-3000:]
    plans = out / "logs" / "retry" / "gpu-1"
    seen = set()
    for run_file in plans.glob("*/run"):
        run = run_file.read_text().split()[0]
        limit = (run_file.parent / "limit").read_text().strip()
        assert limit == ("3" if run in GPU_RUNS else "20"), run_file.parent.name
        seen.add(run in GPU_RUNS)
    assert seen == {True, False}


def test_the_watcher_applies_the_limit_its_plan_names(tmp_path):
    steps = [
        [
            "again",
            "rb-x-16g-again",
            "--mem=16G",
            "run_rung.sbatch",
            "@EARLIER@",
            "again",
        ]
    ]
    ends = {0: None, 1: "REGBENCH_END kind=run status=ok cause= whose="}
    done, sent, _ = _watch(
        tmp_path / "a", ends, steps, range_="0-1", plan_files={"limit": "3\n"}
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert sent and "--array=0%3" in sent[0].split()
    done, sent, _ = _watch(tmp_path / "b", ends, steps, range_="0-1")
    assert sent and "--array=0%20" in sent[0].split()  # an older plan: the array limit


def test_unset_the_gpu_limit_is_the_array_limit(site):
    sub = {"REGBENCH_SUBMISSION": "same-id"}
    plain, _, _ = launch(site, env=sub, log="gpu-unset")
    same, _, _ = launch(
        site, env={**sub, "REGBENCH_GPU_ARRAY_LIMIT": "20"}, log="gpu-20"
    )
    a = [ln.replace("out-gpu-unset", "X") for ln in plain.stdout.splitlines()]
    b = [ln.replace("out-gpu-20", "X") for ln in same.stdout.splitlines()]
    assert a == b


def test_the_nuclei_array_on_a_gpu_carries_the_gpu_limit(site, tmp_path):
    env = {
        "REGBENCH_MANIFEST": str(_manifest(tmp_path)),
        "REGBENCH_STARDIST_MODEL": "site_model",
        "REGBENCH_GPU_ARRAY_LIMIT": "3",
    }
    done, _, _ = launch(site, "--prepare", env=env, log="nucgpu")
    assert jobs_of(done)["rb-nuclei"].opt("--array") == "0-2%3"
    cpu, _, _ = launch(
        site, "--prepare", env={**env, "REGBENCH_STARDIST_CPU": "1"}, log="nuccpu"
    )
    assert jobs_of(cpu)["rb-nuclei"].opt("--array") == "0-2%20"


# ── every knob of launch.sbatch is explained in the README ────────────────────────
def _settings_knobs():
    text = LAUNCH.read_text()
    block = text.split("SETTINGS: edit here", 1)[1].split("\n# ----", 1)[0]
    knobs = set(re.findall(r':\s*"\$\{(REGBENCH_[A-Z0-9_]+):?=', block))
    knobs |= set(re.findall(r"^#\s*(REGBENCH_[A-Z0-9_]+)=", block, re.M))
    return knobs


def test_the_readme_explains_every_knob_launch_sbatch_sets():
    knobs = _settings_knobs()
    for must in ("REGBENCH_RUN_LANES", "REGBENCH_ARRAY_LIMIT", "REGBENCH_GPU_ARRAY_LIMIT",
                 "REGBENCH_STAGE_QUEUE_MAX", "REGBENCH_STAGE_ARRAY_MAX",
                 "REGBENCH_STAGE_FEEDERS", "REGBENCH_COMPUTE_LANES",
                 "REGBENCH_RETRY_WATCHERS", "REGBENCH_WAVES", "REGBENCH_REPEAT",
                 "REGBENCH_CPUS", "REGBENCH_TIME", "REGBENCH_SCORE_CPUS",
                 "REGBENCH_SCORE_MEM", "REGBENCH_INPUT", "REGBENCH_SITE_CONFIG"):  # fmt: skip
        assert must in knobs, must
    text = README.read_text()
    assert "## The whole benchmark in one command" in text
    section = text.split("## The whole benchmark in one command", 1)[1].split(
        "\n## ", 1
    )[0]
    for knob in sorted(knobs):
        assert re.search(rf"\b{knob}\b", section), knob
    for said in ("launch.sbatch --all", "squeue --me -n regbench-launch", "scancel",
                 "logs/chain/", "Task 32"):  # fmt: skip
        assert said in section, said


def test_the_settings_set_the_gpu_limit_to_four():
    assert re.search(r':\s*"\$\{REGBENCH_GPU_ARRAY_LIMIT:=4\}"', LAUNCH.read_text())


# ── fix round 1 ───────────────────────────────────────────────────────────────────
def test_no_head_is_ever_requeued():
    """I1: leaving --requeue out does not stop a requeue (slurm.conf JobRequeue
    defaults to 1); a requeued head would run its stage a second time."""
    head = LAUNCH.read_text().split("\nset -uo pipefail", 1)[0]
    assert "\n#SBATCH --no-requeue\n" in head
    assert "--no-requeue" in code_of(CHAIN)


def test_the_chain_record_is_written_before_the_stage_runs(site, tmp_path):
    """I2: a head killed mid-stage still left its ids file in the record, and the
    scancel line printed before the stage names it."""
    done, calls, root = _head(site, tmp_path, "--all", state="none")
    assert done.returncode == 0, done.stderr[-3000:]
    during = [
        ln.split("\t") for ln in (tmp_path / "during.tsv").read_text().splitlines()
    ]
    assert during[1][0] == "inputs" and during[1][4] == "running"
    assert during[1][2].endswith(".inputs.ids") and during[1][3] == "-"
    first_sbatch = (
        done.stdout.index("--job-name=rb-inputs")
        if "--job-name=rb-inputs" in done.stdout
        else None
    )
    lines = done.stdout.splitlines()
    early = [i for i, ln in enumerate(lines) if "scancel" in ln]
    stage_at = next(i for i, ln in enumerate(lines) if ln.startswith("rb-inputs\t"))
    assert early and early[0] < stage_at, first_sbatch
    assert f'"{during[1][2]}"' in lines[early[0]]
    final = [
        ln.split("\t")
        for ln in next((root / "logs" / "chain").glob("*.tsv")).read_text().splitlines()
    ]
    assert len(final) == 2 and final[1][4] == "done"
    # a stage that fails is recorded as failed, not left running
    bad, _, root2 = _head(site, tmp_path / "b", "--all", env={"REGBENCH_INPUT": None})
    rows = [
        ln.split("\t")
        for ln in next((root2 / "logs" / "chain").glob("*.tsv"))
        .read_text()
        .splitlines()
    ]
    assert bad.returncode != 0 and rows[1][4] == "failed"


def test_an_interrupted_prepare_is_run_again_on_resume(site, tmp_path):
    """I3: cases on disk without the record that prepare completed (a prepare that was
    cancelled or timed out half-way) is not a done prepare."""
    done, calls, _ = _head(site, tmp_path, "--all", "--dry-run", state="partial")
    assert done.returncode == 0, done.stderr[-3000:]
    assert "this head runs the stage prepare" in done.stdout
    assert "did not record completing" in done.stdout


def _prepare_task(tmp_path, ok, *index):
    cases = tmp_path / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    marker = cases / "prepared-multiplex.ok"
    marker.write_text("an earlier prepare\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    (bin_ / "python").write_text("#!/bin/bash\n" + ("exit 0\n" if ok else "exit 3\n"))
    (bin_ / "python").chmod(0o755)
    env = {
        **{
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("REGBENCH_", "SLURM_"))
        },
        "PATH": f"{bin_}:{os.environ['PATH']}",
        "REGBENCH_REPO": str(REPO),
        "REGBENCH_CASES": str(cases),
        "REGBENCH_OUT": str(tmp_path / "out"),
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        **({"SLURM_ARRAY_TASK_ID": index[0]} if index else {}),
    }
    done = subprocess.run(
        ["bash", str(SLURM / "prepare.sbatch"), "multiplex"], env=env,
        capture_output=True, text=True,
    )  # fmt: skip
    return done, marker


def test_prepare_records_its_completion_and_only_its_completion(tmp_path):
    done, marker = _prepare_task(tmp_path / "ok", True)
    assert done.returncode == 0, done.stderr
    assert marker.exists() and marker.read_text() != "an earlier prepare\n"
    failed, marker = _prepare_task(tmp_path / "bad", False)
    assert failed.returncode != 0 and not marker.exists()


def test_each_stage_has_a_submission_id_of_its_own(site, tmp_path):
    """M5: the nuclei chain's plan is the prepare stage's, never the launch's."""
    done, calls, root = _head(site, tmp_path, "--all", state="inputs")
    assert done.returncode == 0, done.stderr[-3000:]
    chain = next((root / "logs" / "chain").glob("*.tsv")).stem
    submitted = root / "out" / "logs" / "submitted"
    assert (submitted / f"{chain}-prepare.prepare.ids").is_file()
    assert (root / "out" / "logs" / "retry" / f"{chain}-prepare" / "rb-nuclei").is_dir()
    assert not (root / "out" / "logs" / "retry" / chain).exists()
    first, _, root1 = _head(site, tmp_path / "i", "--all", state="none")
    chain1 = next((root1 / "logs" / "chain").glob("*.tsv")).stem
    assert (
        root1 / "out" / "logs" / "submitted" / f"{chain1}-inputs.inputs.ids"
    ).is_file()


def test_a_notice_after_the_next_heads_id_is_not_a_refusal(site, tmp_path):
    """M6: the id is the line that is one, wherever the site prints its notices."""
    done, calls, root = _head(
        site, tmp_path, "--all", state="none", env={"NEXT_WARN": "1"}
    )
    assert done.returncode == 0, done.stderr[-3000:]
    nxt, _ = _next_head(calls)
    rows = next((root / "logs" / "chain").glob("*.tsv")).read_text().splitlines()
    assert rows[1].split("\t")[3] == nxt


def test_a_refused_next_head_prints_a_quoted_line_to_submit_it(site, tmp_path):
    done, calls, _ = _head(
        site, tmp_path, "--all", "--runs", "a b", state="none", env={"NEXT_REFUSE": "1"}
    )
    assert done.returncode != 0
    assert "--runs a\\ b" in done.stderr, done.stderr[-2000:]


def test_options_are_copied_only_from_a_batch_job(site, tmp_path):
    """M7: inside salloc/srun the job is not a head: nothing of it is copied."""
    done, calls, _ = _head(
        site, tmp_path, "--all", state="none", env={"FAKE_BATCHFLAG": "0"}
    )
    assert done.returncode == 0, done.stderr[-3000:]
    _, argv = _next_head(calls)
    assert not any(
        w.startswith(("--partition", "--account", "--qos", "--time")) for w in argv
    )
    assert "not a batch job" in done.stdout + done.stderr
