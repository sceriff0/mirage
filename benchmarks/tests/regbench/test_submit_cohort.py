"""submit_cohort.sh: the job graph of the cohort, read off its dry run.

The dry run prints every ``sbatch`` command with a job name in angle brackets where a job
id would be, and submits nothing. A stand-in ``sbatch`` on PATH records every call, so
"submits nothing" is observed, and one test runs the real path through it.

The fixture is two patients: P1 with three rounds, P2 with one (four cases).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.regbench import ladder, stages, variants
from benchmarks.regbench.cases import Case, write_json

REPO = Path(__file__).resolve().parents[3]
SLURM = REPO / "benchmarks" / "regbench" / "slurm"
SCRIPT = SLURM / "submit_cohort.sh"
GPU = "--gres=gpu:nvidia_h200:1"
RUNS = {variants.run_name(r): r for r in variants.active_runs()}
GATING = sorted({variants.gated_by(n) for n in RUNS})
# the runs whose registration is several jobs per round: their last job (the writer) has
# the four attempts of the pipeline's allocation for STITCH where the others have the
# memory ladder
STAGED = {n: variants.stages_of(r) for n, r in RUNS.items() if variants.stages_of(r)}
FEEDERS = 2  # tile submitters chained per staged run and dataset
N_JOBS = 425  # sbatch calls of the default submission on the fixture
# the last job of a staged round has the benchmark's ladder (its own work is in it)
STITCH_GB = ladder.RUNGS_GB
# the repository's one quote-aware "where does the comment start" (CLAUDE.md, rule 7): a
# guard that asks "does this command run" reads the comment-stripped text
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))
ci_actions = importlib.import_module("ci_actions")


def code_of(path):
    return ci_actions.strip_shell_comments(Path(path).read_text())


THREADS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


# every fixture slide: 2 x 3 tiles of 2048 px, so a round is six REG-TILE jobs
REF_HW = (4096, 6144)
TILES = 6


def _case(root, dataset, case_id, group, ref_hw=REF_HW):
    case = Case(
        dataset=dataset, case_id=case_id, group=group, ref_nuclear="", mov_nuclear="",
        ref_image="", mov_image="", modality="fluorescence", diagonal=1.0,
        ref_hw=list(ref_hw),
    )  # fmt: skip
    d = root / dataset / case_id
    d.mkdir(parents=True)
    case.dir = str(d)
    write_json(d / "case.json", asdict(case))


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """Prepared cases, and a bin directory whose `sbatch` only records its calls."""
    tmp = tmp_path_factory.mktemp("cohort")
    cases = tmp / "cases"
    for case_id, group in (
        ("P1_a", "P1"),
        ("P1_b", "P1"),
        ("P1_c", "P1"),
        ("P2_a", "P2"),
    ):
        _case(cases, "multiplex", case_id, group)
    _case(cases, "synthetic", "blob_grid_p0", "blob_grid")
    # prepared, and DORMANT: the default submission must leave it alone
    _case(cases, "semisynth", "P1_w0_grid_p0", "P1_w0_grid")
    fake = tmp / "bin"
    fake.mkdir()
    sbatch = fake / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        'n=$(( $(cat "$SBATCH_LOG.n" 2>/dev/null || echo 1000) + 1 ))\n'
        # SBATCH_REFUSE_AT: the scheduler takes no job from that id on
        'if [ -n "${SBATCH_REFUSE_AT:-}" ] && [ "$n" -ge "$SBATCH_REFUSE_AT" ]; then\n'
        '  echo "sbatch: error: QOSMaxSubmitJobPerUserLimit" >&2; exit 1\nfi\n'
        'echo "$n" > "$SBATCH_LOG.n"\n'
        'printf "%s\\t%s\\n" "$n" "$*" >> "$SBATCH_LOG"\n'
        # SBATCH_SLEEP_AT: the scheduler takes that job only after a while
        'if [ -n "${SBATCH_SLEEP_AT:-}" ] && [ "$n" -eq "$SBATCH_SLEEP_AT" ]; then\n'
        "  sleep 3\nfi\n"
        # SBATCH_BREAK_AT: from that id on the launcher's own job list cannot be
        # written (something other than the scheduler stops the launch)
        'if [ -n "${SBATCH_BREAK_AT:-}" ] && [ "$n" -ge "$SBATCH_BREAK_AT" ]; then\n'
        '  chmod a-w "$SBATCH_BREAK_FILE"\nfi\n'
        # SBATCH_NOTICE: the site's submit plugin says something before the id
        'if [ -n "${SBATCH_NOTICE:-}" ]; then\n'
        '  echo "sbatch: slurm_job_submit: Job assigned to GPU nodes: gpu" >&2\nfi\n'
        'echo "$n;cluster"\n'
    )
    sbatch.chmod(0o755)
    # no test reaches the network: a launch that fetches anything names its own curl
    curl = fake / "curl"
    curl.write_text(
        '#!/bin/bash\necho "curl $*" >> "$SBATCH_LOG.curl"\n'
        'echo "curl: the test site has no network" >&2\nexit 22\n'
    )
    curl.chmod(0o755)
    # the queue: the job ids in SQUEUE_FILE (none without it), as `-o "%F %A"` prints
    squeue = fake / "squeue"
    squeue.write_text(
        "#!/bin/bash\n"
        '[ -n "${SQUEUE_FAILS:-}" ] && exit 1\n'
        '[ -n "${SQUEUE_FILE:-}" ] && cat "$SQUEUE_FILE"\nexit 0\n'
    )
    squeue.chmod(0o755)
    return tmp, cases, fake


def launch(site, *args, env=None, dry=True, log="calls"):
    argv, base, out = _launch_command(site, args, env, dry, log)
    done = subprocess.run(argv, env=base, capture_output=True, text=True, cwd=REPO)
    return done, Path(base["SBATCH_LOG"]), out


def _launch_command(site, args, env, dry, log):
    tmp, cases, fake = site
    out = tmp / f"out-{log}"
    base = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("REGBENCH_") and k != "SBATCH_LOG"
    }
    base.update(
        PATH=f"{fake}:{Path(sys.executable).parent}:{base['PATH']}",
        REGBENCH_CASES=str(cases),
        REGBENCH_OUT=str(out),
        REGBENCH_SBATCH_GPU=GPU,
        # the run list's own order, which these tests are written against: the
        # launcher's default order has tests of its own
        REGBENCH_RUN_ORDER="",
        REGBENCH_SITE_CONFIG="none",  # never the real site overlay of a checkout
        # the stand-in "images" are this machine's python, which has not the methods'
        # packages: the preflight has its own tests
        REGBENCH_PREFLIGHT="0",
        SBATCH_LOG=str(tmp / f"{log}.tsv"),
    )
    base.update(env or {})
    for key in [k for k, v in base.items() if v is None]:
        del base[key]
    argv = ["bash", str(SCRIPT), *(["--dry-run"] if dry else []), *args]
    return argv, base, out


class Job:
    def __init__(self, line):
        self.line = line
        self.words = shlex.split(line)
        self.name = self.opt("--job-name")
        script = next(i for i, w in enumerate(self.words) if w.endswith(".sbatch"))
        self.script = Path(self.words[script]).name
        self.args = self.words[script + 1 :]

    def opt(self, flag):
        for w in self.words:
            if w.startswith(flag + "="):
                return w.split("=", 1)[1]
        return None

    @property
    def deps(self):
        """``{(kind, job name)}`` of the dependency list."""
        out = set()
        for item in (self.opt("--dependency") or "").split(","):
            if item:
                kind, *names = item.split(":")
                out |= {(kind, n) for n in names}
        return out


def jobs_of(done):
    assert done.returncode == 0, done.stderr + done.stdout
    return {j.name: j for j in map(Job, _sbatch_lines(done.stdout))}


def _sbatch_lines(text):
    return [ln for ln in text.splitlines() if ln.startswith("sbatch ")]


@pytest.fixture(scope="module")
def dry(site):
    done, log, _ = launch(site)
    return jobs_of(done), done, log


def rung_gb(run):
    return ladder.RUNGS_GB


def _last(run, suffix):
    return suffix + ("-last" if run in STAGED else "")


def rungs(jobs, run, dataset="multiplex", suffix=""):
    """The attempt of each rung (the `-again` arrays are `agains`). For a run that is
    several jobs per round: the rungs of its last job."""
    tail = _last(run, suffix)
    return [jobs[f"rb-{run}-{dataset}{tail}-{gb}g"] for gb in rung_gb(run)]


def agains(jobs, run, dataset="multiplex", suffix=""):
    tail = _last(run, suffix)
    return [jobs[f"rb-{run}-{dataset}{tail}-{gb}g-again"] for gb in rung_gb(run)]


def last_job(run, dataset="multiplex"):
    """The name of the last job a run submits on a dataset: what a run that reads its
    results, the next dataset, the next run of its lane and the settle job wait for."""
    return f"rb-{run}-{dataset}{_last(run, '')}-{rung_gb(run)[-1]}g-again"


def launch_jobs(spec):
    """How many jobs the launcher submits for one run on one dataset, after its gate:
    the ladder's eight, or for a run in several jobs per round four attempts of each
    stage that is an array over the rounds, its tile submitters, and the last job's
    eight."""
    names = variants.stages_of(spec)
    if not names:
        return 2 * len(ladder.RUNGS_GB)
    arrays = sum(st in names for st in ("prep", "coarse", "solve"))
    return 4 * arrays + FEEDERS + 2 * len(STITCH_GB)


def first_rung(spec, dataset):
    """The name of the first array of a run's ladder (its last job's, when staged)."""
    name = variants.run_name(spec)
    if variants.stages_of(spec):
        return f"rb-{name}-{dataset}-last-{STITCH_GB[0]}g"
    return f"rb-{name}-{dataset}-{ladder.RUNGS_GB[0]}g"


def chain(jobs, run, dataset="multiplex", suffix=""):
    """The ladder in the order it runs: attempt, again, attempt, again, ..."""
    both = zip(rungs(jobs, run, dataset, suffix), agains(jobs, run, dataset, suffix))
    return [j for pair in both for j in pair]


# ── nothing is submitted by a dry run ─────────────────────────────────────────
def test_a_dry_run_submits_nothing(dry):
    jobs, done, log = dry
    assert len(jobs) > 100 and not log.exists()
    assert f"# {len(jobs)} job(s) would be submitted; nothing was" in done.stdout
    # nothing waits on the login node, and no job waits on a job it submitted
    assert all("--wait" not in j.words for j in jobs.values())
    for path in [SCRIPT, *SLURM.glob("*.sbatch")]:
        assert "--wait" not in code_of(path), path.name
    # ONE kind of job submits jobs: the tile submitter
    assert {p.name for p in SLURM.glob("*.sbatch") if "sbatch " in code_of(p)} == {
        "feed_tiles.sbatch"
    }


def test_the_launch_job_is_the_launcher_run_from_a_copy(site, dry, tmp_path):
    """launch.sbatch, run as a scheduler runs it (a COPY, somewhere else), submits the
    same jobs as the launcher called directly; and says so where it finds no launcher."""
    import shutil

    copy = tmp_path / "slurm_script"
    shutil.copy(SLURM / "launch.sbatch", copy)
    # a site config of made-up values: the GPU request and the bound folders are READ
    # from it; comment lines are not
    config = tmp_path / "site.config"
    config.write_text(
        "params {\n    // gpu_type = 'never:9'\n    gpu_type   = 'card_x:1'\n}\n"
        "singularity {\n    runOptions = '--writable-tmpfs --bind /a/b --bind /c/d'\n}\n"
    )
    env = {
        "REGBENCH_SBATCH_GPU": None,
        "REGBENCH_CASES": None,
        "REGBENCH_OUT": None,
        "REGBENCH_SITE_CONFIG": str(config),
        # the stand-in "images" are this machine's python (the images have a test below)
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
    }
    argv, base, _ = _launch_command(site, (), env, True, "viajob")
    argv[1] = str(copy)
    root = tmp_path / "root"
    shutil.copytree(site[1], root / "cases")
    base = {**base, "SLURM_SUBMIT_DIR": str(REPO), "REGBENCH_ROOT": str(root)}
    base.pop("REGBENCH_RUN_ORDER")
    done = subprocess.run(argv, env=base, capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    jobs = jobs_of(done)
    assert sorted(jobs) == sorted(dry[0])
    # the settings inside the script reached the launcher: lanes, the site's GPU
    # request, the results under the one folder, the default order
    assert "GPU request --gres=gpu:card_x:1, 3 folder(s) bound" in done.stdout
    assert f"results {root}/out" in done.stdout
    assert any("--gres=gpu:card_x:1" in j.words for j in jobs.values())
    assert not any("never" in w for j in jobs.values() for w in j.words)
    assert _ladder_order(jobs)[0] != list(RUNS)[0]
    laned = [j for j in jobs.values() if j.script == "run_rung.sbatch"]
    assert max(len(j.deps) for j in laned) > max(
        len(j.deps) for j in dry[0].values() if j.script == "run_rung.sbatch"
    )
    # submitted from ANOTHER folder: the repository is found from the path given to
    # the scheduler, and the benchmark lives where the job was submitted from
    away = tmp_path / "away"
    shutil.copytree(site[1], away / "regbench" / "cases")
    (site[2] / "scontrol").write_text(
        f'#!/bin/bash\necho "   Command={SLURM / "launch.sbatch"} --prepare"\n'
    )
    (site[2] / "scontrol").chmod(0o755)
    far = {**base, "SLURM_SUBMIT_DIR": str(away), "SLURM_JOB_ID": "7"}
    far.pop("REGBENCH_ROOT")
    there = subprocess.run(argv, env=far, capture_output=True, text=True, cwd=away)
    (site[2] / "scontrol").unlink()
    assert there.returncode == 0, there.stderr
    assert f"results {away}/regbench/out" in there.stdout
    assert sorted(jobs_of(there)) == sorted(dry[0])
    lost = subprocess.run(
        argv,
        env={**base, "SLURM_SUBMIT_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert lost.returncode == 2 and "Nothing was submitted" in lost.stderr
    assert not _sbatch_lines(lost.stdout)
    head = (SLURM / "launch.sbatch").read_text()
    assert "\n#SBATCH --export=ALL\n" in head


def test_the_launch_job_finds_or_pulls_every_image_and_never_uses_path_python(
    site, tmp_path
):
    """The image folder is the site config's cacheDir. An image that is there is taken,
    one that is not is pulled into it under Nextflow's name, and a pull that fails
    stops the job before the launcher is started."""
    import shutil

    copy = tmp_path / "slurm_script"
    shutil.copy(SLURM / "launch.sbatch", copy)
    images = tmp_path / "site images"
    (images / "regbench").mkdir(parents=True)
    (images / "bolt3x-mirage-palom-1.0.0.img").write_text("x")
    (images / "regbench" / "ashlar.sif").write_text("x")
    config = tmp_path / "site.config"
    config.write_text(f"singularity {{\n    cacheDir = '{images}'\n}}\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    pulls = tmp_path / "pulls.txt"
    runtime = bin_ / "apptainer"
    runtime.write_text(
        f'#!/bin/bash\n[ "$1" = pull ] || exit 0\necho "$3" >> "{pulls}"\n'
        '[ -z "${PULL_FAILS:-}" ] || exit 1\necho x > "$2"\n'
    )
    runtime.chmod(0o755)
    env = {"REGBENCH_SITE_CONFIG": str(config), "REGBENCH_CASES": None}
    argv, base, _ = _launch_command(site, ("--no-such-option",), env, True, "img")
    argv[1] = str(copy)
    argv.remove("--dry-run")
    base = {
        **base,
        "PATH": f"{bin_}:{base['PATH']}",
        "SLURM_SUBMIT_DIR": str(REPO),
        "REGBENCH_ROOT": str(tmp_path / "root"),
    }
    failed = subprocess.run(
        argv, env={**base, "PULL_FAILS": "1"}, capture_output=True, text=True
    )
    assert failed.returncode == 2 and "could not be pulled" in failed.stderr
    assert "unknown option" not in failed.stdout + failed.stderr  # never started
    assert not list(images.glob("*.part"))
    pulls.unlink()
    done = subprocess.run(argv, env=base, capture_output=True, text=True)
    assert "unknown option --no-such-option" in done.stdout + done.stderr  # started
    pulled = pulls.read_text().split()
    assert len(pulled) == 4 and not any("palom" in r or "ashlar" in r for r in pulled)
    assert not any("stardist" in r for r in pulled)  # --prepare only
    assert f"PALOM -> {images}/bolt3x-mirage-palom-1.0.0.img" in done.stdout
    assert f"ASHLAR -> {images}/regbench/ashlar.sif" in done.stdout
    assert (images / "bolt3x-mirage-stare-regbench-1.0.0.img").exists()
    assert any(
        p.name.startswith("cdgatenbee-valis-wsi@sha256-") for p in images.iterdir()
    )
    # no runtime at all: said, and nothing started
    if not (shutil.which("apptainer") or shutil.which("singularity")):
        bare = {**base, "PATH": base["PATH"].split(":", 1)[1]}
        none = subprocess.run(argv, env=bare, capture_output=True, text=True)
        assert (
            none.returncode == 2 and "neither apptainer nor singularity" in none.stderr
        )


def test_an_image_without_a_container_runtime_is_said(site):
    """An image is named and neither runtime is on PATH: the launch stops WITH a line
    (it stopped without one: the lookup failed under `set -e`)."""
    import shutil

    if shutil.which("apptainer") or shutil.which("singularity"):
        pytest.skip("a container runtime is on PATH here")
    done, log, _ = launch(
        site, env={"REGBENCH_SIF_STARE_REGBENCH": "/images/x.sif"}, log="noruntime"
    )
    assert done.returncode != 0 and not _sbatch_lines(done.stdout) and not log.exists()
    assert "neither apptainer nor singularity is on PATH" in done.stderr


def test_the_environment_switch_is_the_same_dry_run(site, dry):
    done, log, _ = launch(site, env={"REGBENCH_DRY": "1"}, dry=False, log="envdry")
    same = [ln.replace("out-envdry", "out-calls") for ln in _sbatch_lines(done.stdout)]
    assert same == _sbatch_lines(dry[1].stdout)
    assert not log.exists()


# ── the run list is the harness's, never the script's ─────────────────────────
def test_the_script_names_no_method():
    """Every run comes from `python -m benchmarks.regbench runs`."""
    shell = [SCRIPT, *SLURM.glob("*.sbatch")]
    shell = [p for p in shell if p.name in (
        "submit_cohort.sh", "run_rung.sbatch", "selfcheck.sbatch", "gate.sbatch",
        "score_cohort.sbatch",
    )]  # fmt: skip
    assert len(shell) == 5
    names = {r.method for r in variants.RUNS} | set(RUNS)
    for path in shell:
        code = code_of(path)
        for name in names:
            assert not re.search(rf"\b{re.escape(name)}\b", code), (path.name, name)
    assert 'regbench_py "$REGBENCH_HEAD_IMAGE" runs > "$table"' in code_of(SCRIPT)


def test_every_run_of_the_list_is_submitted_and_nothing_else(dry):
    jobs, _, _ = dry
    laddered = {j.args[0] for j in jobs.values() if j.script == "run_rung.sbatch"}
    assert laddered == set(RUNS) and len(RUNS) == 18
    gated = {j.args[1] for j in jobs.values() if j.args[:1] == ["selfcheck"]}
    assert gated == set(RUNS)


def test_a_paused_run_is_not_submitted(site, monkeypatch, tmp_path):
    """The script submits what `runs` prints, and `runs` leaves a paused run out."""
    table = variants.runs_table()
    kept = [row for row in table if row[0] != "deeperhistreg_cpu"]
    stub = tmp_path / "benchmarks" / "regbench"
    stub.mkdir(parents=True)
    (tmp_path / "benchmarks" / "__init__.py").write_text("")
    (stub / "__init__.py").write_text("")
    lines = "\\n".join("\\t".join(r) for r in kept)
    alloc = "\\n".join(
        "\\t".join(
            map(
                str,
                (
                    st,
                    a,
                    *list(stages.allocation(st, a).values())[:2],
                    stages.allocation(st, a)["time"],
                ),
            )
        )
        for st in stages.ALLOCATION
        for a in (1, 2, 3, 4)
    )
    (stub / "__main__.py").write_text(
        "import sys\n"
        "a = sys.argv[1:]\n"
        "if a[:1] == ['runs'] and '--stage-allocation' in a:\n"
        f"    print('{alloc}')\n"
        "elif a[:1] == ['stages']:\n"
        f"    print('4 {4 * TILES}')\n"
        "elif a[:1] == ['runs']:\n"
        f"    print('{variants.SCORER_IMAGE}' if '--scorer-image' in a else '{lines}')\n"
        "elif a[:1] == ['list']:\n"
        "    print('4 2 3' if 'multiplex' in a else '0 0 0')\n"
    )
    done, _, _ = launch(site, env={"REGBENCH_REPO": str(tmp_path)}, log="paused")
    ran = {j.args[0] for j in jobs_of(done).values() if j.script == "run_rung.sbatch"}
    assert "deeperhistreg_cpu" not in ran and ran == set(RUNS) - {"deeperhistreg_cpu"}


# ── the ladder ────────────────────────────────────────────────────────────────
def test_every_run_has_four_chained_arrays_at_four_cpus(dry):
    jobs, _, _ = dry
    for run in RUNS:
        got = rungs(jobs, run)
        gbs = rung_gb(run)
        # the benchmark's ladder for every run: the last job of a staged round holds
        # the benchmark's own work beside the method's writer
        assert gbs == (16, 64, 256, 512), run
        assert [j.opt("--mem") for j in got] == [f"{gb}G" for gb in gbs], run
        assert [j.opt("--cpus-per-task") for j in got] == ["4"] * 4, run
        # attempt and memory reach the task: REGBENCH_ATTEMPT, REGBENCH_MEM_GB
        assert [j.args[7:9] for j in got] == [
            [str(i), str(gb)] for i, gb in enumerate(gbs, 1)
        ], run
        # only the last job of a run in several jobs is told to collect them
        assert {j.args[13] for j in got} == {"jobs" if run in STAGED else "-"}, run
        assert [j.args[11] for j in got] == ["first", "climb", "climb", "climb"], run
        assert got[0].args[10] == "-"
        assert all("--wait" not in j.words for j in got)


def test_every_rung_is_tried_again_once_at_the_same_memory(dry):
    """Infrastructure is not the method: a unit the scheduler lost gets the SAME rung
    once more, in an array of its own right behind the attempt."""
    jobs, _, _ = dry
    assert ladder.RETRIES_PER_RUNG == 1
    for run in RUNS:
        first, again = rungs(jobs, run), agains(jobs, run)
        for a, b in zip(first, again):
            assert b.opt("--mem") == a.opt("--mem") and b.args[7:9] == a.args[7:9]
            assert b.opt("--array") == a.opt("--array") and b.args[11] == "again"
            assert b.opt("--time") == a.opt("--time") == a.args[12] == b.args[12]
        seq = chain(jobs, run)
        for i, (lower, upper) in enumerate(zip(seq, seq[1:]), 1):
            assert ("afterany", f"<{lower.name}>") in upper.deps
            # every earlier job of the ladder, in order: the task asks the scheduler
            # what became of its unit in each
            assert upper.args[10] == ",".join(f"<{j.name}>" for j in seq[:i])


def test_the_rung_task_passes_the_rule_and_the_allocation_on():
    code = code_of(SLURM / "run_rung.sbatch")
    assert 'export REGBENCH_ATTEMPT="$attempt" REGBENCH_MEM_GB="$mem_gb"' in code
    assert 'export REGBENCH_RETRY="$retry" REGBENCH_TIME_LIMIT="$limit"' in code
    assert re.search(
        r'if \[ "\$task" != first \]; then\s+args\+=\(--ladder "\$task"\)', code
    )
    assert "-o JobID,State,ElapsedRaw,TotalCPU" in code
    assert 'look "$ids" && acct_args+=(--accounting "$acct")' in code
    assert 'regbench_py "$image" "${args[@]}" ${acct_args[@]+"${acct_args[@]}"}' in code
    assert '[ "${REGBENCH_FORCE:-0}" = 1 ] && args+=(--force)' in code
    env = code_of(SLURM / "env.sh")
    for name in ("REGBENCH_RETRY", "REGBENCH_SUBMISSION", "REGBENCH_TIME_LIMIT"):
        assert name in env, f"{name} must reach the inside of an image"


def test_the_default_is_multiplex_first_then_synthetic_and_nothing_dormant(dry):
    """Ruling R3.14. The site has semisynth cases prepared; they are not submitted."""
    jobs, done, _ = dry
    datasets = {j.args[6] for j in jobs.values() if j.script == "run_rung.sbatch"}
    assert datasets == {"multiplex", "synthetic"}
    assert not [n for n in jobs if "semisynth" in n or "hyreco" in n]
    for run in RUNS:
        last = agains(jobs, run)[-1].name
        for job in chain(jobs, run, "synthetic"):
            assert ("afterany", f"<{last}>") in job.deps, (run, job.name)
        waits = {n for j in chain(jobs, run) for _, n in j.deps}
        assert not [n for n in waits if "synthetic" in n], run
    order = [Job(ln).args[6] for ln in _sbatch_lines(done.stdout) if "run_rung" in ln]
    assert order.index("synthetic") > max(
        i for i, d in enumerate(order) if d == "multiplex"
    )


def test_a_dormant_dataset_is_submitted_only_when_named(site):
    done, _, _ = launch(site, "--datasets", "synthetic semisynth multiplex", log="dorm")
    jobs = jobs_of(done)
    assert "# semisynth is DORMANT: submitted because it was named." in done.stdout
    order = []
    for ln in _sbatch_lines(done.stdout):
        job = Job(ln)
        if job.script == "run_rung.sbatch" and job.args[6] not in order:
            order.append(job.args[6])
    assert order == ["multiplex", "synthetic", "semisynth"]  # the cohort first
    first = rungs(jobs, "stare", "semisynth")[0]
    assert ("afterany", f"<{last_job('stare', 'synthetic')}>") in first.deps
    code = code_of(SCRIPT)
    assert 'DEFAULT_DATASETS="multiplex synthetic"' in code
    assert 'DORMANT_DATASETS="semisynth hyreco_a hyreco_b"' in code


def test_no_job_asks_for_more_than_seven_days(site):
    done, _, _ = launch(site, env={"REGBENCH_TIME": "3-00:00:00"}, log="long")
    jobs = jobs_of(done)
    for job in jobs.values():
        days, clock = job.opt("--time").split("-")
        h, m, s = (int(v) for v in clock.split(":"))
        assert int(days) * 86400 + h * 3600 + m * 60 + s <= 7 * 86400, job.name
    # three rounds at three days each would be nine: capped at the site's maximum
    assert rungs(jobs, "valis_mirage", suffix="-first")[0].opt("--time") == "7-00:00:00"
    assert rungs(jobs, "palom")[0].opt("--time") == "3-00:00:00"
    # the last job of a staged round: one round's time plus the writer's 40 h
    assert rungs(jobs, "stare")[0].opt("--time") == "4-16:00:00"


def test_patient_runs_get_one_rounds_time_for_every_round(dry):
    jobs, _, _ = dry
    for run, spec in RUNS.items():
        per_patient = variants.scope(spec.method) == "group"
        job = rungs(jobs, run)[0]
        # P1 has three rounds: 3 x 24 h for a run that registers a patient at once
        want = "3-00:00:00" if per_patient else "1-00:00:00"
        if run in STAGED:
            # the round's last job: one round's time for the benchmark's own work,
            # and the four attempts the pipeline gives the writer (4 + 8 + 12 + 16 h)
            hours = sum(stages.allocation("stitch", a)["hours"] for a in (1, 2, 3, 4))
            assert hours == 40
            want = "2-16:00:00"
        assert job.opt("--time") == want, run


def test_pair_runs_are_arrays_over_cases_and_patient_runs_over_patients(dry):
    jobs, _, _ = dry
    for run, spec in RUNS.items():
        per_patient = variants.scope(spec.method) == "group"
        gated = variants.realcheck_gate(run) is not None
        first = rungs(jobs, run)[0].opt("--array")
        want = {(True, False): "0-1", (True, True): "1-1", (False, False): "0-3"}
        assert first == want[(per_patient, gated)] + "%20", run
    assert rungs(jobs, "stare", "synthetic")[0].opt("--array") == "0-0%20"


def test_arrays_are_throttled(site):
    done, _, _ = launch(site, env={"REGBENCH_ARRAY_LIMIT": "7"}, log="throttle")
    arrays = [j for j in jobs_of(done).values() if j.script == "run_rung.sbatch"]
    assert arrays and all(j.opt("--array").endswith("%7") for j in arrays)


# ── the selfcheck, and the gate ───────────────────────────────────────────────
def test_a_limit_on_runs_in_flight_chains_the_runs_onto_lanes(site, dry):
    """REGBENCH_RUN_LANES=N: the runs are dealt onto N lanes in the order of the list and
    a run starts only after the run before it on its lane has ended, so at most N are in
    flight. Without the variable no run waits for an unrelated one."""
    lanes = 4

    def waits(job):
        return {name.strip("<>") for _, name in job.deps}

    def head(jobs, run, dataset="multiplex"):
        if run in STAGED:  # the first job of its first stage
            return jobs[f"rb-{run}-{dataset}-{STAGED[run][0]}-a1"]
        return jobs.get(f"rb-{run}-{dataset}-first-16g") or rungs(jobs, run, dataset)[0]

    done, _, _ = launch(site, env={"REGBENCH_RUN_LANES": str(lanes)}, log="lanes")
    jobs = jobs_of(done)
    runs = [r for r in variants.RUNS if r.active]
    names = [variants.run_name(r) for r in runs]
    last = {n: chain(jobs, n)[-1].name for n in names}
    for k, name in enumerate(names):
        if k >= lanes:
            assert last[names[k - lanes]] in waits(head(jobs, name)), name
    # the next dataset goes on down the same lanes
    on_first_lane = [n for k, n in enumerate(names) if k % lanes == 0]
    assert last[on_first_lane[-1]] in waits(head(jobs, names[0], "synthetic"))
    # off: a run that reads no other run's results waits for no other run's arrays
    plain, _, _ = dry
    arrays = {j.name for n in names for j in chain(plain, n)}
    for r in runs:
        if r.after is None:
            own = {j.name for j in chain(plain, variants.run_name(r))}
            assert not (waits(head(plain, variants.run_name(r))) & (arrays - own)), r


def test_every_gating_run_has_a_selfcheck_that_climbs_in_memory(dry):
    jobs, _, _ = dry
    checks = [j for j in jobs.values() if j.script == "selfcheck.sbatch"]
    assert {j.args[0] for j in checks} == set(GATING) and len(GATING) == 16
    for run in GATING:
        got = [jobs[f"rb-sc-{run}-{gb}g"] for gb in ladder.SELFCHECK_RUNGS_GB]
        assert [j.opt("--mem") for j in got] == ["64G", "256G", "512G"], run
        assert [j.opt("--cpus-per-task") for j in got] == ["4"] * 3, run
        assert [j.args[8:10] for j in got] == [["1", "64"], ["2", "256"], ["3", "512"]]
        for lower, upper in zip(got, got[1:]):
            assert ("afterany", f"<{lower.name}>") in upper.deps
    code = code_of(SLURM / "selfcheck.sbatch")
    assert re.search(
        r'if \[ "\$attempt" -gt 1 \]; then\s+args\+=\(--retry-memory\)', code
    )


def test_each_selfcheck_is_given_its_runs_own_options_and_mask(dry):
    jobs, _, _ = dry
    for run in GATING:
        spec, job = RUNS[run], jobs[f"rb-sc-{run}-64g"]
        name, method, label, opts, image, gpu, mask, anchor = job.args[:8]
        assert (name, method, label) == (run, spec.method, spec.label or "-")
        assert opts == (",".join(variants.opt_args(spec)) or "-")
        assert (image, gpu) == (spec.image, "1" if spec.gpu else "0")
        assert mask == (variants.selfcheck_mask(spec.method) or "-")
        assert anchor == (spec.after or "-")
    assert jobs["rb-sc-ashlar-64g"].args[6] == "overlap"
    assert jobs["rb-sc-palom-64g"].args[6] == "seams"


def test_an_anchored_selfcheck_waits_for_the_anchors_and_reads_its_work(dry):
    jobs, _, _ = dry
    job = jobs["rb-sc-affine_ashlar-64g"]
    assert job.args[7] == "stare"
    assert ("afterok", "<rb-sc-stare-512g>") in job.deps
    code = code_of(SLURM / "selfcheck.sbatch")
    assert '[ "$anchor" != - ] && args+=(--m0-from "$root/$anchor")' in code


def test_every_run_waits_for_its_own_gate_and_the_gate_for_the_selfcheck(dry):
    jobs, _, _ = dry
    for run in RUNS:
        gate = jobs[f"rb-gate-{run}"]
        gating = variants.gated_by(run)
        assert gate.script == "gate.sbatch"
        assert gate.args == ["selfcheck", run, gating, "0"]
        assert gate.deps == {("afterany", f"<rb-sc-{gating}-512g>")}
        mine = [
            j
            for j in jobs.values()
            if j.script == "run_rung.sbatch" and j.args[0] == run
        ]
        assert len(mine) >= 8  # two datasets, four rungs
        for job in mine:  # every rung, so a blocked run is blocked at each of them
            assert ("afterok", f"<rb-gate-{run}>") in job.deps, job.name
            others = {n for k, n in job.deps if "gate" in n and k == "afterok"}
            assert others <= {f"<rb-gate-{run}>", f"<rb-realgate-{run}>"}, job.name
    # the two clean ASHLAR rows stand behind the noisy rows' selfchecks
    assert jobs["rb-gate-ashlar_clean"].args[2] == "ashlar"
    assert jobs["rb-gate-affine_ashlar_clean"].args[2] == "affine_ashlar"


def test_a_failed_gate_blocks_that_run_only(dry):
    """No job of one run waits `afterok` on another run's gate."""
    jobs, _, _ = dry
    for job in jobs.values():
        if job.script != "run_rung.sbatch":
            continue
        for kind, name in job.deps:
            if kind == "afterok":
                assert name.strip("<>").split("gate-")[-1] == job.args[0], job.line
    assert '[ "$allowed" = 1 ] && args+=(--allow-unchecked)' in code_of(
        SLURM / "gate.sbatch"
    )
    # and the task itself refuses, should the scheduler let it through
    assert 'if [ "$blocked" != 0 ]; then' in code_of(SLURM / "run_rung.sbatch")


def test_an_allowed_run_is_told_so_and_no_other(site):
    done, _, _ = launch(
        site, env={"REGBENCH_ALLOW_UNCHECKED": "valis120 palom"}, log="allow"
    )
    jobs = jobs_of(done)
    allowed = {
        j.args[1]
        for j in jobs.values()
        if j.args[:1] == ["selfcheck"] and j.args[3] == "1"
    }
    assert allowed == {"valis120", "palom"}
    assert "allowed to start unchecked: valis120 palom" in done.stdout
    bad, _, _ = launch(site, env={"REGBENCH_ALLOW_UNCHECKED": "valis"}, log="allowbad")
    assert bad.returncode == 2 and "no run named 'valis'" in bad.stderr
    assert not _sbatch_lines(bad.stdout)


def test_every_dependent_job_is_killed_on_an_invalid_dependency(dry):
    jobs, _, _ = dry
    waiting = [j for j in jobs.values() if j.opt("--dependency")]
    assert len(waiting) > 100
    assert all(j.opt("--kill-on-invalid-dep") == "yes" for j in waiting)


def test_no_job_is_ever_requeued(dry):
    """A requeued task would find only the placeholder of its first start and record
    the method's death: a lost task ends, and the ladder's `again` task is the retry."""
    jobs, done, _ = dry
    assert len(jobs) == N_JOBS
    assert all("--no-requeue" in j.words for j in jobs.values())
    assert "--requeue" not in code_of(SCRIPT)
    assert "--requeue" not in code_of(SLURM / "feed_tiles.sbatch")


# ── order ─────────────────────────────────────────────────────────────────────
def test_both_anchored_rows_wait_for_the_run_they_read(dry):
    jobs, _, _ = dry
    for run in ("affine_ashlar", "affine_ashlar_clean"):
        assert RUNS[run].after == "stare"
        for dataset in ("multiplex", "synthetic"):
            first = rungs(jobs, run, dataset)[0]
            assert ("afterany", f"<{last_job('stare', dataset)}>") in first.deps, run
            # the anchor's LAST job, the writer's last rung: none of its earlier ones
            assert {n for _, n in first.deps if "rb-stare-" in n} == {
                f"<{last_job('stare', dataset)}>"
            }
    assert not any("stare-" in n for _, n in rungs(jobs, "ashlar")[0].deps)


def test_the_lines_come_in_the_order_the_work_does(dry):
    _, done, _ = dry
    order = [Job(ln).name for ln in _sbatch_lines(done.stdout)]
    last_selfcheck = max(i for i, n in enumerate(order) if n.startswith("rb-sc-"))
    first_gate = min(i for i, n in enumerate(order) if n.startswith("rb-gate-"))
    last_gate = max(i for i, n in enumerate(order) if n.startswith("rb-gate-"))
    first_array = min(
        i for i, n in enumerate(order) if re.search(r"-\d+g$", n) and "-sc-" not in n
    )
    assert last_selfcheck < first_gate <= last_gate < first_array
    assert order[-2:] == ["rb-tissue", "rb-score"]
    # settle, then the case arrays, then tissue, then the tables
    assert order[-5:-2] == [
        "rb-settle",
        "rb-score-case-multiplex",
        "rb-score-case-synthetic",
    ]
    # a job never waits for one printed after it
    seen = set()
    for line in _sbatch_lines(done.stdout):
        job = Job(line)
        assert {n.strip("<>") for _, n in job.deps} <= seen, job.name
        seen.add(job.name)


def test_the_killed_results_are_settled_before_any_case_is_scored(dry):
    """`regbench settle` rewrites the run.json of a result the scheduler ended. It is a
    job of its own that waits for every registration and every gate, and every scoring
    job waits for it: no case is scored from a result that is about to change."""
    jobs, _, _ = dry
    settle = jobs["rb-settle"]
    assert settle.script == "score_cohort.sbatch" and settle.args == ["settle"]
    assert (settle.opt("--cpus-per-task"), settle.opt("--mem")) == ("1", "4G")
    assert {k for k, _ in settle.deps} == {"afterany"}  # whatever became of them
    waited = {n.strip("<>") for _, n in settle.deps}
    for run in RUNS:
        assert f"rb-gate-{run}" in waited
        for dataset in ("multiplex", "synthetic"):
            assert last_job(run, dataset) in waited, (run, dataset)
    # every array of the per-case scoring waits for it, and for nothing else directly
    arrays = [j for n, j in jobs.items() if n.startswith("rb-score-case-")]
    assert len(arrays) == 2
    for job in arrays:
        assert job.deps == {("afterany", "<rb-settle>")}, job.name
    # ... so the tissue job and the scorer come after it too, through the arrays
    after = {"rb-settle"} | {j.name for j in arrays}
    for name in ("rb-tissue", "rb-score"):
        assert {n.strip("<>") for _, n in jobs[name].deps} <= after | {"rb-tissue"}
        assert {n.strip("<>") for _, n in jobs[name].deps} >= {j.name for j in arrays}
    # no job that registers, checks or gates anything waits for the settle job
    for name, job in jobs.items():
        if name not in ("rb-tissue", "rb-score") and job not in arrays:
            assert ("afterany", "<rb-settle>") not in job.deps, name


def test_the_settle_step_of_the_batch_script():
    code = code_of(SLURM / "score_cohort.sbatch")
    # ONE place asks the scheduler and settles; the settle job and the scorer call it
    assert (
        code.count("settle --out") == 1
        and code.count("sacct -n -P -j") == 1
        and "sacct -n -X" not in code
    )
    assert code.count("settle_killed") == 3  # defined, the settle job, the scorer
    job = code[
        code.index('if [ "$kind" = settle ]') : code.index('if [ "$kind" = case ]')
    ]
    assert "settle_killed" in job and "exit 0" in job  # never blocks the scoring
    assert "cause=settle_unfinished whose=benchmark" in job
    # it says for which submission it was done, and only after it was
    done = code[code.index("settle_killed() {") : code.index("already_settled() {")]
    assert done.index("settle --out") < done.index('> "$settled"')
    assert 'echo "${REGBENCH_SUBMISSION:-}" > "$settled"' in done
    # the scorer settles ONLY when the job left no word of it, and before the tables
    scorer = code[code.index("if already_settled; then") :]
    assert scorer.index("settle_killed") < scorer.index(
        'regbench_py "$REGBENCH_SCORER_IMAGE" "${args[@]}"'
    )
    assert scorer.index("else") < scorer.index("settle_killed")
    # a case task never settles: it reads what the settle job left
    case = code[
        code.index('if [ "$kind" = case ]') : code.index('if [ "$kind" = tissue ]')
    ]
    assert "settle" not in case


def test_tissue_then_score_run_last_in_the_scorers_image(dry):
    jobs, _, _ = dry
    tissue, scorer = jobs["rb-tissue"], jobs["rb-score"]
    assert tissue.args == ["tissue"] and scorer.args == ["score"]
    assert scorer.deps == {
        ("afterany", "<rb-tissue>"),
        ("afterany", "<rb-score-case-multiplex>"),
        ("afterany", "<rb-score-case-synthetic>"),
    }
    assert tissue.deps == {
        ("afterany", "<rb-score-case-multiplex>"),
        ("afterany", "<rb-score-case-synthetic>"),
    }
    code = code_of(SLURM / "score_cohort.sbatch")
    assert code.count('regbench_py "$REGBENCH_SCORER_IMAGE"') == 5
    # only what this submission ran or reused is scored
    assert 'shared+=(--submission "$REGBENCH_SUBMISSION")' in code
    assert 'shared+=(--drop "$v")' in code and '--gates "$gates"' in code
    assert "pipeline" not in code.lower()  # no cost is read from another run


# ── one scoring task per case, then the tables ────────────────────────────────
def test_every_case_is_scored_by_a_task_of_its_own_before_the_tables(dry):
    """One array per dataset, a task per case; it waits for the settle job (and so for
    everything before it), and the scorer waits for it whatever becomes of it."""
    jobs, _, _ = dry
    tissue, scorer = jobs["rb-tissue"], jobs["rb-score"]
    sizes = {"multiplex": 4, "synthetic": 1}
    arrays = {n: j for n, j in jobs.items() if n.startswith("rb-score-case-")}
    assert set(arrays) == {f"rb-score-case-{ds}" for ds in sizes}
    for ds, n in sizes.items():
        job = arrays[f"rb-score-case-{ds}"]
        assert job.script == "score_cohort.sbatch" and job.args == ["case", ds]
        assert job.opt("--array") == f"0-{n - 1}%20"
        assert (job.opt("--cpus-per-task"), job.opt("--mem")) == ("8", "64G")
        assert job.opt("--time") == "1-00:00:00"
        assert job.deps == {("afterany", "<rb-settle>")}
        # the tissue job comes after the arrays (each task makes its case's pieces; two
        # jobs never write the same ones)
        assert tissue.deps == {("afterany", f"<{n}>") for n in arrays}
        assert "%A_%a.log" in job.opt("--output")
        assert ("afterany", f"<{job.name}>") in scorer.deps
    # the tables keep their allocation: the cases were scored before them
    assert (scorer.opt("--cpus-per-task"), scorer.opt("--time")) == ("4", "1-00:00:00")
    # the default submission: the settle job, an array per dataset, tissue, the scorer
    # a run in several jobs per round has, per dataset, 4 attempts of COARSE and of SOLVE,
    # its two tile submitters (the tile jobs are theirs to submit), and the last job's 8
    # (and 4 of its preprocessing) where another run has the ladder's 8
    more = sum(4 + FEEDERS + 4 + (4 if "prep" in st else 0) for st in STAGED.values())
    assert more == 10 + 14 and len(STAGED) == 2
    assert (
        len(jobs)
        == 18 + 48 + 18 * 2 * 8 + 2 * (8 + 1) + 1 + len(sizes) + 2 + 2 * more
        == N_JOBS
    )


def test_the_case_arrays_have_their_own_allocation_and_are_throttled(site):
    env = {
        "REGBENCH_SCORE_CPUS": "16",
        "REGBENCH_SCORE_MEM": "128G",
        "REGBENCH_SCORE_TIME": "9-00:00:00",  # above the site's longest job
        "REGBENCH_ARRAY_LIMIT": "3",
    }
    jobs = jobs_of(launch(site, "--runs", "stare", env=env, log="score-alloc")[0])
    job = jobs["rb-score-case-multiplex"]
    assert job.opt("--array") == "0-3%3" and job.opt("--cpus-per-task") == "16"
    assert job.opt("--mem") == "128G" and job.opt("--time") == "7-00:00:00"
    # the memory is the tissue and score jobs' too; their CPUs and day are not touched
    for name in ("rb-tissue", "rb-score"):
        assert jobs[name].opt("--mem") == "128G"
        assert jobs[name].opt("--cpus-per-task") == "4"
        assert jobs[name].opt("--time") == "1-00:00:00"


def test_the_case_task_and_the_scorer_are_told_the_same_things():
    """A case's record is reused only by a scorer given the same submission and the same
    dropped rows: both read them from ONE place in the batch script."""
    code = code_of(SLURM / "score_cohort.sbatch")
    assert code.count("shared_args\n") == 2  # the case task, and the scorer
    case = code[
        code.index('if [ "$kind" = case ]') : code.index('if [ "$kind" = tissue ]')
    ]
    assert "score-case" in case and '--index "$SLURM_ARRAY_TASK_ID"' in case
    assert '--dataset "$external"' in case and '--workers "$workers"' in case
    assert '${shared[@]+"${shared[@]}"}' in case
    scorer = code[code.index("args=(score ") :]
    assert '${shared[@]+"${shared[@]}"}' in scorer.split("\n\n")[0]
    # the task says how it ended, and a task that died says whose that is
    assert "cause=score_case_unfinished whose=benchmark" in case
    assert 'exit "$code"' in case
    # the case's tissue pieces first, and a failure there never stops its scoring
    tissue = case.index('tissue --cases "$REGBENCH_CASES"')
    assert tissue < case.index("|| echo") < case.index("score-case")
    assert '--index "$SLURM_ARRAY_TASK_ID"' in case[tissue : case.index("|| echo")]


def _records_site(site, tmp_path, name, ids=("20261011T220000-77",), last=None):
    """A results folder as an earlier submission left it: selfcheck records carrying
    its id, and (from this version on) logs/submission."""
    out = site[0] / f"out-{name}"
    for i, sub in enumerate(ids):
        d = out / "logs" / "selfcheck" / f"run{i}"
        d.mkdir(parents=True)
        write_json(d / "selfcheck.json", {"status": "ok", "submission": sub})
    if last:
        (out / "logs" / "submission").write_text(last + "\n")
    return out


def test_score_only_submits_the_scoring_and_nothing_before_it(site, tmp_path):
    """For a folder whose registrations are done: the settle job, the case arrays,
    tissue, the scorer. No selfcheck, no gate, no ladder, and nothing waits for a job
    that is not there."""
    _records_site(site, tmp_path, "so", ids=("20261012T010203-9",))
    (site[0] / "out-so" / "logs" / "submission").write_text("20261012T010203-9\n")
    done, log, out = launch(site, "--score-only", log="so")
    jobs = jobs_of(done)
    assert list(jobs) == [
        "rb-settle",
        "rb-score-case-multiplex",
        "rb-score-case-synthetic",
        "rb-tissue",
        "rb-score",
    ]
    assert jobs["rb-settle"].deps == set()  # nothing was submitted before it
    for name in list(jobs)[1:3]:
        assert jobs[name].deps == {("afterany", "<rb-settle>")}, name
    assert jobs["rb-tissue"].deps == {
        ("afterany", "<rb-score-case-multiplex>"),
        ("afterany", "<rb-score-case-synthetic>"),
    }
    assert jobs["rb-score"].deps == {
        ("afterany", "<rb-tissue>"),
        ("afterany", "<rb-score-case-multiplex>"),
        ("afterany", "<rb-score-case-synthetic>"),
    }
    assert "# 5 job(s) would be submitted; nothing was" in done.stdout
    assert "WARNING" not in done.stdout
    # it scores the LAST submission into the folder, and says which
    assert "# --score-only: scoring submission 20261012T010203-9" in done.stdout
    assert "nothing is run again" in done.stdout
    # only the scorer's image is asked for, and no GPU
    assert "REGBENCH_SIF_STARE_REGBENCH" in done.stdout
    assert "REGBENCH_SIF_ASHLAR" not in done.stdout + done.stderr
    no_gpu, _, _ = launch(
        site, "--score-only", env={"REGBENCH_SBATCH_GPU": ""}, log="so"
    )
    assert no_gpu.returncode == 0 and len(jobs_of(no_gpu)) == 5
    # one dataset, when named
    one = jobs_of(launch(site, "--score-only", "--datasets", "multiplex", log="so")[0])
    assert list(one) == [
        "rb-settle",
        "rb-score-case-multiplex",
        "rb-tissue",
        "rb-score",
    ]
    # --runs changes nothing (the scorer scores every run), and that is said
    runs, _, _ = launch(site, "--score-only", "--runs", "stare", log="so")
    assert list(jobs_of(runs)) == list(jobs) and "--runs is not used" in runs.stdout
    both, _, _ = launch(site, "--score-only", "--prepare", log="so")
    assert both.returncode == 2 and "two different steps" in both.stderr


def test_score_only_finds_the_submission_or_refuses(site, tmp_path):
    # a folder filled before logs/submission existed: the one id its records carry
    _records_site(site, tmp_path, "so1")
    done, _, _ = launch(site, "--score-only", log="so1")
    assert done.returncode == 0
    assert "# --score-only: scoring submission 20261011T220000-77" in done.stdout
    # an exported id wins over both
    named, _, _ = launch(
        site, "--score-only", env={"REGBENCH_SUBMISSION": "mine-1"}, log="so1"
    )
    assert "# submission  -> mine-1 (scored as it is" in named.stdout
    assert "scoring submission" not in named.stdout
    # two ids and nothing that says which was last: refused, both named
    _records_site(site, tmp_path, "so2", ids=("A-1", "B-2"))
    two, log, _ = launch(site, "--score-only", log="so2")
    assert two.returncode == 2 and "A-1 B-2" in two.stderr and not log.exists()
    assert "Nothing was submitted" in two.stderr and not _sbatch_lines(two.stdout)
    # nothing at all: refused, with where the id is to be found
    none, _, _ = launch(site, "--score-only", log="so3")
    assert none.returncode == 2 and "REGBENCH_SUBMISSION=" in none.stderr
    assert not _sbatch_lines(none.stdout)


def _real_record(out, run, submission, monkeypatch):
    """A selfcheck.json as the selfcheck itself writes it (`selfcheck.run`, on the
    identity method and a small pair), where the launcher's selfcheck job puts it."""
    from benchmarks.regbench import selfcheck as sc

    if submission is None:
        monkeypatch.delenv("REGBENCH_SUBMISSION", raising=False)
    else:
        monkeypatch.setenv("REGBENCH_SUBMISSION", submission)
    work = out / "logs" / "selfcheck" / run
    rec = sc.run("initial", work, sizes=[((256, 320), None)])
    monkeypatch.delenv("REGBENCH_SUBMISSION", raising=False)
    assert rec["submission"] == (submission or "")
    return work / "selfcheck.json"


def test_score_only_reads_the_id_from_records_the_selfcheck_itself_wrote(
    site, monkeypatch
):
    """The id is PARSED out of each record by the reader the gates use
    (`ladder.submissions_of` through `regbench submissions`), not searched for as text."""
    out = site[0] / "out-real"
    path = _real_record(out, "stare", "20261012T090909-41", monkeypatch)
    rec = json.loads(path.read_text())
    # the real record names its submission more than once, and holds other text
    assert path.read_text().count('"submission"') >= 2
    assert rec["attempts"][-1]["submission"] == rec["submission"]
    assert ladder.selfcheck_submissions(out) == ({"20261012T090909-41": ["stare"]}, [])
    done, _, _ = launch(site, "--score-only", log="real")
    assert done.returncode == 0, done.stderr
    assert "# --score-only: scoring submission 20261012T090909-41" in done.stdout
    assert "WARNING" not in done.stdout
    # a second run of the SAME submission: still one id
    _real_record(out, "palom", "20261012T090909-41", monkeypatch)
    done, _, _ = launch(site, "--score-only", log="real")
    assert "# --score-only: scoring submission 20261012T090909-41" in done.stdout
    # text inside a record that only LOOKS like an id is not one: a message quoting
    # another submission, and a key of that name deeper in the record
    rec["pairs"][0]["error"] = 'the gate said "submission": "20260101T000000-1"'
    rec["pairs"][0]["note"] = {"submission": "20260202T000000-2"}
    write_json(path, rec)
    assert '"submission": "20260202T000000-2"' in path.read_text()
    done, _, _ = launch(site, "--score-only", log="real")
    assert done.returncode == 0, done.stderr
    assert "# --score-only: scoring submission 20261012T090909-41" in done.stdout
    # a record cut short is no evidence, and is named
    (out / "logs" / "selfcheck" / "ashlar").mkdir()
    cut = out / "logs" / "selfcheck" / "ashlar" / "selfcheck.json"
    whole = path.read_text().replace("090909-41", "777777-7")
    cut.write_text(whole[: whole.index("777777-7") + 12])  # just past its own id
    assert "777777-7" in cut.read_text()
    found, unreadable = ladder.selfcheck_submissions(out)
    assert list(found) == ["20261012T090909-41"] and unreadable == [str(cut)]
    done, _, _ = launch(site, "--score-only", log="real")
    assert "# --score-only: scoring submission 20261012T090909-41" in done.stdout
    assert f"not a readable record: {cut}" in done.stderr


def test_score_only_with_real_records_of_no_one_or_several_submissions(
    site, monkeypatch
):
    # no selfcheck file at all: refused
    none, _, _ = launch(site, "--score-only", log="real0")
    assert none.returncode == 2 and "no submission to score" in none.stderr
    assert not _sbatch_lines(none.stdout)
    # a selfcheck run by hand, outside the launcher, names no submission: refused too
    out = site[0] / "out-real0"
    _real_record(out, "stare", None, monkeypatch)
    assert ladder.selfcheck_submissions(out) == ({"": ["stare"]}, [])
    by_hand, _, _ = launch(site, "--score-only", log="real0")
    assert by_hand.returncode == 2 and "no submission to score" in by_hand.stderr
    # several runs with different submission ids and nothing that says which was last
    out = site[0] / "out-real2"
    _real_record(out, "stare", "A-1", monkeypatch)
    _real_record(out, "palom", "B-2", monkeypatch)
    _real_record(out, "ashlar", "A-1", monkeypatch)
    assert ladder.selfcheck_submissions(out) == (
        {"A-1": ["ashlar", "stare"], "B-2": ["palom"]},
        [],
    )
    two, log, _ = launch(site, "--score-only", log="real2")
    assert two.returncode == 2 and "more than one submission (A-1 B-2)" in two.stderr
    assert not log.exists() and not _sbatch_lines(two.stdout)
    # ... logs/submission says which was last, and that one is scored without a warning
    (out / "logs" / "submission").write_text("B-2\n")
    last, _, _ = launch(site, "--score-only", log="real2")
    assert last.returncode == 0, last.stderr
    assert "# --score-only: scoring submission B-2" in last.stdout
    assert "WARNING" not in last.stdout


def test_score_only_says_when_every_record_is_another_submissions(site, monkeypatch):
    """A file from another submission is never taken for the one being scored: the id
    asked for is kept, and the launcher says that no record is of it."""
    out = site[0] / "out-real3"
    _real_record(out, "stare", "theirs-1", monkeypatch)
    named, _, _ = launch(
        site, "--score-only", env={"REGBENCH_SUBMISSION": "mine-2"}, log="real3"
    )
    assert named.returncode == 0, named.stderr
    assert "# submission  -> mine-2 (scored as it is" in named.stdout
    (warning,) = [ln for ln in named.stdout.splitlines() if "WARNING" in ln]
    assert "no selfcheck record" in warning and "is of submission mine-2" in warning
    assert "the records there are of: theirs-1" in warning
    # the same when it is logs/submission that names a submission with no record
    (out / "logs" / "submission").write_text("gone-3\n")
    stale, _, _ = launch(site, "--score-only", log="real3")
    assert "# --score-only: scoring submission gone-3" in stale.stdout
    assert "is of submission gone-3 (the records there are of: theirs-1)" in (
        stale.stdout
    )
    # and none when the id asked for is one the records carry
    mine, _, _ = launch(
        site, "--score-only", env={"REGBENCH_SUBMISSION": "theirs-1"}, log="real3"
    )
    assert mine.returncode == 0 and "WARNING" not in mine.stdout


def test_a_submission_leaves_its_id_for_score_only_which_never_replaces_it(site):
    env = {"REGBENCH_ALLOW_PATH_PYTHON": "1", "REGBENCH_SUBMISSION": "first-1"}
    done, _, out = launch(
        site, "--runs", "stare", "--datasets", "multiplex", env=env, dry=False, log="id"
    )
    assert done.returncode == 0, done.stderr
    assert (out / "logs" / "submission").read_text() == "first-1\n"
    # score-only reads it, passes it to its jobs' environment, and submits four jobs
    env = {"REGBENCH_ALLOW_PATH_PYTHON": "1"}
    again, log, _ = launch(
        site, "--score-only", "--datasets", "multiplex", env=env, dry=False, log="id"
    )
    assert again.returncode == 0, again.stderr
    assert "# --score-only: scoring submission first-1" in again.stdout
    assert "# 4 job(s) submitted" in again.stdout
    names = [
        Job("sbatch " + ln.split("\t", 1)[1]).name
        for ln in log.read_text().splitlines()
    ]
    assert names[-4:] == [
        "rb-settle",
        "rb-score-case-multiplex",
        "rb-tissue",
        "rb-score",
    ]
    # ... and another id given by hand does not become "the last submission"
    env["REGBENCH_SUBMISSION"] = "other-2"
    launch(
        site, "--score-only", "--datasets", "multiplex", env=env, dry=False, log="id"
    )
    assert (out / "logs" / "submission").read_text() == "first-1\n"


# ── the order the runs are submitted in ───────────────────────────────────────
USER_ORDER = "ashlar palom deeperhistreg valis120 stare valis_mirage"
ORDERS = (
    USER_ORDER,
    "affine_ashlar ashlar stare",  # an anchored family named BEFORE its anchor
    "affine_ashlar_clean deeperhistreg_cpu",  # run names; one anchored row alone
)


def _sorted_topologically(deps):
    """The jobs of ``{job: {jobs it waits for}}`` in an order in which each comes after
    all it waits for; raises when there is none (a cycle, or a job that does not exist)."""
    waiting = {job: set(needs) for job, needs in deps.items()}
    unknown = {n for needs in waiting.values() for n in needs} - set(waiting)
    assert not unknown, f"jobs waited for that were never submitted: {sorted(unknown)}"
    done = []
    while waiting:
        ready = sorted(job for job, needs in waiting.items() if not needs)
        assert ready, f"a cycle: nothing can start among {sorted(waiting)}"
        for job in ready:
            del waiting[job]
        for needs in waiting.values():
            needs.difference_update(ready)
        done += ready
    return done


def _ladder_order(jobs):
    """The runs in the order their ladders were submitted (multiplex)."""
    first = [j.args[0] for j in jobs.values() if j.script == "run_rung.sbatch"]
    return list(dict.fromkeys(first))


def test_the_default_order_is_the_run_lists(dry):
    jobs, done, _ = dry
    assert _ladder_order(jobs) == list(RUNS) and "# run order" not in done.stdout


def test_unset_the_order_is_ashlar_first_and_a_gpu_task_asks_for_one_gpu(site):
    """The launcher's own defaults: what a launch with neither variable set does."""
    done, _, _ = launch(
        site,
        env={"REGBENCH_RUN_ORDER": None, "REGBENCH_SBATCH_GPU": None},
        log="defaults",
    )
    assert done.returncode == 0, done.stderr
    jobs = jobs_of(done)
    order = _ladder_order(jobs)
    assert order[:5] == [
        "ashlar", "ashlar_clean", "stare", "affine_ashlar", "affine_ashlar_clean",
    ]  # fmt: skip
    assert sorted(order) == sorted(RUNS)
    asked = same = 0
    for job in jobs.values():
        if job.script == "run_rung.sbatch":
            assert ("--gres=gpu:1" in job.words) is RUNS[job.args[0]].gpu, job.name
            asked += RUNS[job.args[0]].gpu
            same += 1
    assert asked and same > asked
    # the same as naming it
    named, _, _ = launch(
        site, env={"REGBENCH_RUN_ORDER": "ashlar affine_ashlar"}, log="named"
    )
    assert _ladder_order(jobs_of(named)) == order


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("lanes", [0, 1, 2, 3, 4])
def test_an_order_changes_the_order_of_submission_and_nothing_else(
    site, dry, order, lanes
):
    """REGBENCH_RUN_ORDER on 0 to 4 lanes: the same jobs, submitted in that order; a
    run that reads another's results still comes after it and waits for it; and the
    graph of what waits for what can be run (it has no cycle)."""
    env = {"REGBENCH_RUN_ORDER": order, "REGBENCH_RUN_LANES": str(lanes)}
    done, _, _ = launch(site, env=env, log=f"order-{lanes}")
    jobs = jobs_of(done)
    wanted = [variants.run_name(r) for r in variants.submission_order(order.split())[0]]
    assert sorted(jobs) == sorted(dry[0])  # the same jobs as without an order
    assert _ladder_order(jobs) == wanted != list(RUNS)
    assert f"# run order: {' '.join(wanted)}" in done.stdout
    # the selfchecks and the gates are submitted in that order too
    checks = [j.args[0] for j in jobs.values() if j.script == "selfcheck.sbatch"]
    assert list(dict.fromkeys(checks)) == [n for n in wanted if n in GATING]
    # every job waits only for jobs submitted before it: nothing can wait for ever
    seen = set()
    for job in jobs.values():
        assert {n.strip("<>") for _, n in job.deps} <= seen, job.name
        seen.add(job.name)
    _sorted_topologically(
        {j.name: {n.strip("<>") for _, n in j.deps} for j in jobs.values()}
    )
    # the anchored rows: after their anchor in the order, waiting for its last array
    # and (the one with a selfcheck of its own) for its last selfcheck attempt
    for run in ("affine_ashlar", "affine_ashlar_clean"):
        assert wanted.index(run) > wanted.index("stare")
        for dataset in ("multiplex", "synthetic"):
            first = rungs(jobs, run, dataset)[0]
            assert ("afterany", f"<{last_job('stare', dataset)}>") in first.deps
    anchored = jobs["rb-sc-affine_ashlar-64g"]
    assert ("afterok", "<rb-sc-stare-512g>") in anchored.deps
    # on lanes, each run starts behind the run dealt onto its lane before it
    if lanes:
        for i, run in enumerate(wanted):
            waits = {n.strip("<>") for _, n in rungs(jobs, run)[0].deps}
            suffix = "-first" if variants.realcheck_gate(run) else ""
            if suffix:
                waits = {
                    n.strip("<>") for _, n in rungs(jobs, run, suffix=suffix)[0].deps
                }
            before = wanted[i - lanes] if i >= lanes else None
            ends = {
                w for w in waits if before and w.startswith(f"rb-{before}-multiplex")
            }
            ends |= {w for w in waits if before and w == f"rb-realgate-{before}"}
            assert bool(ends) is (before is not None), (run, before, waits)


def test_an_anchored_run_named_first_is_moved_and_that_is_said(site):
    done, _, _ = launch(site, env={"REGBENCH_RUN_ORDER": ORDERS[1]}, log="order-moved")
    for run in ("affine_ashlar", "affine_ashlar_clean"):
        said = f"# run order: {run} reads the results of stare: it is submitted directly after stare"
        assert said in done.stdout
    order = _ladder_order(jobs_of(done))
    at = order.index("stare")
    assert order[at : at + 3] == ["stare", "affine_ashlar", "affine_ashlar_clean"]
    assert order[0] == "ashlar"  # the rest of what was named keeps its place
    # the user's order moves nothing, and says nothing of the kind
    plain, _, _ = launch(site, env={"REGBENCH_RUN_ORDER": USER_ORDER}, log="order-user")
    assert "reads the results of" not in plain.stdout
    assert _ladder_order(jobs_of(plain))[:4] == [
        "ashlar", "ashlar_clean", "palom", "deeperhistreg_gpu",
    ]  # fmt: skip


def test_an_order_with_an_unknown_name_is_refused(site):
    done, log, _ = launch(
        site, env={"REGBENCH_RUN_ORDER": "ashlar valis"}, log="order-bad"
    )
    assert done.returncode == 2 and not _sbatch_lines(done.stdout) and not log.exists()
    assert "no family and no run named 'valis'" in done.stderr
    assert "Families: stare, valis_mirage, valis120" in done.stderr
    assert "Nothing was submitted" in done.stderr


def test_the_order_is_of_submission_only():
    """`runs`, the expected names and the primaries are what they were: no table and no
    result can depend on REGBENCH_RUN_ORDER."""
    before = variants.runs_table(), variants.expected_variants(), variants.PRIMARY
    ordered = variants.runs_table(order=USER_ORDER.split())
    assert sorted(ordered) == sorted(before[0]) and ordered != before[0]
    assert variants.runs_table(order=[]) == before[0]  # no name: the run list's order
    assert (
        variants.runs_table(),
        variants.expected_variants(),
        variants.PRIMARY,
    ) == before
    code = code_of(SLURM / "score_cohort.sbatch") + code_of(SLURM / "run_rung.sbatch")
    assert "REGBENCH_RUN_ORDER" not in code and "--order" not in code


@pytest.mark.parametrize("lanes", [1, 2, 3, 4])
def test_the_submitted_graph_of_the_users_order_has_no_cycle(site, lanes):
    """The real path: what the stand-in sbatch was GIVEN as --dependency, by job id."""
    env = {
        "REGBENCH_RUN_ORDER": USER_ORDER,
        "REGBENCH_RUN_LANES": str(lanes),
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
    }
    done, log, _ = launch(site, env=env, dry=False, log=f"order-real-{lanes}")
    assert done.returncode == 0, done.stderr
    deps, names = {}, {}
    for line in log.read_text().splitlines():
        job_id, argv = line.split("\t", 1)
        job = Job("sbatch " + argv)
        names[job_id] = job.name
        deps[job_id] = {n for _, n in job.deps}
    order = _sorted_topologically(deps)
    assert len(order) == len(deps) == N_JOBS
    # a cycle would be found: the check is not vacuous
    a, b = list(deps)[:2]
    with pytest.raises(AssertionError, match="a cycle"):
        _sorted_topologically({**deps, a: {b}, b: {a}})
    # an anchored run and its anchor on ONE lane: the anchor's ladder comes first
    by_name = {n: i for i, n in names.items()}
    anchored = deps[by_name["rb-affine_ashlar-multiplex-16g"]]
    assert by_name[last_job("stare")] in anchored
    # ... and of the anchor's jobs it waits for that one alone: not COARSE, not SOLVE
    assert {names[i] for i in anchored if names[i].startswith("rb-stare-")} == {
        last_job("stare")
    }


# ── a run whose registration is several jobs per round ────────────────────────
def attempts(jobs, run, stage, dataset="multiplex"):
    return [jobs[f"rb-{run}-{dataset}-{stage}-a{a}"] for a in (1, 2, 3, 4)]


def feeders(jobs, run, dataset="multiplex"):
    """The chain of tile submitters of a run on a dataset, in order."""
    return [jobs[f"rb-{run}-{dataset}-feed-{i}"] for i in range(1, FEEDERS + 1)]


def after(job):
    return {n.strip("<>") for kind, n in job.deps if kind == "afterany"}


def test_stare_is_the_pipelines_four_stages_as_jobs_of_their_own(dry):
    """COARSE, one REG-TILE task per tile, SOLVE, then the writer: each with the
    allocation the pipeline gives that process, each attempt after the one before."""
    jobs, _, _ = dry
    assert STAGED == {
        "stare": ("coarse", "tile", "solve", "stitch"),
        "stare_prep": ("prep", "coarse", "tile", "solve", "stitch"),
    }
    for run, names in STAGED.items():
        spec = RUNS[run]
        gate = ("afterok", f"<rb-gate-{run}>")
        before = None  # the last job of the stage before
        for stage in names[:-1]:
            if stage == "tile":
                # no tile job is submitted by the launcher: the run's tile submitters
                # are, one after the other, between COARSE and SOLVE
                assert not [n for n in jobs if n.startswith(f"rb-{run}-multiplex-tile")]
                for link, job in enumerate(feeders(jobs, run), 1):
                    assert job.script == "feed_tiles.sbatch" and gate in job.deps
                    assert after(job) == {before}, job.name
                    assert job.opt("--array") is None and "--wait" not in job.words
                    assert (job.opt("--cpus-per-task"), job.opt("--mem")) == ("1", "1G")
                    assert job.opt("--time") == "7-00:00:00"
                    assert job.args[:6] == [
                        run,
                        spec.method,
                        spec.label or "-",
                        ",".join(variants.opt_args(spec)) or "-",
                        spec.image,
                        "multiplex",
                    ]
                    assert job.args[6].endswith(
                        f"/logs/stages/<submission>/{run}-multiplex.tsv"
                    )
                    # four rounds of six tiles, this link, and the pipeline's
                    # allocation of a tile job at each of its four attempts
                    assert job.args[7:10] == ["4", str(4 * TILES), str(link)]
                    assert job.args[10:] == [
                        "{cpus}:{mem_gb}:{time}".format(**stages.allocation("tile", a))
                        for a in (1, 2, 3, 4)
                    ]
                    before = job.name
                continue
            chained = attempts(jobs, run, stage)
            assert [j.opt("--array") for j in chained] == ["0-3%20"] * 4, stage
            for attempt, job in enumerate(chained, 1):
                want = stages.allocation(stage, attempt)
                assert job.script == "stage.sbatch" and gate in job.deps
                assert job.opt("--cpus-per-task") == str(want["cpus"]), (stage, attempt)
                assert job.opt("--mem") == f"{want['mem_gb']}G", (stage, attempt)
                assert job.opt("--time") == want["time"], (stage, attempt)
                assert job.args[:10] == [
                    run,
                    spec.method,
                    spec.label or "-",
                    ",".join(variants.opt_args(spec)) or "-",
                    spec.image,
                    "multiplex",
                    stage,
                    str(attempt),
                    str(want["mem_gb"]),
                    want["time"],
                ]
                assert "--wait" not in job.words and GPU not in job.words
                # each job after the one before it, whatever became of that one
                assert after(job) == ({before} if before else set()), job.name
                before = job.name
        # the writer: the round's last job, after SOLVE's last attempt
        last = rungs(jobs, run)[0]
        assert last.script == "run_rung.sbatch" and after(last) == {before}
        assert last.args[13] == "jobs" and last.opt("--array") == "0-3%20"
    # the allocations, as numbers: the pipeline's at the high tier and stride 64
    got = {
        st: [
            (j.opt("--cpus-per-task"), j.opt("--mem"))
            for j in attempts(jobs, "stare", st)
        ]
        for st in ("coarse", "solve")
    }
    assert got["coarse"] == [("2", "2G"), ("2", "4G"), ("2", "6G"), ("2", "8G")]
    assert got["solve"] == [("1", "1G"), ("1", "2G"), ("1", "3G"), ("1", "4G")]
    assert [j.opt("--time") for j in attempts(jobs, "stare", "solve")] == [
        "0-08:00:00", "0-16:00:00", "1-00:00:00", "1-08:00:00"
    ]  # fmt: skip
    assert feeders(jobs, "stare")[0].args[10:] == [
        "2:4:0-02:00:00", "2:8:0-04:00:00", "2:12:0-06:00:00", "2:16:0-08:00:00"
    ]  # fmt: skip
    # SOLVE waits for the LAST tile submitter and for nothing else of the run
    assert after(jobs["rb-stare-multiplex-solve-a1"]) == {
        f"rb-stare-multiplex-feed-{FEEDERS}"
    }
    # the preprocessing is the benchmark's own step, with the benchmark's allocation
    prep = attempts(jobs, "stare_prep", "prep")
    assert [(j.opt("--cpus-per-task"), j.opt("--mem")) for j in prep] == [
        ("4", "16G"), ("4", "32G"), ("4", "48G"), ("4", "64G")
    ]  # fmt: skip
    assert "rb-stare-multiplex-prep-a1" not in jobs


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """Cases of other sizes: `multiplex` three rounds of 1, 6 and 2 tiles, `synthetic`
    one round of a single tile."""
    cases = tmp_path_factory.mktemp("small")
    _case(cases, "multiplex", "P1_a", "P1", (1024, 1024))
    _case(cases, "multiplex", "P1_b", "P1")
    _case(cases, "multiplex", "P2_a", "P2", (2048, 4096))
    _case(cases, "synthetic", "blob_grid_p0", "blob_grid", (1024, 1024))
    return cases


def test_the_dry_run_says_what_is_submitted_now_and_what_is_fed_in_later(site, small):
    env = {
        "REGBENCH_CASES": str(small),
        "REGBENCH_STAGE_ARRAY_MAX": "7",
        "REGBENCH_STAGE_QUEUE_MAX": "30",
    }
    done, _, _ = launch(site, "--runs", "stare stare_prep", env=env, log="totals")
    jobs = jobs_of(done)
    tasks = sum(
        int(r.split("-")[1]) - int(r.split("-")[0]) + 1
        for j in jobs.values()
        for r in [(j.opt("--array") or "0-0").split("%")[0]]
    )
    said = done.stdout
    # multiplex: rounds of 1, 6 and 2 tiles; synthetic: one tile; for each of two runs
    assert (
        "# stare on multiplex: 3 round(s) as separate jobs (coarse,tile,solve,stitch); "
        "9 tile job(s) fed in later by its tile submitter, in arrays of at most 7, at "
        "most 30 in the queue" in said
    )
    assert (
        f"# {len(jobs)} job(s) would be submitted; nothing was, {tasks} task(s) in them "
        f"(an array counts once as a job and once per task): {len(jobs)} sbatch "
        "call(s) would be made now" in said
    )
    assert (
        "# later, by the tile submitters (feed_tiles.sbatch): 20 tile task(s), one per "
        "tile, in arrays of at most 7; plus, per attempt 2 to 4, one retry task per "
        "round that has a tile to retry" in said
    )
    assert (
        f"# the most in the queue at once: the {tasks} task(s) above plus at most 30 "
        f"tile task(s) for each of the 2 run(s) with a tile submitter (stare "
        f"stare_prep): {tasks + 2 * 30} task(s)" in said
    )
    # no tile task is among what is submitted now
    assert not [n for n in jobs if "-tile-" in n]
    # a submission without such a run says nothing about tile submitters
    done, _, _ = launch(site, "--runs", "palom", log="totals-none")
    assert "tile submitter" not in done.stdout


@pytest.mark.parametrize(
    "name, value",
    [
        ("REGBENCH_STAGE_ARRAY_MAX", "0"),
        ("REGBENCH_STAGE_ARRAY_MAX", "ten"),
        ("REGBENCH_STAGE_QUEUE_MAX", "0"),
        ("REGBENCH_STAGE_QUEUE_MAX", "-5"),
        ("REGBENCH_ARRAY_LIMIT", "0"),
        ("REGBENCH_ARRAY_LIMIT", "2.5"),
        ("REGBENCH_STAGE_FEEDERS", "0"),
    ],
)
def test_a_limit_that_is_not_a_whole_number_of_at_least_one_is_refused(
    site, name, value
):
    done, log, _ = launch(site, env={name: value}, dry=False, log="limits")
    assert done.returncode == 2 and not log.exists()
    assert f"{name}='{value}': a whole number of at least 1" in done.stderr
    assert "Nothing was submitted." in done.stderr


def test_the_tile_submitter_reads_the_layout_the_submission_wrote(site, small):
    """The real path: the layout file exists where the tile submitters are told to read
    it, under the results folder, in a folder of this submission; the caller's own
    sbatch arguments are beside it, for the jobs the submitter submits."""
    env = {
        "REGBENCH_CASES": str(small),
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "REGBENCH_SUBMISSION": "S-7",
    }
    done, log, out = launch(
        site,
        "--runs",
        "stare",
        "--datasets",
        "multiplex",
        "--",
        "--account=lab",
        "--partition=long",
        env=env,
        dry=False,
        log="layout",
    )
    assert done.returncode == 0, done.stderr
    calls = [
        Job("sbatch " + ln.split("\t", 1)[1]) for ln in log.read_text().splitlines()
    ]
    assert not [j for j in calls if "-tile-" in j.name]
    links = [j for j in calls if j.name.startswith("rb-stare-multiplex-feed-")]
    layout = out / "logs" / "stages" / "S-7" / "stare-multiplex.tsv"
    assert len(links) == FEEDERS and {j.args[6] for j in links} == {str(layout)}
    assert [j.args[7:9] for j in links] == [["3", "9"]] * FEEDERS
    assert layout == Path(f"{stages.feed_state(out, 'S-7', 'stare', 'multiplex')}.tsv")
    assert stages.read_layout(layout) == [("P1_a", 1), ("P1_b", 6), ("P2_a", 2)]
    assert layout.with_suffix(".sbatch_args").read_text().splitlines() == [
        "--account=lab",
        "--partition=long",
    ]
    assert [stages.locate([1, 6, 2], i) for i in (0, 1, 6, 7, 8, 9)] == [
        (0, 0), (1, 0), (1, 5), (2, 0), (2, 1), None
    ]  # fmt: skip
    # this call's job ids, one per line: what a refusal tells the user to cancel
    ids = (out / "logs" / "submitted" / "S-7.ids").read_text().split()
    assert (
        len(ids) == len(calls)
        and f'scancel $(cat "{out}/logs/submitted/S-7.ids")' in done.stdout
    )


# ── one id, one submission ────────────────────────────────────────────────────
def test_a_submission_id_is_used_once_per_results_folder(site):
    """The files of a tile submitter are kept under the id: a second submission under
    the same id would take the first one's lists for its own and feed nothing."""
    env = {"REGBENCH_ALLOW_PATH_PYTHON": "1", "REGBENCH_SUBMISSION": "S-A"}
    args = ("--runs", "stare", "--datasets", "multiplex")
    first, log, out = launch(site, *args, env=env, dry=False, log="once")
    assert first.returncode == 0, first.stderr
    n = len(log.read_text().splitlines())
    state = stages.feed_state(out, "S-A", "stare", "multiplex")
    Path(f"{state}.fed.tsv").write_text("1\t0\t9\t0\t4242\n")
    Path(f"{state}.due.a1").write_text("")
    again, _, _ = launch(site, *args, env=env, dry=False, log="once")
    assert again.returncode == 2 and len(log.read_text().splitlines()) == n
    assert "the submission id 'S-A' was already used" in again.stderr
    assert str(out / "logs" / "submitted" / "S-A.ids") in again.stderr
    assert "Unset REGBENCH_SUBMISSION" in again.stderr
    assert "Nothing was submitted." in again.stderr
    assert Path(f"{state}.fed.tsv").exists()  # nothing of the first one was touched
    # scoring that submission is what its id is for; a dry run submits nothing
    scoring, _, _ = launch(site, "--score-only", env=env, dry=False, log="once")
    assert scoring.returncode == 0, scoring.stderr
    assert launch(site, *args, env=env, log="once")[0].returncode == 0
    # another id: a submission of its own, with its own files
    other, _, _ = launch(
        site, *args, env={**env, "REGBENCH_SUBMISSION": "S-B"}, dry=False, log="once"
    )
    assert other.returncode == 0, other.stderr
    assert (out / "logs" / "submission").read_text() == "S-B\n"
    assert not Path(
        f"{stages.feed_state(out, 'S-B', 'stare', 'multiplex')}.fed.tsv"
    ).exists()


# ── nothing that can stop the script is left for after the first sbatch ───────
def _python_that_fails_on(tmp_path, word):
    """A `python` that is this machine's, except that a regbench call with ``word``
    among its arguments fails."""
    wrap = tmp_path / "wrap"
    wrap.mkdir()
    python = wrap / "python"
    python.write_text(
        "#!/bin/bash\n"
        f'for a in "$@"; do [ "$a" = {shlex.quote(word)} ] && '
        '{ echo "boom" >&2; exit 1; }; done\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    python.chmod(0o755)
    return wrap


@pytest.mark.parametrize(
    "word, said",
    [
        ("--stage-allocation", "cannot read the pipeline's allocation per stage"),
        ("stages", "cannot size the tile arrays of stare on multiplex"),
    ],
)
def test_a_read_that_fails_stops_the_launch_before_anything_is_queued(
    site, tmp_path, word, said
):
    wrap = _python_that_fails_on(tmp_path, word)
    env = {
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "PATH": f"{wrap}:{site[2]}:{Path(sys.executable).parent}:{os.environ['PATH']}",
    }
    name = f"early-{word.strip('-')}"
    done, log, out = launch(site, env=env, dry=False, log=name)
    assert done.returncode == 2 and not log.exists()  # not one sbatch
    assert said in done.stderr and "Nothing was submitted." in done.stderr
    assert "already queued" not in done.stderr
    assert not (out / "logs" / "submission").exists()
    assert not (out / "logs" / "jobs.tsv").exists()


def test_every_read_of_the_launcher_comes_before_its_first_sbatch():
    """In the script itself: after the selfchecks start, nothing asks the harness for
    anything, nothing refuses, no time is parsed and no variable of the caller is
    read for the first time."""
    code = code_of(SCRIPT)
    start = code.index("sc_time=")
    tail = code[start:]
    assert code.count("sc_time=") == 1
    assert "regbench_py" not in tail
    assert not re.search(r"(^|[;&|]\s*|\s)die\s", tail)
    assert "fetch_stare_source\n    preflight\n" in code[:start]
    # a time that is not one must stop the script, which only a plain assignment
    # before the first sbatch does: none is nested in another substitution anywhere
    assert not re.search(r"\$\(\s*seconds\s", tail)
    assert not re.search(r"\$\([^()]*\$\(\s*seconds\s", code)
    for call in re.findall(r"^.*\$\(\s*seconds\s.*$", code, re.M):
        assert re.match(r"^\s*\w+=\"\$\(seconds [^()]*\)\"$", call), call
    later = set(re.findall(r"\$\{?!?(REGBENCH_[A-Z0-9_]+)", tail))
    earlier = set(re.findall(r"REGBENCH_[A-Z0-9_]+", code[:start]))
    assert later <= earlier, sorted(later - earlier)


@pytest.mark.parametrize(
    "name, value",
    [
        ("REGBENCH_SCORE_TIME", "tomorrow"),
        ("REGBENCH_SCORE_TIME", "24:00"),
        ("REGBENCH_STAGE_FEED_TIME", "7d"),
        ("REGBENCH_STAGE_FEED_TIME", "1-xx:00:00"),
        ("REGBENCH_NUCLEI_TIME", "8h"),
    ],
)
def test_a_time_that_is_not_one_stops_the_launch_before_anything_is_queued(
    site, name, value
):
    env = {"REGBENCH_ALLOW_PATH_PYTHON": "1", name: value}
    done, log, out = launch(site, env=env, dry=False, log=f"time-{name}")
    assert done.returncode == 2 and not log.exists(), done.stderr
    assert f"a time must be [D-]HH:MM:SS, got '{value}'" in done.stderr
    assert "Nothing was submitted." in done.stderr
    assert "already queued" not in done.stderr
    assert not (out / "logs" / "submission").exists()


@pytest.mark.parametrize(
    "name, value, least",
    [
        ("REGBENCH_STAGE_STALL_S", "1h", 1),
        ("REGBENCH_STAGE_STALL_MAX_S", "0", 1),
        ("REGBENCH_STAGE_SQUEUE_TRIES", "", 1),
        ("REGBENCH_ACCT_WAIT_S", "ten", 0),
        ("REGBENCH_ACCT_POLL_S", "0", 1),
    ],
)
def test_a_number_the_jobs_read_is_checked_by_the_launcher(site, name, value, least):
    env = {"REGBENCH_ALLOW_PATH_PYTHON": "1", name: value}
    done, log, _ = launch(site, env=env, dry=False, log=f"number-{name}")
    assert done.returncode == 2 and not log.exists(), done.stderr
    assert f"{name}='{value}': a whole number of at least {least}" in done.stderr


def test_the_stitch_time_is_summed_before_the_first_sbatch(dry):
    """The last job of a staged round has one round's time plus the writer's four
    attempts (4 + 8 + 12 + 16 h), each parsed before anything is submitted."""
    code = code_of(SCRIPT)
    assert "STITCH_S=$((STITCH_S + stage_s))" in code[: code.index("sc_time=")]
    jobs, _, _ = dry
    for run in STAGED:
        assert rungs(jobs, run)[0].opt("--time") == "2-16:00:00"


@pytest.mark.parametrize("signum, code", [(15, 143), (1, 129), (2, 130)])
def test_a_signal_in_the_middle_of_a_submission_says_what_is_queued(site, signum, code):
    """The launcher is signalled while the scheduler is taking its fifth job: that job
    is in the ids file, and the script's last words name what is queued."""
    env = {
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "REGBENCH_SUBMISSION": f"S-sig{signum}",
        "SBATCH_SLEEP_AT": "1005",
    }
    argv, base, out = _launch_command(site, (), env, False, f"signal{signum}")
    log = Path(base["SBATCH_LOG"])
    own = subprocess.Popen(
        argv, env=base, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=REPO,
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline and own.poll() is None:
            if log.exists() and len(log.read_text().splitlines()) >= 5:
                break
            time.sleep(0.05)
        assert own.poll() is None, own.communicate()
        own.send_signal(signum)  # this test's own launcher, by the pid it started
        stdout, said = own.communicate(timeout=120)
    finally:
        if own.poll() is None:
            own.kill()
            own.communicate()
    assert own.returncode == code, said
    assert len(log.read_text().splitlines()) == 5
    ids = out / "logs" / "submitted" / f"S-sig{signum}.ids"
    assert ids.read_text().split() == [str(i) for i in range(1001, 1006)]
    name = {15: "TERM", 1: "HUP", 2: "INT"}[signum]
    assert f"stopped with exit {code} after the first sbatch (signal {name})" in said
    assert (
        f"5 job(s) of this call are already queued (submission S-sig{signum})" in said
    )
    assert "the rest was NOT submitted, so the graph is incomplete" in said
    assert f'scancel $(cat "{ids}")' in said
    assert "job(s) submitted" not in stdout
    assert (out / "logs" / "submission").read_text() == f"S-sig{signum}\n"


def test_a_stray_statement_of_the_stage_launcher_reaches_no_job():
    """`--export=ALL` carries the caller's shell into every job: the launcher drops
    REGBENCH_STAGES, and the one script that states it starts from nothing."""
    code = code_of(SCRIPT)
    assert "unset REGBENCH_STAGES\n" in code[: code.index("RUNGS_GB=")]
    rung = code_of(SLURM / "run_rung.sbatch")
    stated = rung.index('export REGBENCH_STAGES="$stages"')
    assert "unset REGBENCH_STAGES\n" in rung[:stated]
    done = subprocess.run(
        ["bash", "-c", f'set -- n m - - I 0 d 1 16 0 - first t -; {rung[: stated].split("retry=0")[0]}'
         '[ "$stages" = - ] || export REGBENCH_STAGES="$stages"; echo "[${REGBENCH_STAGES-unset}]"'],
        env={**os.environ, "REGBENCH_STAGES": "local", "REGBENCH_REPO": str(REPO)},
        capture_output=True, text=True, cwd=REPO,
    )  # fmt: skip
    assert done.stdout.strip().endswith("[unset]"), done.stdout + done.stderr


def test_a_notice_of_the_sites_submit_plugin_is_not_a_refusal(site):
    """sbatch prints a line of the site's own before the job id (seen on the cluster,
    for every job that asks for a GPU): the job is queued, and the launch goes on."""
    env = {"REGBENCH_ALLOW_PATH_PYTHON": "1"}
    plain, plain_log, _ = launch(site, env=env, dry=False, log="quiet")
    said, log, _ = launch(
        site, env={**env, "SBATCH_NOTICE": "1"}, dry=False, log="said"
    )
    assert plain.returncode == 0 and said.returncode == 0, said.stderr
    assert "REFUSED" not in said.stderr
    assert (
        "# sbatch said: sbatch: slurm_job_submit: Job assigned to GPU nodes"
        in said.stderr
    )
    rows = [line.split("\t")[0] for line in log.read_text().splitlines()]
    assert rows and len(rows) == len(plain_log.read_text().splitlines())
    # every job the launch printed has a plain id, and waits on plain ids
    for line in said.stdout.splitlines():
        if line.startswith("rb-"):
            assert line.split("\t")[1].isdigit(), line
    assert "slurm_job_submit" not in log.read_text()


def test_a_first_sbatch_that_is_refused_leaves_no_submission(site):
    env = {
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "REGBENCH_SUBMISSION": "S-none",
        "SBATCH_REFUSE_AT": "1001",
    }
    done, log, out = launch(site, env=env, dry=False, log="refused-first")
    assert done.returncode == 2 and not log.exists()
    assert "sbatch REFUSED the job 'rb-" in done.stderr
    # nothing is queued: nothing to cancel, and no submission for --score-only to find
    assert "after cancelling what is queued" not in done.stderr
    assert done.stderr.rstrip().endswith("Nothing was submitted.")
    assert "already queued" not in done.stderr
    assert not (out / "logs" / "submission").exists()


@pytest.mark.parametrize("how", [(), ("--score-only",)])
def test_synthetic_cases_of_another_truth_spec_are_refused_at_the_launch(
    site, tmp_path, how
):
    """The scorer refuses such cases, and then writes no table for any dataset: the
    launcher asks the same question before its first job."""
    cases = tmp_path / "cases"
    _case(cases, "multiplex", "P1_a", "P1")
    _case(cases, "synthetic", "blob_grid_p0", "blob_grid")
    record = cases / "synthetic" / "blob_grid_p0" / "case.json"
    case = json.loads(record.read_text())
    case["extra"] = {"truth_spec_sha256": "0" * 64}
    record.write_text(json.dumps(case))
    env = {
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "REGBENCH_CASES": str(cases),
        "REGBENCH_SUBMISSION": "S-spec",
    }
    name = "spec" + "-".join(how)
    done, log, out = launch(site, *how, env=env, dry=False, log=name)
    assert done.returncode == 2 and not log.exists(), done.stderr
    assert "this case was prepared from truth_spec.json 000000000000" in done.stderr
    assert "Nothing was prepared" not in done.stderr
    assert done.stderr.rstrip().endswith("Nothing was submitted.")
    assert not (out / "logs" / "submission").exists()
    # the cohort alone is not stopped by cases it does not submit
    done, log, _ = launch(
        site, *how, "--datasets", "multiplex", "--runs", "palom", env=env, log=name
    )
    assert done.returncode == 0, done.stderr


def test_whatever_stops_the_launch_after_the_first_sbatch_says_what_is_queued(site):
    """Not the scheduler this time: the launcher's own list of jobs cannot be written
    from the sixth job on. The script stops, and its last words are the same."""
    env = {
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "REGBENCH_SUBMISSION": "S-10",
        "SBATCH_BREAK_AT": "1006",
        "SBATCH_BREAK_FILE": str(site[0] / "out-broken" / "logs" / "jobs.tsv"),
    }
    done, log, out = launch(site, env=env, dry=False, log="broken")
    try:
        assert done.returncode not in (0, 2)
        assert len(log.read_text().splitlines()) == 6
        ids = out / "logs" / "submitted" / "S-10.ids"
        # the sixth job is in the file the notice names, though its line in the job
        # list could not be written
        assert ids.read_text().split() == [str(i) for i in range(1001, 1007)]
        said = done.stderr
        assert "stopped with exit" in said and "after the first sbatch" in said
        assert "6 job(s) of this call are already queued (submission S-10)" in said
        assert "the rest was NOT submitted, so the graph is incomplete" in said
        assert f'scancel $(cat "{ids}")' in said
        assert "job(s) submitted" not in done.stdout
    finally:
        (out / "logs" / "jobs.tsv").chmod(0o644)


# ── the pinned STARE source is fetched whether or not the preflight runs ──────
def _stare_release(tmp_path):
    """A stand-in `curl` that serves an archive laid out as the pinned release's."""
    import tarfile

    release = tmp_path / "stare-9.9.9" / "src" / "stare"
    release.mkdir(parents=True)
    (release / "slide_io.py").write_text("WHAT = 'the pinned source'\n")
    archive = tmp_path / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tmp_path / "stare-9.9.9", arcname="stare-9.9.9")
    fake = tmp_path / "net"
    fake.mkdir()
    curl = fake / "curl"
    curl.write_text(
        f'#!/bin/bash\necho "$*" >> "{tmp_path}/curl.log"\n'
        f'[ -e "{tmp_path}/offline" ] && exit 22\ncat "{archive}"\n'
    )
    curl.chmod(0o755)
    return fake


@pytest.mark.parametrize("preflight", ["0", "1"])
def test_the_stare_source_is_fetched_with_and_without_the_preflight(
    site, tmp_path, preflight
):
    """ASHLAR's image has no `stare` package: its jobs import the pinned source from
    the results folder, so it is there before the first sbatch, preflight or not."""
    net = _stare_release(tmp_path)
    runtime = _image_runtime(tmp_path, ())  # an image that does not start
    env = {
        "REGBENCH_PREFLIGHT": preflight,
        "REGBENCH_SIF_ASHLAR": "/images/ashlar.sif",
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "PATH": f"{net}:{runtime}:{site[2]}:{Path(sys.executable).parent}:"
        f"{os.environ['PATH']}",
    }
    args = ("--runs", "ashlar", "--datasets", "multiplex")
    done, log, out = launch(site, *args, env=env, dry=False, log=f"src-{preflight}")
    asked = (tmp_path / "curl.log").read_text().splitlines()
    assert len(asked) == 1 and "sceriff0/stare/archive/refs/tags/" in asked[0]
    tag = asked[0].rsplit("/", 1)[1][: -len(".tar.gz")]  # the pinned release
    src = out / "cache" / f"stare-{tag}" / "src"
    assert (src / "stare" / "slide_io.py").read_text().startswith("WHAT")
    line = f"# STARE source for the images without one (ASHLAR VALIS_MIRAGE) -> {src}"
    assert line in done.stdout
    if preflight == "0":
        # skipped, said, and the launch goes on with the source in place
        assert done.returncode == 0, done.stderr
        assert "# WARNING preflight SKIPPED" in done.stdout
        assert done.stdout.index(line) < done.stdout.index("rb-sc-")
    else:
        # the image does not start here: the preflight stops the launch, after the fetch
        assert done.returncode == 2 and not log.exists()
        assert "preflight FAIL [ASHLAR ashlar]" in done.stdout
    # a second launch finds it and fetches nothing
    launch(site, *args, env=env, dry=False, log=f"src-{preflight}")
    assert len((tmp_path / "curl.log").read_text().splitlines()) == 1


def test_a_source_that_cannot_be_fetched_stops_the_launch_without_the_preflight(
    site, tmp_path
):
    net = _stare_release(tmp_path)
    (tmp_path / "offline").write_text("")
    env = {
        "REGBENCH_PREFLIGHT": "0",
        "REGBENCH_SIF_ASHLAR": "/images/ashlar.sif",
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "PATH": f"{net}:{site[2]}:{Path(sys.executable).parent}:{os.environ['PATH']}",
    }
    done, log, out = launch(
        site, "--runs", "ashlar", "--datasets", "multiplex", env=env, dry=False,
        log="src-offline",
    )  # fmt: skip
    assert done.returncode == 2 and not log.exists()
    assert "the pinned STARE source could not be fetched" in done.stderr
    assert "Nothing was submitted." in done.stderr
    assert not (out / "logs" / "submission").exists()
    # an image that is not named as a file has its own STARE: nothing is fetched
    env.pop("REGBENCH_SIF_ASHLAR")
    done, log, _ = launch(
        site, "--runs", "ashlar", "--datasets", "multiplex", env=env, dry=False,
        log="src-none",
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert len((tmp_path / "curl.log").read_text().splitlines()) == 1


# ── a launch that cannot leave a silent partial graph ─────────────────────────
def test_a_refused_sbatch_stops_the_launch_and_says_how_to_undo_it(site):
    env = {
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "REGBENCH_SUBMISSION": "S-9",
        "SBATCH_REFUSE_AT": "1031",  # the scheduler takes 30 jobs, then no more
    }
    done, log, out = launch(site, env=env, dry=False, log="refused")
    assert done.returncode == 2
    assert len(log.read_text().splitlines()) == 30
    ids = out / "logs" / "submitted" / "S-9.ids"
    assert ids.read_text().split() == [str(i) for i in range(1001, 1031)]
    said = done.stderr
    assert "sbatch REFUSED the job 'rb-" in said
    assert "QOSMaxSubmitJobPerUserLimit" in said
    assert "30 job(s) of this call are already queued (submission S-9)" in said
    assert "the rest was NOT submitted, so the graph is incomplete" in said
    assert f'scancel $(cat "{ids}")' in said and str(out / "logs" / "jobs.tsv") in said
    assert "job(s) submitted" not in done.stdout  # the closing lines never appear


def test_a_second_submission_is_refused_while_the_first_is_still_queued(site, tmp_path):
    env = {"REGBENCH_ALLOW_PATH_PYTHON": "1"}
    first, log, out = launch(site, "--runs", "palom", env=env, dry=False, log="twice")
    assert first.returncode == 0, first.stderr
    n = len(log.read_text().splitlines())
    ids = (out / "logs" / "jobs.tsv").read_text().splitlines()
    one = ids[3].split("\t")[2]
    record = out / "logs" / "gates" / "palom.tsv"
    record.write_text("palom\tok\t-\t0\t-\n")
    queue = tmp_path / "queue"
    # an array task of the first submission is still running; another user's job is not ours
    queue.write_text(f"{one} {one}_4\n999999 999999\n")
    again, _, _ = launch(
        site,
        "--runs",
        "palom",
        env={**env, "SQUEUE_FILE": str(queue)},
        dry=False,
        log="twice",
    )
    assert again.returncode == 2 and len(log.read_text().splitlines()) == n
    assert (
        f"1 job(s) of an earlier submission into {out} are still pending or running "
        f"({one})" in again.stderr
    )
    assert (
        f"scancel {one} ;" in again.stderr and "Nothing was submitted." in again.stderr
    )
    # the gate record the first submission's tasks read is as its gate job left it
    assert record.read_text() == "palom\tok\t-\t0\t-\n"
    # a queue that cannot be read is not an empty queue
    blind, _, _ = launch(
        site,
        "--runs",
        "palom",
        env={**env, "SQUEUE_FAILS": "1"},
        dry=False,
        log="twice",
    )
    assert blind.returncode == 2 and "the queue cannot be read (squeue)" in blind.stderr
    # --score-only is refused the same way, unless it is told to score all the same
    queue.write_text(f"{one} {one}_4\n999999 999999\n")
    env_q = {**env, "SQUEUE_FILE": str(queue)}
    scoring, _, _ = launch(site, "--score-only", env=env_q, dry=False, log="twice")
    assert scoring.returncode == 2 and len(log.read_text().splitlines()) == n
    assert (
        f"--score-only: 1 job(s) of an earlier call into {out} are still pending or "
        f"running ({one})" in scoring.stderr
    )
    assert "REGBENCH_SCORE_WHILE_QUEUED=1" in scoring.stderr
    assert "Nothing was submitted." in scoring.stderr
    blind, _, _ = launch(
        site, "--score-only", env={**env, "SQUEUE_FAILS": "1"}, dry=False, log="twice"
    )
    assert blind.returncode == 2 and "the queue cannot be read (squeue)" in blind.stderr
    forced, _, _ = launch(
        site,
        "--score-only",
        env={**env_q, "REGBENCH_SCORE_WHILE_QUEUED": "1", "SQUEUE_FAILS": "1"},
        dry=False,
        log="twice",
    )
    assert forced.returncode == 0, forced.stderr
    n = len(log.read_text().splitlines())
    assert "# 5 job(s) submitted" in forced.stdout
    # a dry run is not refused; and once the queue is empty, neither is a scoring nor
    # a second submission
    queue.write_text("999999 999999\n")
    for args, dry in (
        (["--score-only"], False),
        ([], True),
        (["--runs", "palom"], False),
    ):
        done, _, _ = launch(
            site, *args, env={**env, "SQUEUE_FILE": str(queue)}, dry=dry, log="twice"
        )
        assert done.returncode == 0, (args, done.stderr)


def test_the_stage_task_and_the_last_job_are_told_what_they_are():
    code = code_of(SLURM / "stage.sbatch")
    assert 'args+=(--flat "$((offset + task))" --layout "$layout")' in code
    assert 'args+=(--index "$task")' in code
    assert '--stage "$stage" --attempt "$attempt"' in code
    assert 'if [ "$blocked" != 0 ]; then' in code  # the second lock behind the gate job
    assert code.rstrip().endswith("exit 0")
    assert "sbatch " not in code and "--wait" not in code  # a tile job submits none
    rung = code_of(SLURM / "run_rung.sbatch")
    assert 'stages="${14:--}"' in rung
    assert '[ "$stages" = - ] || export REGBENCH_STAGES="$stages"' in rung
    # ... and it reaches the process inside the image
    assert re.search(r"for v in [^;]*\bREGBENCH_STAGES\b", code_of(SLURM / "env.sh"))
    launcher = code_of(SCRIPT)
    assert "sbatch --wait" not in launcher
    assert ': "${REGBENCH_STAGE_ARRAY_MAX:=100}"' in launcher
    assert ': "${REGBENCH_STAGE_QUEUE_MAX:=100}"' in launcher


# ── the realcheck as a gate ───────────────────────────────────────────────────
def test_a_realcheck_gated_run_sends_one_unit_ahead(dry):
    jobs, _, _ = dry
    gated = {n: variants.realcheck_gate(n) for n in RUNS if variants.realcheck_gate(n)}
    assert gated == {"valis_mirage": "valis_mirage", "valis120_hires": "valis120_hires"}
    for run, variant in gated.items():
        first = rungs(jobs, run, suffix="-first")
        assert [j.opt("--array") for j in first] == ["0-0%20"] * 4
        assert all(j.args[9] == "1" for j in first)  # --realcheck
        gate = jobs[f"rb-realgate-{run}"]
        assert gate.args[:2] == ["realcheck", run] and gate.args[4:6] == [
            "multiplex",
            variant,
        ]
        assert gate.deps == {("afterany", f"<rb-{run}-multiplex-first-512g-again>")}
        for dataset in ("multiplex", "synthetic"):
            for job in rungs(jobs, run, dataset):
                assert ("afterok", f"<rb-realgate-{run}>") in job.deps
                assert job.args[9] == "0"
    others = [
        j
        for j in jobs.values()
        if j.script == "run_rung.sbatch" and j.args[0] not in gated
    ]
    assert others and all(j.args[9] == "0" for j in others)


# ── GPUs and images ───────────────────────────────────────────────────────────
def test_a_gpu_is_asked_for_only_where_the_run_list_says(dry):
    jobs, _, _ = dry
    with_gpu = {j.args[0] for j in jobs.values() if GPU in j.words}
    assert with_gpu == {n for n, r in RUNS.items() if r.gpu} and len(with_gpu) == 4
    for job in jobs.values():
        if job.script in ("run_rung.sbatch", "selfcheck.sbatch"):
            assert job.args[5] == ("1" if RUNS[job.args[0]].gpu else "0")
            assert (GPU in job.words) is RUNS[job.args[0]].gpu, job.name


def test_gpu_runs_are_refused_without_a_request_string(site):
    done, log, _ = launch(site, env={"REGBENCH_SBATCH_GPU": ""}, log="nogpu")
    assert done.returncode == 2 and not _sbatch_lines(done.stdout) and not log.exists()
    assert "REGBENCH_SBATCH_GPU is empty" in done.stderr
    for run, spec in RUNS.items():
        assert (
            f" {run}" in done.stderr.split("is empty:")[1].split(".")[0]
        ) is spec.gpu
    assert "Nothing was submitted" in done.stderr
    # the same refusal on the real path, and the runs that need none can still go
    real, log, _ = launch(
        site, env={"REGBENCH_SBATCH_GPU": ""}, dry=False, log="nogpu2"
    )
    assert real.returncode == 2 and not log.exists()
    cpu = " ".join(n for n, r in RUNS.items() if not r.gpu)
    ok, _, _ = launch(site, "--runs", cpu, env={"REGBENCH_SBATCH_GPU": ""}, log="cpu")
    assert ok.returncode == 0 and GPU not in ok.stdout


def _exec_line(tmp_path, image, **env):
    """What `regbench_py` starts for an image, with a stand-in container runtime."""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    runtime = fake / "apptainer"
    runtime.write_text('#!/bin/bash\necho "apptainer $*"\n')
    runtime.chmod(0o755)
    base = {k: v for k, v in os.environ.items() if not k.startswith("REGBENCH_")}
    base.pop("CUDA_VISIBLE_DEVICES", None)
    base.update(
        PATH=f"{fake}:{base['PATH']}",
        REGBENCH_REPO=str(REPO),
        REGBENCH_CASES="/c",
        REGBENCH_OUT="/o",
        **{f"REGBENCH_SIF_{image}": f"/images/{image}.sif"},
        **env,
    )
    done = subprocess.run(
        ["bash", "-c", f'source "{SLURM}/env.sh" && regbench_py {image} runs'],
        env=base, capture_output=True, text=True,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    return done.stdout


def _inside(tmp_path, image, **env):
    """What reaches the inside of an image: the runtime's own arguments, and every
    variable the launcher sets for it (`APPTAINERENV_<name>`), with a stand-in runtime
    that prints both."""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    runtime = fake / "apptainer"
    runtime.write_text(
        '#!/bin/bash\necho "ARGS $*"\nenv | grep "^APPTAINERENV_" | sort\n'
    )
    runtime.chmod(0o755)
    base = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("REGBENCH_", "APPTAINERENV_", "SINGULARITYENV_"))
    }
    base.pop("CUDA_VISIBLE_DEVICES", None)
    base.update(
        PATH=f"{fake}:{base['PATH']}",
        REGBENCH_REPO=str(REPO),
        REGBENCH_CASES="/c",
        REGBENCH_OUT="/o",
        PYTHONPATH="/home/someone/my-own-packages",
        **{f"REGBENCH_SIF_{image}": f"/images/{image}.sif"},
        **env,
    )
    done = subprocess.run(
        ["bash", "-c", f'source "{SLURM}/env.sh" && regbench_py {image} runs'],
        env=base, capture_output=True, text=True,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    inside = dict(
        ln[len("APPTAINERENV_") :].split("=", 1) for ln in lines if "=" in ln[:60]
    )
    return lines[0], inside


def test_no_package_of_the_host_can_shadow_one_inside_an_image(tmp_path):
    """PYTHONNOUSERSITE=1 in every image, and a PYTHONPATH that holds the checkout (and
    the STARE source where the image has none) and nothing of the host's."""
    images = {r.image for r in variants.RUNS} | {variants.SCORER_IMAGE}
    without = {"ASHLAR", "VALIS_MIRAGE"}  # pulled as their projects publish them
    assert without <= images
    tag = re.search(
        r"tags/([^\s]+)\.tar\.gz", (REPO / "requirements" / "stare.txt").read_text()
    )[1]
    for image in sorted(images):
        args, inside = _inside(tmp_path, image)
        assert "--cleanenv" in args
        assert inside["PYTHONNOUSERSITE"] == "1", image
        want = str(REPO)
        if image in without:
            want += f":/o/cache/stare-{tag}/src"
        assert inside["PYTHONPATH"] == want, image
        assert "my-own-packages" not in "".join(inside.values())
    # the same switch for an environment that is not an image
    code = code_of(SLURM / "env.sh")
    assert code.count('PYTHONNOUSERSITE=1 "$@"') == 1
    assert 'PYTHONNOUSERSITE=1 PATH="$venv/bin:$PATH" "$@"' in code


def test_the_work_folder_the_temp_folder_and_the_plugin_switch_reach_the_image(
    tmp_path,
):
    scratch, private = tmp_path / "scratch", tmp_path / "private"
    scratch.mkdir()
    args, inside = _inside(
        tmp_path,
        "PALOM",
        TMPDIR=f"{scratch}/",
        REGBENCH_WORK="/work",
        REGBENCH_PRIVATE="1",
        REGBENCH_PRIVATE_DIR=str(private),
    )
    assert inside["TMPDIR"] == f"{scratch}/" and inside["REGBENCH_WORK"] == "/work"
    assert inside["REGBENCH_PRIVATE"] == "1"
    assert inside["REGBENCH_PRIVATE_DIR"] == str(private)
    bound = args.split("--bind ", 1)[1].split()[0].split(",")
    assert {"/work", str(scratch), str(private), "/c", "/o", str(REPO)} <= set(bound)
    # unset, none of them is invented: the plug-in stays off
    args, inside = _inside(tmp_path, "PALOM", TMPDIR="/no/such/folder")
    assert "REGBENCH_PRIVATE" not in inside and "REGBENCH_WORK" not in inside
    assert "/no/such/folder" not in args.split("--bind ", 1)[1].split()[0]


def test_the_pinned_stare_source_is_fetched_once_for_the_images_without_one(tmp_path):
    """`regbench_fetch_stare`, with a stand-in `curl` that serves an archive laid out
    as the release's: the source of the pinned tag, under the results folder's cache."""
    import tarfile

    release = tmp_path / "stare-9.9.9" / "src" / "stare"
    release.mkdir(parents=True)
    (release / "slide_io.py").write_text("WHAT = 'the pinned source'\n")
    (release / "__init__.py").write_text("")
    archive = tmp_path / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tmp_path / "stare-9.9.9", arcname="stare-9.9.9")
    fake = tmp_path / "bin"
    fake.mkdir()
    curl = fake / "curl"
    curl.write_text(
        f'#!/bin/bash\necho "$*" >> "{tmp_path}/curl.log"\ncat "{archive}"\n'
    )
    curl.chmod(0o755)
    out = tmp_path / "out"
    env = {k: v for k, v in os.environ.items() if not k.startswith("REGBENCH_")}
    env.update(
        PATH=f"{fake}:{env['PATH']}", REGBENCH_REPO=str(REPO), REGBENCH_OUT=str(out)
    )
    call = f'source "{SLURM}/env.sh" && regbench_fetch_stare && regbench_stare_src'
    url = re.search(
        r"https://github\.com/sceriff0/stare/\S+\.tar\.gz",
        (REPO / "requirements" / "stare.txt").read_text(),
    )[0]
    for _ in (1, 2):
        done = subprocess.run(
            ["bash", "-c", call], env=env, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr
        first, second = done.stdout.split()
        assert first == second and Path(first).parent.parent == out / "cache"
        assert Path(first, "stare", "slide_io.py").read_text().startswith("WHAT")
    asked = (tmp_path / "curl.log").read_text().splitlines()
    assert len(asked) == 1 and asked[0].endswith(url)  # once, the pinned release
    # nothing fetched, nothing there: said, and non-zero
    curl.write_text("#!/bin/bash\nexit 22\n")
    env["REGBENCH_OUT"] = str(tmp_path / "other")
    done = subprocess.run(["bash", "-c", call], env=env, capture_output=True, text=True)
    assert done.returncode != 0 and "could not fetch STARE from" in done.stderr


# ── the preflight: before anything is submitted ───────────────────────────────
def test_the_preflight_stops_the_launch_with_one_line_per_failure(site, tmp_path):
    """The stand-in "image" is this machine's python: it has STARE and not PALOM."""
    env = {
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "REGBENCH_PREFLIGHT": "1",
        "REGBENCH_WORK": str(tmp_path / "work"),
    }
    bad, log, out = launch(
        site, "--runs", "stare palom", env=env, dry=False, log="preflight-bad"
    )
    assert bad.returncode == 2 and not log.exists()  # no sbatch was called
    lines = [ln for ln in bad.stdout.splitlines() if ln.startswith("preflight")]
    fails = [ln for ln in lines if ln.startswith("preflight FAIL")]
    assert fails == [
        ln
        for ln in fails
        if ln.startswith("preflight FAIL [PALOM palom] cannot import")
    ] and any("cannot import palom:" in ln for ln in fails)
    assert "preflight [STARE_REGBENCH stare] ok" in lines
    assert any(
        ln.startswith("preflight [PALOM palom] ") and "failure(s)" in ln for ln in lines
    )
    assert (
        "the preflight failed" in bad.stderr and "Nothing was submitted." in bad.stderr
    )
    assert not (out / "logs" / "jobs.tsv").exists()
    # a launch the preflight stopped is not "the last submission" of the folder
    assert not (out / "logs" / "submission").exists()
    # the same run list without the image that fails: the launch goes on
    good, log, out = launch(
        site, "--runs", "stare", env=env, dry=False, log="preflight-ok"
    )
    assert good.returncode == 0, good.stderr
    assert (out / "logs" / "submission").read_text().strip()
    assert "# preflight: ok" in good.stdout and log.exists()
    assert good.stdout.index("# preflight: ok") < good.stdout.index("rb-sc-stare-64g")
    assert (tmp_path / "work").is_dir()  # the work folder was written to
    # alone: the same checks, nothing submitted, no submission id left behind
    alone, log, out = launch(
        site,
        "--preflight",
        "--runs",
        "stare",
        env=env,
        dry=False,
        log="preflight-alone",
    )
    assert alone.returncode == 0 and not log.exists()
    assert "# --preflight: nothing was submitted" in alone.stdout
    assert not (out / "logs" / "submission").exists()
    # a dry run does not start an image, and says so; skipping is said aloud
    dry_run, _, _ = launch(site, "--runs", "stare", env=env, log="preflight-dry")
    assert "# preflight: not run in a dry run" in dry_run.stdout
    skipped, _, _ = launch(
        site, "--runs", "palom", env={**env, "REGBENCH_PREFLIGHT": "0"}, dry=False,
        log="preflight-skip",
    )  # fmt: skip
    assert skipped.returncode == 0 and "# WARNING preflight SKIPPED" in skipped.stdout


def test_an_image_that_does_not_start_is_one_preflight_failure(site, tmp_path):
    fake = _image_runtime(tmp_path, ())  # an image with no interpreter at all
    env = {
        "REGBENCH_PREFLIGHT": "1",
        "REGBENCH_SIF_PALOM": "/images/palom.sif",
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "PATH": f"{fake}:{site[2]}:{Path(sys.executable).parent}:{os.environ['PATH']}",
    }
    done, log, _ = launch(
        site, "--preflight", "--runs", "palom", env=env, dry=False, log="noimage"
    )
    assert done.returncode == 2 and not log.exists()
    fails = [ln for ln in done.stdout.splitlines() if ln.startswith("preflight FAIL")]
    assert len(fails) == 1 and fails[0].startswith(
        "preflight FAIL [PALOM palom] the image did not start, or has neither python nor python3"
    )


def test_valis_120_is_started_with_a_writable_tmpfs_and_no_other_image(tmp_path):
    images = {r.image for r in variants.RUNS} | {variants.SCORER_IMAGE}
    assert "VALIS120" in images
    for image in sorted(images):
        line = _exec_line(tmp_path, image)
        # the interpreter is chosen inside the image (`sh -c`), never named from outside
        assert f"/images/{image}.sif sh -c " in line
        assert line.rstrip().endswith("regbench-python -m benchmarks.regbench runs")
        assert ("--writable-tmpfs" in line) is (image == "VALIS120"), image


def _image_runtime(tmp_path, commands):
    """A stand-in container runtime whose "image" has exactly these commands: what
    follows the image path is run with PATH set to the image's bin and nothing else."""
    fake, image = tmp_path / "bin", tmp_path / "image-bin"
    fake.mkdir(exist_ok=True)
    image.mkdir(exist_ok=True)
    (image / "sh").symlink_to("/bin/sh")
    for name in commands:
        tool = image / name
        tool.write_text(f'#!/bin/sh\necho "{name} $*"\n')
        tool.chmod(0o755)
    runtime = fake / "apptainer"
    runtime.write_text(
        "#!/bin/bash\n"
        'while [ $# -gt 0 ]; do a="$1"; shift; case "$a" in *.sif) break ;; esac; done\n'
        f'PATH="{image}" exec "$@"\n'
    )
    runtime.chmod(0o755)
    return fake


@pytest.mark.parametrize(
    "commands, wanted",
    [
        (("python3",), "python3"),  # labsyspharm/ashlar:1.20.0: Ubuntu, python3 only
        (("python", "python3"), "python"),
        (("python",), "python"),
    ],
)
def test_the_interpreter_is_the_images_own_python_or_python3(
    tmp_path, commands, wanted
):
    """`regbench_py` must not ask an image for a `python` it does not have: ASHLAR's
    official image has `python3` alone, and its selfcheck died on exactly that."""
    fake = _image_runtime(tmp_path, commands)
    base = {k: v for k, v in os.environ.items() if not k.startswith("REGBENCH_")}
    base.update(
        PATH=f"{fake}:{base['PATH']}",
        REGBENCH_REPO=str(REPO),
        REGBENCH_CASES="/c",
        REGBENCH_OUT="/o",
        REGBENCH_SIF_ASHLAR="/images/ASHLAR.sif",
    )
    done = subprocess.run(
        ["bash", "-c", f'source "{SLURM}/env.sh" && regbench_py ASHLAR selfcheck --x "a b"'],
        env=base, capture_output=True, text=True,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert done.stdout == f"{wanted} -m benchmarks.regbench selfcheck --x a b\n"


def test_an_image_with_no_interpreter_fails_and_says_so(tmp_path):
    fake = _image_runtime(tmp_path, ())
    base = {k: v for k, v in os.environ.items() if not k.startswith("REGBENCH_")}
    base.update(
        PATH=f"{fake}:{base['PATH']}", REGBENCH_REPO=str(REPO), REGBENCH_CASES="/c",
        REGBENCH_OUT="/o", REGBENCH_SIF_ASHLAR="/images/ASHLAR.sif",
    )  # fmt: skip
    done = subprocess.run(
        ["bash", "-c", f'source "{SLURM}/env.sh" && regbench_py ASHLAR runs'],
        env=base, capture_output=True, text=True,
    )  # fmt: skip
    assert done.returncode != 0 and "python3" in done.stderr


def test_the_stardist_command_also_falls_back_to_python3():
    """nuclei.sbatch runs the pipeline's `python segment.py ...` line inside the
    StarDist image through a script: the same rule, as a shell function there."""
    code = code_of(SLURM / "nuclei.sbatch")
    assert 'command -v python >/dev/null 2>&1 || python() { python3 "$@"; }' in code
    script = 'command -v python >/dev/null 2>&1 || python() { python3 "$@"; }\npython x.py 1\n'
    done = subprocess.run(
        ["/bin/sh", "-c", script], capture_output=True, text=True,
        env={"PATH": "/nonexistent"},
    )  # fmt: skip
    assert "python3" in done.stderr and done.returncode != 0  # it asked for python3


def test_a_gpu_reaches_an_image_only_when_the_run_asked_for_one(tmp_path):
    for image in ("DEEPERHISTREG", "VALIS120"):
        assert "--nv" in _exec_line(tmp_path, image, REGBENCH_GPU="1")
        # a device visible in the job is not a reason: the run list decides
        hidden = _exec_line(tmp_path, image, REGBENCH_GPU="0", CUDA_VISIBLE_DEVICES="0")
        assert "--nv" not in hidden
    assert 'REGBENCH_GPU="$gpu"' in code_of(SLURM / "run_rung.sbatch")


def test_every_image_of_the_run_list_needs_its_path_before_anything_is_submitted(site):
    done, log, _ = launch(site, dry=False, log="noimages")
    assert done.returncode == 2 and not log.exists()
    for image in {r.image for r in RUNS.values()}:
        assert f"REGBENCH_SIF_{image}" in done.stderr
    # the dry run goes on and says so
    dry_run, _, _ = launch(site, log="noimages-dry")
    assert dry_run.returncode == 0 and "# not set: REGBENCH_SIF_" in dry_run.stdout


# ── the real path, through a stand-in sbatch ──────────────────────────────────
def test_the_real_path_submits_the_same_graph_with_job_ids(site, dry):
    jobs, _, _ = dry
    done, log, out = launch(
        site, dry=False, log="real", env={"REGBENCH_ALLOW_PATH_PYTHON": "1"}
    )
    assert done.returncode == 0, done.stderr
    calls = [ln.split("\t", 1) for ln in log.read_text().splitlines()]
    assert len(calls) == len(jobs)
    ids = {}
    for (job_id, argv), wanted in zip(calls, jobs.values()):
        real = Job("sbatch " + argv)
        assert real.name == wanted.name
        ids[job_id] = real.name
        # every dependency is the id sbatch returned for the job the dry run named
        assert {(k, f"<{ids[n]}>") for k, n in real.deps} == wanted.deps, real.name
    recorded = (out / "logs" / "jobs.tsv").read_text().splitlines()
    assert [ln.split("\t")[1:] for ln in recorded] == [[n, i] for i, n in ids.items()]
    assert f"# {len(jobs)} job(s) submitted" in done.stdout


# ── subsets, prepare, and what is never exported ──────────────────────────────
def test_a_subset_brings_the_selfchecks_it_stands_behind(site):
    done, _, _ = launch(site, "--runs", "affine_ashlar_clean", log="subset")
    jobs = jobs_of(done)
    assert {j.args[0] for j in jobs.values() if j.script == "run_rung.sbatch"} == {
        "affine_ashlar_clean"
    }
    checks = {j.args[0] for j in jobs.values() if j.script == "selfcheck.sbatch"}
    assert checks == {"affine_ashlar", "stare"}  # its gate, and that gate's anchor
    assert "reads the results of stare, which is not submitted here" in done.stdout
    bad, _, _ = launch(site, "--runs", "nope", log="subset-bad")
    assert bad.returncode == 2 and "no run named 'nope'" in bad.stderr


def test_unprepared_cases_are_refused(site, tmp_path):
    done, log, _ = launch(site, env={"REGBENCH_CASES": str(tmp_path)}, log="nocases")
    assert done.returncode == 2 and "Run with --prepare first" in done.stderr
    assert not _sbatch_lines(done.stdout) and not log.exists()


def test_prepare_is_a_step_of_its_own(site, tmp_path):
    env = {
        "REGBENCH_MANIFEST": "/data/rounds.csv",
        "REGBENCH_TRUTH_PATIENTS": "P1,P2",
        # the bare-nuclei step has its own test below
        "REGBENCH_NUCLEI_SKIP": "1",
    }
    done, _, _ = launch(site, "--prepare", env=env, log="prepare")
    jobs = jobs_of(done)
    assert set(jobs) == {"rb-prepare-multiplex", "rb-prepare-synthetic"}
    named = "multiplex semisynth synthetic"
    done, _, _ = launch(site, "--prepare", "--datasets", named, env=env, log="prep3")
    jobs = jobs_of(done)
    assert set(jobs) == {
        "rb-prepare-multiplex",
        "rb-prepare-semisynth",
        "rb-prepare-synthetic",
    }
    assert jobs["rb-prepare-multiplex"].args == [
        "multiplex",
        "--manifest",
        "/data/rounds.csv",
    ]
    assert jobs["rb-prepare-semisynth"].args == [
        "semisynth", "--truth", "--manifest", "/data/rounds.csv", "--patients", "P1,P2",
    ]  # fmt: skip
    assert jobs["rb-prepare-synthetic"].args == ["synthetic", "--suite", "truth"]
    assert "run this script again without --prepare" in done.stdout
    none, _, _ = launch(site, "--prepare", log="prepare-bad")
    assert none.returncode == 2 and "REGBENCH_MANIFEST" in none.stderr
    assert not _sbatch_lines(none.stdout)


def _manifest(tmp_path):
    path = tmp_path / "rounds.csv"
    path.write_text(
        "case_id,reference,moving,group\n"
        "P1_b,/s/P1.a.ome.tif,/s/P1.b.ome.tif,P1\n"
        "P1_c,/s/P1.a.ome.tif,/s/P1.c.ome.tif,P1\n"
    )
    return path


def test_prepare_resolves_the_bare_nuclei_of_every_slide_first(site, tmp_path):
    """Ruling R3.18: an array over the manifest's SLIDES (three here: one reference, two
    rounds) checks what exists and re-runs StarDist where there is nothing, and `prepare
    multiplex` waits for it and reads what it left."""
    env = {
        "REGBENCH_MANIFEST": str(_manifest(tmp_path)),
        "REGBENCH_STARDIST_MODEL": "site_model",
        "REGBENCH_STARDIST_MODEL_DIR": "/models",
    }
    done, log, _ = launch(
        site, "--prepare", "--datasets", "multiplex", env=env, log="nuc"
    )
    jobs = jobs_of(done)
    assert list(jobs) == ["rb-nuclei", "rb-prepare-multiplex"]  # in this order
    nuc, prep = jobs["rb-nuclei"], jobs["rb-prepare-multiplex"]
    assert nuc.script == "nuclei.sbatch" and nuc.args == []
    assert nuc.opt("--array") == "0-2%20"
    assert (nuc.opt("--cpus-per-task"), nuc.opt("--mem")) == ("8", "128G")
    assert GPU in nuc.line  # StarDist asks for the GPU the arms had
    assert prep.deps == {("afterany", "<rb-nuclei>")}
    nuclei_dir = str(site[1] / "_nuclei")
    assert prep.args == [
        "multiplex", "--manifest", env["REGBENCH_MANIFEST"], "--nuclei-dir", nuclei_dir,
    ]  # fmt: skip
    assert "REGBENCH_SIF_STARDIST" in done.stdout  # named, since it is not set
    assert not log.exists()
    # the dry run says where the model and each setting came from
    said = "# StarDist re-run: segmentation_model\tsite_model\tenvironment"
    assert said in done.stdout
    assert "# StarDist re-run: seg_pmax\t99.8\trepository default" in done.stdout
    # from a site config when the two variables are not exported
    config = tmp_path / "site.config"
    config.write_text(
        "params {\n    segmentation_model     = 'site_net'\n"
        "    segmentation_model_dir = '/site/models'\n    seg_pmax = 99.5\n}\n"
    )
    from_site = {
        "REGBENCH_MANIFEST": env["REGBENCH_MANIFEST"],
        "REGBENCH_SITE_CONFIG": str(config),
    }
    done, _, _ = launch(
        site, "--prepare", "--datasets", "multiplex", env=from_site, log="nuc1"
    )
    assert "rb-nuclei" in jobs_of(done)
    where = f"site config ({config})"
    assert f"# StarDist re-run: segmentation_model\tsite_net\t{where}" in done.stdout
    assert f"# StarDist re-run: seg_pmax\t99.5\t{where} [differs" in done.stdout
    # on CPUs when asked; the shrunk outline only when asked
    more = {**env, "REGBENCH_STARDIST_CPU": "1", "REGBENCH_NUCLEI_FALLBACK": "shrink"}
    done, _, _ = launch(
        site, "--prepare", "--datasets", "multiplex", env=more, log="nuc2"
    )
    jobs = jobs_of(done)
    assert GPU not in jobs["rb-nuclei"].line
    assert jobs["rb-prepare-multiplex"].args[-2:] == ["--nuclei-fallback", "shrink"]


def test_prepare_leaves_the_manifest_where_the_score_job_finds_it(site, tmp_path):
    """`attrition.csv: manifest_rounds` needs the manifest in the score job, which is
    submitted by a later call that has no REGBENCH_MANIFEST."""
    cases = tmp_path / "cases"
    manifest = _manifest(tmp_path)
    env = {
        "REGBENCH_MANIFEST": str(manifest),
        "REGBENCH_CASES": str(cases),
        "REGBENCH_NUCLEI_SKIP": "1",
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
    }
    done, _, _ = launch(
        site, "--prepare", "--datasets", "multiplex", env=env, dry=False, log="keep"
    )
    assert done.returncode == 0, done.stderr
    assert (cases / "manifest.csv").read_text() == manifest.read_text()
    # the score step: the variable when it is there, else that copy
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "python").write_text('#!/bin/bash\necho "$*" >> "$FAKE/python.log"\n')
    (fake / "python").chmod(0o755)
    base = {
        k: v for k, v in os.environ.items() if not k.startswith(("REGBENCH_", "SLURM_"))
    }
    base.update(
        PATH=f"{fake}:/usr/bin:/bin", FAKE=str(fake), REGBENCH_REPO=str(REPO),
        REGBENCH_CASES=str(cases), REGBENCH_OUT=str(out),
        REGBENCH_SCORER_IMAGE="STARE_REGBENCH", REGBENCH_ALLOW_PATH_PYTHON="1",
    )  # fmt: skip
    subprocess.run(
        ["bash", str(SLURM / "score_cohort.sbatch"), "score"],
        env=base, capture_output=True, text=True, cwd=REPO, timeout=120,
    )  # fmt: skip
    (call,) = [
        ln for ln in (fake / "python.log").read_text().splitlines() if " score " in ln
    ]
    assert f"--manifest {out}/logs/manifest.csv" in call
    assert (out / "logs" / "manifest.csv").read_text() == manifest.read_text()


def test_the_nuclei_step_is_refused_without_its_model_or_its_gpu(site, tmp_path):
    env = {"REGBENCH_MANIFEST": str(_manifest(tmp_path))}
    none, _, _ = launch(site, "--prepare", "--datasets", "multiplex", env=env, log="n3")
    assert none.returncode == 2 and "REGBENCH_STARDIST_MODEL" in none.stderr
    assert "REGBENCH_NUCLEI_SKIP=1" in none.stderr and not _sbatch_lines(none.stdout)
    env["REGBENCH_STARDIST_MODEL"] = "site_model"
    no_gpu, _, _ = launch(
        site,
        "--prepare",
        "--datasets",
        "multiplex",
        env={**env, "REGBENCH_SBATCH_GPU": ""},
        log="n4",
    )
    assert no_gpu.returncode == 2 and "REGBENCH_STARDIST_CPU=1" in no_gpu.stderr
    assert not _sbatch_lines(no_gpu.stdout)
    # a real submission also needs the image
    real, log, _ = launch(
        site, "--prepare", "--datasets", "multiplex", env=env, dry=False, log="n5"
    )
    assert real.returncode == 2 and "REGBENCH_SIF_STARDIST" in real.stderr
    assert not log.exists()


def test_the_nuclei_task_runs_stardist_only_for_a_slide_that_needs_it():
    """A guard on nuclei.sbatch, comments stripped: it checks what exists first, asks
    for the command, stops when there is none, and runs it in the StarDist image."""
    lines = [
        ln.split(" #")[0].strip()
        for ln in (SLURM / "nuclei.sbatch").read_text().splitlines()
        if not ln.lstrip().startswith("#")
    ]
    code = "\n".join(ln for ln in lines if ln)
    # the first check may exit 1 (a refused re-run): the task still asks what is needed
    first = code.index(
        'regbench_py "$scorer" "${what[@]}" ${force[@]+"${force[@]}"} || first=$?'
    )
    asked = code.index('line="$(regbench_py "$scorer" "${what[@]}" --command')
    stop = code.index('if [ -z "$line" ]; then')
    run = code.index('regbench_exec STARDIST bash "$script"')
    check = code.rindex('regbench_py "$scorer" "${what[@]}"')
    assert first < asked < stop < run < check
    assert "exit 0" in code[stop:run] and 'exit "$first"' in code[stop:run]
    assert '[ "${REGBENCH_NUCLEI_FORCE:-0}" = 1 ] && force=(--force)' in code[:first]
    assert '--index "$SLURM_ARRAY_TASK_ID"' in code


def test_no_thread_counts_are_exported():
    for path in [*SLURM.glob("*.sh"), *SLURM.glob("*.sbatch")]:
        text = path.read_text()
        for name in THREADS:
            assert name not in text, (path.name, name)


def test_extra_sbatch_arguments_reach_every_job(site):
    done, _, _ = launch(
        site, "--", "--partition=cpu", "--constraint=icelake", log="extra"
    )
    jobs = jobs_of(done)
    assert all(j.opt("--partition") == "cpu" for j in jobs.values())
    assert all(j.opt("--constraint") == "icelake" for j in jobs.values())
