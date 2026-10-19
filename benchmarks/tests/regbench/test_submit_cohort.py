"""submit_cohort.sh: the job graph of the cohort, read off its dry run.

The dry run prints every ``sbatch`` command with a job name in angle brackets where a job
id would be, and submits nothing. A stand-in ``sbatch`` on PATH records every call, so
"submits nothing" is observed, and one test runs the real path through it.

The fixture is two patients: P1 with three rounds, P2 with one (four cases).
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.regbench import ladder, resources, stages, variants
from benchmarks.regbench.cases import Case, write_json

REPO = Path(__file__).resolve().parents[3]
SLURM = REPO / "benchmarks" / "regbench" / "slurm"
SCRIPT = SLURM / "submit_cohort.sh"
GPU = "--gres=gpu:nvidia_h200:1"
# the cohort's list: its regime's runs (the bf-only STARE runs are not submitted for it)
RUNS = {variants.run_name(r): r for r in variants.runs_for(None)}
# the runs whose selfcheck checks a run (reported, never a gate)
CHECKERS = sorted({variants.checked_by(n) for n in RUNS})
# the runs whose registration is several jobs per round: their last job (the writer) has
# the four attempts of the pipeline's allocation for STITCH where the others have the
# memory ladder
STAGED = {n: variants.stages_of(r) for n, r in RUNS.items() if variants.stages_of(r)}
# every run, whatever its dataset
STAGED_ALL = {
    variants.run_name(r): variants.stages_of(r)
    for r in variants.RUNS
    if variants.stages_of(r)
}
FEEDERS = 2  # tile submitters chained per staged run and dataset
# links of each chain's watcher (REGBENCH_RETRY_WATCHERS): the first after the chain's
# first job, each later one after the link before it, so a watcher ended at its time
# limit or lost with its node is taken over by its successor
LINKS = 2
PAIR = 1 + LINKS  # a chain's first job and its watcher's links
# the default submission on the fixture (the cohort, `multiplex`),
# derived from the run list: per run and dataset the ladder's first array and its watcher;
# per distinct checker a selfcheck and its watcher; per run with a realcheck the ladder of
# its first unit (2 jobs, beside the rest); the settle job, an array per dataset, tissue
# and the scorer; and
# per staged run the stages' extra jobs per dataset (two feeders, the last job's pair, and
# the preprocessing's pair where it has one)
COHORT_DATASETS = ("multiplex",)
MORE = sum(PAIR + FEEDERS + PAIR + (PAIR if "prep" in st else 0) for st in STAGED.values())
N_REALCHECKED = len([n for n in RUNS if variants.realcheck_row(n)])
N_JOBS = (
    PAIR * len(CHECKERS)
    + len(RUNS) * PAIR * len(COHORT_DATASETS)
    + PAIR * N_REALCHECKED
    + 1
    + len(COHORT_DATASETS)
    + 2
    + 1  # rb-rescore, after the scorer and every selfcheck (Task 28, B5)
    + len(COHORT_DATASETS) * MORE
)
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


# every fixture slide: 2 x 3 tiles of 2048 px, so a round is six REG-TILE jobs at the
# high tier (`TILES_PER_ROUND` for a run at another tier)
REF_HW = (4096, 6144)
TILES = 6


# the tier runs' REG-TILE jobs per round on that slide, written out (STARE's tile plan
# at 1024 / 192 and 512 / 128 px): every other STARE run is the high tier's TILES
TILES_PER_ROUND = {"stare": TILES, "stare_medium": 24, "stare_low": 96}


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
def public_launcher(tmp_path_factory):
    """The launcher with the two TEST-ONLY public datasets of ``testpublic`` (``nucset``,
    ``tissueset``) added to its ``PUBLIC_DATASETS`` and ``PUBLIC_KINDS``, as ``(script,
    env)``: ``env`` makes a python the launcher starts know them too. The real lists are
    held equal to ``regbench.PUBLIC`` by the tests below, on the real script."""
    from . import testpublic

    return testpublic.patched_launcher(tmp_path_factory.mktemp("public_launcher"))


def launch_public(site, public_launcher, *args, env=None, **kw):
    script, more = public_launcher
    return launch(site, *args, env={**more, **(env or {})}, script=script, **kw)


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


def launch(site, *args, env=None, dry=True, log="calls", script=None):
    argv, base, out = _launch_command(site, args, env, dry, log, script)
    done = subprocess.run(argv, env=base, capture_output=True, text=True, cwd=REPO)
    return done, Path(base["SBATCH_LOG"]), out


def _launch_command(site, args, env, dry, log, script=None):
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
    argv = ["bash", str(script or SCRIPT), *(["--dry-run"] if dry else []), *args]
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


class Jobs(dict):
    """The jobs a dry run would submit, by name. ``later``: the retry steps it prints
    and does not submit (the plans of the chains' watchers)."""

    later: dict


def jobs_of(done):
    assert done.returncode == 0, done.stderr + done.stdout
    jobs = Jobs({j.name: j for j in map(Job, _sbatch_lines(done.stdout))})
    jobs.later = later_of(done)
    return jobs


def _any(jobs, name):
    """A job by name, submitted or planned for later."""
    return jobs[name] if name in jobs else getattr(jobs, "later", {})[name]


LATER = "# later, only after a failure: "


def _sbatch_lines(text):
    return [ln for ln in text.splitlines() if ln.startswith("sbatch ")]


def later_of(done):
    """The retry steps a dry run prints and does NOT submit: the plans of the chains'
    watchers (retry.sbatch), by job name."""
    lines = [
        ln[len(LATER) :] for ln in done.stdout.splitlines() if ln.startswith(LATER)
    ]
    return {j.name: j for j in map(Job, lines)}


@pytest.fixture(scope="module")
def dry(site):
    done, log, _ = launch(site)
    return jobs_of(done), done, log


def rung_gb(run):
    return ladder.RUNGS_GB


def _last(run, suffix):
    return suffix + ("-last" if run in STAGED_ALL else "")


def rungs(jobs, run, dataset="multiplex", suffix=""):
    """The attempt of each rung (the `-again` arrays are `agains`): the first is
    submitted, the others are steps of the watcher's plan. For a run that is several
    jobs per round: the rungs of its last job."""
    tail = _last(run, suffix)
    return [_any(jobs, f"rb-{run}-{dataset}{tail}-{gb}g") for gb in rung_gb(run)]


def agains(jobs, run, dataset="multiplex", suffix=""):
    tail = _last(run, suffix)
    return [_any(jobs, f"rb-{run}-{dataset}{tail}-{gb}g-again") for gb in rung_gb(run)]


def last_job(run, dataset="multiplex"):
    """The name of the last job a run submits on a dataset: what a run that reads its
    results, the next dataset, the next run of its lane and the settle job wait for."""
    return f"rb-{run}-{dataset}{_last(run, '')}-{retry_link(LINKS)}"


def retry_link(n):
    """The name tail of a watcher's link `n` (1 is `retry`, a successor `retry-<n>`)."""
    return "retry" if n == 1 else f"retry-{n}"


def links(jobs, prefix):
    """A chain's watcher, link by link: `<prefix>-retry`, `<prefix>-retry-2`, ...
    Link 1 waits for the chain's first job, each later link for the link before it,
    and each is told its number (the 6th argument)."""
    got = [jobs[f"{prefix}-{retry_link(n)}"] for n in range(1, LINKS + 1)]
    for n, link in enumerate(got, 1):
        assert link.script == "retry.sbatch" and link.args[5] == str(n), link.name
        if n > 1:
            assert ("afterany", f"<{got[n - 2].name}>") in link.deps, link.name
            assert link.args[:5] == got[0].args[:5], link.name
    return got


def launch_jobs(spec):
    """How many jobs the launcher submits for one run on one dataset:
    the ladder's first array and its watcher, or for a run in several jobs per round
    the first attempt and the watcher of each stage that is an array over the rounds,
    its tile submitters, and the last job's first array and watcher. No retry is
    among them."""
    names = variants.stages_of(spec)
    if not names:
        return PAIR
    arrays = sum(st in names for st in ("prep", "coarse", "solve"))
    return PAIR * arrays + FEEDERS + PAIR


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
    # ONE kind of job submits jobs on its own: the tile submitter. launch.sbatch is the
    # launcher itself run as a job (it names sbatch in what it prints, and calls none)
    assert {p.name for p in SLURM.glob("*.sbatch") if "sbatch " in code_of(p)} == {
        "feed_tiles.sbatch",
        "launch.sbatch",
        "retry.sbatch",  # a chain's watcher: the retries, after a failure
    }
    launch_code = code_of(SLURM / "launch.sbatch")
    assert not re.search(r"(^|[;&|(]\s*)sbatch\s", launch_code, re.M)
    assert 'exec bash "$launcher" "$@"' in launch_code


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
    # containers/valis100, never the pipeline's bare upstream image (no zarr inside)
    assert (images / "bolt3x-mirage-valis100-1.0.0.img").exists()
    assert not any(p.name.startswith("cdgatenbee-valis-wsi") for p in images.iterdir())
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
        "submit_cohort.sh", "run_rung.sbatch", "selfcheck.sbatch",
        "score_cohort.sbatch",
    )]  # fmt: skip
    assert len(shell) == 4
    names = {r.method for r in variants.RUNS} | set(RUNS)
    for path in shell:
        code = code_of(path)
        for name in names:
            assert not re.search(rf"\b{re.escape(name)}\b", code), (path.name, name)
    assert 'regbench_py "$REGBENCH_HEAD_IMAGE" runs > "$table"' in code_of(SCRIPT)


def test_every_run_of_the_list_is_submitted_and_nothing_else(dry):
    jobs, _, _ = dry
    laddered = {j.args[0] for j in jobs.values() if j.script == "run_rung.sbatch"}
    assert laddered == set(RUNS) and len(RUNS) == len(variants.runs_for(None))
    checked = {j.args[0] for j in jobs.values() if j.script == "selfcheck.sbatch"}
    assert checked == set(CHECKERS)
    assert not [j for j in jobs.values() if j.script == "gate.sbatch"]


def test_a_run_the_list_leaves_out_is_not_submitted(site, monkeypatch, tmp_path):
    """The script submits what `runs` prints and nothing else."""
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
    done, _, _ = launch(site, env={"REGBENCH_REPO": str(tmp_path)}, log="left-out")
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
        assert gbs == (16, 64, 256, 512, 700), run
        assert [j.opt("--mem") for j in got] == [f"{gb}G" for gb in gbs], run
        assert [j.opt("--cpus-per-task") for j in got] == ["4"] * 5, run
        # attempt and memory reach the task: REGBENCH_ATTEMPT, REGBENCH_MEM_GB
        assert [j.args[7:9] for j in got] == [
            [str(i), str(gb)] for i, gb in enumerate(gbs, 1)
        ], run
        # only the last job of a run in several jobs is told to collect them
        # (stare_local4: its stages inside its one job, T31 R2)
        want = "jobs" if run in STAGED else "local" if run in variants.LOCAL_4CPU else "-"
        assert {j.args[13] for j in got} == {want}, run
        assert [j.args[11] for j in got] == ["first"] + ["climb"] * 4, run
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
            assert b.args[11] == "again"
            assert b.opt("--time") == a.opt("--time") == a.args[12] == b.args[12]
        seq = chain(jobs, run)
        # ONE array of the ladder is submitted. Every other one is a step of the plan:
        # no array (the watcher names the units that failed), no dependency (the
        # watcher waits for the step before), and the ids of the ladder's earlier
        # jobs are filled in by the watcher when it submits the step
        assert seq[0].name in jobs and seq[0].opt("--array")
        for step in seq[1:]:
            assert step.name in jobs.later and step.name not in jobs
            assert step.opt("--array") is None and step.opt("--dependency") is None
            assert step.args[10] == "@EARLIER@"
            assert "--no-requeue" in step.words
        watcher = links(jobs, last_job(run).rsplit("-retry", 1)[0])[0]
        assert jobs[last_job(run)] is not watcher
        assert watcher.script == "retry.sbatch" and watcher.opt("--array") is None
        assert ("afterany", f"<{seq[0].name}>") in watcher.deps
        # the watcher has the first array's own dependencies too
        assert {d for d in seq[0].deps} <= watcher.deps
        assert (watcher.opt("--cpus-per-task"), watcher.opt("--mem")) == ("1", "1G")
        assert watcher.args[0] == run and watcher.args[2:4] == [
            f"<{seq[0].name}>",
            seq[0].name,
        ]
        assert watcher.args[4] == seq[0].opt("--array").split("%")[0]


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


def test_the_default_is_the_multiplex_cohort_alone(dry):
    """Ruling R3.14. The default submission is the cohort and nothing dormant."""
    jobs, done, _ = dry
    datasets = {j.args[6] for j in jobs.values() if j.script == "run_rung.sbatch"}
    assert datasets == {"multiplex"}
    code = code_of(SCRIPT)
    assert 'DEFAULT_DATASETS="multiplex"' in code


@pytest.mark.parametrize("named", ["anhir", "synthetics", "multiplex x", "mcmicro@raw"])
def test_a_dataset_of_no_kind_is_refused_up_front_in_one_line(site, named):
    """Neither the cohort nor a public dataset: refused before anything is read or
    submitted, in one line naming the allowed values, with no argparse usage and no
    "cannot list" from a later step."""
    done, _, _ = launch(site, "--datasets", named, log="ds-bad")
    assert done.returncode == 2
    lines = [x for x in done.stderr.splitlines() if x.strip()]
    assert len(lines) == 1 and "the cohort is multiplex" in lines[0], lines
    assert "the public datasets are hyreco, mcmicro" in lines[0]
    assert "usage" not in done.stderr.lower() and "cannot list" not in done.stderr
    assert not _sbatch_lines(done.stdout)


def test_the_launchers_public_datasets_are_regbench_public():
    from benchmarks.regbench import PUBLIC

    (line,) = [x for x in code_of(SCRIPT).splitlines() if x.startswith("PUBLIC_DATASETS=")]
    assert line.split("=", 1)[1].strip('"').split() == list(PUBLIC)


def test_a_public_dataset_needs_its_cases_folder(site, public_launcher):
    done, _, _ = launch_public(site, public_launcher, "--datasets", "multiplex nucset", log="ds-nocases")
    assert done.returncode == 2
    assert "REGBENCH_CASES_NUCSET is not set" in done.stderr
    assert not _sbatch_lines(done.stdout)


def test_a_public_dataset_is_refused_by_prepare(site, public_launcher, tmp_path):
    done, _, _ = launch_public(
        site, public_launcher, "--prepare", "--datasets", "nucset",
        env={"REGBENCH_CASES_NUCSET": str(tmp_path)}, log="ds-prep",
    )  # fmt: skip
    assert done.returncode == 2
    assert "prepares the cohort (multiplex) and the synthetic set only" in done.stderr
    assert not _sbatch_lines(done.stdout)


def test_a_public_dataset_runs_its_own_list_from_its_own_cases(site, public_launcher, public_datasets, tmp_path):
    """``--datasets "multiplex tissueset"``: the tissue test set's runs are its own list (no ASHLAR, no
    PALOM: no input rule for a brightfield case), read from REGBENCH_CASES_TISSUESET, and
    the scorer is told both datasets (each public one is scored alone)."""
    cases = tmp_path / "tissueset_cases"
    _case(cases, "tissueset", "A1", "A1")
    _case(cases, "tissueset", "A2", "A2")
    done, _, _ = launch_public(
        site, public_launcher, "--datasets", "multiplex tissueset",
        env={"REGBENCH_CASES_TISSUESET": str(cases)}, log="ds-tissueset",
    )  # fmt: skip
    jobs = jobs_of(done)
    on = {}
    for j in [*jobs.values(), *jobs.later.values()]:
        if j.script == "run_rung.sbatch":
            on.setdefault(j.args[6], set()).add(j.args[0])
    want = {variants.run_name(r) for r in variants.runs_for("tissueset")}
    assert set(on) == {"multiplex", "tissueset"}
    assert on["tissueset"] <= want and not any(
        n.startswith(("ashlar", "affine_ashlar", "palom")) for n in on["tissueset"]
    )
    # a run of the tissue test set's regime alone, never the cohort's: its own list, not the cohort's
    assert "stare_grey_clahe" in on["tissueset"] and "stare_grey_clahe" not in on["multiplex"]
    assert "stare_clahe" in on["multiplex"] and "stare_clahe" not in on["tissueset"]
    assert jobs["rb-score"].args == ["score", "multiplex", "tissueset"]
    assert jobs["rb-tissue"].args == ["tissue", "multiplex", "tissueset"]
    assert "rb-score-case-tissueset" in jobs
    assert "# tissueset: 2 case(s)" in done.stdout


def test_a_public_dataset_alone_selects_no_cohort_only_run(site, public_launcher, public_datasets, tmp_path):
    """``--datasets tissueset`` alone: the selected runs, their selfchecks and the stage
    sizing come from the tissue test set's list; no cohort-only run (ASHLAR, PALOM, ``stare_clahe``)
    is selected or checked, and the tissue test set's own runs are."""
    cases = tmp_path / "tissueset_alone"
    _case(cases, "tissueset", "A1", "A1")
    done, _, _ = launch_public(
        site, public_launcher, "--datasets", "tissueset",
        env={"REGBENCH_CASES_TISSUESET": str(cases)}, log="ds-tissueset-alone",
    )  # fmt: skip
    jobs = jobs_of(done)
    names = set(jobs) | set(jobs.later)
    checked = {n.split("-", 2)[2].rsplit("-", 1)[0] for n in names if n.startswith("rb-sc-")}
    assert "stare" in checked
    assert not checked & {"ashlar", "affine_ashlar", "palom"}, checked
    ran = {j.args[0] for j in [*jobs.values(), *jobs.later.values()]
           if j.script == "run_rung.sbatch"}  # fmt: skip
    assert "stare_grey_clahe" in ran
    assert not ran & {"ashlar", "palom", "stare_clahe"}, ran


# the landmark set (hyreco) launches: test_submit_landmarks.py (ruling 2026-10-19)


def test_the_launchers_kinds_are_regbench_public_and_scored_kinds():
    from benchmarks.regbench import PUBLIC, score

    code = code_of(SCRIPT).splitlines()
    (kinds,) = [x for x in code if x.startswith("PUBLIC_KINDS=")]
    got = dict(i.split("=") for i in kinds.split("=", 1)[1].strip('"').split())
    assert got == {k: v["kind"] for k, v in PUBLIC.items()}
    (scored,) = [x for x in code if x.startswith("SCORED_KINDS=")]
    assert tuple(scored.split("=", 1)[1].strip('"').split()) == score.SCORED_KINDS


def test_a_public_cases_folder_that_is_not_there_is_refused(site, public_launcher, tmp_path):
    env = {"REGBENCH_CASES_NUCSET": str(tmp_path / "not_there")}
    done, _, _ = launch_public(site, public_launcher, "--datasets", "nucset", env=env, log="ds-gone")
    assert done.returncode == 2 and "not_there" in done.stderr
    assert "does not exist" in done.stderr and not _sbatch_lines(done.stdout)


def test_only_the_named_datasets_folders_are_bound(tmp_path):
    """env.sh binds the cases folders of the submission's public datasets
    (REGBENCH_PUBLIC_SETS) only: a stale REGBENCH_CASES_<NAME> left in a profile is never
    bound, and a named folder that is not there is said and left out."""
    (tmp_path / "nucset").mkdir()
    script = f'source "{SLURM / "env.sh"}"; _regbench_public_cases; echo'
    env = {k: v for k, v in os.environ.items() if not k.startswith("REGBENCH_")}
    env.update(
        REGBENCH_CASES=str(tmp_path / "c"), REGBENCH_OUT=str(tmp_path / "o"),
        REGBENCH_CASES_NUCSET=str(tmp_path / "nucset"),
        REGBENCH_CASES_MCMICRO=str(tmp_path / "stale"),
    )  # fmt: skip

    def bound(**more):
        done = subprocess.run(
            ["bash", "-c", script], env={**env, **more}, capture_output=True,
            text=True, check=True,
        )  # fmt: skip
        return done.stdout.strip(), done.stderr

    assert bound() == ("", "")
    assert bound(REGBENCH_PUBLIC_SETS="nucset")[0] == str(tmp_path / "nucset")
    got, said = bound(REGBENCH_PUBLIC_SETS="nucset mcmicro")
    assert got == str(tmp_path / "nucset") and "stale" in said


def test_the_jobs_resolve_a_datasets_cases_folder_before_using_it():
    """A job that names a dataset resolves its cases folder once, and stops (exit 2,
    naming the variable) when it is not set, never with an empty --cases; retry.sbatch
    says the units could not be written down and goes on."""
    for name in ("run_rung.sbatch", "stage.sbatch", "feed_tiles.sbatch",
                 "score_cohort.sbatch", "retry.sbatch"):  # fmt: skip
        code = code_of(SLURM / name)
        assert '--cases "$(regbench_cases_of' not in code, name
        assert re.search(r'cases="\$\(regbench_cases_of "\$[{a-z]', code), name
        assert "REGBENCH_CASES_<NAME> not set for" in code, name
    retry = code_of(SLURM / "retry.sbatch")
    cut = retry[retry.index("mark_cut() {") :]
    cut = cut[: cut.index("\n}\n")]
    assert "exit 2" not in cut and "could not be written down" in cut


def test_the_chain_summary_goes_to_the_cohorts_summary_only_when_it_was_scored():
    code = code_of(SLURM / "score_cohort.sbatch")
    at = code.index('--summary-to "$REGBENCH_OUT/tables/summary.md"')
    guard = code[code.rindex("if ", 0, at) : at]
    assert "cohort_scored" in guard, guard


def test_the_default_launch_tells_the_scorer_no_dataset(dry):
    jobs, _, _ = dry
    assert jobs["rb-score"].args == ["score"]
    assert jobs["rb-tissue"].args == ["tissue"]


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
        real = variants.realcheck_row(run) is not None
        first = rungs(jobs, run)[0].opt("--array")
        want = {(True, False): "0-1", (True, True): "1-1", (False, False): "0-3"}
        assert first == want[(per_patient, real)] + "%20", run


def test_arrays_are_throttled(site):
    done, _, _ = launch(site, env={"REGBENCH_ARRAY_LIMIT": "7"}, log="throttle")
    arrays = [j for j in jobs_of(done).values() if j.script == "run_rung.sbatch"]
    assert arrays and all(j.opt("--array").endswith("%7") for j in arrays)


# ── the selfcheck: reported, never a gate ───────────────────────────────────────────────
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
    runs = list(variants.runs_for(None))
    names = [variants.run_name(r) for r in runs]
    last = {n: last_job(n) for n in names}
    for k, name in enumerate(names):
        if k >= lanes:
            assert last[names[k - lanes]] in waits(head(jobs, name)), name
    # off: a run that reads no other run's results waits for no other run's arrays
    plain, _, _ = dry
    arrays = {x for n in names for x in (chain(plain, n)[0].name, last_job(n))}
    for r in runs:
        if r.after is None:
            own = {chain(plain, variants.run_name(r))[0].name}
            own.add(last_job(variants.run_name(r)))
            assert not (waits(head(plain, variants.run_name(r))) & (arrays - own)), r


def test_every_checking_run_has_a_selfcheck_that_climbs_in_memory(dry):
    jobs, _, _ = dry
    checks = [j for j in jobs.values() if j.script == "selfcheck.sbatch"]
    assert {j.args[0] for j in checks} == set(CHECKERS) and set(CHECKERS) <= set(RUNS)
    # one selfcheck per distinct checker, and a checker is checked by its own selfcheck
    assert all(variants.checked_by(g) == g for g in CHECKERS)
    for run in CHECKERS:
        got = [_any(jobs, f"rb-sc-{run}-{gb}g") for gb in ladder.SELFCHECK_RUNGS_GB]
        # the first is submitted; the others are steps of the selfcheck's watcher
        assert [j.name in jobs for j in got] == [True, False, False], run
        assert got[1].args[10] == got[2].args[10] == "@PREVIOUS@"
        watcher = links(jobs, f"rb-sc-{run}")[0]
        assert ("afterany", f"<{got[0].name}>") in watcher.deps
        # a selfcheck's chain (no run of its own), of single jobs, not arrays
        assert watcher.args[0] == "-" and watcher.args[4] == "-"
        assert [j.opt("--mem") for j in got] == ["64G", "256G", "512G"], run
        assert [j.opt("--cpus-per-task") for j in got] == ["4"] * 3, run
        assert [j.args[8:10] for j in got] == [["1", "64"], ["2", "256"], ["3", "512"]]
        # a later rung has no dependency: the watcher waits for the one before it
        assert all(j.opt("--dependency") is None for j in got[1:])
    code = code_of(SLURM / "selfcheck.sbatch")
    assert re.search(
        r'if \[ "\$attempt" -gt 1 \]; then\s+args\+=\(--retry-memory\)', code
    )


def test_each_selfcheck_is_given_its_runs_own_options_and_mask(dry):
    jobs, _, _ = dry
    for run in CHECKERS:
        spec, job = RUNS[run], jobs[f"rb-sc-{run}-64g"]
        name, method, label, opts, image, gpu, mask, anchor = job.args[:8]
        assert (name, method, label) == (run, spec.method, spec.label or "-")
        assert opts == (",".join(variants.opt_args(spec)) or "-")
        assert (image, gpu) == (spec.image, "1" if spec.gpu else "0")
        assert mask == (variants.selfcheck_mask(spec.method) or "-")
        assert anchor == (spec.after or "-")
    assert jobs["rb-sc-ashlar-64g"].args[6] == "overlap"
    assert jobs["rb-sc-palom-64g"].args[6] == "seams"


def test_an_anchored_selfcheck_waits_afterany(dry):
    """It reads the anchor's selfcheck work (`--m0-from`), so it waits for it: with
    `afterany`, whatever the anchor's selfcheck said. One selfcheck's verdict never
    stops another (it did with `afterok`: a failed STARE selfcheck left affine_ashlar's
    pending for ever)."""
    jobs, _, _ = dry
    job = jobs["rb-sc-affine_ashlar-64g"]
    assert job.args[7] == "stare"
    stare_last = f"<rb-sc-stare-{retry_link(LINKS)}>"
    assert job.deps == {("afterany", stare_last)}
    # its watcher has the same dependency, and so does nothing else of a selfcheck's
    assert ("afterany", stare_last) in jobs["rb-sc-affine_ashlar-retry"].deps
    for name, j in jobs.items():
        if name.startswith("rb-sc-"):
            assert {k for k, _ in j.deps} <= {"afterany"}, name
    code = code_of(SLURM / "selfcheck.sbatch")
    assert '[ "$anchor" != - ] && args+=(--m0-from "$root/$anchor")' in code


def test_no_run_waits_for_a_selfcheck(dry):
    """The selfcheck is REPORTED, NEVER A GATE: no gate job exists, no ladder, stage or
    tile job waits for a selfcheck, and nothing in the whole graph waits `afterok`."""
    jobs, _, _ = dry
    assert not [n for n in jobs if n.startswith(("rb-gate-", "rb-realgate-"))]
    assert not [n for n in getattr(jobs, "later", {}) if "gate" in n]
    for name, job in jobs.items():
        assert "afterok" not in {k for k, _ in job.deps}, name
        if job.script in ("run_rung.sbatch", "stage.sbatch", "feed_tiles.sbatch") or (
            job.script == "retry.sbatch" and not name.startswith("rb-sc-")
        ):
            assert not [n for _, n in job.deps if n.startswith("<rb-sc-")], name
    # a run that reads no other run's results and is on no lane waits for nothing
    for run, spec in RUNS.items():
        if spec.after is None and run not in STAGED:
            assert rungs(jobs, run)[0].deps == set(), run
    # every other ASHLAR cell is checked by the selfcheck of the run named for its
    # affine level, and every generated STARE cell by `stare`'s
    from benchmarks.regbench.methods import ashlar as ashlar_adapter

    for name, (affine, _, _) in ashlar_adapter.ASHLAR_CELLS.items():
        want = "affine_ashlar" if affine == "stare" else "ashlar"
        assert variants.checked_by(name) == want
    for name in RUNS:
        if RUNS[name].method == "stare":
            assert variants.checked_by(name) == "stare"


def test_a_failed_selfcheck_starts_the_run_and_is_in_the_verdicts(
    tmp_path, monkeypatch
):
    """The tasks of a run never read a verdict: with a selfcheck record that says
    inconsistent beside it, a ladder's watcher still submits the retries its units
    need. And `regbench verdict` reports the record."""
    from benchmarks.regbench import cli

    for name in (
        "run_rung.sbatch",
        "stage.sbatch",
        "retry.sbatch",
        "feed_tiles.sbatch",
    ):
        code = code_of(SLURM / name)
        assert "logs/gates" not in code and "blocked" not in code, name
    ends = {
        0: "REGBENCH_END kind=run status=ok cause= whose= wants=-",
        1: "REGBENCH_END kind=run status=failed cause=oom whose=method wants=climb",
    }
    out = tmp_path / "out"
    rec = out / "logs" / "selfcheck" / "stare" / "selfcheck.json"
    rec.parent.mkdir(parents=True)
    rec.write_text(
        json.dumps({"variants": {"stare_rigid": "ok", "stare": "inconsistent"}})
    )
    done, sent, _ = _watch(tmp_path, ends, STEPS, range_="0-1")
    assert done.returncode == 0, done.stdout + done.stderr
    # the unit that ran out of memory climbs, as it would without any selfcheck
    assert [line.split()[1] for line in sent] == [
        "--job-name=rb-x-64g",
        "--job-name=rb-x-64g-again",
    ]
    assert "blocked" not in done.stdout
    monkeypatch.delenv("REGBENCH_SUBMISSION", raising=False)
    assert cli.main(["verdict", "--out", str(out), "--runs", "stare"]) == 0
    lines = (out / "logs" / "verdicts.tsv").read_text().splitlines()
    assert lines[1].split("\t")[:3] == ["stare", "selfcheck", "inconsistent"]


def test_the_old_gate_switches_are_said_to_be_ignored(site, dry):
    env = {
        "REGBENCH_SELFCHECK_ADVISORY": "1",
        "REGBENCH_ALLOW_UNCHECKED": "valis120 palom",
        "REGBENCH_ALLOW_INCONSISTENT": "nope",  # once refused as no run's name
    }
    done, _, _ = launch(site, env=env, log="ignored")
    for var in env:
        assert f"# ignored: {var} (the selfcheck no longer gates)" in done.stdout
    assert sorted(jobs_of(done)) == sorted(dry[0])  # and nothing else changes


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
    for run in variants.members("affine_ashlar"):
        assert RUNS[run].after == "stare"
        for dataset in COHORT_DATASETS:
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
    first_array = min(
        i for i, n in enumerate(order) if re.search(r"-\d+g$", n) and "-sc-" not in n
    )
    assert last_selfcheck < first_array
    # settle, then the case arrays, then tissue, then the tables; then (Task 28, B5)
    # the light re-score, after the scorer and every selfcheck
    assert order[-5:] == [
        "rb-settle", "rb-score-case-multiplex", "rb-tissue", "rb-score", "rb-rescore"
    ]  # fmt: skip
    # a job never waits for one printed after it
    seen = set()
    for line in _sbatch_lines(done.stdout):
        job = Job(line)
        assert {n.strip("<>") for _, n in job.deps} <= seen, job.name
        seen.add(job.name)


def test_the_killed_results_are_settled_before_any_case_is_scored(dry):
    """`regbench settle` rewrites the run.json of a result the scheduler ended. It is a
    job of its own that waits for every registration, and every scoring job waits for
    it: no case is scored from a result that is about to change. It waits for NO
    selfcheck (Task 28, B5): `rb-rescore` waits for the scorer and every selfcheck's
    watcher, and writes the tables again with every verdict."""
    jobs, _, _ = dry
    settle = jobs["rb-settle"]
    assert settle.script == "score_cohort.sbatch" and settle.args == ["settle"]
    assert (settle.opt("--cpus-per-task"), settle.opt("--mem")) == ("1", "4G")
    assert {k for k, _ in settle.deps} == {"afterany"}  # whatever became of them
    waited = {n.strip("<>") for _, n in settle.deps}
    assert not any(n.startswith("rb-sc-") for n in waited)
    rescore = jobs["rb-rescore"]
    # it settles again first (ruling: a selfcheck chain still running at the first
    # settle is marked by the second), then writes the tables
    assert rescore.script == "score_cohort.sbatch" and rescore.args[:1] == ["rescore"]
    rewaited = {n.strip("<>") for _, n in rescore.deps}
    assert {k for k, _ in rescore.deps} == {"afterany"} and "rb-score" in rewaited
    for run in RUNS:
        # the re-score reports every selfcheck's verdict: it waits for each of them
        assert f"rb-sc-{variants.checked_by(run)}-{retry_link(LINKS)}" in rewaited
        for dataset in COHORT_DATASETS:
            assert last_job(run, dataset) in waited, (run, dataset)
    # every array of the per-case scoring waits for it, and for nothing else directly
    arrays = [j for n, j in jobs.items() if n.startswith("rb-score-case-")]
    assert len(arrays) == len(COHORT_DATASETS)
    for job in arrays:
        assert job.deps == {("afterany", "<rb-settle>")}, job.name
    # ... so the tissue job and the scorer come after it too, through the arrays
    after = {"rb-settle"} | {j.name for j in arrays}
    for name in ("rb-tissue", "rb-score"):
        assert {n.strip("<>") for _, n in jobs[name].deps} <= after | {"rb-tissue"}
        assert {n.strip("<>") for _, n in jobs[name].deps} >= {j.name for j in arrays}
    # no job that registers or checks anything waits for the settle job
    for name, job in jobs.items():
        if name not in ("rb-tissue", "rb-score", "rb-rescore") and job not in arrays:
            assert ("afterany", "<rb-settle>") not in job.deps, name


def test_the_settle_step_of_the_batch_script():
    code = code_of(SLURM / "score_cohort.sbatch")
    # ONE place settles (the settle job and the scorer call it); the scheduler is asked
    # twice there, with the steps both times (never -X: no CPU time, no MaxRSS): the run
    # jobs (settle_killed) and, since T31 R6/R7, the stage jobs (settle_stages)
    assert code.count("settle --out") == 1 and code.count("sacct -n -P -j") == 2
    assert "sacct -X" not in code and "sacct -n -X" not in code
    killed = code[code.index("settle_killed() {") : code.index("already_settled() {")]
    stages = code[code.index("settle_stages() {") : code.index("settle_killed() {")]
    assert killed.count("sacct -n -P -j") == 1 and stages.count("sacct -n -P -j") == 1
    # both through the one bounded call (T31 fix I1), never a bare sacct of their own
    assert "regbench_sacct -n -P -j" in killed and "regbench_sacct -n -P -j" in stages
    # defined, then run right AFTER each of the three settle_killed calls (the settle
    # job, the re-score, the scorer): a killed rung is always settled first
    assert code.count("settle_stages") == 4
    calls = [i for i in range(len(code)) if code.startswith("settle_killed", i)][1:]
    for i in calls:
        after = code[i:]
        assert after.index("settle_stages") < after.index("\nfi") + 1, code[i : i + 200]
    # defined, the settle job, the re-score (always), the scorer (when not settled)
    assert code.count("settle_killed") == 4
    rescore = code[code.index('if [ "$kind" = rescore ]') :]
    assert rescore.index("settle_killed") < rescore.index(
        'regbench_py "$REGBENCH_SCORER_IMAGE" "${args[@]}"'
    )
    job = code[
        code.index('if [ "$kind" = settle ]') : code.index('if [ "$kind" = case ]')
    ]
    assert "settle_killed" in job and "exit 0" in job  # never blocks the scoring
    assert job.index("settle_killed") < job.index("settle_stages") < job.index("regbench_say_end")
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


def test_the_settle_job_asks_the_accounting_of_this_submissions_run_tasks_only(
    dry, tmp_path
):
    """Fleet wave 2, F6: jobs.tsv holds every job ever submitted into the folder, the
    tile arrays included (hundreds of tasks each). Only a run task writes a run.json
    that can still say killed, so the settle job asks the accounting of THIS
    submission's run tasks alone, and it has hours, not one."""
    jobs, _, _ = dry
    assert jobs["rb-settle"].opt("--time") == "0-04:00:00"
    out, bin_ = tmp_path / "out", tmp_path / "bin"
    (out / "logs" / "submitted").mkdir(parents=True)
    bin_.mkdir()
    (out / "logs" / "jobs.tsv").write_text(
        "t\trb-palom-multiplex-16g\t11\n"  # this submission's run array
        "t\trb-stare-multiplex-tile-1-0\t12\n"  # its tile array
        "t\trb-palom-multiplex-64g-again\t13\n"  # a step its watcher submitted
        "t\trb-stare-multiplex-last-16g\t14\n"  # a staged round's last job
        "t\trb-palom-multiplex-16g\t9\n"  # an EARLIER submission's
    )
    (out / "logs" / "submitted" / "now.ids").write_text("11\n12\n13\n14\n")
    calls = tmp_path / "sacct.args"
    (bin_ / "sacct").write_text(f'#!/bin/bash\necho "$*" >> "{calls}"\n')
    (bin_ / "sacct").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        "REGBENCH_REPO": str(REPO),
        "REGBENCH_OUT": str(out),
        "REGBENCH_SUBMISSION": "now",
        "REGBENCH_SCORER_IMAGE": "STARE_REGBENCH",
    }
    done = subprocess.run(
        ["bash", str(SLURM / "score_cohort.sbatch"), "settle"],
        env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    (asked,) = calls.read_text().splitlines()
    assert asked.split()[asked.split().index("-j") + 1] == "11,13,14"



# a stand-in for coreutils' timeout (absent on some build hosts): the same contract,
# exit 124 when the command outlives its limit
TIMEOUT_SHIM = """#!{py}
import subprocess, sys
try:
    sys.exit(subprocess.run(sys.argv[2:], timeout=float(sys.argv[1])).returncode)
except subprocess.TimeoutExpired:
    sys.exit(124)
"""


def _settle_folder(tmp_path, sacct_body):
    out, bin_ = tmp_path / "out", tmp_path / "bin"
    (out / "logs" / "submitted").mkdir(parents=True)
    bin_.mkdir()
    (out / "logs" / "jobs.tsv").write_text(
        "t\trb-palom-multiplex-16g\t11\n"  # this submission's run task
        "t\trb-stare-multiplex-P1_r1-coarse-a1\t21\n"  # its stage jobs
        "t\trb-stare-multiplex-P1_r1-tile-a1-o0\t22\n"
        "t\trb-stare-multiplex-P0_r1-solve-a1\t8\n"  # an EARLIER submission's
    )
    (out / "logs" / "submitted" / "now.ids").write_text("11\n21\n22\n")
    calls = tmp_path / "sacct.args"
    (bin_ / "sacct").write_text(f'#!/bin/bash\necho "$*" >> "{calls}"\n' + sacct_body)
    (bin_ / "sacct").chmod(0o755)
    (bin_ / "timeout").write_text(TIMEOUT_SHIM.format(py=sys.executable))
    (bin_ / "timeout").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        "REGBENCH_REPO": str(REPO),
        "REGBENCH_OUT": str(out),
        "REGBENCH_SUBMISSION": "now",
        "REGBENCH_SCORER_IMAGE": "STARE_REGBENCH",
        "REGBENCH_SACCT_TIMEOUT_S": "1",
    }
    done = subprocess.run(
        ["bash", str(SLURM / "score_cohort.sbatch"), "settle"],
        env=env, capture_output=True, text=True,
    )  # fmt: skip
    return out, calls, done


def test_the_stage_accounting_is_this_submissions_after_the_run_tasks_and_bounded(tmp_path):
    """T31 fix I1: the stage jobs' accounting is asked AFTER the run tasks' (a killed
    rung is always settled first), for THIS submission's stage jobs only, with the
    same bound as the run tasks' query; it is never a reason for the settle to fail."""
    out, calls, done = _settle_folder(
        tmp_path, 'case "$*" in *21,22*) echo "21|COMPLETED|10|00:00:05|1|4G|" ;; esac\n')
    assert done.returncode == 0, done.stdout + done.stderr
    asked = calls.read_text().splitlines()
    ids = [a.split()[a.split().index("-j") + 1] for a in asked]
    assert ids == ["11", "21,22"]  # run tasks first; the earlier submission's job 8 is not asked
    assert all(" -X" not in a for a in asked)
    assert (out / "logs" / "accounting_stages.psv").read_text().startswith("21|COMPLETED")


def test_a_stage_accounting_that_outlives_its_bound_is_a_warning(tmp_path):
    out, calls, done = _settle_folder(
        tmp_path, 'case "$*" in *21,22*) sleep 5 ;; esac\n')
    assert done.returncode == 0, done.stdout + done.stderr
    assert "settle: WARNING: the stage jobs' accounting" in done.stdout
    assert "did not end within 1 s" in done.stdout
    assert not (out / "logs" / "accounting_stages.psv").exists()
    # the run tasks were still asked, first, and the settle job still says how it ended
    first = calls.read_text().splitlines()[0].split()
    assert first[first.index("-j") + 1] == "11"
    assert "REGBENCH_END kind=settle" in done.stdout + done.stderr

def test_tissue_then_score_run_last_in_the_scorers_image(dry):
    jobs, _, _ = dry
    tissue, scorer = jobs["rb-tissue"], jobs["rb-score"]
    assert tissue.args == ["tissue"] and scorer.args == ["score"]
    assert scorer.deps == {
        ("afterany", "<rb-tissue>"),
        ("afterany", "<rb-score-case-multiplex>"),
    }
    assert tissue.deps == {("afterany", "<rb-score-case-multiplex>")}
    code = code_of(SLURM / "score_cohort.sbatch")
    # settle, tissue x3 (the case's; the cohort's; each named dataset's), score-case,
    # the verdict, score x2 (the cohort's; each public dataset's alone) and the chain
    # summary (status)
    assert code.count('regbench_py "$REGBENCH_SCORER_IMAGE"') == 9
    # only what this submission ran or reused is scored, and nothing is dropped
    assert 'shared+=(--submission "$REGBENCH_SUBMISSION")' in code
    assert "--drop" not in code and "--gates" not in code
    assert "pipeline" not in code.lower()  # no cost is read from another run


# ── one scoring task per case, then the tables ────────────────────────────────
def test_the_scorer_runs_verdict_before_score():
    """The `score` kind first writes what every run's selfcheck and realcheck said
    (`regbench verdict`, always exit 0), then scores with `--verdicts` that file."""
    code = code_of(SLURM / "score_cohort.sbatch")
    scorer = code[code.index("if already_settled; then") :]
    verdict = scorer.index('regbench_py "$REGBENCH_SCORER_IMAGE" "${verdict[@]}"')
    score = scorer.index('regbench_py "$REGBENCH_SCORER_IMAGE" "${args[@]}"')
    assert verdict < score
    assert 'verdict=(verdict --out "$REGBENCH_OUT")' in scorer
    assert 'verdicts="$REGBENCH_OUT/logs/verdicts.tsv"' in code
    assert 'args+=(--verdicts "$verdicts")' in scorer[verdict:score]
    # the runs of the submission, from the file the launcher left beside its job ids
    assert '.runs"' in scorer and '--runs "$(head -n 1 "$runs_file")"' in scorer
    assert ".runs" in code_of(SCRIPT)
    # a case task gets no drop list and no verdict: its record does not rest on them
    case = code[
        code.index('if [ "$kind" = case ]') : code.index('if [ "$kind" = tissue ]')
    ]
    assert "verdict" not in case and "--drop" not in case


def test_every_case_is_scored_by_a_task_of_its_own_before_the_tables(dry):
    """One array per dataset, a task per case; it waits for the settle job (and so for
    everything before it), and the scorer waits for it whatever becomes of it."""
    jobs, _, _ = dry
    tissue, scorer = jobs["rb-tissue"], jobs["rb-score"]
    sizes = {"multiplex": 4}
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
    # a run in several jobs per round has, per dataset, the first attempt and the
    # watcher of COARSE and of SOLVE, its two tile submitters (the tile jobs are theirs
    # to submit), and the last job's first array and watcher (and 2 of its
    # preprocessing) where another run has the ladder's 2. No retry is submitted.
    assert set(STAGED) == {n for n, r in RUNS.items()
                           if r.method == "stare" and n not in variants.LOCAL_4CPU}
    assert len(sizes) == len(COHORT_DATASETS)
    assert (
        len(jobs)
        == N_JOBS
        == (
            PAIR * len(CHECKERS)
            + len(RUNS) * PAIR * len(sizes)
            + PAIR * N_REALCHECKED
            + 1
            + len(sizes)
            + 2
            + 1  # rb-rescore (Task 28, B5)
            + len(sizes) * MORE
        )
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
    """A case's record is reused only by a scorer given the same submission: both read
    it from ONE place in the batch script."""
    code = code_of(SLURM / "score_cohort.sbatch")
    assert code.count("shared_args\n") == 2  # the case task, and the scorer
    case = code[
        code.index('if [ "$kind" = case ]') : code.index('if [ "$kind" = tissue ]')
    ]
    assert "score-case" in case and '--index "$SLURM_ARRAY_TASK_ID"' in case
    assert '--dataset "$dataset"' in case and '--workers "$workers"' in case
    assert '${shared[@]+"${shared[@]}"}' in case
    scorer = code[code.index("args=(score ") :]
    assert '${shared[@]+"${shared[@]}"}' in scorer.split("\n\n")[0]
    # the task says how it ended, and a task that died says whose that is
    assert "cause=score_case_unfinished whose=benchmark" in case
    assert 'exit "$code"' in case
    # the case's tissue pieces first, and a failure there never stops its scoring
    assert case.index('cases="$(regbench_cases_of "$dataset")"') < case.index("tissue --cases")
    tissue = case.index('tissue --cases "$cases"')
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
    tissue, the scorer. No selfcheck, no ladder, and nothing waits for a job
    that is not there."""
    _records_site(site, tmp_path, "so", ids=("20261012T010203-9",))
    (site[0] / "out-so" / "logs" / "submission").write_text("20261012T010203-9\n")
    done, log, out = launch(site, "--score-only", log="so")
    jobs = jobs_of(done)
    assert list(jobs) == [
        "rb-settle",
        "rb-score-case-multiplex",
        "rb-tissue",
        "rb-score",
    ]
    assert jobs["rb-settle"].deps == set()  # nothing was submitted before it
    assert jobs["rb-score-case-multiplex"].deps == {("afterany", "<rb-settle>")}
    assert jobs["rb-tissue"].deps == {("afterany", "<rb-score-case-multiplex>")}
    assert jobs["rb-score"].deps == {
        ("afterany", "<rb-tissue>"),
        ("afterany", "<rb-score-case-multiplex>"),
    }
    assert "# 4 job(s) would be submitted; nothing was" in done.stdout
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
    assert no_gpu.returncode == 0 and len(jobs_of(no_gpu)) == 4
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
    """The id is PARSED out of each record by the reader the verdicts use
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
    plain = variants.members("ashlar")
    anchored = variants.members("affine_ashlar")
    # the plain ASHLAR cells (the primary row first), then STARE, then the anchored cells
    # that read it, directly after it
    assert order[: len(plain)] == plain and plain[0] == "ashlar"
    assert order[len(plain)] == "stare"
    assert order[len(plain) + 1 : len(plain) + 1 + len(anchored)] == anchored
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
    # the selfchecks are submitted in that order too
    checks = [j.args[0] for j in jobs.values() if j.script == "selfcheck.sbatch"]
    assert list(dict.fromkeys(checks)) == [n for n in wanted if n in CHECKERS]
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
    for run in variants.members("affine_ashlar"):
        assert wanted.index(run) > wanted.index("stare")
        for dataset in COHORT_DATASETS:
            first = rungs(jobs, run, dataset)[0]
            assert ("afterany", f"<{last_job('stare', dataset)}>") in first.deps
    anchored = jobs["rb-sc-affine_ashlar-64g"]
    assert ("afterany", f"<rb-sc-stare-{retry_link(LINKS)}>") in anchored.deps
    # on lanes, each run starts behind the run dealt onto its lane before it
    if lanes:
        for i, run in enumerate(wanted):
            waits = {n.strip("<>") for _, n in rungs(jobs, run)[0].deps}
            suffix = "-first" if variants.realcheck_row(run) else ""
            if suffix:
                waits = {
                    n.strip("<>") for _, n in rungs(jobs, run, suffix=suffix)[0].deps
                }
            before = wanted[i - lanes] if i >= lanes else None
            ends = {
                w for w in waits if before and w.startswith(f"rb-{before}-multiplex")
            }
            assert bool(ends) is (before is not None), (run, before, waits)


def test_an_anchored_run_named_first_is_moved_and_that_is_said(site):
    done, _, _ = launch(site, env={"REGBENCH_RUN_ORDER": ORDERS[1]}, log="order-moved")
    anchored = variants.members("affine_ashlar")
    for run in anchored:
        said = f"# run order: {run} reads the results of stare: it is submitted directly after stare"
        assert said in done.stdout
    order = _ladder_order(jobs_of(done))
    at = order.index("stare")
    assert order[at : at + 1 + len(anchored)] == ["stare", *anchored]
    assert order[0] == "ashlar"  # the rest of what was named keeps its place
    # the user's order moves nothing, and says nothing of the kind
    plain, _, _ = launch(site, env={"REGBENCH_RUN_ORDER": USER_ORDER}, log="order-user")
    assert "reads the results of" not in plain.stdout
    assert _ladder_order(jobs_of(plain))[: len(variants.members("ashlar")) + 2] == [
        *variants.members("ashlar"), "palom", "deeperhistreg_gpu",
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
    return [_any(jobs, f"rb-{run}-{dataset}-{stage}-a{a}") for a in (1, 2, 3, 4)]


def feeders(jobs, run, dataset="multiplex"):
    """The chain of tile submitters of a run on a dataset, in order."""
    return [jobs[f"rb-{run}-{dataset}-feed-{i}"] for i in range(1, FEEDERS + 1)]


def after(job):
    return {n.strip("<>") for kind, n in job.deps if kind == "afterany"}


def test_stare_is_the_pipelines_four_stages_as_jobs_of_their_own(dry):
    """COARSE, one REG-TILE task per tile, SOLVE, then the writer: each with the
    allocation the pipeline gives that process, each attempt after the one before."""
    jobs, _, _ = dry
    # every STARE run
    assert STAGED == {
        n: ("coarse", "tile", "solve", "stitch")
        for n, r in RUNS.items()
        if r.method == "stare" and n not in variants.LOCAL_4CPU
    }
    for run, names in STAGED.items():
        spec = RUNS[run]
        before = None  # the last job of the stage before
        for stage in names[:-1]:
            if stage == "tile":
                # no tile job is submitted by the launcher: the run's tile submitters
                # are, one after the other, between COARSE and SOLVE
                assert not [n for n in jobs if n.startswith(f"rb-{run}-multiplex-tile")]
                for link, job in enumerate(feeders(jobs, run), 1):
                    assert job.script == "feed_tiles.sbatch"
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
                    # four rounds of six tiles (high; more at a smaller tier), this
                    # link, and the pipeline's allocation of a tile job at each of its
                    # four attempts
                    n_tiles = TILES_PER_ROUND.get(run, TILES)
                    assert job.args[7:10] == ["4", str(4 * n_tiles), str(link)]
                    assert job.args[10:] == [
                        "{cpus}:{mem_gb}:{time}".format(**stages.allocation("tile", a))
                        for a in (1, 2, 3, 4)
                    ]
                    before = job.name
                continue
            chained = attempts(jobs, run, stage)
            # attempt 1 is submitted; 2 to 4 are steps of the stage's watcher
            assert [j.opt("--array") for j in chained] == ["0-3%20", None, None, None]
            assert [j.name in jobs for j in chained] == [True, False, False, False]
            for attempt, job in enumerate(chained, 1):
                want = stages.allocation(stage, attempt)
                assert job.script == "stage.sbatch"
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
            # the first attempt after the stage before, whatever became of that one;
            # the watcher after the first attempt, and the next stage after the watcher
            assert after(chained[0]) == ({before} if before else set()), stage
            chain_links = links(jobs, f"rb-{run}-multiplex-{stage}")
            watcher = chain_links[0]
            assert after(watcher) == after(chained[0]) | {chained[0].name}
            before = chain_links[-1].name
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
    # the cohort is fluorescence: no run of it has a proxy to write, so no prep job
    assert not any("prep" in st for st in STAGED.values())
    assert not [n for n in jobs if "-prep-" in n]


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """Cases of other sizes: `multiplex` three rounds of 1, 6 and 2 tiles."""
    cases = tmp_path_factory.mktemp("small")
    _case(cases, "multiplex", "P1_a", "P1", (1024, 1024))
    _case(cases, "multiplex", "P1_b", "P1")
    _case(cases, "multiplex", "P2_a", "P2", (2048, 4096))
    return cases


def test_the_dry_run_says_what_is_submitted_now_and_what_is_fed_in_later(site, small):
    env = {
        "REGBENCH_CASES": str(small),
        "REGBENCH_STAGE_ARRAY_MAX": "7",
        "REGBENCH_STAGE_QUEUE_MAX": "30",
    }
    done, _, _ = launch(site, "--runs", "stare stare_clahe", env=env, log="totals")
    jobs = jobs_of(done)
    tasks = sum(
        int(r.split("-")[1]) - int(r.split("-")[0]) + 1
        for j in jobs.values()
        for r in [(j.opt("--array") or "0-0").split("%")[0]]
    )
    said = done.stdout
    # multiplex: rounds of 1, 6 and 2 tiles; for each of two runs
    assert (
        "# stare on multiplex: 3 round(s) as separate jobs (coarse,tile,solve,stitch); "
        "9 tile job(s) fed in later by its tile submitter, in arrays of at most 7, at "
        "most 15 in the queue (its share of the total 30)" in said
    )
    assert (
        f"# {len(jobs)} job(s) would be submitted; nothing was, {tasks} task(s) in them "
        f"(an array counts once as a job and once per task): {len(jobs)} sbatch "
        "call(s) would be made now" in said
    )
    assert (
        "# later, by the tile submitters (feed_tiles.sbatch): 18 tile task(s), one per "
        "tile, in arrays of at most 7; plus, per attempt 2 to 4, one retry task per "
        "round that has a tile to retry" in said
    )
    assert (
        f"# the most in the queue at once: the {tasks} task(s) above plus at most 15 "
        f"tile task(s) for each of the 2 run(s) with a tile submitter (stare "
        f"stare_clahe; REGBENCH_STAGE_QUEUE_MAX=30 is the total they share): "
        f"{tasks + 2 * 15} task(s)" in said
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


def _dropped(out):
    import csv as _csv

    path = out / "submit_dropped.csv"
    with open(path, newline="") as fh:
        return list(_csv.DictReader(fh))


def test_a_run_whose_tile_arrays_cannot_be_sized_is_dropped_not_the_launch(
    site, tmp_path
):
    """Task 28, B4: a per-run configuration problem drops THAT run (a row of
    submit_dropped.csv, listed again at the end); the others are submitted."""
    wrap = _python_that_fails_on(tmp_path, "stages")
    env = {
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "PATH": f"{wrap}:{site[2]}:{Path(sys.executable).parent}:{os.environ['PATH']}",
    }
    done, log, out = launch(site, "--runs", "stare palom", env=env, dry=False,
                            log="early-stages")  # fmt: skip
    assert done.returncode == 0, done.stderr
    sent = log.read_text()
    assert "rb-palom-" in sent and "rb-stare-" not in sent
    rows = _dropped(out)
    assert [(r["run"], r["reason"]) for r in rows] == [("stare", "run_config")]
    assert "cannot size its tile arrays on multiplex" in rows[0]["detail"]
    tail = done.stdout[done.stdout.index("# DROPPED from this submission"):]
    assert "stare,run_config," in tail
    # every run dropped: then, and only then, the launch stops
    alone, log, _ = launch(site, "--runs", "stare", env=env, dry=False,
                           log="early-stages-alone")  # fmt: skip
    assert alone.returncode == 2 and not log.exists()
    assert "every selected run was dropped" in alone.stderr


def test_a_run_whose_image_has_no_path_is_dropped_not_the_launch(site, tmp_path):
    """Task 28, B4: the runs of an image with no path (no SIF, conda or venv) are
    dropped; the scorer's image with none still stops the launch."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    (venv / "bin" / "python3").symlink_to(sys.executable)
    env = {"REGBENCH_VENV_STARE_REGBENCH": str(venv)}
    done, log, out = launch(site, "--runs", "stare palom", env=env, dry=False,
                            log="no-path")  # fmt: skip
    assert done.returncode == 0, done.stderr
    rows = _dropped(out)
    assert [(r["run"], r["reason"]) for r in rows] == [("palom", "run_config")]
    assert "REGBENCH_SIF_PALOM" in rows[0]["detail"]
    sent = log.read_text()
    assert "rb-sc-stare-64g" in sent and "palom" not in sent
    scorer, log, _ = launch(site, "--runs", "stare palom", env={}, dry=False,
                            log="no-path-scorer")  # fmt: skip
    assert scorer.returncode == 2 and not log.exists()
    assert "The scorer's image needs its path" in scorer.stderr
    # M8: a hard stop, and it names itself as one (the README's integrity refusals)
    assert "integrity refusal (the scorer's image)" in scorer.stderr


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


@pytest.mark.parametrize("total", [None, 100, 7, 3, 1])
def test_the_tile_queue_ceiling_is_a_total_split_over_the_runs_with_a_submitter(
    site, total
):
    """REGBENCH_STAGE_QUEUE_MAX is for the whole launch: each tile submitter gets
    total // (runs with a tile submitter), at least 1. The count is read off the run
    list, not written down."""
    from benchmarks.regbench import stages

    fed = [
        variants.run_name(r)
        for r in variants.runs_for(None)
        if "tile" in variants.stages_of(r)
    ]
    assert len(fed) > 2  # more than the two the per-run ceiling was sized for
    env = {} if total is None else {"REGBENCH_STAGE_QUEUE_MAX": str(total)}
    done, _, _ = launch(site, env=env, log=f"share-{total}")
    assert done.returncode == 0, done.stderr
    whole = stages.QUEUE_MAX if total is None else total
    share = max(1, whole // len(fed))
    for name in fed:
        assert (
            f"in arrays of at most {stages.ARRAY_MAX}, at most {share} in the queue "
            f"(its share of the total {whole})"
        ) in next(
            ln for ln in done.stdout.splitlines() if ln.startswith(f"# {name} on ")
        )
    said = next(
        ln
        for ln in done.stdout.splitlines()
        if ln.startswith("# the most in the queue at once")
    )
    assert f"at most {share} tile task(s) for each of the {len(fed)} run(s)" in said
    assert f"REGBENCH_STAGE_QUEUE_MAX={whole} is the total they share" in said
    # what the submitters are told is the share (their variable), set before any sbatch
    code = code_of(SCRIPT)
    assert 'export REGBENCH_STAGE_QUEUE_RUN="$STAGE_QUEUE_RUN"' in code
    assert code.index("export REGBENCH_STAGE_QUEUE_RUN") < code.index("sc_time=")


@pytest.mark.parametrize("lanes", [4, 2, 20])
def test_the_tile_queue_is_split_over_the_runs_that_feed_at_once(site, lanes):
    """With REGBENCH_RUN_LANES=L at most L runs are in flight, so at most L tile
    submitters feed at once: the total is divided by min(runs with a submitter, L), and
    the dry run prints that share. More lanes than such runs: divided by the runs."""
    fed = [
        variants.run_name(r)
        for r in variants.runs_for(None)
        if "tile" in variants.stages_of(r)
    ]
    assert len(fed) == 11  # the cohort's STARE runs: 9 of the cross, 2 tiers
    total = 100
    env = {"REGBENCH_STAGE_QUEUE_MAX": str(total), "REGBENCH_RUN_LANES": str(lanes)}
    done, _, _ = launch(site, env=env, log=f"lanes-share-{lanes}")
    assert done.returncode == 0, done.stderr
    feeding = min(len(fed), lanes)
    share = total // feeding
    lines = done.stdout.splitlines()
    assert (
        f"# tile queue: REGBENCH_STAGE_QUEUE_MAX={total} split over {feeding} tile "
        f"submitter(s) feeding at once ({len(fed)} run(s) with one, "
        f"REGBENCH_RUN_LANES={lanes}): {share} each"
    ) in lines
    for name in fed:
        line = next(ln for ln in lines if ln.startswith(f"# {name} on "))
        assert f"at most {share} in the queue (its share of the total {total})" in line
    said = next(ln for ln in lines if ln.startswith("# the most in the queue at once"))
    if feeding < len(fed):
        assert (
            f"at most {share} tile task(s) for each of the {feeding} of the {len(fed)} "
            "run(s) with a tile submitter that feed at once" in said
        )
    else:
        assert f"for each of the {len(fed)} run(s) with a tile submitter" in said
    assert feeding * share <= total  # the total stays a ceiling


def test_a_runs_subset_divides_the_tile_queue_by_the_subset(site):
    """`--runs` with two STARE runs and four lanes: two submitters feed at once, so each
    gets half the total, not a ninth of it."""
    env = {"REGBENCH_STAGE_QUEUE_MAX": "100", "REGBENCH_RUN_LANES": "4"}
    done, _, _ = launch(
        site, "--runs", "stare stare_clahe palom", env=env, log="lanes-subset"
    )
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert (
        "# tile queue: REGBENCH_STAGE_QUEUE_MAX=100 split over 2 tile submitter(s) "
        "feeding at once (2 run(s) with one, REGBENCH_RUN_LANES=4): 50 each"
    ) in lines
    for name in ("stare", "stare_clahe"):
        line = next(ln for ln in lines if ln.startswith(f"# {name} on "))
        assert "at most 50 in the queue (its share of the total 100)" in line


def test_a_run_dropped_at_tile_sizing_on_a_later_dataset_leaves_the_tile_count(
    site, public_launcher, public_datasets, tmp_path
):
    """Fix round 1 (review M1): a run sized on the cohort and then dropped at tile
    sizing on a later dataset is not counted among the runs that feed the tile queue:
    TILE_RUNS is rebuilt from the runs left, so the share is the total over one."""
    cases = tmp_path / "tissueset_tiles"
    _case(cases, "tissueset", "A1", "A1")
    wrap = tmp_path / "wrap"
    wrap.mkdir()
    (wrap / "python").write_text(
        "#!/bin/bash\n"
        'case " $* " in *" stages "*"tissueset"*) echo boom >&2; exit 1 ;; esac\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    (wrap / "python").chmod(0o755)
    env = {
        "REGBENCH_CASES_TISSUESET": str(cases),
        "REGBENCH_STAGE_QUEUE_MAX": "100",
        "REGBENCH_RUN_LANES": "4",
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "PATH": f"{wrap}:{site[2]}:{Path(sys.executable).parent}:{os.environ['PATH']}",
    }
    done, _, _ = launch_public(
        site, public_launcher, "--runs", "stare stare_clahe", "--datasets", "multiplex tissueset",
        env=env, log="tiles-later",
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert "# DROPPED stare (run_config): cannot size its tile arrays on tissueset" in (
        done.stdout
    )
    assert (
        "# tile queue: REGBENCH_STAGE_QUEUE_MAX=100 split over 1 tile submitter(s) "
        "feeding at once (1 run(s) with one, REGBENCH_RUN_LANES=4): 100 each"
    ) in done.stdout.splitlines()


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
    REGBENCH_STAGES, and the one script that states it starts from nothing. A `local`
    task (the CPU cross) states it on its own command line, never in the variable."""
    code = code_of(SCRIPT)
    assert "unset REGBENCH_STAGES\n" in code[: code.index("RUNGS_GB=")]
    rung = code_of(SLURM / "run_rung.sbatch")
    stated = rung.index('*) export REGBENCH_STAGES="$stages" ;;')
    assert "unset REGBENCH_STAGES\n" in rung[:stated]
    head = rung.split("retry=0")[0]
    assert stated < len(head)
    for column, said in (("-", "[unset] []"), ("local", "[unset] [--stages local]"),
                         ("jobs", "[jobs] []")):
        done = subprocess.run(
            ["bash", "-c", f"set -- n m - - I 0 d 1 16 0 - first t {column}; {head}"
             'echo "[${REGBENCH_STAGES-unset}] [${stated[*]+${stated[*]}}]"'],
            env={**os.environ, "REGBENCH_STAGES": "local", "REGBENCH_REPO": str(REPO)},
            capture_output=True, text=True, cwd=REPO,
        )  # fmt: skip
        assert done.stdout.strip().endswith(said), column + done.stdout + done.stderr


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
    # the runs the first submission selected, for its scorer's verdicts
    (record,) = (out / "logs" / "submitted").glob("*.runs")
    assert record.read_text() == "palom\n"
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
    # the first submission's record is as it left it, and the refused call left none
    assert record.read_text() == "palom\n"
    assert list((out / "logs" / "submitted").glob("*.runs")) == [record]
    # a queue that cannot be read is not an empty queue
    blind, _, _ = launch(
        site,
        "--runs",
        "palom",
        env={**env, "SQUEUE_FAILS": "1", "REGBENCH_SQUEUE_WAIT_S": "0"},
        dry=False,
        log="twice",
    )
    assert blind.returncode == 2 and "the queue cannot be read (squeue)" in blind.stderr
    # B13: it was asked again before the launch stopped
    assert "asked 5 times" in blind.stderr
    assert blind.stdout.count("the queue could not be read (squeue), try") == 4
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
    assert "# 4 job(s) submitted" in forced.stdout
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
    assert "blocked" not in code and "logs/gates" not in code  # no gate, no lock
    assert code.rstrip().endswith("exit 0")
    assert "sbatch " not in code and "--wait" not in code  # a tile job submits none
    rung = code_of(SLURM / "run_rung.sbatch")
    assert 'stages="${14:--}"' in rung
    assert '*) export REGBENCH_STAGES="$stages" ;;' in rung
    assert "local) stated=(--stages local) ;;" in rung
    assert 'args+=(${stated[@]+"${stated[@]}"})' in rung
    # ... and it reaches the process inside the image
    assert re.search(r"for v in [^;]*\bREGBENCH_STAGES\b", code_of(SLURM / "env.sh"))
    launcher = code_of(SCRIPT)
    assert "sbatch --wait" not in launcher
    assert ': "${REGBENCH_STAGE_ARRAY_MAX:=100}"' in launcher
    assert ': "${REGBENCH_STAGE_QUEUE_MAX:=200}"' in launcher


# ── the realcheck: reported, never a gate ─────────────────────────────────────
def test_a_realcheck_runs_on_the_first_unit_without_holding_the_rest(dry):
    jobs, _, _ = dry
    real = {n: variants.realcheck_row(n) for n in RUNS if variants.realcheck_row(n)}
    assert real == {"valis_mirage": "valis_mirage", "valis120_hires": "valis120_hires"}
    settle = {n.strip("<>") for _, n in jobs["rb-settle"].deps}
    for run in real:
        first = rungs(jobs, run, suffix="-first")
        assert [j.opt("--array") for j in first] == ["0-0%20"] + [None] * 4
        assert all(j.args[9] == "1" for j in first)  # --realcheck
        for dataset in COHORT_DATASETS:
            rest = rungs(jobs, run, dataset)
            assert rest[0].opt("--array") == "1-1%20"
            assert all(job.args[9] == "0" for job in rest)
            # side by side: the rest waits for nothing of the first unit
            for job in (rest[0], jobs[last_job(run, dataset)]):
                assert not [n for _, n in job.deps if "-first-" in n], job.name
            assert rest[0].deps == first[0].deps
            # and what waits for the run waits for both
            assert {last_job(run, dataset), f"rb-{run}-{dataset}-first-{retry_link(LINKS)}"} <= settle
    others = [
        j
        for j in jobs.values()
        if j.script == "run_rung.sbatch" and j.args[0] not in real
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


def test_gpu_runs_are_dropped_without_a_request_string(site):
    """Task 28, B4: a run that needs a GPU with no request string is dropped (a row of
    submit_dropped.csv), and the runs that need none are submitted."""
    done, log, out = launch(site, env={"REGBENCH_SBATCH_GPU": ""}, log="nogpu")
    assert done.returncode == 0, done.stderr
    names = {Job(ln).name for ln in _sbatch_lines(done.stdout)}
    for run, spec in RUNS.items():
        dropped = f"# DROPPED {run} (run_config): it needs a GPU" in done.stdout
        assert dropped is spec.gpu, run
        assert any(n.startswith(f"rb-{run}-") for n in names) is (not spec.gpu), run
    assert not any(GPU in ln for ln in _sbatch_lines(done.stdout))
    # the real path writes the table
    real, log, out = launch(
        site, env={"REGBENCH_SBATCH_GPU": "", "REGBENCH_ALLOW_PATH_PYTHON": "1"},
        dry=False, log="nogpu2",
    )  # fmt: skip
    assert real.returncode == 0 and log.exists(), real.stderr
    rows = _dropped(out)
    assert {r["run"] for r in rows} == {n for n, r in RUNS.items() if r.gpu}
    assert {r["reason"] for r in rows} == {"run_config"}
    # only GPU runs selected: nothing is left, the launch stops
    gpu = " ".join(n for n, r in RUNS.items() if r.gpu)
    none, log, _ = launch(site, "--runs", gpu, env={"REGBENCH_SBATCH_GPU": ""},
                          log="gpu-only")  # fmt: skip
    assert none.returncode == 2 and "every selected run was dropped" in none.stderr


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
def test_the_preflight_drops_the_runs_it_fails_with_one_line_per_failure(
    site, tmp_path
):
    """The stand-in "image" is this machine's python: it has STARE and not PALOM.
    Task 28, B3: the runs of the (image, method) that failed are dropped (rows of
    submit_dropped.csv); the others are submitted. Only with no run left does the
    launch stop."""
    env = {
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "REGBENCH_PREFLIGHT": "1",
        "REGBENCH_WORK": str(tmp_path / "work"),
    }
    bad, log, out = launch(
        site, "--runs", "stare palom", env=env, dry=False, log="preflight-bad"
    )
    assert bad.returncode == 0 and log.exists(), bad.stderr  # stare was submitted
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
    rows = _dropped(out)
    assert [(r["run"], r["reason"]) for r in rows] == [("palom", "preflight_failed")]
    assert rows[0]["detail"].startswith("[PALOM palom] preflight FAIL [PALOM palom]")
    sent = log.read_text()
    assert "rb-sc-stare-64g" in sent and "palom" not in sent
    # palom alone: no run left, the launch stops with nothing queued
    stop, log, out = launch(
        site, "--runs", "palom", env=env, dry=False, log="preflight-stop"
    )
    assert stop.returncode == 2 and not log.exists()
    assert "every selected run was dropped (the preflight" in stop.stderr
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
    assert checks == {"affine_ashlar", "stare"}  # its checker, and that one's anchor
    assert "reads the results of stare, which is not submitted here" in done.stdout
    bad, _, _ = launch(site, "--runs", "nope", log="subset-bad")
    assert bad.returncode == 2 and "no run named 'nope'" in bad.stderr


def test_unprepared_cases_are_refused(site, tmp_path):
    done, log, _ = launch(site, env={"REGBENCH_CASES": str(tmp_path)}, log="nocases")
    assert done.returncode == 2 and "Run with --prepare first" in done.stderr
    assert not _sbatch_lines(done.stdout) and not log.exists()


def test_prepare_is_a_step_of_its_own(site, tmp_path):
    env = {
        "REGBENCH_MANIFEST": str(_manifest(tmp_path)),
        "REGBENCH_STARDIST_MODEL": "site_model",
    }
    done, _, _ = launch(site, "--prepare", env=env, log="prepare")
    jobs = jobs_of(done)
    # the nuclei step is never left out: it traces the outlines every case needs
    assert set(jobs) == {
        "rb-nuclei", "rb-nuclei-retry", "rb-nuclei-retry-2", "rb-prepare-multiplex",
    }  # fmt: skip
    assert jobs["rb-prepare-multiplex"].args[:3] == [
        "multiplex",
        "--manifest",
        env["REGBENCH_MANIFEST"],
    ]
    assert "run this script again without --prepare" in done.stdout
    assert "taken as it is" in done.stdout  # a REGBENCH_MANIFEST named by hand
    gone, _, _ = launch(
        site, "--prepare", "--datasets", "nosuchset", env=env, log="gone"
    )
    assert gone.returncode == 2 and "the cohort is multiplex" in gone.stderr
    none, _, _ = launch(site, "--prepare", log="prepare-bad")
    assert none.returncode == 2 and "REGBENCH_INPUTS" in none.stderr
    assert "REGBENCH_MANIFEST" in none.stderr
    assert not _sbatch_lines(none.stdout)
    skip, _, _ = launch(
        site, "--prepare", env={**env, "REGBENCH_NUCLEI_SKIP": "1"}, log="skip"
    )
    assert "rb-nuclei" in jobs_of(skip) and "no longer read" in skip.stdout


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
    # in this order: the array, its watcher's links, then prepare
    assert list(jobs) == [
        "rb-nuclei", "rb-nuclei-retry", "rb-nuclei-retry-2", "rb-prepare-multiplex",
    ]  # fmt: skip
    nuc, prep = jobs["rb-nuclei"], jobs["rb-prepare-multiplex"]
    assert nuc.script == "nuclei.sbatch" and nuc.args == []
    assert nuc.opt("--array") == "0-2%20"
    assert (nuc.opt("--cpus-per-task"), nuc.opt("--mem")) == ("8", "128G")
    assert GPU in nuc.line  # StarDist asks for the GPU, as the pipeline does
    # the slides StarDist ran out of memory on climb (F7): 256 GB, then 512 GB, each
    # a step of the watcher's plan, for those slides only
    watch = links(jobs, "rb-nuclei")
    assert watch[0].args[0] == "-" and watch[0].args[4] == "0-2"
    climbs = [jobs.later[f"rb-nuclei-{gb}g"] for gb in (256, 512)]
    assert [j.opt("--mem") for j in climbs] == ["256G", "512G"]
    assert all(j.script == "nuclei.sbatch" and GPU in j.line for j in climbs)
    # prepare waits for the chain's last link: every rung has run
    assert prep.deps == {("afterany", f"<rb-nuclei-{retry_link(LINKS)}>")}
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


def test_a_nuclei_request_no_node_can_grant_stops_the_launch(site, tmp_path):
    """T5: the nuclei array's FIRST request (REGBENCH_NUCLEI_MEM, 128G) is checked
    against the nodes it may have, the GPU ones here, like every rung above it: one no
    node can grant would pend for ever. The launch stops, and nothing is submitted."""
    env = {
        **_with_sinfo(site, tmp_path, [
            "cpu*|1000000|icelake|(null)", "gpu|100000|h200|gpu:h200:4",
        ]),
        "REGBENCH_MANIFEST": str(_manifest(tmp_path)),
        "REGBENCH_STARDIST_MODEL": "site_model",
        "REGBENCH_STARDIST_MODEL_DIR": "/models",
        "REGBENCH_SBATCH_GPU": "--partition=gpu --gres=gpu:h200:1",
    }  # fmt: skip
    done, log, _ = launch(site, "--prepare", "--datasets", "multiplex", env=env,
                          log="nuc-big")  # fmt: skip
    assert done.returncode != 0, done.stdout
    assert "REGBENCH_NUCLEI_MEM=128G" in done.stderr
    assert "partition gpu with gres gpu:h200 has 100000 MB" in done.stderr
    assert _sbatch_lines(done.stdout) == [] and not log.exists()  # nothing submitted


def test_prepare_leaves_the_manifest_where_the_score_job_finds_it(site, tmp_path):
    """`attrition.csv: manifest_rounds` needs the manifest in the score job, which is
    submitted by a later call that has no REGBENCH_MANIFEST."""
    cases = tmp_path / "cases"
    manifest = _manifest(tmp_path)
    env = {
        "REGBENCH_MANIFEST": str(manifest),
        "REGBENCH_CASES": str(cases),
        "REGBENCH_STARDIST_MODEL": "site_model",
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
    assert "the site's StarDist model" in none.stderr and not _sbatch_lines(none.stdout)
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


# ── the watcher: a retry is submitted after a failure, for the units that failed ──
def _watch(
    tmp_path, ends, steps, range_="0-4", run="stare", single=None, link=None,
    squeue="exit 0", python=None, sbatch_fails=None, plan_files=None, extra_env=None,
):  # fmt: skip
    """Run retry.sbatch against stand-in sbatch and squeue. ``ends``: {index: end line
    of the unit's task in the first array, or None for no end line}. ``link``: the
    watcher link's number (the 6th argument); ``squeue``: the stand-in's body;
    ``python``: a stand-in `python` that records its arguments in ``python.log``;
    ``sbatch_fails``: a job name sbatch refuses; ``plan_files``: {name: text} an
    earlier link left in the plan folder."""
    out, bin_, plan = tmp_path / "out", tmp_path / "bin", tmp_path / "plan"
    for d in (out / "logs" / "end", bin_, plan):
        d.mkdir(parents=True, exist_ok=True)
    for idx, line in ends.items():
        if line is not None:
            (out / "logs" / "end" / f"rb-x-16g-50_{idx}.end").write_text(line + "\n")
    if single is not None:  # a chain of single jobs: the first job's end line
        (out / "logs" / "end" / "rb-x-16g-50.end").write_text(single + "\n")
    for k, items in enumerate(steps, 1):
        (plan / f"step-{k}").write_text("".join(f"{i}\n" for i in items))
    for fname, text in (plan_files or {}).items():
        (plan / fname).write_text(text)
    calls = tmp_path / "sbatch.log"
    refuse = (
        f'case "$*" in *--job-name={sbatch_fails}\\ *) echo "sbatch: error: '
        'Memory specification can not be satisfied" >&2; exit 1 ;; esac\n'
        if sbatch_fails
        else ""
    )
    (bin_ / "sbatch").write_text(
        "#!/bin/bash\n"
        + refuse
        + f'n=$(( $(cat "{tmp_path}/n" 2>/dev/null || echo 100) + 1 ))\n'
        f'echo $n > "{tmp_path}/n"; echo "$*" >> "{calls}"\n'
        'echo "sbatch: slurm_job_submit: a notice"; echo $n\n'
    )
    (bin_ / "squeue").write_text(f"#!/bin/bash\n{squeue}\n")
    if python is not None:
        (bin_ / "python").write_text(
            f'#!/bin/bash\necho "$*" >> "{tmp_path}/python.log"\n{python}\n'
        )
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_}:{os.environ['PATH']}",
        "REGBENCH_REPO": str(REPO),
        "REGBENCH_OUT": str(out),
        "REGBENCH_RETRY_POLL_S": "0",
        "REGBENCH_SUBMISSION": "t",
        "REGBENCH_CASES": str(tmp_path / "cases"),
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        **(extra_env or {}),
    }
    argv = ["bash", str(SLURM / "retry.sbatch"), run, str(plan), "50", "rb-x-16g"]
    argv += [range_] + ([str(link)] if link is not None else [])
    done = subprocess.run(argv, env=env, capture_output=True, text=True)
    sent = calls.read_text().splitlines() if calls.exists() else []
    return done, sent, out


STEPS = [
    ["again", "rb-x-16g-again", "--mem=16G", "run_rung.sbatch", "@EARLIER@", "again"],
    ["climb", "rb-x-64g", "--mem=64G", "run_rung.sbatch", "@EARLIER@", "climb"],
    ["again", "rb-x-64g-again", "--mem=64G", "run_rung.sbatch", "@EARLIER@", "again"],
]


def test_the_watcher_submits_nothing_when_every_unit_is_ok(tmp_path):
    ends = {i: "REGBENCH_END kind=run status=ok cause= whose=" for i in range(5)}
    done, sent, _ = _watch(tmp_path, ends, STEPS)
    assert done.returncode == 0 and sent == [], done.stdout + done.stderr
    assert "0 of 3 retry step(s) were needed" in done.stdout
    assert done.stdout.splitlines()[-1].startswith("REGBENCH_END kind=retry")


def test_the_watcher_submits_a_step_only_for_the_units_that_failed_that_way(tmp_path):
    ends = {
        0: "REGBENCH_END kind=run status=ok cause= whose=",
        1: "REGBENCH_END kind=run status=failed cause=oom whose=method wants=climb",
        2: "REGBENCH_END kind=run status=failed cause=exception whose=method wants=-",
        3: None,  # the task died or never started
        4: "REGBENCH_END kind=run status=ok cause= whose=",
    }
    done, sent, out = _watch(tmp_path, ends, STEPS)
    assert done.returncode == 0, done.stdout + done.stderr
    # the same rung again for the lost unit alone; the next rung for the unit that ran
    # out of memory and the one still without an end line; nothing for the unit that
    # failed for another reason, and nothing for the ones that are ok
    assert sent == [
        "--parsable --job-name=rb-x-16g-again --array=3%20 --mem=16G run_rung.sbatch "
        "50 again",
        "--parsable --job-name=rb-x-64g --array=1,3%20 --mem=64G run_rung.sbatch "
        "50,101 climb",
        "--parsable --job-name=rb-x-64g-again --array=1,3%20 --mem=64G "
        "run_rung.sbatch 50,101,102 again",
    ]
    # its jobs are in the list `scancel` is given
    assert (out / "logs" / "submitted" / "t.ids").read_text().split() == [
        "101",
        "102",
        "103",
    ]


def test_the_watcher_of_single_jobs_ends_as_the_chains_last_job_did(tmp_path):
    """A selfcheck's chain: jobs, not arrays. Nothing is submitted when the first ended
    ok; otherwise the next rung is, without --array and with the id of the job before
    it, and the watcher ends ok only when the chain's last job did (nothing waits for it
    with `afterok`: the selfcheck is reported, never a gate)."""
    steps = [
        ["notok", "rb-sc-x-256g", "--mem=256G", "selfcheck.sbatch", "2", "@PREVIOUS@"],
        ["notok", "rb-sc-x-512g", "--mem=512G", "selfcheck.sbatch", "3", "@PREVIOUS@"],
    ]
    kw = {"range_": "-", "run": "-"}
    ok = "REGBENCH_END kind=selfcheck status=ok cause= whose="
    done, sent, _ = _watch(tmp_path / "ok", {}, steps, single=ok, **kw)
    assert done.returncode == 0 and sent == [], done.stdout + done.stderr
    # a selfcheck that did not end ok: the next rungs are submitted and decide for
    # themselves
    bad = "REGBENCH_END kind=selfcheck status=not_checked cause=selfcheck_not_checked"
    done, sent, _ = _watch(tmp_path / "bad", {}, steps, single=bad, **kw)
    assert sent == [
        "--parsable --job-name=rb-sc-x-256g --mem=256G selfcheck.sbatch 2 50",
        "--parsable --job-name=rb-sc-x-512g --mem=512G selfcheck.sbatch 3 101",
    ]
    # the last job left no end line here: the chain did not end ok
    assert done.returncode == 1 and "not ok" in done.stdout


def test_the_watcher_keeps_no_cause_list_of_its_own():
    """Fleet wave 2, F3: the filter is ONE predicate (failure_log.chain_wants), written
    on the run task's end line as `wants=`; the watcher reads it and keeps no list of
    causes that could drift from the ladder's."""
    code = code_of(SLURM / "retry.sbatch")
    assert "AGAIN_CAUSES" not in code and "CLIMB_CAUSES" not in code
    assert "wants=" in code
    assert set(resources.MEMORY_CAUSES) < set(ladder.CLIMB_CAUSES)
    assert not set(resources.MEMORY_CAUSES) & set(ladder.AGAIN_CAUSES)


def test_the_watcher_reads_what_a_unit_still_wants_off_its_end_line(tmp_path):
    """F3: a patient whose first failed round is the writer's and whose second ran out
    of memory climbs (its end line says wants=climb). F4: a unit no later task can run
    (wants=-: killed with an end state, lost twice, the method's own cause) is
    submitted for no step, and a unit that ran out of memory is not submitted again at
    the same rung. A run end line written before `wants=` existed is taken by every
    step, as one with no end line is."""
    end = "REGBENCH_END kind=run status=failed whose=method rung=1"
    ends = {
        0: f"{end} cause=writer wants=climb",  # F3
        1: f"{end} cause=killed wants=-",  # F4: final
        2: f"{end} cause=oom wants=climb",  # F4: no same-rung retry
        3: f"{end} cause=infrastructure wants=again,climb",
        4: f"{end} cause=exception",  # no wants= key: every step
    }
    done, sent, _ = _watch(tmp_path, ends, STEPS)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [s.split()[1:3] for s in sent] == [
        ["--job-name=rb-x-16g-again", "--array=3-4%20"],
        ["--job-name=rb-x-64g", "--array=0,2-4%20"],
        ["--job-name=rb-x-64g-again", "--array=0,2-4%20"],
    ]


RUNG = ["/r/run_rung.sbatch", "palom", "palom", "-", "-", "PALOM", "0", "multiplex"]


def test_a_later_link_of_a_finished_chain_ends_as_the_chain_did(tmp_path):
    """Fleet wave 2, F1: the link that ends the chain writes `done`; a later link (it
    waits `afterany` on the link before it) submits nothing and ends as the chain did,
    so what waits for the last link sees the chain's outcome."""
    ends = {1: "REGBENCH_END kind=run status=failed cause=oom whose=method wants=climb"}
    done, sent, _ = _watch(
        tmp_path, ends, STEPS, link=2, plan_files={"done": "failed not_submitted 1\n"}
    )
    assert sent == [] and done.returncode == 1, done.stdout + done.stderr
    assert "finished by an earlier link (failed)" in done.stdout
    last = done.stdout.splitlines()[-1]
    assert last.startswith("REGBENCH_END kind=retry") and " link=2 " in last
    assert "status=failed cause=not_submitted whose=launcher" in last


def test_a_successor_link_takes_over_the_step_an_earlier_link_submitted(tmp_path):
    """F1: a link ended at its time limit (or lost with its node) after submitting a
    step leaves `state`; its successor waits for that job, records it once, and goes
    on with the NEXT step for the units still due, with every earlier id."""
    end = "REGBENCH_END kind=run status=failed cause=oom whose=method wants=climb"
    ok = "REGBENCH_END kind=run status=ok cause= whose="
    ends = {0: ok, 1: end, 2: ok, 3: end, 4: ok}
    plan_files = {
        "state": "k=2\njob=rb-x-64g\nid=90\nearlier=50,90\nprevious=90\n",
        "due-2": "1\n3\n",
        "last": "".join(f"{i} rb-x-16g-50\n" for i in range(5)),
    }
    out = tmp_path / "out" / "logs" / "end"
    out.mkdir(parents=True)
    # at 64 GB unit 1 ran out of memory again; unit 3 left no end line (lost)
    (out / "rb-x-64g-90_1.end").write_text(end + "\n")
    done, sent, out = _watch(tmp_path, ends, STEPS, link=2, plan_files=plan_files)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "TAKING OVER the chain rb-x-16g" in done.stdout
    # step 3 only (steps 1 and 2 were the earlier link's), read off the job the
    # taken-over step submitted: the same rung again for the lost unit alone, never
    # for the one that wants to climb
    assert sent == [
        "--parsable --job-name=rb-x-64g-again --array=3%20 --mem=64G "
        "run_rung.sbatch 50,90 again"
    ]
    assert (out / "logs" / "submitted" / "t.ids").read_text().split() == ["90", "101"]
    assert (tmp_path / "plan" / "done").read_text().split() == ["ok", "-", "0"]
    # and a link after this one ends at once
    again, sent2, _ = _watch(tmp_path, ends, STEPS, link=3)
    assert again.returncode == 0 and sent2 == sent, again.stdout


def test_a_step_an_earlier_link_was_submitting_is_looked_for_in_the_queue(tmp_path):
    """F1: a link ended between sbatch and recording the id leaves `submitting`; the
    successor finds the job by its name in the queue and does not submit it twice.
    Not in the queue, it is submitted again."""
    ends = {i: None for i in range(2)}
    found = (
        'case "$*" in *--name=rb-x-16g-again*) echo 77 ;; esac; exit 0'
    )
    done, sent, out = _watch(
        tmp_path / "found", ends, STEPS, range_="0-1", link=2, squeue=found,
        plan_files={"submitting": "1 rb-x-16g-again\n"},
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    assert "it is in the queue as 77" in done.stdout
    assert [s.split()[1] for s in sent] == [
        "--job-name=rb-x-64g",
        "--job-name=rb-x-64g-again",
    ]
    assert " 50,77,101 " in sent[1] + " "  # the found job is among the earlier ids
    done, sent, _ = _watch(
        tmp_path / "lost", ends, STEPS, range_="0-1", link=2,
        plan_files={"submitting": "1 rb-x-16g-again\n"},
    )  # fmt: skip
    assert "may not have reached the queue" in done.stdout
    assert [s.split()[1] for s in sent][0] == "--job-name=rb-x-16g-again"


def test_a_takeover_that_cannot_read_the_queue_looks_again_and_never_submits_twice(
    tmp_path,
):
    """T2: the successor's look for a half-submitted step is retried as `gone()` retries
    its own: a queue that fails once and then answers finds the job; a queue that keeps
    failing ends the link loudly, with NOTHING submitted (resubmitting on a failed look
    would put the step in the queue twice)."""
    ends = {i: None for i in range(2)}
    once = (
        f'n=$(cat "{tmp_path}/q" 2>/dev/null || echo 0); echo $((n + 1)) > "{tmp_path}/q"\n'
        'case "$*" in *--name=rb-x-16g-again*)\n'
        '  [ "$n" = 0 ] && { echo "slurm_load_jobs error: Socket timed out" >&2; exit 1; }\n'
        "  echo 77 ;; esac; exit 0"
    )
    done, sent, _ = _watch(
        tmp_path / "once", ends, STEPS, range_="0-1", link=2, squeue=once,
        plan_files={"submitting": "1 rb-x-16g-again\n"},
        extra_env={"REGBENCH_RETRY_SQUEUE_TRIES": "3"},
    )  # fmt: skip
    assert "it is in the queue as 77" in done.stdout, done.stdout + done.stderr
    assert "--job-name=rb-x-16g-again" not in " ".join(sent)
    never = (
        'case "$*" in *--name=*) echo "slurm_load_jobs error: Socket timed out" >&2;'
        " exit 1 ;; esac; exit 0"
    )
    done, sent, _ = _watch(
        tmp_path / "never", ends, STEPS, range_="0-1", link=2, squeue=never,
        plan_files={"submitting": "1 rb-x-16g-again\n"},
        extra_env={"REGBENCH_RETRY_SQUEUE_TRIES": "3"},
    )  # fmt: skip
    assert done.returncode != 0 and sent == [], done.stdout + done.stderr
    assert "could not be read 3 times in a row" in done.stdout
    assert "NOT submitted again" in done.stdout
    assert done.stdout.splitlines()[-1].startswith("REGBENCH_END kind=retry")


def test_a_takeover_never_takes_a_squeue_warning_for_a_job_id(tmp_path):
    """A squeue that answers AND warns on stderr (a slurm.conf notice, say) still gives
    the job id: the warning line is never recorded as the step's job."""
    ends = {i: None for i in range(2)}
    warns = (
        'case "$*" in *--name=rb-x-16g-again*)\n'
        '  echo "squeue: warning: slurm.conf option deprecated" >&2; echo 77 ;; esac; exit 0'
    )
    done, sent, _ = _watch(
        tmp_path, ends, STEPS, range_="0-1", link=2, squeue=warns,
        plan_files={"submitting": "1 rb-x-16g-again\n"},
    )  # fmt: skip
    assert "it is in the queue as 77" in done.stdout, done.stdout + done.stderr
    assert "--job-name=rb-x-16g-again" not in " ".join(sent)


def test_a_rung_the_partition_cannot_grant_is_written_down_as_capped(tmp_path):
    """F2, then Task 28 B6: a `cut` step is never submitted. The units that would climb
    to it are written down by `regbench chain-mark` as capped (the method's, imputed:
    the ladder reached the largest rung the partition grants), and the chain ends ok:
    the ladder stops at the rung below."""
    steps = [
        STEPS[0],
        ["cut", "rb-x-64g", "the partition's largest node has 32000 MB", "--mem=64G"]
        + RUNG + ["2", "64", "0", "@EARLIER@"],
        STEPS[2],
    ]
    ends = {
        0: "REGBENCH_END kind=run status=failed cause=oom whose=method wants=climb",
        1: "REGBENCH_END kind=run status=failed cause=writer whose=method wants=-",
        2: "REGBENCH_END kind=run status=ok cause= whose=",
    }
    done, sent, _ = _watch(tmp_path, ends, steps, range_="0-2", python="exit 0")
    assert done.returncode == 0 and sent == [], done.stdout + done.stderr
    assert "NOT SUBMITTED, the partition cannot grant this rung" in done.stdout
    (call,) = (tmp_path / "python.log").read_text().splitlines()
    words = call.split()
    assert words[words.index("-m") + 2] == "chain-mark"
    got = dict(zip(words[::1], words[1::1]))
    assert got["--method"] == "palom" and got["--dataset"] == "multiplex"
    assert got["--rung-gb"] == "64" and got["--cause"] == "capped"
    assert words.count("--index") == 1 and got["--index"] == "0"
    assert "--label" not in words
    assert (tmp_path / "plan" / "done").read_text().split() == ["ok", "-", "0"]


REFUSED_STEPS = [
    STEPS[0],
    ["climb", "rb-x-64g", "--mem=64G"] + RUNG + ["2", "64", "0", "@EARLIER@"],
]
OOM_END = "REGBENCH_END kind=run status=failed cause=oom whose=method wants=climb"
FAST_RETRY = {"REGBENCH_RETRY_SUBMIT_WAIT_S": "0", "REGBENCH_RETRY_SUBMIT_TRIES": "3"}


def test_a_step_sbatch_refuses_every_time_is_written_down_as_submit_refused(tmp_path):
    """F2, then Task 28 B7: a climb step sbatch refuses is asked again (with a wait
    that doubles), and only after its last try are its units written down, as
    submit_refused (not the method's); the chain stops there."""
    done, sent, _ = _watch(
        tmp_path, {0: OOM_END}, REFUSED_STEPS, range_="0-0", python="exit 0",
        sbatch_fails="rb-x-64g", extra_env=FAST_RETRY,
    )  # fmt: skip
    assert done.returncode == 1 and sent == [], done.stdout + done.stderr
    assert "sbatch REFUSED rb-x-64g (--array=0), try 1 of 3" in done.stdout
    assert "try 3 of 3" in done.stdout and "try 4" not in done.stdout
    call = (tmp_path / "python.log").read_text()
    assert "chain-mark" in call and "--cause submit_refused" in call
    assert "Memory specification can not be satisfied" in call
    assert (tmp_path / "plan" / "done").read_text().split() == [
        "failed", "submit_refused", "1"
    ]  # fmt: skip
    assert "cause=submit_refused whose=infrastructure" in done.stdout.splitlines()[-1]


def test_a_refused_selfcheck_step_is_written_down_too(tmp_path):
    """Minor (fix round 1): a refused `notok` / `stage` step is not a step of a run's
    ladder, so chain-mark cannot write it on a record; its units are rows of
    logs/refused_units.tsv, and the scorer makes each a loss row of its cause."""
    from benchmarks.regbench import score

    steps = [["notok", "rb-sc-x-64g", "--mem=64G", "selfcheck.sbatch", "x"]]
    fail = "REGBENCH_END kind=selfcheck status=failed cause=oom whose=method"
    done, sent, out = _watch(
        tmp_path, {0: fail, 1: None}, steps, range_="0-1", python="exit 0",
        sbatch_fails="rb-sc-x-64g", extra_env=FAST_RETRY,
    )  # fmt: skip
    assert done.returncode == 1 and sent == [], done.stdout + done.stderr
    with open(out / "logs" / "refused_units.tsv", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert [(r["plan"], r["step"], r["job"], r["index"], r["cause"]) for r in rows] == [
        ("plan", "1", "rb-sc-x-64g", "0", "submit_refused"),
        ("plan", "1", "rb-sc-x-64g", "1", "submit_refused"),
    ]
    assert "Memory specification" in rows[0]["why"]
    lost = score.loss_rows(out, tmp_path / "cases", ["multiplex"], None, [], [], [],
                           [], [])  # fmt: skip
    got = [(r["run"], r["case_id"], r["cause"], r["source"]) for r in lost
           if r["source"] == "logs/refused_units.tsv"]  # fmt: skip
    assert got == [
        ("plan", "0", "submit_refused", "logs/refused_units.tsv"),
        ("plan", "1", "submit_refused", "logs/refused_units.tsv"),
    ]


def test_a_step_sbatch_refuses_once_is_submitted_on_its_next_try(tmp_path):
    """B7: a refusal that does not last (a busy slurmctld) costs a wait, not the
    units: the next try goes through, and nothing is written down."""
    flag = tmp_path / "refused-once"
    sbatch_once = tmp_path / "bin-pre"
    sbatch_once.mkdir(parents=True)
    real = tmp_path / "bin" / "sbatch"
    (sbatch_once / "sbatch").write_text(
        "#!/bin/bash\n"
        f'if [ ! -e "{flag}" ]; then touch "{flag}"; '
        'echo "sbatch: error: Socket timed out on send/recv" >&2; exit 1; fi\n'
        f'exec "{real}" "$@"\n'
    )
    (sbatch_once / "sbatch").chmod(0o755)
    path = f"{sbatch_once}:{tmp_path / 'bin'}:{os.environ['PATH']}"
    env = {**FAST_RETRY, "PATH": path}
    done, sent, _ = _watch(
        tmp_path, {0: OOM_END}, REFUSED_STEPS, range_="0-0", python="exit 0",
        extra_env=env,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    assert "try 1 of 3" in done.stdout and "try 2" not in done.stdout
    assert [s.split()[1] for s in sent] == ["--job-name=rb-x-64g"]
    assert not (tmp_path / "python.log").exists()


def test_a_refusal_whose_job_is_in_the_queue_after_all_is_never_submitted_twice(
    tmp_path,
):
    """B7: before trying again the queue is asked for the step by its name; a refused
    call that queued the job anyway is taken, not submitted a second time."""
    found = 'case "$*" in *--name=rb-x-64g*) echo 77 ;; esac; exit 0'
    done, sent, _ = _watch(
        tmp_path, {0: OOM_END}, REFUSED_STEPS, range_="0-0", python="exit 0",
        sbatch_fails="rb-x-64g", squeue=found, extra_env=FAST_RETRY,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    assert "the queue has it as 77: taken" in done.stdout
    assert "try 2" not in done.stdout and sent == []
    assert "77" in (tmp_path / "out" / "logs" / "submitted" / "t.ids").read_text()


def test_a_watcher_that_gives_up_says_why_for_the_settle_job(tmp_path):
    """S3: a watcher that cannot read the queue writes `gave_up` (why) before it ends
    without `done`: the settle job writes what was due as watcher_gave_up. A link that
    takes the chain over removes it."""
    never = 'echo "slurm_load_jobs error: Socket timed out" >&2; exit 1'
    done, sent, _ = _watch(
        tmp_path, {0: OOM_END}, REFUSED_STEPS, range_="0-0", python="exit 0",
        squeue=never, extra_env={"REGBENCH_RETRY_SQUEUE_TRIES": "2"},
    )  # fmt: skip
    assert done.returncode == 1 and len(sent) == 1, done.stdout + done.stderr
    plan = tmp_path / "plan"
    assert "the queue could not be read while waiting for step 2" in (
        plan / "gave_up"
    ).read_text()
    assert not (plan / "done").exists()
    assert "cause=watcher_gave_up whose=infrastructure" in done.stdout.splitlines()[-1]
    # M6: the watcher itself writes the units it was waiting for down first, so a
    # settle job that fails does not leave them the method's
    call = (tmp_path / "python.log").read_text()
    assert "chain-mark" in call and "--cause watcher_gave_up" in call
    assert "--index 0" in call
    again, _, _ = _watch(tmp_path, {0: OOM_END}, REFUSED_STEPS, range_="0-0",
                         python="exit 0", link=2)  # fmt: skip
    assert again.returncode == 0, again.stdout + again.stderr
    assert not (plan / "gave_up").exists()


def _rung_task(tmp_path, python_body):
    """run_rung.sbatch as the scheduler starts it, with a stand-in `python`."""
    bin_ = tmp_path / "bin"
    bin_.mkdir(parents=True, exist_ok=True)
    (bin_ / "python").write_text("#!/bin/bash\n" + python_body + "\n")
    (bin_ / "python").chmod(0o755)
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("REGBENCH_")},
        "PATH": f"{bin_}:/usr/bin:/bin",
        "REGBENCH_REPO": str(REPO),
        "REGBENCH_OUT": str(tmp_path / "out"),
        "REGBENCH_CASES": str(tmp_path / "cases"),
        "SLURM_JOB_ID": "51",
        "SLURM_JOB_NAME": "rb-palom-multiplex-16g",
        "SLURM_ARRAY_JOB_ID": "50",
        "SLURM_ARRAY_TASK_ID": "3",
    }
    argv = ["bash", str(SLURM / "run_rung.sbatch"), "palom", "palom", "-", "-"]
    argv += ["PALOM", "0", "multiplex", "1", "16", "0", "-", "first", "1-00:00:00", "-"]
    done = subprocess.run(argv, env=env, capture_output=True, text=True)
    end = tmp_path / "out" / "logs" / "end" / "rb-palom-multiplex-16g-50_3.end"
    return done, end


def test_a_task_the_kernel_killed_for_memory_says_so_on_its_end_line(tmp_path):
    """Fleet wave 2, F4: Python killed by signal 9 while the script goes on is the
    kernel's out-of-memory killer (the note ladder.settle reads). Its end line says
    so, and that only the next rung can run the unit: the watcher submits no
    same-rung retry for it, which would hold a full allocation only to exit."""
    done, end = _rung_task(tmp_path, "kill -9 $$")
    assert done.returncode == 0, done.stdout + done.stderr
    line = end.read_text().strip()
    assert " cause=oom whose=method " in f"{line} " and " wants=climb" in line
    assert "exit=137" in line and "unit=index_3" in line and "rung=1" in line
    assert [ln for ln in done.stdout.splitlines() if "REGBENCH_END" in ln][-1] == line
    note = tmp_path / "out" / "logs" / "accounting" / "exits" / "50_3"
    assert note.read_text().strip() == "137"
    # any other death leaves no end line of the step's own: the fallback, every step
    done, end = _rung_task(tmp_path / "other", "exit 3")
    assert not end.exists()
    last = [ln for ln in done.stdout.splitlines() if "REGBENCH_END" in ln][-1]
    assert "status=no_end_line cause=killed" in last and "wants=" not in last


def _with_sinfo(site, tmp_path, nodes):
    """An env whose PATH has a stand-in `sinfo` printing ``nodes`` (`%P|%m|%f` lines)
    and recording its arguments."""
    bin_ = tmp_path / "sinfo-bin"
    bin_.mkdir()
    (bin_ / "sinfo").write_text(
        f'#!/bin/bash\necho "$*" >> "{tmp_path}/sinfo.log"\ncat <<\'EOF\'\n'
        + "".join(f"{n}\n" for n in nodes)
        + "EOF\n"
    )
    (bin_ / "sinfo").chmod(0o755)
    _, _, fake = site
    return {"PATH": f"{bin_}:{fake}:{Path(sys.executable).parent}:{os.environ['PATH']}"}


NODES = [
    "cpu*|515000|icelake",  # the default partition: 503 GB nodes
    "cpu*|257000|icelake",
    "big|720000|sapphire",  # the site's 700 GB node, in a partition of its own
]


def test_a_rung_no_node_can_grant_is_not_planned_and_said_loudly(site, tmp_path):
    """Fleet wave 2, F2: the launcher reads the partition's largest node off `sinfo`
    before anything is submitted. A rung above it is never planned as a job (it
    would pend for ever, or be refused days into the run): it is a `cut` step, and
    the launch says so on stdout AND stderr."""
    done, _, _ = launch(site, env=_with_sinfo(site, tmp_path, NODES), log="cut")
    assert done.returncode == 0, done.stderr
    jobs = jobs_of(done)
    warn = "can NEVER be granted there"
    assert warn in done.stdout and warn in done.stderr
    assert "the rung(s) 512 GB, 700 GB can NEVER" in done.stdout
    assert "# NOT PLANNED, no node of partition <the default> can grant it" in done.stdout
    # the 512 GB rung of every run is a cut step, and nothing above it is planned
    everything = {**jobs, **jobs.later}
    assert not [n for n in everything if "-700g" in n]
    assert "rb-palom-multiplex-512g" not in everything
    assert "rb-palom-multiplex-256g-again" in jobs.later
    # the selfcheck's 512 GB rung too
    assert "rb-sc-stare-512g" not in everything and "rb-sc-stare-256g" in jobs.later
    assert "ladder(s) stop below a rung no node of" in done.stdout
    assert (tmp_path / "sinfo.log").read_text().split()[:4] == [
        "-h", "-N", "-o", "%P|%m|%f|%G"
    ]  # fmt: skip
    # named on the command line, the partition with the large node plans the ladder
    done, _, _ = launch(
        site, "--", "--partition=big", env=_with_sinfo(site, tmp_path / "b", NODES),
        log="big",
    ) if (tmp_path / "b").mkdir() is None else None  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert "-p big" in (tmp_path / "b" / "sinfo.log").read_text()
    assert "every rung (16, 64, 256, 512, 700 GB" in done.stdout
    assert "rb-palom-multiplex-700g" in jobs_of(done).later


def test_a_gpu_run_is_checked_against_the_nodes_it_can_have(site, tmp_path):
    """T4: a run that needs a GPU is given REGBENCH_SBATCH_GPU (here a partition of its
    own and a typed gres): its rungs are checked against the largest node of THAT
    partition that HAS the gres, not against the CPU partition. A larger node there
    without the card does not count."""
    nodes = [
        "cpu*|1000000|icelake|(null)",  # the CPU runs' partition: every rung fits
        "gpu|300000|h200|gpu:nvidia_h200:4(S:0-1)",  # 292 GB with the card
        "gpu|900000|h200|(null)",  # more memory, no card
    ]
    env = {
        **_with_sinfo(site, tmp_path, nodes),
        "REGBENCH_SBATCH_GPU": "--partition=gpu --gres=gpu:nvidia_h200:1",
    }
    done, _, _ = launch(site, env=env, log="gpu")
    assert done.returncode == 0, done.stderr
    everything = {**jobs_of(done), **jobs_of(done).later}
    gpu_runs = sorted({n.rsplit("-", 1)[0] for n in everything
                       if n.endswith("_gpu-multiplex-16g")})  # fmt: skip
    assert gpu_runs, sorted(everything)
    for run in gpu_runs:
        assert f"{run}-256g" in everything and f"{run}-512g" not in everything, run
    assert "rb-palom-multiplex-700g" in everything  # the CPU partition grants it
    assert (
        "GPU runs: the largest node of partition gpu with gres gpu:nvidia_h200 has "
        "300000 MB (sinfo), so the rung(s) 512 GB, 700 GB can NEVER be granted there"
    ) in done.stdout


def test_a_partition_that_cannot_grant_the_first_rung_stops_the_launch(site, tmp_path):
    """F2: no run could start; nothing is submitted, in a dry run as in --preflight."""
    env = _with_sinfo(site, tmp_path, ["tiny*|8000|"])
    done, log, _ = launch(site, env=env, log="tiny")
    assert done.returncode != 0 and "first rung, 16 GB" in done.stderr
    assert not log.exists()
    env["REGBENCH_ALLOW_PATH_PYTHON"] = "1"
    done, log, _ = launch(site, "--preflight", env=env, dry=False, log="tiny-pre")
    assert done.returncode != 0 and "first rung, 16 GB" in done.stderr
    assert not log.exists()


def test_an_earlier_default_results_folder_of_the_same_cases_is_named(site, tmp_path):
    """Fleet wave 2, B3: the default REGBENCH_OUT is named after the commit, so a launch
    from a newer commit starts an empty folder. An earlier default-named folder of the
    same cases is named loudly, with the export that reuses it; the default stays."""
    _, cases, _ = site
    home = tmp_path / "home"
    old = home / "regbench_results" / "mirage-0ld0ld0"
    (old / "logs").mkdir(parents=True)
    (old / "logs" / "cases").write_text(f"{cases}\n")
    (old / "palom" / "multiplex" / "P1_a").mkdir(parents=True)
    (old / "palom" / "multiplex" / "P1_a" / "run.json").write_text("{}")
    other = home / "regbench_results" / "mirage-0th3r00"
    (other / "logs").mkdir(parents=True)
    (other / "logs" / "cases").write_text("/somewhere/else\n")
    env = {"HOME": str(home), "REGBENCH_OUT": None}
    done, _, _ = launch(site, "--runs", "palom", env=env, log="default-out")
    assert done.returncode == 0, done.stderr
    warn = f"an EARLIER default-named folder holds 1 result(s) of the same cases ({cases}): {old}"
    assert warn in done.stdout and warn in done.stderr
    assert f"export REGBENCH_OUT={old}" in done.stdout
    assert str(other) not in done.stdout
    # named by hand, nothing is said
    done, _, _ = launch(site, "--runs", "palom", env={"HOME": str(home)}, log="named")
    assert "EARLIER default-named folder" not in done.stdout
    # a real launch writes down the cases its folder is of
    done, _, out = launch(site, "--runs", "palom", dry=False, log="marker",
                          env={"REGBENCH_ALLOW_PATH_PYTHON": "1"})  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert (out / "logs" / "cases").read_text().strip() == str(cases)


def test_the_dry_run_plans_a_700g_rung(dry):
    """User ruling 2026-10-16: the ladder ends at the site's 700 GB node, for every
    method; only the selfcheck stops at 512 GB."""
    jobs, _, _ = dry
    first = _any(jobs, "rb-palom-multiplex-16g")
    assert first.opt("--mem") == "16G"
    steps = [_any(jobs, f"rb-palom-multiplex-{gb}g") for gb in (16, 64, 256, 512, 700)]
    assert [j.opt("--mem") for j in steps] == ["16G", "64G", "256G", "512G", "700G"]
    assert [j.args[7:9] for j in steps][-2:] == [["4", "512"], ["5", "700"]]
    assert _any(jobs, "rb-palom-multiplex-700g-again").opt("--mem") == "700G"
    assert _any(jobs, "rb-stare-multiplex-last-700g").opt("--mem") == "700G"
    assert _any(jobs, "rb-stare-multiplex-last-700g-again").opt("--mem") == "700G"
    everything = {**jobs, **jobs.later}
    assert not [n for n in everything if n.startswith("rb-sc-") and "700g" in n]


# ── the CPU cross (plan F, task 9): REGBENCH_CPUS and STARE's stages in one job ─────
def _ladder_jobs(jobs):
    """Every ladder task, submitted or planned for later (run_rung.sbatch)."""
    everything = {**jobs, **jobs.later}
    return {n: j for n, j in everything.items() if j.script == "run_rung.sbatch"}


def test_regbench_cpus_is_every_ladder_tasks_allocation_and_nothing_elses(site):
    done, _, _ = launch(site, env={"REGBENCH_CPUS": "8"}, log="cpus8")
    jobs = jobs_of(done)
    ladders = _ladder_jobs(jobs)
    assert ladders and {j.opt("--cpus-per-task") for j in ladders.values()} == {"8"}
    # the jobs outside the ladder keep their own allocation (selfcheck, settle, score)
    plain, _, _ = launch(site, log="cpus-default")
    base = {**jobs_of(plain), **jobs_of(plain).later}
    for name, job in {**jobs, **jobs.later}.items():
        if job.script != "run_rung.sbatch":
            assert job.opt("--cpus-per-task") == base[name].opt("--cpus-per-task"), name


@pytest.mark.parametrize("value", ["6", "2", "0", "16", "eight", "4 8"])
def test_a_cpu_count_outside_the_cross_is_refused_before_any_sbatch(site, value):
    done, log, _ = launch(
        site, env={"REGBENCH_CPUS": value}, log="cpus-bad-" + value.replace(" ", "_")
    )
    assert done.returncode == 2
    assert "REGBENCH_CPUS" in done.stderr and "4 or 8" in done.stderr
    assert not _sbatch_lines(done.stdout)
    assert not log.exists() or not log.read_text()


def test_a_stare_execution_other_than_jobs_or_local_is_refused(site):
    done, _, _ = launch(
        site, env={"REGBENCH_STARE_EXECUTION": "laptop"}, log="exec-bad"
    )
    assert done.returncode == 2 and "REGBENCH_STARE_EXECUTION" in done.stderr
    assert not _sbatch_lines(done.stdout)


def test_stare_local_is_one_job_per_case_on_the_ordinary_ladder(site):
    env = {"REGBENCH_CPUS": "8", "REGBENCH_STARE_EXECUTION": "local"}
    done, _, _ = launch(site, env=env, log="stare-local")
    jobs = jobs_of(done)
    everything = {**jobs, **jobs.later}
    assert "# STARE: stages run inside one job of 8 CPUs (local)" in done.stdout
    for run in STAGED:
        # no stage chain, no tile submitter, no `-last` job
        assert not [
            n for n in everything if n.startswith(f"rb-{run}-multiplex-") and (
                any(f"-{st}-a" in n for st in ("coarse", "tile", "solve", "stitch"))
                or "-feed-" in n or "-last-" in n
            )
        ], run  # fmt: skip
        # the ordinary ladder's names (rungs() would look for the staged `-last` job)
        got = [_any(jobs, f"rb-{run}-multiplex-{gb}g") for gb in ladder.RUNGS_GB]
        again = [
            _any(jobs, f"rb-{run}-multiplex-{gb}g-again") for gb in ladder.RUNGS_GB
        ]
        first = got[0]
        assert (
            first.name in jobs
            and [j.args[11] for j in got] == ["first"] + ["climb"] * 4
        )
        # an array over the cases (STARE runs per pair), at the plain round's time
        assert first.opt("--array").split("%")[0] == "0-3", run
        assert first.opt("--time") == jobs["rb-palom-multiplex-16g"].opt("--time")
        for j in got + again:
            assert j.args[13] == "local", (run, j.name)
            assert "workers=8" in j.args[3].split(","), (run, j.args[3])
            assert j.opt("--cpus-per-task") == "8"
    assert "rb-stare-multiplex-coarse-a1" not in everything
    assert not [n for n in everything if n.startswith("rb-stare-multiplex-coarse")]
    # the runs that are one job per unit are the same jobs as under `jobs`; one that
    # reads a STARE run's results waits for that run's (now plain) ladder instead
    staged_off, _, _ = launch(site, env={"REGBENCH_CPUS": "8"}, log="stare-jobs8")
    other = jobs_of(staged_off)

    def without_deps(job, out):
        return [
            w.replace(out, "OUT")
            for w in job.words
            if not w.startswith("--dependency=")
        ]

    for name, job in jobs.items():
        if job.script == "run_rung.sbatch" and job.args[0] not in STAGED:
            assert without_deps(job, "out-stare-local") == without_deps(
                other[name], "out-stare-jobs8"
            ), name
            plain = {(k, d.replace("-last-", "-")) for k, d in other[name].deps}
            assert job.deps == plain, name
    assert "STARE: stages run inside one job" not in staged_off.stdout


def test_stare_local_at_four_cpus_names_four_workers(site):
    done, _, _ = launch(site, "--runs", "stare", env={"REGBENCH_STARE_EXECUTION": "local"},
                        log="stare-local4")  # fmt: skip
    jobs = jobs_of(done)
    first = jobs["rb-stare-multiplex-16g"]
    assert first.opt("--cpus-per-task") == "4" and first.args[13] == "local"
    assert first.args[3].split(",")[-1] == "workers=4"
    assert "# STARE: stages run inside one job of 4 CPUs (local)" in done.stdout


def test_the_explicit_defaults_are_the_default_launch(site, dry):
    env = {"REGBENCH_CPUS": "4", "REGBENCH_STARE_EXECUTION": "jobs"}
    done, _, _ = launch(site, env=env, log="explicit-defaults")
    same = [
        ln.replace("out-explicit-defaults", "out-calls")
        for ln in _sbatch_lines(done.stdout)
    ]
    assert same == _sbatch_lines(dry[1].stdout)
    assert "STARE: stages run inside one job" not in dry[1].stdout


# the commit before the CPU cross: its launcher is what every main-run result was
# submitted with, so the default launch must submit exactly what it did
BEFORE_CROSS = "275edda0"


def test_the_default_launch_is_byte_identical_to_the_launcher_before_the_cross(
    site, dry, tmp_path
):
    have = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-e", f"{BEFORE_CROSS}^{{commit}}"],
        capture_output=True,
    )
    if have.returncode != 0:
        pytest.skip(f"{BEFORE_CROSS} is not in this clone (shallow)")
    old = tmp_path / "old"
    old.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", BEFORE_CROSS, "benchmarks/regbench/slurm"],
        capture_output=True, check=True,
    )  # fmt: skip
    subprocess.run(["tar", "-x", "-C", str(old)], input=archive.stdout, check=True)
    old_slurm = old / "benchmarks" / "regbench" / "slurm"
    # T31 R2 adds ONE run to the default launch, stare_local4 (STARE at 4 CPUs in one
    # job, a cost row of stare's; its run-list column `local` is not one the old
    # launcher knows): the old launcher submits every OTHER run, and the new default
    # launch must be exactly that plus stare_local4's own jobs, with nothing else
    # waiting for them but the settle and scoring jobs that wait for every run
    others = " ".join(n for n in RUNS if n not in variants.LOCAL_4CPU)
    argv, base, _ = _launch_command(
        site, ("--runs", others), {"REGBENCH_REPO": str(REPO)}, True, "before"
    )
    argv[1] = str(old_slurm / "submit_cohort.sh")
    done = subprocess.run(argv, env=base, capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stderr
    then = [
        ln.replace(str(old_slurm), str(SLURM)).replace("out-before", "out-calls")
        for ln in _sbatch_lines(done.stdout)
    ]
    # Task 28, B5, the one change to the default graph: the settle job no longer
    # waits for the selfchecks' watchers, and rb-rescore (after the scorer and them)
    # is added
    local = tuple(variants.LOCAL_4CPU)

    def own(ln):
        return any(f"rb-{r}-" in Job(ln).name for r in local)

    def without(ln):
        for r in local:
            ln = re.sub(rf":<rb-{r}-[^>]*>", "", ln)
        return ln

    every_line = _sbatch_lines(dry[1].stdout)
    assert any(own(ln) for ln in every_line)  # it IS in the default launch
    now_lines = [without(ln) for ln in every_line if not own(ln)]
    rescore = [ln for ln in now_lines if Job(ln).name == "rb-rescore"]
    assert len(rescore) == 1 and "<rb-score>" in rescore[0]
    then = [
        re.sub(r":<rb-sc-[^>]*>", "", ln) if Job(ln).name == "rb-settle" else ln
        for ln in then
    ]
    assert then and then == [ln for ln in now_lines if ln not in rescore]
    later = [ln for ln in done.stdout.splitlines() if ln.startswith(LATER)]
    now = [without(ln) for ln in dry[1].stdout.splitlines()
           if ln.startswith(LATER) and not own(ln[len(LATER):])]
    assert [
        ln.replace(str(old_slurm), str(SLURM)).replace("out-before", "out-calls")
        for ln in later
    ] == now


def _stare_record(out, record, variant="stare", case="P1_a"):
    d = Path(out) / variant / "multiplex" / case
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(record))
    return d


@pytest.mark.parametrize(
    "record",
    [
        {"method": "stare", "ok": True, "reuse_key": {"execution": "jobs"}},
        # a failed unit carries the key too: it is still the staged run's folder
        {"method": "stare", "ok": False, "reuse_key": {"execution": "jobs"}},
        # a record that names the execution only in what the jobs added up to
        {"method": "stare", "ok": True, "info": {"jobs": {"execution": "jobs"}}},
    ],
)
def test_a_local_stare_launch_over_staged_stare_results_is_refused(site, record):
    """Fix round 1 of task 9: a `local` launch into a folder of staged STARE results
    would run every STARE unit again (another reuse key) and replace them. Refused
    before any sbatch, naming the folder and the way out."""
    tmp, _, _ = site
    out = tmp / "out-local-over-staged"
    shutil.rmtree(out, ignore_errors=True)
    _stare_record(out, record, variant="stare_medium")
    done, log, _ = launch(
        site, env={"REGBENCH_STARE_EXECUTION": "local"}, log="local-over-staged"
    )
    assert done.returncode == 2, done.stdout + done.stderr
    assert str(out) in done.stderr and "separate REGBENCH_OUT" in done.stderr
    assert "stare_medium/multiplex/P1_a" in done.stderr
    assert not _sbatch_lines(done.stdout)
    assert not log.exists() or not log.read_text()


def test_staged_and_local_launches_over_results_that_do_not_clash_go_on(site):
    tmp, _, _ = site
    # a staged launch over local results: allowed (the README's cross folders)
    out = tmp / "out-staged-over-local"
    shutil.rmtree(out, ignore_errors=True)
    _stare_record(out, {"method": "stare", "ok": True, "reuse_key": {"execution": "local"}})
    done, _, _ = launch(site, log="staged-over-local")
    assert done.returncode == 0, done.stderr
    # a local launch over local results, and over another method's staged-looking key
    out = tmp / "out-local-over-local"
    shutil.rmtree(out, ignore_errors=True)
    _stare_record(out, {"method": "stare", "ok": True, "reuse_key": {"execution": "local"}})
    _stare_record(
        out, {"method": "palom", "ok": True, "reuse_key": {"execution": "jobs"}}, "palom"
    )
    (out / "stare" / "multiplex" / "P2_a").mkdir(parents=True)
    (out / "stare" / "multiplex" / "P2_a" / "run.json").write_text("not json")
    done, _, _ = launch(site, env={"REGBENCH_STARE_EXECUTION": "local"}, log="local-over-local")
    assert done.returncode == 0, done.stderr


def test_a_datasets_cases_folder_is_the_cohorts_or_its_own(tmp_path):
    """env.sh's regbench_cases_of: REGBENCH_CASES for the cohort, REGBENCH_CASES_<NAME>
    for a public dataset by its base name, exit 1 when that is not set."""
    script = (
        f'source "{SLURM / "env.sh"}"; '
        'regbench_cases_of multiplex; regbench_cases_of mcmicro@raw; '
        'regbench_cases_of nucset || echo "nucset: none"'
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("REGBENCH_")}
    env.update(
        REGBENCH_CASES=str(tmp_path / "c"), REGBENCH_CASES_MCMICRO=str(tmp_path / "m"),
        REGBENCH_OUT=str(tmp_path / "o"),
    )  # fmt: skip
    done = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=True
    )
    assert done.stdout.split("\n")[:3] == [
        str(tmp_path / "c"), str(tmp_path / "m"), "nucset: none"
    ]  # fmt: skip


def test_stare_local4_is_one_job_per_case_at_4_cpus_with_its_own_workers(dry):
    """T31 R2: STARE at 4 CPUs in one job, its stages inside it (``--stages local``
    from the run list's column `local`), on the ordinary ladder; its options name its
    workers (4), never appended to."""
    jobs, _, _ = dry
    for run in variants.LOCAL_4CPU:
        got = rungs(jobs, run)
        assert [j.opt("--cpus-per-task") for j in got] == ["4"] * 5
        assert {j.args[13] for j in got} == {"local"}
        assert not [n for n in jobs if n.startswith(f"rb-{run}-") and ("-coarse-" in n or "-feed-" in n)]


def test_stare_local4_is_dropped_at_another_cpu_count(site):
    done, _, _ = launch(site, env={"REGBENCH_CPUS": "8"}, log="local4-at-8")
    jobs = jobs_of(done)
    assert not [n for n in jobs if any(n.startswith(f"rb-{r}-") for r in variants.LOCAL_4CPU)]
    assert "DROPPED stare_local4 (run_config)" in done.stdout


def test_waves_submit_one_method_after_the_other(site, dry):
    """T31 R14, REGBENCH_WAVES=1: a method's first jobs wait for every job of the method
    before it in the run list (a timing replicate with no other method on the nodes);
    off by default, the plain launch has no such wait."""

    def waits(job):
        return {name.strip("<>") for _, name in job.deps}

    def head(jobs, run, dataset="multiplex"):
        if run in STAGED:
            return jobs[f"rb-{run}-{dataset}-{STAGED[run][0]}-a1"]
        return jobs.get(f"rb-{run}-{dataset}-first-16g") or rungs(jobs, run, dataset)[0]

    done, _, _ = launch(site, env={"REGBENCH_WAVES": "1"}, log="waves")
    jobs = jobs_of(done)
    runs = list(variants.runs_for(None))
    for prev, run in zip(runs, runs[1:]):
        if prev.method == run.method:
            continue
        before = [variants.run_name(r) for r in runs[: runs.index(run)] if r.method == prev.method]
        got = waits(head(jobs, variants.run_name(run)))
        # the wave waits for the last job of every run of the method before it
        assert {last_job(n) for n in before} <= got, variants.run_name(run)
        plain = waits(head(dry[0], variants.run_name(run)))
        assert not ({last_job(n) for n in before} <= plain) or run.after, variants.run_name(run)
    bad, _, _ = launch(site, env={"REGBENCH_WAVES": "2"}, log="waves-bad")
    assert bad.returncode != 0 and "REGBENCH_WAVES='2'" in bad.stderr + bad.stdout


def test_a_submitted_repeat_waits_for_the_ids_of_the_repeat_before(site):
    """T31 fix M3, submitted (not dry): rep2's first job names rep1's scorers' ids."""
    env = {"REGBENCH_REPEAT": "2", "REGBENCH_ALLOW_PATH_PYTHON": "1"}
    done, calls, out = launch(site, env=env, dry=False, log="repeat-real")
    assert done.returncode == 0, done.stderr
    rows = [ln.split("\t", 1) for ln in calls.read_text().splitlines()]
    name = {i: re.search(r"--job-name=(\S+)", a).group(1) for i, a in rows}
    rep1 = [i for i, a in rows if "/rep1/" in a]
    sinks = [i for i in rep1 if name[i] in ("rb-tissue", "rb-score", "rb-rescore")]
    assert len(sinks) == 3
    rep2_first = [a for i, a in rows if "/rep2/" in a and i not in rep1][0]
    assert f"--dependency=afterany:{':'.join(sinks)}" in rep2_first


def test_a_repeat_is_the_same_submission_into_its_own_folder(site):
    """T31 R13, REGBENCH_REPEAT=2: the launch twice, into rep1 and rep2 of REGBENCH_OUT
    (each repeat its own results: no repeat reuses another's)."""
    done, _, out = launch(site, env={"REGBENCH_REPEAT": "2"}, log="repeat")
    assert done.returncode == 0, done.stderr
    assert f"REGBENCH_OUT={out}/rep1" in done.stdout and f"REGBENCH_OUT={out}/rep2" in done.stdout
    lines = _sbatch_lines(done.stdout)
    half = [ln for ln in lines if "/rep1" in ln]
    assert half and len(lines) == 2 * len(half)
    # T31 fix M3 (ruling): the repeats run one after the other, never beside each other:
    # every first job of rep2 (one with no dependency of its own) waits afterany for
    # rep1's last jobs, its scorers; nothing of rep1 waits for anything of rep2
    first = [ln for ln in half if "--dependency=" not in ln]
    assert first
    after = " --dependency=afterany:<rb-tissue>:<rb-score>:<rb-rescore>"
    rep2 = [ln for ln in lines if "/rep2" in ln]
    assert all(after in ln for ln in rep2 if ln.replace(after, "").replace("/rep2", "/rep1")
               in first), rep2
    assert sum(after in ln for ln in rep2) == len(first)
    assert sorted(ln.replace("/rep1", "/rep2") for ln in half) == sorted(
        ln.replace(after, "") for ln in rep2)


# ---- the synthetic set (T32: cost only, named only, prepared here) ------------------------

BEFORE_SYNTHETIC = "887e18fc"


def test_the_launchers_cost_only_sets_are_the_clis():
    from benchmarks.regbench import cli

    (line,) = [x for x in code_of(SCRIPT).splitlines() if x.startswith("COST_ONLY_SETS=")]
    assert tuple(line.split("=", 1)[1].strip('"').split()) == cli.COST_ONLY_SETS == ("synthetic",)


def test_cases_there_names_the_synthetic_set_named_only(site, tmp_path):
    """--all (launch.sbatch) launches what --cases-there says yes to: never the synthetic
    set, even with its cases there."""
    done, _, _ = launch(
        site, "--cases-there", env={"REGBENCH_CASES_SYNTHETIC": str(tmp_path)}, log="there-syn"
    )
    assert done.returncode == 0, done.stderr
    (line,) = [x for x in done.stdout.splitlines() if x.startswith("synthetic\t")]
    assert line.split("\t")[1] == "no" and "named only" in line
    assert any(x.startswith("multiplex\tyes") for x in done.stdout.splitlines())


def _syn_source(tmp_path):
    from .test_synth import _source

    return _source(tmp_path)


def test_prepare_synthetic_submits_plan_then_array_then_finish(site, tmp_path):
    to = tmp_path / "syn"
    src = _syn_source(tmp_path)
    env = {"REGBENCH_CASES_SYNTHETIC": str(to), "REGBENCH_SYNTH_SOURCE": str(src),
           "REGBENCH_BIND": str(tmp_path)}
    done, _, _ = launch(site, "--prepare", "--datasets", "synthetic", env=env, log="syn-prep")
    jobs = jobs_of(done)
    assert set(jobs) == {"rb-synth-plan", "rb-synth", "rb-synth-finish"}
    plan, array, finish = jobs["rb-synth-plan"], jobs["rb-synth"], jobs["rb-synth-finish"]
    assert plan.script == array.script == finish.script == "synth.sbatch"
    assert plan.args == ["plan", str(to), "--source", str(src), "--grid", "default", "--seed", "0"]
    # one task per (patient, panel): 5 sizes x 11 panels
    assert array.opt("--array").startswith("0-54%") and array.args == ["unit", str(to)]
    assert finish.args == ["finish", str(to)]
    # the array only after a plan that ended ok; the finish after the array, whatever
    # became of it (it names the slides that are missing)
    assert array.opt("--dependency") == "afterok:<rb-synth-plan>"
    assert finish.opt("--dependency") == "afterany:<rb-synth>"
    # the cohort's rounds are never read for the synthetic set alone
    assert "rb-prepare-multiplex" not in jobs and "rb-nuclei" not in jobs
    small, _, _ = launch(site, "--prepare", "--datasets", "synthetic",
                         env={**env, "REGBENCH_SYNTH_GRID": "small"}, log="syn-small")  # fmt: skip
    assert jobs_of(small)["rb-synth"].opt("--array").startswith("0-10%")
    # a dry run makes no folder
    assert not to.exists()


def test_prepare_synthetic_needs_its_source(site, tmp_path):
    env = {"REGBENCH_CASES_SYNTHETIC": str(tmp_path / "syn")}
    done, _, _ = launch(site, "--prepare", "--datasets", "synthetic", env=env, log="syn-nosrc")
    assert done.returncode == 2 and "REGBENCH_SYNTH_SOURCE" in done.stderr
    assert not _sbatch_lines(done.stdout)
    unset, _, _ = launch(site, "--prepare", "--datasets", "synthetic", log="syn-novar")
    assert unset.returncode == 2 and "REGBENCH_CASES_SYNTHETIC is not set" in unset.stderr


def test_prepare_synthetic_continues_a_planned_set_and_reuses_a_complete_one(site, tmp_path):
    from benchmarks.regbench import synth

    from .test_synth import _plan

    src = _syn_source(tmp_path)
    to = _plan(tmp_path, src, sizes=(64,))
    env = {"REGBENCH_CASES_SYNTHETIC": str(to)}  # no source: a planned set needs none
    done, _, _ = launch(site, "--prepare", "--datasets", "synthetic", env=env, log="syn-planned")
    jobs = jobs_of(done)
    assert set(jobs) == {"rb-synth", "rb-synth-finish"}
    # the plan's units (1 size x 11 panels), not the grid's
    assert jobs["rb-synth"].opt("--array").startswith("0-10%")
    assert "planned, not complete" in done.stdout
    for i in range(synth.n_units(to)):
        synth.render_unit(to, i)
    synth.finish(to)
    again, _, _ = launch(site, "--prepare", "--datasets", "synthetic", env=env, log="syn-complete")
    assert again.returncode == 0, again.stderr
    assert not _sbatch_lines(again.stdout) and "complete: reused" in again.stdout


def test_prepare_synthetic_refuses_a_folder_without_its_plan(site, tmp_path):
    (tmp_path / "synthetic").mkdir()
    env = {"REGBENCH_CASES_SYNTHETIC": str(tmp_path), "REGBENCH_SYNTH_SOURCE": "x.tif"}
    done, _, _ = launch(site, "--prepare", "--datasets", "synthetic", env=env, log="syn-partial")
    assert done.returncode == 2 and "without its plan" in done.stderr
    assert not _sbatch_lines(done.stdout)


def test_a_synthetic_launch_runs_its_list_and_has_no_case_or_tissue_scoring(site, tmp_path):
    cases = tmp_path / "syn_cases"
    _case(cases, "synthetic", "s04096_r01", "s04096")
    _case(cases, "synthetic", "s04096_r02", "s04096")
    done, _, _ = launch(site, "--datasets", "synthetic",
                        env={"REGBENCH_CASES_SYNTHETIC": str(cases)}, log="syn-launch")  # fmt: skip
    jobs = jobs_of(done)
    ran = {j.args[0] for j in [*jobs.values(), *jobs.later.values()]
           if j.script == "run_rung.sbatch"}  # fmt: skip
    assert ran and ran <= set(variants.SYNTHETIC_RUNS)
    assert {"stare", "valis_mirage", "valis120_hires"} <= ran
    assert not any(n.startswith("rb-score-case") for n in jobs) and "rb-tissue" not in jobs
    assert jobs["rb-score"].args == ["score", "synthetic"]
    assert "<rb-settle>" in jobs["rb-score"].opt("--dependency")
    # beside the cohort: the cohort keeps its per-case and tissue jobs, synthetic has none
    both, _, _ = launch(site, "--datasets", "multiplex synthetic",
                        env={"REGBENCH_CASES_SYNTHETIC": str(cases)}, log="syn-both")  # fmt: skip
    jb = jobs_of(both)
    assert "rb-score-case-multiplex" in jb and "rb-score-case-synthetic" not in jb
    assert jb["rb-tissue"].args == ["tissue", "multiplex", "synthetic"]
    assert jb["rb-score"].args == ["score", "multiplex", "synthetic"]


def test_the_default_launch_is_byte_identical_to_the_launcher_before_the_synthetic_set(
    site, dry, tmp_path
):
    """T32 adds the synthetic set to the launcher; the default launch (the cohort alone)
    submits and plans exactly what the launcher at BASE did."""
    have = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-e", f"{BEFORE_SYNTHETIC}^{{commit}}"],
        capture_output=True,
    )
    if have.returncode != 0:
        pytest.skip(f"{BEFORE_SYNTHETIC} is not in this clone (shallow)")
    old = tmp_path / "old"
    old.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", BEFORE_SYNTHETIC, "benchmarks/regbench/slurm"],
        capture_output=True, check=True,
    )  # fmt: skip
    subprocess.run(["tar", "-x", "-C", str(old)], input=archive.stdout, check=True)
    old_slurm = old / "benchmarks" / "regbench" / "slurm"
    argv, base, _ = _launch_command(site, (), {"REGBENCH_REPO": str(REPO)}, True, "before-syn")
    argv[1] = str(old_slurm / "submit_cohort.sh")
    done = subprocess.run(argv, env=base, capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stderr

    def norm(text):
        return [
            ln.replace(str(old_slurm), str(SLURM)).replace("out-before-syn", "out-calls")
            for ln in text.splitlines()
            if ln.startswith(("sbatch ", LATER))
        ]

    now = norm(dry[1].stdout)
    assert now and norm(done.stdout) == now


def test_prepare_synthetic_refuses_a_source_the_images_cannot_read(site, tmp_path):
    """REGBENCH_SYNTH_SOURCE must lie (its realpath) under REGBENCH_INPUTS, REGBENCH_CASES
    or a folder of REGBENCH_BIND: the plan job reads it inside the image."""
    src = _syn_source(tmp_path)
    to = tmp_path / "syn"
    base = {"REGBENCH_CASES_SYNTHETIC": str(to), "REGBENCH_SYNTH_SOURCE": str(src)}
    done, _, _ = launch(site, "--prepare", "--datasets", "synthetic", env=base, log="syn-unbound")
    assert done.returncode == 2 and "REGBENCH_SYNTH_SOURCE" in done.stderr
    assert "REGBENCH_BIND" in done.stderr and not _sbatch_lines(done.stdout)
    # under one folder of a comma-separated REGBENCH_BIND
    ok, _, _ = launch(site, "--prepare", "--datasets", "synthetic",
                      env={**base, "REGBENCH_BIND": f"/nowhere,{tmp_path}"}, log="syn-bind")  # fmt: skip
    assert "rb-synth-plan" in jobs_of(ok)
    # under REGBENCH_INPUTS
    ok2, _, _ = launch(site, "--prepare", "--datasets", "synthetic",
                       env={**base, "REGBENCH_INPUTS": str(tmp_path)}, log="syn-inputs")  # fmt: skip
    assert "rb-synth-plan" in jobs_of(ok2)
    # a symlink under a bound folder to a file outside every one is refused (its realpath)
    bound = tmp_path / "bound"
    bound.mkdir()
    (bound / "link.ome.tif").symlink_to(src)
    link, _, _ = launch(site, "--prepare", "--datasets", "synthetic",
                        env={**base, "REGBENCH_SYNTH_SOURCE": str(bound / "link.ome.tif"),
                             "REGBENCH_BIND": str(bound)}, log="syn-link")  # fmt: skip
    assert link.returncode == 2 and "REGBENCH_SYNTH_SOURCE" in link.stderr


def test_the_synthetic_plan_jobs_time_is_its_own_setting(site, tmp_path):
    src = _syn_source(tmp_path)
    env = {"REGBENCH_CASES_SYNTHETIC": str(tmp_path / "syn"), "REGBENCH_SYNTH_SOURCE": str(src),
           "REGBENCH_BIND": str(tmp_path)}
    default, _, _ = launch(site, "--prepare", "--datasets", "synthetic", env=env, log="syn-pt0")
    assert jobs_of(default)["rb-synth-plan"].opt("--time") == "0-04:00:00"
    longer, _, _ = launch(site, "--prepare", "--datasets", "synthetic",
                          env={**env, "REGBENCH_SYNTH_PLAN_TIME": "10:30:00"}, log="syn-pt1")  # fmt: skip
    jobs = jobs_of(longer)
    assert jobs["rb-synth-plan"].opt("--time") == "0-10:30:00"
    # the array's and the finish's limits are not that one
    assert jobs["rb-synth"].opt("--time") == "0-04:00:00"
    assert jobs["rb-synth-finish"].opt("--time") == "0-01:00:00"


def test_a_synthetic_launch_prints_its_results_estimate_and_the_free_space(site, tmp_path):
    """Final review I1: before the first sbatch, the bytes STARE's uncompressed rounds
    will take (synth.results_estimate, over the selected STARE runs of the set) and the
    free space of REGBENCH_OUT. Printed only: nothing is refused."""
    from benchmarks.regbench import synth

    cases = tmp_path / "syn_cases"
    _case(cases, "synthetic", "s04096_r01", "s04096")
    env = {"REGBENCH_CASES_SYNTHETIC": str(cases)}
    done, _, out = launch(site, "--datasets", "synthetic", env=env, log="syn-disk")
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    est = synth.results_estimate(synth.GRIDS["default"], 9)
    (said,) = [i for i, x in enumerate(lines) if x.startswith("# synthetic: results")]
    assert str(est["total_bytes"]) in lines[said] and "9 STARE run(s)" in lines[said]
    (free,) = [i for i, x in enumerate(lines) if x.startswith("# synthetic: free space of REGBENCH_OUT")]
    assert str(out) in lines[free] and "bytes" in lines[free]
    first = next(i for i, x in enumerate(lines) if x.startswith("sbatch "))
    assert said < first and free < first
    one, _, _ = launch(site, "--datasets", "synthetic", "--runs", "stare valis_mirage",
                       env=env, log="syn-disk1")  # fmt: skip
    one_est = synth.results_estimate(synth.GRIDS["default"], 1)
    assert any(
        x.startswith("# synthetic: results") and str(one_est["total_bytes"]) in x
        for x in one.stdout.splitlines()
    )
    # the cohort's launch says nothing of it
    cohort, _, _ = launch(site, log="syn-disk-cohort")
    assert "# synthetic:" not in cohort.stdout
