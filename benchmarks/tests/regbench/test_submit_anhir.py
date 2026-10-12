"""submit_cohort.sh on ANHIR's training pairs: ``--datasets anhir`` and ``anhir_tune``.

Read off the dry run, like ``test_submit_hyreco.py``.
"""

from __future__ import annotations

import pytest

from benchmarks.regbench import ladder, prep, variants

from .test_submit_cohort import SCRIPT, _case, jobs_of, launch

RUNS = [variants.run_name(r) for r in variants.runs_for("anhir")]
CANDIDATES = [variants.run_name(r) for r in variants.TUNING_RUNS]
PER_RUN = len(ladder.SELFCHECK_RUNGS_GB) + 1 + 2 * len(ladder.RUNGS_GB)
SOURCE = {"REGBENCH_ANHIR": "/data/anhir"}  # bound into every image: the cases name it


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("anhir_site")
    cases = tmp / "cases"
    for cid in ("000", "001", "002", "003"):
        _case(cases, "anhir", cid, cid)  # each pair is its own group
    for cid in ("000", "002"):
        _case(cases, "anhir_tune", cid, cid)
    _case(cases, "multiplex", "P1_a", "P1")
    fake = tmp / "bin"
    fake.mkdir()
    sbatch = fake / "sbatch"
    sbatch.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$SBATCH_LOG"\necho "1;cluster"\n'
    )
    sbatch.chmod(0o755)
    return tmp, cases, fake


def test_anhir_runs_its_own_list_on_every_training_pair(site):
    done, log, out = launch(site, "--datasets", "anhir", env=SOURCE, log="anhir")
    assert done.returncode == 0, done.stderr
    jobs = jobs_of(done)
    assert not log.exists()
    laddered = {j.args[0] for j in jobs.values() if j.script == "run_rung.sbatch"}
    assert laddered == set(RUNS) and len(RUNS) == 8
    # ... the settle job and the set's own scorer
    assert len(jobs) == len(RUNS) * PER_RUN + 2
    assert "# anhir: 4 case(s), 4 patient(s), at most 1 round(s)" in done.stdout
    for run in RUNS:
        first = jobs[f"rb-{run}-anhir-16g"]
        assert first.opt("--array") == "0-3%20" and first.opt("--cpus-per-task") == "4"
    # no tissue job, no unit sent ahead for a realcheck, the set's own tables
    assert not [j for j in jobs if "tissue" in j or "realgate" in j]
    score = jobs["rb-score-anhir"]
    assert score.args == ["score", "anhir"]
    assert f"{out}/tables/anhir/summary.md" in done.stdout
    assert "TRAINING pairs" in done.stdout and "leaderboard" in done.stdout
    left = [ln for ln in done.stdout.splitlines() if " is not run (" in ln]
    assert len(left) == len(variants.active_runs()) - len(RUNS)
    assert any(ln.startswith("# anhir: palom is not run (") for ln in left)
    # the GPU rows ask for the site's GPU, the others for none
    assert jobs["rb-deeperhistreg_hr_gpu-anhir-16g"].opt("--gres")
    assert not jobs["rb-stare_prep-anhir-16g"].opt("--gres")


def test_the_provisional_preprocessing_is_said_aloud(site):
    done, _, _ = launch(site, "--datasets", "anhir", env=SOURCE, log="warn")
    warned = [ln for ln in done.stdout.splitlines() if ln.startswith("# WARNING")]
    record = prep.load_frozen()
    if record["status"] == "provisional":
        assert len(warned) == 1 and "PROVISIONAL" in warned[0]
        assert f"'{record['selected']}'" in warned[0] and prep.ROW in warned[0]
    else:
        assert not warned and "selected on ANHIR's tuning side" in done.stdout
    # a submission without that row says nothing about it
    done, _, _ = launch(
        site, "--datasets", "anhir", "--runs", "stare", env=SOURCE, log="quiet"
    )
    assert done.returncode == 0 and "WARNING" not in done.stdout
    # and the cohort's default, which holds the row, says it too
    done, _, _ = launch(site, "--runs", f"stare {prep.ROW}", log="cohort")
    assert done.returncode == 0, done.stderr
    assert ("# WARNING" in done.stdout) == (record["status"] == "provisional")
    assert f"rb-{prep.ROW}-multiplex-16g" in jobs_of(done)


def test_the_tuning_side_runs_the_candidates_and_nothing_else(site):
    done, _, out = launch(site, "--datasets", "anhir_tune", env=SOURCE, log="tune")
    assert done.returncode == 0, done.stderr
    jobs = jobs_of(done)
    laddered = {j.args[0] for j in jobs.values() if j.script == "run_rung.sbatch"}
    assert laddered == set(CANDIDATES) and len(CANDIDATES) == len(prep.CANDIDATES)
    checked = {j.args[0] for j in jobs.values() if j.script == "selfcheck.sbatch"}
    assert checked == set(CANDIDATES)  # each candidate behind its own selfcheck
    assert len(jobs) == len(CANDIDATES) * PER_RUN + 2
    for name, c in zip(CANDIDATES, prep.CANDIDATES):
        first = jobs[f"rb-{name}-anhir_tune-16g"]
        assert first.args[1:4] == ["stare", name, f"prep={c}"]
        assert first.opt("--array") == "0-1%20"  # the two tuning-side pairs
    assert "Nothing in its tables is a result".lower() in done.stdout.lower()
    assert "prep-select --cases" in done.stdout and "--freeze" in done.stdout
    assert "WARNING" not in done.stdout  # the frozen row is not part of this list
    assert jobs["rb-score-anhir_tune"].args == ["score", "anhir_tune"]


@pytest.mark.parametrize(
    "named", ["anhir multiplex", "anhir anhir_tune", "hyreco_b anhir"]
)
def test_anhir_is_never_submitted_with_anything_else(site, named):
    done, log, _ = launch(site, "--datasets", named, env=SOURCE, log="mixed")
    assert done.returncode == 2 and "submitted alone" in done.stderr
    assert not log.exists() and "sbatch" not in done.stdout


def test_the_default_submission_holds_no_anhir_job(site):
    done, _, _ = launch(site, log="default")
    assert done.returncode == 0, done.stderr
    assert not [name for name in jobs_of(done) if "anhir" in name]


def test_prepare_is_one_pair_per_task_from_the_source_folder(site, tmp_path):
    from .test_anhir import build

    done, _, _ = launch(site, "--prepare", "--datasets", "anhir", log="noprep")
    assert done.returncode == 2 and "needs REGBENCH_ANHIR" in done.stderr
    # sized from the cover table: the tracked split refuses the stand-in's table, so
    # the size is asked of the real one where it is on this machine
    from .test_anhir import REAL

    if not (REAL / "dataset_medium.csv").is_file():
        source = build(tmp_path / "source")
        done, _, _ = launch(
            site,
            "--prepare",
            "--datasets",
            "anhir",
            env={"REGBENCH_ANHIR": str(source)},
        )
        assert (
            done.returncode == 2
            and "cannot size the anhir prepare array" in done.stderr
        )
        return
    env = {"REGBENCH_ANHIR": str(REAL), "REGBENCH_ANHIR_ARGS": "--allow-incomplete"}
    done, _, _ = launch(site, "--prepare", "--datasets", "anhir", env=env, log="prep")
    assert done.returncode == 0, done.stderr
    (job,) = jobs_of(done).values()
    assert job.name == "rb-prepare-anhir" and job.opt("--array") == "0-229%20"
    assert job.args == ["anhir", "--source", str(REAL), "--allow-incomplete"]


def test_the_runs_need_the_source_folder_too(site):
    """The cases name the original images inside it; without it bound, VALIS,
    DeeperHistReg and the preprocessing row would fail every pair while plain STARE
    (cached grey images) would not."""
    for ds in ("anhir", "anhir_tune"):
        done, log, _ = launch(site, "--datasets", ds, log="nosource")
        assert (
            done.returncode == 2
            and "needs REGBENCH_ANHIR for the runs too" in done.stderr
        )
        assert not log.exists() and "sbatch" not in done.stdout
    text = (SCRIPT.parent / "env.sh").read_text()
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert 'bind="$bind${REGBENCH_ANHIR:+,$REGBENCH_ANHIR}"' in code
