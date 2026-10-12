"""submit_cohort.sh on an external landmark set: the job graph of ``--datasets hyreco_b``.

Read off the dry run, like ``test_submit_cohort.py`` (whose stand-in ``sbatch`` and
helpers are used here). The fixture is three prepared pairs of each subset.
"""

from __future__ import annotations

import pytest

from benchmarks.regbench import ladder, variants

from .test_submit_cohort import (
    SCRIPT,
    _case,
    _sbatch_lines,
    code_of,
    first_rung,
    jobs_of,
    last_job,
    launch,
    launch_jobs,
    rungs,
)

RUNS = ["stare", "valis120", "valis120_hires"]
DATA = {
    "REGBENCH_HYRECO_A": "/data/HyReCo",
    "REGBENCH_HYRECO_B_HE": "/data/HyReCo-Additional/HE",
    "REGBENCH_HYRECO_B_PHH3": "/data/HyReCo-Additional/PHH3",
}


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hyreco_site")
    cases = tmp / "cases"
    for ds in ("hyreco_a", "hyreco_b"):
        for cid in ("8", "29", "31"):
            _case(cases, ds, cid, cid)  # each pair is its own patient
    _case(cases, "multiplex", "P1_a", "P1")
    fake = tmp / "bin"
    fake.mkdir()
    sbatch = fake / "sbatch"
    sbatch.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$SBATCH_LOG"\necho "1;cluster"\n'
    )
    sbatch.chmod(0o755)
    return tmp, cases, fake


@pytest.fixture(scope="module")
def dry(site):
    done, log, out = launch(site, "--datasets", "hyreco_b", log="hb")
    return jobs_of(done), done, log, out


def test_the_dry_run_submits_nothing_and_says_where_the_results_go(dry):
    jobs, done, log, out = dry
    assert not log.exists()
    # per run: three selfcheck attempts, a gate, and its own jobs (the ladder's eight;
    # for STARE its stage arrays, tile submitters and last job); then the settle job
    # and the set's scorer
    own = [launch_jobs(variants.run_of(r)) for r in RUNS]
    assert own == [18, 8, 8] and len(jobs) == sum(3 + 1 + n for n in own) + 2 == 48
    assert f"# {len(jobs)} job(s) would be submitted; nothing was" in done.stdout
    assert "# hyreco_b: 3 case(s), 3 patient(s), at most 1 round(s)" in done.stdout
    assert f"{out}/tables/hyreco_b/summary.md" in done.stdout


def test_only_the_runs_with_a_brightfield_rule_are_submitted(dry):
    jobs, done, _, _ = dry
    laddered = {j.args[0] for j in jobs.values() if j.script == "run_rung.sbatch"}
    assert laddered == set(RUNS)
    checked = {j.args[0] for j in jobs.values() if j.script == "selfcheck.sbatch"}
    assert checked == set(RUNS)
    # every other run of the cohort's list is named with its reason, not submitted
    left = [ln for ln in done.stdout.splitlines() if " is not run (" in ln]
    assert len(left) == len(variants.active_runs()) - len(RUNS) == 15
    assert any(ln.startswith("# hyreco_b: ashlar is not run (") for ln in left)


def test_each_run_has_the_existing_ladder_on_the_set(dry):
    jobs, _, _, _ = dry
    for run in RUNS:
        ladder_of = rungs(jobs, run, "hyreco_b")
        assert [j.opt("--mem") for j in ladder_of] == [
            f"{gb}G" for gb in ladder.RUNGS_GB
        ]
        assert all(j.opt("--cpus-per-task") == "4" for j in ladder_of)
        # three pairs, one task each: a pair is its own patient for the VALIS rows too
        assert all(j.opt("--array") == "0-2%20" for j in ladder_of)
        assert all(j.args[6] == "hyreco_b" for j in ladder_of)
        assert ("afterok", f"<rb-gate-{run}>") in ladder_of[0].deps
        for below, above in zip(ladder_of, ladder_of[1:]):
            assert ("afterany", f"<{below.name}-again>") in above.deps
    # no run of the set needs a GPU
    assert all(j.args[5] == "0" for j in jobs.values() if j.script == "run_rung.sbatch")
    assert not any(w.startswith("--gres") for j in jobs.values() for w in j.words)


def test_no_unit_goes_ahead_for_a_realcheck_and_no_tissue_job_is_submitted(dry):
    """Both are defined on nuclei, which these cases do not have."""
    jobs, _, _, _ = dry
    assert not [n for n in jobs if "realgate" in n or n.endswith("-first-16g")]
    assert all(j.args[9] == "0" for j in jobs.values() if j.script == "run_rung.sbatch")
    assert "rb-tissue" not in jobs and "rb-score" not in jobs


def test_the_scorer_is_the_sets_own_and_waits_for_everything(dry):
    jobs, _, _, _ = dry
    score = jobs["rb-score-hyreco_b"]
    assert score.script == "score_cohort.sbatch" and score.args == ["score", "hyreco_b"]
    # through the settle job, which waits for every ladder and gate
    assert score.deps == {("afterany", "<rb-settle>")}
    settle = jobs["rb-settle"]
    assert settle.args == ["settle"]
    waited = {name for kind, name in settle.deps if kind == "afterany"}
    for run in RUNS:
        assert f"<{last_job(run, 'hyreco_b')}>" in waited
        assert f"<rb-gate-{run}>" in waited
    code = code_of(SCRIPT.parent / "score_cohort.sbatch")
    assert 'args+=(--dataset "$external")' in code


def test_the_tuning_set_is_submitted_the_same_way_and_labelled(site):
    done, _, out = launch(site, "--datasets", "hyreco_a", log="ha")
    jobs = jobs_of(done)
    assert jobs["rb-score-hyreco_a"].args == ["score", "hyreco_a"]
    assert first_rung(variants.run_of("stare"), "hyreco_a") in jobs
    assert "# hyreco_a is the TUNING set: nothing in its tables is a result" in (
        done.stdout
    )


@pytest.mark.parametrize(
    "named", ["hyreco_b multiplex", "hyreco_a hyreco_b", "multiplex hyreco_a"]
)
def test_an_external_set_is_never_submitted_with_anything_else(site, named):
    done, log, _ = launch(site, "--datasets", named, log="mixed")
    assert done.returncode == 2 and not _sbatch_lines(done.stdout)
    assert (
        "is submitted alone" in done.stderr and "Nothing was submitted" in done.stderr
    )
    assert not log.exists()


def test_the_cohorts_default_submission_has_no_external_set(site):
    done, _, _ = launch(site, log="default")
    jobs = jobs_of(done)
    assert not [n for n in jobs if "hyreco" in n]
    assert "rb-tissue" in jobs and "rb-score" in jobs


def test_prepare_is_one_pair_per_task_with_the_subsets_own_arguments(site):
    done, _, _ = launch(site, "--prepare", "--datasets", "hyreco_a", env=DATA, log="pa")
    (job,) = jobs_of(done).values()
    assert job.name == "rb-prepare-hyreco_a" and job.opt("--array") == "0-8%20"
    assert job.args == ["hyreco", "--subset", "a", "--data-root", "/data/HyReCo"]
    tried = {**DATA, "REGBENCH_HYRECO_A_ARGS": "--proxy haematoxylin --reduce 2"}
    done, _, _ = launch(
        site, "--prepare", "--datasets", "hyreco_a", env=tried, log="pa2"
    )
    (job,) = jobs_of(done).values()
    assert job.args[-4:] == ["--proxy", "haematoxylin", "--reduce", "2"]

    # the tuning set's trial arguments are still exported: the report set ignores them
    done, _, _ = launch(
        site, "--prepare", "--datasets", "hyreco_b", env=tried, log="pb"
    )
    (job,) = jobs_of(done).values()
    assert job.name == "rb-prepare-hyreco_b" and job.opt("--array") == "0-53%20"
    cases = site[1]
    assert job.args == [
        "hyreco", "--subset", "b",
        "--ref-dir", DATA["REGBENCH_HYRECO_B_HE"],
        "--mov-dir", DATA["REGBENCH_HYRECO_B_PHH3"],
        "--settings-from", str(cases),
    ]  # fmt: skip
    # the report set never gets the tuning set's trial arguments
    assert "--proxy" not in job.args and "--reduce" not in job.args
    code = code_of(SCRIPT.parent / "prepare.sbatch")
    assert '--index "$SLURM_ARRAY_TASK_ID"' in code


def test_prepare_takes_the_tuned_cases_from_where_they_are(site):
    env = {**DATA, "REGBENCH_HYRECO_TUNED": "/scratch/tuned_cases"}
    done, _, _ = launch(site, "--prepare", "--datasets", "hyreco_b", env=env, log="pt")
    (job,) = jobs_of(done).values()
    assert job.args[job.args.index("--settings-from") + 1] == "/scratch/tuned_cases"


@pytest.mark.parametrize(
    "ds, missing",
    [("hyreco_a", "REGBENCH_HYRECO_A"), ("hyreco_b", "REGBENCH_HYRECO_B_PHH3")],
)
def test_prepare_without_the_data_folders_is_refused(site, ds, missing):
    env = {k: v for k, v in DATA.items() if k != missing}
    done, log, _ = launch(site, "--prepare", "--datasets", ds, env=env, log="pm")
    assert done.returncode == 2 and missing in done.stderr
    assert not _sbatch_lines(done.stdout) and not log.exists()


def test_a_real_submission_goes_through_sbatch(site):
    env = {"REGBENCH_ALLOW_PATH_PYTHON": "1"}
    done, log, out = launch(
        site, "--datasets", "hyreco_b", env=env, dry=False, log="real"
    )
    assert done.returncode == 0, done.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == len(dry_count(site))
    assert sum("score_cohort.sbatch score hyreco_b" in c for c in calls) == 1
    assert (out / "logs" / "jobs.tsv").exists()


def dry_count(site):
    done, _, _ = launch(site, "--datasets", "hyreco_b", log="count")
    return _sbatch_lines(done.stdout)
