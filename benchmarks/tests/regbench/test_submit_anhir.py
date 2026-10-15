"""submit_cohort.sh on ANHIR's training pairs: ``--datasets anhir`` and ``anhir_tune``.

Read off the dry run, like ``test_submit_hyreco.py``.
"""

from __future__ import annotations

import pytest

from benchmarks.regbench import prep, variants

from .test_submit_cohort import (
    SCRIPT,
    _case,
    first_rung,
    jobs_of,
    launch,
    launch_jobs,
)

SPECS = {variants.run_name(r): r for r in variants.runs_for("anhir")}
RUNS = list(SPECS)
TUNING = {variants.run_name(r): r for r in variants.TUNING_RUNS}
CANDIDATES = list(TUNING)


def submitted(specs):
    """The jobs one launch of these runs submits, read off the run table and the job graph:
    per run that gates a run (and the run it is anchored on) the selfcheck's first attempt
    and its watcher (the later attempts are submitted only after a failure), per run its
    gate (advisory or not, the gate job is there), the run's own jobs, and at the end the
    settle job and the set's scorer."""
    specs = list(specs)
    gating = {variants.gated_by(variants.run_name(r)) for r in specs}
    gating |= {
        variants.run_of(g).after for g in list(gating) if variants.run_of(g).after
    }
    return 2 * len(gating) + len(specs) + sum(launch_jobs(r) for r in specs) + 2


SOURCE = {"REGBENCH_ANHIR": "/data/anhir"}  # bound into every image: the cases name it
# the tuning side is PAUSED (variants.TUNING_PAUSED): submitted only when asked for
ASKED = {**SOURCE, "REGBENCH_ANHIR_TUNE": "1"}


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
    assert laddered == set(RUNS)
    # ... the settle job and the set's own scorer
    assert len(jobs) == submitted(SPECS.values())
    assert "# anhir: 4 case(s), 4 patient(s), at most 1 round(s)" in done.stdout
    for run in RUNS:
        first = jobs[first_rung(SPECS[run], "anhir")]
        assert first.opt("--array") == "0-3%20" and first.opt("--cpus-per-task") == "4"
    # no tissue job, no unit sent ahead for a realcheck, the set's own tables
    assert not [j for j in jobs if "tissue" in j or "realgate" in j]
    score = jobs["rb-score-anhir"]
    assert score.args == ["score", "anhir"]
    assert f"{out}/tables/anhir/summary.md" in done.stdout
    assert "TRAINING pairs" in done.stdout and "leaderboard" in done.stdout
    left = [ln for ln in done.stdout.splitlines() if " is not run (" in ln]
    # one line per active run the set leaves out (`excluded_for`); a run of another regime
    # (the cyclic-IF cells) is not on this set's list at all and is not "left out" of it
    gone = dict(variants.excluded_for("anhir"))
    assert {ln.split()[2] for ln in left} == set(gone) and len(left) == len(gone)
    assert not set(gone) & set(RUNS)
    assert any(ln.startswith("# anhir: palom is not run (") for ln in left)
    # the GPU rows ask for the site's GPU, the others for none
    assert jobs["rb-deeperhistreg_hr_gpu-anhir-16g"].opt("--gres")
    assert not jobs[first_rung(SPECS["stare"], "anhir")].opt("--gres")


def test_the_paused_preprocessing_row_is_not_submitted_and_so_not_warned_about(site):
    """`stare_prep` is paused (2026-10-14): no launch holds it, so the provisional-candidate
    warning, which is printed for a SELECTED run only, has nothing to say; the row cannot be
    asked for by name either (a paused run is not in `runs`). The tuning side still runs
    its candidates, and still ends in the frozen record the row would read."""
    assert not variants.run_of(prep.ROW).active
    for args in (("--datasets", "anhir"), ("--runs", "stare")):
        done, _, _ = launch(site, *args, env=SOURCE, log="warn")
        assert done.returncode == 0, done.stderr
        assert "WARNING" not in done.stdout
    done, _, _ = launch(site, "--runs", f"stare {prep.ROW}", log="cohort")
    assert (
        done.returncode != 0
        and f"no run named '{prep.ROW}'" in done.stderr + done.stdout
    )


def test_the_tuning_side_is_paused_and_submitted_only_when_asked_for(site):
    """`anhir_tune` is paused with `stare_prep`, the only row that reads it: the
    launcher refuses it, submitting nothing, unless REGBENCH_ANHIR_TUNE=1 asks for it
    (the explicit path, whose jobs the next test counts)."""
    assert variants.TUNING_PAUSED
    for env in (SOURCE, {**SOURCE, "REGBENCH_ANHIR_TUNE": "0"}):
        done, log, _ = launch(site, "--datasets", "anhir_tune", env=env, log="paused")
        assert done.returncode == 2 and "PAUSED" in done.stderr
        assert "REGBENCH_ANHIR_TUNE=1" in done.stderr
        assert not log.exists() and "sbatch" not in done.stdout
    # ANHIR's report runs are not paused
    done, _, _ = launch(site, "--datasets", "anhir", env=SOURCE, log="notpaused")
    assert done.returncode == 0, done.stderr


def test_the_tuning_side_runs_the_candidates_and_nothing_else(site):
    """The explicit path (REGBENCH_ANHIR_TUNE=1): the job count of the paused side."""
    done, _, out = launch(site, "--datasets", "anhir_tune", env=ASKED, log="tune")
    assert done.returncode == 0, done.stderr
    jobs = jobs_of(done)
    laddered = {j.args[0] for j in jobs.values() if j.script == "run_rung.sbatch"}
    assert laddered == set(CANDIDATES) and len(CANDIDATES) == len(prep.CANDIDATES)
    checked = {j.args[0] for j in jobs.values() if j.script == "selfcheck.sbatch"}
    assert checked == set(CANDIDATES)  # each candidate behind its own selfcheck
    assert len(jobs) == submitted(TUNING.values())
    for name, c in zip(CANDIDATES, prep.CANDIDATES):
        first = jobs[first_rung(TUNING[name], "anhir_tune")]
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
        env = {"REGBENCH_ANHIR_TUNE": "1"}  # the paused side, asked for
        done, log, _ = launch(site, "--datasets", ds, env=env, log="nosource")
        assert (
            done.returncode == 2
            and "needs REGBENCH_ANHIR for the runs too" in done.stderr
        )
        assert not log.exists() and "sbatch" not in done.stdout
    text = (SCRIPT.parent / "env.sh").read_text()
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert 'bind="$bind${REGBENCH_ANHIR:+,$REGBENCH_ANHIR}"' in code
