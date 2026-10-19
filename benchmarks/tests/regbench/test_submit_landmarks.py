"""submit_cohort.sh on the public LANDMARK set (``hyreco``, the only one since ANHIR was
removed by the user's ruling of 2026-10-19): read off the dry run with
``test_submit_cohort.py``'s stand-in ``sbatch`` and helpers.

Rulings 2026-10-19: the set runs its own 50-run list (STARE's whole tier x stride x
level cross over ``bf_same``'s five levels, the other methods at their defaults), STARE
first in the default order, and is scored into its own tables; ``hyreco`` is one dataset
and HYRECO_ADDITIONAL is a download switch the launcher never reads.
"""

from __future__ import annotations

import pytest

from benchmarks.regbench import variants

from .test_submit_cohort import SCRIPT, _case, _sbatch_lines, code_of, jobs_of, launch

SETS = ("hyreco",)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("landmark_site")
    cases = tmp / "cases"
    _case(cases, "multiplex", "P1_a", "P1")
    for cid in ("a_29", "a_533", "b_29"):
        _case(tmp / "hyreco_cases", "hyreco", cid, cid)
    fake = tmp / "bin"
    fake.mkdir()
    sbatch = fake / "sbatch"
    sbatch.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$SBATCH_LOG"\necho "1;cluster"\n'
    )
    sbatch.chmod(0o755)
    return tmp, cases, fake


def _env(site, ds, **more):
    return {f"REGBENCH_CASES_{ds.upper()}": str(site[0] / f"{ds}_cases"), **more}


def _laddered(jobs):
    """Run names in the order their ladders were submitted."""
    every = [*jobs.values(), *jobs.later.values()]
    return list(
        dict.fromkeys(j.args[0] for j in every if j.script == "run_rung.sbatch")
    )


@pytest.mark.parametrize("ds", SETS)
def test_the_set_runs_its_own_50_runs_stare_first(site, ds):
    done, log, out = launch(
        site,
        "--datasets",
        ds,
        env=_env(site, ds, REGBENCH_RUN_ORDER=None),
        log=f"{ds}-default",
    )
    assert done.returncode == 0, done.stderr
    assert not log.exists()
    ran = _laddered(jobs_of(done))
    want = [variants.run_name(r) for r in variants.runs_for(ds)]
    assert len(ran) == 50 and set(ran) == set(want)
    assert sum(variants.family(r) == "stare" for r in ran) == 45
    assert "stare_haem_none" in ran  # the manuscript's HyReCo STARE row
    assert ran[0] == "stare" and variants.family(ran[0]) == "stare"
    first_other = next(i for i, r in enumerate(ran) if variants.family(r) != "stare")
    assert all(variants.family(r) == "stare" for r in ran[:first_other])
    (line,) = [
        x for x in done.stdout.splitlines() if x.startswith(f"# run order ({ds}): ")
    ]
    assert line.split(": ", 1)[1].split() == ran
    assert f"# {ds}: 50 run(s) of its own list" in done.stdout
    assert f"{out}/tables/{ds}/" in done.stdout


@pytest.mark.parametrize("ds", SETS)
def test_the_set_is_scored_alone_into_its_own_tables(site, ds):
    done, _, _ = launch(site, "--datasets", ds, env=_env(site, ds), log=f"{ds}-score")
    jobs = jobs_of(done)
    assert jobs["rb-score"].args == ["score", ds]
    # no per-case record and no tissue job: the landmark scorer reads neither (final
    # review I3), so the scorer waits for the settle job alone
    assert f"rb-score-case-{ds}" not in jobs and "rb-tissue" not in jobs
    assert not any(n.startswith("rb-score-case") for n in jobs)
    assert jobs["rb-score"].deps == {("afterany", "<rb-settle>")}
    assert not any(
        j.args[6] == "multiplex" for j in jobs.values() if j.script == "run_rung.sbatch"
    )


def test_hyreco_additional_is_never_read_by_the_launcher(site):
    plain, _, _ = launch(
        site, "--datasets", "hyreco", env=_env(site, "hyreco"), log="hy-0"
    )
    extra, _, _ = launch(
        site,
        "--datasets",
        "hyreco",
        env=_env(site, "hyreco", HYRECO_ADDITIONAL="1"),
        log="hy-1",
    )
    assert plain.returncode == extra.returncode == 0
    assert len(_sbatch_lines(plain.stdout)) == len(_sbatch_lines(extra.stdout))
    assert "HYRECO_ADDITIONAL" not in code_of(SCRIPT)


@pytest.mark.parametrize("old", ["hyreco_a", "hyreco_b", "anhir"])
def test_the_old_landmark_sets_are_no_dataset(site, old):
    done, _, _ = launch(site, "--datasets", old, log=f"old-{old}")
    assert done.returncode == 2 and "no such dataset" in done.stderr
    assert "the public datasets are hyreco, mcmicro, phenocycler, rashid" in done.stderr
    assert not _sbatch_lines(done.stdout)


def test_cases_there_says_yes_for_prepared_landmark_sets(site):
    env = _env(site, "hyreco")
    done, _, _ = launch(site, "--cases-there", env=env, log="there")
    assert done.returncode == 0, done.stderr
    got = dict(ln.split("\t")[:2] for ln in done.stdout.splitlines() if "\t" in ln)
    assert got["hyreco"] == "yes"
    assert "hyreco_a" not in got and "hyreco_b" not in got and "anhir" not in got


def test_beside_the_cohort_the_landmark_set_has_no_case_or_tissue_scoring(site):
    """--datasets "multiplex hyreco": the cohort keeps its per-case jobs and the tissue job;
    hyreco has no per-case job (the tissue job skips it itself: cli._tissue)."""
    done, _, _ = launch(
        site, "--datasets", "multiplex hyreco", env=_env(site, "hyreco"), log="hy-both"
    )
    jobs = jobs_of(done)
    assert "rb-score-case-multiplex" in jobs and "rb-score-case-hyreco" not in jobs
    assert jobs["rb-tissue"].args == ["tissue", "multiplex", "hyreco"]
    assert jobs["rb-score"].args == ["score", "multiplex", "hyreco"]


def test_the_launchers_landmark_sets_are_the_public_landmark_kind():
    from benchmarks.regbench import PUBLIC

    (line,) = [x for x in code_of(SCRIPT).splitlines() if x.startswith("LANDMARK_SETS=")]
    want = tuple(d for d, s in PUBLIC.items() if s["kind"] == "landmarks")
    assert tuple(line.split("=", 1)[1].strip('"').split()) == want == ("hyreco",)


def test_score_case_and_tissue_return_early_on_the_landmark_set(site, monkeypatch, capsys):
    """The CLI mirrors the launcher (final review I3): `score-case` and `tissue` on a
    landmark set say why and compute nothing."""
    from benchmarks.regbench import cli, score, tissue

    def never(*a, **k):
        raise AssertionError("computed on a landmark set")

    monkeypatch.setattr(score, "score_case", never)
    monkeypatch.setattr(tissue, "score_all", never)
    cases, out = str(site[0] / "hyreco_cases"), str(site[0] / "out-cli")
    argv = ["--cases", cases, "--out", out, "--dataset", "hyreco"]
    assert cli.main(["score-case", *argv, "--index", "0"]) == 0
    said = capsys.readouterr().out
    assert "score-case: hyreco is a landmark set" in said and "regbench.landmarks" in said
    assert cli.main(["tissue", *argv]) == 0
    said = capsys.readouterr().out
    assert "tissue: hyreco is a landmark set" in said


# ---- HyReCo's prepare as an sbatch job (final review) -------------------------------------


def _hyreco_data(root):
    for s in ("HE", "PHH3"):
        (root / "HyReCo" / s).mkdir(parents=True, exist_ok=True)
    return root


def test_prepare_hyreco_is_one_prepare_job_with_its_data_root(site, tmp_path):
    data = _hyreco_data(tmp_path / "hyreco_data")
    env = {"REGBENCH_CASES_HYRECO": str(tmp_path / "hy"), "REGBENCH_HYRECO_DATA": str(data)}
    done, _, _ = launch(site, "--prepare", "--datasets", "hyreco", env=env, log="hy-prep")
    jobs = jobs_of(done)
    assert set(jobs) == {"rb-prepare-hyreco"}
    job = jobs["rb-prepare-hyreco"]
    assert job.script == "prepare.sbatch"
    assert job.args == ["hyreco", "--data-root", str(data)]
    assert job.opt("--export") == "ALL"
    # the data root is bound into the job's image: said, with the variable that carries it
    (said,) = [x for x in done.stdout.splitlines() if x.startswith("# hyreco: prepared from")]
    assert str(data) in said and "REGBENCH_BIND" in said
    # a dry run makes no folder
    assert not (tmp_path / "hy").exists()


def test_prepare_hyreco_needs_its_data_root(site, tmp_path):
    env = {"REGBENCH_CASES_HYRECO": str(tmp_path / "hy")}
    done, _, _ = launch(site, "--prepare", "--datasets", "hyreco", env=env, log="hy-nodata")
    assert done.returncode == 2 and "REGBENCH_HYRECO_DATA" in done.stderr
    assert not _sbatch_lines(done.stdout)
    empty = tmp_path / "empty"
    empty.mkdir()
    done, _, _ = launch(site, "--prepare", "--datasets", "hyreco",
                        env={**env, "REGBENCH_HYRECO_DATA": str(empty)}, log="hy-empty")  # fmt: skip
    assert done.returncode == 2 and "HyReCo/HE" in done.stderr
    assert not _sbatch_lines(done.stdout)


def test_prepare_sbatch_writes_hyreco_under_its_own_cases_and_binds_them(tmp_path):
    """The job by hand, as the README writes it: `prepare hyreco` writes into
    REGBENCH_CASES_HYRECO (never REGBENCH_CASES), which is bound with the data root
    (REGBENCH_BIND), and the completion record is written beside its cases."""
    import os
    import subprocess

    from .test_submit_cohort import REPO, SLURM

    data = _hyreco_data(tmp_path / "data")
    hy = tmp_path / "hy_cases"
    fake = tmp_path / "bin"
    fake.mkdir()
    runtime = fake / "apptainer"
    runtime.write_text(
        '#!/bin/bash\necho "ARGS $*" >> "$RUNTIME_LOG"\n'
        'env | grep "^APPTAINERENV_" | sort >> "$RUNTIME_LOG"\n'
    )
    runtime.chmod(0o755)
    log = tmp_path / "runtime.log"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("REGBENCH_", "SLURM_", "APPTAINERENV_", "SINGULARITYENV_"))
    }
    env.update(
        PATH=f"{fake}:{env['PATH']}",
        RUNTIME_LOG=str(log),
        REGBENCH_REPO=str(REPO),
        REGBENCH_CASES=str(tmp_path / "cohort_cases"),
        REGBENCH_OUT=str(tmp_path / "out"),
        REGBENCH_CASES_HYRECO=str(hy),
        REGBENCH_BIND=str(data),
        REGBENCH_PREPARE_IMAGE="STARE_REGBENCH",
        REGBENCH_SIF_STARE_REGBENCH="/images/stare_regbench.sif",
        SLURM_CPUS_PER_TASK="8",
    )
    done = subprocess.run(
        ["bash", str(SLURM / "prepare.sbatch"), "hyreco", "--data-root", str(data)],
        env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    (args,) = [x for x in log.read_text().splitlines() if x.startswith("ARGS ")]
    bind = args.split("--bind ", 1)[1].split()[0].split(",")
    assert str(data) in bind and str(hy) in bind
    tail = args.split("benchmarks.regbench ", 1)[1].split()
    assert tail == ["prepare", "hyreco", "--cases", str(hy), "--workers", "8",
                    "--data-root", str(data)]  # fmt: skip
    assert (hy / "prepared-hyreco.ok").is_file()
    assert not (tmp_path / "cohort_cases" / "prepared-hyreco.ok").exists()
    assert "REGBENCH_END kind=prepare dataset=hyreco status=ok" in done.stdout + done.stderr


def test_prepare_sbatch_names_the_unset_cases_folder_of_a_public_set(tmp_path):
    import os
    import subprocess

    from .test_submit_cohort import REPO, SLURM

    env = {k: v for k, v in os.environ.items() if not k.startswith(("REGBENCH_", "SLURM_"))}
    env.update(REGBENCH_REPO=str(REPO), REGBENCH_CASES=str(tmp_path / "c"),
               REGBENCH_OUT=str(tmp_path / "o"))  # fmt: skip
    done = subprocess.run(
        ["bash", str(SLURM / "prepare.sbatch"), "hyreco", "--data-root", "/x"],
        env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert done.returncode == 2 and "REGBENCH_CASES_HYRECO" in done.stderr


def _head_images(site, tmp_path, *args, **env):
    """launch.sbatch as a head job with its images resolved (not REGBENCH_ALLOW_PATH_PYTHON):
    a stand-in runtime whose `pull` logs the image and makes the file, and whose `exec`
    runs the command on this machine."""
    import subprocess

    from .test_submit_cohort import SLURM, _launch_command

    bin_ = tmp_path / "bin"
    bin_.mkdir(parents=True)
    rt = bin_ / "apptainer"
    rt.write_text(
        "#!/bin/bash\n"
        'case $1 in\n'
        '  pull) echo "$3" >> "$PULL_LOG"; : > "$2"; exit 0 ;;\n'
        "  exec) shift\n"
        "    while [ $# -gt 0 ]; do case $1 in --bind) shift 2 ;; --*) shift ;; *) shift; break ;; esac; done\n"
        '    exec "$@" ;;\n'
        "esac\nexit 1\n"
    )
    rt.chmod(0o755)
    pulls = tmp_path / "pulls.log"
    more = {
        "REGBENCH_CASES": None,
        "REGBENCH_OUT": None,
        "REGBENCH_ROOT": str(tmp_path / "root"),
        "REGBENCH_IMAGES": str(tmp_path / "images"),
        "PULL_LOG": str(pulls),
        "SLURM_SUBMIT_DIR": str(tmp_path),
        **env,
    }
    argv, base, _ = _launch_command(site, args, more, True, f"img-{tmp_path.name}")
    argv[1] = str(SLURM / "launch.sbatch")
    argv.remove("--dry-run")
    argv.insert(2, "--dry-run")
    base["PATH"] = f"{bin_}:{base['PATH']}"
    base = {k: v for k, v in base.items() if not k.startswith("REGBENCH_SIF_")}
    done = subprocess.run(argv, env=base, capture_output=True, text=True, cwd=tmp_path)
    got = pulls.read_text().split() if pulls.exists() else []
    return done, got


def test_a_head_that_prepares_hyreco_alone_never_resolves_stardist(site, tmp_path):
    """launch.sbatch --prepare resolves (pulls when missing) StarDist's image only for a
    prepare that has the cohort: a hyreco-only prepare runs no StarDist, and a node
    without network must not stop it on an image it never uses."""
    data = _hyreco_data(tmp_path / "data")
    env = {"REGBENCH_CASES_HYRECO": str(tmp_path / "hy"), "REGBENCH_HYRECO_DATA": str(data)}
    done, pulled = _head_images(
        site, tmp_path / "a", "--prepare", "--datasets", "hyreco", **env
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert "docker://bolt3x/mirage-stare-regbench:1.0.0" in pulled
    assert not any("stardist" in p for p in pulled)
    assert "rb-prepare-hyreco" in done.stdout
    # the cohort's prepare (the default dataset) still has it resolved
    _, pulled = _head_images(site, tmp_path / "b", "--prepare")
    assert "docker://bolt3x/mirage-stardist:1.0.0" in pulled
