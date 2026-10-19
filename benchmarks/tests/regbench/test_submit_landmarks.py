"""submit_cohort.sh on the public LANDMARK sets (``anhir``, ``hyreco``): read off the dry
run with ``test_submit_cohort.py``'s stand-in ``sbatch`` and helpers.

Ruling 2026-10-19: each set runs its own 41-run list (STARE's whole tier x stride x
level cross, the other methods at their defaults), STARE first in the default order,
and is scored into its own tables; ``hyreco`` is one dataset and HYRECO_ADDITIONAL is a
download switch the launcher never reads.
"""

from __future__ import annotations

import pytest

from benchmarks.regbench import variants

from .test_submit_cohort import SCRIPT, _case, _sbatch_lines, code_of, jobs_of, launch

SETS = ("anhir", "hyreco")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("landmark_site")
    cases = tmp / "cases"
    _case(cases, "multiplex", "P1_a", "P1")
    for cid in ("004", "012", "370"):
        _case(tmp / "anhir_cases", "anhir", cid, cid)  # each pair is its own patient
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
def test_the_set_runs_its_own_41_runs_stare_first(site, ds):
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
    assert len(ran) == 41 and set(ran) == set(want)
    assert sum(variants.family(r) == "stare" for r in ran) == 36
    assert ran[0] == "stare" and variants.family(ran[0]) == "stare"
    first_other = next(i for i, r in enumerate(ran) if variants.family(r) != "stare")
    assert all(variants.family(r) == "stare" for r in ran[:first_other])
    (line,) = [
        x for x in done.stdout.splitlines() if x.startswith(f"# run order ({ds}): ")
    ]
    assert line.split(": ", 1)[1].split() == ran
    assert f"# {ds}: 41 run(s) of its own list" in done.stdout
    assert f"{out}/tables/{ds}/" in done.stdout


@pytest.mark.parametrize("ds", SETS)
def test_the_set_is_scored_alone_into_its_own_tables(site, ds):
    done, _, _ = launch(site, "--datasets", ds, env=_env(site, ds), log=f"{ds}-score")
    jobs = jobs_of(done)
    assert jobs["rb-score"].args == ["score", ds]
    # after the settle job, through the per-case records and the tissue job (both a
    # no-op on a landmark set: these cases have no nuclear channel)
    deps = {name for _, name in jobs["rb-score"].deps}
    assert {"<rb-tissue>", f"<rb-score-case-{ds}>"} <= deps
    assert ("afterany", "<rb-settle>") in jobs[f"rb-score-case-{ds}"].deps
    assert jobs[f"rb-score-case-{ds}"].args == ["case", ds]
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


@pytest.mark.parametrize("old", ["hyreco_a", "hyreco_b"])
def test_the_two_old_hyreco_sets_are_no_dataset(site, old):
    done, _, _ = launch(site, "--datasets", old, log=f"old-{old}")
    assert done.returncode == 2 and "no such dataset" in done.stderr
    assert not _sbatch_lines(done.stdout)


def test_cases_there_says_yes_for_prepared_landmark_sets(site):
    env = {**_env(site, "anhir"), **_env(site, "hyreco")}
    done, _, _ = launch(site, "--cases-there", env=env, log="there")
    assert done.returncode == 0, done.stderr
    got = dict(ln.split("\t")[:2] for ln in done.stdout.splitlines() if "\t" in ln)
    assert got["anhir"] == "yes" and got["hyreco"] == "yes"
    assert "hyreco_a" not in got and "hyreco_b" not in got
