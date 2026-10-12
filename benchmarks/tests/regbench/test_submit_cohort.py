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
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.regbench import ladder, variants
from benchmarks.regbench.cases import Case, write_json

REPO = Path(__file__).resolve().parents[3]
SLURM = REPO / "benchmarks" / "regbench" / "slurm"
SCRIPT = SLURM / "submit_cohort.sh"
GPU = "--gres=gpu:nvidia_h200:1"
RUNS = {variants.run_name(r): r for r in variants.active_runs()}
GATING = sorted({variants.gated_by(n) for n in RUNS})
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


def _case(root, dataset, case_id, group):
    case = Case(
        dataset=dataset, case_id=case_id, group=group, ref_nuclear="", mov_nuclear="",
        ref_image="", mov_image="", modality="fluorescence", diagonal=1.0,
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
        'echo "$n" > "$SBATCH_LOG.n"\n'
        'printf "%s\\t%s\\n" "$n" "$*" >> "$SBATCH_LOG"\n'
        'echo "$n;cluster"\n'
    )
    sbatch.chmod(0o755)
    return tmp, cases, fake


def launch(site, *args, env=None, dry=True, log="calls"):
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
        REGBENCH_SITE_CONFIG="none",  # never the real site overlay of a checkout
        SBATCH_LOG=str(tmp / f"{log}.tsv"),
    )
    base.update(env or {})
    for key in [k for k, v in base.items() if v is None]:
        del base[key]
    argv = ["bash", str(SCRIPT), *(["--dry-run"] if dry else []), *args]
    done = subprocess.run(argv, env=base, capture_output=True, text=True, cwd=REPO)
    return done, Path(base["SBATCH_LOG"]), out


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


def rungs(jobs, run, dataset="multiplex", suffix=""):
    """The attempt of each rung (the `-again` arrays are `agains`)."""
    return [jobs[f"rb-{run}-{dataset}{suffix}-{gb}g"] for gb in ladder.RUNGS_GB]


def agains(jobs, run, dataset="multiplex", suffix=""):
    return [jobs[f"rb-{run}-{dataset}{suffix}-{gb}g-again"] for gb in ladder.RUNGS_GB]


def chain(jobs, run, dataset="multiplex", suffix=""):
    """The ladder in the order it runs: attempt, again, attempt, again, ..."""
    both = zip(rungs(jobs, run, dataset, suffix), agains(jobs, run, dataset, suffix))
    return [j for pair in both for j in pair]


# ── nothing is submitted by a dry run ─────────────────────────────────────────
def test_a_dry_run_submits_nothing(dry):
    jobs, done, log = dry
    assert len(jobs) > 100 and not log.exists()
    assert f"# {len(jobs)} job(s) would be submitted; nothing was" in done.stdout
    # nothing waits on the login node, and no job submits a job
    assert all("--wait" not in j.words for j in jobs.values())
    for path in [SCRIPT, *SLURM.glob("*.sbatch")]:
        assert "--wait" not in code_of(path), path.name
    for path in SLURM.glob("*.sbatch"):
        assert "sbatch " not in code_of(path), path.name


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
    assert laddered == set(RUNS) and len(RUNS) == 19
    gated = {j.args[1] for j in jobs.values() if j.args[:1] == ["selfcheck"]}
    assert gated == set(RUNS)


def test_a_paused_run_is_not_submitted(site, monkeypatch, tmp_path):
    """The script submits what `runs` prints, and `runs` leaves a paused run out."""
    table = variants.runs_table()
    kept = [row for row in table if row[0] != "palom_b1024"]
    stub = tmp_path / "benchmarks" / "regbench"
    stub.mkdir(parents=True)
    (tmp_path / "benchmarks" / "__init__.py").write_text("")
    (stub / "__init__.py").write_text("")
    lines = "\\n".join("\\t".join(r) for r in kept)
    (stub / "__main__.py").write_text(
        "import sys\n"
        "a = sys.argv[1:]\n"
        "if a[:1] == ['runs']:\n"
        f"    print('{variants.SCORER_IMAGE}' if '--scorer-image' in a else '{lines}')\n"
        "elif a[:1] == ['list']:\n"
        "    print('4 2 3' if 'multiplex' in a else '0 0 0')\n"
    )
    done, _, _ = launch(site, env={"REGBENCH_REPO": str(tmp_path)}, log="paused")
    ran = {j.args[0] for j in jobs_of(done).values() if j.script == "run_rung.sbatch"}
    assert "palom_b1024" not in ran and ran == set(RUNS) - {"palom_b1024"}


# ── the ladder ────────────────────────────────────────────────────────────────
def test_every_run_has_four_chained_arrays_at_four_cpus(dry):
    jobs, _, _ = dry
    for run in RUNS:
        got = rungs(jobs, run)
        assert [j.opt("--mem") for j in got] == ["16G", "64G", "256G", "512G"], run
        assert [j.opt("--cpus-per-task") for j in got] == ["4"] * 4, run
        # attempt and memory reach the task: REGBENCH_ATTEMPT, REGBENCH_MEM_GB
        assert [j.args[7:9] for j in got] == [
            ["1", "16"], ["2", "64"], ["3", "256"], ["4", "512"]
        ], run  # fmt: skip
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
    assert 'args+=(--accounting "$acct")' in code
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
    assert ("afterany", "<rb-stare-synthetic-512g-again>") in first.deps
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
    assert rungs(jobs, "stare")[0].opt("--time") == "3-00:00:00"


def test_patient_runs_get_one_rounds_time_for_every_round(dry):
    jobs, _, _ = dry
    for run, spec in RUNS.items():
        per_patient = variants.scope(spec.method) == "group"
        job = rungs(jobs, run)[0]
        # P1 has three rounds: 3 x 24 h for a run that registers a patient at once
        assert job.opt("--time") == ("3-00:00:00" if per_patient else "1-00:00:00"), run


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
    assert {j.args[0] for j in checks} == set(GATING) and len(GATING) == 17
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
    assert (
        jobs["rb-sc-palom-64g"].args[6]
        == jobs["rb-sc-palom_b1024-64g"].args[6]
        == "seams"
    )


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


# ── order ─────────────────────────────────────────────────────────────────────
def test_both_anchored_rows_wait_for_the_run_they_read(dry):
    jobs, _, _ = dry
    for run in ("affine_ashlar", "affine_ashlar_clean"):
        assert RUNS[run].after == "stare"
        for dataset in ("multiplex", "synthetic"):
            first = rungs(jobs, run, dataset)[0]
            assert ("afterany", f"<rb-stare-{dataset}-512g-again>") in first.deps, run
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
            assert f"rb-{run}-{dataset}-512g-again" in waited, (run, dataset)
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
    assert code.count("settle --out") == 1 and code.count("sacct -n -X -P") == 1
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
    assert len(jobs) == 19 + 51 + 19 * 2 * 8 + 2 * (8 + 1) + 1 + len(sizes) + 2 == 397


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
        site, "--score-only", env={"REGBENCH_SBATCH_GPU": None}, log="so"
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
    "affine_ashlar_clean palom_b1024",  # run names; one anchored row alone
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
            assert ("afterany", f"<rb-stare-{dataset}-512g-again>") in first.deps
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
        "ashlar", "ashlar_clean", "palom", "palom_b1024",
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
    assert len(order) == len(deps) == 397
    # a cycle would be found: the check is not vacuous
    a, b = list(deps)[:2]
    with pytest.raises(AssertionError, match="a cycle"):
        _sorted_topologically({**deps, a: {b}, b: {a}})
    # an anchored run and its anchor on ONE lane: the anchor's ladder comes first
    by_name = {n: i for i, n in names.items()}
    anchored = deps[by_name["rb-affine_ashlar-multiplex-16g"]]
    assert by_name["rb-stare-multiplex-512g-again"] in anchored


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
    done, log, _ = launch(site, env={"REGBENCH_SBATCH_GPU": None}, log="nogpu")
    assert done.returncode == 2 and not _sbatch_lines(done.stdout) and not log.exists()
    assert "REGBENCH_SBATCH_GPU is not set" in done.stderr
    for run, spec in RUNS.items():
        assert (f" {run}" in done.stderr.split("not set:")[1].split(".")[0]) is spec.gpu
    assert "Nothing was submitted" in done.stderr
    # the same refusal on the real path, and the runs that need none can still go
    real, log, _ = launch(
        site, env={"REGBENCH_SBATCH_GPU": None}, dry=False, log="nogpu2"
    )
    assert real.returncode == 2 and not log.exists()
    cpu = " ".join(n for n, r in RUNS.items() if not r.gpu)
    ok, _, _ = launch(site, "--runs", cpu, env={"REGBENCH_SBATCH_GPU": None}, log="cpu")
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
        env={**env, "REGBENCH_SBATCH_GPU": None},
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
