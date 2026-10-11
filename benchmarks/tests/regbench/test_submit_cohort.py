"""submit_cohort.sh: the job graph of the cohort, read off its dry run.

The dry run prints every ``sbatch`` command with a job name in angle brackets where a job
id would be, and submits nothing. A stand-in ``sbatch`` on PATH records every call, so
"submits nothing" is observed, and one test runs the real path through it.

The fixture is two patients: P1 with three rounds, P2 with one (four cases).
"""

from __future__ import annotations

import importlib
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
    return [jobs[f"rb-{run}-{dataset}{suffix}-{gb}g"] for gb in ladder.RUNGS_GB]


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
    assert laddered == set(RUNS) and len(RUNS) == 18
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
        for lower, upper in zip(got, got[1:]):
            assert ("afterany", f"<{lower.name}>") in upper.deps
            assert upper.args[10] == f"<{lower.name}>"  # whose state sacct is asked for
        assert got[0].args[10] == "-"
        assert all("--wait" not in j.words for j in got)


def test_the_rung_task_passes_the_rule_and_the_allocation_on():
    code = code_of(SLURM / "run_rung.sbatch")
    assert 'export REGBENCH_ATTEMPT="$attempt" REGBENCH_MEM_GB="$mem_gb"' in code
    assert re.search(r'if \[ "\$attempt" -gt 1 \]; then\s+args\+=\(--retry-oom\)', code)
    assert "--previous-state" in code and "sacct" in code


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
            assert ("afterany", f"<rb-stare-{dataset}-512g>") in first.deps, run
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
    # a job never waits for one printed after it
    seen = set()
    for line in _sbatch_lines(done.stdout):
        job = Job(line)
        assert {n.strip("<>") for _, n in job.deps} <= seen, job.name
        seen.add(job.name)


def test_tissue_then_score_run_last_in_the_scorers_image(dry):
    jobs, _, _ = dry
    tissue, scorer = jobs["rb-tissue"], jobs["rb-score"]
    assert tissue.args == ["tissue"] and scorer.args == ["score"]
    assert scorer.deps == {("afterany", "<rb-tissue>")}
    waited = {n.strip("<>") for k, n in tissue.deps if k == "afterany"}
    assert {k for k, _ in tissue.deps} == {"afterany"}
    for run in RUNS:
        assert f"rb-gate-{run}" in waited
        for dataset in ("multiplex", "synthetic"):
            assert f"rb-{run}-{dataset}-512g" in waited, (run, dataset)
    code = code_of(SLURM / "score_cohort.sbatch")
    assert code.count('regbench_py "$REGBENCH_SCORER_IMAGE"') == 2
    assert 'args+=(--drop "$v")' in code and '--gates "$gates"' in code
    assert '--pipeline-trace "$REGBENCH_PIPELINE_TRACE"' in code


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
        assert gate.deps == {("afterany", f"<rb-{run}-multiplex-first-512g>")}
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
        assert f"/images/{image}.sif python -m benchmarks.regbench runs" in line
        assert ("--writable-tmpfs" in line) is (image == "VALIS120"), image


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
    env = {"REGBENCH_MANIFEST": "/data/rounds.csv", "REGBENCH_TRUTH_PATIENTS": "P1,P2"}
    done, _, _ = launch(site, "--prepare", env=env, log="prepare")
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
