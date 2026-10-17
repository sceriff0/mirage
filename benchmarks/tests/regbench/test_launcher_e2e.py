"""The launcher end to end (fleet wave 2, C): `submit_cohort.sh` for real (not
--dry-run) on one toy case, against a stand-in SLURM (`slurm_sim.py`), with the jobs'
bodies run on the host python (REGBENCH_ALLOW_PATH_PYTHON=1) and `fake_palom.py` as the
method. Nothing downloads and no container starts. The selfcheck jobs are stubbed (they
are reported, never a gate, and the real one takes minutes); so are the per-case
scoring jobs except in the first scenario, which runs settle, tissue and the scorer.

Scenarios: (1) all ok; (2) out of memory at 16 GB climbs to 64 GB; (3) out of memory to
the top rung is the method's `oom`, and with a partition whose largest node cannot
grant 512 GB the units are the launcher's `not_submitted`; (4) a watcher killed
mid-chain is taken over by its successor link; (5) a second submission into the same
folder runs the method for nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.regbench import status
from benchmarks.tests.regbench import slurm_sim

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "benchmarks" / "regbench" / "slurm" / "submit_cohort.sh"
CASE = "crop"
SCORING = "rb-score-case-* rb-tissue rb-score"

SITECUSTOMIZE = """
import os
if os.environ.get("REGBENCH_E2E_FAKE") == "palom":
    from benchmarks.tests.regbench import fake_palom as fp
    fp.install()
    below = float(os.environ.get("REGBENCH_E2E_OOM_BELOW_GB") or 0)
    orig = fp.Aligner.coarse_register_affine
    def coarse(self, **kw):
        with open(os.environ["REGBENCH_E2E_CALLS"], "a") as fh:
            fh.write(os.environ.get("REGBENCH_MEM_GB", "?") + "\\n")
        if below and float(os.environ.get("REGBENCH_MEM_GB") or 0) < below:
            raise MemoryError("e2e: out of memory below %g GB" % below)
        return orig(self, **kw)
    fp.Aligner.coarse_register_affine = coarse
"""


@pytest.fixture(scope="module")
def one_case(cases_root, tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e-cases")
    shutil.copytree(Path(cases_root) / "multiplex" / CASE, root / "multiplex" / CASE)
    return root


def cohort(tmp_path, cases, oom_below=0, sinfo=None, kill=None, submission="e2e-1",
           out=None, stub=SCORING):  # fmt: skip
    """Launch, then let the stand-in scheduler run everything. Returns
    ``(launcher, out, sim env)``."""
    site = tmp_path / "site"
    site.mkdir(parents=True, exist_ok=True)
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
    bin_ = slurm_sim.install(tmp_path / "bin")
    out = out or tmp_path / "out"
    keep = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("REGBENCH_", "SLURM_", "SIM_"))
    }
    env = {
        **keep,
        "PATH": f"{bin_}:{Path(sys.executable).parent}:/usr/bin:/bin",
        "PYTHONPATH": os.pathsep.join(
            [str(site), *filter(None, [os.environ.get("PYTHONPATH")])]
        ),
        "SIM_DIR": str(tmp_path / "sim"),
        "SIM_STUB": f"rb-sc-* {stub}",
        "REGBENCH_CASES": str(cases),
        "REGBENCH_OUT": str(out),
        "REGBENCH_ALLOW_PATH_PYTHON": "1",
        "REGBENCH_PREFLIGHT": "0",
        "REGBENCH_SITE_CONFIG": "none",
        "REGBENCH_RUN_ORDER": "",
        "REGBENCH_SUBMISSION": submission,
        "REGBENCH_E2E_FAKE": "palom",
        "REGBENCH_E2E_OOM_BELOW_GB": str(oom_below),
        "REGBENCH_E2E_CALLS": str(tmp_path / "calls.txt"),
    }
    if sinfo:
        env["SIM_SINFO"] = sinfo
    if kill:
        env["SIM_KILL_ON_SQUEUE"] = kill
    done = subprocess.run(
        ["bash", str(SCRIPT), "--runs", "palom"], env=env, capture_output=True,
        text=True, cwd=REPO,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    old = os.environ.get("SIM_DIR")
    os.environ["SIM_DIR"] = env["SIM_DIR"]
    try:
        slurm_sim.drain()
        jobs = slurm_sim.jobs()
    finally:
        if old is None:
            os.environ.pop("SIM_DIR", None)
        else:
            os.environ["SIM_DIR"] = old
    return done, out, jobs


def record(out, variant="palom"):
    return json.loads((out / variant / "multiplex" / CASE / "run.json").read_text())


def by_name(jobs):
    return {j["name"]: j for j in jobs.values()}


def log_of(jobs, name, task="0"):
    job = by_name(jobs)[name]
    t = job["tasks"].get(task) or job["tasks"]["-"]
    return Path(t["log"]).read_text()


def test_1_all_ok_settles_scores_and_says_so(tmp_path, one_case, monkeypatch):
    done, out, jobs = cohort(tmp_path, one_case, stub="")
    named = by_name(jobs)
    assert all(j["state"] == "COMPLETED" for j in jobs.values()), {
        n: j["state"] for n, j in named.items()
    }
    for v in ("palom", "palom_affine"):
        assert record(out, v)["ok"] is True and record(out, v)["attempt"] == 1
    # nothing climbed: the watcher submitted no step
    assert sorted(
        n for n in named if n.startswith("rb-palom-multiplex-") and "retry" not in n
    ) == ["rb-palom-multiplex-16g"]
    assert (out / "tables" / "summary.md").is_file()
    assert "REGBENCH_END kind=settle status=ok" in log_of(jobs, "rb-settle")
    # the operator's lines: one start and one end per job, the end before REGBENCH_END
    run_log = log_of(jobs, "rb-palom-multiplex-16g").splitlines()
    start = next(ln for ln in run_log if ln.startswith("[regbench] start"))
    assert " kind=run " in start and " rung_gb=16 attempt=1 " in start
    assert f" out={out} " in start and " commit=" in start
    end = [ln for ln in run_log if ln.startswith("[regbench] end")][-1]
    assert " outcome=ok " in end and f"record={out}/" in end
    assert run_log[-1].startswith("REGBENCH_END ") and run_log[-2] == end
    monkeypatch.setenv("PATH", "/nonexistent")  # no squeue: said, not guessed
    said = status.report(out, one_case, runs=["palom"])
    assert "# 2 result(s) expected: 2 ok (0 reused), 0 failed, 0 missing" in said
    assert "# next: nothing; every expected result is ok" in said
    (tmp_path / "example.txt").write_text(start + "\n" + end + "\n" + said)


def test_2_out_of_memory_at_16_gb_climbs_to_64_and_succeeds(tmp_path, one_case):
    _, out, jobs = cohort(tmp_path, one_case, oom_below=64)
    named = by_name(jobs)
    rec = record(out)
    assert rec["ok"] is True and rec["attempt"] == 2 and rec["mem_requested_gb"] == 64
    # the watcher submitted the next rung, and no same-rung retry (wants=climb)
    assert named["rb-palom-multiplex-64g"]["by"] == "rb-palom-multiplex-retry"
    assert "rb-palom-multiplex-16g-again" not in named
    assert "wants=climb" in log_of(jobs, "rb-palom-multiplex-16g")


def test_3_out_of_memory_to_the_top_rung_is_the_methods(tmp_path, one_case):
    _, out, jobs = cohort(tmp_path, one_case, oom_below=100000)
    named = by_name(jobs)
    for gb in (64, 256, 512, 700):
        assert f"rb-palom-multiplex-{gb}g" in named, gb
    rec = record(out)
    assert rec["ok"] is False and rec["cause"] == "oom" and rec["attempt"] == 5
    assert "ladder_cut_gb" not in rec


def test_3_a_rung_no_node_can_grant_is_the_launchers(tmp_path, one_case):
    """The partition's largest node has 300000 MB: 512 and 700 GB are never planned;
    the unit that ran out of memory at 256 GB is written down as not_submitted."""
    done, out, jobs = cohort(tmp_path, one_case, oom_below=100000, sinfo="cpu*|300000|")
    assert "can NEVER be granted there" in done.stdout + done.stderr
    named = by_name(jobs)
    assert "rb-palom-multiplex-256g" in named and "rb-palom-multiplex-512g" not in named
    rec = record(out)
    assert rec["cause"] == "not_submitted" and rec["ladder_cut_gb"] == 512
    assert rec["cause_from"].startswith("oom at rung 3 (256 GB)")
    assert "NOT SUBMITTED, the partition cannot grant this rung" in log_of(
        jobs, "rb-palom-multiplex-retry"
    )


def test_4_a_watcher_killed_mid_chain_is_taken_over_by_its_successor(
    tmp_path, one_case
):
    _, out, jobs = cohort(
        tmp_path, one_case, oom_below=64, kill="rb-palom-multiplex-retry"
    )
    named = by_name(jobs)
    # link 1 submitted the 64 GB rung, then was killed (its time limit, its node)
    assert named["rb-palom-multiplex-retry"]["state"] == "TIMEOUT"
    assert named["rb-palom-multiplex-64g"]["by"] == "rb-palom-multiplex-retry"
    link2 = log_of(jobs, "rb-palom-multiplex-retry-2")
    assert "TAKING OVER the chain rb-palom-multiplex-16g" in link2
    assert named["rb-palom-multiplex-retry-2"]["state"] == "COMPLETED"
    plan = out / "logs" / "retry" / "e2e-1" / "rb-palom-multiplex"
    assert (plan / "done").read_text().split() == ["ok", "-", "0"]
    assert record(out)["ok"] is True and record(out)["attempt"] == 2
    # the 64 GB rung ran once: the successor did not submit it again
    assert [j["name"] for j in jobs.values()].count("rb-palom-multiplex-64g") == 1
    assert named["rb-settle"]["state"] == "COMPLETED"


def test_5_a_second_submission_into_the_same_folder_reuses_everything(tmp_path, one_case,
                                                                      monkeypatch):  # fmt: skip
    _, out, _ = cohort(
        tmp_path / "a", one_case, submission="e2e-1", out=tmp_path / "out"
    )
    calls = (tmp_path / "a" / "calls.txt").read_text()
    assert calls  # the method ran
    _, out, jobs = cohort(
        tmp_path / "b", one_case, submission="e2e-2", out=tmp_path / "out"
    )
    # the second submission's run task reused both results: the method never ran
    assert not (tmp_path / "b" / "calls.txt").exists()
    assert "reused" in log_of(jobs, "rb-palom-multiplex-16g")
    for v in ("palom", "palom_affine"):
        rec = record(out, v)
        assert rec["ok"] and [e["submission"] for e in rec["reused_by"]] == ["e2e-2"]
    monkeypatch.setenv("PATH", "/nonexistent")
    said = status.report(out, one_case, submission="e2e-2", runs=["palom"])
    assert "# 2 result(s) expected: 2 ok (2 reused), 0 failed, 0 missing" in said
