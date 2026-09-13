"""The benchmark head jobs must fit the Nextflow heads they launch.

Raising ARMS_CONCURRENCY / SWEEP_CONCURRENCY is the one lever that raises
cluster-wide throughput (the per-process clamps are per head), and the head
job's memory is the ONLY thing it can OOM: the process jobs are sized per task in
conf/modules.config. So two things are pinned here, for BOTH submitters:

1. statically, the shipped defaults fit: heads x (-Xmx + overhead) <= #SBATCH --mem,
   and the CPU allocation is not absurd for that many JVMs;
2. behaviourally, benchmarks/head_sizing.sh refuses a launch that does not fit,
   passes one that does, and keeps the total in-flight job count at its target
   when the head count changes.

Comment-blind where it matters (the SBATCH header IS a comment, so it is read on
purpose from the raw lines; everything else through ci_actions.strip_line_comment).
"""

from __future__ import annotations

import importlib
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks"
# The one quote-aware comment stripper lives in tests/ci_actions.py. Resolved the way
# test_ci_installs_what_benchmarks_import.py resolves it, so that guard's import scan
# (which reads `import` statements) does not mistake a first-party module for a
# distribution benchmarks.yml must install.
sys.path.insert(0, str(REPO / "tests"))
strip_line_comment = importlib.import_module("ci_actions").strip_line_comment

SUBMITTERS = {
    "submit_arms.sh": "ARMS_CONCURRENCY",
    "submit_sweep.sh": "SWEEP_CONCURRENCY",
}
OVERHEAD_GB = 0.75


def _sbatch(text: str, flag: str) -> str:
    m = re.search(rf"^#SBATCH\s+{re.escape(flag)}=(\S+)", text, re.M)
    assert m, f"no #SBATCH {flag} line"
    return m.group(1)


def _mem_gb(value: str) -> float:
    m = re.fullmatch(r"(\d+)([GgMm])", value)
    assert m, value
    n, unit = int(m.group(1)), m.group(2).lower()
    return n if unit == "g" else n / 1024


def _code(text: str) -> str:
    return "\n".join(strip_line_comment(ln) for ln in text.splitlines())


def _default(code: str, var: str) -> str:
    m = re.search(rf"\$\{{{re.escape(var)}:-([^}}]*)\}}", code)
    assert m, f"no ${{{var}:-default}} in the script"
    return m.group(1)


def _xmx_gb(nxf_opts: str) -> float:
    m = re.search(r"-Xmx(\d+)([gGmM])", nxf_opts)
    assert m, f"no -Xmx in NXF_OPTS default {nxf_opts!r}"
    n, unit = int(m.group(1)), m.group(2).lower()
    return n if unit == "g" else n / 1024


@pytest.mark.parametrize("script,var", sorted(SUBMITTERS.items()))
def test_shipped_defaults_fit_the_head_job(script, var):
    text = (BENCH / script).read_text()
    code = _code(text)
    heads = int(_default(code, var))
    heap = _xmx_gb(_default(code, "NXF_OPTS"))
    mem = _mem_gb(_sbatch(text, "--mem"))
    cpus = int(_sbatch(text, "--cpus-per-task"))
    need = heads * (heap + OVERHEAD_GB)
    assert need <= mem, (
        f"{script}: {heads} heads x ({heap} + {OVERHEAD_GB}) GB = {need} GB "
        f"exceeds #SBATCH --mem={mem} GB"
    )
    assert heads >= 16, f"{script}: {heads} heads is not 'greatly increased'"
    assert cpus >= math.ceil(heads / 8), (
        f"{script}: {cpus} cpus for {heads} JVMs -- give at least one core per 8 heads"
    )


@pytest.mark.parametrize("script", sorted(SUBMITTERS))
def test_submitter_sources_the_guard_and_refuses_before_launching(script):
    code = _code((BENCH / script).read_text())
    assert re.search(r"source\s+\"\$SRC_DIR/benchmarks/head_sizing\.sh\"", code), (
        f"{script} does not source head_sizing.sh (comment-stripped view)"
    )
    check = re.search(
        r'check_head_memory\s+"\$CONCURRENCY"\s+"\$NXF_OPTS"\s*\|\|\s*exit 1', code
    )
    assert check, f"{script}: check_head_memory is not called with `|| exit 1`"
    launch = re.search(r"benchmarks/run_(arms|sweep)\.sh", code)
    assert launch and check.start() < launch.start(), (
        f"{script}: the memory check must run BEFORE the launcher is invoked"
    )
    assert re.search(r"derive_queue_size\s+\"\$CONCURRENCY\"", code), (
        f"{script}: QUEUE_SIZE is not derived from the head count"
    )


def _run(fn_call: str, env: dict | None = None):
    return subprocess.run(
        ["bash", "-c", f'source "{BENCH / "head_sizing.sh"}"; {fn_call}'],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **(env or {})},
    )


def test_guard_refuses_heads_that_outgrow_the_allocation_and_names_the_fix():
    r = _run(
        'check_head_memory 32 "-Xms512m -Xmx3g"', {"SLURM_MEM_PER_NODE": str(64 * 1024)}
    )
    assert r.returncode == 1
    assert "120.00 GB exceeds the head job's 64.00 GB" in r.stderr
    assert "--mem' to at least 121G" in r.stderr
    assert "sized per task in conf/modules.config" in r.stderr


def test_guard_passes_heads_that_fit_and_says_so():
    r = _run(
        'check_head_memory 32 "-Xms256m -Xmx2g"',
        {"SLURM_MEM_PER_NODE": str(128 * 1024)},
    )
    assert r.returncode == 0, r.stderr
    assert "88.00 GB of 128.00 GB -- fits" in r.stdout


def test_guard_reads_a_per_cpu_allocation_too():
    env = {"SLURM_MEM_PER_CPU": "4096", "SLURM_CPUS_ON_NODE": "8"}  # 32 GB
    assert _run('check_head_memory 8 "-Xmx3g"', env).returncode == 0  # 30 GB
    assert _run('check_head_memory 9 "-Xmx3g"', env).returncode == 1  # 33.75 GB


def test_guard_refuses_a_missing_xmx_rather_than_trusting_the_jvm_default():
    r = _run('check_head_memory 4 "-Xms1g"', {"SLURM_MEM_PER_NODE": "999999"})
    assert r.returncode == 1 and "sets no -Xmx" in r.stderr


def test_guard_is_advisory_outside_slurm():
    r = _run('check_head_memory 64 "-Xmx3g"')
    assert r.returncode == 0 and "not checked" in r.stdout


def test_queue_size_keeps_the_total_in_flight_at_the_target_floored_at_max_forks():
    assert _run("derive_queue_size 32 20 800").stdout.strip() == "25"
    assert _run("derive_queue_size 40 20 1600").stdout.strip() == "40"
    assert _run("derive_queue_size 80 20 800").stdout.strip() == "20"  # floor, not 10
    assert _run("derive_queue_size 1 20 800").stdout.strip() == "800"


def test_heap_parser_accepts_megabytes_and_last_xmx_wins():
    assert _run('head_heap_gb "-Xmx2048m"').stdout.strip() == "2.00"
    assert _run('head_heap_gb "-Xmx1g -Xmx4G"').stdout.strip() == "4"
    assert _run('head_heap_gb "-Xms1g"').stdout.strip() == "0"
