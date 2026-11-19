#!/usr/bin/env python3
"""Guard: the nf-test (stub) matrix's shard list and its `--shard` divisor agree.

`_test-suite.yml`'s `nf-test-stub` job runs as a matrix using nf-test's
`--shard i/n --shard-strategy round-robin`. `strategy.matrix.shard: [1, 2, 3, 4]`
and the run step's `--shard ${{ matrix.shard }}/4` both encode the shard count,
and nothing else forces them to agree. With `shard: [1, 2, 3]` and `/4`, a
quarter of the suite would run on no leg while every leg stays green.

The divisor is read from the job's `run:` bodies via `ci_actions.job_run_scripts`
(comments stripped), so a `--shard` left only in a comment does not count.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
ci_actions = importlib.import_module("ci_actions")

SUITE = ci_actions.WORKFLOWS_DIR / "_test-suite.yml"
JOB_ID = "nf-test-stub"

# `--shard 2/4`, `--shard ${{ matrix.shard }}/4`: the divisor follows the nearest
# `/` after `--shard `. `.*?` rather than `\S*?` because `${{ matrix.shard }}`
# contains spaces; the trailing whitespace anchor keeps `--shard-strategy` out.
SHARD_RE = re.compile(r"--shard\s+.*?/(\d+)")


def _nf_test_stub_job() -> dict:
    jobs = ci_actions.load_jobs(SUITE)
    assert JOB_ID in jobs, (
        f"{SUITE.name} no longer has a `{JOB_ID}` job -- this guard, and "
        "`suite-gate`'s `needs:`, both key off that id staying stable across a "
        "matrix rewrite."
    )
    return jobs[JOB_ID]


def test_shard_matrix_list_is_contiguous_from_one():
    """`strategy.matrix.shard` must be exactly `[1, 2, ..., n]` for its length n.

    A gap (`[1, 2, 4]`) or a duplicate (`[1, 1, 2, 3]`) either strands a shard
    number nf-test never sees or runs one leg's slice twice, while every leg
    still reports green.
    """
    job = _nf_test_stub_job()
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    shard_list = matrix.get("shard")
    assert isinstance(shard_list, list) and shard_list, (
        f"{JOB_ID} has no non-empty `strategy.matrix.shard` list"
    )
    n = len(shard_list)
    assert shard_list == list(range(1, n + 1)), (
        f"{JOB_ID}'s matrix.shard is {shard_list!r}, expected the contiguous "
        f"list [1..{n}]. A gap or duplicate here silently drops or double-runs "
        "a quarter of the suite with no test failing."
    )


def test_shard_divisor_matches_the_matrix_length():
    """Every `--shard i/n` invocation in the job must use n == len(matrix.shard).

    Read through `ci_actions.job_run_scripts`, which strips shell comments.
    """
    job = _nf_test_stub_job()
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    shard_list = matrix.get("shard") or []
    n = len(shard_list)
    assert n, f"{JOB_ID} has no `strategy.matrix.shard` list to compare against"

    script = ci_actions.job_run_scripts(job)
    divisors = {int(d) for d in SHARD_RE.findall(script)}
    assert divisors, (
        f"no `--shard i/n` invocation found in {JOB_ID}'s `run:` bodies "
        "(comments stripped) -- the sharded nf-test command may have been "
        "removed or commented out."
    )
    assert divisors == {n}, (
        f"{JOB_ID}'s `--shard` divisor(s) {sorted(divisors)} do not match its "
        f"matrix length ({n} legs declared in `strategy.matrix.shard`). "
        f"matrix.shard and the `--shard i/n` divisor must be typed as the same "
        "number: a mismatch either strands tests (divisor > matrix length) or "
        "runs some of them on every leg while others run on none (divisor < "
        "matrix length)."
    )


def test_shard_strategy_is_round_robin():
    """The job's `run:` body must actually pass `--shard-strategy round-robin`.

    Without it, `--shard i/n` alone falls back to nf-test's default splitting
    strategy, which shards by file rather than by test; with unevenly sized
    nf-test files that unbalances the legs. Read through the comment-stripped
    `ci_actions.job_run_scripts`.
    """
    job = _nf_test_stub_job()
    script = ci_actions.job_run_scripts(job)
    assert "--shard-strategy round-robin" in script, (
        f"{JOB_ID}'s `run:` bodies (comments stripped) do not contain "
        "`--shard-strategy round-robin` -- without it, `--shard i/n` falls "
        "back to nf-test's default per-file splitting, which can silently "
        "starve or overload individual shard legs while every leg still "
        "reports green."
    )
