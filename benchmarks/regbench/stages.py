"""A registration that runs as several scheduler jobs, the way the pipeline runs STARE.

The pipeline registers one moving round in four processes
(``subworkflows/local/adapters/tiled_adapter.nf``): ``TILED_COARSE``, then one
``TILED_REG_TILE`` task per tile, then ``TILED_SOLVE``, then ``TILED_STITCH``. The
benchmark runs the same four stage commands as separate jobs, each with the allocation
the pipeline gives that process, and this module holds everything about that which is
not STARE's own command line: the allocations, the retry rule, the record each job
leaves, the layout of the tile arrays, and the cost of a round summed over its jobs.

**Allocation** (``ALLOCATION``). The CPUs, memory and time ``conf/modules.config`` gives
each process at the ``high`` tier and stride 64: the ``withLabel:`` block of the label
the module declares, and the ``memory`` closure of the process's ``withName:`` block.
Memory and time are multiplied by the attempt, as there. ``prep`` is not a pipeline
process: it is the benchmark's own preprocessing of the ``stare_prep`` row and has the
benchmark's allocation (``ladder.CPUS``, the first rung). ``tests/regbench/
test_stages.py`` reads both files and fails when this table and they differ.

**Retries** (``RETRY_EXIT``, ``MAX_RETRIES``). None of the four processes sets an
``errorStrategy``; they inherit the process-level one of ``conf/base.config``: an exit
status of 104, 134, 135, 137, 139, 140 or 143 is retried, anything else ends the run,
and ``maxRetries`` is 3. A job the scheduler killed before it could record its exit
status (its record still says ``killed``) and a job that left no record are retried too:
the scheduler ends a job with a signal, and every such status is in the list.

**One record per job** (``<stage dir>/jobs/<stage>[_<row>].json``, ``run_command``).
Written as ``ok: false, cause: killed`` before the command starts, so a killed job
leaves a failed record, and rewritten when the command ends with its exit status, wall
time, CPU time and peak memory (``os.wait4``: the command's own process and the
children it waited for, nothing of the benchmark's). ``history`` keeps every earlier
attempt of the same job.

**The cost of a round** (``summarise``), from the records of COARSE, every tile and
SOLVE, every attempt of each (STITCH is the writer and is timed apart):

``elapsed_with_queue_s``        from the start of the first COARSE attempt to the end
                                of SOLVE, on the wall clock: queue waits between jobs
                                and the array's concurrency cap are in it
``wall_unlimited_parallel_s``   COARSE + the LONGEST tile + SOLVE, each the sum of its
                                attempts: the wall time with every tile running at
                                once and no queue. This is the round's ``register_s``
``cpu_h_allocated``             sum over every attempt of allocated CPUs x its duration
``cpu_h_measured``              sum over every attempt of its measured CPU time
``largest_job_peak_mb``, ``largest_job``  the peak memory of the largest single job, and
                                which stage it was (the round's ``peak_rss_mb``)
``n_jobs``, ``n_tiles``, ``n_retries``  attempts run, tiles, attempts beyond the first
``peak_cpus_at_once``           the most allocated CPUs in jobs of this round that
                                overlapped in time
``unmeasured_attempts``         attempts that were killed before they could record a
                                duration: counted, in no sum
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from . import ladder, resources

PREP, COARSE, TILE, SOLVE, STITCH = "prep", "coarse", "tile", "solve", "stitch"
# the jobs up to the transform, in order; STITCH (the writer) is the driver's own job
TO_TRANSFORM = (COARSE, TILE, SOLVE)

# conf/modules.config at reg_tiled_mode=high, reg_tiled_stride=64 (attempt 1; memory and
# time x attempt). `label` is the one the module declares (modules/local/tiled_*.nf).
ALLOCATION = {
    COARSE: {
        "process": "TILED_COARSE",
        "label": "process_low",
        "cpus": 2,
        "mem_gb": 2,
        "hours": 2,
    },
    TILE: {
        "process": "TILED_REG_TILE",
        "label": "process_low",
        "cpus": 2,
        "mem_gb": 4,
        "hours": 2,
    },
    SOLVE: {
        "process": "TILED_SOLVE",
        "label": "process_single",
        "cpus": 1,
        "mem_gb": 1,
        "hours": 8,
    },
    STITCH: {
        "process": "TILED_STITCH",
        "label": "process_medium",
        "cpus": 4,
        "mem_gb": 4,
        "hours": 4,
    },
    # the benchmark's own step, not a pipeline process
    PREP: {
        "process": "",
        "label": "",
        "cpus": ladder.CPUS,
        "mem_gb": ladder.RUNGS_GB[0],
        "hours": 24,
    },
}
# conf/base.config: errorStrategy and maxRetries of every process without its own
RETRY_EXIT = (104, 134, 135, 137, 139, 140, 143)
MAX_RETRIES = 3
ATTEMPTS = MAX_RETRIES + 1
KILLED = "killed"
ARRAY_MAX = 1000  # tasks per tile array unless REGBENCH_STAGE_ARRAY_MAX says otherwise


class StageFailed(RuntimeError):
    """A stage job of a round failed after its retries, or left nothing usable. The
    driver records ``regbench_cause`` as the round's cause."""

    def __init__(self, message, cause="exception"):
        super().__init__(message)
        self.regbench_cause = cause if cause in resources.CAUSES else "exception"


def allocation(stage, attempt=1):
    """``{"cpus", "mem_gb", "hours", "time"}`` of one attempt of a stage's job;
    ``time`` as ``D-HH:MM:SS``, never above the site's longest job."""
    row = ALLOCATION[stage]
    attempt = int(attempt)
    if not 1 <= attempt <= ATTEMPTS:
        raise ValueError(f"attempt {attempt}: 1 to {ATTEMPTS}")
    hours = min(row["hours"] * attempt, ladder.MAX_HOURS)
    return {
        "cpus": row["cpus"],
        "mem_gb": row["mem_gb"] * attempt,
        "hours": hours,
        "time": f"{hours // 24}-{hours % 24:02d}:00:00",
    }


# ── the tile arrays ───────────────────────────────────────────────────────────
def chunks(total, most=ARRAY_MAX):
    """``[(offset, size), ...]``: ``total`` array tasks in arrays of at most ``most``."""
    total, most = int(total), int(most)
    if most < 1:
        raise ValueError(f"an array holds at least one task, got {most}")
    return [(o, min(most, total - o)) for o in range(0, total, most)]


def locate(counts, flat):
    """``(case index, tile row)`` of the ``flat``-th tile task, the cases' tiles laid
    end to end in the order of ``counts``; None past the last one."""
    flat = int(flat)
    if flat < 0:
        return None
    for index, n in enumerate(counts):
        if flat < n:
            return index, flat
        flat -= n
    return None


def write_layout(path, rows):
    """``rows`` = ``[(case id, tiles)]`` in case order, as the launcher's tile arrays
    are laid out: one line per case, ``index, case id, tiles, offset``."""
    offset, lines = 0, []
    for index, (case_id, n) in enumerate(rows):
        lines.append(f"{index}\t{case_id}\t{int(n)}\t{offset}")
        offset += int(n)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(line + "\n" for line in lines))
    return offset


def read_layout(path):
    """``[(case id, tiles)]`` of a layout file."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            _, case_id, n, _ = line.split("\t")
            rows.append((case_id, int(n)))
    return rows


# ── one job's record ──────────────────────────────────────────────────────────
def record_path(stage_dir, stage, row=None):
    name = stage if row is None else f"{stage}_{int(row):06d}"
    return Path(stage_dir) / "jobs" / f"{name}.json"


def read_record(stage_dir, stage, row=None):
    try:
        rec = json.loads(record_path(stage_dir, stage, row).read_text())
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def _write(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, path)


def retryable(rec):
    """Would the pipeline run this job again? (module docstring, "Retries")"""
    if rec is None:
        return True
    if rec.get("ok"):
        return False
    if rec.get("cause") == KILLED and rec.get("exit") is None:
        return True
    return rec.get("exit") in RETRY_EXIT


def decide(rec, attempt, command, submission="", force=False):
    """``(run now?, why)`` for the job of attempt ``attempt`` (1 is the first).

    A record another submission left is evidence of nothing, except an ``ok`` one for
    the very same command, which is reused (``force`` runs it again)."""
    attempt = int(attempt)
    same = rec is not None and rec.get("command") == list(command)
    if rec is not None and rec.get("ok") and same and not (force and attempt == 1):
        return False, "already ok for this command"
    mine = rec is not None and same and ladder.mine(rec, submission)
    if attempt == 1:
        return True, "the first attempt"
    if not mine:
        return True, "no record of this submission: the attempt before never started"
    done = int(rec.get("attempt") or 1)
    if done >= attempt:
        return False, f"attempt {done} already ran"
    if retryable(rec):
        return True, f"attempt {done} ended with exit {rec.get('exit')}: retried"
    return False, f"attempt {done} failed with exit {rec.get('exit')}: not retried"


def cause_of(exit_code, error=""):
    """The benchmark's cause for a stage command that ended with ``exit_code``."""
    if exit_code == 137 or resources.classify_text(error) == "oom":
        return "oom"
    if exit_code in (140, 143):
        return "timeout"
    return "exception"


def _summary(rec):
    keys = (
        "attempt",
        "ok",
        "exit",
        "cause",
        "started",
        "ended",
        "wall_s",
        "cpu_s",
        "peak_rss_mb",
        "cpus",
        "mem_gb",
        "launch_id",
        "log_path",
    )
    return {k: rec.get(k) for k in keys}


def run_command(
    stage_dir, stage, command, row=None, attempt=1, execution="jobs", env=None
):
    """Run one stage command as a process of its own and leave its record; returns it.

    ``command`` is the whole command line. ``execution`` is ``jobs`` (a scheduler job
    of its own) or ``local`` (one of several inside a single run)."""
    path = record_path(stage_dir, stage, row)
    old = read_record(stage_dir, stage, row)
    host = resources.host_info()
    history = []
    if (
        old
        and old.get("command") == list(command)
        and ladder.mine(old, host["submission"])
        and int(old.get("attempt") or 1) < int(attempt)
    ):
        history = [*(old.get("history") or []), _summary(old)]
    alloc = allocation(stage, attempt)
    rec = {
        "stage": stage,
        "row": row,
        "attempt": int(attempt),
        "ok": False,
        "exit": None,
        "cause": KILLED,
        "error": "killed before the command ended (out of memory or time limit?)",
        "command": list(command),
        "execution": execution,
        "cpus": alloc["cpus"],
        "mem_gb": alloc["mem_gb"],
        "time_limit": alloc["time"],
        "launch_id": host["launch_id"],
        "host": host["host"],
        "log_path": host["log_path"],
        "submission": host["submission"],
        "started": time.time(),
        "history": history,
    }
    _write(path, rec)
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    tail = []
    for line in proc.stdout:
        text = line.decode(errors="replace")
        if execution == "jobs":
            sys.stdout.write(text)
        tail = [*tail[-39:], text]
    proc.stdout.close()
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = code = os.waitstatus_to_exitcode(status)
    if code < 0:  # ended by a signal: the status a shell (and Nextflow) reports
        code = 128 - code
    wall = time.perf_counter() - t0
    # ru_maxrss: bytes on macOS, KiB on Linux
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    error = "" if code == 0 else "".join(tail)[-2000:].strip() or f"exit {code}"
    rec.update(
        ok=code == 0,
        exit=code,
        cause="" if code == 0 else cause_of(code, error),
        error=error,
        ended=time.time(),
        wall_s=wall,
        cpu_s=usage.ru_utime + usage.ru_stime,
        peak_rss_mb=usage.ru_maxrss * scale / resources.MB,
    )
    _write(path, rec)
    return rec


# ── the cost of a round ───────────────────────────────────────────────────────
def attempts_of(rec):
    """Every attempt of one job, the last one last."""
    return [*(rec.get("history") or []), _summary(rec)]


def _number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def summarise(coarse, tiles, solve):
    """The cost of one round from its job records (module docstring)."""
    stages = [(COARSE, None, coarse), *((TILE, i, t) for i, t in enumerate(tiles))]
    stages.append((SOLVE, None, solve))
    every, spent, unmeasured = [], {COARSE: 0.0, SOLVE: 0.0}, 0
    longest_tile = 0.0
    largest = {"mb": math.nan, "stage": "", "row": None}
    for stage, row, rec in stages:
        own = 0.0
        for a in attempts_of(rec):
            if not _number(a.get("wall_s")):
                unmeasured += 1
                continue
            every.append((stage, a))
            own += a["wall_s"]
            peak = a.get("peak_rss_mb")
            if _number(peak) and not peak <= largest["mb"]:
                largest = {"mb": float(peak), "stage": stage, "row": row}
        if stage == TILE:
            longest_tile = max(longest_tile, own)
        else:
            spent[stage] += own
    starts = [a["started"] for a in attempts_of(coarse) if _number(a.get("started"))]
    elapsed = (
        solve["ended"] - min(starts) if starts and _number(solve.get("ended")) else None
    )
    # the most CPUs allocated to jobs of this round at one moment
    edges = []
    for _, a in every:
        if _number(a.get("started")) and _number(a.get("ended")):
            edges += [(a["started"], 1, a["cpus"]), (a["ended"], 0, -a["cpus"])]
    at_once = now = 0
    for _, _, step in sorted(edges):
        now += step
        at_once = max(at_once, now)
    n_jobs = len(every) + unmeasured
    return {
        "elapsed_with_queue_s": elapsed,
        "wall_unlimited_parallel_s": spent[COARSE] + longest_tile + spent[SOLVE],
        "coarse_s": spent[COARSE],
        "longest_tile_s": longest_tile,
        "solve_s": spent[SOLVE],
        "cpu_h_allocated": sum(a["cpus"] * a["wall_s"] for _, a in every) / 3600.0,
        "cpu_h_measured": sum(a.get("cpu_s") or 0.0 for _, a in every) / 3600.0,
        "largest_job_peak_mb": largest["mb"],
        "largest_job": largest["stage"],
        "largest_job_row": largest["row"],
        "n_jobs": n_jobs,
        "n_tiles": len(tiles),
        "n_retries": n_jobs - len(stages),
        "peak_cpus_at_once": at_once,
        "unmeasured_attempts": unmeasured,
        "allocation": {s: allocation(s) for s in (*TO_TRANSFORM, STITCH)},
    }
