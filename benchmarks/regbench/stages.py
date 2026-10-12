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
attempt of the same job. ``identity`` is the caller's digest of what the job is computed
from beyond its command line (its inputs' bytes, the files the stage before wrote, the
version of the code): ``decide`` reuses an ``ok`` record only of the same command AND
the same identity.

**The cost of a round** (``summarise``), from the records of COARSE, every tile and
SOLVE, every attempt of each (STITCH is the writer and is timed apart):

``elapsed_with_queue_s``        from the start of the first COARSE attempt to the end
                                of SOLVE, on the wall clock: queue waits between jobs
                                and the submitter's ceiling are in it
``wall_unlimited_parallel_s``   COARSE + the LONGEST tile + SOLVE, each the attempt its
                                job ended ok with: the wall time with every tile
                                running at once and no queue. This is the round's
                                ``register_s``
``cpu_h_allocated``             sum over those attempts of allocated CPUs x duration
``cpu_h_measured``              sum over those attempts of measured CPU time
``wasted_attempts``, ``wasted_wall_s``, ``wasted_cpu_h_allocated``,
``wasted_cpu_h_measured``       every attempt that did not end its job ok (a retried
                                one, and the last one of a job that failed): the same
                                rule as for a method that is one job, whose failed
                                attempt is ``wasted_*`` and in no cost
``largest_job_peak_mb``, ``largest_job``  the peak memory of the largest single attempt,
                                failed ones included, and which stage it was (the
                                round's ``peak_rss_mb``)
``n_jobs``, ``n_tiles``, ``n_retries``  attempts run, tiles, attempts beyond the first
``peak_cpus_at_once``           the most allocated CPUs in jobs of this round that
                                overlapped in time
``unmeasured_attempts``         attempts that were killed before they could record a
                                duration (a job ended at its time limit): counted, in
                                no sum, so every sum of that round is a LOWER BOUND
``elapsed_with_queue_s``        blank when COARSE and SOLVE are of two submissions (a
                                reused stage is not a queue wait)
``complete``                    COARSE, every tile and SOLVE each ended ok
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
# The tile jobs reach the queue through a throttled submitter (slurm/feed_tiles.sbatch),
# as Nextflow's queue size does for the pipeline: arrays of at most ARRAY_MAX tasks, at
# most QUEUE_MAX tile tasks of a run pending or running. Both are the defaults of the
# launcher's REGBENCH_STAGE_ARRAY_MAX and REGBENCH_STAGE_QUEUE_MAX.
ARRAY_MAX = 100
QUEUE_MAX = 100
# a stage job that was never submitted: the launcher's failure (regbench.failure_log)
NOT_SUBMITTED = "not_submitted"


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


def feed_state(out, submission, run, dataset):
    """The path prefix of a run's tile submitter on a dataset: ``<prefix>.tsv`` is the
    layout, ``.fed.tsv`` what it submitted, ``.done`` that nothing is left, ``.refused``
    the scheduler's refusal (slurm/feed_tiles.sbatch)."""
    return Path(out) / "logs" / "stages" / str(submission) / f"{run}-{dataset}"


def feed_problem(out, submission, run, dataset):
    """Why a tile of this submission may never have been submitted: the text of the
    submitter's refusal or of its giving up (a queue it could not read, tile tasks
    that did not leave it), a note that it did not finish, or ``""`` when it finished
    (or when there is no submission: a run outside the launcher)."""
    if not submission:
        return ""
    state = feed_state(out, submission, run, dataset)
    done, refused = Path(f"{state}.done"), Path(f"{state}.refused")
    if done.exists():
        return ""
    try:
        lines = refused.read_text().splitlines()
    except OSError:
        return (
            "the tile submitter did not finish (it was ended at its time limit, or "
            f"lost): no {done.name} under {done.parent}"
        )
    gave_up = any(ln.startswith("gave up:") for ln in lines)
    return (
        (
            "the tile submitter gave up"
            if gave_up
            else "the tile submitter was refused by the scheduler"
        )
        + ": "
        + " | ".join(lines[:4])
    )


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


def same_job(rec, command, identity=None):
    """Is ``rec`` the record of THIS job: the same command line, computed from the same
    things (``identity``, below)?"""
    return (
        rec is not None
        and rec.get("command") == list(command)
        and rec.get("identity") == identity
    )


def decide(rec, attempt, command, submission="", force=False, identity=None):
    """``(run now?, why)`` for the job of attempt ``attempt`` (1 is the first).

    ``identity`` says what the job is computed FROM, beyond its command line: the
    caller's digest of its input files' bytes, of the files the stage before it wrote,
    and of the version of the code that runs (``methods.stare.Round.identity``). A
    record with another identity is the record of another job.

    A record another submission left is evidence of nothing, except an ``ok`` one of
    this very job, which is reused. ``force`` (``REGBENCH_FORCE=1``) reuses an ``ok``
    record only when THIS submission made it, at every attempt: without a submission
    id (a run outside the launcher) it runs the job again at attempt 1."""
    attempt = int(attempt)
    same = same_job(rec, command, identity)
    if rec is not None and rec.get("ok") and same:
        ours = bool(submission) and ladder.mine(rec, submission)
        if not force or ours or (attempt > 1 and not submission):
            return False, "already ok for this command, these inputs and this code"
        return True, "forced: the ok record is not of this submission"
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
        "time_limit",
        "error",
    )
    out = {k: rec.get(k) for k in keys}
    out["error"] = str(out["error"] or "")[-300:]
    return out


def run_command(
    stage_dir,
    stage,
    command,
    row=None,
    attempt=1,
    execution="jobs",
    env=None,
    identity=None,
    timeout_s=None,
):
    """Run one stage command as a process of its own and leave its record; returns it.

    ``command`` is the whole command line. ``execution`` is ``jobs`` (a scheduler job
    of its own) or ``local`` (one of several inside a single run). ``identity`` is
    kept in the record (``decide``). ``timeout_s`` ends the command with SIGTERM (exit
    143, a retried status) when it runs longer: the time limit of a process that
    shares its job with other work."""
    path = record_path(stage_dir, stage, row)
    old = read_record(stage_dir, stage, row)
    host = resources.host_info()
    history = []
    if (
        same_job(old, command, identity)
        and ladder.mine(old, host["submission"])
        and not old.get("ok")
    ):
        # every failed attempt of this job in this submission, whichever job of the
        # scheduler ran it
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
        "identity": identity,
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
    timer = None
    if timeout_s:
        import threading

        timer = threading.Timer(float(timeout_s), proc.terminate)
        timer.daemon = True
        timer.start()
    tail = []
    for line in proc.stdout:
        text = line.decode(errors="replace")
        if execution == "jobs":
            sys.stdout.write(text)
        tail = [*tail[-39:], text]
    proc.stdout.close()
    _, status, usage = os.wait4(proc.pid, 0)
    if timer is not None:
        timer.cancel()
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


FAILED_LISTED = (
    500  # failed stage attempts kept per round in run.json (the rest counted)
)


def attempt_cause(attempt, after=None):
    """``(cause, where it is from)`` of a stage attempt that did not end its job ok.

    An attempt that recorded its exit says so itself (``cause_of``). One that recorded
    none still holds the placeholder ``killed``: the scheduler ended the job. It is
    ``timeout`` when the evidence allows: the NEXT attempt of the same job started no
    earlier than this one's start plus its time limit (attempts are chained, so the
    next one starts when this one ends). Without a next attempt, or when the next
    one started sooner, it stays ``killed``."""
    cause = str(attempt.get("cause") or "exception")
    if cause != KILLED or attempt.get("exit") is not None:
        return cause, "its own record"
    limit = ladder._seconds(attempt.get("time_limit"))
    began, then = attempt.get("started"), (after or {}).get("started")
    if limit and _number(began) and _number(then) and then - began >= limit:
        return "timeout", (
            "no end recorded, and the next attempt started after this one's time "
            f"limit ({attempt.get('time_limit')})"
        )
    return KILLED, "no end recorded (the scheduler ended the job)"


def summarise(coarse, tiles, solve):
    """The cost of one round from its job records (module docstring). A record that is
    missing (None: the job never ran) contributes nothing; ``complete`` says whether
    COARSE, every tile and SOLVE each ended ok."""
    stages = [(COARSE, None, coarse), *((TILE, i, t) for i, t in enumerate(tiles))]
    stages.append((SOLVE, None, solve))
    good, bad, spent, unmeasured = [], [], {COARSE: 0.0, SOLVE: 0.0}, 0
    failed = []  # every attempt that did not end its job ok, for the failure log
    longest_tile = 0.0
    largest = {"mb": math.nan, "stage": "", "row": None}
    for stage, row, rec in stages:
        if rec is None:
            continue
        runs = attempts_of(rec)
        for i, a in enumerate(runs):
            last_ok = i == len(runs) - 1 and bool(a.get("ok"))
            if not last_ok:
                after = runs[i + 1] if i + 1 < len(runs) else None
                cause, how = attempt_cause(a, after)
                failed.append(
                    {
                        "stage": stage,
                        "row": row,
                        "attempt": a.get("attempt"),
                        "exit": a.get("exit"),
                        "cause": cause,
                        "cause_from": how,
                        "wall_s": a.get("wall_s"),
                        "peak_rss_mb": a.get("peak_rss_mb"),
                        "mem_gb": a.get("mem_gb"),
                        "launch_id": a.get("launch_id"),
                        "log_path": a.get("log_path"),
                        "error": a.get("error") or "",
                        "retried": after is not None,
                    }
                )
            if not _number(a.get("wall_s")):
                unmeasured += 1
                continue
            # the attempt a job ended ok with is its cost; every other one is wasted
            ok = i == len(runs) - 1 and bool(a.get("ok"))
            (good if ok else bad).append((stage, a))
            peak = a.get("peak_rss_mb")
            if _number(peak) and not peak <= largest["mb"]:
                largest = {"mb": float(peak), "stage": stage, "row": row}
            if not ok:
                continue
            if stage == TILE:
                longest_tile = max(longest_tile, a["wall_s"])
            else:
                spent[stage] += a["wall_s"]
    every = [*good, *bad]
    first = coarse or {}
    starts = [a["started"] for a in attempts_of(first) if _number(a.get("started"))]
    last = solve or {}
    # within ONE submission: a COARSE an earlier submission ran and this one reused
    # started before this submission's queue, and the difference is not a queue wait
    same = (first.get("submission") or "") == (last.get("submission") or "")
    elapsed = (
        last["ended"] - min(starts)
        if starts and same and _number(last.get("ended")) and last.get("ok")
        else None
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
    ran = sum(rec is not None for _, _, rec in stages)

    def cpu_h(rows, measured=False):
        if measured:
            return sum(a.get("cpu_s") or 0.0 for _, a in rows) / 3600.0
        return sum(a["cpus"] * a["wall_s"] for _, a in rows) / 3600.0

    return {
        "failed_attempts": failed[:FAILED_LISTED],
        "failed_attempts_not_listed": max(0, len(failed) - FAILED_LISTED),
        "complete": all(rec is not None and rec.get("ok") for _, _, rec in stages),
        "elapsed_with_queue_s": elapsed,
        "wall_unlimited_parallel_s": spent[COARSE] + longest_tile + spent[SOLVE],
        "coarse_s": spent[COARSE],
        "longest_tile_s": longest_tile,
        "solve_s": spent[SOLVE],
        "cpu_h_allocated": cpu_h(good),
        "cpu_h_measured": cpu_h(good, True),
        # every attempt that did not end its job ok: in no cost above
        "wasted_attempts": len(bad),
        "wasted_wall_s": sum(a["wall_s"] for _, a in bad),
        "wasted_cpu_h_allocated": cpu_h(bad),
        "wasted_cpu_h_measured": cpu_h(bad, True),
        "largest_job_peak_mb": largest["mb"],
        "largest_job": largest["stage"],
        "largest_job_row": largest["row"],
        "n_jobs": n_jobs,
        "n_tiles": len(tiles),
        "n_retries": max(0, n_jobs - ran),
        "peak_cpus_at_once": at_once,
        "unmeasured_attempts": unmeasured,
        "allocation": {s: allocation(s) for s in (*TO_TRANSFORM, STITCH)},
    }
