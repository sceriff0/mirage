"""CPU-hours and memory-hours, defined once and computed the same way for every method.

**The scope is the registration, for every method.** Estimating the transform: what the
registration clock covers (``regbench.cost.CLOCK`` says what that is per method). The
method's own writer has its own pair of columns. The benchmark's own work (moving the
nuclei through the map, the tissue reduction, scoring) is in neither.

**An attempt** is one try at a registration, or at a part of one: for a method that
registers in one process, the job of one rung of the memory ladder; for a registration
that runs as several scheduler jobs per round (``regbench.stages``), one attempt of one
stage job. Each attempt has the CPUs and the memory it was given, how long it ran, and
the CPU time its processes used. An attempt is *good* when its result is part of the
registration the row reports, *lost* when it ended for a cause that is not the method's
(``regbench.resources.NOT_THE_METHOD``: a node that failed, a cancelled job, the
benchmark's own step), and *wasted* otherwise (a rung that ran out of memory, a tile
attempt that was retried, the last attempt of a registration that failed).

==========================  ========================================================
``cpu_h_used``              user + system CPU time of the good attempts' processes
``cpu_h_reserved``          allocated CPUs x duration, summed over the good attempts
``cpu_efficiency``          ``cpu_h_used / cpu_h_reserved``
``gb_h_reserved``           requested memory (GB) x duration, over the same attempts
``wasted_cpu_h_used``,
``wasted_cpu_h_reserved``,
``wasted_gb_h_reserved``    the same three over the wasted attempts
``total_*``                 good + wasted: what the registration cost the cluster
``lost_cpu_h_used``,
``lost_cpu_h_reserved``,
``lost_gb_h_reserved``      the same three over the attempts lost to causes that are
                            not the method's: in NO ``wasted_*`` and NO ``total_*``
``write_cpu_h_used``,
``write_cpu_h_reserved``    the method's own writer, the attempt that wrote the image
``write_wasted_cpu_h_reserved``  writer attempts that failed, where they are recorded
``cpu_h_scheduler``         the scheduler's own CPU time of the jobs that ran the good
                            attempts: a cross-check (below)
``cpu_h_used_vs_scheduler`` ``cpu_h_used / cpu_h_scheduler``
``cpu_h_used_per_gpx``      ``cpu_h_used`` per gigapixel of moving image registered
``device``                  ``gpu`` when the registration had a GPU, else ``cpu``
==========================  ========================================================

**How CPU time is measured.** By a counter the kernel keeps, never by sampling
(``regbench.resources``, which says what each one is and cites it). A registration in
one process reads, at the start and end of every stretch it bills, the CPU counter of
its own scheduler step's cgroup (``cgroup_v2``: ``usage_usec`` of ``cpu.stat``;
``cgroup_v1``: ``cpuacct.usage``), which counts every process of the task however
short its life. Where there is no such counter (a laptop, a site without a cgroup per
job) it reads ``os.times()``, this process and every child that has been waited for,
plus the processes still alive (``rusage+live``). A registration in several jobs is
measured by ``os.wait4`` of each stage command: the same kind of figure. The column
``cpu_instrument`` (``run.json: resources.cpu_instrument``) says which one a row had.
Only ``polled (may undercount)``, the last resort where neither exists, can miss CPU:
there ``cpu_h_used`` is a LOWER BOUND. The stretches taken off the method's clock are
read with the same counter: ``write_cpu_h_used`` is the writer's (``billing_note``
says when the stretch holds more than the writer), and the benchmark's own point
mapping is ``run.json: warp_cpu_s``, in no column of the method.

**The scheduler's figure is a cross-check, not the number.** ``cpu_h_scheduler`` is
``TotalCPU`` of ``sacct`` for the jobs concerned (``logs/accounting.psv``), asked WITH
the jobs' steps: "Without including steps, utilization statistics for job
allocation(s) will be reported as zero" (``-X``), and ``TotalCPU`` is "the sum of the
SystemCPU and UserCPU time used by the job or job step" (sacct manual,
slurm.schedmd.com/sacct.html ``[PARTIAL: the -X / --allocations option, the entries
of JobID, JobIDRaw, ElapsedRaw, State, SystemCPU, TotalCPU and UserCPU and the
--allocations example read from the page itself; the other options and fields not
read]``). A job's figure is the larger of its own line and the sum of its steps'
lines (``ladder.parse_accounting``). For a row that is one job, that job also holds
the benchmark's own work, the method's writer and every other result of the same
launch, so it is larger than the registration by an amount that differs between
methods: it cannot be the main number. ``run.json: resources.launch_cpu_s`` is the
whole launch as the row's own counter read it, the figure to hold against it. It is
blank when the accounting has no CPU time for one of the jobs. Slurm's own caveat:
"For the steps interrupted by signal (e.g. scancel, job timeout) TotalCPU provides a
measure of the task's parent process and may not include CPU time of child
processes". A zero is therefore read as "nothing", never as a ratio.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

GOOD, WASTED, LOST = "good", "wasted", "lost"
GPU, CPU = "gpu", "cpu"
# the three sums of a set of attempts: (column stem, key of an attempt)
SUMS = (
    ("cpu_h_used", "used_h"),
    ("cpu_h_reserved", "reserved_h"),
    ("gb_h_reserved", "gb_h"),
)
COLUMNS = (
    "device",
    "device_name",
    *(stem for stem, _ in SUMS),
    *(f"wasted_{stem}" for stem, _ in SUMS),
    *(f"total_{stem}" for stem, _ in SUMS),
    *(f"lost_{stem}" for stem, _ in SUMS),
    "write_cpu_h_used",
    "write_cpu_h_reserved",
    "write_wasted_cpu_h_reserved",
    "moving_gpx",
    "cpu_h_used_per_gpx",
    "cpu_h_scheduler",
    "scheduler_jobs",
    "cpu_h_used_vs_scheduler",
    "billing_note",
)
NOT_IN_RECORDS = (
    "gb_h_reserved is blank: the stage jobs' records were not read (they are read by "
    "`score`, from the results folder)"
)
# `run.json: write_cpu_from`: what `write_cpu_s` is the CPU time of. The writer's own
# call, alone; or the stretches the method's adapter took off the clock around a writer
# it ran at a yield, shared between the launch's written rows in proportion to `write_s`
WRITE_CPU_CALL = "writer_call"
WRITE_CPU_STRETCH = "paused_stretches"
WRITE_CPU_IS_THE_STRETCH = (
    "write_cpu_h_used is the CPU of the stretches the method's adapter took off the "
    "clock around its writer (the writer, and the adapter's own bookkeeping beside "
    "it), shared between the launch's written rows as write_s is"
)
WRITE_CPU_HOLDS = (
    "write_cpu_h_used is the CPU of the whole writing step, which also holds the "
    "parts write_s leaves out ({})"
)
RECORDS_CHANGED = (
    "gb_h_reserved and cpu_h_scheduler are blank: the stage jobs' records on disk are "
    "not the ones this result was made from"
)


def _n(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan
    return v


def attempt(kind, cpus, mem_gb, wall_s, cpu_s, job=""):
    """One attempt, as hours: ``used_h`` (measured CPU time), ``reserved_h`` (allocated
    CPUs x duration), ``gb_h`` (requested memory x duration). A figure nobody recorded
    is NaN, never zero."""
    wall = _n(wall_s)
    return {
        "kind": kind,
        "used_h": _n(cpu_s) / 3600.0,
        "reserved_h": _n(cpus) * wall / 3600.0,
        "gb_h": _n(mem_gb) * wall / 3600.0,
        "job": str(job or ""),
    }


def hours(kind, used_h, reserved_h, gb_h=math.nan):
    """Attempts that are known only as sums (a round's stage jobs, as its run recorded
    them)."""
    return {
        "kind": kind,
        "used_h": _n(used_h),
        "reserved_h": _n(reserved_h),
        "gb_h": _n(gb_h),
        "job": "",
    }


def kind_of(cause, failed=WASTED):
    """The kind of an attempt that did not end in the row's registration: lost when
    its cause is not the method's, else ``failed``."""
    from . import losses, resources

    cause = str(cause or "")
    lost = cause in resources.NOT_THE_METHOD or cause in losses.NOT_THE_METHOD_EXTRA
    # R26: an unsettled kill is not the method's for the scores, but it is billed
    lost = lost and cause not in losses.BILLED_AS_WASTED
    return LOST if lost else failed


def _recorded(attempts, key):
    """The sum of what a set of failed attempts recorded: 0 without an attempt, NaN
    when there are attempts and none has the figure (nobody knows what they cost; such
    an attempt is counted in ``wasted_unknown``)."""
    known = [a[key] for a in attempts if a[key] == a[key]]
    return math.fsum(known) if known or not attempts else math.nan


def sums(attempts):
    """The twelve columns of a row's attempts. Good: NaN when the row has no good
    attempt or one of them lacks the figure. Wasted, and lost: the sum of what was
    recorded (:func:`_recorded`). Total: good + wasted, never the lost attempts: NaN
    when the row has neither kind of attempt or neither has a figure, and NaN when a
    good attempt's figure is missing (a sum without it would be the wasted part alone)."""
    good = [a for a in attempts if a["kind"] == GOOD]
    bad = [a for a in attempts if a["kind"] == WASTED]
    lost = [a for a in attempts if a["kind"] == LOST]
    out = {}
    for stem, key in SUMS:
        g = sum(a[key] for a in good) if good else math.nan
        w = _recorded(bad, key)
        out[stem] = g
        out[f"wasted_{stem}"] = w
        parts = [v for v in (g, w) if v == v]
        if not parts or not (good or bad) or good and g != g:
            out[f"total_{stem}"] = math.nan
        else:
            out[f"total_{stem}"] = math.fsum(parts)
        out[f"lost_{stem}"] = _recorded(lost, key)
    return out


# ── a registration in several scheduler jobs: its attempts, from its records ──────
def stage_attempts(stage_dir):
    """Every attempt of COARSE, each tile and SOLVE of one round, from the records its
    stage jobs left (``<stage dir>/jobs/*.json``, ``regbench.stages``): ``(attempts,
    unmeasured)``. An attempt is good when it is the one its job ended ok with, and
    lost when it failed for a cause that is not the method's (:func:`kind_of`). One
    without a recorded duration is counted in ``unmeasured`` and is in no sum. None
    when the folder holds no record."""
    from . import stages

    folder = Path(stage_dir) / "jobs"
    out, unmeasured, found = [], 0, False
    for path in sorted(folder.glob("*.json")):
        name = path.stem.split("_")[0]
        if name not in stages.TO_TRANSFORM:
            continue  # the writer and the benchmark's own step are not the registration
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict):
            continue
        found = True
        runs = stages.attempts_of(rec)
        for i, a in enumerate(runs):
            if _n(a.get("wall_s")) != _n(a.get("wall_s")):
                unmeasured += 1
                continue
            ok = i == len(runs) - 1 and bool(a.get("ok"))
            out.append(
                attempt(
                    GOOD if ok else kind_of(a.get("cause")),
                    a.get("cpus"),
                    a.get("mem_gb"),
                    a.get("wall_s"),
                    a.get("cpu_s"),
                    a.get("launch_id"),
                )
            )
    return (out, unmeasured) if found else None


def unreadable_stage_records(stage_dir):
    """How many of a round's registration stage records (``<stage dir>/jobs/*.json``
    of COARSE, a tile or SOLVE) could not be read: :func:`stage_attempts` has none of
    their attempts, so where it is not 0 the row's times and CPU-hours are lower bounds
    (``resources.csv: unmeasured_records``, S5). 0 for a folder with no record."""
    from . import stages

    n = 0
    for path in sorted((Path(stage_dir) / "jobs").glob("*.json")):
        if path.stem.split("_")[0] not in stages.TO_TRANSFORM:
            continue
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            n += 1
            continue
        n += not isinstance(rec, dict)
    return n


def find_stage_dir(out, dataset, case_id, variant, launch_id):
    """The folder holding the stage records of the round a result row is of. Every
    result of one launch shares it, and it sits beside the launch's last result: the
    row's own folder, else the folder of a result of the same case and launch."""
    own = Path(out) / variant / dataset / str(case_id) / "stages"
    if own.is_dir():
        return own
    for other in sorted(Path(out).glob(f"*/{dataset}/{case_id}/stages")):
        try:
            run = json.loads((other.parent / "run.json").read_text())
        except (OSError, ValueError):
            continue
        if launch_id and run.get("launch_id") == launch_id:
            return other
    return None


# ── the scheduler's own figure ────────────────────────────────────────────────────
def scheduler_hours(jobs, accounting):
    """``(CPU-hours, jobs)`` the scheduler billed the distinct ``jobs``; ``(NaN, n)``
    when it has no CPU time for one of them (absent, unreadable, or zero: module
    docstring)."""
    jobs = sorted({j for j in jobs if j})
    if not jobs:
        return math.nan, 0
    total = 0.0
    for job in jobs:
        cpu = _n((accounting.get(job) or {}).get("cpu_s"))
        if not cpu > 0:
            return math.nan, len(jobs)
        total += cpu
    return total / 3600.0, len(jobs)


def ratio(a, b):
    a, b = _n(a), _n(b)
    return a / b if a == a and b > 0 else math.nan
