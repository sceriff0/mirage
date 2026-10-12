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
registration the row reports, and *wasted* otherwise (a rung that ran out of memory, a
tile attempt that was retried, the last attempt of a registration that failed).

==========================  ========================================================
``cpu_h_used``              user + system CPU time of the good attempts' processes
``cpu_h_reserved``          allocated CPUs x duration, summed over the good attempts
``cpu_efficiency``          ``cpu_h_used / cpu_h_reserved``
``gb_h_reserved``           requested memory (GB) x duration, over the same attempts
``wasted_cpu_h_used``,
``wasted_cpu_h_reserved``,
``wasted_gb_h_reserved``    the same three over the wasted attempts
``total_*``                 good + wasted: what the registration cost the cluster
``write_cpu_h_used``,
``write_cpu_h_reserved``    the method's own writer, the attempt that wrote the image
``write_wasted_cpu_h_reserved``  writer attempts that failed, where they are recorded
``cpu_h_scheduler``         the scheduler's own CPU time of the jobs that ran the good
                            attempts: a cross-check (below)
``cpu_h_used_vs_scheduler`` ``cpu_h_used / cpu_h_scheduler``
``cpu_h_used_per_gpx``      ``cpu_h_used`` per gigapixel of moving image registered
``device``                  ``gpu`` when the registration had a GPU, else ``cpu``
==========================  ========================================================

**How CPU time is measured, and what that can miss.** A registration in one process is
measured by ``resources.Monitor``: a thread reads the CPU time of the process and of
every descendant five times a second and keeps each process's last reading. CPU a
child spends between its last reading and its exit is not seen, and a child that
starts and ends between two readings is not seen at all: measured on one laptop, 40
children of 0.05 s CPU each, one after another, were billed 33 % of their CPU time,
children of 0.5 s 80 %, four workers of 3 s 99 %. So ``cpu_h_used`` of such a row is a
LOWER BOUND, close for a method whose work is in long-lived processes and threads. A
registration in several jobs is measured by ``os.wait4`` of each stage command, which
has no such gap. A writer's CPU time is measured only where the writer is a process of
its own (STARE's STITCH); elsewhere ``write_cpu_h_used`` is blank.

**The scheduler's figure is the detector, not the number.** ``cpu_h_scheduler`` is
``TotalCPU`` of ``sacct`` for the jobs concerned (``logs/accounting.psv``). For a row
that is one job, that job also holds the benchmark's own work, the method's writer and
every other result of the same launch, so it is larger than the registration by an
amount that differs between methods: it cannot be the main number. What it shows is a
``cpu_h_used_vs_scheduler`` far below the other rows', which is what an undercount by
the reader above looks like. It is blank when the accounting has no CPU time for one
of the jobs. Slurm's own caveat: for a step ended by a signal, ``TotalCPU`` "may not
include CPU time of child processes", and with ``-X`` "utilization statistics for job
allocation(s) will be reported as zero" (sacct manual, slurm.schedmd.com/sacct.html
``[PARTIAL: the definitions of TotalCPU, UserCPU, SystemCPU, CPUTime, ElapsedRaw,
AllocCPUS and of -X, read through a summarising fetch that quoted them; the rest of
the page not read]``). A zero is therefore read as "nothing", never as a ratio.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

GOOD, WASTED = "good", "wasted"
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


def sums(attempts):
    """The nine columns of a row's attempts. Good: NaN when the row has no good attempt
    or one of them lacks the figure. Wasted: the sum of what was recorded (0 without a
    wasted attempt; an attempt nobody could bill is counted elsewhere, in
    ``wasted_unknown``). Total: good + wasted, NaN when the row has neither."""
    good = [a for a in attempts if a["kind"] == GOOD]
    bad = [a for a in attempts if a["kind"] == WASTED]
    out = {}
    for stem, key in SUMS:
        g = sum(a[key] for a in good) if good else math.nan
        w = math.fsum(a[key] for a in bad if a[key] == a[key])
        out[stem] = g
        out[f"wasted_{stem}"] = w
        out[f"total_{stem}"] = (
            math.nan if not attempts else math.fsum(v for v in (g, w) if v == v)
        )
    return out


# ── a registration in several scheduler jobs: its attempts, from its records ──────
def stage_attempts(stage_dir):
    """Every attempt of COARSE, each tile and SOLVE of one round, from the records its
    stage jobs left (``<stage dir>/jobs/*.json``, ``regbench.stages``): ``(attempts,
    unmeasured)``. An attempt is good when it is the one its job ended ok with. One
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
                    GOOD if ok else WASTED,
                    a.get("cpus"),
                    a.get("mem_gb"),
                    a.get("wall_s"),
                    a.get("cpu_s"),
                    a.get("launch_id"),
                )
            )
    return (out, unmeasured) if found else None


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
