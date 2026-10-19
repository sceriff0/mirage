"""What each scheduler job saw of its node, from the job's own shell (R14, R16, R5).

``slurm/env.sh`` writes one file per run task and per stage job,
``<out>/logs/telemetry/<launch id>.tsv`` (the launch id is ``run.json: launch_id`` and a
stage record's: ``<array job>_<task>``, else the job id), one ``key<TAB>value`` line
each, from ``regbench_start`` and ``regbench_end``:

=====================  ================================================================
``loadavg_start``,     ``/proc/loadavg`` at the job's start and end (the whole line; its
``loadavg_end``        first figure, the 1-minute load, is what the tables read)
``other_jobs_start``,  the jobs RUNNING on the node at the start of a RUN job (and at
``other_jobs_end``     its end only with ``REGBENCH_TEL_SQUEUE_END=1``), this one left
                       out (``squeue -h -w <node> -t RUNNING``); a stage job never asks
``cpu_model``          the node's CPU (``/proc/cpuinfo: model name``)
``nproc``,             the CPUs the job's shell may use, and the node's
``nproc_all``
``cpus_per_task``      ``SLURM_CPUS_PER_TASK``
``cgroup_peak_mb``     the job cgroup's ``memory.peak`` (v2) or
                       ``memory.max_usage_in_bytes`` (v1) at the end: the whole job
``hwm_peak_mb``        the largest sum, over the job cgroup's processes, of their
                       ``VmHWM`` (``/proc/<pid>/status``), sampled every
                       ``REGBENCH_HWM_POLL_S`` (1) seconds by a loop in the job's shell
``hwm_samples``        how many samples that loop took
=====================  ================================================================

``VmHWM`` is the kernel's high-water mark of one process's resident set (proc(5),
"Peak resident set size" ``[MEMORY]``): unlike a sampled RSS it cannot miss an
allocation shorter than the sampling interval, but a sum over processes adds peaks
that need not have happened at the same time, so it is an UPPER bound of the job's
concurrent peak, and a process that started and ended between two samples is in no
sample. The driver's own sampler (``resources.Monitor``) is unchanged: it is part of
the code every method's results are keyed on (``methods.code_files``), and a change
there re-runs every result. These files are read at table time only and are in no key.

A job without a file (a site without ``/proc``, a job that died before
``regbench_end``, a run outside the launcher) has blank columns and ``telemetry = no``,
counted in the header line (``contention_line``), never a zero. ``squeue`` is asked
under ``timeout 20`` where the node has ``timeout``.
"""

from __future__ import annotations

import math
from pathlib import Path

FOLDER = ("logs", "telemetry")
KEYS = (
    "kind",
    "host",
    "loadavg_start",
    "loadavg_end",
    "other_jobs_start",
    "other_jobs_end",
    "cpu_model",
    "nproc",
    "nproc_all",
    "cpus_per_task",
    "cgroup_peak_mb",
    "hwm_peak_mb",
    "hwm_samples",
)
# what a resource row gets from its own job's file (``columns``)
COLUMNS = (
    "load_start",
    "load_end",
    "other_jobs_start",
    "other_jobs_end",
    "peak_hwm_gb",
    "telemetry",
)


def folder(out):
    return Path(out).joinpath(*FOLDER)


def parse(text):
    """``{key: value}`` of one file; a line that is no ``key<TAB>value`` of ``KEYS``
    is skipped, and a later line of a key wins."""
    got = {}
    for line in str(text or "").splitlines():
        key, tab, value = line.partition("\t")
        if tab and key in KEYS:
            got[key] = value.strip()
    return got


def read(out):
    """``{launch id: {key: value}}`` of every file under ``<out>/logs/telemetry``."""
    got = {}
    base = folder(out)
    if not base.is_dir():
        return got
    for path in sorted(base.glob("*.tsv")):
        try:
            got[path.stem] = parse(path.read_text())
        except (OSError, UnicodeDecodeError):
            continue
    return got


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def load(value):
    """The 1-minute load of a ``/proc/loadavg`` line; NaN without one."""
    return _num(str(value or "").split(" ", 1)[0])


def of(tel, launch_id):
    """The file of one job, ``{}`` when there is none."""
    return (tel or {}).get(str(launch_id or "")) or {}


def columns(tel, launch_id):
    """``COLUMNS`` of a resource row whose job is ``launch_id``."""
    t = of(tel, launch_id)

    def blank(v):  # a figure the file does not have is blank, as in every cost table
        return v if v == v else ""

    return {
        "load_start": blank(load(t.get("loadavg_start"))),
        "load_end": blank(load(t.get("loadavg_end"))),
        "other_jobs_start": blank(_num(t.get("other_jobs_start"))),
        "other_jobs_end": blank(_num(t.get("other_jobs_end"))),
        "peak_hwm_gb": blank(_num(t.get("hwm_peak_mb")) / 1024.0),
        "telemetry": "yes" if t else "no",
    }


def _median(values):
    v = sorted(x for x in values if x == x)
    if not v:
        return math.nan
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2.0


def contention_line(res_rows):
    """The header line of every cost table (R14): how loaded the nodes were, from the
    rows' own jobs. Wall times are of SHARED nodes; the headline is CPU-hours."""
    ran = [r for r in res_rows if r.get("cause") != "not_run"]
    told = [r for r in ran if r.get("telemetry") == "yes"]
    head = (
        "contention (shared nodes): every wall time here was measured on nodes shared "
        "with other jobs, this benchmark's own among them, so the headline is "
        "CPU-hours (`cpu_h_used`), which co-location moves far less than wall time. "
    )
    if not told:
        return head + (
            f"No job of the {len(ran)} results left its node telemetry "
            "(`logs/telemetry/<launch id>.tsv`): the load is unknown."
        )
    loads = [_num(r.get(k)) for r in told for k in ("load_start", "load_end")]
    others = [_num(r.get(k)) for r in told for k in ("other_jobs_start", "other_jobs_end")]
    loads = [v for v in loads if v == v]
    others = [v for v in others if v == v]
    return head + (
        f"{len(told)} of {len(ran)} results' jobs left their telemetry"
        + (
            f": 1-minute load at start and end median {_median(loads):g}, at most "
            f"{max(loads):g}"
            if loads
            else ""
        )
        + (
            f"; other running jobs on the node median {_median(others):g}, at most "
            f"{int(max(others))}"
            if others
            else ""
        )
        + f"; {len(ran) - len(told)} without (`telemetry = no`). `load_start`, "
        "`load_end`, `other_jobs_start` and `other_jobs_end` give it per row; "
        "`REGBENCH_WAVES=1` is the timing replicate, one method family at a time."
    )
