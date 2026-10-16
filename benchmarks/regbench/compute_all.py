"""compute-all: what the whole Mirage pipeline costs, per patient, per registration backend.

The full pipeline (no ``--start``/``--stop``: convert, preprocess, register, segment,
quantify, merge and pyramid, export) runs once per patient with
``registration_method=valis`` and once with ``registration_method=tiled`` (STARE), each
run traced by Nextflow. The launcher is ``slurm/compute_all.sbatch`` (one Nextflow head
job per run), submitted by ``submit_cohort.sh --compute-all`` LAST, after the scorer, and
ALONE on ``REGBENCH_COMPUTE_LANES`` lanes: a cost measured under contention the harness
caused itself is not a cost (the arms' precedent, run_arms.sh's "timed ALONE").

**What is pinned.** Everything is at the shipped defaults of ``nextflow.config`` but:

* ``registration_method``: the backend, the one thing compared;
* ``PINNED``: values restated AT their shipped default so that a site profile cannot
  move them (``enable_trace``: tracing off would remove the measurement). The guard
  ``test_compute_all.py::test_the_pinned_params_are_real_and_at_their_shipped_default``
  fails when one is not a schema param, not of its type, or not the default any more;
* ``RULED``: the user's ruling of 2026-10-15, ``seg_method=stardist`` (the benchmark's
  own nuclei are StarDist's, and so was the arms' compute run). NOT the shipped default
  (``instantseg``): a deliberate difference, checked against the schema's enum.

Every one of them reaches Nextflow through ``-params-file`` (typed by
``params_json.build`` against ``nextflow_schema.json``): Nextflow 26 delivers a
command-line param as a String, and a boolean there is refused.

**Each run has its own launch directory, work directory, trace directory and run name**
(``plan``): two Nextflow runs sharing a trace directory corrupt ``trace.txt``, and a
``-resume`` would turn tasks into ``CACHED`` rows that cost nothing. The launcher never
resumes, and the tables count ``CACHED`` and ``FAILED`` tasks and keep such a run out
of the backend's total.

**The tables** (``score`` writes them next to ``cost_end_to_end.csv`` when there is a
trace under ``<out>/compute``):

``pipeline_cost.csv`` (``COLUMNS``)
    one row per task of each trace: Nextflow's own figures (``realtime_s``,
    ``duration_s``, ``pcpu`` = ``%cpu``, ``cpus``, ``memory_gb``, ``peak_rss_gb``,
    ``peak_vmem_gb``, ``read_gb``, ``write_gb``, ``submit``/``start``/``complete``) and
    three products of them, named as ``billing.py`` names the regbench methods' and
    defined the same way (CPU time used; allocated CPUs x time; requested memory x time):
    ``cpu_h_used = pcpu/100 * realtime_s/3600``, ``cpu_h_reserved = cpus *
    realtime_s/3600``, ``gb_h_reserved = memory_gb * realtime_s/3600``. They are NOT the
    same measurement: ``%cpu`` is what Nextflow's task wrapper reports, not the cgroup
    counter billing.py reads, and ``realtime`` is the task's own run time inside the
    allocation Nextflow asked for. So the two are never added.
``pipeline_cost_summary.csv`` (``SUMMARY_COLUMNS``)
    one row per (backend, patient) and one per backend with ``patient = all``.
``pipeline_cost_by_process.csv`` (``BY_PROCESS_COLUMNS``)
    one row per (backend, process) over the patients in the ``all`` row.
"""

from __future__ import annotations

import csv
import math
import re
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from . import trace

BACKENDS = ("valis", "tiled")
# restated AT the shipped default (nextflow.config), so a site profile cannot remove it
PINNED = {"enable_trace": True}
# the user's ruling (2026-10-15): the benchmark's nuclei and the arms' compute run were
# StarDist's. Not the shipped default (instantseg): a ruling, not a restatement
RULED = {"seg_method": "stardist"}
REGISTRATION_PROCESSES = {
    "valis": ("REGISTER",),
    "tiled": ("TILED_COARSE", "TILED_REG_TILE", "TILED_SOLVE", "TILED_STITCH"),
}
SHEET_PATIENT = "patient_id"
SHEET_SLIDE = "path_to_file"

Run = namedtuple(
    "Run",
    "backend patient launch_dir sheet outdir trace_dir work_dir params_file name",
)

COLUMNS = (
    "backend",
    "patient",
    "process",
    "process_path",
    "task_id",
    "tag",
    "status",
    "exit",
    "cpus",
    "memory_gb",
    "realtime_s",
    "duration_s",
    "pcpu",
    "cpu_h_used",
    "cpu_h_reserved",
    "gb_h_reserved",
    "peak_rss_gb",
    "peak_vmem_gb",
    "read_gb",
    "write_gb",
    "submit",
    "start",
    "complete",
    "is_registration",
)
SUMMARY_COLUMNS = (
    "backend",
    "patient",
    "n_patients",
    "n_tasks",
    "n_completed",
    "n_failed",
    "n_retried",
    "n_cached",
    "complete_run",
    "wall_s",
    "cpu_h_used",
    "cpu_h_reserved",
    "gb_h_reserved",
    "peak_rss_gb",
    "registration_wall_s",
    "registration_cpu_h_used",
    "registration_cpu_h_reserved",
    "registration_share_cpu_h_used",
    "input_gb",
)
BY_PROCESS_COLUMNS = (
    "backend",
    "process",
    "is_registration",
    "n_patients",
    "n_tasks",
    "n_failed",
    "realtime_s",
    "cpu_h_used",
    "cpu_h_reserved",
    "gb_h_reserved",
    "pcpu_weighted",
    "peak_rss_gb",
    "peak_vmem_gb",
    "read_gb",
    "write_gb",
)
FAILED = ("FAILED", "ABORTED")
_TIME = "%Y-%m-%d %H:%M:%S.%f"


# ── the plan ──────────────────────────────────────────────────────────────────
def params(backend) -> dict:
    """The params file of one run: the pins, the ruling and the backend."""
    if backend not in BACKENDS:
        raise ValueError(f"backend {backend!r}: one of {', '.join(BACKENDS)}")
    return {**PINNED, **RULED, "registration_method": backend}


def run_name(backend, patient) -> str:
    """``rb-compute-<backend>-<patient>`` in Nextflow's run-name alphabet: lower case,
    anything else a single ``-``. The head job adds the submission and its job id."""
    name = f"rb-compute-{backend}-{patient}".lower()
    name = re.sub(r"[^a-z0-9_-]+", "-", name)
    return re.sub(r"-{2,}", "-", name).strip("-")


def _read_sheet(samplesheet):
    """``(header line, [(patient, line, row dict), ...])``: the samplesheet's lines as
    they are (a carriage return taken off), each with its patient. The patient is the
    ``patient_id`` column, the column the pipeline reads (on the cohort's sheets the
    first one, which is what run_arms.sh's ``filtered_sheet`` compared)."""
    text = Path(samplesheet).read_text().replace("\r", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        raise ValueError(f"{samplesheet}: empty")
    header = lines[0].lstrip("﻿")
    names = next(csv.reader([header]))
    if SHEET_PATIENT not in names:
        raise ValueError(f"{samplesheet}: no {SHEET_PATIENT} column in {names}")
    out = []
    for line in lines[1:]:
        row = dict(zip(names, next(csv.reader([line]))))
        patient = (row.get(SHEET_PATIENT) or "").strip()
        if patient:
            out.append((patient, line, row))
    return header, out


def patients(samplesheet) -> list[str]:
    """The samplesheet's patients, in its order."""
    return list(dict.fromkeys(p for p, _, _ in _read_sheet(samplesheet)[1]))


def plan(samplesheet, out) -> list[Run]:
    """One run per (backend, patient): every ``valis`` patient, then every ``tiled``
    one, each in a launch directory of its own under ``<out>/compute``."""
    names = {}
    runs = []
    found = patients(samplesheet)
    for patient in found:
        if patient in (".", "..") or "/" in patient or patient.startswith("."):
            raise ValueError(f"patient {patient!r}: not usable as a folder name")
    for backend in BACKENDS:
        for patient in found:
            name = run_name(backend, patient)
            if name in names:
                raise ValueError(
                    f"patients {names[name]!r} and {patient!r} give the same run name "
                    f"{name!r}: rename one in the samplesheet"
                )
            names[name] = patient
            d = Path(out) / "compute" / backend / patient
            runs.append(
                Run(
                    backend,
                    patient,
                    str(d),
                    str(d / "samplesheet.csv"),
                    str(d / "results"),
                    str(d / "trace"),
                    str(d / "work"),
                    str(d / "params.json"),
                    name,
                )
            )
    return runs


def _size(path):
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return None


def sheet(samplesheet, patient, out_csv, sizes=None) -> int:
    """Write the header and that patient's rows to ``out_csv``; return how many rows.
    With ``sizes``: write there the sum of the patient's slide files (``path_to_file``)
    in GB (2**30 bytes), ``nan`` when one is not there or the column is missing."""
    header, rows = _read_sheet(samplesheet)
    mine = [(line, row) for p, line, row in rows if p == patient]
    if not mine:
        raise ValueError(f"{samplesheet}: no row of patient {patient!r}")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_csv).write_text("\n".join([header, *(line for line, _ in mine)]) + "\n")
    if sizes is not None:
        total = 0
        for _, row in mine:
            got = _size(row.get(SHEET_SLIDE) or "")
            if got is None:
                total = None
                break
            total += got
        Path(sizes).write_text(("nan" if total is None else repr(total / 2**30)) + "\n")
    return len(mine)


# ── the rows ──────────────────────────────────────────────────────────────────
class Rows(list):
    """``pipeline_cost.csv``'s rows, and beside them what is in no column: per
    (backend, patient) what the run left beside its trace (``runs``: ``ended_ok``,
    ``input_gb``), and per (backend, patient, task_id) the task's name (``names``),
    which tells a retried task (Nextflow keeps the name, the retry gets a later id)."""

    runs: dict
    names: dict


def _ended_ok(out, backend, patient):
    end = Path(out) / "logs" / "compute" / backend / f"{patient}.end"
    try:
        text = end.read_text()
    except OSError:
        return False
    return any(
        line.startswith("REGBENCH_END kind=compute") and " status=ok" in line
        for line in text.splitlines()
    )


def _input_gb(d):
    try:
        return float((Path(d) / "input_gb.txt").read_text().strip())
    except (OSError, ValueError):
        return math.nan


def _mul(*values):
    out = 1.0
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return math.nan
        out *= v
    return out


def rows(out) -> Rows:
    """One row per task of every ``<out>/compute/<backend>/<patient>/trace/trace.txt``
    (a folder moved aside to ``compute/.replaced`` is not read)."""
    got = Rows()
    got.runs, got.names = {}, {}
    for backend in BACKENDS:
        root = Path(out) / "compute" / backend
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            path = d / "trace" / "trace.txt"
            if not path.is_file():
                continue
            patient = d.name
            got.runs[(backend, patient)] = {
                "ended_ok": _ended_ok(out, backend, patient),
                "input_gb": _input_gb(d),
            }
            for t in trace.parse(path):
                process = t["process"].split(":")[-1]
                hours = _mul(t["realtime_s"], 1 / 3600)
                got.append(
                    {
                        "backend": backend,
                        "patient": patient,
                        "process": process,
                        "process_path": t["process"],
                        "task_id": t["task_id"],
                        "tag": t["tag"],
                        "status": t["status"],
                        "exit": t["exit"],
                        "cpus": t["cpus"],
                        "memory_gb": t["memory_gb"],
                        "realtime_s": t["realtime_s"],
                        "duration_s": t["duration_s"],
                        "pcpu": t["pcpu"],
                        "cpu_h_used": _mul(t["pcpu"], 1 / 100, hours),
                        "cpu_h_reserved": _mul(t["cpus"], hours),
                        "gb_h_reserved": _mul(t["memory_gb"], hours),
                        "peak_rss_gb": t["peak_rss_gb"],
                        "peak_vmem_gb": t["peak_vmem_gb"],
                        "read_gb": t["read_gb"],
                        "write_gb": t["write_gb"],
                        "submit": t["submit"],
                        "start": t["start"],
                        "complete": t["complete"],
                        "is_registration": int(
                            process in REGISTRATION_PROCESSES[backend]
                        ),
                    }
                )
                got.names[(backend, patient, t["task_id"])] = t["name"]
    return got


def _num(v):
    return v if isinstance(v, (int, float)) and not math.isnan(v) else None


def _sum(values):
    got = [x for x in (_num(v) for v in values) if x is not None]
    return float(sum(got)) if got else math.nan


def _max(values):
    got = [x for x in (_num(v) for v in values) if x is not None]
    return float(max(got)) if got else math.nan


def _ts(text):
    try:
        return datetime.strptime(text, _TIME)
    except (TypeError, ValueError):
        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            return None


def _wall(task_rows):
    """``max(complete) - min(submit)`` in seconds; NaN without both."""
    submits = [s for s in (_ts(r["submit"]) for r in task_rows) if s]
    completes = [c for c in (_ts(r["complete"]) for r in task_rows) if c]
    if not submits or not completes:
        return math.nan
    return (max(completes) - min(submits)).total_seconds()


def _ratio(a, b):
    return a / b if _num(a) is not None and _num(b) else math.nan


def _retried(task_rows, names):
    """How many failed tasks were run again and completed: Nextflow keeps a retried
    task's name and gives the new attempt a later task id. Without the names (rows
    not from ``rows``), a task is its process and tag."""

    def key(r):
        got = names.get((r["backend"], r["patient"], r["task_id"]))
        return got if got else (r["process_path"], r["tag"])

    def tid(r):
        return r["task_id"] if _num(r["task_id"]) is not None else -1

    done = {}
    for r in task_rows:
        if r["status"] == "COMPLETED":
            done[key(r)] = max(done.get(key(r), -1), tid(r))
    return sum(
        1 for r in task_rows if r["status"] in FAILED and done.get(key(r), -1) > tid(r)
    )


def _patient_row(backend, patient, task_rows, run, names):
    reg = [r for r in task_rows if r["is_registration"] == 1]
    n_failed = sum(r["status"] in FAILED for r in task_rows)
    n_retried = _retried(task_rows, names)
    n_cached = sum(r["status"] == "CACHED" for r in task_rows)
    used = _sum(r["cpu_h_used"] for r in task_rows)
    reg_used = _sum(r["cpu_h_used"] for r in reg)
    return {
        "backend": backend,
        "patient": patient,
        "n_patients": 1,
        "n_tasks": len(task_rows),
        "n_completed": sum(r["status"] == "COMPLETED" for r in task_rows),
        "n_failed": n_failed,
        "n_retried": n_retried,
        "n_cached": n_cached,
        # an unrecovered failure, a cached task or no ok end line: not a whole run
        "complete_run": int(
            n_failed == n_retried and n_cached == 0 and bool(run.get("ended_ok"))
        ),
        "wall_s": _wall(task_rows),
        "cpu_h_used": used,
        "cpu_h_reserved": _sum(r["cpu_h_reserved"] for r in task_rows),
        "gb_h_reserved": _sum(r["gb_h_reserved"] for r in task_rows),
        "peak_rss_gb": _max(r["peak_rss_gb"] for r in task_rows),
        "registration_wall_s": _wall(reg),
        "registration_cpu_h_used": reg_used,
        "registration_cpu_h_reserved": _sum(r["cpu_h_reserved"] for r in reg),
        "registration_share_cpu_h_used": _ratio(reg_used, used),
        "input_gb": run.get("input_gb", math.nan),
    }


SUMMED = (
    "n_tasks",
    "n_completed",
    "n_failed",
    "n_retried",
    "n_cached",
    "wall_s",
    "cpu_h_used",
    "cpu_h_reserved",
    "gb_h_reserved",
    "registration_wall_s",
    "registration_cpu_h_used",
    "registration_cpu_h_reserved",
    "input_gb",
)


def summarise(task_rows, runs=None) -> list[dict]:
    """``pipeline_cost_summary.csv``: per (backend, patient), then per backend the
    ``all`` row over the patients whose run is complete (``complete_run == 1``): sums,
    the largest peak, the share from the sums. Its ``complete_run`` is 1 when every
    patient listed is in it."""
    runs = getattr(task_rows, "runs", {}) if runs is None else runs
    names = getattr(task_rows, "names", {})
    out = []
    for backend in BACKENDS:
        keys = [k for k in runs if k[0] == backend]
        keys += sorted(
            {(r["backend"], r["patient"]) for r in task_rows if r["backend"] == backend}
            - set(keys)
        )
        per = []
        for key in keys:
            mine = [r for r in task_rows if (r["backend"], r["patient"]) == key]
            per.append(_patient_row(backend, key[1], mine, runs.get(key, {}), names))
        if not per:
            continue
        out += per
        whole = [p for p in per if p["complete_run"] == 1]
        total = {"backend": backend, "patient": "all", "n_patients": len(whole)}
        for col in SUMMED:
            total[col] = _sum(p[col] for p in whole) if whole else math.nan
        for col in ("n_tasks", "n_completed", "n_failed", "n_retried", "n_cached"):
            total[col] = int(total[col]) if whole else 0
        total["complete_run"] = int(len(whole) == len(per))
        total["peak_rss_gb"] = _max(p["peak_rss_gb"] for p in whole)
        total["registration_share_cpu_h_used"] = _ratio(
            total["registration_cpu_h_used"], total["cpu_h_used"]
        )
        out.append({c: total.get(c, math.nan) for c in SUMMARY_COLUMNS})
    return out


def by_process(task_rows, summary) -> list[dict]:
    """``pipeline_cost_by_process.csv``: per (backend, process), the tasks of the
    patients in the backend's ``all`` row. ``pcpu_weighted`` is Nextflow's ``%cpu``
    weighted by run time (``100 * cpu_h_used / realtime hours``)."""
    whole = {
        (s["backend"], s["patient"])
        for s in summary
        if s["patient"] != "all" and s["complete_run"] == 1
    }
    out = []
    for backend in BACKENDS:
        mine = [
            r
            for r in task_rows
            if r["backend"] == backend and (backend, r["patient"]) in whole
        ]
        for process in dict.fromkeys(r["process"] for r in mine):
            these = [r for r in mine if r["process"] == process]
            real = _sum(r["realtime_s"] for r in these)
            used = _sum(r["cpu_h_used"] for r in these)
            out.append(
                {
                    "backend": backend,
                    "process": process,
                    "is_registration": these[0]["is_registration"],
                    "n_patients": len({r["patient"] for r in these}),
                    "n_tasks": len(these),
                    "n_failed": sum(r["status"] in FAILED for r in these),
                    "realtime_s": real,
                    "cpu_h_used": used,
                    "cpu_h_reserved": _sum(r["cpu_h_reserved"] for r in these),
                    "gb_h_reserved": _sum(r["gb_h_reserved"] for r in these),
                    "pcpu_weighted": _ratio(100 * used, real / 3600)
                    if _num(real)
                    else math.nan,
                    "peak_rss_gb": _max(r["peak_rss_gb"] for r in these),
                    "peak_vmem_gb": _max(r["peak_vmem_gb"] for r in these),
                    "read_gb": _sum(r["read_gb"] for r in these),
                    "write_gb": _sum(r["write_gb"] for r in these),
                }
            )
    return out


# ── summary.md ────────────────────────────────────────────────────────────────
def _fmt(v, digits=3):
    if isinstance(v, float):
        return "-" if math.isnan(v) else f"{v:.{digits}g}"
    return str(v)


def _table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in body]
    return lines


def markdown(task_rows, summary) -> list[str]:
    """The ``## Full pipeline cost, per backend (secondary)`` block of ``summary.md``."""
    parts = ["## Full pipeline cost, per backend (secondary)", ""]
    by = by_process(task_rows, summary)
    for backend in BACKENDS:
        mine = [s for s in summary if s["backend"] == backend]
        if not mine:
            continue
        parts += [f"### `registration_method={backend}`", ""]
        body = []
        for s in mine:
            body.append(
                (
                    s["patient"]
                    if s["patient"] != "all"
                    else f"all ({s['n_patients']} patient(s))",
                    _ratio(s["wall_s"], 3600),
                    s["cpu_h_used"],
                    s["cpu_h_reserved"],
                    s["gb_h_reserved"],
                    s["peak_rss_gb"],
                    s["registration_share_cpu_h_used"],
                    "yes" if s["complete_run"] == 1 else "no",
                    f"{s['n_failed']}/{s['n_cached']}",
                )
            )
        parts += _table(
            (
                "patient",
                "wall h",
                "CPU-h used",
                "CPU-h reserved",
                "GB-h reserved",
                "peak GB",
                "registration share (CPU-h used)",
                "complete",
                "failed/cached tasks",
            ),
            body,
        )
        procs = [b for b in by if b["backend"] == backend]
        if procs:
            parts += [
                "",
                f"Per process, over the patients of the `all` row ({backend}):",
                "",
            ]
            parts += _table(
                ("process", "registration", "tasks", "CPU-h used", "peak GB"),
                [
                    (
                        b["process"],
                        "yes" if b["is_registration"] else "",
                        b["n_tasks"],
                        b["cpu_h_used"],
                        b["peak_rss_gb"],
                    )
                    for b in procs
                ],
            )
        parts.append("")
    parts += [
        "These are Nextflow tasks at the pipeline's own allocations, read from its trace "
        "(`pipeline_cost.csv`, one row per task; `pipeline_cost_summary.csv`; "
        "`pipeline_cost_by_process.csv`): a different clock from regbench's per-method "
        "registration cost above, measured another way (`%cpu` of Nextflow's task "
        "wrapper, not the cgroup counter), so no total adds the two. CPU-h used is "
        "`%cpu/100 x realtime`, CPU-h reserved `cpus x realtime`, GB-h reserved "
        "`memory x realtime`; the registration share is that of the registration "
        f"processes ({', '.join(f'{b}: ' + '/'.join(p) for b, p in REGISTRATION_PROCESSES.items())}). "
        "A failed task was paid for: it counts in its patient's row. A patient with a "
        "failed task that was not retried to success, a cached task, or no `status=ok` "
        "end line (`logs/compute/<backend>/<patient>.end`) is listed but kept out of "
        "the `all` row. Each run was alone on its lane, after the benchmark's own jobs "
        "(`submit_cohort.sh --compute-all`); `seg_method=stardist` (ruling 2026-10-15), "
        "everything else at the pipeline's shipped defaults and the site profile.",
        "",
    ]
    return parts
