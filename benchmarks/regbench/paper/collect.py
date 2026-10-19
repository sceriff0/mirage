"""The interim collector: the results that already exist (the retired arms, the retired
synthetic sweep, and regbench folders a scorer is still filling) in the columns of the paper
tables, and ``NA`` for every measurement still to come.

    python -m benchmarks.regbench collect --to DIR [--arms ROOT] [--sweep ROOT
        --sweep-plan CSV] [--regbench FOLDER ...] [--cases CASES]

Writes ``<to>/<table>.csv`` for EVERY table of ``schema.TABLES`` (a header even with no row)
and ``<to>/README.txt``. **Interim only**: these are not regbench rows, ``score`` never
reads them, and ``paper`` (``tables.py``) builds the paper's tables from the scorer's own
output, never from here. ``to`` may not be, or lie inside, any folder read (a regbench
results folder, its ``tables/`` or ``paper/``, the arms root or the sweep root): a
collection is never written into a result.

**Expected rows.** With ``cases``, ``per_pair`` holds one row per (cohort case, terminal
row of the CURRENT run list (``tables.terminal_rows``), set ``outlines``), ``NA`` but the
keys and ``case_id``, ``source = expected``: the final table's shape is visible now. A row
filled from any source replaces the expected row of its key.

**Dice is copied as found.** A NaN (no matched pair) is written ``nan``, distinct from
``NA`` (no value). Counting a NaN Dice as 0 (``tables.py``'s rule) happens when the paper
tables are BUILT, not here.

**Arms** (``<arms>/<arm>/<patient>/qc/registration/*_seg_qc.json``): the Dice of the
``full_transform`` record, else of the most-registered stage (``STAGE_RANK``); the panel is
the ``channels`` of ``<arm>/csv/registered.csv`` with ``|`` -> ``_``; the row by
``arm_row``. One difference from the ported ``harvest``: a ``full_transform`` record whose
Dice is NaN is KEPT (NaN) rather than skipped for a lower stage, which would credit the arm
with a stage it did not ship. Cost: ``cost_points`` per traced task of ``REGISTER`` (VALIS
arms, one per patient) or ``TILED_*`` (tiled arms, aggregated per patient: T31 P14,
``_tiled_patient_points``), completed tasks only, with ``input_gb`` the per-process mean of
``size_logs/input_sizes.csv`` where present (a VALIS arm's).

**Sweep** (``<sweep>/<run_id>/trace/trace.txt``, the run plan's ``run_id``): every completed
task as a ``cost_points`` row of ``dataset = synthetic_sweep``, ``row = run_id``. The
synthetic sweep is removed from the paper; README.txt says so.

**Interim regbench** (each ``--regbench`` folder): ``tables/cells.csv`` (and
``cells_nuclei.csv``) when present; otherwise the finished case records
``cache/scores/<dataset>/<case>/case.rec`` (a sha256 line, then a pickle of ``{"key",
"payload"}``; unpickled ONLY after the sha256 matches, a record that does not match is
skipped with a note; the payload is ``score._score_case``'s tuple: ``[0]`` the outlines
rows, ``[4]["cells"]`` the nuclei rows). Off-seam rows (``<row>:non_<owner>``) are left out,
as ``tables.per_pair`` leaves them; the full pairings of Task 21b (``tables.FULL_PAIR_COLUMNS``,
``full_lsa_status``) are copied with the rest, from the table or from the record (a case
record's rows carry them: ``score._full_records``). Costs from every ``<row>/<dataset>/<case>/run.json``
that registered, one launch counted once per row (by ``launch_id`` and ``attempt``).

**crosscheck_arms**: every (patient, panel, row) present from both an arm and a regbench
folder (outlines), both Dice and their difference (regbench - arm).

Ported, standard library only (the imports from this package are the schema, the run list
and the trace reader):

* ``STAGE_RANK``, ``stem``, ``panel_names`` and the stage rule of ``harvest``:
  ``774e1c9d:benchmarks/arms_dice_stare_vs_valis.py``.
* ``parse_trace`` (``2413456d:benchmarks/analysis/lib/load.py``) was already ported to the
  standard library as ``benchmarks/regbench/trace.py::parse``, which is used here;
  ``parse_size_logs`` (per-process MEAN of the task bytes, 2^30) and ``only_successful``
  (status COMPLETED/CACHED, exit 0) of the same file are ported below.
* The case-record and ``run.json`` readings follow ``regbench_peek_dice.py`` and
  ``regbench_peek_cost.py`` (scripts outside the repository), checked against
  ``score._write_record`` / ``score._read_record`` and ``score.resource_row``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import re
import shutil
from pathlib import Path

from .. import trace, variants
from . import costs, schema, tables

COHORT = "multiplex"
SWEEP_DATASET = "synthetic_sweep"
OUTLINES, NUCLEI = "outlines", "nuclei"
EXPECTED = "expected"

STAGE_RANK = {"native": 0, "rigid": 1, "non_rigid": 2, "refined": 2, "micro": 3}
EXTS = (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".ome.zarr", ".zarr")
FULL = "full_transform"
# an arm stage in the per_pair `stage` vocabulary (full_transform: the row's own stage)
ARM_STAGE = {
    "native": "initial",
    "rigid": "rigid",
    "non_rigid": "nonrigid",
    "refined": "nonrigid",
    "micro": "micro",
}
# the same-named numbers of an arm's stage record that per_pair has a column for
ARM_COPIED = ("iou_p50", "pair_fraction", "displacement_um_p50", "displacement_um_p90", "n_pairs")
NOT_RESULTS = {"logs", "cache", "tables", "paper"}
OK_STATUS = {"COMPLETED", "CACHED"}
README = "README.txt"


# ── values ───────────────────────────────────────────────────────────────────
def _as_found(v):
    """A value as its source held it. A float NaN is written ``nan`` (``schema.write``
    would write it ``NA``, the mark of a value that is not there)."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return v


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _nonempty(v):
    return v not in (None, "", schema.NA)


# ── the arm -> regbench row table ────────────────────────────────────────────
_TILED = re.compile(r"^tiled_(high|medium|low)_s(64|128|256)$")
_VALIS = re.compile(r"^valis_(high|medium|low)_micro([0-2])$")


def arm_row(arm):
    """The regbench row an arm's numbers stand for, or None (no such row)."""
    m = _TILED.match(arm)
    if m:
        tier, stride = m.groups()
        if stride == "64":
            return "stare" if tier == "high" else f"stare_{tier}"
        return f"stare_s{stride}_none" if tier == "high" else None
    m = _VALIS.match(arm)
    if m:
        tier, k = m.groups()
        return "valis_mirage" if (tier, k) == ("high", "2") else f"valis_mirage_{tier}_m{k}"
    return None


def _family(row):
    return variants.family(row) if row else None


# ── arms (port of arms_dice_stare_vs_valis.py) ───────────────────────────────
def stem(name):
    name = Path(str(name or "")).name
    for e in EXTS:
        if name.lower().endswith(e):
            return name[: -len(e)]
    return name


def panel_names(arm_dir):
    """Every name a moving slide may carry in this arm -> its channel set (``A_B_C``)."""
    names = {}
    p = Path(arm_dir) / "csv" / "registered.csv"
    if not p.is_file():
        return names
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r.get("channels") or "").replace("|", "_")
            if not key:
                continue
            pid = (r.get("patient_id") or "").strip()
            cands = {r.get("id"), key, f"{pid}_{key}"}
            cands |= {stem(v) for v in r.values() if v and ("/" in v or "." in v)}
            for c in cands:
                if c:
                    names[(pid, stem(c))] = key
    return names


def _headline(d):
    """``(stage, record)`` of a seg_qc JSON: the full_transform record, else the
    most-registered stage that has a Dice; None when there is none."""
    full = d.get(FULL)
    if isinstance(full, dict) and "dice_matched" in full:
        return FULL, full
    stages = d.get("stages") if isinstance(d.get("stages"), dict) else {}
    order = [
        s
        for s in (d.get("stage_order") or list(stages))
        if isinstance(stages.get(s), dict) and "dice_matched" in stages[s]
    ]
    if not order:
        return None
    stage = max(order, key=lambda s: STAGE_RANK.get(s, -1))
    return stage, stages[stage]


def harvest(arm_dir, notes):
    """``{(patient, panel): {...}}`` of one arm; what could not be read goes to ``notes``."""
    arm_dir = Path(arm_dir)
    arm = arm_dir.name
    names = panel_names(arm_dir)
    if not names:
        notes.append(f"arms:{arm}: no readable csv/registered.csv, panels named by file stem")
    found = sorted(arm_dir.glob("*/qc/registration/*_seg_qc.json"))
    if not found:
        notes.append(f"arms:{arm}: no <patient>/qc/registration/*_seg_qc.json, no Dice read")
    out = {}
    for j in found:
        try:
            d = json.loads(j.read_text())
        except (OSError, ValueError) as exc:
            notes.append(f"arms:{arm}: {j}: unreadable ({type(exc).__name__}), skipped")
            continue
        if not isinstance(d, dict):
            notes.append(f"arms:{arm}: {j}: not a JSON object, skipped")
            continue
        got = _headline(d)
        if got is None:
            notes.append(f"arms:{arm}: {j}: no stage with a dice_matched, skipped")
            continue
        stage, rec = got
        pid = str(d.get("patient_id") or j.parts[-4])
        mv = stem(d.get("moving"))
        panel = names.get((pid, mv)) or names.get((pid, mv.removeprefix(f"{pid}_")))
        if panel is None:
            panel = mv
            notes.append(f"arms:{arm}: {j}: {mv!r} not in csv/registered.csv, panel = the stem")
        if (pid, panel) in out:
            notes.append(
                f"arms:{arm}: {pid}:{panel} scored twice ({out[(pid, panel)]['moving']}, "
                f"{mv}); the first kept"
            )
            continue
        out[(pid, panel)] = {"stage": stage, "record": rec, "moving": mv}
    return out


def _arm_pair_rows(arm, harvested):
    row = arm_row(arm)
    rows = []
    for (pid, panel), h in sorted(harvested.items()):
        rec, stage = h["record"], h["stage"]
        if stage == FULL:
            pp_stage = tables.stage_of(row) if row else "final"
        else:
            pp_stage = ARM_STAGE.get(stage, stage)
        rows.append(
            schema.empty_row(
                "per_pair",
                source=f"arms:{arm}",
                dataset=COHORT,
                patient_id=pid,
                panel=panel,
                row=row,
                family=_family(row),
                set=OUTLINES,
                stage=pp_stage,
                dice_matched=_as_found(rec.get("dice_matched")),
                **{k: _as_found(rec.get(k)) for k in ARM_COPIED if k in rec},
            )
        )
    return rows


# ── traces (parse_size_logs and only_successful of load.py) ──────────────────
def _simple(process):
    return str(process).rsplit(":", 1)[-1]


def size_logs(path):
    """``{process: mean input GB over its tasks}`` of one ``input_sizes.csv`` (2^30)."""
    sums = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            b = _finite(r.get("bytes"))
            if b is None:
                continue
            s = sums.setdefault(_simple(r.get("process", "")), [0.0, 0])
            s[0] += b
            s[1] += 1
    return {p: total / n / 2**30 for p, (total, n) in sums.items() if n}


def successful(tasks):
    """The tasks that completed: status COMPLETED or CACHED (or none recorded), exit 0
    (or none recorded). A failed task's peak and time are partial."""
    out = []
    for t in tasks:
        if t.get("status") and t["status"] not in OK_STATUS:
            continue
        ex = t.get("exit")
        if isinstance(ex, int) and ex != 0:
            continue
        out.append(t)
    return out


def _task_points(tasks, sizes, keep, **ident):
    rows, seen = [], set()
    for t in successful(tasks):
        proc = _simple(t["process"])
        if not keep(proc):
            continue
        unit = f"{proc}:{t.get('tag') or t.get('name') or t.get('task_id')}"
        if unit in seen:
            continue
        seen.add(unit)
        wall, pcpu = _finite(t.get("realtime_s")), _finite(t.get("pcpu"))
        rows.append(
            schema.empty_row(
                "cost_points",
                unit_kind="patient" if proc == "REGISTER" else "task",
                unit_id=unit,
                input_gb=sizes.get(proc),
                wall_s=wall,
                peak_rss_gb=_finite(t.get("peak_rss_gb")),
                cpu_h_used=wall * pcpu / 100.0 / 3600.0
                if wall is not None and pcpu is not None
                else None,
                **ident,
            )
        )
    return rows


def _tiled_patient_points(tasks, **ident):
    """A tiled arm's cost per PATIENT (T31 P14: the patient is the one unit): its
    completed ``TILED_*`` tasks, grouped by the patient of their tag (the modules tag
    ``<patient_id>:...``). ``cpu_h_used`` is their sum, blank when one task has no
    ``%cpu`` (``score._whole``'s rule); ``wall_s`` is the critical path with every task
    at once and no queue (``clock = critical_path_unlimited``: per process its longest
    task, summed over the processes); ``peak_rss_gb`` the largest task's (the memory
    one job needs). ``input_gb`` is blank: the size log is per process, not per
    patient."""
    by = {}
    for t in successful(tasks):
        proc = _simple(t["process"])
        if not proc.startswith("TILED_"):
            continue
        tag = str(t.get("tag") or t.get("name") or "")
        patient = tag.split(":", 1)[0].strip()
        if patient:
            by.setdefault(patient, []).append((proc, t))
    rows = []
    for patient, mine in sorted(by.items()):
        walls = [_finite(t.get("realtime_s")) for _, t in mine]
        cpus = []
        for _, t in mine:
            wall, pcpu = _finite(t.get("realtime_s")), _finite(t.get("pcpu"))
            cpus.append(wall * pcpu / 100.0 / 3600.0 if wall is not None and pcpu is not None else None)
        longest = {}
        for (proc, _), w in zip(mine, walls):
            longest.setdefault(proc, []).append(w)
        whole = all(w is not None for w in walls)
        peaks = [x for x in (_finite(t.get("peak_rss_gb")) for _, t in mine) if x is not None]
        rows.append(
            schema.empty_row(
                "cost_points",
                unit_kind="patient",
                unit="patient",
                unit_id=patient,
                status="complete",
                clock=costs.CRITICAL,
                wall_s=sum(max(ws) for ws in longest.values()) if whole else None,
                peak_rss_gb=max(peaks) if peaks else None,
                cpu_h_used=sum(cpus) if all(c is not None for c in cpus) else None,
                lower_bound="" if len(peaks) == len(mine) else "a task without a peak",
                **ident,
            )
        )
    return rows


def _sizes_at(*candidates):
    for p in candidates:
        if p.is_file():
            return size_logs(p)
    return {}


def _read_trace(path, label, notes):
    if not path.is_file():
        notes.append(f"{label}: no {path}, no cost read")
        return None
    try:
        return trace.parse(path)
    except (OSError, ValueError, csv.Error) as exc:
        notes.append(f"{label}: {path}: unreadable trace ({type(exc).__name__})")
        return None


def _arm_cost(arm_dir, notes):
    arm = arm_dir.name
    row = arm_row(arm)
    if arm.startswith("valis_"):
        keep = lambda p: p == "REGISTER"  # noqa: E731
    elif arm.startswith("tiled_"):
        keep = lambda p: p.startswith("TILED_")  # noqa: E731
    else:
        notes.append(f"arms:{arm}: neither a valis_ nor a tiled_ arm, no cost read")
        return []
    tasks = _read_trace(arm_dir / "trace" / "trace.txt", f"arms:{arm}", notes)
    if tasks is None:
        return []
    ident = dict(source=f"arms:{arm}", dataset=COHORT, family=_family(row), row=row)
    if arm.startswith("tiled_"):  # T31 P14: per patient, like every other cost point
        return _tiled_patient_points(tasks, **ident)
    sizes = _sizes_at(arm_dir / "size_logs" / "input_sizes.csv")
    return _task_points(tasks, sizes, keep, **ident)


def collect_arms(root, notes):
    """``(per_pair rows, cost_points rows, {(patient, panel, row): arm values})``."""
    pairs, costs, by_key = [], [], {}
    for arm_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if arm_dir.name.startswith("."):
            continue
        arm = arm_dir.name
        if arm_row(arm) is None:
            notes.append(f"arms:{arm}: no regbench row for this arm (arm_row): its rows have row NA")
        harvested = harvest(arm_dir, notes)
        rows = _arm_pair_rows(arm, harvested)
        pairs += rows
        for r, ((pid, panel), h) in zip(rows, sorted(harvested.items())):
            if _nonempty(r["row"]):
                by_key.setdefault((pid, panel, r["row"]), (arm, h, r))
        costs += _arm_cost(arm_dir, notes)
    return pairs, costs, by_key


# ── the sweep ────────────────────────────────────────────────────────────────
def collect_sweep(root, plan, notes):
    root = Path(root)
    if plan is not None:
        with open(plan, newline="") as fh:
            run_ids = [r["run_id"] for r in csv.DictReader(fh) if r.get("run_id")]
    else:
        run_ids = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    rows = []
    for run_id in run_ids:
        tasks = _read_trace(root / run_id / "trace" / "trace.txt", f"sweep:{run_id}", notes)
        if tasks is None:
            continue
        sizes = _sizes_at(
            root / run_id / "out" / "size_logs" / "input_sizes.csv",
            root / run_id / "size_logs" / "input_sizes.csv",
        )
        rows += _task_points(
            tasks, sizes, lambda p: True, source=f"sweep:{run_id}", dataset=SWEEP_DATASET,
            row=run_id,
        )
    return rows


# ── interim regbench folders ─────────────────────────────────────────────────
def _pair_row(r, source, default_set):
    """A cell row of the scorer (a ``cells.csv`` line or a case record's dict) as a
    ``per_pair`` row, the columns of ``tables.per_pair``, values as found."""
    method = str(r.get("method") or "")
    g = lambda k: _as_found(r.get(k))  # noqa: E731
    return schema.empty_row(
        "per_pair",
        source=source,
        dataset=g("dataset"),
        patient_id=g("patient_id") if _nonempty(r.get("patient_id")) else g("group"),
        panel=g("panel"),
        case_id=g("case_id"),
        row=method or None,
        family=_family(method),
        set=g("set") if _nonempty(r.get("set")) else default_set,
        stage=tables.stage_of(method) if method else None,
        tool_version=g("tool_version"),
        frame=g("frame"),
        dice_matched=g("dice_matched"),
        iou_p50=g("iou_p50"),
        pair_fraction=g("pair_fraction"),
        pair_fraction_null_max=g("pair_fraction_null_max"),
        displacement_um_p50=g("displacement_um_p50"),
        displacement_um_p90=g("displacement_um_p90"),
        displacement_readable=g("displacement_readable"),
        within_1um=g("within_1um"),
        n_pairs=g("n_pairs"),
        n_ref=g("n_ref"),
        n_mov=g("n_moving"),
        imputed_initial=g("imputed_initial"),
        excluded_from_summaries=g("excluded_from_summaries"),
        cause=g("cause"),
        selfcheck_status=g("selfcheck_status"),
        # Task 21b: the full pairings and the fallback share, copied as found
        **{k: g(c) for k, c in tables.FULL_PAIR_COLUMNS.items()},
        full_lsa_status=g("full_lsa_status"),
    )


def _cell_rows(rows, source, default_set):
    return [
        _pair_row(r, source, default_set)
        for r in rows
        if isinstance(r, dict) and ":" not in str(r.get("method") or "")
    ]


def read_record(path):
    """The ``{"key", "payload"}`` of a case record, or ``(None, why)``. The pickle is
    loaded only after its sha256 line matches."""
    try:
        digest, _, body = Path(path).read_bytes().partition(b"\n")
    except OSError as exc:
        return None, f"unreadable ({type(exc).__name__})"
    if hashlib.sha256(body).hexdigest().encode() != digest:
        return None, "sha256 line does not match (cut short or damaged)"
    try:
        return pickle.loads(body), ""
    except Exception as exc:  # whatever a damaged pickle raises
        return None, f"unreadable pickle ({type(exc).__name__})"


def _record_rows(folder, source, notes):
    rows = []
    for rec in sorted(Path(folder).glob("cache/scores/*/*/case.rec")):
        held, why = read_record(rec)
        if held is None:
            notes.append(f"{source}: {rec}: {why}, skipped")
            continue
        payload = held.get("payload") if isinstance(held, dict) else None
        if not isinstance(payload, (tuple, list)) or not payload:
            notes.append(f"{source}: {rec}: not a case record's payload, skipped")
            continue
        rows += _cell_rows(payload[0] or [], source, OUTLINES)
        nuc = payload[4] if len(payload) > 4 and isinstance(payload[4], dict) else {}
        rows += _cell_rows(nuc.get("cells") or [], source, NUCLEI)
    return rows


def _table_rows(folder, source):
    rows = []
    for name, default_set in (("cells.csv", OUTLINES), ("cells_nuclei.csv", NUCLEI)):
        p = Path(folder) / "tables" / name
        if p.is_file():
            with open(p, newline="") as fh:
                rows += _cell_rows(list(csv.DictReader(fh)), source, default_set)
    return rows


def _device(info):
    d = str((info or {}).get("device") or "")
    if not d:
        return None
    return "gpu" if d.startswith("cuda") else "cpu"


def _run_cost(folder, source, notes):
    rows, seen = [], set()
    for p in sorted(Path(folder).glob("*/*/*/run.json")):
        top = p.parts[-4]
        if top in NOT_RESULTS or top.startswith(("partial-", ".")):
            continue
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError) as exc:
            notes.append(f"{source}: {p}: unreadable ({type(exc).__name__}), skipped")
            continue
        if not isinstance(r, dict) or not r.get("ok"):
            continue
        row = str(r.get("variant") or top)
        dataset = str(r.get("dataset") or p.parts[-3])
        launch = r.get("launch_id") or tuple(r.get("group_cases") or []) or str(p)
        key = (row, dataset, str(launch), r.get("attempt"))
        if key in seen:
            continue
        seen.add(key)
        res = r.get("resources") if isinstance(r.get("resources"), dict) else {}
        info = r.get("info") if isinstance(r.get("info"), dict) else {}
        jobs = info.get("jobs") if isinstance(info.get("jobs"), dict) else {}
        if jobs.get("execution") == "jobs":
            wall = _finite(jobs.get("wall_unlimited_parallel_s"))
            cpu_h = _finite(jobs.get("cpu_h_measured"))
        else:
            wall = _finite(r.get("register_s", res.get("wall_s")))
            cpu_s = _finite(res.get("cpu_s"))
            cpu_h = cpu_s / 3600.0 if cpu_s is not None else None
        rss = _finite(res.get("peak_rss_mb"))
        patient = r.get("unit") == "patient"
        rows.append(
            schema.empty_row(
                "cost_points",
                source=source,
                dataset=dataset,
                family=_family(row),
                row=row,
                unit_kind="patient" if patient else "round",
                unit_id=r.get("group") if patient else (r.get("case_id") or p.parts[-2]),
                wall_s=wall,
                peak_rss_gb=rss / 1024.0 if rss is not None else None,
                cpu_h_used=cpu_h,
                device=_device(info),
            )
        )
    return rows


def collect_regbench(folder, notes):
    """``(per_pair rows, cost_points rows)`` of one interim regbench folder."""
    source = f"regbench-interim:{folder}"
    has_table = (Path(folder) / "tables" / "cells.csv").is_file()
    pairs = _table_rows(folder, source) if has_table else _record_rows(folder, source, notes)
    if not pairs:
        notes.append(f"{source}: no tables/cells.csv and no finished case record")
    return pairs, _run_cost(folder, source, notes)


# ── expected rows and the crosscheck ─────────────────────────────────────────
def _cases(root):
    from ..cases import list_cases

    return list_cases(root, COHORT)


def _key(r):
    return (r["dataset"], r["patient_id"], r["panel"], r["row"], r["set"])


def expected_rows(cases_root):
    from ..score import panel_of

    rows = []
    terminal = list(dict.fromkeys(tables.terminal_rows(COHORT).values()))
    for case in _cases(cases_root):
        for row in terminal:
            rows.append(
                schema.empty_row(
                    "per_pair",
                    source=EXPECTED,
                    dataset=getattr(case, "dataset", COHORT),
                    patient_id=case.group,
                    panel=panel_of(case),
                    case_id=str(case.case_id),
                    row=row,
                    set=OUTLINES,
                )
            )
    return rows


def crosscheck(arm_values, regbench_rows):
    first = {}
    for r in regbench_rows:
        if r["set"] == OUTLINES and _nonempty(r["row"]):
            first.setdefault((r["patient_id"], r["panel"], r["row"]), r)
    rows = []
    for key in sorted(set(arm_values) & set(first)):
        arm, h, a = arm_values[key]
        rb = first[key]
        d_rb, d_arm = _finite(rb["dice_matched"]), _finite(a["dice_matched"])
        note = f"regbench from {rb['source']}"
        if d_rb is None or d_arm is None:
            note += "; a Dice is NaN or missing, no difference"
        rows.append(
            schema.empty_row(
                "crosscheck_arms",
                source=f"arms:{arm}",
                patient_id=key[0],
                panel=key[1],
                row=key[2],
                arm=arm,
                dice_regbench=rb["dice_matched"],
                dice_arm=a["dice_matched"],
                dice_diff=d_rb - d_arm if d_rb is not None and d_arm is not None else None,
                disp_um_regbench=rb["displacement_um_p50"],
                disp_um_arm=a["displacement_um_p50"],
                arm_stage=h["stage"],
                note=note,
            )
        )
    return rows


# ── the refusal and the run ──────────────────────────────────────────────────
def _refuse_inside(to, folders):
    to = Path(to).resolve()
    for f in folders:
        f = Path(f).resolve()
        if to == f or f in to.parents:
            raise SystemExit(
                f"collect: refusing --to {to}: it is, or lies inside, {f}, a folder this "
                "collection reads. An interim collection is never written into a result."
            )


def _readme(to, counts, inputs, notes):
    lines = [
        "interim only: not regbench rows, never read by `score`; every measurement still "
        "to come is NA",
        "",
        "Written by `python -m benchmarks.regbench collect` (benchmarks/regbench/paper/collect.py).",
        "Every table of the paper schema is here with exactly its columns, a header even "
        "with no row. `source` says where each row comes from: arms:<arm>, sweep:<run_id>, "
        "regbench-interim:<folder>, or expected (a per_pair row the current run list says "
        "should exist and no source filled).",
        "",
        "Dice is copied as found: NaN (written `nan`) means the scorer found no matched pair; "
        "NA means no value. The paper tables count a NaN Dice as 0 when they are BUILT "
        "(`paper`, tables.py); this collection does not.",
        "The synthetic sweep is removed from the paper: its rows (cost_points, dataset "
        f"{SWEEP_DATASET}) are here for reference only.",
        "synthetic/: the synthetic scaling set's own tables (synthetic_cost, _scaling, "
        "_extrapolation: cost only, never pooled with the cohort), copied as the scorer "
        "wrote them when a --regbench folder has tables/synthetic/.",
        "cost_points: unit_kind `task` is one Nextflow task of the sweep (every sweep "
        "process); an arm's point is one patient (REGISTER, or a tiled arm's TILED_* jobs "
        "summed per patient, wall = the critical path with every task at once). Regbench cost "
        "points have no input_gb / input_gpx here (the slide sizes come with the paper "
        "tables). Case-record rows have no selfcheck_status or excluded_from_summaries (the "
        "scorer sets those when it writes its tables).",
        "",
        "Read:",
        *[f"  {k}: {v}" for k, v in inputs.items()],
        "",
        "Rows per table:",
        *[f"  {name}: {n}" for name, n in counts.items()],
        "",
        "Notes:",
        *([f"  - {n}" for n in notes] or ["  (none)"]),
    ]
    (Path(to) / README).write_text("\n".join(lines) + "\n")


SYNTHETIC_DIR = "synthetic"


def copy_synthetic(to, regbench, notes):
    """The synthetic set's own tables (T32, cost only, never pooled), copied as they are
    into ``<to>/synthetic/``, where ``plots`` looks for them. From
    ``<out>/paper/synthetic/`` when it has them (``paper --dataset synthetic``, whose
    ``synthetic_extrapolation`` reads the cohort's ``paper/cost_points.csv``), else from
    the scorer's ``<out>/tables/synthetic/`` (final review I4: never by the order the
    folders were written in). A second results folder that has them goes into
    ``<to>/synthetic/<its name>/`` and is noted. The paths copied."""
    copied = []
    for folder in regbench:
        found = []
        for where in ("paper", "tables"):
            found = sorted((Path(folder) / where / SYNTHETIC_DIR).glob("synthetic_*.csv"))
            if found:
                break
        if not found:
            continue
        dest = Path(to) / SYNTHETIC_DIR
        if any(dest.glob("synthetic_*.csv")):
            dest = dest / Path(folder).name
            notes.append(f"{folder}: a second synthetic set, kept in {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        for f in found:
            shutil.copyfile(f, dest / f.name)
            copied.append(dest / f.name)
    return copied


def run(to, arms, sweep, sweep_plan, regbench, cases):
    """Write every table of ``schema.TABLES`` and README.txt under ``to``; ``{table: rows}``."""
    regbench = [Path(f) for f in (regbench or [])]
    _refuse_inside(to, [*regbench, *(p for p in (arms, sweep) if p is not None)])
    if sweep_plan is not None and sweep is None:
        raise SystemExit("collect: --sweep-plan needs --sweep (the folder the runs are under)")
    notes = []
    built = {name: [] for name in schema.TABLES}
    arm_values = {}
    if arms is not None:
        pairs, costs, arm_values = collect_arms(arms, notes)
        built["per_pair"] += pairs
        built["cost_points"] += costs
    if sweep is not None:
        built["cost_points"] += collect_sweep(sweep, sweep_plan, notes)
    regbench_pairs = []
    for folder in regbench:
        pairs, costs = collect_regbench(folder, notes)
        regbench_pairs += pairs
        built["cost_points"] += costs
    built["per_pair"] += regbench_pairs
    built["crosscheck_arms"] = crosscheck(arm_values, regbench_pairs)
    if cases is not None:
        filled = {_key(r) for r in built["per_pair"]}
        built["per_pair"] = [r for r in expected_rows(cases) if _key(r) not in filled] + built[
            "per_pair"
        ]
    to = Path(to)
    to.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in schema.TABLES:
        schema.write(to / f"{name}.csv", name, built[name])
        counts[name] = len(built[name])
    synthetic = copy_synthetic(to, regbench, notes)
    if synthetic:
        counts["synthetic (copied)"] = len(synthetic)
    inputs = {
        "arms": arms,
        "sweep": sweep,
        "sweep plan": sweep_plan,
        "regbench": ", ".join(map(str, regbench)) or None,
        "cases": cases,
    }
    _readme(to, counts, {k: v if v is not None else "-" for k, v in inputs.items()}, notes)
    return counts


def main(a):
    counts = run(
        Path(a.to),
        Path(a.arms) if a.arms else None,
        Path(a.sweep) if a.sweep else None,
        Path(a.sweep_plan) if a.sweep_plan else None,
        [Path(f) for f in (a.regbench or [])],
        Path(a.cases) if a.cases else None,
    )
    for name, n in counts.items():
        print(f"{name}\t{n}")
    return 0
