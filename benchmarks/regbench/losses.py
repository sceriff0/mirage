"""Every loss and every flag of a scoring, as ROWS of one table (``tables/losses.csv``).

USER 2026-10-18: "no blocking at all, just report in final csv / results" and "make sure
in outputs i see everything". So nothing that goes wrong with one unit (a case, a run, a
slide, a round of the manifest) stops the others, and every such thing is a row here,
never only a line of a log.

``losses.csv`` (``COLUMNS``, in this order)::

    stage,dataset,patient_id,case_id,run,method,cause,blame,effect,detail,source

``stage``   where it happened (``STAGES``): submit, manifest, prepare, nuclei, run,
            settle, score, full_pairs (the scorer's full LSA beside the headline,
            ``regbench.full_pairs``), paper
``blame``   whose it is (``BLAMES``): ``method`` (the package under test),
            ``infrastructure`` (the scheduler, the node, the launcher, the benchmark's
            own steps), ``input`` (the cohort's slides, manifest or nuclei) or
            ``integrity`` (a result that is not of the case as it is now)
``effect``  what became of the unit (``EFFECTS``): ``dropped`` (in no summary: the case,
            run or round is gone, or a round lost for a cause that is not the method's),
            ``imputed`` (scored at the unregistered pose, like any failure of the
            method), ``flagged`` (kept and scored, with a flag), ``capped`` (the memory
            ladder stopped at the partition's largest rung: imputed, and the cap is in
            ``detail``), ``not_run`` (nobody ran it)
``source``  the file or record that carries the evidence, relative to the results folder
            (or to the cases folder, for the prepare and nuclei stages)

``CAUSES`` is the one list of causes, each with its blame and its stage. ``score`` writes
the table for every dataset it scores (the cohort into ``tables/``, a public dataset into
``tables/<dataset>/``), a header alone when nothing was lost: an absent table is not
"none". It is the UNION of the per-stage tables, which still exist:

``submit_dropped.csv``    (results folder) the runs the launcher dropped: a failed
                          preflight, a configuration problem of that run
``manifest_dropped.csv``  (beside the manifest) the rounds the manifest could not list
``prepare_dropped.csv``   (cases folder) the rounds ``prepare multiplex`` dropped or
                          failed on
``nuclei_checks.csv``     (tables) one row per slide: its nuclei's agreement with its
                          outlines; a slide under the threshold is kept and flagged
``score_errors.csv``      (tables) a case whose scoring raised: the last frame and the
                          message
``logs/unmarked_units.csv``  the units the settle job could not list (S2)
``logs/refused_units.tsv``   the units of a refused or given-up selfcheck/stage step

and of the records of every result (``run.json``: a failure, a unit held in the queue, a
refused submission, a watcher that gave up, an unsettled kill).

Stdlib only.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

COLUMNS = (
    "stage",
    "dataset",
    "patient_id",
    "case_id",
    "run",
    "method",
    "cause",
    "blame",
    "effect",
    "detail",
    "source",
)
STAGES = (
    "submit",
    "manifest",
    "prepare",
    "nuclei",
    "run",
    "settle",
    "score",
    "full_pairs",
    "paper",
)
BLAMES = ("method", "infrastructure", "input", "integrity")
EFFECTS = ("dropped", "imputed", "flagged", "capped", "not_run")

METHOD, INFRA, INPUT, INTEGRITY = BLAMES
# cause -> (blame, stage, what it is). EVERY cause a loss row can carry, once.
CAUSES = {
    # the method's: scored at the unregistered pose (imputed)
    "exception": (METHOD, "run", "the method raised"),
    "killed": (
        METHOD,
        "run",
        "the process died and the scheduler's accounting names no cause",
    ),
    "oom": (METHOD, "run", "out of memory"),
    "jvm_heap": (
        METHOD,
        "run",
        "VALIS's JVM ran out of its fixed 10 GB heap (never a climb: the heap does not "
        "grow with the rung)",
    ),
    "gpu_oom": (METHOD, "run", "out of GPU memory"),
    "timeout": (METHOD, "run", "out of time"),
    "writer": (METHOD, "run", "the method's writer failed"),
    "writer_oom": (METHOD, "run", "the method's writer ran out of memory"),
    "capped": (
        METHOD,
        "run",
        "it needed more memory than the largest rung the partition grants: the "
        "ladder is the same for every method, so this is a cost result (imputed; the "
        "cap is in detail)",
    ),
    # not the method's (score.NOT_THE_METHOD): in no summary of that method
    "benchmark_oom": (INFRA, "run", "the benchmark's own point step ran out of memory"),
    "benchmark": (INFRA, "run", "the benchmark's own step failed"),
    "identical_overlap": (
        INFRA,
        "run",
        "pixel-identical tile overlaps, made by the benchmark's tiling",
    ),
    "infrastructure": (INFRA, "run", "the scheduler or the node lost the attempt"),
    "not_submitted": (INFRA, "run", "the launcher never submitted the unit"),
    "killed_unsettled": (
        INFRA,
        "settle",
        "killed, and never settled against the scheduler's accounting (no job id, "
        "or sacct had none): the cause is not known",
    ),
    "watcher_gave_up": (
        INFRA,
        "settle",
        "the chain's watcher could not read the queue and gave up before the unit's "
        "next step",
    ),
    "result_unreadable": (
        INFRA,
        "score",
        "the result's run.json or warped.npz could not be read, or an ok result has "
        "no warped.npz",
    ),
    "submit_refused": (
        INFRA,
        "submit",
        "sbatch refused the unit's next step after every retry",
    ),
    "held": (
        INFRA,
        "run",
        "its jobs were held or stalled in the queue (the squeue reason is in detail)",
    ),
    "score_error": (
        INFRA,
        "score",
        "the scoring of the case raised (score_errors.csv): the case has no row",
    ),
    "table_unreadable": (
        INFRA,
        "score",
        "a per-stage table is there and cannot be read: the losses it holds are not "
        "rows here (its path is in source)",
    ),
    # Task 21b (regbench.full_pairs): the full LSA beside the headline, never ranked;
    # the headline row stays, its full-LSA numbers are NA
    "full_lsa_not_certified": (
        INFRA,
        "full_pairs",
        "the full LSA (no fallback, no cap) of a pair record has fewer pairs than the "
        "candidates allow: its numbers are NA, the headline row is kept (flagged)",
    ),
    "full_lsa_failed": (
        INFRA,
        "full_pairs",
        "the full-LSA solve of a pair record raised (memory, anything; the exception is "
        "in detail): its numbers are NA, the headline row is kept (flagged) and the "
        "next scoring tries again",
    ),
    "full_lsa_killed": (
        INFRA,
        "full_pairs",
        "the process was killed during the full-LSA solve of a pair record (memory, "
        "walltime: it left its in-flight marker and no result): the solve is not tried "
        "again, its numbers are NA and the headline row is kept (flagged)",
    ),
    "tissue_missing": (
        INFRA,
        "score",
        "the tissue block could not score a written image (tissue_missing.csv)",
    ),
    "realcheck_unreadable": (
        INFRA,
        "score",
        "a realcheck.json (or the run.json beside it) cannot be read: the run's "
        "realcheck verdict is `unreadable` (verdicts.tsv)",
    ),
    "fit_points_dropped": (
        INFRA,
        "paper",
        "a cost fit left points out (no finite target or predictor, or no input size): "
        "n_dropped of cost_fits / pipeline_cost_fits (S12)",
    ),
    "paper_record_unreadable": (
        INFRA,
        "paper",
        "a run.json the paper tables could not read: n_unreadable of versions / "
        "run_parameters (S13)",
    ),
    # T31 rule 6: what a cost table left out of a median is a row, never a count alone
    "cost_figure_missing": (
        INFRA,
        "paper",
        "a common patient the row completed has no figure for a cost the row has for "
        "other patients: left out of that method_summary median (paper/losses.csv)",
    ),
    "cost_peak_not_comparable": (
        INFRA,
        "paper",
        "a common patient's peak is a rusage lower bound: left out of every peak median "
        "of method_summary (paper/losses.csv)",
    ),
    "cost_lower_bound": (
        INFRA,
        "paper",
        "a common patient's times and CPU-hours are lower bounds (cost_points.lower_bound): "
        "kept in method_summary's medians and flagged (paper/losses.csv)",
    ),
    # T31 rule 6: what a cost figure could not draw is a row, never a silent gap
    "plot_points_dropped": (
        INFRA,
        "paper",
        "a cost figure could not draw a patient's value (no figure, <= 0 on a log axis, "
        "a rusage peak in a peak comparison, a GPU row without a CPU twin on a CPU axis): "
        "paper/losses.csv",
    ),
    "plot_lower_bound": (
        INFRA,
        "paper",
        "a cost figure drew a patient's value that is a lower bound "
        "(cost_points.lower_bound): kept and flagged, paper/losses.csv",
    ),
    "plot_placeholder": (
        INFRA,
        "paper",
        "a cost figure is its placeholder: its input table is absent or all NA "
        "(paper/losses.csv)",
    ),
    "stare_local4_differs": (
        INTEGRITY,
        "paper",
        "stare_local4 (STARE's stages in one job of 4 CPUs) scored a case differently "
        "from stare: the cost row is not of the same registration (paper/losses.csv)",
    ),
    "not_run": (INFRA, "run", "no result of this submission: nobody ran it"),
    "stale_result": (
        INTEGRITY,
        "run",
        "a result made for an earlier version of the case or by another submission",
    ),
    # the launcher's drops of a whole run (submit_dropped.csv)
    "preflight_failed": (
        INFRA,
        "submit",
        "the run's image failed the preflight: the run was not submitted",
    ),
    "run_config": (
        INFRA,
        "submit",
        "a configuration problem of that run (its image, its GPU request, its stages): "
        "the run was not submitted",
    ),
    "unmarked_unit": (
        INFRA,
        "settle",
        "a chain the settle job could not list: its units were not checked for tasks "
        "still in flight",
    ),
    "no_trace": (
        INFRA,
        "run",
        "a compute-all patient with no Nextflow trace (pipeline_cost_summary: "
        "complete_run 0)",
    ),
    "compute_incomplete": (
        INFRA,
        "run",
        "a compute-all patient whose run is not whole (a failed task no retry "
        "recovered, a cached task, no ok end line): out of the backend's `all` row",
    ),
    # the inputs
    "manifest_dropped": (INPUT, "manifest", "the manifest could not list the round"),
    "prepare_dropped": (
        INPUT,
        "prepare",
        "prepare dropped the round (multiplex) or the pair (hyreco)",
    ),
    "prepare_failed": (
        INPUT,
        "prepare",
        "prepare failed on the round (multiplex) or the pair (hyreco)",
    ),
    "landmark_flagged": (
        INPUT,
        "prepare",
        "a landmark sanity check of a public landmark set failed (a landmark outside "
        "the image, a diagonal the table disagrees with, unequal landmark files, a "
        "pixel size off the published one): the pair is kept, scored and flagged",
    ),
    "frame_check_failed": (
        INPUT,
        "run",
        "a STARE proxy moved image content in prep's frame check: written, run and "
        "flagged (stages/prep/record.json frame_check=failed; B10, no blocking)",
    ),
    "low_agreement": (
        INPUT,
        "nuclei",
        "the slide's nuclei agree with its outlines under the threshold: kept, "
        "scored and flagged (nuclei_checks.csv)",
    ),
    "pixel_size_fallback": (
        INPUT,
        "manifest",
        "the checkpoint has no pixel size for the round: 0.325 um is used",
    ),
    # T32: the synthetic scaling set's slides grow to 65536 px a side
    "refused_size": (
        INPUT,
        "run",
        "the run's method declares a largest slide side (MAX_SIDE_PX) under this "
        "slide's: it refused the size by its own limit (not run; the synthetic set)",
    ),
}
# the causes added to `score.NOT_THE_METHOD` by Task 28 (billed as lost, except
# BILLED_AS_WASTED, R26); `capped` is the method's (ruling 2026-10-18) and is not here
NOT_THE_METHOD_EXTRA = (
    "killed_unsettled",
    "watcher_gave_up",
    "result_unreadable",
    "submit_refused",
    "held",
    "score_error",
)
# what the tile submitter writes on its give-up line when a task was held or stalled
# (slurm/feed_tiles.sbatch): it reaches the round's record through `stages.feed_problem`
HELD_MARK = "held (squeue reason"
# R26 (resources review): a cause that is not the method's for ACCURACY (the case is
# left out as infrastructure) but whose attempt is still BILLED as wasted: an OOM kill
# nobody settled would otherwise be billed to no one, favouring rows that ran without
# the OOM watcher. Billed by the scheduler's upper bound where the accounting has it,
# and the row says `cost_bound = upper` (score.bill_from_disk)
BILLED_AS_WASTED = ("killed_unsettled",)
# what stages.feed_problem (frozen) says of a tile submitter sbatch refused after its
# REGBENCH_STAGE_SUBMIT_TRIES: that unit's cause is `submit_refused` (B7)
REFUSED_MARK = "the tile submitter was refused by the scheduler"


def blame_of(cause):
    """The blame of a cause. KeyError for one ``CAUSES`` does not know: add it there."""
    return CAUSES[cause][0]


def stage_of(cause):
    return CAUSES[cause][1]


def _ended(state):
    state = str(state or "").strip().upper()
    not_ended = ("PENDING", "RUNNING", "RESIZING", "SUSPENDED", "COMPLETING")
    return bool(state) and not state.startswith(not_ended)


def reclassify(run):
    """A result's record as the tables read it: the causes the frozen writers cannot
    say themselves (``killed_unsettled``, ``held``). Returns ``run`` itself when nothing
    changes, else a copy with ``cause`` and a ``cause_from`` saying why.

    * ``killed`` whose task the scheduler's accounting never ended (no ``scheduler``
      state: no job id, or sacct had none) is ``killed_unsettled``: the cause is not
      known, so it is not the method's. A ``killed`` the accounting DID end with a state
      that names no cause (``COMPLETED``, ``FAILED``) stays the method's.
    * ``not_submitted`` whose record names a tile submitter that gave up on held or
      stalled tasks (``HELD_MARK``) is ``held``; the reason stays in the record.
    * ``not_submitted`` whose record names a tile submitter sbatch refused after its
      tries (``REFUSED_MARK``) is ``submit_refused``."""
    if not isinstance(run, dict) or not run or run.get("ok"):
        return run
    cause = run.get("cause")
    if cause == "killed":
        state = (run.get("scheduler") or {}).get("state")
        if not _ended(state):
            return {
                **run,
                "cause": "killed_unsettled",
                "cause_from": (
                    (run.get("cause_from") or "the driver's record said killed")
                    + "; never settled against the scheduler's accounting"
                ),
            }
    if cause == "not_submitted":
        said = " ".join(str(run.get(k) or "") for k in ("error", "cause_from"))
        if HELD_MARK in said:
            return {**run, "cause": "held"}
        if REFUSED_MARK in said:
            return {**run, "cause": "submit_refused"}
    return run


def row(**kw):
    """One loss row: every column, ``blame`` from the cause (``CAUSES``)."""
    out = dict.fromkeys(COLUMNS, "")
    out.update({k: ("" if v is None else v) for k, v in kw.items() if k in COLUMNS})
    out["blame"] = blame_of(out["cause"])
    if not out["stage"]:
        out["stage"] = stage_of(out["cause"])
    if out["effect"] not in EFFECTS:
        raise ValueError(f"effect {out['effect']!r}: one of {EFFECTS}")
    if out["stage"] not in STAGES:
        raise ValueError(f"stage {out['stage']!r}: one of {STAGES}")
    return out


def read_csv(path, bad=None, delimiter=","):
    """The rows of a CSV, ``[]`` when it is not there. One that is there and cannot be
    read is said, and its path is appended to ``bad`` (I5: the caller makes it a
    ``table_unreadable`` row, so its losses are never silently missing)."""
    if not path:
        return []
    path = Path(path)
    if not path.is_file():
        return []
    try:
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh, delimiter=delimiter))
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        print(f"losses: {path} could not be read ({exc}); its rows are missing")
        if bad is not None:
            bad.append(path)
        return []


def unreadable_rows(paths, shown=str):
    """One ``table_unreadable`` row per per-stage table that could not be read."""
    return [
        row(
            cause="table_unreadable",
            effect="flagged",
            detail=f"{Path(p).name} is there and cannot be read",
            source=shown(p),
        )
        for p in paths
    ]


def find(name, *dirs):
    """The first ``<dir>/<name>`` that exists, else None."""
    for d in dirs:
        if d and (Path(d) / name).is_file():
            return Path(d) / name
    return None


PREPARE_DROPPED = "prepare_dropped.csv"
PREPARE_DROPPED_COLUMNS = ("dataset", "patient_id", "case_id", "status", "reason")
# one file per round under the cases folder: the prepare array's tasks write their own
# (no two tasks write one file), and `prepare_dropped.csv` is their union
PREPARE_DROPPED_DIR = ("_cache", "prepare_dropped")


def prepare_dropped_dir(cases_root):
    return Path(cases_root, *PREPARE_DROPPED_DIR)


def read_prepare_dropped(cases_root, bad=None):
    """``(rows, where)``: the rounds ``prepare`` dropped or failed on, from the one file
    per round (the source of truth), else from ``prepare_dropped.csv``; ``([], None)``
    when neither is there (no prepare wrote one)."""
    if not cases_root:
        return [], None
    folder = prepare_dropped_dir(cases_root)
    rows = []
    if folder.is_dir():
        for path in sorted(folder.glob("*/*.csv")):
            rows += read_csv(path, bad)
        return rows, Path(cases_root) / PREPARE_DROPPED
    path = find(PREPARE_DROPPED, cases_root)
    return (read_csv(path, bad), path) if path else ([], None)


def write_prepare_dropped(cases_root, dataset, case_id, row):
    """Record (``row``: a dict of ``PREPARE_DROPPED_COLUMNS``) or clear (``None``) one
    round's drop, then rewrite ``prepare_dropped.csv`` as the union (atomically)."""
    folder = prepare_dropped_dir(cases_root) / dataset
    one = folder / f"{case_id}.csv"
    if row is None:
        if not one.exists():
            return
        one.unlink()
    else:
        folder.mkdir(parents=True, exist_ok=True)
        tmp = one.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(PREPARE_DROPPED_COLUMNS))
            w.writeheader()
            w.writerow({k: row.get(k, "") for k in PREPARE_DROPPED_COLUMNS})
        tmp.replace(one)
    rows, _ = read_prepare_dropped(cases_root)
    union = Path(cases_root) / PREPARE_DROPPED
    tmp = union.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(PREPARE_DROPPED_COLUMNS), restval="")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(union)


def write(path, rows):
    """``losses.csv``: ALWAYS written, a header alone when nothing was lost."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS), restval="")
        w.writeheader()
        w.writerows(rows)


def merge(path, rows, causes):
    """``losses.csv`` at ``path`` with ``rows`` in place of its rows of ``causes``; every
    other row kept. Two writers share ``paper/losses.csv`` (the tables, the figures):
    each replaces its own causes only (T31 fix M7). ALWAYS written."""
    causes = set(causes)
    kept = [r for r in read_csv(path) if r.get("cause") not in causes]
    write(path, [*kept, *rows])
    return path


def counts(rows):
    """``[{stage, cause, blame, effect, n}]``: the loss summary, in the order of
    ``STAGES`` then ``CAUSES``."""
    got = {}
    for r in rows:
        key = (r["stage"], r["cause"], r["blame"], r["effect"])
        got[key] = got.get(key, 0) + 1
    order = {c: i for i, c in enumerate(CAUSES)}
    keys = sorted(
        got, key=lambda k: (STAGES.index(k[0]), order.get(k[1], len(order)), k[3])
    )
    return [
        {"stage": s, "cause": c, "blame": b, "effect": e, "n": got[(s, c, b, e)]}
        for s, c, b, e in keys
    ]


def summary_text(rows, where="losses.csv"):
    """The loss summary ``score`` prints and writes into ``summary.md``."""
    got = counts(rows)
    lines = ["## Where every loss shows up\n"]
    if not got:
        lines.append(f"No loss and no flag: `{where}` has its header alone.\n")
        return "\n".join(lines) + "\n"
    lines.append(
        f"{len(rows)} row(s) in `{where}`, by stage, cause and blame (every one is a "
        "row there; README.txt beside it lists the causes):\n"
    )
    lines.append("| stage | cause | blame | effect | rows |")
    lines.append("|---|---|---|---|---|")
    lines += [
        f"| {g['stage']} | {g['cause']} | {g['blame']} | {g['effect']} | {g['n']} |"
        for g in got
    ]
    return "\n".join(lines) + "\n\n"


def readme_lines():
    """The cause list for ``tables/README.txt``."""
    lines = [
        "losses.csv: one row per loss or flag. Columns: " + ",".join(COLUMNS),
        "  stage  " + ", ".join(STAGES),
        "  blame  " + ", ".join(BLAMES),
        "  effect " + ", ".join(EFFECTS),
        "  causes (blame, stage: what it is):",
    ]
    lines += [f"    {c}  ({b}, {s}): {what}" for c, (b, s, what) in CAUSES.items()]
    return lines


# ── the per-stage tables, as loss rows ───────────────────────────────────────────────
def submit_rows(rows, source="submit_dropped.csv"):
    """``submit_dropped.csv`` (``run,reason,detail``): one row per dropped run."""
    out = []
    for r in rows:
        cause = r.get("reason") or "run_config"
        if cause not in CAUSES:
            cause = "run_config"
        out.append(
            row(
                stage="submit",
                run=r.get("run", ""),
                cause=cause,
                effect="not_run",
                detail=r.get("detail", ""),
                source=source,
            )
        )
    return out


def manifest_rows(rows, source="manifest_dropped.csv"):
    """``manifest_dropped.csv``: one row per round (or patient) the manifest dropped."""
    return [
        row(
            stage="manifest",
            dataset="multiplex",
            patient_id=r.get("patient_id", ""),
            case_id=r.get("slide", ""),
            cause="manifest_dropped",
            effect="dropped",
            detail=f"{r.get('reason', '')} ({r.get('path', '')}; "
            f"{r.get('n_rounds', '?')} round(s))",
            source=source,
        )
        for r in rows
    ]


def prepare_rows(rows, source="prepare_dropped.csv"):
    """``prepare_dropped.csv``: one row per round ``prepare`` dropped or failed on."""
    return [
        row(
            stage="prepare",
            dataset=r.get("dataset", "multiplex"),
            patient_id=r.get("patient_id", ""),
            case_id=r.get("case_id", ""),
            cause="prepare_failed" if r.get("status") == "FAILED" else "prepare_dropped",
            effect="dropped",
            detail=r.get("reason", ""),
            source=source,
        )
        for r in rows
    ]


def unmarked_rows(rows, source="logs/unmarked_units.csv"):
    """``logs/unmarked_units.csv`` (the settle job, ``chains.mark_in_flight``)."""
    return [
        row(
            stage="settle",
            run=r.get("plan", ""),
            case_id=r.get("index", ""),
            cause="unmarked_unit",
            effect="flagged",
            detail=r.get("reason", ""),
            source=source,
        )
        for r in rows
    ]


REFUSED_UNITS = "refused_units.tsv"  # <out>/logs/: retry.sbatch's mark_cut


def refused_rows(rows, source=f"logs/{REFUSED_UNITS}"):
    """``logs/refused_units.tsv``: the units of a refused (or given-up) selfcheck or
    stage step, which no record can carry (retry.sbatch ``mark_cut``)."""
    return [
        row(
            run=r.get("plan", ""),
            case_id=r.get("index", ""),
            cause=r.get("cause", ""),
            effect="not_run",
            detail=f"step {r.get('step', '')} ({r.get('job', '')}): {r.get('why', '')}",
            source=source,
        )
        for r in rows
        if r.get("cause") in CAUSES
    ]


def score_error_rows(rows, source="score_errors.csv"):
    return [
        row(
            stage="score",
            dataset=r.get("dataset", ""),
            patient_id=r.get("patient_id", ""),
            case_id=r.get("case_id", ""),
            cause="score_error",
            effect="dropped",
            detail=f"{r.get('error', '')} at {r.get('where', '')}",
            source=source,
        )
        for r in rows
    ]
