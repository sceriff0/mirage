"""One row per failed or lost event, with whose failure it is (ruling R3.17, 2026-10-11).

``failures.csv`` counts. This module writes what the counts are made of, so that every
kind of failure can be told from every other: ``tables/failure_log.csv`` has ONE ROW PER
EVENT:

* every failed attempt of every unit: the attempt a result ended with, and every earlier
  attempt of the memory ladder kept in its ``run.json: previous_attempts`` (those appear
  even when the unit ended ``ok`` at a higher rung);
* every selfcheck attempt that did not end ``ok``;
* every selfcheck or realcheck VERDICT that is not ok (``logs/verdicts.tsv``): reported,
  never a gate, so the run went on and its rows are scored (``REPORTED``); such a row
  ends no unit and is never a method failure;
* every written image the tissue block could not score, and every record the scorer
  could not read.

A per-patient unit that fails loses every round at once: that is one event per result
name, with the rounds listed in ``case_id``.

**Whose.** ``WHOSE`` maps every cause to exactly one of

``method``          the package under test did it: it raised, ran out of memory or time,
                    its writer failed, its process died
``benchmark``       the benchmark's own code or instrument did it: its point step ran out
                    of memory, its check of the adapter did not pass or could not run, its
                    tissue block or its scorer could not read something
``infrastructure``  the scheduler or the node lost the attempt
``launcher``        the launcher decided it: a unit was never submitted

``whose`` is total by construction: ``classify`` raises on a cause that is not in the
table, and a test walks every vocabulary in the code base against it, so a new cause
cannot be added without saying whose it is. A selfcheck that did not pass is the
``benchmark``'s: it means the benchmark cannot show that ITS points follow the method's
image, which is not evidence that the method registers badly.

**The same words in the logs.** Every batch script ends with one line
``REGBENCH_END key=value ...`` (``end_line``): the task's kind, run, unit, status, cause
and whose, and the path of its log. ``grep REGBENCH_END <out>/logs/*.log`` is the same
classification as the table.

Stdlib only: the launcher's shell and every method image can import it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

METHOD, BENCHMARK, INFRASTRUCTURE, LAUNCHER = (
    "method",
    "benchmark",
    "infrastructure",
    "launcher",
)
OWNERS = (METHOD, BENCHMARK, INFRASTRUCTURE, LAUNCHER)

# cause -> whose. EVERY cause any record can carry is here, once.
WHOSE = {
    # a registration attempt (resources.CAUSES)
    "exception": METHOD,
    "killed": METHOD,  # the process died and nobody said why; `settle` may rename it
    "oom": METHOD,
    "jvm_heap": METHOD,
    "gpu_oom": METHOD,
    "timeout": METHOD,
    "writer": METHOD,
    "writer_oom": METHOD,
    "benchmark_oom": BENCHMARK,
    "benchmark": BENCHMARK,  # its own step failed or was ended, not for memory
    # pixel-identical overlaps exist only because the benchmark cuts tiles from a
    # stitched slide (ruling R6)
    "identical_overlap": BENCHMARK,
    "infrastructure": INFRASTRUCTURE,
    "not_run": LAUNCHER,
    # a stage job never reached the queue: the tile submitter was refused or ended
    "not_submitted": LAUNCHER,
    # the checks: reported, never a gate
    "selfcheck_inconsistent": BENCHMARK,
    "selfcheck_not_checked": BENCHMARK,
    "selfcheck_crashed": BENCHMARK,
    "selfcheck_lost": INFRASTRUCTURE,
    "realcheck_flagged": BENCHMARK,
    "realcheck_not_checked": BENCHMARK,
    # the scorer's own steps
    "tissue_missing": BENCHMARK,
    "unreadable_record": BENCHMARK,
    # a per-case scoring task that did not finish (time, memory, an error): what it
    # had saved is kept, and the final scorer computes the rest
    "score_case_unfinished": BENCHMARK,
    # the settle job could not read the scheduler's accounting; the scorer tries again
    "settle_unfinished": BENCHMARK,
    # Task 28 (regbench.losses): the ladder stopped at the partition's cap (the
    # method's: the ladder is the same for every method, so it is a cost result)
    "capped": METHOD,
    # ... and what is not the method's
    "killed_unsettled": INFRASTRUCTURE,
    "watcher_gave_up": INFRASTRUCTURE,
    "submit_refused": INFRASTRUCTURE,
    "held": INFRASTRUCTURE,
    "result_unreadable": BENCHMARK,
    "score_error": BENCHMARK,
    "realcheck_unreadable": BENCHMARK,
    # rows of losses.csv that are not an attempt: what the launcher dropped, what the
    # inputs lacked, what the benchmark's own steps could not read
    "preflight_failed": LAUNCHER,
    "run_config": LAUNCHER,
    "unmarked_unit": BENCHMARK,
    "no_trace": BENCHMARK,
    "compute_incomplete": BENCHMARK,
    "stale_result": BENCHMARK,
    "manifest_dropped": BENCHMARK,
    "prepare_dropped": BENCHMARK,
    "prepare_failed": BENCHMARK,
    "table_unreadable": BENCHMARK,
    "fit_points_dropped": BENCHMARK,
    "paper_record_unreadable": BENCHMARK,
    "cost_figure_missing": BENCHMARK,
    "cost_peak_not_comparable": BENCHMARK,
    "cost_lower_bound": BENCHMARK,
    "plot_points_dropped": BENCHMARK,
    "plot_lower_bound": BENCHMARK,
    "plot_placeholder": BENCHMARK,
    "stare_local4_differs": BENCHMARK,
    "low_agreement": BENCHMARK,
    "landmark_flagged": BENCHMARK,  # a landmark set's sanity flag: kept, scored (Task 21)
    # the scorer's full LSA beside the headline (Task 21b, regbench.full_pairs)
    "full_lsa_not_certified": BENCHMARK,
    "full_lsa_failed": BENCHMARK,
    "full_lsa_killed": BENCHMARK,
    "frame_check_failed": BENCHMARK,
    "pixel_size_fallback": BENCHMARK,
    "refused_size": BENCHMARK,  # T32: a size the method's own limit refuses (synthetic)
}
STAGES = (
    "selfcheck",
    "realcheck",
    "register",
    "writer",
    "benchmark_points",
    "tissue",
    "score",
)
COLUMNS = (
    "dataset",
    "run",
    "variant",
    "unit",
    "group",
    "case_id",
    "attempt",
    "rung_gb",
    "retry",
    "submission",
    "launch_id",
    "stage",
    "cause",
    "whose",
    "cause_from",
    "scheduler_state",
    "exit_code",
    "elapsed_s",
    "peak_rss_gb",
    "message",
    "log_path",
    "climbed_to",
    "final_outcome_of_unit",
)
NONE = "-"
END = "REGBENCH_END"
LOG_ENV = "REGBENCH_LOG"
END_FILE_ENV = "REGBENCH_END_FILE"


def classify(cause):
    """Whose a cause is. KeyError for a cause the table does not know: add it there."""
    return WHOSE[cause]


def stage_of(cause, phase=""):
    """Where in a registration attempt a cause arose."""
    if cause in ("writer", "writer_oom") or phase == "writer":
        return "writer"
    if cause in ("benchmark_oom", "benchmark"):
        return "benchmark_points"
    return "register"


def outcome(ok, cause, attempt=None):
    """``final_outcome_of_unit``: ``ok at rung N`` | ``failed`` | ``lost``."""
    if ok:
        return f"ok at rung {attempt if attempt is not None else '?'}"
    return "failed" if classify(cause or "exception") == METHOD else "lost"


def first_line(text, limit=240):
    for line in str(text or "").splitlines():
        if line.strip():
            return line.strip()[:limit]
    return ""


def _gb(mb):
    return "" if mb is None else round(float(mb) / 1024.0, 3)


def _blank(value):
    return "" if value is None else value


def _row(**kw):
    row = {c: "" for c in COLUMNS}
    row.update(kw)
    row["whose"] = classify(row["cause"])
    return row


def run_events(run, dataset="", case_id="", variant=""):
    """The events one ``run.json`` holds: each earlier attempt, then its own end when it
    is not ok. ``run`` is the parsed record; an empty one is a result nobody ran."""
    if not run:
        return [
            _row(
                dataset=dataset,
                run=variant,
                variant=variant,
                case_id=case_id,
                stage="register",
                cause="not_run",
                cause_from="no run.json of this submission",
                message="not run",
                climbed_to=NONE,
                final_outcome_of_unit=outcome(False, "not_run"),
            )
        ]
    ok = bool(run.get("ok"))
    cause = "" if ok else (run.get("cause") or "exception")
    final = outcome(ok, cause, run.get("attempt"))
    head = {
        "dataset": run.get("dataset", dataset),
        "run": run.get("label") or run.get("method") or variant,
        "variant": run.get("variant", variant),
        "unit": run.get("unit", ""),
        "group": run.get("group", ""),
        "case_id": run.get("case_id", case_id),
        "final_outcome_of_unit": final,
    }
    attempts = list(run.get("previous_attempts") or [])
    rungs = [p.get("mem_requested_gb") for p in attempts] + [
        run.get("mem_requested_gb")
    ]
    out = []
    for i, p in enumerate(attempts):
        c = p.get("cause") or "exception"
        elapsed = p.get("wall_s")
        if elapsed is None:
            elapsed = p.get("scheduler_elapsed_s")
        after = rungs[i + 1]
        out.append(
            _row(
                **head,
                attempt=_blank(p.get("attempt")),
                rung_gb=_blank(p.get("mem_requested_gb")),
                retry=_blank(p.get("retry")),
                submission=run.get("submission", ""),
                launch_id=_blank(p.get("launch_id")),
                stage=stage_of(c, p.get("phase", "")),
                cause=c,
                cause_from=p.get("cause_from") or "its own record",
                scheduler_state=_blank(p.get("scheduler_state")),
                exit_code=_blank(p.get("exit_code")),
                elapsed_s=_blank(elapsed),
                peak_rss_gb=_gb(p.get("peak_rss_mb")),
                message=first_line(p.get("error")),
                log_path=p.get("log_path") or "",
                climbed_to=f"{after:g} GB"
                if isinstance(after, (int, float))
                and after != p.get("mem_requested_gb")
                else ("the same rung again" if after is not None else NONE),
            )
        )
    if not ok:
        res = run.get("resources") or {}
        sched = run.get("scheduler") or {}
        elapsed = res.get("wall_s")
        if elapsed is None:
            elapsed = sched.get("elapsed_s")
        out.append(
            _row(
                **head,
                attempt=_blank(run.get("attempt")),
                rung_gb=_blank(run.get("mem_requested_gb")),
                retry=_blank(run.get("retry")),
                submission=run.get("submission", ""),
                launch_id=run.get("launch_id") or "",
                stage=stage_of(cause, run.get("phase", "")),
                cause=cause,
                cause_from=run.get("cause_from") or "its own record",
                scheduler_state=_blank(sched.get("state")),
                exit_code=_blank(sched.get("exit_code")),
                elapsed_s=_blank(elapsed),
                peak_rss_gb=_gb(res.get("peak_rss_mb")),
                message=first_line(run.get("error")),
                log_path=run.get("log_path") or "",
                climbed_to=NONE,
            )
        )
    out += stage_events(run, head)
    check = run.get("realcheck") if isinstance(run.get("realcheck"), dict) else {}
    if ok and check.get("flag"):
        out.append(
            _row(
                **head,
                attempt=_blank(run.get("attempt")),
                rung_gb=_blank(run.get("mem_requested_gb")),
                retry=_blank(run.get("retry")),
                submission=run.get("submission", ""),
                launch_id=run.get("launch_id") or "",
                stage="realcheck",
                cause="realcheck_flagged",
                cause_from="realcheck.json of this result",
                message=first_line(check.get("status")),
                log_path=run.get("log_path") or "",
                climbed_to=NONE,
            )
        )
    return out


# a stage of a registration that runs as several scheduler jobs -> the table's stage
STAGE_OF_JOB = {"stitch": "writer"}


def stage_events(run, head):
    """One row per stage attempt that did not end its job ok, for a registration that
    ran as several scheduler jobs per round (``run.json: info.jobs.failed_attempts``,
    ``native.stitch_job.failed_attempts``): a retried attempt of a round that ended
    ok is there too. A job that never recorded its end is ``timeout`` where the
    next attempt's start shows it ran to its limit, else ``killed``
    (``stages.attempt_cause``). The rows carry ``retry`` = ``stage`` so that they are
    told from the rungs of the memory ladder, and are listed once per round."""
    info = run.get("info") if isinstance(run.get("info"), dict) else {}
    jobs = info.get("jobs") if isinstance(info.get("jobs"), dict) else {}
    native = run.get("native") if isinstance(run.get("native"), dict) else {}
    stitch = native.get("stitch_job") if isinstance(native, dict) else None
    listed = list(jobs.get("failed_attempts") or [])
    for a in (stitch or {}).get("failed_attempts") or []:
        listed.append({**a, "stage": "stitch", "cause_from": "its own record"})
    out = []
    for a in listed:
        job, cause = str(a.get("stage") or ""), str(a.get("cause") or "exception")
        if job == "stitch":
            cause = "writer_oom" if cause == "oom" else "writer"
        where = job.upper() + (f" row {a['row']}" if a.get("row") is not None else "")
        out.append(
            _row(
                **head,
                attempt=_blank(a.get("attempt")),
                rung_gb=_blank(a.get("mem_gb")),
                retry="stage",
                submission=run.get("submission", ""),
                launch_id=_blank(a.get("launch_id")),
                stage=STAGE_OF_JOB.get(job, "register"),
                cause=cause,
                cause_from=a.get("cause_from") or "its own record",
                exit_code=_blank(a.get("exit")),
                elapsed_s=_blank(a.get("wall_s")),
                peak_rss_gb=_gb(a.get("peak_rss_mb")),
                message=f"stage job {where}: " + first_line(a.get("error") or cause),
                log_path=a.get("log_path") or "",
                climbed_to="the next attempt of that job" if a.get("retried") else NONE,
            )
        )
    return out


def once_per_round(rows):
    """A round's stage attempts are in the record of each of its result names: keep the
    first of each."""
    seen, out = set(), []
    for r in rows:
        if r.get("retry") == "stage":
            key = (r["dataset"], r["run"], r["case_id"], r["message"], r["attempt"])
            if key in seen:
                continue
            seen.add(key)
        out.append(r)
    return out


def collapse(rows):
    """One row per event: the rounds a per-patient launch lost together are one event
    per result name, with the rounds joined in ``case_id``."""
    out, seen = [], {}
    for r in rows:
        if r["unit"] != "patient":
            out.append(r)
            continue
        key = tuple(r[c] for c in COLUMNS if c not in ("case_id", "message"))
        if key in seen:
            first = seen[key]
            first["case_id"] = f"{first['case_id']};{r['case_id']}"
            continue
        seen[key] = r
        out.append(r)
    return out


def result_events(out, case, variant, mine=None):
    """The events of one (case, result name) on disk. ``mine(run)`` says whether a
    record belongs to what is being scored; one that does not is ``not_run``."""
    path = Path(out) / variant / case.dataset / case.case_id / "run.json"
    try:
        run = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        return [
            _row(
                dataset=case.dataset,
                run=variant,
                variant=variant,
                group=case.group,
                case_id=case.case_id,
                stage="score",
                cause="unreadable_record",
                cause_from="the scorer could not parse run.json",
                message=str(path),
                climbed_to=NONE,
                final_outcome_of_unit="lost",
            )
        ]
    if run and mine is not None and not mine(run):
        run = {}
    from . import losses

    # the causes the frozen writers cannot say (killed_unsettled, held), as score reads
    rows = run_events(losses.reclassify(run), case.dataset, case.case_id, variant)
    for r in rows:
        r["group"] = r["group"] or case.group
    return rows


# what a check's verdict did to the run: nothing. It ends no unit and is never counted
# as one that ended (``not_ok``, ``count_table``).
REPORTED = "reported; the run went on"
REALCHECK_CAUSE = {
    "flagged": "realcheck_flagged",
    "not_checked": "realcheck_not_checked",
    "unreadable": "realcheck_unreadable",
}


def selfcheck_events(verdicts, run_variants=None):
    """One row per selfcheck or realcheck VERDICT that is not ok (``score.read_verdicts``
    of ``logs/verdicts.tsv``). ``stage`` is the check (``selfcheck`` | ``realcheck``),
    ``whose`` the benchmark's: a check that did not pass says the benchmark cannot show
    that ITS points follow the method's image, which is no evidence about the method.
    Reported, never a gate: ``final_outcome_of_unit`` is ``REPORTED``, the run went on
    and its rows are scored."""
    run_variants = run_variants or {}
    out = []
    for g in verdicts or []:
        cause = SELFCHECK_CAUSE.get(g.get("selfcheck") or "not_checked")
        if cause:
            out.append(
                _row(
                    run=g["run"],
                    variant=";".join(run_variants.get(g["run"], [])),
                    stage="selfcheck",
                    cause=cause,
                    cause_from="verdicts.tsv",
                    message=f"selfcheck: {g.get('selfcheck') or 'not_checked'}"
                    + (f" ({g['log']})" if g.get("log") else ""),
                    climbed_to=NONE,
                    final_outcome_of_unit=REPORTED,
                )
            )
        cause = REALCHECK_CAUSE.get(g.get("realcheck") or "")
        if cause:
            out.append(
                _row(
                    run=g["run"],
                    variant=g.get("realcheck_row", ""),
                    stage="realcheck",
                    cause=cause,
                    cause_from="verdicts.tsv",
                    message=f"realcheck: {g['realcheck']}"
                    + (f" ({g['realcheck_log']})" if g.get("realcheck_log") else ""),
                    climbed_to=NONE,
                    final_outcome_of_unit=REPORTED,
                )
            )
    return out


SELFCHECK_CAUSE = {
    "inconsistent": "selfcheck_inconsistent",
    "not_checked": "selfcheck_not_checked",
    "crashed": "selfcheck_crashed",
}


def selfcheck_attempt_events(directory, mine=None):
    """Rows for every selfcheck attempt that did not end ``ok``
    (``<out>/logs/selfcheck/<run>/selfcheck.json``: ``attempts``, this one last)."""
    out = []
    directory = Path(directory) if directory else None
    if directory is None or not directory.is_dir():
        return out
    for path in sorted(directory.glob("*/selfcheck.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            rec = None
        run = path.parent.name
        if rec is None:
            out.append(
                _row(
                    run=run,
                    stage="selfcheck",
                    cause="unreadable_record",
                    cause_from="the scorer could not parse selfcheck.json",
                    message=str(path),
                    climbed_to=NONE,
                    final_outcome_of_unit="lost",
                )
            )
            continue
        if mine is not None and not mine(rec):
            continue
        final = rec.get("status", "not_checked")
        attempts = list(rec.get("attempts") or [{"status": final}])
        bad = {
            k: v for k, v in (rec.get("variants") or {}).items() if v in SELFCHECK_CAUSE
        }
        for i, a in enumerate(attempts):
            status = a.get("status", final)
            if status not in SELFCHECK_CAUSE:
                continue
            last = i == len(attempts) - 1
            nxt = None if last else attempts[i + 1].get("mem_requested_gb")
            out.append(
                _row(
                    run=run,
                    variant=";".join(bad) if last else "",
                    attempt=_blank(a.get("attempt")),
                    rung_gb=_blank(a.get("mem_requested_gb")),
                    submission=a.get("submission", rec.get("submission", "")),
                    launch_id=_blank(a.get("launch_id")),
                    stage="selfcheck",
                    cause=SELFCHECK_CAUSE[status],
                    cause_from="selfcheck.json",
                    message=first_line(
                        a.get("why")
                        or "; ".join(f"{k}: {v}" for k, v in bad.items())
                        or status
                    ),
                    log_path=a.get("log_path") or "",
                    climbed_to=NONE if nxt is None else f"{nxt:g} GB",
                    final_outcome_of_unit="ok" if final == "ok" else final,
                )
            )
    return out


# a tissue row is missing by design for a result that wrote no image (a stage, a failed
# run): only a reason that names an ERROR is an event
TISSUE_ERRORS = ("could not", "Error", "Exception", "is gone", "not readable")


def tissue_events(missing):
    """Rows for the written images the tissue block could not score because something
    went wrong (not for results that have no image by design)."""
    return [
        _row(
            dataset=m.get("dataset", ""),
            run=m.get("variant", ""),
            variant=m.get("variant", ""),
            group=m.get("group", ""),
            case_id=m.get("case_id", ""),
            stage="tissue",
            cause="tissue_missing",
            cause_from="tissue_missing.csv",
            message=first_line(m.get("reason")),
            climbed_to=NONE,
            final_outcome_of_unit="no tissue row",
        )
        for m in missing
        if any(word in str(m.get("reason", "")) for word in TISSUE_ERRORS)
    ]


def _fold_not_run(rows):
    """The results nobody ran, one row per (dataset, result name) with their count: a
    submission of some of the runs would otherwise list every case of every other."""
    out, seen = [], {}
    for r in rows:
        if r["cause"] != "not_run" or r["launch_id"]:
            out.append(r)
            continue
        key = (r["dataset"], r["variant"], r["final_outcome_of_unit"])
        if key in seen:
            seen[key][0] += 1
            continue
        seen[key] = [1, r]
        out.append(r)
    for n, r in seen.values():
        r["case_id"], r["group"] = f"{n} case(s)", ""
    return out


def build(out, jobs, verdicts=(), run_variants=None, tissue_missing=(), mine=None):
    """Every event of what is being scored. ``jobs`` are ``(case, [result names])``;
    ``verdicts`` is ``score.read_verdicts`` of ``logs/verdicts.tsv``."""
    rows = []
    for case, names in jobs:
        for name in names:
            if name == "initial":
                continue
            rows += result_events(out, case, name, mine)
    rows = _fold_not_run(collapse(once_per_round(rows)))
    rows += selfcheck_events(verdicts, run_variants)
    rows += selfcheck_attempt_events(Path(out) / "logs" / "selfcheck", mine)
    rows += tissue_events(tissue_missing)
    return rows


def count_table(rows):
    """One row per (run, cause, whose) with the number of events and of units."""
    out = {}
    for r in rows:
        key = (r["run"], r["cause"], r["whose"], r["stage"])
        rec = out.setdefault(
            key,
            {
                "run": r["run"],
                "stage": r["stage"],
                "cause": r["cause"],
                "whose": r["whose"],
                "events": 0,
                "ended_the_unit": 0,
            },
        )
        rec["events"] += 1
        rec["ended_the_unit"] += _ended(r)
    return list(out.values())


def _ended(r):
    """Did this event end its unit: not an attempt it climbed past, not an ok, and not
    a check's verdict (``REPORTED``: the run went on)."""
    outcome = str(r["final_outcome_of_unit"])
    return (
        r["climbed_to"] == NONE and not outcome.startswith("ok") and outcome != REPORTED
    )


def not_ok(rows):
    """The events a unit ENDED with (not the attempts it climbed past, and not the
    checks' verdicts, which end nothing)."""
    return [r for r in rows if _ended(r)]


def markdown(rows, table, limit=200):
    """The "Failures" section of ``summary.md``; ``table(rows, columns)`` renders."""
    parts = ["## Failures: every event, and whose it is\n"]
    if not rows:
        return [*parts, "No failed or lost event, and no check that was not ok.\n"]
    parts.append(
        "`failure_log.csv` has one row per event: each failed attempt of each unit "
        "(earlier rungs of the memory ladder included, even when the unit ended ok), "
        "each selfcheck attempt that did not end ok, each selfcheck or realcheck "
        "verdict that is not ok (reported: the run went on and its rows are scored; "
        f"`final_outcome_of_unit` = `{REPORTED}`), each image the tissue block "
        "could not score. `whose`: `method` (the package under test raised, ran out "
        "of memory or time, its writer failed, its process died); `benchmark` (the "
        "benchmark's own step or check: not evidence about the method's "
        "registration); `infrastructure` (the scheduler or node lost the attempt); "
        "`launcher` (the unit was never submitted). `grep REGBENCH_END` in the task "
        "logs gives the same words.\n"
    )
    parts += [
        table(
            count_table(rows),
            ["run", "stage", "cause", "whose", "events", "ended_the_unit"],
        ),
        "",
    ]
    final = not_ok(rows)
    other = [r for r in final if r["whose"] != METHOD and r["cause"] != "not_run"]
    if other:
        parts.append(
            f"**{len(other)} unit(s) ended for a reason that is NOT the method's "
            "(benchmark, infrastructure or launcher): read their rows below before "
            "any number of those runs, and submit the lost ones again.**\n"
        )
    else:
        parts.append(
            "No unit ended for a reason that is not the method's (a unit that was "
            "never submitted aside).\n"
        )
    if final:
        parts.append("### Units whose final outcome is not ok\n")
        cols = [
            "dataset",
            "run",
            "variant",
            "group",
            "case_id",
            "stage",
            "cause",
            "whose",
            "rung_gb",
            "final_outcome_of_unit",
            "message",
            "log_path",
        ]
        parts += [table(final[:limit], cols), ""]
        if len(final) > limit:
            parts.append(
                f"{len(final) - limit} more in `failure_log.csv` (the table above "
                f"shows the first {limit}).\n"
            )
    return parts


# ── the last line of every task's log ────────────────────────────────────────────────
def end_line(kind, status, cause="", **fields):
    """``REGBENCH_END kind=... status=... cause=... whose=... key=value ...``: one line,
    no spaces inside a value."""

    def clean(v):
        return str(v).replace(" ", "_").replace("\t", "_").replace("\n", "_") or NONE

    cause = cause or NONE
    items = {
        "kind": kind,
        **fields,
        "status": status,
        "cause": cause,
        "whose": classify(cause) if cause != NONE else NONE,
        "log": os.environ.get(LOG_ENV) or NONE,
    }
    return END + " " + " ".join(f"{k}={clean(v)}" for k, v in items.items())


def emit_end(kind, status, cause="", **fields):
    """Print the task's end line, and leave it where the batch script repeats it as the
    log's LAST line (``REGBENCH_END_FILE``; the script writes a fallback when the
    process died before this)."""
    line = end_line(kind, status, cause, **fields)
    print(line, flush=True)
    path = os.environ.get(END_FILE_ENV)
    if path:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(line + "\n")
        except OSError:
            pass
    return line


def chain_wants(records):
    """The steps of a chain's watcher (``slurm/retry.sbatch``) that may still run this
    unit, from the ``run.json`` records of its results after its last task: a tuple of
    ``"again"`` (the same rung once more) and ``"climb"`` (the next rung), empty when
    no later task of the chain can run it. Written on the task's end line
    (``wants=``), and the watcher submits a step only for the units that want it.

    THE ONE PREDICATE of the watcher's filter. It is read from every record of the
    unit, not from the first failed one: a patient goes up as a whole when ANY of its
    records ran out of memory (``ladder.unit_decision``), whatever cause another round
    failed with first. It answers for a SUPERSET of what the ladder will run (a test
    holds it to that: ``test_failure_log.py``): the task submitted still decides for
    itself, and a unit the filter takes may find nothing to do; a unit it leaves out
    would be one the ladder never got to decide.

    * no record of this submission: the task never started; both (``again`` runs it at
      rung 1, ``climb`` writes it down as lost)
    * a memory cause (``resources.MEMORY_CAUSES``): ``climb``; never ``again``, which
      retries an attempt lost to the scheduler and not one that ran out of memory
    * ``infrastructure`` not yet tried again (``retry`` 0): both; tried again: neither
    * ``timeout`` not yet tried again: ``again`` (with more time, ruling 2026-10-21),
      unless a record of the unit ran out of memory (it goes up whole instead)
    * ``killed`` with no end state recorded: both (the accounting may still say memory,
      a lost node or nothing); with one (``scheduler.state``, written when it was
      settled) that names no cause, not yet tried again: ``again`` (a process killed by
      a signal, ruling 2026-10-21; not when the unit goes up for memory); tried again:
      neither
    * a record already retried where it failed (``no_rerun``) and any other cause:
      neither
    """
    from . import ladder, resources

    records = [r for r in records if r]
    if not records:
        return (ladder.AGAIN, ladder.CLIMB)
    wants = set()
    retry_here = memory = False  # a timeout or a signal: again, unless memory climbs it
    for r in records:
        if r.get("ok") or r.get("no_rerun"):
            continue
        cause = r.get("cause") or "exception"
        if cause in resources.MEMORY_CAUSES:
            wants.add(ladder.CLIMB)
            memory = True
        elif cause == ladder.INFRASTRUCTURE:
            if int(r.get("retry") or 0) < ladder.RETRIES_PER_RUNG:
                wants.update((ladder.AGAIN, ladder.CLIMB))
        elif cause == ladder.TIMEOUT:
            if int(r.get("retry") or 0) < ladder.RETRIES_PER_RUNG:
                retry_here = True
        elif cause == ladder.KILLED:
            if not ladder.ended((r.get("scheduler") or {}).get("state")):
                wants.update((ladder.AGAIN, ladder.CLIMB))
            elif int(r.get("retry") or 0) < ladder.RETRIES_PER_RUNG:
                retry_here = True
    if retry_here and not memory:  # as ladder.unit_decision: memory goes up instead
        wants.add(ladder.AGAIN)
    return tuple(k for k in (ladder.AGAIN, ladder.CLIMB) if k in wants)


def unit_end(records):
    """``(status, cause)`` of one unit from the ``run.json`` records of its results:
    ``ok`` when every one is, else the first failed one's cause and ``failed`` (the
    method's) or ``lost`` (anything else)."""
    records = [r for r in records if r]
    if not records:
        return "lost", "not_run"
    for r in records:
        if not r.get("ok"):
            cause = r.get("cause") or "exception"
            return outcome(False, cause), cause
    return "ok", ""
