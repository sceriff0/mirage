"""The chains of a cohort submission as the harness reads them back: what each chain's
watcher (``slurm/retry.sbatch``) left, which chains no link of it finished, and the units
written down as the launcher's or the scheduler's because their ladder could not go on.

A CHAIN is a run's memory ladder on one dataset (or a selfcheck's, or one stage of a run
whose registration is several jobs per round). ``submit_cohort.sh`` writes its plan to
``<out>/logs/retry/<submission>/<job prefix>/`` (``step-<k>``, and for a run's ladder the
row it is of, ``run``), and the chain's watcher leaves there:

``last``      ``<index> <job name>-<job id>`` per unit: the last job that held the unit
``state``     the step the chain is at (a successor link goes on from it)
``done``      ``<status> <cause> <exit code>``: the chain is over. A plan with no ``done``
              is a chain NO LINK FINISHED: its watcher's links were all ended (time
              limit, a lost node) or never ran, and what was still due was not submitted

Two writers here, both of RECORDS (``run.json``) and both in the vocabulary that exists
(``resources.NOT_THE_METHOD``: no new cause):

:func:`mark`           the units a chain could not take up its ladder (a rung the
                       partition cannot grant, a step sbatch refused): ``not_submitted``,
                       the launcher's. The memory failure of the rung below is kept in
                       ``cause_from``, and ``ladder_cut_gb`` is the rung that was not
                       submitted. ``regbench chain-mark`` (the watcher calls it).
:func:`mark_in_flight` at settle time, the results of tasks the scheduler had NOT ENDED
                       (a chain no link finished leaves them in the queue):
                       ``infrastructure``, never the method's ``killed``.
"""

from __future__ import annotations

from pathlib import Path

DONE, LAST, STATE, RUN = "done", "last", "state", "run"
LADDER_CUT = "ladder_cut_gb"


def retry_root(out, submission):
    return Path(out) / "logs" / "retry" / str(submission or "none")


def plans(out, submission):
    """Every chain's plan folder of a submission, sorted by name."""
    root = retry_root(out, submission)
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir())


def read_run(plan):
    """``{name, method, label, opts, dataset}`` of a run's ladder, or None (a selfcheck's
    or a stage's chain, or a plan written before the file existed)."""
    try:
        lines = Path(plan, RUN).read_text().splitlines()
    except OSError:
        return None
    if len(lines) < 5:
        return None
    keys = ("name", "method", "label", "opts", "dataset")
    row = dict(zip(keys, lines))
    for k in ("label", "opts"):
        if row[k] == "-":
            row[k] = ""
    return row


def done(plan):
    """``(status, cause, exit code)`` of a finished chain, or None."""
    try:
        words = Path(plan, DONE).read_text().split()
    except OSError:
        return None
    return tuple(words[:3]) if words else None


def last_jobs(plan):
    """``{index: "<job name>-<job id>"}``: the last job that held each unit."""
    try:
        lines = Path(plan, LAST).read_text().splitlines()
    except OSError:
        return {}
    out = {}
    for line in lines:
        idx, _, job = line.partition(" ")
        if idx and job:
            out[idx] = job
    return out


def unfinished(out, submission):
    """The plan folders of the chains no link of whose watcher finished."""
    return [p for p in plans(out, submission) if done(p) is None]


def _units(cases_root, dataset, method):
    from .cases import groups, list_cases
    from .methods import scope_of

    cases = list_cases(cases_root, dataset)
    if scope_of(method) == "group":
        return list(groups(cases).values())
    return [[c] for c in cases]


def _records(paths, submission):
    from . import ladder

    for path in paths:
        rec = ladder._read(path)
        if isinstance(rec, dict) and ladder.mine(rec, submission):
            yield path, rec


def mark(out, cases_root, dataset, method, label, indices, rung_gb, cause, why,
         submission=None):  # fmt: skip
    """Write the units ``indices`` (array task ids) of a run down as ``cause``: their
    ladder was not taken to ``rung_gb``. Only the failed records of this submission
    that the next rung would have run again change (a memory cause, or a ``killed`` or
    ``infrastructure`` one). A ``killed`` record is settled first by its script's note
    (``ladder.read_exits``), as the next rung's task would have. Returns how many
    records changed."""
    from . import ladder, resources
    from .cases import write_json

    units = _units(cases_root, dataset, method)
    exits = ladder.read_exits(out)
    climbs = (*resources.MEMORY_CAUSES, ladder.KILLED, ladder.INFRASTRUCTURE)
    n = 0
    for index in indices:
        if not 0 <= int(index) < len(units):
            continue
        paths = ladder.unit_paths(out, method, label, units[int(index)])
        for p in paths:
            ladder.settle(p, "", None, submission, exits)
        for path, rec in _records(paths, submission):
            was = str(rec.get("cause") or "")
            if rec.get("ok") or rec.get("no_rerun") or was not in climbs:
                continue
            gb = rec.get("mem_requested_gb")
            rec["cause_from"] = (
                f"{was} at rung {rec.get('attempt') or '?'}"
                + (f" ({float(gb):g} GB)" if isinstance(gb, (int, float)) else "")
                + (f" [{rec['cause_from']}]" if rec.get("cause_from") else "")
                + f"; the next rung, {float(rung_gb):g} GB, was not submitted by the "
                f"launcher: {why}"
            )
            rec["cause"] = cause
            rec[LADDER_CUT] = float(rung_gb)
            write_json(path, rec)
            n += 1
    return n


IN_FLIGHT = (
    "the scheduler had not ended its task ({state}) when the benchmark was settled: "
    "the chain's watcher did not finish ({plan}), so what it was waiting for is still "
    "in the queue; not the method's"
)


def mark_in_flight(out, accounting, submission=None, cases_root=None):
    """At settle time: the results whose task the scheduler had not ended.

    * a ``killed`` record (the placeholder of a task that started) whose task the
      accounting shows in a state of ``ladder.STATES_NOT_ENDED``;
    * with ``cases_root``, for a run's ladder no link of whose watcher finished: every
      failed record of a unit whose LAST job (``last``) the accounting does not show
      ended (pending, running, or not there yet), the rung below's failure included,
      since a later rung of it is still to run.

    Each becomes ``infrastructure`` (``cause_from`` says why). Returns
    ``{plan or "-": records changed}``."""
    from . import ladder
    from .cases import write_json

    changed = {}
    pending = {}
    if cases_root:
        for plan in unfinished(out, submission):
            row = read_run(plan)
            if row is None:
                continue
            for idx, job in last_jobs(plan).items():
                job_id = job.rsplit("-", 1)[-1]
                state = (accounting.get(f"{job_id}_{idx}") or {}).get("state", "")
                if not ladder.ended(state):
                    pending.setdefault(plan, []).append((idx, state or "not in sacct"))
    for path in sorted(Path(out).glob("*/*/*/run.json")):
        rec = ladder._read(path)
        if not isinstance(rec, dict) or not ladder.mine(rec, submission):
            continue
        if rec.get("ok") or rec.get("cause") != ladder.KILLED:
            continue
        state = (accounting.get(str(rec.get("launch_id") or "")) or {}).get("state")
        if state and not ladder.ended(state):
            rec["cause_from"] = IN_FLIGHT.format(state=state, plan="its chain")
            rec["cause"] = ladder.INFRASTRUCTURE
            write_json(path, rec)
            changed["-"] = changed.get("-", 0) + 1
    for plan, units in pending.items():
        row = read_run(plan)
        try:
            all_units = _units(cases_root, row["dataset"], row["method"])
        except (SystemExit, Exception):  # noqa: BLE001 - a run this checkout lacks
            continue
        for idx, state in units:
            if not 0 <= int(idx) < len(all_units):
                continue
            paths = ladder.unit_paths(
                out, row["method"], row["label"], all_units[int(idx)]
            )
            for path, rec in _records(paths, submission):
                if rec.get("ok") or rec.get("cause") == ladder.INFRASTRUCTURE:
                    continue
                rec["cause_from"] = (
                    f"{rec.get('cause') or 'exception'} at rung "
                    f"{rec.get('attempt') or '?'}; "
                    + IN_FLIGHT.format(state=state, plan=plan.name)
                )
                rec["cause"] = ladder.INFRASTRUCTURE
                write_json(path, rec)
                changed[plan.name] = changed.get(plan.name, 0) + 1
    return changed


def summary_lines(out, submission):
    """The block the scorer's summary gets when a chain was not finished (empty
    otherwise)."""
    left = unfinished(out, submission)
    if not left:
        return []
    lines = [
        "## Chains no watcher finished\n",
        f"**{len(left)} chain(s) of submission {submission} were not finished by any "
        "link of their watcher** (each link was ended at its time limit, lost with its "
        "node, or never ran). What they still had to submit was not submitted, and "
        "their units' results that the scheduler had not ended were recorded as "
        "`infrastructure` (not the method's) by the settle job. Submit again into the "
        "same results folder: every unit that is ok is reused.\n",
    ]
    for plan in left:
        state = {}
        try:
            for line in Path(plan, STATE).read_text().splitlines():
                key, _, value = line.partition("=")
                state[key] = value
        except OSError:
            pass
        at = (
            f"at step {state['k']} ({state.get('job', '?')}, job {state.get('id', '?')})"
            if state.get("k")
            else "before its first retry step"
        )
        lines.append(f"- `{plan.name}`: {at}")
    lines.append("")
    return lines
