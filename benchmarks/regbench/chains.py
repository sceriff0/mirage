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

``gave_up``  the watcher gave up (the queue could not be read): the reason. Set by
              ``retry.sbatch`` before it exits without ``done``; the settle job then
              writes what was still due as ``watcher_gave_up`` (Task 28, S3)

Two writers here, both of RECORDS (``run.json``). Their causes are
``resources.NOT_THE_METHOD``'s and, since Task 28, ``losses.NOT_THE_METHOD_EXTRA``'s
(``watcher_gave_up``, ``submit_refused``) and ``capped`` (the method's: the ladder
reached the largest rung the partition grants):

:func:`mark`           the units a chain could not take up its ladder (a rung the
                       partition cannot grant, a step sbatch refused): ``not_submitted``,
                       the launcher's. The memory failure of the rung below is kept in
                       ``cause_from``, and ``ladder_cut_gb`` is the rung that was not
                       submitted. ``regbench chain-mark`` (the watcher calls it).
:func:`mark_in_flight` at settle time, the results of tasks the scheduler had NOT ENDED
                       (a chain no link finished leaves them in the queue):
                       ``infrastructure``, never the method's ``killed``. And, of a
                       chain no link finished, a unit whose task WAS ended but whose
                       next step (``failure_log.chain_wants``) was never submitted:
                       ``not_submitted`` through :func:`mark`, the rungs above it untried.
"""

from __future__ import annotations

from pathlib import Path

DONE, LAST, STATE, RUN, GAVE_UP = "done", "last", "state", "run", "gave_up"
LADDER_CUT = "ladder_cut_gb"
# the causes ``mark`` writes (``regbench chain-mark --cause``)
NOT_SUBMITTED, CAPPED = "not_submitted", "capped"
SUBMIT_REFUSED, WATCHER_GAVE_UP = "submit_refused", "watcher_gave_up"
MARK_CAUSES = (NOT_SUBMITTED, CAPPED, SUBMIT_REFUSED, WATCHER_GAVE_UP)
UNMARKED = "unmarked_units.csv"
UNMARKED_COLUMNS = ("plan", "index", "reason")


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


def gave_up(plan):
    """Why the chain's watcher gave up (``gave_up``), or None."""
    try:
        text = Path(plan, GAVE_UP).read_text().strip()
    except OSError:
        return None
    return text or "the watcher gave up"


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


def read_state(plan):
    """``{k, job, id}`` of the step the chain is at (``state``), ``{}`` before any."""
    state = {}
    try:
        for line in Path(plan, STATE).read_text().splitlines():
            key, _, value = line.partition("=")
            state[key] = value
    except OSError:
        pass
    return state


def later_steps(plan):
    """``[(k, kind, job name)]`` of the plan's steps after the one it is at, in order."""
    try:
        at = int(read_state(plan).get("k") or 0)
    except ValueError:
        at = 0
    steps = []
    for f in Path(plan).glob("step-*"):
        try:
            k = int(f.name.split("-", 1)[1])
            lines = f.read_text().splitlines()
        except (ValueError, OSError):
            continue
        if k > at and len(lines) >= 2:
            steps.append((k, lines[0], lines[1]))
    return sorted(steps)


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
    records changed.

    ``capped`` (a rung the partition cannot grant, B6) is written only over a MEMORY
    failure (the ladder would have climbed for it); the record's ``error`` then says
    ``capped at <the rung below> GB`` and the unit is the method's, imputed. Any other
    failure below a cut rung is ``not_submitted``, as before."""
    from . import ladder, resources
    from .cases import write_json

    units = _units(cases_root, dataset, method)
    exits = ladder.read_exits(out)
    climbs = (*resources.MEMORY_CAUSES, ladder.KILLED, ladder.INFRASTRUCTURE)
    n = 0
    if cause not in MARK_CAUSES:
        raise ValueError(f"chains.mark: {cause!r} is not one of {MARK_CAUSES}")
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
            if cause == CAPPED and was not in resources.MEMORY_CAUSES:
                rec["cause"] = NOT_SUBMITTED
            elif cause == CAPPED:
                below = f"{float(gb):g} GB" if isinstance(gb, (int, float)) else (
                    "the rung below"
                )
                rec["error"] = (
                    f"capped at {below}: the next rung, {float(rung_gb):g} GB, cannot "
                    f"be granted ({why}); last failure {was}"
                )
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

    Each becomes ``infrastructure`` (``cause_from`` says why).

    * with ``cases_root``, for the same chains: a unit none of whose tasks is still in
      flight, whose records want a later step (``failure_log.chain_wants``) the plan
      has and no link submitted (after ``state``'s step): ``not_submitted`` (:func:`mark`,
      ``ladder_cut_gb`` the rung of that step). An oom at 16 GB whose 64 GB step was
      never submitted is the launcher's, not the method's.

    A chain whose watcher GAVE UP (``gave_up``: the queue could not be read) has both
    kinds written as ``watcher_gave_up`` instead (S3): what it was waiting for, and what
    it never submitted, were lost to the watcher, not to the method or the launcher.

    A run's chain whose units cannot be listed (its ``run`` file is unreadable, or the
    cases of its dataset cannot be read) is no longer passed over in silence (S2): each
    of its units is a row of ``<out>/logs/unmarked_units.csv`` (``plan,index,reason``;
    written at every settle, header alone when there is none), and the scorer makes
    each an ``unmarked_unit`` row of ``losses.csv``.

    Returns ``{plan or "-": records changed}`` (an :class:`InFlight`), whose
    ``n_unmarked_units`` is the number of those rows (S2)."""
    import re

    from . import failure_log, ladder
    from .cases import write_json

    changed = InFlight()
    pending = {}
    cut = []
    unmarked = {}

    def cannot_list(plan, reason, only=None):
        idxs = sorted(last_jobs(plan), key=lambda i: (len(i), i)) or [""]
        for idx in idxs if only is None else only:
            unmarked.setdefault((plan.name, idx), reason)

    def outside(plan, all_units):
        """The plan's units its case list does not have (S2)."""
        beyond = [
            i for i in sorted(last_jobs(plan), key=lambda i: (len(i), i))
            if not (i.isdigit() and int(i) < len(all_units))
        ]
        if beyond:
            cannot_list(
                plan, f"not among the {len(all_units)} unit(s) its cases list", beyond
            )

    if cases_root:
        for plan in unfinished(out, submission):
            row = read_run(plan)
            if row is None:
                if Path(plan, RUN).exists():  # a run's ladder whose row cannot be read
                    cannot_list(plan, f"its {RUN} file is unreadable or short")
                continue
            for idx, job in last_jobs(plan).items():
                job_id = job.rsplit("-", 1)[-1]
                state = (accounting.get(f"{job_id}_{idx}") or {}).get("state", "")
                if not ladder.ended(state):
                    pending.setdefault(plan, []).append((idx, state or "not in sacct"))
            later = later_steps(plan)
            if not later:
                continue
            try:
                all_units = _units(cases_root, row["dataset"], row["method"])
            except (SystemExit, Exception) as exc:  # noqa: BLE001 - logged (S2)
                cannot_list(plan, f"its units cannot be listed: {exc}")
                continue
            outside(plan, all_units)
            held = {str(i) for i, _ in pending.get(plan, [])}
            for index, unit in enumerate(all_units):
                if str(index) in held:
                    continue
                paths = ladder.unit_paths(out, row["method"], row["label"], unit)
                records = [rec for _, rec in _records(paths, submission)]
                if any(
                    r.get("cause") == ladder.KILLED
                    and not ladder.ended(
                        (accounting.get(str(r.get("launch_id") or "")) or {}).get(
                            "state"
                        )
                        or "COMPLETED"
                    )
                    for r in records
                    if not r.get("ok")
                ):
                    continue  # still in flight: the pass below makes it infrastructure
                wants = failure_log.chain_wants(records)
                for k, kind, jobname in later:
                    taken = kind if kind != "cut" else ladder.CLIMB
                    if taken not in wants:
                        continue
                    gb = re.search(r"-(\d+)g(-again)?$", jobname)
                    if gb:
                        cut.append((plan, row, index, k, jobname, int(gb.group(1))))
                    break
    for plan, row, index, k, jobname, gb in cut:
        quit_ = gave_up(plan)
        n = mark(
            out, cases_root, row["dataset"], row["method"], row["label"], [index], gb,
            WATCHER_GAVE_UP if quit_ else NOT_SUBMITTED,
            f"the chain's watcher did not finish ({plan.name}"
            + (f"; it gave up: {quit_}" if quit_ else "")
            + f"): step {k} ({jobname}) was never submitted by any of its links",
            submission,
        )  # fmt: skip
        if n:
            changed[plan.name] = changed.get(plan.name, 0) + n
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
        quit_ = gave_up(plan)
        try:
            all_units = _units(cases_root, row["dataset"], row["method"])
        except (SystemExit, Exception) as exc:  # noqa: BLE001 - logged (S2)
            cannot_list(plan, f"its units cannot be listed: {exc}")
            continue
        outside(plan, all_units)
        for idx, state in units:
            if not 0 <= int(idx) < len(all_units):
                continue
            paths = ladder.unit_paths(
                out, row["method"], row["label"], all_units[int(idx)]
            )
            for path, rec in _records(paths, submission):
                if rec.get("ok") or rec.get("cause") == ladder.INFRASTRUCTURE:
                    continue
                if rec.get("cause") == WATCHER_GAVE_UP:
                    continue
                rec["cause_from"] = (
                    f"{rec.get('cause') or 'exception'} at rung "
                    f"{rec.get('attempt') or '?'}; "
                    + IN_FLIGHT.format(state=state, plan=plan.name)
                    + (f"; the watcher gave up: {quit_}" if quit_ else "")
                )
                rec["cause"] = WATCHER_GAVE_UP if quit_ else ladder.INFRASTRUCTURE
                write_json(path, rec)
                changed[plan.name] = changed.get(plan.name, 0) + 1
    if cases_root:
        write_unmarked(out, unmarked)
    changed.n_unmarked_units = len(unmarked)
    return changed


class InFlight(dict):
    """``{plan or "-": records changed}`` plus ``n_unmarked_units``: the units of
    ``logs/unmarked_units.csv`` this settle wrote (S2)."""

    n_unmarked_units = 0


def write_unmarked(out, unmarked):
    """``<out>/logs/unmarked_units.csv`` from ``{(plan, index): reason}`` (header alone
    when empty)."""
    import csv

    path = Path(out) / "logs" / UNMARKED
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(UNMARKED_COLUMNS)
        for (plan, idx), reason in sorted(unmarked.items()):
            w.writerow((plan, idx, reason))


def read_unmarked(out):
    """The rows of ``<out>/logs/unmarked_units.csv`` (``[]`` without it, or when it
    cannot be read: the scorer then writes a ``table_unreadable`` row for it)."""
    import csv

    try:
        with open(Path(out) / "logs" / UNMARKED, newline="") as fh:
            return list(csv.DictReader(fh))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


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
        state = read_state(plan)
        at = (
            f"at step {state['k']} ({state.get('job', '?')}, job {state.get('id', '?')})"
            if state.get("k")
            else "before its first retry step"
        )
        lines.append(f"- `{plan.name}`: {at}")
    lines.append("")
    return lines
