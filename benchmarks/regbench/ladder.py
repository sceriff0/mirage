"""The memory ladder's decisions: does a unit go up a rung, is a lost attempt tried again,
and how long may a job run.

The cohort launcher (``slurm/submit_cohort.sh``) submits the jobs of a run up front,
each waiting for the one before it (``afterany``); only the tile jobs of a staged run
come later, from its tile submitter. Nothing waits on a login node. So a
task must decide FOR ITSELF, when it starts, whether its unit has anything to do. That
decision is here, in Python, so that it can be tested without a scheduler.

**The chain of one unit.** Per rung (16, 64, 256, 512 GB) there are two array tasks:

``first``  (rung 1) / ``climb`` (rungs 2-4)   the attempt at that rung
``again``                                      the SAME rung once more, and only when the
                                               attempt before it was lost to the
                                               scheduler, not to the method

**Evidence** is the driver's own record, ``run.json: cause``, of THIS submission
(``run.json: submissions`` holds the launcher's id; a record another submission left is
not evidence of anything, see :func:`mine`):

a memory cause (``resources.MEMORY_CAUSES``: ``oom``, ``writer_oom``, ``benchmark_oom``)
             the unit goes up a rung.
``killed``   the placeholder the driver writes before it starts; it is still there when
             the process was killed before it could say why. Then, and only then, the
             scheduler's END state for the task THAT WROTE THE RECORD decides (matched
             by ``launch_id`` in ``--accounting``, or ``--previous-state`` for the task
             just before): ``OUT_OF_MEMORY`` makes it ``oom``, ``TIMEOUT`` makes it
             ``timeout``, and a state of :data:`STATES_INFRASTRUCTURE` (node failure,
             preemption, a cancelled job, a failed boot) makes it ``infrastructure``.
             The record is rewritten to say so, with ``cause_from`` naming the state and
             ``scheduler`` holding the task's elapsed and CPU time, so the attempt's
             cost is the scheduler's figure and not zero. The rule is the same for
             every record, whichever method and whichever job of it wrote the
             placeholder. An end state that names no cause (``FAILED``, ``COMPLETED``)
             is settled by the task's own script where it left a note
             (:func:`read_exits`: Python ended by signal 9 while the script went on,
             which is the kernel's out-of-memory killer: ``oom``); with no note it
             stays ``killed``: a process that died on its own is the method's failure.
no end state  the accounting does not have the task yet, or shows it in a state of
             :data:`STATES_NOT_ENDED`. The rung task asks again until it has one
             (``slurm/run_rung.sbatch``: ``REGBENCH_ACCT_WAIT_S``). A record still
             ``killed`` without an end state after that wait is NOT final: the
             ``climb`` task runs the unit again at its own rung's memory, and the
             record it replaces says so (``cause_from``, carried into
             ``previous_attempts``). The cause was never known, so it is neither
             counted as a memory cause nor left as the method's last word.
``infrastructure``  the ``again`` task of the same rung runs the unit once more. ONE
             retry per rung: an attempt lost twice stays ``infrastructure``, which the
             tables count apart from the method's failures. When the accounting names
             the lost attempt only after that ``again`` task has decided, the retry is
             the next rung's ``climb`` task: it runs the unit once, at its own rung's
             memory (a task has one allocation), and the record it replaces says that
             this is the retry of a lost attempt and not a climb for memory
             (``cause_from``, carried into ``previous_attempts``). Memory stays the
             only CAUSE that sends a unit up. After the last rung there is no such
             task.
no record of this submission   the task that should have written it never started. At
             rung 1 the ``again`` task runs the unit; at a higher rung it does when the
             unit's last record is a memory cause of the rung below (the ``climb`` task
             was lost before its placeholder). A unit still without a record after
             both is written down as ``infrastructure`` by the next ``climb`` task
             (:func:`mark_lost`), so that it is never read as "not run" or as failed.
             A record another submission left there is set aside, not destroyed.
anything else (``ok``, ``exception``, ``timeout``, ``gpu_oom``, ``writer``): the unit is
             never re-run.
``no_rerun: true``  whatever its cause, the record is of a failure that was already
             retried where it happened: a stage job of a registration that runs as
             several jobs (``regbench.stages``). No rung runs the unit again.

A unit is one case for a method that runs per pair and one patient for a method that runs
per patient; a patient goes up as a whole when any of its rounds ran out of memory.

The selfcheck climbs too (ruling R3.5): 64, then 256, then 512 GB. A later attempt runs
when the earlier one ended for memory or left no record of this submission
(``selfcheck_retry``): a selfcheck lost to the scheduler is run again, not read as
``not_checked``.

Frozen: the rungs, the CPUs, the one retry per rung and the site's longest job. Changing
one after a cohort run is a post hoc change.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import resources

RUNGS_GB = (16, 64, 256, 512)
SELFCHECK_RUNGS_GB = (64, 256, 512)
CPUS = 4  # every method, every rung
MAX_HOURS = 168  # the site's longest job: 7 days
RETRIES_PER_RUNG = 1  # a rung is tried again once when the scheduler lost the attempt

OOM = "oom"
KILLED = "killed"
INFRASTRUCTURE = "infrastructure"
FIRST, CLIMB, AGAIN = "first", "climb", "again"
TASKS = (FIRST, CLIMB, AGAIN)
# What the chain's watcher (slurm/retry.sbatch) reads off a unit's last end line before it
# SUBMITS a step for it: a filter, never the decision (the task decides, unit_decision).
# A unit whose last cause is in neither list is not put in the queue again; a unit with
# no end line at all (its task died or never started) is taken by every step.
AGAIN_CAUSES = (KILLED, INFRASTRUCTURE)
CLIMB_CAUSES = (OOM, "writer_oom", "benchmark_oom", KILLED, INFRASTRUCTURE)
# sacct's State, matched by its first word ("CANCELLED by 123", "OUT_OF_MEMORY+" are
# seen: the manual says a state too long for its column is followed by "+"). The names
# are those of the SLURM `sacct` manual's JOB STATE CODES `[PARTIAL: read the JOB STATE
# CODES list and the -S, -j, -X, --name entries; missing the rest of the page]`.
STATE_OOM = "OUT_OF_ME"
STATE_TIMEOUT = "TIMEOUT"
STATES_INFRASTRUCTURE = (
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "CANCELLED",
    "REQUEUED",
    "REVOKED",
)
# States of a task that has not ended: no evidence of how it will. PENDING, RUNNING,
# RESIZING and SUSPENDED are sacct's (the same list); COMPLETING is squeue's name for a job whose processes are still ending, kept
# here so that it is never read as an end. slurm/run_rung.sbatch has the same list.
STATES_NOT_ENDED = ("PENDING", "RUNNING", "RESIZING", "SUSPENDED", "COMPLETING")
# What the task's own script notes when Python ended by signal 9 and the script went
# on (slurm/run_rung.sbatch): <out>/logs/accounting/exits/<launch id> holds the exit
# code. The scheduler's own kill (time limit, scancel) ends the script too, so a note
# is the kernel's out-of-memory killer taking the largest process of the task.
EXITS_DIR = ("logs", "accounting", "exits")
EXIT_SIGKILL = 137
# `run --accounting-wanted`: the exit code that asks the script to wait for the
# accounting and call again (sysexits.h EX_TEMPFAIL)
EXIT_ACCOUNTING_PENDING = 75
NO_END_STATE = "no end state in the scheduler's accounting"


def _state(previous_state):
    return str(previous_state or "").strip().upper()


def scheduler_cause(state):
    """The cause a scheduler state gives a record that still says ``killed``; None for a
    state that says nothing about why (``COMPLETED``, ``FAILED``, unknown)."""
    state = _state(state)
    if state.startswith(STATE_OOM):
        return OOM
    if state.startswith(STATE_TIMEOUT):
        return "timeout"
    if state.startswith(STATES_INFRASTRUCTURE):
        return INFRASTRUCTURE
    return None


def ended(state):
    """Is this a state the scheduler gives a task that is over? Not an empty one, and
    not one of :data:`STATES_NOT_ENDED`."""
    state = _state(state)
    return bool(state) and not state.startswith(STATES_NOT_ENDED)


def exits_dir(out):
    return Path(out).joinpath(*EXITS_DIR)


def read_exits(out):
    """``{launch id: exit code}`` of the notes the rung tasks' scripts left under
    ``<out>/logs/accounting/exits``: one file per task whose Python step ended by
    signal 9 while the script itself went on."""
    found = {}
    try:
        paths = sorted(exits_dir(out).iterdir())
    except OSError:
        return found
    for path in paths:
        try:
            found[path.name] = int(path.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            continue
    return found


def _seconds(text):
    """``[D-]HH:MM:SS[.fff]``, ``MM:SS[.fff]`` or plain seconds; None when unreadable."""
    text = str(text or "").strip()
    if not text:
        return None
    try:
        days = 0.0
        if "-" in text:
            d, text = text.split("-", 1)
            days = float(d)
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        return None
    if len(parts) > 3:
        return None
    total = 0.0
    for p in parts:
        total = total * 60.0 + p
    return days * 86400.0 + total


def parse_accounting(text):
    """``{job id: {"state", "elapsed_s", "cpu_s"}}`` from
    ``sacct -n -P -o JobID,State,ElapsedRaw,TotalCPU``, asked with or without ``-X``:
    ``|``-separated lines, one per job or array task (``<array job>_<task>``, which is
    also a record's ``launch_id``) and, without ``-X``, one per step of it
    (``<job>.batch``, ``<job>.extern``, ``<job>.<n>``).

    State and elapsed time are the job's own line's. ``cpu_s`` is the job's total CPU
    time: the larger of its own line's ``TotalCPU`` and the sum of its steps', because
    the job's line alone carries none when the steps were left out ("Without including
    steps, utilization statistics for job allocation(s) will be reported as zero",
    sacct manual, ``regbench.billing`` cites it) and carries the steps' total when it
    was asked with them (`[MEMORY]`: that sacct adds the steps up into the job's line
    is not in the passages read; taking the larger is right either way). A total of
    zero is None: nothing was reported, which is not "no CPU was used". Steps of a job
    with no line of its own, and lines that do not parse, are skipped."""
    out, steps = {}, {}
    for line in str(text or "").splitlines():
        fields = [f.strip() for f in line.split("|")]
        if len(fields) < 2 or not fields[0] or not fields[1]:
            continue
        cpu = _seconds(fields[3]) if len(fields) > 3 else None
        job, _, step = fields[0].partition(".")
        if step:
            steps[job] = steps.get(job, 0.0) + (cpu or 0.0)
            continue
        out[job] = {
            "state": _state(fields[1]),
            "elapsed_s": _seconds(fields[2]) if len(fields) > 2 else None,
            "cpu_s": cpu,
        }
    for job, rec in out.items():
        total = max(rec["cpu_s"] or 0.0, steps.get(job, 0.0))
        rec["cpu_s"] = total if total > 0 else None
    return out


def read_accounting(path):
    if not path:
        return {}
    try:
        return parse_accounting(Path(path).read_text())
    except OSError:
        return {}


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def submissions_of(rec):
    """The submission ids a record (``run.json``, ``selfcheck.json``) carries, in the
    order it lists them: ``submissions`` (every submission that ran or reused the
    result), then ``submission`` (the one that wrote it). ``[]`` for anything else."""
    if not isinstance(rec, dict):
        return []
    known = [str(v) for v in rec.get("submissions") or [] if v]
    if rec.get("submission"):
        known.append(str(rec["submission"]))
    return list(dict.fromkeys(known))


def mine(rec, submission):
    """Does a record belong to this submission? Always, when there is no submission id
    (a run outside the cohort launcher). The driver adds the id of a submission that
    reuses an ``ok`` result to ``submissions``."""
    if not submission:
        return True
    return submission in submissions_of(rec)


def selfcheck_submissions(out):
    """``({submission id: [runs]}, [unreadable files])`` of the selfcheck records under
    ``<out>/logs/selfcheck/<run>/selfcheck.json``, each PARSED and read with the reader
    the verdicts use (``submissions_of``; a verdict takes a record only when ``mine``
    says so). A record that names no submission (a selfcheck run by hand, outside the
    launcher) is under ``""``. This is what ``submit_cohort.sh --score-only`` asks when
    the folder has no ``logs/submission``."""
    found, unreadable = {}, []
    for path in sorted(Path(out).glob("logs/selfcheck/*/selfcheck.json")):
        rec = _read(path)
        if not isinstance(rec, dict):
            unreadable.append(str(path))
            continue
        for sub in submissions_of(rec) or [""]:
            found.setdefault(sub, []).append(path.parent.name)
    return found, unreadable


def _end_state(rec, previous_state="", accounting=None):
    """``(end state or "", the accounting's line or None)`` of the task that wrote a
    record. A state of a task that has not ended is no state."""
    found = (accounting or {}).get(str(rec.get("launch_id") or ""))
    state = found["state"] if found else _state(previous_state)
    if not ended(state):
        return "", None
    return state, found


def settle(
    path,
    previous_state="",
    accounting=None,
    submission=None,
    exits=None,
):
    """The cause one ``run.json`` ends with, after the scheduler has been heard: ``""``
    for a run that is ok, None when there is no record of this submission. A ``killed``
    placeholder whose task the scheduler ended for memory, for time or by losing it is
    REWRITTEN with that cause (``cause_from`` says where it came from). So is one whose
    end state names no cause while the task's script noted that Python ended by signal
    9 (``exits``, :func:`read_exits`): ``oom``.

    The state is that of the task that WROTE the record (``accounting[launch_id]``);
    ``previous_state`` stands in when the accounting does not have it. The rule does
    not ask which method, or which job of a method, wrote the placeholder."""
    from .cases import write_json

    rec = _read(path)
    if rec is None or not mine(rec, submission):
        return None
    if rec.get("ok"):
        return ""
    cause = str(rec.get("cause") or "exception")
    if cause != KILLED:
        return cause
    state, found = _end_state(rec, previous_state, accounting)
    new = scheduler_cause(state)
    how = f"scheduler state {state} (the driver's record said killed)"
    code = (exits or {}).get(str(rec.get("launch_id") or ""))
    if new is None and code == EXIT_SIGKILL:
        new = OOM
        how = (
            f"the task's own script saw Python end by signal 9 (exit {code}) and went "
            "on: the kernel's out-of-memory killer (the scheduler's state "
            f"{state or 'was not known'}; the driver's record said killed)"
        )
    if found or new:
        if new:
            rec["cause"] = cause = new
            rec["cause_from"] = how
        rec["scheduler"] = {
            "state": state,
            "elapsed_s": (found or {}).get("elapsed_s"),
            "cpu_s": (found or {}).get("cpu_s"),
        }
        write_json(path, rec)
    return cause


def unsettled(paths, previous_state="", accounting=None, submission=None, exits=None):
    """The ``killed`` records of this submission among ``paths`` that nothing settles
    yet: no end state of the task that wrote them, and no note of its script. Each is
    ``(path, launch id)``. These are what the rung task waits for the accounting for."""
    out = []
    for p in paths:
        rec = _read(p)
        if rec is None or not mine(rec, submission) or rec.get("ok"):
            continue
        if rec.get("cause") != KILLED or rec.get("no_rerun"):
            continue
        launch = str(rec.get("launch_id") or "")
        state, _ = _end_state(rec, previous_state, accounting)
        if not state and (exits or {}).get(launch) != EXIT_SIGKILL:
            out.append((Path(p), launch))
    return out


def mark_unsettled(paths, mem_gb=None):
    """Say in each record why it is run again: it still said ``killed`` and the
    accounting named no end state after the wait. The cause stays ``killed`` (it was
    never known); ``cause_from`` and ``scheduler.state`` (``""``) carry the rest into
    the next attempt's ``previous_attempts``."""
    from .cases import write_json

    for p in paths:
        rec = _read(p)
        if rec is None:
            continue
        rec["cause_from"] = (
            f"{NO_END_STATE} after the wait: the cause is not known, and the unit is "
            "run again"
            + (f" at {mem_gb:g} GB" if mem_gb else "")
            + " (the driver's record said killed)"
        )
        rec["scheduler"] = {"state": "", "elapsed_s": None, "cpu_s": None}
        write_json(p, rec)


LATE_LOST = "the retry of an attempt lost to the scheduler, not a climb for memory"


def mark_late_lost(paths, attempt, mem_gb=None, submission=None):
    """Say in each record lost to the scheduler why the next rung's ``climb`` task runs
    its unit: the accounting named the lost attempt after the rung's own retry had
    decided. The cause stays ``infrastructure``."""
    from .cases import write_json

    for p in paths:
        rec = _read(p)
        if rec is None or not mine(rec, submission) or not _late_lost(rec, attempt):
            continue
        rec["cause_from"] = (
            f"{rec.get('cause_from') or INFRASTRUCTURE}; found after this rung's retry "
            f"had decided, so the next rung's task runs the unit once more"
            + (f" at {mem_gb:g} GB" if mem_gb else "")
            + f": {LATE_LOST}"
        )
        write_json(p, rec)


def _late_lost(rec, attempt):
    """A failed record of an attempt below ``attempt`` that the scheduler lost and that
    was not tried again at its own rung."""
    return (
        not rec.get("ok")
        and not rec.get("no_rerun")
        and rec.get("cause") == INFRASTRUCTURE
        and int(rec.get("attempt") or 1) < attempt
        and int(rec.get("retry") or 0) < RETRIES_PER_RUNG
    )


def _records(paths, submission):
    """The records of this submission among ``paths``, read after settling."""
    out = []
    for p in paths:
        rec = _read(p)
        if rec is not None and mine(rec, submission):
            out.append(rec)
    return out


def unit_decision(
    paths,
    task=CLIMB,
    attempt=2,
    previous_state="",
    accounting=None,
    submission=None,
    exits=None,
    waited=False,
    mem_gb=None,
    late=True,
):
    """``(run the unit now?, why, lost?)`` for one array task of the chain.

    ``late``: a ``climb`` task also runs a unit whose attempt at the rung below the
    accounting named as lost only after that rung's retry had decided (module
    docstring). False asks about memory alone (:func:`unit_retry`).

    ``lost`` is True when this is a ``climb`` task and the unit has NO record of this
    submission: both tries of the rung below were lost before they could write one
    (the caller writes it down with :func:`mark_lost`).

    ``waited``: the accounting is as complete as it will be for this task (the caller
    waited for it, or there is none to ask). Then a record that still says ``killed``
    without an end state is not final: a ``climb`` task runs the unit again at its own
    rung (``mem_gb``, for the record)."""
    if task not in TASKS:
        raise ValueError(f"task {task!r}: one of {TASKS}")
    if task == FIRST:
        return True, "the first attempt", False
    for p in paths:
        settle(p, previous_state, accounting, submission, exits)
    unknown = [
        p for p, _ in unsettled(paths, previous_state, accounting, submission, exits)
    ]
    recs = _records(paths, submission)
    failed = [r for r in recs if not r.get("ok")]
    # `no_rerun`: the failure was retried where it happened (a stage job of its own)
    again = [r for r in failed if not r.get("no_rerun")]
    memory = [r for r in again if r.get("cause") in resources.MEMORY_CAUSES]
    here = [r for r in recs if int(r.get("attempt") or 1) >= attempt]
    if task == AGAIN:
        lost = [
            r
            for r in again
            if r.get("cause") == INFRASTRUCTURE
            and int(r.get("attempt") or 1) == attempt
            and int(r.get("retry") or 0) < RETRIES_PER_RUNG
        ]
        if lost:
            return (
                True,
                f"{len(lost)} result(s) of this rung were lost to the scheduler "
                f"({lost[0].get('cause_from') or INFRASTRUCTURE}): the same rung again",
                False,
            )
        if not recs and attempt == 1:
            return True, "no record of this submission: the unit never started", False
        if memory and not here:
            return (
                True,
                "the unit ran out of memory at the rung below and this rung's attempt "
                "left no record: it never started",
                False,
            )
        return False, "the attempt before was not lost to the scheduler", False
    # CLIMB
    if not recs:
        return (
            False,
            "no record of this submission after the rung below and its retry: lost "
            "to the scheduler, not re-run",
            True,
        )
    if memory:
        causes = sorted({r["cause"] for r in memory})
        return (
            True,
            f"{len(memory)} of {len(recs)} result(s) ran out of memory "
            f"({', '.join(causes)})",
            False,
        )
    late = [r for r in again if _late_lost(r, attempt)] if late else []
    if late:
        mark_late_lost(paths, attempt, mem_gb, submission)
        return (
            True,
            f"{len(late)} result(s) of the rung below were lost to the scheduler "
            f"({late[0].get('cause_from') or INFRASTRUCTURE}) and its retry had "
            f"decided before the accounting said so: run once at this rung, {LATE_LOST}",
            False,
        )
    if unknown and waited:
        mark_unsettled(unknown, mem_gb)
        return (
            True,
            f"{len(unknown)} result(s) still say killed and the scheduler's accounting "
            "named no end state after the wait: the cause is not known, the unit is "
            "run again at this rung's memory",
            False,
        )
    other = sorted({str(r.get("cause") or "exception") for r in failed})
    if other:
        return (
            False,
            f"failed for another reason ({', '.join(other)}): not re-run",
            False,
        )
    return False, "every result is ok", False


def unit_retry(
    paths, previous_state="", accounting=None, submission=None, exits=None, waited=False
):
    """``(go up?, why)`` for one unit, from the ``run.json`` of each of its results."""
    go, why, _ = unit_decision(
        paths,
        CLIMB,
        len(RUNGS_GB) + 1,
        previous_state,
        accounting,
        submission,
        exits,
        waited,
        late=False,
    )
    return go, why


def unit_heads(out, method, label, cases):
    """``{run.json path: identity}`` of every result one unit writes: each variant of
    each of its cases."""
    from . import methods
    from .cases import result_dir

    name = methods.method_name(method)
    unit = methods.UNIT[methods.scope_of(method)]
    heads = {}
    for case in cases:
        for variant in methods.variant_names(method, label):
            heads[result_dir(out, variant, case) / "run.json"] = {
                "method": name,
                "label": label or "",
                "variant": variant,
                "dataset": case.dataset,
                "case_id": case.case_id,
                "group": case.group,
                "group_cases": [c.case_id for c in cases],
                "unit": unit,
            }
    return heads


def unit_paths(out, method, label, cases):
    """Every ``run.json`` one unit writes: each variant of each of its cases."""
    return list(unit_heads(out, method, label, cases))


LOST = (
    "lost to the scheduler: no attempt of this unit left a record at the rung below or "
    "at its retry (node failure, preemption, a cancelled job?). Not the method's "
    "failure; submit the unit again"
)


SET_ASIDE = "earlier"


def set_aside(path):
    """Move a result's ``run.json`` and ``warped.npz`` into ``<result>/earlier/<n>/``
    (the first free ``n``) and return that folder, or None when there is no record.
    The image, the stage files and everything else of the result stay where they are:
    moving the two files back restores it."""
    path = Path(path)
    if not path.exists():
        return None
    n = 1
    while (path.parent / SET_ASIDE / str(n)).exists():
        n += 1
    dest = path.parent / SET_ASIDE / str(n)
    dest.mkdir(parents=True)
    for name in ("run.json", "warped.npz"):
        if (path.parent / name).exists():
            (path.parent / name).rename(dest / name)
    return dest


def mark_lost(heads, attempt, submission=None):
    """Write an ``infrastructure`` record for every result of a unit that has none of
    this submission. ``attempt`` is the rung that was lost. Returns how many.

    A record ANOTHER submission left there is not destroyed: it and its
    ``warped.npz`` are set aside (``set_aside``), and the new record names where
    (``earlier_result``, with that record's outcome). The row of THIS submission is
    ``infrastructure``: it ran nothing, so it has no result of its own, and an
    earlier ``ok`` one is not scored as this submission's."""
    from .cases import write_json

    n = 0
    for path, head in heads.items():
        rec = _read(path)
        if rec is not None and mine(rec, submission):
            continue
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        kept = set_aside(path)
        (path.parent / "warped.npz").unlink(missing_ok=True)
        new = {
            **head,
            "ok": False,
            "error": LOST
            + (
                f". An earlier result of this unit ({'ok' if rec.get('ok') else 'not ok'}"
                f", submissions {', '.join(submissions_of(rec)) or 'none'}) is kept "
                f"under {kept}"
                if kept and isinstance(rec, dict)
                else ""
            ),
            "cause": INFRASTRUCTURE,
            "cause_from": "no record of this submission",
            "attempt": int(attempt),
            "retry": RETRIES_PER_RUNG,
            "submission": submission or "",
            "submissions": [submission] if submission else [],
            "launch_id": "",
            "previous_attempts": [],
        }
        if kept:
            new["earlier_result"] = {
                "kept": str(kept),
                "ok": bool(isinstance(rec, dict) and rec.get("ok")),
                "submissions": submissions_of(rec),
            }
        write_json(path, new)
        n += 1
    return n


def settle_all(out, accounting, submission=None):
    """After everything: every ``run.json`` under ``out`` that still says ``killed`` is
    settled against the scheduler's accounting and the scripts' notes (the last rung
    has no later task to do it). Returns ``{cause: count}`` of the records it
    rewrote."""
    changed = {}
    exits = read_exits(out)
    for path in sorted(Path(out).glob("*/*/*/run.json")):
        rec = _read(path)
        if not rec or rec.get("ok") or rec.get("cause") != KILLED:
            continue
        cause = settle(path, "", accounting, submission, exits)
        if cause and cause != KILLED:
            changed[cause] = changed.get(cause, 0) + 1
    return changed


def selfcheck_retry(path, previous_state="", submission=None):
    """``(run again?, why)`` from an earlier attempt's ``selfcheck.json``.

    Yes when a pair or a result of it ended for memory: the driver's ``oom``, a pair's
    process killed outright, or a ``killed`` placeholder while the scheduler's state says
    out of memory. Yes also when there is NO record of this submission, unless the
    scheduler says the job ran out of time: the selfcheck removes the old record when it
    starts and writes its own last, so a job the cgroup limit ended, a job the scheduler
    lost and a job that never started all leave none. Any other record stands as it is.
    """
    state = _state(previous_state)
    rec = _read(path)
    if rec is not None and not mine(rec, submission):
        rec = None
    if rec is None:
        if state.startswith(STATE_TIMEOUT):
            return False, f"no selfcheck.json and the scheduler says {state}"
        return True, (
            "no selfcheck.json of this submission: the earlier attempt did not finish"
            + (f" (scheduler: {state})" if state else "")
        )
    for pair in rec.get("pairs") or []:
        if "killed" in str(pair.get("cause") or "") and not pair.get("variants"):
            return True, "a pair's process was killed (out of memory?)"
        for name, v in (pair.get("variants") or {}).items():
            cause = str((v or {}).get("cause") or "")
            if cause in resources.MEMORY_CAUSES or cause.startswith("oom:"):
                return True, f"{name} ran out of memory"
            if cause == INFRASTRUCTURE:
                return True, f"{name} was lost to the scheduler"
            if cause == KILLED and scheduler_cause(state) in (OOM, INFRASTRUCTURE):
                return True, f"{name} was killed and the scheduler says {state}"
    return False, f"the earlier attempt ended as {rec.get('status', '?')}: it stands"
