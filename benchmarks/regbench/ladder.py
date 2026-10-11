"""The memory ladder's decisions: does a unit go up a rung, is a lost attempt tried again,
and how long may a job run.

The cohort launcher (``slurm/submit_cohort.sh``) submits every job of a run up front,
each waiting for the one before it (``afterany``). Nothing waits on a login node. So a
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
             scheduler's state for the task THAT WROTE THE RECORD decides (matched by
             ``launch_id`` in ``--accounting``, or ``--previous-state`` for the task
             just before): ``OUT_OF_MEMORY`` makes it ``oom``, ``TIMEOUT`` makes it
             ``timeout``, and a state of :data:`STATES_INFRASTRUCTURE` (node failure,
             preemption, a cancelled job, a failed boot) makes it ``infrastructure``.
             The record is rewritten to say so, with ``cause_from`` naming the state and
             ``scheduler`` holding the task's elapsed and CPU time, so the attempt's
             cost is the scheduler's figure and not zero. Any other state leaves it
             ``killed``: a process that died on its own is the method's failure.
``infrastructure``  the ``again`` task of the same rung runs the unit once more. ONE
             retry per rung: an attempt lost twice stays ``infrastructure``, which the
             tables count apart from the method's failures.
no record of this submission   the task that should have written it never started. At
             rung 1 the ``again`` task runs the unit; at a higher rung it does when the
             unit's last record is a memory cause of the rung below (the ``climb`` task
             was lost before its placeholder). A unit still without a record after
             both is written down as ``infrastructure`` by the next ``climb`` task
             (:func:`mark_lost`), so that it is never read as "not run" or as failed.
anything else (``ok``, ``exception``, ``timeout``, ``gpu_oom``, ``writer``): the unit is
             never re-run.

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
# sacct's State, matched by its first word ("CANCELLED by 123", "OUT_OF_MEMORY+" are
# seen). SLURM `sacct` manual, JOB STATE CODES `[MEMORY]`: the names as remembered, not
# retrieved while this was written; README "Not verified" lists them.
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
    ``sacct -n -X -P -o JobID,State,ElapsedRaw,TotalCPU``: one ``|``-separated line per
    job or array task (``<array job>_<task>``, which is also a record's ``launch_id``).
    Lines that do not parse are skipped."""
    out = {}
    for line in str(text or "").splitlines():
        fields = [f.strip() for f in line.split("|")]
        if len(fields) < 2 or not fields[0] or not fields[1]:
            continue
        out[fields[0]] = {
            "state": _state(fields[1]),
            "elapsed_s": _seconds(fields[2]) if len(fields) > 2 else None,
            "cpu_s": _seconds(fields[3]) if len(fields) > 3 else None,
        }
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


def mine(rec, submission):
    """Does a record belong to this submission? Always, when there is no submission id
    (a run outside the cohort launcher). The driver adds the id of a submission that
    reuses an ``ok`` result to ``submissions``."""
    if not submission:
        return True
    if not isinstance(rec, dict):
        return False
    known = list(rec.get("submissions") or [])
    if rec.get("submission"):
        known.append(rec["submission"])
    return submission in known


def settle(path, previous_state="", accounting=None, submission=None):
    """The cause one ``run.json`` ends with, after the scheduler has been heard: ``""``
    for a run that is ok, None when there is no record of this submission. A ``killed``
    placeholder whose task the scheduler ended for memory, for time or by losing it is
    REWRITTEN with that cause (``cause_from`` says where it came from).

    The state is that of the task that WROTE the record (``accounting[launch_id]``);
    ``previous_state`` stands in when the accounting does not have it."""
    rec = _read(path)
    if rec is None or not mine(rec, submission):
        return None
    if rec.get("ok"):
        return ""
    cause = str(rec.get("cause") or "exception")
    if cause != KILLED:
        return cause
    found = (accounting or {}).get(str(rec.get("launch_id") or ""))
    state = found["state"] if found else _state(previous_state)
    new = scheduler_cause(state)
    if found or new:
        if new:
            rec["cause"] = cause = new
            rec["cause_from"] = (
                f"scheduler state {state} (the driver's record said killed)"
            )
        rec["scheduler"] = {
            "state": state,
            "elapsed_s": (found or {}).get("elapsed_s"),
            "cpu_s": (found or {}).get("cpu_s"),
        }
        Path(path).write_text(json.dumps(rec, indent=1))
    return cause


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
):
    """``(run the unit now?, why, lost?)`` for one array task of the chain.

    ``lost`` is True when this is a ``climb`` task and the unit has NO record of this
    submission: both tries of the rung below were lost before they could write one
    (the caller writes it down with :func:`mark_lost`)."""
    if task not in TASKS:
        raise ValueError(f"task {task!r}: one of {TASKS}")
    if task == FIRST:
        return True, "the first attempt", False
    for p in paths:
        settle(p, previous_state, accounting, submission)
    recs = _records(paths, submission)
    failed = [r for r in recs if not r.get("ok")]
    memory = [r for r in failed if r.get("cause") in resources.MEMORY_CAUSES]
    here = [r for r in recs if int(r.get("attempt") or 1) >= attempt]
    if task == AGAIN:
        lost = [
            r
            for r in failed
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
    other = sorted({str(r.get("cause") or "exception") for r in failed})
    if other:
        return (
            False,
            f"failed for another reason ({', '.join(other)}): not re-run",
            False,
        )
    return False, "every result is ok", False


def unit_retry(paths, previous_state="", accounting=None, submission=None):
    """``(go up?, why)`` for one unit, from the ``run.json`` of each of its results."""
    go, why, _ = unit_decision(
        paths, CLIMB, len(RUNGS_GB) + 1, previous_state, accounting, submission
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


def mark_lost(heads, attempt, submission=None):
    """Write an ``infrastructure`` record for every result of a unit that has none of
    this submission. ``attempt`` is the rung that was lost. Returns how many."""
    n = 0
    for path, head in heads.items():
        rec = _read(path)
        if rec is not None and mine(rec, submission):
            continue
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "warped.npz").unlink(missing_ok=True)
        new = {
            **head,
            "ok": False,
            "error": LOST,
            "cause": INFRASTRUCTURE,
            "cause_from": "no record of this submission",
            "attempt": int(attempt),
            "retry": RETRIES_PER_RUNG,
            "submission": submission or "",
            "submissions": [submission] if submission else [],
            "launch_id": "",
            "previous_attempts": [],
        }
        path.write_text(json.dumps(new, indent=1))
        n += 1
    return n


def settle_all(out, accounting, submission=None):
    """After everything: every ``run.json`` under ``out`` that still says ``killed`` is
    settled against the scheduler's accounting (the last rung has no later task to do
    it). Returns ``{cause: count}`` of the records it rewrote."""
    changed = {}
    for path in sorted(Path(out).glob("*/*/*/run.json")):
        rec = _read(path)
        if not rec or rec.get("ok") or rec.get("cause") != KILLED:
            continue
        cause = settle(path, "", accounting, submission)
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
