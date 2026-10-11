"""The memory ladder's decisions: does a unit go up a rung, and how long may a job run.

The cohort launcher (``slurm/submit_cohort.sh``) submits every rung of a run up front,
each waiting for the one before it (``afterany``). Nothing waits on a login node. So a
task of a later rung must decide FOR ITSELF, when it starts, whether its unit goes up: it
does when the unit's last attempt ran out of memory, and exits at once otherwise. That
decision is here, in Python, so that it can be tested without a scheduler.

**Evidence of an out-of-memory end** is the driver's own record, ``run.json: cause``:

``oom``      the unit goes up a rung.
``killed``   the placeholder the driver writes before it starts; it is still there when
             the process was killed before it could say why. Then, and only then, the
             scheduler's state for the previous rung's task decides
             (``--previous-state``): ``OUT_OF_MEMORY`` makes it ``oom`` and ``TIMEOUT``
             makes it ``timeout``. The record is rewritten to say so, with
             ``cause_from`` naming the scheduler state, so the scorer counts the
             attempt under its real cause.
anything else (``ok``, ``exception``, ``timeout``, ``gpu_oom``, no record at all): the
             unit is never re-run. A unit with no ``run.json`` never started; more
             memory is not what it lacked.

A unit is one case for a method that runs per pair and one patient for a method that runs
per patient; a patient goes up as a whole when any of its rounds ran out of memory.

The selfcheck climbs too (ruling R3.5): 64, then 256, then 512 GB. A later attempt runs
only when the earlier one ended for memory (``selfcheck_retry``).

Frozen: the rungs, the CPUs and the site's longest job. Changing one after a cohort run
is a post hoc change.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNGS_GB = (16, 64, 256, 512)
SELFCHECK_RUNGS_GB = (64, 256, 512)
CPUS = 4  # every method, every rung
MAX_HOURS = 168  # the site's longest job: 7 days

OOM = "oom"
KILLED = "killed"
# sacct's State for a job the cgroup limit ended, and for one that ran out of time; a
# state is matched by its first word ("CANCELLED by 123", "OUT_OF_MEMORY+" are seen)
STATE_OOM = "OUT_OF_ME"
STATE_TIMEOUT = "TIMEOUT"


def _state(previous_state):
    return str(previous_state or "").strip().upper()


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def settle(path, previous_state=""):
    """The cause one ``run.json`` ends with, after the scheduler's state has been heard:
    ``""`` for a run that is ok, None when there is no record. A ``killed`` placeholder
    whose task the scheduler ended for memory or for time is REWRITTEN with that cause
    (``cause_from`` says where it came from)."""
    rec = _read(path)
    if rec is None:
        return None
    if rec.get("ok"):
        return ""
    cause = str(rec.get("cause") or "exception")
    state = _state(previous_state)
    if cause == KILLED and state:
        new = (
            OOM
            if state.startswith(STATE_OOM)
            else "timeout"
            if state.startswith(STATE_TIMEOUT)
            else None
        )
        if new:
            rec["cause"] = cause = new
            rec["cause_from"] = (
                f"scheduler state {state} (the driver's record said killed)"
            )
            Path(path).write_text(json.dumps(rec, indent=1))
    return cause


def unit_retry(paths, previous_state=""):
    """``(go up?, why)`` for one unit, from the ``run.json`` of each of its results."""
    causes = {str(p): settle(p, previous_state) for p in paths}
    found = [c for c in causes.values() if c is not None]
    if not found:
        return False, "no run.json: the unit never started, nothing to retry"
    if OOM in found:
        n = sum(c == OOM for c in found)
        return True, f"{n} of {len(found)} result(s) ran out of memory"
    failed = sorted({c for c in found if c})
    if failed:
        return False, f"failed for another reason ({', '.join(failed)}): not re-run"
    return False, "every result is ok"


def unit_paths(out, method, label, cases):
    """Every ``run.json`` one unit writes: each variant of each of its cases."""
    from . import methods
    from .cases import result_dir

    return [
        result_dir(out, variant, case) / "run.json"
        for case in cases
        for variant in methods.variant_names(method, label)
    ]


def selfcheck_retry(path, previous_state=""):
    """``(run again with more memory?, why)`` from an earlier attempt's
    ``selfcheck.json``.

    Yes when a pair or a result of it ended for memory: the driver's ``oom``, a pair's
    process killed outright, or a ``killed`` placeholder while the scheduler's state says
    out of memory. Yes also when there is NO record and the scheduler does not say the
    job ran out of time or was cancelled: the selfcheck writes its record last, and a job
    the cgroup limit ended leaves none. Any other record stands as it is.
    """
    state = _state(previous_state)
    rec = _read(path)
    if rec is None:
        if state and not state.startswith(STATE_OOM):
            return False, f"no selfcheck.json and the scheduler says {state}"
        return True, "no selfcheck.json: the earlier attempt did not finish"
    for pair in rec.get("pairs") or []:
        if "killed" in str(pair.get("cause") or "") and not pair.get("variants"):
            return True, "a pair's process was killed (out of memory?)"
        for name, v in (pair.get("variants") or {}).items():
            cause = str((v or {}).get("cause") or "")
            if cause == OOM or cause.startswith("oom:"):
                return True, f"{name} ran out of memory"
            if cause == KILLED and state.startswith(STATE_OOM):
                return True, f"{name} was killed and the scheduler says {state}"
    return False, f"the earlier attempt ended as {rec.get('status', '?')}: it stands"
