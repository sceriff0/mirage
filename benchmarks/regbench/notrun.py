"""What a scorer does with a pair that was never registered for a reason that is not the
method's: the one definition, shared by ANHIR's scorer (``challenge``) and HyReCo's
(``landmarks``).

A pair whose run was blocked by a gate, whose task never started, or whose attempt was
lost to the scheduler or to the benchmark's own step is in NO median of the method (it
says nothing about it); it is counted (``n_not_run``) and listed (``*_not_run.csv``). A
pair the method itself failed on counts at its unregistered landmarks. The causes are
``failure_log.WHOSE``'s launcher / infrastructure / benchmark ones (a test walks them).
"""

from __future__ import annotations

from . import resources, variants
from . import score as cohort
from .mapping import FRAME, MOV_TO_REF

# A pair that was never registered for a reason that is not the method's: its run was
# blocked by a gate, its task never started, or the attempt was lost to the scheduler or
# to the benchmark's own step. Such a pair is in NO median of the method (it says
# nothing about it); it is counted (`n_not_run`) and listed (`anhir_not_run.csv`,
# `hyreco_not_run.csv`). A pair the method itself failed on counts at its unregistered
# landmarks, as the challenge counts a missing registration.
NOT_RUN_CAUSES = ("not_run", "gate_blocked", *resources.NOT_THE_METHOD)
UNKNOWN_FRAME = "unknown (no pair of this row registered)"


def settle_frames(pairs, blocked=None):
    """Fill in what a failed row cannot know by itself, in place.

    **Frame.** A pair that did not register has no map; its row takes the frame of the
    same result name's registered pairs (a method maps one way), else the direction its
    run record declares, else ``UNKNOWN_FRAME``. Never a default: a reference -> moving
    method's failed pair is not "the challenge's own quantity".

    **Blocked.** ``blocked`` is ``{result name: why}`` for the rows of a run its gate did
    not let start: their pairs are ``not_run`` with cause ``gate_blocked``, whatever is on
    disk from an earlier submission. Returns ``{result name: frames}``."""
    frames = {}
    for r in pairs:
        if not r["failed"] and r["frame"]:
            frames.setdefault(r["method"], set()).add(r["frame"])
    reference = FRAME[MOV_TO_REF]
    for r in pairs:
        why = (blocked or {}).get(r["method"])
        if why and r["method"] != "initial":
            r.update(failed=True, not_run=True, cause="gate_blocked", problem=why[:200])
            r["robustness"] = 0.0
        if not r["failed"]:
            continue
        known = frames.get(r["method"]) or ({r["frame"]} if r["frame"] else set())
        r["frame"] = "+".join(sorted(known)) if known else UNKNOWN_FRAME
        r["challenge_quantity"] = known == {reference}
    return {
        m: sorted({r["frame"] for r in pairs if r["method"] == m})
        for m in dict.fromkeys(r["method"] for r in pairs)
    }


def not_run_rows(pairs):
    """One row per result name with pairs that were never registered for a reason that
    is not the method's (``anhir_not_run.csv``), by cause."""
    out = []
    for m in dict.fromkeys(r["method"] for r in pairs):
        mine = [r for r in pairs if r["method"] == m]
        for cause in dict.fromkeys(r["cause"] for r in mine if r.get("not_run")):
            hit = [r for r in mine if r.get("not_run") and r["cause"] == cause]
            out.append(
                {
                    "method": m,
                    "cause": cause,
                    "n_pairs": len(hit),
                    "of": len(mine),
                    "in_no_median": True,
                    "why": hit[0]["problem"],
                    "pairs": " ".join(r["case_id"] for r in hit[:20])
                    + (" ..." if len(hit) > 20 else ""),
                }
            )
    return out


def blocked_rows(gates):
    """``{result name: why}`` for every row of a run a gate of this submission blocked
    (``<out>/logs/gates/*.tsv``, as the cohort's scorer reads them)."""
    out = {}
    for g in cohort.read_gates(gates):
        if not g["blocked"]:
            continue
        try:
            run = variants.run_of(g["run"])
        except KeyError:
            continue
        for v in variants.variants_of(run):
            out[v] = (
                f"the run {g['run']} was blocked by its {g['gate']} gate "
                f"({g['status']}): no pair was registered"
            )
    return out
