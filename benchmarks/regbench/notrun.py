"""What a landmark scorer does with a pair that was never registered for a reason that is
not the method's: the one definition, shared by ANHIR's scorer (``challenge``) and
HyReCo's (``landmarks``).

Restored from commit 5a46f778 (``benchmarks/regbench/notrun.py``) without
``blocked_rows``: the gates it read are gone (every check is reported, never a gate).

A pair whose task never started, whose attempt was lost to the scheduler or to the
benchmark's own step, or whose result was made for other landmarks or another version
of the case (``STALE_TRUTH``, an integrity refusal) is in NO median of the method (it
says nothing about it): it is counted (``n_not_run``), listed (``*_not_run.csv``) and a
row of ``losses.csv``. A pair the method itself failed on counts at its unregistered
landmarks, as the challenge counts a missing registration. The causes are the cohort
scorer's (``score.NOT_THE_METHOD``) and ``not_run``.
"""

from __future__ import annotations

from . import score as cohort
from .mapping import FRAME, MOV_TO_REF

NOT_RUN_CAUSES = ("not_run", *cohort.NOT_THE_METHOD)
UNKNOWN_FRAME = "unknown (no pair of this row registered)"
# the integrity refusal: a result whose run.json names other landmarks (truth_sha256)
STALE_TRUTH = (
    "made for other landmarks: the result's truth_sha256 is not the case's truth.npz "
    "as it is now (the case was prepared again): not run on the case as it is"
)


def settle_frames(pairs):
    """Fill in what a failed row cannot know by itself, in place: its frame.

    A pair that did not register has no map; its row takes the frame of the same result
    name's registered pairs (a method maps one way), else the direction its run record
    declares, else ``UNKNOWN_FRAME``. Never a default: a reference -> moving method's
    failed pair is not "the challenge's own quantity". Returns ``{result name: frames}``."""
    frames = {}
    for r in pairs:
        if not r["failed"] and r["frame"]:
            frames.setdefault(r["method"], set()).add(r["frame"])
    reference = FRAME[MOV_TO_REF]
    for r in pairs:
        if not r["failed"]:
            continue
        known = frames.get(r["method"]) or ({r["frame"]} if r["frame"] else set())
        r["frame"] = "+".join(sorted(known)) if known else UNKNOWN_FRAME
        if "challenge_quantity" in r:
            r["challenge_quantity"] = known == {reference}
    return {
        m: sorted({r["frame"] for r in pairs if r["method"] == m})
        for m in dict.fromkeys(r["method"] for r in pairs)
    }


def not_run_rows(pairs):
    """One row per result name with pairs that were never registered for a reason that
    is not the method's (``<set>_not_run.csv``), by cause."""
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
