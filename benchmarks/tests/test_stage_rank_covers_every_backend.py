"""_STAGE_RANK must rank every stage any backend can emit.

The reduction in `registration_accuracy_per_run` picks each run's "final, most-registered"
stage by mapping `stage` through `_STAGE_RANK` and taking the last row after a sort. An
unranked stage maps to -1 via `.fillna(-1)` -- the SAME rank as `native`, which is also
unranked when missing -- so the terminal stage is then chosen by nothing more than the
stable-sort tie-break.

That was not hypothetical: `refined` (STARE's terminal stage, and now ashlar's) was absent,
so every STARE run in the paper tables was one upstream reordering away from reporting its
UNREGISTERED accuracy as its headline number. Nothing failed, and nothing would have.

This guard reads the stage vocabularies out of lib/WarpBackends.groovy rather than restating
them, so adding a fourth backend with a new stage name fails here instead of silently
degrading the table.
"""

from __future__ import annotations

import re
from pathlib import Path

from benchmarks.analysis.lib.quality import _STAGE_RANK, FULL_TRANSFORM_STAGE

REPO = Path(__file__).resolve().parents[2]
WARP_BACKENDS = REPO / "lib" / "WarpBackends.groovy"

_STAGES_LINE = re.compile(r"stages\s*:\s*\[([^\]]*)\]")


def backend_stages() -> set[str]:
    """Every stage name any WarpBackends entry declares."""
    text = WARP_BACKENDS.read_text()
    stages: set[str] = set()
    for block in _STAGES_LINE.findall(text):
        stages |= {s.strip().strip("'\"") for s in block.split(",") if s.strip()}
    return stages


def test_the_scan_actually_finds_the_backend_stages():
    """A guard whose regex silently matched nothing would pass forever."""
    found = backend_stages()
    assert len(found) >= 4, (
        f"only found {sorted(found)} in {WARP_BACKENDS.name} -- the `stages : [...]` scan "
        "pattern is stale, so the coverage assertion below is checking nothing."
    )
    # The two vocabularies that exist today, as a sanity anchor on the parse.
    assert {"native", "rigid", "non_rigid", "micro", "refined"} <= found


def test_every_backend_stage_is_ranked():
    unranked = sorted(backend_stages() - set(_STAGE_RANK))
    assert not unranked, (
        f"stages emitted by a WarpBackends entry but absent from _STAGE_RANK: {unranked}. "
        "They map to -1, tie with `native`, and the 'final stage' reduction then picks the "
        "terminal stage only by stable-sort accident. Add each with the rank that says how "
        "much registration it represents."
    )


def test_no_stale_rank_entries():
    """The other direction: a rank for a stage nothing emits is a stale claim.

    `full_transform` is the one entry that is not a backend stage: it is the scorer's
    final-stage record PAIRED AFTER the whole transform (bin/warp_seg_qc.py), which the
    harvest turns into a row of its own.
    """
    stale = sorted(set(_STAGE_RANK) - backend_stages() - {FULL_TRANSFORM_STAGE})
    assert not stale, (
        f"_STAGE_RANK ranks stages no backend emits: {stale}. Remove them, or the table "
        "documents a vocabulary the pipeline does not have."
    )


def test_native_is_the_lowest_and_terminal_stages_outrank_rigid():
    """The ranks must encode 'how much registration has been applied', not just be distinct."""
    assert _STAGE_RANK["native"] == min(_STAGE_RANK.values())
    for terminal in ("micro", "refined"):
        assert _STAGE_RANK[terminal] > _STAGE_RANK["rigid"], (
            f"{terminal} must outrank rigid, or the reduction reports the rigid stage as the "
            "headline accuracy for the backend whose terminal stage it is."
        )


def test_the_full_transform_row_outranks_every_backend_stage():
    """It is the headline: every 'final stage' reduction must pick it when it exists."""
    others = [r for s, r in _STAGE_RANK.items() if s != FULL_TRANSFORM_STAGE]
    assert _STAGE_RANK[FULL_TRANSFORM_STAGE] > max(others)
