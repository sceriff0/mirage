"""The full pairings beside the headline (Task 21b): full LSA and full MNN.

USER 2026-10-18: "I would like score to also test full lsa and full mnn". The headline is
unchanged: it is the pipeline's pairing (``bin/utils/cell_pairs.py``: an assignment per
connected component of the candidate pairs, mutual nearest neighbour in a component of
more than ``cell_pairs.DEFAULT_MAX_COMPONENT_CELLS`` cells; ``fallback_frac`` of a record
is the share of a case's cells that sat in such a component). Beside it:

``full_lsa``  the exact minimum-weight assignment (most pairs first, then the least total
              distance: the pipeline's objective) in EVERY component, with no fallback and
              no work cap: ``pairs.exact_pairs(..., max_work=math.inf)``. The pair
              record's own ``*_exact`` is the same assignment under the bound
              ``pairs.EXACT_MAX_WORK``, so the two differ only where the bound was hit.
              Hence a record whose ``exact_status`` is ``EXACT_SAME`` or ``EXACT_SOLVED``
              is COPIED (``FROM_EXACT``: no solve), and only a record left blank at the
              bound (``EXACT_TOO_LARGE``), or one made before the exact columns existed,
              is solved here, from the same polygon sets and the same gate
              (``match_radius_px`` of the record). A record the exact solver did not
              certify (``EXACT_NOT_CERTIFIED``) is not solved again: the same solver on
              the same graph gives the same uncertified result, so it is
              ``NOT_CERTIFIED`` here too.
``full_mnn``  mutual nearest neighbour on the whole case: the record's own ``*_mnn``
              columns (``cell_pairs.match_cells(method="mutual_nn")`` is uncapped), never
              recomputed. ``score`` reports them under this name.

The cost of a solve is recorded, never capped: ``full_lsa_seconds``,
``full_lsa_largest_component_cells`` (the largest component of the candidate graph) and
``full_lsa_work`` (``pairs.exact_work`` of the costliest component the fallback took, in
the bound's unit). There is no timeout and no approximation. A result the solver does
not certify has NaN numbers and status ``NOT_CERTIFIED``; a solve that raises (memory,
anything) has NaN numbers and status ``FAILED`` and is not kept, so the next scoring
tries it again. Either is a row of ``losses.csv`` (``LOSS_CAUSE``), never a crash of the
scorer and never a dropped headline row.

Only the registered placement has these numbers: like ``*_exact``, the null's eight
shifted placements have none.

The records are ``score``'s, kept beside the pair records in a folder of their own
(``RECORDS``) under a key of their own (``score.full_key``: the pair record's key and the
sha256 of this file and the package modules it imports that are not already pair code),
so a change here makes the full records again and no pair record.
"""

from __future__ import annotations

import math
import time

import numpy as np

from . import pairs

RECORDS = "full_pairs"
# the scores that depend on which cells are paired (``pairs._EXACT_KEYS``' stems)
SCORED = tuple(k for k in pairs._SCORE_KEYS if k not in ("n_ref", "n_moving"))
KEYS = tuple(f"{k}_full_lsa" for k in SCORED)
ROW_KEYS = (
    *KEYS,
    "full_lsa_status",
    "full_lsa_detail",
    "full_lsa_seconds",
    "full_lsa_largest_component_cells",
    "full_lsa_work",
)
FROM_EXACT = "full_lsa_from_exact"
SOLVED = "full_lsa_solved"
NOT_CERTIFIED = "not_certified"
FAILED = "failed"
# what a scoring keeps as a record: a failure is tried again
KEPT = (FROM_EXACT, SOLVED, NOT_CERTIFIED)
# the losses.csv cause of a full-LSA number that is not reported
LOSS_CAUSE = {
    NOT_CERTIFIED: "full_lsa_not_certified",
    FAILED: "full_lsa_failed",
}

_TOO_LARGE = pairs.EXACT_TOO_LARGE.split("{")[0]
_NOT_CERTIFIED = pairs.EXACT_NOT_CERTIFIED.split("{")[0]


def needs_solve(pair_rec):
    """True where the full LSA is not the record's own ``*_exact``: the exact solver
    stopped at its bound, or the record has no exact columns at all."""
    if "exact_status" not in pair_rec:
        return True
    return str(pair_rec["exact_status"]).startswith(_TOO_LARGE)


def _blank(status, detail=""):
    out = {k: math.nan for k in KEYS}
    out.update(
        full_lsa_status=status,
        full_lsa_detail=detail,
        full_lsa_seconds=math.nan,
        full_lsa_largest_component_cells=math.nan,
        full_lsa_work=math.nan,
    )
    return out


def from_exact(pair_rec):
    """The record's own exact numbers, as the full LSA (no solve)."""
    said = str(pair_rec.get("exact_status", ""))
    if said.startswith(_NOT_CERTIFIED):
        return _blank(NOT_CERTIFIED, said)
    out = _blank(FROM_EXACT, said)
    out.update({f"{k}_full_lsa": pair_rec.get(f"{k}_exact", math.nan) for k in SCORED})
    out["full_lsa_largest_component_cells"] = pair_rec.get(
        "largest_component_cells", math.nan
    )
    return out


def _components(cent_a, cent_b, radius):
    """``(largest component's cells, work of the costliest component the pipeline's
    pairing leaves to its fallback)``, from the candidate graph as the pipeline builds
    it (the same k-d tree query and components as ``pairs.exact_pairs``)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    a, _ = pairs.cp._finite_rows(cent_a)
    b, _ = pairs.cp._finite_rows(cent_b)
    na, nb = len(a), len(b)
    if not na or not nb:
        return 0, 0.0
    found = cKDTree(a).query_ball_tree(cKDTree(b), radius)
    cols = np.fromiter((j for js in found for j in js), dtype=np.int64)
    rows = np.repeat(np.arange(na, dtype=np.int64), [len(js) for js in found])
    adj = coo_matrix((np.ones(cols.size), (rows, na + cols)), shape=(na + nb, na + nb))
    _, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels)
    side_a = np.bincount(labels[:na], minlength=len(sizes))
    large = sizes > pairs.cp.DEFAULT_MAX_COMPONENT_CELLS
    work = [
        pairs.exact_work(int(side_a[k]), int(sizes[k] - side_a[k]))
        for k in np.flatnonzero(large)
    ]
    return int(sizes.max()), float(max(work, default=0.0))


def full_lsa_pairs(cent_a, cent_b, radius):
    """``(idx_a, idx_b, dist, why, largest component cells, work)``: the assignment in
    every component, no fallback, no bound. ``idx_a`` is None when it is not certified
    (``why`` says so)."""
    ia, ib, _, stats = pairs.cp.match_cells(
        cent_a,
        cent_b,
        radius,
        method="lsa",
        max_component_cells=pairs.cp.DEFAULT_MAX_COMPONENT_CELLS,
    )
    largest, work = _components(cent_a, cent_b, radius)
    ea, eb, ed, why = pairs.exact_pairs(
        cent_a, cent_b, radius, ia, ib, stats, max_work=math.inf
    )
    return ea, eb, ed, why, largest, work


def solve(ref_in, mov_in, radius, pixel_size_um, geometry=None):
    """The full-LSA numbers of one placement: ``ref_in``, ``mov_in`` in one frame,
    ``radius`` the record's gate."""
    started = time.perf_counter()
    geo = pairs.Geometry(geometry)
    a, b = pairs.as_polyset(ref_in), pairs.as_polyset(mov_in)
    out = _blank(SOLVED)
    if not a.n_features or not b.n_features or not radius > 0:
        out.update(n_pairs_full_lsa=0, n_pairs_scored_full_lsa=0)
        out["pair_fraction_full_lsa"] = 0.0
        out.update(full_lsa_seconds=time.perf_counter() - started)
        out.update(full_lsa_largest_component_cells=0, full_lsa_work=0.0)
        return out
    _, cent_a = pairs.area_centroid(a, geo)
    _, cent_b = pairs.area_centroid(b, geo)
    ea, eb, ed, why, largest, work = full_lsa_pairs(cent_a, cent_b, radius)
    if ea is None:
        out = _blank(NOT_CERTIFIED, why)
    else:
        got = pairs._pair_scores(a, b, ea, eb, ed, pixel_size_um, None, geo)
        out.update({f"{k}_full_lsa": v for k, v in got.items()})
        out["full_lsa_detail"] = why
    out.update(
        full_lsa_seconds=time.perf_counter() - started,
        full_lsa_largest_component_cells=largest,
        full_lsa_work=work,
    )
    return out


def full_record(pair_rec, inputs):
    """The full-LSA columns (``ROW_KEYS``) of one pair record. ``inputs()`` gives
    ``(ref native, mov native, ref in frame, mov in frame, pixel size, geometry)``, the
    sets the pair record was made from; it is called only when a solve is needed.
    Never raises: a failure is ``FAILED`` with the exception in ``full_lsa_detail``."""
    if not needs_solve(pair_rec):
        return from_exact(pair_rec)
    try:
        ref, mov, ref_in, mov_in, px, geometry = inputs()
        radius = pair_rec.get("match_radius_px")
        if radius is None:
            radius = pairs.gate_px(ref, mov, pairs.Geometry(geometry))[0]
        return solve(ref_in, mov_in, float(radius), px, geometry)
    except Exception as exc:  # memory, anything: a loss row, never a crash
        return _blank(FAILED, f"{type(exc).__name__}: {exc}"[:300])
