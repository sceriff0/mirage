"""Cell scores for one (case, variant): mirage's pairing and IoU, a fixed gate, a null and a
pairing-free curve.

The pairing and the per-pair IoU are ``bin/utils/cell_pairs.py``, imported, so the benchmark
cannot drift from the pipeline's reg_qc=2 scorer. What is added here:

- the gate is sized ONCE per case from the two NATIVE sets of nuclei, so every method is
  judged with the same gate (the pipeline sizes it from the warped cells);
- the two sets are compared in whichever frame the method's transform leads to;
- ``_null``: the same score with the moving side shifted ``NULL_SHIFT_RADII`` nuclear radii,
  averaged over ``NULL_DIRECTIONS`` directions: what chance pairing gives in that tissue.
  One direction is not enough (rows of nuclei pair along the row and not across it);
- ``pair_fraction_excess``: pair fraction above the null, as a share of the room above it;
- ``within_<x>um``: the share of nuclei with a partner centroid within x um, computed from
  each side (``_ref``, ``_mov``) and reported as the smaller. One side alone is flattered by
  a map that piles nuclei onto one place; the smaller of the two is not. Read it beside
  ``count_ratio``, the smaller cell count over the larger.

Nothing is computed twice on the same polygons: ``Geometry`` keeps the area and centroid
of every set a record touches (the native sets for a whole case, when the caller hands
one in), and it keeps them by the set's identity, so a cached value is the value the
pipeline's function returned for exactly those vertices. No ``*_null`` number is derived
from another: each shifted placement has its own centroids, pairing and IoU.

``cell_pairs.py`` is loaded as ONE file. Putting ``bin/utils`` on ``sys.path`` would expose
its other modules (``logger``, ``retry``, ``progress``, ...) as top-level names inside the
method images, ahead of the methods' own.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import pathlib
import sys

import numpy as np

CELL_PAIRS_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "bin" / "utils" / "cell_pairs.py"
)


def _load_cell_pairs():
    """The pipeline's ``cell_pairs`` module, registered as ``sys.modules["cell_pairs"]``."""
    loaded = sys.modules.get("cell_pairs")
    if loaded is not None:
        origin = pathlib.Path(getattr(loaded, "__file__", "") or "").resolve()
        if origin != CELL_PAIRS_PATH:
            raise ImportError(
                f"a different `cell_pairs` is already imported ({origin}); "
                f"the benchmark scores with {CELL_PAIRS_PATH} only"
            )
        return loaded
    spec = importlib.util.spec_from_file_location("cell_pairs", CELL_PAIRS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cell_pairs"] = module  # before exec: its dataclass looks itself up
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules["cell_pairs"]
        raise
    return module


cp = _load_cell_pairs()

GATE_FACTOR = 1.5
NULL_SHIFT_RADII = 4.0
NULL_DIRECTIONS = 8
CURVE_UM = (1.0, 2.0, 3.0, 5.0)
IOU_THRESH = 0.5
# displacement is read only where pair fraction clears the null by this
READABLE_MARGIN = 0.10
_COUNT_KEYS = ("n_ref", "n_moving", "n_pairs", "n_pairs_scored", "pair_fraction")
_NAN_KEYS = (
    "dice_matched",
    "iou_p50",
    "frac_iou_ge_0.5",
    "displacement_px_p50",
    "displacement_px_p90",
    "displacement_um_p50",
    "displacement_um_p90",
)
_SCORE_KEYS = (*_COUNT_KEYS, *_NAN_KEYS)
_CURVE_KEYS = tuple(
    f"within_{t:g}um{side}" for t in CURVE_UM for side in ("", "_ref", "_mov")
)
RECORD_KEYS = (
    "match_radius_px",
    "median_cell_radius_px",
    "count_ratio",
    *_SCORE_KEYS,
    *(f"{k}_mnn" for k in _SCORE_KEYS),
    *(f"{k}_null" for k in _SCORE_KEYS),
    *(f"{k}_mnn_null" for k in _SCORE_KEYS),
    *_CURVE_KEYS,
    *(f"{k}_null" for k in _CURVE_KEYS),
    "pair_fraction_null_min",
    "pair_fraction_null_max",
    "pair_fraction_excess",
    "displacement_readable",
    "n_unmapped",
    "unmapped_frac",
)


def as_polyset(ps):
    """A ``cell_pairs.PolySet`` as is; a regbench ``Polys`` (one outer ring per cell) converted."""
    if isinstance(ps, cp.PolySet):
        return ps
    k = len(ps)
    return cp.PolySet(
        xy=np.asarray(ps.xy, dtype=float).reshape(-1, 2),
        ring_off=np.asarray(ps.off, dtype=np.int64),
        ring_feat=np.arange(k, dtype=np.int64),
        ring_hole=np.zeros(k, dtype=bool),
        n_features=k,
    )


def n_cells(ps):
    return int(as_polyset(ps).n_features)


def with_xy(ps, xy):
    ps = as_polyset(ps)
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) != len(ps.xy):
        raise ValueError(f"{len(xy)} vertices for a polygon set of {len(ps.xy)}")
    return dataclasses.replace(ps, xy=xy)


def _area_cent(ps):
    ps = as_polyset(ps)
    if not ps.n_features:
        return np.empty(0), np.empty((0, 2))
    return cp.feature_area_centroid(ps)


class Geometry:
    """``(area, centroid)`` of polygon sets, each computed once.

    ``cell_pairs.feature_area_centroid`` is a Python loop over every ring, and one
    record asked for it 58 times on a handful of sets. A set is recognised by IDENTITY
    (the object, which is kept alive here), never by value: a shifted or warped copy is
    another object and is computed for itself. What is kept is read, never written.

    ``parent`` is looked in first and never added to: a caller keeps the native sets of
    a case in one ``Geometry`` and hands it to every record, each of which keeps its
    own sets in a child that dies with it."""

    def __init__(self, parent=None):
        self._kept = {}
        self._parent = parent

    def _find(self, ps):
        hit = self._kept.get(id(ps))
        if hit is not None and hit[0] is ps:
            return hit[1]
        return None if self._parent is None else self._parent._find(ps)

    def area_cent(self, ps):
        ps = as_polyset(ps)
        got = self._find(ps)
        if got is None:
            got = _area_cent(ps)
            self._kept[id(ps)] = (ps, got)
        return got


def area_centroid(ps, geometry=None):
    """``_area_cent`` through a ``Geometry`` when there is one."""
    return _area_cent(ps) if geometry is None else geometry.area_cent(ps)


def gate_px(ref_native, mov_native, geometry=None):
    """``(match radius, median nuclear radius)`` in px, from the two NATIVE sets."""
    area_r, _ = area_centroid(ref_native, geometry)
    area_m, _ = area_centroid(mov_native, geometry)
    cell_radius = cp.median_equivalent_radius(np.concatenate([area_r, area_m]))
    return GATE_FACTOR * cell_radius, cell_radius


def _pct(v, q):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else math.nan


def _pair_iou(a, b, ia, ib, seen):
    """``cell_pairs.pair_iou`` for these pairs, computing only those not in ``seen``.

    A pair's IoU is rasterised in the window of its own two cells and depends on nothing
    else, so the value of a pair both pairings chose (most of them) is the same number
    and is computed once. ``seen`` is ``{"key", "iou", "scored"}`` of one ``(a, b)`` and
    is updated; None computes everything."""
    ia, ib = np.asarray(ia, dtype=np.int64), np.asarray(ib, dtype=np.int64)
    if seen is None:
        return cp.pair_iou(a, b, ia, ib)
    key = ia * (int(b.n_features) + 1) + ib
    iou = np.full(len(key), np.nan)
    scored = np.zeros(len(key), dtype=bool)
    have = np.zeros(len(key), dtype=bool)
    if len(seen.get("key", ())):
        pos = np.minimum(np.searchsorted(seen["key"], key), len(seen["key"]) - 1)
        have = seen["key"][pos] == key
        iou[have], scored[have] = seen["iou"][pos[have]], seen["scored"][pos[have]]
    new = ~have
    if new.any():
        iou[new], scored[new] = cp.pair_iou(a, b, ia[new], ib[new])
        keys = np.concatenate([seen.get("key", np.empty(0, np.int64)), key[new]])
        order = np.argsort(keys, kind="stable")
        seen["key"] = keys[order]
        seen["iou"] = np.concatenate([seen.get("iou", np.empty(0)), iou[new]])[order]
        seen["scored"] = np.concatenate(
            [seen.get("scored", np.empty(0, bool)), scored[new]]
        )[order]
    return iou, scored


def score(a, b, radius_px, pixel_size_um, pairing="lsa", seen=None, geometry=None):
    """``a`` is the reference side, ``b`` the moving side, both already in ONE frame.
    ``seen`` shares the per-pair IoU between calls on the same two sets (``_pair_iou``);
    ``geometry`` shares their areas and centroids (``Geometry``)."""
    a, b = as_polyset(a), as_polyset(b)
    n_r, n_m = a.n_features, b.n_features
    out = {
        "n_ref": n_r,
        "n_moving": n_m,
        "n_pairs": 0,
        "n_pairs_scored": 0,
        "pair_fraction": 0.0,
        **{k: math.nan for k in _NAN_KEYS},
    }
    if not n_r or not n_m or not radius_px > 0:
        return out
    # both sets have cells here, so this is cell_pairs.feature_area_centroid itself
    area_a, cent_a = area_centroid(a, geometry)
    area_b, cent_b = area_centroid(b, geometry)
    ia, ib, dist, _ = cp.match_cells(cent_a, cent_b, radius_px, method=pairing)
    out["n_pairs"] = int(ia.size)
    out["pair_fraction"] = ia.size / min(n_r, n_m)
    if not ia.size:
        return out
    iou, scored = _pair_iou(a, b, ia, ib, seen)
    out["n_pairs_scored"] = int(scored.sum())
    out["displacement_px_p50"] = _pct(dist, 50)
    out["displacement_px_p90"] = _pct(dist, 90)
    if pixel_size_um:
        um = dist * float(pixel_size_um)
        out["displacement_um_p50"] = _pct(um, 50)
        out["displacement_um_p90"] = _pct(um, 90)
    if scored.any():
        i = iou[scored]
        out["iou_p50"] = _pct(i, 50)
        out["frac_iou_ge_0.5"] = float((i >= IOU_THRESH).mean())
        # inter = iou * union and union = a + b - inter  =>  inter = iou * (a + b) / (1 + iou)
        tot = area_a[ia][scored] + area_b[ib][scored]
        if tot.sum() > 0:
            out["dice_matched"] = float(2.0 * (i * tot / (1.0 + i)).sum() / tot.sum())
    return out


def _one_way(cent_from, cent_to, pixel_size_um):
    """Share of ``cent_from`` (all of them) with a ``cent_to`` centroid within each threshold."""
    from scipy.spatial import cKDTree

    n = len(cent_from)
    ok = np.isfinite(cent_from).all(axis=1)
    to = cent_to[np.isfinite(cent_to).all(axis=1)]
    if not n:
        return [math.nan] * len(CURVE_UM)
    if not len(to) or not ok.any():
        return [0.0] * len(CURVE_UM)
    d, _ = cKDTree(to).query(cent_from[ok], k=1)
    um = d * float(pixel_size_um)
    return [float((um <= t).sum() / n) for t in CURVE_UM]


def within_curve(a, b, pixel_size_um, geometry=None):
    """The pairing-free curve; NaN without a pixel size, never zero."""
    out = {k: math.nan for k in _CURVE_KEYS}
    if not pixel_size_um or not n_cells(a):
        return out
    _, cent_r = area_centroid(a, geometry)
    _, cent_m = area_centroid(b, geometry)
    if not n_cells(b):
        return {k: 0.0 for k in _CURVE_KEYS}
    from_ref = _one_way(cent_r, cent_m, pixel_size_um)
    from_mov = _one_way(cent_m, cent_r, pixel_size_um)
    for t, r, m in zip(CURVE_UM, from_ref, from_mov):
        out[f"within_{t:g}um_ref"], out[f"within_{t:g}um_mov"] = r, m
        out[f"within_{t:g}um"] = min(r, m)
    return out


def _nanmean(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else math.nan


def _both_pairings(a, b, radius, pixel_size_um, geometry=None):
    one, seen = {}, {}
    for pairing, mid in (("lsa", ""), ("mutual_nn", "_mnn")):
        got = score(a, b, radius, pixel_size_um, pairing, seen, geometry)
        one.update({f"{k}{mid}": v for k, v in got.items()})
    return one


def record(
    ref_native, mov_native, ref_in_frame, mov_in_frame, pixel_size_um, geometry=None
):
    """Every score of one (case, variant); the keys are ``RECORD_KEYS``.

    Reference frame: ``(reference native, moving warped)``. Moving frame: ``(reference
    warped, moving native)``. The gate always comes from the two native sets.

    ``geometry`` is a ``Geometry`` that already holds sets this record will meet again
    (a case's native sets); it is read, and nothing of this record is left in it.
    """
    geo = Geometry(geometry)
    radius, cell_radius = gate_px(ref_native, mov_native, geo)
    a, b = as_polyset(ref_in_frame), as_polyset(mov_in_frame)
    n_r, n_m = a.n_features, b.n_features
    rec = {
        "match_radius_px": radius,
        "median_cell_radius_px": cell_radius,
        "count_ratio": (min(n_r, n_m) / max(n_r, n_m)) if max(n_r, n_m) else math.nan,
    }
    rec.update(_both_pairings(a, b, radius, pixel_size_um, geo))
    rec.update(within_curve(a, b, pixel_size_um, geo))
    # Cells the method's map sent nowhere (a non-finite centroid: outside the image it
    # wrote). They stay in every denominator as unpaired cells, in both directions; the
    # count is reported because pair fraction alone cannot show a loss on the LARGER set.
    lost = sum(
        int((~np.isfinite(geo.area_cent(side)[1]).all(axis=1)).sum()) for side in (a, b)
    )
    rec["n_unmapped"] = lost
    rec["unmapped_frac"] = lost / (n_r + n_m) if n_r + n_m else math.nan

    per_direction = []
    for k in range(NULL_DIRECTIONS):
        ang = 2.0 * math.pi * k / NULL_DIRECTIONS
        step = NULL_SHIFT_RADII * cell_radius * np.array([math.cos(ang), math.sin(ang)])
        shifted = with_xy(b, b.xy + step) if b.n_features else b
        # the shifted set is this placement's alone: its geometry is dropped with it
        placed = Geometry(geo)
        one = _both_pairings(a, shifted, radius, pixel_size_um, placed)
        one.update(within_curve(a, shifted, pixel_size_um, placed))
        per_direction.append(one)
    for key in per_direction[0]:
        rec[f"{key}_null"] = _nanmean([d[key] for d in per_direction])
    pf_null = [d["pair_fraction"] for d in per_direction]
    rec["pair_fraction_null_min"] = min(pf_null)
    rec["pair_fraction_null_max"] = max(pf_null)
    null = rec["pair_fraction_null"]
    rec["pair_fraction_excess"] = (
        (rec["pair_fraction"] - null) / (1.0 - null) if null < 1.0 else math.nan
    )
    rec["displacement_readable"] = bool(
        rec["pair_fraction"] - rec["pair_fraction_null_max"] >= READABLE_MARGIN
    )
    return rec
