"""Fixed-correspondence cell-pair metrics for staged registration QC.

The staged QC (see ``bin/warp_seg_qc.py``) establishes cell-to-cell correspondence **once**,
at the rigid stage, and then re-scores *those same pairs* after each subsequent registration
stage. This module owns the three pieces that requires:

1. :class:`PolySet` — a flat, vectorizable representation of a cell GeoJSON. All ring vertices
   of every feature live in one ``(N, 2)`` array, so a warp is a single call to
   ``valis.warp_tools.warp_xy`` instead of a per-feature Python loop. Warping a set returns a
   new :class:`PolySet` that *shares* the ring bookkeeping, which is what makes feature index
   ``i`` mean the same cell at every stage.

2. :func:`match_mutual_nn` — mutual-nearest-centroid pairing within a radius. Two cells pair
   only if each is the other's nearest neighbour *and* they are within ``radius`` px. Mutuality
   is what stops a dense cluster from collapsing onto one popular target; the radius is what
   stops a cell with no true partner from pairing with whatever happens to be closest.

3. :func:`pair_iou` — per-pair IoU by rasterizing each pair in its **own bounding-box window**,
   not on a shared slide-sized grid. Peak RAM is one nucleus, not one slide, which is the whole
   reason this replaces the previous whole-slide label-image approach. ``supersample`` recovers
   sub-pixel precision: a 30 px nucleus scored at 2x is a 60 px grid, so the IoU quantum is
   ~1% instead of ~4%.

Why a fixed pairing at all: if the matching is re-derived at each stage, a stage's score can
move because *different cells got paired*, or because a cell that the rasterizer occluded at
one stage reappears at another. Neither is a registration effect. Holding the pairing makes
every per-stage change a pure function of geometry.

numpy + scipy + scikit-image only — no shapely, no VALIS. Unit-testable without a container.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# A pair whose two cells span a window larger than this (after supersampling) is not scored.
# Real nuclei are a few thousand px; anything at this scale is a warp artifact or a merged
# blob, and rasterizing it would reintroduce exactly the unbounded allocation this design
# exists to remove. Such pairs are counted and reported, never silently dropped.
DEFAULT_MAX_PAIR_WINDOW_PX = 4_000_000

# A connected component of the candidate graph larger than this is not solved exactly — it
# falls back to mutual-NN inside that component. Percolation is the only way to get one:
# after a 1.5-radius gate on rigid-aligned tissue a component is normally 1-4 cells, but a
# badly-registered region can chain them. 5000 members is a <=2500x2500 dense sub-matrix
# (~50 MB), which is the same "bounded, counted, never silently dropped" posture as
# DEFAULT_MAX_PAIR_WINDOW_PX above.
DEFAULT_MAX_COMPONENT_CELLS = 5_000


# ── flat polygon representation ────────────────────────────────────────────────
@dataclass(frozen=True)
class PolySet:
    """All rings of all features of one GeoJSON, flattened.

    Attributes
    ----------
    xy : (N, 2) float64
        Every ring vertex of every feature, concatenated in feature order.
    ring_off : (R + 1,) int64
        Ring ``r`` occupies ``xy[ring_off[r]:ring_off[r + 1]]``.
    ring_feat : (R,) int64
        Feature index each ring belongs to. Non-decreasing.
    ring_hole : (R,) bool
        True for interior rings (holes).
    n_features : int
        Feature count, including features that contributed no ring (degenerate geometry).
        Kept so that feature indices stay aligned with the source file's feature order.
    """

    xy: np.ndarray
    ring_off: np.ndarray
    ring_feat: np.ndarray
    ring_hole: np.ndarray
    n_features: int

    @property
    def n_rings(self) -> int:
        """The total number of rings across every feature in this set."""
        return int(self.ring_feat.size)

    def with_xy(self, xy) -> "PolySet":
        """Same rings, new vertex coordinates — i.e. this set after a warp."""
        xy = np.asarray(xy, dtype=float)
        if xy.shape != self.xy.shape:
            raise ValueError(
                f"warped xy has shape {xy.shape}, expected {self.xy.shape}"
            )
        return PolySet(
            xy, self.ring_off, self.ring_feat, self.ring_hole, self.n_features
        )


def _iter_rings(geometry):
    """Yield ``(outer_ring, [hole_rings])`` per polygon part in a GeoJSON geometry."""
    if not geometry:
        return
    gtype, coords = geometry.get("type"), geometry.get("coordinates")
    if gtype == "Polygon" and coords:
        yield coords[0], list(coords[1:])
    elif gtype == "MultiPolygon" and coords:
        for poly in coords:
            if poly:
                yield poly[0], list(poly[1:])


def from_feature_collection(fc) -> PolySet:
    """Flatten a GeoJSON FeatureCollection into a :class:`PolySet`.

    Rings with fewer than three vertices are dropped — they cover no area and would only
    add empty slices to the bookkeeping. A feature that loses all of its rings still keeps
    its index, so ``n_features`` continues to match the file.
    """
    features = fc.get("features", [])
    chunks: list = []
    offs = [0]
    feats: list = []
    holes: list = []
    total = 0
    for i, ft in enumerate(features):
        for outer, hole_rings in _iter_rings(ft.get("geometry")):
            for ring, is_hole in [(outer, False)] + [(h, True) for h in hole_rings]:
                arr = np.asarray(ring, dtype=float)
                if arr.ndim != 2 or arr.shape[0] < 3:
                    continue
                arr = arr[:, :2]
                chunks.append(arr)
                total += arr.shape[0]
                offs.append(total)
                feats.append(i)
                holes.append(is_hole)
    xy = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2), dtype=float)
    return PolySet(
        xy=xy,
        ring_off=np.asarray(offs, dtype=np.int64),
        ring_feat=np.asarray(feats, dtype=np.int64),
        ring_hole=np.asarray(holes, dtype=bool),
        n_features=len(features),
    )


def _feature_ring_slices(ps: PolySet) -> np.ndarray:
    """``(n_features + 1,)`` offsets into the ring arrays: feature ``f``'s rings are
    ``range(out[f], out[f + 1])``.

    ``ring_feat`` is non-decreasing by construction, so this is a searchsorted, not a group-by.
    """
    return np.searchsorted(ps.ring_feat, np.arange(ps.n_features + 1), side="left")


# ── per-ring geometry (shoelace) ───────────────────────────────────────────────
def _ring_area_centroid(ps: PolySet):
    """Signed area and centroid of every ring."""
    n = ps.n_rings
    area = np.zeros(n, dtype=float)
    cent = np.zeros((n, 2), dtype=float)
    for r in range(n):
        p = ps.xy[ps.ring_off[r] : ps.ring_off[r + 1]]
        # Close the ring if the source did not repeat the first vertex.
        if p.shape[0] >= 2 and not np.array_equal(p[0], p[-1]):
            p = np.vstack([p, p[:1]])
        x, y = p[:, 0], p[:, 1]
        cross = x[:-1] * y[1:] - x[1:] * y[:-1]
        a = 0.5 * cross.sum()
        area[r] = a
        if a == 0:
            cent[r] = p[:-1].mean(axis=0) if p.shape[0] > 1 else p[0]
        else:
            cent[r, 0] = ((x[:-1] + x[1:]) * cross).sum() / (6.0 * a)
            cent[r, 1] = ((y[:-1] + y[1:]) * cross).sum() / (6.0 * a)
    return area, cent


def feature_area_centroid(ps: PolySet):
    """Per-feature ``(area, centroid)``; holes subtract from both.

    Ring orientation in GeoJSON is not reliably enforced by segmentation writers, so areas
    are taken as magnitudes and the hole flag — not the winding direction — decides the sign.
    """
    r_area, r_cent = _ring_area_centroid(ps)
    signed = np.abs(r_area) * np.where(ps.ring_hole, -1.0, 1.0)

    area = np.zeros(ps.n_features, dtype=float)
    moment = np.zeros((ps.n_features, 2), dtype=float)
    np.add.at(area, ps.ring_feat, signed)
    np.add.at(moment, ps.ring_feat, signed[:, None] * r_cent)

    cent = np.full((ps.n_features, 2), np.nan, dtype=float)
    ok = area != 0
    cent[ok] = moment[ok] / area[ok, None]

    # Degenerate feature (all rings collinear, or holes exactly cancelling the outer ring):
    # fall back to the mean vertex so it still has a position and can still be matched.
    bad = ~ok
    if bad.any():
        slices = _feature_ring_slices(ps)
        for f in np.flatnonzero(bad):
            lo, hi = int(slices[f]), int(slices[f + 1])
            if hi > lo:
                cent[f] = ps.xy[ps.ring_off[lo] : ps.ring_off[hi]].mean(axis=0)
    return np.abs(area), cent


def feature_bboxes(ps: PolySet) -> np.ndarray:
    """``(n_features, 4)`` array of ``[minx, miny, maxx, maxy]``; NaN for empty features."""
    out = np.full((ps.n_features, 4), np.nan, dtype=float)
    slices = _feature_ring_slices(ps)
    for f in range(ps.n_features):
        lo, hi = int(slices[f]), int(slices[f + 1])
        if hi <= lo:
            continue
        p = ps.xy[ps.ring_off[lo] : ps.ring_off[hi]]
        out[f] = [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
    return out


def median_equivalent_radius(area: np.ndarray) -> float:
    """Median nuclear radius implied by the areas — the natural unit for a match radius."""
    a = np.asarray(area, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0:
        return 0.0
    return float(math.sqrt(float(np.median(a)) / math.pi))


# ── correspondence ─────────────────────────────────────────────────────────────
def _finite_rows(cent):
    """``(rows, index)``: the rows of ``cent`` whose coordinates are all finite, and where
    they came from. This is :func:`match_lsa`'s NaN filter; :func:`match_mutual_nn` keeps
    its own inline equivalent (``np.flatnonzero(np.isfinite(...).all(axis=1))``) rather than
    calling this helper, and is left that way deliberately."""
    cent = np.asarray(cent, dtype=float)
    if cent.size == 0:
        return cent.reshape(0, 2), np.empty(0, dtype=np.int64)
    ok = np.flatnonzero(np.isfinite(cent).all(axis=1))
    return cent[ok], ok


def match_mutual_nn(cent_a, cent_b, radius):
    """Pair cells that are each other's nearest centroid and within ``radius``.

    Returns ``(idx_a, idx_b, dist)``. Mutual nearest neighbour is the cheapest pairing that is
    symmetric in the two slides — swapping the arguments yields the same pairs — which matters
    because "reference" and "moving" are not physically privileged here: both were segmented
    independently on their own native image.
    """
    from scipy.spatial import cKDTree

    cent_a = np.asarray(cent_a, dtype=float)
    cent_b = np.asarray(cent_b, dtype=float)
    empty = (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=float),
    )
    if cent_a.size == 0 or cent_b.size == 0 or not (radius > 0):
        return empty
    ok_a = np.flatnonzero(np.isfinite(cent_a).all(axis=1))
    ok_b = np.flatnonzero(np.isfinite(cent_b).all(axis=1))
    if ok_a.size == 0 or ok_b.size == 0:
        return empty

    tree_a = cKDTree(cent_a[ok_a])
    tree_b = cKDTree(cent_b[ok_b])
    d_ab, nn_ab = tree_b.query(cent_a[ok_a], k=1, distance_upper_bound=radius)
    _d_ba, nn_ba = tree_a.query(cent_b[ok_b], k=1, distance_upper_bound=radius)

    # cKDTree signals "nothing within the cap" with index == n and distance == inf.
    src = np.flatnonzero(np.isfinite(d_ab) & (nn_ab < ok_b.size))
    if src.size == 0:
        return empty
    tgt = nn_ab[src]
    mutual = (nn_ba[tgt] < ok_a.size) & (nn_ba[tgt] == src)
    src, tgt = src[mutual], tgt[mutual]
    return ok_a[src], ok_b[tgt], d_ab[src]


def match_lsa(cent_a, cent_b, radius, max_component_cells=DEFAULT_MAX_COMPONENT_CELLS):
    """Minimum-total-distance one-to-one pairing within ``radius``.

    Returns ``(idx_a, idx_b, dist, stats)``.

    This is the **exact** optimum, not an approximation, and it is computed without ever
    materializing the n x m cost matrix that would make it unrunnable on a real slide (two
    10^5-nucleus slides is 80 GB). The radius is a hard gate, so a forbidden pair can never be
    chosen, so no assignment edge crosses a connected component of the candidate graph -- and
    therefore the global minimum-cost assignment is exactly the concatenation of the
    per-component minima. Components are 1-4 cells on rigid-aligned tissue.

    Inside a component, entries outside the radius get a finite ``big_m`` rather than ``inf``:
    ``linear_sum_assignment`` raises ValueError on an infeasible matrix, and a component need
    not be a complete bipartite subgraph. ``big_m`` exceeds the sum of every real cost the
    component could contain (at most ``n_members`` of them, each <= ``radius``), so an
    assignment using k forbidden entries always costs more than one using k-1. That yields
    maximum cardinality first, then minimum distance -- from a bound, not a tuned constant.

    ``stats["n_components"]`` counts only components with candidates on **both** sides -- the
    ones the assignment loop actually solves. Singleton nodes (a cell with no candidate partner
    within ``radius``) are their own connected component in the underlying graph but are
    skipped before counting, so they are not included.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    stats = {
        "n_components": 0,
        "largest_component_cells": 0,
        "n_fallback_components": 0,
        "n_fallback_cells": 0,
    }
    empty = (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=float),
        stats,
    )
    a, ok_a = _finite_rows(cent_a)
    b, ok_b = _finite_rows(cent_b)
    if ok_a.size == 0 or ok_b.size == 0 or not (radius > 0):
        return empty

    na = ok_a.size
    neigh = cKDTree(a).query_ball_tree(cKDTree(b), radius)
    cols = np.fromiter((j for js in neigh for j in js), dtype=np.int64)
    if cols.size == 0:
        return empty
    rows = np.repeat(np.arange(na, dtype=np.int64), [len(js) for js in neigh])

    # One undirected graph over na + nb nodes: a-cell i is node i, b-cell j is node na + j.
    n_nodes = na + ok_b.size
    adj = coo_matrix((np.ones(cols.size), (rows, na + cols)), shape=(n_nodes, n_nodes))
    _, labels = connected_components(adj, directed=False)

    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    out_a, out_b = [], []
    for members in np.split(order, bounds):
        ai = members[members < na]
        bi = members[members >= na] - na
        if ai.size == 0 or bi.size == 0:
            continue  # an isolated cell: no candidate on the other slide
        stats["n_components"] += 1
        n_members = int(ai.size + bi.size)
        stats["largest_component_cells"] = max(
            stats["largest_component_cells"], n_members
        )
        if n_members > max_component_cells:
            fa, fb, _ = match_mutual_nn(a[ai], b[bi], radius)
            out_a.append(ai[fa])
            out_b.append(bi[fb])
            stats["n_fallback_components"] += 1
            stats["n_fallback_cells"] += n_members
            continue
        sub_a, sub_b = a[ai], b[bi]
        d = np.hypot(
            sub_a[:, None, 0] - sub_b[None, :, 0],
            sub_a[:, None, 1] - sub_b[None, :, 1],
        )
        big_m = n_members * float(radius) + 1.0
        cost = np.where(d <= radius, d, big_m)
        ri, ci = linear_sum_assignment(cost)
        keep = cost[ri, ci] < big_m
        out_a.append(ai[ri[keep]])
        out_b.append(bi[ci[keep]])

    if not out_a:
        return empty
    ia = np.concatenate(out_a)
    ib = np.concatenate(out_b)
    # Sort by the reference index so the pair order -- and the per-cell residual CSV built
    # from it -- is reproducible run to run.
    s = np.argsort(ia, kind="stable")
    ia, ib = ia[s], ib[s]
    dist = np.hypot(a[ia][:, 0] - b[ib][:, 0], a[ia][:, 1] - b[ib][:, 1])
    return ok_a[ia], ok_b[ib], dist, stats


PAIRING_METHODS = ("lsa", "mutual_nn")
DEFAULT_PAIRING = "lsa"


def match_cells(
    cent_a,
    cent_b,
    radius,
    method=DEFAULT_PAIRING,
    max_component_cells=DEFAULT_MAX_COMPONENT_CELLS,
):
    """Pair cells within ``radius`` using the named backend.

    Returns ``(idx_a, idx_b, dist, stats)``. ``stats`` is merged wholesale into the QC JSON's
    ``matching`` block by bin/warp_seg_qc.py, which is how a backend reports its own
    diagnostics without a second plumbing path.
    """
    if method == "mutual_nn":
        idx_a, idx_b, dist = match_mutual_nn(cent_a, cent_b, radius)
        return idx_a, idx_b, dist, {"method": "mutual_nn_centroid"}
    if method == "lsa":
        idx_a, idx_b, dist, stats = match_lsa(
            cent_a, cent_b, radius, max_component_cells=max_component_cells
        )
        return idx_a, idx_b, dist, {"method": "lsa_centroid", **stats}
    raise ValueError(
        f"unknown pairing method {method!r}; expected one of {PAIRING_METHODS}"
    )


def centroid_distance(cent_a, cent_b, idx_a, idx_b) -> np.ndarray:
    """Euclidean distance between the members of each pair, in the current frame."""
    ca = np.asarray(cent_a, dtype=float)[np.asarray(idx_a, dtype=np.int64)]
    cb = np.asarray(cent_b, dtype=float)[np.asarray(idx_b, dtype=np.int64)]
    if ca.size == 0:
        return np.empty(0, dtype=float)
    return np.linalg.norm(ca - cb, axis=1)


# ── per-pair IoU on a local window ─────────────────────────────────────────────
def _raster_feature(ps: PolySet, slices, f, origin_xy, shape_rc, scale):
    """Boolean mask of feature ``f`` inside a local window. Holes punched, outer rings OR'd."""
    from skimage.draw import polygon as sk_polygon

    mask = np.zeros(shape_rc, dtype=bool)
    hole = np.zeros(shape_rc, dtype=bool)
    ox, oy = origin_xy
    for r in range(int(slices[f]), int(slices[f + 1])):
        p = ps.xy[ps.ring_off[r] : ps.ring_off[r + 1]]
        rr, cc = sk_polygon(
            (p[:, 1] - oy) * scale, (p[:, 0] - ox) * scale, shape=shape_rc
        )
        if rr.size == 0:
            continue
        (hole if ps.ring_hole[r] else mask)[rr, cc] = True
    if hole.any():
        mask &= ~hole
    return mask


def pair_iou(
    ps_a: PolySet,
    ps_b: PolySet,
    idx_a,
    idx_b,
    supersample: int = 2,
    max_window_px: int = DEFAULT_MAX_PAIR_WINDOW_PX,
):
    """IoU of each matched pair, each scored in its own bounding-box window.

    Returns ``(iou, scored)``: ``iou`` is NaN wherever ``scored`` is False — a pair whose two
    cells are far enough apart (or distorted enough) to exceed ``max_window_px``, or whose
    geometry rasterizes to nothing. Callers report the unscored count rather than letting it
    vanish into a mean.
    """
    idx_a = np.asarray(idx_a, dtype=np.int64)
    idx_b = np.asarray(idx_b, dtype=np.int64)
    n = idx_a.size
    iou = np.full(n, np.nan, dtype=float)
    scored = np.zeros(n, dtype=bool)
    if n == 0:
        return iou, scored

    ss = max(1, int(supersample))
    sl_a, sl_b = _feature_ring_slices(ps_a), _feature_ring_slices(ps_b)
    bb_a, bb_b = feature_bboxes(ps_a), feature_bboxes(ps_b)

    for k in range(n):
        fa, fb = int(idx_a[k]), int(idx_b[k])
        ba, bb = bb_a[fa], bb_b[fb]
        if not (np.isfinite(ba).all() and np.isfinite(bb).all()):
            continue
        minx, miny = min(ba[0], bb[0]), min(ba[1], bb[1])
        maxx, maxy = max(ba[2], bb[2]), max(ba[3], bb[3])
        h = int(math.ceil((maxy - miny) * ss)) + 2
        w = int(math.ceil((maxx - minx) * ss)) + 2
        if h * w > max_window_px:
            continue
        origin = (minx - 1.0 / ss, miny - 1.0 / ss)
        ma = _raster_feature(ps_a, sl_a, fa, origin, (h, w), ss)
        mb = _raster_feature(ps_b, sl_b, fb, origin, (h, w), ss)
        union = int(np.count_nonzero(ma | mb))
        if union == 0:
            continue
        iou[k] = int(np.count_nonzero(ma & mb)) / union
        scored[k] = True
    return iou, scored


# ── summarisation ──────────────────────────────────────────────────────────────
def _dist_stats(values, prefix, percentiles=(10, 50, 90)) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"{prefix}_n": 0}
    out = {f"{prefix}_n": int(v.size), f"{prefix}_mean": float(v.mean())}
    for p in percentiles:
        out[f"{prefix}_p{p}"] = float(np.percentile(v, p))
    out[f"{prefix}_max"] = float(v.max())
    return out


def summarize_stage(
    iou, scored, dist_px, areas_a, areas_b, iou_thresh=0.5, pixel_size_um=None
) -> dict:
    """Collapse one stage's per-pair arrays into the reported record.

    ``dice_matched`` is the areal Dice restricted to the matched pairs: ``2*sum(inter) /
    sum(area_a + area_b)``, with ``inter`` recovered from each pair's IoU and areas. It is
    *not* the whole-slide foreground Dice — unmatched cells contribute nothing — so it is
    named to say so.
    """
    iou = np.asarray(iou, dtype=float)
    scored = np.asarray(scored, dtype=bool)
    dist_px = np.asarray(dist_px, dtype=float)
    aa = np.asarray(areas_a, dtype=float)
    ab = np.asarray(areas_b, dtype=float)

    rec: dict = {"n_pairs": int(iou.size), "n_pairs_scored": int(scored.sum())}
    rec.update(_dist_stats(iou[scored], "iou"))
    rec.update(_dist_stats(dist_px, "displacement_px"))
    if pixel_size_um:
        rec.update(_dist_stats(dist_px * float(pixel_size_um), "displacement_um"))

    if scored.any():
        i = iou[scored]
        rec[f"frac_iou_ge_{iou_thresh:g}"] = float((i >= iou_thresh).mean())
        # inter = iou * union and union = a + b - inter  =>  inter = iou*(a+b)/(1+iou)
        tot = aa[scored] + ab[scored]
        inter = i * tot / (1.0 + i)
        denom = float(tot.sum())
        rec["dice_matched"] = (
            float(2.0 * inter.sum() / denom) if denom else float("nan")
        )
    return rec
