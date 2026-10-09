"""Nuclei as polygons: segment a slide on its native grid, pair cells, score Dice and displacement.

The scoring is the mirage pipeline's reg_qc=2 "full transform" record (``bin/warp_seg_qc.py``
``score_full_transform`` over ``bin/utils/cell_pairs.py``), restated here so the benchmark
imports nothing from the pipeline:

* each slide's nuclei are segmented on its NATIVE image, never on a warped one, so no method
  is scored on pixels it interpolated itself;
* the moving slide's polygon vertices are pushed through the method's point map;
* cells are paired one-to-one by minimum total centroid distance inside a hard radius of
  ``1.5 x`` the median nuclear radius (:func:`match_lsa`);
* each pair is rasterised in its own bounding-box window for an IoU, and the pairs give
  ``dice_matched = 2 * sum(intersection) / sum(area_ref + area_moving)`` and the centroid
  displacement percentiles.

One simplification against mirage: a cell is ONE outer ring. Holes are dropped and a
MultiPolygon keeps its largest part, which is what a nucleus is.

Coordinates are X, Y with pixel CENTRES on the integers, the convention of
``skimage.measure.find_contours`` and of the ANHIR landmark files.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

MATCH_RADIUS_FACTOR = 1.5
SUPERSAMPLE = 2
IOU_THRESH = 0.5
MAX_PAIR_WINDOW_PX = 4_000_000
MAX_COMPONENT_CELLS = 5_000


@dataclass
class Polys:
    """``K`` polygons: ``xy`` is every vertex, ``off[k]:off[k + 1]`` the ring of cell ``k``."""

    xy: np.ndarray
    off: np.ndarray

    def __len__(self):
        return len(self.off) - 1

    def with_xy(self, xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if len(xy) != len(self.xy):
            raise ValueError(f"{len(xy)} vertices for a polygon set of {len(self.xy)}")
        return Polys(xy, self.off)

    def take(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        lens = (self.off[1:] - self.off[:-1])[idx]
        off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        if not len(idx):
            return Polys(np.empty((0, 2)), off)
        rows = np.concatenate([np.arange(self.off[i], self.off[i + 1]) for i in idx])
        return Polys(self.xy[rows], off)


def empty_polys():
    return Polys(np.empty((0, 2), dtype=float), np.zeros(1, dtype=np.int64))


def from_rings(rings):
    rings = [np.asarray(r, dtype=float).reshape(-1, 2) for r in rings]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return empty_polys()
    off = np.concatenate([[0], np.cumsum([len(r) for r in rings])]).astype(np.int64)
    return Polys(np.concatenate(rings), off)


def concat(parts):
    parts = [p for p in parts if len(p)]
    if not parts:
        return empty_polys()
    off = [np.zeros(1, dtype=np.int64)]
    base = 0
    for p in parts:
        off.append(p.off[1:] + base)
        base += len(p.xy)
    return Polys(np.concatenate([p.xy for p in parts]), np.concatenate(off))


# ── geometry ──────────────────────────────────────────────────────────────────
def area_centroid(ps):
    """Shoelace ``(area (K,), centroid (K, 2))``; a degenerate ring falls back to its vertex mean."""
    k = len(ps)
    if k == 0:
        return np.empty(0), np.empty((0, 2))
    x, y = ps.xy[:, 0], ps.xy[:, 1]
    nxt = np.arange(len(x)) + 1
    nxt[ps.off[1:] - 1] = ps.off[:-1]
    # Shift each ring to its own first vertex: the cross terms of slide-scale coordinates
    # (1e4..1e5 px) otherwise cancel to a nucleus-scale area with float error to match.
    first = np.repeat(ps.off[:-1], np.diff(ps.off))
    x, y = x - x[first], y - y[first]
    cross = x * y[nxt] - x[nxt] * y
    start = ps.off[:-1]
    a2 = np.add.reduceat(cross, start)
    cx = np.add.reduceat((x + x[nxt]) * cross, start)
    cy = np.add.reduceat((y + y[nxt]) * cross, start)
    n = np.diff(ps.off)
    mean = np.stack([np.add.reduceat(x, start) / n, np.add.reduceat(y, start) / n], axis=1)
    ok = np.abs(a2) > 1e-9
    cent = mean.copy()
    cent[ok, 0] = cx[ok] / (3.0 * a2[ok])
    cent[ok, 1] = cy[ok] / (3.0 * a2[ok])
    return np.abs(a2) / 2.0, cent + ps.xy[ps.off[:-1]]


def median_equivalent_radius(area):
    a = np.asarray(area, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    return float(math.sqrt(float(np.median(a)) / math.pi)) if a.size else 0.0


# ── segmentation ──────────────────────────────────────────────────────────────
def segment_tile(img, threshold, diameter):
    """Label the nuclei of one float image: smooth, threshold, split touching nuclei by watershed."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    sm = ndi.gaussian_filter(np.asarray(img, dtype=np.float32), max(diameter / 10.0, 0.5))
    fg = sm > threshold
    if not fg.any():
        return np.zeros(fg.shape, dtype=np.int32)
    dist = ndi.distance_transform_edt(fg)
    comp, _ = ndi.label(fg)
    peaks = peak_local_max(dist, min_distance=max(int(round(diameter / 3.0)), 1), labels=comp,
                           exclude_border=False)
    markers = np.zeros(fg.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-dist, markers, mask=fg)


def label_polys(labels, origin_xy=(0.0, 0.0), core=None, min_area=0.0):
    """One outline per label; with ``core=(x0, y0, x1, y1)`` only cells whose centroid lies in it.

    ``core`` is in the tile's own pixels. It is how a slide is segmented tile by tile without
    duplicates: tiles overlap by a halo wider than a nucleus, and every nucleus belongs to
    the one tile whose core holds its centroid, where it is also complete.
    """
    from scipy import ndimage as ndi
    from skimage.measure import find_contours

    rings = []
    ox, oy = origin_xy
    for lab, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        mask = labels[sl] == lab
        area = int(mask.sum())
        if area < max(min_area, 3):
            continue
        ys, xs = np.nonzero(mask)
        cx, cy = xs.mean() + sl[1].start, ys.mean() + sl[0].start
        if core is not None and not (core[0] <= cx < core[2] and core[1] <= cy < core[3]):
            continue
        contours = find_contours(np.pad(mask, 1).astype(np.float32), 0.5)
        if not contours:
            continue
        c = max(contours, key=len)[:-1]  # closed contour: drop the repeated last vertex
        if len(c) < 3:
            continue
        rings.append(np.stack([c[:, 1] - 1 + sl[1].start + ox, c[:, 0] - 1 + sl[0].start + oy], axis=1))
    return from_rings(rings)


def _tile_job(job):
    from stare.slide_io import open_lazy

    path, channel, mode, threshold, diameter, (x0, y0, x1, y1), halo, (h, w) = job
    ax0, ay0, ax1, ay1 = max(x0 - halo, 0), max(y0 - halo, 0), min(x1 + halo, w), min(y1 + halo, h)
    arr, _, close = open_lazy(path)
    try:
        tile = np.asarray(arr[channel, slice(ay0, ay1), slice(ax0, ax1)])
    finally:
        close()
    min_area = math.pi * (diameter / 4.0) ** 2
    labels = tile if mode == "labels" else segment_tile(tile, threshold, diameter)
    core = (x0 - ax0, y0 - ay0, x1 - ax0, y1 - ay0)
    return label_polys(labels, (ax0, ay0), core, min_area if mode == "image" else 0.0)


def _over_tiles(path, channel, mode, threshold, diameter, tile, workers):
    from stare.slide_io import open_lazy

    arr, _, close = open_lazy(path)
    _, h, w = arr.shape
    close()
    halo = int(max(2 * diameter, 32))
    jobs = [(str(path), channel, mode, threshold, diameter,
             (x0, y0, min(x0 + tile, w), min(y0 + tile, h)), halo, (h, w))
            for y0 in range(0, h, tile) for x0 in range(0, w, tile)]
    if workers > 1 and len(jobs) > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            parts = pool.map(_tile_job, jobs)
    else:
        parts = [_tile_job(j) for j in jobs]
    return concat(parts)


def slide_threshold(path, channel=0, max_dim=4096):
    """Otsu on a decimated view of the whole slide, so every tile shares one threshold."""
    from skimage.filters import threshold_otsu

    from stare.slide_io import decimation_factor, open_lazy, read_decimated

    arr, _, close = open_lazy(path)
    try:
        thumb = read_decimated(arr, channel, decimation_factor([arr.shape[1:]], max_dim))
    finally:
        close()
    return float(threshold_otsu(thumb)) if thumb.max() > thumb.min() else float(thumb.max())


def segment_slide(path, channel=0, diameter=20.0, tile=2048, workers=1, threshold=None):
    """Nuclear outlines of one channel of an OME-TIFF, tile by tile (bounded memory)."""
    if threshold is None:
        threshold = slide_threshold(path, channel)
    return _over_tiles(path, channel, "image", threshold, float(diameter), tile, workers)


def polys_from_label_image(path, diameter=20.0, tile=2048, workers=1):
    """Outlines of an integer label mask (TIFF), tile by tile; ``diameter`` only sizes the halo."""
    return _over_tiles(path, 0, "labels", None, float(diameter), tile, workers)


def polys_from_geojson(path):
    """A cell GeoJSON (FeatureCollection of Polygon / MultiPolygon), outer rings only."""
    with open(path) as fh:
        fc = json.load(fh)
    feats = fc["features"] if isinstance(fc, dict) else fc
    rings = []
    for f in feats:
        g = f.get("geometry") or {}
        if g.get("type") == "Polygon":
            outers = [g["coordinates"][0]] if g.get("coordinates") else []
        elif g.get("type") == "MultiPolygon":
            outers = [p[0] for p in g.get("coordinates", []) if p]
        else:
            continue
        if not outers:
            continue
        ring = np.asarray(max(outers, key=len), dtype=float)[:, :2]
        if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]  # GeoJSON closes its rings
        rings.append(ring)
    return from_rings(rings)


def load_cells(path, diameter=20.0, tile=2048, workers=1):
    """Cells from a GeoJSON or a label TIFF, by extension."""
    if str(path).lower().endswith((".geojson", ".json")):
        return polys_from_geojson(path)
    return polys_from_label_image(path, diameter, tile, workers)


# ── correspondence ────────────────────────────────────────────────────────────
def _mutual_nn(a, b, radius):
    from scipy.spatial import cKDTree

    d_ab, nn_ab = cKDTree(b).query(a, k=1, distance_upper_bound=radius)
    _, nn_ba = cKDTree(a).query(b, k=1, distance_upper_bound=radius)
    src = np.flatnonzero(np.isfinite(d_ab))
    tgt = nn_ab[src]
    keep = nn_ba[tgt] == src
    return src[keep], tgt[keep]


def match_lsa(cent_a, cent_b, radius, max_component_cells=MAX_COMPONENT_CELLS):
    """Minimum-total-distance one-to-one pairing within ``radius``: ``(idx_a, idx_b, dist)``.

    Exact without the n x m cost matrix: the radius is a hard gate, so no assignment edge
    crosses a connected component of the candidate graph, and the global optimum is the
    concatenation of the per-component optima. Inside a component a forbidden pair costs
    more than every real cost combined, which gives maximum cardinality first and minimum
    distance second. A component above ``max_component_cells`` falls back to mutual nearest
    neighbour.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0))
    cent_a = np.asarray(cent_a, dtype=float).reshape(-1, 2)
    cent_b = np.asarray(cent_b, dtype=float).reshape(-1, 2)
    ok_a = np.flatnonzero(np.isfinite(cent_a).all(axis=1))
    ok_b = np.flatnonzero(np.isfinite(cent_b).all(axis=1))
    if not ok_a.size or not ok_b.size or not radius > 0:
        return empty
    a, b = cent_a[ok_a], cent_b[ok_b]
    na = len(a)
    neigh = cKDTree(a).query_ball_tree(cKDTree(b), radius)
    cols = np.fromiter((j for js in neigh for j in js), dtype=np.int64)
    if not cols.size:
        return empty
    rows = np.repeat(np.arange(na, dtype=np.int64), [len(js) for js in neigh])
    n_nodes = na + len(b)
    adj = coo_matrix((np.ones(cols.size), (rows, na + cols)), shape=(n_nodes, n_nodes))
    _, labels = connected_components(adj, directed=False)
    order = np.argsort(labels, kind="stable")
    out_a, out_b = [], []
    for members in np.split(order, np.flatnonzero(np.diff(labels[order])) + 1):
        ai, bi = members[members < na], members[members >= na] - na
        if not ai.size or not bi.size:
            continue
        if ai.size + bi.size > max_component_cells:
            fa, fb = _mutual_nn(a[ai], b[bi], radius)
            out_a.append(ai[fa])
            out_b.append(bi[fb])
            continue
        d = np.hypot(a[ai][:, None, 0] - b[bi][None, :, 0], a[ai][:, None, 1] - b[bi][None, :, 1])
        big_m = (ai.size + bi.size) * float(radius) + 1.0
        cost = np.where(d <= radius, d, big_m)
        ri, ci = linear_sum_assignment(cost)
        keep = cost[ri, ci] < big_m
        out_a.append(ai[ri[keep]])
        out_b.append(bi[ci[keep]])
    if not out_a:
        return empty
    ia, ib = np.concatenate(out_a), np.concatenate(out_b)
    s = np.argsort(ia, kind="stable")
    ia, ib = ia[s], ib[s]
    return ok_a[ia], ok_b[ib], np.hypot(*(a[ia] - b[ib]).T)


# ── per-pair IoU ──────────────────────────────────────────────────────────────
def pair_iou(ps_a, ps_b, idx_a, idx_b, supersample=SUPERSAMPLE, max_window_px=MAX_PAIR_WINDOW_PX):
    """IoU of each pair in its own bounding-box window; NaN where a pair could not be scored."""
    from skimage.draw import polygon as sk_polygon

    iou = np.full(len(idx_a), np.nan)
    ss = max(1, int(supersample))
    for k, (fa, fb) in enumerate(zip(idx_a, idx_b)):
        pa = ps_a.xy[ps_a.off[fa] : ps_a.off[fa + 1]]
        pb = ps_b.xy[ps_b.off[fb] : ps_b.off[fb + 1]]
        both = np.concatenate([pa, pb])
        if not np.isfinite(both).all():
            continue
        (minx, miny), (maxx, maxy) = both.min(axis=0), both.max(axis=0)
        h = int(math.ceil((maxy - miny) * ss)) + 2
        w = int(math.ceil((maxx - minx) * ss)) + 2
        if h * w > max_window_px:
            continue
        ox, oy = minx - 1.0 / ss, miny - 1.0 / ss
        masks = []
        for p in (pa, pb):
            m = np.zeros((h, w), dtype=bool)
            rr, cc = sk_polygon((p[:, 1] - oy) * ss, (p[:, 0] - ox) * ss, shape=(h, w))
            m[rr, cc] = True
            masks.append(m)
        union = int(np.count_nonzero(masks[0] | masks[1]))
        if union:
            iou[k] = int(np.count_nonzero(masks[0] & masks[1])) / union
    return iou


# ── the record ────────────────────────────────────────────────────────────────
def _dist(values, prefix):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return {f"{prefix}_{k}": math.nan for k in ("mean", "p50", "p90", "max")}
    return {f"{prefix}_mean": float(v.mean()), f"{prefix}_p50": float(np.percentile(v, 50)),
            f"{prefix}_p90": float(np.percentile(v, 90)), f"{prefix}_max": float(v.max())}


def score_cells(ps_ref, ps_mov, pixel_size_um=None, radius_factor=MATCH_RADIUS_FACTOR,
                iou_thresh=IOU_THRESH):
    """Pair reference cells with the (already warped) moving cells and score the pairs.

    ``dice_matched`` covers the matched pairs only, so read it together with
    ``pair_fraction``: a registration that is off by more than the match radius pairs few
    cells, or pairs neighbours by chance, and its Dice over those pairs is not its quality.
    """
    area_r, cent_r = area_centroid(ps_ref)
    area_m, cent_m = area_centroid(ps_mov)
    cell_radius = median_equivalent_radius(np.concatenate([area_r, area_m]))
    radius = radius_factor * cell_radius
    ia, ib, dist = match_lsa(cent_r, cent_m, radius)
    iou = pair_iou(ps_ref, ps_mov, ia, ib)
    scored = np.isfinite(iou)
    n_r, n_m = len(ps_ref), len(ps_mov)
    rec = {
        "n_ref": n_r, "n_moving": n_m, "n_pairs": len(ia), "n_pairs_scored": int(scored.sum()),
        "pair_fraction": len(ia) / (min(n_r, n_m) or 1),
        "pair_fraction_moving": len(ia) / (n_m or 1),
        "match_radius_px": radius, "median_cell_radius_px": cell_radius,
        "dice_matched": math.nan, f"frac_iou_ge_{iou_thresh:g}": math.nan,
        **_dist(iou[scored], "iou"), **_dist(dist, "displacement_px"),
    }
    if pixel_size_um:
        rec.update(_dist(dist * float(pixel_size_um), "displacement_um"))
    else:
        rec.update(_dist([], "displacement_um"))
    if scored.any():
        i = iou[scored]
        rec[f"frac_iou_ge_{iou_thresh:g}"] = float((i >= iou_thresh).mean())
        # inter = iou * union and union = a + b - inter  =>  inter = iou * (a + b) / (1 + iou)
        tot = area_r[ia][scored] + area_m[ib][scored]
        if tot.sum() > 0:
            rec["dice_matched"] = float(2.0 * (i * tot / (1.0 + i)).sum() / tot.sum())
    return rec
