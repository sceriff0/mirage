"""Nuclei as polygons: segment a slide on its native grid, tile by tile.

This module makes the cells; it does not score them. Pairing, IoU and every score are
``regbench.pairs``, which calls the pipeline's own ``bin/utils/cell_pairs.py``.

* each slide's nuclei are segmented on its NATIVE image, never on a warped one, so no method
  is scored on pixels it interpolated itself;
* a slide has TWO polygon sets (``cases.SETS``): the bare nuclei, and the outlines, each
  nucleus grown by ``EXPAND_PX`` until it meets a neighbour (:func:`grow`: what the
  pipeline's StarDist backend publishes as its cell mask). :func:`segment_slide_sets` makes
  both from one segmentation, nucleus ``k`` and outline ``k`` being the same object;
* :class:`Polys` is one outer ring per cell, which is what the built-in segmenter and a
  label image give. Cells read from a GeoJSON (:func:`cells_from_geojson`) are a
  ``cell_pairs.PolySet`` from the pipeline's own ``from_feature_collection``: they keep
  their holes and parts. ``cases.write_case`` stores either kind.

Coordinates are X, Y with pixel CENTRES on the integers, the convention of
``skimage.measure.find_contours``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np


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
    mean = np.stack(
        [np.add.reduceat(x, start) / n, np.add.reduceat(y, start) / n], axis=1
    )
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

    sm = ndi.gaussian_filter(
        np.asarray(img, dtype=np.float32), max(diameter / 10.0, 0.5)
    )
    fg = sm > threshold
    if not fg.any():
        return np.zeros(fg.shape, dtype=np.int32)
    dist = ndi.distance_transform_edt(fg)
    comp, _ = ndi.label(fg)
    peaks = peak_local_max(
        dist,
        min_distance=max(int(round(diameter / 3.0)), 1),
        labels=comp,
        exclude_border=False,
    )
    markers = np.zeros(fg.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-dist, markers, mask=fg)


# The pipeline's cell mask is its nuclei grown by this many pixels: `seg_expand_distance`
# (nextflow.config; `segment.py --expand-distance`), 3.25 um at the cohort's 0.325 um.
EXPAND_PX = 10


def grow(labels, distance=EXPAND_PX):
    """Each label grown by ``distance`` px, no label growing into another: the call the
    pipeline makes (``bin/utils/segment_io.py``, ``expand_labels_tiled``, whose result is
    by its own test that of one whole-image ``skimage.segmentation.expand_labels``)."""
    from skimage.segmentation import expand_labels

    return expand_labels(np.asarray(labels), distance=int(distance))


# The pipeline publishes its cell mask as polygons simplified to this tolerance
# (bin/mask_to_geojson.py: `approximate_polygon(contour, tolerance=1.0)`). The grown
# outlines made here are traced the same way: it is what the cohort's outlines are, and
# an unsimplified grown contour has about 125 vertices where the simplified one has
# about 10, which the pipeline's per-pair raster IoU pays for four times over.
OUTLINE_TOLERANCE_PX = 1.0


def _ring(mask, x0, y0, tolerance=0.0):
    """The longest contour of a boolean mask whose first pixel is at ``(x0, y0)``, or
    None; with ``tolerance`` > 0 simplified as the pipeline simplifies its outlines."""
    from skimage.measure import approximate_polygon, find_contours

    contours = find_contours(np.pad(mask, 1).astype(np.float32), 0.5)
    if not contours:
        return None
    c = max(contours, key=len)
    if tolerance > 0:
        simple = approximate_polygon(c, tolerance=tolerance)
        c = simple if len(simple) >= 4 else c
    c = c[:-1]  # closed contour: drop the repeated last vertex
    if len(c) < 3:
        return None
    return np.stack([c[:, 1] - 1 + x0, c[:, 0] - 1 + y0], axis=1)


def label_polys(labels, origin_xy=(0.0, 0.0), core=None, min_area=0.0, grown=None):
    """One outline per label; with ``core=(x0, y0, x1, y1)`` only cells whose centroid lies in it.

    ``core`` is in the tile's own pixels. It is how a slide is segmented tile by tile without
    duplicates: tiles overlap by a halo wider than a nucleus, and every nucleus belongs to
    the one tile whose core holds its centroid, where it is also complete.

    With ``grown`` (the same labels after :func:`grow`) the result is ``(nuclei,
    outlines)``: the labels kept are decided on the NUCLEI alone, and outline ``k`` is the
    grown label of nucleus ``k``, simplified to ``OUTLINE_TOLERANCE_PX`` as the pipeline's
    published outlines are. The nuclei are never simplified.
    """
    from scipy import ndimage as ndi

    rings, big = [], []
    ox, oy = origin_xy
    grown_at = ndi.find_objects(grown) if grown is not None else None
    for lab, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        mask = labels[sl] == lab
        area = int(mask.sum())
        if area < max(min_area, 3):
            continue
        ys, xs = np.nonzero(mask)
        cx, cy = xs.mean() + sl[1].start, ys.mean() + sl[0].start
        if core is not None and not (
            core[0] <= cx < core[2] and core[1] <= cy < core[3]
        ):
            continue
        ring = _ring(mask, sl[1].start + ox, sl[0].start + oy)
        if ring is None:
            continue
        rings.append(ring)
        if grown is not None:
            gsl = grown_at[lab - 1] if lab <= len(grown_at) else None
            outer = (
                None
                if gsl is None
                else _ring(
                    grown[gsl] == lab,
                    gsl[1].start + ox,
                    gsl[0].start + oy,
                    OUTLINE_TOLERANCE_PX,
                )
            )
            big.append(ring if outer is None else outer)
    if grown is None:
        return from_rings(rings)
    return from_rings(rings), from_rings(big)


def _tile_job(job):
    from stare.slide_io import open_lazy

    path, channel, mode, threshold, diameter, (x0, y0, x1, y1), halo, (h, w) = job[:8]
    expand = job[8] if len(job) > 8 else None
    ax0, ay0, ax1, ay1 = (
        max(x0 - halo, 0),
        max(y0 - halo, 0),
        min(x1 + halo, w),
        min(y1 + halo, h),
    )
    arr, _, close = open_lazy(path)
    try:
        tile = np.asarray(arr[channel, slice(ay0, ay1), slice(ax0, ax1)])
    finally:
        close()
    min_area = math.pi * (diameter / 4.0) ** 2
    labels = tile if mode == "labels" else segment_tile(tile, threshold, diameter)
    core = (x0 - ax0, y0 - ay0, x1 - ax0, y1 - ay0)
    return label_polys(
        labels,
        (ax0, ay0),
        core,
        min_area if mode == "image" else 0.0,
        grow(labels, expand) if expand else None,
    )


def _over_tiles(path, channel, mode, threshold, diameter, tile, workers, expand=None):
    from stare.slide_io import open_lazy

    arr, _, close = open_lazy(path)
    _, h, w = arr.shape
    close()
    halo = int(max(2 * diameter, 32))
    jobs = [
        (
            str(path),
            channel,
            mode,
            threshold,
            diameter,
            (x0, y0, min(x0 + tile, w), min(y0 + tile, h)),
            halo,
            (h, w),
            expand,
        )
        for y0 in range(0, h, tile)
        for x0 in range(0, w, tile)
    ]
    if workers > 1 and len(jobs) > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            parts = pool.map(_tile_job, jobs)
    else:
        parts = [_tile_job(j) for j in jobs]
    if expand:
        return concat([p[0] for p in parts]), concat([p[1] for p in parts])
    return concat(parts)


def slide_threshold(path, channel=0, max_dim=4096):
    """Otsu on a decimated view of the whole slide, so every tile shares one threshold."""
    from skimage.filters import threshold_otsu
    from stare.slide_io import decimation_factor, open_lazy, read_decimated

    arr, _, close = open_lazy(path)
    try:
        thumb = read_decimated(
            arr, channel, decimation_factor([arr.shape[1:]], max_dim)
        )
    finally:
        close()
    return (
        float(threshold_otsu(thumb))
        if thumb.max() > thumb.min()
        else float(thumb.max())
    )


def segment_slide(path, channel=0, diameter=20.0, tile=2048, workers=1, threshold=None):
    """Nuclear outlines of one channel of an OME-TIFF, tile by tile (bounded memory)."""
    if threshold is None:
        threshold = slide_threshold(path, channel)
    return _over_tiles(
        path, channel, "image", threshold, float(diameter), tile, workers
    )


def segment_slide_sets(
    path,
    channel=0,
    diameter=20.0,
    tile=2048,
    workers=1,
    threshold=None,
    expand=EXPAND_PX,
):
    """``(nuclei, outlines)`` of one channel: the nuclei exactly as :func:`segment_slide`
    gives them, and each one's label grown by ``expand`` px (:func:`grow`), in the same
    order. The halo of a tile is not widened for the growth, so the nuclei do not depend
    on it; it is at least 32 px, wider than a nucleus's reach plus the 10 px rim.
    """
    if threshold is None:
        threshold = slide_threshold(path, channel)
    nuclei, outlines = _over_tiles(
        path, channel, "image", threshold, float(diameter), tile, workers, int(expand)
    )
    if len(nuclei) != len(outlines):  # from_rings drops nothing label_polys kept
        raise RuntimeError(
            f"{len(nuclei)} nuclei and {len(outlines)} outlines from one segmentation"
        )
    return nuclei, outlines


def polys_from_label_image(path, diameter=20.0, tile=2048, workers=1):
    """Outlines of an integer label mask (TIFF), tile by tile; ``diameter`` only sizes the halo."""
    return _over_tiles(path, 0, "labels", None, float(diameter), tile, workers)


def is_geojson(path):
    return str(path).lower().endswith((".geojson", ".json"))


def cells_from_geojson(path, hw=None):
    """A cell GeoJSON as a ``cell_pairs.PolySet``: every ring of every feature, holes and
    multi-part cells included, read by the pipeline's own ``from_feature_collection``.

    With ``hw``, the ``(H, W)`` of the slide the cells were segmented on, the file must fit
    that slide: every vertex within ``[-1, W] x [-1, H]`` (the pipeline's writer traces a
    border nucleus half a pixel outside the raster, ``bin/mask_to_geojson.py``). A file
    that is unreadable, holds no cell, or does not fit raises ``ValueError`` naming it.
    """
    from . import pairs

    try:
        with open(path) as fh:
            fc = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: cannot be read as GeoJSON ({exc})") from exc
    if isinstance(fc, list):
        fc = {"features": fc}
    if not isinstance(fc, dict):
        raise ValueError(f"{path}: not a GeoJSON FeatureCollection")
    ps = pairs.cp.from_feature_collection(fc)
    if not ps.n_rings:
        raise ValueError(f"{path}: no cell ({ps.n_features} features, no polygon)")
    if not np.isfinite(ps.xy).all():
        raise ValueError(f"{path}: a vertex is not finite")
    if hw is not None:
        h, w = (int(v) for v in hw)
        (x0, y0), (x1, y1) = ps.xy.min(axis=0), ps.xy.max(axis=0)
        if x0 < -1 or y0 < -1 or x1 > w or y1 > h:
            raise ValueError(
                f"{path}: cells lie outside their slide: x {x0:g}..{x1:g}, "
                f"y {y0:g}..{y1:g} on a slide {w} px wide and {h} px high"
            )
    return ps


def load_cells(path, diameter=20.0, tile=2048, workers=1, hw=None):
    """Cells from a GeoJSON (a ``PolySet``, checked against ``hw``) or a label TIFF (a
    :class:`Polys`), by extension."""
    if is_geojson(path):
        return cells_from_geojson(path, hw)
    return polys_from_label_image(path, diameter, tile, workers)


# ── points in polygons, and the outline shrunk back ───────────────────────────
def inside(ps, feat, pts, chunk=4_000_000):
    """For each pair ``(feat[i], pts[i])``: is the point inside that feature of a
    ``PolySet``? Even-odd over every ring of the feature, so a hole is outside."""
    from . import pairs

    ps = pairs.as_polyset(ps)
    feat = np.asarray(feat, dtype=np.int64).ravel()
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    out = np.zeros(len(feat), dtype=bool)
    if not len(feat) or not len(ps.xy):
        return out
    lens = np.diff(ps.ring_off)
    nxt = np.arange(len(ps.xy)) + 1
    nxt[ps.ring_off[1:] - 1] = ps.ring_off[:-1]
    edge_feat = np.repeat(ps.ring_feat, lens)
    order = np.argsort(edge_feat, kind="stable")
    count = np.bincount(edge_feat, minlength=ps.n_features)
    start = np.concatenate([[0], np.cumsum(count)])[:-1]
    x1, y1 = ps.xy[order, 0], ps.xy[order, 1]
    x2, y2 = ps.xy[nxt[order], 0], ps.xy[nxt[order], 1]
    n_edges = count[feat]
    bounds = np.concatenate([[0], np.cumsum(n_edges)])
    lo = 0
    while lo < len(feat):
        hi = max(
            int(np.searchsorted(bounds, bounds[lo] + chunk, side="right")) - 1, lo + 1
        )
        n = n_edges[lo:hi]
        pair = np.repeat(np.arange(hi - lo), n)
        within = np.arange(int(n.sum())) - np.repeat(bounds[lo:hi] - bounds[lo], n)
        e = start[feat[lo:hi]][pair] + within
        px, py = pts[lo:hi, 0][pair], pts[lo:hi, 1][pair]
        ya, yb, xa, xb = y1[e], y2[e], x1[e], x2[e]
        straddles = (ya > py) != (yb > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (xb - xa) * (py - ya) / (yb - ya) + xa
        crossings = np.bincount(pair[straddles & (px < x_at)], minlength=hi - lo)
        out[lo:hi] = crossings % 2 == 1
        lo = hi
    return out


def shrink(ps, distance=EXPAND_PX):
    """``(polys, parent)``: each outline of a ``PolySet`` with the rim of ``distance`` px
    taken off again, and the outline each result came from.

    NOT the nucleus the outline was grown from: growth stops where two cells meet, so
    taking the full rim off a clipped side cuts into the nucleus, and a small outline
    between neighbours can vanish (it then has no entry). It is the fallback of
    ``prepare multiplex --nuclei-fallback shrink`` and is labelled ``shrunk_outlines``.
    """
    from scipy import ndimage as ndi
    from skimage.draw import polygon as fill

    from . import pairs

    ps = pairs.as_polyset(ps)
    rings_of = [[] for _ in range(ps.n_features)]
    for r, f in enumerate(ps.ring_feat):
        rings_of[f].append(ps.xy[ps.ring_off[r] : ps.ring_off[r + 1]])
    rings, parent = [], []
    pad = 2
    for f, own in enumerate(rings_of):
        if not own:
            continue
        xy = np.concatenate(own)
        if not np.isfinite(xy).all():
            continue
        x0, y0 = np.floor(xy.min(axis=0)).astype(int) - pad
        x1, y1 = np.ceil(xy.max(axis=0)).astype(int) + pad
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
        for ring in own:  # even-odd: a hole clears what its outer ring set
            rr, cc = fill(ring[:, 1] - y0, ring[:, 0] - x0, mask.shape)
            mask[rr, cc] ^= True
        core = ndi.distance_transform_edt(mask) > distance
        if not core.any():
            continue
        lab, n = ndi.label(core)
        if n > 1:
            core = lab == 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
        ring = _ring(core, x0, y0)
        if ring is None:
            continue
        rings.append(ring)
        parent.append(f)
    return from_rings(rings), np.asarray(parent, dtype=np.int64)
