"""Nuclei as polygons: segment a slide on its native grid, tile by tile.

This module makes the cells; it does not score them. Pairing, IoU and every score are
``regbench.pairs``, which calls the pipeline's own ``bin/utils/cell_pairs.py``.

* each slide's nuclei are segmented on its NATIVE image, never on a warped one, so no method
  is scored on pixels it interpolated itself;
* :class:`Polys` is one outer ring per cell, which is what the built-in segmenter and a
  label image give. Cells read from a GeoJSON through ``cell_pairs.from_feature_collection``
  keep their holes and parts; ``cases.write_case`` stores either kind.

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
        if core is not None and not (
            core[0] <= cx < core[2] and core[1] <= cy < core[3]
        ):
            continue
        contours = find_contours(np.pad(mask, 1).astype(np.float32), 0.5)
        if not contours:
            continue
        c = max(contours, key=len)[:-1]  # closed contour: drop the repeated last vertex
        if len(c) < 3:
            continue
        rings.append(
            np.stack(
                [c[:, 1] - 1 + sl[1].start + ox, c[:, 0] - 1 + sl[0].start + oy], axis=1
            )
        )
    return from_rings(rings)


def _tile_job(job):
    from stare.slide_io import open_lazy

    path, channel, mode, threshold, diameter, (x0, y0, x1, y1), halo, (h, w) = job
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
    return label_polys(labels, (ax0, ay0), core, min_area if mode == "image" else 0.0)


def _over_tiles(path, channel, mode, threshold, diameter, tile, workers):
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
