"""Nuclei per RAW tile, the deduplicated nuclei of a front-end's mosaic, and the stitching
error between neighbour tiles.

Three steps, none of which holds an image across tiles:

* :func:`tile_nuclei` segments each tile of a cycle ONCE (method-independent, one tile read
  and segmented at a time), keeping only polygons in TILE pixels, cached per tile;
* :func:`owned` keeps one copy of every nucleus that lies in an overlap, decided once on the
  stage layout; :func:`mosaic_nuclei` carries that set into a front-end's mosaic at the SAME
  whole-pixel origins :func:`frontends.write_mosaic` pastes at (``frontends.placed_origins``);
* :func:`seam_error` pairs, for every neighbour pair, the nuclei both tiles saw in their
  overlap -- placed by the front-end's own (float) positions -- and reports how far apart
  the two copies land. That distance is the front-end's stitching error.

A "polygon" is a ring of ``cells.Polys`` (one set per tile).

Memory (user ruling 2026-10-14): ``tile_nuclei`` opens one tile at a time; ``mosaic_nuclei``
and ``seam_error`` take polygons only. ``test_stitching.py`` measures ``tile_nuclei`` on a
2x2 and a 6x6 grid: the traced peak grows with the polygons, not the pixels.

Neighbour rule of :func:`seam_error` (consistent with ``stare.stages.mosaic.neighbour_pairs``
of STARE 1.3.0 [PARTIAL: read the function, not the solver]): two placed tiles are neighbours
when they overlap by at least ``min_overlap_px`` (32) on BOTH axes and, on one axis, over at
least half the smaller tile -- a strip along a shared edge, not a diagonal corner square.
Ownership (:func:`owned`, used by :func:`mosaic_nuclei`) is geometric, on the stage layout, (which tile's centre is nearest) and looks
at EVERY overlapping tile, corners included.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from . import cells, frontends, nuclei, pairs

MIN_OVERLAP_PX = 32
DIAMETER_UM = 6.5  # a nucleus, as the cohort's 20 px at 0.325 um
CACHE_VERSION = 1


# ── segmenters: callable(tile TIFF path) -> cells.Polys in tile px, with a ``tag`` ────
class Segmenter:
    def __init__(self, fn, tag):
        self._fn, self.tag = fn, str(tag)

    def __call__(self, path):
        return self._fn(Path(path))


def builtin_segmenter(pixel_size_um, diameter_um=DIAMETER_UM):
    """``cells.py``'s segmenter on one tile: Otsu of THAT tile (a tile is all it sees, so
    thresholds differ between tiles), watershed split, no halo (a nucleus cut by the tile
    edge is a partial polygon; :func:`seam_error` leaves those out)."""
    diameter = max(float(diameter_um) / float(pixel_size_um), 4.0)

    def run(path):
        from skimage.filters import threshold_otsu

        img = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
        if img.ndim != 2 or not img.max() > img.min():
            return cells.empty_polys()
        labels = cells.segment_tile(img, float(threshold_otsu(img)), diameter)
        return cells.label_polys(labels, min_area=math.pi * (diameter / 4.0) ** 2)

    return Segmenter(run, f"builtin:d{diameter:.3f}")


def stardist_segmenter(work, env=None):
    """The StarDist route of ``nuclei.py`` (``segment.py`` with the arms' model and
    settings), run on one tile file at a time in ``work``. Needs the StarDist image."""
    env = os.environ if env is None else env
    work = Path(work)

    def run(path):
        slide = {"image": str(path), "channel": 0}
        dest = work / Path(path).stem
        dest.mkdir(parents=True, exist_ok=True)
        argv = nuclei.command(slide, dest, env)
        argv[0] = sys.executable
        res = subprocess.run(argv, capture_output=True, text=True)
        if res.returncode:
            tail = "\n".join((res.stderr or res.stdout or "").strip().splitlines()[-10:])
            raise RuntimeError(f"segment.py exited {res.returncode} on {path}: {tail}")
        mask, _ = nuclei.expected_masks(slide, dest)
        labels = tifffile.imread(str(mask))
        return cells.label_polys(labels)

    return Segmenter(run, f"stardist:{env.get(nuclei.ENV_MODEL, '')}")


def default_segmenter(cycle, work, env=None):
    """StarDist when ``REGBENCH_STARDIST_MODEL`` is set, else the built-in segmenter."""
    env = os.environ if env is None else env
    if (env.get(nuclei.ENV_MODEL) or "").strip():
        return stardist_segmenter(work, env)
    return builtin_segmenter(cycle.pixel_size_um)


# ── the cache ─────────────────────────────────────────────────────────────────
def _key(segmenter, row):
    st = os.stat(row.path)
    tag = getattr(segmenter, "tag", None) or f"{type(segmenter).__module__}.{type(segmenter).__qualname__}"
    return f"v{CACHE_VERSION}|{tag}|{st.st_size}|{st.st_mtime_ns}"


def _cache_file(cache_dir, row):
    return Path(cache_dir) / f"t{row.tile:05d}.npz"


def _load(path, key):
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["key"]) != key:
                return None
            return cells.Polys(np.array(z["xy"], dtype=float), np.array(z["off"], dtype=np.int64))
    except (OSError, ValueError, KeyError):
        return None


def _store(path, key, polys):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, key=np.array(key), xy=polys.xy, off=polys.off)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def tile_nuclei(cycle, segmenter, cache_dir=None):
    """``{tile: cells.Polys}``: the nuclei of every tile of ``cycle`` in TILE pixels, each
    tile segmented once by ``segmenter(path)`` (``.tag`` names it for the cache). With
    ``cache_dir`` (``<cases>/_cache/tiles/<dataset>/<cycle>/``) a tile whose cached set
    matches the segmenter and the tile file (size, mtime) is not segmented again."""
    out = {}
    for row in cycle.tiles:
        polys = None
        if cache_dir is not None:
            key = _key(segmenter, row)
            polys = _load(_cache_file(cache_dir, row), key)
        if polys is None:
            polys = segmenter(row.path)
            if cache_dir is not None:
                _store(_cache_file(cache_dir, row), key, polys)
        out[row.tile] = polys
    return out


# ── geometry ──────────────────────────────────────────────────────────────────
def _overlapping(boxes, *, strip, min_overlap_px):
    """``(i, j, (x0, y0, x1, y1))`` for ``i < j`` over ``boxes`` (n, 4: x0, y0, x1, y1).
    ``strip``: STARE's neighbour rule; else any positive-area overlap."""
    order = np.argsort(boxes[:, 0], kind="stable")
    out = []
    for p, i in enumerate(order):
        for j in order[p + 1:]:
            if boxes[j, 0] >= boxes[i, 2]:
                break
            x0, y0 = max(boxes[i, 0], boxes[j, 0]), max(boxes[i, 1], boxes[j, 1])
            x1, y1 = min(boxes[i, 2], boxes[j, 2]), min(boxes[i, 3], boxes[j, 3])
            ox, oy = x1 - x0, y1 - y0
            if ox <= 0 or oy <= 0:
                continue
            if strip:
                wa, wb = boxes[i, 2] - boxes[i, 0], boxes[j, 2] - boxes[j, 0]
                ha, hb = boxes[i, 3] - boxes[i, 1], boxes[j, 3] - boxes[j, 1]
                if ox < min_overlap_px or oy < min_overlap_px:
                    continue
                if not (ox >= 0.5 * min(wa, wb) or oy >= 0.5 * min(ha, hb)):
                    continue
            a, b = sorted((int(i), int(j)))
            out.append((a, b, (x0, y0, x1, y1)))
    return sorted(out, key=lambda t: t[:2])


def _inside(cent, rect):
    x0, y0, x1, y1 = rect
    return np.flatnonzero(
        (cent[:, 0] >= x0) & (cent[:, 0] < x1) & (cent[:, 1] >= y0) & (cent[:, 1] < y1)
    )


def _bboxes(polys):
    """``(K, 4)`` x0, y0, x1, y1 of every polygon."""
    if not len(polys):
        return np.empty((0, 4))
    st = polys.off[:-1]
    return np.stack([
        np.minimum.reduceat(polys.xy[:, 0], st), np.minimum.reduceat(polys.xy[:, 1], st),
        np.maximum.reduceat(polys.xy[:, 0], st), np.maximum.reduceat(polys.xy[:, 1], st),
    ], axis=1)


def _within(bbox, rect, inset):
    """Polygons whose bounding box lies inside ``rect`` shrunk by ``inset`` px."""
    x0, y0, x1, y1 = rect
    return np.flatnonzero(
        (bbox[:, 0] >= x0 + inset) & (bbox[:, 1] >= y0 + inset)
        & (bbox[:, 2] <= x1 - inset) & (bbox[:, 3] <= y1 - inset)
    )


def _shifted(polys, dx, dy):
    return polys.with_xy(polys.xy + np.array([dx, dy], dtype=float))


def _check(tile_polys, positions, tiles):
    ids = [r.tile for r in tiles]
    for what, got in (("tile_polys", tile_polys), ("positions", positions)):
        if set(got) != set(ids):
            raise ValueError(f"{what} and the tiles disagree: {sorted(got)} vs {sorted(ids)}")


# ── mosaic nuclei ─────────────────────────────────────────────────────────────
def owned(tile_polys, tiles):
    """``{tile: cells.Polys}``: the polygons each tile OWNS, still in TILE pixels.

    Ownership is decided once, on the cycle's STAGE layout (``tiles.csv`` positions through
    ``frontends.placed_origins``), so it does not depend on any front-end: a polygon is kept
    iff, among the tiles whose stage rectangle contains its stage-mosaic centroid, its own
    tile's centre is the nearest; an exact tie (within 1e-9 px) goes to the lower tile id.
    No matching, no radius. Compute it once per cycle and reuse it for every front-end.
    Only polygons are read."""
    tiles = tuple(tiles)
    ids = [r.tile for r in tiles]
    if set(tile_polys) != set(ids):
        raise ValueError(f"tile_polys and the tiles disagree: {sorted(tile_polys)} vs {sorted(ids)}")
    origins = frontends.placed_origins(tiles, {r.tile: (r.x_px, r.y_px) for r in tiles})
    boxes = np.array([[*origins[k], origins[k, 0] + r.w, origins[k, 1] + r.h]
                      for k, r in enumerate(tiles)], dtype=float).reshape(-1, 4)
    mids = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], 1)
    cent = [cells.area_centroid(tile_polys[r.tile])[1] + origins[k] for k, r in enumerate(tiles)]
    keep = [np.ones(len(c), bool) for c in cent]
    neighbours = [[] for _ in tiles]
    for a, b, _ in _overlapping(boxes, strip=False, min_overlap_px=1):
        neighbours[a].append(b)
        neighbours[b].append(a)
    for k, r in enumerate(tiles):
        if not len(cent[k]):
            continue
        own = np.hypot(*(cent[k] - mids[k]).T)
        for j in neighbours[k]:
            inside = np.zeros(len(cent[k]), bool)
            inside[_inside(cent[k], tuple(boxes[j]))] = True
            other = np.hypot(*(cent[k] - mids[j]).T)
            closer = other < own - 1e-9
            tied = (np.abs(other - own) <= 1e-9) & (tiles[j].tile < r.tile)
            keep[k] &= ~(inside & (closer | tied))
    return {r.tile: tile_polys[r.tile].take(np.flatnonzero(keep[k])) for k, r in enumerate(tiles)}


def mosaic_nuclei(tile_polys, positions, tiles, owned_polys=None):
    """The nuclei of a front-end's mosaic as one ``cells.Polys``, in mosaic pixels.

    The SET of nuclei is :func:`owned` (decided on the stage layout, the same for every
    front-end; pass ``owned_polys`` to reuse it); each is moved by the whole-pixel origin
    ``write_mosaic`` pastes its tile at (``frontends.placed_origins`` of ``positions``). A
    front-end's misplacement therefore shows up as the nuclei's POSITION error, never as a
    different population: the count is the stage layout's for every front-end. Only
    polygons are read."""
    tiles = tuple(tiles)
    _check(tile_polys, positions, tiles)
    kept = owned(tile_polys, tiles) if owned_polys is None else owned_polys
    origins = frontends.placed_origins(tiles, positions)
    return cells.concat([_shifted(kept[r.tile], *origins[k]) for k, r in enumerate(tiles)])


# ── stitching error ───────────────────────────────────────────────────────────
def _pct(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def seam_error(tile_polys, positions, tiles, pixel_size_um, *, radius_um=15.0,
               min_overlap_px=MIN_OVERLAP_PX):
    """The stitching error between neighbour tiles, from polygons only.

    For each neighbour pair (the STARE strip rule, see the module docstring) the polygons
    of both tiles whose bounding boxes lie inside the overlap of the tiles' placed rectangles
    (inset 1 px: a nucleus the overlap's edge cuts is partial in one tile, and leaving it
    out of one side only would hand its partner to a neighbour) --
    ``positions`` AS GIVEN, not rounded (the error is the front-end's, not the paste's) --
    are paired by ``cell_pairs.match_lsa`` within ``radius_um`` (default 15 um, the
    front-ends' ``max_shift_um``). Returns ``{"n_pairs", "n_matched", "n_candidates", "disp_um_p50", "disp_um_p90",
    "per_edge"}``; ``n_pairs`` counts neighbour edges, ``n_candidates`` the nuclei the two
    sides offered (the smaller side, summed), ``per_edge`` one dict per edge with the same
    percentiles (``None`` when nothing matched) and the median signed offset ``dx_um``,
    ``dy_um`` of tile ``b``'s copy from ``a``'s. A radius too wide for a dense tissue can
    pair the wrong neighbours."""
    tiles = tuple(tiles)
    _check(tile_polys, positions, tiles)
    ps = float(pixel_size_um)
    if not (math.isfinite(ps) and ps > 0):
        raise ValueError(f"pixel_size_um must be finite and > 0, got {pixel_size_um}")
    radius_px = float(radius_um) / ps
    pos = np.array([positions[r.tile] for r in tiles], dtype=float)
    boxes = np.array([[x, y, x + r.w, y + r.h] for (x, y), r in zip(pos, tiles, strict=True)],
                     dtype=float).reshape(-1, 4)
    cent, bbox = [], []
    for k, r in enumerate(tiles):
        polys = tile_polys[r.tile]
        _, c = cells.area_centroid(polys)
        cent.append(c + pos[k])
        bbox.append(_bboxes(polys) + np.tile(pos[k], 2))
    edges, all_d, n_match, n_cand = [], [], 0, 0
    for a, b, rect in _overlapping(boxes, strip=True, min_overlap_px=min_overlap_px):
        ia, ib = _within(bbox[a], rect, 1.0), _within(bbox[b], rect, 1.0)
        d = np.empty(0)
        off = np.empty((0, 2))
        if ia.size and ib.size:
            ma, mb, dd, _ = pairs.cp.match_lsa(cent[a][ia], cent[b][ib], radius_px)
            ma, mb = np.asarray(ma, int), np.asarray(mb, int)
            off = cent[b][ib][mb] - cent[a][ia][ma]
            d = np.hypot(off[:, 0], off[:, 1]) * ps
        all_d.append(d)
        n_match += len(d)
        n_cand += min(ia.size, ib.size)
        edges.append({
            "a": tiles[a].tile, "b": tiles[b].tile,
            "n_a": int(ia.size), "n_b": int(ib.size), "n_matched": int(len(d)),
            "disp_um_p50": _pct(d, 50), "disp_um_p90": _pct(d, 90),
            "dx_um": float(np.median(off[:, 0]) * ps) if len(d) else None,
            "dy_um": float(np.median(off[:, 1]) * ps) if len(d) else None,
        })
    flat = np.concatenate(all_d) if all_d else np.empty(0)
    return {
        "n_pairs": len(edges), "n_matched": int(n_match), "n_candidates": int(n_cand),
        "disp_um_p50": _pct(flat, 50), "disp_um_p90": _pct(flat, 90), "per_edge": edges,
    }
