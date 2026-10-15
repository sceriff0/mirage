"""Tile geometry for the ASHLAR adapter. NumPy and SciPy only; (y, x) for tiles, (x, y) for
points.

ASHLAR places every tile rigidly: ``Mosaic.assemble_channel`` pastes tile ``i`` of a cycle
at ``aligner.positions[i]`` (``ashlar/reg.py:1147-1153``), and ``utils.paste`` puts the
tile's pixel ``l`` at mosaic pixel ``position + l``, whole part by slicing and fractional
part by ``scipy.ndimage.shift`` (``ashlar/utils.py:149-194``). Where a tile is the only one
drawn, that is the whole transform and it is exact. Source: labsyspharm/ashlar, the files
the plan cites as v1.20.0 ``[PARTIAL: scripts/ashlar.py, utils.py and thumbnail.py in
full; reg.py lines 1-119, 436-810 and 836-1375; filepattern.py not read]``.

Where tiles overlap ASHLAR cross-fades them (``pastefunc_blend``, ``utils.py:197-216``: a
weight that grows with the distance to the nearest unfilled pixel, tile after tile). The
point map here follows a simpler rule there: the mean of the covering tiles' shifts,
weighted by how deep the point lies inside each. That matches ASHLAR's blend only in
two-tile bands on a non-zero background. At four-tile corners ASHLAR's weights depend on
the paste order, and on a zero background the blend keeps the first copy at full intensity
where the later tile is zero (``utils.py:215``), so two shifts that differ show an object
twice. The map inside a band is therefore an APPROXIMATION and ``on_seam`` exists to leave
those points out: the rows scored without them are the ones to lean on.

**Two steps, both ASHLAR's own placement** (``placement_maps``). The mosaic is not on the
reference slide's grid: ASHLAR re-stitches the reference cycle, so a reference tile whose
pixels came from ``ref_true`` in the slide sits at ``ref_pos`` in the mosaic. A moving
point therefore goes

    moving slide --(+ mov_pos[t] - mov_true[t])--> mosaic --(- (ref_pos[s] - ref_true[s]))--> reference slide

with ``t`` the moving tile that holds the point and ``s`` the reference tile drawn where it
lands IN THE MOSAIC. Each step is one translation per tile, so its inverse is exact too.
"""

from __future__ import annotations

import numpy as np


def anchor_tile_origins(m0, ref_origins_yx, tile_size):
    """Where to cut each moving tile so it shows what its reference tile shows.

    ``m0`` (3 x 3, x-y) maps moving pixels to reference pixels. The centre of reference tile
    ``k`` is carried back into the moving slide and the tile is cut around it, unrotated, at
    whole pixels. Nothing is resampled; a rotation is left inside each tile for ASHLAR.
    """
    inv = np.linalg.inv(np.asarray(m0, dtype=float))
    half = tile_size / 2.0
    c_ref_xy = np.asarray(ref_origins_yx, dtype=float)[:, ::-1] + half
    c_mov_xy = c_ref_xy @ inv[:2, :2].T + inv[:2, 2]
    return np.round(c_mov_xy[:, ::-1] - half).astype(np.int64)


def half_size_xy(tile_size):
    """Half a tile, ``(x, y)``: ``tile_size`` is one number (a square tile, the retiled
    slides) or ``(h, w)`` (a camera's tile, e.g. 1008 x 1344 on CODEX: the raw-tile
    case sets)."""
    t = np.asarray(tile_size, dtype=float)
    if t.ndim == 0:
        return np.array([float(t), float(t)]) / 2.0
    if t.shape != (2,):
        raise ValueError(f"tile size must be one number or (h, w), got {tile_size!r}")
    return np.array([t[1], t[0]]) / 2.0


def _signed_depths(origins_yx, tile_size, xy, k):
    """``(depth (N, k), tile (N, k))``: how deep each point lies inside its ``k`` nearest
    tiles (max-norm), negative outside; ``-inf`` and tile ``-1`` for a missing neighbour.

    A rectangular tile: its depth is the distance to its nearest edge, ``min(w/2 - |dx|,
    h/2 - |dy|)``; the ``k`` candidates are the nearest in the max-norm scaled to the tile
    (every tile that holds the point is among them) and come deepest first."""
    from scipy.spatial import cKDTree

    half = half_size_xy(tile_size)
    centres_xy = np.asarray(origins_yx, dtype=float)[:, ::-1] + half
    k = min(k, len(centres_xy))
    if half[0] == half[1]:
        dist, idx = cKDTree(centres_xy).query(xy, k=k, p=np.inf)
        dist, idx = dist.reshape(len(xy), k), idx.reshape(len(xy), k)
        return half[0] - dist, np.where(np.isfinite(dist), idx, -1)
    scale = half.max() / half
    _, idx = cKDTree(centres_xy * scale).query(np.asarray(xy) * scale, k=k, p=np.inf)
    idx = np.asarray(idx).reshape(len(xy), k)
    found = idx < len(centres_xy)
    safe = np.where(found, idx, 0)
    off = np.abs(np.asarray(xy, dtype=float)[:, None, :] - centres_xy[safe])
    depth = np.min(half - off, axis=2)
    depth[~found] = -np.inf
    order = np.argsort(-depth, axis=1, kind="stable")
    depth = np.take_along_axis(depth, order, axis=1)
    idx = np.take_along_axis(np.where(found, idx, -1), order, axis=1)
    return depth, idx


def _depths(mov_origins_yx, tile_size, xy, k=4):
    """As ``_signed_depths``, zero where a point is outside the tile."""
    depth, idx = _signed_depths(mov_origins_yx, tile_size, xy, k)
    depth = np.clip(depth, 0.0, None)
    depth[~np.isfinite(depth)] = 0.0
    return depth, idx


def tile_point_map(mov_origins_yx, tile_size, d_yx):
    """``warp(xy) -> xy``: each point moves by the depth-weighted mean shift of the tiles
    covering it (one tile: that tile's shift, exactly)."""
    d_xy = np.asarray(d_yx, dtype=float)[:, ::-1]

    def warp(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if not len(xy):
            return xy.copy()
        depth, idx = _depths(mov_origins_yx, tile_size, xy)
        total = depth.sum(axis=1)
        outside = total <= 0
        # a point no tile covers takes its nearest tile's shift
        depth[outside, 0] = 1.0
        total[outside] = 1.0
        shift = (depth[..., None] * d_xy[np.clip(idx, 0, None)]).sum(axis=1)
        return xy + shift / total[:, None]

    return warp


def in_band(mov_origins_yx, tile_size, xy):
    """True where a point is covered by more than one tile (ASHLAR blends two placements there)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not len(xy):
        return np.zeros(0, dtype=bool)
    depth, _ = _depths(mov_origins_yx, tile_size, xy, k=2)
    return (depth > 0).sum(axis=1) > 1


def single_cover(origins_yx, tile_size, xy, margin=0.0):
    """True where exactly one tile holds the point, with ``margin`` px to spare: more than
    ``margin`` inside that tile and more than ``margin`` outside every other."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not len(xy):
        return np.zeros(0, dtype=bool)
    depth, _ = _signed_depths(origins_yx, tile_size, xy, k=2)
    inside = depth[:, 0] > margin
    if depth.shape[1] == 1:
        return inside
    return inside & (depth[:, 1] < -margin)


def tile_spec(tile):
    """A placement's ``tile``: one number (square) or ``[h, w]``."""
    if np.ndim(tile) == 0:
        return float(tile)
    return tuple(float(v) for v in tile)


def placement_maps(placement):
    """``(to_reference, to_mosaic)`` of one moving cycle (module docstring).

    ``placement`` holds, per tile in ASHLAR's order and in ``(y, x)``: ``mov_true_yx`` and
    ``ref_true_yx`` (where each tile's pixels came from in its slide), ``mov_pos_yx`` and
    ``ref_pos_yx`` (where ASHLAR drew it in the mosaic), and ``tile``. ``to_mosaic`` is
    moving slide -> mosaic; ``to_reference`` is moving slide -> reference slide.
    """
    tile = tile_spec(placement["tile"])
    mov_true = np.asarray(placement["mov_true_yx"], dtype=float)
    mov_pos = np.asarray(placement["mov_pos_yx"], dtype=float)
    ref_true = np.asarray(placement["ref_true_yx"], dtype=float)
    ref_pos = np.asarray(placement["ref_pos_yx"], dtype=float)
    to_mosaic = tile_point_map(mov_true, tile, mov_pos - mov_true)
    # looked up by the reference tiles' footprints IN THE MOSAIC, hence origins = ref_pos
    mosaic_to_ref = tile_point_map(ref_pos, tile, ref_true - ref_pos)

    def to_reference(xy):
        return mosaic_to_ref(to_mosaic(xy))

    return to_reference, to_mosaic


def on_seam(placement, xy, margin=1.0):
    """True where the mosaic does NOT show a moving point as one unblended copy over one
    unblended reference tile, so the point map is not exactly the image there.

    A point is off every seam when all three hold, each with ``margin`` px to spare: one
    moving tile holds it in the slide (two would draw it twice); it lands under that one
    moving tile's footprint in the mosaic and under no other's (neighbours placed with
    different shifts overlap elsewhere than the tiles were cut, and the content is then
    cross-faded with another tile's); and one reference tile is drawn where it lands.
    Only ASHLAR's geometry enters: the tile grid and the positions it accepted.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not len(xy):
        return np.zeros(0, dtype=bool)
    tile = tile_spec(placement["tile"])
    _, to_mosaic = placement_maps(placement)
    m = to_mosaic(xy)
    clean = (
        single_cover(placement["mov_true_yx"], tile, xy, margin)
        & single_cover(placement["mov_pos_yx"], tile, m, margin)
        & single_cover(placement["ref_pos_yx"], tile, m, margin)
    )
    return ~clean


def neighbour_jumps(d_yx, n_rows, n_cols):
    """Length of the difference between the shifts of grid-neighbouring tiles (row-major
    ``d_yx``), one value per horizontal and per vertical neighbour pair."""
    d = np.asarray(d_yx, dtype=float).reshape(n_rows, n_cols, 2)
    across = np.linalg.norm(d[:, 1:] - d[:, :-1], axis=-1).ravel()
    down = np.linalg.norm(d[1:] - d[:-1], axis=-1).ravel()
    return np.concatenate([across, down])
