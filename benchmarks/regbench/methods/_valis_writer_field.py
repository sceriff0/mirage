"""The non-rigid field of VALIS's point map, sampled the way VALIS's WRITER samples it.

``_valis_points.image_map`` sends a point through ``warp_tools.warp_xy_inv``, which reads
the backward field with a cubic ``RectBivariateSpline`` at ``X / k`` (``warp_xy_non_rigid``,
``warp_tools.py:1969-1992`` at 1.2.0). The writer reads the SAME field another way
(``warp_tools.warp_img``, ``:1199-1222`` at 1.2.0, the same lines at 1.0.0): it multiplies
the field by ``k`` (canvas / field shape, ``get_warp_scaling_factors``), upsamples it onto
the canvas with ``pyvips.Image.affine([kx, 0, 0, ky], interpolate=bicubic)``, and adds the
result to the pixel index before ``mapim``. Source read: the PyPI sdists ``valis-wsi==1.0.0``
and ``==1.2.0`` ``[PARTIAL: warp_tools.warp_img, get_warp_scaling_factors, _warp_xy_numpy,
warp_xy, warp_xy_inv; slide_tools.warp_slide]``.

Where the field is smooth the two readings agree (measured 2026-10-24 with libvips 8.17 and
scipy on a synthetic field: 0.000 px p95 more than 8 field px from a step). VALIS's field
is NOT smooth at the tissue edge, and there they part: p95 1.0 to 1.8 canvas px within 4
field px of a step at ``k`` = 14, and the cluster's selfcheck failed exactly there (whole
image p95 <= 0.18 px, the 512 px blocks on the tissue boundary 0.4 to 1.9 px; 2026-10-24,
every ``valis120*`` and ``valis_mirage*`` non-rigid row, every rigid row ok).

``image_map`` here therefore takes the field from libvips itself: the writer's own
``affine`` call (same matrix, same interpolator, same ``premultiplied``) over an ``oarea``
window around the points, which is the writer's field at those integer pixels exactly; a
point between pixels reads it bilinearly. The rigid rest of the chain is VALIS's own
``warp_xy_inv`` with no field, which the selfcheck passes (p95 0.08 px). Only the two VALIS
adapters import this module, so no other method's ``reuse_key`` moves with it.
"""

from __future__ import annotations

import numpy as np

from ..mapping import REF_TO_MOV, PointMap
from . import _valis_points as vp

INTERP = "bicubic"  # warp_and_save_slides / warp_slide's default, passed to warp_img
WINDOW = 2048  # canvas px per side of one affine window: 2 x 2048^2 doubles, 64 MB


def _rigid_kwargs(snap):
    return {
        "M": snap["M"],
        "transformation_src_shape_rc": snap["processed_rc"],
        "transformation_dst_shape_rc": snap["reg_rc"],
        "src_shape_rc": snap["src_rc"],
        "dst_shape_rc": snap["aligned_rc"],
    }


def writer_field(snap, warp_tools):
    """``(vips image, affine matrix)``: the field scaled to canvas px, as ``warp_img``
    builds ``vips_dxdy`` and ``S``."""
    field = np.asarray(snap["bk_dxdy"])
    _, dst_sxy, _, _ = warp_tools.get_warp_scaling_factors(
        **{k: v for k, v in _rigid_kwargs(snap).items() if k != "M"}, bk_dxdy=field
    )
    sx, sy = (1.0, 1.0) if dst_sxy is None else (float(v) for v in dst_sxy)
    vips_dxdy = warp_tools.numpy2vips(np.dstack([sx * field[0], sy * field[1]]))
    return vips_dxdy, [sx, 0.0, 0.0, sy]


def displacement(xy, vips_dxdy, matrix, warp_tools, interp=INTERP, window=WINDOW):
    """``(N, 2)`` the writer's displacement at canvas points ``xy``: its upsampled field
    at the integer pixels around each point (one ``affine`` per ``window`` block that
    holds points), bilinear between them. NaN for a point that is not finite."""
    import pyvips

    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    out = np.full_like(xy, np.nan)
    ok = np.isfinite(xy).all(axis=1)
    if not ok.any():
        return out
    interpolate = pyvips.Interpolate.new(interp)
    idx = np.flatnonzero(ok)
    base = np.floor(xy[idx]).astype(np.int64)
    key = np.floor_divide(base, window)
    order = np.lexsort((key[:, 1], key[:, 0]))
    cuts = np.flatnonzero(np.any(np.diff(key[order], axis=0) != 0, axis=1)) + 1
    for rows in np.split(order, cuts):
        b = base[rows]
        x0, y0 = b.min(axis=0)
        w, h = b.max(axis=0) - (x0, y0) + 2  # the pixel at +1 of the last point too
        up = vips_dxdy.affine(
            matrix,
            oarea=[int(x0), int(y0), int(w), int(h)],
            interpolate=interpolate,
            premultiplied=True,
        )
        a = np.asarray(vp.numpy_field(up, warp_tools), dtype=float)  # (2, h, w)
        f = xy[idx[rows]] - (x0, y0)
        i = np.floor(f).astype(np.int64)
        t = f - i
        cx, cy = i[:, 0], i[:, 1]
        for band in (0, 1):
            g = a[band]
            top = g[cy, cx] * (1 - t[:, 0]) + g[cy, cx + 1] * t[:, 0]
            bot = g[cy + 1, cx] * (1 - t[:, 0]) + g[cy + 1, cx + 1] * t[:, 0]
            out[idx[rows], band] = top * (1 - t[:, 1]) + bot * t[:, 1]
    return out


def image_map(snap, warp_tools, interp=INTERP, window=WINDOW):
    """The map of the image VALIS's writer draws for this stage, reference-grid pixel ->
    moving slide pixel, with the field read as the writer reads it (module docstring).
    A rigid stage (no field) is ``_valis_points.image_map`` unchanged."""
    if snap["bk_dxdy"] is None:
        return vp.image_map(snap, warp_tools)
    origin = np.asarray(snap["origin_xy"], dtype=float)
    vips_dxdy, matrix = writer_field(snap, warp_tools)
    rigid = _rigid_kwargs(snap)

    def fn(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if not len(xy):
            return xy.copy()
        canvas = xy + origin
        moved = canvas + displacement(
            canvas, vips_dxdy, matrix, warp_tools, interp, window
        )
        out = warp_tools.warp_xy_inv(moved, bk_dxdy=None, **rigid)
        return np.asarray(out, dtype=float).reshape(-1, 2)

    return PointMap(fn, REF_TO_MOV)


def point_map(snap, warp_tools, points="image"):
    """``_valis_points.point_map`` with ``image`` read as the writer reads the field;
    ``from_to`` (VALIS's own point function) is left as it is."""
    if points not in vp.POINTS:
        raise ValueError(f"points={points!r}: expected one of {vp.POINTS}")
    if points == "image":
        return image_map(snap, warp_tools)
    return vp.from_to_map(snap, warp_tools)
