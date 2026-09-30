"""Image warp for the STARE registration method (the WARP_TILE core).

Warps a moving image into the reference frame through the *same* transform the reg_qc=2 stage
warper uses — global affine ``M0`` plus the smooth mesh residual ``F`` — so the QC measures
exactly what ships. Warping resamples by the **inverse** map: for each reference-frame output
pixel ``u`` it finds the source moving coordinate ``x`` solving the forward relation
``u = M0·x + F(x)``, then samples the image there with :func:`mesh_field.resample_bilinear`. Because
``F`` is a contraction (SOLVE's fold certificate accepts Lipschitz ``L < 0.5``), the inverse is
found by the fixed-point iteration ``v <- u - F(v)``, ``x = M0^-1 v``, run until the step
``max |v_k - v_(k-1)|`` is below ``INVERSE_TOL_PX`` (or ``INVERSE_MAX_ITERATIONS``); with no mesh
it is the exact affine inverse.

Bilinear resampling keeps a non-negative image non-negative — the guarantee downstream marker
quantification relies on. Pure NumPy.
"""

from __future__ import annotations

import numpy as np

from stare.mesh_field import resample_bilinear

__all__ = [
    "INVERSE_MAX_ITERATIONS",
    "INVERSE_TOL_PX",
    "new_inverse_stats",
    "source_coords",
    "source_region",
    "warp_image",
]


def _apply_affine(m, xy):
    homog = np.column_stack([xy, np.ones(len(xy))])
    return (homog @ np.asarray(m, dtype=float).T)[:, :2]


# The inverse map's fixed point v = u - F(v) is iterated to a TOLERANCE, not a fixed count.
# The error after k steps is <= L^k |F| (Banach; Behrmann et al. 2019 eq. 1), so a fixed 3
# steps is only good for small L: SOLVE certifies L < 0.5 (FOLD_CERTIFICATE_LIPSCHITZ), and at
# L = 0.5 with a 100 px field three steps leave up to 12.5 px (4.56 px measured,
# research/stare-papers/oa/field/NOTES.md F2c). Real fields (L ~ 0.02) stop in 2-4 steps. The
# step |v_k - v_(k-1)| bounds the returned point's fixed-point residual (by L times it) and its
# error (by L / (1 - L) times it, <= the step itself for L <= 0.5), so it is what is recorded.
# The cap only bounds the work on a field outside the certificate; hitting it is reported.
INVERSE_TOL_PX = 1e-3
INVERSE_MAX_ITERATIONS = 50


def new_inverse_stats():
    """An empty accumulator for :func:`_invert`'s convergence record (see ``stats`` there)."""
    return {
        "inverse_residual_px": 0.0,
        "inverse_iterations_max": 0,
        "inverse_calls": 0,
        "inverse_cap_hits": 0,
    }


def _invert(m0, mesh, u, inverse_tol=INVERSE_TOL_PX, stats=None):
    """Source (moving) coordinates for reference-frame points ``u``: x = M0^-1 (u - F(v)).

    ``v <- u - F(v)`` is iterated, per point, until that point's ``|v_k - v_(k-1)|`` (max
    norm) is below ``inverse_tol`` px, or ``INVERSE_MAX_ITERATIONS``. ``stats`` (from
    :func:`new_inverse_stats`), when given, accumulates the largest final step
    (``inverse_residual_px``), the most iterations, the call count and how many calls
    stopped at the cap with a point still unconverged.
    """
    u = np.asarray(u, dtype=float)
    v = u.copy()
    if mesh is not None and v.size:
        # Per POINT: a point stops once its own step is below tolerance, so its result does
        # not depend on which other points share the call -- the tiles of a streamed stitch
        # and a whole-image warp give identical coordinates (no seam at a tile edge).
        active = np.ones(len(v), dtype=bool)
        last = np.zeros(len(v))
        it = 0
        while it < INVERSE_MAX_ITERATIONS and active.any():
            ua, va = u[active], v[active]
            vn = ua - mesh.displacement(va)
            step = np.abs(vn - va).max(axis=1)
            v[active] = vn
            last[active] = step
            active[np.flatnonzero(active)[step < inverse_tol]] = False
            it += 1
        if stats is not None:
            stats["inverse_residual_px"] = max(
                stats["inverse_residual_px"], float(last.max())
            )
            stats["inverse_iterations_max"] = max(stats["inverse_iterations_max"], it)
            stats["inverse_calls"] += 1
            stats["inverse_cap_hits"] += int(active.any())
    return _apply_affine(np.linalg.inv(np.asarray(m0, dtype=float)), v)


def source_region(
    m0,
    mesh,
    out_origin,
    out_shape,
    margin=8,
    src_shape=None,
    inverse_tol=INVERSE_TOL_PX,
):
    """Bounding box (in moving pixels) that an output tile draws from.

    Inverse-maps the output tile's corners to moving coordinates and pads by ``margin`` (which must
    cover the mesh's residual displacement plus a bilinear pixel). Returns integer
    ``(x0, y0, x1, y1)``, clamped to ``src_shape`` = ``(H, W)`` when given. This lets the streaming
    stitch read only the moving pixels a tile needs, never the whole slide.
    """
    ox, oy = out_origin
    out_h, out_w = out_shape
    corners = np.array(
        [[ox, oy], [ox + out_w, oy], [ox, oy + out_h], [ox + out_w, oy + out_h]],
        dtype=float,
    )
    x = _invert(m0, mesh, corners, inverse_tol)
    x0 = int(np.floor(x[:, 0].min())) - margin
    y0 = int(np.floor(x[:, 1].min())) - margin
    x1 = int(np.ceil(x[:, 0].max())) + margin
    y1 = int(np.ceil(x[:, 1].max())) + margin
    if src_shape is not None:
        h, w = src_shape
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(int(w), x1), min(int(h), y1)
    return x0, y0, x1, y1


def _axis_upsampler(n_out, origin, step):
    """Sub-grid nodes on one axis, and each output pixel's lower node + linear weight.

    Nodes sit at GLOBAL multiples of ``step`` (not tile-relative), so a pixel is interpolated
    from the same nodes whichever output tile it falls in: the streamed stitch equals a
    whole-image warp at the same ``step``, tile seams included.
    """
    k0 = origin // step
    k1 = max(-(-(origin + n_out - 1) // step), k0 + 1)
    nodes = np.arange(k0, k1 + 1, dtype=float) * step
    t = (np.arange(n_out, dtype=float) + origin - k0 * step) / step
    i0 = np.minimum(np.floor(t).astype(int), nodes.size - 2)
    return nodes, i0, t - i0


def source_coords(
    m0,
    mesh,
    out_shape,
    out_origin=(0, 0),
    inverse_tol=INVERSE_TOL_PX,
    field_step=None,
    stats=None,
):
    """Moving coordinates ``(H, W, 2)`` (x, y) that each reference-frame output pixel samples.

    ``field_step=None`` inverts the map exactly at every pixel. An integer ``field_step`` inverts
    it only on a sub-grid every ``field_step`` px (global multiples) and bilinearly upsamples the
    result: exact for the affine part, and for the mesh part off by at most
    ``(h^2/8)(max|u_xx| + max|u_yy|)``, the bilinear interpolation bound (a cubic mesh; a bilinear one adds ``h |jump u'| / 4`` at its cell edges) -- 0.0015 px
    worst of 10k pixels on SOLVE's cubic meshes at the stitch's ``h = 8``
    (``stages.stitch.FIELD_STEP``; ``tests/test_tiled_warp.py`` pins it). The QC seam
    (``stage_warp``) keeps evaluating the exact field at its points. The fixed-point inverse
    runs to ``inverse_tol`` px at the nodes (``_invert``; ``stats`` records its convergence),
    so on the sub-grid path the iteration costs one field evaluation per node per step.
    """
    out_h, out_w = out_shape
    ox, oy = out_origin
    if mesh is None or not field_step:
        ys, xs = np.mgrid[0:out_h, 0:out_w]
        u = np.column_stack(
            [(xs.ravel() + ox).astype(float), (ys.ravel() + oy).astype(float)]
        )
        return _invert(m0, mesh, u, inverse_tol, stats).reshape(out_h, out_w, 2)
    step = int(field_step)
    nx, ix, fx = _axis_upsampler(out_w, int(ox), step)
    ny, iy, fy = _axis_upsampler(out_h, int(oy), step)
    gx, gy = np.meshgrid(nx, ny)
    xc = _invert(
        m0, mesh, np.column_stack([gx.ravel(), gy.ravel()]), inverse_tol, stats
    ).reshape(ny.size, nx.size, 2)
    # separable linear upsampling: along x on the node rows, then along y
    fx, fy = fx[None, :, None], fy[:, None, None]
    rows = xc[:, ix] * (1.0 - fx) + xc[:, ix + 1] * fx
    return rows[iy] * (1.0 - fy) + rows[iy + 1] * fy


def warp_image(
    image,
    m0,
    mesh,
    out_shape,
    inverse_tol=INVERSE_TOL_PX,
    out_origin=(0, 0),
    src_origin=(0, 0),
    field_step=None,
    stats=None,
):
    """Warp ``image`` (moving) into an ``out_shape`` = ``(H, W)`` reference-frame raster.

    Parameters
    ----------
    image : ndarray, ``(H, W)`` or ``(H, W, C)``
        The moving image (non-negative).
    m0 : 3x3 array
        Global forward affine, moving -> reference.
    mesh : MeshField or None
        Residual displacement field in the *reference frame*, or None for a rigid warp.
    out_shape : (int, int)
        Output raster size ``(height, width)`` in the reference frame.
    out_origin : (int, int)
        ``(x, y)`` top-left of the output window in the reference frame (default ``(0, 0)``).
        Warping a window equals cropping the full-frame warp, so per-tile warps reassemble
        seamlessly and each tile task holds only its own output in memory.
    src_origin : (int, int)
        ``(x, y)`` of ``image``'s top-left in moving coordinates, when ``image`` is a crop of the
        moving slide (streaming stitch). Sample points are shifted into the crop's local frame.
    field_step : int or None
        Evaluate the inverse map on a sub-grid this many px apart and upsample it
        (:func:`source_coords`); ``None`` (default) evaluates it at every pixel.
    inverse_tol : float
        Fixed-point tolerance (px) of the inverse map (``INVERSE_TOL_PX``; see ``_invert``).
    stats : dict or None
        :func:`new_inverse_stats` accumulator for the inverse's convergence record.
    """
    image = np.asarray(image, dtype=float)
    out_h, out_w = out_shape
    sox, soy = src_origin

    # Forward map is u = v + F(v) with v = M0 x the rigid (ref-frame) position. Invert in two
    # decoupled steps: solve v = u - F(v) by fixed-point (F is a contraction), then x = M0^-1 v.
    x = source_coords(
        m0, mesh, out_shape, out_origin, inverse_tol, field_step, stats
    ).reshape(-1, 2)
    # ``image`` may be a crop of the moving slide whose top-left sits at ``src_origin`` in moving
    # coordinates — shift the sample points into the crop's local frame.
    if sox or soy:
        x = x - np.array([sox, soy], dtype=float)

    vals = resample_bilinear(image, x)
    if image.ndim == 3:
        return vals.reshape(out_h, out_w, image.shape[2])
    return vals.reshape(out_h, out_w)
