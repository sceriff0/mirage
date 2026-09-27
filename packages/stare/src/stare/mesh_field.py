"""Smooth control-grid displacement field + a non-negative bilinear resampler.

These are the geometry primitives of the tiled ('STARE') registration method. The tiled
registrar measures, per tile, one residual displacement at the tile centre; those control
points form a regular grid, and :class:`MeshField` interpolates them into a *single continuous*
displacement field over the whole slide. Warping points or images through one continuous field
is what keeps a cell straddling a tile boundary from being torn — there is never a per-tile
discontinuity to fall into.

Both the forward point-warp (used by the reg_qc=2 stage warper) and the image warp (used by the
per-tile WARP step) sample the *same* field -- both build it with :meth:`MeshField.from_spec` --
so the QC measures exactly the transform that shipped. The field is bilinear unless the mesh
spec says ``"interp": "cubic"`` (what SOLVE's ``dctpls`` writes for its vector lattice).

Nothing here imports VALIS, pyvips, or a JVM — pure NumPy, a few MB at most.
"""

from __future__ import annotations

import numpy as np

__all__ = ["INTERPS", "MeshField", "resample_bilinear"]

#: Interpolants a manifest mesh spec may name in ``"interp"``; absent means ``"bilinear"``.
INTERPS = ("bilinear", "cubic")


def _fractional_index(coord, grid):
    """Fractional node index of ``coord`` on a 1-D ascending ``grid``, edge-clamped.

    ``np.interp`` maps a non-uniform grid (the per-tile grid's short last tile) to index space
    exactly and clamps past the outermost nodes, so the cubic field extrapolates as a constant
    like the bilinear one. A length-1 grid maps everything to 0.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.size == 1:
        return np.zeros(np.shape(coord), dtype=float)
    return np.interp(coord, grid, np.arange(grid.size, dtype=float))


def _bilinear_weights(coord, grid):
    """Lower-node index and fractional weight for ``coord`` on a 1-D ascending ``grid``.

    Coordinates are edge-clamped to ``[grid[0], grid[-1]]`` so the field extrapolates as a
    constant past its outermost control points rather than blowing up. A length-1 grid degenerates
    to "always node 0, weight 0" (no interpolation along that axis).
    """
    grid = np.asarray(grid, dtype=float)
    n = grid.size
    coord = np.clip(np.asarray(coord, dtype=float), grid[0], grid[-1])
    if n == 1:
        i0 = np.zeros(coord.shape, dtype=int)
        return i0, i0, np.zeros(coord.shape, dtype=float)
    i1 = np.clip(np.searchsorted(grid, coord, side="right"), 1, n - 1)
    i0 = i1 - 1
    span = grid[i1] - grid[i0]
    t = np.where(span > 0, (coord - grid[i0]) / np.where(span > 0, span, 1.0), 0.0)
    return i0, i1, t


class MeshField:
    """A regular-grid displacement field, bilinearly (default) or cubic-B-spline interpolated.

    Parameters
    ----------
    grid_x, grid_y : 1-D ascending arrays
        Control-node coordinates (tile centres) along x and y.
    displacements : array, shape ``(len(grid_y), len(grid_x), 2)``
        ``(dx, dy)`` displacement at each node.
    interp : ``"bilinear"`` or ``"cubic"``
        ``"cubic"`` is an interpolating cubic B-spline on the node indices (mirror boundary,
        prefiltered once here); it passes through every node like the bilinear field.
    """

    def __init__(self, grid_x, grid_y, displacements, interp="bilinear"):
        self.grid_x = np.asarray(grid_x, dtype=float)
        self.grid_y = np.asarray(grid_y, dtype=float)
        self.disp = np.asarray(displacements, dtype=float)
        expected = (self.grid_y.size, self.grid_x.size, 2)
        if self.disp.shape != expected:
            raise ValueError(
                f"displacements shape {self.disp.shape} != expected {expected} "
                f"(len(grid_y), len(grid_x), 2)"
            )
        if interp not in INTERPS:
            raise ValueError(
                f"unknown mesh interp {interp!r}; expected one of {INTERPS}"
            )
        self.interp = interp
        self._coeffs = None
        if interp == "cubic":
            from scipy.ndimage import spline_filter

            self._coeffs = [
                spline_filter(self.disp[..., k], order=3, mode="mirror")
                for k in range(2)
            ]

    @classmethod
    def from_spec(cls, spec):
        """The field a manifest mesh spec describes, or ``None`` for a rigid-only entry.

        The ONE constructor both manifest readers use -- ``stage_warp.make_warper`` (the QC
        seam) and ``stages/stitch`` (the image) -- so they cannot sample different fields.
        """
        if spec is None:
            return None
        return cls(
            spec["grid_x"],
            spec["grid_y"],
            spec["displacements"],
            interp=spec.get("interp", "bilinear"),
        )

    @classmethod
    def zero(cls, grid_x, grid_y):
        """A field that displaces nothing — the identity warp."""
        gx = np.asarray(grid_x, dtype=float)
        gy = np.asarray(grid_y, dtype=float)
        return cls(gx, gy, np.zeros((gy.size, gx.size, 2)))

    def displacement(self, xy):
        """Interpolated ``(dx, dy)`` at each ``(N, 2)`` point."""
        xy = np.asarray(xy, dtype=float)
        if xy.size == 0:
            return xy.reshape(-1, 2).copy()
        if self._coeffs is not None:
            from scipy.ndimage import map_coordinates

            idx = [
                _fractional_index(xy[:, 1], self.grid_y),
                _fractional_index(xy[:, 0], self.grid_x),
            ]
            return np.stack(
                [
                    map_coordinates(c, idx, order=3, prefilter=False, mode="mirror")
                    for c in self._coeffs
                ],
                axis=1,
            )
        ix0, ix1, tx = _bilinear_weights(xy[:, 0], self.grid_x)
        iy0, iy1, ty = _bilinear_weights(xy[:, 1], self.grid_y)

        d = self.disp
        d00 = d[iy0, ix0]
        d01 = d[iy0, ix1]
        d10 = d[iy1, ix0]
        d11 = d[iy1, ix1]
        tx = tx[:, None]
        ty = ty[:, None]
        top = d00 * (1 - tx) + d01 * tx
        bot = d10 * (1 - tx) + d11 * tx
        return top * (1 - ty) + bot * ty

    def warp_points(self, xy):
        """Move each point by its interpolated displacement: ``xy + displacement(xy)``."""
        xy = np.asarray(xy, dtype=float)
        if xy.size == 0:
            return xy.reshape(-1, 2).copy()
        return xy + self.displacement(xy)


def resample_bilinear(image, xy):
    """Bilinearly sample ``image`` at ``(N, 2)`` ``(x=col, y=row)`` coordinates.

    Returns ``(N,)`` for a 2-D image or ``(N, C)`` for ``(H, W, C)``. Coordinates are edge-clamped
    to the image, so out-of-frame samples take the nearest border value.

    The result is a **convex combination** of the four neighbouring source pixels — the four
    weights are non-negative and sum to 1 — so it always lies within ``[min, max]`` of those
    pixels. A non-negative image therefore stays non-negative: no bicubic/Lanczos overshoot to
    manufacture the negative values that would corrupt downstream marker quantification. This is
    the resampler the per-tile warp and stitch use, deliberately, instead of a higher-order kernel.

    The kernel is ``scipy.ndimage.map_coordinates(order=1)`` on edge-clamped coordinates --
    the same four-tap convex combination as the explicit NumPy form it replaced (equal to
    ~1e-11 on 60000-scale data) at under half the time on a 1024^2 output tile.
    """
    from scipy.ndimage import map_coordinates

    image = np.asarray(image, dtype=float)
    xy = np.asarray(xy, dtype=float)
    h, w = image.shape[:2]
    coords = [np.clip(xy[:, 1], 0.0, h - 1.0), np.clip(xy[:, 0], 0.0, w - 1.0)]
    if image.ndim == 3:
        return np.stack(
            [
                map_coordinates(image[..., c], coords, order=1, mode="nearest")
                for c in range(image.shape[2])
            ],
            axis=1,
        )
    return map_coordinates(image, coords, order=1, mode="nearest")
