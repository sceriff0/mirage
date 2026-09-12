"""Smooth control-grid displacement field + a non-negative bilinear resampler.

These are the geometry primitives of the tiled ('STARE') registration method. The tiled
registrar measures, per tile, one residual displacement at the tile centre; those control
points form a regular grid, and :class:`MeshField` interpolates them into a *single continuous*
displacement field over the whole slide. Warping points or images through one continuous field
is what keeps a cell straddling a tile boundary from being torn — there is never a per-tile
discontinuity to fall into.

Both the forward point-warp (used by the reg_qc=2 stage warper) and the image warp (used by the
per-tile WARP step) sample the *same* field, so the QC measures exactly the transform that
shipped.

Nothing here imports VALIS, pyvips, or a JVM — pure NumPy, a few MB at most.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MeshField", "resample_bilinear"]


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
    """A regular-grid displacement field, bilinearly interpolated.

    Parameters
    ----------
    grid_x, grid_y : 1-D ascending arrays
        Control-node coordinates (tile centres) along x and y.
    displacements : array, shape ``(len(grid_y), len(grid_x), 2)``
        ``(dx, dy)`` displacement at each node.
    """

    def __init__(self, grid_x, grid_y, displacements):
        self.grid_x = np.asarray(grid_x, dtype=float)
        self.grid_y = np.asarray(grid_y, dtype=float)
        self.disp = np.asarray(displacements, dtype=float)
        expected = (self.grid_y.size, self.grid_x.size, 2)
        if self.disp.shape != expected:
            raise ValueError(
                f"displacements shape {self.disp.shape} != expected {expected} "
                f"(len(grid_y), len(grid_x), 2)"
            )

    @classmethod
    def zero(cls, grid_x, grid_y):
        """A field that displaces nothing — the identity warp."""
        gx = np.asarray(grid_x, dtype=float)
        gy = np.asarray(grid_y, dtype=float)
        return cls(gx, gy, np.zeros((gy.size, gx.size, 2)))

    def displacement(self, xy):
        """Bilinearly interpolated ``(dx, dy)`` at each ``(N, 2)`` point."""
        xy = np.asarray(xy, dtype=float)
        if xy.size == 0:
            return xy.reshape(-1, 2).copy()
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
    """
    image = np.asarray(image, dtype=float)
    xy = np.asarray(xy, dtype=float)
    h, w = image.shape[:2]
    multichannel = image.ndim == 3

    x = np.clip(xy[:, 0], 0.0, w - 1.0)
    y = np.clip(xy[:, 1], 0.0, h - 1.0)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    tx = x - x0
    ty = y - y0

    ia = image[y0, x0]
    ib = image[y0, x1]
    ic = image[y1, x0]
    id_ = image[y1, x1]
    if multichannel:
        tx = tx[:, None]
        ty = ty[:, None]
    top = ia * (1 - tx) + ib * tx
    bot = ic * (1 - tx) + id_ * tx
    return top * (1 - ty) + bot * ty
