"""Non-overlapping FOV tiling: the grid BASICPY's inputs are built from.

Illumination correction runs TILE_FOR_BASIC -> BASICPY -> APPLY_PROFILES. Tiling a
stitched slide into pseudo-FOVs gives BaSiC the multi-field population it fits on.

* ``bin/tile_for_basic.py`` uses ``count_fovs`` + ``fov_positions`` for the grid and
  ``iter_padded_fovs`` to stream the FOV stack;
* ``bin/apply_basic_profiles.py`` uses ``fov_positions`` and ``fov_overlaps`` to apply the
  profile one output tile at a time.

``fov_positions`` is pure arithmetic on ``(H, W, n_fovs_y, n_fovs_x)`` and reads no
pixels, which is what lets both callers stream.

The split is exact and non-overlapping: remainder pixels are spread across tiles (some
get +1), so ``reconstruct_image_from_fovs(split_image_into_fovs(x))`` is the identity.
``count_fovs`` is ``ceil(size / fov)`` per axis.

Edge tiles are zero-padded to the largest tile size because BaSiCPy's ``fit()`` takes one
``(N, Y, X)`` array. The padding is at most a 1 px border, a ~0.1% effect on the
mean-normalised flatfield; do not remove it, and do not adopt a grid whose tiles differ by
more than a pixel.

Not ``tile_grid.py``'s grid: that one uses a fixed tile size with a short last cell (a
4097 px axis at 2048 gives 2048, 2048, 1) and a halo for registration context. This one
balances cells (1366, 1366, 1365) because a degenerate field is a degenerate BaSiC sample,
and has no overlap, which would double-count pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Fov",
    "count_fovs",
    "fov_sizes",
    "fov_positions",
    "fov_grid",
    "fov_overlaps",
    "iter_padded_fovs",
    "split_image_into_fovs",
    "reconstruct_image_from_fovs",
]


@dataclass(frozen=True)
class Fov:
    """One pseudo-FOV of the grid.

    Shaped like :class:`tile_grid.Tile`, without a ``read`` field (this grid has no halo).

    Attributes
    ----------
    ix, iy : int
        Column / row index into the FOV grid.
    y0, x0 : int
        Top-left corner of the owned region, in image pixel coordinates.
    h, w : int
        Height and width of the owned region. Cells differ by at most one pixel per axis.
    index : int
        Row-major site index, ``iy * n_fovs_x + ix``: the order of ``positions`` and of
        the sites in the ``CZYX`` stack.
    """

    iy: int
    ix: int
    y0: int
    x0: int
    h: int
    w: int
    index: int

    @property
    def position(self) -> Tuple[int, int, int, int]:
        """The sidecar's ``(y_start, x_start, h, w)`` 4-tuple for this FOV."""
        return (self.y0, self.x0, self.h, self.w)


def count_fovs(
    image_shape: Tuple[int, int], fov_size: Tuple[int, int]
) -> Tuple[int, int]:
    """Calculate how many FOVs are needed to cover an image with a given FOV size.

    Non-overlapping, matching split_image_into_fovs.
    """
    height, width = image_shape[:2]
    fov_h, fov_w = fov_size

    if fov_h <= 0 or fov_w <= 0:
        raise ValueError(f"FOV size must be positive, got {fov_size}")

    n_fovs_y = math.ceil(height / fov_h)
    n_fovs_x = math.ceil(width / fov_w)

    return n_fovs_y, n_fovs_x


def fov_sizes(size: int, n_fovs: int) -> List[int]:
    """Per-cell extents along one axis: ``size // n`` each, with the first ``size % n``
    cells taking one extra pixel. Sums to ``size`` exactly -- no padding, no overlap.
    """
    if n_fovs <= 0:
        raise ValueError("Number of FOVs must be positive")
    base, remainder = divmod(size, n_fovs)
    return [base + (1 if i < remainder else 0) for i in range(n_fovs)]


def fov_grid(image_shape, n_fovs_y: int, n_fovs_x: int) -> List[Fov]:
    """Build the list of :class:`Fov` covering a ``height`` x ``width`` image.

    Pure arithmetic, no pixels read. Row-major, so the order is the stack's site order.
    """
    height, width = image_shape[:2]
    heights = fov_sizes(int(height), n_fovs_y)
    widths = fov_sizes(int(width), n_fovs_x)

    fovs: List[Fov] = []
    y0 = 0
    index = 0
    for iy in range(n_fovs_y):
        x0 = 0
        for ix in range(n_fovs_x):
            fovs.append(
                Fov(
                    iy=iy,
                    ix=ix,
                    y0=y0,
                    x0=x0,
                    h=heights[iy],
                    w=widths[ix],
                    index=index,
                )
            )
            x0 += widths[ix]
            index += 1
        y0 += heights[iy]
    return fovs


def fov_positions(image_shape, n_fovs_y: int, n_fovs_x: int):
    """``(positions, tile_shape)`` for a grid -- WITHOUT reading or allocating the image.

    ``positions`` is the sidecar's list of ``(y_start, x_start, h, w)`` in site order and
    ``tile_shape`` the ``(max_h, max_w)`` each tile is padded to; both equal what
    ``split_image_into_fovs`` returns for the same grid.
    """
    fovs = fov_grid(image_shape, n_fovs_y, n_fovs_x)
    positions = [f.position for f in fovs]
    max_h = max(f.h for f in fovs)
    max_w = max(f.w for f in fovs)
    return positions, (max_h, max_w)


def fov_overlaps(positions, y0: int, x0: int, h: int, w: int):
    """Which FOVs a ``(y0, x0, h, w)`` output window touches, and where.

    Used by ``bin/apply_basic_profiles.py``: profiles are in FOV-local coordinates, so a
    window straddling FOVs is corrected piecewise.

    Yields ``(index, window_slices, profile_slices)`` where ``window_slices`` index into
    the ``(h, w)`` window and ``profile_slices`` index into that FOV's ``(max_h, max_w)``
    profile plane. The yielded windows partition the overlap exactly, so assigning through
    all of them writes every pixel of the window once.
    """
    y1, x1 = y0 + h, x0 + w
    for index, (fy, fx, fh, fw) in enumerate(positions):
        iy0, iy1 = max(y0, fy), min(y1, fy + fh)
        ix0, ix1 = max(x0, fx), min(x1, fx + fw)
        if iy1 <= iy0 or ix1 <= ix0:
            continue
        window = (slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0))
        profile = (slice(iy0 - fy, iy1 - fy), slice(ix0 - fx, ix1 - fx))
        yield index, window, profile


def iter_padded_fovs(read_region, positions, tile_shape, dtype) -> Iterator[NDArray]:
    """Yield each FOV as a zero-padded ``tile_shape`` plane, one at a time.

    ``read_region(y0, x0, h, w)`` returns that region of one channel (typically a lazy
    zarr read). Peak memory is one padded tile.

    A fresh buffer per tile: tifffile's iterator-fed writer may still hold the previous
    one, and reusing it would silently duplicate tiles.
    """
    for y0, x0, h, w in positions:
        tile = np.zeros(tile_shape, dtype=dtype)
        tile[:h, :w] = read_region(y0, x0, h, w)
        yield tile


def split_image_into_fovs(
    image: NDArray, n_fovs_x: int, n_fovs_y: int
) -> Tuple[NDArray, List[Tuple[int, int, int, int]], Tuple[int, int]]:
    """Split image (H, W) into FOV tiles with adaptive sizing to handle remainders.

    Remainder pixels go to the first tiles (+1 each); every tile is zero-padded to the
    largest size so the stack is the uniform ``(N, Y, X)`` BaSiC requires.

    Holds the whole stack in memory: it is the reference definition the tests check
    against. Production code streams via ``fov_positions`` + ``iter_padded_fovs``.
    """
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D, got shape {image.shape}")

    if n_fovs_x <= 0 or n_fovs_y <= 0:
        raise ValueError("Number of FOVs must be positive")

    positions, tile_shape = fov_positions(image.shape, n_fovs_y, n_fovs_x)
    max_h, max_w = tile_shape

    fov_stack = np.zeros((len(positions), max_h, max_w), dtype=image.dtype)
    for idx, (y_start, x_start, h, w) in enumerate(positions):
        fov_stack[idx, :h, :w] = image[y_start : y_start + h, x_start : x_start + w]

    return fov_stack, positions, tile_shape


def reconstruct_image_from_fovs(
    fov_stack: NDArray,
    positions: List[Tuple[int, int, int, int]],
    original_shape: Tuple[int, ...],
) -> NDArray:
    """Reconstruct 2D image from 3D FOV tiles stack.

    Exact inverse of :func:`split_image_into_fovs`; holds the whole image, so it is a test
    oracle, not a production step.
    """
    reconstructed = np.zeros(original_shape, dtype=fov_stack.dtype)

    for idx, (row_start, col_start, h, w) in enumerate(positions):
        # fov_stack is 3D (N, fov_h, fov_w), reconstructed is 2D (H, W)
        reconstructed[row_start : row_start + h, col_start : col_start + w] = fov_stack[
            idx, :h, :w
        ]

    return reconstructed
