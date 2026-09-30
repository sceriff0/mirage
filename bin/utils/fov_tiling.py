"""Non-overlapping FOV tiling: the grid BASICPY's inputs are built from.

These functions used to live in ``bin/preprocess.py``, next to the in-process
``BaSiC.fit_transform`` call they were written for. That path is gone -- illumination
correction now runs through nf-core's BASICPY module (``TILE_FOR_BASIC`` -> ``BASICPY``
-> ``APPLY_PROFILES``) -- but the tiling itself is unchanged and is still what makes a
stitched mirage slide look like the multi-site acquisition BaSiC needs to fit a profile.

They live under ``bin/utils/`` because both halves of that three-process path need them
and neither owns the other:

* ``bin/tile_for_basic.py`` calls ``count_fovs`` + ``fov_positions`` to lay out the grid
  and ``iter_padded_fovs`` to stream the pseudo-FOV stack BASICPY fits on;
* ``bin/apply_basic_profiles.py`` calls ``fov_positions`` and then reads the profile back
  through ``fov_overlaps``, one output write-tile at a time.

GEOMETRY IS SEPARATE FROM PIXELS, and that is the whole point of this module's shape.
``fov_positions`` is pure arithmetic on ``(H, W, n_fovs_y, n_fovs_x)``: it allocates
nothing, imports no image library, and can be called before a single byte is read. That
is what lets both callers stream. It is the same split ``bin/utils/tile_grid.py`` makes
for the STARE registration path, deliberately -- the two tilings solve different problems
(see "WHY THIS GRID IS NOT tile_grid's" below) but they are built the same way, so a
reader who has understood one has understood the other.

The split is EXACT and non-overlapping: remainder pixels are distributed across tiles
(some get +1 pixel) rather than padded, so ``reconstruct_image_from_fovs`` applied to
``split_image_into_fovs``' output is the identity. ``count_fovs`` is the matching grid
count, ``ceil(size / fov)`` on each axis.

THE ZERO-PADDING OF EDGE TILES IS REQUIRED BY BaSiC, not inherited by accident.
``split_image_into_fovs`` and ``iter_padded_fovs`` both pad every tile up to the grid's
maximum tile size. BaSiCPy's ``fit()`` takes ONE ``(N, Y, X)`` ndarray and resizes it with
``_resize_to_working_size``; there is no ragged-input path, so fields of differing size
cannot be passed at all. Since the remainder distribution above makes tiles differ by at
most one pixel per axis, the padding is at most a one-pixel border on some tiles.

It is not free, and the cost is worth stating: BaSiCPy normalises the fitted flatfield by
its own mean (``S = S / torch.mean(S)``), so padded zeros pull that mean down slightly.
With a <=1 px border on a ~1950 px tile that is a ~0.1% effect at the very edge of the
profile; growing the padding -- e.g. by adopting a grid whose tiles differ by more than a
pixel -- would grow it in proportion. Do not "fix" the padding away: removing it does not
produce a smaller input, it produces one ``fit()`` cannot accept.

WHY THIS GRID IS NOT ``tile_grid.py``'s. They are not interchangeable and unifying them
would change what BaSiC fits on:

* ``tile_grid`` uses a FIXED tile size with a short last cell, because the STARE stitch
  needs cores that partition the image at predictable offsets. On a 4097 px axis at
  tile=2048 that yields cells of 2048, 2048 and **1**.
* This module BALANCES the cells (1366, 1366, 1365 for the same axis), because a one-pixel
  wide "field of view" is not a field of view -- BaSiC estimates shading across a
  POPULATION of comparable fields, and a degenerate cell is a degenerate sample.
* ``tile_grid`` cells carry a HALO for registration context. These do not: illumination
  profiles are fitted per field, and overlap between fields would double-count pixels.
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

    Deliberately shaped like :class:`tile_grid.Tile`: index into the grid, then the owned
    region. There is no ``read`` field because this grid has no halo -- ``core`` is the
    whole of it.

    Attributes
    ----------
    ix, iy : int
        Column / row index into the FOV grid.
    y0, x0 : int
        Top-left corner of the owned region, in image pixel coordinates.
    h, w : int
        Height and width of the owned region. Cells differ by at most one pixel per axis.
    index : int
        Position on the tile stack's site axis -- row-major, ``iy * n_fovs_x + ix``. This
        is the ordering ``positions`` is written in and the order the sites must be
        written to the ``CZYX`` stack in.
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

    Tiling is non-overlapping, matching split_image_into_fovs, which extracts
    adjacent tiles and distributes the remainder pixels across them.
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

    Pure arithmetic: allocates nothing image-sized and touches no pixels. Row-major, so
    the returned order IS the site order on the tile stack's Z axis.
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

    This is the function that replaces the old "split a channel just to find out where the
    tiles are" step. ``positions`` is the sidecar's list of ``(y_start, x_start, h, w)``
    in site order and ``tile_shape`` is the ``(max_h, max_w)`` every tile is padded up to;
    both are identical to what ``split_image_into_fovs`` returns for the same grid, which
    is pinned by ``tests/test_fov_tiling.py``.
    """
    fovs = fov_grid(image_shape, n_fovs_y, n_fovs_x)
    positions = [f.position for f in fovs]
    max_h = max(f.h for f in fovs)
    max_w = max(f.w for f in fovs)
    return positions, (max_h, max_w)


def fov_overlaps(positions, y0: int, x0: int, h: int, w: int):
    """Which FOVs a ``(y0, x0, h, w)`` output window touches, and where.

    The inverse lookup ``bin/apply_basic_profiles.py`` needs to correct one write-tile at a
    time: an illumination profile is indexed in FOV-LOCAL coordinates, so a window that
    straddles a FOV boundary must be corrected piecewise. This is the analogue of
    ``tiled_warp.source_region`` on the STARE path -- geometry that says which source
    pixels an output tile draws from, computed before anything is read.

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

    ``read_region(y0, x0, h, w)`` returns that region of one channel -- typically a lazy
    zarr region read, so only the FOV is ever decoded. Peak memory is ONE padded tile,
    against ``split_image_into_fovs``' whole padded stack plus the whole source plane.

    A FRESH buffer is allocated per tile rather than a reused one: the consumer is
    ``tifffile``'s iterator-fed writer, and handing it a buffer that the next iteration
    overwrites is the kind of aliasing bug that shows up as a silently duplicated tile.
    ``bin/tiled_stitch.py:stream_tiles`` allocates per tile for the same reason.
    """
    for y0, x0, h, w in positions:
        tile = np.zeros(tile_shape, dtype=dtype)
        tile[:h, :w] = read_region(y0, x0, h, w)
        yield tile


def split_image_into_fovs(
    image: NDArray, n_fovs_x: int, n_fovs_y: int
) -> Tuple[NDArray, List[Tuple[int, int, int, int]], Tuple[int, int]]:
    """Split image (H, W) into FOV tiles with adaptive sizing to handle remainders.

    The image is divided into n_fovs_y * n_fovs_x tiles. Remainder pixels are
    distributed across tiles (some tiles get +1 pixel) to exactly cover the image
    without padding, and each tile is then zero-padded up to the grid's maximum tile
    size so the stack is the uniform ``(N, Y, X)`` BaSiC requires.

    THIS MATERIALISES THE WHOLE STACK and is retained as the reference definition of the
    grid -- it is what ``tests/test_fov_tiling.py`` pins ``fov_positions`` against, and
    what ``tests/test_tile_for_basic.py`` reconstructs from. Production callers stream
    through ``fov_positions`` + ``iter_padded_fovs`` instead; a slide-sized caller of this
    function is a memory bug.
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

    The exact inverse of :func:`split_image_into_fovs`. Like it, this holds the whole
    image; it survives as the round-trip oracle the tests check the streaming path
    against, not as a production reassembly step -- ``bin/apply_basic_profiles.py`` writes
    its output tile by tile and never reassembles anything in memory.
    """
    reconstructed = np.zeros(original_shape, dtype=fov_stack.dtype)

    for idx, (row_start, col_start, h, w) in enumerate(positions):
        # fov_stack is 3D (N, fov_h, fov_w), reconstructed is 2D (H, W)
        reconstructed[row_start : row_start + h, col_start : col_start + w] = fov_stack[
            idx, :h, :w
        ]

    return reconstructed
