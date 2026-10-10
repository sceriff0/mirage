"""A stand-in for the ``palom`` package, for the adapter's tests where it is not installed.

It does three things and nothing else:

* **records** every call the adapter makes into it (``CALLS``): the readers and their
  arguments, ``Aligner`` with its keywords, the three registration calls with theirs, the
  two writer calls;
* keeps the package's **bookkeeping**, written from labsyspharm/palom at ``v2026.4.1``
  ``[PARTIAL: the functions named here]``: how the reader turns a TIFF into dask arrays
  and builds its own levels for a one-level file (``reader.py:41-87, 111-136``),
  ``Aligner.affine_matrix`` (``align.py:172-184``), ``grid_shape`` (``:213-215``),
  ``block_affine_matrices`` and ``block_affine_matrices_da`` (``:113-135``),
  ``block_affine_transformed_moving_img`` (``:13-30``) and what ``constrain_shifts`` keeps
  (``:221-227``);
* **draws** the registered image the way ``block_affine.py:8-113`` is written to
  (``draw_block``): one call per dask block of the reference's chunks, with that block's
  own 3 x 3 matrix; the block's corners taken through the matrix's inverse to find the
  source region (one pixel more on each side, clipped at zero); the source sampled at the
  inverse of each output pixel centre, linearly, zero outside the region. The sampling is
  ``scipy.ndimage.map_coordinates`` here and ``cv2.warpAffine`` in the package: an
  emulation written from the source, never the real writer.

It does NOT register anything: ``coarse_register_affine`` invents an affine and
``compute_shifts`` invents whole-pixel shifts, some of them blank (infinite), and
``constrain_shifts`` replaces the blank ones and a few others by fractional values. The
selfcheck needs a writer and a point path, not a good registration.
"""

from __future__ import annotations

import math
import re
import sys
import types
from pathlib import Path

import numpy as np

CALLS = []
# What a test may set before the adapter runs.
CONFIG = {
    "version": "2026.4.1",
    "rotation_deg": 1.0,  # the invented affine, moving -> reference
    "scale": 1.002,
    "shift_px": (11.0, -7.0),
    "max_shift": 5,  # block shifts are whole pixels in [-max_shift, max_shift]
    "blank": 0.1,  # share of blocks whose measured shift is (inf, inf)
    "replaced": 0.2,  # share of the other blocks the filter replaces
    "identity": False,  # feature matching failed: the identity placeholder
    "leave_nonfinite": False,  # the filter leaves a blank block as it is
    "constrain_raises": None,  # the text of an exception raised by constrain_shifts
    "figure": False,  # the coarse call opens a matplotlib figure, as the package does
    "closed_file": False,  # reads raise "seek of closed file" until STATE says otherwise
    "seed": 0,
}
DEFAULTS = dict(CONFIG)
STATE = {"closed": False}
TILE = 1024  # write_pyramid's default tile_size (pyramid.py:113-125)


def configure(**config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown fake settings {sorted(unknown)}")
    CONFIG.clear()
    CONFIG.update(DEFAULTS)
    CONFIG.update(config)
    STATE["closed"] = bool(CONFIG["closed_file"])
    del CALLS[:]


def record(name, **kw):
    CALLS.append({"call": name, **kw})


def calls(name):
    return [c for c in CALLS if c["call"] == name]


# ── the reader (reader.py:17-165) ─────────────────────────────────────────────
def num_levels(base_shape, max_size=1024, factor=2):
    """``pyramid.py:34-36`` (``PyramidSetting.num_levels``)."""
    return max(math.ceil(math.log(max(base_shape) / max_size, factor)) + 1, 1)


def _gate(block):
    if STATE["closed"]:
        raise ValueError("seek of closed file")
    return block


class OmePyramidReader:
    def __init__(self, path, pixel_size=None):
        import dask.array as da
        import tifffile
        import zarr

        record("OmePyramidReader", path=str(path), pixel_size=pixel_size)
        self.path = Path(path)
        self._pixel_size = pixel_size
        with tifffile.TiffFile(self.path) as tif:  # closed before any pixel is read
            levels = tif.series[0].levels if len(tif.series) == 1 else tif.series
            stores = [zarr.open(level.aszarr(), "r") for level in levels]
        pyramid = []
        for z in stores:
            level = da.from_zarr(z[0] if hasattr(z, "keys") else z, name=False)
            level = level.squeeze()
            if level.ndim == 2:
                level = level.reshape(1, *level.shape)
            if CONFIG["closed_file"]:
                level = level.map_blocks(_gate, dtype=level.dtype, meta=level._meta)
            pyramid.append(level)
        first = pyramid[0]
        self.file_levels = len(pyramid)
        if len(pyramid) == 1 and max(first.shape) >= 1024:
            # auto_format_pyramid (reader.py:45-68): its own levels, by 2 x 2 means
            if first.numblocks[1:3] == (1, 1):
                first = first.rechunk((1, 1024, 1024))
            pyramid = [
                da.coarsen(
                    np.mean, first, {0: 1, 1: 2**i, 2: 2**i}, trim_excess=True
                ).astype(first.dtype)
                for i in range(num_levels(first.shape[1:3]))
            ]
        self.pyramid = pyramid

    def read_level_channels(self, level, channels):
        record(
            "read_level_channels", path=str(self.path), level=level, channels=channels
        )
        return self.pyramid[level][channels]

    @property
    def level_downsamples(self):
        shapes = [level.shape[1:3] for level in self.pyramid]
        out, total = {}, 1.0
        for i, shape in enumerate(shapes):
            if i:
                total *= float(np.divide(shapes[i - 1], shape).mean())
            out[i] = total
        return out

    def get_thumbnail_level_of_size(self, size):
        record("get_thumbnail_level_of_size", path=str(self.path), size=size)
        gaps = [abs(np.mean(level.shape[1:3]) - size) for level in self.pyramid]
        return np.argmin(gaps)  # a numpy integer, as in the package

    @property
    def pixel_size(self):
        if self._pixel_size is None:
            import tifffile

            with tifffile.TiffFile(self.path) as tif:
                found = re.search(
                    r'PhysicalSizeX="([0-9.eE+-]+)"', tif.ome_metadata or ""
                )
            self._pixel_size = float(found.group(1)) if found else 1
        return self._pixel_size


# ── the aligner (align.py:113-240) ────────────────────────────────────────────
def block_affine_matrices(mx, shifts):
    """One matrix per block: ``T(dx, dy) @ mx`` for a shift given as ``(dy, dx)``."""
    out = []
    for shift in shifts:
        y, x = shift
        move = np.eye(3)
        move[:2, 2] = x, y
        out.append(move @ mx)
    return np.array(out)


def block_affine_matrices_da(mxs, grid_shape):
    """The matrices as one array of 3 x 3 chunks, block ``(r, c)`` at chunk ``(r, c)``."""
    import dask.array as da

    rows, cols = grid_shape
    grid = np.arange(rows * cols).reshape(grid_shape)
    return da.from_array(np.vstack([np.hstack(mxs[r]) for r in grid]), chunks=3)


def draw_block(affine_matrix, src_array=None, block_info=None, is_mask=False):
    """One output block, as ``block_affine_dask`` and ``block_affine`` draw it."""
    from scipy.ndimage import map_coordinates

    (y0, y1), (x0, x1) = block_info[None]["array-location"]
    height, width = block_info[None]["chunk-shape"]
    matrix = np.asarray(affine_matrix, dtype=float)
    inverse = np.linalg.inv(matrix)
    corners = np.array(
        [[x, y, 1.0] for x in (x0, x0 + width) for y in (y0, y0 + height)]
    )
    back = corners @ inverse.T
    sx1, sy1 = np.ceil(np.clip(back[:, :2].max(axis=0) + 1, 0, None)).astype(int)
    sx0, sy0 = np.floor(np.clip(back[:, :2].min(axis=0) - 1, 0, None)).astype(int)
    source = np.asarray(src_array[sy0:sy1, sx0:sx1])
    if 0 in source.shape:
        return np.zeros((height, width), dtype=src_array.dtype)
    yy, xx = np.mgrid[y0 : y0 + height, x0 : x0 + width].astype(float)
    at_x = inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2] - sx0
    at_y = inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2] - sy0
    drawn = map_coordinates(
        source.astype(np.float64),
        [at_y, at_x],
        order=0 if is_mask else 1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    if np.issubdtype(src_array.dtype, np.integer):
        drawn = np.rint(drawn)
    return drawn.astype(src_array.dtype)


def block_affine_transformed_moving_img(ref_img, moving_img, mxs, is_mask=False):
    import dask.array as da

    record(
        "block_affine_transformed_moving_img",
        ref_chunks=tuple(tuple(int(v) for v in c) for c in ref_img.chunks),
        moving_shape=tuple(int(v) for v in moving_img.shape),
        mxs_shape=tuple(int(v) for v in mxs.shape),
        mxs_is_dask=isinstance(mxs, da.Array),
    )
    assert ref_img.ndim == 2
    one = moving_img.ndim == 2
    if one:
        moving_img = moving_img[np.newaxis]
    drawn = da.array(
        [
            da.map_blocks(
                draw_block,
                mxs,
                src_array=channel,
                chunks=ref_img.chunks,
                dtype=moving_img.dtype,
                is_mask=is_mask,
            )
            for channel in moving_img
        ]
    )
    return drawn[0] if one else drawn


def invented_affine(ref_hw):
    """moving (x, y, 1) -> reference (x, y, 1): a small rotation, scale and shift."""
    if CONFIG["identity"]:
        return np.eye(3)
    th = math.radians(CONFIG["rotation_deg"])
    s = float(CONFIG["scale"])
    tx, ty = CONFIG["shift_px"]
    return np.array(
        [
            [s * math.cos(th), -s * math.sin(th), tx],
            [s * math.sin(th), s * math.cos(th), ty],
            [0.0, 0.0, 1.0],
        ]
    )


class Aligner:
    def __init__(self, **keywords):
        record("Aligner", keywords=sorted(keywords))
        for key, value in keywords.items():
            setattr(self, key, value)

    def coarse_register_affine(self, **kwargs):
        record("coarse_register_affine", kwargs=dict(kwargs))
        if CONFIG["figure"]:
            import matplotlib.pyplot as plt

            plt.subplots()
        full = invented_affine(self.ref_img.shape)
        if CONFIG["identity"]:  # register.py:152-154
            self.coarse_affine_matrix = np.eye(3)
            return
        to_ref = np.diag([1 / self.ref_thumbnail_down_factor] * 2 + [1.0])
        to_mov = np.diag([1 / self.moving_thumbnail_down_factor] * 2 + [1.0])
        self.coarse_affine_matrix = to_ref @ full @ np.linalg.inv(to_mov)

    @property
    def affine_matrix(self):
        to_ref = np.diag([1 / self.ref_thumbnail_down_factor] * 2 + [1.0])
        to_mov = np.diag([1 / self.moving_thumbnail_down_factor] * 2 + [1.0])
        return np.linalg.inv(to_ref) @ self.coarse_affine_matrix.copy() @ to_mov

    @property
    def grid_shape(self):
        return self.ref_img.numblocks

    def compute_shifts(self, *args, **kwargs):
        record("compute_shifts", args=args, kwargs=dict(kwargs))
        if STATE["closed"]:
            raise ValueError("seek of closed file")
        rows, cols = self.grid_shape
        rng = np.random.default_rng([int(CONFIG["seed"]), rows, cols])
        m = int(CONFIG["max_shift"])
        shifts = rng.integers(-m, m + 1, (rows * cols, 2)).astype(np.float32)
        blank = rng.random(rows * cols) < float(CONFIG["blank"])
        shifts[blank] = np.inf
        self.shifts = shifts

    def constrain_shifts(self, *args, **kwargs):
        record("constrain_shifts", args=args, kwargs=dict(kwargs))
        if CONFIG["constrain_raises"]:
            raise ValueError(CONFIG["constrain_raises"])
        if not hasattr(self, "original_shifts"):
            self.original_shifts = self.shifts.copy()
        rows, cols = self.grid_shape
        shifts = self.original_shifts.copy()
        rng = np.random.default_rng([int(CONFIG["seed"]), 7])
        blank = ~np.isfinite(shifts).all(axis=1)
        replace = rng.random(rows * cols) < float(CONFIG["replaced"])
        if not CONFIG["leave_nonfinite"]:
            replace |= blank
        coords = np.indices((rows, cols)).reshape(2, -1).T
        fitted = np.stack(
            [0.37 * coords[:, 0] - 0.21 * coords[:, 1] + 0.4, 0.5 - 0.3 * coords[:, 1]],
            axis=1,
        )
        if not CONFIG["max_shift"]:
            fitted[:] = 0.0
        shifts[replace] = fitted[replace]
        self.shifts = shifts.astype(np.float32)

    @property
    def block_affine_matrices(self):
        return block_affine_matrices(self.affine_matrix, self.shifts)

    @property
    def block_affine_matrices_da(self):
        return block_affine_matrices_da(self.block_affine_matrices, self.grid_shape)


# ── the writer (pyramid.py:113-220) ───────────────────────────────────────────
def write_pyramid(mosaics, output_path, **kwargs):
    """Every channel of every mosaic into one tiled OME-TIFF with one reduced level in a
    SubIFD: the layout of the package's file, not its reduction."""
    import tifffile

    record(
        "write_pyramid",
        output_path=str(output_path),
        kwargs=dict(kwargs),
        n_mosaics=len(mosaics),
        shapes=[tuple(int(v) for v in m.shape) for m in mosaics],
    )
    planes = []
    for m in mosaics:
        m = m[np.newaxis] if m.ndim == 2 else m
        planes.append(np.asarray(m.compute() if hasattr(m, "compute") else m))
    data = np.concatenate(planes)
    px = float(kwargs.get("pixel_size", 1))
    tile = int(kwargs.get("tile_size", TILE))
    meta = {"axes": "CYX", "PhysicalSizeX": px, "PhysicalSizeY": px}
    with tifffile.TiffWriter(str(output_path), bigtiff=True) as tif:
        common = {
            "tile": (tile, tile),
            "compression": "zlib",
            "photometric": "minisblack",
        }
        tif.write(data, subifds=1, metadata=meta, **common)
        tif.write(data[:, ::2, ::2], subfiletype=1, **common)


# ── installing ────────────────────────────────────────────────────────────────
def modules():
    """The fake modules, by the names the adapter imports."""
    package = types.ModuleType("palom")
    package.__version__ = CONFIG["version"]
    package.__file__ = str(Path(__file__).with_name("palom") / "__init__.py")
    package.reader = types.SimpleNamespace(OmePyramidReader=OmePyramidReader)
    package.align = types.SimpleNamespace(
        Aligner=Aligner,
        block_affine_matrices=block_affine_matrices,
        block_affine_matrices_da=block_affine_matrices_da,
        block_affine_transformed_moving_img=block_affine_transformed_moving_img,
    )
    package.pyramid = types.SimpleNamespace(write_pyramid=write_pyramid)
    return {"palom": package}


def install(setitem=None, **config):
    """Configure the fake and put it in ``sys.modules`` (through ``monkeypatch.setitem``
    when given, so a test leaves nothing behind)."""
    configure(**config)
    put = setitem or (lambda mapping, key, value: mapping.__setitem__(key, value))
    for name, mod in modules().items():
        put(sys.modules, name, mod)
