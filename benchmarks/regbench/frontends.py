"""Three stitching front-ends over one tile series (``tiles.py``), one shared mosaic writer.

``stitch(cycle, name, out)`` places every tile of one cycle and writes the cycle's mosaic to
``out/mosaic.ome.tif``. The front-ends differ ONLY in the positions they find:

* ``stage``        the stage positions of ``tiles.csv``;
* ``ashlar_edge``  ASHLAR's ``reg.EdgeAligner`` (in the ASHLAR image, ``REGBENCH_SIF_ASHLAR``)
                   over ``reader_of(cycle)``, a minimal ``reg.Reader``/``reg.Metadata`` over
                   the same tile files, so ASHLAR sees exactly the tiles the others see;
* ``stare_mosaic`` ``stare mosaic`` (STARE >= 1.3.0), as a subprocess of the interpreter that
                   runs this (in ``REGBENCH_SIF_STARE_REGBENCH``, where ``prepare`` runs; the
                   same ``[sys.executable, "-m", "stare.cli", <stage>]`` as ``methods/stare.py``).

Every front-end's positions are moved to a zero origin (a reader may give a negative one:
``bioformats_tiles`` keeps ASHLAR's frame). ``stare mosaic`` is handed the stage positions as
they are: its solve moves its result to a zero origin itself (``stare/mosaic.py:281``,
``pos -= pos.min(axis=0)``; STARE 1.3.0 ``[PARTIAL: stare/mosaic.py 75-160, 260-289;
stare/stages/mosaic.py in full]``) and its pair geometry is relative, which
``test_frontends.py::test_stare_mosaic_recovers_the_jitter`` checks on a negative origin.

Units, ASHLAR 1.20.0 (read from the sdist, not run: ``[PARTIAL: reg.py 50-120, 180-200,
430-840, 1036-1175; utils.py 92-98, 149-237; thumbnail.py 1-80]``; skimage's float/int
conversion ``[PARTIAL: skimage 0.24 util/dtype.py _convert]``): ``metadata.positions`` are
PIXELS (row, col);
``EdgeAligner(max_shift=...)`` is in MICROMETRES (``reg.py:504-506``: ``# Unit is
micrometers.``, ``self.max_shift_pixels = self.max_shift / self.metadata.pixel_size``).

Illumination: when ``cycle.ffp``/``cycle.dfp`` are set, every front-end reads tiles corrected
with ASHLAR's own arithmetic (``reg.Mosaic._load_correction_profiles`` and
``correct_illumination``, ``reg.py:1124-1134, 1165-1171``, then ``utils.paste``'s conversion,
``utils.py:186-188``)::

    dfp /= np.iinfo(dtype).max                       # once, at load
    img = skimage.util.img_as_float(img)             # raw * (1 / max), float64
    img -= dfp ; img /= ffp ; img.clip(0, 1)         # ffp used as given (BaSiC's is mean 1)
    dtype_convert(img, dtype)                        # rint(img * max), clipped

A missing profile is ASHLAR's neutral one (dark 0, flat 1). A profile holding every channel
(``(C, H, W)``, as MCMICRO ships per cycle) gives its nuclear plane. ``stare_mosaic`` is given
corrected tile files, written once under ``out/corrected/``.

MEMORY (user ruling 2026-10-14): O(one tile + one output write-tile), independent of the grid.
``write_mosaic`` assembles ``out_tile``-px blocks one at a time from only the tiles whose box
intersects the block, each read as that WINDOW (``tifffile.memmap``; a compressed tile is
read whole, one at a time), corrected per window (the profiles are one tile, loaded once),
and streamed into a tiled BigTIFF by ``imageio.write_nuclear_tiles`` (one encoder thread).
ASHLAR's ``EdgeAligner`` wraps the reader it is given in a ``CachingReader``
(``reg.py:502``: ``self.reader = CachingReader(reader, self.channel)``) whose ``read`` keeps
every tile of the align channel (``reg.py:457-462``) -- O(n_tiles). The aligner's reader is
therefore set back to the uncached tile-series reader before ``run()``; ASHLAR's thumbnail
(``thumbnail.make_thumbnail``, 5 % scale or <= 1000 px long side) stays its only whole-slide
buffer and its size is recorded in the report.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tifffile

from . import imageio, tiles

FRONT_ENDS = ("stage", "ashlar_edge", "stare_mosaic")
DEFAULT_MAX_SHIFT_UM = 15.0
OUT_TILE = 1024
MOSAIC = "mosaic.ome.tif"
STARE_POSITIONS = "stare_positions.json"
CORRECTED = "corrected"
STARE_MIN_VERSION = (1, 3, 0)
PASTE = "last-wins in tiles.csv order, at positions rounded to whole pixels"
ASHLAR_FORMULA = (
    "ashlar 1.20.0 Mosaic.correct_illumination: clip((raw/max - dfp/max) / ffp, 0, 1), "
    "then rint(* max)"
)


class Stitched(NamedTuple):
    mosaic: Path
    positions: dict[int, tuple[float, float]]  # tile -> (x, y) px, zero origin
    report: dict


# ── positions ────────────────────────────────────────────────────────────────
def _zero_origin(positions):
    if not positions:
        raise ValueError("no positions")
    x0 = min(float(x) for x, _ in positions.values())
    y0 = min(float(y) for _, y in positions.values())
    return {int(t): (float(x) - x0, float(y) - y0) for t, (x, y) in positions.items()}


def stage_positions(cycle: tiles.Cycle) -> dict[int, tuple[float, float]]:
    """The stage positions of ``tiles.csv``, moved to a zero origin."""
    return _zero_origin({r.tile: (r.x_px, r.y_px) for r in cycle.tiles})


# ── reading one tile, or one window of it ────────────────────────────────────
def _read_window(row: tiles.TileRow, rows=slice(None), cols=slice(None)):
    """A copy of ``row``'s plane in ``[rows, cols]``: a read-only mapping when the TIFF is
    uncompressed and contiguous (only those pages are touched), else the one plane read
    whole. Nothing references the file afterwards."""
    try:
        arr = tifffile.memmap(str(row.path), mode="r")
    except ValueError:  # compressed or not contiguous
        arr = tifffile.imread(str(row.path))
    if arr.ndim == 3:
        arr = arr[row.channel]
    win = np.array(arr[rows, cols])
    del arr
    return win


def _tile_dtype(cycle):
    with tifffile.TiffFile(str(cycle.tiles[0].path)) as tf:
        return np.dtype(tf.pages[0].dtype)


# ── illumination ─────────────────────────────────────────────────────────────
class Illumination:
    """A cycle's flat-/dark-field profiles (its nuclear plane, one tile in size, loaded
    once) and ASHLAR's correction of any window of a tile."""

    def __init__(self, ffp, dfp, dtype, paths):
        self.ffp, self.dfp, self.dtype, self.paths = ffp, dfp, np.dtype(dtype), paths
        self.imax = np.iinfo(self.dtype).max

    @classmethod
    def of(cls, cycle: tiles.Cycle):
        """None when the cycle names no profile."""
        if cycle.ffp is None and cycle.dfp is None:
            return None
        dtype = _tile_dtype(cycle)
        if dtype.kind != "u":
            raise ValueError(
                f"{cycle.dir}: illumination correction needs unsigned-integer tiles "
                f"(ASHLAR's arithmetic divides by the dtype's maximum), got {dtype}"
            )
        hw = (cycle.tiles[0].h, cycle.tiles[0].w)
        ffp = _profile(cycle, cycle.ffp, hw, "flat-field")
        dfp = _profile(cycle, cycle.dfp, hw, "dark-field")
        if ffp is not None and not (np.isfinite(ffp).all() and (ffp > 0).all()):
            raise ValueError(f"{cycle.ffp}: the flat-field profile has values <= 0 or not finite")
        if dfp is not None:
            if not np.isfinite(dfp).all():
                raise ValueError(f"{cycle.dfp}: the dark-field profile is not finite")
            dfp /= np.iinfo(dtype).max  # reg.py:1134, in the profile's own float dtype
        paths = {
            "ffp": None if cycle.ffp is None else str(_resolve(cycle, cycle.ffp)),
            "dfp": None if cycle.dfp is None else str(_resolve(cycle, cycle.dfp)),
        }
        return cls(ffp, dfp, dtype, paths)

    def correct(self, raw, rows=slice(None), cols=slice(None)):
        """``raw`` is the tile's window ``[rows, cols]``; returns it corrected, same dtype."""
        img = np.multiply(raw, 1.0 / self.imax, dtype=np.float64)  # img_as_float
        if self.dfp is not None:
            img -= self.dfp[rows, cols]
        if self.ffp is not None:
            img /= self.ffp[rows, cols]
        np.clip(img, 0, 1, out=img)
        np.multiply(img, self.imax, out=img)  # skimage dtype convert, float -> unsigned
        np.rint(img, out=img)
        np.clip(img, 0, self.imax, out=img)
        return img.astype(self.dtype)

    def describe(self):
        out = {**self.paths, "formula": ASHLAR_FORMULA}
        if self.ffp is not None:
            out["ffp_mean"] = float(self.ffp.mean())
        return out


def _resolve(cycle, path):
    p = Path(path)
    return p if p.is_absolute() else Path(cycle.dir) / p


def _profile(cycle, path, hw, what):
    if path is None:
        return None
    p = _resolve(cycle, path)
    try:
        arr = tifffile.imread(str(p))
    except (OSError, ValueError, tifffile.TiffFileError) as e:
        raise ValueError(f"{p}: cannot read the {what} profile ({e})") from None
    nuc = int(cycle.nuclear_channel)
    if arr.ndim == 3 and tuple(arr.shape[1:]) == hw and nuc < arr.shape[0]:
        arr = arr[nuc]
    elif arr.ndim == 3 and tuple(arr.shape[:2]) == hw and nuc < arr.shape[2]:
        arr = arr[..., nuc]
    if arr.ndim != 2 or tuple(arr.shape) != hw:
        raise ValueError(
            f"{p}: the {what} profile's shape {arr.shape} does not match the tiles' {hw} "
            f"(nor a (C, H, W) stack holding the nuclear channel {nuc})"
        )
    arr = np.array(arr)  # own copy: one plane, not the stack
    if arr.dtype.kind != "f":
        arr = arr.astype(np.float64)
    return arr


def _read_tile(row, illum, rows=slice(None), cols=slice(None)):
    win = _read_window(row, rows, cols)
    return win if illum is None else illum.correct(win, rows, cols)


# ── the shared writer ────────────────────────────────────────────────────────
_FROM_CYCLE = object()


def placed_origins(rows, positions) -> np.ndarray:
    """``(n, 2)`` int64 (x, y): where :func:`write_mosaic` pastes each of ``rows`` -- the
    positions rounded to whole pixels, then moved to a zero origin. The ONE definition of
    "on its pixels": nuclei carried into the mosaic use it too (``stitching.py``)."""
    xy = np.array([positions[r.tile] for r in rows], dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError("non-finite tile position")
    o = np.rint(xy).astype(np.int64)
    return o - o.min(axis=0)


def write_mosaic(cycle: tiles.Cycle, positions, out, *, out_tile: int = OUT_TILE,
                 illumination=_FROM_CYCLE) -> Path:
    """The cycle's mosaic at ``positions`` (``tile -> (x, y)`` px) as a tiled, one-channel
    BigTIFF OME at ``out``, streamed: ``out_tile`` blocks, each from the intersecting tiles'
    windows only, pasted last-wins in ``tiles.csv`` order at whole-pixel positions (rounded,
    then moved to a zero origin). ``illumination`` defaults to the cycle's own profiles."""
    out_tile = int(out_tile)
    if out_tile <= 0 or out_tile % 16:
        raise ValueError(f"out_tile must be a positive multiple of 16 (a TIFF tile), got {out_tile}")
    rows = cycle.tiles
    ids = {r.tile for r in rows}
    if set(positions) != ids:
        lack = sorted(ids - set(positions))
        extra = sorted(set(positions) - ids)
        raise ValueError(f"positions lack tile(s) {lack} / name tile(s) the cycle lacks {extra}")
    illum = Illumination.of(cycle) if illumination is _FROM_CYCLE else illumination
    o = placed_origins(rows, positions)
    hw = np.array([[r.w, r.h] for r in rows], np.int64)
    boxes = np.concatenate([o, o + hw], axis=1)  # x0, y0, x1, y1
    out_h, out_w = int(boxes[:, 3].max()), int(boxes[:, 2].max())
    dtype = _tile_dtype(cycle)

    def blocks():
        for by in range(0, out_h, out_tile):
            for bx in range(0, out_w, out_tile):
                block = np.zeros((out_tile, out_tile), dtype)
                hit = np.flatnonzero(
                    (boxes[:, 0] < bx + out_tile) & (boxes[:, 2] > bx)
                    & (boxes[:, 1] < by + out_tile) & (boxes[:, 3] > by)
                )
                for i in hit:  # ascending: a later tile pastes over an earlier one
                    x0, y0, x1, y1 = (int(v) for v in boxes[i])
                    ix0, iy0 = max(bx, x0), max(by, y0)
                    ix1, iy1 = min(bx + out_tile, x1), min(by + out_tile, y1)
                    rs, cs = slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0)
                    block[iy0 - by:iy1 - by, ix0 - bx:ix1 - bx] = _read_tile(rows[i], illum, rs, cs)
                yield block

    return imageio.write_nuclear_tiles(
        out, (out_h, out_w), dtype, blocks(), out_tile, cycle.pixel_size_um
    )


# ── ashlar_edge ──────────────────────────────────────────────────────────────
def reader_of(cycle: tiles.Cycle, illumination=None):
    """A minimal ``ashlar.reg.Reader`` over the tile series: ``metadata`` (a
    ``reg.Metadata``: the zero-origin stage positions as (row, col) px, the pixel size, the
    tile size and dtype) and an UNCACHED ``read(series, c)`` (one tile, corrected when
    ``illumination`` is given). The files hold the nuclear plane only, so ``c`` must be
    ``cycle.nuclear_channel``. Needs ``ashlar`` importable (the ASHLAR image)."""
    reg = importlib.import_module("ashlar.reg")
    rows = cycle.tiles
    nuclear = int(cycle.nuclear_channel)
    n_channels = max(len(cycle.channels), nuclear + 1)

    class TileSeriesReader(reg.Reader):
        def __init__(self):
            self.metadata = _series_metadata(reg, cycle, n_channels)

        def read(self, series, c):
            if int(c) != nuclear:
                raise ValueError(
                    f"{cycle.dir}: the tile files hold only the nuclear channel {nuclear}; "
                    f"channel {c} was asked for"
                )
            return _read_tile(rows[int(series)], illumination)

    return TileSeriesReader()


def _series_metadata(reg, cycle, n_channels):
    """A ``reg.Metadata`` of the tile series: the zero-origin stage positions as (row,
    col) px, the pixel size, the tile size and dtype, ``n_channels`` channels."""
    rows = cycle.tiles
    stage = stage_positions(cycle)
    pos_rc = np.array([[stage[r.tile][1], stage[r.tile][0]] for r in rows], dtype=float)
    dtype = _tile_dtype(cycle)
    pixel_size = float(cycle.pixel_size_um)

    class TileSeriesMetadata(reg.Metadata):
        @property
        def _num_images(self):
            return len(rows)

        @property
        def num_channels(self):
            return int(n_channels)

        @property
        def pixel_size(self):
            return pixel_size

        @property
        def pixel_dtype(self):
            return dtype

        def tile_position(self, i):
            return pos_rc[i]

        def tile_size(self, i):
            return np.array([rows[i].h, rows[i].w])

    return TileSeriesMetadata()


SERIES_READER = "regbench_tiles"


def series_reader_class():
    """The reader ASHLAR's own COMMAND reads a tile series with (the ``@raw`` cases of
    ``methods/ashlar.py``): a ``reg.Reader`` subclass built from a cycle folder,
    ``Reader(path)``, registered in ``ashlar.scripts.ashlar.readers`` under
    ``SERIES_READER`` so that the argument ``regbench_tiles|<cycle folder>`` selects it
    (ASHLAR's documented ``reader|path`` syntax, ``scripts/ashlar.py:342-376``).

    It serves EVERY channel the tile files hold -- one plane, the nuclear channel, served
    as channel 0 -- uncorrected (ASHLAR as published applies flat-/dark-field profiles in
    its ``Mosaic`` only, never to alignment), and it is NOT uncached: ASHLAR's own
    ``EdgeAligner`` wraps it in its ``CachingReader`` as it wraps any reader, and the
    method's memory is a measured result. ``reader_of`` (the front-end's reader) is
    unchanged. Needs ``ashlar`` importable (the ASHLAR image)."""
    reg = importlib.import_module("ashlar.reg")

    class TileSeriesFileReader(reg.Reader):
        def __init__(self, path, **kwargs):
            if kwargs:
                raise ValueError(
                    f"{SERIES_READER} takes a cycle folder and no options, got {kwargs}"
                )
            self.path = str(path)
            self.cycle = tiles.read_cycle(path)
            self.metadata = _series_metadata(reg, self.cycle, 1)

        def read(self, series, c):
            if int(c) != 0:
                raise ValueError(
                    f"{self.path}: the tile files hold one plane (the nuclear channel, "
                    f"served as channel 0); channel {c} was asked for"
                )
            return _read_tile(self.cycle.tiles[int(series)], None)

    return TileSeriesFileReader


def _ashlar_edge(cycle, max_shift_um, illum):
    from .methods.ashlar import ZERO_OVERLAP_WARNINGS, fallback_stage_positions

    reg = importlib.import_module("ashlar.reg")
    reader = reader_of(cycle, illum)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aligner = reg.EdgeAligner(reader, channel=cycle.nuclear_channel, max_shift=max_shift_um)
        wrapped = type(aligner.reader).__name__
        if not isinstance(aligner.reader, reg.CachingReader):
            raise RuntimeError(
                f"ashlar's EdgeAligner holds a {wrapped}, not reg.CachingReader: the "
                "uncached-reader swap (the memory contract) would be a silent no-op, so "
                "ashlar_edge refuses until it is re-checked against this ASHLAR"
            )
        aligner.reader = reader  # no CachingReader: see the module docstring
        aligner.run()
    warned = [str(w.message) for w in caught if issubclass(w.category, reg.Warning)]
    pos = np.asarray(aligner.positions, dtype=float)  # (row, col) px
    positions = _zero_origin(
        {r.tile: (float(pos[i, 1]), float(pos[i, 0])) for i, r in enumerate(cycle.tiles)}
    )
    thumb = getattr(reader, "thumbnail", None)
    side = [
        f"reg.EdgeAligner wraps its reader in {wrapped} (reg.py:502), which keeps every tile "
        "of the align channel; the aligner's reader is set back to the uncached tile-series "
        "reader before run() (the memory contract)",
        "the mosaic is written by regbench's shared streaming writer ("
        + PASTE + "), not by ASHLAR's Mosaic (sub-pixel paste, linear blend)",
    ]
    if illum is not None:
        side.append(
            "tiles are illumination-corrected before alignment too (ASHLAR corrects only "
            "in Mosaic.assemble_channel)"
        )
    report = {
        "ashlar": getattr(importlib.import_module("ashlar"), "__version__", None),
        "max_shift_um": float(max_shift_um),
        "ashlar_warnings": warned,
        "fallback_stage_positions": fallback_stage_positions(warned),
        "zero_overlap": any(z in m for m in warned for z in ZERO_OVERLAP_WARNINGS),
        "thumbnail": None if thumb is None else {
            "shape": [int(v) for v in np.shape(thumb)],
            "bytes": int(np.asarray(thumb).nbytes),
            "scale": getattr(reader, "thumbnail_scale", None),
        },
        "benchmark_side": side,
    }
    del aligner, reader, thumb
    return positions, report


# ── stare_mosaic ─────────────────────────────────────────────────────────────
def _version_tuple(v):
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", str(v))
    return tuple(int(g or 0) for g in m.groups()) if m else (0,)


def _require_stare():
    """STARE's version, refusing one without ``stare mosaic`` (< 1.3.0)."""
    try:
        stare = importlib.import_module("stare")
    except ImportError:
        raise RuntimeError(
            "stare_mosaic needs STARE >= 1.3.0 (`stare mosaic`), and `stare` is not "
            "importable here: it runs in REGBENCH_SIF_STARE_REGBENCH"
        ) from None
    v = str(getattr(stare, "__version__", "0"))
    if _version_tuple(v) < STARE_MIN_VERSION:
        raise RuntimeError(f"stare {v} has no `stare mosaic`: stare_mosaic needs STARE >= 1.3.0")
    return v


def _run_stare_mosaic(argv):
    """``stare mosaic <argv>`` in this interpreter's environment."""
    cmd = [sys.executable, "-m", "stare.cli", "mosaic", *argv]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        tail = "\n".join((res.stderr or res.stdout or "").strip().splitlines()[-20:])
        raise RuntimeError(f"stare mosaic exited {res.returncode}: {tail}")


def _corrected_copy(cycle, illum, dest):
    """Every tile corrected, one at a time, as a new tile series (no profiles: already
    applied) for ``stare mosaic``."""

    def planes():
        for r in cycle.tiles:
            yield r.tile, _read_tile(r, illum), r.x_px, r.y_px

    return tiles.write_cycle(
        dest, planes(), pixel_size_um=cycle.pixel_size_um, channels=cycle.channels,
        nuclear_channel=cycle.nuclear_channel,
        extra={"corrected_from": str(cycle.dir), "illumination": illum.describe()},
    )


def _stare_mosaic(cycle, out, max_shift_um, illum):
    version = _require_stare()
    src = cycle if illum is None else _corrected_copy(cycle, illum, out / CORRECTED)
    json_path = out / STARE_POSITIONS
    _run_stare_mosaic([
        "--tiles", str(Path(src.dir) / tiles.TILES_CSV),
        "--out", str(json_path),
        "--max-shift-um", repr(float(max_shift_um)),
        "--pixel-size", repr(float(cycle.pixel_size_um)),
    ])
    doc = json.loads(json_path.read_text())
    got = {int(t["tile"]): (float(t["x"]), float(t["y"])) for t in doc["tiles"]}
    ids = {r.tile for r in cycle.tiles}
    if set(got) != ids:
        raise RuntimeError(
            f"{json_path}: stare mosaic placed tiles {sorted(got)}, the cycle has {sorted(ids)}"
        )
    report = dict(doc.get("report") or {})
    # STARE's own report says "mosaic": None (it was not asked to write one); here the
    # mosaic IS written (by the shared writer), so the key would mislead
    if "mosaic" in report:
        report["stare_mosaic"] = report.pop("mosaic")
    report.update(
        stare=version,
        stare_positions=str(json_path),
        benchmark_side=[
            "the mosaic is written by regbench's shared streaming writer (" + PASTE
            + "), not by `stare mosaic --write-mosaic`"
        ],
    )
    return _zero_origin(got), report


# ── the entry point ──────────────────────────────────────────────────────────
def stitch(cycle: tiles.Cycle, name: str, out, *, max_shift_um: float = DEFAULT_MAX_SHIFT_UM,
           out_tile: int = OUT_TILE) -> Stitched:
    """Place ``cycle``'s tiles with front-end ``name`` and write its mosaic in ``out``."""
    if name not in FRONT_ENDS:
        raise ValueError(f"unknown stitching front-end {name!r}; one of {FRONT_ENDS}")
    max_shift_um = float(max_shift_um)
    if not (math.isfinite(max_shift_um) and max_shift_um > 0):
        raise ValueError(f"max_shift_um must be finite and > 0, got {max_shift_um}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    illum = Illumination.of(cycle)
    if name == "stage":
        positions, report = stage_positions(cycle), {}
    elif name == "ashlar_edge":
        positions, report = _ashlar_edge(cycle, max_shift_um, illum)
    else:
        positions, report = _stare_mosaic(cycle, out, max_shift_um, illum)
    mosaic = write_mosaic(cycle, positions, out / MOSAIC, out_tile=out_tile, illumination=illum)
    with tifffile.TiffFile(str(mosaic)) as tf:
        shape = [int(v) for v in tf.series[0].shape[-2:]]
    report = {
        **report,
        "front_end": name,
        "n_tiles": len(cycle.tiles),
        "illumination": None if illum is None else illum.describe(),
        "paste": PASTE,
        "out_tile": int(out_tile),
        "mosaic_shape": shape,
    }
    return Stitched(Path(mosaic), positions, report)
