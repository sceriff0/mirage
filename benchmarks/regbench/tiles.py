"""The tile-series format every raw-tile reader writes and every stitching front-end reads.

One folder per cycle::

    <cycle>/tiles.csv    tile,path,channel,x_px,y_px   (path relative to the CSV)
    <cycle>/cycle.json   pixel_size_um, channels, nuclear_channel, ffp, dfp, + reader extras
    <cycle>/t<id>.tif    one single-plane nuclear TIFF per tile

``tiles.csv`` is exactly the input of ``stare mosaic`` (``--tiles``). ``x_px``/``y_px`` are
the tile's top-left corner in pixels of that tile (x = column, y = row). ``channel`` is the
plane inside the TIFF, always 0: the files hold the nuclear plane alone.

Memory (user ruling 2026-10-14): ``write_cycle`` consumes its planes ONE at a time and writes
each before asking for the next; ``read_cycle`` reads only TIFF headers, never pixels.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tifffile

TILES_CSV = "tiles.csv"
CYCLE_JSON = "cycle.json"
COLUMNS = ("tile", "path", "channel", "x_px", "y_px")


class TileRow(NamedTuple):
    tile: int
    path: Path
    channel: int
    x_px: float
    y_px: float
    h: int
    w: int


class Cycle(NamedTuple):
    dir: Path
    pixel_size_um: float
    channels: tuple[str, ...]
    nuclear_channel: int
    tiles: tuple[TileRow, ...]
    ffp: Path | None
    dfp: Path | None


def _atomic_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _finite_positive(value, what: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: not a number: {value!r}") from None
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{what}: must be finite and > 0, got {value!r}")
    return v


def write_cycle(
    dest: Path,
    planes: Iterable[tuple[int, np.ndarray, float, float]],
    *,
    pixel_size_um: float,
    channels,
    nuclear_channel: int,
    ffp: Path | None = None,
    dfp: Path | None = None,
    extra: dict | None = None,
) -> Cycle:
    """Write ``planes`` -- ``(tile id, 2-D array, x_px, y_px)`` -- as one cycle in ``dest``.

    ``planes`` may be a generator: each plane is written to disk before the next is asked
    for. ``extra`` is merged into ``cycle.json`` (a reader's provenance, e.g. the chosen z
    per tile); ``read_cycle`` ignores it. ``ffp``/``dfp`` are stored as given.
    """
    dest = Path(dest)
    pixel_size_um = _finite_positive(pixel_size_um, "pixel_size_um")
    channels = tuple(str(c) for c in channels)
    if not 0 <= int(nuclear_channel) < max(len(channels), 1):
        raise ValueError(
            f"nuclear_channel {nuclear_channel} is not an index into {len(channels)} channels"
        )
    dest.mkdir(parents=True, exist_ok=True)
    seen: set[int] = set()
    rows: list[tuple[int, str, float, float]] = []
    for tile, plane, x_px, y_px in planes:
        tile = int(tile)
        if tile in seen:
            raise ValueError(f"duplicate tile id {tile}")
        seen.add(tile)
        x_px, y_px = float(x_px), float(y_px)
        if not (math.isfinite(x_px) and math.isfinite(y_px)):
            raise ValueError(f"tile {tile}: non-finite position ({x_px}, {y_px})")
        arr = np.asarray(plane)
        if arr.ndim != 2:
            raise ValueError(f"tile {tile}: expected a 2-D plane, got shape {arr.shape}")
        name = f"t{tile:05d}.tif"
        tifffile.imwrite(dest / name, arr, photometric="minisblack")
        rows.append((tile, name, x_px, y_px))
        del arr, plane
    if not rows:
        raise ValueError("no tiles to write")
    buf = [",".join(COLUMNS)]
    buf += [f"{t},{p},0,{x!r},{y!r}" for t, p, x, y in rows]
    _atomic_text(dest / TILES_CSV, "\n".join(buf) + "\n")
    doc = dict(extra or {})
    doc.update(
        pixel_size_um=pixel_size_um,
        channels=list(channels),
        nuclear_channel=int(nuclear_channel),
        ffp=None if ffp is None else str(ffp),
        dfp=None if dfp is None else str(dfp),
    )
    _atomic_text(dest / CYCLE_JSON, json.dumps(doc, indent=2) + "\n")
    return read_cycle(dest)


def read_cycle(dir: Path) -> Cycle:  # noqa: A002 - the interface's name
    """Read a cycle folder (headers only). Refuses a missing or non-positive pixel size, a
    duplicate tile id, a tile whose shape differs from the first, a non-finite position."""
    d = Path(dir)
    try:
        doc = json.loads((d / CYCLE_JSON).read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"{d}: cannot read {CYCLE_JSON} ({e})") from None
    if doc.get("pixel_size_um") is None:
        raise ValueError(f"{d / CYCLE_JSON}: no pixel_size_um")
    pixel = _finite_positive(doc["pixel_size_um"], f"{d / CYCLE_JSON} pixel_size_um")
    try:
        with open(d / TILES_CSV, newline="") as fh:
            raw = list(csv.DictReader(fh))
    except OSError as e:
        raise ValueError(f"{d}: cannot read {TILES_CSV} ({e})") from None
    if not raw:
        raise ValueError(f"{d / TILES_CSV}: no tiles")
    missing = set(COLUMNS) - set(raw[0])
    if missing:
        raise ValueError(f"{d / TILES_CSV}: missing column(s) {sorted(missing)}")
    seen: set[int] = set()
    out: list[TileRow] = []
    shape = None
    for i, r in enumerate(raw):
        where = f"{d / TILES_CSV} row {i + 2}"
        try:
            tile = int(r["tile"])
        except (TypeError, ValueError):
            raise ValueError(f"{where}: tile id is not an integer: {r['tile']!r}") from None
        if tile in seen:
            raise ValueError(f"{where}: duplicate tile id {tile}")
        seen.add(tile)
        try:
            x, y = float(r["x_px"]), float(r["y_px"])
        except (TypeError, ValueError):
            raise ValueError(f"{where}: position is not a number") from None
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"{where}: non-finite position ({r['x_px']}, {r['y_px']})")
        path = d / r["path"]
        try:
            with tifffile.TiffFile(path) as tf:
                hw = tuple(int(v) for v in tf.pages[0].shape[-2:])
        except (OSError, ValueError, tifffile.TiffFileError) as e:
            raise ValueError(f"{where}: cannot read {path} ({e})") from None
        if shape is None:
            shape = hw
        elif hw != shape:
            raise ValueError(
                f"{where}: tile {tile} is {hw[0]}x{hw[1]}, the first tile is "
                f"{shape[0]}x{shape[1]}"
            )
        out.append(TileRow(tile, path, int(r["channel"]), x, y, hw[0], hw[1]))
    return Cycle(
        dir=d,
        pixel_size_um=pixel,
        channels=tuple(doc.get("channels") or ()),
        nuclear_channel=int(doc.get("nuclear_channel", 0)),
        tiles=tuple(out),
        ffp=Path(doc["ffp"]) if doc.get("ffp") else None,
        dfp=Path(doc["dfp"]) if doc.get("dfp") else None,
    )
