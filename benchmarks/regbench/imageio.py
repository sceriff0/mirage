"""Writing and extracting the OME-TIFFs the methods are given: the one-channel nuclear slide,
and slides with several named channels (the selfcheck's pair).

The RGB readers (``_rgb_view``, ``read_rgb_reduced``: a brightfield slide read in
rectangles, or reduced) are restored as they were at ``5a46f778``: DeeperHistReg's and
STARE's brightfield input rules read with them, and so will the brightfield dataset
modules."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

BAND_ROWS = 2048
NUCLEAR_NAME = ("NUCLEAR",)


def _ome_meta(names, pixel_size_um):
    meta = {"axes": "CYX", "Channel": {"Name": [str(n) for n in names]}}
    if pixel_size_um:
        meta.update(
            {
                "PhysicalSizeX": float(pixel_size_um),
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": float(pixel_size_um),
                "PhysicalSizeYUnit": "µm",
            }
        )
    return meta


def _atomic_tiff(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".ome.tif", dir=path.parent)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def write_channels(path, planes, names, pixel_size_um=None):
    """``(C, H, W)`` planes as a tiled, compressed OME-TIFF with NAMED channels (atomic)."""
    import tifffile

    planes = np.asarray(planes)
    if planes.ndim != 3 or len(names) != planes.shape[0]:
        raise ValueError(
            f"{len(names)} channel names for planes of shape {planes.shape}"
        )
    return _atomic_tiff(
        path,
        lambda tmp: tifffile.imwrite(
            tmp,
            planes,
            ome=True,
            bigtiff=True,
            tile=(512, 512),
            compression="zlib",
            photometric="minisblack",
            metadata=_ome_meta(names, pixel_size_um),
        ),
    )


def write_channel_tiles(path, hw, dtype, tiles, tile, names, pixel_size_um=None):
    """As :func:`write_channels`, from an iterator of tiles: every tile of channel 0 in
    row-major order, then every tile of channel 1, and so on.

    The slide is never held in memory, which is what lets a 65536 px slide be written from a
    generator. Edge tiles are padded to the full tile, as TIFF stores them.
    """
    import tifffile

    def padded():
        for t in tiles:
            if t.shape != (tile, tile):
                full = np.zeros((tile, tile), dtype)
                full[: t.shape[0], : t.shape[1]] = t
                t = full
            yield t

    return _atomic_tiff(
        path,
        lambda tmp: tifffile.imwrite(
            tmp,
            padded(),
            shape=(len(names), *hw),
            dtype=dtype,
            ome=True,
            bigtiff=True,
            tile=(tile, tile),
            compression="zlib",
            photometric="minisblack",
            metadata=_ome_meta(names, pixel_size_um),
            # one encoder thread is a memory bound, not a tuning choice: tifffile's
            # default pool encodes ahead of the writer, and the tiles it queues grow
            # with the slide (measured on tifffile 2023.4.12: 58 / 199 / 269 MiB peak
            # for 48 / 192 / 384 tiles of 1024 px; tests/test_merge_pyramid_streaming.py
            # pins the same argument for the pipeline's pyramid writer)
            maxworkers=1,
        ),
    )


def write_nuclear(path, plane, pixel_size_um=None):
    """One ``(H, W)`` plane as a tiled, compressed, one-channel OME-TIFF (atomic)."""
    return write_channels(path, np.asarray(plane)[None], NUCLEAR_NAME, pixel_size_um)


def write_nuclear_tiles(path, hw, dtype, tiles, tile, pixel_size_um=None):
    """As :func:`write_nuclear`, from an iterator of tiles in row-major order."""
    return write_channel_tiles(
        path, hw, dtype, tiles, tile, NUCLEAR_NAME, pixel_size_um
    )


def channel_names(path):
    """The channel names in an OME-TIFF's header ([] when it names none)."""
    import re

    import tifffile

    with tifffile.TiffFile(path) as tif:
        xml = tif.ome_metadata or ""
    return re.findall(r'<Channel\b[^>]*?\bName="([^"]*)"', xml)


def shape_of(path):
    """``(C, H, W)`` of an OME-TIFF without reading its pixels."""
    from stare.slide_io import open_lazy

    arr, _, close = open_lazy(path)
    try:
        return tuple(int(v) for v in arr.shape)
    finally:
        close()


def extract_channel(src, channel, dst, pixel_size_um=None):
    """Copy one channel of a (possibly multichannel, pyramidal) OME-TIFF to ``dst``, in row bands.

    A source that already is one channel is used in place: nothing is copied and ``src`` is
    returned.
    """
    from stare.slide_io import open_lazy

    arr, dtype, close = open_lazy(src)
    try:
        c, h, w = arr.shape
        if not 0 <= channel < c:
            raise ValueError(f"{src}: channel {channel} out of range for C={c}")
        if c == 1:
            return Path(src)
        if Path(dst).exists():
            return Path(dst)
        plane = np.empty((h, w), dtype=dtype)
        for y0 in range(0, h, BAND_ROWS):
            plane[y0 : y0 + BAND_ROWS] = arr[
                channel, slice(y0, min(y0 + BAND_ROWS, h)), slice(0, w)
            ]
    finally:
        close()
    return write_nuclear(dst, plane, pixel_size_um)


def pixel_size_um(path):
    """The OME header's pixel size in µm, or None when the file does not say."""
    from stare.ome import read_ome_pixel_size

    try:
        px = read_ome_pixel_size(str(path))
    except Exception:
        return None
    if px is None:
        return None
    if isinstance(px, (tuple, list)):
        px = px[0]
    return float(px) if px else None


def _rgb_view(level, lazy_only=False):
    """``(read, (H, W))`` of one TIFF pyramid level: ``read(y0, y1, x0=0, x1=W)`` gives
    that rectangle as ``(h, w, 3)``. Read lazily (only the TIFF's tiles or strips the
    rectangle touches) if zarr is installed, and whole otherwise; ``lazy_only`` refuses
    the whole read (``ImportError``) for a caller that must stream."""
    axes = level.axes.replace("S", "C")
    if set(axes) != set("YXC") or level.shape[axes.index("C")] < 3:
        raise ValueError(
            f"expected an RGB image, got axes {level.axes} shape {level.shape}"
        )
    order = [axes.index(c) for c in "YXC"]
    h, w = level.shape[order[0]], level.shape[order[1]]
    try:
        import zarr

        arr = zarr.open(level.aszarr(), mode="r")
    except ImportError:
        if lazy_only:
            raise
        arr = level.asarray()

    def read(y0, y1, x0=0, x1=None):
        idx = [slice(None)] * 3
        idx[order[0]] = slice(y0, y1)
        idx[order[1]] = slice(x0, w if x1 is None else x1)
        return np.transpose(np.asarray(arr[tuple(idx)]), order)[..., :3]

    return read, (h, w)


def read_rgb_reduced(path, max_dim, rows=64):
    """An RGB whole-slide TIFF as uint8 ``(h, w, 3)`` with long side <= ``max_dim``, and the
    integer factor it was reduced by: the coarsest pyramid level still at least that large,
    then block-averaged in bands of rows."""
    import tifffile

    with tifffile.TiffFile(path) as tif:
        levels = tif.series[0].levels
        h0, w0 = _rgb_view(levels[0])[1]
        pick, f_level = levels[0], 1
        for lv in levels[1:]:
            f = round(h0 / _rgb_view(lv)[1][0])
            if (
                max(_rgb_view(lv)[1]) >= max_dim
                and abs(h0 / f - _rgb_view(lv)[1][0]) < 1.0
            ):
                pick, f_level = lv, f
        read, (h, w) = _rgb_view(pick)
        f = max(1, -(-max(h, w) // max_dim))
        hh, ww = (h // f) * f, (w // f) * f
        out = np.empty((hh // f, ww // f, 3), np.uint8)
        for y0 in range(0, hh, rows * f):
            y1 = min(y0 + rows * f, hh)
            band = read(y0, y1)[:, :ww].astype(np.float32)
            out[y0 // f : y1 // f] = (
                band.reshape((y1 - y0) // f, f, ww // f, f, 3)
                .mean(axis=(1, 3))
                .round()
                .clip(0, 255)
            )
    return out, f * f_level


def tiff_mm_per_px(path):
    """Millimetres per full-resolution pixel from a TIFF's resolution tags, or None."""
    import tifffile

    with tifffile.TiffFile(path) as tif:
        tags = tif.series[0].levels[0].pages[0].tags
        res, unit = tags.get("XResolution"), tags.get("ResolutionUnit")
        if res is None or unit is None:
            return None
        num, den = res.value
        per_unit = {2: 25.4, 3: 10.0}.get(int(unit.value))  # inch, centimetre
        if not per_unit or not num:
            return None
        return per_unit * den / num


def describe_written(path):
    """What a writer left on disk, read from the file's own headers: ``channels``,
    ``levels`` (1 without a pyramid), ``pyramid``, ``compression``, ``tile``, ``bigtiff``,
    ``dtype``, ``shape`` and ``bytes``. ``{"unread": reason}`` when it cannot be read.

    A pyramid is counted both ways writers store one: as reduced levels of the first
    series (SubIFDs, what OME-TIFF writers do) and as further series smaller than the
    first (top-level pages, what libvips does without SubIFDs).
    """
    import tifffile

    try:
        path = Path(path)
        with tifffile.TiffFile(str(path)) as tif:
            series, page = tif.series[0], tif.pages[0]
            axes = str(getattr(series, "axes", ""))
            shape = [int(v) for v in series.shape]
            channels = shape[axes.index("C")] if "C" in axes else 1
            if "S" in axes and "C" not in axes:
                channels = shape[axes.index("S")]
            levels = len(getattr(series, "levels", [series]))
            if levels == 1:
                plane = shape[-2] * shape[-1] if "S" not in axes else 0
                levels += sum(
                    1
                    for other in tif.series[1:]
                    if plane and int(np.prod(other.shape[-2:])) < plane
                )
            return {
                "channels": int(channels),
                "levels": int(levels),
                "pyramid": bool(levels > 1),
                "compression": str(getattr(page.compression, "name", page.compression)),
                "tile": [int(page.tilelength), int(page.tilewidth)]
                if page.is_tiled
                else None,
                "bigtiff": bool(tif.is_bigtiff),
                "dtype": str(np.dtype(series.dtype)),
                "shape": shape,
                "bytes": int(path.stat().st_size),
            }
    except Exception as exc:
        return {"unread": f"{type(exc).__name__}: {exc}"}
