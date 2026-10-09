"""Writing and extracting the one-channel nuclear OME-TIFF every method is given."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

BAND_ROWS = 2048


def write_nuclear(path, plane, pixel_size_um=None):
    """One ``(H, W)`` plane as a tiled, compressed, one-channel OME-TIFF (atomic)."""
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"axes": "CYX", "Channel": {"Name": ["NUCLEAR"]}}
    if pixel_size_um:
        meta.update({"PhysicalSizeX": float(pixel_size_um), "PhysicalSizeXUnit": "µm",
                     "PhysicalSizeY": float(pixel_size_um), "PhysicalSizeYUnit": "µm"})
    fd, tmp = tempfile.mkstemp(suffix=".ome.tif", dir=path.parent)
    os.close(fd)
    try:
        tifffile.imwrite(tmp, np.asarray(plane)[None], ome=True, bigtiff=True, tile=(512, 512),
                         compression="zlib", metadata=meta)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def write_nuclear_tiles(path, hw, dtype, tiles, tile, pixel_size_um=None):
    """As :func:`write_nuclear`, from an iterator of tiles in row-major order.

    The slide is never held in memory, which is what lets a 65536 px slide be written from a
    generator. Edge tiles are padded to the full tile, as TIFF stores them.
    """
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"axes": "CYX", "Channel": {"Name": ["NUCLEAR"]}}
    if pixel_size_um:
        meta.update({"PhysicalSizeX": float(pixel_size_um), "PhysicalSizeXUnit": "µm",
                     "PhysicalSizeY": float(pixel_size_um), "PhysicalSizeYUnit": "µm"})

    def padded():
        for t in tiles:
            if t.shape != (tile, tile):
                full = np.zeros((tile, tile), dtype)
                full[: t.shape[0], : t.shape[1]] = t
                t = full
            yield t

    fd, tmp = tempfile.mkstemp(suffix=".ome.tif", dir=path.parent)
    os.close(fd)
    try:
        tifffile.imwrite(tmp, padded(), shape=(1, *hw), dtype=dtype, ome=True, bigtiff=True,
                         tile=(tile, tile), compression="zlib", metadata=meta)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


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
            plane[y0 : y0 + BAND_ROWS] = arr[channel, slice(y0, min(y0 + BAND_ROWS, h)), slice(0, w)]
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


def _rgb_view(level):
    """``(array-like (H, W, 3), (H, W))`` of one TIFF pyramid level, read lazily if zarr is
    installed and whole otherwise."""
    axes = level.axes.replace("S", "C")
    if set(axes) != set("YXC") or level.shape[axes.index("C")] < 3:
        raise ValueError(f"expected an RGB image, got axes {level.axes} shape {level.shape}")
    order = [axes.index(c) for c in "YXC"]
    h, w = level.shape[order[0]], level.shape[order[1]]
    try:
        import zarr

        arr = zarr.open(level.aszarr(), mode="r")
    except ImportError:
        arr = level.asarray()

    def read(y0, y1):
        idx = [slice(None)] * 3
        idx[order[0]] = slice(y0, y1)
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
            if max(_rgb_view(lv)[1]) >= max_dim and abs(h0 / f - _rgb_view(lv)[1][0]) < 1.0:
                pick, f_level = lv, f
        read, (h, w) = _rgb_view(pick)
        f = max(1, -(-max(h, w) // max_dim))
        hh, ww = (h // f) * f, (w // f) * f
        out = np.empty((hh // f, ww // f, 3), np.uint8)
        for y0 in range(0, hh, rows * f):
            y1 = min(y0 + rows * f, hh)
            band = read(y0, y1)[:, :ww].astype(np.float32)
            out[y0 // f : y1 // f] = band.reshape((y1 - y0) // f, f, ww // f, f, 3).mean(
                axis=(1, 3)).round().clip(0, 255)
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
