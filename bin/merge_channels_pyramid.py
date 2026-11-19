#!/usr/bin/env python3
"""Merge single-channel TIFF files into a pyramidal multi-channel OME-TIFF."""

from __future__ import annotations

import argparse
import colorsys
import gc
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile

# Add path for utils
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from logger import configure_logging, get_logger
from ome_io import ome_metadata, ome_tiff_writer
from pixel_size import resolve_pixel_size
from tiled_io import open_lazy
from validation import StreamingWrappedValues

# Unguarded imports: bin/utils/validation.py needs only numpy and the stdlib, so a
# fallback would only hide a skipped wrapped-value check. tiled_io.open_lazy needs
# zarr, which the lazy pyramid build requires.

os.environ["NUMBA_DISABLE_JIT"] = "0"
os.environ["NUMBA_CACHE_DIR"] = tempfile.gettempdir() + "/numba_cache"
os.environ["NUMBA_DISABLE_CACHING"] = "1"

logger = get_logger(__name__)

__all__ = ["main"]


def log(msg):
    """Emit one INFO line through this module's logger.

    Takes a preformatted string so the dense progress-reporting call sites stay one
    line.
    """
    logger.info(msg)


MARKER_COLORS = {
    "DAPI": (0, 0, 255),
    "CD45": (0, 255, 0),
    "CD3": (255, 255, 0),
    "CD8": (255, 0, 255),
    "CD4": (0, 255, 255),
    "CD14": (255, 128, 0),
    "CD163": (255, 0, 128),
    "FOXP3": (128, 255, 0),
    "PANCK": (255, 0, 0),
    "VIMENTIN": (128, 0, 255),
    "SMA": (0, 128, 255),
    "GZMB": (255, 128, 128),
    "PD1": (128, 255, 128),
    "PDL1": (255, 200, 100),
    "L1CAM": (100, 200, 255),
    "PAX2": (200, 100, 255),
    "CD74": (255, 255, 128),
    "Segmentation": (255, 255, 255),
}


def generate_channel_color(name: str, index: int) -> Tuple[int, int, int]:
    """Generate a color for a channel based on name or index."""
    # Predefined colors first (exact match, case-insensitive), so "CD4" does not
    # match "CD45".
    name_upper = name.upper()
    for key, rgb in MARKER_COLORS.items():
        if key.upper() == name_upper:
            return rgb

    # Generate color using golden ratio
    h = (index * 0.618033988749895) % 1.0
    s = 0.7 + (index % 3) * 0.1
    v = 0.85 + (index % 2) * 0.1
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


#: Integer dtypes `downsample_image` routes through a float32 intermediate before
#: rounding back; shared by `_downsample_plane_f32` and `downsample_image`.
_ROUNDED_DTYPES = (
    np.uint8,
    np.uint16,
    np.uint32,
    np.int8,
    np.int16,
    np.int32,
)


def _downsample_plane_f32(plane_f32: np.ndarray, factor: int, out_dtype) -> np.ndarray:
    """Downsample ONE already-float32 plane by `factor`, returning it in `out_dtype`.

    Must be a whole plane, never a band or tile: `cv2.INTER_AREA`'s vertical ratio is
    `h / (h // factor)`, so when `factor` does not divide `h` the resample support
    crosses any band boundary and banded results differ from the whole-plane answer.
    That is why levels are derived per channel from whole planes and only the write is
    tiled (see `_level_tiles`).

    Takes float32 so the caller can fill the float32 buffer band by band. This is the
    arithmetic of `downsample_image`'s integer branch, which calls this function. cv2 is
    imported locally; `downsample_image` falls back to block averaging without it.
    """
    import cv2

    h, w = plane_f32.shape
    resized = cv2.resize(
        plane_f32, (w // factor, h // factor), interpolation=cv2.INTER_AREA
    )
    out_dtype = np.dtype(out_dtype)
    if out_dtype.type in _ROUNDED_DTYPES:
        return np.round(resized).astype(out_dtype)
    return resized.astype(out_dtype)


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2D or 3D (CYX) image by a given factor."""
    try:
        import cv2

        if image.ndim == 2:
            h, w = image.shape
            new_h, new_w = h // factor, w // factor
            # cv2.INTER_AREA is good for downsampling but may introduce small rounding errors
            # For integer types, use float intermediate then round
            if image.dtype in _ROUNDED_DTYPES:
                return _downsample_plane_f32(
                    image.astype(np.float32), factor, image.dtype
                )
            else:
                return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif image.ndim == 3:
            # CYX format - downsample each channel
            c, h, w = image.shape
            new_h, new_w = h // factor, w // factor
            result = np.zeros((c, new_h, new_w), dtype=image.dtype)
            for i in range(c):
                if image.dtype in _ROUNDED_DTYPES:
                    result[i] = _downsample_plane_f32(
                        image[i].astype(np.float32), factor, image.dtype
                    )
                else:
                    result[i] = cv2.resize(
                        image[i], (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
            return result
    except ImportError:
        pass

    # Fallback: block averaging
    if image.ndim == 2:
        h, w = image.shape
        new_h, new_w = h // factor, w // factor
        trimmed = image[: new_h * factor, : new_w * factor]
        reshaped = trimmed.reshape(new_h, factor, new_w, factor)
        if image.dtype in [np.float32, np.float64]:
            return reshaped.mean(axis=(1, 3)).astype(image.dtype)
        else:
            # Round before converting to integer dtype to avoid truncation artifacts
            return np.round(reshaped.mean(axis=(1, 3))).astype(image.dtype)
    elif image.ndim == 3:
        c, h, w = image.shape
        new_h, new_w = h // factor, w // factor
        result = np.zeros((c, new_h, new_w), dtype=image.dtype)
        for i in range(c):
            trimmed = image[i, : new_h * factor, : new_w * factor]
            reshaped = trimmed.reshape(new_h, factor, new_w, factor)
            # Round before converting to integer dtype to avoid truncation artifacts
            if image.dtype in [np.float32, np.float64]:
                result[i] = reshaped.mean(axis=(1, 3)).astype(image.dtype)
            else:
                result[i] = np.round(reshaped.mean(axis=(1, 3))).astype(image.dtype)
        return result
    else:
        raise ValueError(
            f"downsample_image only supports 2D or 3D (CYX) images, got ndim={image.ndim}"
        )


def calculate_pyramid_levels(
    height: int,
    width: int,
    min_size: int = 64,
    max_levels: int = 10,
    scale_factor: int = 2,
) -> List[Tuple[int, int]]:
    """Calculate pyramid level dimensions."""
    levels = [(height, width)]
    h, w = height, width

    for _ in range(max_levels - 1):
        h = h // scale_factor
        w = w // scale_factor
        if h < min_size or w < min_size:
            break
        levels.append((h, w))

    return levels


def to_uint16(data):
    """The one float->uint16 rule for this pipeline: clip to range, round, then cast.

    `.astype()` truncates toward zero, a one-sided bias of mean -0.5 LSB. Rounding is
    `np.round` (half-to-even), the same convention as `bin/apply_basic_profiles.py`'s
    `_to_storage_dtype`. Integer input is returned untouched. Clipping comes before
    rounding because 65535.6 rounds to 65536, which wraps to 0 in uint16
    (tests/test_dtype_rounding_contract.py).
    """
    if not np.issubdtype(data.dtype, np.floating):
        return data
    return np.round(np.clip(data, 0, 65535)).astype(np.uint16)


#: Byte budget for one streamed source row band. Peak for the base pass is one band
#: plus one write tile; a byte budget keeps that independent of slide width.
#: Owned here rather than imported, because the other scripts that stream bands run
#: in different container images. Same value as `tiled_io.DEFAULT_BAND_BYTES`; one
#: 512px tile row of a 40000px-wide slide is 41 MB.
READ_BAND_BYTES = 64 * 1024 * 1024


def _band_rows(width: int, itemsize: int, align: int = 1) -> int:
    """Rows per streamed band: the most that fit in READ_BAND_BYTES, snapped DOWN to a
    multiple of `align` and never below it.

    `align` is the write-tile size on the base pass, so a band always contains whole tile
    rows and no tile is ever assembled from two reads. It is 1 when the band is only
    filling a float32 buffer, where nothing depends on the boundary.
    """
    align = max(1, int(align))
    per_row = max(1, int(width) * max(1, int(itemsize)))
    rows = max(align, int(READ_BAND_BYTES) // per_row)
    return max(align, (rows // align) * align)


class _ChannelValidator:
    """Per-channel validation folded over row bands, logged once after the last band.

    Not `validation.StreamingNegativeClip`: this script's log lines differ (indented,
    naming the channel file, reporting the min). The wrapped-value half uses
    validation.py's `StreamingWrappedValues`.

    The clip is applied to every band by the source; `np.clip(x, 0, None)` is a no-op on
    non-negative data, while the decision to log waits for the last band.
    """

    def __init__(self, name: str, dtype):
        self.name = name
        dt = np.dtype(dtype)
        self._min_before = None
        self._min_after = None
        # The wrapped-value check runs on uint16 channels only, and only when the channel
        # had no negatives.
        self._wrapped = StreamingWrappedValues(dt) if dt == np.uint16 else None

    def observe(self, raw: np.ndarray, clipped: np.ndarray) -> None:
        """Fold one band's raw (pre-clip) and clipped values into the aggregates."""
        if raw.size == 0:
            return
        min_before = raw.min()
        if self._min_before is None or min_before < self._min_before:
            self._min_before = min_before
        min_after = clipped.min()
        if self._min_after is None or min_after < self._min_after:
            self._min_after = min_after
        if self._wrapped is not None:
            self._wrapped.process(raw)

    def finalize(self) -> None:
        """Emit the channel's aggregate line(s) -- the same ones, with the same numbers."""
        if self._min_before is None:
            return
        if self._min_before < 0:
            log(
                f"    WARNING: Negative values detected in {self.name}: min={self._min_before}"
            )
            log(f"    Clipped to 0. New min={self._min_after}")
        elif self._wrapped is not None:
            # Percentages against the whole channel: counts and sizes are summed across bands
            # and divided once, in finalize().
            has_wrapped, wrap_count, wrap_pct = self._wrapped.finalize()
            if has_wrapped:
                log(
                    f"    WARNING: {wrap_count} potential wrapped negative pixels ({wrap_pct:.4f}%) in {self.name}"
                )


class _BandSource:
    """A (C, H, W) pixel source that hands out row bands already in the storage dtype.

    Subclasses supply `read_band`. The writer does not know whether pixels come from
    files or a resident array, so both entry points share one writer.
    """

    #: (C, H, W)
    shape: Tuple[int, int, int] = (0, 0, 0)
    #: dtype the pyramid is written in -- what `read_band` returns.
    dtype = np.dtype(np.uint16)
    #: itemsize of the SOURCE pixels, which is what a band read materialises. Differs
    #: from `dtype.itemsize` when a float32 source is stored as uint16. (The float path
    #: still under-counts: `read_band` briefly holds ~10 bytes/px.)
    src_itemsize = 2

    def read_band(self, c: int, y0: int, y1: int, validator=None) -> np.ndarray:
        raise NotImplementedError

    def release(self) -> None:
        """Drop any open handle. Safe to call at any time; `read_band` reopens."""

    def plane_float32(self, c: int) -> np.ndarray:
        """Channel `c` as ONE float32 plane, assembled band by band.

        Each band is cast on assignment into the preallocated buffer, so the storage-dtype
        plane is never held beside the float32 one.
        """
        _, h, w = self.shape
        buf = np.empty((h, w), dtype=np.float32)
        band = _band_rows(w, self.src_itemsize)
        for y0 in range(0, h, band):
            y1 = min(y0 + band, h)
            buf[y0:y1] = self.read_band(c, y0, y1)
        self.release()
        return buf

    def level_chain(self, c: int, scale: int, n_levels: int):
        """Every pyramid level of channel `c`, as ``[level 1, level 2, ...]``.

        Iterated halving with a round to the storage dtype between levels, not one
        cumulative resize by ``scale**k``; the two are different computations.

        The base plane is transient (one float32 buffer for the first resize); the chain
        the caller keeps is about a third of one plane.
        """
        planes = [_downsample_plane_f32(self.plane_float32(c), scale, self.dtype)]
        for _ in range(n_levels - 1):
            planes.append(
                _downsample_plane_f32(planes[-1].astype(np.float32), scale, self.dtype)
            )
        return planes


class _ArraySource(_BandSource):
    """A resident (C, H, W) array behind the band interface.

    Float input is cast to uint16 one band at a time with `to_uint16`. Negative values
    are not clipped, unlike `_ChannelFileSource`: signed-integer input round-trips
    unchanged (tests/test_merge_pyramid_streaming.py::
    test_the_array_entry_point_does_not_clip_signed_input).
    """

    def __init__(self, data: np.ndarray):
        self._data = data
        src_dtype = np.dtype(data.dtype)
        self._need_cast = src_dtype in (np.dtype(np.float32), np.dtype(np.float64))
        self.dtype = np.dtype(np.uint16) if self._need_cast else src_dtype
        self.src_itemsize = src_dtype.itemsize
        self.shape = tuple(int(n) for n in data.shape)
        if self._need_cast:
            log("  Casting float data to uint16 for QuPath compatibility (per-band)")

    def read_band(self, c: int, y0: int, y1: int, validator=None) -> np.ndarray:
        band = self._data[c, y0:y1]
        if validator is not None:
            validator.observe(band, band)
        return to_uint16(band) if self._need_cast else band


class _ChannelFileSource(_BandSource):
    """The single-channel TIFFs SPLIT_CHANNELS wrote, read one row band at a time.

    Reads through `open_lazy` (bin/utils/tiled_io.py). One handle stays open across the
    bands of a channel and is closed when another channel is selected, because
    `open_lazy` rebuilds the store on every call.
    """

    def __init__(self, channel_files, height: int, width: int, dtype):
        self.channel_files = [str(p) for p in channel_files]
        self.shape = (len(self.channel_files), int(height), int(width))
        src_dtype = np.dtype(dtype)
        # uint16 when the input is float, so nothing downstream pays for a bulk cast.
        self._need_cast = src_dtype in (np.dtype(np.float32), np.dtype(np.float64))
        self._signed = not np.issubdtype(src_dtype, np.unsignedinteger)
        self.dtype = np.dtype(np.uint16) if self._need_cast else src_dtype
        self.src_itemsize = src_dtype.itemsize
        self._open_index = None
        self._arr = None
        self._close = None

    def _select(self, c: int) -> None:
        if self._open_index == c:
            return
        self.release()
        arr, _dtype, close = open_lazy(self.channel_files[c])
        if arr.shape[0] != 1:
            close()
            raise ValueError(
                f"{self.channel_files[c]}: expected a single-plane channel file, got "
                f"{arr.shape[0]} planes. SPLIT_CHANNELS writes one plane per marker."
            )
        self._arr, self._close, self._open_index = arr, close, c

    def release(self) -> None:
        if self._close is not None:
            self._close()
        self._arr = None
        self._close = None
        self._open_index = None

    def read_band(self, c: int, y0: int, y1: int, validator=None) -> np.ndarray:
        self._select(c)
        width = self.shape[2]
        raw = np.asarray(self._arr[0, slice(y0, y1), slice(0, width)])
        band = np.clip(raw, 0, None) if self._signed else raw
        if validator is not None:
            validator.observe(raw, band)
        return to_uint16(band) if self._need_cast else band


class _ArrayMaskSource(_BandSource):
    """A resident (2, H, W) mask stack behind the band interface."""

    dtype = np.dtype(np.uint32)
    src_itemsize = 4

    def __init__(self, mask_stack: np.ndarray):
        self._data = mask_stack
        self.shape = tuple(int(n) for n in mask_stack.shape)

    def read_band(self, c: int, y0: int, y1: int, validator=None) -> np.ndarray:
        return self._data[c, y0:y1].astype(np.uint32)


class _MaskFileSource(_BandSource):
    """The cell and nuclei mask TIFFs, read one row band at a time.

    The mask stack is uint32, 8 bytes per pixel for the pair, so holding it resident
    would dominate this process's memory. No pyramid is built: mean-downsampling would
    corrupt categorical label IDs.
    """

    dtype = np.dtype(np.uint32)
    src_itemsize = 4

    def __init__(self, cell_path: str, nuclei_path: str, height: int, width: int):
        self._paths = [str(cell_path), str(nuclei_path)]
        self.shape = (2, int(height), int(width))
        self._open_index = None
        self._arr = None
        self._close = None

    def _select(self, c: int) -> None:
        if self._open_index == c:
            return
        self.release()
        arr, _dtype, close = open_lazy(self._paths[c])
        # `_open_mask_source` sees only the first page's Y/X, so the plane count is
        # checked here; otherwise a multi-plane mask would embed plane 0 silently.
        if arr.shape[0] != 1:
            close()
            raise ValueError(
                f"{self._paths[c]}: expected a single-plane mask, got {arr.shape[0]} "
                "planes. SEGMENT writes one plane per mask."
            )
        self._arr, self._close, self._open_index = arr, close, c

    def release(self) -> None:
        if self._close is not None:
            self._close()
        self._arr = None
        self._close = None
        self._open_index = None

    def read_band(self, c: int, y0: int, y1: int, validator=None) -> np.ndarray:
        self._select(c)
        width = self.shape[2]
        return np.asarray(self._arr[0, slice(y0, y1), slice(0, width)]).astype(
            np.uint32
        )


def _plane_tiles(source: _BandSource, tile: int, validators=None):
    """Yield a (C, H, W) source as full-resolution tiles in tifffile's order.

    Order is (channel, tile row, tile column); each tile is the full `tile` x `tile` size
    with the slide edge zero-padded. A fresh buffer per tile, because the writer
    consumes lazily and a reused buffer would be overwritten before encoding.

    Bands are read in whole tile rows and dropped before the next, so peak is one band
    plus one tile.
    """
    c_n, h, w = source.shape
    band_rows = _band_rows(w, source.src_itemsize, tile)
    for c in range(c_n):
        validator = validators[c] if validators else None
        if validator is not None:
            log(f"  Streaming: {validator.name}")
        for band_y0 in range(0, h, band_rows):
            band_y1 = min(band_y0 + band_rows, h)
            band = source.read_band(c, band_y0, band_y1, validator=validator)
            for y0 in range(band_y0, band_y1, tile):
                th = min(tile, band_y1 - y0)
                for x0 in range(0, w, tile):
                    tw = min(tile, w - x0)
                    out = np.zeros((tile, tile), dtype=source.dtype)
                    out[:th, :tw] = band[y0 - band_y0 : y0 - band_y0 + th, x0 : x0 + tw]
                    yield out
            del band
        # Nothing that must happen may go after the last `yield`: tifffile stops calling
        # next() once it has the final tile, so the tail never runs for the last channel.
        # The caller finalizes validators after `write` returns.
        source.release()
        gc.collect()


class _PyramidLevels:
    """Derives every pyramid level once, per channel, and serves them in tifffile's order.

    tifffile fills the SubIFDs one level at a time (all channels of level 1, then level
    2, ...), while level k must be derived from level k-1 of the same channel. So level
    1's pass reads each base plane, derives that channel's whole chain (`level_chain`),
    hands level 1 to the writer, and keeps levels 2..K.

    Levels 2 and deeper sum to about `H*W/6` bytes per channel and are dropped (`drop`)
    as soon as written. Keeping every level would hold about `base/3`; re-reading the
    base for each level would make peak independent of channel count but re-decode every
    plane once per level, which is much slower.
    """

    def __init__(self, source: _BandSource, scale: int, n_sublevels: int):
        self._source = source
        self._scale = scale
        self._n = n_sublevels
        #: (channel, level) -> plane, for levels 2..K only.
        self._stash: Dict[Tuple[int, int], np.ndarray] = {}

    def tiles(self, level_index: int, level_shape, tile: int):
        """Yield level `level_index` as tiles, in tifffile's (channel, row, col) order."""
        c_n = self._source.shape[0]
        lh, lw = level_shape
        for c in range(c_n):
            if level_index == 1:
                chain = self._source.level_chain(c, self._scale, self._n)
                plane = chain[0]
                for offset, deeper in enumerate(chain[1:], start=2):
                    self._stash[(c, offset)] = deeper
                del chain
            else:
                plane = self._stash[(c, level_index)]
            for y0 in range(0, lh, tile):
                th = min(tile, lh - y0)
                for x0 in range(0, lw, tile):
                    tw = min(tile, lw - x0)
                    out = np.zeros((tile, tile), dtype=self._source.dtype)
                    out[:th, :tw] = plane[y0 : y0 + th, x0 : x0 + tw]
                    yield out
            del plane
            # Nothing that must run goes after the last `yield` (see `_plane_tiles`).
            self._source.release()
            gc.collect()

    def drop(self, level_index: int) -> None:
        """Release a level once it is on disk. Called by the writer, not by `tiles`, because
        a generator's tail does not run for the last channel."""
        for key in [k for k in self._stash if k[1] == level_index]:
            del self._stash[key]
        gc.collect()


def write_pyramidal_ome_tiff_from_source(
    source: _BandSource,
    output_path: str,
    channel_names: List[str],
    channel_colors: List[Tuple[int, int, int]],
    physical_size_x: float = 0.325,
    physical_size_y: float = 0.325,
    pyramid_resolutions: int = 5,
    pyramid_scale: int = 2,
    tile_size: int = 512,
    compression: str = "zstd",
    compressionargs: Optional[Dict] = None,
    mask_source: Optional[_BandSource] = None,
    mask_names: Optional[List[str]] = None,
    validators=None,
):
    """Write the pyramidal OME-TIFF without ever holding the slide.

    Structure as Bio-Formats/QuPath expect: base resolution as one page per channel,
    pyramid levels in the reserved SubIFDs, and an optional second OME image for the
    masks. Every write is fed from a tile generator:

      * the base comes straight from the source's row bands;
      * each level is derived per channel from a whole plane (see
        `_downsample_plane_f32`) and tiled only on the way out;
      * the mask series streams the two mask files the same way.

    Peak is one float32 plane plus its level chain, plus levels 2..K of each channel
    until written (`_PyramidLevels`), so it grows slightly with channel count.

    Parameters are the same as `write_pyramidal_ome_tiff`, except that `source` and
    `mask_source` replace the `data` and `mask_stack` arrays, and `validators` carries the
    optional per-channel `_ChannelValidator` objects folded during the base pass.
    """
    num_channels, height, width = source.shape

    # Calculate pyramid levels
    levels = calculate_pyramid_levels(
        height,
        width,
        min_size=64,
        max_levels=pyramid_resolutions,
        scale_factor=pyramid_scale,
    )
    num_subresolutions = len(levels) - 1

    log(f"Writing pyramidal OME-TIFF with {len(levels)} resolution levels:")
    for i, (h, w) in enumerate(levels):
        log(f"  Level {i}: {w} x {h}")

    # bin/utils/ome_io.py builds the OME metadata.
    metadata = ome_metadata(channel_names, (physical_size_x, physical_size_y))

    # Common write options
    options = dict(
        tile=(tile_size, tile_size),
        compression=compression,
        photometric="minisblack",
        resolutionunit="CENTIMETER",
        # One compressor thread. tifffile defaults `maxworkers` to os.cpu_count() (not the
        # cgroup) for a compressed tiled write, and its encoded-tile queue then grows peak
        # memory with channel count. maxworkers=1 keeps it flat; wall clock is about the
        # same at this process's 2 CPUs, and written bytes are unchanged.
        maxworkers=1,
    )
    if compressionargs:
        options["compressionargs"] = compressionargs

    with ome_tiff_writer(output_path, bigtiff=True, ome=True) as tif:
        # Write base resolution with all channels
        # subifds parameter reserves space for pyramid levels
        log(f"  Writing base resolution ({width} x {height})...")
        tif.write(
            _plane_tiles(source, tile_size, validators=validators),
            shape=(num_channels, height, width),
            dtype=source.dtype,
            subifds=num_subresolutions,
            resolution=(1e4 / physical_size_x, 1e4 / physical_size_y),
            metadata=metadata,
            **options,
        )

        # Per-channel validation lines are emitted here, not from the tail of
        # `_plane_tiles`, whose code after the final `yield` never runs for the last
        # channel. One line per channel, in channel order.
        for validator in validators or ():
            validator.finalize()

        # Generate and write pyramid levels
        pyramid = _PyramidLevels(source, pyramid_scale, num_subresolutions)
        for level_idx in range(1, len(levels)):
            level_h, level_w = levels[level_idx]
            log(f"  Writing pyramid level {level_idx} ({level_w} x {level_h})...")

            tif.write(
                pyramid.tiles(level_idx, levels[level_idx], tile_size),
                shape=(num_channels, level_h, level_w),
                dtype=source.dtype,
                subfiletype=1,  # REDUCEDIMAGE flag for pyramid level
                resolution=(
                    1e4 / (physical_size_x * (pyramid_scale**level_idx)),
                    1e4 / (physical_size_y * (pyramid_scale**level_idx)),
                ),
                **options,
            )
            # Freed here, not in the generator (whose tail does not run for the last
            # channel), so peak stays at the deepest live pair of levels.
            pyramid.drop(level_idx)

        # Optional second series: segmentation masks (cell + nuclei), uint32,
        # single full-resolution. This is a fresh top-level tif.write (no
        # subifds/subfiletype) which tifffile records as a NEW OME image
        # (Image:1) rather than a pyramid level of Image:0. Mean-downsampling
        # would corrupt categorical label IDs, so no pyramid is built here.
        if mask_source is not None:
            log(f"  Writing mask series (Image:1): {mask_source.shape} uint32")
            tif.write(
                _plane_tiles(mask_source, tile_size),
                shape=mask_source.shape,
                dtype=mask_source.dtype,
                metadata={
                    "axes": "CYX",
                    "Channel": {"Name": mask_names or ["cell_mask", "nuclei_mask"]},
                },
                tile=(tile_size, tile_size),
                compression=compression,
                photometric="minisblack",
                resolutionunit="CENTIMETER",
                maxworkers=1,  # see `options` above
            )

    log(f"Pyramidal OME-TIFF complete: {output_path}")

    # Verify the output
    verify_ome_tiff(output_path)


def write_pyramidal_ome_tiff(
    data: np.ndarray,  # Full CYX array
    output_path: str,
    channel_names: List[str],
    channel_colors: List[Tuple[int, int, int]],
    physical_size_x: float = 0.325,
    physical_size_y: float = 0.325,
    pyramid_resolutions: int = 5,
    pyramid_scale: int = 2,
    tile_size: int = 512,
    compression: str = "zstd",
    compressionargs: Optional[Dict] = None,
    mask_stack: Optional[np.ndarray] = None,
    mask_names: Optional[List[str]] = None,
):
    """
    Write a pyramidal OME-TIFF with proper QuPath-compatible structure, from arrays.

    Entry point for callers and tests that already hold a small stack. Wraps the arrays
    in `_ArraySource` / `_ArrayMaskSource` and defers to
    `write_pyramidal_ome_tiff_from_source`; the pipeline uses the file source.

    Parameters
    ----------
    data : ndarray
        3D numpy array in CYX order (channels, height, width).
    output_path : str
        Output file path.
    channel_names : list of str
        List of channel names.
    channel_colors : list of tuple
        List of (R, G, B) tuples.
    physical_size_x : float
        Pixel size in X (micrometers).
    physical_size_y : float
        Pixel size in Y (micrometers).
    pyramid_resolutions : int
        Number of pyramid levels.
    pyramid_scale : int
        Downscaling factor between levels.
    tile_size : int
        Tile size for efficient access.
    compression : str
        Compression algorithm.
    mask_stack : ndarray, optional
        Optional (2, H, W) uint32 array of segmentation masks (cell + nuclei
        labels). Written as a SECOND, single-resolution OME series (Image:1)
        so categorical label IDs are never mean-downsampled. Does not affect
        the intensity series (Image:0).
    mask_names : list of str, optional
        Channel names for the mask series (default ['cell_mask', 'nuclei_mask']).
    """
    write_pyramidal_ome_tiff_from_source(
        _ArraySource(data),
        output_path,
        channel_names,
        channel_colors,
        physical_size_x=physical_size_x,
        physical_size_y=physical_size_y,
        pyramid_resolutions=pyramid_resolutions,
        pyramid_scale=pyramid_scale,
        tile_size=tile_size,
        compression=compression,
        compressionargs=compressionargs,
        mask_source=None if mask_stack is None else _ArrayMaskSource(mask_stack),
        mask_names=mask_names,
    )


def verify_ome_tiff(path: str):
    """Verify the OME-TIFF structure is correct."""
    log("Verifying OME-TIFF structure...")
    with tifffile.TiffFile(path) as tif:
        log(f"  Number of pages: {len(tif.pages)}")
        log(f"  Number of series: {len(tif.series)}")

        if tif.series:
            series = tif.series[0]
            log(f"  Series 0 shape: {series.shape}")
            log(f"  Series 0 axes: {series.axes}")
            log(f"  Series 0 is_pyramidal: {series.is_pyramidal}")

            if hasattr(series, "levels") and series.levels:
                log(f"  Pyramid levels: {len(series.levels)}")
                for i, level in enumerate(series.levels):
                    log(f"    Level {i}: shape={level.shape}")

        if tif.ome_metadata:
            log("  OME-XML metadata: present")
            # Parse and show channel names
            import xml.etree.ElementTree as ET

            try:
                root = ET.fromstring(tif.ome_metadata)
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                channels = root.findall(".//ome:Channel", ns)
                if channels:
                    log(f"  Channels in OME-XML: {len(channels)}")
                    for ch in channels[:5]:  # Show first 5
                        log(f"    - {ch.get('Name')}")
                    if len(channels) > 5:
                        log(f"    ... and {len(channels) - 5} more")
            except Exception as e:
                log(f"  Warning: Could not parse OME-XML: {e}")
        else:
            log("  WARNING: No OME-XML metadata found!")


def _scan_channel_metadata(input_dir: str):
    """Pass 1: discover channel files and scan their metadata (order, dtype,
    shape).
    """
    # Find all single-channel TIFF files
    channel_files = sorted(
        list(Path(input_dir).glob("*.tif")) + list(Path(input_dir).glob("*.tiff"))
    )

    if not channel_files:
        raise ValueError(f"No TIFF files found in {input_dir}")

    log(f"Found {len(channel_files)} channel files")

    # PASS 1: Collect metadata
    log("-" * 50)
    log("Pass 1: Scanning channels for metadata...")
    channel_names = []
    channel_colors = []
    height, width, dtype = None, None, None

    for i, channel_file in enumerate(channel_files):
        channel_name = channel_file.stem
        channel_names.append(channel_name)
        color = generate_channel_color(channel_name, i)
        channel_colors.append(color)
        log(f"  [{i}] {channel_name}: RGB{color}")

        with tifffile.TiffFile(str(channel_file)) as tif:
            page = tif.pages[0]
            if height is None:
                if len(page.shape) == 2:
                    height, width = page.shape
                else:
                    height, width = page.shape[-2:]
                dtype = page.dtype
                log(f"  Image dimensions: {height} x {width}, dtype: {dtype}")
            else:
                if len(page.shape) == 2:
                    h, w = page.shape
                else:
                    h, w = page.shape[-2:]
                if (h, w) != (height, width):
                    raise ValueError(f"Dimension mismatch: {channel_name}")

    num_output_channels = len(channel_names)
    log(f"Total output channels: {num_output_channels}")

    return (
        channel_files,
        channel_names,
        channel_colors,
        height,
        width,
        dtype,
        num_output_channels,
    )


def _channel_source(channel_files, height, width, dtype):
    """Pass 2: a lazy view over the channel files, not a loaded stack.

    Nothing is read here; pixels arrive one row band at a time while the pyramid is
    written.
    """
    log("-" * 50)
    log("Pass 2/3: Streaming channels into the pyramidal OME-TIFF...")

    source = _ChannelFileSource(channel_files, height, width, dtype)
    # One validator per channel, folded during the base pass and emitted once per
    # channel after its last band.
    validators = [_ChannelValidator(Path(f).stem, dtype) for f in source.channel_files]
    return source, validators


def _open_mask_source(height, width, masks_dir):
    """Open the optional cell/nuclei mask pair that becomes a second OME series.

    Read only when masks_dir contains both expected files; a single full-resolution uint32
    series, never pyramided. Shapes come from the TIFF headers so a mismatch fails fast;
    pixels are streamed by `_MaskFileSource`.
    """
    if not masks_dir:
        return None, None

    cell_matches = sorted(Path(masks_dir).glob("*cell_mask.tif"))
    nuclei_matches = sorted(Path(masks_dir).glob("*nuclei_mask.tif"))
    if not cell_matches or not nuclei_matches:
        # --masks-dir is only passed when the mask series is REQUIRED
        # (embed_masks + expanded compartment quant). Missing masks here is a
        # hard error, not a silent skip — surface it with the dir contents.
        found = [p.name for p in Path(masks_dir).glob("*")]
        raise ValueError(
            f"--masks-dir '{masks_dir}' given but no *cell_mask.tif / *nuclei_mask.tif found. "
            f"Directory contains: {found}. Expected SEGMENT outputs like <patient>_cell_mask.tif."
        )
    cell_mask_path, nuclei_mask_path = cell_matches[0], nuclei_matches[0]

    log(f"Loading masks for second OME series from: {masks_dir}")
    shapes = []
    for path in (cell_mask_path, nuclei_mask_path):
        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            shapes.append(tuple(page.shape[-2:]))

    if shapes[0] != (height, width) or shapes[1] != (height, width):
        raise ValueError(
            "Mask/intensity dimension mismatch: intensity is "
            f"{(height, width)}, cell_mask is {shapes[0]}, "
            f"nuclei_mask is {shapes[1]}"
        )

    mask_source = _MaskFileSource(cell_mask_path, nuclei_mask_path, height, width)
    mask_names = ["cell_mask", "nuclei_mask"]
    log(f"  Mask stack shape: {mask_source.shape}, dtype: {mask_source.dtype}")
    return mask_source, mask_names


def _write_pyramid(
    output_path,
    source,
    channel_names,
    channel_colors,
    physical_size_x,
    physical_size_y,
    pyramid_resolutions,
    pyramid_scale,
    tile_size,
    compression,
    compressionargs,
    mask_source,
    mask_names,
    validators,
):
    """Pass 3: write the pyramidal OME-TIFF to a temp path (atomic write:
    write tmp → validate → rename is completed by the caller).
    """
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # PASS 3: Write pyramidal OME-TIFF (atomic: write tmp → validate → rename)
    # If the process is killed mid-write, only the .tmp file exists.
    # Nextflow won't find the expected output → marks as failed → retries.
    log("-" * 50)
    log("Pass 3: Writing pyramidal OME-TIFF...")
    tmp_path = output_path + ".tmp"

    write_pyramidal_ome_tiff_from_source(
        source,
        tmp_path,
        channel_names=channel_names,
        channel_colors=channel_colors,
        physical_size_x=physical_size_x,
        physical_size_y=physical_size_y,
        pyramid_resolutions=pyramid_resolutions,
        pyramid_scale=pyramid_scale,
        tile_size=tile_size,
        compression=compression,
        compressionargs=compressionargs,
        mask_source=mask_source,
        mask_names=mask_names,
        validators=validators,
    )

    return tmp_path


def _validate_written_pyramid(tmp_path, num_output_channels, channel_names, mask_shape):
    """Post-write validation: check the file has the right number of channels
    and is readable, including the optional mask series.
    """
    # Validate: check the file has the right number of channels and is readable.
    # Reads only the first tile per channel. Use tif.series[0] (the intensity series),
    # not tif.pages, since an optional mask series (Image:1) adds its own pages.
    # `mask_shape` is the expected (2, H, W); the masks were streamed, not held.
    log("Validating written file...")
    with tifffile.TiffFile(tmp_path) as tif:
        base_pages = list(tif.series[0].pages)
        n_written = len(base_pages)
        if n_written != num_output_channels:
            os.remove(tmp_path)
            raise RuntimeError(
                f"Pyramid validation failed: expected {num_output_channels} channels, "
                f"got {n_written} pages. File was likely truncated."
            )
        for i, page in enumerate(base_pages):
            # Read a single tile (first one) to verify the page is intact
            try:
                seg = page.asarray(maxworkers=1)[:256, :256]
                del seg
            except Exception as e:
                os.remove(tmp_path)
                raise RuntimeError(
                    f"Pyramid validation failed: channel {i} ({channel_names[i]}) "
                    f"is unreadable: {e}"
                )

        if mask_shape is not None:
            if len(tif.series) < 2:
                os.remove(tmp_path)
                raise RuntimeError(
                    "Pyramid validation failed: mask series was requested but "
                    "no second OME series was found in the written file."
                )
            mask_series = tif.series[1]
            if mask_series.dtype != np.uint32 or tuple(mask_series.shape) != tuple(
                mask_shape
            ):
                os.remove(tmp_path)
                raise RuntimeError(
                    "Pyramid validation failed: mask series shape/dtype mismatch "
                    f"(expected {tuple(mask_shape)} uint32, got {mask_series.shape} {mask_series.dtype})"
                )
            try:
                seg = mask_series.pages[0].asarray(maxworkers=1)[:256, :256]
                del seg
            except Exception as e:
                os.remove(tmp_path)
                raise RuntimeError(
                    f"Pyramid validation failed: mask series is unreadable: {e}"
                )
    log(
        f"  Validation passed: {n_written} channels intact"
        + (
            f", mask series {tuple(mask_shape)} intact"
            if mask_shape is not None
            else ""
        )
    )


def merge_channels(
    input_dir: str,
    output_path: str,
    physical_size_x: float = 0.325,
    physical_size_y: float = 0.325,
    pyramid_resolutions: int = 5,
    pyramid_scale: int = 2,
    tile_size: int = 512,
    compression: str = "zstd",
    compressionargs: Optional[Dict] = None,
    masks_dir: Optional[str] = None,
):
    """
    Merge all single-channel TIFF files into a single pyramidal OME-TIFF.
    """
    log("=" * 70)
    log("MERGE CHANNELS - Pyramidal OME-TIFF (QuPath Compatible)")
    log("=" * 70)
    log(f"Input directory: {input_dir}")
    log(f"Output: {output_path}")
    log(f"Pyramid: {pyramid_resolutions} levels, scale factor {pyramid_scale}")

    (
        channel_files,
        channel_names,
        channel_colors,
        height,
        width,
        dtype,
        num_output_channels,
    ) = _scan_channel_metadata(input_dir)

    source, validators = _channel_source(channel_files, height, width, dtype)

    mask_source, mask_names = _open_mask_source(height, width, masks_dir)

    try:
        tmp_path = _write_pyramid(
            output_path,
            source,
            channel_names,
            channel_colors,
            physical_size_x,
            physical_size_y,
            pyramid_resolutions,
            pyramid_scale,
            tile_size,
            compression,
            compressionargs,
            mask_source,
            mask_names,
            validators,
        )
    finally:
        source.release()
        if mask_source is not None:
            mask_source.release()

    _validate_written_pyramid(
        tmp_path,
        num_output_channels,
        channel_names,
        None if mask_source is None else mask_source.shape,
    )

    # Atomic rename — same filesystem, so this is instantaneous
    os.replace(tmp_path, output_path)

    gc.collect()

    # Summary
    file_size = Path(output_path).stat().st_size / (1024 * 1024 * 1024)
    log("=" * 70)
    log(f"SUCCESS: {output_path}")
    log(f"  Size: {file_size:.2f} GB")
    log(f"  Channels: {num_output_channels}")
    log(f"  Channel list: {', '.join(channel_names)}")
    log("=" * 70)

    return channel_names


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge single-channel TIFFs into pyramidal OME-TIFF (QuPath compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory with single-channel TIFF files"
    )
    parser.add_argument("--output", required=True, help="Output OME-TIFF path")
    parser.add_argument(
        "--masks-dir",
        help="Directory containing cell_mask.tif and nuclei_mask.tif "
        "to embed as a second (uint32, single-resolution) OME series",
    )
    parser.add_argument(
        "--physical-size-x",
        # str, not float: may be the literal 'auto', resolved against the first channel
        # TIFF in --input-dir. No default: see nextflow.config's pixel_size note.
        type=str,
        default=None,
        help="Pixel size in X (micrometres), or 'auto' to read it from the input",
    )
    parser.add_argument(
        "--physical-size-y",
        # str, not float: may be the literal 'auto', resolved against the first channel
        # TIFF in --input-dir. No default: see nextflow.config's pixel_size note.
        type=str,
        default=None,
        help="Pixel size in Y (micrometres), or 'auto' to read it from the input",
    )
    parser.add_argument(
        "--pyramid-resolutions",
        type=int,
        default=8,
        help="Number of pyramid levels (default: 8)",
    )
    parser.add_argument(
        "--pyramid-scale",
        type=int,
        default=2,
        help="Downscaling factor between levels (default: 2)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=512, help="Tile size (default: 512)"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["lzw", "zlib", "zstd", "jpeg", "none"],
        help="Compression algorithm (default: zstd)",
    )
    return parser.parse_args()


def main() -> int:
    """Run merge and pyramid CLI."""
    configure_logging()
    # zarr is required for the lazy read; a missing zarr fails at the first read.
    args = parse_args()

    # 'auto' reads the scale off the first single-channel TIFF SPLIT_CHANNELS wrote;
    # they all came from one slide, so any of them carries the same header.
    _tiles = sorted(Path(args.input_dir).glob("*.tif")) + sorted(
        Path(args.input_dir).glob("*.tiff")
    )
    _probe = str(_tiles[0]) if _tiles else None
    args.physical_size_x = resolve_pixel_size(
        args.physical_size_x, _probe, source=args.input_dir
    )
    args.physical_size_y = resolve_pixel_size(
        args.physical_size_y, _probe, source=args.input_dir
    )
    compression = None if args.compression == "none" else args.compression

    # Set compression-specific arguments
    compressionargs = None
    if compression == "zstd":
        compressionargs = {"level": 2}  # Fast decompression for QuPath

    try:
        merge_channels(
            args.input_dir,
            args.output,
            physical_size_x=args.physical_size_x,
            physical_size_y=args.physical_size_y,
            pyramid_resolutions=args.pyramid_resolutions,
            pyramid_scale=args.pyramid_scale,
            tile_size=args.tile_size,
            compression=compression,
            compressionargs=compressionargs,
            masks_dir=args.masks_dir,
        )
        log("Complete!")
        return 0
    except Exception as e:
        log(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
