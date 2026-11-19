"""The one place this pipeline decides how to read an image, and the only place it writes one.

**Which reader claims a file.** `detect_reader` answers from one table and raises
`UnsupportedFormatError` for anything unclaimed, naming the suffix and the supported list.

**Every TIFF this pipeline writes.** `tests/test_ome_io_is_the_only_writer.py` fails on
any `tifffile` write call outside this file.

Lazy imports are required: CI installs `tifffile`, `numpy`, `zarr` and `dask` but not
`bioio`, `bioio-bioformats` or `h5py` (those are in `containers/convert` only). A
module-scope import of them would deselect every test that imports this module. So only
`tifffile` and `numpy` are hard imports; vendor readers import their backend inside the
function. `jvm_cache` is safe at module scope: its `scyjava` import is lazy.

Imported flat (`from ome_io import ...`) by scripts that put `bin/utils` on `sys.path`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile
from jvm_cache import point_jvm_cache_off_readonly_home
from pixel_size import read_ome_pixel_size
from tiled_io import open_lazy

__all__ = [
    "CONVERT_TIFF_TILE",
    "READERS",
    "ImageInfo",
    "UnsupportedFormatError",
    "detect_reader",
    "ome_metadata",
    "ome_tiff_writer",
    "parse_ndpis",
    "read_info",
    "read_plane",
    "require_reader",
    "write_ome_tiff",
    "write_tiff",
]

logger = logging.getLogger(__name__)


class UnsupportedFormatError(ValueError):
    """Raised by `detect_reader` for an extension no installed reader claims.

    A `ValueError` subclass so a caller that already wraps a read in
    `except ValueError` keeps working, and its own type so a caller can tell
    "this pipeline does not read that format" apart from "this file is corrupt".
    """


#: The reader names `detect_reader` returns and `ImageInfo.reader` reports.
READERS = ("bioio", "bioio-bioformats", "tifffile", "hdf5")

#: suffix -> reader. The dispatch table; there is no fall-through.
#:
#: `.ome.tif`/`.ome.tiff` are listed separately from `.tif`/`.tiff` so an OME-specific
#: path can be split off later without reparsing the double suffix.
#:
#: The `bioio-bioformats` formats need that plugin, installed only in `containers/convert`;
#: `require_reader` turns its absence into a message naming the plugin.
_SUFFIX_TO_READER = {
    ".ome.tif": "bioio",
    ".ome.tiff": "bioio",
    ".tif": "bioio",
    ".tiff": "bioio",
    ".nd2": "bioio",
    ".czi": "bioio",
    ".lif": "bioio",
    ".ndpi": "tifffile",
    ".ndpis": "tifffile",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".svs": "bioio-bioformats",
    ".qptiff": "bioio-bioformats",
    ".vsi": "bioio-bioformats",
    ".scn": "bioio-bioformats",
    ".mrxs": "bioio-bioformats",
    ".bif": "bioio-bioformats",
    ".ims": "bioio-bioformats",
}

#: reader -> the importable module names it needs. `bioio-bioformats` needs BOTH the
#: `bioio` core and the `bioio_bioformats` plugin; naming only the plugin would let a
#: half-installed environment fail later and less clearly.
_READER_MODULES = {
    "bioio": ("bioio",),
    "bioio-bioformats": ("bioio", "bioio_bioformats"),
    "tifffile": ("tifffile",),
    "hdf5": ("h5py",),
}

#: pip distribution name per module, for the message (`bioio_bioformats` installs as
#: `bioio-bioformats`).
_MODULE_TO_DISTRIBUTION = {
    "bioio": "bioio",
    "bioio_bioformats": "bioio-bioformats",
    "tifffile": "tifffile",
    "h5py": "h5py",
}


def _module_is_importable(name: str) -> bool:
    """Whether `name` can be imported, without importing it.

    `find_spec` rather than a try/import: importing `bioio` pulls a heavy, JVM-adjacent
    stack. Also used by tests/test_ome_io.py before asserting on the absent-plugin message.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # find_spec raises for a namespace package whose parent is missing.
        return False


def detect_reader(path) -> str:
    """Which of `READERS` claims `path`, by extension alone.

    Extension alone: this decides how to open the file, and sniffing a vendor format would
    need the reader being chosen. A file whose bytes disagree with its name fails at the read.

    Raises `UnsupportedFormatError` for anything not in the table; there is no fall-through.
    """
    name = Path(path).name.lower()
    for double in (".ome.tif", ".ome.tiff"):
        if name.endswith(double):
            return _SUFFIX_TO_READER[double]

    suffix = Path(name).suffix
    reader = _SUFFIX_TO_READER.get(suffix)
    if reader is None:
        supported = ", ".join(sorted(_SUFFIX_TO_READER))
        raise UnsupportedFormatError(
            f"{Path(path).name}: no reader claims the extension "
            f"{suffix or '<none>'!r}. Supported extensions: {supported}."
        )
    return reader


def require_reader(reader: str) -> None:
    """Fail now, naming the plugin, if `reader`'s backend is not installed.

    The `.svs/.qptiff/.vsi/.scn/.mrxs/.bif/.ims` route needs `bioio-bioformats`, installed
    only in `containers/convert`. Without this check a run in the wrong image dies with a
    bare `ModuleNotFoundError` that does not say the fix is the image.
    """
    modules = _READER_MODULES.get(reader)
    if modules is None:
        raise UnsupportedFormatError(
            f"Unknown reader {reader!r}. Valid readers: {', '.join(READERS)}."
        )
    missing = [m for m in modules if not _module_is_importable(m)]
    if missing:
        distributions = ", ".join(_MODULE_TO_DISTRIBUTION[m] for m in missing)
        raise ImportError(
            f"the {reader!r} route needs {distributions}, which is not installed in this "
            f"environment. This format is read by containers/convert "
            f"(bolt3x/mirage-convert), which installs it; a run outside that image, or an "
            f"image built before {distributions} was added to requirements, cannot open it."
        )


#: TIFF tile size (px) for the canonical converted intermediate: a layout choice, not a
#: processing tile size (contrast ``params.preproc_tile_size``).
#:
#: A striped TIFF's zarr view has one chunk per whole plane, so every region read
#: downstream decodes the entire plane (a 2048^2 read on a 6000x6000 uint16 plane peaks
#: at 76.7 MiB striped vs 16.0 MiB tiled). Pinned by ``tests/test_convert_streaming_write.py``.
CONVERT_TIFF_TILE = 2048

#: Above this many bytes a classic TIFF's 32-bit offsets overflow. 2 GiB rather than
#: 4 GiB because the limit is on the largest offset, which includes header/IFD overhead.
BIGTIFF_THRESHOLD_BYTES = 2**31

_MICRON = "µm"


def ome_metadata(
    channels: Optional[Sequence[str]],
    pixel_size_um: Union[float, Tuple[Optional[float], Optional[float]], None],
    *,
    axes: str = "CYX",
    pixel_size_z_um: Optional[float] = None,
) -> dict:
    """The OME metadata dict tifffile serialises into an image's header.

    The single builder of this dict, so unit spelling, key order and omitted keys cannot
    drift between writers.

    Key order is part of the output: tifffile walks the dict to build the XML, so this is
    the attribute order in every header. ``tests/test_ome_io.py`` pins it.

    ``None`` omits rather than substitutes: an unknown scale or channel list is left out,
    never invented (see ``bin/utils/pixel_size.py``).

    ``pixel_size_um`` may be one number (X and Y equal) or an ``(x, y)`` pair; vendor
    formats report the two separately and they can differ.
    """
    if isinstance(pixel_size_um, tuple):
        px_x, px_y = pixel_size_um
    else:
        px_x = px_y = pixel_size_um

    md: dict = {"axes": axes}
    if channels:
        md["Channel"] = {"Name": list(channels)}
    if px_x is not None:
        md["PhysicalSizeX"] = px_x
        md["PhysicalSizeXUnit"] = _MICRON
    if px_y is not None:
        md["PhysicalSizeY"] = px_y
        md["PhysicalSizeYUnit"] = _MICRON
    if pixel_size_z_um is not None:
        md["PhysicalSizeZ"] = pixel_size_z_um
        md["PhysicalSizeZUnit"] = _MICRON
    return md


def _wants_bigtiff(bigtiff: Optional[bool], nbytes: int) -> bool:
    """Resolve the tri-state ``bigtiff`` argument.

    ``None`` (the default) means BigTIFF iff the payload exceeds
    ``BIGTIFF_THRESHOLD_BYTES``, so small files stay classic TIFFs. ``True``/``False`` are
    the caller's choice and win (``bin/convert_image.py`` always passes ``True``).
    """
    if bigtiff is not None:
        return bool(bigtiff)
    return nbytes > BIGTIFF_THRESHOLD_BYTES


@contextlib.contextmanager
def ome_tiff_writer(path, *, bigtiff: bool = True, ome: bool = True):
    """An open ``tifffile.TiffWriter``, for a caller that must issue SEVERAL writes.

    ``write_ome_tiff`` covers the single-shot case. This is for callers that need more:

    * ``bin/merge_channels_pyramid.py`` writes the base with ``subifds=N`` and then each
      pyramid level onto the same open writer.
    * ``bin/apply_basic_profiles.py`` and ``bin/tiled_stitch.py`` feed generators that
      compute each tile, so there is no array for ``write_ome_tiff`` to walk.
    """
    writer = tifffile.TiffWriter(str(path), bigtiff=bigtiff, ome=ome)
    try:
        yield writer
    finally:
        writer.close()


def _axis0_slabs(chunk_sizes: Tuple[int, ...]) -> Iterator[Tuple[int, int]]:
    """Turn a dask axis-0 chunk-size tuple into ``(start, stop)`` half-open spans."""
    start = 0
    for size in chunk_sizes:
        yield start, start + size
        start += size


def _iter_planes(image_data: Any, shape: Tuple[int, ...]) -> Iterator[np.ndarray]:
    """Yield the stack's 2-D ``(Y, X)`` planes in write order, ONE SOURCE CHUNK AT A TIME.

    ``shape[:-2]`` are the non-spatial axes (``C``, and ``Z``/``T`` when the caller has
    them); they are walked in C order, which is the page order tifffile expects for
    ``shape=``.

    Memory is bounded by one axis-0 chunk of what the reader returned: one plane when the
    reader chunks per plane, the whole stack when it returns a single chunk.

    Planes are not fetched one index at a time: dask does not cache across ``compute()``
    calls, so on a single-chunk array ``image_data[i]`` would decode the whole stack once
    per plane. ``.rechunk()`` does not help either (each plane still re-reads the source
    chunk). Iterating in chunk-sized slabs decodes each chunk exactly once;
    ``tests/test_convert_streaming_write.py::test_a_chunk_is_decoded_once_whatever_the_chunking``
    pins it.

    Array-likes with no ``chunks`` (a plain ndarray, the NDPI/HDF5 readers' output) keep
    the per-index walk.
    """
    lead = shape[:-2]
    chunks = getattr(image_data, "chunks", None)

    if not lead or not chunks:
        for index in itertools.product(*(range(n) for n in lead)):
            yield np.asarray(image_data[index])
        return

    trailing = tuple(range(n) for n in lead[1:])
    for start, stop in _axis0_slabs(chunks[0]):
        slab = np.asarray(image_data[start:stop])
        for index in itertools.product(range(stop - start), *trailing):
            yield slab[index]


def _iter_tiles(
    image_data: Any, shape: Tuple[int, ...], tile: Tuple[int, int]
) -> Iterator[np.ndarray]:
    """Yield TILES, row-major within each plane -- what tifffile's tiled writer wants.

    With ``tile=`` set, tifffile's iterator mode is tile-wise and raises
    ``ValueError('tile is too large')`` on any item bigger than one tile. Feeding it planes
    only works while a plane fits in one tile, which small fixtures hide; tests write a
    plane larger than one tile.

    Partial edge tiles are yielded short and tifffile pads them. Peak memory is still one
    source chunk, as in ``_iter_planes``.
    """
    th, tw = tile
    for plane in _iter_planes(image_data, shape):
        h, w = plane.shape[-2:]
        for y in range(0, h, th):
            for x in range(0, w, tw):
                yield plane[..., y : y + th, x : x + tw]


def _warn_if_the_reader_chunked_the_whole_stack_together(
    image_data: Any, shape: Tuple[int, ...]
) -> None:
    """Say so, loudly, when the lazy read cannot actually save any memory.

    With a single-chunk array, the one decode ``_iter_planes`` does is the whole
    decompressed stack. Whether a bioio plugin chunks per plane is outside this pipeline's
    control, so the operator is warned here rather than by an OOM.
    """
    chunks = getattr(image_data, "chunks", None)
    if not chunks or len(shape) < 3 or shape[0] < 2 or len(chunks[0]) != 1:
        return

    itemsize = np.dtype(image_data.dtype).itemsize
    total_gb = int(np.prod(shape)) * itemsize / 1e9
    logger.warning(
        f"The reader returned all {shape[0]} planes as one dask chunk, so the streamed "
        f"write must decode them together: peak memory is the whole {total_gb:.2f} GB "
        "decompressed stack, not one plane. Size this task from the DECOMPRESSED slide."
    )


def _downsample_plane(plane: np.ndarray, factor: int) -> np.ndarray:
    """Strided decimation for a pyramid level. Strided, not interpolated: this is also used
    on label images, where any interpolation invents label ids that never existed."""
    return plane[::factor, ::factor]


def write_ome_tiff(
    path,
    data: Any,
    *,
    channels: Optional[List[str]],
    pixel_size_um: Union[float, Tuple[Optional[float], Optional[float]], None],
    tile: int = CONVERT_TIFF_TILE,
    pyramid_levels: int = 1,
    bigtiff: Optional[bool] = None,
    axes: str = "CYX",
    pixel_size_z_um: Optional[float] = None,
) -> None:
    """Write ``data`` as a TILED OME-TIFF, streaming, never holding the whole slide.

    ``data`` is ``(C, Y, X)`` for the default ``axes='CYX'``; it may be a plain ndarray OR
    a lazy array-like (anything with ``.shape``, ``.dtype`` and per-index/slab indexing,
    typically dask). A whole-array ``tifffile.imwrite`` would materialise it via
    ``numpy.asarray``, so this hands ``TiffWriter.write`` an iterator of tiles with an
    explicit ``shape=`` and ``dtype=``.

    The write is tiled, not striped (see ``CONVERT_TIFF_TILE``).

    ``photometric="minisblack"`` makes one page equal one channel, which every per-page
    read downstream depends on; without it a ``(C, H, W)`` array becomes one page with C
    samples. ``tests/test_slide_io_seam.py`` pins it.

    ``pyramid_levels`` > 1 writes extra levels as subIFDs of the base page (the layout
    QuPath reads), each a strided 2x decimation of its parent, with shapes taken from the
    actual downsampled arrays so odd sizes work. The pyramid branch is not streaming: it
    materialises the base stack. ``bin/merge_channels_pyramid.py`` uses ``ome_tiff_writer``
    instead, for per-level compression and resolution tags.

    ``bigtiff=None`` means BigTIFF iff the payload exceeds ``BIGTIFF_THRESHOLD_BYTES``;
    pass ``True``/``False`` to assert.
    """
    shape = tuple(int(n) for n in data.shape)
    if len(shape) < 2:
        raise ValueError(
            f"write_ome_tiff needs at least a 2-D array, got shape {shape}"
        )

    if channels is not None and "C" in axes:
        n_channels = shape[axes.index("C")]
        if len(channels) != n_channels:
            raise ValueError(
                f"write_ome_tiff: {len(channels)} channel name(s) {channels} for a stack "
                f"with {n_channels} channel(s) (axes={axes!r}, shape={shape}). The names "
                "are read positionally by every downstream consumer, so a mismatched "
                "header is silently wrong rather than merely incomplete."
            )

    dtype = np.dtype(data.dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    _warn_if_the_reader_chunked_the_whole_stack_together(data, shape)

    metadata = ome_metadata(
        channels, pixel_size_um, axes=axes, pixel_size_z_um=pixel_size_z_um
    )
    levels = max(1, int(pyramid_levels))

    with ome_tiff_writer(
        path, bigtiff=_wants_bigtiff(bigtiff, nbytes), ome=True
    ) as writer:
        writer.write(
            _iter_tiles(data, shape, (tile, tile)),
            shape=shape,
            dtype=dtype,
            metadata=metadata,
            photometric="minisblack",
            tile=(tile, tile),
            subifds=levels - 1 if levels > 1 else None,
        )

        # Each reduced level is derived from the level above it (geometric cost). The base
        # is materialised here for the first time; unavoidable once a level is computed.
        if levels > 1:
            lead = shape[:-2]
            current = np.stack(list(_iter_planes(data, shape))).reshape(shape)
            for _level in range(1, levels):
                planes = [
                    _downsample_plane(plane, 2)
                    for plane in current.reshape((-1,) + current.shape[-2:])
                ]
                current = np.stack(planes).reshape(lead + planes[0].shape)
                writer.write(
                    _iter_tiles(current, tuple(current.shape), (tile, tile)),
                    shape=tuple(current.shape),
                    dtype=dtype,
                    photometric="minisblack",
                    tile=(tile, tile),
                    subfiletype=1,  # REDUCEDIMAGE
                )


def write_tiff(path, data: Any, **kwargs) -> Path:
    """Write a NON-OME TIFF, forwarding every keyword argument to tifffile verbatim.

    It sets nothing: the non-OME writers (label masks, single-channel planes, the BaSiC
    pseudo-FOV stack, ImageJ QC composites) each make their own TIFF choices, some of which
    conflict (an ImageJ composite cannot be BigTIFF). Its purpose is to keep
    ``tests/test_ome_io_is_the_only_writer.py`` a closed rule. OME slides belong in
    ``write_ome_tiff``.
    """
    tifffile.imwrite(str(path), data, **kwargs)
    return Path(path)


@dataclass(frozen=True)
class ImageInfo:
    """What is known about an image without decoding its pixels.

    ``shape_cyx`` is always 3-D and channel-first, a 2-D file promoted to ``C=1``, so
    ``shape_cyx[0]`` can never be mistaken for the height.

    ``channels is None`` (no channel names) and ``pixel_size_um is None`` (no declared
    scale) are both ordinary; see ``bin/utils/pixel_size.py`` for why a missing scale must
    stay distinguishable.

    ``reader`` is ``detect_reader``'s answer, i.e. what a conversion would use; a
    ``.ome.tif`` routes to ``bioio`` for conversion but is inspected with ``tifffile``.

    ``channels_are_samples`` is True for an interleaved RGB-like file, which bioio reports
    as C=1 with S colour samples. ``read_info`` maps S onto the C position, and
    ``read_plane`` indexes it the same way; the flag lets a caller tell 3 channels from
    3 colour samples.
    """

    path: Path
    shape_cyx: Tuple[int, int, int]
    dtype: np.dtype
    channels: Optional[List[str]]
    pixel_size_um: Optional[float]
    reader: str
    channels_are_samples: bool = False


#: Extensions this module can inspect and read planes from with `tifffile` alone.
#: `.ndpis` is NOT here: it is a manifest, not a TIFF, and is handled by delegating to
#: the first NDPI it names.
_TIFFFILE_READABLE = (".ome.tif", ".ome.tiff", ".tif", ".tiff", ".ndpi")


def _is_tifffile_readable(path) -> bool:
    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in _TIFFFILE_READABLE)


def _samples_axis_index(path) -> Optional[int]:
    """Where the SAMPLES axis sits in the array ``tifffile`` hands back, or ``None``.

    ``photometric="rgb"`` writes ``YXS`` (S last), ``planarconfig="separate"`` writes
    ``SYX`` (S first), so ``read_plane`` re-derives the position from ``series.axes``.
    Not an ``ImageInfo`` field: only the tifffile branch needs it.
    """
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0] if tf.series else tf.pages[0]
        axes = getattr(series, "axes", None) or ""
        shape = tuple(series.shape)
    if "S" not in axes or len(axes) != len(shape) or len(shape) != 3:
        return None
    return axes.index("S")


def parse_ndpis(ndpis_path) -> List[Path]:
    """The ``.ndpi`` files a Hamamatsu ``.ndpis`` manifest names, in declared order.

    The manifest format::

        [NanoZoomer Digital Pathology Image Set]
        NoImages=4
        Image0=2025-10-22 10.53.32-CY5.ndpi
        Image1=2025-10-22 10.53.32-TRITC.ndpi

    Each entry names a SIBLING file, so the manifest is ``resolve()``d first: Nextflow
    stages inputs as symlinks into a task directory while the ``.ndpi`` files stay where
    they were, and resolving is what makes the sibling lookup find them.
    """
    ndpis_path = Path(ndpis_path)
    parent = ndpis_path.resolve().parent
    entries: List[Path] = []
    for raw in ndpis_path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("Image") and "=" in line:
            entries.append(parent / line.split("=", 1)[1].strip())
    logger.info(f"Parsed NDPIS: {len(entries)} images (from {parent})")
    return entries


def _channel_names_from_ome(xml: Optional[str]) -> Optional[List[str]]:
    """OME ``Channel/@Name`` values, or None when the header names none.

    ``xml.etree`` rather than a regex: a channel name is operator-supplied text and may
    legitimately contain the characters a regex would have to guess about.
    """
    if not xml or not isinstance(xml, str):
        return None
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    names = [c.get("Name") for c in root.iter() if c.tag.endswith("}Channel")]
    named = [n for n in names if n]
    return named or None


def _pixel_size_from_ndpi_resolution_tags(x_res, y_res, res_unit) -> Optional[float]:
    """The classic-TIFF resolution convention NDPI actually carries, in µm.

    Mirrors ``bin/convert_image.py::read_single_ndpi`` (RESUNIT 2 = inch, 3 = centimetre;
    anything else means no usable scale). Only used for ``.ndpi``/``.ndpis``, which carry no
    OME header.
    """
    x_val = x_res[0] / x_res[1] if isinstance(x_res, tuple) else x_res
    y_val = y_res[0] / y_res[1] if isinstance(y_res, tuple) else y_res
    if not x_val or not y_val:
        return None
    if res_unit == 3:  # centimetre
        px_x = 10000.0 / x_val
    elif res_unit == 2:  # inch
        px_x = 25400.0 / x_val
    else:
        return None
    return px_x


def _info_via_tifffile(path: Path, reader: str) -> ImageInfo:
    """``ImageInfo`` for anything opened directly with ``tifffile``, INCLUDING the S->C
    remap this pipeline's contract needs for an interleaved file (RGB and similar).

    Plain ``.tif``/``.tiff``/``.ndpi`` come here (tifffile is installed everywhere, bioio
    in one image), so this has its own copy of ``_info_via_bioio``'s remap; the two report
    axes differently and share no object to factor it onto.

    Without the remap an interleaved ``YXS`` file would report ``(Y, X, S)``. Channel
    names default to R/G/B for 3 samples, ``S0``.. otherwise, when the file names none.
    """
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0] if tf.series else tf.pages[0]
        shape = tuple(int(n) for n in series.shape)
        axes = getattr(series, "axes", None) or ""
        dtype = np.dtype(series.dtype)
        channels = _channel_names_from_ome(getattr(tf, "ome_metadata", None))
        # NDPI has no OME header, only classic resolution tags: read them while the
        # file is open, convert after.
        resolution_values = None
        if reader == "tifffile":
            page0 = tf.pages[0]
            x_res = page0.tags.get("XResolution")
            y_res = page0.tags.get("YResolution")
            res_unit = page0.tags.get("ResolutionUnit")
            if x_res is not None and y_res is not None:
                resolution_values = (
                    x_res.value,
                    y_res.value,
                    getattr(res_unit, "value", None),
                )

    channels_are_samples = False
    if "S" in axes and len(axes) == len(shape):
        s_idx = axes.index("S")
        n_samples = shape[s_idx]
        if n_samples > 1:
            rest = [shape[i] for i in range(len(shape)) if i != s_idx]
            if len(rest) != 2:
                raise ValueError(
                    f"{path.name}: axes {axes!r} shape {shape} has a samples axis plus "
                    "more than two remaining (Y, X) axes; this pipeline's contract is "
                    "(C, Y, X)."
                )
            shape = (n_samples,) + tuple(rest)
            channels_are_samples = True
            if not (channels and len(channels) == n_samples):
                channels = (
                    ["R", "G", "B"]
                    if n_samples == 3
                    else [f"S{i}" for i in range(n_samples)]
                )

    if not channels_are_samples:
        if len(shape) == 2:
            shape = (1,) + shape
        elif len(shape) > 3:
            # Squeeze leading singletons (a TCZYX file whose T and Z are 1); anything
            # left over is genuinely more than this pipeline's (C, Y, X) contract covers.
            lead = [n for n in shape[:-2] if n != 1] or [1]
            if len(lead) != 1:
                raise ValueError(
                    f"{path.name}: shape {shape} has more than one non-singleton "
                    "non-spatial axis; this pipeline's contract is (C, Y, X)."
                )
            shape = (lead[0], shape[-2], shape[-1])

    px_x, px_y = read_ome_pixel_size(path)
    pixel_size = px_x if px_x is not None else px_y
    if pixel_size is None and resolution_values is not None:
        pixel_size = _pixel_size_from_ndpi_resolution_tags(*resolution_values)

    return ImageInfo(
        path=path,
        shape_cyx=shape,
        dtype=dtype,
        channels=channels,
        pixel_size_um=pixel_size,
        reader=reader,
        channels_are_samples=channels_are_samples,
    )


def _info_via_hdf5(path: Path, reader: str) -> ImageInfo:
    require_reader("hdf5")
    import h5py

    with h5py.File(str(path), "r") as handle:
        dataset = _first_image_dataset(handle)
        if dataset is None:
            raise ValueError(f"No image dataset found in HDF5 file: {path}")
        shape = tuple(int(n) for n in dataset.shape)
        dtype = np.dtype(dataset.dtype)
        px = None
        for obj in (dataset, handle):
            if "element_size_um" in obj.attrs:
                value = obj.attrs["element_size_um"]
                px = float(value[-1]) if hasattr(value, "__len__") else float(value)
                break
        channels = None
        for obj in (dataset, handle):
            if "channel_names" in obj.attrs:
                raw = obj.attrs["channel_names"]
                if hasattr(raw, "__len__") and len(raw):
                    channels = [str(n) for n in raw]
                    break

    if len(shape) == 2:
        shape = (1,) + shape
    elif len(shape) > 3:
        shape = (shape[-3], shape[-2], shape[-1])

    return ImageInfo(
        path=path,
        shape_cyx=shape,
        dtype=dtype,
        channels=channels,
        pixel_size_um=px,
        reader=reader,
    )


def _first_image_dataset(group):
    """The first >=2-D numeric dataset in an HDF5 group, depth first."""
    import h5py

    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            if item.ndim >= 2 and np.issubdtype(item.dtype, np.number):
                return item
        elif isinstance(item, h5py.Group):
            found = _first_image_dataset(item)
            if found is not None:
                return found
    return None


def _open_bioio(path: Path, reader: str):
    """The one place bioio is imported and a ``BioImage`` constructed.

    ``require_reader`` is called here, not by callers, so a test that monkeypatches this
    function replaces the whole bioio round trip and can exercise the S->C remap without
    bioio installed.

    For ``bioio-bioformats`` only, ``jvm_cache.point_jvm_cache_off_readonly_home()`` runs
    before ``BioImage`` is constructed: scyjava derives its jgo/Maven cache from
    ``Path.home()``, which is read-only on the cluster, and the redirect must happen before
    the first Maven-triggering call. Plain ``bioio`` starts no JVM.
    """
    require_reader(reader)
    if reader == "bioio-bioformats":
        point_jvm_cache_off_readonly_home()
    from bioio import BioImage

    return BioImage(str(path))


def _info_via_bioio(path: Path, reader: str) -> ImageInfo:
    """``ImageInfo`` for anything routed through bioio, INCLUDING the S->C remap.

    bioio reports an interleaved RGB-like file as ``TCZYXS`` with ``C=1`` and ``S`` colour
    samples. When ``S > 1`` and ``C == 1``, ``S`` is mapped onto the C position and channel
    names default to R/G/B for 3 samples (``S0``.. otherwise) if the file names none. Y and
    X are found by name in ``dims.order``, since S can trail them (``...YXS``).
    """
    img = _open_bioio(path, reader)
    order = getattr(img.dims, "order", None) or ""
    shape = tuple(int(n) for n in img.shape)
    y = shape[order.index("Y")] if "Y" in order else shape[-2]
    x = shape[order.index("X")] if "X" in order else shape[-1]
    c = int(img.dims.C)
    s = int(img.dims.S) if "S" in order else 1

    channels_are_samples = s > 1 and c == 1
    n_channels = s if channels_are_samples else c
    raw_names = list(img.channel_names) if img.channel_names else None
    if channels_are_samples:
        if raw_names and len(raw_names) == n_channels:
            channels = raw_names
        elif n_channels == 3:
            channels = ["R", "G", "B"]
        else:
            channels = [f"S{i}" for i in range(n_channels)]
    else:
        channels = raw_names

    px = getattr(img.physical_pixel_sizes, "X", None)
    return ImageInfo(
        path=path,
        shape_cyx=(n_channels, y, x),
        dtype=np.dtype(img.dtype),
        channels=channels,
        pixel_size_um=float(px) if px is not None else None,
        reader=reader,
        channels_are_samples=channels_are_samples,
    )


def read_info(path) -> ImageInfo:
    """Everything known about an image without decoding its pixels.

    The inspecting reader is not always the converting one: anything ``tifffile`` can open
    is inspected with it (installed everywhere), while ``ImageInfo.reader`` still reports
    ``detect_reader``'s answer (``bioio`` for a ``.ome.tif``).

    The scale comes from ``pixel_size.read_ome_pixel_size``, which owns the unit rules.
    """
    path = Path(path)
    reader = detect_reader(path)

    if path.name.lower().endswith(".ndpis"):
        entries = parse_ndpis(path)
        if not entries:
            raise ValueError(f"No NDPI files found in NDPIS manifest: {path}")
        first = _info_via_tifffile(entries[0], reader)
        # One NDPI per channel; the manifest's entry count IS the channel count.
        return ImageInfo(
            path=path,
            shape_cyx=(len(entries), first.shape_cyx[1], first.shape_cyx[2]),
            dtype=first.dtype,
            channels=None,  # the manifest names files, not markers
            pixel_size_um=first.pixel_size_um,
            reader=reader,
        )

    if _is_tifffile_readable(path):
        return _info_via_tifffile(path, reader)
    if reader == "hdf5":
        return _info_via_hdf5(path, reader)
    return _info_via_bioio(path, reader)


def read_plane(path, channel: int, *, lazy: bool = True):
    """One 2-D ``(Y, X)`` plane.

    ``lazy=True`` (the default) reads through ``tiled_io.open_lazy``, so only that
    channel's tiles are decoded. ``TiffFile.asarray(out="memmap")`` would decode the whole
    compressed image into a full-size temp file first.

    ``lazy=False`` is for formats ``open_lazy`` cannot open (HDF5, bioio vendor formats)
    or a caller that wants the plane materialised.

    When ``read_info`` reports ``channels_are_samples``, ``channel`` indexes the sample
    axis, in all three branches. The remap is done here rather than in ``open_lazy``,
    whose other callers never see interleaved data.
    """
    path = Path(path)
    info = read_info(path)
    n_channels = info.shape_cyx[0]
    if not 0 <= channel < n_channels:
        raise ValueError(
            f"{path.name}: channel {channel} out of range for C={n_channels}"
        )

    if _is_tifffile_readable(path):
        # `None` for the ordinary planar case, so `key` stays (channel, :, :).
        s_idx = _samples_axis_index(path) if info.channels_are_samples else None
        if s_idx is None:
            key = (channel, slice(None), slice(None))
        else:
            key = tuple(channel if axis == s_idx else slice(None) for axis in range(3))

        if lazy:
            # `key` must stay a plain 3-tuple (c, ys, xs): tiled_io.open_lazy's
            # `_CHW.__getitem__` unpacks it positionally (`c, ys, xs = key`), not by name.
            view, _dtype, close = open_lazy(str(path))
            try:
                return np.array(view[key], copy=True)
            finally:
                close()

        stack = np.asarray(tifffile.imread(str(path)))
        if stack.ndim == 2:
            return stack
        return np.asarray(stack[key])

    if info.reader == "hdf5":
        require_reader("hdf5")
        import h5py

        with h5py.File(str(path), "r") as handle:
            dataset = _first_image_dataset(handle)
            data = dataset[()]
        if data.ndim == 2:
            return data
        return np.asarray(data[channel])

    img = _open_bioio(path, info.reader)
    if info.channels_are_samples:
        return np.asarray(img.get_image_data("YX", C=0, S=channel))
    return np.asarray(img.get_image_data("YX", C=channel))
