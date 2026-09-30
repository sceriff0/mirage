"""The one place this pipeline decides how to read an image, and the only place it writes one.

TWO OWNERSHIPS, ONE MODULE.

**Which reader claims a file.** `bin/convert_image.py` used to answer this with three
module-level sets and a silent `return "bioio"` fall-through, so an unrecognised
extension -- a .png dropped into a samplesheet -- was handed to BioImage and failed
several frames inside a plugin, naming a problem that was not the problem.
`detect_reader` answers it from ONE table and raises `UnsupportedFormatError` for
anything unclaimed, with the suffix and the supported list in the message.

**Every TIFF this pipeline writes.** Before this module there was no writer helper at
all: ten scripts called `tifffile.imwrite`/`tifffile.TiffWriter` directly, and the OME
metadata dict (PhysicalSizeX/Y + unit + channel names + axes) was rebuilt independently
in four of them -- plus a fifth, hand-written `<OME ...>` XML template in
`merge_channels_pyramid.py` that nothing ever called. `tests/test_ome_io_is_the_only_writer.py`
now fails on any `tifffile` write call outside this file.

LAZY IMPORTS ARE A HARD REQUIREMENT, NOT A STYLE CHOICE. `requirements/ci.txt` installs
`tifffile`, `numpy`, `zarr` and `dask`, and installs NONE of `bioio`, `bioio-bioformats`
or `h5py` (they belong to `containers/convert` alone). A module-scope `import bioio` here
would break every pytest that imports this module, in CI, at collection time -- silently,
because a module-scope failure deselects the whole file rather than failing it (see
tests/conftest.py's floor). So `tifffile` and `numpy` are the only hard imports; every
vendor-format reader imports its backend INSIDE the function that needs it.

IMPORT CONVENTION. Flat (`from ome_io import ...`) by scripts that
`sys.path.insert(0, .../bin/utils)` first, matching `pixel_size.py`, `tiled_io.py` and
`segment_io.py`.

`jvm_cache` is a third module-scope sibling import, alongside `pixel_size` and `tiled_io`:
it is stdlib-only at import time (its own `scyjava` import is lazy, inside the function
body), so importing it here does not violate the lazy-heavy-dependency rule above. It is
called by `_open_bioio` for the `bioio-bioformats` route -- see that function's docstring.
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

#: suffix -> reader. THE dispatch table; there is no fall-through.
#:
#: `.ome.tif`/`.ome.tiff` are listed separately from `.tif`/`.tiff` even though both
#: resolve to the same reader, because the two-part suffix is the one this pipeline's own
#: intermediates carry and a future split (an OME-specific fast path) must not have to
#: reintroduce the double-suffix parse to make it.
#:
#: The `bioio-bioformats` block is the R2 route: "any Bio-Formats-compatible format",
#: honoured by installing `bioio-bioformats` in `containers/convert` and baking its jars
#: at build time (phase 06). Declaring the route here BEFORE the plugin exists is
#: deliberate -- `require_reader` turns its absence into a message naming the plugin,
#: which is strictly better than a suffix that silently routes somewhere wrong.
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

#: pip distribution name per module, for the message. `bioio_bioformats` is installed as
#: `bioio-bioformats`; telling an operator to `pip install bioio_bioformats` works by
#: accident today and is not what the container's requirements file says.
_MODULE_TO_DISTRIBUTION = {
    "bioio": "bioio",
    "bioio_bioformats": "bioio-bioformats",
    "tifffile": "tifffile",
    "h5py": "h5py",
}


def _module_is_importable(name: str) -> bool:
    """Whether `name` can be imported, WITHOUT importing it.

    `importlib.util.find_spec` rather than a try/import: importing `bioio` pulls a
    five-plugin, JVM-adjacent stack, and this is called on a path where the answer is
    usually "no". Exported (no underscore prefix would be nicer, but it is genuinely
    internal) only so tests/test_ome_io.py can ask the same question before asserting on
    the absent-plugin message.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # find_spec raises for a namespace package whose parent is missing.
        return False


def detect_reader(path) -> str:
    """Which of `READERS` claims `path`, by extension alone.

    Extension alone, deliberately: this is called before the file is opened -- it is what
    DECIDES how to open it -- and content sniffing a vendor format needs the very reader
    this function is choosing. A file whose bytes disagree with its name fails at the
    read, loudly, which is the right place for that failure.

    Raises `UnsupportedFormatError` for anything not in the table. There is no
    fall-through: `convert_image.get_file_format`'s silent `return "bioio"` is the
    behaviour this replaces.
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

    The `.svs/.qptiff/.vsi/.scn/.mrxs/.bif/.ims` route needs `bioio-bioformats`, which
    `containers/convert` installs (with its Bio-Formats jars baked at build time) and no
    other image does. Without this check a run in the wrong image dies with a bare
    `ModuleNotFoundError: No module named 'bioio_bioformats'` several frames inside a
    reader, which does not tell an operator that the fix is the image.
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


#: TIFF tile size (px) for the pipeline's canonical converted intermediate -- a TIFF
#: LAYOUT choice, not a processing tile size (contrast ``params.preproc_tile_size``,
#: BaSiC's pseudo-FOV grid).
#:
#: tifffile's zarr view of a STRIPED (untiled) TIFF reports ``chunks=(1, H, W)``, one
#: chunk per whole plane, so every "read just this region" call downstream -- the BaSiC
#: path, the STARE registration path, SPLIT_CHANNELS, the QC processes -- decodes the
#: entire plane to slice it. Measured on a 6000x6000 uint16 plane: a single 2048^2 region
#: read peaks at 76.7 MiB striped vs 16.0 MiB tiled. CONVERT_IMAGE writes the pipeline's
#: canonical intermediate, so an untiled write there is the origin of that cost for every
#: reader below it. Pinned by ``tests/test_convert_streaming_write.py``.
CONVERT_TIFF_TILE = 2048

#: Above this many bytes a classic TIFF's 32-bit offsets overflow
#: (``struct.error: 'I' format requires 0 <= number <= 4294967295``). 2 GiB rather than
#: 4 GiB because the ceiling applies to the largest OFFSET, not to the payload, and the
#: header/IFD overhead of a tiled multi-page file is not accounted for anywhere else.
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

    THIS IS THE FINDING THIS MODULE EXISTS FOR. The same six-key dict was rebuilt
    independently in ``convert_image.py``, ``merge_channels_pyramid.py``,
    ``apply_basic_profiles.py`` and ``tiled_stitch.py`` -- four copies of one convention,
    each free to drift in unit spelling, key order or which keys it omits. A fifth copy,
    ``merge_channels_pyramid.build_ome_xml``, hand-wrote the ``<OME ...>`` XML itself and
    was called by nothing at all.

    KEY ORDER IS PART OF THE OUTPUT. tifffile walks this dict to build the XML, so the
    order below is the attribute order in the header of every file the pipeline writes.
    It is the order ``convert_image.py`` used; changing it is a diff in a published
    artifact for no behavioural reason. ``tests/test_ome_io.py`` pins it.

    ``None`` OMITS RATHER THAN SUBSTITUTES. A missing scale means "the file did not say",
    and inventing one is the exact failure ``bin/utils/pixel_size.py`` exists to prevent
    (its module docstring has the full argument). Same for channel names: an anonymous
    header is honest, a mislabelled one is not.

    ``pixel_size_um`` may be one number (X and Y equal) or an ``(x, y)`` pair -- the
    vendor formats ``convert_image.py`` reads report the two separately and they can
    genuinely differ.
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

    ``None`` -- the default -- means "BigTIFF iff the payload exceeds
    ``BIGTIFF_THRESHOLD_BYTES``", so a small file stays a classic TIFF (which every reader
    accepts) and a real whole-slide image does not overflow. ``True``/``False`` are the
    caller asserting, and win: ``bin/convert_image.py`` passes ``True`` unconditionally
    because that is what it has always written and its outputs are compared against a
    reference writer.
    """
    if bigtiff is not None:
        return bool(bigtiff)
    return nbytes > BIGTIFF_THRESHOLD_BYTES


@contextlib.contextmanager
def ome_tiff_writer(path, *, bigtiff: bool = True, ome: bool = True):
    """An open ``tifffile.TiffWriter``, for a caller that must issue SEVERAL writes.

    ``write_ome_tiff`` below covers the single-shot case and is what most callers want.
    This exists for the two shapes it cannot express:

    * ``bin/merge_channels_pyramid.py`` writes the base resolution with ``subifds=N`` and
      then each pyramid level onto the SAME open writer -- several ``.write()`` calls that
      have to share one file handle.
    * ``bin/apply_basic_profiles.py`` and ``bin/tiled_stitch.py`` feed generators that
      COMPUTE each tile (an illumination correction, a warp) rather than slicing a source
      array, so there is no array for ``write_ome_tiff`` to walk.

    The caller keeps its generator and its per-write arguments; what moves here is the
    writer's construction, which is the part that was being re-decided per file.
    ``tests/test_ome_io_is_the_only_writer.py`` is what makes this the only route.
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

    THE ASSUMPTION THIS FUNCTION MAKES EXPLICIT: how much memory the write costs is the
    READER's decision, not this function's. It is bounded by ONE axis-0 chunk of whatever
    the reader handed back -- one plane when the reader chunks per plane (the good case),
    the whole stack when it returns a single chunk.

    So the planes are NOT fetched one index at a time. Doing that is a REGRESSION, not
    merely a missed optimisation: dask does not cache across independent ``compute()``
    calls, so on a single-chunk array ``image_data[i]`` decodes the WHOLE stack, once per
    plane -- same peak as an eager write, times N the work. Measured on an 8-plane
    single-chunk fixture: 8 whole-stack decodes.

    A plain ``.rechunk()`` does not fix that either (measured: still 8 decodes).
    Rechunking splits the graph DOWNSTREAM of the source read, and every plane's
    ``compute()`` still re-reads the source chunk. Iterating in chunk-sized slabs is what
    actually bounds it, and it bounds it at exactly one decode per chunk for every
    chunking --
    ``tests/test_convert_streaming_write.py::test_a_chunk_is_decoded_once_whatever_the_chunking``
    pins 1/2/8-plane chunks at 8/4/1 decodes.

    Array-likes with no ``chunks`` (a plain ndarray, or the NDPI/HDF5 readers' output)
    keep the plain per-index walk: there is no decode to amortise, and the caller may only
    support scalar-tuple indexing.
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

    tifffile's iterator mode is tile-wise, not plane-wise, whenever ``tile=`` is set:
    it walks ``numtiles`` items per page and raises ``ValueError('tile is too large')``
    the moment one is bigger than a single tile. Feeding it ``_iter_planes`` therefore
    only works while a whole plane happens to FIT IN ONE TILE, which is exactly the case
    a small fixture creates -- tifffile pads an undersized item without complaint. So the
    striped-vs-tiled write was once covered by a test that could not fail, and the first
    real slide (30552 x 32072) failed at CONVERT_IMAGE with "tile is too large".
    ``tests/test_convert_streaming_write.py`` and ``tests/test_ome_io.py`` both now write
    a plane LARGER than one tile.

    Partial edge tiles are yielded short and tifffile pads them, so the output is
    identical for shapes that are not tile multiples.

    Peak memory is unchanged: this only re-slices what ``_iter_planes`` already decoded,
    so the bound is still ONE SOURCE CHUNK, and all of that function's chunk reasoning
    applies here untouched.
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

    ``_iter_planes`` bounds the write at one axis-0 chunk, so a single-chunk array is
    never WORSE than an eager write. But it is not better either: that one decode is the
    whole decompressed stack. The saving is the reader's to give, and whether a given
    bioio plugin gives it is not something this pipeline controls -- so an operator who
    sized CONVERT_IMAGE expecting a per-plane peak is told here rather than finding out
    from an OOM.
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
    """Strided decimation for a pyramid level. Strided, not interpolated, on purpose:
    a pyramid level is a NAVIGATION aid, and this function is also used on LABEL images
    where any interpolation invents label ids that never existed."""
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
    a LAZY array-like (anything with ``.shape``, ``.dtype`` and per-index/slab indexing --
    a dask array is the case this was built for). Reading lazily and then writing eagerly
    saves nothing: a whole-array ``tifffile.imwrite`` passes its argument through
    ``numpy.asarray``, so the slide would simply be materialised at the write instead of
    at the read. Both halves had to change together, and this is the second half.

    The streaming form hands ``TiffWriter.write`` an ITERATOR of tiles together with an
    explicit ``shape=`` and ``dtype=``, which it cannot infer from a generator.

    THE WRITE IS TILED, NOT STRIPED -- see ``CONVERT_TIFF_TILE`` for the measurement.
    Tiling changes the on-disk layout, not the pixels.

    ``photometric="minisblack"`` is the precondition that makes ONE PAGE EQUAL ONE
    CHANNEL, which every downstream per-page read in this pipeline depends on. Without it
    tifffile stores a ``(C, H, W)`` array as one page with C samples and both ``key=`` and
    ``pages[i]`` raise IndexError. ``tests/test_slide_io_seam.py`` pins it.

    ``pyramid_levels`` > 1 writes the extra levels as subIFDs of the base page -- the
    layout QuPath reads -- each a strided 2x decimation of its parent. Each level's shape
    is derived from the ACTUAL downsampled array (``planes[0].shape``), never recomputed
    by ceil-division arithmetic, so an odd, non-square plane decimates correctly.
    ``bin/merge_channels_pyramid.py`` does NOT use this path: it needs per-level
    compression arguments, per-level resolution tags and its own validators, and goes
    through ``ome_tiff_writer`` instead.

    THE PYRAMID BRANCH IS NOT STREAMING. Building level 1 needs the whole base plane
    stack, so it materialises it in full (``np.stack`` of every plane) before decimating
    -- unlike the base-level write above, which stays tile-by-tile. No production caller
    passes ``pyramid_levels > 1`` today: ``convert_image.py`` is this function's only
    caller and always writes a single level; ``merge_channels_pyramid.py`` (the one
    multi-level writer) uses ``ome_tiff_writer`` directly, not this function.

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

        # The reduced levels. Each is derived from the level ABOVE it, not from the
        # base, so the decimation cost is geometric rather than quadratic -- and it is
        # materialised one level at a time, which for a >=2x reduction is always
        # smaller than the base the caller already handed us. The base itself was
        # never materialised as a WHOLE array above (it was streamed tile by tile), so
        # this is the first point the full base is held in memory; it is unavoidable
        # once a reduced level must be computed from it.
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

    DELIBERATELY THIN, AND HERE FOR ONE REASON: it makes the seam TOTAL. Eight call sites
    in ``bin/`` write something that is not an OME slide -- the segmentation label masks
    (three backends plus the ``add_cycle`` re-extraction), one single-channel plane per
    marker, the multi-site pseudo-FOV stack BASICPY fits on, and two ImageJ QC composites
    -- and each makes decisions this module has no opinion about (a mask is zlib+BigTIFF
    and tiled; an ImageJ composite is neither, and in fact CANNOT be BigTIFF, tifffile
    refuses the combination). Inventing defaults here would change all eight the day they
    were migrated.

    So it sets NOTHING. What it buys is that ``tests/test_ome_io_is_the_only_writer.py``
    can be a closed rule -- no allowlist, no per-file exception -- and that a new writer
    anywhere in ``bin/`` is a call into this module rather than a fresh, unreviewed set of
    TIFF decisions. Anything OME-shaped belongs in ``write_ome_tiff``, which does have
    opinions and states them.
    """
    tifffile.imwrite(str(path), data, **kwargs)
    return Path(path)


@dataclass(frozen=True)
class ImageInfo:
    """What is known about an image without decoding its pixels.

    ``shape_cyx`` is ALWAYS three-dimensional and channel-first, even for a 2-D file
    (promoted to ``C=1``). Every consumer in this pipeline indexes ``(C, Y, X)``, and a
    2-D image reported as ``(Y, X)`` makes ``shape_cyx[0]`` the image height -- a bug that
    reads as a plausible channel count.

    ``channels is None`` means the file carries no channel names, which is ORDINARY: most
    intermediates in this pipeline carry none, and `meta.channels` is what the pipeline
    actually trusts. ``pixel_size_um is None`` means the file declares no scale, which is
    likewise ordinary -- see ``bin/utils/pixel_size.py`` for why that must stay
    distinguishable from a scale that happens to equal the configured one.

    ``reader`` is ``detect_reader``'s answer for this path -- what a CONVERSION of it
    would use. It is not necessarily the module that produced this ``ImageInfo``: a
    ``.ome.tif`` routes to ``bioio`` for conversion but is inspected here through
    ``tifffile``, which needs no plugin and is installed everywhere.

    ``channels_are_samples`` is True for an interleaved RGB (or RGBA/CMYK-style) file --
    bioio reports one of those as TCZYXS with C=1 and S (samples) equal to the number of
    colour components, which is NOT the same axis this pipeline's own multichannel
    intermediates carry as C. ``read_info`` REMAPS S onto ``shape_cyx``'s C position (so
    ``shape_cyx == (S, Y, X)``, never the nonsense ``(1, X, S)`` a naive C/last-two-axes
    read would produce) and ``read_plane`` follows the same remap when indexing, so a
    caller never has to know which axis it actually came from. This flag exists so a
    caller that DOES care (a caller writing an RGB thumbnail, say) can tell "3 real
    channels" apart from "3 colour samples of one channel" after the fact.
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

    ``read_info`` reports THAT a file is interleaved (``channels_are_samples``) but not
    WHERE the samples axis is, and the two are not the same question: ``photometric="rgb"``
    writes ``YXS`` (S last) while ``planarconfig="separate"`` writes ``SYX`` (S first).
    ``read_plane`` needs the position, so it re-derives it from ``series.axes`` -- the
    same string ``_info_via_tifffile`` reads -- rather than assuming ``-1``.

    Deliberately NOT a field on ``ImageInfo``: it is meaningful only for the tifffile
    branch (bioio answers the same question through ``get_image_data("YX", S=...)``, which
    needs no index), and ``ImageInfo``'s shape is part of the published contract in
    ``docs/superpowers/plans/2026-09-02-release-1.0.0/00-master.md``.
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

    Mirrors ``bin/convert_image.py::read_single_ndpi``'s conversion exactly (RESUNIT 2 =
    inch, 3 = centimetre; anything else -- including the "no unit" default every plain
    TIFF this repo writes carries -- means no usable scale). Only reached for
    ``reader == "tifffile"`` (``.ndpi``/``.ndpis``), which has no OME header to answer
    ``read_ome_pixel_size`` and no other convention available; see that function's
    docstring for why it does not read these tags itself.
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

    ``_is_tifffile_readable`` diverts every plain ``.tif``/``.tiff``/``.ndpi`` here rather
    than through ``_info_via_bioio`` -- ``tifffile`` is installed in every image and in
    CI, ``bioio`` in exactly one -- so this function needs its OWN copy of the remap
    ``_info_via_bioio`` performs, not a shared one: the two readers report axes under
    different names (``img.dims.order``/``.S``/``.C`` for bioio, ``series.axes`` here) and
    there is no common object to factor the logic onto.

    ``series.axes`` names a samples axis ``'S'`` (e.g. ``'YXS'`` for an interleaved RGB
    TIFF written with ``photometric="rgb"``). Left unhandled, the samples axis stays
    wherever tifffile put it -- trailing, for ``YXS`` -- and ``shape_cyx`` comes out
    ``(Y, X, S)`` instead of ``(S, Y, X)``: a channel count that is actually the image
    height. Mirrors ``_info_via_bioio``'s R/G/B default for a 3-sample file, ``S0``..
    otherwise, only when the file names no channels itself.
    """
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0] if tf.series else tf.pages[0]
        shape = tuple(int(n) for n in series.shape)
        axes = getattr(series, "axes", None) or ""
        dtype = np.dtype(series.dtype)
        channels = _channel_names_from_ome(getattr(tf, "ome_metadata", None))
        # NDPI (reader == "tifffile") carries no OME header, only the classic TIFF
        # XResolution/YResolution/ResolutionUnit tags -- read the RAW values here,
        # inside the file handle's lifetime, and convert after it closes.
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

    ``require_reader`` lives INSIDE this function, not in its callers, so that a test
    monkeypatching ``_open_bioio`` bypasses the bioio/bioio-bioformats installed-ness
    check entirely -- a fake reader object handed back by the test stands in for the
    WHOLE bioio round trip, not just the ``BioImage(...)`` construction, which is what
    makes the S->C remap (see ``ImageInfo.channels_are_samples``) testable without bioio
    installed anywhere in this environment.

    For the ``bioio-bioformats`` route ONLY, this calls
    ``jvm_cache.point_jvm_cache_off_readonly_home()`` BEFORE constructing ``BioImage`` --
    that is the only call in the repository that starts the Bio-Formats JVM through bioio
    rather than through valis, and it is the same read-only-``$HOME`` trap
    ``bin/utils/valis_config.py::init_jvm`` already guards: scyjava derives its jgo
    ``cache_dir``/``m2_repo`` from ``Path.home()`` at import time and never consults
    ``JGO_CACHE_DIR``/``M2_REPO``, so nothing short of this call points it at the image's
    baked ``/root/.jgo`` + ``/root/.m2`` cache before the first Maven-triggering CALL --
    ``BioImage(str(path))`` below. Ordering versus the *import* of ``bioio``/``scyjava`` is
    immaterial: ``scyjava.config``'s setters rebind module globals that are only read when
    Maven actually resolves a dependency, which happens on the first JVM-touching call, not
    on import. Plain ``bioio`` (OME-TIFF/TIFF/ND2/CZI/LIF) starts no JVM, so the call is
    skipped there.
    """
    require_reader(reader)
    if reader == "bioio-bioformats":
        point_jvm_cache_off_readonly_home()
    from bioio import BioImage

    return BioImage(str(path))


def _info_via_bioio(path: Path, reader: str) -> ImageInfo:
    """``ImageInfo`` for anything routed through bioio, INCLUDING the S->C remap.

    bioio reports an interleaved RGB (or similar) file as ``TCZYXS`` with ``C=1`` and
    ``S`` (samples) equal to the number of colour components -- three colour SAMPLES of
    one channel, not three channels. Reading ``C`` and the trailing two shape entries
    without accounting for ``S`` (the pre-fix behaviour) reports ``(1, X, S)``: a
    channel count of 1, a "height" that is actually the image width, and a "width" that
    is actually the sample count. Whenever ``S`` is present and greater than 1 while
    ``C`` is exactly 1, this remaps ``S`` onto ``shape_cyx``'s C position instead, and
    defaults channel names to ``R``/``G``/``B`` for a 3-sample file (``S0``.. otherwise)
    when the file names none. Y and X are located by NAME in ``dims.order`` rather than
    assumed to be the trailing two shape entries, because a samples axis can itself
    trail Y/X (``...YXS``).
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

    THE READER THAT INSPECTS IS NOT ALWAYS THE READER THAT CONVERTS, and that is
    deliberate. ``detect_reader`` names what a CONVERSION would use -- ``bioio`` for a
    ``.ome.tif``, because the conversion path wants BioImage's dimension handling. But
    ``bioio`` is installed in exactly one image, and this pipeline's own intermediates are
    ``.ome.tif``, so routing inspection through it would make ``read_info`` unusable
    everywhere except ``containers/convert``. Anything ``tifffile`` can open is therefore
    inspected with ``tifffile``, which is installed in every image and in CI, while
    ``ImageInfo.reader`` still reports ``detect_reader``'s answer.

    ``pixel_size.read_ome_pixel_size`` is asked for the scale rather than the header being
    re-parsed here: that module owns the unit table, the OME default-unit rule and the
    non-positive rejection, and a second copy of any of them is a silent scale error in
    whichever copy falls behind.
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

    ``lazy=True`` (the default) reads through ``tiled_io.open_lazy`` -- tifffile's zarr
    view -- so ONLY that channel's tiles are decoded. The eager alternative,
    ``TiffFile.asarray(out="memmap")``, is misleadingly named: on compressed input it
    decodes the ENTIRE image into a new full-size uncompressed temp file and memory-maps
    THAT, so "extract one channel" costs a full decode and a full-size scratch file. On a
    20 GB slide that is 20 GB of temp disk to use one plane. ``bin/utils/segment_io.py``'s
    docstring records the measurement.

    ``lazy=False`` exists for the formats ``open_lazy`` cannot open (HDF5, and the vendor
    formats behind bioio) and for a caller that wants the plane materialised anyway.

    On a file where ``read_info`` found ``channels_are_samples`` True (an interleaved
    RGB-like file), ``channel`` indexes the SAMPLE axis, not C -- the same remap
    ``read_info`` applies, so a caller never has to know which axis it actually came
    from. **All three branches honour it**, which they did not before plan 07: the
    ``_is_tifffile_readable`` branches assumed the channel axis was axis 0 of the array
    tifffile hands back, which for an interleaved ``YXS`` file (``fmt_rgb.tiff``) is Y,
    so ``read_plane("fmt_rgb.tiff", 1)`` returned shape ``(64, 3)`` -- one ROW and all
    three samples -- instead of the ``(64, 64)`` single-sample plane promised above. A
    wrong-shaped array, not an error: the caller most likely to hit it is a QC or
    thumbnail path that would have gone on to write it out.

    The remap is applied HERE rather than inside ``tiled_io.open_lazy`` deliberately.
    ``open_lazy`` is shared by ``tiled_stitch``, ``merge_channels_pyramid``,
    ``registration`` and ``qc.py``, none of which ever sees interleaved bytes -- the main
    path runs through ``CONVERT_IMAGE``, which normalises to planar ``(C, Y, X)`` first --
    so changing what ``open_lazy`` presents would alter a view four callers depend on in
    order to fix a case none of them reach. ``_samples_axis_index`` re-derives the axis
    position from ``series.axes`` instead, which costs one header read and leaves
    ``open_lazy``'s contract untouched.
    """
    path = Path(path)
    info = read_info(path)
    n_channels = info.shape_cyx[0]
    if not 0 <= channel < n_channels:
        raise ValueError(
            f"{path.name}: channel {channel} out of range for C={n_channels}"
        )

    if _is_tifffile_readable(path):
        # `None` for the ordinary planar case, so `key` stays (channel, :, :) and this
        # is byte-for-byte the pre-plan-07 index for every non-interleaved file.
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
