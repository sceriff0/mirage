"""``bin/convert_image.py:convert_to_ome_tiff`` must write the slide one plane at a time.

BOTH HALVES OR NOTHING. Swapping ``img.data`` for ``img.dask_data`` (pinned by
``tests/test_convert_lazy_read.py``) on its own saves no memory at all: the lazy handle
is handed straight to ``tifffile.imwrite``, which calls ``numpy.asarray`` on it and
materialises the whole slide at the write instead of at the read. The write has to
stream -- ``tifffile.TiffWriter(...).write(<iterator of planes>, shape=..., dtype=...)``,
the pattern ``bin/tiled_stitch.py:156-170`` already uses for the gigapixel stitch.

This file pins the write half and, crucially, the OUTPUT EQUIVALENCE: the streamed file
must have the same decoded PIXELS and the same OME-XML header as what the previous
``tifffile.imwrite(path, whole_array, ...)`` call produced from the same pixels and the
same OME metadata.

That equivalence used to be checked byte-for-byte (masking only the UUID field below).
It no longer is, because of a *second*, later change layered on top of the streaming
write this docstring otherwise describes: ``write_ome_tiff`` now writes a TILED TIFF
(``tile=(convert_image.CONVERT_TIFF_TILE,) * 2``) instead of the striped layout
``tifffile.imwrite`` defaults to -- see ``convert_image.CONVERT_TIFF_TILE`` for why
(striped output makes every downstream region read decode a whole plane). Tiling changes
which TIFF tags get written (``TileWidth``/``TileLength``/``TileOffsets`` replace
``RowsPerStrip``/``StripOffsets``/``StripByteCounts``) and therefore the raw bytes, on
top of the UUID -- so a masked byte comparison would now fail for a reason that is
CORRECT, not a regression. The equivalence tests below therefore decode both files and
compare pixels (``tifffile.imread``) and compare the OME-XML text (masking the UUID as
before); ``test_the_write_is_tiled`` is the one that positively pins the new layout, so
that reverting the ``tile=`` argument still fails a test rather than silently passing
this weakened equivalence check.

The one byte-level caveat that DOES remain relevant, measured rather than assumed:
tifffile stamps a fresh time-based UUID into the OME-XML header
(``UUID="urn:uuid:..."``) on every write, so no two OME-TIFFs tifffile ever writes are
byte-identical, including two runs of the same writer.
``test_the_only_nondeterminism_is_the_ome_uuid`` is the control that establishes that --
it writes the same array twice through the OLD writer and asserts the ONLY differing
bytes fall inside that UUID field. The OME-XML comparison below masks that one
36-character field, which is exactly as strict as an XML-text comparison can be made.
"""

from __future__ import annotations

import importlib.machinery
import logging
import re
import sys
import types
from pathlib import Path

import dask.array as da
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin" / "utils"))

import ome_io  # noqa: E402

from bin import convert_image  # noqa: E402

_UUID_RE = re.compile(rb"urn:uuid:[0-9a-fA-F-]{36}")
_UUID_MASK = b"urn:uuid:00000000-0000-0000-0000-000000000000"


def _mask_uuid(raw: bytes) -> bytes:
    """Blank the one field tifffile randomises per write (see the module docstring)."""
    return _UUID_RE.sub(_UUID_MASK, raw)


def _legacy_write(
    output_filename,
    image_data,
    *,
    channels,
    pixel_size_um,
    tile=None,
    pyramid_levels=1,
    bigtiff=None,
    axes="CYX",
    pixel_size_z_um=None,
):
    """The write this task replaces, verbatim, kept only as the equivalence reference.

    ``bin/convert_image.py`` before the streaming rewrite::

        tifffile.imwrite(output_filename, image_data, metadata=ome_metadata,
                         photometric="minisblack", ome=True, bigtiff=True)

    It takes the SAME keyword signature the production writer now has (it is
    monkeypatched in over ``convert_image.write_ome_tiff``), and builds its metadata
    through ``ome_io.ome_metadata`` -- so what is being compared is the WRITE, not two
    different headers. ``tifffile.imwrite`` here is in ``tests/``, which
    ``tests/test_ome_io_is_the_only_writer.py`` deliberately does not scan: the point of
    a reference implementation is that it is the old code.
    """
    tifffile.imwrite(
        output_filename,
        np.asarray(image_data),
        metadata=ome_io.ome_metadata(
            channels, pixel_size_um, axes=axes, pixel_size_z_um=pixel_size_z_um
        ),
        photometric="minisblack",
        ome=True,
        bigtiff=True,
    )


# ---------------------------------------------------------------------------
# fixtures: drive the real convert_to_ome_tiff through a controlled read
# ---------------------------------------------------------------------------


class _RefusesWholeArray:
    """An array-like that serves 2-D planes but refuses whole-array materialisation.

    This is the tripwire that makes "the write streams" a behavioural assertion rather
    than a grep for ``TiffWriter``. ``tifffile.imwrite(path, array, ...)`` calls
    ``numpy.asarray`` on its argument; a writer that still does so trips ``__array__``
    and fails with the reason.
    """

    def __init__(self, array):
        self._array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.ndim = array.ndim
        self.planes_read = []

    def __getitem__(self, index):
        self.planes_read.append(index)
        plane = self._array[index]
        assert plane.ndim == 2, f"the writer asked for {plane.ndim}-D, not one plane"
        return plane

    def __array__(self, *_args, **_kwargs):
        raise AssertionError(
            "the writer materialised the whole stack; convert_to_ome_tiff must stream planes"
        )


class _FakeDims:
    def __init__(self, order, sizes):
        self.order = order
        self._sizes = sizes

    def __getattr__(self, name):
        sizes = self.__dict__.get("_sizes", {})
        if name in sizes:
            return sizes[name]
        raise AttributeError(name)

    def __repr__(self):
        return (
            "<Dimensions ["
            + " ".join(f"{d}: {self._sizes[d]}" for d in self.order)
            + "]>"
        )


class _FakePixelSizes:
    def __init__(self, x=None, y=None, z=None):
        self.X, self.Y, self.Z = x, y, z


class _FakeBioImage:
    def __init__(self, array, order, sizes, channel_names, pixel_sizes):
        self._array = array
        self.dims = _FakeDims(order, sizes)
        self.shape = array.shape
        self.channel_names = channel_names
        self.physical_pixel_sizes = pixel_sizes

    @property
    def data(self):
        raise AssertionError("BioImage.data materialises the entire slide")

    @property
    def dask_data(self):
        # One dask chunk per plane, which is what a real BioImage reader produces for a
        # planar file and what makes a per-plane write actually read per-plane.
        chunks = (1,) + self._array.shape[1:]
        return da.from_array(self._array, chunks=chunks)


def _install_fake_bioio(monkeypatch, img):
    module = types.ModuleType("bioio")
    module.BioImage = lambda _path: img
    # ome_io.require_reader (now called by convert_image.read_image before dispatch)
    # asks importlib.util.find_spec("bioio"), which -- for a module already present in
    # sys.modules -- returns sys.modules["bioio"].__spec__ rather than probing the
    # filesystem. A bare types.ModuleType has that attribute set to None, so find_spec
    # raises ValueError("bioio.__spec__ is None") and require_reader would report a
    # real bioio as "not installed" even though this fake module answers every call the
    # rest of this file makes on it. A real (if loader-less) ModuleSpec is enough to
    # satisfy find_spec without pulling in the actual package.
    module.__spec__ = importlib.machinery.ModuleSpec("bioio", loader=None)
    monkeypatch.setitem(sys.modules, "bioio", module)


def _planar_bioimage(seed=11, shape=(1, 3, 1, 24, 20)):
    """``TCZYX``, three channels, nuclear marker NOT in position 0 (forces the reorder)."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 4000, size=shape, dtype=np.uint16)
    return array, _FakeBioImage(
        array,
        order="TCZYX",
        sizes=dict(zip("TCZYX", shape)),
        channel_names=["CD3", "DAPI", "CD8"],
        pixel_sizes=_FakePixelSizes(0.325, 0.325, None),
    )


def _s_as_c_bioimage(seed=13, shape=(1, 1, 1, 24, 20, 3)):
    """``TCZYXS``: singleton ``C``, three ``S`` samples -- the RGB-ish vendor file."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=shape, dtype=np.uint16)
    return array, _FakeBioImage(
        array,
        order="TCZYXS",
        sizes=dict(zip("TCZYXS", shape)),
        channel_names=["CD3", "DAPI", "CD8"],
        pixel_sizes=_FakePixelSizes(0.325, 0.325, None),
    )


def _convert(monkeypatch, img, out_dir, channels, writer=None):
    """Run the real ``convert_to_ome_tiff``, optionally with the legacy writer patched in."""
    _install_fake_bioio(monkeypatch, img)
    args = (out_dir / "input.czi", out_dir, "P1")
    if writer is None:
        return convert_image.convert_to_ome_tiff(
            *args, channel_names=list(channels), pixel_size_um=0.325
        )
    # A context, not a bare setattr: monkeypatch.setattr lasts the whole test, so a
    # patched reference write followed by an "unpatched" one would silently compare the
    # legacy writer against itself and pass vacuously.
    with monkeypatch.context() as patched:
        patched.setattr(convert_image, "write_ome_tiff", writer)
        return convert_image.convert_to_ome_tiff(
            *args, channel_names=list(channels), pixel_size_um=0.325
        )


# ---------------------------------------------------------------------------
# the write streams
# ---------------------------------------------------------------------------


def test_the_write_never_materialises_the_whole_stack(monkeypatch, tmp_path):
    rng = np.random.default_rng(17)
    array = rng.integers(0, 4000, size=(3, 24, 20), dtype=np.uint16)
    guard = _RefusesWholeArray(array)

    monkeypatch.setattr(
        convert_image,
        "read_image",
        lambda _path: (
            guard,
            {
                "num_channels": 3,
                "physical_pixel_size_x": 0.325,
                "physical_pixel_size_y": 0.325,
                "physical_pixel_size_z": None,
                "channel_names_from_file": ["DAPI", "CD3", "CD8"],
                "original_dims": "CYX",
            },
        ),
    )

    out, channels = convert_image.convert_to_ome_tiff(
        tmp_path / "input.czi",
        tmp_path,
        "P1",
        channel_names=["DAPI", "CD3", "CD8"],
        pixel_size_um=0.325,
    )

    assert channels == ["DAPI", "CD3", "CD8"]
    assert len(guard.planes_read) == 3, (
        f"expected one read per plane, got {guard.planes_read}"
    )
    np.testing.assert_array_equal(tifffile.imread(str(out)), array)


# ---------------------------------------------------------------------------
# output equivalence
# ---------------------------------------------------------------------------


def test_streamed_pixels_match_the_eager_writer_for_a_multichannel_stack(
    monkeypatch, tmp_path
):
    array, img = _planar_bioimage()
    channels = ["CD3", "DAPI", "CD8"]

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    new_dir = tmp_path / "new"
    new_dir.mkdir()

    ref_path, ref_channels = _convert(
        monkeypatch, img, ref_dir, channels, writer=_legacy_write
    )
    new_path, new_channels = _convert(monkeypatch, img, new_dir, channels)

    assert ref_channels == new_channels == ["DAPI", "CD3", "CD8"]
    assert ref_path.name == new_path.name
    # Not a byte comparison: the streamed writer now tiles its output (see the module
    # docstring), so ``ref_path`` (striped) and ``new_path`` (tiled) legitimately differ
    # at the byte level even with the UUID masked. Decoded pixels are the equivalence
    # that must hold.
    np.testing.assert_array_equal(
        tifffile.imread(str(new_path)), tifffile.imread(str(ref_path))
    )

    # ... and the pixels really are the reordered source, not merely equal to each other.
    expected = np.take(array[0, :, 0], [1, 0, 2], axis=0)
    np.testing.assert_array_equal(tifffile.imread(str(new_path)), expected)


def test_streamed_pixels_match_the_eager_writer_for_the_s_as_c_case(
    monkeypatch, tmp_path
):
    array, img = _s_as_c_bioimage()
    channels = ["CD3", "DAPI", "CD8"]

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    new_dir = tmp_path / "new"
    new_dir.mkdir()

    ref_path, _ = _convert(monkeypatch, img, ref_dir, channels, writer=_legacy_write)
    new_path, _ = _convert(monkeypatch, img, new_dir, channels)

    # See test_streamed_pixels_match_the_eager_writer_for_a_multichannel_stack: the tiled
    # layout means the raw bytes differ from the striped reference on purpose, so this
    # compares decoded pixels rather than masked bytes.
    np.testing.assert_array_equal(
        tifffile.imread(str(new_path)), tifffile.imread(str(ref_path))
    )

    # TCZYXS -> squeeze C -> TZYXC -> transpose C in front of Y -> TCZYX -> squeeze T, Z
    expected = np.take(array[0, 0, 0].transpose(2, 0, 1), [1, 0, 2], axis=0)
    np.testing.assert_array_equal(tifffile.imread(str(new_path)), expected)


def test_the_ome_header_is_unchanged(monkeypatch, tmp_path):
    """Channel names and PhysicalSize live in the OME-XML; the streamed write must keep them."""
    _array, img = _planar_bioimage()
    channels = ["CD3", "DAPI", "CD8"]

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    new_dir = tmp_path / "new"
    new_dir.mkdir()

    ref_path, _ = _convert(monkeypatch, img, ref_dir, channels, writer=_legacy_write)
    new_path, _ = _convert(monkeypatch, img, new_dir, channels)

    with (
        tifffile.TiffFile(str(ref_path)) as ref,
        tifffile.TiffFile(str(new_path)) as new,
    ):
        ref_xml = _mask_uuid(ref.ome_metadata.encode())
        new_xml = _mask_uuid(new.ome_metadata.encode())

    assert new_xml == ref_xml
    assert b'Name="DAPI"' in new_xml
    assert b'PhysicalSizeX="0.325"' in new_xml


def test_the_only_nondeterminism_is_the_ome_uuid(tmp_path):
    """The control for the masking above: prove the OLD writer is not byte-stable either.

    Without this, ``_mask_uuid`` would be an unexamined escape hatch -- it could be
    hiding a real difference. Two runs of the *reference* writer over identical pixels
    are compared here, and the only differing byte offsets must fall inside the masked
    UUID field. If tifffile ever becomes deterministic, this test fails and the masking
    can be deleted.
    """
    rng = np.random.default_rng(23)
    array = rng.integers(0, 4000, size=(3, 24, 20), dtype=np.uint16)
    channels = ["DAPI", "CD3", "CD8"]
    pixel_size_um = 0.325

    first = tmp_path / "one.ome.tif"
    second = tmp_path / "two.ome.tif"
    _legacy_write(first, array, channels=channels, pixel_size_um=pixel_size_um)
    _legacy_write(second, array, channels=channels, pixel_size_um=pixel_size_um)

    raw_a = first.read_bytes()
    raw_b = second.read_bytes()
    assert raw_a != raw_b, (
        "tifffile now writes byte-identical OME-TIFFs; drop _mask_uuid and compare raw bytes"
    )

    differing = [i for i in range(len(raw_a)) if raw_a[i] != raw_b[i]]
    uuid_spans = [m.span() for m in _UUID_RE.finditer(raw_a)]
    assert uuid_spans, "no urn:uuid field found in the reference file"
    assert all(any(lo <= i < hi for lo, hi in uuid_spans) for i in differing), (
        f"bytes outside the OME UUID differ between two identical writes: {differing[:20]}"
    )

    assert _mask_uuid(raw_a) == _mask_uuid(raw_b)


def test_the_numpy_reader_branches_also_write_identical_pixels(monkeypatch, tmp_path):
    """The NDPI/HDF5 readers still hand over a decoded ``np.ndarray``.

    They go through the same streamed writer now, so their published pixels are in scope
    too -- a change confined to the BioIO branch would still have moved them. (Not a byte
    comparison -- see the module docstring: the streamed writer tiles, the reference
    writer does not, so the raw bytes differ on purpose.)
    """
    rng = np.random.default_rng(29)
    array = rng.integers(0, 4000, size=(3, 24, 20), dtype=np.uint16)
    metadata = {
        "num_channels": 3,
        "physical_pixel_size_x": None,
        "physical_pixel_size_y": None,
        "physical_pixel_size_z": None,
        "channel_names_from_file": None,
        "original_dims": "CYX",
    }
    monkeypatch.setattr(convert_image, "read_image", lambda _p: (array, dict(metadata)))

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    new_dir = tmp_path / "new"
    new_dir.mkdir()

    with monkeypatch.context() as patched:
        patched.setattr(convert_image, "write_ome_tiff", _legacy_write)
        ref_path, _ = convert_image.convert_to_ome_tiff(
            tmp_path / "in.ndpi",
            ref_dir,
            "P1",
            channel_names=["CD3", "DAPI", "CD8"],
            pixel_size_um=0.325,
        )
    new_path, _ = convert_image.convert_to_ome_tiff(
        tmp_path / "in.ndpi",
        new_dir,
        "P1",
        channel_names=["CD3", "DAPI", "CD8"],
        pixel_size_um=0.325,
    )

    np.testing.assert_array_equal(
        tifffile.imread(str(new_path)), tifffile.imread(str(ref_path))
    )
    np.testing.assert_array_equal(
        tifffile.imread(str(new_path)), np.take(array, [1, 0, 2], axis=0)
    )


# ---------------------------------------------------------------------------
# the tiled layout -- what this task exists to change
# ---------------------------------------------------------------------------


def test_the_write_is_tiled(monkeypatch, tmp_path):
    """Pin the layout change this task exists to make, and state why.

    ``write_ome_tiff`` used to write a STRIPED TIFF (tifffile's default): a plain
    ``tifffile.imread(path, aszarr=True)`` view of that reports ``chunks=(1, H, W)``, one
    chunk per whole plane, so a single "read just this 2048x2048 region" call downstream
    (the BaSiC path, the STARE registration path, SPLIT_CHANNELS, the QC processes) has to
    decode the ENTIRE plane to slice it out. Measured on a 6000x6000 uint16 plane: a single
    2048^2 region read peaks at 76.7 MiB striped vs 16.0 MiB tiled.

    Checked two ways, so a revert of the ``tile=`` argument in ``write_ome_tiff`` fails
    here rather than silently reintroducing that cost:

    1. the TIFF tags tifffile itself writes (``TileWidth``/``TileLength``), against
       ``convert_image.CONVERT_TIFF_TILE`` rather than a bare ``2048``, so the two stay in
       sync if the constant ever changes;
    2. the zarr chunk shape the pipeline's own region reader actually sees
       (``bin/utils/tiled_io.py:open_lazy`` uses exactly this ``aszarr=True`` + ``zarr.open``
       pattern), which is the property the whole task is about -- the TIFF tags could in
       principle be present without the zarr view honouring them, and this check is the one
       that would catch that.

    The un-tiled reference writer (``_legacy_write``, i.e. the writer this task's write
    replaced) is asserted to be striped, as the contrast: this is not merely "some file
    happens to be tiled", it is "the streamed writer changed layout and the old one did
    not".
    """
    import zarr

    array, img = _planar_bioimage()
    channels = ["CD3", "DAPI", "CD8"]

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    new_dir = tmp_path / "new"
    new_dir.mkdir()

    ref_path, _ = _convert(monkeypatch, img, ref_dir, channels, writer=_legacy_write)
    new_path, _ = _convert(monkeypatch, img, new_dir, channels)

    with tifffile.TiffFile(str(new_path)) as new_tif:
        new_tags = new_tif.pages[0].tags
        assert new_tags["TileWidth"].value == ome_io.CONVERT_TIFF_TILE
        assert new_tags["TileLength"].value == ome_io.CONVERT_TIFF_TILE

    with tifffile.TiffFile(str(ref_path)) as ref_tif:
        ref_tag_names = {t.name for t in ref_tif.pages[0].tags}
        assert "TileWidth" not in ref_tag_names, (
            "the reference writer is supposed to be striped, not tiled -- if this fires, "
            "tifffile's default changed and the contrast this test relies on is gone"
        )

    def _chunks(path):
        store = tifffile.imread(str(path), aszarr=True)
        try:
            grp = zarr.open(store, mode="r")
            arr = grp[0] if isinstance(grp, zarr.Group) else grp
            return arr.chunks
        finally:
            store.close()

    assert _chunks(new_path) == (1, ome_io.CONVERT_TIFF_TILE, ome_io.CONVERT_TIFF_TILE)
    assert _chunks(ref_path) == (1,) + array.shape[-2:]


# ---------------------------------------------------------------------------
# the pathological chunking: one whole-slide chunk
# ---------------------------------------------------------------------------


class _CountingSource:
    """Counts how many times the underlying storage is asked to decode.

    ``dask.array.from_array`` probes the source once while building the graph; tests
    reset ``decodes`` after construction so only the write is counted.
    """

    def __init__(self, array):
        self._array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.ndim = array.ndim
        self.decodes = 0

    def __getitem__(self, key):
        self.decodes += 1
        return self._array[key]


def _chunked_read(monkeypatch, array, chunks, channel_names):
    """Point convert_to_ome_tiff at a dask array with a chosen chunking."""
    source = _CountingSource(array)
    lazy = da.from_array(source, chunks=chunks)
    source.decodes = 0
    monkeypatch.setattr(
        convert_image,
        "read_image",
        lambda _p: (
            lazy,
            {
                "num_channels": array.shape[0],
                "physical_pixel_size_x": 0.325,
                "physical_pixel_size_y": 0.325,
                "physical_pixel_size_z": None,
                "channel_names_from_file": list(channel_names),
                "original_dims": "CYX",
            },
        ),
    )
    return source


def test_a_whole_slide_chunk_is_decoded_once_not_once_per_plane(monkeypatch, tmp_path):
    """The regression a naive per-plane loop introduces, pinned.

    If the reader hands back the stack as ONE dask chunk, asking for plane ``i`` decodes
    that whole chunk -- and dask does not cache across independent ``compute()`` calls, so
    a per-plane loop pays a FULL-STACK decode PER PLANE. That is strictly worse than the
    eager write it replaced: same peak memory, N times the work. Whether it happens is up
    to the bioio plugin's chunking, and ``bioio-czi`` is exactly the format PERF-PLAN's
    6 GB -> 20 GB example was about, so it cannot be assumed away.

    Measured before the fix on this fixture: 8 planes produced 8 whole-stack decodes.
    A plain ``.rechunk()`` does NOT fix it (measured: still 8) -- rechunking splits the
    graph downstream of the source read, and each plane's ``compute()`` re-reads the
    source chunk anyway. Only iterating in CHUNK-SIZED SLABS does.
    """
    rng = np.random.default_rng(31)
    array = rng.integers(0, 4000, size=(8, 24, 20), dtype=np.uint16)
    names = ["DAPI"] + [f"M{i}" for i in range(1, 8)]
    source = _chunked_read(monkeypatch, array, array.shape, names)

    out, _ = convert_image.convert_to_ome_tiff(
        tmp_path / "in.czi",
        tmp_path,
        "P1",
        channel_names=list(names),
        pixel_size_um=0.325,
    )

    assert source.decodes == 1, (
        f"the single whole-stack chunk was decoded {source.decodes} times for "
        f"{array.shape[0]} planes; a chunk must be decoded once"
    )
    np.testing.assert_array_equal(tifffile.imread(str(out)), array)


def test_a_chunk_is_decoded_once_whatever_the_chunking(monkeypatch, tmp_path):
    """One decode per chunk -- not per plane, and not per chunk per plane."""
    rng = np.random.default_rng(37)
    array = rng.integers(0, 4000, size=(8, 24, 20), dtype=np.uint16)
    names = ["DAPI"] + [f"M{i}" for i in range(1, 8)]

    for chunk_planes, expected in ((1, 8), (2, 4), (8, 1)):
        out_dir = tmp_path / f"c{chunk_planes}"
        out_dir.mkdir()
        with monkeypatch.context() as patched:
            source = _chunked_read(
                patched, array, (chunk_planes,) + array.shape[1:], names
            )
            out, _ = convert_image.convert_to_ome_tiff(
                tmp_path / "in.czi",
                out_dir,
                "P1",
                channel_names=list(names),
                pixel_size_um=0.325,
            )
        assert source.decodes == expected, (
            f"chunks of {chunk_planes} planes: {source.decodes} decodes, expected {expected}"
        )
        np.testing.assert_array_equal(tifffile.imread(str(out)), array)


def test_a_whole_slide_chunk_still_writes_identical_pixels(monkeypatch, tmp_path):
    """The slab path must not change the output, only the number of decodes.

    Compares decoded pixels, not raw bytes: the streamed writer tiles its output (see the
    module docstring), so its bytes differ from the striped reference writer on purpose.
    """
    rng = np.random.default_rng(41)
    array = rng.integers(0, 4000, size=(8, 24, 20), dtype=np.uint16)
    names = ["DAPI"] + [f"M{i}" for i in range(1, 8)]

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    new_dir = tmp_path / "new"
    new_dir.mkdir()

    with monkeypatch.context() as patched:
        _chunked_read(patched, array, array.shape, names)
        patched.setattr(convert_image, "write_ome_tiff", _legacy_write)
        ref_path, _ = convert_image.convert_to_ome_tiff(
            tmp_path / "in.czi",
            ref_dir,
            "P1",
            channel_names=list(names),
            pixel_size_um=0.325,
        )
    _chunked_read(monkeypatch, array, array.shape, names)
    new_path, _ = convert_image.convert_to_ome_tiff(
        tmp_path / "in.czi",
        new_dir,
        "P1",
        channel_names=list(names),
        pixel_size_um=0.325,
    )

    np.testing.assert_array_equal(
        tifffile.imread(str(new_path)), tifffile.imread(str(ref_path))
    )
    np.testing.assert_array_equal(tifffile.imread(str(new_path)), array)


def test_a_whole_slide_chunk_is_warned_about_by_name(monkeypatch, tmp_path, caplog):
    """One decode is the floor, not a fix: that decode is still the WHOLE stack.

    The slab loop makes the pathological chunking no worse than the eager write it
    replaced, but it cannot make it better -- the memory saving is the reader's to give.
    An operator who sized the task expecting per-plane peak has to be told.
    """
    rng = np.random.default_rng(43)
    array = rng.integers(0, 4000, size=(8, 24, 20), dtype=np.uint16)
    names = ["DAPI"] + [f"M{i}" for i in range(1, 8)]
    _chunked_read(monkeypatch, array, array.shape, names)

    with caplog.at_level(logging.WARNING):
        convert_image.convert_to_ome_tiff(
            tmp_path / "in.czi",
            tmp_path,
            "P1",
            channel_names=list(names),
            pixel_size_um=0.325,
        )

    assert "8 planes" in caplog.text and "one dask chunk" in caplog.text, caplog.text
    assert "peak" in caplog.text.lower()


def test_a_per_plane_chunked_read_is_not_warned_about(monkeypatch, tmp_path, caplog):
    """The control: a warning that always fires is noise, not a signal."""
    rng = np.random.default_rng(47)
    array = rng.integers(0, 4000, size=(8, 24, 20), dtype=np.uint16)
    names = ["DAPI"] + [f"M{i}" for i in range(1, 8)]
    _chunked_read(monkeypatch, array, (1,) + array.shape[1:], names)

    with caplog.at_level(logging.WARNING):
        convert_image.convert_to_ome_tiff(
            tmp_path / "in.czi",
            tmp_path,
            "P1",
            channel_names=list(names),
            pixel_size_um=0.325,
        )

    assert "one dask chunk" not in caplog.text


def test_a_plane_larger_than_one_tile_still_writes(tmp_path):
    """The case every other test in this file was too small to reach.

    tifffile's iterator mode is TILE-wise, not plane-wise, whenever ``tile=`` is set:
    it walks ``numtiles`` items per page and raises ``ValueError('tile is too large')``
    the moment one exceeds a single tile. Every fixture here is 24x20 -- smaller than
    one 2048x2048 tile -- so tifffile PADDED the plane without complaint and
    ``test_the_write_is_tiled`` above passed while the writer was feeding it whole
    planes. The first real slide (30552 x 32072) failed at CONVERT_IMAGE with
    "tile is too large", after the scheduler had granted the task its memory.

    Shrinking the tile rather than growing the image keeps this fast and additionally
    exercises PARTIAL EDGE TILES: 24x20 over a 16px tile is 2x2 tiles of which three
    are short, which is the other thing a plane-fed writer never reached.
    """
    import numpy as np
    import tifffile

    shape = (2, 24, 20)
    data = (np.arange(int(np.prod(shape)), dtype=np.uint16) % 4096).reshape(shape)
    out = tmp_path / "big_plane.ome.tif"
    # tile= as an ARGUMENT rather than monkeypatching the constant: the writer moved
    # to bin/utils/ome_io.py, and patching convert_image's imported binding would not
    # reach the default in ome_io.write_ome_tiff's own signature.
    convert_image.write_ome_tiff(
        out, data, channels=["a", "b"], pixel_size_um=None, tile=16
    )

    back = tifffile.imread(out)
    assert back.shape == shape
    assert np.array_equal(back, data), "pixels must survive the tile walk exactly"

    with tifffile.TiffFile(out) as tf:
        page = tf.pages[0]
        assert page.is_tiled
        assert (page.tilelength, page.tilewidth) == (16, 16)
