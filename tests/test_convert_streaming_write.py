"""``bin/convert_image.py:convert_to_ome_tiff`` must write the slide one plane at a time.

A lazy read (``tests/test_convert_lazy_read.py``) saves nothing on its own:
``tifffile.imwrite`` calls ``numpy.asarray`` on the handle and materialises the slide at
the write. The write has to stream through ``tifffile.TiffWriter(...).write(<iterator>)``.

This file pins the streamed write and its output equivalence with the eager
``tifffile.imwrite`` reference: same decoded pixels, same OME-XML. Bytes are not
compared because the streamed writer tiles its output (``CONVERT_TIFF_TILE``) and the
reference is striped; ``test_the_write_is_tiled`` pins the tiling itself.

tifffile stamps a fresh UUID into the OME-XML on every write, so the XML comparison
masks that one field; ``test_the_only_nondeterminism_is_the_ome_uuid`` is the control.
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
    """The eager ``tifffile.imwrite`` write, kept as the equivalence reference.

    Same keyword signature as ``convert_image.write_ome_tiff`` (it is monkeypatched in
    over it) and the same ``ome_io.ome_metadata``, so only the write differs.
    ``tests/test_ome_io_is_the_only_writer.py`` does not scan ``tests/``.
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
    # ome_io.require_reader calls find_spec("bioio"), which returns the module's
    # __spec__; a bare ModuleType has None there and find_spec raises. A loader-less
    # ModuleSpec satisfies it.
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
    # Decoded pixels, not bytes: the reference is striped, the streamed write tiled.
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

    # Decoded pixels, not bytes: the reference is striped, the streamed write tiled.
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

    They go through the same streamed writer, so their pixels are in scope too.
    Compared decoded, not byte for byte (see the module docstring).
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
# the tiled layout
# ---------------------------------------------------------------------------


def test_the_write_is_tiled(monkeypatch, tmp_path):
    """The output is tiled, so a downstream region read decodes tiles, not whole planes.

    A striped TIFF has one zarr chunk per plane, and every region read (BaSiC, STARE,
    SPLIT_CHANNELS, QC) would decode the full plane. Checked two ways:

    1. the TIFF tags (``TileWidth``/``TileLength``) against ``CONVERT_TIFF_TILE``;
    2. the zarr chunk shape ``bin/utils/tiled_io.py:open_lazy`` actually sees.

    The striped reference writer is the contrast.
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
    """A one-chunk stack must not be decoded once per plane.

    dask does not cache across ``compute()`` calls, so a per-plane loop over a single
    whole-stack chunk decodes the full stack N times. Readers such as ``bioio-czi``
    can return that chunking. ``.rechunk()`` does not help (each plane still re-reads
    the source chunk); iterating in chunk-sized slabs does.
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
    """A plane larger than one tile, with partial edge tiles.

    With ``tile=`` set, tifffile's iterator mode is tile-wise and raises
    ``ValueError('tile is too large')`` if fed whole planes; a plane smaller than one
    tile is padded silently, so the other fixtures cannot catch it. A 16px tile over
    24x20 gives 2x2 tiles, three of them short.
    """
    import numpy as np
    import tifffile

    shape = (2, 24, 20)
    data = (np.arange(int(np.prod(shape)), dtype=np.uint16) % 4096).reshape(shape)
    out = tmp_path / "big_plane.ome.tif"
    # tile= as an argument: patching convert_image's imported constant would not reach
    # the default in ome_io.write_ome_tiff's signature.
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
