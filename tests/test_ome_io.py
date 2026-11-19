"""Unit tests for bin/utils/ome_io.py -- the pipeline's one image I/O module.

Both bin and bin/utils go on sys.path: bin/utils for the flat sibling imports
ome_io itself performs, matching tests/test_tiled_io.py.

`bioio`, `bioio-bioformats` and `h5py` are not installed in CI, so every test
here runs without them; ome_io imports all three lazily, inside functions.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
    ),
)
tifffile = pytest.importorskip("tifffile")

import ome_io  # noqa: E402

# ---------------------------------------------------------------------------
# detect_reader -- the format dispatch table
# ---------------------------------------------------------------------------

BIOIO = [".ome.tif", ".ome.tiff", ".tif", ".tiff", ".nd2", ".czi", ".lif"]
TIFFFILE = [".ndpi", ".ndpis"]
HDF5 = [".h5", ".hdf5"]
BIOFORMATS = [".svs", ".qptiff", ".vsi", ".scn", ".mrxs", ".bif", ".ims"]


@pytest.mark.parametrize("suffix", BIOIO)
def test_bioio_native_suffixes(suffix):
    assert ome_io.detect_reader(f"/x/slide{suffix}") == "bioio"


@pytest.mark.parametrize("suffix", TIFFFILE)
def test_hamamatsu_suffixes_go_to_tifffile(suffix):
    assert ome_io.detect_reader(f"/x/slide{suffix}") == "tifffile"


@pytest.mark.parametrize("suffix", HDF5)
def test_hdf5_suffixes(suffix):
    assert ome_io.detect_reader(f"/x/slide{suffix}") == "hdf5"


@pytest.mark.parametrize("suffix", BIOFORMATS)
def test_bioformats_only_suffixes(suffix):
    assert ome_io.detect_reader(f"/x/slide{suffix}") == "bioio-bioformats"


def test_every_answer_is_one_of_READERS():
    for suffix in BIOIO + TIFFFILE + HDF5 + BIOFORMATS:
        assert ome_io.detect_reader(f"/x/s{suffix}") in ome_io.READERS


def test_the_dispatch_is_case_insensitive():
    assert ome_io.detect_reader("/x/SLIDE.SVS") == "bioio-bioformats"
    assert ome_io.detect_reader("/x/SLIDE.OME.TIFF") == "bioio"


def test_an_unclaimed_suffix_raises_and_names_it():
    """An unknown suffix fails here, naming it, rather than being handed to
    BioImage and failing several frames deep."""
    with pytest.raises(ome_io.UnsupportedFormatError) as exc:
        ome_io.detect_reader("/x/thumbnail.png")
    message = str(exc.value)
    assert ".png" in message, "the message must name the suffix that was refused"
    assert ".czi" in message, "the message must list what IS supported"


def test_a_file_with_no_suffix_raises():
    with pytest.raises(ome_io.UnsupportedFormatError):
        ome_io.detect_reader("/x/slide")


def test_UnsupportedFormatError_is_a_ValueError():
    """Callers that already catch ValueError around a read keep working."""
    assert issubclass(ome_io.UnsupportedFormatError, ValueError)


# ---------------------------------------------------------------------------
# require_reader -- the plugin precondition
# ---------------------------------------------------------------------------


def test_require_reader_accepts_a_reader_whose_modules_are_installed():
    ome_io.require_reader("tifffile")  # must not raise


def test_require_reader_names_the_missing_plugin():
    """`.svs` routes to bioio-bioformats, which only containers/convert installs.
    The failure has to say the plugin's name, not
    `ModuleNotFoundError: bioio_bioformats` several frames deep."""
    if ome_io._module_is_importable("bioio_bioformats"):
        pytest.skip("bioio-bioformats is installed here; nothing to assert")
    with pytest.raises(ImportError) as exc:
        ome_io.require_reader("bioio-bioformats")
    assert "bioio-bioformats" in str(exc.value)


def test_require_reader_rejects_an_unknown_reader_name():
    with pytest.raises(ome_io.UnsupportedFormatError):
        ome_io.require_reader("openslide")


# ---------------------------------------------------------------------------
# ome_metadata -- the shared OME metadata dict
# ---------------------------------------------------------------------------


def test_ome_metadata_key_order_is_the_shape_tifffile_serialises():
    """Key ORDER is asserted, not just contents. tifffile builds the OME-XML by
    walking this dict, so a reordering changes the attribute order in the header
    of every file this pipeline writes, a diff in a published artifact for no
    behavioural reason."""
    md = ome_io.ome_metadata(["DAPI", "CD3"], 0.325)
    assert list(md) == [
        "axes",
        "Channel",
        "PhysicalSizeX",
        "PhysicalSizeXUnit",
        "PhysicalSizeY",
        "PhysicalSizeYUnit",
    ]
    assert md["axes"] == "CYX"
    assert md["Channel"] == {"Name": ["DAPI", "CD3"]}
    assert md["PhysicalSizeX"] == 0.325
    assert md["PhysicalSizeXUnit"] == "µm"
    assert md["PhysicalSizeY"] == 0.325


def test_ome_metadata_accepts_separate_x_and_y():
    md = ome_io.ome_metadata(["DAPI"], (0.5, 0.25))
    assert md["PhysicalSizeX"] == 0.5
    assert md["PhysicalSizeY"] == 0.25


def test_ome_metadata_omits_physical_size_when_there_is_none():
    """None means "the file did not say". Writing a placeholder number would be a
    scale the pipeline invented -- bin/utils/pixel_size.py exists to stop exactly
    that."""
    md = ome_io.ome_metadata(["DAPI"], None)
    assert "PhysicalSizeX" not in md
    assert "PhysicalSizeXUnit" not in md
    assert md["Channel"] == {"Name": ["DAPI"]}


def test_ome_metadata_omits_channel_names_when_there_are_none():
    md = ome_io.ome_metadata(None, 0.325)
    assert "Channel" not in md
    assert md["PhysicalSizeX"] == 0.325


def test_ome_metadata_carries_z_when_asked():
    md = ome_io.ome_metadata(["DAPI"], 0.325, axes="CZYX", pixel_size_z_um=1.5)
    assert md["axes"] == "CZYX"
    assert md["PhysicalSizeZ"] == 1.5
    assert md["PhysicalSizeZUnit"] == "µm"


# ---------------------------------------------------------------------------
# write_ome_tiff -- the round trip
# ---------------------------------------------------------------------------


def _stack(c=3, h=24, w=20, dtype=np.uint16, seed=11):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4000, size=(c, h, w)).astype(dtype)


def test_write_ome_tiff_round_trips_pixels_channels_scale_and_dtype(tmp_path):
    data = _stack()
    out = tmp_path / "round_trip.ome.tif"

    ome_io.write_ome_tiff(
        out, data, channels=["DAPI", "CD3", "CD8"], pixel_size_um=0.325, tile=16
    )

    back = tifffile.imread(out)
    assert back.shape == data.shape
    assert back.dtype == data.dtype
    np.testing.assert_array_equal(back, data)

    info = ome_io.read_info(out) if hasattr(ome_io, "read_info") else None
    with tifffile.TiffFile(out) as tf:
        xml = tf.ome_metadata
    assert xml, "write_ome_tiff must produce an OME header"
    for name in ("DAPI", "CD3", "CD8"):
        assert f'Name="{name}"' in xml
    assert 'PhysicalSizeX="0.325"' in xml
    assert 'PhysicalSizeXUnit="µm"' in xml
    assert info is None or info.channels == ["DAPI", "CD3", "CD8"]


def test_write_ome_tiff_writes_tiled_at_the_requested_tile(tmp_path):
    """A striped write makes every downstream region read decode a whole plane.
    Checked on the TIFF tags and on the zarr chunk shape tiled_io.open_lazy sees,
    since the tags could be present without the zarr view honouring them."""
    zarr = pytest.importorskip("zarr")
    data = _stack(c=2, h=40, w=40)
    out = tmp_path / "tiled.ome.tif"

    ome_io.write_ome_tiff(out, data, channels=["A", "B"], pixel_size_um=0.5, tile=16)

    with tifffile.TiffFile(out) as tf:
        page = tf.pages[0]
        assert page.is_tiled
        assert (page.tilelength, page.tilewidth) == (16, 16)

    store = tifffile.imread(str(out), aszarr=True)
    try:
        grp = zarr.open(store, mode="r")
        arr = grp[0] if isinstance(grp, zarr.Group) else grp
        assert arr.chunks == (1, 16, 16)
    finally:
        store.close()


def test_write_ome_tiff_handles_a_plane_larger_than_one_tile(tmp_path):
    """tifffile's iterator mode is TILE-wise whenever tile= is set: it walks
    numtiles items per page and raises ValueError('tile is too large') the moment
    one is bigger than a tile. A plane-fed writer therefore passes on any fixture
    smaller than one tile and fails on the first real slide. 24x20 over a 16px
    tile is 2x2 tiles, three of them short -- so this also exercises the partial
    edge tiles tifffile pads."""
    data = _stack(c=2, h=24, w=20)
    out = tmp_path / "big_plane.ome.tif"

    ome_io.write_ome_tiff(out, data, channels=["A", "B"], pixel_size_um=None, tile=16)

    np.testing.assert_array_equal(tifffile.imread(out), data)


def test_write_ome_tiff_defaults_bigtiff_off_below_the_threshold(tmp_path):
    """bigtiff=None means "BigTIFF iff nbytes > 2**31". A tiny stack must not
    silently become BigTIFF: a classic TIFF is what every reader accepts."""
    out = tmp_path / "small.ome.tif"
    ome_io.write_ome_tiff(
        out, _stack(), channels=["a", "b", "c"], pixel_size_um=0.325, tile=16
    )
    with tifffile.TiffFile(out) as tf:
        assert tf.is_bigtiff is False


def test_write_ome_tiff_switches_to_bigtiff_above_the_threshold(tmp_path):
    """Asserted on the DECISION function rather than by writing a 2 GB file:
    materialising one in a unit test is not proportionate, and the threshold is
    the part that can regress."""
    assert ome_io._wants_bigtiff(None, 2**31 + 1) is True
    assert ome_io._wants_bigtiff(None, 2**31) is False
    assert ome_io._wants_bigtiff(True, 0) is True
    assert ome_io._wants_bigtiff(False, 2**40) is False


def test_write_ome_tiff_writes_pyramid_levels_as_subifds(tmp_path):
    """pyramid_levels=2 means base + one reduced level, carried as subIFDs of the
    base page (the layout QuPath reads), not as extra top-level series."""
    data = _stack(c=2, h=64, w=64)
    out = tmp_path / "pyramid.ome.tif"

    ome_io.write_ome_tiff(
        out, data, channels=["A", "B"], pixel_size_um=0.325, tile=16, pyramid_levels=2
    )

    with tifffile.TiffFile(out) as tf:
        base = tf.series[0]
        assert len(base.levels) == 2, (
            f"expected base + 1 reduced level, got {len(base.levels)}"
        )
        assert base.levels[1].shape[-2:] == (32, 32)
        np.testing.assert_array_equal(base.levels[0].asarray(), data)


def test_write_ome_tiff_writes_pyramid_levels_on_an_odd_non_square_plane(tmp_path):
    """The reshape arithmetic that derives each reduced level's shape must not
    assume an even, square plane. 37x51 (both odd, both different) forces
    ceil-division: level 1 must be (19, 26), i.e. ceil(37/2) x ceil(51/2)."""
    data = _stack(c=3, h=37, w=51)
    out = tmp_path / "odd_pyramid.ome.tif"

    ome_io.write_ome_tiff(
        out,
        data,
        channels=["A", "B", "C"],
        pixel_size_um=0.25,
        tile=16,
        pyramid_levels=2,
    )

    with tifffile.TiffFile(out) as tf:
        base = tf.series[0]
        assert len(base.levels) == 2
        assert base.levels[1].shape[-2:] == (19, 26)
        np.testing.assert_array_equal(base.levels[0].asarray(), data)
        np.testing.assert_array_equal(base.levels[1].asarray(), data[:, ::2, ::2])


def test_write_ome_tiff_rejects_a_channel_count_that_does_not_match_the_data(tmp_path):
    """Naming three channels on a two-channel stack writes a header that lies, and
    every downstream consumer reads the names positionally."""
    with pytest.raises(ValueError, match="channel"):
        ome_io.write_ome_tiff(
            tmp_path / "bad.ome.tif",
            _stack(c=2),
            channels=["A", "B", "C"],
            pixel_size_um=0.325,
            tile=16,
        )


def test_write_ome_tiff_never_materialises_a_lazy_stack(tmp_path):
    """The property the streaming write exists for. An array-like whose
    __array__ raises stands in for a lazy handle; a writer that still calls
    numpy.asarray on the whole stack trips it."""

    class _RefusesWholeArray:
        def __init__(self, array):
            self._array = array
            self.shape = array.shape
            self.dtype = array.dtype
            self.ndim = array.ndim

        def __getitem__(self, index):
            return self._array[index]

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("the writer materialised the whole stack")

    data = _stack(c=3, h=24, w=20)
    out = tmp_path / "lazy.ome.tif"
    ome_io.write_ome_tiff(
        out,
        _RefusesWholeArray(data),
        channels=["a", "b", "c"],
        pixel_size_um=0.325,
        tile=16,
    )
    np.testing.assert_array_equal(tifffile.imread(out), data)


# ---------------------------------------------------------------------------
# write_tiff -- the non-OME escape hatch
# ---------------------------------------------------------------------------


def test_write_tiff_forwards_its_kwargs_verbatim(tmp_path):
    """write_tiff sets no defaults of its own: call sites pass tifffile.imwrite's
    keyword arguments through unchanged."""
    mask = np.arange(256, dtype=np.uint32).reshape(16, 16)
    out = tmp_path / "mask.tif"

    returned = ome_io.write_tiff(
        out, mask, compression="zlib", bigtiff=True, tile=(16, 16)
    )

    assert returned == out
    np.testing.assert_array_equal(tifffile.imread(out), mask)
    with tifffile.TiffFile(out) as tf:
        assert tf.is_bigtiff
        assert tf.pages[0].is_tiled


def test_write_tiff_does_not_force_bigtiff(tmp_path):
    """An imagej=True write (bin/utils/qc.py's composite) is INCOMPATIBLE with
    BigTIFF (tifffile refuses the combination), so bigtiff=True must not be a
    default."""
    rgb = np.zeros((3, 8, 8), dtype=np.uint8)
    out = tmp_path / "composite.tif"
    ome_io.write_tiff(
        out, rgb, imagej=True, metadata={"axes": "CYX", "mode": "composite"}
    )
    with tifffile.TiffFile(out) as tf:
        assert tf.is_bigtiff is False


# ---------------------------------------------------------------------------
# read_info / read_plane
# ---------------------------------------------------------------------------


def _written(tmp_path, name="src.ome.tif", c=3, h=32, w=24, px=0.325, channels=None):
    data = _stack(c=c, h=h, w=w)
    out = tmp_path / name
    ome_io.write_ome_tiff(
        out,
        data,
        channels=channels if channels is not None else [f"C{i}" for i in range(c)],
        pixel_size_um=px,
        tile=16,
    )
    return out, data


def test_read_info_reports_shape_dtype_channels_scale_and_reader(tmp_path):
    path, data = _written(tmp_path, channels=["DAPI", "CD3", "CD8"])

    info = ome_io.read_info(path)

    assert info.path == path
    assert info.shape_cyx == data.shape
    assert info.dtype == data.dtype
    assert info.channels == ["DAPI", "CD3", "CD8"]
    assert info.pixel_size_um == pytest.approx(0.325)
    assert info.reader in ome_io.READERS


def test_read_info_delegates_the_scale_to_pixel_size_py(tmp_path, monkeypatch):
    """bin/utils/pixel_size.py is the read-side owner of the OME PhysicalSize rule
    -- units, the absent-attribute default, the non-positive rejection. read_info
    must call it, not re-derive it."""
    import pixel_size

    path, _ = _written(tmp_path)
    calls = []
    real = pixel_size.read_ome_pixel_size

    def _spy(p):
        calls.append(str(p))
        return real(p)

    monkeypatch.setattr(ome_io, "read_ome_pixel_size", _spy)
    ome_io.read_info(path)
    assert calls, "read_info must go through pixel_size.read_ome_pixel_size"


def test_read_info_reports_none_for_a_file_that_carries_no_scale(tmp_path):
    """None means "the file did not say" and must stay distinguishable from a
    scale that genuinely equals the configured one."""
    out = tmp_path / "no_scale.ome.tif"
    ome_io.write_ome_tiff(
        out, _stack(c=2), channels=["a", "b"], pixel_size_um=None, tile=16
    )
    assert ome_io.read_info(out).pixel_size_um is None


def test_read_info_promotes_a_two_dimensional_image_to_c_equals_one(tmp_path):
    """Every consumer in this pipeline indexes (C, Y, X). A 2-D file arriving as
    (Y, X) would make `info.shape_cyx[0]` the image HEIGHT."""
    out = tmp_path / "flat.tif"
    ome_io.write_tiff(out, np.zeros((10, 12), dtype=np.uint16))
    assert ome_io.read_info(out).shape_cyx == (1, 10, 12)


def test_read_plane_returns_one_two_dimensional_plane(tmp_path):
    path, data = _written(tmp_path)
    plane = ome_io.read_plane(path, 1)
    assert plane.ndim == 2
    np.testing.assert_array_equal(plane, data[1])


def test_read_plane_lazy_and_eager_agree(tmp_path):
    path, data = _written(tmp_path)
    np.testing.assert_array_equal(
        ome_io.read_plane(path, 2, lazy=True), ome_io.read_plane(path, 2, lazy=False)
    )
    np.testing.assert_array_equal(ome_io.read_plane(path, 2, lazy=False), data[2])


def test_read_plane_goes_through_open_lazy_when_lazy(tmp_path, monkeypatch):
    """tiled_io.open_lazy is the lazy backend behind read_plane -- a tifffile zarr
    view, so only the requested channel's tiles are decoded. A read_plane that
    quietly decoded the whole stack would be indistinguishable on a fixture and
    catastrophic on a 20 GB slide."""
    import tiled_io

    path, _ = _written(tmp_path)
    calls = []
    real = tiled_io.open_lazy

    def _spy(p):
        calls.append(str(p))
        return real(p)

    monkeypatch.setattr(ome_io, "open_lazy", _spy)
    ome_io.read_plane(path, 0, lazy=True)
    assert calls, "read_plane(lazy=True) must go through tiled_io.open_lazy"


def test_read_plane_rejects_an_out_of_range_channel(tmp_path):
    path, _ = _written(tmp_path, c=2)
    with pytest.raises(ValueError, match="out of range"):
        ome_io.read_plane(path, 5)


def test_read_info_refuses_an_unsupported_extension(tmp_path):
    bad = tmp_path / "thumb.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ome_io.UnsupportedFormatError):
        ome_io.read_info(bad)


# ---------------------------------------------------------------------------
# parse_ndpis -- the Hamamatsu manifest
# ---------------------------------------------------------------------------


def test_parse_ndpis_reads_the_image_entries_in_order(tmp_path):
    (tmp_path / "a-CY5.ndpi").write_bytes(b"")
    (tmp_path / "a-TRITC.ndpi").write_bytes(b"")
    manifest = tmp_path / "a.ndpis"
    manifest.write_text(
        "[NanoZoomer Digital Pathology Image Set]\n"
        "NoImages=2\n"
        "Image0=a-CY5.ndpi\n"
        "Image1=a-TRITC.ndpi\n"
    )

    entries = ome_io.parse_ndpis(manifest)

    assert [p.name for p in entries] == ["a-CY5.ndpi", "a-TRITC.ndpi"]
    assert all(p.parent == tmp_path for p in entries)


def test_parse_ndpis_resolves_entries_against_the_real_directory(tmp_path):
    """The manifest is routinely staged by Nextflow as a SYMLINK into a task
    directory while the .ndpi files it names stay where they were. Resolving the
    manifest first is what makes the sibling lookup find them."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "s-DAPI.ndpi").write_bytes(b"")
    manifest = real / "s.ndpis"
    manifest.write_text("[Set]\nNoImages=1\nImage0=s-DAPI.ndpi\n")

    staged = tmp_path / "staged.ndpis"
    staged.symlink_to(manifest)

    assert ome_io.parse_ndpis(staged) == [real / "s-DAPI.ndpi"]


# ---------------------------------------------------------------------------
# The S->C (samples->channels) remap for interleaved RGB
# ---------------------------------------------------------------------------


class _FakeBioImage:
    """Stands in for ``bioio.BioImage``, exposing exactly the attributes
    ``ome_io``'s bioio path reads (``dims.C``/``.S``/``.order``, ``.shape``, ``.dtype``,
    ``.channel_names``, ``.physical_pixel_sizes.X``, ``.get_image_data``), so the S->C
    remap can be exercised without bioio installed anywhere in this test environment.
    """

    def __init__(self, array, order, channel_names=None, pixel_size_x=None):
        self._array = array
        sizes = dict(zip(order, array.shape))
        self.dims = types.SimpleNamespace(order=order, **sizes)
        self.shape = array.shape
        self.dtype = array.dtype
        self.channel_names = channel_names
        self.physical_pixel_sizes = types.SimpleNamespace(X=pixel_size_x)

    def get_image_data(self, dims_str, **kwargs):
        index = []
        for axis in self.dims.order:
            if axis in kwargs:
                index.append(kwargs[axis])
            elif axis in dims_str:
                index.append(slice(None))
            else:
                index.append(0)
        return self._array[tuple(index)]


def test_read_info_remaps_interleaved_rgb_samples_to_channels(tmp_path, monkeypatch):
    """bioio reports an interleaved RGB TIFF as TCZYXS with C=1, S=3 -- three SAMPLES of
    one channel, not three channels. read_info must remap S onto shape_cyx's C axis
    (shape_cyx=(3, Y, X)), not report (1, X, 3), and
    default channel names to R/G/B (S==3) when the file names none."""
    rng = np.random.default_rng(3)
    array = rng.integers(0, 255, size=(1, 1, 1, 5, 7, 3)).astype(np.uint8)
    fake = _FakeBioImage(array, order="TCZYXS")
    monkeypatch.setattr(ome_io, "_open_bioio", lambda path, reader: fake)

    info = ome_io.read_info(tmp_path / "rgb.svs")

    assert info.shape_cyx == (3, 5, 7)
    assert info.channels == ["R", "G", "B"]
    assert info.channels_are_samples is True


def test_read_info_names_non_rgb_sample_counts_S0_S1(tmp_path, monkeypatch):
    """S!=3 has no R/G/B convention to fall back on."""
    rng = np.random.default_rng(7)
    array = rng.integers(0, 255, size=(1, 1, 5, 7, 4)).astype(np.uint8)
    fake = _FakeBioImage(array, order="CZYXS")
    monkeypatch.setattr(ome_io, "_open_bioio", lambda path, reader: fake)

    info = ome_io.read_info(tmp_path / "cmyk.svs")

    assert info.shape_cyx == (4, 5, 7)
    assert info.channels == ["S0", "S1", "S2", "S3"]
    assert info.channels_are_samples is True


def test_read_plane_selects_the_sample_for_an_interleaved_rgb_file(
    tmp_path, monkeypatch
):
    """A caller must never see the remap: read_plane(path, 2) on an S=3 file returns
    the third SAMPLE, indexed on the S axis, not a (nonexistent) third channel."""
    rng = np.random.default_rng(4)
    array = rng.integers(0, 255, size=(1, 1, 1, 5, 7, 3)).astype(np.uint8)
    fake = _FakeBioImage(array, order="TCZYXS")
    monkeypatch.setattr(ome_io, "_open_bioio", lambda path, reader: fake)

    plane = ome_io.read_plane(tmp_path / "rgb.svs", 2)

    assert plane.shape == (5, 7)
    np.testing.assert_array_equal(plane, array[0, 0, 0, :, :, 2])


def test_read_info_leaves_an_ordinary_multichannel_file_unremapped(
    tmp_path, monkeypatch
):
    """C=4, S=1 (S absent from the axis order) is an ordinary multichannel file -- no
    samples axis to remap, so shape_cyx/channels/channels_are_samples are unchanged."""
    rng = np.random.default_rng(5)
    array = rng.integers(0, 4000, size=(4, 1, 1, 6, 9)).astype(np.uint16)
    fake = _FakeBioImage(
        array, order="CZTYX", channel_names=["DAPI", "CD3", "CD8", "CD20"]
    )
    monkeypatch.setattr(ome_io, "_open_bioio", lambda path, reader: fake)

    info = ome_io.read_info(tmp_path / "slide.czi")

    assert info.shape_cyx == (4, 6, 9)
    assert info.channels == ["DAPI", "CD3", "CD8", "CD20"]
    assert info.channels_are_samples is False


def test_read_plane_still_selects_by_channel_when_there_is_no_samples_axis(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(6)
    array = rng.integers(0, 4000, size=(4, 1, 1, 6, 9)).astype(np.uint16)
    fake = _FakeBioImage(array, order="CZTYX")
    monkeypatch.setattr(ome_io, "_open_bioio", lambda path, reader: fake)

    plane = ome_io.read_plane(tmp_path / "slide.czi", 1)

    assert plane.shape == (6, 9)
    np.testing.assert_array_equal(plane, array[1, 0, 0, :, :])


# ---------------------------------------------------------------------------
# The pyramid chain, exercised three levels deep
# ---------------------------------------------------------------------------


def test_write_ome_tiff_chains_pyramid_levels_forward_not_from_the_base(tmp_path):
    """Level 2 must derive from level 1, not be recomputed from the base each time.
    A loop that forgets to reassign `current` (or resets it to the base every
    iteration) is invisible at pyramid_levels=2 -- level 1 looks identical either way
    -- and only shows up chained three deep: level 2 would then have level 1's SHAPE
    (not further halved) and level 1's CONTENT, instead of being level 1's own
    downsampling."""
    data = _stack(c=2, h=37, w=51)
    out = tmp_path / "chain_pyramid.ome.tif"

    ome_io.write_ome_tiff(
        out, data, channels=["A", "B"], pixel_size_um=0.2, tile=16, pyramid_levels=3
    )

    with tifffile.TiffFile(out) as tf:
        base = tf.series[0]
        assert len(base.levels) == 3
        level1 = base.levels[1].asarray()
        level2 = base.levels[2].asarray()

    assert level1.shape[-2:] == (19, 26)  # ceil(37/2), ceil(51/2)
    assert level2.shape[-2:] == (10, 13)  # ceil(19/2), ceil(26/2) -- NOT level1's shape
    np.testing.assert_array_equal(level1, data[:, ::2, ::2])
    np.testing.assert_array_equal(level2, level1[:, ::2, ::2])
