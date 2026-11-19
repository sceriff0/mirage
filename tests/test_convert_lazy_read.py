"""``bin/convert_image.py:read_image_bioio`` must stop materialising the whole slide.

``img.data`` decodes the whole slide and the squeeze/transpose makes a second copy,
so peak is about twice the decompressed slide, while the task's memory is sized from
the compressed input. ``BioImage.dask_data`` has the same shape and dimension order.
This file pins the read half:

1. ``read_image_bioio`` reads ``img.dask_data``. ``img.data`` is never touched -- the
   fake below raises if it is, so this is a behavioural assertion, not a grep.
2. The returned handle is still lazy when it leaves the function (nothing in
   ``read_image_bioio`` computes it).
3. The ``S``-as-``C`` remapping and the singleton-``C`` squeeze -- the branch that makes
   RGB-ish vendor files work -- survive unchanged, *including* their logged messages.
4. ``physical_pixel_sizes`` still passes ``None`` through as ``None``. ``None`` means
   "the file did not say"; the single fallback lives in ``convert_to_ome_tiff`` so that
   a missing scale and a scale that genuinely equals the default stay distinguishable
   (``warn_on_pixel_size_mismatch`` has to tell them apart).
5. ``channel_names_from_file`` and ``original_dims`` are still in the returned metadata.

A fake ``bioio`` module stands in because CI does not install bioio (only
``containers/convert`` does), and an ``importorskip`` would never run in the gate.
``read_image_bioio`` imports ``BioImage`` inside the function, so injecting
``sys.modules["bioio"]`` drives the real function. ``dask`` is installed in CI and
imported unconditionally, so dropping it fails the gate instead of skipping.
"""

from __future__ import annotations

import logging
import sys
import types

import dask.array as da
import numpy as np

from bin import convert_image


class _FakeDims:
    """Stands in for ``bioio.dimensions.Dimensions``: an ``order`` plus per-axis sizes."""

    def __init__(self, order: str, sizes: dict):
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
    """``bioio``'s ``PhysicalPixelSizes`` namedtuple surface: ``.X/.Y/.Z``, any may be None."""

    def __init__(self, x=None, y=None, z=None):
        self.X = x
        self.Y = y
        self.Z = z


class _FakeBioImage:
    """A ``BioImage`` whose eager ``.data`` is a tripwire.

    Reading ``.data`` raises instead of returning pixels, so a converter that still
    materialises the slide fails loudly with the reason rather than passing on a
    small fixture where eager and lazy are indistinguishable.
    """

    def __init__(self, array, order, sizes, channel_names, pixel_sizes, chunks=None):
        self._array = array
        self._chunks = chunks if chunks is not None else array.shape
        self.dims = _FakeDims(order, sizes)
        self.shape = array.shape
        self.channel_names = channel_names
        self.physical_pixel_sizes = pixel_sizes

    @property
    def data(self):
        raise AssertionError(
            "BioImage.data materialises the entire slide; the converter must read dask_data"
        )

    @property
    def dask_data(self):
        return da.from_array(self._array, chunks=self._chunks)


def _install_fake_bioio(monkeypatch, img):
    module = types.ModuleType("bioio")
    module.BioImage = lambda _path: img
    monkeypatch.setitem(sys.modules, "bioio", module)


def _planar(seed=3, shape=(1, 3, 1, 12, 10)):
    """A ``TCZYX`` stack with per-channel-distinct content."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4000, size=shape, dtype=np.uint16)


def _sample_last(seed=5, shape=(1, 1, 1, 12, 10, 3)):
    """A ``TCZYXS`` stack: singleton ``C``, three ``S`` samples -- the RGB-ish vendor case."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=shape, dtype=np.uint16)


def _planar_image(array):
    return _FakeBioImage(
        array,
        order="TCZYX",
        sizes={
            "T": array.shape[0],
            "C": array.shape[1],
            "Z": array.shape[2],
            "Y": array.shape[3],
            "X": array.shape[4],
        },
        channel_names=["DAPI", "CD3", "CD8"],
        pixel_sizes=_FakePixelSizes(0.5, 0.5, None),
    )


def _s_as_c_image(array):
    return _FakeBioImage(
        array,
        order="TCZYXS",
        sizes={
            "T": array.shape[0],
            "C": array.shape[1],
            "Z": array.shape[2],
            "Y": array.shape[3],
            "X": array.shape[4],
            "S": array.shape[5],
        },
        channel_names=["DAPI"],
        pixel_sizes=_FakePixelSizes(None, None, None),
    )


# ---------------------------------------------------------------------------
# 1 + 2: the read is lazy
# ---------------------------------------------------------------------------


def test_read_image_bioio_never_touches_eager_data(monkeypatch, tmp_path):
    array = _planar()
    _install_fake_bioio(monkeypatch, _planar_image(array))

    data, _metadata = convert_image.read_image_bioio(tmp_path / "slide.czi")

    assert isinstance(data, da.Array), (
        f"read_image_bioio returned {type(data).__name__}, so the whole slide was "
        "materialised before the writer ever saw it"
    )
    np.testing.assert_array_equal(np.asarray(data), array)


def test_the_returned_handle_is_still_unmaterialised(monkeypatch, tmp_path):
    """Laziness that ends at the function boundary saves nothing -- the write half needs it."""
    array = _planar()
    _install_fake_bioio(monkeypatch, _planar_image(array))

    data, _metadata = convert_image.read_image_bioio(tmp_path / "slide.czi")

    assert hasattr(data, "compute") and hasattr(data, "dask"), (
        "the handle must still be a lazy graph when it leaves read_image_bioio"
    )


# ---------------------------------------------------------------------------
# 3: the S-as-C branch, including its messages
# ---------------------------------------------------------------------------


def test_s_as_c_remap_and_singleton_squeeze_are_preserved(
    monkeypatch, tmp_path, caplog
):
    array = _sample_last()
    _install_fake_bioio(monkeypatch, _s_as_c_image(array))

    with caplog.at_level(logging.INFO):
        data, metadata = convert_image.read_image_bioio(tmp_path / "rgbish.tif")

    assert metadata["num_channels"] == 3, "S must be adopted as the channel count"
    assert metadata["original_dims"] == "TZYXC", (
        "the singleton C is squeezed out and S is renamed C, so the order ends in C"
    )
    np.testing.assert_array_equal(np.asarray(data), np.squeeze(array, axis=1))

    text = caplog.text
    assert "Detected 'S' dimension (3) used as channels instead of 'C' (1)" in text
    assert "Squeezed singleton C dimension at position 1" in text
    assert "Remapped dimension order: TZYXC" in text


def test_s_as_c_without_a_singleton_c_keeps_the_full_stack(
    monkeypatch, tmp_path, caplog
):
    """The `else` half of the same branch: no C in the order, so nothing is squeezed."""
    array = _sample_last(shape=(1, 1, 12, 10, 3))
    img = _FakeBioImage(
        array,
        order="TZYXS",
        sizes={"T": 1, "Z": 1, "Y": 12, "X": 10, "S": 3, "C": 1},
        channel_names=["DAPI"],
        pixel_sizes=_FakePixelSizes(None, None, None),
    )
    _install_fake_bioio(monkeypatch, img)

    with caplog.at_level(logging.INFO):
        data, metadata = convert_image.read_image_bioio(tmp_path / "rgbish.tif")

    assert metadata["num_channels"] == 3
    assert metadata["original_dims"] == "TZYXC"
    np.testing.assert_array_equal(np.asarray(data), array)
    assert "Squeezed singleton C dimension" not in caplog.text


def test_a_plain_planar_stack_is_left_alone(monkeypatch, tmp_path):
    array = _planar()
    _install_fake_bioio(monkeypatch, _planar_image(array))

    data, metadata = convert_image.read_image_bioio(tmp_path / "slide.czi")

    assert metadata["original_dims"] == "TCZYX"
    assert metadata["num_channels"] == 3
    np.testing.assert_array_equal(np.asarray(data), array)


# ---------------------------------------------------------------------------
# 4 + 5: the metadata contract
# ---------------------------------------------------------------------------


def test_absent_physical_pixel_sizes_stay_none(monkeypatch, tmp_path):
    """None must not be collapsed into PIXEL_SIZE_UM here; that fallback is downstream."""
    array = _sample_last()
    _install_fake_bioio(monkeypatch, _s_as_c_image(array))

    _data, metadata = convert_image.read_image_bioio(tmp_path / "rgbish.tif")

    assert metadata["physical_pixel_size_x"] is None
    assert metadata["physical_pixel_size_y"] is None
    assert metadata["physical_pixel_size_z"] is None


def test_declared_physical_pixel_sizes_are_passed_through(monkeypatch, tmp_path):
    array = _planar()
    _install_fake_bioio(monkeypatch, _planar_image(array))

    _data, metadata = convert_image.read_image_bioio(tmp_path / "slide.czi")

    assert metadata["physical_pixel_size_x"] == 0.5
    assert metadata["physical_pixel_size_y"] == 0.5
    assert metadata["physical_pixel_size_z"] is None


def test_channel_names_and_original_dims_survive(monkeypatch, tmp_path):
    array = _planar()
    _install_fake_bioio(monkeypatch, _planar_image(array))

    _data, metadata = convert_image.read_image_bioio(tmp_path / "slide.czi")

    assert metadata["channel_names_from_file"] == ["DAPI", "CD3", "CD8"]
    assert metadata["original_dims"] == "TCZYX"
