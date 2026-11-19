"""The bioio-bioformats read path must point scyjava's jgo cache off $HOME BEFORE the
first ``BioImage`` construction that can trigger a Maven resolve.

Both openers that reach ``BioImage()`` must call
``bin/utils/jvm_cache.py::point_jvm_cache_off_readonly_home()`` first:

  * ``bin/utils/ome_io.py::_open_bioio`` -- used by ``read_info``/``read_plane``.
  * ``bin/convert_image.py::read_image`` -> ``read_image_bioio`` -- CONVERT_IMAGE's own,
    separate opener.

``bioio`` is faked via ``sys.modules`` injection (not installed in CI) and the guard
function is replaced with a spy; ``tests/test_jvm_cache_guard.py`` covers the guard's
own behaviour.
"""

from __future__ import annotations

import os
import sys
import types

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
pytest.importorskip("tifffile")

import ome_io  # noqa: E402

from bin import convert_image  # noqa: E402


class _FakeBioImage:
    def __init__(self, *_args, **_kwargs):
        self.dims = types.SimpleNamespace(order="TCZYX", C=1, S=1)
        self.shape = (1, 1, 1, 4, 4)
        self.channel_names = ["DAPI"]
        self.physical_pixel_sizes = types.SimpleNamespace(X=None, Y=None, Z=None)

    @property
    def dask_data(self):
        import numpy as np

        return np.zeros((1, 1, 1, 4, 4), dtype="uint16")


def _install_fake_bioio(monkeypatch):
    module = types.ModuleType("bioio")
    module.BioImage = _FakeBioImage
    monkeypatch.setitem(sys.modules, "bioio", module)


# ---------------------------------------------------------------------------
# ome_io._open_bioio -- the read_info/read_plane opener
# ---------------------------------------------------------------------------


def test_open_bioio_points_the_jvm_cache_before_constructing_bioimage_for_bioformats(
    monkeypatch,
):
    _install_fake_bioio(monkeypatch)
    monkeypatch.setattr(ome_io, "require_reader", lambda reader: None)
    calls = []
    monkeypatch.setattr(
        ome_io, "point_jvm_cache_off_readonly_home", lambda: calls.append("called")
    )

    ome_io._open_bioio(ome_io.Path("/x/slide.svs"), "bioio-bioformats")

    assert calls == ["called"], (
        "_open_bioio must call point_jvm_cache_off_readonly_home() for the "
        "bioio-bioformats route -- without it the first Maven-triggering BioImage() "
        "call hits jgo's os.makedirs($HOME/.jgo) on a read-only $HOME"
    )


def test_open_bioio_skips_the_jvm_cache_guard_for_plain_bioio(monkeypatch):
    """Plain bioio (OME-TIFF/TIFF/ND2/CZI/LIF) starts no JVM, so the guard must not
    fire there -- it would be dead work on every non-Bio-Formats read."""
    _install_fake_bioio(monkeypatch)
    monkeypatch.setattr(ome_io, "require_reader", lambda reader: None)

    def _boom():
        raise AssertionError(
            "point_jvm_cache_off_readonly_home must not run for the plain bioio reader"
        )

    monkeypatch.setattr(ome_io, "point_jvm_cache_off_readonly_home", _boom)

    ome_io._open_bioio(ome_io.Path("/x/slide.czi"), "bioio")


# ---------------------------------------------------------------------------
# convert_image.read_image -- CONVERT_IMAGE's own opener (a SEPARATE call site)
# ---------------------------------------------------------------------------


def test_convert_image_read_image_points_the_jvm_cache_for_the_bioformats_route(
    monkeypatch, tmp_path
):
    _install_fake_bioio(monkeypatch)
    monkeypatch.setattr(convert_image, "require_reader", lambda reader: None)
    calls = []
    monkeypatch.setattr(
        convert_image,
        "point_jvm_cache_off_readonly_home",
        lambda: calls.append("called"),
    )

    convert_image.read_image(tmp_path / "slide.svs")

    assert calls == ["called"], (
        "convert_image.read_image is CONVERT_IMAGE's actual read path and is a "
        "DIFFERENT opener from ome_io._open_bioio -- it must call "
        "point_jvm_cache_off_readonly_home() itself for the .svs/.qptiff/... route, "
        "or the fix in ome_io.py never reaches a real CONVERT_IMAGE run"
    )


def test_convert_image_read_image_skips_the_guard_for_plain_bioio_extensions(
    monkeypatch, tmp_path
):
    _install_fake_bioio(monkeypatch)
    monkeypatch.setattr(convert_image, "require_reader", lambda reader: None)

    def _boom():
        raise AssertionError(
            "point_jvm_cache_off_readonly_home must not run for a plain-bioio "
            "extension such as .czi"
        )

    monkeypatch.setattr(convert_image, "point_jvm_cache_off_readonly_home", _boom)

    convert_image.read_image(tmp_path / "slide.czi")
