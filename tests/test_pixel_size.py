"""Tests for bin/utils/pixel_size.py — the one pixel-size rule.

`params.pixel_size` owns every µm conversion in the pipeline. These tests cover reading
the scale an image claims, and warning when it disagrees with the configured one.
Neither changes a number: the configured value stays authoritative.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pytest
import tifffile

_SPEC = importlib.util.spec_from_file_location(
    "pixel_size",
    Path(__file__).resolve().parent.parent / "bin" / "utils" / "pixel_size.py",
)
ps = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ps)


def _write_ome(path: Path, **kwargs) -> Path:
    tifffile.imwrite(
        path, np.zeros((4, 4), dtype=np.uint16), metadata={"axes": "YX", **kwargs}
    )
    return path


# ── read_ome_pixel_size ────────────────────────────────────────────────────────
def test_reads_physical_size_in_um(tmp_path):
    p = _write_ome(tmp_path / "a.ome.tif", PhysicalSizeX=0.2125, PhysicalSizeY=0.2125)
    assert ps.read_ome_pixel_size(p) == pytest.approx((0.2125, 0.2125), rel=1e-6)


def test_unit_is_honoured_not_assumed(tmp_path):
    """A value without its unit is a number, not a scale.

    OME writers do use nm; reading 212.5 nm as 212.5 µm is a 1000x error that still
    looks like a plausible overview pixel size.
    """
    p = _write_ome(
        tmp_path / "nm.ome.tif",
        PhysicalSizeX=212.5,
        PhysicalSizeXUnit="nm",
        PhysicalSizeY=212.5,
        PhysicalSizeYUnit="nm",
    )
    assert ps.read_ome_pixel_size(p) == pytest.approx((0.2125, 0.2125), rel=1e-6)


def test_unrecognised_unit_reads_as_absent(tmp_path):
    p = _write_ome(
        tmp_path / "weird.ome.tif", PhysicalSizeX=1.0, PhysicalSizeXUnit="parsec"
    )
    assert ps.read_ome_pixel_size(p)[0] is None


def test_no_ome_header_is_not_an_error(tmp_path):
    """The normal case for this pipeline's own intermediates."""
    p = tmp_path / "plain.tiff"
    tifffile.imwrite(p, np.zeros((4, 4), dtype=np.uint16))
    assert ps.read_ome_pixel_size(p) == (None, None)


def test_unreadable_file_is_not_an_error(tmp_path):
    p = tmp_path / "not-a-tiff.tiff"
    p.write_bytes(b"definitely not a tiff")
    assert ps.read_ome_pixel_size(p) == (None, None)


def test_nonsense_values_read_as_absent(tmp_path):
    p = _write_ome(tmp_path / "zero.ome.tif", PhysicalSizeX=0.0)
    assert ps.read_ome_pixel_size(p)[0] is None


# ── warn_on_pixel_size_mismatch ────────────────────────────────────────────────
def test_disagreement_warns_and_names_both_numbers(caplog):
    with caplog.at_level(logging.WARNING):
        fired = ps.warn_on_pixel_size_mismatch(
            (0.2125, 0.2125), 0.325, source="P001.ome.tiff", logger=logging.getLogger()
        )
    assert fired
    text = caplog.text
    assert "SCALE MISMATCH" in text and "P001.ome.tiff" in text
    assert "0.2125" in text and "0.325" in text


def test_float_noise_is_not_a_disagreement(caplog):
    """OME headers routinely serialise 0.325 as 0.32499998807907104."""
    with caplog.at_level(logging.WARNING):
        fired = ps.warn_on_pixel_size_mismatch(
            0.32499998807907104, 0.325, source="x", logger=logging.getLogger()
        )
    assert not fired
    assert "SCALE MISMATCH" not in caplog.text


def test_absent_scale_is_not_a_disagreement(caplog):
    """Warning here would train operators to ignore the warning that matters."""
    with caplog.at_level(logging.WARNING):
        fired = ps.warn_on_pixel_size_mismatch(
            (None, None), 0.325, source="x", logger=logging.getLogger()
        )
    assert not fired
    assert caplog.text == ""


def test_one_axis_disagreeing_is_enough(caplog):
    with caplog.at_level(logging.WARNING):
        fired = ps.warn_on_pixel_size_mismatch(
            (0.325, 0.65), 0.325, source="x", logger=logging.getLogger()
        )
    assert fired
    assert caplog.text.count("SCALE MISMATCH") == 1  # only the Y axis


def test_never_returns_a_scale_to_use():
    """The signature is the guard: a caller cannot accidentally prefer the metadata.

    This keeps `params.pixel_size` the single owner; a function returning "the resolved
    pixel size" would invite drift.
    """
    assert ps.warn_on_pixel_size_mismatch(
        0.1, 0.325, source="x", logger=logging.getLogger()
    ) in (True, False)
