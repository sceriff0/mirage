"""`ome_io.detect_reader` is the whole format claim, in one function.

An unknown suffix must be refused here with a message naming the supported
formats, not fall through to BioImage and fail with a plugin-installation error.

This file needs no image reader at all -- detect_reader is a suffix table --
which is why it lives in the main suite while every other format test lives in
tests/integration/formats/ behind the convert image's reader stack.

Import convention: `ome_io` is a bin/utils module and imports its siblings
(`jvm_cache`, `pixel_size`) with a flat `from jvm_cache import ...`, so both
`bin` and `bin/utils` must be on `sys.path` -- matching tests/test_ome_io.py
and tests/test_tiled_io.py, not a plain `from utils import ome_io`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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

import ome_io  # noqa: E402


@pytest.mark.parametrize(
    "name,expected",
    [
        ("slide.ome.tif", "bioio"),
        ("slide.ome.tiff", "bioio"),
        ("slide.tif", "bioio"),
        ("slide.tiff", "bioio"),
        ("slide.nd2", "bioio"),
        ("slide.czi", "bioio"),
        ("slide.lif", "bioio"),
        ("slide.ndpi", "tifffile"),
        ("slide.ndpis", "tifffile"),
        ("slide.h5", "hdf5"),
        ("slide.hdf5", "hdf5"),
        ("slide.svs", "bioio-bioformats"),
        ("slide.qptiff", "bioio-bioformats"),
        ("slide.vsi", "bioio-bioformats"),
        ("slide.scn", "bioio-bioformats"),
        ("slide.mrxs", "bioio-bioformats"),
        ("slide.bif", "bioio-bioformats"),
        ("slide.ims", "bioio-bioformats"),
    ],
)
def test_every_claimed_suffix_routes_to_a_named_reader(name, expected):
    assert ome_io.detect_reader(Path(name)) == expected


def test_every_reader_the_table_names_is_in_READERS():
    """A typo'd reader name would route a real format to a branch that does not
    exist, failing as an AttributeError deep in read_info."""
    routed = {
        ome_io.detect_reader(Path(f"slide{suffix}"))
        for suffix in (
            ".ome.tif",
            ".tif",
            ".nd2",
            ".czi",
            ".lif",
            ".ndpi",
            ".ndpis",
            ".h5",
            ".hdf5",
            ".svs",
            ".qptiff",
            ".vsi",
            ".scn",
            ".mrxs",
            ".bif",
            ".ims",
        )
    }
    assert routed <= set(ome_io.READERS)
    # Non-vacuity: all four readers must actually be reachable from the table.
    assert routed == set(ome_io.READERS)


@pytest.mark.parametrize("name", ["shot.png", "shot.jpg", "store.zarr", "notes.txt"])
def test_an_unclaimed_suffix_raises_and_says_what_is_supported(name):
    with pytest.raises(ome_io.UnsupportedFormatError) as exc:
        ome_io.detect_reader(Path(name))
    message = str(exc.value)
    suffix = Path(name).suffix
    assert suffix in message, (
        f"the error does not name the offending suffix {suffix!r}: {message!r}"
    )
    # The operator must be able to act on it: at least the four vendor suffixes
    # the pipeline does claim have to appear.
    for claimed in (".ome.tif", ".czi", ".ndpi", ".h5"):
        assert claimed in message, (
            f"the error does not list {claimed} as supported: {message!r}"
        )


def test_case_is_not_significant():
    """Vendor exports are routinely upper-cased (.CZI off a Zeiss share, .TIF off
    a scanner). A case-sensitive table rejects a file the pipeline can read."""
    assert ome_io.detect_reader(Path("SLIDE.CZI")) == "bioio"
    assert ome_io.detect_reader(Path("SLIDE.OME.TIFF")) == "bioio"
