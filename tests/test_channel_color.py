"""generate_channel_color must not resolve substring collisions.

A substring match would let "CD4" take CD45's color and "CD1" take CD14's; only an
exact (case-insensitive) match against the marker color map counts.
"""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "merge_channels_pyramid",
    Path(__file__).resolve().parent.parent / "bin" / "merge_channels_pyramid.py",
)
mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mcp)


def test_cd4_and_cd45_get_distinct_colors():
    # "CD4" is a substring of "CD45".
    assert mcp.generate_channel_color("CD4", 0) != mcp.generate_channel_color("CD45", 1)


def test_cd1_and_cd14_get_distinct_colors():
    # "CD1" is a substring of "CD14".
    assert mcp.generate_channel_color("CD1", 0) != mcp.generate_channel_color("CD14", 1)


def test_known_key_maps_to_its_predefined_color():
    assert mcp.generate_channel_color("CD45", 0) == mcp.MARKER_COLORS["CD45"]
    assert mcp.generate_channel_color("cd45", 0) == mcp.MARKER_COLORS["CD45"]


def test_unknown_channel_falls_back_to_generated_color():
    # Unknown names should not raise and should not collide with a predefined key.
    color = mcp.generate_channel_color("SomeUnknownMarker", 5)
    assert isinstance(color, tuple) and len(color) == 3
