"""Tests for bin/utils/tile_grid.py — the tiling geometry of the STARE registration method.

A slide is split into core tiles that *partition* it (no gaps, no overlaps) so stitching just
places each core back; each core is read with a halo (clamped to the image) so per-tile
registration has surrounding context. Tile centres are the control points of the mesh field.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
    ),
)

from tile_grid import tile_grid


def test_cores_partition_the_image_with_no_gaps_or_overlaps():
    w, h, tile = 100, 60, 40
    tiles = tile_grid(w, h, tile, halo=8)
    # 3 columns (0-40, 40-80, 80-100), 2 rows (0-40, 40-60)
    assert len(tiles) == 3 * 2

    covered = set()
    for t in tiles:
        x0, y0, x1, y1 = t.core
        assert 0 <= x0 < x1 <= w
        assert 0 <= y0 < y1 <= h
        for x in range(x0, x1):
            for y in range(y0, y1):
                key = (x, y)
                assert key not in covered  # no overlap
                covered.add(key)
    assert len(covered) == w * h  # no gaps: every pixel owned by exactly one core


def test_read_box_is_the_core_expanded_by_halo_and_clamped_to_the_image():
    w, h, tile, halo = 100, 60, 40, 8
    tiles = tile_grid(w, h, tile, halo)

    for t in tiles:
        cx0, cy0, cx1, cy1 = t.core
        rx0, ry0, rx1, ry1 = t.read
        # never leaves the image
        assert 0 <= rx0 and rx1 <= w and 0 <= ry0 and ry1 <= h
        # expands the core by up to `halo`, clamped at the borders
        assert rx0 == max(0, cx0 - halo)
        assert ry0 == max(0, cy0 - halo)
        assert rx1 == min(w, cx1 + halo)
        assert ry1 == min(h, cy1 + halo)
