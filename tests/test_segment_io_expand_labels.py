"""bin/utils/segment_io.py:expand_labels_tiled -- one implementation, and covered.

The function existed TWICE, verbatim apart from formatting: bin/segment.py:200-244
(StarDist) and bin/segment_cellsam.py:99-142 (CellSAM), the latter's docstring
saying outright "Identical to segment.py:expand_labels_tiled". Neither copy was
covered by any test, because segment.py imports stardist/csbdeep at module scope
and is therefore not importable in CI at all -- so the two could drift with
nothing to notice.

Its neighbour extract_dapi_channel had already been extracted here for exactly
that reason (see this module's docstring). This is the same extraction for the
same reason, and the tests below are the coverage neither copy ever had.

THE PROPERTY THAT MATTERS is the one the docstring claims and nothing checked:
the tiled expansion must produce IDENTICAL output to a full-image
skimage.segmentation.expand_labels. A map_overlap whose depth is smaller than the
expansion distance produces a seam at every tile boundary -- correct-looking
labels, wrong where it counts, and invisible on any fixture smaller than one tile.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
    ),
)
# "skimage", not "skimage.segmentation": the requirements-guard scan
# (tests/test_ci_stack_pinned.py) matches importorskip names against DISTRIBUTION names
# via an exact-string lookup, and a submodule-qualified name matches nothing a
# requirements file declares -- silently deselecting this whole file in CI. dask needs no
# importorskip at all: it is installed UNCONDITIONALLY there (see
# tests/test_convert_lazy_read.py's docstring), and segment_io.py already imports it at
# module scope, so its absence would be a loud ImportError, not a silent skip.
pytest.importorskip("skimage")

from segment_io import expand_labels_tiled  # noqa: E402
from skimage import segmentation  # noqa: E402


def _labels(h=64, w=64, n=7, seed=3):
    """A label image with `n` well-separated seeds, background 0."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), dtype=np.uint32)
    for label in range(1, n + 1):
        y = int(rng.integers(3, h - 3))
        x = int(rng.integers(3, w - 3))
        out[y, x] = label
    return out


@pytest.mark.parametrize("distance", [1, 2, 5])
def test_tiled_expansion_equals_full_image_expansion(distance):
    """The claim the docstring made and nothing checked."""
    labels = _labels()
    expected = segmentation.expand_labels(labels, distance=distance)
    got = expand_labels_tiled(labels, distance=distance, tile_size=16)
    np.testing.assert_array_equal(got, expected)


def test_a_tile_smaller_than_the_overlap_falls_back_to_the_full_image():
    """`tile_size <= 2 * (distance + 1)` cannot carry the overlap map_overlap needs,
    so the function must fall back rather than emit a seamed result."""
    labels = _labels(h=24, w=24)
    got = expand_labels_tiled(labels, distance=5, tile_size=8)
    np.testing.assert_array_equal(got, segmentation.expand_labels(labels, distance=5))


def test_the_result_is_uint32():
    """Label ids are written into a uint32 mask by every backend; a narrower dtype
    silently wraps once a slide has more than 65535 cells, which real ones do."""
    assert expand_labels_tiled(_labels(), distance=2, tile_size=16).dtype == np.uint32


def test_the_shape_is_preserved():
    labels = _labels(h=37, w=53)  # not a tile multiple, on purpose
    assert expand_labels_tiled(labels, distance=2, tile_size=16).shape == labels.shape


def test_background_stays_background_where_nothing_reaches_it():
    """A pixel further than `distance` from every seed must stay 0. Without this,
    an expansion that flooded the whole image would pass every test above."""
    labels = np.zeros((32, 32), dtype=np.uint32)
    labels[16, 16] = 1
    got = expand_labels_tiled(labels, distance=2, tile_size=16)
    assert got[0, 0] == 0
    assert got[16, 16] == 1
    assert got[16, 18] == 1


def test_only_one_implementation_survives():
    """The extraction is only worth anything if the copies are gone. A second
    definition anywhere under bin/ is the drift this move exists to end."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "bin"
    definers = sorted(
        p.relative_to(root.parent).as_posix()
        for p in root.rglob("*.py")
        if "def expand_labels_tiled" in p.read_text()
    )
    assert definers == ["bin/utils/segment_io.py"], (
        f"expand_labels_tiled is defined in {definers}; it must be defined once, in "
        "bin/utils/segment_io.py, and imported by both segmentation backends"
    )


def test_labels_beyond_uint16_survive_the_tiled_expansion():
    """A checkerboard of 80 000 one-pixel labels: more than uint16 can hold. The
    expansion fills the gaps and must keep every label id and the uint32 dtype."""
    side = 520  # a 260 x 260 lattice: 67 600 labels, above 65 535
    half = side // 2
    labels = np.zeros((side, side), dtype=np.uint32)
    labels[::2, ::2] = np.arange(1, half * half + 1, dtype=np.uint32).reshape(
        half, half
    )
    n = int(labels.max())
    assert n > 65_535, (
        "the fixture must cross the uint16 boundary or the test proves nothing"
    )
    out = expand_labels_tiled(labels, distance=1, tile_size=128)
    assert out.dtype == np.uint32
    assert int(out.max()) == n
    assert set(np.unique(out)) - {0} == set(np.unique(labels)) - {0}
