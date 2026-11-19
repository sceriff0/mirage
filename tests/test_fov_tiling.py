"""The FOV grid arithmetic in ``bin/utils/fov_tiling.py``.

``count_fovs`` is plain ``ceil(size / fov)``, non-overlapping. The nuclear/fiducial
skip decision is tested in ``tests/test_tile_for_basic.py``.
"""

import pytest

from bin.utils.fov_tiling import count_fovs


def test_fov_grid_covers_image():
    """The FOV grid must cover the image, with a partial row/column when it
    does not divide evenly."""
    assert count_fovs((4000, 4000), (1950, 1950)) == (3, 3)
    assert count_fovs((3900, 3900), (1950, 1950)) == (2, 2)  # exact fit, no extra tile
    assert count_fovs((1000, 1000), (1950, 1950)) == (1, 1)  # smaller than one FOV


def test_fov_size_must_be_positive():
    with pytest.raises(ValueError):
        count_fovs((4000, 4000), (0, 1950))
