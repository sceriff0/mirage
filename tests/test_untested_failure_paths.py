"""Six failure paths, each covered directly against the real production function whose
`raise` is under test.

  - ``bin/utils/tiled_io.py:28``  (``nuclear_channel``)   -- out-of-range ``--nuclear-index``
  - ``bin/utils/tiled_io.py:175`` (``read_decimated``)     -- out-of-range ``--nuclear-index``
  - ``bin/quantify.py:415``  (``run_quantification``) -- missing channel image
  - ``bin/quantify.py:438``  (``run_quantification``) -- missing nuclei mask
  - ``bin/utils/image_utils.py:115`` (``normalize_image_dimensions``) -- 4-D array
  - ``bin/utils/fov_tiling.py:254`` (``split_image_into_fovs``)       -- 3-D array

The two ``tiled_io.py`` cases are separate tests because the same check is copied into
two functions (``nuclear_channel`` and ``read_decimated``); each copy needs its own
coverage. A third copy in ``bin/tiled_reg_tile.py`` is covered by
``tests/test_tiled_reg_tile_lazy.py::test_nuclear_index_out_of_range_raises_same_message``.
"""

from __future__ import annotations

import os
import re
import sys

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

pytest.importorskip("numpy")
pytest.importorskip("tifffile")
import quantify as q  # noqa: E402
import tifffile  # noqa: E402
from fov_tiling import split_image_into_fovs  # noqa: E402
from image_utils import normalize_image_dimensions  # noqa: E402
from tiled_io import nuclear_channel, read_decimated  # noqa: E402


# ---------------------------------------------------------------------------
# bin/utils/tiled_io.py:28 -- nuclear_channel's own copy of the range check.
# ---------------------------------------------------------------------------
def test_tiled_io_nuclear_channel_rejects_out_of_range_index():
    arr_chw = np.zeros((2, 8, 8), dtype=np.uint16)
    with pytest.raises(ValueError, match=r"--nuclear-index 5 out of range for C=2"):
        nuclear_channel(arr_chw, 5)


# ---------------------------------------------------------------------------
# bin/utils/tiled_io.py:175 -- read_decimated's separate copy of the same check, with
# its own callers (``bin/utils/qc.py``, ``bin/generate_preprocess_qc.py``).
# ---------------------------------------------------------------------------
class _FakeLazyView:
    """Stand-in for open_lazy's (C, H, W) view. read_decimated reads ``.shape`` before
    it ever indexes into the array, so the out-of-range branch needs nothing else."""

    def __init__(self, shape):
        self.shape = shape


def test_tiled_io_read_decimated_rejects_out_of_range_index():
    src = _FakeLazyView((2, 8, 8))
    with pytest.raises(ValueError, match=r"--nuclear-index 5 out of range for C=2"):
        read_decimated(src, 5, factor=1)


# ---------------------------------------------------------------------------
# bin/quantify.py:415 -- run_quantification, missing channel image.
# ---------------------------------------------------------------------------
def _write_mask_npy(path, shape=(8, 8)):
    mask = np.zeros(shape, dtype=np.uint16)
    mask[2:6, 2:6] = 1
    np.save(path, mask)
    return path


def _write_channel_tif(path, shape=(8, 8)):
    tifffile.imwrite(str(path), np.zeros(shape, dtype=np.uint16))
    return path


def test_quantify_missing_channel_image_raises_filenotfounderror(tmp_path):
    mask_path = tmp_path / "mask.npy"
    _write_mask_npy(mask_path)
    missing_channel = tmp_path / "does_not_exist_channel.tif"

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"Channel image not found: {missing_channel}"),
    ):
        q.run_quantification(
            mask_path=str(mask_path),
            channel_path=str(missing_channel),
            output_path=str(tmp_path / "out.csv"),
            channel_name="DAPI",
        )


# ---------------------------------------------------------------------------
# bin/quantify.py:438 -- run_quantification, missing nuclei mask.
# ---------------------------------------------------------------------------
def test_quantify_missing_nuclei_mask_raises_filenotfounderror(tmp_path):
    mask_path = tmp_path / "mask.npy"
    _write_mask_npy(mask_path)
    channel_path = tmp_path / "channel.tif"
    _write_channel_tif(channel_path)
    missing_nuclei = tmp_path / "does_not_exist_nuclei.npy"

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"Nuclei mask not found: {missing_nuclei}"),
    ):
        q.run_quantification(
            mask_path=str(mask_path),
            channel_path=str(channel_path),
            output_path=str(tmp_path / "out.csv"),
            channel_name="DAPI",
            nuclei_mask_path=str(missing_nuclei),
        )


# ---------------------------------------------------------------------------
# bin/utils/image_utils.py:115 -- normalize_image_dimensions, a 4-D array.
# ---------------------------------------------------------------------------
def test_image_utils_rejects_4d_array():
    arr = np.zeros((2, 3, 4, 5))
    with pytest.raises(
        ValueError, match=r"Image must be 2D or 3D, got 4D with shape \(2, 3, 4, 5\)"
    ):
        normalize_image_dimensions(arr)


# ---------------------------------------------------------------------------
# bin/utils/fov_tiling.py:254 -- split_image_into_fovs, a 3-D array where 2-D is
# required.
# ---------------------------------------------------------------------------
def test_fov_tiling_rejects_3d_array():
    arr = np.zeros((3, 8, 8))
    with pytest.raises(ValueError, match=r"Image must be 2D, got shape \(3, 8, 8\)"):
        split_image_into_fovs(arr, n_fovs_x=2, n_fovs_y=2)
