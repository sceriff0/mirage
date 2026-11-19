"""The segmentation backends decode only the channel they use.

``tifffile.TiffFile(...).asarray(out="memmap")`` does not memory-map compressed
input: it decodes the whole image into a full-size uncompressed temp file and
maps that, so a 20 GB slide spills ~20 GB of scratch to read one plane.

Both backends (``bin/segment.py``, ``bin/segment_cellsam.py``) delegate their
channel read to ``bin/utils/segment_io.py:extract_dapi_channel``, which reads
through ``tiled_io.open_lazy`` (a tifffile zarr view) so only the requested
channel's tiles are decoded. ``segment_io.py`` imports no ML stack, so these
tests exercise the production read path directly in CI.

``bin/segment_cellsam.py`` imports cellSAM lazily and is tested directly.
Whether ``bin/segment.py`` imports depends on the environment (stardist,
TensorFlow), so its delegation is checked structurally always
(``test_segment_py_delegates_to_segment_io_and_drops_memmap``) and by identity
where the import succeeds (``test_segment_py_delegation_is_live_when_importable``).
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = str(REPO_ROOT / "bin")
UTILS_DIR = str(REPO_ROOT / "bin" / "utils")
for _dir in (BIN_DIR, UTILS_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")

import segment_io  # noqa: E402
from validation import clip_negative_values  # noqa: E402

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _write_3d_stack(
    tmp_path,
    n_channels=3,
    h=48,
    w=40,
    dtype=np.uint16,
    compression=None,
    seed=11,
    name="stack.ome.tiff",
):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        lo = max(info.min, 0)
        stack = rng.integers(lo, info.max, size=(n_channels, h, w), dtype=dtype)
    else:
        stack = (rng.random(size=(n_channels, h, w)).astype(dtype)) * 1000.0

    path = tmp_path / name
    tifffile.imwrite(
        str(path),
        stack,
        ome=True,
        photometric="minisblack",
        compression=compression,
        metadata={
            "axes": "CYX",
            "Channel": {"Name": [f"C{i}" for i in range(n_channels)]},
        },
    )
    return path, stack


def _write_2d_plane(
    tmp_path, h=48, w=40, dtype=np.uint16, compression=None, seed=13, name="plane.tif"
):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        lo = max(info.min, 0)
        plane = rng.integers(lo, info.max, size=(h, w), dtype=dtype)
    else:
        plane = (rng.random(size=(h, w)).astype(dtype)) * 1000.0

    path = tmp_path / name
    tifffile.imwrite(
        str(path), plane, photometric="minisblack", compression=compression
    )
    return path, plane


# ---------------------------------------------------------------------------
# Frozen copies of the memmap-based implementations, used as the independent
# reference for the equivalence tests below.
# ---------------------------------------------------------------------------


def _old_segment_extract_dapi_channel(multichannel_image_path, dapi_channel_index=0):
    """Frozen memmap-based copy of bin/segment.py:extract_dapi_channel."""
    logger = logging.getLogger("old_segment")

    with tifffile.TiffFile(multichannel_image_path) as tif:
        if tif.series:
            source = tif.series[0]
        else:
            if not tif.pages:
                raise ValueError(
                    f"TIFF file appears corrupted: {multichannel_image_path}"
                )
            source = tif.pages[0]

        image_shape = source.shape

        metadata = {}
        if hasattr(tif, "ome_metadata") and tif.ome_metadata:
            metadata["ome"] = tif.ome_metadata

        image_memmap = tif.asarray(out="memmap")

        if image_memmap.ndim == 2:
            dapi_image = np.array(image_memmap, copy=True)
            dapi_image = clip_negative_values(
                dapi_image, logger, stage_name="extract_dapi"
            )
        elif image_memmap.ndim == 3:
            n_channels = image_memmap.shape[0]
            if dapi_channel_index >= n_channels:
                raise ValueError(
                    f"DAPI channel index {dapi_channel_index} out of range "
                    f"for image with {n_channels} channels"
                )
            dapi_image = np.array(image_memmap[dapi_channel_index, :, :], copy=True)
            dapi_image = clip_negative_values(
                dapi_image, logger, stage_name="extract_dapi"
            )
        else:
            raise ValueError(
                f"Unexpected image dimensions: {image_shape}. "
                f"Expected 2D (Y, X) or 3D (C, Y, X)"
            )

    return dapi_image, metadata


def _old_segment_cellsam_extract_dapi_channel(
    multichannel_image_path, dapi_channel_index=0
):
    """Frozen memmap-based copy of bin/segment_cellsam.py:extract_dapi_channel."""
    logger = logging.getLogger("old_segment_cellsam")

    with tifffile.TiffFile(multichannel_image_path) as tif:
        image_memmap = tif.asarray(out="memmap")

        if image_memmap.ndim == 2:
            dapi_image = np.array(image_memmap, copy=True)
        elif image_memmap.ndim == 3:
            n_channels = image_memmap.shape[0]
            if dapi_channel_index >= n_channels:
                raise ValueError(
                    f"DAPI channel index {dapi_channel_index} out of range "
                    f"for image with {n_channels} channels"
                )
            dapi_image = np.array(image_memmap[dapi_channel_index, :, :], copy=True)
        else:
            raise ValueError(
                f"Unexpected image dimensions: {image_memmap.shape}. "
                f"Expected 2D (Y, X) or 3D (C, Y, X)"
            )

    dapi_image = clip_negative_values(dapi_image, logger, stage_name="extract_dapi")
    return dapi_image


# ---------------------------------------------------------------------------
# 1. Behavioral: the lazy path is actually used, not asarray(out="memmap")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_extract_dapi_channel_never_calls_asarray_out_memmap(
    tmp_path, monkeypatch, compression
):
    """segment_io never calls tif.asarray(out="memmap"), which decodes the whole
    image into a full-size temp file, for a compressed or uncompressed source.
    """
    path, _stack = _write_3d_stack(tmp_path, compression=compression)

    calls = []
    orig_asarray = tifffile.TiffFile.asarray

    def spying_asarray(self, *args, **kwargs):
        calls.append(kwargs)
        return orig_asarray(self, *args, **kwargs)

    monkeypatch.setattr(tifffile.TiffFile, "asarray", spying_asarray)

    segment_io.extract_dapi_channel(str(path), dapi_channel_index=1)

    memmap_calls = [c for c in calls if c.get("out") == "memmap"]
    assert memmap_calls == [], (
        f"extract_dapi_channel called tif.asarray(out='memmap') {len(memmap_calls)} "
        "time(s) -- this decodes the WHOLE image into a full-size temp file"
    )


def test_extract_dapi_channel_routes_through_open_lazy(tmp_path, monkeypatch):
    """The read goes through tiled_io.open_lazy, and only the requested channel
    index is sliced out of it.
    """
    path, _stack = _write_3d_stack(tmp_path, n_channels=4)

    seen_paths = []
    seen_indices = []
    orig_open_lazy = segment_io.open_lazy

    def spying_open_lazy(p):
        seen_paths.append(str(p))
        arr, dtype, close = orig_open_lazy(p)

        class _Spy:
            shape = arr.shape
            dtype = arr.dtype

            def __getitem__(self, key):
                seen_indices.append(key[0])
                return arr[key]

        return _Spy(), dtype, close

    monkeypatch.setattr(segment_io, "open_lazy", spying_open_lazy)

    segment_io.extract_dapi_channel(str(path), dapi_channel_index=2)

    assert str(path) in seen_paths, "image never went through open_lazy"
    assert seen_indices == [2], (
        f"expected only channel 2 to be read, got {seen_indices}"
    )


# ---------------------------------------------------------------------------
# 2. Output equivalence: new lazy read == old memmap-then-slice read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compression", [None, "zlib"])
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_3d_branch_equivalence_segment(tmp_path, compression, dtype):
    """3-D (C, Y, X) branch: new segment_io read must exactly match the old
    memmap-then-slice read, for both a compressed and an uncompressed
    source, and both an integer and a float dtype.
    """
    path, stack = _write_3d_stack(tmp_path, dtype=dtype, compression=compression)
    channel_index = 1

    old_image, old_meta = _old_segment_extract_dapi_channel(str(path), channel_index)
    new_image, new_meta = segment_io.extract_dapi_channel(str(path), channel_index)

    assert new_image.dtype == old_image.dtype
    assert new_image.shape == old_image.shape == (stack.shape[1], stack.shape[2])
    np.testing.assert_array_equal(new_image, old_image)
    # Ground truth: the raw pixels actually written for that channel.
    np.testing.assert_array_equal(new_image, stack[channel_index])
    assert bool(old_meta.get("ome")) == bool(new_meta.get("ome"))


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_2d_branch_equivalence_segment(tmp_path, compression):
    """2-D (Y, X) single-channel branch: new segment_io read must exactly
    match the old memmap-then-slice read, compressed and uncompressed.
    """
    path, plane = _write_2d_plane(tmp_path, compression=compression)

    old_image, old_meta = _old_segment_extract_dapi_channel(str(path))
    new_image, new_meta = segment_io.extract_dapi_channel(str(path))

    assert new_image.dtype == old_image.dtype
    assert new_image.shape == old_image.shape == plane.shape
    np.testing.assert_array_equal(new_image, old_image)
    np.testing.assert_array_equal(new_image, plane)
    assert old_meta == new_meta == {}


def test_out_of_range_channel_index_raises_with_original_message(tmp_path):
    path, _stack = _write_3d_stack(tmp_path, n_channels=3)

    with pytest.raises(ValueError) as old_exc:
        _old_segment_extract_dapi_channel(str(path), dapi_channel_index=5)
    with pytest.raises(ValueError) as new_exc:
        segment_io.extract_dapi_channel(str(path), dapi_channel_index=5)

    assert str(old_exc.value) == str(new_exc.value)
    assert str(new_exc.value) == (
        "DAPI channel index 5 out of range for image with 3 channels"
    )


# ---------------------------------------------------------------------------
# 3. segment_cellsam.py: importable in CI (no heavy top-level imports). The
#    delegation identity check proves it is wired to segment_io.
# ---------------------------------------------------------------------------


def test_segment_cellsam_module_has_no_heavy_top_level_imports():
    """cellSAM stays a deferred import inside run_cellsam, so segment_cellsam is
    importable in CI.
    """
    import segment_cellsam

    src = Path(segment_cellsam.__file__).read_text()
    # cellSAM must not be imported at MODULE level (column 0) -- an indented
    # import inside a function body is fine and expected (see run_cellsam).
    for line in src.splitlines():
        if line.startswith("from cellSAM") or line.startswith("import cellSAM"):
            pytest.fail(f"cellSAM import found at module level: {line!r}")


def test_segment_cellsam_delegates_to_segment_io():
    """segment_cellsam.py delegates its channel read to segment_io (same shape as
    test_segment_py_delegation_is_live_when_importable). The tests that follow
    call segment_io directly, so only this proves segment_cellsam is wired to it.
    """
    import segment_cellsam

    assert segment_cellsam.extract_dapi_channel is segment_io.extract_dapi_channel, (
        "bin/segment_cellsam.py no longer delegates its channel read to "
        "segment_io.extract_dapi_channel"
    )


@pytest.mark.parametrize("compression", [None, "zlib"])
@pytest.mark.parametrize("dims", ["2d", "3d"])
def test_segment_cellsam_extract_dapi_channel_equivalence(tmp_path, compression, dims):
    if dims == "3d":
        path, stack = _write_3d_stack(tmp_path, compression=compression)
        channel_index = 0
        expected = stack[channel_index]
    else:
        path, plane = _write_2d_plane(tmp_path, compression=compression)
        channel_index = 0
        expected = plane

    old_image = _old_segment_cellsam_extract_dapi_channel(str(path), channel_index)
    new_image, _new_meta = segment_io.extract_dapi_channel(str(path), channel_index)

    assert new_image.dtype == old_image.dtype
    np.testing.assert_array_equal(new_image, old_image)
    np.testing.assert_array_equal(new_image, expected)


def test_segment_cellsam_extract_dapi_channel_uses_lazy_path(tmp_path, monkeypatch):
    path, _stack = _write_3d_stack(tmp_path, n_channels=3)

    calls = []
    orig_asarray = tifffile.TiffFile.asarray

    def spying_asarray(self, *args, **kwargs):
        calls.append(kwargs)
        return orig_asarray(self, *args, **kwargs)

    monkeypatch.setattr(tifffile.TiffFile, "asarray", spying_asarray)

    segment_io.extract_dapi_channel(str(path), dapi_channel_index=0)

    memmap_calls = [c for c in calls if c.get("out") == "memmap"]
    assert memmap_calls == []


# ---------------------------------------------------------------------------
# 4. segment.py: importable only where stardist/csbdeep/TensorFlow are. The
#    wiring is verified structurally, and by identity where possible.
# ---------------------------------------------------------------------------


def _segment_py_is_importable():
    """Whether ``bin/segment.py`` can actually be imported in THIS environment.

    Imports the module rather than probing a list of packages: ``import stardist`` can
    succeed while ``stardist.models`` raises ``RuntimeError: Please install TensorFlow``.
    `except Exception`, not ImportError, so a collection-time probe skips rather than
    crashing collection.
    """
    try:
        import segment  # noqa: F401, PLC0415
    except Exception:
        return False
    return True


def test_segment_py_delegates_to_segment_io_and_drops_memmap():
    """bin/segment.py must delegate its channel read and not decode eagerly.

    Checked structurally (source text), which works in every environment.
    """
    src = (REPO_ROOT / "bin" / "segment.py").read_text()
    # The eager decode assignment itself, not the bare substring 'out="memmap"',
    # which a docstring may name in prose.
    assert "image_memmap = tif.asarray(out=" not in src, (
        "bin/segment.py still calls asarray(out='memmap') directly"
    )
    # `from segment_io import extract_dapi_channel`, possibly sharing the line with a
    # sibling name (ruff's isort merges same-module imports).
    assert re.search(
        r"^from segment_io import .*\bextract_dapi_channel\b", src, re.MULTILINE
    ), "bin/segment.py no longer delegates to segment_io.extract_dapi_channel"


@pytest.mark.skipif(
    not _segment_py_is_importable(),
    reason="stardist/csbdeep not importable here; the structural test above is the "
    "environment-independent check and always runs",
)
def test_segment_py_delegation_is_live_when_importable():
    """The stronger form of the check above, where the environment permits it.

    Additive to the structural assertions, never a replacement. bin/segment.py imports
    the name directly, so identity against segment_io's function is the delegation.
    """
    import segment  # noqa: PLC0415

    assert segment.extract_dapi_channel is segment_io.extract_dapi_channel, (
        "bin/segment.py no longer delegates its channel read to "
        "segment_io.extract_dapi_channel"
    )
