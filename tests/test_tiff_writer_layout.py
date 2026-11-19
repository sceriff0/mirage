"""Every mask-writing TIFF the segmentation backends produce must be tiled, not striped.

tifffile's zarr view of a striped TIFF reports one chunk per whole plane (``chunks=(H, W)``),
so any region read has to decode the entire plane (see ``bin/convert_image.py`` and
``tests/test_convert_streaming_write.py`` for CONVERT_IMAGE). The segmentation mask is the
highest-fan-out artifact: read back by ``QUANTIFY`` (once per marker),
``EXTRACT_CELL_PROPERTIES``, ``EXTRACT_NUCLEI_PROPERTIES``, ``SEG_QC_GEOJSON``,
``EXPORT_GEOJSON``, ``EXPORT_SPATIALDATA``, ``SEG_QUALITY_EVAL`` and ``WARP_SEG_QC``.

Four writers:

* ``bin/segment.py`` (StarDist backend)
* ``bin/segment_cellsam.py`` (CellSAM backend)
* ``bin/segment_instantseg.py`` (InstanSeg backend)
* ``bin/extract_mask_series.py`` (mask re-read on the ``add_cycle`` path)

``bin/segment_cellsam.py``, ``bin/segment_instantseg.py`` and ``bin/extract_mask_series.py``
import cleanly here, so their ``MASK_TIFF_TILE`` constants are read directly.
``bin/segment.py`` imports ``stardist``/``csbdeep`` at module level and is read by AST
instead. It is not stubbed into ``sys.modules``: pytest imports every test module at
collection, so a stub would break ``tests/test_segment_lazy_dapi_read.py``'s premise check.

The ML-backend writers need a real model, so their round-trip and chunk geometry are
exercised with an ``imwrite`` carrying the same keyword arguments as the real call
(``compression="zlib"``, ``bigtiff=True``, the writer's tile size).
``test_the_real_call_sites_pass_the_tile_constant`` checks by AST that each backend's real
call sites pass the constant. ``extract_mask_series.py`` is driven through its real
``main()``.

Two properties are pinned for every writer:

1. **G1: decoded pixels are bit-identical** (exact equality on label ids) after tiling,
   including a mask smaller than one 1024x1024 tile, which must decode at its original shape.
2. **The chunk geometry is tiled**, via the TIFF tags (``TileWidth``/``TileLength``) and the
   zarr chunk shape region readers see (``tifffile.imread(path, aszarr=True)``).
   ``test_a_striped_write_of_the_same_array_fails_the_chunk_geometry_assertion`` is the
   control: the same assertion fails on a striped write.
"""

from __future__ import annotations

import ast
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

tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")

import extract_mask_series as ems  # noqa: E402
import segment_cellsam  # noqa: E402
import segment_instantseg  # noqa: E402

SEGMENT_PY = REPO_ROOT / "bin" / "segment.py"
SEGMENT_CELLSAM_PY = REPO_ROOT / "bin" / "segment_cellsam.py"
SEGMENT_INSTANTSEG_PY = REPO_ROOT / "bin" / "segment_instantseg.py"


# ---------------------------------------------------------------------------
# AST helpers for bin/segment.py, which this file never imports (see module docstring)
# ---------------------------------------------------------------------------


def _module_constant(path: Path, name: str):
    """The value of a module-level ``name = <literal>`` assignment in ``path``."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"no module-level assignment to {name!r} found in {path}")


def _count_imwrite_tile_kwarg_call_sites(path: Path, tile_name: str) -> int:
    """How many ``tifffile.imwrite(...)`` or ``ome_io.write_tiff(...)`` calls in ``path``
    pass ``tile=(tile_name, tile_name)``.

    Both spellings: ``bin/utils/ome_io.py``'s ``write_tiff`` forwards verbatim to
    ``tifffile.imwrite``. An AST ``Call`` match, so ``tile=`` quoted in prose is not counted.
    """
    tree = ast.parse(path.read_text())
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in ("imwrite", "write_tiff"):
            continue
        for kw in node.keywords:
            if (
                kw.arg == "tile"
                and isinstance(kw.value, ast.Tuple)
                and len(kw.value.elts) == 2
                and all(
                    isinstance(elt, ast.Name) and elt.id == tile_name
                    for elt in kw.value.elts
                )
            ):
                count += 1
    return count


_SEGMENT_PY_TILE = _module_constant(SEGMENT_PY, "MASK_TIFF_TILE")

# writer name -> tile size in effect, used to drive the parametrized tests below without
# importing bin/segment.py.
_TILE_BY_WRITER = {
    "segment": _SEGMENT_PY_TILE,
    "segment_cellsam": segment_cellsam.MASK_TIFF_TILE,
    "segment_instantseg": segment_instantseg.MASK_TIFF_TILE,
}

# writer name -> source path, for the AST call-site check: proves the real imwrite calls
# inside each backend's run function pass tile=(MASK_TIFF_TILE, MASK_TIFF_TILE), not just
# that the constant exists.
_SOURCE_PATH_BY_WRITER = {
    "segment": SEGMENT_PY,
    "segment_cellsam": SEGMENT_CELLSAM_PY,
    "segment_instantseg": SEGMENT_INSTANTSEG_PY,
}


def _chunks(path):
    """The zarr chunk shape a real reader (``bin/utils/tiled_io.py:open_lazy``) sees."""
    store = tifffile.imread(str(path), aszarr=True)
    try:
        grp = zarr.open(store, mode="r")
        arr = grp[0] if isinstance(grp, zarr.Group) else grp
        return arr.chunks
    finally:
        store.close()


def _label_mask(shape=(40, 30), seed=1):
    """Smaller than one 1024x1024 tile in both dims: tifffile pads the tile internally,
    and the decoded array must come back at this original (unpadded) shape.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 5000, size=shape, dtype=np.uint32)


def _write_like(path, array, tile):
    """Round-trip ``array`` through an imwrite call with the same keyword arguments as
    the writers' real mask writes (``compression="zlib"``, ``bigtiff=True``), plus the
    tile size in effect for the given writer.
    """
    tifffile.imwrite(path, array, compression="zlib", bigtiff=True, tile=(tile, tile))


# ---------------------------------------------------------------------------
# the constant itself, and that both real call sites in each ML backend use it
# ---------------------------------------------------------------------------


_ALL_WRITER_TILES = [
    *_TILE_BY_WRITER.items(),
    ("extract_mask_series", ems.MASK_TIFF_TILE),
]


@pytest.mark.parametrize("writer, tile", _ALL_WRITER_TILES)
def test_the_tile_constant_is_a_valid_tifffile_tile_size(writer, tile):
    # tifffile requires each tile dimension to be a multiple of 16. 1024 is a tuning
    # choice, so the constraint is asserted, not the value.
    assert tile % 16 == 0


@pytest.mark.parametrize("writer, path", _SOURCE_PATH_BY_WRITER.items())
def test_the_real_call_sites_pass_the_tile_constant(writer, path):
    """The AST call-site check, applied to all three ML-backend writers.

    The tests above read ``MASK_TIFF_TILE`` and rebuild an independent call
    (``_write_like``), so they prove only that the constant is well-formed. This checks
    the real mask writes in each backend actually pass it.
    """
    n = _count_imwrite_tile_kwarg_call_sites(path, "MASK_TIFF_TILE")
    assert n == 2, (
        f"expected both the nuclei-mask and cell-mask tifffile.imwrite calls in "
        f"{path.name} to pass tile=(MASK_TIFF_TILE, MASK_TIFF_TILE), found {n} that do"
    )


# ---------------------------------------------------------------------------
# G1: decoded pixels are bit-identical, including the smaller-than-one-tile case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("writer, tile", _TILE_BY_WRITER.items())
def test_decoded_pixels_are_bit_identical_after_tiling(tmp_path, writer, tile):
    array = _label_mask()
    path = tmp_path / f"{writer}_mask.tif"
    _write_like(path, array, tile)

    back = tifffile.imread(str(path))
    np.testing.assert_array_equal(back, array)
    assert back.shape == array.shape, (
        "a mask smaller than one tile must still decode at its original (unpadded) shape"
    )
    assert back.dtype == array.dtype


# ---------------------------------------------------------------------------
# the chunk geometry is actually tiled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("writer, tile", _TILE_BY_WRITER.items())
def test_the_write_reports_a_tiled_chunk_geometry(tmp_path, writer, tile):
    array = _label_mask()
    path = tmp_path / f"{writer}_mask.tif"
    _write_like(path, array, tile)

    with tifffile.TiffFile(str(path)) as tif:
        tags = tif.pages[0].tags
        assert tags["TileWidth"].value == tile
        assert tags["TileLength"].value == tile

    assert _chunks(path) == (tile, tile), (
        "the zarr view a real reader (tiled_io.open_lazy) sees must report the TILE "
        "shape as its chunk size, not a full-plane span"
    )


def test_a_striped_write_of_the_same_array_fails_the_chunk_geometry_assertion(tmp_path):
    """The control: the same assertion applied to a striped write.

    Same array, same compression, no ``tile=`` (tifffile's striped default). A test that
    only checked round-trip equality would pass on this file too.
    """
    array = _label_mask()
    path = tmp_path / "striped_mask.tif"
    tifffile.imwrite(path, array, compression="zlib", bigtiff=True)

    with tifffile.TiffFile(str(path)) as tif:
        tag_names = {t.name for t in tif.pages[0].tags}
        assert "TileWidth" not in tag_names

    assert _chunks(path) == array.shape, (
        "a striped TIFF reports one chunk per whole plane -- this is the cost the tile= "
        "fix removes"
    )


# ---------------------------------------------------------------------------
# extract_mask_series.py: cheaply callable, exercised through its real code path
# ---------------------------------------------------------------------------


def test_extract_mask_series_writes_tiled_masks_through_its_own_code_path(
    tmp_path, monkeypatch
):
    """No heavy ML deps, so drive the real ``main()``.

    Mirrors ``tests/test_mask_series_write_contract.py``'s fixture: a two-series OME-TIFF
    (an intensity pyramid, then the uint32 mask series), smaller than one tile in both
    dims to exercise the same pad-to-tile edge case as the parametrized tests above.
    """
    h, w = 40, 30
    image = np.zeros((2, h, w), dtype=np.uint16)
    masks = np.stack(
        [
            np.arange(h * w, dtype=np.uint32).reshape(h, w) % 97,
            np.arange(h * w, dtype=np.uint32).reshape(h, w) % 31,
        ]
    )
    pyramid = tmp_path / "merged.ome.tiff"
    with tifffile.TiffWriter(str(pyramid), bigtiff=True) as tw:
        tw.write(image, photometric="minisblack")
        tw.write(masks, photometric="minisblack")

    outdir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["extract_mask_series.py", "--pyramid", str(pyramid), "--outdir", str(outdir)],
    )
    ems.main()

    tile = ems.MASK_TIFF_TILE
    for name, expected in (("cell_mask.tif", masks[0]), ("nuclei_mask.tif", masks[1])):
        path = outdir / name
        back = tifffile.imread(str(path))
        np.testing.assert_array_equal(back, expected)
        assert back.shape == expected.shape

        with tifffile.TiffFile(str(path)) as tif:
            tags = tif.pages[0].tags
            assert tags["TileWidth"].value == tile
            assert tags["TileLength"].value == tile

        assert _chunks(path) == (tile, tile)
