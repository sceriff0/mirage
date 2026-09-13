"""`extract_mask_series.py` must write masks the way every other mask writer here does.

`bin/extract_mask_series.py:50` wrote full-resolution uint32 label masks with no keyword
arguments at all -- uncompressed, and in classic TIFF. Both matter, and neither is a judgment
call about compression policy: this one writer had drifted from three siblings and a rule the
repo states elsewhere.

  * **Compression.** `segment.py:476`, `segment_cellsam.py:304` and `segment_instantseg.py:252`
    all write their masks `compression="zlib"`. Label masks compress extremely well -- they are
    large runs of identical integers -- so this is close to free.

  * **BigTIFF.** `tiled_stitch.py:118` puts it plainly: bigtiff is "mandatory, not an
    optimisation ... classic TIFF's 32-bit offsets overflow (struct.error: 'I' format requires
    0 <= number <= 4294967295) the moment the output crosses 4 GB". A 40000x40000 uint32 mask is
    40000 * 40000 * 4 = 6.4 GB on its own, so the *uncompressed* write overflows before the pair
    is considered.

This is on the add_cycle path, which reads masks back out of a prior run's pyramid -- so the
failure lands on someone extending a completed patient run, at the point where the prior run is
largest.
"""

import os
import sys

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

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")

import extract_mask_series as ems  # noqa: E402


def _pyramid_with_mask_series(path, shape=(64, 64)):
    """A two-series OME-TIFF: an intensity pyramid, then the uint32 mask series."""
    h, w = shape
    image = np.zeros((2, h, w), dtype=np.uint16)
    masks = np.stack(
        [
            np.arange(h * w, dtype=np.uint32).reshape(h, w) % 97,
            np.arange(h * w, dtype=np.uint32).reshape(h, w) % 31,
        ]
    )
    with tifffile.TiffWriter(str(path), bigtiff=True) as tw:
        tw.write(image, photometric="minisblack")
        tw.write(masks, photometric="minisblack")
    return masks


def _run(tmp_path, monkeypatch):
    # extract_mask_series.main() reads sys.argv directly rather than taking argv, so drive it
    # the way the process does instead of changing production code for test convenience.
    pyramid = tmp_path / "merged.ome.tiff"
    expected = _pyramid_with_mask_series(pyramid)
    outdir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["extract_mask_series.py", "--pyramid", str(pyramid), "--outdir", str(outdir)],
    )
    ems.main()
    return outdir, expected


def test_masks_are_written_compressed_like_every_other_mask_writer(
    tmp_path, monkeypatch
):
    outdir, _ = _run(tmp_path, monkeypatch)

    for name in ("cell_mask.tif", "nuclei_mask.tif"):
        with tifffile.TiffFile(str(outdir / name)) as tif:
            compression = tif.pages[0].compression
            assert compression != 1, (
                f"{name} is written uncompressed (compression={compression!r}); "
                "segment.py, segment_cellsam.py and segment_instantseg.py all use zlib"
            )


def test_masks_are_written_bigtiff_so_a_full_resolution_pair_cannot_overflow(
    tmp_path, monkeypatch
):
    outdir, _ = _run(tmp_path, monkeypatch)

    for name in ("cell_mask.tif", "nuclei_mask.tif"):
        with tifffile.TiffFile(str(outdir / name)) as tif:
            assert tif.is_bigtiff, (
                f"{name} is classic TIFF; a 40000x40000 uint32 mask is 6.4 GB and overflows "
                "classic TIFF's 32-bit offsets, exactly as tiled_stitch.py:118 documents"
            )


def test_the_mask_values_survive_the_write_unchanged(tmp_path, monkeypatch):
    """Compression must be lossless -- these are label ids, not intensities."""
    outdir, expected = _run(tmp_path, monkeypatch)

    cell = tifffile.imread(str(outdir / "cell_mask.tif"))
    nuclei = tifffile.imread(str(outdir / "nuclei_mask.tif"))

    assert cell.dtype == np.uint32
    np.testing.assert_array_equal(cell, expected[0])
    np.testing.assert_array_equal(nuclei, expected[1])
