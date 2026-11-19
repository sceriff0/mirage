#!/usr/bin/env python3
"""The reg_qc=2 GeoJSON is the same whether the mask comes from memory or from a TIFF.

SEG_QC_GEOJSON reads the ``*_cell_mask.tif`` that SEG_QC_SEGMENT (``SEGMENT`` under an
alias) wrote and calls ``mask_to_geojson.mask_to_feature_collection`` via
``bin/mask_to_geojson.py``'s CLI. That is polygon-preserving only if the label mask
survives the TIFF round trip byte-for-byte, written with the exact call
``segment.run_segmentation`` makes (``tifffile.imwrite(..., compression="zlib")``).

Equivalence holds for the same backend only: the QC follows the run's own segmenter,
and on the stardist backend the nuclear channel is chosen positionally
(``--dapi-channel 0``, guarded by ``SegBackends``' channel-0 check).

``bin/segment.py`` imports StarDist/csbdeep/dask at module scope and is not importable
here, which is why the mask is synthetic rather than segmented.
"""

from __future__ import annotations

import json

import mask_to_geojson
import numpy as np
import pytest

# The one label dtype segment.py writes: `segment_nuclei` returns uint32 label images
# (bin/segment.py's docstring for it says so) and `run_segmentation` imwrites them as-is.
LABEL_DTYPE = np.uint32


def _synthetic_cell_mask() -> np.ndarray:
    """A label image with several non-trivial, non-convex blobs.

    Each blob has a bite taken out of it, so the contour has real vertices for
    `approximate_polygon` to keep or drop and a tolerance change is observable.
    """
    mask = np.zeros((120, 160), dtype=LABEL_DTYPE)
    for i, (r, c) in enumerate([(10, 10), (10, 70), (60, 30), (60, 100)], start=1):
        mask[r : r + 40, c : c + 45] = i
        mask[r + 5 : r + 20, c + 5 : c + 22] = 0  # concave notch
        mask[r + 30 : r + 40, c + 35 : c + 45] = 0  # clipped corner
    # A single-pixel label: at tolerance <= 0.5 it survives as a degenerate zero-area
    # three-vertex ring (the `len(ring) < 3` filter misses it); at >= 1.0 it is dropped.
    # Kept so the round trip covers that branch too.
    mask[110, 150] = 5
    return mask


def _write_like_segment_py(path, mask) -> None:
    """Write the mask exactly as ``segment.run_segmentation`` does."""
    import tifffile

    tifffile.imwrite(path, mask, compression="zlib")


@pytest.mark.parametrize("tolerance", [0.0, 0.5, 1.0, 2.0])
def test_mask_tiff_round_trip_preserves_every_polygon(tmp_path, tolerance):
    """TIFF path (mask -> TIFF -> GeoJSON) == in-memory path (mask -> GeoJSON), exactly."""
    mask = _synthetic_cell_mask()

    # In-memory path: mask straight to GeoJSON.
    expected = mask_to_geojson.mask_to_feature_collection(
        mask, simplify_tolerance=tolerance
    )

    # TIFF path: SEG_QC_SEGMENT writes the mask, SEG_QC_GEOJSON reads it back.
    mask_tif = tmp_path / "P001_slide1.ome_cell_mask.tif"
    _write_like_segment_py(mask_tif, mask)
    out = tmp_path / "P001_slide1.geojson"
    n = mask_to_geojson.write_geojson(mask_tif, out, simplify_tolerance=tolerance)

    actual = json.loads(out.read_text())

    assert actual == expected, "the TIFF round trip changed the traced geometry"
    assert n == len(expected["features"])

    # Non-vacuity: `{} == {}` would satisfy the equality above, so pin that the four real
    # cells are actually present AND that each is a real polygon rather than a sliver.
    by_label = {f["properties"]["label"]: f for f in actual["features"]}
    assert {1, 2, 3, 4} <= set(by_label), f"lost cells: {sorted(by_label)}"
    for label in (1, 2, 3, 4):
        ring = by_label[label]["geometry"]["coordinates"][0]
        assert len(ring) >= 5, f"cell {label} traced to a {len(ring)}-vertex sliver"


def test_the_written_label_image_is_bit_identical(tmp_path):
    """The equality above rests on this: zlib TIFF is lossless for uint32 labels.

    Asserted separately so a change of compression or dtype in
    ``segment.run_segmentation`` fails here, naming the cause.
    """
    import tifffile

    mask = _synthetic_cell_mask()
    path = tmp_path / "cell_mask.tif"
    _write_like_segment_py(path, mask)

    read_back = tifffile.imread(path)
    assert read_back.dtype == mask.dtype
    np.testing.assert_array_equal(read_back, mask)


def test_tolerance_actually_reaches_the_tracer(tmp_path):
    """--tolerance must not be silently dropped on the way through the CLI.

    conf/modules.config passes ``--tolerance ${params.simplify_tolerance}``. The checks
    above compare like-for-like at one tolerance, so pin that the value changes the
    output; otherwise the QC could simplify at the script's default unnoticed.
    """
    mask = _synthetic_cell_mask()
    path = tmp_path / "cell_mask.tif"
    _write_like_segment_py(path, mask)

    loose = tmp_path / "loose.geojson"
    tight = tmp_path / "tight.geojson"
    mask_to_geojson.write_geojson(path, loose, simplify_tolerance=8.0)
    mask_to_geojson.write_geojson(path, tight, simplify_tolerance=0.0)

    assert json.loads(loose.read_text()) != json.loads(tight.read_text())


def test_cli_defaults_match_the_pipeline_tolerance(tmp_path, monkeypatch):
    """The CLI is the only way this conversion is reached, so drive it end to end."""
    import sys

    mask = _synthetic_cell_mask()
    path = tmp_path / "cell_mask.tif"
    _write_like_segment_py(path, mask)
    out = tmp_path / "slide.geojson"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mask_to_geojson.py",
            "--mask",
            str(path),
            "--out",
            str(out),
            "--tolerance",
            "1.0",
        ],
    )
    mask_to_geojson.main()

    assert json.loads(out.read_text()) == mask_to_geojson.mask_to_feature_collection(
        mask, simplify_tolerance=1.0
    )
