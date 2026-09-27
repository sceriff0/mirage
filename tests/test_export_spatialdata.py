"""Unit tests for bin/export_spatialdata.py.

Covers the pure-logic layer: measurement-key parsing, QC sanitizing, and the
spatial join of registration residuals onto segmentation labels. The full
SpatialData round-trip needs `spatialdata`/`geopandas`, which live only in the
export container, so it is exercised by the nf-test suite rather than here.

One exception, at the bottom of this file: the pixel-convention invariant runs
`build_shapes` and `build_table` for real against stubbed element MODELS
(`ShapesModel`/`TableModel` are pass-through wrappers around a GeoDataFrame and
an AnnData). Faking the models rather than the geometry is what makes it a real
test -- it is the coordinates those two functions compute, not the container's
serialization, that the invariant is about.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import tifffile

BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))
sys.path.insert(0, str(BIN / "utils"))

spec = importlib.util.spec_from_file_location(
    "export_spatialdata", BIN / "export_spatialdata.py"
)
esd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(esd)


# ── measurement keys ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "key,expected",
    [
        ("CD3: Nucleus: Median", ("CD3", "Nucleus", "Median")),
        ("CD8: Cytoplasm: Mean", ("CD8", "Cytoplasm", "Mean")),
        ("PanCK: Cell: Sum", ("PanCK", "Cell", "Sum")),
        # A bare marker is NOT assumed to be whole-cell mean — var must not assert
        # a compartment the pipeline never measured.
        ("DAPI", ("DAPI", None, None)),
        # Marker names may themselves contain ": "; only the last two tokens count.
        (
            "CD3: clone UCHT1: Nucleus: Median",
            ("CD3: clone UCHT1", "Nucleus", "Median"),
        ),
    ],
)
def test_parse_measurement_key(key, expected):
    assert esd.parse_measurement_key(key) == expected


def test_parse_measurement_key_rejects_empty_marker():
    """': Nucleus: Median' has no marker, so it is not a compartment key."""
    assert esd.parse_measurement_key(": Nucleus: Median") == (
        ": Nucleus: Median",
        None,
        None,
    )


def test_identify_marker_columns_excludes_morphology():
    df = pd.DataFrame(
        {
            "label": [1, 2],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "area": [10, 20],
            "CD3: Cell: Median": [1.5, 2.5],
            "DAPI": [7.0, 8.0],
            "some_text": ["a", "b"],
        }
    )
    assert esd.identify_marker_columns(df) == ["CD3: Cell: Median", "DAPI"]


# ── QC sanitizing ──────────────────────────────────────────────────────────────
def test_sanitize_converts_none_to_nan():
    """warp_seg_qc emits `radius_factor: None`; None does not survive AnnData->Zarr."""
    out = esd.sanitize_for_uns({"radius_factor": None, "n_pairs": 12})
    assert np.isnan(out["radius_factor"])
    assert out["n_pairs"] == 12


def test_sanitize_coerces_numpy_scalars():
    out = esd.sanitize_for_uns(
        {"a": np.int64(3), "b": np.float32(1.5), "c": np.bool_(True)}
    )
    assert (
        isinstance(out["a"], int) and isinstance(out["b"], float) and out["c"] is True
    )


def test_sanitize_recurses_and_stringifies_keys():
    out = esd.sanitize_for_uns({1: {"deep": [np.int64(1), None]}})
    assert "1" in out
    assert out["1"]["deep"][0] == 1


def test_load_qc_keeps_verbatim_json_and_survives_bad_files(tmp_path):
    good = tmp_path / "a_seg_qc.json"
    good.write_text(
        json.dumps({"moving": "cycle2", "matching": {"radius_factor": None}})
    )
    bad = tmp_path / "broken.json"
    bad.write_text("{not json")

    qc, extras = esd.load_qc([str(good), str(bad)], [])
    assert "cycle2" in qc["registration"]
    assert np.isnan(qc["registration"]["cycle2"]["matching"]["radius_factor"])
    # verbatim copy is preserved alongside the flattened form
    assert (
        json.loads(extras["qc_json"]["registration"]["cycle2"])["matching"][
            "radius_factor"
        ]
        is None
    )


# ── mask & pyramid reads (Task 4: one read strategy) ────────────────────────────
# `read_mask` and `read_pyramid_lazy` both need to be dask-backed, aszarr-opened
# region reads -- NOT a whole-array `tifffile.imread` -- because their only
# consumers (`Labels2DModel.parse` / `Image2DModel.parse`) wrap the array
# structurally and never compute across it. These tests exercise both call sites
# for real, against fixtures written the same way the pipeline's own writers
# (`bin/segment.py`, `bin/merge_channels_pyramid.py`) produce them, and assert:
# (1) the pixels read back are identical to a naive whole-array read, and (2) the
# read never eagerly decoded the whole file (the strategy itself, not just its
# output, is pinned).
#
# Pixel equality is NOT the same as "exported output is unchanged", though:
# `spatialdata`'s writer defaults the *exported* zarr's chunk grid to the dask
# array's own chunksize, so whatever native chunk grid the source file happens to
# read back with would otherwise propagate straight into the export. A per-row
# (or per-tile) chunk grid is still pixel-identical, so a pixel-only check would
# not catch that leak; `test_read_mask_rechunks_away_from_...` below exercises
# `read_mask`'s explicit rechunk-to-2048 specifically, using a fixture written
# WITHOUT `tile=` (see `_write_mask`) so it reads back on the un-tiled strip grid
# `bin/segment.py`/`bin/segment_cellsam.py` produced before `bd5861c` tiled those
# writers -- i.e. it deliberately exercises the worst case the rechunk guards
# against, not what those writers emit today (they now write `MASK_TIFF_TILE =
# 1024` tiles; the rechunk still runs, onto the pipeline's common 2048 grid).
#
# `spatialdata`/`geopandas` aren't needed for any of this -- only `tifffile` and
# `dask`, which are real dependencies of this script and importable here.


def _write_mask(path, h=48, w=64, seed=0):
    """A label mask exactly as `bin/segment.py`/`bin/segment_cellsam.py` write one:
    single-page uint32, zlib, bigtiff. No `tile=` -- these are written as strips."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 500, size=(h, w), dtype=np.uint32)
    tifffile.imwrite(str(path), arr, compression="zlib", bigtiff=True)
    return arr


def _write_pyramid(path, c=3, h=64, w=96, seed=1):
    """A minimal real pyramidal OME-TIFF (base level + one half-scale level), built
    the same way `bin/merge_channels_pyramid.py` does: `TiffWriter` + `subifds`.
    Only level 0 needs to be correct for these tests; the second level exists so
    the file is genuinely pyramidal (subifds present), matching production shape.
    """
    rng = np.random.default_rng(seed)
    stack = rng.integers(0, 65000, size=(c, h, w), dtype=np.uint16)
    half = stack[:, ::2, ::2]
    metadata = {"axes": "CYX", "Channel": {"Name": [f"c{i}" for i in range(c)]}}
    options = dict(tile=(32, 32), compression="zlib", photometric="minisblack")
    with tifffile.TiffWriter(str(path), bigtiff=True, ome=True) as tif:
        tif.write(stack, subifds=1, metadata=metadata, **options)
        tif.write(half, subfiletype=1, **options)
    return stack


def _spy_on_imread(monkeypatch):
    """Record the kwargs of every `tifffile.imread` call made through the module
    attribute (which is how `read_mask`/`read_pyramid_lazy` call it), while still
    performing the real read."""
    calls = []
    real_imread = tifffile.imread

    def spying(p, *a, **kw):
        calls.append(kw)
        return real_imread(p, *a, **kw)

    monkeypatch.setattr(tifffile, "imread", spying)
    return calls


def test_read_mask_matches_a_whole_array_read(tmp_path, monkeypatch):
    path = tmp_path / "P001_cell_mask.tif"
    expected = _write_mask(path)
    calls = _spy_on_imread(monkeypatch)

    result = esd.read_mask(str(path))

    assert np.array_equal(np.asarray(result), expected)
    # A slice of the lazy result must match the same slice of the reference --
    # this is what "the consumer can work region-wise" means, exercised for real.
    assert np.array_equal(np.asarray(result[10:20, 5:15]), expected[10:20, 5:15])
    assert calls and calls[0].get("aszarr") is True, (
        "read_mask must open the file through tifffile's zarr view (aszarr=True), "
        "not decode the whole array up front"
    )


def test_read_mask_stays_a_dask_array_until_computed(tmp_path):
    """The strategy itself, not just the pixels: read_mask must return something
    still lazy (a dask array) rather than a materialized numpy array -- otherwise
    it is a whole-array read wearing a zarr costume."""
    path = tmp_path / "mask.tif"
    _write_mask(path)

    result = esd.read_mask(str(path))

    assert isinstance(result, da.Array)


def test_read_mask_squeezes_a_singleton_leading_axis(tmp_path):
    """Some writers add a leading (1, H, W); read_mask must squeeze it away exactly
    like the eager `np.squeeze(tifffile.imread(path))` it replaced did."""
    path = tmp_path / "leading_axis_mask.tif"
    arr = np.random.default_rng(2).integers(0, 50, size=(1, 20, 30)).astype(np.uint32)
    tifffile.imwrite(str(path), arr, compression="zlib", bigtiff=True)

    result = esd.read_mask(str(path))

    assert result.ndim == 2
    assert np.array_equal(np.asarray(result), np.squeeze(arr))


def test_read_mask_rejects_a_non_2d_mask(tmp_path):
    path = tmp_path / "multi_channel_mask.tif"
    arr = np.random.default_rng(3).integers(0, 50, size=(2, 20, 30)).astype(np.uint32)
    tifffile.imwrite(str(path), arr, compression="zlib", bigtiff=True)

    with pytest.raises(ValueError, match="2-D label mask"):
        esd.read_mask(str(path))


def test_read_pyramid_lazy_matches_the_base_level(tmp_path, monkeypatch):
    path = tmp_path / "pyramid.ome.tif"
    expected = _write_pyramid(path)
    calls = _spy_on_imread(monkeypatch)

    result = esd.read_pyramid_lazy(str(path))

    assert isinstance(result, da.Array)
    assert np.array_equal(np.asarray(result), expected)
    assert np.array_equal(np.asarray(result[:, 10:20, 5:15]), expected[:, 10:20, 5:15])
    assert calls and calls[0].get("aszarr") is True and calls[0].get("level") == 0


def test_read_mask_rechunks_away_from_the_file_s_strip_layout(tmp_path):
    """Guards the chunk-grid regression: at slide widths, `bin/segment.py`'s
    un-tiled `tifffile.imwrite` collapses to ~1-row TIFF strips (measured here at
    width 40000 -- Task 4's finding's own cited width -- RowsPerStrip=1 at both
    tifffile 2023.4.12 and 2025.5.10), and `spatialdata`'s `sdata.write()`
    passes no `storage_options`, so `ome_zarr.writer` defaults the exported zarr's
    chunk grid to the dask array's chunksize (`chunks_opt = level.chunksize`).
    Left uncorrected, that turns one mask into tens of thousands of tiny zarr chunk
    objects. `read_mask` must rechunk before returning, so this checks the
    RETURNED array's chunksize, not just its pixels -- pixel equality alone is
    blind to this, since a per-row chunk grid is still bit-identical to a
    coarser one.
    """
    path = tmp_path / "wide_mask.tif"
    h, w = 64, 40000
    arr = np.random.default_rng(7).integers(0, 500, size=(h, w)).astype(np.uint32)
    tifffile.imwrite(str(path), arr, compression="zlib", bigtiff=True)

    # Sanity check on the FIXTURE itself: confirm the raw layout is strictly finer
    # than what `read_mask` returns, so the final assertion below is not vacuous.
    # NOT a fixed row count: tifffile's default strip byte target moved between
    # 2023.4.12 and 2025.5.10 (RowsPerStrip went 1 -> 3 at this file's OLD w=20000,
    # which is why this check is now written against the invariant it actually
    # needs rather than a magic number -- see git history for the failure that
    # caught it). 2048 rows x 40000 x 4 bytes/px is 327 MB per strip, not a layout
    # any plausible strip-byte target will emit, so this still fails loudly if the
    # collapsed-strip regime genuinely disappears.
    raw_store = tifffile.imread(str(path), aszarr=True, level=0)
    raw_chunksize = da.from_zarr(raw_store).chunksize
    raw_store.close()
    assert raw_chunksize[0] < 2048 and raw_chunksize != (min(2048, h), min(2048, w)), (
        f"fixture no longer exercises the strip-collapse regime (raw chunksize "
        f"{raw_chunksize}); widen it until RowsPerStrip collapses again"
    )

    result = esd.read_mask(str(path))

    assert np.array_equal(np.asarray(result), arr)
    assert result.chunksize == (min(2048, h), min(2048, w)), (
        f"read_mask's output chunksize is {result.chunksize} -- the raw file's "
        "collapsed strip layout leaked into the exported array uncorrected"
    )


# ── spatial join of registration residuals ─────────────────────────────────────
def _residual_csv(path, rows, moving="cycle2"):
    pd.DataFrame(
        [
            {
                "moving": moving,
                "ref_x": x,
                "ref_y": y,
                "residual_px": r,
                "stage": "micro",
            }
            for x, y, r in rows
        ]
    ).to_csv(path, index=False)
    return str(path)


def test_join_reg_residuals_matches_nearest_centroid(tmp_path):
    centroids = np.array([[0.0, 0.0], [100.0, 100.0]])
    labels = np.array([7, 9])
    csv = _residual_csv(tmp_path / "r.csv", [(0.5, 0.5, 2.0), (100.2, 99.8, 5.0)])

    out, stats = esd.join_reg_residuals([csv], centroids, labels, max_dist_px=15.0)
    assert list(out.index) == [7, 9]
    assert out["cycle2"].tolist() == [2.0, 5.0]
    assert stats["slides"]["cycle2"]["joined"] == 2


def test_join_reg_residuals_drops_pairs_beyond_radius(tmp_path):
    """An unmatched cell means 'no QC evidence', never 'well registered'."""
    centroids = np.array([[0.0, 0.0]])
    labels = np.array([1])
    csv = _residual_csv(tmp_path / "r.csv", [(500.0, 500.0, 3.0)])

    out, stats = esd.join_reg_residuals([csv], centroids, labels, max_dist_px=15.0)
    assert np.isnan(out["cycle2"].iloc[0])
    assert stats["slides"]["cycle2"]["joined"] == 0
    assert stats["slides"]["cycle2"]["cell_coverage"] == 0.0


def test_join_reg_residuals_keeps_worst_on_collision(tmp_path):
    """Two QC cells on one mask cell -> keep the WORST residual.

    This column exists to exclude badly-registered cells; taking the optimistic
    value would hide exactly the cells it is meant to flag.
    """
    centroids = np.array([[0.0, 0.0]])
    labels = np.array([1])
    csv = _residual_csv(tmp_path / "r.csv", [(0.1, 0.1, 1.0), (0.2, 0.2, 9.0)])

    out, _ = esd.join_reg_residuals([csv], centroids, labels, max_dist_px=15.0)
    assert out["cycle2"].iloc[0] == 9.0


def test_join_reg_residuals_one_column_per_moving_slide(tmp_path):
    centroids = np.array([[0.0, 0.0]])
    labels = np.array([1])
    c2 = _residual_csv(tmp_path / "c2.csv", [(0.0, 0.0, 2.0)], moving="cycle2")
    c3 = _residual_csv(tmp_path / "c3.csv", [(0.0, 0.0, 4.0)], moving="cycle3")

    out, stats = esd.join_reg_residuals([c2, c3], centroids, labels, max_dist_px=15.0)
    assert sorted(out.columns) == ["cycle2", "cycle3"]
    assert out.loc[1, "cycle2"] == 2.0 and out.loc[1, "cycle3"] == 4.0
    assert set(stats["slides"]) == {"cycle2", "cycle3"}


def test_join_reg_residuals_handles_no_inputs():
    out, stats = esd.join_reg_residuals([], np.empty((0, 2)), np.array([1, 2]), 15.0)
    assert out.empty and stats["slides"] == {}


def test_join_reg_residuals_skips_header_only_csv(tmp_path):
    """The stub (and a slide that paired nothing) writes a header-only CSV."""
    p = tmp_path / "empty.csv"
    p.write_text("moving,ref_x,ref_y,residual_px,stage\n")
    out, stats = esd.join_reg_residuals(
        [str(p)], np.array([[0.0, 0.0]]), np.array([1]), 15.0
    )
    assert out.empty and stats["slides"] == {}


# ── pixel convention: the point and the polygon must agree ─────────────────────
# The .zarr carries each cell twice: as a Shapes polygon (from build_shapes,
# reading EXTRACT_CELL_PROPERTIES' contours.json) and as a row of
# ``obsm["spatial"]`` (from build_table, reading the quantification CSV's x/y).
# Those two sources are written in DIFFERENT pixel conventions -- contours.json
# is corner-of-pixel, the CSV's x/y are raw regionprops centroids and therefore
# centre-of-pixel -- so one of them has to be converted before it is stored, or
# every cell's point sits half a pixel up-and-left of its own polygon.
# bin/utils/pixel_convention.py owns the rule.
#
# What is asserted below is the INVARIANT -- a cell's point falls inside that
# same cell's polygon -- not the offset or its direction. That holds under
# either convention as long as both elements use the same one, so this test
# survives a later decision to flip which way the .zarr is normalised.
#
# geopandas / shapely / spatialdata live only in the export container, so the
# element MODELS are faked here (they are pass-through wrappers). The two
# functions under test are the real ones, and so is the contour producer.

_ecp_spec = importlib.util.spec_from_file_location(
    "extract_cell_properties", BIN / "extract_cell_properties.py"
)
ecp = importlib.util.module_from_spec(_ecp_spec)
_ecp_spec.loader.exec_module(ecp)


class _FakePolygon:
    """As much shapely as ``build_shapes`` uses: construction from a ring."""

    def __init__(self, ring):
        self.ring = [(float(x), float(y)) for x, y in ring]


class _FakeGeoDataFrame:
    """As much geopandas as ``build_shapes`` uses: a geometry column + index."""

    def __init__(self, data, index=None):
        self.geometry = list(data["geometry"])
        self.index = list(index)


@pytest.fixture
def fake_geo_stack(monkeypatch):
    """Stub geopandas/shapely/spatialdata so the real builders can run here."""
    import types

    gpd = types.ModuleType("geopandas")
    gpd.GeoDataFrame = _FakeGeoDataFrame

    geometry = types.ModuleType("shapely.geometry")
    geometry.Polygon = _FakePolygon
    shapely = types.ModuleType("shapely")
    shapely.geometry = geometry

    class _TableModel:
        @staticmethod
        def parse(adata, **_kwargs):
            return adata

    models = types.ModuleType("spatialdata.models")
    models.TableModel = _TableModel
    spatialdata = types.ModuleType("spatialdata")
    spatialdata.models = models

    for name, module in (
        ("geopandas", gpd),
        ("shapely", shapely),
        ("shapely.geometry", geometry),
        ("spatialdata", spatialdata),
        ("spatialdata.models", models),
    ):
        monkeypatch.setitem(sys.modules, name, module)


# Cells drawn as ASCII, one per 14x14 tile. The first two are thin
# ANTI-diagonals: the centre->corner translation is (+0.5, +0.5), which is
# perpendicular to their long axis, so half a pixel is the entire margin and
# their centroid lands outside their own polygon the moment the two conventions
# disagree. The rest are ordinary blobs, for which the invariant must hold too.
_CELL_PATTERNS = (
    (
        ".##",
        "##.",
        "#..",
    ),
    (
        "..##",
        ".##.",
        "##..",
    ),
    (
        ".###.",
        "#####",
        "#####",
        "#####",
        ".###.",
    ),
    (
        "###",
        "###",
        "###",
    ),
    (
        ".##.",
        "####",
        "####",
        ".##.",
    ),
)


def _mask_from_patterns(patterns, tile=14):
    """One labelled cell per tile, so no two cells touch."""
    mask = np.zeros((tile * len(patterns), tile), dtype=np.int32)
    for i, pattern in enumerate(patterns):
        for r, row in enumerate(pattern):
            for c, char in enumerate(row):
                if char == "#":
                    mask[i * tile + 4 + r, 4 + c] = i + 1
    return mask


def _point_in_ring(point, ring):
    """Ray-casting point-in-polygon (shapely is not importable here)."""
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        if (y1 > y) != (y2 > y):
            crossing = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing:
                inside = not inside
    return inside


def _build_zarr_elements(tmp_path):
    """Run the real contour producer, then the real shapes/table builders."""
    mask = _mask_from_patterns(_CELL_PATTERNS)
    props, filtered, _ = ecp.extract_morphology(mask)
    contours = ecp.extract_contours(filtered, props, 1.0, 2)

    contours_json = tmp_path / "contours.json"
    contours_json.write_text(json.dumps(contours))

    df = props.reset_index()[
        ["label", "y", "x", "area", "eccentricity", "perimeter", "solidity"]
    ].copy()
    # One marker column: build_table refuses a CSV with no intensities at all.
    df["CD3: Cell: Mean"] = np.arange(len(df), dtype=float)
    quant_csv = tmp_path / "quant.csv"
    df.to_csv(quant_csv, index=False)

    labels = df["label"].to_numpy(dtype=np.int64)
    gdf = esd.build_shapes(str(contours_json), labels, "cells")
    adata = esd.build_table(
        str(quant_csv),
        "P001",
        None,
        {},
        {"qc_json": "{}", "versions": ""},
        {},
    )
    return contours, gdf, adata


def test_table_points_fall_inside_their_own_shape_polygons(tmp_path, fake_geo_stack):
    """Every ``obsm["spatial"]`` row lies inside that cell's Shapes polygon."""
    _contours, gdf, adata = _build_zarr_elements(tmp_path)

    rings = {int(lab): poly.ring for lab, poly in zip(gdf.index, gdf.geometry)}
    points = adata.obsm["spatial"]
    labels = adata.obs[esd.INSTANCE_KEY].to_numpy()

    # Not vacuous: every cell drawn above must have survived contour
    # simplification and be present in BOTH elements, or this asserts nothing.
    assert len(rings) == len(_CELL_PATTERNS), (
        f"only {len(rings)} of {len(_CELL_PATTERNS)} cells produced a polygon -- "
        "the invariant below would be checked on a subset chosen by accident"
    )
    assert set(labels) == set(rings)
    assert points.shape == (len(_CELL_PATTERNS), 2)

    outside = [
        int(lab)
        for lab, point in zip(labels, points)
        if not _point_in_ring(tuple(point), rings[int(lab)])
    ]
    assert not outside, (
        f"cells {outside} have a spatial point outside their own polygon. The "
        "Shapes element and obsm['spatial'] are in different pixel conventions "
        "-- see bin/utils/pixel_convention.py."
    )


def test_shapes_keep_the_contour_ring_verbatim(tmp_path, fake_geo_stack):
    """The polygons are contours.json unmodified; the TABLE is what converts.

    This pins the DIRECTION chosen in bin/utils/pixel_convention.py. The same
    corner-of-pixel rings are shipped to QuPath as ``cells.geojson``, a contract
    with the sibling ``qupath-extension-flowpath`` repo, so making the two
    elements agree by shifting the geometry would fix the .zarr by breaking the
    GeoJSON. Any offset introduced in ``build_shapes`` fails here.
    """
    contours, gdf, _adata = _build_zarr_elements(tmp_path)

    assert len(gdf.geometry) == len(_CELL_PATTERNS)
    for lab, poly in zip(gdf.index, gdf.geometry):
        expected = [(float(x), float(y)) for x, y in contours[str(int(lab))]]
        assert poly.ring == expected, (
            f"cell {lab}: build_shapes altered the contour ring it was given"
        )


# ── QC: columns land in obs, not var (2026-09-27 spec) ──────────────────────────
def test_build_table_puts_qc_columns_in_obs_not_var(tmp_path, fake_geo_stack):
    """A `QC: ...` column in the quantification CSV becomes an `obs` column,
    verbatim, and is never treated as a marker (excluded from `X`/`var`)."""
    df = pd.DataFrame(
        {
            "label": [1, 2],
            "x": [1.0, 2.0],
            "y": [1.0, 2.0],
            "area": [10.0, 20.0],
            "CD3: Cell: Median": [1.0, 2.0],
            "QC: Total intensity": [5.0, float("nan")],
            "QC: Registration Dice: [CD3]": [0.9, 0.8],
        }
    )
    quant_csv = tmp_path / "quant.csv"
    df.to_csv(quant_csv, index=False)

    adata = esd.build_table(
        str(quant_csv), "P001", None, {}, {"qc_json": "{}", "versions": ""}, {}
    )

    assert "QC: Total intensity" in adata.obs.columns
    assert "QC: Registration Dice: [CD3]" in adata.obs.columns
    np.testing.assert_array_equal(
        adata.obs["QC: Registration Dice: [CD3]"].to_numpy(), [0.9, 0.8]
    )
    assert np.isnan(adata.obs["QC: Total intensity"].to_numpy()[1])

    assert "QC: Total intensity" not in adata.var_names
    assert "QC: Registration Dice: [CD3]" not in adata.var_names
    assert list(adata.var_names) == ["CD3: Cell: Median"]
