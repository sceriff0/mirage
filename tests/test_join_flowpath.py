"""Unit tests for bin/join_flowpath.py.

Covers the join strategies, positivity extraction, and cohort flattening. The
zarr I/O layer needs `spatialdata` (export container only); everything tested
here operates on AnnData and plain frames.

One test builds its `obsm["spatial"]` with the REAL
`export_spatialdata.build_table`, stubbing only `TableModel` (a pass-through
wrapper): `join_by_centroid`'s inverse is only correct relative to the pixel
convention `build_table` stores, which a hand-written array would not track.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# CI installs anndata; the skip is for dev environments without it, where an
# ImportError would take the whole pytest run down.
ad = pytest.importorskip("anndata")

BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))
sys.path.insert(0, str(BIN / "utils"))

spec = importlib.util.spec_from_file_location("join_flowpath", BIN / "join_flowpath.py")
jf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jf)

_esd_spec = importlib.util.spec_from_file_location(
    "export_spatialdata", BIN / "export_spatialdata.py"
)
esd = importlib.util.module_from_spec(_esd_spec)
_esd_spec.loader.exec_module(esd)

PX = 0.325


def flowpath_frame(n=3, with_label=False, signs=None, phenotypes=None):
    """A minimal PhenotypeCsvExporter-shaped frame."""
    df = pd.DataFrame(
        {
            "cell_id": range(n),
            "phenotype": phenotypes if phenotypes is not None else ["T cell"] * n,
            "Out_of_annotation": [False] * n,
            "Outlier": [False] * n,
            # µm, as MIRAGE wrote them. export_geojson's "Centroid X µm" is
            # corner_px * pixel_size, and corner = regionprops centroid + 0.5
            # (bin/utils/pixel_convention.py), so a cell whose centroid sits at
            # centre-of-pixel i*10 is exported at (i*10 + 0.5) * PX.
            "centroid_x": [(i * 10 + 0.5) * PX for i in range(n)],
            "centroid_y": [(i * 10 + 0.5) * PX for i in range(n)],
            "area": [100.0] * n,
            "perimeter": [40.0] * n,
            "eccentricity": [0.5] * n,
            "solidity": [0.9] * n,
        }
    )
    if with_label:
        df["label"] = [10 + i for i in range(n)]
    for col, vals in (signs or {}).items():
        df[col] = vals
    return df


# ── positivity ─────────────────────────────────────────────────────────────────
def test_gated_sign_columns_ignores_all_blank():
    """A blank sign column means no gate touched it — not 'all negative'."""
    df = flowpath_frame(3, signs={"CD3_sign": ["+", "", "+"], "CD8_sign": ["", "", ""]})
    assert jf.gated_sign_columns(df) == ["CD3_sign"]


def test_build_positivity_matrix_and_names():
    df = flowpath_frame(
        3, signs={"CD3_sign": ["+", "", "+"], "CD8_sign": ["", "+", ""]}
    )
    mat, names = jf.build_positivity(df)
    assert names == ["CD3", "CD8"]
    assert mat.dtype == bool
    assert mat[:, 0].tolist() == [True, False, True]
    assert mat[:, 1].tolist() == [False, True, False]


def test_build_positivity_handles_no_gates():
    mat, names = jf.build_positivity(flowpath_frame(2))
    assert names == [] and mat.shape == (2, 0)


def test_build_positivity_strips_whitespace():
    df = flowpath_frame(2, signs={"CD3_sign": [" + ", " "]})
    mat, _ = jf.build_positivity(df)
    assert mat[:, 0].tolist() == [True, False]


def test_read_flowpath_csv_rejects_wrong_schema(tmp_path):
    p = tmp_path / "not_flowpath.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="FlowPath phenotype export"):
        jf.read_flowpath_csv(str(p))


# ── joins ──────────────────────────────────────────────────────────────────────
def test_join_by_label_is_order_independent():
    """The whole point of a label join: row order must not matter."""
    df = flowpath_frame(3, with_label=True).iloc[::-1].reset_index(drop=True)
    idx, stats = jf.join_by_label(df, np.array([10, 11, 12]))
    assert stats["method"] == "label" and stats["matched"] == 3
    assert df["label"].to_numpy()[idx].tolist() == [10, 11, 12]


def test_join_by_label_marks_missing_as_negative_one():
    df = flowpath_frame(2, with_label=True)  # labels 10, 11
    idx, stats = jf.join_by_label(df, np.array([10, 99]))
    assert idx[0] == 0 and idx[1] == -1
    assert stats["matched"] == 1


@pytest.fixture
def stub_table_model(monkeypatch):
    """spatialdata is container-only; TableModel.parse is a pass-through."""
    import types

    class _TableModel:
        @staticmethod
        def parse(adata, **_kwargs):
            return adata

    models = types.ModuleType("spatialdata.models")
    models.TableModel = _TableModel
    spatialdata = types.ModuleType("spatialdata")
    spatialdata.models = models
    monkeypatch.setitem(sys.modules, "spatialdata", spatialdata)
    monkeypatch.setitem(sys.modules, "spatialdata.models", models)


def _mirage_obsm_spatial(tmp_path, centre_px):
    """The real obsm["spatial"] EXPORT_SPATIALDATA publishes for these cells.

    Built by the real build_table from a quantification CSV carrying raw
    regionprops (centre-of-pixel) centroids, so this array tracks the store's
    convention instead of asserting a remembered one.
    """
    pd.DataFrame(
        {
            "label": np.arange(1, len(centre_px) + 1),
            "x": centre_px[:, 0],
            "y": centre_px[:, 1],
            "area": 50.0,
            "CD3: Cell: Mean": 1.0,
        }
    ).to_csv(tmp_path / "quant.csv", index=False)
    table = esd.build_table(
        str(tmp_path / "quant.csv"),
        "P001",
        None,
        {},
        {"qc_json": "{}", "versions": ""},
        {},
    )
    return table.obsm["spatial"]


def test_join_by_centroid_inverts_mirage_convention(tmp_path, stub_table_model):
    """The inverse must land ON obsm["spatial"], not merely near it.

    Both sides are corner-of-pixel: FlowPath's µm centroids come from
    export_geojson's "Centroid X µm", and obsm["spatial"] is converted by
    build_table. A half-pixel correction on either side is not a rounding
    detail -- it is a constant bias that spends part of the
    --centroid-join-max-px budget and narrows the margin between a true match
    and its neighbour.
    """
    # What regionprops measured, in skimage's centre-of-pixel convention.
    centre_px = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    centroids = _mirage_obsm_spatial(tmp_path, centre_px)

    df = flowpath_frame(3)
    idx, stats = jf.join_by_centroid(df, centroids, PX, max_dist_px=2.0)
    assert stats["method"] == "centroid"
    assert idx.tolist() == [0, 1, 2]

    # The assertion that actually pins the convention. A generous radius pairs
    # the cells correctly even with a half-pixel bias (0.707 px < 2 px), so it
    # is the TIGHT radius that distinguishes "exact" from "close enough".
    idx_exact, stats_exact = jf.join_by_centroid(df, centroids, PX, max_dist_px=0.001)
    assert idx_exact.tolist() == [0, 1, 2], (
        "the centroid inverse does not land exactly on obsm['spatial']: it is "
        f"off by a constant ({stats_exact['matched']}/3 matched within 0.001 px). "
        "Both sides are corner-of-pixel -- see bin/utils/pixel_convention.py."
    )


def test_join_by_centroid_is_mutual():
    """A far-away table cell must not steal a FlowPath row already claimed."""
    df = flowpath_frame(1)  # one cell, corner-of-pixel, at (0.5, 0.5)
    centroids = np.array([[0.5, 0.5], [1.1, 0.5]])
    idx, _ = jf.join_by_centroid(df, centroids, PX, max_dist_px=5.0)
    assert idx.tolist() == [0, -1]


def test_join_by_centroid_respects_max_distance():
    df = flowpath_frame(1)
    centroids = np.array([[500.0, 500.0]])
    idx, stats = jf.join_by_centroid(df, centroids, PX, max_dist_px=5.0)
    assert idx.tolist() == [-1] and stats["matched"] == 0


def test_resolve_join_prefers_label_over_centroid():
    df = flowpath_frame(2, with_label=True)
    _, stats = jf.resolve_join(df, np.array([10, 11]), np.zeros((2, 2)), PX, 10.0, 0.5)
    assert stats["method"] == "label"


def test_resolve_join_refuses_positional_when_nothing_to_match_on():
    df = flowpath_frame(2)
    with pytest.raises(ValueError, match="Refusing to align positionally"):
        jf.resolve_join(df, np.array([1, 2]), None, PX, 10.0, 0.5)


def test_resolve_join_fails_below_min_match_fraction():
    """A CSV from the wrong patient must fail loudly, not label 5% of cells."""
    df = flowpath_frame(2, with_label=True)  # labels 10, 11
    with pytest.raises(ValueError, match="Refusing to write"):
        jf.resolve_join(df, np.array([900, 901, 902, 903]), None, PX, 10.0, 0.5)


# ── applying to AnnData ────────────────────────────────────────────────────────
def _adata(n=3, labels=None):
    a = ad.AnnData(
        X=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        obs=pd.DataFrame(
            {"label": labels if labels is not None else np.arange(10, 10 + n)},
            index=[str(i) for i in range(n)],
        ),
        var=pd.DataFrame(index=["CD3", "CD8"]),
    )
    a.obsm["spatial"] = np.array([[i * 10.0, i * 10.0] for i in range(n)])
    a.layers["zscore"] = np.zeros((n, 2), dtype=np.float32)
    return a


def test_apply_flowpath_sets_phenotype_and_positivity():
    a = _adata(3)
    df = flowpath_frame(3, with_label=True, signs={"CD3_sign": ["+", "", "+"]})
    idx, stats = jf.join_by_label(df, a.obs["label"].to_numpy())
    jf.apply_flowpath(a, df, idx, stats)

    assert list(a.obs["phenotype"]) == ["T cell"] * 3
    assert a.uns["positivity_columns"] == ["CD3"]
    assert a.obsm["positivity"][:, 0].tolist() == [True, False, True]
    assert a.obs["fp_matched"].all()


def test_apply_flowpath_marks_unmatched_cells_unclassified():
    """An unmatched cell is 'unclassified', and fp_matched records why."""
    a = _adata(2, labels=np.array([10, 999]))
    df = flowpath_frame(2, with_label=True, signs={"CD3_sign": ["+", "+"]})
    idx, stats = jf.join_by_label(df, a.obs["label"].to_numpy())
    jf.apply_flowpath(a, df, idx, stats)

    assert list(a.obs["phenotype"]) == ["T cell", "unclassified"]
    assert a.obs["fp_matched"].tolist() == [True, False]
    # An unmatched cell must never inherit a neighbour's positivity.
    assert a.obsm["positivity"][1, 0] is np.False_ or not a.obsm["positivity"][1, 0]


def test_apply_flowpath_blank_phenotype_becomes_unclassified():
    a = _adata(2)
    df = flowpath_frame(2, with_label=True, phenotypes=["", "B cell"])
    idx, stats = jf.join_by_label(df, a.obs["label"].to_numpy())
    jf.apply_flowpath(a, df, idx, stats)
    assert list(a.obs["phenotype"]) == ["unclassified", "B cell"]


# ── flattening ─────────────────────────────────────────────────────────────────
def test_flatten_emits_intensities_zscores_and_positivity():
    a = _adata(3)
    df = flowpath_frame(3, with_label=True, signs={"CD3_sign": ["+", "", "+"]})
    idx, stats = jf.join_by_label(df, a.obs["label"].to_numpy())
    jf.apply_flowpath(a, df, idx, stats)

    flat = jf.flatten(a)
    assert len(flat) == 3
    for col in (
        "label",
        "phenotype",
        "x_px",
        "y_px",
        "CD3",
        "CD8",
        "CD3_zscore",
        "CD3_positive",
    ):
        assert col in flat.columns, col
    assert flat["CD3_positive"].tolist() == [True, False, True]


# ── pairing ────────────────────────────────────────────────────────────────────
def test_pair_inputs_matches_by_stem():
    pairs = jf.pair_inputs(
        ["out/P002/P002.zarr", "out/P001/P001.zarr"],
        ["pheno/P001_phenotypes.csv", "pheno/P002_phenotypes.csv"],
    )
    assert dict((Path(z).stem, Path(c).stem) for z, c in pairs) == {
        "P001": "P001_phenotypes",
        "P002": "P002_phenotypes",
    }


def test_pair_inputs_single_pair_passes_through():
    assert jf.pair_inputs(["a.zarr"], ["b.csv"]) == [("a.zarr", "b.csv")]


def test_pair_inputs_refuses_ambiguous_counts():
    with pytest.raises(SystemExit, match="cannot pair inputs"):
        jf.pair_inputs(["P001.zarr"], ["x.csv", "y.csv"])
