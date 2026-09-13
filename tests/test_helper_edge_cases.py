"""Boundary cases for the small pure helpers the pipeline's scripts share.

Each expected value comes from the helper's docstring or its first lines; a
failure here means the code and its own documentation disagree, which is a
finding to report, not a test to relax.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bin"))
sys.path.insert(0, os.path.join(ROOT, "bin", "utils"))

from pixel_size import _to_um, unit_to_um  # noqa: E402
from tile_grid import _edges, tile_grid  # noqa: E402
from tre_report import _pct  # noqa: E402

# ── tile_grid._edges ────────────────────────────────────────────────────────────


def test_edges_exact_multiple_ends_on_size():
    assert _edges(8, 4) == [0, 4, 8]


def test_edges_last_cell_may_be_short():
    assert _edges(10, 4) == [0, 4, 8, 10]


def test_edges_size_smaller_than_tile_is_one_cell():
    assert _edges(3, 4) == [0, 3]


def test_edges_zero_size_is_a_single_boundary():
    assert _edges(0, 4) == [0]


@pytest.mark.parametrize("tile", [0, -1])
def test_edges_rejects_a_non_positive_tile(tile):
    with pytest.raises(ValueError, match="tile size must be positive"):
        _edges(8, tile)


def test_tile_grid_read_window_is_clamped_at_the_image_edge():
    tiles = tile_grid(8, 8, 4, halo=3)
    first = tiles[0]
    # core (0,0,4,4); a halo of 3 cannot go below 0 on the top-left tile
    assert first.read[0] == 0 and first.read[1] == 0
    assert len(tiles) == 4


# ── pixel_size._to_um / unit_to_um ──────────────────────────────────────────────


@pytest.mark.parametrize("raw", [None, "", "abc", "0", "-0.5"])
def test_to_um_returns_none_for_an_untrustworthy_value(raw):
    assert _to_um(raw, "µm") is None


def test_to_um_defaults_to_micrometres_when_the_unit_is_absent():
    # OME's default unit when the attribute is absent is µm (2016-06 schema).
    assert _to_um("0.5", None) == pytest.approx(0.5)


def test_to_um_scales_by_the_unit_table():
    assert unit_to_um("µm") == pytest.approx(1.0)
    assert _to_um("2", "µm") == pytest.approx(2.0)


def test_unit_to_um_is_none_for_an_unknown_unit():
    assert unit_to_um("furlong") is None
    assert _to_um("1", "furlong") is None


# ── quantify._safe_mean ─────────────────────────────────────────────────────────


def test_safe_mean_is_nan_where_the_count_is_zero_and_a_ratio_elsewhere():
    """bin/quantify.py:_safe_mean's docstring says 'Element-wise sums / counts, NaN

    where counts is 0' (bin/quantify.py:46). A nonzero sum over a zero count
    (sums[2]=5.0, counts[2]=0) must yield NaN, not +inf, matching the docstring's
    own 'NaN, not 0.0' framing for an unmeasured compartment.
    """
    from quantify import _safe_mean

    sums = np.array([10.0, 0.0, 5.0])
    counts = np.array([2, 0, 0])
    out = _safe_mean(sums, counts)
    assert out[0] == pytest.approx(5.0)
    assert np.isnan(out[1]) and np.isnan(out[2]), (
        "an unmeasured compartment is NaN, never 0.0"
    )


# ── tre_report._pct ─────────────────────────────────────────────────────────────


def test_pct_of_nothing_is_all_none():
    assert _pct([]) == {"mean": None, "p50": None, "p90": None, "max": None}


def test_pct_of_one_value_collapses_every_statistic_onto_it():
    out = _pct([3.0])
    assert out["mean"] == out["p50"] == out["p90"] == out["max"] == pytest.approx(3.0)


def test_pct_accepts_a_generator():
    out = _pct(v for v in (1.0, 2.0, 3.0))
    assert out["max"] == pytest.approx(3.0)


# ── merge_quant_csvs._load_intensity_csvs ───────────────────────────────────────


def test_load_intensity_csvs_explicit_list_wins_over_directory(tmp_path):
    from merge_quant_csvs import _load_intensity_csvs

    (tmp_path / "b_quant.csv").write_text("cell_id\n1\n")
    (tmp_path / "a_quant.csv").write_text("cell_id\n1\n")
    explicit = _load_intensity_csvs(
        csvs_dir=tmp_path, csv_files_list=[str(tmp_path / "b_quant.csv")]
    )
    assert [p.name for p in explicit] == ["b_quant.csv"]


def test_load_intensity_csvs_directory_glob_is_sorted(tmp_path):
    from merge_quant_csvs import _load_intensity_csvs

    (tmp_path / "b_quant.csv").write_text("cell_id\n1\n")
    (tmp_path / "a_quant.csv").write_text("cell_id\n1\n")
    (tmp_path / "ignored.csv").write_text("cell_id\n1\n")
    assert [p.name for p in _load_intensity_csvs(csvs_dir=tmp_path)] == [
        "a_quant.csv",
        "b_quant.csv",
    ]


def test_load_intensity_csvs_with_an_empty_directory_raises_system_exit(tmp_path):
    """An empty directory has no `*_quant.csv` files to load. Pinned to what
    `_load_intensity_csvs` actually does (observed directly: `csv_files` comes
    back `[]` from the glob, `bin/merge_quant_csvs.py`'s `if not csv_files:`
    branch logs an error and calls `sys.exit(1)`) -- it must not yield a
    phantom file, and it must not silently return an empty list either."""
    from merge_quant_csvs import _load_intensity_csvs

    with pytest.raises(SystemExit, match=r"^1$"):
        _load_intensity_csvs(csvs_dir=tmp_path)
