# benchmarks/tests/regbench/test_paper_schema.py
import math

import pytest

from benchmarks.regbench.paper import schema

EXPECTED = {
    "per_pair", "method_best", "method_summary", "valis_stages", "stare_tiers",
    "stare_stage_peaks", "ashlar_discard", "cost_points", "cost_fits",
    "pipeline_cost_fits", "cpu_scaling", "laptop", "versions", "run_parameters",
    "hosts", "stare_tier_table", "pipeline_params", "cohort_counts",
    "crosscheck_arms", "overlay_rois", "public_landmarks", "public_tissue",
}


def test_the_tables_are_the_paper_tables():
    assert set(schema.TABLES) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_table_opens_with_source_and_names_each_column_once(name):
    cols = schema.columns(name)
    assert cols[0] == "source"
    assert len(cols) == len(set(cols))
    assert set(schema.TABLES[name].key) <= set(cols)
    assert all(c.what for c in schema.TABLES[name].cols)


def test_empty_row_is_all_na_but_the_given_keys():
    row = schema.empty_row("per_pair", source="regbench", dataset="multiplex")
    assert row["source"] == "regbench" and row["dataset"] == "multiplex"
    assert all(v == schema.NA for k, v in row.items() if k not in ("source", "dataset"))
    with pytest.raises(KeyError):
        schema.empty_row("per_pair", not_a_column=1)


def test_write_and_read_round_trip_na(tmp_path):
    p = tmp_path / "laptop.csv"
    row = schema.empty_row("laptop", source="regbench", wall_s=12.5)
    row["peak_rss_gb"] = math.nan
    schema.write(p, "laptop", [row])
    text = p.read_text().splitlines()
    assert text[0].split(",") == schema.columns("laptop")
    back = schema.read(p)[0]
    assert back["wall_s"] == "12.5" and back["peak_rss_gb"] is None
    with pytest.raises(KeyError):
        schema.write(p, "laptop", [{"source": "x", "bogus": 1}])


def test_cpu_scaling_key_names_the_execution():
    assert schema.TABLES["cpu_scaling"].key == ("row", "cpus", "patient_id", "execution")


def test_source_documents_expected_and_the_reexport_is_the_scorer_side_module():
    from benchmarks.regbench import table_schema

    assert "expected" in schema.TABLES["per_pair"].cols[0].what
    assert schema.TABLES is table_schema.TABLES
