# benchmarks/tests/regbench/test_paper_schema.py
import math

import pytest

from benchmarks.regbench.paper import schema

EXPECTED = {
    "per_pair", "method_best", "method_best_per_method", "method_summary", "valis_stages", "stare_tiers",
    "stare_stage_peaks", "ashlar_discard", "cost_points", "cost_fits",
    "pipeline_cost_fits", "cpu_scaling", "laptop", "versions", "run_parameters",
    "hosts", "stare_tier_table", "pipeline_params", "cohort_counts",
    "crosscheck_arms", "overlay_rois", "public_landmarks", "public_landmark_pairs",
    "public_tissue",
    "end_to_end",
    # T32: the synthetic scaling set (cost only)
    "synthetic_cost", "synthetic_scaling", "synthetic_extrapolation",
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


def test_cpu_scaling_key_names_the_execution_and_the_repeat():
    # T31 R13: two repeats per point, never one row over the other
    assert schema.TABLES["cpu_scaling"].key == ("row", "cpus", "patient_id", "execution", "repeat")


def test_source_documents_expected_and_the_reexport_is_the_scorer_side_module():
    from benchmarks.regbench import table_schema

    assert "expected" in schema.TABLES["per_pair"].cols[0].what
    assert schema.TABLES is table_schema.TABLES


def test_a_duplicate_table_name_is_refused():
    from benchmarks.regbench import table_schema

    name, table = table_schema._t("x", ("dataset",), "w", ("dataset", "", ""))
    with pytest.raises(ValueError):
        table_schema._unique([(name, table), (name, table)])


def test_star_import_exports_only_the_interface():
    from benchmarks.regbench import table_schema

    assert set(table_schema.__all__) == {
        "NA", "Col", "Table", "TABLES", "columns", "empty_row", "write", "read"}


def test_cell_values_and_a_failed_write_leaves_the_file(tmp_path):
    p = tmp_path / "laptop.csv"
    row = schema.empty_row("laptop", source="regbench")
    for v, want in [(0, "0"), (0.0, "0.0"), ("0", "0"), ("", None), (None, None),
                    (math.nan, None)]:
        row["wall_s"] = v
        schema.write(p, "laptop", [row])
        assert schema.read(p)[0]["wall_s"] == want, v
    before = p.read_bytes()
    with pytest.raises(KeyError):
        schema.write(p, "laptop", [row, {"source": "x", "bogus": 1}])
    assert p.read_bytes() == before


# T31 fix I5: rules 4 and 5 hold in EVERY paper table, not only method_summary
COST_UNITS = {"CPU-h", "GB-h", "GPU-h", "s", "GB"}
# a GB that is no measured cost: a configured request, or the machine's memory
NOT_A_COST = {("stare_tier_table", "reg_tile_mem_gb"), ("stare_tier_table", "stitch_mem_gb"),
              ("laptop", "mem_total_gb")}
_SUFFIXES = ("_common_med", "_survivors_med", "_common_ci_low", "_common_ci_high",
             "_survivors_ci_low", "_survivors_ci_high", "_ci_low", "_ci_high", "_med", "_max",
             "_p50")


def _stem(name):
    for suf in _SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_cost_column_names_its_clock_and_unit_and_a_median_its_interval(name):
    cols = {c.name: c for c in schema.TABLES[name].cols}
    bad = []
    for c in cols.values():
        if c.unit not in COST_UNITS or (name, c.name) in NOT_A_COST:
            continue
        stem = _stem(c.name)
        if not {"clock", f"{stem}_clock"} & cols.keys():
            bad.append(f"{c.name}: no clock")
        if not {"unit", f"{stem}_unit"} & cols.keys():
            bad.append(f"{c.name}: no unit")
        if c.name.endswith("_med") and "DEPRECATED" not in c.what:
            base = c.name[: -len("_med")]
            # its own interval, or it is the alias of a headline that has one
            if not ({f"{base}_ci_low", f"{base}_ci_high"} <= cols.keys()
                    or {f"{base}_common_ci_low", f"{base}_common_ci_high"} <= cols.keys()):
                bad.append(f"{c.name}: a median with no interval")
    assert not bad, bad
