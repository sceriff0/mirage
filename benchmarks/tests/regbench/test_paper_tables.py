# benchmarks/tests/regbench/test_paper_tables.py
"""The paper tables (``benchmarks.regbench.paper.tables``), on the scored test cohort.

``scored_cohort`` has the rows ``stare``, ``rtm``, ``tiles``, ``blocks``, ``crash`` and
``dropme`` (and ``ghost``, expected and never run). They are put in two toy families:
``stare`` (``stare``, ``dropme``) and ``rtm`` (``rtm``, ``crash``). ``crash`` raised on
every case and ``rtm`` was never run on ``P2_r1``: both are imputed failures, and an
imputed failure is KEPT. No row of the cohort is excluded from the summaries, so the test
that needs one marks it in ``cells.csv``.
"""
import csv

import pytest

from benchmarks.regbench import score, variants
from benchmarks.regbench.paper import schema, tables
from benchmarks.tests.regbench import scored_cohort

FAMILY = {"stare": "stare", "dropme": "stare", "rtm": "rtm", "crash": "rtm"}


@pytest.fixture
def scored(tmp_path, monkeypatch):
    root, out, made = scored_cohort.build(tmp_path)
    score.score(root, out, **scored_cohort.SCORE_KW)
    monkeypatch.setattr(tables, "family_of", lambda v: FAMILY.get(v))
    monkeypatch.setattr(tables, "terminal_rows", lambda ds: {k: k for k in FAMILY})
    return out


def _exclude(out, method, case_id):
    """Mark one round of ``method`` in both cells tables as lost for a cause that is not
    the method's (``excluded_from_summaries``), as the scorer would."""
    for name in ("cells.csv", "cells_nuclei.csv"):
        path = out / "tables" / name
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
            cols = list(rows[0])
        for r in rows:
            if r["method"] == method and r["case_id"] == case_id:
                r.update(imputed_initial="True", cause="node_failure",
                         excluded_from_summaries="True", dice_matched="nan")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


def test_every_table_is_written_with_the_schemas_columns(scored):
    tables.write_all(scored, "multiplex")
    for name in tables.WRITTEN:
        header = (scored / "paper" / f"{name}.csv").read_text().splitlines()[0].split(",")
        assert header == schema.columns(name)


def test_every_row_says_regbench(scored):
    built = tables.build(scored, "multiplex")
    assert set(built) == set(tables.WRITTEN)
    for name, rows in built.items():
        assert all(r["source"] == "regbench" for r in rows), name


def test_best_counts_imputed_rounds_and_leaves_out_excluded(scored):
    built = tables.build(scored, "multiplex")
    best = {r["family"]: r for r in built["method_best"]}
    # `crash` raised on every case: imputed at the unregistered pose, so it cannot beat `rtm`
    assert best["rtm"]["best_row"] == "rtm"
    assert best["stare"]["in_sample_selection"] is True
    assert "median over patients" in best["stare"]["rule"]
    # rtm's never-run P2_r1 is an imputed failure: it COUNTS, so the common cases are 4
    assert best["rtm"]["n_cases_common"] == 4


def test_an_excluded_row_is_left_out_and_an_imputed_one_kept(scored):
    _exclude(scored, "dropme", "P1_r1")
    built = tables.build(scored, "multiplex")
    pp = [r for r in built["per_pair"] if r["set"] == "outlines"]
    # the per-pair table keeps every row, the excluded one flagged
    gone = [r for r in pp if r["row"] == "dropme" and r["case_id"] == "P1_r1"]
    assert len(gone) == 1 and gone[0]["excluded_from_summaries"] is True
    # best_of compares on the cases every candidate has a NON-excluded row on
    best = tables.best_of(pp, "stare", "multiplex", {})
    assert best["n_cases_common"] == 3 and best["best_row"] == "dropme"
    rows = {(r["family"], r["which"]): r for r in built["method_summary"]}
    assert rows[("stare", "best")]["row"] == "dropme"
    assert rows[("stare", "best")]["n_pairs"] == 3  # P1_r1 left out
    assert rows[("stare", "primary")]["n_pairs"] == 4
    # rtm's imputed (never-run) P2_r1 is KEPT: four cases, three registered
    rtm = rows[("rtm", "best")]
    assert rtm["row"] == "rtm" and rtm["n_pairs"] == 4 and rtm["n_ok"] == 3
    imputed = [r for r in pp if r["row"] == "rtm" and r["case_id"] == "P2_r1"]
    assert imputed[0]["imputed_initial"] is True
    assert imputed[0]["excluded_from_summaries"] is False


def test_best_of_on_hand_rows_excluded_never_decides():
    """An excluded row with a high Dice would win if it counted; it does not."""

    def r(row, case, patient, dice, excluded=False, imputed=False):
        return {"dataset": "multiplex", "set": "outlines", "row": row, "case_id": case,
                "patient_id": patient, "dice_matched": dice,
                "excluded_from_summaries": excluded, "imputed_initial": imputed}

    pp = [
        r("a", "c1", "p1", 0.80), r("a", "c2", "p2", 0.80), r("a", "c3", "p3", 0.80),
        r("b", "c1", "p1", 0.70), r("b", "c2", "p2", 0.70),
        r("b", "c3", "p3", 0.99, excluded=True, imputed=True),
    ]
    fam = {"a": "f", "b": "f"}
    old_f, old_t = tables.family_of, tables.terminal_rows
    tables.family_of, tables.terminal_rows = fam.get, lambda ds: {"a": "a", "b": "b"}
    try:
        best = tables.best_of(pp, "f", "multiplex", {})
        assert best["n_cases_common"] == 2 and best["best_row"] == "a"
        # an imputed failure with a low Dice counts against its row
        pp2 = pp[:-1] + [r("b", "c3", "p3", 0.10, imputed=True)]
        pp2[0] = r("a", "c1", "p1", 0.65)
        best = tables.best_of(pp2, "f", "multiplex", {})
        assert best["n_cases_common"] == 3
        # a: patients 0.65, 0.80, 0.80 -> 0.80; b: 0.70, 0.70, 0.10 -> 0.70
        assert best["best_row"] == "a" and best["criterion_value"] == pytest.approx(0.80)
        # ties go to the lower CPU-hours, then to the name
        tie = [x for x in pp if x["row"] == "a"] + [
            {**x, "row": "b"} for x in pp if x["row"] == "a"]
        assert tables.best_of(tie, "f", "multiplex", {})["best_row"] == "a"
        cost = {"a": [2.0, 2.0], "b": [1.0, 1.0]}
        assert tables.best_of(tie, "f", "multiplex", cost)["best_row"] == "b"
    finally:
        tables.family_of, tables.terminal_rows = old_f, old_t


def test_summary_has_best_and_primary_rows_and_the_paired_difference(scored):
    built = tables.build(scored, "multiplex")
    rows = {(r["family"], r["which"]): r for r in built["method_summary"]}
    assert ("stare", "best") in rows and ("stare", "primary") in rows
    assert ("rtm", "primary") not in rows  # no primary row in the toy family
    me = rows[("stare", "best")]
    assert me["d_dice_vs_stare_med"] == 0.0 and me["n_pairs_stare_ge"] == me["n_pairs_vs_stare"]
    assert me["peak_rss_unit"] == "per round" and me["wall_kind"] == "register_s"
    assert me["cpu_model"] == "arm"


def test_terminal_rows_are_the_last_variant_of_each_run():
    rows = tables.terminal_rows("multiplex")
    for run in variants.runs_for("multiplex"):
        assert rows[variants.run_name(run)] == variants.variants_of(run)[-1]
    # DeeperHistReg has ONE result per run: the run's own name (ruling 2026-10-17)
    assert rows["deeperhistreg_hr_gpu"] == "deeperhistreg_hr_gpu"
    assert not any("asdrawn" in v for v in rows.values())
    assert rows["valis_mirage"] == "valis_mirage" and rows["palom"] == "palom"


def test_valis_stages_are_honest_about_the_depth(scored, monkeypatch):
    """R18: depth 1 is micro-RIGID only (inside the `_rigid` row); a non-rigid `micro`
    stage exists only at depth 2; at depth 0 there is no micro stage at all."""
    monkeypatch.setattr(tables, "family_of", variants.family)
    monkeypatch.setattr(
        tables, "terminal_rows",
        lambda ds: {n: n for n in ("valis_mirage", "valis_mirage_high_m1",
                                   "valis_mirage_low_m0")})
    built = tables.build(scored, "multiplex")
    st = {(r["row"], r["stage"]): r for r in built["valis_stages"]}
    assert st[("valis_mirage", "micro")]["stage_row"] == "valis_mirage"
    assert st[("valis_mirage", "micro_rigid")]["stage_row"] == "valis_mirage_rigid"
    assert st[("valis_mirage", "rigid")]["stage_row"] == schema.NA
    assert st[("valis_mirage_high_m1", "micro_rigid")]["stage_row"] == "valis_mirage_high_m1_rigid"
    assert st[("valis_mirage_high_m1", "micro")]["stage_row"] == schema.NA
    assert st[("valis_mirage_low_m0", "rigid")]["stage_row"] == "valis_mirage_low_m0_rigid"
    assert st[("valis_mirage_low_m0", "micro_rigid")]["stage_row"] == schema.NA
    assert st[("valis_mirage_low_m0", "micro")]["stage_row"] == schema.NA
    assert st[("valis_mirage_low_m0", "nonrigid")]["micro_reg"] == "0"
    assert st[("valis_mirage_high_m1", "initial")]["stage_row"] == "initial"
    allowed = schema.TABLES["valis_stages"].cols[schema.columns("valis_stages").index("stage")]
    for stage in ("micro_rigid", "micro"):
        assert stage in allowed.what


def test_per_pair_stage_names(scored):
    assert tables.stage_of("valis_mirage_high_m1_rigid") == "micro_rigid"
    assert tables.stage_of("valis_mirage_low_m0_rigid") == "rigid"
    assert tables.stage_of("valis_mirage_high_m1") == "nonrigid"
    assert tables.stage_of("valis_mirage") == "micro"
    assert tables.stage_of("valis_mirage_nonrigid") == "nonrigid"
    assert tables.stage_of("stare_rigid") == "rigid"
    assert tables.stage_of("stare") == "final"
    assert tables.stage_of("initial") == "initial"
    assert tables.stage_of("deeperhistreg_hr_gpu") == "final"


def test_stare_tier_table_is_s3():
    rows = {r["tier"]: r for r in tables.stare_tier_table()}
    assert rows["medium"]["halo"] == 192 and rows["low"]["coarse_max_dim"] == 512
    assert rows["high"]["read_box_px_s64"] == 2432
    assert rows["high"]["reg_tile_mem_gb"] == 4 and rows["low"]["stitch_mem_gb"] == 4


def test_pipeline_params_lists_every_schema_param():
    rows = tables.pipeline_params()
    names = {r["name"] for r in rows}
    assert {"registration_method", "reg_tiled_mode", "memory_mode", "pixel_size"} <= names
    assert len(names) == len(rows)


def test_versions_and_run_parameters_and_hosts(scored):
    built = tables.build(scored, "multiplex")
    v = {r["row"]: r for r in built["versions"]}
    assert v["stare"]["n_results"] == 4 and v["stare"]["image"] == "REGBENCH_SIF_STARE_REGBENCH"
    assert v["crash"]["tool_version"] == schema.NA
    h = {r["row"]: r for r in built["hosts"]}
    assert h["stare"]["cpu_model"] == "arm"
    p = [r for r in built["run_parameters"] if r["row"] == "dropme"]
    assert p and p[0]["from"].startswith("the run's options")


def test_cohort_counts_per_patient(scored):
    built = tables.build(scored, "multiplex")
    c = {(r["patient_id"], r["row"]): r for r in built["cohort_counts"]}
    assert c[("P1", "stare")]["n_panels"] == 2
    assert c[("P1", "stare")]["n_pairs_min"] <= c[("P1", "stare")]["n_pairs_max"]


def test_the_paper_cli_writes_the_tables(scored, monkeypatch):
    from benchmarks.regbench import cli

    assert cli.main(["paper", "--out", str(scored)]) == 0
    assert (scored / "paper" / "method_best.csv").exists()
