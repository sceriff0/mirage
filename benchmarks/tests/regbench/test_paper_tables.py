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
import json

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
    best = {r["family"]: r for r in built["method_best"] if tables.is_headline(r)}
    # `crash` raised on every case: imputed at the unregistered pose, so it cannot beat `rtm`
    assert best["rtm"]["best_row"] == "rtm"
    assert best["stare"]["in_sample_selection"] is True
    assert "median over patients" in best["stare"]["rule"]
    # rtm's never-run P2_r1 is an imputed failure with no pair; so is crash's: a case where
    # EVERY candidate has no pair is the case's, not a method's, and leaves the common set
    assert best["rtm"]["n_cases_common"] == 3
    assert "counts as 0" in best["rtm"]["rule"]


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
    # the case pooling counts every case of the row; the headline (common) only the
    # cases every compared row has
    rows = {(r["family"], r["which"]): r for r in built["method_summary"]
            if r["set"] == "outlines" and r["pooling"] == "case"}
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
        # T31 R11: a tie goes to the name, device-blind: no cost decides it (a GPU row's
        # CPU-hours would otherwise win every tie it is in)
        tie = [x for x in pp if x["row"] == "a"] + [
            {**x, "row": "b"} for x in pp if x["row"] == "a"]
        assert tables.best_of(tie, "f", "multiplex", {})["best_row"] == "a"
        cost = {"a": {"p1": 2.0, "p2": 2.0}, "b": {"p1": 1.0, "p2": 1.0}}
        assert tables.best_of(tie, "f", "multiplex", cost)["best_row"] == "a"
        assert "device-blind" in tables.TIE_BREAK
    finally:
        tables.family_of, tables.terminal_rows = old_f, old_t


def test_summary_has_best_and_primary_rows_and_the_paired_difference(scored):
    built = tables.build(scored, "multiplex")
    rows = {(r["family"], r["which"]): r for r in built["method_summary"]
            if tables.is_headline(r)}
    assert ("stare", "best") in rows and ("stare", "primary") in rows
    assert ("rtm", "primary") not in rows  # no primary row in the toy family
    me = rows[("stare", "best")]
    assert me["d_dice_vs_stare_med"] == 0.0 and me["n_pairs_stare_ge"] == me["n_pairs_vs_stare"]
    assert me["peak_rss_unit"] == "per round" and me["wall_kind"] == "register_s"
    assert me["cpu_model"] == "arm"


def test_terminal_rows_are_the_last_variant_of_each_run():
    rows = tables.terminal_rows("multiplex")
    for run in variants.runs_for("multiplex"):
        if variants.run_name(run) in variants.COST_ONLY:
            # a cost row is in no accuracy table (T31 fix C1): the cost tables add it
            assert variants.run_name(run) not in rows
            continue
        assert rows[variants.run_name(run)] == variants.variants_of(run)[-1]
    assert variants.STARE_LOCAL4 in variants.COST_ONLY
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


def test_an_unreadable_record_is_counted_in_versions_and_run_parameters(scored):
    """Task 28, S13: an unreadable run.json was skipped silently by ``_records``."""
    assert all(r["n_unreadable"] == 0 for r in tables.build(scored, "multiplex")["versions"])
    (path,) = sorted((scored / "stare" / "multiplex").glob("*/run.json"))[:1]
    path.write_text("{trunc")
    built = tables.build(scored, "multiplex")
    v = {r["row"]: r for r in built["versions"]}
    assert v["stare"]["n_unreadable"] == 1 and v["stare"]["n_results"] == 3
    assert v["rtm"]["n_unreadable"] == 0
    p = [r for r in built["run_parameters"] if r["row"] == "stare"]
    assert p and all(r["n_unreadable"] == 1 for r in p)


def test_cohort_counts_per_patient(scored):
    built = tables.build(scored, "multiplex")
    c = {(r["patient_id"], r["row"]): r for r in built["cohort_counts"]}
    assert c[("P1", "stare")]["n_panels"] == 2
    assert c[("P1", "stare")]["n_pairs_min"] <= c[("P1", "stare")]["n_pairs_max"]


def test_the_paper_cli_writes_the_tables(scored, monkeypatch):
    from benchmarks.regbench import cli

    assert cli.main(["paper", "--out", str(scored)]) == 0
    assert (scored / "paper" / "method_best.csv").exists()


def _hand(row, case, patient, dice, excluded=False, imputed=False):
    return {"dataset": "multiplex", "set": "outlines", "row": row, "case_id": case,
            "patient_id": patient, "dice_matched": dice, "excluded_from_summaries": excluded,
            "imputed_initial": imputed, "displacement_readable": False,
            "displacement_um_p50": float("nan"), "tool_version": "1", "n_pairs": 0}


@pytest.fixture
def toy_family(monkeypatch):
    monkeypatch.setattr(tables, "family_of", {"a": "f", "b": "f"}.get)
    monkeypatch.setattr(tables, "terminal_rows", lambda ds: {"a": "a", "b": "b"})


def test_a_nan_dice_counts_as_zero_in_best_of(toy_family):
    """The review's probe: `a` is 0.9 on c1 and has no pair on c2 (an imputed failure);
    `b` is 0.8 / 0.7. Scored on its easy round alone `a` would win; it does not."""
    nan = float("nan")
    pp = [_hand("a", "c1", "p1", 0.9), _hand("a", "c2", "p2", nan, imputed=True),
          _hand("b", "c1", "p1", 0.8), _hand("b", "c2", "p2", 0.7)]
    best = tables.best_of(pp, "f", "multiplex", {})
    assert best["best_row"] == "b" and best["n_cases_common"] == 2
    assert best["criterion_value"] == pytest.approx(0.75) and best["n_dice_zeroed"] == 0
    # a case where EVERY candidate has no pair is left out of the common cases
    pp += [_hand("a", "c3", "p3", nan, imputed=True), _hand("b", "c3", "p3", nan)]
    best = tables.best_of(pp, "f", "multiplex", {})
    assert best["n_cases_common"] == 2 and best["best_row"] == "b"
    # the chosen row's own NaN rounds counted as 0 are counted
    pp2 = [_hand("a", "c1", "p1", 0.9), _hand("a", "c2", "p2", 0.9),
           _hand("b", "c1", "p1", 0.5), _hand("b", "c2", "p2", nan)]
    best = tables.best_of(pp2, "f", "multiplex", {})
    assert best["best_row"] == "a" and best["n_dice_zeroed"] == 0
    pp3 = [_hand("a", "c1", "p1", 0.9), _hand("a", "c2", "p2", nan),
           _hand("b", "c1", "p1", 0.2), _hand("b", "c2", "p2", 0.2)]
    best = tables.best_of(pp3, "f", "multiplex", {})
    assert best["best_row"] == "a" and best["n_dice_zeroed"] == 1


def test_the_paired_difference_has_its_sign():
    nan = float("nan")
    pp = [_hand("x", "c1", "p1", 0.9), _hand("x", "c2", "p2", 0.8),
          _hand("s", "c1", "p1", 0.7), _hand("s", "c2", "p2", 0.7),
          _hand("x", "c3", "p3", nan), _hand("s", "c3", "p3", nan),  # both: the case's
          _hand("x", "c4", "p4", nan), _hand("s", "c4", "p4", 0.5)]  # x has no pair: 0
    got = tables._paired(pp, "x", "s", "multiplex")
    # diffs 0.2, 0.1, -0.5
    assert got["d_dice_vs_stare_med"] == pytest.approx(0.1)
    assert got["n_pairs_vs_stare"] == 3 and got["n_pairs_stare_ge"] == 1


def _sr(method, group, complete=True, **kw):
    base = {"dataset": "multiplex", "group": group, "method": method, "unit": "pair",
            "complete": str(complete), "n_jobs": "", "peak_rss_gb": "1.0",
            "cpu_h_used": "1.0", "register_s": "10.0", "wall_unlimited_parallel_s": "",
            "gpu_h": "0.0", "device": "cpu", "gpu_model": ""}
    base.update({k: str(v) for k, v in kw.items()})
    return base


def test_cost_routes_read_complete_patients_only():
    rows = [
        _sr("j", "p1", n_jobs=40, wall_unlimited_parallel_s=100.0, register_s=1.0,
            peak_rss_gb=3.0, cpu_h_used=2.0),
        _sr("j", "p2", n_jobs=50, wall_unlimited_parallel_s=300.0, register_s=1.0,
            peak_rss_gb=5.0, cpu_h_used=4.0),
        _sr("j", "p3", complete=False, n_jobs=5, wall_unlimited_parallel_s=1.0,
            peak_rss_gb=99.0, cpu_h_used=0.01),
        _sr("g", "p1", unit="patient", peak_rss_gb=8.0, cpu_h_used=6.0, register_s=60.0),
        _sr("r", "p1", peak_rss_gb=2.0, register_s=20.0),
    ]
    j = tables._cost(rows, [], "j")
    assert j["peak_rss_unit"] == "per task (largest job)"
    assert j["wall_kind"] == "time_unlimited_parallel_tiles_s"
    assert j["wall_s_med"] == pytest.approx(200.0)
    assert j["peak_rss_gb_med"] == pytest.approx(4.0) and j["peak_rss_gb_max"] == 5.0
    assert j["cpu_h_used_med"] == pytest.approx(3.0) and j["n_patients_complete"] == 2
    g = tables._cost(rows, [], "g")
    assert g["peak_rss_unit"] == "per patient launch" and g["wall_s_med"] == 60.0
    r = tables._cost(rows, [], "r")
    assert r["peak_rss_unit"] == "per round" and r["wall_kind"] == "register_s"
    assert tables._cpu_by_row(rows)["j"] == {"p1": 2.0, "p2": 4.0}


def test_cpu_by_row_from_the_scorers_file_skips_incomplete_patients(scored):
    rows = tables._cost_rows(scored, "multiplex")
    cpu = tables._cpu_by_row(rows)
    assert set(cpu["rtm"]) == {"P1", "P3"}  # P2 never ran: not complete
    assert set(cpu["stare"]) == {"P1", "P2", "P3"}
    assert "crash" not in cpu


def test_summary_zeroes_nan_dice_and_counts_complete_patients(scored):
    built = tables.build(scored, "multiplex")
    rows = {(r["family"], r["which"]): r for r in built["method_summary"]
            if tables.is_headline(r)}
    by_case = {(r["family"], r["which"]): r for r in built["method_summary"]
               if r["set"] == "outlines" and r["pooling"] == "case"}
    rtm = by_case[("rtm", "best")]
    pp = {r["case_id"]: r["dice_matched"] for r in built["per_pair"]
          if r["row"] == "rtm" and r["set"] == "outlines"}
    want = sorted([pp["P1_r1"], pp["P1_r2"], pp["P3_r1"], 0.0])
    assert rtm["n_dice_zeroed"] == 1 and rtm["dice_med"] == pytest.approx((want[1] + want[2]) / 2)
    # per patient first (the headline pools so, on the common cases): P1's median, P2's 0
    # (its one round has no pair), P3's
    head = rows[("rtm", "best")]
    per = sorted([(pp["P1_r1"] + pp["P1_r2"]) / 2, 0.0, pp["P3_r1"]])
    assert head["pooling"] == "common" and head["dice_med"] == pytest.approx(per[1])
    rtm = head
    assert rtm["n_patients_complete"] == 2 and rows[("stare", "best")]["n_patients_complete"] == 3
    cost = [r for r in tables._cost_rows(scored, "multiplex")
            if r["method"] == "rtm" and r["complete"] == "True"]
    import statistics
    assert rtm["peak_rss_gb_med"] == pytest.approx(
        statistics.median(float(r["peak_rss_gb"]) for r in cost))


def test_excluded_rounds_leave_hosts_and_cost(scored):
    _exclude(scored, "dropme", "P1_r1")
    built = tables.build(scored, "multiplex")
    h = {r["row"]: r for r in built["hosts"]}
    assert h["dropme"]["n_results"] == 3 and h["stare"]["n_results"] == 4


def test_cohort_counts_numbers(scored):
    built = tables.build(scored, "multiplex")
    c = {(r["patient_id"], r["row"]): r for r in built["cohort_counts"]}
    assert (c[("P1", "stare")]["n_pairs_min"], c[("P1", "stare")]["n_pairs_max"]) == (81, 81)
    assert c[("P1", "stare")]["n_pairs_med"] == 81.0
    assert c[("P2", "rtm")]["n_panels"] == 1 and c[("P2", "rtm")]["n_pairs_max"] == 0


def test_stare_tiers_defaults_and_options(scored, monkeypatch):
    monkeypatch.setattr(tables, "family_of", variants.family)
    monkeypatch.setattr(tables, "terminal_rows", lambda ds: {
        n: n for n in ("stare", "stare_medium", "stare_s128_clahe")})
    t = {r["row"]: r for r in tables.build(scored, "multiplex")["stare_tiers"]}
    assert (t["stare"]["tier"], t["stare"]["stride"], t["stare"]["sprep"]) == ("high", 64, "none")
    assert t["stare_medium"]["tier"] == "medium" and t["stare_medium"]["stride"] == 64
    assert (t["stare_s128_clahe"]["stride"], t["stare_s128_clahe"]["sprep"]) == (128, "clahe")


def test_primary_of_reads_the_dataset():
    from benchmarks.regbench import regimes

    public = next(d for d, reg in regimes.DATASET_REGIME.items()
                  if reg != regimes.COHORT_REGIME)
    assert tables.primary_of("valis_mirage", "multiplex") == "valis_mirage"
    assert tables.primary_of("valis_mirage", public) == "valis_mirage"  # runs as shipped
    # ASHLAR has no input rule for a brightfield case: never run there (excluded_for)
    assert tables.primary_of("ashlar", "multiplex") == "ashlar"
    assert tables.primary_of("ashlar", public) is None
    assert tables.primary_of("stare", public) == "stare"


def test_best_of_has_a_per_method_common_set_beside_the_strict_one(toy_family):
    """B2 (M2): a round `a` lost for a cause that is not its own (excluded) takes c2
    out of the STRICT common set for every candidate, but out of `a`'s own set alone
    in the PER_METHOD one. The strict row is the headline; both say which they are."""
    pp = [_hand("a", "c1", "p1", 0.9), _hand("a", "c2", "p2", 0.0, excluded=True,
                                             imputed=True),
          _hand("b", "c1", "p1", 0.8), _hand("b", "c2", "p2", 0.7)]  # fmt: skip
    strict = tables.best_of(pp, "f", "multiplex", {})
    per = tables.best_of(pp, "f", "multiplex", {}, rule="per_method")
    assert strict["common_block"] == "strict" and per["common_block"] == "per_method"
    assert strict["n_cases_common"] == 1
    assert per["best_row"] == "a" and per["n_cases_common"] == 1  # a's own: c1
    built = tables.build_best(pp, "multiplex", {})
    assert {r["common_block"] for r in built["method_best"]} == {"strict"}
    assert {r["common_block"] for r in built["method_best_per_method"]} == {
        "per_method"
    }


def test_the_paper_writes_its_own_losses_csv(scored):
    """M3 (ruling): the paper stage's losses are rows of ``paper/losses.csv`` (same
    columns as tables/losses.csv, stage ``paper``): the points a cost fit dropped
    (S12) and the run.json files the paper tables could not read (S13)."""
    from benchmarks.regbench import losses

    (path,) = sorted((scored / "stare" / "multiplex").glob("*/run.json"))[:1]
    path.write_text("{trunc")
    paths = tables.write_all(scored, "multiplex")
    got = paths["losses"]
    with open(got, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fh.seek(0)
        assert next(csv.reader(fh)) == list(losses.COLUMNS)
    assert rows and {r["stage"] for r in rows} == {"paper"}
    (bad,) = [r for r in rows if r["cause"] == "paper_record_unreadable"]
    assert bad["run"] == "stare" and "1" in bad["detail"]
    assert bad["source"] == "paper/versions.csv"
    fits_rows = [r for r in rows if r["cause"] == "fit_points_dropped"]
    built = tables.build(scored, "multiplex")
    want = [f for f in built["cost_fits"] + built["pipeline_cost_fits"]
            if f["n_dropped"]]  # fmt: skip
    assert len(fits_rows) == len(want) > 0  # no cases folder: no input_gb


def _pt(row, pid, status, cpu, peak):
    return {"row": row, "unit_kind": "patient", "unit_id": pid, "status": status,
            "cpu_h_used": cpu, "peak_rss_gb": peak, "peak_rss_comparable": True}


def test_stare_tiers_cost_is_over_the_common_patients_with_its_interval_clock_and_unit(monkeypatch):
    """T31 fix I5: the tier table's cost follows the cost rules (2, 4, 5): the headline
    is over the common patients, never the row's survivors, with an interval."""
    monkeypatch.setattr(tables, "family_of", variants.family)
    monkeypatch.setattr(tables, "terminal_rows", lambda ds: {"stare_medium": "stare_medium"})
    points = [_pt("stare_medium", "P1", "complete", 1.0, 2.0),
              _pt("stare_medium", "P2", "complete", 3.0, 4.0),
              _pt("stare_medium", "P3", "complete", 100.0, 200.0)]  # not common: left out
    cost = [{"method": "stare_medium", "n_jobs": "5", "group": p} for p in ("P1", "P2", "P3")]
    (r,) = [r for r in tables.stare_tiers([], cost, "multiplex", points, {"P1", "P2"})
            if tables.is_headline(r)]
    assert r["cpu_h_used_common_med"] == 2.0 and r["peak_rss_gb_common_med"] == 3.0
    assert r["cpu_h_used_common_ci_low"] <= 2.0 <= r["cpu_h_used_common_ci_high"]
    assert (r["n_patients_common"], r["n_patients_survivors"]) == (2, 3)
    assert r["clock"] == "critical_path_unlimited" and r["unit"] == "patient"


def test_end_to_end_reads_the_common_patients_the_row_completed_with_intervals():
    """T31 fix I5: a common patient this row did not complete (a twin's or local4's
    failed patient) is no part of its medians; every median has its interval."""
    summary = [{"row": "x", "family": "palom", "row_kind": "cpu_twin", "clock": "whole_process",
                "prior_cpu_h": None, "convert_inside_clock": "no"}]
    points = [_pt("x", "P1", "complete", 1, 1), _pt("x", "P2", "complete", 1, 1),
              _pt("x", "P3", "failed", 1, 1)]
    e2e = [{"method": "x", "group": p, "method_convert_s": s, "estimate_s": s, "write_s": s}
           for p, s in (("P1", 10), ("P2", 20), ("P3", 1000))]
    (r,) = tables.end_to_end(summary, [], [], e2e, [], "multiplex", {"P1", "P2", "P3"}, points)
    assert r["n_patients"] == 2 and r["estimate_s_med"] == 15.0
    assert r["estimate_s_ci_low"] <= 15.0 <= r["estimate_s_ci_high"]
    assert r["unit"] == "patient" and r["clock"] == "whole_process"


def test_the_prior_counts_the_rounds_it_has_no_line_for_and_is_summed_per_patient():
    """T31 fix M8: an affine=stare prior sums STARE's COARSE over the patient's rounds; a
    round with no stare_stage_peaks line makes it a lower bound, counted; and
    with_prior_cpu_h is the median of the per-patient sums, not a median plus a median."""
    res = [{"method": "stare", "group": p, "case_id": c}
           for p, c in (("P1", "a"), ("P1", "b"), ("P2", "c"), ("P3", "d"))]
    peaks = [{"row": "stare", "patient_id": p, "case_id": c, "coarse_cpu_h": h, "coarse_s": 1}
             for p, c, h in (("P1", "a", 1.0), ("P2", "c", 10.0), ("P3", "d", 0.0))]
    got = tables._prior_by_patient("affine_ashlar", [], res, {"P1", "P2", "P3"}, peaks)
    assert got["P1"] == (1.0, 1.0, 1) and got["P2"] == (10.0, 1.0, 0)
    own = {"P1": 0.0, "P2": 0.0, "P3": 10.0}
    # sums 1, 10, 10 -> 10; median + median would be 1 + 0 = 1
    assert tables._with_prior(own, got) == 10.0
    assert tables._prior_by_patient("ashlar", [], res, {"P1"}, peaks) == {}


def test_a_table_side_drop_or_lower_bound_is_a_paper_loss_row():
    """T31 fix I6 (rule 6): what method_summary's medians leave out of a common patient
    the row completed (a figure it has for other patients, a rusage peak) and what is a
    lower bound there is a (patient, run, cause) row of paper/losses.csv, not a count."""
    pts = [dict(_pt("x", "P1", "complete", 1.0, 2.0), in_common=True, gpu_peak_gb=3.0),
           dict(_pt("x", "P2", "complete", None, 2.0), in_common=True, gpu_peak_gb=None,
                lower_bound="unmeasured"),
           dict(_pt("x", "P3", "complete", 1.0, 2.0), in_common=True, gpu_peak_gb=3.0,
                peak_rss_comparable=False),
           dict(_pt("x", "P4", "failed", None, None), in_common=True),  # not in a median
           dict(_pt("x", "P5", "complete", None, None), in_common=False)]  # not common
    built = {k: [] for k in ("cost_fits", "pipeline_cost_fits", "per_pair", "versions",
                             "run_parameters")}
    built.update(cost_points=pts, method_summary=[{"row": "x", "family": "palom"}])
    rows = tables.paper_losses(built, "multiplex")
    got = {(r["run"], r["patient_id"], r["cause"], r["effect"]) for r in rows}
    assert got == {("x", "P2", "cost_figure_missing", "dropped"),
                   ("x", "P2", "cost_lower_bound", "flagged"),
                   ("x", "P3", "cost_peak_not_comparable", "dropped")}
    (miss,) = [r for r in rows if r["cause"] == "cost_figure_missing"]
    assert "cpu_h_used" in miss["detail"] and "gpu_peak_gb" in miss["detail"]
    assert {r["stage"] for r in rows} == {"paper"}
    assert all(r["source"] == "paper/method_summary.csv" for r in rows)


def test_the_tables_keep_the_figures_loss_rows_in_paper_losses_csv(scored):
    """T31 fix M7: each writer of paper/losses.csv replaces its own rows only."""
    from benchmarks.regbench import losses

    paths = tables.write_all(scored, "multiplex")
    old = list(csv.DictReader(open(paths["losses"])))
    plot = losses.row(stage="paper", dataset="multiplex", run="r", cause="plot_placeholder",
                      effect="dropped", detail="fig", source="figures/fig")
    losses.write(paths["losses"], [*old, plot])
    tables.write_all(scored, "multiplex")
    again = list(csv.DictReader(open(paths["losses"])))
    assert [r for r in again if r["cause"] == "plot_placeholder"] == [plot]
    assert len(again) == len(old) + 1


# ── T29: the fair comparison (accuracy side) ─────────────────────────────────
def _row(row, case, patient, dice, set_="outlines", readable=False, disp=float("nan"),
         **kw):
    r = _hand(row, case, patient, dice, **kw)
    r.update(set=set_, displacement_readable=readable, displacement_um_p50=disp)
    return r


def _pool_fixture():
    """a: P1 0.9, 0.8, 0.7 and P2 0.1 (four cases); b misses c3. By case the median is
    0.75; per patient first median(0.8, 0.1) = 0.45; on the common cases (c3 out)
    median(0.85, 0.1) = 0.475."""
    return [_row("a", "c1", "p1", 0.9), _row("a", "c2", "p1", 0.8),
            _row("a", "c3", "p1", 0.7), _row("a", "c4", "p2", 0.1),
            _row("b", "c1", "p1", 0.5), _row("b", "c2", "p1", 0.5),
            _row("b", "c4", "p2", 0.5)]


def test_the_three_poolings_differ_where_cases_and_patients_differ():
    pp = _pool_fixture()
    common = tables.common_sets(pp, "multiplex", ["a", "b"])["outlines"]
    assert common["cases"] == {"c1", "c2", "c4"}
    got = {p: tables._accuracy(pp, "a", "multiplex", "outlines", p, common)[1]
           for p in tables.POOLINGS}
    assert got["case"]["dice_med"] == pytest.approx(0.75)
    assert got["patient"]["dice_med"] == pytest.approx(0.45)
    assert got["common"]["dice_med"] == pytest.approx(0.475)
    assert (got["case"]["n_units"], got["patient"]["n_units"]) == (4, 2)
    assert got["common"]["n_pairs"] == 3 and got["case"]["n_pairs"] == 4
    assert tables.POOLINGS[0] == tables.HEADLINE_POOLING == "common"


def test_dice_is_reported_zero_filled_and_skipped_with_the_count():
    nan = float("nan")
    pp = [_row("a", "c1", "p1", 0.8), _row("a", "c2", "p2", nan, imputed=True),
          _row("a", "c3", "p3", 0.6)]
    acc = tables._accuracy(pp, "a", "multiplex", "outlines", "patient",
                           tables.common_sets(pp, "multiplex", ["a"])["outlines"])[1]
    assert acc["dice_med"] == pytest.approx(0.6) == acc["dice_zero_med"]  # 0.8, 0, 0.6
    assert acc["dice_skip_med"] == pytest.approx(0.7)  # 0.8, 0.6
    assert acc["n_dice_zeroed"] == 1


def test_displacement_has_a_common_readable_version_beside_the_per_row_one():
    pp = [_row("a", "c1", "p1", 0.9, readable=True, disp=1.0),
          _row("a", "c2", "p2", 0.9, readable=True, disp=3.0),
          _row("b", "c1", "p1", 0.9, readable=True, disp=2.0),
          _row("b", "c2", "p2", 0.9, readable=False, disp=9.0)]
    common = tables.common_sets(pp, "multiplex", ["a", "b"])["outlines"]
    assert common["readable"] == {"c1"}
    acc = tables._accuracy(pp, "a", "multiplex", "outlines", "case", common)[1]
    assert acc["disp_um_med"] == pytest.approx(2.0) and acc["n_readable"] == 2
    assert acc["disp_um_common_med"] == pytest.approx(1.0) and acc["n_readable_common"] == 1


def test_the_paired_difference_is_per_patient_with_its_unit_and_tie_rule():
    """The paired table's unit is the patient (stated in `paired_unit`), and a tie goes
    to STARE: `n_units_stare_ge` counts b >= a (WT5)."""
    pp = [_row("x", "c1", "p1", 0.9), _row("x", "c2", "p1", 0.7),
          _row("s", "c1", "p1", 0.6), _row("s", "c2", "p1", 0.8),  # p1: 0.8 vs 0.7
          _row("x", "c3", "p2", 0.5), _row("s", "c3", "p2", 0.5)]  # p2: a tie
    common = tables.common_sets(pp, "multiplex", ["x", "s"])["outlines"]
    got = tables._paired(pp, "x", "s", "multiplex", "outlines", "patient", common)
    assert got["paired_unit"] == "patient" and got["n_units_vs_stare"] == 2
    assert got["d_dice_vs_stare_med"] == pytest.approx(0.05)  # median(0.1, 0.0)
    assert got["n_units_stare_ge"] == 1  # the tie only
    assert got["n_pairs_vs_stare"] == 3
    by_case = tables._paired(pp, "x", "s", "multiplex", "outlines", "case", common)
    assert by_case["paired_unit"] == "case" and by_case["n_units_stare_ge"] == 2


def test_every_summary_row_has_its_set_and_pooling_and_the_headline_first(scored):
    built = tables.build(scored, "multiplex")
    rows = [r for r in built["method_summary"] if not tables.is_cost_only(r["row"])]
    seen = {(r["row"], r["which"], r["set"], r["pooling"]) for r in rows}
    for row, which in {(r["row"], r["which"]) for r in rows}:
        assert {(s, p) for rr, w, s, p in seen if (rr, w) == (row, which)} == {
            (s, p) for s in ("outlines", "nuclei") for p in tables.POOLINGS}
    head = [r for r in rows if tables.is_headline(r)]
    assert head and all((r["set"], r["pooling"]) == ("outlines", "common") for r in head)
    for name in ("valis_stages", "stare_tiers", "cohort_counts", "method_best"):
        assert "set" in schema.columns(name), name
    for name in ("valis_stages", "stare_tiers"):
        assert "pooling" in schema.columns(name), name
    assert {r["unit"] for r in built["method_best"]} == {"patient"}
    assert {r["set"] for r in built["method_best"]} == {"outlines", "nuclei"}
    pp = [r for r in built["per_pair"] if r["row"] == "stare" and r["n_pairs"] > 0]
    assert pp and all(r["pair_fraction_sym"] == pytest.approx(
        2 * r["n_pairs"] / (r["n_ref"] + r["n_mov"])) for r in pp)
    assert "pair_fraction_sym_med" in head[0] and "pair_fraction_maxn_med" in head[0]


def test_a_best_row_carries_its_candidates_and_says_it_is_a_ceiling(scored):
    built = tables.build(scored, "multiplex")
    best = {r["family"]: r for r in built["method_best"] if tables.is_headline(r)}
    for r in built["method_summary"]:
        if r["which"] == "best":
            assert r["n_candidates"] == best[r["family"]]["n_candidates"] == 2
            assert r["in_sample_selection"] is True and "ceiling" in r["selection"]
        elif r["which"] == "primary":
            assert r["n_candidates"] == schema.NA and r["in_sample_selection"] is False


def test_n_candidates_is_every_terminal_row_of_the_family_on_the_cohort():
    """WT4: the real family map, not a toy one: every terminal row of a family that has
    a row competes (the audit's counts, WD1)."""
    # stare: 27 since the full cross on the cohort (tier x stride x level, 2026-10-20)
    want = {"ashlar": 12, "affine_ashlar": 12, "stare": 27, "valis_mirage": 9,
            "deeperhistreg": 8, "valis120": 2, "palom": 1}
    rows = tables.terminal_rows("multiplex").values()
    pp = [_hand(t, "c1", "p1", 0.5) for t in rows]
    for fam, n in want.items():
        assert tables.best_of(pp, fam, "multiplex", {})["n_candidates"] == n, fam
    # a candidate with no row on the dataset does not compete
    assert tables.best_of(pp[1:], variants.family(pp[0]["row"]), "multiplex", {})[
        "n_candidates"] == want[variants.family(pp[0]["row"])] - 1


def test_deeperhistregs_primary_is_its_best_as_drawn_row():
    """Ruling 193 (user 2026-10-18 'the best one as_drawn'): DeeperHistReg's primary is
    the dataset's method_best row, said and flagged in-sample; variants.PRIMARY's
    placeholder is not what the tables use."""
    rows = [t for t in tables.terminal_rows("multiplex").values()
            if variants.family(t) == "deeperhistreg"]
    pp = [_hand(t, "c1", "p1", 0.9 if t == "deeperhistreg_fluo_cpu" else 0.5) for t in rows]
    best = [tables.best_of(pp, "deeperhistreg", "multiplex", {})]
    assert best[0]["best_row"] == "deeperhistreg_fluo_cpu"
    assert tables.primary_row("deeperhistreg", "multiplex", best) == "deeperhistreg_fluo_cpu"
    assert tables.primary_row("stare", "multiplex", best) == "stare"
    assert "user 2026-10-18" in tables.BEST_IS_PRIMARY["deeperhistreg"]
    # without a best row it falls back to variants' placeholder, never to nothing
    assert tables.primary_row("deeperhistreg", "multiplex", []) == variants.PRIMARY_DEEPERHISTREG


def test_versions_hold_the_image_tag_digest_python_and_stardist(scored):
    """T29 (M7): what ran, as far as anything records it; `not recorded` where nothing
    does (never a guess)."""
    built = tables.build(scored, "multiplex")
    v = {r["row"]: r for r in built["versions"]}
    assert v["stare"]["image_tag"] == "docker://bolt3x/mirage-stare-regbench:1.0.0"
    assert v["stare"]["image_digest"] == tables.NOT_RECORDED
    assert v["stare"]["python"] == tables.NOT_RECORDED
    assert v["stare"]["stardist_image"] == "bolt3x/mirage-stardist:1.0.0"
    assert v["stare"]["stardist_version"] == tables.NOT_RECORDED
    assert v["crash"]["image_tag"] == tables.NOT_RECORDED  # a toy row: no launcher pin
    # a record that says them is read
    for path in sorted((scored / "stare" / "multiplex").glob("*/run.json")):
        rec = json.loads(path.read_text())
        rec.setdefault("info", {})["python"] = "3.11.9"
        rec["image_digest"] = "sha256:abc"
        path.write_text(json.dumps(rec))
    v = {r["row"]: r for r in tables.build(scored, "multiplex")["versions"]}
    assert v["stare"]["python"] == "3.11.9" and v["stare"]["image_digest"] == "sha256:abc"


def test_an_excluded_run_is_an_na_row_with_its_reason(scored, monkeypatch):
    """T29 (M8): a run of excluded_runs.csv is in the dataset's paper tables as a row of
    NA with the reason, never silently absent."""
    monkeypatch.setattr(tables, "family_of", lambda v: {
        **FAMILY, "stare_gone": "stare", "palom_x": "palom", "stare_medium": "stare",
        "valis_mirage_low_m0": "valis_mirage"}.get(v))
    with open(scored / "tables" / "excluded_runs.csv", "w", newline="") as fh:
        fh.write("run,reason\nstare_gone,not on this regime\npalom_x,defaults only\n"
                 "stare_medium,one tier only\nvalis_mirage_low_m0,defaults only\n")
    built = tables.build(scored, "multiplex")
    gone = {"stare_gone", "palom_x", "stare_medium", "valis_mirage_low_m0"}
    na = (None, schema.NA)
    # method_summary: one NA row per (set, pooling) slot, each with the reason
    ex = [r for r in built["method_summary"] if r["which"] == "excluded"]
    slots = {(r["set"], r["pooling"]) for r in ex}
    assert slots == {(st, p) for st in tables.SETS for p in tables.POOLINGS}
    assert {r["row"] for r in ex} == gone and len(ex) == len(gone) * len(slots)
    s = {r["row"]: r for r in ex if tables.is_headline(r)}
    assert s["stare_gone"]["excluded_reason"] == "not on this regime"
    assert s["stare_gone"]["family"] == "stare" and s["stare_gone"]["dice_med"] in na
    assert all(r["excluded_reason"] not in na for r in ex)
    # valis_stages and stare_tiers: the family's excluded runs, a row of NA per slot,
    # with the settings the method module gives them
    vs = [r for r in built["valis_stages"] if r["row"] == "valis_mirage_low_m0"]
    assert {(r["set"], r["pooling"]) for r in vs} == slots
    assert all(r["excluded_reason"] == "defaults only" and r["dice_med"] in na for r in vs)
    assert {(r["memory_mode"], r["micro_reg"]) for r in vs} == {("low", "0")}
    st = [r for r in built["stare_tiers"] if r["row"] == "stare_medium"]
    assert {(r["set"], r["pooling"]) for r in st} == slots
    assert {(r["tier"], r["excluded_reason"]) for r in st} == {("medium", "one tier only")}
    assert not [r for r in built["stare_tiers"] if r["row"] == "palom_x"]
    # cohort_counts: one row per set; run_parameters: the module's settings, with reason
    cc = [r for r in built["cohort_counts"] if r["row"] == "stare_medium"]
    assert {r["set"] for r in cc} == set(tables.SETS)
    assert all(r["excluded_reason"] == "one tier only" and r["n_panels"] in na for r in cc)
    rp = [r for r in built["run_parameters"] if r["row"] == "stare_medium"]
    assert {r["key"]: r["value"] for r in rp}["tier"] == "medium"
    assert all(r["excluded_reason"] == "one tier only" for r in rp)
    (toy,) = [r for r in built["run_parameters"] if r["row"] == "palom_x"]
    assert toy["key"] in na and toy["excluded_reason"] == "defaults only"
    v = {r["row"]: r for r in built["versions"]}
    assert v["palom_x"]["excluded_reason"] == "defaults only"
    assert v["palom_x"]["image_tag"] == tables.NOT_RECORDED
    best = {(r["family"], r["set"]): r for r in built["method_best"]}
    # a family with other runs keeps its best row and names what was left out
    assert best[("stare", "outlines")]["best_row"] not in (None, schema.NA, "stare_gone")
    assert "stare_gone: not on this regime" in best[("stare", "outlines")]["excluded_runs"]
    # a family with no run left is a row of NA with the reason, in each set
    for st in tables.SETS:
        assert best[("palom", st)]["best_row"] in (None, schema.NA)
        assert best[("palom", st)]["excluded_runs"] == "palom_x: defaults only"
    assert "stare_gone: not on this regime" in best[("stare", "nuclei")]["excluded_runs"]
    # every table still has the schema's columns
    tables.write_all(scored, "multiplex")


def test_a_landmark_datasets_excluded_runs_are_na_rows_too(tmp_path):
    """T29 (M8) on a landmark dataset: the runs its excluded_runs.csv names (a public
    dataset's list: variants.excluded_for) are NA rows with the reason."""
    from benchmarks.regbench.paper import landmarks

    t = tmp_path / "tables" / "hyreco"
    t.mkdir(parents=True)
    (t / "hyreco_pairs.csv").write_text("method,pair_id\n")
    ex = variants.excluded_for("hyreco")
    assert ex  # the ruling of 2026-10-19: every method but STARE runs its defaults only
    with open(t / "excluded_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "reason"])
        w.writerows(ex)
    built = landmarks.build(tmp_path, "hyreco")
    got = {r["row"]: r for r in built["public_landmarks"]}
    name, reason = ex[0]
    final = list(variants.variants_of(variants.run_of(name)))[-1]
    assert got[final]["excluded_reason"] == reason and got[final]["set"] == "all_pairs"
    assert len(got) == len(ex)
    fams = {r["family"] for r in built["method_best"]}
    assert tables.family_of(final) in fams
    for name in landmarks.LANDMARK_WRITTEN:
        for r in built[name]:
            assert set(r) == set(schema.columns(name)), name
