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
    rows = {(r["family"], r["which"]): r for r in built["method_summary"]}
    rtm = rows[("rtm", "best")]
    pp = {r["case_id"]: r["dice_matched"] for r in built["per_pair"]
          if r["row"] == "rtm" and r["set"] == "outlines"}
    want = sorted([pp["P1_r1"], pp["P1_r2"], pp["P3_r1"], 0.0])
    assert rtm["n_dice_zeroed"] == 1 and rtm["dice_med"] == pytest.approx((want[1] + want[2]) / 2)
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
    (r,) = tables.stare_tiers([], cost, "multiplex", points, {"P1", "P2"})
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
