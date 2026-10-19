"""The synthetic scaling set end to end (T32): its run list, its COST-only scorer, its
three tables, its refusals and its figures, on a test-only grid of tiny slides with two
fake methods (``stare`` and ``valis_mirage``) run by the real driver."""

from __future__ import annotations

import csv
import math
import types
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import PUBLIC, losses, methods, score, synth, variants
from benchmarks.regbench.cases import list_cases
from benchmarks.regbench.mapping import MOV_TO_REF, PointMap
from benchmarks.regbench.paper import schema
from benchmarks.regbench.paper import synthetic as S
from benchmarks.regbench.paper import tables as T
from benchmarks.tests.regbench import test_synth as ts

SIZES = (64, 128, 256)
TWELVE = {
    "stare",
    "stare_s128_none",
    "stare_s256_none",
    "stare_medium",
    "stare_medium_s128_none",
    "stare_medium_s256_none",
    "stare_low",
    "stare_low_s128_none",
    "stare_low_s256_none",
    "valis_mirage",
    "valis120",
    "valis120_hires",
}


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ---- the run list -------------------------------------------------------------------------


def test_the_set_is_public_cost_only_and_fluorescence():
    assert PUBLIC["synthetic"] == {"kind": "synthetic", "modality": "fluorescence"}
    assert variants.is_cost_only_set("synthetic")
    assert not variants.is_cost_only_set("mcmicro") and not variants.is_cost_only_set(
        None
    )


def test_the_synthetic_run_list_is_exactly_the_twelve_runs():
    got = {variants.run_name(r) for r in variants.runs_for("synthetic")}
    assert got == TWELVE == set(variants.SYNTHETIC_RUNS)


def test_every_other_run_is_excluded_with_the_reason():
    why = dict(variants.excluded_for("synthetic"))
    every = {variants.run_name(r) for r in variants.active_runs()}
    assert set(why) == every - TWELVE
    assert set(why.values()) == {
        "synthetic: STARE (tier x stride, no prep) and VALIS (mirage, 1.2.0 default, "
        "1.2.0 hires) only (user 2026-10-19)"
    }
    rows = score.excluded_rows(["synthetic"])
    assert {r["run"] for r in rows} == every - TWELVE


def test_the_cohort_and_hyreco_run_lists_are_unchanged():
    """No run was added for the set: the 96 runs, the cohort's 56, HyReCo's 50."""
    assert len(variants.RUNS) == 96
    assert len(variants.runs_for(None)) == 56
    assert len(variants.runs_for("hyreco")) == 50
    assert variants.excluded_for(None) == []
    assert variants.runs_for("mcmicro") == variants.runs_for(None)


def test_the_cpu_cross_names_two_runs_of_the_set_at_two_sizes():
    assert set(variants.SYNTHETIC_CROSS_RUNS) <= TWELVE
    assert set(variants.SYNTHETIC_CROSS_SIZES) <= set(synth.GRIDS["default"])


# ---- the stand-in set and two fake methods -----------------------------------------------


def _fake(name, info=None):
    def register(case, work, opts):
        def fn(xy):
            return np.asarray(xy, dtype=float) + 1.0

        yield name, PointMap(fn, MOV_TO_REF), dict(info(case) if info else {})

    return types.SimpleNamespace(
        NAME=name, SCOPE="pair", VARIANTS=(name,), register=register
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("synthset")
    src = ts._source(tmp)
    cases_root = tmp / "cases"
    synth.plan(src, cases_root, sizes=SIZES, tile=32, texture_side=128)
    for i in range(synth.n_units(cases_root)):
        synth.render_unit(cases_root, i)
    synth.finish(cases_root)
    out = tmp / "out"
    cs = list_cases(cases_root, "synthetic")
    for case in cs:
        methods.run_case(_fake("stare"), case, out)
        if case.group != synth.patient_of(64):  # valis_mirage misses the smallest size
            methods.run_case(_fake("valis_mirage"), case, out)
    return cases_root, out, cs


@pytest.fixture(scope="module")
def scored(built):
    cases_root, out, _ = built
    got = score.score(
        cases_root,
        out,
        ["synthetic"],
        expected=variants.expected_variants("synthetic"),
        primary=variants.primary_for("synthetic"),
        run_variants=variants.run_variants(),
    )
    return got["public"]["synthetic"], Path(out) / "tables" / "synthetic"


def test_the_scorer_writes_the_cost_tables_into_tables_synthetic(scored):
    _, tables = scored
    for name in (
        "resources",
        "summary_resources",
        "stare_stage_peaks",
        "excluded_runs",
        "losses",
        *score.SYNTHETIC_TABLES,
    ):
        assert (tables / f"{name}.csv").is_file(), name
    # and never an accuracy table: no pair, nucleus or landmark was scored
    for name in (
        "cells",
        "summary_cells",
        "cells_nuclei",
        "common_block",
        "hyreco_pairs",
        "tissue",
    ):
        assert not (tables / f"{name}.csv").exists(), name
    assert set(score.SYNTHETIC_TABLES) <= set(score.TABLES)


def test_the_scorer_never_reads_pairs_or_nuclei(built, monkeypatch, tmp_path):
    cases_root, out, _ = built

    def boom(*a, **k):
        raise AssertionError("the synthetic scorer scored a pair or a nucleus")

    for name in (
        "score_jobs",
        "_score_case",
        "case_jobs",
        "_score_cells",
        "_score_tissue",
        "_score_landmarks",
    ):
        monkeypatch.setattr(score, name, boom)
    monkeypatch.setattr(score.pairs, "score_pair", boom, raising=False)
    monkeypatch.setattr(score, "load_points", boom)
    score.score(
        cases_root, out, ["synthetic"], expected=variants.expected_variants("synthetic")
    )


def test_synthetic_cost_has_a_row_per_run_and_size(scored):
    _, tables = scored
    rows = schema.read(tables / "synthetic_cost.csv")
    mine = [r for r in rows if r["excluded_reason"] is None]
    stare = [r for r in mine if r["row"] == "stare"]
    assert sorted(int(r["size_px"]) for r in stare) == list(SIZES)
    for r in stare:
        size = int(r["size_px"])
        assert r["status"] == "complete" and r["patient_id"] == synth.patient_of(size)
        assert float(r["input_gpx"]) == pytest.approx(10 * size * size / 1e9)
        assert int(r["input_bytes"]) == 11 * 3 * 2 * size * size
        assert (r["n_channels"], r["n_panels"], r["n_repeats"]) == ("3", "11", "1")
        assert r["n_rounds_ok"] == "10"
        assert r["clock"] and r["unit"] == "patient"
        assert float(r["cpu_h_used"]) > 0 and float(r["peak_rss_gb"]) > 0
        assert r["caveat"] == synth.CAVEAT
    # every Task 31 stem is a column, with clock and unit beside
    for stem in S.costs.STEMS:
        assert stem in rows[0]
    # valis_mirage never ran on s00064: a row that says so, not a silent absence
    v64 = [r for r in mine if r["row"] == "valis_mirage" and r["size_px"] == "64"]
    assert len(v64) == 1 and v64[0]["status"] == "not_run"
    # the ten runs that ran nowhere are rows of their status too
    assert {r["row"] for r in mine} == set(T.terminal_rows("synthetic").values())


def test_excluded_runs_are_na_rows_with_the_reason_in_every_table(scored):
    _, tables = scored
    reason = variants.SYNTHETIC_REASON
    n_ex = len(variants.excluded_for("synthetic"))
    for name in score.SYNTHETIC_TABLES:
        rows = schema.read(tables / f"{name}.csv")
        ex = [r for r in rows if r["excluded_reason"] is not None]
        assert len(ex) == n_ex, name
        assert {r["excluded_reason"] for r in ex} == {reason}
        assert all(r["caveat"] == synth.CAVEAT for r in rows), name
    assert len(_read(tables / "excluded_runs.csv")) == n_ex


def test_the_scaling_fits_three_sizes_and_says_why_not_with_two(scored):
    _, tables = scored
    rows = schema.read(tables / "synthetic_scaling.csv")
    by = {(r["row"], r["quantity"]): r for r in rows if r["excluded_reason"] is None}
    s = by[("stare", "cpu_h_used")]
    assert s["fit_ok"] == "True" and s["n_sizes_used"] == "3"
    assert s["sizes_used"] == "64|128|256" and s["min_n"] == "3"
    assert s["slope_ci_low"] is not None and s["n_repeats"] == "1"
    v = by[("valis_mirage", "cpu_h_used")]
    assert v["fit_ok"] == "False" and v["n_sizes_used"] == "2"
    assert "a fit needs 3" in v["no_fit_reason"]
    nothing = by[("valis120_hires", "wall_s")]
    assert (
        nothing["fit_ok"] == "False" and "no size complete" in nothing["no_fit_reason"]
    )


def test_a_fit_on_a_known_power_law_recovers_its_slope():
    pts = [
        {
            "status": "complete",
            "input_gpx": x,
            "patient_id": f"s{i}",
            "cpu_h_used": 3 * x**1.5,
        }
        for i, x in enumerate((0.1, 0.4, 1.6, 6.4, 25.8))
    ]
    got, pairs = S._fit_of(pts, "cpu_h_used")
    assert got["fit_ok"] and got["slope"] == pytest.approx(1.5)
    assert 10 ** got["intercept"] == pytest.approx(3.0)
    assert len(pairs) == 5
    got2, pairs2 = S._fit_of(pts[:2], "cpu_h_used")
    assert not got2["fit_ok"]
    assert "a fit needs 3" in S._no_fit_reason(got2, pairs2, 2, "cpu_h_used", 2)


def _cohort_points(out, values):
    """A stand-in ``paper/cost_points.csv`` of the cohort: one complete common patient
    per (input_gpx, cpu_h_used) of ``values``, row ``stare``."""
    path = Path(out) / "paper" / "cost_points.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        schema.empty_row(
            "cost_points",
            source="regbench",
            dataset="multiplex",
            family="stare",
            row="stare",
            unit_kind="patient",
            unit="patient",
            unit_id=f"P{i}",
            status="complete",
            in_common=True,
            clock="critical_path_unlimited",
            input_gpx=x,
            cpu_h_used=c,
            wall_s=c * 100,
            peak_rss_gb=1.0,
            gb_h_used=c,
        )
        for i, (x, c) in enumerate(values)
    ]
    schema.write(path, "cost_points", rows)
    return path


def test_the_extrapolation_flag_and_the_gap(tmp_path):
    scaling = [
        schema.empty_row(
            "synthetic_scaling",
            source="regbench",
            dataset="synthetic",
            family="stare",
            row="stare",
            quantity="cpu_h_used",
            quantity_unit="CPU-h",
            fit_ok=True,
            slope=1.0,
            intercept=0.0,
            fit_lo_gpx=0.1,
            fit_hi_gpx=2.0,
            clock="x",
        )
    ]
    # the cohort's patients: 0.5, 1.0 and 4.0 Gpx -> median 1.0 (inside), largest 4.0
    _cohort_points(tmp_path, [(0.5, 0.6), (1.0, 1.2), (4.0, 3.0)])
    rows = {r["at"]: r for r in S.extrapolation_table(scaling, tmp_path)}
    med, big = rows["cohort_median"], rows["cohort_largest"]
    assert med["at_gpx"] == pytest.approx(1.0) and med["extrapolated"] is False
    assert med["fitted"] == pytest.approx(1.0)
    assert med["measured_cohort"] == pytest.approx(1.2)
    assert med["gap"] == pytest.approx(-0.2)
    assert big["at_gpx"] == pytest.approx(4.0) and big["extrapolated"] is True
    assert big["fitted"] == pytest.approx(4.0) and big[
        "measured_cohort"
    ] == pytest.approx(3.0)
    assert big["measured_patient"] == "P2" and big["gap_ratio"] == pytest.approx(4 / 3)
    assert med["cohort_source"] == S.COHORT_POINTS


def test_the_extrapolation_without_a_cohort_says_so(scored):
    _, tables = scored
    rows = [
        r
        for r in schema.read(tables / "synthetic_extrapolation.csv")
        if r["excluded_reason"] is None
    ]
    assert rows and all(r["cohort_source"] == "absent" for r in rows)
    assert all("no cohort cost" in r["note"] for r in rows)
    assert {r["at"] for r in rows} == {"cohort_median", "cohort_largest"}


def test_a_size_a_method_declares_it_cannot_take_is_a_refused_size_row(
    built, monkeypatch
):
    cases_root, out, _ = built
    # VALIS 1.2.0's adapter declares a 100 px limit (no method declares one today)
    monkeypatch.setattr(methods.load("valis"), "MAX_SIDE_PX", 100, raising=False)
    assert (
        score.declared_limit("valis") == 100 and score.declared_limit("stare") is None
    )
    got = score.score(
        cases_root,
        out,
        ["synthetic"],
        expected=variants.expected_variants("synthetic"),
        run_variants=variants.run_variants(),
    )
    rows = got["public"]["synthetic"]["losses"]
    refused = [r for r in rows if r["cause"] == "refused_size"]
    # valis120 / valis120_hires: never run here, their sizes over 100 px refused
    assert refused and {r["run"] for r in refused} == {"valis120", "valis120_hires"}
    assert {r["blame"] for r in refused} == {"input"}
    assert {r["effect"] for r in refused} == {"not_run"}
    assert {r["case_id"].split("_")[0] for r in refused} == {"s00128", "s00256"}
    assert "MAX_SIDE_PX = 100" in refused[0]["detail"]
    # those rounds are not also a not_run row; the 64 px ones still are
    nr = [r for r in rows if r["cause"] == "not_run" and r["run"] == "valis120"]
    assert nr and all("s00064" in r["case_id"] or "case(s)" in r["case_id"] for r in nr)
    assert losses.CAUSES["refused_size"][0] == "input"


def test_the_set_is_never_pooled_with_the_cohort(built, scored):
    cases_root, out, _ = built
    _, tables = scored
    assert score.tables_dir(out, "synthetic") == Path(out) / "tables" / "synthetic"
    # nothing of the set reaches a cohort table
    top = Path(out) / "tables"
    for path in top.glob("*.csv"):
        assert "synthetic" not in path.read_text(), path
    for path in tables.glob("*.csv"):
        for r in _read(path):
            if "dataset" in r and r["dataset"] not in ("", "NA"):
                assert r["dataset"] == "synthetic", (path, r["dataset"])


def test_paper_dataset_synthetic_writes_its_tables_under_paper_synthetic(built, scored):
    _, out, _ = built
    paths = T.write_all(out, "synthetic")
    assert set(paths) == {*S.TABLES, "losses"}
    for name, path in paths.items():
        assert path.parent == Path(out) / "paper" / "synthetic", name
        assert path.is_file()


def test_score_case_and_tissue_skip_the_cost_only_set(built, capsys):
    from benchmarks.regbench import cli

    cases_root, out, _ = built
    assert (
        cli.main(
            [
                "score-case",
                "--cases",
                str(cases_root),
                "--out",
                str(out),
                "--dataset",
                "synthetic",
                "--index",
                "0",
            ]
        )
        == 0
    )
    assert "cost alone" in capsys.readouterr().out


def test_the_cli_scores_the_set_alone(built):
    from benchmarks.regbench import cli

    cases_root, out, _ = built
    assert (
        cli.main(
            [
                "score",
                "--cases",
                str(cases_root),
                "--out",
                str(out),
                "--dataset",
                "synthetic",
            ]
        )
        == 0
    )
    assert (Path(out) / "tables" / "synthetic" / "synthetic_cost.csv").is_file()


def test_the_cross_commands_take_the_set():
    from benchmarks.regbench import cli

    p = cli.build_parser()
    a = p.parse_args(
        ["cpu-scaling", "--point", "4=x", "--to", "y", "--dataset", "synthetic"]
    )
    assert a.dataset == "synthetic"
    a = p.parse_args(
        [
            "subset-cases",
            "--cases",
            "c",
            "--groups",
            "s04096",
            "--to",
            "d",
            "--dataset",
            "synthetic",
        ]
    )
    assert a.dataset == "synthetic"


def test_the_fit_slope_recovers_a_known_power_law_end_to_end(tmp_path):
    """synthetic_scaling from a stand-in summary_resources.csv whose CPU-hours grow as
    gpx^0.8 over five sizes."""
    tables = tmp_path / "tables" / "synthetic"
    tables.mkdir(parents=True)
    rows = []
    for size in synth.GRIDS["default"]:
        gpx = 10 * size * size / 1e9
        rows.append(
            {
                "dataset": "synthetic",
                "group": synth.patient_of(size),
                "method": "stare",
                "complete": "True",
                "launched": "True",
                "n_ok": "10",
                "n_rounds": "10",
                "cpu_h_used": str(2 * gpx**0.8),
                "register_s": "5",
                "peak_rss_gb": "1",
                "moving_gpx": str(gpx),
            }
        )
    score.write_csv(tables / "summary_resources.csv", rows)
    score.write_csv(tables / "resources.csv", [])
    (tables / "excluded_runs.csv").write_text("run,reason\n")
    built_ = S.build(tmp_path)
    fit = [
        r
        for r in built_["synthetic_scaling"]
        if r["row"] == "stare" and r["quantity"] == "cpu_h_used"
    ][0]
    assert fit["fit_ok"] is True and fit["slope"] == pytest.approx(0.8)
    assert fit["n_sizes_used"] == 5
    assert math.isclose(fit["fit_hi_gpx"], 10 * 65536**2 / 1e9)
