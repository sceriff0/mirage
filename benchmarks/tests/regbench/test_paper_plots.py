# benchmarks/tests/regbench/test_paper_plots.py
"""The paper's plots (``benchmarks.regbench.paper.plots``) on a tiny ``paper/`` folder:
three families with rows, a fourth with none, five patients of two cases each."""
import csv
import math

import pytest

pytest.importorskip("matplotlib")

from benchmarks.regbench.paper import plots, schema, tables  # noqa: E402

FAMILY = {"stare": "stare", "stare_medium": "stare", "valis_mirage": "valis_mirage",
          "ashlar": "ashlar", "ashlar_clean": "ashlar", "initial": None}
PATIENTS = [f"P{i}" for i in range(5)]
CASES = [(p, f"{p}_r{k}") for p in PATIENTS for k in (1, 2)]


def _pair(row, patient, case, dice, **kw):
    base = dict(source="regbench", dataset="multiplex", patient_id=patient, panel="a",
                case_id=case, row=row, family=FAMILY[row], set="outlines", stage="final",
                dice_matched=dice, displacement_um_p50=2.0 + dice, displacement_readable=True,
                n_pairs=50, excluded_from_summaries=False, imputed_initial=False)
    base.update(kw)
    return schema.empty_row("per_pair", **base)


@pytest.fixture
def paper(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "family_of", lambda v: FAMILY.get(v))
    monkeypatch.setattr(tables, "terminal_rows",
                        lambda ds: {k: k for k in FAMILY if k != "initial"})
    d = tmp_path / "paper"
    d.mkdir()
    pp = []
    level = {"stare": 0.9, "stare_medium": 0.8, "valis_mirage": 0.7, "ashlar": 0.5, "ashlar_clean": 0.6}
    for row, base in level.items():
        for i, (p, c) in enumerate(CASES):
            dice = base - 0.01 * i
            if row == "ashlar" and i == 0:
                dice = math.nan  # no matched pair: counted as 0
            pp.append(_pair(row, p, c, dice))
    # a case with no matched pair on every row is left out
    for row in level:
        pp.append(_pair(row, "P0", "P0_r3", math.nan))
    schema.write(d / "per_pair.csv", "per_pair", pp)
    best = [
        schema.empty_row("method_best", source="regbench", dataset="multiplex", family="stare",
                         best_row="stare"),
        schema.empty_row("method_best", source="regbench", dataset="multiplex",
                         family="valis_mirage", best_row="valis_mirage"),
        schema.empty_row("method_best", source="regbench", dataset="multiplex", family="ashlar",
                         best_row="ashlar_clean"),
        schema.empty_row("method_best", source="regbench", dataset="multiplex", family="palom",
                         best_row=None),
    ]
    schema.write(d / "method_best.csv", "method_best", best)
    schema.write(d / "method_summary.csv", "method_summary", [
        schema.empty_row("method_summary", source="regbench", dataset="multiplex", family="stare",
                         row="stare", which="best", peak_rss_unit="per task (largest job)",
                         device="cpu"),
    ])
    vs = []
    for tier, depth in (("high", 2), ("low", 0)):
        row = f"valis_{tier}"
        for stage in plots.STAGES:
            has = stage in ("initial", "nonrigid") or (stage == "micro" and depth == 2) \
                or (stage == "rigid" and depth == 0) or (stage == "micro_rigid" and depth >= 1)
            vs.append(schema.empty_row("valis_stages", source="regbench", dataset="multiplex",
                                       row=row, memory_mode=tier, micro_reg=str(depth), stage=stage,
                                       stage_row=("x" if has else None),
                                       dice_med=0.5 if has else None, n_pairs=10))
    schema.write(d / "valis_stages.csv", "valis_stages", vs)
    schema.write(d / "stare_tiers.csv", "stare_tiers", [
        schema.empty_row("stare_tiers", source="regbench", dataset="multiplex", row="stare",
                         tier="high", stride=64, sprep="none"),
        schema.empty_row("stare_tiers", source="regbench", dataset="multiplex", row="stare_medium",
                         tier="medium", stride=64, sprep="none"),
    ])
    cost = []
    for row, fam in (("stare", "stare"), ("valis_mirage", "valis_mirage"), ("ashlar_clean", "ashlar")):
        for i, p in enumerate(PATIENTS):
            cost.append(schema.empty_row("cost_points", source="regbench", dataset="multiplex",
                                         family=fam, row=row, unit_kind="patient", unit_id=p,
                                         input_gb=1.0 + i, wall_s=100.0 * (i + 1),
                                         peak_rss_gb=2.0 + i, cpu_h_used=0.5 * (i + 1)))
    schema.write(d / "cost_points.csv", "cost_points", cost)
    schema.write(d / "cost_fits.csv", "cost_fits", [
        schema.empty_row("cost_fits", source="regbench", dataset="multiplex", family="stare",
                         row="stare", target="peak_rss_gb", predictor="input_gb", slope=1.0,
                         intercept=1.0, n=5, n_distinct_x=5, fit_ok=True)])
    schema.write(d / "pipeline_cost_fits.csv", "pipeline_cost_fits", [
        schema.empty_row("pipeline_cost_fits", source="regbench", backend="tiled",
                         process="TILED_STITCH", target="peak_rss_gb", predictor="input_gb",
                         slope=0.5, intercept=1.0, n=6, n_distinct_x=3, fit_ok=True)])
    return d


def _source(res, name):
    path = next(f for f in res["files"] if f.endswith("_source_data.csv"))
    return list(csv.DictReader(open(path)))


def test_every_plot_writes_all_formats_or_a_placeholder(paper):
    figs = paper / "figures"
    for name, fn in plots.PLOTS.items():
        res = fn(paper, figs)
        suffixes = {f.rsplit(".", 1)[1] for f in res["files"]}
        assert {"pdf", "png", "svg", "csv"} <= suffixes, name
        assert res["whiskers"] == "1.5 x IQR"
        assert res.get("placeholder", False) == (name in MISSING), name


MISSING = {"figS_cpu_scaling", "public_dice_best", "public_landmarks_best"}


def test_a_missing_input_is_a_placeholder_that_names_the_table_and_path(paper):
    res = plots.PLOTS["figS_cpu_scaling"](paper, paper / "figures")
    assert res["placeholder"] is True
    rows = _source(res, "figS_cpu_scaling")
    assert rows[0]["table"] == "cpu_scaling"
    assert rows[0]["expected_path"].endswith("cpu_scaling.csv")
    assert (paper / "figures" / "figS_cpu_scaling.png").stat().st_size > 0


def test_all_variants_has_more_boxes_than_best(paper):
    figs = paper / "figures"
    best = plots.PLOTS["fig3b_dice_best"](paper, figs)
    every = plots.PLOTS["fig3b_dice_all"](paper, figs)
    rows = lambda res: {r["row"] for r in _source(res, "")}  # noqa: E731
    assert len(rows(every)) > len(rows(best))
    assert rows(best) == {"stare", "valis_mirage", "ashlar_clean"}


def test_a_family_with_no_rows_is_absent_but_counted_zero(paper):
    res = plots.PLOTS["fig3b_dice_best"](paper, paper / "figures")
    assert res["n"]["palom"] == 0
    assert "palom" not in {r["family"] for r in _source(res, "")}
    assert res["n"]["stare"] == 10


def test_dice_zeroes_a_missing_pair_and_drops_a_case_empty_everywhere(paper):
    res = plots.PLOTS["fig3b_dice_all"](paper, paper / "figures")
    rows = _source(res, "")
    ash = [r for r in rows if r["row"] == "ashlar"]
    assert len(ash) == 10  # P0_r3 is NaN on every row: left out
    assert "P0_r3" not in {r["case_id"] for r in rows}
    zeroed = [r for r in ash if r["zeroed"] == "True"]
    assert len(zeroed) == 1 and float(zeroed[0]["dice"]) == 0.0


def test_an_excluded_row_never_enters(paper):
    rows = schema.read(paper / "per_pair.csv")
    for r in rows:
        if r["row"] == "stare" and r["case_id"] == "P1_r1":
            r["excluded_from_summaries"] = "True"
    with open(paper / "per_pair.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: ("NA" if v is None else v) for k, v in r.items()} for r in rows)
    res = plots.PLOTS["fig3b_dice_best"](paper, paper / "figures")
    assert res["n"]["stare"] == 9


def test_valis_stage_a_run_lacks_is_a_gap_not_a_point(paper):
    res = plots.PLOTS["fig3d_valis_stages"](paper, paper / "figures")
    rows = _source(res, "")
    low = {r["stage"]: r["dice_med"] for r in rows if r["row"] == "valis_low"}
    assert low["micro"] in ("", "NA", "nan") and low["micro_rigid"] in ("", "NA", "nan")
    assert low["rigid"] not in ("", "NA", "nan")


def test_colours_are_locked_across_figures(paper):
    import json

    figs = paper / "figures"
    plots.PLOTS["fig3b_dice_best"](paper, figs)
    first = json.loads((figs / ".palette.json").read_text())["default"]
    plots.PLOTS["fig3e_peak_rss_best"](paper, figs)
    assert json.loads((figs / ".palette.json").read_text())["default"] == first


def test_the_cli_writes_the_figures_folder(paper, capsys):
    from benchmarks.regbench import cli

    out = paper.parent
    assert cli.main(["plots", "--out", str(out), "--only", "fig3b_dice_best,figS_cpu_scaling"]) == 0
    assert (out / "paper" / "figures" / "fig3b_dice_best.pdf").is_file()
    assert "placeholder" in capsys.readouterr().out
