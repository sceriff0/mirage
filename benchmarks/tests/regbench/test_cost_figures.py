# benchmarks/tests/regbench/test_cost_figures.py
"""The paper's cost figures (``paper/cost_plots.py``, Task 31): one unit per axis, the
common patients in every tick, the primary row filled and the best row hollow, no GPU row
on a CPU axis, STARE at 4 CPUs on the wall axis, and every point not drawn a row of
``figures/losses.csv``. Hand-made ``cost_points`` first, then end to end from the scored
test cohort through ``tables.write_all`` and ``plots.main``."""
import csv
import math

import pytest

pytest.importorskip("matplotlib")

from benchmarks.regbench import score, variants  # noqa: E402
from benchmarks.regbench.paper import (  # noqa: E402
    cost_plots,
    costs,
    plots,
    schema,
    tables,
)
from benchmarks.tests.regbench import scored_cohort  # noqa: E402

PATIENTS = [f"P{i}" for i in range(6)]
FAMILY = {
    "stare": "stare",
    variants.STARE_LOCAL4: "stare",
    "dhr_gpu": "deeperhistreg",
    "dhr_cpu": "deeperhistreg",
    "valis_a": "valis_mirage",
    "valis_b": "valis_mirage",
}


def _pt(row, p, kind, primary, best, **kw):
    base = dict(source="regbench", dataset="multiplex", family=FAMILY[row], row=row,
                row_kind=kind, is_primary=primary, is_best=best, unit_kind="patient",
                unit="patient", unit_id=p, status="complete", in_common=True,
                clock=costs.WHOLE, input_gpx=1.0 + PATIENTS.index(p), device="cpu",
                peak_rss_comparable=True, cpu_h_used=1.0, cpu_h_reserved=2.0,
                total_cpu_h_reserved=2.5, gb_h_used=3.0, gb_h_reserved=16.0,
                wasted_gb_h_reserved=1.0, wall_s=600.0, peak_rss_gb=4.0)
    base.update(kw)
    return schema.empty_row("cost_points", **base)


@pytest.fixture
def paper(tmp_path, monkeypatch):
    monkeypatch.setattr(variants, "cpu_twin", lambda r: "dhr_cpu" if r == "dhr_gpu" else None)
    d = tmp_path / "paper"
    d.mkdir()
    pts = []
    for p in PATIENTS:
        pts.append(_pt("stare", p, costs.PRIMARY, True, True, clock=costs.CRITICAL,
                       elapsed_with_queue_s=3000.0, peak_concurrent_gb=9.0, peak_rss_gb=2.0))
        pts.append(_pt(variants.STARE_LOCAL4, p, costs.LOCAL_4CPU, False, False,
                       clock=costs.LOCAL4, wall_s=900.0))
        pts.append(_pt("dhr_gpu", p, costs.PRIMARY, True, True, device="gpu",
                       gpu_model="A100", gpu_h=0.2))
        pts.append(_pt("dhr_cpu", p, costs.CPU_TWIN, False, False, cpu_h_used=5.0))
        pts.append(_pt("valis_a", p, costs.PRIMARY, True, False))
        pts.append(_pt("valis_b", p, costs.BEST, False, True, cpu_h_used=2.0))
    # P5 of valis_a: not in the common set (drawn nowhere, counted in the tick's /all)
    pts = [dict(r, in_common=False) if (r["row"], r["unit_id"]) == ("valis_a", "P5") else r
           for r in pts]
    # P0 of valis_b: no CPU-hours (a loss row); P1: a lower bound (drawn, flagged)
    for r in pts:
        if (r["row"], r["unit_id"]) == ("valis_b", "P0"):
            r["cpu_h_used"] = None
        if (r["row"], r["unit_id"]) == ("valis_b", "P1"):
            r["lower_bound"] = "unmeasured stage attempts"
    # a rusage peak is in no peak comparison
    for r in pts:
        if (r["row"], r["unit_id"]) == ("valis_a", "P2"):
            r["peak_rss_comparable"] = False
    schema.write(d / "cost_points.csv", "cost_points", pts)
    schema.write(d / "method_best.csv", "method_best", [
        schema.empty_row("method_best", source="regbench", dataset="multiplex", family=f,
                         best_row=b)
        for f, b in (("stare", "stare"), ("deeperhistreg", "dhr_gpu"), ("valis_mirage", "valis_b"))
    ])
    schema.write(d / "cost_fits.csv", "cost_fits", [])
    return d


def _source(res):
    path = next(f for f in res["files"] if f.endswith("_source_data.csv"))
    return list(csv.DictReader(open(path)))


def _ticks(monkeypatch):
    seen = []
    real = plots._group_axes

    def spy(ax, labels, *a, **kw):
        seen.append(list(labels))
        return real(ax, labels, *a, **kw)

    monkeypatch.setattr(plots, "_group_axes", spy)
    return seen


def test_fig3f_one_unit_no_gpu_row_on_a_cpu_axis_stare_at_4_cpus(paper, monkeypatch):
    ticks = _ticks(monkeypatch)
    res = cost_plots.fig3f_cpu_wall_best(paper, paper / "figures")
    rows = _source(res)
    assert {r["unit"] for r in rows} == {"patient"}  # one unit per axis
    cpu = {r["row"] for r in rows if r["panel"] == "cpu_h"}
    assert "dhr_gpu" not in cpu and "dhr_cpu" in cpu  # R11: its CPU twin
    assert {r["row"] for r in rows if r["panel"] == "gpu_h"} == {"dhr_gpu"}
    wall = {(r["row"], r["clock"]) for r in rows if r["panel"] == "wall" and r["quantity"] == "wall_s"}
    assert (variants.STARE_LOCAL4, costs.LOCAL4) in wall  # R2: STARE at 4 CPUs
    assert ("stare", costs.CRITICAL) in wall  # the staged critical path: a marker
    assert {r["clock"] for r in rows if r["quantity"] == "elapsed_with_queue_s"} == {costs.QUEUE}
    # rule 2: every tick names the common count over all patients
    cpu_ticks = ticks[0]
    assert any("Valis" in t or "valis" in t.lower() for t in cpu_ticks)
    valis_a = next(t for t in cpu_ticks if "n=5/6" in t)
    assert "capped 0, failed 0" in valis_a
    # rule 3: the best row is labelled the ceiling
    assert any(cost_plots.CEILING in t for t in cpu_ticks)
    kinds = {(r["row"], r["row_kind"]) for r in rows if r["panel"] == "cpu_h"}
    assert ("valis_b", costs.BEST) in kinds and ("valis_a", costs.PRIMARY) in kinds


def test_fig3e_leaves_a_rusage_peak_out_and_marks_stares_concurrent_memory(paper):
    res = cost_plots.fig3e_peak_rss_best(paper, paper / "figures")
    rows = _source(res)
    assert ("valis_a", "P2") not in {(r["row"], r["unit_id"]) for r in rows
                                     if r["quantity"] == "peak_rss_gb"}
    assert {r["row"] for r in rows if r["quantity"] == "peak_concurrent_gb"} == {"stare"}
    causes = {(x["run"], x["patient_id"], x["cause"]) for x in res["losses"]}
    assert ("valis_a", "P2", "plot_points_dropped") in causes


def test_every_point_not_drawn_and_every_lower_bound_is_a_loss_row(paper):
    figs = paper / "figures"
    got = {"fig3f_cpu_wall_best": cost_plots.fig3f_cpu_wall_best(paper, figs),
           "figS_end_to_end": cost_plots.figS_end_to_end(paper, figs)}  # a placeholder
    from benchmarks.regbench import losses

    # T31 fix M7: the figures' rows go into paper/losses.csv, beside the tables' own,
    # replacing the figures' rows of an earlier draw and nothing else (no third file)
    losses.write(paper / "losses.csv", [
        losses.row(stage="paper", dataset="multiplex", run="r", cause="fit_points_dropped",
                   effect="dropped", detail="kept", source="paper/cost_fits.csv"),
        losses.row(stage="paper", dataset="multiplex", run="old", cause="plot_placeholder",
                   effect="dropped", detail="an earlier draw", source="figures/x")])
    path = cost_plots.write_losses(paper / "losses.csv", got, "multiplex")
    assert path == paper / "losses.csv" and not (figs / "losses.csv").exists()
    rows = list(csv.DictReader(open(path)))
    found = {(r["run"], r["patient_id"], r["cause"], r["effect"]) for r in rows}
    assert ("valis_b", "P0", "plot_points_dropped", "dropped") in found
    assert ("valis_b", "P1", "plot_lower_bound", "flagged") in found
    assert any(r["cause"] == "plot_placeholder" and "figS_end_to_end" in r["detail"] for r in rows)
    assert {r["stage"] for r in rows} == {"paper"}
    assert [r["detail"] for r in rows if r["cause"] == "fit_points_dropped"] == ["kept"]
    assert not any(r["run"] == "old" for r in rows)


def test_a_tick_reads_common_over_all_with_capped_and_failed():
    t = cost_plots.tick("STARE", {"common": 3, "all": 5, "capped": 1, "failed": 1, "not_drawn": 2})
    assert t == "STARE\nn=3/5 (capped 1, failed 1)\n+2 not drawn"


def test_fold_peak_vs_stare_is_gone():
    cols = schema.columns("method_summary")
    assert "fold_peak_vs_stare" not in cols and "fold_peak_vs_stare_local4" in cols


# ── end to end: the scored cohort, tables then figures ───────────────────────
FAM = {"stare": "stare", "dropme": "stare", "rtm": "rtm", "crash": "rtm"}


def test_the_scored_cohort_draws_every_cost_figure_on_patients(tmp_path, monkeypatch, capsys):
    from benchmarks.regbench import cli

    root, out, _ = scored_cohort.build(tmp_path)
    score.score(root, out, **scored_cohort.SCORE_KW)
    monkeypatch.setattr(tables, "family_of", lambda v: FAM.get(v))
    monkeypatch.setattr(tables, "terminal_rows", lambda ds: {k: k for k in FAM})
    tables.write_all(out, "multiplex")
    pts = schema.read(out / "paper" / "cost_points.csv")
    assert pts and {p["unit_kind"] for p in pts} <= {"patient", "round"}
    assert all(p["row_kind"] in costs.ROW_KINDS for p in pts)
    ticks = _ticks(monkeypatch)
    assert cli.main(["plots", "--out", str(out), "--only",
                     "fig3e_peak_rss_best,fig3f_cpu_wall_best,fig2c_cost_vs_input"]) == 0
    printed = capsys.readouterr().out
    figs = out / "paper" / "figures"
    for name in ("fig3e_peak_rss_best", "fig3f_cpu_wall_best"):
        assert f"{name}\tdrawn" in printed, printed
        src = list(csv.DictReader(open(figs / f"{name}_source_data.csv")))
        assert src and {r["unit"] for r in src} == {"patient"}, name
        drawn = {r["unit_id"] for r in src}
        assert drawn <= {p["unit_id"] for p in pts if p["unit_kind"] == "patient"}
    assert ticks and all("n=" in t and "/" in t for group in ticks for t in group)
    assert (out / "paper" / "losses.csv").is_file() and not (figs / "losses.csv").exists()
    assert "fold_peak_vs_stare," not in (out / "paper" / "method_summary.csv").read_text()
    summary = schema.read(out / "paper" / "method_summary.csv")
    assert summary and all(not math.isnan(float(r["n_patients_common"]))
                           for r in summary if r.get("n_patients_common") is not None)


def test_stare_local4_scoring_a_case_differently_is_an_integrity_loss():
    def pp(row, case, dice):
        return schema.empty_row("per_pair", source="regbench", dataset="multiplex",
                                patient_id="P1", case_id=case, row=row, set="outlines",
                                stage="final", dice_matched=dice)

    rows = [pp("stare", "P1_r1", 0.8), pp(variants.STARE_LOCAL4, "P1_r1", 0.8),
            pp("stare", "P1_r2", 0.8), pp(variants.STARE_LOCAL4, "P1_r2", 0.7)]
    got = tables.local4_differs(rows, "multiplex")
    assert [(r["case_id"], r["cause"], r["blame"], r["effect"]) for r in got] == [
        ("P1_r2", "stare_local4_differs", "integrity", "flagged")]


# ── T31 fix I3: the fold against stare_local4 ────────────────────────────────
def _fold_pts(row, peaks, clock=costs.WHOLE, device="cpu", status="complete"):
    return [{"row": row, "unit_kind": "patient", "unit_id": p, "status": status,
             "clock": clock, "device": device, "peak_rss_gb": v, "peak_rss_comparable": True}
            for p, v in peaks.items()]


def test_the_fold_is_over_the_patients_both_rows_completed():
    local = variants.STARE_LOCAL4
    # stare_local4 lost P3 (not complete): the fold reads P1, P2 for both rows, never
    # one median over {P1, P2, P3} against another over {P1, P2}
    pts = (_fold_pts(local, {"P1": 2.0, "P2": 4.0})
           + _fold_pts(local, {"P3": 1.0}, status="failed")
           + _fold_pts("valis_a", {"P1": 6.0, "P2": 12.0, "P3": 100.0}))
    common = {"P1", "P2", "P3"}
    got = tables.fold_vs_local4(pts, "valis_a", local, common)
    assert got["fold_peak_vs_stare_local4"] == pytest.approx(9.0 / 3.0)
    assert got["n_patients_fold"] == 2 and got["fold_note"] == ""
    # STARE's own staged row: a stage job's high-water mark is not the sampler's: none
    staged = _fold_pts("stare", {"P1": 1.0, "P2": 1.0}, clock=costs.CRITICAL)
    got = tables.fold_vs_local4(pts + staged, "stare", local, common)
    assert math.isnan(got["fold_peak_vs_stare_local4"]) and got["n_patients_fold"] == 0
    assert "stage" in got["fold_note"]
    # a GPU row: its host RSS only, and it says so
    gpu = _fold_pts("dhr_gpu", {"P1": 4.0, "P2": 8.0}, device="gpu")
    got = tables.fold_vs_local4(pts + gpu, "dhr_gpu", local, common)
    assert got["fold_peak_vs_stare_local4"] == pytest.approx(2.0)
    assert "GPU" in got["fold_note"] and "host" in got["fold_note"]
    # a patient outside the common set is in no fold
    got = tables.fold_vs_local4(pts, "valis_a", local, {"P1"})
    assert got["n_patients_fold"] == 1 and got["fold_peak_vs_stare_local4"] == pytest.approx(3.0)


# ── T31 fix I4, M2: fig. 2c by the same rules ────────────────────────────────
def _round(row, case, patient_in_common, status="complete", **kw):
    return _pt(row, "P0", costs.PRIMARY, True, False, unit_kind="round", unit="round",
               unit_id=case, in_common=patient_in_common, status=status, **kw)


def test_fig2c_draws_the_common_patients_with_their_counts_and_no_gpu_row_on_cpu_hours(paper):
    pts = schema.read(paper / "cost_points.csv")
    pts += [_round("valis_a", "P0_r1", True), _round("valis_a", "P5_r1", False),
            _round("valis_a", "P1_r1", True, status=costs.NOT_SPLIT, cpu_h_used=None,
                   peak_rss_gb=None, wall_s=None)]
    schema.write(paper / "cost_points.csv", "cost_points", pts)
    schema.write(paper / "cost_fits.csv", "cost_fits", [schema.empty_row(
        "cost_fits", source="regbench", dataset="multiplex", family="valis_mirage",
        row="valis_a", target="cpu_h_used", predictor="input_gpx", fit_ok=False)])
    res = cost_plots.fig2c_cost_vs_input(paper, paper / "figures")
    rows = _source(res)
    drawn = {(r["panel"], r["row"], r["unit_id"]) for r in rows if r["fit"] in ("False", "")}
    # rule 2 / P13: the common patients only, on both facets
    assert ("patient:cpu_h_used", "valis_a", "P0") in drawn
    assert ("patient:cpu_h_used", "valis_a", "P5") not in drawn
    assert ("round:cpu_h_used", "valis_a", "P0_r1") in drawn
    assert ("round:cpu_h_used", "valis_a", "P5_r1") not in drawn
    # R11: a GPU row's CPU-hours are never on a CPU-hours axis: its CPU twin is
    cpu = {r for p, r, _ in drawn if p.endswith("cpu_h_used")}
    assert "dhr_gpu" not in cpu and "dhr_cpu" in cpu
    # P10: every row's count, n=<common>/<all> (capped, failed), in the legend
    assert any("n=5/6 (capped 0, failed 0)" in t for t in res["legend"]), res["legend"]
    # M2, rule 6: a common patient's round with no per-round figure is a loss row
    lost = {(x["run"], x["case_id"], x["cause"]) for x in res["losses"]}
    assert ("valis_a", "P1_r1", "plot_points_dropped") in lost


def test_a_per_patient_methods_round_without_its_own_figure_is_a_point_that_says_so():
    res = [{"method": "valis_a", "group": "P1", "case_id": "P1_r1", "ok": "True",
            "unit": "patient", "register_s_per_round_from": "split"}]
    pts, without = costs.round_points("valis_a", "valis_mirage", res, set(),
                                      (costs.PRIMARY, True, False), common={"P1"})
    assert without == 1
    assert [(p["unit_id"], p["status"], p["in_common"]) for p in pts] == [
        ("P1_r1", costs.NOT_SPLIT, True)]
