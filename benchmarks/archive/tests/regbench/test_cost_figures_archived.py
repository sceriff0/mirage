"""ARCHIVED from benchmarks/tests/regbench/test_cost_figures.py (2026-11-06; not
collected, see benchmarks/archive/README.md): the test as it was while figS_cpu_scaling
was a figure, verbatim. The live file keeps its end-to-end half."""


def test_the_supplementary_cost_figures_name_their_clock_and_end_to_end_its_intervals(
        tmp_path, monkeypatch):
    d = tmp_path / "paper"
    d.mkdir()
    schema.write(d / "end_to_end.csv", "end_to_end", [schema.empty_row(
        "end_to_end", source="regbench", dataset="multiplex", family="valis120",
        row="valis_a", row_kind="primary", clock=costs.WHOLE, unit="patient", n_patients=3,
        estimate_s_med=600, estimate_s_ci_low=500, estimate_s_ci_high=700, write_s_med=60,
        write_s_ci_low=50, write_s_ci_high=70)])
    schema.write(d / "cpu_scaling.csv", "cpu_scaling", [schema.empty_row(
        "cpu_scaling", source="regbench", family="valis120", row="valis_a", cpus=c,
        patient_id="P1", execution="one process", repeat=1, wall_s=w, wall_clock=costs.WHOLE,
        clock=costs.WHOLE, unit="patient", cpu_h_used=1.0, peak_rss_gb=2.0, speedup_vs_4=s)
        for c, w, s in ((4, 100, 1.0), (8, 60, 100 / 60))])
    seen = _labels(monkeypatch)
    e2e = cost_plots.figS_end_to_end(d, d / "figures")
    src = _source(e2e)
    reg = [r for r in src if r["part"] == "estimate_s_med"]
    assert reg and float(reg[0]["ci_low_min"]) == pytest.approx(500 / 60)
    assert reg[0]["clock"] == costs.WHOLE
    cost_plots.figS_cpu_scaling(d, d / "figures")
    named = [lab for pair in seen for lab in pair if lab]
    for want in ("Minutes", "Speed-up", "CPU-hours", "Peak memory"):
        (lab,) = [x for x in named if want in x]
        assert costs.WHOLE in lab, lab
