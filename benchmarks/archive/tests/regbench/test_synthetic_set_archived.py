"""ARCHIVED from benchmarks/tests/regbench/test_synthetic_set.py (2026-11-06; not
collected, see benchmarks/archive/README.md): the two tests as they were before the CPU
cross (``cpu-scaling``) and the figures figS_synth_cross_scaling / figS_synth_vs_cohort
were archived, verbatim. The live file keeps their remaining halves."""


SYNTH_FIGS = ("figS_synth_cost_scaling", "figS_synth_cross_scaling", "figS_synth_vs_cohort")


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


def test_the_synthetic_figures_draw_from_the_sets_tables(built, scored, tmp_path):
    _, out, _ = built
    T.write_all(out, "synthetic")
    figs = tmp_path / "figs"
    # from the scorer's tables/synthetic/ beside paper/, and from paper/synthetic/
    for where in (Path(out) / "paper", Path(out) / "paper" / "synthetic"):
        for name in SYNTH_FIGS[:2]:
            res = P.draw(name, where, figs)
            assert not res.get("placeholder"), (where, name)
            assert res["n"] and Path(res["files"][0]).is_file()
    # no cohort cost yet: the extrapolation has no measured value, a placeholder
    res = P.draw("figS_synth_vs_cohort", Path(out) / "paper", figs)
    assert res.get("placeholder")
    # with the cohort's cost points, drawn: fitted against measured, y = x
    _cohort_points(out, [(0.0005, 1e-6), (0.001, 2e-6), (0.004, 9e-6)])
    T.write_all(out, "synthetic")
    res = P.draw("figS_synth_vs_cohort", Path(out) / "paper", figs)
    assert not res.get("placeholder")
    src = _read(next(f for f in res["files"] if f.endswith("_source_data.csv")))
    assert {r["row"] for r in src} == {"stare"} and {r["at"] for r in src} == set(S.AT)
    (Path(out) / "paper" / "cost_points.csv").unlink()
    T.write_all(out, "synthetic")
