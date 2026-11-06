"""ARCHIVED from benchmarks/tests/regbench/test_paper_plots.py (2026-11-06; not collected, see
benchmarks/archive/README.md): the tests of an archived dataset or module, verbatim.
Restoring them is pasting them back beside the module they test."""


def test_an_excluded_runs_na_rows_are_drawn_nowhere(paper):
    """T29 fix 2: excluded runs are rows of NA in stare_tiers; Fig. 3d draws none of them
    (an excluded medium-tier STARE run must not take the tier's slot)."""
    st = schema.read(paper / "stare_tiers.csv")
    st.append(schema.empty_row("stare_tiers", source="regbench", dataset="multiplex",
                               row="stare_gone", tier="medium", stride=64, sprep="none",
                               set="outlines", pooling="common", excluded_reason="x"))
    schema.write(paper / "stare_tiers.csv", "stare_tiers", st)
    tiers = plots.PLOTS["fig3d_stare_tiers"](paper, paper / "figures")
    assert tiers["n"]["medium"] == 10  # stare_medium's cases, not the excluded row's 0
