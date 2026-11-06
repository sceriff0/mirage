"""ARCHIVED from benchmarks/tests/regbench/test_public_never_pooled.py (2026-11-06; not collected, see
benchmarks/archive/README.md): the tests of an archived dataset or module, verbatim.
Restoring them is pasting them back beside the module they test."""


@pytest.mark.parametrize("name", ["hyreco"])
def test_a_landmark_set_is_scored_into_its_own_tables_never_the_cohorts(tmp_path, name):
    """Task 21: HyReCo is scored alone into ``tables/<name>/``."""
    from benchmarks.regbench import cli

    fixtures = pathlib.Path(__file__).resolve().parent / "fixtures" / name
    root, out, made = scored_cohort.build(tmp_path)
    flag = ["--data-root", str(fixtures), "--allow-incomplete"]
    assert cli.main(["prepare", name, *flag, "--cases", str(root)]) == 0
    score.score(root, out, datasets=["multiplex", name], **scored_cohort.SCORE_KW)
    assert (out / "tables" / name / f"{name}_pairs.csv").is_file()
    for path in (out / "tables").glob("*.csv"):
        assert name not in path.read_text(), path.name


def test_the_landmark_scorer_writes_only_its_own_tables():
    """Task 21 brought HyReCo's scorer back: every table it writes is named for its
    dataset, so none can be taken for a cohort table."""
    mod = importlib.import_module("benchmarks.regbench.landmarks")
    assert mod.TABLES and all(t.startswith("hyreco_") for t in mod.TABLES)
