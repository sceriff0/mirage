"""ARCHIVED from benchmarks/tests/regbench/test_paper_tables.py (2026-11-06; not collected, see
benchmarks/archive/README.md): the tests of an archived dataset or module, verbatim.
Restoring them is pasting them back beside the module they test."""


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
