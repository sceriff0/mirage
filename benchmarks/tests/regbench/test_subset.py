"""subset.py: the few patients of the CPU cross, and a cases folder of them alone."""

from __future__ import annotations

import json

import pytest

from benchmarks.regbench import cli, subset
from benchmarks.tests.regbench import scored_cohort


def test_pick_takes_smallest_median_largest(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    picked = subset.pick(root)
    assert len(picked) == len({g for g, _ in picked}) <= 3
    sizes = [s for _, s in picked]
    assert sizes == sorted(sizes)


def _resize(root, case_id, hw):
    path = root / "multiplex" / case_id / "case.json"
    case = json.loads(path.read_text())
    case["mov_hw"] = list(hw)
    path.write_text(json.dumps(case))


def test_pick_ranks_by_total_moving_gigapixels_ties_by_name(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    # P1: two rounds of 440 x 440 (the largest in total); P2 and P3 one each, tied
    picked = subset.pick(root)
    assert [g for g, _ in picked] == ["P2", "P3", "P1"]
    assert picked[2][1] == pytest.approx(2 * 440 * 440 / 1e9)
    _resize(root, "P3_r1", (100000, 100000))
    assert [g for g, _ in subset.pick(root)] == ["P2", "P1", "P3"]
    # n other than 3: evenly spaced ranks with both ends
    assert [g for g, _ in subset.pick(root, n=2)] == ["P2", "P3"]
    assert [g for g, _ in subset.pick(root, n=9)] == ["P2", "P1", "P3"]


def test_pick_patients_prints_the_rule_and_the_sizes(tmp_path, capsys):
    root, out, made = scored_cohort.build(tmp_path)
    assert cli.main(["pick-patients", "--cases", str(root)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "# rule: smallest, median and largest by total moving gigapixels"
    assert [ln.split("\t")[0] for ln in lines[1:]] == ["P2", "P3", "P1"]


def test_subset_links_only_the_named_patients(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    n = subset.write(root, ["P1"], tmp_path / "sub")
    names = sorted(p.name for p in (tmp_path / "sub" / "multiplex").iterdir())
    assert n == 2 and names == ["P1_r1", "P1_r2"]
    assert all((tmp_path / "sub" / "multiplex" / x).is_symlink() for x in names)


def test_subset_cases_copies_the_manifest_and_lists_as_cases(tmp_path, capsys):
    from benchmarks.regbench.cases import list_cases

    root, out, made = scored_cohort.build(tmp_path)
    (root / "manifest.csv").write_text("patient_id\nP1\n")
    dest = tmp_path / "sub"
    code = cli.main(
        ["subset-cases", "--cases", str(root), "--groups", "P1,P3", "--to", str(dest)]
    )
    assert code == 0
    assert (dest / "manifest.csv").read_text() == "patient_id\nP1\n"
    assert [c.case_id for c in list_cases(dest, "multiplex")] == [
        "P1_r1",
        "P1_r2",
        "P3_r1",
    ]


def test_subset_refuses_a_folder_with_content(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x").write_text("x")
    with pytest.raises(SystemExit):
        subset.write(root, ["P1"], tmp_path / "sub")


def test_subset_refuses_a_patient_that_is_not_prepared(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    with pytest.raises(SystemExit, match="P9"):
        subset.write(root, ["P1", "P9"], tmp_path / "sub")
    assert not (tmp_path / "sub").exists()
