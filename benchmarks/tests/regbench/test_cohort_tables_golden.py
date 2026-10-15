import json
from pathlib import Path

from benchmarks.regbench import methods

from . import make_cohort_golden, scored_cohort

GOLDEN = Path(__file__).parent / "fixtures" / "cohort_golden.json"


def test_the_cohort_tables_are_the_golden_ones(tmp_path):
    want = json.loads(GOLDEN.read_text())
    got = make_cohort_golden.snapshot(tmp_path)
    assert sorted(got) == sorted(want)
    for table in want:
        assert got[table] == want[table], table


def test_the_snapshot_has_rows_of_every_table():
    # every table has multiplex rows today; one that loses them has changed
    want = json.loads(GOLDEN.read_text())
    assert sorted(want) == sorted(make_cohort_golden.TABLES)
    assert [t for t in want if not want[t]] == []


def _multiplex_only(tmp):
    """``scored_cohort.build`` without its ``synthetic`` case."""
    sc = scored_cohort
    root, out = Path(tmp) / "cases", Path(tmp) / "out"
    made = [
        sc.multiplex_case(root, "P1_r1", "P1", 1, parents=True),
        sc.multiplex_case(root, "P1_r2", "P1", 2, parents=True),
        sc.multiplex_case(root, "P2_r1", "P2", 3),
        sc.multiplex_case(root, "P3_r1", "P3", 4, nuclei=False),
    ]
    for case in made:
        methods.run_case(sc.forward("stare", 0.4), case, out)
        if case.case_id != "P2_r1":
            methods.run_case(sc.backward("rtm", 0.9), case, out)
        if case.case_id != "P3_r1":
            methods.run_case(sc.forward("tiles", 1.5, sc._moving_mask), case, out)
        methods.run_case(sc.forward("blocks", 2.5, sc._reference_mask), case, out)
        methods.run_case(sc.crashing("crash"), case, out)
        methods.run_case(sc.forward("dropme", 0.1), case, out)
    return root, out, made


def test_the_golden_does_not_need_the_synthetic_case(tmp_path):
    """Removing the ``synthetic`` dataset must not move a multiplex number."""
    want = json.loads(GOLDEN.read_text())
    got = make_cohort_golden.snapshot(tmp_path, build=_multiplex_only)
    for table in want:
        assert got[table] == want[table], table
