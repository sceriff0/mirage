import json
from pathlib import Path

from . import make_cohort_golden

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
