"""The golden snapshot of the cohort's score tables, the reference every removal task is
checked against: the ``multiplex`` rows of the score tables, on their key and value
columns only (timing, resource and ``selfcheck_*`` columns differ between builds).

Regenerate with ``python -m benchmarks.tests.regbench.make_cohort_golden``."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

from benchmarks.regbench import score as score_mod

from . import scored_cohort

# The brief's keys (``variant`` is called ``method`` in the tables) and the other columns
# that tell two rows of a table apart; a column a table lacks is skipped.
KEY_COLUMNS = (
    "dataset",
    "variant",
    "method",
    "case_id",
    "group",
    "unit",
    "tier",
    "set",
    "family",
    "pose",
    "subset",
    "against",
    "owner",
    "in_common_block",
)
_VALUES = (
    "pair_fraction pair_fraction_excess dice_matched iou_p50 frac_iou_ge_0.5 "
    "displacement_um_p50 displacement_um_p90 within_1um within_3um count_ratio "
    "fold_pct sd_log_jac n_unmapped coverage nn_distance fallback_frac "
    "imputed_initial excluded_from_summaries"
).split()
VALUE_COLUMNS = tuple(
    name + suffix for name in _VALUES for suffix in ("", "_shared", "_ranked", "_exact")
)
_BASE = (
    "cells summary_cells common_block paired_vs_primary paired_summary "
    "nonband_cells nonband_summary"
).split()
TABLES = tuple(_BASE + [name + "_nuclei" for name in _BASE])
GOLDEN = Path(__file__).parent / "fixtures" / "cohort_golden.json"


def _value(text):
    try:
        number = float(text)
    except ValueError:
        return text
    # NaN is not equal to itself, so it is stored as null
    return None if number != number else round(number, 9)


def snapshot(out_dir):
    root, out, _ = scored_cohort.build(out_dir)
    score_mod.score(root, out, **scored_cohort.SCORE_KW)
    got = {}
    for table in TABLES:
        path = Path(out) / "tables" / f"{table}.csv"
        if not path.exists():
            got[table] = []
            continue
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            names = reader.fieldnames or []
            # the paired tables' own value columns are the ``d_*`` differences
            values = [c for c in names if c in VALUE_COLUMNS or c.startswith("d_")]
            keep = [c for c in KEY_COLUMNS if c in names] + values
            rows = [
                {c: (_value(r[c]) if c in values else r[c]) for c in keep}
                for r in reader
                if r.get("dataset") == "multiplex"
            ]
        rows.sort(key=lambda r: tuple(str(r.get(k, "")) for k in KEY_COLUMNS))
        got[table] = rows
    return got


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        snap = snapshot(Path(tmp))
    GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
    print({t: len(r) for t, r in snap.items()}, file=sys.stderr)
