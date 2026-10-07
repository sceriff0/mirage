#!/usr/bin/env python3
"""Keep EXACTLY the rows of an arm plan whose run_id matches a regex.

``build_arm_plan.py --only`` selects rows AND everything that depends on them: naming
``valis_high_micro2``, whose nuclei every other arm is scored on, selects 117 of the 119
runs. That is right for a re-run after a change and wrong for "run these two arms, here":
this writes the named rows alone. Nothing they read is added either, so what they read
(``<root>/<from_arm>/csv/...``, another row's nuclei) must already exist or be among them.

    plan_exact.py <plan.csv> <out.csv> '^(valis_high_micro2|tiled_high_s64)$'
"""

from __future__ import annotations

import csv
import re
import sys


def exact_rows(rows: list[dict], pattern: str) -> list[dict]:
    """The rows whose run_id matches (re.search), in plan order; raises when a kept row
    is scored on the nuclei of a row that was not kept."""
    rx = re.compile(pattern)
    kept = [r for r in rows if rx.search(r["run_id"])]
    ids = {r["run_id"] for r in kept}
    missing = sorted(
        {r["seg_qc_nuclei_from"] for r in kept if r.get("seg_qc_nuclei_from")} - ids
    )
    if missing:
        raise SystemExit(
            f"{sorted(ids)} are scored on the nuclei of {missing}, which the pattern "
            "leaves out: name them too"
        )
    return kept


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        raise SystemExit(__doc__)
    src, dst, pattern = args
    with open(src, newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    kept = exact_rows(rows, pattern)
    if not kept:
        raise SystemExit(
            f"{pattern!r} matches no run_id of the {len(rows)}-row plan {src}"
        )
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    print(
        f"Exact plan: {len(kept)} of {len(rows)} rows -> {dst}: {[r['run_id'] for r in kept]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
