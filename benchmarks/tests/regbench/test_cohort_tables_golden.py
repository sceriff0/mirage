import hashlib
import json
import math
from pathlib import Path

import numpy as np

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


# T8 (the selfcheck runs and reports, never gates): the cohort's `dropme` row, once left
# out of every table by `score(dropped=...)`, is scored like any other. These are the
# tables it adds rows to, and how many: nothing else of the golden moved, apart from
# one column renamed in `failures` (`n_not_run_or_blocked` -> `n_not_run_or_lost`:
# there is no blocked run any more). Measured against the T7 golden (7e86bf10).
DROPME_ROWS = {
    "attrition": 1,
    "cells": 4,
    "cells_nuclei": 3,
    "cost_summary": 1,
    "failures": 1,
    "nonband_cells": 7,
    "nonband_cells_nuclei": 6,
    "nonband_summary": 2,
    "nonband_summary_nuclei": 2,
    "paired_summary": 1,
    "paired_summary_nuclei": 1,
    "paired_vs_primary": 3,
    "paired_vs_primary_nuclei": 2,
    "resources": 4,
    "summary_cells": 2,
    "summary_cells_nuclei": 2,
    "summary_resources": 3,
    "tissue_missing": 4,
}


def _dropme(row):
    return any(
        str(row.get(k, "")).startswith("dropme") for k in ("variant", "method", "run")
    )


def test_the_dropme_rows_move_no_other_row(tmp_path):
    """The golden minus its `dropme` rows is EXACTLY the snapshot of the same cohort
    without `dropme`'s results: the row it gained moved no other row, no common block
    and no paired row (`dropme` is no primary row)."""
    want = json.loads(GOLDEN.read_text())
    got = {t: sum(map(_dropme, rows)) for t, rows in want.items()}
    assert {t: n for t, n in got.items() if n} == DROPME_ROWS

    def without_dropme(out_dir):
        import shutil

        root, out, made = make_cohort_golden.scored_cohort.build(out_dir)
        shutil.rmtree(out / "dropme")
        return root, out, made

    alone = make_cohort_golden.snapshot(tmp_path, build=without_dropme)
    for table, rows in want.items():
        assert alone[table] == [r for r in rows if not _dropme(r)], table
    # the one column renamed with the blocked state, and nothing named after it
    assert all("n_not_run_or_lost" in r for r in want["failures"])
    assert not any("blocked" in k for rows in want.values() for r in rows for k in r)


# sha256 of the T7 golden (7e86bf10) in canonical JSON. Checked equal to the golden here
# minus its `dropme` rows with `n_not_run_or_lost` renamed back to `n_not_run_or_blocked`
# before this constant was written; the golden holds no summary header to rename.
T7_GOLDEN_SHA256 = "3fd78aea36cb84e7d1f731049d671835a35e8fa2a4fa04c279ab21b99928b731"


# the three columns T10 added (patient_id, panel, tool_version): not in the T7 golden
T10_COLUMNS = ("patient_id", "panel", "tool_version")
# the columns Task 28 added (attrition, failures, resources, summary_resources): with them
# stripped too the golden is still the T7 golden, so no value of an older column moved
T28_COLUMNS = (
    "checkpoint_rounds",
    "not_prepared",
    "killed_unsettled",
    "settled",
    "launches_failed_capped",
    "launches_failed_held",
    "launches_failed_killed_unsettled",
    "launches_failed_result_unreadable",
    "launches_failed_submit_refused",
    "launches_failed_watcher_gave_up",
    "cost_bound",
    "unmeasured_records",
    # fix round 1: B2's block label and the rounds lost to a cause not the method's
    "common_block",
    "n_lost_not_method",
)


# the columns Task 31 added to resources and summary_resources (the cost review): with
# them stripped too the golden is still the T7 golden, so no value of an older column
# moved (the golden changed only by added columns)
T31_COLUMNS = (
    "cost_blank_reason",
    "jvm_peak_heap_gb",
    "peak_concurrent_gb",
    "peak_hwm_gb",
    "peak_largest_job_gb",
    "register_s_per_round_from",
)


# Task 29: the paired tables' Dice difference is the ZERO-FILLED one (the headline rule);
# the old NaN-skipping difference is kept beside it as `d_dice_matched_skip*`, and the
# three pair fractions' differences were added. Read back: the T29 columns dropped and
# `d_dice_matched*` taken from `d_dice_matched_skip*`, the golden is the T7 golden, so no
# other value moved (measured: the old `d_dice_matched*` equal the new `_skip` ones in
# every row; 61 cells of `d_dice_matched*` moved to the zero-filled rule).
T29_ADDED = ("d_pair_fraction_maxn", "d_pair_fraction_sym")
# fix round 1: the generator also keeps the other two pair fractions (with excess) and the
# summaries' headline medians (make_cohort_golden.SUMMARY_MEDS) and the summary's own
# pair_fraction_max (SUMMARY_STATS; round 2 renamed the per-case metric pair_fraction_maxn):
# none is in the T7 golden
T29_F1_ADDED = ("pair_fraction_maxn", "pair_fraction_sym")
T29_SKIP = "d_dice_matched_skip"


def _before_t29(row):
    out = {}
    for k, v in row.items():
        if (k.startswith(T29_ADDED + T29_F1_ADDED) or k in make_cohort_golden.SUMMARY_MEDS
                or k in make_cohort_golden.SUMMARY_STATS):
            continue
        if k.startswith(T29_SKIP):
            out["d_dice_matched" + k[len(T29_SKIP):]] = v
        elif k.startswith("d_dice_matched") and T29_SKIP + k[len("d_dice_matched"):] in row:
            continue  # the zero-filled value; the skipping one is read back above
        else:
            out[k] = v
    return out


def test_the_golden_minus_dropme_is_the_t7_golden():
    want = {t: [_before_t29(r) for r in rows]
            for t, rows in json.loads(GOLDEN.read_text()).items()}
    back = {
        table: [
            {
                ("n_not_run_or_blocked" if k == "n_not_run_or_lost" else k): v
                for k, v in row.items()
                if k not in T10_COLUMNS and k not in T28_COLUMNS and k not in T31_COLUMNS
            }
            for row in rows
            if not _dropme(row)
        ]
        for table, rows in want.items()
    }
    blob = json.dumps(back, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == T7_GOLDEN_SHA256


# ── WT9 (T29): golden cells computed by hand, not by the scorer ───────────────────
# The golden above only says the tables did not CHANGE. These cells say they are RIGHT:
# each is recomputed here from the cohort's own generator (scored_cohort: the seeds, the
# shift, the smooth error of each toy map) with nothing of regbench but those constants.
_SEEDS = {"P1_r1": 1, "P1_r2": 2, "P2_r1": 3, "P3_r1": 4}
_AMPLITUDE = {"stare": 0.4, "tiles": 1.5, "blocks": 2.5, "dropme": 0.1}
_PIXEL_UM = 0.5


def _shoelace_centroid(ring):
    x, y = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    c = x * y1 - x1 * y
    a = c.sum() / 2.0
    return np.array([((x + x1) * c).sum() / (6 * a), ((y + y1) * c).sum() / (6 * a)])


def _hand_displacements_um(case_id, amplitude):
    """Each moving outline (a square of half-side OUTLINE_HALF) carried by the toy map
    vertex by vertex (xy + SHIFT + the smooth error), its centroid against its own
    reference centre: one cell per pair (the cells are 40 px apart, the error < 3 px)."""
    sc = make_cohort_golden.scored_cohort
    ref, mov = sc._centres(_SEEDS[case_id])
    h = sc.OUTLINE_HALF
    out = []
    for (rx, ry), (mx, my) in zip(ref, mov):
        v = np.array([[mx - h, my - h], [mx + h, my - h], [mx + h, my + h], [mx - h, my + h]])
        w = v + sc.SHIFT + sc._wobble(v, amplitude)
        out.append(float(np.hypot(*(_shoelace_centroid(w) - [rx, ry]))) * _PIXEL_UM)
    return np.array(out)


def _golden_cell(case_id, method):
    want = json.loads(GOLDEN.read_text())
    (row,) = [r for r in want["cells"] if r["case_id"] == case_id
              and r["method"] == method and r["set"] == "outlines"]
    return row


def test_hand_computed_golden_cells():
    for case_id in _SEEDS:
        for method, amp in _AMPLITUDE.items():
            if method == "tiles" and case_id == "P3_r1":
                continue  # never run there: an imputed row, below
            row, d = _golden_cell(case_id, method), _hand_displacements_um(case_id, amp)
            # every cell pairs its own: the pair fraction and the count ratio are 1
            assert row["pair_fraction"] == 1.0 and row["count_ratio"] == 1.0
            # within 1 um: the share of the hand displacements under 1 um
            assert abs(row["within_1um"] - float((d < 1.0).mean())) < 1e-8, (case_id, method)
            assert row["within_3um"] == 1.0
            # the median and the 90th percentile, to 1e-4 px (the centroid of a ring
            # warped at its vertices, not the warped centre: that differs by ~1e-3 um)
            assert abs(row["displacement_um_p50"] - float(np.median(d))) < 5e-5, (case_id, method)
            assert abs(row["displacement_um_p90"] - float(np.percentile(d, 90))) < 5e-5
    # the unregistered pose: SHIFT is 15 px, beyond the 1.5 x sqrt(144 / pi) = 10.16 px
    # gate and beyond 3 um (6 px): no pair, nothing within, no Dice; a failure (crash)
    # and a run never made (ghost, tiles on P3_r1) are scored at that pose
    gate = 1.5 * math.sqrt((2 * make_cohort_golden.scored_cohort.OUTLINE_HALF) ** 2 / math.pi)
    assert float(np.hypot(*make_cohort_golden.scored_cohort.SHIFT)) - 3.0 > gate
    for case_id in _SEEDS:
        for method in ("initial", "crash", "ghost") + (("tiles",) if case_id == "P3_r1" else ()):
            row = _golden_cell(case_id, method)
            assert (row["pair_fraction"], row["within_1um"], row["within_3um"]) == (0.0, 0.0, 0.0)
            assert row["dice_matched"] is None and row["displacement_um_p50"] is None
            assert row["imputed_initial"] == ("False" if method == "initial" else "True")
