"""The scorer's subsets, units, uncertainty and cost totals (review items 6, 7, 8, 11).

Hand-made rows where the rule is arithmetic; the toy cases of ``test_pipeline`` where the
rule is about what is on disk.
"""

import math

import numpy as np
import pytest

from benchmarks.regbench import methods, pairs, score
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap
from benchmarks.tests.regbench.test_pipeline import (
    GRID,
    SHIFT,
    fake,
    good,
    squares,
    toy_case,
    true_map,
)


def cell(method, case_id, group, **kw):
    """One row of cells.csv, registered and clear of its null unless told otherwise."""
    row = {
        "dataset": "multiplex",
        "case_id": case_id,
        "group": group,
        "unit": group,
        "family": "",
        "method": method,
        "imputed_initial": False,
        "cause": "",
        "pair_fraction": 0.9,
        "pair_fraction_null": 0.1,
        "pair_fraction_null_max": 0.2,
        "pair_fraction_excess": 0.8,
        "dice_matched": 0.8,
        "dice_matched_null": 0.1,
        "within_1um": 0.7,
        "within_1um_null": 0.0,
        "within_3um": 0.9,
        "within_3um_null": 0.1,
        "displacement_readable": True,
        "displacement_um_p50": 0.5,
        "fold_pct": 0.0,
        "coverage": 0.3,
        "unmapped_frac": 0.0,
    }
    row.update(kw)
    return row


def by(rows, **want):
    (r,) = [r for r in rows if all(r[k] == v for k, v in want.items())]
    return r


# ── item 6: the number that is ranked is on the same cases for every row ─────────────
def test_ranked_dice_is_read_on_the_cases_every_compared_row_clears():
    rows = []
    for k in range(4):
        rows.append(cell("a", f"P{k}_r1", f"P{k}", dice_matched=0.5 + 0.1 * k))
        # b fails its two hard cases: under its null there, and good on the easy ones
        hard = k >= 2
        rows.append(
            cell(
                "b",
                f"P{k}_r1",
                f"P{k}",
                dice_matched=0.9,
                pair_fraction=0.1 if hard else 0.9,
                displacement_readable=not hard,
            )
        )
    summary = score.summarise_cells(rows, primary=["a", "b"])
    a, b = (by(summary, method=m, subset="all") for m in ("a", "b"))
    # each on its own cases: a on four, b on its two easy ones
    assert (a["n_dice_ranked"], b["n_dice_ranked"]) == (4, 2)
    assert a["dice_matched_ranked_med"] == pytest.approx(0.65)
    # shared: the two cases both clear, for both
    assert (a["n_dice_shared"], b["n_dice_shared"]) == (2, 2)
    assert a["dice_matched_shared_med"] == pytest.approx(0.55)
    assert b["dice_matched_shared_med"] == pytest.approx(0.9)
    assert (a["n_readable"], a["n_readable_shared"]) == (4, 2)
    # a row that is not compared is read on the compared rows' cases too
    summary = score.summarise_cells(
        [*rows, *(cell("c", f"P{k}_r1", f"P{k}") for k in range(4))],
        primary=["a", "b"],
    )
    assert by(summary, method="c", subset="all")["n_dice_shared"] == 2


def test_a_case_without_a_value_is_counted_not_dropped_in_silence():
    rows = [cell("a", f"P{k}_r1", f"P{k}") for k in range(3)]
    rows[1]["dice_matched"] = math.nan  # no pair at all
    (rec,) = [r for r in score.summarise_cells(rows) if r["subset"] == "all"]
    assert rec["dice_matched_n_nan"] == 1 and rec["pair_fraction_n_nan"] == 0
    table = score._summary_table([rec], None)
    assert "dice NaN cases" in table.splitlines()[0]


# ── item 7: the common block is among the primary rows that ran ──────────────────────
def test_the_fallback_and_the_exact_assignment_are_beside_the_headline():
    """A summary says in how many cases the pipeline's pairing fell back and on how
    large a share of the cells, and gives the exact assignment's numbers as extra
    columns: the headline, the shared subset and the differences do not read them."""
    rows = [
        cell("a", "P1_r1", "P1"),
        cell(
            "a",
            "P1_r2",
            "P1",
            fallback_frac=0.6,
            pair_fraction_exact=0.95,
            dice_matched_exact=0.7,
        ),
        cell(
            "a",
            "P2_r1",
            "P2",
            fallback_frac=0.25,
            pair_fraction_exact=0.99,
            dice_matched_exact=0.5,
        ),
        cell("a", "P2_r2", "P2", fallback_frac=0.0, pair_fraction_exact=0.9),
    ]
    plain = [
        {k: v for k, v in r.items() if "exact" not in k and k != "fallback_frac"}
        for r in rows
    ]
    got = by(score.summarise_cells(rows, ["a"]), method="a", subset="all")
    assert (got["n_fallback_cases"], got["fallback_frac_max"]) == (2, 0.6)
    assert got["n_exact"] == 3 and got["n_cases"] == 4
    # per unit the median over its rounds (P1: 0.95; P2: 0.945), then over units
    assert got["pair_fraction_exact_med"] == pytest.approx((0.95 + 0.945) / 2)
    assert got["dice_matched_exact_med"] == pytest.approx(0.6)
    without = by(score.summarise_cells(plain, ["a"]), method="a", subset="all")
    assert without["n_fallback_cases"] == 0 and without["n_exact"] == 0
    assert math.isnan(without["pair_fraction_exact_med"])
    for key, value in without.items():
        if "exact" not in key and "fallback" not in key:
            same = value == got[key] or (value != value and got[key] != got[key])
            assert same, key
    assert score.paired_vs_primary(rows) == score.paired_vs_primary(plain)
    text = score._summary_table([got], ["a"])
    assert "| 2 (0.6) |" in text and "| 3/4 |" in text
    assert "extra numbers, in no ranking" in score.PAIRING_RULE
    assert "exact_max_work" in score._settings()


def test_the_common_block_is_among_the_primary_rows_that_ran():
    rows = []
    for k in range(3):
        rows.append(cell("a", f"P{k}_r1", f"P{k}"))
        rows.append(cell("b", f"P{k}_r1", f"P{k}", imputed_initial=k == 2, cause=""))
        # c was never launched: every case is `not run`
        rows.append(
            cell("c", f"P{k}_r1", f"P{k}", imputed_initial=True, cause="not_run")
        )
        # d ran, and a gate blocked its run (older results would still be on disk)
        rows.append(cell("d", f"P{k}_r1", f"P{k}"))
    primary = ["a", "b", "c", "d", "e"]
    blocked = {"d": "its run `d` was blocked (selfcheck: inconsistent)"}
    inside, outside = score.compared_rows(rows, primary, blocked)
    assert inside == ["a", "b"]
    assert outside["c"] == score.NOT_LAUNCHED and outside["e"] == score.NO_ROW
    assert "blocked" in outside["d"]
    assert score.common_cases(rows, primary, blocked) == {"P0_r1", "P1_r1"}
    summary = score.summarise_cells(rows, primary, blocked)
    assert by(summary, method="a", subset="common")["n_cases"] == 2
    # the rule before: one primary that did not run emptied the block for everyone
    assert score.common_cases(rows, ["a", "b"]) != set()
    block = score.common_block(rows, primary, blocked)
    assert [r["method"] for r in block if r["in_common_block"]] == ["a", "b"]
    text = score.summary_markdown(summary, [], [], primary, common=block)
    assert "### Cases every compared row registered" in text
    assert "`c` (not launched by this submission" in text
    assert "`d` (its run `d` was blocked" in text


def test_the_block_is_named_even_when_no_primary_ran():
    rows = [
        cell("a", "P0_r1", "P0", imputed_initial=True, cause="not_run"),
        cell("initial", "P0_r1", "P0"),
    ]
    summary = score.summarise_cells(rows, ["a"])
    assert not [r for r in summary if r["subset"] == "common"]
    text = score.summary_markdown(
        summary, [], [], ["a"], common=score.common_block(rows, ["a"])
    )
    assert "No compared row ran in what is scored here" in text


def test_a_submission_of_some_runs_scores_without_only_run(tmp_path):
    """The walking skeleton: two runs launched, the full list expected."""
    root, out = tmp_path / "cases", tmp_path / "out"
    cases = [toy_case(root, f"P{k}_r1", f"P{k}") for k in range(2)]
    for c in cases:
        methods.run_case(good("stare"), c, out)
        methods.run_case(good("ashlar"), c, out)
    gates = out / "logs" / "gates"
    gates.mkdir(parents=True)
    (gates / "stare.tsv").write_text("stare\tok\t-\t0\t-\t/logs/rb-gate-stare-7.log\n")
    (gates / "valis120.tsv").write_text("valis120\tinconsistent\t-\t1\t-\n")
    res = score.score(
        root,
        out,
        ["multiplex"],
        expected=["stare", "ashlar", "valis120", "palom"],
        primary=["stare", "ashlar", "valis120", "palom"],
        gates=gates,
        run_variants={"valis120": ["valis120_rigid", "valis120"]},
        with_tissue=False,
    )
    block = {r["method"]: r for r in res["common_block"]}
    assert block["stare"]["in_common_block"] and block["ashlar"]["in_common_block"]
    assert "blocked" in block["valis120"]["why_not"]
    assert block["palom"]["why_not"] == score.NOT_LAUNCHED
    assert block["stare"]["n_cases_common"] == 2
    common = by(res["summary_cells"], method="stare", subset="common")
    assert common["n_cases"] == 2 and common["n_dice_shared"] == 2
    assert res["launch"][0]["log_path"] == "/logs/rb-gate-stare-7.log"
    # where the rounds went, as a table: three of the manifest's rounds were prepared
    # into two cases, both with outlines and neither with bare nuclei
    manifest = tmp_path / "rounds.csv"
    manifest.write_text("case_id,reference\nP0_r1,a\nP1_r1,b\nP2_r1,c\n,\n")
    assert score.manifest_rounds(manifest) == 3
    assert score.manifest_rounds(tmp_path / "absent.csv") is None
    assert [r["method"] for r in res["attrition"]] == [
        "ashlar",
        "stare",
        "valis120",
        "palom",
    ]
    for r in res["attrition"]:
        assert (r["manifest_rounds"], r["prepared"], r["with_outlines"]) == ("", 2, 2)
        assert r["with_nuclei"] == 0 and r["scored_nuclei"] == 0
        ran = 2 if r["method"] in ("stare", "ashlar") else 0
        assert (r["run"], r["registered"], r["scored"]) == (ran, ran, ran)
    table = (out / "tables" / "attrition.csv").read_text().splitlines()
    assert table[0] == ",".join(score.ATTRITION_COLUMNS) and len(table) == 5
    assert (
        "## From the manifest to a scored round"
        in (out / "tables" / "summary.md").read_text()
    )
    jobs = [("", c) for c in cases]
    got = score.attrition(jobs, res["cells"], [], [], res["resources"], 3)
    assert {r["manifest_rounds"] for r in got} == {3}
    # a toy case records nothing about its polygon sets: nothing is marked
    assert {(r["set_inexact"], r["set_shrunk"]) for r in res["cells"]} == {
        (False, False)
    }


def test_inexact_and_shrunk_nuclei_are_marked_on_every_row_and_counted():
    """A case whose bare nuclei are the outlines with the rim taken off, or are
    declared inexact, says so on every cell row computed on them; a summary counts
    those cases beside its numbers."""
    import types

    case = types.SimpleNamespace(
        extra={
            "cell_sets": {
                "ref": {
                    "outlines": {"source": "arm_geojson", "exact": True},
                    "nuclei": {"source": "label_image:/x/ref.tif", "exact": True},
                },
                "mov": {
                    "outlines": {"source": "arm_geojson", "exact": True},
                    "nuclei": {
                        "source": "shrunk_outlines",
                        "exact": False,
                        "shrunk_by_px": 10,
                    },
                },
            }
        }
    )
    assert score.set_marks(case, "outlines") == {
        "set_source_ref": "arm_geojson",
        "set_source_mov": "arm_geojson",
        "set_inexact": False,
        "set_shrunk": False,
    }
    marks = score.set_marks(case, "nuclei")
    assert marks["set_source_mov"] == "shrunk_outlines"
    assert marks["set_inexact"] is True and marks["set_shrunk"] is True
    case.extra["cell_sets"]["mov"]["nuclei"] = {
        "source": "stardist_rerun",
        "exact": False,
    }
    marks = score.set_marks(case, "nuclei")
    assert marks["set_inexact"] is True and marks["set_shrunk"] is False
    assert score.set_marks(types.SimpleNamespace(extra={}), "nuclei") == {
        "set_source_ref": "",
        "set_source_mov": "",
        "set_inexact": False,
        "set_shrunk": False,
    }
    rows = [
        cell("a", "P1_r1", "P1", set_inexact=True, set_shrunk=True),
        cell("a", "P1_r2", "P1", set_inexact=True, set_shrunk=False),
        cell("a", "P2_r1", "P2", set_inexact=False, set_shrunk=False),
    ]
    got = by(score.summarise_cells(rows, ["a"]), method="a", subset="all")
    assert (got["n_inexact_cases"], got["n_shrunk_cases"]) == (2, 1)
    assert "| 2 (1) |" in score._summary_table([got], ["a"])


def test_the_cohorts_caption_names_the_rows_tuned_on_it():
    rows = [cell(m, "P1_r1", "P1") for m in ("stare", "valis_mirage", "palom")]
    text = score.summary_markdown(score.summarise_cells(rows, ["stare"]), [], [])
    assert "**In-sample: `stare` and `valis_mirage`.**" in text
    assert "No other row was tuned on these slides." in text
    assert text.index("In-sample") < text.index("### All cases")
    # one of the two alone, and none: the line names what is there
    (line,) = score.in_sample_lines("multiplex", ["palom", "valis_mirage"])
    assert line.startswith("**In-sample: `valis_mirage`.** The settings of this row")
    assert score.in_sample_lines("multiplex", ["palom", "ashlar"]) == []
    # a dataset that is not the cohort the settings were chosen on has no caption
    assert score.in_sample_lines("elsewhere", ["stare"]) == []


def test_the_gate_block_counts_the_narrow_dots_of_each_selfcheck():
    def row(run, variant, case_id, dots, detected, narrow):
        return {
            "run": run,
            "variant": variant,
            "case_id": case_id,
            "set": dots,
            "n_detected": detected,
            "n_narrow": narrow,
        }

    rows = [
        # one comparison has a row per dot set: it is counted once
        row("stare", "stare", "pair", "all", 8891, 1),
        row("stare", "stare", "pair", "unmasked", 8891, 1),
        row("stare", "stare", "patient_round1", "all", 100, 2),
        row("ashlar", "ashlar", "pair", "all", 500, 0),
        row("valis120_hires", "valis120_hires_rigid", "pair", "all", 70, 4),
        {"run": "old", "variant": "old", "case_id": "pair", "set": "all"},
    ]
    assert score.narrow_dots(rows) == [
        {"run": "stare", "variant": "stare", "n_detected": 8991, "n_narrow": 3},
        {"run": "ashlar", "variant": "ashlar", "n_detected": 500, "n_narrow": 0},
        {
            "run": "valis120_hires",
            "variant": "valis120_hires_rigid",
            "n_detected": 70,
            "n_narrow": 4,
        },
    ]
    # and the rows are the selfcheck records' own: the count of each comparison
    from benchmarks.regbench import selfcheck

    record = {
        "pairs": [
            {
                "case_id": "pair",
                "variants": {
                    "stare": {"n_detected": 8891, "n_narrow": 1, "sets": {}},
                    "stare_rigid": {"status": "no_stage_image"},
                },
            }
        ]
    }
    made = selfcheck.mask_rows(record, "stare")
    assert {(r["variant"], r["n_detected"], r["n_narrow"]) for r in made} == {
        ("stare", 8891, 1)
    }
    assert score.narrow_dots(made) == [
        {"run": "stare", "variant": "stare", "n_detected": 8891, "n_narrow": 1}
    ]
    (line,) = score.narrow_lines(rows)
    assert "`stare` 3 of 8991 detected;" in line
    assert "`valis120_hires_rigid` 4 of 70 detected (run `valis120_hires`)" in line
    assert "None in the other 1 checked result(s)." in line
    assert score.narrow_lines(rows[3:4])[0].endswith(
        "None in the 1 checked result(s).\n"
    )
    assert score.narrow_lines([]) == [] and score.narrow_lines(rows[-1:]) == []
    # in the block of the gates, with or without a gate record
    gate = {"run": "stare", "gate": "selfcheck", "status": "ok", "blocked": False}
    block = score.launch_markdown([{**gate, "dropped": ""}], rows)
    assert block[0] == "## What the gates decided\n" and block[-1] == line
    assert score.launch_markdown([], rows) == ["## What the gates decided\n", line]
    assert score.launch_markdown([], []) == []


# ── item 8: the unit ─────────────────────────────────────────────────────────────────
def test_the_cohort_is_not_split_by_family():
    cohort = score.summarise_cells([cell("a", "P0_r1", "P0", family="x")])
    assert {r["family"] for r in cohort} == {""}


def test_unit_of_a_case(tmp_path):
    case = toy_case(tmp_path, "P1_r1", "P1")
    assert case.dataset == "multiplex"
    assert score.unit_of(case) == "P1"


# ── item 8: a paired bootstrap over units ────────────────────────────────────────────
def test_paired_differences_keep_a_units_rounds_together():
    rows = []
    for k, (a, b) in enumerate([(0.5, 0.7), (0.6, 0.9), (0.4, 0.5)]):
        for r in (1, 2):
            rows.append(cell("stare", f"P{k}_r{r}", f"P{k}", within_1um=a))
            rows.append(cell("other", f"P{k}_r{r}", f"P{k}", within_1um=b))
    paired = score.paired_vs_primary(rows)
    assert len(paired) == 3 and {p["n_rounds"] for p in paired} == {2}
    (rec,) = score.summarise_paired(paired)
    assert rec["n_patients"] == 3 and rec["d_within_1um_med"] == pytest.approx(0.2)
    lo, hi = rec["d_within_1um_lo"], rec["d_within_1um_hi"]
    assert 0.1 - 1e-9 <= lo <= 0.2 <= hi <= 0.3 + 1e-9 and lo < hi
    # deterministic, and nothing to resample with one unit
    assert score.summarise_paired(paired)[0]["d_within_1um_lo"] == lo
    one = score.summarise_paired(paired[:1])[0]
    assert math.isnan(one["d_within_1um_lo"]) and one[
        "d_within_1um_med"
    ] == pytest.approx(paired[0]["d_within_1um"])
    assert rec["d_within_1um_mean"] == pytest.approx(0.2)
    text = "\n".join(score._paired_markdown([rec], "multiplex"))
    # the caption says what the interval is over, and that it is too narrow
    assert "percentile bootstrap OVER UNITS" in text
    assert "It is narrower than the truth" in text
    assert score.DESCRIPTIVE.capitalize() in text
    assert "p-value" not in text and "significant" not in text
    # and it says "no ranking" with many units as with few
    many = "\n".join(score._paired_markdown([{**rec, "n_patients": 40}], "multiplex"))
    assert score.DESCRIPTIVE.capitalize() in many


def test_a_method_that_fails_on_some_patients_shows_on_the_line_of_its_median():
    """Five patients, two of them failed (scored at the unregistered pose): the median
    over patients does not move, so the mean and ok/cases stand beside it."""
    rows = [cell("a", f"P{k}_r1", f"P{k}", within_1um=0.7) for k in range(3)]
    rows += [
        cell(
            "a", f"P{k}_r1", f"P{k}", within_1um=0.0, imputed_initial=True, cause="oom"
        )
        for k in (3, 4)
    ]
    whole = [cell("b", f"P{k}_r1", f"P{k}", within_1um=0.7) for k in range(5)]
    summary = score.summarise_cells(rows + whole, ["a", "b"])
    a, b = (by(summary, method=m, subset="all") for m in "ab")
    assert a["within_1um_med"] == b["within_1um_med"] == 0.7
    assert a["within_1um_mean"] == pytest.approx(0.42) and b["within_1um_mean"] == 0.7
    assert (a["n_ok"], a["n_cases"]) == (3, 5)
    table = score._summary_table([a, b], ["a", "b"]).splitlines()
    (line_a,) = [ln for ln in table if ln.startswith("| a |")]
    (line_b,) = [ln for ln in table if ln.startswith("| b |")]
    assert "| 3/5 |" in line_a and "| 0.7 (0.42) [0] |" in line_a
    assert "| 5/5 |" in line_b and "| 0.7 (0.7) [0.7] |" in line_b
    text = score.summary_markdown(summary, [], [], ["a", "b"])
    assert "`median over units (mean over the same units) [minimum over cases]`" in text
    assert f"At most 5 unit(s): {score.DESCRIPTIVE}." in text
    # with eight patients the line is still there: no size of cohort ranks anything
    more = [cell("a", f"P{k}_r1", f"P{k}") for k in range(8)]
    text = score.summary_markdown(score.summarise_cells(more, ["a"]), [], [], ["a"])
    assert f"At most 8 unit(s): {score.DESCRIPTIVE}." in text
    assert not hasattr(score, "MIN_PATIENTS_TO_RANK")


def test_the_interval_is_the_units_spread_not_the_rounds():
    """Ten rounds of one unit are one number: resampling units cannot use them as ten."""
    assert score.bootstrap_interval([0.3, 0.3, 0.3]) == (0.3, 0.3)
    lo, hi = score.bootstrap_interval([0.0, 1.0])
    assert (lo, hi) == (0.0, 1.0)
    assert all(math.isnan(v) for v in score.bootstrap_interval([0.4]))


# ── rounds lost for a cause that is not the method's ─────────────────────────────────
def _with_a_lost_round():
    rows = []
    for k in range(4):
        lost = k == 3
        rows.append(cell("a", f"P{k}_r1", f"P{k}"))
        rows.append(
            cell(
                "b",
                f"P{k}_r1",
                f"P{k}",
                imputed_initial=lost,
                cause="infrastructure" if lost else "",
                within_1um=0.0 if lost else 0.7,
            )
        )
    return rows


def test_a_round_lost_to_the_scheduler_is_left_out_and_named():
    rows = _with_a_lost_round()
    summary = score.summarise_cells(rows, ["a", "b"])
    b = by(summary, method="b", subset="all")
    assert (b["n_cases"], b["n_ok"], b["n_excluded_not_method"]) == (3, 3, 1)
    assert b["within_1um_min_case"] == pytest.approx(0.7)  # not the imputed 0.0
    # and out of the common block for EVERY method
    assert by(summary, method="a", subset="common")["n_cases"] == 3
    assert by(summary, method="a", subset="all")["n_cases"] == 4
    named = score.excluded_rounds(rows)
    assert [(e["method"], e["case_id"], e["cause"]) for e in named] == [
        ("b", "P3_r1", "infrastructure")
    ]
    paired = score.paired_vs_primary(rows, against="a")
    assert {p["group"] for p in paired} == {"P0", "P1", "P2"}
    text = score.summary_markdown(summary, [], [], ["a", "b"], excluded=named)
    assert "`b` multiplex/P3_r1 (infrastructure)" in text


def test_a_failure_of_the_method_is_still_imputed():
    rows = _with_a_lost_round()
    for r in rows:
        if r["cause"]:
            r["cause"] = "oom"
    b = by(score.summarise_cells(rows, ["a", "b"]), method="b", subset="all")
    assert (b["n_cases"], b["n_ok"], b["n_excluded_not_method"]) == (4, 3, 0)
    assert b["within_1um_min_case"] == 0.0


def test_one_constant_switches_the_rule(monkeypatch):
    monkeypatch.setattr(score, "NOT_THE_METHOD_ROUNDS", "impute")
    rows = _with_a_lost_round()
    b = by(score.summarise_cells(rows, ["a", "b"]), method="b", subset="all")
    assert (b["n_cases"], b["n_excluded_not_method"]) == (4, 0)
    assert score.excluded_rounds(rows) == []


# ── item 11: cost ────────────────────────────────────────────────────────────────────
def res(method, group, case_id, unit="patient", ok=True, **kw):
    row = {
        "dataset": "multiplex",
        "group": group,
        "case_id": case_id,
        "method": method,
        "unit": unit,
        "ok": ok,
        "cause": "" if ok else "oom",
        "attempt": 1,
        "mem_requested_gb": 16.0,
        "cpus_requested": 4.0,
        "register_s": 100.0,
        "cpu_h": 0.1,
        "gpu_h": 0.0,
        "gpu_model": "",
        "write_s": 30.0,
        "total_to_image_s": 130.0,
        "cpu_efficiency": 0.9,
        "peak_rss_gb": 2.0,
        "peak_rss_after_register_gb": 2.0,
        "cgroup_peak_gb": 3.0,
        "wasted_attempts": 0,
        "wasted_cpu_h": 0.0,
        "wasted_wall_s": 0.0,
        "wasted_from_scheduler": 0,
        "wasted_unknown": 0,
    }
    row.update(kw)
    return row


def test_a_patient_launch_reports_the_whole_writer_time():
    """Five rounds, one writer call of 150 s recorded as five shares of 30 s."""
    rows = [res("group", "P1", f"P1_r{k}") for k in range(5)]
    (rec,) = score.summarise_resources(rows)
    assert rec["register_s"] == 100.0  # the launch's own total, on every round
    assert rec["write_s"] == pytest.approx(150.0)
    assert rec["total_to_image_s"] == pytest.approx(250.0)  # it was 130: one share
    # a pair method sums both
    rows = [res("pair", "P1", f"P1_r{k}", unit="pair") for k in range(5)]
    (rec,) = score.summarise_resources(rows)
    assert (rec["register_s"], rec["total_to_image_s"]) == (500.0, 650.0)


def test_cost_is_given_on_survivors_and_on_the_common_patients():
    rows = []
    for g, seconds in (("small", 100.0), ("large", 900.0)):
        rows.append(res("sturdy", g, f"{g}_r1", unit="pair", register_s=seconds))
        # `fragile` dies on the large patient: over its survivors it looks cheap
        rows.append(
            res("fragile", g, f"{g}_r1", unit="pair", register_s=50.0, ok=g == "small")
        )
    per_patient = score.summarise_resources(rows)
    summary = {r["method"]: r for r in score.summarise_cost(per_patient)}
    sturdy, fragile = summary["sturdy"], summary["fragile"]
    assert (fragile["n_patients_complete"], fragile["n_patients"]) == (1, 2)
    assert sturdy["register_s_med"] == 500.0  # both patients
    assert sturdy["n_patients_common"] == fragile["n_patients_common"] == 1
    assert sturdy["register_s_common_med"] == 100.0  # the patient both completed
    assert fragile["register_s_common_med"] == 50.0
    text = score.summary_markdown(
        [], [], per_patient, cost_summary=list(summary.values())
    )
    assert "| 1/2 |" in text and "over survivors" in text


def test_a_patient_with_a_failed_round_is_not_a_complete_cost():
    rows = [res("m", "P1", "P1_r1"), res("m", "P1", "P1_r2", ok=False)]
    (rec,) = score.summarise_resources(rows)
    assert rec["complete"] is False
    (cost_row,) = score.summarise_cost([rec])
    assert (cost_row["n_patients_complete"], cost_row["n_patients_partial"]) == (0, 1)
    assert math.isnan(cost_row["register_s_med"])


# ── pair fraction: the same lost cells give the same number in both directions ──────
def _lose(ps, cells_lost):
    xy = pairs.as_polyset(ps).xy.copy()
    for i in cells_lost:
        xy[4 * i : 4 * i + 4] = np.nan
    return pairs.with_xy(ps, xy)


@pytest.mark.parametrize("surplus", ["none", "moving", "reference"])
def test_cells_mapped_nowhere_count_the_same_in_both_directions(surplus):
    partners = np.array(
        [(x, y) for x in range(40, 1040, 40) for y in range(40, 440, 40)], dtype=float
    )
    extra = partners[:100] + [20.0, 20.0]  # nuclei with no partner on the other slide
    ref = squares(np.vstack([partners, extra]) if surplus == "reference" else partners)
    mov = squares(np.vstack([partners, extra]) if surplus == "moving" else partners)
    lost = range(80)  # the same 80 PARTNERED cells, lost by either kind of method
    forward = pairs.record(ref, mov, ref, _lose(mov, lost), 0.5)  # mov_to_ref
    backward = pairs.record(ref, mov, _lose(ref, lost), mov, 0.5)  # ref_to_mov
    assert forward["pair_fraction"] == backward["pair_fraction"] == pytest.approx(0.68)
    assert forward["within_1um"] == backward["within_1um"]
    # the lost cells stay in the denominator and are counted, in both directions
    assert forward["n_moving"] == pairs.n_cells(mov)
    assert backward["n_ref"] == pairs.n_cells(ref)
    assert forward["n_unmapped"] == backward["n_unmapped"] == 80


def test_losing_only_surplus_cells_does_not_lower_pair_fraction_but_is_counted():
    """The blind spot the review named: it is the same in both directions, and visible
    in ``n_unmapped`` and in ``within_*um``, never in pair fraction."""
    partners = np.array(
        [(x, y) for x in range(40, 1040, 40) for y in range(40, 440, 40)], dtype=float
    )
    extra = partners[:100] + [20.0, 20.0]
    few, many = squares(partners), squares(np.vstack([partners, extra]))
    surplus = range(len(partners), len(partners) + 100)
    forward = pairs.record(few, many, few, _lose(many, surplus), 0.5)
    backward = pairs.record(many, few, _lose(many, surplus), few, 0.5)
    assert forward["pair_fraction"] == backward["pair_fraction"] == 1.0
    assert forward["n_unmapped"] == backward["n_unmapped"] == 100
    assert forward["within_1um"] == backward["within_1um"] < 0.75


# ── the off-seam rows: what the mask alone allows ────────────────────────────────────
def test_the_off_seam_summary_reports_the_ceiling_the_mask_imposes(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = toy_case(root, "P1_r1", "P1")
    band = np.zeros(len(GRID), dtype=bool)
    band[:27] = True  # a third of the 81 moving nuclei are on the owner's seams

    def register(case, work, opts):
        yield "tiles", true_map(), {"band_mov": band}

    methods.run_case(fake("tiles", ["tiles"], register), case, out)
    methods.run_case(good(), case, out)
    got = score.score(
        root, out, ["multiplex"], seam_owners=("tiles",), with_tissue=False
    )
    rows = {r["method"]: r for r in got["nonband_summary"]}
    for name in ("tiles:non_tiles", "good:non_tiles"):
        # 54 moving nuclei kept against 81 reference nuclei: 2/3 for every method
        assert rows[name]["within_ceiling"] == pytest.approx(54 / 81)
        assert rows[name]["within_1um_med"] <= rows[name]["within_ceiling"] + 1e-9
        assert rows[name]["pair_fraction_med"] == 1.0  # over the smaller kept set
    text = (out / "tables" / "summary.md").read_text()
    assert "caps `within_*um` at about 0.6667" in text


def test_a_ref_to_mov_row_and_a_mov_to_ref_row_share_one_summary(tmp_path):
    """Sanity for the directions: both score 1.0 on the same toy case."""
    root, out = tmp_path / "cases", tmp_path / "out"
    case = toy_case(root, "P1_r1", "P1")

    def back(case, work, opts):
        yield (
            "back",
            PointMap(lambda xy: np.asarray(xy, dtype=float) - SHIFT, REF_TO_MOV),
            {},
        )

    methods.run_case(fake("back", ["back"], back), case, out)
    methods.run_case(good(), case, out)
    got = score.score(root, out, ["multiplex"], with_tissue=False)
    rows = {r["method"]: r for r in got["summary_cells"] if r["subset"] == "all"}
    assert rows["back"]["pair_fraction_med"] == rows["good"]["pair_fraction_med"] == 1.0
    assert rows["back"]["unmapped_frac_max"] == 0.0
    assert MOV_TO_REF != REF_TO_MOV
