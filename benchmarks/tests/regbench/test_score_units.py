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


# ── item 8: the unit ─────────────────────────────────────────────────────────────────
def _known_truth_rows(dataset, units, families=("grid", "seams")):
    rows = []
    for fam in families:
        for unit in units:
            for window in (0, 1):
                for pose in ("shift", "rot"):
                    group = (
                        f"{unit}_w{window}_{fam}"
                        if dataset == "semisynth"
                        else f"blob_{fam}"
                    )
                    case = f"{group}_{pose}"
                    for m in ("stare", "other"):
                        rows.append(
                            cell(
                                m,
                                case,
                                group,
                                dataset=dataset,
                                family=fam,
                                unit=unit if dataset == "semisynth" else group,
                                within_1um=0.2
                                if (m, fam) == ("stare", "seams")
                                else 0.9,
                            )
                        )
    return rows


def test_a_semisynthetic_patient_is_one_unit_whatever_its_windows():
    rows = _known_truth_rows("semisynth", ["046", "052"])
    summary = score.summarise_cells(rows, ["stare", "other"])
    rec = by(summary, method="stare", family="grid", subset="all")
    # 2 patients x 2 windows x 2 poses = 8 cases of 2 real patients, not of 4 "patients"
    assert (rec["n_cases"], rec["n_patients"]) == (8, 2)
    assert len({r["group"] for r in rows if r["family"] == "grid"}) == 4
    text = score.summary_markdown(summary, [], [], ["stare", "other"])
    assert "The independent unit is the real patient" in text
    assert f"At most 2 unit(s) per family: {score.DESCRIPTIVE}" in text


def test_a_synthetic_family_is_one_scene_and_is_never_ranked():
    rows = [
        r
        for r in _known_truth_rows("synthetic", ["scene"])
        if "_w0_" not in r["group"] and r["case_id"].count("blob") == 1
    ]
    summary = score.summarise_cells(rows, ["stare", "other"])
    assert {r["n_patients"] for r in summary} == {1}
    text = score.summary_markdown(summary, [], [], ["stare", "other"])
    assert "The independent unit is the generated scene" in text
    assert score.DESCRIPTIVE in text


def test_known_truth_cell_summaries_are_per_family_never_pooled():
    rows = _known_truth_rows("semisynth", ["046", "052"])
    summary = score.summarise_cells(rows, ["stare", "other"])
    seams = by(summary, method="stare", family="seams", subset="all")
    grid = by(summary, method="stare", family="grid", subset="all")
    assert seams["within_1um_med"] == pytest.approx(0.2)
    assert grid["within_1um_med"] == pytest.approx(0.9)
    assert not [r for r in summary if r["family"] == ""]
    # the cohort is not split: its family is empty
    cohort = score.summarise_cells([cell("a", "P0_r1", "P0", family="x")])
    assert {r["family"] for r in cohort} == {""}


def test_unit_of_a_case(tmp_path):
    case = toy_case(tmp_path, "P1_r1", "P1")
    assert score.unit_of(case) == "P1"
    case.dataset, case.group = "semisynth", "046_w1_grid"
    case.extra["truth_patient"] = "046"
    assert score.unit_of(case) == "046"


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
    text = "\n".join(score._paired_markdown([rec], "multiplex"))
    assert "bootstrap interval over units" in text
    assert score.DESCRIPTIVE.capitalize() in text
    assert "p-value" not in text and "significant" not in text


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


def test_known_truth_rows_leave_the_lost_round_out_too():
    base = {
        "dataset": "synthetic",
        "family": "grid",
        "pose": "shift",
        "in_stare_model": True,
        "method": "b",
        "tre_um_p50": 1.0,
        "tre_um_p90": 2.0,
        "tre_um_max": 3.0,
        "frac_under_1um": 0.5,
        "frac_under_2um": 0.9,
        "initial_tre_um_p50": 9.0,
        "fold_pct": 0.0,
        "imputed_initial": False,
        "cause": "",
    }
    lost = {**base, "imputed_initial": True, "cause": "benchmark_oom", "tre_um_p50": 9}
    (rec,) = score.truth_by_family([base, lost])
    assert (rec["n_cases"], rec["n_excluded_not_method"]) == (1, 1)
    assert rec["tre_um_p50_med"] == 1.0
    (rec,) = score.truth_by_family([lost])
    assert (rec["n_cases"], rec["n_excluded_not_method"]) == (0, 1)


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
