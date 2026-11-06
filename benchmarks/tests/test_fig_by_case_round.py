"""Supplementary Fig. S6 (benchmarks/figures/fig_by_case_round.py) on a tiny cells.csv:
two cases of two rounds each, three methods, every value known in advance."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from benchmarks.figures import fig_by_case_round as F  # noqa: E402
from benchmarks.regbench.cases import Case, write_json  # noqa: E402

# (case_id, patient, the moving round's channels; DAPI is the nuclear one)
CASES = [
    ("101_r2", "101", ["DAPI", "CD3", "CD8"]),
    ("101_r3", "101", ["DAPI", "CD45", "FOXP3"]),
    ("202_r2", "202", ["DAPI", "CD3", "CD8"]),
    ("202_r3", "202", ["DAPI", "CD45", "FOXP3"]),
]
# per method: Dice, then distance, per case in CASES order
SCORES = {
    "initial": ([0.10, 0.12, 0.08, 0.09], [9.0, 8.0, 10.0, 9.5]),
    "stare": ([0.80, 0.78, 0.82, 0.79], [0.5, 0.6, 0.4, 0.55]),
    "valis120": ([0.70, 0.75, 0.60, 0.72], [0.9, 0.8, 1.2, 0.85]),
}


@pytest.fixture
def fixture(tmp_path):
    cases = tmp_path / "cases"
    for case_id, group, channels in CASES:
        d = cases / "multiplex" / case_id
        d.mkdir(parents=True)
        case = Case(
            dataset="multiplex", case_id=case_id, group=group, ref_nuclear="",
            mov_nuclear="", ref_image="", mov_image="", modality="fluorescence",
            diagonal=1.0, extra={"channels_mov": channels, "mov_channel": 0}, dir=str(d),
        )  # fmt: skip
        write_json(d / "case.json", asdict(case))
    rows = []
    for method, (dice, dist) in SCORES.items():
        for (case_id, group, _), di, ds in zip(CASES, dice, dist):
            rows.append({"dataset": "multiplex", "set": "outlines", "method": method,
                         "case_id": case_id, "group": group, "dice_matched": di,
                         "displacement_um_p50": ds, "excluded_from_summaries": False})  # fmt: skip
    # a row the summaries leave out is drawn nowhere
    rows.append({**rows[0], "dice_matched": 0.99, "excluded_from_summaries": True})
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame(rows).to_csv(tables / "cells.csv", index=False)
    return tables, cases


def test_the_rounds_are_named_by_their_markers(fixture):
    _, cases = fixture
    labels, order = F.round_labels(cases, "multiplex")
    assert labels["101_r2"] == "CD3 / CD8" and labels["202_r3"] == "CD45 / FOXP3"
    assert order == ["CD3 / CD8", "CD45 / FOXP3"]


def test_each_panel_is_its_own_file_with_its_source_data(fixture, tmp_path):
    tables, cases = fixture
    out = tmp_path / "figs"
    F.main([str(tables), "--cases", str(cases), "--outdir", str(out),
            "--methods", "Unregistered=initial,STARE=stare,VALIS=valis120"])  # fmt: skip
    for p in "abcd":
        for ext in ("pdf", "png", "svg"):
            assert (out / f"by_case_round_{p}.{ext}").stat().st_size > 0, (p, ext)
        src = pd.read_csv(out / f"by_case_round_{p}_source_data.csv")
        assert set(src["panel"]) == {p}
    a = pd.read_csv(out / "by_case_round_a_source_data.csv")
    # every kept row once: 3 methods x 4 round pairs, the excluded row nowhere
    assert len(a) == 12 and 0.99 not in set(a["dice_matched"])
    assert set(a["group"].astype(str)) == {"101", "202"}
    b = pd.read_csv(out / "by_case_round_b_source_data.csv")
    assert set(b["round"]) == {"CD3 / CD8", "CD45 / FOXP3"}
    # the distance panels leave Unregistered out, as Fig. 3c does
    for p in "cd":
        src = pd.read_csv(out / f"by_case_round_{p}_source_data.csv")
        assert set(src["method"]) == {"STARE", "VALIS"} and len(src) == 8


def test_the_medians_say_which_method_is_best_in_each_group(fixture, tmp_path):
    tables, cases = fixture
    out = tmp_path / "figs"
    F.main([str(tables), "--cases", str(cases), "--outdir", str(out),
            "--methods", "Unregistered=initial,STARE=stare,VALIS=valis120"])  # fmt: skip
    med = pd.read_csv(out / "by_case_round_medians.csv")
    best = med[med["best_in_group"]]
    # STARE has the highest Dice and the lowest distance in every case and round
    assert set(best["method"]) == {"STARE"} and len(best) == 2 * 4
    assert set(med.loc[med["panel"] == "b", "grouped_by"]) == {"round"}
    row = med[(med["panel"] == "a") & (med["group"].astype(str) == "101")
              & (med["method"] == "VALIS")]  # fmt: skip
    assert float(row["median"].iloc[0]) == pytest.approx((0.70 + 0.75) / 2)
    assert int(row["n"].iloc[0]) == 2
