# benchmarks/tests/regbench/test_cost_only_rows.py
"""A cost-only row (``variants.COST_ONLY``: ``stare_local4``, STARE's stages in one job of
4 CPUs) is in the cost tables and figures and in NO accuracy table or figure (T31 fix
C1, controller ruling 2026-10-19). One guard over every output of ``tables.write_all``
and ``plots.main``, on the scored test cohort with a ``stare_local4`` result beside
``stare``: the cost outputs are named here, everything else is accuracy."""
import csv

import pytest

pytest.importorskip("matplotlib")

from benchmarks.regbench import methods, score, variants  # noqa: E402
from benchmarks.regbench.paper import cost_plots, plots, tables  # noqa: E402
from benchmarks.tests.regbench import scored_cohort  # noqa: E402

LOCAL = variants.STARE_LOCAL4
# the toy methods stand for real families (the figures' palette is the paper's)
FAM = {"stare": "stare", LOCAL: "stare", "rtm": "palom", "crash": "palom", "dropme": "stare"}
# the paper tables a cost-only row belongs in: its cost
COST_TABLES = {"cost_points", "cost_fits", "end_to_end", "method_summary",
               "losses", "stare_stage_peaks"}
# the columns of method_summary that are accuracy (blank on a cost-only row)
ACCURACY = ("dice_med", "dice_q25", "dice_q75", "dice_mean", "disp_um_med", "n_pairs")


def _names(path):
    """Every cell of a CSV: a row name can sit in any column (row, best_row, stage_row,
    row_name, run)."""
    with open(path, newline="") as fh:
        return [c for r in csv.reader(fh) for c in r]


@pytest.fixture
def scored(tmp_path, monkeypatch):
    root, out, made = scored_cohort.build(tmp_path)
    for case in made:  # the same registration as stare's, run as the 4-CPU cost row
        methods.run_case(scored_cohort.forward(LOCAL, 0.4), case, out)
    score.score(root, out, **scored_cohort.SCORE_KW)
    monkeypatch.setattr(tables, "family_of", lambda v: FAM.get(v))
    # the run list, the cost-only run in it: the filter under test is terminal_rows'
    monkeypatch.setattr(tables, "_run_terminals", lambda ds: {k: k for k in FAM}, raising=False)
    tables.write_all(out, "multiplex")
    return out


def test_a_cost_only_row_is_in_no_accuracy_table_or_figure(scored, monkeypatch):
    from benchmarks.regbench import cli

    paper = scored / "paper"
    # the cost tables do carry it (or the guard below proves nothing)
    assert LOCAL in {r["row"] for r in csv.DictReader(open(paper / "cost_points.csv"))}
    leaked = [p.name for p in sorted(paper.glob("*.csv"))
              if p.stem not in COST_TABLES and LOCAL in _names(p)]
    assert not leaked, leaked
    # method_summary: its cost row, with no accuracy of its own
    mine = [r for r in csv.DictReader(open(paper / "method_summary.csv")) if r["row"] == LOCAL]
    assert mine and all(r[c] in ("", "NA", "nan") for r in mine for c in ACCURACY if c in r), mine
    # fig. 3d's high tier is stare's own row, never the cost row (the review's case)
    picks = {}
    for r in csv.DictReader(open(paper / "stare_tiers.csv")):
        picks[r["tier"]] = r["row"]
    assert picks.get("high") != LOCAL
    assert cli.main(["plots", "--out", str(scored)]) == 0
    figs = paper / "figures"
    srcs = sorted(figs.glob("*_source_data.csv"))
    assert srcs
    accuracy_figs = [p for p in srcs if not any(p.name.startswith(f) for f in cost_plots.FIGURES)]
    assert accuracy_figs, [p.name for p in srcs]
    leaked = [p.name for p in accuracy_figs if LOCAL in _names(p)]
    assert not leaked, leaked
    # and it is in the cost table as what it is: STARE's 4-CPU cost row
    assert {r["row_kind"] for r in mine} == {"local_4cpu"}

def test_an_old_per_pair_csv_with_the_cost_row_draws_no_accuracy_figure_of_it():
    """A per_pair.csv written before the fix (or collected from a draft) still names the
    cost row: the figures' own reader leaves it out."""
    pp = [{"set": tables.OUTLINES, "row": r, "case_id": "c", "excluded_from_summaries": ""}
          for r in ("stare", LOCAL)]
    assert [r["row"] for r in plots._outlines(pp)] == ["stare"]
    assert tables.is_cost_only(LOCAL) and not tables.is_cost_only("stare")
    assert [r["row"] for r in tables.accuracy_pairs(pp)] == ["stare"]
