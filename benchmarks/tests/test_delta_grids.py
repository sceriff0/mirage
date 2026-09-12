"""delta_grids: new method-variant cells APPENDED to a launched sweep.

The sweep was launched before STARE's `robust` solver existed, with
reg_tiled_solver pinned to `legacy` in the baseline. The resource curves of the
new solver must come from the same 9 STARE cells re-run at `robust` -- without
re-running anything else, and without moving a single run id of the launched
plan (run ids are assigned by enumeration; a block inserted anywhere but the
end renumbers everything after it, and the re-run lands in the wrong dirs).
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from benchmarks.analysis.lib import contract
from benchmarks.build_run_plan import build_run_plan, select_runs

BENCH = Path(__file__).resolve().parents[1]
SWEEP = BENCH / "configs" / "sweep.yaml"


@pytest.fixture(scope="module")
def sweep():
    return yaml.safe_load(SWEEP.read_text())


def _without_delta(sweep):
    s = copy.deepcopy(sweep)
    s.pop("delta_grids", None)
    return s


@pytest.mark.parametrize("repeats", [1, 3])
def test_delta_rows_are_appended_so_launched_run_ids_never_move(sweep, repeats):
    before = build_run_plan(_without_delta(sweep), repeats=repeats)
    after = build_run_plan(sweep, repeats=repeats)
    assert len(after) > len(before)
    assert after[: len(before)] == before, "a launched run id or config moved"
    tail = after[len(before) :]
    assert all(r["varied_axis"].startswith("delta_grid:") for r in tail)


def test_solver_robust_replicates_the_nine_stare_cells_at_robust(sweep):
    plan = build_run_plan(sweep, repeats=1)
    delta = [r for r in plan if r["varied_axis"] == "delta_grid:solver_robust"]
    stare = [r for r in plan if r["varied_axis"] == "registration_method_grid:tiled"]
    assert len(stare) == 9 and len(delta) == 9
    key = ("reg_tiled_mode", "reg_tiled_gate_tre")
    assert {tuple(r[k] for k in key) for r in delta} == {
        tuple(r[k] for k in key) for r in stare
    }
    assert {r["reg_tiled_solver"] for r in delta} == {"robust"}
    assert {r["reg_tiled_solver"] for r in stare} == {"legacy"}
    # everything else identical to the launched cell it replicates
    skip = {"run_id", "config_id", "varied_axis", "reg_tiled_solver"}
    by_cell = {tuple(r[k] for k in key): r for r in stare}
    for r in delta:
        base = by_cell[tuple(r[k] for k in key)]
        assert {k: v for k, v in r.items() if k not in skip} == {
            k: v for k, v in base.items() if k not in skip
        }


def test_the_solver_becomes_an_identity_column_so_rows_cannot_collapse(sweep):
    assert "reg_tiled_solver" in contract.axes(sweep)
    assert "reg_tiled_solver" in contract.identity_columns(sweep)
    assert "reg_tiled_solver" not in contract.axes(_without_delta(sweep))


def test_only_regex_selects_exactly_the_delta_block(sweep):
    plan = build_run_plan(sweep, repeats=3)
    sub = select_runs(plan, [], "delta_grid:solver_robust")
    assert len(sub) == 27 and all(
        r["varied_axis"] == "delta_grid:solver_robust" for r in sub
    )


def test_cli_writes_the_delta_subset_as_lines_of_the_full_plan(tmp_path):
    full = tmp_path / "full.csv"
    sub = tmp_path / "sub.csv"
    for out, extra in ((full, []), (sub, ["--only", "delta_grid:solver_robust"])):
        subprocess.run(
            [
                sys.executable,
                str(BENCH / "build_run_plan.py"),
                "--sweep",
                str(SWEEP),
                "--out",
                str(out),
                "--repeats",
                "3",
                *extra,
            ],
            check=True,
            capture_output=True,
            cwd=BENCH.parent,
        )
    full_lines = full.read_text().splitlines()
    sub_lines = sub.read_text().splitlines()
    assert sub_lines[0] == full_lines[0]
    assert len(sub_lines) == 28
    assert set(sub_lines[1:]) <= set(full_lines[1:])


def test_a_delta_block_must_name_a_real_grid_and_change_something(sweep):
    s = copy.deepcopy(sweep)
    s["delta_grids"] = {"x": {"registration_method": "elastix", "params": {"a": [1]}}}
    with pytest.raises(ValueError, match="no 'elastix' entry"):
        build_run_plan(s)
    s["delta_grids"] = {"x": {"registration_method": "tiled", "params": {}}}
    with pytest.raises(ValueError, match="params is empty"):
        build_run_plan(s)
    s["delta_grids"] = {
        "x": {
            "from": "axes",
            "registration_method": "tiled",
            "params": {"reg_tiled_solver": ["robust"]},
        }
    }
    with pytest.raises(ValueError, match="only registration_method_grid"):
        build_run_plan(s)
