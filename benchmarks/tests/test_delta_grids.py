"""delta_grids: new method-variant cells APPENDED to a launched sweep.

When a method variant arrives after the sweep was launched, its resource curves must come
from the same per-method cells re-run with the variant's params -- without re-running
anything else, and without moving a single run id of the launched plan (run ids are
assigned by enumeration; a block inserted anywhere but the end renumbers everything after
it, and the re-run lands in the wrong dirs).

The shipped sweep.yaml declares no delta grid since STARE v2 retired the one it had
(`solver_robust`, the old STARE cells at reg_tiled_solver=robust). The mechanism stays, so
these tests exercise it with a probe block injected into a copy of the real sweep:
the 9 STARE cells again at a non-default range gate.
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


DELTA = "maxdisp_probe"
PROBE = {
    DELTA: {
        "from": "registration_method_grid",
        "registration_method": "tiled",
        "params": {"reg_tiled_max_disp": [64]},
    }
}


@pytest.fixture(scope="module")
def shipped():
    return yaml.safe_load(SWEEP.read_text())


@pytest.fixture(scope="module")
def sweep(shipped):
    s = copy.deepcopy(shipped)
    s["delta_grids"] = copy.deepcopy(PROBE)
    return s


def test_the_shipped_sweep_carries_no_retired_solver_grid(shipped):
    grids = shipped.get("delta_grids") or {}
    assert "solver_robust" not in grids
    for block in grids.values():
        assert "reg_tiled_solver" not in (block.get("params") or {})


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


def test_a_delta_grid_replicates_the_nine_stare_cells_with_its_params(sweep):
    plan = build_run_plan(sweep, repeats=1)
    delta = [r for r in plan if r["varied_axis"] == f"delta_grid:{DELTA}"]
    stare = [r for r in plan if r["varied_axis"] == "registration_method_grid:tiled"]
    assert len(stare) == 9 and len(delta) == 9
    key = ("reg_tiled_mode", "reg_tiled_stride")
    assert {tuple(r[k] for k in key) for r in delta} == {
        tuple(r[k] for k in key) for r in stare
    }
    assert {r["reg_tiled_max_disp"] for r in delta} == {64}
    assert {r["reg_tiled_max_disp"] for r in stare} == {None}
    # everything else identical to the launched cell it replicates
    skip = {"run_id", "config_id", "varied_axis", "reg_tiled_max_disp"}
    by_cell = {tuple(r[k] for k in key): r for r in stare}
    for r in delta:
        base = by_cell[tuple(r[k] for k in key)]
        assert {k: v for k, v in r.items() if k not in skip} == {
            k: v for k, v in base.items() if k not in skip
        }


def test_the_delta_param_becomes_an_identity_column_so_rows_cannot_collapse(sweep):
    assert "reg_tiled_max_disp" in contract.axes(sweep)
    assert "reg_tiled_max_disp" in contract.identity_columns(sweep)
    assert "reg_tiled_max_disp" not in contract.axes(_without_delta(sweep))


def test_only_regex_selects_exactly_the_delta_block(sweep):
    plan = build_run_plan(sweep, repeats=3)
    sub = select_runs(plan, [], f"delta_grid:{DELTA}")
    assert len(sub) == 27 and all(
        r["varied_axis"] == f"delta_grid:{DELTA}" for r in sub
    )


def test_cli_writes_the_delta_subset_as_lines_of_the_full_plan(tmp_path, sweep):
    sweep_f = tmp_path / "sweep.yaml"
    sweep_f.write_text(yaml.safe_dump(sweep, sort_keys=False))
    full = tmp_path / "full.csv"
    sub = tmp_path / "sub.csv"
    for out, extra in ((full, []), (sub, ["--only", f"delta_grid:{DELTA}"])):
        subprocess.run(
            [
                sys.executable,
                str(BENCH / "build_run_plan.py"),
                "--sweep",
                str(sweep_f),
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
            "params": {"reg_tiled_max_disp": [64]},
        }
    }
    with pytest.raises(ValueError, match="only registration_method_grid"):
        build_run_plan(s)
