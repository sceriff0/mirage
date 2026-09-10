import importlib.util
import re
import sys
from pathlib import Path

import pytest

from benchmarks.build_run_plan import build_run_plan

REPO_ROOT = Path(__file__).resolve().parents[2]


def _param_checker():
    """Load tests/check_param_consistency.py (not an importable package) by path.

    That module already parses nextflow.config's `params {}` block into Python values and is the
    pipeline's own source of truth for "what is the shipped default". Reusing it keeps one Groovy
    literal parser in the repo instead of a second, drifting copy here.
    """
    path = REPO_ROOT / "tests" / "check_param_consistency.py"
    # That module imports its sibling `_code_view` flat, the way everything in tests/
    # does -- but loading a file by path does not put its directory on sys.path, so the
    # sibling import fails with ModuleNotFoundError before a single assertion runs.
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_check_param_consistency", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CHECKER = _param_checker()
Unparsed = _CHECKER.Unparsed


def pipeline_param_defaults() -> dict:
    """param name -> shipped default, straight from nextflow.config."""
    return _CHECKER.extract_config_defaults((REPO_ROOT / "nextflow.config").read_text())


SWEEP = {
    "strategy": "ofat",
    "baseline": {"target_px": 4096, "n_channels": 2, "memory_mode": "medium"},
    "axes": {
        "target_px": [4096, 8192],
        "n_channels": [2, 4],
        "memory_mode": ["medium", "high"],
    },
}

GRID_SWEEP = {
    "strategy": "ofat",
    "scaling_grid": {"target_px": [2048, 4096, 8192], "n_channels": [2, 4]},
    "baseline": {"target_px": 4096, "n_channels": 2, "memory_mode": "medium"},
    "axes": {"memory_mode": ["medium", "high"]},
}

GRID_SWEEP2 = {
    "strategy": "ofat",
    "scaling_grid": {"target_px": [2048, 4096], "n_channels": [2, 4]},
    "registration_grid": {
        "target_px": [2048, 4096],
        "n_register_images": [2, 4, 8],
        "n_channels": 2,
    },
    "baseline": {
        "target_px": 4096,
        "n_channels": 2,
        "n_register_images": 2,
        "memory_mode": "medium",
    },
    "axes": {"memory_mode": ["medium", "high"]},
}


def test_ofat_includes_one_baseline_run():
    plan = build_run_plan(SWEEP)
    baseline_runs = [r for r in plan if r["varied_axis"] == "baseline"]
    assert len(baseline_runs) == 1
    assert baseline_runs[0]["memory_mode"] == "medium"


def test_ofat_varies_one_axis_at_a_time_off_baseline():
    plan = build_run_plan(SWEEP)
    # non-baseline values: target_px=8192, n_channels=4, memory_mode=high -> 3 runs
    varied = [r for r in plan if r["varied_axis"] != "baseline"]
    assert len(varied) == 3
    hi = [r for r in varied if r["varied_axis"] == "memory_mode"][0]
    # only memory_mode differs from baseline; other axes stay at baseline values
    assert (
        hi["memory_mode"] == "high"
        and hi["target_px"] == 4096
        and hi["n_channels"] == 2
    )


def test_run_ids_are_unique():
    plan = build_run_plan(SWEEP)
    ids = [r["run_id"] for r in plan]
    assert len(ids) == len(set(ids))


def test_grid_is_full_cross_product():
    sweep = dict(SWEEP, strategy="grid")
    plan = build_run_plan(sweep)
    # 2 * 2 * 2 = 8 cells
    assert len(plan) == 8


def test_repeats_expands_each_config_with_unique_ids():
    plan = build_run_plan(SWEEP, repeats=3)
    # 4 configs (baseline + 3 OFAT axes) x 3 repeats = 12 runs
    assert len(plan) == 12
    assert len({r["run_id"] for r in plan}) == 12  # unique launch dirs
    by_cfg = {}
    for r in plan:
        by_cfg.setdefault(r["config_id"], []).append(r["rep"])
    assert len(by_cfg) == 4  # 4 distinct configs
    assert all(sorted(reps) == [0, 1, 2] for reps in by_cfg.values())


def test_repeats_default_one_reproduces_single_shot():
    assert len(build_run_plan(SWEEP)) == 4  # baseline + 3 varied, no dup


def test_repeats_below_one_rejected():
    with pytest.raises(ValueError):
        build_run_plan(SWEEP, repeats=0)


def test_scaling_grid_is_full_size_channel_cross():
    plan = build_run_plan(GRID_SWEEP, repeats=1)
    grid_cells = {
        (r["target_px"], r["n_channels"])
        for r in plan
        if r["varied_axis"] in ("scaling_grid", "baseline")
    }
    assert grid_cells == {
        (2048, 2),
        (2048, 4),
        (4096, 2),
        (4096, 4),
        (8192, 2),
        (8192, 4),
    }
    # the baseline cell (4096, 2) is emitted exactly once, labelled baseline
    base = [r for r in plan if r["varied_axis"] == "baseline"]
    assert len(base) == 1
    assert (base[0]["target_px"], base[0]["n_channels"]) == (4096, 2)
    # OFAT knobs still run, held at the baseline input cell
    mm = [r for r in plan if r["varied_axis"] == "memory_mode"]
    assert len(mm) == 1 and mm[0]["target_px"] == 4096 and mm[0]["n_channels"] == 2


def test_registration_grid_crosses_size_and_rounds():
    plan = build_run_plan(GRID_SWEEP2, repeats=1)
    reg = [
        (r["target_px"], r["n_register_images"])
        for r in plan
        if r["varied_axis"] == "registration_grid"
    ]
    # 2 sizes x rounds {4,8} (N=2 skipped — covered by the scaling grid) = 4 configs,
    # all at the fixed 2 channels
    assert set(reg) == {(2048, 4), (2048, 8), (4096, 4), (4096, 8)}
    assert all(
        r["n_channels"] == 2 for r in plan if r["varied_axis"] == "registration_grid"
    )


def test_registration_grid_crosses_channels_when_list():
    sweep = {
        "strategy": "ofat",
        "scaling_grid": {"target_px": [2048, 4096], "n_channels": [2, 4]},
        "registration_grid": {
            "target_px": [2048, 4096],
            "n_register_images": [4, 8],
            "n_channels": [2, 4],
        },
        "baseline": {
            "target_px": 4096,
            "n_channels": 2,
            "n_register_images": 2,
            "memory_mode": "medium",
        },
        "axes": {"memory_mode": ["medium", "high"]},
    }
    plan = build_run_plan(sweep, repeats=1)
    reg = {
        (r["target_px"], r["n_channels"], r["n_register_images"])
        for r in plan
        if r["varied_axis"] == "registration_grid"
    }
    # 2 sizes x {2,4} channels x {4,8} rounds = 8 configs (N=2 skipped, covered by scaling grid)
    assert reg == {
        (2048, 2, 4),
        (2048, 2, 8),
        (2048, 4, 4),
        (2048, 4, 8),
        (4096, 2, 4),
        (4096, 2, 8),
        (4096, 4, 4),
        (4096, 4, 8),
    }


def test_registration_method_grid_crosses_each_method_own_knobs():
    """ONE grid, BOTH methods, each pinned to its own registration_method and crossed in full.

    Replaces test_registration_param_grid_crosses_params_classic. That grid crossed memory_mode x
    reg_micro_reg only, VALIS-only, and lived beside a flat reg_max_image_dim axis -- the split that
    left VALIS on 11 of 27 cells against STARE's 27. Both methods now go through the same generic
    per-method loop, which is the mechanism this asserts.
    """
    sweep = {
        "strategy": "ofat",
        "baseline": {
            "target_px": 4096,
            "n_channels": 2,
            "n_register_images": 2,
            "registration_method": "valis",
            "memory_mode": "medium",
            "reg_micro_reg": 0,
            "reg_max_image_dim": 4000,
            "reg_tiled_mode": "high",
            "reg_tiled_tile": None,
        },
        "registration_method_grid": {
            "valis": {
                "memory_mode": ["low", "medium", "high"],
                "reg_micro_reg": [0, 1, 2],
                "reg_max_image_dim": [2000, 4000, 8000],
            },
            "tiled": {"reg_tiled_mode": ["custom"], "reg_tiled_tile": [1024, 2048]},
        },
        "axes": {},
    }
    plan = build_run_plan(sweep, repeats=1)
    v = [r for r in plan if r["varied_axis"] == "registration_method_grid:valis"]
    t = [r for r in plan if r["varied_axis"] == "registration_method_grid:tiled"]

    assert len(v) == 27, (
        "the valis entry must be a FULL 3x3x3 cross, not a plane plus a spur"
    )
    assert len(t) == 2
    assert all(r["registration_method"] == "valis" for r in v)
    assert all(r["registration_method"] == "tiled" for r in t)
    assert {
        (r["memory_mode"], r["reg_micro_reg"], r["reg_max_image_dim"]) for r in v
    } == {
        (mm, rmr, d)
        for mm in ("low", "medium", "high")
        for rmr in (0, 1, 2)
        for d in (2000, 4000, 8000)
    }
    # the distributed knobs must not appear anywhere
    assert all(
        "reg_distributed_tiling" not in r and "reg_dist_force_tiling" not in r
        for r in v + t
    )


def test_project_sweep_exercises_the_tiled_fanout_path():
    """The STARE fan-out (TILED_COARSE/REG_TILE/SOLVE/STITCH) must be measured.

    These four processes only ever run under registration_method=tiled, so if the plan carries no
    tiled runs they are never benchmarked at all. This used to also assert that both sides of the
    old execution-shape flag appeared; that flag was removed along with the single-task
    TILED_REGISTER path, so the fan-out is the only shape and there is nothing left to cross.
    """
    import yaml

    sweep = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )
    plan = build_run_plan(sweep, repeats=1)
    tiled = [r for r in plan if r.get("registration_method") == "tiled"]
    assert tiled, "no tiled/STARE runs in the plan"


def test_project_sweep_covers_every_shipped_segmentation_backend():
    import yaml

    sweep = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )
    # nextflow.config's seg_method enum -- all three must be benchmarked somewhere in the plan.
    plan = build_run_plan(sweep, repeats=1)
    assert {r["seg_method"] for r in plan} == {"stardist", "instantseg", "cellsam"}


def stare_preset_row(mode):
    """Parse one row of RegPresets.STARE out of lib/RegPresets.groovy.

    The STARE tier knobs are null-declared in nextflow.config -- their shipped default now lives
    in that table, not in the config -- so a test that wants "the value a default run actually
    uses" has to read it from there. Parsed rather than mirrored as literals here for the same
    reason test_project_sweep_baseline_matches_pipeline_defaults reads nextflow.config: a mirrored
    copy is a second home that silently goes stale.

    test_stare_preset_parser_actually_parses below fails if this ever returns nothing, so a
    regex that stops matching cannot quietly turn every caller into a no-op assertion.
    """
    text = (Path(__file__).parents[2] / "lib" / "RegPresets.groovy").read_text()
    body = re.search(r"STARE\s*=\s*\[(.*?)\n    \]", text, re.S)
    assert body, "could not locate the STARE table in lib/RegPresets.groovy"
    row = re.search(rf"^\s*{mode}\s*:\s*\[([^\]]*)\]", body.group(1), re.M)
    assert row, f"could not locate the '{mode}' row in RegPresets.STARE"
    return {
        k.strip(): int(v) for k, v in re.findall(r"(\w+)\s*:\s*(\d+)", row.group(1))
    }


def stare_preset_modes():
    """The tier NAMES in RegPresets.STARE -- the owner of which tiers exist.

    Shares stare_preset_row's parse deliberately. A second private regex over the same
    Groovy file is the pattern tests/test_nfmodel.py exists to stop: seven guards each
    carried their own parse and each had a different blind spot. 'custom' is absent from
    the table by construction -- it is not a row, it is "start from high and apply
    overrides" -- so callers get exactly the tiers that have values.
    """
    text = (Path(__file__).parents[2] / "lib" / "RegPresets.groovy").read_text()
    body = re.search(r"STARE\s*=\s*\[(.*?)\n    \]", text, re.S)
    assert body, "could not locate the STARE table in lib/RegPresets.groovy"
    modes = re.findall(r"^\s*(\w+)\s*:\s*\[", body.group(1), re.M)
    assert modes, "parsed no tier names out of RegPresets.STARE"
    return modes


def test_stare_preset_modes_are_the_documented_three():
    """Watched failing: the parser must find the real tiers, not an empty list."""
    assert set(stare_preset_modes()) == {"high", "medium", "low"}


def test_stare_preset_parser_actually_parses():
    """The parser above must find a complete table, or every test using it silently passes."""
    rows = {m: stare_preset_row(m) for m in ("high", "medium", "low")}
    for mode, row in rows.items():
        assert set(row) == {"tile", "halo", "out_tile", "coarse_max_dim", "upsample"}, (
            f"RegPresets.STARE['{mode}'] parsed as {row} -- the table shape changed"
        )
    assert rows["high"]["tile"] > rows["low"]["tile"], "tiers are not ordered by cost"


def test_project_stare_resolution_axis_mirrors_the_valis_one():
    """Both methods' transform-resolution knob must be swept, or the head-to-head is biased.

    reg_max_image_dim (VALIS) and reg_tiled_coarse_max_dim (STARE) are the same thing for their
    respective methods: the resolution the global transform is solved at, and the dominant
    runtime-and-accuracy axis of each. Sweeping one and holding the other at its default compares
    a TUNED VALIS against an UNTUNED STARE -- a bias that is invisible in the output tables.
    """
    import yaml

    sweep = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )
    rmg = sweep["registration_method_grid"]
    # BOTH now live in the same grid. reg_max_image_dim used to be read out of flat `axes:`; that
    # is exactly the split this asymmetry hid behind, so read them the same way or the comparison
    # is being made between two different kinds of coverage again.
    valis_values = set(rmg["valis"].get("reg_max_image_dim", []))
    stare_values = set(rmg["tiled"].get("reg_tiled_coarse_max_dim", []))
    assert "reg_max_image_dim" not in sweep.get("axes", {}), (
        "reg_max_image_dim is back in flat `axes:` — as an OFAT axis it varies at ONE "
        "(memory_mode, reg_micro_reg) point, which is the 11-of-27 coverage this grid fixed"
    )
    assert len(valis_values) >= 3, (
        f"the VALIS resolution axis lost points: {sorted(valis_values)}"
    )
    assert len(stare_values) >= 3, (
        "reg_max_image_dim is swept but reg_tiled_coarse_max_dim is not (or has <3 points): "
        f"{sorted(stare_values)}"
    )
    valis_base = sweep["baseline"]["reg_max_image_dim"]
    assert min(valis_values) < valis_base < max(valis_values), (
        f"the VALIS resolution axis must bracket its default {valis_base} in both directions "
        f"— the same bar the STARE one is held to below: {sorted(valis_values)}"
    )
    # The baseline leaves this null: reg_tiled_mode='high' supplies it. Compare the axis against
    # the value a default run ACTUALLY uses -- the 'high' row -- not against the literal null.
    assert sweep["baseline"]["reg_tiled_coarse_max_dim"] is None, (
        "baseline now pins reg_tiled_coarse_max_dim directly; if that is deliberate it must also "
        "pin reg_tiled_mode='custom', or ParamUtils.validateRegPresets rejects every baseline run"
    )
    base = stare_preset_row("high")["coarse_max_dim"]
    assert min(stare_values) < base < max(stare_values), (
        f"the STARE resolution axis must bracket its default {base} in both directions, "
        f"as reg_max_image_dim does: {sorted(stare_values)}"
    )


def test_project_sweep_backends_are_crossed_at_equal_dimensionality():
    """Every backend in registration_method_grid must be crossed as richly as every other.

    THE PROPERTY THE MARGINAL GUARD ABOVE CANNOT SEE. That one asks "is each method's resolution
    axis swept, and does it bracket its default?" -- true of both backends, and it passed for the
    entire period this was broken. It compares MARGINALS. The asymmetry lived in the JOINT
    coverage:

        STARE  27/27 cells   full cube, up to 3 knobs off-default at once (8 such cells)
        VALIS  11/27 cells   a 3x3 plane + a 1-D spur, NEVER more than 2 off-default at once

    because VALIS's three knobs were split across registration_param_grid (2 of them) and a flat
    `axes:` entry (the third, varied at ONE point of the other two).

    Why it is not cosmetic: "which method wins" becomes entangled with "which method got more
    cells", and the direction flips with the estimator -- a BEST-CELL comparison flatters the
    method with more draws, a MEAN-OVER-CELLS comparison flatters the method whose cells cluster
    near its default. Neither is safe, so neither can be fixed by choosing the statistic.

    Asserts the two things that make the cross a cross: same knob count, same level count per
    knob, and a cell count equal to the full product (no holes).
    """
    import math

    import yaml

    sweep = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )
    grid = sweep["registration_method_grid"]
    assert len(grid) >= 2, "a head-to-head needs at least two backends in the grid"

    shape = {}
    for method, knobs in grid.items():
        # A pin (one value) is not an axis -- reg_tiled_mode: [custom] exists to make the other
        # knobs legal, not to be crossed. Counting it as an axis would let a backend claim a
        # dimension it does not actually vary.
        levels = sorted((len(v) for v in knobs.values() if len(v) > 1), reverse=True)
        cells = math.prod(levels) if levels else 0
        shape[method] = (len(levels), levels, cells)

    n_axes = {m: v[0] for m, v in shape.items()}
    assert len(set(n_axes.values())) == 1, (
        "backends are crossed over different NUMBERS of knobs, so one has interaction terms the "
        f"other cannot have: {n_axes}"
    )

    levels = {m: tuple(v[1]) for m, v in shape.items()}
    assert len(set(levels.values())) == 1, (
        f"backends are crossed at different numbers of LEVELS per knob: {levels}"
    )

    # And the plan must actually contain the full product for each -- a grid can be declared
    # symmetric and still be expanded into holes.
    plan = build_run_plan(sweep, repeats=1)
    for method, (_, _, cells) in shape.items():
        got = len(
            {
                tuple(sorted((k, v) for k, v in r.items() if k in grid[method]))
                for r in plan
                if r["varied_axis"] == f"registration_method_grid:{method}"
            }
        )
        assert got == cells, (
            f"{method} declares a {cells}-cell cross but the plan expands {got} distinct cells"
        )


def test_project_sweep_has_no_dead_axes():
    # An OFAT axis on a param that is gated behind another param is a no-op run. These belong in
    # a per-method grid (gate pinned on), never in flat `axes`.
    import yaml

    sweep = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )
    gated = {
        "reg_tiled_tile",
        "reg_tiled_gate_tre",  # registration_method=tiled
        "reg_tiled_halo",
        "reg_tiled_upsample",
        "reg_tiled_out_tile",
        "reg_tiled_coarse_max_dim",
        "seg_n_tiles_x",
        "seg_n_tiles_y",  # seg_method=stardist
        "seg_instantseg_tile_size",
        "seg_instantseg_batch_size",  # seg_method=instantseg
        "seg_cellsam_block_size",
        "seg_cellsam_bbox_threshold",  # seg_method=cellsam
    }
    dead = gated & set(sweep["axes"])
    assert not dead, (
        f"gated params used as flat OFAT axes (they would measure nothing): {sorted(dead)}"
    )


def test_segmentation_grid_pins_method_and_crosses_own_params():
    sweep = {
        "strategy": "ofat",
        "baseline": {
            "target_px": 4096,
            "n_channels": 2,
            "seg_method": "stardist",
            "seg_n_tiles_x": 16,
            "seg_n_tiles_y": 16,
        },
        "segmentation_grid": {
            "stardist": {"seg_n_tiles_x": [8, 16, 32], "seg_n_tiles_y": [8, 16, 32]},
            "cellsam": {"seg_cellsam_block_size": [256, 400, 512]},
            "instantseg": {"seg_instantseg_tile_size": [256, 512, 1024]},
        },
        "axes": {},
    }
    plan = build_run_plan(sweep, repeats=1)
    sd = [r for r in plan if r["varied_axis"] == "segmentation_grid:stardist"]
    cs = [r for r in plan if r["varied_axis"] == "segmentation_grid:cellsam"]
    ins = [r for r in plan if r["varied_axis"] == "segmentation_grid:instantseg"]
    assert len(sd) == 9 and len(cs) == 3 and len(ins) == 3  # 3x3, 3, 3
    assert all(r["seg_method"] == "stardist" for r in sd)
    assert all(r["seg_method"] == "cellsam" for r in cs)
    assert all(r["seg_method"] == "instantseg" for r in ins)
    assert {(r["seg_n_tiles_x"], r["seg_n_tiles_y"]) for r in sd} == {
        (x, y) for x in (8, 16, 32) for y in (8, 16, 32)
    }
    assert {r["seg_cellsam_block_size"] for r in cs} == {256, 400, 512}


def test_project_sweep_has_no_distributed_surface():
    # the distributed/tiled registration path was archived out of the pipeline; the sweep must not
    # reference any of its params or grids (they would pass dead --param flags to the pipeline).
    import yaml

    sweep = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )
    assert "distributed_grid" not in sweep
    assert "distributed_tiling_grid" not in sweep
    plan = build_run_plan(sweep, repeats=1)
    for r in plan:
        for k in r:
            assert not k.startswith("reg_dist"), (
                f"distributed param {k} leaked into the run plan"
            )
        assert "reg_distributed_tiling" not in r


def test_project_sweep_enables_qc_signals():
    # the paper's accuracy + segmentation-quality tables are harvested from pipeline QC, so the
    # baseline must turn those QC steps on.
    import yaml

    sweep = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )
    assert (
        sweep["baseline"]["reg_qc"] == 2
    )  # staged registration QC (dice + displacement)


# Baseline values that deliberately DIFFER from the shipped default, each with the
# reason. Kept deliberately small and specific: the baseline run is supposed to BE
# the shipped config, and every entry here is a place where it is not.
#
# An entry whose pipeline default has caught up with the baseline is a hard failure
# below, not a silent pass -- the same shrink-only discipline the debt allowlists in
# tests/ follow. A stale exemption is how the seg_method desync survived.
BASELINE_DEVIATIONS = {
}


def test_project_sweep_baseline_matches_pipeline_defaults():
    """Every baseline value must equal the shipped nextflow.config default.

    Read straight from nextflow.config rather than mirrored as literals here. The previous
    hand-maintained assertion list said "update BOTH sides together if a pipeline default changes",
    and that is exactly what failed: main flipped seg_method from 'stardist' to 'instantseg' and the
    sweep kept claiming stardist was the pipeline default, because seg_method was never asserted.
    Deriving the expected values means a new pipeline default cannot silently desync the baseline --
    the same approach tests/check_param_consistency.py takes for the schema.

    BASELINE_DEVIATIONS above is the escape hatch, and it is narrow on purpose: a
    deviation has to be written down with its reason, and it stops being allowed the
    moment the pipeline default agrees with it again.
    """
    import yaml

    base = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )["baseline"]
    config_defaults = pipeline_param_defaults()

    # Synthetic matrix dimensions: owned by generate_matrix.py, not pipeline params.
    synthetic = {"target_px", "n_channels", "n_register_images"}

    checked = 0
    for name, baseline_value in base.items():
        if name in synthetic:
            continue
        if name in BASELINE_DEVIATIONS:
            continue
        assert name in config_defaults, (
            f"sweep baseline sets '{name}', which is not a pipeline param in nextflow.config"
        )
        expected = config_defaults[name]
        if isinstance(expected, Unparsed):
            continue  # Groovy literal the parser cannot evaluate; nothing to compare
        assert baseline_value == expected and (
            isinstance(baseline_value, bool) == isinstance(expected, bool)
        ), (
            f"sweep baseline '{name}' = {baseline_value!r} but nextflow.config default is "
            f"{expected!r}. The baseline run must BE the shipped config -- update sweep.yaml "
            f"(or, if the pipeline default moved deliberately, update both together)."
        )
        checked += 1

    assert checked > 20, (
        f"only {checked} baseline params checked -- parser likely broke"
    )


def test_every_baseline_deviation_is_still_a_deviation():
    """A deviation the pipeline has caught up with is a stale exemption, and a stale
    exemption is how the seg_method desync survived: the entry keeps a param out of
    the check above forever, so the NEXT divergence on that param is invisible.

    Also asserts each entry is a real baseline key -- an entry naming a param the
    sweep no longer sets exempts nothing and should be deleted.
    """
    import yaml

    base = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )["baseline"]
    config_defaults = pipeline_param_defaults()

    problems = []
    for name in BASELINE_DEVIATIONS:
        if name not in base:
            problems.append(
                f"BASELINE_DEVIATIONS names '{name}', which the sweep baseline does not "
                f"set -- the entry exempts nothing; delete it"
            )
            continue
        expected = config_defaults.get(name)
        if isinstance(expected, Unparsed):
            continue
        if base[name] == expected:
            problems.append(
                f"BASELINE_DEVIATIONS names '{name}', but the baseline value "
                f"{base[name]!r} now EQUALS the pipeline default. The deviation is "
                f"gone; delete the entry so the param is checked again."
            )
    assert not problems, "\n".join(problems)


def test_project_sweep_all_configs_share_columns():
    # run_sweep.sh maps EVERY csv column to --param, so a config missing a key another config has would
    # write a blank cell -> `--param ""` for those runs. Guard: all configs carry the same key set.
    import yaml

    sweep = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )
    plan = build_run_plan(sweep, repeats=1)
    keysets = {frozenset(r) for r in plan}
    assert len(keysets) == 1, (
        "configs have differing columns (baseline must declare every swept param)"
    )


def test_project_sweep_caps_and_grids():
    import yaml

    sweep = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )
    sg = sweep["scaling_grid"]
    assert (
        max(sg["target_px"]) == 65536
    )  # largest benchmarked size (see sweep.yaml scaling_grid)
    assert sg["n_channels"] == [2, 4]  # 1 not benchmarked, max 4
    # registration is measured ACROSS sizes, not only at baseline
    rg = sweep["registration_grid"]
    assert max(rg["target_px"]) == 65536
    assert rg["n_register_images"] == [4, 8]
    # input-scaling dimensions are owned by the grids, not double-covered as OFAT axes
    for k in ("target_px", "n_channels", "n_register_images"):
        assert k not in sweep["axes"], f"{k} should not be an OFAT axis"


# ---------------------------------------------------------------------------
# Coverage registry: every pipeline param is either SWEPT or excluded here with
# a reason. test_every_pipeline_param_is_swept_or_excused fails on any param that
# is in neither set, so a param added to nextflow.config cannot silently escape
# the benchmark. This is the coverage counterpart to
# test_project_sweep_baseline_matches_pipeline_defaults (which pins the values).
# ---------------------------------------------------------------------------
NOT_SWEPT = {
    # --- run identity / where the data is. Not knobs: they select the job, not its cost. ---
    "input": "samplesheet path — supplied per run by run_sweep.sh",
    "outdir": "output path — supplied per run by run_sweep.sh",
    "mode": "'standard' is the only value on this branch (add_cycle lives on dev) — a run label, not a knob",
    "config_profile_name": (
        "nf-core provenance string: a display string echoed into the run header and the QC "
        "report. Null by default and set by a site profile; it reaches no process and has "
        "no cost curve."
    ),
    "config_profile_description": "nf-core provenance display string — see config_profile_name",
    "reg_valis_max_keypoints": (
        "keypoints per image for SuperPoint/SuperGlue. null = VALIS's own 20000, what "
        "every measured run has used; the tma profile pins 2000. Lowering it changes "
        "WHICH MATCHES EXIST, i.e. the registration itself, so an arm at a different "
        "value is a different method, not a cheaper run of the same one -- it belongs "
        "in a registration-quality study with landmark TRE, not in the cost sweep."
    ),
    "start": "step gate — every sweep run is a full pipeline; a partial run is not comparable",
    "stop": "step gate — see start",
    "cleanup_work": (
        "post-run disk hygiene, and ACTIVELY HOSTILE to benchmarking. It sets Nextflow's "
        "`cleanup`, which deletes the work directory once a run succeeds -- and the work "
        "directory is where the trace lives. Every number the analysis layer computes is "
        "parsed out of <run>/trace/trace.txt, so a swept-on arm would report nothing at "
        "all while exiting 0. It also cannot change what a task costs: it runs after the "
        "last task has finished."
    ),
    # --- site / scheduler. Properties of the cluster, not of the pipeline. ---
    # max_cpus and max_memory ARE params again, and are excused rather than swept.
    # This note previously said they were "no longer params at all"; that was true of the
    # window in which they were undeclared, and stopped being true at :bug: 90bfdbe
    # ("Declare the resource ceilings null so schema and params agree"). They are now
    # DECLARED NULL in nextflow.config (:396-397) and listed in nf-schema's `required`,
    # so a run must still supply them -- the rejection message is identical, only the
    # mechanism moved from "undeclared" to "declared null + required".
    #
    # The reason they are not swept is unchanged and is the same one max_time carries: a
    # resource ceiling is a claim about someone else's hardware, and varying it benchmarks
    # the machine rather than the pipeline. They are supplied by the SWEEP_PROFILE
    # (`docker,local` by default, `singularity,ieo` on the cluster) and NOT by
    # benchmark.config, because a `-c` file overrides a profile's params and would clobber
    # the site ceiling.
    "max_cpus": "resourceLimits clamp — a site ceiling; varying it benchmarks the machine",
    "max_memory": "resourceLimits clamp — see max_cpus",
    "max_time": "resourceLimits clamp — see max_cpus",
    "max_forks": (
        "cluster concurrency, not pipeline cost. It caps how many tasks of one process run "
        "at once, so varying it changes how fast the SWEEP drains, never what a task costs. "
        "Same category as max_time: it benchmarks the machine. It is also "
        "read EAGERLY at conf/modules.config parse time (Math.min(own, params.max_forks)), so "
        "run_sweep.sh must pass it on the CLI -- a -c file sets it too late to reach the clamps."
    ),
    "queue_size": (
        "cluster concurrency, not pipeline cost -- see max_forks. The two are a pair and the "
        "lower binds, so benchmark.config raises both together on the CLI."
    ),
    "concurrency": (
        "cluster concurrency, not pipeline cost -- the COUPLED knob that sets the max_forks/"
        "queue_size pair above (nextflow.config derives maxForks from it, and queueSize from "
        "it x4, unless either is set explicitly). Excused for exactly the reason its two "
        "derivatives are: it changes how fast the SWEEP drains, never what a task costs. "
        "benchmark.config keeps setting max_forks/queue_size directly on the CLI rather than "
        "going through this knob, because the clamps in conf/modules.config read max_forks "
        "EAGERLY at parse time -- see max_forks.\n"
        "Note also WHERE the derivation lives: in the executor/process scopes, not in the "
        "params block, because the params block is evaluated BEFORE the CLI is applied and a "
        "derived default written there would silently ignore --concurrency."
    ),
    "slurm_account": "scheduler identity",
    "slurm_partition": "scheduler identity",
    "slurm_qos": "scheduler identity",
    "gpu_type": "site GPU selector — the hardware under test, not a pipeline knob",
    # --- asset paths. Site-local files; a path string has no cost curve. ---
    "segmentation_model": "StarDist model NAME — an asset, not a knob",
    "segmentation_model_dir": "StarDist model dir — site-local asset path",
    "instanseg_model_dir": "InstanSeg model dir — site-local asset path",
    "cellsam_model_path": "CellSAM weights — site-local asset path",
    # The automated-phenotyping params (panel_spec, panel_model, pheno_alpha,
    # pheno_max_enumerate, pheno_min_cal) used to be excused here. They left
    # nextflow.config when that feature was extracted from main, and this registry is
    # not allowed to carry excuses for params that no longer exist -- a dead excuse is
    # indistinguishable from a deliberate coverage gap. If phenotyping returns, the
    # other direction of this guard will demand they be re-added.
    # --- properties of the INPUT DATA, fixed by generate_matrix.py. ---
    "pixel_size": (
        "a property of the synthetic images, not a cost knob. It no longer has a pipeline "
        "default (it was 0.325, one scanner's value standing in for everyone's), so it is "
        "REQUIRED and benchmarks/configs/benchmark.config pins it to the scale "
        "generate_matrix.py actually draws at."
    ),
    "nuclear_markers": "channel naming contract — the matrix generator fixes the panel",
    # allow_auto_reference was excused here as "samplesheet semantics — the generated sheet
    # always names a reference". The param is gone, and so is the promotion it enabled: a
    # patient with no declared reference is now an error everywhere but add_cycle.
    # --- the CSE segmentation-quality scorer (restored on this branch, opt-in). ---
    # Not a cost knob of the PIPELINE: SEG_QUALITY_EVAL is a scorer bolted onto the
    # end, and the synthetic sweep cannot exercise it meaningfully anyway — CSE's
    # metrics assume multiple cell types differing in channel expression, and the
    # matrix's extra channels are channel 0 duplicated with jitter. It is measured
    # on REAL slides by the segmentation arm in benchmarks/configs/arms.yaml
    # (score_with: cse), which is where a segmentation-quality number is defensible.
    "skip_seg_quality_eval": "opt-in scorer; exercised by arms.yaml's segmentation arm, not the synthetic sweep",
    "cse_pixel_size_um": "an image property, not a knob — inferred from metadata when null",
    "cse_max_pixels": (
        "a MEASUREMENT setting, and not a comparable one: the composite QualityScore "
        "shifts with the binning factor, so sweeping it would produce scores that "
        "cannot be read against each other. Held fixed across a cohort by design."
    ),
    # segeval_tag was excused here as "container tag -- an asset selector, not a cost
    # knob". The param is GONE: v1.0.0's container work (release plan 06) tags the
    # segeval image with manifest.version instead of a standalone param, so there is
    # nothing left to excuse. The other direction of this guard would have demanded
    # its return if the param had survived.
    # --- developer / trace plumbing. ---
    "debug_channels": "debug output only",
    "dry_run": "dry_run does not execute the pipeline",
    "enable_trace": "held ON — the sweep's own measurement plumbing (size_logs feed the regression)",
    "trace_dir": "trace output path",
    # --- gated behind a backend/method choice: a flat OFAT run would be a no-op. ---
    # Same rule test_project_sweep_has_no_dead_axes enforces: pin the gate, cross the knob.
    "seg_pmin": "seg_method=stardist only (starDistCommonFlags) — belongs in segmentation_grid.stardist",
    "seg_pmax": "seg_method=stardist only — see seg_pmin",
    "seg_prob_thresh": "seg_method=stardist only — see seg_pmin",
    "seg_expand_distance": (
        "reaches SEGMENT only via starDistCommonFlags; at the instantseg baseline it would move "
        "only SEG_QC_GEOJSON's contours — i.e. change the accuracy MEASUREMENT rather than the "
        "pipeline's own segmentation. Belongs in segmentation_grid.stardist."
    ),
    "seg_instantseg_model": "seg_method=instantseg only — an asset name, not a cost knob",
    "seg_instantseg_target": "seg_method=instantseg only — selects which outputs are produced",
    "seg_cellsam_overlap": "seg_method=cellsam only — add to segmentation_grid.cellsam if needed",
    "seg_cellsam_use_wsi": "seg_method=cellsam only — see seg_cellsam_overlap",
    "reg_tiled_halo": "registration_method=tiled only — secondary accuracy knob, held at default",
    "reg_tiled_upsample": "registration_method=tiled only — see reg_tiled_halo",
    "reg_tiled_out_tile": "registration_method=tiled only — write-side I/O, held at default",
    "reg_tiled_nuclear_index": (
        "registration_method=tiled only — a channel index, not a cost knob. Renamed "
        "from reg_tiled_dapi_index, and its default is now null (resolve from the "
        "slide's channel metadata) rather than 0; sweeping it would only re-test "
        "MarkerUtils' resolution, which tests/ already covers."
    ),
    # --- secondary knobs deliberately held at defaults (add an axis if a curve is ever needed). ---
    "reg_micro_reg_fraction": "secondary micro-registration knob; reg_micro_reg (the DEPTH) is crossed instead",
    "preprocess_qc_scale_factor": "preprocess-QC render scale; qc_scale_factor already prices QC rendering",
    "spatialdata_residual_join_max_px": "join tolerance — changes the export's matching, not its cost",
    "embed_masks": "embeds masks in the OME-TIFF; a niche output-format toggle",
    # --- QC gates held ON: the paper's quality tables are harvested from these artifacts. ---
    # Flipping them off yields runs absent from every quality table (same caveat as reg_qc<2).
    "skip_preprocess_qc": "held ON — turning QC off drops the run from the quality tables",
    "skip_registration_qc": "held ON — see skip_preprocess_qc",
    "skip_postprocessing_qc": "held ON — see skip_preprocess_qc",
    "skip_final_qc_report": "held ON — the final report is where versions/size_logs are aggregated",
}


def _sweep_covered_params(sweep) -> set:
    """Every pipeline param the sweep touches, in the baseline or any axis/grid."""
    covered = set(sweep["baseline"]) | set(sweep.get("axes", {}))
    for grid in ("scaling_grid", "registration_grid"):
        covered |= set(sweep.get(grid, {}))
    for grid in ("segmentation_grid", "registration_method_grid"):
        for per_method in sweep.get(grid, {}).values():
            covered |= set(per_method)
    return covered


def test_every_pipeline_param_is_swept_or_excused():
    """No pipeline param may be silently unbenchmarked.

    The benchmark's claim is that it characterises the pipeline. A param that is neither
    swept nor listed in NOT_SWEPT is a hole in that claim -- and the failure mode is silent,
    because nothing else in the suite reads nextflow.config's full param list. Adding a param
    to nextflow.config now forces a decision here: sweep it, or say why not.
    """
    import yaml

    sweep = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )
    params = set(pipeline_param_defaults())

    unaccounted = params - _sweep_covered_params(sweep) - set(NOT_SWEPT)
    assert not unaccounted, (
        "pipeline params neither swept nor excused: "
        + ", ".join(sorted(unaccounted))
        + "\nEither add an axis/baseline entry in benchmarks/configs/sweep.yaml, or add the "
        "param to NOT_SWEPT with the reason it is not benchmarkable."
    )


def test_not_swept_registry_has_no_stale_entries():
    """NOT_SWEPT must not excuse a param that no longer exists, or one now swept.

    Without this, a deleted param leaves a dead excuse behind (and a param later promoted to
    a real axis keeps a stale 'not benchmarked' note contradicting the sweep).
    """
    import yaml

    sweep = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )
    params = set(pipeline_param_defaults())

    ghosts = set(NOT_SWEPT) - params
    assert not ghosts, (
        f"NOT_SWEPT excuses params that no longer exist in nextflow.config: {sorted(ghosts)}"
    )

    contradictory = set(NOT_SWEPT) & _sweep_covered_params(sweep)
    assert not contradictory, (
        f"params are BOTH swept and listed as not-swept: {sorted(contradictory)} -- drop the NOT_SWEPT entry"
    )


def test_every_not_swept_entry_states_a_reason():
    for name, reason in NOT_SWEPT.items():
        assert isinstance(reason, str) and len(reason) > 15, (
            f"NOT_SWEPT['{name}'] must state a real reason, got {reason!r}"
        )


def _schema_enum(name):
    """The enum nextflow_schema.json declares for a param."""
    import json

    doc = json.loads((REPO_ROOT / "nextflow_schema.json").read_text())

    def walk(node):
        if isinstance(node, dict):
            entry = (node.get("properties") or {}).get(name)
            if isinstance(entry, dict) and "enum" in entry:
                return entry["enum"]
            for value in node.values():
                hit = walk(value)
                if hit:
                    return hit
        elif isinstance(node, list):
            for value in node:
                hit = walk(value)
                if hit:
                    return hit
        return None

    return walk(doc)


def test_every_shipped_registration_method_is_actually_swept():
    """Every method in registration_method's enum is either the baseline or has a
    grid block. Nothing is offered by the pipeline and left unmeasured.

    WRITTEN FOR `ashlar`, WHICH IS NO LONGER THE CASE IT CATCHES. When this was
    added, ashlar was a schema-valid registration_method with three params and its
    own adapter, and registration_method_grid contained only `tiled`; the coverage
    guard passed anyway, because it counts a param as covered when it merely
    appears in `baseline` -- so "declared and never varied" was indistinguishable
    from "swept". Appearing in baseline is a DEFAULT, not coverage: every run
    shares it, so nothing is being measured against anything.

    :fire: 6a54479 then removed the ashlar BACKEND from the pipeline for v1.0.0.
    The enum is now ['valis', 'tiled'] and ashlar survives as an EXTERNAL arm
    (benchmarks/ashlar/, arms.yaml) rather than a registration_method, so the
    original offender is gone -- but the enum is READ FROM THE SCHEMA rather than
    restated here, so this keeps working across that change and covers a third
    backend the day one is added. That is the reason it is not deleted with the
    case that motivated it.
    """
    import yaml

    shipped = set(_schema_enum("registration_method") or [])
    assert shipped, "registration_method has no enum in nextflow_schema.json"
    sweep = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )
    grid = sweep.get("registration_method_grid") or {}
    swept = (
        set(grid)
        if isinstance(grid, dict)
        else {e.get("registration_method") for e in grid}
    )
    # The baseline method is the anchor every OFAT run uses, so it IS measured
    # even though it has no grid block of its own.
    swept.add(sweep["baseline"].get("registration_method"))
    missing = shipped - swept
    assert not missing, (
        f"shipped but never benchmarked: {sorted(missing)} -- a backend the "
        f"pipeline offers and the paper never measures. Add a "
        f"registration_method_grid block for it, or remove it from the schema "
        f"enum; a third option is not honest."
    )


def test_baseline_membership_does_not_count_as_coverage():
    """The specific weakness the check above closes, asserted on the mechanism.

    A param that appears ONLY in `baseline` must not show up in the swept-axis
    set. If it did, `identity_columns` would claim it identifies a run while
    every run shares its value -- and the coverage guard would count it as
    benchmarked.
    """
    import yaml

    from benchmarks.analysis.lib import contract

    sweep = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )
    axes = contract.axes(sweep)
    baseline_only = set(sweep.get("baseline") or {}) - axes
    assert baseline_only, (
        "every baseline entry is also swept, so this guard is checking an empty "
        "set -- cleanup_level and the tier-owned nulls should be in here"
    )
    assert not (baseline_only & axes), (
        f"declared-only params leaked into the axis set: {sorted(baseline_only & axes)}"
    )
