# Mirage benchmarking

`regbench` (`benchmarks/regbench/`) is the only benchmarking harness. Its own
README is `benchmarks/regbench/README.md`; this file only points there and
records what was removed.

## Removed harnesses

The arms, sweep, figure and ASHLAR-arm launchers and the analysis layer were
removed when `regbench` became the only harness. They are kept at the git tag
`archive/benchmarks-arms-sweep` (to be tagged at 2413456d, the last commit
before the removal). Check that tag out to read or run any of them.

Removed:

- Launchers and planners: `run_arms.sh`, `submit_arms.sh`, `run_sweep.sh`,
  `submit_sweep.sh`, `build_arm_plan.py`, `build_run_plan.py`,
  `generate_matrix.py`, `submit_matrix.sh`, `impact.py`, `plan_exact.py`,
  `arms_status.py`, `head_sizing.sh`, `redo_window.sh`, `pull_run_resources.sh`,
  `pull_to_ihc_method.sh`, `placeholder_card.py`, `trace_step.py` and the
  `params_json.py` shim (the module lives on as `benchmarks.regbench.params_json`).
- `benchmarks/configs/` (`arms.yaml`, `sweep.yaml`, `figures*.yaml`,
  `supplementary.yaml`, `benchmark.config`, `seg_arms.config`) and
  `benchmarks/analysis/` (`make_tables.py`, `make_figures.py`, `run_resources.py`
  and `lib/`), with `benchmarks/ihc/supplementary.R`.
- Figures: `reg_crop.py`, `reg_mosaic.py`, `reg_overlay.py`, `reg_zoom.py`,
  `draw_mosaics.sh`, the `submit_*` figure launchers, `build_figure_plan.py`,
  `supplementary.py`, `seg_zoom_rois.sh`, `before_after.sh`, `before_after_pairs.py`.
- The ASHLAR arm: `run_ashlar_arm.sh`, `run_ashlar_original_arm.sh`,
  `submit_ashlar.sh`, `submit_degraded.sh`, `ashlar/solve.py`, `ashlar/degrade.py`.
  `ashlar/{retile,original}.py` stay: regbench imports them.
- Earlier removals recorded here for the guards that point at this section: the
  ANHIR/ACROBAT landmark harness (`benchmarks/registration_eval/`, later
  `benchmarks/anhir/`) and `benchmarks/stare_bench/` no longer exist on any
  branch. Nothing under `benchmarks/` measures accuracy against ground truth
  other than what `regbench` scores itself.

## Tests

    python -m pytest benchmarks/tests -q
