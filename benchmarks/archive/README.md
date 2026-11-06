# benchmarks/archive — what the paper does not report

Everything here was part of the regbench harness and is **not reported in the 261105
manuscripts** (the main text and the supplement). It was archived on 2026-11-06 at the
user's ruling ("Still dont delete, archive if you can") instead of being deleted.

**Nothing here is imported, run, linted or tested.** `ruff.toml` excludes this
directory, `conftest.py` here stops pytest collecting it, and no file outside it imports
from it. The paths mirror where each file used to live (`benchmarks/archive/regbench/x.py`
was `benchmarks/regbench/x.py`), so a restore is a `git mv` back plus the import or
dispatch line named below.

The whole harness as it stood before the archive, every file at its live path, is the tag
**`archive/integrate-benchmarking-pre-prune-2026-11-06`**. Check that tag out to run any
of this as it ran.

## Runs: still in `variants.py`, paused

The unreported runs were **not** moved. They stay in `benchmarks/regbench/variants.py`'s
`RUNS` with `active=False`: not launched, expected or counted, but `run_of` still resolves
them and `regbench runs --all` lists them. `variants.REPORTED_RUNS` names the 11 the
paper reports; `variants.ARCHIVED_RUNS` names the other 88. Restoring one means adding its
name to `REPORTED_RUNS`.

| family | archived runs | why |
|---|---|---|
| STARE | the preprocessing x stride cross (`stare_s128_*`, `stare_s256_*`, `stare_clahe`, ...) and the `stare_medium*` / `stare_low*` tiers | the paper reports `stare` and `stare_local4`; the pipeline has had no STARE tiers since the merge |
| DeeperHistReg | `deeperhistreg_{cpu,gpu,fluo_cpu,fluo_gpu}` (the default, not HR, settings) | the paper reports the HR rows |
| ASHLAR | `ashlar_clean`, `affine_ashlar_clean` and the `ashlar_anone_*` / `ashlar_astare_*` noise x step cross | the paper reports `ashlar` and `affine_ashlar` as shipped |

The synthetic scaling set keeps `stare`, `valis120` and `valis120_hires` active; its
eight other STARE cells are `variants.ARCHIVED_SYNTHETIC_RUNS`.

## Public datasets: out of `regbench.PUBLIC`

`hyreco` (landmarks), `mcmicro`, `phenocycler` and `rashid` (nuclei) moved from
`regbench.PUBLIC` to `regbench.ARCHIVED_PUBLIC`. `PUBLIC` holds `synthetic` alone, and
`submit_cohort.sh`'s `PUBLIC_DATASETS` / `PUBLIC_KINDS` / `LANDMARK_SETS` follow it, so a
launch, prepare or score naming an archived set is refused. The landmark scoring that
frozen files still carry (`landmark_io.py`, the landmark branches of `cases.py` and
`regimes.py`) stays where it is, unreachable: those files are fingerprinted in
`benchmarks/tests/regbench/method_code_freeze.json` and editing them would invalidate
the results.

## Files moved here

| file | was | what it did | restore |
|---|---|---|---|
| `regbench/datasets/hyreco.py` | `benchmarks/regbench/datasets/` | `regbench prepare hyreco`: the HyReCo cases | its `prepare` parser, in `regbench/cli_archived.txt` |
| `regbench/download/hyreco.sh`, `common.sh` | `benchmarks/regbench/download/` | fetched HyReCo (`common.sh` was its only user) | — |
| `regbench/landmarks.py` | `benchmarks/regbench/` | the HyReCo landmark scorer (`tables/hyreco/`) | `_score_landmarks` in `regbench/score_archived.py`, `"landmarks"` back in `score.SCORED_KINDS` |
| `regbench/notrun.py`, `regbench/truth.py` | `benchmarks/regbench/` | the landmark scorer's "not run" rows and landmark truth | imported by `landmarks.py` only |
| `regbench/paper/landmarks.py` | `benchmarks/regbench/paper/` | the landmark paper tables (`public_landmarks`, ...) | its branch in `paper.tables.write_all` |
| `regbench/paper/cpu_scaling.py` | `benchmarks/regbench/paper/` | `regbench cpu-scaling`: the CPU-count scaling table | its parser, in `regbench/cli_archived.txt`; `figS_cpu_scaling` |
| `regbench/paper/plots_archived.py` | functions of `paper/plots.py` and `paper/cost_plots.py` | `fig3d_stare_tiers`, `public_dice_best`, `public_landmarks_best` (and `_public_dirs`), `figS_synth_cross_scaling`, `figS_synth_vs_cohort`, `figS_cpu_scaling` | paste back, re-add to `PLOTS`/`NEEDS` or `cost_plots.FIGURES` |
| `regbench/score_archived.py` | a function and a branch of `score.py` | `_score_landmarks` and its dispatch line | as above |
| `regbench/cli_archived.txt` | parsers of `cli.py` | `prepare hyreco`, `cross-runs`, `cpu-scaling` and their dispatch, verbatim | paste back |
| `tests/regbench/test_hyreco.py`, `test_submit_landmarks.py`, `make_landmark_fixtures.py`, `fixtures/hyreco/` | `benchmarks/tests/regbench/` | the HyReCo preparation, scorer and launch tests and their fixtures | with the modules |
| `tests/regbench/test_cpu_scaling.py` | `benchmarks/tests/regbench/` | the CPU-scaling table's tests | with `cpu_scaling.py` |
| `tests/regbench/test_download_scripts.py` | `benchmarks/tests/regbench/` | the download scripts' tests | with `download/` |

Two things were left in place although the paper does not report them, because reported
code still needs them: `regbench/subset.py` (`staged-results`, which the local STARE
launch uses, calls `subset.staged_stare`) and the `stare_tiers` paper table (entangled
with `paper.tables.add_excluded` and the schema; it is now fed by `stare` alone).
`score.HYRECO_TABLES` also stays in `score.TABLES`, so a clear still removes an older
scoring's files.

The one file actually deleted on this branch is `methods/valis_mirage.py` (with its runs
and `containers/valis100`), at an earlier explicit ruling; it is in the tag above and in
`git log`.
