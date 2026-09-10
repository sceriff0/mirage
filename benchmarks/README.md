# Mirage Benchmarking

Produces the **DATA** a method paper needs — tidy CSV tables, not plots — for the
*shipped* pipeline (classic VALIS registration). Three signals, all harvested from
QC the pipeline already emits (the benchmark invents no metrics):

1. **Resource scaling** — peak RAM & wall-time vs input size / channels / rounds (Nextflow trace).
2. **Registration accuracy** — `dice_matched` + centroid **displacement (µm)** from the staged
   DAPI-nuclei QC (`reg_qc=2`; `bin/warp_seg_qc.py`). Landmark-free — and *only* landmark-free:
   this benchmark carries no ground truth of any kind, so its accuracy numbers rank
   configurations against each other and cannot say any of them is correct. See section B, and
   Supplementary Figure S6 (`docs/figures/accuracy-schematic.html`).

The heavy step (the sweep) runs on a **cluster** with the pipeline containers;
matrix generation and the DATA emit run **locally**.

```
   generate_matrix.py --sweep  ->  build_run_plan.py --sweep  ->  run_sweep.sh  (cluster)
                                                                       |
                                             per-run trace + size logs + QC JSONs
                                                                       |
                                        make_tables.py  ->  benchmarks/paper_data/*.csv (+ *.dict.md)
```

!!! note
    The distributed/tiled registration path was archived out of the pipeline (git tag
    `archive/tiled-valis-2026-07-24`); registration is classic `REGISTER` only, so the
    old classic-vs-distributed benchmark was removed.

!!! note "This is the SYNTHETIC sweep"
    It measures how cost SCALES. To choose a configuration — an arm ranking on the
    REAL study slides — see `docs/benchmarks_real.md` (`benchmarks/configs/arms.yaml`).

**Where the plots come from.** This layer emits **data**, plus `scaling_*.pdf/svg`
from `make_figures`. Every other figure is rendered by the consumer,
`../ihc_method`, whose `code/benchmark_plots.R` + `code/registration_accuracy_plots.R`
are the maintained descendants of a `plots.R` that used to live here. That copy was
deleted rather than kept in sync: it was referenced by nothing, tested by nothing,
and had drifted well behind the two files that actually render.

Design record: `docs/superpowers/specs/2026-07-24-benchmark-paper-data-design.md`.
Rendered overview: `docs/benchmarks.md`.

## Prerequisites

- **Local (DATA emit + tests):** numpy, pandas, scikit-learn, tifffile, PyYAML.
  (`matplotlib` only for the optional figures.) No real data
  needed for the test suite.
- **Cluster (the sweeps):** Nextflow >= 25.04, Docker/Singularity, the pipeline
  containers (including the VALIS image), and `bash`. `run_sweep.sh` /
  `run_arms.sh` use indexed-array lookups (no `declare -A`), so macOS
  `/bin/bash` 3.2 works for the tested paths; if you hit an array error, `brew install bash`.

Verify the whole harness with no data at all:

    python -m pytest benchmarks/tests -q          # -> all green on synthetic fixtures

---

## A. Resource / parameter sweep

### A1 — Generate the input matrix

    python benchmarks/generate_matrix.py \
        --source /path/to/your_source.ome.tif \
        --outdir bench_matrix \
        --sweep benchmarks/configs/sweep.yaml

- **`--sweep` is the easy path:** it reads the sweep and generates exactly the cells +
  moving panels the sweep needs — deriving `--target-px`, `--n-channels`, `--paired`, and
  `--n-moving` (= `max(n_register_images) - 1`) automatically. Input-scale cells come from
  the sweep's `scaling_grid` (sizes × channels); the shipped sweep caps size at **65536**
  and channels at **{2, 4}** (1-channel isn't benchmarked), so `--sweep` builds a **6 × 2 =
  12-cell** matrix. Moving panels are generated **per cell, plan-aware**: each cell gets
  exactly as many panels as the runs that touch it consume (derived from the real run
  configs). Cells that only pair-register get 1; cells the `registration_grid` runs at 8
  rounds get 7 — including the big ones, since those *are* measured at multiple rounds. No
  panels are made that no run opens. No manual sync between the matrix and the sweep.
  (Explicit `--target-px` / `--n-channels` / `--n-moving` (forces a uniform count) /
  `--paired` still override; `--seed 0` by default.)
- **Input:** one image you supply (any Bio-Formats/tifffile-readable format; **uint8 or uint16**).
- **Output:** `bench_matrix/px{px}_ch{n}.ome.tif` per (size x channel) cell; with pairing,
  also `px{px}_ch{n}_moving{j}.ome.tif` (j = 1..n-moving) for n>=2 cells, each a distinct
  panel with distinct channel names; plus `bench_matrix/matrix_manifest.csv`
  (`cell_id,target_px,width,height,n_channels,bytes,path[,moving_paths]`, where `moving_paths`
  is a `;`-joined list). The N-image registration sweep (`n_register_images: [2, 4, 8]`)
  needs ≥7 moving panels — `--sweep` provides them.

### A2 — Expand the sweep into a run plan

    python benchmarks/build_run_plan.py \
        --sweep benchmarks/configs/sweep.yaml \
        --out bench_run_plan.csv \
        --repeats 3                       # replicate runs per config (default 3)

- **Input:** `benchmarks/configs/sweep.yaml` — edit it: `strategy: ofat|grid`, a `baseline:`
  map, the OFAT `axes:` (one knob varied off baseline), and two input-scaling grids:
  `scaling_grid:` (`target_px` × `n_channels`) and `registration_grid:` (`target_px` ×
  `n_register_images`, at a fixed channel count). Both are full cross-products because the
  three input-scaling dimensions — pixels, channels-per-round, and **number of registered
  rounds** — each drive memory (REGISTER's input is the sum of all rounds; merged channels
  grow with round count), so each must be measured across sizes, not at one baseline cell.
  Do NOT also list `target_px` / `n_channels` / `n_register_images` under `axes:` — the
  grids own input scale, and double-listing would confound the regression.
- **A knob that is only live under another setting must NOT be a flat OFAT axis.** Varying
  it off the baseline changes a value the pipeline never reads: the run costs full price and
  measures nothing. `sweep.yaml` has two structures for these, and
  `test_project_sweep_has_no_dead_axes` enforces the rule:
  - `segmentation_grid:` — pins `seg_method` per backend and crosses that backend's own knobs
    (StarDist tile grid; InstanSeg tile size × batch size; CellSAM block size × bbox threshold).
    All three shipped backends are covered.
  - `registration_method_grid:` — pins `registration_method` and crosses that method's **tier**
    with its **refinement knob**. **One grid, both methods, at equal dimensionality** — 2 knobs ×
    3 levels = **9 cells each**: `valis` crosses `memory_mode` {low, medium, high} ×
    `reg_micro_reg` {0, 1, 2}; `tiled`/STARE crosses `reg_tiled_mode` {low, medium, high} ×
    `reg_tiled_gate_tre` {0.5, 1.0, 2.0}. Both tiers are the same three-rung resolution ladder
    (high → medium → low = 2048 → 1024 → 512 px; VALIS uses SuperPoint+SuperGlue at every rung, STARE's
    row moves tile, halo, out_tile, coarse anchor and upsample together — `lib/RegPresets.groovy`),
    so `low` against `low` is the same question asked of both methods. `reg_max_image_dim`
    (VALIS, ungated, live at every tier) is an OFAT axis.

    **Why the tier and not the knobs (2026-09-10).** The grid used to pin `reg_tiled_mode=custom`
    and cross `reg_tiled_tile` × `reg_tiled_gate_tre` × `reg_tiled_coarse_max_dim` (27 cells)
    against VALIS's `memory_mode` × `reg_micro_reg` × `reg_max_image_dim` (27). Equal in count —
    but STARE's shipped presets never ran: `custom` starts from the `high` row, so only the
    (2048, 2048) cell was a tier; the 1024 cell kept `high`'s halo/out_tile/upsample and 512 was
    not a level. The tier knobs are tier-owned (`ParamUtils.validateRegPresets` refuses them under
    any tier but `custom`), so a tier axis and a knob cross cannot share a grid; the tier won,
    because it is what an operator chooses and what the real-sample arms compare.

    **The older asymmetry this grid fixed still matters, and is still guarded.** VALIS's knobs
    were once split across a VALIS-only grid and a flat `axes:` entry, measuring VALIS on 11 of
    its 27 cells against STARE's full 27 — "which method wins" confounded with "which method got
    more cells", with the direction flipping by estimator. `test_project_sweep_backends_are_crossed_at_equal_dimensionality`
    enforces equal knob count, equal levels and a hole-free product;
    `test_project_tiers_are_crossed_on_the_same_rungs` enforces that the first knob of each entry is
    its tier, that both list the full shipped ladder, that the two ladders resolve to the same
    pixel sizes (read from `RegPresets.groovy` and `valis_config.py`), and that no tier-owned knob
    is crossed beside a tier.

    STARE has a single execution shape (the per-tile fan-out
    `TILED_COARSE`/`TILED_REG_TILE`/`TILED_SOLVE`/`TILED_STITCH`); the `reg_tiled_fanout` flag and
    the single-task `TILED_REGISTER` path it selected were removed upstream, so there is no
    execution shape left to cross.
- **The `baseline:` map must equal the shipped `nextflow.config` defaults**, and must declare
  every param any grid or axis varies (so all configs share CSV columns).
  `test_project_sweep_baseline_matches_pipeline_defaults` reads `nextflow.config` directly and
  fails on any divergence — it is not a hand-maintained mirror.
- Some axes are **measurement settings**, not pipeline knobs: `reg_qc` changes *what is
  measured*, so its runs are comparable on cost but **not** on quality. Keep the cohort at the
  baseline value for any quality table. It carries this caveat inline.
- **`--repeats N`** emits N replicate launches per config so the analysis can report
  per-config variance (timing at n=1 is noisy — cache state, node contention). `N=3` is the
  default; `N=1` is a single-shot plan. Replicates share a `config_id`; `rep` is the index.
  Repeats multiply *runs*, not *images* (the cell image is reused across a config's reps).
- **Output:** `bench_run_plan.csv` — one row per pipeline run
  (`run_id,varied_axis,config_id,rep,<param columns incl target_px,n_channels>`).

### A3 — Launch the sweep (cluster)

    benchmarks/run_sweep.sh  bench_run_plan.csv  bench_matrix/matrix_manifest.csv  bench_results \
        [extra nextflow args, e.g. -profile singularity]

- **What it does:** for each run-plan row, resolves the cell image, writes a per-run
  samplesheet (reference + `n_register_images-1` moving panels when paired), and launches the pipeline once with
  `-profile docker -c benchmarks/configs/benchmark.config` (enables `enable_trace` +
  `enable_size_logs`), isolating each run's outputs.
- **Output per run:**
  - `bench_results/<run_id>/trace/trace.txt` — peak_rss, realtime, cpus (+ `report.html`, `timeline.html`)
  - `bench_results/<run_id>/out/size_logs/input_sizes.csv` — per-process input bytes
  - `bench_results/<run_id>/out/...` — the normal pipeline outputs

> `run_sweep.sh` parses CSV headers with indexed bash arrays. macOS `/bin/bash` (3.2)
> works; if you see an array error use a newer bash (`brew install bash`).

> **Registration in the resource sweep.** With `--paired`, `run_sweep.sh` emits a
> reference + one-or-more moving panels per patient with **distinct channel names**
> (reference `DAPI|ch1|...`, moving panel _p_ `DAPI|m{p}_1|...` — distinct so the
> pipeline's duplicate-channel guard accepts them), so VALIS registration runs and the
> registration axes (`memory_mode`, `reg_micro_reg`, `n_register_images`) are
> exercised. **`n_register_images`** (default `[2, 4, 8]`) sets how many slides register
> together (1 reference + N−1 moving panels) — this is how registration is benchmarked on
> N images, not just a pair. Generating the matrix with `--sweep` provides the moving
> panels automatically (no `--n-moving` to compute).
> (`reg_use_tiled_registration`, `reg_tile_size`, `reg_n_workers`, `reg_parallel_warping`
> were dropped — VALIS auto-tiles internally and warping is sequential, so they had no
> effect.) Pairing applies to n_channels >= 2 cells; n_channels == 1 cells are
> reference-only. Without `--paired`, registration is a single-slide passthrough.

### A4 — Emit the paper DATA (local)

    python -m benchmarks.analysis.make_tables \
        --results-root bench_results \
        --run-plan     bench_run_plan.csv \
        --outdir       benchmarks/paper_data

- **Output:** `benchmarks/paper_data/{runs_master,scaling_fits,registration_accuracy,
  registration_valis_rtre,segmentation_agreement,param_matrix}.csv`,
  each with a sibling `.dict.md` data dictionary. **This is the method-paper deliverable.**
  Column-by-column meaning is in each `.dict.md` and summarised in `docs/benchmarks.md`.
- **Two registration-accuracy signals:** `registration_accuracy.csv` (segmentation-based
  Dice/displacement from `reg_qc: 2`) and `registration_valis_rtre.csv` (VALIS's *own*
  feature-based rTRE, harvested from the summary CSVs it writes — the same numbers the
  final QC report shows). Both need registration to have actually run (paired matrix).

### A5 — Optional: figures + derived config (local)

    python -m benchmarks.analysis.make_figures \
        --results-root bench_results \
        --run-plan     bench_run_plan.csv \
        --outdir       benchmarks/analysis

> **Run it any time — the sweep does not have to be finished.** Both `make_tables` and
> `make_figures` read whatever
> runs have completed: it skips run dirs with no `trace.txt` yet, and (because Nextflow only writes
> a trace row when a *process* finishes) an in-flight run contributes just its completed processes.
> `only_successful` then drops any non-`COMPLETED` row, so the fits/means never see a partial process.
> Early on, size-varying runs may be too few for a real fit (`r2` empty ⇒ `n<3` flat fallback) — that
> is expected, not an error. Re-run it as more runs land to watch the fits firm up.

- **Output:**
  - `benchmarks/analysis/measurements.csv` — **the primary data artifact**: one tidy row per
    (run × process) with `peak_rss_gb`, `peak_vmem_gb`, `realtime_s`, `duration_s`, `cpus`,
    `input_gb`, and every swept param column joined in (`run_id`, `varied_axis`, `target_px`,
    `n_channels`, …). Read straight into R (`read.csv`); `../ihc_method` consumes exactly this file.
  - `benchmarks/analysis/resource_models.csv` — one row per process with the fitted
    `slope`, `intercept`, `r2`, `sigma`, `n` (empty `r2` ⇒ `n<3`, flat fallback). Fit uses
    only the size-varying runs (`scaling_grid` + `baseline`); OFAT-knob runs are excluded so
    they don't pile points at one input size and confound the regression.
  - `benchmarks/analysis/resource_stats.csv` — per-`(process, config_id)` replicate
    variance: `n_reps` and `mean`/`std`/`cv` of `peak_rss_gb`, `realtime_s`, etc. This is the
    error bar `--repeats` buys you (empty/degenerate when the plan has no replicates).
  - `benchmarks/analysis/figures/scaling_<PROCESS>.pdf` + `.svg` — peak-RSS-vs-input fits per process.
  - `benchmarks/analysis/modules.optimized.config` — regression-derived memory directives,
    `memory = { ( <input_gb>*slope + intercept + sigma*task.attempt ).GB }`. No `check_max()`
    wrapper — that helper does not exist in this repo; `process.resourceLimits` (top-level in
    `nextflow.config`) clamps every request against `params.max_*` centrally.
    Processes with a known input expression are **live** blocks; others are emitted as
    **commented** blocks for you to fill in (so the file is valid Groovy as-is). It never
    overwrites the live config — review and diff:

        diff conf/modules.config benchmarks/analysis/modules.optimized.config

---

## B. Ground-truth registration validation — REMOVED (twice). Nothing here is runnable.

> **Historical record.** Neither harness exists on any branch. `benchmarks/registration_eval/`
> and `benchmarks/stare_bench/` return zero paths from `git ls-tree -r benchmarking benchmarks`.
> Nothing in this section can be run; it is kept because what was LOST is a property of the
> current benchmark.

Two harnesses have occupied this slot and both are gone. The section is kept because
what was LOST is a property of the current benchmark, not history.

**1. The ANHIR/ACROBAT landmark harness** (`benchmarks/registration_eval/`) drove
`bin/register.py` and `bin/tiled_register.py` on public challenge pairs and scored true
landmark TRE/rTRE. It went because `bin/tiled_register.py` — the single-task STARE entry
point — was removed when STARE became the four-stage fan-out and exists on no branch, so
the STARE half of every pair errored. Its data was also doubly account-gated, so it never
ran in CI.

**2. The synthetic ground-truth rung** (`benchmarks/stare_bench/`) replaced it: generated
image pairs from a known displacement field, plus physics (photobleach, PSF, shot/read
noise, autofluorescence, seams, focus) and an 11-point blank-fraction sweep, scored by
exact endpoint error, landmark TRE, field quality and a gate ROC. It was frozen at
`1.0.0` and fully tested (135 tests) — and **never run above the unit rung**: zero of its
528 planned records were ever scored, no competitor was ever scored through it, and its
headline claim (STARE peak RSS ≤ 8 GB at gigapixel scale) was never measured.

**Why it went.** Its numbers could not be ranked against the arm table. It scored exact
EPE against an injected field; `docs/benchmarks_real.md` ranks configurations by tissue
Dice and centroid displacement. Two metric families sharing no column, one of which had
produced no data — and it was the ONLY consumer of the ashlar comparator, which is the
number the manuscript actually needs. ASHLAR now runs as an arm of the real-sample
benchmark instead (`arm_kind=external`, `benchmarks/run_ashlar_arm.sh`), scored by the
pipeline's own `reg_qc=2` scorer into the same table as VALIS and STARE.

**What was lost, and it is not nothing.** There is now **no ground truth of any kind** in
this benchmark — no public landmarks, no synthetic field. Every accuracy number is a
reference-free agreement measure on tissue: `dice_matched` and centroid displacement
between paired nuclei, plus VALIS's self-reported rTRE. Those rank configurations against
each other; they cannot say any of them is *correct*. Two specific consequences:

- A reviewer asking "what is the absolute registration error in microns against known
  correspondences?" has no answer here.
- The **gate-ROC family is gone entirely** — STARE's per-tile accept/reject decision was
  the one metric no competitor reports, and it was scorable only against a known
  registrable/not-registrable label. `Finding 1` from the freeze record (the default
  `reg_tiled_gate_tre=1.0` collapsing the mesh toward identity on sub-pixel fields) was
  produced by that harness and is not reproducible by anything remaining.

Recoverable from history if either is wanted again: the landmark primitives
(`landmarks.py`, `tre.py`) and the whole generator went with the deletion --
`benchmarks/registration_eval/` at `61e26ec`, `benchmarks/stare_bench/` at `cacc850`.


## C. One real run's resource profile

Independent of the sweep and the arms. Any finished mirage run with tracing on has a
`trace.txt` and a `size_logs/input_sizes.csv`; `benchmarks/analysis/run_resources.py` joins
them into `run_resources_{tasks,processes,fits,summary}.csv` (+ `.dict.md`): CPU-hours used
vs reserved, peak RSS vs the memory requested, wall-time, each against the task's input
size, with within-run fits where a process ran on ≥ 3 inputs of different size.

    make run-resources RUN=<run --outdir> [TRACE=<trace_dir>/trace.txt] IHC=../ihc_method
    # -> ihc_method/data/run_resources/, read by analysis/run_resources.Rmd there

It reuses `bin/generate_resource_report.py`'s parsers (the same code that renders the run's
`qc/mirage_resource_report.html`) and the sweep's `regress.fit_memory_model`, so its fits
are the sweep's fits applied to one run. See `docs/benchmarks_real.md` §3b.

## Inputs -> outputs at a glance

| Step | You provide | You get |
|---|---|---|
| `generate_matrix.py` | 1 source image | matrix of OME-TIFFs + `matrix_manifest.csv` |
| `build_run_plan.py` | `sweep.yaml` | `run_plan.csv` |
| `run_sweep.sh` | manifest + run plan | per-run `trace.txt` + `input_sizes.csv` + QC JSONs (`*_seg_qc.json`) |
| **`make_tables`** | results + run plan | **`paper_data/{runs_master,scaling_fits,registration_accuracy,registration_valis_rtre,segmentation_agreement,param_matrix}.csv` (+ `.dict.md`)** — the paper DATA |
| `make_figures` (optional) | results + run plan | `measurements.csv` + `resource_models.csv` + `resource_stats.csv` + `scaling_*.pdf/svg` + `modules.optimized.config` |
| `build_arm_plan.py` | `arms.yaml` + real `input.csv` | `arm_plan.csv` + `arms.csv` (label manifest) |
| `run_arms.sh` | arm plan + real `input.csv` | per-arm `<root>/<arm>/<patient>/qc/registration/*_seg_qc.json` |
| `run_ashlar_arm.sh` | preprocessed CSV + a registration arm's QC nuclei | the ashlar external baseline, in that SAME tree |

The last three are the real-sample arm sweep — see `docs/benchmarks_real.md`. The rows for
`prepare_pairs.py` / `run_registration.sh` / `aggregate_eval` were removed: those scripts
went with the landmark harness (Section B) and had been listed here naming nothing.

The fastest way to see every component work end-to-end **without any real data** is
`python -m pytest benchmarks/tests -q` — the suite drives the whole harness on tiny
synthetic fixtures.
