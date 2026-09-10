# Real-sample benchmark — the arm sweep

The synthetic sweep ([Benchmarks](benchmarks.md)) answers **"how does cost
scale?"**. It cannot answer **"which configuration do we ship?"**, because its
registration offset is injected rather than biological — a known shift on a
duplicated channel is not tissue difficulty.

This page is the other half: **the real study slides, registered once per
configuration**, plus a per-process cost profile on the same slides. It produces
the arm ranking the manuscript's registration figure needs, and it feeds
`ihc_method` directly.

| | synthetic sweep | **this** |
|---|---|---|
| input | one image rescaled across a size × channel matrix | your real `input.csv` |
| question | how does cost scale with input? | which configuration, and what does a real slide cost? |
| accuracy | a known injected offset | tissue, scored by `reg_qc=2` |
| consumer page | `benchmark_pipeline`, `benchmark_registration` | `registration_arms` |

Neither replaces the other. The scaling regression needs a controlled size axis
that real slides do not provide; the arm ranking needs tissue that synthetic
images cannot imitate.

---

## The arms

Defined in `benchmarks/configs/arms.yaml`. They are **factored, not crossed** —
registration is the expensive half, so it is paid for once.

At the shipped settings that is **23 launches**: 1 shared preprocessing, 14
registration (12 arms + 2 QC-segmenter cross), 4 external (ASHLAR), 3
segmentation, 1 compute profile. **All but the compute profile launch the whole
cohort**, so the launch count is not the run count — for a 6-patient cohort,
22 × 6 = 132 patient-runs plus the compute launch. `build_arm_plan.py` prints
this multiplier; read the arm counts below as per-patient multipliers.

### 1. Registration arms — *which configuration aligns real tissue best?*

`--start registration --stop registration`, at `reg_qc = 2`, resuming from **one
shared preprocessing run**. No arm axis touches a `preproc_*` param, so running
preprocessing nine times would repeat the expensive half of a real-WSI run to
vary something it does not affect — the same factoring the segmentation arms use.
Segmentation and export are not run either: nothing downstream of registration
changes the staged registration QC. **12 arms**:

- **VALIS preset × micro-depth = 9.** `memory_mode` is a **resolution ladder** —
  `high` / `medium` / `low` detect features and solve the non-rigid field at 2048 /
  1024 / 512 px, with the same SuperPoint+SuperGlue matcher at every rung
  (`bin/utils/valis_config.py`; an earlier `low` used BRISK/RANSAC, which is why older
  notes call the tiers "different matchers") — crossed with `reg_micro_reg` (a
  **depth**: 0 none, 1 micro-rigid, 2 + micro non-rigid). A depth is why this is
  3 × 3, not 3 × 2. STARE's ladder is the same three rungs (`lib/RegPresets.groovy`),
  so `low` against `low` is a like-for-like comparison — the same axis the synthetic
  sweep's `registration_method_grid` crosses.
- **STARE (`registration_method = tiled`) × tier = 3.** A different *backend*, not
  three more cells of the grid: `memory_mode` and `reg_micro_reg` do not exist
  there, so these arms carry neither. It fans out over `reg_tiled_mode`
  (`low|medium|high`) because a single defaults arm left the ranking tuning VALIS
  across six configurations and STARE across none — the tuned-vs-untuned bias the
  synthetic sweep has an explicit guard against, and which was unguarded here.
  The **tier** is the right granularity rather than the individual knobs: it is what
  an operator picks, and each row of `RegPresets.STARE` moves all five tier-owned
  values coherently. The synthetic sweep crosses the same tiers with the refinement
  gate (`reg_tiled_gate_tre`) where a cell is cheap; here a cell is a real WSI.
### 1b. ASHLAR — the external baseline, **4 runs**

ASHLAR is not a *registration* arm: `v1.0.0` removed it as a backend
(`registration_method` is now `valis | tiled`), so a registration arm would be
rejected at launch. It is an `arm_kind=external` row instead — planned by
`build_arm_plan.py::_external_arms`, run by `benchmarks/run_ashlar_arm.sh`, and
configured under `external_baseline:` in `arms.yaml`. Comparing against an
external tool should not require the pipeline to adopt it as a backend.

`tile_size` [1024, 4096] × `maximum_shift_um` [30, 60] = **4 runs**, each over
the whole cohort. It runs in its own pass, after the registration arms, because
it **reuses `from_arm`'s published QC nuclei**
(`<root>/<from_arm>/<patient>/qc/registration/geojson/`) rather than
re-segmenting — the same factoring the segmentation arms use, and for the same
reason: scoring ASHLAR against different nuclei than the arms it is ranked
against would confound the comparison with segmenter noise.

`maximum_shift_um` is two values because it **fails silently**: ASHLAR's
`LayerAligner` substitutes a model prediction for an out-of-range tile rather
than erroring, so a shift-starved run does not crash — it quietly degrades and
lands in the table as a genuine loss for ASHLAR. If 30 and 60 agree the budget
was not binding; if they disagree, the 30 row must not be reported as ASHLAR's
accuracy.

ASHLAR is comparable at all only because `benchmarks/ashlar/solve.py` rewrites
its per-tile placements into the same `M0` + mesh manifest STARE emits — so
`bin/warp_seg_qc.py --method tiled`, the pipeline's **own** `reg_qc=2` scorer,
reads it unchanged and writes the same `*_seg_qc.json` into the same tree. Same
metric family, same columns, one layout, and `ihc_method` picks it up with no
path added. Scored any other way it would land in a different metric family that
shares no column with this table — which is what the deleted synthetic
ground-truth rung did, and why it could never be ranked against these arms.
Its `tile_size` stays a **fairness** knob, not a cost one — ASHLAR takes one
independent shift per tile, so a finer grid buys it more local freedom, the
direct analogue of STARE's `reg_tiled_tile`, which the synthetic sweep varies
over `[1024, 2048, 4096]`; `[1024, 4096]` brackets that range at both ends.
**Read it against VALIS's `rigid` stage** for the like-for-like number and
against `micro` to quantify what non-rigid buys: ASHLAR attempts no non-rigid
warp at all, so reporting only the second overstates VALIS's advantage.

### 2. QC-segmenter cross — *does the verdict depend on who found the nuclei?*

`subworkflows/local/seg_qc.nf` segments the native slides with **the run's own
segmenter**, so `params.seg_method` selects which nuclei registration accuracy is
measured on. Varying it leaves the registration byte-identical, which makes this a
**robustness** axis on the headline number — never a quality claim about
registration. The same category `seg_qc_pairing` occupies in the synthetic sweep.

`cross: reference` (the default) runs the extra segmenters against one arm:
**14 registration runs** (12 arms + 2 extra segmenters on the reference arm).
`cross: all` crosses all twelve and costs **36**. `arms.yaml`'s cost gate and
`test_cross_all_crosses_every_arm` carry the same two numbers. Start at
`reference` — if the number is stable there, crossing everything buys a denser
null result.

### 3. Segmentation arms — *which backend segments real tissue best?*

`--start segmentation`, resuming from `<root>/<from_arm>/csv/registered.csv`. So
registration happens once, and the comparison is not confounded by arms that
registered differently. **3 runs** (instantseg / stardist / cellsam).

Scored two ways, neither needing ground truth:

- **Cross-method agreement** — `segmentation_agreement.csv` from `make_tables.py`
  (cell-count ratio, instance F1). Already built; answers *do the backends agree,
  and where not?*
- **CSE** — the reference-free `QualityScore`, which can **rank** rather than only
  compare. Opt-in; see [Parameters](parameters.md#cse-opt-in).

### 4. Compute profile — *what does a real slide actually cost?*

The **full** pipeline, no step gate, under tracing — the only arm that prices
`SEGMENT`, quantification, export and `MERGE_AND_PYRAMID`. Feeds the same
`make_figures` path as the synthetic sweep, so `measurements.csv` gains
real-tissue rows tagged `varied_axis=real_compute`.

Prefer **two patients bracketing the cohort's size range** over one: a single
patient gives a per-process breakdown but no slope, so it cannot say whether the
synthetic scaling fits transfer to real tissue. Two points can.

---

## Running it — the short version

```bash
make arm-plan   INPUT=real_input.csv ROOT=arm_results          # seconds, local
make arm-run    INPUT=real_input.csv ROOT=arm_results          # hours-days, cluster
make arm-tables ROOT=arm_results                               # minutes, local
make arm-pull   ROOT=arm_results IHC=../ihc_method             # seconds
```

Then knit the four pages in `ihc_method` (step 5 below). The Make targets write
the plan to `<ROOT>_plan.csv`; the long form below spells out the same four steps
with explicit paths.

`arm-run` is deliberately **not** a prerequisite of `arm-tables` — it is
hours-to-days of cluster time, and the tables are meant to be regenerated
repeatedly while it is still going.

## Running it — the long version

### 0. Your samplesheet

The ordinary mirage samplesheet — nothing benchmark-specific:

```csv
patient_id,path_to_file,is_reference,channels
046,/hpcnfs/.../046_cycle1.ome.tif,true,DAPI|PANCK|CD8
046,/hpcnfs/.../046_cycle2.ome.tif,false,DAPI|CD68|FOXP3
24086,/hpcnfs/.../24086_cycle1.ome.tif,true,DAPI|PANCK|CD8
24086,/hpcnfs/.../24086_cycle2.ome.tif,false,DAPI|CD68|FOXP3
```

### 1. Expand the arms into a run plan

```bash
python benchmarks/build_arm_plan.py \
    --arms         benchmarks/configs/arms.yaml \
    --input        real_input.csv \
    --out          arm_plan.csv \
    --results-root arm_results
```

Writes `arm_plan.csv` (one row per launch) and `arm_results/arms.csv` (the label
manifest the consumer reads). It prints the multiplier out loud — every
registration and segmentation arm runs the **whole cohort**, so 12 launches over 2
patients is 24 patient-runs.

`--results-root` matters: `arms.csv` must sit at the root the runs publish into,
because that is where `registration_arms.R` looks.

### 2. Launch (cluster)

On SLURM, use the submitter — it sets the profile, the JVM heap per head, the
Singularity cache and the CellSAM token check for you:

```bash
cd /beegfs/scratch/$USER/analysis_runs/method_paper/benchmark
mkdir -p logs && sbatch mirage/benchmarks/submit_arms.sh
```

Edit the `EDIT THESE FOR YOUR SITE` block at the top first. Keep everything on the
large filesystem: `$HOME` is small, and read-only inside the containers.

Or drive it directly:

```bash
ARMS_PROFILE="singularity,ieo" ARMS_CONCURRENCY=4 \
  benchmarks/run_arms.sh arm_plan.csv real_input.csv arm_results -c conf/ieo.config
```

Passes run in order — `preprocess`, `registration`, `segmentation`, `compute` —
with a barrier between them. That order is a **dependency**: each pass resumes
from a checkpoint the previous one wrote. The compute arm runs last and alone so
it is not timed under contention from the QC arms.

!!! warning "`ARMS_CONCURRENCY` is heads, and heads share the head job's memory"
    Each concurrent arm is one Nextflow JVM. The `-Xmx32g` that suits a
    single-run launcher would blow a 32 GB head job at two arms; `submit_arms.sh`
    sets `-Xmx3g` per head instead. Raise `--mem` before raising concurrency.

`ARMS_CONCURRENCY` is how many Nextflow heads run at once; each still submits its
own SLURM jobs, so measurements stay clean.

To enable CSE on the segmentation arms, publish the `segeval` image once
(Actions → *Build & Push Container Images* → Run workflow) and append:

```bash
  ... arm_results -params-file params/seg_quality_eval.json
```

### 3. Emit the paper tables

Unchanged from the synthetic sweep — the same readers, pointed at the arm results:

```bash
python -m benchmarks.analysis.make_tables \
    --results-root arm_results --run-plan arm_plan.csv --outdir benchmarks/paper_data

python -m benchmarks.analysis.make_figures \
    --results-root arm_results --run-plan arm_plan.csv \
    --reg-eval none --outdir benchmarks/analysis
```

`--reg-eval` is required. `none` is the normal path — this repository ships no
ground-truth registration-accuracy harness (see `benchmarks/README.md` section
B) — and it writes a `NO_GROUND_TRUTH.txt` marker into the output so a cost-only
result says so in the deliverable. If you have an externally produced landmark
TRE CSV (one row per pair_id/mode; see `load.GROUND_TRUTH_COLS`), pass its path
instead of `none`.

Both read whatever has finished; re-run them as arms land.

### 3b. What ONE run cost — the resource profile

Not an arm, and not the sweep: any finished mirage run — the compute arm, a plain
production run — has Nextflow's `trace.txt` and the pipeline's `size_logs/input_sizes.csv`,
and `benchmarks/analysis/run_resources.py` turns those two into per-task, per-process and
per-run tables (CPU-hours used against reserved, peak RSS against the memory requested,
wall-time, and both against each task's input size), plus within-run fits of peak RSS and
wall-time on input size for every process that ran on ≥ 3 inputs of different size:

```bash
make run-resources RUN=/path/to/run_outdir [TRACE=/path/to/.trace/trace.txt] IHC=../ihc_method
# = benchmarks/pull_run_resources.sh <run_outdir> ../ihc_method [--trace ...]
```

`TRACE` matters more often than it looks: `trace_dir` defaults to `.trace` beside the
**launch** directory, not inside `--outdir`, so a bare run's trace is found only when it
sits at `<outdir>/../.trace`; the benchmark launchers point it into the results tree
(`<run>/trace/`), which is found directly. The tables land in
`ihc_method/data/run_resources/` with a `.dict.md` defining every column, and
`analysis/run_resources.Rmd` there plots them. One run at a time, by design — the script
replaces the tables, and a fit inside one run must not mix configurations.
`pull_to_ihc_method.sh` runs the same step (4b) for the run it publishes as `data/mirage/`.

### 4. Hand off to `ihc_method`

```bash
# arms only
benchmarks/pull_to_ihc_method.sh arm_results ../ihc_method

# arms + the resource sweep, regenerating both sets of tables first
benchmarks/pull_to_ihc_method.sh arm_results ../ihc_method \
    --sweep sweep_results --build

# or, via make
make arm-pull ROOT=arm_results IHC=../ihc_method SWEEP=sweep_results
```

Copies **only the small artifacts** — QC JSON/CSV, VALIS summaries, traces, the
cell tables, the paper tables — into `ihc_method/data/`. The registered
OME-TIFFs and masks stay where they are: no analysis page reads them, and they
are multi-GB per patient.

It lands them where each page expects:

| destination | read by |
|---|---|
| `data/registration_arms/<arm>/<patient>/…` | `registration_arms.Rmd` |
| `data/registration_arms/arms.csv` | ditto — the arm labels |
| `data/registration_arms/*.csv` | the arm tables, if `--build` made them |
| `data/benchmark/*.csv` | `benchmark_pipeline.Rmd`, `benchmark_registration.Rmd` |
| `data/mirage/<patient>/qc,csv` | `registration_run_qc.Rmd` |
| `data/mirage/<patient>/quantification,cell_properties` | the mirage cell pages |

#### The arms and the sweep must not share an outdir

`make_tables.py` and `make_figures.py` both default to one outdir pair. Run them
for the arms and then for the sweep and the second silently overwrites the
first — `measurements.csv` becomes whichever ran last. Nothing errors: the file
exists and carries the right columns, so the scaling pages render normally with
data that answers a different question.

`benchmark_plots.R` keys on the **sweep's** axes (`scaling_grid`, `target_px`,
`n_channels`), so `data/benchmark/` must hold the sweep's tables. The script
therefore stages each experiment separately under `benchmarks/_handoff/`
(`--handoff` to relocate, e.g. a read-only cluster checkout) and copies the arm
tables **beside the arms** instead. Order of operations can no longer decide
what a page reads.

| option | meaning |
|---|---|
| `--sweep <dir>` | the sweep results root (`run_sweep.sh`'s third argument) |
| `--sweep-plan <csv>` | its run plan; defaults to `<sweep>_plan.csv` |
| `--arm-plan <csv>` | the arm plan; defaults to `<arm_root>_plan.csv` |
| `--run <dir>` | which run becomes `data/mirage/`; defaults to the `compute_*` arm |
| `--handoff <dir>` | where built tables are staged |
| `--build` | regenerate the tables before copying (minutes on a full sweep) |

### 5. Knit the pages — this is where the plots appear

```r
# in ../ihc_method
renv::restore()                       # first time only
workflowr::wflow_build(c("analysis/registration_arms.Rmd",
                         "analysis/benchmark_pipeline.Rmd",
                         "analysis/benchmark_registration.Rmd",
                         "analysis/registration_run_qc.Rmd",
                         "analysis/run_resources.Rmd"))
```

Each page renders its figures **inline from the CSVs** — there are no PNGs on
disk to wire up — and writes PDFs to `output/figures/<page>/` via
`export_pdf_figures()`. Open `docs/<page>.html` to read them.

A page whose input is missing does not fail: it prints what it wanted and skips
the figure. So a partial pull gives a partial page, never a broken build.

`data/` is gitignored there — nothing copied is committed.

| page | answers |
|---|---|
| `registration_arms` | which registration configuration to ship, ranked on real tissue |
| `benchmark_registration` | cost-vs-accuracy across the sweep; the two independent accuracy signals agreeing |
| `benchmark_pipeline` | resource scaling, cost, segmentation-method comparison |
| `registration_run_qc` | was this cohort registered well enough to analyse? |
| `run_resources` | what one run of the cohort cost, per process, against input size |

---

## Three traps the consumer already guards, and why the producer respects them

These are properties of the *measurement*, not of the plumbing. They are restated
here because a producer that ignores them emits data that plots cleanly with the
conclusion inverted.

1. **`rigid` is not comparable across micro-depths.** mirage defines the QC
   `rigid` stage as the rigid transform *after* `MicroRigidRegistrar` refined it.
   At depth 0 that is affine alone; at depth ≥ 1 it is affine ∘ micro-rigid.
   Putting `rigid` on one axis across a depth-crossed sweep plots two different
   transforms and reads as a micro-registration effect that is really a definition
   change. **A true rigid-only baseline exists only in the depth-0 arms.**

2. **The backends do not share a stage vocabulary.** `lib/WarpBackends.groovy`:
   VALIS is `native → rigid → non_rigid → micro`; STARE and ASHLAR are both
   `native → rigid → refined` (they serialize the same `M0` + mesh manifest, so
   they read through the same warper). Only `native` and `rigid` are shared as
   both a spelling and a meaning across all three. Each arm is therefore ranked on
   **its own final stage** — which is also why depths 0 and 1, emitting no `micro`
   stage at all, are not silently dropped.

   `benchmarks/analysis/lib/quality.py`'s `_STAGE_RANK` is what performs that
   reduction, and it is guarded by
   `benchmarks/tests/test_stage_rank_covers_every_backend.py`, which reads the
   vocabularies out of `WarpBackends` rather than restating them. It was added
   after `refined` was found MISSING from that table: an unranked stage maps to
   `-1`, ties with `native`, and the "final stage" pick then rests on a stable-sort
   accident — so every STARE run was one upstream reordering away from reporting
   its *unregistered* accuracy as its headline number.

3. **Label the arms explicitly.** `arms.csv` is always written, because the
   consumer's fallback parses directory names for `high`/`low` and a depth. A
   mislabelled arm does not fail — it renders a clean figure with the conclusion
   inverted. The QC-segmenter-crossed names (`valis_high_micro2_segstardist`) are
   exactly what that fallback reads wrong.

---

## Cost, before you launch

Per patient, at the shipped `arms.yaml`:

| arm | runs | pipeline extent |
|---|---|---|
| shared preprocessing | 1 | preprocessing only |
| registration (9 + 2 crossed) | 11 | registration only (resumed) |
| segmentation | 3 | segmentation → export (resumed) |
| compute profile | 1 | full pipeline |

Preprocessing is paid for **once**, not ten times. The compute-profile arm still
runs it, because it is the arm that prices every process.

Real WSI runs are not sweep cells: `REGISTER` has been observed at **483 GB** and
`MERGE_AND_PYRAMID` at **6.5 h**. Multiply by your cohort before submitting, and
prefer `qc_segmenter_cross.cross: reference` until the headline number proves
unstable.
