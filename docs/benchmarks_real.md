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

At the shipped settings that is **90 launches**: 1 shared preprocessing, 18
registration arms (9 VALIS + 9 STARE), 54 QC instrument crosses (which **resume**
their base arm and re-run only the QC chain — see §2), 9 solver crosses (the STARE
arms with the `robust` SOLVE stage, `arm_kind=registration_solver`; they resume
their base too but re-run the tiled stages — see §1c), 4 external (ASHLAR), 3
segmentation, 1 compute profile. **All but the compute profile launch the whole
cohort**, so the launch count is not the run count — for a 6-patient cohort,
80 × 6 = 480 patient-runs plus the compute launch, of which 54 × 6 are QC-only.
`build_arm_plan.py` prints this multiplier; read the arm counts below as
per-patient multipliers.

### 1. Registration arms — *which configuration aligns real tissue best?*

`--start registration --stop registration`, at `reg_qc = 2`, resuming from **one
shared preprocessing run**. No arm axis touches a `preproc_*` param, so running
preprocessing nine times would repeat the expensive half of a real-WSI run to
vary something it does not affect — the same factoring the segmentation arms use.
Segmentation and export are not run either: nothing downstream of registration
changes the staged registration QC. **18 arms**, nine per backend:

- **VALIS preset × micro-depth = 9.** `memory_mode` is a **resolution ladder** —
  `high` / `medium` / `low` detect features and solve the non-rigid field at 2048 /
  1024 / 512 px, with the same SuperPoint+SuperGlue matcher at every rung
  (`bin/utils/valis_config.py`; an earlier `low` used BRISK/RANSAC, which is why older
  notes call the tiers "different matchers") — crossed with `reg_micro_reg` (a
  **depth**: 0 none, 1 micro-rigid, 2 + micro non-rigid). A depth is why this is
  3 × 3, not 3 × 2. STARE's ladder is the same three rungs (`lib/RegPresets.groovy`),
  so `low` against `low` is a like-for-like comparison — the same axis the synthetic
  sweep's `registration_method_grid` crosses.
- **STARE (`registration_method = tiled`) tier × refinement gate = 9.** A different
  *backend*: `memory_mode` and `reg_micro_reg` do not exist there, so these arms
  carry neither. `reg_tiled_mode` (`low|medium|high`) is the same 512 / 1024 / 2048 px
  ladder as VALIS's tiers, and `reg_tiled_gate_tre` {0.5, 1.0, 2.0} is STARE's
  refinement depth — the rigid-stage TRE above which a tile is non-rigidly refined —
  the counterpart of `reg_micro_reg`. Nine against nine (since 2026-09-10; it was
  three tier-only arms against nine, which handed VALIS three times the draws in a
  best-cell ranking). The **tier** rather than its five knobs because the tier is
  what an operator picks, each `RegPresets.STARE` row moves all five coherently, and
  `validateRegPresets` refuses a per-knob override under any tier but `custom`. The
  synthetic sweep crosses exactly the same two axes on synthetic images.
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

**It has a cost row too.** Because it is not a Nextflow run, nothing wrote a
trace for it, and ASHLAR sat in the accuracy table with no entry in
`measurements.csv` / `run_cost` / `resource_stats`. `run_ashlar_arm.sh` now runs
each heavy step through `benchmarks/trace_step.py`, which records wall-clock,
peak RSS, CPU time and exit status as Nextflow-format rows in the same
`<root>/<arm>/trace/trace.txt` every other arm has (processes `ASHLAR_RETILE`,
`ASHLAR_SOLVE`, `ASHLAR_SEG_QC`, tag = patient) plus the step's input size in
`size_logs/input_sizes.csv`, so the analysis reads it with no ASHLAR-specific
path. A root built before this change lacks those rows; collect them once with
`make arm-rerun ONLY='ashlar.*'` — nothing depends on an external arm, so that
re-runs the four ASHLAR arms and nothing else (see "Re-running a subset").

### 1c. The SOLVE-stage cross — **9 runs**

`reg_tiled_solver` selects STARE's SOLVE stage: `legacy` (gates + median filter,
byte-identical to every manifest produced before 2026-09-12) or `robust`
(neighbour-consistency rejection, in-fill of dropped tiles, Tikhonov smoothing,
invertibility check — `stare.solve`, `docs/parallel_registration_design.md` §6b).
The pipeline default is `robust`; the 9 STARE base arms are pinned to `legacy` in
`arms.yaml`'s baseline because they were launched before the solver existed and
*are* that path, so their results stay valid. `robust` enters as a **solver cross**
(`solver_cross`, `arm_kind=registration_solver`): one row per STARE base arm, named
`<base>_solver_robust`, resuming the base arm's launch directory. Unlike the QC
crosses of §2 it changes the registration, not how it is measured, so it is a third
kind rather than a `registration_qc` row, and the VALIS-vs-STARE draw count stays 9
against 9.

Cost: the tile modules reference `params` in their script blocks, so under
`-resume` the tiled stages re-run — a solver cross is one full STARE registration,
not a QC-only resume. Nine launches, never a cohort. After any change to
`stare.solve`, `--changed solve` selects exactly these nine (see "Re-running a subset after a code change").

### 2. QC instrument crosses — *does the verdict depend on how it was measured?*

The arm ranking reads the staged seg-overlap QC (`reg_qc = 2`): `subworkflows/local/seg_qc.nf`
segments the native slides with **the run's own segmenter** (`params.seg_method`
selects it), the nuclei are paired across slides by **`params.seg_qc_pairing`**
(`lsa`, linear sum assignment — the default — or `mutual_nn`, mutual nearest
neighbours; `bin/utils/cell_pairs.py`), and the pairs are held fixed through every
stage. Two measuring instruments, then: *who found the nuclei* and *how they were
paired*. Varying either leaves the registration byte-identical, which makes both
**robustness** axes on the headline number — never a quality claim about
registration. If the arm ranking changes with the segmenter or the pairing rule,
the verdict is fragile and the paper should say so; if it does not, the null result
is the evidence.

**They resume, they do not re-register.** A cross arm is planned as
`arm_kind = registration_qc` with `resume_run` naming its base arm; `run_arms.sh`
launches it *inside the base arm's launch directory* with `-resume <that session>`,
so `REGISTER` — up to 483 GB on a real WSI — is served from the cache and only the
QC chain runs again (`SEG_QC_SEGMENT` for a segmenter cross, `WARP_SEG_QC` for a
pairing cross, then the aggregation), publishing into the cross arm's own directory.
Cross arms of **one base run one after another** (two runs resuming the same session
at once fight over Nextflow's cache-DB lock); different bases run concurrently.

**One instrument at a time, not a factorial.** `qc_segmenter_cross` varies the
segmenter at the baseline pairing; `qc_pairing_cross` varies the pairing at the
baseline segmenter. Per base arm that is (3 − 1) + (2 − 1) = **3 QC-only runs**; both
blocks ship at `cross: all`, so 18 base arms give **54** cross runs and **72**
registration-step launches per cohort. `cross: reference` restricts either block to
`reference_arm` (18 + 2 + 1 = 21). `test_cross_all_crosses_every_arm`,
`test_qc_cross_arms_resume_their_base_arm` and
`test_run_arms_chains_the_qc_crosses_of_one_base_and_resumes_its_session` carry these
numbers and the chaining.

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

# after a code change confined to one component -- see "Re-running a subset"
make arm-rerun  CHANGED=tiled INPUT=real_input.csv ROOT=arm_results
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

## Re-running a subset after a code change

A change confined to one component does not move every arm, and a real WSI arm
is days of cluster time. The harness can re-run **only what the change
affects** and have `make arm-tables` produce exactly what a full re-run would.
The rule is in code, not here: `benchmarks/impact.py`, guarded by
`benchmarks/tests/test_subset_rerun_equivalence.py`, which builds a synthetic
results root in the real arm layout, replaces only the affected arms' files, and
asserts every table is byte-identical to a full re-run's.

### Which components map to which arms

`build_arm_plan.py --changed <component>` (repeatable) seeds the selection with
every row its predicate matches, then takes the **transitive closure** over the
plan's dependency columns: a row that `resume_run`s an affected arm is affected
(a QC cross re-scores the re-run base — left out, it would keep the OLD score
from the OLD session beside the base's new one), a row whose `from_arm` is
affected is affected (it resumes that arm's checkpoint), and an external row
whose `ext_from_arm` is affected is affected (it scores on that arm's nuclei).
Nothing else is.

| `--changed` | seeds | closure adds | at the shipped `arms.yaml` |
|---|---|---|---|
| `solve` | the STARE rows that run the `robust` SOLVE stage — the 9 solver crosses. The 9 STARE base arms are the `legacy` solver, pinned byte-identical to the code that produced them, so a change to `stare.solve` does not reach them | nothing resumes a solver cross | **9 of 90** — this is the re-run after a SOLVE change |
| `tiled` / `stare` | every row at `registration_method=tiled` — the 9 bases, their 27 QC crosses and their 9 solver crosses, which all carry the backend column | (the crosses would be added by closure if they did not) | 9 + 27 + 9 = **45 of 90**; no VALIS arm, no preprocessing, no segmentation arm, no ASHLAR arm (all scored on the VALIS reference) |
| `valis` | every VALIS arm | their crosses, the segmentation arms (`from_arm`), the ASHLAR arms (`ext_from_arm`), the compute profile (baseline backend) | 44 of 90 |
| `seg:<method>` | every row whose `seg_method` is that backend — the segmentation arm *and* every `_seg<method>` cross, since `SEG_QC_SEGMENT` is `SEGMENT` under an alias | — | `seg:stardist`: 19 of 90 |
| `ashlar` | the external arms | nothing depends on them | 4 of 90 |
| `qc` | every row that runs the `reg_qc` scorer, ASHLAR included | — | 89 of 90 (all but `preprocess_shared`) |
| `preprocess` | `preprocess_shared` and the compute profile | everything resumes from it | 90 of 90 |
| `--only <regex>` | rows whose `arm`/`run_id` matches (`re.search`) | the same closure | `--only 'ashlar.*'` → the 4 external arms |

The subset plan's rows are **byte-identical lines of the full plan** — same
`run_id`, same `arm`, same params, same `resume_run`, under the full plan's
header — so a re-run lands in the same `<root>/<arm>` directories the full run
wrote. `test_subset_plan_is_a_row_identical_subset_under_the_full_header`
asserts it.

### The three commands

```bash
make arm-plan-subset CHANGED=tiled INPUT=real_input.csv ROOT=arm_results   # -> arm_results_plan.subset.csv
make arm-rerun       CHANGED=tiled INPUT=real_input.csv ROOT=arm_results   # = ARMS_REPLACE=1 run_arms.sh <subset plan> ...
make arm-tables      ROOT=arm_results                                      # reads arm_results_plan.csv -- the FULL plan
```

or, driven directly / on SLURM:

```bash
python benchmarks/build_arm_plan.py --arms benchmarks/configs/arms.yaml \
    --input real_input.csv --out arm_plan.subset.csv --results-root arm_results --changed tiled
ARMS_REPLACE=1 ARMS_PROFILE="singularity,ieo" \
    benchmarks/run_arms.sh arm_plan.subset.csv real_input.csv arm_results -c conf/ieo.config
# or: CHANGED=tiled ARMS_REPLACE=1 sbatch benchmarks/submit_arms.sh   (writes arm_plan.subset.csv beside the full plan)
```

`ARMS_REPLACE=1` is opt-in. For each row of the plan it **moves** the previous
`<root>/<arm>` output directory and, for a base arm, its `<root>/.launch/<run_id>`
launch directory (work dir, cache, history) to `<root>/.replaced/<timestamp>/…`
before launching — never deletes: the old result is what you compare the new
one against, and `.replaced/` is yours to remove once you have. A cross arm has
no launch directory of its own; when its base is *not* in the plan, only its
run-name line is freed from the base's history (a copy is kept under
`.replaced/`). Without `ARMS_REPLACE`, `run_arms.sh` keeps refusing a run name
its launch directory already holds, naming the remedy.

**The refusal.** `run_arms.sh` refuses to replace a base arm whose QC crosses
the plan does not carry, naming them — the launch directory's history is the
ground truth of which runs resumed that base. A plan built with `--changed` /
`--only` cannot hit this (the closure adds them); a hand-filtered CSV can.

### Why `arms.csv` is untouched

`arms.csv` is the consumer's label manifest, and `registration_arms.R` labels
every arm it finds under `<root>` from it. A subset re-run leaves the unaffected
arms' results in place, so the manifest must still name them: `build_arm_plan.py`
writes `arms.csv` **from the full plan** on every build, subset or not, and a
subset build's `arms.csv` is byte-identical to a full build's
(`test_subset_plan_is_a_row_identical_subset_under_the_full_header`). The tables
likewise read the **full** plan — `make arm-tables` reads `<ROOT>_plan.csv`, and
the subset is written to `<ROOT>_plan.subset.csv` precisely so it never
overwrites it — and `load_runs` / the QC harvesters take the union of whatever
each arm directory holds.

### The sweep: accuracy columns, not resource curves

`build_run_plan.py --only-method tiled` (or `--changed` / `--only <regex>` on
`run_id`, `config_id`, `varied_axis`) writes the same row-identical subset of the
synthetic sweep (9 of 98 runs at the shipped `sweep.yaml`), and `run_sweep.sh`
takes `SWEEP_REPLACE=1` to move a run directory aside to
`<root>/.replaced/<timestamp>/<run_id>` before relaunching (it otherwise skips a
run whose name its directory already holds). But be clear about **what a
SOLVE-only change can move there**: the resource/scaling curves are dominated by
`TILED_COARSE` / `TILED_REG_TILE` / `TILED_STITCH` and `REGISTER`'s peak RSS, and
a different solver moves `TILED_SOLVE`'s peak by kilobytes. The sweep's tiled rows need re-running for the
**accuracy columns** (`registration_accuracy.csv`, `param_matrix.csv`'s
`reg_*`), not for the resource curves, which will come back within replicate
noise of the old ones. Do not read a moved scaling fit as a solver effect.

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
| registration (9 VALIS + 9 STARE) | 18 | registration only (resumed from preprocessing) |
| QC instrument crosses (2 segmenters + 1 pairing per arm) | 54 | QC chain only (resumes the base arm's session) |
| ASHLAR external baseline | 4 | ASHLAR + the pipeline's QC scorer |
| segmentation | 3 | segmentation → export (resumed) |
| compute profile | 1 | full pipeline |

Preprocessing is paid for **once**, not eighteen times, and registration is paid for eighteen times, not seventy-two. The compute-profile arm still
runs it, because it is the arm that prices every process.

Real WSI runs are not sweep cells: `REGISTER` has been observed at **483 GB** and
`MERGE_AND_PYRAMID` at **6.5 h**. Multiply by your cohort before submitting, and
prefer `qc_segmenter_cross.cross: reference` until the headline number proves
unstable.
