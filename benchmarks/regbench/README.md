# regbench: registration methods, each run natively, judged on the same nuclei

`benchmarks/regbench/` compares registration methods on nuclear overlap and on
registration-only cost. It was moved here from `sceriff0/stare` (branch `benchmarking`,
commit `f5ce62e`, archived there as the tag `archive/benchmarking-2026-10-08`).

## The whole benchmark in one command

From the folder the benchmark should live in (its `regbench/` holds the inputs, the cases
and the results), with the site's container module loaded and `REGBENCH_INPUT` naming the
cohort's samplesheet:

```bash
REGBENCH_INPUT=/path/to/input.csv \
  sbatch [--partition=... --account=... --qos=... --time=...] \
  <repository>/benchmarks/regbench/slurm/launch.sbatch --all [--runs "a b"] [-- extra sbatch args]
```

`--all --dry-run` submits nothing and prints the stage plan as read from disk, the next
head's full `sbatch` line, and the launcher's own dry run of the stage it would run now (and
of the launch as well, once the inputs and the cohort's cases are there). When the stage
it would run is prepare and `REGBENCH_MANIFEST` is not set, the dry run still rebuilds
`$REGBENCH_INPUTS/rounds.csv` from `preprocessed.csv`, as `--prepare --dry-run` always has.
That file is the deterministic product of `preprocessed.csv`, and a real prepare writes it
again.

**What runs, in order.** `--all` is a chain of short head jobs, all named `regbench-launch`.
Each head runs one stage, submits the next head (`launch.sbatch --all-stage=<next>`) and
ends. No head sleeps waiting for another, nothing waits on the login node, and nothing uses
`sbatch --wait`.

| stage | done when (read by the first head) | the head runs | the next head waits for |
|---|---|---|---|
| inputs | `$REGBENCH_INPUTS/preprocess/results/csv/preprocessed.csv` exists | `--inputs`: one Nextflow head, the pipeline's preprocessing | `afterok` on that job (`rb-inputs`) |
| prepare | the rounds file exists, the cohort has prepared cases, and its last prepare ended ok (`$REGBENCH_CASES/prepared-multiplex.ok`, which `prepare.sbatch` removes when it starts and writes only on success) | `--prepare`: the rounds, the nuclei array (StarDist on its memory ladder), `rb-prepare-multiplex` | `afterok` on `rb-prepare-*` only, never on the nuclei tasks: one slide that fails is tolerated (`afterany` upstream) |
| launch | - | the launch, with `--datasets "multiplex <every public dataset whose cases are prepared>"` and `--compute-all` when `REGBENCH_INPUT` is set | nothing: the chain ends, and the launcher's own jobs run |

The first head runs the first stage that is not done (a done prepare stage counts the
inputs as done too). Cases on disk without that record (a prepare that was cancelled or ran
out of time half-way) do not count as done: prepare runs again and reuses the cases that are
there, so the launch never runs on part of the cohort. Every later head is given its stage, so the chain cannot loop or skip.
The datasets left out of the launch are listed with the reason; without `REGBENCH_INPUT`, a
line says compute-all is skipped and why. `--runs`, `--datasets` and the launcher's other
arguments go to the launch, and the arguments after `--` go to every stage's jobs. Each next
head is submitted with `--parsable --export=ALL --kill-on-invalid-dep=yes
--dependency=afterok:<ids> --job-name=regbench-launch`, the first head's submit folder as
`--chdir`, an absolute `--output` (`$REGBENCH_ROOT/logs/regbench-launch-<id>.log`), and the
first head's own `--partition`, `--account`, `--qos` and `--time`, which it reads from
`scontrol show job`. `--kill-on-invalid-dep=yes` makes a failed stage cancel the next head
rather than leave it pending for ever as `DependencyNeverSatisfied`. Heads are never
requeued: `launch.sbatch` has `#SBATCH --no-requeue`, and every next head is given
`--no-requeue` too. Leaving `--requeue` out would not be enough, because Slurm's default
`JobRequeue=1` requeues a batch job after a node failure, and a requeued head would queue its
stage a second time.

**Which sbatch options reach the next heads.** Only `--partition`, `--account`, `--qos` and
`--time`, read from the first head, and only when it is a batch job (`BatchFlag=1`). Run
inside `salloc` or an interactive session, nothing is copied, and a line says so. Any other
option of the first `sbatch` (`--reservation`, `--constraint`, `-M/--clusters`, `--mail-*`,
...) is NOT copied to the heads. To reach every head, export it as sbatch's own input
variable before the first `sbatch` (`SBATCH_RESERVATION`, `SBATCH_CONSTRAINT`,
`SBATCH_CLUSTERS`, ...; `--export=ALL` carries them to every head), or export
`REGBENCH_CHAIN_SBATCH="--partition=... --reservation=..."`, which then replaces what is read
from scontrol. To give it to the stages' own jobs too, repeat it after `--`. A first head
submitted to several partitions (`-p a,b`) shows, once running, only the partition it runs
in, so every later head goes to that one. A `TimeLimit` of `UNLIMITED` is not copied (the
next head then takes the `#SBATCH --time` of the file, or the site's default).

Each stage has a submission id of its own: `<chain id>-inputs`, `<chain id>-prepare`, and
`<chain id>` for the launch. Its ids file and retry plans therefore never mix with
another stage's (with `REGBENCH_SUBMISSION` exported before the first `sbatch`, that id
replaces the chain id in the three names). If a stage queues nothing (for example, the launcher refused it), the chain
ends with a non-zero exit and says why. It never submits a head that would wait on nothing.

**Watching it.** `squeue --me -n regbench-launch` shows the heads. The chain's record is
`$REGBENCH_ROOT/logs/chain/<chain id>.tsv` (stage, head_job, ids_file, next_head, state, at),
with one line per head. Each head writes its line before its stage runs (state `running`,
with the ids file the stage will write) and sets it to `done` or `failed` afterwards. When
`squeue` shows no head while the last line is `done` with a `next_head`, a stage failed, and
`--kill-on-invalid-dep=yes` has cancelled that next head without a word:
`sacct -j <next_head> -o JobID,State,Reason` shows it CANCELLED
(`DependencyNeverSatisfied`), and the stage's own job log under `$REGBENCH_OUT/logs/`
(`rb-inputs-<id>.log`, `rb-prepare-multiplex-<id>.log`) says why. Fix that cause and
submit `--all` again. The heads' logs are `$REGBENCH_ROOT/logs/regbench-launch-<id>.log` (the
first one is where it was submitted from), and the stages' own jobs log under
`$REGBENCH_OUT/logs/`.

**Stopping it.** Every head prints the line
`scancel <heads> $(cat "<ids file>" ... 2>/dev/null)`, which removes everything the chain
has queued so far: the heads, and every job id in every ids file the record names (the tile
submitters add theirs to the launch's file as they feed). It is printed once before the stage
runs and once after it, so a head killed in the middle of its stage has already printed a
line that covers that stage's jobs. The last head's line covers the whole chain.

**Resuming after a stop.** Submit `--all` again. The stages that are done are skipped, and
the launch reuses every unit that is already ok in `REGBENCH_OUT` and runs the rest. A new
chain gets a new id (its submission id too). The launcher refuses a second submission while
jobs of the first are still queued, and says which ones.

**Concurrency.** Every knob is set, or shown with its default, in the SETTINGS of
`slurm/launch.sbatch`, with what it bounds and when to change it. An exported variable wins
over the file (`--export=ALL`). At the launch stage the head prints `# peak: ...` from the
actual settings and run list before its first `sbatch`, and the launcher's dry run gives the
exact queue figure. With the shipped settings, on the cohort's list (56 runs, 4 of them on a
GPU, 2 with a realcheck):

| knob | launch.sbatch | bounds | change it when |
|---|---|---|---|
| `REGBENCH_RUN_LANES` | 4 (launcher: 0, no limit) | runs in flight: each run waits `afterany` for the one before it on its lane | the cluster is empty (raise it) or the per-user queue limit is hit (lower it) |
| `REGBENCH_ARRAY_LIMIT` | 20 | tasks of one array at once (`%N`): CPU ladders, the nuclei array on CPUs, the per-case scoring | one run's tasks crowd out the others |
| `REGBENCH_GPU_ARRAY_LIMIT` | 4 (launcher: `REGBENCH_ARRAY_LIMIT`) | the same for every array of a GPU run (first rung, the watcher's retry arrays, realcheck) and the nuclei array on a GPU: one GPU per task | the site lets one user hold more GPUs |
| `REGBENCH_STAGE_QUEUE_MAX` | 100 (launcher: 200) | STARE tile tasks pending or running in the whole launch, split over the STARE runs feeding at once (25 each over 4 lanes) | the tiles fill the queue limit |
| `REGBENCH_STAGE_ARRAY_MAX` | 100 | the tasks in one tile array | the site caps an array's size |
| `REGBENCH_STAGE_FEEDERS` | 2 | tile submitters chained per STARE run and dataset (7 days each) | one run's tiles take more than 7 days to feed |
| `REGBENCH_COMPUTE_LANES` | 2 | compute-all Nextflow heads at once, last and alone | rarely: more lanes means more contention in a cost measurement |
| `REGBENCH_RETRY_WATCHERS` | 2 | links of each chain's watcher (1 CPU, 1 GB, 7 days each) | a ladder climbs for more than 14 days |
| `REGBENCH_WAVES` | 0 | 1: one method at a time (a timing replicate) | a timing replicate |
| `REGBENCH_REPEAT` | 1 | the same submission N times, into `rep<k>`, one after the other (each repeat waits `afterany` on the previous one's scorers); refused with `--all` | the CPU cross's repetitions |
| `REGBENCH_CPUS` | 4 | every ladder task's CPUs (4 or 8) | the CPU cross only |
| `REGBENCH_TIME` | 24:00:00 | a ladder task's time per round | a method needs more than a day per round |
| `REGBENCH_SCORE_CPUS`, `REGBENCH_SCORE_MEM` | 8, 64G | each per-case scoring task (the memory also of the tissue and score jobs) | the scorer runs out of memory |

The memory ladder rungs are not a setting: 16, 64, 256, 512, then 700 GB at 4 CPUs
(`ladder.py`, frozen). The selfchecks climb 64, 256, 512 GB, and the nuclei step goes from
`REGBENCH_NUCLEI_MEM` (128G) to 256 and then 512 GB. Only the first rung is submitted; a higher
rung is submitted only for the units that ran out of memory on the rung below it.

The peak, written out (shipped settings, the numbers the head printed on the fixture's run
list):

- ladder tasks at once ≤ `RUN_LANES` × `ARRAY_LIMIT` = 4 × 20 = 80, since each run in flight
  holds one first-rung array. A run with a realcheck holds one task more (its own one-unit
  ladder), so the bound is 82.
- GPU tasks ≤ (GPU runs in flight) × `GPU_ARRAY_LIMIT` = 4 × 4 = 16, one GPU each.
- plus every selfcheck: 15 jobs, outside the lanes and all at once (64 GB, 4 CPUs each, 4 of
  them on a GPU).
- plus tile tasks ≤ `STAGE_QUEUE_MAX` = 100.
- plus the small jobs: a watcher link per chain and a tile submitter per STARE run in
  flight, 1 CPU and 1 GB each.
- In total, at most 197 tasks: 588 CPUs (tasks × `REGBENCH_CPUS` for the ladder and the
  selfchecks, the pipeline's 2 per tile task), at most 20 GPUs, and 2672 GB at the first rung
  (16 GB a ladder task, 64 GB a selfcheck, 4 GB a tile task).

The site settings: `REGBENCH_SITE_CONFIG` (default `conf/ieo.config` of the checkout, which is
not in the repository; `none` reads no file) is where the GPU request, the image folder and the
bound folders are read from. `REGBENCH_INPUT` is the samplesheet, which the inputs stage
needs; when it is set, the launch adds compute-all. The folders are `REGBENCH_ROOT` (default
`regbench/` in the submit folder), `REGBENCH_INPUTS`, `REGBENCH_CASES` and `REGBENCH_OUT`
under it, and `REGBENCH_NF_PROFILE` (`singularity,ieo`) is the pipeline's profile for the
inputs and compute-all.

**What is not in it yet.** The generated known-displacement dataset of plan Task 32 has not
been built, so `--all` cannot run it. The public datasets the user has not prepared are also
left out: their data is downloaded and prepared by hand (see "State"), and `--all`
includes each one only when its cases are prepared under `REGBENCH_CASES_<NAME>`, saying
which datasets it left out and why. The landmark set (HyReCo) is launched like any other
public set once its cases are prepared ("Landmark set", below).

## The rule

**Each method is run through its own entry point, with one of its own shipped parameter
sets, its own reader and its own writer, and is judged with the transform it estimated,
moved by its own code in its own direction. Every departure from that is listed per run
in `run.json: info.implementation.benchmark_side`.** Where a method's writer departs
from its own transform, that is stated in prose in the method's section, not recorded
per run. DeeperHistReg, whose shipped writer draws its image off its own registration,
is scored on the image as drawn only (user instruction of 2026-10-17).

The departures are listed, none is hidden; "Caveats that travel with the results" and
each method's section give them in full. In short:

- **Parameter sets.** Not every primary row is its package's default: `valis120_hires`
  is upstream's high-resolution example, `deeperhistreg_hr_gpu` a shipped 8192
  px set on a GPU, and `stare` and `valis_mirage` run the pipeline's settings, which
  their authors chose on this cohort with this metric.
- **ASHLAR.** Its nuclei are moved by a map the benchmark builds from ASHLAR's tile
  shifts; in a tile overlap that map is a weighted mean of two tiles' shifts while the
  image is a cross-fade of two copies. Its tiles are cut from stitched slides, and the
  reference cycle's tiles get noise.
- **VALIS.** Its nuclei go through VALIS's own point function with the slide's
  matrices and field copied at the yield, not through the image writer; the VALIS 1.2.0
  rows start the JVM with a heap the benchmark sets.
- **DeeperHistReg.** One row per run, under the run's own label: its nuclei go through
  the writer's arithmetic, re-stated by the benchmark, so that the image on disk is what
  is scored. That writer draws the image about S/2 px plus a few whole pixels off the
  package's own registration (roughly 1 to 3 µm on the cohort), so a good registration
  can score low.

## State

`regbench` is the official, and the only, benchmarking harness of Mirage. It scores
registration methods on the user's cyclic-IF cohort (`multiplex`), which is what every
command takes by default. The public datasets of the paper (`regbench.PUBLIC`: HyReCo,
MCMICRO, PhenoCycler, Rashid) are named only on request and are
scored BESIDE the cohort, each into tables of its own, never pooled with it (see "Public
datasets"); HyReCo has a download script (`download/`) and a landmark scorer of its own
("Landmark set"); the other public sets are downloaded and prepared by hand. The
generated rungs of the earlier harnesses are gone. **Nothing here has run on the
cluster**: every part is tested against stand-ins (small trees laid out like the pipeline's
published output, a stand-in `segment.py` / `nextflow`, recording stand-ins for each
method's package) and through the launcher's dry run. The earlier harnesses (arms, sweep,
figures, the ASHLAR arm) were removed; `benchmarks/README.md` names the tag that keeps them.

ACROBAT and IBEX were removed from `regbench.PUBLIC` by the user's ruling on 2026-10-18 (plan F tasks 22 and 25 are dropped); the `tissue` scoring kind stays as generic machinery, with no dataset of that kind.

The whole harness is drawn in one figure, from the inputs to the tables and the caveats:
`docs/figures/regbench-harness.html` (self-contained; `test_harness_diagram.py` keeps it in
step with the run list, `score.TABLES` and the launcher's job names).

| part | state |
|---|---|
| contract (`mapping.PointMap`, per-pair and per-patient methods, `run.json`) | here |
| the cohort's inputs and cases (`slurm/inputs.sbatch`, `manifest.py`, `outlines.py`, `nuclei.py`, `datasets/multiplex.py`) | here: regbench's OWN pipeline preprocessing of the cohort (`submit_cohort.sh --inputs`, `pixel_size` pinned at 0.325 µm by user ruling) and its own StarDist outlines (the cell mask traced as the pipeline's QC traces it: see "What the cohort's nuclei are"). Inputs carry fingerprints. Never run on the cohort itself |
| scoring (`pairs.py`, `score.py`) and cost (`resources.py`, `billing.py`) | here |
| adapters `stare` (the pipeline's `high` tier on the original slides, its own STITCH deferred until the clock has stopped, the option `stride`), `valis_mirage` (VALIS 1.0.0 through the pipeline's function), `valis` (1.2.0 as shipped: `valis120`, `valis120_hires`), `initial` (the identity), `deeperhistreg` (1.0.1: eight labels, one row each: the image its writer draws), `ashlar` (1.20.0: `ashlar`, `ashlar_clean`, `affine_ashlar`, `affine_ashlar_clean` and a 20-cell factorial), `palom` (2026.4.1, one run) | here, as in "Methods" and each method's section. **Only `stare` has ever been run with its package** (the only one installed where this was written); the others are written against the package's source and tested with a recording stand-in. Their first real run is the selfcheck inside each method's image |
| selfcheck and realcheck (`selfcheck.py`) | here. **Reported, never a gate** (see "selfcheck and realcheck"). Really run for `stare` only; for every other method only with fake methods |
| off-seam rows (`<variant>:non_<owner>`, `nonband_*.csv`) | here for ASHLAR's moving-side mask and PALOM's reference-side mask |
| tissue-level agreement of the written images (`tissue.py`, secondary) | here. Run on drawn images and fake methods' images only |
| the run list (`variants.py`, `python -m benchmarks.regbench runs`) | here: 55 runs (STARE strides x preprocessing levels and its medium and low tiers, ASHLAR's 24 cells, the pipeline's VALIS at three tiers x micro depth 0, 1, 2 and VALIS 1.2.0, DeeperHistReg, PALOM), six primary rows |
| the cohort launcher (`slurm/submit_cohort.sh`, `ladder.py`) | here, as in "Running the cohort on a cluster". **Never submitted**: tested through its dry run and through a stand-in `sbatch` that records its calls |
| cost by step (`cost.py`, `cost_end_to_end.csv`, secondary) | here: every number in it is measured by this benchmark |
| full pipeline cost (`compute_all.py`, `trace.py`, `slurm/compute_all.sbatch`, `pipeline_cost*.csv`, secondary) | here, as in "Cost". Reads Nextflow traces of the whole pipeline run per patient and backend; tested on generated traces, never run |

## Layout

```
mapping.py     PointMap: a point mapping and the direction it was estimated in
cases.py       the on-disk contract: case.json + points.npz in, warped.npz + run.json out
scene.py       the cohort's slides as a scene: what a patient's rounds are and which is the reference
manifest.py    the cohort as a `prepare multiplex` manifest, one row per moving round, from
               regbench's own preprocessing (slurm/inputs.sbatch); replaces `arms_manifest`
               (a `rounds.csv` of the old kind is still read: REGBENCH_MANIFEST)
outlines.py    a StarDist cell mask traced to outlines by bin/mask_to_geojson.py itself
nuclei.py      nuclei and outlines per native slide: the pipeline's StarDist, run by
               regbench, its cell mask traced to the slide's outlines ("Bare nuclei")
params_json.py a Nextflow -params-file typed against nextflow_schema.json
cells.py       nuclei of a native slide: label images, cell GeoJSON
pairs.py       the scores of one (case, variant), on bin/utils/cell_pairs.py
resources.py   the allocation, the process-tree monitor, out-of-memory detection
methods/       one module per method, and the driver that runs any of them
datasets/      multiplex writes the cohort's cases; hyreco the landmark set's pairs
landmark_io.py the landmarks of a case (truth.npz), written before case.json
landmarks.py   HyReCo's scorer (<out>/tables/hyreco/)
truth.py       per-pair error statistics of a landmark scorer; notrun.py its not-run pairs
download/      hyreco.sh: fetch and unpack the landmark set (common.sh: rb_fetch)
score.py       reads files only; writes <out>/tables, and keeps what it computed for each
               case under <out>/cache/scores ("Records: a case is scored once")
selfcheck.py   do a method's points follow the image its own writer produced?
               (REPORTED, never a gate)
tissue.py      the written images against the reference at 4000 px (ACCREDIT's score; secondary)
variants.py    the run list; regimes.py each regime's preprocessing levels for STARE
ladder.py      the memory ladder's rule: a unit goes up a rung only when it ran out of memory
billing.py     CPU-hours and memory-hours: one definition of an attempt for every method
cost.py        cost by step, from the cohort's files to each method's own image (secondary)
trace.py       a Nextflow trace.txt read into typed rows
compute_all.py the whole pipeline per patient and backend, traced; pipeline_cost*.csv (secondary)
failure_log.py every failed or lost event, and whose it is
slurm/         the cohort launcher (submit_cohort.sh) and what its jobs run
```

Three seams: datasets write cases, method adapters run the method's own software and return
a point mapping in the method's own direction, the scorer reads files only.

## Running it

From the repository root:

```bash
python -m benchmarks.tests.regbench.toy_cases CASES    # four generated test cases
python -m benchmarks.regbench list --cases CASES --dataset multiplex          # count
python -m benchmarks.regbench list --cases CASES --dataset multiplex --groups  # patients
python -m benchmarks.regbench run --cases CASES --out OUT --dataset multiplex \
    --method stare --index 0
python -m benchmarks.regbench tissue --cases CASES --out OUT   # needs OpenCV; see below
python -m benchmarks.regbench score --cases CASES --out OUT
python -m benchmarks.regbench selfcheck --method stare --work WORK
python -m benchmarks.regbench runs                     # the run list
```

The order for the cohort, on the cluster (`submit_cohort.sh`, "Running the cohort on a
cluster" has each step's job graph):

1. `submit_cohort.sh --inputs`: the input step, the pipeline's own preprocessing of the
   cohort's samplesheet (`REGBENCH_INPUT`), one Nextflow head job.
2. `submit_cohort.sh --prepare`: StarDist and the trace of every slide's outlines, then the
   cases (`prepare multiplex`).
3. `submit_cohort.sh` (the main submission; `--dry-run` first): every run on its memory
   ladder, the selfchecks beside them, then the scorer.
4. `submit_cohort.sh --compute-all` (or `--compute-only` for a results folder that is
   already scored): the whole pipeline once per patient and backend, traced, alone on its
   lanes after the scorer ("Full pipeline cost").

By hand, the same inputs and cases:

```bash
# the input step: the pipeline's preprocessing of the cohort's samplesheet, ONE Nextflow
# head job (slurm/inputs.sbatch; REGBENCH_INPUT = the samplesheet, REGBENCH_INPUTS = a folder)
python -m benchmarks.regbench.manifest --preprocessed INPUTS/preprocess/results/csv/preprocessed.csv \
    --outlines INPUTS/outlines --out rounds.csv
python -m benchmarks.regbench nuclei --manifest rounds.csv --out NUCLEI --segment   # StarDist + trace
python -m benchmarks.regbench prepare multiplex --manifest rounds.csv --cases CASES --nuclei-dir NUCLEI
```

The input step runs `--start preprocessing --stop preprocessing` at the shipped defaults
but one: `pixel_size` is pinned to 0.325 µm (user ruling 2026-10-15), through a
`-params-file` (`params_json.py`: Nextflow 26 delivers a command-line param as a
String). `cleanup_level` stays `none`, so `csv/preprocessed.csv` is written; `cleanup_work`
stays on (the published tree is what is read). It refuses to run over an earlier one
unless `REGBENCH_INPUTS_FORCE=1`, which moves it to `preprocess.replaced/<time>/`.

The manifest reads the input step's checkpoint and writes one row per moving round:
`case_id` = `<patient>_<slide>`, `group` = the patient, `ref_cells` / `mov_cells` =
`INPUTS/outlines/<patient>/<slide>.geojson`, which the nuclei step fills (they need not
exist yet). A patient without exactly one reference, a slide that is not on disk and a
slide whose channel list does not name exactly one nuclear channel (a case-insensitive
substring match of `DAPI` or `CELLTOX`; `--nuclear` narrows it) are dropped, each printed
with its reason and path; the exit code is non-zero only when no row is left. The pixel
size is the checkpoint's `pixel_size` column, 0.325 µm where a row has none
(`pixel_size_source` says which). `prepare multiplex` DROPS a round whose outlines were
not traced (`outlines not traced yet: run the nuclei step (submit_cohort.sh --prepare)`), or
were traced from another file than the slide is now; it never reads them as an empty set.
Each case carries a fingerprint of its inputs (`cases.file_fingerprint`: a file's size,
modification time and a sha256 over 16 sampled blocks), so a case prepared from other files
than the ones now on disk is recognised. For the outlines the fingerprint is that of the cell mask they were traced from (their `.traced.json`), so a re-trace from a changed mask makes the case stale and `prepare multiplex` refuses it unasked; a case prepared before fingerprints existed records none and is judged by the input paths alone.

**A `rounds.csv` written from the arms' results before 2026-10-16** (the old
`arms_manifest`) has the same columns and is still read as it was: export
`REGBENCH_MANIFEST=<it>` and `--prepare` takes it as it is. Its `*_cells` name the
arm's published GeoJSON; the nuclei step never rewrites a GeoJSON it did not trace (no
`.traced.json` beside it) and compares its StarDist run with it as before
(`outlines_from: "manifest"`). Its `*_nuclei` columns are no longer read. Cases already
prepared stay scoreable; preparing a case again from OTHER inputs (regbench's own) into the
same cases folder is refused with the reason, unless `REGBENCH_PREPARE_FORCE=1` or another
`REGBENCH_ROOT`.

The second writes the cases. Each one names the original multichannel slides
(`ref_image`, `mov_image`; `extra["ref_channel"]`, `["mov_channel"]`, `["channels_ref"]`,
`["channels_mov"]`, and `["channel_names"]` = the moving slide's) and the nuclear channel
alone (`ref_nuclear`, `mov_nuclear`). The nuclei are read with the pipeline's
`cell_pairs.from_feature_collection`, holes and multi-part cells included. A nuclei file
that is unreadable, empty, or has a vertex outside `[-1, W] x [-1, H]` of its slide refuses
that case with the file's path, and the rest are still prepared.

`run` takes one of `--case-id` (one pair), `--group` (every round of one patient) or
`--index` (the n-th case for a method that runs per pair, the n-th patient for one that
runs per patient). `--label NAME` renames a method's variants
(`variant.replace(method, label, 1)`), so two configurations sit side by side.

### The run list (`variants.py`)

`python -m benchmarks.regbench runs [--header] [--all]` prints what the cohort is
registered with: one tab-separated line per run, in a documented column order that only
ever grows at the end (the module's docstring; `-` for "none", never an empty field).
A run's options are `--opt key=value` items, and a key its method does not read is
refused by `run`, `stage`, `stages` and `selfcheck` before any work, with the keys it
does read (each adapter's `OPTIONS` constant, plus the driver's `workers` and
`stage_images`): a mistyped option is never a default run under another name.
56 runs on the cohort, 4 on a GPU, 35 per patient (`variants.counts()`; the list is generated from
tables, so the totals move with the tables; `test_variants.py` pins the total as the
literal `N_RUNS`, which a change to the list must update). The cohort has one
regime (`regimes.COHORT_REGIME`, cyclic IF). `regimes.py` also names the brightfield
regime (`bf_same`: one section re-stained; `bf_consecutive`, consecutive sections, was
removed with its only dataset on 2026-10-19) and the regime of every dataset of the paper; a run carries the
regimes that take it (`Run.regimes`: every regime by default, a generated STARE cell the
regimes whose levels make it, and the two STARE tier runs the cohort's and the landmark
set's), so the brightfield regime's 12 high-tier STARE runs and its 28 medium and low
cells are in `RUNS` (`runs --all`) and never in the cohort's list (`variants.runs_for`). `score` expects
exactly the cohort's result names and opens its summary on the six primary rows:

`stare`, `valis_mirage`, `valis120_hires`, `deeperhistreg_hr_gpu`, `ashlar`, `palom`.

- **Two methods run at two settings, and the high-resolution one is primary** (the
  user's ruling of 2026-10-11, fixed before any cohort run). VALIS 1.2.0: `valis120` is
  upstream's default pipeline, `valis120_hires` upstream's own high-resolution example.
  DeeperHistReg: the `deeperhistreg_{gpu,cpu}` rows are the package's default parameter
  set (4096 px), the `_hr` rows its shipped 8192 px set. Both settings of both methods
  are run, scored and shown; the default-setting rows are in the same family and are
  never hidden. `variants.PRIMARY_VALIS120` and `variants.PRIMARY_DEEPERHISTREG` are the
  one line that names each.
- **DeeperHistReg's primary is stated two ways, on purpose** (ruling 2026-10-19). The
  SCORER pairs against `variants.PRIMARY`, fixed before any scoring:
  `deeperhistreg_hr_gpu` is the anchor of `paired_vs_primary`, `common_block` and every
  `*_shared` column. The PAPER's DeeperHistReg primary is its best row as drawn (user
  ruling 2026-10-18, flagged in-sample: `paper/tables.py` `BEST_IS_PRIMARY`). That row is
  chosen from the scorer's tables, so the scorer cannot use it without a cycle. The
  scorer's common case set and the paper's `common` pooling can therefore differ;
  `summary.md` says so in its header (`score.PRIMARY_ANCHOR_RULE`).
- **DeeperHistReg keeps one row per run, the image its writer draws** (user instruction
  of 2026-10-17, which replaced option A of 2026-10-15). The row is named after the run
  (`deeperhistreg_hr_gpu`); no row of the estimated field is written.
- **STARE runs are generated: stride x preprocessing level**
  (`regimes.py`; `variants._stare_runs`). `STARE_STRIDES = (64, 128, 256)`. The cohort's
  regime has the levels `baseline` (STARE unchanged: the fluorescence channel as it is),
  `default` (the same), `alt` (CLAHE) and `pct` (percentile), deduplicated
  (`regimes.stare_levels`) to `none`, `clahe` and `percentile`: nine runs. A run is named
  `stare` (stride 64, no preprocessing), `stare_<level>` (stride 64) or
  `stare_s<stride>_<level>`. Beside them are two tier runs, `stare_medium` and
  `stare_low` (`variants.STARE_TIERS`; stride 64, no preprocessing, `--opt tier=<tier>`),
  the medium and low rows of `lib/RegPresets.groovy`; `stare` itself is the high tier.
  A STARE run is checked by `stare`'s selfcheck (one adapter, one pointing path).
- **STARE's run options `proxy` and `sprep`** (STARE 1.3.0). `sprep=<none|clahe|percentile>`
  is STARE's OWN `--prep`, passed to `stare coarse` and `stare reg-tile` (default `none`;
  `--pixel-size` goes with it, as the case's size, else `auto`). It is the package's
  option, so it is NOT a benchmark-side departure and is not listed in `benchmark_side`;
  it is recorded in `info.input.sprep`. `proxy=` is the one-channel image STARE reads:
  on a fluorescence case `channel` (the case's nuclear channel, the only value); on an
  RGB case `grey` (the case's own inverted-luminance proxy, the default), `haem` or
  `grey_smooth`, the last two written by the benchmark (`regbench.prep`) before COARSE,
  unbilled and listed in `benchmark_side`. Refused: a proxy of the other modality, values
  outside `PROXIES`/`SPREPS`; and, as the benchmark's cause (`benchmark`, not the
  method's), `sprep=clahe` where the pixel size is unknown, a written proxy on a case
  with a reduced proxy, and a named `proxy=grey` on a case whose own proxy is not grey.
- **What a selfcheck does not exercise.** A generated STARE cell is checked by `stare`'s
  selfcheck and an ASHLAR cell by that of `ashlar` (affine none) or `affine_ashlar`
  (affine stare), which run the 15 um / 0.002 cell. The check covers the adapter's
  pointing path, not the other `--prep` levels, shifts or noises: those cells run on
  trust in that path, and their verdict is the one of the run that checks them.
- **A run can be paused** (`active=False` on its entry). A paused run stays implemented
  and tested and its names still resolve, but it is not launched, not expected by
  `score`, not counted and not printed by `runs`; `runs --all` prints it with `0` in the
  `active` column. No run is paused now. A list in which a paused run writes a primary row,
  is what an active run's selfcheck belongs to or is what an active run waits for is
  refused (`variants.check`, at import).
- **Images** (the `image` column is the suffix of `REGBENCH_SIF_<image>`, the variable
  that holds an image's path on the cluster). Built here, the suffix being the directory
  under `containers/` in capitals with `-` written `_`: `STARE_REGBENCH` (the `stare`
  run, and `tissue` and `score`: `variants.SCORER_IMAGE`), `VALIS120` (built FROM
  upstream's `cdgatenbee/valis-wsi:1.2.0`; needs `--writable-tmpfs`), `DEEPERHISTREG`,
  `PALOM`. Upstream's published image plus the slide reader's codecs: `VALIS_MIRAGE`
  (`containers/valis100`: `cdgatenbee/valis-wsi` 1.0.0, the pipeline's digest) and
  `ASHLAR` (`containers/ashlar120`: `labsyspharm/ashlar:1.20.0`). `containers/README.md`
  has what each holds. Inside an image the harness is started with the image's own
  `python`, or with `python3` where it has no `python`: the choice is made inside the
  image (`slurm/env.sh: regbench_py`), never on the host. ASHLAR's image is the case
  that needs it: Ubuntu 22.04 with `python3-dev` and `python3-pip` and nothing called
  `python` (its Dockerfile at tag v1.20.0, `[FULL]`), and its first selfcheck on the
  cluster ended with `"python": executable file not found in $PATH`. That Dockerfile
  also sets `ENV OMP_NUM_THREADS 1`: the harness sets no thread variable, but this one
  comes with the image ("Cost", "Threads the method's own image sets").
- **Which rows a run's selfcheck status is read from** (`variants.checked_rows`,
  column `checked_rows`): all of the run's rows. Nothing is dropped:
  `variants.selfcheck_status(name, record)` is the status that is REPORTED, and no row
  is ever left out of a table because of it.
- **Which runs have a realcheck** (`realcheck`, column 15): the primary rows of the two
  VALIS adapters, `valis_mirage` and `valis120_hires` (`variants.REALCHECK_METHODS`).
  Their point path is derived by the benchmark from VALIS's matrices and fields, so it
  is checked once more on a real case: the launcher sends the run's first patient
  ahead with `run --realcheck`, beside the rest of the run, which does not wait for it.
  The verdict (`variants.realcheck_status`) is read from that row's own `realcheck.json`
  and is reported in `verdicts.tsv`; a first patient that failed or wrote no image is
  `realcheck_not_checked`. `valis120` has no primary row and no realcheck.

## The contract

**A case** is `<cases>/<dataset>/<case_id>/case.json` plus `points.npz`. `case.json` names the
two slides, the pixel size, `group` (the PATIENT, in every dataset) and, kept so an old file loads,
`family` and `pose` (empty on the cohort). `points.npz` holds the nuclei segmented on each NATIVE slide as
full polygon sets (`cells_<side>_xy`, `_ring_off`, `_ring_feat`, `_ring_hole`, `_n`; `side`
is `ref` or `mov`), read back with `cases.polys(points, side)`.

Since 2026-10-11 a side has **two polygon sets** (`cases.SETS`). The arrays above are the
**outlines**; the **bare nuclei** are a second set with the same five arrays under
`nuclei_<side>_*`, read with `cases.polys(points, side, "nuclei")`. `nuclei_<side>_outline`
(the outline each nucleus belongs to) is written only where that is known by construction.
`case.json: extra.cell_sets.<side>.<set>` records where each set came from: `source` is
`arm_geojson`, `label_image:<path>`, `stardist_rerun`, `shrunk_outlines` or `builtin`, and
the nuclei also carry `exact` (true or false) and, for a checked label image or a re-run,
the measured `agreement`; a side without nuclei has `{"source": "", "reason": ...}`.
**A case written before the second set existed still loads and scores**: it has no
`nuclei_*` array and no `cell_sets`, its outline rows are what they were, and its nuclei
rows are absent with the reason (`nuclei_missing.csv`;
`test_nuclei_sets.py::test_a_case_written_before_the_second_set_still_loads_and_scores`).

**A method** is a module with `SCOPE` (`"pair"` or `"group"`), `VARIANTS`, and one generator:

- pair: `register(case, work, opts)` yields `(variant, PointMap, info)`;
- group: `register_group(cases, work, opts)` yields `(case_id, variant, PointMap | None,
  info)`. `cases` is every case of one patient. A failed round is
  `(case_id, variant, None, {"error": msg})` (`"cause"` may say why: a memory cause
  goes up the ladder); a degraded one carries `info["problem"]`.

`mapping.PointMap(fn, direction)`: `fn(xy (N, 2)) -> (N, 2)` in full-resolution pixels of the
native slides; `direction` is `mapping.MOV_TO_REF` or `mapping.REF_TO_MOV`.

Optional: `write_native(state, case_id, dest_dir)` (the method's own writer, called after
the generator ends with the clock paused; `state` is what the method put in
`info["_state"]`), or an image written at a yield inside `opts["monitor"].pause()` /
`.resume()` and handed over as `info["native"]`. **A row is `ok` only when its image
exists, for every method alike:** a writer that fails, at a yield (`native: {"error":
...}`) or in `write_native`, fails the row it was writing for (`cause: writer`, or
`writer_oom` when the error names memory), and while `write_native` runs that row's
`run.json` says `ok: false, cause: killed, phase: writer`, so a kill there leaves a failed
row too. The registration clock is not touched: the writer stays outside it.

The driver hands a method three things through `opts` that are not options of the run and
are not recorded as such: `monitor` (the clock), `native_dir` (`f(case_id, variant)`: where
an image written at a yield belongs, `<out>/<variant>/<dataset>/<case>/native/`; the work
directory is deleted after the run) and `label`. A method whose label does not rename its
variants by the replace rule declares the names in `VARIANT_LABELS` (`valis120_hires`).

**The driver** (`methods.run_case`, `methods.run_group`), at each yield, before it resumes
the generator and with the clock paused: moves the nuclei through the map (the moving
nuclei for `mov_to_ref`, the reference nuclei for `ref_to_mov`), writes `warped.npz`
(`cells_xy`, and `nuclei_xy` = the bare nuclei of the same side through the same map;
`direction`, `band` when `info["band_mov"]` is given and `band_ref` when
`info["band_ref"]` is: a method's own seam mask, one bool per moving or per reference
nucleus; `grid`, `grid_shape`,
`grid_step` = the map on a lattice for the Jacobian statistics; `band_nuc` /
`band_ref_nuc` = the owner's seam mask on the bare nuclei, see "Off the seams"), writes
`run.json` (`nuclei_band` says how that mask was got), and calls `on_variant`. The second
set costs the method nothing: it is moved with the clock paused, like the first.

**`run.json`**: `ok`, `error`, `cause`, `problem`, `direction`, `launch_id`, `unit`
(`pair` | `patient`), `group_cases`, `register_s`, `write_s`, `resources`, `native`,
`selfcheck_status` (vestigial: always `not_checked` since selfchecks stopped gating; the tables' `selfcheck_status` comes from `verdicts.tsv`), `attempt`, `mem_requested_gb`, `cpus_requested`, `cpus_affinity`,
`blas`, `cpu_model`, `previous_attempts`, `info`, and `realcheck` (`status`, `flag`) when
`run --realcheck` was given; `retry`, `submissions` (the launcher's id of every submission
that ran or reused the result), `reused_by`, `time_limit`, `oom_watch` (`readable`,
`started`, `path`: was the cgroup's out-of-memory counter readable, did the watcher
start), and, where the scheduler had to say how the task ended, `cause_from` and
`scheduler` (`state`, `elapsed_s`, `cpu_s`). It is written as
`ok: false, cause: killed` before the method starts, so a task the scheduler kills leaves a
failed row; `realcheck.json` of an earlier attempt is removed at the same moment.

| `cause` | what happened | goes up a rung | counted as |
|---|---|---|---|
| `exception` | the method raised, or reported a failed round; or its own point mapping raised, or returned something that is not one X, Y pair per point | no | the method's |
| `oom` | the method ran out of host memory while registering or inside its own point mapping (its own error text, `resources.MEMORY_ERRORS`; the cgroup counter; the scheduler's `OUT_OF_MEMORY`; the task's script noting that Python ended by signal 9) | yes | the method's |
| `writer_oom` | the method's own writer ran out of memory: no image | yes | the method's |
| `writer` | the method's own writer failed for another reason: no image | no | the method's |
| `gpu_oom`, `timeout` | as named | no | the method's |
| `killed` | the process died, and the scheduler's end state for its task names no cause (`FAILED`, `COMPLETED`) | no | the method's |
| `killed`, no end state | the process died and the scheduler's accounting named no end state for its task within the wait (`REGBENCH_ACCT_WAIT_S`) | not as a memory cause: the `climb` task runs the unit once more at its own rung, and `cause_from` says so | the method's while it stands |
| `identical_overlap` | ASHLAR's error metric raised `correlation > total_amplitude` on pixel-identical overlaps, which exist only because the benchmark cuts tiles from a stitched slide (a cell without noise only; with noise the same raise is `exception`) | no | NOT the method's |
| `benchmark_oom` | the BENCHMARK's own lines ran out of memory with the clock stopped (the lattice, the arrays and the files around the method's point mapping; never the mapping itself) | yes | NOT the method's |
| `benchmark` | the benchmark's own lines around the method's point mapping failed for another reason | no | NOT the method's |
| `not_submitted` | a stage job of a registration in several jobs never reached the queue, or was left without the attempt the pipeline would have given it next (the tile submitter was refused or did not finish) | no | NOT the method's (the launcher's) |
| `infrastructure` | the scheduler lost the attempt (`NODE_FAIL`, `PREEMPTED`, `BOOT_FAIL`, `CANCELLED`, `REQUEUED`, `REVOKED`), twice at one rung, or the task never started | no: the same rung once more, then it stands | NOT the method's |
| `not_run` | the scorer's word for a result with no record, or with a record of another submission | | neither |

`failures.csv` and the summary count the causes that are not the method's apart
(`launches_failed_method`, `launches_lost_not_method`, `rounds_lost_not_method`). A round
lost for a cause that is not the method's (`resources.NOT_THE_METHOD`) is LEFT OUT of that
method's summaries and differences, and is in no common block; its row
stays in `cells.csv` with `excluded_from_summaries`, it is named in `excluded_rounds.csv`,
and the summary says so in bold above the failure table: submit it again before reading
those rows ("Scoring" has the rule; `score.NOT_THE_METHOD_ROUNDS` is the switch).

**One rule of attribution for every method.** A task that dies leaves the placeholder
(`killed`), and the scheduler's end state for that task names the cause: the same for
every method and for every job of a method, the last job of a STARE round included
(`OUT_OF_MEMORY` is `oom`, `TIMEOUT` is `timeout`, both the method's). `benchmark_oom`
and `benchmark` are recorded only by the driver itself, where it can show that its own
lines were running: the watcher saw the cgroup counter rise, or an error was raised,
outside the method's generator, outside its point mapping and outside its writer.

A memory error raised inside a method's generator is `oom` whatever the method was doing
at that moment (the driver cannot see inside it). `writer_oom` is a writer that REPORTED
the failure (VALIS's, ASHLAR's split), `write_native`, or a kill the watcher saw while the
method itself had the clock paused.

### Private method plug-ins (`plugins.py`)

A method that cannot be published is benchmarked without entering the repository. Its
adapter lives in `benchmarks/regbench/private/` (ignored by git), with a `plugin.py` that
declares its methods, runs, seam owners, selfcheck cases and dataset exclusions
(`plugins.py` lists the names). It is read only with `REGBENCH_PRIVATE=1`; without the
variable, or without the folder, the run list, the counts, the tables and the tests are
exactly those of this document. With it, `runs`, `run`, `selfcheck`, `verdict`, `score` and
the launcher pick the plug-in's runs up through that one hook: an image is named like any
other (`REGBENCH_SIF_<image>`), and the launcher's jobs inherit the environment. A
plug-in's rows are never primary. Nothing of a plug-in is built, pushed or tested by CI.

## Methods (the design; see "State" for what exists)

| variant(s) | software | run unit | native transform | how nuclei are moved | an RGB (brightfield) case |
|---|---|---|---|---|---|
| `stare_rigid`, `stare` | pinned `stare-registration`, mirage's `high` tier, stride 64 | pair | affine + mesh, moving -> reference | moving nuclei through `stage_warp.make_warper`; reference frame | registered on a one-channel proxy of each slide: the case's own grey proxy, or a haematoxylin / smoothed proxy the benchmark writes; STARE's own `--prep` on it; the image written is the registered proxy |
| `valis_mirage_rigid`, `valis_mirage_nonrigid`, `valis_mirage` | VALIS 1.0.0 through the pipeline's own call | patient | affine + backward field `bk_dxdy` | reference nuclei (pixels of the written image's grid) through VALIS's `warp_tools.warp_xy_inv`; moving frame | taken as it is: VALIS picks its brightfield processing when its reader calls the slide RGB (three samples per pixel); `info.valis_img_type` records what it picked, and a `problem` says so when it read the slide as fluorescence |
| `valis120_rigid`, `valis120`, `valis120_micro` | VALIS 1.2.0, **upstream's default setting** (`register()` then `register_micro()` with no argument). Run and shown; not primary | patient | same | same | same |
| `valis120_hires_rigid`, `valis120_hires_nonrigid`, `valis120_hires` | VALIS 1.2.0, upstream's high-resolution example (`--opt hires=1`). **`valis120_hires` is VALIS 1.2.0's primary row**, by the user's ruling | patient | same | same | same |
| `deeperhistreg_{gpu,cpu}`, `deeperhistreg_hr_{gpu,cpu}`, and the same four with `_fluo` | DeeperHistReg 1.0.1 (the method is RegWSI), a shipped parameter set with only the device changed; `_hr` is the 8192 px set, `_fluo` adds two benchmark-set keys. One row per run, under the run's own label | pair | the image its own writer puts on disk | reference-grid pixels of the written image through the writer's own arithmetic; moving frame. **`deeperhistreg_hr_gpu` is DeeperHistReg's primary row in the scorer** (the user's ruling of 2026-10-11; the 4096 px set is the package's default); the paper's DeeperHistReg primary is its best row (2026-10-18; see "stated two ways" above). **It follows the writer**, which by the adapter's reading sits about half a field pixel from the registration (2.5 px = 0.81 µm per axis at this 8192 px set, 1.15 µm along the diagonal; 4.9 px = 1.59 µm per axis at the 4096 px set), plus a whole-pixel padding term on uneven sizes, so its `within_1um` can be low with a perfect registration | taken as it is (an RGB TIFF, JPEG or PNG, re-contained as one RGB TIFF page with its values untouched), with its shipped (brightfield) parameters; `_fluo` is refused on it |
| `ashlar`, `ashlar_clean`, `affine_ashlar`, `affine_ashlar_clean` | ASHLAR 1.20.0, its own command, on tiles cut from stitched slides | patient | one shift per tile | moving nuclei by their tiles' shifts; reference frame | refused, with the reason (one translation per tile of unstitched tiles: two whole slides are not its input) |
| `palom_affine`, `palom` (**primary**) | PALOM 2026.4.1, its documented recipe; blocks are the reference file's TIFF tiles (2048 px on the cohort) | pair | an affine, then one shift per block of the reference | reference nuclei through the affine and their block's shift; moving frame | refused, with the reason (the README's recipe reads one channel by index; which of its colour modes is the input rule is not decided) |
| `initial` | none | | identity | |  |

Honest limits of "natively": ASHLAR is run on tiles cut from stitched slides, not raw tiles;
DeeperHistReg is run on fluorescence with brightfield defaults; STARE reads one channel,
so an RGB case is a one-channel proxy of it; STARE is run as the
pipeline's four stages in scheduler jobs of their own, with a throttled tile submitter
in place of Nextflow's queue, a retried tile grouped with its round's other retries, and
STITCH as a process inside the round's last job; `valis120` at upstream defaults is upstream's
low-resolution pipeline (which is why its high-resolution example is the primary row).

VALIS registers a patient's rounds together: its rigid step chains through
similarity-sorted neighbours (`serial_rigid.py`, `serial_non_rigid.py` in valis-wsi 1.0.0
and 1.2.0 `[PARTIAL: confirmed by grep of the calls; bodies not read]`). A round's VALIS
result therefore depends on the patient's other rounds, which is why `group` is the patient
and a per-patient method gets every round in one run.

## The transform on the nuclei

A non-rigid transform has one exact direction. Going the other way needs an inverse, which
is an approximation. Nuclei are moved only in the method's own direction, and the two sets
are compared in the frame that leads to: the reference frame for `mov_to_ref`, the moving
frame for `ref_to_mov` (`frame` in `cells.csv`).

The two frames do not give identical numbers. Pair fraction agrees to about 1e-4 between
frames; matched Dice differs by up to 1e-3; displacement p50 by 0.5 % from a 0.5 % scale
alone; displacement p90 by 0.14 µm (11 %) at 25 % local stretch; `within_1um` by 0.022 when
the residual sits at the threshold (simulations recorded in the port's plan, not re-run
here).

## selfcheck and realcheck

A method is scored through its point mapping. That is only fair if the mapping IS the
transform of the image the method wrote to disk. Two checks measure exactly that;
registration accuracy enters neither.

### selfcheck (reported, never a gate)

```bash
python -m benchmarks.regbench selfcheck --method M [--label L] [--opt k=v ...] --work W \
    [--mask overlap|seams] [--m0-from DIR] [--size HxW[:HxW[:HxW]] ...]
```

Exit 0 `ok`, 1 `inconsistent`, 2 `not_checked`, 3 `crashed`. **The exit code is REPORTED
and gates nothing** (user ruling: run, report, never gate): on the cohort no run waits for
its selfcheck, no run is kept from starting and no row is left out of a table because of
it. `regbench verdict` collects what every run's selfcheck and realcheck said into
`<out>/logs/verdicts.tsv`, and `score --verdicts` puts it in the tables. It writes `W/selfcheck.json`: `cases` (each case's folder, group
and number of rounds) and `pairs`, every variant's numbers per ROUND and per 512 px block
(one entry for a pair, one per moving round for a patient, each with `case_id`, `group`,
`round`, `rounds`).

**Every round carries every row of the run** (`selfcheck.every_row`). A round whose
process died, or that has no entry for a row, gets that row with the round's own status
when it blocks and `not_checked` otherwise (`"filled": true`), and a round's status is
never better than its rows'. The reported status (`variants.selfcheck_status`) reads
its rows twice, from the record's summary and from every round of every case, a
missing row counting as `not_checked`: the status is never better than what any case
says. A case that died can therefore not pass as checked through the rows of another
case. A missing record (or another submission's) is `not_checked`, with the reason in the
verdict's `log` column.

**The pair** (`selfcheck.make_pair`): two-channel 16-bit OME-TIFFs at 0.325 µm per pixel,
channel 0 nuclei (`DAPI`), channel 1 `dots`, on a canvas that is not square (2048 x 2560 by
default) with tissue that does not start at the origin. The moving slide is the reference
seen through a known map (1.5 degrees, a shift, a smooth 10 px field). Its dots channel
holds Gaussian dots (sigma 1.5 px, 20000 counts, zero background) on a 96 px lattice, each
jittered by up to 24 px so a mapping wrong by a whole lattice period cannot pass. The
reference's dots channel is empty. Dots are addressed by channel INDEX, never by name.
Methods with a size-dependent code path are checked on more than one case
(`selfcheck.SIZES`): `stare` also at 4096 x 5120 (a four-tile junction) and on ASHLAR's
two cases at 8192 x 10240 (below), both VALIS adapters also on a pair with a 30000 px
side, every DeeperHistReg label on a pair of
40000 x 50000 against 40000 x 48000 (the cohort's size class, with sizes that divide
exactly; see "DeeperHistReg").

Two pairs have a moving slide LARGER than its reference (added 2026-10-11; a cohort round
that overhangs its reference was otherwise first met on the cluster):

- DeeperHistReg: the same two canvases swapped (reference 40000 x 48000, moving 40000 x
  50000), so the REFERENCE is the padded one and the writer's crop back to the reference
  grid runs with a non-zero pad. Same memory as the first pair; it doubles that
  selfcheck's time (unmeasured: DeeperHistReg has not run here).
- both VALIS adapters: reference 2048 x 2560, moving 2305 x 2817. The overhang is odd
  (257 px each way), so a centred placement is half a pixel off the grid; whether VALIS's
  crop origin is then fractional is by reading only. Seconds to minutes, under the first
  rung.

**One patient of three slides for the methods that register a patient together** (user
ruling R3.19, 2026-10-11). The VALIS adapters (`valis_mirage`, `valis120`,
`valis120_hires`) and ASHLAR take every round of a patient in one run, and every cohort
patient has more than one moving round, so a pair alone never exercised the path the
cohort takes. Their `SIZES` now hold a case of the form `(reference, moving, moving)`:
one reference and two moving rounds that share it (`selfcheck.make_group`; round 2 keeps
the tissue and has its own rotation, shift, smooth field and dots: -1.0 degree, (-14, 9)
px). The whole patient goes to the driver in ONE `run_group` call, natively, and **each
moving round is checked on its own**: its points against its image.

| rows | the patient | what it adds to the pairs |
|---|---|---|
| `valis_mirage`, `valis120`, `valis120_hires` | 2048 x 2560, three slides (`patient_2048x2560`), before the large pair | VALIS orders the slides and, at upstream's defaults (`valis120*`), registers each to its neighbour in that order, so a round may reach the reference through the other round; the pipeline's settings register each to the reference. One `Valis` object, one `register`, a written image per round |
| the four ASHLAR rows | 8192 x 10240, three slides (`patient_8192x10240`) | two moving cycles in one command and one mosaic: each cycle's planes, the shared `origin_xy`, the shared reference tiles. The anchored rows read STARE's matrix of the same round |
| `stare` | ASHLAR's pair and ASHLAR's patient, the patient run round by round | nothing for STARE itself (it is a pair method): it is what the anchored ASHLAR rows read (`--m0-from`), at the cost of three more registrations of ASHLAR's canvas |

Pair methods keep their pair cases. A case's folder under `--work` is `<H>x<W>` for a pair
and the group's name for a patient (`patient_8192x10240`); case ids are `pair_<H>x<W>` and
`patient_<H>x<W>_round<i>`. `--size 8192x10240:8192x10240:8192x10240` runs one patient by
hand. A method that runs each case in its own process (`SELFCHECK_SUBPROCESS`: VALIS)
runs a patient in ONE process.

**ASHLAR is checked on one canvas, 8192 x 10240** (`selfcheck.ASHLAR_HW`): 5 x 6 of the
1950 px tiles it cuts on the cohort (pitch 1755 px), with twelve interior tiles and
four-tile corners; the selfcheck and the cohort read the same `TILE`. A canvas of 3 x 3
tiles or fewer, with all but one tile on the border, is not one of ASHLAR's cases. The
last row and column of that grid are sliver tiles (1172 and 1465 px of slide in a 1950 px tile), as on
every cohort slide; smaller canvases with thinner slivers are covered by tests against
the stand-in ("ASHLAR", "Sliver tiles").

**What ASHLAR's canvas costs STARE's selfcheck** (measured once, 2026-10-12, this laptop:
macOS arm64, 12 cores, `workers` 12, stare-registration 1.2.1). The whole `selfcheck
--method stare` (five registrations, their images, the comparisons) took 338 s and its
largest registration held 6.4 GB (process tree, `run.json: resources.peak_rss_mb`):

| case | registration s | STITCH s | peak GB |
|---|---|---|---|
| `pair_2048x2560` | 6.5 | 0.9 | 1.9 |
| `pair_4096x5120` | 21.4 | 3.3 | 4.9 |
| `pair_8192x10240` | 71.2 | 13.4 | 5.7 |
| `patient_8192x10240_round1` | 70.4 | 13.4 | 6.4 |
| `patient_8192x10240_round2` | 68.4 | 13.4 | 6.4 |

The three cases on ASHLAR's canvas are 250 s of those 338 s (210 s of registration, 40
s of STITCH). On the cluster the job has 4 CPUs, not 12, and was NOT timed: it will be
slower than this by an unmeasured factor, and stays far inside the selfcheck's first
rung (64 GB) and its time limit (24 h). The test suite runs STARE's selfcheck for real
on STARE's own two pairs only (45 s); the three cases on ASHLAR's canvas are covered
there by their place in the list and by the lookup of their matrices against a stand-in
record (`test_ashlar_adapter.py::test_stares_selfcheck_holds_the_anchors_of_ashlars_own_cases`),
and the registrations themselves run on the cluster. A real run of that canvas is not
in the suite, not even marked `slow`: the CI budget guard
(`benchmarks/tests/test_ci_time_budget.py`) lets no marker be skipped, so it would add
its 5 minutes to every run. `pair_8192x10240` and
round 1 of the patient are the same two slides under two case ids (the anchored rows look
their matrices up by case), so one of the three registrations repeats another.

**`palom` is checked on the block size its cohort row has.** PALOM's blocks are the
reference file's TIFF tiles, 2048 px on the cohort. Its selfcheck pairs are files with
2048 px tiles (`methods/palom.py::SELFCHECK_PAIR_TILE`): 16384 x 20480 (and against a
15566 x 19456 moving slide), the smallest canvas with 8 x 10 blocks of 2048 px, so the
tiling has 7 x 9 interior block corners and the seam rule is exercised at that block
size (cap 0.05 there, below).

**The check.** The method runs natively through the normal driver, writer included. The
dots are found blind in the method's own output (`find_dots`: a 15 px median background,
peaks above 8 robust s.d. and a quarter of the brightest, a second-moment width of 0.7 to
1.8 px in a 7 px window, an intensity-weighted centroid) and compared with the mapping:

- `mov_to_ref`: `fn(dot)` against the nearest detection within 3 px;
- `ref_to_mov`: `fn(detection)` against the nearest original dot within 3 px.

The residual is **the point mapping minus the image**, in the frame the mapping leads to.
A method whose writer is known to depart from its mapping declares that departure as a
function of position (`SELFCHECK_OFFSET`); it is subtracted point by point, and its sign
is part of the claim.

**A dot is a dot at the size the writer drew it** (`selfcheck.WIDTH_PX`, 0.7 to 1.8 px).
A dot as drawn measures 1.39 to 1.50 px in that window, a nucleus 1.92, one hot pixel 0,
two adjacent ones 0.35. The band takes a dot from half size to about twice its size
(measured widths of dots resampled through a linear map, bilinear and cubic: 0.60 at
scale 0.4, 0.75 at 0.5, 0.90 at 0.6, 1.12 at 0.75, 1.40 at 1, 1.57 at 1.2, 1.71 at 1.5;
the centroid stays within 0.1 px of the mapped centre down to 0.4), because the check
asks WHERE a dot is and registration accuracy is not its business: a band that only took
dots near their drawn size would make the verdict depend on the local scale of the
registration. That is not hypothetical. On the 8192 x 10240 canvas, 110 px from the
image's right edge and outside the tissue, STARE's mesh has nothing to hold on to and
shrinks the image to 0.64 x 0.89; ONE dot of 8891 is drawn there 1.19 px wide, 0.09 px
from where STARE's own map puts it. With a lower bound of 1.2 px it is not counted as a
dot, its 512 px block reads 28 of 29 found, and STARE reads `inconsistent` with every
whole-image number inside the thresholds. Detections under 1.2 px are counted in the record
(`n_narrow`), so a writer that shrinks dots is visible (per checked result in
`selfcheck_masks.csv`, and by name in the block "What the selfchecks said" of
`summary.md`); each of them is held to both
checks like any other, so a narrow fragment a writer leaves where the map puts no dot
fails. Tests: `test_selfcheck.py::test_a_dot_the_writer_shrank_is_still_a_dot`,
`::test_a_shrunk_dot_in_the_wrong_place_is_still_inconsistent`,
`::test_what_is_narrower_than_half_a_dot_is_not_a_dot`.

**The thresholds (frozen).** 95th percentile of the residual at most 0.40 px, mean signed
residual at most 0.15 px per axis, at least 99 % of the points found: on the whole image
AND in every 512 px block holding at least 8 of them.

**It is the same two checks in both directions, with the same thresholds** (since
2026-10-11; before, a `ref_to_mov` method was only held to the first, so an image with
almost half of it undrawn passed):

| | (1) every point the comparison starts from lands on its partner (99 %, whole image and per block) | (2) nothing is left over on the other side |
|---|---|---|
| `mov_to_ref` | starts from the DRAWN dots the map puts inside the image (16 px from its edge) | at most 1 % of the detections may be dots the mapping puts nothing on (`n_unexplained`) |
| `ref_to_mov` | starts from the DETECTIONS | at least 99 % of the drawn dots that belong in the image must have a detection coming back to them, whole image and per 512 px block (`n_expected`, `n_covered`; failures read "drawn dots the image does not show", "uncovered block (x, y)") |

Which drawn dots "belong in the image" is the one asymmetry left, and its reason: a
`mov_to_ref` map says itself where a dot lands; a `ref_to_mov` map cannot without being
inverted, and the benchmark never inverts a method's map. So for `ref_to_mov` the
GENERATOR's own moving -> reference map says it, and a dot is demanded when that puts it
64 px inside the image (`COVER_MARGIN_PX`; and inside `valid_box` where the writer
declares one). A registration wrong by more than 64 px at the border of the selfcheck
pair would be reported as uncovered dots. A comparison given no such map (none of the
adapters') records `coverage_checked: false`.

**A mask** (`--mask overlap` for ASHLAR, `--mask seams` for PALOM) takes points out of
check (1) only:

- **A mask never hides a result** (ruling R3.22, 2026-10-11). For every case and round
  the record holds the numbers of check (1) on THREE sets of dots (`sets`): `all` (no
  mask at all), `unmasked` (what the verdict rests on) and `masked` (the masked dots
  alone), each with its 95th percentile, mean, found share and worst 512 px block, and
  the masked share (`masked_frac`). The scorer writes them as `selfcheck_masks.csv`
  (one row per run, case, result name and set) and the summary shows them under
  "Selfcheck masks".
- **Its cap is a flag, not a verdict** (`selfcheck.MASK_MAX`, frozen: `overlap` 0.50,
  `seams` 0.10 of the dots). A row whose mask hid more than the cap KEEPS its verdict
  (and, like every verdict, it gates nothing: the run starts and is scored). It is flagged instead
  (`mask_over_cap` in the record; column `selfcheck_mask_over_cap` on every row of
  every table: `no`, or `yes: 0.57 masked, cap 0.50`, with `(via <run>)` for a run
  checked by another run's selfcheck), and `summary.md` names it. What still fails the
  selfcheck is unchanged: the accuracy thresholds on the unmasked dots (95th percentile at most
  0.40 px, mean offset at most 0.15 px per axis, at least 99 % found, per block too),
  the coverage of the drawn dots and the image's shape. Until this ruling a share
  above the cap read `not_checked`.
- **The verdict under a fixed list of caps** (`selfcheck.MASK_CAPS`): `overlap`: none,
  0.70, 0.50, 0.45; `seams`: none, 0.10, 0.05. One column per cap (`caps` in the record,
  `passes_cap_<cap>` in the table): would the row pass the accuracy thresholds on its
  unmasked dots if the mask could hide at most that share. `passes_cap_none` is the
  verdict in force.
  - **`overlap` 0.50**, beside shares measured on ASHLAR's only selfcheck canvas (8192
    x 10240: 5 x 6 tiles of 1950 px at a 1755 px pitch) with the adapter's own
    geometry: its tile grid, `tilemap.on_seam` at its 8 px dot margin, on the
    selfcheck's own dots, both rounds of the patient. No image enters, so ASHLAR is not
    needed
    (`test_selfcheck_masks.py::test_the_overlap_cap_stands_above_the_measured_share`
    repeats it).

    | where ASHLAR draws the moving tiles | share of the dots masked |
    |---|---|
    | where they were cut | 0.19 to 0.20 |
    | each tile following the pair's true map | 0.24 to 0.26 |
    | the true map plus 46 px on both axes on every tile (15 µm, ASHLAR's whole maximum shift), worst of four signs | 0.28 |
    | the same with 92 px | 0.31 |
    | the anchored rows: tiles cut by the anchor, drawn on the grid or following the true map | 0.20 to 0.21 |
    | the anchored rows, the true map plus 46 px, worst of four signs | 0.25 |

    The anchor of that measurement is the rigid part of the round's map; with the
    matrices STARE itself found on these cases the shares are the same, 0.20 to 0.21.
    The share grows with the shifts because a moving tile drawn away from where it was
    cut lies over other reference tiles; an anchored tile is cut where it belongs and
    is drawn within the map's non-rigid residual (under 10 px here) of the grid. The
    cap stands 0.22 above the worst case within ASHLAR's own limit; it is a flag, and at
    least half of the dots are always judged. What ASHLAR really does on
    this canvas is not known until it runs: a share above 0.50 is flagged on every row
    of the run, with the share, and the numbers on all dots and on the masked dots
    alone are in the record either way.
  - **`seams`**: PALOM's adapter sets the cap from the blocks a pair really has
    (`SELFCHECK_MASK_MAX`): 3.2 times the share the seams hold when every block edge is
    one (8 px either side: 1 - (1 - 16/2048)^2 = 0.016 on 2048 px blocks), so 0.05 for
    `palom` on its 2048 px blocks. 0.10 is the cap of a result that records no blocks.
- **`masked_frac` is in the record** for every variant, so the share that was not looked
  at is never implicit.
- **Check (2) stays on outside the mask.** A detection is excused only when the mapped
  dot nearest to it is itself masked (a masked dot may be drawn twice, or torn); a drawn
  dot only when it, or the detection found where it belongs, is masked. Before, a mask
  switched the leftover check off entirely.
- What a mask hides stays unseen: a map wrong only under the mask passes. For ASHLAR that
  is exactly the overlap bands; see "ASHLAR", "What the overlap bands cost".

**The image's shape** is checked against what the method declares (`mosaic_shape`) and,
when the image claims to be on the reference grid (no `origin_xy`, or a zero one), against
the reference's own shape; a declared `valid_box` must lie inside the image. A mismatch is
`inconsistent` ("image shape: ..."). The origin of an image that is NOT on the reference
grid is checked only where a method redraws the reference (ASHLAR:
`SELFCHECK_REFERENCE`, `expect_origin_xy`).

**A stage that was never checked says so in every table.** `run.json:
selfcheck_status` is now always `not_checked`: a run is never told its selfcheck's verdict,
because it does not wait for it. The tables' `selfcheck_status` column comes from
`logs/verdicts.tsv` (`regbench verdict`, read by `score --verdicts`), the run's verdict,
which `stare_rigid` shares with `stare`. The scorer also reads each ROW's own status from
the selfcheck records (`variants.row_checks`) into a column of its own, `selfcheck_row`, in `cells.csv`,
`summary_cells.csv` and the off-seam tables, and
`summary.md` prints it beside the row's name: `ok`, `no_stage_image` (shown as **NEVER
CHECKED**), `via <run>: <status>` for a row whose run is checked by another run's selfcheck
(`ashlar_clean`), or `no selfcheck record`.

| status | meaning |
|---|---|
| `ok` | the thresholds hold |
| `inconsistent` | the points do not follow the image |
| `crashed` | the method raised |
| `not_checked` | nothing could be measured: the dots were not detected, the method wrote no image, or the run ran out of memory or time (`cause` says which). Never reported as `inconsistent` |
| `no_stage_image` | a stage the method's own writer cannot draw (`stare_rigid`: STITCH writes one image). Reported, never counted as checked |

The overall status is the worst over the checked variants.

**Measured on `stare` 1.2.1** (2026-10-09, and again on 2026-10-11 after the gate changed:
every number of the first two rows came out the same, `n_unexplained` 0, `masked_frac` 0,
image shape equal to the reference's; macOS arm64, `--opt workers=4`, 46 s for both
pairs; the only method whose selfcheck has really run). The 8192 x 10240 rows are of
2026-10-12 (`workers` 12), read from the kept images with the detector as it is now:

| case | dots found | residual p50 / p95 px | mean x / y px | worst block p95 | worst block mean |
|---|---|---|---|---|---|
| 2048 x 2560 | 537 / 537 | 0.055 / 0.090 | +0.003 / +0.002 | 0.104 (20 blocks) | 0.020 |
| 4096 x 5120 | 2202 / 2202 | 0.057 / 0.091 | 0.000 / 0.000 | 0.108 (80 blocks) | 0.025 |
| 8192 x 10240, the pair and round 1 of the patient | 8891 / 8891 (one narrow) | 0.057 / 0.091 | +0.001 / 0.000 | 0.119 (320 blocks) | 0.033 |
| 8192 x 10240, round 2 of the patient | 8911 / 8911 | 0.057 / 0.091 | 0.000 / 0.000 | 0.117 (320 blocks) | 0.026 |

`stare_rigid` is `no_stage_image`. The same image, with STARE's own mapping deliberately
changed, on the 2048 x 2560 pair:

| mapping checked against STARE's image | status | found | p95 px | mean x px | blocks failed |
|---|---|---|---|---|---|
| STARE's own | `ok` | 537 / 537 | 0.090 | +0.003 | 0 / 20 |
| + 0.10 px on x | `ok` (inside the thresholds) | 537 / 537 | 0.175 | +0.103 | 0 / 20 |
| + 0.15 px on x | `inconsistent` | 537 / 537 | 0.224 | +0.153 | 12 / 20 |
| + 0.30 px on x | `inconsistent` | 537 / 537 | 0.373 | +0.303 | 20 / 20 |
| + 0.5 px on both axes | `inconsistent` | 538 / 538 | 0.781 | +0.503 | 20 / 20 |
| scale error of 0.05 % about the centre | `inconsistent` | 537 / 537 | 0.685 | +0.005 | 18 / 20 |
| scale error of 0.1 % about the centre | `inconsistent` | 537 / 537 | 1.354 | +0.008 | 20 / 20 |
| + 96 px (one lattice period) | `inconsistent` | 7 / 519 | | | 20 / 20 |
| 5 px wrong on the 8 % of dots along one edge | `inconsistent` | 496 / 537 | 0.090 | +0.004 | 4 / 20 |
| 1 px wrong on 4 % of dots, one neighbourhood | `inconsistent` | 537 / 537 | 0.101 | +0.042 | 1 / 20 |
| the rigid stage's mapping (`M0` only) | `inconsistent` | 74 / 539 | | | 20 / 20 |
| the identity | `inconsistent` | 0 / 543 | | | 20 / 20 |
| the same function declared `ref_to_mov` | `inconsistent` | 2 / 537 | | | 20 / 20 |

The rows of this second table were measured on 2026-10-09 and not re-run: every one is a
`mov_to_ref` comparison without a mask, which the change of 2026-10-11 does not touch
(the first table, re-run, is the evidence). What that change adds is not measurable on
STARE, which is `mov_to_ref`; it is shown on drawn dots, where the image is known exactly
(`test_selfcheck_masks.py`, a 1024 x 1536 image, the points right wherever something is
drawn):

| image | `mov_to_ref` | `ref_to_mov`, before | `ref_to_mov`, now |
|---|---|---|---|
| complete | `ok` | `ok` | `ok` |
| 7 % of it undrawn | `inconsistent` | `ok` | `inconsistent` (drawn dots the image does not show; uncovered blocks) |
| 20 % undrawn | `inconsistent` | `ok` | `inconsistent` |
| 45 % undrawn | `inconsistent` | `ok` | `inconsistent` |
| a mask hiding 93 % of the dots, nothing else wrong | `ok`, flagged `mask_over_cap` (R3.22; `not_checked` between 2026-10-11 and that ruling) | `ok` | `ok`, flagged |
| 12 stray detections outside a mask that hides a quarter | not looked at before, `inconsistent` now | | |

So the smallest constant offset the selfcheck objects to is 0.15 px per axis, the detector's
own floor being 0.09 px at the 95th percentile. The row "1 px wrong on 4 % of dots" is
caught by one block alone, with every whole-image number inside the thresholds.

**What it cannot see.** A defect the writer and the point path SHARE passes (both read
the same wrong field); a stage with no image of its own is not checked; the default pair
does not exercise whole-slide code paths, which is what the larger pairs and the
realcheck are for.

**What a method provides** for it (all optional; signatures in `selfcheck.py`):
`write_native`, or an image handed over at a yield as `info["native"]` (with
`opts["stage_images"]` set, also for its earlier stages); `write_stage`; `warp_extra` (its
own writer applied to the one-channel dots image, when its output has one channel);
`SELFCHECK_OFFSET(info)`; `SELFCHECK_MASK(info, mask)` (`--mask overlap`, `--mask seams`);
`SELFCHECK_SUBPROCESS = True` (one process per pair). A native dict may carry
`dots_file`, `dots_channel` and `origin_xy`.

### realcheck (one real case per method)

```bash
python -m benchmarks.regbench run ... --realcheck      # exit 3 when a realcheck is flagged
```

After the method's writer, with the clock stopped, for every variant that has a native
image: nuclei are detected with the built-in segmenter in five 2048 px crops of the native
nuclear output (the four corners of the tissue's bounding box and its centre) and matched,
within one nuclear radius, to the point-mapped native nuclei (`mov_to_ref`: detections
against the mapped moving nuclei; `ref_to_mov`: detections mapped back, against the moving
nuclei). The baseline is the same detector on the unwarped slide of the frame the mapping
leads to (for `ref_to_mov` the crop is carried there by the mapping itself; nothing is
inverted). `realcheck.json` holds, per crop, `n`, `matched_fraction`, `baseline_fraction`,
`offset_px_mean_x/y` (after the variant's declared offset) and `err_px_p50`. **`flag`** is
true when a crop's mean offset exceeds 0.3 px on an axis or its matched fraction is under
0.7 of the baseline's; `status` is `ok`, `flagged` or `not_checked` (no crop held 20
nuclei). The median distance is shown and not used: it measures segmentation noise.
`run.json: realcheck` repeats `status` and `flag`. For the VALIS primary rows the flag is
REPORTED (it is the only check at cohort field scales), never a gate:
`slurm/submit_cohort.sh` runs the first unit of such a run ahead with `--realcheck`,
in a ladder of its own beside the rest of the run, which does not wait for it
("Running the cohort on a cluster", step 4). `regbench verdict` writes its status into
`logs/verdicts.tsv` and the summary prints it. It has only run on generated slides, and
never on a scheduler.

## Scoring

Nuclei are segmented once on each native slide, before registration, so they do not depend
on any method's output. Pairing and IoU are `bin/utils/cell_pairs.py`, imported
(`pairs.py` loads that one file; `bin/utils` is not put on `sys.path`).

- **Two polygon sets, every number on both.** The **outlines** are the headline: on the
  cohort, each nucleus grown by 10 px, which is what the pipeline's QC publishes; every table and column that existed means the outlines and
  is unchanged. The **bare nuclei** get the same numbers in tables of their own (below,
  "Outlines and bare nuclei"). Neither set depends on any method.
- **Gate:** `1.5 x median_equivalent_radius` of the two native sets, once per case, the same
  for every method. Each polygon set has its OWN gate, from its own areas: the nuclei's
  is 1.5 x the median equivalent radius of the nuclei.
- **Pairing:** the pipeline's (`cell_pairs.match_lsa`), with mutual nearest neighbour
  beside it (`_mnn`). The pipeline solves a linear sum assignment per connected
  component of the candidate pairs (most pairs first, then least total distance) and,
  in a component of more than 5,000 cells, pairs by mutual nearest neighbour instead.
  Every row says how much of its pairing was that fallback: `n_fallback_cells`,
  `n_fallback_components`, `largest_component_cells`, `n_components`, `fallback_frac`
  (cells in such a component over all cells of both sides; `fallback_frac_null` for
  the null's placements), and the summary counts the cases (`n_fallback_cases`,
  `fallback_frac_max`). On a jittered lattice of nuclei 26 px apart under a 30 px gate
  the whole tissue is one component, so there the headline is mutual nearest neighbour.
- **The exact assignment, as extra columns (`*_exact`, `exact_status`).** The same gate
  and the same objective with no fallback: `n_pairs_exact`, `pair_fraction_exact`,
  `dice_matched_exact`, `iou_p50_exact`, `frac_iou_ge_0.5_exact` and the four
  `displacement_*_exact`, for the registered placement (the null has none). They are
  never used to rank, to choose a shared subset or in a difference; the headline stays
  the pipeline's pairing. Where no component fell back the two are one pairing and the
  columns repeat the headline. A component the pipeline left to its fallback is solved
  as a sparse minimum-weight full matching with the pipeline's own `big_m`
  (`pairs.solve_component`); the result is checked against the largest number of pairs
  the candidates allow, and against the pipeline's dense solver in the tests. Its time
  grows with (cells of the smaller side) x (cells of both sides): measured on one
  laptop core 2 s at 19,000 cells a side, 44 s at 85,000 (0.2 GB), 225 s at 190,000.
  A component above `pairs.EXACT_MAX_WORK` (1.5e10: about 85,000 cells a side) is not
  solved: that row's `*_exact` cells are blank and `exact_status` says why; nothing is
  approximated. With more pairs the exact assignment also has LONGER pairs than mutual
  nearest neighbour (in that lattice 5 % more pairs at more than twice the mean
  distance): most-pairs-first chains cells past a missing one.
- **The full pairings, beside the headline (Task 21b, `full_pairs.py`).** USER
  2026-10-18: "I would like score to also test full lsa and full mnn". `full_lsa` is
  the exact assignment in EVERY component with no fallback AND no cap
  (`pairs.exact_pairs(max_work=inf)`): `*_full_lsa` in `cells.csv` (the same score
  stems as `*_exact`). It differs from `*_exact` only where the cap was hit, so a row
  whose `exact_status` is "exact here" or "solved exactly" COPIES `*_exact`
  (`full_lsa_status = full_lsa_from_exact`, no solve) and only a row above the cap (or a
  record made before the exact columns) is solved, from the same polygon sets and
  gate. Its cost is recorded, never capped: `full_lsa_seconds`,
  `full_lsa_largest_component_cells`, `full_lsa_work` (in the cap's unit). No timeout,
  no approximation: a result the solver does not certify is NA (`not_certified`), a
  solve that raises is NA (`failed`, tried again by the next scoring), a solve that
  killed its process (OOM killer, walltime: its `full_pairs/<key>.started` marker is
  left with no record) is NA (`interrupted`, never tried again, or the same kill would
  take the case's headline rows with it; `fresh` tries again), and each is a
  `losses.csv` row of stage `full_pairs` with its headline row kept. `full_mnn` is
  mutual nearest neighbour on the whole case: the `*_mnn` columns, never recomputed.
  The registered placement only (the null has none). The summaries carry, beside every
  headline, `<h>_full_lsa_med` and `<h>_full_mnn_med` for `pair_fraction` and
  `dice_matched` (aggregated as `*_exact_med`), `fallback_frac_med`,
  `fallback_frac_max` and `n_full_lsa`; the paper tables `dice_full_lsa_med`,
  `dice_full_mnn_med`, `fallback_frac_med`, `fallback_frac_max`, `n_full_lsa`
  (`table_schema._FULL`). Nothing ranks on them: no best row, no common block, no
  shared subset, no difference. Their records are kept beside the pair records
  (`cache/scores/<dataset>/<case>/full_pairs/`) under a key of their own (the pair
  record's key plus `full_pairs.py`'s sha256), so changing `full_pairs.py` makes them
  again and no pair record.
- **Per case and variant:** pair fraction, matched Dice, IoU p50, share of pairs with
  IoU >= 0.5, centroid displacement in µm (p50, p90).
- **Pair fraction is pairs over the smaller NATIVE count, in both directions.** A cell the
  method's map sends nowhere (outside the image it wrote: a non-finite centroid) stays in
  the denominator as an unpaired cell whether the method maps the moving cells
  (`mov_to_ref`) or the reference cells (`ref_to_mov`): the same lost cells give the same
  number (`test_score_units.py::test_cells_mapped_nowhere_count_the_same_in_both_directions`,
  with equal and unequal counts). The review's reading that the two directions are
  treated differently was checked and is NOT what the code does; what is true, in both
  directions alike, is that losing only cells of the LARGER set (cells that had no
  partner to lose) does not lower pair fraction. Since 2026-10-11 every row therefore
  also reports `n_unmapped` and `unmapped_frac` (cells mapped nowhere, over all cells of
  both sides), the summary has `unmapped_frac_max`, and `within_*um` (below) does object.
  Nothing changed in `pair_fraction` itself.
- **Null (`_null`):** the same with one set shifted four nuclear radii, averaged over eight
  directions, with the minimum and maximum. The null depends strongly on density: every
  case reports `coverage` (nuclear area over the area of the 16-radius bins that hold
  nuclei) and `nn_distance_um_p50` beside these numbers.
- **`pair_fraction_excess`** = (pair fraction - null) / (1 - null). It is density-dependent
  too: where nuclei touch, a perfect registration scores near zero.
- **Three pair fractions, side by side** (Task 29, user 2026-10-18 "fix both, report
  all"): `pair_fraction` (pairs / min n: the headline, unchanged), `pair_fraction_maxn`
  (pairs / max n: a loss on the LARGER side lowers it) and `pair_fraction_sym`
  (2 pairs / (n_ref + n_mov)), each with its `_null`, `_null_max` and `_excess`. The two
  new ones are computed by `score.add_pair_fractions` at table time from the recorded
  `n_pairs`, `n_ref`, `n_moving`, `n_pairs_null` and `pair_fraction_null_max` (the null
  moves the moving side only, so its counts are the record's): no pair record changes.
- **A folding map is rewarded by the paired numbers.** Snapping each moving nucleus onto its
  nearest reference nucleus from a failed pose pairs well and has zero displacement; only
  the two-sided curve objects. So `fold_pct` and `sd_log_jac` (forward evaluation of the
  map on a lattice, no inverse) are columns of every row.
- **Pairing-free curve (`within_<x>um`):** share of nuclei with a partner centroid within
  x µm, from each side (`_ref`, `_mov`), reported as the smaller, with `count_ratio`.
  `within_1um` and matched Dice are the two numbers that resolve a sub-nuclear error;
  `within_3um` does not, and `count_ratio` is not a ceiling. None of them ranks: the
  summary claims no ranking at any number of units.
- **Reading rule, fixed before any cohort run:** displacement is censored at the gate and
  conditional on being paired, so it is not used to rank methods. It is published for every
  case with the `displacement_readable` flag (pair fraction exceeds the null's maximum by at
  least 0.10). Dice is read only where pair fraction clears the null's maximum.
- **Both are read on a subset of cases, and the subset is the SAME for every row.**
  `dice_matched_shared` and `displacement_um_p50_shared` are over the cases where EVERY
  compared row qualifies (below: the primary rows that ran), with `n_dice_shared` and
  `n_readable_shared`: these are the numbers that may be set beside another row's. The
  per-row versions (`dice_matched_ranked` with `n_dice_ranked`, `displacement_um_p50` with
  `n_readable`: the cases where that row alone qualifies) stay beside them and are NOT
  comparable between rows: a method that fails its hard cases is read there on its easy
  ones. A case with no pair has no Dice; it is counted (`<headline>_n_nan`, "dice NaN
  cases" in `summary.md`) instead of vanishing, and read BOTH ways (Task 29, user
  2026-10-18 "report both ways"): `dice_matched_med` (and `_mean`, `_min`, `_max`,
  `_min_case`, `_null_med`, and the paired `d_dice_matched`) zero-fill it (a failure
  scores 0, the paper's rule: the headline), `dice_matched_zero_*` says so explicitly,
  and `dice_matched_skip_*` (and `d_dice_matched_skip`) leave it out (finite values only,
  the rule before). Both are printed. In a PAIRED difference (`d_dice_matched` in
  `paired_vs_primary`, and the paper's `d_dice_vs_stare_*`) a case where BOTH rows have
  no pair carries no comparison and is left out under both rules (the same rule in the
  scorer and the paper since T29's fix round 1). `dice_matched` is the only headline a
  failure can make NaN (`within_*um` is NaN only without a pixel size).
- **No pixel size:** µm columns and the curve are NaN, not zero.

A failed, killed or never-started run is a row scored at the unregistered pose
(`imputed_initial`), with `problem` and `cause` filled; never a gap.

**A round lost for a cause that is not the method's is not imputed** (`infrastructure`: the
scheduler lost the attempt; `benchmark_oom` and `benchmark`: the benchmark's own lines
ran out of memory or failed; `not_submitted`: the launcher never queued a stage job).
Scoring it at the unregistered pose would charge the method for the cluster, and dropping
it silently would hide it, so: it is left out of that method's summaries and differences
(`n_excluded_not_method`, "left out" in `summary.md`), it is in no common
block for any method (it is unregistered), it is named at the top of `summary.md` and in
`excluded_rounds.csv`, and its row stays in `cells.csv` with `excluded_from_summaries`.
This is the coordinator's decision pending the owner's; `score.NOT_THE_METHOD_ROUNDS`
(`"exclude"` | `"impute"`) is the one switch.

**The unit.** `n_patients`, the line "descriptive; no ranking is claimed" (printed for
every dataset, whatever its number of units) and every across-unit statistic count the
independent unit (`score.UNIT_IS`, column `unit` of `cells.csv`), not the case's group:

| dataset | one unit is | so |
|---|---|---|
| `multiplex` | a patient | rounds of a patient are reduced to their median first |

**The common block** (`subset=common`, `common_block.csv`) is the cases every COMPARED row
registered. A row is compared when it is a primary row (any registration row when no
primary list is given) that ran in what is being scored: at least one of its cases has a record of
this submission. A primary run that was never launched (`submit_cohort.sh
--runs "stare ashlar"`) is named with its reason and is not allowed to empty the block
for everyone. What a selfcheck said never takes a row out: the checks are reported, never a
gate. The block is always printed, even when it is empty.

**Uncertainty** (`paired_summary.csv`, "Differences against `stare`" in `summary.md`): for
each row, the median over units of its paired difference against the `stare` row (each
unit's rounds reduced to their median on the cases both rows have), with a 95 % percentile
bootstrap interval over UNITS (2000 resamples, seed 20261011; a unit's rounds are never
split), and the mean beside the median (`d_<headline>_mean`). The interval is over units
and is narrower than the truth: a resample of a handful of units takes few values, and
the measured units stand in for all units; under two units there is none. It describes
the spread between units. No p-value is computed and no ranking is derived; the line
"descriptive; no ranking is claimed" is printed with it at every number of units.

**Beside every median: the mean, and `ok/cases` on the same line.** A median over
patients does not move when a method fails on a minority of them. Each headline of
`summary.md` is therefore printed as `median (mean over the same units) [minimum over
cases]` on the line that carries `ok/cases` (`<headline>_mean` in `summary_cells.csv`),
and the cost medians have their means (`*_mean`, `*_common_mean`) and the
patients completed.

It is self-consistency on the registration channel: there is no held-out structure and no
landmarks on real slides.

### What the cohort's nuclei are

**The definition (regbench's own, since 2026-10-16; user ruling 2026-10-15: the pipeline's
StarDist model and settings, nuclei grown by 10 px).** The nuclei step
(`nuclei.py`, `slurm/nuclei.sbatch`) segments ONE channel of each native slide, the nuclear
marker, with `bin/segment.py` in the pipeline's StarDist image, with every flag
SEGMENT's `ext.args` renders for StarDist and the same params (the site's model and model
folder, `seg_expand_distance`, default 10 px), on the preprocessed slides of regbench's own
input step (the pipeline's preprocessing, `pixel_size` pinned to 0.325 µm). Its
`*_cell_mask.tif` (the StarDist nuclei labels grown by `seg_expand_distance` px, no label
growing into another) is traced to outlines by the `write_geojson` of
`bin/mask_to_geojson.py` at the pipeline's `simplify_tolerance` (`outlines.py`). That
GeoJSON is the slide's outline set, `ref_cells` / `mov_cells` of the case. The bare nuclei
are the same run's nuclei mask (next section).

So the outlines are **nuclei found on the nuclear channel alone, each grown by 10 px (3.25
µm at 0.325 µm per pixel) until it meets a neighbour**. They are not whole cells found on
membrane or marker channels, so outlines do not differ between rounds because the markers
differ (they would at the pipeline's default segmenter, InstanSeg on every channel). They
are also not bare nuclei, and three things follow for every method alike:

- the gate (1.5 x the median equivalent radius) and the null's shift (4 radii) are sized
  from the GROWN outline, so both are larger than a nuclear radius would make them;
- matched Dice is the overlap of grown outlines, whose shape in dense tissue is set by the
  neighbours as much as by the nucleus; it has a ceiling below 1 that is not
  registration;
- centroids are those of the grown outline: close to the nuclear centroid for an
  isolated nucleus, pulled away from it where neighbours block the growth on one side.
  `within_1um` inherits that.

**Its equivalence to the outlines the pipeline publishes is DERIVED from the code, link by
link, and was NOT measured on cohort slides.** The pipeline's QC (`reg_qc: 2`) segments each
native slide before registration (`SEG_QC_SEGMENT` is `SEGMENT` under an alias,
`subworkflows/local/seg_qc.nf`), on channel 0 (`lib/SegBackends.groovy`), traces the **cell**
mask (`SEG_QC_GEOJSON`), and publishes only that GeoJSON. Regbench's run gives the outline
the pipeline would have published from such an image with the same model and settings.
`test_outline_definition.py` pins each link:

| link | pinned by |
|---|---|
| the expansion is `seg_expand_distance` (10) | `test_the_rerun_grows_nuclei_by_the_pipelines_expand_distance` |
| every flag SEGMENT's `ext.args` renders for StarDist is rendered, from the same params | `test_the_rerun_passes_every_flag_...`, `test_the_model_and_settings_are_the_params_ext_args_reads` |
| channel 0, the pipeline's StarDist image, the GPU | `test_the_rerun_runs_in_the_pipelines_stardist_image_on_channel_zero` |
| the cell mask is one whole-image `expand_labels` of the nuclei | `test_the_cell_mask_is_skimage_expand_labels_of_the_nuclei` |
| the trace is `bin/mask_to_geojson.py`'s `write_geojson` at `simplify_tolerance` (site config, else 1.0) | `test_outlines_are_traced_by_...`, `test_the_tolerance_follows_the_site_config_like_the_pipeline`, `test_seg_qc_geojson_still_traces_...` |

What no test can pin: the site's model files (the same name and folder are rendered, the
files are the site's), the library versions inside the images (the trace runs in the
scorer's image, the pipeline's in `bolt3x/mirage-quantify`), a GPU's run-to-run
differences, and `REGBENCH_STARDIST_CPU=1`, which drops `--use-gpu` (the pipeline's QC
runs on the GPU, `seg_gpu = true`): StarDist on CPU need not give the GPU's labels.
`--dapi-channel` is the one rendered difference: the pipeline passes 0 (and
refuses a slide whose channel 0 is not nuclear), regbench the manifest's nuclear index,
which is 0 on every slide the pipeline would segment.

### Outlines and bare nuclei

**The two definitions.**

| set | on the cohort |
|---|---|
| `outlines` (headline) | regbench's own outline of each native slide, defined to be the pipeline's QC outline: a StarDist nucleus grown by 10 px until it meets a neighbour |
| `nuclei` | the StarDist nuclei themselves, from the first source below that applies |

**Why both.** Matched Dice on grown outlines is dominated by the rim: at a nuclear radius
of about 12 px a 10 px rim is roughly two thirds of the outline's area, so a given
displacement costs a grown outline a smaller share of its area than it costs the nucleus,
and in dense tissue the outline's shape is clipped by its neighbours and not by the
nucleus at all. Dice on the bare nuclei is stricter per micrometre and has no clipping.
The outlines stay the headline because they are what the pipeline's own QC reports; the nuclei say how much of an outline number is rim. Both sets are
segmented on the NATIVE slides before any registration, so neither depends on a method.

**Where the nuclei and outlines of a cohort slide come from** (since 2026-10-16 the one
source; the label images an arm's work directory or a segmentation arm offered are no longer read):

1. **A StarDist run** (`python -m benchmarks.regbench nuclei`, `slurm/nuclei.sbatch`:
   an array over SLIDES): the pipeline's `bin/segment.py`, unchanged, on the native slide,
   keeping its `*_nuclei_mask.tif` and tracing its `*_cell_mask.tif` to the slide's
   outlines (with a `.traced.json` beside them: the mask, its fingerprint, the slide's
   fingerprint, the tolerance). The rendered command, and where each word is from:

   ```
   python <repo>/bin/segment.py --image <slide> --output-dir <nuclei dir>/<group>/<slide>/rerun \
     --dapi-channel <c> --model-dir $REGBENCH_STARDIST_MODEL_DIR --model-name $REGBENCH_STARDIST_MODEL \
     --n-tiles 16 16 --expand-distance 10 --pmin 1.0 --pmax 99.8 --use-gpu
   ```

   | word | from |
   |---|---|
   | `segment.py --image … --output-dir …` | `lib/SegBackends.groovy:100` (`entrypoint`), `modules/local/segment.nf:77-79` |
   | `--dapi-channel` | `lib/SegBackends.groovy:101`, `modules/local/segment.nf:80`: the pipeline passes `0`, its preprocessing having put the nuclear channel first; here it is the manifest's `ref_channel` / `mov_channel`, which is 0 on every slide the pipeline segments |
   | `--model-dir`, `--model-name` | `conf/modules.config:1021-1022` (`segmentation_model_dir`, `segmentation_model`); the site takes them from its overlay (`conf/ieo.config` at IEO), which is not in the repository: the two environment variables, or, when they are unset, that file's two keys (below) |
   | `--n-tiles 16 16` | `conf/modules.config:1023`; `nextflow.config:273-274` |
   | `--expand-distance 10` | `conf/modules.config:1024`; `nextflow.config:275` |
   | `--pmin 1.0 --pmax 99.8` | `conf/modules.config:1025-1026`; `nextflow.config:268-269` |
   | `--prob-thresh` | absent: `conf/modules.config:1029`, `nextflow.config:272` (`null`); `REGBENCH_STARDIST_PROB_THRESH` adds it if the site sets one |
   | `--use-gpu` | `conf/modules.config:1057`, `nextflow.config:267` (`seg_gpu = true`); `REGBENCH_STARDIST_CPU=1` drops it |

   None of these is set by the benchmark itself
   (`test_nuclei_cohort.py::test_the_rerun_follows_the_pipelines_own_configuration` reads
   the defaults out of `nextflow.config` and the flag list out of `conf/modules.config`).
   The outlines are the trace of the cell mask the same call grows from its nuclei, so
   the comparison below is a SELF-CONSISTENCY check (a mismatch is a tracing bug, not
   another segmentation; with an old manifest's published outlines it is the comparison
   with those, as before): the counts within 1 %, and at least
   0.99 of the outlines' centroids with a re-grown centroid within 1 px
   (`RERUN_MATCH_PX`, `EXACT_MATCH`), beside the one-nucleus share above. What a re-run
   is then (user ruling 2026-10-11):

   | the re-run's share of published outlines holding exactly one of its nuclei | what happens |
   |---|---|
   | under 0.90 (`ACCEPT_ONE_NUCLEUS`) | **REFUSED**: not used. A `REFUSED <slide>` line with the share and both counts is printed, the task exits 1, `--list` shows status `refused` with that reason, the slide has no nuclei rows, and the reason is in its cases' `cell_sets` and in `nuclei_missing.csv`. It is not segmented again unless `--force` is given (`REGBENCH_NUCLEI_FORCE=1` through the launcher, which the array task passes on): the same command gives the same nuclei, so the model, the channel or a site setting is what to look at. The array task of such a slide says so and exits 1 without running StarDist |
   | 0.90 or more, but short of any exact threshold (0.99 of outlines, counts within 1 %, 0.99 of centroids within 1 px) | kept, marked `exact: false`, printed as `DISAGREEMENT <slide>` with the numbers |
   | every exact threshold met | kept, `exact: true` |

   **A kept re-run is the re-run of its settings.** When the command of a re-run is
   rendered, the model and every setting it is rendered with are written beside its
   masks (`rerun/produced_by.json`), and the slide's record keeps THAT (`settings`,
   `settings_recorded`), not what resolves on the day the mask is checked. When the
   model or a setting later resolves to another value (another `REGBENCH_STARDIST_*`,
   another site config), a kept re-run (usable or refused) is not reused: a `STALE
   <slide>` line names what changed, the old masks are set aside under `rerun_stale/`,
   and the slide needs its re-run again.

   **A record is of one image.** A slide's record holds the fingerprint of the slide
   file its nuclei were computed from (`image_fingerprint`: size, modification time
   and sampled bytes, `cases.file_fingerprint`) and, when they were traced from a
   label image, that file's (`label_image_fingerprint`). When the file under that name
   is another one now, or the record names none, the record is not used: a `STALE
   <slide>` line says so, what is kept is set aside (`rerun_stale/`) and the slide is
   decided again (a label image is traced and checked again; a re-run is run again).
   `prepare multiplex` reads the same fingerprints and does not take the nuclei of
   another file: the case then has no nuclei, with that reason.

   **`--force` decides every slide again** (`REGBENCH_NUCLEI_FORCE=1` through the
   launcher), whatever its record says, usable ones included: a `FORCED <slide>` line,
   the kept re-run set aside, a label image of the manifest traced and checked again,
   and StarDist run again for every slide that has no usable label image. The polygons
   and the record are then those of the new run.

   **The model and the settings come from the site config** when the two variables are
   not exported (`benchmarks/regbench/site_config.py`). The site overlay holds, in its
   params block, `segmentation_model = '<name>'` and `segmentation_model_dir =
   '<dir>'` (as `conf/site.config.template` does). Order, per value: an exported
   `REGBENCH_STARDIST_MODEL` / `REGBENCH_STARDIST_MODEL_DIR` /
   `REGBENCH_STARDIST_PROB_THRESH` wins; else the site config, which is
   `REGBENCH_SITE_CONFIG` when set (`none` reads no file) and otherwise
   `conf/ieo.config` of the checkout if it is there; else, for the settings only, the
   default in `nextflow.config`. From the same file the re-run takes any override of
   `seg_n_tiles_y`, `seg_n_tiles_x`, `seg_expand_distance`, `seg_pmin`, `seg_pmax` and
   `seg_prob_thresh`, renders the site's value into the `segment.py` call, and keeps
   every value with its source in the slide's record (`nuclei.json: settings`). The
   file is read comment-blind, those keys only, as plain `key = literal` lines; a key
   set twice to different values (two profiles), or to an expression, is refused with
   the way out instead of guessed. `nuclei --settings` prints `name, value, source`
   for each (a value that differs from the repository's default says so), `nuclei
   --command` prints the same on stderr, and the launcher prints them as `# StarDist
   re-run:` lines, in a dry run too. The site config is read, never written, and
   nothing from it is copied into the repository.
3. **Fallback, only when asked** (`prepare multiplex --nuclei-fallback shrink`,
   `REGBENCH_NUCLEI_FALLBACK=shrink`): the outline with a 10 px rim taken off again,
   labelled `shrunk_outlines`, `exact: false`. It is NOT the nucleus: growth stops where
   two cells meet, so taking the full rim off a clipped side cuts into the nucleus, and
   it is too small wherever cells touch. Never used silently.

A slide with nuclei from no allowed source is still prepared: its cases are scored on the
outlines, their nuclei numbers are absent, and `nuclei_missing.csv` and the top of
`summary.md` give the reason. Nothing is imputed.

**Tracing.** The published outlines are polygons the pipeline simplified to 1 px
(`bin/mask_to_geojson.py`); nuclei read from a label image are traced by
`cells.polys_from_label_image`, unsimplified. The grown
outlines are simplified to the same 1 px (`cells.OUTLINE_TOLERANCE_PX`), as the cohort's
are, and the nuclei are not (an unsimplified grown contour has about 125 vertices against
about 10, and the pipeline's raster IoU, which scores every pair 18 times with the null,
costs four times as much on it). Both rounds of a pair are traced the same
way within a set, so this does not favour a round; it is one more reason not to compare
an outline number with a nuclei number as if only the rim differed.

**Tables.** The nuclei have the SAME tables as the outlines, with the same columns, in
files of their own: `cells_nuclei.csv`, `summary_cells_nuclei.csv`,
`common_block_nuclei.csv`, `paired_vs_primary_nuclei.csv`, `paired_summary_nuclei.csv`,
`nonband_cells_nuclei.csv`, `nonband_summary_nuclei.csv`. Every cell table of either set
carries a `set` column (`outlines` | `nuclei`), so two files can be stacked. This was
chosen over `_nuc` columns (which would double the about 100 columns of `cells.csv`) and
over one file with both sets (which would change the meaning of "one row per case and
variant" for everything that reads `cells.csv`). On the nuclei the gate, the null, the
excess, matched Dice, IoU, `within_<x>um`, displacement, the shared-subset numbers, the
common block, the off-seam rows and the paired differences are all computed from the
nuclei alone: the shared subset is the cases where every compared row qualifies ON THE
NUCLEI. `summary.md` prints, under each dataset's table, "Outlines and bare nuclei, side
by side" for the primary rows, each cell `outlines / nuclei`:

```
| method | cases: outlines / nuclei | pair_fraction: outlines / nuclei | ... | dice_matched: outlines / nuclei | dice_shared: outlines / nuclei | within_1um: outlines / nuclei | ...
| stare | 1 / 1 | 1 / 1 | ... | 0.84 / 0.6923 | 0.84 / 0.6923 | 1 / 1 | ...
```

(the numbers of `test_nuclei_sets.py::test_the_summary_prints_the_nuclei_beside_the_outlines`:
squares of 12 px and 6 px, 2 px off). A result made before its case had nuclei has outline
rows and no nuclei row (`nuclei_missing.csv`: "Run it again"); it is not scored at the
unregistered pose.

**What it costs the scorer.** The pipeline's IoU rasterises each pair in a Python loop,
and a row scores its pairs 18 times (two pairings, once registered and in eight null
directions). Measured on one 625-cell generated case on a laptop: 2.5 s per row on the
bare nuclei, and on the grown outlines 25 s unsimplified, about 9 s simplified. Two
things were done about it, neither of which changes a number: the grown outlines are
simplified as the pipeline's are (above), and a pair that both pairings chose is
rasterised once (`pairs._pair_iou`; a pair's IoU depends on its two cells alone;
`test_pairs.py::test_a_pair_both_pairings_share_is_rasterised_once`), which halves the
IoU work on the cohort too.

**Commands on the cluster** (after exporting what "What to export" lists):

```bash
# 0. the input step, then the rounds (launch.sbatch --inputs / --prepare do both)
python3 -m benchmarks.regbench.manifest --preprocessed "$REGBENCH_INPUTS/preprocess/results/csv/preprocessed.csv" \
    --outlines "$REGBENCH_INPUTS/outlines" --out rounds.csv
# 1 + 2. StarDist and the trace of every slide, then prepare: one call
REGBENCH_MANIFEST=$PWD/rounds.csv benchmarks/regbench/slurm/submit_cohort.sh --prepare -- <site options>
#    (by hand: N=$(regbench_py STARE_REGBENCH nuclei --manifest rounds.csv --out "$REGBENCH_NUCLEI" --count)
#     sbatch --array=0-$((N-1)) -c 8 --mem=128G -t 8:00:00 $REGBENCH_SBATCH_GPU benchmarks/regbench/slurm/nuclei.sbatch
#     then: regbench_py STARE_REGBENCH prepare multiplex --cases "$REGBENCH_CASES" --manifest rounds.csv --nuclei-dir "$REGBENCH_NUCLEI")
# afterwards: every slide, its source, whether it is exact, or why it has none
regbench_py STARE_REGBENCH nuclei --manifest rounds.csv --out "$REGBENCH_NUCLEI" --list
grep -h DISAGREEMENT "$REGBENCH_OUT"/logs/rb-nuclei-*.log
```

The array task looks at what is kept for its slide first and runs StarDist only when the
slide has no usable run. `--list` prints `index, slide,
status (usable | needs_rerun | refused | none), source, exact, one_nucleus_frac,
n_nuclei, reason`. The allocation of the array (8 CPUs, 128 GB, 8 h, a GPU) follows the
pipeline's `SEGMENT` block (`conf/modules.config:1062-1067`: 8 CPUs, 32 to 128 GB by file
size, 4 h per attempt) at its upper end; `REGBENCH_NUCLEI_CPUS`, `_MEM`, `_TIME` change it.

**Not verified until the cluster** (none of it was run: StarDist is not installed where
this was written, the site's model is not in the repository):
that the re-run reproduces the published outlines at all (same model, same image, the
site's settings as its config's plain `key = value` lines give them: the `REFUSED` and
`DISAGREEMENT` lines and `exact` will say); that the site overlay is readable that way
(the real file was never read here: a key set in two profiles, or by an expression, is
refused with a message, not guessed); that `bolt3x/mirage-stardist:1.0.0` runs `bin/segment.py` from a bound
checkout under `apptainer exec --cleanenv --nv`; that StarDist is deterministic enough
between two GPU runs for the 0.99 thresholds; what a work directory still holds; the memory and time of tracing
two label images of a whole slide in the scorer's image (`cells.polys_from_label_image`,
tile by tile).

### Records: a case is scored once

Scoring a slide is slow (measured below), so the scorer keeps what it computes, under
`<out>/cache/scores/<dataset>/<case>/`, and no table depends on whether it did.

- **A pair record** is one call of `pairs.record`: one polygon set (outlines or nuclei)
  of one case, in the frame of one source (the unregistered pose, the true map, or one
  variant's result), whole or without the nuclei on one owner's seams. It is nine
  placements (the real one and the eight shifted ones of the null), each with both
  pairings and its own per-pair IoU. `pairs/<key>.rec` holds one, written the moment it
  is computed.
- **A case record** (`case.rec`) is every row of the case, as the tables take them.
  `case.json` beside it is the same key for a person to read, with the seconds it took
  and which seam masks were found.
- **Nothing is computed twice.** Every variant that failed, was killed or never ran is
  scored at the unregistered pose: all of them read the ONE pair record of that pose,
  and one more per seam owner. (Before, each such variant computed its own off-seam
  record: with a run list of dozens of rows, most of a case's time.) The areas and
  centroids of a polygon set are computed once per set, not once per use. No `*_null`
  number is derived from another: each shifted placement has its own centroids,
  pairing and IoU, as before.
- **When a record is reused.** Only when everything it rests on is unchanged, and the
  key is compared whole, never guessed from a file being there.
  A pair record's key: the bytes of the case's `points.npz`, the set, the pixel size,
  the bytes of the `warped.npz` that moved the polygons, the owner's mask itself, the
  pipeline's `cell_pairs.py`; `pairs.py`, `cases.py` and `mapping.py` with every module
  of the package they import (`score.code_files` follows the relative imports); of
  `score.py` the source of the helpers that read a result and put a record's inputs
  together (`score.PAIR_CODE_MEMBERS`: `_load_run` and the seam masks among them) and
  the values of the names its arrays are read under (`score.PAIR_CODE_CONSTANTS`); the
  versions of Python, NumPy, SciPy and scikit-image, the scoring constants (the exact
  assignment's bound among them), and `score.RECORD_VERSION`.
  A case record's key: all of that for every result the case reads (`run.json` AND
  `warped.npz`, by content), plus `case.json`, the list of variants in order, the seam
  owners, each row's selfcheck status, the submission being scored
  (`--submission`), every file of the package's top level and every module `score.py`
  imports inside the package, followed through. So a re-run of a method, a changed
  run list or another submission makes the case's ROWS again, in
  seconds to minutes, and computes only the pair records whose inputs changed: a
  `run.json` that says something else changes rows and no pair record.
  A record that is cut short, damaged or made for other inputs is computed again and
  replaced, and the log says so (`started (made for other inputs: results.stare...)`).
  `score --fresh` reads no record at all.
- **`score-case`** scores one case and writes no table:
  `python -m benchmarks.regbench score-case --cases CASES --out OUT --dataset multiplex
  --index 3 --workers 8` (or `--case-id`). `score` then reads that case's record,
  provided it is given the same `--submission` and `--only-run`; it computes
  every case that has no valid record itself, and saves each as it finishes, so an
  interrupted `score` keeps the cases it finished and a killed `score-case` keeps the
  pair records it finished.
- **`--workers N`** runs at ONE level, never two: whole cases side by side when at
  least N cases are to be computed, else one case after the other with its pair
  records on N processes. Each process reads the case's files itself; what crosses
  between processes is a record's name one way and its numbers the other. `--workers
  1` starts no process. The tables are byte-identical for every N
  (`test_score_records.py`), and were byte-identical to those of the scorer before
  the records existed, on the test cohort.
- **Progress** goes to stderr: each case as it starts (with the reason it had no
  record) and ends, and its pair records as they finish, at most one line per 30 s.

Measured on a laptop (Apple M-series, one generated case: 37,913 reference and 38,096
moving cells over 22,000 x 22,000 px, outlines of about 18 px radius and nuclei of
about 8 px, three results of which one is a seam owner; pair fraction 0.95, null
0.12). The case has 16 pair records (four sources, whole and off the owner's seams, on
two sets).

| `--workers` | one case, seconds | against one worker |
|---|---|---|
| 1 | 912.6 | 1 |
| 4 | 246.4 | 3.7 x |
| 8 | 132.3 | 6.9 x |

One pair record of the outlines took 65 to 121 s alone and one of the nuclei 17 to
22 s; with 8 workers the case took as long as its longest pair record plus the
reading. Where the time goes (profile of one outline record, 146.7 s before the
geometry was shared): 81 % rasterising the pairs for their IoU (`cell_pairs.pair_iou`,
two scikit-image polygons per pair, 0.77 ms each at this size), 17 % areas and
centroids (now computed once per set: that record takes 127.8 s, the nuclei's 25.0 s
instead of 45.6 s), about 2 % pairing. The IoU is the pipeline's code and is not
touched; it is linear in the number of pairs (19.1 s at 4,738 cells, 146.7 s at
37,913). **Extrapolated, not measured:** at 300,000 nuclei per slide, linear in cells
and at this density, an outline record is about 1,000 s and a nuclei record about
200 s on this laptop's cores. The first cohort patient ran two runs (4 sources, one
seam owner: 16 pair records, about 2.7 h of work), so about 25 to 30 minutes per case
on 8 cores; the whole run list (about 36 sources and 2 seam owners: about 216 pair
records, about 36 h of work) is about 4.5 h per case on 8 cores. Real tissue is denser
than this case (more pairs under the null), and the cluster's cores are not this
laptop's: read the first case's `seconds` in `case.json` before sizing the rest. Each
worker holds the case's polygons (hundreds of megabytes at that size) and the result
it is scoring.

### Tables (`<out>/tables/`)

| file | one row per | what |
|---|---|---|
| `cells.csv` | case x variant | `dataset, tier, case_id, group, patient_id` (= `group`), `panel` (the moving slide's channels as `A_B_C`, or `not recorded`), `tool_version` (the run's `info.version`, `not recorded` when it has none, `-` for `initial`; the same three are in `cells_nuclei.csv`, `nonband_cells*.csv` and `resources.csv`), `family, pose, method, frame, selfcheck_status`, every key of `pairs.RECORD_KEYS`, `coverage, nn_distance_um_p50, fold_pct, sd_log_jac, imputed_initial, problem, cause`, and since 2026-10-11 `unit` (the independent unit), `n_unmapped, unmapped_frac`, `excluded_from_summaries`, `set` (`outlines`: this file is the outlines; the nuclei are in `cells_nuclei.csv`), and what the case records of that set: `set_source_ref`, `set_source_mov` (where each side's objects come from), `set_inexact` (either side is declared not to be, object for object, the segmentation it stands for) and `set_shrunk` (either side is the outlines with the 10 px rim taken off, the fallback for a slide without bare nuclei). Among the record keys: the pipeline pairing's own account (`n_components`, `largest_component_cells`, `n_fallback_components`, `n_fallback_cells`, `fallback_frac`, `fallback_frac_null`) and the exact assignment's extra columns (`*_exact`, `exact_status`); then the full LSA (`*_full_lsa`, `full_lsa_status`, `full_lsa_detail`, `full_lsa_seconds`, `full_lsa_largest_component_cells`, `full_lsa_work`: `full_pairs.ROW_KEYS`, Task 21b). Every row scores ALL nuclei |
| `nonband_cells.csv` | case x variant x owner | the same columns for `<variant>:non_<owner>`, plus `variant, owner, n_mov_kept, n_mov_all, n_ref_kept, n_ref_all`. See "Off the seams" |
| `nonband_summary.csv` | dataset x owner x variant | the summary of those rows, with the nuclei kept and `within_ceiling`: what the mask alone lets `within_*um` reach for any method |
| `cells_nuclei.csv`, `summary_cells_nuclei.csv`, `common_block_nuclei.csv`, `paired_vs_primary_nuclei.csv`, `paired_summary_nuclei.csv`, `nonband_cells_nuclei.csv`, `nonband_summary_nuclei.csv` | as the table of the same name | the same columns on the BARE NUCLEI (`set = nuclei`), with their own gate, null and shared subset. Absent when no case has nuclei. See "Outlines and bare nuclei" |
| `nuclei_missing.csv` | case, or case x variant | why there is no nuclei row: `dataset, case_id, group, …, method, reason`. `method` is empty when the CASE has no nuclei on a side (the prepared reason), and names a result that was made before its case had them |
| `nonband_missing.csv` | dataset x owner | `n_cases`, and `n_cases_without_mask`: the cases the owner's primary run did not register, which have no off-seam row for any method |
| `summary_cells.csv` | dataset x family x variant x subset | per unit the median over rounds, then median, min, max over units; `family` (empty on the cohort); `n_patients` (units), `n_cases, n_ok, n_excluded_not_method`; the minimum over cases and `<headline>_n_nan`; each headline beside its null; `dice_matched_shared_med, n_dice_shared` and `displacement_um_p50_shared_med, n_readable_shared` (the cases every compared row qualifies on) beside the per-row `dice_matched_ranked_med, n_dice_ranked`, `displacement_um_p50_med, n_readable`; `unmapped_frac_max`; `n_fallback_cases`, `fallback_frac_max` and the exact assignment's `pair_fraction_exact_med`, `dice_matched_exact_med`, `n_exact` (extra numbers, never ranked); `fallback_frac_med`, `pair_fraction_full_lsa_med`, `dice_matched_full_lsa_med`, `pair_fraction_full_mnn_med`, `dice_matched_full_mnn_med`, `n_full_lsa` (Task 21b: the full pairings, never ranked); `n_inexact_cases`, `n_shrunk_cases` (cases scored on objects marked `set_inexact` / `set_shrunk`). `subset=common` repeats it on the cases every compared row registered. Task 29: `dice_matched_*` zero-fills a case with no pair (the headline), `dice_matched_zero_*` / `dice_matched_skip_*` give both rules; `pair_fraction_maxn_*` and `pair_fraction_sym_*` (with null and excess) beside `pair_fraction_*` |
| `common_block.csv` | dataset x primary row | `in_common_block`, `why_not`, `n_cases_common`, `n_cases` |
| `paired_vs_primary.csv` | unit x variant (x family) | the difference of each headline against the `stare` row, on the cases both have |
| `paired_summary.csv` | dataset x family x variant | the median and the mean over units of each difference with its bootstrap interval over units (`d_<headline>_med, _mean, _lo, _hi`), `n_patients` |
| `excluded_rounds.csv` | lost round | written only when a round was lost for a cause that is not the method's: `dataset, method, case_id, group, cause` |
| `selfcheck_masks.csv` | selfcheck run x case (round) x variant x dot set | the selfcheck's numbers on `all` dots, the `unmasked` dots and the `masked` dots alone (`n_dots, found_frac, err_px_p95, err_px_mean, worst_block_err_px_p95, worst_block_found_frac, passes_accuracy`), `masked_frac, mask_cap, mask_over_cap`, the comparison's `n_detected` and `n_narrow` (what the detector took for a dot in the written image, and how many of those were narrower than 1.2 px), and one `passes_cap_<cap>` column per cap of `selfcheck.MASK_CAPS`. Written when the scorer is given the selfchecks (the launcher's `score`). Every table with a `method` column also carries `selfcheck_mask_over_cap` then |
| `attrition.csv` | dataset x registration row | where the rounds go: `manifest_rounds` (the rounds of the cohort's manifest; blank unless the scorer is given it, `score --manifest`: the launcher's score job gives it `REGBENCH_MANIFEST` when that is set, else `$REGBENCH_CASES/manifest.csv`, the copy `submit_cohort.sh --prepare` leaves beside the cases it prepares), `prepared`, `with_outlines`, `with_nuclei`, then per row `run` (a record that is not `not run`), `registered` (`ok`), `scored` and `scored_nuclei` (the row is its own result: not the unregistered pose, not left out). Also a table of `summary.md` |
| `failures.csv` | variant | launches made, launches failed by cause, rounds lost; `n_failed_method`, `n_failed_benchmark` and `n_not_run_or_lost` (the launcher's and the infrastructure's: `not_run`, `not_submitted`, a lost attempt; the summary's column is `not run / lost`) count CASES. A selfcheck's verdict is in none of them |
| `failure_log.csv` | failed or lost EVENT | always written (a header alone when nothing failed). See "The failure log" |
| `resources.csv` | case x variant | the cost recorded at that variant's yield, with the billing columns of "CPU-hours: one definition for every method" |
| `summary_resources.csv` | patient x variant | the billing columns per patient ("CPU-hours: one definition for every method"), `jvm_heap_asked_gb` and `jvm_max_heap_gb` for a method that starts a JVM; sum over rounds for a pair method, the launch's total for a per-patient one; `write_s` and `total_to_image_s` are the patient's totals for both; peaks are maxima, never divided by rounds; `complete` (every round registered), `launched` |
| `stare_stage_peaks.csv` | case x STARE result | the peak memory (GB) of each stage job of a STARE round, in the attempt the job ENDED OK with (`None`/`NA` when it did not): `coarse_peak_gb`, `tile_peak_gb_p50` and `_max` over the `n_tiles` tiles, `solve_peak_gb`, and STITCH's `stitch_peak_gb` (`run.json: native.stitch_job.peak_rss_mb`); with `tier`, `stride`, and the read box STARE gives an interior tile (`read_box_px` = tile + 2 x 3 x stride, `read_box_mpx`) beside the window the pipeline sizes a tile job for (`pipeline_window_px` = tile + 2 x max(halo, 3 x stride)). Written by the score side (`regbench.stage_peaks`); not in the cohort golden |
| `ashlar_discard.csv` | case x ASHLAR result (one per result name that has a `run.json`) | the tiles ASHLAR's stage model replaced instead of the cycle's own measurement, from `run.json: info`: `discard_frac` (their share of `n_tiles`), its three parts `discard_infinite_error`, `discard_zero_offset`, `discard_over_budget`, `cycle_fully_modelled`; with the cell (`affine`, `noise_frac`, `max_shift_um` = 15, 60 or 240) and, beside it, the case's `rigid_offset_um` (\|M0(c) - c\| x pixel size at the reference centre, M0 = STARE's `info.m0` read by the ASHLAR adapter's own `read_anchor`; the file in `rigid_offset_from`, `NA` when STARE left none). A round without those keys has `NA`, never 0. Written by the score side (`regbench.ashlar_discard`); not in the cohort golden |
| `registration_channels.csv` | case x run (Task 29, WD6) | the channel each run registered on, read from what its `run.json` records (`regbench.reg_channels`; no method's behaviour changes): VALIS's `info.nuclear_channel` (per slide, matched to the case's reference / moving file by name; VALIS picks by a `dapi` regex and falls to channel 0, so a CELLTOX round can register on CELLTOX), STARE's `info.nuclear_index` (both slides), ASHLAR's `native.nuclear_channel` (every cycle), PALOM's and DeeperHistReg's `native.nuclear_channel` (the moving slide; the reference's is not recorded). Beside the case's own nuclear channel (`case.extra`), `registration_channel_is_nuclear` is False where a recorded side differs, True where every recorded side matches, `not recorded` where nothing is |
| `cost_summary.csv` | dataset x variant | the median and the mean over patients of each cost twice: over the patients this row completed (`n_patients_complete` of `n_patients`; `*_med`, `*_mean`) and over the patients every compared row completed (`*_common_med`, `*_common_mean`, `n_patients_common`); CPU-hours first, `device`, `gpu_model`; for a row in several jobs `time_unlimited_parallel_tiles_s_*`, `to_image_unlimited_parallel_tiles_s_*` and `peak_cpus_at_once` in place of `register_s_*` and `total_to_image_s_*` |
| `tissue.csv`, `tissue_summary.csv`, `tissue_missing.csv` | see "Tissue-level agreement of the written images" | ACCREDIT's score and its pieces on each written image. Secondary: in no table above, no paired difference, no ranking |
| `cost_end_to_end.csv` | patient x result that wrote an image | the steps apart, all measured here: `method_convert_s`, `estimate_s`, `write_s`, their sum `measured_here_s`, `benchmark_copy_s`, the peaks, `device`, `gpu_model`, `written`, `merged_image`. Secondary; see "Cost" |
| `cost_end_to_end_summary.csv` | dataset x result | the median over patients of each step and of `measured_here_s`, the largest peaks (`estimate_peak_gb_max`, `job_peak_gb_max`), `n_patients` of `n_patients_cohort`, and the same medians on the common patients (`*_common`, `n_patients_common`) |
| `pipeline_cost.csv` | task of a compute-all trace (backend x patient x Nextflow task) | written only when there is a trace under `<out>/compute`. Nextflow's own figures and three products of them; see "Full pipeline cost, per backend (compute-all)" |
| `pipeline_cost_summary.csv` | backend x patient, and backend (`patient = all`) | counts, completeness, wall time, CPU-h, GB-h, peak, registration share, input size; see the same section |
| `pipeline_cost_by_process.csv` | backend x process | the tasks of the patients in the backend's `all` row, per process; see the same section |
| `launch.csv` | run | written only with `score --verdicts`: what each run's selfcheck and realcheck said, from `logs/verdicts.tsv` (`run, selfcheck, realcheck, reference, log`). Reported, never a gate: no run and no row is left out because of it |
| `excluded_runs.csv` | run left out of the dataset | `run, reason` (`variants.excluded_for`): a run whose method has no input rule for the dataset's modality, or one the dataset does not take. Always written; on the cohort a header alone (nothing is left out of it). On a public dataset the summary lists them too |
| `summary.md` | | the summary: the unit of each dataset and the line "descriptive; no ranking is claimed" (always), the compared rows and who is not among them, the paired differences with their interval, the failure log's section, cost on survivors and on the common patients |

### Keeping old results, freeing the rest (`keep_results.sh`)

One command keeps what the plots and tables need from earlier regbench, arms or sweep
results and frees the images and work folders (`collect`, `plots` and `slim` in order):

```bash
benchmarks/regbench/keep_results.sh --to $ROOT/keep \
    --regbench $ROOT/out --arms $ROOT/arms_results      # a dry run: nothing is deleted
benchmarks/regbench/keep_results.sh --to $ROOT/keep \
    --regbench $ROOT/out --arms $ROOT/arms_results --apply
```

It writes `KEEP/tables/` (the interim collection), `KEEP/figures/` (draft figures from it),
and `KEEP/raw/<root>/`: every `csv tsv json txt yml yaml log html` of each root, tree kept,
with Nextflow `work/` skipped. It checks that every one is there with the same bytes, and
stops before anything is deleted if not. Then it slims each root:
- a regbench root at `images`, so the maps and records stay and it can be scored again;
- an arms or sweep root at `csv`.

`LEVEL=images|csv` sets one level for every root. `--sweep` needs `--sweep-plan`. A
refusal names itself. `KEEP/README.txt` and `KEEP/keep_results.log` record what was kept
and freed.

### Public datasets (`<out>/tables/<dataset>/`)

A public dataset (`regbench.PUBLIC`) is scored on its own, never pooled with the cohort:
`score --dataset mcmicro` (or `--dataset multiplex mcmicro`) writes the cohort's tables into
`<out>/tables/` as always and MCMICRO's into `<out>/tables/mcmicro/`; a table of the cohort never
holds a row of a public dataset. `paper --dataset <name>` reads `<out>/tables/<name>/` and
writes `<out>/paper/<name>/`. Each public dataset is scored with its own run list
(`variants.expected_variants(<name>)`, `variants.primary_for(<name>)`), and goes to the
scorer of its `kind`: `nuclei` (MCMICRO, PhenoCycler, Rashid) to the matched cells,
every table above; `tissue` (no dataset has this kind now) to the tissue-level block alone;
`landmarks` (HyReCo) to its own landmark scorer ("Landmark set", below). A kind
with no scorer would be refused with a message naming the dataset, and nothing written. A
set of unstitched tiles is named `<name>@<front end>` and is looked up by its
base name (`regbench.base_name`). On a brightfield dataset ASHLAR and PALOM have no input
rule and are left out, and so are the `fluo=1` DeeperHistReg rows (their adapter refuses a
brightfield case); `excluded_runs.csv` and the summary say so, run by run.

### The synthetic scaling set (`synthetic`: cost only, `<out>/tables/synthetic/`)

A supplementary set that measures how the COST of STARE and VALIS grows with the size of a
slide. **Accuracy is not measured on it**: no truth map, no field error, no Dice; the
transform each moving panel was made with is kept in its case's `synth_transform.json` for
reproducibility only. It is a public entry (`regbench.PUBLIC["synthetic"]`, kind
`synthetic`), scored for cost alone and never pooled with the cohort.

- **Design** (`synth.py`; user, 2026-10-19): one synthetic patient per size, square,
  4096, 8192, 16384, 32768 and 65536 px (`--grid small` is 4096 alone); 11 panels per
  patient as the cohort has (panel 0 the reference, 1..10 moving), each ONE image of 3
  channels `DAPI`, `S1`, `S2`, uint16 at 0.325 µm. The texture is a cohort slide's
  nuclear channel, a centred crop of up to 16384 px (the source's extent when smaller; the
  plan records which), QUILTED to the size: 2048 px blocks taken at offsets seeded by the
  block's position, blended over a 256 px feathered seam. The quilt is aperiodic and has
  no 180-degree symmetry (a mirror tiling, which an earlier draft used, made the reference
  symmetric and periodic as it grew and sent STARE's coarse step to its fallback from 8192
  px up: the code path would have changed along the size axis), and nuclei keep their size
  at every size. A seeded irregular TISSUE MASK leaves half of every slide inside it
  (`foreground_fraction` 0.5, the same shape scaled to the size; a column of
  `synthetic_cost` and `synthetic_extrapolation`, and each case's measured value in its
  `synth_transform.json`); outside it is the source's background level (1st percentile)
  plus the noise. Moving panel k is the reference, mask included, moved by a rotation of
  up to 1 degree, a translation of up to 2 % of the side and a smooth deformation of 5 µm,
  seeded by (seed, size, k). The slides are single-level tiled OME-TIFFs, like the cohort's
  inputs; the same seed gives byte-identical files.
- **Runs**: STARE's tier x stride cross with no preprocessing (`stare`, `stare_medium`,
  `stare_low`, each at stride 64, 128 and 256) and the three VALIS runs (`valis_mirage`,
  `valis120`, `valis120_hires`). Every other run is in `excluded_runs.csv` with the
  reason. A method that declares `MAX_SIDE_PX` and is given a larger slide is a
  `refused_size` loss (cause `input`), not a failure.
- **Tables** (`<out>/tables/synthetic/`): `synthetic_cost` (one row per run and size),
  `synthetic_scaling` (per run and quantity, the log-log slope on gigapixels with its 95%
  interval, from at least 3 sizes; at 3 sizes the interval is wide) and
  `synthetic_extrapolation` (the fit at the cohort's median and largest patient beside the
  cohort's measured cost, `extrapolated` when outside the fitted range). One repeat. The
  figures are `figS_synth_cost_scaling`, `figS_synth_cross_scaling` and
  `figS_synth_vs_cohort`; `fig2c_cost_vs_input` adds the synthetic peaks as hollow markers
  when the set is there.
- **Disk**: the INPUTS are about 378 GB uncompressed for the default grid, 284 GB of it
  the 65536 px patient (`regbench synth --estimate --to D` prints it; `plan` prints it
  again before it writes anything). On disk the slides are zlib-compressed.
  The RESULTS are far larger: STARE writes every moving panel as an uncompressed,
  single-level round TIFF, 10 per size, so ONE STARE run is 343.3 GB on the default grid
  and the set's nine STARE runs 3089.4 GB (about 3.1 TB), VALIS's results on top. These
  numbers are computed from the sizes in code (`synth.results_estimate`; `regbench synth
  --to D --results-estimate [--runs ...]` prints them), and `submit_cohort.sh --datasets
  synthetic` prints the estimate for the runs it selected beside the free space of
  `REGBENCH_OUT` before its first sbatch (printed only, never a refusal).
- **Caveat**: moving panels carry a cohort-like rigid offset and a mild deformation so
  methods do real work; accuracy is not measured on this set; one repeat.

It is never in `--all` (`--cases-there` says `named only`). Made once, then launched by
name:

```bash
export REGBENCH_CASES_SYNTHETIC=$ROOT/cases_synthetic
# a cohort slide whose nuclear channel is the texture; its realpath must lie under a
# folder the images bind (REGBENCH_INPUTS, REGBENCH_CASES or REGBENCH_BIND), or the launch
# is refused before any sbatch
export REGBENCH_SYNTH_SOURCE=$REGBENCH_INPUTS/<a preprocessed slide>.ome.tif
# the plan job, an array of one task per (patient, panel), then the finish job; a
# complete set is reused, a planned one continued (REGBENCH_SYNTH_GRID=small: 4096 alone)
benchmarks/regbench/slurm/submit_cohort.sh --prepare --datasets synthetic -- <site options>
# when those are done: its own run list, scored for cost alone (no per-case scoring),
# INTO THE COHORT'S REGBENCH_OUT (default $REGBENCH_ROOT/out, here $ROOT/out)
REGBENCH_OUT=$ROOT/out \
    benchmarks/regbench/slurm/submit_cohort.sh --datasets synthetic -- <site options>
# only AFTER the cohort has been scored in that same REGBENCH_OUT: the set's paper tables
regbench_py STARE_REGBENCH paper --dataset synthetic --out "$ROOT/out"
```

**Where and in what order.** The set is scored into the cohort's `REGBENCH_OUT` (the
default `$REGBENCH_ROOT/out`), never a folder of its own: `synthetic_extrapolation` sets
the fit beside the cohort's measured cost, and reads that cost from the same folder
(`paper/cost_points.csv`, else the scorer's `tables/summary_resources.csv`). So `paper
--dataset synthetic` runs AFTER the cohort has been scored there. Run earlier, or in a
folder with no cohort, every extrapolation row says so in its `note` ("cohort not scored in
this REGBENCH_OUT") and has no measured value. The paper's copy (`paper/collect`) takes
`<out>/paper/synthetic/` when it is there and the scorer's `<out>/tables/synthetic/` only
otherwise, whatever order the two were written in.

The array starts only after a plan job that ended ok (`afterok`); the finish job waits
for the array whatever became of it (`afterany`) and names any slide that is missing.
Limits: the plan job `REGBENCH_SYNTH_PLAN_TIME` (4 h), each array task
`REGBENCH_SYNTH_TIME` (4 h, 1 CPU) and `REGBENCH_SYNTH_MEM` (8G). Measured on a
laptop (Apple silicon, one process, 2026-10-19): one 16384 px moving panel takes about
2.7 min (155.7 and 166.7 s, two panels) at a peak of 1.6 GB, so a 65536 px panel (16x the
pixels) is about 45 min: inside the 4 h limit with room for a slower node.

By hand, the same three steps are `regbench synth --source S --to D --plan`, `--unit I`
for every `I` below `--units`, and `--finish` (`--status` says `absent`, `planned` or
`complete`).

**Its CPU cross** runs `stare` and `valis_mirage` at 4 and 8 CPUs on the 4096 and 32768 px
patients, as the cohort's cross does (one folder per count, two repeats). The subset's
cases are symlinks: bind the original folder too. The cross folders (`out_synth_c4`,
`out_synth_c8`) hold no cohort, so the `synthetic_extrapolation` their scorer writes is NOT
the one to use (its rows say "cohort not scored in this REGBENCH_OUT"); the cross's own
table, `cpu_scaling.csv`, goes beside the set's paper tables in the cohort's folder.

```bash
regbench_py STARE_REGBENCH subset-cases --dataset synthetic \
    --cases "$REGBENCH_CASES_SYNTHETIC" --groups s04096,s32768 --to "$ROOT/cases_synthetic_cross"
for c in 4 8; do
    REGBENCH_CPUS=$c REGBENCH_REPEAT=2 REGBENCH_STARE_EXECUTION=local \
    REGBENCH_OUT=$ROOT/out_synth_c$c REGBENCH_CASES_SYNTHETIC=$ROOT/cases_synthetic_cross \
    REGBENCH_BIND=$ROOT/cases_synthetic \
        benchmarks/regbench/slurm/submit_cohort.sh --datasets synthetic \
        --runs "stare valis_mirage" -- <site options>
done
regbench_py STARE_REGBENCH cpu-scaling --dataset synthetic --rows stare,valis_mirage \
    --point 4=$ROOT/out_synth_c4 --point 8=$ROOT/out_synth_c8 \
    --to "$ROOT/out/paper/synthetic/cpu_scaling.csv"
```

### Landmark set (`hyreco`)

HyReCo is scored on its manual landmarks by a scorer of its own (`regbench.landmarks`),
into `<out>/tables/hyreco/` (`score.HYRECO_TABLES`), never into a cohort table. The other
landmark set of Task 21 was removed with its scorer, its download script and its regime
(`bf_consecutive`) by the user's ruling of 2026-10-19: it is in no figure of the
manuscript. Rulings 2026-10-19:

- **HyReCo, the full set and the Additional set** (`regbench.landmarks`): ONE dataset,
  `hyreco`. `prepare hyreco --data-root <dir>` reads subset A (`HyReCo/{HE,PHH3}`) and
  subset B (`HyReCo-Additional/{HE,PHH3}`) into one cases folder, case ids `a_<id>` and
  `b_<id>`; the download script fetches both by default. An id both subsets list is kept
  in both and flagged (`shared_id`, counted as `n_shared_id`); nothing is dropped. The
  summary has three sets, `all_pairs`, `subset_a` and `subset_b`, and the authors'
  published numbers stand beside them (`hyreco_published`), never inside. Errors are in
  micrometres (per-pair median, and one pooled median over all landmarks); each pair row
  also carries the median and maximum rTRE (TRE over the target image's diagonal) and
  robustness; a pair the method failed on counts at its unregistered landmarks.
- **Nothing blocks.** A landmark sanity check that fails (a landmark outside its image,
  two landmark files of different lengths, a pixel size off the published one) is a flag
  on the pair (`flags`, `n_flagged`) and a `landmark_flagged` row of `losses.csv`; the
  pair is scored. A pair prepare could not read, or whose images are not there
  (`--allow-incomplete`), is a row of `prepare_dropped.csv`. A result whose `run.json:
  truth_sha256` is not the case's `truth.npz` as it is now was made for other landmarks:
  it is `not run` on the case (`*_not_run.csv`, a `stale_result` loss), in no statistic.
- **Run list, 50 runs.** STARE runs its whole cross: tier {high, medium, low} x stride
  {64, 128, 256} x the five levels of `bf_same` (`regimes.stare_levels`: `none` (grey,
  STARE unchanged), `haem_clahe`, `grey_clahe`, `haem_percentile` and `haem_none`), 45
  runs. The other methods run their defaults only: `valis_mirage`, `valis120`,
  `valis120_hires`, `deeperhistreg_gpu`, `deeperhistreg_hr_gpu`; every other run is listed
  in `excluded_runs.csv` with the ruling as its reason. The cohort keeps its 56 runs and
  the fluorescence sets theirs; `RUNS` holds 96 (the cohort's 56, `bf_same`'s 12 high-tier
  cells and its 28 medium and low cells). On a public set `runs --order default` puts
  STARE first; on the cohort ASHLAR stays first.
- **The manuscript's HyReCo STARE row** (Fig. 3f) is `stare_haem_none`: haematoxylin from
  colour deconvolution, high tier, stride 64, NO intensity preprocessing (`--opt
  proxy=haem --opt sprep=none`; level `none_haem` of `bf_same`, added by the user's
  ruling of 2026-10-19). The harness has no per-dataset primary row, so `stare` (grey, no
  preprocessing) stays STARE's primary in `summary.md`; the manuscript's row is read by
  its name: the `row` column of `public_landmarks` / `public_landmark_pairs`, and
  `hyreco_summary.csv`'s `method` column.
- **Paper tables** (`paper --dataset hyreco`): `public_landmarks` (one row per set and
  row), `public_landmark_pairs` (one row per pair and row) and `method_best`: each
  family's lowest median over pairs of the per-pair median rTRE, every terminal row of the
  family a candidate (STARE's 45), chosen on these same pairs (`in_sample_selection`
  True). A family with no row scored has no best row (`NA`).

Getting the data (nothing in the repository downloads it for you):

```bash
# HyReCo: the IEEE DataPort links, one `url<TAB>name[<TAB>md5]` line each. By default the
# FULL set is fetched: HyReCo.zip (233.36 GB) AND HyReCo-Additional-HE.zip and -PHH3.zip
# (273.73 + 232.58 GB); the free-space hint is twice their sum, 1479.3 GB.
# HYRECO_ADDITIONAL=0 fetches HyReCo.zip alone (a quick core-only download, 466.7 GB).
# prepare takes subset B whenever it is there
HYRECO_URLS=hyreco_urls.tsv benchmarks/regbench/download/hyreco.sh /data/hyreco
python -m benchmarks.regbench prepare hyreco --data-root /data/hyreco --cases /data/cases_hyreco

REGBENCH_CASES_HYRECO=/data/cases_hyreco benchmarks/regbench/slurm/submit_cohort.sh --datasets hyreco
```

The download is resumable (`curl -C -` into `<name>.part`; an md5 given on the line must
match before the file takes its name), and a failure stops the script, never swallowed.
The committed fixture (`benchmarks/tests/regbench/fixtures/hyreco/`) holds made-up images
and landmarks in the dataset's layout; its `README.md` cites where every format detail
comes from.

### Off the seams

A method that builds its image from tiles or blocks draws some nuclei across a join, and
its point map is not exactly its image there. Such a method (`score.SEAM_OWNERS`: `ashlar`,
`palom`) writes a **seam mask** with its primary row:
one bool per nucleus, from where it put its tiles or blocks and from no score (ASHLAR: the
tile grid and the positions it accepted; PALOM: its block grid and the shifts it accepted).
The mask is therefore NOT fixed before the run: it depends on the owner's own outcome on
that case (which positions and shifts it accepted), though on no other method's and on no
score.
ASHLAR's is on the moving nuclei (`info["band_mov"]`, `band` in `warped.npz`); PALOM's is
on the reference nuclei (`info["band_ref"]`, `band_ref`).

**The mask on the bare nuclei** (`band_nuc`, `band_ref_nuc`; rows in
`nonband_cells_nuclei.csv`). Where the case says which outline each nucleus belongs to
(`nuclei_<side>_outline`: the built-in segmenter and the shrunk
fallback, where the two sets are the same objects by construction) **a nucleus inherits
its outline's flag**, so both sets drop the same objects. Otherwise (the cohort's label
images and re-runs, whose nuclei are not indexed like the published outlines) the mask is
**recomputed from the owner's geometry**: the owner applies its own seam rule to the
nuclei's polygons (`info["band_mov_nuc"]` / `info["band_ref_nuc"]`; ASHLAR: any vertex of
the nucleus on a seam; PALOM: the nucleus's own bounding box, grown by its margin, over
blocks with different shifts). A nucleus is smaller than its outline, so the recomputed
mask marks fewer objects than the outlines' mask does. `run.json: nuclei_band` says which
rule gave the mask (`inherited` | `geometry`).

**Which rows a reference-side mask removes.**

- `nonband_cells.csv`: the marked reference nuclei are taken out of the reference set
  before pairing (`n_ref_kept` of `n_ref_all`); every moving nucleus stays
  (`n_mov_kept == n_mov_all`). A moving nucleus whose partner was removed is then
  unpaired, for every method alike.
The scorer then scores **every** variant of that case a second time on the nuclei the
mask leaves: `stare:non_ashlar`, `valis120:non_ashlar`, `ashlar:non_ashlar`, `initial:
non_ashlar`, ... All rows of one owner use identical nuclei, and each row says how many
(`n_mov_kept` of `n_mov_all`). The gate still comes from all native nuclei. A variant that
failed keeps its row, at the unregistered pose.

- Primary rows always score all nuclei. The off-seam rows live in `nonband_*.csv` and in
  their own block of `summary.md` ("Off the seams"): they are in no primary table, no
  paired difference and no ranking, and an off-seam number is never set beside a
  full-slide number.
- The mask comes from the owner's primary row (`<out>/<owner>/...`) on that case. Where
  that run failed there is no off-seam row for ANY method on the case, and the summary
  prints how many such cases there are (`nonband_missing.csv`).
- The old name `:nonband`, and a second row only for the method that has the band, are
  gone.
- **The mask caps `within_*um` for every row of the block.** The curve is the smaller of
  the two one-sided shares, and a mask that removes nuclei on one side only (ASHLAR's:
  moving nuclei; PALOM's: reference nuclei) leaves at most (fewer kept) / (more kept) of
  the other side with a partner, one partner per nucleus. `nonband_summary.csv:
  within_ceiling` is that ratio per owner, and the block's header prints it: with a
  quarter of the moving nuclei removed no method can exceed about 0.75 there. Pair
  fraction is over the smaller kept set and has no such cap. Read an off-seam
  `within_*um` against its ceiling, never against a
  full-slide number.

## The failure log (`failure_log.csv`): every event, and whose it is

`failures.csv` counts. `failure_log.csv` (ruling R3.17, `failure_log.py`) is what the
counts are made of: **one row per failed or lost event**, written by every `score` (a
header alone when nothing failed).

- every failed attempt of every unit: the attempt a result ended with, and every earlier
  attempt of the memory ladder kept in `run.json: previous_attempts`, even when the unit
  ended `ok` at a higher rung;
- every selfcheck attempt that did not end `ok` (`selfcheck_inconsistent`,
  `selfcheck_not_checked`, `selfcheck_crashed`) and every flagged realcheck
  (`realcheck_flagged`), from `logs/verdicts.tsv`: reported, never a gate, so the run went
  on and its rows are scored (such an event ends no unit and is never a method failure);
- every written image the tissue block could not score because of an error
  (`tissue_missing`), every record the scorer could not parse (`unreadable_record`), and,
  folded into one row per result name, the results nobody ran (`not_run`).

A per-patient launch that fails loses every round at once: one event per result name, the
rounds joined in `case_id`.

Columns: `dataset, run, variant, unit, group, case_id, attempt, rung_gb, retry, submission,
launch_id, stage` (`selfcheck | realcheck | register | writer | benchmark_points |
tissue | score`), `cause, whose, cause_from` (how the cause was decided: the attempt's own
record, the scheduler's state, a verdict), `scheduler_state, exit_code, elapsed_s,
peak_rss_gb, message` (the first line of the error), `log_path` (the scheduler's log of
that task), `climbed_to` (the next rung, "the same rung again", or `-` when the unit ended
there), `final_outcome_of_unit` (`ok at rung N | failed | lost | reported; the run went on`).

**`whose` is total and is one table** (`failure_log.WHOSE`; a test walks every cause the
code can write against it, so a new cause cannot be added unclassified):

| whose | causes | means |
|---|---|---|
| `method` | `exception`, `oom`, `gpu_oom`, `timeout`, `writer`, `writer_oom`, `killed` | the package under test raised, ran out of memory or time, its point mapping or its writer failed, its process died. `killed` is a process that died with nobody saying why; `regbench settle` renames it when the scheduler's accounting names a cause (memory, time, a lost node) or the task's script noted a Python ended by signal 9 |
| `benchmark` | `benchmark_oom`, `benchmark`, `identical_overlap`, `selfcheck_inconsistent`, `selfcheck_not_checked`, `selfcheck_crashed`, `realcheck_flagged`, `tissue_missing`, `unreadable_record` | the benchmark's own step or check. A selfcheck that does not pass means the benchmark cannot show that ITS points follow the method's image; it is not evidence that the method registers badly |
| `infrastructure` | `infrastructure`, `selfcheck_lost` | the scheduler or the node lost the attempt |
| `launcher` | `not_run`, `not_submitted` | the unit was never submitted, a stage job never reached the queue |

`summary.md` has a "Failures: every event, and whose it is" section: the count of events
per run, stage, cause and whose; then every unit whose FINAL outcome is not ok, with its
cause, whose, message and log path; and a bold line when a unit ended for a reason that
is not the method's.

**Where the log path comes from.** Every batch script sources `slurm/env.sh`, which
derives the task's own log name from the scheduler's variables by the launcher's pattern
(`$REGBENCH_OUT/logs/%x-%A_%a.log` for an array task, `%x-%j.log` otherwise) and exports
it as `REGBENCH_LOG`. The driver writes it into `run.json` (`log_path`, and into each
`previous_attempts` entry with that attempt's error text, phase and peak), the selfcheck
into `selfcheck.json` (`log_path`, and per attempt with `why`), and the verdict into
`verdicts.tsv` (its `log` column). It was not checked on a cluster that `%x` equals
`SLURM_JOB_NAME` for these jobs ("Not verified").

**The same words in the logs.** Every batch script ends with one machine-readable line:

```
REGBENCH_END kind=run run=palom dataset=multiplex unit=P03_r2 rung=2 status=failed cause=writer_oom whose=method log=/.../rb-palom-multiplex-r2-812_4.log
```

`kind` is `run`, `selfcheck`, `tissue` or `score`; `status` is `ok`, `failed`
(the method's), `lost` (anything else), a selfcheck status, or
`no_end_line`. The Python step writes the line; the script repeats it as the log's last
line, and writes a fallback itself (`status=no_end_line cause=killed whose=method
exit=<code>`) when the step died before it could. `grep -h REGBENCH_END
$REGBENCH_OUT/logs/*.log` is the same classification as the table. Not recorded:
`exit_code` (the scheduler's accounting is asked for state, elapsed and CPU time only),
and the cost of a selfcheck attempt.

## Where every loss shows up

USER 2026-10-18: "no blocking at all, just report in final csv / results" and "make sure in
outputs i see everything" (Task 28). Nothing that goes wrong with one unit (a case, a run, a
slide, a round of the manifest) stops the others. Every such thing is a ROW of
`tables/losses.csv` (`losses.py`), never only a line of a log; `score` prints the counts
by stage, cause and blame at its end, writes them into `summary.json: losses` and into
`summary.md` ("Where every loss shows up"), and `tables/README.txt` lists every cause.
`losses.csv` is written by every `score`, a header alone when nothing was lost: an absent
table is not "none". Every public dataset (the tissue ones too) has its own
`tables/<dataset>/losses.csv`, `README.txt` and `summary.json`. A row of no dataset (a
dropped run, an unmarked unit, a compute-all patient, a run-level table that cannot be
read) is the RUN's: it is in the cohort's `losses.csv` and in `tables/losses_run.csv`
only, so a union of every dataset's table counts it once. `score` exits non-zero only
when not one case could be scored (its end line says `cause=score_error`), and writes
`losses.csv` first; a dataset with no case is "nothing scored" for itself (its tables
are cleared, `n_scored_cases=0`) and never stops another. The `paper` command writes
`paper/losses.csv` (same columns, stage `paper`) beside its tables.

Columns: `stage,dataset,patient_id,case_id,run,method,cause,blame,effect,detail,source`.
`stage` is one of `submit`, `manifest`, `prepare`, `nuclei`, `run`, `settle`, `score`, `full_pairs`, `paper`; `blame` one of
`method`, `infrastructure`, `input`, `integrity`; `effect` one of
`dropped`, `imputed`, `flagged`, `capped`, `not_run` (`dropped`: in no summary; `imputed`: scored
at the unregistered pose like any failure of the method; `flagged`: kept and scored, with
a flag on its rows; `capped`: the memory ladder stopped at the partition's largest rung,
imputed, the cap in `detail`; `not_run`: nobody ran it). `source` names the file that
carries the evidence (the one that failed: `warped.npz` when that is what could not be
read). **The one fold:** the results nobody ran (`not_run`) are ONE row per (dataset,
method), its `case_id` reading `N case(s)` and `patient_id` blank, since a method nobody
launched would otherwise be a row per case of the cohort; every other loss is a row of
its own.

Every cause, once (`losses.CAUSES`; a test walks it against `failure_log.WHOSE`):

| cause | blame | stage | what it is |
|---|---|---|---|
| `exception` | method | run | the method raised |
| `killed` | method | run | the process died and the scheduler's accounting names no cause |
| `oom` | method | run | out of memory |
| `gpu_oom` | method | run | out of GPU memory |
| `timeout` | method | run | out of time |
| `writer` | method | run | the method's writer failed |
| `writer_oom` | method | run | the method's writer ran out of memory |
| `capped` | method | run | it needed more memory than the largest rung the partition grants: the ladder is the same for every method, so this is a cost result (imputed; the cap is in detail) |
| `benchmark_oom` | infrastructure | run | the benchmark's own point step ran out of memory |
| `benchmark` | infrastructure | run | the benchmark's own step failed |
| `identical_overlap` | infrastructure | run | pixel-identical tile overlaps, made by the benchmark's tiling |
| `infrastructure` | infrastructure | run | the scheduler or the node lost the attempt |
| `not_submitted` | infrastructure | run | the launcher never submitted the unit |
| `killed_unsettled` | infrastructure | settle | killed, and never settled against the scheduler's accounting (no job id, or sacct had none): the cause is not known |
| `watcher_gave_up` | infrastructure | settle | the chain's watcher could not read the queue and gave up before the unit's next step |
| `result_unreadable` | infrastructure | score | the result's run.json or warped.npz could not be read, or an ok result has no warped.npz |
| `submit_refused` | infrastructure | submit | sbatch refused the unit's next step after every retry |
| `held` | infrastructure | run | its jobs were held or stalled in the queue (the squeue reason is in detail) |
| `score_error` | infrastructure | score | the scoring of the case raised (score_errors.csv): the case has no row |
| `table_unreadable` | infrastructure | score | a per-stage table is there and cannot be read: the losses it holds are not rows here (its path is in source) |
| `full_lsa_not_certified` | infrastructure | full_pairs | the full LSA (no fallback, no cap) of a pair record has fewer pairs than the candidates allow: its numbers are NA, the headline row is kept (flagged) |
| `full_lsa_failed` | infrastructure | full_pairs | the full-LSA solve of a pair record raised (memory, anything; the exception is in detail): its numbers are NA, the headline row is kept (flagged) and the next scoring tries again |
| `full_lsa_killed` | infrastructure | full_pairs | the process was killed during the full-LSA solve of a pair record (memory, walltime: it left its in-flight marker and no result): the solve is not tried again, its numbers are NA and the headline row is kept (flagged) |
| `tissue_missing` | infrastructure | score | the tissue block could not score a written image (tissue_missing.csv) |
| `realcheck_unreadable` | infrastructure | score | a realcheck.json (or the run.json beside it) cannot be read: the run's realcheck verdict is `unreadable` (verdicts.tsv) |
| `fit_points_dropped` | infrastructure | paper | a cost fit left points out (no finite target or predictor, or no input size): `n_dropped` of cost_fits / pipeline_cost_fits (S12; `paper/losses.csv`) |
| `paper_record_unreadable` | infrastructure | paper | a run.json the paper tables could not read: `n_unreadable` of versions / run_parameters, one row per run (S13; `paper/losses.csv`) |
| `cost_figure_missing` | infrastructure | paper | a common patient the row completed has no figure for a cost the row has for other patients: left out of that `method_summary` median (`paper/losses.csv`, T31) |
| `cost_peak_not_comparable` | infrastructure | paper | a common patient's peak is a `rusage` lower bound: left out of every peak median of `method_summary` (`paper/losses.csv`) |
| `cost_lower_bound` | infrastructure | paper | a common patient's times and CPU-hours are lower bounds: kept in the medians, flagged (`paper/losses.csv`) |
| `plot_points_dropped` | infrastructure | paper | a cost figure could not draw a patient's value (no figure, `<= 0` on a log axis, a `rusage` peak in a peak comparison, a GPU row without a CPU twin on a CPU axis): `paper/losses.csv` (T31) |
| `plot_lower_bound` | infrastructure | paper | a cost figure drew a value that is a lower bound (`cost_points.lower_bound`): kept, flagged (`paper/losses.csv`) |
| `plot_placeholder` | infrastructure | paper | a cost figure is its placeholder: its input is absent or all NA (`paper/losses.csv`) |
| `stare_local4_differs` | integrity | paper | `stare_local4` scored a case differently from `stare` (`paper/losses.csv`, effect `flagged`) |
| `not_run` | infrastructure | run | no result of this submission: nobody ran it |
| `stale_result` | integrity | run | a result made for an earlier version of the case or by another submission |
| `preflight_failed` | infrastructure | submit | the run's image failed the preflight: the run was not submitted |
| `run_config` | infrastructure | submit | a configuration problem of that run (its image, its GPU request, its stages): the run was not submitted |
| `unmarked_unit` | infrastructure | settle | a chain the settle job could not list: its units were not checked for tasks still in flight |
| `no_trace` | infrastructure | run | a compute-all patient with no Nextflow trace (pipeline_cost_summary: complete_run 0) |
| `compute_incomplete` | infrastructure | run | a compute-all patient whose run is not whole (a failed task no retry recovered, a cached task, no ok end line): out of the backend's `all` row |
| `manifest_dropped` | input | manifest | the manifest could not list the round |
| `prepare_dropped` | input | prepare | prepare dropped the round (multiplex) or the pair (hyreco) |
| `prepare_failed` | input | prepare | prepare failed on the round (multiplex) or the pair (hyreco) |
| `landmark_flagged` | input | prepare | a landmark sanity check of a public landmark set failed (a landmark outside the image, a diagonal the table disagrees with, unequal landmark files, a pixel size off the published one): the pair is kept, scored and flagged |
| `frame_check_failed` | input | run | a STARE proxy moved image content in prep's frame check: written, run and flagged (`stages/prep/record.json` `frame_check=failed`; B10, user ruling "no blocking") |
| `low_agreement` | input | nuclei | the slide's nuclei agree with its outlines under the threshold: kept, scored and flagged (nuclei_checks.csv) |
| `pixel_size_fallback` | input | manifest | the checkpoint has no pixel size for the round: 0.325 um is used |
| `refused_size` | input | run | the run's method declares a largest slide side (`MAX_SIDE_PX`) under this slide's: it refused the size by its own limit (effect `not_run`; the synthetic scaling set) |

`killed_unsettled`, `watcher_gave_up`, `result_unreadable`, `submit_refused`, `held` and
`score_error` are not the method's (`score.NOT_THE_METHOD`): those rounds are in no
summary of the method. `capped` IS the method's (the ladder is the same for every method,
so needing more than its top is a cost result). **Billing (R26):** a `killed_unsettled`
attempt is billed as WASTED, not lost, at the scheduler's upper bound (elapsed x allocated
CPUs and memory from `logs/accounting.psv` when it is there), and its `resources.csv` row
says `cost_bound=upper`.

**The per-stage tables** (`losses.csv` is their union; they still exist):

| table | where | what |
|---|---|---|
| `submit_dropped.csv` | results folder | the runs the launcher dropped (`preflight_failed`, `run_config`: a GPU run with no GPU request, an image with no path, a staged run it cannot size); the others were submitted |
| `manifest_dropped.csv` | beside the manifest (copied to the cases folder by `--prepare`) | every round the manifest could not list, with `n_rounds` |
| `prepare_dropped.csv` | cases folder | every round `prepare multiplex` dropped or failed on (per-round files under `_cache/prepare_dropped/`, their union beside them) |
| `nuclei_checks.csv` | tables (and the nuclei folder, from `nuclei --list`) | one row per slide: its one-nucleus fraction against its outlines, the threshold, `low_agreement` |
| `score_errors.csv` | tables | a case whose scoring raised: the last frame and the message; the other cases are scored |
| `logs/unmarked_units.csv` | results folder | the units the settle job could not list (an unreadable run file, a chain beyond the case list); `chains.mark_in_flight` returns their number as `n_unmarked_units` |
| `logs/refused_units.tsv` | results folder | the units of a refused (or given-up) selfcheck or stage step, which no record can carry (`retry.sbatch` `mark_cut`): a row each, of its cause. A watcher that gives up writes the units of the step it was at down itself (`watcher_gave_up`) before it exits, so a failed settle job does not leave them the method's |
| `logs/compute/plan.tsv` | results folder | the compute-all plan the launcher submitted: a planned patient that left no folder is a `no_trace` row (so is one with an end line and no folder) |
| `tables/losses_run.csv` | tables | the run-level rows (dataset blank) alone |
| `paper/losses.csv` | paper | the paper stage's rows: the tables' (`fit_points_dropped`, `paper_record_unreadable`, `stare_local4_differs`, `cost_figure_missing`, `cost_peak_not_comparable`, `cost_lower_bound`) and the cost figures' (`plot_points_dropped`, `plot_lower_bound`, `plot_placeholder`); each writer replaces only its own rows |
| `pipeline_cost_summary.csv` | tables | `reason` and `n_tasks_unmeasured` per compute-all patient (`no_trace`, `failed_task`, `cached`, `no_ok_end`) |
| `summary.json` | tables | `losses`: the counts by stage, cause, blame and effect |

Beside them, nothing is skipped silently any more: `resources.csv: unmeasured_records`
(stage records that could not be read), `*_n_nan` columns beside every averaged number,
`cost_fits`/`pipeline_cost_fits: n_dropped` (and "k dropped" in the figures),
`versions`/`run_parameters: n_unreadable`, a realcheck verdict `unreadable`, and
`pixel_size_source` and `low_agreement` on every scored row of a case.

**Two common blocks (B2).** A round lost to a cause that is not the method's
(`score.NOT_THE_METHOD`) leaves the strict common block for EVERY method (the headline:
`cost_summary.csv`, `cost_end_to_end_summary.csv`, `paper/method_best.csv`, column
`common_block=strict`). Beside each is a `per_method` variant (`cost_summary_per_method.csv`,
`cost_end_to_end_summary_per_method.csv`, `paper/method_best_per_method.csv`,
`common_block=per_method`) where such a round leaves only the block of the method it was
lost to; `n_lost_not_method` in `summary_resources` counts those rounds.

**The integrity refusals that remain** (the ONLY hard stops; each names itself in its
message): a second submission into one results root; a reused run id; a slimmed or
truncated cells csv (`refuse_slimmed`); the launcher's "the queue cannot be read
(squeue)", only after 5 tries 10 s apart (`REGBENCH_SQUEUE_TRIES`,
`REGBENCH_SQUEUE_WAIT_S`); the scorer's image with no path, or failing its preflight
("integrity refusal (the scorer's image)": nothing could be scored, so nothing is
submitted). A `truth_sha256` mismatch is no crash: the result is
`stale_result`, `not_run`. A launch in which the launcher dropped EVERY run stops, since
nothing is left to submit.

**Selfchecks and realchecks never gate**: they run and report (`verdicts.tsv`,
`failure_log.csv`), and scoring waits only for the registration chains. The scorer
(`rb-score`) runs once they end; `rb-rescore` runs after the selfchecks end (afterany),
SETTLES again (a kill whose accounting arrived since) and then scores again, so their
verdicts reach the tables without anything waiting on them; it reads every case's record
(the selfcheck status is set at table time, not in the case key). With `--compute-all`,
the first compute head of each lane waits for `rb-score` AND `rb-rescore` (compute runs
last and alone), and `rb-score-compute` for every head and `rb-rescore`. A failed frame
check of a STARE proxy is no longer a refusal (B10): it is a `frame_check_failed` row.

## Tissue-level agreement of the written images (secondary)

**ACCREDIT's composite (Zhou et al. 2026, bioRxiv, released code), designed for H&E
references; weights, ranges and thresholds not calibrated on this data; secondary, not
part of any ranking.** That sentence is printed with every table of this block.

For each variant that has an image on disk, the nuclear channel of that image is compared
with the nuclear channel of the reference slide at 4000 px (`tissue.py`). It is the only
part of the benchmark that looks at the written image as a whole: a point map can be
right where the image is blank, cropped or filled. It is computed here between two
fluorescence images of the same channel, which is **outside what the score was designed
and reported for** (every use in its source has an H&E reference). Not claimed: accuracy
at nuclear scale (one evaluation pixel is about 4 µm on a 50000 px slide), or that the
composite tracks registration error (the QCS report of this port, whose author read the
whole preprint, found no check of it against landmarks; the preprint itself calls it "a
proxy metric").

### The definition: the released code, and where the preprint's text differs

Implemented from github.com/LeeZhou-bearway/ACCREDIT at commit `9293473` (MIT licence;
re-implemented, nothing vendored): `benchmark/evaluation/compute_qcs_4methods.py`
`[FULL]` (the script that scored the other methods' images), `quality/quality_metrics.py`
`[FULL]`, `benchmark/evaluation/qcs_formula_v2.py` `[FULL]`,
`benchmark/evaluation/recompute_qcs_v2_from_existing.py` `[FULL]`; the header and all 96
rows of `benchmark/evaluation/metrics_flat_all_4000px.csv` `[FULL]` were parsed and the
composite recomputed from its pieces (91 numeric rows, largest difference 0.00015, the
table's rounding). The preprint (doi 10.64898/2026.08.08.743602, v1) was read for its
"Composite Quality Score (QCS)" section and its limitations paragraph only `[PARTIAL: read
those two passages from the PDF's extracted text, equations partly garbled; missing every
other section and the supplement]`. Term by term, with the file and lines, in
`tissue.py`'s docstring. Where the two disagree the code is implemented (ruling R3.3):

| term | preprint, Methods text | released code (implemented) |
|---|---|---|
| boundary Dice | Dice of two bands, `B_r(M) = dilate(dM, r)`, r = 8 | band = dilate minus erode (17 px ellipse); Dice of the two MASKS restricted to the union of the two bands (`quality_metrics.py:34-54`) |
| Hausdorff | the symmetric maximum: "the worst boundary mismatch, not the average mismatch" | the MEAN of the pooled nearest-neighbour distances between the single longest external contour of each mask, each cut to 2000 evenly taken points (`:86-119`, `:260`). A detached piece of tissue is not seen at all |
| coverage | `c_ref = \|M & M_r\| / \|M_r\|`, penalised under 20 % of the reference tissue | written-image mask / ALL pixels of the frame (`compute_qcs_4methods.py:99`). A tissue that fills 5 % of the frame is scaled by 0.25 even when it matches exactly |
| background | `\|M & not M_r\| / \|M\|` | `\|M & not M_r\| / \|not M_r\|`, the reference's background (`:100`) |
| where the pieces are computed | not stated | inside the bounding box of the written image's mask (`:87-92`); only the two penalties use the frame. Reference tissue outside that box is not seen: an image holding one of two identical pieces of tissue has a tissue Dice of 1 |
| NMI | 64-bin joint histogram | the same, on the pixels above 0 in BOTH 8-bit images only (`quality_metrics.py:238`) |

The released code also disagrees with itself: ACCREDIT's own pipelines score their output
through `compute_quality_metrics_v2` (`quality_metrics.py:288-296`: no bounding-box crop)
with a mask that has one more dilation and no component filter
(`pipelines/dapi/run_dapi_ROI_flip_pipeline.py:74-79` `[PARTIAL: read lines 70-82]`). The
script implemented here is the one the repository ships for other methods' images. The
script that scored the VALIS and ACCREDIT rows of the published table is not in the
repository (`compute_qcs_4methods.py:2`).

### What the benchmark does around it

- **Reference:** the nuclear channel of the reference slide, not an inverted H&E.
- **Evaluation size:** the longest edge of the REFERENCE reduced to 4000 px (never
  enlarged), by an exact area mean in row bands; only this size (ruling R3.3).
- **8-bit:** both planes go through the released `norm8` (1st to 99th percentile of the
  positive pixels). In the released script that function is defined and never called,
  because its inputs were 8-bit already.
- **The written image as it is:** read from the method's own file at full resolution
  (`native.files[0]`, or `nuclear_file`), channel `nuclear_channel` (else the moving
  slide's nuclear index). The released script enlarged 2000 px PNGs, and for one method
  applied an affine itself.
- **Fill:** pixels outside the writer's declared `valid_box` are set to 0 before the
  reduction. The released `== 255 -> 0` rule for one method is not copied.
- **Off the reference grid:** an image with `origin_xy` (ASHLAR's mosaic only) is placed
  on the reference canvas first; what the image does not cover is 0. ASHLAR is therefore
  scored against the benchmark's reference slide, like every other method.
- **A blank image** (under 100 mask pixels) is a row with `status: no_tissue` and a
  composite of 0; the released script stores no score there.
- **Frozen before any cohort run** (`tissue.FROZEN`): 4000 px, r = 8, 64 bins, ORB 5000
  features, RANSAC 5.0 px, the mask recipe (Otsu, 15 px closing twice, components under
  1e-4 of the frame dropped), 2000 contour points, the weights and ranges.

### Rows and tables

| row | image |
|---|---|
| every variant with a written image | the method's own file |
| `initial` | the moving slide's nuclear channel placed on the reference canvas by the identity, drawn by the benchmark: for scale |

A variant with no image has NO row and one line in `tissue_missing.csv` with the reason:
`not run`, `the run failed (<cause>): no image`, `the method writes no image for this
variant` (`stare_rigid`, `palom_affine`, the VALIS rigid and non-rigid stages), `the
method's writer failed: ...`. A failed run is NOT scored at
the unregistered pose here, so `n_scored` of `n_cases` is printed beside every median.

| file | one row per | columns |
|---|---|---|
| `tissue.csv` | case x variant | `dataset, case_id, group, family, pose, variant, source` (`method` or `benchmark`), `status` (`ok` or `no_tissue`), `qcs`, `tissue_dice, boundary_dice, hausdorff_mean, hausdorff_p95` (evaluation px), `hausdorff_mean_um, hausdorff_p95_um`, `inlier_ratio, n_inliers, n_matches, nmi, ncc` (stored by the source, not in the composite), `coverage, bg_coverage`, the bounded terms `s_td, s_bd, s_hd, s_ir, s_nmi, cov_factor, bg_penalty`, `eval_h, eval_w, eval_um_per_px, ref_tissue_frac, mov_tissue_frac, ref_otsu, mov_otsu, crop_box, opencv, definition` |
| `tissue_summary.csv` | dataset x variant | `source, n_scored, n_cases, n_no_tissue, n_below_025`, and for each piece `<piece>_median` and `<piece>_worst` over the scored CASES (minimum; maximum for the two distances) |
| `tissue_missing.csv` | case x variant without a row | `dataset, case_id, group, variant, reason` |

The numeric composite and the count under 0.25 are printed; the source's good /
acceptable / poor classes are not. A 50000 px slide at 4000 px is 4.06 µm per
evaluation pixel.

### Where it runs, and what it needs

1. **The reduction** runs in the driver, right after a method's writer, clock stopped
   (`run.json: tissue`; `<variant dir>/tissue/moving_eval.tif`, 16-bit). NumPy and
   tifffile through `stare.slide_io.open_lazy` only, which every method image already
   needs. One channel is read `tissue.BAND_ROWS` rows at a time; the image is never
   loaded whole. A selfcheck run is not reduced. A plane that is missing is made by the
   scorer from the written image when that still exists. Measured once, on a laptop
   (macOS, tifffile 2023.4.12, zarr 2.18.3), never on the cluster: one channel of a
   three-channel 40000 x 50000 16-bit tiled OME-TIFF reduced in 16 s at a peak of 1.5 GB,
   most of it the reader's (a bare loop of band reads of that file reaches 0.9 to
   1.2 GB; one channel whole would be 3.7 GB); the pieces on the two 3200 x 4000 planes
   took 2 s at a peak of 0.9 GB.
2. **The pieces** need OpenCV (`cv2`). `tissue` and `score` run in the benchmark's own
   STARE image (`containers/stare-regbench`, `variants.SCORER_IMAGE`), which has the
   pinned `opencv-python-headless 4.10.0.82` of `requirements/constraints.txt` beside the
   same pinned STARE; the pipeline's STARE image (`containers/stare`) has no OpenCV and is
   not used by the benchmark.
   `python -m benchmarks.regbench tissue --cases C --out O` computes them in any
   environment that has `cv2`, NumPy, tifffile, zarr and `stare.slide_io`, and leaves
   `tissue.json` beside each reduced plane (`<out>/cache/tissue/...` for the reference
   and `initial`). `score` reads those; what it finds neither on disk nor
   computable is listed in `tissue_missing.csv` with the reason, never left out silently.
   Use ONE OpenCV version for a whole result: the version is recorded in each row
   (`opencv`), and ORB and RANSAC are OpenCV's. That is why the pieces are computed in
   one image, whatever OpenCV a method's own image carries (upstream's VALIS 1.2.0 image
   has 4.9.0.80; the PALOM image and CI have the pinned 4.10.0.82).
3. The benchmark's own planes under `<out>/cache/tissue` (the reference and `initial`)
   are kept, each with a record of what it was made from (`<plane>.json`: the
   image file's size, time and sampled bytes, the channel, the reference grid, the
   evaluation size, the sha256 of `tissue.py`). A plane is read again only while that
   record is unchanged; after a case is prepared again with another image, or after
   `tissue.py` changes, it is made again, and every `tissue.json` holds the same code
   hash and is computed again with it. Nothing has to be deleted by hand.

### What to expect of the pieces (not measured on real slides)

- The **inlier ratio** fits its own similarity, so it is blind to any shift, rotation or
  scale left in the image: the unregistered `initial` row saturates the term (tested).
- **Tissue Dice, coverage and background** see gross failure and missing or misplaced
  tissue, not micrometres. **Boundary Dice** sees only the outline of the tissue.
- **NMI** is self-consistency on the registration channel, like the nuclear scores.
- The composite is `1.0` for a perfect match only when the tissue fills at least 20 % of
  the frame.

## Cost

Per round and per patient, in this order: CPU-hours used and reserved by the
registration, their ratio, GPU-hours with the device, memory-hours, then the times
(registration, writer, to the registered image) and the peak memory. No single column
is "the" cost.

### The paper's accuracy: poolings, sets, Dice rules (`paper/tables.py`, T29)

Every paper accuracy table (`method_summary`, `valis_stages`, `stare_tiers`) carries a
`set` column (`outlines`, the headline, and `nuclei`, the bare nuclei) and a `pooling`
column, one row per (set, pooling):

| `pooling` | the median of | over |
|---|---|---|
| `common` (**headline**) | the per-patient medians | the cases every compared row of the dataset has (and, for the displacement, has readable) |
| `patient` | the per-patient medians | every case of the row |
| `case` | the cases | every case of the row |

`n_units` counts what the pooling is over. Dice is printed both ways: `dice_zero_med`
(no matched pair counts as 0: the headline `dice_med`) and `dice_skip_med` (finite
values only), with `n_dice_zeroed` beside. The three pair fractions
(`pair_fraction_med`, `pair_fraction_maxn_med`, `pair_fraction_sym_med`) are side by side.
`disp_um_common_*` is the displacement on the common readable cases (WD11), beside
`disp_um_*` on the row's own. `method_best` has one row per family per set, with
`unit = patient`, `n_candidates` and `in_sample_selection`: **the best row is an
in-sample ceiling over `n_candidates` rows**, chosen on this cohort with this metric.
Every other table and figure reads the outlines' best row; Fig. 3b's nuclei panel
(`fig3b_dice_best_nuclei`) reads the nuclei's. The paired differences against STARE are
per patient (`paired_unit`), a tie counted as STARE at least as good. DeeperHistReg's
primary is its best row ("primary = best", user 2026-10-18; flagged in-sample);
ASHLAR's primary is unchanged.

**What ran, with which settings (T29).** `run_parameters` names each run's settings
as its method module resolves them (`tables.module_parameters`: the module's own
constants and resolvers, imported read-only, never a list restated here): STARE's tier,
stride, tile/halo/out_tile/max_dim and `sprep`; VALIS's `ARGS` at the run's tier and
micro depth, with the tier's sizes from `bin/utils/valis_config.py` `MEMORY_PRESETS`
(read from its source: importing it builds VALIS's matchers); PALOM's thumbnail size and
keypoints; ASHLAR's max shift, noise and affine prior; DeeperHistReg's preset, device and
`fluo`. VALIS 1.2.0's processing, non-rigid and micro sizes read `not recorded`, with
why: they are upstream's `Valis()` / `register_micro()` defaults (no constant of
`methods.valis`; valis is not importable score-side), and the hires micro size is
computed per patient at run time (`MICRO_REG_FRACTION` x the smallest slide's longest
side) and not recorded. The recorded or optional keys they do not name follow. `versions` gives the image
tag the launcher pins (`slurm/launch.sbatch` `IMAGES`), and the image digest, the Python
and the StarDist version as far as a record gives them: **no result records any of the
three today**, so they read `not recorded`. A run of a dataset's `excluded_runs.csv` is a
row of NA with its reason (`excluded_reason`): in `method_summary` one per (set,
pooling) slot (`which = excluded`), in `valis_stages` / `stare_tiers` one per slot for a
VALIS / STARE run (with the settings its module gives it; no figure draws them), in
`cohort_counts` one per set, in `run_parameters` the run's module settings, in
`versions` and, on a landmark dataset, in `public_landmarks`. `method_best` names the
family's left-out runs (`excluded_runs`) on its rows of each set, and a family with no
row in a set is a row of NA there.

### The paper's cost: the rules every number obeys (`paper/costs.py`)

The paper tables (`method_summary`, `cost_points`, `cost_fits`, `end_to_end`) and the
cost figures (Fig. 3e, 3f, 2c and the supplementary CPU cross and end-to-end panels)
read the tables above by six rules, the same for every method (user ruling
2026-10-18, "fair and sound for all methods"):

1. **One unit: the patient.** Every method has one point per patient. A pair method
   (STARE, DeeperHistReg, PALOM) is the patient's sums from `summary_resources.csv`
   (CPU-hours, GB-hours, GPU-hours, wall time); its peak is the largest of its rounds.
   A per-patient method is its launch. Per-round points are drawn only in Fig. 2c,
   where every method is per round (a per-patient method only where its launch timed
   each round, `register_s_per_round_from = info`), and the per-patient facet beside it
   holds every method. `cost_fits` are per patient for every method.
2. **Common patients are the headline; survivors are second.** The common set is
   computed once per dataset, over every row the paper compares (each family's primary
   row and its best row), by the scorer's own rule (`score.common_patients`, strict).
   Every cost column of `method_summary` is `<stem>_common_med` (the headline) with
   `<stem>_survivors_med` beside it, and every row carries `n_patients_common`,
   `n_patients_survivors`, `n_capped`, `n_failed` and `n_excluded`. The figures draw
   the common set; each tick reads `n=<common>/<all> (capped c, failed f)`.
3. **Rows.** A family's cost is its PRIMARY row (drawn filled); its best-Dice row is
   drawn hollow and labelled "best (ceiling)". `method_summary.row_kind` says which
   (`primary`, `best`, `local_4cpu`, `cpu_twin`); `which` is the old name of the same
   column, kept for one release (deprecated).
4. **Clock and unit are columns.** Every cost stem has `<stem>_clock` and
   `<stem>_unit`; `cost_points` has `clock` and `unit`. `wall_kind` is generalised into
   `clock` (deprecated):

   | `clock` | what the time is | rows |
   |---|---|---|
   | `whole_process` | the registration in one process, in one job of 4 CPUs (`register_s`) | every method but STARE's stage jobs |
   | `local_4cpu` | STARE's stages inside ONE job of 4 CPUs, its tiles 4 at a time (`stare_local4`) | `stare_local4` |
   | `critical_path_unlimited` | STARE's stage jobs: COARSE + the longest tile + SOLVE, every tile at once, no queue (`wall_unlimited_parallel_s`) | STARE's staged rows |
   | `with_queue` | STARE's stage jobs from the first COARSE start to the end of SOLVE, queue waits in (`elapsed_with_queue_s`) | STARE's staged rows |

   `unit` is `patient` everywhere but Fig. 2c's per-round facet (`round`). Every axis
   label names its clock, and every wall time was measured on SHARED nodes (the
   `contention` header line): the headline is CPU-hours.
5. **Uncertainty.** Every median has the bootstrap 95% interval over patients of
   `score.bootstrap_interval`, the accuracy tables' own (2000 resamples, seed
   20261011): `<stem>_<common|survivors>_ci_low` / `_ci_high` in the tables, a bar
   beside each median in the figures.
6. **Nothing silent.** A patient without a figure is counted
   (`n_<stem>_<set>_missing`); a lower bound is flagged (`cost_points.lower_bound`:
   unmeasured stage attempts or records, a polled CPU counter, a wasted attempt nobody
   billed) and counted (`n_lower_bound`); a peak read by `rusage` alone is flagged
   (`peak_rss_comparable = False`), counted (`n_peak_not_comparable`) and in no peak
   comparison; a round that registered and was scored as an imputed failure is costed
   (it used the resources) and counted (`n_imputed_costed`). Each of those patients of
   `method_summary` is also a row (`cost_figure_missing`, `cost_peak_not_comparable`,
   `cost_lower_bound`). Every point a figure could
   not draw (no figure, or `<= 0` on a log axis), every lower-bound point it drew and
   every placeholder is a row of `paper/losses.csv` (stage `paper`).

**`stare_local4`: the wall time at 4 CPUs.** A cost row of STARE's: the `stare` run with
`--stages local` and `workers=4`, every case of the cohort in ONE job of 4 CPUs on the
ordinary memory ladder, under the same `resources.Monitor` as every one-process method.
It is scored like every row, and its accuracy must equal `stare`'s: the paper tables
compare the two case by case and a mismatch is a loss row (`stare_local4_differs`,
blame `integrity`, effect `flagged`). It is never a best-row candidate. The launcher
submits it by default (its own reuse key: no other row's moves); it needs
`REGBENCH_CPUS=4` and is dropped from a launch at another count. Fig. 3f's wall panel is
every method at 4 CPUs with STARE as `stare_local4`; STARE's staged critical path and
its time with the queue are two extra markers, each labelled, never on that axis
without their label.

**Memory one job needs, and the cluster-wide demand.** `fold_peak_vs_stare` is gone (it
divided a sampled tree peak by a single stage job's high-water mark). In its place:
`fold_peak_vs_stare_local4` (a one-process peak over `stare_local4`'s, the same Monitor,
both over the SAME patients: common, complete for both, a comparable peak,
`n_patients_fold`; NaN with a `fold_note` for STARE's stage-job row) and, for STARE's stage jobs, `peak_largest_job_gb` (the largest
single attempt: what a node must give one job) and `peak_concurrent_gb` (the most the
jobs of one round held at once: at every moment the sum of the peaks of the attempts
running then, as `peak_cpus_at_once` is computed). Fig. 3e shows the first, with the
second as a labelled marker.

**Used against reserved.** `gb_h_used` is peak x duration per attempt (the attempt's own
peak: a one-process row's `peak_rss_gb`, a stage attempt's `peak_rss_mb`), beside
`gb_h_reserved`, which measures the REQUEST policy: STARE's stage jobs at the pipeline's
per-process requests (COARSE 2 GB, tile 4 GB, SOLVE 1 GB, x attempt), every other method
on the ladder's 16 GB floor. `cpu_h_reserved_scheduler` and `gb_h_reserved_scheduler`
are "what the cluster held": sacct's `AllocCPUS x ElapsedRaw` and `ReqMem x ElapsedRaw`
over every job of the row's good attempts (STARE: its good stage jobs AND the round's
driver job; the others: the launch), blank and counted without sacct. Fig. 3f draws
CPU-hours reserved as the bar with used inside it and the total (good + wasted) as a
marker, and GB-hours as used with the wasted request stacked on it.

**The same work, stated.** `method_summary` gives `convert_s_*` (the method's own
conversion outside its clock), `convert_inside_clock` (`yes`: inside the registration
clock, `no`: timed apart, `none`: the method converts nothing), `working_px` and
`channels_read`; an `affine_ashlar` row gives `prior_cpu_h` (the anchor row's cost up
to the stage it reads: STARE's COARSE attempts, or VALIS's `_rigid` row) and
`with_prior_cpu_h`. `end_to_end` and `figS_end_to_end` stack conversion, registration,
writer and the prior per row.

### CPU-hours: one definition for every method (`billing.py`)

The scope is the REGISTRATION for every method: what the registration clock covers
("What each method's clock includes"). The method's own writer has a pair of its own.
The benchmark's own work (moving the nuclei through the map, the tissue reduction,
scoring) is in neither.

An **attempt** is one try at a registration or at a part of one, with the CPUs and the
memory it was given, its duration and the CPU time its processes used. For a method
that registers in one process an attempt is the job of one rung of the memory ladder;
for a registration in several scheduler jobs (STARE) it is one attempt of one stage
job. An attempt is *good* when its result is part of the registration the row reports,
and *wasted* otherwise: a rung that ran out of memory, a retried tile, the last attempt
of a registration that failed. A writer that fails after a good registration does not
make the registration a wasted attempt.

| column | definition | in |
|---|---|---|
| `cpu_h_used` | user + system CPU time of the good attempts' processes, summed over every job of the row. The first cost column of `summary.md` | all three tables |
| `cpu_h_reserved` | allocated CPUs x duration, summed over the good attempts. One process: the job's CPUs x `register_s`. Several jobs: each stage attempt's CPUs x its own duration | all three |
| `cpu_efficiency` | `cpu_h_used / cpu_h_reserved` (per patient: of the patient's sums) | all three |
| `gb_h_reserved` | requested memory (GB) x duration over the same attempts. One process: the rung's memory x `register_s`. Several jobs: each stage attempt's own memory x its duration, read from the stage jobs' records when the tables are written (blank in a row whose records are gone or are another attempt's: `billing_note`) | all three |
| `wasted_cpu_h_used`, `wasted_cpu_h_reserved`, `wasted_gb_h_reserved` | the same three over the wasted attempts. An attempt the scheduler killed before it could bill itself is billed by the scheduler's elapsed and CPU time of the whole task (`wasted_from_scheduler`, an upper bound); one nobody recorded is in no sum (`wasted_unknown`), and a sum whose attempts all lack the figure is blank, not 0 | `resources.csv`, `summary_resources.csv` |
| `total_cpu_h_used`, `total_cpu_h_reserved`, `total_gb_h_reserved` | good + wasted. Blank when a good attempt lacks the figure (never the wasted part alone) | all three |
| `lost_cpu_h_used`, `lost_cpu_h_reserved`, `lost_gb_h_reserved`, `lost_attempts` | the same three over the attempts lost to causes that are not the method's (`resources.NOT_THE_METHOD`: a node that failed, a cancelled job, the benchmark's own step), and how many they are | `resources.csv`, `summary_resources.csv` |
| `write_cpu_h_reserved`, `write_cpu_h_used` | the method's own writer, the attempt that wrote the image: the CPUs of the job it ran in x `write_s` (STARE: STITCH's own 4 CPUs x its duration), and its measured CPU time: STITCH's own process for STARE; for a writer inside the row's own job, what the run's CPU counter advanced by while it ran (`run.json: write_cpu_s`; `billing_note` says when that stretch holds more than the writer). Blank for a row that failed | `resources.csv`, `summary_resources.csv` |
| `write_wasted_cpu_h_reserved` | writer attempts that failed, where they are recorded (STITCH's retries; blank elsewhere: there a failed writer fails the row) | the same two |
| `gpu_h`, `device`, `device_name`, `gpu_model` | GPU-hours are a column of their own (registration time x GPUs). `device` is `gpu` for a row whose registration had a GPU and `cpu` otherwise (`device_name` is what the run recorded, `cuda:0`). The `device` column stands beside the CPU columns in `summary.md`: a `gpu` row's CPU-hours are never read without it | all three |
| `moving_gpx`, `cpu_h_used_per_gpx` | gigapixels of the round's moving image (one plane: height x width), and CPU-hours used per gigapixel. Per round for a method that runs per pair; a per-patient launch has one clock for all its rounds, so there it is per patient only: the launch's CPU-hours over the moving images of the rounds that registered | all three |
| `cpu_h_scheduler`, `scheduler_jobs`, `cpu_h_used_vs_scheduler` | the scheduler's own CPU time (`TotalCPU`) of the jobs that ran the good attempts, how many jobs that is, and `cpu_h_used` over it. A cross-check, below | all three |

`cpu_h`, `cpu_h_allocated` (STARE rows), `write_cpu_h_allocated` and `wasted_cpu_h` are
columns of `resources.csv` and `summary_resources.csv` too: `cpu_h` is the measured
CPU time of the row's own last attempt whatever became of it (`cpu_h_used` when it
registered), `cpu_h_allocated` and `write_cpu_h_allocated` are `cpu_h_reserved` and
`write_cpu_h_reserved` of a registration in several jobs, and `wasted_cpu_h` is
`wasted_cpu_h_used` of a row that registered. `wasted_wall_s` of a STARE row also
holds STITCH's failed attempts; the `wasted_*_reserved` columns do not, because STITCH
is the writer (`write_wasted_cpu_h_reserved`).

**Attempts lost to the cluster are not the method's cost.** An attempt that ended for
a cause that is not the method's (a node that failed, a pre-empted or cancelled job, a
task that never started, the benchmark's own step) is billed in the `lost_*` columns
and counted in `lost_attempts`. It is in no `wasted_*` and no `total_*` figure, so a
method is not charged for the node it happened to run on.

**Aggregation.** Per patient: the sum over its rounds for a method that runs per pair,
the launch's own figure for a per-patient method. The good figures are over the rounds
that registered, a round whose writer then failed included, and a patient's figure is
blank when one of those rounds lacks it (`gb_h_reserved` without a memory request or a
stage record, `cpu_h_scheduler` without the job in the accounting): every column of a
patient is over the same rounds. A patient's wasted attempts are every
failed attempt of it; a failed round inside a per-patient launch that otherwise
finished shares that launch's clock and is no attempt of its own. Per method: the
median over patients, given twice (`cost_summary.csv`): over the patients that method
completed, and over the patients EVERY compared row completed (`*_common_med`, with
`n_patients_common`), each with its mean. A cost a complete patient can lack
(`gb_h_reserved`, `total_gb_h_reserved`, `cpu_h_scheduler`, `cpu_h_used_vs_scheduler`)
has the number of patients each median is over beside it (`*_n`, `*_common_n`). Only the second may be placed beside another
method's, and `summary.md` prints the two counts on the same line.

**Which ratios are valid.** CPU-hours (used, reserved, total) between rows, on the
common patients, with `device` stated. Not valid: STARE's time with unlimited parallel
tiles against a single process's time; a `gpu` row against a `cpu` row as if they had
the same resources; totals that end at different deliverables (`merged_image`, "Cost
by step").

**The instrument: a counter the kernel keeps, never a sample.** A registration in one
process (`resources.Monitor`) reads the CPU counter of its own scheduler step's
cgroup at the start and end of every stretch it bills (`cgroup_v2`: `usage_usec` of
`cpu.stat`; `cgroup_v1`: `cpuacct.usage`); it counts every process of the task,
however short its life. Where there is no such counter (a laptop, a site that gives a
job no cgroup of its own) it reads `os.times()`, this process and every child that
has been waited for, plus the processes still alive, each in exactly one of the two
(`rusage+live`). With that instrument, and with `rusage` (no way to read live
descendants), a child's time is counted when the child is reaped, not when it was
spent: a child that exits before a pause of the clock and is reaped during it is
billed to the paused stretch (the writer's, or the benchmark's own), not to the
registration; the cgroup counters do not have this. A registration in several jobs is
measured by `os.wait4` of each stage command, the same kind of figure. `cpu_instrument` (`resources.csv`,
`run.json: resources.cpu_instrument`) says which one a row had, and only its last
resort, `polled (may undercount)` (no counter and no `os.times` children figures),
samples the process tree and makes `cpu_h_used` a LOWER BOUND. The stretches taken
off the method's clock are read with the same counter: the writer's is
`write_cpu_h_used` (`billing_note` says when that stretch holds more than the
writer: an adapter that pauses the clock around its writer AND its own bookkeeping,
or times the benchmark's copying inside its writing step), and the benchmark's own
point mapping is `run.json: warp_cpu_s`, in no column of the method. The sources are
cited in `resources.py`, whose bytes are part of what a finished unit is reused by
(`reuse_key`).

**The scheduler's figure is a cross-check, not the number** (`cpu_h_scheduler`). The
launcher's settle step asks `sacct` for every job of the submission WITH its steps
(`logs/accounting.psv`; no `-X`), and the scorer reads each row's jobs from it: the
row's own launch for a method in one process, the jobs of the good stage attempts for
STARE. A job's CPU time is the larger of its own line's `TotalCPU` and the sum of its
steps' (`ladder.parse_accounting`). It is not the main number because, for a row that
is one job, that job also holds the benchmark's own work (moving the nuclei, the
tissue reduction), the method's writer and every other result of the same launch (a
per-patient launch: every round and every stage), so it exceeds the registration by
an amount that differs between methods; `run.json: resources.launch_cpu_s`, the whole
launch as the row's own counter read it, is the figure to hold against it. It is
blank when the accounting has no CPU time for one of the row's jobs; a zero is read
as nothing. Slurm's own words: `TotalCPU` is "the sum of the SystemCPU and UserCPU
time used by the job or job step"; for a step ended by a signal it "may not include
CPU time of child processes"; and with `-X` "utilization statistics for job
allocation(s) will be reported as zero", which is why the steps are asked for (sacct
manual, slurm.schedmd.com/sacct.html `[PARTIAL: the -X / --allocations option, the
entries of JobID, JobIDRaw, ElapsedRaw, State, SystemCPU, TotalCPU and UserCPU and
the --allocations example read from the page itself; the other options and fields
not read]`). Not known until the first patient has run: whether this site's `sacct`
adds the steps up into the job's own line (the parser is right either way), and
whether the site puts a job's processes in a step cgroup whose counter the container
can read (`cpu_instrument` of the first rows says).

### The other columns

- **A patient's writer time and time-to-image are the patient's totals, for both kinds of
  unit.** A per-patient method records each round's SHARE of one writer call (ASHLAR:
  the call divided by the rounds; VALIS 1.2.0: by the slides written), so the patient's
  `write_s` is the sum of the shares and `total_to_image_s` is the launch's registration
  time plus that sum. Until 2026-10-11 `total_to_image_s` of a per-patient launch was the
  largest single round, i.e. the registration plus ONE share: with five rounds, four
  fifths of the writer were missing. `resources.csv` keeps the per-round share.
- **Survivors.** A cost exists only for what completed. `summary.md` and `cost_summary.csv`
  therefore give each method's median cost twice: over the patients that method
  completed (every round registered; `n_patients_complete` of `n_patients`), and over
  the patients EVERY compared row completed (`n_patients_common`). Only the second is on
  the same patients for every method; in the first a method that fails on the largest
  patients looks cheaper. The cost-by-step table has the same pair (`*_common`).

- **The clock and the peak are per variant.** Each `run.json` holds the cumulative time and
  the process-tree RSS peak up to its own yield. Warping the benchmark's points and a
  method's own writer run with both paused (`resources.Monitor.pause()` / `.resume()`).
- **`peak_rss_gb` is the process tree's peak up to that result, and it is NOT writer-free
  for every row.** A pause keeps out what a paused stretch allocates and frees. What that
  stretch leaves resident is seen again once the clock runs. So for a method whose writer
  runs between two results (VALIS, ASHLAR: the writer is inside the generator), every
  result AFTER the first written one has a peak that can include what an earlier writer
  left in memory; the first written result, and every method whose writer runs after the
  generator has ended (STARE, PALOM, DeeperHistReg), have a peak without any writer.
  ASHLAR also records `peak_rss_after_register_gb`, read before its writer starts.
  `cgroup_peak_gb` is the whole job's high-water mark, writers included, for every
  method: it cannot be paused.
- **The ladder.** Every method runs at 4 CPUs; the four DeeperHistReg `_gpu` runs (one of
  them a primary row) have one GPU as well, which no other method has (`device` and
  `gpu_model` are columns of the cost tables, and their `_cpu` twins are shown beside
  them). Memory starts at 16 GB and moves up one rung
  (64, 256, 512, 700 GB) only when the run ran out of memory. `attempt`, `mem_requested_gb` and
  `cpus_requested` are recorded per run; earlier failed attempts are kept in
  `previous_attempts` and reported as `wasted_*`. `slurm/submit_cohort.sh` submits each
  rung as two chained job arrays (the attempt, then the same rung once more for a unit
  the scheduler lost); a task of a later rung starts, reads its unit's last `run.json`,
  and does nothing unless that names a memory cause (`ladder.py`, `run --ladder
  climb|again`). The driver reads `REGBENCH_ATTEMPT`, `REGBENCH_RETRY` and
  `REGBENCH_MEM_GB`.
- **`wasted_*` for an attempt the scheduler killed.** Such an attempt could not bill
  itself: its record is the placeholder. The next task of the ladder (and, for the last
  rung, `regbench settle` before the scorer) asks `sacct` for that task's state, elapsed
  and CPU time and writes them into the record (`scheduler`). The attempt is then billed
  by the scheduler's elapsed time of the WHOLE task, an upper bound (the container's
  start and the benchmark's own steps are in it), and counted in `wasted_from_scheduler`.
  Where `sacct` gave no end state within the wait the attempt is counted in
  `wasted_unknown` and is in no wasted total: unknown, never zero.
- **The time limit is one rule for both kinds of unit, with one consequence to know.**
  One `REGBENCH_TIME` per round: a pair task gets it once; a patient task gets it times
  the rounds of the LARGEST patient of its array (an array has one limit), never above 7
  days. So a patient with fewer rounds than the largest has more time per round than a
  pair has, and a patient with more than seven rounds less (at the default 24 h). Each
  `run.json` records its task's limit (`time_limit`), the cost header prints the limits
  seen per kind of unit, and `timeout` is never retried.
- **What counts as out of memory.** The driver watches the task's own cgroup counter
  (`oom_kill`), from before the method starts until its writer has finished, and ends the
  run when it rises, because the kernel may kill one worker and leave the parent waiting:
  `oom` while the method registers, `writer_oom` while its writer runs, `benchmark_oom`
  while the benchmark moves the nuclei. `resources.MEMORY_ERRORS` is the fixed list of
  self-reported memory errors that are also `oom`; VALIS prints its exception and
  returns, so what it printed is put back into the error first (below, "VALIS"). All
  three memory causes go up a rung. `timeout`, `exception`, `writer` and `gpu_oom` never
  escalate.
- **The watcher can be absent, and the tables say so.** Where the counter is not readable
  from inside the job (not observed under `apptainer exec --cleanenv`), no watcher
  starts; `run.json: oom_watch` records it, and the cost tables open with `out-of-memory
  watcher: N of M results ran WITHOUT one`. A `timeout` among those runs may be a worker
  the kernel killed.
- **A unit that is already done is not run again.** `run` (and so every array task) leaves
  a unit alone when every one of its results is `ok` for the same method, label, options
  and cases: nothing runs, no placeholder is written, and each record gains the
  submission's id (`submissions`, `reused_by`). `REGBENCH_FORCE=1` or `run --force` runs
  it anyway. The cost tables open with `reused: N of M results`.
  A result is also made FOR its case's points: `run.json: points_sha256` holds a hash of
  each group of arrays of the case's `points.npz` (`outlines` and `nuclei`). A
  case prepared again with other points (a forced re-prepare; outlines whose meaning
  changed) makes its earlier results stale: they are not reused, and the scorer counts
  them as `not_run` ("made for an earlier version of this case"), never as a failure of
  the method. Adding a group the result never had (bare nuclei to a case with outlines
  only) leaves it valid.
- **Threads.** No thread variable is exported. Where a method has its own core-count option
  it is set to `resources.n_cpus()` = `min(SLURM_CPUS_PER_TASK, affinity)`; both numbers
  are recorded. `cpu_efficiency` (`cpu_h_used / cpu_h_reserved`) is reported because a method
  that starts one thread per node core can exceed its allocation where cores are not
  confined.
- **Threads the method's own image sets (ASHLAR: one OpenMP thread on its 4 CPUs).** The
  ruling stands that each method runs as its official distribution configures it and
  the harness sets and overrides nothing. For ASHLAR that configuration includes a
  thread limit:
  - its official Dockerfile ends with `ENV OMP_NUM_THREADS 1` (labsyspharm/ashlar at the
    tag v1.20.0, `Dockerfile` line 21 `[FULL: all 27 lines, fetched 2026-10-12]`; that
    `labsyspharm/ashlar:1.20.0` on Docker Hub was built from this file is assumed, not
    checked);
  - the launcher starts every image with `--cleanenv`, which keeps host variables out
    but not the image's own: "The environment variables from the base image or
    definition file used to build a container always apply", and the list of steps in
    which the environment is built clears the host's first and restores it last only
    without `--cleanenv` (Apptainer user guide, "Environment and Metadata" `[PARTIAL:
    the sections "Environment Overview" and "Environment Variable Precedence", read
    through a summarising fetch that quoted them; the page itself was not read line by
    line, and the site's runtime may be Singularity 3.x, whose guide was not read]`).

  So inside its job ASHLAR finds `OMP_NUM_THREADS=1` and whatever of it is threaded
  through OpenMP runs on one thread, while its allocation, like every method's, is 4
  CPUs. ASHLAR has no core-count option, and the four files of its source read for the
  adapter start no thread or process pool of their own. Its correlations are
  `scipy.fft.fft2` / `ifft2` calls with no `workers` argument (`reg.py:802-803`,
  `:991-992`), so they are serial as written. `pyfftw` plays no part: the one function
  that names it, `utils.fourier_shift` (`utils.py:113-139`), has no caller in those
  four files, and `utils.py` does not import `pyfftw` (ASHLAR 1.20.0, the files
  `reg.py`, `utils.py`, `scripts/ashlar.py` and `thumbnail.py` of the official source
  `[PARTIAL: searched for pyfftw, fourier_shift, fft and for thread or process pools;
  not read line by line; scikit-image's phase_cross_correlation, which thumbnail.py
  calls, and scipy.fft's default of one worker were not read]`). How far NumPy's BLAS
  and scikit-image there obey that variable was not read from their sources `[MEMORY]`. The
  consequence for the cost tables: ASHLAR's wall time is that of a (nearly)
  single-threaded program, and its CPU-hours as billed (4 CPUs x wall) are about four
  times what it used; `cpu_efficiency` near 0.25 is what that looks like. This is
  ASHLAR as published, not a handicap the benchmark adds, and it is not corrected.
  What makes it visible: `run.json: threads_env` holds the three thread variables AS
  THE RUN FOUND THEM in its own process, inside the image (`null` when unset); the
  scorer carries the ones that are set into `resources.csv` (`threads_env`), and the
  cost tables of `summary.md` open with a line naming every method whose job had one
  ("thread variables: the benchmark sets none. Found set inside the job, so by the
  method's own image: ashlar OMP_NUM_THREADS=1"). Never observed on the cluster: ASHLAR
  has not run.
- **Cores confined.** The cost tables in `summary.md` open with one line, `cores
  confined: yes | no | partly | unknown`, computed from what every run recorded
  (`cpus_requested`, what the scheduler was asked for, against `cpus_affinity`, the cores
  the process could run on), with the workers each method's own pool had. It is a
  precondition of comparing wall times: STARE's coarse FFT (`workers=-1`) and PALOM's
  dask pool take every core they can see.

### What each method's clock includes

Methods are not given a common reader or writer (ruling R3.13): each adapter runs the
package's own. So the clocks do not cover the same work, and this table says what each
covers today. "Not separable" means the package does that step inside its registration
call; it is then part of the estimate and no number is guessed for it.

| method (adapter) | reading and conversion | its own conversion, reported as | the registration clock | the writer | what it writes |
|---|---|---|---|---|---|
| `stare` | inside the clock: STARE's own reader on the cohort's OME-TIFFs | none | COARSE, one REG-TILE job per tile and SOLVE, each a scheduler job of its own with the pipeline's allocation: COARSE + the longest tile + SOLVE, each the attempt its job ended ok with (`wall_unlimited_parallel_s`; a failed attempt is `wasted_*`; queue waits are in `elapsed_with_queue_s`) | STITCH, a process of its own inside the round's last job once the transform exists, timed on its own (the attempt that wrote the image) | one OME-TIFF per round, every channel, BigTIFF, uncompressed, no pyramid |
| `valis_mirage` | inside the clock: VALIS's readers, its conversion of every slide to its working images, and the JVM start. Outside it: the benchmark placing the slides in the folder VALIS reads (`info.place_s`: a hard link, or a copy) | not separable | the pipeline's `valis_registration` up to the writer (rigid, non-rigid, micro) | `Slide.warp_and_save_slide`, clock and peak paused, timed per slide | one pyramidal OME-TIFF per round, every channel (libvips, LZW) |
| `valis` | inside the clock: VALIS's readers, its conversion of every slide to its working images, and the JVM start. Outside it: the benchmark placing the slides in the folder VALIS reads (`info.place_s`: a hard link, or a copy) | not separable | `Valis.register()` and `register_micro()` | `warp_and_save_slides`, clock and peak paused; one call writes the patient, a round's share is the call over the slides written | one pyramidal OME-TIFF per round, every channel (libvips, deflate) |
| `deeperhistreg` | the package's own loading is inside the clock (its log's `Loading time` is kept in `stage_s`) | `convert_s` | `run_registration` with the final saving off (its diagnostic images and intermediate fields included, as shipped) | `apply_deformation_pyvips` once per channel; the conversion of each further channel before it and the benchmark's copy of those files into one after it are timed apart (`native.write_parts_s`) | one single-channel pyramidal TIFF per channel (the package's); the benchmark stacks their first pages into one OME-TIFF without a pyramid |
| `ashlar` | inside the clock: ASHLAR's reading of the tiles it is given | `retile_s` | the command up to the last aligner; a round's share is its `layer_s` plus the reference stitch over the rounds | ASHLAR's writer, inside the same command, timed from the last aligner; then the benchmark's split of the mosaic per round, timed apart (`native.write_parts_s`) | one pyramidal OME-TIFF for the patient, every cycle's channels (BigTIFF, deflate); the benchmark's per-round copies have no pyramid |
| `palom` | inside the clock: PALOM's reader, lazily, on the cohort's OME-TIFFs | none | `Aligner.coarse_register_affine`, `compute_shifts`, `constrain_shifts` | `block_affine_transformed_moving_img` and `write_pyramid`, timed on their own | one pyramidal OME-TIFF per round, every channel (BigTIFF, zlib) |

What a writer leaves is described twice: here from each package's source (the tags are
in `cost.py`), and per run from the file itself (`run.json: native.written`: channels,
pyramid levels, compression, tile, bytes, read with tifffile after the writer has
finished and the clock has stopped). Where the file handed to the scorer is the
benchmark's copy and not the package's own output (ASHLAR's mosaic cut per round,
DeeperHistReg's per-channel files stacked), the adapter describes the package's own
file before it copies it, and `native.write_parts_s` holds the three parts of the write
step apart: `writer` (the package's), `convert` (a conversion that writer needed) and
`benchmark` (the benchmark's own copying).

### A registration in several jobs (STARE)

STARE's registration of a round is COARSE, one job per tile and SOLVE, each a scheduler
job with the pipeline's allocation ("STARE"), so its cost is summed over those jobs
(`run.json: info.jobs`, `stages.summarise`) and its wall time is not a 4-CPU wall time.
Every cost table opens with a line that says so and how many CPUs one round had at once.
The columns, in `resources.csv` per round and summed per patient in
`summary_resources.csv` (blank for every other method):

| column | definition |
|---|---|
| `elapsed_with_queue_s` | from the start of the first COARSE attempt to the end of SOLVE, on the wall clock: the queue waits between the jobs and the tile submitter's ceiling are in it. Blank when COARSE and SOLVE are of two submissions (a reused stage is not a queue wait) |
| `wall_unlimited_parallel_s` | COARSE + the LONGEST tile + SOLVE, each the attempt its job ended ok with: the wall time with every tile running at once and no queue |
| `cpu_h_reserved` (also as `cpu_h_allocated`) | sum over those attempts of allocated CPUs x duration |
| `cpu_h_used` (also as `cpu_h`) | sum over the same attempts of their measured CPU time |
| `gb_h_reserved` | sum over the same attempts of each one's memory x duration (attempt 2 of a stage job has twice the memory of attempt 1), from the stage jobs' records |
| `unmeasured_attempts` | stage attempts that ended without a recorded duration (a job the scheduler ended at its time limit): in no sum, so where it is not 0 every time and CPU figure of the row is a LOWER BOUND |
| `largest_job`, `peak_rss_gb` | the stage of the single attempt with the highest peak memory, failed ones included, and that peak |
| `n_jobs`, `n_tiles`, `n_job_retries` | attempts run, tiles, attempts beyond the first of each job |
| `peak_cpus_at_once` | the most allocated CPUs in jobs of the round that overlapped in time (per patient: the largest round's) |
| `write_cpu_h_reserved` (also as `write_cpu_h_allocated`), `write_cpu_h_used`, `write_peak_gb` | STITCH's own process, the attempt that wrote the image: the pipeline's 4 CPUs x its duration, its measured CPU time, and its peak |

For these rows the time columns are defined as follows. `register_s` is
`wall_unlimited_parallel_s`: the queue-free figure, because the waits belong to the
cluster's load and not to the method. `total_to_image_s` is `register_s` plus
`write_s`, STITCH's wall time in the round's last job. Neither is a single process's
time, so in `summary.md` and `cost_summary.csv` they are not under those names: they
are `time_unlimited_parallel_tiles_s` and `to_image_unlimited_parallel_tiles_s`, with
`peak_cpus_at_once` beside them, and the plain columns are blank for such a row. `peak_rss_gb` is the largest
single job's peak (no job of the round holds more), and `cgroup_peak_gb` is the last
job's, which holds STITCH and the benchmark's own steps. `cpu_efficiency` is measured
over allocated CPU time. A duration is the stage command's own process (Python start
to exit), not the scheduler's elapsed time of the job, which also has the container's
start. `attempt`, `mem_requested_gb` and `cpus_requested` of such a row describe the
round's LAST job (the benchmark's allocation), not the registration.

**One retry rule for every method.** A failed attempt's seconds and CPU-hours go to
`wasted_*`, and `register_s` is the attempts that succeeded. For a method that is one
job the attempts are the rungs of the memory ladder (`previous_attempts`); for STARE
they are the stage jobs' own attempts (a retried tile, a COARSE that ran twice) and
STITCH's, so `coarse_s`, `longest_tile_s` and `solve_s` are the successful attempt of
each and every other attempt is in `info.jobs.wasted_*` and in the row's `wasted_*`.
A round that FAILED shows what its stage jobs cost (`info.jobs`, kept with the
failure), not this job's own clock. A writer that fails after a good registration does
not make the registration a wasted attempt. Every retried or failed stage attempt is a
row of `failure_log.csv` (`retry` = `stage`, the job named in `message`): a stage job that never
recorded its end is `timeout` when the next attempt of that job started no earlier than
its start plus its time limit (the attempts are chained, so that is when it ended), and
`killed` otherwise, also when there is no later attempt to compare with.

### Cost by step, as measured here (`cost_end_to_end.csv`, secondary)

`write_s` here IS the headline `write_s`: the method's own writer where its adapter
times it apart from the benchmark's own copying, and otherwise its write step. The
copying is in `benchmark_copy_s` and in no total, here or in the headline block
(`test_cost.py::test_the_headline_write_time_and_the_by_step_one_are_the_same_number`).
The medians are over the patients that wrote an image for EVERY round
(`n_patients_partial` counts the others), the completeness rule of the per-method block.
For both VALIS adapters the benchmark's placing of the slides in the folder VALIS reads
(a hard link, or a full copy of every slide across file systems) runs with the clock
paused and is recorded as `info.place_s`, in no cost.

The headline cost is that of ESTIMATING the transform, in CPU-hours first
("CPU-hours: one definition for every method");
the headline tables also carry the times, the time to the written image and the memory peaks
(`summary_resources.csv`, `cost_summary.csv` and the block "Cost per method" of
`summary.md`: `total_to_image_s`, `peak_rss_gb`, `cgroup_peak_gb`). A second table puts
every step in its own column, per patient and per result that wrote an image:

| column | what | measured |
|---|---|---|
| `method_convert_s` | a conversion the method needs before it can read the slides: DeeperHistReg's single-page TIFFs (`convert_s`, plus the further channels its writer needs), ASHLAR's tiles (`retile_s`, once per patient). `none` when it reads the cohort's files as they are; `not separable` for VALIS | here, 4 CPUs, clock paused |
| `estimate_s` | the registration clock | here, 4 CPUs |
| `write_s` | the package's own writer | here, 4 CPUs, clock paused |
| `measured_here_s` | `method_convert_s + estimate_s + write_s`: the total measured time of that result, from the cohort's files to the image it wrote | here |
| `benchmark_copy_s` | the benchmark's own copying of a package's output (ASHLAR, DeeperHistReg). In NO total | here |
| `estimate_peak_gb`, `job_peak_gb` | the process tree's peak while registering; the whole job's peak, writer included (the cgroup's) | here |
| `written` | what the method's writer left: channels, pyramid or not, compression, size | the file |
| `merged_image` | is that image already ONE pyramidal file for the patient with every round's channels: `yes`, `no`, or `not recorded` | the file |
| `device`, `gpu_model` | DeeperHistReg's CPU and GPU rows are in this one table, told apart here | |

- **Every number is this benchmark's own.** Nothing is read from a pipeline run. Until
  2026-10-12 `score --pipeline-trace` (and `REGBENCH_PIPELINE_TRACE`) imported the
  pipeline's `CONVERT_IMAGE` and `MERGE_AND_PYRAMID` times from a Nextflow trace into
  `pipeline_convert_s`, `pipeline_merge_s` and a `total_s`; the owner ruled that out, and
  the option, the variable, the reader and those columns are removed. A total that
  added seconds measured under another run's allocation no longer exists.
- **Where the total stops, and what that leaves out.** `measured_here_s` ends at the
  image each method writes itself, and those images are not the same deliverable. A
  pipeline's product is one pyramidal file per PATIENT with every round's channels;
  `merged_image` says, from the file each run wrote and never from the method's name
  (`cost.complete_image`, ruling R3.14), whether a result's own image is that: the file
  has a pyramid, it is one file, and it holds at least as many channels as the patient's
  slides hold together (`native.written.patient_channels`). With the adapters as built:
  ASHLAR's mosaic is (`yes`; an anchored row that dropped a round is not); STARE (one
  flat file per round), both VALIS adapters and PALOM (one pyramid per ROUND) and
  DeeperHistReg (one file per channel) are not (`no`), and would still need a
  merge-and-pyramid step that this benchmark neither runs nor times. So ASHLAR's total
  covers one step more than the others': read `merged_image` beside every total. The
  conversion that made the cohort's files (the same for every method) is not in any
  number either.
- **Nothing is decided on a guess.** A run that recorded no description of its file, or
  a case that does not list its channels, reads `not recorded` in `merged_image`.

### Full pipeline cost, per backend (compute-all; `pipeline_cost*.csv`, secondary)

**Not run on the cluster yet**: what is tested is the job graph of the dry run, the head
job against a stand-in `nextflow`, and the tables on hand-written traces
(`test_compute_all.py`, `test_trace.py`).

`submit_cohort.sh --compute-all` (or `launch.sbatch --compute-all`; both need
`REGBENCH_INPUT`, the cohort's samplesheet) adds, after the scorer, one Nextflow head job
per patient and registration backend (`slurm/compute_all.sbatch`, 2 CPU, 8 G,
`REGBENCH_COMPUTE_TIME`, default 7 days): the WHOLE pipeline, no `--start`/`--stop`,
once with `registration_method=valis` and once with `tiled` (STARE), every `valis`
patient first. The heads are dealt onto `REGBENCH_COMPUTE_LANES` lanes (default 2), each
after the head before it on its lane and the first ones after the scorer, so they run
LAST and ALONE; then `rb-score-compute` scores again and adds the tables.
`--compute-only` submits those jobs alone (the heads start at once) and rescores the last
submission, like `--score-only`.

- **What is pinned**, through `-params-file` (`regbench compute-all params`, typed by
  `nextflow_schema.json`): `registration_method`; `enable_trace=true`, its shipped
  default restated so a site profile cannot switch the measurement off; and
  two rulings that are NOT the shipped defaults: `seg_method=stardist` (the user's
  ruling of 2026-10-15: the benchmark's nuclei are StarDist's; shipped `instantseg`) and `pixel_size=0.325` (the input step's ruling, extended
  to compute-all: the scale of the inputs the benchmark scores, and no run stopped by a
  slide without metadata; shipped `auto`, which the pipeline's preflight refuses when it
  cannot read a scale). Everything else is the shipped default and the site profile
  (`REGBENCH_NF_PROFILE`, `REGBENCH_SITE_CONFIG`). **Registration therefore runs at the
  shipped tier `memory_mode='high'` unless the site profile overrides it**, which may
  differ from the settings the site uses in production (caveat 12).
  `test_compute_all.py::test_the_pinned_params_are_real_and_at_their_shipped_default`
  fails when a pinned name is not a schema param, a value is not of its schema type,
  enum, pattern or bounds, a restated value is no longer the shipped default, or the
  pixel size is not the one `slurm/inputs.sbatch` pins.
- **One folder per run**: `<out>/compute/<backend>/<patient>/` holds its samplesheet
  (the header and that patient's rows), `params.json`, `work/`, `trace/`, `results/`,
  `input_gb.txt` and Nextflow's logs; its run name is unique (submission and job id).
  It is never resumed: a folder that is there is moved to `<out>/compute/.replaced/`
  (never deleted), with its end line `<out>/logs/compute/<backend>/<patient>.end`.

- **Where the files must be.** `compute-all plan` and `compute-all sheet` (the patient's
  samplesheet, and `input_gb.txt` from the sizes of its slide files) run inside the
  `STARE_REGBENCH` image, as every `regbench_py` call does (`slurm/env.sh`,
  `regbench_exec`). That image sees only the folders bound into it: `REGBENCH_REPO`,
  `REGBENCH_CASES`, `REGBENCH_OUT`, `REGBENCH_WORK`, `REGBENCH_INPUTS` and every folder
  of `REGBENCH_BIND` (comma-separated; `launch.sbatch` fills it from the site config's
  `singularity.runOptions` `--bind` folders and adds `REGBENCH_ROOT`). So
  `REGBENCH_INPUT` and the slides it names must sit under one of those folders: if the
  samplesheet is not, the launch stops before the first sbatch; if a slide is not,
  `input_gb` is blank (NaN). Nextflow itself reads them through the site profile's own
  binds.
- **Disk, per run (an ESTIMATE from the published layout, not measured).** The shipped
  publish behaviour is kept, because publishing is part of what the run costs:
  `cleanup_level` stays `none`, so each run publishes every kind of `lib/Layout.groovy`.
  About six of them are image-sized copies of the patient's slides (`converted/`,
  `preprocessed/`, `registered/`, `split_channels/`, `pyramid/` with its lower levels,
  and the SpatialData `.zarr`), the rest (masks, quantification, GeoJSON, QC) smaller.
  Count on the order of 6 to 7 times the patient's `input_gb.txt` under `results/`, per
  backend, plus up to about as much again in `work/` while the run lasts
  (`cleanup_work` stays on: the work folder is emptied when a run succeeds and kept when
  it fails). Compression differs between the stages, so the real figure can be off
  either way; the first run's `du -sh <out>/compute/<backend>/<patient>` is the number.
  The traces sit outside `work/` and are always kept.

`pipeline_cost.csv`, one row per task (`compute_all.COLUMNS`): `backend, patient,
process` (the last `:` segment), `process_path` (the full name), `task_id, tag, status,
exit, cpus, memory_gb, realtime_s, duration_s, pcpu` (Nextflow's `%cpu`), `cpu_h_used,
cpu_h_reserved, gb_h_reserved, peak_rss_gb, peak_vmem_gb, read_gb, write_gb` (`rchar`,
`wchar`), `submit, start, complete`, `is_registration` (`REGISTER` for valis;
`TILED_COARSE`, `TILED_REG_TILE`, `TILED_SOLVE`, `TILED_STITCH` for tiled). GB are
Nextflow's (2^30 bytes). The three products are named as `billing.py` names the methods'
and defined the same way, from Nextflow's figures:

| column | Nextflow trace | billing.py's definition it mirrors |
|---|---|---|
| `cpu_h_used` | `%cpu/100 x realtime / 3600` | CPU time used |
| `cpu_h_reserved` | `cpus x realtime / 3600` | allocated CPUs x time |
| `gb_h_reserved` | `memory x realtime / 3600` | requested memory x time |

They are NOT the same measurement: `%cpu` is what Nextflow's task wrapper reports, not
the cgroup counter, and `realtime` is the task's own run time inside the allocation the
pipeline asked for. It is a different clock from the methods' registration cost, so
**no total adds the two**.

`pipeline_cost_summary.csv` (`compute_all.SUMMARY_COLUMNS`), one row per (backend,
patient) and one per backend with `patient = all`: `n_patients, n_tasks, n_completed,
n_failed, n_retried` (failed tasks whose task was run again and completed, the
pipeline's own retry), `n_cached, complete_run, wall_s` (`max(complete) - min(submit)`),
`cpu_h_used, cpu_h_reserved, gb_h_reserved, peak_rss_gb, registration_wall_s,
registration_cpu_h_used, registration_cpu_h_reserved, registration_share_cpu_h_used,
input_gb` (the patient's slide files, written by the head). A failed task was paid for:
it counts in its patient's row. `complete_run` is 1 only with every failure retried to
success, no cached task and a `status=ok` end line; the `all` row sums those patients
only (`n_patients` says how many; its counts are sums over them) and its own
`complete_run` is 1 when that is every patient listed.

**Reading the `all` row.** Its `wall_s`, `registration_wall_s` and `input_gb` are the SUM of
the per-patient values (over the patients in the row), not the wall time of one run: the
heads run on `REGBENCH_COMPUTE_LANES` lanes, so the elapsed time of the whole is shorter.
A patient's `wall_s` is measured from its first task's submission, so it **includes SLURM
queue time**; the CPU-h and GB-h columns are built on Nextflow's `realtime`, which does not.

`pipeline_cost_by_process.csv` (`compute_all.BY_PROCESS_COLUMNS`): per (backend,
process) over the patients of the `all` row: `is_registration, n_patients, n_tasks,
n_failed, realtime_s, cpu_h_used, cpu_h_reserved, gb_h_reserved, pcpu_weighted` (`%cpu`
weighted by run time), the peaks and the I/O.

`summary.md` has the block `## Full pipeline cost, per backend (secondary)` when there is
a trace, and otherwise the one line "Full pipeline cost: not measured".

## Running the cohort on a cluster

`slurm/submit_cohort.sh` submits from a login node and returns; nothing waits there and
no job waits on `sbatch --wait`. One kind of job submits jobs: the tile submitter of a
run whose registration is several jobs per round (`feed_tiles.sbatch`, below). **It has
never been submitted**: what is tested is the job graph its dry run prints, the same
graph through a stand-in `sbatch` (`test_submit_cohort.py`), the tile submitter against
a stand-in `sbatch` and `squeue` that keep a queue (`test_feed_tiles.py`), and the rules
its jobs apply (`test_ladder.py`, `test_failures.py`, `test_verdicts.py`).

### What it submits

The run list comes from `python -m benchmarks.regbench runs` and from nowhere else; the
script names no method.

1. **Selfchecks**, one per run that checks a run (15; `checked_by`), in that run's image
   with that run's options and mask (`overlap` for ASHLAR, `seams` for PALOM), at 64 GB
   and 4 CPUs. **Reported, never a gate: no ladder waits for a selfcheck, no run is kept
   from starting and no row is left out because of one** (user ruling: run, report,
   never gate; the gate jobs, their records and the switches that let a run past one are
   gone). Two more
   attempts wait behind the first at 256 and 512 GB; each runs only when the one before
   ended for memory or left no record of this submission (it was lost to the scheduler,
   or never started: `selfcheck --retry-memory`), and otherwise leaves the record as it
   is. `selfcheck.json: attempts` holds the allocation and outcome of each. The anchored
   ASHLAR rows' selfcheck waits for STARE's (`afterany`: whatever it said) and reads its
   work (`--m0-from`), case by case: STARE's selfcheck runs ASHLAR's patient of three
   slides too, for that reason. One job per run covers all of its cases, the patient
   included. Records: `<out>/logs/selfcheck/<run>/selfcheck.json`. Every job carries
   `--kill-on-invalid-dep=yes`, so that a job whose dependency can never be met is
   cancelled instead of pending for ever, and `--no-requeue`: the scheduler never starts
   a task a second time, so a task on a failed node ends as `NODE_FAIL` and the `-again`
   array of the ladder is the one retry (sbatch manual, `--no-requeue`: "the batch job
   should never be requeued under any circumstances" `[PARTIAL: read the --no-requeue,
   --requeue, --kill-on-invalid-dep and --dependency entries; missing the rest of the
   page]`; the manual notes that a site's `ForceRequeueOnFail` prolog flag overrides it).
2. **(Removed; the other steps keep their numbers.)** The gate job per run was removed; the step number is
   kept so that the numbering of the steps below, and the scripts' own comments, stay
   what they were.
3. **The ladder**: per run and dataset, the rungs are 16, 64, 256, 512 and 700 GB at 4 CPUs (700 GB is the site's largest node; the rungs changed on 2026-10-16 by user ruling, before any cohort run of this branch),
   each with an attempt (`run --ladder first` at 16 GB, `climb` above) and `-again`, the
   same rung once more (`run --ladder again`). One task is one case for a run that works
   per pair and one patient for a run that works per patient.
   **A retry is submitted after a failure, never queued before it** (since 2026-10-13).
   The launcher submits ONE array, the 16 GB attempt, and one small job after it, the
   ladder's watcher (`rb-<run>-<ds>-retry`, `slurm/retry.sbatch`, 1 CPU, 1 GB, on a
   compute node, calling `sbatch` itself as the tile submitter does). The seven other
   steps are written down as a plan (`<out>/logs/retry/<submission>/<job prefix>/`).
   When the first array has ended, the watcher reads each unit's last end line
   (`<out>/logs/end/`) and submits a step only for the units that step can retry, as
   `--array=<those units>`, waits for it to leave the queue, and goes on. Which step a
   unit can still use is written ON ITS END LINE by the task (`wants=again,climb`,
   `wants=climb`, `wants=-`), from ONE predicate, `failure_log.chain_wants`, read off every
   record of the unit; a test holds it to never keeping out a unit `ladder.unit_decision`
   would run (fleet wave 2, F3: the watcher kept cause lists of its own that disagreed
   with the ladder, so a patient whose first failed round was the writer's never
   climbed). A task the kernel killed for memory says `cause=oom wants=climb` on its end
   line (F4), so no same-rung retry holds a full allocation only to exit. Every step
   takes a unit that left no end line. A ladder whose units are all ok puts nothing
   more in the queue.
   **The watcher is a chain of links** (`REGBENCH_RETRY_WATCHERS`, default 2; F1): link
   1 (`<prefix>-retry`) waits for the first array, each later link
   (`<prefix>-retry-<n>`) for the link before it (`afterany`), and what waits for the
   chain waits for the LAST link. Each step is written down in the plan folder
   (`state`; `submitting` while sbatch runs) before anything else, so a link ended at
   its 7-day limit, or lost with its node, leaves its successor to go on from the step it
   was at; the link that ends the chain writes `done`, and a later link ends as the chain
   did. A chain no link finished is named by the settle job, by `regbench status` and in
   `summary.md` ("Chains no watcher finished"), and the settle job writes the results
   whose task the scheduler had not ended as `infrastructure` (`chains.mark_in_flight`).
   **A rung no node can grant** (F2): before the first sbatch (and in `--preflight` and
   a dry run) the launcher reads the largest node of the partition the jobs are given
   (`sinfo -N -o '%P|%m|%f|%G'`; the caller's `--partition`/`--constraint`, or
   SBATCH_PARTITION/SBATCH_CONSTRAINT, else the default partition). A run that needs a
   GPU is checked against the nodes `REGBENCH_SBATCH_GPU` gives it: its partition and
   constraint where it names them, and only nodes that HAVE its gres (`%G`); when sinfo
   lists none, against the CPU runs' nodes as before. The nuclei array's first request
   (`REGBENCH_NUCLEI_MEM`, 128G) is checked the same way and stops `--prepare` when no
   node can grant it. It stops when the 16
   GB rung cannot be granted, and says loudly (stdout and stderr) which rungs cannot be.
   Those are not planned: the watcher writes the units that would climb to them down as
   the launcher's (`regbench chain-mark`: cause `not_submitted`, the memory failure below
   kept in `cause_from`, `ladder_cut_gb`), and does the same for the units of a step
   `sbatch` refuses. `REGBENCH_RUNG_CHECK=0` skips the check. At settle time, a chain
   no link finished has its units that wanted a later step no link submitted written
   down the same way (`chains.mark_in_flight`). A successor link that looks in the
   queue for a step an earlier link was submitting asks again when squeue fails, and
   stops (submitting nothing) after 30 failed looks: a failed look is not "not there".
   That reading is a filter and not the decision: the task that is submitted decides
   for itself by the two rules below, as before. Everything that waited for a ladder's
   last array waits for its watcher. The attempts 2 to 4 of a stage job (COARSE, SOLVE) are submitted the same way, by that stage's watcher. The selfchecks' three
   rungs are still three chained jobs. A watcher that is refused by `sbatch`, or cannot
   read the queue 30 times in a row, stops (exit 1): the steps after it are not
   submitted and what is downstream goes on.
   - A `climb` task exits at once unless its unit's last `run.json` names a memory cause
     (`oom`, `writer_oom`, `benchmark_oom`).
   - An `again` task exits at once unless the attempt before it was lost to the
     scheduler: the record says `infrastructure`, or there is no record of this
     submission where one should be (the task never started). **One retry per rung**
     (`ladder.RETRIES_PER_RUNG`); an attempt lost twice stays `infrastructure`, and a
     unit that never left a record is written down as `infrastructure` by the next
     `climb` task. Infrastructure is never read as the method's failure and never sends
     a unit up a rung.
   - Where a record still says `killed` (the process died before it could say why), the
     scheduler's state for THE TASK THAT WROTE IT decides. Each task asks `sacct` for
     the state, elapsed and CPU time of its unit in every earlier job of the ladder
     (`<out>/logs/accounting/`), and a record is matched to its task by `launch_id`:
     `OUT_OF_MEMORY` makes it `oom`, `TIMEOUT` makes it `timeout`, a lost task makes it
     `infrastructure`, any other END state leaves it `killed`. The record is rewritten
     with `cause_from` and `scheduler`. The last rung has no later task, so the
     `rb-settle` job (step 5) runs `regbench settle` on the accounting of every job
     before anything is scored.
   - **One look at the accounting is not the answer.** The accounting can lag behind a
     task that has just ended. When a record says `killed` and `sacct` has no end state
     for the task that wrote it (no line, or `PENDING`, `RUNNING`, `RESIZING`,
     `SUSPENDED`, `COMPLETING`), the rung task decides nothing yet: it asks again every
     `REGBENCH_ACCT_POLL_S` (30) seconds for at most `REGBENCH_ACCT_WAIT_S` (600), and
     its log says what it got (`accounting: ...`). With an end state the rules above
     apply. Without one after the wait (or on a node with no `sacct`) the record is NOT
     final: the `climb` task runs the unit once more at its own rung's memory, the
     `again` task never does, and the replaced attempt is kept in `previous_attempts`
     with `cause: killed` and a `cause_from` that says no end state was named.
   - **The script's own note.** `run_rung.sbatch` exits 0 whatever became of Python, so
     a Python step the kernel's out-of-memory killer ended can leave a task the
     scheduler calls `COMPLETED`. When Python ends with exit 137 (signal 9) and the
     script goes on, it writes `<out>/logs/accounting/exits/<launch id>`; a `killed`
     record of that task whose end state names no cause is then `oom` (it climbs), with
     `cause_from` saying so. A cause the scheduler does name stands. The scheduler's own
     kill (time limit, `scancel`) ends the script as well, so it leaves no note.
   - The output directory is kept between rungs, so the cost of a failed attempt is
     kept (`previous_attempts`). `REGBENCH_ATTEMPT`, `REGBENCH_RETRY` and
     `REGBENCH_MEM_GB` reach the task (both VALIS adapters size a JVM heap from them).
   - The anchored ASHLAR rows wait for STARE's last array on the same dataset.
   **A run whose registration is several jobs per round** (`runs`, column `stages`:
   every STARE run: `variants.stages_of`, the generated set, 9 on the cohort) has, per dataset, a chain of stage jobs in place of that
   ladder, every job after the one before it (`afterany`). With R rounds and T tiles in all (a round's tile count is
   `ceil(H / 2048) x ceil(W / 2048)` of its reference, known before anything runs:
   `regbench stages` writes one line per round to
   `<out>/logs/stages/<submission>/<run>-<dataset>.tsv`):

   | jobs | what | CPUs, memory, time per task |
   |---|---|---|
   | `rb-<run>-<ds>-coarse-a1..a4` | 4 arrays of R tasks | 2; 2, 4, 6, 8 GB; 2, 4, 6, 8 h |
   | `rb-<run>-<ds>-feed-1..2` | the TILE SUBMITTERS (`feed_tiles.sbatch`), `REGBENCH_STAGE_FEEDERS` of them chained: not a tile job, the job that submits them | 1; 1 GB; 7 days |
   | `rb-<run>-<ds>-tile-a1-o<n>` (submitted by the tile submitter, not by the launcher) | ONE TASK PER TILE: T tasks in arrays of at most `REGBENCH_STAGE_ARRAY_MAX` (100), at most its share of `REGBENCH_STAGE_QUEUE_MAX` (a total over the launch, split over the runs feeding at once: 200 is 22 each for 9 runs without lanes; 100 over 4 lanes is 25 each) of them pending or running | 2; 4 GB; 2 h |
   | `rb-<run>-<ds>-tile-a<2..4>-o<n>` (the same) | a RETRY JOB: one task per round that has a tile the pipeline would retry; it runs that round's due tiles one after another | 2; 8, 12, 16 GB; 4, 6, 8 h |
   | `rb-<run>-<ds>-solve-a1..a4` | 4 arrays of R tasks | 1; 1, 2, 3, 4 GB; 8, 16, 24, 32 h |
   | `rb-<run>-<ds>-last-<16,64,256,512,700>g[-again]` | 8 arrays of R tasks (`run_rung.sbatch ... jobs`): the round's LAST JOB reads the stage records, moves the benchmark's points, runs STITCH as a process of its own and reduces the tissue plane | 4; 16, 64, 256, 512, 700 GB (the benchmark's ladder); one round's `REGBENCH_TIME` plus 40 h |

   **The round's last job, in three sentences.** It holds the benchmark's own work
   (loading and moving the outlines, the tissue reduction), so it has the benchmark's
   allocation and ladder, and a death of it is attributed by the rule of every other
   unit's task: the scheduler's `OUT_OF_MEMORY` is `oom` (it goes up a rung), `TIMEOUT`
   is `timeout`, an unexplained death `killed`, all counted against STARE as they are
   against any other method; `benchmark_oom` and `benchmark` only where the driver saw
   its own lines fail. STITCH runs
   inside it as the pipeline's process: measured alone (its own time, CPU time and
   peak, `native.stitch_job`), ended at the pipeline's time limit for its attempt
   (4 h x attempt) and run again on the pipeline's retry list, the time limit included,
   up to four attempts; the job is not STITCH's cgroup, so the pipeline's memory for an
   attempt (4 GB x attempt) is reported against (`stitch_job.over_allocation`), not
   enforced. A STITCH that fails after a good registration fails the row as the
   writer's (`writer`, `writer_oom`; `registered: true`), no rung runs it again, and
   the registration is not booked as a wasted attempt: `total_to_image_s` stays the
   registration plus the STITCH attempt that wrote the image.

   A STITCH killed for memory while its allocation plus the job's own memory exceeded
   what the job had (a 16 GB job, STITCH at its fourth attempt) is the job's limit:
   `benchmark_oom`, and the next rung runs the last job again. What a lost last job
   cost is its own time (`resources.driver_wall_s`), not the registration's.

   Attempt 1 of a stage runs it; a task of attempts 2 to 4 exits at once unless the
   attempt before ended the way the pipeline retries. A task whose stage before it is
   not ok does nothing. The arrays the launcher submits are throttled by
   `REGBENCH_ARRAY_LIMIT` (`%N`). A run that reads this run's results, the same run on
   the next dataset, the next run of its lane and the settle job all wait for its LAST
   job, the last `stitch` array.

   **The tile submitter** (owner's ruling: the tile jobs reach the queue through a
   throttled submitter, as Nextflow's queue size does for the pipeline). One task per
   tile for a cohort is hundreds of thousands of tasks; submitted up front they would
   all be pending at once. So the launcher submits, between COARSE's last attempt and
   SOLVE's first, a small job per run and dataset that runs on a compute node OUTSIDE
   any image and calls `sbatch` and `squeue` itself, the way a Nextflow head job does.
   It asks the harness what is due (`regbench stages --due <attempt>`, in the run's
   image), and then:

   - attempt 1: one task per tile that is due (every tile of every round whose COARSE
     is ok, less the tiles a reusable record already covers), in arrays of at most
     `REGBENCH_STAGE_ARRAY_MAX` tasks, each task with the pipeline's allocation for
     `TILED_REG_TILE`. Before each `sbatch` it counts this run's tile tasks that are
     pending or running (`squeue -r`, by job name) and submits only what keeps the count
     at or below its share of `REGBENCH_STAGE_QUEUE_MAX` (`REGBENCH_STAGE_QUEUE_RUN` inside the submitter; the launcher sets it); it waits for `REGBENCH_STAGE_FEED_MIN`
     free slots (25) so that it does not submit one-task arrays, and looks again every
     `REGBENCH_STAGE_POLL_S` seconds (60). A queue it cannot read is never taken for an
     empty one (below). The tile arrays carry no `%N`: the ceiling is their limit;
   - attempts 2 to 4, each once the attempt before has left the queue: a RETRY JOB is
     one task per round that has a tile the pipeline would retry, and it runs that
     round's due tiles one after another with the attempt's memory and time (x attempt).
     A retried tile is therefore not a job of its own, which the pipeline's is;
   - it ends when an attempt has nothing due, or after attempt 4, and leaves
     `<out>/logs/stages/<submission>/<run>-<dataset>.done`.

   It is RESTARTABLE from two files beside the layout: `.due.a<attempt>` (what is due,
   written once) and `.fed.tsv` (one line before and one after each `sbatch`). Its time
   limit is `REGBENCH_STAGE_FEED_TIME` (7 days, the site's longest job). When the
   scheduler ends it there, the next submitter of the chain starts (`afterany`), counts
   the tile tasks still in the queue, and continues where the files say the first
   stopped; nothing is submitted twice, except one array whose `sbatch` was cut off
   before its id was written and which is found neither in the queue nor in the
   accounting (a tile whose record is already ok then exits at once). The accounting
   is asked from the time the chain's first link started (`<run>-<dataset>.started`,
   passed as `sacct -S`): without a start `sacct --name` looks from midnight of the
   current day only (sacct manual, DEFAULT TIME WINDOW; tag under "Not verified"), and
   a link that restarts on a later day would not find an array that has ended. With
   `REGBENCH_STAGE_FEEDERS=2` that is 14 days of feeding; a run that needs more is
   continued by a new submission, which reuses every stage that is ok. SOLVE and the
   round's last job wait for the LAST submitter (`afterany`). A round whose tiles
   never reached the queue fails as `not_submitted`: the launcher's failure, counted
   apart from the method's, and never retried on a higher rung.

   An `sbatch` the scheduler refuses inside the submitter is tried
   `REGBENCH_STAGE_SUBMIT_TRIES` times (5), a poll interval apart. Then the submitter
   writes `<run>-<dataset>.refused` (what was refused, the scheduler's words, how many
   tile tasks are queued, the `scancel` line), ends with `REGBENCH_END ...
   status=refused cause=not_submitted whose=launcher`, and the next submitter of the
   chain tries again from the same files.

   **It never waits in silence, and never for ever.** Two waits could otherwise last
   until the link's time limit, with SOLVE and everything behind it waiting too:

   - *A queue that cannot be read* (`squeue` fails). Nothing is submitted, every such
     look is a line of the log, and after `REGBENCH_STAGE_SQUEUE_TRIES` (30) in a row,
     a poll interval apart, the submitter gives up.
   - *Tile tasks that do not leave the queue* (while it waits for room, and while it
     waits for an attempt to be over before listing the next). When none has left for
     `REGBENCH_STAGE_STALL_S` (3600) seconds it prints every tile task of the run still
     there with its state and the scheduler's reason (`squeue -o '%i|%j|%T|%r'`). If
     one of them has a reason it never leaves by itself (`REGBENCH_STAGE_HELD_REASONS`:
     `JobHeldAdmin`, `JobHeldUser`, `JobHoldMaxRequeue`, `DependencyNeverSatisfied`,
     `JobLaunchFailure`) it gives up at once; if none has, it waits on, says so, and
     prints the list again after the next stretch, until
     `REGBENCH_STAGE_STALL_MAX_S` (172800: two days) have passed without one task
     leaving, and then gives up. A stretch in which EVERY task still there is only
     waiting its turn (`REGBENCH_STAGE_WAIT_REASONS`: `Priority`, `Resources`,
     `QOSMaxJobsPerUserLimit`, `AssocMaxJobsLimit`, `BeginTime`, `Dependency`, `None`)
     is not counted in those two days: a busy cluster is no stall, and the submitter
     says each stretch what is waiting. On such a queue it waits until its own time
     limit, and the next link goes on.

   Giving up is the safer of the two ways out, and the one chosen: going on would list
   the next attempt, or write `.done`, over tiles that are still in the queue, and
   their rounds would then be read as the method's failures. The submitter writes
   `<run>-<dataset>.refused` (a `gave up:` line with the reason, how many tile tasks
   are still queued, the `scancel` line), ends with `status=refused
   cause=not_submitted whose=launcher` and exit 1, and LEAVES the queued tile tasks
   alone. The next link of the chain starts from the same files; when it gives up too,
   SOLVE and the last jobs run, and every round with a tile that is not done ends as
   `not_submitted`. Release or cancel the held tasks and submit again: what is ok is
   reused.
4. **The realcheck** of `valis_mirage` and `valis120_hires` (reported, never a gate): the
   first patient of the first dataset goes alone, on its own ladder
   (`rb-<run>-<ds>-first-<rung>g`), with `run --realcheck`, SUBMITTED BESIDE the rest of
   the run, which does not wait for it. What waits for the run waits for both.
5. **`settle`, per-case scoring and `tissue`, then `score`**, in the scorer's image
   (`variants.SCORER_IMAGE`), after everything, whatever failed. FIRST one small job,
   `rb-settle` (`score_cohort.sbatch settle`, 1 CPU, 4 GB, one hour), which waits for
   every ladder and every selfcheck with `afterany`: it asks the scheduler's accounting what
   became of every job of the submission and gives each result that still says `killed`
   the cause and the elapsed time the scheduler names (`regbench settle`: the last rung
   has no later task to ask). That rewrites those results' `run.json`, so every scoring
   job waits for it, and it leaves `logs/settled` naming the submission once it has
   done so. Then per dataset ONE ARRAY
   (`rb-score-case-<dataset>`, `score_cohort.sbatch case`), a task per case, runs
   `score-case`: the case's rows, kept as its record ("Records: a case is scored
   once"), its pair records computed side by side on the task's CPUs
   (`REGBENCH_SCORE_CPUS`, 8). Each task first makes the tissue-level pieces of its
   case's written images (`tissue --index`), so those are spread over the tasks too;
   `rb-tissue` runs AFTER the arrays and makes the pieces of any case whose task did
   not (two jobs never write the same pieces). `rb-score` waits for all of them with
   `afterany`: it reads the records that are
   valid, computes every case that has none (a task that failed, was killed or ran out
   of time left its finished pair records), and writes the tables. The tables do not
   depend on the array: they are the same bytes with it, with half of it, or without.
   Until 2026-10-12 the settling was done inside `rb-score`, AFTER the case tasks had
   scored those results: no table was ever built from a stale record (a case's record
   is read only when the bytes of every `run.json` it rests on are unchanged, so a
   settled result's case was scored again in the score job, from its kept pair
   records), but every such case was scored twice. Now `rb-score` settles only when
   `logs/settled` does not name its submission (the settle job was lost, or its node
   has no `sacct`), says so in its log, and scores again whatever case that changes
   (`test_score_records.py::test_a_record_made_before_a_result_was_settled_is_never_reused`,
   `test_submit_cohort.py::test_the_killed_results_are_settled_before_any_case_is_scored`).
   `score` is given
   `--submission` (below) and `--verdicts`: the scorer FIRST writes what every run's
   selfcheck and realcheck said (`regbench verdict` -> `<out>/logs/verdicts.tsv`, reading
   `logs/submitted/<submission>.runs` for the runs of the submission), and the tables
   then carry it: every row has its run's `selfcheck_status` from `verdicts.tsv`, and
   the summary's block "What the selfchecks said" lists each run's selfcheck and
   realcheck. No row is dropped.
6. **With `--compute-all`**, after the scorer: compute-all, the WHOLE pipeline traced per
   patient and registration backend (`slurm/compute_all.sbatch`, one Nextflow head job
   per run, `rb-compute-<backend>-<patient>`; `python -m benchmarks.regbench compute-all
   plan` lists them, every valis patient then every tiled one). The heads are dealt
   onto `REGBENCH_COMPUTE_LANES` lanes (default 2): each waits (`afterany`) for the head
   before it on its lane, and the first of each lane for the scorer of step 5. So they run
   LAST and ALONE: a cost measured under contention the benchmark made itself is not a
   cost. Needs `REGBENCH_INPUT`. "Full pipeline cost" has what is pinned and what is
   written.
7. **`rb-score-compute`**, the scorer again (`score_cohort.sbatch score`) after every
   head: the case records are kept, so it only adds `pipeline_cost*.csv` and their block
   of `summary.md`.

`--compute-only` submits steps 6 and 7 and nothing else, for a results folder whose
benchmark is done: the heads start at once (on their lanes) and the scorer scores the LAST
submission into `REGBENCH_OUT`, as `--score-only` does.

**The cohort (`multiplex`) is the default**, and the default submission is the cohort
alone. `--datasets` may also name a public dataset (`regbench.PUBLIC`), whose prepared
cases are under `REGBENCH_CASES_<NAME>` (the name in capitals, e.g. `REGBENCH_CASES_MCMICRO`;
refused up front when it is not set). Such a dataset gets its own run list (`runs --dataset
<name>`), and the scorer scores it into `<out>/tables/<name>/`. `--prepare` prepares the
cohort only; any other name is refused.

**One submission, one id, and no stale record.** Each call of the script has an id
(`REGBENCH_SUBMISSION`, default the time and the process id) that every job carries into
its records (`run.json: submissions`, `selfcheck.json: submission`). `REGBENCH_OUT`
defaults to the commit, so a second submission at the same commit writes into the same
folder; these rules keep one from answering for the other:

- a selfcheck removes the record in its folder before it runs and writes its own last; a
  later attempt (`--retry-memory`) and the verdict take a record of another submission as
  no record. A selfcheck attempt the scheduler lost, or that never started, is run again
  at the next of its three attempts, not read as `not_checked`;
- an attempt removes `realcheck.json` before the method starts, and the verdict
  counts a realcheck only beside an `ok` result of this submission;
- the ladder takes a result of another submission as no record;
- the scorer is given `--submission`: a result this submission neither ran nor reused is
  `not run`. A run this submission did not select, with older folders, is therefore not
  scored as if it had run.
  Reuse is the exception that is meant: a unit already `ok` for the same configuration is
  not run again, its records gain this submission's id, and it is scored
  (`REGBENCH_FORCE=1` runs everything again).

**What "the same" means for reuse: what was computed, not the command line.**

- *A unit* (`run.json: reuse_key`, `methods.reuse_key`) is reused only when, besides
  the method, label, options, cases and points, four things are equal: `execution`
  (`one process`; for STARE, whose registration is several stage processes, who
  starts them: `jobs` or `local`, as `REGBENCH_STAGES` states it), `code` (a sha256 of the driver,
  the driver's own modules that change what a result holds (`methods.DRIVER_CODE`:
  `cases.py`, `landmark_io.py`, `mapping.py`, `resources.py`), the adapter module, the
  modules of `methods/` it imports, the `regbench` modules it imports directly, the
  `benchmarks.*` modules it imports by absolute name and, for an adapter that runs
  outside code, its `CODE_FILES` with the `bin/utils` modules they import: see
  `methods.code_files`), `inputs` (for every case, the fingerprint of each image file it
  names) and `method` (what the adapter adds: STARE's version and the identity of its
  SOLVE job; for ASHLAR's anchored rows, the STARE matrix each round read). A result made by other code, in another execution shape or from other
  bytes is run again; so is a result of before the key existed.
- *A stage job* of a staged run (`stages/jobs/*.json: identity`,
  `methods.stare.Round.identity`) is reused only for the same command AND the same
  STARE version AND the same bytes of what it reads: the two slides for COARSE and a
  tile, the anchor and the tile plan for a tile, the anchor and every control file for
  SOLVE. A stage
  whose stage before it ran again and wrote other bytes is therefore run again too.
- *A file's fingerprint* (`cases.file_fingerprint`) is its size, its modification time
  and a sha256 over 16 blocks of 64 KiB spread evenly from its first byte to its last.
  A whole-slide file is tens of gigabytes and a tile job reads a few megabytes of it,
  so hashing it whole in every job would cost more than the job. Any rewrite changes
  the time; a file put back with the same size and time is caught where it differs
  inside a sampled block. **Residual risk:** a file of the same size and the same
  modification time that differs only between the samples is taken for the same file.
  The small files a stage writes (anchor, plan, control files) are hashed whole.
- `REGBENCH_FORCE=1` holds at every attempt: a forced submission reuses an `ok` stage
  record only when it made that record itself, so a first attempt lost to the scheduler
  does not let a later attempt fall back on an earlier submission's record.
- **Not in any key:** the version of a package inside an image, except STARE's. After
  rebuilding or re-pulling an image, submit with `REGBENCH_FORCE=1`.
- A unit the scheduler lost (`ladder.mark_lost`) gets an `infrastructure` record for
  the current submission. A record another submission left in that folder is not
  destroyed: `run.json`, `warped.npz` and `native/` (the method's own image) are moved
  to `<result>/earlier/<n>/` and the new record names the folder (`earlier_result`).
  The current submission's row reads `infrastructure`, not that earlier result; moving
  the three back restores it. The rest of the result folder (the tissue plane, stage
  files) is not kept: a re-run overwrites it, and it is made again from the image.
- A unit whose input images cannot be read (`methods.check_inputs`) is not run and
  touches no `ok` record: those stay as they are until the inputs can be read and are
  known to have changed. Its other rows read `infrastructure`. An `ok` record of another
  submission that a re-run would overwrite is moved to `<result>/earlier/<n>/` first,
  as above.

- *The method code is frozen* (plan F, Task 8).
  `benchmarks/tests/regbench/method_code_freeze.json` holds, for every method in
  `benchmarks.regbench.METHODS`, the `code` fingerprint of `methods.code_fingerprint` and
  the files it covers (`methods.code_files`, paths from the repository root);
  `test_method_code_freeze.py` fails, naming the method and the files, when a method has
  another fingerprint. **A frozen file changes only in a commit of its own that
  regenerates the freeze file** (`python -m
  benchmarks.tests.regbench.make_method_code_freeze >
  benchmarks/tests/regbench/method_code_freeze.json`) **and says why**: every cluster
  result of the methods that file belongs to will run again. A method added later is
  appended by its own task. The files common to all methods are `cases.py`,
  `landmark_io.py`, `mapping.py`, `resources.py` and `methods/__init__.py`; the rest, per
  method (the adapter, its helpers, `imageio.py`, `bin/register.py` and the `bin/utils`
  modules it runs, the tile cutting of `benchmarks/ashlar`), is the `files` list of that
  method in the freeze file, which the generator prints.

**Reuse: what is computed again, and why.** A unit result is reused only when its
`run.json: reuse_key` equals the key the current code computes. `code` is a sha256 over
the driver `methods/__init__.py` and `methods.DRIVER_CODE`, the adapter, the `methods/`
modules it imports, the `regbench` and absolute `benchmarks.*` modules it imports, and
the outside code it runs (`valis_mirage`: `bin/register.py` and the `bin/utils` modules
on its path; ASHLAR: `benchmarks/ashlar/` and the tile cutter); `billing.py`,
`plugins.py` and the reporting modules are left out because they change cost
accounting or reports, not a result (`methods.DRIVER_CODE_LEFT_OUT`); `inputs` is the fingerprint of every image the
cases name; STARE adds its version and its SOLVE identity. So **any byte change to one
of those files makes every earlier result of that method run again**, and results made
before plan E (on the arms' inputs) are not reused by this branch's code. Two finer
caches survive code changes: a STARE stage job (COARSE, a tile, SOLVE) is reused when its
command, STARE's version and the bytes it reads are equal (`stages/jobs/*.json:
identity`, no code hash), and a pair record of the scorer is reused when `pairs.py`,
`cases.py`, `mapping.py`, `cell_pairs.py` and the `PAIR_CODE_MEMBERS` are unchanged; its
full-pairs record (`full_pairs/`, Task 21b) when, besides, `full_pairs.py` is unchanged
(`score.full_key`). `regbench reuse` reports results, not scorer records, so it names
neither.
There is no switch that ignores a code change.

`python -m benchmarks.regbench reuse --cases C --out O [--dataset multiplex] [--csv PATH]`
reports, read-only, for every finished result whether the launcher would reuse it now
(`reusable`: `yes`/`no`) and which part differs (`differs`: any of `execution`, `code`,
`inputs`, `method`, `points`, `record`; `record` = not `ok` or no `warped.npz`). A method
whose adapter cannot be loaded on this machine (a stand-in, a plug-in that is not enabled)
is `no` with `code` and the detail `key not computable here`. The last line is
`# reusable: N of M ok results`. It writes nothing under `--out`. It reports **per
result, not per unit; `opts` and a requested realcheck are not compared**: the launcher
reuses a unit only when every variant of every case of it is `ok` (with the same
`group_cases`, `opts` and, if asked, a realcheck), so a row whose sibling variants
include a `no` carries the detail `sibling variant not ok`. The per-record test is
`methods.record_gaps`, the same function the launcher's reuse uses.

A GPU is asked for only for the runs whose `gpu` column is 1, with the site's own
request string, and the script refuses to submit them without it. An image is given the
GPU (`--nv`) only for those runs. VALIS 1.2.0's image is started with `--writable-tmpfs`.
No job asks for more than 7 days: a run that works per patient gets `REGBENCH_TIME`
times the rounds of the largest patient, capped there ("Cost" says what that means per
round).

### What a job prints, and `regbench status`

Every regbench job prints one `[regbench] start` line first and one `[regbench] end`
line just before its `REGBENCH_END` line (which stays the log's last), in one format
(`slurm/env.sh`: `regbench_start`, `regbench_say_end`; fleet wave 2, B1):

```
[regbench] start kind=<kind> job=<name>-<id> run=<run|-> label=<label|-> dataset=<ds|-> unit=<unit|-> rung_gb=<GB|-> attempt=<n|-> [<key>=<value> ...] out=<REGBENCH_OUT> commit=<sha> host=<node> at=<time>
[regbench] end kind=<kind> job=<name>-<id> outcome=<ok|status/cause> wall=<s>s peak_rss=<MB>MB|- record=<path|-> exit=<code>
```

`peak_rss` is the run's own peak (`run.json: resources.peak_rss_mb`, carried on its end
line) or the job cgroup's peak where it can be read from inside the job, else `-`. A job
that ends without its end line (a refusal under `set -e`, a signal) still prints its
`[regbench] end` line from the EXIT trap (`outcome=exit_<code>`). To find every job that
did not end ok:

```bash
grep -h '^\[regbench\] end' "$REGBENCH_OUT"/logs/*.log | grep -v ' outcome=ok '
```

`python -m benchmarks.regbench status --out "$REGBENCH_OUT" [--cases C] [--dataset D]
[--runs "a b"] [--csv]` (read-only, `status.py`; B2) prints, per run, result name and
dataset, how many results are expected (the case list), `ok` (and of those `reused` by
the last submission), `failed` (with the causes), `missing`, and two counts the scorer
would NOT load for the submission, so that `ok` agrees with the tables: `other_sub`
(an ok record of another submission, not reused by this one) and `stale` (made for
points or landmarks the case no longer has). Then the folder's jobs still in the queue,
matched by the job ids in `logs/jobs.tsv` (not by name: another folder names its jobs
the same), when `squeue` is there (`# queue: not read (no squeue on this node)`
otherwise); the chains no watcher finished and none of whose watcher links is still
queued; and a last line of what to do next: wait, or submit the FULL run list again
into the same folder (`REGBENCH_OUT=... submit_cohort.sh`, which reuses every ok unit).
A `--runs` subset needs its own `REGBENCH_OUT`: submitted into this one, it would leave
every other run `not run` in the tables. `--csv` gives the table as CSV.

When `REGBENCH_OUT` is the commit-hash default and an earlier default-named folder under
`~/regbench_results/` holds results of the same cases (`<out>/logs/cases`, written by
every launch), the launcher says so loudly with the `export REGBENCH_OUT=...` that
reuses it (B3). The default itself is unchanged.

The nuclei step (`--prepare`) is submitted only for the slides that have anything to do
(`nuclei --pending`; F9), and has a memory ladder (128, then 256 and 512 GB, by its own
watcher; F7): a StarDist run that fails is written down (`nuclei --failed`), so the
slide's record and `prepare multiplex`'s drop reason name the exit code instead of
"outlines not traced yet".

### What to export

| variable | what | default |
|---|---|---|
| `REGBENCH_CASES` | prepared cases | `$HOME/regbench_cases` |
| `REGBENCH_OUT` | results, logs | `$HOME/regbench_results/mirage-<commit>` |
| `REGBENCH_BIND` | paths to bind into every image, comma-separated: where the cohort's slides and nuclei live | none |
| `REGBENCH_WORK` | scratch for a method's intermediates | `$TMPDIR` |
| `REGBENCH_SIF_STARE_REGBENCH`, `REGBENCH_SIF_VALIS_MIRAGE`, `REGBENCH_SIF_VALIS120`, `REGBENCH_SIF_DEEPERHISTREG`, `REGBENCH_SIF_ASHLAR`, `REGBENCH_SIF_PALOM` | the six images of the run list (a seventh, `REGBENCH_SIF_STARDIST`, is for `--prepare` only); the script refuses to submit when one a selected run needs is missing. Local `.sif` paths: "Before the first real submission" | none |
| `REGBENCH_SBATCH_GPU` | what a task of the four GPU runs asks for: one GPU. Export the site's own form where it names the card, e.g. `--gres=gpu:nvidia_h200:1`; set empty, the GPU runs are refused | `--gres=gpu:1` |
| `REGBENCH_TIME` | per round, every rung | `24:00:00` |
| `REGBENCH_SELFCHECK_TIME` | every selfcheck attempt | `24:00:00` |
| `REGBENCH_ARRAY_LIMIT` | tasks of one array running at once (`%N`) | `20` |
| `REGBENCH_ACCT_WAIT_S`, `REGBENCH_ACCT_POLL_S` | how long a ladder task waits for the scheduler's accounting to name the end state of an earlier task whose result still says `killed`, and the seconds between two looks ("The ladder") | `600`, `30` |
| `REGBENCH_ARRAY_LIMIT` (above), `REGBENCH_STAGE_ARRAY_MAX`, `REGBENCH_STAGE_QUEUE_MAX`, `REGBENCH_STAGE_FEED_MIN`, `REGBENCH_STAGE_POLL_S`, `REGBENCH_STAGE_FEEDERS` | must be whole numbers of at least 1: the script refuses anything else before it submits | |
| `REGBENCH_STAGE_ARRAY_MAX` | the most tasks in one tile array of a run that is several jobs per round | `100` |
| `REGBENCH_STAGE_QUEUE_MAX` | the most tile tasks pending or running at a time in the WHOLE LAUNCH: a total, divided by the number of runs with a tile submitter that feed at once (every STARE run is such a run: 9 on the cohort; `REGBENCH_RUN_LANES` when that is fewer, since at most that many runs are in flight; floor, at least 1 each) and each submitter keeps to its share (`REGBENCH_STAGE_QUEUE_RUN` inside the submitter, which the launcher sets; do not set it yourself). It was a per-run ceiling of 100 when two runs had a submitter; the default total is 200 = 2 x 100, so the cluster load of a launch is what it was, however many runs it holds | `200` |
| `REGBENCH_STAGE_FEED_MIN` | free slots the tile submitter waits for before it submits an array (fewer when fewer tiles are left) | `25` |
| `REGBENCH_STAGE_POLL_S` | seconds between two looks at the queue | `60` |
| `REGBENCH_STAGE_FEEDERS`, `REGBENCH_STAGE_FEED_TIME` | tile submitters chained per run and dataset, and the time limit of each | `2`, `7-00:00:00` |
| `REGBENCH_STAGE_SUBMIT_TRIES`, `REGBENCH_STAGE_ARRAY_SPAN` | read by the tile submitter: tries of a refused `sbatch`; the widest span of indices in one tile array (below the site's `MaxArraySize`) | `5`, `1000` |
| `REGBENCH_STAGE_SQUEUE_TRIES` | read by the tile submitter: looks in a row at a queue that cannot be read before it gives up | `30` |
| `REGBENCH_STAGE_STALL_S`, `REGBENCH_STAGE_STALL_MAX_S` | read by the tile submitter: seconds without a tile task leaving the queue before it lists what is left (and gives up on a held one); and before it gives up whatever the reasons | `3600`, `172800` |
| `REGBENCH_STAGE_WAIT_REASONS` | the scheduler's reasons the tile submitter takes for "only waiting its turn": while every tile task left has one of them the time is not counted towards `REGBENCH_STAGE_STALL_MAX_S`; empty counts every stretch | `Priority Resources QOSMaxJobsPerUserLimit AssocMaxJobsLimit BeginTime Dependency None` |
| `REGBENCH_STAGE_HELD_REASONS` | the scheduler's reasons the tile submitter takes for "does not leave the queue by itself", space-separated | `JobHeldAdmin JobHeldUser JobHoldMaxRequeue DependencyNeverSatisfied JobLaunchFailure` |
| `REGBENCH_RUN_ORDER` | the order the runs are SUBMITTED in (and dealt onto the lanes): family names (`stare`, `valis_mirage`, `valis120`, `deeperhistreg`, `ashlar`, `affine_ashlar`, `palom`: every run of the family) or run names, space-separated; the runs not named follow in the run list's order. A run that reads another's results (the two `affine_ashlar` rows read `stare`'s) is never submitted before it: it is put directly after it and a `# run order:` line says so. An unknown name is refused with the list. It changes the order of submission and nothing else: not `runs`, not a table, not a result | `default` on the cohort: the two ASHLAR runs, then STARE and the two affine ASHLAR runs that read it, then the rest; on a public set STARE first (`runs --order default --dataset <name>` prints it, and the launcher a `# run order (<name>):` line). Set empty: the run list's order |
| `REGBENCH_RUN_LANES` | runs in flight at once: the selected runs are dealt onto that many lanes in the order of the run list, and a run starts when the run before it on its lane has ended, whatever became of it. With `REGBENCH_ARRAY_LIMIT` this bounds the tasks running at once (lanes x limit, plus the selfchecks, which are not laned) | `0` (no limit) |
| `REGBENCH_FORCE` | `1`: run every unit again, also those whose results are all `ok` already | `0` (reuse) |
| `REGBENCH_SUBMISSION` | this submission's id. Set it only with `--score-only` (the submission to score) or to run a unit by hand under a submission's id: the launcher refuses an id the folder already holds a submission of | time and process id |
| `REGBENCH_SCORE_WHILE_QUEUED` | `1`: `--score-only` although jobs of an earlier call into the folder are still pending or running | `0` (refused) |
| `REGBENCH_MANIFEST` | with `--prepare`: the cohort's `rounds.csv` | none |
| `REGBENCH_PREPARE_ARGS`, `REGBENCH_PREPARE_FORCE`, `REGBENCH_PREPARE_MEM`, `REGBENCH_PREPARE_TIME` | with `--prepare`: more `prepare multiplex` options; `1` to redo cases that exist; the allocation | none, `0`, `64G`, `24:00:00` |
| `REGBENCH_NUCLEI` | with `--prepare`: where each slide's bare nuclei are kept ("Outlines and bare nuclei") | `$REGBENCH_CASES/_nuclei` |
| `REGBENCH_STARDIST_MODEL`, `REGBENCH_STARDIST_MODEL_DIR` | with `--prepare`: the StarDist model the pipeline's QC runs (the pipeline's `segmentation_model` and `segmentation_model_dir`, from the site's config); the folder is bound into the images | the site config's `segmentation_model` and `segmentation_model_dir`; a name is required from one of the two |
| `REGBENCH_SITE_CONFIG` | with `--prepare`: the site config the StarDist model and any override of the StarDist settings are read from when the two variables above are unset; `none` reads no file | `conf/ieo.config` of the checkout, if it is there |
| `REGBENCH_SIF_STARDIST` | the pipeline's StarDist image (`bolt3x/mirage-stardist:1.0.0`), for the re-run only | none |
| `REGBENCH_STARDIST_CPU`, `REGBENCH_STARDIST_PROB_THRESH` | `1`: StarDist without a GPU (else the array asks for `REGBENCH_SBATCH_GPU`); a detection threshold, only if the site's config sets one | `0`, none |
| `REGBENCH_NUCLEI_FALLBACK` | `shrink`: the shrunk outline for a slide still without nuclei (the nuclei step itself is never left out: it traces the outlines) | none |
| `REGBENCH_INPUT`, `REGBENCH_INPUTS` | `--inputs`: the cohort's samplesheet; the input folder (preprocess/, rounds.csv, outlines/) | none; `$REGBENCH_ROOT/inputs` (launch.sbatch) |
| `REGBENCH_NF_PROFILE`, `REGBENCH_INPUTS_TIME`, `REGBENCH_INPUTS_FORCE` | `--inputs` and compute-all: the pipeline's profile; the head job's time; `1` runs the input step again over an earlier one (moved aside) | `singularity,ieo`, `2-00:00:00`, `0` |
| `REGBENCH_COMPUTE_LANES`, `REGBENCH_COMPUTE_TIME` | `--compute-all` / `--compute-only`: compute-all heads in flight at once (each is a Nextflow head whose processes are jobs of their own); the time limit of one head | `2`, `7-00:00:00` |
| `REGBENCH_NUCLEI_CPUS`, `REGBENCH_NUCLEI_MEM`, `REGBENCH_NUCLEI_TIME` | the allocation of one slide's task | `8`, `128G`, `08:00:00` |
| `REGBENCH_SCORE_CPUS` | CPUs of one task of the per-case scoring arrays: a case's pair records are computed on that many processes. The `tissue` and `score` jobs keep 4 | `8` |
| `REGBENCH_SCORE_MEM` | memory of one task of those arrays, and of the `tissue` and `score` jobs | `64G` |
| `REGBENCH_SCORE_TIME` | time limit of one task of those arrays (never above 7 days). A task that runs out keeps the pair records it finished; `--score-only` goes on from there | `24:00:00` |
| `REGBENCH_HEAD_IMAGE` | the image the script itself reads the run list with, on the login node | `STARE_REGBENCH` |
| `REGBENCH_WRITABLE_TMPFS_IMAGES` | images started with `--writable-tmpfs` | `VALIS120` |
| `REGBENCH_STARE_SRC_IMAGES` | images that have no `stare` package inside: the pinned STARE source is put on their `PYTHONPATH` | `ASHLAR VALIS_MIRAGE` |
| `REGBENCH_STARE_SRC` | where that source is; fetched once by the launcher on the login node when it is not there, before the first `sbatch` of every real submission that uses such an image as a `.sif`, with or without the preflight | `$REGBENCH_OUT/cache/stare-<tag>/src` |
| `REGBENCH_PREFLIGHT` | `0`: submit without the preflight (a login node without the container runtime); said aloud. The STARE source is fetched all the same | `1` |
| `REGBENCH_PRIVATE`, `REGBENCH_PRIVATE_DIR` | the private plug-in's switch and folder (`plugins.py`); passed into every image, the folder bound | unset |

Partition, account and node type go after `--` and reach every job, e.g.
`-- --partition=cpu --constraint=<one node type>`. Pin one node type, or the timings
compare CPUs and not methods. The rungs, the 4 CPUs and the 7-day limit are not
variables: they are frozen (`ladder.py`).

### Before the first real submission

None of this is checked by the script; each is a way a launch goes wrong that no job
reports.

**The cost instruments, on the first real rows (T31; only the cluster can say).** Read
the first scored `resources.csv` / `summary_resources.csv` before trusting any cost
figure; each item names the column that answers it.

- [ ] **C1. The CPU counter** is the step cgroup's: `cpu_instrument` is `cgroup_v2` or
  `cgroup_v1`; inside `apptainer exec`, `/proc/self/cgroup` shows a `step_batch` path
  and its `cpu.stat` can be read.
- [ ] **C2. Confinement and allocation**: `cpus_affinity == cpus_requested` (the
  `cores_confined` header line now counts stage jobs too); with hyperthreads, how many
  physical cores 4 logical CPUs are; whether a 1-CPU SOLVE or a 2-CPU tile is rounded
  up (`AllocCPUS` in `logs/accounting_stages.psv` against `cpus_requested`).
- [ ] **C3. The threads each library started** under that cpuset (`workers_seen`,
  `torch_num_threads`, `dask_cpu_count`, `cv2_num_threads`, `vips_concurrency`, the
  JVM's `availableProcessors`), and `cpu_efficiency` per stage job
  (`stare_stage_peaks.csv`: `coarse_/tile_/solve_cpu_efficiency`).
- [ ] **C4. sacct**: steps rolled into the job line or not (the parser copes either way),
  and `TotalCPU` non-zero for the batch step; `AllocCPUS`, `ReqMem`, `MaxRSS` filled
  (`cpu_h_reserved_scheduler`, `gb_h_reserved_scheduler`; blank and counted in
  `reservation_jobs` when not).
- [ ] **C5. What `memory.peak` measures**: task or job level, page cache in or out
  (`cgroup_peak_gb`, `stage_cgroup_peak_gb`, `driver_job_peak_gb`).
- [ ] **C6. GPU**: `torch.cuda.device_count() == 1` under `--gres=gpu:1` (else `gpu_h`
  is multiplied); the GPU nodes' CPU model (`cpu_model`).
- [ ] **C7. Mixed hardware**: `cpu_model` per method and row (`stage_cpu_model` for
  STARE's stage jobs); GPU and CPU rows on different generations.
- [ ] **C8.** `OMP_NUM_THREADS=1` reaches the ASHLAR job (`threads_env`).
- [ ] **C9. STARE's per-job overhead**: `cpu_h_reserved_scheduler` against
  `cpu_h_reserved` on the staged rows.
- [ ] **C10. VALIS's heap**: the heap a rung asked for against what the JVM got.
  `jvm_peak_heap_gb` is `NA` (no instrument reads it: a label, not a figure); until one
  does, `peak_rss_gb` of a JVM row is bounded by `-Xmx` (the `JVM_RSS` header line). The
  heap is inside the in-process RSS, and reading it would move the VALIS and ASHLAR
  fingerprints: NA is the controller's ruling (T31, 2026-10-19).
- [ ] **C11. The tile throttle's cost**: `elapsed_with_queue_s` against
  `wall_unlimited_parallel_s`, `peak_cpus_at_once` against `n_tiles x 2`.
- [ ] **C12.** How often compute-all's traces leave `%cpu` blank (`pipeline_cost.csv`).
- [ ] **C13. The GPU partition's rungs** (`GRANT_GPU_MB`): DeeperHistReg's GPU ladder may
  end below 700 GB where its CPU ladder does not (`n_capped`).
- [ ] **C14. The OOM watcher** can be read (`oom_watch`): it decides whether a kill is
  `wasted` or `killed_unsettled` (billed from the scheduler, `wasted_from_scheduler`).
- [ ] **Contention** (R14): the `contention` header line (`load_start`/`load_end`,
  `other_jobs_start` of the row's own job, `load_max_attempts`/`other_jobs_max_attempts`
  over every attempt with `telemetry_attempts` = how many left a file; a stage job asks
  no `squeue`; `telemetry` yes/partly/no). Wall
  times are on shared nodes; the headline is CPU-hours. `REGBENCH_WAVES=1` gives a
  timing replicate with one method on the nodes at a time.

- **A checkout of its own, left alone until the tables exist.** Every job imports the
  harness from the checkout at the moment it starts (`REGBENCH_REPO` is bound into the
  image and put on `PYTHONPATH`), not from a copy made at submission. A `git pull`, a
  branch switch or an edit while jobs are queued changes what the later jobs run, and
  the code fingerprint in each result's `reuse_key` then differs between units of one
  submission. Clone the branch into a folder nothing else uses, submit from there, and
  touch it again only after `tables/summary.md` is written.
- **`REGBENCH_CASES`, `REGBENCH_OUT` and `REGBENCH_WORK`, exported and created, on
  scratch.** The defaults are under `$HOME` (`REGBENCH_OUT` with the commit in its name,
  so it moves with the checkout) and, for `REGBENCH_WORK`, each job's `TMPDIR`, which
  the preflight cannot check. A cohort does not fit a home quota:

  ```bash
  export REGBENCH_CASES=<scratch>/regbench/cases
  export REGBENCH_OUT=<scratch>/regbench/out
  export REGBENCH_WORK=<scratch>/regbench/work
  mkdir -p "$REGBENCH_CASES" "$REGBENCH_OUT" "$REGBENCH_WORK"
  ```

- **The images as local `.sif` files, pulled ONE AT A TIME** on a node with network,
  before anything is submitted (never by a job, and never two pulls at once into one
  cache). The names are the ones this repository builds or pins: the four it builds
  (`containers/README.md`), the pipeline's StarDist image, ASHLAR's upstream tag, and
  the pipeline's VALIS 1.0.0 by the digest of `modules/local/register.nf`:

  ```bash
  mkdir -p <scratch>/regbench/sif && cd <scratch>/regbench/sif
  apptainer pull stare-regbench.sif docker://bolt3x/mirage-stare-regbench:1.0.0
  apptainer pull valis120.sif       docker://bolt3x/mirage-valis120:1.0.0
  apptainer pull deeperhistreg.sif  docker://bolt3x/mirage-deeperhistreg:1.0.0
  apptainer pull palom.sif          docker://bolt3x/mirage-palom:1.0.0
  apptainer pull ashlar.sif         docker://bolt3x/mirage-ashlar120:1.0.0
  apptainer pull valis-mirage.sif   docker://bolt3x/mirage-valis100:1.0.0
  apptainer pull stardist.sif       docker://bolt3x/mirage-stardist:1.0.0
  export REGBENCH_SIF_STARE_REGBENCH=$PWD/stare-regbench.sif
  export REGBENCH_SIF_VALIS120=$PWD/valis120.sif
  export REGBENCH_SIF_DEEPERHISTREG=$PWD/deeperhistreg.sif
  export REGBENCH_SIF_PALOM=$PWD/palom.sif
  export REGBENCH_SIF_ASHLAR=$PWD/ashlar.sif
  export REGBENCH_SIF_VALIS_MIRAGE=$PWD/valis-mirage.sif
  export REGBENCH_SIF_STARDIST=$PWD/stardist.sif      # --prepare only
  ```

  (`apptainer pull <file> docker://<reference>`: the form of Apptainer's user guide
  `[MEMORY]`, not retrieved while this was written; `singularity pull` takes the same
  arguments. None of the seven pulls was run from here.) `submit_cohort.sh --preflight`
  then starts each of them as a job would.
- **Never an `--export=` after `--`.** Every job is submitted with `--export=ALL`, which
  is how `REGBENCH_*` reaches it, and the caller's own options come after it on the
  `sbatch` line: an `--export=NONE` or `--export=VAR=...` there replaces it, and the
  jobs start without their paths and images.
- **Or as one job, everything set inside it.** `sbatch benchmarks/regbench/slurm/launch.sbatch [the launcher's arguments]` runs the launcher on a compute node (`--export=ALL`); its log is `regbench-launch-<job id>.log`. It can be submitted from any folder: the repository is found from the path given to `sbatch`, the benchmark lives in `regbench/` under the folder submitted from, `--inputs` submits the input step (one Nextflow head job, no image of the benchmark needed), and `--prepare` writes `$REGBENCH_INPUTS/rounds.csv` from it unless `REGBENCH_MANIFEST` is exported. The SETTINGS block at the top of the script is the one place to edit: `REGBENCH_ROOT` (one folder holding `inputs/`, `cases/` and `out/`), the lanes (4), the array limit (20) and the tile queue (100, a total over the launch's stage-chain runs). The site is read from the site config when the job runs (`REGBENCH_SITE_CONFIG`, default `conf/ieo.config` of the checkout): the GPU request from `gpu_type`, the bound folders from the `--bind` entries of `runOptions`, the image folder from `cacheDir` (`REGBENCH_IMAGES`): each image is looked for there under the name Nextflow gives a pulled image (or the names above, also in `regbench/` under it), and one that is missing is pulled into it before the launcher starts. Without every image, or without a container runtime, the job stops with a line and the python on PATH is never used; nothing of it is written in the script. A variable exported before `sbatch` wins over the script's line. The node needs the scheduler's commands and the container runtime (a missing runtime stops the launch with a line saying so), and the pinned STARE source must already be there (one `--preflight` from a login node fetches it). Never run on a cluster.
- **Inside `tmux` (or `screen`).** A real submission starts every image once (the
  preflight) and then makes several hundred `sbatch` calls, one after the other; it
  takes minutes. A login session that drops in the middle ends the script with part of
  the graph queued. The ids of what was queued are then in
  `$REGBENCH_OUT/logs/submitted/<id>.ids`, the newest file of that folder (each id is
  written there as soon as its `sbatch` returns): `scancel $(cat <that file>)`, and
  submit again.
- **`REGBENCH_STAGE_QUEUE_MAX` is a total for the submission, split over the runs.** Each
  run whose registration is several jobs per round (every STARE run, 9 on the cohort)
  has a tile submitter of its own; a run's datasets follow one
  another, so a run is one submitter's worth of queue. The launcher divides the total by
  the number of such runs that can FEED AT ONCE: the runs it submits, or
  `REGBENCH_RUN_LANES` when that is fewer (floor, at least 1), and gives every submitter
  that share as its ceiling. Without lanes: at the default 200, 22 each on the cohort (9
  runs). With lanes (`launch.sbatch`: 4 lanes, a
  total of 100): 25 each. The total stays a ceiling because a laned run's jobs, its
  feeders included, wait for the LAST job of the run before it on its lane, and that job
  waits for that run's last feeder, which ends only once each of its tile tasks has left
  the queue: at most `lanes` submitters feed at a time. (Not so when a submitter GIVES UP
  or is ended at its time limit with tasks still queued: those tasks stay, uncounted by
  the next run on the lane.) A `--runs` subset divides by the runs in it. A minimum of 1
  each means a total below that number is exceeded by the difference. The dry run prints
  the share (`# tile queue: ...`) and, in its last lines, the sum to hold against the
  site's limit.
- **A subset submitted into a results folder makes every other run `not run` there.**
  The scorer takes what the LAST submission into `REGBENCH_OUT` ran or reused and
  nothing else (`--submission`). `--runs "stare ashlar"` into a folder that holds a
  full cohort leaves, in the new tables, every other run as `not run`; the earlier
  results stay on disk, and the earlier tables are replaced. Give a subset its own
  `REGBENCH_OUT`, or submit the full run list again (every unit that is ok is reused,
  not run).
- **The scheduler's state says nothing about the outcome.** The registration tasks
  (`run_rung.sbatch`, `stage.sbatch`) exit 0 whether the method succeeded, failed or
  was skipped, so `sacct` shows them `COMPLETED` in all three cases; a state other
  than that (`OUT_OF_MEMORY`, `TIMEOUT`, `NODE_FAIL`, `CANCELLED`) is the scheduler's
  own verdict. What became of a unit is in its `run.json` (`ok`, `cause`) and in the
  last line of its log: `grep -h REGBENCH_END "$REGBENCH_OUT"/logs/*.log`.
- **Inodes.** A tile task leaves four small files: its log
  (`logs/rb-<run>-<ds>-tile-...log`), its end line (`logs/end/`), its job record
  (`stages/jobs/tile_<row>.json`) and the tile's control file. With T tile tasks
  (the dry run prints T) that is about 4 x T files under `REGBENCH_OUT`, for each of
  the two staged runs: check the scratch file system's inode quota against it, not
  only its bytes.

### In order, the first time

```bash
cd <checkout of this branch>
source benchmarks/regbench/slurm/env.sh        # after exporting the variables above

# 1. the input step (REGBENCH_INPUT = the samplesheet), then the cohort's list (standard
#    library only: the login node's python will do)
benchmarks/regbench/slurm/submit_cohort.sh --inputs -- <site options>   # wait for it
python3 -m benchmarks.regbench.manifest --preprocessed "$REGBENCH_INPUTS/preprocess/results/csv/preprocessed.csv" \
    --outlines "$REGBENCH_INPUTS/outlines" --out rounds.csv

# 2. a walking skeleton: ONE patient, the cohort's cases only
awk -F, 'NR==1 {print; next} NR==2 {p=$4} $4==p' rounds.csv > one.csv
REGBENCH_MANIFEST=$PWD/one.csv benchmarks/regbench/slurm/submit_cohort.sh --prepare --datasets multiplex -- <site options>
#    (this first runs the bare-nuclei array, one task per slide: it needs
#    REGBENCH_STARDIST_MODEL, REGBENCH_STARDIST_MODEL_DIR, REGBENCH_SIF_STARDIST and the GPU
#    request; it traces the outlines every case needs)
#    ok: no DISAGREEMENT line in $REGBENCH_OUT/logs/rb-nuclei-*.log, `regbench_py
#        STARE_REGBENCH nuclei --manifest one.csv --out "$REGBENCH_NUCLEI" --list` says
#        `usable` and `True` (exact) for every slide; the prepare log ends without a
#        refused case, and
regbench_py STARE_REGBENCH list --cases "$REGBENCH_CASES" --dataset multiplex --sizes
#    prints "<rounds> 1 <rounds>"

# 3. STARE and ASHLAR on that patient: the images first (nothing is submitted), then
#    what would be submitted, then for real
benchmarks/regbench/slurm/submit_cohort.sh --preflight --runs "stare ashlar" --datasets multiplex
#    ok: no "preflight FAIL" line, and "# preflight: ok"
benchmarks/regbench/slurm/submit_cohort.sh --dry-run --runs "stare ashlar" --datasets multiplex -- <site options>
benchmarks/regbench/slurm/submit_cohort.sh --runs "stare ashlar" --datasets multiplex -- <site options>
#    (the dry run ends with "# N job(s) ..., M task(s) in them: N sbatch call(s) would
#    be made now", then "# later, by the tile submitters: T tile task(s) ..." and
#    "# the most in the queue at once: ... task(s)": that last number must be within the
#    site's submit limits, "What the site must allow")
#    after it: $REGBENCH_OUT/logs/verdicts.tsv says what each run's selfcheck said (reported,
#    never a gate)
#    ok: every $REGBENCH_OUT/<stare|ashlar>/multiplex/<case>/run.json has "ok": true;
#        every $REGBENCH_OUT/stare/multiplex/<case>/stages/jobs/ holds coarse.json,
#        solve.json, stitch.json and one tile_<row>.json per tile, each "ok": true, and
#        that run.json has info.jobs.execution "jobs", n_tiles = the tile count and
#        register_s = info.jobs.wall_unlimited_parallel_s;
#        $REGBENCH_OUT/tables/summary.md exists and its cost block opens with
#        "cores confined: yes"

# 3b. only if the scoring has to be done again (a score job that ran out of time, a
#     changed scorer): the scoring alone, for the registrations that are there. No
#     selfcheck or registration is submitted; a case whose record is still valid
#     is not computed again
benchmarks/regbench/slurm/submit_cohort.sh --score-only --datasets multiplex -- <site options>

# 4. the whole cohort: prepare (a NEW $REGBENCH_CASES), look at the dry run, submit
REGBENCH_MANIFEST=$PWD/rounds.csv benchmarks/regbench/slurm/submit_cohort.sh --prepare -- <site options>
benchmarks/regbench/slurm/submit_cohort.sh --preflight
benchmarks/regbench/slurm/submit_cohort.sh --dry-run -- <site options> | less
benchmarks/regbench/slurm/submit_cohort.sh -- <site options>

# 5. the cost of the whole pipeline, per patient and backend (steps 6 and 7 of the job
#    graph), once the benchmark is scored: alone on its lanes, then scored again
#    (REGBENCH_INPUT = the samplesheet)
benchmarks/regbench/slurm/submit_cohort.sh --compute-only -- <site options>
#    (or --compute-all in the main submission: the heads then wait for its scorer)
```

`--score-only` scores what the LAST submission into `$REGBENCH_OUT` ran or reused: its
id is kept in `$REGBENCH_OUT/logs/submission`. For a folder filled before that file was
written, the script takes the id its selfcheck records carry when they all carry the
same one, and otherwise asks for `REGBENCH_SUBMISSION=<id>` (the id is printed by every
submission as `# submission  -> ...` and is in every `run.json`, key `submissions`). The
records are parsed, each `logs/selfcheck/<run>/selfcheck.json` by the reader
`verdict` uses (`python -m benchmarks.regbench submissions --out OUT`, `ladder.submissions_of`);
until 2026-10-12 the id was searched for as text. A record that cannot be read, or that
names no submission (a selfcheck run by hand), is evidence of none. When the id being
scored, exported or read from `logs/submission`, is one no selfcheck record carries
while records of other submissions are there, the script says so in a `# WARNING` line
and goes on with the id it was given: a file of another submission is never taken for
it. It submits `rb-settle`, the case arrays, `rb-tissue` and `rb-score`; it reads the
selfcheck records that submission left, and it needs `REGBENCH_SIF_STARE_REGBENCH` and
nothing else.

`rounds.csv`'s fourth column is the patient (`group`); the `awk` line keeps the first
patient's rounds. `--runs` takes any subset of run names; the selfchecks a subset stands
behind are submitted with it. A run whose results another reads but which is not in the
subset is assumed to be under `$REGBENCH_OUT` already, and the script says so.

### The CPU cross

4 CPUs is every method's allocation. The CPU cross runs each family's BEST-DICE row
again, at 4 AND at 8 CPUs, on a few patients, to show what more CPUs buy: speed-up,
CPU-hours and peak memory per patient (user ruling 2026-10-16: `REGBENCH_CPUS` exists
for this alone; user ruling 2026-10-18: 4 and 8 CPUs only, not 16; the cross's rows
are the best-Dice rows, DeeperHistReg's among its CPU rows, T31 R12).

- **The rows**: `regbench cross-runs --out MAIN [--dataset multiplex]` prints each
  family's best-Dice row of the MAIN run's tables (`tables.best_of`, the paper's own
  rule), DeeperHistReg's among its CPU rows, one `family<TAB>row` line each and a
  `# --rows a,b,...` line. It refuses a MAIN with no scored cells table.
  `regbench cpu-scaling --main MAIN` takes the same list; `--rows` names others. A GPU
  row is refused (`cpu_scaling.build`): the cross measures CPUs, and a GPU row's host
  side is not the method. `stare_local4` is never one (it is a cost row of `stare`).
- **The patients** are a rule: the smallest, median and largest by total moving
  gigapixels (the sum of `mov_hw` over a patient's rounds, ties by name; `subset.py`).
  `regbench pick-patients --cases C [--n 3]` prints them, one `patient<TAB>gpx` line
  each, under `# rule: ...`; for an `--n` other than 3, evenly spaced ranks with both
  ends.
- **Their cases** are a folder of symlinks to the prepared case folders (and a copy of
  `manifest.csv`): `regbench subset-cases --cases C --groups P1,P2,P3 --to D`. It
  refuses a `D` that is not empty. Nothing is prepared again.
- **Both counts, the same submission style** (T31 R13): every cross row runs at 4 AND
  at 8 CPUs, each count in its own folder (`out_c4`, `out_c8`), so no point is the main
  run's result under another load. Each point is run twice: `REGBENCH_REPEAT=2` is the
  same submission into `rep1/` and `rep2/` of the folder (each repeat its own results;
  `rep2`'s first jobs wait `afterany` for `rep1`'s last ones, its scorers, so the two
  never share the nodes' time with each other: the figure draws their spread).
- **`REGBENCH_CPUS`** (4 or 8; anything else, 16 included, stops the launch before any `sbatch`)
  is every ladder task's `--cpus-per-task`, recorded in `run.json` as
  `cpus_requested`. The selfchecks and the prepare, settle, tissue and score jobs keep
  their own allocation. `stare_local4` is dropped from a launch at 8 CPUs
  (`submit_dropped.csv`, `run_config`): it is the 4-CPU row by definition.
- **`REGBENCH_STARE_EXECUTION=local`** makes every STARE run one job per case on the
  ordinary ladder: its stages start inside that job (`run --stages local`, option
  `workers=$REGBENCH_CPUS`, so the tiles run that many at a time). There is no stage
  chain and no tile submitter. The launch prints `# STARE: stages run inside one job
  of N CPUs (local)`. Such a result's reuse key has the execution `local`, so it is
  never reused for a staged result, and a staged one is never reused for it.
- **What `cpu_scaling` checks, per point** (never hidden): `cpus_checked` (every
  record's `cpus_requested` against the point's count: `yes`, or `no: <seen>`),
  `cpu_model` and `cpu_model_matches` (one CPU model over every count and repeat of the
  patient), and `n_patients_incomplete` (patients not complete at every count, left
  out and counted). Speed-up is taken inside one (row, execution, repeat) only.

```bash
regbench_py STARE_REGBENCH cross-runs --out "$MAIN"          # the rows
regbench_py STARE_REGBENCH pick-patients --cases "$REGBENCH_CASES"
regbench_py STARE_REGBENCH subset-cases --cases "$REGBENCH_CASES" \
    --groups <the three patients> --to "$ROOT/cases_cross"
RUNS="<the rows cross-runs printed, space-separated>"
for c in 4 8; do
    REGBENCH_CPUS=$c REGBENCH_REPEAT=2 REGBENCH_STARE_EXECUTION=local \
    REGBENCH_OUT=$ROOT/out_c$c REGBENCH_CASES=$ROOT/cases_cross \
        benchmarks/regbench/slurm/submit_cohort.sh --runs "$RUNS" -- <site options>
done
regbench_py STARE_REGBENCH cpu-scaling --main "$MAIN" \
    --point 4=$ROOT/out_c4 --point 8=$ROOT/out_c8 --to "$MAIN/paper/cpu_scaling.csv"
```

Never point a `local` launch at the main `REGBENCH_OUT`: its STARE results have another
reuse key, so every STARE unit there would run again and replace the staged result.
The launcher refuses it before any `sbatch` when the folder holds a STARE result of its
stage jobs (`regbench staged-results --out O` lists them); a staged launch over `local`
results is not refused, and `stare_local4` (its own reuse key) never triggers it.
Without `REGBENCH_CPUS`, `REGBENCH_STARE_EXECUTION`, `REGBENCH_REPEAT` and
`REGBENCH_WAVES` the launcher submits exactly what it submitted before the cross
existed, plus `stare_local4`'s own jobs (T31 R2). A test compares the default dry run
with the one from the launcher at 275edda0, `stare_local4`'s jobs set aside.

### The preflight, and what every image is started with

`submit_cohort.sh --preflight [--runs ...]` starts every image a selected run uses, on
the login node, exactly as a job starts it, and asks inside it (`regbench preflight`,
`preflight.py`): do `numpy`, `tifffile`, `zarr`, `imagecodecs` and `stare.slide_io`
import, does the method's package import with what its adapter imports beside it, is
the package the version the adapter enforces, do `numpy` and the package resolve inside
the image and not under `~/.local`, and can `REGBENCH_OUT` and `REGBENCH_WORK` be
written. It prints one `preflight FAIL [<image> <method>] ...` line per failure; an image
that does not start or has neither `python` nor `python3` is one such line. A real
submission runs the same preflight before its first `sbatch` and stops on a failure
with nothing submitted; a dry run does not start an image and says so. The pinned
STARE source of the images without a `stare` package is not part of the preflight: it
is fetched just before it, and also when `REGBENCH_PREFLIGHT=0` skips the checks,
because the jobs of those images import it either way; a source that cannot be fetched
stops the launch with nothing submitted.

- **The version check is the adapter's, shown early.** The preflight reads the version
  the way the adapter reads it and compares it with the adapter's `VERSION`; the adapter
  is unchanged and still refuses any other version when a unit runs. ASHLAR's version
  string comes from versioneer `[MEMORY: the ASHLAR repository's setup, not retrieved
  while this was written]`: if the official image prints something other than exactly
  `1.20.0`, the preflight line quotes what it prints, and the owner decides (the
  adapter's `--opt any_version=1` exists for a reviewed exception).
- **`stare` in the images that do not ship it.** The harness reads every image through
  `stare.slide_io`, and the benchmark's ASHLAR command path imports STARE's manifest
  code. The official ASHLAR image and upstream's VALIS 1.0.0 image have no `stare`
  (`REGBENCH_STARE_SRC_IMAGES`). For those the launcher fetches the release pinned in
  `requirements/stare.txt` once, as source, into `$REGBENCH_OUT/cache/stare-<tag>/src`
  (`regbench_fetch_stare`, on the login node; a job never downloads), and puts that
  folder on `PYTHONPATH` inside the image. That is STARE's pure-Python code only: the
  image must already hold what it imports that is compiled or third-party, `numpy`,
  `tifffile`, `zarr` (with `numcodecs`) and `imagecodecs`; `psutil` where the memory
  monitor is to use it (without it the monitor reads `/proc`, `rss_method: proc`).
  Neither published image holds all of them (the preflight, 2026-10-13: the official
  ASHLAR image has no `imagecodecs`, upstream's VALIS 1.0.0 image has neither `zarr`
  nor `imagecodecs`), so the two are run from `containers/ashlar120` and
  `containers/valis100`: the published image by digest, plus those packages, with a
  freeze of the base as the install's constraint.
- **Inside every image** (`slurm/env.sh`, through `APPTAINERENV_<name>`, which sets a
  variable under `--cleanenv`): `PYTHONNOUSERSITE=1`, so the packages in the host
  user's `~/.local` (the home folder is mounted) cannot shadow a pinned one;
  `PYTHONPATH` is the checkout plus that STARE source and nothing of the host's own
  `PYTHONPATH`; `TMPDIR` is the job's, bound and passed on; `REGBENCH_WORK`,
  `REGBENCH_PRIVATE` and `REGBENCH_PRIVATE_DIR` are passed on as exported (the plug-in
  folder is bound; a plug-in that is switched on and cannot be read stops the run as
  before). An environment that is not an image (`REGBENCH_CONDA_X`, `REGBENCH_VENV_X`)
  gets `PYTHONNOUSERSITE=1` too.

### What the site must allow

The dry run prints three numbers for a submission; each has a limit on the site.

| what the dry run prints | what it is | the site's limit, and how to read it |
|---|---|---|
| `N sbatch call(s) would be made now` and `M task(s) in them` | everything the launcher submits at once for the default submission (the cohort alone), and one task per unit and rung for each run, per case for the scoring arrays | array tasks count against the per-user and per-association `MaxSubmitJobs` and the cluster's `MaxJobCount`: `sacctmgr show assoc user=$USER format=Account,MaxSubmitJobs,MaxJobs`, `sacctmgr show qos format=Name,MaxSubmitPU,MaxWall`, `scontrol show config \| grep -E "MaxArraySize\|MaxJobCount"` |
| `T tile task(s)` fed in later | one per tile, never in the queue together | none: they pass through the ceiling |
| `the most in the queue at once` | M plus each tile submitter's share of `REGBENCH_STAGE_QUEUE_MAX` (at most the total itself) | the same `MaxSubmitJobs`. Lower it with `REGBENCH_STAGE_QUEUE_MAX`, `--runs` or `--datasets` |

Besides that:

- **A compute node may call `sbatch` and `squeue`** (this site's Nextflow head jobs do).
  Check: `srun -c 1 --mem=1G --time=5 bash -c 'sbatch --version && squeue -h -u $USER | head -n 3'`.
- **A partition with 700 GB nodes**, given after `--`: the ladder's last rung asks for 700 GB at 4 CPUs
  for every method, and a job asking for more than any node has stays pending forever.
- **A job of 7 days at 1 CPU and 1 GB** is accepted on the partition given after `--`
  (`REGBENCH_STAGE_FEED_TIME` shortens it; the chain of `REGBENCH_STAGE_FEEDERS`
  submitters then covers less time).
- **`MaxArraySize`** is above `REGBENCH_STAGE_ARRAY_SPAN` (1000; SLURM's default
  `MaxArraySize` is 1001 `[MEMORY]`) and above the rounds of a dataset.

**Everything that can stop the launcher is read before its first `sbatch`**: the run
list, the prepared cases, the STARE source, the preflight, the pipeline's allocation
per stage (`runs --stage-allocation`) and the tile layout of every run whose
registration is several jobs per round (`regbench stages`, which writes
`<out>/logs/stages/<submission>/<run>-<dataset>.tsv`). A failure of any of them ends
with "Nothing was submitted." and nothing queued; `logs/submission` is written only
after all of them, so a call that was stopped is not "the last submission".

**When the launcher stops after its first `sbatch`** (the scheduler refuses a job: a
limit above, a wrong partition; or the script itself fails, e.g. on a results folder
it can no longer write), it says so on stderr: the reason (for a refused `sbatch`, the
job and the scheduler's words), how many jobs of this call are already queued, the
file holding their ids (`<out>/logs/submitted/<submission>.ids`, one per line, written
before anything else about a job; the tile submitters add theirs), and the line that
removes them, `scancel $(cat <that file>)`. The jobs before the stop ARE queued and
wait for jobs that do not exist: cancel them, remove the reason, submit again.

**A submission id is used once per results folder.** A call that would run anything
under an id the folder already holds a submission of (`logs/submitted/<id>.ids`) is
refused: a tile submitter keeps what is due and what it fed under the id, and would
take an earlier call's lists for its own and feed nothing. Leave `REGBENCH_SUBMISSION`
unset; every unit that is ok is reused whatever the id. `--score-only` takes the id of
the submission it scores, and `--prepare` keeps its job ids in
`logs/submitted/<id>.prepare.ids`.

**One submission at a time into a results folder.** A second `submit_cohort.sh` into a
`REGBENCH_OUT` whose earlier jobs are still pending or running is REFUSED before
anything is touched: it would replace the stage files and retry plans those jobs read. The
refusal names the jobs and the `scancel` line. So the cohort is not submitted patient
by patient into one folder while the first patient runs: wait for it, or give each its
own `REGBENCH_OUT`. A queue that cannot be read (`squeue` fails) refuses too.
`--score-only` is refused in the same two cases: it would score, and settle against the
accounting, results whose tasks have not ended, and two scorings at once would write the
same tables. `REGBENCH_SCORE_WHILE_QUEUED=1` scores what is there all the same (the
tables then say `not run` for what is still running). `--prepare` and `--dry-run` are
not refused.

### Not verified: check these on the cluster before the real submission

The script was written against SLURM's documented behaviour (slurm.schedmd.com: the
`sbatch` manual `[PARTIAL: read the --no-requeue, --requeue, --kill-on-invalid-dep and
--dependency entries; missing the rest of the page, the filename patterns and the array
section included]`, the `sacct` manual `[PARTIAL: read the -S, -j, -X and --name entries,
the JOB STATE CODES list and the State, ElapsedRaw and TotalCPU fields; missing the rest
of the page]`, the `squeue` manual `[PARTIAL: read the -h, -r, -u, -n, -o entries and
the %j, %A, %F, %i, %K, %r, %T format fields; missing the rest of the page; it does not
say what squeue exits with when the controller does not answer]`, "Job Reason Codes"
`[PARTIAL: read the entries for the five reasons of REGBENCH_STAGE_HELD_REASONS and
for Priority, Resources, Dependency, BeginTime, None and the QOS* and Assoc* families;
missing the rest of the list]`). None of the following was observed:

- **A job cancelled for an invalid dependency satisfies `afterany`.** The scorer waits
  `afterany` on every array and every selfcheck; if a cancelled job did not release it, the
  scorer would be cancelled too. Three dummy jobs show it:
  `a=$(sbatch --parsable --wrap 'exit 1'); b=$(sbatch --parsable --dependency=afterok:$a --kill-on-invalid-dep=yes --wrap true); sbatch --dependency=afterany:$b --kill-on-invalid-dep=yes --wrap 'echo ran'`:
  the second must end `CANCELLED` and the third must run.
- **What the tile submitter's bounds rest on.** That `squeue` exits non-zero when it
  cannot reach the controller (the manual does not say); that `%r` prints the reason
  names of the "Job Reason Codes" page as they are spelled there, one word, last in the
  line; that `sacct -S YYYY-MM-DDTHH:MM:SS --name <job name>` finds an array that ended
  on an earlier day (the format is one of the manual's; the manual says `-S` selects
  "jobs in any state after the specified time").
- **The log names.** `slurm/env.sh` rebuilds each task's log path as
  `$REGBENCH_OUT/logs/$SLURM_JOB_NAME-<job or array_task>.log`, assuming `%x`, `%A_%a` and
  `%j` expand to `SLURM_JOB_NAME`, `SLURM_ARRAY_JOB_ID_SLURM_ARRAY_TASK_ID` and
  `SLURM_JOB_ID` (sbatch manual, filename patterns `[MEMORY]`). After the first job:
  `tail -n 1` of any task log must be a `REGBENCH_END` line whose `log=` is that file.
- **Cores are confined** (`ConstrainCores`): `sbatch -c 4 --wrap 'python3 -c "import os; print(len(os.sched_getaffinity(0)), os.cpu_count())"'` should print `4` and the node's
  count. The cost tables report what each run saw either way.
- **Memory is enforced through cgroups and a killed job is `OUT_OF_MEMORY` in `sacct`**,
  and `sacct` can be called from inside a job (every later task of a ladder asks it about
  the earlier ones). Without `sacct` a unit whose record still says `killed` is run once
  more by the next rung's `climb` task (or settled as `oom` by the script's own note of
  a Python ended by signal 9), and the cost of the attempt it replaces is
  `wasted_unknown`. Not observed either: that `apptainer exec` returns 137 for a Python
  the kernel killed, and that the batch script outlives it.
- **The states of a lost task.** `ladder.STATES_INFRASTRUCTURE` (`NODE_FAIL`,
  `PREEMPTED`, `BOOT_FAIL`, `CANCELLED`, `REQUEUED`, `REVOKED`) and the states of a task
  that has not ended (`PENDING`, `RUNNING`, `RESIZING`, `SUSPENDED`) are names of the
  `sacct` manual's JOB STATE CODES (tag above); the manual says a state too long for its
  column is followed by `+`, and a state matches by prefix here (`CANCELLED by
  <uid>`). Check on the site: what `sacct -n -P -j <array job>_<task> -o
  JobID,State,ElapsedRaw,TotalCPU` prints for an array task (the first field must be
  `<array job>_<task>`, which is what a record's `launch_id` is built from:
  `SLURM_ARRAY_JOB_ID` and `SLURM_ARRAY_TASK_ID`), for a task killed for memory, and for
  one cancelled with `scancel`. A `scancel` by the user is also read as `infrastructure`
  and retried once: cancel the whole submission, not single tasks, to stop it. Every job
  is submitted with `--no-requeue`; that a preempted or node-failed task then ends in
  one of those states instead of starting again under the same job id was not observed
  (a site's `ForceRequeueOnFail` overrides the option).
- **The out-of-memory counter is readable inside the image.** `run.json: oom_watch` and
  the first line of the cost tables say whether it was. Not observed under `apptainer
  exec --cleanenv`. One STARE unit is enough to see it.
- **The JVM heap VALIS really gets.** As read in both sdists, `valis.registration.
  init_jvm(jar=None, mem_gb=10)` passes the literal 10 on and ignores its argument
  (1.0.0 `registration.py:115-118`, 1.2.0 `:132-135` `[PARTIAL: those lines and
  slide_io.init_jvm]`). For VALIS 1.2.0 the adapter therefore starts the JVM itself
  through `valis.slide_io.init_jvm(mem_gb=...)`. For the pipeline's VALIS the adapter
  changes nothing (it runs the pipeline's call), so the heap the pipeline asks for may
  never arrive. `run.json: info.jvm_max_heap_gb` is the JVM's own
  `Runtime.maxMemory()`: read it on the first `valis_mirage` and `valis120` patient. If
  it is 10 for `valis_mirage` at every rung, a Java heap error there repeats on all four
  rungs; that is a defect of the pipeline (`bin/utils/valis_config.py::init_jvm`), not
  of this benchmark, and it is the owner's to rule on.
- **The tile submitter** (`feed_tiles.sbatch`) was run only against stand-ins. On the
  site, before the first staged run, see "What the site must allow": that a compute
  node can call `sbatch` and `squeue`, that `squeue -h -r -u $USER -o %j` prints one
  line per array TASK, pending ones included, with the job's full name, and that
  `squeue -h -u $USER -o '%F %A'` prints the array's job id first (the launcher
  matches either field against the ids it recorded). The `squeue` and `sbatch`
  manuals were not retrieved while this was written `[MEMORY]`.
- **The per-case scoring array** (`score_cohort.sbatch case`), `--score-only`, and the
  `python3` fallback inside ASHLAR's image: tested through stand-ins and through the
  real CLI on a laptop, never on a scheduler or inside that image.
- **ASHLAR's image sets `OMP_NUM_THREADS=1`** ("Cost", "Threads the method's own image
  sets"). Read from the Dockerfile and from Apptainer's documentation, never observed:
  `run.json: threads_env` of the first ASHLAR patient must read `"OMP_NUM_THREADS":
  "1"`, the header of the cost tables must name it, and its `cpu_efficiency` says how
  much of the 4 CPUs it used. The `rb-settle` job needs `sacct` on the node it runs on;
  where it has none the scorer says "settling here" in its log.
- **`%j`, `%A_%a` in `--output`** and `--export=ALL` carrying the `REGBENCH_*` variables
  into the job.

## Caveats that travel with the results

1. **Whose settings.** STARE's settings and the pipeline's VALIS settings were chosen by the
   authors on this cohort with this metric: `stare` and `valis_mirage` are in-sample, and
   the caption of the cohort's tables in `summary.md` says so above the numbers
   (`score.IN_SAMPLE`). The other methods were NOT tuned on it, but
   "package defaults" describes only some of them: three primary rows are not a default.
   `valis120_hires` is upstream's high-resolution example, not its default
   (`valis120` is, and is shown beside it). `deeperhistreg_hr_gpu` is a shipped
   8192 px parameter set, not the 4096 px default, on a GPU. `ashlar` is ASHLAR's own
   command at its defaults, but on tiles the benchmark cut from stitched slides, with
   noise added to their empty background. `palom` is its documented recipe. None of these
   choices was made by looking at a score; all were made by STARE's authors. This is the
   largest asymmetry in the comparison.
2. **Two frames**, with the differences quoted above.
3. **Per-patient and pairwise methods** see different problems and fail in different units,
   which is why failures are counted both as launches and as rounds.
4. **Self-consistency, not accuracy** on real slides.
5. **The ladder:** methods need not finish at the same rung, so they need not have had the
   same memory request; the tables show each run's rung.
6. **About a fifth to two fifths of ASHLAR's nuclei are scored through an interpolated
   map** (the tile overlaps; see "ASHLAR", "What the overlap bands cost"). It has no bias
   when a round needs only a translation, and it is optimistic for ASHLAR when a rotation
   is left inside its tiles: at 0.5 degrees, 5.2 px of map error against 8.8 px for the
   copies the image shows. The selfcheck masks exactly those points. The
   `<variant>:non_ashlar` rows are the measure of how much it matters.
7. **DeeperHistReg's rows follow its writer, not its registration.** By the adapter's
   reading the written image sits half a field pixel right and down of the
   registration: 2.5 px (0.81 µm) per axis, 1.15 µm along the diagonal, at the primary
   8192 px set; 4.9 px (1.59 µm) per axis at the 4096 px set; plus a whole-pixel padding
   term on uneven sizes (measured on the cohort on 2026-10-15: roughly 1 to 3 µm in all).
   `within_1um` of `deeperhistreg_hr_gpu` can therefore be low with a perfect
   registration. The benchmark scores what the package puts on disk; no row of the
   registration itself is written.
8. **The cohort's "nuclei" are StarDist nuclei grown by 10 px** ("What the cohort's nuclei
   are"): Dice and `within_1um` have a ceiling that is not registration, the same for
   every method.
9. **No ranking is statistically supported.** The paired differences carry a bootstrap
   interval over patients, which is coarse with few of them; there is no test and no
   p-value.
10. **The tissue-level block is secondary**: ACCREDIT's score on same-modality
   fluorescence, outside what it was validated for, by its released code where that
   differs from the preprint's text. It ranks nothing.
11. **Nothing here has run on the cluster.** The cgroup counter path and whether a site
   confines cores are unverified.
12. **Compute-all is at the shipped defaults and may differ from the site's production
   settings.** The full-pipeline cost (`pipeline_cost*.csv`, secondary) runs the pipeline
   with `seg_method=stardist` and `pixel_size=0.325` (rulings) and everything else at the
   shipped defaults and the site profile; registration is at the shipped tier
   `memory_mode='high'` unless the site profile overrides it. A site that runs production
   at another tier, another segmenter or another resource profile has another cost. The
   `all` row's wall time is a sum over patients that includes SLURM queue time, and the
   realtime-based CPU-h does not; the two clocks are never added.

### STARE (`methods/stare.py`)

**STARE runs the way the pipeline runs it** (`subworkflows/local/adapters/tiled_adapter.nf`):
per moving round, `stare coarse`, then one `stare reg-tile` process per tile, then
`stare solve`, then `stare stitch`, on the slides as the case has them (`ref_image`,
`mov_image`), with `--nuclear-index` = the case's `mov_channel` (a case whose reference
has its nuclei at another index is refused: the commands take one index). On the cohort
each of the four is a scheduler job of its own (`regbench.stages`,
`slurm/stage.sbatch`; "Running the cohort on a cluster" has the job graph):

| stage | pipeline process | label | CPUs | memory, attempt 1 | time, attempt 1 | where the numbers are |
|---|---|---|---|---|---|---|
| COARSE | `TILED_COARSE` | `process_low` | 2 | 2 GB | 2 h | `conf/modules.config:137-141` (label), `:571-590` (memory: `ceil(max(2, (0.5 + 0.1 Mpx) x 2))`, 1024 px thumbnail) |
| REG-TILE, per tile | `TILED_REG_TILE` | `process_low` | 2 | 4 GB | 2 h | `:137-141`, `:515-544` (memory: `ceil(max(4, (0.5 + 0.13 Mpx) x 2))`, window 2048 + 2 x max(256, 3 x 64) = 2560 px) |
| SOLVE | `TILED_SOLVE` | `process_single` | 1 | 1 GB | 8 h | `:131-135`, `:600-604` |
| STITCH | `TILED_STITCH` | `process_medium` | 4 | 4 GB | 4 h | `:143-147` (CPUs, time), `:635-657` (memory: `ceil(max(4, (0.6 + 0.17 Mpx) x 2))`, 1024 px output tile) |

Memory and time are multiplied by the attempt, as there. `stages.ALLOCATION` holds the
table and `test_stages.py` works each number out again from the text of
`conf/modules.config` and `lib/RegPresets.groovy`: it fails when they differ.

- **Retries.** None of the four processes sets an `errorStrategy`: they inherit the
  process-level one of `conf/base.config:11-22`, which retries an exit status of 104,
  134, 135, 137, 139, 140 or 143 and ends the run on any other, with `maxRetries = 3`.
  It is none of the four policies `conf/modules.config` names; in their terms it is
  `retry-then-fail` with a shorter list of statuses. The benchmark applies it to
  COARSE, every tile and SOLVE: up to four attempts, the next with the next attempt's
  memory and time, and a job the scheduler killed before it could record its status, or
  that never started, is retried too. A stage that still fails fails the round: its
  `run.json` has the stage, the last exit status and memory in `error`, the cause
  (`oom` for exit 137 or a memory error, `timeout` for 140 and 143, `exception`
  otherwise, `infrastructure` when no attempt left a record) and `no_rerun: true`, and
  the round is imputed as every failed round is. A tile that never reached the queue
  is `not_submitted` (the launcher's), and so is a tile that had not recorded its end,
  or had ended the way the pipeline retries with attempts left, when the round's last
  job ran while the tile submitter had not finished: the attempt after it was never
  submitted. Once the submitter has finished (it writes its `.done` only after every
  tile task has left the queue) the tile's own record decides. STITCH runs as a process of its own inside the
  round's last job, which has the benchmark's allocation and ladder ("Running the
  cohort on a cluster", "The round's last job, in three sentences"): STITCH is retried
  there on the same list of statuses with the pipeline's time limit per attempt, a
  STITCH that still fails is the writer's and is run at no further rung, and the job
  itself goes up a rung for a memory cause (`oom` when the scheduler ended it for
  memory, as for any other unit's task) and is tried again once per rung when the
  scheduler lost it.
- **One record per job**: `<out>/<row>/<dataset>/<case>/stages/jobs/<stage>[_<tile
  row>].json`, with the command, its exit status, wall time, CPU time and peak memory
  (`os.wait4`: the stage command's own process), the allocation, the job id and every
  earlier attempt. The stage files (anchor, tile plan, one control file per tile,
  manifest) stay beside them: tens of kilobytes per tile (a control file holds the
  tile's whole lattice of window vectors; a job's record with its command and the tail
  of what it printed is a few more).
- **One path to the transform, and a stated launcher.** COARSE, every tile and SOLVE
  always run as processes of their own through one function (`run_stage`), and the
  round's transform is always obtained by reading the records they left (`_collect`).
  The benchmark never runs `stare register`. What differs between two runs is who
  starts the stage processes, and the run says so in `REGBENCH_STAGES`; there is no
  default:

  | `REGBENCH_STAGES` | who starts the stages | stated by | cost of the round |
  |---|---|---|---|
  | `jobs` | the scheduler: one job per stage, the pipeline's allocation and retries | `slurm/run_rung.sbatch`, from the launcher's `stages` column | what the stage jobs recorded |
  | `local` | the run itself: the same `run_stage` calls, the tiles `workers` at a time, retried by the same rule, no memory or time limit per stage | `regbench selfcheck` for its own run; `regbench run --stages local` for a run by hand; a test | the run's own process tree, as for any method |

  A run of STARE that states neither is refused before a stage process is started:
  `regbench run` touches no record and ends with `REGBENCH_END ... status=lost
  cause=not_submitted whose=launcher`, and nothing is registered in the stage jobs'
  place. `run.json: info.jobs.execution` names the launcher that ran. The selfcheck
  therefore checks the path the cohort's results are made by: the same commands, the
  same records, the same collector.
- **Both launchers give one transform.** `test_stages.py` registers the 2048 x 2560
  pair through `regbench stage`, one call per stage as `slurm/stage.sbatch` makes it,
  and through the local launcher: the anchor and the tile plan are equal byte for byte
  and the transform within 0.001 px (the manifest's numbers and the mapped points of a
  64 px lattice). It is not equal to the byte because one `stare reg-tile` command on
  identical inputs is not: between two fresh processes its control values, rounded to 4
  decimals, differ by one unit of the fourth in places (measured; more often on a busy
  machine), and the transform then differed by at most 0.00008 px in ten runs of
  the test on a machine under heavy load. Each stage's argument list is compared whole with the one in the source of
  `stare.cli.register` (stare-registration 1.2.1, installed package `[FULL: cli.py]`),
  which is read and not run.

- **Parameters.** `TIER_HIGH_STRIDE64` = the `high` row of the STARE table in
  `lib/RegPresets.groovy` (`tile` 2048, `halo` 256, `out_tile` 1024, `max_dim` 1024) plus
  `stride` 64, all five passed explicitly; a test reads the row from that file. The two
  STITCH flags the pipeline always passes are passed too: `--pixel-size` (the case's) and
  `--channel-names` (the case's, else the names in the moving slide's header). SOLVE is
  given `--max-disp` = the halo, as `conf/modules.config` gives `TILED_SOLVE`. A tile job
  names its tile by `--plan <COARSE's tile plan> --row <n>`, which reads the values the
  pipeline's module passes one by one. `info["cpu_count"]` records what COARSE's FFTs
  can use on the node.
- **Tier.** `--opt tier=<high|medium|low>`; unset is `high`; any other value fails the
  case. `TIERS` holds the three rows of the STARE table in `lib/RegPresets.groovy`, each
  read back from that file by a test: `high` 2048 / 256 / 1024 / 1024, `medium`
  1024 / 192 / 768 / 1024, `low` 512 / 128 / 512 / 512 (`tile` / `halo` / `out_tile` /
  `max_dim`). A tier moves those four values and nothing else, so SOLVE's `--max-disp`
  is that tier's halo; the stride is crossed with it. The high argument list is the one
  every earlier result was made with. The tier is in `info.implementation.tier`, and
  `changed_from_tier` lists changes from the run's own tier. The stage allocation table
  above serves all three tiers: `test_stages.py` evaluates the three memory closures at
  each tier and every one lands on its floor.
- **Stride.** `--opt stride=<64|128|256>`; unset is 64; any other value fails the case.
  With `--label stare_s128` the results are named `stare_s128_rigid` and `stare_s128`.
  The rigid stage does not depend on the stride (COARSE is never given it and SOLVE stores
  the anchor unchanged; `stare/cli.py:79-100`, `stare/stages/solve.py:77-89`,
  `stare/stage_warp.py:85-87` at 1.2.1 `[PARTIAL: cli.py and stage_warp.py in full, main()
  of stages/solve.py and stages/stitch.py, manifest.slide_entry; stages/coarse.py and
  coarse_align.py only searched for "stride" (no match) and for the argument list]`), and a test finds `info["m0"]` and the rigid
  row's warped nuclei identical to the last bit at stride 64 and 256. So a stride run adds
  one result, its final variant; its `_rigid` row repeats `stare_rigid`.
- **Output.** STARE's own STITCH on the reference grid: no `origin_xy`, nothing padded
  (0 where the moving slide has no pixels). It runs once the transform exists and its
  time is `write_s`. `run.json: info.m0` is the anchor, moving px -> reference px.
- **Cost.** `run.json: info.jobs` is the round's cost over its jobs ("Cost", "A
  registration in several jobs").

Selfcheck per stride (2026-10-09, macOS arm64, `--opt workers=4`, `stare` 1.2.1), every one
`ok` with no failed block, `stare_rigid` `no_stage_image`:

| stride | pair | dots found | residual p50 / p95 px | mean x / y px | worst block p95 | worst block mean |
|---|---|---|---|---|---|---|
| 64 | 2048 x 2560 | 537 / 537 | 0.055 / 0.090 | +0.003 / +0.002 | 0.104 (20 blocks) | 0.020 |
| 64 | 4096 x 5120 | 2202 / 2202 | 0.057 / 0.091 | 0.000 / 0.000 | 0.108 (80 blocks) | 0.025 |
| 128 | 2048 x 2560 | 537 / 537 | 0.055 / 0.089 | +0.001 / +0.001 | 0.096 (20 blocks) | 0.016 |
| 128 | 4096 x 5120 | 2200 / 2200 | 0.058 / 0.093 | -0.001 / -0.001 | 0.107 (80 blocks) | 0.022 |
| 256 | 2048 x 2560 | 539 / 539 | 0.058 / 0.089 | -0.001 / +0.001 | 0.098 (20 blocks) | 0.021 |
| 256 | 4096 x 5120 | 2200 / 2200 | 0.056 / 0.089 | -0.001 / 0.000 | 0.103 (80 blocks) | 0.022 |

### VALIS (`methods/valis_mirage.py`, `methods/valis.py`, `methods/_valis_points.py`)

**Never run with VALIS.** Both adapters were written from the PyPI sdists
`valis-wsi==1.0.0` and `==1.2.0` `[PARTIAL: registration.py (Slide's crop, field getters
and warp functions, Valis.__init__, non_rigid_register, register, register_micro,
warp_and_save_slides), warp_tools.py (warp_img, the point functions), slide_tools.warp_slide;
the registrars, slide_io and libvips were not read]` and tested against a recording
stand-in (`tests/regbench/fake_valis.py`). Everything in this section about VALIS's
behaviour is a reading of that source; the selfcheck inside each VALIS image is what
tests it.

**Three rows, one patient per run** (`run --group`, `selfcheck`). For VALIS 1.2.0,
`valis120` is upstream's default setting and `valis120_hires` upstream's high-resolution
example; both run, and `valis120_hires` is the primary row by the user's ruling of
2026-10-11 (`variants.PRIMARY_VALIS120`):

| run | call | what the benchmark sets |
|---|---|---|
| `--method valis_mirage` | `bin/register.py::valis_registration`, the function the pipeline's REGISTER runs, with `valis_mirage.ARGS` (= `nextflow.config`'s defaults: `memory_mode` high, micro fraction 0.125, `max_image_dim_px` 4000, `micro_reg` 2). `align_to_reference=True`, `crop="reference"`, SuperPoint + SuperGlue and the micro-rigid registrar are inside that function | the folder, the reference's file name, `n_cpus` = the allocation, `jvm_heap_gb` = `min(32 + 16 x attempt, GiB - 4)`, attempt being the pipeline's own attempt for that memory (REGISTER asks `64 GB x 2^(attempt-1)`; the module's formula: 12, 48, 80, 96, 96 GiB on the five rungs), `stage_checkpoint_dir=None` (files only) |
| `--method valis --label valis120` | `Valis(src, dst, img_list=..., reference_img_f=...)`, `.register()`, `.register_micro()` with no argument | the reference slide. `align_to_reference` stays `False` |
| `--method valis --label valis120_hires --opt hires=1` | upstream's high-resolution example (`docs/examples.rst:122-149` at v1.2.0 `[PARTIAL: lines 100-175]`): `micro_rigid_registrar_cls=MicroRigidRegistrar`, then `register_micro(max_non_rigid_registration_dim_px=floor(0.25 x the smallest slide's longest side))` | the reference slide. The label and the option must come together or the run is refused |

**The pipeline's VALIS at three tiers x micro-registration depth 0, 1, 2** (Plan F, the
paper's tier / micro table). `--opt memory_mode=<high|medium|low>` and
`--opt micro_reg=<0|1|2>` (unset: `high` and 2) go into the call through
`valis_mirage.call_args`, which changes those two values of `ARGS` and nothing else, as
the pipeline's own tier moves `--memory-mode` alone (`modules/local/register.nf`). The
tier picks a row of `bin/utils/valis_config.py` `MEMORY_PRESETS`: processed / non-rigid
size `high` 2048 / 2048, `medium` 1024 / 1024, `low` 512 / 512 px. `max_image_dim_px`
4000, the micro fraction 0.125, the keypoint ceiling, the JVM heap formula and the
pipeline's REGISTER allocation are the same at every tier. The depth is
`bin/register.py`'s `micro_reg` (`bin/register.py:253-257`): 0 is neither micro pass,
**1 is micro-RIGID only** (the `MicroRigidRegistrar` inside `register()`), and 2 adds the
micro non-rigid pass (`register_micro`). A non-rigid `micro` stage exists only at depth 2,
so at depth 0 and 1 the `_nonrigid` row and the final row are the same transform, and at
depth 0 the `_rigid` row is the affine alone. `run.json: info.implementation` records
`tier`, `micro_reg` and `stages` (`valis_mirage.STAGES_AT_DEPTH`). The runs are
`valis_mirage` (`high`, 2: the shipped call, unchanged) and eight more,
`valis_mirage_<tier>_m<depth>` (`variants._valis_mirage_runs`), each writing
`<label>_rigid`, `<label>_nonrigid` and `<label>`, checked by `valis_mirage`'s selfcheck,
with no realcheck (only the primary row has one). A label of that form must agree with
the two options, or the run is refused. Each is one job per patient on the ladder, at 4
CPUs.

**Memory failures, and the heap.** `Valis.register()` catches its own exception, prints
it and returns `(None, None, None)`; the adapters used to raise a fixed text, so a Java
`OutOfMemoryError` or a NumPy memory error was `cause: exception` and VALIS never went up
the ladder while STARE's `MemoryError` did. Both adapters now wrap `valtils.print_warning`
for the duration of the call (`_valis_points.swallowed_errors`: it notes what VALIS hands
it and calls the original with the same arguments) and put that text into the error, so a
memory failure is `oom` and climbs. VALIS 1.0.0 also reports a failed Bio-Formats tile
read with a bare `print(e)`, which `print_warning` never sees, so for the call stdout is
teed (`_valis_points.printed_errors`: what it printed passes on unchanged, and up to 50
chunks that read as a memory error are kept, 400 characters each) and that text is read
too, also for the micro pass `bin/register.py` catches. In `valis_mirage`, a slide the
pipeline's writer failed on keeps its reason as `writer` (`writer_oom` on memory), and a
failure of the benchmark's own capture (the crop probe, the field load) is `benchmark`
(`benchmark_oom`), not the method's `exception`; the same holds for `valis120`'s capture.

- `valis120`, `valis120_hires`: the JVM is started by the adapter through VALIS's own
  `slide_io.init_jvm(mem_gb=...)` before `Valis(...)` is built, with
  `min(32 + 16 x attempt, GiB - 4)`, attempt being the pipeline's attempt for that
  memory (REGISTER asks `64 GB x 2^(attempt-1)`: attempt 1 below 64 GB): 12, 48, 80, 96,
  96 GiB at 16, 64, 256, 512 and 700 GB, the pipeline module's formula, because VALIS's `registration.init_jvm` passes the literal 10 on
  whatever it is given. This is the ONE thing besides the reference slide that the
  benchmark sets for those rows, and it is not a registration parameter. Off the ladder
  (no memory request known) nothing is started and VALIS's 10 GB stands. What each run
  really had is in the tables as the JVM reported it: `jvm_max_heap_gb` beside
  `jvm_heap_asked_gb` in `resources.csv` and `summary_resources.csv`, and one line above
  the resources table of `summary.md` per row of a JVM method.
- `valis_mirage`: the pipeline's call is run as it is, with `jvm_heap_gb` from the same
  formula. Whether that value reaches the JVM is recorded, not assumed
  (`info.jvm_max_heap_gb`, column `jvm_max_heap_gb`; "Not verified"): where the column
  reads 10 while `jvm_heap_asked_gb` reads more, the request did not arrive, and the
  row ran with VALIS's own 10 GB.
- `valis_mirage`'s micro pass: `bin/register.py` catches ANY exception of
  `register_micro` and writes the non-rigid result. When that exception is a memory
  error the `valis_mirage` row is FAILED with `cause: oom` and the patient goes up a
  rung, instead of being kept with a `problem`: the row's name is the full method, every
  other method's memory failure is retried with more memory, and a cheaper stage under
  the full method's name would misdescribe the method and flatter its cost. The
  `valis_mirage_nonrigid` row still holds that stage. A micro pass that fails for another
  reason is kept as the pipeline ships it, with `problem`.

**Which file each row reads.** The patient's ORIGINAL multichannel slides
(`case.ref_image`, every round's `case.mov_image`), hard-linked under their own names
into one folder (a copy where a link is impossible, never a symlink), and read by VALIS's
own readers. That is what the pipeline's REGISTER is given. VALIS picks its registration
channel by name (`dapi`): 1.0.0 by a regular expression, channel 0 on no match or on
several (`slide_io.py:723-747`); 1.2.0 by a fuzzy match (`slide_io.py:1074-1099`). The
index it picked is in `run.json: info.nuclear_channel`.

**The point map is the map of the written image.** For output pixel `P` of a slide
written with `crop="reference"`, VALIS's writer (`warp_tools.warp_img`) samples the moving
slide at `warp_xy_inv(P + c)`: `c` is the crop box's origin as the writer used it, and
`warp_tools.warp_xy_inv` reads the backward field `bk_dxdy` in its own grid and applies
the slide's matrix. So every VALIS row is `REF_TO_MOV`, nothing is inverted, and the
function that moves a point is VALIS's own.

- **It is not `Slide.warp_xy_from_to(reference -> moving)`**, which the plan named. That
  function places a reference pixel on the registered canvas through the reference
  slide's own matrix, `(P / s - t) x canvas / reg`; the writer places it at `P + c`. The
  two differ by the rounding of the crop origin (up to 1 px) and by the ceiling in the
  canvas size (up to 1 px at the far edge). On the stand-in's geometry that is 0.46 px,
  which the selfcheck reports as `inconsistent`. Each run records the gap on its own
  geometry (`info.stage.valis_from_to_minus_writer_px`), and `--opt points=from_to`
  scores VALIS's own function instead, so the claim can be measured in the image.
- **The rounding of the crop origin is measured, not assumed.** 1.2.0 takes `np.ceil` of
  the box (`warp_tools.py:1118`); 1.0.0 hands the real number to pyvips (`:1097`). The
  adapter passes an 8 px probe through the installed `warp_tools.warp_img` and reads
  where the crop began. `--opt crop_round=<ceil|trunc|round>` states the rule instead.
- **In 1.0.0 the reference slide is itself resampled** by the writer (through its matrix
  and the same crop), so the pipeline's two registered files agree with each other
  through `warp_xy_from_to`, while the benchmark's reference nuclei are on the NATIVE
  reference grid. This map relates the written moving image to that native grid. In 1.2.0
  the reference is written unwarped (`registration.py:955-974`).
- **Two differences the selfcheck measures** and nothing corrects: points read the field
  through VALIS's cubic spline, the image through libvips' bicubic; the writer's rigid
  step goes through a four-corner projective fit of the same chain.
- **Fields are copied at the yield**, through `Slide.bk_dxdy` as the writer reads it; a
  stored (tiled) field is loaded whole as an array, so points take VALIS's vectorised
  branch instead of its per-point loop.

**Stages, images, cost.**

| variant | what it is | image |
|---|---|---|
| `valis_mirage_rigid` | the matrix after the pipeline's micro-rigid refinement, no field | selfcheck only |
| `valis_mirage_nonrigid` | the field as it stood when the pipeline called `register_micro` | selfcheck only |
| `valis_mirage` | the final transform | always: the file the pipeline's function wrote |
| `valis120_rigid` / `valis120_hires_rigid` | the matrix (after micro-rigid for hires) | selfcheck only |
| `valis120` / `valis120_hires_nonrigid` | after `register()` | always, written BEFORE `register_micro` rewrites the fields |
| `valis120_micro` / `valis120_hires` | after `register_micro()` | always |

- `valis_mirage` captures its stages inside the one call through three wrappers
  (`Valis.register`, `Valis.register_micro`, `Slide.warp_and_save_slide`) that copy
  matrices and fields out, pause the clock around the writer, and change no argument and
  no result. `register_micro`'s wrapper always calls the original.
- A written image has the reference slide's size and is on its grid: **no `origin_xy`**.
  Pixels the moving slide does not cover are **filled** with the slide's background colour
  (the value of its darkest pixel; zero when the writer is given another path), so the
  native record carries **`valid_box`**: the bounding box of the lattice points the map
  sends inside the moving slide, grown by one step.
- Cost: `valis120`'s three results are billed at their own yields. All three
  `valis_mirage` results carry the cost of the whole call; `info.cost_after_register` is
  the bill when `Valis.register()` returned. The writer is never billed.
- CPUs: nothing is exported. `valis_mirage` passes the allocation as `n_cpus`, which the
  pipeline's `bound_cpu_count` turns into `multiprocessing.cpu_count()` (VALIS 1.0.0's
  pools use that minus one). VALIS 1.2.0 has no option and takes the affinity mask minus
  one. `info.workers_seen` is what VALIS saw.
- `info` also records: `micro_ran` (and `problem` when the micro pass did not run),
  `repair_fired`, the µm per pixel of the rigid image and of the field
  (`info.stage.um_per_px`), whether the field was stored (the tiler), the non-rigid
  bounding box, and for 1.2.0 the last slide of `slide_dict` (its micro pass prepares
  every slide's image with the last slide's field, `registration.py:3944-3968, 4045-4062`).

**What the selfcheck reaches.** `SIZES` checks every VALIS row on 2048 x 2560 (a pair, a
pair with an overhanging moving slide, and one patient of three slides: see "selfcheck")
and on 24000 x 30000 against 23000 x 29000. On the small pair `valis120_hires`'s micro pass
cannot run (its size, 640 px, is under the first pass's 2048 px, and 1.2.0 returns
early: `registration.py:4862-4871`), and the pipeline's runs on a coarser grid than the
first pass; on the large pair all three micro passes run and the first field is applied
at 14x. Not reached by any pair: stored (tiled) fields, which need more than 10 GB of
images at registration size. That path, and field scales of 29x, are first exercised by
the realcheck on a cohort patient.

**First commands in each image** (from the repository root; `ok` is exit 0, and
`selfcheck [<variant>]: ok` for all three variants of the row):

```bash
# the pipeline's VALIS image (valis-wsi 1.0.0)
python -m benchmarks.regbench selfcheck --method valis_mirage --work W/valis_mirage
# the VALIS 1.2.0 image
python -m benchmarks.regbench selfcheck --method valis --label valis120 --work W/valis120
python -m benchmarks.regbench selfcheck --method valis --label valis120_hires --opt hires=1 \
    --work W/valis120_hires
```

Add `--size 2048x2560` for the small pair alone (minutes, a few GB), or `--size
2048x2560:2048x2560:2048x2560` for the patient of three slides alone; without it every
case runs, the large pair included. `python -m pytest benchmarks/tests/regbench/test_external_methods.py`
runs the small pair of each row whose VALIS version is installed.

### ASHLAR (`methods/ashlar.py`, `tilemap.py`, `benchmarks/ashlar/`)

**What runs.** ASHLAR's own command, once per patient, through
`benchmarks.ashlar.original.run_ashlar` (`ashlar.scripts.ashlar.main`, unmodified):

```
ashlar 'filepattern|<reference tiles>|...' 'filepattern|<round 1 tiles>|...' ... \
    -o ashlar_output.ome.tif -c <nuclear channel index> -m 15.0
```

The reference cycle is stitched by `EdgeAligner`, each later cycle aligned to it by
`LayerAligner`, and ONE mosaic is written by `PyramidWriter`. `-m 15` is ASHLAR's default.
Because it is one process writing one file, a patient's rounds fail together.

| run it as | options | what it is |
|---|---|---|
| `--method ashlar` | `affine=none noise_frac=0.002 max_shift_um=15` | slight noise on the reference tiles. **ASHLAR's primary row** (rule R3.12) |
| `--method ashlar --label ashlar_clean` | `noise_frac=0` | no noise: **ASHLAR AS PUBLISHED, the default cell** (`-m 15`, no benchmark-side noise) |
| `--method ashlar --label affine_ashlar` | `affine=stare` | moving tiles placed by STARE's anchor, slight noise. A hybrid, never ranked as ASHLAR |
| `--method ashlar --label affine_ashlar_clean` | `affine=stare noise_frac=0` | the same hybrid, no noise |
| `--method ashlar --label ashlar_a<affine>_n<0\|002\|01\|05>_s<shift>` | the other 20 cells | the rest of the factorial |

**The factorial** (ruling R5; `ashlar.ASHLAR_CELLS`, `variants._ashlar_runs`): `affine`
{none, stare} x `noise_frac` {0, 0.002, 0.01, 0.05} x `max_shift_um` {15, 60, 240} =
2 x 4 x 3 = 24 runs, every cell run (the as-published cell is in it, so every cross contains
each factor's default). The four names above keep their cells; every other cell is
`ashlar_a<affine>_n<tag>_s<shift>`. Why these factors, and no others: `noise_frac` decides
whether ASHLAR's error metric can raise on pixel-identical overlaps (`identical_overlap`,
`utils.py:74-86`); `max_shift_um` decides when ASHLAR silently falls back to its linear model
for a tile whose shift is over the limit (`-m` is in micrometres, converted by the reader's
pixel size); `affine` decides whether a rotation reaches the shift budget at all (the
anchored rows place the moving tiles where STARE's matrix puts them). `filter_sigma` causes
no error and is not crossed. The older `--opt noise=ref|none` is still accepted (0.002 | 0).
`ashlar_a*` rows whose `affine` is `stare` form the family `affine_ashlar` and wait for
`stare`; the others form `ashlar`; the primary stays `ashlar`.

A label and its options come together; a mismatch is refused.

**The tiles are the benchmark's** (ASHLAR "requires unstitched individual tile images as
input", README at v1.20.0 `[PARTIAL: lines 1-60]`; the slides exist only stitched).
`benchmarks.ashlar.retile.write_tiles` cuts every slide of the patient on one regular
grid: 1950 px tiles at a 1755 px pitch (10 % overlap), all channels, whole pixels, no
stage error. 1950 px is the tile of the scanner the cohort was acquired on at 0.325 µm
per pixel, and the overlap is the owner's ruling (`ashlar.TILE`, `ashlar.OVERLAP`).

- *Sliver tiles.* The grid has ceil(size / 1755) tiles per axis and every tile is a full
  1950 px square (ASHLAR reads one tile size for the whole grid), so the last row and the
  last column are padded with zeros wherever the slide ends: on a 60000 x 80000 px slide
  35 x 46 tiles, 80 of them partly padding, the last row holding 330 px of slide and the
  last column 1025 px. When the slide ends within 195 px of a tile start, the last tile
  holds nothing the tile before it does not. On the reference cycle the padding carries
  the per-tile noise too. Tested against the stand-in on two such canvases, at the
  frozen tile size (`test_ashlar_adapter.py::test_sliver_tiles_on_both_axes`: 2000 x
  2060, whose last tiles hold 50 and 110 px of their own, and 1905 x 5400, whose last
  row and column hold nothing of their own): the map, the seam mask against plain rectangle tests, the
  origin and `valid_box`. What ASHLAR itself does with a tile that is mostly padding
  (whether it registers its edges or places it by its linear model) has not been
  observed: the stand-in registers nothing.

- *Slight noise* is per-tile Gaussian noise on the REFERENCE cycle's tiles only, s.d. =
  max(0.2 % of the channel's 1st to 99th percentile range, 3 counts; the floor assumes
  16-bit data). On pixel-identical overlaps ASHLAR's error metric can raise
  `RuntimeError("correlation > total_amplitude ...")` (`utils.py:74-86`); its command
  catches `ProcessingError` only (`scripts/ashlar.py:231`), and only the reference cycle's
  edge alignment can raise (`:261-262`). So the two clean rows fail together, per patient,
  where they fail; the adapter catches the exception and the patient's rows carry its
  message (`info["ashlar_raise"]`) and the cause `identical_overlap` (the benchmark's: the
  identical overlaps exist only because it cuts tiles from a stitched slide). That cause
  is given only on a cell without noise (`noise_frac` 0): with noise on the reference the
  overlaps are not identical, and the same raise is ASHLAR's own, `exception`. Any other
  `RuntimeError` stays `exception`; a memory error is not caught: it stays `oom`.
- *Silent fallbacks* are flagged, never blocking. `info["ashlar_warnings"]` holds ASHLAR's
  `DataWarning`s; `fallback_stage_positions` is true when the reference stitch printed
  "Could not align enough edges, proceeding anyway with original stage positions."
  (`reg.py:698-699`, `fit_model`), `zero_overlap` when it printed "No tiles overlap, ..."
  or "Some neighboring tiles have zero overlap." (`reg.py:545-547`, `check_overlaps`).
  Per round, `discard_frac` is the share of the cycle's tiles that ASHLAR replaced by its
  linear model, and `cycle_fully_modelled` is true at 1.0: the cycle's own alignment
  decided nothing.
- *The anchor* (`affine=stare`): each moving tile is cut, unrotated and uninterpolated,
  where STARE's rigid matrix (`info["m0"]` in `<out>/stare/<dataset>/<case>/run.json`) puts
  the matching reference tile's centre; ASHLAR is told the regular grid. A round whose
  `stare` result is missing, failed or has no matrix fails alone as `anchor unavailable:
  <path>` with `cause: benchmark` (not ASHLAR's failure) and `anchor_cause` naming
  STARE's (`no record`, or the cause in STARE's `run.json`), and the patient's other
  rounds proceed. An anchored row is reused only while the matrix each of its rounds
  read is unchanged (`ashlar.REUSE_KEY`).
- Reference tiles are cut once per noise level and shared by the plain and the anchored
  row, under `<out>/cache/ashlar_tiles/<dataset>__<patient>/` (delete it when both rows
  have run; it is large). The moving cycles' tiles (noise-free in every cell) are cut once
  the same way, keyed by the slide, its cycle, the canvas and, for an anchored cell, the
  tile origins STARE's matrix gives (`mov_<hash>/cycle_<k>`): the 12 cells of one affine
  share them (`info["moving_tiles"]`, `moving_tiles_reused`), and `retile_moving_s` is what
  cutting them cost whoever cut them. A cut is written into a temporary folder of the
  job's own (`tempfile.mkdtemp`, never the process id, which two nodes can share) and
  published by one rename. ASHLAR's reference STITCH is not shared: each command does its
  own, and `info["edge_positions_identical"]` says whether the two came out equal. The
  reference tiles depend on the noise level only, so they are shared by every cell of one
  `noise_frac` (six cells: 2 affine x 3 `max_shift_um`); the stitch comparison is made only
  between cells at the same tiles AND the same `max_shift_um` (the edge files are keyed by
  both), because a different shift can stitch the same tiles differently. The cache name of
  the tiles changed with the factorial (`ref_n<frac>_...`, was `ref_ref_` / `ref_none_`): an
  existing cache misses once and the tiles are re-cut (outside the clock).

**The point map is where ASHLAR drew each tile, exactly.** `Mosaic.assemble_channel`
pastes tile `i` at `aligner.positions[i]` (`reg.py:1147-1153`) and `utils.paste` puts tile
pixel `l` on mosaic pixel `position + l`, the whole part by slicing and the fractional
part by `scipy.ndimage.shift` (`utils.py:149-194`). The wrapper notes both cycles'
`positions` once each aligner has returned, and `grid.json` records where each tile's
pixels came from. A moving point goes, in ASHLAR's own direction (`mov_to_ref`):

```
moving slide --(+ L[t] - c_mov[t])--> mosaic --(- (E[s] - c_ref[s]))--> reference slide
```

`t` is the moving tile that holds the point, `s` the reference tile drawn where the point
lands in the mosaic. Both steps are one translation per tile; nothing is inverted
numerically, fitted or resampled. The tile order is read back from ASHLAR's own positions
(`filepattern.py:90-100`: row-major), not assumed. `run.json: info.placement` holds the
four arrays: it is the whole transform.

**The mosaic is not on the reference slide's grid.** ASHLAR re-stitches the reference and
moves its origin to zero (`reg.py:713-716`). The native record says where the image is:
`origin_xy` = where the mosaic's first pixel sits in the reference slide's frame (minus
the median over tiles of `E - c_ref`, each tile weighted by the slide area it holds),
`valid_box` = the reference slide's extent inside the mosaic (outside it the image is the
tiles' zero padding), `reference_file` = the reference cycle as ASHLAR redrew it. How far
single tiles depart from the plain median is `info["reference_stitch_error"]`; the scored
map uses each tile's own placement, not the median.

The weight is what makes one origin describe the image on a canvas with sliver tiles.
Counted one tile one vote, the slivers of the last row and column are a majority on any
canvas of up to 3 x 3 tiles, and the origin would then be that of a few hundred pixels at
the slide's edge (with the stand-in's stitch on a 2 x 2 canvas: (-6.9, -4) reported, no
tile's own offset on x, while the image read off the matched dots says (0, 0), and the
selfcheck's reference comparison objects). On a cohort slide (35 x 46 tiles, 80 slivers
of 1610) the two medians agree unless ASHLAR placed its tiles very unevenly.
`info.placement.ref_valid_wh` holds the weights.

**Where the map is not exact: seams.** Where two tiles of a cycle cover one mosaic pixel
ASHLAR cross-fades them (`pastefunc_blend`, `utils.py:197-216`), and where two tiles hold
one point it may be drawn twice. The map there is the mean of the covering tiles' shifts
weighted by depth: an approximation, which matches ASHLAR's blend only in two-tile bands
on a non-zero background. A moving point is ON a seam unless one moving tile holds it in
the slide, it lands under that one moving tile's footprint and no other's in the mosaic,
and one reference tile is drawn where it lands (`tilemap.on_seam`, with 1 px to spare).
`info["band_mov"]` marks the moving nuclei with any vertex on a seam, and the rows
`<variant>:non_ashlar` are every method scored without them ("Off the seams"). They are
the rows to lean on for ASHLAR; `info["band_fraction"]` says how many nuclei that is.

**What the overlap bands cost (ruling R3.15: the primary row keeps scoring ALL nuclei).**
Re-measured on 2026-10-11 with `tilemap.py` on uniform points (the review's item 5; its
numbers hold):

- **How many.** At 1950 px tiles and a 1755 px pitch a point is in a moving-tile overlap
  with probability 1 - (1 - 195/1755)^2 = 0.21 on an endless mosaic, and 0.21 on a 40000
  x 50000 px slide (667 tiles) and on a 60000 x 80000 px one (1610 tiles; uniform
  points). `on_seam` also asks for one reference tile where the
  point lands: when the moving tiles sit in the mosaic offset from the reference tiles
  that grows, to 0.30 at 100 px and 0.39 to 0.40 from 300 px on. So between a fifth and
  two fifths of the nuclei of ASHLAR's primary row are scored through a map the image
  does not show exactly. These are exactly the dots the selfcheck masks, so that part of
  the map is not checked against any image.
- **No bias under pure translation.** When every tile needs the same shift, the two
  drawn copies coincide and the map equals both: the error is zero (measured 0.00 px).
- **With a residual rotation inside the tiles the map is OPTIMISTIC for ASHLAR.** ASHLAR
  moves each tile rigidly, so a rotation of the round is left inside every tile: the
  tile is right at its centre and wrong by (rotation) x (distance from the centre)
  elsewhere. In a band the image is a cross-fade of two such copies, each wrong by about
  half a pitch times the rotation in OPPOSITE directions along the line between the two
  centres; the map is their depth-weighted mean, which lies between the copies and so
  cancels that component. Measured, as mean error against the true rotation (uniform
  points on a 40000 x 50000 px slide, every tile right at its centre): at 0.5 degrees
  the map is off by 5.2 px in the bands while the copies actually drawn are off by 8.8
  px (off the bands both are 5.2 px); at 0.1 degrees 1.03 px against 1.77 px. So in the
  bands the nuclei are scored as better registered than either copy the image shows, by
  about 40 % of the copies' error.
- **How much it matters is the `:non_ashlar` rows.** They score every method, ASHLAR
  included, on the nuclei off the bands, where the map is exactly the image. If
  `ashlar:non_ashlar` and `ashlar` agree, the bands did not flatter it; if `ashlar` is
  the better of the two by more than the other methods' rows differ between the two
  blocks, that difference is the optimism. `affine_ashlar` removes most of the rotation
  before ASHLAR sees the tiles and is the row where the effect should be smallest.

**Also reported per round:** `discard_infinite_error`, `discard_zero_offset`,
`discard_over_budget` (why ASHLAR replaced a tile's position by its linear model, one
cause per tile in that order of precedence, rebuilt from `reg.py:910-946`;
`discard_rebuild_matches` is false if the three do not add up to the flag ASHLAR kept),
`jump_px_p50`, `jump_px_p95` (difference of `D(t)` between neighbouring tiles), ASHLAR's
data warnings, the exact command.

**Cost.** Every aligner finishes before the writer starts (`scripts/ashlar.py:262-296`),
so the clock and the memory peak stop after the last aligner. `register_s` is the whole
command up to there; `info["register_s_round"]` = that round's `layer_s` + `edge_s` /
rounds (an anchored row that dropped a round divides by fewer); `info
["peak_rss_after_register_mb"]` is the peak read at that moment. Tile cutting is not
billed (`retile_s`). `write_s` is ASHLAR's writer plus the copy, divided by the rounds.
ASHLAR has no core-count option and none is set; its official image sets
`OMP_NUM_THREADS=1`, which reaches the job and is left as it is, so ASHLAR runs on one
OpenMP thread of its 4 CPUs ("Cost", "Threads the method's own image sets"). It starts
a JVM on import with `-Xms10m -XX:+UseSerialGC` and no heap limit (`reg.py:29-38`); the
reader used here does not call it.

**Selfcheck: two comparisons against the file ASHLAR wrote, both from the image.**

1. The moving dots through the scored map, against the moving cycle's planes of the
   mosaic located by `origin_xy`. This is the comparison that tests the first step
   (`L`, `c_mov`, the tile index). The per-tile departure of the reference from the median placement is declared
   (`declared_offset: true`), so `origin_xy` cancels out of this comparison.
2. The REFERENCE slide of ASHLAR's selfcheck pair carries its own dots
   (`SELFCHECK_REF_DOTS`). Every dot found in the reference cycle's planes is sent back by
   the second step of the map and must land on a dot of the reference slide; and the
   reported `origin_xy` must equal the origin read off those matched dots
   (`reference.image_origin_px_median_x/y`), within 0.15 px. This is the comparison that
   tests `E`, `c_ref`, the lookup and the origin.

Neither has proved anything about ASHLAR yet: both have only run against a stand-in
written from the same reading of ASHLAR's source as the adapter (`fake_ashlar.py`), so
they show that the adapter and the comparison agree with each other. Moreover the
selfcheck tiles are cut on the exact grid, so on the real package the reference step
`-(E - c_ref)` is the identity there and its sign and origin cannot fail: the cohort is
its first test.

The variant's status is the worse of the two. Dots within 8 px of a seam are left out
(`--mask overlap`, `n_masked`); without the mask the check fails, as it should, because
band dots are drawn twice. With a stand-in for ASHLAR whose reference is stitched 7 px
off its slide's grid, a negated origin, a reversed map, a map labelled the other
direction, two tiles swapped, a one-tile offset, a shifted reference index and a reference
term of the wrong sign are each reported `inconsistent` (`test_ashlar_adapter.py`).

**Not checked by reading alone:** nothing here has run ASHLAR. Unverified until the
selfcheck runs in the image: that ASHLAR 1.20.0 stitches the noisy tiles of the selfcheck
pair at all; how often the clean rows raise on the cluster's BLAS; the rounding of
`scipy.ndimage.shift` at the pinned SciPy against the 0.15 px threshold; run time and memory.

**First commands in the image** (`labsyspharm/ashlar:1.20.0`, from the repository root,
with the pinned STARE release, `tifffile` and `zarr` importable: `benchmarks/ashlar`
imports `stare.manifest`):

```bash
python -m benchmarks.regbench selfcheck --method ashlar --mask overlap --work W/ashlar
python -m benchmarks.regbench selfcheck --method ashlar --label ashlar_clean \
    --opt noise=none --mask overlap --work W/ashlar_clean
# the anchored rows read STARE's matrix from a STARE selfcheck on the same cases
python -m benchmarks.regbench selfcheck --method stare --work W/stare          # STARE's image
python -m benchmarks.regbench selfcheck --method ashlar --label affine_ashlar \
    --opt affine=stare --mask overlap --m0-from W/stare --work W/affine_ashlar
python -m benchmarks.regbench selfcheck --method ashlar --label affine_ashlar_clean \
    --opt affine=stare --opt noise=none --mask overlap --m0-from W/stare \
    --work W/affine_ashlar_clean
```

`ok` is exit 0 and `"status": "ok"` in `W/<row>/selfcheck.json`, for all three entries of
`pairs`: the 8192 x 10240 pair (5 x 6 tiles of 1950 px with four-tile corners) and both
rounds of the patient of three slides on the same canvas (two moving cycles in one
mosaic); in each entry, `variants.<row>.reference.status` is `ok` too and `masked_frac`
is at most 0.50 (expected 0.19 to 0.28, anchored rows 0.20 to 0.25: see "selfcheck"). A
clean row may end `crashed` with
ASHLAR's `correlation > total_amplitude`: that is the known numerical failure, it is
published and reported (`ashlar_clean` is checked by `ashlar`'s selfcheck,
`affine_ashlar_clean` on `affine_ashlar`'s: the point-map code is the same).

### DeeperHistReg (`methods/deeperhistreg.py`)

Everything below describes the cohort's one-channel path.

**The software is DeeperHistReg 1.0.1; the method is RegWSI** (Wodzinski et al. 2024,
doi:10.1016/j.cmpb.2024.108187). There is no separate RegWSI code: the paper says the
method "is incorporated into the DeeperHistReg framework", and the package is where it
ships. What follows about the paper is taken from this port's RegWSI report, which read
arXiv:2404.13108v2 `[FULL]` (TeX, the three table files, all captions) and the journal
version `[ABSTRACT]`; the paper was not re-read for the adapter. The code was read at
MWod/DeeperHistReg main @ `42e7c9d` `[PARTIAL: the files listed in the adapter's
docstring]`, which the report found identical to the PyPI 1.0.1 wheel but for two data
paths.

**What runs.** `deeperhistreg.run_registration` in its native roles (source = the moving
slide, target = the reference) with one of two shipped parameter factories, deep-copied,
in which only these keys are changed (`changed_keys`, recorded per run with the whole
parameter set): every `device` and `cuda` key, and `save_final_images = False` (the
writer is run afterwards, outside the clock, as the package's own example does).

| rows | factory | working size (short side) | in the paper |
|---|---|---|---|
| `deeperhistreg_gpu`, `_cpu` | `default_initial_nonrigid` | 4096 px, 8 levels, 900 iterations | the 4096 px point of its resolution ablation: 1.01 µm median error on the paper's re-stained-tissue landmark set |
| `deeperhistreg_hr_gpu`, `_cpu` | `default_initial_nonrigid_high_resolution` | 8192 px, 9 levels, 1100 iterations | the resolution of its 0.59 µm result on that set |
| the same four with `_fluo` | the same, plus `pad_value = 0` and `flip_intensity = False` | | not in the paper: configured by the benchmark |

- **These are the shipped sets, not the paper's.** The paper's exact parameters are not
  published. The maintainer says "the configs used to produce the results reported in
  RegWSI paper were slightly different than the default ones" (issue #15 `[FULL]`). Where
  the shipped sets visibly depart from the paper's text is recorded with every run
  (`info.implementation.departures_from_paper_text`): a scale-invariant feature
  transform + RANSAC candidate also runs in the initial alignment; the transform is a four-parameter similarity where the
  paper says affine; the 8192 px set has nine initial-alignment sizes where the paper's
  tables stop at eight; the writer enlarges the field linearly where the paper says
  B-splines; the intensity flip is not in the paper's text.
- **"8192" is the short side**, and the package never enlarges
  (`dhr_utils/utils.py:530-540`). It loads at 0.2 and then resamples the short side down
  to the working size. On this cohort (0.325 µm per pixel):

  | slide short side | 4096 px set | 8192 px set |
  |---|---|---|
  | 40000 px | S = 9.77, 3.17 µm per working pixel | S = 5.0, 1.63 µm (loaded 8000 px is already below 8192) |
  | 50000 px | S = 12.21, 3.97 µm | S = 6.10, 1.98 µm |

  `S` is full-resolution pixels per working pixel; each run records its own (`info.S_x`,
  `info.S_y`, `info.implementation.working_um_per_px`, `initial_resample_ratio`).
- **Fluorescence.** The authors state that DAPI needs no parameter change (the paper, per
  the report; the package README lists fluorescence microscopy and DAPI as tested,
  `README.md:16-30` `[PARTIAL: lines 10-32]`; issue #8 `[FULL]`: "it should be possible to
  register any H&E and IHC stains (and in fact also others, e.g. ExM or DAPI)"), so the
  unmodified rows are
  the authors' stated use and **the `_fluo` rows are the benchmark's**: brightfield after
  the flip is bright tissue on black, which is unflipped DAPI, so `flip_intensity = False`
  restores the polarity the method was built for. They are never primary.
- **Which setting is upstream's default, and which row is primary.** The package's
  default is the 4096 px set: `default_initial_nonrigid` is the factory its README's usage
  example calls (the repository's `README.md:103` `[PARTIAL: lines 88-112 and 222-236]`), and the
  8192 px `default_initial_nonrigid_high_resolution` is another shipped factory, listed
  there as "High Resolution". Both are run,
  on the GPU and on the CPU, each with its own selfcheck. The primary row is the 8192 px
  one, `deeperhistreg_hr_gpu`, by the user's ruling of 2026-10-11 (it is the
  resolution of the paper's headline result); the 4096 px rows stand beside it in every
  table. `variants.PRIMARY_DEEPERHISTREG` is the one line that names it.

**Inputs.** The package does not read OME-TIFF (issue #4 `[FULL]`) and its TIFF loader
counts top-level pages as pyramid levels (`tiff_loader.py:34-38`). Each slide's nuclear
channel is written as a tiled TIFF with ONE top-level page, same dtype and values,
uncompressed (`single_page_tiff`); the clock is paused meanwhile and `convert_s` recorded.
`info.grey_levels` is the number of 8-bit levels each channel keeps under the package's
min-max rule (counted on every fourth row and column): a few bright pixels can leave
16-bit DAPI in a handful of levels.

**One row per run, reference -> moving, not inverted** (user instruction of 2026-10-17).
`<label>` scores the image **as written**: for every pixel of the file on disk, the
moving-slide position the package's writer sampled (`asdrawn_map`). Its points are the
transform of the image, and its selfcheck declares no offset.

**Why DeeperHistReg can score low with a good registration:** its shipped writer is not
its registration. The field lives on the padded,
resampled grid, `warped(x) = source(x + u(x))`. The writer (`apply_deformation_pyvips`,
`dhr_utils/warping.py:224-239`) turns `grid + u` into an index image, enlarges it to the
full-resolution canvas with libvips' linear kernel, multiplies by the enlargement and
calls `mapim`. libvips enlarges with an input displacement of half a pixel, so output
pixel `X` reads the field at `X / S - 0.5` (libvips v8.15.2 `resize.c:233-277`,
`affine.c:445-480`, `:320-349` `[PARTIAL: those ranges, transform.c:158-252 and
mapim.c:225-300, 455-480, 589-594; not executed]`). Three consequences:

1. **Half a field pixel.** The written image samples `X - S/2 + S u(X/S - 0.5)`: its
   content sits about S/2 px right and down of the registration, 4.9 px at S = 9.77 and
   2.5 px at S = 5. The gap from the registration is `(S/2) x Jacobian x (1, 1)`, a
   function of position.
2. **The stretch.** The writer enlarges by `canvas / field size`. The field's size is a
   floor (torch's `interpolate` with a scale factor), so on a slide whose loaded size
   does not divide by the resample ratio the written image is stretched relative to the
   registration: with torch 2.3.1, a loaded 7974 x 11975 slide gives a field of 4095 x
   6151, not 4096 rows, and the written image is 2.6e-4 too tall, about 10 px at the
   bottom edge of a 39871 px slide.
3. **The pads.** The writer pads at full resolution, the registration padded the loaded
   images; the two agree only where the slides' sizes divide evenly.

The map uses the writer's own numbers for all three (`asdrawn_map`: full-resolution
centre pads, `S = canvas / field size`, the index image read at `X / S - 0.5` linearly
with the edge copied). Measured on the cohort on 2026-10-15, the written image sits
`S/2 + k` px off the package's own registration (`k` whole pixels from the padding
computed twice, with different rounding on uneven sizes), roughly 1 to 3 µm: the
benchmark scores what the package puts on disk, and that loss is the package's.

**The written image** (`write_native`): the package's writer with the arguments its own
`save_final` passes, once per channel of the moving slide (the maintainer's route for
multichannel images, issue #6 `[FULL]`), the first page of each of its files stacked into
`native/registered.ome.tif`, values unchanged. It has the reference slide's size and is
on its grid, because the writer crops the padded canvas back by the reference's own pads
(`pair_full_saver.py:76-80`, `dhr_utils/utils.py:299-311`): `origin_xy` is `(0, 0)` and
the pads enter the map, not the origin. Pixels from outside the moving slide are the pad
value (255 counts, or 0 for `_fluo`) inside the padded canvas and 0 beyond it;
`valid_box` bounds the pixels that come from the moving slide. A failed writer fails
the row, `<label>` (`writer`, or `writer_oom` for a memory error), as for every method
(a row is ok only when its image exists).

**Cost.** The clock runs over `run_registration` only, including the package's own
loading, its diagnostic JPEGs and intermediate fields; its own per-stage seconds are in
`info.stage_s`. The package has no core-count option; `info.torch_num_threads` and
`info.vips_concurrency` are what it saw. A `_gpu` row without CUDA is refused before the
package is called, because the package would fall back silently
(`full_resolution.py:58`).

**Failures.** `run_registration` prints the pipeline's exception and carries on
(`run.py:40-46`); the caller sees a missing `displacement_field.mha` (`:59-60`). stdout is
captured and the error raised with that text, so a CUDA out-of-memory is `gpu_oom`
(reported, never escalated), a host allocation failure is `oom` (the ladder), anything
else `exception`. The package prints only `str(e)`, so a bare `MemoryError` leaves
`Exception: ` with nothing after it and libvips' failure reads `out of memory`: both are
read as a host memory failure (`deeperhistreg.printed_memory_failure`; `CUDA out of
memory` stays `gpu_oom`).

**The checks.** The selfcheck compares the row with the one image the writer produced,
as it is (no declared offset). With a stand-in for the package whose writer is drawn
from the source, a wrong direction, dropped or exchanged pads, a 1 % scale error, a
missing half pixel and a field of the wrong sign are each reported `inconsistent`
(`test_dhr_adapter.py`). The realcheck runs on that row.

**What the selfcheck reaches.** One pair for all eight labels, 40000 x 50000 against
40000 x 48000: S = 9.77 at the 4096 px set (the cohort's class) and S = 5.0 at the
8192 px set, with every division exact. A coarsest pyramid level of 31 to 32 px. Not
reached: the 8192 px set's second resampling (a slide whose short side exceeds 40960 px:
S between 5 and about 6.1), and sizes that leave remainders, where the row is expected to
hold (emulated in `test_dhr_adapter.py`); on the cohort that is the realcheck's ground.
The 8192 px set on the CPU is 1100 full-field iterations at 80 megapixels: hours, and
tens of GB.

**Not checked by reading alone:** nothing here has run DeeperHistReg or libvips.
Unverified until the checks run in the image: the half-pixel offset itself (the settling
measurement below); that the package's loader and writer accept a one-channel 16-bit
TIFF (its pad colour is a three-element vector, `dhr_utils/utils.py:220-221`); that
`tifffile` reads the writer's LZW pyramid; run time and memory at any size.

**First commands in the image** (from the repository root; DeeperHistReg 1.0.1 with
`torch >= 2.3`, SimpleITK and pyvips, plus the pinned STARE release, `tifffile`, `zarr`
and `imagecodecs`):

```bash
# 1. the settling measurement, seconds: a zero field must be written 2.5 px right and down
python -m pytest benchmarks/tests/regbench/test_external_methods.py -q -rs \
    -k "zero_field or factory or centre_pads" --basetemp=/tmp/rb_dhr
# 2. the small pair on the CPU, the run's row, shipped and fluo (S = 5)
python -m pytest benchmarks/tests/regbench/test_external_methods.py -q -rs \
    -k deeperhistreg_points --basetemp=/tmp/rb_dhr2
# 3. the selfcheck of each label (the cohort-class pair)
python -m benchmarks.regbench selfcheck --method deeperhistreg --label deeperhistreg_cpu \
    --opt device=cpu --work W/deeperhistreg_cpu
python -m benchmarks.regbench selfcheck --method deeperhistreg --label deeperhistreg_gpu \
    --opt device=cuda:0 --work W/deeperhistreg_gpu
python -m benchmarks.regbench selfcheck --method deeperhistreg --label deeperhistreg_hr_gpu \
    --opt device=cuda:0 --opt config=default_initial_nonrigid_high_resolution \
    --work W/deeperhistreg_hr_gpu
# ... and _hr_cpu, _fluo_gpu, _fluo_cpu, _hr_fluo_gpu, _hr_fluo_cpu (add --opt fluo=1)
```

`ok` is exit 0 and, in `W/<label>/selfcheck.json`, `"status": "ok"` with
`variants.<label>` `ok`. Add `--size 8200x10240:7780x9720`
for a quick S = 5 run of any label. If step 1 fails, the offset is not what the source
reading predicts: report the measured shift and stop.

### PALOM (`methods/palom.py`)

Sources: labsyspharm/palom at tag `v2026.4.1` (= PyPI `palom==2026.4.1`) `[PARTIAL: read in
full align.py, block_affine.py, pyramid.py, __init__.py; register.py:1-316;
reader.py:1-175; img_util.py:1-47; README.md:214-284; cli/align_he.py:150-180; missing:
the multi-resolution and multi-object aligners, register_dev.py, register_util.py, the
stores and the other command-line tools]`; PALOM issues #12 and #13 `[PARTIAL: the
maintainer's comments]`; the maintainer's example script for multi-cycle IF,
`Yu-AnChen/dump` `2023-09/palom-example-script.py` at `61c8e1a2` and `333ebbb5` `[FULL]`
(**the maintainer's personal script, not an official PALOM source**). There is no
publication; the README is the method's description. Nothing here was run with PALOM.

| run | what it is | set by the benchmark |
|---|---|---|
| `--method palom` | the README's scripting recipe (`README.md:220-284`): `OmePyramidReader(path)`, `Aligner` with its six keywords, `coarse_register_affine(n_keypoints=4000)`, `compute_shifts()`, `constrain_shifts()`. PALOM's primary row | the nuclear channel by index; `LEVEL = 0`; the thumbnail level from `get_thumbnail_level_of_size(2000)` (the README's suggestion) |

The run yields `palom_affine` (the coarse affine alone) and `palom` (the affine and one
shift per block), both `ref_to_mov`. Pair scope, no GPU, no JVM.

**What PALOM does.** A full six-parameter affine from ORB keypoints on thumbnails, then
one whole-pixel shift per block by cross-correlation of Laplacian-filtered blocks, then a
filter that replaces the shifts that disagree with a linear trend over the block grid
(and those of blank blocks) by the trend's value. **A block is one dask chunk of the
reference array, which is the reference file's TIFF tile**: 2048 px (666 µm) on the
cohort's preprocessed slides, 1024 px on the generated cases. PALOM has no block-size
parameter and the benchmark sets none: the block of the `palom` row is that of each
file's layout. Inside a block everything moves together; neighbouring blocks may jump.

**The point map** is exact and closed-form, reference -> moving:
`written(P) = bilinear(moving, A^-1 (P - s_b(P)))`, with `A` = `aligner.affine_matrix`
(moving -> reference, homogeneous x, y), `s_b` = `aligner.shifts[b]` given as `(dy, dx)`
and row-major over `aligner.grid_shape`, and `b(P)` the block of `aligner.ref_img.chunks`
that holds pixel `P` (line references in the module's docstring). The only inverse is
the 3 x 3 affine's, which PALOM's writer takes itself. The other direction is not a
function: where two neighbouring shifts differ, some moving points are drawn twice and
some never. For a point between pixel centres a block edge sits half a pixel before the
edge pixel. Before writing, the adapter checks that the aligner's affine, shifts and
chunks are still the ones the map was made from, and hands the writer `aligner.ref_img`,
so the image is drawn on the blocks of the map.

**Seams.** The map jumps at an edge between two blocks whose shifts differ, and the image
is torn there. `band_ref` marks the reference nuclei whose outline's bounding box, grown
by 1 px, touches blocks that do not all carry the same shift. An edge between two
blocks with the same shift is not a seam. Every method gets `<variant>:non_palom` rows on
identical nuclei.

**The written image** is PALOM's: `block_affine_transformed_moving_img` then
`write_pyramid(mosaics=[it], output_path, pixel_size)` with the writer's defaults
(pyramidal BigTIFF, zlib, 1024 px tiles, channels anonymous): every channel of the moving
slide on the reference grid, 0 where nothing lands. `native` has `valid_box` and no
`origin_xy`. The affine stage is drawn by the same calls with `mxs=aligner.affine_matrix`
(the package's coarse-only path), for the selfcheck only.

**What a run records** (`run.json: info`): `version`, `RECIPE`, `affine`,
`coarse_identity_placeholder`, `thumbnail_level`, `thumb_um_per_px`, `ref_down_factor`,
`mov_down_factor`, `n_levels`, `file_levels`, `auto_pyramid`, `block_px`,
`grid_shape`, `n_blocks`, `n_blocks_blank`, `n_blocks_replaced`,
`integer_shifts`, `shift_nonzero_fraction`, `shift_px_p50/p95/max`, `n_block_edges`,
`n_seams`, `seam_jump_px_p50/p95`, `band_fraction`, `pixel_size_detected`, `workers_seen`
and `concurrency` (what dask, OpenCV and `os.cpu_count()` saw; `psutil` is required and
nothing is set), `figures`, `closed_file`, `blocks` (the affine, block edges and shifts
the map is made of) and `implementation.benchmark_side`.

**Cost.** The clock stops after `constrain_shifts()`; `palom_affine`'s clock stops after
the coarse call. The writer runs with the clock paused. PALOM has no worker option: dask's
thread pool takes every core it can see, and it sees the job's affinity mask only when
`psutil` is importable, which is why the adapter refuses to run without it. The matplotlib
figure the documented coarse call opens is saved to the work folder and closed with the
clock paused.

**Known risks, reported as they are.**

- *Silent identity.* Failed feature matching logs a warning and continues with the
  identity (`register.py:152-154`). The rows are kept and carry `problem: coarse matching
  failed; PALOM continued with the identity`.
- *Equal shifts crash the filter.* A transcription of `align.py:54-80` run here with
  scikit-image 0.24.0 and scikit-learn 1.7.2 (not PALOM, not its pinned scikit-image
  0.19.3) raises `ValueError: Found array with 0 sample(s)` when every finite block shift
  is the same, all-zero included: no block is under the threshold, so the regression has
  nothing to fit. That is what a purely rigid map, or a pair the affine
  already aligns to the pixel, may give. The `palom` row then fails with PALOM's own
  message and `palom_affine` is kept; nothing is excluded.
- *`seek of closed file` on a file without pyramid levels* (issue #12, 2023), which is
  the cohort's file type. Whether 2026.4.1 still does it is unknown. The adapter runs
  the recipe as it is; only if a call raises that error does it apply the maintainer's
  workaround (read the first block of level 0 of each slide once and drop it, from their
  personal script), repeat the failed call, record it (`closed_file`) and list it under
  `benchmark_side`. `--opt preread=1` applies it up front, `--opt preread=0` never.
- *The maintainer's script is for an older PALOM.* It uses `get_aligner`, 5000 keypoints,
  a thumbnail of 2000 px (500 in the version written for the issue's files), channel 0
  (2 there) and `detect_flip_rotate=True`, an argument `coarse_register_affine` no longer
  takes at 2026.4.1; no mask and no other channel handling. The primary row stays the
  README's recipe.
- *Memory.* A whole full-resolution moving channel is passed to each block as a keyword
  and is held in memory; the writer does that per channel. Unmeasured.

**Selfcheck.** Always with `--mask seams`: a dot drawn across two blocks with different
shifts is torn in the image, and its centroid follows neither half (without the mask the
stand-in's `palom` row is `inconsistent` for that reason alone). The mask leaves out
detections within 8 px of such an edge; the affine stage has none. Pairs: two,
8 x 10 blocks each (the second with a smaller moving slide, so the two thumbnail
down-factors differ), in files with 2048 px tiles, the cohort's layout: 16384 x 20480,
so the blocks are the 2048 px ones of the cohort row. If under 10 %
of the blocks have a shift of at least 0.5 px (`MIN_SHIFT_PX`; `run.json:
info.shift_meaningful_fraction`) the `palom` variant is `not_checked`: the block map
then equals the affine map to within the threshold and a wrong sign or block order could not
show. Until 2026-10-11 the guard counted every non-zero shift, which a block filled from
PALOM's regression with a few hundredths of a pixel satisfies while testing nothing; at
0.5 px a reversed sign moves a dot by 1 px, 2.5 times the threshold.

What the dots cannot see, and what covers it:

- a block edge one pixel off: those pixels are on the seams the mask leaves out. The
  pixel-for-pixel test draws a ramp through PALOM's own
  `block_affine_transformed_moving_img` and compares every output pixel with the map;
- content PALOM draws twice or never at a seam (a backward check counts neither);
  `seam_jump_px_*` and `band_fraction` say how much is at stake;
- a slide of the cohort's size (the `palom` row's 2048 px blocks are checked since
  2026-10-11, on 8 x 10 of them; a cohort slide has about 30 x 30): the realcheck is
  the check there.

**First commands in a PALOM image** (from the repository root; Python 3.10,
`palom==2026.4.1` with its own pins, `psutil`, plus the pinned STARE release installed
without its dependencies, since it asks for a newer scikit-image than PALOM allows):

```bash
# 1. the pin, and the cohort's file type: a one-level OME-TIFF must be readable
python -m pytest benchmarks/tests/regbench/test_external_methods.py -q -rs \
    -k "pinned_one or single_level or tile_grid" --basetemp=/tmp/rb_palom1
# 2. PALOM's own writer against the point map, pixel for pixel (seconds)
python -m pytest benchmarks/tests/regbench/test_external_methods.py -q -rs -W always \
    -k "pixel_for_pixel or equal_shifts" --basetemp=/tmp/rb_palom2
# 3. the selfcheck
python -m benchmarks.regbench selfcheck --method palom --mask seams --work W/palom
```

`ok` for 1 and 2 is every test passed and none skipped; step 2 prints, as a warning, what
PALOM's filter does on equal shifts. If step 1 fails with `seek of closed file`, the
workaround is needed: step 3 still runs (the adapter applies it) and `closed_file.applied`
is true in each `run.json`. `ok` for 3 is exit 0 and, in `W/palom/selfcheck.json`,
`"status": "ok"` with `variants.palom_affine` and `variants.palom` both `ok` on both
pairs, `n_masked` a few percent of `n_detected` for `palom` (under 5 %: the cap) and 0
for `palom_affine`.
`inconsistent` on `palom` with step 2 green means torn dots outside the mask or a shift
the image does not show: report the numbers and stop.

## Tests

```bash
python -m pytest benchmarks/tests/regbench -q --basetemp=/tmp/rb_<unique>
```

`test_external_methods.py` skips where VALIS, ASHLAR, DeeperHistReg or PALOM (each at the
pinned version of the row) is not importable. `test_palom_adapter.py` runs the PALOM
adapter against `fake_palom.py`, a stand-in that records every call, keeps the package's
bookkeeping (its reader's levels and chunks, the affine's rescaling, the block matrices)
and draws the image block by block the way `block_affine.py` is written to. `test_dhr_adapter.py` runs the DeeperHistReg adapter
against `fake_dhr.py`, a stand-in that records every call, keeps the package's bookkeeping
(pads, resample ratio, field size, its swallowed exception) and draws the warped image the
way the package's writer and libvips are written to. `test_valis_adapters.py` runs both VALIS adapters against
`fake_valis.py`, a stand-in that records every call and draws images the way VALIS's
writer is written to; for `valis_mirage` it executes the real `bin/register.py` on that
stand-in, in a fresh process.
`test_ashlar_adapter.py` runs the ASHLAR adapter against `fake_ashlar.py`, the same kind of
stand-in (ASHLAR's reader, positions and paste rule transcribed from its source at
v1.20.0); the one test that needs ASHLAR itself is in `test_external_methods.py`.
`test_tissue.py` checks the tissue-level block on drawn masks and images with known
answers (identical, a known shift, a known rotation, concentric discs, a blank image, an
image off the reference grid, a bright fill outside the valid box), the reduction against
`cv2.INTER_AREA` and a brute-force integral, and the block's place in the scorer with fake
methods that write images; it needs OpenCV and skips without it.
`test_nuclei_sets.py` checks the two polygon sets on hand-made squares (a nucleus inside
each outline): the contract and its backward compatibility, the driver moving both, the
nuclei's own gate, the seam mask by inheritance and by geometry, the side-by-side lines
of `summary.md`, and that growing the outlines changes the nuclei in no way. `test_nuclei_cohort.py` checks the cohort's sources on small label images: a mask of
the slide is used, one of another size or another frame is refused with the reason, the
rendered `segment.py` call against `nextflow.config` and `conf/modules.config`, and the
re-run itself against a STAND-IN `segment.py` that writes the two masks under the real
script's names (StarDist is not installed here; the real script was never executed).
`test_compute_all.py`, `test_trace.py` and `test_params_json.py` check compute-all on generated Nextflow
traces (the tables, the pins against the schema, the plan of the runs, the `-params-file`),
`test_inputs_launcher.py`, `test_manifest.py`, `test_outline_definition.py` and
`test_scene.py` the input step, the manifest and the outline definition, and
`test_docs_are_cohort_only.py` that this README, `benchmarks/README.md` and `docs/benchmarks.md`
name nothing that was removed and only `submit_cohort.sh` flags that exist.
`test_launcher_e2e.py` runs `submit_cohort.sh` for real (not `--dry-run`) on the toy
cases against a stand-in SLURM (`slurm_sim.py`: `sbatch`, `squeue`, `sacct`, `sinfo`,
`scancel`; a job submitted from inside a job runs at once, the launcher's own are run in
submission order after it, which honours every dependency), with `fake_palom.py` as the
method (out of memory below a set GB) and the jobs on the host python
(`REGBENCH_ALLOW_PATH_PYTHON=1`); the selfcheck jobs are stubbed. Its scenarios: all ok
(settle and the scorer run, `regbench status` says all ok), an out-of-memory at 16 GB
that climbs to 64 GB, a unit out of memory up to the top rung (the method's `oom`) and
one whose next rung no node can grant (`not_submitted`, the launcher's), a watcher
killed mid-chain whose successor link finishes the chain, and a second submission into
the same folder that runs the method for nothing.
`test_selfcheck.py` runs STARE once for real, on a 1024 x 1280 pair;
`test_stare_adapter.py` runs STARE's selfcheck on its own two pairs and two real
registrations; `test_stages.py` registers the 2048 x 2560 pair stage by stage, as the
cohort's jobs do, and once more with the local launcher, and compares the two.
