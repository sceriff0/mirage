# Archived sections of benchmarks/regbench/README.md

Moved here verbatim on 2026-11-06 with the code they describe (`benchmarks/archive/README.md`).
Paths in them are the OLD live paths; the files now sit under `benchmarks/archive/`.

## From "The synthetic scaling set": its CPU cross

**Its CPU cross** runs `stare` and `valis120` at 4 and 8 CPUs on the 4096 and 32768 px
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
        --runs "stare valis120" -- <site options>
done
regbench_py STARE_REGBENCH cpu-scaling --dataset synthetic --rows stare,valis120 \
    --point 4=$ROOT/out_synth_c4 --point 8=$ROOT/out_synth_c8 \
    --to "$ROOT/out/paper/synthetic/cpu_scaling.csv"
```

## From "Scoring"

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
  runs. The other methods run their defaults only: `valis120`,
  `valis120_hires`, `deeperhistreg_gpu`, `deeperhistreg_hr_gpu`; every other run is listed
  in `excluded_runs.csv` with the ruling as its reason. The cohort keeps its 72 runs and
  the fluorescence sets theirs; `RUNS` holds 108 (the cohort's 72, `bf_same`'s 12 high-tier
  cells and its 24 medium and low cells not shared with the cohort). Since 2026-10-20 the
  cohort runs STARE's FULL cross too (user ruling): tier {high, medium, low} x stride
  {64, 128, 256} x level {none, clahe, percentile}, 27 cells plus `stare_local4`; the four
  `none` cells of medium and low at stride 128 and 256 are one run of both regimes. On a public set `runs --order default` puts
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

REGBENCH_CASES_HYRECO=/data/cases_hyreco benchmarks/regbench/slurm/submit_cohort.sh --datasets hyreco
```

With the full set, the script checks after the unzip that subset B landed in
`HyReCo-Additional/HE` and `HyReCo-Additional/PHH3` (the folders `prepare hyreco` reads),
each non-empty; if not, it fails and names the top-level folders the two Additional zips
really hold, so they can be moved there by hand.

**Preparing HyReCo is ONE sbatch job** (`slurm/prepare.sbatch`, in the scorer's image
`STARE_REGBENCH`): subset A and, when it is there, subset B, into `REGBENCH_CASES_HYRECO`
(the folder `submit_cohort.sh --datasets hyreco` reads; never `REGBENCH_CASES`, which is
the cohort's). The data root is read INSIDE the image, so it must be bound. The launcher
does both for you: `--prepare --datasets hyreco` needs `REGBENCH_HYRECO_DATA` (the
folder `hyreco.sh` unpacked into; refused before any sbatch when it is not set or has no
`HyReCo/HE` and `HyReCo/PHH3`), adds it to `REGBENCH_BIND` and submits the one job
`rb-prepare-hyreco` (`--export=ALL`; its size `REGBENCH_PREPARE_MEM`, 64G, and
`REGBENCH_PREPARE_TIME`, 24 h). Through the head job, from the folder the benchmark lives
in (`REGBENCH_ROOT` is `regbench/` under it; the head finds the images):

```bash
cd /beegfs/scratch/ieo7660
sbatch --export=ALL,REGBENCH_CASES_HYRECO=/beegfs/scratch/ieo7660/regbench/cases_hyreco,REGBENCH_HYRECO_DATA=/beegfs/scratch/ieo7660/hyreco \
    ~/pipelines/mirage/benchmarks/regbench/slurm/launch.sbatch --prepare --datasets hyreco
```

Or the same prepare job by hand, with nothing in between (name the image's path, the
cases folder and the data root yourself; `REGBENCH_BIND` binds the data root):

```bash
sbatch -c 8 --mem=32G -t 1-00:00:00 \
    --export=ALL,REGBENCH_REPO=$HOME/pipelines/mirage,REGBENCH_CASES_HYRECO=/beegfs/scratch/ieo7660/regbench/cases_hyreco,REGBENCH_BIND=/beegfs/scratch/ieo7660/hyreco,REGBENCH_PREPARE_IMAGE=STARE_REGBENCH,REGBENCH_SIF_STARE_REGBENCH=<the stare-regbench .sif> \
    ~/pipelines/mirage/benchmarks/regbench/slurm/prepare.sbatch hyreco --data-root /beegfs/scratch/ieo7660/hyreco
```

Either way the job writes `prepared-hyreco.ok` beside the cases when the whole prepare
ended ok. Then launch the set by name (`--datasets hyreco`, with the same
`REGBENCH_CASES_HYRECO`).

The download is resumable (`curl -C -` into `<name>.part`; an md5 given on the line must
match before the file takes its name), and a failure stops the script, never swallowed.
The committed fixture (`benchmarks/tests/regbench/fixtures/hyreco/`) holds made-up images
and landmarks in the dataset's layout; its `README.md` cites where every format detail
comes from.

## From "Running the cohort on a cluster"

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
