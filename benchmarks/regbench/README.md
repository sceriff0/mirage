# regbench: registration methods, each run natively, judged on the same nuclei

`benchmarks/regbench/` compares registration methods on nuclear overlap and on
registration-only cost. It was moved here from `sceriff0/stare` (branch `benchmarking`,
commit `f5ce62e`, archived there as the tag `archive/benchmarking-2026-10-08`).

## The rule

**Each method is run through its own entry point, with its own defaults, reader and writer,
and is judged with the transform it estimated, moved by its own code in its own exact
direction.** The benchmark does not re-implement, resample, invert or approximate that
transform. Where a method's writer departs from its own transform, both are scored and the
difference is reported.

What the benchmark does around a method is listed per run in
`run.json: info.implementation.benchmark_side`.

## State of this port

This package is being ported in steps. What exists now:

| part | state |
|---|---|
| contract (`mapping.PointMap`, per-pair and per-patient methods, `run.json`) | here |
| scoring (`pairs.py`, `score.py`) and cost (`resources.py`) | here |
| datasets `synthetic`, `semisynth`, `multiplex` | here. The frozen known-truth cases (`--suite truth`, `--truth`) are real patients with two rounds; in the copied suites each case is still its own patient |
| the cohort's cases (`arms_manifest.py`, `datasets/multiplex.py`) | here: the arms' original multichannel slides and the QC outlines of one arm (StarDist nuclei on the nuclear channel, grown by 10 px: see "What the cohort's nuclei are"). Built and tested against small trees laid out like the arms' results; never run on the cohort itself |
| known-truth error per deformation family (`truth.py`, `truth_spec.json`) | here; the frozen 8192 px cases have not been run through any method |
| adapter `stare` | here, as in "Methods" below: the pipeline's `high` tier, the original slides, its own STITCH deferred until the clock has stopped, the option `stride` |
| adapters `valis_mirage` (VALIS 1.0.0 through the pipeline's function) and `valis` (1.2.0 as shipped: `valis120`, `valis120_hires`) | here, as in "VALIS" below. **Written against VALIS's source and never run with VALIS** (it is not installed where this was written): tested with a recording stand-in, and for `valis_mirage` with the real `bin/register.py` on that stand-in. Their first real run is the selfcheck inside each VALIS image |
| adapter `initial` | here (the identity) |
| adapter `deeperhistreg` (DeeperHistReg 1.0.1, the software that ships RegWSI: eight labels, each with an estimated-field row and an `_asdrawn` row) | here, as in "DeeperHistReg" below. **Written against DeeperHistReg's and libvips' source and never run with either** (DeeperHistReg, SimpleITK and pyvips are not installed where this was written): tested with a recording stand-in that draws the warped image the way the package's writer is written to. Its first real runs are the settling measurement and the selfcheck inside the DeeperHistReg image |
| selfcheck and realcheck (`selfcheck.py`) | here. Really run for `stare` only (its package is the one installed where this was written); for every other method the check has only been exercised with fake methods |
| adapter `ashlar` (ASHLAR 1.20.0, its own command per patient: `ashlar`, `ashlar_clean`, `affine_ashlar`, `affine_ashlar_clean`) | here, as in "ASHLAR" below. **Written against ASHLAR's source and never run with ASHLAR** (it is not installed where this was written): tested with a recording stand-in that draws the mosaic the way ASHLAR's writer is written to. Its first real run is the selfcheck inside `labsyspharm/ashlar:1.20.0` |
| adapter `palom` (PALOM 2026.4.1 by its README recipe: `palom`, and `palom_b1024` with the reference rechunked to 1024 px blocks) | here, as in "PALOM" below. **Written against PALOM's source and never run with PALOM** (it is not installed where this was written): tested with a recording stand-in that draws the registered image the way PALOM's writer is written to. Its first real runs are the single-level read test, the pixel-for-pixel writer test and the selfcheck inside a PALOM image |
| off-seam rows (`<variant>:non_<owner>`, `nonband_*.csv`) | here for ASHLAR's moving-side mask and PALOM's reference-side mask (nuclei through `band_ref`, known-truth probes through `band_truth`) |
| tissue-level agreement of the written images (`tissue.py`: ACCREDIT's score and its pieces, secondary) | here, as in "Tissue-level agreement of the written images" below. Run on drawn images and on fake methods' written images only; never on a real method's output or on the cohort |
| the run list (`variants.py`, `python -m benchmarks.regbench runs`) | here, as in "The run list" below: 19 runs, 37 result names, six primary rows, a flag that pauses a run (nothing is paused) |
| the cohort launcher (`slurm/submit_cohort.sh`, `ladder.py`) | here, as in "Running the cohort on a cluster" below: every run behind its own gate, eight chained job arrays per run and dataset (four rungs, each with one retry for an attempt the scheduler lost), `multiplex` before `synthetic`, the scorer last. **Never submitted**: tested through its dry run and through a stand-in `sbatch` that records its calls |
| the cost-by-step view (`cost.py`, `cost_end_to_end.csv`) | here, as in "Cost" below. Every number in it is measured by this benchmark; nothing is read from a pipeline run (the import of a Nextflow trace was removed on 2026-10-12) |
| HyReCo, the external landmark set (`datasets/hyreco.py`, `landmarks.py`: `hyreco_a` to tune, `hyreco_b` to report) | here, as in "HyReCo" below. **Written against the dataset page's layout and never run on the files**: they are not on this machine (740 GB behind a login). Tested on stand-in files the tests write in that layout; STARE really registers one such pair. `stare`, `valis120` and `valis120_hires` have an input rule for it; the other methods refuse it, each with its reason |

Nothing here has run on the cluster.

## Layout

```
mapping.py     PointMap: a point mapping and the direction it was estimated in
cases.py       the on-disk contract: case.json + points.npz in, warped.npz + run.json out
cells.py       nuclei of a native slide: the built-in segmenter, label images, cell GeoJSON
arms_manifest.py  the arms cohort as a `prepare multiplex` manifest, one row per moving round
nuclei.py      bare nuclei per native slide: check the label images that exist, re-run the
               pipeline's StarDist where there is none ("Bare nuclei")
pairs.py       the scores of one (case, variant), on bin/utils/cell_pairs.py
resources.py   the allocation, the process-tree monitor, out-of-memory detection
methods/       one module per method, and the driver that runs any of them
datasets/      synthetic, semisynth, multiplex: each writes cases; hyreco and anhir write
               the external landmark sets (hyreco_a, hyreco_b; anhir, anhir_tune)
prep.py        the stare_prep row: the preprocessing candidates, the frozen choice
               (prep_frozen.json), the check that none of them moves image content
score.py       reads files only; writes <out>/tables, and keeps what it computed for each
               case under <out>/cache/scores ("Records: a case is scored once")
landmarks.py   the scorer of HyReCo: <out>/tables/<dataset>/, never the cohort's tables
challenge.py   the scorer of ANHIR's training pairs (rTRE, robustness, micrometres), its
               own tables too; and the choice of the stare_prep preprocessing
selfcheck.py   do a method's points follow the image its own writer produced? (a gate)
tissue.py      the written images against the reference at 4000 px (ACCREDIT's score; secondary)
ladder.py      the memory ladder's rule: a unit goes up a rung only when it ran out of memory
cost.py        cost by step, from the cohort's files to each method's own image (secondary)
slurm/         the cohort launcher (submit_cohort.sh) and what its jobs run
```

Three seams: datasets write cases, method adapters run the method's own software and return
a point mapping in the method's own direction, the scorer reads files only.

## Running it

From the repository root:

```bash
python -m benchmarks.regbench prepare synthetic --suite test --cases CASES
python -m benchmarks.regbench list --cases CASES --dataset synthetic          # count
python -m benchmarks.regbench list --cases CASES --dataset synthetic --groups  # patients
python -m benchmarks.regbench run --cases CASES --out OUT --dataset synthetic \
    --method stare --index 0
python -m benchmarks.regbench tissue --cases CASES --out OUT   # needs OpenCV; see below
python -m benchmarks.regbench score --cases CASES --out OUT
python -m benchmarks.regbench selfcheck --method stare --work WORK
```

The cohort (the arms' results folder, `ROOT`):

```bash
python -m benchmarks.regbench.arms_manifest --root ROOT --out rounds.csv
python -m benchmarks.regbench prepare multiplex --manifest rounds.csv --cases CASES
```

The first command reads `ROOT/preprocess_shared/csv/preprocessed.csv` and the QC outlines
`ROOT/valis_high_micro2/<patient>/qc/registration/geojson/<slide>.geojson` (`--from-arm`
names another arm), and writes one row per moving round: `case_id` = `<patient>_<slide>`,
`group` = the patient. A patient without exactly one reference, a slide that is not on
disk, a slide whose channel list does not name exactly one nuclear channel (a
case-insensitive substring match of `DAPI` or `CELLTOX`; `--nuclear` narrows it) and a
round whose nuclei are missing are dropped, each printed with its reason and path; the exit
code is non-zero only when no row is left. The pixel size is the checkpoint's `pixel_size`
column, 0.325 µm where a row has none (`pixel_size_source` says which).

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
19 runs, 37 result names, 4 runs on a GPU, 7 per patient. `score` expects exactly those
names and opens its summary on the six primary rows:

`stare`, `valis_mirage`, `valis120_hires`, `deeperhistreg_hr_gpu_asdrawn`, `ashlar`, `palom`.

- **Two methods run at two settings, and the high-resolution one is primary** (the
  user's ruling of 2026-10-11, fixed before any cohort run). VALIS 1.2.0: `valis120` is
  upstream's default pipeline, `valis120_hires` upstream's own high-resolution example.
  DeeperHistReg: the `deeperhistreg_{gpu,cpu}` rows are the package's default parameter
  set (4096 px), the `_hr` rows its shipped 8192 px set. Both settings of both methods
  are run, scored and shown; the default-setting rows are in the same family and are
  never hidden. `variants.PRIMARY_VALIS120` and `variants.PRIMARY_DEEPERHISTREG` are the
  one line that names each.
- **STARE has two rows, and one STARE.** `stare` is the method as the pipeline runs it.
  `stare_prep` (ruling R3.21) is the SAME package, version and parameters on a
  benchmark-side preprocessing of the one image STARE registers on (`prep.py`: a small
  closed candidate list; the candidate in force is frozen in `prep_frozen.json` and is
  the run's one option, `prep=<candidate>`). It is an active, NON-primary row of STARE's
  family on `multiplex`, on `synthetic` and on ANHIR. The candidate is chosen on ANHIR's
  tuning-side pairs and on nothing else, never on the cohort; on the cohort it is applied
  to the nuclear channel STARE reads, where each candidate has a stated meaning (two of
  the five are the identity there). The image it writes is the ORIGINAL moving slide
  moved by the transform estimated on the preprocessed copy, so every score is defined
  as for `stare`. **Today the frozen candidate is PROVISIONAL (`grey_clahe`, chosen by no
  evidence): the selection has not been run.** The launcher and every summary say so;
  see "ANHIR: the challenge's training pairs".
- **A run can be paused** (`active=False` on its entry). A paused run stays implemented
  and tested and its names still resolve, but it is not launched, not expected by
  `score`, not counted and not printed by `runs`; `runs --all` prints it with `0` in the
  `active` column. Nothing is paused. A list in which a paused run writes a primary row,
  gates an active run's selfcheck or is what an active run waits for is refused
  (`variants.check`, at import).
- **Images** (the `image` column is the suffix of `REGBENCH_SIF_<image>`, the variable
  that holds an image's path on the cluster). Built here, the suffix being the directory
  under `containers/` in capitals with `-` written `_`: `STARE_REGBENCH` (the `stare`
  run, and `tissue` and `score`: `variants.SCORER_IMAGE`), `VALIS120` (built FROM
  upstream's `cdgatenbee/valis-wsi:1.2.0`; needs `--writable-tmpfs`), `DEEPERHISTREG`,
  `PALOM`. Pulled from upstream: `VALIS_MIRAGE` (`cdgatenbee/valis-wsi` 1.0.0, the
  pipeline's digest) and `ASHLAR` (`labsyspharm/ashlar:1.20.0`). `containers/README.md`
  has what each holds. Inside an image the harness is started with the image's own
  `python`, or with `python3` where it has no `python`: the choice is made inside the
  image (`slurm/env.sh: regbench_py`), never on the host. ASHLAR's image is the case
  that needs it: Ubuntu 22.04 with `python3-dev` and `python3-pip` and nothing called
  `python` (its Dockerfile at tag v1.20.0, `[FULL]`), and its first selfcheck on the
  cluster ended with `"python": executable file not found in $PATH`. That Dockerfile
  also sets `ENV OMP_NUM_THREADS 1`: the harness sets no thread variable, but this one
  comes with the image ("Cost", "Threads the method's own image sets").
- **Which rows gate a run** (`gate_variants`, the last column). A failed selfcheck blocks
  the method. For every method but one that means any of the run's rows. DeeperHistReg's
  gate reads its `_asdrawn` row alone, the row of the image its writer puts on disk: when
  only the estimated-field row fails, that row is reported, left out of the results
  (`variants.dropped_variants`) and the method still runs. `variants.gate_status(name,
  selfcheck_record)` is the status that decides. From a shell:

  ```bash
  python -m benchmarks.regbench gate --name deeperhistreg_hr_cpu --selfcheck W/deeperhistreg_hr_cpu/selfcheck.json
  # prints "<status>\t<rows to drop, comma-joined, or ->"; exit 0 ok, 1 inconsistent,
  # 2 not_checked, 3 crashed (the status of the gating rows, not of the whole record)
  python -m benchmarks.regbench score --cases CASES --out OUT --drop deeperhistreg_hr_cpu
  ```

  A dropped row is in no table; `dropped.csv` and the summary's "Rows left out" name it,
  so a row is never missing without a reason. `--drop` refuses a name that is no result
  name of the run list, and a row that gates its run.
- **Which runs have a realcheck as a second gate** (`realcheck`, column 15): the primary
  rows of the two VALIS adapters, `valis_mirage` and `valis120_hires`
  (`variants.realcheck_gate`, `variants.REALCHECK_METHODS`). Their point path is derived
  by the benchmark from VALIS's matrices and fields, so it is checked once more on a real
  case before the cohort is spent on it: the launcher sends the run's first patient
  ahead with `run --realcheck`, and `python -m benchmarks.regbench realgate` reads that
  row's own `realcheck.json` (not the exit code of `run --realcheck`, which is also 3
  when a non-primary row is flagged). Only a measured flag blocks; a first patient that
  failed, or wrote no image, is `realcheck_not_checked` and the run goes on. `valis120`
  has no primary row and no such gate.

## The contract

**A case** is `<cases>/<dataset>/<case_id>/case.json` plus `points.npz`. `case.json` names the
two slides, the pixel size, `group` (the PATIENT, in every dataset) and, on known-truth
cases, `family` and `pose`. `points.npz` holds the nuclei segmented on each NATIVE slide as
full polygon sets (`cells_<side>_xy`, `_ring_off`, `_ring_feat`, `_ring_hole`, `_n`; `side`
is `ref` or `mov`), read back with `cases.polys(points, side)`. A known-truth case also
holds paired probes: `truth_mov`, `truth_ref` and `truth_cell` (see "Known-truth error").

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
nucleus; `band_truth` when `info["band_truth"]` is: the same seams by known-truth probe,
for a reference-side mask; `truth_xy` = the map
applied to the truth probes of the side it starts from, and `grid`, `grid_shape`,
`grid_step` = the map on a lattice for the Jacobian statistics; `band_nuc` /
`band_ref_nuc` = the owner's seam mask on the bare nuclei, see "Off the seams"), writes
`run.json` (`nuclei_band` says how that mask was got), and calls `on_variant`. The second
set costs the method nothing: it is moved with the clock paused, like the first.

**`run.json`**: `ok`, `error`, `cause`, `problem`, `direction`, `launch_id`, `unit`
(`pair` | `patient`), `group_cases`, `register_s`, `write_s`, `resources`, `native`,
`selfcheck_status`, `attempt`, `mem_requested_gb`, `cpus_requested`, `cpus_affinity`,
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
| `benchmark_oom` | the BENCHMARK's own lines ran out of memory with the clock stopped (the lattice, the arrays and the files around the method's point mapping; never the mapping itself) | yes | NOT the method's |
| `benchmark` | the benchmark's own lines around the method's point mapping failed for another reason | no | NOT the method's |
| `not_submitted` | a stage job of a registration in several jobs never reached the queue, or was left without the attempt the pipeline would have given it next (the tile submitter was refused or did not finish) | no | NOT the method's (the launcher's) |
| `infrastructure` | the scheduler lost the attempt (`NODE_FAIL`, `PREEMPTED`, `BOOT_FAIL`, `CANCELLED`, `REQUEUED`, `REVOKED`), twice at one rung, or the task never started | no: the same rung once more, then it stands | NOT the method's |
| `not_run` | the scorer's word for a result with no record, or with a record of another submission | | neither |

`failures.csv` and the summary count the causes that are not the method's apart
(`launches_failed_method`, `launches_lost_not_method`, `rounds_lost_not_method`). A round
lost for a cause that is not the method's (`resources.NOT_THE_METHOD`) is LEFT OUT of that
method's summaries, differences and known-truth rows, and is in no common block; its row
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
exactly those of this document. With it, `runs`, `run`, `selfcheck`, `gate`, `score` and
the launcher pick the plug-in's runs up through that one hook: an image is named like any
other (`REGBENCH_SIF_<image>`), and the launcher's jobs inherit the environment. A
plug-in's rows are never primary. Nothing of a plug-in is built, pushed or tested by CI.

## Methods (the design; see "State of this port" for what exists)

| variant(s) | software | run unit | native transform | how nuclei are moved |
|---|---|---|---|---|
| `stare_rigid`, `stare` | pinned `stare-registration`, mirage's `high` tier, stride 64 | pair | affine + mesh, moving -> reference | moving nuclei through `stage_warp.make_warper`; reference frame |
| `valis_mirage_rigid`, `valis_mirage_nonrigid`, `valis_mirage` | VALIS 1.0.0 through the pipeline's own call | patient | affine + backward field `bk_dxdy` | reference nuclei (pixels of the written image's grid) through VALIS's `warp_tools.warp_xy_inv`; moving frame |
| `valis120_rigid`, `valis120`, `valis120_micro` | VALIS 1.2.0, **upstream's default setting** (`register()` then `register_micro()` with no argument). Run and shown; not primary | patient | same | same |
| `valis120_hires_rigid`, `valis120_hires_nonrigid`, `valis120_hires` | VALIS 1.2.0, upstream's high-resolution example (`--opt hires=1`). **`valis120_hires` is VALIS 1.2.0's primary row**, by the user's ruling | patient | same | same |
| `deeperhistreg_{gpu,cpu}`, `deeperhistreg_hr_{gpu,cpu}`, and the same four with `_fluo` | DeeperHistReg 1.0.1 (the method is RegWSI), a shipped parameter set with only the device changed; `_hr` is the 8192 px set, `_fluo` adds two benchmark-set keys | pair | backward field on the padded, resampled reference grid | reference nuclei through its `warp_landmarks` and the coordinate chain of its issue #12; moving frame. Secondary |
| the same eight with `_asdrawn` | the same runs | pair | the image its own writer puts on disk | reference-grid pixels of the written image through the writer's own arithmetic; moving frame. The rows that can be primary: **`deeperhistreg_hr_gpu_asdrawn` is DeeperHistReg's primary row**, by the user's ruling (the 4096 px set is the package's default). **It follows the writer**, which by the adapter's reading sits about half a field pixel from the registration (2.5 px = 0.81 µm per axis at this 8192 px set, 1.15 µm along the diagonal; 4.9 px = 1.59 µm per axis at the 4096 px set), so its `within_1um` can be low with a perfect registration; `deeperhistreg_hr_gpu`, the estimated-field row, shows the registration itself |
| `ashlar`, `ashlar_clean`, `affine_ashlar`, `affine_ashlar_clean` | ASHLAR 1.20.0, its own command, on tiles cut from stitched slides | patient | one shift per tile | moving nuclei by their tiles' shifts; reference frame |
| `palom_affine`, `palom` (**primary**), `palom_b1024_affine`, `palom_b1024` | PALOM 2026.4.1, its documented recipe; blocks are the reference file's TIFF tiles (2048 px on the cohort) for `palom`, re-cut to 1024 px for `palom_b1024` | pair | an affine, then one shift per block of the reference | reference nuclei through the affine and their block's shift; moving frame |
| `initial` | none | | identity | |

Honest limits of "natively": ASHLAR is run on tiles cut from stitched slides, not raw tiles;
DeeperHistReg is run on fluorescence with brightfield defaults; STARE is run as the
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

### selfcheck (a gate)

```bash
python -m benchmarks.regbench selfcheck --method M [--label L] [--opt k=v ...] --work W \
    [--mask overlap|seams] [--m0-from DIR] [--size HxW[:HxW[:HxW]] ...]
```

Exit 0 `ok`, 1 `inconsistent`, 2 `not_checked`, 3 `crashed`. **Anything but 0 blocks the
method** on the cohort. It writes `W/selfcheck.json`: `cases` (each case's folder, group
and number of rounds) and `pairs`, every variant's numbers per ROUND and per 512 px block
(one entry for a pair, one per moving round for a patient, each with `case_id`, `group`,
`round`, `rounds`).

**Every round carries every row of the run** (`selfcheck.every_row`). A round whose
process died, or that has no entry for a row, gets that row with the round's own status
when it blocks and `not_checked` otherwise (`"filled": true`), and a round's status is
never better than its rows'. The gate (`regbench gate`, `variants.gate_status`) reads
its gating rows twice, from the record's summary and from every round of every case, a
missing row counting as `not_checked`: its verdict is never better than what any case
says. A case that died can therefore not pass as checked through the rows of another
case. `REGBENCH_ALLOW_UNCHECKED` lets a run start on `not_checked` from a selfcheck that
ran, and on nothing else: `inconsistent`, `crashed`, a flagged realcheck and a missing
record (or another submission's) block it all the same.

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
moving round is checked against the gate on its own**: its points against its image.

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
reference file's TIFF tiles, 2048 px on the cohort. Both rows now read files with 2048
px tiles (`methods/palom.py::SELFCHECK_PAIR_TILE`): `palom` on 16384 x 20480 (and
against a 15566 x 19456 moving slide), the smallest canvas with 8 x 10 blocks of 2048
px, so the tiling has 7 x 9 interior block corners and the seam rule is exercised at
that block size (cap 0.05 there, below); `palom_b1024` on 8192 x 10240 as before, the
same tiles cut to 1024 px blocks. Before, `palom` was checked on 1024 px tiles and its
cohort blocks were first met on the cluster.

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
whole-image number inside the gate. Detections under 1.2 px are counted in the record
(`n_narrow`), so a writer that shrinks dots is visible (per checked result in
`selfcheck_masks.csv`, and by name in the block "What the gates decided" of
`summary.md`); each of them is held to both
checks like any other, so a narrow fragment a writer leaves where the map puts no dot
fails. Tests: `test_selfcheck.py::test_a_dot_the_writer_shrank_is_still_a_dot`,
`::test_a_shrunk_dot_in_the_wrong_place_is_still_inconsistent`,
`::test_what_is_narrower_than_half_a_dot_is_not_a_dot`.

**The gate (frozen).** 95th percentile of the residual at most 0.40 px, mean signed
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
- **Its cap is a flag, not a gate** (`selfcheck.MASK_MAX`, frozen: `overlap` 0.50,
  `seams` 0.10 of the dots). A row whose mask hid more than the cap KEEPS its verdict:
  the run starts on every dataset and is scored. It is flagged instead
  (`mask_over_cap` in the record; column `selfcheck_mask_over_cap` on every row of
  every table: `no`, or `yes: 0.57 masked, cap 0.50`, with `(via <run>)` for a run
  gated by another run's selfcheck), and `summary.md` names it. What still blocks is
  unchanged: the accuracy thresholds on the unmasked dots (95th percentile at most
  0.40 px, mean offset at most 0.15 px per axis, at least 99 % found, per block too),
  the coverage of the drawn dots and the image's shape. Until this ruling a share
  above the cap read `not_checked`, which blocked the run.
- **The verdict under a fixed list of caps** (`selfcheck.MASK_CAPS`): `overlap`: none,
  0.70, 0.50, 0.45; `seams`: none, 0.10, 0.05. One column per cap (`caps` in the record,
  `passes_cap_<cap>` in the table): would the row pass the accuracy gate on its
  unmasked dots if the mask could hide at most that share. `passes_cap_none` is the
  verdict in force.
  - **`overlap` 0.50**, beside shares measured on ASHLAR's only selfcheck canvas (8192
    x 10240: 5 x 6 tiles of 1950 px at a 1755 px pitch) with the adapter's own
    geometry: its tile grid, `tilemap.on_seam` at its 8 px dot margin, on the
    selfcheck's own dots, both rounds of the patient. No image enters, so ASHLAR is not
    needed
    (`test_selfcheck_gate.py::test_the_overlap_cap_stands_above_the_measured_share`
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
    cap stands 0.22 above the worst case within ASHLAR's own limit; it is a flag, never
    a gate, and at least half of the dots are always judged. What ASHLAR really does on
    this canvas is not known until it runs: a share above 0.50 is flagged on every row
    of the run, with the share, and the numbers on all dots and on the masked dots
    alone are in the record either way.
  - **`seams` 0.10** is 3.2 times the share the seams of 1024 px blocks hold when every
    block edge is one (8 px either side: 1 - (1 - 16/1024)^2 = 0.03). PALOM's adapter
    applies the same factor to the blocks a pair really has (`SELFCHECK_MASK_MAX`):
    0.10 for `palom_b1024`, 0.05 for `palom` on its 2048 px blocks.
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
selfcheck_status` is the run's gate, which `stare_rigid` shares with `stare`. The scorer
therefore reads each ROW's own status from the selfcheck records
(`variants.row_checks`) into a column of its own, `selfcheck_row`, in `cells.csv`,
`truth.csv`, `summary_cells.csv`, `truth_by_family.csv` and the off-seam tables, and
`summary.md` prints it beside the row's name: `ok`, `no_stage_image` (shown as **NEVER
CHECKED**), `via <run>: <status>` for a row whose run is gated by another run's selfcheck
(`ashlar_clean`), or `no selfcheck record`.

| status | meaning | blocks the method |
|---|---|---|
| `ok` | the gate holds | no |
| `inconsistent` | the points do not follow the image | yes |
| `crashed` | the method raised | yes |
| `not_checked` | nothing could be measured: the dots were not detected, the method wrote no image, or the run ran out of memory or time (`cause` says which). Never reported as `inconsistent` | yes |
| `no_stage_image` | a stage the method's own writer cannot draw (`stare_rigid`: STITCH writes one image). Reported, never counted as checked | no |

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
| + 0.10 px on x | `ok` (inside the gate) | 537 / 537 | 0.175 | +0.103 | 0 / 20 |
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
(`test_selfcheck_gate.py`, a 1024 x 1536 image, the points right wherever something is
drawn):

| image | `mov_to_ref` | `ref_to_mov`, before | `ref_to_mov`, now |
|---|---|---|---|
| complete | `ok` | `ok` | `ok` |
| 7 % of it undrawn | `inconsistent` | `ok` | `inconsistent` (drawn dots the image does not show; uncovered blocks) |
| 20 % undrawn | `inconsistent` | `ok` | `inconsistent` |
| 45 % undrawn | `inconsistent` | `ok` | `inconsistent` |
| a mask hiding 93 % of the dots, nothing else wrong | `ok`, flagged `mask_over_cap` (R3.22; `not_checked` between 2026-10-11 and that ruling) | `ok` | `ok`, flagged |
| 12 stray detections outside a mask that hides a quarter | not looked at before, `inconsistent` now | | |

So the smallest constant offset the gate objects to is 0.15 px per axis, the detector's
own floor being 0.09 px at the 95th percentile. The row "1 px wrong on 4 % of dots" is
caught by one block alone, with every whole-image number inside the gate.

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
a gate (it is the only check at cohort field scales): `slurm/submit_cohort.sh` runs the
first unit of such a run ahead with `--realcheck`, and a gate job (`regbench realgate`)
blocks the rest when that realcheck is flagged ("Running the cohort on a cluster", step
4). It has only run on synthetic slides, and the gate has never run on a scheduler.

## Scoring

Nuclei are segmented once on each native slide, before registration, so they do not depend
on any method's output. Pairing and IoU are `bin/utils/cell_pairs.py`, imported
(`pairs.py` loads that one file; `bin/utils` is not put on `sys.path`).

- **Two polygon sets, every number on both.** The **outlines** are the headline: on the
  cohort, each nucleus grown by 10 px, which is what the pipeline's QC publishes and what
  the arms benchmark scores; every table and column that existed means the outlines and
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
  ones. A case with no pair has no Dice; it cannot enter a median and is counted
  (`<headline>_n_nan`, "dice NaN cases" in `summary.md`) instead of vanishing.
- **No pixel size:** µm columns and the curve are NaN, not zero.

A failed, killed or never-started run is a row scored at the unregistered pose
(`imputed_initial`), with `problem` and `cause` filled; never a gap.

**A round lost for a cause that is not the method's is not imputed** (`infrastructure`: the
scheduler lost the attempt; `benchmark_oom` and `benchmark`: the benchmark's own lines
ran out of memory or failed; `not_submitted`: the launcher never queued a stage job).
Scoring it at the unregistered pose would charge the method for the cluster, and dropping
it silently would hide it, so: it is left out of that method's summaries, differences and
known-truth rows (`n_excluded_not_method`, "left out" in `summary.md`), it is in no common
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
| `synthetic` | a generated scene: ONE per deformation family (its two poses are the same scene) | every family has one unit: always descriptive, never ranked |
| `semisynth` (dormant) | the real patient the window was cut from | its windows, families and poses are one unit, however many cases they make |

On the two known-truth datasets the cell summaries are **per deformation family** and never
pooled over families, as the known-truth table already was: a family outside a method's
model (and ASHLAR's `rot` rows) shows in its own row.

**The common block** (`subset=common`, `common_block.csv`) is the cases every COMPARED row
registered. A row is compared when it is a primary row (any registration row when no
primary list is given) that ran in what is being scored: no gate blocked its run (the
launcher's gate records, `score --gates`) and at least one of its cases has a record of
this submission. A primary run that was blocked, or never launched (`submit_cohort.sh
--runs "stare ashlar"`), is named with its reason and is not allowed to empty the block
for everyone, which it did before. The block is always printed, even when it is empty.

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
the known-truth table has `tre_um_p50_mean` beside `tre_um_p50_med` with `n_ok` and
`n_cases`, and the cost medians have their means (`*_mean`, `*_common_mean`) and the
patients completed.

It is self-consistency on the registration channel: there is no held-out structure and no
landmarks on real slides.

### What the cohort's nuclei are

Read from the repository on 2026-10-11 (review section D; nothing was run, and the site
overlay `conf/ieo.config` is not in the repository, so a setting it overrides would not
show here):

| step | what | where |
|---|---|---|
| the file | `<arm>/<patient>/qc/registration/geojson/<slide>.geojson`, arm `valis_high_micro2` | `arms_manifest.py:44, 185-186` |
| the arm's QC | `reg_qc: 2`, `seg_method: stardist` (the site's StarDist model, not the pipeline's default InstanSeg) | `benchmarks/configs/arms.yaml:42-51`; the arm varies only the micro-registration |
| what is segmented | each NATIVE slide, before registration, by the run's own segmenter | `subworkflows/local/seg_qc.nf:8-20, 136` (`SEG_QC_SEGMENT` is `SEGMENT` under an alias) |
| on which channel | ONE channel, index 0, which must be the nuclear marker (the run refuses otherwise) | `lib/SegBackends.groovy:93-117` (`segment.py --dapi-channel 0`) |
| which mask is traced | the **cell** mask, not the nuclei mask | `subworkflows/local/seg_qc.nf:140` (`SEG_QC_GEOJSON(SEG_QC_SEGMENT.out.cell_mask)`); `modules/local/segment.nf:51-52` emits both |
| what the cell mask is | the StarDist nuclei labels grown by `seg_expand_distance` px, no label growing into another | `bin/segment.py:162, 223-225` (`expand_labels_tiled`); `conf/modules.config:1046`; default 10 px (`nextflow.config:275`), not set in `arms.yaml` or `benchmark.config` |

So the outlines are **nuclei found on the nuclear channel alone, each grown by 10 px (3.25
µm at 0.325 µm per pixel) until it meets a neighbour**. They are not whole cells found on
membrane or marker channels: the review's worry that outlines differ between rounds
because the markers differ does not apply to this arm (it would at the pipeline's default
segmenter, InstanSeg on every channel). They are also not bare nuclei, and three things
follow for every method alike:

- the gate (1.5 x the median equivalent radius) and the null's shift (4 radii) are sized
  from the GROWN outline, so both are larger than a nuclear radius would make them;
- matched Dice is the overlap of grown outlines, whose shape in dense tissue is set by the
  neighbours as much as by the nucleus; it has a ceiling below 1 that is not
  registration;
- centroids are those of the grown outline: close to the nuclear centroid for an
  isolated nucleus, pulled away from it where neighbours block the growth on one side.
  `within_1um` inherits that.

A nuclear mask is NOT among the arm's published files: `SEG_QC_SEGMENT` publishes nothing
(`conf/modules.config:1140-1149`), and only the GeoJSON of the cell mask is published. The
bare nuclei are therefore found or made by the benchmark, slide by slide: next section.

### Outlines and bare nuclei

**The two definitions.**

| set | on the cohort | on `synthetic` (and dormant `semisynth`) |
|---|---|---|
| `outlines` (headline) | the arm's published QC outline of each native slide: a StarDist nucleus grown by 10 px until it meets a neighbour | the built-in segmenter's objects grown the same way: `skimage.segmentation.expand_labels(labels, distance=10)`, the call `bin/utils/segment_io.py::expand_labels_tiled` makes (`cells.grow`, `cells.EXPAND_PX`) |
| `nuclei` | the StarDist nuclei themselves, from the first source below that applies | the built-in segmenter's objects (what these datasets scored before 2026-10-11) |

**Why both.** Matched Dice on grown outlines is dominated by the rim: at a nuclear radius
of about 12 px a 10 px rim is roughly two thirds of the outline's area, so a given
displacement costs a grown outline a smaller share of its area than it costs the nucleus,
and in dense tissue the outline's shape is clipped by its neighbours and not by the
nucleus at all. Dice on the bare nuclei is stricter per micrometre and has no clipping.
The outlines stay the headline because they are what the pipeline's own QC and the arms
benchmark report; the nuclei say how much of an outline number is rim. Both sets are
segmented on the NATIVE slides before any registration, so neither depends on a method.

**Where the nuclei of a cohort slide come from**, in this order (user's ruling: reuse what
exists, segment only what has none):

1. **An existing label image of that slide.** `arms_manifest --nuclei-glob PATTERN` finds
   masks of the QC segmentation kept in a Nextflow work directory (`{patient}`, `{slide}`,
   `{stem}`; `segment.py` names its mask `<image stem>_nuclei_mask.tif`, e.g.
   `P1.r2.ome_nuclei_mask.tif`); `--seg-arm ARM` offers, for each patient's REFERENCE
   slide only, the `nuclei_mask` a segmentation arm lists in `csv/segmented.csv` (made on
   the registered image, whose reference round is likely the native reference slide
   unchanged: not verified, hence the check). A manifest of your own names them in
   `ref_nuclei` / `mov_nuclei`. A label image is **used only if** its shape is the
   slide's (refused with both sizes otherwise) **and** at least **0.90** of the slide's
   published outlines hold exactly one of its nuclei's centroids
   (`nuclei.ACCEPT_ONE_NUCLEUS`; the measured share is recorded) **and** its objects
   are the bare nuclei, not the cell mask grown from them and not nuclei in a shifted
   frame (`nuclei.identity`; the grown cell mask, and a nuclear mask 3 px off, both put
   one centroid in every outline). Over the objects that sit alone in an outline, with
   fixed thresholds: the median area ratio object / outline at most **0.90** (a
   nucleus of 12 px radius in an outline grown by 10 px gives 0.30; the cell mask
   gives 1.00 and is refused as "looks like the cell mask", with the number); at least
   **0.90** of the objects' vertices inside their own outline; and the median offset
   of an object's centroid from its outline's at most **1.0 px** on each axis. The
   three numbers are in the slide's record (`agreement.identity`), and `exact` needs
   them too. The second check is
   applied to every label image, not only to a segmentation arm's: two rounds of a patient
   have the same shape, and a mask of the wrong round would pass the first. Why 0.90: an
   outline is its nucleus grown outwards, so the true nuclei put one centroid in nearly
   every outline, while a mask of another slide or another frame puts centroids at random
   (in the test: under 0.1). It is `exact` when that share is at least 0.99 and the two
   counts agree within 1 % (`EXACT_ONE_NUCLEUS`, `EXACT_COUNT_TOL`: room for concave
   nuclei, the 1 px simplification of the published polygons and a GPU's run-to-run
   differences). These numbers were fixed before any cohort run and are not tuned.
2. **A StarDist re-run** (`python -m benchmarks.regbench nuclei`, `slurm/nuclei.sbatch`:
   an array over SLIDES): the pipeline's `bin/segment.py`, unchanged, on the native slide,
   keeping its `*_nuclei_mask.tif`. The rendered command, and where each word is from:

   ```
   python <repo>/bin/segment.py --image <slide> --output-dir <nuclei dir>/<group>/<slide>/rerun \
     --dapi-channel <c> --model-dir $REGBENCH_STARDIST_MODEL_DIR --model-name $REGBENCH_STARDIST_MODEL \
     --n-tiles 16 16 --expand-distance 10 --pmin 1.0 --pmax 99.8 --use-gpu
   ```

   | word | from |
   |---|---|
   | `segment.py --image … --output-dir …` | `lib/SegBackends.groovy:100` (`entrypoint`), `modules/local/segment.nf:77-79` |
   | `--dapi-channel` | `lib/SegBackends.groovy:101`, `modules/local/segment.nf:80`: the pipeline passes `0`, its preprocessing having put the nuclear channel first; here it is the manifest's `ref_channel` / `mov_channel`, which is 0 on the arms' slides |
   | `--model-dir`, `--model-name` | `conf/modules.config:1021-1022` (`segmentation_model_dir`, `segmentation_model`); the arms take them from the site overlay `conf/ieo.config`, which is not in the repository: the two environment variables, or, when they are unset, that file's two keys (below) |
   | `--n-tiles 16 16` | `conf/modules.config:1023`; `nextflow.config:273-274` |
   | `--expand-distance 10` | `conf/modules.config:1024`; `nextflow.config:275` |
   | `--pmin 1.0 --pmax 99.8` | `conf/modules.config:1025-1026`; `nextflow.config:268-269` |
   | `--prob-thresh` | absent: `conf/modules.config:1029`, `nextflow.config:272` (`null`); `REGBENCH_STARDIST_PROB_THRESH` adds it if the site sets one |
   | `--use-gpu` | `conf/modules.config:1057`, `nextflow.config:267` (`seg_gpu = true`); `REGBENCH_STARDIST_CPU=1` drops it |

   None of these is set in `benchmarks/configs/arms.yaml` or `benchmark.config`
   (`test_nuclei_cohort.py::test_the_rerun_follows_the_pipelines_own_configuration` reads
   the defaults out of `nextflow.config` and the flag list out of `conf/modules.config`).
   The same call also writes the cell mask it grows from its nuclei, and that is compared
   with the arm's published outlines of the slide: the counts within 1 %, and at least
   0.99 of the published centroids with a re-grown centroid within 1 px
   (`RERUN_MATCH_PX`, `EXACT_MATCH`), beside the one-nucleus share above. What a re-run
   is then (user ruling 2026-10-11):

   | the re-run's share of published outlines holding exactly one of its nuclei | what happens |
   |---|---|
   | under 0.90 (`ACCEPT_ONE_NUCLEUS`, the threshold every label image is held to) | **REFUSED**: not used. A `REFUSED <slide>` line with the share and both counts is printed, the task exits 1, `--list` shows status `refused` with that reason, the slide has no nuclei rows, and the reason is in its cases' `cell_sets` and in `nuclei_missing.csv`. It is not segmented again unless `--force` is given (`REGBENCH_NUCLEI_FORCE=1` through the launcher, which the array task passes on): the same command gives the same nuclei, so the model, the channel or a site setting is what to look at. The array task of such a slide says so and exits 1 without running StarDist |
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
`cells.polys_from_label_image`, unsimplified. On the known-truth datasets the grown
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
directions). Measured on one 625-cell synthetic case on a laptop: 2.5 s per row on the
bare nuclei, and on the grown outlines 25 s unsimplified, about 9 s simplified. Two
things were done about it, neither of which changes a number: the grown outlines are
simplified as the pipeline's are (above), and a pair that both pairings chose is
rasterised once (`pairs._pair_iou`; a pair's IoU depends on its two cells alone;
`test_pairs.py::test_a_pair_both_pairings_share_is_rasterised_once`), which halves the
IoU work on the cohort too. Scoring the known-truth datasets is still slower than before
2026-10-11, because their outline rows are now grown outlines and a nuclei row is added.

**Known truth.** The probes are the centroids of the NUCLEI, as before, and the truth
spec is unchanged by it (now version 5, `1bade785…`): the expansion distance is the pipeline's
default, not a frozen parameter of the maps, the probes or the seeds, so it is recorded
per case (`extra.cell_sets.<side>.outlines.expand_px`) and not in `truth_spec.json`.
What DID change on `synthetic`: the columns that mean "outlines" (`cells.csv`,
`summary_cells.csv`, the off-seam rows) are now the grown objects and no longer the bare
ones; the numbers these datasets had before 2026-10-11 are the `_nuclei` tables. Cases
must be prepared again (the launcher's `--prepare`).

**Commands on the cluster** (after exporting what "What to export" lists):

```bash
# 0. the manifest, offering what already exists (both options are optional)
python3 -m benchmarks.regbench.arms_manifest --root <arm_results> --out rounds.csv \
    --nuclei-glob '<nextflow work>/*/*/{stem}_nuclei_mask.tif' --seg-arm <segmentation arm>
# 1. which slides already have a usable mask (checks every offered label image; runs no StarDist)
regbench_py STARE_REGBENCH nuclei --manifest rounds.csv --out "$REGBENCH_NUCLEI" --workers 4
regbench_py STARE_REGBENCH nuclei --manifest rounds.csv --out "$REGBENCH_NUCLEI" --list
# 2 + 3. the re-run for the rest, then prepare: one call
REGBENCH_MANIFEST=$PWD/rounds.csv benchmarks/regbench/slurm/submit_cohort.sh --prepare -- <site options>
#    (by hand: N=$(regbench_py STARE_REGBENCH nuclei --manifest rounds.csv --out "$REGBENCH_NUCLEI" --count)
#     sbatch --array=0-$((N-1)) -c 8 --mem=128G -t 8:00:00 $REGBENCH_SBATCH_GPU benchmarks/regbench/slurm/nuclei.sbatch
#     then: regbench_py STARE_REGBENCH prepare multiplex --cases "$REGBENCH_CASES" --manifest rounds.csv --nuclei-dir "$REGBENCH_NUCLEI")
# afterwards: every slide, its source, whether it is exact, or why it has none
regbench_py STARE_REGBENCH nuclei --manifest rounds.csv --out "$REGBENCH_NUCLEI" --list
grep -h DISAGREEMENT "$REGBENCH_OUT"/logs/rb-nuclei-*.log
```

Step 1 is optional: the array of step 2 does the same check for its slide first and runs
StarDist only when the slide is left without usable nuclei. `--list` prints `index, slide,
status (usable | needs_rerun | refused | candidate | none), source, exact, one_nucleus_frac,
n_nuclei, reason`. The allocation of the array (8 CPUs, 128 GB, 8 h, a GPU) follows the
pipeline's `SEGMENT` block (`conf/modules.config:1062-1067`: 8 CPUs, 32 to 128 GB by file
size, 4 h per attempt) at its upper end; `REGBENCH_NUCLEI_CPUS`, `_MEM`, `_TIME` change it.

**Not verified until the cluster** (none of it was run: StarDist is not installed where
this was written, the site's model is not in the repository, no arm's files were read):
that the re-run reproduces the published outlines at all (same model, same image, the
site's settings as its config's plain `key = value` lines give them: the `REFUSED` and
`DISAGREEMENT` lines and `exact` will say); that the site overlay is readable that way
(the real file was never read here: a key set in two profiles, or by an expression, is
refused with a message, not guessed); that `bolt3x/mirage-stardist:1.0.0` runs `bin/segment.py` from a bound
checkout under `apptainer exec --cleanenv --nv`; that StarDist is deterministic enough
between two GPU runs for the 0.99 thresholds; that a segmentation arm's mask passes for
its reference slide; what a work directory still holds; the memory and time of tracing
two label images of a whole slide in the scorer's image (`cells.polys_from_label_image`,
tile by tile).

## Known-truth error

The one independent check: on synthetic and semi-synthetic cases the map that made the
moving slide is known, so a method's error can be measured against it. Everything here is
fixed by `truth_spec.json`, which carries the SHA-256 of its own content, except which
cohort patients the semi-synthetic cases are cut from: that is the cohort's own list, or
a subset declared before any run ("Cases" below).

**Probes.** `truth_mov` are centroids of nuclei segmented on the native MOVING slide, and
`truth_ref = g(truth_mov)` with `g` the generating map (moving -> reference, exact). A
probe is kept when its true position lies at least 32 px inside the reference; at most
20000 per case, by a seeded draw. `truth_cell` says which moving nucleus each probe is.

**Direction, with no inverse of anything.** For a `mov_to_ref` method the error is
`fn(truth_mov) - truth_ref`; for a `ref_to_mov` method it is `fn(truth_ref) - truth_mov`.
The same pairs serve both. Neither the method's map nor `g` is inverted, which also keeps
the discontinuous `seams` family well defined.

**Families** (frozen; each is reported on its own, never pooled):

| family | amplitude | scale | note |
|---|---|---|---|
| `rigid` | none | | control: the pose alone |
| `grid` | s.d. 5 px (1.6 µm) | 9 x 9 nodes, 1024 px apart | nearly Casamitjana et al.'s generator, its pixel value carried over without rescaling ("Where the families come from") |
| `multiscale` | RMS 2.0 µm | 333, 166, 83 µm, fixed in µm | the three-octave structure is SynthMorph's; the amplitude and the scales in µm are the benchmark's |
| `bumps100` | 0.2 to 0.6 sigma | sigma 100 px, 120 bumps per 8192 px window | **the benchmark's own** (no published generator); inside STARE's model; 24 % of the window moves by more than 1 µm |
| `bumps25` | 0.2 to 0.6 sigma | sigma 25 px, 4000 bumps per 8192 px window | **the benchmark's own**; **outside STARE's model** (finer than its 64 px lattice); 23 % of the window moves by more than 1 µm |
| `seams` | s.d. 3 px (0.98 µm) per axis | 1950 px blocks, the cohort scanner's tile | **outside STARE's model** (discontinuous); no seam on any method's grid line |

Spec **version 4** (ruling R3.16, 2026-10-11): the family `wave` (a 10 px sinusoid of 1500
px) is REMOVED, because nothing read uses a sinusoidal field and it was a third family
inside a spline mesh's model. Nothing else changed: every remaining family has the
parameters, the scene and the seeds it had in version 3 (each family carries its own
`index` in the spec, which the seed rules read; `wave`'s, 1, is retired, so no remaining
case moved to another seed). Six families: 6 synthetic scenes, 12 cases.

### Where the families come from

Every source below was read for the port's literature report of 2026-10-11, not again
when this section was written; the tag after each is that report's record of how much of
the source was read, carried over unchanged. Where a statement rests on a partial read,
the sentence says so.

**The principle: test deformations should include ones the method cannot represent.**
Three sources state it, and it is why two of the six families (`bumps25`, `seams`) are
outside a 64 px mesh:

- Sorzano, Thévenaz, Unser, IEEE TBME 52:652, 2005 `[PARTIAL: read pp. 657 and 659
  (§IV-A) as images plus one discussion paragraph; missing §II–III]`: "The true
  deformations were picked from a deformation space that cannot be represented exactly by
  B-splines. The goal is to test the algorithm performance in a challenging situation that
  does not allow for an exact registration since this would bias our estimate of the
  accuracy." (§IV-A.1, p. 657). Their methods sections were not read.
- Casamitjana et al., arXiv:2104.14873 `[PARTIAL: read abstract, §3.1.1, §3.2–3.3; missing
  §2 methods, results, supplement]`: "We explicitly avoid using velocity fields to prevent
  imitating the deformation model used in the algorithm." (§3.3).
- Borovec et al., ANHIR, IEEE TMI 2020, PMC7584382 `[PARTIAL: read abstract, parts of
  intro, conclusion paragraph, reference list; keyword scan]`: "we would recommend that
  manually obtained landmarks are combined with … synthetically generated (and therefore
  known) deformations" (Conclusions). Only that paragraph and the abstract carry this
  here; the body was keyword-scanned.

**`grid`** nearly matches Casamitjana et al. (same tag): "normally distributed,
low-resolution deformation fields of size 9×9×2 … Each element in the low-resolution
field is an independent Gaussian variable with zero mean and a standard deviation which
is constant for each slice, and which is sampled from a uniform distribution U[3, 7] (in
pixels). Each deformation field is then resized to the original image size using B-Spline
interpolation." (§3.3). Two differences, both stated: the paper DRAWS the s.d. per slice
from U[3, 7] and the benchmark fixes 5, the midpoint, which is not itself a published
value; and the pixel value is transferred WITHOUT rescaling, from brain MRI slices at 1 mm
per pixel to 8192 px windows at 0.325 µm per pixel, so 5 px is 1.6 µm on nodes 1024 px
apart here and 5 mm there. The paper's methods and results were not read.

**`multiscale`**: three octave-spaced random fields summed is the structure of
SynthMorph's generator, Hoffmann et al., IEEE TMI 41:543 (arXiv:2004.10282) `[PARTIAL:
read §II-C, §II-D.1, Table I (page image), §III-G; missing the rest]`: "we sample several
SVFs v_m ∼ N(0, σ_v²) at resolutions r_v ∈ {1:8, 1:16, 1:32}, drawing a different σ_v for
each … The upsampled SVFs are then combined additively" (§II-C.3). Only the structure is
taken: SynthMorph integrates velocity fields and draws σ at random, the benchmark sums
displacements at a fixed RMS, and the 2 µm, the three scales in µm and their equal energy
are the benchmark's.

**The bump families are the benchmark's own.** No published generator was found for
them; the count, the width and the "about a quarter of the window above 1 µm" target are
the benchmark's. Their motivation is local damage as described, not measured, in:
Muhlich et al. 2022, ASHLAR, Bioinformatics 38:4613, PMC9525007 `[PARTIAL: read the full
main text; missing Supplementary Tables S1–S4 and Figs S1–S7, figures not viewed]`: "the
few regions with high error … corresponded to parts of the tissue in which cells had
become physically distorted or detached from the slide" (§3.4); and Krishna et al., MELBA
2026, PMC13599393 `[PARTIAL: read abstract, parts of intro, the residual-mask method
paragraph, one results sentence]`, where a mask of regions "exceeding 10 pixels" of
residual "excluded 6.65 ± 4.46% of the WSI area" between the first and the last of 17 to
19 rounds (from one results sentence of a partial read, measured by that method's own
fields).

**`seams`.** The 1950 px block is the tile size of the scanner the cohort was acquired on
(the owner's statement, 2026-10-11; it is not derived from a file here). The shift, s.d.
3 px = 0.98 µm per axis, sits at the low end of the stage repeatability the MIST paper
gives, "stage repeatability, that amounts to 1 to 2 microns for good microscopes" (MIST,
PMC5504007 `[PARTIAL: keyword scan only]`): that sentence was found by a keyword scan and
its context was not read, so it supports the order of magnitude and no more. That seam
error exists and changes at tile boundaries is from ASHLAR (tag above): "the field
direction was often consistent across large regions but could change dramatically at
tile boundaries" (§3.4). Independent shifts per block are the benchmark's simplification;
by the same passage real seam error is correlated over large regions.

**Amplitudes, against what has been measured.** Lotz, Weiss, van der Laak, Heldmann,
arXiv:2106.13150v2 `[FULL]` (main text; the PDF has no supplement), Table 3: on re-stained
sections "HyReCo restained (subset B) 1.60 µm [MTRE affine] 0.86 µm [MTRE deformable]",
i.e. an affine fit leaves a median 1.60 µm, of which 0.86 µm is at the level of the
landmarks' own noise. `grid` (1.6 µm s.d.), `multiscale` (2 µm RMS) and `seams` (0.98 µm
per axis) are of that size. Two caveats travel with the comparison: Lotz et al. is
BRIGHTFIELD (H&E re-stained to PHH3, coverslip removed), not cyclic immunofluorescence;
and ASHLAR's own figure for cyclic IF after per-tile translation, "0.119 µm (∼0.2
pixels)" median on 200 x 200 px blocks (§3.4, tag above; the authors add that it "only
measures the consistency of a stitching/registration algorithm against itself"),
suggests that rounds of cyclic IF move much less. So the known-truth test is probably
HARDER than the cohort, and an error here is not a forecast of an error there.

**What could not be grounded in anything read** (the report's list): the 2 degree
rotation and the (30, -20) px shift of the poses; both bump generators (counts, widths,
the quarter-of-the-window target); the three `multiscale` scales in µm and their equal
energy; the scene (uniform random nucleus placement, one per 900 px², the tissue
silhouette, amplitudes 0.5 to 1.5); any Jacobian threshold (Lotz et al. supports "no
folds between re-stained sections", not a number such as the re-seed rule's 0.05); a
per-round fraction of damaged area; and any synthetic-deformation protocol published by a
microscopy registration paper (none was found). The 1950 px block is grounded in the
owner's statement only.

**Seams.** With 1950 px blocks the seams of an 8192 px window fall at 1950, 3900, 5850 and
7800 px. None of these positions is a multiple of any grid the spec lists for a method
(`method_grids_px`): STARE's 64 px vector lattice and 2048 px tile, ASHLAR's 1755 px
tile pitch, PALOM's 1024 and 2048 px blocks. The test
`test_no_seam_falls_on_a_method_grid` checks every seam position against every listed
grid (`truth.seams_on_a_grid`), not the block size alone, and holds the spec's pitch to
the one the adapter cuts with. "Grid" here means multiples of the listed size from the
window's origin, where a tile or block STARTS. The first seam, at 1950 px, is where
ASHLAR's first tile ends, because the seam block and ASHLAR's tile are both the
scanner's 1950 px tile. A window smaller than 1950 px has no seam at all: a reduced copy
of this family is a constant shift.

**Bumps.** The counts are for the 8192 px window and were set so that about a quarter of
it moves by more than 1 µm, measured on the generated field (the pose not counted;
`synthetic.field_stats`, a lattice every 4 px):

| family | bumps | moved by > 1 µm, the two synthetic cases | over other seeds | smallest Jacobian determinant |
|---|---|---|---|---|
| `bumps100` | 120 | 23.7 %, 24.6 % | 21.8 to 25.0 % (82 seeds) | 0.48, 0.46 in the synthetic cases; 0.37 over the 82 seeds |
| `bumps25` | 4000 | 23.2 %, 22.9 % | 22.9 to 23.7 % (182 seeds) | 0.25, 0.32 in the synthetic cases; see below |

Neither synthetic case folds, and the scorer's own `fold_pct` of the true map is 0 for
both families. For `bumps25` a fold is a rare coincidence of bumps, not a matter of the
count: 1 seed of 182 at 4000 bumps had a negative determinant at one point of a 4.2
million point lattice, and so had 1 seed of 100 at 3400 bumps (0 of 100 at 3600 and
3800), so a lower count inside the 20 to 30 % range does not remove it. Each known-truth
case records what its own map does (`case.json`: `extra.field` = `moved_frac_over_1um`,
`min_jacobian`, `fold_pct` on a lattice every 8 px). The test
`test_bumps_move_about_a_quarter_of_the_window_without_folding` measures both families at
the frozen size and holds them to the fraction the spec records, within 0.02.

**A case whose true map folds is re-seeded** (spec version 3, the `reseed` block; the
user's ruling of 2026-10-11). Where a map folds, two moving points have one true
position, and no registration can be right there. The seeds of the semi-synthetic cases
depend on the patients' ids, so such a seed cannot be ruled out in advance; it is skipped
when the case is made:

- Before anything is drawn, `prepare` measures the smallest Jacobian determinant of the
  case's true map (pose and field) on a lattice every 4 px over the whole window
  (`synthetic.settle`). Above 0.05 the seed its seed rule asks for is used. Otherwise it
  is skipped for `seed + 1000000007`, then `seed + 2 x 1000000007`, up to 8 seeds; if all
  fold the case is refused, never made folded. The step is a prime above every seed a
  seed rule can ask for (the largest is under 1.1e8), so a re-seeded case never takes the
  seed another case asks for.
- The rule depends on the case's spec alone, so every array task that prepares the case
  settles on the same seed. The seed used replaces the asked one everywhere (field, noise,
  probe draw). `case.json` holds the seed used (`extra.spec.seed`) and the record:
  `extra.reseed` = `seed_asked`, `seed`, `skipped` (each skipped seed with its
  determinant), `step`, `lattice_px`, `min_jacobian`, `min_jacobian_required`. The
  summary's known-truth header names every re-seeded case, and still lists any case whose
  recorded `fold_pct` is above 0.
- **Why a margin, and why 4 px.** A determinant at or below 0 is a fold; a lattice only
  sees one where it has a point. Measured on 240 `bumps25` seeds at 8192 px (seeds
  3000000 to 3000239, 2026-10-11): the smallest determinant fell by 0.008 to 0.046
  (median 0.022) from an 8 px to a 4 px lattice, and on four seeds near zero by 0.006 to
  0.010 more at 2 px. Two seeds that an 8 px lattice passes at 0.018 fold at 4 px. A
  case kept above 0.05 at 4 px therefore has no fold between lattice points. Of the 240
  seeds 1 was at or below 0 at 4 px and 5 (2 %) were not above 0.05, so about one
  `bumps25` case in fifty is re-seeded.
- **Which families.** Every continuous one: `rigid`, `grid`, `multiscale`,
  `bumps100`, `bumps25`. Only `bumps25` has been seen to come near (the others' smallest
  determinant, on the frozen synthetic cases at 4 px: 0.90 to 1.0, and 0.46 and 0.48 for
  `bumps100`; 0.29 to 0.67 over 60 further `bumps100` seeds), so for them the rule is a
  guarantee that costs one measurement. **`seams` is not under the rule**: it is
  discontinuous on purpose, and a tile that shifts towards its neighbour overlaps it by
  design; a lattice across a seam reads that as a negative determinant on almost every
  seed (the two frozen cases: -0.85 and -1.0 at 4 px).
- **None of the 12 frozen synthetic cases is re-seeded** (the 10 under the rule: 0.25 to
  1.0); the test `test_no_frozen_synthetic_case_folds` measures all of them at 8192 px
  and holds the list of skipped seeds (empty) written out.

A reduced copy (a test, a smoke run at 1024 px) keeps the bumps' density, not their
number: 4000 x (n / 8192)^2. A bump is summed out to 8 sigma and is exactly zero beyond
(under 1.3e-14 of its peak there), so an 8192 px slide with 4000 bumps costs what the few
bumps near each block cost; the rule is per point, and a point has the same value in any
batch. Preparing one 8192 px `bumps25` case (both slides, segmentation, probes) took 49 s
with 4 workers on the development laptop, and twice in fresh folders gave the same pixels,
nuclei, probes and `case.json` (the files differ only in the OME header's UUID).

**Poses**, one per round: `shift` is (30, -20) px with no rotation; `rot` is 2 degrees plus
that shift. ASHLAR has no rotation, so its `rot` rows measure that limit and are read
beside its `shift` rows.

**Cases.** Windows of 8192 px for every method (4 x 4 STARE tiles). A patient is
(window, family): one reference, kept as one file, and two rounds. A per-pair method runs
each round; a per-patient method gets the patient and so never sees two families at once.

- `synthetic`: one Gaussian-blob scene per family (6 scenes, 12 cases; a scene is the
  independent unit, "Scoring").
- `semisynth`: variant `same`, noise 1.0 x the background s.d., on the 2 windows with most
  tissue of each cohort patient's reference round. **The spec names no patient.** The
  patients are every patient of the manifest given to `prepare`, or a subset declared with
  `--patients` / `--patients-file`; an id that is not in the manifest is an error that
  names it. Cost, stated up front: each patient gives 2 windows x 6 families x 2 poses =
  24 cases of 8192 px (12 per-patient launches in the sense above; ONE independent unit),
  and every method runs each of them.

```bash
python -m benchmarks.regbench prepare synthetic --suite truth --cases CASES
# every patient of the cohort (rounds.csv from `python -m benchmarks.regbench.arms_manifest`)
python -m benchmarks.regbench prepare semisynth --truth --cases CASES --manifest rounds.csv
# a declared subset
python -m benchmarks.regbench prepare semisynth --truth --cases CASES --manifest rounds.csv \
    --patients 046,052            # or --patients-file ids.txt (one id per line, # comments)
# one patient by hand (tests)
python -m benchmarks.regbench prepare semisynth --truth --cases CASES \
    --name <patient id> --image <its reference round> --channel <nuclear index>
```

`--manifest` reads two layouts. The `rounds.csv` of `arms_manifest`: each patient's
`reference` and `ref_channel`, one entry per `group`, pixel size from `pixel_size_um`. The
arms' raw `input.csv` (`patient_id, path_to_file, is_reference, channels`): the row marked
as the reference and the one channel whose name holds `DAPI` or `CELLTOX`; it has no pixel
size, so the file's own header or `--pixel-size-um` gives it, and the raw files must be
TIFFs this reader opens (the pipeline's preprocessed slides of `rounds.csv` always are).
With `--index` (a SLURM array) only the patient that case belongs to is opened; the index
runs over 24 x the number of patients, in the manifest's order.

**Which patients a result rests on.** `prepare` writes `CASES/semisynth/truth_patients.json`
(the ids, the manifest's path and SHA-256, the spec's hash, each patient's image and
channel, and the SHA-256 of ids + manifest content + spec hash) and that list's hash into
every case (`extra.truth_patients_sha256`, with `extra.truth_patient`). A folder that holds
one list refuses cases of another; `score` refuses cases of two lists, or a list file that
does not match its cases, and prints the list in the header of the known-truth table of
`summary.md`, with how many of the listed patients have cases so far. The one-patient form
declares a list of that one patient.

**The freeze.** `prepare` refuses a `truth_spec.json` whose content does not match its
hash and writes the hash into every `case.json` it makes from it
(`extra.truth_spec_sha256`); `score` refuses cases of two hashes, and refuses a
`synthetic` case whose hash is not the current `truth_spec.json`'s. `prepare synthetic`
refuses a cases folder that already holds a case of another spec, with or without
`--force`: such a case is neither kept as done nor written over (the results registered
on it would rest on images that are gone), and the message names the folder and says to
prepare into a new, empty one. The hash covers the
parameters, not the generator's code: the commit is what fixes that, and it is to be tagged
`regbench-truth-frozen-<date>` before any method runs. The spec has been re-hashed four
times, each before any method was scored on it: 2026-10-10 (version 2: seam block, bump
counts, no patient named), 2026-10-11 (version 3: the re-seed rule), 2026-10-11
(version 4: `wave` removed, a fixed `index` per family, the sources recorded) and
2026-10-12 (version 5: `method_grids_px` lists the pitch ASHLAR is run with and the
`seams` note says where its first seam falls; no family, field, seed or amplitude
changed, so every deformation is the one version 4 made). Version 5 is sha256
`1bade785322e5472314c8d2f05934c41a93e07b0b86521b5cdceff7def16dcbf`
(`test_version_five_changes_the_record_and_no_deformation` holds the hash, and a hash of
everything that makes a deformation, which is version 4's;
`test_wave_is_gone_and_no_other_case_moved` the seeds every family had in version 3). Cases prepared from an older version are
refused beside newer ones: prepare them again.

**Numbers** (`truth.error_stats`), per case and variant, in micrometres: `tre_um_p50`,
`tre_um_p90`, `tre_um_max`, `frac_under_1um`, `frac_under_2um`, `initial_tre_um_p50` (the
unregistered pose) and `frac_improved`, with `n_probes` and `n_nonfinite`. A point a method
returns as NaN counts against it: its error is infinite. A failed run is scored at the
unregistered pose. Without a pixel size the micrometre numbers are NaN. A variant with
seam mask's owner on that case gets a second row `<variant>:non_<owner>` in
`nonband_truth.csv`, on the probes the mask leaves (see "Off the seams": the probes of
the moving nuclei a moving-side mask leaves; for a reference-side mask, the probes the
owner's run did not mark).

**What these numbers are not.** The amplitudes are set against published measurements
("Where the families come from"), not STARE's output on the cohort. The bias is
elsewhere: the families, their smoothness and the multiscale spectrum were chosen by
STARE's authors after its design was fixed; `grid`, `multiscale` and `bumps100` are
fields a B-spline mesh on a 64 px lattice can represent; the `same` variant reuses the
reference's pixels, so every error is a lower bound. Any amplitude later estimated from a
method's fields is post hoc and names that method.

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
  owners, each row's selfcheck, the dropped rows, the submission being scored
  (`--submission`), every file of the package's top level and every module `score.py`
  imports inside the package, followed through. So a re-run of a method, a changed
  run list, another `--drop` or another submission makes the case's ROWS again, in
  seconds to minutes, and computes only the pair records whose inputs changed: a
  `run.json` that says something else changes rows and no pair record.
  A record that is cut short, damaged or made for other inputs is computed again and
  replaced, and the log says so (`started (made for other inputs: results.stare...)`).
  `score --fresh` reads no record at all.
- **`score-case`** scores one case and writes no table:
  `python -m benchmarks.regbench score-case --cases CASES --out OUT --dataset multiplex
  --index 3 --workers 8` (or `--case-id`). `score` then reads that case's record,
  provided it is given the same `--drop`, `--submission` and `--only-run`; it computes
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

Measured on a laptop (Apple M-series, one synthetic case: 37,913 reference and 38,096
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
on 8 cores; the whole run list (about 40 sources and 3 seam owners: about 320 pair
records, about 53 h of work) is about 7 h per case on 8 cores. Real tissue is denser
than this case (more pairs under the null), and the cluster's cores are not this
laptop's: read the first case's `seconds` in `case.json` before sizing the rest. Each
worker holds the case's polygons (hundreds of megabytes at that size) and the result
it is scoring.

### Tables (`<out>/tables/`)

| file | one row per | what |
|---|---|---|
| `cells.csv` | case x variant | `dataset, tier, case_id, group, family, pose, method, frame, selfcheck_status`, every key of `pairs.RECORD_KEYS`, `coverage, nn_distance_um_p50, fold_pct, sd_log_jac, imputed_initial, problem, cause`, and since 2026-10-11 `unit` (the independent unit), `n_unmapped, unmapped_frac`, `excluded_from_summaries`, `set` (`outlines`: this file is the outlines; the nuclei are in `cells_nuclei.csv`), and what the case records of that set: `set_source_ref`, `set_source_mov` (where each side's objects come from), `set_inexact` (either side is declared not to be, object for object, the segmentation it stands for) and `set_shrunk` (either side is the outlines with the 10 px rim taken off, the fallback for a slide without bare nuclei). Among the record keys: the pipeline pairing's own account (`n_components`, `largest_component_cells`, `n_fallback_components`, `n_fallback_cells`, `fallback_frac`, `fallback_frac_null`) and the exact assignment's extra columns (`*_exact`, `exact_status`). Every row scores ALL nuclei |
| `nonband_cells.csv` | case x variant x owner | the same columns for `<variant>:non_<owner>`, plus `variant, owner, n_mov_kept, n_mov_all, n_ref_kept, n_ref_all`. See "Off the seams" |
| `nonband_summary.csv` | dataset x owner x variant | the summary of those rows, with the nuclei kept and `within_ceiling`: what the mask alone lets `within_*um` reach for any method |
| `nonband_truth.csv`, `nonband_truth_by_family.csv` | as `truth.csv`, `truth_by_family.csv` | the known-truth error on the probes a moving-side mask leaves, with `owner, n_probes_all` |
| `cells_nuclei.csv`, `summary_cells_nuclei.csv`, `common_block_nuclei.csv`, `paired_vs_primary_nuclei.csv`, `paired_summary_nuclei.csv`, `nonband_cells_nuclei.csv`, `nonband_summary_nuclei.csv` | as the table of the same name | the same columns on the BARE NUCLEI (`set = nuclei`), with their own gate, null and shared subset. Absent when no case has nuclei. See "Outlines and bare nuclei" |
| `nuclei_missing.csv` | case, or case x variant | why there is no nuclei row: `dataset, case_id, group, …, method, reason`. `method` is empty when the CASE has no nuclei on a side (the prepared reason), and names a result that was made before its case had them |
| `dropped.csv` | variant | written only when `score --drop` names rows: `variant, reason, had_results`. Rows whose own selfcheck failed without blocking their method; they are in no other table |
| `nonband_missing.csv` | dataset x owner | `n_cases`, and `n_cases_without_mask`: the cases the owner's primary run did not register, which have no off-seam row for any method |
| `summary_cells.csv` | dataset x family x variant x subset | per unit the median over rounds, then median, min, max over units; `family` (empty on the cohort); `n_patients` (units), `n_cases, n_ok, n_excluded_not_method`; the minimum over cases and `<headline>_n_nan`; each headline beside its null; `dice_matched_shared_med, n_dice_shared` and `displacement_um_p50_shared_med, n_readable_shared` (the cases every compared row qualifies on) beside the per-row `dice_matched_ranked_med, n_dice_ranked`, `displacement_um_p50_med, n_readable`; `unmapped_frac_max`; `n_fallback_cases`, `fallback_frac_max` and the exact assignment's `pair_fraction_exact_med`, `dice_matched_exact_med`, `n_exact` (extra numbers, never ranked); `n_inexact_cases`, `n_shrunk_cases` (cases scored on objects marked `set_inexact` / `set_shrunk`). `subset=common` repeats it on the cases every compared row registered |
| `common_block.csv` | dataset x primary row | `in_common_block`, `why_not`, `n_cases_common`, `n_cases` |
| `paired_vs_primary.csv` | unit x variant (x family) | the difference of each headline against the `stare` row, on the cases both have |
| `paired_summary.csv` | dataset x family x variant | the median and the mean over units of each difference with its bootstrap interval over units (`d_<headline>_med, _mean, _lo, _hi`), `n_patients` |
| `excluded_rounds.csv` | lost round | written only when a round was lost for a cause that is not the method's: `dataset, method, case_id, group, cause` |
| `selfcheck_masks.csv` | selfcheck run x case (round) x variant x dot set | the selfcheck's numbers on `all` dots, the `unmasked` dots and the `masked` dots alone (`n_dots, found_frac, err_px_p95, err_px_mean, worst_block_err_px_p95, worst_block_found_frac, passes_accuracy`), `masked_frac, mask_cap, mask_over_cap`, the comparison's `n_detected` and `n_narrow` (what the detector took for a dot in the written image, and how many of those were narrower than 1.2 px), and one `passes_cap_<cap>` column per cap of `selfcheck.MASK_CAPS`. Written when the scorer is given the selfchecks (the launcher's `score`). Every table with a `method` column also carries `selfcheck_mask_over_cap` then |
| `attrition.csv` | dataset x registration row | where the rounds go: `manifest_rounds` (the rounds of the cohort's manifest; blank unless the scorer is given it, `score --manifest`, which the launcher's score job does when `REGBENCH_MANIFEST` is set), `prepared`, `with_outlines`, `with_nuclei`, then per row `run` (a record that is not `not run`), `registered` (`ok`), `scored` and `scored_nuclei` (the row is its own result: not the unregistered pose, not left out). Also a table of `summary.md` |
| `failures.csv` | variant | launches made, launches failed by cause, rounds lost |
| `failure_log.csv` | failed or lost EVENT | always written (a header alone when nothing failed). See "The failure log" |
| `resources.csv` | case x variant | the cost recorded at that variant's yield |
| `summary_resources.csv` | patient x variant | sum over rounds for a pair method, the launch's total for a per-patient one; `write_s` and `total_to_image_s` are the patient's totals for both; peaks are maxima, never divided by rounds; `complete` (every round registered), `launched` |
| `cost_summary.csv` | dataset x variant | the median over patients of each cost twice: over the patients this row completed (`n_patients_complete` of `n_patients`) and over the patients every compared row completed (`*_common_med`, `n_patients_common`) |
| `truth.csv` | known-truth case x variant | `dataset, tier, case_id, group, family, pose, in_stare_model, truth_spec, method, frame`, the numbers of "Known-truth error", `fold_pct, sd_log_jac, imputed_initial, problem, cause`; plus the pseudo-rows `initial` and `truth` (zero by construction) |
| `truth_by_family.csv` | dataset x family x pose x variant | `n_cases, n_ok`, median and maximum of p50, median p90, maximum of max, median shares under 1 and 2 µm, median `fold_pct`. No row over families |
| `tissue.csv`, `tissue_summary.csv`, `tissue_missing.csv`, `tissue_dose.csv`, `tissue_dose_summary.csv` | see "Tissue-level agreement of the written images" | ACCREDIT's score and its pieces on each written image. Secondary: in no table above, no paired difference, no ranking |
| `cost_end_to_end.csv` | patient x result that wrote an image | the steps apart, all measured here: `method_convert_s`, `benchmark_prep_s`, `estimate_s`, `write_s`, their sum `measured_here_s`, `benchmark_copy_s`, the peaks, `device`, `gpu_model`, `written`, `merged_image`. Secondary; see "Cost" |
| `cost_end_to_end_summary.csv` | dataset x result | the median over patients of each step and of `measured_here_s`, the largest peaks (`estimate_peak_gb_max`, `job_peak_gb_max`), `n_patients` of `n_patients_cohort`, and the same medians on the common patients (`*_common`, `n_patients_common`) |
| `launch.csv` | gate record | written only with `score --gates`: what each of the launcher's gates decided (`run, gate, status, blocked, dropped, reference, log_path`) |
| `summary.md` | | the summary: the unit of each dataset and the line "descriptive; no ranking is claimed" (always), the compared rows and who is not among them, the paired differences with their interval, the known-truth table, the failure log's section, cost on survivors and on the common patients |

### Off the seams

A method that builds its image from tiles or blocks draws some nuclei across a join, and
its point map is not exactly its image there. Such a method (`score.SEAM_OWNERS`: `ashlar`,
`palom`, `palom_b1024`) writes a **seam mask** with its primary row:
one bool per nucleus, from where it put its tiles or blocks and from no score (ASHLAR: the
tile grid and the positions it accepted; PALOM: its block grid and the shifts it accepted).
The mask is therefore NOT fixed before the run: it depends on the owner's own outcome on
that case (which positions and shifts it accepted), though on no other method's and on no
score.
ASHLAR's is on the moving nuclei (`info["band_mov"]`, `band` in `warped.npz`); PALOM's is
on the reference nuclei (`info["band_ref"]`, `band_ref`).

**The mask on the bare nuclei** (`band_nuc`, `band_ref_nuc`; rows in
`nonband_cells_nuclei.csv`). Where the case says which outline each nucleus belongs to
(`nuclei_<side>_outline`: the known-truth datasets, the built-in segmenter and the shrunk
fallback, where the two sets are the same objects by construction) **a nucleus inherits
its outline's flag**, so both sets drop the same objects. Otherwise (the cohort's label
images and re-runs, whose nuclei are not indexed like the published outlines) the mask is
**recomputed from the owner's geometry**: the owner applies its own seam rule to the
nuclei's polygons (`info["band_mov_nuc"]` / `info["band_ref_nuc"]`; ASHLAR: any vertex of
the nucleus on a seam; PALOM: the nucleus's own bounding box, grown by its margin, over
blocks with different shifts). A nucleus is smaller than its outline, so the recomputed
mask marks fewer objects than the outlines' mask does. `run.json: nuclei_band` says which
rule gave the mask (`inherited` | `geometry`). The known-truth probes are scored once,
with the outlines' mask.

**Which rows a reference-side mask removes.**

- `nonband_cells.csv`: the marked reference nuclei are taken out of the reference set
  before pairing (`n_ref_kept` of `n_ref_all`); every moving nucleus stays
  (`n_mov_kept == n_mov_all`). A moving nucleus whose partner was removed is then
  unpaired, for every method alike.
- `nonband_truth.csv`: a probe is a pair, the centroid of a moving nucleus (`truth_mov`,
  nucleus `truth_cell`) and where the truth puts it on the reference (`truth_ref`).
  Reference nuclei carry no probe index, so the owner's run marks the probes itself
  (`info["band_truth"]`, one bool per probe, `band_truth` in its `warped.npz`) and the
  scorer drops exactly those (`score._truth_keep`). PALOM's rule
  (`methods/palom.py::probe_band`) is its nucleus rule applied to the probe's own
  nucleus: the bounding box of moving nucleus `truth_cell`, moved so that its centroid
  sits at `truth_ref`, grown by 1 px, touches blocks that do not all carry the same
  shift. An owner with `band_ref` and no `band_truth` gets cell rows and no truth rows.

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
  `within_*um` against its ceiling and against the `truth` row, never against a
  full-slide number.

The cases of the frozen known-truth spec are left out of `resources.csv` and
`summary_resources.csv` (cost is read on the cohort); their failures are still counted in
`failures.csv`.

## The failure log (`failure_log.csv`): every event, and whose it is

`failures.csv` counts. `failure_log.csv` (ruling R3.17, `failure_log.py`) is what the
counts are made of: **one row per failed or lost event**, written by every `score` (a
header alone when nothing failed).

- every failed attempt of every unit: the attempt a result ended with, and every earlier
  attempt of the memory ladder kept in `run.json: previous_attempts`, even when the unit
  ended `ok` at a higher rung;
- every run a gate blocked (`gate_blocked`) and every row a gate dropped (`dropped_row`);
- every selfcheck attempt that did not end `ok` (`selfcheck_inconsistent`,
  `selfcheck_not_checked`, `selfcheck_crashed`) and every flagged realcheck
  (`realcheck_flagged`);
- every written image the tissue block could not score because of an error
  (`tissue_missing`), every record the scorer could not parse (`unreadable_record`), and,
  folded into one row per result name, the results nobody ran (`not_run`).

A per-patient launch that fails loses every round at once: one event per result name, the
rounds joined in `case_id`.

Columns: `dataset, run, variant, unit, group, case_id, attempt, rung_gb, retry, submission,
launch_id, stage` (`selfcheck | gate | realcheck | register | writer | benchmark_points |
tissue | score`), `cause, whose, cause_from` (how the cause was decided: the attempt's own
record, the scheduler's state, a gate's record), `scheduler_state, exit_code, elapsed_s,
peak_rss_gb, message` (the first line of the error), `log_path` (the scheduler's log of
that task), `climbed_to` (the next rung, "the same rung again", or `-` when the unit ended
there), `final_outcome_of_unit` (`ok at rung N | failed | lost | blocked`).

**`whose` is total and is one table** (`failure_log.WHOSE`; a test walks every cause the
code can write against it, so a new cause cannot be added unclassified):

| whose | causes | means |
|---|---|---|
| `method` | `exception`, `oom`, `gpu_oom`, `timeout`, `writer`, `writer_oom`, `killed` | the package under test raised, ran out of memory or time, its point mapping or its writer failed, its process died. `killed` is a process that died with nobody saying why; `regbench settle` renames it when the scheduler's accounting names a cause (memory, time, a lost node) or the task's script noted a Python ended by signal 9 |
| `benchmark` | `benchmark_oom`, `benchmark`, `selfcheck_inconsistent`, `selfcheck_not_checked`, `selfcheck_crashed`, `realcheck_flagged`, `tissue_missing`, `unreadable_record` | the benchmark's own step or check. A selfcheck that does not pass means the benchmark cannot show that ITS points follow the method's image; it is not evidence that the method registers badly |
| `infrastructure` | `infrastructure`, `selfcheck_lost` | the scheduler or the node lost the attempt |
| `launcher` | `gate_blocked`, `dropped_row`, `not_run`, `not_submitted` | a gate kept the run from starting, a row was dropped, the unit was never submitted, a stage job never reached the queue |

`summary.md` has a "Failures: every event, and whose it is" section: the count of events
per run, stage, cause and whose; then every unit whose FINAL outcome is not ok, with its
cause, whose, message and log path; and a bold line when a unit ended for a reason that
is not the method's.

**Where the log path comes from.** Every batch script sources `slurm/env.sh`, which
derives the task's own log name from the scheduler's variables by the launcher's pattern
(`$REGBENCH_OUT/logs/%x-%A_%a.log` for an array task, `%x-%j.log` otherwise) and exports
it as `REGBENCH_LOG`. The driver writes it into `run.json` (`log_path`, and into each
`previous_attempts` entry with that attempt's error text, phase and peak), the selfcheck
into `selfcheck.json` (`log_path`, and per attempt with `why`), and the gate into its
record (a sixth tab-separated field). It was not checked on a cluster that `%x` equals
`SLURM_JOB_NAME` for these jobs ("Not verified").

**The same words in the logs.** Every batch script ends with one machine-readable line:

```
REGBENCH_END kind=run run=palom dataset=multiplex unit=P03_r2 rung=2 status=failed cause=writer_oom whose=method log=/.../rb-palom-multiplex-r2-812_4.log
```

`kind` is `run`, `selfcheck`, `gate`, `tissue` or `score`; `status` is `ok`, `failed`
(the method's), `lost` (anything else), `blocked`, `may_start`, a selfcheck status, or
`no_end_line`. The Python step writes the line; the script repeats it as the log's last
line, and writes a fallback itself (`status=no_end_line cause=killed whose=method
exit=<code>`) when the step died before it could. `grep -h REGBENCH_END
$REGBENCH_OUT/logs/*.log` is the same classification as the table. Not recorded:
`exit_code` (the scheduler's accounting is asked for state, elapsed and CPU time only),
and the cost of a selfcheck attempt.

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
| `truth` (known-truth cases) | the moving slide resampled by the generating map, drawn by the benchmark (bilinear, at the evaluation size; the inverse of the generator's map is taken on every 4th evaluation pixel) |

A variant with no image has NO row and one line in `tissue_missing.csv` with the reason:
`not run`, `the run failed (<cause>): no image`, `the method writes no image for this
variant` (`stare_rigid`, `palom_affine`, the VALIS rigid and non-rigid stages, DeeperHistReg's
estimated-field rows), `the method's writer failed: ...`. A failed run is NOT scored at
the unregistered pose here, so `n_scored` of `n_cases` is printed beside every median.

| file | one row per | columns |
|---|---|---|
| `tissue.csv` | case x variant | `dataset, case_id, group, family, pose, variant, source` (`method` or `benchmark`), `status` (`ok` or `no_tissue`), `qcs`, `tissue_dice, boundary_dice, hausdorff_mean, hausdorff_p95` (evaluation px), `hausdorff_mean_um, hausdorff_p95_um`, `inlier_ratio, n_inliers, n_matches, nmi, ncc` (stored by the source, not in the composite), `coverage, bg_coverage`, the bounded terms `s_td, s_bd, s_hd, s_ir, s_nmi, cov_factor, bg_penalty`, `eval_h, eval_w, eval_um_per_px, ref_tissue_frac, mov_tissue_frac, ref_otsu, mov_otsu, crop_box, opencv, definition` |
| `tissue_summary.csv` | dataset x variant | `source, n_scored, n_cases, n_no_tissue, n_below_025`, and for each piece `<piece>_median` and `<piece>_worst` over the scored CASES (minimum; maximum for the two distances) |
| `tissue_missing.csv` | case x variant without a row | `dataset, case_id, group, variant, reason` |
| `tissue_dose.csv` | known-truth case x shift | the columns of `tissue.csv` plus `shift_um`: the `truth` plane moved by 0, 1, 3, 10, 30 and 100 µm along x |
| `tissue_dose_summary.csv` | dataset x shift | the summary of those rows: what the block can resolve |

The numeric composite and the count under 0.25 are printed; the source's good /
acceptable / poor classes are not. The known-truth datasets are separate rows (`dataset`),
and their evaluation pixel is not the cohort's: an 8192 px window at 4000 px is 0.67 µm per
evaluation pixel, a 50000 px slide 4.06 µm.

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
   `tissue.json` beside each reduced plane (`<out>/cache/tissue/...` for the reference,
   `initial` and `truth`). `score` reads those; what it finds neither on disk nor
   computable is listed in `tissue_missing.csv` with the reason, never left out silently.
   Use ONE OpenCV version for a whole result: the version is recorded in each row
   (`opencv`), and ORB and RANSAC are OpenCV's. That is why the pieces are computed in
   one image, whatever OpenCV a method's own image carries (upstream's VALIS 1.2.0 image
   has 4.9.0.80; the PALOM image and CI have the pinned 4.10.0.82).
3. The benchmark's own planes under `<out>/cache/tissue` (the reference, `initial`,
   `truth`) are kept, each with a record of what it was made from (`<plane>.json`: the
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

Per round and per patient: registration wall time, CPU-hours of the whole process tree,
peak memory, GPU-hours with the device named, and total time to the registered image
(`register_s + write_s`). No single column is "the" cost.

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
- **The ladder.** Every method runs at 4 CPUs; the two DeeperHistReg `_gpu` runs (one of
  them a primary row) have one GPU as well, which no other method has (`device` and
  `gpu_model` are columns of the cost tables, and their `_cpu` twins are shown beside
  them). Memory starts at 16 GB and moves up one rung
  (64, 256, 512 GB) only when the run ran out of memory. `attempt`, `mem_requested_gb` and
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
  each group of arrays of the case's `points.npz` (`outlines`, `nuclei`, `truth`). A
  case prepared again with other points (a forced re-prepare; outlines whose meaning
  changed) makes its earlier results stale: they are not reused, and the scorer counts
  them as `not_run` ("made for an earlier version of this case"), never as a failure of
  the method. Adding a group the result never had (bare nuclei to a case with outlines
  only) leaves it valid.
- **Threads.** No thread variable is exported. Where a method has its own core-count option
  it is set to `resources.n_cpus()` = `min(SLURM_CPUS_PER_TASK, affinity)`; both numbers
  are recorded. `cpu_efficiency` (CPU time over wall x CPUs) is reported because a method
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

**The `stare_prep` row** runs the `stare` adapter, so its line is `stare`'s, with one
step before it: both images are preprocessed by the benchmark (`prep.py`) BEFORE the
registration clock starts. Its `register_s` (the headline cost) is therefore STARE's
alone and does not hold that step, while DeeperHistReg's and VALIS's own preprocessing
are inside their clocks. So the step is recorded apart (`run.json: info.prep.prep_s`,
`peak_rss_mb`), shown as `benchmark_prep_s` / `benchmark_prep_peak_gb` in
`cost_end_to_end.csv`, and added to that row's `measured_here_s`: read that total, not
`register_s`, to compare `stare_prep`'s cost with another method's.

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
| `cpu_h_allocated` | sum over those attempts of allocated CPUs x duration |
| `cpu_h` | sum over the same attempts of their measured CPU time |
| `unmeasured_attempts` | stage attempts that ended without a recorded duration (a job the scheduler ended at its time limit): in no sum, so where it is not 0 every time and CPU figure of the row is a LOWER BOUND |
| `largest_job`, `peak_rss_gb` | the stage of the single attempt with the highest peak memory, failed ones included, and that peak |
| `n_jobs`, `n_tiles`, `n_job_retries` | attempts run, tiles, attempts beyond the first of each job |
| `peak_cpus_at_once` | the most allocated CPUs in jobs of the round that overlapped in time (per patient: the largest round's) |
| `write_cpu_h_allocated`, `write_peak_gb` | STITCH's own process, the attempt that wrote the image: the pipeline's 4 CPUs x its duration, and its peak |

For these rows the headline columns are defined as follows. `register_s` is
`wall_unlimited_parallel_s`: the queue-free figure, because the waits belong to the
cluster's load and not to the method. `total_to_image_s` is `register_s` plus
`write_s`, STITCH's wall time in the round's last job. `peak_rss_gb` is the largest
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
row of `failure_log.csv` (`retry` = `stage`, the job named in `message`): a failed
`prep` job is the benchmark's (`benchmark`, `benchmark_oom`); a stage job that never
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

The headline cost is the time to ESTIMATE the transform (`register_s`), and stays so;
the headline tables also carry the time to the written image and the memory peaks
(`summary_resources.csv`, `cost_summary.csv` and the block "Cost per method" of
`summary.md`: `total_to_image_s`, `peak_rss_gb`, `cgroup_peak_gb`). A second table puts
every step in its own column, per patient and per result that wrote an image:

| column | what | measured |
|---|---|---|
| `method_convert_s` | a conversion the method needs before it can read the slides: DeeperHistReg's single-page TIFFs (`convert_s`, plus the further channels its writer needs), ASHLAR's tiles (`retile_s`, once per patient). `none` when it reads the cohort's files as they are; `not separable` for VALIS | here, 4 CPUs, clock paused |
| `benchmark_prep_s` | the `stare_prep` row only (`none` elsewhere): the benchmark's preprocessing of both images before STARE starts (`run.json: info.prep.prep_s`), with the process's peak at its end in `benchmark_prep_peak_gb`. It is outside the registration clock and INSIDE this row's `measured_here_s`: the other methods' own preprocessing is inside their clocks | here, 4 CPUs |
| `estimate_s` | the registration clock | here, 4 CPUs |
| `write_s` | the package's own writer | here, 4 CPUs, clock paused |
| `measured_here_s` | `method_convert_s + benchmark_prep_s + estimate_s + write_s`: the total measured time of that result, from the cohort's files to the image it wrote | here |
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

## Running the cohort on a cluster

`slurm/submit_cohort.sh` submits from a login node and returns; nothing waits there and
no job waits on `sbatch --wait`. One kind of job submits jobs: the tile submitter of a
run whose registration is several jobs per round (`feed_tiles.sbatch`, below). **It has
never been submitted**: what is tested is the job graph its dry run prints, the same
graph through a stand-in `sbatch` (`test_submit_cohort.py`), the tile submitter against
a stand-in `sbatch` and `squeue` that keep a queue (`test_feed_tiles.py`), and the rules
its jobs apply (`test_ladder.py`, `test_failures.py`, `test_gates.py`).

### What it submits

The run list comes from `python -m benchmarks.regbench runs` and from nowhere else; the
script names no method.

1. **Selfchecks**, one per run that gates a run (17), in that run's image with that run's
   options and mask (`overlap` for ASHLAR, `seams` for PALOM), at 64 GB and 4 CPUs. Two
   more attempts wait behind it at 256 and 512 GB; each runs only when the one before
   ended for memory or left no record of this submission (it was lost to the scheduler,
   or never started: `selfcheck --retry-memory`), and otherwise leaves the record as it
   is. `selfcheck.json: attempts` holds the allocation and outcome of each. The anchored
   ASHLAR rows' selfcheck waits for STARE's and reads its work (`--m0-from`), case by
   case: STARE's selfcheck runs ASHLAR's patient of three slides too, for that reason.
   One job per run covers all of its cases, the patient included, so the number of jobs
   and the rungs (64 / 256 / 512 GB) are unchanged. Records:
   `<out>/logs/selfcheck/<run>/selfcheck.json`.
2. **A gate per run** (19; `gate.sbatch`, `regbench gate`). It fails when the selfcheck
   that gates the run is anything but `ok`, a selfcheck that left no record included.
   Then that run, and only that run, never starts: every one of its arrays waits for its
   gate with `afterok`, and every job carries `--kill-on-invalid-dep=yes` so that it is
   cancelled instead of pending for ever. Every job, the tile jobs included, also
   carries `--no-requeue`: the scheduler never starts a task a second time, so a task
   on a failed node ends as `NODE_FAIL` and the `-again` array of the ladder is the one
   retry (sbatch manual, `--no-requeue`: "the batch job should never be requeued under
   any circumstances" `[PARTIAL: read the --no-requeue, --requeue, --kill-on-invalid-dep
   and --dependency entries; missing the rest of the page]`; the manual notes that a
   site's `ForceRequeueOnFail` prolog flag overrides it). The task itself reads the gate's record again
   before it runs. `<out>/logs/gates/<run>.tsv`: run, status, rows to drop, blocked,
   the reference comparison.
3. **The ladder**: per run and dataset, EIGHT job arrays, two per rung (16, 64, 256 and
   512 GB, 4 CPUs each), each waiting for the one before with `afterany`: the attempt at
   that rung (`run --ladder first` at 16 GB, `climb` above), then `-again`, the same
   rung once more (`run --ladder again`). One task is one case for a run that works per
   pair and one patient for a run that works per patient.
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
   - **`multiplex` first.** Every array of a run on the next dataset waits for that
     run's last array on the one before, and `multiplex` is always first: the cohort is
     done before the known-truth cases start. The anchored ASHLAR rows also wait for
     STARE's last array on the same dataset.
   **A run whose registration is several jobs per round** (`runs`, column `stages`:
   `stare` and `stare_prep`) has, per dataset, a chain of stage jobs in place of that
   ladder, every job after the one before it (`afterany`) and all of them behind the
   run's gate. With R rounds and T tiles in all (a round's tile count is
   `ceil(H / 2048) x ceil(W / 2048)` of its reference, known before anything runs:
   `regbench stages` writes one line per round to
   `<out>/logs/stages/<submission>/<run>-<dataset>.tsv`):

   | jobs | what | CPUs, memory, time per task |
   |---|---|---|
   | `rb-<run>-<ds>-prep-a1..a4` (`stare_prep` only) | 4 arrays of R tasks | 4; 16, 32, 48, 64 GB; 1 to 4 days |
   | `rb-<run>-<ds>-coarse-a1..a4` | 4 arrays of R tasks | 2; 2, 4, 6, 8 GB; 2, 4, 6, 8 h |
   | `rb-<run>-<ds>-feed-1..2` | the TILE SUBMITTERS (`feed_tiles.sbatch`), `REGBENCH_STAGE_FEEDERS` of them chained: not a tile job, the job that submits them | 1; 1 GB; 7 days |
   | `rb-<run>-<ds>-tile-a1-o<n>` (submitted by the tile submitter, not by the launcher) | ONE TASK PER TILE: T tasks in arrays of at most `REGBENCH_STAGE_ARRAY_MAX` (100), at most `REGBENCH_STAGE_QUEUE_MAX` (100) of them pending or running | 2; 4 GB; 2 h |
   | `rb-<run>-<ds>-tile-a<2..4>-o<n>` (the same) | a RETRY JOB: one task per round that has a tile the pipeline would retry; it runs that round's due tiles one after another | 2; 8, 12, 16 GB; 4, 6, 8 h |
   | `rb-<run>-<ds>-solve-a1..a4` | 4 arrays of R tasks | 1; 1, 2, 3, 4 GB; 8, 16, 24, 32 h |
   | `rb-<run>-<ds>-last-<16,64,256,512>g[-again]` | 8 arrays of R tasks (`run_rung.sbatch ... jobs`): the round's LAST JOB reads the stage records, moves the benchmark's points, runs STITCH as a process of its own and reduces the tissue plane | 4; 16, 64, 256, 512 GB (the benchmark's ladder); one round's `REGBENCH_TIME` plus 40 h |

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
     at or below `REGBENCH_STAGE_QUEUE_MAX`; it waits for `REGBENCH_STAGE_FEED_MIN`
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
     leaving, and then gives up whatever the reasons.

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
4. **The realcheck gate** of `valis_mirage` and `valis120_hires`: the first patient of
   the first dataset goes alone, on its own ladder, with `run --realcheck`; a second gate
   job reads the primary row's `realcheck.json`; everything else of that run waits for
   it.
5. **`settle`, per-case scoring and `tissue`, then `score`**, in the scorer's image
   (`variants.SCORER_IMAGE`), after everything, whatever failed. FIRST one small job,
   `rb-settle` (`score_cohort.sbatch settle`, 1 CPU, 4 GB, one hour), which waits for
   every ladder and every gate with `afterany`: it asks the scheduler's accounting what
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
   `--submission` (below), `--drop` for every row a gate said to drop and `--gates` (the
   summary's block "What the gates decided" names every blocked run and every dropped
   row).

**What is submitted by default: `multiplex`, then `synthetic`.** `semisynth` and the two
HyReCo sets are DORMANT (ruling R3.14): implemented and tested, never part of the default,
and submitted only when `--datasets` names them, with a line saying so. Read "Dormant
datasets: known defects before any use" first.

**One submission, one id, and no stale record.** Each call of the script has an id
(`REGBENCH_SUBMISSION`, default the time and the process id) that every job carries into
its records (`run.json: submissions`, `selfcheck.json: submission`). `REGBENCH_OUT`
defaults to the commit, so a second submission at the same commit writes into the same
folder; these rules keep one from answering for the other:

- a selfcheck removes the record in its folder before it runs and writes its own last; a
  later attempt (`--retry-memory`) and the gate take a record of another submission as
  no record. A selfcheck attempt the scheduler lost, or that never started, is run again
  at the next of its three attempts, not read as `not_checked`;
- the gate records of the selected runs are removed when the script submits;
- an attempt removes `realcheck.json` before the method starts, and the realcheck gate
  counts a realcheck only beside an `ok` result of this submission;
- the ladder takes a result of another submission as no record;
- the scorer is given `--submission`: a result this submission neither ran nor reused is
  `not run`. A blocked run's older folders are therefore not scored as if it had run.
  Reuse is the exception that is meant: a unit already `ok` for the same configuration is
  not run again, its records gain this submission's id, and it is scored
  (`REGBENCH_FORCE=1` runs everything again).

**What "the same" means for reuse: what was computed, not the command line.**

- *A unit* (`run.json: reuse_key`, `methods.reuse_key`) is reused only when, besides
  the method, label, options, cases and points, four things are equal: `execution`
  (one process, or the scheduler jobs of a staged run), `code` (a sha256 of the driver,
  the adapter module, the modules of `methods/` it imports and the `regbench` modules
  it imports directly), `inputs` (for every case, the fingerprint of each image file it
  names) and `method` (what the adapter adds: STARE's version and the identity of its
  SOLVE job). A result made by other code, in another execution shape or from other
  bytes is run again; so is a result of before the key existed.
- *A stage job* of a staged run (`stages/jobs/*.json: identity`,
  `methods.stare.Round.identity`) is reused only for the same command AND the same
  STARE version AND the same bytes of what it reads: the two slides for COARSE and a
  tile, the anchor and the tile plan for a tile, the anchor and every control file for
  SOLVE, and for `prep` the slides, the candidate and a sha256 of `prep.py`. A stage
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
  destroyed: `run.json` and `warped.npz` are moved to `<result>/earlier/<n>/` and the
  new record names the folder (`earlier_result`). The current submission's row reads
  `infrastructure`, not that earlier result; moving the two files back restores it.

A GPU is asked for only for the runs whose `gpu` column is 1, with the site's own
request string, and the script refuses to submit them without it. An image is given the
GPU (`--nv`) only for those runs. VALIS 1.2.0's image is started with `--writable-tmpfs`.
No job asks for more than 7 days: a run that works per patient gets `REGBENCH_TIME`
times the rounds of the largest patient, capped there ("Cost" says what that means per
round).

### What to export

| variable | what | default |
|---|---|---|
| `REGBENCH_CASES` | prepared cases | `$HOME/regbench_cases` |
| `REGBENCH_OUT` | results, logs, gates | `$HOME/regbench_results/mirage-<commit>` |
| `REGBENCH_BIND` | paths to bind into every image, comma-separated: where the cohort's slides and nuclei live | none |
| `REGBENCH_WORK` | scratch for a method's intermediates | `$TMPDIR` |
| `REGBENCH_SIF_STARE_REGBENCH`, `REGBENCH_SIF_VALIS_MIRAGE`, `REGBENCH_SIF_VALIS120`, `REGBENCH_SIF_DEEPERHISTREG`, `REGBENCH_SIF_ASHLAR`, `REGBENCH_SIF_PALOM` | the six images of the run list (a seventh, `REGBENCH_SIF_STARDIST`, is for `--prepare` only); the script refuses to submit when one a selected run needs is missing. Local `.sif` paths: "Before the first real submission" | none |
| `REGBENCH_SBATCH_GPU` | the site's GPU request, e.g. `--gres=gpu:nvidia_h200:1`; required for the four GPU runs | none, on purpose |
| `REGBENCH_TIME` | per round, every rung | `24:00:00` |
| `REGBENCH_SELFCHECK_TIME` | every selfcheck attempt | `24:00:00` |
| `REGBENCH_ARRAY_LIMIT` | tasks of one array running at once (`%N`) | `20` |
| `REGBENCH_ACCT_WAIT_S`, `REGBENCH_ACCT_POLL_S` | how long a ladder task waits for the scheduler's accounting to name the end state of an earlier task whose result still says `killed`, and the seconds between two looks ("The ladder") | `600`, `30` |
| `REGBENCH_ARRAY_LIMIT` (above), `REGBENCH_STAGE_ARRAY_MAX`, `REGBENCH_STAGE_QUEUE_MAX`, `REGBENCH_STAGE_FEED_MIN`, `REGBENCH_STAGE_POLL_S`, `REGBENCH_STAGE_FEEDERS` | must be whole numbers of at least 1: the script refuses anything else before it submits | |
| `REGBENCH_STAGE_ARRAY_MAX` | the most tasks in one tile array of a run that is several jobs per round | `100` |
| `REGBENCH_STAGE_QUEUE_MAX` | the most tile tasks of ONE such run pending or running at a time (two such runs, `stare` and `stare_prep`, have a ceiling each) | `100` |
| `REGBENCH_STAGE_FEED_MIN` | free slots the tile submitter waits for before it submits an array (fewer when fewer tiles are left) | `25` |
| `REGBENCH_STAGE_POLL_S` | seconds between two looks at the queue | `60` |
| `REGBENCH_STAGE_FEEDERS`, `REGBENCH_STAGE_FEED_TIME` | tile submitters chained per run and dataset, and the time limit of each | `2`, `7-00:00:00` |
| `REGBENCH_STAGE_SUBMIT_TRIES`, `REGBENCH_STAGE_ARRAY_SPAN` | read by the tile submitter: tries of a refused `sbatch`; the widest span of indices in one tile array (below the site's `MaxArraySize`) | `5`, `1000` |
| `REGBENCH_STAGE_SQUEUE_TRIES` | read by the tile submitter: looks in a row at a queue that cannot be read before it gives up | `30` |
| `REGBENCH_STAGE_STALL_S`, `REGBENCH_STAGE_STALL_MAX_S` | read by the tile submitter: seconds without a tile task leaving the queue before it lists what is left (and gives up on a held one); and before it gives up whatever the reasons | `3600`, `172800` |
| `REGBENCH_STAGE_HELD_REASONS` | the scheduler's reasons the tile submitter takes for "does not leave the queue by itself", space-separated | `JobHeldAdmin JobHeldUser JobHoldMaxRequeue DependencyNeverSatisfied JobLaunchFailure` |
| `REGBENCH_RUN_ORDER` | the order the runs are SUBMITTED in (and dealt onto the lanes): family names (`stare`, `valis_mirage`, `valis120`, `deeperhistreg`, `ashlar`, `affine_ashlar`, `palom`: every run of the family) or run names, space-separated; the runs not named follow in the run list's order. A run that reads another's results (the two `affine_ashlar` rows read `stare`'s) is never submitted before it: it is put directly after it and a `# run order:` line says so. An unknown name is refused with the list. It changes the order of submission and nothing else: not `runs`, not a table, not a result | the run list's order |
| `REGBENCH_RUN_LANES` | runs in flight at once: the selected runs are dealt onto that many lanes in the order of the run list, and a run starts when the run before it on its lane has ended, whatever became of it; the next dataset goes on down the same lanes. With `REGBENCH_ARRAY_LIMIT` this bounds the tasks running at once (lanes x limit, plus the selfchecks, which are not laned) | `0` (no limit) |
| `REGBENCH_FORCE` | `1`: run every unit again, also those whose results are all `ok` already | `0` (reuse) |
| `REGBENCH_SUBMISSION` | this submission's id. Set it only with `--score-only` (the submission to score) or to run a unit by hand under a submission's id: the launcher refuses an id the folder already holds a submission of | time and process id |
| `REGBENCH_SCORE_WHILE_QUEUED` | `1`: `--score-only` although jobs of an earlier call into the folder are still pending or running | `0` (refused) |
| `REGBENCH_ALLOW_UNCHECKED` | run names that start although their selfcheck RAN and could not decide (`not_checked`); the verdict is still recorded and shown. Nothing else is let through: `inconsistent`, `crashed`, a flagged realcheck and a selfcheck that left no record of this submission block the run all the same | none |
| `REGBENCH_MANIFEST` | with `--prepare`: the cohort's `rounds.csv` | none |
| `REGBENCH_TRUTH_PATIENTS` | with `--prepare`: the patients of the semisynthetic known-truth cases, comma-separated | all |
| `REGBENCH_PREPARE_ARGS`, `REGBENCH_PREPARE_FORCE`, `REGBENCH_PREPARE_MEM`, `REGBENCH_PREPARE_TIME` | with `--prepare`: more `prepare multiplex` options; `1` to redo cases that exist; the allocation | none, `0`, `64G`, `24:00:00` |
| `REGBENCH_NUCLEI` | with `--prepare`: where each slide's bare nuclei are kept ("Outlines and bare nuclei") | `$REGBENCH_CASES/_nuclei` |
| `REGBENCH_STARDIST_MODEL`, `REGBENCH_STARDIST_MODEL_DIR` | with `--prepare`: the StarDist model the arms ran (the pipeline's `segmentation_model` and `segmentation_model_dir`, from the site's config); the folder is bound into the images | the site config's `segmentation_model` and `segmentation_model_dir`; a name is required from one of the two |
| `REGBENCH_SITE_CONFIG` | with `--prepare`: the site config the StarDist model and any override of the StarDist settings are read from when the two variables above are unset; `none` reads no file | `conf/ieo.config` of the checkout, if it is there |
| `REGBENCH_SIF_STARDIST` | the pipeline's StarDist image (`bolt3x/mirage-stardist:1.0.0`), for the re-run only | none |
| `REGBENCH_STARDIST_CPU`, `REGBENCH_STARDIST_PROB_THRESH` | `1`: StarDist without a GPU (else the array asks for `REGBENCH_SBATCH_GPU`, as the arms did); a detection threshold, only if the site's config sets one | `0`, none |
| `REGBENCH_NUCLEI_SKIP`, `REGBENCH_NUCLEI_FALLBACK` | `1`: prepare without the nuclei step; `shrink`: the shrunk outline for a slide still without nuclei | `0`, none |
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
  apptainer pull ashlar.sif         docker://labsyspharm/ashlar:1.20.0
  apptainer pull valis-mirage.sif   docker://cdgatenbee/valis-wsi@sha256:eac27cc599ae0e54aa01c1bef97538301994ce1abd4da44be3f3130ab85a40e6
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
- **Inside `tmux` (or `screen`).** A real submission starts every image once (the
  preflight) and then makes several hundred `sbatch` calls, one after the other; it
  takes minutes. A login session that drops in the middle ends the script with part of
  the graph queued. The ids of what was queued are then in
  `$REGBENCH_OUT/logs/submitted/<id>.ids`, the newest file of that folder (each id is
  written there as soon as its `sbatch` returns): `scancel $(cat <that file>)`, and
  submit again.
- **`REGBENCH_STAGE_QUEUE_MAX` is per run, not for the submission.** Each run whose
  registration is several jobs per round (`stare`, `stare_prep`) has, on each dataset,
  a tile submitter of its own with a ceiling of its own; a run's datasets follow one
  another, so the most tile tasks in the queue at once is the ceiling times the number
  of such runs (two: 200 at the default 100), on top of everything the launcher itself
  queued. The dry run's last lines give the sum to hold against the site's limit.
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

# 1. the cohort's list (standard library only: the login node's python will do).
#    --nuclei-glob / --seg-arm offer label images of bare nuclei that already exist
#    ("Outlines and bare nuclei"); without them every slide is segmented again
python3 -m benchmarks.regbench.arms_manifest --root <arm_results> --out rounds.csv

# 2. a walking skeleton: ONE patient, the cohort's cases only
awk -F, 'NR==1 {print; next} NR==2 {p=$4} $4==p' rounds.csv > one.csv
REGBENCH_MANIFEST=$PWD/one.csv benchmarks/regbench/slurm/submit_cohort.sh --prepare --datasets multiplex -- <site options>
#    (this first runs the bare-nuclei array, one task per slide: it needs
#    REGBENCH_STARDIST_MODEL, REGBENCH_STARDIST_MODEL_DIR, REGBENCH_SIF_STARDIST and the GPU
#    request; REGBENCH_NUCLEI_SKIP=1 leaves it out)
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
#    ok: $REGBENCH_OUT/logs/gates/stare.tsv and ashlar.tsv read "<run> ok - 0 -";
#        every $REGBENCH_OUT/<stare|ashlar>/multiplex/<case>/run.json has "ok": true;
#        every $REGBENCH_OUT/stare/multiplex/<case>/stages/jobs/ holds coarse.json,
#        solve.json, stitch.json and one tile_<row>.json per tile, each "ok": true, and
#        that run.json has info.jobs.execution "jobs", n_tiles = the tile count and
#        register_s = info.jobs.wall_unlimited_parallel_s;
#        $REGBENCH_OUT/tables/summary.md exists and its cost block opens with
#        "cores confined: yes"

# 3b. only if the scoring has to be done again (a score job that ran out of time, a
#     changed scorer): the scoring alone, for the registrations that are there. No
#     selfcheck, gate or registration is submitted; a case whose record is still valid
#     is not computed again
benchmarks/regbench/slurm/submit_cohort.sh --score-only --datasets multiplex -- <site options>

# 4. the whole cohort: prepare (a NEW $REGBENCH_CASES: the known-truth cases of an older
#    truth spec are refused), look at the dry run, submit
REGBENCH_MANIFEST=$PWD/rounds.csv benchmarks/regbench/slurm/submit_cohort.sh --prepare -- <site options>
benchmarks/regbench/slurm/submit_cohort.sh --preflight
benchmarks/regbench/slurm/submit_cohort.sh --dry-run -- <site options> | less
benchmarks/regbench/slurm/submit_cohort.sh -- <site options>
```

`--score-only` scores what the LAST submission into `$REGBENCH_OUT` ran or reused: its
id is kept in `$REGBENCH_OUT/logs/submission`. For a folder filled before that file was
written, the script takes the id its selfcheck records carry when they all carry the
same one, and otherwise asks for `REGBENCH_SUBMISSION=<id>` (the id is printed by every
submission as `# submission  -> ...` and is in every `run.json`, key `submissions`). The
records are parsed, each `logs/selfcheck/<run>/selfcheck.json` by the reader the gates
use (`python -m benchmarks.regbench submissions --out OUT`, `ladder.submissions_of`);
until 2026-10-12 the id was searched for as text. A record that cannot be read, or that
names no submission (a selfcheck run by hand), is evidence of none. When the id being
scored, exported or read from `logs/submission`, is one no selfcheck record carries
while records of other submissions are there, the script says so in a `# WARNING` line
and goes on with the id it was given: a file of another submission is never taken for
it. It submits `rb-settle`, the case arrays, `rb-tissue` and `rb-score`; it reads the
gate records that submission left, and it needs `REGBENCH_SIF_STARE_REGBENCH` and
nothing else.

`rounds.csv`'s fourth column is the patient (`group`); the `awk` line keeps the first
patient's rounds. `--runs` takes any subset of run names; the selfchecks a subset stands
behind are submitted with it. A run whose results another reads but which is not in the
subset is assumed to be under `$REGBENCH_OUT` already, and the script says so.

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
  Whether those two images hold them was not checked from here: the preflight is
  where it shows.
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
| `N sbatch call(s) would be made now` and `M task(s) in them` | everything the launcher submits at once: 445 calls for the default submission on two datasets, and one task per unit and rung for each run, per case for the scoring arrays | array tasks count against the per-user and per-association `MaxSubmitJobs` and the cluster's `MaxJobCount`: `sacctmgr show assoc user=$USER format=Account,MaxSubmitJobs,MaxJobs`, `sacctmgr show qos format=Name,MaxSubmitPU,MaxWall`, `scontrol show config \| grep -E "MaxArraySize\|MaxJobCount"` |
| `T tile task(s)` fed in later | one per tile, never in the queue together | none: they pass through the ceiling |
| `the most in the queue at once` | M plus `REGBENCH_STAGE_QUEUE_MAX` for each run with a tile submitter | the same `MaxSubmitJobs`. Lower it with `REGBENCH_STAGE_QUEUE_MAX`, `--runs` or `--datasets` |

Besides that:

- **A compute node may call `sbatch` and `squeue`** (this site's Nextflow head jobs do).
  Check: `srun -c 1 --mem=1G --time=5 bash -c 'sbatch --version && squeue -h -u $USER | head -n 3'`.
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
anything is touched: it would remove the gate records those jobs' tasks read. The
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
  `afterany` on every array and every gate; if a cancelled job did not release it, the
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
  <uid>`). Check on the site: what `sacct -n -X -P -j <array job>_<task> -o
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

## HyReCo: an external landmark set (its own tables)

HyReCo is the one public dataset found in which the SAME section was imaged twice and
annotated by hand: H&E, bleached, re-stained PHH3, scanned again, with landmarks placed
on nuclei. It is brightfield RGB, human tongue, one lab, about 95,000 x 220,000 px per
slide at 0.24 um. Dataset: van der Laak, Lotz, Weiss, Heldmann, IEEE DataPort 2021,
doi:10.21227/pzj5-bs61 `[FULL: the dataset page's text, fetched 2026-10-10; no data file
was ever opened]`. Paper: Lotz et al. 2023, J Med Imaging 10(6):067501,
doi:10.1117/1.JMI.10.6.067501 `[PARTIAL: the Europe PMC text searched for the pixel size,
the landmarks, the grey-scale conversion, MTRE and the observer errors, and those
paragraphs read; figures and supplement not seen]`. Treated as CC BY-SA 4.0 (page and
paper; the DataCite record of the same DOI says CC BY 4.0). ANHIR, whose sets are made
of different (consecutive) sections, is a separate external set with its own section
below.

**It is never part of the cohort's results.** `score` without `--dataset` reads the
cohort's datasets only; `score --dataset hyreco_b` writes `<out>/tables/hyreco_b/` and
nothing else; the two kinds cannot be named in one call, in `score` or in the launcher.
No HyReCo number is in the cohort's tables, in a paired comparison with them or in a
ranking.

### The rule: tuned on subset A, reported on subset B

| dataset | what | role |
|---|---|---|
| `hyreco_a` | subset A's nine re-stained H&E / PHH3 pairs (ids 29, 108, 361, 464, 533, 611, 628, 644, 679) | the TUNING set: every setting is chosen here. Its tables say "nothing on this page is a result" |
| `hyreco_b` | subset B ("HyReCo-Additional"): 54 re-stained pairs | the REPORT set: nothing is chosen on it |

What can be chosen is what `prepare hyreco --subset a` takes: `--proxy` (the one-channel
image STARE is given: `luminance`, the pre-declared default, or `haematoxylin`) and
`--reduce` (1, or 2 for a 2 x 2 block average). A method's own options are fixed by the
run list, as for the cohort. The split is enforced, not only stated:

- A folder is accepted for a subset only when its case ids are that subset's (the two
  lists of the dataset page, in `datasets/hyreco.py`). Subset B's folder cannot be
  prepared as the tuning set, and subset A's cannot be reported.
- `prepare hyreco --subset b` has no setting of its own: `--proxy` and `--reduce` are
  refused there. It copies the settings of the prepared `hyreco_a` cases
  (`--settings-from <cases folder>`), which must be all nine pairs under one setting.
- Subset B cannot be prepared a second time under other settings (the cases folder is
  refused until `hyreco_b` and its results are removed), and `score --dataset hyreco_b`
  refuses cases of two settings, cases that name no tuning set, and a tuning set that
  no longer carries the settings the report set was prepared with.
- To try a second setting on subset A, prepare it into a cases folder of its own and
  score it into an output folder of its own; then point subset B at the one chosen.

**The four shared ids.** 29, 108, 361 and 464 are in both lists; no source read says
whether they are the same slides. They are prepared in `hyreco_b`, left OUT of its
report set (50 pairs), and kept in the pooled line over all 54 pairs, which is the
published definition's set. `--keep-shared-ids` puts them back, for use only once the
files have shown they are different slides.

### The files, and what is assumed about them

```
HyReCo/{CD45,CD8,HE,KI67,PHH3}/<id>.tif|.csv    subset A   HyReCo.zip                    233 GB
HyReCo-Additional/HE/<id>.tif|.csv              subset B   HyReCo-Additional-HE.zip      274 GB
HyReCo-Additional/PHH3/<id>.tif|.csv            subset B   HyReCo-Additional-PHH3.zip    233 GB
```

Verified on the dataset page: the tree above, the two id lists, BigTIFF plus one CSV per
image, and "Each line describes a point annotation in world coordinates (x,y,z). The
unit is mm. The origin of the coordinate system is the upper left corner of the image."
Verified in the paper: 0.24 um per pixel. Everything else is an assumption, listed in
the module's docstring; each one is checked on the files and REFUSES the case when it
does not hold:

| assumed | what happens when it does not hold |
|---|---|
| the first TIFF series' first level is the full-resolution image, RGB, 8 bit | the case is refused |
| landmarks are paired by line (the page implies it, does not say it) | the two files of a pair must have the same number of points, else the case is refused (never truncated). A line that is not `x,y[,z]` refuses the case wherever it is, except one header line; so does a blank line between points, and a file with more than three columns (it would have to be paired by an id) |
| pixel = mm / pixel size, no half-pixel term | not checkable; the same convention on both slides moves both landmarks of a pair alike |
| the pixel size is the FILE's (resolution tags in inch or cm, or `MPP = ` in the description) and agrees with the paper's 0.24 within 2 % | refused when it disagrees. A file that states none falls back to 0.24 exactly, and the summary says so in bold: 1 % of pixel size moves a landmark 50 mm from the origin by 2000 px. `--pixel-size-um` states a measured value, under the same check |
| the two slides of a pair have one pixel size (within 0.1 %) | the case is refused |
| every landmark lies inside its image | the case is refused: the pixel size or the origin is wrong |
| only `HE/` (reference: scanned first) and `PHH3/` (moving) are a pair | any other folder name is refused; subset A's consecutive sections are not part of this module |

A refused case is printed as `REFUSED` with the reason, the other cases are prepared,
and `prepare` exits 1. **Look at the first prepared case before any array is
submitted:**

```bash
python -m benchmarks.regbench.datasets.hyreco --cases $REGBENCH_CASES \
    --dataset hyreco_a --case-id 29 --out look/
```

writes, for the first eight landmarks, the crop of each slide around its landmark, side
by side (H&E left, PHH3 right) with a cross on it. The two crosses must sit on the same
nucleus. If they do not, an assumption above is wrong and no number means anything.

### Which methods run on it, and what each is given

It is brightfield, and the rule of this benchmark holds: a method is fed as its package
intends. `python -m benchmarks.regbench runs --dataset hyreco_b` prints the runs;
`--excluded` prints the rest with their reasons (`methods.NO_BRIGHTFIELD` for a method
without a brightfield rule, `variants.NOT_ON` for a run this set does not take; also in
`hyreco_excluded.csv`). An adapter without a rule REFUSES a brightfield case in the same
words, before it imports its package, so no method can run on a wrong channel by hand.

| run | on HyReCo | what it is given |
|---|---|---|
| `stare` | yes | a ONE-CHANNEL PROXY of each slide, written by `prepare` (benchmark-side preprocessing, recorded in `run.json: info.input` and `benchmark_side`): `stare register` reads one channel per slide. Default: inverted luminance, `255 - (0.299 R + 0.587 G + 0.114 B)`, which is what the dataset's authors feed their own method ("converted from color to gray scale and inverted to obtain a black background", Lotz et al. 2023 `[PARTIAL]`) and what STARE's archived harness did (`benchmarks/regbench/datasets/hyreco.py` at the tag `archive/benchmarking-2026-10-08` of sceriff0/stare, local copy `[FULL: that file]`). On a reduced proxy STARE works in the reduced frame and the adapter scales its map; the image it writes is the registered proxy |
| `valis120`, `valis120_hires` | listed, but as built they FAIL on every pair (both slides of a pair have the same basename and the adapter refuses that: "Dormant datasets", item 1); never executed | the ORIGINAL RGB slides, as for the cohort. VALIS decides how to process a slide from the slide itself; its default for brightfield is `preprocessing.OD` (valis-wsi 1.2.0 `registration.py:63-64` `[PARTIAL: lines 63-66, 2679-2682, 4593-4596]`). The adapter adds nothing. Whether VALIS's reader opens these BigTIFFs is not known |
| `valis_mirage` | no | it is the pipeline's own VALIS call, and the pipeline registers multichannel fluorescence rounds only: there is no shipped brightfield configuration of it |
| `stare_prep` | no | the preprocessing row is defined for ANHIR, the cohort and the synthetic suite (R3.21); HyReCo is dormant and keeps its own proxy settings (`variants.NOT_ON`) |
| `deeperhistreg_*` (8 runs) | no | the package IS built for brightfield RGB (RegWSI reports a HyReCo result, arXiv:2404.13108, per the port's HyReCo report `[PARTIAL]`; not re-read for this section). The adapter's RGB input rule was written for ANHIR's JPEG / PNG images (R3.20) and has not been extended to these BigTIFF slides: the adapter refuses a brightfield TIFF, and the runs are left out here by name (`variants.NOT_ON`) |
| `ashlar`, `ashlar_clean`, `affine_ashlar`, `affine_ashlar_clean` | no | ruled out (R3.8). ASHLAR "requires unstitched individual tile images as input" (its README at v1.20.0, as quoted in `methods/ashlar.py` by the task that read it `[PARTIAL: lines 1-60]`; not re-read for this section) and places each tile by one translation; a pair of whole slides re-mounted between two scans is not that |
| `palom`, `palom_b1024` | no, not yet | PALOM does apply to a pair of whole slides and has colour modes of its own for an RGB slide (`palom/color.py`, `HaxProcessor`: `grayscale`, `hematoxylin`, `aec`, `dab`; palom 2026.4.1 sdist `[PARTIAL: lines 1-40]`; its `palom-align-he` command is documented for an IF reference and an H&E moving slide, per the port's PALOM report), but this adapter follows the README recipe on one channel BY INDEX: on an RGB slide that would be the red channel, chosen by nobody. Which colour mode is the input rule is undecided |

So three runs, eight result rows: `stare_rigid`, `stare`, `valis120_rigid`, `valis120`,
`valis120_micro`, `valis120_hires_rigid`, `valis120_hires_nonrigid`, `valis120_hires`.
The selfcheck gate applies as for the cohort, but it is made on a synthetic
FLUORESCENCE pair: it says nothing about reading RGB. The realcheck gate and the
tissue-level block are not applied (both are defined on nuclear channels and nuclei,
`run.json: tissue.skipped`), and there are no off-seam rows (no nuclei, and neither
seam owner runs).

### Scoring and tables (`<out>/tables/<dataset>/`)

Target registration error in micrometres: a method's map moves the landmarks of the side
it starts from (PHH3 for a moving -> reference map, H&E for a reference -> moving one)
and each is compared with its partner, through the same code as the known-truth error
(`truth.error_stats`). A point a method could not map has an infinite error; a run that
failed or never ran is scored at the unregistered pose and counted (`n_ok`).

| table | one row per | columns |
|---|---|---|
| `hyreco_pairs.csv` | pair and variant | `dataset, case_id, group, report_set, shared_id_with_subset_a, method, frame, n_probes, n_nonfinite, tre_um_p50, tre_um_p90, tre_um_max, frac_under_1um, frac_under_2um, initial_tre_um_p50, frac_improved, fold_pct, sd_log_jac, pixel_size_um, pixel_size_source, imputed_initial, problem, cause` |
| `hyreco_points.csv` | landmark and variant | `..., landmark, tre_um, initial_tre_um, imputed_initial` |
| `hyreco_summary.csv` | set and variant | `dataset, set, method, n_pairs, n_ok, n_landmarks, per_pair_median_tre_um_med / _min / _max, per_pair_p90_tre_um_med, pooled_median_tre_um, pooled_p90_tre_um, pooled_frac_under_1um, pooled_frac_improved, initial_pooled_median_tre_um, n_nonfinite, fold_pct_max` |
| `hyreco_published.csv` | quoted number | `dataset, set, method, n_pairs, n_landmarks, pooled_median_tre_um, status, conditions, source` |
| `hyreco_excluded.csv` | run that is not run | `dataset, run, reason` |
| `hyreco_failures.csv`, `hyreco_resources.csv`, `hyreco_summary_resources.csv` | as the cohort's `failures`, `resources`, `summary_resources` | the same columns |
| `summary.md` | | the block below |

**Two aggregations, never mixed.** `per_pair_*` is each pair's median error, then the
median, minimum and maximum over pairs. `pooled_*` is ONE median over all landmarks of
all pairs of the set, which weights a pair by its landmark count. Sets (`set`):
`tuning` for `hyreco_a`; for `hyreco_b`, `report` (50 pairs) and `all_pairs` (all 54).

**The one published number, like for like.** Lotz et al. define "MTRE" as the median
"over all N_images image pairs and over all available landmarks". Their own method,
HistokatFusion (Fraunhofer MEVIS), reaches **0.9 um** deformable and 1.7 um affine on
subset B (54 pairs, 2357 landmark pairs; best results at a pixel size <= 1.9 um;
parameters set on ANHIR, not tuned on HyReCo), and 2.2 um on subset A's nine re-stained
pairs at 1.0 um/px (`[PARTIAL]`, as above). The summary's "Like for like" table sets
`pooled_median_tre_um` of every run made here on `all_pairs` beside those rows, which
are marked **published, not reproduced here**: the software is licensed, has no public
code, and was not run. When fewer than 54 pairs are scored the table says "NOT like for
like". The landmark count found in the files is printed next to it (the page and the
preprint say 2303, the paper 2357). CORE's 0.41 um is not in the table: its
aggregation ("AMTRE") is not defined and its subset is not stated. Neither CORE nor
HistokatFusion is run. RegWSI's 0.59 um and CORE's VALIS 2.50 um are not quoted either
(subset or configuration unstated).

**The annotation floor.** Two observers differ by 0.57 +- 0.36 um on the same section
(intra-observer 0.53 +- 0.32 um; about 20 landmarks; Lotz et al. 2023 `[PARTIAL]`). It
is printed above every table. A difference between two methods below about 0.5 um is
annotation noise; no ranking is claimed.

**What these numbers are not.** Not fluorescence accuracy (brightfield, and for STARE
through a proxy); not a blind test (the landmarks are public: the frozen split is what
makes it credible); not a reproduction of any published number (resolution, parameters
and, for STARE, the input differ); nothing about low-contrast tissue (the authors note
landmarks sit where contrast is high).

### Running it, once the data is downloaded

Download the three archives from the dataset page in a browser (it needs an IEEE
login; nothing in this repository downloads them) and unpack them so the tree above
exists. Subset B is two archives: its `HE/` and `PHH3/` folders are given separately.

```bash
S=benchmarks/regbench/slurm
export REGBENCH_BIND=/data/hyreco                       # wherever the slides are
export REGBENCH_CASES=/scratch/hyreco_cases
export REGBENCH_SIF_STARE_REGBENCH=... REGBENCH_SIF_VALIS120=...

# 1. the tuning set
export REGBENCH_HYRECO_A=/data/hyreco/HyReCo            # holds HE/ and PHH3/
export REGBENCH_OUT=/scratch/hyreco_out_a
$S/submit_cohort.sh --prepare --datasets hyreco_a       # 1 job: an array of 9, one pair each
python -m benchmarks.regbench list --cases $REGBENCH_CASES --dataset hyreco_a   # 9
python -m benchmarks.regbench.datasets.hyreco --cases $REGBENCH_CASES \
    --dataset hyreco_a --case-id 29 --out look/         # and LOOK at the crops
$S/submit_cohort.sh --dry-run --datasets hyreco_a       # 25 jobs, nothing submitted
$S/submit_cohort.sh --datasets hyreco_a
#    -> $REGBENCH_OUT/tables/hyreco_a/summary.md
#    another setting: a cases folder and an output folder of its own, e.g.
#    REGBENCH_CASES=/scratch/hyreco_cases_r2 REGBENCH_OUT=/scratch/hyreco_out_a_r2 \
#    REGBENCH_HYRECO_A_ARGS="--reduce 2" $S/submit_cohort.sh --prepare --datasets hyreco_a

# 2. the report set, once, with the cases folder of the setting that was chosen
export REGBENCH_CASES=/scratch/hyreco_cases             # the chosen tuning cases are here
export REGBENCH_HYRECO_B_HE=/data/hyreco/HyReCo-Additional/HE
export REGBENCH_HYRECO_B_PHH3=/data/hyreco/HyReCo-Additional/PHH3
export REGBENCH_OUT=/scratch/hyreco_out_b
$S/submit_cohort.sh --prepare --datasets hyreco_b       # 1 job: an array of 54
$S/submit_cohort.sh --dry-run --datasets hyreco_b
$S/submit_cohort.sh --datasets hyreco_b
#    -> $REGBENCH_OUT/tables/hyreco_b/summary.md
```

`REGBENCH_HYRECO_TUNED` names the chosen tuning cases when they are not under
`REGBENCH_CASES`. By hand, without a scheduler: `python -m benchmarks.regbench prepare
hyreco --subset a --data-root ... --cases ...`, `run --dataset hyreco_a --method stare
--index 0`, `score --dataset hyreco_a`.

What `ok` looks like. Each prepare task prints `hyreco_a/<id> -> <folder>` and
`hyreco_a (tuning set): 1/1 case(s) prepared ...`, and exits 0; a `REFUSED` line is a
failed task with its reason. `list` prints 9 (54). The dry run prints `# hyreco_b: 54
case(s), 54 patient(s), at most 1 round(s) per patient`, one `# hyreco_b: <run> is not
run (<reason>)` line per excluded run (15), and 25 `sbatch` lines: per run three
selfcheck attempts, one gate and the four rungs of the ladder (16 / 64 / 256 / 512 GB,
4 CPUs, an array of 54), then `rb-score-hyreco_b`. The summary then has a row per run,
an `initial` row far above the runs that registered, and for `hyreco_b` the "Like for
like" table. As built, only `stare` can have `n_ok` equal to `n_pairs`: both VALIS rows
fail on every pair and are scored at the initial pose ("Dormant datasets", item 1), so
the table is not a comparison until that is fixed.

### Storage and memory to expect (none of it measured)

- **Download**: 233 GB (subset A), 506 GB (subset B), no checksums published. Unpacked
  size unknown; if the TIFFs are already compressed, about the same again. Budget
  0.5 TB for A and 1.0 to 1.5 TB for B, or delete each archive once unpacked.
- **Proxies** (`<cases>/_cache/hyreco_*/`): one 8-bit plane per slide, about 21 GB
  uncompressed at 95,000 x 220,000 px, written tiled and zlib-compressed. 18 slides for
  A, 108 for B: at most 0.4 TB and 2.3 TB uncompressed; how far compression takes that
  down depends on how much of a slide is glass. A quarter of it with `--reduce 2`;
  twice with `--proxy haematoxylin` (16 bit).
- **Written images** are the larger risk. VALIS writes a registered full-resolution
  slide per pair for `valis120`, `valis120_micro`, `valis120_hires_nonrigid` and
  `valis120_hires` (about 63 GB each before compression); STARE one registered proxy
  per pair. Size `REGBENCH_OUT` from the tuning set before submitting 54 pairs.
- **Preparing**: one band of 1024 rows in memory (about 3 GB at 220,000 px wide); the
  prepare job's 64 GB is ample. Time per slide unknown.
- **Registering**: a slide is about 21 gigapixels, seven times a 55,000 px cohort
  slide. STARE at stride 64 measures about 5 million windows per pair at full
  resolution. Expect units to leave the 16 GB rung; the ladder takes each up one rung
  per out-of-memory end, and each failed attempt costs its time. The dataset's authors
  report more than 180 GB for their own method below about 1 um/px (`[PARTIAL]`), which
  says nothing about these methods. `REGBENCH_TIME` (24 h per unit) may be too short.
  Run two or three pairs of the tuning set by hand first (`run --index`), and decide
  `--reduce` from them.

## ANHIR: the challenge's training pairs (its own tables)

ANHIR (Automatic Non-rigid Histological Image Registration, ISBI 2019,
https://anhir.grand-challenge.org) registers differently stained CONSECUTIVE sections:
481 pairs in 49 tissue sets (lung lesions and lobes, mammary glands, mouse and human
kidney, colon adenocarcinoma, gastric, breast), at about 1 to 9 um per pixel in its
"medium" archive. The 230 pairs marked `training` in its cover table have public
landmarks on both images; the 251 `evaluation` pairs do not, and are not part of this
benchmark. Sources: the challenge's Data page and Evaluation page `[FULL: saved copies of
both, fetched 2026-10-10]`; its evaluation code, Borda/BIRL at `f1648b29` `[PARTIAL: the
functions that compute TRE, robustness and the aggregate scores; listed in
challenge.py]`; the challenge paper, Borovec et al. 2020, IEEE TMI 39(10),
doi:10.1109/TMI.2020.2986331 `[MEMORY: not read for this section; nothing here rests on
it but the citation]`. Licence CC BY-NC-SA.

**What these numbers are, and are not.**

- They are computed on TRAINING pairs. Published leaderboard numbers are on the hidden
  evaluation pairs (with a training column beside them) and are **not comparable row for
  row** with anything here. Nothing is exported for submission and no rank among
  published methods is claimed.
- **The `stare_prep` row's preprocessing is tuned on part of this same training set.**
  Its tuning-side numbers are fitted; its report-side numbers come from tissue sets the
  choice never saw, but from the same challenge's public data. Neither side is blind.
- STARE is built for re-imaged sections that hold the same nuclei in both rounds.
  Consecutive sections do not, and at these pixel sizes a nucleus is a few pixels: both
  STARE rows are an out-of-domain test here.
- It is never part of the cohort's results and never part of the default submission: it
  runs when named, alone (`--datasets anhir`), into `<out>/tables/anhir/`.

### The cases, and the split

`prepare anhir --source DIR` (`DIR` holds the cover table `dataset_medium.csv`, the
unpacked image archive under `images/` and the unpacked landmark archive under
`landmarks/`; it is only read) writes two datasets:

- `anhir`: the 230 training pairs. The cover table's `Source image` is the moving image,
  its `Target image` the reference; landmarks are paired by row, the common prefix
  where the two files differ in length (the official rule). Each pair is its own group:
  a set has several targets, so VALIS registers pair by pair with the target fixed,
  which is not how a groupwise leaderboard entry is made.
- `anhir_tune`: the tuning-side pairs again, alone.

**The split is by tissue SET, never by pair** (pairs of one set share images, and most
sets also hold hidden evaluation pairs), and nobody picked it: inside each tissue the
sets are ordered by `sha256("regbench-anhir-split-v1:" + set name)`, the first
`ceil(n / 4)` are the tuning side, and a tissue with one set is wholly on the report
side. It is frozen in `anhir_split.json` (sha256 `c0fb810e29f2...`): **15 sets and 82
pairs on the tuning side, 34 sets and 148 pairs on the report side.**

| tissue | tuning side | pairs (tune / report) |
|---|---|---|
| COAD | `COAD_02`, `_08`, `_10`, `_11`, `_14` | 25 / 59 |
| breast | `breast_3`, `_4` | 2 / 3 |
| gastric | `gastric_2`, `_4`, `_5` | 5 / 8 |
| kidney | `kidney_1`, `_5` | 2 / 3 |
| lung-lesion | `lung-lesion_3` | 10 / 20 |
| lung-lobes | `lung-lobes_2` | 10 / 30 |
| mammary-gland | `mammary-gland_2` | 28 / 10 |
| mice-kidney | none (one set) | 0 / 15 |

`prepare` recomputes the split from the cover table and refuses a table that gives
another; the scorer refuses cases of another split. **Headline numbers are the report
side's.** The tuning side and both sides together are shown beside it, labelled.

### Pixel size, per tissue

The Data page gives one resolution per tissue, of the ORIGINAL image; the archive holds
each set at the percentage its folder names. Micrometres per pixel are that resolution
divided by the folder's fraction: derived arithmetic, recorded as such in every case.

| tissue | original, um/px (Data page) | folder | um/px used | status |
|---|---|---|---|---|
| lung-lesion | 0.174 | 25 % / 50 % | 0.696 / 0.348 | derived |
| lung-lobes | 1.274 | 100 % | 1.274 | as stated |
| mice-kidney | 0.227 | 25 % | 0.908 | derived |
| COAD | 0.468 | 25 % | 1.872 | derived |
| gastric | 0.2528 | 15 % | 1.685 | derived |
| breast | 0.2528 | 20 % | 1.264 | derived |
| kidney | 0.2528 | 25 % | 1.011 | derived |
| mammary-gland | 2.294 | 25 % | **none** | **NOT established: pixels only** |

For the mammary glands the page states 2.294 um/pixel and, in the same paragraph, that
the images "were acquired with the same microscope and set of acquisition parameters as
the mice lung lobes" (1.274 um/pixel, the same 10x objective). Both cannot hold, and no
other official source read settles it. Those 38 pairs are in every rTRE and pixel
column and in no micrometre column (`n_pairs_um` says how many pairs a micrometre
number rests on).

### Which runs, and what each is given

`python -m benchmarks.regbench runs --dataset anhir` prints them; `--excluded` the rest
with reasons (also in `anhir_excluded.csv`). Eight runs, 18 result names.

| run | on ANHIR | what it is given |
|---|---|---|
| `stare` | yes | each image as ONE channel, written by `prepare`: **inverted luminance, `255 - (0.299 R + 0.587 G + 0.114 B)`, 8 bit, full resolution** (`prep.grey_plane`). `stare register` reads one channel per slide, so this conversion is part of the benchmark's input, not STARE's. The image it writes is that grey image, registered. For the mammary glands STITCH's header gets a placeholder pixel size of 1.0 (STARE uses the pixel size for nothing else) |
| `stare_prep` | yes | the frozen candidate of `prep.py` instead of that grey image; STARE unchanged; the image it writes is the same plain grey image moved by its own transform |
| `valis120`, `valis120_hires` | yes | the ORIGINAL RGB images (JPEG; PNG for the lung lobes), as for the cohort: upstream defaults, the target as the reference. VALIS picks its own brightfield processing. Never executed here; whether its reader opens these files as the adapter places them is by reading only |
| `deeperhistreg_gpu`, `_cpu`, `_hr_gpu`, `_hr_cpu` | yes | the RGB image WHOLE, with the shipped parameter set as it stands (see below) |
| `deeperhistreg_*_fluo_*` (4 runs) | no | `fluo=1` is the benchmark's configuration for a fluorescence channel (a black pad, no intensity flip); on RGB the shipped parameters are the ones to run |
| `valis_mirage` | no | the pipeline's own VALIS call registers multichannel fluorescence rounds only |
| `ashlar` (4 runs) | no | ASHLAR stitches unstitched tiles by one translation each; two whole images of two sections are not its input |
| `palom`, `palom_b1024` | no | the adapter follows PALOM's README recipe on one channel by index (on RGB: the red channel, chosen by nobody); which of PALOM's own colour modes would be the input rule is undecided, and its README describes it as made for re-imaged slides of the same section |

**DeeperHistReg's brightfield input rule**, from its own repository (MWod/DeeperHistReg
main @ `42e7c9d` `[PARTIAL: README.md:1-128, registration_params.py:484-520 and 652-690,
tiff_loader.py:20-120, general_preprocessing.py:1-66, dhr_utils/utils.py:25-37 and
374-375, input_output.py; never executed]`): the package takes the source and the target
image by path; the shipped sets load through `'tiff'` (`pyvips.Image.tiffload`), pad
with 255, normalise each channel to its own range, convert to grey, invert and apply
CLAHE. So the adapter hands over the RGB image, three 8-bit channels, values untouched,
and changes no parameter beyond the two every row changes (device, `save_final_images`).
Its one benchmark-side step is the container: each JPEG / PNG is decoded and written as
a tiled, uncompressed RGB TIFF with one top-level page (clock paused, `convert_s`),
because the shipped loader reads TIFF; switching the loader to the package's `'pil'`
one would have been a changed parameter. The package's writer then draws the RGB image
once and its own file is kept.

The selfcheck gate applies as for the cohort, on a synthetic FLUORESCENCE pair: it says
nothing about reading RGB. No realcheck gate, no tissue-level block, no off-seam rows
(all three are defined on nuclei).

### The `stare_prep` preprocessing: candidates, criterion, what is frozen

STARE is not changed. Before it runs, both images are rewritten as one-channel files by
one candidate; STARE registers those; the transform is applied to the original.

| candidate | on an RGB image | on one channel (DAPI, synthetic) |
|---|---|---|
| `grey` | inverted luminance, nothing else (= the `stare` row) | identity (= the `stare` row) |
| `haem` | the haematoxylin channel of a colour deconvolution (`skimage.color.rgb2hed`, channel 0, clipped to [0, 0.255], 8 bit) | identity: there is no colour to separate |
| `grey_clahe` | CLAHE on `grey` | CLAHE on the channel |
| `haem_clahe` | CLAHE on `haem` | CLAHE on the channel (the same as `grey_clahe`) |
| `grey_smooth` | a Gaussian, sigma 2 px, on `grey` | the same Gaussian on the channel |

CLAHE is OpenCV's, clip limit 2.0, contextual regions of 256 px, on 8 bits; a 16-bit
channel is first mapped linearly to 0..255 between 0 and its own 99.9th percentile, so on
the cohort the CLAHE candidates also reduce DAPI to 8 bits (stated, not hidden). The
list is closed and was declared before any run. Why these five: `grey` so the choice
can end on "nothing helps"; `haem` because the counterstain is the one signal two stains
may share (whether every stain of these sets has one is not established by any source
read); CLAHE to even out a pale and a dark section; `grey_smooth` to drop detail that
consecutive sections do not share. No candidate resamples, crops or shifts:
`prep.frame_check` measures, before every case, how far each candidate moves drawn dots
(gate 0.02 px) and the run is refused otherwise, and the output's shape must be the
input's. The selfcheck of the row compares, as for `stare`, the points with the image
STARE's writer draws.

**Criterion** (`prep.CRITERION`), on the tuning-side pairs only: the lowest median over
pairs of the per-pair median rTRE, a pair that did not register counting at its
unregistered landmarks; ties by the higher mean robustness, then by the order of the
list. One candidate for all tissues.

**Frozen today: `grey_clahe`, PROVISIONAL.** The selection has not been run. That value
was chosen by no evidence, only because it is not the identity on either modality, so
the row can be told from `stare` everywhere. To run the selection:

```bash
S=benchmarks/regbench/slurm; export REGBENCH_ANHIR=/data/anhir   # dataset_medium.csv, images/, landmarks/
#    REGBENCH_ANHIR stays exported for EVERY step below: the cases name the original
#    images inside it, and it is bound into every image from that variable (a run that
#    cannot read them is refused at its start, cause `infrastructure`, not the method's)
export REGBENCH_CASES=/scratch/anhir_cases REGBENCH_OUT=/scratch/anhir_out   # of their own
$S/submit_cohort.sh --prepare --datasets anhir      # 230 tasks; writes anhir AND anhir_tune
$S/submit_cohort.sh --dry-run --datasets anhir_tune && $S/submit_cohort.sh --datasets anhir_tune
#    the five candidates on the 82 tuning pairs, each behind its own selfcheck
#    -> $REGBENCH_OUT/tables/anhir_tune/summary.md (nothing in it is a result)
python -m benchmarks.regbench prep-select --cases "$REGBENCH_CASES" --out "$REGBENCH_OUT"            # the table
python -m benchmarks.regbench prep-select --cases "$REGBENCH_CASES" --out "$REGBENCH_OUT" --freeze   # writes prep_frozen.json
#    put benchmarks/regbench/prep_frozen.json under version control: the choice, with its evidence table
$S/submit_cohort.sh --datasets anhir                # every run ANHIR takes, on all 230 pairs
#    -> $REGBENCH_OUT/tables/anhir/summary.md
```

`prep-select` refuses to choose while any candidate lacks a result on any tuning pair,
or while the prepared tuning side is not the whole of the split's. The run list names
the candidate itself (`prep=<candidate>`), and a reused result must have the same
options: a `stare_prep` result made with another candidate, on the cohort too, is run
again, never reused. A cohort submitted BEFORE the selection carries `stare_prep` rows
made with the provisional candidate (marked in `run.json: info.prep.frozen` and in the
summary); resubmitting after the freeze re-runs that one run.

A pilot on this laptop (two tuning pairs of one set, `mammary-gland_2`; all five
candidates through the installed STARE 1.2.1, about a minute each) ran end to end and
is the proof that the machinery works, nothing more: two pairs choose nothing.

### Scores and tables (`<out>/tables/anhir/`)

Per pair, by the challenge's definitions (`challenge.py` cites the page and the code):
TRE is the distance between a landmark's target position and its position after
registration; **rTRE** is TRE over the diagonal of the target image; **robustness** is
the share of a pair's landmarks whose TRE is strictly below their initial TRE. Per group,
the official aggregates: the mean and the median over pairs of the per-pair median, mean
and maximum rTRE and of robustness, and the same over the "robust" pairs (robustness
above 0.5). Beside them, the landmark error in micrometres where the pixel size is
established, and in pixels everywhere.

**Failures, the official way and apart.** "All missing or incomplete registrations in
the submission are considered to have the initial rTRE": a pair the METHOD did not
register counts at its unregistered landmarks with robustness 0 in every official
column, and a landmark returned as non-finite counts at its unregistered position.
`n_failed`, `n_nonfinite`, the `*_registered` columns (the same statistic over the pairs
that did register) and `anhir_failures.csv` show them separately. In the micrometre
column a pair that lost most of its landmarks has an infinite median and COUNTS (it is
the worst there is); only a pair without a pixel size is left out of it.

**Not run is not failed.** A pair that was never registered for a reason that is not the
method's is in NO median: its run was blocked by its selfcheck gate (the scorer reads
the launcher's gates, as the cohort's does), its task never started, the attempt was
lost to the scheduler or to the benchmark's own step, or its images could not be read
where the run was (`infrastructure`: the source folder not bound into the image). Such
pairs are counted (`n_not_run`), listed by cause in `anhir_not_run.csv` and in a block
of `summary.md`; a row with no other pair is in no table at all, so a blocked run can
never read as "fails on every pair". This is a departure from the challenge's rule for
those pairs only, and it is why `n_pairs` can be below the side's pair count.

**Frame.** STARE maps moving to reference: its numbers are the challenge's own quantity.
VALIS and DeeperHistReg map reference to moving: the target landmarks are moved into the
source image and compared there, exactly and with no inverted transform, but in the
source image's pixels (column `frame`; the summary names those rows). A pair that did
not register has no map of its own: its row takes the frame of the same result name's
registered pairs, else the direction its run record declares, else `unknown`; never a
default, so a failed pair of a reference -> moving method is not called the challenge's
quantity.

Not computed: the challenge's ranking criterion (a method's average RANK by median rTRE
among all submitted methods), which needs the other methods' results on the same pairs.

| file | content |
|---|---|
| `summary.md` | the report side first, then the tuning side, then both; what these numbers are not; the frozen preprocessing and whether it is provisional |
| `anhir_pairs.csv` | one row per pair and result name: `rtre_median/mean/max`, `robustness`, `tre_um_p50/p90`, `tre_px_p50`, `frame`, `failed`, `cause` |
| `anhir_summary.csv` | one row per side (`report`, `tune`, `all`), tissue (each, and `all`) and result name: the official aggregates, the robust subset, the registered-only columns, micrometres |
| `anhir_by_set.csv` | the same per tissue SET, with the set's side |
| `anhir_failures.csv`, `anhir_resources.csv`, `anhir_summary_resources.csv` | failures and cost, as for the cohort |
| `anhir_excluded.csv` | the runs left out, each with its reason |
| `anhir_not_run.csv` | per result name and cause: the pairs never registered for a reason that is not the method's (in no median) |
| `anhir_prep_evidence.csv` | `anhir_tune` only: the candidates in the criterion's order |

### Not verified (nothing of this ran outside this laptop)

- **No ANHIR pair was registered by VALIS or DeeperHistReg** (neither is installed
  here). By reading only: that VALIS 1.2.0 reads the JPEG / PNG files as placed; that
  `pyvips.Image.tiffload` reads the adapter's RGB TIFF as three bands; that the package's
  writer writes a three-channel file of the reference's size; every DeeperHistReg
  number on RGB.
- **Only 57 of the 245 images are unpacked on the laptop** (22 of the 49 sets). The
  cover table, all 160 landmark files and the split were checked in full; the prepare
  of the other 172 pairs was not.
- **The selection.** Five candidates on 82 pairs have not run. Time and memory per pair
  on the cluster are unmeasured; the laptop's pairs (20 megapixels) took about a minute
  with 4 workers, and the largest images are 258 megapixels.
- **CLAHE on a cohort slide** (about 2 gigapixels, 16 bit): the block-wise code is
  tested for independence of the block size on small images only; its time and memory
  on a whole slide are unmeasured (it runs with the clock paused, `info.prep.prep_s`).
- The mammary glands' pixel size.

## Dormant datasets: known defects before any use

`semisynth`, `hyreco_a` and `hyreco_b` are DORMANT (ruling R3.14, 2026-10-11): the code
and its tests are kept, the default submission never includes them, and
`submit_cohort.sh --datasets <name>` is the only way to submit one. The independent
review of 2026-10-11 found the defects below. **None of them is fixed.** Each must be
fixed, or ruled acceptable by the owner, before a number from that dataset is read. Where
a sentence elsewhere in this README disagrees with this section, this section is right.

### `hyreco_a`, `hyreco_b` (review items 1 and 10)

1. **VALIS fails on every HyReCo pair, and is then scored as unregistered.** A HyReCo
   pair is `HE/<id>.tif` and `PHH3/<id>.tif`: both slides have the same basename.
   `methods/_valis_points.py::patient_slides` refuses two slides of one name ("VALIS
   names a slide by its basename"), so every VALIS row on every pair is `ok: false` and
   imputed at the initial pose. The table is then STARE against a competitor that never
   ran. Removing the refusal is not enough: `place_slides` links by basename into one
   folder and would overwrite the first slide with the second. The HyReCo section's "yes,
   by construction" for `valis120` / `valis120_hires` and its expectation of `n_ok` equal
   to `n_pairs` cannot hold as built. The test that should catch it builds its VALIS case
   with distinct names.
2. **The A/B split is enforced per cases folder only, and the method's configuration is
   not frozen.**
   - The settings hash holds `{proxy, reduce, version}` and nothing else. `run --dataset
     hyreco_b --opt ... --label ...` is accepted, and the scorer scores any result
     folder it finds.
   - Several B folders can be prepared from several A settings; each prints "REPORT SET.
     Nothing was chosen on it."
   - `--pixel-size-um` and `--keep-shared-ids` are free on B; inside the 2 % pixel-size
     window that moves a TRE by micrometres.
   - The tuned-set check is skipped when the tuning folder is absent.
   - A refused pair leaves every denominator instead of being scored at the initial
     pose; only the like-for-like block notices.
   - `--reduce` is in effect a STARE hyperparameter (it doubles the lattice and the
     capture range in micrometres). VALIS has no equivalent, and its primary row there
     is the one most likely to end for memory on 21-gigapixel slides.

3. **A blocked or never-started run reads as a method that fails everywhere.**
   `regbench.landmarks` (HyReCo's scorer) scores every pair of every expected row, and a
   pair without a result counts at the unregistered pose whatever the reason: a run its
   selfcheck gate blocked, an array task that never started and a method's own failure
   look the same in `hyreco_summary.csv`. ANHIR's scorer reads the gates and lists such
   pairs apart (`anhir_not_run.csv`); HyReCo's was left as it is while the set is
   dormant. It also takes the frame of a failed pair of a reference -> moving method
   to be `reference` (the same default ANHIR's scorer no longer has).

### `semisynth` (review item 9)

3. **The `same` cases hand pixel-correlation methods a signal real rounds never have.**
   The moving slide is a cubic resample of the reference's OWN pixels plus fresh noise;
   the reference gets no noise. For the `shift` pose (an integer translation) with a
   rigid or low-gradient field the moving slide is a pixel-exact shifted copy, so phase
   correlation on an empty background window recovers the shift (the reviewer measured a
   peak over s.d. of 579 there, against 4.6 for independent noise). STARE, ASHLAR and
   PALOM can be "right" on empty tiles; VALIS and DeeperHistReg, which work on
   downsampled features, gain nothing. "Every error is a lower bound" does not say this.
4. **The noise can be exactly zero.** Its s.d. is a MAD and is 0 on zero-filled
   background; the generator adds noise only `if self.noise_sd`, so every case of such a
   patient is noise-free without a refusal or a note.

In the scorer, the unit of a `semisynth` case is the real patient (not the group
`<patient>_w<i>_<family>`), with cell summaries per family, and the paired differences
carry a bootstrap interval over units ("Scoring").

## Caveats that travel with the results

1. **Whose settings.** STARE's settings and the pipeline's VALIS settings were chosen by the
   authors on this cohort with this metric. The other methods were NOT tuned on it, but
   "package defaults" describes only some of them: three primary rows are not a default.
   `valis120_hires` is upstream's high-resolution example, not its default
   (`valis120` is, and is shown beside it). `deeperhistreg_hr_gpu_asdrawn` is a shipped
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
6. **Known-truth cases** are a check chosen by STARE's authors; see "What these numbers
   are not" under "Known-truth error".
7. **About a fifth to two fifths of ASHLAR's nuclei are scored through an interpolated
   map** (the tile overlaps; see "ASHLAR", "What the overlap bands cost"). It has no bias
   when a round needs only a translation, and it is optimistic for ASHLAR when a rotation
   is left inside its tiles: at 0.5 degrees, 5.2 px of map error against 8.8 px for the
   copies the image shows. The selfcheck masks exactly those points. The
   `<variant>:non_ashlar` rows are the measure of how much it matters.
8. **DeeperHistReg's primary row follows its writer, not its registration.** By the
   adapter's reading the written image sits half a field pixel right and down of the
   registration: 2.5 px (0.81 µm) per axis, 1.15 µm along the diagonal, at the primary
   8192 px set; 4.9 px (1.59 µm) per axis at the 4096 px set. `within_1um` of
   `deeperhistreg_hr_gpu_asdrawn` can therefore be low with a perfect registration. The
   estimated-field row `deeperhistreg_hr_gpu` beside it shows the registration itself.
9. **The cohort's "nuclei" are StarDist nuclei grown by 10 px** ("What the cohort's nuclei
   are"): Dice and `within_1um` have a ceiling that is not registration, the same for
   every method.
10. **No ranking is statistically supported.** The paired differences carry a bootstrap
   interval over patients, which is coarse with few of them; there is no test and no
   p-value.
11. **The tissue-level block is secondary**: ACCREDIT's score on same-modality
   fluorescence, outside what it was validated for, by its released code where that
   differs from the preprint's text. It ranks nothing.
12. **Nothing here has run on the cluster.** The cgroup counter path and whether a site
   confines cores are unverified.

### STARE (`methods/stare.py`)

**Two rows.** Everything below is the `stare` row. The `stare_prep` row is the same
adapter with `--label stare_prep --opt prep=<frozen candidate>`: both images are first
rewritten by `prep.py` (a job of its own before COARSE, with the benchmark's allocation:
4 CPUs, 16 GB; `info.prep`), STARE registers those files at index 0 with the same
parameters in the same jobs, and STITCH draws what the plain row draws (the original
moving slide; on a brightfield case its plain grey image) with this row's transform. A label and its option come together: `stare_prep` takes the frozen candidate
and no other, `stare_prep_<candidate>` (ANHIR's tuning runs) that candidate, and `prep`
without one of these labels is refused. See "ANHIR: the challenge's training pairs".

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
- **The same transform as one `stare register`.** `test_stages.py` registers the 2048 x
  2560 pair stage by stage and with one `stare register`, and finds the anchor, the tile
  plan, every tile's control points and the manifest equal byte for byte, SOLVE's report
  equal value for value except its own clock (`solve.solve_seconds`), and the written
  image equal pixel for pixel. Each stage's argument list is compared whole with the
  one `stare.cli.register` hands that stage.
- **Outside the cohort's launcher** (the selfcheck, a laptop, `run` by hand) the same
  four commands run inside one job, each a process of its own, the tiles `workers` at a
  time (`info.jobs.execution: local`): the selfcheck gates the commands the cohort's
  jobs run.

- **Parameters.** `TIER_HIGH_STRIDE64` = the `high` row of the STARE table in
  `lib/RegPresets.groovy` (`tile` 2048, `halo` 256, `out_tile` 1024, `max_dim` 1024) plus
  `stride` 64, all five passed explicitly; a test reads the row from that file. The two
  STITCH flags the pipeline always passes are passed too: `--pixel-size` (the case's) and
  `--channel-names` (the case's, else the names in the moving slide's header). SOLVE is
  given `--max-disp` = the halo, as `conf/modules.config` gives `TILED_SOLVE`. A tile job
  names its tile by `--plan <COARSE's tile plan> --row <n>`, which reads the values the
  pipeline's module passes one by one. `info["cpu_count"]` records what COARSE's FFTs
  can use on the node.
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
| `--method valis_mirage` | `bin/register.py::valis_registration`, the function the pipeline's REGISTER runs, with `valis_mirage.ARGS` (= `nextflow.config`'s defaults: `memory_mode` high, micro fraction 0.125, `max_image_dim_px` 4000, `micro_reg` 2). `align_to_reference=True`, `crop="reference"`, SuperPoint + SuperGlue and the micro-rigid registrar are inside that function | the folder, the reference's file name, `n_cpus` = the allocation, `jvm_heap_gb` = `min(32 + 16 x rung, GiB - 4)` (the module's formula: 12, 60, 80, 96 GiB on the four rungs), `stage_checkpoint_dir=None` (files only) |
| `--method valis --label valis120` | `Valis(src, dst, img_list=..., reference_img_f=...)`, `.register()`, `.register_micro()` with no argument | the reference slide. `align_to_reference` stays `False` |
| `--method valis --label valis120_hires --opt hires=1` | upstream's high-resolution example (`docs/examples.rst:122-149` at v1.2.0 `[PARTIAL: lines 100-175]`): `micro_rigid_registrar_cls=MicroRigidRegistrar`, then `register_micro(max_non_rigid_registration_dim_px=floor(0.25 x the smallest slide's longest side))` | the reference slide. The label and the option must come together or the run is refused |

**Memory failures, and the heap.** `Valis.register()` catches its own exception, prints
it and returns `(None, None, None)`; the adapters used to raise a fixed text, so a Java
`OutOfMemoryError` or a NumPy memory error was `cause: exception` and VALIS never went up
the ladder while STARE's `MemoryError` did. Both adapters now wrap `valtils.print_warning`
for the duration of the call (`_valis_points.swallowed_errors`: it notes what VALIS hands
it and calls the original with the same arguments) and put that text into the error, so a
memory failure is `oom` and climbs.

- `valis120`, `valis120_hires`: the JVM is started by the adapter through VALIS's own
  `slide_io.init_jvm(mem_gb=...)` before `Valis(...)` is built, with
  `min(32 + 16 x rung, GiB - 4)` (12, 60, 80, 96 GiB on the four rungs: the pipeline
  module's formula), because VALIS's `registration.init_jvm` passes the literal 10 on
  whatever it is given. This is the ONE thing besides the reference slide that the
  benchmark sets for those rows, and it is not a registration parameter. Off the ladder
  (no memory request known) nothing is started and VALIS's 10 GB stands.
- `valis_mirage`: the pipeline's call is run as it is, with `jvm_heap_gb` from the same
  formula. Whether that value reaches the JVM is recorded, not assumed
  (`info.jvm_max_heap_gb`; "Not verified").
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
| `--method ashlar` | (none) | slight noise on the reference tiles. **ASHLAR's primary row** |
| `--method ashlar --label ashlar_clean` | `--opt noise=none` | no noise |
| `--method ashlar --label affine_ashlar` | `--opt affine=stare` | moving tiles placed by STARE's anchor, slight noise. A hybrid, never ranked as ASHLAR |
| `--method ashlar --label affine_ashlar_clean` | `--opt affine=stare --opt noise=none` | the same hybrid, no noise |

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
  message (`info["ashlar_raise"]`). A memory error is not caught: it stays `oom`.
- *The anchor* (`affine=stare`): each moving tile is cut, unrotated and uninterpolated,
  where STARE's rigid matrix (`info["m0"]` in `<out>/stare/<dataset>/<case>/run.json`) puts
  the matching reference tile's centre; ASHLAR is told the regular grid. A round whose
  `stare` result is missing, failed or has no matrix fails alone as `anchor unavailable:
  <path>`, and the patient's other rounds proceed.
- Reference tiles are cut once per noise level and shared by the plain and the anchored
  row, under `<out>/cache/ashlar_tiles/<dataset>__<patient>/` (delete it when both rows
  have run; it is large). ASHLAR's reference STITCH is not shared: each command does its
  own, and `info["edge_positions_identical"]` says whether the two came out equal.

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
`scipy.ndimage.shift` at the pinned SciPy against the 0.15 px gate; run time and memory.

**First commands in the image** (`labsyspharm/ashlar:1.20.0`, from the repository root,
with the pinned STARE release, `tifffile` and `zarr` importable: `benchmarks/ashlar`
imports `stare.manifest`, as `run_ashlar_original_arm.sh:63-80` arranges):

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
published, and it does not block (`ashlar_clean` is gated on `ashlar`'s check,
`affine_ashlar_clean` on `affine_ashlar`'s: the point-map code is the same).

### DeeperHistReg (`methods/deeperhistreg.py`)

**On a brightfield case (ANHIR)** the adapter hands the package the RGB image whole,
with the shipped parameters: "ANHIR: the challenge's training pairs", "Which runs, and
what each is given". Everything below describes the cohort's one-channel path.

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
| `deeperhistreg_gpu`, `_cpu` | `default_initial_nonrigid` | 4096 px, 8 levels, 900 iterations | the 4096 px point of its resolution ablation: 1.01 µm median error on re-stained HyReCo |
| `deeperhistreg_hr_gpu`, `_cpu` | `default_initial_nonrigid_high_resolution` | 8192 px, 9 levels, 1100 iterations | the resolution of its 0.59 µm result on re-stained HyReCo |
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
  one, `deeperhistreg_hr_gpu_asdrawn`, by the user's ruling of 2026-10-11 (it is the
  resolution of the paper's headline result); the 4096 px rows stand beside it in every
  table. `variants.PRIMARY_DEEPERHISTREG` is the one line that names it.
- **The selfcheck gate reads the `_asdrawn` row only** (`variants.gate_variants`). The
  estimated-field row is still checked and reported; when it alone fails it is left out
  of the results and the method runs.

**Inputs.** The package does not read OME-TIFF (issue #4 `[FULL]`) and its TIFF loader
counts top-level pages as pyramid levels (`tiff_loader.py:34-38`). Each slide's nuclear
channel is written as a tiled TIFF with ONE top-level page, same dtype and values,
uncompressed (`single_page_tiff`); the clock is paused meanwhile and `convert_s` recorded.
`info.grey_levels` is the number of 8-bit levels each channel keeps under the package's
min-max rule (counted on every fourth row and column): a few bright pixels can leave
16-bit DAPI in a handful of levels.

**Two rows per run, both reference -> moving, neither inverted.**

- `<label>` scores the field the method **estimated**: reference pixel -> moving pixel
  through the package's own `warp_landmarks` and the coordinate chain of issue #12 (a
  user's function the maintainer endorsed; the package itself never calls
  `warp_landmarks`). The chain leaves out the half-pixel terms of the two resamplings
  (`info.gap.chain_halfpixel_px_p95`).
- `<label>_asdrawn` scores the image **as written**: for every pixel of the file on disk,
  the moving-slide position the package's writer sampled. Its points are the transform of
  the image, and its selfcheck declares no offset.

They differ because the writer is not the registration. The field lives on the padded,
resampled grid, `warped(x) = source(x + u(x))`. The writer (`apply_deformation_pyvips`,
`dhr_utils/warping.py:224-239`) turns `grid + u` into an index image, enlarges it to the
full-resolution canvas with libvips' linear kernel, multiplies by the enlargement and
calls `mapim`. libvips enlarges with an input displacement of half a pixel, so output
pixel `X` reads the field at `X / S - 0.5` (libvips v8.15.2 `resize.c:233-277`,
`affine.c:445-480`, `:320-349` `[PARTIAL: those ranges, transform.c:158-252 and
mapim.c:225-300, 455-480, 589-594; not executed]`). Three consequences:

1. **Half a field pixel.** The written image samples `X - S/2 + S u(X/S - 0.5)`: its
   content sits about S/2 px right and down of the registration, 4.9 px at S = 9.77 and
   2.5 px at S = 5. The gap between the two maps is `(S/2) x Jacobian x (1, 1)`, a
   function of position.
2. **The stretch.** The writer enlarges by `canvas / field size`. The field's size is a
   floor (torch's `interpolate` with a scale factor), so on a slide whose loaded size
   does not divide by the resample ratio the written image is stretched relative to the
   registration: with torch 2.3.1, a loaded 7974 x 11975 slide gives a field of 4095 x
   6151, not 4096 rows, and the written image is 2.6e-4 too tall, about 10 px at the
   bottom edge of a 39871 px slide.
3. **The pads.** The writer pads at full resolution, the registration padded the loaded
   images; the two agree only where the slides' sizes divide evenly.

The as-drawn map uses the writer's own numbers for all three (`asdrawn_map`: full-
resolution centre pads, `S = canvas / field size`, the index image read at `X / S - 0.5`
linearly with the edge copied). The estimated-field row declares only the first
(`declared_offset`: the registration differenced over half a field pixel), and each run
records how far the two maps are apart and how much of that the declaration does not
explain (`info.gap.mean_px`, `beyond_declared_px_p95`). On real slides the second and
third terms are expected to be several pixels, which is why the as-drawn rows are the
ones that describe the image on disk.

**The written image** (`write_native`): the package's writer with the arguments its own
`save_final` passes, once per channel of the moving slide (the maintainer's route for
multichannel images, issue #6 `[FULL]`), the first page of each of its files stacked into
`native/registered.ome.tif`, values unchanged. It has the reference slide's size and is
on its grid, because the writer crops the padded canvas back by the reference's own pads
(`pair_full_saver.py:76-80`, `dhr_utils/utils.py:299-311`): `origin_xy` is `(0, 0)` and
the pads enter the map, not the origin. Pixels from outside the moving slide are the pad
value (255 counts, or 0 for `_fluo`) inside the padded canvas and 0 beyond it;
`valid_box` bounds the pixels that come from the moving slide. A failed writer fails
the row it was writing for, `<label>_asdrawn` (`writer`, or `writer_oom` for a memory
error), as for every method (a row is ok only when its image exists); the other row,
which has no image of its own, stays `ok`.

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
else `exception`.

**The checks.** The selfcheck compares both rows with the one image the writer produced:
`_asdrawn` as it is, the estimated-field row minus its declared offset. With a stand-in
for the package whose writer is drawn from the source, a wrong direction, dropped or
exchanged pads, a 1 % scale error, a missing half pixel and a field of the wrong sign
are each reported `inconsistent` on the `_asdrawn` row; an undeclared offset and a wrong
chain are each reported `inconsistent` on the estimated-field row alone
(`test_dhr_adapter.py`). The realcheck runs on the `_asdrawn` row (the one with the
image).

**What the selfcheck reaches.** One pair for all eight labels, 40000 x 50000 against
40000 x 48000: S = 9.77 at the 4096 px set (the cohort's class) and S = 5.0 at the
8192 px set, with every division exact, which the estimated-field row's declaration
needs. A coarsest pyramid level of 31 to 32 px. Not reached: the 8192 px set's second
resampling (a slide whose short side exceeds 40960 px: S between 5 and about 6.1), and
sizes that leave remainders, where the estimated-field row is expected to fail its
declaration and the `_asdrawn` row to hold (emulated in `test_dhr_adapter.py`); on the
cohort that is the realcheck's ground. The 8192 px set on the CPU is 1100 full-field
iterations at 80 megapixels: hours, and tens of GB.

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
# 2. the small pair on the CPU, both rows, shipped and fluo (S = 5)
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
`variants.<label>` and `variants.<label>_asdrawn` both `ok`; in the pair's record,
`variants.<label>.declared_px_mean_x/y` is the measured offset between the two maps
(4.9 px at the 4096 px set, 2.5 px at the 8192 px set). Add `--size 8200x10240:7780x9720`
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
| `--method palom --label palom_b1024 --opt block=1024` | the same recipe with the reference array PALOM's reader returned rechunked to 1024 px blocks (`ref_img.rechunk((1024, 1024))`) before `Aligner`. **Configured by the benchmark**, never primary | the above and the block size. The label and the option must come together or the run is refused |

Each run yields `palom_affine` (the coarse affine alone) and `palom` (the affine and one
shift per block), both `ref_to_mov`. Pair scope, no GPU, no JVM.

**What PALOM does.** A full six-parameter affine from ORB keypoints on thumbnails, then
one whole-pixel shift per block by cross-correlation of Laplacian-filtered blocks, then a
filter that replaces the shifts that disagree with a linear trend over the block grid
(and those of blank blocks) by the trend's value. **A block is one dask chunk of the
reference array, which is the reference file's TIFF tile**: 2048 px (666 µm) on the
cohort's preprocessed slides, 1024 px on the generated cases. PALOM has no block-size
parameter; the block of the `palom` row is therefore an accident of each file's layout,
and `palom_b1024` is the row with a stated block. Inside a block everything moves
together; neighbouring blocks may jump.

**The point map** is exact and closed-form, reference -> moving:
`written(P) = bilinear(moving, A^-1 (P - s_b(P)))`, with `A` = `aligner.affine_matrix`
(moving -> reference, homogeneous x, y), `s_b` = `aligner.shifts[b]` given as `(dy, dx)`
and row-major over `aligner.grid_shape`, and `b(P)` the block of `aligner.ref_img.chunks`
that holds pixel `P` (line references in the module's docstring). The only inverse is
the 3 x 3 affine's, which PALOM's writer takes itself. The other direction is not a
function: where two neighbouring shifts differ, some moving points are drawn twice and
some never. For a point between pixel centres a block edge sits half a pixel before the
edge pixel. Before writing, the adapter checks that the aligner's affine, shifts and
chunks are still the ones the map was made from, and hands the writer `aligner.ref_img`
(the rechunked array for `palom_b1024`), so the image is drawn on the blocks of the map.

**Seams.** The map jumps at an edge between two blocks whose shifts differ, and the image
is torn there. `band_ref` marks the reference nuclei whose outline's bounding box, grown
by 1 px, touches blocks that do not all carry the same shift; `band_truth` the known-truth
probes by the same rule on their own nucleus ("Off the seams"). An edge between two
blocks with the same shift is not a seam. Every method gets `<variant>:non_palom` and
`<variant>:non_palom_b1024` rows on identical nuclei.

**The written image** is PALOM's: `block_affine_transformed_moving_img` then
`write_pyramid(mosaics=[it], output_path, pixel_size)` with the writer's defaults
(pyramidal BigTIFF, zlib, 1024 px tiles, channels anonymous): every channel of the moving
slide on the reference grid, 0 where nothing lands. `native` has `valid_box` and no
`origin_xy`. The affine stage is drawn by the same calls with `mxs=aligner.affine_matrix`
(the package's coarse-only path), for the selfcheck only.

**What a run records** (`run.json: info`): `version`, `RECIPE`, `affine`,
`coarse_identity_placeholder`, `thumbnail_level`, `thumb_um_per_px`, `ref_down_factor`,
`mov_down_factor`, `n_levels`, `file_levels`, `auto_pyramid`, `block_px`, `file_block_px`,
`block_option`, `grid_shape`, `n_blocks`, `n_blocks_blank`, `n_blocks_replaced`,
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
  nothing to fit. That is what the `rigid` known-truth family, or a pair the affine
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
detections within 8 px of such an edge; the affine stage has none. Pairs: two per row,
8 x 10 blocks each (the second with a smaller moving slide, so the two thumbnail
down-factors differ). Both rows read files with 2048 px tiles, the cohort's layout:
`palom` on 16384 x 20480, so its blocks are the 2048 px ones of its cohort row (since
2026-10-11); `palom_b1024` on 8192 x 10240, cut to 1024 px blocks. If under 10 %
of the blocks have a shift of at least 0.5 px (`MIN_SHIFT_PX`; `run.json:
info.shift_meaningful_fraction`) the `palom` variant is `not_checked`: the block map
then equals the affine map to within the gate and a wrong sign or block order could not
show. Until 2026-10-11 the guard counted every non-zero shift, which a block filled from
PALOM's regression with a few hundredths of a pixel satisfies while testing nothing; at
0.5 px a reversed sign moves a dot by 1 px, 2.5 times the gate.

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
# 3. the selfcheck of each row
python -m benchmarks.regbench selfcheck --method palom --mask seams --work W/palom
python -m benchmarks.regbench selfcheck --method palom --label palom_b1024 \
    --opt block=1024 --mask seams --work W/palom_b1024
```

`ok` for 1 and 2 is every test passed and none skipped; step 2 prints, as a warning, what
PALOM's filter does on equal shifts. If step 1 fails with `seek of closed file`, the
workaround is needed: step 3 still runs (the adapter applies it) and `closed_file.applied`
is true in each `run.json`. `ok` for 3 is exit 0 and, in `W/<row>/selfcheck.json`,
`"status": "ok"` with `variants.<row>_affine` and `variants.<row>` both `ok` on both
pairs, `n_masked` a few percent of `n_detected` for `<row>` (under 5 % for `palom`,
under 10 % for `palom_b1024`: the caps) and 0 for `<row>_affine`.
`inconsistent` on `<row>` with step 2 green means torn dots outside the mask or a shift
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
`test_hyreco.py` writes stand-in files in the dataset page's layout (small RGB TIFFs,
millimetre CSVs) and checks the module's rules on them: the subset split, pairing by
line, the millimetre conversion, every refusal, the two aggregations, the tables'
separation from the cohort's; it registers one stand-in pair with the installed STARE
(at full resolution and on a 2 x reduced proxy) and runs the VALIS 1.2.0 adapter on a
brightfield case against `fake_valis.py`. `test_submit_hyreco.py` reads the job graph
of `submit_cohort.sh --datasets hyreco_b` off its dry run.
`test_nuclei_sets.py` checks the two polygon sets on hand-made squares (a nucleus inside
each outline): the contract and its backward compatibility, the driver moving both, the
nuclei's own gate, the seam mask by inheritance and by geometry, the side-by-side lines
of `summary.md`, and that growing the outlines changes neither the nuclei nor the truth
spec. `test_nuclei_cohort.py` checks the cohort's sources on small label images: a mask of
the slide is used, one of another size or another frame is refused with the reason, the
rendered `segment.py` call against `nextflow.config` and `conf/modules.config`, and the
re-run itself against a STAND-IN `segment.py` that writes the two masks under the real
script's names (StarDist is not installed here; the real script was never executed).
`test_anhir.py` builds a stand-in for the challenge's folder (a cover table, small JPEG /
PNG images, landmark files) and checks the split and its hash, the roles, the common
prefix, the pixel sizes, the grey conversion, the run lists, the scores against a
transcription of the official evaluation's functions, failures, the tables' wording and
the choice of the preprocessing with its refusals; one test reads the real cover table
when it is on this machine and skips otherwise. `test_prep.py` checks every candidate
(no shift, independence of the block size, the identity where stated), the frozen file,
and registers one synthetic pair three times with the installed STARE (plain, the
identity candidate, CLAHE). `test_submit_anhir.py` reads the job graphs of
`--datasets anhir` and `--datasets anhir_tune` off the dry run.
`test_selfcheck.py` runs STARE once for real, on a 1024 x 1280 pair;
`test_stare_adapter.py` runs STARE's selfcheck on its own two pairs and two real
registrations; `test_stages.py` registers the 2048 x 2560 pair stage by stage, as the
cohort's jobs do, and once with a single `stare register`, and compares the two.
