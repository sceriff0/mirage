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
| the cohort's cases (`arms_manifest.py`, `datasets/multiplex.py`) | here: the arms' original multichannel slides and the QC nuclei of one arm. Built and tested against small trees laid out like the arms' results; never run on the cohort itself |
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
| the run list (`variants.py`, `python -m benchmarks.regbench runs`) | here, as in "The run list" below: 18 runs, 35 result names, six primary rows, a flag that pauses a run (nothing is paused) |
| the cohort launcher (`slurm/submit_cohort.sh`, `ladder.py`) | here, as in "Running the cohort on a cluster" below: every run behind its own gate, four chained job arrays per run and dataset, the scorer last. **Never submitted**: tested through its dry run and through a stand-in `sbatch` that records its calls |
| the end-to-end cost view (`cost.py`, `cost_end_to_end.csv`) | here, as in "Cost" below. The pipeline's own steps are read from a Nextflow trace, tested with a trace written by hand; no real trace has been read |

Nothing here has run on the cluster.

## Layout

```
mapping.py     PointMap: a point mapping and the direction it was estimated in
cases.py       the on-disk contract: case.json + points.npz in, warped.npz + run.json out
cells.py       nuclei of a native slide: the built-in segmenter, label images, cell GeoJSON
arms_manifest.py  the arms cohort as a `prepare multiplex` manifest, one row per moving round
pairs.py       the scores of one (case, variant), on bin/utils/cell_pairs.py
resources.py   the allocation, the process-tree monitor, out-of-memory detection
methods/       one module per method, and the driver that runs any of them
datasets/      synthetic, semisynth, multiplex: each writes cases
score.py       reads files only; writes <out>/tables
selfcheck.py   do a method's points follow the image its own writer produced? (a gate)
tissue.py      the written images against the reference at 4000 px (ACCREDIT's score; secondary)
ladder.py      the memory ladder's rule: a unit goes up a rung only when it ran out of memory
cost.py        cost from the cohort's files to a merged image, step by step (secondary)
slurm/         the cohort launcher (submit_cohort.sh) and what its jobs run; two older
               one-allocation launchers (submit.sh, submit_scale.sh)
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

The first command reads `ROOT/preprocess_shared/csv/preprocessed.csv` and the QC nuclei
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
18 runs, 35 result names, 4 runs on a GPU, 7 per patient. `score` expects exactly those
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
  has what each holds.
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

**A method** is a module with `SCOPE` (`"pair"` or `"group"`), `VARIANTS`, and one generator:

- pair: `register(case, work, opts)` yields `(variant, PointMap, info)`;
- group: `register_group(cases, work, opts)` yields `(case_id, variant, PointMap | None,
  info)`. `cases` is every case of one patient. A failed round is
  `(case_id, variant, None, {"error": msg})`; a degraded one carries `info["problem"]`.

`mapping.PointMap(fn, direction)`: `fn(xy (N, 2)) -> (N, 2)` in full-resolution pixels of the
native slides; `direction` is `mapping.MOV_TO_REF` or `mapping.REF_TO_MOV`.

Optional: `write_native(state, case_id, dest_dir)` (the method's own writer, called after
the generator ends with the clock paused; `state` is what the method put in
`info["_state"]`), or an image written at a yield inside `opts["monitor"].pause()` /
`.resume()` and handed over as `info["native"]`.

The driver hands a method three things through `opts` that are not options of the run and
are not recorded as such: `monitor` (the clock), `native_dir` (`f(case_id, variant)`: where
an image written at a yield belongs, `<out>/<variant>/<dataset>/<case>/native/`; the work
directory is deleted after the run) and `label`. A method whose label does not rename its
variants by the replace rule declares the names in `VARIANT_LABELS` (`valis120_hires`).

**The driver** (`methods.run_case`, `methods.run_group`), at each yield, before it resumes
the generator and with the clock paused: moves the nuclei through the map (the moving
nuclei for `mov_to_ref`, the reference nuclei for `ref_to_mov`), writes `warped.npz`
(`cells_xy`, `direction`, `band` when `info["band_mov"]` is given and `band_ref` when
`info["band_ref"]` is: a method's own seam mask, one bool per moving or per reference
nucleus; `band_truth` when `info["band_truth"]` is: the same seams by known-truth probe,
for a reference-side mask; `truth_xy` = the map
applied to the truth probes of the side it starts from, and `grid`, `grid_shape`,
`grid_step` = the map on a lattice for the Jacobian statistics), writes `run.json`, and
calls `on_variant`.

**`run.json`**: `ok`, `error`, `cause`, `problem`, `direction`, `launch_id`, `unit`
(`pair` | `patient`), `group_cases`, `register_s`, `write_s`, `resources`, `native`,
`selfcheck_status`, `attempt`, `mem_requested_gb`, `cpus_requested`, `cpus_affinity`,
`blas`, `cpu_model`, `previous_attempts`, `info`, and `realcheck` (`status`, `flag`) when
`run --realcheck` was given. It is written as
`ok: false, cause: killed` before the method starts, so a task the scheduler kills leaves a
failed row. `cause` is one of `exception`, `killed`, `not_run`, `oom`, `timeout`,
`gpu_oom`.

## Methods (the design; see "State of this port" for what exists)

| variant(s) | software | run unit | native transform | how nuclei are moved |
|---|---|---|---|---|
| `stare_rigid`, `stare` | pinned `stare-registration`, mirage's `high` tier, stride 64 | pair | affine + mesh, moving -> reference | moving nuclei through `stage_warp.make_warper`; reference frame |
| `valis_mirage_rigid`, `valis_mirage_nonrigid`, `valis_mirage` | VALIS 1.0.0 through the pipeline's own call | patient | affine + backward field `bk_dxdy` | reference nuclei (pixels of the written image's grid) through VALIS's `warp_tools.warp_xy_inv`; moving frame |
| `valis120_rigid`, `valis120`, `valis120_micro` | VALIS 1.2.0, **upstream's default setting** (`register()` then `register_micro()` with no argument). Run and shown; not primary | patient | same | same |
| `valis120_hires_rigid`, `valis120_hires_nonrigid`, `valis120_hires` | VALIS 1.2.0, upstream's high-resolution example (`--opt hires=1`). **`valis120_hires` is VALIS 1.2.0's primary row**, by the user's ruling | patient | same | same |
| `deeperhistreg_{gpu,cpu}`, `deeperhistreg_hr_{gpu,cpu}`, and the same four with `_fluo` | DeeperHistReg 1.0.1 (the method is RegWSI), a shipped parameter set with only the device changed; `_hr` is the 8192 px set, `_fluo` adds two benchmark-set keys | pair | backward field on the padded, resampled reference grid | reference nuclei through its `warp_landmarks` and the coordinate chain of its issue #12; moving frame. Secondary |
| the same eight with `_asdrawn` | the same runs | pair | the image its own writer puts on disk | reference-grid pixels of the written image through the writer's own arithmetic; moving frame. The rows that can be primary: **`deeperhistreg_hr_gpu_asdrawn` is DeeperHistReg's primary row**, by the user's ruling (the 4096 px set is the package's default) |
| `ashlar`, `ashlar_clean`, `affine_ashlar`, `affine_ashlar_clean` | ASHLAR 1.20.0, its own command, on tiles cut from stitched slides | patient | one shift per tile | moving nuclei by their tiles' shifts; reference frame |
| `initial` | none | | identity | |

Honest limits of "natively": ASHLAR is run on tiles cut from stitched slides, not raw tiles;
DeeperHistReg is run on fluorescence with brightfield defaults; STARE is run as one process,
not as the pipeline's cluster jobs; `valis120` at upstream defaults is upstream's
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
    [--mask overlap] [--m0-from DIR] [--size HxW[:HxW] ...]
```

Exit 0 `ok`, 1 `inconsistent`, 2 `not_checked`, 3 `crashed`. **Anything but 0 blocks the
method** on the cohort. It writes `W/selfcheck.json` (every variant's numbers, per pair and
per 512 px block).

**The pair** (`selfcheck.make_pair`): two-channel 16-bit OME-TIFFs at 0.325 µm per pixel,
channel 0 nuclei (`DAPI`), channel 1 `dots`, on a canvas that is not square (2048 x 2560 by
default) with tissue that does not start at the origin. The moving slide is the reference
seen through a known map (1.5 degrees, a shift, a smooth 10 px field). Its dots channel
holds Gaussian dots (sigma 1.5 px, 20000 counts, zero background) on a 96 px lattice, each
jittered by up to 24 px so a mapping wrong by a whole lattice period cannot pass. The
reference's dots channel is empty. Dots are addressed by channel INDEX, never by name.
Methods with a size-dependent code path are checked on more than one pair
(`selfcheck.SIZES`): `stare` also at 4096 x 5120 (a four-tile junction), both VALIS
adapters also on a pair with a 30000 px side, every DeeperHistReg label on one pair of
40000 x 50000 against 40000 x 48000 (the cohort's size class, with sizes that divide
exactly; see "DeeperHistReg").

**The check.** The method runs natively through the normal driver, writer included. The
dots are found blind in the method's own output (`find_dots`: a 15 px median background,
peaks above 8 robust s.d. and a quarter of the brightest, a second-moment width of 1.2 to
1.8 px in a 7 px window, an intensity-weighted centroid) and compared with the mapping:

- `mov_to_ref`: `fn(dot)` against the nearest detection within 3 px;
- `ref_to_mov`: `fn(detection)` against the nearest original dot within 3 px.

The residual is **the point mapping minus the image**, in the frame the mapping leads to.
A method whose writer is known to depart from its mapping declares that departure as a
function of position (`SELFCHECK_OFFSET`); it is subtracted point by point, and its sign
is part of the claim.

**The gate (frozen).** 95th percentile of the residual at most 0.40 px, mean signed
residual at most 0.15 px per axis, at least 99 % of the dots found: on the whole image AND
in every 512 px block holding at least 8 dots. For `mov_to_ref`, at most 1 % of the
detections may be dots the mapping puts nothing on.

| status | meaning | blocks the method |
|---|---|---|
| `ok` | the gate holds | no |
| `inconsistent` | the points do not follow the image | yes |
| `crashed` | the method raised | yes |
| `not_checked` | nothing could be measured: the dots were not detected, the method wrote no image, or the run ran out of memory or time (`cause` says which). Never reported as `inconsistent` | yes |
| `no_stage_image` | a stage the method's own writer cannot draw (`stare_rigid`: `stare register` writes one image). Reported, never counted as checked | no |

The overall status is the worst over the checked variants.

**Measured on `stare` 1.2.1** (2026-10-09, macOS arm64, `--opt workers=4`; the only method
whose selfcheck has really run):

| pair | dots found | residual p50 / p95 px | mean x / y px | worst block p95 | worst block mean |
|---|---|---|---|---|---|
| 2048 x 2560 | 537 / 537 | 0.055 / 0.090 | +0.003 / +0.002 | 0.104 (20 blocks) | 0.020 |
| 4096 x 5120 | 2202 / 2202 | 0.057 / 0.091 | 0.000 / 0.000 | 0.108 (80 blocks) | 0.025 |

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
`SELFCHECK_OFFSET(info)`; `SELFCHECK_MASK(info, mask)` (`--mask overlap`);
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
a gate (it is the only check at cohort field scales); the launcher that applies it is not
here yet. It has only run on synthetic slides.

## Scoring

Nuclei are segmented once on each native slide, before registration, so they do not depend
on any method's output. Pairing and IoU are `bin/utils/cell_pairs.py`, imported
(`pairs.py` loads that one file; `bin/utils` is not put on `sys.path`).

- **Gate:** `1.5 x median_equivalent_radius` of the two native sets, once per case, the same
  for every method.
- **Pairing:** linear sum assignment; mutual nearest neighbour beside it (`_mnn`).
- **Per case and variant:** pair fraction, matched Dice, IoU p50, share of pairs with
  IoU >= 0.5, centroid displacement in µm (p50, p90).
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
  **`within_1um` and matched Dice are the fine-ranking numbers.** `within_3um` is not, and
  `count_ratio` is not a ceiling.
- **Reading rule, fixed before any cohort run:** displacement is censored at the gate and
  conditional on being paired, so it is not used to rank methods. It is published for every
  case with the `displacement_readable` flag (pair fraction exceeds the null's maximum by at
  least 0.10) and summarised over the readable cases only, with their count. Dice is ranked
  only where pair fraction clears the null's maximum (`dice_matched_ranked`).
- **No pixel size:** µm columns and the curve are NaN, not zero.

A failed, killed or never-started run is a row scored at the unregistered pose
(`imputed_initial`), with `problem` and `cause` filled; never a gap.

It is self-consistency on the registration channel: there is no held-out structure and no
landmarks on real slides.

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
| `wave` | 10 px peak | 1500 px | the benchmark's choice |
| `grid` | s.d. 5 px | 9 x 9 nodes | literature value as recorded in STARE's README, not re-read |
| `multiscale` | RMS 2.0 µm | 333, 166, 83 µm, fixed in µm | amplitude as above; the spectrum is the benchmark's choice |
| `bumps100` | 0.2 to 0.6 sigma | sigma 100 px, 120 bumps per 8192 px window | the benchmark's choice; inside STARE's model; 24 % of the window moves by more than 1 µm |
| `bumps25` | 0.2 to 0.6 sigma | sigma 25 px, 4000 bumps per 8192 px window | **outside STARE's model** (finer than its 64 px lattice); 23 % of the window moves by more than 1 µm |
| `seams` | s.d. 3 px | 1950 px blocks | **outside STARE's model** (discontinuous); no seam on any method's grid |

**Seams.** With 1950 px blocks the seams of an 8192 px window fall at 1950, 3900, 5850 and
7800 px. None of these positions is a multiple of any grid a method works on
(`method_grids_px` in the spec): STARE's 64 px vector lattice and 2048 px tile, ASHLAR's
922 px tile pitch, PALOM's 1024 and 2048 px blocks. The test
`test_no_seam_falls_on_a_method_grid` checks every seam position against every listed
grid (`truth.seams_on_a_grid`), not the block size alone. "Grid" here means multiples of
the listed size from the window's origin; ASHLAR's tile ENDS (922 k + 1024) are not in the
list. A window smaller than 1950 px has no seam at all: a reduced copy of this family is
a constant shift.

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
- **Which families.** Every continuous one: `rigid`, `wave`, `grid`, `multiscale`,
  `bumps100`, `bumps25`. Only `bumps25` has been seen to come near (the others' smallest
  determinant, on the frozen synthetic cases at 4 px: 0.90 to 1.0, and 0.46 and 0.48 for
  `bumps100`; 0.29 to 0.67 over 60 further `bumps100` seeds), so for them the rule is a
  guarantee that costs one measurement. **`seams` is not under the rule**: it is
  discontinuous on purpose, and a tile that shifts towards its neighbour overlaps it by
  design; a lattice across a seam reads that as a negative determinant on almost every
  seed (the two frozen cases: -0.85 and -1.0 at 4 px).
- **None of the 14 frozen synthetic cases is re-seeded** (the 12 under the rule: 0.25 to
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

- `synthetic`: one Gaussian-blob scene per family (7 patients, 14 cases).
- `semisynth`: variant `same`, noise 1.0 x the background s.d., on the 2 windows with most
  tissue of each cohort patient's reference round. **The spec names no patient.** The
  patients are every patient of the manifest given to `prepare`, or a subset declared with
  `--patients` / `--patients-file`; an id that is not in the manifest is an error that
  names it. Cost, stated up front: each patient gives 2 windows x 7 families x 2 poses =
  28 cases of 8192 px (14 patients in the sense above), and every method runs each of them.

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
runs over 28 x the number of patients, in the manifest's order.

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
(`extra.truth_spec_sha256`); `score` refuses cases of two hashes. The hash covers the
parameters, not the generator's code: the commit is what fixes that, and it is to be tagged
`regbench-truth-frozen-<date>` before any method runs. The spec was re-hashed once on
2026-10-10 (version 2: seam block, bump counts, no patient named), before any method was
scored on it.

**Numbers** (`truth.error_stats`), per case and variant, in micrometres: `tre_um_p50`,
`tre_um_p90`, `tre_um_max`, `frac_under_1um`, `frac_under_2um`, `initial_tre_um_p50` (the
unregistered pose) and `frac_improved`, with `n_probes` and `n_nonfinite`. A point a method
returns as NaN counts against it: its error is infinite. A failed run is scored at the
unregistered pose. Without a pixel size the micrometre numbers are NaN. A variant with
seam mask's owner on that case gets a second row `<variant>:non_<owner>` in
`nonband_truth.csv`, on the probes the mask leaves (see "Off the seams": the probes of
the moving nuclei a moving-side mask leaves; for a reference-side mask, the probes the
owner's run did not mark).

**What these numbers are not.** The amplitudes are literature values as recorded in
STARE's README (the papers were not re-read for this port), not STARE's output on the
cohort. The bias is elsewhere: the families, their smoothness and the multiscale spectrum
were chosen by STARE's authors after its design was fixed; `grid`, `multiscale` and `wave`
are fields a B-spline mesh on a 64 px lattice can represent; the `same` variant reuses the
reference's pixels, so every error is a lower bound. Any amplitude later estimated from a
method's fields is post hoc and names that method.

### Tables (`<out>/tables/`)

| file | one row per | what |
|---|---|---|
| `cells.csv` | case x variant | `dataset, tier, case_id, group, family, pose, method, frame, selfcheck_status`, every key of `pairs.RECORD_KEYS`, `coverage, nn_distance_um_p50, fold_pct, sd_log_jac, imputed_initial, problem, cause`. Every row scores ALL nuclei |
| `nonband_cells.csv` | case x variant x owner | the same columns for `<variant>:non_<owner>`, plus `variant, owner, n_mov_kept, n_mov_all, n_ref_kept, n_ref_all`. See "Off the seams" |
| `nonband_summary.csv` | dataset x owner x variant | the summary of those rows, with the nuclei kept |
| `nonband_truth.csv`, `nonband_truth_by_family.csv` | as `truth.csv`, `truth_by_family.csv` | the known-truth error on the probes a moving-side mask leaves, with `owner, n_probes_all` |
| `dropped.csv` | variant | written only when `score --drop` names rows: `variant, reason, had_results`. Rows whose own selfcheck failed without blocking their method; they are in no other table |
| `nonband_missing.csv` | dataset x owner | `n_cases`, and `n_cases_without_mask`: the cases the owner's primary run did not register, which have no off-seam row for any method |
| `summary_cells.csv` | dataset x variant x subset | per patient the median over rounds, then median, min, max over patients; `n_patients, n_cases, n_ok`; the minimum over cases; each headline beside its null; Dice over the cases that clear the null; displacement over the readable cases with `n_readable`. `subset=common` repeats it on the cases every primary variant registered |
| `paired_vs_primary.csv` | patient x variant | the difference of each headline against the `stare` row |
| `failures.csv` | variant | launches made, launches failed by cause, rounds lost |
| `resources.csv` | case x variant | the cost recorded at that variant's yield |
| `summary_resources.csv` | patient x variant | sum over rounds for a pair method, the launch's total for a per-patient one; peaks are maxima, never divided by rounds |
| `truth.csv` | known-truth case x variant | `dataset, tier, case_id, group, family, pose, in_stare_model, truth_spec, method, frame`, the numbers of "Known-truth error", `fold_pct, sd_log_jac, imputed_initial, problem, cause`; plus the pseudo-rows `initial` and `truth` (zero by construction) |
| `truth_by_family.csv` | dataset x family x pose x variant | `n_cases, n_ok`, median and maximum of p50, median p90, maximum of max, median shares under 1 and 2 µm, median `fold_pct`. No row over families |
| `tissue.csv`, `tissue_summary.csv`, `tissue_missing.csv`, `tissue_dose.csv`, `tissue_dose_summary.csv` | see "Tissue-level agreement of the written images" | ACCREDIT's score and its pieces on each written image. Secondary: in no table above, no paired difference, no ranking |
| `cost_end_to_end.csv` | patient x result that wrote an image | the steps apart: `method_convert_s`, `estimate_s`, `write_s`, `benchmark_copy_s`, `pipeline_convert_s`, `pipeline_merge_s`, `total_s`, the peaks, `device`, `gpu_model`, `written`, and where each imported number came from. Secondary; see "Cost" |
| `cost_end_to_end_summary.csv` | dataset x result | the median over patients of each step, and how many patients have a total |
| `launch.csv` | gate record | written only with `score --gates`: what each of the launcher's gates decided (`run, gate, status, blocked, dropped, reference`) |
| `summary.md` | | the summary, with the line "descriptive; no ranking is claimed" under six patients, and the known-truth table |

### Off the seams

A method that builds its image from tiles or blocks draws some nuclei across a join, and
its point map is not exactly its image there. Such a method (`score.SEAM_OWNERS`: `ashlar`,
`palom`, `palom_b1024`) writes a **seam mask** with its primary row:
one bool per nucleus, from its own geometry alone (ASHLAR: the tile grid and the positions
it accepted; PALOM: its block grid and the shifts it accepted; never any method's scores).
ASHLAR's is on the moving nuclei (`info["band_mov"]`, `band` in `warped.npz`); PALOM's is
on the reference nuclei (`info["band_ref"]`, `band_ref`).

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

The cases of the frozen known-truth spec are left out of `resources.csv` and
`summary_resources.csv` (cost is read on the cohort); their failures are still counted in
`failures.csv`.

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
3. Delete `<out>/cache/tissue` after re-preparing cases: the reference's plane is made
   once per case and kept.

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

- **The clock and the peak are per variant.** Each `run.json` holds the cumulative time and
  the process-tree RSS peak up to its own yield. Warping the benchmark's points and a
  method's own writer run with both paused (`resources.Monitor.pause()` / `.resume()`).
- **`peak_rss_gb` is the cross-method memory number.** `cgroup_peak_gb` is the whole job's
  high-water mark, writers included, for every method: it cannot be paused.
- **The ladder.** Every method runs at 4 CPUs. Memory starts at 16 GB and moves up one rung
  (64, 256, 512 GB) only when the run ran out of memory. `attempt`, `mem_requested_gb` and
  `cpus_requested` are recorded per run; earlier failed attempts are kept in
  `previous_attempts` and reported as `wasted_*`. `slurm/submit_cohort.sh` submits the
  four rungs as four chained job arrays; a task of a later rung starts, reads its unit's
  last `run.json`, and does nothing unless that says `oom` (`ladder.py`, `run
  --retry-oom`). The driver reads `REGBENCH_ATTEMPT` and `REGBENCH_MEM_GB`.
- **What counts as out of memory.** The driver watches the task's own cgroup counter
  (`oom_kill`) and ends the run as `cause: oom` when it rises, because the kernel may kill
  one worker and leave the parent waiting. `resources.MEMORY_ERRORS` is the fixed list of
  self-reported memory errors that are also `oom`. `timeout`, `exception` and `gpu_oom`
  never escalate.
- **Threads.** No thread variable is exported. Where a method has its own core-count option
  it is set to `resources.n_cpus()` = `min(SLURM_CPUS_PER_TASK, affinity)`; both numbers
  are recorded. `cpu_efficiency` (CPU time over wall x CPUs) is reported because a method
  that starts one thread per node core can exceed its allocation where cores are not
  confined.
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
| `stare` | inside the clock: STARE's own reader on the cohort's OME-TIFFs | none | `stare register` up to SOLVE's manifest (COARSE, REG-TILE, SOLVE) | STITCH, deferred until the clock has stopped, timed on its own | one OME-TIFF per round, every channel, BigTIFF, uncompressed, no pyramid |
| `valis_mirage` | inside the clock: VALIS's readers, its conversion of every slide to its working images, and the JVM start | not separable | the pipeline's `valis_registration` up to the writer (rigid, non-rigid, micro) | `Slide.warp_and_save_slide`, clock and peak paused, timed per slide | one pyramidal OME-TIFF per round, every channel (libvips, LZW) |
| `valis` | inside the clock: VALIS's readers, its conversion of every slide to its working images, and the JVM start | not separable | `Valis.register()` and `register_micro()` | `warp_and_save_slides`, clock and peak paused; one call writes the patient, a round's share is the call over the slides written | one pyramidal OME-TIFF per round, every channel (libvips, deflate) |
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

### From the cohort's files to a merged image (`cost_end_to_end.csv`, secondary)

The headline cost is the time to ESTIMATE the transform (`register_s`), and stays so. A
second table puts the other steps beside it, each in its own column, per patient and per
result that wrote an image:

| column | what | measured |
|---|---|---|
| `method_convert_s` | a conversion the method needs before it can read the slides: DeeperHistReg's single-page TIFFs (`convert_s`, plus the further channels its writer needs), ASHLAR's tiles (`retile_s`, once per patient). `none` when it reads the cohort's files as they are; `not separable` for VALIS | here, 4 CPUs, clock paused |
| `estimate_s` | the registration clock | here, 4 CPUs |
| `write_s` | the package's own writer | here, 4 CPUs, clock paused |
| `benchmark_copy_s` | the benchmark's own copying of a package's output (ASHLAR, DeeperHistReg). In NO total | here |
| `pipeline_convert_s` | the pipeline's `CONVERT_IMAGE` on that patient's slides: the files every method reads | IMPORTED from a trace |
| `pipeline_merge_s` | the pipeline's `MERGE_AND_PYRAMID` on that patient, added only where the method's own written image is not already a pyramid that holds every channel in one file; `not needed` otherwise | IMPORTED from a trace |
| `measured_here_s` | `method_convert_s + estimate_s + write_s` | |
| `total_s` | the sum of all of the above that apply; empty for a patient the trace does not have | |
| `written` | what the method's writer left: channels, pyramid or not, compression, size | the file |
| `device`, `gpu_model` | DeeperHistReg's CPU and GPU rows are in this one table, told apart here | |

```bash
python -m benchmarks.regbench score --cases CASES --out OUT \
    --pipeline-trace /path/to/arm/trace/trace.txt --pipeline-trace-label "arms: compute_all"
```

- **Nothing is run for the pipeline's steps.** They are read from the `trace.txt` of an
  earlier pipeline run on the same cohort (the arms benchmark's compute arm), by header
  name, never by column position. A patient is matched by the task's `tag`, which is the
  patient in both processes (`modules/local/convert_image.nf`, `merge_and_pyramid.nf`).
  Only `COMPLETED` and `CACHED` tasks are used; a cached task repeats the figures of the
  run that produced it, and the row says how many were cached.
- **Every imported number names its source and its allocation** (`pipeline_source`,
  `pipeline_convert_alloc`, `pipeline_merge_alloc`: the CPUs and memory that task was
  GIVEN), because those tasks did not run at this benchmark's 4 CPUs. A total therefore
  adds seconds measured under two allocations, and says so.
- **Without a trace** the two pipeline columns read `not available`, there is no total,
  and everything measured here is unchanged.
- **A merge is never added on a guess.** Whether a method's image is already a complete
  pyramid is read from the file it wrote. A run that recorded no description gets
  `not available` there, not an assumption either way.
- **What the rule does and does not say.** VALIS and PALOM write one pyramidal file per
  round with every channel of that round; by the rule above no merge is added for them.
  The pipeline's own product is one file per PATIENT with every round's channels, which
  only ASHLAR's mosaic is. The table follows the rule as ruled and `written` shows the
  difference; it does not claim the outputs are equivalent.
- Other process names: `--pipeline-convert-process`, `--pipeline-merge-process`
  (repeatable; the last part of the trace's `process` column). The pipeline's
  `SPLIT_CHANNELS`, which feeds its merge step, is not counted by default.

## Running the cohort on a cluster

`slurm/submit_cohort.sh` submits everything from a login node and returns; nothing waits
there, and no job submits a job. **It has never been submitted**: what is tested is the
job graph its dry run prints, and the same graph through a stand-in `sbatch`
(`test_submit_cohort.py`), plus the rules its jobs apply (`test_ladder.py`,
`test_gates.py`).

### What it submits

The run list comes from `python -m benchmarks.regbench runs` and from nowhere else; the
script names no method.

1. **Selfchecks**, one per run that gates a run (16), in that run's image with that run's
   options and mask (`overlap` for ASHLAR, `seams` for PALOM), at 64 GB and 4 CPUs. Two
   more attempts wait behind it at 256 and 512 GB; each runs only when the one before
   ended for memory (`selfcheck --retry-memory`) and otherwise leaves the record as it
   is. `selfcheck.json: attempts` holds the allocation and outcome of each. The anchored
   ASHLAR rows' selfcheck waits for STARE's and reads its work (`--m0-from`). Records:
   `<out>/logs/selfcheck/<run>/selfcheck.json`.
2. **A gate per run** (18; `gate.sbatch`, `regbench gate`). It fails when the selfcheck
   that gates the run is anything but `ok`, a selfcheck that left no record included.
   Then that run, and only that run, never starts: every one of its arrays waits for its
   gate with `afterok`, and every job carries `--kill-on-invalid-dep=yes` so that it is
   cancelled instead of pending for ever. The task itself reads the gate's record again
   before it runs. `<out>/logs/gates/<run>.tsv`: run, status, rows to drop, blocked,
   the reference comparison.
3. **The ladder**: per run and dataset, four job arrays at 16, 64, 256 and 512 GB, 4 CPUs
   each, each waiting for the one before with `afterany`. One task is one case for a run
   that works per pair and one patient for a run that works per patient. A task of a
   later array exits at once unless its unit's last `run.json` says `oom`; where the
   record still says `killed` (the process died before it could say why), the scheduler's
   state for that task at the rung below decides: `OUT_OF_MEMORY` makes it `oom`,
   `TIMEOUT` makes it `timeout`, and the record is rewritten with `cause_from`. The
   output directory is kept between rungs, so the cost of a failed attempt is kept
   (`previous_attempts`). `REGBENCH_ATTEMPT` and `REGBENCH_MEM_GB` reach the task (the
   pipeline's VALIS sizes its JVM heap from them). The anchored ASHLAR rows wait for
   STARE's last array on the same dataset.
4. **The realcheck gate** of `valis_mirage` and `valis120_hires`: the first patient of
   the first dataset goes alone, on its own ladder, with `run --realcheck`; a second gate
   job reads the primary row's `realcheck.json`; everything else of that run waits for
   it.
5. **`tissue`, then `score`**, in the scorer's image (`variants.SCORER_IMAGE`), after
   everything, whatever failed. `score` is given `--drop` for every row a gate said to
   drop, `--gates` (the summary's block "What the gates decided" names every blocked run
   and every dropped row) and, when `REGBENCH_PIPELINE_TRACE` is set, `--pipeline-trace`.

A GPU is asked for only for the runs whose `gpu` column is 1, with the site's own
request string, and the script refuses to submit them without it. An image is given the
GPU (`--nv`) only for those runs. VALIS 1.2.0's image is started with `--writable-tmpfs`.
No job asks for more than 7 days: a run that works per patient gets `REGBENCH_TIME`
times the rounds of the largest patient, capped there.

### What to export

| variable | what | default |
|---|---|---|
| `REGBENCH_CASES` | prepared cases | `$HOME/regbench_cases` |
| `REGBENCH_OUT` | results, logs, gates | `$HOME/regbench_results/mirage-<commit>` |
| `REGBENCH_BIND` | paths to bind into every image, comma-separated: where the cohort's slides, nuclei and the pipeline trace live | none |
| `REGBENCH_WORK` | scratch for a method's intermediates | `$TMPDIR` |
| `REGBENCH_SIF_STARE_REGBENCH`, `REGBENCH_SIF_VALIS_MIRAGE`, `REGBENCH_SIF_VALIS120`, `REGBENCH_SIF_DEEPERHISTREG`, `REGBENCH_SIF_ASHLAR`, `REGBENCH_SIF_PALOM` | the six images; the script refuses to submit when one a selected run needs is missing | none |
| `REGBENCH_SBATCH_GPU` | the site's GPU request, e.g. `--gres=gpu:nvidia_h200:1`; required for the four GPU runs | none, on purpose |
| `REGBENCH_TIME` | per round, every rung | `24:00:00` |
| `REGBENCH_SELFCHECK_TIME` | every selfcheck attempt | `24:00:00` |
| `REGBENCH_ARRAY_LIMIT` | tasks of one array running at once (`%N`) | `20` |
| `REGBENCH_ALLOW_UNCHECKED` | run names that start whatever their gates say (the verdict is still recorded and shown) | none |
| `REGBENCH_PIPELINE_TRACE`, `REGBENCH_PIPELINE_TRACE_LABEL` | a `trace.txt` of a pipeline run on this cohort, and what to call it ("Cost") | none |
| `REGBENCH_MANIFEST` | with `--prepare`: the cohort's `rounds.csv` | none |
| `REGBENCH_TRUTH_PATIENTS` | with `--prepare`: the patients of the semisynthetic known-truth cases, comma-separated | all |
| `REGBENCH_PREPARE_ARGS`, `REGBENCH_PREPARE_FORCE`, `REGBENCH_PREPARE_MEM`, `REGBENCH_PREPARE_TIME` | with `--prepare`: more `prepare multiplex` options; `1` to redo cases that exist; the allocation | none, `0`, `64G`, `24:00:00` |
| `REGBENCH_SCORE_MEM` | memory of the `tissue` and `score` jobs | `64G` |
| `REGBENCH_HEAD_IMAGE` | the image the script itself reads the run list with, on the login node | `STARE_REGBENCH` |
| `REGBENCH_WRITABLE_TMPFS_IMAGES` | images started with `--writable-tmpfs` | `VALIS120` |

Partition, account and node type go after `--` and reach every job, e.g.
`-- --partition=cpu --constraint=<one node type>`. Pin one node type, or the timings
compare CPUs and not methods. The rungs, the 4 CPUs and the 7-day limit are not
variables: they are frozen (`ladder.py`).

### In order, the first time

```bash
cd <checkout of this branch>
source benchmarks/regbench/slurm/env.sh        # after exporting the variables above

# 1. the cohort's list (standard library only: the login node's python will do)
python3 -m benchmarks.regbench.arms_manifest --root <arm_results> --out rounds.csv

# 2. a walking skeleton: ONE patient, the cohort's cases only
awk -F, 'NR==1 {print; next} NR==2 {p=$4} $4==p' rounds.csv > one.csv
REGBENCH_MANIFEST=$PWD/one.csv benchmarks/regbench/slurm/submit_cohort.sh --prepare --datasets multiplex -- <site options>
#    ok: the log ends without a refused case, and
regbench_py STARE_REGBENCH list --cases "$REGBENCH_CASES" --dataset multiplex --sizes
#    prints "<rounds> 1 <rounds>"

# 3. STARE and ASHLAR on that patient: first what would be submitted, then for real
benchmarks/regbench/slurm/submit_cohort.sh --dry-run --runs "stare ashlar" --datasets multiplex -- <site options>
benchmarks/regbench/slurm/submit_cohort.sh --runs "stare ashlar" --datasets multiplex -- <site options>
#    ok: $REGBENCH_OUT/logs/gates/stare.tsv and ashlar.tsv read "<run> ok - 0 -";
#        every $REGBENCH_OUT/<stare|ashlar>/multiplex/<case>/run.json has "ok": true;
#        $REGBENCH_OUT/tables/summary.md exists and its cost block opens with
#        "cores confined: yes"

# 4. the whole cohort: prepare (a NEW $REGBENCH_CASES: the known-truth cases of an older
#    truth spec are refused), look at the dry run, submit
REGBENCH_MANIFEST=$PWD/rounds.csv benchmarks/regbench/slurm/submit_cohort.sh --prepare -- <site options>
benchmarks/regbench/slurm/submit_cohort.sh --dry-run -- <site options> | less
benchmarks/regbench/slurm/submit_cohort.sh -- <site options>
```

`rounds.csv`'s fourth column is the patient (`group`); the `awk` line keeps the first
patient's rounds. `--runs` takes any subset of run names; the selfchecks a subset stands
behind are submitted with it. A run whose results another reads but which is not in the
subset is assumed to be under `$REGBENCH_OUT` already, and the script says so.

### Not verified: check these on the cluster before the real submission

The script was written against SLURM's documented behaviour (`sbatch` and `sacct`
manuals, slurm.schedmd.com `[MEMORY]`: not retrieved while this was written; the port's
plan records an earlier partial reading of the `--dependency`, `--kill-on-invalid-dep`
and array sections). None of the following was observed:

- **A job cancelled for an invalid dependency satisfies `afterany`.** The scorer waits
  `afterany` on every array and every gate; if a cancelled job did not release it, the
  scorer would be cancelled too. Three dummy jobs show it:
  `a=$(sbatch --parsable --wrap 'exit 1'); b=$(sbatch --parsable --dependency=afterok:$a --kill-on-invalid-dep=yes --wrap true); sbatch --dependency=afterany:$b --kill-on-invalid-dep=yes --wrap 'echo ran'`:
  the second must end `CANCELLED` and the third must run.
- **Cores are confined** (`ConstrainCores`): `sbatch -c 4 --wrap 'python3 -c "import os; print(len(os.sched_getaffinity(0)), os.cpu_count())"'` should print `4` and the node's
  count. The cost tables report what each run saw either way.
- **Memory is enforced through cgroups and a killed job is `OUT_OF_MEMORY` in `sacct`**,
  and `sacct` can be called from inside a job (the later rungs ask it for the state of the
  rung below). Without `sacct` a unit whose record still says `killed` is not retried.
- **`MaxSubmit`.** The default submission is 18 gates, 48 selfcheck attempts, the tissue
  and score jobs, and four arrays per run and dataset: with three datasets about 290
  jobs, and array TASKS count toward a per-user limit on many sites. Submit by
  `--datasets` or `--runs` if the limit is lower.
- **`%j`, `%A_%a` in `--output`** and `--export=ALL` carrying the `REGBENCH_*` variables
  into the job, as the two older launchers already assume.

## Caveats that travel with the results

1. **Whose settings.** STARE's settings and the pipeline's VALIS settings were chosen by the
   authors on this cohort with this metric; the other methods run at package defaults. This
   is the largest asymmetry in the comparison.
2. **Two frames**, with the differences quoted above.
3. **Per-patient and pairwise methods** see different problems and fail in different units,
   which is why failures are counted both as launches and as rounds.
4. **Self-consistency, not accuracy** on real slides.
5. **The ladder:** methods need not finish at the same rung, so they need not have had the
   same memory request; the tables show each run's rung.
6. **Known-truth cases** are a check chosen by STARE's authors; see "What these numbers
   are not" under "Known-truth error".
7. **The tissue-level block is secondary**: ACCREDIT's score on same-modality
   fluorescence, outside what it was validated for, by its released code where that
   differs from the preprint's text. It ranks nothing.
8. **Nothing here has run on the cluster.** The cgroup counter path and whether a site
   confines cores are unverified.

### STARE (`methods/stare.py`)

One `stare register` process per pair, on the slides as the case has them
(`ref_image`, `mov_image`), with `--nuclear-index` = the case's `mov_channel` (a case whose
reference has its nuclei at another index is refused: the command takes one index).

- **Parameters.** `TIER_HIGH_STRIDE64` = the `high` row of the STARE table in
  `lib/RegPresets.groovy` (`tile` 2048, `halo` 256, `out_tile` 1024, `max_dim` 1024) plus
  `stride` 64, all five passed explicitly; a test reads the row from that file. The two
  STITCH flags the pipeline always passes are passed too: `--pixel-size` (the case's) and
  `--channel-names` (the case's, else the names in the moving slide's header). `--workers`
  is the allocation; `info["cpu_count"]` records what COARSE's FFTs can use regardless.
- **Stride.** `--opt stride=<64|128|256>`; unset is 64; any other value fails the case.
  With `--label stare_s128` the results are named `stare_s128_rigid` and `stare_s128`.
  The rigid stage does not depend on the stride (COARSE is never given it and SOLVE stores
  the anchor unchanged; `stare/cli.py:79-100`, `stare/stages/solve.py:77-89`,
  `stare/stage_warp.py:85-87` at 1.2.1 `[PARTIAL: cli.py and stage_warp.py in full, main()
  of stages/solve.py and stages/stitch.py, manifest.slide_entry; stages/coarse.py and
  coarse_align.py only searched for "stride" (no match) and for the argument list]`), and a test finds `info["m0"]` and the rigid
  row's warped nuclei identical to the last bit at stride 64 and 256. So a stride run adds
  one result, its final variant; its `_rigid` row repeats `stare_rigid` (with another cost,
  because both variants are yielded after SOLVE).
- **Output.** STARE's own STITCH on the reference grid: no `origin_xy`, nothing padded
  (0 where the moving slide has no pixels). `run.json: info.m0` is the anchor, moving px ->
  reference px.

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

**What the selfcheck reaches.** `SIZES` checks every VALIS row on 2048 x 2560 and on
24000 x 30000 against 23000 x 29000. On the small pair `valis120_hires`'s micro pass
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

Add `--size 2048x2560` for the small pair alone (minutes, a few GB); without it the
large pair runs too. `python -m pytest benchmarks/tests/regbench/test_external_methods.py`
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
grid: 1024 px tiles at a 922 px pitch, all channels, whole pixels, no stage error.

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
the median over tiles of `E - c_ref`), `valid_box` = the reference slide's extent inside
the mosaic (outside it the image is the tiles' zero padding), `reference_file` = the
reference cycle as ASHLAR redrew it. How far single tiles depart from that one origin is
`info["reference_stitch_error"]`; the scored map uses each tile's own placement, not the
median.

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
ASHLAR has no core-count option and none is set. It starts a JVM on import with
`-Xms10m -XX:+UseSerialGC` and no heap limit (`reg.py:29-38`); the reader used here does
not call it.

**Selfcheck: two comparisons against the file ASHLAR wrote, both from the image.**

1. The moving dots through the scored map, against the moving cycle's planes of the
   mosaic located by `origin_xy`. This proves the first step (`L`, `c_mov`, the tile
   index). The per-tile departure of the reference from the median placement is declared
   (`declared_offset: true`), so `origin_xy` cancels out of this comparison.
2. The REFERENCE slide of ASHLAR's selfcheck pair carries its own dots
   (`SELFCHECK_REF_DOTS`). Every dot found in the reference cycle's planes is sent back by
   the second step of the map and must land on a dot of the reference slide; and the
   reported `origin_xy` must equal the origin read off those matched dots
   (`reference.image_origin_px_median_x/y`), within 0.15 px. This proves `E`, `c_ref`, the
   lookup and the origin.

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
# the anchored rows read STARE's matrix from a STARE selfcheck on the same pairs
python -m benchmarks.regbench selfcheck --method stare --work W/stare          # STARE's image
python -m benchmarks.regbench selfcheck --method ashlar --label affine_ashlar \
    --opt affine=stare --mask overlap --m0-from W/stare --work W/affine_ashlar
python -m benchmarks.regbench selfcheck --method ashlar --label affine_ashlar_clean \
    --opt affine=stare --opt noise=none --mask overlap --m0-from W/stare \
    --work W/affine_ashlar_clean
```

`ok` is exit 0 and `"status": "ok"` in `W/<row>/selfcheck.json`, for both pairs (2048 x
2560: 3 x 3 tiles; 4096 x 5120: 5 x 6 tiles with four-tile corners); in each pair's
record, `variants.<row>.reference.status` is `ok` too. A clean row may end `crashed` with
ASHLAR's `correlation > total_amplitude`: that is the known numerical failure, it is
published, and it does not block (`ashlar_clean` is gated on `ashlar`'s check,
`affine_ashlar_clean` on `affine_ashlar`'s: the point-map code is the same).

### DeeperHistReg (`methods/deeperhistreg.py`)

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
`valid_box` bounds the pixels that come from the moving slide. A failed writer leaves
both rows `ok` with `native.error`.

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
detections within 8 px of such an edge; the affine stage has none. Pairs: 8192 x 10240
twice (8 x 10 blocks of 1024 px; the second with a smaller moving slide, so the two
thumbnail down-factors differ). `palom` reads files with 1024 px tiles; `palom_b1024`
reads files with 2048 px tiles, the cohort's layout, and cuts them to 1024. If under 10 %
of the blocks have a non-zero shift the `palom` variant is `not_checked`: the block map
then equals the affine map and a wrong sign or block order could not show.

What the dots cannot see, and what covers it:

- a block edge one pixel off: those pixels are on the seams the mask leaves out. The
  pixel-for-pixel test draws a ramp through PALOM's own
  `block_affine_transformed_moving_img` and compares every output pixel with the map;
- content PALOM draws twice or never at a seam (a backward check counts neither);
  `seam_jump_px_*` and `band_fraction` say how much is at stake;
- the cohort's 2048 px blocks for the `palom` row: the realcheck is the only check there.

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
pairs, `n_masked` a few percent of `n_detected` for `<row>` and 0 for `<row>_affine`.
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
`test_selfcheck.py` runs `stare register` once for real, on a 1024 x 1280 pair;
`test_stare_adapter.py` runs STARE's whole selfcheck (both pairs) and two real registrations.
