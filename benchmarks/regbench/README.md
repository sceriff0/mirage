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
| adapters `initial`, `deeperhistreg` | converted to the contract only. `deeperhistreg` still computes what the copied harness computed (it is still asked for a moving -> reference map). The native adapter of "Methods" below replaces it |
| selfcheck and realcheck (`selfcheck.py`) | here. Really run for `stare` only (its package is the one installed where this was written); for every other method the check has only been exercised with fake methods |
| adapter `ashlar` (ASHLAR 1.20.0, its own command per patient: `ashlar`, `ashlar_clean`, `affine_ashlar`, `affine_ashlar_clean`) | here, as in "ASHLAR" below. **Written against ASHLAR's source and never run with ASHLAR** (it is not installed where this was written): tested with a recording stand-in that draws the mosaic the way ASHLAR's writer is written to. Its first real run is the selfcheck inside `labsyspharm/ashlar:1.20.0` |
| off-seam rows (`<variant>:non_<owner>`, `nonband_*.csv`) | here for ASHLAR's moving-side mask; the driver stores a reference-side mask (`band_ref`) and the scorer applies it to the nuclei, for PALOM's task |
| the run list and the cohort launcher | not here yet |

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
slurm/         sbatch wrappers (first rung of the ladder; no escalation yet)
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
nucleus; `truth_xy` = the map
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
| `stare_s128`, `stare_s256` | the same, `--opt stride=128` / `256` and nothing else (a sensitivity analysis, never primary) | pair | same | same |
| `valis_mirage_rigid`, `valis_mirage_nonrigid`, `valis_mirage` | VALIS 1.0.0 through the pipeline's own call | patient | affine + backward field `bk_dxdy` | reference nuclei (pixels of the written image's grid) through VALIS's `warp_tools.warp_xy_inv`; moving frame |
| `valis120_rigid`, `valis120`, `valis120_micro` | VALIS 1.2.0 from PyPI, upstream defaults | patient | same | same |
| `valis120_hires_rigid`, `valis120_hires_nonrigid`, `valis120_hires` | VALIS 1.2.0, upstream's high-resolution example (`--opt hires=1`) | patient | same | same |
| `deeperhistreg_*` | DeeperHistReg 1.0.1 | pair | backward field on the reference grid | reference nuclei through its `warp_landmarks`; moving frame. Each also has an `_asdrawn` row, the image as written |
| `ashlar`, `ashlar_clean`, `affine_ashlar`, `affine_ashlar_clean` | ASHLAR 1.20.0, its own command, on tiles cut from stitched slides | patient | one shift per tile | moving nuclei by their tiles' shifts; reference frame |
| `initial` | none | | identity | |

Honest limits of "natively": ASHLAR is run on tiles cut from stitched slides, not raw tiles;
DeeperHistReg is run on fluorescence with brightfield defaults; STARE is run as one process,
not as the pipeline's cluster jobs; `valis120` at upstream defaults is upstream's
low-resolution pipeline.

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
adapters also on a pair with a 30000 px side, DeeperHistReg on pairs whose sides are
multiples of 20.

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
3800), so a lower count inside the 20 to 30 % range does not remove it. The seeds of the
semi-synthetic cases depend on the patients' ids, so each known-truth case records what
its own map does (`case.json`: `extra.field` = `moved_frac_over_1um`, `min_jacobian`,
`fold_pct` on a lattice every 8 px), and the summary's known-truth header lists any case
whose true map folds. The test
`test_bumps_move_about_a_quarter_of_the_window_without_folding` measures both families at
the frozen size and holds them to the fraction the spec records, within 0.02.

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
`nonband_truth.csv`, on the probes of the moving nuclei the mask leaves (see "Off the
seams").

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
| `nonband_missing.csv` | dataset x owner | `n_cases`, and `n_cases_without_mask`: the cases the owner's primary run did not register, which have no off-seam row for any method |
| `summary_cells.csv` | dataset x variant x subset | per patient the median over rounds, then median, min, max over patients; `n_patients, n_cases, n_ok`; the minimum over cases; each headline beside its null; Dice over the cases that clear the null; displacement over the readable cases with `n_readable`. `subset=common` repeats it on the cases every primary variant registered |
| `paired_vs_primary.csv` | patient x variant | the difference of each headline against the `stare` row |
| `failures.csv` | variant | launches made, launches failed by cause, rounds lost |
| `resources.csv` | case x variant | the cost recorded at that variant's yield |
| `summary_resources.csv` | patient x variant | sum over rounds for a pair method, the launch's total for a per-patient one; peaks are maxima, never divided by rounds |
| `truth.csv` | known-truth case x variant | `dataset, tier, case_id, group, family, pose, in_stare_model, truth_spec, method, frame`, the numbers of "Known-truth error", `fold_pct, sd_log_jac, imputed_initial, problem, cause`; plus the pseudo-rows `initial` and `truth` (zero by construction) |
| `truth_by_family.csv` | dataset x family x pose x variant | `n_cases, n_ok`, median and maximum of p50, median p90, maximum of max, median shares under 1 and 2 µm, median `fold_pct`. No row over families |
| `summary.md` | | the summary, with the line "descriptive; no ranking is claimed" under six patients, and the known-truth table |

### Off the seams

A method that builds its image from tiles or blocks draws some nuclei across a join, and
its point map is not exactly its image there. Such a method (`score.SEAM_OWNERS`: `ashlar`
now, PALOM's two rows when its adapter lands) writes a **seam mask** with its primary row:
one bool per nucleus, from its own geometry alone (ASHLAR: the tile grid and the positions
it accepted; never any method's scores). ASHLAR's is on the moving nuclei
(`info["band_mov"]`, `band` in `warped.npz`); a reference-side mask is `band_ref`.

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
  `previous_attempts` and reported as `wasted_*`. The launcher that escalates is not here
  yet; the driver reads `REGBENCH_ATTEMPT` and `REGBENCH_MEM_GB`.
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
7. **Nothing here has run on the cluster.** The cgroup counter path and whether a site
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

**Three rows, one patient per run** (`run --group`, `selfcheck`):

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

## Tests

```bash
python -m pytest benchmarks/tests/regbench -q --basetemp=/tmp/rb_<unique>
```

`test_external_methods.py` skips where VALIS (at the pinned version of the row) or
DeeperHistReg is not importable. `test_valis_adapters.py` runs both VALIS adapters against
`fake_valis.py`, a stand-in that records every call and draws images the way VALIS's
writer is written to; for `valis_mirage` it executes the real `bin/register.py` on that
stand-in, in a fresh process.
`test_ashlar_adapter.py` runs the ASHLAR adapter against `fake_ashlar.py`, the same kind of
stand-in (ASHLAR's reader, positions and paste rule transcribed from its source at
v1.20.0); the one test that needs ASHLAR itself is in `test_external_methods.py`.
`test_selfcheck.py` runs `stare register` once for real, on a 1024 x 1280 pair;
`test_stare_adapter.py` runs STARE's whole selfcheck (both pairs) and two real registrations.
