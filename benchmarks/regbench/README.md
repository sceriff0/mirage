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
| datasets `synthetic`, `semisynth`, `multiplex` | here. The frozen known-truth cases (`--suite truth`, `--truth`) are real patients with two rounds; in the copied suites each case is still its own patient, and the cohort's cases are not built yet |
| known-truth error per deformation family (`truth.py`, `truth_spec.json`) | here; the frozen 8192 px cases have not been run through any method |
| adapters `initial`, `stare`, `valis`, `deeperhistreg` | converted to the contract only. They still compute what the copied harness computed (for example VALIS and DeeperHistReg are still asked for a moving -> reference map). The native adapters of "Methods" below replace them. `stare` already writes its native image: its own STITCH, deferred until the clock has stopped |
| selfcheck and realcheck (`selfcheck.py`) | here. Really run for `stare` only (its package is the one installed where this was written); for every other method the check has only been exercised with fake methods |
| ASHLAR, the cohort's cases, the run list and the cohort launcher | not here yet |

Nothing here has run on the cluster.

## Layout

```
mapping.py     PointMap: a point mapping and the direction it was estimated in
cases.py       the on-disk contract: case.json + points.npz in, warped.npz + run.json out
cells.py       segment nuclei on a native slide (the built-in segmenter, label images)
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

**The driver** (`methods.run_case`, `methods.run_group`), at each yield, before it resumes
the generator and with the clock paused: moves the nuclei through the map (the moving
nuclei for `mov_to_ref`, the reference nuclei for `ref_to_mov`), writes `warped.npz`
(`cells_xy`, `direction`, `band` when `info["band_mov"]` is given, `truth_xy` = the map
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
| `stare_rigid`, `stare` | pinned `stare-registration` | pair | affine + mesh, moving -> reference | moving nuclei through `stage_warp.make_warper`; reference frame |
| `valis_mirage_rigid`, `valis_mirage_nonrigid`, `valis_mirage` | VALIS 1.0.0 through the pipeline's own call | patient | affine + backward field `bk_dxdy` | reference nuclei through VALIS's own point warp; moving frame |
| `valis120_rigid`, `valis120`, `valis120_micro` | VALIS 1.2.0 from PyPI, upstream defaults | patient | same | same |
| `valis120_hires_*` | VALIS 1.2.0, upstream's high-resolution example | patient | same | same |
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
fixed by `truth_spec.json`, which carries the SHA-256 of its own content.

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
| `bumps100` | 0.2 to 0.6 sigma | sigma 100 px, 20 bumps | the benchmark's choice; inside STARE's model |
| `bumps25` | 0.2 to 0.6 sigma | sigma 25 px, 60 bumps | **outside STARE's model** (finer than its 64 px lattice) |
| `seams` | s.d. 3 px | 1536 px blocks | **outside STARE's model** (discontinuous) |

The 1536 px block is neither STARE's 2048 px tile nor ASHLAR's 922 px tile pitch. It is a
multiple of STARE's 64 px lattice, and in an 8192 px window one seam (6144 px) falls on a
STARE tile edge. Bumps cover a small part of an 8192 px window, so their effect shows in
`tre_um_max` and the shares under a threshold, hardly in the median.

**Poses**, one per round: `shift` is (30, -20) px with no rotation; `rot` is 2 degrees plus
that shift. ASHLAR has no rotation, so its `rot` rows measure that limit and are read
beside its `shift` rows.

**Cases.** Windows of 8192 px for every method (4 x 4 STARE tiles). A patient is
(window, family): one reference, kept as one file, and two rounds. `synthetic`: one
Gaussian-blob scene per family (7 patients, 14 cases). `semisynth`: 2 windows from each of
the 2 cohort patients the spec names, variant `same`, noise 1.0 x the background s.d.
(28 patients, 56 cases). A per-pair method runs each round; a per-patient method gets the
patient and so never sees two families at once.

```bash
python -m benchmarks.regbench prepare synthetic --suite truth --cases CASES
python -m benchmarks.regbench prepare semisynth --truth --cases CASES \
    --name <a patient the spec names> --image <its reference round> --channel <nuclear>
```

**The freeze.** `prepare` refuses a `truth_spec.json` whose content does not match its
hash and writes the hash into every `case.json` it makes from it
(`extra.truth_spec_sha256`); `score` refuses cases of two hashes. The hash covers the
parameters, not the generator's code: the commit is what fixes that, and it is to be tagged
`regbench-truth-frozen-<date>` before any method runs.

**Numbers** (`truth.error_stats`), per case and variant, in micrometres: `tre_um_p50`,
`tre_um_p90`, `tre_um_max`, `frac_under_1um`, `frac_under_2um`, `initial_tre_um_p50` (the
unregistered pose) and `frac_improved`, with `n_probes` and `n_nonfinite`. A point a method
returns as NaN counts against it: its error is infinite. A failed run is scored at the
unregistered pose. Without a pixel size the micrometre numbers are NaN. A variant with
`band` gets a second row `<variant>:nonband` on the probes outside bands.

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
| `cells.csv` | case x variant | `dataset, tier, case_id, group, family, pose, method, frame, selfcheck_status`, every key of `pairs.RECORD_KEYS`, `coverage, nn_distance_um_p50, fold_pct, sd_log_jac, imputed_initial, problem, cause`. A variant whose `warped.npz` has `band` also gets a row `<variant>:nonband`, scored without the moving nuclei in bands |
| `summary_cells.csv` | dataset x variant x subset | per patient the median over rounds, then median, min, max over patients; `n_patients, n_cases, n_ok`; the minimum over cases; each headline beside its null; Dice over the cases that clear the null; displacement over the readable cases with `n_readable`. `subset=common` repeats it on the cases every primary variant registered |
| `paired_vs_primary.csv` | patient x variant | the difference of each headline against the `stare` row |
| `failures.csv` | variant | launches made, launches failed by cause, rounds lost |
| `resources.csv` | case x variant | the cost recorded at that variant's yield |
| `summary_resources.csv` | patient x variant | sum over rounds for a pair method, the launch's total for a per-patient one; peaks are maxima, never divided by rounds |
| `truth.csv` | known-truth case x variant | `dataset, tier, case_id, group, family, pose, in_stare_model, truth_spec, method, frame`, the numbers of "Known-truth error", `fold_pct, sd_log_jac, imputed_initial, problem, cause`; plus the pseudo-rows `initial` and `truth` (zero by construction) |
| `truth_by_family.csv` | dataset x family x pose x variant | `n_cases, n_ok`, median and maximum of p50, median p90, maximum of max, median shares under 1 and 2 µm, median `fold_pct`. No row over families |
| `summary.md` | | the summary, with the line "descriptive; no ranking is claimed" under six patients, and the known-truth table |

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

## Tests

```bash
python -m pytest benchmarks/tests/regbench -q --basetemp=/tmp/rb_<unique>
```

`test_external_methods.py` skips where VALIS or DeeperHistReg is not importable.
`test_selfcheck.py` runs `stare register` once for real, on a 1024 x 1280 pair.
