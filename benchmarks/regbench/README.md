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
| datasets `synthetic`, `semisynth`, `multiplex` | here, as copied; each case is its own patient until the known-truth and cohort cases are built |
| adapters `initial`, `stare`, `valis`, `deeperhistreg` | converted to the contract only. They still compute what the copied harness computed (for example VALIS and DeeperHistReg are still asked for a moving -> reference map). The native adapters of "Methods" below replace them |
| known-truth error per deformation family, selfcheck and realcheck, ASHLAR, the cohort's cases, the run list and the cohort launcher | not here yet |

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
is `ref` or `mov`), read back with `cases.polys(points, side)`.

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
(`cells_xy`, `direction`, `band` when `info["band_mov"]` is given, and the map on a lattice
for the Jacobian statistics), writes `run.json`, and calls `on_variant`.

**`run.json`**: `ok`, `error`, `cause`, `problem`, `direction`, `launch_id`, `unit`
(`pair` | `patient`), `group_cases`, `register_s`, `write_s`, `resources`, `native`,
`selfcheck_status`, `attempt`, `mem_requested_gb`, `cpus_requested`, `cpus_affinity`,
`blas`, `cpu_model`, `previous_attempts`, `info`. It is written as
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

### Tables (`<out>/tables/`)

| file | one row per | what |
|---|---|---|
| `cells.csv` | case x variant | `dataset, tier, case_id, group, family, pose, method, frame, selfcheck_status`, every key of `pairs.RECORD_KEYS`, `coverage, nn_distance_um_p50, fold_pct, sd_log_jac, imputed_initial, problem, cause`. A variant whose `warped.npz` has `band` also gets a row `<variant>:nonband`, scored without the moving nuclei in bands |
| `summary_cells.csv` | dataset x variant x subset | per patient the median over rounds, then median, min, max over patients; `n_patients, n_cases, n_ok`; the minimum over cases; each headline beside its null; Dice over the cases that clear the null; displacement over the readable cases with `n_readable`. `subset=common` repeats it on the cases every primary variant registered |
| `paired_vs_primary.csv` | patient x variant | the difference of each headline against the `stare` row |
| `failures.csv` | variant | launches made, launches failed by cause, rounds lost |
| `resources.csv` | case x variant | the cost recorded at that variant's yield |
| `summary_resources.csv` | patient x variant | sum over rounds for a pair method, the launch's total for a per-patient one; peaks are maxima, never divided by rounds |
| `summary.md` | | the summary, with the line "descriptive; no ranking is claimed" under six patients |

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
6. **Nothing here has run on the cluster.** The cgroup counter path and whether a site
   confines cores are unverified.

## Tests

```bash
python -m pytest benchmarks/tests/regbench -q --basetemp=/tmp/rb_<unique>
```

`test_external_methods.py` skips where VALIS or DeeperHistReg is not importable.
