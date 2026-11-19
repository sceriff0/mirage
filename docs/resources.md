# Resources

<p class="standfirst">What every process actually asks the scheduler for, where that number comes from,
how it grows on a retry, and what clamps it. Read this before sizing a cluster allocation or
diagnosing an out-of-memory kill.</p>

!!! abstract "Canonical sources"
    - **Per-process requests** — `conf/modules.config` (`withLabel:` and `withName:` blocks)
    - **Fallback defaults & retry policy** — `conf/base.config`
    - **Global ceilings** — `nextflow.config` (`params.max_*`, `process.resourceLimits`)

---

## The one-owner rule

A process's `cpus` / `memory` / `time` come from **either** a resource `label`
**or** a `withName:` block in `conf/modules.config` — never both.

`withName:` wins over `withLabel:`, so a label on a fully-overridden process is
inert and misleading; those have been removed. What remains is three cases:

<div class="gate">
  <div class="g"><div class="k">case 1</div><div class="v">withName owns all three</div><div class="d">No label. The block sets cpus, memory and time. 16 processes.</div></div>
  <div class="g"><div class="k">case 2</div><div class="v">label owns all three</div><div class="d">The withName block, if any, sets only publishDir / ext.args. 8 processes.</div></div>
  <div class="g"><div class="k">case 3</div><div class="v">partial override</div><div class="d">withName sets one or two fields; a label supplies the rest. 6 processes.</div></div>
</div>

**These three counts cover `modules/local/*.nf` only**, because that is what
`tests/test_resource_label_coverage.py` scans and the counts are checked against
that scan — raising one to include a process outside `modules/local/` makes the
build fail. One process is therefore outside all three numbers: **`BASICPY`**,
in `modules/nf-core/basicpy/`, which is a **case-3 partial override** — upstream's
`label 'process_single'` with `memory` raised to `32 GB × attempt` by its
`withName:` block. Nothing guards that sentence, so check it by hand if you edit
either side.

Case 3 is the one that surprises people: `TILED_SOLVE` carries
`process_single` **and** a `withName:` block, but that block sets `memory`
only — so the label still owns its `cpus` and `time`. All four tiled/STARE
processes work this way, and so do `GENERATE_REGISTRATION_QC` (`withName` sets
`cpus` and `memory`; the `process_high` label still owns `time`) and
`EXPORT_SPATIALDATA` (`withName` sets `time` alone).

`EXPORT_SPATIALDATA` is also the one process in the repo whose `label` is an
*expression* rather than a literal —
`params.spatialdata_include_image ? 'process_high' : 'process_medium'`
(`modules/local/export_spatialdata.nf:22`) — so its `cpus` and `memory` follow
that flag.

---

## Resource labels

The four labels defined in `conf/modules.config`. Every value scales with
`task.attempt`, so attempt 2 is the second column, attempt 3 the third, up to
`maxRetries = 3`.

| Label | `cpus` | `memory` | `time` |
|---|---|---|---|
| `process_single` | `1` | `12.GB × attempt` | `8.h × attempt` |
| `process_low` | `2` | `32.GB × attempt` | `2.h × attempt` |
| `process_medium` | `4` | `100.GB + 100.GB × attempt` | `4.h × attempt` |
| `process_high` | `8` | `200.GB + 100.GB × attempt` | `12.h × attempt` |

Note the `+` in `process_medium` and `process_high`: the first attempt already
gets 200 GB and 300 GB respectively, and each retry adds 100 GB rather than
doubling.

### Fallback defaults

A process with neither a label nor a `withName:` field for a given resource
falls through to `conf/base.config`:

| Resource | Default |
|---|---|
| `cpus` | `1 × attempt` |
| `memory` | `6.GB × attempt` |
| `time` | `4.h × attempt` |

**No process currently relies on this.** Every process has a `label` or a
`withName:` field covering each of `cpus` / `memory` / `time`, and
`tests/test_resource_label_coverage.py` fails the build if one does not. The
table above is what a *new* process would silently get if it were added with
neither — which is why that guard exists.

---

## Per-process requests

Effective values on **attempt 1**. `f` denotes the relevant input size in GiB
(rounded down, minimum 1).

### Preprocessing

| Process | `cpus` | `memory` (attempt 1) | `time` | Owner |
|---|---|---|---|---|
| `CONVERT_IMAGE` | `1` | `24 GB` + tier: `f<4` → +0, `f<12` → +24, `f<24` → +48, else +64 GB | `2.h × attempt` | `withName` |
| `TILE_FOR_BASIC` | `2` | derived from `preproc_tile_size`, `× attempt` *(withName)* — floor 6 GB | `2.h × attempt` | `withName` |
| `APPLY_PROFILES` | `2` | derived from `preproc_tile_size`, `× attempt` *(withName)* — floor 8 GB | `3.h × attempt` | `withName` |
| `GENERATE_PREPROCESS_QC` | `4` | `200 GB` | `4.h × attempt` | `process_medium` |

`BASICPY` is **not** in the table above and cannot be: the guard behind these tables
(`tests/test_resource_label_coverage.py`) reads `modules/local/*.nf`, and `BASICPY` lives in
`modules/nf-core/basicpy/`, vendored unmodified. Its resources are its upstream
`label 'process_single'` (1 cpu, `8.h × attempt`) with `memory` raised to `32 GB × attempt`
by a partial `withName:` override in `conf/modules.config` — it runs Bio-Formats under a JVM
and materialises one channel's tile stack, which does not fit the single tier's 12 GB on a
real slide. Changing either number changes an unguarded figure, so change it here too.

### Registration — VALIS

| Process | `cpus` | `memory` (attempt 1) | `time` | Owner |
|---|---|---|---|---|
| `REGISTER` | `8` | `64 GB × 2^(attempt − 1)` (64 → 128 → 256 → 512 GB); at `reg_valis_mode hires` `128 GB × 2^(attempt − 1)` (128 → 256 → 512 → 1024 GB) | `24.h × attempt` | `withName` |

`REGISTER` pins its own retry budget (at most four attempts, so the ramp stops at its fourth rung) rather than inheriting a profile's, and every rung is still clamped by `max_memory`.
The first rung is sized per mode from peak RSS measured per case on the 8-case regbench
cohort (VALIS 1.2.0, 4 CPUs): `default` median 31.6 GB, max 51.4 → 64 GB; `hires` median
81.7 GB, max 106.4 → 128 GB, the smallest rung of the same doubling ramp above the cohort's max.

`REGISTER` runs valis-wsi 1.2.0 at its defaults (rigid 512 px, non-rigid 2048 px, VALIS's own
detector and matcher, no micro-registration), or at `reg_valis_mode hires` its high-resolution
example (`MicroRigidRegistrar`, then `register_micro` at 0.25 × the smallest slide's longest
side), with a fixed 10 GB Bio-Formats JVM heap at every
attempt (`reg_jvm_heap_gb` overrides it).

`REGISTER` also carries a per-process `maxForks` cap of 10 (in `nextflow.config`'s
concurrency block, after the profiles) and its own error strategy — see
[Retry policy](#retry-policy) and [Execution & concurrency](#execution-concurrency).

### Registration — tiled / STARE

The default backend (`registration_method = 'tiled'`). Small everywhere: the tiled backend is JVM-free and tile-streamed, so every step needs a few GB
at most even for large slides. `TILED_COARSE`'s anchor is an FFT NCC rotation sweep (ORB
fallback) on a thumbnail bounded by `reg_tiled_coarse_max_dim`: 0.25–0.43 GB peak RSS measured
for the whole stage at 512–2048 px, so the row below asks 2 GB at the 1024 px default.

| Process | `cpus` | `memory` (attempt 1) | `time` | Owner | `maxForks` |
|---|---|---|---|---|---|
| `TILED_COARSE` | `2` *(label)* | derived from `reg_tiled_coarse_max_dim`, `× attempt` *(withName)* — 2 GB at defaults | `2.h × attempt` *(label)* | partial | `20` |
| `TILED_REG_TILE` | `2` *(label)* | derived from `reg_tiled_tile` + 2×max(`reg_tiled_halo`, 3×`reg_tiled_stride`), `× attempt` *(withName)* — 4 GB at defaults | `2.h × attempt` *(label)* | partial | `20` |
| `TILED_SOLVE` | `1` *(label)* | `2 GB × attempt` *(withName)* | `8.h × attempt` *(label)* | partial | — |
| `TILED_STITCH` | `4` *(label)* | derived from `reg_tiled_out_tile`, `× attempt` *(withName)* — 4 GB at defaults | `4.h × attempt` *(label)* | partial | `10` |

`TILED_COARSE` / `TILED_REG_TILE` / `TILED_SOLVE` / `TILED_STITCH` are the STARE method —
the only shape it has.

`TILED_REG_TILE`, `TILED_STITCH`, `TILE_FOR_BASIC`, `APPLY_PROFILES` and
`MERGE_AND_PYRAMID` are the
processes whose memory
request is **derived from a parameter** instead of being a constant: the first
scales with `reg_tiled_tile + 2 × max(reg_tiled_halo, 3 × reg_tiled_stride)`, the second with
`reg_tiled_out_tile`, each the measured linear fit doubled and floored at 4 GB.
Raising a tile size therefore raises the reservation rather than producing a
SIGKILL. The arithmetic is written out inside each process' own
`memory = { … }` closure in `conf/modules.config`, immediately under the block
comment that derives it; the two closures are near-identical and that duplication
is forced — Nextflow 26's strict config parser rejects a function declaration in
a config file, so there is no legal way to share a helper between them.

`MERGE_AND_PYRAMID` does not hold the slide: it streams the base resolution from a generator of tiles, so its request is built from **one
decoded plane** — estimated as 4× the largest single channel file, since a
config closure cannot know the compression ratio — plus the pyramid levels that
must stay resident while `tifffile` fills the SubIFDs one level at a time. That
second term is a geometric series in `pyramid_scale` and disappears below three
`pyramid_resolutions`, which is why both parameters appear in the row. Floor 8 GB.

Both `APPLY_PROFILES`' write-tile-buffer term and `MERGE_AND_PYRAMID`'s
decoded-plane term assume tifffile's compressor pool is pinned to `maxworkers=1`
on the write itself (`bin/apply_basic_profiles.py`,
`bin/merge_channels_pyramid.py:826-847`) — without that pin, the container's
tifffile version reintroduces a term that scales with the channel count, which
these figures do not budget for. See the `maxworkers=1` comment in each
process' `conf/modules.config` block for the measured numbers.

The 4× is a **floor estimate, not a bound**, and the block comment in
`conf/modules.config` records the measured counterexamples: zlib at
SPLIT_CHANNELS' settings reaches 4.2× on a plane that is 75 % true-black
background, and 796× on a near-empty channel. A WSI is mostly empty glass and
every channel shares that background, so taking the largest file does not rescue
the estimate. What backstops it is `conf/base.config`'s exit-137 retry with
`maxRetries = 3` against a request that is multiplied by `task.attempt` — four
attempts cover a shortfall of up to 4×, and beyond that the task fails loudly.
Measure the real ratio against a production `channels/` directory and raise the
coefficient if one becomes available.

The "4 GB at defaults" figures above are the shipped-default evaluation of those
formulas, not independent constants — the **parameter names**, not the numbers,
are what `tests/test_resource_label_coverage.py` checks for all five of these
param-derived rows.

The memory of the STARE tile and resampling tasks is bounded by a window, not by
the slide. Measured peak RSS on uint16 slides (macOS arm64):
`TILED_REG_TILE` 1.27 GB at the default stride of 64 px (a 2432² px read window;
1.65 GB at stride 128, 2.17 GB at 256) and `TILED_STITCH` 1.42 GB at the 1024 px
default output tile — each set by a parameter (`reg_tiled_tile` +
`reg_tiled_stride` / `reg_tiled_halo`, `reg_tiled_out_tile`) rather than by slide
dimensions. `TILED_SOLVE` is the exception: it holds one displacement vector per
lattice node plus lattice-sized working arrays, so its memory grows with slide
area — 0.19–1.20 GB per round pair on the paper's eight-case cohort, which is why
it asks a fixed 2 GB. `TILED_COARSE` is bounded by a parameter too, by `reg_tiled_coarse_max_dim`, and is
small again: its FFT NCC rotation-sweep anchor measured 0.25 GB at 512 px,
0.43 GB at 1024 and 0.40 GB at 2048 on a 16384² tiled OME-TIFF pair.

### Registration QC

| Process | `cpus` | `memory` (attempt 1) | `time` | Owner |
|---|---|---|---|---|
| `GENERATE_REGISTRATION_QC` | `1` *(withName)* | tier on `registered + native + reference`: `f<20` → 100, `f<50` → 200, else 300 GB, `× attempt` | `12.h × attempt` *(label `process_high`)* | partial |
| `SEG_QC_SEGMENT` | `8` | tier on image: `f<10` → 32, `f<30` → 64, else 128 GB, `× attempt` | `4.h × attempt` | `withName` (`SEGMENT`'s, matched via the alias) |
| `SEG_QC_GEOJSON` | `1` | `64 GB × attempt` | `4.h × attempt` | `withName` |
| `WARP_SEG_QC` | `2` | `32 GB × 2^(attempt−1)` → 32 / 64 / 128 / 256 | `3.h × attempt` | `withName` |

`GENERATE_REGISTRATION_QC` is the one process whose tier is keyed on the
**combined** size of *three* inputs (the registered image, its native
pre-registration counterpart, and the reference), not on a single file — it
holds all three to build the before/after pair. Per full-resolution pixel it
carries three float32 planes plus a six-plane uint8 two-panel composite, about
1.6× what the single-panel version held; summing the native's bytes into the
tier is what tracks that. An input read but not summed here is a silent
under-request, which is why `tests/test_registration_qc_wiring.py` fails when
the closure omits one of the process's `path()` inputs.

`WARP_SEG_QC` uses a *doubling* ramp rather than a linear one. It scores each
nucleus pair inside its own bounding box, so peak RAM is one nucleus regardless of
slide size. 32 GB is generous, and the
ramp exists only for pathological inputs. Runtime, not memory, is the binding
constraint.

### Segmentation

| Process | `cpus` | `memory` (attempt 1) | `time` | Owner |
|---|---|---|---|---|
| `SEGMENT` | `8` | tier: `f<10` → 32, `f<30` → 64, else 128 GB, `× attempt` | `4.h × attempt` | `withName` |
| `EXTRACT_CELL_PROPERTIES` | `1` | `64 GB × attempt` | `12.h × attempt` | `withName` |
| `EXTRACT_NUCLEI_PROPERTIES` | `1` | `64 GB × attempt` | `12.h × attempt` | `withName` |
| `EXTRACT_MASK_SERIES` | `2` | `32 GB × attempt` | `2.h × attempt` | `process_low` |
| `SEG_QUALITY_EVAL` | `8` | tier on image: `f<10` → 128, `f<30` → 256, else 448 GB, `× attempt` | `4.h × attempt` | `withName` |

`SEGMENT` asks for 8 CPUs so a CPU-only path — and the CPU-bound label expansion
and Dask tiling either side of inference — stays tolerable. GPU inference is
unaffected by that number.

`SEG_QUALITY_EVAL` is opt-in (`-params-file params/seg_quality_eval.json`) and sized off the
image rather than the mask: CSE's cost is driven by per-pixel index structures on
the DECOMPRESSED masks, so a well-compressed WSI needs far more RAM than its file
size suggests. `--cse_max_pixels` bins the input to cap that; both it and
`SEG_QUALITY_EVAL` retry three times before dropping, and the drop is logged
rather than silent — see [Retry policy](#retry-policy).

### Postprocessing

| Process | `cpus` | `memory` (attempt 1) | `time` | Owner |
|---|---|---|---|---|
| `SPLIT_CHANNELS` | `1` | tier: `f<5` → 32, `f<15` → 64, else 128 GB, `× attempt` | `2.h × attempt` | `withName` |
| `QUANTIFY` | `1` | `128 GB × attempt` | `12.h × attempt` | `withName` |
| `NUCLEAR_RETENTION` | `1` | `128 GB × attempt` | `12.h × attempt` | `withName` |
| `MERGE_QUANT_CSVS` | `2` | `32 GB × attempt` | `2.h × attempt` | `process_low` |
| `CELL_QC` | `2` | `32 GB × attempt` | `2.h × attempt` | `process_low` |
| `EXPORT_GEOJSON` | `1` | `32 GB × attempt` | `2.h × attempt` | `withName` |
| `MERGE_AND_PYRAMID` | `2` | derived from the largest single channel file + `pyramid_resolutions` and `pyramid_scale`, `× attempt` *(withName)* — floor 8 GB | `8.h × attempt` | `withName` |
| `EXPORT_SPATIALDATA` | `4` *(label)* | `200 GB` *(label)* | `4.h × attempt` *(withName)* | partial |
| `GENERATE_POSTPROCESSING_QC` | `4` | `200 GB` | `4.h × attempt` | `process_medium` |

`EXPORT_SPATIALDATA`'s label is chosen at runtime —
`params.spatialdata_include_image ? 'process_high' : 'process_medium'`. The row
above is the default (`false` → `process_medium`); with `--spatialdata_include_image`
it asks for `8` cpus and `300 GB` instead.

#### `MERGE_AND_PYRAMID`'s memory coefficient is unmeasured

The `memory` closure's `plane * 3.25d` term (`conf/modules.config`, the
`MERGE_AND_PYRAMID` block) was calibrated once, on one host, and the closure's own
comment calls part of it "a guess made without one". This is a **node-memory cliff, not
cgroup pressure** — a large-slide run that under-reserves gets SIGKILLed at roughly the
observed ~450 GB ceiling, and the retry ramp (`× task.attempt`, capped at 4 attempts by
`conf/base.config`'s `maxRetries`) is the only safety net.

`workflows/mirage.nf` logs a `log.warn` at launch whenever the run reaches
`MERGE_AND_PYRAMID` — gated on the same `run_postprocessing` boolean
(`ParamUtils.shouldRun('postprocessing', ...)`, against the `ParamUtils.STEPS` table) that
routes the standard start/stop path, so it also fires under `mode=add_cycle`, which
reaches the same process through `ASSEMBLE_EXPORT` without ever setting `--start`/`--stop`.

**To measure the real coefficient on your own data**, run one representative real slide
through `SPLIT_CHANNELS` (or use an existing run's per-channel TIFFs) and, per channel:

1. Read `H` and `W` (pixel height/width) from the channel TIFF.
2. Read the on-disk `file_size` (bytes) of that channel's compressed TIFF from the
   Nextflow trace (`rchar`/`wchar`, or just `stat` the published file).
3. Compute `r = (H * W * 2) / file_size` — the ratio of one uncompressed uint16 plane to
   the compressed file size.
4. Report the **observed maximum** `r` across channels and across slides in the cohort;
   that maximum, not the mean, is what should replace the `3.25d` guess once there is
   real data behind it. Until then, set `--max_memory` generously above the current
   estimate for any slide over ~40 GB.

### Run-level

| Process | `cpus` | `memory` (attempt 1) | `time` | Owner |
|---|---|---|---|---|
| `PREFLIGHT_SCALE` | `1` | `12 GB × attempt` | `8.h × attempt` | `process_single` |
| `GENERATE_QC_REPORT` | `2` | `32 GB × attempt` | `2.h × attempt` | `process_low` |
| `AGGREGATE_SIZE_LOGS` | `1` | `12 GB × attempt` | `8.h × attempt` | `process_single` |
| `MERGE_SEG_EVAL` | `1` | `4 GB × attempt` | `1.h × attempt` | `withName` |

---

## Global ceilings

Every request is clamped by `process.resourceLimits`, which reads three
parameters:

| Parameter | Default | Effect |
|---|---|---|
| `max_cpus` | *(required, no default)* | Upper bound on any process's `cpus` |
| `max_memory` | *(required, no default)* | Upper bound on any process's `memory` |
| `max_time` | `240.h` | Upper bound on any process's `time` |

`max_cpus` and `max_memory` are declared `null` in `nextflow.config` and marked
`required` in `nextflow_schema.json`, so a run that sets neither is refused at
launch. Supply them from a `site.config` (`-c site.config`, copied from
`conf/site.config.template`) or from a profile that pins them:

| Profile / example | `max_cpus` | `max_memory` | `max_time` |
|---|---|---|---|
| *(none)* | — required — | — required — | `240.h` |
| `local` | `4` | `16.GB` | `72.h` |
| `test` | `2` | `6.GB` | `1.h` |
| `test_full` | `8` | `32.GB` | `6.h` |
| `shipped_defaults_test` | `2` | `6.GB` | `1.h` |
| *(a `site.config` sized for a large SLURM partition, e.g.)* | `128` | `700.GB` | `240.h` |

!!! note "There is no shipped site profile"
    Size a cluster run with your own `site.config`, made from
    `conf/site.config.template`. The row above is only an example of what a
    large-partition `site.config` sets. See [Make a site config](hpc.md#size-your-run).

!!! note "Why `-c site.config` reaches the ceiling"
    The top-level `process.resourceLimits` in `nextflow.config` is a **closure**,
    so it is evaluated at task-submission time — after the whole profile stack has
    merged, and after a `-c` file has been layered. That is what makes
    `-c site.config` work: the closure reads `params.max_*` back lazily.

    The `slurm` profile sets **no** `resourceLimits` of its own, so
    `-profile slurm,singularity -c site.config` is clamped by the site's ceiling
    like every other route. A plain-map `resourceLimits` in a profile would be
    evaluated while `nextflow.config` is parsed — before any `-c` file exists — and
    freeze the ceiling at `null`; `tests/test_frozen_config_params.py` forbids one
    in any config file.

---

## Retry policy

### Default (`conf/base.config`)

`maxRetries = 3`. A task retries when its exit status is one of:

| Exit | Meaning |
|---|---|
| `104` | Connection reset |
| `134` | `SIGABRT` |
| `135` | `SIGBUS` |
| `137` | `SIGKILL` — usually the OOM killer |
| `139` | `SIGSEGV` |
| `140` | `SIGUSR2` — SLURM's pre-walltime warning |
| `143` | `SIGTERM` — job killed |

Anything else is `finish`: the pipeline stops submitting new work and lets
running tasks complete.

Because memory and time both scale with `task.attempt`, a retry after an OOM
kill automatically climbs the ramp.

### `REGISTER` — retries on exit 1 as well

VALIS tile-read failures and JVM out-of-heap conditions surface as a plain
exit 1, which the default strategy would treat as fatal. `REGISTER` therefore
retries on `[1, 104, 134, 135, 137, 139, 140, 143]`.

### QC processes

`GENERATE_PREPROCESS_QC`, `GENERATE_REGISTRATION_QC`,
`GENERATE_POSTPROCESSING_QC`, `GENERATE_QC_REPORT`, `SEG_QC_SEGMENT`,
`SEG_QC_GEOJSON` and `WARP_SEG_QC` share one policy — **`retry-then-fail`**: retry
a signal kill up to `maxRetries`, then **`finish`**.

```groovy
errorStrategy = { task.exitStatus in ((130..145) + 104) && task.attempt <= 3 ? 'retry' : 'finish' }
```

`130..145` is the whole "killed by a signal" range, not a hand-picked subset.
`'finish'` rather than `'terminate'` so in-flight tasks drain and the run
reports every failure at once rather than only the first.

!!! note "Why a failed QC task fails the run"
    QC outputs are aggregated with `collect()` / `collectFile()` / `combine()` /
    `join()`, none of which require a fixed item count, so a missing QC output is
    simply absent and never deadlocks the DAG. An `'ignore'` fallback would
    therefore swallow a genuine OOM or walltime kill and still leave a
    full-looking output tree with an exit code of 0. With `'finish'`, a broken QC
    task fails the run, and a green run implies a complete QC tree.

    **Two opt-in processes drop on purpose** — `SEG_QUALITY_EVAL` and
    `MERGE_SEG_EVAL` carry `retry-then-drop`
    (`{ task.attempt <= 3 ? 'retry' : 'ignore' }`), because losing a QualityScore
    degrades QC and nothing else. When either drops, `main.nf`'s `onComplete`
    warns via `workflow.stats.ignoredCount` rather than leaving it to be inferred
    from an absent column.

---

## Execution & concurrency

| Setting | Value | Where |
|---|---|---|
| `process.maxForks` | `params.max_forks` if set, else `params.concurrency` (`5`) | `nextflow.config` |
| `process.stageInMode` | `symlink` | `nextflow.config` — zero-overhead, works cross-filesystem |
| `executor.queueSize` | `params.queue_size` if set, else `params.concurrency * 4` (`20`) | `nextflow.config` — max concurrent scheduler submissions |
| `executor.exitReadTimeout` | `1 day` | `conf/base.config` — SLURM status-poll timeout |

**`--concurrency` is the one knob to tune.** It drives `max_forks` and `queue_size`
together, preserving the shipped 5:20 ratio (`--concurrency 20` → `max_forks 20,
queue_size 80`). `--max_forks` and `--queue_size` remain available and **override**
`--concurrency` individually — for the asymmetric case, e.g. a wide queue with a tight
per-process cap. `max_forks`/`queue_size` are declared `null` in `nextflow.config`, not a
numeric default: the params block is evaluated *before* the CLI is applied, so a default
computed there would use `concurrency`'s own default and silently ignore `--concurrency`.
The derivation instead lives in `nextflow.config`'s concurrency block **after
`profiles {}`**, together with the four per-process caps, so the CLI, a `-params-file` and a
profile all reach it.

!!! warning "A `-c site.config` pin of these does NOT arrive — and the run says so"
    `queueSize` and `maxForks` are scalars evaluated while `nextflow.config` is parsed,
    which is before any `-c` file is merged. So `params { concurrency = 20 }` in a
    `site.config` changes the param and reaches nothing that reads it (the same
    holds for `cleanup_work`, `enable_trace` and `trace_dir`).
    Rather than ignore the pin, the pipeline refuses the run at launch
    (`ParamUtils.validateFrozenConfig`) and names the route. Pass these six with
    `-params-file` or on the command line, never in a `-c` file.

**`max_forks` and `queue_size` are a pair, and the LOWER one binds.** `max_forks` caps how
many tasks of any ONE process run at once; `queue_size` caps how many run at once across
the WHOLE pipeline. At the shipped defaults `max_forks` (5) is far below `queue_size`
(20), so **`max_forks` is what binds** and raising `queue_size` alone changes nothing —
raise `max_forks` (or `concurrency`), or raise both.

Because every per-process override is `Math.min(its own cap, the resolved max_forks)`, a
resolved `max_forks` of 5 clamps ALL of them: the `REGISTER` / `TILED_STITCH` 10 and the
`TILED_COARSE` / `TILED_REG_TILE` 20 below all run at 5. That is a deliberately
conservative default — raise it with `--concurrency` or `--max_forks` when the cluster
can take it.

Per-process `maxForks` overrides (in `nextflow.config`'s concurrency block, not in
`conf/modules.config` — that file is included before the profiles, and a cap frozen there
overrode the profile's value): `REGISTER`, `TILED_STITCH` at `10`; `TILED_COARSE` /
`TILED_REG_TILE` at `20`. These bound how many memory-heavy registration tasks can be in
flight at once. Each is written `Math.min(<its own limit>, the resolved max_forks)`, so
**lowering** `--max_forks` (or `--concurrency`) really does throttle every module, while
**raising** it never lifts one of these past the limit its own block sets for its own
reasons. On the test profile: at the default, `REGISTER` runs at 10 and
everything else at 5; at `--max_forks 4` every process runs at 4; at `--max_forks 50`,
`REGISTER` stays at 10.

`executor.queueSize` is assigned in `nextflow.config`, not in `conf/base.config` where the
rest of the executor scope lives. That is deliberate and load-bearing — see
[Why the includes sit after the params block](#why-the-includes-sit-after-the-params-block).

### Why the includes sit after the params block

`conf/base.config` and `conf/modules.config` are included **after** `nextflow.config`'s
`params` block, not at the top of the file. Anything in those files that reads `params.*`
depends on it:

* A `params.x` reference evaluated **before** the params block exists does not read `null`.
  Nextflow resolves it to an empty `ConfigObject` — a Map. Inside a closure that is
  harmless, because closures run at task-submission time. Evaluated eagerly it is
  not: `params.x as int` on a Map throws *"Cannot coerce a map to class
  java.lang.Integer"* and the entire config fails to parse.
* Inside an `executor { }` scope it is worse than an error. `queueSize = params.queue_size`
  parsed from an early-included file is read as the opening of a **nested scope named
  `params`**, and the `queueSize` setting vanishes from the resolved config with no error
  at all — silently falling back to Nextflow's own default.
* `maxForks` cannot dodge this the way `memory` does, because it is **not a dynamic
  directive**: Nextflow compares it against `0` in `TaskProcessor`'s constructor, so a
  closure throws *"Cannot compare ... Closure ... and java.lang.Integer with value '0'"*.

Both files are included before the `executor` and `process` blocks, so those take
precedence. Guarded by `tests/test_concurrency_params.py`.

---

## Containers

Every process pins a fixed image tag — never `:latest`. The `bolt3x/mirage-*` image
NAMES (one Docker Hub repository per image, e.g. `bolt3x/mirage-preprocess`,
`bolt3x/mirage-stare`) are content-descriptive; the TAG on each is an immutable
SemVer version (`1.0.0`), tied to `manifest.version` — see
[Running on a cluster → Pre-pulling container images](hpc.md#pre-pulling-container-images).

| Image | Processes |
|---|---|
| `bolt3x/mirage-convert:1.0.0` | `CONVERT_IMAGE` |
| `bolt3x/mirage-preprocess:1.0.0` | `TILE_FOR_BASIC`, `APPLY_PROFILES`, `SPLIT_CHANNELS`, `GENERATE_PREPROCESS_QC`, `GENERATE_QC_REPORT`, `PREFLIGHT_SCALE`, `AGGREGATE_SIZE_LOGS` |
| `docker.io/labsyspharm/basicpy-docker-mcmicro:1.2.0-patch5` | `BASICPY` (vendored nf-core module; pulls its own image, and errors under `-profile conda`) |
| `bolt3x/mirage-valis:1.0.0` (built on the digest-pinned upstream `cdgatenbee/valis-wsi:1.2.0`) | `REGISTER`, and `WARP_SEG_QC`'s VALIS path |
| `bolt3x/mirage-stare:1.0.0` | `TILED_COARSE`, `TILED_REG_TILE`, `TILED_SOLVE`, `TILED_STITCH`, and `WARP_SEG_QC`'s STARE path |
| `bolt3x/mirage-regqc:1.0.0` | `GENERATE_REGISTRATION_QC` |
| `bolt3x/mirage-stardist:1.0.0` | `SEGMENT` / `SEG_QC_SEGMENT` when `--seg_method stardist` *(default)* |
| `bolt3x/mirage-instanseg:1.0.0` | `SEGMENT` / `SEG_QC_SEGMENT` when `--seg_method instantseg` |
| `bolt3x/mirage-cellsam:1.0.0` | `SEGMENT` / `SEG_QC_SEGMENT` when `--seg_method cellsam` |
| *(per backend, `lib/WarpBackends.groovy`)* | `WARP_SEG_QC` |
| `bolt3x/mirage-quantify:1.0.0` | `SEG_QC_GEOJSON`, `QUANTIFY`, `NUCLEAR_RETENTION`, `MERGE_QUANT_CSVS`, `CELL_QC`, `EXTRACT_CELL_PROPERTIES`, `EXTRACT_NUCLEI_PROPERTIES`, `EXPORT_GEOJSON`, `GENERATE_POSTPROCESSING_QC` |
| `bolt3x/mirage-merge:1.0.0` | `MERGE_AND_PYRAMID`, `EXTRACT_MASK_SERIES` |
| `bolt3x/mirage-spatialdata:1.0.0` | `EXPORT_SPATIALDATA` |

`SEGMENT` and `WARP_SEG_QC` resolve their image from a backend table
(`lib/SegBackends.groovy`, `lib/WarpBackends.groovy`) rather than a literal, so
the container follows `--seg_method` and the registration method respectively.
An unrecognised `--seg_method` is rejected by name rather than silently falling
back to a different segmenter.

---

## Measuring what a run actually used

With `enable_trace` on (the shipped default), every process emits a per-task
input-size log, `AGGREGATE_SIZE_LOGS` collates them to
`<outdir>/size_logs/input_sizes.csv`, and Nextflow writes `trace.txt` into
`trace_dir` (default `.trace`, resolved against the launch directory) with
`cpus`, `memory`, `peak_rss`, `peak_vmem` and `realtime` per task.

`<outdir>/qc/mirage_resource_report.html` joins the two into run totals and
**four plots**:

| Panel | What it answers | What to do with it |
|---|---|---|
| **Wall-time by Process** | Where did the run's time go? Ranked bars, longest first. | The top bar is the only process worth optimising first. Each bar's `%` is that process's share of the RUN's total wall-time, not of the longest bar — the longest bar does not read 100% unless it really is the whole run. |
| **Memory Headroom** | What was reserved and never used? A light track per process is the request, the dark overlay the observed peak. | A wide track with a sliver of dark is an over-sized `withName:` request — lower it. A dark bar overrunning its track is drawn red: that is the OOM-retry precursor, raise it. |
| **Retries & Failures** | What did the failures cost, in reserved GB·hours? | A failed attempt holds its full reservation for its whole wall-time and delivers nothing. Across one real run this was 19.6% of all reserved GB·h — invisible in a green build. |
| **Input Size vs Runtime** | Does this step's cost scale with the data, and where does it stop being linear? | Log-log. The legend is ranked by POINT COUNT, not by first appearance: the processes with the most plotted points get their own colour and a legend entry (up to 8 by default), and everything past that is drawn in one shared grey "other (K processes)" row — so the legend never silently omits the process that actually dominates the run. On the real `traces/tiled_run/trace.txt` fixture, `TILED_REG_TILE` is 95% of all plottable points and is correctly the first legend entry. Each process's own points are additionally capped to 2,000 (a deterministic stride subsample, stated in a caption when it thins the series) so one huge process cannot blow the report up to megabytes. A series that bends upward is super-linear; that is where a size-tiered `memory` closure earns its keep. Zero-size, zero-runtime, and non-finite (`nan`/`inf`) tasks cannot be placed on a log axis and are dropped, with the count stated on the panel — and if EVERY point is dropped, the panel says so rather than falling back to the generic "no size logs matched" sentence. |

Re-runnable by hand against any completed run:

```bash
python3 bin/generate_resource_report.py \
  --trace .trace/trace.txt \
  --size-log results/size_logs/input_sizes.csv \
  --output resource_report.html
```

The script is **standard-library only, deliberately and permanently.** It runs
on the head node from `workflow.onComplete`, outside every container and under
whatever `python3` the operator has, so a single third-party import would make
the report silently unavailable on most deployments —
`tests/test_resource_report_is_stdlib_only.py` fails the build on one.

### When no report appears

The run log says which of the two reasons applies, and names the exact path it
looked for:

```
WARN  No resource report was generated: no trace at /path/to/launch/.trace/trace.txt.
      enable_trace was OFF for this run, so Nextflow wrote no trace -- turn it on
      with a -params-file or a profile to collect one.
```

`trace_dir` is resolved against the **launch directory**, which is what Nextflow
itself resolves `trace.file` against — so a relative `trace_dir` means the same
place to the report as it does to the engine, whatever directory the pipeline
was launched from.

---

## See also

- :material-tune: **Every parameter** — [Parameters](parameters.md)
- :material-file-tree: **Every output** — [Outputs](outputs.md)
- :material-console: **Cluster invocations** — [Running on a cluster](hpc.md)
- :material-expansion-card: **GPU requests** — [GPU or no GPU](gpu.md)
