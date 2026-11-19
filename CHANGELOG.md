# Changelog

All notable changes to the MIRAGE pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Upgrading from 1.0.0

- **The defaults changed.** Registration is STARE (`registration_method = 'tiled'`, was
  `valis`), segmentation is StarDist (`seg_method = 'stardist'`, was `instantseg`),
  `skip_preprocessing` is `true` (BaSiC correction is opt-in) and `cleanup_level` is `'none'`
  (was `'final'`). Pin the old values in a params file or profile to keep 1.0.0 behaviour.
- **Registration results are not comparable with 1.0.0**, for either backend: VALIS runs
  valis-wsi 1.2.0 at its own defaults, and STARE is a new method version (below).
- **Several parameters are gone and are refused at launch** if still set: `memory_mode`,
  `reg_valis_max_processed_dim`, `reg_valis_max_non_rigid_dim`, `reg_valis_max_keypoints`,
  `reg_micro_reg`, `reg_micro_reg_fraction`, `reg_max_image_dim`, `reg_tiled_solver`,
  `reg_tiled_gate_tre`, `reg_tiled_max_error`, `reg_tiled_upsample` and `reg_tiled_mode`. The
  `tma` profile is gone.
- **Two images must be published or pulled again before a real run**:
  `bolt3x/mirage-valis:1.0.0` (new) and `bolt3x/mirage-stare:1.0.0` (replaces
  `bolt3x/mirage-tiled`). Delete a cached Singularity `.img` so it is pulled again.
- **Point FlowPath >= 0.10.0 at output from this version** (see `cells.geojson` below).

### Added

- **`reg_valis_mode`**: `default` (valis-wsi 1.2.0's defaults) or `hires` (valis-wsi 1.2.0's
  high-resolution example: `MicroRigidRegistrar`, then `register_micro` at a quarter of the
  smallest slide's longest side). `REGISTER` asks 128 GB on its first attempt under `hires`
  (128 → 256 → 512 → 1024 GB). Under `hires`, staged registration QC (`reg_qc = 2`) scores a
  `micro` stage again and the pre-micro error summary is saved as `*_summary_premicro.csv`.
- **A samplesheet that gives one patient the same non-nuclear channel on two slides is
  refused at launch**, naming the patient, the channel, both rows and the remedy. Before, one
  of the two acquisitions was silently dropped. Nuclear markers (`DAPI`, `CELLTOX`) are
  expected on every slide and are exempt. A name repeated within one slide's `channels` cell
  is refused too. Comparison is trimmed and case-insensitive. The check applies at every
  `--start`; a checkpoint CSV that trips it was already losing data, so re-run the earlier
  step from a corrected samplesheet.
- **Per-cell QC keys** in `merged_quant.csv`, `cells.geojson` and the SpatialData table:
  `QC: Total intensity` per cell, and per moving imaging round `QC: Nuclear retention: [...]`,
  `QC: Registration displacement µm: [...]` and `QC: Registration Dice: [...]`, where the
  trailing list names the markers a failing round should blank. Two new processes produce
  them: `NUCLEAR_RETENTION` (per moving slide, publishes `<id>_nuclear_retention.csv` under
  `quantify/`; meaningful only for re-stained rounds of the same section) and `CELL_QC` (per
  patient, also writes `quantification/<patient_id>_round_qc.csv` and
  `quantification/<patient_id>_rounds.json`). All three quantification files survive
  `--cleanup_level final`. Registration displacement and Dice need `--reg_qc 2` (the default).
  NaN always means "no evidence", never "good", and is omitted from `cells.geojson`. See
  `docs/outputs.md`, "Per-cell QC".
- **`reg_tiled_stride`** (default 64) and **`reg_tiled_max_disp`** (default: the halo) for
  STARE's vector lattice and range gate.

### Changed

- **STARE (`registration_method = 'tiled'`) is the default registration method.** VALIS
  stays available with `--registration_method valis`. The `full_pipeline` and
  `registration_only` presets follow the default.
- **STARE is an external package**, `stare-registration` v1.3.0 from
  github.com/sceriff0/stare, pinned in `requirements/stare.txt`. Its image is
  `bolt3x/mirage-stare:1.0.0` (built from `containers/stare`), which carries no torch, kornia
  or learned-matcher weights. `registration_method = 'tiled'`, the `reg_tiled_*` parameters
  and the `TILED_*` process names are unchanged.
- **STARE's registration method changed.**
  - REG_TILE measures one displacement vector every `reg_tiled_stride` px (window
    2 × stride) on a slide-global lattice; SOLVE drops vectors beyond `reg_tiled_max_disp` and
    fits robust affine + robust DCT-PLS, writing a cubic B-spline mesh. The `legacy` and
    `robust` solvers and the TRE gate are removed. SOLVE refuses a control JSON written by a
    1.0.0 REG_TILE; re-run REG_TILE (a `-resume` after upgrading does).
  - COARSE anchors each slide with a normalised cross-correlation rotation sweep on a
    tissue-masked thumbnail, with an ORB + RANSAC fallback, instead of DISK + LightGlue. Its
    peak memory is well under 1 GB. When neither passes its gates, COARSE checks every
    candidate at full resolution on small nuclear patches; if none is confirmed it continues
    with the best guess, logs `UNVERIFIED ANCHOR` naming both slides and writes
    `coarse_trusted: false` into the M0 JSON. Such a slide can come out mis-registered with
    exit 0, so check its registration QC. Only a blank nuclear plane is refused. The M0 JSON
    gains `coarse_method`, `coarse_peak_ncc`, `coarse_peak_ratio`, `coarse_angle_deg` and
    `coarse_probe`.
  - STARE has one configuration, no cost tiers: `reg_tiled_tile` 2048, `reg_tiled_halo` 256,
    `reg_tiled_out_tile` 1024, `reg_tiled_coarse_max_dim` 1024, each settable directly. A
    `reg_tiled_mode` of `medium`, `low` or `custom` is refused at launch with the retired
    row's values in the message (medium 1024/192/768/1024, low 512/128/512/512); `high` only
    logs a warning, since it is what runs anyway.
  - The default lattice (`reg_tiled_stride` 64) gives four times as many vectors per tile as
    stride 128 and halves the anchor error a tile can recover to ±64 px. Pass
    `reg_tiled_stride: 128` for the coarser lattice.
  - Memory at attempt 1: `TILED_COARSE` 2 GB, `TILED_REG_TILE` 4 GB, `TILED_SOLVE` 2 GB,
    `TILED_STITCH` 4 GB.
- **VALIS runs valis-wsi 1.2.0 at its defaults and has no tuning settings.** `REGISTER`
  passes the patient's slides and reference, registers at VALIS's own sizes (rigid 512 px,
  non-rigid 2048 px) with its own detector and matcher, no micro-registration, and warps
  every slide in the reference frame. The one non-default argument is the registration
  channel: the nuclear channel by index 0, not VALIS's name match on `"dapi"`. The
  Bio-Formats JVM heap is a fixed 10 GB (`reg_jvm_heap_gb` still overrides it). Staged
  registration QC scores `native`, `rigid` and `non_rigid` at `default`. The old tuned
  configuration is archived at the git tag `archive/valis-mirage-tuned`.
- **New image `bolt3x/mirage-valis:1.0.0`** for `REGISTER` and `WARP_SEG_QC`'s VALIS path:
  upstream `cdgatenbee/valis-wsi:1.2.0`, digest-pinned, with only the fixes a non-root user
  needs under Apptainer. Both processes run with `--writable-tmpfs` under
  Singularity/Apptainer. Publish it (`containers.yml` workflow_dispatch) before a real VALIS
  run.
- **`REGISTER` memory starts at 64 GB and doubles per attempt, at most four attempts
  (64 → 128 → 256 → 512 GB)**, clamped by `max_memory`. A profile's generic `maxRetries` does
  not change this ramp.
- **StarDist is the default segmentation backend**, with the trained model
  `hf://DSUnit/stardist@68b94406104bb069a58b64f4c890375c7f2f15df` fetched from the Hugging
  Face Hub on first use (into `segmentation_model_dir` when set, otherwise the task
  directory). A copy already in `segmentation_model_dir` is used without network access,
  which is the route for air-gapped clusters; `HF_ENDPOINT` selects a mirror. The reference
  must name a 40-hex commit; a branch such as `@main` is refused. A plain
  `segmentation_model` name still means a folder under `segmentation_model_dir` or a StarDist
  built-in.
- **`skip_preprocessing` defaults to `true`**: a default run converts every slide to
  OME-TIFF and goes straight to registration. BaSiC correction is opt-in
  (`"skip_preprocessing": false` in a params file or profile, not on the CLI).
  `csv/preprocessed.csv` then records the converted image.
- **`cleanup_level` defaults to `'none'`**: a run's output is re-enterable by `--start`
  without opting in. `'final'` is the opt-in cleaning level, with the same publish gates.
- **`cells.geojson` measurements**: morphology is named `MORPH: Area µm²`,
  `MORPH: Eccentricity`, etc. (was the bare name), and every `QC: ...` column is carried
  through, after markers and before `MORPH:`. Area shows 4 decimal places (was 3); the value
  is unchanged. The SpatialData table carries `QC: ...` columns in `obs` verbatim.
  **Requires FlowPath >= 0.10.0**: an older FlowPath reads a `QC:` key as a phantom marker
  and misses `MORPH: Area µm²` in its `area` lookup, and there is no dual naming on the
  pipeline side.
- **`*_reg_residuals.csv` (written at `--reg_qc 2`) gains an `iou` column**:
  `moving, ref_x, ref_y, residual_px, iou, stage`. A CSV without it still reads; Dice then
  comes back NaN.
- **`quantification/merged_quant.csv` is published by `CELL_QC`**, with the `QC: ...`
  columns added, at the same path as before.

### Fixed

- **A `-c site.config` pin of `cleanup_work`, `enable_trace`, `trace_dir`, `concurrency`,
  `max_forks` or `queue_size` was silently ignored.** Such a run is now refused at launch,
  naming the route; `-params-file`, the CLI and profiles all work.
- **A profile pin of `concurrency`, `max_forks` or `queue_size` did not take effect.** It does
  now.
- **`-profile slurm` ran with no resource ceiling** on the documented
  `-profile slurm,singularity -c site.config` route. The `max_*` ceiling now applies.
- **`--cleanup_level final` left seven empty intermediate directories per patient.** They are
  removed at the end of the run.

## [1.0.0] - 2026-09-08

### Migration — read before comparing any output across this release

Seven changes alter what a run produces or what its outputs mean, and every one of them is
silent from the consumer's side. The first six are detailed in
[`docs/_archive/migration-2026-08-24.md`](docs/_archive/migration-2026-08-24.md):

1. **Darkfield correction is gone** — nf-core `BASICPY` runs at `get_darkfield=False`; the
   earlier in-house path used `True`. Corrected pixel values are not comparable across this
   release.
2. **NaN measurements are omitted, not written as `0.0`** — a cell with no nuclear overlap
   carries fewer keys than its neighbours. FlowPath handles this and gates call the same
   cells positive, but every percentile-derived threshold moves, because the artificial zeros
   have left the distribution.
3. **STARE meshes differ, by design** — two new control-point gates reject points the
   previous code accepted. Registration output is not bit-comparable.
4. **Checkpoint CSVs carry an `id` column** — an existing `--prior_outdir`, or an existing
   tree used with `--start`, now fails at launch instead of re-deriving an identity that may
   differ.
5. **A run publishes final outputs only and deletes its work directory** — `cleanup_level`
   defaults to `'final'` and `cleanup_work` to `true`. No checkpoint manifest is written at a
   cleaning level. Pass `--cleanup_level none` on any run whose output will be re-entered.
6. **Two invocations that used to be accepted now fail at launch** —
   `--seg_instantseg_target cells|nuclei` (it produced one mask, replicated into both outputs,
   which zeroed every cytoplasmic measurement), and an OME header whose `PhysicalSizeXUnit`
   is not recognised. A recognised non-µm unit is now converted rather than read as µm (an
   `nm` header was a 1000× scale error).
7. **`pixel_size` defaults to `'auto'`** (was `null`, which forced an explicit value).
   `'auto'` reads each image's own OME `PhysicalSizeX`, so a run that previously refused to
   start may now proceed using the file's scale. `PREFLIGHT_SCALE` resolves every slide's
   scale before heavy work starts: it fails when `auto` cannot resolve a slide, and warns
   (never fails) when a supplied number disagrees with the file, cannot be confirmed, or when
   slides in one run resolve to different scales. Every checkpoint manifest carries a
   `pixel_size` column; a checkpoint written by an earlier version is refused on re-entry
   ("this checkpoint predates scale tracking"): re-run the step that wrote it. Measurements
   are unchanged for a run that already passed an explicit `--pixel_size`.

### Added

- **`-profile tma`** for tissue-microarray input: pins `memory_mode = 'custom'` and
  `reg_valis_max_non_rigid_dim = 1024`, composes with a site profile
  (`-profile slurm,ieo,tma`), and sizes `REGISTER` at 32 GB × attempt. Needed because VALIS
  cannot register a slide whose full resolution is no larger than its non-rigid size; safe
  for cores of 2048 px or more.
- **`PREFLIGHT_SCALE`**, run once over every input slide before heavy work. It reads only OME
  metadata, resolves the scale `--pixel_size` will use per slide, and records it in the meta
  map so `EXPORT_GEOJSON`, `EXPORT_SPATIALDATA`, `SEG_QUALITY_EVAL` and InstanSeg see a
  number, not `'auto'`. A legitimate mixed-magnification cohort is never refused, only
  reported. See the migration note above.
- **STARE's COARSE anchor is DISK + LightGlue**, the only front-end. Its peak memory grows
  with thumbnail area, so the `reg_tiled_coarse_max_dim` tiers are 2048 / 1024 / 512 and
  `TILED_COARSE`'s memory request is derived from that bound (about 48 GB at `high`, 5 GB at
  `low`). The first `-resume` after upgrading re-runs STARE's COARSE-and-downstream chain.
  The development-only `reg_tiled_frontend` parameter and `stare_ml` profile never shipped;
  if a script used either, drop it (the schema and Nextflow refuse them).
- **`torch`, `kornia` and the DISK/LightGlue weights ship inside `bolt3x/mirage-tiled`**, so
  COARSE needs no profile, second image or download at first use (the target cluster has a
  read-only `$HOME` and no network).
- **Two new published registration artifacts** under `<outdir>/<patient_id>/registered/`:
  `transform/preprocessed/data/<patient_id>_registrar.pickle` (the VALIS registrar) and
  `controls/<patient_id>_<channels>_<ix>_<iy>_ctrl.json` (STARE's per-tile control points).
  The control-JSON count scales as (slide/tile)²: about 100 files per 20480² slide at `high`,
  about 1600 at `low`. Total bytes are negligible, but the file count can load a networked
  filesystem's metadata server.
- **A new `segmentation` step** between `registration` and `postprocessing`, with its own
  resumable checkpoint, `csv/segmented.csv` (one row per registered slide). `--start` /
  `--stop` accept `segmentation`. `--start postprocessing` now needs `cell_mask` and
  `nuclei_mask` columns, so a plain `registered.csv` fails validation at launch. At
  `--start postprocessing`, `csv/postprocessed.csv`'s `cell_mask` column names a path under
  the prior run's `--outdir`. The other checkpoint CSVs are unchanged.
- **`nuclear_markers`** (default `['DAPI', 'CELLTOX']`): an ordered preference list naming the
  nuclear/fiducial channel. The first marker present (from channel metadata, never the
  filename) is moved to channel 0 by `CONVERT_IMAGE`; the run fails if none is present
  (single-channel images excepted).
- **An `--mode add_cycle` run writes `csv/registered.csv` into its `--outdir`.** Chaining two
  `add_cycle` runs is still not supported (`csv/postprocessed.csv` is not written).
- **Two new supplementary figures**: `docs/figures/coarse-schematic.html` (S5, the coarse
  front-end) and `docs/figures/accuracy-schematic.html` (S6, what each accuracy measure
  measures).
- **A samplesheet with a mistyped or extra column warns, naming it**, on both the linear and
  `add_cycle` paths. A legitimate checkpoint header warns about nothing.
- **Format coverage**: pyramidal OME-TIFF, BigTIFF, interleaved RGB, 8-bit, float32, HDF5 and
  NDPI/NDPIS are tested on real bytes. For `.czi`, `.nd2`, `.lif`, `.ndpi` and `.svs`, which
  cannot be synthesised, `tests/cluster/` ships a kit an operator runs against private images
  to produce `docs/validation/format_validation.md`.
- **Corrected image-reading results**: an interleaved-RGB plain TIFF reports
  `shape_cyx=(3, Y, X)`, an `.ndpi` carries its pixel size, and reading one channel of an RGB
  image selects a sample. The pipeline's own reads were unaffected.

### Changed

- **The VALIS tiers use one size for both stages: `high` 2048, `medium` 1024, `low` 512 px**
  (non-rigid was 4096 / 4096 / 1024, matching 2048 / 1024 / 256). Registration output changes
  for every VALIS run that used a tier; `custom` runs that set both sizes are unaffected.
- **The resource report is plots, not tables**: wall time by process, memory headroom
  (requested vs peak RSS), reserved GB·hours lost to failures and retries, and input size
  against runtime. Run Totals remains as a table and gains peak single-task %CPU. A
  zero-byte input is dropped from the log-log scatter (the panel says how many), and the
  per-task status table is gone.
- **The HTML QC report is plots, not tables**: per-stage registration-error distributions, a
  feature-TRE vs cell-displacement scatter with its 3× divergence band, and a per-cell
  residual distribution.
- **The Software Versions and Sample Manifest sections are gone from the report.** Their
  sources are still published: `qc/mirage_qc_data_<timestamp>/collated_versions.yml` and
  `run_summary.json`.
- **`pixel_size` default: `null` → `'auto'`** (see Migration item 7). `conf/test.config` and
  `conf/test_full.config` pin a number because the synthetic fixtures carry no OME
  `PhysicalSize`.
- **`reg_tiled_dapi_index` → `reg_tiled_nuclear_index`, default `0` → `null`.** `null`
  resolves the index from the slide's channel metadata; an integer overrides. `TILED_COARSE`
  and `TILED_REG_TILE` fail, naming the configured markers, when no nuclear channel is
  present. **Breaking**: rename the parameter in commands and params files. A negative index
  is now rejected instead of reading the last channel.
- **STARE memory requests follow the parameters that set them**: `TILED_REG_TILE` and
  `TILED_STITCH` ask 4 GB at the defaults (was 8 GB) and more when `reg_tiled_tile` or
  `reg_tiled_out_tile` is raised; `TILED_SOLVE` asks 1 GB (was 4 GB).
- **The STARE size parameters are validated at launch.** `reg_tiled_out_tile` must be a
  multiple of 16; any other value used to fail only in the final `TILED_STITCH`.
- **The registration method's intrinsic TRE is no longer named after VALIS.** In the
  `qc/mirage_qc_data_*/` bundle, `valis_summary/` is renamed `registration_tre/`.
- **`nuclear_markers` is honoured by every consumer** (`SPLIT_CHANNELS`, `SEGMENT`,
  `SEG_QC_GEOJSON`, samplesheet validation and channel counting). Under the default
  `['DAPI', 'CELLTOX']`, a channel whose name contains `CELLTOX` is now dropped from
  non-reference slides, as `DAPI` always was. Set `--nuclear_markers DAPI` for the old
  behaviour.
- **`--nuclear_markers` accepts a comma/space-separated string**, so
  `--nuclear_markers CELLTOX` works on the command line.
- **`reg_qc=2` registration QC simplifies contours at `simplify_tolerance` and expands cells
  by `seg_expand_distance`**, the same settings as the cells it is compared against. Scores
  from before this change are not comparable.
- **`reg_micro_reg` defaults to `2`** (micro-rigid + micro non-rigid, was `0`). Expect
  ~30–120 min more per registration.
- **`--seg_method` with an unrecognised value fails** instead of running StarDist.
- **`params/full_pipeline.json` sets `"seg_method": "stardist"`**, so its StarDist settings
  are used. Pass `--seg_method instantseg` for the previous behaviour.
- **`<outdir>/size_logs/versions.yml`'s key is `MIRAGE:FINAL_QC:AGGREGATE_SIZE_LOGS`.**
- **`--embed_masks true` requires both `--quantify_compartments` and
  `--expanded_quantification`.** It used to exit 0 and publish a pyramid with no mask series.
  Drop `--embed_masks`, or set both.
- **`SPLIT_CHANNELS` and `QUANTIFY` publish per patient**:
  `<outdir>/<patient_id>/split_channels/<marker>.tiff` and
  `<outdir>/<patient_id>/quantify/`. Before, two patients sharing a marker overwrote each
  other's published file.
- **`add_cycle`: `SPLIT_PRIOR_PYRAMID` publishes to
  `<outdir>/<patient_id>/split_channels/prior/`**, so it no longer overwrites the new
  cycle's split channels.
- **`EXTRACT_MASK_SERIES` (`add_cycle`) publishes to `<outdir>/<patient_id>/segmentation/`**
  (was `<outdir>/extract_mask_series/`, where patients overwrote each other).
- **A process with no explicit `publishDir` publishes to `<outdir>/_UNROUTED_PUBLISH/`.**
  `<outdir>/tiled_coarse/` and `<outdir>/tiled_reg_tile/` are no longer published; their
  diagnostics are carried in the `*_tre.json` and manifest `TILED_SOLVE` publishes.
- **`versions.yml`** is rendered the same way for every process and records the `container:`
  the task ran in. On a real run, `extract_mask_series` and `merge_and_pyramid` probe
  `python` (was `python3`) and report `unknown` instead of failing when a probe import fails.
- **`WARP_SEG_QC_TILED` is merged into `WARP_SEG_QC`.** In
  `<outdir>/size_logs/input_sizes.csv` the tiled path's `process` column reads `WARP_SEG_QC`.
  The `*_seg_qc.json` and `*_reg_residuals.csv` outputs are unchanged.
- **Stub-mode size-log rows carry the real process name**, and `QUANTIFY`'s `sample_id` is the
  patient id.
- **The measured byte set changed** for `REGISTER`, `PREFLIGHT_SCALE`, `MERGE_AND_PYRAMID` and
  `WARP_SEG_QC`, so size-vs-runtime plots are not comparable across this release.
- **An unsupported samplesheet extension fails at launch** with `UnsupportedFormatError`
  naming the suffix. `.svs`/`.qptiff`/`.vsi`/`.scn`/`.mrxs`/`.bif`/`.ims` convert through
  Bio-Formats.
- **`CONVERT_IMAGE`'s `versions.yml` records `bioio`** (was `aicsimageio`).
- **`APPLY_PROFILES` writes OME-XML attributes in `convert_image`'s order.** Pixels are
  unchanged, but the header bytes differ and `-resume` across the upgrade re-runs that step.
- **Registration QC is a before/after pair**: the composite shows reference + native next to
  reference + registered, reconciled by pad-or-crop, never rescaling. File names are
  unchanged; the figure is twice as wide. Both TIFF outputs carry a real pixel size. Anyone
  calling `GENERATE_REGISTRATION_QC` directly must pass the native image as a fourth input.
- **Every container base image and both external runtime images
  (`cdgatenbee/valis-wsi`, `labsyspharm/basicpy-docker-mcmicro`) are pinned by digest.**
- **`AGGREGATE_SIZE_LOGS` runs in `bolt3x/mirage-preprocess`** (was `ubuntu:22.04`); every
  first-party image installs `procps`; `containers/regqc` carries only what
  `GENERATE_REGISTRATION_QC` runs; StarDist's `gputools`/`edt` are pinned; all first-party
  images carry OCI provenance labels.
- **`containers/convert` installs `bioio-bioformats`** with the Bio-Formats jars baked in and
  the JVM cache moved off a read-only `$HOME`, so `.svs`/`.qptiff`/`.vsi`/`.scn`/`.mrxs`/
  `.bif`/`.ims` convert offline.
- **`requirements/spatialdata.txt` is locked to exact versions.**
- **`docs/figures/` matches the current pipeline**: two registration backends, the
  DISK + LightGlue coarse front-end, the plots-first QC report and the two-panel registration
  QC.

### Fixed

- **VALIS registration of small slides (TMA cores) died with "JVM is not running"** after the
  whole rigid stage. A VALIS defect picks pyramid level -1 for a slide whose full resolution
  is below the non-rigid source size, swallows the resulting error and returns
  `(None, None, None)`. `REGISTER` now refuses such slides at start, naming them and the
  remedies, and treats a `None` result from `register()` as the failure it is.
- **`cleanup_level` set in a profile or a `-c` site config was ignored by the publish
  gates**, so a site config pinning `'none'` produced a tree with no intermediates and no
  checkpoint manifest. It is honoured now; the published layout at either level is
  unchanged.
- **The resource report is found where Nextflow wrote it**, also when a wrapper or `sbatch`
  changes directory. When no trace is found the run warns, naming the path and whether tracing
  was on, and writes no page.
- **The release archive and images are built from the tagged commit**, after the version
  bump.
- **The nuclear/fiducial channel is resolved from `nuclear_markers` everywhere.** The
  registration QC overlay no longer warns `No DAPI channel found` on CellTox panels, and
  `register.py --reference-markers` no longer defaults to a filename match on
  `["DAPI", "SMA"]`.
- **A scale disagreement between an image and `pixel_size` was silent.** `CONVERT_IMAGE` logs
  `[SCALE MISMATCH]` when the OME `PhysicalSize` differs from the configured value by more
  than 1%. `SPLIT_CHANNELS` and `TILED_STITCH` outputs now carry the scale as TIFF resolution
  tags (they claimed 1 px = 1 unit). No published number changes.
- **`cells.geojson` carries the segmentation `label` as a measurement**, so it joins exactly
  to `cells_data.csv` (a cell with a NaN centroid is in the CSV but not the GeoJSON, so a
  positional join was wrong).
- **Registration QC was silently missing for about half of patients on contended runs** (a
  `groupTuple()` key type mismatch in the warp fan-in). Fixed; `-resume` and existing work
  directories are unaffected.
- **Two concurrent `REGISTER` tasks on one node could share a staging file list** under
  Singularity (`/tmp` is host-mounted). The list now lives in the task directory.
- **`reg_qc=2` registration QC segments with the run's own `--seg_method`**, and runs at
  stock defaults (it raised `FileNotFoundError` and produced no QC at all). On the StarDist
  backend the nuclear channel is channel 0; a slide whose nuclear marker is elsewhere, or
  absent, now fails that QC task instead of being scored on the wrong channel. Published
  output is unchanged.
- **The QC alias of `SEGMENT` no longer inherits `SEGMENT`'s `publishDir`**, and its size log
  is named per slide.
- **`TILED_COARSE` never finished inside its 2 h walltime**: ORB ran on raw uint16 intensities.
  Each plane is now normalised over a robust percentile window first, which also stops two
  differently exposed cycles getting different keypoint thresholds.
- **`TILED_COARSE` logs each stage** (inputs, slide sizes, intensity percentiles, keypoint
  counts, M0) and warns on conditions that used to exit 0 silently: a residual at or beyond
  the halo, fewer than 10 inliers, too few matches, a non-finite residual, an out-of-range
  nuclear index.
- **`TILED_COARSE` was OOM-killed on real slides**: it read the whole slide. It now reads a
  thumbnail (`reg_tiled_coarse_max_dim`, both slides at one shared decimation factor).
- **The `tiled` image installs `procps`**: every `registration_method='tiled'` task failed
  with exit 1 and no traceback without it. Delete a cached `.img`/`.sif` and re-pull.
- **`EXTRACT_NUCLEI_PROPERTIES`'s size log publishes to `<outdir>/<pid>/cell_properties/`**
  (was `.../cell_properties/nuclei/`). Its other outputs are unchanged.
- **Per-cell registration residuals reach the QC report**, on the linear path and on
  `add_cycle`. A declared QC artifact kind with no consumer now fails the run instead of
  being dropped.
- **`--mode add_cycle` rejects `--start`/`--stop`** instead of accepting and ignoring them.
  Omit both flags. Its run summary records `stop: postprocessing`.
- **`add_cycle`'s pyramid-channel grouping streams like the linear path's.** No output change.
- **The checkpoint manifests name files that exist**: `csv/postprocessed.csv`'s `cell_csv`
  and `cell_geojson` paths include the `export/` subdirectory.
- **The checkpoint CSVs have a deterministic row order** (patient id, then path).
- **A quoted samplesheet path (e.g. a filename with a space) was read wrong**; it is read
  correctly now, also by `PREFLIGHT_SCALE`. CRLF, a UTF-8 BOM and non-ASCII filenames are
  handled.
- **Registration QC fails when no configured nuclear marker matches a slide**, instead of
  silently using channel 0. See [`docs/registration_qc.md`](docs/registration_qc.md).

### Removed

- **The `ashlar` registration backend**: 1.0.0 ships `valis` and `tiled`.
  `--registration_method ashlar` and the `reg_ashlar_*` parameters are refused at launch.
  ASHLAR remains a benchmark baseline on the `benchmarking` branch.
- **Automated panel-agnostic phenotyping** (work in progress, parked on
  `feat/automated-phenotyping`): `COMPILE_PANEL`, `PHENOTYPE`, the parameters `panel_spec`,
  `panel_model`, `pheno_alpha`, `pheno_min_cal`, `pheno_max_enumerate`, and the
  `phenotyping/` output. `EXPORT_GEOJSON` emits the constant `"Cell"` classification with no
  `panel_model.json`; gate downstream in QuPath/FlowPath. The FlowPath-side phenotype join
  (`bin/join_flowpath.py`, `export_spatialdata.py --attach-phenotypes`) is unaffected.
- **`reg_tiled_fanout` and the single-task `TILED_REGISTER` path.** STARE always runs the
  per-tile fan-out `TILED_COARSE` → `TILED_REG_TILE` (× tiles) → `TILED_SOLVE` →
  `TILED_STITCH`, whose memory is bounded by parameters, not slide size. At the default
  `reg_tiled_tile = 2048` a 26k² slide plans about 169 `TILED_REG_TILE` tasks per moving
  slide; raise `reg_tiled_tile` to trade task count against mesh resolution.
- **`reg_reference_markers`**: the reference slide is chosen by `is_reference`; the nuclear
  channel by `nuclear_markers`.
- **`padding` / `pad_mode`** and the `PAD_IMAGES`, `GET_IMAGE_DIMS` and `MAX_DIM` processes:
  both registration backends align inputs of different sizes natively.
- **`PREPROCESS`**: BaSiC illumination correction runs through nf-core's `BASICPY` module
  (`TILE_FOR_BASIC` → `BASICPY` → `APPLY_PROFILES`). It runs at `get_darkfield=False`, so no
  darkfield is estimated or removed (see Migration item 1). It uses the third-party image
  `docker.io/labsyspharm/basicpy-docker-mcmicro:1.2.0-patch5`.
- **`bin/registration_benchmark.py`**: on the `benchmarking` branch only.
- **`segeval_tag`**: the segeval image follows `manifest.version`. Delete the key from old
  params files; schema validation refuses it.
- **`docs/figures/failure-schematic.html`**, an unpublished draft.

`CONVERT_IMAGE` also reads input lazily now, with byte-identical output and a lower memory
peak.

### The 1.0.0 feature set
- **Preprocessing** — BaSiC illumination correction with FOV tiling; Bio-Formats/ND2 →
  OME-TIFF conversion; multi-channel parallel processing.
- **Registration** — two backends selected by `--registration_method`:
  - `valis` (default): VALIS graph-based whole-slide alignment, with optional staged
    micro-registration (`reg_micro_reg` = 0/1/2).
  - `tiled`: STARE — JVM-free, internally tiled, fully parallel registration
    (`reg_tiled_*` params), usable on a laptop.
- **Staged registration QC** (`reg_qc` = 0/1/2) — DAPI overlay plus segmentation-overlap
  dice/displacement attributed per registration stage.
- **Segmentation** — three backends via `--seg_method`: InstanSeg (default), StarDist,
  and CellSAM; GPU or CPU; configurable nuclei→whole-cell expansion.
- **Quantification** — per-cell morphology and per-channel intensity; optional
  per-compartment (Nucleus / Cytoplasm / Cell) signal and expanded Mean/Sum statistics.
- **Reference-free segmentation-quality evaluation** — CellSegmentationEvaluator
  `QualityScore` with a `cse_max_pixels` downsample cap.
- **Export** — QuPath-compatible `cells.geojson` and pyramidal OME-TIFF; configurable
  contour simplification (`simplify_tolerance`) and coordinate precision; an additive
  scverse-native SpatialData `.zarr` store, written by default.
- **Incremental cyclic-IF** (`--mode add_cycle`) — fold a new imaging cycle into a
  completed patient run, reusing the prior reference, segmentation mask, and old-marker
  quantification; recomputes only the new cycle.

### Infrastructure
- DSL2 Nextflow architecture with step-based execution (`--start` / `--stop`).
- Checkpoint-based restart (preprocessing, registration, postprocessing).
- SLURM/Singularity and Docker execution profiles.
- Per-process version tracking, aggregated QC report, and computational-resource report.
- JSON-schema parameter definitions plus Groovy validation utilities.

[Unreleased]: https://github.com/sceriff0/mirage/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/sceriff0/mirage/releases/tag/v1.0.0
