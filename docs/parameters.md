# Parameters

<p class="standfirst">The complete parameter surface — all 89 operator-settable parameters of the
91 that <code>nextflow.config</code> declares (the two omitted are the nf-core
<code>config_profile_name</code> / <code>config_profile_description</code> identity strings, which
a site profile sets, not a run), grouped by pipeline
stage, each with the default that ships in <code>nextflow.config</code>. That file is the single
source of truth: <code>tests/test_no_duplicate_param_defaults.py</code> forbids a second
declaration of any default in the pipeline code it scans (<code>main.nf</code>,
<code>modules/local/</code>, <code>conf/</code>, <code>workflows/</code>,
<code>subworkflows/</code>, <code>lib/</code>), and <code>tests/check_param_consistency.py</code>
holds <code>nextflow_schema.json</code> to the same defaults.
<code>tests/test_parameters_doc_matches_schema.py</code> holds this page's
<em>parameter list</em> to <code>nextflow_schema.json</code> in both directions, so a
parameter cannot be added or removed without this page moving, and checks that
every <em>literal</em> default cell (most of them) agrees with
<code>nextflow.config</code>. A narrative cell — a per-tier default, or
<em>required, no default</em> — documents a <code>null</code> config default in
prose instead of restating <code>null</code>, and that prose is <em>not</em> machine-checked;
neither are the descriptions below. Update those by hand when behaviour changes,
and read <code>nextflow config -flat</code> for the values a specific run
actually resolved.</p>

!!! abstract "Canonical sources"
    - **Defaults** — `nextflow.config` (`params { … }`)
    - **Validation & enums** — `lib/ParamUtils.groovy`
    - **Schema** — `nextflow_schema.json`

    Pass values on the command line (`--param value`), or load a preset with
    `-params-file params/full_pipeline.json` and override individual values. Note
    the convention: pipeline parameters use a **double dash** (`--input`);
    Nextflow's own options use a **single dash** (`-profile`, `-resume`,
    `-params-file`).

## Input / output

| Parameter | Default | Description |
|---|---|---|
| `input` | `null` | Path to the samplesheet CSV. **Required.** Columns depend on `--start` — see [Samplesheet & Input](usage.md#the-samplesheet). |
| `outdir` | `null` | Output root directory. **Required** — no default. Checkpoint CSVs are written to `<outdir>/csv/`. |

## Workflow control

| Parameter | Default | Description |
|---|---|---|
| `start` | `preprocessing` | Entry stage: `preprocessing`, `registration`, `segmentation`, or `postprocessing`. |
| `stop` | `null` | Stop after this stage. `null` = run to the end. |
| `dry_run` | `false` | Validate inputs and exit without launching tasks. |
| `debug_channels` | `false` | Emit `.view` channel-topology debug output. |

!!! tip "Run a single stage"
    `--start registration --stop registration` runs registration only, reading a
    `<outdir>/csv/preprocessed.csv` checkpoint. See [Restartability](usage.md#checkpoints-resuming).

## Preprocessing

Bio-Formats conversion + BaSiC illumination correction.

| Parameter | Default | Description |
|---|---|---|
| `pixel_size` | `'auto'` | Micrometres per pixel, and the **single owner** of every µm conversion in the pipeline: GeoJSON centroids and areas, the published pyramid's `PhysicalSize`, and InstantSeg's rescaling all use this value and nothing else. `'auto'` (the default) reads `PhysicalSizeX` from each image's own OME metadata; a positive number overrides it for every slide instead, and `CONVERT_IMAGE`/`APPLY_PROFILES` then log a `[SCALE MISMATCH]` warning naming both numbers if the override disagrees with the file's own metadata by more than 1%. See `bin/utils/pixel_size.py`. |
| `skip_preprocessing` | `false` | Skip BaSiC illumination correction entirely. Conversion still runs — everything downstream assumes the standardised OME-TIFF layout — so the step still emits one image per input and `csv/preprocessed.csv` still has a row per slide; the row points at `<pid>/converted/` instead of `<pid>/preprocessed/`. |
| `preproc_skip_nuclear` | `true` | Leave the nuclear/fiducial channels named by [`nuclear_markers`](#common) uncorrected. Those channels drive both registration and segmentation, so correcting them changes what both consume. |
| `preproc_tile_size` | `1950` | BaSiC FOV tile size (px). |

BaSiC otherwise runs at nf-core BASICPY's upstream defaults — no darkfield estimation,
no autotune, non-overlapping FOVs. The knobs that once exposed those
(`preproc_autotune`, `preproc_n_iter`, `preproc_overlap`, `preproc_no_darkfield`,
`preproc_pool_workers`) were removed with the in-process BaSiC path;
`preproc_pool_workers` additionally set the correction process's `cpus`, which is now
owned outright by `conf/modules.config`. See `docs/basic_illumination.md`.

## Registration

Two backends, selected by `--registration_method`: **VALIS** (default, graph-based
whole-slide alignment) and **STARE tiled** (JVM-free, fully parallel). STARE is **not**
laptop-sized at its shipped tier — see the memory note under [Tiled / STARE](#tiled-stare-registration_methodtiled).

### Common

| Parameter | Default | Description |
|---|---|---|
| `registration_method` | `valis` | Registration backend: `valis` (VALIS whole-slide) or `tiled` (STARE — see [Tiled / STARE](#tiled-stare-registration_methodtiled)). |
| `nuclear_markers` | `['DAPI','CELLTOX']` | Ordered preference of nuclear/fiducial marker names. The first present (resolved from channel metadata, never the filename) is moved to channel 0 and drives both cell segmentation and the registration fiducial. Fails fast if none present (single-channel images excepted). Accepts a **list** (config or `-params-file`) or a **comma/space-separated string**, which is the only shape a command line can produce: `--nuclear_markers CELLTOX` and `--nuclear_markers DAPI,CELLTOX` both work. Matching is case-insensitive **substring**, so `DAPI_nuclear` counts as nuclear. |

!!! warning "The reference is declared, never inferred"
    There is **no parameter** that picks a registration reference for you. Every
    patient must declare exactly one `is_reference=true` row; a patient with none
    is a **hard error at launch**, at every entry point, and two `true` rows are
    the same error. `allow_auto_reference`, which promoted a patient's first
    samplesheet row, was **removed** — two sheets differing only in row order
    produced two different alignments and nothing recorded that the pipeline had
    chosen. Passing the flag now **fails the launch** as an unrecognised parameter
    (`validation.logging.unrecognisedParams = 'error'`).

### VALIS

| Parameter | Default | Description |
|---|---|---|
| `memory_mode` | `high` | VALIS cost/accuracy tier: `high` \| `medium` \| `low` \| `custom` (processed / non-rigid dims): `high` = 2048/2048 px, `medium` = 1024/1024 px, `low` = 512/512 px — one size per tier, used for both stages. All three use SuperPoint + SuperGlue with 5000 features — the tier changes resolution, not the feature matcher. `custom` starts from `high` and applies the `reg_valis_*` overrides below. Source: `MEMORY_PRESETS` in `bin/utils/valis_config.py`. See [Tiers](#tiers). |
| `reg_valis_max_processed_dim` | tier (`high`: 2048) | Feature detection/matching working size (px). **Tier-owned** — only settable under `--memory_mode custom`. |
| `reg_valis_max_non_rigid_dim` | tier (`high`: 2048) | Non-rigid registration size (px). **Tier-owned.** Must be **smaller than the full resolution of every slide**: VALIS 1.0.0–1.2.0 reads a slide no larger than this at pyramid level −1 and kills its JVM (its own size clamp does not prevent it), so `REGISTER` refuses such input at start with the offending slides named. Lower it below the smallest slide on small-format input such as TMA cores, with a margin — or use `registration_method = 'tiled'`. |
| `reg_micro_reg_fraction` | `0.125` | Image fraction used for micro-registration. |
| `reg_max_image_dim` | `4000` | Max cached image dimension during registration. |
| `reg_micro_reg` | `1` | Micro-registration depth (nested, default `1`): `0` = none, `1` = micro-rigid only (refines `slide.M`) — default, `2` = + micro non-rigid (`register_micro`). At `>=1` the QC `rigid` stage means affine ∘ micro-rigid. |
| `reg_jvm_heap_gb` | `null` | Explicit JVM heap (GB) for VALIS. `null` auto-estimates from input size. |
| `reg_qc` | `2` | Registration QC depth: `0` = none, `1` = the [before/after DAPI overlay](registration_qc.md#the-reg_qc-1-overlay-is-a-beforeafter-pair) only, `2` = that overlay + [staged segmentation-overlap metrics](registration_qc.md). |

At `reg_qc = 2` the pipeline segments each slide's DAPI on its **native** image, pairs the
nuclei once after rigid registration, and then re-scores those same pairs after every later
stage — so per-pair IoU and centroid residual can be attributed to `rigid`, `non_rigid` and
`micro` individually. See [Staged registration QC](registration_qc.md) for the output schema
and how to read it.

### Tiled / STARE (`registration_method=tiled`)

STARE is an alternative registration backend that is **JVM-free** (no Bio-Formats/Java
heap), tiles the slide internally, and runs fully in parallel. These params apply only when
`--registration_method tiled`.

**Sizing warning.** STARE is *not* a laptop backend at the shipped `high` tier. Every step
except the coarse anchor is region-streamed and stays under a few GB, but `TILED_COARSE`
matches with DISK — a U-Net whose peak is linear in thumbnail **area** — and asks **~48 GB**
at `reg_tiled_coarse_max_dim` 2048. `--reg_tiled_mode low` (512 px, ~5 GB) is the
workstation-viable tier. Because `max_memory` is a required parameter, a site sized from the
older "≤8 GB / laptop-friendly" wording gets a clamp and then a deterministic OOM on
`TILED_COARSE`, not a slow run.

| Parameter | Default | Description |
|---|---|---|
| `reg_tiled_mode` | `high` | Cost/accuracy tier: `high` \| `medium` \| `low` \| `custom`. Supplies every tier-owned knob below. `custom` starts from `high` and applies only the knobs you set. Table: `RegPresets.STARE` in `lib/RegPresets.groovy`. |
| `reg_tiled_nuclear_index` | `null` | Nuclear/fiducial channel index used to estimate the transform. `null` resolves it from the slide's channel metadata against `nuclear_markers`; set an integer only to override. |
| `reg_tiled_tile` | tier (`high`: 2048) | Tile core size (px); also the mesh-grid resolution. **Tier-owned** — only settable under `--reg_tiled_mode custom`. |
| `reg_tiled_halo` | tier (`high`: 256) | Per-tile read halo (px) for registration context. **Tier-owned.** |
| `reg_tiled_gate_tre` | `1.0` | Refine only tiles whose rigid-stage TRE (px) exceeds this. Not tier-owned: it decides which control points are acceptable, which is a correctness question rather than a cost trade. |
| `reg_tiled_max_error` | `0.99` | **Confidence gate.** Drop a mesh control point whose phase-correlation error exceeds this (`0` = perfect match, `~1` = no shared structure). This, and *not* displacement magnitude, is what keeps background and section-edge tiles out of the deformation mesh — those produce plausibly *small* displacements. Measured on the synthetic tiles in `tests/test_tile_residual_confidence.py`: real tissue `0.041`, background `0.9995`, section edge `0.99996`, empty crop `NaN`. Not tier-owned. |
| `reg_tiled_max_disp` | `null` | **Range gate.** Drop a mesh control point whose displacement magnitude (px) is at or beyond this — the match was never inside the read window, so the peak is an artefact. `null` uses `reg_tiled_halo`, the physically-motivated bound. A cheap secondary net only; `reg_tiled_max_error` is the load-bearing gate. Not tier-owned. |
| `reg_tiled_upsample` | tier (`high`: 10) | Phase-correlation sub-pixel upsample factor (per tile). **Tier-owned.** |
| `reg_tiled_out_tile` | tier (`high`: 1024) | Streaming stitch write-tile size (px) — gigapixel-safe. **Tier-owned.** |
| `reg_tiled_coarse_max_dim` | tier (`high`: 2048) | Longest side (px) of the thumbnail the coarse anchor (M0) is estimated on. **This is what bounds COARSE memory**, not tile size: the anchor's matcher is DISK, a U-Net, so its peak is linear in thumbnail AREA — `GB ~= 1.1 + 7.3 * Mpx` (3.03 GB at 512 px, 8.78 GB at 1024 px, measured on the pinned stack). `TILED_COARSE`'s memory request in `conf/modules.config` is derived from this value, so raising it raises the reservation too. Lower = cheaper and coarser; the M0 residual grows with the decimation factor and must stay well inside `reg_tiled_halo`. **A 256 px floor is enforced at launch** by `ParamUtils.validateRegPresets` — anything below it (including `0`, which used to disable decimation and would now hand a full-resolution plane to a U-Net) aborts before any process is instantiated. Use 512 (the `low` tier) or higher. **Tier-owned.** |

### Tiers

Both registration backends take the same four values. `high` is the shipped default and, with
**one exception**, is the behaviour that shipped before tiers existed; `medium` and `low` trade
accuracy for memory and wall-clock; `custom` starts from `high` and applies only the knobs you
set. The exception is `reg_tiled_coarse_max_dim`, which the whole column moved down one tier for
v1.0.0 — `high` is **2048**, not the 4096 that used to ship — because COARSE's matcher became a
U-Net and 4096 px extrapolates to ~123 GB of need (~185 GB of request). See
`lib/RegPresets.groovy`.

```bash
--reg_tiled_mode low                                   # every STARE knob from the low row
--reg_tiled_mode custom --reg_tiled_tile 4096          # high everywhere else, tile overridden
--memory_mode custom --reg_valis_max_non_rigid_dim 1024
-profile tma                                           # the same pair for TMA cores: custom + 1024
```

**Tissue microarrays.** The `tma` profile pins `memory_mode = 'custom'` and
`reg_valis_max_non_rigid_dim = 1024` (the `medium` row's size), because VALIS
cannot register a slide whose full resolution is no larger than that size and a TMA
core is typically 2000–3000 px on its long side. It describes the *data*, so it composes
with a site profile: `-profile slurm,ieo,tma`. It is safe for cores of 2048 px or more
on the long side; a CLI `--reg_valis_max_non_rigid_dim` still outranks it when your
cores are larger and you want a finer non-rigid stage. It also lowers `REGISTER`'s
memory request from 300 GB to 64 GB per attempt and pins the Bio-Formats JVM heap
flat at 16 GiB (`reg_jvm_heap_gb`), so the request goes to the Python side where
REGISTER's real peak — SuperGlue matching — lives; see [Resources](resources.md).

Setting a tier-owned knob under any tier **other than** `custom` is rejected before the first
process starts (`ParamUtils.validateRegPresets`). That is deliberate: a run that reports
`--reg_tiled_mode low` while using a hand-set tile value would be reported under a tier it is
not using, and the tier name is what reaches the QC report and the benchmark tables.

Tier values live in two places, because they have to:
`lib/RegPresets.groovy` (STARE) and `bin/utils/valis_config.py` (VALIS — its rows hold Python
objects that cannot be expressed in Groovy). `conf/modules.config` additionally inlines the STARE
table, because config files cannot see `lib/*.groovy`. All copies are pinned together by
`tests/test_reg_presets_inlined_in_config.py`.

## Segmentation

One process, `SEGMENT`, dispatching to three genuinely different segmenters via
`--seg_method`. They are **not** drop-in equivalents: they read different things,
build the whole-cell mask differently, and have different prerequisites.

### Choosing a backend

| | `instantseg` **(default)** | `stardist` | `cellsam` |
|---|---|---|---|
| **Reads** | The full multichannel image — **channel-invariant** | One channel, hardcoded to **index 0** | One nuclear channel, **resolved by marker name** |
| **Whole-cell mask** | Predicted natively, alongside nuclei | Expanded from nuclei by `seg_expand_distance` | Expanded from nuclei by `seg_expand_distance` |
| **Precondition** | A writable model cache | Channel 0 **must** be a nuclear marker — otherwise the task exits 1 | A nuclear channel must exist — a negative index exits 1 |
| **Weights** | Apache-2.0, auto-downloaded from BioImage.IO | **None ship with the repo** | **Gated** — needs `DEEPCELL_ACCESS_TOKEN` |
| **Runs on a fresh clone?** | **Yes** | No — set `segmentation_model_dir` | No — set `cellsam_model_path` or export a token |
| **Container image** | `bolt3x/mirage-instanseg:1.0.0` | `bolt3x/mirage-stardist:1.0.0` | `bolt3x/mirage-cellsam:1.0.0` |
| **Entry point** | `segment_instantseg.py` | `segment.py` | `segment_cellsam.py` |

!!! tip "Why InstanSeg is the default"
    It is the only backend that produces a mask on an unconfigured clone. The
    shipped `segmentation_model` name is **not** a StarDist built-in, so without
    `segmentation_model_dir` StarDist raises `FileNotFoundError`; CellSAM's
    weights sit behind a DeepCell token. InstanSeg's are Apache-2.0 and
    unencumbered.

    It is also the only one that predicts the whole-cell mask directly rather
    than dilating nuclei by a fixed radius — so **`seg_expand_distance` has no
    effect when InstanSeg is selected**. An unrecognised `--seg_method` is
    rejected by name at launch rather than silently running a different
    segmenter.

All three write both a `*_cell_mask.tif` and a `*_nuclei_mask.tif` regardless of
backend, so everything downstream is backend-agnostic.

### Backend selection & shared

These three apply to every backend.

| Parameter | Default | Description |
|---|---|---|
| `seg_method` | `instantseg` | Backend: `instantseg`, `stardist`, or `cellsam`. The container, entry point and preconditions all follow from this one value. |
| `seg_gpu` | `true` | Use GPU — adds SLURM `--gres=gpu:$gpu_type` plus Docker `--gpus all` / Singularity `--nv`. Set `false` to force CPU. |
| `seg_expand_distance` | `10` | Pixels to expand nuclei into the whole-cell mask. **StarDist and CellSAM only** — ignored by InstanSeg, which predicts cells natively. |

### StarDist (`seg_method=stardist`)

Consumes the **DAPI channel** (must be channel 0 — guaranteed upstream).

| Parameter | Default | Description |
|---|---|---|
| `segmentation_model` | `stardist_full_e200_…` | StarDist model name. **Not a StarDist built-in** — it names a trained model that must exist under `segmentation_model_dir`. |
| `segmentation_model_dir` | `null` | Directory holding the trained model. **Required for this backend**: with `null`, `segment.py` raises `FileNotFoundError` unless `segmentation_model` names a StarDist built-in (`2D_versatile_fluo`, `2D_versatile_he`, `2D_paper_dsb2018`, `2D_demo`). No model is bundled with the repo. |
| `seg_pmin` | `1.0` | Lower percentile for normalization. |
| `seg_pmax` | `99.8` | Upper percentile for normalization. |
| `seg_n_tiles_x` | `16` | Inference tiles along X. |
| `seg_n_tiles_y` | `16` | Inference tiles along Y. |
| `seg_prob_thresh` | `null` | Detection probability threshold. `null` uses the model's built-in default. |

### InstanSeg (`seg_method=instantseg`)

Channel-invariant — consumes the full multichannel image.

| Parameter | Default | Description |
|---|---|---|
| `seg_instantseg_model` | `fluorescence_nuclei_and_cells` | Pretrained InstanSeg model. |
| `seg_instantseg_target` | `all_outputs` | `all_outputs`, `cells`, or `nuclei`. |
| `seg_instantseg_tile_size` | `512` | Sliding-window tile size (px). |
| `seg_instantseg_batch_size` | `16` | Tiles per forward pass. Raise on big GPUs; lower to avoid OOM. |
| `instanseg_model_dir` | `null` | Writable BioImage.IO model cache (exported as `INSTANSEG_BIOIMAGEIO_PATH`). |

### CellSAM (`seg_method=cellsam`)

SAM foundation model on the DAPI channel (located by name).

| Parameter | Default | Description |
|---|---|---|
| `seg_cellsam_bbox_threshold` | `0.4` | Bounding-box confidence — the main precision/recall knob. |
| `seg_cellsam_use_wsi` | `true` | Enable CellSAM native whole-slide tiling. |
| `seg_cellsam_block_size` | `1024` | Tile side (px) in WSI mode (recommend 256–2048). |
| `seg_cellsam_overlap` | `56` | Tile overlap (px) in WSI mode. |
| `cellsam_model_path` | `null` | Pre-downloaded weights. If `null`, weights auto-download and require `DEEPCELL_ACCESS_TOKEN`. |

!!! warning "CellSAM weights"
    With `cellsam_model_path = null`, export `DEEPCELL_ACCESS_TOKEN` before
    launching — the pipeline forwards it into the container. On clusters without
    compute-node internet, pre-download the weights and set `cellsam_model_path`.

## Quantification

Per-cell marker intensity.

| Parameter | Default | Description |
|---|---|---|
| `quantify_compartments` | `true` | Emit per-compartment signal (Nucleus / Cytoplasm / Cell) by routing the nuclear mask into quantification. |
| `expanded_quantification` | `false` | Also emit Mean and Sum per compartment (per-compartment Median is always emitted). **Requires** `quantify_compartments=true`. Off by default: Median is the statistic FlowPath's selector defaults to, and Mean+Sum triple the column count of every quantification table for a statistic most gating never reads. |

!!! danger "Validation rule"
    Setting `expanded_quantification = true` without `quantify_compartments = true`
    fails at launch with a clear error.

## Visualization & export

Pyramidal OME-TIFF assembly and GeoJSON.

| Parameter | Default | Description |
|---|---|---|
| `tilex` | `512` | Tile size (px, square) for the pyramidal OME-TIFF. |
| `pyramid_resolutions` | `8` | Number of pyramid resolution levels. |
| `pyramid_scale` | `2` | Downsampling factor between levels. |
| `compression` | `zstd` | Codec: `zstd`, `lzw`, `zlib`, `jpeg`, `none`. |
| `simplify_tolerance` | `1.0` | Douglas–Peucker tolerance (px) for GeoJSON contour simplification. Display-only — quantification is mask-derived, so this does not affect measurements. |
| `geojson_coord_precision` | `2` | Decimal places for GeoJSON coordinates. Display-only. |

## SpatialData export

An **additive** scverse-native `.zarr` store, written by default. OME-TIFF and
GeoJSON remain the primary artifacts; this recomputes nothing, it re-serializes
the same masks, polygons, measurements and QC. Full store layout:
[SpatialData & FlowPath](spatialdata.md).

| Parameter | Default | Description |
|---|---|---|
| `skip_spatialdata_export` | `false` | Skip the `.zarr` export. The store **is** written by default. |
| `spatialdata_include_image` | `false` | Also embed the pyramid image in the store. Off by default because `spatialdata.write()` materializes every element, so including it duplicates the largest artifact the run produces. Turn on for a self-contained, depositable object. |
| `spatialdata_residual_join_max_px` | `15.0` | Max centroid distance (px) when joining `WARP_SEG_QC`'s per-cell registration residuals onto `cell_mask`. The QC segmentation is a separate native-image run and shares no label space with `cell_mask`, so the join is **spatial**, not by label. |

## Quality control & reports

| Parameter | Default | Description |
|---|---|---|
| `skip_preprocess_qc` | `false` | Skip preprocessing QC thumbnails. |
| `preprocess_qc_scale_factor` | `0.25` | Downsample factor for preprocessing QC images. |
| `skip_registration_qc` | `false` | Skip registration QC overlays. |
| `qc_scale_factor` | `0.25` | Downsample factor for registration QC images. |
| `skip_postprocessing_qc` | `false` | Skip segmentation/intensity QC plots. |
| `skip_seg_quality_eval` | `true` | Skip reference-free cell-segmentation quality scoring (CSE). **Opt-in:** set `false` to enable. |
| `cse_pixel_size_um` | `null` | Pixel size (µm) passed to CSE. `null` = infer from image metadata. |
| `cse_max_pixels` | `50000000` | Bin image+masks so CSE scores at most this many pixels. `null` = full resolution. |
| `skip_final_qc_report` | `false` | Skip the aggregated HTML QC report. |
| `seg_qc_pairing` | `lsa` | Cell correspondence backend for `reg_qc=2`'s fixed anchor pairing: `lsa` = optimal one-to-one assignment (exact, per connected component), `mutual_nn` = the older mutual-nearest-centroid rule. See [Staged registration QC](registration_qc.md). |
| `seg_qc_match_radius_factor` | `1.5` | Match radius for `reg_qc=2` pairing, in median nuclear radii. |

### Reports

- **`qc/mirage_qc_report_<timestamp>.html`** — aggregated QC report: run summary,
  pipeline-stage status, preprocessing QC images, registration QC (overlays, the
  VALIS rTRE summary, per-stage STARE TRE distributions with the per-tile spatial
  heatmap, and per-stage cell-displacement distributions from the warp-seg QC),
  the feature-TRE vs cell-displacement reconciliation scatter, the per-cell
  residual distribution, segmentation overlays, and postprocessing QC.
  Controlled by `skip_final_qc_report`.
- **`qc/mirage_qc_data_<timestamp>/`** — the report's inputs, copied verbatim:
  `collated_versions.yml`, `run_summary.json` (which carries the sample manifest),
  and the `registration_tre/`, `seg_qc/`, `seg_residuals/`, `*_qc/` folders. The
  report renders plots; this folder is what a script reads. Software versions and
  the sample manifest are **here**, not in the HTML — a 200-row version table and a
  per-patient count table were two renderings of files that already existed.

### Reference-free segmentation quality (CSE) — opt-in { #cse-opt-in }

`SEG_QUALITY_EVAL` scores the shipped cell+nucleus masks against the reference
image with the [CellSegmentationEvaluator][cse] (vendored 1.5.19 subset), giving a
composite `QualityScore` with **no ground-truth annotation**. It is the signal the
segmentation arm of `benchmarks/configs/arms.yaml` uses to rank backends on real
tissue.

It is **off by default** (`skip_seg_quality_eval = true`) for one operational
reason: it needs the `segeval` image on Docker Hub, and `.github/workflows/containers.yml`
only *pushes* on a release or a manual `workflow_dispatch` — a push to `main`
builds without publishing. A default-on stage whose container may not exist yet
would fail every run of a fresh clone.

```bash
# publish the image once (Actions -> Build & Push Container Images -> Run workflow),
# then:
# not-runnable: the trailing `...` elides more flags -- illustrative, not paste-ready
nextflow run . --input samplesheet.csv --outdir results -c site.config -params-file params/seg_quality_eval.json --cse_max_pixels 50000000 ...
```

Two things to know before reading the numbers:

- **`cse_max_pixels` is not comparable across values.** Binning preserves the
  geometry and area metrics, but the composite `QualityScore` shifts with the
  factor. Keep it fixed across a cohort.
- **A dropped score is announced, not silent.** CSE's usual failure is resource
  exhaustion on full-WSI masks. Both processes retry three times (climbing the
  memory ramp) and then `ignore`, logging a warning naming the patient. Previously
  the ignore was silent, which is how the scorer came to be described as "always
  ignored" while runs stayed green.

[cse]: https://doi.org/10.1091/mbc.E22-08-0364
- **`qc/mirage_resource_report.html`** — computational-resource report built from
  the per-task size logs and Nextflow `trace.txt`: run totals plus four plots:
  wall-time by process, memory headroom (requested vs observed), the reserved
  GB·hours lost to failures and retries, and input size against runtime.
  Generated at run completion when `enable_trace` is set; re-runnable by hand
  via `bin/generate_resource_report.py`. Complements Nextflow's native
  `report.html` / `timeline.html`.

## Cluster & resources

Per-process requests, resource labels, retry policy and containers:
**[Resources](resources.md)**. Cluster invocations:
[Running on HPC / SLURM](usage.md#running-on-hpc).

| Parameter | Default | Description |
|---|---|---|
| `slurm_partition` | `null` | SLURM partition (becomes the queue). |
| `slurm_account` | `null` | SLURM account (`--account`). |
| `slurm_qos` | `null` | SLURM QoS (`--qos`). |
| `gpu_type` | `1` | GRES string for GPU jobs (`--gres=gpu:<value>`). Bare `1` = any one GPU, which is portable; set a typed string (e.g. `nvidia_a100:1`) in your site config to match `sinfo -o "%G"`. |
| `max_memory` | *(required, no default)* | Global memory ceiling. Clamps every process's request via `process.resourceLimits`. Declared `null` in `nextflow.config` on purpose — a default here would be a guess about someone else's machine — and marked `required` in the schema, so a run that omits it is refused at launch. Set it in your `site.config`. |
| `max_cpus` | *(required, no default)* | Global CPU ceiling. Same rule as `max_memory`: required, no default, set it in your `site.config`. |
| `max_time` | `240.h` | Global walltime ceiling. This one keeps a real default: overshooting a walltime ceiling costs a queue slot, not a dead run. |
| `concurrency` | `5` | The one knob to tune: drives both `max_forks` and `queue_size` together, preserving the shipped 5:20 ratio. |
| `max_forks` | `null` | Max concurrent tasks **per process**. `null` = derive from `concurrency`; also an upper bound on every per-process `maxForks`, so lowering it throttles every module. |
| `queue_size` | `null` | Max concurrent tasks across the **whole pipeline** (`executor.queueSize`). `null` = derive from `concurrency` (`concurrency * 4`). Normally the binding constraint. |

`max_forks` and `queue_size` are a pair: the lower of the two binds, and they are only
meaningful together. `--concurrency` moves both at once; `--max_forks` / `--queue_size`
override it individually for the asymmetric case. At the shipped defaults `max_forks` (5)
is far lower, so raising `queue_size` alone has no effect — raise `max_forks` (or
`concurrency`), or raise both. See
[Resources → Execution & concurrency](resources.md#execution-concurrency).

!!! warning "Not in a `-c site.config` — refused at launch"
    `concurrency`, `max_forks`, `queue_size`, `cleanup_work`, `enable_trace` and
    `trace_dir` each drive a scalar evaluated while `nextflow.config` is parsed, before
    any `-c` file is merged, so a pin there changes the param and not the setting. The
    pipeline refuses such a run at launch (`ParamUtils.validateFrozenConfig`) instead of
    silently ignoring it. Pass them with `-params-file` or on the command line.

!!! info "How resources scale"
    Per-process memory and time scale with `task.attempt`, bounded by the
    ceilings above, so a retry after an OOM kill automatically climbs the ramp.
    `-profile local` lowers the ceilings to 4 CPUs / 16 GB / 72 h. Every
    process's `cpus`/`memory`/`time` has exactly one owner — a resource label
    **or** a `withName:` block, never both. Full table:
    [Resources → Per-process requests](resources.md#per-process-requests).

## Tracing

| Parameter | Default | Description |
|---|---|---|
| `enable_trace` | `true` | Write Nextflow `trace.txt`, `report.html`, `timeline.html`, and per-task size logs. |
| `trace_dir` | `.trace` | Directory for Nextflow's own `trace.txt` / `report.html` / `timeline.html`. Resolved against the **launch directory**, not `--outdir` — a relative value therefore follows where you ran `nextflow`, not where results go. The computational-resource report itself is written to `<outdir>/qc/` (see [Cluster & resources](#cluster-resources)); this directory only supplies its `trace.txt` input. |

## Output cleanup

| Parameter | Default | Description |
|---|---|---|
| `cleanup_level` | `'none'` | Which published outputs a run keeps: `none` (everything) or `final` (final artifacts only). |
| `cleanup_work` | `true` | Delete the Nextflow work directory after a **successful** run. |

### `cleanup_level`

At `none` — the default since 2026-09-10 (it was `final` from 2026-08-25) — a run
publishes everything, and its output can be re-entered by `--start`.
At `final`, the opt-in cleaning level, a run publishes only what it was run to produce:

| Kept | Dropped |
|---|---|
| `<pid>/pyramid/`, `<pid>/geojson/`, `<pid>/quantification/`, `<pid>/spatialdata/` | `<pid>/converted/`, `<pid>/preprocessed/`, `<pid>/registered/`, `<pid>/segmentation/`, `<pid>/split_channels/`, `<pid>/quantify/`, `<pid>/cell_properties/` |
| `<pid>/qc/`, run-level `qc/`, `size_logs/` | — |
| `csv/README.txt` | every `csv/*.csv` checkpoint manifest |

The intermediates are **never published**, not published and then deleted. Every
`publishDir` uses `mode: 'copy'`, so publish-then-delete would pay a full copy out
of `work/` for a file nobody reads.

The surviving set has one owner, `Layout.FINAL_KINDS`; the gate at each
`publishDir` is an inlined copy of one predicate, held to it by
`tests/test_cleanup_publish_gates.py`. The predicate sits inside a `saveAs:`
closure, which Nextflow resolves per file at publish time, so the level takes
effect however it arrives — `--cleanup_level` on the command line, a
`-params-file`, a profile, or a `-c site.config` (`tests/cleanup_level_pin.sh`
runs all three of the latter against the stub).

**No checkpoint manifest is written at `final`.** A manifest names *where* a step's
artifacts landed, and at this level they landed nowhere — so every row would record
a path that does not exist. That includes `postprocessed.csv`, which is not obvious:
its `cell_csv`/`cell_geojson`/`merged_csv`/`pyramid` columns are all final artifacts,
but its `cell_mask` column names a segmentation mask, which is an intermediate.
`csv/README.txt` is written instead so an empty `csv/` is not a mystery.

Keep the default `none` (or pass `--cleanup_level none` explicitly) when this run's
output will be **re-entered**:

* `--start <step>` reads the published intermediates of a previous run.

`--start` past `preprocessing` only warns: this run reads a *prior* tree fine; what
it cannot do is be re-entered the same way itself.

### `cleanup_work`

On by default. Nextflow removes the work directory during session teardown once the
run completes successfully. A **failed** run keeps it, so failure evidence always
survives.

**Safe to enable:** published outputs are unaffected. Every `publishDir` in
`conf/modules.config` uses `mode: 'copy'`, so nothing under `--outdir` is a symlink
into `work/`, and no checkpoint CSV records a path inside the work directory
(asserted by `tests/checkpoint_manifest.nf.test`). Restarting a later stage with
`--start` therefore still works after cleanup, because `--start` reads published
paths under `--outdir`.

**What it does not remove.** Nextflow's cleanup reclaims *task* directories. The
`collectFile()` scratch under `work/tmp/` and `work/collect-file/`, plus the empty
two-character task-directory shells, survive. On the stub dataset that is 40 KB out
of 1.0 MB — the task outputs, which are the part that actually grows with real
slides, are all gone. Use `nextflow clean` or remove `work/` yourself if you need
the directory to disappear entirely.

!!! warning "Mutually exclusive with `-resume`"
    A cleaned run leaves no cache to resume from, and **this does not fail loudly**:
    re-running with `-resume` against a cleaned work directory exits 0 and silently
    re-runs every task. Verified on the stub dataset — 57 cache hits against an
    intact `work/`, 0 against a cleaned one, both green. If you iterate with
    `-resume`, leave `cleanup_work` off. See
    [Checkpoints & resuming](usage.md#checkpoints-resuming).

!!! note "Why this used to default to off"
    `conf/modules.config`'s QC `errorStrategy` used to end in `'ignore'`, so a run
    could exit 0 with a QC task having failed — and cleanup deleted that task's work
    directory, the only place its logs and partial outputs lived. That branch is
    gone: the QC selector is `retry-then-fail`, so a broken QC task fails the run
    and a green run implies a complete output tree. Two processes still carry a
    deliberate `retry-then-drop` policy (`SEG_QUALITY_EVAL`, `MERGE_SEG_EVAL`, both
    opt-in); when either is dropped, `main.nf`'s `onComplete` says so by name in the
    run log rather than leaving it to be inferred from an absent column.

## Run mode and mask embedding

| Parameter | Default | Description |
|---|---|---|
| `mode` | `standard` | Recorded in `qc/run_summary.json`. Only `standard` (the `--start`/`--stop` step flow) exists on this branch; the `dev` branch adds `add_cycle` (incremental cyclic-IF). |
| `embed_masks` | `false` | Embed the segmentation masks as a second uint32 series in the pyramid OME-TIFF. Written only when `embed_masks && quantify_compartments && expanded_quantification`; `embed_masks = true` with either sibling off is rejected **at launch** (`ParamUtils.validateCompartmentQuant`) rather than silently producing a plain pyramid. |

## Parameter presets

Ready-made `-params-file` JSON presets in `params/`:

| Preset | Purpose |
|---|---|
| `params/full_pipeline.json` | All stages from preprocessing, on the InstanSeg backend (see the file's `_comment_seg_method` for the StarDist switch) |
| `params/preprocessing_only.json` | Preprocessing only |
| `params/registration_only.json` | Registration from a checkpoint |
| `params/postprocessing_only.json` | Postprocessing from a checkpoint |
| `params/test.json` | Minimal self-contained test run (carries its own small ceilings) |
| `params/dry_run.json` | `dry_run` alone — combine with any other flags |
| `params/seg_quality_eval.json` | Turns the opt-in CSE segmentation-quality scorer on |

```bash
nextflow run . -profile slurm,singularity -c site.config \
  -params-file params/full_pipeline.json \
  --input samplesheet.csv --outdir results \
  --seg_method cellsam            # override any preset value on the CLI
```

!!! warning "Presets never carry site sizing — that's `-c site.config`'s job (R4)"
    `full_pipeline.json`, `preprocessing_only.json`, `registration_only.json` and
    `postprocessing_only.json` deliberately do **not** set `max_cpus`/`max_memory`/
    `max_time` (their `_comment_resources` entry says so). A `-params-file` value
    **overrides a `-c` config file** (Nextflow's own precedence order), so a
    preset that baked in one cluster's ceiling would silently defeat any
    `site.config` layered on top of it on a different cluster — that mistake
    shipped in these four presets before ruling R4. `max_cpus`/`max_memory` have
    no default and are `required` in `nextflow_schema.json`, so every command
    that loads one of these presets needs `-c site.config`
    (`cp conf/site.config.template site.config`, then edit the ceiling) or the
    run is refused at launch with `Missing required parameter(s)`.
    `tests/test_param_presets_are_coherent.py` enforces this for every preset
    except `test.json`, which intentionally pins tiny self-contained values for
    the test suite.

## See also

- :material-sitemap: **What each process does, in order** — [The pipeline](pipeline.md)
- :material-file-tree: **Every input column and output path** — [Inputs & outputs](outputs.md)
- :material-server: **Every resource request, retry rule and container** — [Resources](resources.md)
- :material-console: **Command recipes & the samplesheet** — [Usage](usage.md)
- :material-help-circle: **Common questions** — [Usage → Troubleshooting & FAQ](usage.md#troubleshooting-faq)
- :material-book-open-variant: **How to cite** — [Citation](citation.md)
