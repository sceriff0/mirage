# Outputs

<p class="standfirst">Everything a run writes: the tables you will analyse, the complete published
tree, what a run records about itself, and the measurement-key grammar that the QuPath/FlowPath side
depends on.</p>

What a run reads is on [The samplesheet](samplesheet.md); the checkpoint CSVs are on
[Checkpoints & resuming](resuming.md).

!!! abstract "Canonical sources"
    - **Published paths** — `lib/Layout.groovy` + the `publishDir` blocks in `conf/modules.config`
    - **Measurement keys** — `bin/utils/measurements.py`

    `lib/Layout.groovy` is the single owner of published-path and checkpoint-CSV
    layout. `tests/test_layout.py` forbids hand-written paths elsewhere, and
    `tests/checkpoint_manifest.nf.test` asserts that **every path recorded in
    every checkpoint CSV exists on disk and does not lie inside the work
    directory**.

---

## The tables you will analyse

Two locations: per-patient results under `<outdir>/<patient_id>/`, and run-level
aggregates at the `<outdir>` root.

| File | What it is |
|---|---|
| `quantification/merged_quant.csv` | One row per cell; morphology joined in. Per marker: a bare column (whole-cell mean, kept for FlowPath's fast path) plus `<MARKER>: <Compartment>: <Statistic>` keys. **Median is always emitted**; `--expanded_quantification` adds Mean and Sum; `--quantify_compartments` adds the Nucleus and Cytoplasm compartments alongside Cell. |
| `geojson/export/cells.geojson` | One QuPath feature per cell: whole-cell polygon + measurement array (centroid µm, marker intensities, morphology). Carries `nucleusGeometry` in compartment mode. |
| `geojson/export/cells_wholecell.geojson` | The same detections without nucleus geometry — lighter and faster to import. Compartment mode only. |
| `geojson/export/cells_data.csv` | The cell table with per-marker **z-scores** added. |
| `segmentation/*_cell_mask.tif` | Whole-cell instance labels (uint32); each non-zero value is one cell. |

These open directly in QuPath, napari, and OMERO, and feed
[FlowPath](https://flowpath.readthedocs.io/) for interactive gating. The exact
key grammar is a cross-repository contract — see
[The measurement-key contract](#the-measurement-key-contract).


### What a default run publishes

`--cleanup_level` decides how much of the tree below is written. The default,
`none`, publishes the complete tree shown in the next
section. The opt-in `final` publishes only what the run was asked to produce:

```text
results/                              # = --outdir, at --cleanup_level=final
├── <patient_id>/
│   ├── quantification/               # merged_quant.csv
│   ├── geojson/export/               # cells.geojson, cells_wholecell.geojson, cells_data.csv
│   ├── pyramid/                      # pyramid.ome.tiff
│   ├── spatialdata/                  # <patient_id>.zarr
│   └── qc/                           # the full per-patient QC tree
├── qc/                               # run-level report + resource report
├── csv/README.txt                    # why there is no checkpoint manifest
└── size_logs/
```

Everything else in the full tree below — `converted/`, `preprocessed/`,
`registered/`, `segmentation/`, `cell_properties/`, `split_channels/`,
`quantify/`, and every `csv/*.csv` manifest — is an **intermediate**, and at
`final` it is never published rather than published and then deleted. Every
`publishDir` is `mode: 'copy'`, so publish-then-delete would pay a full copy out
of `work/` for a file nobody reads.

`--cleanup_level none`, the default, publishes the complete tree below. Keep it whenever the output will be re-entered by `--start
<step>`. Details: [Output cleanup](parameters.md#output-cleanup).

### The published tree (`--cleanup_level none`)

The per-patient leaf directories are the closed vocabulary declared by
`Layout.PUBLISHED_KINDS` — a process publishing anywhere else under
`<outdir>/<patient_id>/` fails `tests/test_layout.py`. There are two per-patient
exceptions: `spatialdata/`, where `EXPORT_SPATIALDATA` publishes at the patient
root with its own `pattern:` supplying the directory name (no `Layout` kind for
it), and `qc/`, which `tests/test_layout.py` explicitly excludes from the check
because no checkpoint CSV names it — it is likewise absent from
`Layout.PUBLISHED_KINDS`.

```text
results/                              # = --outdir
├── <patient_id>/
│   ├── converted/                    # <name>.ome.tif        — CONVERT_IMAGE (nuclear → ch0)
│   ├── preprocessed/                 # *_corrected.ome.tif   — APPLY_PROFILES (BaSiC)
│   │                                 #   absent by default (skip_preprocessing); csv/preprocessed.csv
│   │                                 #   then points at converted/ instead
│   ├── registered/
│   │   ├── registered_slides/        # *_registered.ome.tiff — REGISTER (VALIS)
│   │   ├── registered/               # *_registered.ome.tiff — TILED_STITCH
│   │   ├── manifest/                 # *_manifest.json       — tiled backend
│   │   ├── summary/                  # *.csv                 — VALIS error summary
│   │   ├── transform/                # *_registrar.pickle    — REGISTER (VALIS); see caveat below
│   │   └── controls/                 # *_ctrl.json, per tile — TILED_REG_TILE (tiled backend)
│   ├── segmentation/                 # *_cell_mask.tif, *_nuclei_mask.tif  — SEGMENT
│   ├── cell_properties/
│   │   ├── morphology.csv            # EXTRACT_CELL_PROPERTIES
│   │   ├── contours.json
│   │   └── nuclei/                   # morphology.csv, contours.json — EXTRACT_NUCLEI_PROPERTIES
│   ├── split_channels/               # <MARKER>.tiff, one per marker — SPLIT_CHANNELS
│   ├── quantify/                     # <id>_quant.csv, per-marker, pre-merge — QUANTIFY
│   ├── quantification/               # merged_quant.csv      — MERGE_QUANT_CSVS
│   ├── geojson/
│   │   └── export/                   # cells.geojson, cells_wholecell.geojson,
│   │                                 #   cells_data.csv        — EXPORT_GEOJSON
│   ├── pyramid/                      # pyramid.ome.tiff      — MERGE_AND_PYRAMID
│   ├── spatialdata/                  # <patient_id>.zarr     — EXPORT_SPATIALDATA
│   └── qc/
│       ├── preprocess/
│       │   └── qc/                   # *.png — GENERATE_PREPROCESS_QC
│       ├── registration/             # *_seg_qc.json — WARP_SEG_QC
│       │   │                         # *_tre.json    — TILED_SOLVE (tiled backend)
│       │   ├── qc/                   # *_QC_RGB.{png,tif}, *_QC_RGB_fullres.tif
│       │   │                         #   (one 2-panel before/after figure per
│       │   │                         #   moving slide) — GENERATE_REGISTRATION_QC
│       │   └── geojson/              # *.geojson — SEG_QC_GEOJSON (reg_qc=2)
│       └── postprocessing/
│           └── qc/                   # *.png — GENERATE_POSTPROCESSING_QC
├── csv/                              # checkpoint CSVs, all patients
├── size_logs/                        # input_sizes.csv, versions.yml — AGGREGATE_SIZE_LOGS
└── qc/                               # mirage_qc_report_<timestamp>.html — GENERATE_QC_REPORT
                                      # mirage_resource_report.html — main.nf's
                                      #   workflow.onComplete handler, not a process
```

!!! info "The QC report is plots; the data folder beside it is data"
    `mirage_qc_report_<timestamp>.html` is a self-contained page of images and
    hand-rolled inline SVG: per-stage registration-error distributions, the
    per-tile spatial heatmap, a log-log scatter of feature-TRE against cell
    displacement with the 3× divergence band drawn, and the per-cell residual
    distribution. It carries **no** software-version table and **no** sample
    manifest table, because both are files: `mirage_qc_data_<timestamp>/`
    (published beside the report) holds `collated_versions.yml` and
    `run_summary.json` verbatim, along with every CSV and JSON the plots were
    built from. Read the page; parse the folder.

!!! info "Why `geojson/export/`, `registered/registered_slides/` and the repeated `qc/`"
    A process that writes into a named subdirectory of its task directory and
    publishes with a `pattern:` naming that subdirectory gets the subdirectory
    carried into the published path. `Layout.publishedPath` reproduces that
    faithfully rather than flattening it, so a checkpoint row always names the
    real file (`<pid>/geojson/export/cells.geojson`).

    The same rule is why the three QC renderers land under a *second* `qc/`:
    each declares `path("qc/*.png")` and publishes with `pattern: "qc/*.png"`,
    so the real path is `<pid>/qc/preprocess/qc/*.png`, not
    `<pid>/qc/preprocess/*.png`.
    `WARP_SEG_QC` and `SEG_QC_GEOJSON` declare flat patterns and are unaffected.

!!! warning "`registered/transform/*_registrar.pickle` is a Python pickle"
    It is the pipeline's own output — this is a caveat, not a blocker — but a
    pickle's `load()` executes arbitrary code as a side effect of deserializing,
    and this one embeds absolute host paths and slide filenames from the run
    that produced it. Only load a registrar pickle from an `--outdir` you trust.

### Trace outputs

Written to `--trace_dir` (default `.trace`, **independent of `--outdir`**) when
`--enable_trace` is on (the default):

| File | Content |
|---|---|
| `.trace/trace.txt` | Nextflow per-task trace: status, exit, duration, `peak_rss`, `peak_vmem`, `rchar`, `wchar` |
| `.trace/report.html` | Nextflow execution report |
| `.trace/timeline.html` | Nextflow timeline |

### The primary artifacts

<div class="outs">
  <div class="out">
    <div class="p">geojson/export/cells.geojson</div>
    <div class="role">primary · QuPath / FlowPath</div>
    <ul>
      <li>One feature per cell</li>
      <li>Whole-cell polygon + <code>nucleusGeometry</code> in compartment mode</li>
      <li>Native QuPath measurement array</li>
    </ul>
  </div>
  <div class="out">
    <div class="p">pyramid/pyramid.ome.tiff</div>
    <div class="role">primary · image</div>
    <ul>
      <li>512 px tiles, up to 8 levels, scale 2, zstd</li>
      <li>Channel names preserved</li>
      <li>Optional uint32 mask series (<code>embed_masks</code>)</li>
    </ul>
  </div>
  <div class="out">
    <div class="p">quantification/merged_quant.csv</div>
    <div class="role">primary · table</div>
    <ul>
      <li>One row per cell</li>
      <li>All markers × compartments × statistics</li>
      <li>Morphology joined in</li>
    </ul>
  </div>
  <div class="out">
    <div class="p">spatialdata/&lt;pid&gt;.zarr</div>
    <div class="role">additive · scverse</div>
    <ul>
      <li>Masks, polygons, AnnData table</li>
      <li>Centroids + per-cell residual</li>
      <li>QC verbatim + provenance</li>
    </ul>
  </div>
</div>

`geojson/export/` holds three files (`bin/export_geojson.py`):

| File | Content |
|---|---|
| `cells.geojson` | QuPath-native detections: each whole-cell polygon (simplified at `simplify_tolerance`, default 1.0 px; coordinates to `geojson_coord_precision`, default 2 decimals) with its measurements, and the nuclear polygon as `nucleusGeometry` |
| `cells_wholecell.geojson` | The same cells without `nucleusGeometry`, for faster loading |
| `cells_data.csv` | One row per cell: the raw values plus a `<column>_zscore` for each marker column, computed column-wise across all cells of the patient |

---

## Provenance: what a run records about itself

Every process emits a `versions.yml`, and every process emits one row into a
`*.size.csv` (`process,sample_id,filename,bytes` —
`lib/ProcessEnvelope.groovy`'s `SIZE_LOG_COLUMNS`). Both are aggregated at the
end of the run.

```yaml
"MIRAGE:PREPROCESSING:CONVERT_IMAGE":
    python: 3.10.14
    container: bolt3x/mirage-convert:1.0.0
    tifffile: 2024.12.12
    bioio: 3.5.0
    h5py: 3.11.0
```

The `container:` line is `task.container` as Nextflow resolved it, so it records
the image that *actually ran* — not just the one a config file names. Every
`FROM` in `containers/*/Dockerfile` is digest-pinned, so this field plus the
Dockerfile identifies the exact base bytes a result came from (a first-party
`bolt3x/mirage-*` image is itself tag-pinned rather than digest-pinned until a
release publishes it — see
[Running on a cluster → Pre-pulling container images](hpc.md#pre-pulling-container-images)).
In a stub run every value reads `stub`, which is how a stubbed tree is told
apart from a real one.

| Artifact | Where | What it is |
|---|---|---|
| `versions.yml` | per task, aggregated into `qc/mirage_qc_data_<timestamp>/collated_versions.yml` | tool versions + the resolved container, per process |
| `*.size.csv` | per task, aggregated into `size_logs/input_sizes.csv` | one row per task: `process,sample_id,filename,bytes` — the byte total of that task's inputs |
| `qc/` (run level) | `<outdir>/qc/` | the aggregated HTML QC report plus its `mirage_qc_data_<timestamp>/` folder, and the self-contained `mirage_resource_report.html` |
| `<trace_dir>/` | resolved against the **launch directory**, not `--outdir` | Nextflow's own `trace.txt`, `report.html`, `timeline.html` only |

!!! note "The resource report lives in `<outdir>/qc/`; only its INPUT lives under `trace_dir`"
    `mirage_resource_report.html` is written to `<outdir>/qc/` by `main.nf`'s
    `workflow.onComplete` handler (`Layout.runDir(cfg.outdir, 'qc')`), not into
    `trace_dir` — `--trace_dir` only names where it *reads* `trace.txt` from,
    and where it points its own `report.html`/`timeline.html` links. `--trace_dir`
    defaults to `.trace` and is resolved against the launch directory, matching
    where Nextflow itself resolves `trace.file`. A run launched from a different
    directory reads its traces from somewhere else, and the resource report names
    the exact path it looked in when it finds nothing there — see
    [Resources → When no report appears](resources.md#when-no-report-appears) —
    rather than silently producing an empty report.

## The measurement-key contract

`quantify.py` produces, and `export_geojson.py` / `export_spatialdata.py`
consume, a single key grammar:

```text
"<marker>: <Compartment>: <Statistic>"
```

| Vocabulary | Values | Governed by |
|---|---|---|
| `<Compartment>` | `Nucleus`, `Cytoplasm`, `Cell` | `--quantify_compartments` — when `false`, only `Cell` is produced |
| `<Statistic>` | `Median`, `Mean`, `Sum` | `Median` is **always** produced; `--expanded_quantification` adds `Mean` and `Sum` |

So a **default** run (`quantify_compartments=true`, `expanded_quantification=false`)
emits three keys per marker:

```text
CD3: Nucleus: Median      CD3: Cytoplasm: Median      CD3: Cell: Median
```

Turning `expanded_quantification` on (in a `-params-file`; it requires
`quantify_compartments`) adds Mean and Sum in each compartment, for nine:

```text
CD3: Nucleus: Median      CD3: Nucleus: Mean      CD3: Nucleus: Sum
CD3: Cytoplasm: Median    CD3: Cytoplasm: Mean    CD3: Cytoplasm: Sum
CD3: Cell: Median         CD3: Cell: Mean         CD3: Cell: Sum
```

Plus one **bare** column per marker (`CD3`), which is the whole-cell *mean* and is
written in every mode — FlowPath's bare-key fast path is hard-wired to
(whole cell, Mean), so it stays a mean even when Mean is not otherwise emitted.

!!! danger "Do not change this format"
    The grammar is **case- and space-sensitive** and is consumed by the sibling
    repository `qupath-extension-flowpath`. `measurement_key()` in
    `bin/utils/measurements.py` is the single producer of the string; the
    `COMPARTMENTS` and `STATISTICS` tuples are the single vocabulary. Changing
    any of the three requires a coordinated change on the FlowPath side.

!!! warning "A missing key is a real state, not a zero"
    `export_geojson.py` writes a measurement only when its value is present
    (`if pd.notna(val)`), so **a cell with no nuclear overlap carries fewer keys
    than its neighbours** rather than carrying them as `0.0`. That is correct — a
    cell with no nucleus has no nuclear median, and `0.0` is a measurement, not an
    absence — but it means a consumer must treat "absent" as a state rather than
    reading a number. A language whose subscript raises (Python) needs `.get()`;
    one that returns null (Java, as `qupath-extension-flowpath` does) already
    handles it, and the effect there is on distributions instead — percentiles
    and z-scores exclude the missing cells rather than counting artificial `0.0`s
    in the low tail.

    The same applies to the morphology keys (`Eccentricity`, `Perimeter µm`,
    `Solidity`, `Convex Area µm²`, `Major/Minor Axis Length µm`, `Area µm²`).

Cytoplasm is computed as `Cell − Nucleus` by subtraction, per cell. A cell with
no nuclear overlap has a **missing** (`NaN`) Nucleus value and a Cytoplasm value
equal to its whole-cell value; a compartment with no pixels is `NaN`, never `0.0`
(`bin/quantify.py`, guarded by `tests/test_absent_compartment_is_nan.py`). The
row is kept either way, so the row count is identical across every compartment.

### Non-marker columns

`MORPHOLOGY_COLS` — the twelve columns in `merged_quant.csv` that are geometry
or identity rather than marker signal, and are therefore excluded from every
marker sweep:

```text
label · y · x · area · eccentricity · perimeter · convex_area
axis_major_length · axis_minor_length · solidity · fov · cell_size
```

`label` is the segmentation instance id — the **only stable cell identifier the
pipeline produces**. It is the SpatialData store's `instance_key`, so every join
is keyed on it and a mismatch is fatal rather than silently positional.

---

## See also

- :material-tune: **Every parameter** — [Parameters](parameters.md)
- :material-table: **What a run reads** — [The samplesheet](samplesheet.md)
- :material-restart: **The checkpoint CSVs** — [Checkpoints & resuming](resuming.md)
- :material-server: **Every resource request** — [Resources](resources.md)
- :material-database: **The `.zarr` layout in detail** — [SpatialData & FlowPath](spatialdata.md)
- :material-chart-scatter-plot: **`*_seg_qc.json` schema** — [Registration QC](registration_qc.md)
