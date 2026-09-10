# The pipeline

<p class="standfirst">A step-gated Nextflow DSL2 workflow that aligns multiplexed immunofluorescence
slides, segments every cell, quantifies each marker per compartment, and exports QuPath-compatible
GeoJSON, a pyramidal OME-TIFF, and a scverse-native SpatialData store. The nuclear/fiducial channel is
resolved by marker name from image metadata and drives both segmentation and registration.</p>

This page is the site rendering of **Supplementary Figure S1** —
[open the figure itself](figures/pipeline-schematic.html){ target=_blank }, a
single self-contained page that also opens straight from the filesystem. Every
process, default and path below is read from the pipeline source —
`nextflow.config`, `conf/modules.config`, `lib/ParamUtils.groovy` and
`lib/Layout.groovy`.

!!! abstract "The supplementary figure set"
    - **S1 · pipeline** — [figure](figures/pipeline-schematic.html){ target=_blank } · this page
    - **S2 · registration** — [figure](figures/registration-schematic.html){ target=_blank } ·
      the two backends, step by step
    - **S3 · quality control** — [figure](figures/qc-schematic.html){ target=_blank } ·
      the tagged artifact stream, the kind vocabulary, and the retry-then-fail contract
    - **S4 · lazy reads** — [figure](figures/zarr-schematic.html){ target=_blank } ·
      where lazy zarr reads cut peak memory, and every place they cannot help
    - **S5 · coarse alignment** — [figure](figures/coarse-schematic.html){ target=_blank } ·
      why the STARE global pose is a learned matcher, and what that costs in memory
    - **S6 · accuracy measures** — [figure](figures/accuracy-schematic.html){ target=_blank } ·
      the four registration-accuracy numbers, which two are scored on the registrar's own features, and why none is ground truth

---

## a — Input: the samplesheet

Every run is driven by a samplesheet CSV — no images are needed to describe the
design. Each row is one image; exactly one image per patient is the registration
reference (`is_reference=true`). Channel identity is declared metadata, never
parsed from filenames.

| `patient_id` | `path_to_file` | `is_reference` | `channels` |
|---|---|---|---|
| `P001` | `c1.ome.tif` | `true` | `DAPI\|PanCK\|CD45` |
| `P001` | `c2.ome.tif` | `false` | `DAPI\|CD8\|CD68` |
| `P001` | `c3.ome.tif` | `false` | `DAPI\|PD-L1\|Ki-67` |
| `P002` | `p2c1.ome.tif` | `true` | `DAPI\|PanCK\|CD45` |
| `P002` | `p2c2.ome.tif` | `false` | `CELLTOX\|CD3\|FoxP3` |

With `nuclear_markers = ['DAPI', 'CELLTOX']`, DAPI is used as the nuclear /
fiducial channel where present; the last row has no DAPI, so CELLTOX is used
instead. That channel is moved to index 0 at conversion. Per-patient image counts
are read from this file up front and injected into channel metadata, so every
downstream grouping **streams** — patient B preprocesses while patient A
registers — instead of waiting for the whole cohort.

Full column contract, including the different columns each `--start` requires:
[Inputs & outputs → The samplesheet](outputs.md#the-samplesheet).

---

## b — Execution gate: `--start` / `--stop`

Four ordered steps. Each writes a checkpoint CSV, so any step can resume from a
prior run's output.

<div class="gate">
  <div class="g"><div class="k">step 1 · default start</div><div class="v">preprocessing</div><div class="d">Standardize to OME-TIFF; BaSiC illumination correction (optional).</div></div>
  <div class="g"><div class="k">step 2</div><div class="v">registration</div><div class="d">Align each moving slide to the patient's reference.</div></div>
  <div class="g"><div class="k">step 3</div><div class="v">segmentation</div><div class="d">Detect every cell; extract contours and morphology.</div></div>
  <div class="g"><div class="k">step 4 · default stop</div><div class="v">postprocessing</div><div class="d">Quantify and export.</div></div>
</div>

<div class="gnote">
  <span><b>--start preprocessing</b> (default)</span>
  <span><b>--stop</b> omitted → run to end</span>
  <span><b>--start X --stop X</b> → one step</span>
</div>

!!! info "The step vocabulary has exactly one owner"
    `ParamUtils.STEPS` is the single ordered table of what a step *is* — its
    `name`, the samplesheet `requiredColumns` when it is the entry point, the
    checkpoint `entryColumn` it reads, and the `qcKinds` it contributes.
    `STEP_ORDER`, `requiredColumnsForStep`, the entry-column lookup in
    `workflows/mirage.nf` and `ParamUtils.knownArtifactKinds()` (called by
    `final_qc.nf`) are all
    derived from it. `--stop` naming an earlier step than `--start` is rejected
    at launch.

---

## c — Modules and default parameters

Each lane is one step's subworkflow; boxes are processes in execution order.
Chips give real defaults. Every process runs in a pinned container and emits
`versions.yml`.

<div class="flow">

  <div class="lane">
    <h3><span>PREPROCESSING</span><span>step 1</span></h3>
    <div class="body">
      <div class="mod"><div class="n">CONVERT_IMAGE</div>
        <div class="x">Standardize any vendor format to OME-TIFF. Resolves the nuclear marker by
          name from metadata and moves it to channel 0; fails fast if none present.</div>
        <div class="pp"><span>nuclear_markers <b>DAPI,CELLTOX</b></span></div></div>
      <div class="mod"><div class="n">TILE_FOR_BASIC → BASICPY → APPLY_PROFILES <span class="tag">opt</span></div>
        <div class="x">BaSiC illumination correction, as three processes rather than one: nf-core's
          <b>BASICPY</b> computes flatfield/darkfield <i>profiles only</i> (mcmicro applies them
          downstream inside ASHLAR, which mirage does not have) and refuses a single-sited image,
          which every stitched mirage slide is. <b>TILE_FOR_BASIC</b> writes the non-overlapping
          pseudo-FOV grid onto the Z axis so the module sees one site per tile and decides the
          fiducial skip once; <b>APPLY_PROFILES</b> does the division/subtraction and reassembles
          the slide. The nuclear/fiducial channels are left uncorrected by default: they drive both
          registration and segmentation, so correcting them changes what each consumes.
          <b>skip_preprocessing</b> (the default since 1.0.0's final round) turns the correction off
          entirely — conversion still runs, so
          the step still emits one image per slide and the checkpoint still has a row per slide,
          pointing at <code>converted/</code> instead.</div>
        <div class="pp"><span>skip_preprocessing <b>true</b></span><span>skip_nuclear <b>true</b></span><span>tile <b>1950</b></span></div></div>
      <div class="mod"><div class="n">GENERATE_PREPROCESS_QC <span class="tag">opt</span></div>
        <div class="x">Per-channel downsampled PNG for visual inspection.</div>
        <div class="pp"><span>scale <b>0.25</b></span></div></div>
      <div class="ckpt">→ csv/preprocessed.csv</div>
    </div>
  </div>

  <div class="lane">
    <h3><span>REGISTRATION</span><span>step 2</span></h3>
    <div class="body">
      <div class="mod"><div class="n">group by patient · resolve reference</div>
        <div class="x">The reference is <b>declared, never inferred</b>: exactly one
          <code>is_reference=true</code> row per patient, resolved once at samplesheet read.
          None is a hard error at launch, at every entry point; there is no promotion rule.
          Patients whose only slide is the reference bypass the backend entirely —
          a lone image has no transform to solve — and pass through unregistered.</div>
        <div class="pp"><span>is_reference <b>exactly one</b></span></div></div>
      <div class="grp">
        <div class="bh">◇ registration_method</div>
        <div class="opt"><div class="oh"><span>valis → REGISTER</span><span class="tag-def">default</span></div>
          <div class="ox">Feature-based rigid + non-rigid warp (JVM / Bio-Formats), resolving all
            slides into a shared space. Micro-registration at micro-rigid depth by default
            (<code>reg_micro_reg=1</code>); <code>2</code> adds the non-rigid pass.</div>
          <div class="pp"><span>memory_mode <b>high</b></span><span>micro_reg <b>1</b></span><span>max_dim <b>4000</b></span></div></div>
        <div class="opt"><div class="oh"><span>tiled → STARE</span><span class="tag-alt">method=tiled</span></div>
          <div class="ox">JVM-free tiled rigid + mesh warp into the reference's shape; fiducial is
            channel 0. <b>Not</b> laptop-sized at the shipped tier — COARSE asks 48 GB at
            <code>reg_tiled_mode=high</code>, ~5 GB at <code>low</code>. The per-tile fan-out
            is the only shape.</div>
          <div class="pp"><span>tile <b>2048</b></span><span>halo <b>256</b></span><span>gate_tre <b>1.0</b></span></div></div>
      </div>
      <div class="mod"><div class="n">GENERATE_REGISTRATION_QC <span class="tag tag-on">reg_qc≥1</span></div>
        <div class="x">Nuclear-channel overlay, registered vs reference.</div></div>
      <div class="mod"><div class="n">SEG_QC_SEGMENT → SEG_QC_GEOJSON → WARP_SEG_QC <span class="tag tag-on">reg_qc=2</span></div>
        <div class="x">Segment the native slides with <b>the run's own segmenter</b>
          (<code>SEGMENT</code> aliased, so <code>--seg_method</code> applies here too), trace
          the cell mask into polygons, warp them through the transform and score Dice and
          centroid displacement per stage. Also emits a <b>per-cell residual</b> table
          consumed by the SpatialData export. Default on.</div></div>
      <div class="ckpt">→ csv/registered.csv</div>
    </div>
  </div>

  <div class="lane">
    <h3><span>SEGMENTATION</span><span>step 3</span></h3>
    <div class="body">
      <div class="mod"><div class="n">SEGMENT <span class="tag">reference slide</span></div>
        <div class="x">Detect every cell. Always emits both a whole-cell mask and a nuclei mask,
          whichever backend runs.</div>
        <div class="pp"><span>gpu <b>true</b></span></div></div>
      <div class="grp">
        <div class="bh">fork — seg_method</div>
        <div class="opt"><div class="oh"><span>instantseg</span><span class="tag-def">default</span></div>
          <div class="ox">Channel-invariant; native cell + nuclei output.</div></div>
        <div class="opt"><div class="oh"><span>stardist</span><span class="tag-alt">needs model</span></div>
          <div class="ox">Nuclear channel 0 only; cells expanded from nuclei.</div></div>
        <div class="opt"><div class="oh"><span>cellsam</span><span class="tag-alt">gated weights</span></div>
          <div class="ox">Nuclear channel by name; cells expanded from nuclei.</div></div>
      </div>
      <div class="mod"><div class="n">EXTRACT_CELL_PROPERTIES</div>
        <div class="x"><code>regionprops</code> once, not once per marker: <code>morphology.csv</code>
          plus simplified cell contours.</div>
        <div class="pp"><span>simplify <b>1.0</b></span><span>precision <b>2</b></span></div></div>
      <div class="mod"><div class="n">EXTRACT_NUCLEI_PROPERTIES <span class="tag tag-on">compartments</span></div>
        <div class="x">Re-keys nucleus contours to cell labels so the GeoJSON can carry both
          geometries per detection.</div></div>
      <div class="ckpt">→ csv/segmented.csv</div>
    </div>
  </div>

  <div class="lane">
    <h3><span>POSTPROCESSING</span><span>step 4</span></h3>
    <div class="body">
      <div class="mod"><div class="n">SPLIT_CHANNELS → QUANTIFY</div>
        <div class="x">Per-cell marker intensity against the mask, one task per channel, merged
          per patient by <code>MERGE_QUANT_CSVS</code>. Median always; Mean/Sum added by
          <code>expanded_quantification</code>.</div>
        <div class="pp"><span>compartments <b>true</b></span><span>expanded <b>false</b></span></div></div>
      <div class="mod"><div class="n">EXPORT_GEOJSON</div>
        <div class="x">QuPath / FlowPath <code>cells.geojson</code> with raw per-marker
          measurements, plus a lighter whole-cell-only variant.</div>
        <div class="pp"><span>pixel_size <b>auto</b></span></div></div>
      <div class="mod"><div class="n">MERGE_AND_PYRAMID</div>
        <div class="x">Pyramidal OME-TIFF preserving channel metadata; masks optionally embedded
          as a second series.</div>
        <div class="pp"><span>tilex <b>512</b></span><span>levels <b>8</b></span><span>zstd</span></div></div>
      <div class="mod"><div class="n">EXPORT_SPATIALDATA <span class="tag tag-on">default on</span></div>
        <div class="x">Serialize the run into a scverse-native <code>.zarr</code>. Recomputes
          nothing.</div></div>
      <div class="mod"><div class="n">GENERATE_POSTPROCESSING_QC <span class="tag">opt</span></div>
        <div class="x">Mask + quantification overlays for visual inspection.</div></div>
      <div class="ckpt">→ csv/postprocessed.csv</div>
    </div>
  </div>

</div>

<div class="mod"><div class="n">GENERATE_QC_REPORT</div>
  <div class="x">Aggregate every step's QC and every <code>versions.yml</code> into one HTML report.
    Skipped by <code>--skip_final_qc_report</code>.</div></div>
<div class="mod"><div class="n">AGGREGATE_SIZE_LOGS <span class="tag">enable_trace</span></div>
  <div class="x">Per-process input-size logs (default on) — the raw resource data behind the
    resource report.</div></div>

!!! warning "The three segmentation backends are not drop-in equivalents"
    They read different inputs, build the whole-cell mask differently, and have
    different prerequisites — only InstanSeg runs on an unconfigured clone, and
    `seg_expand_distance` is ignored when it is selected. Side-by-side
    comparison and the 17 backend-specific parameters:
    [Parameters → Choosing a backend](parameters.md#choosing-a-backend).

!!! note "The adapter seam"
    `VALIS_ADAPTER` and `TILED_ADAPTER` present the *same* interface — same input
    tuple, same emit names (`transform`, `transform_by_slide`, `intrinsic_tre`,
    `stage_checkpoint`, `registered`, `size_logs`, `versions`).
    `params.registration_method` is read **once on the linear registration path**, in
    `subworkflows/local/registration.nf`, and passed down as an argument.
    Optional emits use a null object: an adapter for a
    method that produces no TRE emits `Channel.empty()`, and consumers tolerate
    zero artifacts.

---

## d — What a completed run leaves on disk

Per patient, under `--outdir`. The GeoJSON and the OME-TIFF are the primary
artifacts — they are what QuPath and FlowPath open. The SpatialData store is
additive: the same run, readable by `scanpy`, `squidpy`, `napari-spatialdata`
and Vitessce without conversion.

<div class="outs">
  <div class="out">
    <div class="p">geojson/export/cells.geojson</div><div class="role">primary · QuPath / FlowPath</div>
    <ul><li>One detection per cell</li><li>Cell + nucleus geometry</li><li><code>"CD3: Nucleus: Median"</code> keys</li></ul>
  </div>
  <div class="out">
    <div class="p">pyramid/pyramid.ome.tiff</div><div class="role">primary · image</div>
    <ul><li>8 pyramid levels, zstd</li><li>Channel names preserved</li><li>Optional second mask series</li></ul>
  </div>
  <div class="out">
    <div class="p">spatialdata/&lt;pid&gt;.zarr</div><div class="role">additive · scverse</div>
    <ul><li><code>labels</code> cell + nuclei masks</li><li><code>shapes</code> cell + nucleus polygons</li><li><code>table</code> AnnData</li><li><code>obsm</code> centroids + residuals</li><li><code>uns</code> QC + provenance</li></ul>
  </div>
  <div class="out">
    <div class="p">qc/ · segmentation/ · csv/</div><div class="role">evidence</div>
    <ul><li>Aggregated HTML QC report</li><li>Per-stage Dice / displacement</li><li>Resource report</li><li>Resumable checkpoint CSVs</li></ul>
  </div>
</div>

Complete tree, filenames and the measurement-key contract:
[Inputs & outputs](outputs.md).

---

## f — Resource model

Every process's `cpus` / `memory` / `time` has **exactly one owner**: either a
resource `label` or a `withName:` block in `conf/modules.config`, never both.
Memory and time scale with `task.attempt`, so a retry climbs the ramp, and every
request is clamped to `params.max_cpus` / `max_memory` / `max_time`.

Per-process figures, the retry policy, and the QC `retry-then-fail` rule (the
seven QC processes fail the run; only the two opt-in segmentation evaluators are
dropped): [Resources](resources.md).

---

## See also

- :material-tune: **Every parameter with its default** — [Parameters](parameters.md)
- :material-file-tree: **Every input and output path** — [Inputs & outputs](outputs.md)
- :material-server: **Every resource request** — [Resources](resources.md)
- :material-console: **Command recipes** — [Usage](usage.md)
- :material-image-outline: **Supplementary Figure S2 · registration** —
  [figure](figures/registration-schematic.html){ target=_blank }
- :material-image-outline: **Supplementary Figure S3 · quality control** —
  [figure](figures/qc-schematic.html){ target=_blank }
