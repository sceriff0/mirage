/*
========================================================================================
    SUBWORKFLOW: SEGMENTATION
========================================================================================
    SEGMENT the reference image, extract cell (and optionally nucleus)
    contours/morphology, and write the `segmented` checkpoint
    (Layout.SEGMENTED / Checkpoint.columns('segmented')).

    Input:
        ch_registered: [meta, file] — ALL slides (reference + moving) that reached
                       the registered stream. Only the reference subset is segmented;
                       every slide gets a row in the checkpoint (see the writer below).

    Output:
        cell_mask        [meta, file]         — SEGMENT.out.cell_mask, unchanged
        nuclei_mask      [meta, file]         — SEGMENT.out.nuclei_mask, unchanged
        contours         [patient_id, file]   — EXTRACT_CELL_PROPERTIES.out.contours, re-keyed
        nucleus_contours [patient_id, file]   — EXTRACT_NUCLEI_PROPERTIES.out.contours, re-keyed;
                                                 Channel.empty() when --quantify_compartments is false
        checkpoint_csv                        — csv/segmented.csv
        size_logs, versions                   — SEGMENT + EXTRACT_CELL_PROPERTIES (+
                                                 EXTRACT_NUCLEI_PROPERTIES when compartments run)

    Also defines READ_SEGMENTED_CHECKPOINT, the `--start postprocessing` reader:
    segmented.csv carries mask columns that INPUT_CHECK's `[meta, one_file]` shape
    cannot express.
========================================================================================
*/

include { SEGMENT                   } from '../../modules/local/segment'
include { EXTRACT_CELL_PROPERTIES   } from '../../modules/local/extract_cell_properties'
include { EXTRACT_NUCLEI_PROPERTIES } from '../../modules/local/extract_nuclei_properties'
include { CHECKPOINT_WRITER         } from './checkpoint_writer'

workflow SEGMENTATION {
    take:
    ch_registered       // [meta, file] — all slides (reference + moving)
    compartment_mode    // ParamUtils.compartmentMode(params); only `.compartments` is read

    main:

    // ========================================================================
    // SEGMENTATION - Process reference images only
    // ========================================================================
    ch_references = ch_registered
        .filter { meta, file -> meta.is_reference }

    // No ifEmpty { error } here: a missing reference is rejected at launch, and ifEmpty
    // also fires when upstream failed, hiding the real error
    // (tests/test_no_shadowing_empty_guards.py).

    // Resolve the backend params here: a bare `params` inside SEGMENT's script would hash
    // the whole map into its cache key. SegBackends.CTX_PARAM_KEYS picks the keys.
    def seg_params = SegBackends.ctxParams(params)

    SEGMENT(ch_references.map { meta, f -> tuple(meta, f, seg_params) })

    def ch_cell_mask   = SEGMENT.out.cell_mask
    def ch_nuclei_mask = SEGMENT.out.nuclei_mask

    // ========================================================================
    // CELL PROPERTIES - morphology + contours from the mask, computed once (not per marker)
    // ========================================================================
    EXTRACT_CELL_PROPERTIES(ch_cell_mask)

    def ch_contours = EXTRACT_CELL_PROPERTIES.out.contours
        .map { meta, json_file -> [meta.patient_id, json_file] }

    // Nucleus contours (re-keyed to cell labels) for the GeoJSON export, only when
    // per-compartment quantification is enabled.
    def ch_nucleus_contours = Channel.empty()
    if (compartment_mode.compartments) {
        ch_nuclei_props_in = ch_nuclei_mask
            .map { meta, mask -> [meta.patient_id, meta, mask] }
            .join(ch_cell_mask.map { meta, mask -> [meta.patient_id, mask] }, by: 0)
            .map { _patient_id, meta, nuclei_mask, cell_mask -> [meta, nuclei_mask, cell_mask] }
        EXTRACT_NUCLEI_PROPERTIES(ch_nuclei_props_in)
        ch_nucleus_contours = EXTRACT_NUCLEI_PROPERTIES.out.contours
            .map { meta, json_file -> [meta.patient_id, json_file] }
    }

    // ========================================================================
    // CHECKPOINT - csv/segmented.csv
    // ========================================================================
    // One row per slide (the next step needs each registered image); per-patient mask
    // columns repeat across a patient's rows.
    // publishedOrAsIs: at --start segmentation the registered paths are already published
    // by a prior run, and INPUT_CHECK never sets is_passthrough.
    ch_registered_for_ckpt = ch_registered
        .map { meta, file ->
            def published_path = meta.is_passthrough
                ? Layout.passthroughPath(params.outdir, meta.patient_id, file, params.skip_preprocessing as boolean)
                : Layout.publishedOrAsIs(params.outdir, meta.patient_id, Layout.REGISTERED, file)
            [meta.patient_id, meta, published_path]
        }

    // cell_mask, nuclei_mask and contours are always recorded: SEGMENT always emits both
    // masks, and postprocess.nf inner-joins them, so a missing nuclei_mask would drop every
    // patient. SEGMENT emits one tuple per input, so the combines below drop no row.
    ch_cell_mask_path = ch_cell_mask.map { meta, m ->
        [meta.patient_id, Layout.publishedPath(params.outdir, meta.patient_id, 'segmentation', m)]
    }
    ch_nuclei_mask_path = ch_nuclei_mask.map { meta, m ->
        [meta.patient_id, Layout.publishedPath(params.outdir, meta.patient_id, 'segmentation', m)]
    }
    ch_contours_path = ch_contours.map { pid, j ->
        [pid, Layout.publishedPath(params.outdir, pid, 'cell_properties', j)]
    }

    // nucleus_contours exists only under quantify_compartments. Join with remainder against
    // a per-patient placeholder so a missing value becomes '' instead of dropping the row.
    ch_nucleus_contours_value = compartment_mode.compartments
        ? ch_nucleus_contours.map { pid, j -> [pid, Layout.publishedPath(params.outdir, pid, 'cell_properties', j)] }
        : Channel.empty()

    ch_nucleus_contours_total = ch_cell_mask_path
        .map { pid, _path -> tuple(pid, '') }
        .join(ch_nucleus_contours_value, by: 0, remainder: true)
        .map { pid, placeholder, real -> tuple(pid, real ?: placeholder) }

    ch_checkpoint_rows = ch_registered_for_ckpt
        .combine(ch_cell_mask_path, by: 0)
        .combine(ch_nuclei_mask_path, by: 0)
        .combine(ch_contours_path, by: 0)
        .combine(ch_nucleus_contours_total, by: 0)
        .map { pid, meta, reg_path, cell_mask, nuclei_mask, contours, nucleus_contours ->
            [
                patient_id      : pid,
                // From meta, never re-derived from the image basename.
                id              : meta.id,
                registered_image: reg_path,
                is_reference    : meta.is_reference,
                channels        : meta.channels.join('|'),
                cell_mask       : cell_mask,
                nuclei_mask     : nuclei_mask,
                contours        : contours,
                nucleus_contours: nucleus_contours,
                pixel_size      : meta.pixel_size,
            ]
        }

    // Not written at a cleaning level, where these paths are not published.
    CHECKPOINT_WRITER(Layout.SEGMENTED, ch_checkpoint_rows)
    ch_checkpoint_csv = CHECKPOINT_WRITER.out.csv

    ch_size_logs = Channel.empty()
        .mix(SEGMENT.out.size_log)
        .mix(EXTRACT_CELL_PROPERTIES.out.size_log)
    if (compartment_mode.compartments) {
        ch_size_logs = ch_size_logs.mix(EXTRACT_NUCLEI_PROPERTIES.out.size_log)
    }

    ch_versions = Channel.empty()
        .mix(SEGMENT.out.versions.first())
        .mix(EXTRACT_CELL_PROPERTIES.out.versions.first())
    if (compartment_mode.compartments) {
        ch_versions = ch_versions.mix(EXTRACT_NUCLEI_PROPERTIES.out.versions.first())
    }

    emit:
    cell_mask        = ch_cell_mask
    nuclei_mask      = ch_nuclei_mask
    contours         = ch_contours
    nucleus_contours = ch_nucleus_contours
    // [meta, file]: not checkpointed; POSTPROCESSING joins it with the per-marker CSVs.
    morphology       = EXTRACT_CELL_PROPERTIES.out.morphology
    checkpoint_csv   = ch_checkpoint_csv
    size_logs        = ch_size_logs
    versions         = ch_versions
}

/*
========================================================================================
    WORKFLOW: READ_SEGMENTED_CHECKPOINT
========================================================================================
    The `--start postprocessing` reader for csv/segmented.csv. Columns come from
    lib/Checkpoint.groovy; meta is built with Meta.fromCheckpointRow, from a ctx derived
    from this checkpoint's own rows (CsvUtils.metaContextFromCheckpoint).

    Contours and morphology are re-derived from the masks, not read back:
    morphology.csv is not a checkpoint column, and add_cycle re-derives them the same way.

    Input:
        csv_path: path to a `segmented` checkpoint CSV (segmented.csv or add_cycle's
                  future analogue re-using this same shape)

    Output:
        samples          [meta, file]        — same shape as INPUT_CHECK.out.samples
        cell_mask        [meta, file]        — reference rows only
        nuclei_mask      [meta, file]        — reference rows only (SEGMENT always
                                                produces it; see Checkpoint's
                                                'segmented' columns)
        contours         [patient_id, file]  — re-derived from cell_mask
        nucleus_contours [patient_id, file]  — re-derived from nuclei_mask;
                                                Channel.empty() when
                                                --quantify_compartments is false
        morphology       [meta, file]        — re-derived from cell_mask
        size_logs, versions                  — of the re-run EXTRACT_CELL_PROPERTIES
                                                (+ EXTRACT_NUCLEI_PROPERTIES when
                                                compartments run); workflows/mirage.nf
                                                mixes these into FINAL_QC at the
                                                postprocessing-entry point, where
                                                SEGMENTATION never ran this session
========================================================================================
*/

workflow READ_SEGMENTED_CHECKPOINT {
    take:
    csv_path
    compartment_mode    // ParamUtils.compartmentMode(params); only `.compartments` is read

    main:
    // Fail if the checkpoint schema drifts from the columns this reader indexes.
    Checkpoint.requireColumns(Layout.SEGMENTED,
        ['patient_id', 'id', 'registered_image', 'is_reference', 'channels',
         'cell_mask', 'nuclei_mask'])

    // Meta ctx from this checkpoint's own rows: a derived file's basename cannot
    // reproduce the samplesheet identity.
    def meta_ctx = CsvUtils.metaContextFromCheckpoint(csv_path, Layout.SEGMENTED)

    ch_rows = Channel
        .fromPath(csv_path, checkIfExists: true)
        .splitCsv(header: true)
        .map { row ->
            // Build meta through Meta, never inline, so the key set matches a full run.
            [Meta.fromCheckpointRow(row, Layout.SEGMENTED, meta_ctx), row]
        }

    ch_samples = ch_rows.map { meta, row -> [meta, file(row.registered_image)] }

    // Mask columns repeat across a patient's rows; read them off the reference row only.
    ch_ref_rows = ch_rows.filter { meta, _row -> meta.is_reference }

    ch_cell_mask   = ch_ref_rows.map { meta, row -> [meta, file(row.cell_mask)] }
    // Never empty: SEGMENT always produces nuclei_mask.
    ch_nuclei_mask = ch_ref_rows.map { meta, row -> [meta, file(row.nuclei_mask)] }

    // Re-derive contours + morphology from the reused cell mask.
    EXTRACT_CELL_PROPERTIES(ch_cell_mask)
    ch_contours   = EXTRACT_CELL_PROPERTIES.out.contours.map { meta, j -> [meta.patient_id, j] }
    ch_morphology = EXTRACT_CELL_PROPERTIES.out.morphology

    ch_size_logs = Channel.empty().mix(EXTRACT_CELL_PROPERTIES.out.size_log)
    ch_versions  = Channel.empty().mix(EXTRACT_CELL_PROPERTIES.out.versions.first())

    // Nucleus contours: only under --quantify_compartments.
    ch_nucleus_contours = Channel.empty()
    if (compartment_mode.compartments) {
        ch_nuclei_props_in = ch_nuclei_mask
            .map { meta, mask -> [meta.patient_id, meta, mask] }
            .join(ch_cell_mask.map { meta, mask -> [meta.patient_id, mask] }, by: 0)
            .map { _patient_id, meta, nuclei_mask, cell_mask -> [meta, nuclei_mask, cell_mask] }
        EXTRACT_NUCLEI_PROPERTIES(ch_nuclei_props_in)
        ch_nucleus_contours = EXTRACT_NUCLEI_PROPERTIES.out.contours
            .map { meta, j -> [meta.patient_id, j] }
        ch_size_logs = ch_size_logs.mix(EXTRACT_NUCLEI_PROPERTIES.out.size_log)
        ch_versions  = ch_versions.mix(EXTRACT_NUCLEI_PROPERTIES.out.versions.first())
    }

    emit:
    samples          = ch_samples
    cell_mask        = ch_cell_mask
    nuclei_mask      = ch_nuclei_mask
    contours         = ch_contours
    nucleus_contours = ch_nucleus_contours
    morphology       = ch_morphology
    // Versions/size logs of the re-run, for FINAL_QC (SEGMENTATION did not run this session).
    size_logs        = ch_size_logs
    versions         = ch_versions
}
