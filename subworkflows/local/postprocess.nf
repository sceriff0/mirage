
/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { SPLIT_CHANNELS           } from '../../modules/local/split_channels'
include { MERGE_QUANT_CSVS         } from '../../modules/local/merge_quant_csvs'
include { GENERATE_POSTPROCESSING_QC    } from '../../modules/local/generate_postprocessing_qc'
include { EXPORT_SPATIALDATA } from '../../modules/local/export_spatialdata'
include { SEG_QUALITY_EVAL         } from '../../modules/local/seg_quality_eval'
include { MERGE_SEG_EVAL           } from '../../modules/local/merge_seg_eval'
include { NUCLEAR_RETENTION        } from '../../modules/local/nuclear_retention'
include { CELL_QC                  } from '../../modules/local/cell_qc'
// Shared with add_cycle.nf. groupTiffsByPatient is a plain function.
include { QUANTIFY_MARKERS; groupTiffsByPatient } from './quantify_markers'
include { ASSEMBLE_EXPORT          } from './assemble_export'
// Shared with add_cycle.nf, so both write csv/postprocessed.csv.
include { POSTPROCESSED_CHECKPOINT } from './postprocessed_checkpoint'

/*
========================================================================================
    SUBWORKFLOW:POSTPROCESSING
========================================================================================
    Description:
        Splits multichannel images to single channels, quantifies marker intensities
        per cell against masks produced upstream by subworkflows/local/segmentation.nf,
        merges results, and exports QuPath-compatible GeoJSON with raw measurements
        for FlowPath gating.

    Input:
        ch_registered:       [meta, file] — all registered slides (reference + moving)
        ch_cell_mask:        [meta, file] — from segmentation.nf
        ch_nuclei_mask:      [meta, file] — from segmentation.nf
        ch_contours:         [patient_id, file] — from segmentation.nf
        ch_nucleus_contours: [patient_id, file] — from segmentation.nf;
                              Channel.empty() when --quantify_compartments is false
        ch_morphology:       [meta, file] — from segmentation.nf

    Output:
        checkpoint_csv: file — the collected 'postprocessed' checkpoint (see
                        Layout.POSTPROCESSED / Checkpoint.columns), one row per patient
        postprocess_qc: GENERATE_POSTPROCESSING_QC's per-patient QC artifacts
        size_logs:      input size-log CSVs from this step's processes
        versions:       versions.yml from this step's processes
========================================================================================
*/

workflow POSTPROCESSING {
    take:
    ch_registered       // Channel of [meta, file] tuples
    ch_cell_mask        // [meta, file] — subworkflows/local/segmentation.nf's SEGMENT.out.cell_mask
    ch_nuclei_mask      // [meta, file] — segmentation.nf's SEGMENT.out.nuclei_mask
    ch_contours         // [patient_id, file] — segmentation.nf's EXTRACT_CELL_PROPERTIES.out.contours
    ch_nucleus_contours // [patient_id, file] — segmentation.nf's EXTRACT_NUCLEI_PROPERTIES.out.contours;
                        // Channel.empty() when !params.quantify_compartments
    ch_morphology       // [meta, file] — segmentation.nf's EXTRACT_CELL_PROPERTIES.out.morphology
    ch_reg_qc           // Registration QC JSONs (may be empty)
    ch_reg_residuals    // Per-cell registration residual CSVs (may be empty)
    compartment_mode    // ParamUtils.compartmentMode(params), resolved in workflows/mirage.nf

    main:

    // ========================================================================
    // CHANNEL SPLITTING - split all multichannel images
    // ========================================================================
    // A slide with an empty keep-set adds no new markers (earlier slides claimed them all)
    // and is filtered out, so SPLIT_CHANNELS' mandatory *.tiff output stays a real guard.
    // channels_count already counts such a slide as zero.
    SPLIT_CHANNELS(
        ch_registered
            .filter { meta, _f ->
                if (meta.keep_channels != null && meta.keep_channels.isEmpty()) {
                    log.warn "POSTPROCESSING(${meta.patient_id}): slide ${meta.id} contributes no new " +
                             "markers -- every channel it declares was already claimed by an earlier " +
                             "slide of this patient. Not splitting it; channels_count already counts " +
                             "it as zero."
                    return false
                }
                return true
            }
            .map { meta, file -> [meta, file, meta.is_reference] }
    )

    // ========================================================================
    // SEGMENTATION QUALITY EVAL (CSE) - reference-free, informational, OPT-IN
    // ========================================================================
    // Scores the shipped masks against the reference image they were derived from.
    // Gated by ext.when on params.skip_seg_quality_eval (default true).
    ch_seg_eval_in = ch_cell_mask
        .map { meta, cmask -> [meta.patient_id, meta, cmask] }
        .join(ch_nuclei_mask.map { meta, nmask -> [meta.patient_id, nmask] }, by: 0)
        .join(ch_registered.filter { meta, _img -> meta.is_reference }
                           .map { meta, img -> [meta.patient_id, img] }, by: 0)
        .map { _pid, meta, cmask, nmask, img -> [meta, cmask, nmask, img] }

    SEG_QUALITY_EVAL(ch_seg_eval_in)

    // .ifEmpty([]) so MERGE_SEG_EVAL still writes an (empty) summary if every score was dropped.
    MERGE_SEG_EVAL(
        SEG_QUALITY_EVAL.out.metrics.map { _meta, json -> json }.collect().ifEmpty([])
    )
    ch_seg_eval_metrics = MERGE_SEG_EVAL.out.csv

    // ========================================================================
    // QUANTIFICATION - Join channels with their patient's mask
    // ========================================================================
    // Both masks keyed by patient_id (SEGMENT always emits the nuclear one). Also feeds
    // ASSEMBLE_EXPORT's embed_masks gate.
    ch_mask = ch_cell_mask
        .map { meta, mask -> [meta.patient_id, mask] }
        .join(ch_nuclei_mask.map { meta, mask -> [meta.patient_id, mask] }, by: 0)

    // Per-marker fan-out, QUANTIFY and per-patient grouping (shared with add_cycle.nf).
    QUANTIFY_MARKERS(SPLIT_CHANNELS.out.channels, ch_mask)
    ch_grouped_csvs = QUANTIFY_MARKERS.out.grouped_csv

    // Join grouped intensity CSVs with morphology.csv.
    ch_morphology_by_patient = ch_morphology
        .map { meta, csv -> [meta.patient_id, csv] }

    ch_for_quant_merge = ch_grouped_csvs
        .map { meta, csvs -> [meta.patient_id, meta, csvs] }
        .join(ch_morphology_by_patient, by: 0)
        .map { _patient_id, meta, csvs, morphology_csv -> [meta, csvs, morphology_csv] }

    MERGE_QUANT_CSVS(ch_for_quant_merge)

    // ========================================================================
    // CELL QC - per-round nuclear retention + registration residuals, total intensity
    // ========================================================================
    // One NUCLEAR_RETENTION task per MOVING slide that contributes markers (the same
    // empty-keep-set rule SPLIT_CHANNELS' filter applies above). The reference is the frame.
    def contributes = { meta -> !(meta.keep_channels != null && meta.keep_channels.isEmpty()) }

    ch_retention_in = ch_registered
        .filter { meta, _f -> !meta.is_reference && contributes(meta) }
        .map { meta, img -> [meta.patient_id, meta, img] }
        .combine(ch_mask, by: 0)
        .map { _pid, meta, img, cmask, nmask -> [meta, img, cmask, nmask] }
    NUCLEAR_RETENTION(ch_retention_in)

    // Per slide: its markers and its retention / residual CSV names, keyed by
    // [patient_id, id] (ids are unique only within a patient). remainder: true because the
    // reference has neither CSV, and residuals exist only at reg_qc 2.
    ch_round_rows = SPLIT_CHANNELS.out.channels
        .map { meta, tiffs ->
            def list = tiffs instanceof List ? tiffs : [tiffs]
            [[meta.patient_id, meta.id], meta.patient_id, meta.is_reference as boolean, list.collect { it.baseName }.toSorted()]
        }
        .join(NUCLEAR_RETENTION.out.csv.map { meta, csv -> [[meta.patient_id, meta.id], csv] }, by: 0, remainder: true)
        .join(ch_reg_residuals.map { meta, csv -> [[meta.patient_id, meta.id], csv] }, by: 0, remainder: true)
        .filter { row -> row[1] != null }     // a residual whose slide never split (keep-set empty)
        .map { key, pid, is_ref, markers, ret_csv, res_csv ->
            [pid, [round_id: key[1], is_reference: is_ref, markers: markers,
                   retention_csv: ret_csv ? ret_csv.name : null,
                   residual_csv : res_csv ? res_csv.name : null],
             ret_csv, res_csv]
        }
        // A cross-patient barrier: the remainder joins release only on channel close.
        // MERGE_AND_PYRAMID does not wait on it.
        .groupTuple(by: 0)
        .map { pid, entries, ret_csvs, res_csvs ->
            // Sort for -resume. The file lists pair with entries by name, not position,
            // so sorting them separately is safe.
            [pid,
             entries.toSorted { it.round_id },
             ret_csvs.findAll { it != null }.toSorted { it.name },
             res_csvs.findAll { it != null }.toSorted { it.name }]
        }

    ch_cell_qc_in = MERGE_QUANT_CSVS.out.merged_csv
        .map { meta, csv -> [meta.patient_id, meta, csv] }
        .join(ch_round_rows, by: 0, remainder: true)
        .filter { row -> row[1] != null }
        .map { _pid, meta, csv, rounds, rets, ress ->
            [meta, csv, rounds ?: [], rets ?: [], ress ?: [], []]
        }
    CELL_QC(ch_cell_qc_in)
    ch_merged_with_qc = CELL_QC.out.merged_csv

    ch_nuc_contours_for_export = compartment_mode.compartments ? ch_nucleus_contours : ch_contours

    // ========================================================================
    // MERGE - Combine split channel TIFFs with segmentation mask (per patient)
    // ========================================================================
    // Group split TIFFs per patient, deduplicated on [patient_id, marker], with the same
    // channels_count as QUANTIFY_MARKERS' grouping (groupTiffsByPatient). Both groupings must
    // keep that count: losing one would publish a patient without its pyramid.
    // add_cycle's pyramid count differs from its quantification count on purpose.
    ch_split_tagged = SPLIT_CHANNELS.out.channels
        .flatMap { meta, tiffs ->
            // Entries keyed by [patient_id, marker], carrying channels_count for groupKey.
            def tiff_list = tiffs instanceof List ? tiffs : [tiffs]
            tiff_list.collect { tiff ->
                [meta.patient_id, meta.channels_count, tiff.baseName, tiff]
            }
        }
        .unique { patient_id, _channels_count, marker, _tiff -> [patient_id, marker] }  // Keep first occurrence of each patient+marker
        .map { patient_id, channels_count, _marker, tiff -> [patient_id, channels_count, tiff] }
    ch_split_grouped = groupTiffsByPatient(ch_split_tagged)

    ASSEMBLE_EXPORT(
        ch_merged_with_qc,
        ch_contours,
        ch_nuc_contours_for_export,
        ch_split_grouped,
        ch_mask,
        compartment_mode
    )

    // ========================================================================
    // POSTPROCESSING QC (optional)
    // ========================================================================
    ch_postprocess_qc = Channel.empty()
    if (!params.skip_postprocessing_qc) {
        ch_for_postprocess_qc = ch_cell_mask
            .map { meta, mask -> [meta.patient_id, meta, mask] }
            .join(
                ch_merged_with_qc.map { meta, csv -> [meta.patient_id, csv] },
                by: 0
            )
            .map { _patient_id, meta, mask, csv -> [meta, mask, csv] }

        GENERATE_POSTPROCESSING_QC(ch_for_postprocess_qc)
        ch_postprocess_qc = GENERATE_POSTPROCESSING_QC.out.qc.map { meta, pngs -> pngs }
    }

    // ========================================================================
    // SPATIALDATA EXPORT - scverse-native .zarr (additive; OME-TIFF + GeoJSON stay primary)
    // ========================================================================
    if (!params.skip_spatialdata_export) {
        def ch_sd_in = ch_merged_with_qc
            .map { meta, csv -> [meta.patient_id, meta, csv] }
            .join(ch_contours, by: 0)
            .join(ch_nuc_contours_for_export, by: 0)
            .join(ch_cell_mask.map { meta, m -> [meta.patient_id, m] }, by: 0)
            .join(ch_nuclei_mask.map { meta, m -> [meta.patient_id, m] }, by: 0)
            .join(ASSEMBLE_EXPORT.out.pyramid.map { meta, p -> [meta.patient_id, p] }, by: 0)
            .map { _pid, meta, csv, contours, nuc_contours, cmask, nmask, pyramid ->
                [meta, csv, contours, nuc_contours, cmask, nmask, pyramid]
            }

        // QC is collected run-wide; .ifEmpty([]) covers reg_qc=0 and --start postprocessing.
        EXPORT_SPATIALDATA(
            ch_sd_in,
            // collect(sort: true): path lists hash positionally, so arrival order breaks -resume.
            ch_reg_qc.map { it instanceof List ? it[-1] : it }.flatten().collect(sort: true).ifEmpty([]),
            ch_reg_residuals.map { it instanceof List ? it[-1] : it }.flatten().collect(sort: true).ifEmpty([])
        )
    }

    // ========================================================================
    // CHECKPOINT - Collect all outputs by patient
    // ========================================================================
    POSTPROCESSED_CHECKPOINT(
        ASSEMBLE_EXPORT.out.csv,
        ASSEMBLE_EXPORT.out.geojson,
        ch_merged_with_qc,
        ch_cell_mask,
        ASSEMBLE_EXPORT.out.pyramid
    )
    ch_checkpoint_csv = POSTPROCESSED_CHECKPOINT.out.csv

    // Segmentation's size logs and versions are collected by workflows/mirage.nf.
    ch_size_logs = Channel.empty()
        .mix(SEG_QUALITY_EVAL.out.size_log)
        .mix(SPLIT_CHANNELS.out.size_log)
        .mix(QUANTIFY_MARKERS.out.size_logs)
        .mix(MERGE_QUANT_CSVS.out.size_log)
        .mix(NUCLEAR_RETENTION.out.size_log)
        .mix(CELL_QC.out.size_log)
        .mix(ASSEMBLE_EXPORT.out.size_logs)

    if (!params.skip_postprocessing_qc) {
        ch_size_logs = ch_size_logs
            .mix(GENERATE_POSTPROCESSING_QC.out.size_log)
    }

    // QUANTIFY_MARKERS / ASSEMBLE_EXPORT versions are already de-duplicated.
    ch_versions = Channel.empty()
        .mix(SEG_QUALITY_EVAL.out.versions)
        .mix(SPLIT_CHANNELS.out.versions.first())
        .mix(QUANTIFY_MARKERS.out.versions)
        .mix(MERGE_QUANT_CSVS.out.versions.first())
        .mix(NUCLEAR_RETENTION.out.versions.first())
        .mix(CELL_QC.out.versions.first())
        .mix(ASSEMBLE_EXPORT.out.versions)

    if (!params.skip_postprocessing_qc) {
        ch_versions = ch_versions
            .mix(GENERATE_POSTPROCESSING_QC.out.versions.first())
    }

    emit:
    checkpoint_csv    = ch_checkpoint_csv
    seg_eval_metrics  = ch_seg_eval_metrics
    postprocess_qc    = ch_postprocess_qc
    size_logs         = ch_size_logs
    versions          = ch_versions
}
