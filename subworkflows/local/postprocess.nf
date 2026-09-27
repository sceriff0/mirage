
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
// Shared with subworkflows/local/add_cycle.nf — see those files for why the shaping
// lives there rather than being copied into each caller. groupTiffsByPatient is a
// plain function, not a process/workflow, but Nextflow's `include` pulls in either.
include { QUANTIFY_MARKERS; groupTiffsByPatient } from './quantify_markers'
include { ASSEMBLE_EXPORT          } from './assemble_export'
// The checkpoint writer is shared with add_cycle.nf, the same way
// register_patient.nf's CHECKPOINT_WRITER call is shared with it. This file used
// to own the only copy, so an add_cycle run wrote no csv/postprocessed.csv and
// could never be the --prior_outdir of a second cycle.
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

        SEGMENT / EXTRACT_CELL_PROPERTIES / EXTRACT_NUCLEI_PROPERTIES no longer run
        here — they moved to segmentation.nf, their own resumable step
        (Layout.SEGMENTED / Checkpoint.columns('segmented')). This file takes their
        outputs as plain [meta/patient_id, file] inputs instead.

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
    compartment_mode    // ParamUtils.compartmentMode(params) — resolved once by
                        // workflows/mirage.nf and threaded down, the same seam
                        // --registration_method has. Passed straight through to
                        // ASSEMBLE_EXPORT below; `.compartments` is also read here
                        // directly for the nucleus-contour ternary.

    main:

    // ========================================================================
    // CHANNEL SPLITTING - Split all multichannel images (runs in PARALLEL with EXTRACT_CELL_PROPERTIES)
    // ========================================================================
    // A slide whose keep-set resolved EMPTY contributes NO new markers: every channel it
    // declares was already claimed by an earlier slide of the same patient
    // (CsvUtils.resolveKeptChannelsPerSlide walks reference first, then samplesheet
    // order). It is filtered out here rather than split.
    //
    // WHY FILTER RATHER THAN `optional: true`. SPLIT_CHANNELS declares `path("*.tiff")`
    // as a MANDATORY output, and that is a real guard -- a genuinely failed split still
    // trips it. Relaxing it to accommodate a slide that has no work to do would blind the
    // guard for every slide. Dropping the slide is safe for every consumer: the same
    // resolver sizes meta.channels_count (CsvUtils.countChannelsPerPatient sums the
    // per-slide lists), so this slide already contributes ZERO to both channels_count-sized
    // groupKeys below -- the groups still close on exactly the files that arrive.
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
    // Scores the SHIPPED masks against the reference image. Gated by ext.when on
    // params.skip_seg_quality_eval (default true), so on a normal run neither
    // process is instantiated at all.
    //
    // Scored against the REFERENCE slide, not every registered slide: CSE's
    // metrics compare a mask to the multichannel image the mask was derived
    // from, and SEGMENT only ever runs on the reference (segmentation.nf filters
    // is_reference). Joining on patient_id keeps the three inputs together
    // without assuming channel emission order.
    ch_seg_eval_in = ch_cell_mask
        .map { meta, cmask -> [meta.patient_id, meta, cmask] }
        .join(ch_nuclei_mask.map { meta, nmask -> [meta.patient_id, nmask] }, by: 0)
        .join(ch_registered.filter { meta, _img -> meta.is_reference }
                           .map { meta, img -> [meta.patient_id, img] }, by: 0)
        .map { _pid, meta, cmask, nmask, img -> [meta, cmask, nmask, img] }

    SEG_QUALITY_EVAL(ch_seg_eval_in)

    // .ifEmpty([]) so the collect() still emits when every patient's score was
    // dropped (retried out, see conf/modules.config) — otherwise MERGE_SEG_EVAL
    // never runs and the absence is invisible rather than an empty summary.
    MERGE_SEG_EVAL(
        SEG_QUALITY_EVAL.out.metrics.map { _meta, json -> json }.collect().ifEmpty([])
    )
    ch_seg_eval_metrics = MERGE_SEG_EVAL.out.csv

    // ========================================================================
    // QUANTIFICATION - Join channels with their patient's mask
    // ========================================================================
    // Carry BOTH masks (cell + nuclei) keyed by patient_id. The nuclear mask is
    // always available from SEGMENT; QUANTIFY only uses it when
    // params.quantify_compartments is set (per-compartment signal). The same pair
    // feeds ASSEMBLE_EXPORT's embed_masks gate further down.
    ch_mask = ch_cell_mask
        .map { meta, mask -> [meta.patient_id, mask] }
        .join(ch_nuclei_mask.map { meta, mask -> [meta.patient_id, mask] }, by: 0)

    // Per-marker fan-out + QUANTIFY + per-patient grouping (with the groupKey
    // streaming hint) all live in QUANTIFY_MARKERS, shared with add_cycle.nf.
    // The --debug_channels views for this whole chain moved there with it, so this
    // file no longer carries a debug-view helper of its own.
    QUANTIFY_MARKERS(SPLIT_CHANNELS.out.channels, ch_mask)
    ch_grouped_csvs = QUANTIFY_MARKERS.out.grouped_csv

    // Join grouped intensity CSVs with morphology.csv (segmentation.nf's
    // EXTRACT_CELL_PROPERTIES.out.morphology, taken in via ch_morphology)
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

    // Per slide: its markers (the split files' names -- the exact marker prefixes of
    // the quant columns), and the file names of its retention / residual CSVs, keyed by
    // meta.id. join(remainder: true): the reference has neither CSV; with reg_qc < 2 or
    // --start postprocessing there are no residuals at all.
    ch_round_rows = SPLIT_CHANNELS.out.channels
        .map { meta, tiffs ->
            def list = tiffs instanceof List ? tiffs : [tiffs]
            [meta.id, meta.patient_id, meta.is_reference as boolean, list.collect { it.baseName }.toSorted()]
        }
        .join(NUCLEAR_RETENTION.out.csv.map { meta, csv -> [meta.id, csv] }, by: 0, remainder: true)
        .join(ch_reg_residuals.map { meta, csv -> [meta.id, csv] }, by: 0, remainder: true)
        .filter { row -> row[1] != null }     // a residual whose slide never split (keep-set empty)
        .map { id, pid, is_ref, markers, ret_csv, res_csv ->
            [pid, [round_id: id, is_reference: is_ref, markers: markers,
                   retention_csv: ret_csv ? ret_csv.name : null,
                   residual_csv : res_csv ? res_csv.name : null],
             ret_csv, res_csv]
        }
        .groupTuple(by: 0)
        .map { pid, entries, ret_csvs, res_csvs ->
            // CANONICAL ORDER: groupTuple emits in arrival order and val/path lists hash
            // positionally, so an identical rerun would miss -resume. Sort the entries;
            // the file lists are only staged by name (the manifest pairs by NAME, never
            // by position), so sorting them independently is safe here.
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

    // ch_contours arrives via take: (segmentation.nf's EXTRACT_CELL_PROPERTIES.out.contours,
    // already re-keyed to patient_id) — nothing to derive here any more.
    ch_nuc_contours_for_export = compartment_mode.compartments ? ch_nucleus_contours : ch_contours

    // ========================================================================
    // MERGE - Combine split channel TIFFs with segmentation mask (per patient)
    // ========================================================================
    // Group split channel TIFFs by patient for merging
    // SPLIT_CHANNELS already handles DAPI filtering correctly
    // Deduplicate by patient_id + marker to avoid duplicate channel names
    // Use groupKey for streaming - emits as soon as channels_count items collected
    //
    // `remainder: true` for the same reason as QUANTIFY_MARKERS' grouping (same
    // channels_count): an under-count must not be allowed to silently drop the patient
    // from the pyramid outright. But — see the fuller account in QUANTIFY_MARKERS'
    // GROUP comment, which this grouping mirrors — an under-count here does NOT degrade
    // to "late but complete" the way it can for the CSV-merge paths. This grouping feeds
    // MERGE_AND_PYRAMID with a one-file surplus group, which trips that process's memory
    // closure (conf/modules.config:330-337 — pre-existing, NOT fixed here) and the run
    // ABORTS with "No such file or directory: channels". `remainder: true` is kept anyway
    // because keeping only one of the two channels_count-sized groupings (this one, or
    // QUANTIFY_MARKERS') would be worse than keeping neither — the patient would reach
    // geojson/ and quantification/ but not the pyramid, and so be missing from
    // csv/postprocessed.csv, half-published and invisible to any later --start
    // postprocessing or add_cycle run. These two groupings must keep identical
    // channels_count semantics even though their downstream failure modes differ --
    // WITHIN THIS FILE: both read the same meta.channels_count for the same
    // patient. That equality is NOT a cross-file invariant: add_cycle.nf's own
    // pyramid grouping deliberately uses a DIFFERENT total (new_count + prior_count,
    // since it merges two pyramids' worth of channels) than its own QUANTIFY_MARKERS
    // call (new-cycle channels only, because the prior quantification columns are
    // merged onto the CSV separately, not re-quantified) -- the two counts SHOULD
    // differ there. Do not "reconcile" them to match this file's equality.
    // The channels_count-sized groupKey + remainder:true grouping itself is
    // groupTiffsByPatient (subworkflows/local/quantify_markers.nf), shared with
    // add_cycle.nf's own version of this same grouping — see that function's doc
    // comment for why an under-count here is worse than for the CSV-merge paths.
    ch_split_tagged = SPLIT_CHANNELS.out.channels
        .flatMap { meta, tiffs ->
            // Normalize to List and create entries keyed by [patient_id, marker]
            // Carry channels_count for groupKey
            def tiff_list = tiffs instanceof List ? tiffs : [tiffs]
            tiff_list.collect { tiff ->
                [meta.patient_id, meta.channels_count, tiff.baseName, tiff]
            }
        }
        .unique { patient_id, _channels_count, marker, _tiff -> [patient_id, marker] }  // Keep first occurrence of each patient+marker
        .map { patient_id, channels_count, _marker, tiff -> [patient_id, channels_count, tiff] }
    ch_split_grouped = groupTiffsByPatient(ch_split_tagged)

    // EXPORT_GEOJSON tuple assembly + the embed_masks pyramid gate live in
    // ASSEMBLE_EXPORT, shared with add_cycle.nf.
    ASSEMBLE_EXPORT(
        ch_merged_with_qc,
        ch_contours,
        ch_nuc_contours_for_export,
        ch_split_grouped,
        ch_mask,
        compartment_mode
    )

    // ========================================================================
    // POSTPROCESSING QC (optional, runs in PARALLEL with MERGE_AND_PYRAMID)
    // ========================================================================
    ch_postprocess_qc = Channel.empty()
    if (!params.skip_postprocessing_qc) {
        // Join cell mask with merged CSV for QC visualization
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

        // QC is collected run-wide rather than per patient: these channels are already
        // flat file streams by the time they arrive, and `.ifEmpty([])` keeps the export
        // running when reg_qc=0 or the run started at postprocessing.
        EXPORT_SPATIALDATA(
            ch_sd_in,
            // collect(sort: true), not collect(): these lists become EXPORT_SPATIALDATA's
            // `path` inputs, which Nextflow hashes POSITIONALLY, and a bare collect()
            // emits in arrival order -- so an identical rerun re-ran the export.
            ch_reg_qc.map { it instanceof List ? it[-1] : it }.flatten().collect(sort: true).ifEmpty([]),
            ch_reg_residuals.map { it instanceof List ? it[-1] : it }.flatten().collect(sort: true).ifEmpty([])
        )
    }

    // ========================================================================
    // CHECKPOINT - Collect all outputs by patient
    // ========================================================================
    // The join, the publishedPath rules and the collectFile all live in
    // POSTPROCESSED_CHECKPOINT, which add_cycle.nf calls with its own five
    // streams. This file used to own the only copy, which is why an add_cycle
    // run wrote no csv/postprocessed.csv and could not be chained.
    POSTPROCESSED_CHECKPOINT(
        ASSEMBLE_EXPORT.out.csv,
        ASSEMBLE_EXPORT.out.geojson,
        ch_merged_with_qc,
        ch_cell_mask,
        ASSEMBLE_EXPORT.out.pyramid
    )
    ch_checkpoint_csv = POSTPROCESSED_CHECKPOINT.out.csv

    // Collect size logs from all postprocessing processes. SEGMENT /
    // EXTRACT_CELL_PROPERTIES / EXTRACT_NUCLEI_PROPERTIES moved to
    // subworkflows/local/segmentation.nf and report their own size_logs/versions
    // there now -- workflows/mirage.nf mixes SEGMENTATION.out.{size_logs,versions}
    // into the run-wide QC stream directly, the same way it already does for every
    // other step's aggregate output (this file's checkpoint_csv/postprocess_qc
    // pattern), so nothing from that step is double-counted or dropped here.
    ch_size_logs = Channel.empty()
        .mix(SEG_QUALITY_EVAL.out.size_log)
        .mix(SPLIT_CHANNELS.out.size_log)
        .mix(QUANTIFY_MARKERS.out.size_logs)
        .mix(MERGE_QUANT_CSVS.out.size_log)
        .mix(NUCLEAR_RETENTION.out.size_log)
        .mix(CELL_QC.out.size_log)
        .mix(ASSEMBLE_EXPORT.out.size_logs)

    // Add postprocessing QC size logs if enabled
    if (!params.skip_postprocessing_qc) {
        ch_size_logs = ch_size_logs
            .mix(GENERATE_POSTPROCESSING_QC.out.size_log)
    }

    // Collect versions from all postprocessing processes.
    // QUANTIFY_MARKERS / ASSEMBLE_EXPORT already applied `.first()` internally
    // (see the comments on their `versions` emits) — do not re-apply it here.
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
