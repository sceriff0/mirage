nextflow.enable.dsl = 2

/*
================================================================================
IMPORT SUBWORKFLOWS
================================================================================
*/

include { PREPROCESSING       } from './subworkflows/local/preprocess'
include { REGISTRATION        } from './subworkflows/local/registration'
include { POSTPROCESSING      } from './subworkflows/local/postprocess'
include { COPY_RESULTS        } from './modules/local/copy_results'
include { AGGREGATE_SIZE_LOGS } from './modules/local/aggregate_size_logs'

/*
================================================================================
IMPORT and DECLARE HELPERS
================================================================================
*/

import static CsvUtils.*
import static ParamUtils.*

def loadInputChannel(csv_path, image_column, patient_counts = null, channel_counts = null) {
    def ch = Channel
        .fromPath(csv_path, checkIfExists: true)
        .splitCsv(header: true)
        .map { row ->
            // Use CsvUtils to handle the complex metadata parsing
            def meta = CsvUtils.parseMetadata(row, "CSV ${csv_path}")
            return tuple(meta, file(row[image_column]))
        }

    // If patient counts provided, add images_count to meta for streaming groupTuple
    if (patient_counts) {
        def counts_ch = Channel.fromList(patient_counts.collect { k, v -> [k, v] })
        ch = ch
            .map { meta, f -> [meta.patient_id, meta, f] }
            .combine(counts_ch, by: 0)
            .map { patient_id, meta, f, count ->
                def updated_meta = meta.clone()
                updated_meta.images_count = count
                [updated_meta, f]
            }
    }

    // If channel counts provided, add channels_count to meta for streaming groupTuple in postprocessing
    if (channel_counts) {
        def ch_counts = Channel.fromList(channel_counts.collect { k, v -> [k, v] })
        ch = ch
            .map { meta, f -> [meta.patient_id, meta, f] }
            .combine(ch_counts, by: 0)
            .map { patient_id, meta, f, count ->
                def updated_meta = meta.clone()
                updated_meta.channels_count = count
                [updated_meta, f]
            }
    }
    return ch
}

/*
================================================================================
WORKFLOW
================================================================================
*/

workflow {

    /* -------------------- PARAMETER VALIDATION -------------------- */

    if (!params.input)
        error "Please provide --input"

    validateStep(params.step)
    validateRegistrationMethod(params.registration_method)

    // copy_results step doesn't need input CSV validation
    if (params.step != 'copy_results') {
        validateInputCSV(
            params.input,
            requiredColumnsForStep(params.step)
        )
    }

    if (params.dry_run) {
        log.info "DRY RUN: all validations passed"
        return
    }

    /* -------------------- COPY RESULTS ONLY -------------------- */

    if (params.step == 'copy_results') {
        if (!params.savedir) {
            error "Please provide --savedir for copy_results step"
        }
        if (params.savedir == params.outdir) {
            error "savedir and outdir cannot be the same for copy_results step"
        }

        def source_path = params.outdir.startsWith('/') ? params.outdir : "${workflow.launchDir}/${params.outdir}"
        COPY_RESULTS(
            Channel.of('ready'),
            source_path,
            params.savedir
        )
        return
    }

    /* -------------------- PREPROCESSING -------------------- */

    // Pre-count images and channels per patient for streaming groupTuple operations
    def patient_counts = CsvUtils.countImagesPerPatient(params.input)
    def channel_counts = CsvUtils.countChannelsPerPatient(params.input)

    if (params.step == 'preprocessing') {
        ch_input = loadInputChannel(params.input, 'path_to_file', patient_counts, channel_counts)
        PREPROCESSING(ch_input)
    }
    
    /* -------------------- REGISTRATION -------------------- */

    if (params.step in ['preprocessing','registration']) {

        // When starting from registration, params.input is a string path to CSV
        // When continuing from preprocessing, use direct channel (streaming, no wait)
        ch_for_registration = params.step == 'registration'
            ? loadInputChannel(params.input, 'preprocessed_image', patient_counts, channel_counts)
            : PREPROCESSING.out.preprocessed  // Direct channel - enables patient-level parallelism!

        REGISTRATION(
            ch_for_registration,
            params.registration_method
        )
    }

    /* -------------------- POSTPROCESSING -------------------- */

    if (params.step in ['preprocessing','registration','postprocessing']) {

        // When starting from postprocessing, params.input is a string path to CSV
        // When continuing from registration, use direct channel (streaming, no wait)
        ch_for_postprocessing = params.step == 'postprocessing'
            ? loadInputChannel(params.input, 'registered_image', patient_counts, channel_counts)
            : REGISTRATION.out.registered  // Direct channel - enables patient-level parallelism!

        POSTPROCESSING(ch_for_postprocessing)

        /* -------------------- COPY RESULTS TO SAVEDIR -------------------- */

        // Copy results to savedir after postprocessing completes
        if (params.savedir && params.savedir != params.outdir) {
            // Construct absolute source path from workflow launch directory
            def source_path = params.outdir.startsWith('/') ? params.outdir : "${workflow.launchDir}/${params.outdir}"
            COPY_RESULTS(
                POSTPROCESSING.out.checkpoint_csv.map { 'ready' },
                source_path,
                params.savedir
            )
        }
    }

    /* -------------------- TRACE AGGREGATION -------------------- */

    // Aggregate input size logs from all processes (only if tracing enabled)
    if (params.enable_trace) {
        ch_all_sizes = Channel.empty()

        // PREPROCESSING only runs when step == 'preprocessing'
        if (params.step == 'preprocessing') {
            ch_all_sizes = ch_all_sizes.mix(PREPROCESSING.out.size_logs)
        }
        // REGISTRATION runs when step is 'preprocessing' or 'registration'
        if (params.step in ['preprocessing', 'registration']) {
            ch_all_sizes = ch_all_sizes.mix(REGISTRATION.out.size_logs)
        }
        // POSTPROCESSING runs for all three steps
        if (params.step in ['preprocessing', 'registration', 'postprocessing']) {
            ch_all_sizes = ch_all_sizes.mix(POSTPROCESSING.out.size_logs)
        }

        AGGREGATE_SIZE_LOGS(ch_all_sizes.collect())
    }
}

/*
================================================================================
COMPLETION HANDLERS
================================================================================
*/

workflow.onComplete {
    if (workflow.success) {
        log.info "Pipeline completed successfully!"
        // Work directory cleanup handled by nf-boost plugin during execution
    } else {
        log.error "Pipeline failed - work directory preserved for debugging"
    }
}
