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

def loadInputChannel(csv_path, image_column) {
    return Channel
        .fromPath(csv_path, checkIfExists: true)
        .splitCsv(header: true)
        .map { row ->
            // Use CsvUtils to handle the complex metadata parsing
            def meta = CsvUtils.parseMetadata(row, "CSV ${csv_path}") 
            return tuple(meta, file(row[image_column]))
        }
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

    if (params.step == 'preprocessing') {

        ch_input = loadInputChannel(params.input, 'path_to_file')
        PREPROCESSING(ch_input)
        ch_preprocess_csv = PREPROCESSING.out.checkpoint_csv
    }
    
    /* -------------------- REGISTRATION -------------------- */

    if (params.step in ['preprocessing','registration']) {

        // When starting from registration, params.input is a string path
        // When continuing from preprocessing, ch_preprocess_csv is a channel
        ch_for_registration = params.step == 'registration'
            ? loadInputChannel(params.input, 'preprocessed_image')
            : ch_preprocess_csv
                .splitCsv(header: true)
                .map { row ->
                    def meta = CsvUtils.parseMetadata(row, "Checkpoint CSV")
                    return tuple(meta, file(row['preprocessed_image']))
                }

        REGISTRATION(
            ch_for_registration,
            params.registration_method
        )

        ch_registration_csv = REGISTRATION.out.checkpoint_csv
    }

    /* -------------------- POSTPROCESSING -------------------- */

    if (params.step in ['preprocessing','registration','postprocessing']) {

        ch_for_postprocessing = params.step == 'postprocessing'
            ? loadInputChannel(params.input, 'registered_image')
            : ch_registration_csv
                .splitCsv(header: true)
                .map { row ->
                    def meta = CsvUtils.parseMetadata(row, "Checkpoint CSV")
                    return tuple(meta, file(row['registered_image']))
                }

        POSTPROCESSING(ch_for_postprocessing)

        ch_postprocessing_csv = POSTPROCESSING.out.checkpoint_csv

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

        if (params.step in ['preprocessing', 'registration', 'postprocessing']) {
            ch_all_sizes = ch_all_sizes.mix(PREPROCESSING.out.size_logs)
        }
        if (params.step in ['preprocessing', 'registration', 'postprocessing']) {
            ch_all_sizes = ch_all_sizes.mix(REGISTRATION.out.size_logs)
        }
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

        // Clean up work directory if requested
        if (params.cleanup_work) {
            log.info "Cleaning up work directory: ${workflow.workDir}"
            def workDir = new File("${workflow.workDir}")
            if (workDir.exists() && workDir.isDirectory()) {
                try {
                    workDir.deleteDir()
                    log.info "Work directory removed successfully"
                } catch (Exception e) {
                    log.warn "Failed to remove work directory: ${e.message}"
                }
            }
        }
    } else {
        log.error "Pipeline failed - work directory preserved for debugging"
    }
}
