nextflow.enable.dsl = 2

/*
================================================================================
IMPORT SUBWORKFLOWS
================================================================================
*/

include { PREPROCESSING  } from './subworkflows/local/preprocess'
include { REGISTRATION   } from './subworkflows/local/registration'
include { POSTPROCESSING } from './subworkflows/local/postprocess'

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

    validateInputCSV(
        params.input,
        requiredColumnsForStep(params.step)
    )

    if (params.dry_run) {
        log.info "DRY RUN: all validations passed"
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
        ch_qc               = REGISTRATION.out.qc
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

        // Copy results from outdir to savedir
        if (params.savedir && params.savedir != params.outdir) {
            log.info "Copying results from ${params.outdir} to ${params.savedir}..."

            def outdir = new File(params.outdir)
            def savedir = new File(params.savedir)

            // Create savedir if it doesn't exist
            if (!savedir.exists()) {
                savedir.mkdirs()
                log.info "Created save directory: ${params.savedir}"
            }

            // Copy all contents from outdir to savedir using rsync
            try {
                def cmd = ["rsync", "-rL", "--progress", "${params.outdir}/", "${params.savedir}/"]
                def proc = cmd.execute()
                proc.waitFor()

                if (proc.exitValue() == 0) {
                    log.info "Successfully copied results to ${params.savedir}"
                } else {
                    log.error "Failed to copy results to ${params.savedir}"
                    log.error "Error: ${proc.err.text}"
                }
            } catch (Exception e) {
                log.error "Failed to copy results to ${params.savedir}: ${e.message}"
            }
        } else if (!params.savedir) {
            log.info "No savedir specified - results remain in ${params.outdir}"
        } else {
            log.info "savedir is the same as outdir - no copy needed"
        }

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
