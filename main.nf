nextflow.enable.dsl = 2

plugins {
    id 'nf-validation@1.1.3'
}

/*
================================================================================
IMPORT SUBWORKFLOWS
================================================================================
*/

include { PREPROCESSING  } from './subworkflows/local/preprocess.nf'
include { REGISTRATION   } from './subworkflows/local/registration.nf'
include { POSTPROCESSING } from './subworkflows/local/postprocess.nf'
include { RESULTS        } from './subworkflows/local/results.nf'

/*
================================================================================
IMPORT HELPERS
================================================================================
*/

include { parseMetadata } from './lib/metadata'
include {
    validateStep;
    validateRegistrationMethod;
    validateInputCSV;
    requiredColumnsForStep
} from './lib/validation'
include { loadCheckpointCsv } from './lib/csv'

include { validateParameters; paramsSummaryLog } from 'plugin/nf-validation'

/*
================================================================================
WORKFLOW
================================================================================
*/

workflow {

    /* -------------------- PARAMETER VALIDATION -------------------- */

    validateParameters()
    log.info paramsSummaryLog(workflow)

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

        ch_input = channel.fromPath(params.input)
            .splitCsv(header: true)
            .map { row ->
                [ parseMetadata(row), file(row.path_to_file) ]
            }

        PREPROCESSING(ch_input)
        ch_preprocess_csv = PREPROCESSING.out.checkpoint_csv
    }
    '''
    /* -------------------- REGISTRATION -------------------- */

    if (params.step in ['preprocessing','registration']) {

        ch_preprocess_csv = params.step == 'registration'
            ? params.input
            : ch_preprocess_csv

        ch_for_registration =
            loadCheckpointCsv(ch_preprocess_csv, 'preprocessed_image')

        REGISTRATION(
            ch_for_registration,
            params.registration_method
        )

        ch_registration_csv = REGISTRATION.out.checkpoint_csv
        ch_registered       = REGISTRATION.out.registered
        ch_qc               = REGISTRATION.out.qc
    }

    /* -------------------- POSTPROCESSING -------------------- */

    if (params.step in ['preprocessing','registration','postprocessing']) {

        ch_registration_csv = params.step == 'postprocessing'
            ? params.input
            : ch_registration_csv

        ch_for_postprocessing =
            loadCheckpointCsv(ch_registration_csv, 'registered_image')

        POSTPROCESSING(ch_for_postprocessing)

        ch_postprocessing_csv = POSTPROCESSING.out.checkpoint_csv
    }
    
    /* -------------------- RESULTS -------------------- */

    if (params.step == 'results')
        ch_postprocessing_csv = params.input

    RESULTS(
        ch_registered ?: channel.empty(),
        ch_qc ?: channel.empty(),
        POSTPROCESSING.out.cell_mask,
        POSTPROCESSING.out.phenotype_mask,
        POSTPROCESSING.out.phenotype_mapping,
        POSTPROCESSING.out.merged_csv,
        POSTPROCESSING.out.phenotype_csv,
        params.savedir
    )
    '''
}
