nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT SUBWORKFLOWS
========================================================================================
*/

include { PREPROCESSING  } from './subworkflows/local/preprocess'
include { REGISTRATION   } from './subworkflows/local/registration'
include { POSTPROCESSING } from './subworkflows/local/postprocess'
include { RESULTS        } from './subworkflows/local/results'


/*
========================================================================================
    RUN MAIN WORKFLOW
========================================================================================
*/

workflow {

    // Validate input parameters
    if (!params.input) {
        error "Please provide an input glob pattern with --input"
    }

    // 1. Create input channel from glob pattern (ND2 files)
    ch_input = channel.fromPath(params.input, checkIfExists: true)

    // 2. SUBWORKFLOW: Preprocessing (Convert, Preprocess, Max Dimensions, Padding)
    PREPROCESSING ( ch_input )

    // 3. SUBWORKFLOW: Registration (VALIS, GPU, or CPU)
    REGISTRATION (
        PREPROCESSING.out.padded,
        params.registration_method,
        params.reg_reference_markers
    )

    // 4. SUBWORKFLOW: Postprocessing (Segment, Split, Quantify, Merge, Phenotype)
    POSTPROCESSING (
        REGISTRATION.out.registered,
        params.reg_reference_markers
    )

    // 5. SUBWORKFLOW: Results (Merge, Conversion to Pyramid, Save)
    RESULTS (
        REGISTRATION.out.registered,
        POSTPROCESSING.out.cell_mask,
        POSTPROCESSING.out.phenotype_mask,
        POSTPROCESSING.out.merged_csv,
        POSTPROCESSING.out.phenotype_csv,
        params.savedir
    )
}
