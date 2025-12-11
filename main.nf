nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT SUBWORKFLOWS
========================================================================================
*/

include { PREPROCESSING  } from './subworkflows/local/preprocess'
include { REGISTRATION   } from './subworkflows/local/registration'
include { QUANTIFICATION } from './subworkflows/local/quantification'

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { SAVE_RESULTS } from './modules/local/save_results'


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

    // 4. SUBWORKFLOW: Quantification (Segment, Split, Quantify, Merge, Phenotype)
    QUANTIFICATION (
        REGISTRATION.out.registered,
        params.reg_reference_markers
    )

    // 5. MODULE: Save all results to final output directory
    // Collect all outputs to ensure all processes complete before saving
    ch_all_outputs = channel.empty()
        .mix(
            REGISTRATION.out.registered,
            QUANTIFICATION.out.cell_mask,
            QUANTIFICATION.out.merged_csv,
            QUANTIFICATION.out.individual_csvs,
            QUANTIFICATION.out.phenotype_csv,
            QUANTIFICATION.out.phenotype_mask
        )
        .collect()
        .map { _files ->
            // All files are published under the same parent directory
            return file("${params.outdir}")
        }

    SAVE_RESULTS (
        ch_all_outputs,
        params.savedir
    )
}
