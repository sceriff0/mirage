nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT SUBWORKFLOWS
========================================================================================
*/

include { PREPROCESSING  } from './subworkflows/local/preprocess'
include { REGISTRATION   } from './subworkflows/local/registration'
include { POSTPROCESSING } from './subworkflows/local/postprocess'

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { MERGE          } from './modules/local/merge'
include { CONVERSION } from './modules/local/conversion'
include { SAVE_RESULTS   } from './modules/local/save_results'


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

    // 5. MODULE: Merge all registered images into single multichannel OME-TIFF
    MERGE ( REGISTRATION.out.registered.collect() )

    // 6. MODULE: Create pyramidal OME-TIFF with masks
    CONVERSION (
        MERGE.out.merged,
        POSTPROCESSING.out.cell_mask,
        POSTPROCESSING.out.phenotype_mask
    )

    // 7. MODULE: Save all results to final output directory
    // Collect all outputs to ensure all processes complete before saving
    ch_all_outputs = channel.empty()
        .mix(
            REGISTRATION.out.registered,
            POSTPROCESSING.out.cell_mask,
            POSTPROCESSING.out.merged_csv,
            POSTPROCESSING.out.individual_csvs,
            POSTPROCESSING.out.phenotype_csv,
            POSTPROCESSING.out.phenotype_mask,
            MERGE.out.merged,
            CONVERSION.out.pyramid
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
