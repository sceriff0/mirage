nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { CONVERT_ND2 } from './modules/local/convert_nd2'
include { PREPROCESS } from './modules/local/preprocess'
include { REGISTRATION } from './subworkflows/registration'
include { SEGMENT    } from './modules/local/segment'
include { CLASSIFY   } from './modules/local/classify'


/*
========================================================================================
    RUN MAIN WORKFLOW
========================================================================================
*/

workflow {
     
    // TODO Add patient id for multi-patient runs
    // Validate input parameters
    if (!params.input) {
        error "Please provide an input glob pattern with --input"
    }
    // TODO: Frst file is reference
    // 1. Create input channel from glob pattern (ND2 files)
    ch_input = Channel.fromPath(params.input, checkIfExists: true)

    // TODO: Accept multiple input formats (e.g., OME-TIFF) and set metadata 
    // TODO: based on filename and channel order (correct / reversed)
    // 2. MODULE: Convert ND2 to OME-TIFF 
    CONVERT_ND2 ( ch_input )

    // 3. MODULE: Preprocess each converted file
    PREPROCESS ( CONVERT_ND2.out.ome_tiff )

    // TODO: Add classic registration
    // 4. SUBWORKFLOW: 3-step VALIS registration (compute -> micro -> warp+merge)
    //    Collects all preprocessed files and passes them to REGISTRATION
    REGISTRATION ( PREPROCESS.out.preprocessed.collect() )

    // 5. MODULE: Segment the merged WSI
    SEGMENT ( REGISTRATION.out.merged )

    // 6. MODULE: Classify cell types using deepcell-types
    //    Using cell_mask (expanded) for classification
    CLASSIFY (
        REGISTRATION.out.merged,
        SEGMENT.out.cell_mask
    )
}
