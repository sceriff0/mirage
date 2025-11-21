nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES AND SUBWORKFLOWS
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

    // 4. SUBWORKFLOW: Register/merge all preprocessed files + generate QC
    //    Simplified architecture: compute → warp in parallel → merge → QC
    REGISTRATION ( PREPROCESS.out.preprocessed )

    // 5. MODULE: Segment the merged registered image
    SEGMENT ( REGISTRATION.out.merged )

    // 6. MODULE: Classify cell types using deepcell-types
    CLASSIFY (
        REGISTRATION.out.merged,
        SEGMENT.out.cell_mask
    )
}
