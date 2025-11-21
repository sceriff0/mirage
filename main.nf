nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { CONVERT_ND2 } from './modules/local/convert_nd2'
include { PREPROCESS } from './modules/local/preprocess'
include { REGISTER   } from './modules/local/register'
include { MERGE      } from './modules/local/merge'
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
    // 4. MODULE: Register all preprocessed files
    //    Warps each slide individually (RAM-efficient approach)
    //    Collects all preprocessed files and passes them to REGISTER
    REGISTER ( PREPROCESS.out.preprocessed.collect() )

    // 5. MODULE: Merge registered slides into single multi-channel OME-TIFF
    //    Takes all individually warped slides and merges them, skipping duplicates
    MERGE ( REGISTER.out.registered_slides.collect() )

    // 6. MODULE: Segment the merged WSI
    SEGMENT ( MERGE.out.merged )

    // 7. MODULE: Classify cell types using deepcell-types
    //    Using cell_mask (expanded) for classification
    CLASSIFY (
        MERGE.out.merged,
        SEGMENT.out.cell_mask
    )
}
