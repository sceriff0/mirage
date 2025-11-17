nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { CONVERT_ND2 } from './modules/local/convert_nd2'
include { PREPROCESS } from './modules/local/preprocess'
include { REGISTER   } from './modules/local/register'
include { SEGMENT    } from './modules/local/segment'
include { QUANTIFY   } from './modules/local/quantify'
include { PHENOTYPE  } from './modules/local/phenotype'


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
    ch_input = Channel.fromPath(params.input, checkIfExists: true)

    // TODO: Accept multiple input formats (e.g., OME-TIFF) and set metadata 
    // TODO: based on filename and channel order (correct / reversed)
    // 2. MODULE: Convert ND2 to OME-TIFF 
    CONVERT_ND2 ( ch_input )

    // 3. MODULE: Preprocess each converted file
    PREPROCESS ( CONVERT_ND2.out.ome_tiff )

    // 4. MODULE: Register/merge all preprocessed files
    //    Collects all preprocessed files and passes them to REGISTER
    REGISTER ( PREPROCESS.out.preprocessed.collect() )

    // 5. MODULE: Segment the merged WSI
    SEGMENT ( REGISTER.out.merged )

    // 6. MODULE: Quantify cells
    QUANTIFY (
        REGISTER.out.merged,
        SEGMENT.out.mask
    )

    // 7. MODULE: Phenotype cells
    PHENOTYPE (
        QUANTIFY.out.csv,
        SEGMENT.out.mask
    )
}
