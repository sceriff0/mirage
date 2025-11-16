nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { PREPROCESS } from 'modules/local/preprocess'
include { REGISTER   } from 'modules/local/register'
include { SEGMENT    } from 'modules/local/segment'
include { QUANTIFY   } from 'modules/local/quantify'
include { PHENOTYPE  } from 'modules/local/phenotype'
include { GET_PREPROCESS_DIR } from 'modules/local/get_preprocess_dir' // The helper module


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

    // 1. Create input channel from glob pattern
    ch_input = Channel.fromPath(params.input, checkIfExists: true)

    // 2. MODULE: Preprocess each input file (Parallel)
    PREPROCESS ( ch_input )

    // 3. SYNCHRONIZATION STEP: 
    //    Waits for all PREPROCESS tasks to finish, then triggers GET_PREPROCESS_DIR.
    ch_preprocessed_files_list = PREPROCESS.out.preprocessed.collect()
    GET_PREPROCESS_DIR( ch_preprocessed_files_list )
    
    // 4. MODULE: Register/merge all preprocessed files (Serial Merge)
    REGISTER ( GET_PREPROCESS_DIR.out.preprocess_dir )

    // 5. MODULE: Segment the merged WSI (Serial)
    SEGMENT ( REGISTER.out.merged )

    // 6. MODULE: Quantify cells (Serial)
    QUANTIFY (
        REGISTER.out.merged,
        SEGMENT.out.mask
    )

    // 7. MODULE: Phenotype cells (Serial)
    PHENOTYPE (
        QUANTIFY.out.csv,
        SEGMENT.out.mask
    )
}
