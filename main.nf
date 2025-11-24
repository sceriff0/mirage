nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { CONVERT_ND2  } from './modules/local/convert_nd2'
include { PREPROCESS   } from './modules/local/preprocess'
include { PAD_IMAGES   } from './modules/local/pad_images'
include { REGISTER     } from './modules/local/register'
include { GPU_REGISTER } from './modules/local/register_gpu'
include { MERGE        } from './modules/local/merge'
include { SEGMENT      } from './modules/local/segment'
include { CLASSIFY     } from './modules/local/classify'


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

    // 2. MODULE: Convert ND2 to OME-TIFF
    CONVERT_ND2 ( ch_input )

    // 3. MODULE: Preprocess each converted file
    PREPROCESS ( CONVERT_ND2.out.ome_tiff )

    // 4. MODULE: Register using either classic or GPU method
    if (params.registration_method == 'gpu') {
        // GPU registration workflow:
        // Step 1: Pad all images to common maximum dimensions
        PAD_IMAGES ( PREPROCESS.out.preprocessed.collect() )

        // Step 2: Create (reference, moving) pairs from padded images
        ch_pairs = PAD_IMAGES.out.padded
            .flatten()
            .collect()
            .flatMap { files ->
                def reference = files[0]
                def moving_files = files.drop(1)
                return moving_files.collect { m -> tuple(reference, m) }
            }

        GPU_REGISTER ( ch_pairs )

        // Combine reference (unchanged) with registered moving images
        ch_reference = PAD_IMAGES.out.padded.flatten().first()
        ch_registered = ch_reference.concat(GPU_REGISTER.out.registered)

    } else {
        // Classic registration: collect all preprocessed files
        REGISTER ( PREPROCESS.out.preprocessed.collect() )
        ch_registered = REGISTER.out.registered_slides
    }
    
    // 5. MODULE: Merge registered slides into single multi-channel OME-TIFF
    MERGE ( ch_registered.collect() )

    // 6. MODULE: Segment the merged WSI
    SEGMENT ( MERGE.out.merged )

    // 7. MODULE: Classify cell types using deepcell-types
    CLASSIFY (
        MERGE.out.merged,
        SEGMENT.out.cell_mask
    )
}
