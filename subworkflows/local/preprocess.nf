nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { CONVERT_ND2  } from '../../modules/local/convert_nd2'
include { PREPROCESS   } from '../../modules/local/preprocess'
include { MAX_DIM      } from '../../modules/local/max_dim'
include { PAD_IMAGES   } from '../../modules/local/pad_images'

/*
========================================================================================
    SUBWORKFLOW: PREPROCESSING
========================================================================================
    Description:
        Converts ND2 files to OME-TIFF, applies BaSiC illumination correction,
        computes maximum dimensions across all images, and pads images to uniform size.

    Input:
        ch_nd2_files: Channel of ND2 input files

    Output:
        padded: Padded OME-TIFF images with uniform dimensions
========================================================================================
*/

workflow PREPROCESSING {
    take:
    ch_input_files  // Channel of input files (ND2 or TIFF)

    main:
    // Conditionally convert ND2 to OME-TIFF (skip if using TIFF input)
    if (!params.skip_nd2_conversion) {
        CONVERT_ND2 ( ch_input_files )
        ch_for_preprocess = CONVERT_ND2.out.ome_tiff
    } else {
        ch_for_preprocess = ch_input_files
    }

    // Preprocess each file (BaSiC correction)
    PREPROCESS ( ch_for_preprocess )

    // Compute max dimensions from all preprocessed images
    MAX_DIM ( PREPROCESS.out.dims.collect() )

    // Pad each image to max dimensions
    ch_to_pad = PREPROCESS.out.preprocessed
        .combine(MAX_DIM.out.max_dims_file)

    PAD_IMAGES ( ch_to_pad )

    emit:
    padded = PAD_IMAGES.out.padded
    preprocessed = PREPROCESS.out.preprocessed
    max_dims_file = MAX_DIM.out.max_dims_file
}
