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
    ch_nd2_files  // Channel of ND2 input files

    main:
    // Convert ND2 to OME-TIFF
    CONVERT_ND2 ( ch_nd2_files )

    // Preprocess each converted file (BaSiC correction)
    PREPROCESS ( CONVERT_ND2.out.ome_tiff )

    // Compute max dimensions from all preprocessed images
    MAX_DIM ( PREPROCESS.out.dims.collect() )

    // Pad each image to max dimensions
    ch_to_pad = PREPROCESS.out.preprocessed
        .combine(MAX_DIM.out.max_dims_file)

    PAD_IMAGES ( ch_to_pad )

    emit:
    padded = PAD_IMAGES.out.padded
    max_dims_file = MAX_DIM.out.max_dims_file
}
