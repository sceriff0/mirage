nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { CONVERT_IMAGE         } from '../../modules/local/convert_nd2'
include { PREPROCESS            } from '../../modules/local/preprocess'
include { MAX_DIM               } from '../../modules/local/max_dim'
include { PAD_IMAGES            } from '../../modules/local/pad_images'
include { WRITE_CHECKPOINT_CSV  } from '../../modules/local/write_checkpoint_csv'

/*
========================================================================================
    SUBWORKFLOW: PREPROCESSING
========================================================================================
    Description:
        Converts images to standardized OME-TIFF (with DAPI in channel 0),
        applies BaSiC illumination correction, computes maximum dimensions across
        all images, and pads images to uniform size.

    Input:
        ch_input: Channel of [meta, file] tuples where meta contains:
                  - patient_id: Patient identifier
                  - is_reference: Boolean indicating reference image
                  - channels: List of channel names (DAPI must be first)

    Output:
        padded: Padded OME-TIFF images with uniform dimensions
        preprocessed: Preprocessed OME-TIFF images (before padding)
        checkpoint_csv: CSV for restart capability
========================================================================================
*/

workflow PREPROCESSING {
    take:
    ch_input  // Channel of [meta, file] tuples

    main:
    // Conditionally convert to standardized OME-TIFF format
    if (!params.skip_nd2_conversion) {
        CONVERT_IMAGE ( ch_input )

        // Parse output channels from channels.txt file and update metadata
        ch_for_preprocess = CONVERT_IMAGE.out.ome_tiff
            .map { meta, ome_file, channels_file ->
                // Read output channels from file (DAPI will be first)
                def output_channels = channels_file.text.trim().split(',')
                // Update meta with output channel order
                def updated_meta = meta.clone()
                updated_meta.channels = output_channels
                [updated_meta, ome_file]
            }
    } else {
        ch_for_preprocess = ch_input
    }

    // Preprocess each file (BaSiC correction)
    // Extract just the file path for PREPROCESS (legacy module)
    ch_preprocess_input = ch_for_preprocess.map { meta, file -> file }
    PREPROCESS ( ch_preprocess_input )

    // Compute max dimensions from all preprocessed images
    MAX_DIM ( PREPROCESS.out.dims.collect() )

    // Pad each image to max dimensions
    ch_to_pad = PREPROCESS.out.preprocessed
        .combine(MAX_DIM.out.max_dims_file)

    PAD_IMAGES ( ch_to_pad )

    // Reconstruct metadata channel by joining with original metadata
    ch_padded_with_meta = ch_for_preprocess
        .map { meta, file ->
            // Extract basename from original file (before padding)
            def basename = file.name
            return tuple(basename, meta)
        }
        .combine(
            PAD_IMAGES.out.padded.map { padded_file ->
                // Extract basename by removing _padded suffix
                def basename = padded_file.name.replaceAll('_padded', '')
                return tuple(basename, padded_file)
            },
            by: 0
        )
        .map { basename, meta, padded_file ->
            return tuple(meta, padded_file)
        }

    // Generate checkpoint CSV for restart from preprocessing step
    ch_checkpoint_data = ch_padded_with_meta
        .map { meta, file ->
            def abs_path = file.toString()
            def channels = meta.channels.join('|')
            [meta.patient_id, abs_path, meta.is_reference, channels]
        }
        .collect()

    WRITE_CHECKPOINT_CSV(
        'preprocessed',
        'patient_id,padded_image,is_reference,channels',
        ch_checkpoint_data
    )

    emit:
    padded = ch_padded_with_meta
    preprocessed = PREPROCESS.out.preprocessed
    max_dims_file = MAX_DIM.out.max_dims_file
    checkpoint_csv = WRITE_CHECKPOINT_CSV.out.csv
}
