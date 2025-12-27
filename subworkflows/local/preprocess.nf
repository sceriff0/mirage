nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { CONVERT_IMAGE         } from '../../modules/local/convert_nd2'
include { PREPROCESS            } from '../../modules/local/preprocess'
include { WRITE_CHECKPOINT_CSV  } from '../../modules/local/write_checkpoint_csv'

/*
========================================================================================
    SUBWORKFLOW: PREPROCESSING
========================================================================================
    Description:
        Converts images to standardized OME-TIFF (with DAPI in channel 0) and
        applies BaSiC illumination correction.

        Note: Padding is now handled in the registration workflow for GPU/CPU methods only.
        VALIS uses preprocessed images directly without padding.

    Input:
        ch_input: Channel of [meta, file] tuples where meta contains:
                  - patient_id: Patient identifier
                  - is_reference: Boolean indicating reference image
                  - channels: List of channel names (DAPI must be first)

    Output:
        preprocessed: Preprocessed OME-TIFF images with [meta, file] tuples
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

                // FIX EDGE CASE #2: CRITICAL VALIDATION - DAPI must be in channel 0!
                if (output_channels[0].toUpperCase() != 'DAPI') {
                    throw new Exception("""
                    ❌ CRITICAL: DAPI must be in channel 0 after conversion for ${meta.patient_id}!
                    Got channels: ${output_channels}
                    DAPI is in position: ${output_channels.findIndexOf { it.toUpperCase() == 'DAPI' }}
                    💡 This is a bug in the convert_image.py script - it should place DAPI first
                    """.stripIndent())
                }

                // Update meta with output channel order
                def updated_meta = meta.clone()
                updated_meta.channels = output_channels
                [updated_meta, ome_file]
            }
    } else {
        ch_for_preprocess = ch_input
    }

    // Preprocess each file (BaSiC correction)
    // PREPROCESS now accepts [meta, file] tuples and preserves metadata
    PREPROCESS ( ch_for_preprocess )

    // Preprocessed files already have metadata attached
    ch_preprocessed_with_meta = PREPROCESS.out.preprocessed

    // Generate checkpoint CSV for restart from preprocessing step
    ch_checkpoint_data = ch_preprocessed_with_meta
        .map { meta, file ->
            def abs_path = file.toString()
            def channels = meta.channels.join('|')
            [meta.patient_id, abs_path, meta.is_reference, channels]
        }
        .collect()

    WRITE_CHECKPOINT_CSV(
        'preprocessed',
        'patient_id,preprocessed_image,is_reference,channels',
        ch_checkpoint_data
    )

    emit:
    preprocessed = ch_preprocessed_with_meta
    checkpoint_csv = WRITE_CHECKPOINT_CSV.out.csv
}
