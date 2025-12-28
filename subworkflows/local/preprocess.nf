nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { CONVERT_IMAGE         } from '../../modules/local/convert_image'
include { PREPROCESS            } from '../../modules/local/preprocess'
include { WRITE_CHECKPOINT_CSV  } from '../../modules/local/write_checkpoint_csv'

/*
========================================================================================
    SUBWORKFLOW: PREPROCESSING
========================================================================================
    Description:
        Converts images to standardized OME-TIFF (with DAPI in channel 0) and
        applies BaSiC illumination correction.

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
    // Convert to standardized OME-TIFF format
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

    // Preprocess each file (BaSiC correction)
    // PREPROCESS accepts [meta, file] tuples
    PREPROCESS ( ch_for_preprocess )

    // Preprocessed files already have metadata attached
    ch_preprocessed_with_meta = PREPROCESS.out.preprocessed

    // Generate checkpoint CSV for restart from preprocessing step
    ch_checkpoint_data = ch_preprocessed_with_meta
        .map { meta, file ->
            def abs_path = file.toString()
            def channels = meta.channels.toString()
            [meta.patient_id, abs_path, meta.is_reference.toString(), channels]
        }
        .toList()
        .view { data -> "Checkpoint data: $data" }

    WRITE_CHECKPOINT_CSV(
        'preprocessed',
        'patient_id,preprocessed_image,is_reference,channels',
        ch_checkpoint_data
    )

    emit:
    preprocessed = ch_preprocessed_with_meta
    checkpoint_csv = WRITE_CHECKPOINT_CSV.out.csv
}
