nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT SUBWORKFLOWS
========================================================================================
*/

include { PREPROCESSING  } from './subworkflows/local/preprocess'
include { REGISTRATION   } from './subworkflows/local/registration'
include { POSTPROCESSING } from './subworkflows/local/postprocess'
include { RESULTS        } from './subworkflows/local/results'


/*
========================================================================================
    HELPER FUNCTIONS
========================================================================================
*/

// Parse CSV row and create metadata map with channels
// Used for loading from checkpoint CSVs
def parseMetadata(row) {
    def channels = row.channels.split('\\|')
    return [
        patient_id: row.patient_id,
        is_reference: row.is_reference.toBoolean(),
        channels: channels
    ]
}


/*
========================================================================================
    RUN MAIN WORKFLOW
========================================================================================
*/

workflow {

    // Validate input parameters
    if (!params.input) {
        error "Please provide an input glob pattern or checkpoint CSV with --input"
    }

    // Validate step parameter
    def valid_steps = ['preprocessing', 'registration', 'postprocessing', 'results']
    if (!valid_steps.contains(params.step)) {
        error "Invalid step '${params.step}'. Valid steps: ${valid_steps.join(', ')}"
    }

    // ========================================================================
    // STEP: PREPROCESSING
    // ========================================================================
    if (params.step == 'preprocessing') {
        // Start from beginning: Parse input CSV
        // CSV format: patient_id,path_to_file,is_reference,channel_1,channel_2,channel_3 (optional)
        // Note: DAPI can be in any channel position - conversion will place it in channel 0
        ch_input = channel.fromPath(params.input, checkIfExists: true)
            .splitCsv(header: true)
            .map { row ->
                // Extract channel names (collect all channel_* columns)
                def channels = []
                ['channel_1', 'channel_2', 'channel_3'].each { key ->
                    if (row.containsKey(key) && row[key]) {
                        channels.add(row[key])
                    }
                }

                // Validate DAPI is present (can be in any position - will be moved to channel 0 during conversion)
                def has_dapi = channels.any { it.toUpperCase() == 'DAPI' }
                if (!has_dapi) {
                    error "DAPI channel not found for ${row.patient_id}. Channels: ${channels}"
                }

                def meta = [
                    patient_id: row.patient_id,
                    is_reference: row.is_reference.toBoolean(),
                    channels: channels  // Keep original order; conversion will place DAPI first
                ]

                return [meta, file(row.path_to_file)]
            }

        PREPROCESSING ( ch_input )
        ch_padded = PREPROCESSING.out.padded
        ch_preprocessed = PREPROCESSING.out.preprocessed

    } else {
        // Skip preprocessing - will load from checkpoint later
        ch_padded = channel.empty()
        ch_preprocessed = channel.empty()
    }

    // ========================================================================
    // STEP: REGISTRATION
    // ========================================================================
    if (params.step == 'preprocessing') {
        // Continue from preprocessing output (already has metadata)
        ch_for_registration = ch_padded

        // Reconstruct metadata for preprocessed files by matching basenames with padded
        // ch_padded has [meta, file] where file is *_padded.ome.tif
        // preprocessed has files *_corrected.ome.tif
        // We need to match them by removing suffixes
        ch_preprocessed_with_meta = ch_padded
            .map { meta, padded_file ->
                // Extract basename by removing _padded suffix
                def basename = padded_file.name.replaceAll('_padded\\.ome\\.tif.*$', '')
                return tuple(basename, meta)
            }
            .combine(
                PREPROCESSING.out.preprocessed.map { preproc_file ->
                    // Extract basename by removing _corrected suffix
                    def basename = preproc_file.name.replaceAll('_corrected\\.ome\\.tif.*$', '')
                    return tuple(basename, preproc_file)
                },
                by: 0
            )
            .map { basename, meta, preproc_file ->
                return tuple(meta, preproc_file)
            }

    } else if (params.step == 'registration') {
        // Load from preprocessing checkpoint CSV
        ch_for_registration = channel.fromPath(params.input, checkIfExists: true)
            .splitCsv(header: true)
            .map { row -> [parseMetadata(row), file(row.padded_image)] }

        // For VALIS, preprocessed files need to be created from padded
        // Since we're loading from checkpoint, we don't have the preprocessed files
        // Use the padded images with metadata for now
        ch_preprocessed_with_meta = ch_for_registration
    }

    if (params.step in ['preprocessing', 'registration']) {
        // Pass metadata through to registration
        // Both ch_for_registration (padded) and ch_preprocessed_with_meta now have [meta, file] tuples
        REGISTRATION (
            ch_for_registration,
            ch_preprocessed_with_meta,
            params.registration_method
        )
        ch_registered = REGISTRATION.out.registered

    } else {
        // Skip registration - will load from checkpoint later
        ch_registered = channel.empty()
    }

    // ========================================================================
    // STEP: POSTPROCESSING
    // ========================================================================
    if (params.step in ['preprocessing', 'registration']) {
        // Continue from registration output (already has metadata)
        ch_for_postprocessing = ch_registered

    } else if (params.step == 'postprocessing') {
        // Load from registration checkpoint CSV with metadata
        ch_for_postprocessing = channel.fromPath(params.input, checkIfExists: true)
            .splitCsv(header: true)
            .map { row -> [parseMetadata(row), file(row.registered_image)] }
    }

    if (params.step in ['preprocessing', 'registration', 'postprocessing']) {
        POSTPROCESSING ( ch_for_postprocessing )
        ch_phenotype_csv = POSTPROCESSING.out.phenotype_csv
        ch_phenotype_mask = POSTPROCESSING.out.phenotype_mask
        ch_phenotype_mapping = POSTPROCESSING.out.phenotype_mapping
        ch_merged_csv = POSTPROCESSING.out.merged_csv
        ch_cell_mask = POSTPROCESSING.out.cell_mask

    } else {
        // Skip postprocessing - will load from checkpoint later
        ch_phenotype_csv = channel.empty()
        ch_phenotype_mask = channel.empty()
        ch_phenotype_mapping = channel.empty()
        ch_merged_csv = channel.empty()
        ch_cell_mask = channel.empty()
    }

    // ========================================================================
    // STEP: RESULTS
    // ========================================================================
    if (params.step in ['preprocessing', 'registration', 'postprocessing']) {
        // Continue from postprocessing output
        // Use channels from previous step
        ch_registered_for_results = ch_registered
        ch_qc_for_results = REGISTRATION.out.qc

    } else if (params.step == 'results') {
        // Load from postprocessing checkpoint CSV (new format includes patient_id and is_reference)
        ch_checkpoint = channel.fromPath(params.input, checkIfExists: true)
            .splitCsv(header: true)
            .first()

        ch_registered_for_results = channel.empty()
        ch_qc_for_results = channel.empty()
        // New CSV format: patient_id,is_reference,phenotype_csv,phenotype_mask,phenotype_mapping,merged_csv,cell_mask
        ch_phenotype_csv = ch_checkpoint.map { row -> file(row.phenotype_csv) }
        ch_phenotype_mask = ch_checkpoint.map { row -> file(row.phenotype_mask) }
        ch_phenotype_mapping = ch_checkpoint.map { row -> file(row.phenotype_mapping) }
        ch_merged_csv = ch_checkpoint.map { row -> file(row.merged_csv) }
        ch_cell_mask = ch_checkpoint.map { row -> file(row.cell_mask) }
    }

    if (params.step in ['preprocessing', 'registration', 'postprocessing', 'results']) {
        RESULTS (
            ch_registered_for_results,
            ch_qc_for_results,
            ch_cell_mask,
            ch_phenotype_mask,
            ch_phenotype_mapping,
            ch_merged_csv,
            ch_phenotype_csv,
            params.savedir
        )
    }
}
