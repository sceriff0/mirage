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
        // Use checkpoint CSV from preprocessing - ensures CSV is created and used
        ch_preprocess_csv = PREPROCESSING.out.checkpoint_csv
    } else if (params.step == 'registration') {
        // Load from user-provided checkpoint CSV
        ch_preprocess_csv = channel.fromPath(params.input, checkIfExists: true)
    }

    // Always parse from CSV for consistency
    if (params.step in ['preprocessing', 'registration']) {
        ch_for_registration = ch_preprocess_csv
            .splitCsv(header: true)
            .map { row -> [parseMetadata(row), file(row.padded_image)] }

        // For VALIS, preprocessed files need to be the same as padded when loading from checkpoint
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
        // Use checkpoint CSV from registration - ensures CSV is created and used
        ch_registration_csv = REGISTRATION.out.checkpoint_csv
    } else if (params.step == 'postprocessing') {
        // Load from user-provided checkpoint CSV
        ch_registration_csv = channel.fromPath(params.input, checkIfExists: true)
    }

    // Always parse from CSV for consistency
    if (params.step in ['preprocessing', 'registration', 'postprocessing']) {
        ch_for_postprocessing = ch_registration_csv
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
        // Use checkpoint CSV from postprocessing - ensures CSV is created and used
        ch_postprocessing_csv = POSTPROCESSING.out.checkpoint_csv
    } else if (params.step == 'results') {
        // Load from user-provided checkpoint CSV
        ch_postprocessing_csv = channel.fromPath(params.input, checkIfExists: true)
    }

    // Always parse from CSV for results inputs
    if (params.step in ['preprocessing', 'registration', 'postprocessing', 'results']) {
        ch_checkpoint = ch_postprocessing_csv
            .splitCsv(header: true)
            .first()

        // Load outputs from CSV (needed for results step when loading from checkpoint)
        if (params.step == 'results') {
            ch_phenotype_csv = ch_checkpoint.map { row -> file(row.phenotype_csv) }
            ch_phenotype_mask = ch_checkpoint.map { row -> file(row.phenotype_mask) }
            ch_phenotype_mapping = ch_checkpoint.map { row -> file(row.phenotype_mapping) }
            ch_merged_csv = ch_checkpoint.map { row -> file(row.merged_csv) }
            ch_cell_mask = ch_checkpoint.map { row -> file(row.cell_mask) }
        }

        // Set registered and QC channels appropriately
        ch_registered_for_results = (params.step == 'results') ? channel.empty() : ch_registered
        ch_qc_for_results = (params.step == 'results') ? channel.empty() : REGISTRATION.out.qc
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
