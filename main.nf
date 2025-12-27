nextflow.enable.dsl = 2

/*
========================================================================================
    PLUGINS
========================================================================================
*/

plugins {
    id 'nf-validation@1.1.3'
}

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
    IMPORT NF-VALIDATION FUNCTIONS
========================================================================================
*/

include { validateParameters; paramsHelp; paramsSummaryLog } from 'plugin/nf-validation'

/*
========================================================================================
    HELPER FUNCTIONS
========================================================================================
*/

// Parse CSV row and create metadata map with channels
// Used for loading from checkpoint CSVs
def parseMetadata(row) {
    // Parse pipe-delimited channels from CSV
    def channels = row.channels.split('\\|').collect { ch -> ch.trim() }

    // Validate DAPI is present (can be in any position - will be moved to channel 0 during conversion)
    def has_dapi = channels.any { ch -> ch.toUpperCase() == 'DAPI' }
    if (!has_dapi) {
        error "DAPI channel not found for ${row.patient_id}. Channels: ${channels}"
    }

    return [
        patient_id: row.patient_id,
        is_reference: row.is_reference.toBoolean(),
        channels: channels  // Keep original order; conversion will place DAPI first
    ]
}

// Validate input CSV has required columns
def validateInputCSV(csv_path, required_cols) {
    def csv_file = new File(csv_path)

    if (!csv_file.exists()) {
        error "Input CSV not found: ${csv_path}"
    }

    csv_file.withReader { reader ->
        def header = reader.readLine()
        if (!header) {
            error "Input CSV is empty: ${csv_path}"
        }

        def columns = header.split(',').collect { it.trim() }

        required_cols.each { col ->
            if (!(col in columns)) {
                error "Input CSV missing required column: ${col}. Found columns: ${columns}"
            }
        }
    }

    return true
}

// Validate parameter values
def validateParameter(param_name, param_value, valid_values) {
    if (!(param_value in valid_values)) {
        error "Invalid ${param_name}: '${param_value}'. Valid values: ${valid_values.join(', ')}"
    }
    return true
}

// Create formatted error message
def pipelineError(step, patient_id, message, hint = null) {
    def error_msg = """
    ❌ Pipeline Error in ${step}
    📍 Patient: ${patient_id}
    💬 Message: ${message}
    ${hint ? "💡 Hint: ${hint}" : ""}
    """.stripIndent()
    return error_msg
}


/*
========================================================================================
    RUN MAIN WORKFLOW
========================================================================================
*/

workflow {

    // ========================================================================
    // PARAMETER VALIDATION
    // ========================================================================

    // Show help if requested
    if (params.help) {
        log.info paramsHelp("nextflow run main.nf --input input.csv --outdir results")
        exit 0
    }

    // Validate parameters against schema
    validateParameters()

    // Log parameter summary
    log.info paramsSummaryLog(workflow)

    // Validate input parameter
    if (!params.input) {
        error "Please provide an input glob pattern or checkpoint CSV with --input"
    }

    // Validate step parameter
    def valid_steps = ['preprocessing', 'registration', 'postprocessing', 'results']
    validateParameter('step', params.step, valid_steps)

    // Validate registration method
    def valid_methods = ['valis', 'gpu', 'cpu']
    validateParameter('registration_method', params.registration_method, valid_methods)

    // Validate input CSV exists and has correct format
    if (params.step == 'preprocessing') {
        validateInputCSV(params.input, ['patient_id', 'path_to_file', 'is_reference', 'channels'])
    } else if (params.step == 'registration') {
        validateInputCSV(params.input, ['patient_id', 'preprocessed_image', 'is_reference', 'channels'])
    } else if (params.step == 'postprocessing') {
        validateInputCSV(params.input, ['patient_id', 'registered_image', 'is_reference', 'channels'])
    } else if (params.step == 'results') {
        validateInputCSV(params.input, ['patient_id', 'is_reference', 'phenotype_csv', 'phenotype_mask', 'phenotype_mapping', 'merged_csv', 'cell_mask'])
    }

    // Parameter compatibility warnings
    if (params.padding && params.registration_method == 'valis') {
        log.warn "⚠️  Padding enabled but VALIS registration selected. Padding will be applied but may not be optimal for VALIS."
    }

    if (params.seg_gpu && !params.slurm_partition && workflow.profile != 'local') {
        log.warn "⚠️  GPU segmentation enabled but no SLURM partition specified. May fail if no GPU available."
    }

    // DRY RUN MODE (if enabled)
    if (params.dry_run) {
        log.info """
        ========================================================================
        🔍 DRY RUN MODE - Validation Only
        ========================================================================
        ✅ Input file exists: ${params.input}
        ✅ Step parameter valid: ${params.step}
        ✅ Registration method valid: ${params.registration_method}
        ✅ Input CSV format validated

        All validations passed. Pipeline would execute normally.
        Set --dry_run false to run the pipeline.
        ========================================================================
        """.stripIndent()
        return
    }

    // ========================================================================
    // STEP: PREPROCESSING
    // ========================================================================
    if (params.step == 'preprocessing') {
        // Start from beginning: Parse input CSV
        // CSV format: patient_id,path_to_file,is_reference,channels
        // Note: channels is pipe-delimited (e.g., "PANCK|SMA|DAPI")
        // DAPI can be in any position - conversion will place it in channel 0
        ch_input = channel.fromPath(params.input, checkIfExists: true)
            .splitCsv(header: true)
            .map { row ->
                def meta = parseMetadata(row)
                return [meta, file(row.path_to_file)]
            }

        PREPROCESSING ( ch_input )
        ch_preprocessed = PREPROCESSING.out.preprocessed

    } else {
        // Skip preprocessing - will load from checkpoint later
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
            .map { row -> [parseMetadata(row), file(row.preprocessed_image)] }
    }

    if (params.step in ['preprocessing', 'registration']) {
        // Pass preprocessed images to registration
        // Registration will handle padding internally for GPU/CPU methods
        REGISTRATION (
            ch_for_registration,
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
        // Use checkpoint CSV from postprocessing - ensures CSV is created
        ch_postprocessing_csv = POSTPROCESSING.out.checkpoint_csv

        // Also keep these for RESULTS (they come from direct outputs, not CSV)
        ch_registered_for_results = ch_registered
        ch_qc_for_results = REGISTRATION.out.qc

    } else if (params.step == 'results') {
        // Load from user-provided checkpoint CSV
        ch_postprocessing_csv = channel.fromPath(params.input, checkIfExists: true)

        ch_registered_for_results = channel.empty()
        ch_qc_for_results = channel.empty()
    }

    // Parse CSV and reconstruct channels when starting from 'results' step
    if (params.step == 'results') {
        // FIX BUG #4: Use multiMap to avoid multiple channel consumption
        // Split CSV into multiple output channels in a single operation
        ch_from_csv = ch_postprocessing_csv
            .splitCsv(header: true)
            .multiMap { row ->
                def meta = [patient_id: row.patient_id, is_reference: row.is_reference.toBoolean()]

                phenotype_csv: [meta, file(row.phenotype_csv)]
                phenotype_mask: [meta, file(row.phenotype_mask)]
                phenotype_mapping: [meta, file(row.phenotype_mapping)]
                merged_csv: [meta, file(row.merged_csv)]
                cell_mask: [meta, file(row.cell_mask)]
            }

        // Assign to individual channels
        ch_phenotype_csv = ch_from_csv.phenotype_csv
        ch_phenotype_mask = ch_from_csv.phenotype_mask
        ch_phenotype_mapping = ch_from_csv.phenotype_mapping
        ch_merged_csv = ch_from_csv.merged_csv
        ch_cell_mask = ch_from_csv.cell_mask
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
