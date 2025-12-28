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

include { PREPROCESSING  } from './subworkflows/local/preprocess.nf'
include { REGISTRATION   } from './subworkflows/local/registration.nf'
include { POSTPROCESSING } from './subworkflows/local/postprocess.nf'
include { RESULTS        } from './subworkflows/local/results.nf'

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

// Validate metadata map has required fields and correct types
// This prevents cryptic downstream errors from malformed metadata
def validateMeta(meta, required_fields = ['patient_id', 'is_reference', 'channels'], context = 'unknown') {
    // Check all required fields exist
    required_fields.each { field ->
        if (!meta.containsKey(field)) {
            error """
            Missing required metadata field '${field}' in ${context}
            Patient ID: ${meta.patient_id ?: 'unknown'}
            Available fields: ${meta.keySet().join(', ')}
            """.stripIndent()
        }
    }

    // Validate specific field types and values
    if (required_fields.contains('patient_id')) {
        if (!meta.patient_id || meta.patient_id.toString().trim().isEmpty()) {
            error "Invalid patient_id in ${context}: '${meta.patient_id}'"
        }
    }

    if (required_fields.contains('is_reference')) {
        if (!(meta.is_reference instanceof Boolean)) {
            error "is_reference must be boolean in ${context}. Got: ${meta.is_reference} (${meta.is_reference.class})"
        }
    }

    if (required_fields.contains('channels')) {
        // Validate channels is a non-empty list
        if (!(meta.channels instanceof List)) {
            error """
            channels must be a List in ${context}
            Patient: ${meta.patient_id}
            Got type: ${meta.channels?.class}
            Got value: ${meta.channels}
            """.stripIndent()
        }

        if (meta.channels.isEmpty()) {
            error """
            Empty channels list in ${context}
            Patient: ${meta.patient_id}
            Hint: Check your input CSV 'channels' column is not empty
            """.stripIndent()
        }

        // Validate no empty channel names
        def empty_channels = meta.channels.findIndexValues { it == null || it.toString().trim().isEmpty() }
        if (!empty_channels.isEmpty()) {
            error """
            Empty channel name(s) found in ${context}
            Patient: ${meta.patient_id}
            Channels: ${meta.channels}
            Empty at indices: ${empty_channels}
            """.stripIndent()
        }

        // Validate DAPI is first (critical requirement)
        if (meta.channels[0]?.toUpperCase() != 'DAPI') {
            error """
            DAPI must be the first channel in ${context}
            Patient: ${meta.patient_id}
            Current order: ${meta.channels}
            DAPI position: ${meta.channels.findIndexOf { it?.toUpperCase() == 'DAPI' }}

            Fix: Update your input CSV so DAPI is listed first in the 'channels' column
            Example: 'DAPI|PANCK|SMA' instead of 'PANCK|DAPI|SMA'
            """.stripIndent()
        }
    }

    return true
}

// Parse CSV row and create metadata map with channels
// Used for loading from checkpoint CSVs
def parseMetadata(row) {
    // Parse pipe-delimited channels from CSV
    def channels = row.channels.split('\\|').collect { ch -> ch.trim() }

    // Validate channels are not empty after trimming
    if (channels.isEmpty()) {
        error """
        Empty channels list after parsing for patient ${row.patient_id}
        Raw channels value: '${row.channels}'
        Check your CSV has valid pipe-delimited channel names (e.g., 'DAPI|PANCK|SMA')
        """.stripIndent()
    }

    // Check for any empty strings in channels
    def empty_indices = channels.findIndexValues { it.isEmpty() }
    if (!empty_indices.isEmpty()) {
        error """
        Empty channel name(s) found for patient ${row.patient_id}
        Channels: ${channels}
        Empty at positions: ${empty_indices}
        Raw value: '${row.channels}'
        Check for double pipes (||) or trailing pipes in your CSV
        """.stripIndent()
    }

    // Validate DAPI is present (can be in any position - will be moved to channel 0 during conversion)
    def has_dapi = channels.any { ch -> ch.toUpperCase() == 'DAPI' }
    if (!has_dapi) {
        error """
        DAPI channel not found for patient ${row.patient_id}
        Channels found: ${channels}
        DAPI is required for segmentation and must be present in your channel list
        """.stripIndent()
    }

    def meta = [
        patient_id: row.patient_id,
        is_reference: row.is_reference.toBoolean(),
        channels: channels  // Keep original order; conversion will place DAPI first
    ]

    // Validate the constructed metadata
    validateMeta(meta, ['patient_id', 'is_reference', 'channels'], "parseMetadata for ${row.patient_id}")

    return meta
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
    Pipeline Error in ${step}
    Patient: ${patient_id}
    Message: ${message}
    ${hint ? "Hint: ${hint}" : ""}
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
        // Validate base columns (registered_images is optional for backward compatibility)
        validateInputCSV(params.input, ['patient_id', 'is_reference', 'phenotype_csv', 'phenotype_mask', 'phenotype_mapping', 'merged_csv', 'cell_mask'])
    }

    // Parameter compatibility warnings
    if (params.padding && params.registration_method == 'valis') {
        log.warn "Padding enabled but VALIS registration selected. Padding will be applied but may not be optimal for VALIS."
    }

    if (params.seg_gpu && !params.slurm_partition && workflow.profile != 'local') {
        log.warn "GPU segmentation enabled but no SLURM partition specified. May fail if no GPU available."
    }

    // DRY RUN MODE (if enabled)
    if (params.dry_run) {
        log.info """
        ========================================================================
        DRY RUN MODE - Validation Only
        ========================================================================
        Input file exists: ${params.input}
        Step parameter valid: ${params.step}
        Registration method valid: ${params.registration_method}
        Input CSV format validated

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

        // QC is not regenerated when starting from results step
        ch_qc_for_results = channel.empty()
        // Note: ch_registered_for_results will be populated from CSV below
    }

    // Parse CSV and reconstruct channels when starting from 'results' step
    if (params.step == 'results') {
        // FIX ISSUE #4: Refactored channel reconstruction for clarity
        // Strategy: Use multiMap to split CSV into multiple channels in one pass
        // This avoids repeated CSV parsing and potential channel consumption issues

        ch_from_csv = ch_postprocessing_csv
            .splitCsv(header: true)
            .multiMap { row ->
                // Create minimal metadata (no channels needed at this stage)
                def meta = [patient_id: row.patient_id, is_reference: row.is_reference.toBoolean()]

                // Map each CSV column to its own output channel
                phenotype_csv: [meta, file(row.phenotype_csv)]
                phenotype_mask: [meta, file(row.phenotype_mask)]
                phenotype_mapping: [meta, file(row.phenotype_mapping)]
                merged_csv: [meta, file(row.merged_csv)]
                cell_mask: [meta, file(row.cell_mask)]

                // Handle registered_images (optional for backward compatibility)
                registered_patient: [row.patient_id, row.containsKey('registered_images') ? row.registered_images : null]
            }

        // Assign simple outputs
        ch_phenotype_csv = ch_from_csv.phenotype_csv
        ch_phenotype_mask = ch_from_csv.phenotype_mask
        ch_phenotype_mapping = ch_from_csv.phenotype_mapping
        ch_merged_csv = ch_from_csv.merged_csv
        ch_cell_mask = ch_from_csv.cell_mask

        // Reconstruct registered images channel (complex multi-file per patient)
        // Filter, parse, and flatten in explicit steps for clarity
        ch_registered_for_results = ch_from_csv.registered_patient
            .filter { _patient_id, registered_images_str ->
                // Only process patients that have registered_images column with data
                registered_images_str != null && !registered_images_str.trim().isEmpty()
            }
            .flatMap { patient_id, registered_images_str ->
                // Parse pipe-delimited paths and create [meta, file] tuples
                def meta = [patient_id: patient_id, is_reference: false]  // Will be updated per file
                registered_images_str.split('\\|').collect { path ->
                    [meta.clone(), file(path.trim())]
                }
            }

        // Backward compatibility check: warn if any patients missing registered_images
        ch_from_csv.registered_patient
            .map { patient_id, registered_images_str ->
                [patient_id, registered_images_str != null && !registered_images_str.trim().isEmpty()]
            }
            .collect()
            .subscribe { patient_reg_status ->
                def patients_with_images = patient_reg_status.findAll { _patient_id, has_images -> has_images }
                def patients_without = patient_reg_status.findAll { _patient_id, has_images -> !has_images }

                if (!patients_without.isEmpty()) {
                    log.warn """
                    ⚠️  WARNING: Old checkpoint CSV format detected!

                    ${patients_without.size()} patient(s) are missing 'registered_images' column:
                    ${patients_without.collect { entry -> entry[0] }.join(', ')}

                    Pyramid generation will be SKIPPED for these patients.

                    💡 To enable pyramid regeneration:
                       1. Re-run from 'postprocessing' step to generate new checkpoint CSV
                       2. OR manually add 'registered_images' column with pipe-delimited paths

                    ℹ️  Only CSVs and masks will be saved to archive (no pyramidal OME-TIFF).
                    """.stripIndent()
                }

                if (!patients_with_images.isEmpty()) {
                    log.info "✅ Found registered_images for ${patients_with_images.size()} patient(s)"
                }
            }
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
