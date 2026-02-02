nextflow.enable.dsl = 2

/*
========================================================================================
    VALIS REGISTRATION ADAPTER
========================================================================================
    Adapter that converts patient-grouped data to VALIS batch format and back.

    VALIS requires all images for a patient at once to build optimal transformation graph.
    This adapter handles the batch conversion while maintaining the standard interface.

    Input:  ch_grouped_meta - Channel of [patient_id, reference_item, all_items]
            where reference_item = [meta, file] for the reference image
            and all_items = [[meta1, file1], [meta2, file2], ...] for all images
    Output: Channel of [meta, file] tuples (standard format)
========================================================================================
*/

include { REGISTER } from '../../../modules/local/register'

workflow VALIS_ADAPTER {
    take:
    ch_grouped_meta   // Channel of [patient_id, reference_item, all_items] from grouping

    main:
    // ========================================================================
    // CONVERT TO VALIS BATCH FORMAT
    // ========================================================================
    // VALIS needs: [patient_id, reference_file, [all_files], [all_metas]]

    ch_valis_input = ch_grouped_meta
        .map { patient_id, ref_item, all_items ->
            def ref_file = ref_item[1]

            // CRITICAL: VALIS needs ALL images including reference for batch registration
            // We pass reference both separately (for --reference flag) AND in all_files
            // The REGISTER process uses stageAs to avoid filename collision
            def all_files = all_items.collect { item -> item[1] }
            def all_metas = all_items.collect { item -> item[0] }

            tuple(patient_id, ref_file, all_files, all_metas)
        }

    // ========================================================================
    // RUN VALIS BATCH REGISTRATION
    // ========================================================================

    REGISTER(ch_valis_input)

    // ========================================================================
    // CONVERT BACK TO STANDARD FORMAT
    // ========================================================================
    // VALIS outputs: [patient_id, [registered_files], [metas]]
    // Need to convert to: [meta, file]
    //
    // KISS PRINCIPLE: VALIS outputs are sorted by filename, so we sort inputs
    // the same way to maintain 1:1 correspondence

    ch_registered = REGISTER.out.registered
        .flatMap { patient_id, reg_files, metas ->
            // Sanity check: file count must match metadata count
            if (reg_files.size() != metas.size()) {
                def error_msg = """
                ❌ VALIS adapter: File count mismatch for patient ${patient_id}
                📍 Expected ${metas.size()} files but got ${reg_files.size()}
                📋 Metadata entries: ${metas.collect { it.channels.join('_') }.join(', ')}
                📋 Files: ${reg_files.collect { it.name }.join(', ')}
                """.stripIndent()
                throw new Exception(error_msg)
            }

            // CRITICAL: preprocess.py may reorder channel names in filenames
            // Build lookup by marker SET (order-independent)
            def marker_set_to_meta = metas.collectEntries { meta ->
                // Create sorted marker signature for matching
                // IMPORTANT: Use toSorted() instead of sort() to avoid mutating meta.channels
                def marker_key = meta.channels.toSorted().join('_').toLowerCase()
                [(marker_key): meta]
            }

            // Match each registered file by marker set
            reg_files.collect { reg_file ->
                // Files pass through: input -> PREPROCESS (_corrected) -> REGISTER (_registered)
                // Strip suffixes: {patient_id}_{MARKERS}_corrected_registered.ome.tiff -> {patient_id}_{MARKERS}
                def basename = reg_file.name
                    .replaceAll(/_(corrected_)?registered\.ome\.tiff?$/, '')

                // Split into patient_id and markers
                def parts = basename.split('_')
                def file_patient = parts[0]
                def file_markers = parts.drop(1)  // All parts after patient_id

                // Validate patient ID (use toString().trim() to handle GString/String and whitespace issues)
                if (file_patient.toString().trim() != patient_id.toString().trim()) {
                    def error_msg = """
                    ❌ VALIS adapter: Patient ID mismatch
                    📍 Expected: '${patient_id}' (length: ${patient_id.toString().length()})
                    📍 Got: '${file_patient}' (length: ${file_patient.toString().length()})
                    📍 File: ${reg_file.name}
                    """.stripIndent()
                    throw new Exception(error_msg)
                }

                // Create marker signature (sorted, lowercase)
                // Use toSorted() to avoid mutating the file_markers list
                def file_marker_key = file_markers.toSorted().join('_').toLowerCase()

                // Lookup metadata by marker set
                def matched_meta = marker_set_to_meta[file_marker_key]

                if (!matched_meta) {
                    def error_msg = """
                    ❌ VALIS adapter: Could not match metadata for file ${reg_file.name}
                    📍 Patient: ${patient_id}
                    📍 Extracted basename: ${basename}
                    📍 File markers (sorted): ${file_marker_key}
                    📍 Available marker sets: ${marker_set_to_meta.keySet().join(', ')}
                    💡 This suggests a mismatch between input metadata and file outputs
                    """.stripIndent()
                    throw new Exception(error_msg)
                }

                [matched_meta, reg_file]
            }
        }

    // Collect size logs
    ch_size_logs = REGISTER.out.size_log

    emit:
    registered = ch_registered
    size_logs = ch_size_logs
    // QC generation is now decoupled - handled by GENERATE_REGISTRATION_QC module
}
