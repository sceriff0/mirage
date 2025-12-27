nextflow.enable.dsl = 2

/*
========================================================================================
    VALIS REGISTRATION ADAPTER
========================================================================================
    Adapter that converts standard [meta, file] format to VALIS batch format and back.

    VALIS requires all images for a patient at once to build optimal transformation graph.
    This adapter handles the batch conversion while maintaining the standard interface.

    Input:  Channel of [meta, file] tuples (standard format)
    Output: Channel of [meta, file] tuples (standard format)
========================================================================================
*/

include { REGISTER } from '../../../modules/local/register'

workflow VALIS_ADAPTER {
    take:
    ch_images         // Channel of [meta, file]
    ch_grouped_meta   // Channel of [patient_id, reference_item, all_items] from grouping

    main:
    // ========================================================================
    // CONVERT TO VALIS BATCH FORMAT
    // ========================================================================
    // VALIS needs: [patient_id, reference_file, [all_files], [all_metas]]

    ch_valis_input = ch_grouped_meta
        .map { patient_id, ref_item, all_items ->
            def ref_file = ref_item[1]
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

    ch_registered = REGISTER.out.registered
        .flatMap { patient_id, reg_files, metas ->
            // FIX BUG #8: More robust filename matching using fuzzy matching
            // Instead of exact string matching, use patient_id and channel presence

            // First, try to match by count (sanity check)
            if (reg_files.size() != metas.size()) {
                def error_msg = """
                ❌ VALIS adapter: File count mismatch for patient ${patient_id}
                📍 Expected ${metas.size()} files but got ${reg_files.size()}
                📋 Metadata entries: ${metas.collect { it.channels.join('_') }.join(', ')}
                📋 Files: ${reg_files.collect { it.name }.join(', ')}
                """.stripIndent()
                throw new Exception(error_msg)
            }

            // Create multiple matching strategies for robustness
            def filename_to_meta = [:]

            metas.each { meta ->
                // Strategy 1: Expected exact prefix
                def exact_prefix = "${meta.patient_id}_${meta.channels.join('_')}"
                filename_to_meta[exact_prefix] = meta

                // Strategy 2: Normalized prefix (handle case variations)
                def normalized_prefix = exact_prefix.toLowerCase()
                filename_to_meta[normalized_prefix] = meta

                // Strategy 3: Just channel signature (fallback)
                def channel_sig = meta.channels.join('_')
                filename_to_meta[channel_sig] = meta
            }

            // Match each registered file to its metadata
            reg_files.collect { reg_file ->
                // Strip VALIS-added suffixes progressively
                def basename = reg_file.name
                    .replaceAll(/_(registered|corrected|padded)+/, '')  // Remove suffixes
                    .replaceAll(/\.ome\.tiff?$/, '')  // Remove .ome.tif[f]
                    .replaceAll(/\.tiff?$/, '')       // Remove .tif[f]

                // Try matching strategies in order of preference
                def matched_meta = filename_to_meta[basename] ?:
                                   filename_to_meta[basename.toLowerCase()] ?:
                                   filename_to_meta.find { k, v -> basename.contains(k) }?.value

                if (!matched_meta) {
                    // Detailed debugging with all attempted strategies
                    def error_msg = """
                    ❌ VALIS adapter: Could not match metadata for file ${reg_file.name}
                    📍 Patient: ${patient_id}
                    📍 Extracted basename: ${basename}
                    📋 Tried exact match: ${filename_to_meta.keySet().find { it == basename }}
                    📋 Tried lowercase: ${filename_to_meta.keySet().find { it == basename.toLowerCase() }}
                    📋 Tried contains: ${filename_to_meta.keySet().find { basename.contains(it) }}
                    📋 Available keys: ${filename_to_meta.keySet().join(', ')}
                    💡 This likely means VALIS changed the filename format
                    💡 Please report this issue with the above debug info
                    """.stripIndent()
                    throw new Exception(error_msg)
                }

                [matched_meta, reg_file]
            }
        }

    emit:
    registered = ch_registered
    // QC generation is now decoupled - handled by GENERATE_REGISTRATION_QC module
}
