/*
========================================================================================
    MetadataUtils - Utility functions for metadata handling
========================================================================================
*/

class MetadataUtils {

    /**
     * Parse CSV row and create metadata map with channels
     * Used for loading from checkpoint CSVs
     */
    static def parseMetadata(row) {
        // Parse pipe-delimited channels from CSV
        def channels = row.channels.split('\\|').collect { ch -> ch.trim() }

        // Validate DAPI is present (can be in any position - will be moved to channel 0 during conversion)
        def has_dapi = channels.any { ch -> ch.toUpperCase() == 'DAPI' }
        if (!has_dapi) {
            throw new Exception("DAPI channel not found for ${row.patient_id}. Channels: ${channels}")
        }

        return [
            patient_id: row.patient_id,
            is_reference: row.is_reference.toBoolean(),
            channels: channels  // Keep original order; conversion will place DAPI first
        ]
    }

    /**
     * Validate that DAPI is in channel 0
     */
    static def validateDAPIFirst(channels, patient_id) {
        if (channels[0].toUpperCase() != 'DAPI') {
            throw new Exception("CRITICAL: DAPI must be in channel 0 for ${patient_id}! Got: ${channels}")
        }
    }

    /**
     * Validate channel consistency between steps
     */
    static def validateChannels(expected, actual, step_name, patient_id) {
        def expectedSorted = expected.sort()
        def actualSorted = actual.sort()

        if (expectedSorted != actualSorted) {
            throw new Exception("${step_name}: Channel mismatch for ${patient_id}! Expected ${expected}, got ${actual}")
        }
    }
}
