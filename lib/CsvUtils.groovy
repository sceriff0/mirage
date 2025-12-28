import nextflow.Channel
import static nextflow.Nextflow.file
import static nextflow.Nextflow.error
import static nextflow.Nextflow.tuple

class CsvUtils {

    static Map validateMetadata(Map meta, String context = 'unknown') {

        if (!meta.patient_id)
            error "Missing patient_id in ${context}"

        if (!(meta.is_reference instanceof Boolean))
            error "is_reference must be boolean in ${context}"

        if (!(meta.channels instanceof List) || meta.channels.isEmpty())
            error "channels must be a non-empty List in ${context}"

        if (meta.channels.any { it == null || it.trim().isEmpty() })
            error "Empty channel name found for patient ${meta.patient_id}"

        if (meta.channels[0].toUpperCase() != 'DAPI')
            error """
            DAPI must be the first channel
            Patient: ${meta.patient_id}
            Channels: ${meta.channels}
            """.stripIndent()

        return meta
    }

    static Map parseMetadata(Map row, String context = 'parseMetadata') {

        def channels = row.channels
            .split('\\|')
            .collect { it.trim() }

        if (!channels.any { it.toUpperCase() == 'DAPI' })
            error "DAPI channel missing for patient ${row.patient_id}"

        def meta = [
            patient_id  : row.patient_id,
            is_reference: row.is_reference.toBoolean(),
            channels    : channels
        ]

        return validateMetadata(meta, "${context} (${row.patient_id})")
    }

    static void validateInputCSV(def csv, List required_cols) {

        def file = new File(csv)
        if (!file.exists())
            error "Input CSV not found: ${csv}"

        def header = file.readLines().first()?.split(',')*.trim()
        if (!header)
            error "CSV is empty: ${csv}"

        required_cols.each {
            if (!(it in header))
                error "CSV missing required column '${it}'"
        }
    }

    static Channel loadInputCSV(def csv_path, String image_column) {

        Channel
            .fromPath(csv_path, checkIfExists: true)
            .splitCsv(header: true)
            .map { row ->
                tuple(
                    parseMetadata(row, "CSV ${csv_path}"),
                    file(row[image_column])
                )
            }
    }
}
