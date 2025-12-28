def validateMeta(meta, context = 'unknown') {

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

def parseMetadata(row, context = 'parseMetadata') {

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

    return validateMeta(meta, "${context} (${row.patient_id})")
}
