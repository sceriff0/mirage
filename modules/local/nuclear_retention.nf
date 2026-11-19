/*
 * NUCLEAR_RETENTION - per-cell nuclear-channel median of one registered MOVING slide
 *
 * CELL_QC turns it into per-round nuclear retention (moving / reference, normalised per
 * round). Reads the nuclear plane straight from the registered slide, outside the
 * per-marker path, so channels_count and SPLIT_CHANNELS are unaffected. Assumes cyclic IF.
 */
process NUCLEAR_RETENTION {
    tag "${meta.patient_id} - ${meta.id}"

    container "bolt3x/mirage-quantify:1.0.0"

    input:
    tuple val(meta), path(image), path(cell_mask), path(nuclei_mask)

    output:
    tuple val(meta), path("${meta.id}_nuclear_retention.csv"), emit: csv
    path "versions.yml"                                       , emit: versions
    path("*.size.csv")                                        , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def nuclei_arg = params.quantify_compartments ? "--nuclei_mask_file ${nuclei_mask}" : ''
    def nuclear_args = "--nuclear-markers ${MarkerUtils.markerList(params.nuclear_markers).join(' ')}"
    // Same rule as SPLIT_CHANNELS: samplesheet channel names win over OME-XML names,
    // so both agree on the nuclear plane when OME names are generic ("Channel_0").
    def channel_args = (meta.channels && meta.channels instanceof List && !meta.channels.isEmpty()) ?
        "--channels ${meta.channels.join(' ')}" : ""
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${image}", "${cell_mask}"], "${meta.id}.NUCLEAR_RETENTION.size.csv")}

    nuclear_retention.py \\
        --image ${image} \\
        --mask_file ${cell_mask} \\
        ${nuclei_arg} \\
        ${nuclear_args} \\
        ${channel_args} \\
        --output ${meta.id}_nuclear_retention.csv

    ${ProcessEnvelope.versions(task.process, ['pandas', 'tifffile'], task.container)}
    """

    stub:
    """
    printf 'label,Nucleus,Cell\\n' > ${meta.id}_nuclear_retention.csv
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${meta.id}.NUCLEAR_RETENTION.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['pandas', 'tifffile'], task.container)}
    """
}
