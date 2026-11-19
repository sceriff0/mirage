/*
 * QUANTIFY - per-cell intensity statistics of one single-channel TIFF under the
 * segmentation mask; one CSV per channel.
 */
process QUANTIFY {
    tag "${meta.patient_id} - ${meta.channel_name}"

    container "bolt3x/mirage-quantify:1.0.0"

    input:
    tuple val(meta), path(channel_tiff), path(cell_mask), path(nuclei_mask)

    output:
    tuple val(meta), path("${meta.id}_quant.csv"), emit: individual_csv
    path "versions.yml"                           , emit: versions
    path("*.size.csv")                            , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def channel_name = meta.channel_name
    // The nuclear mask path exists only here, so this toggle cannot live in ext.args.
    def nuclei_arg = params.quantify_compartments ? "--nuclei_mask_file ${nuclei_mask}" : ''
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${channel_tiff}", "${cell_mask}"], "${meta.id}.QUANTIFY.size.csv")}

    echo "Sample: ${meta.patient_id}"
    echo "Channel: ${channel_name}"

    # Run quantification on this single channel TIFF
    quantify.py \\
        --channel_tiff ${channel_tiff} \\
        --channel-name '${channel_name}' \\
        --mask_file ${cell_mask} \\
        ${nuclei_arg} \\
        --outdir . \\
        --output_file ${meta.id}_quant.csv \\
        ${args}

    ${ProcessEnvelope.versions(task.process, ['pandas', 'skimage'], task.container)}
    """

    stub:
    """
    touch ${meta.id}_quant.csv
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${meta.id}.QUANTIFY.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['pandas', 'skimage'], task.container)}
    """
}
