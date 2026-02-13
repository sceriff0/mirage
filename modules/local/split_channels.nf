nextflow.enable.dsl = 2

/*
 * SPLIT_CHANNELS - Split multi-channel TIFF into individual channels
 *
 * Extracts individual channel images from multi-channel OME-TIFFs for
 * per-channel processing. Handles DAPI extraction from reference image only.
 *
 * Input: Registered multi-channel OME-TIFF and reference flag
 * Output: Individual single-channel TIFF files per marker
 */
process SPLIT_CHANNELS {
    tag "${meta.patient_id}"
    label 'process_medium'

    container 'docker://bolt3x/attend_image_analysis:preprocess'

    //publishDir "${params.outdir}/${meta.patient_id}/channels", mode: 'copy'

    input:
    tuple val(meta), path(registered_image), val(is_reference)

    output:
    tuple val(meta), path("*.tiff", includeInputs: false), emit: channels
    path "channel_names.txt"                              , emit: channel_manifest, optional: true
    path "versions.yml"                                   , emit: versions
    path("*.size.csv")                                    , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def ref_flag = is_reference ? "--is-reference" : ""
    // Pass channel names from metadata if available and valid
    def channel_args = (meta.channels && meta.channels instanceof List && !meta.channels.isEmpty()) ?
        "--channels ${meta.channels.join(' ')}" : ""
    """
    # Log input size for tracing (-L follows symlinks)
    input_bytes=\$(stat -L --printf="%s" ${registered_image} 2>/dev/null || echo 0)
    echo "${task.process},${meta.patient_id},${registered_image.name},\${input_bytes}" > ${meta.patient_id}_${registered_image.simpleName}.SPLIT_CHANNELS.size.csv

    echo "Sample: ${meta.patient_id}"
    echo "Channels: ${(meta.channels && meta.channels instanceof List) ? meta.channels.join(', ') : 'Will read from OME metadata'}"

    split_multichannel.py \\
        ${registered_image} \\
        . \\
        ${ref_flag} \\
        ${channel_args} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)" 2>/dev/null || echo "unknown")
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    """
    # Create stub channel files based on metadata
    touch DAPI.tiff
    ${meta.channels && meta.channels.size() > 1 ?
        meta.channels.drop(1).collect { "touch ${it}.tiff" }.join('\n    ') :
        'touch Marker1.tiff'}

    ls -1 *.tiff | sort > channel_names.txt
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}_${registered_image.simpleName}.SPLIT_CHANNELS.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        tifffile: stub
        numpy: stub
    END_VERSIONS
    """
}
