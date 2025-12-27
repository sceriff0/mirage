nextflow.enable.dsl = 2

process SPLIT_CHANNELS {
    tag "${meta.patient_id}"
    label 'process_medium'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:preprocess' :
        'docker://bolt3x/attend_image_analysis:preprocess' }"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/channels", mode: 'copy'

    input:
    tuple val(meta), path(registered_image), val(is_reference)

    output:
    tuple val(meta), path("*.tiff"), emit: channels
    path "versions.yml"             , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    def ref_flag = is_reference ? "--is-reference" : ""
    // FIX BUG #5: Add defensive null check for meta.channels
    // Pass channel names from metadata if available and valid
    def channel_args = (meta.channels && meta.channels instanceof List && !meta.channels.isEmpty()) ?
        "--channels ${meta.channels.join(' ')}" : ""
    """
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
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    touch ${prefix}_channel_0.tiff

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        tifffile: stub
        numpy: stub
    END_VERSIONS
    """
}
