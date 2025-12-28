nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    CONVERSION MODULE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Creates a pyramidal OME-TIFF combining registered images with segmentation
    and phenotype masks for efficient visualization.
----------------------------------------------------------------------------------------
*/

process CONVERSION {
    tag "${meta.patient_id}"
    label 'process_high'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:preprocess' :
        'docker://bolt3x/attend_image_analysis:preprocess' }"

    publishDir "${params.outdir}/${meta.patient_id}/pyramid", mode: 'copy'

    input:
    tuple val(meta), path(merged_image)

    output:
    tuple val(meta), path("pyramid.ome.tiff"), emit: pyramid
    path "versions.yml"                       , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    echo "Sample: ${meta.patient_id}"

    bfconvert \\
        -noflat \\
        -bigtiff \\
        -tilex ${params.tilex} \\
        -tiley ${params.tiley} \\
        -pyramid-resolutions ${params.pyramid_resolutions} \\
        -pyramid-scale ${params.pyramid_scale} \\
        ${args} \\
        "${merged_image}" \\
        "pyramid.ome.tiff"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bfconvert: \$(bfconvert -version 2>&1 | head -n1 || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    touch pyramid.ome.tiff

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bfconvert: stub
    END_VERSIONS
    """
}
