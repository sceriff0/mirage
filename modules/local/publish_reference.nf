process PUBLISH_REFERENCE {
    tag "${meta.id ?: meta.patient_id}"
    label 'process_single'

    publishDir "${params.outdir}/${meta.patient_id}/registered", mode: 'copy'

    input:
    tuple val(meta), path(image)

    output:
    tuple val(meta), path(image), emit: published
    path "versions.yml"          , emit: versions

    script:
    """
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bash: \$(bash --version | head -n1 | sed 's/GNU bash, version //')
    END_VERSIONS
    """

    stub:
    """
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bash: stub
    END_VERSIONS
    """
}
