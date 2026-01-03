nextflow.enable.dsl = 2

process GET_PREPROCESS_DIR {
    tag "get_preprocess_dir"
    label 'process_single'

    input:
    val files

    output:
    path "${params.outdir}/${params.id}/${params.registration_method}/preprocessed", emit: preprocess_dir
    path "versions.yml"                                                             , emit: versions

    script:
    """
    # This directory exists because all PREPROCESS tasks published to it
    echo "Preprocessed directory is ready: ${params.outdir}/preprocessed"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bash: \$(bash --version | head -n1 | sed 's/GNU bash, version //')
    END_VERSIONS
    """

    stub:
    """
    echo "STUB: Preprocessed directory path"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bash: stub
    END_VERSIONS
    """
}