nextflow.enable.dsl = 2

/*
 * GET_PREPROCESS_DIR - Get preprocessed directory path
 *
 * Utility process that returns the path to the preprocessed output directory.
 * Used to signal completion of preprocessing and provide path for downstream steps.
 *
 * Input: Completion signal from preprocessing tasks
 * Output: Path to preprocessed directory
 */
process GET_PREPROCESS_DIR {
    tag "get_preprocess_dir"
    label 'process_single'

    input:
    val files

    output:
    path "${params.outdir}/${params.id}/${params.registration_method}/preprocessed", emit: preprocess_dir
    path "versions.yml"                                                             , emit: versions

    when:
    task.ext.when == null || task.ext.when

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