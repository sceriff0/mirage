process PREPROCESS {
    tag "${nd2file.simpleName}"
    label 'process_medium'
    container "${params.container.preprocess}"

    input:
    path nd2file

    output:
    path "preprocessed/${nd2file.simpleName}.preproc.ome.tif", emit: preprocessed
    path "versions.yml"                                       , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    mkdir -p preprocessed
    python3 scripts/preprocess.py \\
        --image ${nd2file} \\
        --fov_size ${params.preproc_tile_size} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //g')
    END_VERSIONS
    """

    stub:
    """
    mkdir -p preprocessed
    touch preprocessed/${nd2file.simpleName}.preproc.ome.tif
    touch versions.yml
    """
}
