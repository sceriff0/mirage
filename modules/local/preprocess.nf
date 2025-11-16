process PREPROCESS {
    tag "${nd2file.simpleName}"
    label 'process_medium'
    container "${params.container.preprocess}"

    input:
    path nd2file

    output:
    path "preprocessed/${nd2file.simpleName}.preproc.ome.tif", emit: preprocessed

    script:
    """
    mkdir -p preprocessed
    preprocess.py \\
        --image ${nd2file} \\
        --fov_size ${params.preproc_tile_size}
    """
}
