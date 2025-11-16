process SEGMENT {
    tag "${merged_file.simpleName}"
    label "${params.seg_gpu ? 'gpu' : 'process_high'}"
    container "${params.container.segmentation}"

    input:
    path merged_file

    output:
    path "segmentation/${merged_file.simpleName}_segmentation.tif", emit: mask
    path "versions.yml"                                           , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    mkdir -p segmentation
    python3 scripts/segment.py \\
        --input ${merged_file} \\
        --out segmentation/${merged_file.simpleName}_segmentation.tif \\
        --use-gpu ${params.seg_gpu} \\
        --model ${params.seg_model} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //g')
    END_VERSIONS
    """

    stub:
    """
    mkdir -p segmentation
    touch segmentation/${merged_file.simpleName}_segmentation.tif
    touch versions.yml
    """
}
