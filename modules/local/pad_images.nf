process PAD_IMAGES {
    tag "${meta.patient_id}"
    label 'process_medium'

    container 'docker://bolt3x/attend_image_analysis:preprocess'

    input:
    tuple val(meta), path(preprocessed_file), path(max_dims_file)

    output:
    tuple val(meta), path("*_padded.ome.tif"), emit: padded
    path "versions.yml"                       , emit: versions
    path("*.size.csv")                        , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    def pad_mode = params.gpu_reg_pad_mode ?: 'constant'
    """
    # Log input size for tracing
    input_bytes=\$(stat --printf="%s" ${preprocessed_file})
    echo "${task.process},${meta.patient_id},${preprocessed_file.name},\${input_bytes}" > ${meta.patient_id}.PAD_IMAGES.size.csv

    # Read max dimensions from file
    MAX_HEIGHT=\$(grep MAX_HEIGHT ${max_dims_file} | awk '{print \$2}')
    MAX_WIDTH=\$(grep MAX_WIDTH ${max_dims_file} | awk '{print \$2}')

    echo "Padding ${meta.patient_id} to \${MAX_HEIGHT}x\${MAX_WIDTH} with mode: ${pad_mode}"

    pad_image.py \\
        --input ${preprocessed_file} \\
        --output ${preprocessed_file.simpleName}_padded.ome.tif \\
        --target-height \${MAX_HEIGHT} \\
        --target-width \${MAX_WIDTH} \\
        --pad-mode ${pad_mode} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    touch ${preprocessed_file.simpleName}_padded.ome.tif
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}.PAD_IMAGES.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        numpy: stub
    END_VERSIONS
    """
}
