nextflow.enable.dsl = 2

process PAD_IMAGES {
    tag "${preprocessed_file.simpleName}"
    label 'process_high'
    container "${params.container.preprocess}"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/padded", mode: 'copy'

    input:
    tuple path(preprocessed_file), path(max_dims_file)

    output:
    path "${preprocessed_file.simpleName}_padded.ome.tif", emit: padded

    script:
    def pad_mode = params.gpu_reg_pad_mode ?: 'constant'
    """
    # Read max dimensions from file
    MAX_HEIGHT=\$(grep MAX_HEIGHT ${max_dims_file} | awk '{print \$2}')
    MAX_WIDTH=\$(grep MAX_WIDTH ${max_dims_file} | awk '{print \$2}')

    pad_image.py \\
        --input ${preprocessed_file} \\
        --output ${preprocessed_file.simpleName}_padded.ome.tif \\
        --target-height \${MAX_HEIGHT} \\
        --target-width \${MAX_WIDTH} \\
        --pad-mode ${pad_mode}
    """
}
