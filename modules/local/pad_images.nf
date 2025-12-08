nextflow.enable.dsl = 2

process PAD_IMAGES {
    tag "${preprocessed_file.simpleName}"
    label 'process_high'
    container "${params.container.preprocess}"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/padded", mode: 'copy'

    input:
    tuple path(preprocessed_file), val(max_height), val(max_width)

    output:
    path "${preprocessed_file.simpleName}_padded.ome.tif", emit: padded

    script:
    def pad_mode = params.gpu_reg_pad_mode ?: 'constant'
    """
    pad_image.py \\
        --input ${preprocessed_file} \\
        --output ${preprocessed_file.simpleName}_padded.ome.tif \\
        --target-height ${max_height} \\
        --target-width ${max_width} \\
        --pad-mode ${pad_mode}
    """
}
