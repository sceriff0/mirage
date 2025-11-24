nextflow.enable.dsl = 2

process PAD_IMAGES {
    tag "pad_all_images"
    label 'process_medium'
    container "${params.container.preprocess}"

    publishDir "${params.outdir}/padded", mode: 'copy'

    input:
    path preprocessed_files

    output:
    path "padded/*.ome.tif*", emit: padded

    script:
    def pad_mode = params.gpu_reg_pad_mode ?: 'constant'
    """
    mkdir -p padded

    pad_images.py \\
        --input ${preprocessed_files} \\
        --output-dir padded \\
        --pad-mode ${pad_mode}
    """
}
