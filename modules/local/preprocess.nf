nextflow.enable.dsl = 2

process PREPROCESS {
    tag "${ome_tiff.simpleName}"
    label 'process_medium'
    container "${params.container.preprocess}"

    publishDir "${params.outdir}/preprocessed", mode: 'copy'

    input:
    path ome_tiff

    output:
    path "${ome_tiff.simpleName}_corrected.ome.tif", emit: preprocessed

    script:
    """
    preprocess.py \\
        --image ${ome_tiff} \\
        --output_dir . \\
        --fov_size ${params.preproc_tile_size}
    """
}