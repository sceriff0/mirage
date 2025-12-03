nextflow.enable.dsl = 2

process PREPROCESS {
    tag "${ome_tiff.simpleName}"
    label 'process_medium'
    container "${params.container.preprocess}"

    publishDir "${params.outdir}/${params.id}/preprocessed", mode: 'copy'

    input:
    path ome_tiff

    output:
    path "${ome_tiff.simpleName}_corrected.ome.tif", emit: preprocessed

    script:
    def skip_dapi_flag = params.preproc_skip_dapi ? '--skip_dapi' : ''
    def autotune_flag = params.preproc_autotune ? '--autotune' : ''
    def no_darkfield_flag = params.preproc_no_darkfield ? '--no_darkfield' : ''
    def overlap = params.preproc_overlap ?: 0
    """
    preprocess.py \\
        --image ${ome_tiff} \\
        --output_dir . \\
        --fov_size ${params.preproc_tile_size} \\
        --n_workers ${params.preproc_pool_workers} \\
        --n_iter ${params.preproc_n_iter} \\
        --overlap ${overlap} \\
        ${skip_dapi_flag} \\
        ${autotune_flag} \\
        ${no_darkfield_flag}
    """
}