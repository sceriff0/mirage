process SEGMENT {
    tag "${image_file.simpleName}"
    label "${params.seg_gpu ? 'gpu' : 'process_high'}"
    container "${params.container.segmentation}"

    publishDir "${params.outdir}/segmentation", mode: 'copy'

    input:
    path image_file

    output:
    path "*_nuclei_mask.tif", emit: nuclei_mask
    path "*_cell_mask.tif"  , emit: cell_mask

    script:
    def use_gpu_flag = params.seg_gpu ? '--use-gpu' : ''
    def dapi_channel = params.seg_dapi_channel ?: 0
    def n_tiles = params.seg_n_tiles ?: '16 16'
    def expand_distance = params.seg_expand_distance ?: 10
    def pmin = params.seg_pmin ?: 1.0
    def pmax = params.seg_pmax ?: 99.8
    """
    segment.py \\
        --image ${image_file} \\
        --output-dir . \\
        --model-dir ${params.segmentation_model_dir} \\
        --model-name ${params.segmentation_model} \\
        --dapi-channel ${dapi_channel} \\
        --n-tiles ${n_tiles} \\
        --expand-distance ${expand_distance} \\
        --pmin ${pmin} \\
        --pmax ${pmax} \\
        ${use_gpu_flag}
    """
}
