process SEGMENT {
    tag "${merged_file.simpleName}"
    label "${params.seg_gpu ? 'gpu' : 'process_high'}"
    container "${params.container.segmentation}"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/segmentation", mode: 'copy'

    input:
    path merged_file

    output:
    path "*_nuclei_mask.tif", emit: nuclei_mask
    path "*_cell_mask.tif"  , emit: cell_mask

    script:
    def use_gpu_flag = params.seg_gpu ? '--use-gpu' : ''
    def pmin = params.seg_pmin ?: 1.0
    def pmax = params.seg_pmax ?: 99.8
    """
    segment.py \\
        --image ${merged_file} \\
        --output-dir . \\
        --model-dir ${params.segmentation_model_dir} \\
        --model-name ${params.segmentation_model} \\
        --dapi-channel 0 \\
        --n-tiles ${params.seg_n_tiles_y ?: 16} ${params.seg_n_tiles_x ?: 16} \\
        --expand-distance ${params.seg_expand_distance ?: 10} \\
        --pmin ${pmin} \\
        --pmax ${pmax} \\
        ${use_gpu_flag}
    """
}
