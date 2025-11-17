process SEGMENT {
    tag "${merged_file.simpleName}"
    label "${params.seg_gpu ? 'gpu' : 'process_high'}"
    container "${params.container.segmentation}"

    input:
    path merged_file

    output:
    path "segmentation/${merged_file.simpleName}_segmentation.tif", emit: mask

    script:
    def use_gpu_flag = params.seg_gpu ? '--use_gpu' : ''
    def whole_image_flag = params.seg_whole_image ? '--whole_image' : ''
    def tophat_radius = params.seg_tophat_radius ?: 50
    def gaussian_sigma = params.seg_gaussian_sigma ?: 1.0
    def pmin = params.seg_pmin ?: 1.0
    def pmax = params.seg_pmax ?: 99.8
    """
    mkdir -p segmentation
    segment.py \\
        --dapi_file ${merged_file} \\
        --output_dir segmentation \\
        --model_dir ${params.segmentation_model_dir} \\
        --model_name ${params.segmentation_model} \\
        --crop_size ${params.seg_crop_size} \\
        --overlap ${params.segmentation_overlap} \\
        --gamma ${params.seg_gamma} \\
        --tophat_radius ${tophat_radius} \\
        --gaussian_sigma ${gaussian_sigma} \\
        --pmin ${pmin} \\
        --pmax ${pmax} \\
        ${use_gpu_flag} \\
        ${whole_image_flag}
    """
}
