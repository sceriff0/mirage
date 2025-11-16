process SEGMENT {
    tag "${merged_file.simpleName}"
    label "${params.seg_gpu ? 'gpu' : 'process_high'}"
    container "${params.container.segmentation}"

    input:
    path merged_file

    output:
    path "segmentation/${merged_file.simpleName}_segmentation.tif", emit: mask

    script:
    def whole_image_flag = params.seg_whole_image ? '--whole_image' : ''
    """
    mkdir -p segmentation
    python3 scripts/segment.py \\
        --input ${merged_file} \\
        --out segmentation/${merged_file.simpleName}_segmentation.tif \\
        --use-gpu ${params.seg_gpu} \\
        --model ${params.seg_model} \\
        --model_dir ${params.segmentation_model_dir} \\
        --crop_size ${params.seg_crop_size} \\
        --overlap ${params.segmentation_overlap} \\
        --gamma ${params.seg_gamma} \\
        ${whole_image_flag}
    """
}
