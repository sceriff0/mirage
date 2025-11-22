nextflow.enable.dsl = 2

process CLASSIFY {
    tag "classify"
    label "${params.classify_gpu ? 'gpu' : 'process_high'}"
    container "${params.container.classify}"

    publishDir "${params.outdir}/classified", mode: 'copy'

    input:
    path merged_image
    path seg_mask

    output:
    path "cell_types.csv", emit: csv

    script:
    def device = params.classify_gpu ? 'cuda:0' : 'cpu'
    def model = params.classify_model ?: 'deepcell-types_2025-06-09_public-data-only'
    def num_workers = params.classify_num_workers ?: 4
    """
    classify.py \\
        --image ${merged_image} \\
        --mask ${seg_mask} \\
        --output cell_types.csv \\
        --pixel-size ${params.pixel_size} \\
        --model ${model} \\
        --device ${device} \\
        --num-workers ${num_workers}
    """
}
