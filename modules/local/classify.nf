nextflow.enable.dsl = 2

process CLASSIFY {
    tag "${meta.patient_id}"
    label "${params.classify_gpu ? 'gpu' : 'process_high'}"

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:deep_cell_types' :
        'docker://bolt3x/attend_image_analysis:deep_cell_types' }"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/classified", mode: 'copy'

    input:
    tuple val(meta), path(merged_image), path(seg_mask)

    output:
    tuple val(meta), path("cell_types.csv"), emit: csv
    path "versions.yml"                     , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    def device = params.classify_gpu ? 'cuda:0' : 'cpu'
    def model = params.classify_model ?: 'deepcell-types_2025-06-09_public-data-only'
    def num_workers = params.classify_num_workers ?: 4
    """
    echo "Sample: ${meta.patient_id}"

    classify.py \\
        --image ${merged_image} \\
        --mask ${seg_mask} \\
        --output cell_types.csv \\
        --pixel-size ${params.pixel_size} \\
        --model ${model} \\
        --device ${device} \\
        --num-workers ${num_workers} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        deepcell: \$(python -c "import deepcell; print(deepcell.__version__)" 2>/dev/null || echo "unknown")
        torch: \$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    touch cell_types.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        deepcell: stub
        torch: stub
    END_VERSIONS
    """
}
