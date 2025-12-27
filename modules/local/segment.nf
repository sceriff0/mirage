process SEGMENT {
    tag "${meta.patient_id}"
    label "${params.seg_gpu ? 'gpu' : 'process_high'}"

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:segmentation_gpu' :
        'docker://bolt3x/attend_image_analysis:segmentation_gpu' }"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/segmentation", mode: 'copy'

    input:
    tuple val(meta), path(merged_file)

    output:
    tuple val(meta), path("*_nuclei_mask.tif"), emit: nuclei_mask
    tuple val(meta), path("*_cell_mask.tif")  , emit: cell_mask
    path "versions.yml"                        , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    def use_gpu_flag = params.seg_gpu ? '--use-gpu' : ''
    def pmin = params.seg_pmin ?: 1.0
    def pmax = params.seg_pmax ?: 99.8
    """
    echo "Sample: ${meta.patient_id}"

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
        ${use_gpu_flag} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        deepcell: \$(python -c "import deepcell; print(deepcell.__version__)" 2>/dev/null || echo "unknown")
        tensorflow: \$(python -c "import tensorflow; print(tensorflow.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    touch ${prefix}_nuclei_mask.tif
    touch ${prefix}_cell_mask.tif

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        deepcell: stub
        tensorflow: stub
    END_VERSIONS
    """
}
