process SEGMENT {
    tag "${meta.patient_id}"
    label "${params.seg_gpu ? 'gpu' : 'process_high'}"

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:segmentation_gpu' :
        'docker://bolt3x/attend_image_analysis:segmentation_gpu' }"

    publishDir "${params.outdir}/${meta.patient_id}/segmentation", mode: 'copy', overwrite: true

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

    // Quadruple n_tiles on each retry attempt to reduce memory usage
    def n_tiles_y = (params.seg_n_tiles_y ?: 1) * Math.pow(4, task.attempt - 1) as Integer
    def n_tiles_x = (params.seg_n_tiles_x ?: 1) * Math.pow(4, task.attempt - 1) as Integer

    // FIX WARNING #1: Validate DAPI is in channel 0
    def dapi_validation = meta.channels && meta.channels[0]?.toUpperCase() == 'DAPI'
    """
    echo "Sample: ${meta.patient_id}"
    echo "Attempt: ${task.attempt} - Using n_tiles_y=${n_tiles_y}, n_tiles_x=${n_tiles_x}"

    # FIX WARNING #1: Runtime validation that DAPI is in channel 0
    if [ "${meta.channels ? meta.channels[0].toUpperCase() : 'UNKNOWN'}" != "DAPI" ]; then
        echo "❌ ERROR: DAPI must be in channel 0 for segmentation!"
        echo "Found channels: ${meta.channels ? meta.channels.join(', ') : 'Unknown'}"
        echo "Channel 0: ${meta.channels ? meta.channels[0] : 'Unknown'}"
        exit 1
    fi
    echo "✅ Validated: DAPI is in channel 0"

    segment.py \\
        --image ${merged_file} \\
        --output-dir . \\
        --model-dir ${params.segmentation_model_dir} \\
        --model-name ${params.segmentation_model} \\
        --dapi-channel 0 \\
        --n-tiles ${n_tiles_y} ${n_tiles_x} \\
        --expand-distance ${params.seg_expand_distance} \\
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
