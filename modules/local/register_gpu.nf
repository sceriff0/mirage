process GPU_REGISTER {
    tag "${meta.patient_id}"
    label 'gpu'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:debug_diffeo' :
        'docker://bolt3x/attend_image_analysis:debug_diffeo' }"

    // Dynamic resource allocation based on input file size
    // Small: <10 GB, Medium: 10-30 GB, Large: >30 GB
    memory {
        def size = moving.size()
        check_max(
            size < 10.GB ? 128.GB * task.attempt :   // Small images
            size < 30.GB ? 256.GB * task.attempt :   // Medium images
            388.GB * task.attempt,                   // Large images
            'memory'
        )
    }

    time {
        def size = moving.size()
        check_max(
            size < 10.GB ? 2.h * task.attempt :      // Small images
            size < 30.GB ? 3.h * task.attempt :      // Medium images
            6.h * task.attempt,                      // Large images
            'time'
        )
    }

    cpus { check_max( 2 * task.attempt, 'cpus' ) }
    clusterOptions '--gres=gpu:nvidia_h200:1'

    input:
    tuple val(meta), path(reference), path(moving)

    output:
    tuple val(meta), path("*_registered.ome.tiff"), emit: registered
    tuple val(meta), path("qc/*_QC_RGB.{png,tif}") , emit: qc, optional: true
    path "versions.yml"                            , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    // New separate crop sizes for affine and diffeomorphic stages
    def affine_crop_size = params.gpu_reg_affine_crop_size ?: (params.gpu_reg_crop_size ?: 2000)
    def diffeo_crop_size = params.gpu_reg_diffeo_crop_size ?: (params.gpu_reg_crop_size ?: 2000)
    def overlap_percent = params.gpu_reg_overlap_percent ?: 10.0
    def n_features = params.gpu_reg_n_features ?: 2000
    def n_workers = params.gpu_reg_n_workers ?: 4
    def opt_tol = params.gpu_reg_opt_tol ?: 1e-5
    def inv_tol = params.gpu_reg_inv_tol ?: 1e-5

    // Determine resource tier for logging
    def file_size_gb = moving.size() / 1024 / 1024 / 1024
    def resource_tier = file_size_gb < 10 ? "SMALL" : file_size_gb < 30 ? "MEDIUM" : "LARGE"
    def allocated_mem = file_size_gb < 10 ? "128GB" : file_size_gb < 30 ? "256GB" : "388GB"
    def allocated_time = file_size_gb < 10 ? "2h" : file_size_gb < 30 ? "3h" : "6h"
    """
    echo "=================================================="
    echo "GPU Registration - Dynamic Resource Allocation"
    echo "=================================================="
    echo "Sample: ${meta.patient_id}"
    echo "Input file: ${moving.simpleName}"
    echo "File size: ${file_size_gb} GB"
    echo "Resource tier: ${resource_tier}"
    echo "Allocated memory: ${allocated_mem}"
    echo "Allocated time: ${allocated_time}"
    echo "GPU: nvidia_h200:1"
    echo "Affine crop size: ${affine_crop_size}"
    echo "Diffeo crop size: ${diffeo_crop_size}"
    echo "=================================================="
    echo ""

    mkdir -p qc

    register_gpu.py \\
        --reference ${reference} \\
        --moving ${moving} \\
        --output ${moving.simpleName}_registered.ome.tiff \\
        --qc-dir qc \\
        --affine-crop-size ${affine_crop_size} \\
        --diffeo-crop-size ${diffeo_crop_size} \\
        --overlap-percent ${overlap_percent} \\
        --n-features ${n_features} \\
        --n-workers ${n_workers} \\
        --opt-tol ${opt_tol} \\
        --inv-tol ${inv_tol} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        cupy: \$(python -c "import cupy; print(cupy.__version__)" 2>/dev/null || echo "unknown")
        torch: \$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    mkdir -p qc
    touch ${moving.simpleName}_registered.ome.tiff
    touch qc/${moving.simpleName}_QC_RGB.png

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        cupy: stub
        torch: stub
    END_VERSIONS
    """
}
