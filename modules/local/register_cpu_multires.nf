process CPU_REGISTER_MULTIRES {
    tag "${meta.patient_id}"
    label 'process_high'
    label 'process_high_memory'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:debug_diffeo' :
        'docker://bolt3x/attend_image_analysis:debug_diffeo' }"

    // Dynamic resource allocation based on input file size
    // CPU version uses more cores and memory than GPU version
    // Small: <10 GB, Medium: 10-30 GB, Large: >30 GB
    memory {
        def size = moving.size()
        check_max(
            size < 10.GB  ? 100.GB * task.attempt :   // Small images
            size < 25.GB  ? 250.GB * task.attempt :   // Medium images
            400.GB * task.attempt,                    // Large images
            'memory'
        )
    }

    time {
        def size = moving.size()
        check_max(
            size < 10.GB  ? 8.h * task.attempt :      // Small images - CPU is slower
            size < 25.GB  ? 10.h * task.attempt :     // Medium images
            24.h * task.attempt,                      // Large images
            'time'
        )
    }

    cpus {
        def size = moving.size()
        check_max(
            size < 10.GB  ? 64 * task.attempt :       // Small images
            size < 25.GB  ? 64 * task.attempt :       // Medium images
            64 * task.attempt,                        // Large images
            'cpus'
        )
    }

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
    // Multi-resolution parameters: coarse → fine affine → diffeo
    def coarse_crop_size = params.cpu_reg_multires_coarse_crop_size ?: (params.cpu_reg_crop_size ?: 3000)
    def fine_crop_size = params.cpu_reg_multires_fine_crop_size ?: (params.cpu_reg_crop_size ?: 2000)
    def diffeo_crop_size = params.cpu_reg_multires_diffeo_crop_size ?: (params.cpu_reg_crop_size ?: 2000)
    def overlap_percent = params.cpu_reg_multires_overlap_percent ?: (params.cpu_reg_overlap_percent ?: 10.0)
    def n_features = params.cpu_reg_multires_n_features ?: (params.cpu_reg_n_features ?: 2000)
    def n_workers = task.cpus  // Use all allocated CPUs
    def opt_tol = params.cpu_reg_multires_opt_tol ?: (params.cpu_reg_opt_tol ?: 1e-5)
    def inv_tol = params.cpu_reg_multires_inv_tol ?: (params.cpu_reg_inv_tol ?: 1e-5)

    // Determine resource tier for logging
    def file_size_gb = moving.size() / 1024 / 1024 / 1024
    def resource_tier = file_size_gb < 10 ? "SMALL" : file_size_gb < 25 ? "MEDIUM" : "LARGE"
    def allocated_mem = file_size_gb < 10 ? "100GB" : file_size_gb < 25 ? "250GB" : "400GB"
    def allocated_time = file_size_gb < 10 ? "8h" : file_size_gb < 25 ? "10h" : "24h"
    """
    echo "=================================================="
    echo "CPU Registration (Multi-Resolution) - Dynamic Resource Allocation"
    echo "=================================================="
    echo "Sample: ${meta.patient_id}"
    echo "Input file: ${moving.simpleName}"
    echo "File size: ${file_size_gb} GB"
    echo "Resource tier: ${resource_tier}"
    echo "Allocated memory: ${allocated_mem}"
    echo "Allocated time: ${allocated_time}"
    echo "Allocated CPUs: ${task.cpus}"
    echo "Worker threads: ${n_workers}"
    echo "Coarse crop size: ${coarse_crop_size}"
    echo "Fine crop size: ${fine_crop_size}"
    echo "Diffeo crop size: ${diffeo_crop_size}"
    echo "=================================================="
    echo ""

    mkdir -p qc

    register_cpu_multires.py \\
        --reference ${reference} \\
        --moving ${moving} \\
        --output ${moving.simpleName}_registered.ome.tiff \\
        --qc-dir qc \\
        --coarse-crop-size ${coarse_crop_size} \\
        --fine-crop-size ${fine_crop_size} \\
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
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
        scipy: \$(python -c "import scipy; print(scipy.__version__)" 2>/dev/null || echo "unknown")
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
        numpy: stub
        scipy: stub
    END_VERSIONS
    """
}
