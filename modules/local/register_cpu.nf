/*
 * CPU_REGISTER - CPU multi-threaded diffeomorphic registration
 *
 * Performs pairwise image registration using multi-threaded CPU computation.
 * Alternative to GPU registration for environments without GPU access.
 * Uses dynamic resource allocation based on input file size.
 *
 * Input: Reference image and moving image to register
 * Output: Registered OME-TIFF aligned to reference coordinate space
 */
process CPU_REGISTER {
    tag "${meta.patient_id}"
    label 'process_high_memory'

    container 'docker://bolt3x/attend_image_analysis:debug_diffeo'

    input:
    tuple val(meta), path(reference), path(moving)

    output:
    tuple val(meta), path("*_registered.ome.tiff"), emit: registered
    path "versions.yml"                            , emit: versions
    path("*.size.csv")                             , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def affine_crop_size = params.cpu_reg_affine_crop_size ?: 2000
    def diffeo_crop_size = params.cpu_reg_diffeo_crop_size ?: 2000
    def overlap_percent = params.cpu_reg_overlap_percent ?: 10.0
    def n_features = params.cpu_reg_n_features ?: 2000
    def n_workers = task.cpus  // Use all allocated CPUs
    def opt_tol = params.cpu_reg_opt_tol ?: 1e-5
    def inv_tol = params.cpu_reg_inv_tol ?: 1e-5

    // Determine resource tier for logging
    def file_size_gb = moving.size() / 1024 / 1024 / 1024
    def resource_tier = file_size_gb < 10 ? "SMALL" : file_size_gb < 25 ? "MEDIUM" : "LARGE"
    def allocated_mem = file_size_gb < 10 ? "100GB" : file_size_gb < 25 ? "250GB" : "400GB"
    def allocated_time = file_size_gb < 10 ? "8h" : file_size_gb < 25 ? "10h" : "24h"
    """
    # Log input sizes for tracing (sum of reference + moving, -L follows symlinks)
    ref_bytes=\$(stat -L --printf="%s" ${reference} 2>/dev/null || echo 0)
    mov_bytes=\$(stat -L --printf="%s" ${moving} 2>/dev/null || echo 0)
    total_bytes=\$((ref_bytes + mov_bytes))
    echo "${task.process},${meta.patient_id},${reference.name}+${moving.name},\${total_bytes}" > ${meta.patient_id}_${moving.simpleName}.CPU_REGISTER.size.csv

    echo "=================================================="
    echo "CPU Registration - Dynamic Resource Allocation"
    echo "=================================================="
    echo "Sample: ${meta.patient_id}"
    echo "Input file: ${moving.simpleName}"
    echo "File size: ${file_size_gb} GB"
    echo "Resource tier: ${resource_tier}"
    echo "Allocated memory: ${allocated_mem}"
    echo "Allocated time: ${allocated_time}"
    echo "Allocated CPUs: ${task.cpus}"
    echo "Worker threads: ${n_workers}"
    echo "Affine crop size: ${affine_crop_size}"
    echo "Diffeo crop size: ${diffeo_crop_size}"
    echo "=================================================="
    echo ""

    register_cpu.py \\
        --reference ${reference} \\
        --moving ${moving} \\
        --output ${moving.simpleName}_registered.ome.tiff \\
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
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
        scipy: \$(python -c "import scipy; print(scipy.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    """
    touch ${moving.simpleName}_registered.ome.tiff
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}_${moving.simpleName}.CPU_REGISTER.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        numpy: stub
        scipy: stub
    END_VERSIONS
    """
}
