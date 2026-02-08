/*
 * GPU_REGISTER - GPU-accelerated diffeomorphic registration
 *
 * Performs pairwise image registration using GPU-accelerated diffeomorphic
 * transformation. Includes dynamic resource allocation based on input file size
 * and automatic retry with reduced crop sizes on OOM errors.
 *
 * Input: Reference image and moving image to register
 * Output: Registered OME-TIFF aligned to reference coordinate space
 */
process GPU_REGISTER {
    tag "${meta.patient_id}"
    label 'gpu'

    container 'docker://bolt3x/attend_image_analysis:debug_diffeo'

    // Retry memory-related failures with reduced crop sizes
    errorStrategy { task.exitStatus in [137, 139, 140, 143] ? 'retry' : 'finish' }
    maxRetries 3

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

    clusterOptions "--gres=gpu:${params.gpu_type}"

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

    // Reduce crop sizes on retry attempts to ease memory pressure
    def base_affine_crop = params.gpu_reg_affine_crop_size ?: 2000
    def base_diffeo_crop = params.gpu_reg_diffeo_crop_size ?: 2000

    // Reduce by 20% per retry attempt
    def reduction_factor = Math.pow(0.8, task.attempt - 1)
    def affine_crop_size = (base_affine_crop * reduction_factor) as Integer
    def diffeo_crop_size = (base_diffeo_crop * reduction_factor) as Integer

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
    def retry_info = task.attempt > 1 ? " (RETRY #${task.attempt}, crops reduced by ${(int)((1-reduction_factor)*100)}%)" : ""
    """
    # Log input sizes for tracing (sum of reference + moving, -L follows symlinks)
    ref_bytes=\$(stat -L --printf="%s" ${reference})
    mov_bytes=\$(stat -L --printf="%s" ${moving})
    total_bytes=\$((ref_bytes + mov_bytes))
    echo "${task.process},${meta.patient_id},${reference.name}+${moving.name},\${total_bytes}" > ${meta.patient_id}_${moving.simpleName}.GPU_REGISTER.size.csv

    echo "=================================================="
    echo "GPU Registration - Dynamic Resource Allocation${retry_info}"
    echo "=================================================="
    echo "Sample: ${meta.patient_id}"
    echo "Input file: ${moving.simpleName}"
    echo "File size: ${file_size_gb} GB"
    echo "Resource tier: ${resource_tier}"
    echo "Allocated memory: ${allocated_mem}"
    echo "Allocated time: ${allocated_time}"
    echo "GPU: ${params.gpu_type}"
    echo "Attempt: ${task.attempt}/${task.maxRetries + 1}"
    echo "Affine crop size: ${affine_crop_size} (base: ${base_affine_crop})"
    echo "Diffeo crop size: ${diffeo_crop_size} (base: ${base_diffeo_crop})"
    echo "=================================================="
    echo ""

    # Validate GPU availability after scheduler allocation
    echo "Checking GPU availability..."
    if ! nvidia-smi &>/dev/null; then
        echo "ERROR: GPU not available but GPU registration requested"
        echo "Run on a GPU node or use --registration_method cpu"
        exit 1
    fi
    echo "GPU available: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo ""

    register_gpu.py \\
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
        cupy: \$(python -c "import cupy; print(cupy.__version__)" 2>/dev/null || echo "unknown")
        torch: \$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    """
    touch ${moving.simpleName}_registered.ome.tiff
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}_${moving.simpleName}.GPU_REGISTER.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        cupy: stub
        torch: stub
    END_VERSIONS
    """
}
