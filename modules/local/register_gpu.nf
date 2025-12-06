nextflow.enable.dsl = 2

process GPU_REGISTER {
    tag "${moving.simpleName}"
    label 'gpu'
    container "${params.container.register_gpu}"

    // Dynamic resource allocation based on input file size
    // Small: <10 GB, Medium: 10-30 GB, Large: >30 GB
    memory {
        moving.size() < 10.GB  ? '128.GB' :   // Small images
        moving.size() < 30.GB  ? '256.GB' :   // Medium images
        '388.GB'                               // Large images
    }

    time {
        moving.size() < 10.GB  ? '2.h' :      // Small images
        moving.size() < 30.GB  ? '3.h' :      // Medium images
        '6.h'                                  // Large images
    }

    cpus 2
    clusterOptions '--gres=gpu:nvidia_h200:1'

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/registered", mode: 'copy', pattern: "*.ome.tiff"
    publishDir "${params.outdir}/${params.id}/${params.registration_method}/registered_qc", mode: 'copy', pattern: "qc/*"

    input:
    tuple path(reference), path(moving)

    output:
    path "${moving.simpleName}_registered.ome.tiff", emit: registered
    path "qc/*_QC_RGB.tif"                         , emit: qc, optional: true

    script:
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
    def allocated_time = file_size_gb < 10 ? "2h" : file_size_gb < 30 ? "3h" : "4h"
    """
    echo "=================================================="
    echo "GPU Registration - Dynamic Resource Allocation"
    echo "=================================================="
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
        --inv-tol ${inv_tol}
    """
}
