nextflow.enable.dsl = 2

process CPU_REGISTER {
    tag "${moving.simpleName}"
    label 'cpu_intensive'

    // Dynamic resource allocation based on input file size
    // CPU version uses more cores and memory than GPU version
    // Small: <10 GB, Medium: 10-30 GB, Large: >30 GB
    memory {
        moving.size() < 10.GB  ? '64.GB'  :   // Small images
        moving.size() < 30.GB  ? '128.GB' :   // Medium images
        '256.GB'                               // Large images
    }

    time {
        moving.size() < 10.GB  ? '4.h' :      // Small images - CPU is slower
        moving.size() < 30.GB  ? '8.h' :      // Medium images
        '12.h'                                 // Large images
    }

    cpus {
        moving.size() < 10.GB  ? 2 :         // Small images
        moving.size() < 30.GB  ? 4 :         // Medium images
        8                                     // Large images - utilize all cores
    }

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/registered", mode: 'copy', pattern: "*.ome.tiff"
    publishDir "${params.outdir}/${params.id}/${params.registration_method}/registered_qc", mode: 'copy', pattern: "qc/*"

    input:
    tuple path(reference), path(moving)

    output:
    path "${moving.simpleName}_registered.ome.tiff", emit: registered
    path "qc/*_QC_RGB.tif"                         , emit: qc, optional: true

    script:
    // New separate crop sizes for affine and diffeomorphic stages
    def affine_crop_size = params.cpu_reg_affine_crop_size ?: (params.cpu_reg_crop_size ?: 2000)
    def diffeo_crop_size = params.cpu_reg_diffeo_crop_size ?: (params.cpu_reg_crop_size ?: 2000)
    def overlap_percent = params.cpu_reg_overlap_percent ?: 10.0
    def n_features = params.cpu_reg_n_features ?: 2000
    def n_workers = task.cpus  // Use all allocated CPUs
    def opt_tol = params.cpu_reg_opt_tol ?: 1e-5
    def inv_tol = params.cpu_reg_inv_tol ?: 1e-5

    // Determine resource tier for logging
    def file_size_gb = moving.size() / 1024 / 1024 / 1024
    def resource_tier = file_size_gb < 10 ? "SMALL" : file_size_gb < 30 ? "MEDIUM" : "LARGE"
    def allocated_mem = file_size_gb < 10 ? "64GB" : file_size_gb < 30 ? "128GB" : "256GB"
    def allocated_time = file_size_gb < 10 ? "4h" : file_size_gb < 30 ? "8h" : "12h"
    def allocated_cpus = file_size_gb < 10 ? "16" : file_size_gb < 30 ? "32" : "48"
    """
    echo "=================================================="
    echo "CPU Registration - Dynamic Resource Allocation"
    echo "=================================================="
    echo "Input file: ${moving.simpleName}"
    echo "File size: ${file_size_gb} GB"
    echo "Resource tier: ${resource_tier}"
    echo "Allocated memory: ${allocated_mem}"
    echo "Allocated time: ${allocated_time}"
    echo "Allocated CPUs: ${allocated_cpus}"
    echo "Worker threads: ${n_workers}"
    echo "Affine crop size: ${affine_crop_size}"
    echo "Diffeo crop size: ${diffeo_crop_size}"
    echo "=================================================="
    echo ""

    mkdir -p qc

    register_cpu.py \\
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
