nextflow.enable.dsl = 2

process CPU_REGISTER_CDM {
    tag "${moving.simpleName}"
    label 'cpu_intensive'

    // Dynamic resource allocation based on input file size
    // CDM requires similar memory to multires
    // Small: <10 GB, Medium: 10-30 GB, Large: >30 GB
    memory {
        moving.size() < 10.GB  ? '120.GB'  :   // Small images
        moving.size() < 25.GB  ? '240.GB' :   // Medium images
        '360.GB'                               // Large images
    }

    time {
        moving.size() < 10.GB  ? '10.h' :      // Small images
        moving.size() < 25.GB  ? '14.h' :      // Medium images
        '20.h'                                 // Large images
    }

    cpus {
        moving.size() < 10.GB  ? 64 :        // Small images
        moving.size() < 25.GB  ? 64 :        // Medium images
        64                                   // Large images
    }

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/registered", mode: 'copy', pattern: "*.ome.tiff"
    publishDir "${params.outdir}/${params.id}/${params.registration_method}/registered_qc", mode: 'copy', pattern: "qc/*"

    input:
    tuple path(reference), path(moving)

    output:
    path "${moving.simpleName}_registered.ome.tiff", emit: registered
    path "qc/*_QC_RGB.png"                         , emit: qc, optional: true

    script:
    // Coarse affine parameters (rough global alignment)
    def coarse_crop_size = params.cpu_cdm_coarse_crop_size ?: 10000
    def coarse_overlap_percent = params.cpu_cdm_coarse_overlap_percent ?: 10.0
    def coarse_n_features = params.cpu_cdm_coarse_n_features ?: 2000

    // Diffeomorphic parameters (primary non-linear registration)
    def diffeo_crop_size = params.cpu_cdm_diffeo_crop_size ?: 2000
    def diffeo_overlap_percent = params.cpu_cdm_diffeo_overlap_percent ?: 20.0
    def diffeo_sigma_diff = params.cpu_cdm_diffeo_sigma_diff ?: 20
    def diffeo_radius = params.cpu_cdm_diffeo_radius ?: 20
    def diffeo_opt_tol = params.cpu_cdm_diffeo_opt_tol ?: 1e-6
    def diffeo_inv_tol = params.cpu_cdm_diffeo_inv_tol ?: 1e-6

    // Micro affine parameters (fine-tune local alignment)
    def micro_crop_size = params.cpu_cdm_micro_crop_size ?: 1000
    def micro_overlap_percent = params.cpu_cdm_micro_overlap_percent ?: 20.0
    def micro_n_features = params.cpu_cdm_micro_n_features ?: 5000

    // General parameters
    def n_workers = task.cpus  // Use all allocated CPUs

    // Determine resource tier for logging
    def file_size_gb = moving.size() / 1024 / 1024 / 1024
    def resource_tier = file_size_gb < 10 ? "SMALL" : file_size_gb < 30 ? "MEDIUM" : "LARGE"
    """
    echo "========================================================================"
    echo "CPU CDM Registration - Dynamic Resource Allocation"
    echo "========================================================================"
    echo "Input file: ${moving.simpleName}"
    echo "File size: ${file_size_gb} GB"
    echo "Resource tier: ${resource_tier}"
    echo "Allocated CPUs: ${task.cpus}"
    echo "Worker threads: ${n_workers}"
    echo ""
    echo "CDM Pipeline (Coarse-Diffeo-Micro):"
    echo "  1. Coarse Affine:  crop_size=${coarse_crop_size}, features=${coarse_n_features}"
    echo "  2. Diffeomorphic:  crop_size=${diffeo_crop_size}, sigma=${diffeo_sigma_diff}, radius=${diffeo_radius}"
    echo "  3. Micro Affine:   crop_size=${micro_crop_size}, features=${micro_n_features}"
    echo "  Optimization tol:  opt=${diffeo_opt_tol}, inv=${diffeo_inv_tol}"
    echo "========================================================================"
    echo ""

    mkdir -p qc

    register_cpu_cdm.py \\
        --reference ${reference} \\
        --moving ${moving} \\
        --output ${moving.simpleName}_registered.ome.tiff \\
        --qc-dir qc \\
        --coarse-crop-size ${coarse_crop_size} \\
        --coarse-overlap-percent ${coarse_overlap_percent} \\
        --coarse-n-features ${coarse_n_features} \\
        --diffeo-crop-size ${diffeo_crop_size} \\
        --diffeo-overlap-percent ${diffeo_overlap_percent} \\
        --diffeo-sigma-diff ${diffeo_sigma_diff} \\
        --diffeo-radius ${diffeo_radius} \\
        --diffeo-opt-tol ${diffeo_opt_tol} \\
        --diffeo-inv-tol ${diffeo_inv_tol} \\
        --micro-crop-size ${micro_crop_size} \\
        --micro-overlap-percent ${micro_overlap_percent} \\
        --micro-n-features ${micro_n_features} \\
        --n-workers ${n_workers}
    """
}
