nextflow.enable.dsl = 2

process CPU_REGISTER_MULTIRES {
    tag "${moving.simpleName}"
    label 'cpu_intensive'

    // Dynamic resource allocation based on input file size
    // Multi-resolution requires more memory due to multiple stages
    // Small: <10 GB, Medium: 10-30 GB, Large: >30 GB
    memory {
        moving.size() < 10.GB  ? '120.GB'  :   // Small images
        moving.size() < 25.GB  ? '240.GB' :   // Medium images
        '360.GB'                               // Large images
    }

    time {
        moving.size() < 10.GB  ? '10.h' :      // Small images - multi-res takes longer
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
    path "qc/*_QC_RGB.{png,tif}"                   , emit: qc, optional: true

    script:
    // Coarse affine parameters (rough global alignment)
    def coarse_crop_size = params.cpu_multires_coarse_crop_size ?: 4000
    def coarse_overlap_percent = params.cpu_multires_coarse_overlap_percent ?: 10.0
    def coarse_n_features = params.cpu_multires_coarse_n_features ?: 1000

    // Fine affine parameters (precise global alignment)
    def fine_crop_size = params.cpu_multires_fine_crop_size ?: 2500
    def fine_overlap_percent = params.cpu_multires_fine_overlap_percent ?: 10.0
    def fine_n_features = params.cpu_multires_fine_n_features ?: 5000

    // Diffeomorphic parameters (local deformations)
    def diffeo_crop_size = params.cpu_multires_diffeo_crop_size ?: 2000
    def diffeo_overlap_percent = params.cpu_multires_diffeo_overlap_percent ?: 15.0
    def diffeo_sigma_diff = params.cpu_multires_diffeo_sigma_diff ?: 20
    def diffeo_radius = params.cpu_multires_diffeo_radius ?: 20
    def opt_tol = params.cpu_multires_opt_tol ?: 1e-6
    def inv_tol = params.cpu_multires_inv_tol ?: 1e-6

    // General parameters
    def n_workers = task.cpus  // Use all allocated CPUs

    // Determine resource tier for logging
    def file_size_gb = moving.size() / 1024 / 1024 / 1024
    def resource_tier = file_size_gb < 10 ? "SMALL" : file_size_gb < 30 ? "MEDIUM" : "LARGE"
    """
    echo "========================================================================"
    echo "CPU Multi-Resolution Registration - Dynamic Resource Allocation"
    echo "========================================================================"
    echo "Input file: ${moving.simpleName}"
    echo "File size: ${file_size_gb} GB"
    echo "Resource tier: ${resource_tier}"
    echo "Allocated CPUs: ${task.cpus}"
    echo "Worker threads: ${n_workers}"
    echo ""
    echo "Multi-Resolution Pipeline:"
    echo "  1. Coarse Affine:  crop_size=${coarse_crop_size}, features=${coarse_n_features}"
    echo "  2. Fine Affine:    crop_size=${fine_crop_size}, features=${fine_n_features}"
    echo "  3. Diffeomorphic:  crop_size=${diffeo_crop_size}, sigma=${diffeo_sigma_diff}, radius=${diffeo_radius}"
    echo "  Optimization tol:  opt=${opt_tol}, inv=${inv_tol}"
    echo "========================================================================"
    echo ""

    mkdir -p qc

    register_cpu_multires.py \\
        --reference ${reference} \\
        --moving ${moving} \\
        --output ${moving.simpleName}_registered.ome.tiff \\
        --qc-dir qc \\
        --coarse-crop-size ${coarse_crop_size} \\
        --coarse-overlap-percent ${coarse_overlap_percent} \\
        --coarse-n-features ${coarse_n_features} \\
        --fine-crop-size ${fine_crop_size} \\
        --fine-overlap-percent ${fine_overlap_percent} \\
        --fine-n-features ${fine_n_features} \\
        --diffeo-crop-size ${diffeo_crop_size} \\
        --diffeo-overlap-percent ${diffeo_overlap_percent} \\
        --diffeo-sigma-diff ${diffeo_sigma_diff} \\
        --diffeo-radius ${diffeo_radius} \\
        --opt-tol ${opt_tol} \\
        --inv-tol ${inv_tol} \\
        --n-workers ${n_workers}
    """
}
