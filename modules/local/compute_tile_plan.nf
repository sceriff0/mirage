/*
 * COMPUTE_TILE_PLAN - Generate tile plan for tiled CPU registration
 *
 * Reads image metadata (without loading data) and generates a JSON tile plan
 * containing coordinates for both affine and diffeomorphic registration stages.
 *
 * This is a lightweight process that only reads TIFF headers.
 *
 * Input: Reference and moving image paths
 * Output: JSON tile plan + original image paths (passed through)
 */
process COMPUTE_TILE_PLAN {
    tag "${meta.patient_id}"
    label 'process_single'

    container 'docker://bolt3x/attend_image_analysis:debug_diffeo'

    input:
    tuple val(meta), path(reference), path(moving)

    output:
    tuple val(meta), path("tile_plan.json"), path(reference), path(moving), emit: plan
    path "versions.yml"                                                   , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def affine_crop_size = params.cpu_reg_affine_crop_size ?: 10000
    def diffeo_crop_size = params.cpu_reg_diffeo_crop_size ?: 2000
    def overlap_percent = params.cpu_reg_overlap_percent ?: 40.0
    """
    echo "=================================================="
    echo "COMPUTE_TILE_PLAN"
    echo "=================================================="
    echo "Patient: ${meta.patient_id}"
    echo "Reference: ${reference}"
    echo "Moving: ${moving}"
    echo "Affine crop size: ${affine_crop_size}"
    echo "Diffeo crop size: ${diffeo_crop_size}"
    echo "Overlap percent: ${overlap_percent}%"
    echo "=================================================="

    compute_tile_plan.py \\
        --reference ${reference} \\
        --moving ${moving} \\
        --output tile_plan.json \\
        --affine-crop-size ${affine_crop_size} \\
        --diffeo-crop-size ${diffeo_crop_size} \\
        --overlap-percent ${overlap_percent} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    """
    echo '{"version": "1.0", "affine_tiles": [], "diffeo_tiles": []}' > tile_plan.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        tifffile: stub
    END_VERSIONS
    """
}
