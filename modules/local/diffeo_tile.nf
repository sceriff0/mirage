/*
 * DIFFEO_TILE - Process a single diffeomorphic tile
 *
 * Computes and applies diffeomorphic deformation for one tile region.
 * Uses DIPY's symmetric diffeomorphic registration.
 *
 * Memory-efficient: Only loads one tile region at a time.
 *
 * Input: Tile ID, tile plan, reference and affine-transformed images
 * Output: Registered tile as .npy file
 */
process DIFFEO_TILE {
    tag "${meta.patient_id}_${tile_id}"
    label 'process_low'

    container 'docker://bolt3x/attend_image_analysis:debug_diffeo'

    input:
    tuple val(meta), val(tile_id), path(tile_plan), path(reference), path(affine_image)

    output:
    tuple val(meta), path("${tile_id}.npy")      , emit: tile
    tuple val(meta), path("${tile_id}_meta.json"), emit: tile_meta
    path "versions.yml"                          , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def opt_tol = params.cpu_reg_opt_tol ?: 1e-5
    def inv_tol = params.cpu_reg_inv_tol ?: 1e-5
    """
    echo "=================================================="
    echo "DIFFEO_TILE: ${tile_id}"
    echo "=================================================="
    echo "Patient: ${meta.patient_id}"
    echo "Tile ID: ${tile_id}"
    echo "Opt tolerance: ${opt_tol}"
    echo "Inv tolerance: ${inv_tol}"
    echo "=================================================="

    diffeo_tile.py \\
        --tile-id ${tile_id} \\
        --tile-plan ${tile_plan} \\
        --reference ${reference} \\
        --affine ${affine_image} \\
        --output-prefix ${tile_id} \\
        --opt-tol ${opt_tol} \\
        --inv-tol ${inv_tol} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        dipy: \$(python -c "import dipy; print(dipy.__version__)" 2>/dev/null || echo "unknown")
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    """
    echo '{}' > ${tile_id}_meta.json
    python -c "import numpy as np; np.save('${tile_id}.npy', np.zeros((1, 100, 100), dtype=np.float32))"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        dipy: stub
        numpy: stub
    END_VERSIONS
    """
}
