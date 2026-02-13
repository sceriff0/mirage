/*
 * AFFINE_TILE - Process a single affine tile
 *
 * Computes and applies affine transformation for one tile region.
 * Uses ORB feature matching to compute the transformation matrix.
 *
 * Memory-efficient: Only loads one tile region at a time.
 *
 * Input: Tile ID, tile plan, reference and moving images
 * Output: Transformed tile as .npy file
 */
process AFFINE_TILE {
    tag "${meta.patient_id}_${tile_id}"
    label 'process_medium'

    container 'docker://bolt3x/attend_image_analysis:debug_diffeo'

    input:
    tuple val(meta), val(tile_id), path(tile_plan), path(reference), path(moving)

    output:
    tuple val(meta), path("${tile_id}.npy")      , emit: tile
    tuple val(meta), path("${tile_id}_meta.json"), emit: tile_meta
    path "versions.yml"                          , emit: versions
    path("*.size.csv")                           , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def n_features = params.cpu_reg_n_features ?: 5000
    """
    # Log input sizes for tracing (sum of reference + moving, -L follows symlinks)
    ref_bytes=\$(stat -L --printf="%s" ${reference} 2>/dev/null || echo 0)
    mov_bytes=\$(stat -L --printf="%s" ${moving} 2>/dev/null || echo 0)
    total_bytes=\$((ref_bytes + mov_bytes))
    echo "${task.process},${meta.patient_id}_${tile_id},${reference.name}+${moving.name},\${total_bytes}" > ${meta.patient_id}_${tile_id}.AFFINE_TILE.size.csv

    echo "=================================================="
    echo "AFFINE_TILE: ${tile_id}"
    echo "=================================================="
    echo "Patient: ${meta.patient_id}"
    echo "Tile ID: ${tile_id}"
    echo "N features: ${n_features}"
    echo "=================================================="

    affine_tile.py \\
        --tile-id ${tile_id} \\
        --tile-plan ${tile_plan} \\
        --reference ${reference} \\
        --moving ${moving} \\
        --output-prefix ${tile_id} \\
        --n-features ${n_features} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        opencv: \$(python -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "unknown")
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    """
    echo '{}' > ${tile_id}_meta.json
    python -c "import numpy as np; np.save('${tile_id}.npy', np.zeros((1, 100, 100), dtype=np.float32))"
    echo "STUB,${meta.patient_id}_${tile_id},stub,0" > ${meta.patient_id}_${tile_id}.AFFINE_TILE.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        opencv: stub
        numpy: stub
    END_VERSIONS
    """
}
