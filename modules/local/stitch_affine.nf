/*
 * STITCH_AFFINE - Assemble affine-transformed tiles into intermediate image
 *
 * Collects all affine tiles and stitches them using hard-cutoff placement.
 * Outputs an intermediate TIFF (not OME) for use by the diffeo stage.
 *
 * Memory-efficient: Uses memmap for streaming output.
 *
 * Input: Tile plan and all affine tile .npy files
 * Output: Stitched affine-transformed image
 */
process STITCH_AFFINE {
    tag "${meta.patient_id}"
    label 'process_high'

    container 'docker://bolt3x/attend_image_analysis:debug_diffeo'

    input:
    tuple val(meta), path(tile_plan), path("tiles/*")

    output:
    tuple val(meta), path("*_affine.tiff"), emit: affine
    path "versions.yml"                   , emit: versions
    path("*.size.csv")                    , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    # Log input size for tracing (sum of all tiles)
    total_bytes=\$(du -sb tiles/ | cut -f1)
    echo "${task.process},${meta.patient_id},tiles/,\${total_bytes}" > ${meta.patient_id}.size.csv

    echo "=================================================="
    echo "STITCH_AFFINE"
    echo "=================================================="
    echo "Patient: ${meta.patient_id}"
    echo "Tile plan: ${tile_plan}"
    echo "Tiles directory: tiles/"
    echo "=================================================="

    # List tiles for debugging
    echo "Tiles found:"
    ls -la tiles/ | head -20

    stitch_tiles.py \\
        --tile-plan ${tile_plan} \\
        --tiles-dir tiles/ \\
        --stage affine \\
        --output ${prefix}_affine.tiff \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)" 2>/dev/null || echo "unknown")
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    touch ${prefix}_affine.tiff
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        tifffile: stub
        numpy: stub
    END_VERSIONS
    """
}
