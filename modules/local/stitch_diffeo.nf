/*
 * STITCH_DIFFEO - Assemble diffeo tiles into final registered image
 *
 * Collects all diffeomorphic tiles and stitches them using hard-cutoff placement.
 * Outputs the final OME-TIFF with proper metadata.
 *
 * Memory-efficient: Uses memmap for streaming output.
 *
 * Input: Tile plan, original moving image (for metadata), and all diffeo tile .npy files
 * Output: Final registered OME-TIFF
 */
process STITCH_DIFFEO {
    tag "${meta.patient_id}"
    label 'process_high'

    container 'docker://bolt3x/attend_image_analysis:debug_diffeo'

    publishDir "${params.outdir}/${meta.patient_id}/registered", mode: params.publish_dir_mode

    input:
    tuple val(meta), path(tile_plan), path(moving), path("tiles/*")

    output:
    tuple val(meta), path("*_registered.ome.tiff"), emit: registered
    path "versions.yml"                           , emit: versions
    path("*.size.csv")                            , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${moving.simpleName}"
    """
    # Log input size for tracing (sum of tiles + moving, -L follows symlinks)
    tiles_bytes=\$(du -sLb tiles/ | cut -f1)
    moving_bytes=\$(stat -L --printf="%s" ${moving})
    total_bytes=\$((tiles_bytes + moving_bytes))
    echo "${task.process},${meta.patient_id},tiles/+${moving.name},\${total_bytes}" > ${meta.patient_id}.STITCH_DIFFEO.size.csv

    echo "=================================================="
    echo "STITCH_DIFFEO"
    echo "=================================================="
    echo "Patient: ${meta.patient_id}"
    echo "Tile plan: ${tile_plan}"
    echo "Moving (for metadata): ${moving}"
    echo "Tiles directory: tiles/"
    echo "=================================================="

    # List tiles for debugging
    echo "Tiles found:"
    ls -la tiles/ | head -20

    stitch_tiles.py \\
        --tile-plan ${tile_plan} \\
        --tiles-dir tiles/ \\
        --stage diffeo \\
        --output ${prefix}_registered.ome.tiff \\
        --moving ${moving} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)" 2>/dev/null || echo "unknown")
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${moving.simpleName}"
    """
    touch ${prefix}_registered.ome.tiff
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}.STITCH_DIFFEO.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        tifffile: stub
        numpy: stub
    END_VERSIONS
    """
}
