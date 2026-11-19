/*
 * TILED_COARSE - STARE fan-out step 1/4: global rigid anchor (M0) + tile plan (per moving slide).
 */
process TILED_COARSE {
    tag "${meta.patient_id}:${meta.channels.join('_')}"
    label 'process_low'

    container 'bolt3x/mirage-stare:1.0.0'

    input:
    tuple val(meta), path(reference, stageAs: 'ref/*'), path(moving, stageAs: 'mov/*')

    output:
    tuple val(meta), path("*_m0.json")   , emit: m0
    tuple val(meta), path("*_tiles.csv") , emit: tiles
    path "versions.yml"                  , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix     = "${meta.patient_id}_${meta.channels.join('_')}"
    // Nuclear channel from this slide's channel metadata (as SEGMENT's CellSAM backend
    // does); params.reg_tiled_nuclear_index is only an override.
    def nuclear_index = params.reg_tiled_nuclear_index != null
        ? params.reg_tiled_nuclear_index
        : MarkerUtils.nuclearIndex(meta.channels ?: [], params.nuclear_markers)
    if (nuclear_index < 0)
        throw new IllegalArgumentException(
            "${task.process}: no nuclear/fiducial channel among ${meta.channels} for " +
            "patient ${meta.patient_id}. Configured nuclear_markers: " +
            "${MarkerUtils.markerList(params.nuclear_markers).join(', ')}. " +
            "Set params.reg_tiled_nuclear_index to override.")
    // itemised params.x reads, never bare `params`: only these values enter the task hash
    def tile       = params.reg_tiled_tile as int
    def halo       = params.reg_tiled_halo as int
    def max_dim    = params.reg_tiled_coarse_max_dim as int
    """
    tiled_coarse.py \\
        --reference ${reference} \\
        --moving ${moving} \\
        --nuclear-index ${nuclear_index} \\
        --tile ${tile} \\
        --halo ${halo} \\
        --max-dim ${max_dim} \\
        --pixel-size ${params.pixel_size} \\
        --out-m0 ${prefix}_m0.json \\
        --out-tiles ${prefix}_tiles.csv

    ${ProcessEnvelope.versions(task.process, ['skimage', 'scipy'], task.container)}
    """

    stub:
    def prefix = "${meta.patient_id}_${meta.channels.join('_')}"
    """
    echo '{"M0":[[1,0,0],[0,1,0],[0,0,1]],"ref_h":16,"ref_w":16,"ref_name":"ref","coarse_tre":0,"n_inliers":0}' > ${prefix}_m0.json
    printf 'ix,iy,cx,cy,x0,y0,x1,y1,rx0,ry0,rx1,ry1\\n0,0,8,8,0,0,16,16,0,0,16,16\\n' > ${prefix}_tiles.csv
    ${ProcessEnvelope.versionsStub(task.process, ['skimage', 'scipy'], task.container)}
    """
}
