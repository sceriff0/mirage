/*
 * TILED_REG_TILE - STARE fan-out step 2/4: one tile's residual (the little-process fan-out).
 *
 * One task per tile: rigid-warps the tile's read box of the moving nuclear channel and
 * measures window vectors on the slide-global lattice (node k at W/2 + k*stride,
 * W = 2*stride). A tile emits only nodes whose centre lies in its core (x0/y0/x1/y1 of
 * the TILED_COARSE plan row), so each node is measured exactly once.
 */
process TILED_REG_TILE {
    tag "${meta.patient_id}:${row.ix}_${row.iy}"
    label 'process_low'

    container 'bolt3x/mirage-stare:1.0.0'

    input:
    tuple val(meta), path(m0), path(reference, stageAs: 'ref/*'), path(moving, stageAs: 'mov/*'), val(row)

    output:
    tuple val(meta), path("*_ctrl.json"), emit: control
    path "versions.yml"                 , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix     = "${meta.patient_id}_${meta.channels.join('_')}_${row.ix}_${row.iy}"
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
    // itemised params.x read: only this value enters the task hash
    def stride     = params.reg_tiled_stride
    """
    tiled_reg_tile.py \\
        --reference ${reference} \\
        --moving ${moving} \\
        --m0 ${m0} \\
        --nuclear-index ${nuclear_index} \\
        --ix ${row.ix} --iy ${row.iy} --cx ${row.cx} --cy ${row.cy} \\
        --rx0 ${row.rx0} --ry0 ${row.ry0} --rx1 ${row.rx1} --ry1 ${row.ry1} \\
        --x0 ${row.x0} --y0 ${row.y0} --x1 ${row.x1} --y1 ${row.y1} \\
        --stride ${stride} \\
        --pixel-size ${params.pixel_size} \\
        --out ${prefix}_ctrl.json

    ${ProcessEnvelope.versions(task.process, [], task.container)}
    """

    stub:
    // TILED_SOLVE refuses a control JSON without "lattice"/"vectors"; one in-range vector
    // per tile, with the real script's summary keys.
    def prefix = "${meta.patient_id}_${meta.channels.join('_')}_${row.ix}_${row.iy}"
    def stride = params.reg_tiled_stride
    def window = 2 * stride
    """
    echo '{"ix":${row.ix},"iy":${row.iy},"cx":${row.cx},"cy":${row.cy},"dx":0,"dy":0,"tre":0,"error":0.0,"ref_fg":0.1,"mov_fg":0.1,"lattice":{"stride":${stride},"window":${window},"origin":${stride}},"vectors":[[${row.ix},${row.iy},${row.cx},${row.cy},0.0,0.0,10.0,2.0,1.0]],"rejected":[]}' > ${prefix}_ctrl.json
    ${ProcessEnvelope.versionsStub(task.process, [], task.container)}
    """
}
