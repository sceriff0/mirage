/*
 * TILED_REG_TILE - DRAPE fan-out step 2/4: one tile's residual (the little-process fan-out).
 *
 * One task per tile: rigid-warps this tile's reference-frame read box of the moving DAPI and
 * measures a grid of window vectors on the slide-global lattice (drape.vector_grid: node k at
 * W/2 + k*stride, W = 2*stride; the tile emits the nodes whose centre lies in its CORE, so
 * every node is measured exactly once across tiles). `row` is a tile-plan CSV row from
 * TILED_COARSE; its core columns x0/y0/x1/y1 are what makes the ownership exact.
 */
process TILED_REG_TILE {
    tag "${meta.patient_id}:${row.ix}_${row.iy}"
    label 'process_low'

    container 'bolt3x/mirage-tiled:1.0.0'

    input:
    tuple val(meta), path(m0), path(reference, stageAs: 'ref/*'), path(moving, stageAs: 'mov/*'), val(row)

    output:
    tuple val(meta), path("*_ctrl.json"), emit: control
    path "versions.yml"                 , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix     = "${meta.patient_id}_${meta.channels.join('_')}_${row.ix}_${row.iy}"
    // Resolve the nuclear/fiducial channel the transform is estimated from, the same
    // way SEGMENT's CellSAM backend does (lib/SegBackends.groovy): from THIS slide's
    // channel metadata against params.nuclear_markers. params.reg_tiled_nuclear_index
    // is an explicit override, not the source of truth -- the old fixed param restated
    // an invariant CONVERT_IMAGE already guarantees, and named it after one marker, so
    // a CELLTOX panel was read through something called "the DAPI index".
    def nuclear_index = params.reg_tiled_nuclear_index != null
        ? params.reg_tiled_nuclear_index
        : MarkerUtils.nuclearIndex(meta.channels ?: [], params.nuclear_markers)
    if (nuclear_index < 0)
        throw new IllegalArgumentException(
            "${task.process}: no nuclear/fiducial channel among ${meta.channels} for " +
            "patient ${meta.patient_id}. Configured nuclear_markers: " +
            "${MarkerUtils.markerList(params.nuclear_markers).join(', ')}. " +
            "Set params.reg_tiled_nuclear_index to override.")
    // NOT tier-owned: the lattice resolution is its own axis (crossed with the tiers in the
    // arms). An itemised params.reg_tiled_stride reference, so only its value enters the hash.
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
        --out ${prefix}_ctrl.json

    ${ProcessEnvelope.versions(task.process, [], task.container)}
    """

    stub:
    // "lattice"/"vectors" are not decoration: drape.solve's SOLVE solves only the vector
    // lattice and REFUSES a control JSON without them, so a stub that omitted them would fail
    // every stub run's TILED_SOLVE. One confident, in-range vector per tile, at node (ix, iy);
    // the per-tile summary keys (dx/dy/tre/error, ref_fg/mov_fg) mirror the real script's.
    // Guarded by tests/test_stub_control_json_contract.py.
    def prefix = "${meta.patient_id}_${meta.channels.join('_')}_${row.ix}_${row.iy}"
    def stride = params.reg_tiled_stride
    def window = 2 * stride
    """
    echo '{"ix":${row.ix},"iy":${row.iy},"cx":${row.cx},"cy":${row.cy},"dx":0,"dy":0,"tre":0,"error":0.0,"ref_fg":0.1,"mov_fg":0.1,"lattice":{"stride":${stride},"window":${window},"origin":${stride}},"vectors":[[${row.ix},${row.iy},${row.cx},${row.cy},0.0,0.0,10.0,2.0,1.0]],"rejected":[]}' > ${prefix}_ctrl.json
    ${ProcessEnvelope.versionsStub(task.process, [], task.container)}
    """
}
