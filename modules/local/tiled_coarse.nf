/*
 * TILED_COARSE - STARE fan-out step 1/4: global rigid anchor (M0) + tile plan (per moving slide).
 */
process TILED_COARSE {
    tag "${meta.patient_id}:${meta.channels.join('_')}"
    label 'process_low'

    container 'bolt3x/mirage-tiled:1.0.0'

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
    // Tier-owned knobs: null means take the value from reg_tiled_mode's row in
    // lib/RegPresets.groovy. Resolved via RegPresets so the tier table has one home;
    // the mode and the override are passed as SCALARS, never the params map, because a
    // script: block that hands `params` to a helper makes Nextflow hash the whole map
    // and re-run the task on any unrelated parameter change (see CLAUDE.md).
    def tile       = RegPresets.stare(params.reg_tiled_mode, 'tile', params.reg_tiled_tile)
    def halo       = RegPresets.stare(params.reg_tiled_mode, 'halo', params.reg_tiled_halo)
    def max_dim    = RegPresets.stare(params.reg_tiled_mode, 'coarse_max_dim', params.reg_tiled_coarse_max_dim)
    """
    tiled_coarse.py \\
        --reference ${reference} \\
        --moving ${moving} \\
        --nuclear-index ${nuclear_index} \\
        --tile ${tile} \\
        --halo ${halo} \\
        --max-dim ${max_dim} \\
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
