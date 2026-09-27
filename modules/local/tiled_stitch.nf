/*
 * TILED_STITCH - DRAPE fan-out step 4/4: warp the moving slide through the manifest.
 *
 * Applies M0 + mesh to every channel of the moving slide (bilinear, non-negative), in row strips
 * so peak memory is a strip. Writes the registered OME-TIFF in the reference frame.
 */
process TILED_STITCH {
    tag "${meta.patient_id}:${meta.channels.join('_')}"
    label 'process_medium'

    container 'bolt3x/mirage-drape:1.0.0'

    input:
    tuple val(meta), path(manifest), path(moving, stageAs: 'mov/*')

    output:
    tuple val(meta), path("registered/*_registered.ome.tiff"), emit: registered
    path "versions.yml"                                      , emit: versions
    path "*.size.csv"                                        , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix    = "${meta.patient_id}_${meta.channels.join('_')}"
    def slidename = meta.channels.join('_')
    // Tier-owned knobs: null means take the value from reg_tiled_mode's row in
    // lib/RegPresets.groovy. Resolved via RegPresets so the tier table has one home;
    // the mode and the override are passed as SCALARS, never the params map, because a
    // script: block that hands `params` to a helper makes Nextflow hash the whole map
    // and re-run the task on any unrelated parameter change (see CLAUDE.md).
    def out_tile  = RegPresets.drape(params.reg_tiled_mode, 'out_tile', params.reg_tiled_out_tile)
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${moving}"], "${prefix}.TILED_STITCH.size.csv")}

    mkdir -p registered
    tiled_stitch.py \\
        --moving ${moving} \\
        --manifest ${manifest} \\
        --moving-name '${slidename}' \\
        --out-tile ${out_tile} \\
        --pixel-size ${params.pixel_size} \\
        --channel-names ${meta.channels.join(' ')} \\
        --out registered/${prefix}_registered.ome.tiff

    ${ProcessEnvelope.versions(task.process, ['tifffile'], task.container)}
    """

    stub:
    def prefix = "${meta.patient_id}_${meta.channels.join('_')}"
    """
    mkdir -p registered
    touch registered/${prefix}_registered.ome.tiff
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${prefix}.TILED_STITCH.size.csv")}
    ${ProcessEnvelope.versionsStub(task.process, ['tifffile'], task.container)}
    """
}
