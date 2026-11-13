/*
 * TILED_STITCH - STARE fan-out step 4/4: warp the moving slide through the manifest.
 *
 * Applies M0 + mesh to every channel of the moving slide (bilinear, non-negative), in row strips
 * so peak memory is a strip. Writes the registered OME-TIFF in the reference frame.
 */
process TILED_STITCH {
    tag "${meta.patient_id}:${meta.channels.join('_')}"
    label 'process_medium'

    container 'bolt3x/mirage-stare:1.0.0'

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
    // A plain param with a nextflow.config default (STARE has no cost tiers since 2026-11-04);
    // an itemised `params.x` read, so only this value enters the task hash.
    def out_tile  = params.reg_tiled_out_tile as int
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
