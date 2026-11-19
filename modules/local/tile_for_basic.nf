/*
 * TILE_FOR_BASIC - write a stitched slide as a multi-SITE OME-TIFF
 *
 * nf-core's BASICPY refuses a single-sited image ("The image is single sited"), and a
 * mirage slide is one stitched plane per channel. This writes the non-overlapping FOV grid
 * (params.preproc_tile_size) onto the Z axis so the module sees one site per tile. Axes are
 * CZYX because the module fits one profile per channel.
 *
 * The nuclear-channel skip is decided here and recorded in the JSON sidecar for
 * APPLY_PROFILES. The input slide is passed through so APPLY_PROFILES needs no meta-keyed
 * join (two slides of one patient can share an identical meta map).
 */
process TILE_FOR_BASIC {
    tag "${meta.patient_id}"

    container "bolt3x/mirage-preprocess:1.0.0"

    input:
    tuple val(meta), path(ome_tiff)

    output:
    tuple val(meta), path("*_tiles.ome.tif")                    , emit: tiles
    tuple val(meta), path(ome_tiff), path("*_tiles.json")       , emit: sidecar
    path "versions.yml"                                         , emit: versions
    path("*.size.csv")                                          , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def skip_nuclear_flag = params.preproc_skip_nuclear ? '--skip_nuclear' : ''
    // Via MarkerUtils, which normalises the String/comma-joined param shapes.
    def nuclear_args = "--nuclear-markers ${MarkerUtils.markerList(params.nuclear_markers).join(' ')}"
    def channels = meta.channels.join(' ')
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${ome_tiff}"], "${meta.patient_id}_${ome_tiff.simpleName}.TILE_FOR_BASIC.size.csv")}

    tile_for_basic.py \\
        --image ${ome_tiff} \\
        --output ${ome_tiff.simpleName}_tiles.ome.tif \\
        --sidecar ${ome_tiff.simpleName}_tiles.json \\
        --channels ${channels} \\
        --fov_size ${params.preproc_tile_size} \\
        ${skip_nuclear_flag} \\
        ${nuclear_args} \\
        ${args}

    ${ProcessEnvelope.versions(task.process, ['numpy', 'tifffile'], task.container)}
    """

    stub:
    """
    touch ${ome_tiff.simpleName}_tiles.ome.tif
    echo '{"format_version": 1}' > ${ome_tiff.simpleName}_tiles.json
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${meta.patient_id}_${ome_tiff.simpleName}.TILE_FOR_BASIC.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['numpy', 'tifffile'], task.container)}
    """
}
