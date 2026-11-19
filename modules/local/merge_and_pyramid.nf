/*
 * MERGE_AND_PYRAMID - merge single-channel TIFFs into a pyramidal OME-TIFF for QuPath.
 *
 * Writes a pyramidal OME-TIFF directly (no bfconvert), preserving channel names, colors, and
 * pixel sizes in the OME-XML. Intensity channels only; cell objects are delivered separately via
 * cells.geojson. Processes large images memory-efficiently.
 */

process MERGE_AND_PYRAMID {
    tag "${meta.patient_id}"

    container "bolt3x/mirage-merge:1.0.0"

    input:
    tuple val(meta), path(split_channels, stageAs: 'channels/*'), path(mask_files, stageAs: 'masks/*')

    output:
    tuple val(meta), path("pyramid.ome.tiff"), emit: pyramid
    path "versions.yml", emit: versions
    path("*.size.csv") , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''

    def pixel_size_x = params.pixel_size
    def pixel_size_y = params.pixel_size
    def pyramid_resolutions = params.pyramid_resolutions
    def pyramid_scale = params.pyramid_scale
    def tile_size = params.tilex
    def compression = params.compression

    // Masks go in as a second, single-resolution uint32 OME series only when staged;
    // assemble_export.nf decides whether to stage them (empty list = no masks/ dir).
    def masks_arg = mask_files ? "--masks-dir masks" : ""

    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ['channels/*'], "${meta.patient_id}.MERGE_AND_PYRAMID.size.csv")}

    echo "Sample: ${meta.patient_id}"
    echo "Input directory: channels/"
    ls -lh channels/

    merge_channels_pyramid.py \\
        --input-dir channels \\
        --output pyramid.ome.tiff \\
        --physical-size-x ${pixel_size_x} \\
        --physical-size-y ${pixel_size_y} \\
        --pyramid-resolutions ${pyramid_resolutions} \\
        --pyramid-scale ${pyramid_scale} \\
        --tile-size ${tile_size} \\
        --compression ${compression} \\
        ${masks_arg} \\
        ${args}

    ${ProcessEnvelope.versions(task.process, ['tifffile', 'numpy'], task.container)}
    """

    stub:
    """
    touch pyramid.ome.tiff
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${meta.patient_id}.MERGE_AND_PYRAMID.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['tifffile', 'numpy'], task.container)}
    """
}
