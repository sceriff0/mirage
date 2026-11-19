/*
 * APPLY_PROFILES - apply BASICPY's illumination profiles and reassemble the slide
 *
 * nf-core's BASICPY only computes profiles (mcmicro applies them inside ASHLAR); this
 * applies corrected = (image - darkfield) / flatfield per channel, per pseudo-FOV,
 * reassembled from TILE_FOR_BASIC's positions sidecar.
 *
 * The nuclear channel is left uncorrected (TILE_FOR_BASIC's decision, from the sidecar),
 * negatives are clipped, and values are rounded before the integer cast.
 * Output <name>_corrected.ome.tif matches what PREPROCESS publishes.
 */
process APPLY_PROFILES {
    tag "${meta.patient_id}"

    container "bolt3x/mirage-preprocess:1.0.0"

    input:
    tuple val(meta), path(ome_tiff), path(sidecar), path(darkfield), path(flatfield)

    output:
    tuple val(meta), path("*_corrected.ome.tif"), emit: preprocessed
    path "versions.yml"                         , emit: versions
    path("*.size.csv")                          , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${ome_tiff}"], "${meta.patient_id}_${ome_tiff.simpleName}.APPLY_PROFILES.size.csv")}

    apply_basic_profiles.py \\
        --image ${ome_tiff} \\
        --sidecar ${sidecar} \\
        --flatfield ${flatfield} \\
        --darkfield ${darkfield} \\
        --output ${ome_tiff.simpleName}_corrected.ome.tif \\
        --pixel-size ${params.pixel_size} \\
        ${args}

    ${ProcessEnvelope.versions(task.process, ['numpy', 'tifffile'], task.container)}
    """

    stub:
    """
    touch ${ome_tiff.simpleName}_corrected.ome.tif
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${meta.patient_id}_${ome_tiff.simpleName}.APPLY_PROFILES.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['numpy', 'tifffile'], task.container)}
    """
}
