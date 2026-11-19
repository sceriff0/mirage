/*
 * SPLIT_CHANNELS - split a registered multi-channel OME-TIFF into one TIFF per marker
 *
 * Which channels are emitted is decided at samplesheet read
 * (CsvUtils.resolveKeptChannelsPerSlide) and arrives as meta.keep_channels.
 */
process SPLIT_CHANNELS {
    tag "${meta.patient_id}"

    container "bolt3x/mirage-preprocess:1.0.0"

    input:
    tuple val(meta), path(registered_image), val(is_reference)

    output:
    tuple val(meta), path("*.tiff", includeInputs: false), emit: channels
    path "versions.yml"                                   , emit: versions
    path("*.size.csv")                                    , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def ref_flag = is_reference ? "--is-reference" : ""
    def channel_args = (meta.channels && meta.channels instanceof List && !meta.channels.isEmpty()) ?
        "--channels ${meta.channels.join(' ')}" : ""
    // The keep-set from meta is emitted as-is, so countChannelsPerPatient's group size
    // is exact. Absent keep-set (SPLIT_PRIOR_PYRAMID): no flag, the script keeps
    // everything. Empty keep-set: callers filter those slides out, so reaching here
    // means a lost filter; fail rather than emit duplicate marker names.
    if (meta.keep_channels != null && meta.keep_channels.isEmpty())
        throw new IllegalArgumentException(
            "SPLIT_CHANNELS(${meta.id ?: meta.patient_id}): meta.keep_channels is an EMPTY list. " +
            "A slide that contributes no new markers must be filtered out of this process's input " +
            "by its caller (see subworkflows/local/postprocess.nf and " +
            "subworkflows/local/add_cycle.nf), not handed here.")
    def keep_args = meta.keep_channels ? "--keep-channels ${meta.keep_channels.join(' ')}" : ""
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${registered_image}"], "${meta.patient_id}_${registered_image.simpleName}.SPLIT_CHANNELS.size.csv")}

    echo "Sample: ${meta.patient_id}"
    echo "Channels: ${(meta.channels && meta.channels instanceof List) ? meta.channels.join(', ') : 'Will read from OME metadata'}"

    split_multichannel.py \\
        ${registered_image} \\
        . \\
        ${ref_flag} \\
        ${channel_args} \\
        ${keep_args} \\
        --pixel-size ${params.pixel_size} \\
        ${args}

    ${ProcessEnvelope.versions(task.process, ['tifffile', 'numpy'], task.container)}
    """

    stub:
    // Emits exactly what the real split emits (the keep-set), with the same
    // absent-vs-empty handling as script:, so stub group sizes match.
    if (meta.keep_channels != null && meta.keep_channels.isEmpty())
        throw new IllegalArgumentException(
            "SPLIT_CHANNELS(${meta.id ?: meta.patient_id}): meta.keep_channels is an EMPTY list. " +
            "A slide that contributes no new markers must be filtered out of this process's input " +
            "by its caller (see subworkflows/local/postprocess.nf and " +
            "subworkflows/local/add_cycle.nf), not handed here.")
    def out_channels = meta.keep_channels != null ? meta.keep_channels : meta.channels
    // Empty meta.channels (SPLIT_PRIOR_PYRAMID reads names from OME-XML at runtime):
    // emit one placeholder so the mandatory `*.tiff` output binds under -stub.
    def touch_cmds = out_channels ? out_channels.collect { "touch ${it}.tiff" }.join('\n    ') : "touch prior_pyramid_channel.tiff"
    """
    # One stub file per channel in the keep-set, which is what the real split emits.
    ${touch_cmds}

    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${meta.patient_id}_${registered_image.simpleName}.SPLIT_CHANNELS.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['tifffile', 'numpy'], task.container)}
    """
}
