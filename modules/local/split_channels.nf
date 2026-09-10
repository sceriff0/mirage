/*
 * SPLIT_CHANNELS - Split multi-channel TIFF into individual channels
 *
 * Extracts individual channel images from multi-channel OME-TIFFs for
 * per-channel processing. WHICH channels are emitted is decided once per slide at
 * samplesheet read by CsvUtils.resolveKeptChannelsPerSlide and carried here as
 * meta.keep_channels; this process applies no nuclear-marker rule of its own.
 *
 * Input: Registered multi-channel OME-TIFF and reference flag
 * Output: Individual single-channel TIFF files per marker
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
    // Pass channel names from metadata if available and valid
    def channel_args = (meta.channels && meta.channels instanceof List && !meta.channels.isEmpty()) ?
        "--channels ${meta.channels.join(' ')}" : ""
    // This process performs NO nuclear reasoning. The keep-set was resolved once per
    // slide at samplesheet read (CsvUtils.resolveKeptChannelsPerSlide) and travels on
    // the meta map, so the process just emits what it was handed. That is what makes
    // CsvUtils.countChannelsPerPatient's groupKey size exact by construction, rather
    // than by three copies of one rule continuing to agree by hand.
    //
    // ABSENT vs EMPTY. An ABSENT keep-set (dev's add_cycle alias SPLIT_PRIOR_PYRAMID, which reads channel names
    // from OME-XML at runtime) means "no keep-set was resolved" and correctly renders no
    // flag, so split_multichannel.py falls back to its is_reference rule -- that caller is
    // is_reference=true, i.e. keep everything. An EMPTY keep-set means "this slide
    // contributes no new markers", and BOTH callers filter such slides out of this
    // process's input (subworkflows/local/postprocess.nf, subworkflows/local/add_cycle.nf)
    // because `path("*.tiff")` is a mandatory output. Reaching here with an empty list
    // therefore means a caller lost that filter. Fail loudly: rendering no flag would
    // silently re-enable the legacy nuclear rule and emit the slide's whole declared
    // panel, duplicating marker NAMES already claimed by another slide of this patient --
    // the one thing CsvUtils.resolveKeptChannelsPerSlide's invariant forbids, and the
    // reason meta.channels_count would then be an under-count.
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
    // Exactly what the real split emits, from the same source of truth: the keep-set
    // resolved at samplesheet read. CsvUtils.countChannelsPerPatient sizes the
    // postprocessing groupKeys from that SAME resolver, so the stub's output count
    // matches the size the group expects.
    // ABSENT vs EMPTY, the SAME distinction as the script: block above (see it for the
    // full account) -- and it must be made HERE too, identically. The stub's own `?:`
    // used to resolve an empty keep-set to meta.channels, i.e. the slide's whole declared
    // panel INCLUDING the nuclear channel, while the real script fell back to the legacy
    // nuclear rule and emitted the non-nuclear ones. Stub and real disagreed, so no -stub
    // test could see the divergence.
    if (meta.keep_channels != null && meta.keep_channels.isEmpty())
        throw new IllegalArgumentException(
            "SPLIT_CHANNELS(${meta.id ?: meta.patient_id}): meta.keep_channels is an EMPTY list. " +
            "A slide that contributes no new markers must be filtered out of this process's input " +
            "by its caller (see subworkflows/local/postprocess.nf and " +
            "subworkflows/local/add_cycle.nf), not handed here.")
    def out_channels = meta.keep_channels != null ? meta.keep_channels : meta.channels
    // When meta.channels is empty (e.g. dev's add_cycle alias SPLIT_PRIOR_PYRAMID, which
    // reads channel names from OME-XML in REAL mode only), still emit a single
    // placeholder so the mandatory `*.tiff` output binds under -stub.
    def touch_cmds = out_channels ? out_channels.collect { "touch ${it}.tiff" }.join('\n    ') : "touch prior_pyramid_channel.tiff"
    """
    # One stub file per channel in the keep-set, which is what the real split emits.
    ${touch_cmds}

    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${meta.patient_id}_${registered_image.simpleName}.SPLIT_CHANNELS.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['tifffile', 'numpy'], task.container)}
    """
}
