/*
 * SEGMENT - cell segmentation
 *
 * Dispatches between three backends, controlled by ``params.seg_method``:
 *   - 'stardist':   StarDist on the nuclear channel (channel-0 hard check).
 *   - 'instantseg': InstanSeg with the ``fluorescence_nuclei_and_cells`` model.
 *                   Channel-invariant: consumes the multichannel image directly.
 *   - 'cellsam':    CellSAM (SAM foundation model) on the resolved nuclear channel.
 *                   Segments nuclei; the whole-cell mask is derived by expanding
 *                   nuclei labels (StarDist-style).
 *
 * Everything backend-specific (container, entry point, flags, precondition, versions.yml
 * tools) comes from ``lib/SegBackends.groovy``; tunables come from ``ext.args``. The
 * process body is backend-agnostic. script: and stub: both render versions.yml from
 * ``backend.versionTools``, resolved by the same ``SegBackends.of(...)`` expression.
 *
 * All backends produce the same outputs (``*_nuclei_mask.tif`` / ``*_cell_mask.tif`` /
 * ``versions.yml`` / ``*.size.csv``).
 */
process SEGMENT {
    tag "${meta.patient_id}"

    container { SegBackends.container(params.seg_method) }

    input:
    // seg_params = SegBackends.ctxParams(params), the slice of params the backends read.
    // An input, not a bare `params` read, so unrelated params stay out of the task hash.
    tuple val(meta), path(merged_file), val(seg_params)

    output:
    tuple val(meta), path("*_nuclei_mask.tif"), emit: nuclei_mask
    tuple val(meta), path("*_cell_mask.tif")  , emit: cell_mask
    path "versions.yml"                        , emit: versions
    path("*.size.csv")                         , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    def backend = SegBackends.of(params.seg_method)
    def ctx = [meta: meta, prefix: prefix, params: seg_params]
    // Fragments are indented here: Nextflow stripIndent()s the finished script, so a
    // fragment carrying its own indentation would flatten the block.
    def guard = backend.guard(ctx).join('\n    ')
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${merged_file}"], "${prefix}.SEGMENT.size.csv")}

    echo "Sample: ${meta.patient_id}"
    echo "Backend: ${params.seg_method} (attempt ${task.attempt})"
    echo "Args: ${args}"

    ${guard}

    ${backend.entrypoint} \\
        --image ${merged_file} \\
        --output-dir . \\
        ${backend.flags(ctx)} \\
        ${args}

    ${ProcessEnvelope.versions(task.process, backend.versionTools, task.container)}
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    // Must match script:'s binding textually (tests/test_versions_envelope.py).
    def backend = SegBackends.of(params.seg_method)
    """
    touch ${prefix}_nuclei_mask.tif
    touch ${prefix}_cell_mask.tif
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${prefix}.SEGMENT.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, backend.versionTools, task.container)}
    """
}
