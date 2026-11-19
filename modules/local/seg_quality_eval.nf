/*
 * SEG_QUALITY_EVAL - reference-free cell-segmentation quality scoring (CSE, 2D).
 * Scores each patient's cell+nucleus mask against its reference image and emits
 * a per-patient metrics JSON (informational QC; no gating).
 */
process SEG_QUALITY_EVAL {
    tag "${meta.patient_id}"

    container 'bolt3x/mirage-segeval:1.0.0'

    input:
    tuple val(meta), path(cell_mask), path(nuclei_mask), path(image)

    output:
    tuple val(meta), path("*_seg_eval.json"), emit: metrics
    path "versions.yml"                      , emit: versions
    path("*.size.csv")                       , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = "${meta.patient_id}"
    // Fallback is meta.pixel_size (resolved per slide), not params.pixel_size: the script
    // calls float() on it, and 'auto' would fail and be dropped by retry-then-drop.
    def px = params.cse_pixel_size_um != null ? params.cse_pixel_size_um : meta.pixel_size
    // Throw rather than omit the flag: retry-then-drop would otherwise drop the artifact silently.
    if (px == null)
        throw new IllegalStateException(
            "SEG_QUALITY_EVAL(${meta.patient_id}): no pixel size available -- meta.pixel_size is " +
            "null and --cse_pixel_size_um was not set. Every meta reaching this process should " +
            "carry a resolved scale by now; a null value here is an upstream bug to fix, not a " +
            "flag to omit.")
    def px_arg = "--pixel-size-um ${px}"
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${image}"], "${prefix}.SEG_QUALITY_EVAL.size.csv")}

    seg_quality_eval.py \\
        --cell-mask ${cell_mask} \\
        --nuclei-mask ${nuclei_mask} \\
        --image ${image} \\
        --id ${prefix} \\
        --out ${prefix}_seg_eval.json \\
        ${px_arg} \\
        ${args}
    ${ProcessEnvelope.versions(task.process, ['sklearn', 'scipy', 'CellSegmentationEvaluator'], task.container)}
    """

    stub:
    def prefix = "${meta.patient_id}"
    """
    echo '{"id": "${prefix}", "QualityScore": 0.0, "metrics": {}}' > ${prefix}_seg_eval.json
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${prefix}.SEG_QUALITY_EVAL.size.csv")}
    ${ProcessEnvelope.versionsStub(task.process, ['sklearn', 'scipy', 'CellSegmentationEvaluator'], task.container)}
    """
}
