/*
 * GENERATE_REGISTRATION_QC - render a before/after registration QC figure.
 *
 * Two composites on the reference canvas, origin-aligned: reference + native moving
 * slide (left), reference + registered slide (right). Differing dimensions are
 * pad-or-cropped, never rescaled (bin/utils/qc.py:compose_on_reference_canvas).
 * The overlay uses the nuclear channel, resolved per image via MarkerUtils.
 *
 * The `native/` stageAs name is also what the registration and add_cycle nf-tests
 * pair on.
 */
process GENERATE_REGISTRATION_QC {
    tag "${meta.patient_id}"
    label 'process_high'

    container "bolt3x/mirage-regqc:1.0.0"

    input:
    // `native` is a Groovy reserved word, hence `native_image`. Each input gets its own
    // stageAs subdirectory because the three slides' basenames may coincide.
    tuple val(meta), path(registered), path(native_image, stageAs: 'native/*'), path(reference)

    output:
    tuple val(meta), path("qc/*_QC_RGB.{png,tif}"), emit: qc
    tuple val(meta), path("qc/*_QC_RGB_fullres.tif"), optional: true, emit: qc_fullres
    path "versions.yml"                           , emit: versions
    path("*.size.csv")                            , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def scale_factor = params.qc_scale_factor
    // MarkerUtils normalises the String form a CLI --nuclear_markers produces.
    def nuclear_args = "--nuclear-markers ${MarkerUtils.markerList(params.nuclear_markers).join(' ')}"
    """
    # Log input sizes for tracing (sum of registered + native + reference, -L follows symlinks)
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${registered}", "${native_image}", "${reference}"], "${meta.patient_id}_${registered.simpleName}.GENERATE_REGISTRATION_QC.size.csv")}

    mkdir -p qc

    generate_registration_qc.py \\
        --reference ${reference} \\
        --registered ${registered} \\
        --native ${native_image} \\
        --pixel-size-um ${meta.pixel_size} \\
        --output qc \\
        --scale-factor ${scale_factor} \\
        ${nuclear_args} \\
        ${args}

    ${ProcessEnvelope.versions(task.process, ['numpy', 'tifffile'], task.container)}
    """

    stub:
    """
    mkdir -p qc
    touch qc/${registered.simpleName}_QC_RGB.png
    touch qc/${registered.simpleName}_QC_RGB_fullres.tif
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${meta.patient_id}_${registered.simpleName}.GENERATE_REGISTRATION_QC.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['numpy', 'tifffile'], task.container)}
    """
}
